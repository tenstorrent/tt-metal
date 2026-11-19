# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only GCC13/ATen topk8 reproduction and logit-key equivalence audit.

The scalar selector uses integer keys and indices only, and deliberately has no
index tie-break. This is a diagnostic specification, not a device implementation.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def integer_keys(tensor):
    """Increasing numeric keys; normalize signed zero and reject non-finite data."""
    flat = tensor.flatten().contiguous()
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("finite input required; non-finite softmax is outside this contract")
    if flat.dtype == torch.bfloat16:
        bits, sign, mask = flat.view(torch.int16).tolist(), 0x8000, 0xFFFF
    elif flat.dtype == torch.float32:
        bits, sign, mask = flat.view(torch.int32).tolist(), 0x80000000, 0xFFFFFFFF
    else:
        raise ValueError("BF16 or FP32 required")
    result = []
    for raw in bits:
        raw &= mask
        if raw & (sign - 1) == 0:
            raw = 0
        result.append((~raw & mask) if raw & sign else (raw ^ sign))
    return result


def select_top8(keys, depth=16):
    """ATen sorted topk8: nth_element(nth=7), then sort [0,7).

    Mirrors GCC13 median-to-first, unguarded partition, depth-limited
    introselect, heap-select fallback, and stable insertion sort on short runs.
    'Greater' compares keys alone. Payload indices never enter comparison.
    """
    assert len(keys) == 256
    queue = list(range(256))
    stats = {"partitions": 0, "heap_fallbacks": 0}

    def greater(left, right):
        return keys[left] > keys[right]

    def swap(left, right):
        queue[left], queue[right] = queue[right], queue[left]

    def insertion(first, last):
        for pos in range(first + 1, last):
            value = queue[pos]
            hole = pos
            while hole > first and greater(value, queue[hole - 1]):
                queue[hole] = queue[hole - 1]
                hole -= 1
            queue[hole] = value

    def push_heap(first, hole, top, value):
        while hole > top:
            parent = (hole - 1) // 2
            if not greater(queue[first + parent], value):
                break
            queue[first + hole] = queue[first + parent]
            hole = parent
        queue[first + hole] = value

    def adjust_heap(first, hole, length, value):
        top = hole
        second_child = hole
        while second_child < (length - 1) // 2:
            second_child = 2 * (second_child + 1)
            if greater(queue[first + second_child], queue[first + second_child - 1]):
                second_child -= 1
            queue[first + hole] = queue[first + second_child]
            hole = second_child
        if length % 2 == 0 and second_child == (length - 2) // 2:
            second_child = 2 * (second_child + 1)
            queue[first + hole] = queue[first + second_child - 1]
            hole = second_child - 1
        push_heap(first, hole, top, value)

    def heap_select(first, middle, last):
        length = middle - first
        if length >= 2:
            for parent in range((length - 2) // 2, -1, -1):
                adjust_heap(first, parent, length, queue[first + parent])
        for pos in range(middle, last):
            if greater(queue[pos], queue[first]):
                value = queue[pos]
                queue[pos] = queue[first]
                adjust_heap(first, 0, length, value)

    first, last, nth = 0, 256, 7
    while last - first > 3:
        if depth == 0:
            stats["heap_fallbacks"] += 1
            heap_select(first, nth + 1, last)
            swap(first, nth)
            break
        depth -= 1
        stats["partitions"] += 1
        a, b, c = first + 1, first + (last - first) // 2, last - 1
        if greater(queue[a], queue[b]):
            median = b if greater(queue[b], queue[c]) else c if greater(queue[a], queue[c]) else a
        elif greater(queue[a], queue[c]):
            median = a
        elif greater(queue[b], queue[c]):
            median = c
        else:
            median = b
        swap(first, median)
        left, right, pivot = first + 1, last, queue[first]
        while True:
            while greater(queue[left], pivot):
                left += 1
            right -= 1
            while greater(pivot, queue[right]):
                right -= 1
            if left >= right:
                cut = left
                break
            swap(left, right)
            left += 1
        if cut <= nth:
            first = cut
        else:
            last = cut
    else:
        insertion(first, last)
    insertion(0, 7)
    return queue[:8], stats


# No standard-library calls, recursion, allocation, or large local arrays.
# Call with 256 caller-owned uint32_t entries: (bf16_bits << 16) | original_id.
# Returned first eight entries are in the same order as installed Torch topk8.
PORTABLE_CPP = r"""
inline bool v6_greater(uint32_t a, uint32_t b) {
    uint32_t x = a >> 16, y = b >> 16;
    bool xn = (x & 0x7fff) > 0x7f80, yn = (y & 0x7fff) > 0x7f80;
    if (xn || yn) return xn && !yn;
    if ((x & 0x7fff) == 0) x = 0;
    if ((y & 0x7fff) == 0) y = 0;
    uint32_t kx = (x & 0x8000) ? ((~x) & 0xffff) : (x ^ 0x8000);
    uint32_t ky = (y & 0x8000) ? ((~y) & 0xffff) : (y ^ 0x8000);
    return kx > ky;
}
inline void v6_swap(uint32_t* q, int a, int b) {
    uint32_t value=q[a]; q[a]=q[b]; q[b]=value;
}
inline void v6_insertion(uint32_t* q, int first, int last) {
    for (int pos=first+1; pos<last; ++pos) {
        uint32_t value=q[pos]; int hole=pos;
        while (hole>first && v6_greater(value,q[hole-1])) {
            q[hole]=q[hole-1]; --hole;
        }
        q[hole]=value;
    }
}
inline void v6_push_heap(uint32_t* q, int first, int hole, int top, uint32_t value) {
    while (hole>top) {
        int parent=(hole-1)/2;
        if (!v6_greater(q[first+parent],value)) break;
        q[first+hole]=q[first+parent]; hole=parent;
    }
    q[first+hole]=value;
}
inline void v6_adjust_heap(uint32_t* q, int first, int hole, int length, uint32_t value) {
    int top=hole, second_child=hole;
    while (second_child<(length-1)/2) {
        second_child=2*(second_child+1);
        if (v6_greater(q[first+second_child],q[first+second_child-1])) --second_child;
        q[first+hole]=q[first+second_child]; hole=second_child;
    }
    if ((length&1)==0 && second_child==(length-2)/2) {
        second_child=2*(second_child+1);
        q[first+hole]=q[first+second_child-1]; hole=second_child-1;
    }
    v6_push_heap(q,first,hole,top,value);
}
inline void v6_heap_select(uint32_t* q, int first, int middle, int last) {
    int length=middle-first;
    if (length>=2) {
        for (int parent=(length-2)/2; parent>=0; --parent)
            v6_adjust_heap(q,first,parent,length,q[first+parent]);
    }
    for (int pos=middle; pos<last; ++pos) {
        if (v6_greater(q[pos],q[first])) {
            uint32_t value=q[pos]; q[pos]=q[first];
            v6_adjust_heap(q,first,0,length,value);
        }
    }
}
inline void v6_select_top8(uint32_t* q, int depth=16) {
    int first=0,last=256,nth=7;
    while (last-first>3) {
        if (depth==0) {
            v6_heap_select(q,first,nth+1,last); v6_swap(q,first,nth);
            v6_insertion(q,0,7); return;
        }
        --depth;
        int a=first+1,b=first+(last-first)/2,c=last-1,median;
        if (v6_greater(q[a],q[b])) {
            if (v6_greater(q[b],q[c])) median=b;
            else if (v6_greater(q[a],q[c])) median=c;
            else median=a;
        } else if (v6_greater(q[a],q[c])) median=a;
        else if (v6_greater(q[b],q[c])) median=c;
        else median=b;
        v6_swap(q,first,median);
        int left=first+1,right=last; uint32_t pivot=q[first];
        while (true) {
            while (v6_greater(q[left],pivot)) ++left;
            --right;
            while (v6_greater(pivot,q[right])) --right;
            if (left>=right) break;
            v6_swap(q,left,right); ++left;
        }
        if (left<=nth) first=left;
        else last=left;
    }
    v6_insertion(q,first,last); v6_insertion(q,0,7);
}
"""

PACKED_DRIVER = r"""
int main() {
    int depth;
    while (std::cin >> depth) {
        uint32_t q[256];
        for (unsigned i=0;i<256;++i) { uint32_t bits; std::cin >> bits; q[i]=(bits<<16)|i; }
        v6_select_top8(q,depth);
        for (unsigned i=0;i<8;++i) std::cout << (q[i]&0xffff) << ' ';
        std::cout << '\n';
    }
}
"""

# Directly calls installed GCC13's algorithm for independent fallback verification.
CPP = r"""
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
int main() {
  int depth;
  while (std::cin >> depth) {
    std::pair<uint64_t, unsigned> q[256];
    for (unsigned i=0;i<256;++i) { std::cin >> q[i].first; q[i].second=i; }
    auto comp=[](const auto& a,const auto& b){return a.first>b.first;};
    std::__introselect(q,q+7,q+256,depth,__gnu_cxx::__ops::__iter_comp_iter(comp));
    std::sort(q,q+7,comp);
    for (unsigned i=0;i<8;++i) std::cout << q[i].second << ' ';
    std::cout << '\n';
  }
}
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    fixtures = []
    proof_path = DOC / "v5_torch_topk_tie_permutation_proof.json"
    proof = json.loads(proof_path.read_text())
    for case in proof["cases"]:
        logits = torch.tensor(case["input_bf16_codes"], dtype=torch.int16).view(torch.bfloat16)
        fixtures.append(("saved_permutation", f"permutation_{case['permutation_number']}", logits))
    captures = []
    for path in sorted(DOC.glob("v5*probe.json.pos*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        captures.append({"path": str(path), "sha256": digest(path)})
        for name, logits in (("hf", data["hf_values"]["router"][0]), ("tt", data["tt_values"]["router"])):
            fixtures.append(("saved_capture", path.name + ":" + name, logits.flatten().bfloat16()))
    for index in range(args.rows):
        for group, logits in (
            ("gaussian_std1", torch.randn(256, generator=generator)),
            ("gaussian_std5", 5 * torch.randn(256, generator=generator)),
            ("quantized_ties", torch.randint(-16, 17, (256,), generator=generator).float() / 8),
            ("wide_underflow", 100 * torch.randn(256, generator=generator)),
            ("tiny_rounding", 1e-8 * torch.randn(256, generator=generator)),
        ):
            fixtures.append((group, str(index), logits.bfloat16()))
    fixtures.extend(
        ("edge", name, values.bfloat16())
        for name, values in (
            ("all_equal", torch.ones(256)),
            ("signed_zero", torch.tensor([0.0, -0.0] * 128)),
            ("ascending", torch.arange(256).float()),
            ("descending", -torch.arange(256).float()),
            ("one_hot_extreme", torch.tensor([1000.0] + [-1000.0] * 255)),
        )
    )
    groups, failures, fallback_cases = {}, [], []
    for group, name, logits in fixtures:
        probability = logits.float().softmax(-1)
        direct, stats = select_top8(integer_keys(logits))
        prob, _ = select_top8(integer_keys(probability))
        expected_direct = logits.float().topk(8).indices.tolist()
        expected_prob = probability.topk(8).indices.tolist()
        row = groups.setdefault(
            group,
            dict(
                rows=0,
                direct_selector_mismatches=0,
                probability_selector_mismatches=0,
                direct_vs_softmax_order_mismatches=0,
                direct_vs_softmax_membership_mismatches=0,
                rows_with_underflow=0,
                natural_heap_fallbacks=0,
                examples=[],
            ),
        )
        row["rows"] += 1
        row["direct_selector_mismatches"] += direct != expected_direct
        row["probability_selector_mismatches"] += prob != expected_prob
        row["direct_vs_softmax_order_mismatches"] += direct != expected_prob
        row["direct_vs_softmax_membership_mismatches"] += set(direct) != set(expected_prob)
        row["rows_with_underflow"] += bool((probability == 0).any())
        row["natural_heap_fallbacks"] += stats["heap_fallbacks"]
        detail = dict(
            name=name,
            direct=direct,
            softmax=expected_prob,
            input_bf16_codes=logits.contiguous().view(torch.int16).tolist(),
        )
        if direct != expected_prob and len(row["examples"]) < 2:
            row["examples"].append(detail)
        if direct != expected_direct or prob != expected_prob:
            failures.append(detail)
        if group in ("saved_capture", "saved_permutation", "edge") or int(name) < 32:
            fallback_cases.append(integer_keys(logits))
    with tempfile.TemporaryDirectory(prefix="v6_selector_") as directory:
        source, binary = Path(directory) / "proof.cpp", Path(directory) / "proof"
        source.write_text(CPP)
        subprocess.run(["g++-13", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
        requests = [(depth, keys) for keys in fallback_cases for depth in (0, 1, 2, 16)]
        result = subprocess.run(
            [str(binary)],
            input="".join(f"{d} " + " ".join(map(str, k)) + "\n" for d, k in requests),
            text=True,
            capture_output=True,
            check=True,
        )
        actual = [list(map(int, line.split())) for line in result.stdout.splitlines()]
        assert len(actual) == len(requests)
        fallback_failures = [
            index
            for index, ((depth, keys), expected) in enumerate(zip(requests, actual))
            if select_top8(keys, depth)[0] != expected
        ]
    with tempfile.TemporaryDirectory(prefix="v6_packed_") as directory:
        source, binary = Path(directory) / "packed.cpp", Path(directory) / "packed"
        source.write_text("#include <cstdint>\n#include <iostream>\n" + PORTABLE_CPP + PACKED_DRIVER)
        subprocess.run(["g++-13", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
        packed_requests = [(16, logits) for _, _, logits in fixtures]
        packed_requests += [
            (depth, logits)
            for group, _, logits in fixtures
            if group in ("saved_capture", "saved_permutation", "edge")
            for depth in (0, 1, 2)
        ]
        nonfinite_rows = []
        for _ in range(64):
            codes = torch.randint(-32768, 32768, (256,), generator=generator, dtype=torch.int16)
            codes[:8] = torch.tensor([0, -32768, 32640, -128, 32641, -127, 32767, -1], dtype=torch.int16)
            nonfinite_rows.append(codes.view(torch.bfloat16))
        packed_requests += [(16, logits) for logits in nonfinite_rows]
        wire = "".join(
            f"{depth} " + " ".join(str(code & 0xFFFF) for code in logits.contiguous().view(torch.int16).tolist()) + "\n"
            for depth, logits in packed_requests
        )
        result = subprocess.run([str(binary)], input=wire, text=True, capture_output=True, check=True)
        packed_actual = [list(map(int, line.split())) for line in result.stdout.splitlines()]
        assert len(packed_actual) == len(packed_requests)
        packed_failures = [
            index
            for index, ((depth, logits), actual_ids) in enumerate(zip(packed_requests, packed_actual))
            if (
                select_top8(integer_keys(logits), depth)[0]
                if bool(torch.isfinite(logits).all())
                else logits.float().topk(8).indices.tolist()
            )
            != actual_ids
        ]
    nonfinite_rejections = {}
    for name, value in (
        ("nan", float("nan")),
        ("positive_infinity", float("inf")),
        ("negative_infinity", -float("inf")),
    ):
        try:
            integer_keys(torch.tensor([value] * 256).bfloat16())
            nonfinite_rejections[name] = False
        except ValueError:
            nonfinite_rejections[name] = True
    sources = [
        Path(torch.__file__).parent / "include/ATen/native/TopKImpl.h",
        Path("/usr/include/c++/13/bits/stl_algo.h"),
        Path("/usr/include/c++/13/bits/stl_heap.h"),
    ]
    result = dict(
        status=(
            "PASS"
            if not failures and not fallback_failures and not packed_failures and all(nonfinite_rejections.values())
            else "FAIL"
        ),
        classification="CPU selector algorithm proof; not model correctness or device performance acceptance",
        script_sha256=digest(__file__),
        seed=args.seed,
        random_rows_per_group=args.rows,
        torch_version=torch.__version__,
        torch_build=torch.__config__.show(),
        sources_sha256={str(p): digest(p) for p in sources},
        permutation_proof_sha256=digest(proof_path),
        captures=captures,
        groups=groups,
        selector_failures=failures[:8],
        cpp_forced_depth_checks=len(requests),
        cpp_depths=[0, 1, 2, 16],
        cpp_mismatches=fallback_failures,
        portable_cpp_sha256=hashlib.sha256(PORTABLE_CPP.encode()).hexdigest(),
        packed_cpp_checks=len(packed_requests),
        packed_cpp_nonfinite_direct_topk_checks=len(nonfinite_rows),
        packed_cpp_mismatches=packed_failures,
        nonfinite_rejections=nonfinite_rejections,
        contract="Integer selector exactly reproduces tested GCC13/ATen ordering for its supplied keys. "
        "BF16 logit keys cannot universally replace FP32-softmax keys: probability rounding or "
        "underflow can alter ties and the partition path. Non-finite values are rejected; signed zero compares equal.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": len(fixtures),
                "groups": {k: {a: b for a, b in v.items() if a != "examples"} for k, v in groups.items()},
                "cpp_checks": len(requests),
                "cpp_mismatches": len(fallback_failures),
            },
            indent=2,
        )
    )
    assert result["status"] == "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DOC / "v6_cpu_topk_selector_proof.json")
    run(parser.parse_args())
