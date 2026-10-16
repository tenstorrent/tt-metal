# Where a minimal_matmul call's cycles go inside the compute kernel's block loop: device zones patched into
# compute_metal2.cpp (restored afterwards) around each output block's init (reconfig + matmul init + intermediate
# reserve), each K block (MM-KB; its in0 / in1 wait_front nested as MM-KW) and the epilogue (intermediate -> out),
# then bench_mm_ablate.py's child per arm under the device profiler, in its own process with a fresh JIT cache.
# Per RISC (TRISC_0 unpack, TRISC_1 math, TRISC_2 pack), cycles per zone (median over cores and calls): the first K block
# of an output block (pipeline fill) vs the rest, the gaps between K blocks, the init and the epilogue, against the LLK
# ideal of 17.4 cycles per tile product. A zone costs ~40 cycles (two timestamp writes); 125 zones per RISC per call fit.
# Usage: bench_mm_zones.py <batch> [arm ...]   arm = <preset>[@M,K,N,sh,sw][+compute] (bench_mm_ablate presets; +compute
#        drops the DM kernels' DRAM / NoC reads and writes, as bench_mm_ablate's 0,0,0 arm), e.g. qkv wo ff2 ff13
import collections
import csv
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_mm_ablate import HDR, patch, preset  # noqa: E402

COMPUTE = os.path.join(os.path.dirname(HDR), "compute_metal2.cpp")
LLK = 17.4

PATCHES = [
    (
        '#include "api/compute/compute_kernel_api.h"\n',
        '#include "api/compute/compute_kernel_api.h"\n#include "api/debug/kernel_profiler.hpp"\n',
    ),
    (
        "            // Reconfig before init: on all but the first block",
        '            {\n            DeviceZoneScopedN("MM-INIT");\n            // Reconfig before init: on all but the first block',
    ),
    (
        "            dfb_intermediate.reserve_back(out_block_num_tiles);\n            for (uint32_t k_block = 0;",
        "            dfb_intermediate.reserve_back(out_block_num_tiles);\n            }\n            for (uint32_t k_block = 0;",
    ),
    (
        "                dfb_in0.wait_front(in0_block_num_tiles);\n                dfb_in1.wait_front(in1_block_num_tiles);\n",
        '                DeviceZoneScopedN("MM-KB");\n                {\n'
        + ('                DeviceZoneScopedN("MM-KW");\n' if os.getenv("MM_ZONES_KW") == "1" else "")
        + "                dfb_in0.wait_front(in0_block_num_tiles);\n                dfb_in1.wait_front(in1_block_num_tiles);\n"
        "                }\n",
    ),
    (
        "            if constexpr (!swiglu_in_k_loop) {\n                dfb_intermediate.push_back(out_block_num_tiles);",
        '            {\n            DeviceZoneScopedN("MM-EPI");\n'
        "            if constexpr (!swiglu_in_k_loop) {\n                dfb_intermediate.push_back(out_block_num_tiles);",
    ),
    (
        "#endif  // FUSE_TERNARY\n            }\n        }\n    }\n}",
        "#endif  // FUSE_TERNARY\n            }\n            }\n        }\n    }\n}",
    ),
]


def patch_compute(src):
    for old, new in PATCHES:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def zones(d):
    """{risc: {name: [durations]}, ...} plus per-(core, risc, run) ordered zone lists"""
    f = open(f"{d}/.logs/profile_log_device.csv")
    next(f)
    r = csv.reader(f)
    ix = {h.strip(): i for i, h in enumerate(next(r))}
    seqs = collections.defaultdict(list)  # (run, core, risc) -> [(name, start, end)]
    stack = collections.defaultdict(list)
    kern = {}
    for row in r:
        name = row[ix["zone name"]].strip()
        if not (name.startswith("MM-") or name == "TRISC-KERNEL"):
            continue
        tid = row[ix["trace id"]].strip()
        if not tid:
            continue
        key = (row[ix["run host ID"]].strip(), row[ix["trace id counter"]].strip())
        core = (row[ix["core_x"]].strip(), row[ix["core_y"]].strip())
        risc = row[ix["RISC processor type"]].strip()
        t = int(row[ix["time[cycles since reset]"]])
        k = (key, core, risc, name)
        if row[ix["type"]].strip() == "ZONE_START":
            stack[k].append(t)
        elif stack[k]:
            t0 = stack[k].pop()
            if name == "TRISC-KERNEL":
                kern[(key, core, risc)] = (t0, t)
            else:
                seqs[(key, core, risc)].append((name, t0, t))
    return seqs, kern


def summarize(seqs, kern, ideal_kb):
    st = collections.defaultdict(lambda: collections.defaultdict(list))
    for (key, core, risc), zs in seqs.items():
        zs.sort(key=lambda z: z[1])
        s = st[risc]
        kbs = [z for z in zs if z[0] == "MM-KB"]
        prev = None
        for z in zs:
            name, t0, t1 = z
            if name == "MM-INIT":
                s["init"].append(t1 - t0)
                prev = None
            elif name == "MM-EPI":
                s["epi"].append(t1 - t0)
            elif name == "MM-KW":
                s["kw"].append(t1 - t0)
            elif name == "MM-KB":
                s["kb_first" if prev is None else "kb_rest"].append(t1 - t0)
                if prev is not None:
                    s["kb_gap"].append(t0 - prev)
                prev = t1
        k = kern.get((key, core, risc))
        k = k[1] - k[0] if k else None
        if k:
            tot = sum(z[2] - z[1] for z in zs if z[0] != "MM-KW")
            s["kernel"].append(k)
            s["outside"].append(k - tot)
            s["n_kb"].append(len(kbs))
    out = []
    for risc in sorted(st):
        s = st[risc]
        med = lambda v: statistics.median(v) if v else float("nan")
        kw_total = sum(s["kw"]) / max(1, len(s["kernel"]))
        out.append(
            f"  {risc:8s} kernel {med(s['kernel']):9.0f}  n_kb {med(s['n_kb']):4.0f}  "
            f"kb first {med(s['kb_first']):7.0f} rest {med(s['kb_rest']):7.0f} (ideal {ideal_kb:5.0f})  "
            f"gap {med(s['kb_gap']):5.0f}  kw {med(s['kw']):6.0f} (sum/core {kw_total:8.0f})  "
            f"init {med(s['init']):6.0f}  epi {med(s['epi']):6.0f}  outside {med(s['outside']):7.0f}"
        )
    return "\n".join(out)


def critical(seqs, kern, risc="TRISC_1"):
    """Whole-call breakdown (cycles) on the slowest and the median core of one call: K blocks, gaps between them,
    inits, epilogues, before the first / after the last zone."""
    calls = collections.defaultdict(list)
    for (key, core, r), zs in seqs.items():
        if r == risc and (key, core, r) in kern:
            calls[key].append((kern[(key, core, r)], core, sorted(zs, key=lambda z: z[1])))
    key = sorted(calls)[len(calls) // 2]
    cores = sorted(calls[key], key=lambda c: c[0][1] - c[0][0])
    out = []
    for tag, (kt, core, zs) in (("slowest", cores[-1]), ("median", cores[len(cores) // 2])):
        cat = collections.Counter()
        n = collections.Counter()
        prev = None
        for name, t0, t1 in zs:
            cat[name] += t1 - t0
            n[name] += 1
            if name == "MM-KB" and prev is not None and prev[0] == "MM-KB":
                cat["gap"] += t0 - prev[2]
            elif prev is not None:
                cat["other"] += t0 - prev[2]
            prev = (name, t0, t1)
        pre, post = zs[0][1] - kt[0], kt[1] - zs[-1][2]
        tot = kt[1] - kt[0]
        out.append(
            f"  {risc} {tag:7s} core {core}: kernel {tot} = KB {cat['MM-KB']} (n {n['MM-KB']}) + gaps {cat['gap']} + "
            f"init {cat['MM-INIT']} (n {n['MM-INIT']}) + epi {cat['MM-EPI']} + between {cat['other']} + pre {pre} + "
            f"post {post}   [zones {sum(n.values())}]"
        )
    return "\n".join(out)


def main():
    B = sys.argv[1]
    arms = sys.argv[2:] or ["qkv", "wo", "ff2", "ff13"]
    orig = open(COMPUTE).read()
    orig_hdr = open(HDR).read()
    try:
        open(COMPUTE, "w").write(patch_compute(orig))
        for arm in arms:
            compute_only = arm.endswith("+compute")
            arm = arm.removesuffix("+compute")
            name, _, blocks = arm.partition("@")
            open(HDR, "w").write(patch(orig_hdr, 0, 0, 0) if compute_only else orig_hdr)
            d = tempfile.mkdtemp(prefix="mm_zones_")
            env = dict(
                os.environ,
                TT_METAL_CACHE=os.path.join(d, "cache"),
                TT_METAL_DEVICE_PROFILER="1",
                TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="40000",
                TT_METAL_PROFILER_DIR=os.path.join(d, "prof"),
            )
            if blocks:
                env["MM_BLOCKS"] = blocks
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_mm_ablate.py"), "--child", name, B, os.path.join(d, "o.pt")],
                env=env, capture_output=True, text=True, timeout=1200,
            )  # fmt: skip
            label = f"{arm}{' compute-only' if compute_only else ''} B{B}"
            if p.returncode != 0:
                err = [l.split("TT_THROW: ")[-1][:160] for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l]
                print(f"RES {label} FAILED {err[:2] or (p.stdout + p.stderr)[-400:]}", flush=True)
                continue
            bl = tuple(int(v) for v in blocks.split(",")) if blocks else preset(name, int(B))[3]
            ideal_kb = bl[0] * bl[1] * bl[2] * LLK
            seqs, kern = zones(os.path.join(d, "prof"))
            print(f"RES {label} blocks {bl} (cycles)\n{summarize(seqs, kern, ideal_kb)}", flush=True)
            print(critical(seqs, kern) + "\n" + critical(seqs, kern, "TRISC_0"), flush=True)
    finally:
        open(COMPUTE, "w").write(orig)
        open(HDR, "w").write(orig_hdr)


if __name__ == "__main__":
    main()
