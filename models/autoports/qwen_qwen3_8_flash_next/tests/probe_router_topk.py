# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared-source CPU proof and optional traced device router-topk parity.

CPU mode records exact HF logits at all95 batch32 rows, checks the same integer
algorithm compiled into the device kernel against torch, and exercises the
introselect heap fallback at artificially exhausted depth. Device mode consumes
that persisted fixture, uses fixed prepared buffers, changes input logits
between trace replays, and requires bit-exact scores and ordered expert IDs.
This isolates selection; it does not claim full decoder or router matmul parity.
"""

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.router_topk import algorithm_source


def cpu_proof(logits):
    source = (
        "#include <algorithm>\n#include <cmath>\n#include <cstring>\n#include <vector>\n"
        + algorithm_source()
        + r"""
extern "C" void selection(const uint32_t* input, int64_t* output, int rows, int depth, int control) {
    std::vector<RouterItem> queue(512);
    auto greater = [](RouterItem a, RouterItem b) {
        float x, y; std::memcpy(&x, &a.bits, 4); std::memcpy(&y, &b.bits, 4);
        return (std::isnan(x) && !std::isnan(y)) || x > y;
    };
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < 512; ++i) queue[i] = {input[row * 512 + i], uint32_t(i)};
        if (control) {
            std::__introselect(queue.begin(), queue.begin() + 9, queue.end(), depth,
                               __gnu_cxx::__ops::__iter_comp_iter(greater));
            std::sort(queue.begin(), queue.begin() + 9, greater);
        } else router_top10(queue.data(), depth);
        for (int i = 0; i < 10; ++i) output[row * 10 + i] = queue[i].index;
    }
}
"""
    )
    with tempfile.TemporaryDirectory(prefix="router-topk-proof-") as temporary:
        cpp, library = Path(temporary) / "proof.cpp", Path(temporary) / "proof.so"
        cpp.write_text(source)
        subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(cpp), "-o", str(library)], check=True)
        function = ctypes.CDLL(str(library)).selection
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

        def execute(values, depth, control):
            values = values.contiguous()
            output = torch.empty((values.shape[0], 10), dtype=torch.int64)
            function(values.data_ptr(), output.data_ptr(), values.shape[0], depth, control)
            return output

        generator = torch.Generator().manual_seed(91238)
        random = torch.randn((4096, 512), generator=generator).bfloat16().float()
        coarse = torch.randint(-4, 5, (4096, 512), generator=generator).float()
        special = torch.zeros(3, 512)
        special[1, ::2] = -0.0
        special[2, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
        values = torch.cat([logits, random, coarse, special]).contiguous()
        target = values.topk(10, dim=-1).indices
        actual = execute(values, 18, 0)
        assert torch.equal(actual, target), torch.where((actual != target).any(-1))[0].tolist()
        for depth in (0, 1, 2):
            assert torch.equal(execute(values, depth, 0), execute(values, depth, 1)), f"Heap fallback depth{depth}"
        return {"passed": True, "rows": values.shape[0], "ordered_indices_exact": True, "forced_heap_depths": [0, 1, 2]}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.fixture_cache.exists():
        fixture = torch.load(args.fixture_cache, map_location="cpu", weights_only=True)
    else:
        if args.snapshot is None:
            parser.error("--snapshot is needed when creating the exact-HF logit fixture")
        from models.autoports.qwen_qwen3_8_flash_next.tests.probe_batch_intermediates import fixture as build_fixture

        _, reference, weights, requests, hashes = build_fixture(args.snapshot, tuple(range(32)))
        values = [
            phase["router_logits"].float() for request in requests.values() for phase in request["phases"].values()
        ]
        logits = torch.cat(values).contiguous()
        fixture = {"logits": logits, "source": "literal HF real layer0 batch32 seed1981", "input_hashes": hashes}
        args.fixture_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixture, args.fixture_cache)
        del reference, weights, requests, values
    logits = fixture["logits"].float().contiguous()
    assert logits.ndim == 2 and logits.shape[1] == 512
    report = {
        "scope": __doc__,
        "fixture": str(args.fixture_cache),
        "fixture_sha256": hashlib.sha256(args.fixture_cache.read_bytes()).hexdigest(),
        "algorithm_sha256": hashlib.sha256(algorithm_source().encode()).hexdigest(),
        "cpu": cpu_proof(logits),
        "device": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ROUTER_CPU_PROOF", report["cpu"], flush=True)
    if args.cpu_only:
        assert "ttnn" not in sys.modules
        return

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.router_topk import RouterTopK

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        for count in (1, 2, 17, 31, 32):
            # Each physical shape sees all exact-HF fixture rows; the final
            # partial bucket wraps to its beginning with ordinary valid rows.
            cases = [
                logits[(torch.arange(count) + start) % len(logits)].reshape(1, 1, count, 512)
                for start in range(0, len(logits), count)
            ]
            cases.append(cases[0].clone())
            host_inputs = [ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT) for value in cases]
            input_device = ttnn.to_device(host_inputs[0], mesh)
            scores = ttnn.empty((1, 1, count, 10), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            indices = ttnn.empty((1, 1, count, 10), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
            operation = RouterTopK(input_device, scores, indices)
            operation.prepare()
            with runtime_audit():
                operation.forward()
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with runtime_audit():
                operation.forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            for value, refresh in zip(cases, host_inputs):
                ttnn.copy_host_to_device_tensor(refresh, input_device)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                expected_scores, expected_indices = value.topk(10, dim=-1)
                actual_scores = ttnn.to_torch(scores)
                actual_indices = ttnn.to_torch(indices).long()
                assert torch.equal(actual_indices, expected_indices), (count, actual_indices, expected_indices)
                assert torch.equal(actual_scores, expected_scores), count
            report["device"].append(
                {
                    "logical_rows": count,
                    "replays": len(cases),
                    "scores_exact": True,
                    "indices_exact": True,
                    "changed_inputs": True,
                    "runtime_audit": True,
                    "repeated_input_exact": True,
                }
            )
            print("ROUTER_DEVICE_PASS", count, flush=True)
            ttnn.release_trace(mesh, trace)
            trace = None
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
