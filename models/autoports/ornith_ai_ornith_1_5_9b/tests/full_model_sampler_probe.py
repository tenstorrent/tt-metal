# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare common sampler's two greedy strategies on real TP4 model logits."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.common.sampling.generator import format_sampling_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen, traces = None, []
    results = {"status": "in_progress", "layers": [0, 3], "regime": "serial", "strategies": {}}
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3])
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=(0,))
        gen.reset()
        gen.prefill_forward(
            torch.arange(100, 133).reshape(1, -1), page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=[33]
        )
        gen._copy(torch.tensor([33], dtype=torch.int32), state.positions[0], "position_refreshes", signed=True)
        token_input = torch.full((1, 1, 1, 32), 234, dtype=torch.int32)
        gen._copy(token_input, state.tokens, "token_refreshes")
        state.logits = gen._model_step(state)
        reference_logits = gen.model.host_logits(state.logits)[0, 0, 0]
        assert torch.isfinite(reference_logits).all() and reference_logits.std() > 0
        expected = int(reference_logits.argmax())
        results["logit_fixture"] = {
            "input_token": 234,
            "minimum": float(reference_logits.min()),
            "maximum": float(reference_logits.max()),
            "std": float(reference_logits.std()),
        }
        params = format_sampling_params(gen.sampling_params, 32)

        def mode(force):
            gen.sampler.tt_sampling._allow_force_argmax_sampling = force
            gen.sampler.reset_sampling_params(params)
            assert gen.sampler.tt_sampling.force_argmax_sampling == force

        # Complete all warmup before capturing either trace.
        supported = []
        for name, force in (("topk_greedy", False), ("force_argmax", True)):
            mode(force)
            try:
                gen._sample_step(state)
                got = int(ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1)[0])
                assert got == expected, (name, expected, got)
                supported.append((name, force))
                results["strategies"][name] = {"status": "warm_pass", "expected": expected, "actual": got}
            except Exception as error:
                results["strategies"][name] = {"status": "failed", "error": str(error)}
                Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
                raise
        for name, force in supported:
            mode(force)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            gen._sample_step(state)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces.append(trace)
            for _ in range(4):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(args.iterations):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - start
            got = int(ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1)[0])
            assert got == expected, (name, expected, got)
            results["strategies"][name].update(status="pass", milliseconds=elapsed * 1000 / args.iterations)
        results["status"] = "pass"
        results["scope"] = "Sampling-only real reduced-model logits; not full-model throughput or broad tie coverage"
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results), flush=True)
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
