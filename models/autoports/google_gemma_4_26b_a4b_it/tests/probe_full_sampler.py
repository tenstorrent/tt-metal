# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone TP4 sampler contract and semantically greedy comparison."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params


def known_logits(batch, round_index, *, physical_batch=None):
    cpu = torch.zeros(1, 1, physical_batch or batch, 262144)
    cpu[:, :, :batch] = -100
    winners = [((row + 2 * round_index) % 4) * 65536 + 4 + 37 * row + round_index for row in range(batch)]
    for row, token in enumerate(winners):
        cpu[0, 0, row, token] = 100
    return cpu, winners


def check_case(mesh, *, batch, force, mode, pad_batch, iterations, results):
    cfg = SimpleNamespace(
        vocab_size=262144,
        padded_vocab_size=262144,
        cluster_shape=(1, 4),
        sampling_all_gather_axis=1,
        sampling_dp=1,
        num_devices=4,
        is_galaxy=False,
        max_batch_size=32,
        max_top_k=32,
        use_topk_logprobs=False,
        model_config={
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": force,
                "num_links": 1,
                "topology": ttnn.Topology.Linear,
            }
        },
    )
    sampler = SamplingGenerator(args=cfg, mesh_device=mesh, tt_ccl=get_tt_ccl(mesh))
    case = dict(batch=batch, pad_batch=pad_batch, allow_force_argmax=force, mode=mode)
    logits = out = None
    try:
        params = (
            SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
            if mode == "greedy"
            else SamplingParams(temperature=[0.7] * batch, top_k=32, top_p=0.95)
        )
        formatted = format_sampling_params(params, 32)
        sampler.reset_sampling_params(formatted)
        assert sampler.tt_sampling.force_argmax_sampling == (force and mode == "greedy")
        if mode == "greedy":
            assert formatted.top_k == [1] * 32
            assert formatted.top_p == [0.0] * 32
            assert formatted.temperature == [1.0] * 32
        cpu, _ = known_logits(batch, 0)
        logits = ttnn.from_torch(
            cpu,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        if pad_batch and batch < 32:
            raw_logits = logits
            logits = ttnn.pad(raw_logits, padding=[(0, 0), (0, 0), (0, 32 - batch), (0, 0)], value=0.0)
            ttnn.deallocate(raw_logits)
        case.update(
            logits_shape=list(logits.shape),
            logits_padded_shape=list(logits.padded_shape),
            offsets_shape=list(sampler.tt_sampling.tt_indices_device_offsets.shape),
            force_argmax_active=sampler.tt_sampling.force_argmax_sampling,
            sampling_k=formatted.top_k[0],
            sampling_p=formatted.top_p[0],
        )
        print(case, flush=True)
        out = ttnn.from_torch(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        sampler.precompile(logits, tt_out_tok=out)
        sampler.capture_trace(logits, tt_out_tok=out, skip_precompile=True)
        assert any(slot["id"] is not None for slot in sampler._trace_states.values())
        assert not sampler.seed_manager.has_active_request_seed()
        for round_index in range(2):
            cpu, winners = known_logits(batch, round_index, physical_batch=32 if pad_batch else batch)
            host = ttnn.from_torch(
                cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3)
            )
            ttnn.copy_host_to_device_tensor(host, logits)
            sampler.sample(logits, tt_out_tok=out, enable_trace=True)
            actual = [ttnn.to_torch(t).flatten()[:batch].tolist() for t in ttnn.get_device_tensors(out)]
            row = dict(
                **case,
                round=round_index,
                expected=winners,
                actual=actual,
                passed=all(tokens == winners for tokens in actual),
            )
            results.append(row)
            print(row, flush=True)
            assert row["passed"], row
        if iterations:
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(iterations):
                sampler.sample(logits, tt_out_tok=out, enable_trace=True)
            ttnn.synchronize_device(mesh)
            results.append(dict(**case, host_wall_replay_us=(time.perf_counter() - start) * 1e6 / iterations))
    except Exception as error:
        results.append(dict(**case, passed=False, error=str(error)))
        raise
    finally:
        sampler.reset_trace()
        for tensor in (logits, out):
            if tensor is not None:
                ttnn.deallocate(tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1], choices=range(1, 33))
    parser.add_argument("--pad-batch", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("greedy", "sampled"), default=["greedy"])
    parser.add_argument("--force-mode", choices=("both", "split", "force"), default="both")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    results = []
    try:
        force_modes = (False, True) if args.force_mode == "both" else (args.force_mode == "force",)
        for batch in args.batch_sizes:
            for force in force_modes:
                for mode in args.modes:
                    check_case(
                        mesh,
                        batch=batch,
                        force=force,
                        mode=mode,
                        pad_batch=args.pad_batch,
                        iterations=args.iterations,
                        results=results,
                    )
        assert all(x.get("passed", True) for x in results)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
