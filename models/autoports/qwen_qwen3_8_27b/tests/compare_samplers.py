# SPDX-License-Identifier: Apache-2.0
"""TP4 sampler-only comparison; synthetic Qwen-sized logits, no model weights.

Reports host wall per traced replay separately from eager setup and semantics.
Use the same CCL, power-of-two padding, parameters, and explicit seeds per arm.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.sampling_ccl import SamplingCCL
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.tt_ccl import TT_CCL
from models.common.sampling.tt_sampling import TTSampling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, choices=[1, 8, 16, 32], default=8)
    p.add_argument("--sampler", choices=["sampling1d", "ttsampling"], required=True)
    p.add_argument(
        "--compact-ttsampling",
        action="store_true",
        help="Stage TTSampling offsets/masks at the actual batch before capture",
    )
    p.add_argument("--ccl", choices=["baseline", "async"], default="baseline")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.compact_ttsampling and a.sampler != "ttsampling":
        p.error("--compact-ttsampling requires --sampler ttsampling")
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024**2)
    traces = []
    result = dict(
        scope="sampler-only synthetic logits; not full-model or SWE performance",
        sampler=a.sampler,
        batch=a.batch,
        ccl=a.ccl,
        vocab=248320,
        padding=True,
        iterations=a.iterations,
        compact_ttsampling=a.compact_ttsampling,
    )
    source_root = Path(__file__).resolve().parents[4]
    result["source_sha256"] = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__).resolve(),
            source_root / "models/common/modules/sampling/sampling_1d.py",
            source_root / "models/common/sampling/tt_sampling.py",
            source_root / "models/autoports/qwen_qwen3_8_27b/tt/sampling_ccl.py",
        ]
    }
    result["ttnn_module"] = ttnn.__file__
    try:

        def put(x, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, shard=False):
            return ttnn.from_torch(
                x,
                device=mesh,
                dtype=dtype,
                layout=layout,
                mesh_mapper=(ttnn.ShardTensorToMesh(mesh, dim=-1) if shard else ttnn.ReplicateTensorToMesh(mesh)),
            )

        ccl = TT_CCL(mesh)
        if a.ccl == "async":
            ccl = SamplingCCL(ccl, persistent=False)
        physical_batch = 32 if a.sampler == "ttsampling" and not a.compact_ttsampling else a.batch
        result["sampler_physical_batch"] = physical_batch
        k = put(torch.full((physical_batch,), 20), ttnn.uint32)
        prob = put(torch.full((physical_batch,), 0.95), ttnn.bfloat16)
        temp = put(torch.ones(physical_batch), ttnn.bfloat16)
        seeds = put(torch.arange(32) + 1234, ttnn.uint32)
        output = put(torch.zeros(1, 1, 1, a.batch, dtype=torch.int32), ttnn.uint32)
        if a.sampler == "sampling1d":
            sampler = Sampling1D(
                vocab_size=248320,
                mesh_device=mesh,
                tt_ccl=ccl,
                max_batch_size=32,
                max_top_k=32,
                allow_force_argmax=False,
                pad_to_power_of_2=True,
            )
            sampler.load_device_buffers()

            def draw(x):
                return sampler(x, k=k, p=prob, temp=temp, seeds=seeds, tt_out_tok=output)

        else:
            args = SimpleNamespace(
                vocab_size=248320,
                padded_vocab_size=248320,
                max_batch_size=32,
                max_top_k=32,
                cluster_shape=mesh.shape,
                pad_logits_to_power_of_2=True,
            )
            sampler = TTSampling(
                mesh,
                ccl,
                args,
                k=torch.full((physical_batch,), 20),
                p=torch.full((physical_batch,), 0.95),
                temp=torch.ones(physical_batch),
            )
            if a.compact_ttsampling and a.batch < 32:
                offsets = sampler.tt_indices_device_offsets
                sampler.tt_indices_device_offsets = ttnn.slice(
                    offsets, [0, 0, 0, 0], [1, 1, a.batch, offsets.shape[-1]]
                )
                ttnn.deallocate(offsets)
            sampler.k_tensor, sampler.p_tensor, sampler.temp_tensor = k, prob, temp
            sampler.seeds_tt_tensor = seeds
            padded_output = (
                output if physical_batch == a.batch else put(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
            )

            def draw(x):
                if physical_batch > a.batch:
                    x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - a.batch), (0, 0)], 0.0)
                sampled = sampler(x, tt_out_tok=padded_output)[0]
                if physical_batch > a.batch:
                    sampled = ttnn.slice(sampled, [0, 0, 0, 0], [1, 1, 1, a.batch])
                    ttnn.copy(sampled, output)
                return output

        torch.manual_seed(701)
        host = torch.randn(1, 1, a.batch, 248320, dtype=torch.bfloat16)
        result["input_sha256"] = hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
        logits = put(host, ttnn.bfloat16, ttnn.TILE_LAYOUT, shard=True)
        for _ in range(3):
            draw(logits)
            ttnn.plus_one(seeds)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traces.append(trace)
        try:
            draw(logits)
            ttnn.plus_one(seeds)
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(a.iterations):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            timings.append((time.perf_counter() - start) * 1e6 / a.iterations)
        result["replay_us"] = timings
        result["median_replay_us"] = statistics.median(timings)
        # Reset seeds outside capture, then retain matched draws for comparison.
        seed_host = ttnn.from_torch(torch.arange(32) + 1234, dtype=ttnn.uint32)
        ttnn.copy_host_to_device_tensor(seed_host, seeds)
        draws = []
        for _ in range(32):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            tokens = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape(-1).tolist()
            assert len(tokens) == a.batch and all(0 <= t < 248320 for t in tokens)
            draws.append(tokens)
        result["draws"] = draws
        # Each greedy row has one maximum on each shard boundary in turn.
        for value, dst, dtype in [
            (torch.ones(physical_batch), k, ttnn.uint32),
            (torch.zeros(physical_batch), prob, ttnn.bfloat16),
        ]:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(value, dtype=dtype), dst)
        boundaries = []
        for token in [7, 62079, 62080, 124160, 186240, 248319]:
            x = torch.full_like(host, -100)
            x[..., token] = 100
            source = ttnn.from_torch(
                x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)
            )
            ttnn.copy_host_to_device_tensor(source, logits)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            tokens = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape(-1).tolist()
            assert tokens == [token] * a.batch, (token, tokens)
            boundaries.append(token)
        result["greedy_boundaries"] = boundaries
        result["status"] = "pass"
    except BaseException as exc:
        result.update(status="fail", error=repr(exc))
        raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
    print(json.dumps({k: v for k, v in result.items() if k != "draws"}), flush=True)


if __name__ == "__main__":
    main()
