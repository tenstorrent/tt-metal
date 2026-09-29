"""Precision-locked LMHead1D working-shard/K-block search on real activations."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cores", type=int, default=32)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--split", type=int, default=16384)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--precision-config", type=Path)
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    trace = None
    result = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    try:
        gen = K2Generator(
            mesh,
            override_num_layers=args.layers,
            head_workers=args.workers,
            head_split_size=args.split,
            head_k=2,
            precision_config=args.precision_config,
        )
        result.update(
            dtype=str(gen.model.head.dtype),
            fidelity=str(gen.model.head_compute.math_fidelity),
            fp32_dest_acc_en=gen.model.head_compute.fp32_dest_acc_en,
            precision_config=gen.model.precision_config,
        )
        saved = []
        original = gen.model.final_logits

        def save_terminal(x, *, decode):
            if decode:
                saved[:] = [x]
            return original(x, decode=decode)

        gen.model.final_logits = save_terminal
        prompt = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 20)[:128]
        gen._ensure_owned_cache(1, 256)
        initial = gen.prefill_forward(
            torch.tensor([prompt]),
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[128],
            sampling_mode="device_logits",
        )
        ttnn.synchronize_device(mesh)
        del initial
        x = saved[0]
        gen.model.final_logits = original
        baseline = original(x, decode=True)
        reference = ttnn.to_torch(baseline, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        del baseline
        cfg = gen.model.head_decode.config
        cfg.input_memcfg = ttnn.create_sharded_memory_config(
            (32, 4096 // args.cores),
            core_grid=ttnn.num_cores_to_corerangeset(args.cores, mesh.compute_with_storage_grid_size(), True),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        cfg.program_configs = [
            ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=args.k, per_core_M=1, per_core_N=args.split // 1024, num_workers_per_dram_bank=args.workers
            )
        ] * (65536 // args.split)
        # Match the complete final_logits consumer path. A wrapper around
        # head_decode would add fixed16 -> candidate after final_logits has
        # already resharded to the old default, biasing non-default layouts.
        gen.model.head_input_memory = cfg.input_memcfg
        result["direct_head_input_layout"] = True
        warm = original(x, decode=True)
        ttnn.synchronize_device(mesh)
        del warm
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = original(x, decode=True)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        times = []
        for _ in range(5):
            begin = time.perf_counter()
            for _ in range(64):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter() - begin) * 1000 / 64)
        actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        a, b = actual.float().flatten(), reference.float().flatten()
        pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
        result.update(
            ms=times,
            median_ms=statistics.median(times),
            pcc=pcc,
            equal=torch.equal(actual, reference),
            top1_equal=int(a.argmax()) == int(b.argmax()),
            input_memory=str(cfg.input_memcfg),
            program=str(cfg.program_configs[0]),
        )
        assert pcc >= 0.9999 and result["top1_equal"]
        result["pass"] = True
    except Exception as error:
        result.update(error=str(error), passed=False)
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
