# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real two-layer wrapper probe for terminal geometry and no-readback replay."""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D

ROOT = Path(__file__).parents[1]


def repack_terminal_weights(model, bank_multiple):
    """Re-upload the same rank-ordered host weights with terminal-only bank padding."""
    original_head = model.lm_head
    cfg = original_head.config
    memories, weights = [], []
    for source, width, memory, program in zip(
        cfg.output_weights, cfg.output_split_sizes, cfg.weights_memcfgs, cfg.program_configs
    ):
        spec = memory.shard_spec
        banks = spec.grid.num_cores()
        width_tiles = width // 32
        bank_tiles = math.ceil(width_tiles / (banks * bank_multiple)) * bank_multiple
        readers = program.num_workers_per_dram_bank
        reader_tiles = math.ceil(width_tiles / (banks * readers))
        if bank_tiles % readers or (readers > 1 and bank_tiles != readers * reader_tiles):
            raise ValueError(
                f"Bank padding gives {bank_tiles} tiles/bank for width {width}; "
                f"{readers} readers require {readers * reader_tiles}"
            )
        repacked_memory = ttnn.MemoryConfig(
            memory.memory_layout,
            memory.buffer_type,
            ttnn.ShardSpec(spec.grid, [cfg.dim, bank_tiles * 32], spec.orientation),
        )
        memories.append(repacked_memory)
        # A fresh LazyWeight is essential: changing the old object's memory_config
        # would reuse its already materialized tensor. Keep the canonical host
        # source, dtype and mesh mapper; only bank storage padding changes.
        weights.append(
            LazyWeight(
                source=source.source,
                pad_value=source.pad_value,
                dtype=source.dtype,
                device=source.device,
                mesh_mapper_config=source.mesh_mapper_config,
                memory_config=repacked_memory,
                layout=source.layout,
            )
        )
    repacked_head = LMHead1D.from_config(replace(cfg, output_weights=weights, weights_memcfgs=memories))
    repacked_head.load_device_weights()
    for weight, source, memory in zip(repacked_head.output_weights, weights, memories):
        if weight.dtype != source.dtype or weight.memory_config() != memory:
            raise RuntimeError("Repacked LM-head weight dtype or storage differs from the requested configuration")
    model.lm_head = repacked_head
    model.lm_decode = repacked_head.output_weights[0]
    for weight in original_head.output_weights:
        ttnn.deallocate(weight)


def terminal_weight_geometry(model):
    cfg = model.lm_head.config
    devices = cfg.mesh_device.get_num_devices()
    geometry = []
    for width, source, weight, program in zip(
        cfg.output_split_sizes, cfg.output_weights, model.lm_head.output_weights, cfg.program_configs
    ):
        memory = weight.memory_config()
        spec = memory.shard_spec
        physical_width = source.source.shape[-1] // devices
        geometry.append(
            dict(
                logical_output_width=width,
                source_shape=list(source.source.shape),
                tensor_shape=list(weight.shape),
                tensor_padded_shape=list(weight.padded_shape),
                per_device_source_width=physical_width,
                dtype=str(weight.dtype),
                memory_config=str(memory),
                dram_banks=spec.grid.num_cores(),
                bank_shard_shape=list(spec.shape),
                bank_width_tiles=spec.shape[-1] // 32,
                storage_columns_per_device=spec.grid.num_cores() * spec.shape[-1],
                readers_per_bank=program.num_workers_per_dram_bank,
                output_per_core_n_tiles=program.per_core_N,
                output_storage_cores=math.ceil(physical_width / (32 * program.per_core_N)),
            )
        )
    return geometry


def main(args):
    if args.bank_multiple is not None and args.bank_multiple < 1:
        raise ValueError("--bank-multiple must be positive")
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    gen = None
    try:
        started = time.perf_counter()
        gen = build_generator(ROOT, mesh, layer_indices=[0, 3], terminal_policy=json.loads(args.policy), prepare=False)
        if args.output_cores:
            policy = gen.model.terminal_policy
            gen.model.lm_head.config.program_configs = [
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=policy["block_w"],
                    per_core_M=1,
                    per_core_N=math.ceil(width / (32 * args.output_cores)),
                    num_workers_per_dram_bank=policy["readers"],
                )
                for width in gen.model.lm_head.config.output_split_sizes
            ]
        if args.bank_multiple is not None:
            repack_terminal_weights(gen.model, args.bank_multiple)
        weight_geometry = terminal_weight_geometry(gen.model)
        print("TERMINAL_WEIGHT_STORAGE", json.dumps(weight_geometry), flush=True)
        gen.prepare()
        result = dict(
            policy=gen.model.terminal_policy,
            output_storage_cores=args.output_cores,
            bank_multiple=args.bank_multiple,
            head_weight_storage=weight_geometry,
            input_storage_cores=gen.model.lm_input_mem.shard_spec.grid.num_cores(),
            input_shard_shape=list(gen.model.lm_input_mem.shard_spec.shape),
            layers=[0, 3],
            setup_s=time.perf_counter() - started,
            tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
            skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "tt/model.py", ROOT / "tt/generator.py", Path(__file__))
            },
        )
        prompt = [42] * 128
        outputs, measurements = {}, []
        for mode in ("sample", "argmax"):
            outputs[mode] = gen.generate(prompt, 32, sampler_mode=mode, stop_on_eos=False)
            for repeat in range(3):
                gen.generate(prompt, 1, stop_on_eos=False)
                before = dict(gen.counters)
                start = time.perf_counter()
                gen.decode_steps(31, sampler_mode=mode)
                ttnn.synchronize_device(mesh)
                elapsed = time.perf_counter() - start
                delta = {k: v - before.get(k, 0) for k, v in gen.counters.items()}
                final = gen._read_tokens()[0]
                assert final == outputs[mode][-1], (final, outputs[mode][-1])
                measurements.append(
                    dict(
                        mode=mode,
                        repeat=repeat,
                        decode_ms=elapsed * 1000 / 31,
                        ttft_ms=gen.last_ttft_seconds * 1000,
                        counters=delta,
                    )
                )
        # Record real checkpoint, real activation logits for same-input policy comparisons.
        gen.generate(prompt, 1, stop_on_eos=False)
        logits = gen._read_logits(gen.logits).flatten().float()
        torch.save(logits, args.output.with_suffix(".logits.pt"))
        result["terminal_replay_ms"] = []
        for _ in range(3):
            started = time.perf_counter()
            for _ in range(32):
                gen._execute(("head", 128))
            ttnn.synchronize_device(mesh)
            result["terminal_replay_ms"].append((time.perf_counter() - started) * 1000 / 32)
        result.update(
            measurements=measurements,
            outputs=outputs,
            median_decode_ms={
                mode: statistics.median(r["decode_ms"] for r in measurements if r["mode"] == mode) for mode in outputs
            },
            actual_head_dtypes=[str(w.dtype) for w in gen.model.lm_head.output_weights],
            head_programs=[str(x) for x in gen.model.lm_head.config.program_configs],
            trace_ids={str(k): str(v) for k, v in gen.traces.items()},
            counters=dict(gen.counters),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("TERMINAL_PROBE_PASS", result["median_decode_ms"], flush=True)
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-cores", type=int)
    parser.add_argument(
        "--bank-multiple",
        type=int,
        help="Repack only LM-head DRAM bank widths to this tile multiple before preparing traces",
    )
    main(parser.parse_args())
