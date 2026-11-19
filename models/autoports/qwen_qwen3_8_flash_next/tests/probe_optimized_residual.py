# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact gated-residual storage candidates through read -> write -> next read.

Serial device experiment; invoke under an external timeout. Real layer-0 weights,
seeded representative BF16 activations, logical rows 1 and 32. Not full-decoder
acceptance. No native RMSNorm or altered projection arithmetic is introduced.
The width-sharded candidate retains the residual in that storage and explicitly
adapts each read consumer to interleaved L1 before reshape/permute. All residual
boundary conversions are inside the measured trace. API errors remain unresolved
experiments, not evidence that an optimization is impossible.
"""

import argparse
import gc
import json
import os
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_optimized_dram import measure
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader, linear, rms
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import (
    _RESIDUAL_WRITE_COMPUTE,
    _RESIDUAL_WRITE_READER,
    _RESIDUAL_WRITE_WRITER,
    FusedPreciseProjection,
    FusedResidualWriter,
)

# Same writer body and kernel sources as fused_decoder.fused_residual_write.
# Only its DRAM-only guard is removed. TensorAccessorArgs describes the actual
# input/output allocations, including L1 and width-sharded tile mappings.


def adapted_residual_write(residual, block, injection):
    tokens = residual.shape[-2]
    if (
        tuple(residual.shape) != (1, 1, tokens, 10240)
        or tuple(block.shape) != (1, 1, tokens, 2560)
        or tuple(injection.shape) != (1, 4, tokens, 1)
        or tokens < 1
        or any(t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT for t in (residual, block, injection))
        or any(t.device() != residual.device() for t in (block, injection))
    ):
        raise ValueError("Residual write requires matching BF16 TILE four-stream inputs")
    padded_tokens = (tokens + 31) // 32 * 32
    for tensor, expected in (
        (residual, (1, 1, padded_tokens, 10240)),
        (block, (1, 1, padded_tokens, 2560)),
        (injection, (1, 4, padded_tokens, 32)),
    ):
        tile = tensor.get_tile()
        if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
            raise ValueError("Residual write requires standard 32x32 tiles")
        if tuple(tensor.padded_shape) != expected:
            raise ValueError("Residual write received unexpected physical padding")
    mesh = residual.device()
    output = ttnn.empty_like(residual)
    mt = (tokens + 31) // 32
    total = mt * 320
    grid = mesh.compute_with_storage_grid_size()
    cores = min(total, grid.x * grid.y)
    ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
            for i in range(cores)
        ]
    )
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(cores):
        x, y = i % grid.x, i // grid.x
        start, end = total * i // cores, total * (i + 1) // cores
        reader[x][y] = [
            block.buffer_address(),
            injection.buffer_address(),
            residual.buffer_address(),
            start,
            end - start,
            mt,
        ]
        compute[x][y] = [end - start]
        writer[x][y] = [output.buffer_address(), start, end - start]
    ct = []
    for t in (block, injection, residual):
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=args,
            runtime_args=rt,
            config=config,
        )
        for source, args, rt, config in [
            (_RESIDUAL_WRITE_READER, ct, reader, ttnn.ReaderConfigDescriptor()),
            (
                _RESIDUAL_WRITE_COMPUTE,
                [],
                compute,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                    dst_full_sync_en=False,
                ),
            ),
            (
                _RESIDUAL_WRITE_WRITER,
                ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                writer,
                ttnn.WriterConfigDescriptor(),
            ),
        ]
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(4)
    ]
    ttnn.generic_op(
        [block, injection, residual, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return output


def adapted_read(component, residual, counts, *, consumer_dram=False):
    """Same exact arithmetic as GatedResidual.read, with explicit boundaries.

    Keep the stored residual untouched. Interleave only the reader's view. The
    expanded SFPU projection products can be tens of MiB at 32 rows, so their
    input moves to DRAM once and is shared by both same-input projections.
    """

    def convert(value, target, reason):
        if value.memory_config() == target:
            return value
        counts[reason] = counts.get(reason, 0) + 1
        return ttnn.to_memory_config(value, target)

    work = ttnn.DRAM_MEMORY_CONFIG if consumer_dram else ttnn.L1_MEMORY_CONFIG
    x = convert(residual, work, "stored_residual_to_read_consumer")
    tokens = x.shape[-2]
    grouped = ttnn.permute(ttnn.reshape(x, (1, tokens, component.streams, component.width)), (0, 2, 1, 3))
    normalized = rms(grouped, component.weight, component.eps)
    flattened = ttnn.reshape(
        ttnn.permute(normalized, (0, 2, 1, 3)), (1, 1, tokens, component.streams * component.width)
    )
    flattened = convert(flattened, ttnn.DRAM_MEMORY_CONFIG, "normalized_to_dram_projection_workspace")
    scaled = ttnn.multiply(component.precise_down(flattened), 1 / component.streams)
    mixing = ttnn.typecast(ttnn.silu(ttnn.typecast(scaled, ttnn.float32)), ttnn.bfloat16)
    mixing = ttnn.typecast(ttnn.sigmoid(ttnn.typecast(component.precise_up(mixing), ttnn.float32)), ttnn.bfloat16)
    mixing = ttnn.permute(ttnn.reshape(mixing, (1, tokens, component.streams, component.width)), (0, 2, 1, 3))
    mixed = ttnn.typecast(
        ttnn.mean(ttnn.typecast(ttnn.multiply(mixing, normalized), ttnn.float32), dim=1, keepdim=True),
        ttnn.bfloat16,
    )
    scaled_injection = ttnn.multiply(linear(flattened, component.inject, component.compute), 1 / component.streams)
    injection = ttnn.multiply(
        ttnn.typecast(ttnn.sigmoid(ttnn.typecast(scaled_injection, ttnn.float32)), ttnn.bfloat16), 2
    )
    injection = ttnn.permute(ttnn.reshape(injection, (1, 1, tokens, component.streams)), (0, 3, 2, 1))
    return mixed, injection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="1,32")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--shard-cores", type=int, default=32)
    parser.add_argument("--projection-group", type=int, default=64)
    parser.add_argument(
        "--variants",
        default="dram,l1,width_sharded,width_sharded_dram_consumer",
        help="Comma separated: dram,l1,width_sharded,width_sharded_dram_consumer",
    )
    args = parser.parse_args()
    from ttnn.tools import trace_allocation_tracker

    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING, "Set TT_METAL_TRACE_ALLOC_TRACKING=1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    assert args.repeats > 0
    torch.set_num_threads(16)
    torch.manual_seed(829)
    rows = [int(value) for value in args.rows.split(",")]
    assert all(1 <= n <= 32 for n in rows)
    prefix = "attn_hyper_connection."
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    state = load_weights(args.snapshot, 0, prefix)
    report = {
        "weights": "real layer0 attn_hyper_connection checkpoint",
        "activations": "seeded representative BF16; not recorded full-model activations",
        "shape": [config.hc_count, config.hidden_size],
        "regime": "serial single-chip component trace",
        "trace_allocation_tracking": True,
        "projection_group": args.projection_group,
        "reference": "Inherited GatedResidual.read and FusedResidualWriter.write with fused default projections",
        "boundary": "read(residual), write(residual, mixed, injection), read(updated_residual)",
        "candidates": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32000000)
    try:
        component = FusedResidualWriter(WeightLoader(state, mesh), prefix, config)
        for role in ("down", "up"):
            name = "precise_" + role
            setattr(
                component,
                name,
                FusedPreciseProjection(
                    getattr(component, name),
                    component.compute,
                    args.projection_group,
                    widen_in_unpack=True,
                    host_weight=state[prefix + "input_mix_weight_" + role + ".weight"],
                ),
            )
        for count in rows:
            host = (torch.randn(1, 1, count, config.hidden_size * config.hc_count) * 0.02).bfloat16()
            initial = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            def baseline():
                mixed, injection = component.read(initial)
                updated = component.write(initial, mixed, injection)
                next_mixed, next_injection = component.read(updated)
                return [mixed, injection, updated, next_mixed, next_injection]

            reference_device = baseline()
            reference = [ttnn.to_torch(value) for value in reference_device]
            del reference_device
            for variant in args.variants.split(","):
                entry = {"variant": variant, "logical_rows": count}
                report["candidates"].append(entry)
                counts = {}
                stored = None
                try:
                    if variant == "dram":
                        run = baseline
                        entry["residual_storage"] = "interleaved DRAM"
                    else:
                        if variant == "l1":
                            memory = ttnn.L1_MEMORY_CONFIG
                        elif variant in ("width_sharded", "width_sharded_dram_consumer"):
                            width = config.hidden_size * config.hc_count
                            assert width % (32 * args.shard_cores) == 0
                            grid = ttnn.num_cores_to_corerangeset(
                                args.shard_cores, mesh.compute_with_storage_grid_size(), row_wise=True
                            )
                            memory = ttnn.MemoryConfig(
                                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                ttnn.BufferType.L1,
                                ttnn.ShardSpec(grid, (32, width // args.shard_cores), ttnn.ShardOrientation.ROW_MAJOR),
                            )
                        else:
                            raise ValueError(f"Unknown variant: {variant}")
                        # Initial admission occurs once, outside trace. Each
                        # repeated stack boundary retains candidate storage.
                        stored = ttnn.to_memory_config(initial, memory)

                        def run():
                            counts.clear()
                            kwargs = {"consumer_dram": variant == "width_sharded_dram_consumer"}
                            mixed, injection = adapted_read(component, stored, counts, **kwargs)
                            updated = adapted_residual_write(stored, mixed, injection)
                            assert updated.memory_config() == memory
                            next_mixed, next_injection = adapted_read(component, updated, counts, **kwargs)
                            return [mixed, injection, updated, next_mixed, next_injection]

                        entry["residual_storage"] = str(memory)
                    entry.update(measure(ttnn, mesh, run, reference, args.repeats))
                    entry["boundary_conversions_per_trace"] = dict(counts)
                    entry["boundary_conversion_count"] = sum(counts.values())
                except Exception as error:
                    entry.update(status="unresolved_error", error=repr(error), traceback=traceback.format_exc())
                    # A DRAM-consumer retry follows width-sharded L1-consumer:
                    # this separates writer mapping from unsupported read ops.
                    entry["decision"] = "Investigate/adapt; first API failure is not a rejection"
                finally:
                    stored = None
                    gc.collect()
                    save()
                print(
                    json.dumps(
                        {
                            k: entry[k]
                            for k in ("variant", "logical_rows", "status", "traced_ms", "whole_pcc", "error")
                            if k in entry
                        }
                    ),
                    flush=True,
                )
            del initial
        save()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
