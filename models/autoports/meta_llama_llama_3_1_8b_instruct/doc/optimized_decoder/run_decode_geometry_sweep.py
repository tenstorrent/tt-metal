# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_optimized_decoder import (
    EMITTED_BATCH,
    HF_MODEL,
    LAYER_IDX,
    _pcc,
    _real_layer_state_dict,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    BF16,
    DRAM,
    OptimizedDecoder,
    _core_coord,
    _width_sharded_memcfg,
)

OUT_DIR = Path(__file__).parent
OUT_JSON = OUT_DIR / "decode_geometry_sweep.json"
OUT_MD = OUT_DIR / "decode_geometry_sweep.md"


def _lofi():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _program_config(mesh_device, *, out_tiles: int, in0_block_w: int, fused_activation=None):
    cores = _core_coord(mesh_device)
    core_count = int(cores.x) * int(cores.y)
    per_core_n = max(1, out_tiles // core_count)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=cores,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=per_core_n,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def _dram_sharded_program_config(mesh_device, *, out_tiles: int, in0_block_w: int, fused_activation=None):
    cores = _core_coord(mesh_device)
    core_count = int(cores.x) * int(cores.y)
    per_core_n = max(1, out_tiles // core_count)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fused_activation=fused_activation,
    )


def _run_mlp(decoder, mlp_input, *, memory_config, input_memory_config=None, program_block_w=None, dram_sharded=False):
    x = mlp_input
    if input_memory_config is not None:
        x = ttnn.to_memory_config(x, input_memory_config)
    gate_cfg = None
    up_cfg = None
    down_cfg = None
    if program_block_w is not None:
        activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        if dram_sharded:
            gate_cfg = _dram_sharded_program_config(
                decoder.mesh_device, out_tiles=448, in0_block_w=program_block_w, fused_activation=activation
            )
            up_cfg = _dram_sharded_program_config(decoder.mesh_device, out_tiles=448, in0_block_w=program_block_w)
            down_cfg = _dram_sharded_program_config(decoder.mesh_device, out_tiles=128, in0_block_w=program_block_w)
        else:
            gate_cfg = _program_config(
                decoder.mesh_device, out_tiles=448, in0_block_w=program_block_w, fused_activation=activation
            )
            up_cfg = _program_config(decoder.mesh_device, out_tiles=448, in0_block_w=program_block_w)
            down_cfg = _program_config(decoder.mesh_device, out_tiles=128, in0_block_w=program_block_w)
    gate = ttnn.linear(
        x,
        decoder.weights["gate_proj"],
        dtype=BF16,
        memory_config=memory_config,
        activation=None if gate_cfg is not None else "silu",
        compute_kernel_config=_lofi(),
        program_config=gate_cfg,
    )
    up = ttnn.linear(
        x,
        decoder.weights["up_proj"],
        dtype=BF16,
        memory_config=memory_config,
        compute_kernel_config=_lofi(),
        program_config=up_cfg,
    )
    prod = ttnn.multiply(gate, up, memory_config=memory_config)
    return ttnn.linear(
        prod,
        decoder.weights["down_proj"],
        dtype=BF16,
        memory_config=memory_config,
        compute_kernel_config=_lofi(),
        program_config=down_cfg,
    )


def _measure(candidate, decoder, mlp_input, golden):
    try:
        out = candidate["runner"](decoder, mlp_input)
        ttnn.synchronize_device(decoder.mesh_device)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            out = candidate["runner"](decoder, mlp_input)
            ttnn.synchronize_device(decoder.mesh_device)
            times.append((time.perf_counter() - start) * 1000.0)
        actual = ttnn.to_torch(out)
        return {
            "name": candidate["name"],
            "status": "ok",
            "best_ms": round(min(times), 3),
            "median_ms": round(sorted(times)[len(times) // 2], 3),
            "pcc_vs_final_mlp": round(_pcc(golden, actual), 6),
            "notes": candidate["notes"],
        }
    except Exception as exc:
        return {
            "name": candidate["name"],
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
            "notes": candidate["notes"],
        }


def main():
    hf_config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            _real_layer_state_dict(LAYER_IDX),
            hf_config=hf_config,
            layer_idx=LAYER_IDX,
            mesh_device=mesh,
            batch=EMITTED_BATCH,
        )
        torch.manual_seed(20260709)
        host_input = torch.randn(1, 1, EMITTED_BATCH, hf_config.hidden_size, dtype=torch.bfloat16)
        mlp_input = ttnn.from_torch(host_input, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        width_4096 = _width_sharded_memcfg(mesh, EMITTED_BATCH, hf_config.hidden_size)
        width_14336 = _width_sharded_memcfg(mesh, EMITTED_BATCH, hf_config.intermediate_size)

        final_out = _run_mlp(decoder, mlp_input, memory_config=DRAM)
        ttnn.synchronize_device(mesh)
        golden = ttnn.to_torch(final_out)

        candidates = [
            {
                "name": "final_dram_interleaved_auto",
                "notes": "Previous auto MLP baseline: BFP4 weights, LoFi compute, DRAM-interleaved activations, auto 1D multicast config.",
                "runner": lambda d, x: _run_mlp(d, x, memory_config=DRAM),
            },
            {
                "name": "l1_width_input_dram_output_auto",
                "notes": "Forge-seeded L1 WIDTH_SHARDED input, but keep DRAM output to avoid downstream reshard.",
                "runner": lambda d, x: _run_mlp(d, x, input_memory_config=width_4096, memory_config=DRAM),
            },
            {
                "name": "l1_width_input_width_output_auto",
                "notes": "Forge-seeded WIDTH_SHARDED residual/MLP path with L1 WIDTH_SHARDED input and output.",
                "runner": lambda d, x: _run_mlp(d, x, input_memory_config=width_4096, memory_config=width_14336),
            },
            *[
                {
                    "name": f"explicit_1d_in0_block_w_{block_w}",
                    "notes": f"Precision-locked BFP4/LoFi MLP with explicit 1D multicast configs and in0_block_w={block_w}.",
                    "runner": lambda d, x, block_w=block_w: _run_mlp(d, x, memory_config=DRAM, program_block_w=block_w),
                }
                for block_w in (1, 2, 4, 8, 16, 32, 64)
            ],
            *[
                {
                    "name": f"l1_width_input_dram_output_explicit_w_{block_w}",
                    "notes": f"L1 WIDTH_SHARDED input with explicit fused-SiLU program configs, DRAM output, in0_block_w={block_w}.",
                    "runner": lambda d, x, block_w=block_w: _run_mlp(
                        d, x, input_memory_config=width_4096, memory_config=DRAM, program_block_w=block_w
                    ),
                }
                for block_w in (1, 2, 4, 8, 16, 32, 64)
            ],
            *[
                {
                    "name": f"l1_width_input_width_output_explicit_w_{block_w}",
                    "notes": f"L1 WIDTH_SHARDED input/output with explicit fused-SiLU program configs, in0_block_w={block_w}.",
                    "runner": lambda d, x, block_w=block_w: _run_mlp(
                        d, x, input_memory_config=width_4096, memory_config=width_14336, program_block_w=block_w
                    ),
                }
                for block_w in (1, 2, 4, 8, 16, 32, 64)
            ],
            {
                "name": "dram_sharded_program_in0_block_w_2",
                "notes": "Forge-seeded legal DRAM-sharded matmul program on BFP4/LoFi MLP with in0_block_w=2.",
                "runner": lambda d, x: _run_mlp(
                    d,
                    x,
                    input_memory_config=width_4096,
                    memory_config=width_14336,
                    program_block_w=2,
                    dram_sharded=True,
                ),
            },
            {
                "name": "dram_sharded_program_in0_block_w_4",
                "notes": "Forge-seeded DRAM-sharded matmul program on BFP4/LoFi MLP.",
                "runner": lambda d, x: _run_mlp(
                    d,
                    x,
                    input_memory_config=width_4096,
                    memory_config=width_14336,
                    program_block_w=4,
                    dram_sharded=True,
                ),
            },
            {
                "name": "dram_sharded_program_in0_block_w_8",
                "notes": "Forge-seeded DRAM-sharded matmul program on BFP4/LoFi MLP with larger in0_block_w=8.",
                "runner": lambda d, x: _run_mlp(
                    d,
                    x,
                    input_memory_config=width_4096,
                    memory_config=width_14336,
                    program_block_w=8,
                    dram_sharded=True,
                ),
            },
        ]
        rows = [_measure(candidate, decoder, mlp_input, golden) for candidate in candidates]
        payload = {
            "model": HF_MODEL,
            "layer_idx": LAYER_IDX,
            "batch": EMITTED_BATCH,
            "policy": "MLP weights BFP4, LoFi compute, BF16 activations",
            "device_grid": str(mesh.compute_with_storage_grid_size()),
            "rows": rows,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
        lines = [
            "# Decode MLP Geometry Sweep",
            "",
            "Policy: MLP weights BFP4, LoFi compute, BF16 activations. Real layer-0 weights.",
            "",
            "| Candidate | Status | Best ms | PCC vs final | Notes / blocker |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for row in rows:
            if row["status"] == "ok":
                detail = row["notes"]
                lines.append(
                    f"| `{row['name']}` | ok | {row['best_ms']:.3f} | {row['pcc_vs_final_mlp']:.6f} | {detail} |"
                )
            else:
                detail = (
                    (row["error"].replace("\n", " ")[:300] + "...")
                    if len(row["error"]) > 300
                    else row["error"].replace("\n", " ")
                )
                lines.append(f"| `{row['name']}` | blocked |  |  | {row['notes']} Blocker: `{detail}` |")
        OUT_MD.write_text("\n".join(lines) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
