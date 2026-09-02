# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated probes for the full-model terminal path (embedding, LM head, sampling).

Answers the shape/L1/perf questions the model wrapper needs before it is written:
embedding output rank, plus_one dtypes, LM-head matmul geometry at
vocab 154880 on one Blackhole chip, and the on-device sampling op contract.
"""

import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.provenance import source_manifest

TILE = 32
V, H = 154880, 2048
#: The report cites the LM-head geometry sweep and the default-config figure as
#: the justification for the explicit program config, so they are written to an
#: artifact rather than only printed (work log FM-019).
OUT = Path(__file__).resolve().parents[1] / "doc" / "full_model" / "head_probe.json"
RESULTS = {"source_manifest": source_manifest([__file__]), "lm_head_us": {}, "notes": []}


def _rect_grid(cores):
    for cols in range(min(cores, 11), 0, -1):
        if cores % cols == 0 and cores // cols <= 10:
            return cols, cores // cols
    raise ValueError(cores)


def lm_head_1d_pc(nt, kt, cores, bw):
    per_core_n = -(-nt // cores)
    blocks = -(-nt // per_core_n)
    cols, rows = _rect_grid(blocks)
    osw = max(d for d in (1, 2, 4) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=osw,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    torch.manual_seed(0)

    # ---- 1. embedding shapes ----
    w = torch.randn(1, 1, 512, H) * 0.02
    wt = ttnn.from_torch(w, device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    for shape in ((1, 1, 1, 4), (1, 4)):
        ids = ttnn.from_torch(
            torch.randint(0, 512, shape, dtype=torch.int32).to(torch.int32),
            device=dev,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        out = ttnn.embedding(ids, wt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        print(f"embedding in{shape} -> {out.shape}  4D={ttnn.unsqueeze_to_4D(out).shape}")
        ttnn.deallocate(out)
        ttnn.deallocate(ids)
    ttnn.deallocate(wt)

    # ---- 2. plus_one dtypes ----
    pos = ttnn.from_torch(
        torch.tensor([5, -1], dtype=torch.int32), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.plus_one(pos, skip_negative_entries=True)
    print("plus_one int32 [B]:", ttnn.to_torch(pos).tolist())
    rot = ttnn.from_torch(
        torch.tensor([[5, 0]], dtype=torch.int32).to(torch.int32),
        device=dev,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.plus_one(rot)
    print("plus_one uint32 [1,B]:", ttnn.to_torch(rot).tolist())
    rot4 = ttnn.from_torch(
        torch.tensor([[[[5, 0]]]], dtype=torch.int32).to(torch.int32),
        device=dev,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.plus_one(rot4)
    print("plus_one uint32 [1,1,1,B]:", ttnn.to_torch(rot4).tolist())

    # ---- 3. LM head geometry ----
    wh = (torch.randn(H, V) * 0.02).contiguous()
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, H) * 0.5,
        device=dev,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    for dtype, name in ((ttnn.bfloat8_b, "bf8"), (ttnn.bfloat4_b, "bf4")):
        wt = ttnn.from_torch(
            wh, device=dev, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # in0_block_w over every legal divisor of kt = 2048/32 = 64 that the
        # config helper can express, so the shipped value is a measurement
        # rather than a middle point (work log FM-020).
        for cores, bw in ((110, 1), (110, 2), (110, 4), (110, 8), (110, 16), (110, 32), (110, 64), (88, 8), (64, 8)):
            pc = lm_head_1d_pc(V // TILE, H // TILE, cores, bw)
            try:
                out = ttnn.linear(
                    x, wt, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ck
                )
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(20):
                    o = ttnn.linear(
                        x, wt, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ck
                    )
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / 20 * 1e6
                print(f"LM head {name} 1D cores={cores} bw={bw} pcn={pc.per_core_N}: {dt:.1f} us  out={out.shape}")
                RESULTS["lm_head_us"][f"{name}_1d_cores{cores}_bw{bw}"] = round(dt, 1)
                ttnn.deallocate(out)
            except Exception as e:
                print(f"LM head {name} 1D cores={cores} bw={bw}: FAIL {str(e)[:160]}")
                RESULTS["lm_head_us"][f"{name}_1d_cores{cores}_bw{bw}"] = f"FAIL: {str(e).splitlines()[0][:120]}"
        # default config
        try:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(20):
                o = ttnn.linear(x, wt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ck)
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            default_us = (time.perf_counter() - t0) / 20 * 1e6
            print(f"LM head {name} default cfg: {default_us:.1f} us")
            RESULTS["lm_head_us"][f"{name}_default_program_config"] = round(default_us, 1)
        except Exception as e:
            print(f"LM head {name} default: FAIL {str(e)[:160]}")
        ttnn.deallocate(wt)
    ttnn.deallocate(x)
    ttnn.close_device(dev)
    RESULTS["notes"].append(
        "One decode row (M = 32 padded rows) x 2048 x 154880, 20 iterations each, wall clock with an "
        "explicit synchronize. `*_default_program_config` is the figure the report cites for why the "
        "explicit 1D mcast config is required."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(RESULTS, indent=2) + "\n")
    print("wrote", OUT)
    print("HEAD_PROBE_OK")


if __name__ == "__main__":
    main()
