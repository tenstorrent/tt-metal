# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: localise TT attention error on captured native CUDA inputs.

RoPE is applied on the host with eager BF16 rounding (bitwise-matched earlier), so
every TT variant receives identical rotated Q/K and V. Host references use FP64.

Usage: attention_breakdown.py CAPTURE_ROOT OUTPUT_DIR [BLOCKS]
CAPTURE_ROOT/<block>/attention.pt holds CUDA q, k, v (pre-RoPE), rope and output.
Opens logical device 0; select the reserved card with TT_VISIBLE_DEVICES.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
BLOCKS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["00", "21", "41"]
OUT.mkdir(parents=True, exist_ok=True)
report = {"scope": "captured native CUDA attention inputs; host BF16-eager RoPE", "results": {}}


def save():
    (OUT / "attention-breakdown.json").write_text(json.dumps(report, indent=2) + "\n")


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    return {
        "pcc": float((ac @ bc) / (ac.norm() * bc.norm()).clamp_min(1e-30)),
        "relative_rms": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max()),
        "gain": float((a @ b) / (b @ b).clamp_min(1e-30)),
        "exact_fraction": float((a == b).double().mean()),
    }


def record(name, value, refs):
    for rname, r in refs.items():
        m = metrics(value, r)
        report["results"][f"{name}|vs_{rname}"] = m
        print(
            f"{name:40s} vs {rname:12s} rRMS={m['relative_rms']*100:.4f}% gain={m['gain']:.5f} exact={m['exact_fraction']:.3f}",
            flush=True,
        )
    save()


def host_rope(x, cos, sin):
    # Eager BF16: each op rounds to BF16 (round-to-nearest-even on CPU).
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


import ttnn as t

mesh = t.open_mesh_device(
    mesh_shape=t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=131072, trace_region_size=0
)
compute = t.init_device_compute_kernel_config(
    mesh.arch(), math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def upload(x, dtype=None):
    return t.from_torch(
        x.contiguous(), device=mesh, dtype=dtype or t.bfloat16, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
    )


def host(x):
    return t.to_torch(t.get_device_tensors(x)[0]).contiguous()


try:
    for block in BLOCKS:
        data = torch.load(ROOT / block / "attention.pt", weights_only=True, map_location="cpu")
        seq, heads = data["q"].shape[1], data["heads"]
        padded = (seq + 31) // 32 * 32
        sin, cos = data["rope"].chunk(2, -1)
        sin = torch.cat((sin, sin), -1).unsqueeze(1)
        cos = torch.cat((cos, cos), -1).unsqueeze(1)
        shape = lambda x: x.reshape(1, seq, heads, 128).permute(0, 2, 1, 3).contiguous()
        q = host_rope(shape(data["q"]), cos, sin)
        k = host_rope(shape(data["k"]), cos, sin)
        v = shape(data["v"])
        cuda = data["output"].reshape(1, seq, heads, 128).permute(0, 2, 1, 3).float()
        scale = 128**-0.5

        # Host references, per head.
        exact = torch.empty(1, heads, seq, 128, dtype=torch.float64)
        flash = torch.empty(1, heads, seq, 128)
        for h in range(heads):
            s = (q[0, h].double() @ k[0, h].double().T) * scale
            exact[0, h] = torch.softmax(s, -1) @ v[0, h].double()
            s32 = (q[0, h].float() @ k[0, h].float().T) * scale
            pr = torch.exp(s32 - s32.amax(-1, keepdim=True))
            flash[0, h] = (pr.bfloat16().float() @ v[0, h].float()) / pr.sum(-1, keepdim=True)
        refs = {"cuda": cuda, "fp64": exact}
        record(f"{block}/host/fp64_bf16out", exact.bfloat16(), {"cuda": cuda})
        record(f"{block}/host/flash_emul_bf16out", flash.bfloat16(), refs)

        pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, padded - seq))
        qt, kt, vt = upload(pad(q)), upload(pad(k)), upload(pad(v))
        mask = torch.full((1, 1, padded, padded), -10000.0, dtype=torch.bfloat16)
        mask[:, :, :, :seq] = 0
        mt = upload(mask)
        crop = lambda x: host(x)[:, :, :seq].float()

        for chunk in (128,):
            for approx in (False,):
                cfg = t.SDPAProgramConfig(
                    compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
                    q_chunk_size=chunk,
                    k_chunk_size=chunk,
                    exp_approx_mode=approx,
                )
                y = t.transformer.scaled_dot_product_attention(
                    qt,
                    kt,
                    vt,
                    attn_mask=mt,
                    is_causal=False,
                    scale=scale,
                    compute_kernel_config=compute,
                    memory_config=t.DRAM_MEMORY_CONFIG,
                    program_config=cfg,
                )
                record(f"{block}/tt/sdpa_q{chunk}_approx{approx}", crop(y), refs)

        # Manual per-head attention with controllable intermediate formats.
        mask_full = upload(mask.float(), t.float32)
        variants = {
            "scores_fp32_P_bf16": (t.float32, t.bfloat16),
            "scores_bf16_P_bf16": (t.bfloat16, t.bfloat16),
            "scores_fp32_P_fp32": (t.float32, t.float32),
        }
        for name, (sdt, pdt) in variants.items():
            outs = []
            for h in range(heads):
                qh, kh, vh = qt[:, h : h + 1], kt[:, h : h + 1], vt[:, h : h + 1]
                s = t.matmul(
                    qh,
                    t.transpose(kh, -2, -1),
                    dtype=sdt,
                    compute_kernel_config=compute,
                    memory_config=t.DRAM_MEMORY_CONFIG,
                )
                s = t.typecast(s, t.float32)
                s = t.multiply(s, scale)
                s = t.add(s, mask_full)
                s = t.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=compute)
                if pdt == t.bfloat16:
                    pm, vm = t.typecast(s, t.bfloat16), vh
                else:
                    pm, vm = s, t.typecast(vh, t.float32)
                o = t.matmul(pm, vm, dtype=t.float32, compute_kernel_config=compute, memory_config=t.DRAM_MEMORY_CONFIG)
                outs.append(host(t.typecast(o, t.bfloat16))[:, :, :seq].float())
                for z in (s, pm, o):
                    z.deallocate()
            record(f"{block}/tt/manual_{name}", torch.cat(outs, 1), refs)
        report["results"][f"{block}/meta"] = {"seq": seq, "padded": padded, "heads": heads}
        save()
    report["status"] = "complete"
    save()
finally:
    t.close_mesh_device(mesh)
