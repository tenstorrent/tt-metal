# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate the token-major attention: device sdpa with the block-diagonal mask vs torch, per row
count (48 = not a tile multiple, 96, 192) and per sdpa program config. Finds which part of the
token-major solve is inexact.

Env: VOXTRAL_DEVICE_ID (0).
"""

import os

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    FM_HEAD_DIM,
    FM_N_HEADS,
    FM_N_KV_HEADS,
    pcc,
)  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as fm  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = pytest.mark.slow  # opens a device

DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))


def _torch_attn(q, k, v, mask):
    rep = FM_N_HEADS // FM_N_KV_HEADS
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    s = q.float() @ k.float().transpose(-1, -2) + mask.float()
    return torch.softmax(s, dim=-1) @ v.float()


def test_tm_attention_exactness():
    dev = open_device(device_id=DEVICE_ID)
    try:
        cc = fm.COMPUTE_CONFIG
        for B2 in (16, 32, 64):
            rows = 3 * B2
            torch.manual_seed(0)
            q = torch.randn(1, FM_N_HEADS, rows, FM_HEAD_DIM) * 0.3
            k = torch.randn(1, FM_N_KV_HEADS, rows, FM_HEAD_DIM) * 0.3
            v = torch.randn(1, FM_N_KV_HEADS, rows, FM_HEAD_DIM)
            r = torch.arange(rows)
            same = (r.reshape(-1, 1) % B2) == (r.reshape(1, -1) % B2)
            mask = torch.where(same, 0.0, -1e9).reshape(1, 1, rows, rows)
            ref = _torch_attn(q.bfloat16(), k.bfloat16(), v.bfloat16(), mask)
            dv = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            qd, kd, vd, md = dv(q), dv(k), dv(v), dv(mask)
            variants = {"default": None}
            for qc, kc in ((32, 32), (32, 64)):
                try:
                    variants[f"q{qc}_k{kc}"] = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                        q_chunk_size=qc,
                        k_chunk_size=kc,
                        exp_approx_mode=False,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[tm-attn] cannot build config q{qc}_k{kc}: {e}")
            for name, prg in variants.items():
                try:
                    kw = {"program_config": prg} if prg is not None else {}
                    out = ttnn.transformer.scaled_dot_product_attention(
                        qd, kd, vd, attn_mask=md, is_causal=False, scale=1.0, compute_kernel_config=cc, **kw
                    )
                    o = ttnn.to_torch(out).float()
                    p = pcc(o, ref)
                    worst_row = min(pcc(o[0, :, i], ref[0, :, i]) for i in range(rows))
                    print(
                        f"[tm-attn] rows {rows:3d} ({'tile-aligned' if rows % 32 == 0 else 'NOT tile-aligned'}) sdpa {name:9s}: PCC {p:.6f}, worst row PCC {worst_row:.6f}, max|diff| {(o - ref).abs().max():.3e}"
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[tm-attn] rows {rows:3d} sdpa {name:9s}: FAILED {type(e).__name__}: {str(e)[:120]}")
            # the fused head split/concat round trip on its own
            qkv = torch.cat(
                [
                    q.permute(0, 2, 1, 3).reshape(1, rows, -1),
                    k.permute(0, 2, 1, 3).reshape(1, rows, -1),
                    v.permute(0, 2, 1, 3).reshape(1, rows, -1),
                ],
                dim=-1,
            )
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
                dv(qkv.reshape(1, 1, rows, -1)),
                num_heads=FM_N_HEADS,
                num_kv_heads=FM_N_KV_HEADS,
                transpose_k_heads=False,
                memory_config=fm._L1,
            )
            print(
                f"[tm-attn] rows {rows:3d} nlp_create_qkv_heads: q PCC {pcc(ttnn.to_torch(qh).float(), q.bfloat16().float()):.6f}, k {pcc(ttnn.to_torch(kh).float(), k.bfloat16().float()):.6f}, v {pcc(ttnn.to_torch(vh).float(), v.bfloat16().float()):.6f}"
            )
            cat = (
                ttnn.to_torch(ttnn.experimental.nlp_concat_heads(qh, memory_config=fm._L1)).float().reshape(1, rows, -1)
            )
            print(
                f"[tm-attn] rows {rows:3d} nlp_concat_heads round trip PCC {pcc(cat, q.bfloat16().float().permute(0, 2, 1, 3).reshape(1, rows, -1)):.6f}"
            )
    finally:
        ttnn.close_device(dev)
