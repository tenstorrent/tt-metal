# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Localize the decode divergence WITHIN attention: q-post-rope vs SDPA output, decode vs prefill.

QWEN_ATTN_CAPTURE=1 makes each full-attn layer record q (post-RoPE) and attn (SDPA out) in both
paths. This runs score_tp (prefill, all positions) then one teacher-forced decode step at position P,
and compares per attention layer, at position P:
  - PCC(q_rope decode, q_rope prefill)   -> RoPE/projection fidelity
  - PCC(attn_out decode, attn_out prefill) -> SDPA/KV-cache fidelity
If q_rope matches but attn_out diverges -> the SDPA-decode kernel / KV read is the culprit; if
q_rope also diverges -> RoPE decode.
Run: QWEN_ATTN_CAPTURE=1 N_LAYERS=8 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_attn_capture.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.attention import tp as attn_tp


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@parametrize_mesh_tp()
def test_attn_capture(mesh_device):
    from loguru import logger
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    os.environ["QWEN_ATTN_CAPTURE"] = "1"
    mesh_device.enable_program_cache()
    nl = int(os.environ.get("N_LAYERS", "8"))
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048, n_layers=nl)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    fa = [i for i, l in enumerate(model.layers) if l.is_full_attention]
    logger.info(f"ATTNCAP n_layers={nl} full_attn_layers={fa}")

    text = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes four into "
            "muffins, then sells the rest at the market for two dollars each. Compute the daily revenue "
            "step by step, tracking every number carefully to reach the final dollar amount.")
    ids = tok(text, return_tensors="pt").input_ids[0]
    T = int(ids.numel())
    P = 48

    def pad128(n):
        return ((n + 127) // 128) * 128

    # ---- prefill capture (all positions) ----
    attn_tp._ATTN_CAP.clear()
    Tp = pad128(T)
    padded = torch.zeros(1, Tp, dtype=torch.long)
    padded[0, :T] = ids
    _ = model.score_tp(padded, valid_len=T)
    pf = list(attn_tp._ATTN_CAP)  # [(prefill, q_rope/attn_out, [4,NHl,S,HD]) ...] per attn layer

    # ---- decode capture: prefill first P, then ONE decode step at position P ----
    attn_tp._ATTN_CAP.clear()
    Pp = pad128(P)
    ppad = torch.zeros(1, Pp, dtype=torch.long)
    ppad[0, :P] = ids[:P]
    model.reset_tp()
    _ = model.prefill_seed_tp(ppad, valid_len=P, batch_slot=0)
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.finalize_seed(1)
    attn_tp._ATTN_CAP.clear()  # drop the prefill_seed captures; keep only the decode step
    _ = model.decode_tp_batched([int(ids[P].item())], [P])
    dec = list(attn_tp._ATTN_CAP)

    # group by (name) in order → one entry per attn layer per name
    def by_name(cap, name):
        return [t for (m, n, t) in cap if n == name]

    for li, lidx in enumerate(fa):
        for name in ("q_rope", "attn_out"):
            pt = by_name(pf, name)[li]   # [4, NHl, S, HD]
            dt = by_name(dec, name)[li]  # [4, 1, NHl, HD] (decode: [.,B,NH,HD])
            pf_pos = pt[:, :, P, :] if pt.dim() == 4 and pt.shape[2] >= P + 1 else pt.reshape(4, -1, pt.shape[-1])[:, min(P, pt.shape[1] - 1), :]
            dt_pos = dt[:, 0, :, :] if dt.dim() == 4 else dt.reshape(4, -1, dt.shape[-1])[:, 0, :]
            logger.info(f"ATTNCAP layer{lidx} {name}: PCC(decode,prefill)={_pcc(dt_pos, pf_pos):.5f}")
