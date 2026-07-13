# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-path vs prefill-path logit fidelity (teacher-forced) — localize the decode drift.

prefill/score path (score_tp, chunk GDN + full attention) is healthy (PPL 9.03). Here we run the
INCREMENTAL decode path (prefill_seed the first P tokens, then feed the TRUE next tokens one by one
via decode_tp_batched) on the SAME sequence and compare its per-position next-token logits to
score_tp's. Both are TT, no external reference. If decode logits diverge from prefill logits as the
decode position grows -> the decode step math (attention paged/RoPE at growing pos; GDN recurrence
is already exonerated) drifts. If they stay aligned -> decode path is faithful and the full-model
"drift" is autoregressive model behavior, not a decode numerics bug.

Reports PCC + argmax-agreement of decode-vs-prefill logits binned by decode step. Run:
  QWEN_GDN_FUSED_DECODE=1 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_decode_vs_prefill.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@parametrize_mesh_tp()
def test_decode_vs_prefill(mesh_device):
    from loguru import logger
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    nl = os.environ.get("N_LAYERS")
    kw = {"n_layers": int(nl)} if nl else {}
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048, **kw)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    logger.info(f"DECVSPRE n_layers={len(model.layers)} full_attn_layers="
                f"{[i for i,l in enumerate(model.layers) if l.is_full_attention]}")

    text = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes four into "
            "muffins. She sells the remainder at the market for two dollars per egg. To find the "
            "answer we first compute how many eggs remain after breakfast and baking, then multiply "
            "the remainder by the price. Sixteen minus three is thirteen, and thirteen minus four is "
            "nine, so nine eggs remain to sell. Nine eggs at two dollars each gives eighteen dollars.")
    ids = tok(text, return_tensors="pt").input_ids[0]
    T = int(ids.numel())
    P = 48  # prefill the first P tokens, teacher-force decode the rest

    def pad128(n):
        return ((n + 127) // 128) * 128

    # ---- reference: prefill/score all-position logits ----
    Tp = pad128(T)
    padded = torch.zeros(1, Tp, dtype=torch.long)
    padded[0, :T] = ids
    ref = model.score_tp(padded, valid_len=T)  # [Tp, vocab]

    # ---- incremental decode path (teacher-forced) ----
    Pp = pad128(P)
    ppad = torch.zeros(1, Pp, dtype=torch.long)
    ppad[0, :P] = ids[:P]
    model.reset_tp()
    _ = model.prefill_seed_tp(ppad, valid_len=P, batch_slot=0)
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.finalize_seed(1)

    rows = []
    for pos in range(P, T - 1):
        true_tok = int(ids[pos].item())
        dl = model.decode_tp_batched([true_tok], [pos])  # [1, vocab] = pred for pos+1
        dl = torch.as_tensor(dl).float().reshape(-1)[: ref.shape[-1]]
        rl = ref[pos].float()
        agree = int(dl.argmax().item() == rl.argmax().item())
        rows.append((pos - P, _pcc(dl, rl), agree))

    # bin by decode-step ranges
    for lo in range(0, len(rows), 20):
        chunk = rows[lo:lo + 20]
        pcc = sum(c[1] for c in chunk) / len(chunk)
        agr = sum(c[2] for c in chunk) / len(chunk)
        logger.info(f"DECVSPRE steps[{lo}-{lo+len(chunk)-1}] meanPCC={pcc:.4f} argmax_agree={agr:.2f}")
    allpcc = sum(c[1] for c in rows) / len(rows)
    allagr = sum(c[2] for c in rows) / len(rows)
    logger.info(f"DECVSPRE_SUMMARY nsteps={len(rows)} meanPCC={allpcc:.4f} argmax_agree={allagr:.3f} "
                f"firstPCC={rows[0][1]:.4f} lastPCC={rows[-1][1]:.4f}")
