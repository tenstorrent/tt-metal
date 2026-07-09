# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-slot state reseed (continuous-batching join) check (Step 3).

Replaces ONE batch lane's sequence mid-generation (reseed_slot_tp) while the other
lanes keep decoding, and verifies:

  * the untouched lane (0) produces the EXACT same tokens whether or not lane 1 was
    reseeded — i.e. reseeding lane 1's KV + GDN state does not perturb lane 0;
  * the reseeded lane (1) follows its NEW prompt correctly (oracle first token 'Paris').

Lanes are independent (per-lane KV cache + per-value-head GDN recurrence), so this
proves the in-place per-lane reseed writes only the target lane. On-device only;
needs a 4-chip P150x4 mesh.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_batched_reseed_tp.py -v -s
"""
import math
import os

import torch

from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_batched_reseed_slot(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    B = 2
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=256)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    ids = lambda s: tok(s, return_tensors="pt").input_ids[0].tolist()
    A = ids("The largest ocean on Earth is")  # lane 0 (must stay unperturbed)
    Bp = ids("Photosynthesis converts sunlight into")  # lane 1 (original)
    C = ids("The capital of France is")  # lane 1 (reseeded → oracle 'Paris')

    def seed(prompts):
        model.reset_tp()
        cur, pos = [], []
        for b, p in enumerate(prompts):
            T = len(p)
            T_pad = max(128, math.ceil(T / 128) * 128)
            padded = list(p) + [0] * (T_pad - T)
            lg = model.prefill_seed_tp(torch.tensor([padded], dtype=torch.long), valid_len=T, batch_slot=b)
            cur.append(int(torch.argmax(lg).item()))
            pos.append(T)
        for layer in model.layers:
            if not layer.is_full_attention:
                layer.attention.finalize_seed(B)
        return cur, pos

    STEPS = 8
    RESEED_AT = 4

    # Reference: no reseed — record lane 0's tokens.
    cur, pos = seed([A, Bp])
    ref0 = [cur[0]]
    for _ in range(STEPS):
        logits = model.decode_tp_batched(cur, pos)
        nxt = [int(torch.argmax(logits[b]).item()) for b in range(B)]
        for b in range(B):
            pos[b] += 1
            cur[b] = nxt[b]
        ref0.append(nxt[0])

    # Reseed run: swap lane 1 → C at RESEED_AT, keep decoding.
    cur, pos = seed([A, Bp])
    res0 = [cur[0]]
    slot1_first = None
    for step in range(STEPS):
        if step == RESEED_AT:
            slot1_first, newpos = model.reseed_slot_tp(C, slot=1)
            cur[1], pos[1] = slot1_first, newpos
        logits = model.decode_tp_batched(cur, pos)
        nxt = [int(torch.argmax(logits[b]).item()) for b in range(B)]
        for b in range(B):
            pos[b] += 1
            cur[b] = nxt[b]
        res0.append(nxt[0])

    logger.info(f"lane0 ref   : {tok.decode(ref0)!r}")
    logger.info(f"lane0 reseed: {tok.decode(res0)!r}")
    first_c = tok.decode([slot1_first]).strip()
    logger.info(f"lane1 first token after reseed to C: {first_c!r}")

    # Lane 0 is untouched by lane 1's reseed → identical tokens.
    assert res0 == ref0, f"lane 0 perturbed by lane 1 reseed\n  ref={ref0}\n  got={res0}"
    # Reseeded lane 1 follows the new prompt C → oracle first token.
    assert first_c == "Paris", f"reseeded lane 1 expected 'Paris', got {first_c!r}"
    logger.info("PASSED: per-slot reseed — lane 0 unperturbed, lane 1 follows new prompt")
