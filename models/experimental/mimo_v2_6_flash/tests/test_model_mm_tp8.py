# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full 48-layer MiMo-V2.6-Flash on one TP=8 replica (row 0 of the Galaxy) with multimodal prompts, checked against the
fp32 reference of reference/make_golden_mm.py (MIMO_MM_GOLDEN, default <golden dir>/<checkpoint>/golden_mm_48l.pt).

Each golden prompt (an image + question, an audio clip + question) goes through the serving path: its item is
encoded again from the saved processor fields by the encoder backend (tt/multimodal.py; the rows must match the
golden's), scattered into the prompt's rows (chunk_embeds) and merged into the prefill (prefill_enqueue(embeds=)).
All but the last TEACHER_FORCED tokens are prefilled, those are teacher-forced through decode, and every logits row
is compared with the reference (top-1, PCC) at the text golden's bar (test_model_tp8_dp4.py). The per-layer hidden
states of the prefill are compared too, and each prompt then continues greedily (the answer is logged).
"""

import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.tests.common import (
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    dp_submeshes,
    load_config_and_ckpt,
    model_path,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt import multimodal as mm
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel

TEACHER_FORCED = 4
GREEDY_TOKENS = 24


def mm_golden_path():
    ckpt_name = os.path.basename(os.path.normpath(model_path()))
    default = os.path.join("/home/zni/models/mimo-golden", ckpt_name, "golden_mm_48l.pt")
    path = os.environ.get("MIMO_MM_GOLDEN", default)
    if not os.path.exists(path):
        pytest.skip(f"multimodal golden {path} not found; run reference/make_golden_mm.py")
    return path


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_model_mm_tp8(mesh_device):
    golden = torch.load(mm_golden_path(), weights_only=False)
    assert golden["num_layers"] == 48 and golden["tail"] > TEACHER_FORCED
    prompts = golden["prompts"]
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path(), trust_remote_code=True)

    # ---- the serving path's host side: encode each item again, scatter it into the prefill rows
    backend = mm.HFHostBackend(model_path(), dtype=torch.float32)
    encoder = mm.ItemEncoder(backend, mm.EmbeddingCache())
    inputs = []
    for p in prompts:
        item = mm.MMItem(p["modality"], p["name"], p["offset"], p["length"], None, p["fields"])
        n = p["token_ids"].numel() - TEACHER_FORCED
        embeds, rows = mm.chunk_embeds([item], 0, n, encoder, cfg.hidden_size)
        assert torch.equal(rows, p["mm_rows"][:n]), f"{p['name']}: multimodal rows differ from the golden's"
        enc_pcc = pcc(embeds[rows].float(), p["inputs_embeds"][:n][rows].float())
        logger.info(
            f"{p['name']}: {int(rows.sum())} {p['modality']} rows re-encoded, PCC vs the golden's {enc_pcc:.6f}"
        )
        assert enc_pcc > 0.999, enc_pcc
        inputs.append((n, embeds, rows))

    row = dp_submeshes(mesh_device)[0]
    t0 = time.time()
    model = TtMiMoModel(row, cfg, ckpt=ckpt, cache_dir=cache_dir("model_bfp4experts"), max_seq_len=1024, decode_users=4)
    model.setup_kv_cache(4)
    logger.info(f"model ready in {time.time() - t0:.0f}s")

    top1, pccs, rows_log = [], [], []

    def score(lg, ref, name, step):
        lg, ref = lg.float(), ref.float()
        top1.append(int(lg.argmax()) == int(ref.argmax()))
        pccs.append(pcc(lg, ref))
        r2 = ref.topk(2)
        rows_log.append((name, step, top1[-1], float(r2.values[0] - r2.values[1]), pccs[-1]))

    # ---- prefill all but the last TEACHER_FORCED tokens, with the per-layer hidden states
    first, positions = {}, {}
    for slot, (p, (n, embeds, rows)) in enumerate(zip(prompts, inputs)):
        ids = p["token_ids"]
        t1 = time.time()
        logits, hidden = model.prefill(ids[:n], slot, return_hidden=True, embeds=embeds, embed_rows=rows)
        logger.info(
            f"{p['name']}: prefill of {n} tokens ({int(rows.sum())} embedded) {1e3 * (time.time() - t1):.0f} ms"
        )
        tail0 = ids.numel() - golden["tail"]
        score(logits, p["logits_tail"][n - 1 - tail0], p["name"], 0)
        first[slot], positions[slot] = logits, n
        ref_hidden = p["hidden"]  # [49, S, hidden]: the embedding, then every layer's output
        layer_pcc = [pcc(h, ref_hidden[k + 1][:n].float()) for k, h in enumerate(hidden)]
        mm_pcc = pcc(hidden[-1][rows], ref_hidden[-1][:n][rows].float())
        logger.info(
            f"{p['name']}: per-layer hidden PCC vs reference: layer 0 {layer_pcc[0]:.5f}, layer 23 {layer_pcc[23]:.5f}, "
            f"layer 47 {layer_pcc[-1]:.5f} (min {min(layer_pcc):.5f}); layer 47 at the {p['modality']} rows {mm_pcc:.5f}"
        )

    # ---- teacher-forced decode over the last TEACHER_FORCED prompt tokens
    max_batch = model.max_batch
    last = {}
    for t in range(TEACHER_FORCED):
        tokens = torch.zeros(max_batch, dtype=torch.int32)
        pos = torch.full((max_batch,), -1, dtype=torch.int32)
        for slot, p in enumerate(prompts):
            tokens[slot], pos[slot] = int(p["token_ids"][positions[slot]]), positions[slot]
        out = model.decode(tokens, pos)
        for slot, p in enumerate(prompts):
            tail0 = p["token_ids"].numel() - golden["tail"]
            score(out[slot], p["logits_tail"][positions[slot] - tail0], p["name"], t + 1)
            last[slot] = out[slot]
            positions[slot] += 1

    top1_rate = sum(top1) / len(top1)
    logger.info(
        f"vs fp32 reference over {len(top1)} logits rows: top-1 {top1_rate:.3f}, PCC min {min(pccs):.4f} mean "
        f"{sum(pccs) / len(pccs):.4f}"
    )
    for name, step, match, margin, p_ in rows_log:
        logger.info(f"{name} step {step}: {'ok  ' if match else 'MISS'} reference margin {margin:.3f}, PCC {p_:.4f}")
    for slot, p in enumerate(prompts):
        ref_next = tok.decode([p["next_token"]])
        logger.info(
            f"{p['name']}: reference next token {ref_next!r}, device {tok.decode([int(last[slot].argmax())])!r}"
        )

    # ---- greedy continuation: the answers
    gen = {slot: [int(last[slot].argmax())] for slot in last}
    for _ in range(GREEDY_TOKENS - 1):
        tokens = torch.zeros(max_batch, dtype=torch.int32)
        pos = torch.full((max_batch,), -1, dtype=torch.int32)
        for slot in gen:
            tokens[slot], pos[slot] = gen[slot][-1], positions[slot]
        out = model.decode(tokens, pos)
        for slot in gen:
            gen[slot].append(int(out[slot].argmax()))
            positions[slot] += 1
    for slot, p in enumerate(prompts):
        logger.info(f"{p['name']}: answer {tok.decode(gen[slot])!r}")

    assert top1_rate >= 0.8, top1_rate
    assert sum(pccs) / len(pccs) > 0.95, pccs
