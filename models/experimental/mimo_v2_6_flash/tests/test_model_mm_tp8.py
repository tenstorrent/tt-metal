# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full 48-layer MiMo-V2.6-Flash on one TP=8 replica (row 0 of the Galaxy) with multimodal prompts, checked against the
fp32 reference of reference/make_golden_mm.py (MIMO_MM_GOLDEN, default <golden dir>/<checkpoint>/golden_mm_48l.pt).

* The LLM: every golden prompt (a text-only control of similar length, an image + question, an audio clip + question)
  is prefilled from the golden's own ``inputs_embeds`` (the fp32 reference encoders' rows, in bf16) through the
  serving path's scatter (chunk_embeds) and merge (prefill_enqueue(embeds=)), all but the last TEACHER_FORCED tokens;
  those are teacher-forced through decode. Every logits row is scored against the reference (top-1, PCC) at the text
  golden's bar (test_model_tp8_dp4.py), and the prefill's per-layer hidden states too (all rows, and the item's rows).
  The text-only control shows what the same metrics look like without multimodal rows; with MIMO_GOLDEN (the text
  golden, make_golden.py) its prompts are scored the same way.
* The serving encoders: the items are encoded again by the serving backends (the device ViT on this row,
  tt/vision/encoder.py; the host audio path, tt/audio/host_backend.py) and compared with the reference rows; the image
  prompt is also run end to end from the device ViT's rows.
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
SLOTS = 4


def mm_golden_path():
    ckpt_name = os.path.basename(os.path.normpath(model_path()))
    default = os.path.join("/home/zni/models/mimo-golden", ckpt_name, "golden_mm_48l.pt")
    path = os.environ.get("MIMO_MM_GOLDEN", default)
    if not os.path.exists(path):
        pytest.skip(f"multimodal golden {path} not found; run reference/make_golden_mm.py")
    return path


def run_group(model, tok, prompts, greedy=0):
    """Prefill (with per-layer hidden) and teacher-force the prompts, one per slot; score each against its reference.
    prompt: name, ids, embeds / rows (or None), ref_hidden [49, S, hidden], ref_logits(position) -> [vocab]."""
    results, positions, last = {}, {}, {}
    for slot, p in enumerate(prompts):
        ids, n = p["ids"], p["ids"].numel() - TEACHER_FORCED
        t0 = time.time()
        logits, hidden = model.prefill(
            ids[:n], slot, return_hidden=True, embeds=p.get("embeds"), embed_rows=p.get("rows")
        )
        r = results[p["name"]] = dict(top1=[], pcc=[], margins=[], ms=1e3 * (time.time() - t0), n=n)
        ref_hidden = p["ref_hidden"]
        r["layer_pcc"] = [pcc(h, ref_hidden[k + 1][:n].float()) for k, h in enumerate(hidden)]
        rows = p.get("rows")
        if rows is not None and bool(rows.any()):
            r["mm_pcc"] = pcc(hidden[-1][rows], ref_hidden[-1][:n][rows].float())
            r["text_pcc"] = pcc(hidden[-1][~rows], ref_hidden[-1][:n][~rows].float())
            # Per layer: PCC at the item rows / at the text rows / at the text rows without the few rows that carry
            # massive activations (|x| > 500 in some layer: they dominate a PCC over the rows they are in), and the
            # reference's largest |activation| at the item / text rows.
            massive = ref_hidden[1:, :n].float().abs().amax(dim=(0, 2)) > 500
            plain = ~rows & ~massive
            r["massive_rows"] = massive.nonzero().flatten().tolist()
            r["curve"] = [
                (
                    pcc(h[rows], ref_hidden[k + 1][:n][rows].float()),
                    pcc(h[~rows], ref_hidden[k + 1][:n][~rows].float()),
                    pcc(h[plain], ref_hidden[k + 1][:n][plain].float()),
                    float(ref_hidden[k + 1][:n][rows].float().abs().max()),
                    float(ref_hidden[k + 1][:n][~rows].float().abs().max()),
                )
                for k, h in enumerate(hidden)
            ]
        _score(r, logits, p["ref_logits"](n - 1))
        positions[slot] = n
    for _ in range(TEACHER_FORCED):
        tokens = torch.zeros(model.max_batch, dtype=torch.int32)
        pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
        for slot, p in enumerate(prompts):
            tokens[slot], pos[slot] = int(p["ids"][positions[slot]]), positions[slot]
        out = model.decode(tokens, pos)
        for slot, p in enumerate(prompts):
            _score(results[p["name"]], out[slot], p["ref_logits"](positions[slot]))
            last[slot] = out[slot]
            positions[slot] += 1
    if greedy:
        gen = {slot: [int(last[slot].argmax())] for slot in last}
        for _ in range(greedy - 1):
            tokens = torch.zeros(model.max_batch, dtype=torch.int32)
            pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
            for slot in gen:
                tokens[slot], pos[slot] = gen[slot][-1], positions[slot]
            out = model.decode(tokens, pos)
            for slot in gen:
                gen[slot].append(int(out[slot].argmax()))
                positions[slot] += 1
        for slot, p in enumerate(prompts):
            results[p["name"]]["answer"] = tok.decode(gen[slot])
    return results


def _score(r, lg, ref):
    lg, ref = lg.float(), ref.float()
    r["top1"].append(int(lg.argmax()) == int(ref.argmax()))
    r["pcc"].append(pcc(lg, ref))
    top2 = ref.topk(2).values
    r["margins"].append(float(top2[0] - top2[1]))


def log_results(results):
    for name, r in results.items():
        lp = r["layer_pcc"]
        extra = ""
        if "mm_pcc" in r:
            extra = f"; layer 47 at the item rows {r['mm_pcc']:.5f}, at the text rows {r['text_pcc']:.5f}"
        logger.info(
            f"{name}: {r['n']} prefill tokens in {r['ms']:.0f} ms; hidden PCC layer 0 {lp[0]:.5f}, layer 23 "
            f"{lp[23]:.5f}, layer 47 {lp[-1]:.5f}{extra}; logits top-1 {sum(r['top1'])}/{len(r['top1'])}, PCC "
            f"mean {sum(r['pcc']) / len(r['pcc']):.4f} min {min(r['pcc']):.4f}; reference margins "
            f"{[round(m, 2) for m in r['margins']]}" + (f"; answer {r['answer']!r}" if "answer" in r else "")
        )
        if "curve" in r:
            logger.info(
                f"{name}: per layer, hidden PCC item rows / text rows / text rows without the massive-activation rows "
                f"{r['massive_rows']} (reference max |activation| item / text): "
                + ", ".join(
                    f"{k}: {a:.4f}/{b:.4f}/{c:.4f} ({d:.0f}/{e:.0f})" for k, (a, b, c, d, e) in enumerate(r["curve"])
                )
            )


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_model_mm_tp8(mesh_device):
    golden = torch.load(mm_golden_path(), weights_only=False)
    assert golden["num_layers"] == 48 and golden["tail"] > TEACHER_FORCED
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path(), trust_remote_code=True)
    row = dp_submeshes(mesh_device)[0]
    t0 = time.time()
    model = TtMiMoModel(row, cfg, ckpt=ckpt, cache_dir=cache_dir("model_bfp4experts"), max_seq_len=1024, decode_users=4)
    model.setup_kv_cache(SLOTS)
    logger.info(f"model ready in {time.time() - t0:.0f}s")

    # ---- the serving encoders against the reference rows
    from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import HostAudioBackend
    from models.experimental.mimo_v2_6_flash.tt.vision.encoder import MiMoVisionEncoder

    serving = mm.CompositeBackend(MiMoVisionEncoder(row, model_path()), HostAudioBackend(model_path()))
    serving_rows = {}
    for p in golden["prompts"]:
        if p["modality"] is None:
            continue
        item = mm.MMItem(p["modality"], p["name"], p["offset"], p["length"], None, p["fields"])
        t1 = time.time()
        rows = mm.encode_item(serving, item)
        ref_rows = p["inputs_embeds"][p["mm_rows"]]
        enc_pcc = pcc(rows.float(), ref_rows.float())
        logger.info(
            f"{p['name']}: serving encoder ({type(getattr(serving, 'vision' if p['modality'] != 'audio' else 'audio')).__name__}) "
            f"{rows.shape[0]} rows in {1e3 * (time.time() - t1):.0f} ms, PCC vs the fp32 reference {enc_pcc:.5f}"
        )
        assert enc_pcc > 0.99, enc_pcc
        serving_rows[p["name"]] = (item, rows)

    # ---- the LLM from the reference rows (and the image end to end from the device ViT's rows)
    def prompt(p, rows_from=None):
        n = p["token_ids"].numel() - TEACHER_FORCED
        tail0 = p["token_ids"].numel() - golden["tail"]
        out = dict(
            name=p["name"] if rows_from is None else f"{p['name']} (device ViT rows)",
            ids=p["token_ids"],
            ref_hidden=p["hidden"],
            ref_logits=lambda q, p=p, tail0=tail0: p["logits_tail"][q - tail0],
        )
        if p["modality"] is not None:
            item = mm.MMItem(p["modality"], p["name"], p["offset"], p["length"], None, p["fields"])
            source = (lambda it: rows_from) if rows_from is not None else (lambda it: p["inputs_embeds"][p["mm_rows"]])
            out["embeds"], out["rows"] = mm.chunk_embeds([item], 0, n, source, cfg.hidden_size)
        return out

    group = [prompt(p) for p in golden["prompts"]]
    image = next(p for p in golden["prompts"] if p["modality"] == "image")
    group.append(prompt(image, rows_from=serving_rows["image"][1]))
    results = run_group(model, tok, group[:SLOTS], greedy=GREEDY_TOKENS)
    log_results(results)

    # ---- the text golden's prompts, the same metrics
    text_golden = os.environ.get("MIMO_GOLDEN")
    if text_golden and os.path.exists(text_golden):
        tg = torch.load(text_golden, weights_only=False)
        tps = [
            dict(name=f"text golden {g}", ids=ids, ref_hidden=h, ref_logits=lambda q, lg=lg: lg[q])
            for g, (ids, h, lg) in enumerate(zip(tg["token_ids"], tg["hidden"], tg["logits"]))
            if ids.numel() > TEACHER_FORCED
        ]
        text_results = {}
        for k in range(0, len(tps), SLOTS):
            text_results.update(run_group(model, tok, tps[k : k + SLOTS]))
        log_results(text_results)
        l47 = [r["layer_pcc"][-1] for r in text_results.values()]
        top1 = [t for r in text_results.values() for t in r["top1"]]
        logger.info(
            f"text golden: layer 47 hidden PCC mean {sum(l47) / len(l47):.5f} (min {min(l47):.5f}); logits top-1 "
            f"{sum(top1)}/{len(top1)}"
        )

    top1 = [t for r in results.values() for t in r["top1"]]
    pccs = [x for r in results.values() for x in r["pcc"]]
    logger.info(f"multimodal golden: top-1 {sum(top1)}/{len(top1)}, logits PCC mean {sum(pccs) / len(pccs):.4f}")
    assert sum(top1) / len(top1) >= 0.8, top1
    assert sum(pccs) / len(pccs) > 0.95, pccs
