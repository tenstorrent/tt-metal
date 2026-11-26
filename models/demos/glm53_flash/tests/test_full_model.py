# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full GLM-5.3-Flash model (45 layers + embedding + head) on ONE chip by weight streaming
(``tt/model.py``), teacher-forced against the HF reference, real weights.

1. ``test_dump_chain``: the 128-token random prompt of ``bringup/references/glm53_flash/L45_S128``
   (layer by layer through the HF stack) goes through the TT chain; every layer's TT output is
   compared with the dump's ``x_out`` of that layer (error accumulation across the stack), the
   logits of all 128 rows with the host head (``lm_head(norm(mean(streams)))`` in fp32) on the
   dump's layer-44 output (PCC, top-1 agreement, top-5 overlap), then the dump's 2 decode steps
   (its tokens at positions 128, 129) run on the TT state with the same per-layer comparison.
2. ``test_generate_vs_hf``: greedy generation of 16 tokens from a short real prompt (tokenizer from
   the snapshot) vs the HF greedy tokens of ``reference/hf_layer_reference.py --generate 16``
   (host, layers streamed per token; ``bringup/references/glm53_flash/FULLMODEL_G16``).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_full_model.py -s --timeout 7000
    python models/demos/glm53_flash/reference/hf_layer_reference.py --generate 16 \\
        --prompt "The capital of France is Paris. The theory of general relativity" \\
        --out bringup/references/glm53_flash/FULLMODEL_G16            # the HF side (CPU, ~1.5 h)

Measured 2026-10-20 (``doc/full_model/README.md``): (1) the chain PCC vs the dump decays 0.9998 (layer 0)
-> 0.967 (19) -> 0.857 (23) -> 0.69 (29-33) -> 0.858 (44), logits top-1 0.234 (head alone 0.99997 / 0.969),
so the > 0.99 / >= 0.9 gates of this test are red on the random prompt; the HF bf16 chain itself with a 1 %
per-layer perturbation decays the same way (``reference/hf_chain_sensitivity.py``), i.e. the model's own
sensitivity on random tokens, not a stack defect. (2) greedy generation: TT and HF agree on the first 7
tokens and diverge on a 0.25-logit near tie (HF's token = TT's #2); the gate is divergence-aware.

Env: ``GLM53_FULLMODEL_CACHE`` = tensorbin cache root (default ``~/ashai/glm53/_cache/fullmodel``,
``none`` disables; the first pass writes ~7.7 GB per MoE layer), ``GLM53_FULLMODEL_LAYERS`` = a
layer prefix such as ``0-4`` for mechanics checks (the head gates are skipped when the stack is
partial), ``GLM53_FULLMODEL_N`` = tokens to generate, ``GLM53_FULLMODEL_PROMPT``,
``GLM53_FULLMODEL_SAVE`` = a directory that receives the TT chain outputs of ``test_dump_chain``
(``tt_layer<i>_prefill.pt`` / ``tt_layer<i>_decode<s>.pt`` with ``x_out`` in the dump's layout, and
the logits) for host-side analysis (``reference/hf_chain_sensitivity.py --tt-chain``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import (
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_tensors,
    shard_headers,
)
from models.demos.glm53_flash.reference.hf_layer_reference import parse_layers
from models.demos.glm53_flash.tests.test_embedding_head import hf_head_reference
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.functional_decoder import device_streams_to_host, host_streams_to_device
from models.demos.glm53_flash.tt.model import GLM53Model, dram_view

DUMP = Path("bringup/references/glm53_flash/L45_S128")
HF_GEN = Path("bringup/references/glm53_flash/FULLMODEL_G16")
PROMPT = os.environ.get("GLM53_FULLMODEL_PROMPT", "The capital of France is Paris. The theory of general relativity")
N_TOKENS = int(os.environ.get("GLM53_FULLMODEL_N", "16"))
_cache_env = os.environ.get("GLM53_FULLMODEL_CACHE", "/home/ttuser/ashai/glm53/_cache/fullmodel")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
LAYERS = parse_layers(os.environ["GLM53_FULLMODEL_LAYERS"]) if os.environ.get("GLM53_FULLMODEL_LAYERS") else None
SAVE_DIR = Path(os.environ["GLM53_FULLMODEL_SAVE"]) if os.environ.get("GLM53_FULLMODEL_SAVE") else None
NUM_DECODE = 2
FINAL_PCC_BAR = 0.99  # the task's bars for the random-prompt chain; measured 0.858 / 0.234 (see the docstring)
TOP1_BAR = 0.9
PREFIX_BAR = int(os.environ.get("GLM53_FULLMODEL_PREFIX_BAR", "4"))


def _row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token PCC over the [4, 4096] stream vector: a, b [1, S, 4, 4096] -> [S]."""
    a = a.detach().float().reshape(a.shape[1], -1)
    b = b.detach().float().reshape(b.shape[1], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)


def _grade_logits(name: str, ref: torch.Tensor, got: torch.Tensor) -> dict:
    """PCC, top-1 agreement and top-5 overlap over the rows of ``[rows, vocab]`` logits."""
    p = pcc(ref, got)
    top1 = float((ref.argmax(-1) == got.argmax(-1)).float().mean())
    r5, g5 = ref.topk(5, dim=-1).indices, got.topk(5, dim=-1).indices
    top5 = float((r5.unsqueeze(-1) == g5.unsqueeze(-2)).any(-1).float().sum(-1).mean() / 5)
    logger.info(
        f"{name}: pcc {p:.6f}, top-1 agreement {top1:.4f}, top-5 overlap {top5:.4f}, "
        f"max abs diff {(ref - got).abs().max().item():.3f} (logit range {ref.abs().max().item():.2f})"
    )
    return {"pcc": p, "top1": top1, "top5": top5}


def _log_dram(device, when: str):
    a, f = dram_view(device)
    logger.info(f"DRAM {when}: {a:.3f} GiB allocated, {f:.2f} GiB free")


def _model(mesh_device, max_seq_len: int) -> GLM53Model:
    snapshot = default_snapshot_dir()
    t0 = time.perf_counter()
    model = GLM53Model.from_snapshot(
        mesh_device,
        snapshot,
        headers=shard_headers(snapshot),
        layers=LAYERS,
        max_batch=1,
        max_seq_len=max_seq_len,
        weight_cache_root=CACHE_ROOT,
    )
    logger.info(
        f"model: layers {model.layers[0]}..{model.layers[-1]} ({len(model.layers)}), cache root {CACHE_ROOT}, "
        f"embedding + head built in {model.embed_head_build_s:.1f} s (total {time.perf_counter() - t0:.1f} s)"
    )
    _log_dram(mesh_device, "embedding + head resident")
    return model


def _report(model: GLM53Model):
    logger.info("per-layer timing (seconds, means over the passes):\n" + model.timing_table())
    builds = [s for t in model.timing.values() for s in t.build_s]
    hits = sum(sum(t.cache_hits) for t in model.timing.values())
    logger.info(
        f"passes {model.passes}; layer builds {len(builds)} ({hits} expert-cache hits), total build {sum(builds):.0f} s, "
        f"total run {sum(s for t in model.timing.values() for _, s in t.run_s):.1f} s, "
        f"total release {sum(s for t in model.timing.values() for s in t.release_s):.1f} s; "
        f"DRAM peak after a build {max(max(t.dram_built_gib) for t in model.timing.values()):.2f} GiB; "
        f"tensorbin cache {model.cache_size_bytes() / 1e9:.1f} GB under {model.weight_cache_root}"
    )


@pytest.mark.timeout(7000)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dump_chain(mesh_device):
    full = LAYERS is None
    pre_in = torch.load(DUMP / "prefill_input.pt")
    prompt_ids = pre_in["prompt_ids"]  # [1, 128] int64
    S = prompt_ids.shape[1]
    model = _model(mesh_device, max_seq_len=S + 64)
    layers = model.layers
    x_outs = {i: torch.load(DUMP / f"layer{i}_prefill.pt")["x_out"] for i in layers}  # [1, S, 4, 4096] bf16
    results: dict[int, dict] = {i: {} for i in layers}

    # ---- (1) prefill: every layer vs the dump chain
    if SAVE_DIR:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def on_prefill(i, y):
        y_h = device_streams_to_host(y, 4, S)
        if SAVE_DIR:
            torch.save({"x_out": y_h.to(torch.bfloat16)}, SAVE_DIR / f"tt_layer{i}_prefill.pt")
        rows = _row_pcc(x_outs[i], y_h)
        results[i]["prefill"] = pcc(x_outs[i], y_h)
        results[i]["prefill_worst_row"] = float(rows.min())
        results[i]["prefill_worst_idx"] = int(rows.argmin())
        logger.info(
            f"layer {i:2d} ({model.timing[i].kind}) prefill S={S}: pcc vs dump x_out {results[i]['prefill']:.6f}, "
            f"worst row {int(rows.argmin())} {float(rows.min()):.5f}; build {model.timing[i].build_s[-1]:.1f} s "
            f"(expert cache {'hit' if model.timing[i].cache_hits[-1] else 'miss'}), run {model.timing[i].run_s[-1][1]:.2f} s"
        )

    # the embedding must reproduce the dump's layer-0 input exactly
    t0 = time.perf_counter()
    logits_dev = model.prefill(prompt_ids, all_rows=True, on_layer=on_prefill)
    t_prefill = time.perf_counter() - t0
    logits = model.host_logits(logits_dev)[:S]  # [S, vocab]
    ttnn.deallocate(logits_dev)
    if SAVE_DIR:
        torch.save({"logits": logits.to(torch.bfloat16)}, SAVE_DIR / "tt_prefill_logits.pt")
    _log_dram(mesh_device, "after the prefill pass (weights released)")
    logger.info(f"prefill pass over {len(layers)} layers: {t_prefill:.0f} s wall")

    last = layers[-1]
    head_w = load_tensors(model.snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], model.headers)
    ref = hf_head_reference(x_outs[last], head_w[FINAL_NORM_KEY], head_w[LM_HEAD_KEY], model.cfg, torch.float32)[0]
    g = _grade_logits(
        f"logits of the {S} rows: TT chain + TT head vs host head on the dump layer-{last} x_out", ref, logits
    )
    # how much of the gap is the head itself: TT head on the dump's own layer-44 streams
    head_only = model.host_logits(model.head(host_streams_to_device(x_outs[last], mesh_device)))[:S]
    g_head = _grade_logits(f"head only: TT head on the dump layer-{last} x_out vs host head", ref, head_only)
    logger.info(
        f"prefill greedy ids (TT chain) first 16 rows: {logits.argmax(-1)[:16].tolist()}; host: {ref.argmax(-1)[:16].tolist()}"
    )

    # ---- (2) the dump's 2 decode steps on the TT state
    dec_results = []
    for step in range(NUM_DECODE):
        recs = {i: torch.load(DUMP / f"layer{i}_decode{step}.pt") for i in layers}
        tok = int(recs[layers[0]]["token"])
        pos = S + step
        assert recs[layers[0]]["position"] == pos

        def on_decode(i, y, recs=recs, step=step):
            y_h = device_streams_to_host(y, 4, 1)
            if SAVE_DIR:
                torch.save({"x_out": y_h.to(torch.bfloat16)}, SAVE_DIR / f"tt_layer{i}_decode{step}.pt")
            results[i][f"decode{step}"] = pcc(recs[i]["x_out"], y_h)
            logger.info(
                f"layer {i:2d} decode step {step} (pos {pos}): pcc vs dump x_out {results[i][f'decode{step}']:.6f}; "
                f"build {model.timing[i].build_s[-1]:.1f} s, run {model.timing[i].run_s[-1][1]:.3f} s"
            )

        t0 = time.perf_counter()
        logits_d, ids_d = model.decode([tok], [pos], on_layer=on_decode)
        t_dec = time.perf_counter() - t0
        ld = model.host_logits(logits_d)[:1]
        dev_id = int(ttnn.to_torch(ids_d).view(torch.int32).reshape(-1)[0])
        ttnn.deallocate(logits_d)
        ttnn.deallocate(ids_d)
        ref_d = hf_head_reference(
            recs[last]["x_out"], head_w[FINAL_NORM_KEY], head_w[LM_HEAD_KEY], model.cfg, torch.float32
        )[0]
        gd = _grade_logits(
            f"decode step {step} (pos {pos}, token {tok}) logits: TT vs host head on the dump", ref_d, ld
        )
        gd["device_greedy"] = dev_id
        gd["host_argmax"] = int(ref_d.argmax())
        gd["tt_argmax"] = int(ld.argmax())
        gd["final_pcc"] = results[last][f"decode{step}"]
        dec_results.append(gd)
        logger.info(
            f"decode step {step}: pass {t_dec:.0f} s; TT argmax {gd['tt_argmax']} (device greedy {dev_id}), "
            f"host argmax {gd['host_argmax']}; TT top-5 {ld.topk(5).indices.tolist()}, host top-5 {ref_d.topk(5).indices.tolist()}"
        )

    # ---- tables
    lines = ["layer kind       prefill_pcc worst_row(idx)  decode0_pcc decode1_pcc"]
    for i in layers:
        r = results[i]
        lines.append(
            f"{i:5d} {model.timing[i].kind:10s} {r['prefill']:.6f}   {r['prefill_worst_row']:.5f} ({r['prefill_worst_idx']:3d})   "
            f"{r.get('decode0', float('nan')):.6f}    {r.get('decode1', float('nan')):.6f}"
        )
    logger.info("TT chain vs HF dump per layer:\n" + "\n".join(lines))
    _report(model)
    summary = {
        "layers": layers,
        "per_layer": results,
        "logits": g,
        "head_only": g_head,
        "decode": dec_results,
        "prefill_pass_s": t_prefill,
    }
    logger.info(f"FULLMODEL_RESULT {json.dumps(summary)}")
    if os.environ.get("GLM53_RESULTS"):
        with open(os.environ["GLM53_RESULTS"], "a") as f:
            f.write(json.dumps(summary) + "\n")

    assert results[last]["prefill"] > FINAL_PCC_BAR, f"layer {last} chain prefill pcc {results[last]['prefill']:.6f}"
    if full:
        assert g["top1"] >= TOP1_BAR, f"top-1 agreement {g['top1']:.4f} < {TOP1_BAR}"
        for step, gd in enumerate(dec_results):
            assert gd["final_pcc"] > FINAL_PCC_BAR, f"decode step {step}: layer-44 pcc {gd['final_pcc']:.6f}"
    else:
        logger.warning(f"partial stack ({layers[0]}..{last}): head gates skipped")


@pytest.mark.timeout(7000)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_generate_vs_hf(mesh_device):
    from transformers import AutoTokenizer

    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids  # [1, S]
    S = prompt_ids.shape[1]
    logger.info(f"prompt {PROMPT!r}: {S} tokens {prompt_ids[0].tolist()}")
    model = _model(mesh_device, max_seq_len=S + N_TOKENS + 32)

    def on_token(rec):
        logger.info(
            f"TT step {rec['step']} (pos {rec['position']}): token {rec['token']} {tok.decode([rec['token']])!r} "
            f"(device greedy {rec['device_greedy']}), top-5 {rec['top5_ids']} {rec['top5_logits']}, pass {rec['pass_s']} s"
        )

    t0 = time.perf_counter()
    tt_ids, records = model.generate(prompt_ids, N_TOKENS, on_token=on_token)
    t_gen = time.perf_counter() - t0
    tt_text = tok.decode(tt_ids)
    logger.info(f"TT greedy {N_TOKENS} tokens in {t_gen:.0f} s: {tt_ids}\nTT text: {tt_text!r}")
    _log_dram(mesh_device, "after generation")
    _report(model)
    for rec in records:
        assert rec["device_greedy"] is None or rec["device_greedy"] == rec["token"], f"device argmax differs: {rec}"

    hf_file = HF_GEN / "generate.json"
    if not hf_file.exists():
        pytest.fail(
            f"HF reference {hf_file} missing: run hf_layer_reference.py --generate {N_TOKENS} (see the docstring)"
        )
    hf = json.loads(hf_file.read_text())
    assert hf["prompt_ids"] == prompt_ids[0].tolist(), "the HF reference was generated from another prompt"
    hf_ids = hf["generated"][:N_TOKENS]
    assert len(hf_ids) == N_TOKENS, f"HF reference holds {len(hf_ids)} tokens (still running?)"
    hf_text = tok.decode(hf_ids)
    matches = sum(int(a == b) for a, b in zip(tt_ids, hf_ids))
    prefix = 0
    while prefix < N_TOKENS and tt_ids[prefix] == hf_ids[prefix]:
        prefix += 1
    logger.info(f"HF greedy: {hf_ids}\nHF text: {hf_text!r}")
    per_step = []
    for k, (rec, hrec) in enumerate(zip(records, hf["records"])):
        same = rec["token"] == hrec["token"]
        in_top5 = hrec["token"] in rec["top5_ids"]
        per_step.append(
            f"{k:2d}: TT {rec['token']:6d} HF {hrec['token']:6d} {'==' if same else '!='} (HF token in TT top-5: {in_top5})"
        )
    logger.info("per step (TT and HF each continue their OWN sequence after a divergence):\n" + "\n".join(per_step))
    logger.info(
        f"GENERATION_RESULT matches {matches}/{N_TOKENS} position-wise, identical prefix {prefix}; "
        f"TT {tt_text!r} vs HF {hf_text!r}"
    )
    # Gate (divergence-aware): greedy sequences are only comparable up to their first divergence (after it the
    # two decoders continue different sentences), so the gate is the identical prefix plus the character of the
    # divergence: HF's token must be inside the TT top-5 there (a near tie, not a wrong distribution). Measured
    # 2026-10-20 (doc/full_model/README.md): prefix 7, divergence at a 0.25-logit tie with HF's token as TT's #2.
    assert tt_ids[0] == hf_ids[0], f"first generated token differs: TT {tt_ids[0]} vs HF {hf_ids[0]}"
    assert prefix >= PREFIX_BAR, f"identical prefix {prefix} < {PREFIX_BAR}"
    if prefix < N_TOKENS:
        assert (
            hf_ids[prefix] in records[prefix]["top5_ids"]
        ), f"at the divergence (step {prefix}) HF's token {hf_ids[prefix]} is not in the TT top-5 {records[prefix]['top5_ids']}"
