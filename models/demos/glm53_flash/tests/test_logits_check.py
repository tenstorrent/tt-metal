# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only check of ``reference/logits_check.py`` on the natural-prompt HF dump (no device, no ttnn).

1. the dump's logits against the host head on the dump's own last-layer ``x_out``: PCC 1.0, top-1 / top-5 1.0,
   KL 0 (the helper and the dump file agree);
2. the same head with the hidden rounded to bf16 (``round_hidden="h"``: the stream mean; ``"hn"``: the normalized
   hidden, a bf16-activation head's matmul input) and, on top, ``lm_head`` rounded to bfp8 the way the device stores
   it (``round_weight="bfp8"``, ``tt/model_head.py``) against the fp32 logits: the CPU floor of the logits metrics,
   which the device head cannot beat; the numbers are logged, not gated;
3. the partial-chain path (``reference_logits`` on a layer without a logits file) is the host head on ``x_out``.

    GLM53_DUMP=L45_NAT_S128 pytest models/demos/glm53_flash/tests/test_logits_check.py -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.reference.logits_check import (
    LOGITS_FILE,
    META_FILE,
    TOPK_FILE,
    compare_logits,
    dump_last_layer,
    host_head,
    load_head_weights,
    reference_logits,
)

_dump = os.environ.get("GLM53_DUMP", "L45_NAT_S128")
DUMP = Path(_dump) if "/" in _dump else Path("bringup/references/glm53_flash") / _dump
S = int(os.environ.get("GLM53_LOGITS_S", "0")) or None  # positions to use (0 = all)


def _eps() -> float:
    from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir

    return float(json.loads((default_snapshot_dir() / "config.json").read_text())["text_config"]["rms_norm_eps"])


@pytest.fixture(scope="module")
def head():
    return load_head_weights()


@pytest.mark.skipif(not (DUMP / LOGITS_FILE).exists(), reason=f"{DUMP / LOGITS_FILE} not produced yet")
def test_dump_logits_self_consistent(head):
    norm_w, lm_w = head
    eps = _eps()
    last = dump_last_layer(DUMP)
    assert last is not None, f"{DUMP / META_FILE} has no logits_layer"
    ref, source = reference_logits(DUMP, last, norm_w, lm_w, eps, S=S)
    assert source == str(DUMP / LOGITS_FILE)
    x_out = torch.load(DUMP / f"layer{last}_prefill.pt")["x_out"][0]
    x_out = x_out if S is None else x_out[:S]
    dev = host_head(x_out, norm_w, lm_w, eps)
    rep = compare_logits(dev, ref)
    logger.info(rep.summary(f"HF dump {DUMP.name} vs itself (fp32 head on layer {last} x_out)"))
    assert rep.min_pcc > 0.999999 and rep.top1_rate == 1.0 and rep.topk_rate == 1.0 and rep.max_kl < 1e-6
    top = torch.load(DUMP / TOPK_FILE)
    assert torch.equal(top["top5_ids"][: rep.n], rep.ref_topk) and torch.equal(top["top1_ids"][: rep.n], rep.ref_top1)
    meta = json.loads((DUMP / META_FILE).read_text())
    assert meta["n_tokens"] == torch.load(DUMP / "prefill_input.pt")["prompt_ids"].numel()
    bf16_top1 = top["top1_ids_bf16_head"][: rep.n]
    logger.info(
        f"HF's own bf16 head path top-1 == fp32 head top-1 at {float((bf16_top1 == rep.ref_top1).float().mean()) * 100:.1f} % "
        f"of {rep.n} positions; prompt: {meta['prompt_text'][:80]!r}...; next-token top-5 {meta['next_token_top5_text']}"
    )


@pytest.mark.skipif(not (DUMP / LOGITS_FILE).exists(), reason=f"{DUMP / LOGITS_FILE} not produced yet")
def test_bf16_hidden_floor(head):
    """The floor: the HF streams' mean / normalized hidden rounded to bf16 before the fp32 head vs the fp32 head."""
    norm_w, lm_w = head
    eps = _eps()
    last = dump_last_layer(DUMP)
    ref, _ = reference_logits(DUMP, last, norm_w, lm_w, eps, S=S)
    x_out = torch.load(DUMP / f"layer{last}_prefill.pt")["x_out"][0]
    x_out = x_out if S is None else x_out[:S]
    floors = {}
    for mode, what, rw in (
        ("h", "stream mean rounded to bf16", None),
        ("hn", "normalized hidden rounded to bf16", None),
        ("hn+bfp8", "bf16 normalized hidden x bfp8-rounded lm_head (the device head's storage)", "bfp8"),
    ):
        rep = compare_logits(host_head(x_out, norm_w, lm_w, eps, round_hidden=mode.split("+")[0], round_weight=rw), ref)
        logger.info(rep.summary(f"FLOOR {what} ({mode})"))
        for line in rep.rows(every=rep.n):  # the top-1 mismatches and the last position
            logger.info("  " + line)
        floors[mode] = rep
        assert rep.mean_pcc > 0.99, f"bf16 rounding of the hidden alone costs more than expected: {rep.summary(mode)}"
    out = os.environ.get("GLM53_LOGITS_FLOOR_JSON")
    if out:
        Path(out).write_text(json.dumps({m: r.as_dict() for m, r in floors.items()}, indent=1))


@pytest.mark.skipif(not (DUMP / "layer5_prefill.pt").exists(), reason=f"{DUMP} has no layer5_prefill.pt")
def test_partial_chain_reference(head):
    """A chain that stops at layer 5: the reference is the host head on that layer's x_out (no logits file)."""
    norm_w, lm_w = head
    eps = _eps()
    ref, source = reference_logits(DUMP, 5, norm_w, lm_w, eps, S=8)
    assert source.startswith("host head on")
    x_out = torch.load(DUMP / "layer5_prefill.pt")["x_out"][0, :8]
    rep = compare_logits(host_head(x_out, norm_w, lm_w, eps), ref)
    assert rep.min_pcc > 0.999999 and rep.top1_rate == 1.0
    logger.info(rep.summary("partial chain (layer 5, 8 positions) vs itself"))
