# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Why the acceptance K numbers sit below ``pcc_target`` — root-caused, not hand-waved. Host only.

The finding: **the golden trace's rope inverse frequencies were rounded to float16 before the
position outer product.** Recompute layer 0's post-RoPE K from the real checkpoint with fp16-rounded
inverse frequencies and it matches the trace at PCC **1.000000** at every position; with the fp32 or
fp64 frequencies HuggingFace actually uses, it matches at 0.9861 overall and degrades monotonically
with position (0.9994 over the first 2048 tokens, 0.9677 over the last 2048).

Why the error grows with position: rounding ``inv_freq`` perturbs a *frequency*, and the resulting
phase error is ``pos * delta_inv`` — linear in position rather than bounded. The fastest unscaled
frequency (j=1, inv=0.814617) rounds to the nearest fp16 value, 1.64e-4 low (a 2.0e-4 relative shift), so
by token 10240 the phase is off by ~1.68 rad. j=0 (inv exactly 1.0) is exactly representable and
shows no drift at all, which is the signature that distinguishes this from ordinary precision loss.

This is layer 0 only, so it needs an embedding lookup, one norm, one projection and the rotation —
seconds, not a CPU forward.

Consequence for the bring-up: the device follows HuggingFace (full-precision frequencies) by
default, so its K is measured against a trace whose phases differ. ``LLAMA_ROPE_FREQ_FP16=1`` flips
both the reference and the device onto the trace's convention; the README records the acceptance
numbers both ways. Nothing here changes the default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.llama_3_1_8b.reference import model as ref
from models.demos.llama_3_1_8b.reference.config import LlamaConfig

CHECKPOINT = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"
TRACE = os.getenv("PREFILL_TRACE_DIR") or f"{CHECKPOINT}/golden/synthetic_10240"

pytestmark = pytest.mark.skipif(
    not (Path(CHECKPOINT, "model.safetensors.index.json").exists() and Path(TRACE, "metadata.json").exists()),
    reason=f"needs the real checkpoint at {CHECKPOINT} and the golden trace at {TRACE}",
)


def _pcc(a, b):
    from models.common.utility_functions import comp_pcc

    return float(comp_pcc(a, b, 0.0)[1])


def _layer0_pre_rope_k(cfg, ids):
    """Layer 0's pre-RoPE K: embedding -> input_layernorm -> k_proj. No attention, no other layer."""
    from safetensors.torch import load_file

    index = json.load(open(Path(CHECKPOINT, "model.safetensors.index.json")))["weight_map"]
    need = {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.k_proj.weight",
    }
    sd = {}
    for shard in sorted({index[k] for k in need}):
        loaded = load_file(str(Path(CHECKPOINT, shard)))
        sd.update({k: loaded[k].to(torch.float16) for k in need if k in loaded})
        del loaded

    n = ids.shape[-1]
    x = torch.nn.functional.embedding(ids, sd["model.embed_tokens.weight"]).float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    x = (x * sd["model.layers.0.input_layernorm.weight"].float()).to(torch.float16)
    k = torch.nn.functional.linear(x, sd["model.layers.0.self_attn.k_proj.weight"])
    return k.view(1, n, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)


def _rope_with(k, inv, n):
    t = torch.arange(n, dtype=torch.float64)
    freqs = torch.outer(t, inv.double())
    emb = torch.cat((freqs, freqs), dim=-1)
    return ref.apply_rope(k, emb.cos().to(torch.float16), emb.sin().to(torch.float16)).float()


def test_golden_trace_rope_frequencies_are_fp16_rounded():
    from safetensors import safe_open

    cfg = LlamaConfig.from_json()
    meta = json.load(open(Path(TRACE, "metadata.json")))
    n = meta["n_tokens"]
    ids = torch.tensor(meta["token_ids"]).unsqueeze(0)

    k_pre = _layer0_pre_rope_k(cfg, ids)
    with safe_open(str(Path(TRACE, "kv_cache", "layer_0.safetensors")), framework="pt") as h:
        golden = h.get_tensor("key_cache_layer_0").float()

    inv = ref.llama3_inv_freq(cfg)
    assert not ref.freqs_rounded_to_fp16(), "this test characterises the DEFAULT (full-precision) path"

    full = _rope_with(k_pre, inv, n)
    fp16 = _rope_with(k_pre, inv.to(torch.float16).double(), n)

    p_full, p_fp16 = _pcc(golden, full), _pcc(golden, fp16)
    logger.info(f"layer-0 K vs golden: full-precision freqs {p_full:.6f}, fp16-rounded freqs {p_fp16:.6f}")
    assert p_fp16 > 0.99999, f"the fp16-frequency hypothesis should be exact; got {p_fp16:.6f}"
    assert p_full < 0.99, f"the full-precision path should NOT match the trace; got {p_full:.6f}"

    # The error is position-proportional, which is what makes it a phase offset and not precision.
    blocks = [_pcc(golden[:, :, s : s + 2048], full[:, :, s : s + 2048]) for s in range(0, n, 2048)]
    logger.info("full-precision per-2048-token PCC: " + ", ".join(f"{p:.5f}" for p in blocks))
    assert blocks == sorted(blocks, reverse=True), "error should grow monotonically with position"
    assert blocks[0] > 0.999 and blocks[-1] < 0.98

    # ... and it is concentrated in the FAST frequencies. The right quantity is the accumulated
    # PHASE error at the last position, ``(n-1) * |delta_inv|``, not the relative frequency shift:
    # the slow (heavily rescaled) frequencies are below fp16's smallest normal, 6.1e-5, so their
    # RELATIVE error is up to 5% — but their angles are so small that the phase barely moves.
    fp16_inv = inv.to(torch.float16).double()
    delta = (fp16_inv - inv).abs()
    phase = delta * (n - 1)
    worst = int(phase.argmax())
    logger.info(
        f"worst accumulated phase error at position {n - 1}: {phase[worst]:.3f} rad at j={worst} "
        f"(inv={inv[worst]:.6f}); slowest-frequency phase error {phase[-1]:.2e} rad"
    )
    assert delta[0] == 0, "inv_freq[0] is exactly 1.0 and must survive the rounding unchanged"
    assert worst == 1, "the fastest UNSCALED frequency should dominate the phase error"
    assert 1.5 < phase[worst] < 2.0, f"expected ~1.7 rad of drift by token {n}, got {phase[worst]:.3f}"
    assert phase[-1] < 0.01, "the slow frequencies must contribute essentially no phase error"

    # Where fp16 is NORMAL (>= 6.1e-5), rounding is bounded by half the spacing. The spacing is
    # 2^(exp-10), so the relative half-spacing ranges from 2^-12 (just below a power of two) up to
    # 2^-11 (just above one) — 4.88e-4 is the ceiling, not 2.44e-4.
    normal = inv >= 6.1e-5
    assert ((delta[normal] / inv[normal]) <= 2.0**-11).all(), "shift exceeds fp16 rounding; another cause"


def test_rope_freq_knob_flips_both_sides_together():
    """The knob is read in ONE place, so the reference and the device cannot straddle it."""
    from models.demos.llama_3_1_8b.tt.rope import inv_freq as device_inv_freq

    assert device_inv_freq is ref.llama3_inv_freq
