# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 — the reference against ground truth, and the golden cache's contract. Host only.

Three things, in increasing cost:

1. **golden-cache round-trip** — a second run loads from disk instead of recomputing, and a changed
   ``ReferenceCacheKey`` field forces a miss rather than silently reusing a stale result. No model.
2. **reference vs the real HF checkpoint** — ground truth for the whole model, not self-consistency.
   Depth-reduced to 2 layers so two copies of the weights fit comfortably; labelled ``REDUCED``.
3. **reference vs the golden trace** — the graded artifact's convention, at full 32-layer depth with
   real weights over a prefix of the trace's own tokens. This is what establishes that the device's
   acceptance comparison is looking at the right tensors in the right layout, and its negative
   controls (a Meta-permuted K and a pre-RoPE K must NOT match) are what stop a layout bug from
   cancelling out on both sides.

The expensive ones skip when the checkpoint or the trace is not reachable, so the host-only suite
still runs on a machine without the shared store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.llama_3_1_8b.reference import golden_cache
from models.demos.llama_3_1_8b.reference import model as ref
from models.demos.llama_3_1_8b.reference.config import LlamaConfig

CHECKPOINT = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"
TRACE = os.getenv("PREFILL_TRACE_DIR") or f"{CHECKPOINT}/golden/synthetic_10240"
TRACE_PREFIX_TOKENS = int(os.getenv("LLAMA_REF_TRACE_TOKENS", "512"))


def _pcc(a, b):
    from models.common.utility_functions import comp_pcc

    return float(comp_pcc(a, b, 0.0)[1])


def _have_checkpoint():
    return Path(CHECKPOINT, "model.safetensors.index.json").exists()


def _have_trace():
    return Path(TRACE, "metadata.json").exists()


requires_checkpoint = pytest.mark.skipif(not _have_checkpoint(), reason=f"no checkpoint at {CHECKPOINT}")
requires_trace = pytest.mark.skipif(not _have_trace(), reason=f"no golden trace at {TRACE}")


# ---------------------------------------------------------------------------------------------
# 1. golden cache round-trip
# ---------------------------------------------------------------------------------------------
def test_golden_cache_roundtrip(tmp_path, monkeypatch):
    """Save, reload identical tensors, and prove a changed key field is a MISS, not a silent reuse."""
    monkeypatch.setenv(golden_cache.VARIANT.ref_cache_env, str(tmp_path))
    key = golden_cache.cache_key(weight_type="random", input_source="unit", isl_total=64, num_layers=2)
    assert not golden_cache.exists(key), "a fresh cache dir must start empty"

    torch.manual_seed(0)
    hidden = [torch.randn(1, 64, 16) for _ in range(2)]
    kv = [(torch.randn(1, 2, 64, 8), torch.randn(1, 2, 64, 8)) for _ in range(2)]
    golden_cache.save(key, hidden, kv)

    assert golden_cache.exists(key)
    got_hidden, got_kv = golden_cache.load(key)
    for a, b in zip(hidden, got_hidden):
        assert torch.equal(a, b)
    for (k, v), (gk, gv) in zip(kv, got_kv):
        assert torch.equal(k, gk) and torch.equal(v, gv)

    # Every field that changes the output must change the filename.
    for changed in (
        golden_cache.cache_key(weight_type="pretrained", input_source="unit", isl_total=64, num_layers=2),
        golden_cache.cache_key(weight_type="random", input_source="other", isl_total=64, num_layers=2),
        golden_cache.cache_key(weight_type="random", input_source="unit", isl_total=128, num_layers=2),
        golden_cache.cache_key(weight_type="random", input_source="unit", isl_total=64, num_layers=3),
    ):
        assert not golden_cache.exists(changed), f"{changed} collided with the stored key"


def test_golden_cache_miss_is_loud(tmp_path, monkeypatch):
    """A miss raises rather than silently recomputing — the CI behaviour the recipe asks for."""
    monkeypatch.setenv(golden_cache.VARIANT.ref_cache_env, str(tmp_path))
    key = golden_cache.cache_key(weight_type="random", input_source="absent", isl_total=8, num_layers=1)
    with pytest.raises(FileNotFoundError):
        golden_cache.load(key)


# ---------------------------------------------------------------------------------------------
# 2. reference vs the real HF checkpoint
# ---------------------------------------------------------------------------------------------
@requires_checkpoint
def test_reference_vs_hf_real_weights():
    """REDUCED (2 of 32 layers), REAL weights: the reference must be HuggingFace, not merely close.

    Depth-reduced only so two copies of the weights are cheap; the width, the head counts, the rope
    and the vocab are all the real ones, which is where a reference usually goes wrong.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig as HFConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from models.demos.llama_3_1_8b.tt.model_config import load_state_dict

    n_layers = 2
    cfg = LlamaConfig.from_json()
    cfg.num_hidden_layers = n_layers
    sd = load_state_dict(CHECKPOINT, num_layers=n_layers)

    ours = ref.load_reference_from_state_dict(cfg, sd)

    hf_cfg = HFConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_scaling=dict(cfg.rope_scaling),
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        attn_implementation="eager",
        dtype=torch.float16,
    )
    hf = LlamaForCausalLM(hf_cfg).to(torch.float16)
    hf.load_state_dict({k: v.to(torch.float16) for k, v in sd.items()}, strict=True)
    hf.eval()

    ids = torch.tensor(json.load(open(Path(TRACE, "metadata.json")))["token_ids"][:256]).unsqueeze(0) if _have_trace() \
        else torch.randint(0, cfg.vocab_size, (1, 256))
    logits, kv = ours(ids)
    with torch.no_grad():
        hf_out = hf(ids, use_cache=True)

    p = _pcc(hf_out.logits, logits)
    logger.info(f"REDUCED (2 layers) reference vs HF LlamaForCausalLM, real weights: logits PCC {p:.6f}")
    assert p > 0.999
    assert torch.equal(hf_out.logits.argmax(-1), logits.argmax(-1)), "argmax token differs from HF"
    for i, (k, v) in enumerate(kv):
        assert _pcc(hf_out.past_key_values.layers[i].keys, k) > 0.999
        assert _pcc(hf_out.past_key_values.layers[i].values, v) > 0.999


# ---------------------------------------------------------------------------------------------
# 3. reference vs the golden trace (full depth, real weights, prefix of the trace)
# ---------------------------------------------------------------------------------------------
@requires_checkpoint
@requires_trace
def test_reference_matches_golden_trace_convention():
    """Full 32-layer depth, REAL weights, the trace's own tokens — but only the first
    ``LLAMA_REF_TRACE_TOKENS`` (512) of them, so this stays a convention check rather than a second
    golden generation. It establishes that the trace stores HF-layout post-RoPE K and raw V, which
    is what the device comparison converts from."""
    from safetensors import safe_open

    from models.demos.llama_3_1_8b.tt.model_config import load_state_dict
    from models.demos.llama_3_1_8b.utils.rope_layout import hf_to_meta_perm

    cfg = LlamaConfig.from_json()
    meta = json.load(open(Path(TRACE, "metadata.json")))
    n = min(TRACE_PREFIX_TOKENS, meta["n_tokens"])
    ids = torch.tensor(meta["token_ids"][:n]).unsqueeze(0)

    model = ref.load_reference_from_state_dict(cfg, load_state_dict(CHECKPOINT))
    _, kv = model(ids, skip_lm_head=True)

    perm = hf_to_meta_perm(cfg.head_dim)
    worst_k = worst_v = 1.0
    for L in range(cfg.num_hidden_layers):
        with safe_open(str(Path(TRACE, "kv_cache", f"layer_{L}.safetensors")), framework="pt") as h:
            g_k = h.get_tensor(f"key_cache_layer_{L}").float()[:, :, :n, :]
            g_v = h.get_tensor(f"value_cache_layer_{L}").float()[:, :, :n, :]
        k, v = kv[L]
        pk, pv = _pcc(g_k, k.float()), _pcc(g_v, v.float())
        worst_k, worst_v = min(worst_k, pk), min(worst_v, pv)
        if L == 0:
            # Controls, on the layer where every other test also looks.
            assert _pcc(g_k[..., perm], k.float()) < 0.9, "golden K matches in the Meta order too — the permutation is a no-op"
            _, k_pre, _ = model.layers[0].self_attn.project(model.embed_tokens(ids))
            assert _pcc(g_k, k_pre.float()) < 0.99, "golden K matches PRE-RoPE K — the trace is not post-RoPE"
    logger.info(f"reference vs golden trace over {n} tokens, 32 layers: min K {worst_k:.6f}, min V {worst_v:.6f}")
    assert worst_v > 0.999, f"golden V should be exact (raw, never rotated); got {worst_v:.6f}"
    assert worst_k > 0.99, f"golden K over the first {n} tokens should be near-exact; got {worst_k:.6f}"
