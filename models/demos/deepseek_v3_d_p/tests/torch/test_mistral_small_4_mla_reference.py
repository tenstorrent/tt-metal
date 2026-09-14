# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for Mistral's llama4 query temperature. No TTNN, no device, no checkpoint.

``mla_reference.MLAReference`` is a vendored copy of DeepSeek's attention. Mistral reuses DeepSeek's
MLA field names, so its dimensions slot in with no error -- and the device is then validated against
DeepSeek's algorithm running on Mistral's numbers. Mistral ships its own implementation in
transformers (``transformers.models.mistral4``), which is an independent second opinion, so these
forward identical weights and input through both and compare them.

Both tests exist because the DEVICE tests cannot see this term. It moves full-output PCC by ~0.002
against test_mla.py's 0.98 gate, so every chunked scenario passes with the temperature entirely
absent. test_mla.py::test_llama4_query_scale_matches_rotated_positions covers the tensor itself;
these cover the arithmetic and its effect on an attention output.
"""

import math

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import mistral4_hf_config
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference

SEED = 42

# One window past original_max_position_embeddings, so positions 8192..8703 carry the temperature
# (1.0693) while 0..8191 stay at exactly 1.0. The smallest length that puts the boundary INSIDE the
# tensor; anything shorter cannot see the term at all. Cheaper than 17x a 512-token run suggests
# (~13 s for the three forwards): MLAReference chunks attention above SEQ_CHUNK=4096.
LONG_SEQ_LEN = 8704


def _mla_weights(config, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    std = config.initializer_range
    qk = config.qk_nope_head_dim + config.qk_rope_head_dim
    return {
        "q_a_proj.weight": (torch.randn(config.q_lora_rank, config.hidden_size) * std).to(torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": (torch.randn(config.num_attention_heads * qk, config.q_lora_rank) * std).to(torch.bfloat16),
        "kv_a_proj_with_mqa.weight": (
            torch.randn(config.kv_lora_rank + config.qk_rope_head_dim, config.hidden_size) * std
        ).to(torch.bfloat16),
        "kv_a_layernorm.weight": torch.ones(config.kv_lora_rank, dtype=torch.bfloat16),
        "kv_b_proj.weight": (
            torch.randn(config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim), config.kv_lora_rank)
            * std
        ).to(torch.bfloat16),
        "o_proj.weight": (torch.randn(config.hidden_size, config.num_attention_heads * config.v_head_dim) * std).to(
            torch.bfloat16
        ),
    }


@pytest.fixture(scope="module")
def config():
    cfg = mistral4_hf_config(max_seq=LONG_SEQ_LEN)
    cfg._attn_implementation = "eager"  # deterministic, and no flash-attn head-dim padding path
    return cfg


@pytest.fixture(scope="module")
def weights(config):
    return _mla_weights(config)


@pytest.fixture(scope="module")
def hidden(config):
    torch.manual_seed(SEED + 1)
    return (torch.randn(1, LONG_SEQ_LEN, config.hidden_size) * 0.02).to(torch.bfloat16)


def _run_vendored(config, weights, hidden, *, force_no_llama4_scale=False):
    """mla_reference.MLAReference -- the vendored DeepSeek attention the device is PCC'd against."""
    ref = create_mla_reference(
        config=config, state_dict={f"model.layers.0.self_attn.{k}": v for k, v in weights.items()}
    )
    if force_no_llama4_scale:
        ref.llama4_beta = None  # revert just the temperature, to attribute it
    ref = ref.eval().to(torch.bfloat16)
    with torch.no_grad():
        out, _, _ = ref(hidden_states=hidden, position_ids=torch.arange(hidden.shape[1])[None])
    return out


def _run_upstream(config, weights, hidden):
    """transformers.models.mistral4.Mistral4Attention -- Mistral's own implementation."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention, Mistral4RotaryEmbedding

    attn = Mistral4Attention(config, layer_idx=0)
    attn.load_state_dict(weights, strict=True)
    attn = attn.eval().to(torch.bfloat16)

    # fp32 rotary is deliberate: a bf16 module quantizes inv_freq and the phase error grows with
    # position, which is exactly the regime this test runs in.
    rotary = Mistral4RotaryEmbedding(config).float()
    assert rotary.inv_freq.dtype is torch.float32, "rotary must stay fp32"
    pos = torch.arange(hidden.shape[1])[None]
    with torch.no_grad():
        cos, sin = rotary(hidden.float(), pos)
    cos, sin = cos.to(hidden.dtype), sin.to(hidden.dtype)

    q_len = hidden.shape[1]
    causal = torch.triu(torch.full((q_len, q_len), float("-inf"), dtype=hidden.dtype), diagonal=1)
    with torch.no_grad():
        out, _ = attn(
            hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=causal[None, None], position_ids=pos
        )
    return out


def _mean_row_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token relative L2 error. PCC is useless here -- see the docstring below."""
    a, b = a.float()[0], b.float()[0]
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)


def test_llama4_scale_formula_matches_upstream(config):
    """mla_reference re-derives the temperature; it must equal upstream's bit-for-bit.

    The cheap guard. mla_reference deliberately does NOT import get_llama_4_attn_scale -- that would
    put the code under test on both sides of the head-to-head below -- so the formula is duplicated,
    and a duplicated formula needs a test pinning it to the original. Pure arithmetic, no attention.
    """
    from transformers.models.mistral4.modeling_mistral4 import get_llama_4_attn_scale

    ref = create_mla_reference(
        config=config, state_dict={f"model.layers.0.self_attn.{k}": v for k, v in _mla_weights(config).items()}
    )
    assert ref.llama4_beta is not None, "llama_4_scaling_beta is not reaching mla_reference"
    assert ref.llama4_orig_max == 8192

    probe = torch.tensor([[0, 1, 8191, 8192, 8193, 16383, 16384, 49152, 51200, 1040000]])
    mine = ref._llama4_attn_scale(probe)
    theirs = get_llama_4_attn_scale(probe, ref.llama4_beta, ref.llama4_orig_max)
    assert torch.equal(mine, theirs), f"max |diff| {(mine - theirs).abs().max().item():.3e} vs upstream"
    logger.info(
        f"scale matches upstream at {probe.numel()} probe positions; 1.0 below 8192, "
        f"{mine.flatten()[3]:.6f} at 8192, {mine.flatten()[-1]:.6f} at 1040000"
    )


def test_llama4_scale_is_the_whole_divergence_above_8192(config, weights, hidden):
    """Drop the temperature and the rows past 8192 must degrade; the rows below must not move at all.

    Two things about this test are forced by the term's shape rather than chosen:

    1. **It needs a sequence past 8192.** The scale is exactly 1.0 below that, so at any shorter length
       both variants are bit-identical and this would pass with the term deleted.
    2. **The metric is per-row error, not PCC.** Measured on this input: whole-tensor PCC reads
       1.000000 with AND without the scale, and even restricted to the rows above 8192 it only falls to
       0.999442. A ~7% uniform scale on Q sharpens the softmax without decorrelating the output, so no
       PCC threshold this file would accept can gate it.

    The below-8192 assertion is exact, not approximate: the term is identically 1.0 there, so disabling
    it must not perturb those rows by a single bit. That is what catches the scale being applied at the
    wrong POSITIONS rather than simply being absent.
    """
    orig_max = config.rope_scaling["original_max_position_embeddings"]
    assert orig_max < LONG_SEQ_LEN, f"LONG_SEQ_LEN {LONG_SEQ_LEN} must exceed orig_max {orig_max}"

    upstream = _run_upstream(config, weights, hidden)
    with_scale = _run_vendored(config, weights, hidden)
    without = _run_vendored(config, weights, hidden, force_no_llama4_scale=True)

    below, above = slice(0, orig_max), slice(orig_max, LONG_SEQ_LEN)
    err_with, err_without = _mean_row_err(with_scale, upstream), _mean_row_err(without, upstream)
    w_below, w_above = err_with[below].mean().item(), err_with[above].mean().item()
    n_below, n_above = err_without[below].mean().item(), err_without[above].mean().item()

    logger.info(f"mean row err vs upstream -- with scale: <{orig_max} {w_below:.6f}, >={orig_max} {w_above:.6f}")
    logger.info(f"mean row err vs upstream -- no scale:   <{orig_max} {n_below:.6f}, >={orig_max} {n_above:.6f}")

    assert torch.equal(with_scale[:, below], without[:, below]), (
        f"disabling the temperature perturbed rows below {orig_max}; it is identically 1.0 there, so it "
        "is being applied at the wrong positions"
    )
    assert w_above < 2 * w_below, (
        f"rows >={orig_max} still disagree ({w_above:.6f}) far more than rows below ({w_below:.6f}); the "
        "temperature is not the whole story above the boundary"
    )
    assert n_above > 5 * w_above, (
        f"dropping the temperature left rows >={orig_max} at {n_above:.6f} vs {w_above:.6f} with it; "
        "either mla_reference stopped applying it or the term no longer matters"
    )


def test_llama4_scale_is_inert_below_original_max_position(config):
    """The term is exactly 1.0 below 8192 -- latent, not absent. Pins why short tests cannot see it."""
    from transformers.models.mistral4.modeling_mistral4 import get_llama_4_attn_scale

    beta = config.rope_scaling["llama_4_scaling_beta"]
    orig_max = config.rope_scaling["original_max_position_embeddings"]
    assert (beta, orig_max) == (0.1, 8192)

    inert = get_llama_4_attn_scale(torch.arange(orig_max)[None], beta, orig_max)
    assert torch.all(inert == 1.0), "expected a no-op below original_max_position_embeddings"

    beyond = get_llama_4_attn_scale(torch.tensor([[orig_max]]), beta, orig_max)
    assert beyond.item() == pytest.approx(1 + beta * math.log(2.0))
