# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Host-side PCC test for the two Kimi-K3 MLA references. No TTNN, no device code.

K3's MLA differs from Kimi-K2.6's in three ways that are all easy to get silently wrong, so both
references are pinned against each other before any device work:

  * **NoPE** -- ``rotary_emb`` is None and the 64 ``qk_rope_head_dim`` columns pass through
    unrotated. They are NOT removed: the cached latent row stays ``kv_lora_rank + qk_rope_head_dim``
    = 576 wide, and ``k_rot`` is broadcast across all heads.
  * **softmax scale** -- plain ``qk_head_dim**-0.5``. K2.6 multiplies by ``mscale**2`` (~2.0), so
    inheriting its scale is a silent 2x error that PCC alone would report as a vague miss.
  * **output gate** -- ``sigmoid(g_proj(hidden))`` multiplies the attention output in
    ``num_heads * v_head_dim`` space, strictly after the V-head expansion and before ``o_proj``.

The two references reach the same answer by different routes, which is what makes the comparison
meaningful:
  * ``reference/kimi_k3/modeling_kimi_k3_mla.KimiMLAAttention`` -- the **unabsorbed** upstream
    module (trimmed verbatim from HF), materializing full per-head K and V.
  * ``reference/mla_reference.MLAReference`` -- the **absorbed** form that mirrors the device op
    order: ``q_nope`` folded through ``wkv_b1`` into latent space, attention over the 576-wide
    latent, then ``wkv_b2`` expanding back to ``v_head_dim``.

If the absorption algebra, the gate placement, or the scale is wrong in either one, this test fails
on host in seconds instead of surfacing as an unattributable device PCC miss.
"""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiMLAAttention
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference

# Absorbed vs unabsorbed differ only by bf16 rounding through two extra matmuls.
REFERENCE_PCC = 0.999


def _mla_weights(config, seed: int = 42) -> dict:
    """Random MLA weights in the same key space / shapes the TT weight dict uses.

    Mirrors the ``random_weights`` fixture in ``tests/conftest.py`` (including ``g_proj``), but
    standalone so this test needs no device fixtures.
    """
    torch.manual_seed(seed)
    std = config.initializer_range
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    return {
        "q_a_proj.weight": (torch.randn(config.q_lora_rank, config.hidden_size) * std).to(torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": (torch.randn(config.num_attention_heads * qk_head_dim, config.q_lora_rank) * std).to(
            torch.bfloat16
        ),
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
        "g_proj.weight": (torch.randn(config.num_attention_heads * config.v_head_dim, config.hidden_size) * std).to(
            torch.bfloat16
        ),
    }


def _run_absorbed(config, weights, hidden):
    ref = create_mla_reference(
        config=config,
        state_dict={f"model.layers.0.self_attn.{k}": v for k, v in weights.items()},
    )
    ref = ref.eval().to(torch.bfloat16)
    with torch.no_grad():
        out, _, _ = ref(hidden_states=hidden, position_ids=torch.arange(hidden.shape[1])[None])
    return out


def _run_unabsorbed(config, weights, hidden):
    attn = KimiMLAAttention(config, layer_idx=0)
    attn.load_state_dict(weights, strict=True)  # strict: every K3 weight must land, g_proj included
    attn = attn.eval().to(torch.bfloat16)
    with torch.no_grad():
        return attn(hidden_states=hidden)


def test_k3_config_shapes():
    """The dims the op audit in docs/KIMI_K3_MLA.md is written against."""
    c = kimi_k3_hf_config(5120)
    qk_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
    assert c.num_attention_heads == 96
    assert qk_head_dim == 192
    assert c.q_lora_rank == 1536 and c.kv_lora_rank == 512
    # q_b_proj widens 3072 -> 4608 per device at tp=4; o_proj's K widens 2048 -> 3072.
    assert c.num_attention_heads * qk_head_dim == 18432
    assert c.num_attention_heads * c.v_head_dim == 12288
    # NoPE does NOT shrink the cached latent row.
    assert c.kv_lora_rank + c.qk_rope_head_dim == 576
    # The scale must carry no mscale factor.
    assert c.rope_scaling is None
    assert c.mla_use_nope is True and c.mla_use_output_gate is True


def test_k3_layer_schedule(expect_error):
    """full_attn_layers is 1-indexed upstream; consumers need 0-indexed, and the tail is irregular."""
    ids = KimiK3Config.mla_layer_ids()
    assert len(ids) == 24
    assert ids[0] == 3 and ids[-1] == 92
    # 91 and 92 are adjacent -- the 3 KDA : 1 MLA pattern breaks at the end of the model.
    assert ids[-2] == 91, "the last two layers are both MLA; a strict stride-4 map would be wrong"
    assert KimiK3Config.mla_kv_slot(3) == 0 and KimiK3Config.mla_kv_slot(92) == 23
    # A KDA layer index must raise rather than return a plausible-but-wrong slot.
    with expect_error(ValueError, "KDA layer"):
        KimiK3Config.mla_kv_slot(0)  # layer 0 is KDA


def test_k3_softmax_scale_has_no_mscale():
    """The absorbed reference's scale must be plain qk_head_dim**-0.5.

    K2.6 reaches ~0.1446 via mscale**2; K3 must be ~0.0722. Pinned explicitly because the failure
    mode is a plausible-looking output, not an exception.
    """
    c = kimi_k3_hf_config(512)
    ref = create_mla_reference(
        config=c, state_dict={f"model.layers.0.self_attn.{k}": v for k, v in _mla_weights(c).items()}
    )
    expected = (c.qk_nope_head_dim + c.qk_rope_head_dim) ** -0.5
    assert ref.attention.softmax_scale == pytest.approx(expected)
    assert ref.attention.rotary_emb is not None, "constructed but inert under NoPE"
    assert ref.use_nope and ref.use_output_gate


@pytest.mark.parametrize("seq_len", [128, 512], ids=["seq128", "seq512"])
def test_k3_absorbed_matches_unabsorbed(seq_len):
    """Absorbed (device-order) vs unabsorbed (upstream) K3 MLA, same weights."""
    config = kimi_k3_hf_config(max_seq=seq_len)
    weights = _mla_weights(config)
    torch.manual_seed(7)
    hidden = (torch.randn(1, seq_len, config.hidden_size) * 0.5).to(torch.bfloat16)

    absorbed = _run_absorbed(config, weights, hidden)
    unabsorbed = _run_unabsorbed(config, weights, hidden)

    assert absorbed.shape == unabsorbed.shape == (1, seq_len, config.hidden_size)
    passing, pcc = comp_pcc(absorbed.float(), unabsorbed.float(), REFERENCE_PCC)
    logger.info(f"K3 MLA absorbed vs unabsorbed @ seq{seq_len}: {pcc}")
    assert passing, f"absorbed vs unabsorbed K3 MLA PCC below {REFERENCE_PCC}: {pcc}"


def test_k3_gate_is_load_bearing():
    """Dropping the gate must change the output.

    Guards against the gate being wired up but never applied -- a no-op gate would still pass the
    parity test above if both references skipped it, so pin it against the ungated path.
    """
    seq_len = 128
    config = kimi_k3_hf_config(max_seq=seq_len)
    weights = _mla_weights(config)
    torch.manual_seed(7)
    hidden = (torch.randn(1, seq_len, config.hidden_size) * 0.5).to(torch.bfloat16)

    gated = _run_absorbed(config, weights, hidden)

    ungated_config = kimi_k3_hf_config(max_seq=seq_len)
    ungated_config.mla_use_output_gate = False
    ungated = _run_absorbed(ungated_config, {k: v for k, v in weights.items() if k != "g_proj.weight"}, hidden)

    passing, pcc = comp_pcc(gated.float(), ungated.float(), 0.99)
    logger.info(f"K3 gated vs ungated PCC: {pcc} (expected to FAIL the 0.99 threshold)")
    assert not passing, "gated and ungated MLA agree to PCC 0.99 -- the output gate is not being applied"


def test_k3_nope_is_load_bearing():
    """Enabling RoPE must change the output, i.e. the NoPE branch really bypasses the rotation."""
    seq_len = 128
    config = kimi_k3_hf_config(max_seq=seq_len)
    weights = _mla_weights(config)
    torch.manual_seed(7)
    hidden = (torch.randn(1, seq_len, config.hidden_size) * 0.5).to(torch.bfloat16)

    nope = _run_absorbed(config, weights, hidden)

    roped_config = kimi_k3_hf_config(max_seq=seq_len)
    roped_config.mla_use_nope = False
    roped = _run_absorbed(roped_config, weights, hidden)

    passing, pcc = comp_pcc(nope.float(), roped.float(), 0.99)
    logger.info(f"K3 NoPE vs RoPE PCC: {pcc} (expected to FAIL the 0.99 threshold)")
    assert not passing, "NoPE and RoPE outputs agree to PCC 0.99 -- the rope bypass is not effective"
