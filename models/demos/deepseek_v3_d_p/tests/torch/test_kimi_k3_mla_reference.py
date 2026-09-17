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

import json
from pathlib import Path

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
    """The dims every tuned Kimi-K3 config and per-device shape is written against."""
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


def test_k3_constants_match_the_vendored_checkpoint_config():
    """Every ``KimiK3Config`` constant must equal the value in the vendored upstream config.json.

    ``KimiK3Config`` is hand-transcribed (upstream's own config class cannot be imported here --
    ``modeling_kimi_linear.py`` raises ImportError without ``fla-core``), so without this the numbers
    are trusted rather than checked and a single mistyped digit would pass the whole suite. Two
    failure modes it catches:

      * a transcription slip -- ``qk_rope_head_dim`` and the 24-entry ``full_attn_layers`` list are the
        risky ones, the latter because its tail breaks the otherwise-strict stride of 4 (92 and 93 are
        adjacent), so it cannot be regenerated from a rule;
      * an upstream revision that changes a dimension -- re-vendor config.json and this test tells you
        exactly which fields moved.

    config.json is vendored from ``huggingface.co/moonshotai/Kimi-K3`` at revision
    9f62e4e9fffbd0a83ddd60e1c209d828994b3569, matching the pattern of
    ``reference/kimi_k2_6/config.json``. Content is unmodified; the only difference from the upstream
    bytes is the trailing newline this repo's end-of-file-fixer hook requires, so a re-vendored copy
    diffs clean apart from that last line. It is the multimodal wrapper, so every LM field lives under
    ``text_config``; KDA sizing lives under ``text_config.linear_attn_config``.
    """
    with open(Path(__file__).parents[2] / "reference" / "kimi_k3" / "config.json") as f:
        text_config = json.load(f)["text_config"]
    linear_attn = text_config["linear_attn_config"]

    expected = {
        "EMB_SIZE": text_config["hidden_size"],
        "MOE_INTERMEDIATE_SIZE": text_config["moe_intermediate_size"],
        "INTERMEDIATE_SIZE": text_config["intermediate_size"],
        "NUM_ROUTED_EXPERTS": text_config["num_experts"],
        "NUM_EXPERTS_PER_TOKEN": text_config["num_experts_per_token"],
        "NUM_SHARED_EXPERTS": text_config["num_shared_experts"],
        "NUM_EXPERT_GROUPS": text_config["num_expert_group"],
        "NUM_LIMITED_GROUPS": text_config["topk_group"],
        "ROUTE_SCALE": text_config["routed_scaling_factor"],
        "ROUTED_EXPERT_HIDDEN_SIZE": text_config["routed_expert_hidden_size"],
        "NUM_LAYERS": text_config["num_hidden_layers"],
        "NUM_DENSE_LAYERS": text_config["first_k_dense_replace"],
        "VOCAB_SIZE": text_config["vocab_size"],
        "NUM_ATTENTION_HEADS": text_config["num_attention_heads"],
        "NUM_KEY_VALUE_HEADS": text_config["num_key_value_heads"],
        "Q_LORA_RANK": text_config["q_lora_rank"],
        "KV_LORA_RANK": text_config["kv_lora_rank"],
        "QK_NOPE_HEAD_DIM": text_config["qk_nope_head_dim"],
        "QK_ROPE_HEAD_DIM": text_config["qk_rope_head_dim"],
        "V_HEAD_DIM": text_config["v_head_dim"],
        "USE_NOPE": text_config["mla_use_nope"],
        "USE_OUTPUT_GATE": text_config["mla_use_output_gate"],
        "RMS_NORM_EPS": text_config["rms_norm_eps"],
        "MAX_POSITION_EMBEDDINGS": text_config["max_position_embeddings"],
        "ATTN_RES_BLOCK_SIZE": text_config["attn_res_block_size"],
        "LATENT_MOE_USE_NORM": text_config["latent_moe_use_norm"],
        "ACTIVATION_SITU_BETA": text_config["activation_situ_beta"],
        "ACTIVATION_SITU_LINEAR_BETA": text_config["activation_situ_linear_beta"],
        "KDA_NUM_HEADS": linear_attn["num_heads"],
        "KDA_HEAD_DIM": linear_attn["head_dim"],
        "KDA_SHORT_CONV_KERNEL_SIZE": linear_attn["short_conv_kernel_size"],
        "KDA_GATE_LOWER_BOUND": linear_attn["gate_lower_bound"],
        "FULL_ATTN_LAYERS_1BASED": linear_attn["full_attn_layers"],
    }
    mismatched = {
        name: (getattr(KimiK3Config, name), want)
        for name, want in expected.items()
        if getattr(KimiK3Config, name) != want
    }
    assert not mismatched, f"KimiK3Config disagrees with the vendored config.json: {mismatched}"

    # rope_theta / rope_scaling must be ABSENT upstream, not merely falsy: their absence is what
    # makes K3's softmax scale plain qk_head_dim**-0.5, and it is why ttMLA keys its YaRN guard on the
    # presence of rope_scaling["factor"] (transformers >= 5 synthesizes a factor-less rope_scaling
    # dict for configs that omit it, so an `is not None` guard would still raise KeyError).
    assert "rope_theta" not in text_config
    assert "rope_scaling" not in text_config

    # The hybrid split must partition the layers exactly, and MLA weights are exempt from the MXFP4
    # quantization -- both load-bearing for the MLA work.
    assert len(linear_attn["kda_layers"]) + len(linear_attn["full_attn_layers"]) == KimiK3Config.NUM_LAYERS
    assert len(linear_attn["full_attn_layers"]) == len(KimiK3Config.mla_layer_ids()) == 24
    assert any("self_attn" in pattern for pattern in text_config["quantization_config"]["ignore"])


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


def test_k3_accuracy_pinned_blocking():
    """in0_block_w values pinned for ACCURACY must not be "optimised" without re-running the depth test.

    This is a tripwire, not a preference. ``kv_a_proj_with_mqa`` is the only tuned matmul feeding the
    KV cache, which every later chunk re-reads, so its rounding degrades accuracy cumulatively rather
    than locally. K2.6's ``in0_block_w=14`` drove the chunked output PCC below 0.98 at kv_actual=3840
    of a 56320-token prefill; ``in0_block_w=1`` passes all 44 chunks (0.98550 at kv_actual=55040). A
    bisect pinned it to that one field -- the other six tuned matmuls and the k_chunk=640 SDPA entry
    are all fine.

    Raising it is exactly the kind of change that looks free and is not. The full sweep of K_t=56's
    divisors, each depth-tested:

        ibw   us (isolated)   single-chunk PCC   depth: fails at kv_actual
          1        44.7          0.9999216       never -- all 44, 0.98550 @ 55040
          2        28.7          0.9999339           20480
          4        21.6          0.9999294            7680
          8        16.5          0.9999034            5120
         14        17.9          0.9998523            3840

    ibw=2 has the BEST single-chunk PCC of any value -- better than ibw=1, the one that works -- and
    still fails at 20480. The per-op PCC ranking is INVERTED against depth behaviour here, so no
    op-level measurement can justify raising this. Only a deep chunked run can:
    test_mla_chunked_prefill[k3-production-50k+5k-cpu-8x4-fabric2d] is 11 x 5120 = 56320 tokens at
    S_loc=640, every iteration asserted.

    Resolved through ttMLA rather than by indexing MLA_MATMUL_CONFIG directly: the 640 slot holds
    several candidates and only _select_cfg / _cfg_matches know which one K3 gets.
    """
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
    from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG

    pinned = {"kv_a_proj_with_mqa": 1}

    mla = object.__new__(ttMLA)
    mla.num_heads = KimiK3Config.NUM_ATTENTION_HEADS
    mla.q_lora_rank = KimiK3Config.Q_LORA_RANK
    mla.is_chunked = True
    mla._is_dsa_family = False

    for name, expected_ibw in pinned.items():
        cfg = mla._select_cfg(MLA_MATMUL_CONFIG[name][640])
        assert cfg is not None, f"{name!r} lost its Kimi-K3 tuned config at seq_len_local=640"
        actual = cfg["program_config"].in0_block_w
        assert actual == expected_ibw, (
            f"{name}: in0_block_w is {actual}, pinned at {expected_ibw} for ACCURACY (see the ladder "
            "in this docstring and in mla_config.py). Raising it fails the 0.98 chunked-prefill PCC "
            "at depth. Re-run the 56320-token depth case before changing this."
        )
