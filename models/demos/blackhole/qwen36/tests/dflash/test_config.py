# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``DFlashDrafterConfig`` -- derived shapes, fail-closed validation, reference agreement.

No device. Only :func:`test_literal_matches_real_checkpoint` touches the network, and it
skips when the checkpoint is unreachable.
"""

from __future__ import annotations

import dataclasses

import pytest

from models.demos.blackhole.qwen36.reference.dflash.dflash import Qwen3DFlashAttention
from models.demos.blackhole.qwen36.tests.dflash.conftest import CHECKPOINT_CONFIG, TARGET_NUM_HIDDEN_LAYERS
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig


def _cfg(**overrides) -> DFlashDrafterConfig:
    """The real config with ``overrides`` merged into the raw dict."""
    return DFlashDrafterConfig.from_dict({**CHECKPOINT_CONFIG, **overrides})


# ---- derived shapes ------------------------------------------------------------------


def test_derived_shapes(drafter_cfg):
    assert drafter_cfg.q_dim == 4096  # 32 heads x 128
    assert drafter_cfg.kv_dim == 1024  # 8 heads x 128
    assert drafter_cfg.num_key_value_groups == 4  # GQA 4:1
    assert drafter_cfg.target_feature_size == 25600  # 5 taps x 5120
    assert drafter_cfg.num_draft_tokens == 15  # block_size 16, minus the anchor


def test_target_feature_size_matches_fc_weight_shape(drafter_cfg):
    """``fc.weight`` is ``[5120, 25600]`` in the checkpoint. If these disagree, weight
    loading fails late and confusingly, so pin it here."""
    assert drafter_cfg.target_feature_size == 5 * drafter_cfg.hidden_size
    assert len(drafter_cfg.target_layer_ids) == 5


def test_kv_heads_shard_cleanly_at_tp8(drafter_cfg):
    """8 KV heads over 8 devices is exactly one KV head per chip, with 4 local Q heads
    each -- so the TP=8 split needs no KV replication."""
    tp = 8
    assert drafter_cfg.num_key_value_heads % tp == 0
    assert drafter_cfg.num_attention_heads % tp == 0
    assert (drafter_cfg.num_attention_heads // tp) == drafter_cfg.num_key_value_groups


def test_hidden_and_intermediate_match_the_target(drafter_cfg):
    """The drafter's MLP is dimensionally the target's, which is why ``tt/mlp.py`` and its
    swept program configs are reusable rather than needing their own."""
    assert drafter_cfg.hidden_size == 5120
    assert drafter_cfg.intermediate_size == 17408


# ---- agreement with the reference ----------------------------------------------------


def test_per_layer_mask_flags_match_reference(drafter_cfg):
    """``is_causal`` / ``sliding_window`` agree with ``Qwen3DFlashAttention.__init__``.

    Built as a live differential against the reference class rather than restated by hand,
    so an upstream change to the derivation shows up here instead of in a PCC number.
    """
    hf_cfg = _hf_config_stub(drafter_cfg)
    for layer_idx in range(drafter_cfg.num_hidden_layers):
        ref = Qwen3DFlashAttention(hf_cfg, layer_idx)
        assert drafter_cfg.is_causal(layer_idx) == ref.is_causal, f"layer {layer_idx} is_causal"
        assert drafter_cfg.window_for(layer_idx) == ref.sliding_window, f"layer {layer_idx} window"


def _hf_config_stub(cfg: DFlashDrafterConfig):
    """Minimal object with the attributes ``Qwen3DFlashAttention.__init__`` reads."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

    return Qwen3Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        rms_norm_eps=cfg.rms_norm_eps,
        layer_types=list(cfg.layer_types),
        sliding_window=cfg.sliding_window,
        # Required: Qwen3Config.__post_init__ nulls sliding_window unless this is set
        # (configuration_qwen3.py:87), and its own default is False.
        use_sliding_window=cfg.sliding_window is not None,
        rope_theta=cfg.rope_theta,
        attention_bias=False,
        attention_dropout=0.0,
    )


def test_is_causal_override_is_honoured(drafter_cfg):
    """A checkpoint that sets top-level ``is_causal`` overrides the per-layer default.

    Dormant for Qwen3.6-27B-DFlash (the key is absent) but live for the Kimi-era drafter,
    and the reference honours it at ``dflash.py:363-364``.
    """
    assert drafter_cfg.is_causal_override is None
    assert drafter_cfg.is_causal(0) is True  # sliding layer, default derivation

    forced = _cfg(is_causal=False)
    assert forced.is_causal_override is False
    assert all(not forced.is_causal(i) for i in range(forced.num_hidden_layers))
    # The window is unaffected by the override -- it still keys off layer_type.
    assert forced.window_for(0) == 2048
    assert forced.window_for(4) is None


def test_use_sliding_window_gates_the_window(drafter_cfg, expect_error):
    """``use_sliding_window`` must be honoured, even though ``dflash.py`` never reads it.

    The reference takes an HF ``Qwen3Config``, whose ``__post_init__`` has already applied
    ``sliding_window = sliding_window if use_sliding_window else None``
    (``configuration_qwen3.py:87``), and its default for the flag is ``False``. So the gate
    is real and lives one level up. This checkpoint sets it true.
    """
    assert CHECKPOINT_CONFIG["use_sliding_window"] is True
    assert drafter_cfg.sliding_window == 2048

    # With the flag off there is no window, so the config refuses rather than silently
    # running the sliding layers unwindowed -- see the note in __post_init__.
    with expect_error(ValueError, "use_sliding_window"):
        _cfg(use_sliding_window=False)

    # An omitted flag behaves like HF's default (False), not like true.
    cfg_without_flag = {k: v for k, v in CHECKPOINT_CONFIG.items() if k != "use_sliding_window"}
    with expect_error(ValueError, "use_sliding_window"):
        DFlashDrafterConfig.from_dict(cfg_without_flag)


def test_rope_theta_is_read_from_either_location(drafter_cfg):
    """``rope_theta`` must be found whether it is top-level or nested in ``rope_parameters``.

    transformers 5.x normalises a top-level ``rope_theta`` into
    ``rope_parameters={"rope_theta": ..., "rope_type": "default"}`` and drops the top-level
    attribute. Raw checkpoint JSON has it at top level; a config round-tripped through
    ``Qwen3Config`` has it only nested. Reading just one location silently yields RoPE base
    10000 instead of 1e7 -- garbage RoPE that degrades with context, the F3 failure mode.
    """
    assert drafter_cfg.rope_theta == 1e7  # from top-level, as the checkpoint ships it

    nested = {k: v for k, v in CHECKPOINT_CONFIG.items() if k != "rope_theta"}
    nested["rope_parameters"] = {"rope_theta": 10000000, "rope_type": "default"}
    assert DFlashDrafterConfig.from_dict(nested).rope_theta == 1e7


def test_from_hf_config_round_trips(drafter_cfg):
    """``from_dict`` -> ``Qwen3Config`` -> ``from_hf_config`` is the identity.

    This is the path that exposed the ``rope_theta`` relocation above: constructing a
    ``Qwen3Config`` moves ``rope_theta`` into ``rope_parameters`` and deletes the top-level
    attribute, so a naive ``from_hf_config`` loses it and falls back to base 10000.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

    hf = Qwen3Config(**CHECKPOINT_CONFIG)
    assert not hasattr(hf, "rope_theta"), "assumption changed: Qwen3Config now keeps rope_theta top-level"
    assert DFlashDrafterConfig.from_hf_config(hf) == drafter_cfg


def test_dflash_config_takes_precedence_over_top_level():
    """Resolution order mirrors the reference's ``_draft_value``: ``dflash_config`` wins."""
    cfg = _cfg(block_size=99, dflash_config={**CHECKPOINT_CONFIG["dflash_config"], "block_size": 8})
    assert cfg.block_size == 8
    # And a key present only at top level is still found.
    assert _cfg(block_size=8, dflash_config={"mask_token_id": 248070, "target_layer_ids": [1]}).block_size == 8


# ---- fail-closed validation ----------------------------------------------------------


def test_no_field_has_a_misleading_default():
    """Every shape-bearing field is required.

    The equivalent Kimi config carries its own dims as defaults and relies on a docstring
    warning; here a checkpoint that omits a key cannot silently inherit another model's
    shape. Only genuinely-optional fields may have defaults.
    """
    optional = {"is_causal_override", "initializer_range", "final_logit_softcapping"}
    for field in dataclasses.fields(DFlashDrafterConfig):
        has_default = field.default is not dataclasses.MISSING
        assert has_default == (field.name in optional), f"{field.name}: unexpected default"


@pytest.mark.parametrize("missing", ["hidden_size", "num_hidden_layers", "layer_types", "num_target_layers"])
def test_missing_required_key_raises(missing, expect_error):
    cfg = {k: v for k, v in CHECKPOINT_CONFIG.items() if k != missing}
    with expect_error(KeyError, missing):
        DFlashDrafterConfig.from_dict(cfg)


def test_layer_types_length_must_match_depth(expect_error):
    with expect_error(ValueError, "layer_types has 4 entries"):
        _cfg(layer_types=["sliding_attention"] * 4)


def test_unknown_layer_type_raises(expect_error):
    with expect_error(ValueError, "unknown layer_types"):
        _cfg(layer_types=["linear_attention"] + list(CHECKPOINT_CONFIG["layer_types"])[1:])


def test_sliding_layers_without_a_window_raises(expect_error):
    with expect_error(ValueError, "no sliding_window"):
        _cfg(sliding_window=None)


def test_target_layer_ids_must_fit_the_target(expect_error):
    """A tap past the target's depth would index a layer that does not exist."""
    with expect_error(ValueError, "outside the target's 64 layers"):
        _cfg(dflash_config={**CHECKPOINT_CONFIG["dflash_config"], "target_layer_ids": [1, 16, 31, 46, 64]})


def test_empty_target_layer_ids_raises(expect_error):
    with expect_error(ValueError, "cannot be conditioned"):
        _cfg(dflash_config={**CHECKPOINT_CONFIG["dflash_config"], "target_layer_ids": []})


def test_gqa_ratio_must_divide(expect_error):
    with expect_error(ValueError, "not a multiple of"):
        _cfg(num_key_value_heads=7)


def test_block_size_must_leave_room_to_draft(expect_error):
    with expect_error(ValueError, "no room to draft"):
        _cfg(block_size=1)


def test_tap_count_need_not_equal_depth():
    """The Gemma4 drafter has 5 layers and 6 taps, so this must not be validated as equal.

    ``fc``'s input width comes from the tap count alone.
    """
    cfg = _cfg(dflash_config={**CHECKPOINT_CONFIG["dflash_config"], "target_layer_ids": [1, 12, 23, 35, 46, 57]})
    assert cfg.num_hidden_layers == 5
    assert cfg.target_feature_size == 6 * cfg.hidden_size


# ---- target pairing ------------------------------------------------------------------


def test_assert_matches_target(drafter_cfg, expect_error):
    drafter_cfg.assert_matches_target(TARGET_NUM_HIDDEN_LAYERS)  # 64: the real pairing
    with expect_error(ValueError, "trained against a 64-layer target"):
        drafter_cfg.assert_matches_target(32)  # e.g. accidentally paired with the 9B


# ---- drift guard ---------------------------------------------------------------------


def test_literal_matches_real_checkpoint():
    """The hermetic literal in ``conftest`` still matches the published ``config.json``.

    This is the only test here that needs the network. Everything else runs offline
    against the literal, which is only safe because this test exists.
    """
    try:
        real = DFlashDrafterConfig.from_pretrained()
    except Exception as exc:  # unreachable checkpoint, offline CI, no HF token
        pytest.skip(f"drafter checkpoint config unavailable: {type(exc).__name__}: {exc}")

    assert real == DFlashDrafterConfig.from_dict(CHECKPOINT_CONFIG)
