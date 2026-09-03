# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-MESH` — `MeshConfig` arithmetic, its refusals, and `llama_hf_config` normalisation.

`MeshConfig` decides how every weight is sharded and how big every per-chip tensor is, so its
arithmetic is worth a device-free test of its own: a wrong `shard_size` shows up much later as a
matmul shape error inside a module, where it looks like a module bug.

Three things are asserted, and each one is load-bearing:

1. **The arithmetic** — `sp`, `tp`, `shard_size(4096) == 512`, `shard_size(14336) == 1792` at the
   deployment target `(4,8)`/TP=8 (`DEC-002`), and the `(1,1)`/TP=1 single-card shape P5 runs on.
2. **The refusal** — `MeshConfig((1,8), tp=4)` must **raise**. `shard_mapper` always shards the
   *entire* TP axis, so a sub-axis TP would build head counts from `tp` while the mapper still
   splits across all 8 devices. `models/demos/minimax_m3/config.py:40` only `logger.warning`s this,
   which would make the gate unfailable; the strict `_validate` is taken from
   `models/demos/gpt_oss_d_p/tt/config.py:38` (raise at `:45`) — `DEC-019`.
3. **`llama_hf_config`** — the normaliser accepts both a dict and a `to_dict()`-able object, agrees
   between them, and refuses a config whose RoPE parameters the repo helper would silently ignore
   (`DEC-010`, `BRINGUP_RECIPE.md` Appendix F.2).

`CCLManager` construction and its semaphore counts live in `test_ccl_semaphores.py` (gate
`G-SEMAPHORE`), which needs a device; everything here is host-only.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_mesh_config.py -x -q
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from loguru import logger

from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, llama_config_dims
from models.demos.llama31_8b_d_p.tt.config import _VALIDATED_MESH_SHAPE, _VALIDATED_TP, MeshConfig
from models.demos.llama31_8b_d_p.tt.model_config import LlamaHFConfig, llama_hf_config

# The two widths every TP shard computation in this model reduces to (03_OUTLINE.md section 3).
HIDDEN = 4096
INTERMEDIATE = 14336


def test_validated_target_matches_dec_002():
    """The module constants are the `DEC-002` deployment target, not the template's."""
    assert (_VALIDATED_MESH_SHAPE, _VALIDATED_TP) == (TestFactory.TARGET_MESH_SHAPE, TestFactory.TARGET_TP)


@pytest.mark.parametrize(
    "mesh_shape, tp, expected_sp",
    [
        ((4, 8), 8, 4),  # the deployment target, DEC-002
        ((1, 1), 1, 1),  # the P5/P6 single-card shape
        ((1, 8), 8, 1),  # the G-TP-PARITY shape
        ((8, 4), 4, 8),  # the legal-but-untested fallback, DEC-002
    ],
    ids=["4x8_tp8", "1x1_tp1", "1x8_tp8", "8x4_tp4"],
)
def test_mesh_config_arithmetic(mesh_shape, tp, expected_sp):
    """`sp`, `tp` and both shard widths, for every mesh shape this bring-up uses."""
    mc = MeshConfig(mesh_shape, tp=tp)

    logger.info(
        f"[G-MESH] {mc!r} -> shard_size({HIDDEN})={mc.shard_size(HIDDEN)}, "
        f"shard_size({INTERMEDIATE})={mc.shard_size(INTERMEDIATE)}"
    )

    assert mc.tp == tp
    assert mc.sp == expected_sp
    assert mc.mesh_shape == tuple(mesh_shape)
    assert mc.total_devices == mesh_shape[0] * mesh_shape[1]
    assert mc.sp_axis == 0 and mc.tp_axis == 1
    assert mc.shard_size(HIDDEN) == HIDDEN // tp
    assert mc.shard_size(INTERMEDIATE) == INTERMEDIATE // tp
    # Both shards must be tile-aligned or every matmul on them is padded silently.
    assert mc.shard_size(HIDDEN) % 32 == 0
    assert mc.shard_size(INTERMEDIATE) % 32 == 0


def test_target_shard_sizes_are_the_gate_numbers():
    """The two exact numbers `G-MESH` names: 4096/8 = 512 and 14336/8 = 1792."""
    mc = MeshConfig((4, 8), tp=8)
    assert mc.sp == 4
    assert mc.tp == 8
    assert mc.shard_size(HIDDEN) == 512
    assert mc.shard_size(INTERMEDIATE) == 1792
    logger.info(f"[G-MESH] target {mc!r}: shard_size(4096)=512, shard_size(14336)=1792 OK")


@pytest.mark.parametrize(
    "mesh_shape, tp",
    [((1, 8), 4), ((4, 8), 4), ((4, 8), 16), ((1, 1), 2)],
    ids=["1x8_tp4", "4x8_tp4", "4x8_tp16", "1x1_tp2"],
)
def test_sub_axis_tp_is_refused(mesh_shape, tp, expect_error):
    """TP must span the whole TP axis; anything else raises rather than warns.

    `MeshConfig((1,8), tp=4)` is the case `G-MESH` names explicitly.
    """
    with expect_error(ValueError, "must equal mesh_"):
        MeshConfig(mesh_shape, tp=tp)
    logger.info(f"[G-MESH] MeshConfig({mesh_shape}, tp={tp}) raised ValueError as required")


def test_tp_axis_zero_flips_the_sp_axis():
    """`tp_axis=0` moves TP onto the rows and SP onto the cols — the only other legal wiring."""
    mc = MeshConfig((8, 4), tp=8, tp_axis=0)
    assert mc.tp_axis == 0 and mc.sp_axis == 1
    assert mc.tp == 8 and mc.sp == 4


def test_reduce_scatter_exists():
    """`reduce_scatter` is the one member gpt-oss lacks; `DEC-019` requires the union to carry it."""
    assert callable(getattr(MeshConfig, "reduce_scatter", None)), "MeshConfig.reduce_scatter is missing (DEC-019)"
    assert not hasattr(MeshConfig((1, 1), tp=1), "ep_axis"), "ep_axis should be dropped for Llama (DEC-022)"


# --------------------------------------------------------------------------------------
# llama_hf_config — the normaliser (DEC-009 / DEC-010)
# --------------------------------------------------------------------------------------
def test_llama_hf_config_from_dict():
    """Every field resolves from the bundled `config.json`, with the real Llama-3.1-8B values."""
    cfg = llama_hf_config(llama_config_dims())

    logger.info(
        f"[G-MESH] llama_hf_config(dict): theta={cfg.rope_theta}, factor={cfg.rope_scaling_factor}, "
        f"orig_ctx={cfg.rope_orig_context_len}, head_dim={cfg.head_dim}"
    )

    assert isinstance(cfg, LlamaHFConfig)
    assert cfg.hidden_size == 4096
    assert cfg.intermediate_size == 14336
    assert cfg.num_hidden_layers == 32
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.gqa_group_size == 4
    assert cfg.vocab_size == 128256
    assert cfg.max_position_embeddings == 131072
    assert cfg.rms_norm_eps == 1e-05
    assert cfg.tie_word_embeddings is False
    assert cfg.hidden_act == "silu"
    assert cfg.attention_bias is False
    assert cfg.mlp_bias is False
    # The trap: theta must be the checkpoint's 500000.0, never a 10000.0 default. Appendix F.2.
    assert cfg.rope_theta == 500000.0
    assert cfg.rope_type == "llama3"
    assert cfg.rope_scaling_factor == 8.0
    assert cfg.rope_orig_context_len == 8192
    assert cfg.rope_low_freq_factor == 1.0
    assert cfg.rope_high_freq_factor == 4.0


def test_llama_hf_config_from_transformers_object():
    """A `LlamaConfig` gives the identical object, though its `to_dict()` has NO `rope_theta` key.

    This is the whole point of `DEC-010`: on transformers 5.12.1 the dict layout differs
    (`rope_parameters` only), and only `get_rope_theta` reads both.
    """
    from transformers import LlamaConfig

    dims = llama_config_dims()
    hf = LlamaConfig(
        hidden_size=dims["hidden_size"],
        intermediate_size=dims["intermediate_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims["num_key_value_heads"],
        head_dim=dims["head_dim"],
        hidden_act=dims["hidden_act"],
        rms_norm_eps=dims["rms_norm_eps"],
        max_position_embeddings=dims["max_position_embeddings"],
        vocab_size=dims["vocab_size"],
        attention_bias=dims["attention_bias"],
        mlp_bias=dims["mlp_bias"],
        tie_word_embeddings=dims["tie_word_embeddings"],
        rope_theta=dims["rope_theta"],
        rope_scaling=dims["rope_scaling"],
    )
    as_dict = hf.to_dict()
    assert (
        "rope_theta" not in as_dict and "rope_scaling" not in as_dict
    ), "transformers layout changed; re-read Appendix F.2 before trusting this test"

    from_obj = llama_hf_config(hf)
    from_dict = llama_hf_config(dims)
    logger.info(
        f"[G-MESH] llama_hf_config(LlamaConfig).rope_theta = {from_obj.rope_theta} (to_dict has no rope_theta key)"
    )
    assert from_obj == from_dict, "dict and object normalisation disagree"


def test_llama_hf_config_rejects_bad_source(expect_error):
    with expect_error(TypeError, "dict or an object with to_dict"):
        llama_hf_config("Llama-3.1-8B-Instruct")


def test_llama_hf_config_rejects_missing_theta(expect_error):
    """A config with no resolvable theta must fail loudly, never fall back to a default."""
    dims = llama_config_dims()
    dims.pop("rope_theta")
    with expect_error(AssertionError, "rope_theta resolved to None"):
        llama_hf_config(dims)


@pytest.mark.parametrize("key, value", [("low_freq_factor", 2.0), ("high_freq_factor", 8.0)], ids=["low", "high"])
def test_llama_hf_config_rejects_unhandled_limb_factors(key, value, expect_error):
    """`compute_llama3_parameters` hard-codes 1 / 4 (`common.py:407-408`); anything else must raise.

    Silently accepting a different factor is exactly the class of bug that produces a plausible
    short-sequence PCC and a collapsed long-context one.
    """
    dims = llama_config_dims()
    dims["rope_scaling"] = dict(dims["rope_scaling"])
    dims["rope_scaling"][key] = value
    with expect_error(AssertionError, key):
        llama_hf_config(dims)


def test_llama_hf_config_is_frozen(expect_error):
    """Frozen, so one module cannot mutate a dimension another module already read."""
    cfg = llama_hf_config(llama_config_dims())
    with expect_error(FrozenInstanceError, "cannot assign to field"):
        cfg.hidden_size = 1  # type: ignore[misc]
