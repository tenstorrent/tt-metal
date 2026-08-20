# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Gates for ``tt/precision.py`` -- the precision config stage 07 sweeps.

The stage-07 goal asks for a selected precision config that "later
full-model/vLLM construction paths actually consume by default", and says
explicitly that **a JSON field ignored by hard-coded model code does not satisfy
this requirement**. So this file is arranged around that sentence:

* :func:`test_default_is_the_shipped_policy` and the alias tests pin that the
  *default* changed nothing -- every constant stages 02-06 measured at is still
  what a no-argument construction produces;
* :func:`test_non_default_precision_reaches_the_device` builds a real model on
  the mesh at a **non-default** value and asserts on what the device actually
  holds: a different weight dtype, a different program-config block width, a
  different compute-kernel fidelity, and a smaller per-die expert allocation.
  That is the assertion the goal is really asking for;
* the round-trip tests pin that config -> JSON -> config is lossless and that
  the JSON carries every field.

The host-only tests need no device. The two device tests build **two-layer**
models -- the observable is per-layer and a 48-layer load is three minutes --
on a module-scoped mesh, exactly as ``test_full_model.py`` does.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

import ttnn

# **Absolute imports, deliberately.** ``tt/generator.py`` imports ``tt.model``
# by absolute path, and under this repo's ``--import-mode=importlib`` a relative
# ``from ..tt import model`` here resolves to a *second* copy of the module (no
# ``models/__init__.py``, so pytest roots the package at this directory).
# The identity assertions below would then be comparing two different classes,
# and the device tests would be inspecting a model built by the other copy.
from models.demos.blackhole.qwen3_coder_30b_a3b.tt import model as M
from models.demos.blackhole.qwen3_coder_30b_a3b.tt import multichip_decoder as MC
from models.demos.blackhole.qwen3_coder_30b_a3b.tt import optimized_decoder as O
from models.demos.blackhole.qwen3_coder_30b_a3b.tt.generator import build_generator
from models.demos.blackhole.qwen3_coder_30b_a3b.tt.model import DEFAULT_TRACE_REGION_SIZE
from models.demos.blackhole.qwen3_coder_30b_a3b.tt.precision import DEFAULT_PRECISION, PrecisionConfig

MODEL_DIR = "models/demos/blackhole/qwen3_coder_30b_a3b"

#: Every field the stage-07 goal enumerates, mapped to the config field(s) that
#: carry it. The test below fails if any of these stops being serialised, which
#: is the cheap way to notice a field being dropped from the artifact.
GOAL_FIELDS = {
    "experts gate_up weight dtype": ["experts_gate_up_dtype"],
    "experts down weight dtype": ["experts_down_dtype"],
    "attention qkv weight dtype": ["attention_qkv_dtype"],
    "attention wo weight dtype": ["attention_wo_dtype"],
    "lm_head weight dtype": ["lm_head_dtype"],
    "router weight dtype": ["router_dtype"],
    "embedding weight dtype": ["embedding_dtype"],
    "per-group compute fidelity": [
        "experts_fidelity",
        "attention_fidelity",
        "router_window_fidelity",
        "lm_head_fidelity",
        "norm_fidelity",
    ],
    "activation/residual dtype": ["activation_dtype"],
    "CCL dtype": ["ccl_dtype"],
    "KV-cache dtype": ["kv_cache_dtype"],
    "logits/sampling dtype": ["logits_dtype", "sampling_dtype"],
}


# -- the default is the shipped policy ----------------------------------------


def test_default_is_the_shipped_policy():
    """The literal values stages 02-06 measured at, re-asserted here.

    Written out rather than compared against the module constants, which are now
    *derived* from this config -- comparing them to each other would be a
    tautology. If a shipped value is ever changed, this test is the thing that
    has to be changed with it, deliberately.
    """
    p = DEFAULT_PRECISION
    assert p.experts_gate_up_dtype is ttnn.bfloat4_b
    assert p.experts_down_dtype is ttnn.bfloat4_b
    # Stage 07: retuned to the full-K ceilings the 48-layer sweep measured.
    assert p.experts_gate_up_in0_block_w == 64
    assert p.experts_down_in0_block_w == 24
    assert p.experts_fidelity is ttnn.MathFidelity.LoFi
    assert p.attention_qkv_dtype is ttnn.bfloat8_b
    assert p.attention_wo_dtype is ttnn.bfloat8_b
    assert p.attention_fidelity is None, "the projections take the op default; see _attention_compute_kernel_config"
    assert p.lm_head_dtype is ttnn.bfloat8_b
    assert p.lm_head_fidelity is ttnn.MathFidelity.HiFi2
    assert p.router_dtype is ttnn.bfloat16
    assert p.router_window_fidelity is ttnn.MathFidelity.HiFi4, "the one-hot window matmul must select, not approximate"
    assert p.embedding_dtype is ttnn.bfloat16
    assert p.norm_weight_dtype is ttnn.bfloat16
    assert p.norm_fidelity is ttnn.MathFidelity.HiFi4
    assert p.activation_dtype is ttnn.bfloat16
    assert p.ccl_dtype is None and p.effective_ccl_dtype is ttnn.bfloat16
    assert p.kv_cache_dtype is ttnn.bfloat16
    assert p.logits_dtype is ttnn.bfloat16
    assert p.sampling_dtype is ttnn.bfloat16


def test_module_constants_still_resolve_to_the_default():
    """The stage-02/04 names are aliases now; they must still read the same.

    Probes under ``doc/`` and several stage-02 tests import these, and the point
    of keeping them was that nothing outside this file had to change.
    """
    assert O.EXPERT_WEIGHT_DTYPE is DEFAULT_PRECISION.experts_gate_up_dtype
    assert O.EXPERT_IN0_BLOCK_W_GATE_UP == DEFAULT_PRECISION.experts_gate_up_in0_block_w
    assert O.EXPERT_IN0_BLOCK_W_DOWN == DEFAULT_PRECISION.experts_down_in0_block_w
    assert O.EXPERT_MATH_FIDELITY is DEFAULT_PRECISION.experts_fidelity
    assert O.ATTENTION_WEIGHT_DTYPE is DEFAULT_PRECISION.attention_qkv_dtype
    assert M.LM_HEAD_WEIGHT_DTYPE is DEFAULT_PRECISION.lm_head_dtype
    assert M.EMBED_WEIGHT_DTYPE is DEFAULT_PRECISION.embedding_dtype


def test_default_construction_paths_all_resolve_to_the_same_object():
    """``None`` means the shipped policy on every entry point that takes one."""
    assert M._resolve_precision(None) is DEFAULT_PRECISION
    assert M._resolve_precision(DEFAULT_PRECISION) is DEFAULT_PRECISION
    assert M._resolve_precision(DEFAULT_PRECISION.to_dict()) == DEFAULT_PRECISION


# -- serialisation -------------------------------------------------------------


def test_json_round_trip_is_lossless():
    for config in (
        DEFAULT_PRECISION,
        DEFAULT_PRECISION.with_overrides(experts_gate_up_dtype="bfloat8_b", experts_gate_up_in0_block_w=32),
        DEFAULT_PRECISION.with_overrides(attention_fidelity="HiFi4", ccl_dtype="bfloat8_b"),
    ):
        assert PrecisionConfig.from_json(config.to_json()) == config
        # and again through a second hop, so an asymmetric coercion cannot hide
        assert PrecisionConfig.from_json(PrecisionConfig.from_json(config.to_json()).to_json()) == config


def test_json_carries_every_field_the_goal_lists():
    payload = json.loads(DEFAULT_PRECISION.to_json())
    declared = {f.name for f in dataclasses.fields(PrecisionConfig)}
    assert set(payload) == declared, "to_dict() must emit exactly the dataclass fields"
    for description, names in GOAL_FIELDS.items():
        for name in names:
            assert name in payload, f"{description} is missing from the serialised config"


def test_json_is_plain_names_not_repr():
    """The artifact has to be readable and diffable, not ``DataType.BFLOAT4_B``."""
    payload = json.loads(DEFAULT_PRECISION.to_json())
    assert payload["experts_gate_up_dtype"] == "bfloat4_b"
    assert payload["experts_fidelity"] == "LoFi"
    assert payload["attention_fidelity"] is None
    assert payload["ccl_dtype"] is None
    assert payload["experts_gate_up_in0_block_w"] == 64


def test_write_and_read_json_file(tmp_path):
    config = DEFAULT_PRECISION.with_overrides(lm_head_dtype="bfloat16")
    path = config.write_json(tmp_path / "nested" / "selected_precision_config.json")
    assert PrecisionConfig.read_json(path) == config
    # the file form is what a construction path is handed
    assert M._resolve_precision(str(path)) == config


def test_unknown_names_are_rejected(expect_error):
    with expect_error(ValueError, "unknown dtype"):
        PrecisionConfig(experts_gate_up_dtype="bfloat3_b")
    with expect_error(ValueError, "unknown math fidelity"):
        PrecisionConfig(experts_fidelity="HiFi9")
    with expect_error(ValueError, "unknown precision fields"):
        PrecisionConfig.from_dict({**DEFAULT_PRECISION.to_dict(), "expert_dtype": "bfloat8_b"})
    with expect_error(ValueError, "unknown precision fields"):
        DEFAULT_PRECISION.with_overrides(expert_weight_dtype="bfloat8_b")
    with expect_error(ValueError, "may not be None"):
        PrecisionConfig(activation_dtype=None)


def test_config_is_frozen(expect_error):
    # the fixture requires a match string; frozen dataclasses name the field
    with expect_error(dataclasses.FrozenInstanceError, "experts_gate_up_dtype"):
        DEFAULT_PRECISION.experts_gate_up_dtype = ttnn.bfloat16


# -- the config is actually consumed (device) ----------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


#: A precision that differs from the shipped one in four independently
#: observable ways: a wider expert weight, a different block width for it, a
#: different expert fidelity, and a narrower lm_head.
NON_DEFAULT = DEFAULT_PRECISION.with_overrides(
    experts_gate_up_dtype="bfloat8_b",
    experts_gate_up_in0_block_w=32,  # also a divisor of 2048/32 = 64
    experts_fidelity="HiFi4",
    lm_head_dtype="bfloat4_b",
    attention_wo_dtype="bfloat16",
)


def _build(mesh_device, precision):
    return build_generator(
        MODEL_DIR,
        mesh_device,
        override_num_layers=2,
        max_context_len=1024,
        max_batch_size=1,
        precision=precision,
    )


def test_non_default_precision_reaches_the_device(mesh_device):
    """Construct at ``NON_DEFAULT`` and assert on what the *device* holds.

    This is the goal's "a JSON field ignored by hard-coded model code does not
    satisfy this requirement" test. Nothing here reads ``model.precision``: every
    assertion is against a dtype read back off an uploaded tensor, a block width
    resolved by ``_tuned_sparse_matmul_config``, or a byte count computed from
    the allocated shape.
    """
    default_gen = _build(mesh_device, None)
    try:
        base = default_gen.model.runtime_fallback_audit()
        base_lm_head = str(default_gen.model.lm_head.dtype)
    finally:
        default_gen.teardown()

    gen = _build(mesh_device, NON_DEFAULT)
    try:
        audit = gen.model.runtime_fallback_audit()
        lm_head_dtype = str(gen.model.lm_head.dtype)

        # 1. a different weight dtype reached the device
        assert base["device_experts_gate_up_dtype"] == str(ttnn.bfloat4_b)
        assert audit["device_experts_gate_up_dtype"] == str(ttnn.bfloat8_b)
        assert audit["device_attention_wo_dtype"] == str(ttnn.bfloat16)
        assert base["device_attention_wo_dtype"] == str(ttnn.bfloat8_b)
        assert base_lm_head == str(ttnn.bfloat8_b)
        assert lm_head_dtype == str(ttnn.bfloat4_b)

        # 2. gate/up moved but down did **not** -- the two expert groups are
        #    genuinely separate fields, not one knob wearing two names
        assert audit["device_experts_down_dtype"] == str(ttnn.bfloat4_b)

        # 3. a different block width in the resolved program config
        assert base["gate_up_in0_block_w"] == 64
        assert audit["gate_up_in0_block_w"] == 32
        assert audit["down_in0_block_w"] == 24, "down's width was not overridden and must not move"

        # 4. a different fidelity in the compute kernel config
        assert base["expert_math_fidelity"] == str(ttnn.MathFidelity.LoFi)
        assert audit["expert_math_fidelity"] == str(ttnn.MathFidelity.HiFi4)

        # 5. an allocation-size change: bfloat4_b -> bfloat8_b on gate/up
        #    roughly doubles the gate/up half of the per-die expert footprint
        assert audit["device_expert_bytes_per_die"] > base["device_expert_bytes_per_die"]
        grew = audit["device_expert_bytes_per_die"] - base["device_expert_bytes_per_die"]
        assert grew > 20 * 1024 * 1024, f"expected tens of MB per die, got {grew}"

        # 6. and it still runs -- a config that reaches the device but wedges it
        #    would not be sweepable
        ids = gen.tokenizer("def fib(n):", add_special_tokens=False)["input_ids"]
        out = gen.generate(ids, 4, enable_trace=True, sampling_mode="device", top_k=1)
        assert len(out) == 4
    finally:
        gen.teardown()


def test_default_construction_audit_matches_the_shipped_values(mesh_device):
    """The default path puts the shipped dtypes and widths on the device."""
    gen = _build(mesh_device, None)
    try:
        audit = gen.model.runtime_fallback_audit()
        assert audit["device_experts_gate_up_dtype"] == str(ttnn.bfloat4_b)
        assert audit["device_experts_down_dtype"] == str(ttnn.bfloat4_b)
        assert audit["device_attention_qkv_dtype"] == str(ttnn.bfloat8_b)
        assert audit["device_attention_wo_dtype"] == str(ttnn.bfloat8_b)
        assert audit["device_attention_qkv_decode_dtype"] == str(ttnn.bfloat8_b)
        assert audit["device_router_dtype"] == str(ttnn.bfloat16)
        assert audit["device_norm_weight_dtype"] == str(ttnn.bfloat16)
        assert audit["gate_up_in0_block_w"] == 64
        assert audit["down_in0_block_w"] == 24
        assert audit["expert_math_fidelity"] == str(ttnn.MathFidelity.LoFi)
        assert audit["attention_math_fidelity"] is None
        assert audit["router_window_math_fidelity"] == str(ttnn.MathFidelity.HiFi4)
        assert audit["ccl_dtype"] == str(ttnn.bfloat16)
        assert audit["activation_dtype"] == str(ttnn.bfloat16)
        assert str(gen.model.ensure_internal_kv_cache()[0].k.dtype) == str(ttnn.bfloat16)
        # the audit also reports the config itself, which is what a sweep row
        # records alongside its measurement
        assert audit["precision"] == DEFAULT_PRECISION.to_dict()
    finally:
        gen.teardown()


#: The four fields stage 07's original selection proof could not see.
#:
#: They had **no audit entry at all**, which made "this lever does nothing" and
#: "this lever is not wired up" indistinguishable -- three sweep rows produced
#: ``device_audit`` blocks byte-identical to the baseline's. For
#: ``norm_fidelity`` it was the second: ``decode_residual_norm`` built its
#: compute config from the module default and never saw ``self.precision``, so
#: the field was a documented knob with no effect and ``R21_norm_hifi2``
#: measured nothing. This config moves all four away from their defaults.
TERMINAL_AND_NORM = DEFAULT_PRECISION.with_overrides(
    norm_fidelity="HiFi2",
    lm_head_fidelity="LoFi",
    logits_dtype="bfloat8_b",
    sampling_dtype="bfloat8_b",
)


def test_fidelity_and_terminal_dtypes_reach_the_device(mesh_device):
    """The four fields that used to change nothing observable must change it now.

    Regression test for a dead config field. Every assertion is against
    something the *device* or the *ops* hold: the fidelities come off the
    ``compute_kernel_config`` objects the norm and lm_head are handed, the two
    dtypes off the tensors the terminal path actually produced during a real
    traced decode. A default-vs-override diff is asserted for each, so a field
    that silently stops being threaded fails here rather than in a sweep row
    three stages later.
    """
    gen = _build(mesh_device, None)
    try:
        ids = gen.tokenizer("def fib(n):", add_special_tokens=False)["input_ids"]
        gen.generate(ids, 4, enable_trace=True, sampling_mode="device", top_k=1)
        base = gen.model.runtime_fallback_audit()
    finally:
        gen.teardown()

    assert base["norm_math_fidelity"] == str(ttnn.MathFidelity.HiFi4)
    assert base["lm_head_math_fidelity"] == str(ttnn.MathFidelity.HiFi2)
    assert base["logits_dtype_observed"] == "bfloat16"
    assert base["sampling_dtype_observed"] == "bfloat16"
    assert base["terminal_dtype_source"] == "device_readback"

    gen = _build(mesh_device, TERMINAL_AND_NORM)
    try:
        ids = gen.tokenizer("def fib(n):", add_special_tokens=False)["input_ids"]
        out = gen.generate(ids, 4, enable_trace=True, sampling_mode="device", top_k=1)
        assert len(out) == 4, "a config that reaches the device but wedges it is not sweepable"
        audit = gen.model.runtime_fallback_audit()
    finally:
        gen.teardown()

    # norm_fidelity: the field that was NOT threaded. It reaches only the decode
    # residual norms -- the prefill norms pass no compute config at all -- which
    # is why the audit name is norm_math_fidelity rather than something global.
    assert audit["norm_math_fidelity"] == str(ttnn.MathFidelity.HiFi2)
    assert audit["lm_head_math_fidelity"] == str(ttnn.MathFidelity.LoFi)
    assert audit["logits_dtype_observed"] == "bfloat8_b"
    assert audit["sampling_dtype_observed"] == "bfloat8_b"
