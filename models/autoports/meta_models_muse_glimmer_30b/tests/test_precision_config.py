# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The selected precision policy, and the guarantee that the build consumes it.

``$datatype-sweep`` accepts a selected precision configuration only if the
runtime construction path actually reads every field.  A JSON file the model
ignores is worse than no file, because it reads like evidence.  So these tests
come in two halves:

* **host-only** -- the artifact schema, its rejections, and the policy algebra
  (per-role math fidelity, layer exceptions).  No device.
* **device** -- a reduced two-layer build from an artifact, whose realised
  precision is read back off the device tensors and compared field by field.
  The negative controls matter as much as the positive one: a build asked for
  BFP4 attention weights must come back BFP4, *and* a build asked for HiFi2 on
  one role must come back with the other roles unchanged.

Run::

    pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_precision_config.py
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
from dataclasses import replace

import pytest

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt import precision_config as pc
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.model import (
    LM_HEAD_CORES,
    LM_HEAD_DTYPE,
    LM_HEAD_FIDELITY,
    LM_HEAD_FP32_ACC,
    LM_HEAD_IN0_BLOCK_W,
    LM_HEAD_MATMUL,
    LM_HEAD_OUTPUT_DTYPE,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_DECODE_CCL_DTYPE,
    DEFAULT_PREFILL_CCL_DTYPE,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import DEFAULT_PRECISION, PROJECTION_ROLES

MODEL_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = MODEL_DIR / "doc/datatype_sweep/configs"
#: One sliding-window layer and one full-attention layer, as in the full-model tests.
REDUCED_LAYERS = (0, 3)
REDUCED_MAX_SEQ = 4096
ATTN_ROLES = ("wqkv", "attn_gate", "o_proj")


def _artifact(**overrides) -> dict:
    """The shipped policy as an artifact, with ``config_from_policy`` overrides."""
    kwargs = {
        "config_id": "test-policy",
        "description": "unit test",
        "policy": DEFAULT_PRECISION,
        "prefill_ccl_dtype": DEFAULT_PREFILL_CCL_DTYPE,
        "decode_ccl_dtype": DEFAULT_DECODE_CCL_DTYPE,
        "lm_head_dtype": LM_HEAD_DTYPE,
        "lm_head_fidelity": LM_HEAD_FIDELITY,
        "lm_head_fp32_acc": LM_HEAD_FP32_ACC,
        "lm_head_output_dtype": LM_HEAD_OUTPUT_DTYPE,
        "lm_head_matmul": LM_HEAD_MATMUL,
        "lm_head_cores": LM_HEAD_CORES,
        "lm_head_in0_block_w": LM_HEAD_IN0_BLOCK_W,
    }
    kwargs.update(overrides)
    return pc.config_from_policy(**kwargs)


# ------------------------------------------------------------------ host only


def test_the_selected_artifact_exists_and_loads():
    """The build path treats it as required, so a missing file is a broken port."""
    config = pc.load_precision_config()
    assert config["schema_version"] == pc.SCHEMA_VERSION
    kwargs = pc.build_kwargs_from_config(config)
    assert isinstance(kwargs["decoder_kwargs"]["precision"], type(DEFAULT_PRECISION))


def test_every_candidate_artifact_round_trips():
    for path in sorted(CONFIG_DIR.glob("*.json")):
        config = pc.load_precision_config(path)
        assert config["config_id"] == path.stem
        pc.build_kwargs_from_config(config)


def test_the_artifact_round_trips_through_the_policy():
    policy = replace(
        DEFAULT_PRECISION,
        name="rt",
        attn_weight_dtype=ttnn.bfloat4_b,
        decode_math_fidelity_by_role=(("wqkv", ttnn.MathFidelity.HiFi2),),
        layer_exceptions=(((0, 51), (("attn_weight_dtype", ttnn.bfloat8_b),)),),
    )
    config = _artifact(config_id="rt", policy=policy)
    back = pc.precision_policy_from_config(config)
    assert back.attn_weight_dtype is ttnn.bfloat4_b
    assert back.decode_fidelity("wqkv") is ttnn.MathFidelity.HiFi2
    assert back.decode_fidelity("mlp_gate") is ttnn.MathFidelity.LoFi
    assert back.for_layer(0).attn_weight_dtype is ttnn.bfloat8_b
    assert back.for_layer(51).attn_weight_dtype is ttnn.bfloat8_b
    assert back.for_layer(1).attn_weight_dtype is ttnn.bfloat4_b


def test_for_layer_is_idempotent_and_drops_its_exceptions():
    """Applying a resolved policy twice must not re-apply an exception."""
    policy = replace(
        DEFAULT_PRECISION,
        attn_weight_dtype=ttnn.bfloat4_b,
        layer_exceptions=(((0,), (("attn_weight_dtype", ttnn.bfloat8_b),)),),
    )
    once = policy.for_layer(0)
    assert once.layer_exceptions == ()
    assert once.for_layer(0) is once


def test_an_unlisted_layer_gets_the_same_policy_object():
    """No exception means no allocation and no cache-key churn."""
    policy = replace(DEFAULT_PRECISION, layer_exceptions=(((7,), (("attn_weight_dtype", ttnn.bfloat4_b),)),))
    assert policy.for_layer(3) is policy
    assert policy.for_layer(7) is not policy


def test_per_role_fidelity_defaults_to_the_scalar_and_rejects_unknown_roles(expect_error):
    policy = replace(DEFAULT_PRECISION, decode_math_fidelity_by_role=(("mlp_down", ttnn.MathFidelity.HiFi4),))
    assert policy.decode_fidelity("mlp_down") is ttnn.MathFidelity.HiFi4
    for role in PROJECTION_ROLES:
        if role != "mlp_down":
            assert policy.decode_fidelity(role) is DEFAULT_PRECISION.decode_math_fidelity
    with expect_error(KeyError, "unknown projection role"):
        policy.decode_fidelity("not_a_role")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.__setitem__("schema_version", 99), "schema_version"),
        (lambda c: c.pop("kv_cache"), "kv_cache"),
        (lambda c: c["weights"]["mlp_down"].__setitem__("dtype", "int8"), "unknown dtype"),
        (lambda c: c["compute_fidelity"]["decode"].__setitem__("default", "HiFi9"), "unknown math fidelity"),
        (
            lambda c: c["compute_fidelity"]["decode"].__setitem__("by_role", {"not_a_role": "LoFi"}),
            "unknown role",
        ),
        (lambda c: c["activations"].__setitem__("residual_dtype", "float32"), "residual_dtype"),
        (lambda c: c["weights"]["embedding"].__setitem__("dtype", "bfloat8_b"), "embedding"),
        (lambda c: c["weights"]["norms"].__setitem__("dtype", "float32"), "norms"),
        (lambda c: c["logits"].__setitem__("sampling_input_dtype", "float32"), "sampling_input_dtype"),
    ],
)
def test_a_malformed_artifact_is_rejected_rather_than_defaulted(tmp_path, expect_error, mutate, message):
    """Every rejection here is a field the build would otherwise silently ignore."""
    config = _artifact()
    mutate(config)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with expect_error(pc.PrecisionConfigError, re.escape(message)):
        pc.build_kwargs_from_config(pc.load_precision_config(path))


def test_a_missing_artifact_names_itself_as_required(tmp_path, expect_error):
    with expect_error(pc.PrecisionConfigError, "required build input"):
        pc.load_precision_config(tmp_path / "absent.json")


def test_check_propagation_catches_a_field_the_build_ignored():
    """The propagation check is only worth running if it can fail."""
    config = _artifact(policy=replace(DEFAULT_PRECISION, attn_weight_dtype=ttnn.bfloat4_b))
    realised = {
        "num_layers": 1,
        "layer_groups": [
            {
                "layers": [0],
                "precision": {
                    "policy_name": "test-policy",
                    "activation_dtype": "DataType.BFLOAT16",
                    "kv_cache_dtype": "DataType.BFLOAT8_B",
                    "kv_cache_dtype_requested": "DataType.BFLOAT8_B",
                    "roles": {
                        role: {
                            # The build ignored the BFP4 request for the attention
                            # projections and packed BFP8 instead.
                            "weight_dtype": "DataType.BFLOAT8_B" if role in ATTN_ROLES else "DataType.BFLOAT4_B",
                            "decode_fidelity": "MathFidelity.LoFi",
                            "prefill_fidelity": "MathFidelity.LoFi",
                            "decode_cores": 16,
                            "decode_in0_block_w": 2,
                        }
                        for role in PROJECTION_ROLES
                    },
                    "ccl": {
                        "prefill_payload_dtype": "DataType.BFLOAT8_B",
                        "decode_payload_dtype": "DataType.BFLOAT16",
                    },
                },
            }
        ],
        "embedding": {"weight_dtype": "DataType.BFLOAT16"},
        "lm_head": {
            "weight_dtype": "DataType.BFLOAT4_B",
            "fidelity": "MathFidelity.LoFi",
            "fp32_dest_acc_en": False,
            "output_dtype": "DataType.BFLOAT16",
            "matmul": LM_HEAD_MATMUL,
            "cores": LM_HEAD_CORES,
            "in0_block_w": LM_HEAD_IN0_BLOCK_W,
        },
        "terminal_norms": {
            "embed_norm_weight_dtype": "DataType.BFLOAT16",
            "final_norm_weight_dtype": "DataType.BFLOAT16",
        },
        "logits": {
            "logits_dtype": "DataType.BFLOAT16",
            "sampling_input_dtype": "DataType.BFLOAT16",
            "sampling_implementation": "models.common.sampling.generator.SamplingGenerator",
        },
    }
    problems = pc.check_propagation(config, realised)
    assert len(problems) == len(ATTN_ROLES)
    assert all("requested bfloat4_b, built bfloat8_b" in p for p in problems)

    # ... and passes once the build agrees.
    agreed = copy.deepcopy(realised)
    for role in ATTN_ROLES:
        agreed["layer_groups"][0]["precision"]["roles"][role]["weight_dtype"] = "DataType.BFLOAT4_B"
    assert pc.check_propagation(config, agreed) == []


# --------------------------------------------------------------------- device


@pytest.fixture(scope="module")
def mesh():
    device = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    yield device
    clear_generator_cache()
    close_multichip_mesh(device)


def _build(mesh, config_path):
    return build_generator(
        MODEL_DIR,
        mesh,
        max_seq_len=REDUCED_MAX_SEQ,
        max_batch_size=1,
        layer_indices=REDUCED_LAYERS,
        precision_config=config_path,
        reuse=False,
    )


@pytest.mark.timeout(900)
def test_the_shipped_artifact_is_the_policy_the_build_runs(mesh):
    """The headline guarantee: no caller passes a dtype and the build has them all."""
    generator = _build(mesh, pc.SELECTED_PRECISION_CONFIG_PATH)
    try:
        config = pc.load_precision_config()
        realised = generator.capability_report()["precision_policy"]
        assert pc.check_propagation(config, realised) == []
        assert realised["selected_config_id"] == config["config_id"]
    finally:
        generator.teardown()


@pytest.mark.timeout(900)
def test_a_per_role_fidelity_request_reaches_only_that_role(mesh, tmp_path):
    """Per-role fidelity is the field a single scalar knob used to flatten."""
    config = _artifact(
        config_id="attn-hifi2",
        policy=replace(
            DEFAULT_PRECISION,
            decode_math_fidelity_by_role=tuple((r, ttnn.MathFidelity.HiFi2) for r in ATTN_ROLES),
        ),
    )
    path = tmp_path / "attn-hifi2.json"
    path.write_text(json.dumps(config))
    generator = _build(mesh, path)
    try:
        assert pc.check_propagation(config, generator.capability_report()["precision_policy"]) == []
        roles = generator.model.layers[0].precision_report()["roles"]
        for role in ATTN_ROLES:
            assert roles[role]["decode_fidelity"] == "MathFidelity.HiFi2"
            # Prefill was not asked to move, and must not have.
            assert roles[role]["prefill_fidelity"] == "MathFidelity.LoFi"
        for role in ("mlp_gate", "mlp_up", "mlp_down"):
            assert roles[role]["decode_fidelity"] == "MathFidelity.LoFi"
    finally:
        generator.teardown()


@pytest.mark.timeout(900)
def test_a_layer_exception_reaches_only_the_layers_it_names(mesh, tmp_path):
    """Layer 0 is excepted, layer 3 is not; both are in the reduced build."""
    config = _artifact(
        config_id="attn4-except-layer0",
        policy=replace(
            DEFAULT_PRECISION,
            attn_weight_dtype=ttnn.bfloat4_b,
            layer_exceptions=(((0,), (("attn_weight_dtype", ttnn.bfloat8_b),)),),
        ),
    )
    path = tmp_path / "except.json"
    path.write_text(json.dumps(config))
    generator = _build(mesh, path)
    try:
        assert pc.check_propagation(config, generator.capability_report()["precision_policy"]) == []
        first, second = (layer.precision_report() for layer in generator.model.layers)
        assert first["layer_idx"] == 0 and second["layer_idx"] == 3
        assert first["roles"]["wqkv"]["weight_dtype"] == "DataType.BFLOAT8_B"
        assert second["roles"]["wqkv"]["weight_dtype"] == "DataType.BFLOAT4_B"
        # The MLP is not part of the exception and must be BFP4 in both.
        assert first["roles"]["mlp_down"]["weight_dtype"] == "DataType.BFLOAT4_B"
        assert second["roles"]["mlp_down"]["weight_dtype"] == "DataType.BFLOAT4_B"
    finally:
        generator.teardown()


@pytest.mark.timeout(900)
def test_the_readiness_factory_path_builds_the_selected_config(mesh):
    """The propagation proof for every *downstream* consumer.

    The readiness runners -- and, by the same contract, the vLLM adapter -- do not
    import ``tt.generator``; they load ``<model_dir>/tt/generator.py`` **by path**
    under a synthetic module name and call its ``build_generator(model_dir,
    mesh_device)`` with no knobs (``models/common/readiness_check/contract.py``).
    That is a different module object from the one the rest of this file imports,
    so "the selected config reaches build_generator" has to be shown through that
    path specifically, not through the convenient one.
    """
    import importlib.util
    import sys

    from models.common.readiness_check.contract import BUILD_GENERATOR_FUNCTION_NAME, GENERATOR_MODULE_RELPATH

    name = "_precision_test_readiness_generator"
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / GENERATOR_MODULE_RELPATH)
    module = importlib.util.module_from_spec(spec)
    # Register before executing, exactly as the runner does: ``GeneratorConfig``
    # is a dataclass with string annotations, and ``@dataclass`` resolves them
    # through ``sys.modules[cls.__module__]``.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    factory = getattr(module, BUILD_GENERATOR_FUNCTION_NAME)

    config = pc.load_precision_config()
    generator = factory(
        MODEL_DIR, mesh, max_seq_len=REDUCED_MAX_SEQ, max_batch_size=1, layer_indices=REDUCED_LAYERS, reuse=False
    )
    try:
        realised = generator.capability_report()["precision_policy"]
        assert realised["selected_config_id"] == config["config_id"]
        assert pc.check_propagation(config, realised) == []
    finally:
        generator.teardown()


@pytest.mark.timeout(900)
def test_a_caller_override_is_recorded_rather_than_silently_applied(mesh):
    """An evidence file must never say "selected policy" about an overridden build."""
    generator = build_generator(
        MODEL_DIR,
        mesh,
        max_seq_len=REDUCED_MAX_SEQ,
        max_batch_size=1,
        layer_indices=REDUCED_LAYERS,
        decoder_kwargs={"kv_cache_dtype": ttnn.bfloat16},
        reuse=False,
    )
    try:
        report = generator.capability_report()
        assert "override(kv_cache_dtype)" in report["precision_policy"]["selected_config_id"]
        assert report["precision_policy"]["layer_groups"][0]["precision"]["kv_cache_dtype"] == "DataType.BFLOAT16"
    finally:
        generator.teardown()
