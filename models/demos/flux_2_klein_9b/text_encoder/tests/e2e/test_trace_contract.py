# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Command 3 -- the trace contract and the fully-on-device check.

Stages come from Source A: `architectures == ["Qwen3ForCausalLM"]` and no
encoder-decoder sub-config, so the phases are ["prefill", "decode"].

  * every stage exposes `<stage>_trace_setup / _trace_step / _trace_inputs /
    _trace_items` on the object `build_pipeline` returns;
  * each stage captures ONE host-op-free step, executes it, matches the eager
    reference by PCC, and RELEASES the trace before the next stage;
  * `host_op_selftest()` proves the model math fires zero host aten ops;
  * the `layers` knob actually caps the built depth (it is not inert).

Run:  ./python_env/bin/python -m pytest \
        models/demos/flux_2_klein_9b_text_encoder/tests/e2e/test_trace_contract.py -s
"""
from __future__ import annotations

import inspect
import os

import pytest
import torch

import ttnn
from models.demos.flux_2_klein_9b.text_encoder.tt import model_ref
from models.demos.flux_2_klein_9b.text_encoder.tt.pipeline import (
    PIPELINE_STAGES,
    Flux2Klein9BTextEncoderPipeline,
    build_pipeline,
)

TP = int(os.environ.get("TT_HW_PLANNER_SHARD_TP", "8"))
# Depth used by the on-device trace/host-op checks. Profiling is per-op, so a
# capped stack surfaces the same op set at a fraction of the cost; the FULL depth
# is what tests/e2e/test_e2e_pipeline.py gates on.
TRACE_TEST_LAYERS = int(os.environ.get("TT_TRACE_TEST_LAYERS", "4"))

_DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "trace_region_size": 90000000,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
}


def test_stage_list_matches_the_reference_config():
    """The stages are DERIVED from Source A, not hardcoded per model."""
    cfg = model_ref.load_hf_model(torch.float32).config
    assert cfg.architectures == ["Qwen3ForCausalLM"]
    assert not getattr(cfg, "is_encoder_decoder", False)
    assert PIPELINE_STAGES == ["prefill", "decode"], PIPELINE_STAGES


def test_contract_surface_is_complete():
    """Every stage has the four hooks, and the two zero-arg ones really are."""
    for stage in PIPELINE_STAGES:
        for suffix in ("trace_setup", "trace_step", "trace_inputs", "trace_items"):
            name = f"{stage}_{suffix}"
            assert hasattr(Flux2Klein9BTextEncoderPipeline, name), f"missing {name}"
        for suffix in ("trace_inputs", "trace_items"):
            sig = inspect.signature(getattr(Flux2Klein9BTextEncoderPipeline, f"{stage}_{suffix}"))
            params = [p for p in sig.parameters if p != "self"]
            assert not params, f"{stage}_{suffix} must be ZERO-ARG, got {params}"
        setup_sig = inspect.signature(getattr(Flux2Klein9BTextEncoderPipeline, f"{stage}_trace_setup"))
        assert [p for p in setup_sig.parameters if p != "self"] == ["inputs"]

    # The AR decode contract lives alongside the trace hooks.
    for name in ("decode_prefill", "decode_step"):
        assert hasattr(Flux2Klein9BTextEncoderPipeline, name), f"missing {name}"

    # The factory must CONSTRUCT, not run.
    sig = inspect.signature(build_pipeline)
    for expected in ("device", "model", "layers", "prefill_layers", "decode_layers"):
        assert expected in sig.parameters, f"build_pipeline is missing `{expected}`"
    assert any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ), "build_pipeline must accept and ignore demo kwargs"
    print(f"[trace] contract surface complete for stages {PIPELINE_STAGES}")


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [TP], indirect=True)
def test_trace_contract_and_on_device(mesh_device):
    pipeline = build_pipeline(mesh_device, layers=TRACE_TEST_LAYERS, text="ignored demo kwarg")

    # --- the factory returned the OBJECT, and the stack is discoverable
    assert isinstance(pipeline, Flux2Klein9BTextEncoderPipeline)
    assert isinstance(pipeline.layers, list) and pipeline.layers
    assert len({type(x) for x in pipeline.layers}) == 1, "the repeated block must be same-typed"
    assert all(hasattr(x, "__dict__") for x in pipeline.layers)
    assert pipeline.hf_model is not None and hasattr(
        pipeline.hf_model.model, "layers"
    ), "the HF reference must stay reachable as section ground truth"

    # --- SV-9: the knob is not inert -- depth capped, everything else intact
    assert len(pipeline.layers) == TRACE_TEST_LAYERS
    assert len(pipeline.hf_model.model.layers) == pipeline.config.num_hidden_layers
    for part in ("token_embed", "rotary_embedding", "encoder_stack", "decoder_head"):
        assert getattr(pipeline, part) is not None, f"{part} must survive a capped build"
    assert pipeline.encoder_stack.final_norm_gamma is not None
    print(f"[trace] layers knob: built {len(pipeline.layers)} of {pipeline.config.num_hidden_layers}")

    # --- items: what one traced step retires
    items = {s: getattr(pipeline, f"{s}_trace_items")() for s in PIPELINE_STAGES}
    assert items["decode"] == pipeline.batch, items
    assert items["prefill"] == pipeline.batch * pipeline.trace_capacity["prefill"], items
    print(f"[trace] items per step: {items}")

    # --- trace_inputs is exactly what trace_setup takes
    for stage in PIPELINE_STAGES:
        payload = getattr(pipeline, f"{stage}_trace_inputs")()
        assert set(payload) == {"input_ids", "position_ids"}, payload.keys()
        getattr(pipeline, f"{stage}_trace_setup")(payload)

    # --- capture / execute / release, one stage at a time
    ok = pipeline.trace_capture_selftest(mesh_device)
    print(f"[trace] report={pipeline.trace_report}")
    assert ok, f"trace capture selftest failed: {pipeline.trace_report}"

    # --- the authoritative fully-on-device check
    verdict = pipeline.host_op_selftest(steps=2)
    print(f"[hostop] on_device={verdict['on_device']} n_host_ops={verdict['n_host_ops']}")
    for task, sub in verdict["per_task"].items():
        print(f"[hostop] task={task} on_device={sub['on_device']} host_ops={sub['host_ops'][:8]}")
    assert verdict["on_device"], verdict["reason"]


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [TP], indirect=True)
def test_per_stage_layer_overrides(mesh_device, expect_error):
    """`<stage>_layers` overrides exist and are honoured; a real conflict is refused."""
    p = build_pipeline(mesh_device, prefill_layers=2, decode_layers=2)
    assert len(p.layers) == 2
    print("[trace] prefill_layers/decode_layers honoured")

    with expect_error(RuntimeError, "share one stack"):
        build_pipeline(mesh_device, prefill_layers=2, decode_layers=3)
    print("[trace] conflicting per-stage depths are refused, not silently resolved")
