# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fused-op style unit test for the Janus-Pro ViT: embedding + transformer + aligner.

Follows models/demos/deepseek_v3/tests/fused_op_unit_tests (AGENTS_GUIDE_ADD_TEST.md) and
shares its helpers. One call to ``TtJanusProTransformerVision.forward_device`` covers the
three modules under optimization:

    forward_device  ->  embeddings.forward_device  ->  patch_embed.forward_device
                    ->  encoder
                    ->  ln_post
                    ->  aligner

``prepare_patches`` (host im2col plus the transfer) is deliberately outside that entry
point: it is torch on host followed by an allocation, neither of which can live inside a
trace region.

Two measurement modes, as in the reference harness:

  * default -- host e2e latency over PERF_MEASURE_ITERS iterations after PERF_WARMUP_ITERS
    warmup, via ``measure_perf_us``;
  * with JANUS_VIT_DEVICE_PERF set -- the e2e block is skipped and the measured iterations
    are wrapped in signposts so a tracy run can report kernel duration and op-to-op latency
    per op. Their sum is the span from the first instruction to the last.

PCC is compared on a single plain call before any perf work, so warmup and the measurement
loops cannot influence it.
"""

import os
import subprocess

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import (
    compare_with_reference,
    get_int_env,
    measure_perf_us,
)
from models.experimental.janus_pro.tt.janus_pro_vision_model import TtJanusProTransformerVision
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PERF_ENV_VAR = "JANUS_VIT_DEVICE_PERF"

# Iteration counts follow AGENTS_GUIDE_ADD_TEST.md: 10/100 for host e2e, 10/10 on device.
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


_MESH_DEVICE_PARAM = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


def janus_vit_reference(model_args, images: torch.Tensor) -> torch.Tensor:
    """float32 HF reference: aligner(vision_model(pixel_values)).

    Depending on the transformers version get_image_features returns the aligned features
    directly or a wrapper holding them in pooler_output.
    """
    reference_model = model_args.reference_vision_transformer(wrap=False)
    reference_model.eval()
    with torch.no_grad():
        image_features = reference_model.model.get_image_features(images)
    return getattr(image_features, "pooler_output", image_features)


def janus_vit_ttnn(tt_model: TtJanusProTransformerVision, patches: ttnn.Tensor) -> ttnn.Tensor:
    """The sequence under test: everything from the patch projection to the aligner."""
    return tt_model.forward_device(patches)


def _run_janus_vit_test(
    mesh_device,
    dummy_weights,
    bsz,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    trace_mode,
    program_cache_enabled,
):
    logger.info(
        f"Janus ViT: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
        f"dummy_weights={dummy_weights}, bsz={bsz}"
    )
    if not program_cache_enabled:
        assert not trace_mode, "trace requires the program cache"
        mesh_device.disable_and_clear_program_cache()

    model_args = ModelArgs(mesh_device, dummy_weights=dummy_weights)
    state_dict = model_args.load_state_dict()

    images = torch.rand(
        (bsz, model_args.vision_in_channels, model_args.vision_chunk_size, model_args.vision_chunk_size)
    )
    reference_output = janus_vit_reference(model_args, images)

    tt_model = TtJanusProTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix="model.",  # the wrapper composes vision_model. and aligner. internally
        dtype=ttnn.bfloat16,
        configuration=model_args,
    )
    del state_dict

    # Host im2col + transfer once. On the traced path this buffer is the trace's input and
    # must keep its address, so it is never reallocated or freed between iterations.
    patches = tt_model.prepare_patches(images)

    def op_fn():
        return janus_vit_ttnn(tt_model, patches)

    # ---- correctness: one plain call, independent of everything below ----
    tt_output = op_fn()
    tt_output_torch = ttnn.to_torch(
        ttnn.from_device(tt_output), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, :, :, :]
    assert (
        tt_output_torch.shape == reference_output.shape
    ), f"Shape mismatch: tt {tuple(tt_output_torch.shape)} vs ref {tuple(reference_output.shape)}"

    pcc_value, max_abs_error = compare_with_reference(
        tt_output_torch,
        reference_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        convert_to_float=True,  # the tower returns bfloat16, the reference is float32
        strict_assert=False,  # PCC gates; assert_close only logs
    )
    ttnn.deallocate(tt_output)

    # ---- perf ----
    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        step_name = (
            f"janus_vit_{'trace' if trace_mode else 'no_trace'}_{'pcache' if program_cache_enabled else 'no_pcache'}"
        )
        warmup_iters = get_int_env("JANUS_VIT_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("JANUS_VIT_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name=step_name,
        )
        logger.info(f"{step_name}: {perf_us:.3f} us/iter over {measure_iters} iterations")

        benchmark_data.add_measurement(
            perf_profiler,
            0,
            step_name,
            f"{step_name}-e2e_duration_us",
            perf_us,
            target=expected_perf_us if expected_perf_us > 0 and not trace_mode and program_cache_enabled else None,
        )
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-pcc", pcc_value)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-max_abs_error", max_abs_error)

        if expected_perf_us > 0 and not trace_mode and program_cache_enabled:
            assert (
                perf_us <= expected_perf_us * 1.1
            ), f"Perf regression: {perf_us:.3f} us exceeds expected {expected_perf_us:.3f} us +10%"
        elif expected_perf_us == 0 and not trace_mode and program_cache_enabled:
            logger.warning("TODO: set expected_perf_us from a measured baseline")
    else:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        # The tracy report lands under generated/profiler/reports/<stamp>/ only after pytest exits,
        # so the run cannot print its own path. Log the commit instead: that is what names the
        # stage file in perf_reports/ and ties the run to a PERF.md change-log row.
        logger.info(f"device-perf run of commit {_head_sha()} -- see PERF.md 'Preserved profiler reports'")
        from tracy import signpost

        for _ in range(PERF_WARMUP_ITERS):
            out = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traced_output = op_fn()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(traced_output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                out = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(out)
            signpost("stop")

    ttnn.deallocate(patches)


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("bsz", [1])
@pytest.mark.parametrize("expected_pcc", [0.95])
@pytest.mark.parametrize("expected_atol, expected_rtol", [(0.1, 0.1)])
# TODO: replace with a target derived from theoretical numbers once a baseline exists.
@pytest.mark.parametrize("expected_perf_us", [0.0])
@pytest.mark.parametrize(
    "trace_mode, program_cache_enabled",
    [(False, True), (True, True), (False, False)],
    ids=["no_trace_pcache", "trace_pcache", "no_trace_no_pcache"],
)
@pytest.mark.parametrize(
    "device_params",
    # trace_region_size 0 asks for dynamic trace-region allocation; if capture reports the
    # region is too small, pin it to the size printed in the log.
    #
    # Profiling this path needs `python -m tracy --op-support-count 10000`: the default
    # per-RISC marker budget is 1000 programs and the device-perf branch runs the tower once
    # for PCC, PERF_WARMUP_ITERS times to warm up, once to capture the trace and
    # DEVICE_PERF_ITERS times to replay it. Overflow surfaces only as tracy post-processing
    # failing to match a host op against the device report.
    [{"fabric_config": True, "trace_region_size": 0}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_DEVICE_PARAM], indirect=True)
def test_janus_vision_tower(
    mesh_device,
    dummy_weights,
    bsz,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    trace_mode,
    program_cache_enabled,
    reset_seeds,
    ensure_gc,
):
    _run_janus_vit_test(
        mesh_device,
        dummy_weights,
        bsz,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
    )
