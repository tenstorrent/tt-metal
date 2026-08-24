# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the Janus-Pro ViT device path.

Same sequence as ``test_vision_tower_janus`` device-perf + trace: PCC, warmup,
then capture. Signposted iterations replay the captured trace so the report
covers already-compiled, already-dispatched programs — not host dispatch.
"""

import os
import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import compare_with_reference
from models.experimental.janus_pro.tt.janus_pro_vision_model import TtJanusProTransformerVision
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

PERF_WARMUP_ITERS = 1
DEVICE_PERF_ITERS = 1


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


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("bsz", [1])
@pytest.mark.parametrize("expected_pcc", [0.95])
@pytest.mark.parametrize("expected_atol, expected_rtol", [(0.1, 0.1)])
@pytest.mark.parametrize(
    "device_params",
    # trace_region_size 0 asks for dynamic trace-region allocation; if capture reports the
    # region is too small, pin it to the size printed in the log.
    [{"fabric_config": True, "trace_region_size": 0}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_DEVICE_PARAM], indirect=True)
def test_janus_vision_model_profile(
    mesh_device,
    dummy_weights,
    bsz,
    expected_pcc,
    expected_atol,
    expected_rtol,
    reset_seeds,
    ensure_gc,
):
    logger.info(f"device-perf run of commit {_head_sha()} -- see PERF.md 'Preserved profiler reports'")

    model_args = ModelArgs(mesh_device, dummy_weights=dummy_weights)
    state_dict = model_args.load_state_dict()

    images = torch.rand(
        (bsz, model_args.vision_in_channels, model_args.vision_chunk_size, model_args.vision_chunk_size)
    )
    reference_model = model_args.reference_vision_transformer(wrap=False)
    reference_model.eval()
    image_features = reference_model.model.get_image_features(images)
    reference_output = getattr(image_features, "pooler_output", image_features)

    tt_model = TtJanusProTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix="model.",
        dtype=ttnn.bfloat16,
        configuration=model_args,
    )
    del state_dict

    # Host im2col + transfer once. This buffer is the trace's input and must keep
    # its address, so it is never reallocated or freed between iterations.
    patches = tt_model.prepare_patches(images)

    def op_fn():
        return tt_model.forward_device(patches)

    tt_output = op_fn()
    tt_output_torch = ttnn.to_torch(
        ttnn.from_device(tt_output), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, :, :, :]
    assert (
        tt_output_torch.shape == reference_output.shape
    ), f"Shape mismatch: tt {tuple(tt_output_torch.shape)} vs ref {tuple(reference_output.shape)}"

    compare_with_reference(
        tt_output_torch,
        reference_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        convert_to_float=True,
        strict_assert=False,
    )
    ttnn.deallocate(tt_output)

    for _ in range(PERF_WARMUP_ITERS):
        out = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)

    ttnn.synchronize_device(mesh_device)
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

    ttnn.deallocate(patches)
