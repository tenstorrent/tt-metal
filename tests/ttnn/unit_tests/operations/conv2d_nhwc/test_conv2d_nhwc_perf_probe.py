# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe (not an acceptance test): BLOCK-sharded shapes for on-device A/B of R4 knobs.
Run under scripts/run_safe_pytest.sh --profile; read per-core kernel durations."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402
import ttnn  # noqa: E402
from test_conv2d_nhwc_sharded import ML, run_sharded  # noqa: E402


def test_perf_probe_block_28x28x256(device):
    run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, True)


def test_perf_probe_block_sdxl_l1(device):
    run_sharded(
        device, (1, 64, 64, 640), 640, 3, 1, 1, ML.BLOCK_SHARDED, True, shard_shape=(410, 64), core_grid=(10, 10)
    )


def test_perf_probe_interleaved_sdxl_l1(device):
    run_sharded(device, (1, 64, 64, 640), 640, 3, 1, 1, ML.INTERLEAVED, True)


def test_perf_probe_block_sdxl_l1_bf16_bf8b(device):
    """The perf-1 contract config: bf16 activations x bf8b weights, bias, pinned 10x10 BLOCK."""
    run_sharded(
        device,
        (1, 64, 64, 640),
        640,
        3,
        1,
        1,
        ML.BLOCK_SHARDED,
        True,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        shard_shape=(410, 64),
        core_grid=(10, 10),
    )


def test_l1_peak_block_sdxl_l1_bf16_bf8b(device):
    """Per-core L1 peak (graph capture, as the eval harness's device_l1_peak_bytes) of the
    perf-1 contract config; the target is production's 618,992 B."""
    device.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        test_perf_probe_block_sdxl_l1_bf16_bf8b(device)
    finally:
        trace = ttnn.graph._cpp_end_graph_capture()
        device.enable_program_cache()
    peak = int(ttnn.graph.extract_resource_usage_per_core(trace).peak_total)
    print(f"L1_PEAK_BYTES={peak}")
    assert peak > 0


# Regression-guard representatives: one per kernel path x layout x placement.
def test_guard_interleaved_bf16_relu(device):
    run_sharded(
        device, (1, 28, 28, 256), 256, 3, 1, 1, ML.INTERLEAVED, True, ttnn.bfloat16, ttnn.bfloat8_b, activation="relu"
    )


def test_guard_block_bf16_nobias(device):
    run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, False, ttnn.bfloat16, ttnn.bfloat8_b)


def test_guard_width_strided(device):
    run_sharded(device, (1, 14, 14, 256), 512, 3, 1, 2, ML.WIDTH_SHARDED, True)


# Refinement 4 depthwise (R11) probes: MobileNet-style shapes, bf16 x bf8b.
def _dw(device, shape, stride, memory_layout):
    from test_conv2d_nhwc_depthwise import run_dw

    run_dw(device, shape, 3, 1, stride, memory_layout, True, ttnn.bfloat16, ttnn.bfloat8_b)


def test_perf_probe_dw_il_112x112x32(device):
    _dw(device, (1, 112, 112, 32), 1, ML.INTERLEAVED)


def test_perf_probe_dw_il_56x56x128(device):
    _dw(device, (1, 56, 56, 128), 1, ML.INTERLEAVED)


def test_perf_probe_dw_il_112x112x64_s2(device):
    _dw(device, (1, 112, 112, 64), 2, ML.INTERLEAVED)


def test_perf_probe_dw_width_7x7x1024(device):
    _dw(device, (1, 7, 7, 1024), 1, ML.WIDTH_SHARDED)


def test_perf_probe_dw_block_28x28x256(device):
    _dw(device, (1, 28, 28, 256), 1, ML.BLOCK_SHARDED)


# Refinement 6: the four ResNet50 HEIGHT perf cases (batch 16, bf16 x bf8b, bias), pinned geometry.
RESNET_HEIGHT = {
    "layer1_3x3": ((16, 56, 56, 64), 64, 3, 1, 1, (480, 64), 105),
    "layer2_ds": ((16, 56, 56, 128), 128, 3, 1, 2, (457, 128), 110),
    "layer2_3x3": ((16, 28, 28, 128), 128, 3, 1, 1, (128, 128), 98),
    "shortcut_1x1_s2": ((16, 56, 56, 64), 256, 1, 0, 2, (457, 64), 110),
}


def _resnet_height(device, name):
    shape, C_out, k, pad, stride, shard, cores = RESNET_HEIGHT[name]
    run_sharded(
        device,
        shape,
        C_out,
        k,
        pad,
        stride,
        ML.HEIGHT_SHARDED,
        True,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        shard_shape=shard,
        num_cores=cores,
    )


def test_perf_probe_resnet_layer1_3x3(device):
    _resnet_height(device, "layer1_3x3")


def test_perf_probe_resnet_layer2_ds(device):
    _resnet_height(device, "layer2_ds")


def test_perf_probe_resnet_layer2_3x3(device):
    _resnet_height(device, "layer2_3x3")


def test_perf_probe_resnet_shortcut_1x1_s2(device):
    _resnet_height(device, "shortcut_1x1_s2")


@pytest.mark.parametrize("name", list(RESNET_HEIGHT))
def test_l1_peak_resnet_height(device, name):
    """Per-core L1 peak (graph capture, as the harness's device_l1_peak_bytes) of a ResNet HEIGHT case."""
    device.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        _resnet_height(device, name)
    finally:
        trace = ttnn.graph._cpp_end_graph_capture()
        device.enable_program_cache()
    peak = int(ttnn.graph.extract_resource_usage_per_core(trace).peak_total)
    print(f"L1_PEAK_BYTES {name}={peak}")
    assert peak > 0
