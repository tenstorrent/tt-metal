# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# mm_help profiling harness — isolated single conv2d for the main-vs-mm_help5 A/B.
# Underscore-prefixed so ambient `pytest tests/...` does NOT collect it; run EXPLICITLY
# under Tracy on each branch-build (A/B is cross-BUILD, not an in-build mode toggle):
#
#   CB_BATCH=20 CB_OUT_CH=256 CB_IN_CH=256 CB_H=14 CB_W=14 CB_FILTER=3 CB_STRIDE=1 \
#   CB_PAD=1,1,1,1 CB_SHARD=BS CB_WEIGHTS_DTYPE=bfloat8_b CB_FIDELITY=LoFi CB_L1_ACC=true \
#   python -m tracy -r -p -v -m pytest tests/ttnn/unit_tests/operations/conv/_conv_mmhelp_bench.py
#
# run_conv PCC-checks every run (a mis-wired trace fails loudly). run_twice=True gives one
# cold + one warm Conv2dDeviceOperation; the extractor takes the warm one. The why-fields
# (SBM/TRM, subblock, per_core, l1_acc) come from the Tracy CSV ATTRIBUTES column with no
# perturbation; on mm_help5 the factory also emits a log_info at cache-miss (warmup only).
#
# DRAM activation slicing (replicates the models' Conv2dSliceConfig so heavyweight
# convs that OOM on a single chip fit): set CB_SLICE_N>0 to build a
# ttnn.Conv2dSliceConfig(slice_type=..., num_slices=N) and pass it (+ use_dram_slicing)
# to run_conv. CB_SLICE_TYPE=height|width (default height). CB_SLICE_N=0 -> no slicing.
# Note: one sliced conv2d call dispatches N Conv2dDeviceOperations (one per slice); the
# extractor sums the warm invocation's N slices to report the whole-conv device time.
import os
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, BS, WS

_SHARD = {"HS": HS, "BS": BS, "WS": WS}
_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}
_FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
_LAYOUT = {"tile": ttnn.TILE_LAYOUT, "row_major": ttnn.ROW_MAJOR_LAYOUT}
_SLICE_TYPE = {"height": ttnn.Conv2dDRAMSliceHeight, "width": ttnn.Conv2dDRAMSliceWidth}


def _e(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


def _bool(v):
    return str(v).lower() in ("1", "true", "yes")


def _opt_int(v):
    return None if str(v).lower() == "none" else int(v)


def _pad(v):
    return tuple(int(x) for x in v.split(","))


CONFIG = dict(
    batch_size=_e("CB_BATCH", 1, int),
    output_channels=_e("CB_OUT_CH", 256, int),
    input_channels=_e("CB_IN_CH", 256, int),
    input_height=_e("CB_H", 14, int),
    input_width=_e("CB_W", 14, int),
    filter=_e("CB_FILTER", 3, int),
    stride=_e("CB_STRIDE", 1, int),
    padding=_e("CB_PAD", (1, 1, 1, 1), _pad),
    shard_layout=_e("CB_SHARD", HS, lambda v: _SHARD[v]),
    act_block_h_override=_e("CB_ABH", None, _opt_int),
    input_dtype=_e("CB_IN_DTYPE", ttnn.bfloat16, lambda v: _DTYPE[v]),
    output_dtype=_e("CB_OUT_DTYPE", ttnn.bfloat16, lambda v: _DTYPE[v]),
    fp32_accum=_e("CB_FP32_ACCUM", False, _bool),
    math_fidelity=_e("CB_FIDELITY", ttnn.MathFidelity.LoFi, lambda v: _FID[v]),
    has_bias=_e("CB_BIAS", True, _bool),
    weights_dtype=_e("CB_WEIGHTS_DTYPE", None, lambda v: None if str(v).lower() == "none" else _DTYPE[v]),
    output_layout=_e("CB_OUT_LAYOUT", ttnn.TILE_LAYOUT, lambda v: _LAYOUT[v]),
    packer_l1_acc=_e("CB_L1_ACC", True, _bool),
    groups=_e("CB_GROUPS", 1, int),
    num_slices=_e("CB_SLICE_N", 0, int),
    slice_type=_e("CB_SLICE_TYPE", "height", lambda v: _SLICE_TYPE[v]),
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_mmhelp_bench(device, torch_tensor_map):
    c = CONFIG
    config_override = {"act_block_h": c["act_block_h_override"]} if c["act_block_h_override"] else None
    slice_config = None
    if c["num_slices"] > 0:
        slice_config = ttnn.Conv2dSliceConfig(slice_type=c["slice_type"], num_slices=c["num_slices"])
    run_conv(
        device,
        torch_tensor_map,
        c["math_fidelity"],
        c["output_dtype"],
        c["weights_dtype"],
        c["batch_size"],
        c["output_channels"],
        c["input_channels"],
        c["input_height"],
        c["input_width"],
        c["filter"],
        c["filter"],
        c["stride"],
        c["stride"],
        c["padding"],
        config_override,
        shard_layout=c["shard_layout"],
        has_bias=c["has_bias"],
        fp32_accum=c["fp32_accum"],
        output_layout=c["output_layout"],
        packer_l1_acc=c["packer_l1_acc"],
        groups=c["groups"],
        slice_config=slice_config,
        use_dram_slicing=slice_config is not None,
        run_twice=True,
        input_dtype=c["input_dtype"],
    )
