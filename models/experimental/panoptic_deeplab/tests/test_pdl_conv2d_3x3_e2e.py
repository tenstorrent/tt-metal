# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Panoptic-DeepLab end to end (110-core Blackhole) with its 3x3 convs on ttnn.conv2d vs
ttnn.operations.conv2d_3x3 (TtConv2d.use_conv2d_3x3).

Seeded random weights stand in for model_final_bd324a.pkl: device time does not depend on weight values, and both
runs and the torch reference see the same weights. Each case runs one warm-up inference (compile, weight
preparation), then:
- MEASURED_RUNS untraced inferences, each on a fresh input (the model frees its input), bracketed by tracy
  signposts `pdl|<label>|<i>` / `pdl|<label>|end`, so a `--profile` run sums the device kernel time of every op in
  one inference;
- the production traced path (CustomTracedModelExecutor, as test_pipeline_e2e.py runs it): TRACED_RUNS replays timed
  one at a time (latency) and back to back (throughput).
Results go to PDL_E2E_RESULTS as JSON lines.
"""

import gc
import json
import os
import sys
import statistics
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB
from models.experimental.panoptic_deeplab.tests.pcc.common import bf16_conv_via_fp32, skip_if_not_blackhole_110_cores
from models.experimental.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    create_host_input_tensors_random,
    create_model_wrapper,
    create_pytorch_model,
    create_ttnn_model,
    get_panoptic_deeplab_config,
    preprocess_nchw_input_tensor,
)
from models.experimental.panoptic_deeplab.tt.tt_custom_pipeline import CustomTracedModelExecutor
from models.tt_cnn.tt.builder import TtConv2d

MEASURED_RUNS = 5
TRACED_RUNS = 20
# PDL_E2E_DEVICE_TIME=1 (with --profile): only the signposted untraced runs, the profiler read after each, so the
# device buffer does not overflow.
DEVICE_TIME_ONLY = os.environ.get("PDL_E2E_DEVICE_TIME") == "1"
RESULTS = os.environ.get("PDL_E2E_RESULTS", "/tmp/pdl_e2e_results.jsonl")


def _in_collect_pass():
    """True during run_safe_pytest's up-front collect pass, which runs test bodies to record programs."""
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _spec(t):
    mc = t.memory_config()
    cores = mc.shard_spec.grid.num_cores() if mc.is_sharded() and mc.shard_spec is not None else None
    where = f"{mc.buffer_type}".split(".")[-1]
    return (
        f"{t.dtype}".split(".")[-1]
        + f", {t.layout}".split(".")[-1].replace("_LAYOUT", "")
        + (f", {where} {mc.memory_layout}".split(".")[-1].replace("_SHARDED", "") + (f" ({cores})" if cores else ""))
    )


def _pcc(torch_nchw, tt_out):
    # The model returns NCHW with channels padded past the real count (validate_outputs_with_pcc slices the same way).
    t = ttnn.to_torch(tt_out).float()[:, : torch_nchw.shape[1]]
    a, r = torch_nchw.double().flatten(), t.double().flatten()
    return torch.corrcoef(torch.stack([a, r]))[0, 1].item()


def _conv_paths(model_convs):
    """(3x3 convs on conv2d_3x3, 3x3 convs on ttnn.conv2d, other convs, why each ttnn 3x3 conv stayed there)."""
    on_new = on_ttnn = other = 0
    skipped = []
    for obj in model_convs:
        c = obj.configuration
        if tuple(c.kernel_size) != (3, 3):
            other += 1
        elif isinstance(obj.conv2d_3x3_state, dict):
            on_new += 1
        else:
            on_ttnn += 1
            skipped.append(
                f"{c.input_height}x{c.input_width} {c.in_channels}->{c.out_channels} s{c.stride[0]} "
                f"d{c.dilation[0]} p{c.padding[0]}: {obj.conv2d_3x3_skip_reason}"
            )
    return on_new, on_ttnn, other, skipped


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("conv", ["ttnn", "conv2d_3x3"])
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE, "trace_region_size": 50_000_000}], indirect=True
)
def test_pdl_conv2d_3x3_e2e(device, conv):
    skip_if_not_blackhole_110_cores(device)
    config = get_panoptic_deeplab_config()
    height, width = config["train_size"]
    model_category = PANOPTIC_DEEPLAB
    label = f"{model_category}-{conv}"

    torch.manual_seed(0)
    pytorch_input = torch.randn(config["batch_size"], 3, height, width, dtype=torch.bfloat16)
    model_args = dict(
        target_size=config["train_size"],
        model_category=model_category,
        num_classes=config["num_classes"],
        common_stride=config["common_stride"],
        project_channels=config["project_channels"],
        decoder_channels=config["decoder_channels"],
        sem_seg_head_channels=config["sem_seg_head_channels"],
        ins_embed_head_channels=config["ins_embed_head_channels"],
    )
    pytorch_model = create_pytorch_model(weights_path=None, **model_args)
    before = {id(o) for o in gc.get_objects() if isinstance(o, TtConv2d)}
    ttnn_model = create_ttnn_model(
        device=device, pytorch_model=pytorch_model, batch_size=config["batch_size"], **model_args
    )
    model_convs = [o for o in gc.get_objects() if isinstance(o, TtConv2d) and id(o) not in before]

    # Tensor specs every 3x3 conv sees (and returns) in the model, recorded on the first inference.
    specs = []
    original_call = TtConv2d.__call__

    def recording_call(self, x, *args, **kwargs):
        x_spec = _spec(x)  # before the call: the conv may free its input
        out = original_call(self, x, *args, **kwargs)
        c = self.configuration
        if tuple(c.kernel_size) == (3, 3) and len(specs) < len(model_convs):
            y = out[0] if isinstance(out, tuple) else out
            specs.append(
                f"{c.input_height}x{c.input_width} {c.in_channels}->{c.out_channels} s{c.stride[0]} "
                f"d{c.dilation[0]} | in: {x_spec} | out: {_spec(y)}"
            )
        return out

    with torch.no_grad(), bf16_conv_via_fp32():
        torch_outputs = pytorch_model.forward(pytorch_input)[:3]

    previous = TtConv2d.use_conv2d_3x3
    TtConv2d.use_conv2d_3x3 = conv == "conv2d_3x3"
    try:
        tt_input = preprocess_nchw_input_tensor(device, pytorch_input)
        TtConv2d.__call__ = recording_call
        try:
            outputs = ttnn_model.forward(tt_input)
        finally:
            TtConv2d.__call__ = original_call
        ttnn.synchronize_device(device)
        if _in_collect_pass():
            return
        pcc = {n: _pcc(t, o) for n, t, o in zip(["semantic", "center", "offset"], torch_outputs, outputs[:3])}

        wall_ms = []
        for i in range(MEASURED_RUNS):
            tt_input = preprocess_nchw_input_tensor(device, pytorch_input)
            ttnn.synchronize_device(device)
            signpost(header=f"pdl|{label}|{i}")
            t0 = time.perf_counter()
            ttnn_model.forward(tt_input)
            ttnn.synchronize_device(device)
            wall_ms.append((time.perf_counter() - t0) * 1000)
            if DEVICE_TIME_ONLY:
                ttnn.ReadDeviceProfiler(device)
        signpost(header=f"pdl|{label}|end")
        if DEVICE_TIME_ONLY:
            return

        host_inputs, _, l1_memory_config = create_host_input_tensors_random(device, 1, height, width, TRACED_RUNS)
        executor = CustomTracedModelExecutor(create_model_wrapper(ttnn_model), device, l1_memory_config)
        executor.compile(host_inputs[0])
        traced_ms = []
        for host_input in host_inputs:
            t0 = time.perf_counter()
            executor._execute_single(host_input)
            ttnn.synchronize_device(device)
            traced_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        for host_input in host_inputs:
            executor._execute_single(host_input)
        ttnn.synchronize_device(device)
        traced_fps = len(host_inputs) / (time.perf_counter() - t0)
        executor.cleanup()
    finally:
        TtConv2d.use_conv2d_3x3 = previous

    on_new, on_ttnn, other, skipped = _conv_paths(model_convs)
    record = {
        "model": model_category,
        "conv": conv,
        "traced_ms_median": statistics.median(traced_ms),
        "traced_fps": traced_fps,
        "untraced_ms_median": statistics.median(wall_ms),
        "untraced_ms": wall_ms,
        "pcc": pcc,
        "conv3x3_on_conv2d_3x3": on_new,
        "conv3x3_on_ttnn_conv2d": on_ttnn,
        "other_convs": other,
        "conv3x3_skip_reasons": sorted(set(skipped)) if conv == "conv2d_3x3" else [],
        "conv3x3_specs": specs,
    }
    logger.info(f"PDL_E2E: {json.dumps(record)}")
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    if conv == "conv2d_3x3":
        assert on_ttnn == 0, f"{on_ttnn} 3x3 convs did not run on conv2d_3x3: {skipped}"
