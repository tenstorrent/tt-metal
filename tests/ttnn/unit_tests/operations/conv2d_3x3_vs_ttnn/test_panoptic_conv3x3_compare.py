# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""conv2d_3x3 vs ttnn.conv2d on the 22 unique Panoptic-DeepLab 3x3 layer configs.

Both ops get the identical pinned L1-sharded input (Panoptic's 110-core geometry from
eval/golden_tests/conv2d_3x3/feature_spec.py LOOSE_CASES) and Panoptic's compute config
(LoFi, fp32 DEST, bf8_b weights). ttnn.conv2d uses the per-layer Conv2dConfig that
test_conv2d_panoptic_110_cores pins for the model. Each op runs one warm-up call (compile,
and ttnn.conv2d's own weight preparation), then MEASURED_ITERS calls bracketed by tracy
signposts `cmp|<layer>|<op>|<i>` so per-call device time can be summed from the profiler CSV.
Run with `scripts/run_safe_pytest.sh --profile`.
"""

import json
import os
import sys

import pytest
import torch
import ttnn
from tracy import signpost

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, REPO)

from eval.golden_tests.conv2d_3x3 import feature_spec  # noqa: E402
from eval.golden_tests.conv2d_3x3.helpers import (  # noqa: E402
    build_input_memory_config,
    create_ttnn_input_tensor,
    make_activation,
    make_bias,
    make_weight,
    output_hw,
    pytorch_conv2d_3x3,
)
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights  # noqa: E402

PDL_L1_SMALL_SIZE = 4 * 1024
MEASURED_ITERS = 3
RESULTS = os.environ.get("CONV_CMP_RESULTS", "/tmp/conv_cmp_results.jsonl")

# Panoptic's act_block_h_override per layer (test_conv2d_panoptic_110_cores; aspp.convs.3 from
# model_configs.py). Merged LOOSE rows use the override of the most frequent member layer.
ACT_BLOCK_H = {
    "stem.conv1": 608,
    "stem.conv2": 64,
    "stem.conv3": 64,
    "res2.{0,1,2}.conv2": 320,
    "res3.0.conv2": 96,
    "res3.{1,2,3}.conv2 + instance res3.fuse_conv.1": 96,
    "res4.0.conv2": 256,
    "res4.{1..5}.conv2": 256,
    "res5.0.conv2": 224,
    "res5.1.conv2": 224,
    "res5.2.conv2": 224,
    "aspp.convs.1 (x2 heads)": 128,
    "aspp.convs.2 (x2 heads)": 64,
    "aspp.convs.3 (x2 heads)": 256,
    "semantic decoder.res3.fuse_conv.0": 32,
    "instance decoder.res3.fuse_conv.0": 32,
    "semantic decoder.res3.fuse_conv.1": 32,
    "semantic decoder.res2.fuse_conv.0": 32,
    "instance decoder.res2.fuse_conv.0": 32,
    "semantic res2.fuse_conv.1 + semantic_head.head.{0,1}": 64,
    "instance res2.fuse_conv.1 + center_head.0 + offset_head.0": 64,
    "center_head.1 + offset_head.1": 64,
}

# Layers Panoptic runs on ttnn.conv2d as channel slices (model_configs.py ChannelSliceStrategyConfiguration):
# the full-K call does not fit ttnn.conv2d's L1. As in TtConv2d, each slice runs without bias or activation and
# the slices are summed afterwards; only the conv calls are timed.
TTNN_CHANNEL_SLICES = {"aspp.convs.3 (x2 heads)": 2}

CASES = [c for c in feature_spec.LOOSE_CASES if "@ LoFi" not in c["extras"]["layer"]]
# Comma-separated CASES indices to run. Selects by env var because `python -m tracy` (used by
# --profile) re-joins argv through a shell and breaks -k quoting on these ids.
_ONLY = {int(i) for i in os.environ.get("CONV_CMP_CASES", "").split(",") if i.strip()}


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c["extras"]["layer"] for c in CASES])
def test_panoptic_conv3x3_compare(device, idx):
    if _ONLY and idx not in _ONLY:
        pytest.skip("not selected by CONV_CMP_CASES")
    case = CASES[idx]
    extras = case["extras"]
    layer = extras["layer"]
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    dtype, layout, mem = case["dtype"], case["layout"], case["memory_layout"]
    weight_dtype = case["weight_dtype"]
    relu = case["activation"] == "relu"
    with_bias = case["bias_mode"] == "with_bias"
    c_in_real = extras.get("c_in_real")

    torch.manual_seed(0)
    torch_input = make_activation(H, W, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
    torch_weight = make_weight(c_out, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
    torch_bias = make_bias(c_out, dtype=torch.bfloat16) if with_bias else None
    expected = pytorch_conv2d_3x3(
        torch_input,
        torch_weight,
        input_height=H,
        input_width=W,
        bias=torch_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if relu else None,
    )
    h_out, w_out = output_hw(H, W, padding=padding, stride=stride, dilation=dilation)

    input_mc = build_input_memory_config(
        (1, 1, H * W, c_in),
        mem,
        layout=layout,
        dtype=dtype,
        device=device,
        extras=extras,
    )
    ttnn_input = create_ttnn_input_tensor(torch_input, device, dtype=dtype, layout=layout, memory_config=input_mc)

    # --- conv2d_3x3 ---
    prep = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=input_mc,
        device=device,
    )
    new_w = prepare_conv2d_weights(torch_weight, **prep)
    new_b = prepare_conv2d_bias(torch_bias, **prep) if with_bias else None
    new_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    new_kwargs = dict(
        input_height=H,
        input_width=W,
        bias=new_b,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if relu else None,
        compute_kernel_config=new_cfg,
    )
    conv2d_3x3(ttnn_input, new_w, **new_kwargs)
    ttnn.synchronize_device(device)
    for i in range(MEASURED_ITERS):
        signpost(header=f"cmp|{layer}|conv2d_3x3|{i}")
        new_out = conv2d_3x3(ttnn_input, new_w, **new_kwargs)
        ttnn.synchronize_device(device)
    signpost(header=f"cmp|{layer}|end_new")
    new_pcc = _pcc(
        ttnn.to_torch(new_out).reshape(-1, new_out.shape[-1])[: h_out * w_out, :c_out], expected.reshape(-1, c_out)
    )
    # ttnn.conv2d runs as Panoptic runs it (deallocate_activation=True: it frees its input after the
    # halo, and its CB budget assumes that), so it gets a fresh input per call and nothing else in L1.
    ttnn.deallocate(new_out)
    ttnn.deallocate(ttnn_input)
    del new_out, ttnn_input

    # --- ttnn.conv2d, Panoptic's per-layer config ---
    num_slices = TTNN_CHANNEL_SLICES.get(layer, 1)
    sliced = num_slices > 1
    slice_c = c_in // num_slices
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weight_dtype,
        shard_layout=mem,
        deallocate_activation=True,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        output_layout=ttnn.TILE_LAYOUT,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu and not sliced else None,
        config_tensors_in_dram=True,
    )
    conv_config.act_block_h_override = ACT_BLOCK_H[layer]
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    slice_mc = input_mc
    if sliced:
        shard_h, shard_w = extras["shard_shape"]
        slice_mc = build_input_memory_config(
            (1, 1, H * W, slice_c),
            mem,
            layout=layout,
            dtype=dtype,
            device=device,
            extras={**extras, "shard_shape": [shard_h, shard_w // num_slices]},
        )
    tt_ws = [
        ttnn.from_torch(torch_weight[:, s * slice_c : (s + 1) * slice_c].float(), ttnn.float32)
        for s in range(num_slices)
    ]
    tt_b = (
        ttnn.from_torch(torch_bias.float().reshape(1, 1, 1, c_out), ttnn.float32) if with_bias and not sliced else None
    )
    ref_kwargs = dict(
        in_channels=slice_c,
        out_channels=c_out,
        device=device,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=(padding, padding),
        dilation=(dilation, dilation),
        batch_size=1,
        input_height=H,
        input_width=W,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=dtype,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    def fresh_inputs():
        return [
            create_ttnn_input_tensor(
                torch_input[..., s * slice_c : (s + 1) * slice_c],
                device,
                dtype=dtype,
                layout=layout,
                memory_config=slice_mc,
            )
            for s in range(num_slices)
        ]

    def run_ref(inputs):
        nonlocal tt_b
        outs = []
        for s, inp in enumerate(inputs):
            out, _, [tt_ws[s], tt_b] = ttnn.conv2d(
                input_tensor=inp, weight_tensor=tt_ws[s], bias_tensor=tt_b, **ref_kwargs
            )
            outs.append(out)
        return outs

    for out in run_ref(fresh_inputs()):
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    ref_outs = []
    for i in range(MEASURED_ITERS):
        for out in ref_outs:
            ttnn.deallocate(out)
        inputs = fresh_inputs()
        signpost(header=f"cmp|{layer}|ttnn.conv2d|{i}")
        ref_outs = run_ref(inputs)
        ttnn.synchronize_device(device)
    signpost(header=f"cmp|{layer}|end_ref")
    ref = sum(ttnn.to_torch(out).float().reshape(-1, out.shape[-1])[: h_out * w_out, :c_out] for out in ref_outs)
    if sliced:
        if with_bias:
            ref = ref + torch_bias.float().reshape(1, c_out)
        if relu:
            ref = torch.relu(ref)
    ref_pcc = _pcc(ref, expected.reshape(-1, c_out))

    ttnn.ReadDeviceProfiler(device)
    with open(RESULTS, "a") as f:
        f.write(json.dumps({"layer": layer, "pcc_conv2d_3x3": new_pcc, "pcc_ttnn_conv2d": ref_pcc}) + "\n")
    assert new_pcc > 0.99 and ref_pcc > 0.99, f"{layer}: pcc conv2d_3x3={new_pcc:.5f} ttnn.conv2d={ref_pcc:.5f}"
