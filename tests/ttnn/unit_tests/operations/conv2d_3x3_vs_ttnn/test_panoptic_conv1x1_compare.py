# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.conv2d on the 23 distinct Panoptic-DeepLab 1x1 conv calls (110-core Blackhole), replayed as the model makes them.

panoptic_conv1x1_cases.json holds, per distinct call, the model's TtConv2d configuration (sharding strategy, act block
override, compute config, buffering, deallocation) and the memory config of the input the model hands the conv. It is
captured from a model run with PDL_E2E_CONV_CAPTURE (models/experimental/panoptic_deeplab/tests/
test_pdl_conv2d_3x3_e2e.py). Each case builds that TtConv2d with seeded random weights, runs one warm-up call (compile,
weight preparation), then MEASURED_ITERS calls, each on a fresh input in the captured memory config and bracketed by
tracy signposts `cmp|<layer>|<op>|<i>`, so the device time of every program a call dispatches (the reshard of an
interleaved input included) can be summed from the profiler CSV. Run with `scripts/run_safe_pytest.sh --profile`.

CONV1X1_TUNE=1 also times single-knob variants of the model's configuration (`ttnn.conv2d <variant>`): the packer
L1 accumulation flipped, ttnn's own act block height, ttnn's own sharding, act_block_w_div 2 (block-sharded) and all
110 cores (height-sharded). A variant may change the output's shard grid; its record says which grid it produced.
"""

import json
import os

import pytest
import torch
import ttnn
from tracy import signpost

from models.tt_cnn.tt import builder
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d

PDL_L1_SMALL_SIZE = 4 * 1024
MEASURED_ITERS = 3
RESULTS = os.environ.get("CONV_CMP_RESULTS", "/tmp/conv1x1_cmp_results.jsonl")
TUNE = os.environ.get("CONV1X1_TUNE") == "1"
CASES = json.load(open(os.path.join(os.path.dirname(__file__), "panoptic_conv1x1_cases.json")))
# Comma-separated CASES indices to run. Selects by env var because `python -m tracy` (used by --profile) re-joins argv
# through a shell and breaks -k quoting on these ids.
_ONLY = {int(i) for i in os.environ.get("CONV_CMP_CASES", "").split(",") if i.strip()}


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _core_range_set(ranges):
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in ranges}
    )


def _memory_config(spec):
    memory_layout = getattr(ttnn.TensorMemoryLayout, spec["memory_layout"])
    buffer_type = getattr(ttnn.BufferType, spec["buffer_type"])
    if spec["shard_grid"] is None:
        return ttnn.MemoryConfig(memory_layout, buffer_type)
    shard_spec = ttnn.ShardSpec(
        _core_range_set(spec["shard_grid"]),
        spec["shard_shape"],
        getattr(ttnn.ShardOrientation, spec["shard_orientation"]),
    )
    return ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)


def _strategy(spec):
    if spec is None:
        return None
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    if kwargs.get("override_core_grid") is not None:
        kwargs["override_core_grid"] = _core_range_set(kwargs["override_core_grid"])
    return getattr(builder, spec["type"])(**kwargs)


def _configuration(case, torch_weight, torch_bias):
    weight, bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(torch_weight, torch_bias)
    act = case["activation"]
    return Conv2dConfiguration(
        input_height=case["input_height"],
        input_width=case["input_width"],
        in_channels=case["in_channels"],
        out_channels=case["out_channels"],
        batch_size=case["batch_size"],
        kernel_size=tuple(case["kernel_size"]),
        weight=weight,
        stride=tuple(case["stride"]),
        padding=tuple(case["padding"]),
        groups=case["groups"],
        dilation=tuple(case["dilation"]),
        activation=None
        if act is None
        else ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, act["op_type"]), *act["params"]),
        bias=bias,
        activation_dtype=getattr(ttnn.DataType, case["activation_dtype"]),
        weights_dtype=getattr(ttnn.DataType, case["weights_dtype"]),
        output_dtype=getattr(ttnn.DataType, case["output_dtype"]),
        output_layout=getattr(ttnn.Layout, case["output_layout"]),
        sharding_strategy=_strategy(case["sharding_strategy"]),
        slice_strategy=_strategy(case["slice_strategy"]),
        math_fidelity=getattr(ttnn.MathFidelity, case["math_fidelity"]),
        fp32_dest_acc_en=case["fp32_dest_acc_en"],
        packer_l1_acc=case["packer_l1_acc"],
        enable_act_double_buffer=case["enable_act_double_buffer"],
        enable_weights_double_buffer=case["enable_weights_double_buffer"],
        deallocate_activation=case["deallocate_activation"],
        reallocate_halo_output=case["reallocate_halo_output"],
        config_tensors_in_dram=case["config_tensors_in_dram"],
    )


def _variants(case):
    """(name, case overrides) for each single-knob change to the model's configuration."""
    strategy = case["sharding_strategy"]
    out = [
        ("l1acc-off" if case["packer_l1_acc"] else "l1acc-on", {"packer_l1_acc": not case["packer_l1_acc"]}),
        ("auto-shard", {"sharding_strategy": {"type": "AutoShardedStrategyConfiguration"}}),
    ]
    if strategy.get("act_block_h_override"):
        out.append(("abh-auto", {"sharding_strategy": strategy | {"act_block_h_override": 0}}))
    if strategy["type"] == "BlockShardedStrategyConfiguration":
        out.append(("wdiv2", {"sharding_strategy": strategy | {"act_block_w_div": 2}}))
    if strategy["type"] == "HeightShardedStrategyConfiguration":
        out.append(("110-cores", {"sharding_strategy": strategy | {"override_core_grid": [[0, 0, 10, 9]]}}))
    return out


def _output_spec(t):
    mc = t.memory_config()
    spec = f"{mc.buffer_type} {mc.memory_layout}".replace("BufferType.", "").replace("TensorMemoryLayout.", "")
    return spec + (f" ({mc.shard_spec.grid.num_cores()})" if mc.is_sharded() else "")


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c["layer"] for c in CASES])
def test_panoptic_conv1x1_compare(device, idx, monkeypatch):
    if _ONLY and idx not in _ONLY:
        pytest.skip("not selected by CONV_CMP_CASES")
    monkeypatch.setattr(TtConv2d, "use_conv2d_3x3", False)
    case = CASES[idx]
    layer = case["layer"]
    spec = case["input"]
    H, W, c_in, c_out = case["input_height"], case["input_width"], case["in_channels"], case["out_channels"]
    stride = case["stride"][0]

    torch.manual_seed(0)
    torch_input = torch.randn(spec["shape"], dtype=torch.bfloat16)
    torch_weight = torch.randn(c_out, c_in, 1, 1) / c_in**0.5
    torch_bias = torch.randn(c_out) if case["has_bias"] else None
    expected = torch.nn.functional.conv2d(
        torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2), torch_weight, torch_bias, stride=stride
    )
    if case["activation"] is not None:
        expected = torch.relu(expected)
    h_out, w_out = expected.shape[2:]
    expected = expected.permute(0, 2, 3, 1).reshape(h_out * w_out, c_out)

    input_mc = _memory_config(spec)
    dtype, layout = getattr(ttnn.DataType, spec["dtype"]), getattr(ttnn.Layout, spec["layout"])

    def fresh_input():
        return ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=input_mc)

    def measure(op, variant_case):
        """Warm-up call, then MEASURED_ITERS signposted calls; returns (pcc, output spec) of the last one."""
        conv = TtConv2d(_configuration(variant_case, torch_weight, torch_bias), device)

        def run(x):
            out = conv(x)
            if not variant_case["deallocate_activation"]:
                ttnn.deallocate(x)
            return out

        ttnn.deallocate(run(fresh_input()))
        ttnn.synchronize_device(device)
        out = None
        for i in range(MEASURED_ITERS):
            if out is not None:
                ttnn.deallocate(out)
            x = fresh_input()
            ttnn.synchronize_device(device)
            signpost(header=f"cmp|{layer}|{op}|{i}")
            out = run(x)
            ttnn.synchronize_device(device)
        signpost(header=f"cmp|{layer}|end_{op}")
        pcc = _pcc(ttnn.to_torch(out).float().reshape(-1, out.shape[-1])[: h_out * w_out, :c_out], expected)
        spec = _output_spec(out)
        ttnn.deallocate(out)
        ttnn.ReadDeviceProfiler(device)
        return pcc, spec

    pcc, spec = measure("ttnn.conv2d", case)
    records = [{"layer": layer, "op": "ttnn.conv2d", "pcc": pcc, "output": spec}]
    for name, overrides in _variants(case) if TUNE else []:
        try:
            v_pcc, v_spec = measure(f"ttnn.conv2d {name}", case | overrides)
            records.append({"layer": layer, "op": f"ttnn.conv2d {name}", "pcc": v_pcc, "output": v_spec})
        except Exception as e:  # a variant ttnn.conv2d rejects is a result, not a test failure
            records.append({"layer": layer, "op": f"ttnn.conv2d {name}", "error": str(e).splitlines()[0][:300]})
    with open(RESULTS, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    assert pcc > 0.99, f"{layer}: pcc ttnn.conv2d={pcc:.5f}"
