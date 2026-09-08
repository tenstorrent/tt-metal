# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the trace JSONs in this directory.

    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/base_functionality/graph_l1_traces/dump_traces.py

Everything is captured in RunMode.NO_DISPATCH except matmul_normal_mode.json, which is the same
matmul in RunMode.NORMAL so the two can be diffed.  Fast runtime mode is disabled so the op
decorator emits the full function_start/function_end nesting.

conv2d.json needs a build where conv2d/halo exist; that test skips otherwise.
"""

import json
import pathlib

import pytest
import torch
import ttnn

OUT_DIR = pathlib.Path(__file__).parent


def _save(name, trace):
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(trace, indent=2))
    print(f"{name}: {len(trace)} nodes -> {path}")


def _capture(fn, mode=ttnn.graph.RunMode.NO_DISPATCH):
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        ttnn.graph.begin_graph_capture(mode)
        try:
            fn()
        finally:
            return ttnn.graph.end_graph_capture()


def _height_sharded(h, w, gx=8, gy=8):
    n = gx * gy
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            [h // n, w],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def test_dump_matmul(device):
    a = ttnn.from_torch(
        torch.rand(1, 1, 1024, 1024, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    _save("matmul_l1_out", _capture(lambda: ttnn.matmul(a, a, memory_config=ttnn.L1_MEMORY_CONFIG)))
    _save(
        "matmul_normal_mode",
        _capture(lambda: ttnn.matmul(a, a, memory_config=ttnn.L1_MEMORY_CONFIG), mode=ttnn.graph.RunMode.NORMAL),
    )


def test_dump_sharded_add(device):
    cfg = _height_sharded(2048, 512)

    def mk():
        return ttnn.from_torch(
            torch.rand(1, 1, 2048, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg,
        )

    a, b = mk(), mk()
    _save("sharded_add", _capture(lambda: ttnn.add(a, b, memory_config=cfg)))

    c, d = mk(), mk()
    _save("sharded_add_in_place", _capture(lambda: ttnn.add(c, d, output_tensor=c, memory_config=cfg)))


@pytest.mark.skipif(not hasattr(ttnn, "conv2d"), reason="conv2d not available in this build")
def test_dump_conv2d(device):
    from ttnn.operations.conv2d import Conv2dConfig

    b, ic, oc, h, w = 1, 32, 64, 128, 128
    x = ttnn.from_torch(
        torch.randn([1, 1, b * h * w, ic], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    weight = ttnn.from_torch(torch.randn([oc, ic, 3, 3], dtype=torch.bfloat16), dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(torch.randn([1, 1, 1, oc], dtype=torch.bfloat16), dtype=ttnn.bfloat16)
    conv_config = Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
    )

    def run():
        return ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=ic,
            out_channels=oc,
            batch_size=b,
            input_height=h,
            input_width=w,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=conv_config,
            dtype=ttnn.bfloat16,
        )

    _save("conv2d", _capture(run))
