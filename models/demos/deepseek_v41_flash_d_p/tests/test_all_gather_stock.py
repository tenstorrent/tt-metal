# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""deepseek_v41_flash_d_p's all-gathers on stock ttnn.experimental.high_bw_all_gather, bit-exact (a regression test of the
stock op at the model's calls). Device test, 2x2 (Blackhole QuietBox), after a build:

    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_all_gather_stock.py -s

The 15 cases are the model's cases of the retired fork ttnn.bringup.fabric_all_gather
(ttnn/ttnn/bringup/fabric_all_gather_ttnn/tests/cases.py, task O.1, sigs from bringup/results/fork_calls.json),
moved unchanged when the fork was retired (2026-11-06; upstream #61387 made the stock op the same program, same
signature and validation). tt/ccl.py:all_gather (CCL_IMPL fabric / fabric_bf16) calls the stock op with the same arguments. Inputs random in the captured dtype;
every chip's output compared bit for bit (incl. a partial gather's untouched fill), then determinism:
models/demos/common/bringup/tests/all_gather_exact.py."""

import pytest

from models.demos.common.bringup.tests.all_gather_exact import parametrize, run_case

# deepseek_v41_flash_d_p O.1 (scaled 2x2 run, ladder rung last: chunk 53248-55296, 1024 rows a chip): every gather of
# tt/ccl.py:all_gather under CCL_IMPL fabric_bf16 (the norms' stats partials, the mHC partial rows on dim 1, the hidden
# columns on dim 3, the indexer's uint32 top-k rows, the experts' q(x) rows over axis 0, ...). The model passes the
# output preallocated (allocate_tensor_on_device), its own pair of L1_SMALL global semaphores (ccl._sems: zero, the
# whole worker grid) and num_links None (every usable link of the axis); its mesh runs a 10880 B fabric packet payload
# (hooks.device_params: the weighted combine's one-row packets). Inputs random in the captured dtype (uint32: random
# 31-bit words); compared bit for bit.
_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576, "fabric_payload_bytes": 10880},
    "num_links": None,
    "semaphores": True,
    "seed": 0,
    "exact": True,
}

CASES = [
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim3-float32-tile-s1024-w64",
        "sig": "0350cd2422",
        "input": {"shape": [1, 1, 1024, 64], "dtype": "FLOAT32", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 128]},
        "dim": 3,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s512-w1792",
        "sig": "2f761f8778",
        "input": {"shape": [1, 1, 512, 1792], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 1792]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-uint32-rm-s512-w512",
        "sig": "3ed44796fc",
        "input": {"shape": [1, 1, 512, 512], "dtype": "UINT32", "layout": "ROW_MAJOR"},
        "output": {"shape": [1, 1, 1024, 512]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-float32-tile-s512-w32",
        "sig": "43f469cea2",
        "input": {"shape": [1, 1, 512, 32], "dtype": "FLOAT32", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 32]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-rm-s1024-w5120",
        "sig": "44b17650b3",
        "input": {"shape": [1, 1, 1024, 5120], "dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "output": {"shape": [1, 1, 2048, 5120]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim3-bfloat16-tile-s1024-w2560",
        "sig": "50b80cde70",
        "input": {"shape": [1, 1, 1024, 2560], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 5120]},
        "dim": 3,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim1-float32-tile-s1024-w32",
        "sig": "5215b10576",
        "input": {"shape": [1, 1, 1024, 32], "dtype": "FLOAT32", "layout": "TILE"},
        "output": {"shape": [1, 2, 1024, 32]},
        "dim": 1,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s256-w512",
        "sig": "6d75743894",
        "input": {"shape": [1, 1, 256, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 512, 512]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s512-w512",
        "sig": "743a405dcd",
        "input": {"shape": [1, 1, 512, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 512]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s1024-w128",
        "sig": "a8b1f5026d",
        "input": {"shape": [1, 1, 1024, 128], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 2048, 128]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s512-w1024",
        "sig": "ad4aac6edc",
        "input": {"shape": [1, 1, 512, 1024], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 1024]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s128-w512",
        "sig": "bc9fc305e3",
        "input": {"shape": [1, 1, 128, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 256, 512]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s512-w512",
        "sig": "d5f476e647",
        "input": {"shape": [1, 1, 512, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 512]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s1024-w512",
        "sig": "e109e2154a",
        "input": {"shape": [1, 1, 1024, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 2048, 512]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s512-w128",
        "sig": "ffe3863107",
        "input": {"shape": [1, 1, 512, 128], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 128]},
        "dim": 2,
        "cluster_axis": 0,
    },
]


@pytest.mark.timeout(900)
@pytest.mark.parametrize(**parametrize(CASES))
def test_all_gather_stock(mesh_device, device_params, case):
    run_case(mesh_device, case)
