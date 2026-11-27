# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of groupnorm_sc_N_1_HW_C_b: native C++ op versus the frozen generic_op route."""
import json
import os

import groupnorm_sc_N_1_HW_C_b_generic  # frozen generic route, provided by the driver
import intree_host_comparison as hc
import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
    TOLERANCES,
    block_shard_memory_config,
    check_output,
    pytorch_groupnorm_sc_N_1_HW_C,
)
from ttnn.operations import groupnorm_sc_N_1_HW_C_b as op_pkg

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

# (shape, num_groups, layout, block-shard grid or None for DRAM interleaved); every case has gamma and beta (bf16 RM).
CASES = [
    # test_acceptance.py dram_tile_single_tile_g1: the smallest supported input.
    pytest.param((1, 1, 32, 32), 1, TILE, None, id="dram_min"),
    # test_acceptance.py dram_vs_block_sharded (A): SDXL attention-norm size, device time matters.
    pytest.param((1, 1, 1024, 1280), 32, TILE, None, id="dram_1024x1280"),
    # test_acceptance.py BS_TILE: L1 block-sharded TILE, 8x8.
    pytest.param((1, 1, 1024, 1280), 32, TILE, (8, 8), id="bs_tile_1024x1280"),
    # test_acceptance.py bs_rm_8x8_narrow_k4: L1 block-sharded ROW_MAJOR, 8x8 (the SDXL resnet norm layout).
    pytest.param((1, 1, 4096, 320), 32, RM, (8, 8), id="bs_rm_4096x320"),
]


@pytest.mark.parametrize(("shape", "num_groups", "layout", "grid"), CASES)
def test_host_perf(device, shape, num_groups, layout, grid, request):
    torch.manual_seed(0)
    C = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    beta = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    expected = pytorch_groupnorm_sc_N_1_HW_C(x, num_groups, gamma=gamma, beta=beta)
    mem = (
        block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        if grid is not None
        else ttnn.DRAM_MEMORY_CONFIG
    )

    def make_input():
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)

    def make_affine(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def validate(out):
        cfg = out.memory_config()
        if grid is not None:
            assert cfg.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            assert cfg.buffer_type == ttnn.BufferType.L1
            assert (
                list(cfg.shard_spec.shape) == list(mem.shard_spec.shape) and cfg.shard_spec.grid == mem.shard_spec.grid
            )
        else:
            assert cfg.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            assert cfg.buffer_type == ttnn.BufferType.DRAM
        check_output(
            out,
            expected,
            shape=list(shape),
            dtype=ttnn.bfloat16,
            expected_layout=layout if grid is not None else TILE,
            tolerance=TOLERANCES[ttnn.bfloat16],
        )

    def route(name, op):
        return hc.RouteAdapter(
            name,
            lambda case, phase, block: hc.BoundCall(
                op,
                validate,
                (make_input(), num_groups),
                {"gamma": make_affine(gamma), "beta": make_affine(beta)},
            ),
        )

    native = lambda *a, **k: op_pkg.groupnorm_sc_N_1_HW_C_b(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", groupnorm_sc_N_1_HW_C_b_generic.groupnorm_sc_N_1_HW_C_b),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
