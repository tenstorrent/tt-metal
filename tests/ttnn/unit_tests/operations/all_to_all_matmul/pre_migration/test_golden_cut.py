# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut for all_to_all_matmul, run every round of the port.

Every value of every SUPPORTED axis appears in at least one case (combined freely, not their product), plus the
smallest and the largest golden INPUTS, the widest-N input and the LOOSE_CASES model configurations; each is built
and checked by the golden suite's own prepare_all_to_all_matmul (its reference, its bfloat8_b-as-held weight and its
tolerances), with the op called through the acceptance suite's ``all_to_all_matmul_op`` route fixture.

SUPPORTED: dtype bf16; weight_dtype bf16 | bf8b; layout TILE; fp32_dest_acc_en True | False; alignment tile_aligned |
m_non_aligned; output_buffer DRAM | L1; memory_layout INTERLEAVED; cluster_axis 1; topology Ring; num_links 2.
The LOOSE_CASES run at this mesh's packet size (their 7232 B payload pin changes only performance; the acceptance
suite's small-packet case covers a non-default payload).
"""

from __future__ import annotations

import pytest
import ttnn

from eval.golden_tests.all_to_all_matmul.feature_spec import GROUP, LOOSE_CASES
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul
from ttnn.operations.ccl import Topology

_AXES = dict(
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    cluster_axis=1,
    topology=Topology.Ring,
    num_links=2,
)
BF16, BF8B, DRAM, L1 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.BufferType.DRAM, ttnn.BufferType.L1

SMALLEST = ((1, 4, 128, 32), (1, 16, 32, 32), GROUP)
LARGEST = ((1, 32, 640, 128), (1, 128, 128, 512), GROUP)  # DeepSeek-V3.2: the most work per call
WIDE_N = ((1, 4, 128, 128), (1, 16, 128, 8192), GROUP)
DEEP_K = ((1, 4, 128, 4096), (1, 16, 4096, 128), GROUP)
NA_40 = ((1, 24, 160, 128), (1, 96, 128, 512), GROUP)
NA_12 = ((1, 8, 48, 64), (1, 32, 64, 128), GROUP)


def _loose(i):
    case = LOOSE_CASES[i]
    extras = {k: v for k, v in case["extras"].items() if k == "math_fidelity"}
    axes = {k: case[k] for k in ("weight_dtype", "fp32_dest_acc_en", "output_buffer")}
    return case["inputs"], axes, extras


# (inputs, {weight_dtype, fp32_dest_acc_en, output_buffer}, extras)
CUT = [
    pytest.param(SMALLEST, dict(weight_dtype=BF16, fp32_dest_acc_en=True, output_buffer=DRAM), {}, id="smallest"),
    pytest.param(
        SMALLEST, dict(weight_dtype=BF8B, fp32_dest_acc_en=False, output_buffer=L1), {}, id="smallest_bf8b_16dest_l1"
    ),
    pytest.param(
        LARGEST, dict(weight_dtype=BF16, fp32_dest_acc_en=True, output_buffer=DRAM), {}, id="largest_deepseek"
    ),
    pytest.param(
        LARGEST,
        dict(weight_dtype=BF8B, fp32_dest_acc_en=False, output_buffer=L1),
        {},
        id="largest_deepseek_bf8b_16dest_l1",
    ),
    pytest.param(WIDE_N, dict(weight_dtype=BF8B, fp32_dest_acc_en=True, output_buffer=DRAM), {}, id="wide_n8192_bf8b"),
    pytest.param(
        DEEP_K, dict(weight_dtype=BF16, fp32_dest_acc_en=False, output_buffer=DRAM), {}, id="deep_k4096_16dest"
    ),
    pytest.param(
        NA_40, dict(weight_dtype=BF16, fp32_dest_acc_en=False, output_buffer=L1), {}, id="non_aligned_40_rows_16dest_l1"
    ),
    pytest.param(
        NA_12, dict(weight_dtype=BF8B, fp32_dest_acc_en=True, output_buffer=DRAM), {}, id="non_aligned_12_rows_bf8b"
    ),
    pytest.param(*_loose(0), id="loose_kimi_model_hifi2"),
    pytest.param(*_loose(1), id="loose_kimi_phase0"),
    pytest.param(*_loose(2), id="loose_deepseek_model_hifi2"),
    pytest.param(*_loose(3), id="loose_glm_model_hifi2"),
]


@pytest.mark.parametrize("inputs,axes,cut_extras", CUT)
def test_golden_cut(mesh_device, all_to_all_matmul_op, inputs, axes, cut_extras):
    def op(*args, **kwargs):
        out = all_to_all_matmul_op(*args, **kwargs)
        if all_to_all_matmul_op.mode == "source":
            assert all_to_all_matmul_op.last_dispatches == 1
        mc = out.memory_config()
        assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED and mc.buffer_type == axes["output_buffer"], mc
        return out

    call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh_device, extras=cut_extras, op=op, **_AXES, **axes)
    check(call())
