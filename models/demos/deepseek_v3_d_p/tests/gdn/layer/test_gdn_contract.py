# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public contract of ttGDN on the toy GDN layer: shapes, dtypes and placement of state and output, read-only input
state, and construction-time rejection of unsupported geometries. Accuracy is owned by test_gdn_accuracy.py."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
    LAYOUTS,
    TOY_GDN_CONFIG,
    build_gdn_case,
    make_gdn_device_case,
    registered_gdn_case,
    synthetic_gdn_weights,
)
from models.demos.deepseek_v3_d_p.tests.gdn.device_utils import gdn_device_params, gdn_local_widths
from models.demos.deepseek_v3_d_p.tests.kda.utils import to_sp_input
from models.demos.deepseek_v3_d_p.tt.gdn.gdn import ttGDN
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.linear_attention import KdaState
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = run_for_blackhole()

_CONTRACT_LAYOUTS = ("1x1", "LB-A")


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


def _assert_state_metadata(state: KdaState, value_heads: int, convolution_width: int, key_dim: int) -> None:
    assert tuple(state.recurrent.shape) == (1, value_heads, key_dim, key_dim)
    assert state.recurrent.dtype == ttnn.float32
    assert state.recurrent.layout == ttnn.TILE_LAYOUT
    assert state.recurrent.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(state.convolution.shape) == (1, 3, convolution_width)
    assert state.convolution.dtype == ttnn.bfloat16
    assert state.convolution.layout == ttnn.ROW_MAJOR_LAYOUT
    assert state.convolution.memory_config() == ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "mesh_device,device_params,layout",
    [
        pytest.param(LAYOUTS[layout][0], gdn_device_params(LAYOUTS[layout][0]), layout, id=layout)
        for layout in _CONTRACT_LAYOUTS
    ],
    indirect=["mesh_device", "device_params"],
)
def test_gdn_state_output_contract_and_read_only_input_state(
    mesh_device: ttnn.MeshDevice, device_params: dict, layout: str
) -> None:
    case = build_gdn_case(registered_gdn_case("toy", layout, "chained3"))
    spec, config = case.spec, case.config
    tensor_parallel_size = spec.mesh_shape[spec.tensor_parallel_axis]
    value_heads = config.num_value_heads // tensor_parallel_size
    convolution_width = sum(gdn_local_widths(config, tensor_parallel_size))
    layer = make_gdn_device_case(mesh_device, case)

    state = layer.allocate_state()
    _assert_state_metadata(state, value_heads, convolution_width, config.head_k_dim)
    local_rows = spec.chunk_tokens // spec.sequence_parallel_size
    for chunk in range(2):
        hidden = to_sp_input(case.chunk_hidden(chunk), mesh_device, spec.sequence_parallel_axis)
        start = make_actual_start(mesh_device, chunk * spec.chunk_tokens)
        before = {"recurrent": _host_shards(state.recurrent), "convolution": _host_shards(state.convolution)}
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output, new_state = layer.forward(hidden, state, start)
        # The input state is only read: no tensor reachable from it changes, including a nonzero carry (chunk 1).
        for name in before:
            after = _host_shards(getattr(state, name))
            assert all(torch.equal(b, a) for b, a in zip(before[name], after, strict=True)), f"input {name} mutated"
        if chunk == 1:
            assert any(
                shard.abs().max() > 0 for shard in before["recurrent"]
            ), "chunk 1 must start from a nonzero carry"
        assert tuple(output.shape) == (1, local_rows, config.hidden_size // tensor_parallel_size)
        assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        _assert_state_metadata(new_state, value_heads, convolution_width, config.head_k_dim)
        assert new_state.recurrent.buffer_address() != state.recurrent.buffer_address()
        state = new_state


@pytest.mark.parametrize(
    "mesh_device,device_params,case_name,message",
    [
        pytest.param((1, 1), {}, "head_dim", "tile-aligned head dims", id="head-dims-must-be-tile-aligned"),
        pytest.param(
            (1, 1),
            {},
            "grouped_nonsquare",
            "grouped GDN affine prefix currently requires K == V",
            id="grouped-needs-k-eq-v",
        ),
        pytest.param((1, 1), {}, "sequence", "positive tile-aligned local physical length", id="sequence-tile-aligned"),
        pytest.param(
            (1, 1), {}, "weight_sources", "either constructed GDNWeights or host state_dict", id="one-weight-source"
        ),
        # No fabric: the TP check fires before any collective or weight placement.
        pytest.param((1, 4), {}, "tensor_parallel", "whole K-head groups", id="tp-must-split-whole-key-groups"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_gdn_rejects_unsupported_construction(
    mesh_device: ttnn.MeshDevice, device_params: dict, case_name: str, message: str, expect_error
) -> None:
    config = TOY_GDN_CONFIG
    kwargs = {"program_config": KDAProgramConfig(), "active_seq_len": 640}
    weights = None
    if case_name == "head_dim":
        config = replace(config, head_k_dim=48, head_v_dim=48)
    elif case_name == "grouped_nonsquare":
        config = replace(config, head_v_dim=64)
        kwargs["program_config"] = KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(local_scan_strategy="grouped", summary_group_chunks=10)
        )
    elif case_name == "sequence":
        kwargs["active_seq_len"] = 650
    elif case_name == "weight_sources":
        weights = object()
    else:
        config = replace(config, num_key_heads=2, num_value_heads=6)
    with expect_error(ValueError, message):
        ttGDN(mesh_device, config, synthetic_gdn_weights(config), weights=weights, **kwargs)
