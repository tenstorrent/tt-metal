# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Galaxy correctness tests for the dense Llama-3.1 SwiGLU MLP."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.device_utils import addresses as _device_addresses
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics as _metrics
from models.demos.llama_3p1_8b_d_p.tests.utils import read_raw_weights
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.mlp import MLP

HF_MODEL = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))
MESH_SHAPE = (4, 8)
SP = MESH_SHAPE[0]
TP = MESH_SHAPE[1]
GLOBAL_CHUNK = 1024
LOCAL_SEQUENCE = GLOBAL_CHUNK // SP
HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
INTERMEDIATE_SIZE = Llama31_8BConfig.INTERMEDIATE_SIZE
LOCAL_INTERMEDIATE = INTERMEDIATE_SIZE // TP
MLP_WEIGHT_NAMES = {
    "gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
    "up_proj.weight": "model.layers.0.mlp.up_proj.weight",
    "down_proj.weight": "model.layers.0.mlp.down_proj.weight",
}
MATMUL_L1_BYTES_PER_CORE = {"gate": 196_608, "up": 196_608, "down": 376_832}
MUL_L1_BYTES_PER_CORE = 14_336


def _load_layer_zero_mlp_weights():
    return read_raw_weights(HF_MODEL, MLP_WEIGHT_NAMES)


def _structured_weights():
    gate = torch.zeros(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    up = torch.zeros_like(gate)
    down = torch.zeros(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16)
    intermediate = torch.arange(INTERMEDIATE_SIZE)
    partition = intermediate // LOCAL_INTERMEDIATE
    gate_columns = intermediate % HIDDEN_SIZE
    up_columns = (intermediate * 13 + 17) % HIDDEN_SIZE
    output_rows = (intermediate * 29 + 11) % HIDDEN_SIZE
    gate_scale = torch.where(partition % 2 == 0, 1.25, -1.125).to(torch.bfloat16)
    up_scale = (0.5 + (intermediate % 7).float() * 0.125).to(torch.bfloat16)
    down_scale = (0.0078125 * (1 + partition % 4)).to(torch.bfloat16)
    gate[intermediate, gate_columns] = gate_scale
    up[intermediate, up_columns] = up_scale
    down[output_rows, intermediate] = down_scale
    return {
        "gate_proj.weight": gate,
        "up_proj.weight": up,
        "down_proj.weight": down,
    }


def _structured_input():
    columns = torch.linspace(-12.0, 12.0, HIDDEN_SIZE, dtype=torch.float32)
    rows = torch.arange(GLOBAL_CHUNK, dtype=torch.float32)
    row_offsets = ((rows % 29) - 14)[:, None] / 64 + (rows // LOCAL_SEQUENCE)[:, None] / 32
    return (columns[None, :] + row_offsets).reshape(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE)


def _real_input():
    torch.manual_seed(20260915)
    values = torch.randn(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE) * 0.2
    for sp_coord in range(SP):
        start = sp_coord * LOCAL_SEQUENCE
        values[:, :, start : start + LOCAL_SEQUENCE, :] += sp_coord * 0.03125
    return values


def _reference_mlp(host_input, state_dict):
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        pretraining_tp=1,
    )
    with torch.device("meta"):
        reference = LlamaMLP(config)
    reference.gate_proj.weight = torch.nn.Parameter(
        state_dict["gate_proj.weight"].to(torch.bfloat16).float(), requires_grad=False
    )
    reference.up_proj.weight = torch.nn.Parameter(
        state_dict["up_proj.weight"].to(torch.bfloat16).float(), requires_grad=False
    )
    reference.down_proj.weight = torch.nn.Parameter(
        state_dict["down_proj.weight"].to(torch.bfloat16).float(), requires_grad=False
    )
    reference.eval()
    with torch.no_grad():
        return reference(host_input.to(torch.bfloat16).float())


def _to_device_input(mesh_device, host_input, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        host_input.to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, None)),
    )


def _assert_live_tp_fabric(mesh_device):
    logger.info(f"MLP live mesh shape={tuple(mesh_device.shape)}, devices={mesh_device.get_num_devices()}")
    assert tuple(mesh_device.shape) == MESH_SHAPE
    for sp_coord in range(SP):
        for tp_coord in range(TP):
            src_coord = ttnn.MeshCoordinate([sp_coord, tp_coord])
            src_node = mesh_device.get_fabric_node_id(src_coord)
            for neighbor_tp in ((tp_coord - 1) % TP, (tp_coord + 1) % TP):
                dst_coord = ttnn.MeshCoordinate([sp_coord, neighbor_tp])
                dst_node = mesh_device.get_fabric_node_id(dst_coord)
                link_indices = tuple(ttnn.get_forwarding_link_indices(src_node, dst_node))
                logger.info(
                    f"MLP TP edge src={src_coord}/{src_node} dst={dst_coord}/{dst_node} "
                    f"forwarding_link_indices={link_indices}"
                )
                assert 0 in link_indices and 1 in link_indices


def _assert_synthetic_sensitivity(host_input, state_dict):
    sample = host_input[:, :, :1, :].to(torch.bfloat16).float()
    gate_weight = state_dict["gate_proj.weight"].float()
    up_weight = state_dict["up_proj.weight"].float()
    down_weight = state_dict["down_proj.weight"].float()
    gate = F.linear(sample, gate_weight)
    up = F.linear(sample, up_weight)
    assert gate.min() < -7.0 and gate.max() > 7.0

    product = F.silu(gate) * up
    partials = []
    for tp_coord in range(TP):
        start = tp_coord * LOCAL_INTERMEDIATE
        end = start + LOCAL_INTERMEDIATE
        partial = F.linear(product[..., start:end], down_weight[:, start:end])
        assert torch.count_nonzero(partial) > 0, f"TP partition {tp_coord} made no contribution"
        partials.append(partial)
    expected = sum(partials)
    assert not torch.allclose(partials[0], expected), "fixture cannot detect omitted TP reduction"

    swapped = F.linear(F.silu(up) * gate, down_weight)
    assert not torch.allclose(swapped, expected), "fixture cannot detect swapped gate/up projections"
    clamped = F.linear(F.silu(gate.clamp(max=7.0)) * up.clamp(min=-7.0, max=7.0), down_weight)
    assert not torch.allclose(clamped, expected), "fixture cannot detect GPT-style clamped SwiGLU"


def _run_mlp_case(mesh_device, mlp, host_input, expected, *, label):
    rounded_input = host_input.to(torch.bfloat16)
    tt_input = _to_device_input(mesh_device, rounded_input)
    input_addresses = _device_addresses(tt_input)
    input_before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tt_input)]
    tt_output = mlp(tt_input)
    ttnn.synchronize_device(mesh_device)
    output_addresses = _device_addresses(tt_output)

    assert tuple(tt_output.shape) == (1, 1, LOCAL_SEQUENCE, HIDDEN_SIZE)
    assert tt_output.dtype == ttnn.bfloat16
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert all(in_addr != out_addr for in_addr, out_addr in zip(input_addresses, output_addresses))

    output_shards = ttnn.get_device_tensors(tt_output)
    input_shards = ttnn.get_device_tensors(tt_input)
    errors = []
    for sp_coord in range(SP):
        seq_start = sp_coord * LOCAL_SEQUENCE
        expected_shard = expected[:, :, seq_start : seq_start + LOCAL_SEQUENCE, :]
        expected_input = rounded_input[:, :, seq_start : seq_start + LOCAL_SEQUENCE, :]
        tp_outputs = []
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            actual = ttnn.to_torch(output_shards[device_idx])[:, :, :LOCAL_SEQUENCE, :HIDDEN_SIZE]
            after = ttnn.to_torch(input_shards[device_idx])[:, :, :LOCAL_SEQUENCE, :HIDDEN_SIZE]
            assert torch.equal(input_before[device_idx][:, :, :LOCAL_SEQUENCE, :HIDDEN_SIZE], after)
            assert torch.equal(after, expected_input), f"{label} device={device_idx}: input/SP rows changed"
            assert torch.isfinite(actual).all(), f"{label} device={device_idx}: nonfinite output"
            tp_outputs.append(actual)
            if torch.count_nonzero(expected_shard) == 0:
                assert torch.count_nonzero(actual) == 0, f"{label} device={device_idx}: zero was not exact"
                errors.append((device_idx, 1.0, 0.0))
            else:
                pcc, nl2 = _metrics(expected_shard, actual)
                errors.append((device_idx, pcc, nl2))
                assert pcc >= 0.999, f"{label} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
                assert nl2 <= 0.02, f"{label} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
        assert all(
            torch.equal(tp_outputs[0], output) for output in tp_outputs[1:]
        ), f"{label} SP row {sp_coord}: TP outputs are not replicated"

    logger.info(
        f"MLP {label}: min_PCC={min(item[1] for item in errors):.7f}, "
        f"max_NL2={max(item[2] for item in errors):.7f} over {len(errors)} devices; "
        f"input_addresses={[hex(address) for address in input_addresses]}; "
        f"output_addresses={[hex(address) for address in output_addresses]}"
    )
    tt_output.deallocate(True)
    tt_input.deallocate(True)
    return input_addresses, output_addresses, min(item[1] for item in errors), max(item[2] for item in errors)


# Sum an exactly representable, independently constructed partial across TP on every SP row; all
# 32 outputs must match and replicate, catching a missing reduction or a reduction on the SP axis.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_tp_all_reduce_exact_sum_on_every_chip(mesh_device):
    _assert_live_tp_fabric(mesh_device)
    local_rows = torch.arange(LOCAL_SEQUENCE).reshape(LOCAL_SEQUENCE, 1)
    columns = torch.arange(HIDDEN_SIZE).reshape(1, HIDDEN_SIZE)
    shards = []
    for sp_coord in range(SP):
        for tp_coord in range(TP):
            values = 4 * (tp_coord + 1) + 32 * sp_coord + 4 * (local_rows % 2) + 4 * (columns % 2)
            shards.append(values.reshape(1, 1, LOCAL_SEQUENCE, HIDDEN_SIZE).to(torch.bfloat16))
    global_partial = torch.cat(
        [torch.cat(shards[sp_coord * TP : (sp_coord + 1) * TP], dim=3) for sp_coord in range(SP)], dim=2
    )
    tt_partial = ttnn.from_torch(
        global_partial,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 3)),
    )
    usable_topology = ttnn.get_usable_topology(tt_partial, topology=ttnn.Topology.Ring, cluster_axis=1)
    logger.info(f"MLP exact-sum usable_topology={usable_topology}")
    assert usable_topology == ttnn.Topology.Ring
    tt_sum = ttnn.all_reduce(
        tt_partial,
        cluster_axis=1,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
    ttnn.synchronize_device(mesh_device)

    output_shards = ttnn.get_device_tensors(tt_sum)
    assert len(output_shards) == SP * TP
    for sp_coord in range(SP):
        expected = (
            (
                sum(4 * (tp_coord + 1) for tp_coord in range(TP))
                + TP * 32 * sp_coord
                + TP * 4 * (local_rows % 2)
                + TP * 4 * (columns % 2)
            )
            .reshape(1, 1, LOCAL_SEQUENCE, HIDDEN_SIZE)
            .to(torch.bfloat16)
        )
        first = None
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            actual = ttnn.to_torch(output_shards[device_idx])[:, :, :LOCAL_SEQUENCE, :HIDDEN_SIZE]
            assert torch.equal(actual, expected), f"exact TP sum mismatch on device {device_idx}"
            if first is None:
                first = actual
            else:
                assert torch.equal(actual, first), f"TP result did not replicate on device {device_idx}"
    tt_sum.deallocate(True)
    tt_partial.deallocate(True)


# Run structured, zero, changed-input, return, and real checkpoint cases through the full MLP on
# all chips; this catches wrong SwiGLU math, sharding/reduction errors, stale addresses, and input reuse.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_dense_mlp_matches_transformers_and_reuses_programs(mesh_device, expect_error):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    synthetic_weights = _structured_weights()
    real_weights = _load_layer_zero_mlp_weights()
    synthetic_mlp = MLP(mesh_device, mesh_config, synthetic_weights)
    real_mlp = MLP(mesh_device, mesh_config, real_weights)

    assert tuple(synthetic_mlp.gate_weight.shape) == (HIDDEN_SIZE, LOCAL_INTERMEDIATE)
    assert tuple(synthetic_mlp.up_weight.shape) == (HIDDEN_SIZE, LOCAL_INTERMEDIATE)
    assert tuple(synthetic_mlp.down_weight.shape) == (LOCAL_INTERMEDIATE, HIDDEN_SIZE)
    for weight in (synthetic_mlp.gate_weight, synthetic_mlp.up_weight, synthetic_mlp.down_weight):
        assert weight.dtype == ttnn.bfloat16
        assert weight.layout == ttnn.TILE_LAYOUT
        assert weight.memory_config() == ttnn.DRAM_MEMORY_CONFIG

    logger.info(
        f"MLP runtime configs: compute={synthetic_mlp.compute_kernel_config}; "
        f"gate_up={synthetic_mlp.gate_up_program_config}; down={synthetic_mlp.down_program_config}; "
        f"validated_tp_edges={len(synthetic_mlp.fabric_links)}"
    )
    assert len(synthetic_mlp.fabric_links) == SP * TP * 2
    l1_before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    required_l1 = max(*MATMUL_L1_BYTES_PER_CORE.values(), MUL_L1_BYTES_PER_CORE)
    logger.info(
        f"MLP live L1 before launch: total_bytes_per_bank={l1_before.total_bytes_per_bank}, "
        f"largest_contiguous_bytes_free_per_bank={l1_before.largest_contiguous_bytes_free_per_bank}, "
        f"required_named_bytes_per_core={required_l1}, configs={MATMUL_L1_BYTES_PER_CORE}, "
        f"mul={MUL_L1_BYTES_PER_CORE}"
    )
    assert l1_before.total_bytes_per_bank >= required_l1
    assert l1_before.largest_contiguous_bytes_free_per_bank >= required_l1

    structured_input = _structured_input()
    changed_structured_input = torch.roll(structured_input, shifts=97, dims=-1) * 0.75
    real_input = _real_input()
    _assert_synthetic_sensitivity(structured_input, synthetic_weights)
    structured_expected = _reference_mlp(structured_input, synthetic_weights)
    changed_structured_expected = _reference_mlp(changed_structured_input, synthetic_weights)
    real_expected = _reference_mlp(real_input, real_weights)
    zero_input = torch.zeros_like(structured_input)
    zero_expected = torch.zeros_like(zero_input)

    mesh_device.enable_program_cache()
    cases = [
        ("synthetic-structured", synthetic_mlp, structured_input, structured_expected),
        ("checkpoint-real", real_mlp, real_input, real_expected),
        ("synthetic-changed", synthetic_mlp, changed_structured_input, changed_structured_expected),
        ("synthetic-zero", synthetic_mlp, zero_input, zero_expected),
        ("synthetic-structured-return", synthetic_mlp, structured_input, structured_expected),
    ]
    address_history = []
    cache_entries = None
    guards = []
    worst_pcc = 1.0
    worst_nl2 = 0.0
    for case_idx, (label, mlp, host_input, expected) in enumerate(cases):
        result = _run_mlp_case(mesh_device, mlp, host_input, expected, label=label)
        address_history.append(result[:2])
        worst_pcc = min(worst_pcc, result[2])
        worst_nl2 = max(worst_nl2, result[3])
        entries = mesh_device.num_program_cache_entries()
        if case_idx == 0:
            cache_entries = entries
            assert cache_entries > 0
            guards = [_to_device_input(mesh_device, torch.zeros_like(structured_input)) for _ in range(2)]
        else:
            assert entries == cache_entries, f"{label}: program cache changed {cache_entries} -> {entries}"

    first_addresses = set(address_history[0][0] + address_history[0][1])
    for case_idx, (input_addresses, output_addresses) in enumerate(address_history[1:], start=1):
        assert first_addresses.isdisjoint(
            input_addresses + output_addresses
        ), f"case {case_idx} reused a guarded first-call input/output address"
    logger.info(
        f"MLP cached-program reuse: entries={cache_entries} across {len(cases)} calls; "
        f"worst_PCC={worst_pcc:.7f}, worst_NL2={worst_nl2:.7f}"
    )
    for guard in guards:
        guard.deallocate(True)

    with expect_error(ValueError, "missing required weights"):
        MLP(mesh_device, mesh_config, {})
    wrong_shape = dict(synthetic_weights)
    wrong_shape["gate_proj.weight"] = torch.ones(1, 1)
    with expect_error(ValueError, "gate_proj.weight must have shape"):
        MLP(mesh_device, mesh_config, wrong_shape)
    with expect_error(ValueError, "requires mesh_shape"):
        MLP(mesh_device, MeshConfig((4, 4), 4), synthetic_weights)

    bad_shape_input = _to_device_input(mesh_device, structured_input[:, :, : GLOBAL_CHUNK - 128, :])
    with expect_error(ValueError, "input must have local shape"):
        synthetic_mlp(bad_shape_input)
    bad_shape_input.deallocate(True)
    bad_dtype_input = _to_device_input(mesh_device, structured_input, dtype=ttnn.float32)
    with expect_error(ValueError, "input must be bfloat16"):
        synthetic_mlp(bad_dtype_input)
    bad_dtype_input.deallocate(True)
    bad_layout_input = _to_device_input(mesh_device, structured_input, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(ValueError, "input must use TILE_LAYOUT"):
        synthetic_mlp(bad_layout_input)
    bad_layout_input.deallocate(True)
    valid_input = _to_device_input(mesh_device, structured_input)
    bad_memory_input = ttnn.to_memory_config(valid_input, ttnn.L1_MEMORY_CONFIG)
    with expect_error(ValueError, "input must use interleaved DRAM"):
        synthetic_mlp(bad_memory_input)
    bad_memory_input.deallocate(True)
    valid_input.deallocate(True)
