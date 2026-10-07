# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy correctness tests for Llama-3.1 attention output projection."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.device_utils import addresses as _addresses
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics as _metrics
from models.demos.llama_3p1_8b_d_p.tests.utils import read_raw_weights
from models.demos.llama_3p1_8b_d_p.tt.attention import AttentionOutputProjection
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig

HF_MODEL = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))
MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
GLOBAL_CHUNK = 1024
LOCAL_SEQUENCE = GLOBAL_CHUNK // SP
HEAD_DIM = Llama31_8BConfig.HEAD_DIM
NUM_HEADS = Llama31_8BConfig.NUM_ATTENTION_HEADS
LOCAL_HEADS = NUM_HEADS // TP
HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
O_PROJ_CHECKPOINT_NAME = "model.layers.0.self_attn.o_proj.weight"
O_MATMUL_L1_BYTES_PER_CORE = 376_832
CONCAT_L1_BYTES_PER_CORE = 65_536


def _load_layer_zero_o_weight():
    return read_raw_weights(HF_MODEL, {"o_proj.weight": O_PROJ_CHECKPOINT_NAME})


def _structured_weight():
    weight = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    columns = torch.arange(HIDDEN_SIZE)
    rows_a = (columns * 29 + 17) % HIDDEN_SIZE
    rows_b = (columns * 47 + 3) % HIDDEN_SIZE
    tp_partition = columns // (LOCAL_HEADS * HEAD_DIM)
    weight[rows_a, columns] = (0.03125 * (1 + tp_partition % 4)).to(torch.bfloat16)
    weight[rows_b, columns] += (-0.015625 * (1 + columns % 3)).to(torch.bfloat16)
    return {"o_proj.weight": weight}


def _structured_heads():
    positions = torch.arange(GLOBAL_CHUNK, dtype=torch.float32)
    heads = torch.arange(NUM_HEADS, dtype=torch.float32)
    dims = torch.arange(HEAD_DIM, dtype=torch.float32)
    values = (
        torch.sin((positions[None, :, None] + 1) * (dims[None, None, :] + 3) / 257.0)
        + heads[:, None, None] * 0.03125
        + (positions[None, :, None] % 19) * 0.00390625
    )
    return values.unsqueeze(0)


def _real_heads():
    generator = torch.Generator().manual_seed(20260916)
    heads = torch.randn(1, NUM_HEADS, GLOBAL_CHUNK, HEAD_DIM, generator=generator) * 0.2
    heads += torch.arange(NUM_HEADS, dtype=torch.float32)[None, :, None, None] * 0.001953125
    return heads


def _reference(host_heads, state_dict):
    rounded_heads = host_heads.to(torch.bfloat16).float()
    concatenated = rounded_heads.transpose(1, 2).reshape(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE)
    weight = state_dict["o_proj.weight"].to(torch.bfloat16).float()
    return F.linear(concatenated, weight)


def _to_heads(mesh_device, host_heads, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        host_heads.to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _run_case(mesh_device, projection, host_heads, expected, *, label, retain_output=False):
    tt_heads = _to_heads(mesh_device, host_heads)
    input_shards = ttnn.get_device_tensors(tt_heads)
    input_before = [ttnn.to_torch(shard).clone() for shard in input_shards]
    input_addresses = _addresses(tt_heads)
    output = projection(tt_heads)
    ttnn.synchronize_device(mesh_device)
    output_addresses = _addresses(output)

    assert tuple(output.shape) == (1, 1, LOCAL_SEQUENCE, HIDDEN_SIZE)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert set(input_addresses).isdisjoint(output_addresses)

    output_shards = ttnn.get_device_tensors(output)
    errors = []
    for sp_coord in range(SP):
        seq_slice = slice(sp_coord * LOCAL_SEQUENCE, (sp_coord + 1) * LOCAL_SEQUENCE)
        wanted = expected[:, :, seq_slice, :]
        tp_outputs = []
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            after = ttnn.to_torch(input_shards[device_idx])
            assert torch.equal(after, input_before[device_idx]), f"{label} device={device_idx}: input changed"
            actual = ttnn.to_torch(output_shards[device_idx]).float()[:, :, :LOCAL_SEQUENCE, :HIDDEN_SIZE]
            tp_outputs.append(actual)
            if torch.count_nonzero(wanted) == 0:
                assert torch.count_nonzero(actual) == 0
                pcc, nl2 = 1.0, 0.0
            else:
                pcc, nl2 = _metrics(wanted, actual)
                assert pcc >= 0.999, f"{label} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
                assert nl2 <= 0.02, f"{label} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
            errors.append((pcc, nl2))
        for replicated in tp_outputs[1:]:
            assert torch.equal(replicated, tp_outputs[0]), f"{label} SP={sp_coord}: TP outputs differ"

    logger.info(
        f"attention O {label}: min_PCC={min(x[0] for x in errors):.7f}, "
        f"max_NL2={max(x[1] for x in errors):.7f}; input={[hex(x) for x in input_addresses]}; "
        f"output={[hex(x) for x in output_addresses]}"
    )
    tt_heads.deallocate(True)
    if not retain_output:
        output.deallocate(True)
    return set(input_addresses + output_addresses), output if retain_output else None


# Project structured, changed, zero, and real layer-0 heads, switch weight instances and allocations,
# then return to the first case; this catches wrong head concatenation, TP row sharding, stale buffers,
# input mutation, missing TP reduction, weak validation, and unstable warm program reuse.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_attention_output_projection_matches_reference_and_replays(mesh_device, expect_error):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    synthetic_weights = _structured_weight()
    real_weights = _load_layer_zero_o_weight()
    structured = _structured_heads()
    changed = torch.roll(structured, shifts=(5, 71), dims=(1, 2)) * 0.75
    real = _real_heads()

    synthetic_projection = AttentionOutputProjection(mesh_device, mesh_config, synthetic_weights)
    real_projection = AttentionOutputProjection(mesh_device, mesh_config, real_weights)
    logger.info(
        f"attention O config: matmul={synthetic_projection.program_config}; "
        f"compute={synthetic_projection.compute_kernel_config}; "
        f"synthetic_weight={[hex(x) for x in _addresses(synthetic_projection.o_weight)]}; "
        f"real_weight={[hex(x) for x in _addresses(real_projection.o_weight)]}"
    )

    l1 = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    required_l1 = max(O_MATMUL_L1_BYTES_PER_CORE, CONCAT_L1_BYTES_PER_CORE)
    logger.info(
        f"attention O live L1 free={l1.total_bytes_free_per_bank}, "
        f"largest_contiguous={l1.largest_contiguous_bytes_free_per_bank}, required={required_l1}"
    )
    assert l1.total_bytes_free_per_bank >= 1024 * 1024
    assert l1.largest_contiguous_bytes_free_per_bank >= 1024 * 1024
    assert l1.largest_contiguous_bytes_free_per_bank >= required_l1

    mesh_device.enable_program_cache()
    cases = [
        ("structured", synthetic_projection, structured, _reference(structured, synthetic_weights)),
        ("real", real_projection, real, _reference(real, real_weights)),
        ("changed", synthetic_projection, changed, _reference(changed, synthetic_weights)),
        ("zero", synthetic_projection, torch.zeros_like(structured), torch.zeros(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE)),
        ("return", synthetic_projection, structured, _reference(structured, synthetic_weights)),
    ]
    first_addresses = None
    first_output = None
    warm_program_count = None
    guards = []
    for index, (label, projection, host_heads, expected) in enumerate(cases):
        addresses, retained_output = _run_case(
            mesh_device,
            projection,
            host_heads,
            expected,
            label=label,
            retain_output=index == 0,
        )
        if index == 0:
            first_addresses = addresses
            first_output = retained_output
            guards = [_to_heads(mesh_device, torch.zeros_like(structured))]
        elif index == 1:
            assert first_addresses.isdisjoint(addresses)
            warm_program_count = mesh_device.num_program_cache_entries()
            assert warm_program_count > 0
        else:
            assert first_addresses.isdisjoint(addresses)
            assert mesh_device.num_program_cache_entries() == warm_program_count
    for guard in guards:
        guard.deallocate(True)
    assert first_output is not None
    first_output.deallocate(True)

    with expect_error(ValueError, "o_proj.weight must have shape"):
        AttentionOutputProjection(mesh_device, mesh_config, {"o_proj.weight": torch.zeros(1, 1)})
    with expect_error(ValueError, "requires mesh_shape"):
        AttentionOutputProjection(
            mesh_device,
            SimpleNamespace(mesh_shape=(8, 4), sp=8, tp=4, sp_axis=0, tp_axis=1),
            synthetic_weights,
        )
    wrong_shape = _to_heads(mesh_device, structured[:, :, :512])
    with expect_error(ValueError, "local shape"):
        synthetic_projection(wrong_shape)
    wrong_shape.deallocate(True)
    wrong_dtype = _to_heads(mesh_device, structured, dtype=ttnn.bfloat8_b)
    with expect_error(ValueError, "must be DataType.BFLOAT16"):
        synthetic_projection(wrong_dtype)
    wrong_dtype.deallocate(True)

    _run_case(
        mesh_device,
        synthetic_projection,
        structured,
        _reference(structured, synthetic_weights),
        label="valid-after-errors",
    )
