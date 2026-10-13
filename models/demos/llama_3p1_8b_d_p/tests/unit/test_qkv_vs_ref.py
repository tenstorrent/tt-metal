# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy correctness tests for Llama-3.1 Q/K/V projection and head splitting."""

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
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.qkv import QKVProjection

HF_MODEL = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))
MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
GLOBAL_CHUNK = 1024
LOCAL_SEQUENCE = GLOBAL_CHUNK // SP
HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
HEAD_DIM = Llama31_8BConfig.HEAD_DIM
NUM_Q_HEADS = Llama31_8BConfig.NUM_ATTENTION_HEADS
NUM_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
LOCAL_Q_HEADS = NUM_Q_HEADS // TP
LOCAL_KV_HEADS = NUM_KV_HEADS // TP
LOCAL_QKV_WIDTH = (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS) * HEAD_DIM
QKV_WEIGHT_NAMES = {
    "q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
    "k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
    "v_proj.weight": "model.layers.0.self_attn.v_proj.weight",
}
QKV_MATMUL_L1_BYTES_PER_CORE = 196_608


def _load_layer_zero_qkv_weights():
    return read_raw_weights(HF_MODEL, QKV_WEIGHT_NAMES)


def _structured_projection(out_features, *, multiplier, offset, sign_period):
    rows = torch.arange(out_features)
    weight = torch.zeros(out_features, HIDDEN_SIZE, dtype=torch.bfloat16)
    columns = (rows * multiplier + offset) % HIDDEN_SIZE
    magnitudes = 0.25 + (rows % 11).float() / 32
    signs = torch.where((rows // sign_period) % 2 == 0, 1.0, -1.0)
    weight[rows, columns] = (magnitudes * signs).to(torch.bfloat16)
    return weight


def _structured_weights():
    return {
        "q_proj.weight": _structured_projection(HIDDEN_SIZE, multiplier=17, offset=3, sign_period=61),
        "k_proj.weight": _structured_projection(NUM_KV_HEADS * HEAD_DIM, multiplier=29, offset=7, sign_period=37),
        "v_proj.weight": _structured_projection(NUM_KV_HEADS * HEAD_DIM, multiplier=43, offset=19, sign_period=23),
    }


def _structured_input():
    columns = torch.arange(HIDDEN_SIZE, dtype=torch.float32)
    rows = torch.arange(GLOBAL_CHUNK, dtype=torch.float32)
    values = ((columns % 31) - 15)[None, :] / 8
    values = values + ((rows % 17) - 8)[:, None] / 16
    values = values + (rows // LOCAL_SEQUENCE)[:, None] / 4
    return values.reshape(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE)


def _real_input():
    values = torch.randn(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE, generator=torch.Generator().manual_seed(20260915))
    values *= 0.125
    for sp_coord in range(SP):
        values[:, :, sp_coord * LOCAL_SEQUENCE : (sp_coord + 1) * LOCAL_SEQUENCE] += sp_coord / 32
    return values


def _half_split_to_adjacent_independent(tensor):
    half = tensor.shape[-1] // 2
    return torch.stack((tensor[..., :half], tensor[..., half:]), dim=-1).reshape(tensor.shape)


def _reference_qkv(host_input, state_dict):
    x = host_input.to(torch.bfloat16).float()
    q_hf = F.linear(x, state_dict["q_proj.weight"].to(torch.bfloat16).float())
    k_hf = F.linear(x, state_dict["k_proj.weight"].to(torch.bfloat16).float())
    v = F.linear(x, state_dict["v_proj.weight"].to(torch.bfloat16).float())
    q = _half_split_to_adjacent_independent(q_hf.reshape(1, 1, GLOBAL_CHUNK, NUM_Q_HEADS, HEAD_DIM))
    k = _half_split_to_adjacent_independent(k_hf.reshape(1, 1, GLOBAL_CHUNK, NUM_KV_HEADS, HEAD_DIM))
    q = q.squeeze(1).transpose(1, 2).contiguous()
    k = k.squeeze(1).transpose(1, 2).contiguous()
    v = v.reshape(1, GLOBAL_CHUNK, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    return q, k, v


def _packed_local_weights_independent(state_dict, tp_coord):
    q = state_dict["q_proj.weight"].to(torch.bfloat16)
    k = state_dict["k_proj.weight"].to(torch.bfloat16)
    v = state_dict["v_proj.weight"].to(torch.bfloat16)
    q = q.reshape(NUM_Q_HEADS, HEAD_DIM, HIDDEN_SIZE).transpose(-2, -1)
    k = k.reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE).transpose(-2, -1)
    q = _half_split_to_adjacent_independent(q).transpose(-2, -1)
    k = _half_split_to_adjacent_independent(k).transpose(-2, -1)
    q0 = tp_coord * LOCAL_Q_HEADS
    q_local = q[q0 : q0 + LOCAL_Q_HEADS].reshape(LOCAL_Q_HEADS * HEAD_DIM, HIDDEN_SIZE)
    k_local = k[tp_coord].reshape(HEAD_DIM, HIDDEN_SIZE)
    v_local = v.reshape(NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE)[tp_coord]
    return torch.cat((q_local, k_local, v_local), dim=0).transpose(-2, -1).contiguous()


def _to_input(mesh_device, host_input, *, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        host_input.to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, None)),
    )


def _to_heads(mesh_device, host_heads):
    return ttnn.from_torch(
        host_heads.to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _assert_packed_weights(qkv, state_dict):
    shards = ttnn.get_device_tensors(qkv.qkv_weight)
    assert len(shards) == SP * TP
    assert tuple(qkv.qkv_weight.shape) == (1, 1, HIDDEN_SIZE, LOCAL_QKV_WIDTH)
    for sp_coord in range(SP):
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            actual = ttnn.to_torch(shards[device_idx])[0, 0, :HIDDEN_SIZE, :LOCAL_QKV_WIDTH]
            assert torch.equal(actual, _packed_local_weights_independent(state_dict, tp_coord))


def _run_case(mesh_device, projection, host_input, expected, *, label):
    tt_input = _to_input(mesh_device, host_input)
    input_shards = ttnn.get_device_tensors(tt_input)
    input_before = [ttnn.to_torch(shard).clone() for shard in input_shards]
    input_addresses = _addresses(tt_input)
    outputs = projection(tt_input)
    ttnn.synchronize_device(mesh_device)
    shapes = (
        (1, LOCAL_Q_HEADS, LOCAL_SEQUENCE, HEAD_DIM),
        (1, LOCAL_KV_HEADS, LOCAL_SEQUENCE, HEAD_DIM),
        (1, LOCAL_KV_HEADS, LOCAL_SEQUENCE, HEAD_DIM),
    )
    for output, shape in zip(outputs, shapes):
        assert tuple(output.shape) == shape
        assert output.dtype == ttnn.bfloat16
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    output_addresses = tuple(address for output in outputs for address in _addresses(output))
    assert set(input_addresses).isdisjoint(output_addresses)
    output_shards = [ttnn.get_device_tensors(output) for output in outputs]
    errors = []
    for sp_coord in range(SP):
        seq_slice = slice(sp_coord * LOCAL_SEQUENCE, (sp_coord + 1) * LOCAL_SEQUENCE)
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            assert torch.equal(ttnn.to_torch(input_shards[device_idx]), input_before[device_idx])
            for shards, reference, name in zip(output_shards, expected, ("Q", "K", "V")):
                count = LOCAL_Q_HEADS if name == "Q" else LOCAL_KV_HEADS
                actual = ttnn.to_torch(shards[device_idx]).float()[:, :count, :LOCAL_SEQUENCE, :HEAD_DIM]
                wanted = reference[:, tp_coord * count : (tp_coord + 1) * count, seq_slice, :]
                if torch.count_nonzero(wanted) == 0:
                    assert torch.count_nonzero(actual) == 0
                    pcc, nl2 = 1.0, 0.0
                else:
                    pcc, nl2 = _metrics(wanted, actual)
                    assert pcc >= 0.9999, f"{label} {name} chip={device_idx} PCC={pcc:.7f} NL2={nl2:.7f}"
                    assert nl2 <= 0.01, f"{label} {name} chip={device_idx} PCC={pcc:.7f} NL2={nl2:.7f}"
                errors.append((pcc, nl2))
    logger.info(
        f"QKV {label}: min_PCC={min(x[0] for x in errors):.7f}, max_NL2={max(x[1] for x in errors):.7f}; "
        f"input_addresses={[hex(x) for x in input_addresses]}; "
        f"output_addresses={[hex(x) for x in output_addresses]}"
    )
    for output in outputs:
        output.deallocate(True)
    tt_input.deallocate(True)
    return input_addresses + output_addresses


# Project structured, zero, changed, and real layer-0 data on all 32 chips, then revisit the first
# program with different allocations; this catches wrong TP group packing, head order, Q/K frame,
# swapped K/V, stale addresses, input mutation, and an accidentally rebuilt program geometry.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_qkv_projection_matches_independent_reference_and_reuses_programs(mesh_device, expect_error):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    synthetic_weights = _structured_weights()
    real_weights = _load_layer_zero_qkv_weights()
    structured_input = _structured_input()
    changed_input = torch.roll(structured_input, shifts=113, dims=-1) * 0.75
    real_input = _real_input()
    synthetic_expected = _reference_qkv(structured_input, synthetic_weights)
    changed_expected = _reference_qkv(changed_input, synthetic_weights)
    real_expected = _reference_qkv(real_input, real_weights)
    zero_expected = tuple(torch.zeros_like(tensor) for tensor in synthetic_expected)
    raw_q = F.linear(structured_input.to(torch.bfloat16).float(), synthetic_weights["q_proj.weight"].float())
    raw_q = raw_q.reshape(1, GLOBAL_CHUNK, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    assert not torch.equal(synthetic_expected[0], raw_q)
    assert not torch.equal(synthetic_expected[1], synthetic_expected[2])

    synthetic_qkv = QKVProjection(mesh_device, mesh_config, synthetic_weights)
    real_qkv = QKVProjection(mesh_device, mesh_config, real_weights)
    _assert_packed_weights(synthetic_qkv, synthetic_weights)
    _assert_packed_weights(real_qkv, real_weights)
    logger.info(f"QKV configs: compute={synthetic_qkv.compute_kernel_config}; matmul={synthetic_qkv.program_config}")
    l1 = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    logger.info(
        f"QKV L1 total={l1.total_bytes_per_bank}, free={l1.largest_contiguous_bytes_free_per_bank}, "
        f"required={QKV_MATMUL_L1_BYTES_PER_CORE}"
    )
    assert l1.total_bytes_per_bank >= QKV_MATMUL_L1_BYTES_PER_CORE
    assert l1.largest_contiguous_bytes_free_per_bank >= QKV_MATMUL_L1_BYTES_PER_CORE

    mesh_device.enable_program_cache()
    cases = [
        ("synthetic", synthetic_qkv, structured_input, synthetic_expected),
        ("real", real_qkv, real_input, real_expected),
        ("changed", synthetic_qkv, changed_input, changed_expected),
        ("zero", synthetic_qkv, torch.zeros_like(structured_input), zero_expected),
        ("return", synthetic_qkv, structured_input, synthetic_expected),
    ]
    first_addresses = None
    cache_entries = None
    guards = []
    for index, (label, projection, host_input, expected) in enumerate(cases):
        addresses = set(_run_case(mesh_device, projection, host_input, expected, label=label))
        if index == 0:
            first_addresses = addresses
            guards = [
                _to_input(mesh_device, torch.zeros_like(structured_input)),
                _to_heads(mesh_device, torch.zeros_like(synthetic_expected[0])),
                _to_heads(mesh_device, torch.zeros_like(synthetic_expected[1])),
                _to_heads(mesh_device, torch.zeros_like(synthetic_expected[2])),
            ]
        elif index == 1:
            assert first_addresses.isdisjoint(addresses)
            cache_entries = mesh_device.num_program_cache_entries()
            assert cache_entries > 0
        else:
            assert first_addresses.isdisjoint(addresses)
            assert mesh_device.num_program_cache_entries() == cache_entries
    for guard in guards:
        guard.deallocate(True)

    with expect_error(ValueError, "q_proj.weight must have shape"):
        bad = dict(synthetic_weights)
        bad["q_proj.weight"] = torch.zeros(1, 1)
        QKVProjection(mesh_device, mesh_config, bad)
    wrong_mesh = SimpleNamespace(mesh_shape=(8, 4), sp=8, tp=4, sp_axis=0, tp_axis=1)
    with expect_error(ValueError, "requires mesh_shape"):
        QKVProjection(mesh_device, wrong_mesh, synthetic_weights)
    wrong_shape = _to_input(mesh_device, torch.zeros(1, 1, 128, HIDDEN_SIZE))
    with expect_error(ValueError, "local shape"):
        synthetic_qkv(wrong_shape)
    wrong_shape.deallocate(True)
    wrong_dtype = _to_input(mesh_device, structured_input, dtype=ttnn.bfloat8_b)
    with expect_error(ValueError, "bfloat16"):
        synthetic_qkv(wrong_dtype)
    wrong_dtype.deallocate(True)
