# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Galaxy correctness tests for Llama-3.1 indexed RoPE."""

import os
from functools import partial

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics
from models.demos.llama_3p1_8b_d_p.tt import rope

_metrics = partial(metrics, dtype=torch.float32)

HF_MODEL = os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct")
MESH_SHAPE = (4, 8)
SP_AXIS = 0
SP = 4
TP = 8
HEAD_DIM = 128
PHYSICAL_CHUNK = 1024
LOCAL_CHUNK = PHYSICAL_CHUNK // SP
MAX_SEQ_LEN = 2048
TABLE_CAPACITY = 3072
STARTS = (0, 32, 224, 256, 768, 1024, 2016, 224)


def _hf_to_meta_independent(tensor):
    half = tensor.shape[-1] // 2
    return torch.stack((tensor[..., :half], tensor[..., half:]), dim=-1).flatten(-2)


def _owned_positions(start):
    """Enumerate the interval, assign slab owners, and preserve encounter order per owner."""
    owned = [[] for _ in range(SP)]
    for position in range(start, start + PHYSICAL_CHUNK):
        owned[(position // LOCAL_CHUNK) % SP].append(position)
    assert all(len(rows) == LOCAL_CHUNK for rows in owned)
    return owned


def _assert_all_device_shards(tt_tensor, expected_meta, owned, *, heads_per_tp, label, start):
    device_tensors = ttnn.get_device_tensors(tt_tensor)
    assert len(device_tensors) == SP * TP
    errors = []
    for sp_coord in range(SP):
        assert len(owned[sp_coord]) == LOCAL_CHUNK
        seq_start = sp_coord * LOCAL_CHUNK
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            actual = ttnn.to_torch(device_tensors[device_idx]).to(torch.float32)
            h0 = tp_coord * heads_per_tp
            expected = expected_meta[:, h0 : h0 + heads_per_tp, seq_start : seq_start + LOCAL_CHUNK, :]
            actual = actual[:, :heads_per_tp, :LOCAL_CHUNK, :HEAD_DIM]
            pcc, nl2 = _metrics(expected, actual)
            errors.append((device_idx, pcc, nl2))
            assert pcc >= 0.9999, f"{label} start={start} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
            assert nl2 <= 0.01, f"{label} start={start} device={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
    logger.info(
        f"{label} start={start}: min_PCC={min(x[1] for x in errors):.7f}, "
        f"max_NL2={max(x[2] for x in errors):.7f} over {len(errors)} devices"
    )


def _assert_table_shards_match_hf(cos_tt, sin_tt, hf_rotary):
    all_positions = torch.arange(TABLE_CAPACITY, dtype=torch.long).unsqueeze(0)
    hf_cos, hf_sin = hf_rotary(torch.empty(1, dtype=torch.float32), all_positions)
    expected_cos = torch.repeat_interleave(hf_cos[..., : HEAD_DIM // 2], 2, dim=-1).unsqueeze(1)
    expected_sin = torch.repeat_interleave(hf_sin[..., : HEAD_DIM // 2], 2, dim=-1).unsqueeze(1)
    cos_devices = ttnn.get_device_tensors(cos_tt)
    sin_devices = ttnn.get_device_tensors(sin_tt)

    for sp_coord in range(SP):
        owned = [p for p in range(TABLE_CAPACITY) if (p // LOCAL_CHUNK) % SP == sp_coord]
        expected_cos_shard = expected_cos[:, :, owned, :]
        expected_sin_shard = expected_sin[:, :, owned, :]
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            actual_cos = ttnn.to_torch(cos_devices[device_idx]).float()[:, :, : len(owned), :HEAD_DIM]
            actual_sin = ttnn.to_torch(sin_devices[device_idx]).float()[:, :, : len(owned), :HEAD_DIM]
            cos_pcc, cos_nl2 = _metrics(expected_cos_shard, actual_cos)
            sin_pcc, sin_nl2 = _metrics(expected_sin_shard, actual_sin)
            assert cos_pcc >= 0.9999 and cos_nl2 <= 0.01
            assert sin_pcc >= 0.9999 and sin_nl2 <= 0.01


# Run reused Q/K programs across every required offset and all 32 chips; tight HF metrics catch stale indexing,
# wrong SP ownership, TP replication, Llama3 frequencies, Meta coordinates, and undersized padded-tail tables.
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_indexed_rope_matches_hf_on_every_q_and_k_shard(mesh_device, expect_error):
    """Catches wrong Llama3 frequencies, coordinate frame, SP ownership, TP mapping, or stale offsets."""
    torch.manual_seed(20260915)
    config = AutoConfig.from_pretrained(HF_MODEL)
    hf_rotary = LlamaRotaryEmbedding(config)

    rope_tables = rope.build_indexed_rope(
        mesh_device,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=PHYSICAL_CHUNK,
        sp_axis=SP_AXIS,
    )
    trans_mat = rope.build_transformation_mat(mesh_device)
    _assert_table_shards_match_hf(rope_tables[0], rope_tables[1], hf_rotary)

    input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1))
    from_torch_kwargs = dict(
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=input_mapper,
    )
    q_hf = torch.randn(1, 32, PHYSICAL_CHUNK, HEAD_DIM).to(torch.bfloat16).float()
    k_hf = torch.randn(1, 8, PHYSICAL_CHUNK, HEAD_DIM).to(torch.bfloat16).float()
    q_meta = _hf_to_meta_independent(q_hf).to(torch.bfloat16)
    k_meta = _hf_to_meta_independent(k_hf).to(torch.bfloat16)
    q_tt = ttnn.from_torch(q_meta, **from_torch_kwargs)
    k_tt = ttnn.from_torch(k_meta, **from_torch_kwargs)

    mesh_device.enable_program_cache()
    entries_after_first_shapes = None
    last_q_out = None
    for iteration, start in enumerate(STARTS):
        owned = _owned_positions(start)
        flat_positions = [position for device_rows in owned for position in device_rows]
        position_ids = torch.tensor([flat_positions], dtype=torch.long)
        hf_cos, hf_sin = hf_rotary(q_hf, position_ids)
        expected_q_hf, expected_k_hf = apply_rotary_pos_emb(q_hf, k_hf, hf_cos, hf_sin)
        expected_q_meta = _hf_to_meta_independent(expected_q_hf)
        expected_k_meta = _hf_to_meta_independent(expected_k_hf)

        q_out = rope.apply_indexed_rope(q_tt, rope_tables, trans_mat, kv_actual_global=start, sp_axis=SP_AXIS)
        k_out = rope.apply_indexed_rope(k_tt, rope_tables, trans_mat, kv_actual_global=start, sp_axis=SP_AXIS)
        ttnn.synchronize_device(mesh_device)
        _assert_all_device_shards(q_out, expected_q_meta, owned, heads_per_tp=4, label="Q", start=start)
        _assert_all_device_shards(k_out, expected_k_meta, owned, heads_per_tp=1, label="K", start=start)
        last_q_out = q_out

        if iteration == 0:
            entries_after_first_shapes = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == entries_after_first_shapes

        if start == 2016:
            logger.info(
                "start=2016 checks all physical rows through 3039; only positions [2016, 2048) "
                "are logically valid for the later runtime"
            )

    with expect_error(RuntimeError, "tile-aligned"):
        rope.apply_indexed_rope(q_tt, rope_tables, trans_mat, kv_actual_global=1, sp_axis=SP_AXIS)

    assert last_q_out is not None
