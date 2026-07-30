# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.functional_decoder import (
    MODEL_ID,
    FunctionalDecoder,
    _LlamaMLP,
)
from models.common.auto_compose import to_torch_auto_compose
from models.common.tensor_utils import get_rot_transformation_mat
from models.common.utility_functions import nearest_32


ARTIFACT_DIR = Path("models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder")
LAYER_IDX = 0
PCC_THRESHOLD = 0.995
LAYER0_WEIGHT_STATS = {
    "input_layernorm.weight": (0.14212070405483246, 0.1614246964454651),
    "post_attention_layernorm.weight": (0.20326919853687286, 0.02741304039955139),
    "self_attn.q_proj.weight": (-1.6576341295149177e-06, 0.036056291311979294),
    "self_attn.k_proj.weight": (-2.900863728427794e-05, 0.0467258095741272),
    "self_attn.v_proj.weight": (1.6981493899947964e-07, 0.009176568128168583),
    "self_attn.o_proj.weight": (-2.563944235589588e-06, 0.01147487573325634),
    "mlp.gate_proj.weight": (1.6329495338140987e-05, 0.019302168861031532),
    "mlp.up_proj.weight": (-7.731669029453769e-07, 0.017218172550201416),
    "mlp.down_proj.weight": (-4.90650427309447e-06, 0.017107686027884483),
}


def _mesh_shape_tuple(mesh_device: ttnn.MeshDevice) -> tuple[int, int]:
    return int(mesh_device.shape[0]), int(mesh_device.shape[1])


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected = expected.float().flatten().double()
    actual = actual.float().flatten().double()
    expected = expected - expected.mean()
    actual = actual - actual.mean()
    denom = torch.linalg.vector_norm(expected) * torch.linalg.vector_norm(actual)
    if denom == 0:
        return 1.0 if torch.allclose(expected, actual) else 0.0
    return float(torch.clamp(torch.dot(expected, actual) / denom, min=-1.0, max=1.0))


def _assert_pcc(name: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float = PCC_THRESHOLD) -> float:
    pcc = _pcc(expected, actual)
    assert pcc >= threshold, f"{name} PCC {pcc:.6f} < {threshold:.6f}"
    return pcc


def _permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = torch.stack((cos[:, : cos.shape[1] // 2], cos[:, : cos.shape[1] // 2]), dim=-1).flatten(-2)
    sin = torch.stack((sin[:, : sin.shape[1] // 2], sin[:, : sin.shape[1] // 2]), dim=-1).flatten(-2)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _hf_and_meta_position_embeddings(
    rotary_emb: LlamaRotaryEmbedding,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    with torch.no_grad():
        cos_hf, sin_hf = rotary_emb(hidden_states, position_ids)
    cos_meta, sin_meta = _permute_to_meta_format(cos_hf.float(), sin_hf.float())
    return (cos_hf, sin_hf), (cos_meta.to(torch.bfloat16), sin_meta.to(torch.bfloat16))


def _meta_rot_mats_prefill(
    rotary_emb: LlamaRotaryEmbedding,
    hidden_states: torch.Tensor,
    start_pos: int,
    seq_len: int,
    mesh_device: ttnn.MeshDevice,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[ttnn.Tensor, ttnn.Tensor]]:
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
    hf_pos_emb, meta_pos_emb = _hf_and_meta_position_embeddings(rotary_emb, hidden_states, position_ids)
    cos_meta, sin_meta = meta_pos_emb
    cos_tt = ttnn.from_torch(
        cos_meta,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin_meta,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return hf_pos_emb, (cos_tt, sin_tt)


class DecodeRotaryHelper:
    def __init__(self, rotary_emb: LlamaRotaryEmbedding, max_seq_len: int, head_dim: int, mesh_device: ttnn.MeshDevice):
        self.mesh_device = mesh_device
        self.head_dim = head_dim
        dummy = torch.zeros(1, max_seq_len, head_dim, dtype=torch.bfloat16)
        position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        _, meta = _hf_and_meta_position_embeddings(rotary_emb, dummy, position_ids)
        cos_meta, sin_meta = meta
        self.cos_matrix = ttnn.from_torch(
            cos_meta,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_matrix = ttnn.from_torch(
            sin_meta,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        trans_mat = get_rot_transformation_mat().repeat(1, 1, 32, 1)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=ttnn.num_cores_to_corerangeset(32, mesh_device.compute_with_storage_grid_size(), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def get_rot_mats(self, position_idxs: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        batch = int(position_idxs.numel())
        padded = F.pad(position_idxs.reshape(1, batch).to(torch.int32), (0, nearest_32(batch) - batch))
        rot_idxs = ttnn.from_torch(
            padded,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        if batch % ttnn.TILE_SIZE != 0:
            cos = cos[:, :batch, :, :]
            sin = sin[:, :batch, :, :]
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(32, self.mesh_device.compute_with_storage_grid_size(), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.interleaved_to_sharded(cos, mem_config), ttnn.interleaved_to_sharded(sin, mem_config)


def _causal_mask(batch: int, query_len: int, key_len: int, *, past_len: int = 0) -> torch.Tensor:
    q_positions = torch.arange(past_len, past_len + query_len).unsqueeze(-1)
    k_positions = torch.arange(key_len).unsqueeze(0)
    blocked = k_positions > q_positions
    mask = torch.zeros(1, 1, query_len, key_len, dtype=torch.float32)
    mask.masked_fill_(blocked.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min)
    return mask.expand(batch, 1, query_len, key_len)


def _make_hf_layer_and_rotary(hf_config, state_dict: dict[str, torch.Tensor]) -> tuple[LlamaDecoderLayer, LlamaRotaryEmbedding]:
    layer = LlamaDecoderLayer(hf_config, LAYER_IDX).to(torch.bfloat16).eval()
    layer.load_state_dict({key.removeprefix(f"model.layers.{LAYER_IDX}."): value for key, value in state_dict.items()})
    return layer, LlamaRotaryEmbedding(hf_config)


def _synthetic_layer_state_dict(hf_config) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(20260615)
    h = hf_config.hidden_size
    kv = hf_config.num_key_value_heads * hf_config.head_dim
    intermediate = hf_config.intermediate_size

    def rand(name, shape):
        mean, std = LAYER0_WEIGHT_STATS[name]
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * std + mean).to(torch.bfloat16)

    return {
        f"model.layers.{LAYER_IDX}.self_attn.q_proj.weight": rand("self_attn.q_proj.weight", (h, h)),
        f"model.layers.{LAYER_IDX}.self_attn.k_proj.weight": rand("self_attn.k_proj.weight", (kv, h)),
        f"model.layers.{LAYER_IDX}.self_attn.v_proj.weight": rand("self_attn.v_proj.weight", (kv, h)),
        f"model.layers.{LAYER_IDX}.self_attn.o_proj.weight": rand("self_attn.o_proj.weight", (h, h)),
        f"model.layers.{LAYER_IDX}.mlp.gate_proj.weight": rand("mlp.gate_proj.weight", (intermediate, h)),
        f"model.layers.{LAYER_IDX}.mlp.up_proj.weight": rand("mlp.up_proj.weight", (intermediate, h)),
        f"model.layers.{LAYER_IDX}.mlp.down_proj.weight": rand("mlp.down_proj.weight", (h, intermediate)),
        f"model.layers.{LAYER_IDX}.input_layernorm.weight": rand("input_layernorm.weight", (h,)),
        f"model.layers.{LAYER_IDX}.post_attention_layernorm.weight": rand("post_attention_layernorm.weight", (h,)),
    }


def _real_layer_state_dict() -> dict[str, torch.Tensor]:
    snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True))
    model_file = snapshot / "model.safetensors"
    prefix = f"model.layers.{LAYER_IDX}."
    state = {}
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key] = f.get_tensor(key).to(torch.bfloat16)
    if not state:
        raise RuntimeError(f"no layer {LAYER_IDX} tensors found in {model_file}")
    return state


def _write_weight_stats(state_dict: dict[str, torch.Tensor]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": float(tensor.float().mean()),
            "std": float(tensor.float().std()),
        }
        for key, tensor in sorted(state_dict.items())
    }
    (ARTIFACT_DIR / "real_weight_stats_layer0.json").write_text(json.dumps(stats, indent=2) + "\n")


def _make_page_table(
    mesh_device: ttnn.MeshDevice,
    *,
    batch: int,
    max_seq_len: int,
    block_size: int,
    seed: int = 17,
) -> tuple[torch.Tensor, ttnn.Tensor]:
    blocks_per_user = math.ceil(max_seq_len / block_size)
    max_num_blocks = batch * blocks_per_user
    gen = torch.Generator().manual_seed(seed)
    page_table = torch.randperm(max_num_blocks, generator=gen, dtype=torch.int32).reshape(batch, blocks_per_user)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=_mesh_shape_tuple(mesh_device)),
    )
    return page_table, page_table_tt


def _to_tt_prefill(hidden_states: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=_mesh_shape_tuple(mesh_device)),
    )


def _to_tt_decode(hidden_states: torch.Tensor, decoder: FunctionalDecoder, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    tt = ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=_mesh_shape_tuple(mesh_device)),
    )
    return ttnn.to_memory_config(tt, decoder.decode_input_memcfg)


def _tt_to_layer_output(tt_tensor: ttnn.Tensor, *, batch: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    out = to_torch_auto_compose(tt_tensor)
    return out[:, 0:1, :seq_len, :hidden_size].reshape(batch, seq_len, hidden_size)


def _run_prefill(
    *,
    decoder: FunctionalDecoder,
    hf_layer: LlamaDecoderLayer,
    rotary_emb: LlamaRotaryEmbedding,
    mesh_device: ttnn.MeshDevice,
    page_table_tt: ttnn.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, hidden_size = hidden_states.shape
    hf_pos_emb, tt_rot_mats = _meta_rot_mats_prefill(rotary_emb, hidden_states, 0, seq_len, mesh_device)
    tt_out = decoder.prefill_forward(
        _to_tt_prefill(hidden_states, mesh_device),
        rot_mats=tt_rot_mats,
        page_table=page_table_tt,
        user_id=0,
    )
    tt_out_torch = _tt_to_layer_output(tt_out, batch=batch, seq_len=seq_len, hidden_size=hidden_size)
    with torch.no_grad():
        hf_out = hf_layer(
            hidden_states,
            attention_mask=_causal_mask(batch, seq_len, seq_len),
            position_embeddings=hf_pos_emb,
        )
    return hf_out, tt_out_torch


def _run_decode_trace(
    *,
    decoder: FunctionalDecoder,
    hf_layer: LlamaDecoderLayer,
    rotary_emb: LlamaRotaryEmbedding,
    mesh_device: ttnn.MeshDevice,
    page_table_tt: ttnn.Tensor,
    prefill_hidden_states: torch.Tensor,
    decode_hidden_states: torch.Tensor,
    prefill_hf_pos_emb: tuple[torch.Tensor, torch.Tensor],
    current_pos_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, hidden_size = decode_hidden_states.shape
    cache = DynamicCache(config=decoder.hf_config)
    with torch.no_grad():
        hf_layer(
            prefill_hidden_states,
            attention_mask=_causal_mask(batch, prefill_hidden_states.shape[1], prefill_hidden_states.shape[1]),
            position_embeddings=prefill_hf_pos_emb,
            past_key_values=cache,
            use_cache=True,
        )
        pos_ids = torch.full((batch, 1), current_pos_value, dtype=torch.long)
        hf_decode_pos_emb, _ = _hf_and_meta_position_embeddings(rotary_emb, decode_hidden_states, pos_ids)
        hf_decode = hf_layer(
            decode_hidden_states,
            attention_mask=_causal_mask(batch, 1, current_pos_value + 1, past_len=current_pos_value),
            position_embeddings=hf_decode_pos_emb,
            past_key_values=cache,
            use_cache=True,
        )

    decode_rope = DecodeRotaryHelper(rotary_emb, max(current_pos_value + 2, decoder.page_block_size), hidden_size // 32, mesh_device)
    current_pos_host = torch.full((batch,), current_pos_value, dtype=torch.int32)
    current_pos_tt = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = decode_rope.get_rot_mats(current_pos_host)
    decode_input_tt = _to_tt_decode(decode_hidden_states, decoder, mesh_device)

    # Compile once before capture.
    eager_out = decoder.decode_forward(
        decode_input_tt,
        current_pos=current_pos_tt,
        rot_mats=rot_mats,
        page_table=page_table_tt,
    )
    ttnn.synchronize_device(mesh_device)
    eager_torch = _tt_to_layer_output(eager_out, batch=batch, seq_len=1, hidden_size=hidden_size)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = decoder.decode_forward(
        decode_input_tt,
        current_pos=current_pos_tt,
        rot_mats=rot_mats,
        page_table=page_table_tt,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    replay_torch_1 = _tt_to_layer_output(traced_out, batch=batch, seq_len=1, hidden_size=hidden_size)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    replay_torch_2 = _tt_to_layer_output(traced_out, batch=batch, seq_len=1, hidden_size=hidden_size)
    ttnn.release_trace(mesh_device, trace_id)
    return hf_decode, eager_torch, replay_torch_1, replay_torch_2


def test_functional_decoder_contract_and_runtime_fallback_audit():
    source = inspect.getsource(FunctionalDecoder.prefill_forward)
    source += inspect.getsource(FunctionalDecoder.decode_forward)
    source += inspect.getsource(FunctionalDecoder.kv_cache.fget)
    source += inspect.getsource(_LlamaMLP._forward)
    assert "torch" not in source
    assert "from_torch" not in source
    assert "to_torch" not in source
    assert "cpu(" not in source


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_runtime_fallback_audit_measured_prefill_and_traced_decode(mesh_device: ttnn.MeshDevice, monkeypatch):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256
    _, page_table_tt = _make_page_table(
        mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=71
    )
    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        weight_dtype=ttnn.bfloat16,
        kv_cache_dtype=ttnn.bfloat16,
    )

    torch.manual_seed(73)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    _, tt_rot_mats = _meta_rot_mats_prefill(rotary_emb, hidden_states, 0, prefill_seq_len, mesh_device)
    tt_prefill_input = _to_tt_prefill(hidden_states, mesh_device)
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    decode_input_tt = _to_tt_decode(decode_hidden, decoder, mesh_device)
    current_pos_host = torch.tensor([prefill_seq_len], dtype=torch.int32)
    current_pos_tt = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = DecodeRotaryHelper(rotary_emb, prefill_seq_len + 2, hf_config.head_dim, mesh_device).get_rot_mats(
        current_pos_host
    )

    # Warm/compile before the guarded measured calls.
    decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)

    def forbidden_host_bridge(*_args, **_kwargs):
        raise AssertionError("host fallback bridge called inside measured TTNN pass")

    monkeypatch.setattr(ttnn, "from_torch", forbidden_host_bridge)
    monkeypatch.setattr(ttnn, "to_torch", forbidden_host_bridge, raising=False)

    prefill_out = decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decode_out = decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, trace_id)

    assert prefill_out is not None
    assert decode_out is not None
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "runtime_fallback_audit.json").write_text(
        json.dumps(
            {
                "prefill_seq_len": prefill_seq_len,
                "decode_current_pos": prefill_seq_len,
                "guarded_python_bridges": ["ttnn.from_torch", "ttnn.to_torch"],
                "measured_passes": ["prefill_forward", "decode_forward_trace_capture_and_replay"],
                "status": "passed",
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_synthetic_paged_prefill_decode_trace_and_determinism(mesh_device: ttnn.MeshDevice):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    hf_layer, rotary_emb = _make_hf_layer_and_rotary(hf_config, state_dict)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256
    page_table, page_table_tt = _make_page_table(
        mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=13
    )
    assert not torch.equal(page_table, torch.arange(page_table.numel(), dtype=torch.int32).reshape_as(page_table))

    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        weight_dtype=ttnn.bfloat16,
        kv_cache_dtype=ttnn.bfloat16,
    )

    torch.manual_seed(5)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_prefill, tt_prefill = _run_prefill(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        hidden_states=hidden_states,
    )
    prefill_pcc = _assert_pcc("synthetic prefill", hf_prefill, tt_prefill)

    prefill_hf_pos_emb, _ = _hf_and_meta_position_embeddings(
        rotary_emb, hidden_states, torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0)
    )
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_decode, eager_decode, replay_decode_1, replay_decode_2 = _run_decode_trace(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        prefill_hidden_states=hidden_states,
        decode_hidden_states=decode_hidden,
        prefill_hf_pos_emb=prefill_hf_pos_emb,
        current_pos_value=prefill_seq_len,
    )
    eager_decode_pcc = _assert_pcc("synthetic eager decode", hf_decode, eager_decode)
    replay_decode_pcc = _assert_pcc("synthetic traced replay decode", hf_decode, replay_decode_1)
    repeated_pcc = _assert_pcc("synthetic repeated traced decode", replay_decode_1, replay_decode_2, threshold=0.9999)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "synthetic_correctness.json").write_text(
        json.dumps(
            {
                "prefill_seq_len": prefill_seq_len,
                "decode_current_pos": prefill_seq_len,
                "page_block_size": page_block_size,
                "page_table": page_table.tolist(),
                "prefill_pcc": prefill_pcc,
                "eager_decode_pcc": eager_decode_pcc,
                "traced_decode_replay_pcc": replay_decode_pcc,
                "repeated_trace_replay_pcc": repeated_pcc,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.real_weights
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weights_paged_prefill_and_decode_trace(mesh_device: ttnn.MeshDevice):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _real_layer_state_dict()
    _write_weight_stats(state_dict)
    hf_layer, rotary_emb = _make_hf_layer_and_rotary(hf_config, state_dict)
    page_block_size = 64
    prefill_seq_len = int(os.getenv("FD_PREFILL_SEQ_LEN", "128"))
    max_seq_len = max(256, ((prefill_seq_len + page_block_size) + page_block_size - 1) // page_block_size * page_block_size)
    _, page_table_tt = _make_page_table(
        mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=29
    )

    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        weight_dtype=ttnn.bfloat16,
        kv_cache_dtype=ttnn.bfloat16,
    )

    torch.manual_seed(99)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_prefill, tt_prefill = _run_prefill(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        hidden_states=hidden_states,
    )
    prefill_pcc = _assert_pcc("real-weight prefill", hf_prefill, tt_prefill)

    prefill_hf_pos_emb, _ = _hf_and_meta_position_embeddings(
        rotary_emb, hidden_states, torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0)
    )
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_decode, _, replay_decode_1, replay_decode_2 = _run_decode_trace(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        prefill_hidden_states=hidden_states,
        decode_hidden_states=decode_hidden,
        prefill_hf_pos_emb=prefill_hf_pos_emb,
        current_pos_value=prefill_seq_len,
    )
    decode_pcc = _assert_pcc("real-weight traced decode", hf_decode, replay_decode_1)
    repeated_pcc = _assert_pcc("real-weight repeated traced decode", replay_decode_1, replay_decode_2, threshold=0.9999)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_name = (
        "real_weight_correctness.json"
        if prefill_seq_len == 128
        else f"real_weight_correctness_prefill_{prefill_seq_len}.json"
    )
    (ARTIFACT_DIR / artifact_name).write_text(
        json.dumps(
            {
                "prefill_seq_len": prefill_seq_len,
                "decode_current_pos": prefill_seq_len,
                "prefill_pcc": prefill_pcc,
                "traced_decode_replay_pcc": decode_pcc,
                "repeated_trace_replay_pcc": repeated_pcc,
                "threshold": PCC_THRESHOLD,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.long_context
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_longest_feasible_prefill_probe(mesh_device: ttnn.MeshDevice):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    hf_layer, rotary_emb = _make_hf_layer_and_rotary(hf_config, state_dict)
    seq_len = int(os.getenv("FD_LONG_SEQ_LEN", "1024"))
    page_block_size = 64
    _, page_table_tt = _make_page_table(mesh_device, batch=1, max_seq_len=seq_len, block_size=page_block_size, seed=41)
    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        page_block_size=page_block_size,
        max_seq_len=seq_len,
        max_batch_size=1,
        weight_dtype=ttnn.bfloat16,
        kv_cache_dtype=ttnn.bfloat16,
    )
    torch.manual_seed(7)
    hidden_states = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_prefill, tt_prefill = _run_prefill(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        hidden_states=hidden_states,
    )
    pcc = _assert_pcc(f"long prefill seq_len={seq_len}", hf_prefill, tt_prefill)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"long_prefill_{seq_len}.json").write_text(json.dumps({"seq_len": seq_len, "pcc": pcc}) + "\n")


@pytest.mark.perf_artifact
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_artifact_signposted_prefill_and_decode(mesh_device: ttnn.MeshDevice):
    from tracy import signpost

    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = int(os.getenv("FD_PERF_PREFILL_SEQ_LEN", "128"))
    max_seq_len = max(256, prefill_seq_len + page_block_size)
    _, page_table_tt = _make_page_table(mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=53)
    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        weight_dtype=ttnn.bfloat16,
        kv_cache_dtype=ttnn.bfloat16,
    )
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    _, tt_rot_mats = _meta_rot_mats_prefill(rotary_emb, hidden_states, 0, prefill_seq_len, mesh_device)
    tt_prefill_input = _to_tt_prefill(hidden_states, mesh_device)

    decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL")
    out = decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL_END")
    assert out is not None

    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    current_pos_host = torch.tensor([prefill_seq_len], dtype=torch.int32)
    current_pos_tt = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = DecodeRotaryHelper(rotary_emb, prefill_seq_len + 2, hf_config.head_dim, mesh_device).get_rot_mats(
        current_pos_host
    )
    decode_input_tt = _to_tt_decode(decode_hidden, decoder, mesh_device)

    decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decode_out = decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_DECODE")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_DECODE_END")
    ttnn.release_trace(mesh_device, trace_id)
    assert decode_out is not None
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "perf_trace_contract.json").write_text(
        json.dumps(
            {
                "prefill_seq_len": prefill_seq_len,
                "decode_current_pos": prefill_seq_len,
                "prefill_signposts": ["PERF_PREFILL", "PERF_PREFILL_END"],
                "decode_signposts": ["PERF_DECODE", "PERF_DECODE_END"],
                "decode_measurement": "single warmed ttnn.execute_trace replay",
            },
            indent=2,
        )
        + "\n"
    )
