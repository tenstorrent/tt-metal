# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import inspect
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    MODEL_ID,
    FunctionalDecoder,
)
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs
from models.common.utility_functions import comp_pcc

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - only absent outside profiling runs

    def signpost(header: str) -> None:
        del header


PCC_THRESHOLD = 0.995
PAGE_BLOCK_SIZE = 64
FULL_CACHE_SEQ_LEN = 128 * 1024


@pytest.fixture(scope="function")
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    param = getattr(request, "param", (1, 1))
    if not isinstance(param, tuple) or len(param) != 2:
        raise ValueError(f"mesh_device fixture expects a (rows, cols) tuple, got {param!r}")
    requested = param[0] * param[1]
    if requested > ttnn.get_num_devices():
        pytest.skip(f"requested {requested} devices, only {ttnn.get_num_devices()} available")

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*param), **device_params)
    original_default_device = ttnn.GetDefaultDevice()
    ttnn.SetDefaultDevice(mesh)
    try:
        yield mesh
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)


LLAMA31_8B_LAYER0_WEIGHT_STATS = {
    "model.layers.0.input_layernorm.weight": {
        "shape": [4096],
        "dtype": "torch.bfloat16",
        "mean": 0.07075785100460052,
        "std": 0.12142692506313324,
    },
    "model.layers.0.self_attn.q_proj.weight": {
        "shape": [4096, 4096],
        "dtype": "torch.bfloat16",
        "mean": 1.4063164144317852e-07,
        "std": 0.01866932027041912,
    },
    "model.layers.0.self_attn.k_proj.weight": {
        "shape": [1024, 4096],
        "dtype": "torch.bfloat16",
        "mean": -9.069692168850452e-06,
        "std": 0.02693898044526577,
    },
    "model.layers.0.self_attn.v_proj.weight": {
        "shape": [1024, 4096],
        "dtype": "torch.bfloat16",
        "mean": -2.834541874108254e-06,
        "std": 0.007224599830806255,
    },
    "model.layers.0.self_attn.o_proj.weight": {
        "shape": [4096, 4096],
        "dtype": "torch.bfloat16",
        "mean": 6.681928255147795e-08,
        "std": 0.008338166400790215,
    },
    "model.layers.0.post_attention_layernorm.weight": {
        "shape": [4096],
        "dtype": "torch.bfloat16",
        "mean": 0.13376958668231964,
        "std": 0.011952084489166737,
    },
    "model.layers.0.mlp.gate_proj.weight": {
        "shape": [14336, 4096],
        "dtype": "torch.bfloat16",
        "mean": 7.519858172599925e-06,
        "std": 0.01280402671545744,
    },
    "model.layers.0.mlp.up_proj.weight": {
        "shape": [14336, 4096],
        "dtype": "torch.bfloat16",
        "mean": -3.887561490500957e-07,
        "std": 0.011733459308743477,
    },
    "model.layers.0.mlp.down_proj.weight": {
        "shape": [4096, 14336],
        "dtype": "torch.bfloat16",
        "mean": -9.766669109012582e-07,
        "std": 0.01175522431731224,
    },
}


def _hf_config():
    cfg = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    cfg._attn_implementation = "eager"
    return cfg


def _synthetic_state_dict(seed: int = 20260615) -> dict[str, torch.Tensor]:
    state_dict = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for name, stats in LLAMA31_8B_LAYER0_WEIGHT_STATS.items():
        shape = tuple(stats["shape"])
        if name.endswith("input_layernorm.weight") or name.endswith("post_attention_layernorm.weight"):
            tensor = torch.normal(
                mean=stats["mean"],
                std=max(stats["std"], 1e-6),
                size=shape,
                generator=generator,
                dtype=torch.float32,
            )
            tensor = tensor.abs()
        else:
            tensor = torch.normal(
                mean=stats["mean"],
                std=stats["std"],
                size=shape,
                generator=generator,
                dtype=torch.float32,
            )
        state_dict[name] = tensor.to(torch.bfloat16)
    return state_dict


def _real_state_dict() -> dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
    wanted = set(LLAMA31_8B_LAYER0_WEIGHT_STATS)
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in wanted}


def _reference_layer(hf_config, state_dict: dict[str, torch.Tensor]) -> LlamaDecoderLayer:
    layer = LlamaDecoderLayer(hf_config, layer_idx=0).to(torch.bfloat16).eval()
    layer_state = {name.removeprefix("model.layers.0."): tensor for name, tensor in state_dict.items()}
    layer.load_state_dict(layer_state, strict=True)
    return layer


def _hf_rotary(hf_config) -> LlamaRotaryEmbedding:
    return LlamaRotaryEmbedding(config=hf_config).eval()


def _permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(cos.shape) == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = torch.stack((cos[:, : cos.shape[1] // 2], cos[:, : cos.shape[1] // 2]), dim=-1).flatten(-2)
    sin = torch.stack((sin[:, : sin.shape[1] // 2], sin[:, : sin.shape[1] // 2]), dim=-1).flatten(-2)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _rope_setup(mesh_device: ttnn.MeshDevice, hf_config, rotary_emb: LlamaRotaryEmbedding, max_seq_len: int, batch: int):
    dummy = torch.zeros(1, 1, max_seq_len, hf_config.head_dim, dtype=torch.bfloat16)
    position_ids = torch.arange(max_seq_len).unsqueeze(0)
    with torch.no_grad():
        cos_hf, sin_hf = rotary_emb(dummy, position_ids)
    cos_meta, sin_meta = _permute_to_meta_format(cos_hf.float(), sin_hf.float())
    cos_lw = LazyWeight(
        source=cos_meta.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_lw = LazyWeight(
        source=sin_meta.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return RotarySetup1D.from_config(
        Rope1DConfig(
            cos_matrix=cos_lw,
            sin_matrix=sin_lw,
            max_batch_size=batch,
            head_dim=hf_config.head_dim,
            device=mesh_device,
            use_qk_fused=False,
            datatype=ttnn.bfloat16,
        )
    )


def _decode_rot_mats(rope_setup: RotarySetup1D, current_pos: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    rot_idxs = prepare_rot_idxs(rope_setup.config, current_pos.to(torch.long), on_host=False)
    return tuple(rope_setup.decode_forward(rot_idxs))


def _tt_tensor(mesh_device: ttnn.MeshDevice, tensor: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _page_table(mesh_device: ttnn.MeshDevice, *, batch: int, max_num_blocks: int) -> tuple[torch.Tensor, ttnn.Tensor]:
    permutation = torch.randperm(max_num_blocks, generator=torch.Generator().manual_seed(11))
    page_table = torch.argsort(permutation).reshape(batch, max_num_blocks // batch).to(torch.int32)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return page_table, page_table_tt


def _causal_prefill_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def _reference_prefill(layer, rotary_emb, hidden_states: torch.Tensor) -> tuple[torch.Tensor, DynamicCache]:
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    with torch.no_grad():
        position_embeddings = rotary_emb(hidden_states, position_ids)
        cache = DynamicCache()
        out = layer(
            hidden_states,
            attention_mask=_causal_prefill_mask(seq_len, hidden_states.dtype),
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
    return out, cache


def _reference_decode(layer, rotary_emb, cache: DynamicCache, hidden_states: torch.Tensor, current_pos: int) -> torch.Tensor:
    position_ids = torch.full((hidden_states.shape[0], 1), current_pos, dtype=torch.long)
    attention_mask = torch.zeros(
        hidden_states.shape[0],
        1,
        1,
        current_pos + 1,
        dtype=hidden_states.dtype,
    )
    with torch.no_grad():
        position_embeddings = rotary_emb(hidden_states, position_ids)
        return layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )


def _assert_pcc(name: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float = PCC_THRESHOLD) -> float:
    passing, pcc = comp_pcc(expected.float(), actual.float(), threshold)
    logger.info(f"{name} PCC={pcc}")
    assert passing, f"{name} PCC {pcc} < {threshold}"
    return float(pcc)


@contextmanager
def _assert_no_host_fallback():
    with (
        patch("ttnn.from_torch", side_effect=AssertionError("ttnn.from_torch called inside hot path")),
        patch("ttnn.as_tensor", side_effect=AssertionError("ttnn.as_tensor called inside hot path")),
        patch("ttnn.to_torch", side_effect=AssertionError("ttnn.to_torch called inside hot path")),
        patch("torch.tensor", side_effect=AssertionError("torch.tensor called inside hot path")),
        patch("torch.as_tensor", side_effect=AssertionError("torch.as_tensor called inside hot path")),
        patch("torch.empty", side_effect=AssertionError("torch.empty called inside hot path")),
        patch("torch.zeros", side_effect=AssertionError("torch.zeros called inside hot path")),
        patch("torch.ones", side_effect=AssertionError("torch.ones called inside hot path")),
        patch("torch.arange", side_effect=AssertionError("torch.arange called inside hot path")),
        patch("torch.full", side_effect=AssertionError("torch.full called inside hot path")),
        patch("torch.cat", side_effect=AssertionError("torch.cat called inside hot path")),
        patch("torch.stack", side_effect=AssertionError("torch.stack called inside hot path")),
        patch("torch.matmul", side_effect=AssertionError("torch.matmul called inside hot path")),
        patch("torch.nn.functional.linear", side_effect=AssertionError("torch.nn.functional.linear called inside hot path")),
    ):
        yield


def test_functional_decoder_contract_and_stats():
    assert Path("models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py").exists()
    assert "rot_mats" in inspect.signature(FunctionalDecoder.prefill_forward).parameters
    assert "page_table" in inspect.signature(FunctionalDecoder.prefill_forward).parameters
    assert "current_pos" in inspect.signature(FunctionalDecoder.decode_forward).parameters
    assert "page_table" in inspect.signature(FunctionalDecoder.decode_forward).parameters
    for name, stats in LLAMA31_8B_LAYER0_WEIGHT_STATS.items():
        assert stats["shape"], name
        assert stats["dtype"] == "torch.bfloat16"
        assert stats["std"] >= 0.0


def _run_prefill_decode_trace_case(
    mesh_device: ttnn.MeshDevice,
    state_dict: dict[str, torch.Tensor],
    *,
    real_weights: bool,
    seq_len: int = 128,
    max_seq_len: int | None = None,
    max_num_blocks: int | None = None,
    emit_perf_signposts: bool = True,
):
    hf_config = _hf_config()
    batch = 1
    assert seq_len % 128 == 0
    max_seq_len = max_seq_len or max(seq_len + 1, 256)
    max_num_blocks = max_num_blocks or max(2, (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE)
    current_pos_value = seq_len

    torch.manual_seed(123)
    reference_layer = _reference_layer(hf_config, state_dict)
    rotary_emb = _hf_rotary(hf_config)
    tt_decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=max_num_blocks,
        weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        kv_cache_dtype=ttnn.bfloat16,
    )

    page_table, page_table_tt = _page_table(mesh_device, batch=batch, max_num_blocks=max_num_blocks)
    assert page_table.shape == (batch, max_num_blocks)
    assert int(page_table[0, 0]) != 0 or int(page_table[0, 1]) != 1

    rope_setup = _rope_setup(mesh_device, hf_config, rotary_emb, max_seq_len + 1, batch)

    prefill_hidden = torch.randn(batch, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05
    ref_prefill, ref_cache = _reference_prefill(reference_layer, rotary_emb, prefill_hidden)
    tt_prefill = tt_decoder.prefill_forward(
        _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0)),
        rot_mats=tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len)),
        page_table=page_table_tt,
        user_id=0,
    )
    tt_prefill_host = to_torch_auto_compose(tt_prefill)[:, 0, :seq_len, :].reshape(batch, seq_len, hf_config.hidden_size)
    prefill_pcc = _assert_pcc("prefill", ref_prefill, tt_prefill_host)

    prefill_audit_input = _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0))
    prefill_audit_rot_mats = tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len))
    with _assert_no_host_fallback():
        tt_prefill_audit = tt_decoder.prefill_forward(
            prefill_audit_input,
            rot_mats=prefill_audit_rot_mats,
            page_table=page_table_tt,
            user_id=0,
        )
    ttnn.synchronize_device(mesh_device)
    del tt_prefill_audit

    if emit_perf_signposts:
        signpost(header="PERF_PREFILL")
    tt_prefill_perf = tt_decoder.prefill_forward(
        _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0)),
        rot_mats=tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len)),
        page_table=page_table_tt,
        user_id=0,
    )
    ttnn.synchronize_device(mesh_device)
    if emit_perf_signposts:
        signpost(header="PERF_PREFILL_END")
    del tt_prefill_perf

    decode_hidden = torch.randn(batch, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05
    ref_decode = _reference_decode(reference_layer, rotary_emb, ref_cache, decode_hidden, current_pos_value)
    current_pos_host = torch.full((batch,), current_pos_value, dtype=torch.int32)
    current_pos = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    decode_rot_mats = _decode_rot_mats(rope_setup, current_pos_host.to(torch.long))
    tt_decode_input = _tt_tensor(mesh_device, decode_hidden.unsqueeze(0))

    # Warm compile and prove the hot path does not need host conversion APIs.
    tt_warm = tt_decoder.decode_forward(
        tt_decode_input,
        current_pos=current_pos,
        rot_mats=decode_rot_mats,
        page_table=page_table_tt,
    )
    with _assert_no_host_fallback():
        tt_audit = tt_decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            rot_mats=decode_rot_mats,
            page_table=page_table_tt,
        )
    del tt_audit

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = tt_decoder.decode_forward(
        tt_decode_input,
        current_pos=current_pos,
        rot_mats=decode_rot_mats,
        page_table=page_table_tt,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first_replay = to_torch_auto_compose(traced_out)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size)
    if emit_perf_signposts:
        signpost(header="PERF_DECODE")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    if emit_perf_signposts:
        signpost(header="PERF_DECODE_END")
    second_replay = to_torch_auto_compose(traced_out)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size)
    ttnn.release_trace(mesh_device, trace_id)

    eager_decode = to_torch_auto_compose(tt_warm)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size)
    decode_pcc = _assert_pcc("decode_trace", ref_decode, first_replay)
    determinism_pcc = _assert_pcc("decode_trace_repeated_input", first_replay, second_replay, threshold=0.9999)
    eager_trace_pcc = _assert_pcc("decode_eager_vs_trace", eager_decode, first_replay, threshold=0.9999)

    return {
        "real_weights": real_weights,
        "prefill_pcc": prefill_pcc,
        "decode_trace_pcc": decode_pcc,
        "determinism_pcc": determinism_pcc,
        "eager_trace_pcc": eager_trace_pcc,
        "seq_len": seq_len,
        "decode_context": current_pos_value + 1,
        "page_block_size": PAGE_BLOCK_SIZE,
        "max_num_blocks": max_num_blocks,
        "max_seq_len": max_seq_len,
        "runtime_fallback_audit": "prefill_decode_clean",
    }


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_functional_decoder_full_context_cache_contract(mesh_device: ttnn.MeshDevice, device_params):
    hf_config = _hf_config()
    decoder = FunctionalDecoder.from_state_dict(
        _synthetic_state_dict(),
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=FULL_CACHE_SEQ_LEN,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE,
        weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        kv_cache_dtype=ttnn.bfloat16,
    )
    decoder.self_attn.load_device_weights()
    key_cache, value_cache = decoder.self_attn.kv_cache
    assert key_cache.shape[0] == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE
    assert value_cache.shape[0] == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE
    assert key_cache.shape[2] == PAGE_BLOCK_SIZE
    assert value_cache.shape[2] == PAGE_BLOCK_SIZE
    assert decoder.self_attn.config.max_seq_len == FULL_CACHE_SEQ_LEN
    assert decoder.self_attn.config.paged_attention_config.max_num_blocks == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_functional_decoder_synthetic_paged_prefill_decode_trace(mesh_device: ttnn.MeshDevice, device_params):
    metrics = _run_prefill_decode_trace_case(mesh_device, _synthetic_state_dict(), real_weights=False)
    logger.info(f"synthetic functional decoder metrics: {metrics}")


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_functional_decoder_synthetic_long_context_paged_prefill_decode_trace(
    mesh_device: ttnn.MeshDevice, device_params
):
    seq_len = int(os.environ.get("LLAMA31_8B_FUNCTIONAL_DECODER_LONG_SEQ_LEN", "32768"))
    max_seq_len = int(os.environ.get("LLAMA31_8B_FUNCTIONAL_DECODER_LONG_MAX_SEQ_LEN", str(seq_len + 128)))
    assert seq_len % 128 == 0
    assert max_seq_len > seq_len
    max_num_blocks = max(2, (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE)
    metrics = _run_prefill_decode_trace_case(
        mesh_device,
        _synthetic_state_dict(),
        real_weights=False,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        max_num_blocks=max_num_blocks,
        emit_perf_signposts=False,
    )
    logger.info(f"synthetic long-context functional decoder metrics: {metrics}")


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_functional_decoder_real_weights_paged_prefill_decode_trace(mesh_device: ttnn.MeshDevice, device_params):
    metrics = _run_prefill_decode_trace_case(mesh_device, _real_state_dict(), real_weights=True)
    logger.info(f"real-weight functional decoder metrics: {metrics}")
