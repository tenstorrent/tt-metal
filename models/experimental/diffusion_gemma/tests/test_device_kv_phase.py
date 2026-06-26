# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for DiffusionGemma KV cache phase discipline (#47474)."""

import os

import pytest

if os.environ.get("DG_RUN_DEVICE") != "1":
    pytest.skip("set DG_RUN_DEVICE=1 to run QB2 KV phase tests", allow_module_level=True)

import torch
import torch.nn.functional as F

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tests.unit.test_model import (
    _create_hf_model,
    _create_hf_text_config,
    _hf_model_state_to_tt_state,
)
from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables
from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.utils_for_testing import assert_with_pcc


pytestmark = pytest.mark.use_module_device


def _cache_region(cache_tensor, start, end, *, is_mesh):
    device_tensors = ttnn.get_device_tensors(cache_tensor) if is_mesh else [cache_tensor]
    return [ttnn.to_torch(t)[:, :, start:end, :].clone() for t in device_tensors]


def _paged_cache_slot(cache_tensor, slot, *, block_size, is_mesh):
    device_tensors = ttnn.get_device_tensors(cache_tensor) if is_mesh else [cache_tensor]
    block = slot // block_size
    offset = slot % block_size
    return [ttnn.to_torch(t)[block : block + 1, :, offset : offset + 1, :].clone() for t in device_tensors]


def _assert_regions_equal(before, after):
    assert len(before) == len(after)
    for idx, (lhs, rhs) in enumerate(zip(before, after)):
        assert torch.equal(lhs, rhs), f"cache prompt region changed on device shard {idx}"


def _assert_regions_changed(before, after):
    assert len(before) == len(after)
    for idx, (lhs, rhs) in enumerate(zip(before, after)):
        assert not torch.equal(lhs, rhs), f"cache region did not change on device shard {idx}"


def _assert_regions_pcc(lhs_regions, rhs_regions, pcc=0.99):
    assert len(lhs_regions) == len(rhs_regions)
    for idx, (lhs, rhs) in enumerate(zip(lhs_regions, rhs_regions)):
        passing, message = assert_with_pcc(lhs.float(), rhs.float(), pcc)
        assert passing, f"cache region PCC failed on device shard {idx}: {message}"


def _embed_tokens(model, tokens, mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    tt_tokens = ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    embeds = model.embed_tokens(tt_tokens)
    embeds = ttnn.reshape(embeds, (1, 1, tokens.shape[1], model.hidden_size))
    return ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)


def _build_tiny_gemma4_state(vocab_size=256):
    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    return model_args, _hf_model_state_to_tt_state(hf_model)


def _build_tiny_gemma4_model(
    mesh_device,
    *,
    vocab_size=256,
    max_seq_len=64,
    model_args=None,
    tt_state=None,
    paged_attention_config=None,
    bounded_sliding_kv_cache=False,
):
    if model_args is None or tt_state is None:
        model_args, tt_state = _build_tiny_gemma4_state(vocab_size=vocab_size)
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        ccl_manager=CCLManager(mesh_device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=1,
        num_layers=1,
        paged_attention_config=paged_attention_config,
        create_kv_cache=True,
        bounded_sliding_kv_cache=bounded_sliding_kv_cache,
    )
    return model


@parametrize_mesh_with_fabric([(1, 4)])
def test_denoise_readonly_prefill_does_not_mutate_frozen_prompt_cache(mesh_device, reset_seeds):
    torch.manual_seed(0)
    prompt_len = 32
    vocab_size = 256
    model = _build_tiny_gemma4_model(mesh_device, vocab_size=vocab_size)

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    prompt_embeds = _embed_tokens(model, prompt_tokens, mesh_device)
    prompt_logits = model(
        prompt_embeds,
        is_decode=False,
        input_ids_torch=prompt_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    prompt_logits.deallocate(True)

    k_cache, v_cache = model.tt_kv_cache[0]
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    k_before = _cache_region(k_cache, 0, prompt_len, is_mesh=is_mesh)
    v_before = _cache_region(v_cache, 0, prompt_len, is_mesh=is_mesh)

    canvas_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    canvas_embeds = _embed_tokens(model, canvas_tokens, mesh_device)
    denoise_logits = model(
        canvas_embeds,
        is_decode=False,
        input_ids_torch=canvas_tokens,
        kv_phase=KVCachePhase.DENOISE_READONLY,
    )
    denoise_logits.deallocate(True)

    _assert_regions_equal(k_before, _cache_region(k_cache, 0, prompt_len, is_mesh=is_mesh))
    _assert_regions_equal(v_before, _cache_region(v_cache, 0, prompt_len, is_mesh=is_mesh))


@parametrize_mesh_with_fabric([(1, 4)])
def test_commit_append_decode_writes_new_position_without_mutating_prompt(mesh_device, reset_seeds):
    torch.manual_seed(1)
    prompt_len = 32
    vocab_size = 256
    model = _build_tiny_gemma4_model(mesh_device, vocab_size=vocab_size)

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    prompt_logits = model(
        _embed_tokens(model, prompt_tokens, mesh_device),
        is_decode=False,
        input_ids_torch=prompt_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    prompt_logits.deallocate(True)

    k_cache, v_cache = model.tt_kv_cache[0]
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    k_prompt_before = _cache_region(k_cache, 0, prompt_len, is_mesh=is_mesh)
    v_prompt_before = _cache_region(v_cache, 0, prompt_len, is_mesh=is_mesh)
    k_append_before = _cache_region(k_cache, prompt_len, prompt_len + 1, is_mesh=is_mesh)
    v_append_before = _cache_region(v_cache, prompt_len, prompt_len + 1, is_mesh=is_mesh)

    append_tokens = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)
    pos_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    pos_u32 = ttnn.from_torch(
        F.pad(torch.tensor([[prompt_len]], dtype=torch.int32), (0, 31), "constant", 0),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=pos_mapper,
    )
    pos_i32 = ttnn.from_torch(
        torch.tensor([prompt_len], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=pos_mapper,
    )
    append_logits = model(
        _embed_tokens(model, append_tokens, mesh_device),
        position_idx=pos_u32,
        position_idx_cache=pos_i32,
        is_decode=True,
        kv_phase=KVCachePhase.COMMIT_APPEND,
    )
    append_logits.deallocate(True)

    _assert_regions_equal(k_prompt_before, _cache_region(k_cache, 0, prompt_len, is_mesh=is_mesh))
    _assert_regions_equal(v_prompt_before, _cache_region(v_cache, 0, prompt_len, is_mesh=is_mesh))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(k_append_before, _cache_region(k_cache, prompt_len, prompt_len + 1, is_mesh=is_mesh))
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(v_append_before, _cache_region(v_cache, prompt_len, prompt_len + 1, is_mesh=is_mesh))
    )


@parametrize_mesh_with_fabric([(1, 4)])
def test_commit_append_decode_writes_full_256_token_canvas(mesh_device, reset_seeds):
    torch.manual_seed(2)
    prompt_len = 32
    canvas_len = 256
    vocab_size = 256
    # SDPA decode requires the allocated K sequence length to be a multiple of
    # k_chunk_size=64; the verified logical region remains prompt_len+canvas_len.
    model = _build_tiny_gemma4_model(mesh_device, vocab_size=vocab_size, max_seq_len=320)

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    prompt_logits = model(
        _embed_tokens(model, prompt_tokens, mesh_device),
        is_decode=False,
        input_ids_torch=prompt_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    prompt_logits.deallocate(True)

    k_cache, v_cache = model.tt_kv_cache[0]
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    k_prompt_before = _cache_region(k_cache, 0, prompt_len, is_mesh=is_mesh)
    v_prompt_before = _cache_region(v_cache, 0, prompt_len, is_mesh=is_mesh)
    k_canvas_before = _cache_region(k_cache, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh)
    v_canvas_before = _cache_region(v_cache, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh)

    for offset in range(canvas_len):
        position = prompt_len + offset
        append_tokens = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)
        pos_u32 = ttnn.from_torch(
            F.pad(torch.tensor([[position]], dtype=torch.int32), (0, 31), "constant", 0),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=mesh_mapper,
        )
        pos_i32 = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=mesh_mapper,
        )
        append_logits = model(
            _embed_tokens(model, append_tokens, mesh_device),
            position_idx=pos_u32,
            position_idx_cache=pos_i32,
            is_decode=True,
            kv_phase=KVCachePhase.COMMIT_APPEND,
        )
        append_logits.deallocate(True)

    _assert_regions_equal(k_prompt_before, _cache_region(k_cache, 0, prompt_len, is_mesh=is_mesh))
    _assert_regions_equal(v_prompt_before, _cache_region(v_cache, 0, prompt_len, is_mesh=is_mesh))
    _assert_regions_changed(
        k_canvas_before, _cache_region(k_cache, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh)
    )
    _assert_regions_changed(
        v_canvas_before, _cache_region(v_cache, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh)
    )


@parametrize_mesh_with_fabric([(1, 4)])
def test_bounded_sliding_commit_append_wraps_cache_slot(mesh_device, reset_seeds):
    torch.manual_seed(23)
    vocab_size = 256
    block_size = 32
    sliding_window = 64
    max_seq_len = sliding_window * 2
    wrap_position = sliding_window
    model_args, tt_state = _build_tiny_gemma4_state(vocab_size=vocab_size)
    model_args.sliding_window = sliding_window
    model_args._hf_text_config.sliding_window = sliding_window
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size,
        max_num_blocks=max_seq_len // block_size,
    )
    model = _build_tiny_gemma4_model(
        mesh_device,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_args=model_args,
        tt_state=tt_state,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=True,
    )
    assert model.layers[0].self_attn.config.cache_position_modulo == sliding_window

    page_tables_per_layer = build_hybrid_page_tables(
        num_layers=1,
        sliding_layers_mask=[True],
        num_users=1,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )
    k_cache, v_cache = model.tt_kv_cache[0]
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    expected_slot = wrap_position % sliding_window
    k_before = _paged_cache_slot(k_cache, expected_slot, block_size=block_size, is_mesh=is_mesh)
    v_before = _paged_cache_slot(v_cache, expected_slot, block_size=block_size, is_mesh=is_mesh)
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    append_tokens = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)
    pos_u32 = ttnn.from_torch(
        F.pad(torch.tensor([[wrap_position]], dtype=torch.int32), (0, 31), "constant", 0),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mesh_mapper,
    )
    pos_i32 = ttnn.from_torch(
        torch.tensor([wrap_position], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=mesh_mapper,
    )

    append_logits, _ = model.ttnn_decode_forward(
        _embed_tokens(model, append_tokens, mesh_device),
        pos_u32,
        pos_i32,
        page_table=None,
        kv_cache=model.tt_kv_cache,
        page_tables_per_layer=page_tables_per_layer,
        kv_phase=KVCachePhase.COMMIT_APPEND,
    )
    append_logits.deallocate(True)

    _assert_regions_changed(k_before, _paged_cache_slot(k_cache, expected_slot, block_size=block_size, is_mesh=is_mesh))
    _assert_regions_changed(v_before, _paged_cache_slot(v_cache, expected_slot, block_size=block_size, is_mesh=is_mesh))


@parametrize_mesh_with_fabric([(1, 4)])
def test_commit_append_canvas_kv_matches_reencode_pcc(mesh_device, reset_seeds):
    torch.manual_seed(3)
    prompt_len = 32
    canvas_len = 256
    vocab_size = 256
    max_seq_len = 320
    model_args, tt_state = _build_tiny_gemma4_state(vocab_size=vocab_size)
    commit_model = _build_tiny_gemma4_model(
        mesh_device, vocab_size=vocab_size, max_seq_len=max_seq_len, model_args=model_args, tt_state=tt_state
    )
    reencode_model = _build_tiny_gemma4_model(
        mesh_device, vocab_size=vocab_size, max_seq_len=max_seq_len, model_args=model_args, tt_state=tt_state
    )

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    canvas_tokens = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    prompt_logits = commit_model(
        _embed_tokens(commit_model, prompt_tokens, mesh_device),
        is_decode=False,
        input_ids_torch=prompt_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    prompt_logits.deallocate(True)

    for offset in range(canvas_len):
        position = prompt_len + offset
        append_tokens = canvas_tokens[:, offset : offset + 1]
        pos_u32 = ttnn.from_torch(
            F.pad(torch.tensor([[position]], dtype=torch.int32), (0, 31), "constant", 0),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=mesh_mapper,
        )
        pos_i32 = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=mesh_mapper,
        )
        append_logits = commit_model(
            _embed_tokens(commit_model, append_tokens, mesh_device),
            position_idx=pos_u32,
            position_idx_cache=pos_i32,
            is_decode=True,
            kv_phase=KVCachePhase.COMMIT_APPEND,
        )
        append_logits.deallocate(True)

    full_tokens = torch.cat([prompt_tokens, canvas_tokens], dim=-1)
    reencode_logits = reencode_model(
        _embed_tokens(reencode_model, full_tokens, mesh_device),
        is_decode=False,
        input_ids_torch=full_tokens,
        kv_phase=KVCachePhase.PREFILL_WRITE,
    )
    reencode_logits.deallocate(True)

    commit_k, commit_v = commit_model.tt_kv_cache[0]
    reencode_k, reencode_v = reencode_model.tt_kv_cache[0]
    _assert_regions_pcc(
        _cache_region(commit_k, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh),
        _cache_region(reencode_k, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh),
    )
    _assert_regions_pcc(
        _cache_region(commit_v, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh),
        _cache_region(reencode_v, prompt_len, prompt_len + canvas_len, is_mesh=is_mesh),
    )
