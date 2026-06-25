# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for DiffusionGemma KV cache phase discipline (#47474)."""

import os

import pytest

if os.environ.get("DG_RUN_DEVICE") != "1":
    pytest.skip("set DG_RUN_DEVICE=1 to run QB2 KV phase tests", allow_module_level=True)

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tests.unit.test_model import (
    _create_hf_model,
    _create_hf_text_config,
    _hf_model_state_to_tt_state,
)
from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


pytestmark = pytest.mark.use_module_device


def _cache_prompt_region(cache_tensor, prompt_len, *, is_mesh):
    device_tensors = ttnn.get_device_tensors(cache_tensor) if is_mesh else [cache_tensor]
    return [ttnn.to_torch(t)[:, :, :prompt_len, :].clone() for t in device_tensors]


def _assert_regions_equal(before, after):
    assert len(before) == len(after)
    for idx, (lhs, rhs) in enumerate(zip(before, after)):
        assert torch.equal(lhs, rhs), f"cache prompt region changed on device shard {idx}"


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


@parametrize_mesh_with_fabric([(1, 4)])
def test_denoise_readonly_prefill_does_not_mutate_frozen_prompt_cache(mesh_device, reset_seeds):
    torch.manual_seed(0)
    prompt_len = 32
    vocab_size = 256

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=_hf_model_state_to_tt_state(hf_model),
        ccl_manager=CCLManager(mesh_device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=64,
        max_local_batch_size=1,
        num_layers=1,
        create_kv_cache=True,
    )

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
    k_before = _cache_prompt_region(k_cache, prompt_len, is_mesh=is_mesh)
    v_before = _cache_prompt_region(v_cache, prompt_len, is_mesh=is_mesh)

    canvas_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    canvas_embeds = _embed_tokens(model, canvas_tokens, mesh_device)
    denoise_logits = model(
        canvas_embeds,
        is_decode=False,
        input_ids_torch=canvas_tokens,
        kv_phase=KVCachePhase.DENOISE_READONLY,
    )
    denoise_logits.deallocate(True)

    _assert_regions_equal(k_before, _cache_prompt_region(k_cache, prompt_len, is_mesh=is_mesh))
    _assert_regions_equal(v_before, _cache_prompt_region(v_cache, prompt_len, is_mesh=is_mesh))
