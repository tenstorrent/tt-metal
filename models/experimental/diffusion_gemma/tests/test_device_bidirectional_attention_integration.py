# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device integration test for the real Gemma4 masked non-causal prefill path."""

import os

import pytest

if os.environ.get("DG_RUN_DEVICE") != "1":
    pytest.skip("set DG_RUN_DEVICE=1 to run QB2 bidirectional attention integration tests", allow_module_level=True)

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
from tests.ttnn.utils_for_testing import assert_with_pcc


pytestmark = pytest.mark.use_module_device


@parametrize_mesh_with_fabric([(1, 4)])
def test_real_attention_prefill_accepts_all_attend_noncausal_mask(mesh_device, reset_seeds):
    torch.manual_seed(4)
    seq_len = 32
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
    tt_model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=_hf_model_state_to_tt_state(hf_model),
        ccl_manager=CCLManager(mesh_device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        max_local_batch_size=1,
        num_layers=1,
        create_kv_cache=True,
    )

    tokens = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    attn_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
    with torch.no_grad():
        hf_logits = hf_model(tokens, attention_mask=attn_mask)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tt_tokens = ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mesh_mapper,
    )
    tt_embeds = tt_model.embed_tokens(tt_tokens)
    tt_embeds = ttnn.reshape(tt_embeds, (1, 1, seq_len, model_args.hidden_size))
    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)
    tt_mask = ttnn.from_torch(
        attn_mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_logits = tt_model(
        tt_embeds,
        is_decode=False,
        input_ids_torch=tokens,
        kv_phase=KVCachePhase.DENOISE_READONLY,
        attn_mask=tt_mask,
    )
    tt_logits_torch = (
        ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]) if is_mesh else ttnn.to_torch(tt_logits)
    ).squeeze(0)

    passing, message = assert_with_pcc(hf_logits.float(), tt_logits_torch.float(), 0.99)
    assert passing, message
