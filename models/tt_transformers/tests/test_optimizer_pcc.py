# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded full-model PCC gate for production Llama optimizer runs."""

from __future__ import annotations

import gc
import os

import pytest
import torch
from transformers import AutoModelForCausalLM

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


INPUT_IDS = [128000] + [1000 + ((index * 7919) % 120000) for index in range(127)]
PCC_THRESHOLD = 0.90
MAX_SEQ_LEN = 2048


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


def _close(mesh_device):
    if mesh_device is not None:
        for submesh in list(mesh_device.get_submeshes()):
            if submesh is not mesh_device:
                ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(300)
def test_optimizer_full_model_pcc():
    """Compare full-depth prefill and one teacher-forced decode with Hugging Face."""
    model_path = os.environ["HF_MODEL"]
    mesh_device = generator = model = model_args = reference = state_dict = tt_kv_cache = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 4),
            trace_region_size=52_000_000,
            num_command_queues=1,
        )
        paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
        optimizations = lambda args: DecodersPrecision.performance(args.n_layers, args.model_name)
        model_args, model, tt_kv_cache, state_dict = create_tt_model(
            mesh_device,
            instruct=True,
            max_batch_size=1,
            optimizations=optimizations,
            max_seq_len=MAX_SEQ_LEN,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            num_layers=None,
            use_prefetcher=False,
            use_hf_rope=False,
        )
        assert model_args.n_layers == 32
        assert len(model.layers) == 32
        assert mesh_device.get_num_devices() == 4
        state_dict = None
        gc.collect()

        reference = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).eval()
        input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)
        with torch.no_grad():
            hf_prefill = reference(input_ids).logits[:, -1, :].float()
            teacher_token = hf_prefill.argmax(dim=-1)
            hf_decode = reference(torch.cat((input_ids, teacher_token[:, None]), dim=1)).logits[:, -1, :].float()

        generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
        page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
            1, paged_attention_config.max_num_blocks
        )
        kv_cache = [tt_kv_cache]
        tt_prefill = generator.prefill_forward_text(
            input_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[len(INPUT_IDS)],
            enable_trace=False,
            warmup_prefill=False,
        ).reshape(1, -1)
        prefill_pcc = _pcc(tt_prefill, hf_prefill)
        tt_decode = generator.decode_forward(
            teacher_token.reshape(1, 1),
            torch.tensor([len(INPUT_IDS)], dtype=torch.int64),
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=False,
            read_from_device=True,
            sampling_params=None,
            reset_batch=True,
        ).reshape(1, -1)
        decode_pcc = _pcc(tt_decode, hf_decode)
        measured_pcc = min(prefill_pcc, decode_pcc)
        print(
            f"Prefill PCC: {prefill_pcc:.6f} | Decode PCC: {decode_pcc:.6f} | PCC: {measured_pcc:.6f}",
            flush=True,
        )
        assert measured_pcc >= PCC_THRESHOLD
    finally:
        reference = None
        generator = None
        model = None
        model_args = None
        state_dict = None
        tt_kv_cache = None
        gc.collect()
        _close(mesh_device)
