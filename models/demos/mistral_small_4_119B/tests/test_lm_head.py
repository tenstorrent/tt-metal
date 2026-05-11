# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LM Head (final norm + vocab projection) parity tests.

Validates ``Mistral4LMHead`` against a torch reference (``Mistral4RMSNorm`` + ``nn.Linear``).
Uses the root ``device`` fixture for single-chip runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[4]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
try:
    from tests.scripts.ompi_singleton_env import apply_ompi_singleton_workaround_env

    apply_ompi_singleton_workaround_env()
except ImportError:
    if os.environ.get("TT_METAL_OMPI_SINGLETON_WORKAROUND", "1") != "0":
        os.environ.setdefault("OMPI_MCA_plm", "isolated")
        os.environ.setdefault("PRTE_MCA_plm", "isolated")

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.lm_head import Mistral4LMHead
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.run_config import create_run_config


def _tiny_mistral4_config():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        rms_norm_eps=1e-6,
    )


def _assert_pcc(tt_output: torch.Tensor, reference_output: torch.Tensor, *, pcc_required: float) -> None:
    tt_out = tt_output.cpu().float()
    ref_out = reference_output.cpu().float()

    while tt_out.ndim < ref_out.ndim:
        tt_out = tt_out.unsqueeze(0)
    while ref_out.ndim < tt_out.ndim:
        ref_out = ref_out.unsqueeze(0)

    # Align batch/seq dims
    seq_or_batch = min(tt_out.shape[-2], ref_out.shape[-2])
    vocab = min(tt_out.shape[-1], ref_out.shape[-1])
    tt_out = tt_out[..., :seq_or_batch, :vocab]
    ref_out = ref_out[..., :seq_or_batch, :vocab]

    passing, pcc = comp_pcc(tt_out, ref_out, pcc_required)
    logger.info(f"lm_head PCC: {pcc}")
    assert passing, f"PCC {pcc} < required {pcc_required}"


def _iter_weight_tensors(weight_config):
    stack = [weight_config]
    while stack:
        node = stack.pop()
        if isinstance(node, ttnn.Tensor):
            yield node
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


def test_mistral4_lm_head_decode_matches_torch(device, tmp_path):
    """``Mistral4LMHead.forward_decode`` vs torch RMSNorm + Linear (decode batch=1)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    hf_config = _tiny_mistral4_config()
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    batch_size = 1

    # Reference modules
    torch.manual_seed(42)
    reference_norm = Mistral4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps).eval().to(torch.bfloat16)
    reference_linear = torch.nn.Linear(hidden_size, vocab_size, bias=False).eval().to(torch.bfloat16)

    # Build state dict matching Mistral4LMHead.convert_weights expectations
    state_dict = {
        "norm.weight": reference_norm.weight.detach().clone(),
        "lm_head.weight": reference_linear.weight.detach().clone(),
    }

    # Reference forward
    torch_input = torch.randn(1, 1, batch_size, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        normed = reference_norm(torch_input)
        reference_output = reference_linear(normed)

    # TT path
    ccl = CCL(device)
    weight_config = Mistral4LMHead.convert_weights(hf_config, (state_dict,), tmp_path / "lm_head", device)
    model_config = Mistral4LMHead.decode_model_config(hf_config, device, batch_size_per_row=batch_size)
    model_state = Mistral4LMHead.create_state(hf_config, device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state, {})

    tt_input = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = Mistral4LMHead.forward_decode(tt_input, run_config)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
        )
        tt_output_torch = tt_output_torch[:1]

        logger.info(f"TT output shape: {tt_output_torch.shape}, Reference shape: {reference_output.shape}")
        _assert_pcc(tt_output_torch, reference_output, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


def test_mistral4_lm_head_prefill_matches_torch(device, tmp_path):
    """``Mistral4LMHead.forward_prefill`` vs torch RMSNorm + Linear (prefill seq_len=32)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    hf_config = _tiny_mistral4_config()
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    seq_len = 32

    # Reference modules
    torch.manual_seed(42)
    reference_norm = Mistral4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps).eval().to(torch.bfloat16)
    reference_linear = torch.nn.Linear(hidden_size, vocab_size, bias=False).eval().to(torch.bfloat16)

    state_dict = {
        "norm.weight": reference_norm.weight.detach().clone(),
        "lm_head.weight": reference_linear.weight.detach().clone(),
    }

    # Reference forward
    torch_input = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        normed = reference_norm(torch_input)
        reference_output = reference_linear(normed)

    # TT path
    ccl = CCL(device)
    weight_config = Mistral4LMHead.convert_weights(hf_config, (state_dict,), tmp_path / "lm_head_prefill", device)
    model_config = Mistral4LMHead.prefill_model_config(hf_config, device)
    model_state = Mistral4LMHead.create_state(hf_config, device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state, {})

    tt_input = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = Mistral4LMHead.forward_prefill(tt_input, run_config)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
        )
        tt_output_torch = tt_output_torch[:1]

        logger.info(f"TT output shape: {tt_output_torch.shape}, Reference shape: {reference_output.shape}")
        _assert_pcc(tt_output_torch, reference_output, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
