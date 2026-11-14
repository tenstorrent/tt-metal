# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic correctness and N150 runtime regression tests for OpenVLA."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import timm
import torch

import ttnn
from models.experimental.openvla.tt import open_vla
from models.experimental.openvla.tt import tt_optimized_openvla_vision
from ttnn.model_preprocessing import preprocess_model_parameters
from tt_py_test_utils_common.utils_for_testing import assert_with_pcc


def _apply_rope_hf(x: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    """Apply Hugging Face split-half RoPE: x1 = x[..., :d/2], x2 = x[..., d/2:]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    return (x * cos) + (rotated * sin)


def _apply_rope_meta(x: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    """Apply Meta interleaved pairwise RoPE: [r0, i0, r1, i1, ...]."""
    cos = torch.repeat_interleave(cos_half, repeats=2, dim=-1)
    sin = torch.repeat_interleave(sin_half, repeats=2, dim=-1)
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-x_pairs[..., 1], x_pairs[..., 0]), dim=-1).flatten(-2)
    return (x * cos) + (rotated * sin)


def test_qk_rope_permutation_preserves_hf_attention_scores():
    """HF Q/K weights permuted via permute_hf_to_meta_qk must match HF RoPE dot products under Meta RoPE."""
    torch.manual_seed(42)
    batch, seq_len, hidden = 1, 6, 4096
    n_heads, n_kv_heads, head_dim = 32, 32, 128

    x_q = torch.randn(batch, seq_len, hidden, dtype=torch.float32)
    x_k = torch.randn(batch, seq_len, hidden, dtype=torch.float32)
    wq_hf = torch.randn(n_heads * head_dim, hidden, dtype=torch.float32)
    wk_hf = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.float32)

    cos_half = torch.cos(torch.randn(1, 1, seq_len, head_dim // 2))
    sin_half = torch.sin(torch.randn(1, 1, seq_len, head_dim // 2))

    # Reference Hugging Face RoPE attention scores
    q_hf = (x_q @ wq_hf.T).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    k_hf = (x_k @ wk_hf.T).view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    scores_hf = torch.matmul(_apply_rope_hf(q_hf, cos_half, sin_half), _apply_rope_hf(k_hf, cos_half, sin_half).transpose(-2, -1))

    # Mapped weights under Meta RoPE
    mapped = open_vla.map_openvla_hf_to_meta_keys(
        {
            "language_model.model.layers.0.self_attn.q_proj.weight": wq_hf,
            "language_model.model.layers.0.self_attn.k_proj.weight": wk_hf,
            "language_model.model.layers.0.self_attn.v_proj.weight": wk_hf.clone(),
        }
    )
    wq_meta = mapped["layers.0.attention.wq.weight"]
    wk_meta = mapped["layers.0.attention.wk.weight"]

    q_meta = (x_q @ wq_meta.T).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    k_meta = (x_k @ wk_meta.T).view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    scores_meta = torch.matmul(_apply_rope_meta(q_meta, cos_half, sin_half), _apply_rope_meta(k_meta, cos_half, sin_half).transpose(-2, -1))

    torch.testing.assert_close(scores_meta, scores_hf, rtol=1e-4, atol=1e-4)
    # Non-Q/K weights remain unmodified
    torch.testing.assert_close(mapped["layers.0.attention.wv.weight"], wk_hf)


def test_action_token_sequence_ordering_includes_prefill_without_eos():
    """OpenVLALanguageModel.__call__ must return [action_0 (prefill), action_1..6 (decode)] with no EOS."""
    vocab_size = 32064
    expected_actions = [31850, 31861, 31872, 31883, 31894, 31905, 31916]
    eos_token_id = 2

    def _make_logits(tok_id: int) -> torch.Tensor:
        logits = torch.full((1, 1, vocab_size), -10.0, dtype=torch.float32)
        logits[0, 0, tok_id] = 10.0
        return logits

    prefill_logits_1d = _make_logits(expected_actions[0]).squeeze(0)  # [1, vocab_size]

    fake_rope = SimpleNamespace(
        cos_matrix_prefill=MagicMock(__getitem__=MagicMock(return_value="cos_prefill")),
        sin_matrix_prefill=MagicMock(__getitem__=MagicMock(return_value="sin_prefill")),
    )
    fake_transformer = MagicMock()
    fake_transformer.rope_setup = fake_rope
    fake_transformer.forward.return_value = "tt_prefill_logits"
    fake_transformer.process_output_prefill.return_value = prefill_logits_1d

    decode_queue = expected_actions[1:] + [eos_token_id]
    decode_calls = []

    def _fake_decode(out_tok, current_pos, **kwargs):
        idx = len(decode_calls)
        decode_calls.append(int(out_tok.item()))
        return (_make_logits(decode_queue[idx]), None)

    lm = open_vla.OpenVLALanguageModel.__new__(open_vla.OpenVLALanguageModel)
    lm.model = [fake_transformer]
    lm.tt_kv_cache = ["kv_cache"]
    lm.page_table = torch.zeros((1, 16), dtype=torch.int32)
    lm.num_actions = 7
    lm.generator = SimpleNamespace(decode_forward=_fake_decode)
    lm._get_prefill_user_page_table = MagicMock(return_value=torch.zeros((1, 8), dtype=torch.int32))

    tt_inputs = MagicMock()
    tt_inputs.shape = (1, 1, 20, 4096)
    tt_inputs.device.return_value = MagicMock()

    orig_from_torch = ttnn.from_torch
    orig_pad = ttnn.pad
    orig_from_device = ttnn.from_device
    try:
        ttnn.from_torch = MagicMock(return_value="tt_page_table")
        ttnn.pad = MagicMock(side_effect=lambda t, *a, **kw: t)
        ttnn.from_device = MagicMock(side_effect=lambda t: t)
        output_toks = lm(inputs_embeds=tt_inputs, use_cache=True, past_key_values="dummy_cache")
    finally:
        ttnn.from_torch = orig_from_torch
        ttnn.pad = orig_pad
        ttnn.from_device = orig_from_device

    assert len(output_toks) == 7
    assert len(decode_calls) == 6
    assert lm._last_generated_tokens == expected_actions
    assert eos_token_id not in lm._last_generated_tokens
    predicted = [int(torch.argmax(tok, dim=-1).item()) for tok in output_toks]
    assert predicted == expected_actions


def test_runtime_blockers_n150_compatibility():
    """Validate pre-init _supports_sdpa, tie_weights(*args, **kwargs), and single-chip device_params."""
    # _supports_sdpa before self.language_model is attached
    uninit = open_vla.PrismaticForConditionalGeneration.__new__(open_vla.PrismaticForConditionalGeneration)
    assert uninit._supports_sdpa is False

    lm = open_vla.OpenVLALanguageModel.__new__(open_vla.OpenVLALanguageModel)
    assert lm.tie_weights(recompute_mapping=True) is None
    uninit.language_model = lm
    assert uninit._supports_sdpa is False
    assert uninit.tie_weights(missing_keys=[]) is None

    # timm version >= 0.9.10 check accepts current repo requirement (1.0.x+)
    from packaging.version import Version

    assert Version(timm.__version__) >= Version("0.9.10")

    # Single-chip device_params must not request FABRIC_1D
    params = open_vla._get_openvla_device_params(is_single_device=True)
    assert params.get("fabric_config") is None


def test_siglip_patch_embeddings_with_pos_embed_device(device):
    """On-device regression test: SigLIP patch embeddings + pos_embed matches timm reference."""
    torch.manual_seed(7)
    model = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=False).eval()
    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        ref = model._pos_embed(model.patch_embed(pixel_values))

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
        custom_preprocessor=tt_optimized_openvla_vision.custom_preprocessor_siglip,
    )
    nhwc = torch.nn.functional.pad(pixel_values.permute(0, 2, 3, 1), (0, 1, 0, 0, 0, 0, 0, 0))
    tt_pixels = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_out = tt_optimized_openvla_vision.siglip_patch_embeddings(
        tt_pixels,
        parameters=parameters.patch_embed.patch_embeddings,
        pos_embed=parameters.pos_embed,
    )
    assert_with_pcc(ref, ttnn.to_torch(tt_out), 0.9995)
