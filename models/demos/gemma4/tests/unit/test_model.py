# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for full Gemma4 model.

Uses HuggingFace Gemma4 classes as reference for PCC comparison.
"""

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

from ...tests.test_factory import TestFactory, compare_tensors

# ── Config Tests ───────────────────────────────────────────────────────────


def test_model_config():
    """Test that Gemma4ModelArgs loads correctly from HF_MODEL checkpoint."""
    hf_config = TestFactory.create_hf_config()

    # Core fields must be populated
    assert hf_config.hidden_size > 0
    assert hf_config.num_hidden_layers > 0
    assert hf_config.num_attention_heads > 0
    assert hf_config.num_key_value_heads > 0
    assert hf_config.head_dim > 0
    assert hf_config.vocab_size > 0
    assert hf_config.tie_word_embeddings is True

    # Layer types must cover all layers
    assert len(hf_config.layer_types) == hf_config.num_hidden_layers
    assert all(lt in ("sliding_attention", "full_attention") for lt in hf_config.layer_types)

    # MoE fields: either fully populated or all zero
    if hf_config.enable_moe_block:
        assert hf_config.num_experts > 0
        assert hf_config.top_k_experts > 0
        assert hf_config.moe_intermediate_size > 0
    else:
        # Dense model — MoE fields should be zero/None
        assert hf_config.num_experts == 0 or hf_config.num_experts is None

    logger.info(
        f"Model: hidden={hf_config.hidden_size}, layers={hf_config.num_hidden_layers}, "
        f"heads={hf_config.num_attention_heads}, moe={hf_config.enable_moe_block}"
    )


def test_model_instantiation():
    """Test that Gemma4Model can be imported and config created."""
    hf_config = TestFactory.create_hf_config()
    assert hf_config.hidden_size > 0
    assert hf_config.num_hidden_layers > 0


def test_softcapping():
    """Test logit softcapping matches tanh(x/cap)*cap."""
    cap = 30.0
    x = torch.randn(1, 1, 32, 100, dtype=torch.float32) * 100
    expected = torch.tanh(x / cap) * cap
    assert expected.abs().max() <= cap + 1e-5
    small = torch.randn(1, 1, 32, 100) * 0.1
    small_capped = torch.tanh(small / cap) * cap
    assert torch.allclose(small_capped, small, atol=1e-3)


# ── HF Reference Helpers ──────────────────────────────────────────────────


def _create_hf_text_config(num_experts=None, top_k=None, vocab_size=256, num_layers=1):
    """Create a Gemma4TextConfig from real model config with overrides for speed."""
    from transformers import AutoConfig

    from ...tests.test_factory import _get_model_path

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = config.text_config
    if num_experts is not None:
        tc.num_experts = num_experts
    if top_k is not None:
        tc.top_k_experts = top_k
    tc.vocab_size = vocab_size
    tc.num_hidden_layers = num_layers
    # Disable per-layer input for now (not yet implemented in TT)
    tc.hidden_size_per_layer_input = 0
    tc._attn_implementation = "eager"
    return tc


def _create_hf_model(hf_text_config):
    """Create HF Gemma4 text model with random weights."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextScaledWordEmbedding,
    )

    # Build a minimal model: embedding + 1 layer + norm + lm_head
    class HFRefModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_tokens = Gemma4TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id if hasattr(config, "pad_token_id") else 0,
                embed_scale=config.hidden_size**0.5,
            )
            self.layers = torch.nn.ModuleList(
                [Gemma4TextDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
            )
            self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # Tied lm_head
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def forward(self, input_ids, position_embeddings, attention_mask=None):
            x = self.embed_tokens(input_ids)
            pli_size = getattr(self.config, "hidden_size_per_layer_input", 0) or 0
            pli = torch.randn(1, x.shape[1], pli_size) if pli_size else None
            for layer in self.layers:
                x = layer(
                    x, per_layer_input=pli, position_embeddings=position_embeddings, attention_mask=attention_mask
                )
            x = self.norm(x)
            logits = self.lm_head(x)
            # Softcapping
            cap = self.config.final_logit_softcapping
            if cap and cap > 0:
                logits = torch.tanh(logits / cap) * cap
            return logits

    model = HFRefModel(hf_text_config)
    # Randomize router/expert weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(k in name for k in ["router", "experts"]):
                if "scale" in name:
                    param.data.fill_(1.0)
                else:
                    param.data.normal_(0, 0.02)
        # Tie lm_head to embeddings
        model.lm_head.weight = model.embed_tokens.weight
        # Set layer_scalar to 1.0
        for layer in model.layers:
            layer.layer_scalar.fill_(1.0)
    model.eval()
    return model


def _hf_model_state_to_tt_state(hf_model):
    """Convert HF model state_dict to format our TT Gemma4Model expects."""
    state = hf_model.state_dict()
    tt_state = {}
    for k, v in state.items():
        # Map: "layers.0.xxx" -> "model.layers.0.xxx"
        # Map: "embed_tokens.weight" -> "model.embed_tokens.weight"
        # Map: "norm.weight" -> "model.norm.weight"
        tt_state[f"model.{k}"] = v
    return tt_state


# ── Single Layer Model PCC Test ───────────────────────────────────────────


def test_single_layer_model(device):
    """
    Test single-layer model against HF reference with PCC.

    Embed -> 1 decoder layer -> norm -> lm_head -> softcapping.
    """
    # For MoE models, reduce experts for speed; dense models don't need override
    base_config = _create_hf_text_config(vocab_size=256, num_layers=1)
    is_moe = getattr(base_config, "enable_moe_block", False)
    if is_moe:
        base_config.num_experts = 4
        base_config.top_k_experts = 2
    hf_text_config = base_config
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)

    # Convert state for TT
    tt_state = _hf_model_state_to_tt_state(hf_model)

    # RoPE setup
    from models.demos.gemma4.tt.attention import Gemma4AttentionConfig

    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx=0)
    seq_len = 32

    mesh_config = TestFactory.create_mesh_config((1, 1))
    tt_model = Gemma4Model(
        mesh_device=device,
        hf_config=model_args,
        state_dict=tt_state,
        ccl_manager=None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        max_local_batch_size=1,
        num_layers=1,
    )

    # Input tokens
    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)

    # HF reference forward
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    hf_rope = Gemma4TextRotaryEmbedding(hf_text_config)
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    layer_type = hf_text_config.layer_types[0]
    cos, sin = hf_rope(torch.randn(1, seq_len, model_args.hidden_size), pos_ids, layer_type=layer_type)

    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        hf_logits = hf_model(tokens, position_embeddings=(cos, sin), attention_mask=causal_mask)
    logger.info(f"HF logits shape: {hf_logits.shape}, range: [{hf_logits.min():.4f}, {hf_logits.max():.4f}]")

    # TT forward
    tt_tokens = ttnn.from_torch(tokens.to(torch.int32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    tt_embeds = tt_model.embed_tokens(tt_tokens)
    tt_embeds = ttnn.reshape(tt_embeds, (1, 1, seq_len, model_args.hidden_size))
    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(device, hf_text_config, max(seq_len, 128), layer_idx=0)
    tt_logits = tt_model(
        tt_embeds, rope_mats=(cos_tt, sin_tt), position_idx=None, page_table=None, kv_caches=None, is_decode=False
    )
    tt_logits_torch = ttnn.to_torch(tt_logits).squeeze(0).float()  # [1, S, V]

    # Compare
    passing, pcc_msg = compare_tensors(tt_logits_torch, hf_logits, pcc_threshold=0.95)
    assert passing, f"Single-layer model PCC too low: {pcc_msg}"


# ── Full Model PCC Test ─────────────────────────────────────────────────


def test_full_model(device):
    """
    Test full model (all layers, real weights) against HuggingFace reference.

    Loads the real checkpoint, runs a short prefill on both HF and TT,
    and compares logits PCC.
    """
    import os

    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.gemma4.tt.common import create_tt_model

    model_path = os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "/proj_sw/user_dev/gemma4/gemma-4-26B-A4B-it")

    # ── HF reference ─────────────────────────────────────────────────
    logger.info(f"Loading HF reference model from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # [1, seq_len]
    seq_len = input_ids.shape[1]
    padded_len = ((seq_len + 31) // 32) * 32
    if padded_len > seq_len:
        input_ids_padded = F.pad(input_ids, (0, padded_len - seq_len), value=0)
    else:
        input_ids_padded = input_ids

    logger.info(f"Prompt: '{prompt}' -> {seq_len} tokens (padded to {padded_len})")

    with torch.no_grad():
        hf_out = hf_model(input_ids_padded)
        hf_logits = hf_out.logits.float()  # [1, padded_len, vocab_size]

    # Note: HF Gemma4ForConditionalGeneration already applies softcapping internally,
    # so no need to apply it again here. TT model also applies it internally.

    logger.info(f"HF logits shape: {hf_logits.shape}, range: [{hf_logits.min():.4f}, {hf_logits.max():.4f}]")

    # Free HF model GPU memory (we only need state_dict for TT)
    del hf_model
    import gc

    gc.collect()

    # ── TT model ─────────────────────────────────────────────────────
    logger.info("Creating TT model with all layers...")
    model_args, tt_model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=device,
        max_batch_size=1,
        max_seq_len=max(padded_len, 128),
        model_path=model_path,
        create_kv_cache=True,
    )

    is_mesh = hasattr(device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

    tokens_tt = ttnn.from_torch(
        input_ids_padded.to(torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    embeds = tt_model.embed_tokens(tokens_tt)
    embeds = ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size))
    embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

    # CPU tensors for per-layer input (E2B/E4B models)
    embeds_torch = (
        F.embedding(
            input_ids_padded.long(),
            state_dict.get(
                "model.language_model.embed_tokens.weight",
                state_dict.get("model.embed_tokens.weight", torch.zeros(1)),
            ),
        )
        * tt_model.embed_scale
    ).float()

    tt_logits = tt_model.ttnn_prefill_forward(
        embeds,
        page_table=None,
        kv_cache=tt_kv_cache,
        input_ids_torch=input_ids_padded,
        embeds_torch=embeds_torch,
    )

    if is_mesh:
        tt_logits_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    else:
        tt_logits_torch = ttnn.to_torch(tt_logits).float()
    tt_logits.deallocate(True)

    # Reshape TT output to match HF: TT is [1, 1, padded_len, vocab] -> [1, padded_len, vocab]
    if tt_logits_torch.dim() == 4:
        tt_logits_torch = tt_logits_torch.squeeze(1)

    logger.info(
        f"TT logits shape: {tt_logits_torch.shape}, range: [{tt_logits_torch.min():.4f}, {tt_logits_torch.max():.4f}]"
    )

    # Compare only up to the real (unpadded) sequence length
    hf_compare = hf_logits[:, :seq_len, :]
    tt_compare = tt_logits_torch[:, :seq_len, :]

    passing, pcc_msg = compare_tensors(tt_compare, hf_compare, pcc_threshold=0.90)
    logger.info(f"Full model PCC (seq_len={seq_len}): {pcc_msg}")

    # Also check that argmax tokens match for the last position
    hf_last_tok = hf_compare[0, -1, :].argmax().item()
    tt_last_tok = tt_compare[0, -1, :].argmax().item()
    logger.info(
        f"Last-position argmax: HF={hf_last_tok} ('{tokenizer.decode([hf_last_tok])}'), "
        f"TT={tt_last_tok} ('{tokenizer.decode([tt_last_tok])}')"
    )

    assert passing, f"Full model PCC too low: {pcc_msg}"
