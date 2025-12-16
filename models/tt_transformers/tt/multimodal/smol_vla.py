"""
smol_vla.py

SmolVLA (Small Vision-Language-Action) model implementation for TT hardware.
Based on HuggingFace's SmolVLA architecture: https://huggingface.co/lerobot/smolvla_base

SmolVLA is a 450M parameter VLA model that combines:
- SmolVLM2-500M vision-language backbone (vision encoder + LLaMA-like text model)
- Action expert layers (16 additional transformer layers)
- Action prediction heads (state projection, action in/out projection, time MLPs)

This implementation mirrors the OpenVLA pattern for TT hardware execution.

References:
    - SmolVLA Paper: https://arxiv.org/abs/2506.01844
    - SmolVLA HF: https://huggingface.co/lerobot/smolvla_base
    - SmolVLM2: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoConfig, AutoProcessor

import ttnn
from models.tt_transformers.tt.common import create_tt_model, get_block_size, get_padded_prefill_len, num_blocks_in_seq
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import ModelArgs

# Get Logger
logger = logging.getLogger(__name__)


# ============================================================================
# Performance Checkpointing (same pattern as OpenVLA)
# ============================================================================
class PerfCheckpoints:
    checkpoints: ClassVar[List[Dict[str, int]]] = None

    def __init__(self):
        self.times = {}
        if self.checkpoints is None:
            self.checkpoints = []
        self.present_keys_counter = {}

    def checkpoint(self, key):
        if key not in self.present_keys_counter:
            self.present_keys_counter[key] = 0
            new_key = f"{key}_{self.present_keys_counter[key]}"
            self.times[new_key] = time.time()
        else:
            self.present_keys_counter[key] += 1
            new_key = f"{key}_{self.present_keys_counter[key]}"
            self.times[new_key] = time.time()

    def get_pairs(self):
        pairs = []
        keys = list(key for key in self.present_keys_counter if key.startswith("start"))
        for key in keys:
            end_key = key.replace("start", "end")
            if end_key in self.present_keys_counter:
                pairs.append((key, end_key))
        return pairs

    def analyze(self, pairs=None):
        results = {}
        if pairs is None:
            pairs = self.get_pairs()
        for pair in pairs:
            assert len(pair) == 2, "Each pair must contain exactly two keys."
            assert pair[0] in self.present_keys_counter, f"Key {pair[0]} not found in checkpoints."
            assert pair[1] in self.present_keys_counter, f"Key {pair[1]} not found in checkpoints."
            assert (
                self.present_keys_counter[pair[0]] == self.present_keys_counter[pair[1]]
            ), f"Key {pair[0]} and {pair[1]} must have the same number of occurrences."
            for counter in range(self.present_keys_counter[pair[0]] + 1):
                key1 = f"{pair[0]}_{counter}"
                key2 = f"{pair[1]}_{counter}"
                results[f"{key1}->{key2}"] = self.times[key2] - self.times[key1]
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        return results

    def reset(self):
        if len(self.times) > 0:
            self.checkpoints.append(self.times)
            self.times = {}
            self.present_keys_counter = {}


CHECKPOINTS = PerfCheckpoints()


# ============================================================================
# SmolVLA Configuration
# ============================================================================
@dataclass
class SmolVLAConfig:
    """Configuration for SmolVLA model."""

    # VLM backbone
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # Text model config (LLaMA-like)
    text_hidden_size: int = 960
    text_num_layers: int = 16  # SmolVLM2-500M has 16 text layers
    text_num_attention_heads: int = 15
    text_num_kv_heads: int = 5
    text_vocab_size: int = 49280
    text_intermediate_size: int = 2560
    text_rms_norm_eps: float = 1e-5

    # Vision model config
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_attention_heads: int = 12
    vision_image_size: int = 512
    vision_patch_size: int = 16
    vision_intermediate_size: int = 3072

    # Expert layers config
    num_expert_layers: int = 16
    expert_hidden_size: int = 720  # Expert layers use smaller hidden dim
    expert_intermediate_size: int = 2048
    expert_num_heads: int = 12  # 720 / 60 = 12 heads
    expert_num_kv_heads: int = 4  # 320 / 80 = 4 KV heads (GQA)

    # Action config
    n_action_steps: int = 50
    max_action_dim: int = 32
    max_state_dim: int = 32
    chunk_size: int = 50

    # Processing config
    tokenizer_max_length: int = 48

    @classmethod
    def from_pretrained(cls, repo_id: str = "lerobot/smolvla_base"):
        """Load config from HuggingFace repo."""
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Also load VLM config
        vlm_name = config_dict.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        vlm_config = AutoConfig.from_pretrained(vlm_name, trust_remote_code=True)

        return cls(
            vlm_model_name=vlm_name,
            text_hidden_size=vlm_config.text_config.hidden_size,
            text_num_layers=vlm_config.text_config.num_hidden_layers,
            text_num_attention_heads=vlm_config.text_config.num_attention_heads,
            text_num_kv_heads=vlm_config.text_config.num_key_value_heads,
            text_vocab_size=vlm_config.text_config.vocab_size,
            text_intermediate_size=vlm_config.text_config.intermediate_size,
            vision_hidden_size=vlm_config.vision_config.hidden_size,
            vision_num_layers=vlm_config.vision_config.num_hidden_layers,
            vision_num_attention_heads=vlm_config.vision_config.num_attention_heads,
            vision_image_size=vlm_config.vision_config.image_size,
            vision_patch_size=vlm_config.vision_config.patch_size,
            num_expert_layers=config_dict.get("num_expert_layers", 0) or config_dict.get("num_vlm_layers", 16),
            n_action_steps=config_dict.get("n_action_steps", 50),
            max_action_dim=config_dict.get("max_action_dim", 32),
            max_state_dim=config_dict.get("max_state_dim", 32),
            chunk_size=config_dict.get("chunk_size", 50),
            tokenizer_max_length=config_dict.get("tokenizer_max_length", 48),
        )


# ============================================================================
# Weight Mapping Functions
# ============================================================================
def map_smolvla_to_tt_text_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map SmolVLA HF state dict keys to TT Meta-style keys for the text model.

    SmolVLA text model keys look like:
        model.vlm_with_expert.vlm.model.text_model.layers.{layer}.self_attn.q_proj.weight
        model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight
        model.vlm_with_expert.vlm.lm_head.weight

    TT Meta-style keys look like:
        layers.{layer}.attention.wq.weight
        tok_embeddings.weight
        output.weight
    """
    hf_to_meta = {
        # Top level mappings
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight": "tok_embeddings.weight",
        "model.vlm_with_expert.vlm.model.text_model.norm.weight": "norm.weight",
        "model.vlm_with_expert.vlm.lm_head.weight": "output.weight",
    }

    # Layer-level mappings template
    layer_mappings = {
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
    }

    meta_state_dict = {}
    text_prefix = "model.vlm_with_expert.vlm.model.text_model."

    for key, tensor in state_dict.items():
        if key in hf_to_meta:
            meta_state_dict[hf_to_meta[key]] = tensor
        elif key.startswith(text_prefix + "layers."):
            # Extract layer number
            rest = key[len(text_prefix + "layers.") :]
            parts = rest.split(".", 1)
            layer_num = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""

            if suffix in layer_mappings:
                meta_key = f"layers.{layer_num}.{layer_mappings[suffix]}"
                meta_state_dict[meta_key] = tensor

    return meta_state_dict


def map_smolvla_expert_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map SmolVLA expert layer keys to a separate state dict.

    Expert keys look like:
        model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.q_proj.weight
    """
    expert_state_dict = {}
    expert_prefix = "model.vlm_with_expert.lm_expert."

    layer_mappings = {
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
    }

    for key, tensor in state_dict.items():
        if key.startswith(expert_prefix + "layers."):
            rest = key[len(expert_prefix + "layers.") :]
            parts = rest.split(".", 1)
            layer_num = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""

            if suffix in layer_mappings:
                meta_key = f"layers.{layer_num}.{layer_mappings[suffix]}"
                expert_state_dict[meta_key] = tensor

    return expert_state_dict


def extract_vision_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract vision encoder weights from SmolVLA state dict.

    State dict keys look like:
        model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight

    SiglipVisionModel expects keys like:
        vision_model.embeddings.patch_embedding.weight
    """
    vision_state_dict = {}
    vision_prefix = "model.vlm_with_expert.vlm.model.vision_model."

    for key, tensor in state_dict.items():
        if key.startswith(vision_prefix):
            # Strip the prefix and add "vision_model." to match SiglipVisionModel structure
            new_key = "vision_model." + key[len(vision_prefix) :]
            vision_state_dict[new_key] = tensor

    return vision_state_dict


def extract_connector_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract connector/projector weights from SmolVLA state dict."""
    connector_state_dict = {}
    connector_prefix = "model.vlm_with_expert.vlm.model.connector."

    for key, tensor in state_dict.items():
        if key.startswith(connector_prefix):
            new_key = key[len(connector_prefix) :]
            connector_state_dict[new_key] = tensor

    return connector_state_dict


def extract_action_heads_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract action head weights from SmolVLA state dict."""
    action_heads = {}
    action_prefixes = [
        "model.action_in_proj.",
        "model.action_out_proj.",
        "model.action_time_mlp_in.",
        "model.action_time_mlp_out.",
        "model.state_proj.",
    ]

    for key, tensor in state_dict.items():
        for prefix in action_prefixes:
            if key.startswith(prefix):
                new_key = key[len("model.") :]
                action_heads[new_key] = tensor
                break

    return action_heads


# ============================================================================
# SmolVLA ModelArgs for TT Transformer
# ============================================================================
def get_SmolVLAModelArgs(state_dict, config: SmolVLAConfig):
    """Create a ModelArgs subclass configured for SmolVLA's text model."""

    class SmolVLATextModelArgs(ModelArgs):
        def __init__(self, *args, **kwargs):
            # Set HF_MODEL to SmolVLM2 for config loading
            os.environ["HF_MODEL"] = config.vlm_model_name
            super().__init__(*args, **kwargs)

        def _set_params_from_dict(self, hf_config):
            # Override with SmolVLA text config
            new_config = {
                "hidden_size": config.text_hidden_size,
                "num_hidden_layers": config.text_num_layers,
                "num_attention_heads": config.text_num_attention_heads,
                "num_key_value_heads": config.text_num_kv_heads,
                "vocab_size": config.text_vocab_size,
                "intermediate_size": config.text_intermediate_size,
                "rms_norm_eps": config.text_rms_norm_eps,
                "hidden_act": "silu",
                "max_position_embeddings": 2048,
                "rope_theta": 10000.0,
                "model_type": "llama",
                "torch_dtype": "bfloat16",
            }
            # Merge with any existing config
            text_config = hf_config.get("text_config", hf_config)
            for key, value in text_config.items():
                if key not in new_config:
                    new_config[key] = value

            return super()._set_params_from_dict(new_config)

        def load_state_dict(self):
            if state_dict is None:
                return super().load_state_dict()
            # Return the already-mapped state dict (passed in from SmolVLATextModel)
            return state_dict

    return SmolVLATextModelArgs


# ============================================================================
# TT-Optimized Vision Encoder Operations
# ============================================================================
def smolvla_patch_embeddings_tt(
    pixel_values,
    proj_weight,
    proj_bias,
    patch_size: int = 16,
    ttnn_device=None,
):
    """
    TT-optimized patch embedding for SmolVLA.

    SmolVLA uses 512x512 images with 16x16 patches = 1024 patches.
    Hidden size = 768.
    """
    # pixel_values shape: [batch, H, W, C+pad] (NHWC, padded to 4 channels)
    batch_size, img_h, img_w, img_c = pixel_values.shape
    patch_count = img_h // patch_size  # 32 for 512x512
    patch_count_all = patch_count * patch_count  # 1024 patches
    stride_h = patch_size
    stride_w = 1

    # Reshape for fold operation
    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    # Linear projection
    patch_embedding_output = ttnn.linear(
        pixel_values,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(pixel_values)

    # Reshape to [batch, num_patches, hidden_size]
    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, -1))

    return patch_embedding_output


def smolvla_attention_tt(
    hidden_states,
    attention_mask,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    num_heads: int = 12,
):
    """
    TT-optimized attention for SmolVLA vision encoder.

    SmolVLA vision uses 12 attention heads with 768 hidden size (64 head dim).
    """
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # Combined QKV projection
    query_key_value = ttnn.linear(
        hidden_states,
        qkv_weight,
        bias=qkv_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.reallocate(hidden_states)

    # Split into Q, K, V and reshape for multi-head attention
    (query, key, value) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    value = ttnn.reallocate(value)

    # Attention scores: Q @ K^T
    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # Scale and softmax
    # Use regular softmax since vision encoder doesn't need masking
    scale_factor = 1.0 / math.sqrt(head_size)
    attention_scores = ttnn.mul(attention_scores, scale_factor)
    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    # Attention output: softmax(Q @ K^T) @ V
    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    # Concatenate heads
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Output projection
    self_output = ttnn.linear(
        context_layer,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(context_layer)

    return self_output


def smolvla_mlp_tt(
    hidden_states,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
):
    """
    TT-optimized MLP for SmolVLA vision encoder.

    Uses GELU activation between fc1 and fc2.
    """
    # FC1 with GELU activation
    intermediate = ttnn.linear(
        hidden_states,
        fc1_weight,
        bias=fc1_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        activation="gelu",
    )

    # FC2
    output = ttnn.linear(
        intermediate,
        fc2_weight,
        bias=fc2_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(intermediate)

    return output


def smolvla_vision_layer_tt(
    hidden_states,
    attention_mask,
    layer_params,
    num_heads: int = 12,
):
    """
    Single TT-optimized vision transformer layer.

    Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    # Pre-attention LayerNorm
    normed = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["norm1_weight"],
        bias=layer_params["norm1_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Self-attention
    attn_output = smolvla_attention_tt(
        normed,
        attention_mask,
        layer_params["qkv_weight"],
        layer_params["qkv_bias"],
        layer_params["proj_weight"],
        layer_params["proj_bias"],
        num_heads=num_heads,
    )

    # Residual connection
    hidden_states = ttnn.add(
        attn_output,
        hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(attn_output)

    # Pre-MLP LayerNorm
    normed = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["norm2_weight"],
        bias=layer_params["norm2_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # MLP
    mlp_output = smolvla_mlp_tt(
        normed,
        layer_params["fc1_weight"],
        layer_params["fc1_bias"],
        layer_params["fc2_weight"],
        layer_params["fc2_bias"],
    )

    # Residual connection
    hidden_states = ttnn.add(
        mlp_output,
        hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(mlp_output)

    return hidden_states


def smolvla_vision_encoder_tt(
    pixel_values,
    embed_params,
    layer_params_list,
    position_embedding,
    num_heads: int = 12,
    num_layers: int = 12,
):
    """
    Full TT-optimized vision encoder forward pass.
    """
    # Patch embeddings
    hidden_states = smolvla_patch_embeddings_tt(
        pixel_values,
        embed_params["proj_weight"],
        embed_params["proj_bias"],
        patch_size=16,
    )

    # Add position embeddings
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
    if position_embedding is not None:
        hidden_states = ttnn.add(
            hidden_states,
            position_embedding,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

    # Create dummy attention mask (no masking for vision)
    attention_mask = None

    # Encoder layers
    for i in range(min(num_layers, len(layer_params_list))):
        hidden_states = smolvla_vision_layer_tt(
            hidden_states,
            attention_mask,
            layer_params_list[i],
            num_heads=num_heads,
        )

    return hidden_states


# ============================================================================
# TT VLM K/V Cache Functions
# ============================================================================
def smolvla_vlm_kv_projection_tt(
    hidden_states,
    k_weight,
    v_weight,
):
    """
    Project VLM hidden states to K/V for expert cross-attention.

    Args:
        hidden_states: [batch, seq_len, 960] VLM hidden states
        k_weight: K projection weight [960, 320]
        v_weight: V projection weight [960, 320]

    Returns:
        k: [batch, seq_len, 320]
        v: [batch, seq_len, 320]
    """
    k = ttnn.linear(
        hidden_states,
        k_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    v = ttnn.linear(
        hidden_states,
        v_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    return k, v


def smolvla_vlm_layer_forward_tt(
    hidden_states,
    layer_params,
    k_proj_weight,
    v_proj_weight,
    num_heads: int = 15,
    head_dim: int = 64,
):
    """
    Single VLM layer forward pass that also returns K/V for expert cross-attention.

    VLM has 15 attention heads with head_dim=64 -> 960 hidden size.
    K/V output is 320-dim (5 KV heads × 64 head_dim).

    Returns:
        hidden_states: [batch, seq_len, 960] output hidden states
        kv_cache: (k, v) tuple where k, v are [batch, seq_len, 320]
    """
    residual = hidden_states

    # Pre-attention layernorm
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params.norm1_weight,
        bias=layer_params.norm1_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Compute K/V for expert cross-attention using VLM's K/V projections
    # These are the 960->320 projections
    k_cache, v_cache = smolvla_vlm_kv_projection_tt(hidden_states, k_proj_weight, v_proj_weight)

    # VLM self-attention (Q, K, V all from hidden_states)
    # For simplicity, we do a basic attention here
    # In production, this would use the full VLM attention implementation
    q = ttnn.linear(
        hidden_states,
        layer_params.q_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    k = ttnn.linear(
        hidden_states,
        layer_params.k_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    v = ttnn.linear(
        hidden_states,
        layer_params.v_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    # Reshape for multi-head attention
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    num_kv_heads = 5  # GQA

    q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT)
    q = ttnn.reshape(q, (batch_size, seq_len, num_heads, head_dim))
    q = ttnn.permute(q, (0, 2, 1, 3))
    q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)

    k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT)
    k = ttnn.reshape(k, (batch_size, seq_len, num_kv_heads, head_dim))
    k = ttnn.permute(k, (0, 2, 1, 3))
    k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)

    v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT)
    v = ttnn.reshape(v, (batch_size, seq_len, num_kv_heads, head_dim))
    v = ttnn.permute(v, (0, 2, 1, 3))
    v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)

    # GQA: repeat K/V to match Q heads (5 -> 15)
    n_rep = num_heads // num_kv_heads  # 3
    k = ttnn.repeat_interleave(k, n_rep, dim=1)
    v = ttnn.repeat_interleave(v, n_rep, dim=1)

    # Attention
    scale = 1.0 / math.sqrt(head_dim)
    k_t = ttnn.permute(k, (0, 1, 3, 2))
    scores = ttnn.matmul(q, k_t, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    scores = ttnn.mul(scores, scale)
    scores = ttnn.softmax(scores, dim=-1)

    attn_output = ttnn.matmul(scores, v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(scores)
    ttnn.deallocate(k_t)

    # Reshape back
    attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
    attn_output = ttnn.to_layout(attn_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, num_heads * head_dim))
    attn_output = ttnn.to_layout(attn_output, layout=ttnn.TILE_LAYOUT)

    # Output projection
    attn_output = ttnn.linear(
        attn_output,
        layer_params.o_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    # Residual
    hidden_states = ttnn.add(residual, attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(attn_output)

    # MLP
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params.norm2_weight,
        bias=layer_params.norm2_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    gate = ttnn.linear(
        hidden_states,
        layer_params.gate_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    up = ttnn.linear(
        hidden_states,
        layer_params.up_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    mlp_output = ttnn.mul(ttnn.silu(gate), up)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    mlp_output = ttnn.linear(
        mlp_output,
        layer_params.down_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    hidden_states = ttnn.add(residual, mlp_output, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(mlp_output)

    return hidden_states, (k_cache, v_cache)


# ============================================================================
# TT Expert Layer Functions
# ============================================================================
def smolvla_expert_attention_tt(
    hidden_states,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    num_q_heads: int = 12,
    num_kv_heads: int = 4,
    is_cross_attn: bool = False,
    cross_attn_k: ttnn.Tensor = None,
    cross_attn_v: ttnn.Tensor = None,
):
    """
    TT-optimized attention for SmolVLA expert layers.

    Expert layers have asymmetric attention:
    - Q: 720 -> 960 (12 heads × 80 head_dim)
    - K/V: 720 -> 320 (4 heads × 80 head_dim) for self-attn
         : 320 -> 320 for cross-attn (K from cross_attn_k, V from cross_attn_v)
    - GQA: repeat K/V 3x to match Q heads
    """
    head_dim = 80  # Fixed head dimension

    # Q projection: [batch, seq, 720] -> [batch, seq, 960]
    query = ttnn.linear(
        hidden_states,
        q_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    # K/V source depends on layer type
    if is_cross_attn and cross_attn_k is not None:
        k_input = cross_attn_k
        v_input = cross_attn_v if cross_attn_v is not None else cross_attn_k
    else:
        k_input = hidden_states
        v_input = hidden_states

    # K projection
    key = ttnn.linear(
        k_input,
        k_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    # V projection
    value = ttnn.linear(
        v_input,
        v_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    # Reshape for multi-head attention
    # Q: [batch, q_seq, 960] -> [batch, 12, q_seq, 80]
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    batch_size = query.shape[0]
    q_seq_len = query.shape[1]  # Q sequence length (e.g., 50 for action chunk)
    query = ttnn.reshape(query, (batch_size, q_seq_len, num_q_heads, head_dim))
    query = ttnn.permute(query, (0, 2, 1, 3))
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)

    # K/V: [batch, kv_seq, 320] -> [batch, 4, kv_seq, 80] -> repeat to [batch, 12, kv_seq, 80]
    # IMPORTANT: For cross-attention, kv_seq_len can differ from q_seq_len!
    # Cross-attn: Q attends to VLM context (kv_seq = 64+ tokens)
    # Self-attn: Q attends to self (kv_seq = q_seq)
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    kv_seq_len = key.shape[1]  # K/V sequence length (may differ from Q for cross-attn)
    key = ttnn.reshape(key, (batch_size, kv_seq_len, num_kv_heads, head_dim))
    key = ttnn.permute(key, (0, 2, 1, 3))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)

    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, kv_seq_len, num_kv_heads, head_dim))
    value = ttnn.permute(value, (0, 2, 1, 3))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)

    # GQA: repeat K/V to match Q heads (4 -> 12)
    n_rep = num_q_heads // num_kv_heads  # 3
    key = ttnn.repeat_interleave(key, n_rep, dim=1)
    value = ttnn.repeat_interleave(value, n_rep, dim=1)

    # Attention: Q @ K^T / sqrt(head_dim)
    key_t = ttnn.permute(key, (0, 1, 3, 2))
    attention_scores = ttnn.matmul(
        query,
        key_t,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)
    ttnn.deallocate(key_t)

    # Apply scaling
    scale_factor = 1.0 / math.sqrt(head_dim)
    attention_scores = ttnn.mul(attention_scores, scale_factor)

    # Apply causal masking for SELF-ATTENTION layers only
    # Cross-attention: action tokens can attend to ALL VLM context tokens (no masking)
    # Self-attention: action tokens can only attend to PAST action tokens (causal mask)
    if not is_cross_attn:
        # Create causal mask: [1, 1, q_seq, kv_seq] where lower triangle = True
        # For self-attn: q_seq == kv_seq
        causal_mask = torch.tril(torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_seq, kv_seq]

        # Convert mask to TT - where mask is False, set to -inf
        # ttnn.where not available, so we use additive masking
        mask_value = -1e9
        mask_tensor = torch.where(causal_mask, 0.0, mask_value).to(torch.bfloat16)
        mask_tt = ttnn.from_torch(
            mask_tensor,
            device=attention_scores.device(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        attention_scores = ttnn.add(attention_scores, mask_tt)
        ttnn.deallocate(mask_tt)

    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    ttnn.deallocate(attention_scores)

    # Attention output: softmax(Q @ K^T) @ V
    context = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    # Reshape back: [batch, 12, q_seq, 80] -> [batch, q_seq, 960]
    # NOTE: Output sequence length = Q sequence length (not K/V length)
    context = ttnn.to_layout(context, layout=ttnn.ROW_MAJOR_LAYOUT)
    context = ttnn.permute(context, (0, 2, 1, 3))
    context = ttnn.reshape(context, (batch_size, q_seq_len, num_q_heads * head_dim))
    context = ttnn.to_layout(context, layout=ttnn.TILE_LAYOUT)

    # Output projection: 960 -> 720
    output = ttnn.linear(
        context,
        o_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(context)

    return output


def smolvla_expert_mlp_tt(
    hidden_states,
    gate_weight,
    up_weight,
    down_weight,
):
    """
    TT-optimized SwiGLU MLP for SmolVLA expert layers.

    Architecture: gate_proj + up_proj -> SiLU(gate) * up -> down_proj
    Hidden: 720, Intermediate: 2048
    """
    # Gate projection with SiLU
    gate = ttnn.linear(
        hidden_states,
        gate_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        activation="silu",
    )

    # Up projection
    up = ttnn.linear(
        hidden_states,
        up_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    # SwiGLU: SiLU(gate) * up
    intermediate = ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    # Down projection
    output = ttnn.linear(
        intermediate,
        down_weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    ttnn.deallocate(intermediate)

    return output


def smolvla_expert_layer_tt(
    hidden_states,
    layer_params,
    layer_idx: int,
    cross_attn_k: ttnn.Tensor = None,
    cross_attn_v: ttnn.Tensor = None,
):
    """
    Single TT-optimized expert transformer layer.

    Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual

    Even layers (0, 2, 4...): Self-attention
    Odd layers (1, 3, 5...): Cross-attention style
    """
    is_cross_attn = layer_idx % 2 == 1
    residual = hidden_states

    # Pre-attention LayerNorm
    normed = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["input_layernorm_weight"],
        bias=layer_params["input_layernorm_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Attention
    attn_output = smolvla_expert_attention_tt(
        normed,
        layer_params["q_weight"],
        layer_params["k_weight"],
        layer_params["v_weight"],
        layer_params["o_weight"],
        num_q_heads=12,
        num_kv_heads=4,
        is_cross_attn=is_cross_attn,
        cross_attn_k=cross_attn_k,
        cross_attn_v=cross_attn_v,
    )
    ttnn.deallocate(normed)

    # Residual connection
    hidden_states = ttnn.add(
        residual,
        attn_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(attn_output)
    residual = hidden_states

    # Pre-MLP LayerNorm
    normed = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["post_attn_layernorm_weight"],
        bias=layer_params["post_attn_layernorm_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # MLP
    mlp_output = smolvla_expert_mlp_tt(
        normed,
        layer_params["gate_weight"],
        layer_params["up_weight"],
        layer_params["down_weight"],
    )
    ttnn.deallocate(normed)

    # Residual connection
    hidden_states = ttnn.add(
        residual,
        mlp_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(mlp_output)

    return hidden_states


def smolvla_expert_layers_tt(
    hidden_states,
    layer_params_list,
    num_layers: int = 16,
    vlm_kv_cache: Optional[Dict[int, Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
):
    """
    Full TT-optimized expert layers forward pass (16 transformer layers).

    Args:
        hidden_states: [batch, seq_len, 720] expert input (action_emb + time_emb)
        layer_params_list: List of layer parameters
        num_layers: Number of expert layers
        vlm_kv_cache: Optional dict mapping layer_idx -> (k, v) from VLM
                     For cross-attention layers, K/V come from VLM (320-dim)
                     For self-attention layers, K/V come from expert hidden (720-dim)
    """
    for layer_idx in range(min(num_layers, len(layer_params_list))):
        is_cross_attn = layer_idx % 2 == 1  # Odd layers use cross-attention

        # Get cross-attention K/V from VLM cache if available
        cross_attn_k = None
        cross_attn_v = None
        if is_cross_attn and vlm_kv_cache is not None and layer_idx in vlm_kv_cache:
            # Use VLM K and V separately (both 320-dim)
            k_cache, v_cache = vlm_kv_cache[layer_idx]
            cross_attn_k = k_cache
            cross_attn_v = v_cache
        elif is_cross_attn:
            # Fallback: use first 320 dims of expert hidden (incorrect but maintains backward compat)
            logger.warning(f"Layer {layer_idx}: No VLM K/V cache, using fallback cross-attn context")
            fallback = ttnn.slice(
                hidden_states,
                (0, 0, 0),
                (hidden_states.shape[0], hidden_states.shape[1], 320),
            )
            cross_attn_k = fallback
            cross_attn_v = fallback

        hidden_states = smolvla_expert_layer_tt(
            hidden_states,
            layer_params_list[layer_idx],
            layer_idx=layer_idx,
            cross_attn_k=cross_attn_k,
            cross_attn_v=cross_attn_v,
        )

    return hidden_states


# ============================================================================
# TT Action Heads Functions
# ============================================================================
def smolvla_time_embedding_tt(
    timestep: ttnn.Tensor,
    time_mlp_in_weight,
    time_mlp_in_bias,
    time_mlp_out_weight,
    time_mlp_out_bias,
    embed_dim: int = 1440,
    min_period: float = 0.004,
    max_period: float = 4.0,
):
    """
    Generate timestep embedding for flow matching on TT.

    Sinusoidal embedding -> MLP: 1440 -> 720 -> 720
    """
    # Note: Sinusoidal embedding is computed on CPU and transferred
    # as it involves complex math operations not easily done on TT

    # MLP: 1440 -> 720 with SiLU
    emb = ttnn.linear(
        timestep,
        time_mlp_in_weight,
        bias=time_mlp_in_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        activation="silu",
    )

    # 720 -> 720
    emb = ttnn.linear(
        emb,
        time_mlp_out_weight,
        bias=time_mlp_out_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )

    return emb


def smolvla_action_heads_tt(
    expert_output,
    action_out_weight,
    action_out_bias,
):
    """
    TT-optimized action output projection.

    Projects from expert hidden (720) to action space (32).
    """
    actions = ttnn.linear(
        expert_output,
        action_out_weight,
        bias=action_out_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )

    return actions


# ============================================================================
# Vision Encoder (CPU fallback + TT path)
# ============================================================================
class SmolVLAVisionEncoder(nn.Module):
    """
    SmolVLA Vision Encoder wrapper with TT hardware acceleration.
    Uses SmolVLM2's native SigLIP-style vision encoder.

    Architecture:
    - Patch embedding: 16x16 patches from 512x512 images -> 1024 patches
    - 12 transformer layers with 768 hidden size, 12 attention heads
    - Position embeddings added after patch embedding
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        ttnn_device: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config
        self.ttnn_device = ttnn_device
        self.embed_dim = config.vision_hidden_size

        # Build CPU vision model (for weight storage and fallback)
        vlm_config = AutoConfig.from_pretrained(config.vlm_model_name, trust_remote_code=True)
        self.vision_model = self._build_vision_model(vlm_config.vision_config)

        # Load weights if provided
        if state_dict is not None:
            vision_weights = extract_vision_state_dict(state_dict)
            if vision_weights:
                self.vision_model.load_state_dict(vision_weights, strict=False)
                logger.info("Loaded vision encoder weights from state dict")

        # Initialize TT parameters if device is available
        self.tt_params_initialized = False
        self.tt_embed_params = None
        self.tt_layer_params = None
        self.tt_position_embedding = None

        if self.ttnn_device is not None:
            self._init_tt_params()

    def _build_vision_model(self, vision_config):
        """Build a simple ViT-style vision encoder."""
        from transformers.models.siglip.modeling_siglip import SiglipVisionConfig, SiglipVisionModel

        siglip_config = SiglipVisionConfig(
            hidden_size=self.config.vision_hidden_size,
            intermediate_size=self.config.vision_intermediate_size,
            num_hidden_layers=self.config.vision_num_layers,
            num_attention_heads=self.config.vision_num_attention_heads,
            image_size=self.config.vision_image_size,
            patch_size=self.config.vision_patch_size,
        )
        return SiglipVisionModel(siglip_config)

    def _init_tt_params(self):
        """Initialize TT tensor parameters from the CPU model."""
        logger.info("Initializing TT vision encoder parameters...")

        device = self.ttnn_device
        vision_model = self.vision_model.vision_model

        # Patch embedding weights
        patch_embed = vision_model.embeddings.patch_embedding

        # Preprocess conv weights for TT linear: [out, in, kH, kW] -> [kH*kW*in, out]
        weight = patch_embed.weight  # [768, 3, 16, 16]
        out_c, in_c, kh, kw = weight.shape

        # Pad input channels from 3 to 4 for TT
        weight_padded = F.pad(weight, (0, 0, 0, 0, 0, 1))  # [768, 4, 16, 16]
        weight_reshaped = weight_padded.permute(2, 3, 1, 0).reshape(-1, out_c)  # [1024, 768]

        self.tt_embed_params = {
            "proj_weight": ttnn.from_torch(
                weight_reshaped.to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            "proj_bias": ttnn.from_torch(
                patch_embed.bias.unsqueeze(0).to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
        }

        # Position embeddings
        pos_embed = vision_model.embeddings.position_embedding.weight  # [1024, 768]
        self.tt_position_embedding = ttnn.from_torch(
            pos_embed.unsqueeze(0).to(torch.bfloat16),  # [1, 1024, 768]
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )

        # Layer parameters
        self.tt_layer_params = []
        encoder = vision_model.encoder

        for i, layer in enumerate(encoder.layers):
            # Combine Q, K, V weights into single QKV weight
            q_weight = layer.self_attn.q_proj.weight
            k_weight = layer.self_attn.k_proj.weight
            v_weight = layer.self_attn.v_proj.weight
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            q_bias = layer.self_attn.q_proj.bias
            k_bias = layer.self_attn.k_proj.bias
            v_bias = layer.self_attn.v_proj.bias
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0) if q_bias is not None else None

            layer_params = {
                "norm1_weight": ttnn.from_torch(
                    layer.layer_norm1.weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "norm1_bias": ttnn.from_torch(
                    layer.layer_norm1.bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "qkv_weight": ttnn.from_torch(
                    qkv_weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "qkv_bias": ttnn.from_torch(
                    qkv_bias.unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                if qkv_bias is not None
                else None,
                "proj_weight": ttnn.from_torch(
                    layer.self_attn.out_proj.weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "proj_bias": ttnn.from_torch(
                    layer.self_attn.out_proj.bias.unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                if layer.self_attn.out_proj.bias is not None
                else None,
                "norm2_weight": ttnn.from_torch(
                    layer.layer_norm2.weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "norm2_bias": ttnn.from_torch(
                    layer.layer_norm2.bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "fc1_weight": ttnn.from_torch(
                    layer.mlp.fc1.weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "fc1_bias": ttnn.from_torch(
                    layer.mlp.fc1.bias.unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "fc2_weight": ttnn.from_torch(
                    layer.mlp.fc2.weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "fc2_bias": ttnn.from_torch(
                    layer.mlp.fc2.bias.unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
            }
            self.tt_layer_params.append(layer_params)

        # Post-encoder LayerNorm (important for normalized output!)
        self.tt_post_layernorm_weight = ttnn.from_torch(
            vision_model.post_layernorm.weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_post_layernorm_bias = ttnn.from_torch(
            vision_model.post_layernorm.bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        self.tt_params_initialized = True
        logger.info(f"TT vision encoder initialized with {len(self.tt_layer_params)} layers")

    def forward(self, pixel_values: torch.Tensor) -> Union[torch.Tensor, ttnn.Tensor]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: [batch, channels, height, width] tensor (NCHW format)

        Returns:
            Vision features [batch, num_patches, hidden_size]
        """
        if self.ttnn_device is not None and self.tt_params_initialized:
            # TT path
            return self._forward_tt(pixel_values)
        else:
            # CPU fallback
            return self._forward_cpu(pixel_values)

    def _forward_cpu(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """CPU forward pass."""
        if hasattr(pixel_values, "cpu"):
            pixel_values = pixel_values.cpu()
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.float()
        outputs = self.vision_model(pixel_values)
        return outputs.last_hidden_state

    def _forward_tt(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        TT-optimized forward pass using sharded attention for 1024 patches.

        Uses block sharding across cores to handle the large attention matrices
        (1024x1024 per head) that would otherwise exceed L1 memory.
        """
        device = self.ttnn_device
        batch_size = pixel_values.shape[0]

        # Convert NCHW -> NHWC and pad channels 3->4 for TT
        if pixel_values.shape[1] == 3:
            pixel_values_nhwc = pixel_values.permute(0, 2, 3, 1)  # [B, H, W, 3]
            pixel_values_nhwc = F.pad(pixel_values_nhwc, (0, 1))  # [B, H, W, 4]
        else:
            pixel_values_nhwc = pixel_values.permute(0, 2, 3, 1)

        # Transfer to TT device
        pixel_values_tt = ttnn.from_torch(
            pixel_values_nhwc.to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # 1. Patch embedding
        hidden_states = self._tt_patch_embed(pixel_values_tt, batch_size)
        ttnn.deallocate(pixel_values_tt)

        # 2. Add position embeddings
        hidden_states = ttnn.add(
            hidden_states,
            self.tt_position_embedding,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # 3. Transformer layers with sharded attention
        hidden_states = self._tt_encoder(hidden_states, batch_size)

        # 4. Post-encoder LayerNorm (critical for normalized output!)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.tt_post_layernorm_weight,
            bias=self.tt_post_layernorm_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return hidden_states

    def _tt_patch_embed(self, pixel_values: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
        """TT patch embedding: fold + linear."""
        patch_size = self.config.vision_patch_size  # 16
        img_h = self.config.vision_image_size  # 512
        patch_count = img_h // patch_size  # 32
        patch_count_all = patch_count * patch_count  # 1024

        # Fold the image into patches
        pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_h // patch_size, 4 * patch_size))
        pixel_values = ttnn.fold(pixel_values, stride_h=patch_size, stride_w=1)
        pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

        # Linear projection
        patch_embed = ttnn.linear(
            pixel_values,
            self.tt_embed_params["proj_weight"],
            bias=self.tt_embed_params["proj_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(pixel_values)

        # Reshape to sequence: [batch, 1024, 768]
        patch_embed = ttnn.to_layout(patch_embed, layout=ttnn.ROW_MAJOR_LAYOUT)
        patch_embed = ttnn.reshape(patch_embed, (batch_size, patch_count_all, self.embed_dim))
        patch_embed = ttnn.to_layout(patch_embed, layout=ttnn.TILE_LAYOUT)

        return patch_embed

    def _tt_encoder(self, hidden_states: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
        """TT encoder with sharded attention for memory efficiency."""
        num_heads = self.config.vision_num_attention_heads  # 12
        head_dim = self.embed_dim // num_heads  # 64
        seq_len = 1024  # Fixed for 512x512 image

        # Setup sharding config for attention
        # Using height sharding to distribute attention across cores
        core_grid = ttnn.CoreGrid(y=8, x=8)

        for layer_idx, layer_params in enumerate(self.tt_layer_params):
            residual = hidden_states

            # Pre-attention LayerNorm
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=layer_params["norm1_weight"],
                bias=layer_params["norm1_bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Attention
            hidden_states = self._tt_attention(hidden_states, layer_params, num_heads, head_dim, batch_size, seq_len)

            # Residual connection
            hidden_states = ttnn.add(
                hidden_states,
                residual,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )

            residual = hidden_states

            # Post-attention LayerNorm
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=layer_params["norm2_weight"],
                bias=layer_params["norm2_bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # MLP
            hidden_states = self._tt_mlp(hidden_states, layer_params)

            # Residual connection
            hidden_states = ttnn.add(
                hidden_states,
                residual,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )

        return hidden_states

    def _tt_attention(
        self,
        hidden_states: ttnn.Tensor,
        layer_params: dict,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """TT attention with sharding for large sequences."""
        # QKV projection
        qkv = ttnn.linear(
            hidden_states,
            layer_params["qkv_weight"],
            bias=layer_params["qkv_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        # Split QKV and reshape for multi-head attention
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=num_heads,
        )
        ttnn.deallocate(qkv)

        # Scaled dot-product attention
        # Note: split_query_key_value_and_split_heads already transposes key for matmul
        scale = 1.0 / math.sqrt(head_dim)

        attn_weights = ttnn.matmul(
            query,
            key,  # Key is already transposed by split_query_key_value_and_split_heads
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        attn_weights = ttnn.mul(attn_weights, scale)
        attn_probs = ttnn.softmax(attn_weights, dim=-1)
        ttnn.deallocate(attn_weights)

        attn_output = ttnn.matmul(
            attn_probs,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(attn_probs)
        ttnn.deallocate(value)

        # Merge heads back
        attn_output = ttnn.transformer.concatenate_heads(
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Output projection
        output = ttnn.linear(
            attn_output,
            layer_params["proj_weight"],
            bias=layer_params["proj_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(attn_output)

        return output

    def _tt_mlp(self, hidden_states: ttnn.Tensor, layer_params: dict) -> ttnn.Tensor:
        """TT MLP forward."""
        # FC1
        hidden = ttnn.linear(
            hidden_states,
            layer_params["fc1_weight"],
            bias=layer_params["fc1_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            activation="gelu",  # SigLIP uses GELU
        )

        # FC2
        output = ttnn.linear(
            hidden,
            layer_params["fc2_weight"],
            bias=layer_params["fc2_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(hidden)

        return output


# ============================================================================
# Connector/Projector
# ============================================================================
class SmolVLAConnector(nn.Module):
    """
    SmolVLA Connector - projects vision features to text model dimension.

    SmolVLM2 uses a connector that:
    1. Pools spatial dimensions by scale_factor (4)
    2. Groups patches together (16 patches -> 1 token)
    3. Projects flattened groups to text dim

    Connector input: 12288 = 768 (vision_dim) * 16 (patches per group)
    Connector output: 960 (text_dim)
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        scale_factor: int = 4,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        ttnn_device: Optional[Any] = None,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.scale_factor = scale_factor
        self.ttnn_device = ttnn_device

        # Number of patches to group together
        self.patches_per_group = scale_factor * scale_factor  # 16

        # Input dimension = vision_dim * patches_per_group
        self.proj_input_dim = vision_dim * self.patches_per_group  # 768 * 16 = 12288

        # SmolVLM2 uses: model.connector.modality_projection.proj
        self.proj = nn.Linear(self.proj_input_dim, text_dim, bias=False)

        if state_dict is not None:
            connector_weights = extract_connector_state_dict(state_dict)
            if "modality_projection.proj.weight" in connector_weights:
                self.proj.weight.data = connector_weights["modality_projection.proj.weight"].float()
            if "modality_projection.proj.bias" in connector_weights:
                self.proj.bias.data = connector_weights["modality_projection.proj.bias"].float()
            logger.info("Loaded connector weights from state dict")

        # TT path
        self.tt_proj_weight = None
        self.tt_proj_bias = None
        if self.ttnn_device is not None:
            self._init_tt_weights()

    def _init_tt_weights(self):
        """Initialize TT tensor weights."""
        self.tt_proj_weight = ttnn.from_torch(
            self.proj.weight.T.contiguous().to(torch.bfloat16),
            device=self.ttnn_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        if self.proj.bias is not None:
            self.tt_proj_bias = ttnn.from_torch(
                self.proj.bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                device=self.ttnn_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

    def _pool_and_group(self, vision_features: Union[torch.Tensor, ttnn.Tensor]) -> Union[torch.Tensor, ttnn.Tensor]:
        """
        Pool vision features spatially and group into connector input format.
        All operations on TT when input is TT tensor.

        Args:
            vision_features: [batch, num_patches, vision_dim] where num_patches = H*W/patch_size^2

        Returns:
            [batch, num_groups, proj_input_dim] ready for linear projection
        """
        is_tt_tensor = isinstance(vision_features, ttnn.Tensor)

        if is_tt_tensor:
            # Full TT path - all reshape/permute ops on TT
            batch_size = vision_features.shape[0]
            num_patches = vision_features.shape[1]
            dim = vision_features.shape[2]

            side_length = int(math.sqrt(num_patches))  # 32
            pooled_h = side_length // self.scale_factor  # 8
            pooled_w = side_length // self.scale_factor  # 8

            # Reshape to spatial grid [batch, H, W, dim]
            x = ttnn.reshape(vision_features, (batch_size, side_length, side_length, dim))

            # Reshape to [batch, pooled_h, scale, pooled_w, scale, dim]
            x = ttnn.reshape(x, (batch_size, pooled_h, self.scale_factor, pooled_w, self.scale_factor, dim))

            # Permute to [batch, pooled_h, pooled_w, scale, scale, dim]
            x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))

            # Reshape to flatten patch groups: [batch, pooled_h * pooled_w, patches_per_group * dim]
            result = ttnn.reshape(x, (batch_size, pooled_h * pooled_w, self.patches_per_group * dim))

            return result
        else:
            # CPU path for torch tensors
            batch_size, num_patches, dim = vision_features.shape
            side_length = int(math.sqrt(num_patches))
            pooled_h = side_length // self.scale_factor
            pooled_w = side_length // self.scale_factor

            x = vision_features.view(batch_size, side_length, side_length, dim)
            x = x.view(batch_size, pooled_h, self.scale_factor, pooled_w, self.scale_factor, dim)
            x = x.permute(0, 1, 3, 2, 4, 5)
            result = x.reshape(batch_size, pooled_h * pooled_w, self.patches_per_group * dim)

            return result

    def forward(self, vision_features: Union[torch.Tensor, ttnn.Tensor]) -> Union[torch.Tensor, ttnn.Tensor]:
        """
        Project vision features to text model dimension.
        100% on TT when input is TT tensor.

        Args:
            vision_features: [batch, num_patches, vision_dim] - can be torch or TT tensor

        Returns:
            [batch, num_output_tokens, text_dim] - TT tensor if input is TT
        """
        is_tt_input = isinstance(vision_features, ttnn.Tensor)

        # Pool and group patches (all on TT if input is TT)
        grouped = self._pool_and_group(vision_features)

        if is_tt_input and self.ttnn_device is not None and self.tt_proj_weight is not None:
            # Full TT path - no CPU transfers
            projected = ttnn.linear(
                grouped,
                self.tt_proj_weight,
                bias=self.tt_proj_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(grouped)
            return projected
        else:
            # CPU path
            if isinstance(grouped, torch.Tensor) and grouped.dtype == torch.bfloat16:
                grouped = grouped.float()
            return self.proj(grouped)


# ============================================================================
# Flow Matching Utilities
# ============================================================================
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    min_period: float = 0.004,
    max_period: float = 4.0,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings for flow matching (CPU version).

    Args:
        timesteps: [batch] tensor of timesteps in [0, 1]
        embedding_dim: Dimension of the embedding
        min_period: Minimum period for the sinusoidal encoding
        max_period: Maximum period for the sinusoidal encoding

    Returns:
        [batch, embedding_dim] tensor of timestep embeddings
    """
    half_dim = embedding_dim // 2

    # Log-spaced frequencies
    freqs = (
        torch.exp(
            -math.log(max_period / min_period)
            * torch.arange(half_dim, dtype=timesteps.dtype, device=timesteps.device)
            / (half_dim - 1)
        )
        / min_period
    )

    # [batch, 1] * [half_dim] -> [batch, half_dim]
    args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)

    # Concatenate sin and cos
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # Pad if necessary
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))

    return embedding


def get_timestep_embedding_tt(
    timesteps: ttnn.Tensor,
    freqs: ttnn.Tensor,  # Pre-computed frequency tensor on TT
    embedding_dim: int,
) -> ttnn.Tensor:
    """
    Create sinusoidal timestep embeddings for flow matching (TT version).
    100% on TT - no CPU transfers.

    Args:
        timesteps: [batch, 1] TT tensor of timesteps in [0, 1]
        freqs: [1, half_dim] pre-computed frequency tensor on TT
        embedding_dim: Dimension of the embedding

    Returns:
        [batch, embedding_dim] TT tensor of timestep embeddings
    """
    # [batch, 1] * [1, half_dim] -> [batch, half_dim]
    args = ttnn.matmul(timesteps, freqs, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Sin and cos
    sin_emb = ttnn.sin(args)
    cos_emb = ttnn.cos(args)

    # Concatenate [batch, half_dim] + [batch, half_dim] -> [batch, embedding_dim]
    embedding = ttnn.concat([sin_emb, cos_emb], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn.deallocate(sin_emb)
    ttnn.deallocate(cos_emb)
    ttnn.deallocate(args)

    return embedding


# ============================================================================
# Expert Layers (Action Transformer)
# ============================================================================
class SmolVLAExpertLayers(nn.Module):
    """
    SmolVLA Expert Layers - 16 additional transformer layers for action prediction.

    These layers process the combined representation of:
    - Action features (noisy actions during flow matching) - 720-dim
    - Time features (timestep embedding) - 720-dim

    Architecture notes:
    - Expert hidden dim: 720
    - Alternating layer types:
      * Even layers (0,2,4...): Self-attention, K/V from 720-dim hidden
      * Odd layers (1,3,5...): Cross-attention style, K/V from 320-dim
    - Q projection: 720 -> 960
    - O projection: 960 -> 720
    - MLP intermediate: 2048
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        ttnn_device: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config
        self.ttnn_device = ttnn_device
        self.num_layers = config.num_expert_layers
        self.hidden_dim = config.expert_hidden_size  # 720
        self.intermediate_size = config.expert_intermediate_size  # 2048

        # Attention dimensions
        self.q_dim = 960  # Q output matches text model hidden
        self.kv_dim = 320  # K/V dimension

        # Build expert transformer layers - alternating types
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            is_cross_attn = i % 2 == 1  # Odd layers have K/V from different source

            # K/V input dimension differs between layer types
            kv_input_dim = self.kv_dim if is_cross_attn else self.hidden_dim

            layer = nn.ModuleDict(
                {
                    "input_layernorm": nn.LayerNorm(self.hidden_dim),
                    "post_attention_layernorm": nn.LayerNorm(self.hidden_dim),
                    "is_cross_attn": nn.Identity(),  # Marker (we'll check layer index instead)
                    "self_attn": nn.ModuleDict(
                        {
                            "q_proj": nn.Linear(self.hidden_dim, self.q_dim, bias=False),
                            "k_proj": nn.Linear(kv_input_dim, self.kv_dim, bias=False),
                            "v_proj": nn.Linear(kv_input_dim, self.kv_dim, bias=False),
                            "o_proj": nn.Linear(self.q_dim, self.hidden_dim, bias=False),
                        }
                    ),
                    "mlp": nn.ModuleDict(
                        {
                            "gate_proj": nn.Linear(self.hidden_dim, self.intermediate_size, bias=False),
                            "up_proj": nn.Linear(self.hidden_dim, self.intermediate_size, bias=False),
                            "down_proj": nn.Linear(self.intermediate_size, self.hidden_dim, bias=False),
                        }
                    ),
                }
            )
            self.layers.append(layer)

        # TT params
        self.tt_layer_params = None
        self.tt_params_initialized = False

        # Load weights if provided
        if state_dict is not None:
            self._load_weights(state_dict)

        # Initialize TT params if device provided
        if self.ttnn_device is not None:
            self._init_tt_params()

    def _init_tt_params(self):
        """Initialize TT tensor parameters for expert layers."""
        logger.info("Initializing TT expert layer parameters...")
        device = self.ttnn_device

        self.tt_layer_params = []
        for i, layer in enumerate(self.layers):
            layer_params = {
                "input_layernorm_weight": ttnn.from_torch(
                    layer["input_layernorm"].weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "input_layernorm_bias": ttnn.from_torch(
                    layer["input_layernorm"].bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "post_attn_layernorm_weight": ttnn.from_torch(
                    layer["post_attention_layernorm"].weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "post_attn_layernorm_bias": ttnn.from_torch(
                    layer["post_attention_layernorm"].bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                # Attention weights (transposed for TT linear)
                "q_weight": ttnn.from_torch(
                    layer["self_attn"]["q_proj"].weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "k_weight": ttnn.from_torch(
                    layer["self_attn"]["k_proj"].weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "v_weight": ttnn.from_torch(
                    layer["self_attn"]["v_proj"].weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "o_weight": ttnn.from_torch(
                    layer["self_attn"]["o_proj"].weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                # MLP weights
                "gate_weight": ttnn.from_torch(
                    layer["mlp"]["gate_proj"].weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "up_weight": ttnn.from_torch(
                    layer["mlp"]["up_proj"].weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "down_weight": ttnn.from_torch(
                    layer["mlp"]["down_proj"].weight.T.contiguous().to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                ),
            }
            self.tt_layer_params.append(layer_params)

        self.tt_params_initialized = True
        logger.info(f"Initialized TT params for {len(self.tt_layer_params)} expert layers")

    def _load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load expert layer weights from state dict."""
        expert_prefix = "model.vlm_with_expert.lm_expert."

        loaded = 0
        for i, layer in enumerate(self.layers):
            layer_prefix = f"{expert_prefix}layers.{i}."

            # Load attention weights
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = f"{layer_prefix}self_attn.{name}.weight"
                if key in state_dict:
                    layer["self_attn"][name].weight.data = state_dict[key].float()
                    loaded += 1

            # Load MLP weights
            for name in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{layer_prefix}mlp.{name}.weight"
                if key in state_dict:
                    layer["mlp"][name].weight.data = state_dict[key].float()
                    loaded += 1

            # Load layernorm weights
            for ln_name, key_name in [
                ("input_layernorm", "input_layernorm"),
                ("post_attention_layernorm", "post_attention_layernorm"),
            ]:
                for param in ["weight", "bias"]:
                    key = f"{layer_prefix}{key_name}.{param}"
                    if key in state_dict:
                        if hasattr(layer[ln_name], param):
                            getattr(layer[ln_name], param).data = state_dict[key].float()
                            loaded += 1

        logger.info(f"Loaded {loaded} expert layer weights")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attn_context: Optional[torch.Tensor] = None,
        vlm_kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through expert layers.

        Args:
            hidden_states: [batch, seq_len, expert_hidden_dim=720]
            cross_attn_context: Optional [batch, seq_len, 320] for cross-attention layers (DEPRECATED)
            vlm_kv_cache: Dict mapping layer_idx -> (k, v) from VLM layers
                         Each k, v is [batch, vlm_seq, 320]
            attention_mask: Optional attention mask

        Returns:
            [batch, seq_len, expert_hidden_dim=720] output features
        """
        # Use TT path if available
        if self.ttnn_device is not None and self.tt_params_initialized:
            return self._forward_tt(hidden_states, vlm_kv_cache=vlm_kv_cache)

        # Ensure float32 for CPU computation
        hidden_states = hidden_states.float()

        # Fallback: create context from hidden states if nothing provided
        if vlm_kv_cache is None and cross_attn_context is None:
            logger.warning("No VLM K/V cache or cross_attn_context provided - using fallback")
            cross_attn_context = hidden_states[:, :, : self.kv_dim]  # Take first 320 dims

        for layer_idx, layer in enumerate(self.layers):
            is_cross_attn = layer_idx % 2 == 1

            # Pre-attention layernorm
            residual = hidden_states
            hidden_states_normed = layer["input_layernorm"](hidden_states)

            # Q always comes from the 720-dim hidden states
            q = layer["self_attn"]["q_proj"](hidden_states_normed)  # [batch, seq, 960]

            # K/V source depends on layer type
            if is_cross_attn:
                # Cross-attention: K/V from VLM cache
                if vlm_kv_cache is not None and layer_idx in vlm_kv_cache:
                    # Use per-layer VLM K/V (correct way)
                    vlm_k, vlm_v = vlm_kv_cache[layer_idx]
                    vlm_k = vlm_k.float()
                    vlm_v = vlm_v.float()
                    # Project VLM K/V through expert projections (320 -> 320)
                    k = layer["self_attn"]["k_proj"](vlm_k)  # [batch, vlm_seq, 320]
                    v = layer["self_attn"]["v_proj"](vlm_v)  # [batch, vlm_seq, 320]
                elif cross_attn_context is not None:
                    # Fallback: use same tensor for K and V (less accurate)
                    ctx = cross_attn_context.float()
                    k = layer["self_attn"]["k_proj"](ctx)
                    v = layer["self_attn"]["v_proj"](ctx)
                else:
                    # Last resort fallback
                    ctx = hidden_states_normed[:, :, : self.kv_dim]
                    k = layer["self_attn"]["k_proj"](ctx)
                    v = layer["self_attn"]["v_proj"](ctx)
            else:
                # Self-attention: K/V from 720-dim hidden states
                k = layer["self_attn"]["k_proj"](hidden_states_normed)  # [batch, seq, 320]
                v = layer["self_attn"]["v_proj"](hidden_states_normed)  # [batch, seq, 320]

            # Reshape for multi-head attention
            batch, q_seq_len, _ = q.shape
            _, kv_seq_len, _ = k.shape  # May differ from q_seq_len for cross-attention!

            # Q has 960 dims, use 12 heads with 80 head_dim
            num_q_heads = 12
            q_head_dim = self.q_dim // num_q_heads  # 80

            # K/V have 320 dims, use 4 heads with 80 head_dim (GQA)
            num_kv_heads = 4
            kv_head_dim = self.kv_dim // num_kv_heads  # 80

            # Q: [batch, q_seq, 960] -> [batch, 12, q_seq, 80]
            q = q.view(batch, q_seq_len, num_q_heads, q_head_dim).transpose(1, 2)
            # K/V: [batch, kv_seq, 320] -> [batch, 4, kv_seq, 80]
            k = k.view(batch, kv_seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
            v = v.view(batch, kv_seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)

            # GQA: repeat K/V to match Q heads (12 Q heads, 4 KV heads -> repeat 3x)
            n_rep = num_q_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)  # [batch, 12, seq, 80]
            v = v.repeat_interleave(n_rep, dim=1)  # [batch, 12, seq, 80]

            # Attention scores
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q_head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # [batch, 12, q_seq, 80]

            # Reshape back to [batch, q_seq, 960] (output length = Q length)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, q_seq_len, self.q_dim)

            # Output projection: 960 -> 720
            attn_output = layer["self_attn"]["o_proj"](attn_output)

            # Residual connection
            hidden_states = residual + attn_output

            # Pre-MLP layernorm
            residual = hidden_states
            hidden_states_normed = layer["post_attention_layernorm"](hidden_states)

            # MLP with SwiGLU
            gate = layer["mlp"]["gate_proj"](hidden_states_normed)
            up = layer["mlp"]["up_proj"](hidden_states_normed)
            hidden_states = F.silu(gate) * up
            hidden_states = layer["mlp"]["down_proj"](hidden_states)

            # Residual connection
            hidden_states = residual + hidden_states

            # Update cross-attention context for next layer (use first 320 dims of current hidden)
            cross_attn_context = hidden_states[:, :, : self.kv_dim]

        return hidden_states

    def _forward_tt(
        self,
        hidden_states: Union[torch.Tensor, ttnn.Tensor],
        vlm_kv_cache: Optional[Dict[int, Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
    ) -> torch.Tensor:
        """TT-optimized forward pass through expert layers."""
        # Convert to TT tensor if needed
        if isinstance(hidden_states, ttnn.Tensor):
            hidden_states_tt = hidden_states
        else:
            hidden_states_tt = ttnn.from_torch(
                hidden_states.to(torch.bfloat16),
                device=self.ttnn_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

        # Run TT expert layers with VLM K/V cache
        output = smolvla_expert_layers_tt(
            hidden_states_tt,
            self.tt_layer_params,
            num_layers=self.num_layers,
            vlm_kv_cache=vlm_kv_cache,
        )

        # Convert back to torch
        output = ttnn.to_torch(output).float()

        return output


# ============================================================================
# Action Heads
# ============================================================================
class SmolVLAActionHeads(nn.Module):
    """
    SmolVLA Action prediction heads for flow matching inference.

    Architecture (based on actual weight shapes):
    - state_proj: 32 -> 960 (projects to context dim)
    - action_in_proj: 32 -> 720 (projects to expert hidden dim)
    - action_out_proj: 720 -> 32 (from expert hidden to action)
    - action_time_mlp_in: 1440 -> 720 (sinusoidal embedding to expert dim)
    - action_time_mlp_out: 720 -> 720
    """

    def __init__(
        self,
        hidden_dim: int = 960,  # Context dim (text model)
        expert_hidden_dim: int = 720,  # Expert layers dim
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        chunk_size: int = 50,
        min_period: float = 0.004,
        max_period: float = 4.0,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        ttnn_device: Optional[Any] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim  # 960 - context dim
        self.expert_hidden_dim = expert_hidden_dim  # 720 - expert dim
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.min_period = min_period
        self.max_period = max_period
        self.ttnn_device = ttnn_device

        # Time embedding dimension (sinusoidal uses 2x expert dim / 2 = expert dim)
        self.time_embed_dim = expert_hidden_dim * 2  # 1440

        # State projection: 32 -> 960 (context space)
        self.state_proj = nn.Linear(max_state_dim, hidden_dim)

        # Action projections: work in expert hidden space (720)
        self.action_in_proj = nn.Linear(max_action_dim, expert_hidden_dim)
        self.action_out_proj = nn.Linear(expert_hidden_dim, max_action_dim)

        # Time embedding MLPs: 1440 -> 720 -> 720
        self.action_time_mlp_in = nn.Linear(self.time_embed_dim, expert_hidden_dim)
        self.action_time_mlp_out = nn.Linear(expert_hidden_dim, expert_hidden_dim)

        # TT params
        self.tt_params = None
        self.tt_params_initialized = False

        # Load weights if provided
        if state_dict is not None:
            action_weights = extract_action_heads_state_dict(state_dict)
            self._load_action_weights(action_weights)

        # Initialize TT params if device provided
        if self.ttnn_device is not None:
            self._init_tt_params()

    def _init_tt_params(self):
        """Initialize TT tensor parameters for action heads."""
        logger.info("Initializing TT action head parameters...")
        device = self.ttnn_device

        self.tt_params = {
            # Action projections
            "action_in_weight": ttnn.from_torch(
                self.action_in_proj.weight.T.contiguous().to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            "action_in_bias": ttnn.from_torch(
                self.action_in_proj.bias.unsqueeze(0).to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            "action_out_weight": ttnn.from_torch(
                self.action_out_proj.weight.T.contiguous().to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            "action_out_bias": ttnn.from_torch(
                self.action_out_proj.bias.unsqueeze(0).to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            # Time embedding MLPs
            "time_mlp_in_weight": ttnn.from_torch(
                self.action_time_mlp_in.weight.T.contiguous().to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            "time_mlp_in_bias": ttnn.from_torch(
                self.action_time_mlp_in.bias.unsqueeze(0).to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            "time_mlp_out_weight": ttnn.from_torch(
                self.action_time_mlp_out.weight.T.contiguous().to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            "time_mlp_out_bias": ttnn.from_torch(
                self.action_time_mlp_out.bias.unsqueeze(0).to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            ),
        }

        # Pre-compute frequency tensor for sinusoidal embedding (100% on TT)
        half_dim = self.time_embed_dim // 2  # 720
        freqs = (
            torch.exp(
                -math.log(self.max_period / self.min_period)
                * torch.arange(half_dim, dtype=torch.float32)
                / (half_dim - 1)
            )
            / self.min_period
        )
        # Shape: [1, half_dim] for broadcasting with timestep [batch, 1]
        self.tt_params["time_freqs"] = ttnn.from_torch(
            freqs.unsqueeze(0).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        self.tt_params_initialized = True
        logger.info("Initialized TT action head parameters (with sinusoidal freqs on TT)")

    def _load_action_weights(self, action_weights: Dict[str, torch.Tensor]):
        """Load action head weights."""
        weight_map = {
            "state_proj.weight": self.state_proj.weight,
            "state_proj.bias": self.state_proj.bias,
            "action_in_proj.weight": self.action_in_proj.weight,
            "action_in_proj.bias": self.action_in_proj.bias,
            "action_out_proj.weight": self.action_out_proj.weight,
            "action_out_proj.bias": self.action_out_proj.bias,
            "action_time_mlp_in.weight": self.action_time_mlp_in.weight,
            "action_time_mlp_in.bias": self.action_time_mlp_in.bias,
            "action_time_mlp_out.weight": self.action_time_mlp_out.weight,
            "action_time_mlp_out.bias": self.action_time_mlp_out.bias,
        }

        for key, param in weight_map.items():
            if key in action_weights and param is not None:
                param.data = action_weights[key].float()
        logger.info("Loaded action head weights")

    def get_time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Generate time embedding for flow matching.

        Args:
            timestep: [batch] tensor of timesteps in [0, 1]

        Returns:
            [batch, expert_hidden_dim=720] time embedding
        """
        # Get sinusoidal embedding of dimension 1440
        emb = get_timestep_embedding(
            timestep,
            self.time_embed_dim,  # 1440
            self.min_period,
            self.max_period,
        )

        # Project through MLPs: 1440 -> 720 -> 720
        emb = self.action_time_mlp_in(emb)
        emb = F.silu(emb)
        emb = self.action_time_mlp_out(emb)

        return emb  # [batch, 720]

    def project_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        """Project robot state to hidden dimension."""
        # Pad state to max_state_dim if necessary
        if robot_state.shape[-1] < self.max_state_dim:
            robot_state = F.pad(robot_state, (0, self.max_state_dim - robot_state.shape[-1]))
        return self.state_proj(robot_state)

    def project_actions(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        """Project noisy actions to hidden dimension."""
        # Pad actions to max_action_dim if necessary
        if noisy_actions.shape[-1] < self.max_action_dim:
            noisy_actions = F.pad(noisy_actions, (0, self.max_action_dim - noisy_actions.shape[-1]))
        return self.action_in_proj(noisy_actions)

    def project_actions_tt(self, noisy_actions: ttnn.Tensor) -> ttnn.Tensor:
        """TT version of project_actions."""
        return ttnn.linear(
            noisy_actions,
            self.tt_params["action_in_weight"],
            bias=self.tt_params["action_in_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

    def predict_velocity(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity (for flow matching) from expert hidden states.

        Args:
            hidden_states: [batch, chunk_size, expert_hidden_dim=720]

        Returns:
            [batch, chunk_size, max_action_dim=32] predicted velocity
        """
        return self.action_out_proj(hidden_states)

    def predict_velocity_tt(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """TT version of predict_velocity."""
        return smolvla_action_heads_tt(
            hidden_states,
            self.tt_params["action_out_weight"],
            self.tt_params["action_out_bias"],
        )

    def get_time_embedding_tt(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """
        TT version of time embedding - 100% on TT, no CPU transfers.

        Args:
            timestep: [batch, 1] TT tensor of timesteps in [0, 1]

        Returns:
            [batch, 720] time embedding on TT
        """
        # Sinusoidal embedding on TT
        sinusoidal_emb = get_timestep_embedding_tt(
            timestep,
            self.tt_params["time_freqs"],
            self.time_embed_dim,
        )

        # MLP on TT: 1440 -> 720 (with SiLU) -> 720
        hidden = ttnn.linear(
            sinusoidal_emb,
            self.tt_params["time_mlp_in_weight"],
            bias=self.tt_params["time_mlp_in_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(sinusoidal_emb)

        hidden = ttnn.silu(hidden)

        output = ttnn.linear(
            hidden,
            self.tt_params["time_mlp_out_weight"],
            bias=self.tt_params["time_mlp_out_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(hidden)

        return output  # [batch, 720]

    def forward(
        self,
        hidden_states: torch.Tensor,
        robot_state: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare inputs for expert layers (does NOT run expert layers).

        This combines action and time embeddings in expert hidden space (720-dim).
        Context (vision/state) is handled separately as it's in 960-dim space.

        Args:
            hidden_states: [batch, seq_len, hidden_dim=960] vision/text features (unused here)
            robot_state: [batch, state_dim] robot proprioception
            noisy_actions: [batch, chunk_size, action_dim] current noisy actions
            timestep: [batch] current timestep in [0, 1]

        Returns:
            [batch, chunk_size, expert_hidden_dim=720] combined action+time embeddings
        """
        batch_size = noisy_actions.shape[0] if noisy_actions is not None else 1

        if noisy_actions is not None and timestep is not None:
            # Project noisy actions: [batch, chunk_size, 720]
            action_emb = self.project_actions(noisy_actions)

            # Get time embedding: [batch, 720]
            time_emb = self.get_time_embedding(timestep)

            # Combine action and time in expert space
            # [batch, chunk_size, 720]
            combined = action_emb + time_emb.unsqueeze(1)

            return combined
        else:
            # Return zeros in expert hidden space
            return torch.zeros(batch_size, self.chunk_size, self.expert_hidden_dim, dtype=torch.float32)


# ============================================================================
# SmolVLA Text Model (TT-accelerated)
# ============================================================================
class SmolVLATextModel:
    """
    SmolVLA Text Model wrapper for TT hardware.
    Uses the TT Transformer infrastructure with SmolVLA-specific config.
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        device,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.config = config
        self.device = device
        self.state_dict = state_dict

        # Map state dict to TT format
        self.tt_state_dict = None
        if state_dict is not None:
            self.tt_state_dict = map_smolvla_to_tt_text_keys(state_dict)

        # Generator args config
        self.generator_args_config = {
            "num_devices": device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1,
            "data_parallel": 1,
            "mesh_device": device,
            "instruct": False,
            "global_batch_size": 1,
            "optimizations": "performance",
            "max_seq_len": 512,
            "page_params": {"page_block_size": 32, "page_max_num_blocks_per_dp": 512},
            "paged_attention": True,
            "num_layers": config.text_num_layers,
        }

        # Create TT model
        self._init_tt_model()

    def _init_tt_model(self):
        """Initialize the TT transformer model."""
        from models.tt_transformers.demo.simple_text_demo import prepare_generator_args

        def model_factory_fn(*args, **kwargs):
            return create_tt_model(
                *args,
                **kwargs,
                ModelArgsClass=get_SmolVLAModelArgs(self.tt_state_dict, self.config),
            )

        (
            self.model_args,
            self.model,
            self.page_table,
            self.tt_kv_cache,
            self.tokenizer,
            self.processor,
        ) = prepare_generator_args(**self.generator_args_config, model_factory_fn=model_factory_fn)

        self.generator = Generator(self.model, self.model_args, self.device, self.tokenizer)

    def get_input_embeddings(self):
        """Get the embedding layer."""
        return self.model[0].embd

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through text model.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size] input embeddings
            attention_mask: Optional attention mask

        Returns:
            Hidden states from the text model
        """
        seq_len = inputs_embeds.shape[1] if inputs_embeds.dim() == 3 else inputs_embeds.shape[2]

        # Pad to valid prefill length
        padding = get_padded_prefill_len(seq_len) - seq_len
        if padding != 0:
            if isinstance(inputs_embeds, torch.Tensor):
                inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, padding), value=0)
            else:
                inputs_embeds = ttnn.pad(inputs_embeds, [(0, 0), (0, 0), (0, padding), (0, 0)], 0)

        # Get rotation matrices
        tt_rot_mats_prefill = [
            self.model[0].rope_setup.cos_matrix[:, :, : seq_len + padding, :],
            self.model[0].rope_setup.sin_matrix[:, :, : seq_len + padding, :],
        ]

        # Get page table
        page_table_user = self._get_prefill_user_page_table(self.page_table, self.tt_kv_cache[0], seq_len)
        tt_page_table = ttnn.from_torch(
            page_table_user,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

        # Forward through model
        hidden_states = self.model[0].forward(
            inputs_embeds,
            None,
            rot_mats_global=tt_rot_mats_prefill,
            mode="prefill",
            page_table=tt_page_table,
            kv_cache=self.tt_kv_cache[0],
            get_last_token=-1,  # Return all tokens
        )

        return hidden_states

    def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len):
        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(prefill_len, block_size)
        return page_table[:, :num_blocks]


# ============================================================================
# Main SmolVLA Model
# ============================================================================
class SmolVLAForActionPrediction(nn.Module):
    """
    SmolVLA model for robot action prediction.

    This combines:
    - Vision encoder (SmolVLM2 SigLIP-style)
    - Connector (linear projection)
    - Text model (LLaMA-like, TT-accelerated)
    - Expert layers (additional transformer layers)
    - Action heads (flow matching output)
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        ttnn_device: Optional[Any] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.config = config
        self.ttnn_device = ttnn_device
        self.state_dict = state_dict

        logger.info("Initializing SmolVLA model components...")

        # Vision encoder
        CHECKPOINTS.checkpoint("start_VISIONINIT")
        self.vision_encoder = SmolVLAVisionEncoder(
            config=config,
            state_dict=state_dict,
            ttnn_device=ttnn_device,
        )
        CHECKPOINTS.checkpoint("end_VISIONINIT")

        # Connector
        CHECKPOINTS.checkpoint("start_CONNECTORINIT")
        self.connector = SmolVLAConnector(
            vision_dim=config.vision_hidden_size,
            text_dim=config.text_hidden_size,
            scale_factor=4,  # SmolVLM2 uses scale_factor=4 for spatial pooling
            state_dict=state_dict,
            ttnn_device=ttnn_device,
        )
        CHECKPOINTS.checkpoint("end_CONNECTORINIT")

        # Text model (TT-accelerated)
        if ttnn_device is not None:
            CHECKPOINTS.checkpoint("start_TEXTMODELINIT")
            self.text_model = SmolVLATextModel(
                config=config,
                device=ttnn_device,
                state_dict=state_dict,
            )
            CHECKPOINTS.checkpoint("end_TEXTMODELINIT")
        else:
            self.text_model = None
            logger.info("TT device not provided, text model not initialized")

        # Expert layers (action transformer)
        CHECKPOINTS.checkpoint("start_EXPERTINIT")
        self.expert_layers = SmolVLAExpertLayers(
            config=config,
            state_dict=state_dict,
            ttnn_device=ttnn_device,
        )
        CHECKPOINTS.checkpoint("end_EXPERTINIT")

        # Action heads
        CHECKPOINTS.checkpoint("start_ACTIONHEADSINIT")
        self.action_heads = SmolVLAActionHeads(
            hidden_dim=config.text_hidden_size,  # 960
            expert_hidden_dim=config.expert_hidden_size,  # 720
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            chunk_size=config.chunk_size,
            state_dict=state_dict,
            ttnn_device=ttnn_device,
        )
        CHECKPOINTS.checkpoint("end_ACTIONHEADSINIT")

        # VLM K/V projections for expert cross-attention
        # These project 960-dim VLM hidden to 320-dim for expert cross-attn
        CHECKPOINTS.checkpoint("start_VLMKVINIT")
        self._init_vlm_kv_projections(state_dict, ttnn_device)
        CHECKPOINTS.checkpoint("end_VLMKVINIT")

        # Flow matching config
        self.num_inference_steps = 10  # Default, can be overridden

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            config.vlm_model_name,
            trust_remote_code=True,
        )

        # Disable image splitting/tiling to reduce vision token count
        # With splitting: 17 tiles × 64 = 1088 vision tokens (instruction diluted to ~4%)
        # Without splitting: 1 tile × 64 = 64 vision tokens (instruction ~11%)
        # This matches LeRobot reference behavior for better instruction following
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_image_splitting = False
            logger.info("Disabled image splitting for better instruction influence")
        if hasattr(self.processor, "video_processor"):
            self.processor.video_processor.do_image_splitting = False

        # Load text embedding layer for instruction token embedding
        self._init_text_embeddings(state_dict)

        logger.info("SmolVLA model initialized successfully")

    def _init_text_embeddings(self, state_dict: Optional[Dict[str, torch.Tensor]]):
        """
        Initialize text embedding layer for instruction token embedding.

        The VLM uses a 49280-dim vocabulary with 960-dim embeddings.
        This allows us to embed instruction tokens before merging with vision.
        """
        if state_dict is None:
            logger.warning("No state dict, skipping text embeddings initialization")
            self.text_embeddings = None
            return

        embed_key = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"
        if embed_key in state_dict:
            embed_weight = state_dict[embed_key]  # [49280, 960]
            logger.info(f"Loading text embeddings: {embed_weight.shape}")

            self.text_embeddings = nn.Embedding(
                num_embeddings=embed_weight.shape[0],
                embedding_dim=embed_weight.shape[1],
            )
            self.text_embeddings.weight.data = embed_weight
        else:
            logger.warning(f"Text embedding weights not found: {embed_key}")
            self.text_embeddings = None

    def _build_prefix_embeddings(
        self,
        projected_features: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Build merged prefix embeddings from vision and instruction tokens.

        The VLM K/V cache must see BOTH vision and instruction tokens.
        Otherwise the model cannot condition on language instructions.

        Strategy:
        1. Get input_ids from processor (contains <image> placeholders)
        2. Embed ALL tokens using text_embeddings (including placeholders)
        3. In-place replace all <image> placeholder positions with projected vision features
        4. Return prefix_embeds with vision embeddings merged into the text sequence

        Note: The processor expands <image> to many placeholders (e.g., 1088 for 1024 patches).
        Our connector pools these to fewer tokens (e.g., 64 after 4x4 pooling), so we
        replace the contiguous block of placeholders with the pooled vision features.

        Args:
            projected_features: [batch, vision_tokens, 960] from connector (pooled)
            inputs: Processor outputs containing input_ids

        Returns:
            prefix_embeds: [batch, total_tokens, 960] with vision embeddings merged in
        """
        if self.text_embeddings is None:
            logger.warning("Text embeddings not available, using vision-only prefix")
            return projected_features

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            logger.warning("No input_ids in inputs, using vision-only prefix")
            return projected_features

        batch_size = projected_features.shape[0]
        vision_tokens = projected_features.shape[1]  # e.g., 64 after connector pooling

        # Get image token ID
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

        # Count image placeholders in input_ids
        image_mask = input_ids == image_token_id
        n_image_placeholders = image_mask.sum(dim=1)  # [batch]

        # Embed all tokens first (including placeholders)
        # Use float32 for stability
        tok_embeds = self.text_embeddings(input_ids).float()  # [batch, seq_len, 960]

        # Critical assertion: embedding dimensions must match
        assert (
            tok_embeds.shape[-1] == projected_features.shape[-1] == 960
        ), f"Embedding dim mismatch: tok_embeds={tok_embeds.shape[-1]}, vision={projected_features.shape[-1]}, expected=960"

        # SmolVLM processor expands <image> to many placeholders (e.g., 1088)
        # But our connector outputs only vision_tokens (e.g., 64 after pooling)
        #
        # We replace the contiguous block of <image> placeholders with pooled vision features
        # This effectively: [text_before] + [1088 placeholders] + [text_after]
        #               -> [text_before] + [64 vision tokens] + [text_after]

        if n_image_placeholders[0].item() > 0:
            # Find positions of image placeholders
            image_positions = image_mask[0].nonzero(as_tuple=True)[0]

            if len(image_positions) > vision_tokens:
                # More placeholders than vision tokens - need to truncate
                # Keep first vision_tokens positions for vision features
                # Remove extra placeholder positions

                # Build new sequence: [before_images] + [vision_features] + [after_images]
                first_image_pos = image_positions[0].item()
                last_image_pos = image_positions[-1].item()

                # Get text embeddings before and after image region
                before_embeds = tok_embeds[:, :first_image_pos, :]  # Before <image>
                after_embeds = tok_embeds[:, last_image_pos + 1 :, :]  # After <image>

                # Merge: [before] + [vision] + [after]
                prefix_embeds = torch.cat(
                    [
                        before_embeds,
                        projected_features,  # [batch, vision_tokens, 960]
                        after_embeds,
                    ],
                    dim=1,
                )

                logger.info(
                    f"Merged prefix: before={before_embeds.shape[1]}, "
                    f"vision={vision_tokens}, after={after_embeds.shape[1]}, "
                    f"total={prefix_embeds.shape[1]}"
                )

            elif len(image_positions) == vision_tokens:
                # Perfect match - in-place replace all <image> positions with vision features
                # Critical assertion: number of placeholders must match vision tokens
                assert (
                    image_mask.sum().item() == projected_features.shape[1]
                ), f"Mismatch: {image_mask.sum().item()} <image> tokens != {projected_features.shape[1]} vision tokens"
                tok_embeds[image_mask] = projected_features.reshape(-1, projected_features.shape[-1])
                prefix_embeds = tok_embeds

            else:
                # Fewer placeholders than vision tokens - pad vision features
                # This shouldn't happen normally
                logger.warning(
                    f"Fewer image placeholders ({len(image_positions)}) than "
                    f"vision tokens ({vision_tokens}), truncating vision features"
                )
                truncated_vision = projected_features[:, : len(image_positions), :]
                tok_embeds[image_mask] = truncated_vision.reshape(-1, truncated_vision.shape[-1])
                prefix_embeds = tok_embeds
        else:
            # No image placeholders - just concatenate vision + text
            logger.warning("No image placeholders found, concatenating vision + text")
            prefix_embeds = torch.cat([projected_features, tok_embeds], dim=1)

        # Final shape validation
        assert (
            prefix_embeds.shape[-1] == 960
        ), f"Final prefix_embeds has wrong embedding dim: {prefix_embeds.shape[-1]}, expected 960"

        return prefix_embeds

    def _init_vlm_kv_projections(self, state_dict: Optional[Dict[str, torch.Tensor]], ttnn_device):
        """
        Initialize VLM layer weights for computing K/V cache on TT.

        This initializes ALL VLM layer weights on TT for full TT-accelerated
        K/V cache computation - no CPU fallback needed.
        """
        self.vlm_kv_weights = {}
        self.vlm_layer_weights_tt = {}  # TT weights per layer

        if state_dict is None:
            logger.warning("No state dict, skipping VLM K/V projection initialization")
            return

        text_prefix = "model.vlm_with_expert.vlm.model.text_model.layers"
        num_layers = self.config.num_expert_layers

        # Initialize TT weights for ALL VLM layers
        if ttnn_device is not None:
            logger.info(f"Initializing TT weights for {num_layers} VLM layers...")
            for layer_idx in range(num_layers):
                layer_key = f"{text_prefix}.{layer_idx}"

                # Check if layer weights exist
                ln_key = f"{layer_key}.input_layernorm.weight"
                if ln_key not in state_dict:
                    logger.warning(f"VLM layer {layer_idx} not found, stopping at {layer_idx} layers")
                    break

                layer_weights = {}

                # Input layernorm - weight shape: [1, SHARD_HEIGHT, dim] where SHARD_HEIGHT=32
                ln_weight = state_dict[f"{layer_key}.input_layernorm.weight"].to(torch.bfloat16)
                ln_weight = ln_weight.unsqueeze(0).view(1, 1, 960).expand(1, 32, 960).contiguous()
                layer_weights["ln_weight"] = ttnn.from_torch(
                    ln_weight,
                    device=ttnn_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )

                # Q/K/V/O projections (transpose for ttnn.linear)
                layer_weights["q_proj"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.self_attn.q_proj.weight"].T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                layer_weights["k_proj"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.self_attn.k_proj.weight"].T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                layer_weights["v_proj"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.self_attn.v_proj.weight"].T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                layer_weights["o_proj"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.self_attn.o_proj.weight"].T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )

                # Post-attention layernorm - weight shape: [1, SHARD_HEIGHT, dim]
                post_ln_weight = state_dict[f"{layer_key}.post_attention_layernorm.weight"].to(torch.bfloat16)
                post_ln_weight = post_ln_weight.unsqueeze(0).view(1, 1, 960).expand(1, 32, 960).contiguous()
                layer_weights["post_ln_weight"] = ttnn.from_torch(
                    post_ln_weight,
                    device=ttnn_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )

                # MLP weights
                layer_weights["gate_proj"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.mlp.gate_proj.weight"].T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                layer_weights["up_proj"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.mlp.up_proj.weight"].T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                layer_weights["down_proj"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.mlp.down_proj.weight"].T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )

                self.vlm_layer_weights_tt[layer_idx] = layer_weights

            logger.info(f"VLM TT weights initialized for {len(self.vlm_layer_weights_tt)} layers")

        # Also keep CPU fallback weights (layer 0 only for backward compat)
        prefix = "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn"
        k_weight_key = f"{prefix}.k_proj.weight"
        v_weight_key = f"{prefix}.v_proj.weight"

        if k_weight_key in state_dict and v_weight_key in state_dict:
            k_weight = state_dict[k_weight_key]
            v_weight = state_dict[v_weight_key]

            self.vlm_k_proj = nn.Linear(960, 320, bias=False)
            self.vlm_k_proj.weight.data = k_weight
            self.vlm_v_proj = nn.Linear(960, 320, bias=False)
            self.vlm_v_proj.weight.data = v_weight

            if ttnn_device is not None:
                self.tt_vlm_k_proj = ttnn.from_torch(
                    k_weight.T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                self.tt_vlm_v_proj = ttnn.from_torch(
                    v_weight.T.contiguous().to(torch.bfloat16),
                    device=ttnn_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
            else:
                self.tt_vlm_k_proj = None
                self.tt_vlm_v_proj = None
        else:
            self.vlm_k_proj = None
            self.vlm_v_proj = None
            self.tt_vlm_k_proj = None
            self.tt_vlm_v_proj = None

    def compute_vlm_kv_context(
        self, projected_features: Union[torch.Tensor, ttnn.Tensor]
    ) -> Tuple[Union[torch.Tensor, ttnn.Tensor], Union[torch.Tensor, ttnn.Tensor]]:
        """
        Compute VLM K/V context from projected vision features for expert cross-attention.

        SIMPLIFIED VERSION: Just projects connector output without running VLM layers.
        For full implementation, use compute_vlm_kv_cache() instead.

        Args:
            projected_features: [batch, seq_len, 960] from connector

        Returns:
            k_context: [batch, seq_len, 320]
            v_context: [batch, seq_len, 320]
        """
        if isinstance(projected_features, ttnn.Tensor) and self.tt_vlm_k_proj is not None:
            # TT path
            k_context = ttnn.linear(
                projected_features,
                self.tt_vlm_k_proj,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            v_context = ttnn.linear(
                projected_features,
                self.tt_vlm_v_proj,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            return k_context, v_context
        elif self.vlm_k_proj is not None:
            # CPU path
            if isinstance(projected_features, ttnn.Tensor):
                projected_features = ttnn.to_torch(projected_features).float()
            k_context = self.vlm_k_proj(projected_features)
            v_context = self.vlm_v_proj(projected_features)
            return k_context, v_context
        else:
            # No VLM projections available - return None
            logger.warning("VLM K/V projections not available")
            return None, None

    def compute_vlm_kv_cache(
        self,
        prefix_embeds: Union[torch.Tensor, ttnn.Tensor],
        num_layers: int = 16,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run VLM text model forward and cache K/V for each layer.

        NOTE: This runs on CPU for now. The VLM forward is complex (16 layers with
        attention + MLP each) and the rms_norm weight requirements in TTNN are
        non-trivial to satisfy. Since the vision encoder, expert layers, and action
        heads are already on TT, this CPU path is acceptable for the overall pipeline.

        This is the CORRECT way to compute cross-attention context:
        1. Run VLM layers on prefix embeddings (vision + optional state/instruction)
        2. Cache K/V from each layer's attention
        3. Expert cross-attn uses cached K/V from corresponding layer

        Args:
            prefix_embeds: [batch, seq_len, 960] prefix embeddings (vision, state, instruction)
            num_layers: Number of VLM layers to run

        Returns:
            Dict mapping layer_idx -> (k_cache, v_cache) where k/v are [batch, seq, 320]
        """
        # Use TT path if TT weights are available
        if self.ttnn_device is not None and self.vlm_layer_weights_tt:
            return self._compute_vlm_kv_cache_tt(prefix_embeds, num_layers)

        return self._compute_vlm_kv_cache_cpu(prefix_embeds, num_layers)

    def _compute_vlm_kv_cache_tt(
        self,
        prefix_embeds: Union[torch.Tensor, ttnn.Tensor],
        num_layers: int = 16,
    ) -> Dict[int, Tuple[ttnn.Tensor, ttnn.Tensor]]:
        """TT-accelerated VLM K/V cache computation - 100% on TT."""
        # Convert to TT tensor if needed
        if isinstance(prefix_embeds, torch.Tensor):
            hidden_states = ttnn.from_torch(
                prefix_embeds.to(torch.bfloat16),
                device=self.ttnn_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            hidden_states = prefix_embeds

        batch_size = 1
        seq_len = hidden_states.shape[1] if hasattr(hidden_states, "shape") else prefix_embeds.shape[1]
        hidden_dim = 960
        num_heads = 15
        num_kv_heads = 5
        head_dim = 64

        kv_cache = {}

        for layer_idx in range(min(num_layers, len(self.vlm_layer_weights_tt))):
            weights = self.vlm_layer_weights_tt[layer_idx]

            # Input layernorm (RMSNorm)
            normed = ttnn.rms_norm(
                hidden_states,
                weight=weights["ln_weight"],
                epsilon=1e-5,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # K/V projections for cross-attention cache
            k = ttnn.linear(normed, weights["k_proj"], memory_config=ttnn.L1_MEMORY_CONFIG)
            v = ttnn.linear(normed, weights["v_proj"], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Only cache for cross-attention layers (odd layers in expert)
            if layer_idx % 2 == 1:
                kv_cache[layer_idx] = (k, v)

            # Q projection for attention
            q = ttnn.linear(normed, weights["q_proj"], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Attention computation
            # Reshape for multi-head attention: [B, S, D] -> [B, H, S, D/H]
            q = ttnn.reshape(q, (batch_size, seq_len, num_heads, head_dim))
            q = ttnn.permute(q, (0, 2, 1, 3))  # [B, H, S, head_dim]

            k_attn = ttnn.reshape(k, (batch_size, seq_len, num_kv_heads, head_dim))
            k_attn = ttnn.permute(k_attn, (0, 2, 1, 3))

            v_attn = ttnn.reshape(v, (batch_size, seq_len, num_kv_heads, head_dim))
            v_attn = ttnn.permute(v_attn, (0, 2, 1, 3))

            # GQA: repeat K/V for each query head group
            n_rep = num_heads // num_kv_heads  # 3
            k_attn = ttnn.repeat(k_attn, (1, n_rep, 1, 1))
            v_attn = ttnn.repeat(v_attn, (1, n_rep, 1, 1))

            # Scaled dot-product attention
            scale = 1.0 / math.sqrt(head_dim)
            k_attn_t = ttnn.permute(k_attn, (0, 1, 3, 2))  # Transpose last two dims
            attn_weights = ttnn.matmul(q, k_attn_t)
            attn_weights = ttnn.mul(attn_weights, scale)
            attn_weights = ttnn.softmax(attn_weights, dim=-1)
            attn_output = ttnn.matmul(attn_weights, v_attn)

            # Reshape back: [B, H, S, head_dim] -> [B, S, D]
            attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
            attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, hidden_dim))

            # Output projection
            attn_output = ttnn.linear(attn_output, weights["o_proj"], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Residual
            hidden_states = ttnn.add(hidden_states, attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)

            # MLP: Post-attention layernorm
            normed = ttnn.rms_norm(
                hidden_states,
                weight=weights["post_ln_weight"],
                epsilon=1e-5,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Gate and up projections
            gate = ttnn.linear(normed, weights["gate_proj"], memory_config=ttnn.L1_MEMORY_CONFIG)
            up = ttnn.linear(normed, weights["up_proj"], memory_config=ttnn.L1_MEMORY_CONFIG)

            # SiLU activation and elementwise multiply
            gate = ttnn.silu(gate)
            mlp_output = ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Down projection
            mlp_output = ttnn.linear(mlp_output, weights["down_proj"], memory_config=ttnn.L1_MEMORY_CONFIG)

            # Residual
            hidden_states = ttnn.add(hidden_states, mlp_output, memory_config=ttnn.L1_MEMORY_CONFIG)

        logger.info(f"Computed VLM K/V cache on TT for {len(kv_cache)} cross-attention layers")
        return kv_cache

    def _compute_vlm_kv_cache_cpu(
        self,
        prefix_embeds: torch.Tensor,
        num_layers: int = 16,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """CPU fallback for VLM K/V cache computation."""
        if self.state_dict is None:
            logger.warning("No state dict available for VLM forward")
            return {}

        # Convert to torch if needed
        if isinstance(prefix_embeds, ttnn.Tensor):
            prefix_embeds = ttnn.to_torch(prefix_embeds).float()

        batch_size, seq_len, hidden_dim = prefix_embeds.shape
        hidden_states = prefix_embeds.float()
        kv_cache = {}

        text_prefix = "model.vlm_with_expert.vlm.model.text_model.layers"

        for layer_idx in range(num_layers):
            layer_key = f"{text_prefix}.{layer_idx}"

            ln_weight_key = f"{layer_key}.input_layernorm.weight"
            if ln_weight_key not in self.state_dict:
                logger.warning(f"VLM layer {layer_idx} weights not found")
                break

            ln_weight = self.state_dict[ln_weight_key].float()
            normed = F.rms_norm(hidden_states, (hidden_dim,), weight=ln_weight, eps=1e-5)

            k_proj_weight = self.state_dict[f"{layer_key}.self_attn.k_proj.weight"].float()
            v_proj_weight = self.state_dict[f"{layer_key}.self_attn.v_proj.weight"].float()

            k = F.linear(normed, k_proj_weight)
            v = F.linear(normed, v_proj_weight)

            if layer_idx % 2 == 1:
                kv_cache[layer_idx] = (k, v)

            q_proj_weight = self.state_dict[f"{layer_key}.self_attn.q_proj.weight"].float()
            o_proj_weight = self.state_dict[f"{layer_key}.self_attn.o_proj.weight"].float()

            q = F.linear(normed, q_proj_weight)

            num_heads = 15
            num_kv_heads = 5
            head_dim = 64

            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k_attn = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v_attn = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

            n_rep = num_heads // num_kv_heads
            k_attn = k_attn.repeat_interleave(n_rep, dim=1)
            v_attn = v_attn.repeat_interleave(n_rep, dim=1)

            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = torch.matmul(q, k_attn.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_attn)

            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
            attn_output = F.linear(attn_output, o_proj_weight)

            hidden_states = hidden_states + attn_output

            post_ln_weight = self.state_dict[f"{layer_key}.post_attention_layernorm.weight"].float()
            normed = F.rms_norm(hidden_states, (hidden_dim,), weight=post_ln_weight, eps=1e-5)

            gate_weight = self.state_dict[f"{layer_key}.mlp.gate_proj.weight"].float()
            up_weight = self.state_dict[f"{layer_key}.mlp.up_proj.weight"].float()
            down_weight = self.state_dict[f"{layer_key}.mlp.down_proj.weight"].float()

            gate = F.linear(normed, gate_weight)
            up = F.linear(normed, up_weight)
            mlp_output = F.silu(gate) * up
            mlp_output = F.linear(mlp_output, down_weight)

            hidden_states = hidden_states + mlp_output

        logger.info(f"Computed VLM K/V cache on CPU for {len(kv_cache)} cross-attention layers")
        return kv_cache

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "lerobot/smolvla_base",
        ttnn_device: Optional[Any] = None,
    ):
        """
        Load SmolVLA from HuggingFace.

        Args:
            repo_id: HuggingFace repo ID (default: lerobot/smolvla_base)
            ttnn_device: TT device for acceleration

        Returns:
            SmolVLAForActionPrediction model
        """
        from huggingface_hub import hf_hub_download

        logger.info(f"Loading SmolVLA from {repo_id}")

        # Load weights first to count actual layers
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = safetensors_load_file(weights_path)
        logger.info(f"Loaded {len(state_dict)} weight tensors from {weights_path}")

        # Count actual layers in the state dict
        max_text_layer = -1
        for key in state_dict.keys():
            if "text_model.layers." in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            max_text_layer = max(max_text_layer, layer_num)
                        except:
                            pass

        actual_text_layers = max_text_layer + 1 if max_text_layer >= 0 else None
        logger.info(f"Detected {actual_text_layers} text layers in state dict")

        # Load config
        config = SmolVLAConfig.from_pretrained(repo_id)

        # Override layer count if different from state dict
        if actual_text_layers is not None and actual_text_layers != config.text_num_layers:
            logger.info(f"Overriding text_num_layers: {config.text_num_layers} -> {actual_text_layers}")
            config.text_num_layers = actual_text_layers

        return cls(
            config=config,
            ttnn_device=ttnn_device,
            state_dict=state_dict,
        )

    def predict_action(
        self,
        images: List[Image.Image],
        robot_state: Optional[torch.Tensor] = None,
        instruction: str = "",
        num_inference_steps: int = 10,
        action_dim: int = 6,
    ) -> np.ndarray:
        """
        Predict robot action from images and optional state using flow matching.

        Flow Matching Inference:
        1. Encode vision features and project through connector
        2. Initialize with noise (or zeros)
        3. Iteratively denoise using the learned velocity field:
           x_{t+dt} = x_t + v(x_t, t) * dt

        Args:
            images: List of PIL images (typically 1-3 camera views)
            robot_state: Optional robot state tensor [state_dim]
            instruction: Optional language instruction
            num_inference_steps: Number of flow matching denoising steps
            action_dim: Actual action dimension (default 6 for robot arm)

        Returns:
            Predicted actions as numpy array [n_action_steps, action_dim]
        """
        CHECKPOINTS.reset()
        CHECKPOINTS.checkpoint("start_ACTIONPREDICTION")

        batch_size = 1
        chunk_size = self.config.chunk_size
        device = "cpu"  # Use CPU for now; TT acceleration in future

        # Preprocess images
        CHECKPOINTS.checkpoint("start_PREPROCESS")

        # SmolVLM requires image tokens in text prompt
        # Format: "<image>" token for each image
        num_images = len(images) if images else 0
        image_tokens = "<image>" * num_images

        # Create prompt with image tokens
        if instruction:
            text_prompt = f"{image_tokens}{instruction}"
        else:
            text_prompt = f"{image_tokens}What action should the robot take?"

        inputs = self.processor(
            images=images if images else None,
            text=text_prompt,
            return_tensors="pt",
        )
        pixel_values = inputs.get("pixel_values", inputs.get("images"))
        if pixel_values is not None:
            pixel_values = pixel_values.to(torch.bfloat16)
        CHECKPOINTS.checkpoint("end_PREPROCESS")

        # Vision encoding
        CHECKPOINTS.checkpoint("start_VISIONFORWARD")
        if pixel_values is not None:
            # SmolVLM processor outputs [batch, num_tiles, C, H, W] for tiled images
            # num_tiles = 17 = 16 local (4x4 grid) + 1 global
            # Each tile: 1024 patches -> pooled to 64 tokens
            # Total: 17 tiles × 64 = 1088 vision tokens (matches 1088 <image> placeholders)

            if pixel_values.dim() == 5:
                # Tiled format: [B, T, C, H, W] -> flatten to [B*T, C, H, W]
                B, T, C, H, W = pixel_values.shape
                logger.info(f"Processing {T} tiles: pixel_values shape {pixel_values.shape}")

                # Flatten tiles into batch dimension
                pv_flat = pixel_values.reshape(B * T, C, H, W)  # [B*T, 3, 512, 512]

                # Process all tiles at once through vision encoder
                vf_flat = self.vision_encoder(pv_flat)  # [B*T, 1024, 768]

                # Project through connector
                pf_flat = self.connector(vf_flat)  # [B*T, 64, 960]

                # Convert to torch if needed
                if isinstance(pf_flat, ttnn.Tensor):
                    pf_flat = ttnn.to_torch(pf_flat)

                # Reshape back: [B*T, 64, 960] -> [B, T*64, 960]
                projected_features = pf_flat.reshape(B, T * 64, 960)  # [B, 1088, 960]
                logger.info(f"All tiles projected: {projected_features.shape} ({T} tiles × 64 = {T * 64} tokens)")
            else:
                # Single image: [batch, C, H, W]
                vision_features = self.vision_encoder(pixel_values)
                projected_features = self.connector(vision_features)
                if isinstance(projected_features, ttnn.Tensor):
                    projected_features = ttnn.to_torch(projected_features)
        else:
            projected_features = torch.zeros(1, 1, self.config.text_hidden_size)
        CHECKPOINTS.checkpoint("end_VISIONFORWARD")

        # ========================================
        # Compute VLM K/V Cache for Expert Cross-Attention
        # ========================================
        # This is the KEY FIX: Vision features must influence expert layers
        # via cross-attention, not be ignored!
        #
        # CORRECT FLOW (from LeRobot SmolVLA):
        # 1. Build prefix embeddings: [vision_tokens, instruction_tokens]
        # 2. Run VLM text model on prefix
        # 3. Each VLM layer computes K/V via k_proj/v_proj (960 -> 320)
        # 4. Expert cross-attn layer i uses K/V from VLM layer i
        CHECKPOINTS.checkpoint("start_VLMKVCOMPUTE")

        # Convert projected_features to torch for VLM forward
        if isinstance(projected_features, ttnn.Tensor):
            projected_features_cpu = ttnn.to_torch(projected_features).float()
        elif projected_features.dtype == torch.bfloat16:
            projected_features_cpu = projected_features.float()
        else:
            projected_features_cpu = projected_features

        # ========================================
        # Build merged prefix embeddings: vision + instruction
        # ========================================
        # The VLM context needs BOTH vision and instruction tokens for
        # the model to follow language instructions correctly.

        prefix_embeds = self._build_prefix_embeddings(
            projected_features_cpu,  # [batch, vision_tokens, 960]
            inputs,  # Contains input_ids for instruction tokens
        )
        logger.info(f"Built prefix embeddings: {prefix_embeds.shape}")

        # Compute full VLM K/V cache by running VLM layers
        # This returns per-layer K/V: {layer_idx: (k, v)} where k, v are [batch, seq, 320]
        vlm_kv_cache_cpu = self.compute_vlm_kv_cache(
            prefix_embeds,  # Now includes vision + instruction!
            num_layers=self.config.num_expert_layers,
        )

        # Use K/V cache - already on TT when compute_vlm_kv_cache uses TT path
        vlm_kv_cache = None
        cross_attn_context_cpu = None

        if self.ttnn_device is not None:
            vlm_kv_cache = {}
            for layer_idx, (k, v) in vlm_kv_cache_cpu.items():
                # Check if already ttnn.Tensor (from TT VLM path)
                if isinstance(k, ttnn.Tensor):
                    vlm_kv_cache[layer_idx] = (k, v)
                else:
                    # Convert from torch to TT
                    k_tt = ttnn.from_torch(
                        k.to(torch.bfloat16),
                        device=self.ttnn_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                    )
                    v_tt = ttnn.from_torch(
                        v.to(torch.bfloat16),
                        device=self.ttnn_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                    )
                    vlm_kv_cache[layer_idx] = (k_tt, v_tt)
            logger.info(f"VLM K/V cache ready for TT, {len(vlm_kv_cache)} layers")
        else:
            # CPU path: use the first available K as cross_attn_context
            # (This is a simplification - each layer should use its own K/V)
            if vlm_kv_cache_cpu:
                first_layer_idx = min(vlm_kv_cache_cpu.keys())
                cross_attn_context_cpu = vlm_kv_cache_cpu[first_layer_idx][0]  # Use K as context
                logger.info(
                    f"Using VLM K from layer {first_layer_idx} as cross-attn context: {cross_attn_context_cpu.shape}"
                )
            else:
                cross_attn_context_cpu = None
                logger.warning("No VLM K/V cache computed, using fallback")

        CHECKPOINTS.checkpoint("end_VLMKVCOMPUTE")

        # ========================================
        # Flow Matching Inference Loop
        # ========================================
        CHECKPOINTS.checkpoint("start_FLOWMATCHING")

        # Initialize noisy actions: start from standard normal noise
        # or zeros (both work, noise gives more diversity)
        noisy_actions = torch.zeros(
            batch_size, chunk_size, self.config.max_action_dim, dtype=torch.float32, device=device
        )

        # Prepare robot state if provided
        if robot_state is not None:
            if robot_state.dim() == 1:
                robot_state = robot_state.unsqueeze(0)  # Add batch dim
            robot_state = robot_state.float()
            # Project state to context space (960-dim) - stored for potential use
            state_context = self.action_heads.project_state(robot_state)

        # Flow matching: integrate from t=0 to t=1
        dt = 1.0 / num_inference_steps

        # Check if TT path is available
        use_tt = (
            self.ttnn_device is not None
            and self.expert_layers.tt_params_initialized
            and self.action_heads.tt_params_initialized
        )

        if use_tt:
            # TT-accelerated flow matching loop - ALL ON TT, NO CPU SYNC UNTIL END

            # Convert noisy_actions to TT ONCE before loop
            noisy_actions_tt = ttnn.from_torch(
                noisy_actions.to(torch.bfloat16),
                device=self.ttnn_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

            # Pre-compute dt as TT scalar for Euler integration
            dt_tt = ttnn.from_torch(
                torch.tensor([[[dt]]], dtype=torch.bfloat16),  # [1, 1, 1] for broadcasting
                device=self.ttnn_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

            for step in range(num_inference_steps):
                t = step / num_inference_steps
                timestep = torch.tensor([t], dtype=torch.float32)

                with torch.no_grad():
                    # Project actions on TT
                    action_emb_tt = self.action_heads.project_actions_tt(noisy_actions_tt)

                    # Get time embedding - 100% on TT
                    timestep_tt = ttnn.from_torch(
                        timestep.unsqueeze(-1).to(torch.bfloat16),  # [batch, 1]
                        device=self.ttnn_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                    )
                    time_emb_tt = self.action_heads.get_time_embedding_tt(timestep_tt)
                    ttnn.deallocate(timestep_tt)

                    # Expand time embedding and add to actions using broadcasting
                    time_emb_3d = ttnn.reshape(time_emb_tt, (1, 1, self.action_heads.expert_hidden_dim))
                    ttnn.deallocate(time_emb_tt)
                    expert_input_tt = ttnn.add(action_emb_tt, time_emb_3d)
                    ttnn.deallocate(action_emb_tt)
                    ttnn.deallocate(time_emb_3d)

                    # Expert layers on TT
                    CHECKPOINTS.checkpoint(f"start_EXPERT_STEP_{step}")
                    expert_output_tt = smolvla_expert_layers_tt(
                        expert_input_tt,
                        self.expert_layers.tt_layer_params,
                        num_layers=self.config.num_expert_layers,
                        vlm_kv_cache=vlm_kv_cache,
                    )
                    CHECKPOINTS.checkpoint(f"end_EXPERT_STEP_{step}")
                    ttnn.deallocate(expert_input_tt)

                    # Predict velocity on TT
                    velocity_tt = self.action_heads.predict_velocity_tt(expert_output_tt)
                    ttnn.deallocate(expert_output_tt)

                    # Euler integration ON TT: noisy_actions += velocity * dt
                    velocity_scaled = ttnn.mul(velocity_tt, dt_tt)
                    ttnn.deallocate(velocity_tt)
                    noisy_actions_new = ttnn.add(noisy_actions_tt, velocity_scaled)
                    ttnn.deallocate(velocity_scaled)
                    ttnn.deallocate(noisy_actions_tt)
                    noisy_actions_tt = noisy_actions_new

            # Convert back to torch ONCE after loop
            ttnn.deallocate(dt_tt)
            noisy_actions = ttnn.to_torch(noisy_actions_tt).float()
            ttnn.deallocate(noisy_actions_tt)
        else:
            # CPU flow matching loop
            for step in range(num_inference_steps):
                t = step / num_inference_steps
                timestep = torch.tensor([t], dtype=torch.float32, device=device)

                with torch.no_grad():
                    # Project noisy actions to expert hidden space (720-dim)
                    action_emb = self.action_heads.project_actions(noisy_actions)

                    # Get time embedding in expert space (720-dim)
                    time_emb = self.action_heads.get_time_embedding(timestep)

                    # Combine action and time in expert space: [batch, chunk_size, 720]
                    expert_input = action_emb + time_emb.unsqueeze(1)

                    # Pass through expert layers (16 transformer layers)
                    # NOW WITH FULL VLM K/V CACHE FOR CROSS-ATTENTION!
                    CHECKPOINTS.checkpoint(f"start_EXPERT_STEP_{step}")
                    expert_output = self.expert_layers(
                        expert_input,
                        vlm_kv_cache=vlm_kv_cache_cpu,  # Pass full per-layer VLM K/V cache
                    )
                    CHECKPOINTS.checkpoint(f"end_EXPERT_STEP_{step}")

                    # Predict velocity: [batch, chunk_size, 32]
                    velocity = self.action_heads.predict_velocity(expert_output)

                    # Euler integration step
                    noisy_actions = noisy_actions + velocity * dt

        CHECKPOINTS.checkpoint("end_FLOWMATCHING")

        # Extract final actions (trim to actual action_dim)
        actions = noisy_actions[:, :, :action_dim]

        CHECKPOINTS.checkpoint("end_ACTIONPREDICTION")

        # Return as numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        return actions[0]  # Remove batch dimension


# ============================================================================
# Pytest Tests
# ============================================================================
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("iterations", [1])
def test_smolvla_model(mesh_device, iterations):
    """
    Test SmolVLA model on TT hardware.

    To run:
        python -m pytest models/tt_transformers/tt/multimodal/smol_vla.py::test_smolvla_model -v

    Prerequisites:
        - SmolVLA weights downloaded (automatic from HuggingFace)
    """
    logger.info("Loading SmolVLA model...")

    # Load model
    vla = SmolVLAForActionPrediction.from_pretrained(
        repo_id="lerobot/smolvla_base",
        ttnn_device=mesh_device,
    )

    # Create test image
    test_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

    # Run inference
    results: List[Dict[str, float]] = []
    for i in range(iterations):
        CHECKPOINTS.reset()
        action = vla.predict_action(
            images=[test_image],
            instruction="pick up the object",
        )
        results.append(CHECKPOINTS.analyze())
        logger.info(f"Iteration {i+1}: action shape = {action.shape}")

    # Combine timing results
    if results:
        combined_results = {k: 0.0 for k in results[0].keys()}
        for r in results[min(iterations - 1, 1) :]:
            for k, v in r.items():
                combined_results[k] += v
        avg_results = {k: round(v / max(len(results) - 1, 1), 6) for k, v in combined_results.items()}
        avg_results = dict(sorted(avg_results.items(), key=lambda x: x[1], reverse=True))

        logger.info(f"Predicted action shape: {action.shape}")
        logger.info(f"Timings after {iterations} iterations: {json.dumps(avg_results, indent=4)}")


@pytest.mark.parametrize("repo_id", ["lerobot/smolvla_base"])
def test_smolvla_cpu_only(repo_id):
    """
    Test SmolVLA model on CPU only (no TT device required).

    To run:
        python -m pytest models/tt_transformers/tt/multimodal/smol_vla.py::test_smolvla_cpu_only -v
    """
    logger.info(f"Loading SmolVLA from {repo_id} (CPU only)...")

    # Load config
    config = SmolVLAConfig.from_pretrained(repo_id)
    logger.info(f"Config loaded: {config.vlm_model_name}")
    logger.info(f"  Text: {config.text_num_layers} layers, {config.text_hidden_size} hidden")
    logger.info(f"  Vision: {config.vision_num_layers} layers, {config.vision_hidden_size} hidden")
    logger.info(f"  Expert: {config.num_expert_layers} layers")

    # Load weights
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    state_dict = safetensors_load_file(weights_path)

    logger.info(f"Loaded {len(state_dict)} weight tensors")

    # Test weight mapping
    text_weights = map_smolvla_to_tt_text_keys(state_dict)
    logger.info(f"Mapped {len(text_weights)} text model weights")

    expert_weights = map_smolvla_expert_keys(state_dict)
    logger.info(f"Mapped {len(expert_weights)} expert layer weights")

    vision_weights = extract_vision_state_dict(state_dict)
    logger.info(f"Extracted {len(vision_weights)} vision encoder weights")

    connector_weights = extract_connector_state_dict(state_dict)
    logger.info(f"Extracted {len(connector_weights)} connector weights")

    action_weights = extract_action_heads_state_dict(state_dict)
    logger.info(f"Extracted {len(action_weights)} action head weights")

    # Verify key mappings
    assert "tok_embeddings.weight" in text_weights, "Missing tok_embeddings"
    assert "output.weight" in text_weights, "Missing output weight"
    assert "layers.0.attention.wq.weight" in text_weights, "Missing layer 0 attention"

    logger.info("CPU test passed!")


if __name__ == "__main__":
    # Run CPU test by default
    logging.basicConfig(level=logging.INFO)
    test_smolvla_cpu_only("lerobot/smolvla_base")
