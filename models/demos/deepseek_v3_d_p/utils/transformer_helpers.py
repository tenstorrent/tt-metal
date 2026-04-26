# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for deepseek_v3_d_p transformer tests.

Provides:
- HF model creation and weight extraction helpers
- Tokenization helpers (JSON prompts, InfiniteBench)
- Host reference caching for PCC validation
- InfiniteBench dataset download and caching
"""

import gc
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
from loguru import logger
from transformers import DynamicCache
from transformers.modeling_utils import no_init_weights

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model, DeepseekV3MoE


@dataclass
class LayerByLayerResult:
    """Result from load_and_compute_layer_by_layer function."""

    state_dict: dict | None  # TT format state dict, or None if extract_tt_weights=False
    ref_snapshots: list[torch.Tensor] | None  # Reference outputs, or None if compute_reference=False
    ref_kvpe_list: list[torch.Tensor] | None  # Reference KV cache, or None if compute_reference=False


def _log_memory(label: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / 1024**3
    logger.info(f"[MEMORY] {label}: {rss_gb:.2f} GB")


# --- Constants ---

PROMPTS_PATH = Path("models/demos/deepseek_v3/demo/test_prompts_1024.json")
ABC_1K_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_prompt_ABC_1k.json")

# Subset name -> JSONL filename on HuggingFace
INFINITEBENCH_SUBSETS = {
    "passkey": "passkey.jsonl",
    "kv_retrieval": "kv_retrieval.jsonl",
    "longdialogue_qa_eng": "longdialogue_qa_eng.jsonl",
    "longbook_qa_eng": "longbook_qa_eng.jsonl",
}

INFINITEBENCH_CACHE_DIR = Path(
    os.environ.get("TT_DS_PREFILL_INFINITEBENCH_CACHE", "/tmp/deepseek_v3_transformer_inputs")
)


# --- HF model helpers ---


def create_hf_model(config, num_layers, n_routed_experts=None):
    """Create HF DeepseekV3Model with num_layers and random weights."""
    test_config = deepcopy(config)
    test_config.num_hidden_layers = num_layers
    test_config._attn_implementation = "eager"
    if n_routed_experts is not None:
        test_config.n_routed_experts = n_routed_experts

    model = DeepseekV3Model(test_config)
    return model.eval().to(torch.bfloat16)


def extract_layer_state_dict(full_sd, layer_idx, hf_layer):
    """Extract one layer's weights from HF state_dict into TtPrefillBlock format."""
    prefix = f"layers.{layer_idx}."
    is_moe = isinstance(hf_layer.mlp, DeepseekV3MoE)

    layer_sd = {
        "attn_norm_weight": full_sd[f"{prefix}input_layernorm.weight"],
        "mla_weights": {
            "q_a_proj.weight": full_sd[f"{prefix}self_attn.q_a_proj.weight"],
            "q_a_layernorm.weight": full_sd[f"{prefix}self_attn.q_a_layernorm.weight"],
            "q_b_proj.weight": full_sd[f"{prefix}self_attn.q_b_proj.weight"],
            "kv_a_proj_with_mqa.weight": full_sd[f"{prefix}self_attn.kv_a_proj_with_mqa.weight"],
            "kv_a_layernorm.weight": full_sd[f"{prefix}self_attn.kv_a_layernorm.weight"],
            "kv_b_proj.weight": full_sd[f"{prefix}self_attn.kv_b_proj.weight"],
            "o_proj.weight": full_sd[f"{prefix}self_attn.o_proj.weight"],
        },
        "ffn_norm_weight": full_sd[f"{prefix}post_attention_layernorm.weight"],
    }

    if is_moe:
        layer_sd["gate_weights"] = {
            "weight": full_sd[f"{prefix}mlp.gate.weight"],
            "e_score_correction_bias": full_sd[f"{prefix}mlp.gate.e_score_correction_bias"],
        }
        layer_sd["routed_expert_weights"] = [
            {
                "gate_proj": full_sd[f"{prefix}mlp.experts.{j}.gate_proj.weight"],
                "up_proj": full_sd[f"{prefix}mlp.experts.{j}.up_proj.weight"],
                "down_proj": full_sd[f"{prefix}mlp.experts.{j}.down_proj.weight"],
            }
            for j in range(len(hf_layer.mlp.experts))
        ]
        layer_sd["shared_expert_weights"] = {
            "gate_proj": full_sd[f"{prefix}mlp.shared_experts.gate_proj.weight"],
            "up_proj": full_sd[f"{prefix}mlp.shared_experts.up_proj.weight"],
            "down_proj": full_sd[f"{prefix}mlp.shared_experts.down_proj.weight"],
        }
    else:
        layer_sd["ffn_weights"] = {
            "gate_proj": full_sd[f"{prefix}mlp.gate_proj.weight"],
            "up_proj": full_sd[f"{prefix}mlp.up_proj.weight"],
            "down_proj": full_sd[f"{prefix}mlp.down_proj.weight"],
        }

    return layer_sd


def extract_tt_state_dict(hf_model):
    """Extract state_dict in TtPrefillTransformer format from HF model."""
    sd = hf_model.state_dict()
    num_layers = len(hf_model.layers)

    result = {
        "embed_weight": sd["embed_tokens.weight"].float(),
        "norm_weight": sd["norm.weight"],
        "layers": [],
    }

    for i in range(num_layers):
        layer_sd = extract_layer_state_dict(sd, i, hf_model.layers[i])
        result["layers"].append(layer_sd)

    return result


def tt_state_dict_to_hf_state_dict(tt_sd):
    """Reverse key mapping from TT format to HF DeepseekV3Model state_dict keys.

    Tensor references are shared (no copy).
    """
    hf_sd = {}
    hf_sd["embed_tokens.weight"] = tt_sd["embed_weight"]
    hf_sd["norm.weight"] = tt_sd["norm_weight"]

    for i, layer in enumerate(tt_sd["layers"]):
        prefix = f"layers.{i}."
        hf_sd[f"{prefix}input_layernorm.weight"] = layer["attn_norm_weight"]

        for key, value in layer["mla_weights"].items():
            hf_sd[f"{prefix}self_attn.{key}"] = value

        hf_sd[f"{prefix}post_attention_layernorm.weight"] = layer["ffn_norm_weight"]

        if "ffn_weights" in layer:
            # Dense FFN
            for key in ("gate_proj", "up_proj", "down_proj"):
                hf_sd[f"{prefix}mlp.{key}.weight"] = layer["ffn_weights"][key]
        else:
            # MoE
            hf_sd[f"{prefix}mlp.gate.weight"] = layer["gate_weights"]["weight"]
            hf_sd[f"{prefix}mlp.gate.e_score_correction_bias"] = layer["gate_weights"]["e_score_correction_bias"]

            for j, expert in enumerate(layer["routed_expert_weights"]):
                for key in ("gate_proj", "up_proj", "down_proj"):
                    hf_sd[f"{prefix}mlp.experts.{j}.{key}.weight"] = expert[key]

            for key in ("gate_proj", "up_proj", "down_proj"):
                hf_sd[f"{prefix}mlp.shared_experts.{key}.weight"] = layer["shared_expert_weights"][key]

    return hf_sd


def create_hf_model_with_weights(config, num_layers, hf_sd):
    """Create HF DeepseekV3Model with pretrained weights (no random init)."""
    test_config = deepcopy(config)
    test_config.num_hidden_layers = num_layers
    test_config._attn_implementation = "eager"

    logger.info(f"Creating DeepseekV3Model with {num_layers} layers...")
    with no_init_weights():
        model = DeepseekV3Model(test_config)
    logger.info("Model structure created successfully")

    # Load state dict layer-by-layer to avoid OOM (loading all layers at once doubles memory usage)
    # strict=False: rotary embedding buffers are computed from config, not from weights
    logger.info(f"Loading state dict with {len(hf_sd)} keys (layer-by-layer to minimize memory)...")

    all_missing = set()
    all_unexpected = set()

    try:
        # Load embeddings first
        embedding_keys = [k for k in hf_sd.keys() if k.startswith("embed_tokens")]
        if embedding_keys:
            logger.info(f"Loading embeddings ({len(embedding_keys)} keys)...")
            embedding_sd = {k: hf_sd[k] for k in embedding_keys}
            missing, unexpected = model.load_state_dict(embedding_sd, strict=False)
            all_missing.update(missing)
            all_unexpected.update(unexpected)
            # Free memory immediately
            for k in embedding_keys:
                del hf_sd[k]
            del embedding_sd
            gc.collect()
            logger.info("Embeddings loaded, memory freed")

        # Load each layer sequentially
        for i in range(num_layers):
            layer_keys = [k for k in hf_sd.keys() if k.startswith(f"layers.{i}.")]
            if layer_keys:
                logger.info(f"Loading layer {i} ({len(layer_keys)} keys)...")
                layer_sd = {k: hf_sd[k] for k in layer_keys}
                missing, unexpected = model.load_state_dict(layer_sd, strict=False)
                all_missing.update(missing)
                all_unexpected.update(unexpected)
                # Free memory immediately
                for k in layer_keys:
                    del hf_sd[k]
                del layer_sd
                gc.collect()
                logger.info(f"Layer {i} loaded, memory freed")

        # Load norm last
        norm_keys = [k for k in hf_sd.keys() if k.startswith("norm")]
        if norm_keys:
            logger.info(f"Loading norm ({len(norm_keys)} keys)...")
            norm_sd = {k: hf_sd[k] for k in norm_keys}
            missing, unexpected = model.load_state_dict(norm_sd, strict=False)
            all_missing.update(missing)
            all_unexpected.update(unexpected)
            # Free memory immediately
            for k in norm_keys:
                del hf_sd[k]
            del norm_sd
            gc.collect()
            logger.info("Norm loaded, memory freed")

        logger.info("State dict loaded successfully (layer-by-layer)")

    except Exception as e:
        logger.error(f"FAILED to load state dict: {type(e).__name__}: {e}")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

    if all_missing:
        real_missing = [k for k in all_missing if "rotary_emb" not in k]
        if real_missing:
            logger.warning(f"Missing weight keys (not rotary_emb): {real_missing}")
        logger.info(f"Skipped {len(all_missing)} model keys not in weights (rotary_emb buffers)")
    if all_unexpected:
        logger.warning(f"Unexpected keys in state dict: {all_unexpected}")

    logger.info("Converting model to eval mode and bfloat16...")
    result = model.eval().to(torch.bfloat16)
    logger.info("Model conversion complete")
    return result


def get_4d_causal_mask(attention_mask, ignore_padding=False):
    "Get 4D causal attention mask for prefill. If ignore_padding=True, returns a purely causal mask that does not account for padding tokens."

    if ignore_padding:
        # torch.where(torch.tril(torch.ones(5,5)) == 1, 0, -1e38)
        # tensor([[ 0.0000e+00, -1.0000e+38, -1.0000e+38, -1.0000e+38, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00, -1.0000e+38, -1.0000e+38, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+38, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]])
        return (
            torch.where(torch.tril(torch.ones(attention_mask.shape[-1], attention_mask.shape[-1])) == 1, 0, -1e38)
            .unsqueeze(0)
            .unsqueeze(0)
        )
    else:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask,
            (1, attention_mask.shape[-1]),
            inputs_embeds=torch.zeros_like(attention_mask, dtype=torch.bfloat16),
            past_key_values_length=0,
        )
        return attention_mask_4d


def load_and_compute_layer_by_layer(
    model_path: Path,
    config,
    num_layers: int,
    token_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    compute_reference: bool = True,
    build_ttnn_cache: bool = True,
    weight_cache_path: Path | None = None,
    mesh_device: ttnn.MeshDevice | None = None,
    seq_len: int = 1024,
    num_links: int = 2,
    topology: ttnn.Topology = ttnn.Topology.Linear,
    sp_axis: int = 0,
    tp_axis: int = 1,
    capacity_factor: int = 32,
    gate_fallback_mode=None,
    routed_expert_activations_dtype=ttnn.bfloat8_b,
    routed_expert_weights_dtype=ttnn.bfloat4_b,
    shared_expert_activations_dtype=ttnn.bfloat16,
    shared_expert_weights_dtype=ttnn.bfloat8_b,
    ignore_padding=True,
) -> LayerByLayerResult:
    """
    Process layers one-at-a-time: load → compute reference → build cache → clear → next.

    Peak memory: ~21GB regardless of num_layers (no accumulation).

    Args:
        model_path: Path to HF model directory with safetensors
        config: HF model config
        num_layers: Number of layers to process
        token_ids: Input tokens [1, seq_len] (required if compute_reference=True)
        compute_reference: If True, compute forward passes and return reference outputs
        build_ttnn_cache: If True, build .ttnn cache files using device=None
        weight_cache_path: Cache directory (required if build_ttnn_cache=True)
        mesh_device: Mesh device reference (required if build_ttnn_cache=True)
        attention_mask: Attention mask [shape TBD] (required if compute_reference=True)

    Returns:
        LayerByLayerResult(state_dict=None, ref_snapshots, ref_kvpe_list)
        Note: state_dict is always None (cache built to disk instead)
    """
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
    from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
    from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
    from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding

    if gate_fallback_mode is None:
        gate_fallback_mode = GateComputeMode.HOST_ALL

    # Validation
    if not (compute_reference or build_ttnn_cache):
        raise ValueError("At least one of compute_reference or build_ttnn_cache must be True")
    if compute_reference and token_ids is None:
        raise ValueError("token_ids required when compute_reference=True")
    if build_ttnn_cache and (weight_cache_path is None or mesh_device is None):
        raise ValueError("weight_cache_path and mesh_device required when build_ttnn_cache=True")

    logger.info(
        f"Processing {num_layers} layers from {model_path} "
        f"(compute_reference={compute_reference}, build_ttnn_cache={build_ttnn_cache})"
    )

    _log_memory("Start of load_and_compute_layer_by_layer")

    # Create LazyStateDict
    lazy_sd = LazyStateDict(Path(model_path))

    # Initialize outputs
    ref_snapshots = [] if compute_reference else None
    ref_kvpe_list = None
    ref_cache = None

    # Create hf_model only if computing reference
    hf_model = None
    h_ref = None
    if compute_reference:
        test_config = deepcopy(config)
        test_config.num_hidden_layers = num_layers
        test_config._attn_implementation = "eager"

        logger.info(f"Creating empty HF model structure for reference computation...")
        with no_init_weights():
            hf_model = DeepseekV3Model(test_config)
        _log_memory("After creating HF model structure")

        # Setup forward pass inputs
        seq_len = token_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        ref_cache = DynamicCache()

    # --- Process Embeddings ---
    logger.info("Processing embeddings...")
    embed_sd = sub_state_dict(lazy_sd, "model.embed_tokens.")
    embed_dequant = dequantize_state_dict(embed_sd, config)

    if compute_reference:
        embed_with_prefix = {f"embed_tokens.{k}": v for k, v in embed_dequant.items()}
        hf_model.load_state_dict(embed_with_prefix, strict=False)
        with torch.no_grad():
            h_ref = hf_model.embed_tokens(token_ids).to(torch.bfloat16)
        ref_snapshots.append(h_ref)
        # Clear embedding weights from hf_model
        hf_model.embed_tokens.weight.data = torch.empty(0)
        del embed_with_prefix

    attention_mask = get_4d_causal_mask(attention_mask, ignore_padding=ignore_padding)

    if build_ttnn_cache:
        # Build embedding cache (device=None, no accumulation!)
        TtParallelEmbedding.build_ttnn_cache(
            torch_weight=embed_dequant["weight"].float(),
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=weight_cache_path,
            tp_axis=tp_axis,
            dtype=ttnn.bfloat16,
        )

    for k in embed_sd.keys():
        lazy_sd.evict(k)
    del embed_sd, embed_dequant
    gc.collect()
    _log_memory("After embeddings processed and cleared")
    logger.debug("Embeddings processed, cache cleared")

    # --- Process Layers ---
    first_k_dense = config.first_k_dense_replace
    n_routed = config.n_routed_experts

    for i in range(num_layers):
        logger.info(f"Processing layer {i}/{num_layers}...")

        layer_sd = sub_state_dict(lazy_sd, f"model.layers.{i}.")
        layer_dequant = dequantize_state_dict(layer_sd, config)

        if compute_reference:
            layer_with_prefix = {f"layers.{i}.{k}": v for k, v in layer_dequant.items()}
            hf_model.load_state_dict(layer_with_prefix, strict=False)

            with torch.no_grad():
                layer_out = hf_model.layers[i](
                    h_ref,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=ref_cache,
                    use_cache=True,
                )
                h_ref = layer_out[0]
            ref_snapshots.append(h_ref)

            # Clear layer weights from hf_model
            for param in hf_model.layers[i].parameters():
                param.data = torch.empty(0)
            del layer_with_prefix

        if build_ttnn_cache:
            # Build TTNN cache for this layer (device=None, no accumulation!)
            is_dense = i < first_k_dense

            # Build layer_dict in same format as before
            layer_dict = {
                "attn_norm_weight": layer_dequant["input_layernorm.weight"],
                "mla_weights": {
                    "q_a_proj.weight": layer_dequant["self_attn.q_a_proj.weight"],
                    "q_a_layernorm.weight": layer_dequant["self_attn.q_a_layernorm.weight"],
                    "q_b_proj.weight": layer_dequant["self_attn.q_b_proj.weight"],
                    "kv_a_proj_with_mqa.weight": layer_dequant["self_attn.kv_a_proj_with_mqa.weight"],
                    "kv_a_layernorm.weight": layer_dequant["self_attn.kv_a_layernorm.weight"],
                    "kv_b_proj.weight": layer_dequant["self_attn.kv_b_proj.weight"],
                    "o_proj.weight": layer_dequant["self_attn.o_proj.weight"],
                },
                "ffn_norm_weight": layer_dequant["post_attention_layernorm.weight"],
            }

            if is_dense:
                layer_dict["ffn_weights"] = {
                    "gate_proj": layer_dequant["mlp.gate_proj.weight"],
                    "up_proj": layer_dequant["mlp.up_proj.weight"],
                    "down_proj": layer_dequant["mlp.down_proj.weight"],
                }
            else:
                layer_dict["gate_weights"] = {
                    "weight": layer_dequant["mlp.gate.weight"],
                    "e_score_correction_bias": layer_dequant["mlp.gate.e_score_correction_bias"],
                }
                layer_dict["routed_expert_weights"] = [
                    {
                        "gate_proj": layer_dequant[f"mlp.experts.{j}.gate_proj.weight"],
                        "up_proj": layer_dequant[f"mlp.experts.{j}.up_proj.weight"],
                        "down_proj": layer_dequant[f"mlp.experts.{j}.down_proj.weight"],
                    }
                    for j in range(n_routed)
                ]
                layer_dict["shared_expert_weights"] = {
                    "gate_proj": layer_dequant["mlp.shared_experts.gate_proj.weight"],
                    "up_proj": layer_dequant["mlp.shared_experts.up_proj.weight"],
                    "down_proj": layer_dequant["mlp.shared_experts.down_proj.weight"],
                }

            # Build TTNN cache (device=None) - NOT accumulated in memory!
            TtPrefillBlock.build_ttnn_cache(
                state_dict=layer_dict,
                layer_idx=i,
                cache_path=weight_cache_path,
                mesh_device=mesh_device,
                config=config,
                seq_len=seq_len,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                capacity_factor=capacity_factor,
                gate_fallback_mode=gate_fallback_mode,
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
            )

            # Free layer_dict immediately!
            del layer_dict

        for k in layer_sd.keys():
            lazy_sd.evict(k)
        del layer_sd, layer_dequant
        gc.collect()
        _log_memory(f"After layer {i} cleared")
        logger.debug(f"Layer {i} processed, cache cleared")

    # Extract KVPE if computed reference
    if compute_reference:
        ref_kvpe_list = [ref_cache.key_cache[i] for i in range(num_layers)]

    # --- Process Norm ---
    logger.info("Processing norm...")
    norm_sd = sub_state_dict(lazy_sd, "model.norm.")
    norm_dequant = dequantize_state_dict(norm_sd, config)

    if compute_reference:
        norm_with_prefix = {f"norm.{k}": v for k, v in norm_dequant.items()}
        hf_model.load_state_dict(norm_with_prefix, strict=False)
        logger.debug(f"[norm] h_ref {h_ref.dtype=}, norm_weight dtype={norm_dequant['weight'].dtype}")
        with torch.no_grad():
            h_ref = hf_model.norm(h_ref)
        ref_snapshots.append(h_ref)
        del norm_with_prefix

    if build_ttnn_cache:
        # Build norm cache
        TtDistributedRmsNorm.build_ttnn_cache(
            torch_weight=norm_dequant["weight"],
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=weight_cache_path,
            cache_name_prefix="norm",
        )

    for k in norm_sd.keys():
        lazy_sd.evict(k)
    del norm_sd, norm_dequant
    gc.collect()

    # --- Process LM Head ---
    logger.info("Processing lm_head...")
    lm_head_sd = sub_state_dict(lazy_sd, "lm_head.")
    lm_head_dequant = dequantize_state_dict(lm_head_sd, config)

    if compute_reference:
        # Apply lm_head projection: logits = h_ref @ lm_head_weight.T
        logger.debug(f"[lm_head] h_ref {h_ref.dtype=}, lm_head_weight.dtype={lm_head_dequant['weight'].dtype}")
        lm_head_weight = lm_head_dequant["weight"].to(torch.bfloat16)
        with torch.no_grad():
            h_ref_lm = torch.nn.functional.linear(h_ref.to(torch.bfloat16), lm_head_weight)
        ref_snapshots.append(h_ref_lm)
        del lm_head_weight

    if build_ttnn_cache:
        TtLMHead.build_ttnn_cache(
            torch_weight=lm_head_dequant["weight"],
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=weight_cache_path,
        )

    for k in lm_head_sd.keys():
        lazy_sd.evict(k)
    del lm_head_sd, lm_head_dequant
    gc.collect()
    _log_memory("After lm_head processed and cleared")

    # Cleanup
    lazy_sd.close()

    _log_memory("End of load_and_compute_layer_by_layer")
    logger.info(f"Processing complete ({num_layers} layers, peak memory minimized)")

    return LayerByLayerResult(
        state_dict=None,  # Never built! Cache saved to disk instead
        ref_snapshots=ref_snapshots,
        ref_kvpe_list=ref_kvpe_list,
    )


def check_reference_cache_exists(cache_key: str) -> bool:
    """
    Check if reference output cache exists for the given cache key.

    Reference cache contains forward pass outputs from HF model for PCC validation.
    This cache is machine-independent and can be generated once and shared.

    Args:
        cache_key: Cache identifier like "pretrained_json_prompts_isl1024_layers24_experts256"

    Returns:
        True if cache file exists, False otherwise
    """
    cache_dir = Path(os.environ.get("TT_DS_PREFILL_HOST_REF_CACHE", "/tmp/deepseek_v3_transformer_ref_cache"))
    cache_path = cache_dir / f"{cache_key}.pt"

    exists = cache_path.exists()

    if exists:
        logger.info(f"Reference cache found: {cache_path}")
    else:
        logger.debug(f"Reference cache not found: {cache_path}")

    return exists


def save_reference_cache(cache_key: str, ref_snapshots, ref_kvpe_list):
    """Save reference outputs to cache file."""
    cache_dir = Path(os.environ.get("TT_DS_PREFILL_HOST_REF_CACHE", "/tmp/deepseek_v3_transformer_ref_cache"))
    cache_path = cache_dir / f"{cache_key}.pt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"ref_snapshots": ref_snapshots, "ref_kvpe_list": ref_kvpe_list}, cache_path)
    logger.info(f"Saved reference to {cache_path} ({len(ref_snapshots)} snapshots, {len(ref_kvpe_list)} KVPE)")


def load_reference_cache(cache_key: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load reference outputs from cache file."""
    cache_dir = Path(os.environ.get("TT_DS_PREFILL_HOST_REF_CACHE", "/tmp/deepseek_v3_transformer_ref_cache"))
    cache_path = cache_dir / f"{cache_key}.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"Reference cache not found: {cache_path}")

    cached = torch.load(cache_path, weights_only=True)

    if "ref_snapshots" not in cached or "ref_kvpe_list" not in cached:
        raise ValueError(f"Invalid cache format in {cache_path}")

    logger.info(f"Loaded reference from {cache_path}")
    return cached["ref_snapshots"], cached["ref_kvpe_list"]


# --- Tokenization helpers ---
def tokenize_prompt_to_isl(
    tokenizer, max_isl: int, prompt_text: str = "Capital of France is", debug: bool = False
) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
    """Tokenize a prompt, padding/truncating to exactly max_isl tokens.

    Returns:
        (input_ids, attention_mask, tokens): input_ids and attention_mask are [1, max_isl] tensors;
        tokens is a list of token strings (only when debug=True, else None).
    """
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",  # return PyTorch tensors
        padding="max_length",  # pad batch to max_length
        truncation=True,  # truncate if longer than max_length
        max_length=max_isl,  # hard cap at <max_isl> tokens
        return_attention_mask=True,
    )

    input_ids: torch.Tensor = inputs.input_ids  # shape [B, 1024] (padded or capped)
    attention_mask: torch.Tensor = inputs.attention_mask
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist()) if debug else None

    logger.debug(f"Input IDs shape: {input_ids.shape}")
    logger.debug(f"Attention mask sum: {attention_mask.sum(dim=1)}")  # actual non‑pad tokens

    return input_ids, attention_mask, tokens


def tokenize_prompt_to_chat_template(
    tokenizer,
    max_isl: int,
    user_prompt: str = "What is the capital of France",
    system_prompt: str = "You are a friendly assistant",
    debug: bool = False,
) -> tuple[torch.Tensor, list[str] | None]:
    """Apply chat template and tokenize, padding/truncating to max_isl tokens.

    Returns:
        (input_ids, tokens): input_ids is a [1, max_isl] tensor;
        tokens is a list of token strings (only when debug=True, else None).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # apply chat template AND tokenize in one shot
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding="max_length",  # pad batch to max_len
        truncation=True,  # truncate if longer
        max_length=max_isl,  # hard cap at <max_isl> tokens
        return_attention_mask=True,  # <-- ignored for some reason
    )
    logger.debug(f"Input IDs shape: {inputs.shape}")
    tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist()) if debug else None

    return inputs, tokens


# --- InfiniteBench download ---


def download_infinitebench_subset(subset: str) -> Path:
    """
    Download one InfiniteBench subset (longest entry) and cache as JSON.

    Args:
        subset: One of the keys in INFINITEBENCH_SUBSETS.

    Returns:
        Path to the cached JSON file containing the prompt.
    """
    if subset not in INFINITEBENCH_SUBSETS:
        raise ValueError(f"Unknown InfiniteBench subset '{subset}'. Choose from: {list(INFINITEBENCH_SUBSETS)}")

    filename = f"infinitebench_{subset}.json"
    cached_path = INFINITEBENCH_CACHE_DIR / filename

    if cached_path.exists():
        logger.info(f"InfiniteBench '{subset}' already cached at {cached_path}")
        return cached_path

    logger.info(f"Downloading InfiniteBench subset '{subset}' from HuggingFace...")

    from datasets import load_dataset

    ds = load_dataset(
        "xinrongzhang2022/InfiniteBench",
        data_files=INFINITEBENCH_SUBSETS[subset],
        split="train",
    )

    # Find the longest entry by character count (avoids tokenizer dependency)
    best = max(ds, key=lambda e: len(e.get("context") or ""))
    ctx = best["context"] or ""

    logger.info(f"Selected entry id={best.get('id', 'N/A')}, {len(ctx):,} chars")

    output = {
        "subset": subset,
        "source": "xinrongzhang2022/InfiniteBench",
        "license": "Apache-2.0",
        "id": best.get("id"),
        "prompt": ctx,
    }

    INFINITEBENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cached_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {cached_path.name} ({cached_path.stat().st_size:,} bytes)")
    return cached_path
