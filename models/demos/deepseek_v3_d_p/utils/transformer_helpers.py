# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from loguru import logger
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3.demo.demo import load_prompts_from_json
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model, DeepseekV3MoE

# --- Constants ---

PROMPTS_PATH = Path("models/demos/deepseek_v3/demo/test_prompts_1024.json")

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

    with no_init_weights():
        model = DeepseekV3Model(test_config)

    # strict=False: rotary embedding buffers are computed from config, not from weights
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    if missing:
        real_missing = [k for k in missing if "rotary_emb" not in k]
        if real_missing:
            logger.warning(f"Missing weight keys (not rotary_emb): {real_missing}")
        logger.info(f"Skipped {len(missing)} model keys not in weights (rotary_emb buffers)")
    if unexpected:
        logger.warning(f"Unexpected keys in state dict: {unexpected}")

    return model.eval().to(torch.bfloat16)


# --- Tokenization helpers ---


def get_pad_id(tokenizer):
    """Resolve pad token ID, matching generator.py logic."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    return int(pad_id)


def tokenize_prompts_to_isl(tokenizer, prompts_path, isl_total, sp_factor):
    """Tokenize prompts from JSON, concatenate, truncate/pad to isl_total."""
    prompts = load_prompts_from_json(str(prompts_path))
    pad_id = get_pad_id(tokenizer)

    all_tokens = []
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        all_tokens.extend(ids)
        if len(all_tokens) >= isl_total:
            break

    # Truncate or pad
    if len(all_tokens) >= isl_total:
        all_tokens = all_tokens[:isl_total]
    else:
        all_tokens.extend([pad_id] * (isl_total - len(all_tokens)))

    # Alignment check
    isl_per_chip = isl_total // sp_factor
    assert (
        isl_per_chip % ttnn.TILE_SIZE == 0
    ), f"isl_per_chip={isl_per_chip} must be divisible by TILE_SIZE={ttnn.TILE_SIZE}"

    return torch.tensor(all_tokens, dtype=torch.int64).unsqueeze(0)  # [1, isl_total]


def tokenize_infinitebench_to_isl(tokenizer, prompt_text, isl_total, sp_factor):
    """Tokenize a raw InfiniteBench prompt string, truncate/pad to isl_total."""
    pad_id = get_pad_id(tokenizer)

    all_tokens = tokenizer.encode(prompt_text)

    if len(all_tokens) >= isl_total:
        all_tokens = all_tokens[:isl_total]
    else:
        all_tokens.extend([pad_id] * (isl_total - len(all_tokens)))

    isl_per_chip = isl_total // sp_factor
    assert (
        isl_per_chip % ttnn.TILE_SIZE == 0
    ), f"isl_per_chip={isl_per_chip} must be divisible by TILE_SIZE={ttnn.TILE_SIZE}"

    return torch.tensor(all_tokens, dtype=torch.int64).unsqueeze(0)  # [1, isl_total]


# --- Host reference caching ---


def get_or_compute_host_reference(
    hf_model,
    token_ids,
    num_layers,
    cache_key,
    is_ci,
):
    """
    Get host reference snapshots, using file cache if available.

    Follows the caching pattern from test_mla.py: first run computes and saves,
    subsequent runs load from cache. CI environments must use pre-computed cache.

    Returns:
        List of ref_snapshot tensors: [embed_out, layer0_out, ..., layerN_out, norm_out]
    """
    cache_dir = Path(os.environ.get("TT_DS_PREFILL_HOST_REF_CACHE", "/tmp/deepseek_v3_transformer_ref_cache"))
    cache_path = cache_dir / f"{cache_key}.pt"

    if cache_path.exists():
        logger.info(f"Loading cached reference results from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        ref_snapshots = cached["ref_snapshots"]
        logger.info(f"Loaded {len(ref_snapshots)} cached reference snapshots")
        return ref_snapshots

    assert not is_ci, (
        f"Host reference cache missing in CI: {cache_path}. " "Run the test locally first to generate the cache."
    )

    # Compute host reference step-by-step
    profiler.start("host_reference_forward")

    seq_len = token_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

    with torch.no_grad():
        h_ref = hf_model.embed_tokens(token_ids).to(torch.bfloat16)
    ref_snapshots = [h_ref]

    for i in range(num_layers):
        with torch.no_grad():
            layer_out = hf_model.layers[i](h_ref, attention_mask=attention_mask, position_ids=position_ids)
            h_ref = layer_out[0]
        ref_snapshots.append(h_ref)

    with torch.no_grad():
        h_ref = hf_model.norm(h_ref)
    ref_snapshots.append(h_ref)

    profiler.end("host_reference_forward")

    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ref_snapshots": ref_snapshots}, cache_path)
    logger.info(f"Saved reference results to {cache_path} ({len(ref_snapshots)} snapshots)")

    return ref_snapshots


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
