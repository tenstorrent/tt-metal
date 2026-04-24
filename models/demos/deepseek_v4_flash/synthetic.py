# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from models.demos.deepseek_v4_flash.fp4 import EXPERT_FP4_BLOCK_SIZE, pack_fp4_indices


def generate_tiny_hf_checkpoint(
    output_dir: str | Path,
    *,
    num_hidden_layers: int = 1,
    num_routed_experts: int = 4,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Synthetic checkpoint already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "inference").mkdir()

    config = tiny_config_dict(num_hidden_layers=num_hidden_layers, num_routed_experts=num_routed_experts)
    inference_config = tiny_inference_config_dict(config)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "inference" / "config.json", inference_config)
    _write_json(output_dir / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(output_dir / "tokenizer.json", {"version": "1.0", "truncation": None})

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensors = _tiny_tensors(config, generator)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(output_dir / shard_name))

    index = {
        "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())},
        "weight_map": {key: shard_name for key in sorted(tensors)},
    }
    _write_json(output_dir / "model.safetensors.index.json", index)
    return output_dir


def tiny_config_dict(*, num_hidden_layers: int = 1, num_routed_experts: int = 4) -> dict:
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be >= 1")
    compress_pattern = [0, 0, 4]
    compress_ratios = [compress_pattern[i] if i < len(compress_pattern) else 4 for i in range(num_hidden_layers)]
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 32,
        "index_head_dim": 8,
        "index_n_heads": 4,
        "index_topk": 8,
        "initializer_range": 0.02,
        "max_position_embeddings": 1024,
        "model_type": "deepseek_v4",
        "moe_intermediate_size": 32,
        "n_routed_experts": num_routed_experts,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": num_hidden_layers,
        "num_hash_layers": min(3, num_hidden_layers),
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 0,
        "o_groups": 4,
        "o_lora_rank": 16,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 4,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 64,
            "type": "yarn",
        },
        "rope_theta": 10000,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "sliding_window": 8,
        "swiglu_limit": 10.0,
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 64,
        "compress_rope_theta": 160000,
        "compress_ratios": compress_ratios,
    }


def tiny_inference_config_dict(config: dict) -> dict:
    rope_scaling = config["rope_scaling"]
    return {
        "vocab_size": config["vocab_size"],
        "dim": config["hidden_size"],
        "moe_inter_dim": config["moe_intermediate_size"],
        "n_layers": config["num_hidden_layers"],
        "n_hash_layers": config["num_hash_layers"],
        "n_heads": config["num_attention_heads"],
        "n_routed_experts": config["n_routed_experts"],
        "n_shared_experts": config["n_shared_experts"],
        "n_activated_experts": config["num_experts_per_tok"],
        "score_func": config["scoring_func"],
        "route_scale": config["routed_scaling_factor"],
        "swiglu_limit": config["swiglu_limit"],
        "q_lora_rank": config["q_lora_rank"],
        "head_dim": config["head_dim"],
        "rope_head_dim": config["qk_rope_head_dim"],
        "o_groups": config["o_groups"],
        "o_lora_rank": config["o_lora_rank"],
        "window_size": config["sliding_window"],
        "original_seq_len": rope_scaling["original_max_position_embeddings"],
        "rope_theta": config["rope_theta"],
        "rope_factor": rope_scaling["factor"],
        "beta_fast": rope_scaling["beta_fast"],
        "beta_slow": rope_scaling["beta_slow"],
        "index_n_heads": config["index_n_heads"],
        "index_head_dim": config["index_head_dim"],
        "index_topk": config["index_topk"],
        "hc_mult": config["hc_mult"],
        "hc_sinkhorn_iters": config["hc_sinkhorn_iters"],
        "dtype": "fp8",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "compress_rope_theta": config["compress_rope_theta"],
        "compress_ratios": config["compress_ratios"],
    }


def _tiny_tensors(config: dict, generator: torch.Generator) -> dict[str, torch.Tensor]:
    hidden = config["hidden_size"]
    inter = config["moe_intermediate_size"]
    vocab = config["vocab_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    q_rank = config["q_lora_rank"]
    o_groups = config["o_groups"]
    o_rank = config["o_lora_rank"]
    hc_mult = config["hc_mult"]
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden
    tensors: dict[str, torch.Tensor] = {
        "embed.weight": _randn((vocab, hidden), generator),
        "head.weight": _randn((vocab, hidden), generator),
        "hc_head_fn": _randn((hc_mult, hc_dim), generator, dtype=torch.float32),
        "hc_head_base": torch.zeros(hc_mult, dtype=torch.float32),
        "hc_head_scale": torch.ones(1, dtype=torch.float32),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"layers.{layer}"
        tensors.update(
            {
                f"{prefix}.attn_norm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.ffn_norm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.attn.attn_sink": torch.zeros(heads, dtype=torch.float32),
                f"{prefix}.attn.q_norm.weight": torch.ones(q_rank, dtype=torch.float32),
                f"{prefix}.attn.kv_norm.weight": torch.ones(head_dim, dtype=torch.float32),
                f"{prefix}.attn.wq_a.weight": _randn((q_rank, hidden), generator),
                f"{prefix}.attn.wq_b.weight": _randn((heads * head_dim, q_rank), generator),
                f"{prefix}.attn.wq_b.scale": torch.ones((1, 1), dtype=torch.float32),
                f"{prefix}.attn.wkv.weight": _randn((head_dim, hidden), generator),
                f"{prefix}.attn.wkv.scale": torch.ones((1, 1), dtype=torch.float32),
                f"{prefix}.attn.wo_a.weight": _randn((o_groups * o_rank, heads * head_dim // o_groups), generator),
                f"{prefix}.attn.wo_a.scale": torch.ones((1, 1), dtype=torch.float32),
                f"{prefix}.attn.wo_b.weight": _randn((hidden, o_groups * o_rank), generator),
                f"{prefix}.attn.wo_b.scale": torch.ones((1, 1), dtype=torch.float32),
                f"{prefix}.ffn.gate.weight": _randn((config["n_routed_experts"], hidden), generator),
                f"{prefix}.ffn.shared_experts.w1.weight": _randn((inter, hidden), generator),
                f"{prefix}.ffn.shared_experts.w1.scale": torch.ones((1, 1), dtype=torch.float32),
                f"{prefix}.ffn.shared_experts.w2.weight": _randn((hidden, inter), generator),
                f"{prefix}.ffn.shared_experts.w2.scale": torch.ones((1, 1), dtype=torch.float32),
                f"{prefix}.ffn.shared_experts.w3.weight": _randn((inter, hidden), generator),
                f"{prefix}.ffn.shared_experts.w3.scale": torch.ones((1, 1), dtype=torch.float32),
                f"{prefix}.hc_attn_fn": _randn((mix_hc, hc_dim), generator, dtype=torch.float32),
                f"{prefix}.hc_attn_base": torch.zeros(mix_hc, dtype=torch.float32),
                f"{prefix}.hc_attn_scale": torch.ones(3, dtype=torch.float32),
                f"{prefix}.hc_ffn_fn": _randn((mix_hc, hc_dim), generator, dtype=torch.float32),
                f"{prefix}.hc_ffn_base": torch.zeros(mix_hc, dtype=torch.float32),
                f"{prefix}.hc_ffn_scale": torch.ones(3, dtype=torch.float32),
            }
        )
        if layer < config["num_hash_layers"]:
            tid2eid = torch.arange(vocab * config["num_experts_per_tok"], dtype=torch.int32).reshape(
                vocab, config["num_experts_per_tok"]
            )
            tensors[f"{prefix}.ffn.gate.tid2eid"] = tid2eid % config["n_routed_experts"]
        else:
            tensors[f"{prefix}.ffn.gate.bias"] = torch.zeros(config["n_routed_experts"], dtype=torch.float32)
        if config["compress_ratios"][layer] != 0:
            tensors.update(_compressor_tensors(prefix, config, generator))
        for expert in range(config["n_routed_experts"]):
            for projection, shape in (("w1", (inter, hidden)), ("w2", (hidden, inter)), ("w3", (inter, hidden))):
                packed, scale = _fp4_debug_weight(shape)
                tensors[f"{prefix}.ffn.experts.{expert}.{projection}.weight"] = packed
                tensors[f"{prefix}.ffn.experts.{expert}.{projection}.scale"] = scale
    return tensors


def _compressor_tensors(prefix: str, config: dict, generator: torch.Generator) -> dict[str, torch.Tensor]:
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    ratio = config["compress_ratios"][int(prefix.split(".")[1])]
    index_heads = config["index_n_heads"]
    index_head_dim = config["index_head_dim"]
    q_rank = config["q_lora_rank"]
    return {
        f"{prefix}.attn.compressor.ape": _randn((ratio, 2 * head_dim), generator, dtype=torch.float32),
        f"{prefix}.attn.compressor.norm.weight": torch.ones(head_dim, dtype=torch.float32),
        f"{prefix}.attn.compressor.wgate.weight": _randn((2 * head_dim, hidden), generator, dtype=torch.float32),
        f"{prefix}.attn.compressor.wkv.weight": _randn((2 * head_dim, hidden), generator, dtype=torch.float32),
        f"{prefix}.attn.indexer.compressor.ape": _randn((ratio, 2 * index_head_dim), generator, dtype=torch.float32),
        f"{prefix}.attn.indexer.compressor.norm.weight": torch.ones(index_head_dim, dtype=torch.float32),
        f"{prefix}.attn.indexer.compressor.wgate.weight": _randn(
            (2 * index_head_dim, hidden), generator, dtype=torch.float32
        ),
        f"{prefix}.attn.indexer.compressor.wkv.weight": _randn(
            (2 * index_head_dim, hidden), generator, dtype=torch.float32
        ),
        f"{prefix}.attn.indexer.weights_proj.weight": _randn((index_heads, hidden), generator),
        f"{prefix}.attn.indexer.wq_b.weight": _randn((index_heads * index_head_dim, q_rank), generator),
        f"{prefix}.attn.indexer.wq_b.scale": torch.ones((1, 1), dtype=torch.float32),
    }


def _fp4_debug_weight(shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if shape[-1] % EXPERT_FP4_BLOCK_SIZE != 0:
        raise ValueError(f"Synthetic FP4 shape must be block-aligned, got {shape}")
    indices = torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape) % 16
    packed = pack_fp4_indices(indices)
    scale = torch.ones((shape[0], shape[1] // EXPERT_FP4_BLOCK_SIZE), dtype=torch.float32)
    return packed, scale


def _randn(shape: tuple[int, ...], generator: torch.Generator, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
