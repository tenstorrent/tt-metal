"""
Test script for model_configs.py to verify the Pydantic models work correctly.

This script tests the parsing and standardization of different model configuration formats.
"""

import json
from model_configs import parse_model_config_from_dict, StandardModelConfig


def test_meta_llama_config():
    """Test Meta format Llama configuration."""
    meta_config = {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim_multiplier": 1.3,
        "multiple_of": 1024,
        "norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "use_scaled_rope": True,
        "rope_scaling_factor": 8
    }
    
    standard_config = parse_model_config_from_dict(meta_config)
    
    print("Meta Llama Config Test:")
    print(f"  Architecture: {standard_config.architecture}")
    print(f"  Dimensions: {standard_config.dim}")
    print(f"  Layers: {standard_config.n_layers}")
    print(f"  Heads: {standard_config.n_heads}")
    print(f"  KV Heads: {standard_config.n_kv_heads}")
    print(f"  Vocab Size: {standard_config.vocab_size}")
    print(f"  RoPE Scaling Factor: {standard_config.rope_scaling_factor}")
    print()


def test_hf_llama_config():
    """Test HuggingFace format Llama configuration."""
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": [128001, 128008, 128009],
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 131072,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.42.3",
        "use_cache": True,
        "vocab_size": 128256
    }
    
    standard_config = parse_model_config_from_dict(hf_config)
    
    print("HuggingFace Llama Config Test:")
    print(f"  Architecture: {standard_config.architecture}")
    print(f"  Dimensions: {standard_config.dim}")
    print(f"  Layers: {standard_config.n_layers}")
    print(f"  Heads: {standard_config.n_heads}")
    print(f"  KV Heads: {standard_config.n_kv_heads}")
    print(f"  Vocab Size: {standard_config.vocab_size}")
    print(f"  Max Position Embeddings: {standard_config.max_position_embeddings}")
    print(f"  RoPE Scaling: {standard_config.rope_scaling}")
    print()


def test_qwen_config():
    """Test Qwen2.5 configuration."""
    qwen_config = {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 29568,
        "max_position_embeddings": 32768,
        "max_window_layers": 70,
        "model_type": "qwen2",
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 131072,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.1",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 152064
    }
    
    standard_config = parse_model_config_from_dict(qwen_config)
    
    print("Qwen2.5 Config Test:")
    print(f"  Architecture: {standard_config.architecture}")
    print(f"  Dimensions: {standard_config.dim}")
    print(f"  Layers: {standard_config.n_layers}")
    print(f"  Heads: {standard_config.n_heads}")
    print(f"  KV Heads: {standard_config.n_kv_heads}")
    print(f"  Vocab Size: {standard_config.vocab_size}")
    print(f"  Max Position Embeddings: {standard_config.max_position_embeddings}")
    print()


def test_deepseek_config():
    """Test DeepSeek V3 configuration."""
    deepseek_config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
            "AutoModel": "modeling_deepseek.DeepseekV3Model",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
        },
        "bos_token_id": 0,
        "eos_token_id": 1,
        "ep_size": 1,
        "first_k_dense_replace": 3,
        "hidden_act": "silu",
        "hidden_size": 7168,
        "initializer_range": 0.02,
        "intermediate_size": 18432,
        "kv_lora_rank": 512,
        "max_position_embeddings": 163840,
        "model_type": "deepseek_v3",
        "moe_intermediate_size": 2048,
        "moe_layer_freq": 1,
        "n_group": 8,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "num_nextn_predict_layers": 1,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        },
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        },
        "rope_theta": 10000,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 4,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.33.1",
        "use_cache": True,
        "v_head_dim": 128,
        "vocab_size": 129280
    }
    
    standard_config = parse_model_config_from_dict(deepseek_config)
    
    print("DeepSeek V3 Config Test:")
    print(f"  Architecture: {standard_config.architecture}")
    print(f"  Dimensions: {standard_config.dim}")
    print(f"  Layers: {standard_config.n_layers}")
    print(f"  Heads: {standard_config.n_heads}")
    print(f"  KV Heads: {standard_config.n_kv_heads}")
    print(f"  Vocab Size: {standard_config.vocab_size}")
    print(f"  Max Position Embeddings: {standard_config.max_position_embeddings}")
    print(f"  MoE Experts: {standard_config.n_routed_experts}")
    print(f"  Experts per Token: {standard_config.num_experts_per_tok}")
    print(f"  RoPE Scaling: {standard_config.rope_scaling}")
    print()


if __name__ == "__main__":
    print("Testing Model Configuration Parsers\n")
    print("=" * 50)
    
    test_meta_llama_config()
    test_hf_llama_config()
    test_qwen_config()
    test_deepseek_config()
    
    print("All tests completed successfully!") 