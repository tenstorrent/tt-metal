import ttnn

def create_model_config(batch_size, seq_len):
    return {
        "hidden_size": 2048,
        "num_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 12288,
        "num_hidden_layers": 16,
        "layer_types": [
            "conv", "conv", "full_attention", 
            "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "full_attention",
            "conv", "full_attention",
            "conv", "full_attention",
            "conv"
        ],
        "vocab_size": 65536,
        "norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "vision_config": {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "patch_size": 16,
            "num_patches": 256,
        },
        "projector_hidden_size": 2048,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }