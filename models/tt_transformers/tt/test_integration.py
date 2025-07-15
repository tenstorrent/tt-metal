#!/usr/bin/env python3
"""
Test script to verify the integration between ModelArgs and the new Pydantic configuration system.
"""

import json
import os
import tempfile

# Mock the ttnn and other dependencies that ModelArgs expects
class MockTTNN:
    @property 
    def get_arch_name(self):
        return lambda: "wormhole_b0"
    
    @property
    def DRAM_MEMORY_CONFIG(self):
        return "DRAM_CONFIG"
    
    @property 
    def L1_MEMORY_CONFIG(self):
        return "L1_CONFIG"
    
    @property
    def TILE_LAYOUT(self):
        return "TILE"
    
    class UnaryOpType:
        GELU = "gelu"

import sys
sys.modules['ttnn'] = MockTTNN()

# Mock other dependencies
class MockTransformer:
    def __init__(self, args):
        pass
    def state_dict(self):
        return {}

sys.modules['models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model'] = type('MockModule', (), {
    'Transformer': MockTransformer
})()

sys.modules['models.tt_transformers.tt.common'] = type('MockModule', (), {
    'calculate_hidden_dim': lambda dim, mult, multiple: int(dim * mult),
    'encode_prompt_hf': lambda *args: [],
    'encode_prompt_instruct': lambda *args: [],
    'get_base_model_name': lambda name: name.split('-')[0] + '-' + name.split('-')[1] if '-' in name else name,
    'get_out_subblock_w': lambda *args: 1,
    'nearest_multiple': lambda x, multiple: ((x + multiple - 1) // multiple) * multiple,
    'num_to_core_range_set': lambda x: x,
})()

sys.modules['models.tt_transformers.tt.load_checkpoints'] = type('MockModule', (), {
    'convert_hf_to_meta': lambda x, y: x,
    'convert_meta_to_hf': lambda x, y: x,
    'load_hf_state_dict': lambda x: {},
    'load_meta_state_dict': lambda x, y: {},
    'reverse_permute': lambda x, y, z, w: x,
    'standardize_hf_keys': lambda x: x,
})()

sys.modules['models.utility_functions'] = type('MockModule', (), {
    'is_blackhole': lambda: False,
    'is_wormhole_b0': lambda: True,
    'nearest_32': lambda x: ((x + 31) // 32) * 32,
})()

sys.modules['loguru'] = type('MockModule', (), {
    'logger': type('MockLogger', (), {
        'info': lambda msg: print(f"INFO: {msg}"),
        'warning': lambda msg: print(f"WARNING: {msg}"),
        'error': lambda msg: print(f"ERROR: {msg}"),
    })()
})()

sys.modules['torch'] = type('MockModule', (), {
    'is_tensor': lambda x: False
})()

# Now we can import ModelArgs
from model_config import ModelArgs

def test_model_args_with_hf_config():
    """Test ModelArgs with a HuggingFace Llama configuration."""
    
    # Create test config directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        
        # HuggingFace Llama config
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
            "vocab_size": 128256,
            "_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"
        }
        
        # Write config file
        with open(config_path, 'w') as f:
            json.dump(hf_config, f)
        
        # Set environment variables to point to our test config
        os.environ['HF_MODEL'] = temp_dir
        
        try:
            # Create ModelArgs instance (this should use the new Pydantic config system)
            model_args = ModelArgs(
                mesh_device=None,  # Mock device
                dummy_weights=True,
                max_batch_size=1,
                max_seq_len=2048
            )
            
            # Verify that the configuration was parsed correctly
            print("✅ ModelArgs Integration Test Results:")
            print(f"  Model Name: {model_args.model_name}")
            print(f"  Dimensions: {model_args.dim}")
            print(f"  Layers: {model_args.n_layers}")
            print(f"  Attention Heads: {model_args.n_heads}")
            print(f"  KV Heads: {model_args.n_kv_heads}")
            print(f"  Vocab Size: {model_args.vocab_size}")
            print(f"  Max Context Length: {model_args.max_context_len}")
            print(f"  Hidden Dimension: {model_args.hidden_dim}")
            print(f"  RoPE Theta: {model_args.rope_theta}")
            print(f"  RoPE Scaling Factor: {model_args.rope_scaling_factor}")
            print(f"  Original Context Length: {model_args.orig_context_len}")
            
            # Verify expected values
            assert model_args.dim == 4096, f"Expected dim=4096, got {model_args.dim}"
            assert model_args.n_layers == 32, f"Expected n_layers=32, got {model_args.n_layers}"
            assert model_args.n_heads == 32, f"Expected n_heads=32, got {model_args.n_heads}"
            assert model_args.n_kv_heads == 8, f"Expected n_kv_heads=8, got {model_args.n_kv_heads}"
            assert model_args.vocab_size == 128256, f"Expected vocab_size=128256, got {model_args.vocab_size}"
            assert model_args.hidden_dim == 14336, f"Expected hidden_dim=14336, got {model_args.hidden_dim}"
            assert model_args.rope_theta == 500000.0, f"Expected rope_theta=500000.0, got {model_args.rope_theta}"
            assert model_args.rope_scaling_factor == 8.0, f"Expected rope_scaling_factor=8.0, got {model_args.rope_scaling_factor}"
            assert model_args.orig_context_len == 8192, f"Expected orig_context_len=8192, got {model_args.orig_context_len}"
            
            print("✅ All assertions passed! New configuration system is working correctly.")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Clean up environment
            if 'HF_MODEL' in os.environ:
                del os.environ['HF_MODEL']
    
    return True

def test_model_args_with_qwen_config():
    """Test ModelArgs with a Qwen2.5 configuration."""
    
    # Create test config directory  
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        
        # Qwen2.5 config
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
            "vocab_size": 152064,
            "_name_or_path": "Qwen/Qwen2.5-72B-Instruct"
        }
        
        # Write config file
        with open(config_path, 'w') as f:
            json.dump(qwen_config, f)
        
        # Set environment variables
        os.environ['HF_MODEL'] = temp_dir
        
        try:
            # Create ModelArgs instance
            model_args = ModelArgs(
                mesh_device=None,  # Mock device
                dummy_weights=True,
                max_batch_size=1,
                max_seq_len=2048
            )
            
            print("✅ Qwen ModelArgs Integration Test Results:")
            print(f"  Model Name: {model_args.model_name}")
            print(f"  Dimensions: {model_args.dim}")
            print(f"  Layers: {model_args.n_layers}")
            print(f"  Vocab Size: {model_args.vocab_size}")
            print(f"  Hidden Dimension: {model_args.hidden_dim}")
            
            # Verify Qwen-specific values
            assert model_args.dim == 8192, f"Expected dim=8192, got {model_args.dim}"
            assert model_args.n_layers == 80, f"Expected n_layers=80, got {model_args.n_layers}"
            assert model_args.vocab_size == 152064, f"Expected vocab_size=152064, got {model_args.vocab_size}"
            assert model_args.hidden_dim == 29568, f"Expected hidden_dim=29568, got {model_args.hidden_dim}"
            
            print("✅ Qwen configuration parsed correctly!")
            
        except Exception as e:
            print(f"❌ Qwen test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Clean up environment
            if 'HF_MODEL' in os.environ:
                del os.environ['HF_MODEL']
    
    return True

if __name__ == "__main__":
    print("Testing ModelArgs Integration with Pydantic Configuration System")
    print("=" * 70)
    
    success1 = test_model_args_with_hf_config()
    print()
    success2 = test_model_args_with_qwen_config()
    
    if success1 and success2:
        print("\n🎉 All integration tests passed! The new configuration system is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.") 