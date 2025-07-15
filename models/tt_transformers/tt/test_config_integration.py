#!/usr/bin/env python3
"""
Simple test to verify that the configuration parsing integration works correctly.
"""

import json
import tempfile
import os
import sys

# Mock the logger
class MockLogger:
    def info(self, msg):
        print(f"INFO: {msg}")
    def warning(self, msg):
        print(f"WARNING: {msg}")
    def error(self, msg):
        print(f"ERROR: {msg}")

sys.modules['loguru'] = type('MockModule', (), {'logger': MockLogger()})()

# Mock torch 
sys.modules['torch'] = type('MockModule', (), {'is_tensor': lambda x: False})()

# Test the configuration parsing methods directly
def test_configuration_parsing():
    """Test just the configuration parsing methods."""
    
    # Create a minimal mock for other dependencies
    class MockNearestMultiple:
        def __call__(self, x, multiple):
            return ((x + multiple - 1) // multiple) * multiple
    
    sys.modules['models.tt_transformers.tt.common'] = type('MockModule', (), {
        'nearest_multiple': MockNearestMultiple(),
        'get_base_model_name': lambda name: name.split('-')[0] + '-' + name.split('-')[1] if '-' in name else name,
        'calculate_hidden_dim': lambda dim, mult, multiple: int(dim * mult) if mult else dim,
    })()
    
    # Import what we need after setting up mocks
    from model_config import ModelArgs
    
    # Create a test config
    test_config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "rms_norm_eps": 1e-05,
        "vocab_size": 128256,
        "intermediate_size": 14336,
        "max_position_embeddings": 131072,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        "model_type": "llama",
        "_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"
    }
    
    # Create a mock ModelArgs instance to test the config parsing methods
    class TestModelArgs:
        def __init__(self):
            self.num_devices = 0  # CPU mode
            self.is_galaxy = False
            self.tile_size = 32
            
        def _get_text_prefix(self):
            return ""
            
        def _set_model_specific_params(self):
            pass
    
    # Create test instance
    test_instance = TestModelArgs()
    
    # Copy the methods we want to test
    test_instance._set_params_from_dict = ModelArgs._set_params_from_dict.__get__(test_instance, TestModelArgs)
    test_instance._set_params_from_standard_config = ModelArgs._set_params_from_standard_config.__get__(test_instance, TestModelArgs)
    test_instance._set_params_from_dict_legacy = ModelArgs._set_params_from_dict_legacy.__get__(test_instance, TestModelArgs)
    
    # Add the base_model_name property
    def get_base_model_name(self):
        if hasattr(self, 'model_name'):
            return self.model_name.split('-')[0] + '-' + self.model_name.split('-')[1] if '-' in self.model_name else self.model_name
        return "Unknown"
    
    test_instance.base_model_name = property(get_base_model_name)
    
    print("Testing Configuration Parsing Integration")
    print("=" * 50)
    
    try:
        # Test the main parsing method
        test_instance._set_params_from_dict(test_config, is_hf=True)
        
        print("✅ Configuration parsing completed successfully!")
        print(f"  Model Name: {getattr(test_instance, 'model_name', 'Not set')}")
        print(f"  Dimensions: {getattr(test_instance, 'dim', 'Not set')}")
        print(f"  Layers: {getattr(test_instance, 'n_layers', 'Not set')}")
        print(f"  Attention Heads: {getattr(test_instance, 'n_heads', 'Not set')}")
        print(f"  KV Heads: {getattr(test_instance, 'n_kv_heads', 'Not set')}")
        print(f"  Vocab Size: {getattr(test_instance, 'vocab_size', 'Not set')}")
        print(f"  Hidden Dimension: {getattr(test_instance, 'hidden_dim', 'Not set')}")
        print(f"  RoPE Theta: {getattr(test_instance, 'rope_theta', 'Not set')}")
        print(f"  RoPE Scaling Factor: {getattr(test_instance, 'rope_scaling_factor', 'Not set')}")
        print(f"  Original Context Length: {getattr(test_instance, 'orig_context_len', 'Not set')}")
        
        # Verify that the values are correct
        assert test_instance.dim == 4096, f"Expected dim=4096, got {test_instance.dim}"
        assert test_instance.n_layers == 32, f"Expected n_layers=32, got {test_instance.n_layers}"
        assert test_instance.n_heads == 32, f"Expected n_heads=32, got {test_instance.n_heads}"
        assert test_instance.n_kv_heads == 8, f"Expected n_kv_heads=8, got {test_instance.n_kv_heads}"
        assert test_instance.vocab_size == 128256, f"Expected vocab_size=128256, got {test_instance.vocab_size}"
        assert test_instance.hidden_dim == 14336, f"Expected hidden_dim=14336, got {test_instance.hidden_dim}"
        assert test_instance.rope_theta == 500000.0, f"Expected rope_theta=500000.0, got {test_instance.rope_theta}"
        assert test_instance.rope_scaling_factor == 8.0, f"Expected rope_scaling_factor=8.0, got {test_instance.rope_scaling_factor}"
        assert test_instance.orig_context_len == 8192, f"Expected orig_context_len=8192, got {test_instance.orig_context_len}"
        
        print("✅ All assertions passed!")
        
        # Test with a Qwen config to make sure it works for different model types
        print("\nTesting with Qwen2.5 config...")
        
        qwen_config = {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_hidden_layers": 80,
            "rms_norm_eps": 1e-06,
            "vocab_size": 152064,
            "intermediate_size": 29568,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "model_type": "qwen2",
            "_name_or_path": "Qwen/Qwen2.5-72B-Instruct"
        }
        
        test_instance2 = TestModelArgs()
        test_instance2._set_params_from_dict = ModelArgs._set_params_from_dict.__get__(test_instance2, TestModelArgs)
        test_instance2._set_params_from_standard_config = ModelArgs._set_params_from_standard_config.__get__(test_instance2, TestModelArgs)
        test_instance2._set_params_from_dict_legacy = ModelArgs._set_params_from_dict_legacy.__get__(test_instance2, TestModelArgs)
        test_instance2.base_model_name = property(get_base_model_name)
        
        test_instance2._set_params_from_dict(qwen_config, is_hf=True)
        
        print(f"  Qwen Dimensions: {test_instance2.dim}")
        print(f"  Qwen Layers: {test_instance2.n_layers}")
        print(f"  Qwen Vocab Size: {test_instance2.vocab_size}")
        
        assert test_instance2.dim == 8192, f"Expected Qwen dim=8192, got {test_instance2.dim}"
        assert test_instance2.n_layers == 80, f"Expected Qwen n_layers=80, got {test_instance2.n_layers}"
        assert test_instance2.vocab_size == 152064, f"Expected Qwen vocab_size=152064, got {test_instance2.vocab_size}"
        
        print("✅ Qwen configuration parsed correctly!")
        print("\n🎉 All integration tests passed! The new configuration system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_configuration_parsing()
    if not success:
        sys.exit(1) 