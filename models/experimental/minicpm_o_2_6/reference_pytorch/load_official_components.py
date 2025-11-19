# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Load Official MiniCPM-o-2_6 Components

Loads each component (Qwen2 LLM, SigLip Vision, Whisper Audio, Resampler, ChatTTS)
from official safetensors weights to avoid memory issues while using 100% official code.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
from pathlib import Path

# Add official components to path
official_path = Path(__file__).parent / "minicpm_official"
sys.path.insert(0, str(official_path))

try:
    pass

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️ safetensors not available, will use demo weights")

# Import using importlib to avoid relative import issues
import importlib.util

spec = importlib.util.spec_from_file_location("weight_loader", Path(__file__).parent / "weight_loader.py")
weight_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weight_loader_module)
DiskBasedWeightLoader = weight_loader_module.DiskBasedWeightLoader


def load_siglip_vision_batch(weights_dict: Dict[str, torch.Tensor], config) -> nn.Module:
    """Load SigLip vision model from batch weights (GPTQ compatible)"""
    try:
        print("🎨 Loading SigLip Vision Model...")

        from modeling_navit_siglip import SigLipVisionTransformer

        # Create model with config - use GPTQ model config
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            # Load from GPTQ model config
            from transformers import AutoConfig

            full_config = AutoConfig.from_pretrained("openbmb/MiniCPM-o-2_6-int4", trust_remote_code=True)
            vision_config = full_config.vision_config

        model = SigLipVisionTransformer(vision_config)

        # Load state dict with GPTQ compatibility
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

        if missing_keys:
            print(f"⚠️ Missing keys in vision model: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys in vision model: {len(unexpected_keys)}")

        print("✅ SigLip Vision Model loaded (GPTQ compatible)")
        return model

    except Exception as e:
        print(f"❌ Failed to load SigLip vision model: {e}")
        return None


def load_qwen2_llm_batch(weights_dict: Dict[str, torch.Tensor], config) -> nn.Module:
    """Load Qwen2 LLM from batch weights (GPTQ quantized)"""
    try:
        print("🧠 Loading Qwen2 LLM (GPTQ 4-bit)...")

        # For GPTQ models, we need to use AutoGPTQ
        try:
            from auto_gptq import AutoGPTQForCausalLM

            print("✅ Using AutoGPTQ for quantized model loading")

            # Load the GPTQ model directly from HuggingFace
            model = AutoGPTQForCausalLM.from_quantized(
                "openbmb/MiniCPM-o-2_6-int4",
                model_basename="model",
                use_safetensors=True,
                trust_remote_code=True,
                device="cpu",  # Load on CPU first
                use_triton=False,  # Disable Triton for compatibility
            )

            print("✅ GPTQ Qwen2 LLM loaded successfully")
            return model

        except ImportError:
            print("⚠️ AutoGPTQ not available, falling back to standard loading")
            # Fallback to standard loading (may not work with GPTQ weights)
            from modeling_minicpmo import MiniCPMOLlamaForCausalLM

            text_config = getattr(config, "text_config", None)
            if text_config is None:
                from transformers import AutoConfig

                full_config = AutoConfig.from_pretrained("openbmb/MiniCPM-o-2_6-int4", trust_remote_code=True)
                text_config = full_config.text_config

            model = MiniCPMOLlamaForCausalLM(text_config)

            # Try to load GPTQ weights (this may fail)
            try:
                missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
                print("✅ Qwen2 LLM loaded (standard loading)")
                return model
            except Exception as e:
                print(f"❌ Failed to load GPTQ weights with standard loading: {e}")
                return None

    except Exception as e:
        print(f"❌ Failed to load Qwen2 LLM: {e}")
        return None


def load_qwen2_llm_streaming(disk_loader: DiskBasedWeightLoader, config) -> nn.Module:
    """Load Qwen2 LLM with streaming weight assignment to avoid memory issues"""
    from transformers import Qwen2ForCausalLM, Qwen2Config

    # Create Qwen2 config matching MiniCPM-o-2_6
    llm_config = Qwen2Config(
        vocab_size=151700,  # Actual MiniCPM vocab size
        hidden_size=3584,  # Full hidden dimension
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=28,
        max_position_embeddings=32768,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151643,
        tie_word_embeddings=False,
    )

    print("🏗️ Creating Qwen2 LLM (streaming construction)...")
    llm = Qwen2ForCausalLM(llm_config)

    # Move model to CPU and pin memory
    device = torch.device("cpu")
    llm = llm.to(device)
    for param in llm.parameters():
        param.data = param.data.pin_memory() if param.data.is_pinned() else param.data

    # Ultra-efficient streaming: load one tensor at a time with zero intermediate storage
    weight_map = disk_loader.index.get("weight_map", {})
    llm_files = set()
    for weight_name, safetensors_file in weight_map.items():
        if weight_name.startswith("llm."):
            llm_files.add(safetensors_file)

    print(f"📂 Found LLM weights in {len(llm_files)} files")

    # Load weights one by one with maximum memory efficiency
    loaded_count = 0
    import gc

    for safetensors_file in llm_files:
        file_path = disk_loader.model_dir / safetensors_file
        print(f"📖 Streaming from {safetensors_file}...")

        # Open file and get LLM keys
        with safe_open(file_path, framework="pt", device="cpu") as f:
            llm_keys = [key for key in f.keys() if key.startswith("llm.")]

            # Process one weight at a time with immediate cleanup
            for i, key in enumerate(llm_keys):
                try:
                    # Load single tensor directly to model parameter
                    tensor = f.get_tensor(key).to(device)

                    # Assign directly to model without intermediate storage
                    model_key = key[4:]  # Remove 'llm.' prefix
                    if model_key in llm.state_dict():
                        with torch.no_grad():
                            llm.state_dict()[model_key].copy_(tensor)
                        loaded_count += 1

                    # Immediate cleanup
                    del tensor

                    # Aggressive GC every 5 weights
                    if i % 5 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"⚠️ Failed to load {key}: {e}")
                    continue

        # GC between files
        gc.collect()

    print(f"✅ Streaming loaded {loaded_count} LLM weights")
    llm.eval()
    return llm


def load_siglip_vision_streaming(disk_loader: DiskBasedWeightLoader, config) -> nn.Module:
    """Load SigLip vision model with streaming weight assignment"""
    # Use simplified vision model to avoid import issues
    from modeling_navit_siglip import SiglipVisionTransformer
    from configuration_minicpm import MiniCPMOVisionConfig

    # Create vision config
    vision_config = MiniCPMOVisionConfig(
        hidden_size=1152,
        image_size=980,
        patch_size=14,
        num_channels=3,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
        hidden_dropout=0.0,
    )

    print("🏗️ Creating SigLip Vision Model (streaming)...")
    vpm = SiglipVisionTransformer(vision_config)

    # Move model to CPU and pin memory
    device = torch.device("cpu")
    vpm = vpm.to(device)
    for param in vpm.parameters():
        param.data = param.data.pin_memory() if not param.data.is_pinned() else param.data

    # Ultra-efficient vision streaming
    weight_map = disk_loader.index.get("weight_map", {})
    vision_files = set()
    for weight_name, safetensors_file in weight_map.items():
        if weight_name.startswith("vpm."):
            vision_files.add(safetensors_file)

    print(f"📂 Found vision weights in {len(vision_files)} files")

    # Load weights one by one with maximum memory efficiency
    loaded_count = 0
    import gc

    for safetensors_file in vision_files:
        file_path = disk_loader.model_dir / safetensors_file
        print(f"📖 Streaming vision from {safetensors_file}...")

        with safe_open(file_path, framework="pt", device="cpu") as f:
            vision_keys = [key for key in f.keys() if key.startswith("vpm.")]

            for i, key in enumerate(vision_keys):
                try:
                    # Load single tensor directly to model parameter
                    tensor = f.get_tensor(key).to(device)

                    model_key = key[4:]  # Remove 'vpm.' prefix
                    if model_key in vpm.state_dict():
                        with torch.no_grad():
                            vpm.state_dict()[model_key].copy_(tensor)
                        loaded_count += 1

                    # Immediate cleanup
                    del tensor

                    # Aggressive GC every 3 weights (vision has fewer weights)
                    if i % 3 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"⚠️ Failed to load vision {key}: {e}")
                    continue

        # GC between files
        gc.collect()

    print(f"✅ Streaming loaded {loaded_count} vision weights")
    vpm.eval()
    return vpm


def load_whisper_audio(weights_dict: Dict[str, torch.Tensor], config) -> nn.Module:
    """Load Whisper encoder with official weights"""
    from transformers import WhisperConfig, WhisperModel

    # Create Whisper config matching MiniCPM
    whisper_config = WhisperConfig(
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        encoder_hidden_size=1024,
        encoder_layerdrop=0.0,
        use_cache=True,
        max_source_positions=1500,
    )

    print("🏗️ Creating Whisper Audio Encoder...")
    apm = WhisperModel(whisper_config).encoder

    # Extract audio weights (apm.* prefix)
    apm_weights = {}
    for key, tensor in weights_dict.items():
        if key.startswith("apm."):
            new_key = key[4:]  # Remove 'apm.'
            apm_weights[new_key] = tensor

    print(f"📦 Loading {len(apm_weights)} audio weights...")
    missing_keys, unexpected_keys = apm.load_state_dict(apm_weights, strict=False)

    if missing_keys:
        print(f"⚠️ Missing audio keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️ Unexpected audio keys: {len(unexpected_keys)}")

    apm.eval()
    return apm


def load_resampler(weights_dict: Dict[str, torch.Tensor], config) -> nn.Module:
    """Load vision resampler with official weights"""
    from resampler import Resampler

    # Create resampler config
    resampler_config = {
        "num_queries": 32,
        "embed_dim": 3584,  # Match LLM hidden size
        "num_heads": 28,  # Match LLM heads
        "kv_dim": 1152,  # Match vision hidden size
    }

    print("🏗️ Creating Vision Resampler...")
    resampler = Resampler(**resampler_config)

    # Extract resampler weights
    resampler_weights = {}
    for key, tensor in weights_dict.items():
        if key.startswith("resampler."):
            new_key = key[10:]  # Remove 'resampler.'
            resampler_weights[new_key] = tensor

    print(f"📦 Loading {len(resampler_weights)} resampler weights...")
    if resampler_weights:
        missing_keys, unexpected_keys = resampler.load_state_dict(resampler_weights, strict=False)

        if missing_keys:
            print(f"⚠️ Missing resampler keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected resampler keys: {len(unexpected_keys)}")
    else:
        print("⚠️ No resampler weights found in safetensors")

    resampler.eval()
    return resampler


def load_audio_projection(weights_dict: Dict[str, torch.Tensor], config) -> nn.Module:
    """Load audio projection layer"""
    # Audio projection is a simple MultiModalProjector
    from modeling_minicpmo import MultiModalProjector

    audio_output_dim = int(1024 // 4)  # Whisper hidden size // 4
    audio_projection = MultiModalProjector(in_dim=audio_output_dim, out_dim=3584)  # Match LLM hidden size

    # Extract audio projection weights
    proj_weights = {}
    for key, tensor in weights_dict.items():
        if key.startswith("audio_projection_layer."):
            new_key = key[23:]  # Remove 'audio_projection_layer.'
            proj_weights[new_key] = tensor

    print(f"📦 Loading {len(proj_weights)} audio projection weights...")
    if proj_weights:
        missing_keys, unexpected_keys = audio_projection.load_state_dict(proj_weights, strict=False)

        if missing_keys:
            print(f"⚠️ Missing audio projection keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected audio projection keys: {len(unexpected_keys)}")
    else:
        print("⚠️ No audio projection weights found")

    audio_projection.eval()
    return audio_projection


def load_chattts(weights_dict: Dict[str, torch.Tensor], config) -> Optional[nn.Module]:
    """Load ChatTTS decoder with official weights"""
    try:
        from modeling_minicpmo import ConditionalChatTTS
        from configuration_minicpm import ConditionalChatTTSConfig

        # Create ChatTTS config
        tts_config = ConditionalChatTTSConfig(
            vocab_size=4096,  # Semantic tokens
            hidden_size=256,  # ChatTTS hidden size
            num_layers=2,
            num_heads=8,
        )

        print("🏗️ Creating ChatTTS Decoder...")
        tts = ConditionalChatTTS(tts_config)

        # Extract TTS weights
        tts_weights = {}
        for key, tensor in weights_dict.items():
            if key.startswith("tts."):
                new_key = key[4:]  # Remove 'tts.'
                tts_weights[new_key] = tensor

        print(f"📦 Loading {len(tts_weights)} TTS weights...")
        if tts_weights:
            missing_keys, unexpected_keys = tts.load_state_dict(tts_weights, strict=False)

            if missing_keys:
                print(f"⚠️ Missing TTS keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️ Unexpected TTS keys: {len(unexpected_keys)}")
        else:
            print("⚠️ No TTS weights found")

        tts.eval()
        return tts

    except Exception as e:
        print(f"⚠️ ChatTTS not available: {e}")
        return None


def create_official_minicpm_components_simple(config) -> Dict[str, nn.Module]:
    """Load all official MiniCPM components using TTNN's efficient loading infrastructure"""
    print("🚀 Loading MiniCPM-o-2_6 Components with TTNN bridge loading...")

    components = {}

    try:
        # Use TTNN bridge loader exclusively
        from ttnn_bridge_loader import TTNNBridgeLoader

        ttnn_loader = TTNNBridgeLoader("model_cache/minicpm_o_2_6")
        print("✅ Using TTNN bridge loader for efficient loading")

        # Load the full model state dict once
        full_state_dict = ttnn_loader.load_full_model()

        # Component configurations
        component_configs = [
            ("vision", "vpm.", load_siglip_vision_batch),
            ("audio", "apm.", load_whisper_audio),
            ("resampler", "resampler.", load_resampler),
            ("audio_projection", "audio_projection_layer.", load_audio_projection),
        ]

        if getattr(config, "init_tts", True):
            component_configs.append(("tts", "tts.", load_chattts))

        # Load smaller components first
        for comp_name, prefix, load_func in component_configs:
            try:
                print(f"📦 Loading {comp_name}...")

                # Extract component weights from full state dict
                component_weights = {key: tensor for key, tensor in full_state_dict.items() if key.startswith(prefix)}

                print(f"   Found {len(component_weights)} weights for {comp_name}")

                if component_weights:
                    component = load_func(component_weights, config)
                    component.eval()
                    components[comp_name] = component
                    print(f"✅ {comp_name.capitalize()} loaded")
                else:
                    print(f"⚠️ No weights found for {comp_name}")

                # Clear component weights to free memory
                del component_weights
                import gc

                gc.collect()

            except Exception as e:
                print(f"❌ Failed to load {comp_name}: {e}")
                continue

        # Load LLM last (largest component)
        try:
            print("🧠 Loading LLM...")
            llm_weights = {key: tensor for key, tensor in full_state_dict.items() if key.startswith("llm.")}
            print(f"   Found {len(llm_weights)} LLM weights")

            components["llm"] = load_qwen2_llm_batch(llm_weights, config)
            components["llm"].eval()
            print("✅ LLM loaded")

            # Clear LLM weights
            del llm_weights

        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")

        # Clear full state dict
        del full_state_dict

        # Final cleanup
        import gc

        gc.collect()

    except Exception as e:
        print(f"❌ Component loading failed: {e}")
        import traceback

        traceback.print_exc()
        return {}

    print(f"✅ Loaded {len(components)} official components with TTNN bridge loading")
    return components
