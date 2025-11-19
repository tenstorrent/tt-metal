# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MiniCPM-o Official Wrapper

Assembles official MiniCPM-o-2_6 components (Qwen2, SigLip, Whisper, Resampler, ChatTTS)
into a unified interface that provides the chat() method for multimodal conversations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Union
from PIL import Image
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

# Import components using importlib to avoid relative import issues
import importlib.util

# Import load_official_components
spec = importlib.util.spec_from_file_location(
    "load_official_components", Path(__file__).parent / "load_official_components.py"
)
load_official_components = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_official_components)

# Import weight_loader
spec2 = importlib.util.spec_from_file_location("weight_loader", Path(__file__).parent / "weight_loader.py")
weight_loader = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(weight_loader)

create_official_minicpm_components_simple = load_official_components.create_official_minicpm_components_simple
DiskBasedWeightLoader = weight_loader.DiskBasedWeightLoader


class MiniCPMOConfig:
    """Configuration for MiniCPM-o components"""

    def __init__(
        self, model_path: str = None, init_vision: bool = True, init_audio: bool = True, init_tts: bool = True
    ):
        self.model_path = model_path or "model_cache/minicpm_o_2_6_int4"
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts

        # Component configurations
        self.llm_config = {
            "vocab_size": 151700,
            "hidden_size": 3584,
            "num_layers": 28,
            "num_heads": 28,
        }

        self.vision_config = {
            "hidden_size": 1152,
            "image_size": 980,
            "patch_size": 14,
            "num_layers": 27,
            "num_heads": 16,
        }

        self.audio_config = {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
        }

        self.tts_config = {
            "vocab_size": 4096,
            "hidden_size": 256,
            "num_layers": 2,
            "num_heads": 8,
        }


class MiniCPMOWrapper(nn.Module):
    """
    Wrapper around official MiniCPM-o-2_6 components

    Provides the chat() interface that matches the official HuggingFace API
    while loading components individually to avoid memory issues.
    """

    def __init__(self, config: MiniCPMOConfig):
        super().__init__()
        self.config = config

        # Special token IDs (matching official implementation)
        self.image_token_id = 151646  # <image> token
        self.audio_token_id = 151647  # <audio> token

        # Load components
        self.components = self._load_components()

        # Initialize Vocos for TTS if available
        self.vocos = None
        if "tts" in self.components:
            self._init_vocos()

    def _load_components(self) -> Dict[str, nn.Module]:
        """Load official MiniCPM components from disk (lazy loading)"""
        print("🚀 Loading MiniCPM-o-2_6 Components from disk...")

        # Create disk-based weight loader (already imported at module level)
        disk_loader = DiskBasedWeightLoader()

        # Load official components with simple batch loading (works for large models)
        components = create_official_minicpm_components_simple(self.config)

        print(f"✅ Loaded {len(components)} components: {list(components.keys())}")
        return components

    def _init_vocos(self):
        """Initialize Vocos vocoder for TTS"""
        try:
            from huggingface_hub import hf_hub_download
            from vocos import Vocos

            # Download Vocos checkpoint
            vocos_path = hf_hub_download(repo_id="openbmb/MiniCPM-o-2_6", subfolder="assets", filename="Vocos.pt")

            # Initialize Vocos
            from vocos.feature_extractors import MelSpectrogramFeatures
            from vocos.models import VocosBackbone
            from vocos.heads import ISTFTHead

            feature_extractor = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100)
            backbone = VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8)
            head = ISTFTHead(dim=512, n_fft=1024, hop_length=256)

            self.vocos = Vocos(feature_extractor, backbone, head)
            self.vocos.load_state_dict(torch.load(vocos_path, weights_only=True, mmap=True))
            self.vocos.eval()

            print("✅ Vocos vocoder initialized")

        except Exception as e:
            print(f"⚠️ Failed to initialize Vocos: {e}")
            self.vocos = None

    def chat(
        self,
        msgs: List[Dict[str, Any]],
        tokenizer=None,
        max_new_tokens: int = 128,
        sampling: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        generate_audio: bool = False,
        output_audio_path: Optional[str] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Multimodal chat interface matching official MiniCPM-o API

        Args:
            msgs: List of message dicts with multimodal content
                  [{'role': 'user', 'content': [image, audio, "question"]}]
            tokenizer: Qwen2 tokenizer (if None, will try to load)
            max_new_tokens: Maximum tokens to generate
            sampling: Whether to use sampling vs greedy
            temperature: Sampling temperature
            top_k: Top-k for sampling
            top_p: Top-p (nucleus) for sampling
            generate_audio: Whether to generate speech output
            output_audio_path: Path to save generated audio

        Returns:
            str: Text response if generate_audio=False
            dict: {'text': str, 'audio_path': str} if generate_audio=True
        """
        if not self.components:
            return "❌ No model components loaded"

        # Get tokenizer if not provided
        if tokenizer is None:
            tokenizer = self._get_tokenizer()

        # Process multimodal inputs
        processed_inputs = self._process_multimodal_inputs(msgs, tokenizer)

        # Generate text response
        text_response = self._generate_text_response(
            processed_inputs, tokenizer, max_new_tokens, sampling, temperature, top_k, top_p
        )

        # Generate audio if requested
        if generate_audio and "tts" in self.components and self.vocos:
            audio_path = self._generate_audio(text_response, output_audio_path)
            return {"text": text_response, "audio_path": audio_path}

        return text_response

    def _get_tokenizer(self):
        """Get Qwen2 tokenizer"""
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)
            return tokenizer
        except Exception as e:
            print(f"⚠️ Could not load tokenizer: {e}")
            return None

    def _process_multimodal_inputs(self, msgs: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:
        """
        Process multimodal inputs into format ready for generation

        Args:
            msgs: [{'role': 'user', 'content': [image, audio, text]}]
            tokenizer: Text tokenizer

        Returns:
            Dict with processed inputs for generation
        """
        if not msgs:
            return {}

        # Get the user message (assume single-turn for now)
        user_msg = msgs[0]
        content = user_msg.get("content", [])

        # Extract different modalities
        text_parts = []
        image_tensor = None
        audio_tensor = None

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, Image.Image):
                # Process image
                image_tensor = self._process_image(item)
            elif isinstance(item, np.ndarray):
                # Process audio
                audio_tensor = self._process_audio(item)

        # Combine text parts
        text_input = " ".join(text_parts) if text_parts else "Describe what you see and hear."

        # Tokenize text with special tokens
        tokens = self._tokenize_with_special_tokens(text_input, tokenizer, image_tensor, audio_tensor)

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "image_embeds": tokens.get("image_embeds"),
            "audio_embeds": tokens.get("audio_embeds"),
        }

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Process PIL image into tensor for vision model"""
        if "vision" not in self.components:
            return None

        # Convert PIL to tensor (simplified preprocessing)
        # In official implementation, this uses the processor
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((980, 980)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Get vision embeddings
        with torch.no_grad():
            vision_embeds = self.components["vision"](image_tensor)

        # Resample to LLM space
        if "resampler" in self.components:
            vision_embeds = self.components["resampler"](vision_embeds)

        return vision_embeds

    def _process_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Process numpy audio into tensor for audio model"""
        if "audio" not in self.components or "audio_projection" not in self.components:
            return None

        # Convert numpy array to mel spectrograms
        # Simplified - official uses proper mel spectrogram extraction
        import librosa

        # Ensure audio is 16kHz mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)  # Convert to mono

        # Resample to 16kHz if needed
        if len(audio) > 0:
            target_sr = 16000
            audio = librosa.resample(audio, orig_sr=44100, target_sr=target_sr)

        # Create mel spectrogram (simplified)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80, n_fft=1024, hop_length=256)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Convert to tensor and add batch/channel dimensions
        audio_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)  # [1, 80, time]

        # Get audio embeddings
        with torch.no_grad():
            audio_embeds = self.components["audio"](audio_tensor)

        # Average pool and project to LLM space
        audio_embeds = torch.mean(audio_embeds, dim=1)  # Average over time
        audio_embeds = self.components["audio_projection"](audio_embeds)

        return audio_embeds

    def _tokenize_with_special_tokens(self, text: str, tokenizer, image_embeds=None, audio_embeds=None) -> Dict:
        """Tokenize text and insert special tokens for multimodal content"""
        # Build token sequence with special tokens
        tokens = []

        # Add BOS token
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            tokens.append(tokenizer.bos_token_id)

        # Add image token if we have image
        if image_embeds is not None:
            tokens.append(self.image_token_id)

        # Add audio token if we have audio
        if audio_embeds is not None:
            tokens.append(self.audio_token_id)

        # Add text tokens
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(text_tokens)

        # Convert to tensor
        input_ids = torch.tensor([tokens])

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Store embeddings for later injection
        if image_embeds is not None:
            result["image_embeds"] = image_embeds
        if audio_embeds is not None:
            result["audio_embeds"] = audio_embeds

        return result

    def _generate_text_response(
        self, inputs: Dict, tokenizer, max_new_tokens: int, sampling: bool, temperature: float, top_k: int, top_p: float
    ) -> str:
        """Generate text response using the LLM"""
        if "llm" not in self.components:
            return "❌ Language model not available"

        # Get input tensors
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Create input embeddings with multimodal injection
        input_embeds = self._create_multimodal_embeddings(input_ids, inputs)

        with torch.no_grad():
            # Generate using Qwen2
            outputs = self.components["llm"].generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=sampling,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response (skip input tokens)
        response_tokens = outputs[0][len(input_ids[0]) :]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response_text.strip()

    def _create_multimodal_embeddings(self, input_ids: torch.Tensor, inputs: Dict) -> torch.Tensor:
        """Create embeddings with multimodal content injected"""
        # Get base text embeddings
        embeds = self.components["llm"].model.embed_tokens(input_ids)

        # Inject image embeddings at image token positions
        if "image_embeds" in inputs:
            image_mask = input_ids == self.image_token_id
            if image_mask.any():
                # Replace image token embeddings with vision features
                # Note: This is simplified - official implementation handles this more carefully
                image_embeds = inputs["image_embeds"]  # [1, 32, 3584]
                # For now, use the first token position of each image
                image_positions = image_mask.nonzero(as_tuple=True)[1]
                if len(image_positions) > 0 and image_embeds is not None:
                    # Take first sequence position for each image token
                    embeds[:, image_positions[0], :] = image_embeds[:, 0, :]

        # Inject audio embeddings at audio token positions
        if "audio_embeds" in inputs:
            audio_mask = input_ids == self.audio_token_id
            if audio_mask.any():
                audio_embeds = inputs["audio_embeds"]  # [1, 3584]
                audio_positions = audio_mask.nonzero(as_tuple=True)[1]
                if len(audio_positions) > 0 and audio_embeds is not None:
                    embeds[:, audio_positions[0], :] = audio_embeds

        return embeds

    def _generate_audio(self, text: str, output_path: str) -> Optional[str]:
        """Generate speech audio from text using ChatTTS + Vocos"""
        if "tts" not in self.components or self.vocos is None:
            print("⚠️ TTS components not available")
            return None

        try:
            # For now, return placeholder - full TTS implementation needs more work
            # This would require:
            # 1. Tokenizing text into semantic tokens
            # 2. Using ChatTTS to predict semantic tokens from LLM hidden states
            # 3. Using Vocos to convert semantic tokens to audio

            print("🎵 Audio generation not yet implemented (placeholder)")
            return None

        except Exception as e:
            print(f"❌ Audio generation failed: {e}")
            return None


# Factory function
def create_official_minicpm_wrapper(model_path: str = None) -> MiniCPMOWrapper:
    """Create MiniCPM-o wrapper with official components"""
    config = MiniCPMOConfig(model_path=model_path)
    model = MiniCPMOWrapper(config)
    return model
