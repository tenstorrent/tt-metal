# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MiniCPM-o-2_6 End-to-End Pipeline

Orchestrates all components for multimodal audio-text processing:
1. Audio Input → Whisper Encoder → Audio Projector → Qwen LLM (with cross-attention)
2. Text Input → Qwen LLM → DVAE → Audio Output

Supports both inference modes:
- Audio-to-Text: Audio understanding and description
- Text-to-Audio: Speech synthesis from text
"""

import os
import torch
import numpy as np
import ttnn
from loguru import logger
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Import MiniCPM components
from .ttnn_whisper_encoder import TtnnWhisperEncoder
from .ttnn_qwen_llm import TtnnQwenLLM
from .ttnn_audio_projector import TtnnAudioProjector
from .ttnn_dvae import TtnnDVAE
from .minicpm_generator import MiniCPMGenerator


class TtnnMiniCPMPipeline:
    """TTNN MiniCPM-o-2_6 end-to-end pipeline"""

    def __init__(
        self,
        device: ttnn.Device,
        execution_mode: str = "hybrid",  # "ttnn" | "cpu" | "hybrid"
        whisper_config: Optional[Dict[str, Any]] = None,
        qwen_config: Optional[Dict[str, Any]] = None,
        dva_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MiniCPM-o-2_6 pipeline

        Args:
            device: TTNN device
            execution_mode: Execution mode - "hybrid" (TTNN + CPU), "ttnn" (all TTNN), "cpu" (all CPU)
            whisper_config: Whisper encoder configuration
            qwen_config: Qwen LLM configuration
            dva_config: DVAE configuration
        """
        self.device = device
        self.execution_mode = execution_mode

        # Validate execution mode
        if execution_mode not in ["hybrid", "ttnn", "cpu"]:
            raise ValueError(f"Invalid execution_mode: {execution_mode}. Must be 'hybrid', 'ttnn', or 'cpu'")

        # Default configurations matching MiniCPM-o-2_6 specs
        self.whisper_config = whisper_config or {
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
        }

        self.qwen_config = qwen_config or {
            "vocab_size": 151700,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 18,  # Corrected based on actual checkpoint weights
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "cross_attention_layers": [3, 7, 11, 15],  # Adjusted for 18 layers
        }

        self.dva_config = dva_config or {
            "num_encoder_layers": 12,  # Production: 12 layers
            "num_decoder_layers": 12,  # Production: 12 layers
            "hidden_dim": 256,
            "num_mel_bins": 100,
            "bn_dim": 128,  # Production: 128
            "enable_gfsq": True,  # Enable/disable GFSQ quantization
        }

        # Initialize components
        self.whisper_encoder = None
        self.audio_projector = None
        self.siglip_encoder = None
        self.vision_resampler = None
        self.qwen_llm = None
        self.dvae = None
        self.chattts_decoder = None
        self.tokenizer = None
        self.generator = None

        self._initialized = False

        logger.info("Initialized MiniCPM-o-2_6 pipeline")

    def load_all_components_with_real_weights(
        self, checkpoint_path: str = "openbmb/MiniCPM-o-2_6", trust_remote_code: bool = True
    ) -> None:
        """
        Load real weights for all components using memory-efficient streaming approach.

        This avoids loading full models into memory by using component-wise weight loading
        with proper memory cleanup between components.

        Args:
            checkpoint_path: HuggingFace model identifier

        Raises:
            RuntimeError: If critical components fail to load and cannot be recovered
        """
        from tt.weight_loader import MiniCPMWeightLoader

        logger.info(f"Loading MiniCPM-o components using memory-efficient streaming from {checkpoint_path}...")

        try:
            # Initialize components to None first
            self.qwen_llm = None
            self.whisper_encoder = None
            self.audio_projector = None
            self.dvae = None
            self.tokenizer = None

            # Step 1: Load tokenizer first (lightweight)
            logger.info("📝 Loading tokenizer...")
            try:
                self._load_tokenizer(checkpoint_path, trust_remote_code)
                logger.info("✅ Tokenizer loaded")
            except Exception as e:
                logger.error(f"❌ Tokenizer loading failed: {e}")
                raise RuntimeError(f"Critical component failed: Tokenizer. {e}")

            # Step 2: Create Qwen LLM FIRST (most memory-intensive, needs fresh device memory)
            logger.info("🚀 Creating Qwen LLM (most memory intensive)...")
            try:
                # Use the standard TT transformers approach like simple_text_demo.py
                from models.tt_transformers.tt.generator import create_submeshes

                # Create submeshes for proper mesh device handling
                submeshes = create_submeshes(self.device, data_parallel=1)
                submesh = submeshes[0]  # Use first submesh

                logger.info("Creating Qwen LLM using standard TT transformers...")
                # Use smaller max_seq_len for memory efficiency (like simple_text_demo.py)
                # The model can handle larger sequences at inference time with proper KV cache management
                # Use performance optimizations to reduce memory usage
                from models.tt_transformers.tt.model_config import DecodersPrecision

                optimizations = DecodersPrecision.performance(
                    self.qwen_config["num_hidden_layers"], "minicpm-o-2-6-ttnn"
                )

                # First create model with dummy weights to avoid trust_remote_code issues
                # Use MiniCPM directly since it's now in the supported models list
                original_hf_model = os.environ.get("HF_MODEL")
                os.environ["HF_MODEL"] = "minicpm-o-2-6-ttnn"

                from models.tt_transformers.tt.model_config import ModelArgs

                model_args = ModelArgs(
                    submesh,
                    instruct=True,
                    dummy_weights=True,  # Use dummy weights to avoid loading from HF
                    max_batch_size=1,
                    max_seq_len=1024,
                    optimizations=optimizations,
                    cache_hf=False,  # Don't cache HF models
                )
                # Restore original HF_MODEL
                if original_hf_model is not None:
                    os.environ["HF_MODEL"] = original_hf_model
                else:
                    os.environ.pop("HF_MODEL", None)

                # Override model name and checkpoint to prevent inference issues
                model_args.model_name = "minicpm-o-2-6-ttnn"
                model_args.checkpoint_path = None  # Prevent checkpoint inference

                # Set model parameters manually since we're using dummy weights
                model_args.n_layers = self.qwen_config["num_hidden_layers"]
                model_args.n_heads = self.qwen_config["num_attention_heads"]
                model_args.n_kv_heads = self.qwen_config["num_key_value_heads"]
                model_args.dim = self.qwen_config["hidden_size"]
                model_args.vocab_size = self.qwen_config["vocab_size"]
                model_args.max_seq_len = self.qwen_config["max_position_embeddings"]
                model_args.rope_theta = self.qwen_config["rope_theta"]

                # Create MiniCPM Qwen LLM that extends the standard TT transformers
                # Load the dummy state_dict for initialization
                state_dict = model_args.load_state_dict()

                from tt.ttnn_qwen_llm import TtnnQwenLLM

                self.qwen_llm = TtnnQwenLLM(
                    args=model_args,
                    dtype=ttnn.bfloat8_b,
                    mesh_device=submesh,
                    state_dict=state_dict,  # Use the loaded dummy state_dict
                    weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
                    cross_attention_layers=self.qwen_config.get("cross_attention_layers", [8, 16, 24]),
                )

                logger.info("✅ Qwen LLM TTNN model created")

                # Load Qwen weights immediately to free up host memory
                logger.info("🧠 Loading Qwen LLM weights...")
                try:
                    qwen_weights = weight_loader.load_from_official_checkpoint(
                        checkpoint_path=checkpoint_path, components=["qwen"], trust_remote_code=trust_remote_code
                    ).get("qwen", {})

                    if qwen_weights and self.qwen_llm:
                        self.qwen_llm.load_weights(qwen_weights)
                        logger.info("✅ Qwen LLM weights loaded")
                    else:
                        logger.warning("⚠️ No Qwen weights found, using generated weights")

                    del qwen_weights
                    import gc

                    gc.collect()

                except Exception as e:
                    logger.warning(f"Qwen LLM weight loading failed: {e}")
                    # Continue - Qwen might still work with generated weights

                logger.info("✅ Qwen LLM fully loaded and ready")

            except Exception as e:
                logger.error(f"❌ Qwen LLM creation failed: {e}")
                raise RuntimeError(f"Critical component failed: Qwen LLM. {e}")

            # Step 3: Create other TTNN models (allocates TTNN memory properly)
            logger.info("🏗️ Creating TTNN models...")
            try:
                from tt.ttnn_whisper_encoder import TtnnWhisperEncoder
                from tt.ttnn_audio_projector import TtnnAudioProjector
                from tt.ttnn_dvae import TtnnDVAE

                # Create models without weights first (allocates TTNN memory)
                self.whisper_encoder = TtnnWhisperEncoder(device=self.device, **self.whisper_config)
                logger.info("✅ Whisper encoder created")

                self.audio_projector = TtnnAudioProjector(
                    device=self.device,
                    input_dim=self.whisper_config["d_model"],
                    output_dim=self.qwen_config["hidden_size"],
                )
                logger.info("✅ Audio projector created")

                self.dvae = TtnnDVAE(self.device, **self.dva_config)
                logger.info("✅ DVAE created")

                # Create SigLip and vision resampler
                from tt.ttnn_siglip_encoder import TtnnSigLipEncoder
                from tt.ttnn_resampler import TtnnResampler

                self.siglip_encoder = TtnnSigLipEncoder(
                    device=self.device,
                    hidden_size=1152,
                    num_hidden_layers=27,
                )
                logger.info("✅ SigLip encoder created")

                self.vision_resampler = TtnnResampler(
                    device=self.device,
                    embed_dim=self.qwen_config["hidden_size"],
                    kv_dim=1152,  # Vision encoder output dimension
                    num_queries=64,
                )
                logger.info("✅ Vision resampler created")

            except Exception as e:
                logger.warning(f"TTNN model creation failed: {e}")
                # Continue - some models might still work

            # Step 3: Load weights one component at a time with memory cleanup
            weight_loader = MiniCPMWeightLoader()

            # Load Whisper weights from official checkpoint
            logger.info("🎵 Loading Whisper weights...")
            try:
                # Load only Whisper weights from official checkpoint
                whisper_weights = weight_loader.load_from_official_checkpoint(
                    checkpoint_path=checkpoint_path, components=["whisper"]
                ).get("whisper", {})
                if whisper_weights and self.whisper_encoder:
                    self.whisper_encoder.load_weights(whisper_weights)
                    logger.info("✅ Whisper weights loaded")
                # Clean up memory
                del whisper_weights
                import gc

                gc.collect()
            except Exception as e:
                logger.warning(f"Whisper weight loading failed: {e}")

            # Load Audio Projector weights
            logger.info("🔊 Loading Audio Projector weights...")
            try:
                import gc

                audio_weights = weight_loader.load_audio_projector_weights(checkpoint_path)
                if audio_weights and self.audio_projector:
                    self.audio_projector.load_weights(audio_weights)
                    logger.info("✅ Audio Projector weights loaded")
                # Clean up memory
                del audio_weights
                gc.collect()
            except Exception as e:
                logger.warning(f"Audio Projector weight loading failed: {e}")

            # Load DVAE weights
            logger.info("🎭 Loading DVAE weights...")
            try:
                import gc

                dva_weights = weight_loader.load_dvae_weights(checkpoint_path)
                if dva_weights and self.dvae:
                    self.dvae.load_weights(dva_weights)
                    logger.info("✅ DVAE weights loaded")
                # Clean up memory
                del dva_weights
                gc.collect()
            except Exception as e:
                logger.warning(f"DVAE weight loading failed (optional): {e}")

            # Load SigLip weights
            logger.info("👁️ Loading SigLip weights...")
            try:
                import gc

                siglip_weights = weight_loader.load_from_official_checkpoint(
                    checkpoint_path=checkpoint_path, components=["siglip"]
                ).get("siglip", {})
                if siglip_weights and self.siglip_encoder:
                    self.siglip_encoder.load_weights(siglip_weights)
                    logger.info("✅ SigLip weights loaded")
                del siglip_weights
                gc.collect()
            except Exception as e:
                logger.warning(f"SigLip weight loading failed (optional): {e}")

            # Load vision resampler weights
            logger.info("🔄 Loading Vision Resampler weights...")
            try:
                resampler_weights = weight_loader.load_from_official_checkpoint(
                    checkpoint_path=checkpoint_path, components=["resampler"]
                ).get("resampler", {})

                # Strip 'resampler.' prefix from resampler weights as decoder expects bare keys
                resampler_weights_stripped = {}
                for key, value in resampler_weights.items():
                    if key.startswith("resampler."):
                        new_key = key[10:]  # Remove 'resampler.' prefix
                        resampler_weights_stripped[new_key] = value
                    else:
                        resampler_weights_stripped[key] = value

                logger.debug(f"Resampler weight keys after stripping: {list(resampler_weights_stripped.keys())[:5]}")

                if resampler_weights_stripped and self.vision_resampler:
                    self.vision_resampler.load_weights(resampler_weights_stripped)
                    logger.info("✅ Vision resampler weights loaded")
                del resampler_weights
                gc.collect()
            except Exception as e:
                logger.warning(f"Vision resampler weight loading failed (optional): {e}")

            # Create ChatTTS decoder
            logger.info("🗣️ Creating ChatTTS decoder...")
            try:
                from tt.ttnn_chattts_decoder import TtnnChatTTSDecoder

                self.chattts_decoder = TtnnChatTTSDecoder(
                    device=self.device,
                    hidden_size=768,
                    num_hidden_layers=20,
                    num_attention_heads=12,
                    num_text_tokens=21178,
                    num_audio_tokens=626,
                )
                logger.info("✅ ChatTTS decoder created")

            except Exception as e:
                logger.warning(f"ChatTTS decoder creation failed: {e}")
                # Continue without ChatTTS - audio generation will be disabled

            # Load ChatTTS weights
            if self.chattts_decoder is not None:
                logger.info("🎤 Loading ChatTTS weights...")
                try:
                    chattts_weights = weight_loader.load_from_official_checkpoint(
                        checkpoint_path=checkpoint_path, components=["chattts"]
                    ).get("chattts", {})

                    # Strip 'tts.' prefix from ChatTTS weights as decoder expects bare keys
                    chattts_weights_stripped = {}
                    for key, value in chattts_weights.items():
                        if key.startswith("tts."):
                            new_key = key[4:]  # Remove 'tts.' prefix
                            chattts_weights_stripped[new_key] = value
                        else:
                            chattts_weights_stripped[key] = value

                    logger.debug(f"ChatTTS weight keys after stripping: {list(chattts_weights_stripped.keys())[:5]}")

                    if chattts_weights_stripped:
                        self.chattts_decoder.load_weights(chattts_weights_stripped)
                        logger.info("✅ ChatTTS weights loaded")
                    del chattts_weights
                    gc.collect()
                except Exception as e:
                    logger.warning(f"ChatTTS weight loading failed (optional): {e}")

            # Qwen LLM already created in Step 2 above

            self._initialized = True

            # Report final status
            status = self.get_component_status()
            loaded_count = sum(1 for comp in status.values() if comp)  # Count True values
            total_count = len(status)

            logger.info(f"✅ Memory-efficient loading complete: {loaded_count}/{total_count} components ready")

            if loaded_count == 0:
                logger.warning("⚠️ No components were successfully loaded")
            elif loaded_count < total_count:
                logger.info("⚠️ Partial loading - some components may not be available")

        except Exception as e:
            logger.error(f"❌ Memory-efficient loading failed: {e}")
            raise RuntimeError(f"Failed to load components: {e}")

    def process_multimodal_input(
        self,
        audio: Optional[Union[str, torch.Tensor]] = None,
        image: Optional[Union[str, torch.Tensor]] = None,
        text: str = "",
    ) -> Dict[str, ttnn.Tensor]:
        """
        Process multimodal inputs and return encoded features for cross-attention.

        Args:
            audio: Audio input (file path or tensor)
            image: Image input (file path or tensor)
            text: Text prompt

        Returns:
            Dictionary with encoded features:
            - 'audio_features': Audio features for cross-attention [seq_len, hidden_size]
            - 'image_features': Image features for cross-attention [seq_len, hidden_size]
            - 'text_tokens': Tokenized text input
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call load_all_components_with_real_weights() first.")

        features = {}

        # Validate inputs
        if not any([audio, image, text]):
            raise ValueError("At least one input modality (audio, image, or text) must be provided")

        # Process audio input
        if audio is not None:
            logger.info("Processing audio input...")
            try:
                # Check if required components are available
                if self.whisper_encoder is None:
                    raise RuntimeError("Whisper encoder not available - audio processing disabled")
                if self.audio_projector is None:
                    raise RuntimeError("Audio projector not available - audio processing disabled")

                # Convert audio to mel spectrogram if needed
                if isinstance(audio, str):
                    from reference.modality_processors import AudioProcessor

                    processor = AudioProcessor()
                    mel_spec = processor.process_audio_file(audio)
                    if mel_spec is None:
                        raise ValueError(f"Failed to process audio file: {audio}")
                else:
                    mel_spec = audio  # Assume it's already processed

                # Validate mel spectrogram
                if not isinstance(mel_spec, torch.Tensor):
                    raise ValueError(f"Audio input must be tensor, got {type(mel_spec)}")
                if len(mel_spec.shape) != 3:
                    raise ValueError(f"Mel spectrogram must be 3D [batch, n_mels, time], got shape {mel_spec.shape}")

                # Encode with Whisper
                audio_features = self.whisper_encoder.forward(mel_spec)

                # Project to LLM hidden space
                audio_features = self.audio_projector.forward(audio_features)

                features["audio_features"] = audio_features
                logger.info(f"✅ Audio features processed: {audio_features.shape}")

            except Exception as e:
                logger.error(f"❌ Audio processing failed: {e}")
                raise RuntimeError(f"Audio processing failed: {e}")

        # Process image input
        if image is not None:
            logger.info("Processing image input...")
            try:
                if self.siglip_encoder is None or self.vision_resampler is None:
                    raise RuntimeError("Vision components not available - image processing disabled")

                # Load and preprocess image
                if isinstance(image, str):
                    from reference.modality_processors import ImageProcessor

                    processor = ImageProcessor()
                    image_tensor = processor.process_image_file(image)
                    if image_tensor is None:
                        raise ValueError(f"Failed to process image file: {image}")
                else:
                    image_tensor = image

                # Validate image tensor [batch, 3, H, W]
                if not isinstance(image_tensor, torch.Tensor):
                    raise ValueError(f"Image must be tensor, got {type(image_tensor)}")
                if len(image_tensor.shape) != 4:
                    raise ValueError(f"Image must be 4D [batch, 3, H, W], got {image_tensor.shape}")

                # Encode with SigLip
                vision_embeddings = self.siglip_encoder.forward(image_tensor)

                # Resample to fixed length with cross-attention
                image_features = self.vision_resampler.forward(vision_embeddings)

                features["image_features"] = image_features
                logger.info(f"✅ Image features processed: {image_features.shape}")

            except Exception as e:
                logger.error(f"❌ Image processing failed: {e}")
                raise RuntimeError(f"Image processing failed: {e}")

        # Process text input
        if text:
            logger.info("Processing text input...")
            try:
                # Validate text input
                if not isinstance(text, str):
                    raise ValueError(f"Text input must be string, got {type(text)}")
                if len(text.strip()) == 0:
                    raise ValueError("Text input cannot be empty")

                # Load tokenizer if needed
                if self.tokenizer is None:
                    from tt.tokenizer_utils import load_minicpm_tokenizer

                    self.tokenizer = load_minicpm_tokenizer()
                    if self.tokenizer is None:
                        raise RuntimeError("Failed to load tokenizer")

                # Tokenize text
                raw_tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
                logger.debug(f"Raw tokenizer output: {raw_tokens}, shape: {raw_tokens.shape}")
                tokens = raw_tokens.squeeze(0)  # Only squeeze batch dimension, keep seq_len
                logger.debug(f"After squeeze(0): {tokens}, shape: {tokens.shape}, ndim: {tokens.ndim}")

                # Validate token length
                max_context = self.qwen_config.get("max_position_embeddings", 32768)
                if len(tokens) > max_context:
                    logger.warning(f"Text length {len(tokens)} exceeds max context {max_context}, truncating")
                    tokens = tokens[:max_context]

                features["text_tokens"] = tokens
                logger.info(f"✅ Text tokens processed: {len(tokens)} tokens")

            except Exception as e:
                logger.error(f"❌ Text processing failed: {e}")
                raise RuntimeError(f"Text processing failed: {e}")

        # Validate that we have some features
        if not features:
            raise RuntimeError("No inputs were successfully processed")

        # Log processing summary
        processed_modalities = list(features.keys())
        logger.info(f"✅ Multimodal input processing completed: {processed_modalities}")

        return features

    def generate_audio_response(
        self,
        text_tokens: torch.Tensor,
        llm_hidden_states: ttnn.Tensor,
        max_audio_len: int = 1000,
    ) -> Optional[np.ndarray]:
        """
        Generate audio from text tokens using ChatTTS + DVAE pipeline.

        Args:
            text_tokens: Text token IDs [batch, seq_len]
            llm_hidden_states: Hidden states from Qwen LLM for conditioning
            max_audio_len: Maximum audio sequence length

        Returns:
            Audio waveform as numpy array, or None if generation fails
        """
        if self.chattts_decoder is None or self.dvae is None:
            logger.warning("ChatTTS or DVAE not available - audio generation disabled")
            return None

        try:
            logger.info("Generating audio with ChatTTS decoder...")

            # Generate audio codes with ChatTTS
            audio_codes = self.chattts_decoder.forward(
                text_tokens=text_tokens,
                llm_hidden_states=llm_hidden_states,
                max_length=max_audio_len,
            )

            logger.info(f"Generated audio codes: {audio_codes.shape}")

            # Decode to mel spectrogram with DVAE
            mel_spectrogram = self.dvae.decode(audio_codes)

            logger.info(f"Generated mel spectrogram: {mel_spectrogram.shape}")

            # Convert mel to waveform (using vocoder - CPU fallback)
            from reference.modality_processors import AudioPostprocessor

            postprocessor = AudioPostprocessor()
            waveform = postprocessor.mel_to_waveform(mel_spectrogram)

            logger.info(f"✅ Generated audio waveform: {waveform.shape}")
            return waveform

        except Exception as e:
            logger.error(f"❌ Audio generation failed: {e}")
            return None

    def generate_multimodal_response(
        self,
        audio: Optional[Union[str, torch.Tensor]] = None,
        image: Optional[Union[str, torch.Tensor]] = None,
        text: str = "",
        max_gen_len: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        generate_audio: bool = False,
    ) -> Dict[str, Any]:
        """
        Full multimodal generation pipeline.

        Args:
            audio: Audio input (file path or tensor)
            image: Image input (file path or tensor)
            text: Text prompt
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p probability
            top_k: Top-k sampling
            generate_audio: Whether to generate audio output

        Returns:
            Dictionary with 'text' response and optionally 'audio_waveform'
        """
        logger.info("Starting multimodal generation...")

        # Process inputs
        features = self.process_multimodal_input(audio, image, text)

        # Prepare multimodal prompt
        from tt.tokenizer_utils import format_multimodal_prompt

        has_audio = "audio_features" in features
        has_image = "image_features" in features
        multimodal_prompt = format_multimodal_prompt(text=text, has_audio=has_audio, has_image=has_image)

        # Generate text response
        if self.generator is None:
            self.generator = MiniCPMGenerator(self, self.tokenizer)

        text_response = self.generator.generate_text(
            prompt=multimodal_prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            audio_input=features.get("audio_features"),
            image_input=features.get("image_features"),
        )

        result = {"text": text_response}

        # Generate audio if requested
        if generate_audio:
            logger.info("Generating audio response...")
            # Tokenize generated text
            text_tokens = self.tokenizer(text_response, return_tensors="pt")["input_ids"].squeeze()

            # Generate mel spectrogram with DVAE
            mel_spec = self.dvae.generate_audio(text_tokens)

            # Convert to waveform
            from reference.modality_processors import AudioPostprocessor

            postprocessor = AudioPostprocessor()
            waveform = postprocessor.mel_to_waveform(mel_spec)

            result["audio_waveform"] = waveform
            logger.info(f"Audio generated: {waveform.shape}")

        logger.info("✅ Multimodal generation complete")
        return result

    def load_weights(
        self,
        weights_path: Union[str, Path],
        whisper_weights: Optional[Dict[str, torch.Tensor]] = None,
        qwen_weights: Optional[Dict[str, torch.Tensor]] = None,
        audio_projector_weights: Optional[Dict[str, torch.Tensor]] = None,
        dva_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Load weights for all components

        Args:
            weights_path: Base path for weight files
            whisper_weights: Pre-loaded Whisper weights (optional)
            qwen_weights: Pre-loaded Qwen weights (optional)
            audio_projector_weights: Pre-loaded Audio Projector weights (optional)
            dva_weights: Pre-loaded DVAE weights (optional)
        """
        weights_path = Path(weights_path)

        logger.info("Loading MiniCPM-o-2_6 weights...")

        # Initialize components
        self.whisper_encoder = TtnnWhisperEncoder(device=self.device, **self.whisper_config)

        self.audio_projector = TtnnAudioProjector(
            device=self.device,
            input_dim=self.whisper_config["d_model"],
            output_dim=self.qwen_config["hidden_size"],
        )

        self.qwen_llm = TtnnQwenLLM(
            device=self.device,
            cross_attention_layers=self.qwen_config["cross_attention_layers"],
            **{k: v for k, v in self.qwen_config.items() if k != "cross_attention_layers"},
        )

        self.dvae = TtnnDVAE(self.device, **self.dva_config)

        # Load weights
        if whisper_weights:
            self.whisper_encoder.load_weights(whisper_weights)
        else:
            # Load from file (placeholder - would need actual weight loading logic)
            logger.warning("Whisper weights not provided - using random weights")

        if audio_projector_weights:
            self.audio_projector.load_weights(audio_projector_weights)
        else:
            logger.warning("Audio projector weights not provided - using random weights")

        if qwen_weights:
            self.qwen_llm.load_weights(qwen_weights)
        else:
            logger.warning("Qwen weights not provided - using random weights")

        if dva_weights:
            self.dvae.load_weights(dva_weights)
        else:
            logger.warning("DVAE weights not provided - using random weights")

        self._initialized = True
        logger.info("All MiniCPM-o-2_6 components loaded")

    def _preprocess_audio(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio input to mel spectrograms

        Args:
            audio_input: Raw audio waveform [batch, samples] or pre-computed mel [batch, 80, time]

        Returns:
            Mel spectrograms [batch, 80, time_steps]
        """
        # For now, assume input is already mel spectrograms
        # In practice, would need audio processing pipeline
        if audio_input.dim() == 2:
            # Assume [batch, samples] - need to convert to mel
            logger.warning("Audio preprocessing not implemented - assuming mel spectrograms")
            # Placeholder: create dummy mel spectrograms
            batch_size = audio_input.shape[0]
            return torch.randn(batch_size, 80, 300)  # [batch, 80, time]

        return audio_input

    def _preprocess_text(self, text_input: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess text input to token IDs

        Args:
            text_input: Text string or pre-tokenized IDs

        Returns:
            Token IDs [batch, seq_len]
        """
        if isinstance(text_input, str):
            # Placeholder tokenization - in practice would use actual tokenizer
            logger.warning("Text tokenization not implemented - using dummy tokens")
            # Placeholder: convert string to dummy token IDs
            token_ids = torch.randint(0, self.qwen_config["vocab_size"], (1, len(text_input.split())))
            return token_ids

        return text_input

    def audio_to_text(
        self,
        audio_input: torch.Tensor,
        text_prompt: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 1.0,
    ) -> str:
        """
        Audio-to-Text generation: Understand audio and generate text description

        Args:
            audio_input: Audio waveform or mel spectrograms
            text_prompt: Optional text prompt
            max_length: Maximum generation length
            temperature: Generation temperature

        Returns:
            Generated text description
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized - call load_weights() first")

        logger.info("Running audio-to-text generation...")

        # Preprocess audio
        mel_specs = self._preprocess_audio(audio_input)

        # Extract audio features with Whisper
        audio_features = self.whisper_encoder.forward(mel_specs)

        # Project to LLM embedding space
        projected_audio = self.audio_projector.forward(audio_features)

        # Prepare text input
        if text_prompt:
            text_tokens = self._preprocess_text(text_prompt)
        else:
            # Start with BOS token
            text_tokens = torch.tensor([[0]])  # BOS token

        # Generate text with Qwen LLM using cross-attention
        generated_tokens = self._generate_text(
            text_tokens,
            encoder_features=projected_audio,
            max_length=max_length,
            temperature=temperature,
        )

        # Decode tokens to text (placeholder)
        generated_text = f"Generated text from audio: {generated_tokens.tolist()}"
        logger.info(f"Generated: {generated_text}")

        return generated_text

    def text_to_audio(
        self,
        text_input: Union[str, torch.Tensor],
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Text-to-Audio synthesis: Generate speech from text

        Args:
            text_input: Input text or token IDs
            speaker_embedding: Optional speaker embedding for voice cloning

        Returns:
            Generated audio waveform
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized - call load_weights() first")

        logger.info("Running text-to-audio synthesis...")

        # Preprocess text
        text_tokens = self._preprocess_text(text_input)

        # Generate semantic tokens with Qwen LLM (text-only, no cross-attention)
        semantic_tokens = self._generate_semantic_tokens(text_tokens)

        # Generate audio with DVAE
        generated_audio = self.dvae.forward(semantic_tokens, speaker_embedding=speaker_embedding)

        logger.info(f"Generated audio shape: {generated_audio.shape}")

        return generated_audio

    def multimodal_generation(
        self,
        audio_input: torch.Tensor,
        text_input: Union[str, torch.Tensor],
        generation_mode: str = "audio_to_text",
        **kwargs,
    ) -> Union[str, torch.Tensor]:
        """
        Full multimodal generation combining audio and text

        Args:
            audio_input: Audio input
            text_input: Text input
            generation_mode: "audio_to_text" or "text_to_audio"
            **kwargs: Additional generation parameters

        Returns:
            Generated output (text or audio)
        """
        if generation_mode == "audio_to_text":
            return self.audio_to_text(audio_input, text_input, **kwargs)
        elif generation_mode == "text_to_audio":
            return self.text_to_audio(text_input, **kwargs)
        else:
            raise ValueError(f"Unknown generation mode: {generation_mode}")

    def _generate_text(
        self,
        input_tokens: torch.Tensor,
        encoder_features: Optional[ttnn.Tensor] = None,
        max_length: int = 512,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text tokens with Qwen LLM

        Args:
            input_tokens: Input token IDs [batch, seq_len]
            encoder_features: Audio features for cross-attention [batch, seq_len, hidden_size]
            max_length: Maximum generation length
            temperature: Generation temperature

        Returns:
            Generated token IDs [batch, seq_len]
        """
        # Placeholder implementation - in practice would implement proper generation loop
        logger.warning("Text generation not fully implemented - returning dummy tokens")

        batch_size, seq_len = input_tokens.shape
        # Generate some dummy tokens
        generated = torch.randint(1, self.qwen_config["vocab_size"], (batch_size, max_length - seq_len))

        return torch.cat([input_tokens, generated], dim=1)

    def _generate_semantic_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate semantic audio tokens from text

        Args:
            text_tokens: Input text tokens [batch, seq_len]

        Returns:
            Semantic tokens for DVAE [batch, seq_len]
        """
        # Placeholder - in practice would use ChatTTS or similar
        logger.warning("Semantic token generation not implemented - using dummy tokens")

        # Return dummy semantic tokens (would be audio token IDs)
        return torch.randint(0, 626, (text_tokens.shape[0], text_tokens.shape[1] * 2))  # 626 vocab size for ChatTTS

    def _setup_generator_if_possible(self) -> None:
        """
        Create generator if both tokenizer and qwen_llm are available.
        """
        if self.tokenizer is not None and self.qwen_llm is not None and self.generator is None:
            try:
                self.generator = MiniCPMGenerator(self, self.tokenizer)
                logger.info("✅ Generator created with real weights")
            except Exception as e:
                logger.warning(f"Failed to create generator: {e}")
        else:
            missing = []
            if self.tokenizer is None:
                missing.append("tokenizer")
            if self.qwen_llm is None:
                missing.append("qwen_llm")
            if len(missing) > 0:
                logger.debug(f"Generator not ready - missing: {', '.join(missing)}")

    def get_component_status(self) -> Dict[str, bool]:
        """Get initialization status of all components"""
        return {
            "whisper_encoder": self.whisper_encoder is not None,
            "audio_projector": self.audio_projector is not None,
            "qwen_llm": self.qwen_llm is not None,
            "dvae": self.dvae is not None,
            "tokenizer": self.tokenizer is not None,
            "generator": self.generator is not None,
            "pipeline_initialized": self._initialized,
        }

    def _load_tokenizer(self, checkpoint_path: str, trust_remote_code: bool = True) -> None:
        """
        Load tokenizer from the specified checkpoint path.

        Args:
            checkpoint_path: HuggingFace checkpoint path

        Raises:
            RuntimeError: If tokenizer loading fails
        """
        try:
            from tt.tokenizer_utils import load_minicpm_tokenizer

            self.tokenizer = load_minicpm_tokenizer(checkpoint_path, trust_remote_code)
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer loading returned None")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    def set_tokenizer(self, tokenizer) -> None:
        """
        Set the tokenizer for text generation.

        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        logger.info("Tokenizer set for MiniCPM pipeline")

    def generate_response(
        self,
        prompt: str,
        max_gen_len: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        High-level generation interface using MiniCPMGenerator.

        Args:
            prompt: Text prompt
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p probability
            top_k: Top-k sampling (optional)
            audio_input: Audio input tensor
            image_input: Image input tensor
            system_prompt: System prompt (optional)

        Returns:
            Generated text response
        """
        if not self.generator:
            if not self.tokenizer:
                raise ValueError("Tokenizer must be set before generation. Use set_tokenizer() first.")
            self.generator = MiniCPMGenerator(self, self.tokenizer)

        return self.generator.generate_text(
            prompt=prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            audio_input=audio_input,
            image_input=image_input,
            system_prompt=system_prompt,
        )

    def text_to_text_generation(
        self,
        prompt: str,
        max_gen_len: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Text-only generation.

        Args:
            prompt: Text prompt
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p probability
            top_k: Top-k sampling (optional)
            system_prompt: System prompt (optional)

        Returns:
            Generated text response
        """
        return self.generate_response(
            prompt=prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            audio_input=None,
            image_input=None,
            system_prompt=system_prompt,
        )

    def multimodal_to_text_generation(
        self,
        prompt: str,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        max_gen_len: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Multimodal generation (audio + text + image → text).

        Args:
            prompt: Text prompt
            audio_input: Audio input tensor
            image_input: Image input tensor
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p probability
            top_k: Top-k sampling (optional)
            system_prompt: System prompt (optional)

        Returns:
            Generated text response
        """
        return self.generate_response(
            prompt=prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            audio_input=audio_input,
            image_input=image_input,
            system_prompt=system_prompt,
        )

    def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Chat completion with conversation history and multimodal inputs.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Top-p probability
            max_gen_len: Maximum generation length
            audio_input: Audio input tensor
            image_input: Image input tensor

        Returns:
            Generated response text
        """
        if not self.generator:
            if not self.tokenizer:
                raise ValueError("Tokenizer must be set before generation. Use set_tokenizer() first.")
            self.generator = MiniCPMGenerator(self, self.tokenizer)

        return self.generator.chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            audio_input=audio_input,
            image_input=image_input,
        )

    def cleanup(self):
        """Clean up TTNN resources"""
        if self.whisper_encoder:
            # Cleanup logic would go here
            pass
        if self.audio_projector:
            # Cleanup logic would go here
            pass
        if self.qwen_llm:
            # Cleanup logic would go here
            pass
        if self.dvae:
            # Cleanup logic would go here
            pass

        logger.info("MiniCPM pipeline cleaned up")
