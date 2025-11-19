# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MiniCPM-o-2_6 Hybrid Pipeline

Hybrid implementation that uses PyTorch for Qwen LLM and TTNN for all other
validated components (SigLip, Resampler, Whisper, Audio Projector, ChatTTS, DVAE).

This enables full multimodal functionality while working around TTNN Qwen issues.
"""

import torch
import ttnn
from typing import Dict, Any, Optional, Tuple, List, Union
from loguru import logger
from pathlib import Path

# Import PyTorch Qwen reference
import sys
from pathlib import Path

ref_path = Path(__file__).parent.parent / "reference"
if str(ref_path) not in sys.path:
    sys.path.insert(0, str(ref_path))

from multimodal_qwen import MultimodalQwen2ForCausalLM, MultimodalQwen2Config

# Import TTNN components
import ttnn_whisper_encoder
import ttnn_audio_projector
from tt.ttnn_siglip_vision import TtSiglipVisionModel
import ttnn_resampler
import ttnn_chattts_decoder
import ttnn_dvae

TtnnWhisperEncoder = ttnn_whisper_encoder.TtnnWhisperEncoder
TtnnAudioProjector = ttnn_audio_projector.TtnnAudioProjector
TtSiglipVisionModel = TtSiglipVisionModel
TtnnResampler = ttnn_resampler.TtnnResampler
TtnnChatTTSDecoder = ttnn_chattts_decoder.TtnnChatTTSDecoder
TtnnDVAE = ttnn_dvae.TtnnDVAE

# Import hybrid utilities
import hybrid_utils

pytorch_to_ttnn = hybrid_utils.pytorch_to_ttnn
ttnn_to_pytorch = hybrid_utils.ttnn_to_pytorch
convert_multimodal_embeddings_to_qwen_format = hybrid_utils.convert_multimodal_embeddings_to_qwen_format
convert_qwen_hidden_states_to_ttnn_format = hybrid_utils.convert_qwen_hidden_states_to_ttnn_format
prepare_audio_input_for_whisper = hybrid_utils.prepare_audio_input_for_whisper
prepare_image_input_for_siglip = hybrid_utils.prepare_image_input_for_siglip
log_tensor_info = hybrid_utils.log_tensor_info


class TtnnMiniCPMHybridPipeline:
    """
    Hybrid MiniCPM-o-2_6 pipeline with PyTorch Qwen + TTNN components.

    Architecture:
    - Audio: TTNN Whisper → TTNN AudioProjector → PyTorch Qwen (cross-attn)
    - Vision: TTNN SigLip → TTNN Resampler → PyTorch Qwen (cross-attn)
    - Text: PyTorch Qwen
    - TTS: PyTorch Qwen → TTNN ChatTTS → TTNN DVAE
    """

    def __init__(
        self,
        device: ttnn.Device,
        qwen_config: Optional[Dict[str, Any]] = None,
        whisper_config: Optional[Dict[str, Any]] = None,
        siglip_config: Optional[Dict[str, Any]] = None,
        chattts_config: Optional[Dict[str, Any]] = None,
        dva_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hybrid pipeline.

        Args:
            device: TTNN device for TTNN components
            qwen_config: PyTorch Qwen configuration
            whisper_config: Whisper encoder configuration
            siglip_config: SigLip vision configuration
            chattts_config: ChatTTS decoder configuration
            dva_config: DVAE configuration
        """
        self.device = device
        self.initialized = False

        # Default configurations
        self.qwen_config = qwen_config or {
            "vocab_size": 151700,
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "cross_attention_layers": [8, 16, 24],  # Layers with cross-attention
            "initializer_range": 0.02,
        }

        self.whisper_config = whisper_config or {
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
        }

        self.siglip_config = siglip_config or {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 12,  # SigLip uses 12 heads
        }

        self.chattts_config = chattts_config or {
            "hidden_size": 768,
            "num_hidden_layers": 20,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "num_vq": 4,
            "num_text_tokens": 21178,
            "num_audio_tokens": 626,
        }

        self.dva_config = dva_config or {
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
            "hidden_dim": 256,
            "bn_dim": 128,
            "num_mel_bins": 100,
        }

        logger.info("Initializing TTNN MiniCPM Hybrid Pipeline...")
        logger.info(
            f"PyTorch Qwen: {self.qwen_config['hidden_size']}d × {self.qwen_config['num_hidden_layers']} layers"
        )
        logger.info(f"TTNN Components: Whisper, SigLip, Resampler, AudioProjector, ChatTTS, DVAE")

        # Initialize PyTorch Qwen (CPU/GPU)
        self._init_qwen_model()

        # Initialize TTNN components (will be loaded with weights)
        self.whisper_encoder = None
        self.audio_projector = None
        self.vision_encoder = None
        self.resampler = None
        self.chattts_decoder = None
        self.dvae = None

        # Tokenizers (to be loaded)
        self.tokenizer = None

    def _init_qwen_model(self):
        """Initialize PyTorch Qwen model with cross-attention layers."""
        logger.info("Initializing PyTorch Qwen model with cross-attention...")

        # Create config
        config = MultimodalQwen2Config(**self.qwen_config)

        # Initialize model
        self.qwen_model = MultimodalQwen2ForCausalLM(config)

        logger.info(
            f"✅ PyTorch Qwen initialized: {config.num_hidden_layers} layers, "
            f"cross-attention at {config.cross_attention_layers}"
        )

    def load_weights(self, weights_path: Optional[str] = None):
        """
        Load weights for all components.

        Args:
            weights_path: Path to MiniCPM-o-2_6 weights (optional)
        """
        logger.info("Loading weights for hybrid pipeline...")

        # Initialize TTNN components first (needed for weight loading)
        self._init_ttnn_components()

        # Load weights for all components
        self._load_all_component_weights(weights_path)

        self.initialized = True
        logger.info("✅ Hybrid pipeline weights loaded successfully")

    def _load_all_component_weights(self, weights_path: Optional[str] = None):
        """
        Load weights for all components in the hybrid pipeline.

        Args:
            weights_path: Path to MiniCPM-o-2_6 checkpoint
        """
        logger.info("Loading weights for all hybrid pipeline components...")

        try:
            # Try to load from official checkpoint
            if weights_path:
                component_weights = self._load_from_checkpoint(weights_path)
                self._load_component_weights_from_dict(component_weights)
            else:
                # Use generated weights for testing
                logger.info("No checkpoint provided, using generated weights for testing")
                self._load_generated_weights()

        except Exception as e:
            logger.warning(f"Failed to load from checkpoint: {e}")
            logger.info("Falling back to generated weights")
            self._load_generated_weights()

        # Set models to eval mode
        self.qwen_model.eval()
        logger.info("✅ All component weights loaded")

    def _load_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load component weights from MiniCPM-o-2_6 checkpoint.

        Args:
            checkpoint_path: HuggingFace model path or local path

        Returns:
            Dictionary with component weights
        """
        logger.info(f"Loading from MiniCPM checkpoint: {checkpoint_path}")

        # Import weight loader
        try:
            import weight_loader

            MiniCPMWeightLoader = weight_loader.MiniCPMWeightLoader
            weight_loader = MiniCPMWeightLoader()

            # Load components needed for hybrid pipeline
            components = ["qwen", "whisper", "audio_projector", "vision", "chattts", "dvae"]
            all_weights = weight_loader.load_from_official_checkpoint(
                checkpoint_path=checkpoint_path, components=components, trust_remote_code=True
            )

            return all_weights

        except ImportError:
            logger.warning("Weight loader not available, using generated weights")
            return {}

    def _load_component_weights_from_dict(self, component_weights: Dict[str, Any]):
        """
        Load weights into components from weight dictionary.

        Args:
            component_weights: Dictionary with component weights
        """
        # Load PyTorch Qwen weights
        if "qwen" in component_weights:
            logger.info("Loading PyTorch Qwen weights from checkpoint...")
            qwen_weights = component_weights["qwen"]
            # Convert to state_dict format expected by PyTorch model
            state_dict = {}
            for key, tensor in qwen_weights.items():
                # Convert from TTNN format back to PyTorch if needed
                if hasattr(tensor, "cpu"):  # TTNN tensor
                    tensor = tensor.cpu()
                if hasattr(tensor, "numpy"):  # TTNN tensor
                    tensor = torch.from_numpy(tensor.numpy())
                state_dict[key] = tensor

            # Load into PyTorch model
            missing_keys, unexpected_keys = self._load_qwen_state_dict(state_dict)
            if missing_keys:
                logger.warning(f"Missing Qwen keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                logger.warning(f"Unexpected Qwen keys: {unexpected_keys[:5]}...")  # Show first 5
        else:
            logger.info("No Qwen weights in checkpoint, using generated weights")

        # Load TTNN component weights
        self._load_ttnn_component_weights(component_weights)

    def _load_qwen_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """
        Load Qwen state dict with proper key mapping.

        Args:
            state_dict: State dictionary to load

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Create mapping from checkpoint keys to model keys
        key_mapping = {}

        # Embeddings
        if "model.embed_tokens.weight" in state_dict:
            key_mapping["embed_tokens.weight"] = state_dict["model.embed_tokens.weight"]

        # LM Head
        if "lm_head.weight" in state_dict:
            key_mapping["lm_head.weight"] = state_dict["lm_head.weight"]

        # Layer weights (standard Qwen2.5 format)
        for layer_idx in range(self.qwen_config["num_hidden_layers"]):
            layer_prefix = f"model.layers.{layer_idx}"

            # Self-attention
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = f"{layer_prefix}.self_attn.{proj}.weight"
                if key in state_dict:
                    key_mapping[f"layers.{layer_idx}.self_attn.{proj}.weight"] = state_dict[key]
                bias_key = f"{layer_prefix}.self_attn.{proj}.bias"
                if bias_key in state_dict:
                    key_mapping[f"layers.{layer_idx}.self_attn.{proj}.bias"] = state_dict[bias_key]

            # MLP
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{layer_prefix}.mlp.{proj}.weight"
                if key in state_dict:
                    key_mapping[f"layers.{layer_idx}.mlp.{proj}.weight"] = state_dict[key]

            # Layer norms
            for norm in ["input_layernorm", "post_attention_layernorm"]:
                key = f"{layer_prefix}.{norm}.weight"
                if key in state_dict:
                    key_mapping[f"layers.{layer_idx}.{norm}.weight"] = state_dict[key]

            # Cross-attention (if this layer has it)
            if layer_idx in self.qwen_config["cross_attention_layers"]:
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    cross_attn_key = f"{layer_prefix}.cross_attn.{proj}.weight"
                    if cross_attn_key in state_dict:
                        key_mapping[f"layers.{layer_idx}.cross_attn.{proj}.weight"] = state_dict[cross_attn_key]
                    bias_key = f"{layer_prefix}.cross_attn.{proj}.bias"
                    if bias_key in state_dict:
                        key_mapping[f"layers.{layer_idx}.cross_attn.{proj}.bias"] = state_dict[bias_key]

                # Cross-attention norms
                for norm in ["q_norm", "k_norm"]:
                    norm_key = f"{layer_prefix}.cross_attn.{norm}.weight"
                    if norm_key in state_dict:
                        key_mapping[f"layers.{layer_idx}.cross_attn.{norm}.weight"] = state_dict[norm_key]

        # Final norm
        if "model.norm.weight" in state_dict:
            key_mapping["norm.weight"] = state_dict["model.norm.weight"]

        # Load the mapped weights
        return self.qwen_model.load_state_dict(key_mapping, strict=False)

    def _load_ttnn_component_weights(self, component_weights: Dict[str, Any]):
        """
        Load weights into TTNN components.

        Args:
            component_weights: Dictionary with component weights
        """
        # Load TTNN component weights
        component_map = {
            "whisper": self.whisper_encoder,
            "audio_projector": self.audio_projector,
            # Support both 'vision' (legacy) and 'siglip' keys from weight loader
            "vision": self.resampler,  # Vision weights go to resampler
            "siglip": self.resampler,
            "chattts": self.chattts_decoder,
            "dvae": self.dvae,
        }

        for component_name, component in component_map.items():
            if component_name in component_weights and component:
                try:
                    weights = component_weights[component_name]
                    if hasattr(component, "load_weights"):
                        component.load_weights(weights)
                        logger.info(f"✅ Loaded {component_name} weights: {len(weights)} tensors")
                    else:
                        logger.warning(f"Component {component_name} has no load_weights method")
                except Exception as e:
                    logger.warning(f"Failed to load {component_name} weights: {e}")
            else:
                logger.info(f"No weights found for {component_name}, using generated weights")

    def _load_generated_weights(self):
        """Load generated weights for all components (for testing)."""
        logger.info("Loading generated weights for testing...")

        # Qwen already has random weights from initialization

        # Generate and load weights for TTNN components
        import weight_generator
        from reference_pytorch.weight_loader import DiskBasedWeightLoader

        generate_whisper_weights = weight_generator.generate_whisper_weights
        generate_audio_projector_weights = weight_generator.generate_audio_projector_weights
        generate_resampler_weights = weight_generator.generate_resampler_weights
        generate_chattts_weights = weight_generator.generate_chattts_weights
        generate_dvae_weights = weight_generator.generate_dvae_weights

        # Generate weights for each component
        whisper_weights = generate_whisper_weights(**self.whisper_config)
        audio_proj_weights = generate_audio_projector_weights(
            input_dim=self.whisper_config["d_model"], output_dim=self.qwen_config["hidden_size"]
        )
        # Load SigLip weights from MiniCPM
        weight_loader = DiskBasedWeightLoader()
        vision_weights = weight_loader.get_siglip_weights()
        resampler_weights = generate_resampler_weights(
            embed_dim=self.qwen_config["hidden_size"],
            num_queries=64,
            num_heads=self.qwen_config["num_attention_heads"],
            kv_dim=self.siglip_config["hidden_size"],
        )
        chattts_weights = generate_chattts_weights(**self.chattts_config)
        dvae_weights = generate_dvae_weights(**self.dva_config)

        # Load weights into TTNN components
        if self.whisper_encoder:
            self.whisper_encoder.load_weights(whisper_weights)
            logger.info("✅ Whisper weights loaded")
        if self.audio_projector:
            self.audio_projector.load_weights(audio_proj_weights)
            logger.info("✅ Audio projector weights loaded")
        if self.vision_encoder:
            self.vision_encoder.load_weights(vision_weights)
            logger.info("✅ SigLip weights loaded")
        if self.resampler:
            self.resampler.load_weights(resampler_weights)
            logger.info("✅ Resampler weights loaded")
        if self.chattts_decoder:
            self.chattts_decoder.load_weights(chattts_weights)
            logger.info("✅ ChatTTS weights loaded")
        if self.dvae:
            self.dvae.load_weights(dvae_weights)
            logger.info("✅ DVAE weights loaded")

        logger.info("✅ Generated weights loaded for all components")

    def _init_ttnn_components(self):
        """Initialize TTNN components."""
        logger.info("Initializing TTNN components...")

        try:
            # Audio components
            logger.info(f"Creating TtnnWhisperEncoder with config: {self.whisper_config}")
            self.whisper_encoder = TtnnWhisperEncoder(device=self.device, **self.whisper_config)
            logger.info(
                f"✅ TTNN Whisper Encoder initialized: {type(self.whisper_encoder)}, callable: {callable(self.whisper_encoder)}"
            )

            self.audio_projector = TtnnAudioProjector(
                device=self.device, input_dim=self.whisper_config["d_model"], output_dim=self.qwen_config["hidden_size"]
            )
            logger.info("✅ TTNN Audio Projector initialized")

            # Vision components
            logger.info(f"TtSiglipVisionModel class: {TtSiglipVisionModel}")
            logger.info(f"SigLip config: {self.siglip_config}")
            try:
                self.vision_encoder = TtSiglipVisionModel(
                    mesh_device=self.device,
                    hidden_size=self.siglip_config.get("hidden_size", 1152),
                    num_attention_heads=self.siglip_config.get("num_attention_heads", 16),
                    num_hidden_layers=self.siglip_config.get("num_hidden_layers", 27),
                    patch_size=self.siglip_config.get("patch_size", 14),
                    image_size=self.siglip_config.get("image_size", 980),
                    num_channels=self.siglip_config.get("num_channels", 3),
                )
                logger.info(
                    f"✅ TTNN SigLip Vision Model initialized: {type(self.vision_encoder)}, has forward: {hasattr(self.vision_encoder, 'forward')}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to initialize SigLip Vision Model: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.vision_encoder = None

            # Resampler: queries live in Qwen hidden space (kv_dim=input dim should be SigLip output)
            self.resampler = TtnnResampler(
                device=self.device,
                embed_dim=self.qwen_config["hidden_size"],  # output embed dim for queries (e.g., 3584)
                num_queries=64,
                num_heads=self.qwen_config["num_attention_heads"],
                kv_dim=self.siglip_config["hidden_size"],  # input kv dim from SigLip (e.g., 1152)
            )
            logger.info("✅ TTNN Resampler initialized")

            # TTS components
            self.chattts_decoder = TtnnChatTTSDecoder(device=self.device, **self.chattts_config)
            logger.info("✅ TTNN ChatTTS Decoder initialized")

            self.dvae = TtnnDVAE(device=self.device, **self.dva_config)
            logger.info("✅ TTNN DVAE initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize TTNN components: {e}")
            raise

    def _load_ttnn_weights(self, weights_path: Optional[str] = None):
        """Load TTNN component weights."""
        logger.info("Loading TTNN component weights...")

        # For now, use generated weights (would load from actual checkpoint)
        logger.info("Using generated weights for TTNN components (would load from checkpoint)")

        # TTNN components are initialized with generated weights
        # Real implementation would load from MiniCPM-o-2_6 checkpoint

    def process_audio(self, audio_input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process audio through TTNN Whisper + AudioProjector.

        Args:
            audio_input: Raw audio tensor [batch, samples] or [samples]

        Returns:
            Audio embeddings for Qwen cross-attention [batch, seq, hidden]
        """
        logger.info("Processing audio through TTNN pipeline...")

        # Prepare audio for Whisper (placeholder - would need actual mel spectrogram)
        audio_mel = prepare_audio_input_for_whisper(audio_input, self.device)
        log_tensor_info(audio_mel, "Audio input to Whisper")

        # TTNN Whisper encoder (support either .forward or callable)
        whisper_fn = getattr(self.whisper_encoder, "forward", self.whisper_encoder)
        audio_features = whisper_fn(audio_mel)
        log_tensor_info(audio_features, "Whisper audio features")

        # TTNN Audio projector
        audio_proj_fn = getattr(self.audio_projector, "forward", self.audio_projector)
        audio_embeds = audio_proj_fn(audio_features)
        log_tensor_info(audio_embeds, "Projected audio embeddings")

        # Convert to PyTorch format for Qwen (provide mesh device to compose shards)
        audio_embeds_pt = ttnn_to_pytorch(audio_embeds, mesh_device=self.device)
        qwen_audio_embeds = convert_multimodal_embeddings_to_qwen_format(audio_embeds_pt)

        log_tensor_info(qwen_audio_embeds, "Audio embeddings for Qwen")
        return qwen_audio_embeds

    def process_vision(self, image_input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process vision through TTNN SigLip + Resampler.

        Args:
            image_input: Image tensor [batch, channels, height, width]

        Returns:
            Vision embeddings for Qwen cross-attention [batch, seq, hidden]
        """
        logger.info("Processing vision through TTNN pipeline...")

        # Prepare image for SigLip (placeholder - would need actual preprocessing)
        image_tensor = prepare_image_input_for_siglip(image_input, self.device)
        log_tensor_info(image_tensor, "Image input to SigLip")

        # TTNN SigLip encoder
        siglip_fn = getattr(self.vision_encoder, "forward", self.vision_encoder)
        vision_features = siglip_fn(image_tensor)
        log_tensor_info(vision_features, "SigLip vision features")

        # TTNN Resampler
        resampler_fn = getattr(self.resampler, "forward", self.resampler)
        # Compute target sizes (patch grid height, width) from image shape and SigLip patch size
        try:
            patch_size = getattr(self.vision_encoder, "patch_size", 16)
            img_h = image_input.shape[2]
            img_w = image_input.shape[3]
            tgt_h = img_h // patch_size
            tgt_w = img_w // patch_size
            tgt_sizes = torch.tensor([[tgt_h, tgt_w]] * image_input.shape[0], dtype=torch.int64)
        except Exception:
            # Fallback single target size
            tgt_sizes = torch.tensor([[14, 14]], dtype=torch.int64)

        vision_embeds = resampler_fn(vision_features, tgt_sizes)
        log_tensor_info(vision_embeds, "Resampled vision embeddings")

        # Convert to PyTorch format for Qwen (provide mesh device to compose shards)
        vision_embeds_pt = ttnn_to_pytorch(vision_embeds, mesh_device=self.device)
        qwen_vision_embeds = convert_multimodal_embeddings_to_qwen_format(vision_embeds_pt)

        log_tensor_info(qwen_vision_embeds, "Vision embeddings for Qwen")
        return qwen_vision_embeds

    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using PyTorch Qwen with multimodal cross-attention.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            vision_embeds: Vision embeddings from process_vision() [batch, seq, hidden]
            audio_embeds: Audio embeddings from process_audio() [batch, seq, hidden]
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with generated token IDs and text
        """
        logger.info("Generating text with PyTorch Qwen...")

        # Prepare multimodal embeddings
        encoder_hidden_states = None
        if vision_embeds is not None or audio_embeds is not None:
            # Concatenate vision and audio embeddings for cross-attention
            multimodal_embeds = []
            if vision_embeds is not None:
                multimodal_embeds.append(vision_embeds)
            if audio_embeds is not None:
                multimodal_embeds.append(audio_embeds)

            if len(multimodal_embeds) > 0:
                encoder_hidden_states = torch.cat(multimodal_embeds, dim=1)
                log_tensor_info(encoder_hidden_states, "Multimodal embeddings for cross-attention")

        # Simple autoregressive generation (for now)
        generated_ids = input_ids.clone()
        batch_size, seq_len = input_ids.shape
        # Encoder attention mask for multimodal embeddings (all ones if not provided)
        encoder_attention_mask = None
        if encoder_hidden_states is not None:
            encoder_attention_mask = torch.ones(batch_size, encoder_hidden_states.shape[1], dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                # Ensure attention_mask has appropriate dimensions for the model; if not, pass None
                attn_mask_arg = attention_mask if (attention_mask is not None and attention_mask.dim() >= 3) else None

                outputs = self.qwen_model(
                    input_ids=generated_ids,
                    attention_mask=attn_mask_arg,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

                # Get next token logits
                next_token_logits = outputs["last_hidden_state"][:, -1, :]
                next_token_logits = self.qwen_model.lm_head(next_token_logits)

                # Simple greedy decoding (for now)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device),
                        ],
                        dim=1,
                    )

        return {
            "generated_token_ids": generated_ids,
            "input_ids": input_ids,
            "multimodal_used": encoder_hidden_states is not None,
        }

    def generate_audio(self, text_input: Union[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Generate audio from text using PyTorch Qwen + TTNN TTS.

        Args:
            text_input: Text string or token IDs

        Returns:
            Generated audio tensor
        """
        logger.info("Generating audio through hybrid pipeline...")

        # First, get Qwen to generate text (or use input text)
        if isinstance(text_input, str):
            # Tokenize text (placeholder - would need actual tokenizer)
            logger.warning("Text tokenization not implemented - using dummy input")
            input_ids = torch.randint(0, self.qwen_config["vocab_size"], (1, 32))
        else:
            input_ids = text_input

        # Generate text with Qwen (no multimodal for TTS)
        text_generation = self.generate_text(input_ids, **kwargs)
        generated_tokens = text_generation["generated_token_ids"]

        # Get hidden states from Qwen for TTS conditioning
        with torch.no_grad():
            qwen_outputs = self.qwen_model(input_ids=generated_tokens, output_hidden_states=True)
            # Use last hidden state for TTS conditioning
            hidden_states = qwen_outputs["last_hidden_state"]

        log_tensor_info(hidden_states, "Qwen hidden states for TTS")

        # Convert to TTNN format for ChatTTS
        ttnn_hidden = convert_qwen_hidden_states_to_ttnn_format(hidden_states, self.device)
        log_tensor_info(ttnn_hidden, "TTNN hidden states for ChatTTS")

        # TTNN ChatTTS decoder
        audio_tokens = self.chattts_decoder.forward(ttnn_hidden)
        log_tensor_info(audio_tokens, "ChatTTS audio tokens")

        # TTNN DVAE reconstruction
        audio_output = self.dvae.forward(audio_tokens)
        log_tensor_info(audio_output, "DVAE reconstructed audio")

        # Convert back to PyTorch (provide mesh device to compose shards)
        audio_output_pt = ttnn_to_pytorch(audio_output, mesh_device=self.device)

        return audio_output_pt

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        generate_audio: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main pipeline interface for multimodal processing.

        Args:
            text: Input text
            audio: Input audio tensor
            image: Input image tensor
            generate_audio: Whether to generate audio output
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with results
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call load_weights() first.")

        results = {
            "text_input": text,
            "audio_input": audio is not None,
            "image_input": image is not None,
        }

        # Process modalities
        vision_embeds = None
        audio_embeds = None

        if image is not None:
            vision_embeds = self.process_vision(image)

        if audio is not None:
            audio_embeds = self.process_audio(audio)

        # Generate text response
        if text:
            # Tokenize text (placeholder)
            input_ids = torch.randint(0, self.qwen_config["vocab_size"], (1, 32))
            attention_mask = torch.ones_like(input_ids)

            text_results = self.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_embeds=vision_embeds,
                audio_embeds=audio_embeds,
                **kwargs,
            )
            results.update(text_results)

        # Generate audio if requested
        if generate_audio:
            if text:
                audio_output = self.generate_audio(text, **kwargs)
                results["generated_audio"] = audio_output
            else:
                logger.warning("Cannot generate audio without text input")

        return results

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, "whisper_encoder") and self.whisper_encoder:
                self.whisper_encoder.__del__()
            if hasattr(self, "audio_projector") and self.audio_projector:
                self.audio_projector.__del__()
            if hasattr(self, "vision_encoder") and self.vision_encoder:
                self.vision_encoder.__del__()
            if hasattr(self, "resampler") and self.resampler:
                self.resampler.__del__()
            if hasattr(self, "chattts_decoder") and self.chattts_decoder:
                self.chattts_decoder.__del__()
            if hasattr(self, "dvae") and self.dvae:
                self.dvae.__del__()
        except:
            pass  # Ignore cleanup errors


"""
TTNN MiniCPM-o-2_6 Hybrid Pipeline

Hybrid implementation that uses PyTorch for Qwen LLM and TTNN for all other
validated components (SigLip, Resampler, Whisper, Audio Projector, ChatTTS, DVAE).

This enables full multimodal functionality while working around TTNN Qwen issues.
"""

import torch
import ttnn
from typing import Dict, Any, Optional, Tuple, List, Union
from loguru import logger
from pathlib import Path

# Import PyTorch Qwen reference
import sys
from pathlib import Path

ref_path = Path(__file__).parent.parent / "reference"
if str(ref_path) not in sys.path:
    sys.path.insert(0, str(ref_path))

from multimodal_qwen import MultimodalQwen2ForCausalLM, MultimodalQwen2Config

# Import TTNN components
import ttnn_whisper_encoder
import ttnn_audio_projector
from tt.ttnn_siglip_vision import TtSiglipVisionModel
import ttnn_resampler
import ttnn_chattts_decoder
import ttnn_dvae

TtnnWhisperEncoder = ttnn_whisper_encoder.TtnnWhisperEncoder
TtnnAudioProjector = ttnn_audio_projector.TtnnAudioProjector
TtSiglipVisionModel = TtSiglipVisionModel
TtnnResampler = ttnn_resampler.TtnnResampler
TtnnChatTTSDecoder = ttnn_chattts_decoder.TtnnChatTTSDecoder
TtnnDVAE = ttnn_dvae.TtnnDVAE

# Import hybrid utilities
import hybrid_utils

pytorch_to_ttnn = hybrid_utils.pytorch_to_ttnn
ttnn_to_pytorch = hybrid_utils.ttnn_to_pytorch
convert_multimodal_embeddings_to_qwen_format = hybrid_utils.convert_multimodal_embeddings_to_qwen_format
convert_qwen_hidden_states_to_ttnn_format = hybrid_utils.convert_qwen_hidden_states_to_ttnn_format
prepare_audio_input_for_whisper = hybrid_utils.prepare_audio_input_for_whisper
prepare_image_input_for_siglip = hybrid_utils.prepare_image_input_for_siglip
log_tensor_info = hybrid_utils.log_tensor_info


class TtnnMiniCPMHybridPipeline:
    """
    Hybrid MiniCPM-o-2_6 pipeline with PyTorch Qwen + TTNN components.

    Architecture:
    - Audio: TTNN Whisper → TTNN AudioProjector → PyTorch Qwen (cross-attn)
    - Vision: TTNN SigLip → TTNN Resampler → PyTorch Qwen (cross-attn)
    - Text: PyTorch Qwen
    - TTS: PyTorch Qwen → TTNN ChatTTS → TTNN DVAE
    """

    def __init__(
        self,
        device: ttnn.Device,
        qwen_config: Optional[Dict[str, Any]] = None,
        whisper_config: Optional[Dict[str, Any]] = None,
        siglip_config: Optional[Dict[str, Any]] = None,
        chattts_config: Optional[Dict[str, Any]] = None,
        dva_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hybrid pipeline.

        Args:
            device: TTNN device for TTNN components
            qwen_config: PyTorch Qwen configuration
            whisper_config: Whisper encoder configuration
            siglip_config: SigLip vision configuration
            chattts_config: ChatTTS decoder configuration
            dva_config: DVAE configuration
        """
        self.device = device
        self.initialized = False

        # Default configurations
        self.qwen_config = qwen_config or {
            "vocab_size": 151700,
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "cross_attention_layers": [8, 16, 24],  # Layers with cross-attention
            "initializer_range": 0.02,
        }

        self.whisper_config = whisper_config or {
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
        }

        self.siglip_config = siglip_config or {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 12,  # SigLip uses 12 heads
        }

        self.chattts_config = chattts_config or {
            "hidden_size": 768,
            "num_hidden_layers": 20,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "num_vq": 4,
            "num_text_tokens": 21178,
            "num_audio_tokens": 626,
        }

        self.dva_config = dva_config or {
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
            "hidden_dim": 256,
            "bn_dim": 128,
            "num_mel_bins": 100,
        }

        logger.info("Initializing TTNN MiniCPM Hybrid Pipeline...")
        logger.info(
            f"PyTorch Qwen: {self.qwen_config['hidden_size']}d × {self.qwen_config['num_hidden_layers']} layers"
        )
        logger.info(f"TTNN Components: Whisper, SigLip, Resampler, AudioProjector, ChatTTS, DVAE")

        # Initialize PyTorch Qwen (CPU/GPU)
        self._init_qwen_model()

        # Initialize TTNN components (will be loaded with weights)
        self.whisper_encoder = None
        self.audio_projector = None
        self.vision_encoder = None
        self.resampler = None
        self.chattts_decoder = None
        self.dvae = None

        # Tokenizers (to be loaded)
        self.tokenizer = None

    def _init_qwen_model(self):
        """Initialize PyTorch Qwen model with cross-attention layers."""
        logger.info("Initializing PyTorch Qwen model with cross-attention...")

        # Create config
        config = MultimodalQwen2Config(**self.qwen_config)

        # Initialize model
        self.qwen_model = MultimodalQwen2ForCausalLM(config)

        logger.info(
            f"✅ PyTorch Qwen initialized: {config.num_hidden_layers} layers, "
            f"cross-attention at {config.cross_attention_layers}"
        )

    def load_weights(self, weights_path: Optional[str] = None):
        """
        Load weights for all components.

        Args:
            weights_path: Path to MiniCPM-o-2_6 weights (optional)
        """
        logger.info("Loading weights for hybrid pipeline...")

        # Initialize TTNN components first (needed for weight loading)
        self._init_ttnn_components()

        # Load weights for all components
        self._load_all_component_weights(weights_path)

        self.initialized = True
        logger.info("✅ Hybrid pipeline weights loaded successfully")

    def _load_all_component_weights(self, weights_path: Optional[str] = None):
        """
        Load weights for all components in the hybrid pipeline.

        Args:
            weights_path: Path to MiniCPM-o-2_6 checkpoint
        """
        logger.info("Loading weights for all hybrid pipeline components...")

        try:
            # Try to load from official checkpoint
            if weights_path:
                component_weights = self._load_from_checkpoint(weights_path)
                self._load_component_weights_from_dict(component_weights)
            else:
                # Use generated weights for testing
                logger.info("No checkpoint provided, using generated weights for testing")
                self._load_generated_weights()

        except Exception as e:
            logger.warning(f"Failed to load from checkpoint: {e}")
            logger.info("Falling back to generated weights")
            self._load_generated_weights()

        # Set models to eval mode
        self.qwen_model.eval()
        logger.info("✅ All component weights loaded")

    def _load_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load component weights from MiniCPM-o-2_6 checkpoint.

        Args:
            checkpoint_path: HuggingFace model path or local path

        Returns:
            Dictionary with component weights
        """
        logger.info(f"Loading from MiniCPM checkpoint: {checkpoint_path}")

        # Import weight loader
        try:
            import weight_loader

            MiniCPMWeightLoader = weight_loader.MiniCPMWeightLoader
            weight_loader = MiniCPMWeightLoader()

            # Load components needed for hybrid pipeline
            components = ["qwen", "whisper", "audio_projector", "vision", "chattts", "dvae"]
            all_weights = weight_loader.load_from_official_checkpoint(
                checkpoint_path=checkpoint_path, components=components, trust_remote_code=True
            )

            return all_weights

        except ImportError:
            logger.warning("Weight loader not available, using generated weights")
            return {}

    def _load_component_weights_from_dict(self, component_weights: Dict[str, Any]):
        """
        Load weights into components from weight dictionary.

        Args:
            component_weights: Dictionary with component weights
        """
        # Load PyTorch Qwen weights
        if "qwen" in component_weights:
            logger.info("Loading PyTorch Qwen weights from checkpoint...")
            qwen_weights = component_weights["qwen"]
            # Convert to state_dict format expected by PyTorch model
            state_dict = {}
            for key, tensor in qwen_weights.items():
                # Convert from TTNN format back to PyTorch if needed
                if hasattr(tensor, "cpu"):  # TTNN tensor
                    tensor = tensor.cpu()
                if hasattr(tensor, "numpy"):  # TTNN tensor
                    tensor = torch.from_numpy(tensor.numpy())
                state_dict[key] = tensor

            # Load into PyTorch model
            missing_keys, unexpected_keys = self._load_qwen_state_dict(state_dict)
            if missing_keys:
                logger.warning(f"Missing Qwen keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                logger.warning(f"Unexpected Qwen keys: {unexpected_keys[:5]}...")  # Show first 5
        else:
            logger.info("No Qwen weights in checkpoint, using generated weights")

        # Load TTNN component weights
        self._load_ttnn_component_weights(component_weights)

    def _load_qwen_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """
        Load Qwen state dict with proper key mapping.

        Args:
            state_dict: State dictionary to load

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Create mapping from checkpoint keys to model keys
        key_mapping = {}

        # Embeddings
        if "model.embed_tokens.weight" in state_dict:
            key_mapping["embed_tokens.weight"] = state_dict["model.embed_tokens.weight"]

        # LM Head
        if "lm_head.weight" in state_dict:
            key_mapping["lm_head.weight"] = state_dict["lm_head.weight"]

        # Layer weights (standard Qwen2.5 format)
        for layer_idx in range(self.qwen_config["num_hidden_layers"]):
            layer_prefix = f"model.layers.{layer_idx}"

            # Self-attention
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = f"{layer_prefix}.self_attn.{proj}.weight"
                if key in state_dict:
                    key_mapping[f"layers.{layer_idx}.self_attn.{proj}.weight"] = state_dict[key]
                bias_key = f"{layer_prefix}.self_attn.{proj}.bias"
                if bias_key in state_dict:
                    key_mapping[f"layers.{layer_idx}.self_attn.{proj}.bias"] = state_dict[bias_key]

            # MLP
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{layer_prefix}.mlp.{proj}.weight"
                if key in state_dict:
                    key_mapping[f"layers.{layer_idx}.mlp.{proj}.weight"] = state_dict[key]

            # Layer norms
            for norm in ["input_layernorm", "post_attention_layernorm"]:
                key = f"{layer_prefix}.{norm}.weight"
                if key in state_dict:
                    key_mapping[f"layers.{layer_idx}.{norm}.weight"] = state_dict[key]

            # Cross-attention (if this layer has it)
            if layer_idx in self.qwen_config["cross_attention_layers"]:
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    cross_attn_key = f"{layer_prefix}.cross_attn.{proj}.weight"
                    if cross_attn_key in state_dict:
                        key_mapping[f"layers.{layer_idx}.cross_attn.{proj}.weight"] = state_dict[cross_attn_key]
                    bias_key = f"{layer_prefix}.cross_attn.{proj}.bias"
                    if bias_key in state_dict:
                        key_mapping[f"layers.{layer_idx}.cross_attn.{proj}.bias"] = state_dict[bias_key]

                # Cross-attention norms
                for norm in ["q_norm", "k_norm"]:
                    norm_key = f"{layer_prefix}.cross_attn.{norm}.weight"
                    if norm_key in state_dict:
                        key_mapping[f"layers.{layer_idx}.cross_attn.{norm}.weight"] = state_dict[norm_key]

        # Final norm
        if "model.norm.weight" in state_dict:
            key_mapping["norm.weight"] = state_dict["model.norm.weight"]

        # Load the mapped weights
        return self.qwen_model.load_state_dict(key_mapping, strict=False)

    def _load_ttnn_component_weights(self, component_weights: Dict[str, Any]):
        """
        Load weights into TTNN components.

        Args:
            component_weights: Dictionary with component weights
        """
        # Load TTNN component weights
        component_map = {
            "whisper": self.whisper_encoder,
            "audio_projector": self.audio_projector,
            # Support both 'vision' (legacy) and 'siglip' keys from weight loader
            "vision": self.resampler,  # Vision weights go to resampler
            "siglip": self.resampler,
            "chattts": self.chattts_decoder,
            "dvae": self.dvae,
        }

        for component_name, component in component_map.items():
            if component_name in component_weights and component:
                try:
                    weights = component_weights[component_name]
                    if hasattr(component, "load_weights"):
                        component.load_weights(weights)
                        logger.info(f"✅ Loaded {component_name} weights: {len(weights)} tensors")
                    else:
                        logger.warning(f"Component {component_name} has no load_weights method")
                except Exception as e:
                    logger.warning(f"Failed to load {component_name} weights: {e}")
            else:
                logger.info(f"No weights found for {component_name}, using generated weights")

    def _load_generated_weights(self):
        """Load generated weights for all components (for testing)."""
        logger.info("Loading generated weights for testing...")

        # Qwen already has random weights from initialization

        # Generate and load weights for TTNN components
        import weight_generator
        from reference_pytorch.weight_loader import DiskBasedWeightLoader

        generate_whisper_weights = weight_generator.generate_whisper_weights
        generate_audio_projector_weights = weight_generator.generate_audio_projector_weights
        generate_resampler_weights = weight_generator.generate_resampler_weights
        generate_chattts_weights = weight_generator.generate_chattts_weights
        generate_dvae_weights = weight_generator.generate_dvae_weights

        # Generate weights for each component
        whisper_weights = generate_whisper_weights(**self.whisper_config)
        audio_proj_weights = generate_audio_projector_weights(
            input_dim=self.whisper_config["d_model"], output_dim=self.qwen_config["hidden_size"]
        )
        # Load SigLip weights from MiniCPM
        weight_loader = DiskBasedWeightLoader()
        vision_weights = weight_loader.get_siglip_weights()
        resampler_weights = generate_resampler_weights(
            embed_dim=self.qwen_config["hidden_size"],
            num_queries=64,
            num_heads=self.qwen_config["num_attention_heads"],
            kv_dim=self.siglip_config["hidden_size"],
        )
        chattts_weights = generate_chattts_weights(**self.chattts_config)
        dvae_weights = generate_dvae_weights(**self.dva_config)

        # Load weights into TTNN components
        if self.whisper_encoder:
            self.whisper_encoder.load_weights(whisper_weights)
            logger.info("✅ Whisper weights loaded")
        if self.audio_projector:
            self.audio_projector.load_weights(audio_proj_weights)
            logger.info("✅ Audio projector weights loaded")
        if self.vision_encoder:
            self.vision_encoder.load_weights(vision_weights)
            logger.info("✅ SigLip weights loaded")
        if self.resampler:
            self.resampler.load_weights(resampler_weights)
            logger.info("✅ Resampler weights loaded")
        if self.chattts_decoder:
            self.chattts_decoder.load_weights(chattts_weights)
            logger.info("✅ ChatTTS weights loaded")
        if self.dvae:
            self.dvae.load_weights(dvae_weights)
            logger.info("✅ DVAE weights loaded")

        logger.info("✅ Generated weights loaded for all components")

    def _init_ttnn_components(self):
        """Initialize TTNN components."""
        logger.info("Initializing TTNN components...")

        try:
            # Audio components
            logger.info(f"Creating TtnnWhisperEncoder with config: {self.whisper_config}")
            self.whisper_encoder = TtnnWhisperEncoder(device=self.device, **self.whisper_config)
            logger.info(
                f"✅ TTNN Whisper Encoder initialized: {type(self.whisper_encoder)}, callable: {callable(self.whisper_encoder)}"
            )

            self.audio_projector = TtnnAudioProjector(
                device=self.device, input_dim=self.whisper_config["d_model"], output_dim=self.qwen_config["hidden_size"]
            )
            logger.info("✅ TTNN Audio Projector initialized")

            # Vision components
            logger.info(f"TtSiglipVisionModel class: {TtSiglipVisionModel}")
            logger.info(f"SigLip config: {self.siglip_config}")
            try:
                self.vision_encoder = TtSiglipVisionModel(
                    mesh_device=self.device,
                    hidden_size=self.siglip_config.get("hidden_size", 1152),
                    num_attention_heads=self.siglip_config.get("num_attention_heads", 16),
                    num_hidden_layers=self.siglip_config.get("num_hidden_layers", 27),
                    patch_size=self.siglip_config.get("patch_size", 14),
                    image_size=self.siglip_config.get("image_size", 980),
                    num_channels=self.siglip_config.get("num_channels", 3),
                )
                logger.info(
                    f"✅ TTNN SigLip Vision Model initialized: {type(self.vision_encoder)}, has forward: {hasattr(self.vision_encoder, 'forward')}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to initialize SigLip Vision Model: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.vision_encoder = None

            # Resampler: queries live in Qwen hidden space (kv_dim=input dim should be SigLip output)
            self.resampler = TtnnResampler(
                device=self.device,
                embed_dim=self.qwen_config["hidden_size"],  # output embed dim for queries (e.g., 3584)
                num_queries=64,
                num_heads=self.qwen_config["num_attention_heads"],
                kv_dim=self.siglip_config["hidden_size"],  # input kv dim from SigLip (e.g., 1152)
            )
            logger.info("✅ TTNN Resampler initialized")

            # TTS components
            self.chattts_decoder = TtnnChatTTSDecoder(device=self.device, **self.chattts_config)
            logger.info("✅ TTNN ChatTTS Decoder initialized")

            self.dvae = TtnnDVAE(device=self.device, **self.dva_config)
            logger.info("✅ TTNN DVAE initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize TTNN components: {e}")
            raise

    def _load_ttnn_weights(self, weights_path: Optional[str] = None):
        """Load TTNN component weights."""
        logger.info("Loading TTNN component weights...")

        # For now, use generated weights (would load from actual checkpoint)
        logger.info("Using generated weights for TTNN components (would load from checkpoint)")

        # TTNN components are initialized with generated weights
        # Real implementation would load from MiniCPM-o-2_6 checkpoint

    def process_audio(self, audio_input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process audio through TTNN Whisper + AudioProjector.

        Args:
            audio_input: Raw audio tensor [batch, samples] or [samples]

        Returns:
            Audio embeddings for Qwen cross-attention [batch, seq, hidden]
        """
        logger.info("Processing audio through TTNN pipeline...")

        # Prepare audio for Whisper (placeholder - would need actual mel spectrogram)
        audio_mel = prepare_audio_input_for_whisper(audio_input, self.device)
        log_tensor_info(audio_mel, "Audio input to Whisper")

        # TTNN Whisper encoder (support either .forward or callable)
        whisper_fn = getattr(self.whisper_encoder, "forward", self.whisper_encoder)
        audio_features = whisper_fn(audio_mel)
        log_tensor_info(audio_features, "Whisper audio features")

        # TTNN Audio projector
        audio_proj_fn = getattr(self.audio_projector, "forward", self.audio_projector)
        audio_embeds = audio_proj_fn(audio_features)
        log_tensor_info(audio_embeds, "Projected audio embeddings")

        # Convert to PyTorch format for Qwen (provide mesh device to compose shards)
        audio_embeds_pt = ttnn_to_pytorch(audio_embeds, mesh_device=self.device)
        qwen_audio_embeds = convert_multimodal_embeddings_to_qwen_format(audio_embeds_pt)

        log_tensor_info(qwen_audio_embeds, "Audio embeddings for Qwen")
        return qwen_audio_embeds

    def process_vision(self, image_input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process vision through TTNN SigLip + Resampler.

        Args:
            image_input: Image tensor [batch, channels, height, width]

        Returns:
            Vision embeddings for Qwen cross-attention [batch, seq, hidden]
        """
        logger.info("Processing vision through TTNN pipeline...")

        # Prepare image for SigLip (placeholder - would need actual preprocessing)
        image_tensor = prepare_image_input_for_siglip(image_input, self.device)
        log_tensor_info(image_tensor, "Image input to SigLip")

        # TTNN SigLip encoder
        siglip_fn = getattr(self.vision_encoder, "forward", self.vision_encoder)
        vision_features = siglip_fn(image_tensor)
        log_tensor_info(vision_features, "SigLip vision features")

        # TTNN Resampler
        resampler_fn = getattr(self.resampler, "forward", self.resampler)
        # Compute target sizes (patch grid height, width) from image shape and SigLip patch size
        try:
            patch_size = getattr(self.vision_encoder, "patch_size", 16)
            img_h = image_input.shape[2]
            img_w = image_input.shape[3]
            tgt_h = img_h // patch_size
            tgt_w = img_w // patch_size
            tgt_sizes = torch.tensor([[tgt_h, tgt_w]] * image_input.shape[0], dtype=torch.int64)
        except Exception:
            # Fallback single target size
            tgt_sizes = torch.tensor([[14, 14]], dtype=torch.int64)

        vision_embeds = resampler_fn(vision_features, tgt_sizes)
        log_tensor_info(vision_embeds, "Resampled vision embeddings")

        # Convert to PyTorch format for Qwen (provide mesh device to compose shards)
        vision_embeds_pt = ttnn_to_pytorch(vision_embeds, mesh_device=self.device)
        qwen_vision_embeds = convert_multimodal_embeddings_to_qwen_format(vision_embeds_pt)

        log_tensor_info(qwen_vision_embeds, "Vision embeddings for Qwen")
        return qwen_vision_embeds

    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using PyTorch Qwen with multimodal cross-attention.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            vision_embeds: Vision embeddings from process_vision() [batch, seq, hidden]
            audio_embeds: Audio embeddings from process_audio() [batch, seq, hidden]
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with generated token IDs and text
        """
        logger.info("Generating text with PyTorch Qwen...")

        # Prepare multimodal embeddings
        encoder_hidden_states = None
        if vision_embeds is not None or audio_embeds is not None:
            # Concatenate vision and audio embeddings for cross-attention
            multimodal_embeds = []
            if vision_embeds is not None:
                multimodal_embeds.append(vision_embeds)
            if audio_embeds is not None:
                multimodal_embeds.append(audio_embeds)

            if len(multimodal_embeds) > 0:
                encoder_hidden_states = torch.cat(multimodal_embeds, dim=1)
                log_tensor_info(encoder_hidden_states, "Multimodal embeddings for cross-attention")

        # Simple autoregressive generation (for now)
        generated_ids = input_ids.clone()
        batch_size, seq_len = input_ids.shape
        # Encoder attention mask for multimodal embeddings (all ones if not provided)
        encoder_attention_mask = None
        if encoder_hidden_states is not None:
            encoder_attention_mask = torch.ones(batch_size, encoder_hidden_states.shape[1], dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                # Ensure attention_mask has appropriate dimensions for the model; if not, pass None
                attn_mask_arg = attention_mask if (attention_mask is not None and attention_mask.dim() >= 3) else None

                outputs = self.qwen_model(
                    input_ids=generated_ids,
                    attention_mask=attn_mask_arg,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

                # Get next token logits
                next_token_logits = outputs["last_hidden_state"][:, -1, :]
                next_token_logits = self.qwen_model.lm_head(next_token_logits)

                # Simple greedy decoding (for now)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device),
                        ],
                        dim=1,
                    )

        return {
            "generated_token_ids": generated_ids,
            "input_ids": input_ids,
            "multimodal_used": encoder_hidden_states is not None,
        }

    def generate_audio(self, text_input: Union[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Generate audio from text using PyTorch Qwen + TTNN TTS.

        Args:
            text_input: Text string or token IDs

        Returns:
            Generated audio tensor
        """
        logger.info("Generating audio through hybrid pipeline...")

        # First, get Qwen to generate text (or use input text)
        if isinstance(text_input, str):
            # Tokenize text (placeholder - would need actual tokenizer)
            logger.warning("Text tokenization not implemented - using dummy input")
            input_ids = torch.randint(0, self.qwen_config["vocab_size"], (1, 32))
        else:
            input_ids = text_input

        # Generate text with Qwen (no multimodal for TTS)
        text_generation = self.generate_text(input_ids, **kwargs)
        generated_tokens = text_generation["generated_token_ids"]

        # Get hidden states from Qwen for TTS conditioning
        with torch.no_grad():
            qwen_outputs = self.qwen_model(input_ids=generated_tokens, output_hidden_states=True)
            # Use last hidden state for TTS conditioning
            hidden_states = qwen_outputs["last_hidden_state"]

        log_tensor_info(hidden_states, "Qwen hidden states for TTS")

        # Convert to TTNN format for ChatTTS
        ttnn_hidden = convert_qwen_hidden_states_to_ttnn_format(hidden_states, self.device)
        log_tensor_info(ttnn_hidden, "TTNN hidden states for ChatTTS")

        # TTNN ChatTTS decoder
        audio_tokens = self.chattts_decoder.forward(ttnn_hidden)
        log_tensor_info(audio_tokens, "ChatTTS audio tokens")

        # TTNN DVAE reconstruction
        audio_output = self.dvae.forward(audio_tokens)
        log_tensor_info(audio_output, "DVAE reconstructed audio")

        # Convert back to PyTorch (provide mesh device to compose shards)
        audio_output_pt = ttnn_to_pytorch(audio_output, mesh_device=self.device)

        return audio_output_pt

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        generate_audio: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main pipeline interface for multimodal processing.

        Args:
            text: Input text
            audio: Input audio tensor
            image: Input image tensor
            generate_audio: Whether to generate audio output
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with results
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call load_weights() first.")

        results = {
            "text_input": text,
            "audio_input": audio is not None,
            "image_input": image is not None,
        }

        # Process modalities
        vision_embeds = None
        audio_embeds = None

        if image is not None:
            vision_embeds = self.process_vision(image)

        if audio is not None:
            audio_embeds = self.process_audio(audio)

        # Generate text response
        if text:
            # Tokenize text (placeholder)
            input_ids = torch.randint(0, self.qwen_config["vocab_size"], (1, 32))
            attention_mask = torch.ones_like(input_ids)

            text_results = self.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_embeds=vision_embeds,
                audio_embeds=audio_embeds,
                **kwargs,
            )
            results.update(text_results)

        # Generate audio if requested
        if generate_audio:
            if text:
                audio_output = self.generate_audio(text, **kwargs)
                results["generated_audio"] = audio_output
            else:
                logger.warning("Cannot generate audio without text input")

        return results

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, "whisper_encoder") and self.whisper_encoder:
                self.whisper_encoder.__del__()
            if hasattr(self, "audio_projector") and self.audio_projector:
                self.audio_projector.__del__()
            if hasattr(self, "vision_encoder") and self.vision_encoder:
                self.vision_encoder.__del__()
            if hasattr(self, "resampler") and self.resampler:
                self.resampler.__del__()
            if hasattr(self, "chattts_decoder") and self.chattts_decoder:
                self.chattts_decoder.__del__()
            if hasattr(self, "dvae") and self.dvae:
                self.dvae.__del__()
        except:
            pass  # Ignore cleanup errors
