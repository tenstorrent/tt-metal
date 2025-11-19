"""
Unified MiniCPM Pipeline

Single pipeline handling text-only, vision+text, audio+text, and full multimodal.
Text-only is special case when image/audio not provided.

Architecture:
1. Multimodal encoders produce embeddings in Qwen's space (dim=3584)
2. Replace text placeholder embeddings with multimodal embeddings
3. Qwen LLM processes unified embedding sequence
4. Optional TTS generation for speech output
"""

import torch
import ttnn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
from loguru import logger
from pathlib import Path
import sys
import os

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_pytorch"))

# Import components
from .minicpm_weight_bridge import MiniCPMWeightBridge
from .ttnn_siglip_vision import TtnnSigLIPEncoder
from .ttnn_resampler import TtnnVisionResampler
from .ttnn_whisper_encoder import TtnnWhisperEncoder
from .ttnn_audio_projector import TtnnAudioProjector
from .ttnn_dvae import TtnnDVAE

# Import tt_transformers
from models.tt_transformers.tt.common import create_tt_model
from models.tt_transformers.tt.model_config import ModelArgs


class UnifiedMiniCPMPipeline:
    """
    Unified pipeline for MiniCPM-o-2_6 with all modalities.

    Modalities:
    - Text: Always present (base capability)
    - Vision: Optional SigLIP + Resampler
    - Audio: Optional Whisper + AudioProjector
    - TTS: Optional DVAE + Vocos

    Usage:
        pipeline = UnifiedMiniCPMPipeline(mesh_device)

        # Text-only
        result = pipeline.generate("What is AI?")

        # Vision + text
        result = pipeline.generate("Describe this", image=img)

        # Audio + text
        result = pipeline.generate("Transcribe", audio=audio_array)

        # With TTS
        result = pipeline.generate("Hello", generate_audio=True)
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        model_config: Optional[Dict[str, Any]] = None,
        lazy_load: bool = True,
    ):
        """
        Initialize unified pipeline.

        Args:
            mesh_device: TTNN mesh device (e.g., N300 with shape (1, 2))
            model_config: Optional model configuration overrides
            lazy_load: If True, load components on first use (default)
        """
        self.mesh_device = mesh_device
        self.config = model_config or self._default_config()
        self.lazy_load = lazy_load

        # Weight bridge (loads once, caches)
        self.weight_bridge = MiniCPMWeightBridge()

        # Set up paged attention configuration
        self.paged_attention_config = self._setup_paged_attention()

        # Core LLM (always loaded)
        logger.info("Initializing Qwen LLM with MiniCPM weights...")
        (
            self.model_args,
            self.model,
            self.page_table,
            self.tt_kv_cache,
            self.tokenizer,
            self.processor,
        ) = self._init_qwen_llm()

        # Create generator
        from models.tt_transformers.tt.generator import Generator

        self.qwen_generator = Generator(
            self.model, self.model_args, self.mesh_device, processor=self.processor, tokenizer=self.tokenizer
        )

        # Multimodal components (lazy loaded)
        self.vision_encoder = None
        self.vision_resampler = None
        self.audio_encoder = None
        self.audio_projector = None
        self.tts_decoder = None
        self.vocos = None

        logger.info("✅ UnifiedMiniCPMPipeline initialized")

    def _setup_paged_attention(self):
        """Set up paged attention configuration"""
        from models.tt_transformers.tt.common import PagedAttentionConfig

        # Use same config as simple_text_demo.py
        page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
        return PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )

    def _create_page_table(self, global_batch_size, data_parallel):
        """Create page table for paged attention"""
        import torch

        page_table = None
        if self.paged_attention_config:
            # Implied shuffling of blocks
            permutation = torch.randperm(self.paged_attention_config.max_num_blocks)
            # Page table which maps virtual blocks to physical
            reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
            page_table = reverse_permutation.reshape(
                global_batch_size, self.paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
            )
        return page_table

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration matching MiniCPM-o-2_6"""
        return {
            # Qwen LLM
            "vocab_size": 151700,
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_seq_len": 1024,  # Default sequence length
            # Vision (SigLIP)
            "vision_hidden_size": 1152,
            "vision_num_queries": 32,  # Resampler outputs 32 tokens
            # Audio (Whisper)
            "audio_hidden_size": 1024,
            "audio_pool_stride": 2,
            # TTS
            "enable_tts": True,
        }

    def _init_qwen_llm(self):
        """
        Initialize Qwen LLM with MiniCPM weights via tt_transformers.

        Returns:
            Tuple of (model_args, model, page_table, tt_kv_cache, tokenizer, processor)
        """
        # Set up data parallel and batch size (single user for now)
        data_parallel = 1
        global_batch_size = 1  # Single user

        logger.info("Loading Qwen weights from MiniCPM checkpoint...")
        qwen_weights = self.weight_bridge.get_qwen_weights()

        logger.info(f"Loaded {len(qwen_weights)} weights. Keys sample: {list(qwen_weights.keys())[:10]}")
        if "tok_embeddings.weight" not in qwen_weights:
            logger.error("❌ tok_embeddings.weight not found in converted weights!")
            logger.error(f"Available keys: {sorted(qwen_weights.keys())}")
            raise ValueError("Missing tok_embeddings.weight in converted MiniCPM weights")

        logger.info("Creating tt_transformers model with custom weights...")
        # Create model args - let it infer model name from environment
        model_args = ModelArgs(
            mesh_device=self.mesh_device,
            instruct=False,
            max_batch_size=global_batch_size // data_parallel,
            max_seq_len=self.config["max_seq_len"],
            dummy_weights=False,  # We're providing custom weights
        )

        # Override max_prefill_chunk_size for MiniCPM compatibility
        model_args.max_prefill_chunk_size = 4

        # Check if we should use dummy weights
        use_dummy_weights = self.config.get("dummy_weights", False)
        weights_to_use = None if use_dummy_weights else qwen_weights

        # Update model_args.dummy_weights based on use_dummy_weights
        model_args.dummy_weights = use_dummy_weights

        # Create page table
        page_table = self._create_page_table(global_batch_size, data_parallel)

        # Create TTNN model with paged attention
        from models.tt_transformers.tt.generator import create_submeshes

        submesh_devices = create_submeshes(self.mesh_device, data_parallel)
        model_args_i, model_i, tt_kv_cache_i, _ = create_tt_model(
            submesh_devices[0],
            instruct=False,  # Not using instruct mode for now
            max_batch_size=global_batch_size // data_parallel,
            optimizations=None,  # Use default optimizations
            max_seq_len=self.config["max_seq_len"],
            paged_attention_config=self.paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=weights_to_use,  # Pass the weights (or None for dummy weights)
            dummy_weights=use_dummy_weights,  # Pass the dummy weights flag
        )
        # Ensure the active ModelArgs respects the runtime MAX_PREFILL_CHUNK_SIZE env var
        try:
            # Allow smaller chunks for debugging the hang issue
            requested = int(os.getenv("MAX_PREFILL_CHUNK_SIZE", "2048"))
            MIN_CHUNK = 32  # Allow much smaller chunks for testing
            # Round up to nearest multiple of MIN_CHUNK
            chunk = max(MIN_CHUNK, ((requested + MIN_CHUNK - 1) // MIN_CHUNK) * MIN_CHUNK)
            model_args_i.max_prefill_chunk_size = chunk
            logger.info(f"Set max_prefill_chunk_size to {chunk} (requested: {requested})")
        except Exception:
            # fallback to library defaults on error
            pass

        # Wrap in lists for compatibility
        model_args = [model_args_i]
        model = [model_i]
        tt_kv_cache = [tt_kv_cache_i]

        # Tokenizer
        tokenizer = self._init_tokenizer()
        processor = None  # Not using processor for now

        logger.info("✅ Qwen LLM initialized with MiniCPM weights")
        return model_args, model, page_table, tt_kv_cache, tokenizer, processor

    def _init_tokenizer(self):
        """Initialize Qwen tokenizer"""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)
        return tokenizer

    def _init_vision_components(self):
        """Lazy load vision encoder + resampler"""
        if self.vision_encoder is not None:
            return  # Already loaded

        logger.info("Loading vision components (SigLIP + Resampler)...")

        # Load weights
        vision_enc_weights = self.weight_bridge.get_vision_encoder_weights()
        vision_res_weights = self.weight_bridge.get_vision_resampler_weights()

        # Create components
        self.vision_encoder = TtnnSigLIPEncoder(
            mesh_device=self.mesh_device,
            weights=vision_enc_weights,
        )

        self.vision_resampler = TtnnVisionResampler(
            mesh_device=self.mesh_device,
            weights=vision_res_weights,
            num_queries=self.config["vision_num_queries"],
            embed_dim=self.config["hidden_size"],  # 3584
            kv_dim=self.config["vision_hidden_size"],  # 1152
        )

        logger.info("✅ Vision components loaded")

    def _init_audio_components(self):
        """Lazy load audio encoder + projector"""
        if self.audio_encoder is not None:
            return  # Already loaded

        logger.info("Loading audio components (Whisper + AudioProjector)...")

        # Load weights
        audio_enc_weights = self.weight_bridge.get_audio_encoder_weights()
        audio_proj_weights = self.weight_bridge.get_audio_projector_weights()

        # Create components
        self.audio_encoder = TtnnWhisperEncoder(
            mesh_device=self.mesh_device,
            weights=audio_enc_weights,
        )

        self.audio_projector = TtnnAudioProjector(
            mesh_device=self.mesh_device,
            weights=audio_proj_weights,
            output_dim=self.config["hidden_size"],  # 3584
        )

        logger.info("✅ Audio components loaded")

    def _init_tts_components(self):
        """Lazy load TTS decoder (DVAE + Vocos)"""
        if self.tts_decoder is not None:
            return  # Already loaded

        logger.info("Loading TTS components (DVAE + Vocos)...")

        # Load weights
        tts_weights = self.weight_bridge.get_tts_weights()

        # Create DVAE
        self.tts_decoder = TtnnDVAE(
            mesh_device=self.mesh_device,
            weights=tts_weights,
        )

        # Load Vocos vocoder (CPU-based, lightweight)
        try:
            from vocos import Vocos

            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        except Exception as e:
            logger.warning(f"Could not load Vocos: {e}")
            self.vocos = None

        logger.info("✅ TTS components loaded")

    def _process_vision(self, image: Image.Image) -> torch.Tensor:
        """
        Process image through vision pipeline.

        Args:
            image: PIL Image

        Returns:
            vision_tokens: (batch=1, 32, 3584) - ready to merge with text
        """
        self._init_vision_components()

        # Preprocess image
        pixel_values = self._preprocess_image(image)

        # SigLIP encoding
        vision_features = self.vision_encoder(pixel_values)  # (1, 4901, 1152)

        # Resampler (compresses to 32 tokens in Qwen space)
        vision_tokens = self.vision_resampler(vision_features)  # (1, 32, 3584)

        return vision_tokens

    def _process_audio(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Process audio through audio pipeline.

        Args:
            audio: Audio waveform (numpy array)
            sr: Sampling rate (default 16kHz for Whisper)

        Returns:
            audio_tokens: (batch=1, seq_len, 3584) - ready to merge with text
        """
        self._init_audio_components()

        # Extract mel features
        mel_features = self._extract_mel_features(audio, sr)

        # Whisper encoding
        audio_features = self.audio_encoder(mel_features)  # (1, seq_len, 1024)

        # Audio projection (to Qwen space with pooling)
        audio_tokens = self.audio_projector(audio_features)  # (1, pooled_len, 3584)

        return audio_tokens

    def _merge_embeddings(
        self,
        text: str,
        vision_tokens: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Merge multimodal embeddings by replacing text placeholders.

        This implements MiniCPM's embedding merge strategy:
        1. Tokenize text with special placeholders (<image>, <audio>)
        2. Get text embeddings from Qwen model
        3. Replace embeddings at placeholder positions with multimodal tokens
        4. Return unified embedding sequence

        Args:
            text: Input text (may contain <image> and <audio> placeholders)
            vision_tokens: Optional (batch, 32, 3584)
            audio_tokens: Optional (batch, seq, 3584)

        Returns:
            unified_embeds: (batch, total_seq_len, 3584)
        """
        import torch

        # Tokenize text (handles special tokens)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        batch_size = input_ids.shape[0]

        # Get text embeddings from Qwen model's embedding layer
        # Access the embedding layer from the first model in the generator
        text_embeds = self.qwen_generator.model[0].embd.weights.to(torch.float32)[input_ids]

        # If no multimodal inputs, return text embeddings as-is
        if vision_tokens is None and audio_tokens is None:
            return text_embeds

        # Find placeholder positions
        image_bounds = self._find_placeholder_bounds(text, "image")
        audio_bounds = self._find_placeholder_bounds(text, "audio")

        # If no placeholders found, return text embeddings
        if not image_bounds and not audio_bounds:
            logger.warning("No multimodal placeholders found in text - returning text embeddings only")
            return text_embeds

        # Build the merged embedding sequence
        merged_embeddings = []
        current_token_idx = 0

        # Sort all bounds by start position
        all_bounds = []
        for start_tok, end_tok in image_bounds:
            all_bounds.append((start_tok, end_tok, "image", vision_tokens))
        for start_tok, end_tok in audio_bounds:
            all_bounds.append((start_tok, end_tok, "audio", audio_tokens))

        all_bounds.sort(key=lambda x: x[0])  # Sort by start token position

        for start_tok, end_tok, modality, modality_tokens in all_bounds:
            # Add text embeddings before this placeholder
            if start_tok > current_token_idx:
                merged_embeddings.append(text_embeds[:, current_token_idx:start_tok, :])

            # Add multimodal tokens (ensure batch dimension matches)
            if modality_tokens is not None:
                # modality_tokens shape: (batch, seq_len, hidden_size)
                # Make sure batch dimension is compatible
                if modality_tokens.shape[0] == 1 and batch_size > 1:
                    modality_tokens = modality_tokens.repeat(batch_size, 1, 1)
                elif modality_tokens.shape[0] != batch_size:
                    logger.warning(
                        f"Batch size mismatch: modality_tokens has {modality_tokens.shape[0]}, expected {batch_size}"
                    )
                    continue

                merged_embeddings.append(modality_tokens)

            # Update current position
            current_token_idx = end_tok

        # Add any remaining text embeddings after the last placeholder
        if current_token_idx < text_embeds.shape[1]:
            merged_embeddings.append(text_embeds[:, current_token_idx:, :])

        # Concatenate all embeddings
        if merged_embeddings:
            unified_embeds = torch.cat(merged_embeddings, dim=1)
        else:
            unified_embeds = text_embeds

        logger.info(
            f"Merged embeddings: text tokens {text_embeds.shape[1]} -> unified tokens {unified_embeds.shape[1]}"
        )
        return unified_embeds

    def generate(
        self,
        text: str,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        generate_audio: bool = False,
    ) -> Dict[str, Any]:
        """
        Unified generation supporting all modalities.

        Args:
            text: Text prompt (required)
            image: Optional image file path
            audio: Optional audio file path
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            generate_audio: Whether to generate TTS output

        Returns:
            Dict with 'text' and optionally 'audio' keys
        """
        logger.info(f"Generating with modalities: text={True}, image={image is not None}, audio={audio is not None}")

        # Process multimodal inputs
        vision_tokens = None
        audio_tokens = None

        if image is not None:
            try:
                from PIL import Image

                pil_image = Image.open(image).convert("RGB")
                vision_tokens = self._process_vision(pil_image)
                logger.info(f"Processed image: {image}")
            except Exception as e:
                logger.error(f"Failed to process image {image}: {e}")
                vision_tokens = None

        if audio is not None:
            try:
                import soundfile as sf

                audio_waveform, sample_rate = sf.read(audio)
                audio_tokens = self._process_audio(audio_waveform, sample_rate)
                logger.info(f"Processed audio: {audio}")
            except Exception as e:
                logger.error(f"Failed to process audio {audio}: {e}")
                audio_tokens = None

        # Try multimodal processing
        if vision_tokens is not None or audio_tokens is not None:
            try:
                # Merge embeddings
                unified_embeds = self._merge_embeddings(text, vision_tokens, audio_tokens)
                logger.info("Multimodal embeddings merged successfully")
                # TODO: Use the merged embeddings in the generation process
                # For now, fall back to text generation since we need to modify the generation loop
                logger.warning(
                    "Multimodal embeddings merged but falling back to text generation (generation loop needs update)"
                )
            except Exception as e:
                logger.warning(f"Multimodal processing failed: {e} - falling back to text-only")
                vision_tokens = None
                audio_tokens = None

        # Prepare input for generation (text-only for now)
        import torch
        from models.tt_transformers.tt.common import preprocess_inputs_prefill
        from models.tt_transformers.tt.generator import SamplingParams

        # Tokenize input text
        input_prompts = [text]
        instruct = False
        max_seq_len = self.config["max_seq_len"]

        # Preprocess inputs
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts, self.tokenizer, self.model_args, instruct, max_tokens, max_prefill_len=max_seq_len
        )

        global_batch_size = 1  # Single user
        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        # Prefill warmup phase
        logger.info("Starting prefill warmup...")
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
        logger.info(
            f"Prefill warmup: input shape {input_tokens_prefill_pt.shape}, page_table shape {self.page_table.shape if self.page_table is not None else None}"
        )
        logger.info(f"Prefill warmup: decoding_pos={decoding_pos}, max_tokens={max_tokens}")

        import time

        warmup_start = time.time()
        logger.info(f"Calling prefill_forward_text for warmup at {warmup_start:.3f}")

        logits = self.qwen_generator.prefill_forward_text(
            input_tokens_prefill_pt,  # Prefill warmup for all users, in case some users have different seqlens than others
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=False,
        )

        warmup_end = time.time()
        logger.info(f"Prefill warmup completed: logits shape {logits.shape} (elapsed {warmup_end - warmup_start:.3f}s)")
        logger.info("Finished prefill warmup")

        # Prefill phase
        logger.info("Starting prefill...")
        print("Starting main prefill...")
        logits = self.qwen_generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=False,
        )
        print(f"Main prefill completed: logits shape {logits.shape}")
        prefilled_token = torch.argmax(logits, dim=-1)
        print(f"Prefill token: {prefilled_token}")

        # Keep track of generated outputs
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(global_batch_size)]
        for user in range(global_batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        # Decode loop
        user_done = [False] * global_batch_size
        argmax_on_device = temperature == 0  # Only support greedy decoding for now
        if argmax_on_device:
            device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        else:
            device_sampling_params = None

        current_pos = torch.tensor([decoding_pos[b] for b in range(global_batch_size)])
        iteration = 0
        users_decoding = True
        out_tok = prefilled_token

        logger.info("Starting decode loop...")

        while users_decoding and iteration < max_tokens:
            # Run decode forward
            logits = self.qwen_generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=True,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=device_sampling_params,
            )

            # Get the next token
            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)
            else:
                # For now, only support argmax
                out_tok = torch.argmax(logits, dim=-1).unsqueeze(-1)

            current_pos += 1

            # Save output token
            for user in range(global_batch_size):
                user_tok = out_tok[user].item()
                if user_tok not in self.tokenizer.stop_tokens and user_done[user] == False:
                    all_outputs[user].append(user_tok)
                else:
                    if True:  # stop_at_eos - always stop for now
                        user_done[user] = True
                        if all(user_done):
                            users_decoding = False

            iteration += 1

            # Upper limit of generated tokens
            if iteration >= max_tokens:
                users_decoding = False

        # Decode final output
        output_text = self.tokenizer.decode(all_outputs[0], skip_special_tokens=True)

        result = {"text": output_text}

        # Optional TTS generation (not implemented yet)
        if generate_audio:
            logger.warning("TTS generation not yet implemented")
            # self._init_tts_components()
            # audio_wav = self._generate_speech(output_text)
            # result['audio'] = audio_wav

        return result

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for SigLIP (980x980)"""
        # TODO: Implement proper preprocessing
        # This should resize, normalize, and convert to tensor format expected by SigLIP
        # For now, return a placeholder tensor
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.Resize((980, 980)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(image).unsqueeze(0)  # Add batch dimension

    def _extract_mel_features(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Extract mel spectrogram features for Whisper"""
        # TODO: Implement mel feature extraction
        # This should use WhisperFeatureExtractor or similar
        # For now, return a placeholder tensor
        try:
            from transformers import WhisperFeatureExtractor

            feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
            features = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
            return features.input_features
        except Exception as e:
            logger.warning(f"Could not extract mel features: {e}")
            # Return placeholder
            return torch.randn(1, 80, 3000)  # Typical Whisper input shape

    def _find_placeholder_bounds(self, text: str, placeholder: str) -> List[Tuple[int, int]]:
        """Find token positions of placeholders in tokenized text"""
        # MiniCPM uses special tags: (<image>./</image>) and (<audio>./</audio>)
        if "image" in placeholder:
            pattern = r"\(<image>\./</image>\)"
        elif "audio" in placeholder:
            pattern = r"\(<audio>\./</audio>\)"
        else:
            return []

        import re

        # Find character positions in text
        matches = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

        # Convert to token positions
        token_bounds = []
        for start_char, end_char in matches:
            # Tokenize prefix to find token position
            prefix_tokens = self.tokenizer.encode(text[:start_char], add_special_tokens=False)
            placeholder_tokens = self.tokenizer.encode(text[start_char:end_char], add_special_tokens=False)

            start_tok = len(prefix_tokens)
            end_tok = start_tok + len(placeholder_tokens)
            token_bounds.append((start_tok, end_tok))

        return token_bounds

    def _generate_speech(self, text: str) -> np.ndarray:
        """Generate speech audio from text via TTS"""
        # TODO: Implement TTS generation
        # This should use DVAE decoder + Vocos to generate audio
        # For now, return a placeholder array
        logger.warning("TTS generation not implemented")
        return np.array([])  # Empty array placeholder
