# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3TTSTool: Wraps Qwen3-TTS for text-to-speech synthesis with voice cloning.

Features:
- Voice cloning: Clone any voice from a reference audio sample
- Multi-language: Supports English, Chinese, Japanese, and more
- High quality: 24kHz audio output
- Bleed detection: Automatically trims reference audio bleed from output

N300 note: Uses TTNN for speaker embedding extraction on chip0 submesh.
Code generation uses reference implementation for shared device compatibility.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from loguru import logger

# Default reference audio/text for voice cloning
DEFAULT_REF_AUDIO = Path(__file__).parent.parent.parent.parent / "qwen3_tts" / "demo" / "jim_reference.wav"
DEFAULT_REF_CACHE = Path(__file__).parent.parent.parent.parent / "qwen3_tts" / "demo" / "jim_reference.refcache.pt"
DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. " "But you know what? You blew it! And thanks to you."
)


class Qwen3TTSTool:
    """
    Qwen3-TTS tool for text-to-speech with voice cloning.

    Uses TTNN for speaker embedding extraction.
    Uses reference implementation for code generation (shared device compatible).
    """

    def __init__(
        self,
        mesh_device,
        warmup_on_init: bool = True,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        """
        Initialize Qwen3-TTS tool.

        Args:
            mesh_device: N300 mesh device (will use chip0 submesh)
            warmup_on_init: Whether to run warmup inference on init
            ref_audio: Path to reference audio for voice cloning (uses default if None)
            ref_text: Transcript of reference audio (uses default if None)
        """
        self.mesh_device = mesh_device

        # Use the full mesh device directly to avoid submesh cleanup issues.
        # The TTNN speaker encoder operations will run on chip 0 of the mesh.
        self.device = mesh_device
        self._owns_submesh = False

        # Reference audio/text for voice cloning
        self.ref_audio = ref_audio or str(DEFAULT_REF_AUDIO)
        self.ref_text = ref_text or DEFAULT_REF_TEXT
        self.ref_cache = str(DEFAULT_REF_CACHE) if ref_audio is None else None

        # Model components
        self.tt_model = None  # TTNN model for speaker encoder
        self.tokenizer = None
        self.main_weights = None
        self.decoder_weights = None
        self.speaker_embedding = None
        self.ref_codes = None
        self.audio_data = None

        self._init_model()

        if warmup_on_init:
            self._warmup()

    def _init_model(self):
        """Initialize the Qwen3-TTS model."""
        logger.info("Loading Qwen3-TTS model...")

        # Load weights
        logger.info("[Qwen3-TTS init] Step 1: Loading model weights...")
        self.main_weights, self.decoder_weights = self._load_weights()

        # Load tokenizer
        logger.info("[Qwen3-TTS init] Step 2: Loading tokenizer...")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            trust_remote_code=True,
        )

        # Initialize TTNN model for speaker encoder
        logger.info("[Qwen3-TTS init] Step 3: Initializing TTNN speaker encoder...")
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        self.tt_model = Qwen3TTS(device=self.device, state_dict=self.main_weights)

        # Pre-compute reference encoding for default voice
        logger.info("[Qwen3-TTS init] Step 4: Encoding reference audio...")
        self._encode_reference()

        logger.info("Qwen3-TTS ready.")

    def _load_weights(self) -> Tuple[dict, dict]:
        """Load model weights from HuggingFace."""
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        model_path = Path(
            snapshot_download(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                allow_patterns=["*.safetensors"],
            )
        )

        # Main model weights - convert to float32 for reference implementation
        main_dict = {}
        for f in model_path.glob("*.safetensors"):
            if "speech_tokenizer" not in str(f):
                loaded = load_file(f)
                main_dict.update({k: v.float() for k, v in loaded.items()})
        logger.info(f"  Loaded {len(main_dict)} main weights (float32)")

        # Speech tokenizer decoder weights
        speech_path = model_path / "speech_tokenizer" / "model.safetensors"
        speech_dict = load_file(speech_path)
        decoder_weights = {k[8:]: v.float() for k, v in speech_dict.items() if k.startswith("decoder.")}
        logger.info(f"  Loaded {len(decoder_weights)} decoder weights")

        return main_dict, decoder_weights

    def _encode_reference(self):
        """Encode reference audio to codes and speaker embedding."""
        from scipy import signal

        from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward_mimi

        # Check for cached reference
        if self.ref_cache and Path(self.ref_cache).exists():
            logger.info(f"  Loading cached reference from {self.ref_cache}")
            cached = torch.load(self.ref_cache, weights_only=True)
            self.ref_codes = cached["ref_codes"]
            self.audio_data = cached["audio_data"]
        else:
            # Load and encode reference audio
            logger.info(f"  Encoding reference audio: {self.ref_audio}")
            audio_data, sr = sf.read(self.ref_audio)
            audio_data = torch.from_numpy(audio_data.astype(np.float32))
            if audio_data.dim() == 2:
                audio_data = audio_data.mean(dim=1)
            if sr != 24000:
                num_samples = int(len(audio_data) * 24000 / sr)
                audio_data = torch.from_numpy(signal.resample(audio_data.numpy(), num_samples).astype(np.float32))

            self.audio_data = audio_data

            # Encode to codes
            self.ref_codes = speech_tokenizer_encoder_forward_mimi(audio_data.unsqueeze(0))
            self.ref_codes = self.ref_codes.squeeze(0).T  # [seq_len, 16]

        # Extract speaker embedding using TTNN
        logger.info("  Extracting speaker embedding (TTNN)...")
        self.speaker_embedding = self.tt_model.extract_speaker_embedding(self.audio_data)
        # Ensure float32 for compatibility with reference implementation
        if self.speaker_embedding.dtype != torch.float32:
            self.speaker_embedding = self.speaker_embedding.float()
        logger.info(f"  Speaker embedding: {self.speaker_embedding.shape} ({self.speaker_embedding.dtype})")

    def _warmup(self):
        """Run a warmup to initialize any lazy components."""
        logger.info("[Qwen3-TTS init] Step 5: Warmup...")
        # No warmup needed for reference generation
        logger.info("  Warmup complete.")

    def synthesize(
        self,
        text: str,
        output_path: str = "/tmp/response.wav",
        language: str = "english",
        max_new_tokens: int = 256,
        auto_trim_bleed: bool = True,
    ) -> str:
        """
        Convert text to speech and save to output_path.

        Args:
            text: Text to synthesize
            output_path: Path to save the output .wav file
            language: Language of the text (english, chinese, japanese, etc.)
            max_new_tokens: Maximum number of codec frames to generate
            auto_trim_bleed: Whether to auto-detect and trim reference audio bleed

        Returns:
            Path to the saved .wav file
        """
        logger.info(f"Synthesizing: '{text[:50]}...' ({language})")

        # Use reference implementation for generation (shared device compatible)
        from models.demos.qwen3_tts.demo.demo_pure_reference_tts import (
            TTSConfig,
            create_icl_embedding,
            decode_audio,
            generate_codes,
        )
        from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning

        config = TTSConfig()
        config.max_new_tokens = max_new_tokens
        config.greedy = False
        config.temperature = 0.9
        config.top_k = 50

        # Get language ID
        lang_lower = language.lower()
        if lang_lower not in config.codec_language_ids:
            logger.warning(f"Unknown language '{language}', defaulting to english")
            lang_lower = "english"

        # Trim reference for ICL conditioning
        ref_codes, audio_data = trim_reference_for_icl_conditioning(
            self.ref_codes.clone(),
            self.audio_data.clone(),
            self.tokenizer,
            self.ref_text,
            text,
        )

        # Create ICL embeddings using reference implementation
        inputs_embeds, trailing_text_hidden, tts_pad_embed = create_icl_embedding(
            target_text=text,
            ref_text=self.ref_text,
            ref_codes=ref_codes,
            tokenizer=self.tokenizer,
            weights=self.main_weights,
            config=config,
            speaker_embedding=self.speaker_embedding,
            language=lang_lower,
        )

        # Generate codes using reference implementation (CPU)
        codes = generate_codes(
            inputs_embeds=inputs_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            weights=self.main_weights,
            config=config,
        )

        if codes is None or len(codes) == 0:
            logger.warning("Generation produced no codes (EOS at start)")
            # Create a short silent audio
            audio_np = np.zeros(24000, dtype=np.float32)  # 1 second of silence
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio_np, samplerate=24000)
            return output_path

        # Decode to audio
        audio = decode_audio(codes, self.decoder_weights)

        # Save audio
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        audio_np = audio.squeeze().cpu().float().numpy()
        sf.write(output_path, audio_np, samplerate=24000)

        # Auto-trim reference audio bleed if enabled
        if auto_trim_bleed:
            try:
                from models.demos.qwen3_tts.demo.bleed_detector import auto_trim_bleed as trim_bleed

                # Extract first word of target text for bleed detection
                first_word = text.split()[0].rstrip(",.!?;:") if text.split() else "Hello"
                output_path, bleed_results = trim_bleed(
                    audio_path=output_path,
                    output_path=output_path,
                    target_first_word=first_word,
                    margin_seconds=0.1,
                )
                if bleed_results.get("trimmed", False):
                    logger.info(
                        f"Trimmed {bleed_results['trim_seconds']:.2f}s bleed "
                        f"(content: '{bleed_results.get('bleed_content', '')}')"
                    )
            except Exception as e:
                logger.warning(f"Bleed detection failed: {e}")

        # Re-read to get final duration
        audio_data_final, _ = sf.read(output_path)
        duration = len(audio_data_final) / 24000
        logger.info(f"Audio saved to {output_path} ({duration:.2f}s)")

        return output_path

    def clone_voice(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        output_path: str = "/tmp/cloned_voice.wav",
        language: str = "english",
    ) -> str:
        """
        Clone a voice from reference audio and synthesize text.

        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio file
            ref_text: Transcript of the reference audio
            output_path: Path to save the output .wav file
            language: Language of the text

        Returns:
            Path to the saved .wav file
        """
        # Temporarily update reference
        old_ref_audio = self.ref_audio
        old_ref_text = self.ref_text
        old_ref_codes = self.ref_codes
        old_audio_data = self.audio_data
        old_speaker_embedding = self.speaker_embedding

        try:
            self.ref_audio = ref_audio
            self.ref_text = ref_text
            self.ref_cache = None  # Don't use cache for new voice
            self._encode_reference()

            return self.synthesize(text, output_path, language)
        finally:
            # Restore original reference
            self.ref_audio = old_ref_audio
            self.ref_text = old_ref_text
            self.ref_codes = old_ref_codes
            self.audio_data = old_audio_data
            self.speaker_embedding = old_speaker_embedding

    def close(self):
        """Release model resources."""
        # Clear TTNN model first - it holds device references
        if self.tt_model is not None:
            if hasattr(self.tt_model, "close"):
                try:
                    self.tt_model.close()
                except Exception:
                    pass
            self.tt_model = None

        # Clear all other attributes
        self.tokenizer = None
        self.main_weights = None
        self.decoder_weights = None
        self.speaker_embedding = None
        self.ref_codes = None
        self.audio_data = None
        self.device = None

        logger.info("Qwen3TTSTool closed.")
