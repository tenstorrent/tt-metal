# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5Tool: Wraps TTNN SpeechT5 TTS for text-to-speech synthesis.

Uses the single-device SpeechT5 implementation from demo_ttnn.py.
N300 note: SpeechT5 is a single-chip model — runs on device[0] of the N300.
"""

from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset
from loguru import logger
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from models.experimental.speecht5_tts.demo_ttnn import (
    DEFAULT_CHUNK_SIZE,
    generate_speech_fp32,
    generate_speech_long_text,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNDecoderConfig,
    TTNNSpeechT5Decoder,
    preprocess_decoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNEncoderConfig,
    TTNNSpeechT5Encoder,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNPostNetConfig,
    TTNNSpeechT5SpeechDecoderPostnet,
    preprocess_postnet_parameters,
)

SPEECHT5_MODEL_ID = "microsoft/speecht5_tts"
HIFIGAN_MODEL_ID = "microsoft/speecht5_hifigan"
SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"
SPEAKER_IDX = 7306
MAX_STEPS = 200


class SpeechT5Tool:
    """
    TTNN-accelerated SpeechT5 TTS wrapper.

    Synthesizes speech from text and saves to a .wav file.
    Runs on a single chip (device[0]) of the N300.
    """

    def __init__(self, mesh_device, warmup_on_init: bool = True):
        self.device = mesh_device

        logger.info("Loading SpeechT5 TTS model...")
        logger.info("[SpeechT5 init] Step 1: load processor")
        self.processor = SpeechT5Processor.from_pretrained(SPEECHT5_MODEL_ID)
        logger.info("[SpeechT5 init] Step 2: load HF SpeechT5 model")
        hf_model = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_MODEL_ID)
        logger.info("[SpeechT5 init] Step 3: load HiFiGAN vocoder")
        self.vocoder = SpeechT5HifiGan.from_pretrained(HIFIGAN_MODEL_ID)
        hf_model.eval()

        logger.info("[SpeechT5 init] Step 4: load speaker embeddings dataset")
        embeddings_dataset = load_dataset(SPEAKER_DATASET, split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[SPEAKER_IDX]["xvector"]).unsqueeze(0)
        logger.info("[SpeechT5 init] Step 5: build TT config objects")

        encoder_config = TTNNEncoderConfig(
            vocab_size=hf_model.config.vocab_size,
            hidden_size=hf_model.config.hidden_size,
            num_layers=hf_model.config.encoder_layers,
            num_heads=hf_model.config.encoder_attention_heads,
            ffn_dim=hf_model.config.encoder_ffn_dim,
            max_position_embeddings=hf_model.config.max_length,
            layer_norm_eps=hf_model.config.layer_norm_eps,
        )
        self.decoder_config = TTNNDecoderConfig(
            hidden_size=hf_model.config.hidden_size,
            num_layers=hf_model.config.decoder_layers,
            num_heads=hf_model.config.decoder_attention_heads,
            ffn_dim=hf_model.config.decoder_ffn_dim,
            max_position_embeddings=hf_model.config.max_length,
            layer_norm_eps=hf_model.config.layer_norm_eps,
            num_mel_bins=hf_model.config.num_mel_bins,
            reduction_factor=hf_model.config.reduction_factor,
            speech_decoder_prenet_units=hf_model.config.speech_decoder_prenet_units,
            speech_decoder_prenet_layers=hf_model.config.speech_decoder_prenet_layers,
            speech_decoder_prenet_dropout=0.5,
            speaker_embedding_dim=hf_model.config.speaker_embedding_dim,
            use_fp32=True,
        )
        postnet_config = TTNNPostNetConfig(
            postnet_units=hf_model.config.speech_decoder_postnet_units,
            postnet_layers=hf_model.config.speech_decoder_postnet_layers,
            postnet_kernel=hf_model.config.speech_decoder_postnet_kernel,
            postnet_dropout=0.5,
            num_mel_bins=hf_model.config.num_mel_bins,
            reduction_factor=hf_model.config.reduction_factor,
        )

        logger.info("[SpeechT5 init] Step 6: preprocess encoder parameters")
        encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, self.device)
        logger.info("[SpeechT5 init] Step 7: create TT encoder")
        self.ttnn_encoder = TTNNSpeechT5Encoder(self.device, encoder_params, encoder_config)
        logger.info("[SpeechT5 init] Step 8: preprocess decoder parameters")
        decoder_params = preprocess_decoder_parameters(
            hf_model.speecht5.decoder, self.decoder_config, self.device, self.speaker_embeddings
        )
        logger.info("[SpeechT5 init] Step 9: create TT decoder")
        self.ttnn_decoder = TTNNSpeechT5Decoder(
            self.device, decoder_params, self.decoder_config, max_sequence_length=512
        )
        logger.info("[SpeechT5 init] Step 10: preprocess postnet parameters")
        postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, self.device)
        logger.info("[SpeechT5 init] Step 11: create TT postnet")
        self.ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(self.device, postnet_params, postnet_config)

        # NOTE: generator (trace+2CQ mode) requires l1_small_size=300_000 which
        # clashes with Whisper's L1 buffers on a shared device.  We run SpeechT5
        # with plain KV-cache mode (generator=None) which works correctly with the
        # shared l1_small_size=24_576 and does not use the 2CQ async path.
        self.generator = None

        if warmup_on_init:
            logger.info("[SpeechT5 init] Step 12: warmup inference start")
            self._warmup()
            logger.info("[SpeechT5 init] Step 13: warmup inference end")
        else:
            logger.info("[SpeechT5 init] Step 12: warmup deferred (warmup_on_init=False)")
        logger.info("SpeechT5 ready.")

    def close(self):
        """
        Drop references to TTNN modules and host-side objects so device memory can be reclaimed.

        Use before running Whisper trace capture on a shared mesh if SpeechT5 residency on chip0
        causes stalls; reload SpeechT5 afterward with a new SpeechT5Tool(...).
        """
        for name in ("ttnn_encoder", "ttnn_decoder", "ttnn_postnet"):
            setattr(self, name, None)
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.generator = None
        self.device = None
        logger.info("SpeechT5Tool.close(): device tensors released (refs cleared)")

    def _warmup(self):
        """Run a warmup inference to compile all kernels (KV-cache, no trace)."""
        generate_speech_fp32(
            "warmup",
            self.speaker_embeddings,
            self.processor,
            self.vocoder,
            self.ttnn_encoder,
            self.ttnn_decoder,
            self.ttnn_postnet,
            self.device,
            max_steps=MAX_STEPS,
            warmup_mode=True,
            generator=None,  # no trace — avoids 2CQ and l1_small_size clash
            use_kv_cache=True,
            decoder_config=self.decoder_config,
        )

    def synthesize(self, text: str, output_path: str = "/tmp/response.wav") -> str:
        """
        Convert text to speech and save to output_path.

        Returns the path to the saved .wav file.
        """
        speech = generate_speech_long_text(
            text=text,
            speaker_embeddings=self.speaker_embeddings,
            processor=self.processor,
            vocoder=self.vocoder,
            ttnn_encoder=self.ttnn_encoder,
            ttnn_decoder=self.ttnn_decoder,
            ttnn_postnet=self.ttnn_postnet,
            device=self.device,
            max_steps=MAX_STEPS,
            generator=None,  # no trace — shared l1_small_size=24_576 compatibility
            use_kv_cache=True,
            decoder_config=self.decoder_config,
            max_chunk_size=DEFAULT_CHUNK_SIZE,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, speech.numpy(), samplerate=16000)
        logger.info(f"Audio saved to {output_path}")
        return output_path

    def release_after_inference(self):
        """
        Optional aggressive cleanup hook.

        If trace mode is enabled in the future (self.generator is not None), this
        releases persistent trace/L1 artifacts so other workloads can reclaim L1.
        """
        if self.generator is not None and hasattr(self.generator, "cleanup_aggressive"):
            self.generator.cleanup_aggressive()
