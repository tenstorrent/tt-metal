#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Binary-search the minimum l1_small_size that allows SpeechT5 trace mode
to complete warmup + one synthesis call.

Run as: python probe_speecht5_l1.py [l1_small_size]
Default: 300000 (known-good from demo).
"""
import sys

sys.path.insert(0, "/home/ubuntu/agentic/tt-metal")

from loguru import logger

import ttnn

L1 = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

logger.info(f"Testing SpeechT5 trace mode with l1_small_size={L1}, max_steps={MAX_STEPS}")

device = ttnn.open_device(
    device_id=0,
    l1_small_size=L1,
    trace_region_size=10_000_000,
    num_command_queues=2,
)
device.enable_program_cache()

import torch
from datasets import load_dataset
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from models.experimental.speecht5_tts.demo_ttnn import generate_speech_fp32
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
from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import SpeechT5Generator
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNPostNetConfig,
    TTNNSpeechT5SpeechDecoderPostnet,
    preprocess_postnet_parameters,
)

SPEECHT5_MODEL_ID = "microsoft/speecht5_tts"
HIFIGAN_MODEL_ID = "microsoft/speecht5_hifigan"
processor = SpeechT5Processor.from_pretrained(SPEECHT5_MODEL_ID)
hf_model = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_MODEL_ID)
vocoder = SpeechT5HifiGan.from_pretrained(HIFIGAN_MODEL_ID)
hf_model.eval()

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

decoder_config = TTNNDecoderConfig(
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
encoder_config = TTNNEncoderConfig(
    vocab_size=hf_model.config.vocab_size,
    hidden_size=hf_model.config.hidden_size,
    num_layers=hf_model.config.encoder_layers,
    num_heads=hf_model.config.encoder_attention_heads,
    ffn_dim=hf_model.config.encoder_ffn_dim,
    max_position_embeddings=hf_model.config.max_length,
    layer_norm_eps=hf_model.config.layer_norm_eps,
)
postnet_config = TTNNPostNetConfig(
    postnet_units=hf_model.config.speech_decoder_postnet_units,
    postnet_layers=hf_model.config.speech_decoder_postnet_layers,
    postnet_kernel=hf_model.config.speech_decoder_postnet_kernel,
    postnet_dropout=0.5,
    num_mel_bins=hf_model.config.num_mel_bins,
    reduction_factor=hf_model.config.reduction_factor,
)

encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, device)
ttnn_encoder = TTNNSpeechT5Encoder(device, encoder_params, encoder_config)
decoder_params = preprocess_decoder_parameters(hf_model.speecht5.decoder, decoder_config, device, speaker_embeddings)
ttnn_decoder = TTNNSpeechT5Decoder(device, decoder_params, decoder_config, max_sequence_length=512)
postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, device)
ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(device, postnet_params, postnet_config)

generator = SpeechT5Generator(
    encoder=ttnn_encoder,
    decoder=ttnn_decoder,
    postnet=ttnn_postnet,
    device=device,
    decoder_config=decoder_config,
    max_steps=MAX_STEPS,
    max_batch_size=1,
    encoder_seq_len=128,
)

logger.info("Running warmup with trace mode...")
generate_speech_fp32(
    "warmup",
    speaker_embeddings,
    processor,
    vocoder,
    ttnn_encoder,
    ttnn_decoder,
    ttnn_postnet,
    device,
    max_steps=MAX_STEPS,
    warmup_mode=True,
    generator=generator,
    use_kv_cache=True,
    decoder_config=decoder_config,
)
generator._reset_kv_caches()

logger.info("Running synthesis...")
import tempfile
import time

import soundfile as sf

t0 = time.time()
speech = generate_speech_fp32(
    "Hello from Tenstorrent hardware.",
    speaker_embeddings,
    processor,
    vocoder,
    ttnn_encoder,
    ttnn_decoder,
    ttnn_postnet,
    device,
    max_steps=MAX_STEPS,
    warmup_mode=False,
    generator=generator,
    use_kv_cache=True,
    decoder_config=decoder_config,
)
elapsed = time.time() - t0
if speech is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, speech.numpy(), samplerate=16000)
    logger.info(f"PASS: l1_small_size={L1}  synthesis took {elapsed:.1f}s  audio={len(speech)/16000:.2f}s")
else:
    logger.warning(f"PASS (no audio): l1_small_size={L1}  time={elapsed:.1f}s")

ttnn.close_device(device)
