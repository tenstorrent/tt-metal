# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-3B Text-to-Speech Demo on Tenstorrent Hardware

This demo runs the Llasa-3B TTS model using the existing tt_transformers infrastructure.
Llasa-3B is architecturally identical to LLaMA-3.2-3B but with an extended vocabulary
that includes 65,536 XCodec2 speech tokens.

Usage:
    # Set environment variable for the model
    export HF_MODEL=HKUSTAudio/Llasa-3B

    # Run zero-shot TTS
    pytest models/demos/llasa3b/demo/llasa_demo.py -k "test_llasa_tts" -s

    # Run prompted TTS (voice cloning)
    pytest models/demos/llasa3b/demo/llasa_demo.py -k "test_llasa_tts_prompted" -s
"""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.llasa3b.tt.llasa_pipeline import prepare_llasa_generator, run_llasa_tts
from models.demos.llasa3b.tt.llasa_utils import decode_speech_to_audio, encode_prompt_audio
from models.tt_transformers.tt.model_config import DecodersPrecision

# ============================================================================
# Test Configuration & Pytest Entry Point
# ============================================================================


def load_input_prompts(prompts_file=None):
    """Load input prompts from JSON file.

    Args:
        prompts_file: Path to JSON file with list of {lang, text} entries.
            Defaults to the bundled input_data.json.

    Returns:
        List of prompt dicts.
    """
    if prompts_file is None:
        prompts_file = Path(__file__).parent / "input_data.json"
    with open(prompts_file, "r") as f:
        return json.load(f)


SAMPLE_INPUTS = load_input_prompts()

# Shared pytest device/optimization parameters
DEVICE_PARAMS = [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 2}]
MESH_SHAPE = [
    {
        "N150": (1, 1),
        "N300": (1, 2),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
]
OPTIMIZATIONS = [
    lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
]


# ── Zero-Shot TTS ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "index, input_text",
    [(i, entry["text"]) for i, entry in enumerate(SAMPLE_INPUTS)],
    ids=[f"{entry['lang']}-{i}" for i, entry in enumerate(SAMPLE_INPUTS)],
)
@pytest.mark.parametrize("max_generated_tokens", [500])
@pytest.mark.parametrize("optimizations", OPTIMIZATIONS, ids=["performance"])
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_SHAPE, indirect=True)
def test_llasa_zero_shot(
    index,
    input_text,
    max_generated_tokens,
    optimizations,
    mesh_device,
):
    """
    End-to-end Llasa-3B zero-shot TTS demo on Tenstorrent hardware.

    Pipeline: text → TTNN inference → speech tokens → XCodec2 → WAV
    """
    logger.info("=== Llasa-3B TTS Demo (Zero-Shot) ===")
    logger.info(f"Input: '{input_text}'")

    # 1. Create model and generator
    model_args, generator, tt_kv_cache, paged_attention_config = prepare_llasa_generator(mesh_device, optimizations)

    # 2. Run the TTS pipeline
    speech_ids, metrics = run_llasa_tts(
        generator, model_args, tt_kv_cache, paged_attention_config, input_text, max_generated_tokens
    )

    # 3. Decode speech tokens to audio
    if speech_ids:
        output_dir = "models/demos/llasa3b/demo/output"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"llasa_output_{index}.wav"
        output_path = os.path.join(output_dir, output_filename)
        result = decode_speech_to_audio(speech_ids, output_path=output_path)

        if result:
            logger.info(f"Audio saved to {result}")
    else:
        logger.warning("No speech tokens found in generated output!")

    logger.info("=== Demo Complete ===")


# ── Prompted TTS (Voice Cloning) ──────────────────────────────

# Official Anna.wav prompt from HuggingFace
ANNA_PROMPT_TEXT = (
    "A chance to leave him alone, but... No. She just wanted to see him again. "
    "Anna, you don't know how it feels to lose a sister. "
    "Anna, I'm sorry, but your father asked me not to tell you anything."
)
ANNA_TARGET_TEXT = (
    "Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, "
    "intending to safeguard some from the harsh truths. One day, I hope you understand the reasons "
    "behind my actions. Until then, Anna, please, bear with me."
)


def download_anna_wav():
    """Download the official Anna.wav prompt from HuggingFace if not already cached."""
    cache_dir = Path("models/demos/llasa3b/demo/prompts")
    wav_path = cache_dir / "Anna.wav"
    if wav_path.exists():
        logger.info(f"Using cached prompt: {wav_path}")
        return str(wav_path)

    logger.info("Downloading Anna.wav from HuggingFace...")
    from huggingface_hub import hf_hub_download

    hf_hub_download("HKUSTAudio/Llasa-3B", "Anna.wav", local_dir=str(cache_dir))
    logger.info(f"Downloaded to {wav_path}")
    return str(wav_path)


@pytest.mark.parametrize("max_generated_tokens", [1000])
@pytest.mark.parametrize("optimizations", OPTIMIZATIONS, ids=["performance"])
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_SHAPE, indirect=True)
def test_llasa_voice_cloning(
    max_generated_tokens,
    optimizations,
    mesh_device,
):
    """
    End-to-end Llasa-3B prompted TTS (voice cloning) demo on Tenstorrent hardware.

    Pipeline: prompt WAV → XCodec2 encode → TTNN inference → speech tokens → XCodec2 decode → WAV
    Uses the official Anna.wav prompt from the Llasa-3B HuggingFace repo.
    """
    logger.info("=== Llasa-3B TTS Demo (Prompted / Voice Cloning) ===")
    logger.info(f"Prompt text: '{ANNA_PROMPT_TEXT[:80]}...'")
    logger.info(f"Target text: '{ANNA_TARGET_TEXT[:80]}...'")

    # 1. Download and encode prompt audio (CPU)
    wav_path = download_anna_wav()
    prompt_speech_tokens = encode_prompt_audio(wav_path)
    if prompt_speech_tokens is None:
        pytest.skip("XCodec2 encoder not available — skipping prompted TTS test")

    logger.info(f"Prompt encoded: {len(prompt_speech_tokens)} speech tokens")

    # 2. Create model and generator
    model_args, generator, tt_kv_cache, paged_attention_config = prepare_llasa_generator(mesh_device, optimizations)

    # 3. Run the TTS pipeline with prompt
    speech_ids, metrics = run_llasa_tts(
        generator,
        model_args,
        tt_kv_cache,
        paged_attention_config,
        ANNA_TARGET_TEXT,
        max_generated_tokens,
        prompt_text=ANNA_PROMPT_TEXT,
        prompt_speech_tokens=prompt_speech_tokens,
    )

    # 4. Decode speech tokens to audio
    if speech_ids:
        output_dir = "models/demos/llasa3b/demo/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "llasa_output_prompted.wav")
        result = decode_speech_to_audio(speech_ids, output_path=output_path)
        if result:
            logger.info(f"Voice-cloned audio saved to {result}")
    else:
        logger.warning("No speech tokens found in generated output!")

    logger.info("=== Demo Complete ===")
