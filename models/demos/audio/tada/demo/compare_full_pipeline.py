#!/usr/bin/env python3
"""
Full CPU reference pipeline for TADA 1B TTS.

Runs the EXACT same AR loop as tada_generator.py but entirely on CPU in float32.
This definitively isolates code bugs from model limitations by eliminating
any TTNN/bfloat16 precision effects.

Usage:
    python3 models/demos/audio/tada/demo/compare_full_pipeline.py

If CPU float32 produces good speech → TTNN has a precision or implementation bug.
If CPU float32 produces same bad speech → model doesn't support unconditional generation.
"""

import math
import os
import re
import sys

# Ensure tt-metal root and tada source are on the path
_TT_METAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, _TT_METAL_ROOT)
sys.path.insert(0, "/home/ttuser/atupe/tada/tada")

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

TADA_MODEL_PATH = os.environ.get("TADA_MODEL_PATH", "/home/ttuser/atupe/tada/tada-1b")
TADA_CODEC_PATH = os.environ.get("TADA_CODEC_PATH", "/home/ttuser/atupe/tada/tada-codec")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_RATE = 24000

HIDDEN_SIZE = 2048
ACOUSTIC_DIM = 512
NUM_TIME_BITS = 8
TIME_DIM = 2 * NUM_TIME_BITS  # 16
LATENT_SIZE = ACOUSTIC_DIM + TIME_DIM  # 528
SHIFT_ACOUSTIC = 5
ACOUSTIC_STD = 1.5
ACOUSTIC_MEAN = 0.0
NUM_TIME_CLASSES = 256
VOCAB_SIZE = 128256
NUM_EOS_TOKENS = SHIFT_ACOUSTIC

GENERATION_TEXT = "This is a test of text to speech on Tenstorrent hardware."
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Gray code utilities (from tada/utils/gray_code.py)
# ---------------------------------------------------------------------------


def _gray_code_to_int(gray):
    binary = gray
    shift = 1
    while shift < 32:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary


def decode_gray_code_to_time(gray_bits, num_bits):
    gray_bits_binary = ((gray_bits + 1.0) / 2.0).round().long()
    gray_code = torch.zeros(*gray_bits_binary.shape[:-1], dtype=torch.long)
    for i in range(num_bits):
        gray_code += gray_bits_binary[..., num_bits - 1 - i] << i
    return _gray_code_to_int(gray_code)


# ---------------------------------------------------------------------------
# Text normalization (from tada_generator.py)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    substitutions = {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2026": "...",
        "\u2039": "<",
        "\u203a": ">",
        "\u00ab": "<<",
        "\u00bb": ">>",
    }
    pattern = re.compile("|".join(re.escape(char) for char in substitutions))
    text = pattern.sub(lambda m: substitutions[m.group(0)], text)
    text = (
        text.replace("; ", ". ")
        .replace('"', "")
        .replace(":", ",")
        .replace("(", "")
        .replace(")", "")
        .replace("--", "-")
        .replace("-", ", ")
        .replace(",,", ",")
        .replace(" '", " ")
        .replace("' ", " ")
        .replace("  ", " ")
    )
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    text = re.sub(r"([.!?]\s*)(\w)", lambda m: m.group(1) + m.group(2).upper(), text.lower())
    if text:
        text = text[0].upper() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Time & CFG schedules
# ---------------------------------------------------------------------------


def build_time_schedule(num_steps, schedule):
    if schedule == "logsnr":
        log_snr = torch.linspace(5.0, -5.0, num_steps + 1)
        t_span = torch.sigmoid(-log_snr / 2)
        t_span[0] = 0.0
        t_span[-1] = 1.0
        return t_span
    return torch.linspace(0, 1, num_steps + 1)


def scheduled_cfg(base_scale, t, schedule):
    if schedule == "constant" or base_scale == 1.0:
        return base_scale
    if schedule == "cosine":
        return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))
    return base_scale


# ---------------------------------------------------------------------------
# Text token sampling (matches tada_generator.py exactly)
# ---------------------------------------------------------------------------


def sample_text_token(logits, input_ids, pad_token_id, temperature=0.6, top_p=0.9, repetition_penalty=1.1):
    token_logits = logits.clone()
    token_logits[:, pad_token_id] = float("-inf")

    # Repetition penalty
    if repetition_penalty != 1.0:
        score = torch.gather(token_logits, 1, input_ids)
        score = torch.where(
            score < 0,
            score * repetition_penalty,
            score / repetition_penalty,
        )
        token_logits = token_logits.scatter(1, input_ids, score)

    # Temperature
    token_logits = token_logits / temperature

    # Top-p (nucleus)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(token_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        token_logits = token_logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_llama_and_tada_cpu(model_path):
    """Load Llama model + TADA embedding layers + LM head on CPU float32."""
    import json

    from transformers import LlamaConfig, LlamaModel

    with open(os.path.join(model_path, "config.json")) as f:
        config_dict = json.load(f)

    llama_config = LlamaConfig(
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        intermediate_size=config_dict["intermediate_size"],
        max_position_embeddings=config_dict.get("max_position_embeddings", 131072),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
        rope_theta=config_dict.get("rope_theta", 500000.0),
    )

    model = LlamaModel(llama_config)
    weights = safetensors_load_file(os.path.join(model_path, "model.safetensors"))

    model_state = {}
    for k, v in weights.items():
        if k.startswith("model."):
            model_state[k[6:]] = v.float()

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    logger.info(f"Llama: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    # TADA embedding layers
    embed_tokens = model.embed_tokens

    acoustic_proj = nn.Linear(ACOUSTIC_DIM, HIDDEN_SIZE, bias=True)
    acoustic_proj.weight = nn.Parameter(weights["acoustic_proj.weight"].float())
    acoustic_proj.bias = nn.Parameter(weights["acoustic_proj.bias"].float())

    acoustic_mask_emb = nn.Embedding(2, HIDDEN_SIZE)
    acoustic_mask_emb.weight = nn.Parameter(weights["acoustic_mask_emb.weight"].float())

    time_start_embed = nn.Embedding(NUM_TIME_CLASSES, HIDDEN_SIZE)
    time_start_embed.weight = nn.Parameter(weights["time_start_embed.weight"].float())

    time_end_embed = nn.Embedding(NUM_TIME_CLASSES, HIDDEN_SIZE)
    time_end_embed.weight = nn.Parameter(weights["time_end_embed.weight"].float())

    # LM head (may be tied to embed_tokens)
    if "lm_head.weight" in weights:
        lm_head_w = weights["lm_head.weight"].float()
    else:
        lm_head_w = weights["model.embed_tokens.weight"].float()
    lm_head = nn.Linear(HIDDEN_SIZE, llama_config.vocab_size, bias=False)
    lm_head.weight = nn.Parameter(lm_head_w)
    lm_head.eval()

    # Bottleneck projection (identity for 1B, present for 3B)
    bottleneck_proj = None
    if "bottleneck_proj.weight" in weights:
        bp_w = weights["bottleneck_proj.weight"].float()
        bottleneck_proj = nn.Linear(bp_w.shape[1], bp_w.shape[0], bias=False)
        bottleneck_proj.weight = nn.Parameter(bp_w)
        bottleneck_proj.eval()
        logger.info(f"Loaded bottleneck_proj: {bp_w.shape}")

    modules = {
        "embed_tokens": embed_tokens,
        "acoustic_proj": acoustic_proj,
        "acoustic_mask_emb": acoustic_mask_emb,
        "time_start_embed": time_start_embed,
        "time_end_embed": time_end_embed,
        "lm_head": lm_head,
        "bottleneck_proj": bottleneck_proj,
    }

    return model, modules


def load_vibevoice_cpu(model_path):
    """Load VibeVoice diffusion head on CPU."""
    sys.path.insert(0, "/home/ttuser/atupe/tada/tada")
    import json

    from tada.nn.vibevoice import VibeVoiceDiffusionHead, VibeVoiceDiffusionHeadConfig

    weights = safetensors_load_file(os.path.join(model_path, "model.safetensors"))

    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    vv_config = VibeVoiceDiffusionHeadConfig(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        head_layers=config.get("prediction_head_num_layers", 6),
        head_ffn_ratio=config.get("prediction_head_ffn_ratio", 4.0),
    )
    vv = VibeVoiceDiffusionHead(vv_config)

    vv_state = {}
    for k, v in weights.items():
        if k.startswith("prediction_head."):
            vv_state[k[len("prediction_head.") :]] = v.float()
    missing, unexpected = vv.load_state_dict(vv_state, strict=False)
    if missing:
        logger.warning(f"Missing VibeVoice keys: {missing}")

    vv.eval()
    return vv


def load_decoder_cpu(codec_path):
    """Load the full Decoder (local attention + DAC CNN) on CPU."""
    from pathlib import Path

    from models.demos.audio.tada.reference.tada_reference import Decoder as RefDecoder

    dec_path = os.path.join(codec_path, "decoder")
    state_dict = {}
    for sf_file in sorted(Path(dec_path).glob("*.safetensors")):
        state_dict.update(safetensors_load_file(str(sf_file)))

    decoder = RefDecoder()

    # Build state dict with correct key mapping
    dec_state = {}
    for k, v in state_dict.items():
        # Skip precomputed masks and rope freqs
        if "_precomputed_mask" in k or "rope_freqs" in k:
            continue
        dec_state[k] = v.float()

    missing, unexpected = decoder.load_state_dict(dec_state, strict=False)
    logger.info(f"Decoder: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        logger.info(f"  Missing: {missing[:10]}...")

    decoder.eval()
    return decoder


# ---------------------------------------------------------------------------
# Building inputs_embeds (matches tada_generator.py)
# ---------------------------------------------------------------------------


def build_inputs_embeds(input_id, acoustic, mask, t_before, t_after, modules):
    """Build single-step inputs_embeds on CPU. All inputs are 1D (batch,)."""
    token_emb = modules["embed_tokens"](input_id.unsqueeze(1))  # (B, 1, H)
    ac_emb = modules["acoustic_proj"](acoustic.unsqueeze(1))  # (B, 1, H)
    mask_emb = modules["acoustic_mask_emb"](mask.unsqueeze(1))  # (B, 1, H)
    t_start = modules["time_start_embed"](t_before.unsqueeze(1))  # (B, 1, H)
    t_end = modules["time_end_embed"](t_after.unsqueeze(1))  # (B, 1, H)
    return token_emb + ac_emb + mask_emb + t_start + t_end


# ---------------------------------------------------------------------------
# Flow matching ODE (matches tada_generator._solve_flow_matching)
# ---------------------------------------------------------------------------


def solve_flow_matching_cpu(
    vv_head,
    cond,
    neg_cond,
    bottleneck_proj=None,
    num_steps=20,
    noise_temp=0.9,
    acoustic_cfg_scale=1.6,
    duration_cfg_scale=1.0,
    seed=None,
):
    """
    Euler ODE solver on CPU. Matches tada_generator._solve_flow_matching exactly.

    Args:
        cond: (B, H) positive conditioning
        neg_cond: (B, H) negative conditioning
        bottleneck_proj: optional projection layer
        seed: random seed for noise initialization (None = no seeding)
    """
    B = cond.shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    speech = torch.randn(B, LATENT_SIZE) * noise_temp
    t_span = build_time_schedule(num_steps, "logsnr")
    t_curr = t_span[0]
    use_cfg = acoustic_cfg_scale != 1.0

    # Apply bottleneck projection if present
    if bottleneck_proj is not None:
        cond_proj = bottleneck_proj(cond)
        neg_cond_proj = bottleneck_proj(neg_cond)
    else:
        cond_proj = cond
        neg_cond_proj = neg_cond

    for i in range(1, len(t_span)):
        dt = t_span[i] - t_curr
        t_val = t_curr.item()
        a_cfg = scheduled_cfg(acoustic_cfg_scale, t_val, "cosine")
        d_cfg = scheduled_cfg(duration_cfg_scale, t_val, "cosine")
        t_torch = t_curr.expand(B)

        if use_cfg:
            speech_doubled = torch.cat([speech, speech], dim=0)
            t_doubled = t_torch.repeat(2)
            cond_combined = torch.cat([cond_proj, neg_cond_proj], dim=0)

            # VibeVoice expects (B, latent_size) for speech, (B,) for t, (B, 1, H) for condition
            velocity_combined = vv_head(speech_doubled, t_doubled, condition=cond_combined)
            vel_pos = velocity_combined[:B]
            vel_neg = velocity_combined[B:]

            velocity = torch.cat(
                [
                    (vel_neg + a_cfg * (vel_pos - vel_neg))[..., :ACOUSTIC_DIM],
                    (vel_neg + d_cfg * (vel_pos - vel_neg))[..., ACOUSTIC_DIM:],
                ],
                dim=-1,
            )
        else:
            velocity = vv_head(speech, t_torch, condition=cond_proj)

        speech = speech + dt * velocity
        t_curr = t_span[i]

    return speech


# ---------------------------------------------------------------------------
# Build input token sequence (matches tada_generator._build_input_ids)
# ---------------------------------------------------------------------------


def build_input_ids(tokenizer, generation_text):
    """Build the token sequence: BOS + prefix + text + EOT tokens."""
    text = normalize_text(generation_text)
    logger.info(f"Normalized text: '{text}'")

    prefix = "<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

    bos_id = tokenizer.bos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    all_tokens = [bos_id] + prefix_tokens + text_tokens + [eot_id] * NUM_EOS_TOKENS
    input_ids = torch.tensor([all_tokens], dtype=torch.long)
    prefix_len = 1 + len(prefix_tokens)  # BOS + prefix

    return input_ids, prefix_len


# ---------------------------------------------------------------------------
# Decode waveform
# ---------------------------------------------------------------------------


def expand_and_decode(decoder, acoustic_cat, time_before_cat):
    """Expand acoustic features by duration and decode to waveform."""

    encoded = acoustic_cat[0]  # (N, 512)
    time_before = time_before_cat[0]  # (N+1,)
    time_before = time_before[: encoded.shape[0] + 1]

    encoded_expanded = []
    for pos in range(encoded.shape[0]):
        gap_len = max(0, int(time_before[pos].item()) - 1)
        if gap_len > 0:
            encoded_expanded.append(torch.zeros(gap_len, encoded.shape[-1]))
        encoded_expanded.append(encoded[pos].unsqueeze(0))

    trail_len = int(time_before[-1].item())
    if trail_len > 0:
        encoded_expanded.append(torch.zeros(trail_len, encoded.shape[-1]))

    encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)  # (1, T, 512)
    token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()

    logger.info(f"Expanded: {encoded_expanded.shape}, masks sum={token_masks.sum()}")

    with torch.no_grad():
        wav = decoder(encoded_expanded.float(), token_masks)

    return wav, time_before


def save_audio(filepath, audio, sample_rate=SAMPLE_RATE):
    if audio.dim() == 3:
        audio = audio.squeeze(0).squeeze(0)
    elif audio.dim() == 2:
        audio = audio.squeeze(0)
    audio = audio.clamp(-1.0, 1.0)
    audio_np = (audio.float().cpu().numpy() * 32767).astype("int16")
    wavfile.write(filepath, sample_rate, audio_np)
    logger.info(f"Saved {filepath} ({audio_np.shape[0] / sample_rate:.2f}s, std={np.std(audio_np):.0f})")


# ---------------------------------------------------------------------------
# Audio quality metrics (PESQ & STOI)
# ---------------------------------------------------------------------------


def load_wav_as_float(filepath, target_sr=SAMPLE_RATE):
    """Load a WAV file and return float32 numpy array at target_sr."""
    sr, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if sr != target_sr:
        # Simple resample via linear interpolation
        import scipy.signal

        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
    return data


def compute_pesq_stoi(ref_audio, deg_audio, sr=SAMPLE_RATE):
    """
    Compute PESQ and STOI between reference and degraded audio signals.

    Args:
        ref_audio: numpy float32 array (reference signal)
        deg_audio: numpy float32 array (degraded signal)
        sr: sample rate

    Returns:
        dict with 'pesq' and 'stoi' scores (or None if computation fails)
    """
    # Align lengths by truncating to the shorter one
    min_len = min(len(ref_audio), len(deg_audio))
    if min_len == 0:
        return {"pesq": None, "stoi": None}
    ref = ref_audio[:min_len]
    deg = deg_audio[:min_len]

    results = {}

    # PESQ (requires 8kHz or 16kHz)
    try:
        from pesq import pesq

        if sr == 16000 or sr == 8000:
            mode = "wb" if sr == 16000 else "nb"
            results["pesq"] = pesq(sr, ref, deg, mode)
        else:
            # Resample to 16kHz for PESQ
            import scipy.signal

            ref_16k = scipy.signal.resample(ref, int(len(ref) * 16000 / sr))
            deg_16k = scipy.signal.resample(deg, int(len(deg) * 16000 / sr))
            results["pesq"] = pesq(16000, ref_16k, deg_16k, "wb")
    except Exception as e:
        logger.warning(f"PESQ computation failed: {e}")
        results["pesq"] = None

    # STOI
    try:
        from pystoi import stoi

        results["stoi"] = stoi(ref, deg, sr, extended=False)
    except Exception as e:
        logger.warning(f"STOI computation failed: {e}")
        results["stoi"] = None

    return results


def compare_audio_files(file_pairs, sr=SAMPLE_RATE):
    """
    Compare pairs of audio files using PESQ and STOI.

    Args:
        file_pairs: list of (name, ref_path, deg_path) tuples
        sr: sample rate
    """
    logger.info(f"\n{'='*70}")
    logger.info("AUDIO QUALITY METRICS (PESQ & STOI)")
    logger.info(f"{'='*70}")
    logger.info(f"  PESQ range: -0.5 to 4.5 (higher = better, >3.0 is 'good')")
    logger.info(f"  STOI range:  0.0 to 1.0 (higher = better, >0.7 is 'intelligible')")
    logger.info(f"{'='*70}")

    for name, ref_path, deg_path in file_pairs:
        if not os.path.exists(ref_path):
            logger.warning(f"  {name}: ref not found: {ref_path}")
            continue
        if not os.path.exists(deg_path):
            logger.warning(f"  {name}: deg not found: {deg_path}")
            continue

        ref_audio = load_wav_as_float(ref_path, sr)
        deg_audio = load_wav_as_float(deg_path, sr)

        metrics = compute_pesq_stoi(ref_audio, deg_audio, sr)

        pesq_str = f"{metrics['pesq']:.3f}" if metrics["pesq"] is not None else "N/A"
        stoi_str = f"{metrics['stoi']:.3f}" if metrics["stoi"] is not None else "N/A"

        logger.info(
            f"  {name:50s} | PESQ={pesq_str:>7s} | STOI={stoi_str:>7s} | "
            f"ref={len(ref_audio)/sr:.2f}s, deg={len(deg_audio)/sr:.2f}s"
        )


# ---------------------------------------------------------------------------
# Reference TADA model oracle
# ---------------------------------------------------------------------------


def run_reference_tada_oracle(output_dir, generation_text=GENERATION_TEXT):
    """
    Run the actual TadaForCausalLM.generate() as gold-standard oracle.

    This tells us definitively whether the model produces good speech
    for unconditional generation (no prompt audio).
    """
    logger.info("=" * 60)
    logger.info("REFERENCE TADA ORACLE (TadaForCausalLM.generate)")
    logger.info("=" * 60)

    try:
        from tada.modules.encoder import EncoderOutput
        from tada.modules.tada import InferenceOptions, TadaForCausalLM
    except ImportError as e:
        logger.error(f"Cannot import TADA reference model: {e}")
        logger.error("Make sure tada source is on sys.path")
        return None

    device = torch.device("cpu")

    logger.info("Loading TadaForCausalLM from pretrained...")
    try:
        model = TadaForCausalLM.from_pretrained(
            TADA_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    except OSError as e:
        if "gated repo" in str(e).lower() or "401" in str(e):
            logger.warning(f"Cannot load reference TADA model — gated HF repo access required: {e}")
            logger.warning("The Aligner component requires access to meta-llama/Llama-3.2-1B tokenizer.")
            logger.warning("To fix: run `huggingface-cli login` with a token that has Llama access.")
            logger.warning("Skipping reference oracle.")
            return None
        raise
    model.eval()
    logger.info("Reference TADA model loaded.")

    # Create empty prompt (unconditional generation)
    prompt = EncoderOutput.empty(device=device, token_dim=512)

    # Set inference options matching our pipeline
    opts = InferenceOptions(
        num_flow_matching_steps=20,
        acoustic_cfg_scale=1.6,
        noise_temperature=0.9,
        text_temperature=0.6,
        text_top_p=0.9,
        text_repetition_penalty=1.1,
    )

    logger.info(f"Generating speech for: '{generation_text}'")
    try:
        with torch.no_grad():
            output = model.generate(
                prompt=prompt,
                text=generation_text,
                inference_options=opts,
                normalize_text=True,
                verbose=True,
            )
    except TypeError as e:
        logger.error(f"Reference TADA generate() failed: {e}")
        logger.error("This is likely a transformers version mismatch with the TADA reference code.")
        logger.error("The CPU pipeline (which re-implements the AR loop) is the better comparison baseline.")
        return None

    # Save output audio
    if output.audio is not None and output.audio.numel() > 0:
        audio = output.audio
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        audio = audio.clamp(-1.0, 1.0)

        ref_path = os.path.join(output_dir, "reference_tada_output.wav")
        audio_np = (audio.float().cpu().numpy() * 32767).astype("int16")
        wavfile.write(ref_path, SAMPLE_RATE, audio_np)
        duration = len(audio_np) / SAMPLE_RATE
        logger.info(f"Saved reference TADA output: {ref_path} ({duration:.2f}s)")
        return ref_path
    else:
        logger.error("Reference TADA model produced no audio!")
        return None


# ---------------------------------------------------------------------------
# Per-step hidden state divergence tracking
# ---------------------------------------------------------------------------


def analyze_divergence(output_dir):
    """
    Compare per-step hidden states between CPU and TTNN pipelines.

    Loads debug_hidden_states.pt from TTNN run and cpu_hidden_states.pt from CPU run,
    then prints a divergence table showing where quality degrades.
    """
    cpu_path = os.path.join(output_dir, "cpu_hidden_states.pt")
    ttnn_path = os.path.join(output_dir, "debug_hidden_states.pt")

    if not os.path.exists(cpu_path):
        logger.warning(f"CPU hidden states not found: {cpu_path}")
        return
    if not os.path.exists(ttnn_path):
        logger.warning(f"TTNN hidden states not found: {ttnn_path}")
        return

    cpu_data = torch.load(cpu_path, weights_only=False)
    ttnn_data = torch.load(ttnn_path, weights_only=False)

    cpu_hidden = cpu_data["hidden_states"]
    ttnn_hidden = ttnn_data["hidden_states"]

    n_steps = min(len(cpu_hidden), len(ttnn_hidden))
    logger.info(f"\n{'='*80}")
    logger.info("PER-STEP HIDDEN STATE DIVERGENCE (CPU vs TTNN)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Step':>5} | {'Token':>20} | {'CPU Norm':>10} | {'TTNN Norm':>10} | {'PCC':>8} | {'CosSim':>8}")
    logger.info("-" * 80)

    tokenizer = None
    tokens = cpu_data.get("tokens") or ttnn_data.get("tokens")

    for step in range(n_steps):
        cpu_h = cpu_hidden[step].float().flatten()
        ttnn_h = ttnn_hidden[step].float().flatten()

        cpu_norm = cpu_h.norm().item()
        ttnn_norm = ttnn_h.norm().item()
        cos_sim = torch.nn.functional.cosine_similarity(cpu_h, ttnn_h, dim=0).item()
        pcc = torch.corrcoef(torch.stack([cpu_h, ttnn_h]))[0, 1].item()

        tok_str = ""
        if tokens is not None and step < len(tokens):
            tok_str = str(tokens[step])[:20]

        logger.info(
            f"{step:>5} | {tok_str:>20} | {cpu_norm:>10.3f} | {ttnn_norm:>10.3f} | {pcc:>8.4f} | {cos_sim:>8.4f}"
        )

    logger.info(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main: full AR loop on CPU
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from transformers import AutoTokenizer

    # -----------------------------------------------------------------------
    # 1. Load models
    # -----------------------------------------------------------------------
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    logger.info("Loading Llama + TADA embeddings on CPU (float32)...")
    llama_model, tada_modules = load_llama_and_tada_cpu(TADA_MODEL_PATH)
    logger.info("Llama loaded.")

    logger.info("Loading VibeVoice on CPU (float32)...")
    vv_head = load_vibevoice_cpu(TADA_MODEL_PATH)
    logger.info("VibeVoice loaded.")

    logger.info("Loading Decoder on CPU...")
    decoder = load_decoder_cpu(TADA_CODEC_PATH)
    logger.info("Decoder loaded.")

    # -----------------------------------------------------------------------
    # 2. Build input IDs (matches tada_generator._build_input_ids)
    # -----------------------------------------------------------------------
    input_ids, prefix_len = build_input_ids(tokenizer, GENERATION_TEXT)
    num_steps = input_ids.shape[1]

    pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eos_token_id = tokenizer.eos_token_id

    logger.info(f"Input sequence ({input_ids.shape[1]} tokens):")
    for pos in range(input_ids.shape[1]):
        tok_id = input_ids[0, pos].item()
        tok_str = tokenizer.decode([tok_id])
        region = "PREFIX" if pos < prefix_len else ("TEXT" if pos < num_steps - NUM_EOS_TOKENS else "EOT")
        logger.info(f"  pos {pos}: {tok_id} = {repr(tok_str)} [{region}]")

    # -----------------------------------------------------------------------
    # 3. Create fake prompt (matches tada_generator.py unconditional logic)
    # -----------------------------------------------------------------------
    B = 1
    ref_prefix_len = prefix_len - 1  # Exclude BOS
    n_prompt_pad = max(0, ref_prefix_len - SHIFT_ACOUSTIC)

    prompt_acoustic_features = None
    prompt_acoustic_masks = None
    prompt_time_len_before = None
    prompt_time_len_after = None

    if n_prompt_pad > 0:
        prompt_acoustic_features = torch.zeros(B, n_prompt_pad, ACOUSTIC_DIM)
        prompt_acoustic_masks = torch.zeros(B, n_prompt_pad, dtype=torch.long)
        prompt_acoustic_masks[:, -1] = 1  # Last position gets mask=1
        prompt_time_len_before = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)
        prompt_time_len_after = torch.zeros(B, n_prompt_pad + 1, dtype=torch.long)
        logger.info(f"Created {n_prompt_pad} zero prompt features (mask={prompt_acoustic_masks[0].tolist()})")

    # Prefix acoustic trim count
    n_prefix_acoustic = n_prompt_pad + 1
    logger.info(f"Prefix len={prefix_len}, will discard first {n_prefix_acoustic} acoustic features")

    # -----------------------------------------------------------------------
    # 4. Initialize AR state
    # -----------------------------------------------------------------------
    acoustic_features = torch.zeros(B, ACOUSTIC_DIM)
    acoustic_masks = torch.zeros(B, dtype=torch.long)
    time_len_before = torch.zeros(B, dtype=torch.long)
    time_len_after = torch.zeros(B, dtype=torch.long)

    pos_kv_cache = None  # HF DynamicCache for positive path
    neg_kv_cache = None  # HF DynamicCache for negative path
    cache_position = torch.tensor([0], dtype=torch.long)

    use_cfg = True  # acoustic_cfg_scale=1.6
    acoustic_cfg_scale = 1.6
    duration_cfg_scale = 1.0
    noise_temp = 0.9
    num_fm_steps = 20

    all_acoustic_features = []
    all_time_before = []
    all_output_token_ids = []
    all_hidden_states = []

    # Also load TTNN hidden states for comparison if available
    debug_path = os.path.join(OUTPUT_DIR, "debug_hidden_states.pt")
    ttnn_data = None
    if os.path.exists(debug_path):
        ttnn_data = torch.load(debug_path, weights_only=False)
        ttnn_hidden = ttnn_data["hidden_states"]
        logger.info(f"Loaded {len(ttnn_hidden)} TTNN hidden states for comparison")

    bottleneck_proj = tada_modules.get("bottleneck_proj")

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting full CPU AR generation for {num_steps} steps")
    logger.info(f"CFG: acoustic={acoustic_cfg_scale}, duration={duration_cfg_scale}")
    logger.info(f"Sampling: temp=0.6, top_p=0.9, rep_penalty=1.1")
    logger.info(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 5. AR loop (matches tada_generator.py lines 1292-1452)
    # -----------------------------------------------------------------------
    for step in range(num_steps):
        # Current text token
        if step < input_ids.shape[1]:
            input_slice = input_ids[:, step]
        else:
            input_slice = input_ids[:, -1]

        # Build embedding
        inputs_embeds = build_inputs_embeds(
            input_slice, acoustic_features, acoustic_masks, time_len_before, time_len_after, tada_modules
        )

        # Llama forward (positive path)
        with torch.no_grad():
            outputs = llama_model(
                inputs_embeds=inputs_embeds,
                past_key_values=pos_kv_cache,
                use_cache=True,
                cache_position=cache_position,
            )
        pos_kv_cache = outputs.past_key_values
        hidden_cpu = outputs.last_hidden_state  # (B, 1, H)
        all_hidden_states.append(hidden_cpu.clone())

        # Negative conditioning path (separate KV cache, like TTNN)
        if use_cfg:
            is_structural = (input_slice == start_header_id) | (input_slice == end_header_id) | (input_slice == eot_id)
            neg_input_slice = torch.where(is_structural, input_slice, torch.full_like(input_slice, pad_token_id))
            neg_embeds = build_inputs_embeds(
                neg_input_slice, acoustic_features, acoustic_masks, time_len_before, time_len_after, tada_modules
            )
            with torch.no_grad():
                neg_outputs = llama_model(
                    inputs_embeds=neg_embeds,
                    past_key_values=neg_kv_cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
            neg_kv_cache = neg_outputs.past_key_values
            neg_hidden = neg_outputs.last_hidden_state  # (B, 1, H)
        else:
            neg_hidden = hidden_cpu  # unused

        # Compare with TTNN hidden states if available
        comparison_str = ""
        if ttnn_data is not None and step < len(ttnn_hidden):
            ttnn_h = ttnn_hidden[step].float()
            cos_sim = nn.functional.cosine_similarity(hidden_cpu.flatten(), ttnn_h.flatten(), dim=0).item()
            pcc = torch.corrcoef(torch.stack([hidden_cpu.flatten(), ttnn_h.flatten()]))[0, 1].item()
            comparison_str = f" | vs_TTNN: cos={cos_sim:.4f}, pcc={pcc:.4f}"

        # LM head for text logits
        with torch.no_grad():
            logits = tada_modules["lm_head"](hidden_cpu)  # (B, 1, vocab)
        logits = logits.squeeze(1)  # (B, vocab)

        # Sample next text token if past prompt
        if step >= input_ids.shape[1] - 1:
            torch.manual_seed(RANDOM_SEED + 10000 + step)
            next_token = sample_text_token(logits, input_ids, pad_token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            all_output_token_ids.append(next_token)

            if next_token.item() == eos_token_id or next_token.item() == eot_id:
                logger.info(f"EOS at step {step}")
                break
        else:
            all_output_token_ids.append(input_ids[:, step + 1 : step + 2])

        # Flow matching ODE — squeeze to (B, H) for VibeVoice condition input
        with torch.no_grad():
            speech = solve_flow_matching_cpu(
                vv_head,
                hidden_cpu.squeeze(1),
                neg_hidden.squeeze(1),
                bottleneck_proj=bottleneck_proj,
                num_steps=num_fm_steps,
                noise_temp=noise_temp,
                acoustic_cfg_scale=acoustic_cfg_scale,
                duration_cfg_scale=duration_cfg_scale,
                seed=RANDOM_SEED + step,
            )

        # Extract time from gray code
        time_gray = speech[..., -TIME_DIM:]
        predicted_time_before = decode_gray_code_to_time(time_gray[..., :NUM_TIME_BITS], NUM_TIME_BITS)
        predicted_time_after = decode_gray_code_to_time(time_gray[..., NUM_TIME_BITS:], NUM_TIME_BITS)

        # Update acoustic state (shift_acoustic delay)
        if step >= SHIFT_ACOUSTIC:
            if prompt_acoustic_features is not None and step - SHIFT_ACOUSTIC < prompt_acoustic_features.shape[1]:
                acoustic_features = prompt_acoustic_features[:, step - SHIFT_ACOUSTIC]
                acoustic_masks = prompt_acoustic_masks[:, step - SHIFT_ACOUSTIC]
            else:
                acoustic_features = speech[..., :ACOUSTIC_DIM]
                acoustic_masks = torch.ones(B, dtype=torch.long)
            all_acoustic_features.append(
                acoustic_features.unsqueeze(1) if acoustic_features.dim() == 2 else acoustic_features
            )

            if prompt_time_len_before is not None and step - SHIFT_ACOUSTIC < prompt_time_len_before.shape[1] - 1:
                time_len_before = prompt_time_len_before[:, step - SHIFT_ACOUSTIC + 1]
                time_len_after = prompt_time_len_after[:, step - SHIFT_ACOUSTIC + 1]
            else:
                time_len_before = predicted_time_before
                time_len_after = predicted_time_after
            all_time_before.append(time_len_before.unsqueeze(1) if time_len_before.dim() == 1 else time_len_before)
        else:
            acoustic_features = torch.zeros(B, ACOUSTIC_DIM)
            acoustic_masks = torch.zeros(B, dtype=torch.long)
            time_len_before = torch.zeros(B, dtype=torch.long)
            time_len_after = torch.zeros(B, dtype=torch.long)

        # Advance cache position
        cache_position = cache_position + 1

        # Log
        tok_name = repr(tokenizer.decode([input_slice[0].item()]))
        src = (
            "prompt"
            if (
                prompt_acoustic_features is not None
                and step >= SHIFT_ACOUSTIC
                and step - SHIFT_ACOUSTIC < prompt_acoustic_features.shape[1]
            )
            else ("predicted" if step >= SHIFT_ACOUSTIC else "zeros")
        )
        if step % 5 == 0 or step < 15:
            logger.info(
                f"  Step {step}/{num_steps} [{tok_name:20s}]: "
                f"speech_norm={speech.norm().item():.3f}, "
                f"af_src={src}, af_norm={acoustic_features.norm().item():.3f}, "
                f"t_before={time_len_before.tolist()}, "
                f"t_after={time_len_after.tolist()}, "
                f"mask={acoustic_masks.tolist()}"
                f"{comparison_str}"
            )

    logger.info("\nAR generation complete.")

    # Save CPU hidden states for divergence tracking
    cpu_hidden_path = os.path.join(OUTPUT_DIR, "cpu_hidden_states.pt")
    torch.save({"hidden_states": all_hidden_states}, cpu_hidden_path)
    logger.info(f"Saved {len(all_hidden_states)} CPU hidden states to {cpu_hidden_path}")

    # -----------------------------------------------------------------------
    # 6. Collect and decode
    # -----------------------------------------------------------------------
    if not all_acoustic_features:
        logger.error("No acoustic features collected!")
        return

    acoustic_cat = torch.cat([f if f.dim() == 3 else f.unsqueeze(1) for f in all_acoustic_features], dim=1)
    # Un-normalize
    acoustic_cat = acoustic_cat * ACOUSTIC_STD + ACOUSTIC_MEAN

    time_before_cat = torch.cat([t if t.dim() == 2 else t.unsqueeze(1) for t in all_time_before], dim=1)
    # Add trailing time_before
    if all_time_before:
        time_before_cat = torch.cat(
            [
                time_before_cat,
                all_time_before[-1] if all_time_before[-1].dim() == 2 else all_time_before[-1].unsqueeze(1),
            ],
            dim=1,
        )

    logger.info(f"Acoustic features (raw): shape={acoustic_cat.shape}, norm={acoustic_cat.norm():.3f}")
    logger.info(f"Time before (raw): {time_before_cat[0].tolist()}")

    # Trim prefix
    if n_prefix_acoustic > 0 and acoustic_cat.shape[1] > n_prefix_acoustic:
        logger.info(f"Slicing off {n_prefix_acoustic} prefix acoustic features")
        acoustic_cat = acoustic_cat[:, n_prefix_acoustic:, :]
        time_before_cat = time_before_cat[:, n_prefix_acoustic:]

    logger.info(f"Acoustic features (trimmed): shape={acoustic_cat.shape}, norm={acoustic_cat.norm():.3f}")
    logger.info(f"Time before (trimmed): {time_before_cat[0].tolist()}")

    # Decode waveform
    try:
        wav, time_before = expand_and_decode(decoder, acoustic_cat, time_before_cat)
        # Remove leading silence
        leading_silence = int(time_before[0].item() * 480)
        if leading_silence > 0 and wav.shape[-1] > leading_silence:
            wav = wav[..., leading_silence:]
            logger.info(f"Trimmed {leading_silence} leading silence samples")

        save_audio(os.path.join(OUTPUT_DIR, "cpu_full_pipeline_output.wav"), wav)
        logger.info(f"CPU full pipeline output: shape={wav.shape}, norm={wav.norm():.4f}")
    except Exception as e:
        logger.error(f"Decoding failed: {e}")
        import traceback

        traceback.print_exc()

    # -----------------------------------------------------------------------
    # 7. Run reference TADA oracle
    # -----------------------------------------------------------------------
    ref_tada_path = run_reference_tada_oracle(OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # 8. Per-step divergence tracking
    # -----------------------------------------------------------------------
    analyze_divergence(OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # 9. Audio quality comparison (PESQ & STOI)
    # -----------------------------------------------------------------------
    cpu_output_path = os.path.join(OUTPUT_DIR, "cpu_full_pipeline_output.wav")
    ttnn_output_path = os.path.join(OUTPUT_DIR, "demo_output.wav")
    ttnn_ref_dec_path = os.path.join(OUTPUT_DIR, "demo_output_reference.wav")

    # Build comparison pairs from all available audio files
    file_pairs = []

    # Reference TADA vs CPU pipeline (validates our CPU re-implementation)
    if ref_tada_path and os.path.exists(cpu_output_path):
        file_pairs.append(("Ref_TADA vs CPU_pipeline", ref_tada_path, cpu_output_path))

    # Reference TADA vs TTNN outputs
    if ref_tada_path and os.path.exists(ttnn_output_path):
        file_pairs.append(("Ref_TADA vs TTNN_decoder", ref_tada_path, ttnn_output_path))

    # CPU pipeline vs TTNN outputs
    if os.path.exists(cpu_output_path) and os.path.exists(ttnn_output_path):
        file_pairs.append(("CPU_pipeline vs TTNN_decoder", cpu_output_path, ttnn_output_path))
    if os.path.exists(cpu_output_path) and os.path.exists(ttnn_ref_dec_path):
        file_pairs.append(("CPU_pipeline vs TTNN_ref_decoder", cpu_output_path, ttnn_ref_dec_path))

    # TTNN decoder vs TTNN reference decoder (isolates decoder difference)
    if os.path.exists(ttnn_output_path) and os.path.exists(ttnn_ref_dec_path):
        file_pairs.append(("TTNN_decoder vs TTNN_ref_decoder", ttnn_output_path, ttnn_ref_dec_path))

    # Self-comparison (sanity check — should be perfect)
    if os.path.exists(cpu_output_path):
        file_pairs.append(("CPU_pipeline vs itself (sanity)", cpu_output_path, cpu_output_path))

    if file_pairs:
        compare_audio_files(file_pairs)
    else:
        logger.warning("No audio files available for PESQ/STOI comparison")

    # -----------------------------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Generated text tokens: {input_ids.shape[1]} total")
    logger.info(f"Acoustic features: {acoustic_cat.shape[1]} (after trimming {n_prefix_acoustic} prefix)")
    logger.info(f"Output saved to: {cpu_output_path}")
    if ref_tada_path:
        logger.info(f"Reference TADA output: {ref_tada_path}")
    logger.info("")
    logger.info("DECISION TREE:")
    logger.info("  Ref TADA good + CPU good → TTNN precision bug (focus on divergence table)")
    logger.info("  Ref TADA bad             → Model doesn't support unconditional gen (need prompt audio)")
    logger.info("  CFG=1.0 better than 1.6  → Disable CFG for unconditional mode")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
