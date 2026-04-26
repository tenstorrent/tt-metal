# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LLVC (Low-Latency Low-Resource Voice Conversion) demo for Tenstorrent hardware.

Usage:
    # On-device demo (requires TT hardware)
    pytest models/demos/audio/llvc/demo/demo.py -k test_llvc_demo -v

    # CPU-only demo (no hardware required)
    pytest models/demos/audio/llvc/demo/demo.py -k test_llvc_demo_cpu -v

This demo generates a synthetic audio signal (sine wave), runs it through the
LLVC model using TTNN ops, and compares the output against the PyTorch reference.
It validates both non-streaming and streaming modes, reports PCC accuracy, and
measures throughput/latency performance metrics.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.llvc.reference.model import Net, get_default_config
from models.demos.audio.llvc.tt.ttnn_functional_llvc import (
    LLVC_L1_SMALL_SIZE,
    _is_mesh_device,
    init_buffers,
    preprocess_model_parameters,
    ttnn_llvc_forward,
)

# Default audio parameters
SAMPLE_RATE = 16000


def _create_model_from_config(config):
    """Instantiate a LLVC reference model from config dict."""
    model = Net(
        label_len=config.get("label_len", 1),
        L=config.get("L", 16),
        enc_dim=config.get("enc_dim", 512),
        num_enc_layers=config.get("num_enc_layers", 8),
        dec_dim=config.get("dec_dim", 256),
        dec_buf_len=config.get("dec_buf_len", 13),
        num_dec_layers=config.get("num_dec_layers", 1),
        dec_chunk_size=config.get("dec_chunk_size", 13),
        out_buf_len=config.get("out_buf_len", 4),
        use_pos_enc=config.get("use_pos_enc", True),
        skip_connection=True,
        proj=True,
        lookahead=True,
        decoder_dropout=config.get("decoder_dropout", 0.1),
        convnet_config=config.get("convnet_config"),
    )
    model.eval()
    return model


def _generate_test_audio(duration_s, sample_rate=SAMPLE_RATE, freq_hz=440.0):
    """
    Generate a synthetic audio signal for testing.

    Creates a sine wave at the given frequency, which is more representative
    of real audio than random noise for validating the voice conversion pipeline.
    """
    num_samples = int(duration_s * sample_rate)
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    # Mix of two frequencies to create a more complex signal
    audio = 0.3 * torch.sin(2 * torch.pi * freq_hz * t) + 0.1 * torch.sin(2 * torch.pi * freq_hz * 2 * t)
    return audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]


def _get_ttnn_config(config):
    """Build the TTNN config dict from model config."""
    return {
        "L": config["L"],
        "enc_dim": config["enc_dim"],
        "out_buf_len": config["out_buf_len"],
        "lookahead": True,
        "num_enc_layers": config["num_enc_layers"],
        "num_dec_layers": config["num_dec_layers"],
        "dec_buf_len": config.get("dec_buf_len", 13),
        "dec_chunk_size": config.get("dec_chunk_size", 13),
        "dec_dim": config.get("dec_dim", 256),
        "use_pos_enc": config.get("use_pos_enc", True),
        "skip_connection": True,
        "proj": True,
    }


def run_llvc_demo(device, config=None, audio_length_seconds=0.1, sample_rate=SAMPLE_RATE):
    """
    Run the LLVC demo (works on both CPU and TT device).

    Args:
        device: TTNN mesh device, or None for CPU-only mode
        config: LLVC model config dict (uses default if None)
        audio_length_seconds: length of synthetic audio in seconds
        sample_rate: sample rate in Hz
    Returns:
        dict with results including PCC scores and performance metrics
    """
    torch.manual_seed(42)

    if config is None:
        config = get_default_config()
        # Use simpler config for fast demo
        config["num_enc_layers"] = 2
        config["num_dec_layers"] = 1
        config["convnet_config"] = None

    L = config["L"]
    is_device = device is not None

    logger.info("=" * 60)
    logger.info("LLVC (Low-Latency Low-Resource Voice Conversion) Demo")
    logger.info(f"Mode: {'TT Device' if is_device else 'CPU-only'}")
    logger.info("=" * 60)

    # --- Model setup ---
    logger.info("Creating LLVC reference model...")
    reference_model = _create_model_from_config(config)
    num_params = sum(p.numel() for p in reference_model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # --- Generate test audio ---
    num_samples = int(audio_length_seconds * sample_rate)
    num_samples = (num_samples // L) * L
    if num_samples == 0:
        num_samples = L * 4

    x_torch = _generate_test_audio(num_samples / sample_rate, sample_rate)
    actual_duration = num_samples / sample_rate
    logger.info(f"Input audio: {x_torch.shape} ({actual_duration:.3f}s @ {sample_rate}Hz)")

    # --- PyTorch reference ---
    logger.info("Running PyTorch reference inference...")
    with torch.no_grad():
        ref_output = reference_model(x_torch, pad=True)
    logger.info(f"Reference output: {ref_output.shape}")

    # --- TTNN non-streaming inference ---
    logger.info("Preprocessing parameters for TTNN...")
    parameters = preprocess_model_parameters(reference_model, device=device)
    ttnn_config = _get_ttnn_config(config)

    if is_device:
        logger.info("Running TTNN inference on Tenstorrent device...")
        if _is_mesh_device(device):
            x_input = ttnn.from_torch(
                x_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            x_input = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Warmup run (includes JIT compilation overhead — not counted in perf)
        _ = ttnn_llvc_forward(
            x_input,
            parameters=parameters,
            config=ttnn_config,
            device=device,
            pad=True,
        )
        # Re-create input for timed run
        if _is_mesh_device(device):
            x_input = ttnn.from_torch(
                x_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            x_input = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        logger.info("Running TTNN inference (CPU mode)...")
        x_input = x_torch

    start = time.perf_counter()
    tt_output = ttnn_llvc_forward(
        x_input,
        parameters=parameters,
        config=ttnn_config,
        device=device,
        pad=True,
    )
    inference_time = time.perf_counter() - start

    if is_device:
        if _is_mesh_device(device):
            tt_output_torch = ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
            )
            tt_output_torch = tt_output_torch[:1]  # Take first replica
        else:
            tt_output_torch = ttnn.to_torch(tt_output)
    else:
        tt_output_torch = tt_output

    logger.info(f"TTNN output: {tt_output_torch.shape}")

    # --- Accuracy comparison ---
    pcc_threshold = 0.96 if is_device else 0.99
    passed, pcc_value = comp_pcc(ref_output, tt_output_torch, pcc_threshold)
    logger.info(f"PCC (TTNN vs PyTorch reference): {pcc_value}")

    # --- Token-level accuracy ---
    # Use isclose with both absolute and relative tolerance for BFloat16:
    #   |ref - tt| <= atol + rtol * |ref|
    # atol=0.01 handles near-zero values, rtol=0.1 handles larger values
    token_close = torch.isclose(ref_output, tt_output_torch.float(), atol=0.01, rtol=0.1)
    token_match = token_close.float().mean().item() * 100
    logger.info(f"Token-level match (isclose atol=0.01 rtol=0.1): {token_match:.1f}%")

    # --- Streaming demo ---
    logger.info("\n--- Streaming Mode Demo ---")
    chunk_samples = L * 4  # ~4ms chunks at 16kHz
    num_chunks = max(1, num_samples // chunk_samples)

    # Reference streaming
    with torch.no_grad():
        ref_enc_buf, ref_dec_buf, ref_out_buf = reference_model.init_buffers(1, x_torch.device)
        ref_chunks_out = []
        for c in range(num_chunks):
            chunk = x_torch[:, :, c * chunk_samples : (c + 1) * chunk_samples]
            out, ref_enc_buf, ref_dec_buf, ref_out_buf, _ = reference_model(
                chunk,
                init_enc_buf=ref_enc_buf,
                init_dec_buf=ref_dec_buf,
                init_out_buf=ref_out_buf,
                pad=False,
            )
            ref_chunks_out.append(out)
    ref_streamed = torch.cat(ref_chunks_out, dim=-1) if ref_chunks_out else torch.zeros(1, 1, 0)

    # TTNN streaming
    tt_enc_buf, tt_dec_buf, tt_out_buf = init_buffers(1, ttnn_config)

    tt_chunks_out = []
    chunk_times = []
    for c in range(num_chunks):
        chunk = x_torch[:, :, c * chunk_samples : (c + 1) * chunk_samples]
        start = time.perf_counter()
        out, tt_enc_buf, tt_dec_buf, tt_out_buf, _ = ttnn_llvc_forward(
            chunk,
            parameters=parameters,
            config=ttnn_config,
            device=device,
            init_enc_buf=tt_enc_buf,
            init_dec_buf=tt_dec_buf,
            init_out_buf=tt_out_buf,
            pad=False,
        )
        chunk_times.append(time.perf_counter() - start)
        if is_device and not isinstance(out, torch.Tensor):
            out = ttnn.to_torch(out).float()[:1]
        tt_chunks_out.append(out)

    tt_streamed = torch.cat(tt_chunks_out, dim=-1) if tt_chunks_out else torch.zeros(1, 1, 0)

    # Streaming accuracy
    if ref_streamed.numel() > 0 and tt_streamed.numel() > 0:
        stream_passed, stream_pcc = comp_pcc(ref_streamed, tt_streamed, 0.99)
        logger.info(f"Streaming PCC (TTNN vs ref): {stream_pcc}")
    else:
        stream_passed, stream_pcc = True, 1.0

    # --- Performance metrics ---
    logger.info("\n--- Performance Metrics ---")
    num_tokens = num_samples // L
    tokens_per_sec = num_tokens / inference_time if inference_time > 0 else 0
    rtf = inference_time / actual_duration if actual_duration > 0 else 0

    logger.info("Non-streaming:")
    logger.info(f"  Inference time: {inference_time * 1000:.1f} ms")
    logger.info(f"  Tokens/sec: {tokens_per_sec:.0f}")
    logger.info(f"  RTF: {rtf:.4f}")

    if chunk_times:
        avg_chunk_ms = sum(chunk_times) / len(chunk_times) * 1000
        chunk_duration_ms = chunk_samples / sample_rate * 1000
        streaming_rtf = (avg_chunk_ms / chunk_duration_ms) if chunk_duration_ms > 0 else 0
        logger.info(f"Streaming ({num_chunks} chunks):")
        logger.info(f"  Avg chunk latency: {avg_chunk_ms:.1f} ms")
        logger.info(f"  Chunk audio duration: {chunk_duration_ms:.1f} ms")
        logger.info(f"  Streaming RTF: {streaming_rtf:.4f}")
    else:
        avg_chunk_ms = 0
        streaming_rtf = 0

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    overall_pass = passed and stream_passed
    status = "PASSED" if overall_pass else "FAILED"
    logger.info(f"DEMO {status}")
    logger.info(f"  Non-streaming PCC: {pcc_value}")
    logger.info(f"  Streaming PCC:     {stream_pcc}")
    logger.info(f"  Token match:       {token_match:.1f}%")
    logger.info(f"  Tokens/sec:        {tokens_per_sec:.0f}")
    logger.info(f"  RTF:               {rtf:.4f}")
    logger.info("=" * 60)

    return {
        "passed": overall_pass,
        "pcc": pcc_value,
        "stream_pcc": stream_pcc,
        "token_match": token_match,
        "tokens_per_sec": tokens_per_sec,
        "rtf": rtf,
        "streaming_rtf": streaming_rtf,
        "avg_chunk_latency_ms": avg_chunk_ms,
        "ref_output": ref_output,
        "tt_output": tt_output_torch,
        "num_params": num_params,
        "audio_length_seconds": actual_duration,
    }


# ============================================================================
# Demo entry points
# ============================================================================


def test_llvc_demo_cpu():
    """LLVC demo test entry point (CPU mode, no TT hardware required)."""
    results = run_llvc_demo(device=None, audio_length_seconds=0.5)
    assert results["passed"], f"Demo failed with PCC {results['pcc']}"


@pytest.mark.skipif(
    not os.path.exists("/dev/tenstorrent"),
    reason="No TT device available",
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": LLVC_L1_SMALL_SIZE}], indirect=True)
def test_llvc_demo(device):
    """LLVC demo test entry point (TT device, single chip via PCIe)."""
    results = run_llvc_demo(device, audio_length_seconds=0.5)
    assert results["passed"], f"Demo failed with PCC {results['pcc']}"
