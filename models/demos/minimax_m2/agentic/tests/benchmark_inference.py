#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark inference times for all agentic models on N300.

Loads all models once, then runs inference on each model 5 times sequentially,
capturing per-iteration and average inference times.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device


def make_wav(path, duration=2.0, sr=16000):
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sf.write(path, (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32), sr)


def make_img(path):
    d = np.zeros((512, 512, 3), dtype=np.uint8)
    d[40:200, 40:200] = [220, 30, 30]  # red block
    d[312:472, 176:336] = [30, 180, 30]  # green block
    Image.fromarray(d, "RGB").save(path)


def benchmark_model(name, inference_fn, iterations=5):
    """Run inference N times and collect timing stats."""
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        result = inference_fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  [{name}] iter {i+1}/{iterations}: {elapsed*1000:.1f} ms")
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference times for all models")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM (requires HF auth)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of inference iterations")
    args = parser.parse_args()

    iterations = args.iterations
    results = {}

    logger.info("=" * 70)
    logger.info(f"BENCHMARK: {iterations} inference iterations per model")
    logger.info("=" * 70)

    mesh = open_n300_device()
    try:
        # Use full mesh - all models now support (1,2) mesh with mesh_composer
        device = mesh

        with tempfile.TemporaryDirectory() as tmpdir:
            wav = f"{tmpdir}/test.wav"
            img = f"{tmpdir}/test.png"
            make_wav(wav)
            make_img(img)

            # ─────────────────────────────────────────────────────────────
            # LOAD ALL MODELS
            # ─────────────────────────────────────────────────────────────
            logger.info("Loading all models...")
            load_start = time.perf_counter()

            from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool
            from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool
            from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool
            from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

            whisper = WhisperTool(mesh_device=device)
            bert = BERTTool(mesh_device=device)
            owlvit = OWLViTTool(mesh_device=device)
            speecht5 = SpeechT5Tool(mesh_device=device, warmup_on_init=False)

            llm = None
            if not args.skip_llm:
                # LLM runs on full (1,2) mesh with fabric enabled for multi-chip parallelism
                from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import LLMTool

                llm = LLMTool(mesh_device=device)

            load_time = time.perf_counter() - load_start
            logger.info(f"All models loaded in {load_time:.2f}s")

            # ─────────────────────────────────────────────────────────────
            # WARMUP (first inference compiles kernels / captures traces)
            # ─────────────────────────────────────────────────────────────
            logger.info("=" * 70)
            logger.info("WARMUP PHASE (first inference for each model)")
            logger.info("=" * 70)

            logger.info("Warming up Whisper...")
            warmup_start = time.perf_counter()
            _ = whisper.transcribe(wav)
            whisper.release_decoder_trace()  # Release for safe coexistence
            logger.info(f"  Whisper warmup: {(time.perf_counter()-warmup_start)*1000:.1f} ms")

            logger.info("Warming up BERT...")
            warmup_start = time.perf_counter()
            _ = bert.qa("Test?", "Test context.")
            logger.info(f"  BERT warmup: {(time.perf_counter()-warmup_start)*1000:.1f} ms")

            logger.info("Warming up OWL-ViT...")
            warmup_start = time.perf_counter()
            _ = owlvit.detect(img, "red block")
            logger.info(f"  OWL-ViT warmup: {(time.perf_counter()-warmup_start)*1000:.1f} ms")

            logger.info("Warming up SpeechT5...")
            warmup_start = time.perf_counter()
            speecht5._warmup()
            _ = speecht5.synthesize("Test.", f"{tmpdir}/warmup.wav")
            logger.info(f"  SpeechT5 warmup: {(time.perf_counter()-warmup_start)*1000:.1f} ms")

            if llm:
                logger.info("Warming up LLM...")
                warmup_start = time.perf_counter()
                _ = llm.generate_response(
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_new_tokens=8,
                )
                logger.info(f"  LLM warmup: {(time.perf_counter()-warmup_start)*1000:.1f} ms")

            # ─────────────────────────────────────────────────────────────
            # BENCHMARK INFERENCE
            # ─────────────────────────────────────────────────────────────
            logger.info("=" * 70)
            logger.info(f"BENCHMARK PHASE ({iterations} iterations per model)")
            logger.info("=" * 70)

            # Whisper
            logger.info("Benchmarking Whisper...")
            results["whisper"] = benchmark_model(
                "Whisper",
                lambda: whisper.transcribe(wav),
                iterations,
            )

            # BERT
            logger.info("Benchmarking BERT...")
            results["bert"] = benchmark_model(
                "BERT",
                lambda: bert.qa(
                    "What chips are in the N300?",
                    "The N300 uses two Wormhole B0 chips connected via Ethernet.",
                ),
                iterations,
            )

            # OWL-ViT
            logger.info("Benchmarking OWL-ViT...")
            results["owlvit"] = benchmark_model(
                "OWL-ViT",
                lambda: owlvit.detect(img, "red block, green block"),
                iterations,
            )

            # SpeechT5
            tts_counter = [0]

            def run_tts():
                tts_counter[0] += 1
                return speecht5.synthesize(
                    "Testing text to speech synthesis on N300.",
                    f"{tmpdir}/tts_{tts_counter[0]}.wav",
                )

            logger.info("Benchmarking SpeechT5...")
            results["speecht5"] = benchmark_model("SpeechT5", run_tts, iterations)

            # LLM (if available)
            if llm:
                logger.info("Benchmarking LLM...")
                results["llm"] = benchmark_model(
                    "LLM",
                    lambda: llm.generate_response(
                        messages=[{"role": "user", "content": "What is 2+2? One word."}],
                        max_new_tokens=16,
                    ),
                    iterations,
                )

            # ─────────────────────────────────────────────────────────────
            # RESULTS SUMMARY
            # ─────────────────────────────────────────────────────────────
            logger.info("=" * 70)
            logger.info("RESULTS SUMMARY")
            logger.info("=" * 70)
            print()
            print(f"{'Model':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
            print("-" * 60)
            for name, times in results.items():
                times_ms = [t * 1000 for t in times]
                avg = mean(times_ms)
                std = stdev(times_ms) if len(times_ms) > 1 else 0
                min_t = min(times_ms)
                max_t = max(times_ms)
                print(f"{name:<12} {avg:<12.1f} {std:<12.1f} {min_t:<12.1f} {max_t:<12.1f}")
            print("-" * 60)

            # Total loop time
            total_avg = sum(mean(times) for times in results.values()) * 1000
            print(f"{'TOTAL':<12} {total_avg:<12.1f} ms per full loop")
            print()

            # Cleanup
            if hasattr(whisper, "close"):
                whisper.close()

    finally:
        ttnn.close_mesh_device(mesh)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
