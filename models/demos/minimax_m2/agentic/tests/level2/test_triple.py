#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Level 2 Test: Three-model combinations on shared N300 device.

Tests selected 3-model combinations:
1. Non-trace models together: BERT + OWL-ViT + SpeechT5
2. One trace + two non-trace: Whisper + BERT + OWL-ViT

Flow:
1. Open device (shared params)
2. Load all three models
3. Warmup each (release trace after trace-model warmup)
4. Run inference for each
5. Close device
"""
import gc
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from loguru import logger

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device

# ─── Test Data Generators ────────────────────────────────────────────────────


def make_test_wav(path: str, duration: float = 2.0, sr: int = 16000):
    """Create a simple sine wave test audio file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(path, audio, sr)


def make_test_image(path: str):
    """Create a simple test image with colored blocks."""
    data = np.zeros((512, 512, 3), dtype=np.uint8)
    data[40:200, 40:200] = [220, 30, 30]
    data[312:472, 176:336] = [30, 180, 30]
    Image.fromarray(data, "RGB").save(path)


# ─── Triple Combinations ─────────────────────────────────────────────────────

TRIPLES = [
    # (name, description, models_in_load_order)
    ("non_trace_trio", "BERT + OWL-ViT + SpeechT5 (all non-trace)", ["bert", "owlvit", "speecht5"]),
    ("whisper_bert_owl", "Whisper + BERT + OWL-ViT (one trace)", ["whisper", "bert", "owlvit"]),
]


# ─── Test Runner ─────────────────────────────────────────────────────────────


def run_triple_test(test_name: str, description: str, model_names: List[str]) -> bool:
    """
    Run a triple test: load all three, warmup all, infer all.

    Staged strategy:
    - For Whisper: load and warmup first, release trace, then load others
    - For BERT without Whisper: load and warmup BERT first, then load chip0 models
    """
    logger.info("=" * 60)
    logger.info(f"Level 2: Triple Test - {test_name}")
    logger.info(f"Description: {description}")
    logger.info(f"Models: {model_names}")
    logger.info("=" * 60)

    mesh = open_n300_device()
    models = {}

    try:
        chip0 = (
            mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)) if mesh.get_num_devices() > 1 else mesh
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = f"{tmpdir}/test.wav"
            img_path = f"{tmpdir}/test.png"
            make_test_wav(wav_path)
            make_test_image(img_path)

            # --- Staged Load Phase ---
            whisper_first = "whisper" in model_names
            bert_first = "bert" in model_names and not whisper_first

            if whisper_first:
                # Load Whisper first
                logger.info("[LOAD] Loading Whisper first (staged strategy)...")
                from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

                models["whisper"] = WhisperTool(mesh_device=mesh)

                logger.info("[WARMUP] Whisper warmup (trace capture)...")
                result = models["whisper"].transcribe(wav_path)
                assert isinstance(result, str)
                logger.info(f"Whisper warmup result: {result!r}")

                logger.info("[TRACE] Releasing Whisper decoder trace...")
                models["whisper"].release_decoder_trace()
                gc.collect()
                try:
                    ttnn.synchronize_device(mesh)
                except Exception as e:
                    logger.warning(f"synchronize_device after trace release: {e}")

            if bert_first:
                # Load and warmup BERT before chip0 models to avoid hang
                logger.info("[LOAD] Loading BERT first (staged strategy)...")
                from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

                models["bert"] = BERTTool(mesh_device=mesh)

                logger.info("[WARMUP] BERT warmup...")
                result = models["bert"].qa(
                    "How many chips?",
                    "The N300 contains two Wormhole B0 chips.",
                )
                assert isinstance(result, str) and len(result.strip()) > 0
                logger.info(f"BERT warmup result: {result!r}")
                gc.collect()

            # Load remaining models
            for name in model_names:
                if name == "whisper" and whisper_first:
                    continue  # Already loaded
                if name == "bert" and bert_first:
                    continue  # Already loaded

                logger.info(f"[LOAD] Loading {name}...")

                if name == "whisper":
                    from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

                    models["whisper"] = WhisperTool(mesh_device=mesh)

                elif name == "bert":
                    from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

                    models["bert"] = BERTTool(mesh_device=mesh)

                elif name == "owlvit":
                    from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

                    models["owlvit"] = OWLViTTool(mesh_device=chip0)

                elif name == "speecht5":
                    from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

                    models["speecht5"] = SpeechT5Tool(mesh_device=chip0, warmup_on_init=False)

            gc.collect()
            logger.info("All models loaded.")

            # --- Warmup Phase (remaining models) ---
            logger.info("[WARMUP] Warming up remaining models...")
            for name in model_names:
                if name == "whisper":
                    if not whisper_first:
                        # Whisper not loaded first, warmup now
                        logger.info(f"[WARMUP] {name}...")
                        result = models["whisper"].transcribe(wav_path)
                        assert isinstance(result, str)
                        logger.info(f"{name} warmup result: {result!r}")
                        # Release trace
                        models["whisper"].release_decoder_trace()
                        gc.collect()
                    # else: already warmed up above
                    continue

                if name == "bert":
                    if bert_first:
                        # Already warmed up during staged load
                        continue
                    logger.info(f"[WARMUP] {name}...")
                    result = models["bert"].qa(
                        "How many chips?",
                        "The N300 contains two Wormhole B0 chips.",
                    )
                    assert isinstance(result, str) and len(result.strip()) > 0
                    logger.info(f"{name} warmup result: {result!r}")
                    continue

                if name == "owlvit":
                    logger.info(f"[WARMUP] {name}...")
                    result = models["owlvit"].detect(img_path, "red block")
                    assert isinstance(result, list)
                    logger.info(f"{name} warmup found {len(result)} detections")

                elif name == "speecht5":
                    logger.info(f"[WARMUP] {name}...")
                    models["speecht5"]._warmup()
                    out_path = f"{tmpdir}/speecht5_warmup.wav"
                    result = models["speecht5"].synthesize("Warmup.", out_path)
                    audio, sr = sf.read(result)
                    assert len(audio) > 0
                    logger.info(f"{name} warmup: {len(audio)/sr:.2f}s")

            # --- Inference Phase ---
            logger.info("[INFER] Running inference on all models...")
            results = {}

            for name in model_names:
                logger.info(f"[INFER] {name}...")

                if name == "whisper":
                    result = models["whisper"].transcribe(wav_path)
                    assert isinstance(result, str)
                    results[name] = result
                    logger.info(f"{name}: {result!r}")

                elif name == "bert":
                    result = models["bert"].qa(
                        "What chips are in N300?",
                        "The N300 uses two Wormhole B0 chips.",
                    )
                    assert isinstance(result, str)
                    results[name] = result
                    logger.info(f"{name}: {result!r}")

                elif name == "owlvit":
                    result = models["owlvit"].detect(img_path, "green block")
                    assert isinstance(result, list)
                    results[name] = f"{len(result)} detections"
                    logger.info(f"{name}: {len(result)} detections")

                elif name == "speecht5":
                    out_path = f"{tmpdir}/speecht5_infer.wav"
                    result = models["speecht5"].synthesize("Testing triple.", out_path)
                    audio, sr = sf.read(result)
                    assert len(audio) > 0
                    results[name] = f"{len(audio)/sr:.2f}s"
                    logger.info(f"{name}: {len(audio)/sr:.2f}s")

            # --- Cleanup ---
            if "whisper" in models and hasattr(models["whisper"], "close"):
                models["whisper"].close()

        logger.info("=" * 60)
        logger.info(f"PASS: {test_name}")
        for name, result in results.items():
            logger.info(f"  {name}: {result}")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: {test_name}: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    """Run all triple tests or a specific one."""
    import argparse

    parser = argparse.ArgumentParser(description="Level 2 Triple Tests")
    parser.add_argument("--test", type=str, help="Specific test name to run")
    parser.add_argument("--list", action="store_true", help="List all tests")
    args = parser.parse_args()

    if args.list:
        print("Available triple tests:")
        for name, desc, models in TRIPLES:
            print(f"  {name}: {desc}")
        return

    if args.test:
        for name, desc, models in TRIPLES:
            if name == args.test:
                success = run_triple_test(name, desc, models)
                sys.exit(0 if success else 1)
        logger.error(f"Unknown test: {args.test}")
        sys.exit(1)

    # Run all triples
    results = {}
    for name, desc, models in TRIPLES:
        try:
            run_triple_test(name, desc, models)
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            # Reset device after failure
            try:
                import subprocess

                subprocess.run(["tt-smi", "-r"], timeout=30)
            except Exception:
                pass

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for test_name, status in results.items():
        logger.info(f"  {test_name:25s}: {status}")

    all_pass = all(s == "PASS" for s in results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
