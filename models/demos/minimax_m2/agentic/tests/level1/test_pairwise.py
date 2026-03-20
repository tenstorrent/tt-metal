#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Level 1 Test: Pairwise model combinations on shared N300 device.

Tests every pair of models:
1. Open device (shared params)
2. Load Model A, warmup A, run inference A
3. For trace models: release trace
4. Load Model B, warmup B, run inference B
5. Close device

Model pairs to test (6 combinations, excluding LLM):
- Whisper + BERT
- Whisper + OWL-ViT
- Whisper + SpeechT5
- BERT + OWL-ViT
- BERT + SpeechT5
- OWL-ViT + SpeechT5
"""
import gc
import sys
import tempfile
from pathlib import Path

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


# ─── Model Runners ───────────────────────────────────────────────────────────


class ModelRunner:
    """Base class for model runners."""

    name: str = "base"
    uses_trace: bool = False
    uses_chip0: bool = False  # If True, use chip0 submesh instead of full mesh

    def __init__(self, mesh, chip0, tmpdir: str):
        self.mesh = mesh
        self.chip0 = chip0
        self.tmpdir = tmpdir
        self.model = None

    def load(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def release_trace(self):
        """Release trace if this model uses one."""

    def close(self):
        """Clean up model resources."""


class WhisperRunner(ModelRunner):
    name = "Whisper"
    uses_trace = True
    uses_chip0 = False

    def load(self):
        from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

        self.wav_path = f"{self.tmpdir}/whisper_test.wav"
        make_test_wav(self.wav_path)
        self.model = WhisperTool(mesh_device=self.mesh)

    def warmup(self):
        result = self.model.transcribe(self.wav_path)
        assert isinstance(result, str)
        return result

    def inference(self):
        result = self.model.transcribe(self.wav_path)
        assert isinstance(result, str)
        return result

    def release_trace(self):
        self.model.release_decoder_trace()

    def close(self):
        if self.model:
            self.model.close()


class BERTRunner(ModelRunner):
    name = "BERT"
    uses_trace = False
    uses_chip0 = False

    def load(self):
        from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

        self.model = BERTTool(mesh_device=self.mesh)

    def warmup(self):
        result = self.model.qa(
            "How many chips does the N300 have?",
            "The N300 contains two Wormhole B0 chips.",
        )
        assert isinstance(result, str) and len(result.strip()) > 0
        return result

    def inference(self):
        result = self.model.qa(
            "What type of chips are in N300?",
            "The N300 uses two Wormhole B0 chips connected by high-bandwidth Ethernet.",
        )
        assert isinstance(result, str)
        return result


class OWLViTRunner(ModelRunner):
    name = "OWL-ViT"
    uses_trace = False
    uses_chip0 = True

    def load(self):
        from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool

        self.img_path = f"{self.tmpdir}/owlvit_test.png"
        make_test_image(self.img_path)
        self.model = OWLViTTool(mesh_device=self.chip0)

    def warmup(self):
        result = self.model.detect(self.img_path, "red block, green block")
        assert isinstance(result, list)
        return result

    def inference(self):
        result = self.model.detect(self.img_path, "colored shape")
        assert isinstance(result, list)
        return result


class SpeechT5Runner(ModelRunner):
    name = "SpeechT5"
    uses_trace = False
    uses_chip0 = True

    def load(self):
        from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

        self.model = SpeechT5Tool(mesh_device=self.chip0, warmup_on_init=False)

    def warmup(self):
        self.model._warmup()
        out_path = f"{self.tmpdir}/speecht5_warmup.wav"
        result = self.model.synthesize("Warmup audio.", out_path)
        audio, sr = sf.read(result)
        assert len(audio) > 0
        return f"{len(audio)/sr:.2f}s"

    def inference(self):
        out_path = f"{self.tmpdir}/speecht5_infer.wav"
        result = self.model.synthesize("Testing speech synthesis.", out_path)
        audio, sr = sf.read(result)
        assert len(audio) > 0
        return f"{len(audio)/sr:.2f}s"


# ─── Model Registry ──────────────────────────────────────────────────────────

MODELS = {
    "whisper": WhisperRunner,
    "bert": BERTRunner,
    "owlvit": OWLViTRunner,
    "speecht5": SpeechT5Runner,
}

# Pairs to test (excluding LLM which requires HF auth)
PAIRS = [
    ("whisper", "bert"),
    ("whisper", "owlvit"),
    ("whisper", "speecht5"),
    ("bert", "owlvit"),
    ("bert", "speecht5"),
    ("owlvit", "speecht5"),
]


# ─── Pairwise Test Runner ────────────────────────────────────────────────────


def run_pairwise_test(model_a_name: str, model_b_name: str) -> bool:
    """
    Run a pairwise test: load A, warmup A, infer A, (release trace if needed), load B, warmup B, infer B.
    """
    logger.info("=" * 60)
    logger.info(f"Level 1: Pairwise Test - {model_a_name.upper()} + {model_b_name.upper()}")
    logger.info("=" * 60)

    mesh = open_n300_device()
    try:
        chip0 = (
            mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)) if mesh.get_num_devices() > 1 else mesh
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Model A ---
            logger.info(f"[A] Loading {model_a_name}...")
            runner_a = MODELS[model_a_name](mesh, chip0, tmpdir)
            runner_a.load()

            logger.info(f"[A] Warmup {model_a_name}...")
            result_a_warm = runner_a.warmup()
            logger.info(f"[A] Warmup result: {result_a_warm!r}")

            logger.info(f"[A] Inference {model_a_name}...")
            result_a_infer = runner_a.inference()
            logger.info(f"[A] Inference result: {result_a_infer!r}")

            # Release trace if needed before loading B
            if runner_a.uses_trace:
                logger.info(f"[A] Releasing {model_a_name} trace...")
                runner_a.release_trace()
                gc.collect()
                try:
                    ttnn.synchronize_device(mesh)
                except Exception as e:
                    logger.warning(f"synchronize_device after trace release: {e}")

            # --- Model B ---
            logger.info(f"[B] Loading {model_b_name}...")
            runner_b = MODELS[model_b_name](mesh, chip0, tmpdir)
            runner_b.load()

            logger.info(f"[B] Warmup {model_b_name}...")
            result_b_warm = runner_b.warmup()
            logger.info(f"[B] Warmup result: {result_b_warm!r}")

            logger.info(f"[B] Inference {model_b_name}...")
            result_b_infer = runner_b.inference()
            logger.info(f"[B] Inference result: {result_b_infer!r}")

            # Cleanup
            runner_a.close()
            runner_b.close()

        logger.info("=" * 60)
        logger.info(f"PASS: {model_a_name.upper()} + {model_b_name.upper()}")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"FAIL: {model_a_name.upper()} + {model_b_name.upper()}: {e}")
        raise
    finally:
        ttnn.close_mesh_device(mesh)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    """Run all pairwise tests or a specific pair."""
    import argparse

    parser = argparse.ArgumentParser(description="Level 1 Pairwise Tests")
    parser.add_argument("--pair", type=str, help="Specific pair to test, e.g., 'whisper+bert'")
    parser.add_argument("--list", action="store_true", help="List all pairs")
    args = parser.parse_args()

    if args.list:
        print("Available pairs:")
        for a, b in PAIRS:
            print(f"  {a}+{b}")
        return

    if args.pair:
        a, b = args.pair.lower().split("+")
        success = run_pairwise_test(a, b)
        sys.exit(0 if success else 1)

    # Run all pairs
    results = {}
    for a, b in PAIRS:
        try:
            run_pairwise_test(a, b)
            results[f"{a}+{b}"] = "PASS"
        except Exception as e:
            results[f"{a}+{b}"] = f"FAIL: {e}"
            # Reset device after failure
            try:
                import subprocess

                subprocess.run(["tt-smi", "-r"], timeout=30)
            except Exception:
                pass

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for pair, status in results.items():
        logger.info(f"  {pair:20s}: {status}")

    all_pass = all(s == "PASS" for s in results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
