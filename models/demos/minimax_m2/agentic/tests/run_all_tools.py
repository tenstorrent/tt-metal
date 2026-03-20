#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run all 5 agentic tools sequentially on one shared N300 (1,2) mesh device.

ARCHITECTURE: All models run on chip0 submesh to avoid deadlocks.
Mixing full-mesh and chip0-submesh models causes hangs. chip1 is unused.

Flow:
  1) Load Whisper + LLM on chip0; warmup Whisper (trace capture).
  2) Release Whisper persistent decoder trace (safe DRAM allocations).
  3) Load BERT, OWL-ViT, SpeechT5 on chip0; warmup all.
  4) Inference phase for each tool on the same open device.
"""
import argparse
import gc
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import soundfile as sf
from loguru import logger
from PIL import Image

import ttnn
from models.demos.minimax_m2.agentic.loader import open_n300_device

# ── helpers ──────────────────────────────────────────────────────────────────


def make_wav(path, duration=2.0, sr=16000):
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sf.write(path, (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32), sr)


def make_img(path):
    d = np.zeros((512, 512, 3), dtype=np.uint8)
    d[40:200, 40:200] = [220, 30, 30]  # red
    d[312:472, 176:336] = [30, 180, 30]  # green
    Image.fromarray(d, "RGB").save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Load all models, then warmup and run sequentially on one N300 device")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM load/inference (useful when HF gated model auth is unavailable).",
    )
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    logger.info("Opening N300 (1,2) mesh device for full agentic run...")
    mesh = open_n300_device()
    results = {}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = f"{tmpdir}/test.wav"
            img = f"{tmpdir}/test.png"
            tts_warmup_wav = f"{tmpdir}/tts_warmup.wav"
            tts_infer_wav = f"{tmpdir}/tts_infer.wav"
            make_wav(wav)
            make_img(img)

            models = SimpleNamespace(llm=None, whisper=None, speecht5=None, owlvit=None, bert=None)

            chip0 = (
                mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
                if mesh.get_num_devices() > 1
                else mesh
            )

            logger.info("=" * 60)
            logger.info("PHASE 0a: Load Whisper (+ LLM only) on chip0 — unified submesh for all models")
            from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

            # NOTE: All models on chip0 to avoid full-mesh vs chip0-submesh conflicts.
            # Full-mesh Whisper would conflict with chip0 BERT/SpeechT5.
            models.whisper = WhisperTool(mesh_device=chip0)

            if not args.skip_llm:
                from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import LLMTool

                models.llm = LLMTool(mesh_device=chip0)

            logger.info("=" * 60)
            logger.info("PHASE 1a: LLM + Whisper warmup (trace capture)")

            if args.skip_llm:
                logger.warning("[1/5] LLM warmup skipped (--skip-llm)")
                results["llm_warmup"] = "SKIP"
            else:
                logger.info("[1/5] LLM warmup call")
                llm_warm = models.llm.generate_response(
                    messages=[{"role": "user", "content": "Respond with exactly: WARMUP_OK"}],
                    max_new_tokens=16,
                )
                assert isinstance(llm_warm, str) and len(llm_warm) > 0
                results["llm_warmup"] = "PASS"

            logger.info("[2/5] Whisper warmup call (trace capture)")
            whisper_warm = models.whisper.transcribe(wav)
            assert isinstance(whisper_warm, str)
            results["whisper_warmup"] = "PASS"

            logger.info("Releasing Whisper decoder trace before loading other models (safe DRAM allocation)")
            models.whisper.release_decoder_trace()
            gc.collect()
            try:
                ttnn.synchronize_device(chip0)
            except Exception as e:
                logger.warning(f"synchronize_device after Whisper trace release: {e}")

            logger.info("=" * 60)
            logger.info(
                "PHASE 0b: Load BERT, OWL-ViT, SpeechT5 on chip0 "
                "(BERT on full mesh conflicts with chip0 submesh models)"
            )
            from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool
            from models.demos.minimax_m2.agentic.tool_wrappers.owlvit_tool import OWLViTTool
            from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool

            # NOTE: BERT must use chip0 — running BERT on full mesh while SpeechT5/OWL-ViT
            # use chip0 submesh causes hangs (see agentic/tests/NOT_POSSIBLE.md)
            models.bert = BERTTool(mesh_device=chip0)
            models.owlvit = OWLViTTool(mesh_device=chip0)
            models.speecht5 = SpeechT5Tool(mesh_device=chip0, warmup_on_init=False)
            gc.collect()
            try:
                ttnn.synchronize_device(chip0)
            except Exception as e:
                logger.warning(f"synchronize_device after loading remaining models: {e}")

            logger.info("=" * 60)
            logger.info("PHASE 1b: Warmup OWL-ViT, BERT, SpeechT5")

            logger.info("[3/5] BERT warmup call")
            bert_warm = models.bert.qa(
                "How many chips does the N300 have?",
                "The N300 contains two Wormhole B0 chips.",
            )
            assert isinstance(bert_warm, str)
            results["bert_warmup"] = "PASS"

            logger.info("[4/5] OWL-ViT warmup call")
            owl_warm = models.owlvit.detect(img, "red block, green block")
            assert isinstance(owl_warm, list)
            results["owlvit_warmup"] = "PASS"

            logger.info("[5/5] SpeechT5 warmup synth call")
            models.speecht5._warmup()
            tts_warm_out = models.speecht5.synthesize("Warmup audio.", tts_warmup_wav)
            warm_audio, warm_sr = sf.read(tts_warm_out)
            assert len(warm_audio) > 0 and warm_sr > 0
            results["speecht5_warmup"] = "PASS"

            logger.info("=" * 60)
            logger.info("PHASE 2: Inference run for all models (sequential, same device)")

            if args.skip_llm:
                logger.warning("[1/5] LLM inference skipped (--skip-llm)")
                results["llm_infer"] = "SKIP"
            else:
                logger.info("[1/5] LLM inference")
                llm_out = models.llm.generate_response(
                    messages=[{"role": "user", "content": "What is 3 + 4? Answer with one token."}],
                    max_new_tokens=16,
                )
                assert isinstance(llm_out, str) and len(llm_out) > 0
                logger.info(f"LLM output: {llm_out!r}")
                results["llm_infer"] = "PASS"

            logger.info("[2/5] Whisper inference")
            whisper_out = models.whisper.transcribe(wav)
            assert isinstance(whisper_out, str)
            logger.info(f"Whisper output: {whisper_out!r}")
            results["whisper_infer"] = "PASS"

            logger.info("[3/5] BERT inference")
            bert_out = models.bert.qa(
                "What chips are used in N300?",
                "The N300 uses two Wormhole B0 chips connected by high-bandwidth Ethernet.",
            )
            assert isinstance(bert_out, str) and len(bert_out.strip()) > 0
            logger.info(f"BERT output: {bert_out!r}")
            results["bert_infer"] = "PASS"

            logger.info("[4/5] OWL-ViT inference")
            detections = models.owlvit.detect(img, "red block, green block")
            assert isinstance(detections, list)
            logger.info(f"OWL-ViT detections: {len(detections)}")
            for d in detections:
                assert "label" in d and "score" in d and "bbox" in d
            results["owlvit_infer"] = "PASS"

            logger.info("[5/5] SpeechT5 inference")
            tts_out = models.speecht5.synthesize("Testing text to speech on shared N300.", tts_infer_wav)
            audio, sr = sf.read(tts_out)
            assert len(audio) > 0, "Empty audio"
            logger.info(f"TTS output: {len(audio)/sr:.2f}s at {sr}Hz")
            results["speecht5_infer"] = "PASS"

            # Explicit whisper close helps avoid teardown edge cases.
            if models.whisper is not None and hasattr(models.whisper, "close"):
                models.whisper.close()

            logger.info("=" * 60)
            for name, status in sorted(results.items()):
                logger.info(f"  {name:18s}: {status}")
            assert all(v in {"PASS", "SKIP"} for v in results.values()), f"Some stages FAILED: {results}"
            logger.info("ALL MODELS: LOAD->WARMUP->INFER PASS")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
