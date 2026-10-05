# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end word error rate of whisper-large-v3 on LibriSpeech's 73-clip validation dummy.

Runs the production pipeline (the demo's create_functional_whisper_for_conditional_generation_inference_pipeline:
encoder trace, two command queues, decode trace) on every clip, one request after another through one
pipeline, and scores the transcripts against the ground truth with Whisper's English text normalizer.
"""

import glob
import os

import jiwer
import pytest
from datasets import load_dataset, load_from_disk
from loguru import logger
from transformers import AutoProcessor

from models.demos.audio.whisper.demo.demo import create_functional_whisper_for_conditional_generation_inference_pipeline
from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_L1_SMALL_SIZE,
    WHISPER_TRACE_REGION_SIZE,
)

MODEL_NAME = "openai/whisper-large-v3"
SAMPLING_RATE = 16000

# Word error rate over all clips together.
MAX_CORPUS_WER = 0.04
# A single clip above this is a collapse (empty text, wrong language, a repetition loop: these score
# near or above 1.0), which the corpus average would hide. One wrong name in a two-word clip is 0.5.
MAX_CLIP_WER = 0.75


def load_librispeech_dummy():
    """LibriSpeech's 73-clip validation dummy, as in test_whisper_modules (Arrow cache first, else download)."""
    hf_datasets = os.path.join(os.environ.get("HF_HOME", ""), "datasets")
    arrow_dirs = (
        [
            os.path.dirname(p)
            for p in glob.glob(
                os.path.join(hf_datasets, "hf-internal-testing___parquet", "clean-*", "**", "dataset_info.json"),
                recursive=True,
            )
        ]
        if os.path.isdir(hf_datasets)
        else []
    )
    if arrow_dirs:
        return load_from_disk(sorted(arrow_dirs)[-1])
    return load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE, "num_command_queues": 2}],
    indirect=True,
)
def test_librispeech_wer(mesh_device):
    pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        mesh_device, MODEL_NAME, language="en", task="transcribe", batch_size_per_device=1
    )
    normalize = AutoProcessor.from_pretrained(MODEL_NAME).tokenizer.normalize
    ds = load_librispeech_dummy()

    references, hypotheses = [], []
    for idx, sample in enumerate(ds):
        text, _, _ = pipeline([(SAMPLING_RATE, sample["audio"]["array"])])
        text = text[0] if isinstance(text, list) else text
        references.append(normalize(sample["text"]))
        hypotheses.append(normalize(text))
        logger.info(f"clip {idx}: WER {jiwer.wer(references[-1], hypotheses[-1]):.3f}  {text.strip()[:80]}")

    corpus_wer = jiwer.wer(references, hypotheses)
    clip_wers = [jiwer.wer(r, h) for r, h in zip(references, hypotheses)]
    collapsed = [i for i, w in enumerate(clip_wers) if w > MAX_CLIP_WER]
    logger.info(f"{len(ds)} clips: corpus WER {corpus_wer:.4f}, worst clip {max(clip_wers):.3f}")
    assert not collapsed, f"clips {collapsed} have WER above {MAX_CLIP_WER}: " + "; ".join(
        f"{i}: {hypotheses[i]!r}" for i in collapsed
    )
    assert corpus_wer <= MAX_CORPUS_WER, f"corpus WER {corpus_wer:.4f} is above {MAX_CORPUS_WER}"
