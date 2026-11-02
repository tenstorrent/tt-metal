# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""One CustomVoice utterance end to end, scored the way CI scores the LLM demos.

Accuracy is teacher forcing: the device is fed the frames the CPU reference sampled
(`reference_outputs/`, written by `generate_reference.py`) and scored on whether its argmax
at each of the 16 codebooks of every frame is the reference's (top-1) or among its own five
most likely (top-5). Determinism: the same seed twice gives the same frames and the same
audio. Perf, warm, at batch 1: the time to the first frame and frames per second, reported
as time-to-token and tokens/s/user.

Accuracy and perf go out as two benchmark payloads, which CI checks against
`models/model_targets.yaml`. Needs the CustomVoice checkpoint at the ambient size.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/e2e/test_e2e.py
"""

import time

import pytest
import torch

from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.tests.checkpoints import use_release
from models.demos.audio.qwen3_tts.tests.e2e import references
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    yield from use_release("custom_voice")


def _speak(pipeline, profiler=None):
    """(waveform, codes, seconds to the first frame) for the reference utterance.

    With a profiler, charges `inference_prefill` up to the first frame and `inference_decode`
    for the rest, the steps the benchmark payload names.
    """
    profiler = profiler or BenchmarkProfiler()
    first = []

    def on_frame(step, frame):
        if not first:
            profiler.end("inference_prefill")
            first.append(time.time() - started)
            profiler.start("inference_decode")

    pipeline.reseed(references.SEED)
    started = time.time()
    profiler.start("inference_prefill")
    waveform, codes = pipeline.generate(
        references.TEXT, speaker=references.SPEAKER, language=references.LANGUAGE, on_frame=on_frame
    )
    profiler.end("inference_decode")
    return waveform, codes, first[0]


def _teacher_forced(pipeline, codes):
    """The device's top five at every codebook of every frame, fed `codes` [T, 16] as its picks."""
    frames = []

    def pick(row, **_):
        if len(frames) == len(codes):
            return pipeline.eos
        frames.append([torch.topk(row, 5).indices])
        return int(codes[len(frames) - 1, 0])

    def inner(row):
        frame = frames[-1]
        frame.append(torch.topk(row, 5).indices)
        return int(codes[len(frames) - 1, len(frame) - 1])

    pipeline._pick, pipeline._inner_pick = pick, lambda: inner
    forced = pipeline.codes(references.TEXT, references.SPEAKER, references.LANGUAGE, max_frames=len(codes) + 1)
    assert torch.equal(forced, codes.long()), "the forced frames did not come back as fed"
    return torch.stack([torch.stack(frame) for frame in frames])


def _save(profiler, ml_model_name, measurements):
    data = BenchmarkData()
    for step, name, value in measurements:
        data.add_measurement(profiler, 0, step, name, value)
    data.save_partial_run_json(
        profiler, run_type="demo", ml_model_name=ml_model_name, ml_model_type="audio", batch_size=1
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_e2e(device):
    reference = torch.load(references.path())
    pipeline = Qwen3TTSPipeline(device, max_frames=references.MAX_FRAMES, seed=references.SEED)
    model = weights.sibling_repo("custom_voice")[0]

    # The first run builds the kernels, the codec's for this length included; the second is warm.
    waveform, codes, _ = _speak(pipeline)
    perf = BenchmarkProfiler()
    perf.start("run")
    again, again_codes, to_first_frame = _speak(pipeline, perf)
    perf.end("run")
    assert torch.equal(codes, again_codes), "the same seed gave different frames"
    assert torch.equal(waveform, again), "the same frames gave different audio"
    timings = dict(pipeline.last_timings)
    frames_per_second = again_codes.shape[0] / timings["decode_s"]

    accuracy = BenchmarkProfiler()
    accuracy.start("run")
    accuracy.start("inference_decode")
    top5 = _teacher_forced(pipeline, reference["codes"])
    accuracy.end("inference_decode")
    accuracy.end("run")
    truth = reference["top1"].long()
    top1_accuracy = 100 * float((top5[..., 0] == truth).float().mean())
    top5_accuracy = 100 * float((top5 == truth[..., None]).any(-1).float().mean())

    print(
        f"\n  {model}: {again_codes.shape[0]} frames, first frame {to_first_frame:.3f} s "
        f"(prefill {timings['prefill_s']:.3f} s, capture {timings['capture_s']:.3f} s), "
        f"{frames_per_second:.1f} frames/s; teacher forced over {truth.numel()} codes: "
        f"top-1 {top1_accuracy:.2f}%, top-5 {top5_accuracy:.2f}%"
    )
    _save(
        perf,
        model,
        [
            ("inference_prefill", "time_to_token", to_first_frame),
            ("inference_decode", "tokens/s/user", frames_per_second),
            ("inference_decode", "tokens/s", frames_per_second),
        ],
    )
    _save(
        accuracy,
        model,
        [
            ("inference_decode", "top1_token_accuracy", top1_accuracy),
            ("inference_decode", "top5_token_accuracy", top5_accuracy),
        ],
    )
