# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The code predictor's frame trace: all 15 codebooks sampled on device in one replay.

The step-by-step path, which reads each head's logits to host and samples there, is the
reference. Greedy must pick what it picks, step for step. Sampling cannot be compared
pick by pick, so the first codebook, whose logits depend only on the frame's inputs, is
drawn many times and its distribution compared with the softmax over the top k.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/pcc/test_predictor_frame_trace.py
"""

import time

import pytest
import torch
from scipy.stats import chisquare

import ttnn
from models.demos.audio.qwen3_tts.tests.reference_helpers import code_predictor_prompt
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor_decode import (
    TtCodePredictorCachedDecoder,
    preprocess_cached_predictor_parameters,
)

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

TEMPERATURE, TOP_K = 0.9, 50
DRAWS = 2000
# A chi-square test of the draws against the exact distribution fails a correct sampler
# once in a thousand runs. `ttnn.sampling`, which never draws past the top ~18 of a
# 50-way draw, is rejected far below this.
MIN_P_VALUE = 1e-3
FRAMES = 50


def _predictor(device, temperature=None, top_k=None):
    predictor = TtCodePredictorCachedDecoder(device, preprocess_cached_predictor_parameters(device))
    if top_k is not None:
        predictor.sample_on_device(temperature, top_k)
    return predictor


def _traced(predictor):
    predictor.warmup()
    predictor.capture()
    return predictor


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_greedy_frames_match_the_step_by_step_path(device):
    talker_hidden, first_code, _, _ = code_predictor_prompt()
    host = _predictor(device)
    expected = list(host.generate(talker_hidden, first_code))
    eager = _predictor(device, 1.0, 1)
    assert list(eager.generate(talker_hidden, first_code)) == expected, "eager frame"
    traced = _traced(eager)
    for replay in range(3):
        assert list(traced.generate(talker_hidden, first_code)) == expected, f"traced frame, replay {replay}"
    traced.release()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_first_codebook_is_drawn_from_the_top_k_softmax(device):
    talker_hidden, first_code, _, _ = code_predictor_prompt()
    host = _predictor(device)
    host.reset()
    host._fill_from_host(talker_hidden.reshape(1, 1, -1))
    host._run(0)
    host._fill_from_table(host.p["talker_codec_embedding_device"], first_code)
    logits = ttnn.to_torch(host._head(host._run(1), 0)).float().reshape(-1)
    scaled = logits / TEMPERATURE
    scaled[scaled < torch.topk(scaled, TOP_K).values[-1]] = -float("inf")
    expected = torch.softmax(scaled, dim=-1)

    traced = _traced(_predictor(device, TEMPERATURE, TOP_K))
    generator = torch.Generator().manual_seed(0)
    draws = torch.tensor([traced.generate(talker_hidden, first_code, generator=generator)[0] for _ in range(DRAWS)])
    traced.release()

    outside = int((expected[draws] == 0).sum())
    assert outside == 0, f"{outside} draws fell outside the top {TOP_K}"
    # Ids expected at least 5 times are their own bins; the rest share one, as chi-square needs.
    observed = torch.bincount(draws, minlength=expected.numel()).double()
    counts = expected.double() / expected.double().sum() * DRAWS
    own = counts >= 5
    rest = counts[~own].sum()
    observed = torch.cat([observed[own], observed[~own].sum().reshape(1)]) if rest > 0 else observed[own]
    counts = torch.cat([counts[own], rest.reshape(1)]) if rest > 0 else counts[own]
    p_value = chisquare(observed.numpy(), counts.numpy()).pvalue
    distance = 0.5 * float((observed - counts).abs().sum()) / DRAWS
    host = torch.multinomial(expected, DRAWS, replacement=True, generator=torch.Generator().manual_seed(1))
    host_distance = 0.5 * float((torch.bincount(host, minlength=expected.numel()) / DRAWS - expected).abs().sum())
    print(
        f"\n  chi-square p {p_value:.3g}, total variation {distance:.4f} "
        f"(host sampler at the same size: {host_distance:.4f}), {len(set(draws.tolist()))} distinct"
    )
    assert p_value >= MIN_P_VALUE, f"chi-square p {p_value:.3g} < {MIN_P_VALUE}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_a_seeded_generator_repeats_its_frames(device):
    talker_hidden, first_code, _, _ = code_predictor_prompt()
    traced = _traced(_predictor(device, TEMPERATURE, TOP_K))
    run = lambda seed: [
        traced.generate(talker_hidden, first_code, generator=generator)
        for generator in [torch.Generator().manual_seed(seed)]
        for _ in range(5)
    ]
    assert run(7) == run(7)
    assert run(7) != run(8)
    traced.release()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_frame_trace_is_faster_than_step_by_step(device):
    talker_hidden, first_code, _, _ = code_predictor_prompt()
    talker_hidden = ttnn.from_torch(talker_hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    generator = torch.Generator().manual_seed(0)

    def per_frame(predictor, **kwargs):
        predictor.generate(talker_hidden, first_code, **kwargs)
        started = time.time()
        for _ in range(FRAMES):
            predictor.generate(talker_hidden, first_code, **kwargs)
        return 1000 * (time.time() - started) / FRAMES

    # One trace at a time: each predictor captures, measures and releases before the next.
    step = _traced(_predictor(device))
    step_ms = per_frame(step, pick=lambda row: int(row.argmax()))
    step.release()
    frame = _traced(_predictor(device, TEMPERATURE, TOP_K))
    frame_ms = per_frame(frame, generator=generator)
    frame.release()
    print(f"\n  step by step {step_ms:.2f} ms/frame, frame trace {frame_ms:.2f} ms/frame")
    assert frame_ms < step_ms
