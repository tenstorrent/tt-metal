# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The cached, traced decode paths against the uncached graphs they replace.

The uncached talker and code predictor are the validated implementations (0.995 and
0.995 against the fp32 reference, in `test_talker_pcc.py` and `test_code_predictor_pcc.py`).
These tests hold the cached ones to those, on the same device and the same inputs, so a
disagreement is the cache, the positions or the trace rather than arithmetic.

**Free-running comparison would measure nothing.** One flipped pick changes the input to
every later step, and both models sit on near-ties constantly: measured 2 of 15 codebooks
agreeing after a single early flip, while the same two graphs agree on all 15 when driven
along one shared prefix. So every test here forces both paths down the same sequence and
judges each step on its own.

Two hazards these cover, both of which cost a board reset to learn:

  * A trace must be captured before any trace executes, and every buffer a warmup
    allocates must exist before the first trace does. Capturing the predictor's trace
    after the talker's had already run hung the device for 16 minutes.
  * Eager work after capture is unsafe in general, so the trace test runs no uncached
    model once a trace is live.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/pcc/test_decode_pcc.py
"""

import pytest
import torch

import ttnn
from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.tests.reference_helpers import code_predictor_prompt, codec_head, talker_prompt
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor import (
    TtCodePredictor,
    preprocess_code_predictor_parameters,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor_decode import (
    TtCodePredictorCachedDecoder,
    preprocess_cached_predictor_parameters,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import TtTalker, preprocess_talker_parameters
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker_decode import (
    TtTalkerCachedDecoder,
    preprocess_cached_talker_parameters,
)

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

STEPS = 8

# On a real dual-track prompt the two talker graphs hold 0.9995 and higher, and 30 steps
# gave identical picks throughout. The prompt here is the text track alone with codebooks
# 1 to 15 held fixed, which is further from the training distribution and measures worse:
# 0.9909 worst over these 8 steps, picks still identical. Both numbers are per step, and
# the prefill itself reproduces the uncached pass to 1.000000.
TALKER_STEP_PCC = 0.99
PREDICTOR_STEP_PCC = 0.99

# Judged on the distribution, not pick equality, which here is luck. Worst measured: 0.28.
SAMPLER_TEMPERATURE = 0.9
MAX_STEP_DISTANCE = 0.45


def sampler_distance(reference_logits, device_logits):
    """Total variation distance between the two sampling distributions, 0 to 1."""
    reference = torch.softmax(reference_logits / SAMPLER_TEMPERATURE, dim=-1)
    device = torch.softmax(device_logits / SAMPLER_TEMPERATURE, dim=-1)
    return float(0.5 * (reference - device).abs().sum())


# A disagreement counts only where the uncached graph prefers its pick by more than this.
MAX_PREFERENCE_GAP = 0.25

# 0.6B measured 0.931 and a 0.875 gap by codebook 14, distance 0.247 against 1.7B's 0.105.
PREDICTOR_GATES = {
    "1b7": {"pcc": PREDICTOR_STEP_PCC, "gap": MAX_PREFERENCE_GAP},
    "0b6": {"pcc": 0.92, "gap": 1.0},
}


def pcc(left, right):
    left, right = left.float().reshape(-1), right.float().reshape(-1)
    return float(torch.corrcoef(torch.stack([left, right]))[0, 1])


def to_device(device, tensor):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.fixture
def frame_tail():
    """Codebooks 1 to 15 of one real frame, summed, plus the `tts_pad` the loop adds.

    A step's input is the sum of 16 codebook embeddings and `tts_pad`. Codebook 0 varies
    per step below, driven by what the models actually pick; the other fifteen are held at
    this one real frame's. Both paths see the identical tensor, which is what these tests
    turn on, and every step still lands in the activation range the weights were trained
    on.
    """
    from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import HostEmbeddings

    tables = HostEmbeddings()
    _, _, codes, _ = code_predictor_prompt()
    total = tables.tts_pad.clone()
    for index, code in enumerate(codes):
        total = total + tables.predictor_tables[index][code].reshape(1, 1, -1)
    return tables, total


def talker_step_inputs(tables, tail, code):
    """The next position: codebook 0's embedding over the held tail."""
    return tables.codec([int(code)]) + tail


# ── the talker ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_cached_talker_tracks_the_uncached_graph(device, frame_tail):
    """Prefill then step, both paths on one shared sequence, judged per step."""
    tables, tail = frame_tail
    head = codec_head()
    embeddings, positions = talker_prompt()
    length = embeddings.shape[1]

    cached = TtTalkerCachedDecoder(device, preprocess_cached_talker_parameters(device), max_seq=length + STEPS + 8)
    plain = TtTalker(device, preprocess_talker_parameters(device))

    def uncached_last(prompt):
        count = prompt.shape[1]
        cos, sin, mask = plain.host_inputs(count)
        hidden = plain(*(to_device(device, tensor) for tensor in (prompt, cos, sin, mask)))
        row = ttnn.slice(hidden, [0, count - 1, 0], [1, count, hidden.shape[2]])
        out = ttnn.to_torch(row).float().reshape(1, 1, -1)
        ttnn.deallocate(hidden)
        return out

    hidden_all = cached.prefill(embeddings, positions)
    row = ttnn.slice(hidden_all, [0, length - 1, 0], [1, length, hidden_all.shape[2]])
    cached_last = ttnn.to_torch(row).float().reshape(1, 1, -1)
    ttnn.deallocate(hidden_all)
    plain_last = uncached_last(embeddings)

    prefill_pcc = pcc(cached_last, plain_last)
    print(f"prefill last hidden pcc {prefill_pcc:.6f}")
    assert prefill_pcc > TALKER_STEP_PCC, "the cached prefill must reproduce the uncached pass"

    prompt = embeddings
    worst, worst_distance, agreed = 1.0, 0.0, 0
    for step in range(STEPS):
        plain_logits = plain_last.reshape(-1) @ head.T
        cached_logits = cached_last.reshape(-1) @ head.T
        cached_pick, plain_pick = int(cached_logits.argmax()), int(plain_logits.argmax())
        score = pcc(cached_last, plain_last)
        distance = sampler_distance(plain_logits, cached_logits)
        worst, worst_distance = min(worst, score), max(worst_distance, distance)
        agreed += cached_pick == plain_pick
        print(
            f"  step {step} position {length + step} pcc {score:.6f} distance {distance:.4f} "
            f"picks {cached_pick} {plain_pick}"
        )

        # Advance both on the uncached pick, so one flip cannot cascade into the rest.
        nxt = talker_step_inputs(tables, tail, plain_pick)
        prompt = torch.cat([prompt, nxt], dim=1)
        cached_last = ttnn.to_torch(cached.step(nxt, length + step)).float().reshape(1, 1, -1)
        plain_last = uncached_last(prompt)

    print(f"worst step pcc {worst:.6f}, worst distance {worst_distance:.4f}, picks agreed {agreed}/{STEPS}")
    assert worst > TALKER_STEP_PCC
    assert worst_distance < MAX_STEP_DISTANCE, f"worst sampling distance {worst_distance:.4f}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_talker_trace_replays_the_eager_step_exactly(device, frame_tail):
    """A captured step must be the step that was captured, bit for bit.

    Runs the eager steps first, then re-prefills the same prompt and replays the same
    inputs through the trace. Nothing eager runs once the trace exists, and the cache is
    cleared between the two passes so the trace starts from the state the eager pass did.
    """
    tables, tail = frame_tail
    embeddings, positions = talker_prompt()
    length = embeddings.shape[1]
    cached = TtTalkerCachedDecoder(device, preprocess_cached_talker_parameters(device), max_seq=length + STEPS + 8)

    codes = [int(code) for code in code_predictor_prompt()[2][:STEPS]]
    inputs = [talker_step_inputs(tables, tail, code) for code in codes]

    ttnn.deallocate(cached.prefill(embeddings, positions))
    eager = [ttnn.to_torch(cached.step(nxt, length + step)).float() for step, nxt in enumerate(inputs)]

    cached.reset()
    ttnn.deallocate(cached.prefill(embeddings, positions))
    cached.warmup()
    cached.capture()
    try:
        for step, nxt in enumerate(inputs):
            replayed = ttnn.to_torch(cached.step(nxt, length + step)).float()
            difference = float((replayed - eager[step]).abs().max())
            print(f"  step {step} max abs difference {difference:.3e}")
            assert difference == 0.0, f"step {step} replayed differently"
    finally:
        ttnn.release_trace(device, cached.trace_id)


# ── the code predictor ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_predictor_carries_nothing_between_frames(device):
    """The same frame twice must decode the same way, and `reset` must really clear.

    A cache that survives a frame would poison every frame after the first, which by ear
    is a sentence that starts clean and dissolves.
    """
    talker_hidden, first_code, _, _ = code_predictor_prompt()
    predictor = TtCodePredictorCachedDecoder(device, preprocess_cached_predictor_parameters(device))

    magnitude = lambda: max(float(ttnn.to_torch(cache).abs().max()) for cache in predictor.k_cache + predictor.v_cache)
    assert magnitude() == 0.0, "a fresh cache must be empty"

    first = list(predictor.generate(talker_hidden, first_code))
    assert magnitude() > 0.0, "a frame must leave keys and values behind"

    predictor.reset()
    assert magnitude() == 0.0, "reset must clear them"

    assert list(predictor.generate(talker_hidden, first_code)) == first, "frame 2 differed from frame 1"
    assert list(predictor.generate(talker_hidden, first_code)) == first, "frame 3 differed from frame 1"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_cached_predictor_tracks_the_uncached_graph(device):
    """All 15 codebooks, both paths forced along the uncached path's own codes."""
    talker_hidden, first_code, _, _ = code_predictor_prompt()
    cached = TtCodePredictorCachedDecoder(device, preprocess_cached_predictor_parameters(device))
    plain = TtCodePredictor(device, preprocess_code_predictor_parameters(device))
    groups = cached.groups

    # The uncached path's codes and logits, one per step.
    reference_codes, reference_logits = [], []
    codes = [int(first_code)]
    for step in range(groups - 1):
        embeddings = plain.build_embeddings(talker_hidden, codes)
        count = embeddings.shape[1]
        cos, sin, mask = plain.host_inputs(count)
        hidden = plain(*(to_device(device, tensor) for tensor in (embeddings, cos, sin, mask)))
        row = ttnn.slice(hidden, [0, count - 1, 0], [1, count, hidden.shape[2]])
        logits = ttnn.to_torch(ttnn.linear(row, plain.p["lm_head"][step])).float().reshape(-1)
        reference_logits.append(logits)
        reference_codes.append(int(logits.argmax()))
        codes.append(reference_codes[-1])

    cached.reset()
    cached._fill_from_host(talker_hidden.reshape(1, 1, -1))
    cached._run(0)
    cached._fill_from_table(cached.p["talker_codec_embedding_device"], first_code)
    hidden = cached._run(1)

    gates = PREDICTOR_GATES[weights.model_size()]
    worst, worst_distance, exact, wide = 1.0, 0.0, 0, []
    for step in range(groups - 1):
        row = ttnn.to_torch(cached._head(hidden, step)).float().reshape(-1)
        pick, reference = int(row.argmax()), reference_codes[step]
        score = pcc(row, reference_logits[step])
        distance = sampler_distance(reference_logits[step], row)
        worst, worst_distance = min(worst, score), max(worst_distance, distance)
        gap = float(reference_logits[step][reference] - reference_logits[step][pick])
        print(f"  codebook {step + 1} pcc {score:.6f} distance {distance:.4f} picks {pick} {reference} gap {gap:.4f}")
        if pick == reference:
            exact += 1
        elif gap > gates["gap"]:
            wide.append(f"codebook {step + 1} gap {gap:.4f}")
        if step < groups - 2:
            # The uncached path's codes, through the device-side lookup the frame loop uses.
            cached._fill_from_table(cached.p["codec_embedding_device"][step], reference)
            hidden = cached._run(2 + step)

    print(
        f"codebooks matching exactly {exact}/{groups - 1} | worst logits pcc {worst:.6f} "
        f"| worst distance {worst_distance:.4f}"
    )
    assert not wide, "codebooks the uncached graph feels strongly about: " + "; ".join(wide)
    assert worst > gates["pcc"]
    assert worst_distance < MAX_STEP_DISTANCE, f"worst sampling distance {worst_distance:.4f}"
