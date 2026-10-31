# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The vLLM talker stage, driven the way the TT vLLM plugin drives it.

No vLLM here: the test plays the plugin. It allocates the KV cache through the adapter,
prefills from a vLLM-Omni `additional_information` payload with the request on scattered
blocks, then feeds codebook 0 back one decode at a time and pops each frame.

Judged against the pipeline's own frame steps on the same codebook-0 sequence and the same
code-predictor seed: the logits and every frame must match exactly. Codebook 0 is forced to
each step's top id, so vLLM's sampler is not needed and both runs see the same inputs.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/pcc/test_generator_vllm.py
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.audio.qwen3_tts import stage_inputs
from models.demos.audio.qwen3_tts.tests.reference_helpers import synthetic_voiced_clip
from models.demos.audio.qwen3_tts.tt.generator_vllm import Qwen3TTSTalkerForConditionalGeneration
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    Qwen3TTSPipeline,
    ReferenceEncoders,
    build_streaming_clone_prefill,
)

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

TEXT = "This voice was cloned on Tenstorrent hardware."
REFERENCE_TEXT = "Good morning. This sentence is being spoken by a language model."
CLIP_SECONDS = 2.0
SAMPLE_RATE = 24000
MAX_MODEL_LEN = 512
BLOCK_SIZE = 64
NUM_BLOCKS = 24
STEPS = 12
SEED = 0
CODEBOOK_SIZE = 2048


def _payload():
    clip = synthetic_voiced_clip(seconds=CLIP_SECONDS)
    return {
        "task_type": ["Base"],
        "text": [TEXT],
        "language": ["English"],
        "ref_text": [REFERENCE_TEXT],
        "ref_audio": [[clip.tolist(), SAMPLE_RATE]],
    }


def _top(row):
    return int(row.reshape(-1).argmax())


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_stage_reproduces_the_pipelines_frames(device, expect_error):
    info = _payload()

    # The reference: the pipeline's frame steps on its own contiguous cache.
    pipeline = Qwen3TTSPipeline(device, max_frames=MAX_MODEL_LEN, codec=False)
    # The stage encodes clips with the host encoders; so does its reference here.
    reference = stage_inputs.clone_reference(info, device, ReferenceEncoders())
    # The length vLLM-Omni's estimate gives this payload: the builder's own, unfitted.
    prompt_len = build_streaming_clone_prefill(TEXT, reference, "English", pipeline.tables)[0].shape[1]
    embeddings, feed = stage_inputs.build_stage_prefill(info, reference, prompt_len, pipeline.tables)
    pipeline.reseed(SEED)
    request, _ = pipeline.start(embeddings, MAX_MODEL_LEN - prompt_len, feed)
    inner = pipeline._inner_pick()
    expected_logits, expected_frames, firsts = [], [], []
    for _ in range(STEPS):
        row = pipeline.codec_logits(request)
        row[0] = -float("inf")
        expected_logits.append(row.clone())
        firsts.append(_top(row))
        frame = pipeline.predict(request, firsts[-1], inner)
        expected_frames.append(frame)
        pipeline.feed_back(request, frame)
    pipeline.release()
    del pipeline, request

    # The stage, as the plugin calls it, with the request on scattered blocks.
    stage = Qwen3TTSTalkerForConditionalGeneration.initialize_vllm_model(
        SimpleNamespace(_name_or_path=""), device, 1, max_seq_len=MAX_MODEL_LEN
    )
    config = stage.model_capabilities
    assert config["supports_stage_inputs"] and config["supports_stage_outputs"]
    kv_cache = stage.allocate_kv_cache((NUM_BLOCKS, 8, BLOCK_SIZE, 128), torch.bfloat16, 28)
    blocks = torch.randperm(NUM_BLOCKS - 1, generator=torch.Generator().manual_seed(1))[: MAX_MODEL_LEN // BLOCK_SIZE]
    page_table = (blocks + 1).to(torch.int32).reshape(1, -1)  # block 0 is vLLM's null block
    stage.pipeline.reseed(SEED)

    logits, rope_deltas = stage.prefill_forward(
        tokens=torch.ones(1, prompt_len, dtype=torch.int32),
        start_pos=torch.zeros(1, dtype=torch.int32),
        prompt_lens=torch.tensor([prompt_len]),
        page_table=page_table,
        kv_cache=kv_cache,
        empty_slots=[0],
        stage_inputs=[info],
    )
    assert rope_deltas.tolist() == [0], "M-RoPE's three axes agree, so no position delta"
    assert stage.pop_stage_outputs() == [], "prefill completes no frame"
    for step in range(STEPS):
        assert tuple(logits.shape) == (1, 1, expected_logits[step].numel())
        difference = float((logits.reshape(-1) - expected_logits[step]).abs().nan_to_num(posinf=0, neginf=0).max())
        assert difference == 0.0, f"step {step} logits differ by {difference:.3e}"
        assert int(logits.reshape(-1).argmax()) == firsts[step]
        logits = stage.decode_forward(
            tokens=torch.tensor([[firsts[step]]], dtype=torch.int32),
            start_pos=torch.tensor([prompt_len + step], dtype=torch.int32),
            page_table=page_table,
            kv_cache=kv_cache,
        )
        (frame,) = stage.pop_stage_outputs()
        assert frame.tolist() == expected_frames[step], f"frame {step} differs"
    print(f"{STEPS} frames through the stage, identical to the pipeline's; prompt {prompt_len} positions")

    with expect_error(ValueError, "decodes at position"):
        stage.decode_forward(
            tokens=torch.tensor([[firsts[0]]], dtype=torch.int32),
            start_pos=torch.tensor([prompt_len], dtype=torch.int32),
            page_table=page_table,
            kv_cache=kv_cache,
        )
    for code in (0, CODEBOOK_SIZE, stage.pipeline.eos):
        with expect_error(ValueError, f"codec id {code}"):
            stage.decode_forward(
                tokens=torch.tensor([[code]], dtype=torch.int32),
                start_pos=torch.tensor([prompt_len + STEPS], dtype=torch.int32),
                page_table=page_table,
                kv_cache=kv_cache,
            )
    stage.release_request(0)
    stage.release_persistent_capture()


SECOND_TEXT = "A second request joins while the first one is still speaking."
# Frames the first request decodes before the second one's prefill lands.
JOIN_AT = 3


def _stage(device, batch, num_blocks):
    stage = Qwen3TTSTalkerForConditionalGeneration.initialize_vllm_model(
        SimpleNamespace(_name_or_path=""), device, batch, max_seq_len=MAX_MODEL_LEN
    )
    kv_cache = stage.allocate_kv_cache((num_blocks, 8, BLOCK_SIZE, 128), torch.bfloat16, 28)
    # Greedy code predictor: a batch draws its sampling noise in another order than one row.
    stage.pipeline.predictor.sample_on_device(1.0, 1)
    return stage, kv_cache


def _prefill(stage, kv_cache, info, length, table, slot):
    logits, _ = stage.prefill_forward(
        tokens=torch.ones(1, length, dtype=torch.int32),
        start_pos=torch.zeros(1, dtype=torch.int32),
        prompt_lens=torch.tensor([length]),
        page_table=table.reshape(1, -1),
        kv_cache=kv_cache,
        empty_slots=[slot],
        stage_inputs=[info],
    )
    return logits[0]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_requests_decoding_together_get_the_frames_they_get_alone(device):
    """Two requests in two slots, the second prefilled while the first is mid-decode: each
    gets, frame for frame, what it gets as the only request."""
    infos = [_payload(), dict(_payload(), text=[SECOND_TEXT])]
    per_request = MAX_MODEL_LEN // BLOCK_SIZE
    num_blocks = 1 + 2 * per_request
    tables = [torch.arange(1 + i * per_request, 1 + (i + 1) * per_request, dtype=torch.int32) for i in range(2)]

    stage, kv_cache = _stage(device, 1, num_blocks)
    lengths = []
    for info in infos:  # the encoders are eager, so before any trace exists
        reference = stage_inputs.clone_reference(info, device)
        text = info["text"][0]
        lengths.append(build_streaming_clone_prefill(text, reference, "English", stage.pipeline.tables)[0].shape[1])
    expected = []
    for info, length, table in zip(infos, lengths, tables):
        logits = _prefill(stage, kv_cache, info, length, table, 0)
        frames = []
        for step in range(STEPS):
            logits = stage.decode_forward(
                tokens=torch.tensor([[_top(logits)]], dtype=torch.int32),
                start_pos=torch.tensor([length + step], dtype=torch.int32),
                page_table=table.reshape(1, -1),
                kv_cache=kv_cache,
            )[0]
            frames.append(stage.pop_stage_outputs()[0].tolist())
        expected.append(frames)
        stage.release_request(0)
    stage.release_persistent_capture()
    del stage, kv_cache

    stage, kv_cache = _stage(device, 2, num_blocks)
    got, logits = [[], []], [None, None]

    def decode(live):
        out = stage.decode_forward(
            tokens=torch.tensor([[_top(logits[s])] if s in live else [0] for s in range(2)], dtype=torch.int32),
            start_pos=torch.tensor([lengths[s] + len(got[s]) if s in live else -1 for s in range(2)]),
            page_table=torch.stack([tables[s] if s in live else torch.zeros_like(tables[s]) for s in range(2)]),
            kv_cache=kv_cache,
        )
        frames = stage.pop_stage_outputs()
        for s in range(2):
            if s in live:
                got[s].append(frames[s].tolist())
                logits[s] = out[s]
            else:
                assert frames[s] is None, f"inactive slot {s} published a frame"

    logits[0] = _prefill(stage, kv_cache, infos[0], lengths[0], tables[0], 0)
    for _ in range(JOIN_AT):
        decode([0])
    logits[1] = _prefill(stage, kv_cache, infos[1], lengths[1], tables[1], 1)
    while any(len(frames) < STEPS for frames in got):
        decode([s for s in range(2) if len(got[s]) < STEPS])
    stage.release_persistent_capture()

    for s in range(2):
        for step, (left, right) in enumerate(zip(got[s], expected[s])):
            assert left == right, f"request {s}, frame {step}: {left} together, {right} alone"
    print(f"\n  2 requests, the second joining at frame {JOIN_AT}: {STEPS} frames each, identical to alone")


# 557 frames (44.6 s) at seed 0: past the 256 frames (20.48 s) a talker with its frame budget
# baked into the trace stopped at.
LONG_TEXT = (
    "The lighthouse keeper climbed the spiral stairs every evening, counting each step under his "
    "breath. At the top he trimmed the wick, polished the great lens, and watched the beam swing "
    "out across the water. Ships he would never meet passed safely in the dark, their captains "
    "trusting a light they could not see the source of. In winter the storms came hard, and salt "
    "crusted the windows until he scraped it away with a knife. He kept a logbook of every vessel, "
    "every gale, and every quiet night when nothing happened at all. Years later, when the light "
    "was automated, they found the logbooks stacked neatly in a cupboard, forty years of weather."
)
OLD_FRAME_CAP = 256
LONG_MODEL_LEN = 2048


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_a_long_utterance_runs_past_the_old_frame_cap_and_stops_at_end_of_speech(device):
    """Sampled as vLLM samples codebook 0, with the checkpoint's settings, the request's frames
    are bounded by the context length alone, and it ends on end-of-speech."""
    info = dict(_payload(), text=[LONG_TEXT])
    stage = Qwen3TTSTalkerForConditionalGeneration.initialize_vllm_model(
        SimpleNamespace(_name_or_path=""), device, 1, max_seq_len=LONG_MODEL_LEN
    )
    num_blocks = 1 + LONG_MODEL_LEN // BLOCK_SIZE
    kv_cache = stage.allocate_kv_cache((num_blocks, 8, BLOCK_SIZE, 128), torch.bfloat16, 28)
    table = torch.arange(1, num_blocks, dtype=torch.int32)
    pipeline = stage.pipeline
    reference = stage_inputs.clone_reference(info, device)
    length = build_streaming_clone_prefill(LONG_TEXT, reference, "English", pipeline.tables)[0].shape[1]
    pipeline.reseed(SEED)
    penalty = pipeline.generation.get("repetition_penalty", 1.0)

    logits = _prefill(stage, kv_cache, info, length, table, 0)
    seen = []
    while length + len(seen) < LONG_MODEL_LEN:
        first = pipeline._pick(logits, seen=seen, penalty=penalty, emitted=len(seen))
        if first == pipeline.eos:
            break
        logits = stage.decode_forward(
            tokens=torch.tensor([[first]], dtype=torch.int32),
            start_pos=torch.tensor([length + len(seen)], dtype=torch.int32),
            page_table=table.reshape(1, -1),
            kv_cache=kv_cache,
        )[0]
        stage.pop_stage_outputs()
        seen.append(first)
    stage.release_request(0)
    stage.release_persistent_capture()

    assert first == pipeline.eos, f"no end-of-speech within {len(seen)} frames, the context length"
    assert len(seen) > OLD_FRAME_CAP, f"stopped at {len(seen)} frames, inside the old cap of {OLD_FRAME_CAP}"
    print(f"\n  {len(LONG_TEXT)} chars: {len(seen)} frames, {len(seen) / 12.5:.1f} s, ended on end-of-speech")
