# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN Qwen3-TTS code predictor against the CPU reference.

Block boundary: embeddings [1, 16, 2048] -> logits [1, 15, 2048], one per codebook 1..15.

Input is a frame the model produced itself: the talker run on a real prompt, its last
hidden state, codebook 0 read off `codec_head`, then codebooks 1..15 decoded greedily by
the reference. Random hidden states and random codes are out of distribution, and the
talker demonstrated what that costs.

Measured here, bf16 on Blackhole P150:

    teacher-forced blocks (projection, 5 layers, norm)   0.9965 to 0.99999
    teacher-forced logits                                0.9946
    per-step logits through a 15-step greedy decode      0.9959 to 0.9986

Greedy decode is scored per step rather than as a sequence. One flipped token changes the
input to every later step, so an end-to-end sequence comparison measures the cascade rather
than the port: the device matches 12 of 15 steps but only 8 of 15 codes. Each disagreement
is judged by how much the reference prefers its own pick over the device's, which is the
question that matters. Every one measured here is a near-tie, the widest gap being 0.10
against logits that span several units, and at the one step where the device picked the
reference's third choice its top three sat within 0.042 of each other.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen import transformers_compat  # noqa: F401  (registers rope "default")
from models.demos.audio.qwen3_tts.reference.qwen3_code_predictor_ref import (
    CodePredictorReference,
    build_input_embeddings,
)
from models.demos.audio.qwen3_tts.reference.qwen.talker import (
    Qwen3TTSRotaryEmbedding,
    Qwen3TTSTalkerCodePredictorConfig,
    apply_rotary_pos_emb,
)
from models.demos.audio.qwen3_tts.tests.reference_helpers import code_predictor_prompt
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor import (
    TtCodePredictor,
    plain_rotary_tables,
    preprocess_code_predictor_parameters,
)

BLOCK_PCC = 0.99
LOGITS_PCC = 0.99

# How far the reference may prefer its own pick over the device's before the disagreement
# stops being rounding. Widest observed is 0.10, against logits spanning several units.
MAX_PREFERENCE_GAP = 0.25

# 0.6B measured 0.9755 and gaps up to 0.82: its residual stream runs near 2665 in bf16 (README).
GREEDY_STEP_PCC = {"1b7": LOGITS_PCC, "0b6": 0.97}
PREFERENCE_GAP = {"1b7": MAX_PREFERENCE_GAP, "0b6": 1.0}

# Sampling-distribution distance, 0 to 1. Worst measured: 0.091 at 1.7B, 0.184 at 0.6B.
SAMPLER_DISTANCE = {"1b7": 0.15, "0b6": 0.25}


@pytest.fixture(scope="module")
def predictor_config():
    return weights.talker_config()["code_predictor_config"]


@pytest.fixture(scope="module")
def frame():
    """talker hidden state, codebook 0, codebooks 1..15, and the 16-position input."""
    return code_predictor_prompt()


@pytest.fixture(scope="module")
def reference_outputs(frame):
    _, _, _, embeddings = frame
    return CodePredictorReference().teacher_forced(embeddings, return_intermediates=True)


def _to_device(device, tensor):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _preference_gap(reference_logits, reference_pick, device_pick):
    """How much the reference prefers its own choice over the device's, in logits."""
    return float(reference_logits[reference_pick] - reference_logits[device_pick])


def _sampler_distance(reference_logits, device_logits):
    """Total variation distance between the two sampling distributions, 0 to 1."""
    temperature = weights.generation_config().get("subtalker_temperature", 1.0)
    reference = torch.softmax(reference_logits.float() / temperature, dim=-1)
    device = torch.softmax(device_logits.float() / temperature, dim=-1)
    return float(0.5 * (reference - device).abs().sum())


# ── host-side tables ────────────────────────────────────────────────────────


def test_plain_rotary_tables_match_the_reference(predictor_config):
    """The predictor rotates with plain RoPE, not the talker's multi-axis version."""
    cfg = dict(predictor_config)
    cfg.setdefault("pad_token_id", None)
    config = Qwen3TTSTalkerCodePredictorConfig(**cfg)
    length = 16

    torch.manual_seed(1)
    query = torch.randn(1, cfg["num_attention_heads"], length, cfg["head_dim"])
    key = torch.randn(1, cfg["num_key_value_heads"], length, cfg["head_dim"])
    positions = torch.arange(length).reshape(1, length)

    cos_ref, sin_ref = Qwen3TTSRotaryEmbedding(config)(query, positions)
    query_ref, key_ref = apply_rotary_pos_emb(query, key, cos_ref, sin_ref)

    cos, sin = plain_rotary_tables(cfg, length)

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    assert torch.equal(query * cos + rotate_half(query) * sin, query_ref)
    assert torch.equal(key * cos + rotate_half(key) * sin, key_ref)


# ── device ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_input_assembly_uses_the_right_embedding_tables(device, frame):
    """Position 1 reads the talker's codec embedding, later positions the predictor's own.

    Mixing the two produces a model that runs and is wrong, so the assembly is pinned
    against the reference's rather than checked by eye.
    """
    hidden, first_code, codes, _ = frame
    model = TtCodePredictor(device, preprocess_code_predictor_parameters(device))

    for count in (1, 2, 8, 15):
        prefix = ([first_code] + list(codes))[:count]
        assert torch.equal(model.build_embeddings(hidden, prefix), build_input_embeddings(hidden, prefix)), count


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_teacher_forced_blocks_match_the_reference(device, frame, reference_outputs):
    """Every block of one pass over all 16 positions."""
    _, _, _, embeddings = frame
    _, gold = reference_outputs
    model = TtCodePredictor(device, preprocess_code_predictor_parameters(device))

    cos, sin, mask = model.host_inputs(embeddings.shape[1])
    _, got = model(*(_to_device(device, t) for t in (embeddings, cos, sin, mask)), return_intermediates=True)

    failures = []
    for name, want in gold.items():
        measured = ttnn.to_torch(got[name]).float().reshape(want.shape)
        passed, message = comp_pcc(want, measured, pcc=BLOCK_PCC)
        print(f"  [{name:11s}] {message}")
        if not passed:
            failures.append(f"{name}: {message}")

    assert not failures, "blocks below PCC gate: " + "; ".join(failures)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_teacher_forced_logits_match_the_reference(device, frame, reference_outputs):
    """All 15 codebook heads, and the tokens they pick."""
    _, _, _, embeddings = frame
    gold, _ = reference_outputs
    model = TtCodePredictor(device, preprocess_code_predictor_parameters(device))

    cos, sin, mask = model.host_inputs(embeddings.shape[1])
    logits = model.teacher_forced_logits(*(_to_device(device, t) for t in (embeddings, cos, sin, mask)))
    measured = ttnn.to_torch(logits).float().reshape(gold.shape)

    passed, message = comp_pcc(gold, measured, pcc=LOGITS_PCC)
    print(f"logits {tuple(measured.shape)}  {message}")

    size = weights.model_size()
    reference_pick, device_pick = gold.argmax(-1)[0], measured.argmax(-1)[0]
    print(f"argmax agreement {int((reference_pick == device_pick).sum())}/{gold.shape[1]}")

    wide = []
    for index in torch.nonzero(reference_pick != device_pick).flatten().tolist():
        gap = _preference_gap(gold[0, index], reference_pick[index], device_pick[index])
        print(f"  codebook {index + 1}: reference prefers its pick by {gap:.4f}")
        if gap > PREFERENCE_GAP[size]:
            wide.append(f"codebook {index + 1} gap {gap:.4f}")

    distance = max(_sampler_distance(gold[0, index], measured[0, index]) for index in range(gold.shape[1]))
    print(f"worst sampling distance {distance:.4f}")

    assert not wide, "disagreements the reference feels strongly about: " + "; ".join(wide)
    assert passed, f"logits below PCC {LOGITS_PCC}: {message}"
    assert distance < SAMPLER_DISTANCE[size], f"sampling distance {distance:.4f}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_greedy_decode_tracks_the_reference_step_by_step(device, frame):
    """The 15-step loop, scored per step rather than as a sequence.

    At each step the device's own prefix goes to both models, so the cascade from an
    earlier flip is removed and what is measured is the step itself.
    """
    hidden, first_code, _, _ = frame
    reference = CodePredictorReference()
    model = TtCodePredictor(device, preprocess_code_predictor_parameters(device))

    size = weights.model_size()
    prefix = [first_code]
    exact, wide, worst_pcc, worst_distance = 0, [], 1.0, 0.0
    for step in range(15):
        embeddings = build_input_embeddings(hidden, prefix)
        length = embeddings.shape[1]
        cos, sin, mask = model.host_inputs(length)
        hidden_states = model(*(_to_device(device, t) for t in (embeddings, cos, sin, mask)))
        row = ttnn.slice(hidden_states, [0, length - 1, 0], [1, length, hidden_states.shape[2]])
        device_logits = ttnn.to_torch(ttnn.linear(row, model.p["lm_head"][step])).float().reshape(-1)

        projected = reference.model.small_to_mtp_projection(embeddings)
        reference_hidden = reference.model.model(inputs_embeds=projected).last_hidden_state[:, -1]
        reference_logits = reference.model.lm_head[step](reference_hidden).reshape(-1)

        device_pick, reference_pick = int(device_logits.argmax()), int(reference_logits.argmax())
        _, message = comp_pcc(reference_logits, device_logits, pcc=0.0)
        worst_pcc = min(worst_pcc, float(str(message)))
        worst_distance = max(worst_distance, _sampler_distance(reference_logits.detach(), device_logits))
        if device_pick == reference_pick:
            exact += 1
        else:
            gap = _preference_gap(reference_logits, reference_pick, device_pick)
            print(f"  step {step:2d}: device {device_pick}, reference {reference_pick}, preferred by {gap:.4f}")
            if gap > PREFERENCE_GAP[size]:
                wide.append(f"step {step} gap {gap:.4f}")
        prefix.append(device_pick)

    print(
        f"steps matching exactly {exact}/15 | worst per-step logits PCC {worst_pcc:.5f} "
        f"| worst sampling distance {worst_distance:.4f}"
    )
    assert not wide, "steps the reference feels strongly about: " + "; ".join(wide)
    assert worst_pcc >= GREEDY_STEP_PCC[size], f"per-step logits PCC fell to {worst_pcc:.5f}"
    assert worst_distance < SAMPLER_DISTANCE[size], f"sampling distance {worst_distance:.4f}"
