# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 2 (flow-matching acoustic transformer) reference tests.

Structural + wiring tests run always (random weights at real checkpoint shapes — 390M fits in
RAM, so the FULL block runs here, not a shortened one). Numerical tests need the checkpoint.

    pytest -svv models/experimental/voxtral_tts/tests/test_flow_pcc.py
"""

import os

import pytest
import torch

from models.experimental.voxtral_tts.reference import voxtral_flow_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    CFG_ALPHA,
    DEFAULT_CKPT,
    EMPTY_AUDIO_ID,
    END_AUDIO_ID,
    FM_DIM,
    FM_INPUT_DIM,
    FM_N_LAYERS,
    FM_SEMANTIC_OUT,
    N_ACOUSTIC_CODEBOOK,
    N_AUDIO_SPECIAL,
    N_DECODING_STEPS,
    SEMANTIC_CODEBOOK_SIZE,
    load_manifest,
    pcc,
    random_state_from_manifest,
)

PREFIX = "acoustic_transformer."
needs_ckpt = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")


@pytest.fixture(scope="module")
def w():
    """Full-size random Block 2 (~390M fp32) plus the recomputed inv_freq buffer."""
    s = random_state_from_manifest(PREFIX, seed=0)
    s["time_embedding.inv_freq"] = ref._inv_freq(FM_DIM, ref.FM_TIME_THETA)
    return s


def test_reference_uses_every_checkpoint_tensor():
    """Block 2 is small enough that the mapping should be exact in BOTH directions: no tensor
    invented, and none of the checkpoint's 33 left unused."""
    man = {k for k in load_manifest() if k.startswith(PREFIX)}
    expect = {
        PREFIX + n
        for n in ["input_projection.weight", "time_projection.weight", "llm_projection.weight",
                  "semantic_codebook_output.weight", "acoustic_codebook_output.weight", "norm.weight"]
    }
    for i in range(FM_N_LAYERS):
        expect |= {
            f"{PREFIX}layers.{i}.{k}"
            for k in ("attention.wq.weight", "attention.wk.weight", "attention.wv.weight",
                      "attention.wo.weight", "attention_norm.weight", "ffn_norm.weight",
                      "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight")
        }
    assert expect == man, f"missing {expect - man}; unused {man - expect}"
    assert len(man) == 33


def test_inv_freq_is_absent_from_checkpoint_and_recomputed():
    """time_embedding.inv_freq is registered persistent=True upstream but is NOT shipped, so a
    port that expects to load it will KeyError. It must be recomputed."""
    assert PREFIX + "time_embedding.inv_freq" not in load_manifest()
    f = ref._inv_freq(FM_DIM, ref.FM_TIME_THETA)
    assert f.shape == (FM_DIM // 2,)
    assert torch.isclose(f[0], torch.tensor(1.0))
    assert f[-1] < f[0], "inv_freq must decay"


def test_semantic_head_is_padded():
    """The semantic head is [8320, 3072] — 8192 codes + 2 specials padded to a 128 multiple.
    The pad rows are live logits and MUST be masked or the model can emit an invalid code."""
    assert tuple(load_manifest()[PREFIX + "semantic_codebook_output.weight"]["shape"]) == (FM_SEMANTIC_OUT, FM_DIM)
    assert FM_SEMANTIC_OUT > N_AUDIO_SPECIAL + SEMANTIC_CODEBOOK_SIZE


def test_velocity_shapes_and_three_token_sequence(w):
    h = torch.randn(2, FM_INPUT_DIM) * 0.1
    x_t = torch.randn(2, N_ACOUSTIC_CODEBOOK)
    t_emb = ref.time_embedding(torch.zeros(2, 1), w["time_embedding.inv_freq"])
    assert t_emb.shape == (2, FM_DIM)
    v = ref.predict_velocity(x_t, h, t_emb, w)
    assert v.shape == (2, N_ACOUSTIC_CODEBOOK), "velocity must be one float per acoustic codebook"
    assert torch.isfinite(v).all()


def test_semantic_mask_forbids_empty_and_pad(w):
    """[EMPTY_AUDIO] must never be selected; [END_AUDIO] must remain selectable (it is the stop
    signal); nothing in the pad region may be selected."""
    h = torch.randn(64, FM_INPUT_DIM)
    codes = ref.semantic_code(h, w)
    assert (codes != EMPTY_AUDIO_ID).all(), "[EMPTY_AUDIO] leaked into the argmax"
    assert (codes < N_AUDIO_SPECIAL + SEMANTIC_CODEBOOK_SIZE).all(), "a pad logit was selected"


def test_frame_is_37_codes_in_valid_ranges(w):
    h, x_0 = ref.make_synthetic_inputs(batch=3)
    frame = ref.reference_frame(h, w, x_0=x_0)
    assert frame.shape == (3, 1 + N_ACOUSTIC_CODEBOOK)
    ac = frame[:, 1:]
    assert (ac >= N_AUDIO_SPECIAL).all() and (ac < N_AUDIO_SPECIAL + ACOUSTIC_CODEBOOK_SIZE).all(), \
        "acoustic codes outside [offset, offset+levels)"


def test_euler_takes_the_configured_number_of_steps(w):
    h, x_0 = ref.make_synthetic_inputs(batch=1)
    sem = ref.semantic_code(h, w)
    _, trace = ref.decode_frame(sem, h, w, x_0=x_0, return_trace=True)
    assert trace.shape == (N_DECODING_STEPS, 1, N_ACOUSTIC_CODEBOOK), \
        f"expected {N_DECODING_STEPS} Euler steps (params.json omits n_decoding_steps; upstream defaults to 7)"


def test_end_audio_frames_are_not_decoded(w):
    """A frame whose semantic code is [END_AUDIO] must have its acoustic slots forced to
    [EMPTY_AUDIO], not whatever the ODE happened to produce."""
    h, x_0 = ref.make_synthetic_inputs(batch=2)
    sem = torch.tensor([[END_AUDIO_ID], [5]])
    codes = ref.decode_frame(sem, h, w, x_0=x_0)
    assert (codes[0] == EMPTY_AUDIO_ID + N_AUDIO_SPECIAL).all(), "END_AUDIO frame was decoded anyway"
    assert not (codes[1] == EMPTY_AUDIO_ID + N_AUDIO_SPECIAL).all(), "normal frame was masked"


def test_cfg_alpha_one_equals_conditional_only(w):
    """With alpha=1 the guidance term must vanish exactly: alpha*v + (1-alpha)*v_uncond -> v."""
    h, x_0 = ref.make_synthetic_inputs(batch=2)
    sem = ref.semantic_code(h, w)
    t_emb = ref.time_embedding(torch.zeros(2, 1), w["time_embedding.inv_freq"])
    v_cond = ref.predict_velocity(x_0, h, t_emb, w)
    _, trace = ref.decode_frame(sem, h, w, cfg_alpha=1.0, x_0=x_0, n_steps=1, return_trace=True)
    expected = x_0 + v_cond * 1.0  # single Euler step of dt=1
    assert pcc(trace[-1], expected) > 0.99999
    assert torch.allclose(trace[-1], expected, atol=1e-4)


def test_cfg_changes_the_trajectory(w):
    """Sanity that guidance is actually wired to the unconditional branch (zeroed h)."""
    h, x_0 = ref.make_synthetic_inputs(batch=2)
    sem = ref.semantic_code(h, w)
    a = ref.decode_frame(sem, h, w, cfg_alpha=1.0, x_0=x_0, return_trace=True)[1][-1]
    b = ref.decode_frame(sem, h, w, cfg_alpha=CFG_ALPHA, x_0=x_0, return_trace=True)[1][-1]
    assert not torch.allclose(a, b, atol=1e-6), "cfg_alpha had no effect"


def test_attention_is_bidirectional(w):
    """No causal mask here: perturbing the LAST sequence position (the LLM conditioning) must
    change the velocity read off position 0. If someone adds a causal mask, this fails."""
    h, x_0 = ref.make_synthetic_inputs(batch=1)
    t_emb = ref.time_embedding(torch.zeros(1, 1), w["time_embedding.inv_freq"])
    v1 = ref.predict_velocity(x_0, h, t_emb, w)
    v2 = ref.predict_velocity(x_0, h * 3.0, t_emb, w)
    assert not torch.allclose(v1, v2, atol=1e-6), "position 0 cannot see the conditioning position"


def test_fsq_quantize_endpoints():
    """clamp+rescale+round must map -1 -> 0 and +1 -> levels-1, and saturate beyond."""
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0] + [0.0] * (N_ACOUSTIC_CODEBOOK - 5)])
    q = ref._fsq_quantize(x)[0, :5]
    assert q.tolist() == [0, 0, (ACOUSTIC_CODEBOOK_SIZE - 1) // 2, ACOUSTIC_CODEBOOK_SIZE - 1,
                          ACOUSTIC_CODEBOOK_SIZE - 1]


@needs_ckpt
def test_real_weights_frame_is_valid():
    w = ref.load_flow_state()
    h, x_0 = ref.make_synthetic_inputs(batch=2)
    frame = ref.reference_frame(h, w, x_0=x_0)
    assert frame.shape == (2, 37)
    assert torch.isfinite(frame.float()).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
