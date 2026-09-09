# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model PCC gate for Qwen3-TTS: the whole device pipeline, chained, against
the fp32 torch reference in ``reference/functional.py``.

``test_qwen3_tts_pcc.py`` is the per-block gate — one block at a time, random
weights, each block handed a clean reference input. This is the other half:
production weights, the demo's own reference clip and prompt, on the device the
model ships on, with every stage feeding the next so precision loss compounds
the way it does in a real render.

Method
------
The same one ``models/tt_transformers/tests/test_model.py`` uses:

* **Chained, not isolated.** The torch reference runs as an independent model —
  its own prefill, its own decode, its own CodePredictor — and never sees a
  device tensor. Each stage's PCC therefore carries everything upstream of it,
  which is the point: per-block numbers can all pass while the chain drifts.
* **Teacher forcing at every sampling point.** Both models advance on the
  *reference's* greedy token. Without it one disagreeing argmax sends the two
  models down different utterances and every later PCC measures divergence
  rather than arithmetic. The device's own greedy picks are still recorded and
  reported as a token-agreement rate, which is the number that actually says
  whether the audio would differ.
* **Measure everything, then assert.** Every stage lands in one table that is
  asserted at the end, so one run names every stage that regressed instead of
  stopping at the first.

Scope
-----
What the demo runs on device, in order::

    speaker encoder (ECAPA, device mel)
      -> Talker prefill, 28 layers          -> codec_head, code 0
      -> CodePredictor prefill + 14 decodes -> codes 1..15
      -> next-frame input embedding         -> Talker decode -> ...

The RVQ speech-tokenizer **encoder** and the audio **decoder** are deliberately
out of scope. ``demo_full_ttnn_tts.py`` runs both in torch
(``speech_tokenizer_encoder_forward_mimi`` for the reference clip,
``speech_tokenizer_decoder_forward`` for the waveform), so on the shipped path
there is no device implementation to compare a reference against.
``tt/speech_tokenizer.py`` and ``tt/ttnn_speech_decoder.py`` exist but the demo
does not call them.

Run (N300, pinned to one card so the rest of the machine stays usable)::

    export MESH_DEVICE=N300
    pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_full_model_pcc.py

``MESH_DEVICE=N150`` for one chip, ``T3K`` for eight; unset it for a plain
non-mesh device.

Knobs
-----
``QWEN3_TTS_PCC_FRAMES``
    Decode frames to walk (default 24). Each frame costs one full 28-layer fp32
    reference recompute of the whole prefix — there is no reference KV cache — so
    the reference side is O(frames x prefix). 24 is the depth at which the
    per-frame trajectory was shown to be a stationary band rather than a drift
    (see below), which is what makes ``FLOOR_PCC`` meaningful.
``QWEN3_TTS_HF_ID``, ``QWEN3_TTS_PROFILE_TARGET_TEXT``
    Checkpoint and prompt, shared with the profile tests.

Prefill length
--------------
Three cases, all run by default:

``demo``
    The shipped prompt: ``jim_reference.wav`` and the default target text, 61 ICL
    tokens into the 64-token bucket. What the demo actually renders.
``max``
    512 — the longest prefill BOTH SKUs run.
``max1024``
    1024, the top of ``SUPPORTED_PREFILL_LENS``. Runs on both SKUs.

Reaching 1024 needs a synthetic prompt, and it is worth being explicit about why.
The ICL length is set by the **reference clip's duration**, not by the target
text: ``create_icl_embedding_ttnn`` builds ``icl_input_embed`` at exactly
``codec_lens = 1 + ref_frames`` and spills any surplus text into
``trailing_text_hidden``. A 1024-token prefill therefore needs ~1014 code frames,
i.e. about 80 s of reference audio at 12.5 fps; the repo's clip is 4 s (51
frames). The long cases tile the real code sequence up to the length they need,
so every embedding row is a genuine codebook entry and both models still receive
byte-identical input — but the sequence does not correspond to real speech. That
is fine for a PCC gate, which compares arithmetic on identical inputs, and it is
NOT a claim about audio quality at that length.

Why 1024 needed a production fix
--------------------------------
1024 is listed in ``SUPPORTED_PREFILL_LENS`` and warmed by
``warmup_all_buckets``, but before this test it **did not run on either SKU**.
Swept bucket by bucket, 128/192/256/384/512 all worked and 1024 failed on both
with the MLP down-proj's static circular buffers clashing with L1. Getting there
took three changes, each found by fixing one op and re-running to see the next:

1. ``mlp.py`` runs prefill longer than ``QWEN3_TTS_MLP_MM_CAP`` (512) in **row
   chunks**, staging the long tensors through DRAM.
2. ``attention.py`` **row-chunks the fused QKV projection** for the same lengths
   and stages it through DRAM. At 1024 on TP=1 that output alone is
   1024 x 4096 bf16 = 8 MB, and the M=1024 matmul's circular buffers clash with
   the layer's resident 1024-row L1 tensors besides.
3. The MLP's chunk SIZE is SKU-aware: 512 on TP=2, **256 on TP=1**. With one chip
   the down-proj is K=6144 per chip against 3072 on TP=2, and even M=512 clashes
   once the caller's 1024-row tensors are live (region ends 721632, highest L1
   buffer 678848).

Row chunking is exact, not an approximation. Gate, up, silu-mul, down, the
row-parallel all_reduce and both projections are row-wise, so no row ever needs
another row — it is the same arithmetic in smaller launches. Confirmation: on
N300 the 1024 numbers are bit-identical before and after the QKV was chunked, so
M=512 and M=1024 accumulate K the same way there.

It has to be a loop, not a reshape. Folding rows into the batch dim — which
``mlp.py`` previously did for seq >= 1024 — makes ``num_blocks_total`` exceed
``num_cores`` and the 1D-mcast matmul rejects it outright, so that path had never
worked either.

Note the separation of GATE from CHUNK. The gate stays at 512 on every SKU, so
every bucket that already worked keeps its exact shapes and program configs. This
is verified, not assumed: all six pre-existing values (demo and 512 on both SKUs)
are identical to the last decimal before and after all three changes.

What was NOT possible: chunked *attention*
------------------------------------------
Chunking the prefill in the attention sense — feed 128 queries at a time and
attend over the whole history — would have solved this more generally, and does
not work on this KV cache. The branch that would carry it
(``prefill_attn_mask`` in ``attention.py``) writes K/V with
``ttnn.update_cache``, which is a single-position decode op. Handed a multi-row
input it runs without error and writes **wrong data** — verified in isolation at
cache depths 256/512/1088 and offsets 0/128, every combination incorrect. That
branch is dead code today (nothing passes ``prefill_attn_mask``; server.py's
traces take the ``is_causal`` path), so the bug is latent rather than live, but
it means chunked attention needs the paged machinery ``tt_transformers`` uses —
``paged_fill_cache`` plus ``ttnn.transformer.chunked_scaled_dot_product_attention``
with a ``chunk_page_table`` — not an offset argument.

Measured
--------
Qwen3-TTS-12Hz-1.7B-Base. ``EXPECTED_PCC`` takes the lower of the two SKUs.
``demo`` case (61 ICL tokens)::

    stage                     N300 (TP=2)   N150 (TP=1)
    speaker_encoder              0.999625     0.999625
    talker_prefill_hidden        0.991840     0.990499
    codec_head_prefill           0.997788     0.998648
    talker_decode_hidden         0.992981     0.991320
    codec_head_decode            0.998061     0.998463
    cp_logits                    0.991980     0.991985
    frame_embed_device           1.000000     1.000000

``max`` (512) and ``max1024`` (1024), both SKUs::

    stage                   512: N300  N150     1024: N300  N150
    speaker_encoder             0.999625  0.999625    0.999625  0.999625
    talker_prefill_hidden       0.981422  0.980476    0.980722  0.978121
    codec_head_prefill          0.993550  0.993934    0.988377  0.988348
    talker_decode_hidden        0.969785  0.967749    0.983402  0.983054
    codec_head_decode           0.992530  0.988894    0.993498  0.993024
    cp_logits                   0.973067  0.968694    0.950089  0.947437
    frame_embed_device          1.000000  1.000000    1.000000  1.000000

``talker_decode_hidden`` is BETTER at 1024 (0.983) than at 512 (0.968): a decode
step attends over a longer, more averaged history, so it is not monotone in
sequence length the way prefill is.

Worst frame over a 24-frame walk — what ``FLOOR_PCC`` is set from::

    stage                    demo N300/N150     512 N300/N150      1024 N300/N150
    talker_decode_hidden     0.991214 0.990976   0.969735 0.961618   0.970165 0.972552
    codec_head_decode        0.997184 0.996345   0.988791 0.983886   0.988170 0.986203
    cp_logits                0.980584 0.980648   0.935413 0.927480   0.932574 0.932510
    frame_embed_device       1.000000 1.000000   1.000000 1.000000   1.000000 1.000000

Eight times the sequence costs roughly a decimal place: the Talker prefill hidden
goes 0.991 -> 0.981, and cp_logits' worst frame 0.981 -> 0.927. The CP is the most
exposed stage because its whole input is one hidden row, so it inherits the
Talker's error undiluted; the codec_head stays above 0.98 throughout because a
projection does not amplify what it is handed.

Two things are worth reading off that table. The Talker hidden state sits at
~0.991 while the codec_head logits it feeds sit at ~0.998: 28 layers of bf16
residual accumulation is where the error is, and the head is a projection that
does not amplify it. And every code-0 token the two models chose agreed exactly
(5/5 across prefill and four frames), which is the independent evidence that
~0.991 is arithmetic noise rather than a mis-specified reference — a wrong RoPE
convention or weight layout does not land tokens on the nose.

The gated value for each per-frame stage is **frame 0**, not the worst frame.
A worst-of-N statistic can only fall as N grows, so a threshold calibrated at one
frame count fails spuriously at a higher one — measured on N300, cp_logits gives
0.9920 worst-of-1, 0.9897 worst-of-2, 0.9839 worst-of-4, 0.9806 worst-of-6. Frame
0 is fixed whatever ``QWEN3_TTS_PCC_FRAMES`` says, and was verified identical to
six decimals at 2, 4 and 6 frames on both SKUs.

Every walked frame is additionally held to a floor (``FLOOR_PCC``), so frames
past 0 are enforced too — just not against frame 0's tighter number.

The frame-to-frame variation is a **stationary band, not cumulative drift**.
Walking 24 frames on both SKUs, no stage trends: cp_logits on N300 runs 0.9920
0.9897 0.9917 0.9839 0.9806 0.9854 ... 0.9919 0.9922 0.9870 0.9875 0.9923,
bottoming at frame 4 and recovering, and N150 bottoms at frame 10 instead. The
two SKUs reach almost exactly the same floor from different frames (0.980584 and
0.980648), which is what a floor looks like rather than a trend.

This corrects a reading taken from six frames, where cp_logits' first five
values do look like a downward slope and were written up as one — the CP
re-reading an ever-worse Talker hidden state. Twenty-four frames show that
mechanism is not operating: the KV cache does not compound error here, because
each CP frame resets its own cache and the Talker's residual stream stays in a
band. Do not re-derive a drift story from a short walk.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_tts.tests.qwen3_tts_profile_demo_common import (
    DEFAULT_REF_AUDIO,
    close_profile_device,
    demo_ref_text,
    demo_target_text,
    hf_id,
    open_profile_device,
)
from models.demos.qwen3_tts.tt.mesh_utils import to_torch as mesh_to_torch

# -----------------------------------------------------------------------------
# Expected PCC per stage, per prefill case. Threshold is the entry minus TOLERANCE, the
# convention test_qwen3_tts_pcc.py uses. See the module docstring for the
# measurements these came from and for why the per-frame stages gate frame 0.
#
# These are CHAINED numbers: each stage's value carries the error of every stage
# before it, so they are NOT comparable to the per-block figures in
# test_qwen3_tts_pcc.py (which hands each block a clean reference input).
# -----------------------------------------------------------------------------
EXPECTED_PCC: Dict[str, Dict[str, float]] = {
    "demo": {
        "speaker_encoder": 0.9996,
        "talker_prefill_hidden": 0.9905,
        "codec_head_prefill": 0.9977,
        "talker_decode_hidden": 0.9913,
        "codec_head_decode": 0.9980,
        "cp_logits": 0.9919,
        "frame_embed_device": 1.0000,
    },
    # Calibrated separately: a 1024-token prefill accumulates over 16x the
    # sequence, and its decode steps attend over a 1024-deep KV cache, so the
    # demo case's numbers do not transfer.
    "max": {
        "speaker_encoder": 0.9996,
        "talker_prefill_hidden": 0.9804,
        "codec_head_prefill": 0.9935,
        "talker_decode_hidden": 0.9677,
        "codec_head_decode": 0.9888,
        "cp_logits": 0.9686,
        "frame_embed_device": 1.0000,
    },
    "max1024": {
        "speaker_encoder": 0.9996,
        "talker_prefill_hidden": 0.9781,
        "codec_head_prefill": 0.9883,
        "talker_decode_hidden": 0.9830,
        "codec_head_decode": 0.9930,
        "cp_logits": 0.9474,
        "frame_embed_device": 1.0000,
    },
}
TOLERANCE = 0.005
# frame_embed_device is not a model stage — it is 16 embedding lookups accumulated
# in float32, which server.py documents as bit-exact with the host sum. Measured
# 1.000000 on both SKUs across every frame, so it gets a gate that would actually
# notice a change rather than one 0.005 wide.
TOLERANCE_OVERRIDE: Dict[str, float] = {"frame_embed_device": 1e-6}

# Floor for EVERY walked frame, not just the gated frame 0. These are the minima
# observed over a 24-frame walk on both SKUs (see the module docstring), so unlike
# a worst-of-N gate calibrated at one frame count they do not tighten as
# QWEN3_TTS_PCC_FRAMES grows — the trajectory is a stationary band and 24 frames
# is enough to have found its bottom twice, independently, on N300 and N150.
FLOOR_PCC: Dict[str, Dict[str, float]] = {
    "demo": {
        "talker_decode_hidden": 0.9909,
        "codec_head_decode": 0.9963,
        "cp_logits": 0.9806,
    },
    "max": {
        "talker_decode_hidden": 0.9616,
        "codec_head_decode": 0.9838,
        "cp_logits": 0.9274,
    },
    "max1024": {
        "talker_decode_hidden": 0.9701,
        "codec_head_decode": 0.9862,
        "cp_logits": 0.9325,
    },
}


def _threshold(case: str, name: str) -> float:
    return EXPECTED_PCC[case][name] - TOLERANCE_OVERRIDE.get(name, TOLERANCE)


def _floor(case: str, name: str) -> float:
    return FLOOR_PCC[case][name] - TOLERANCE


# Greedy token agreement between device and reference is REPORTED, not gated.
# It is not a property bf16 can hold: the codebooks are 2048-way and unordered, a
# 0.997-PCC logit vector reorders near-ties, and the demo does not decode greedily
# anyway (temperature 0.9, top_k 50). Measured 91 % over 2 frames and 77-83 % over
# 4, falling with frame count as the chain drifts — a gate on it would be a gate on
# how far the test happens to walk. The audio-level question ("does it still sound
# right and say the words") has its own gate in test_qwen3_tts_voice_quality.py,
# which scores SIM and WER on a rendered clip. This number is here to make a gross
# breakage obvious at a glance.

# Prefill length per case. "max" is the longest BOTH SKUs run; "max1024" is the
# top of SUPPORTED_PREFILL_LENS and is TP>1 only. QWEN3_TTS_PCC_MAX_PREFILL
# overrides the latter.
_CASE_PREFILL = {"max": 512, "max1024": 1024}
# Above this the codec_head is applied to the last real row only, instead of to the
# whole sequence as the shipped prefill does. seq x 3072 in L1 is 6 MB at 1024, and
# only the last row is ever read. The demo case stays on the full-sequence shape.
_HEAD_FULL_SEQ_LIMIT = 512

# server.generate_codes_ttnn's CP cache width. The CP sequence is 16 positions
# (1 talker hidden + 15 code embeddings), padded to a tile.
_MAX_CP_SEQ = 32


# -----------------------------------------------------------------------------
# Reference chains
# -----------------------------------------------------------------------------
def _ref_talker(embeds: torch.Tensor, talker_w: Dict[str, torch.Tensor], ref_cfg) -> torch.Tensor:
    """Reference Talker over ``[1, seq, hidden]`` input embeddings. Returns the
    post-final-norm hidden state, fp32.

    ``reference.talker_forward`` is not usable here: it starts from token ids and
    its own codec embedding, and it applies MROPE. The shipped Talker is fed
    pre-built ICL embeddings and driven by plain 1-D RoPE (``generate_codes_ttnn``
    builds its tables with ``compute_rope_frequencies``), so the layer chain is
    assembled directly instead.

    The device's Q/K projections and head norms are row-permuted at load time
    (``attention._permute_rope_head_dim_rows``) precisely so that TTNN's
    interleaved rotary kernel computes the same function as the half-split
    ``rotate_half`` convention this reference uses. That is why the raw
    checkpoint weights are correct here and no rearrangement is needed.
    """
    from models.demos.qwen3_tts.reference.functional import compute_rope_frequencies, decoder_layer, rms_norm

    seq = int(embeds.shape[1])
    cos, sin = compute_rope_frequencies(ref_cfg.head_dim, seq, ref_cfg.rope_theta)
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    x = embeds.float()
    for i in range(ref_cfg.num_hidden_layers):
        prefix = f"layers.{i}."
        # Cast per layer: a whole-model fp32 copy of the 1.7B talker is ~7 GB.
        lw = {k[len(prefix) :]: v.float() for k, v in talker_w.items() if k.startswith(prefix)}
        x = decoder_layer(x, lw, cos, sin, ref_cfg, attention_mask=mask, use_mrope=False)
    return rms_norm(x, talker_w["norm.weight"].float(), ref_cfg.rms_norm_eps)


def _ref_cp_hidden(cp_embeds: torch.Tensor, cp_w: Dict[str, torch.Tensor], cp_ref_cfg) -> torch.Tensor:
    """Reference CodePredictor over ``[1, n_pos, cp_hidden]``, already projected.

    The device runs a 2-position prefill then single-token decodes against a KV
    cache; a full causal forward over ``0..n_pos-1`` is the same function, so the
    reference recomputes the prefix and the caller takes the last row.
    """
    from models.demos.qwen3_tts.reference.functional import code_predictor_forward

    return code_predictor_forward(cp_embeds.float(), cp_w, cp_ref_cfg)


def _ref_codes_for_icl_len(target_len: int, ref_codes: torch.Tensor, probe_len: int) -> torch.Tensor:
    """Tile ``ref_codes`` so that the ICL sequence comes out ``target_len`` long.

    ``create_icl_embedding_ttnn`` lays out ``role || prefix || icl_input_embed``
    where only the last part varies in length, and it is always
    ``1 + ref_frames`` rows (surplus target text goes to ``trailing_text_hidden``
    instead of lengthening the prefill). So the ICL length is affine in the frame
    count, and ``probe_len`` — the length a build with the real clip produced —
    pins the constant without hardcoding the prompt's fixed-size parts, which
    would rot the moment the prompt template changes.
    """
    frames = int(ref_codes.shape[0])
    const = probe_len - frames
    need = target_len - const
    if need <= 0:
        raise ValueError(f"target ICL length {target_len} is below the prompt's fixed {const} tokens")
    reps = -(-need // frames)
    return ref_codes.repeat(reps, 1)[:need]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    dev, mesh_shape = open_profile_device()
    yield dev
    close_profile_device(dev, mesh_shape)


@pytest.fixture(scope="module")
def model_and_weights(device):
    from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
    from models.demos.qwen3_tts.tt.server import load_weights

    main_weights, _ = load_weights(hf_id())
    model = Qwen3TTS(device=device, state_dict=main_weights, talker_config=talker_config_for_hf_id(hf_id()))
    ttnn.synchronize_device(device)
    return model, main_weights


# -----------------------------------------------------------------------------
# Device helpers — the shapes and buffers generate_codes_ttnn uses
# -----------------------------------------------------------------------------
def _upload(t: torch.Tensor, device, dtype=ttnn.bfloat16, memcfg=ttnn.L1_MEMORY_CONFIG, mapper=None):
    return ttnn.from_torch(
        t, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memcfg, mesh_mapper=mapper
    )


def _cp_buffers(device, model, cp_num_heads: int) -> Dict:
    """CP KV caches, zero hosts, prefill mask/rope and decode rope/mask tables."""
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_rope_tensors, get_transformation_mat
    from models.demos.qwen3_tts.tt.server import allocate_kv_cache, build_cp_decode_trace_h2d_constants

    cp_cfg = model.code_predictor_config
    caches = allocate_kv_cache(
        device=device,
        num_layers=cp_cfg.num_hidden_layers,
        batch_size=1,
        num_kv_heads=cp_cfg.num_key_value_heads,
        max_seq_len=_MAX_CP_SEQ,
        head_dim=cp_cfg.head_dim,
    )
    zero_hosts = [
        (
            ttnn.from_torch(
                torch.zeros(tuple(k.shape), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                torch.zeros(tuple(v.shape), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
        )
        for k, v in caches
    ]

    # Prefill mask: position 0 sees only itself, position 1 sees both.
    mask = torch.full((1, cp_num_heads, 2, _MAX_CP_SEQ), float("-inf"))
    mask[0, :, 0, 0] = 0.0
    mask[0, :, 1, 0:2] = 0.0

    pf_cos, pf_sin = get_rope_tensors(device, cp_cfg.head_dim, 2, torch.arange(2), cp_cfg.rope_theta)
    cos_tab, sin_tab = compute_rope_frequencies(cp_cfg.head_dim, _MAX_CP_SEQ + 5, cp_cfg.rope_theta)
    dc_cos, dc_sin, dc_mask = build_cp_decode_trace_h2d_constants(
        cos_tab, sin_tab, cp_num_heads, _MAX_CP_SEQ, model.code_predictor_config.num_code_groups - 2
    )
    return {
        "caches": caches,
        "zero_hosts": zero_hosts,
        "trans_mat": get_transformation_mat(cp_cfg.head_dim, device),
        "pf_cos": pf_cos,
        "pf_sin": pf_sin,
        "pf_mask": _upload(mask, device, dtype=ttnn.float32),
        "dc_cos": [ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in dc_cos],
        "dc_sin": [ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in dc_sin],
        "dc_mask": [ttnn.to_device(h, device, memory_config=ttnn.L1_MEMORY_CONFIG) for h in dc_mask],
    }


def _zero_cp_caches(cp: Dict) -> None:
    for (zk, zv), (k, v) in zip(cp["zero_hosts"], cp["caches"]):
        ttnn.copy_host_to_device_tensor(zk, k)
        ttnn.copy_host_to_device_tensor(zv, v)


def _device_frame_embed(device, model, code_row: List[int], trail_row: torch.Tensor, tables) -> torch.Tensor:
    """The fused CP trace's on-device next-input embedding (step 6 of
    ``capture_fused_cp_trace._body``): 16 ``ttnn.embedding`` lookups accumulated in
    float32, plus the trailing-text row, typecast to bfloat16.

    Reproduced here op-for-op because it is on by default
    (``QWEN3_TTS_CP_FUSED=1``) and nothing else in the test suite covers it. The
    claim it rests on is that it is bit-exact with the host ``F.embedding`` + fp32
    sum, so the expected PCC is 1.0 and any drift is a real regression.
    """
    from models.demos.qwen3_tts.tt.server import _replicate_mapper, append_device_embedding

    codec_tt, cp_tts = tables
    talker_h = model.talker_config.hidden_size
    mapper = _replicate_mapper(device)

    def _tok(v: int):
        return ttnn.from_torch(
            torch.tensor([[[v]]], dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    toks = [_tok(c) for c in code_row]
    acc = append_device_embedding(toks[0], codec_tt, talker_h, dtype=ttnn.float32)
    for i in range(1, len(code_row)):
        row = append_device_embedding(toks[i], cp_tts[i - 1], talker_h, dtype=ttnn.float32)
        nxt = ttnn.add(acc, row)
        ttnn.deallocate(acc)
        ttnn.deallocate(row)
        acc = nxt
    trail_tt = _upload(trail_row.reshape(1, 1, 1, -1).float(), device, dtype=ttnn.float32, mapper=mapper)
    with_trail = ttnn.add(acc, trail_tt)
    out = ttnn.typecast(with_trail, ttnn.bfloat16)
    host = mesh_to_torch(out).float().reshape(1, 1, -1)

    for t in (acc, trail_tt, with_trail, out, *toks):
        ttnn.deallocate(t)
    return host


# -----------------------------------------------------------------------------
# The test
# -----------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("prefill_case", ["demo", "max", "max1024"])
def test_full_model_pcc(device, model_and_weights, prefill_case):
    from models.demos.qwen3_tts.reference.functional import Qwen3TTSCodePredictorConfig as RefCPConfig
    from models.demos.qwen3_tts.reference.functional import Qwen3TTSConfig as RefTalkerConfig
    from models.demos.qwen3_tts.reference.functional import SpeakerEncoderConfig as RefSEConfig
    from models.demos.qwen3_tts.reference.functional import (
        compute_mel_spectrogram_qwen,
        extract_code_predictor_weights,
        extract_speaker_encoder_weights,
        extract_talker_weights,
        speaker_encoder_forward,
    )
    from models.demos.qwen3_tts.tests.qwen3_tts_profile_demo_common import allocate_talker_kv, pad_inputs_to_demo_bucket
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size, is_mesh_device
    from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_rope_tensors, get_transformation_mat
    from models.demos.qwen3_tts.tt.server import (
        SUPPORTED_PREFILL_LENS,
        TTSConfig,
        build_talker_decode_trace_h2d_constants,
        create_icl_embedding_ttnn,
        encode_reference_audio,
        upload_embed_tables,
    )

    model, weights = model_and_weights
    talker_cfg = model.talker_config
    cp_cfg = model.code_predictor_config
    ref_talker_cfg = RefTalkerConfig(hidden_size=talker_cfg.hidden_size, intermediate_size=talker_cfg.intermediate_size)
    ref_cp_cfg = RefCPConfig()
    talker_h = talker_cfg.hidden_size
    n_codes = cp_cfg.num_code_groups
    frames = int(os.environ.get("QWEN3_TTS_PCC_FRAMES", "24"))

    tp = get_tp_size(device) if is_mesh_device(device) else 1
    talker_num_heads = talker_cfg.num_attention_heads // tp
    cp_num_heads = cp_cfg.num_attention_heads // tp

    # Collected first, asserted last, so one run names every regression.
    results: List[Tuple[str, float]] = []

    def record(name: str, golden: torch.Tensor, calculated: torch.Tensor) -> float:
        _, pcc = comp_pcc(golden, calculated, _threshold(prefill_case, name))
        results.append((name, float(pcc)))
        logger.info(f"{name:24s} PCC {float(pcc):.6f}")
        return float(pcc)

    trajectories: Dict[str, List[float]] = {}

    def gate_first_frame(name: str, values: List[float]) -> None:
        """Gate frame 0; keep the rest as a printed trajectory.

        The gated value must not depend on ``QWEN3_TTS_PCC_FRAMES``. A
        worst-over-frames statistic does: it can only fall as more frames are
        walked, so a threshold calibrated at one frame count spuriously fails at a
        higher one (measured: cp_logits 0.9897 worst-of-2 against 0.9839
        worst-of-4). Frame 0 is fixed whatever the frame count, and the drift the
        later frames show is what the printed trajectory is for.
        """
        trajectories[name] = values
        results.append((name, values[0]))
        logger.info(f"{name:24s} PCC {values[0]:.6f}  (frame 0; {len(values)} frame(s) walked)")

    def gate_min(name: str, values: List[float]) -> None:
        """Gate the worst frame. Only for stages whose claim is exactness, where
        there is no drift for the frame count to interact with."""
        trajectories[name] = values
        results.append((name, min(values)))
        logger.info(f"{name:24s} PCC {min(values):.6f}  (worst of {len(values)})")

    # =========================================================================
    # Stage 1 — speaker encoder: waveform -> 2048-d embedding.
    # The device does its own mel on device; the reference gets the host mel. The
    # comparison is therefore of the whole stage, mel included, which is what the
    # demo depends on.
    # =========================================================================
    ref_codes, audio = encode_reference_audio(str(DEFAULT_REF_AUDIO))
    spk_dev = model.extract_speaker_embedding(audio)
    spk_ref = speaker_encoder_forward(
        compute_mel_spectrogram_qwen(audio),
        {k: v.float() for k, v in extract_speaker_encoder_weights(weights).items()},
        RefSEConfig(output_dim=talker_h),
    )
    record("speaker_encoder", spk_ref.reshape(1, -1), spk_dev.reshape(1, -1))

    # =========================================================================
    # ICL input construction. Both models are fed the SAME embeddings — they are
    # the demo's real ICL tensor read back off the device — so nothing below is
    # measuring the prompt builder.
    # =========================================================================
    from transformers import AutoTokenizer

    config = TTSConfig()
    config.hidden_size = talker_h
    config.max_new_tokens = max(frames + 8, 32)
    tokenizer = AutoTokenizer.from_pretrained(hf_id(), trust_remote_code=True)

    def build_icl(codes):
        return create_icl_embedding_ttnn(
            target_text=demo_target_text(),
            ref_text=demo_ref_text(),
            ref_codes=codes,
            speaker_embedding=spk_dev,
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
            main_weights=weights,
        )

    inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, code_pred_embeds = build_icl(ref_codes)
    real_seq_len = int(inputs_embeds_tt.shape[2])

    if prefill_case != "demo":
        # Rebuild at a longer prefill. The first build is the probe that pins the
        # prompt's fixed-size part; see _ref_codes_for_icl_len.
        target = _CASE_PREFILL[prefill_case]
        if prefill_case == "max1024":
            target = int(os.environ.get("QWEN3_TTS_PCC_MAX_PREFILL", str(target)))
        assert target in SUPPORTED_PREFILL_LENS, f"{target} is not one of {SUPPORTED_PREFILL_LENS}"
        if real_seq_len > target:
            pytest.skip(f"real prompt is already {real_seq_len} > max bucket {target}")
        ttnn.deallocate(inputs_embeds_tt)
        stretched = _ref_codes_for_icl_len(target, ref_codes, real_seq_len)
        logger.info(f"max prefill: tiling {ref_codes.shape[0]} -> {stretched.shape[0]} reference code frames")
        inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, code_pred_embeds = build_icl(stretched)
        real_seq_len = int(inputs_embeds_tt.shape[2])
        assert real_seq_len == target, f"ICL came out {real_seq_len}, wanted {target}"

    icl_embeds = mesh_to_torch(inputs_embeds_tt).squeeze(1).float()[:, :real_seq_len, :]
    logger.info(f"ICL sequence {real_seq_len} tokens ({prefill_case}), walking {frames} decode frame(s)")

    # Reference-side weights and heads.
    talker_w = extract_talker_weights(weights)
    cp_w_raw = extract_code_predictor_weights(weights)
    cp_w = {(k[len("model.") :] if k.startswith("model.") else k): v.float() for k, v in cp_w_raw.items()}
    codec_head_w = weights["talker.codec_head.weight"].float()
    lm_head_w = [weights[f"talker.code_predictor.lm_head.{g}.weight"].float() for g in range(n_codes - 1)]
    cp_proj_w = weights["talker.code_predictor.small_to_mtp_projection.weight"].float()
    _pb = "talker.code_predictor.small_to_mtp_projection.bias"
    cp_proj_b = weights[_pb].float() if _pb in weights else None
    # Embedding tables: talker's for code 0, the CP's own for codes 1..15. bf16 in
    # the checkpoint, and that is what the device uploads, so keep the same values.
    codec_table = mesh_to_torch(model.talker.codec_embedding).squeeze(0).squeeze(0).float()
    cp_tables = [t.float() for t in code_pred_embeds]

    def cp_project(x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), cp_proj_w, cp_proj_b)

    # =========================================================================
    # Stage 2/3 — Talker prefill (28 layers + final norm) and codec_head.
    # =========================================================================
    talker_trans_mat = get_transformation_mat(talker_cfg.head_dim, device)

    inputs_embeds_tt, padded_seq_len = pad_inputs_to_demo_bucket(device, inputs_embeds_tt, real_seq_len, talker_h)
    talker_kv, max_talker_seq_len = allocate_talker_kv(device, model, padded_seq_len, config.max_new_tokens)
    pf_cos, pf_sin = get_rope_tensors(
        device, talker_cfg.head_dim, padded_seq_len, torch.arange(padded_seq_len), talker_cfg.rope_theta
    )
    hidden_tt, talker_kv = model.talker.forward_from_hidden(
        inputs_embeds_tt, pf_cos, pf_sin, talker_trans_mat, kv_caches=talker_kv, start_pos=0, mode="prefill"
    )
    if padded_seq_len > _HEAD_FULL_SEQ_LIMIT:
        last_row_tt = ttnn.slice(hidden_tt, [0, 0, real_seq_len - 1, 0], [1, 1, real_seq_len, talker_h])
        logits_tt = model.talker.get_codec_logits(last_row_tt)
        ttnn.synchronize_device(device)
        lg_dev = mesh_to_torch(logits_tt).float().reshape(1, -1)
        ttnn.deallocate(last_row_tt)
    else:
        logits_tt = model.talker.get_codec_logits(hidden_tt)
        ttnn.synchronize_device(device)
        lg_dev = mesh_to_torch(logits_tt).squeeze(1).float()[:, real_seq_len - 1, :]
    # Padding sits after every real token and attention is causal, so the real
    # positions are unaffected by the bucket and the reference runs unpadded.
    h_dev = mesh_to_torch(hidden_tt).squeeze(1).float()[:, :real_seq_len, :]
    ttnn.deallocate(pf_cos)
    ttnn.deallocate(pf_sin)

    h_ref = _ref_talker(icl_embeds, talker_w, ref_talker_cfg)
    record("talker_prefill_hidden", h_ref, h_dev)
    lg_ref = F.linear(h_ref[:, -1, :], codec_head_w)
    record("codec_head_prefill", lg_ref, lg_dev)
    ttnn.deallocate(logits_tt)

    code0_ref = int(lg_ref.argmax().item())
    code0_dev = int(lg_dev.argmax().item())
    agree = [code0_ref == code0_dev]
    logger.info(f"prefill code 0: reference {code0_ref}, device {code0_dev}")

    # =========================================================================
    # Frame loop. Ordering is ar_decode_loop's: the CodePredictor consumes the
    # Talker hidden state and code 0 that already exist, then the next Talker
    # input embedding is built, then the Talker decodes one step and emits the
    # code 0 for the following frame.
    # =========================================================================
    cp = _cp_buffers(device, model, cp_num_heads)
    embed_tables = upload_embed_tables(device, codec_table, cp_tables)

    tk_cos, tk_sin, tk_mask, tk_pos = build_talker_decode_trace_h2d_constants(
        *compute_rope_frequencies(talker_cfg.head_dim, max_talker_seq_len + 50, talker_cfg.rope_theta),
        talker_num_heads,
        max_talker_seq_len,
        real_seq_len,
    )
    dc_embed_tt = _upload(torch.zeros(1, 1, 1, talker_h, dtype=torch.bfloat16), device)
    dc_cos_tt = _upload(torch.ones(1, 1, 1, talker_cfg.head_dim, dtype=torch.bfloat16), device)
    dc_sin_tt = _upload(torch.zeros(1, 1, 1, talker_cfg.head_dim, dtype=torch.bfloat16), device)
    dc_mask_tt = _upload(torch.full((1, 1, 1, max_talker_seq_len), float("-inf")).bfloat16(), device)
    dc_pos_tt = ttnn.from_torch(
        torch.tensor([real_seq_len], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cp_worst_per_frame: List[float] = []
    cp_all: List[float] = []  # every code of every frame, for the spread report
    tk_pcc: List[float] = []
    ch_pcc: List[float] = []
    fe_pcc: List[float] = []
    fe_exact: List[bool] = []

    # The running input-embedding sequence. Both models share it: with teacher
    # forcing the codes are identical, and the embedding sum is table lookups
    # only, so the Talker's input is bit-identical on both sides every frame and
    # the drift measured below is the transformer's alone.
    embed_seq_ref = icl_embeds.clone()
    h_ref_last = h_ref[:, -1, :]
    h_dev_last = h_dev[:, -1, :]  # last REAL position, both prefill paths

    for f in range(frames):
        # ---- CodePredictor: codes 1..15 -------------------------------------
        _zero_cp_caches(cp)
        cp_in_dev = torch.cat(
            [h_dev_last.reshape(1, 1, talker_h), codec_table[code0_ref].reshape(1, 1, talker_h)], dim=1
        ).bfloat16()
        cp_pf_tt = _upload(cp_in_dev.reshape(1, 1, 2, talker_h), device)

        # Reference CP sequence, in the device's position order: 0 = Talker
        # hidden, 1 = talker codec_embedding[code0], g+1 = cp_table[g-1][code_g].
        cp_seq_ref = torch.cat(
            [h_ref_last.reshape(1, 1, talker_h), codec_table[code0_ref].reshape(1, 1, talker_h)], dim=1
        )

        cp_pcc: List[float] = []
        code_row_ref = [code0_ref]
        code_row_dev = [code0_dev]
        for g in range(1, n_codes):
            hid_ref = _ref_cp_hidden(cp_project(cp_seq_ref), cp_w, ref_cp_cfg)[:, g, :]
            g_lg_ref = F.linear(hid_ref, lm_head_w[g - 1])

            if g == 1:
                lg_tt, _ = model.code_predictor.forward_single_step(
                    cp_pf_tt,
                    cp["pf_cos"],
                    cp["pf_sin"],
                    cp["trans_mat"],
                    generation_step=1,
                    kv_caches=cp["caches"],
                    start_pos=0,
                    mode="prefill",
                    cp_prefill_mask=cp["pf_mask"],
                )
                vocab = int(lg_tt.shape[3])
                sl = ttnn.slice(lg_tt, [0, 0, 1, 0], [1, 1, 2, vocab])
                g_lg_dev = mesh_to_torch(sl).float().reshape(1, -1)
                ttnn.deallocate(sl)
                ttnn.deallocate(lg_tt)
            else:
                # Position g holds cp_table[g-2][code_{g-1}] — teacher forced, so
                # the device is stepped on the reference's previous code.
                emb = cp_tables[g - 2][code_row_ref[g - 1]].reshape(1, 1, 1, talker_h).bfloat16()
                dc_tt = _upload(emb, device)
                lg_tt, _ = model.code_predictor.forward_single_step(
                    dc_tt,
                    cp["dc_cos"][g - 2],
                    cp["dc_sin"][g - 2],
                    cp["trans_mat"],
                    generation_step=g,
                    kv_caches=cp["caches"],
                    start_pos=g,
                    mode="decode",
                    decode_attn_mask=cp["dc_mask"][g - 2],
                )
                g_lg_dev = mesh_to_torch(lg_tt).float().reshape(1, -1)
                ttnn.deallocate(lg_tt)
                ttnn.deallocate(dc_tt)

            _, p = comp_pcc(g_lg_ref, g_lg_dev, _threshold(prefill_case, "cp_logits"))
            cp_pcc.append(float(p))
            code_row_ref.append(int(g_lg_ref.argmax().item()))
            code_row_dev.append(int(g_lg_dev.argmax().item()))
            cp_seq_ref = torch.cat([cp_seq_ref, cp_tables[g - 1][code_row_ref[g]].reshape(1, 1, talker_h)], dim=1)

        cp_worst_per_frame.append(min(cp_pcc))
        cp_all.extend(cp_pcc)
        ttnn.deallocate(cp_pf_tt)
        agree.extend(a == b for a, b in zip(code_row_ref[1:], code_row_dev[1:]))
        logger.info(f"frame {f} codes: reference {code_row_ref}")
        logger.info(f"frame {f} codes: device    {code_row_dev}")

        # ---- Next Talker input embedding ------------------------------------
        trail_len = int(trailing_text_hidden.shape[1])
        trail_row = trailing_text_hidden[:, f : f + 1, :] if f < trail_len else tts_pad_embed
        host_embed = torch.zeros(1, 1, talker_h, dtype=torch.float32)
        for i, tok in enumerate(code_row_ref):
            table = codec_table if i == 0 else cp_tables[i - 1]
            host_embed += table[tok].reshape(1, 1, talker_h)
        host_embed = (host_embed + trail_row).bfloat16().float()

        dev_embed = _device_frame_embed(device, model, code_row_ref, trail_row, embed_tables)
        _, p = comp_pcc(host_embed, dev_embed, _threshold(prefill_case, "frame_embed_device"))
        fe_pcc.append(float(p))
        fe_exact.append(bool(torch.equal(host_embed, dev_embed)))

        # ---- Talker decode ---------------------------------------------------
        i_h2d = f  # decode position T == real_seq_len + f
        ttnn.copy_host_to_device_tensor(tk_cos[i_h2d], dc_cos_tt)
        ttnn.copy_host_to_device_tensor(tk_sin[i_h2d], dc_sin_tt)
        ttnn.copy_host_to_device_tensor(tk_mask[i_h2d], dc_mask_tt)
        ttnn.copy_host_to_device_tensor(tk_pos[i_h2d], dc_pos_tt)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                host_embed.reshape(1, 1, 1, talker_h).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            dc_embed_tt,
        )
        ttnn.deallocate(hidden_tt)
        hidden_tt, talker_kv = model.talker.forward_from_hidden(
            dc_embed_tt,
            dc_cos_tt,
            dc_sin_tt,
            talker_trans_mat,
            kv_caches=talker_kv,
            cur_pos_tensor=dc_pos_tt,
            decode_attn_mask=dc_mask_tt,
            mode="decode",
        )
        logits_tt = model.talker.get_codec_logits(hidden_tt)
        ttnn.synchronize_device(device)
        h_dev_last = mesh_to_torch(hidden_tt).float().reshape(1, talker_h)
        lg_dev = mesh_to_torch(logits_tt).float().reshape(1, -1)
        ttnn.deallocate(logits_tt)

        embed_seq_ref = torch.cat([embed_seq_ref, host_embed], dim=1)
        h_ref_full = _ref_talker(embed_seq_ref, talker_w, ref_talker_cfg)
        h_ref_last = h_ref_full[:, -1, :]
        lg_ref = F.linear(h_ref_last, codec_head_w)

        _, p = comp_pcc(h_ref_last, h_dev_last, _threshold(prefill_case, "talker_decode_hidden"))
        tk_pcc.append(float(p))
        _, p = comp_pcc(lg_ref, lg_dev, _threshold(prefill_case, "codec_head_decode"))
        ch_pcc.append(float(p))

        code0_ref = int(lg_ref.argmax().item())
        code0_dev = int(lg_dev.argmax().item())
        agree.append(code0_ref == code0_dev)

    gate_first_frame("talker_decode_hidden", tk_pcc)
    gate_first_frame("codec_head_decode", ch_pcc)
    gate_first_frame("cp_logits", cp_worst_per_frame)
    gate_min("frame_embed_device", fe_pcc)

    # =========================================================================
    # Report, then assert.
    # =========================================================================
    token_agreement = sum(agree) / len(agree)
    print(f"\n  checkpoint  {hf_id()}")
    print(f"  device      {os.environ.get('MESH_DEVICE', 'single')}  TP={tp}")
    print(
        f"  prompt      {prefill_case}: {real_seq_len} ICL tokens -> bucket {padded_seq_len}, "
        f"{frames} decode frame(s)"
    )
    print(f"\n  {'stage':<24} {'PCC':>10}  {'threshold':>10}")
    failures = []
    for name, pcc in results:
        threshold = _threshold(prefill_case, name)
        ok = pcc >= threshold
        print(f"  {name:<24} {pcc:>10.6f}  {threshold:>10.4f}  {'' if ok else '<-- FAIL'}")
        if not ok:
            failures.append(
                f"{name}: PCC {pcc:.6f} < {threshold:.4f} (expected {EXPECTED_PCC[prefill_case][name]:.4f})"
            )
    if frames > 1:
        # Worst / best / mean over the frames actually walked. Only `worst` is
        # enforced (against FLOOR_PCC) and only `frame 0` is gated tightly; best and
        # mean are here to show the shape of the spread, because a stage sitting at
        # its floor every frame and a stage that dips there once are different
        # problems and the single gated number cannot tell them apart.
        print(f"\n  per-frame statistics over {frames} frames (frame 0 gated; worst held to FLOOR_PCC)")
        print(f"  {'stage':<24} {'frame 0':>10} {'worst':>10} {'best':>10} {'mean':>10} {'floor':>10}")
        for name, values in trajectories.items():
            has_floor = name in FLOOR_PCC[prefill_case]
            floor = f"{_floor(prefill_case, name):>10.4f}" if has_floor else f"{'-':>10}"
            print(
                f"  {name:<24} {values[0]:>10.6f} {min(values):>10.6f} {max(values):>10.6f} "
                f"{sum(values)/len(values):>10.6f} {floor}"
            )
        print("\n  per-frame trajectory")
        for name, values in trajectories.items():
            print(f"  {name:<24} " + "  ".join(f"{v:.6f}" for v in values))
    for name, values in trajectories.items():
        if name not in FLOOR_PCC[prefill_case]:
            continue
        worst_frame = int(min(range(len(values)), key=lambda i: values[i]))
        if values[worst_frame] < _floor(prefill_case, name):
            failures.append(
                f"{name}: frame {worst_frame} PCC {values[worst_frame]:.6f} < floor "
                f"{_floor(prefill_case, name):.4f} (24-frame minimum was {FLOOR_PCC[prefill_case][name]:.4f})"
            )
    print(
        f"\n  cp_logits across all {len(cp_all)} code x frame comparisons: "
        f"worst {min(cp_all):.6f}  best {max(cp_all):.6f}  mean {sum(cp_all)/len(cp_all):.6f}"
    )
    print(f"  on-device frame embedding bit-equal to host sum:  {all(fe_exact)}")
    print(
        f"  greedy token agreement  {token_agreement*100:.1f} %  "
        f"({sum(agree)}/{len(agree)} codes, reported not gated -- see module docstring)"
    )

    assert not failures, "full-model PCC regressions:\n  " + "\n  ".join(failures)
