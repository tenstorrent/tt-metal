# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 02 correctness tests for ``MusicLLM`` (the Qwen3-8B backbone on one Blackhole chip).

Reference: HF ``Qwen3ForCausalLM`` (transformers 5.x) in bf16 on CPU loaded from ``$HF_MODEL``, plus
the stage-01 golden tensors (fp32 diffusers run) under ``~/mm3-bringup/reference``.

Run (device, serialized):

    source ~/mm3-bringup/common.sh && cd $MM3_WT && \
    with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_llm.py -m "not slow"

Every PCC is written to ``doc/llm/pcc/pcc_results.json`` so the work log quotes measured numbers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.constants import LLM_HIDDEN, LLM_VOCAB, MAX_PROMPT_TOKENS
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.common import Mode

pytestmark = [pytest.mark.hardware, pytest.mark.timeout(3600)]

PCC_LAYER = 0.995
PCC_FULL = 0.99
TEACHER_FORCED_FRAMES = 8

_RESULTS: dict = {}


def _record(evidence_dir: Path, name: str, **fields):
    _RESULTS[name] = fields
    out = evidence_dir / "pcc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pcc_results.json").write_text(json.dumps(_RESULTS, indent=2, sort_keys=True) + "\n")


def _pcc_value(golden, actual) -> float:
    """PCC via ``models.common.utility_functions.comp_pcc`` (returns ``(passing, pcc)``)."""
    _, pcc = comp_pcc(golden.float(), actual.float(), 0.0)
    return float(pcc)


# --------------------------------------------------------------------------- 1. layer 0 vs golden
def test_layer0_prefill_and_decode_vs_golden(music_llm, golden, evidence_dir):
    """Layer 0 alone: prefill of the golden [2, L, 4096] input and the first decode step (position L).

    ``llm_layer0.pt`` holds the fp32 layer-0 input/output of the diffusers run: the prefill call
    (positions 0..L-1) and the first decode call (position L, after the prefill filled the cache).
    """
    d = golden["llm_layer0"]
    pre_in, pre_out = d["prefill_in"], d["prefill_out"]  # [2, L, 4096]
    dec_in, dec_out = d["decode_in"], d["decode_out"]  # [2, 1, 4096]
    batch, seq_len, dim = pre_in.shape
    assert batch == music_llm.max_batch_size and dim == LLM_HIDDEN
    assert int(d["decode_position_ids"].flatten()[0]) == seq_len

    layer = music_llm.model.layers[0]
    from models.tt_transformers.tt.common import get_padded_prefill_len

    seq_pad = get_padded_prefill_len(seq_len)
    rot = music_llm._prefill_rot_mats(0, seq_pad)
    for user in range(batch):
        x = torch.zeros(1, 1, seq_pad, dim)
        x[0, 0, :seq_len] = pre_in[user]
        out = layer(
            music_llm._prefill_input(x),
            None,
            rot_mats_global=rot,
            user_id=user,
            mode=Mode.PREFILL,
            page_table=music_llm.page_table_tt,
        )
        got = ttnn.to_torch(out).float()[0, 0, :seq_len]
        pcc = _pcc_value(pre_out[user], got)
        logger.info(f"layer0 prefill user {user}: PCC {pcc:.6f}")
        _record(evidence_dir, f"layer0_prefill_user{user}", pcc=pcc, bar=PCC_LAYER, seq_len=seq_len)
        assert pcc >= PCC_LAYER, f"layer-0 prefill user {user} PCC {pcc} < {PCC_LAYER}"
        ttnn.deallocate(out)

    # Decode at position L for both rows from the golden decode input.
    pos = torch.full((batch,), seq_len, dtype=torch.int64)
    x_host = music_llm._decode_input_host(dec_in.reshape(batch, dim))
    x = ttnn.to_device(x_host, music_llm.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_memory_config(x, music_llm.args.get_residual_mem_config(Mode.DECODE, None))
    pos_tt = ttnn.from_torch(
        pos.to(torch.int32),
        device=music_llm.mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(music_llm.mesh_device),
    )
    rot_dec = music_llm.model.rope_setup.get_rot_mats(pos)
    out = layer(x, pos_tt, rot_mats_global=rot_dec, mode=Mode.DECODE, page_table=music_llm.page_table_tt)
    got = ttnn.to_torch(out).float()[0, 0, :batch]
    pcc = _pcc_value(dec_out.reshape(batch, dim), got)
    logger.info(f"layer0 decode @pos {seq_len}: PCC {pcc:.6f}")
    _record(evidence_dir, "layer0_decode", pcc=pcc, bar=PCC_LAYER, position=seq_len)
    assert pcc >= PCC_LAYER, f"layer-0 decode PCC {pcc} < {PCC_LAYER}"


# --------------------------------------------------------------------------- 2+3. full stack vs HF
@pytest.fixture(scope="module")
def hf_teacher_forced(hf_model, golden, audio_embeddings):
    text_ids = golden["text_ids"]
    codes = golden["sampled_codes"][:TEACHER_FORCED_FRAMES]
    t0 = time.time()
    hidden0, logits0, steps = R.hf_teacher_forced(hf_model, text_ids, codes, audio_embeddings)
    logger.info(f"HF reference (prefill {tuple(text_ids.shape)} + {len(steps)} frames) took {time.time() - t0:.0f}s")
    return {"hidden": hidden0, "logits": logits0, "steps": steps, "codes": codes}


def test_full_prefill_vs_hf(music_llm, golden, hf_teacher_forced, evidence_dir):
    """Full 36-layer prefill of the golden prompt (batch 2): normed last hidden and logits vs HF."""
    text_ids = golden["text_ids"]
    emb = music_llm.embed_tokens(text_ids)
    t0 = time.time()
    hidden, logits = music_llm.prefill(emb)
    ttnn.synchronize_device(music_llm.mesh_device)
    dt = time.time() - t0
    assert hidden.shape == (text_ids.shape[0], LLM_HIDDEN) and logits.shape == (text_ids.shape[0], LLM_VOCAB)
    pcc_h = _pcc_value(hf_teacher_forced["hidden"], hidden)
    pcc_l = _pcc_value(hf_teacher_forced["logits"], logits)
    top_hf = hf_teacher_forced["logits"].argmax(-1).tolist()
    top_tt = logits.argmax(-1).tolist()
    logger.info(
        f"full prefill: hidden PCC {pcc_h:.6f}, logits PCC {pcc_l:.6f}, argmax HF {top_hf} TT {top_tt}, {dt:.2f}s"
    )
    _record(
        evidence_dir,
        "full_prefill_golden_prompt",
        hidden_pcc=pcc_h,
        logits_pcc=pcc_l,
        bar=PCC_FULL,
        seq_len=int(text_ids.shape[1]),
        argmax_hf=top_hf,
        argmax_tt=top_tt,
        wall_seconds_first_call=dt,
    )
    assert pcc_h >= PCC_FULL, f"prefill hidden PCC {pcc_h} < {PCC_FULL}"
    assert pcc_l >= PCC_FULL, f"prefill logits PCC {pcc_l} < {PCC_FULL}"


def test_teacher_forced_decode_vs_hf(music_llm, golden, hf_model, audio_embeddings, hf_teacher_forced, evidence_dir):
    """Teacher-forced decode over the first golden frames: per-step hidden and logits vs HF.

    Both stacks see the same sequence: the golden prompt, then frame i's codes embedded with the
    exact ``_embed_audio_frame`` formula (semantic code through the backbone embedding, the 7
    residual codes through the depth decoder's ``audio_embeddings``, summed and scaled by 8**-0.5).
    """
    text_ids = golden["text_ids"]
    codes = hf_teacher_forced["codes"]
    batch, seq_len = text_ids.shape
    embed_w = hf_model.model.embed_tokens.weight.detach()

    music_llm.reset_cache()
    music_llm.prefill(music_llm.embed_tokens(text_ids))
    records = []
    for i, (hf_hidden, hf_logits) in enumerate(hf_teacher_forced["steps"]):
        frame = codes[i].unsqueeze(0).expand(batch, -1)
        x = R.embed_audio_frame(embed_w, audio_embeddings, frame)
        hidden, logits = music_llm.decode(x, torch.full((batch,), seq_len + i))
        pcc_h = _pcc_value(hf_hidden, hidden)
        pcc_l = _pcc_value(hf_logits, logits)
        records.append({"step": i, "position": seq_len + i, "hidden_pcc": pcc_h, "logits_pcc": pcc_l})
        logger.info(f"teacher-forced step {i} (pos {seq_len + i}): hidden PCC {pcc_h:.6f} logits PCC {pcc_l:.6f}")
    _record(
        evidence_dir, "teacher_forced_decode", bar=PCC_FULL, steps=records, decode_stats=dict(music_llm.decode_stats)
    )
    for r in records:
        assert r["hidden_pcc"] >= PCC_FULL, r
        assert r["logits_pcc"] >= PCC_FULL, r


def test_embed_frame_matches_formula(music_llm, golden, hf_model, audio_embeddings):
    """The on-device ``embed_frame`` reproduces the host ``_embed_audio_frame`` result (bf16 exact up to rounding)."""
    codes = golden["sampled_codes"][:4]
    embed_w = hf_model.model.embed_tokens.weight.detach()
    for frame in codes:
        fc = frame.unsqueeze(0).expand(music_llm.max_batch_size, -1)
        ref = R.embed_audio_frame(embed_w, audio_embeddings, fc).float()
        res = R.residual_embedding_sum(audio_embeddings, fc)
        got = ttnn.to_torch(music_llm.embed_frame(fc[:, 0], res))[0, 0, : music_llm.max_batch_size].float()
        pcc = _pcc_value(ref, got)
        max_abs = (ref - got).abs().max().item()
        logger.info(f"embed_frame codes {frame.tolist()}: PCC {pcc:.6f}, max abs diff {max_abs:.3e}")
        assert pcc > 0.9999, pcc


# --------------------------------------------------------------------------- 4. odd and max prompt lengths
@pytest.mark.parametrize("seq_len", [1, 333, MAX_PROMPT_TOKENS])
def test_prefill_lengths_run(music_llm, seq_len, evidence_dir):
    """Non-aligned and maximum prompt lengths prefill without error and produce finite outputs, then decode."""
    torch.manual_seed(seq_len)
    emb = torch.randn(music_llm.max_batch_size, seq_len, LLM_HIDDEN) * 0.02  # embedding-table scale
    music_llm.reset_cache()
    t0 = time.time()
    hidden, logits = music_llm.prefill(emb)
    ttnn.synchronize_device(music_llm.mesh_device)
    dt = time.time() - t0
    assert hidden.shape == (music_llm.max_batch_size, LLM_HIDDEN)
    assert logits.shape == (music_llm.max_batch_size, LLM_VOCAB)
    assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
    # One decode step right after the prompt.
    h2, l2 = music_llm.decode(torch.randn(music_llm.max_batch_size, LLM_HIDDEN) * 0.02, seq_len)
    assert torch.isfinite(h2).all() and torch.isfinite(l2).all()
    positions = [seq_len]
    if seq_len == MAX_PROMPT_TOKENS:
        # The last position of the advertised context: the paged SDPA decode reads all 10240 slots.
        last = music_llm.max_seq_len - 1
        h3, l3 = music_llm.decode(torch.randn(music_llm.max_batch_size, LLM_HIDDEN) * 0.02, last)
        assert torch.isfinite(h3).all() and torch.isfinite(l3).all()
        positions.append(last)
    logger.info(f"prefill seq_len {seq_len}: {dt:.2f}s (first call at this padded length includes compile)")
    _record(evidence_dir, f"prefill_len_{seq_len}_runs", seq_len=seq_len, wall_seconds=dt, decode_positions=positions)


@pytest.mark.slow
def test_max_prompt_prefill_vs_hf(music_llm, hf_model, evidence_dir):
    """Maximum prompt (5000 tokens, chunked 2 x 4096 prefill) and the following decode step vs HF.

    Random token ids (seeded) so the prompt is realistic text-embedding input. Slow: the HF bf16
    CPU prefill of 2 x 5000 tokens takes minutes, hence outside the gate.
    """
    torch.manual_seed(5000)
    ids = torch.randint(0, 150000, (music_llm.max_batch_size, MAX_PROMPT_TOKENS))
    t0 = time.time()
    past, hf_hidden, hf_logits = R.hf_prefill(hf_model, hf_model.model.embed_tokens(ids))
    step_in = torch.randn(music_llm.max_batch_size, LLM_HIDDEN) * 0.02
    _, hf_h2, hf_l2 = R.hf_decode_step(hf_model, past, step_in.to(torch.bfloat16).unsqueeze(1))
    logger.info(f"HF 5000-token prefill + decode took {time.time() - t0:.0f}s")
    music_llm.reset_cache()
    hidden, logits = music_llm.prefill(music_llm.embed_tokens(ids))
    h2, l2 = music_llm.decode(step_in, MAX_PROMPT_TOKENS)
    rec = {
        "prefill_hidden_pcc": _pcc_value(hf_hidden, hidden),
        "prefill_logits_pcc": _pcc_value(hf_logits, logits),
        "decode_hidden_pcc": _pcc_value(hf_h2, h2),
        "decode_logits_pcc": _pcc_value(hf_l2, l2),
    }
    logger.info(f"max prompt vs HF: {rec}")
    _record(
        evidence_dir,
        "max_prompt_5000_vs_hf",
        bar=PCC_FULL,
        seq_len=MAX_PROMPT_TOKENS,
        decode_position=MAX_PROMPT_TOKENS,
        **rec,
    )
    for k, v in rec.items():
        assert v >= PCC_FULL, (k, v)


# --------------------------------------------------------------------------- 5. determinism
def test_decode_determinism(music_llm, golden):
    """Two identical traced decode steps (same input, same position) produce bit-identical outputs."""
    text_ids = golden["text_ids"]
    seq_len = text_ids.shape[1]
    music_llm.reset_cache()
    music_llm.prefill(music_llm.embed_tokens(text_ids))
    torch.manual_seed(0)
    x = torch.randn(music_llm.max_batch_size, LLM_HIDDEN) * 0.02
    h1, l1 = music_llm.decode(x, seq_len)
    h2, l2 = music_llm.decode(x, seq_len)
    assert torch.equal(h1, h2), (h1 - h2).abs().max()
    assert torch.equal(l1, l2), (l1 - l2).abs().max()
    # And the replay reacts to a changed input / position (no stale trace inputs).
    h3, _ = music_llm.decode(x * 2.0, seq_len + 1)
    assert not torch.equal(h1, h3)
