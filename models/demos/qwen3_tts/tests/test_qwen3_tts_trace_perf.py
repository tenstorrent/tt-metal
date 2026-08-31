# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Traced Talker + ICL + CodePredictor perf tests for qwen3_tts.

Compile once, capture Metal traces, warm-replay, then replay inside tracy
signposts. Signpost map (use with ``tt-perf-report --start-signpost X
--end-signpost Y``):

Prefill (``test_single_prefill_perf``)::

    start ──► icl_device ──► talker_prefill ──► stop

    full prefill device path:  --start-signpost start --end-signpost stop
    ICL embeds + speaker-enc:  --start-signpost icl_device --end-signpost talker_prefill
    Talker prefill only:       --start-signpost talker_prefill --end-signpost stop

Decode / one AR frame (``test_single_decode_perf``)::

    start ──► cp_prefill ──► cp_decode ──► talker_decode ──► stop

    full AR device path:       --start-signpost start --end-signpost stop
    CP prefill only:           --start-signpost cp_prefill --end-signpost cp_decode
    CP residual decodes:       --start-signpost cp_decode --end-signpost talker_decode
    Talker decode only:        --start-signpost talker_decode --end-signpost stop

Run under Tracy, then summarize with ``tt-perf-report``.
"""

import os

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id
from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat
from models.demos.qwen3_tts.tt.server import SUPPORTED_PREFILL_LENS

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_a, **_k):
        pass


_TILE = 32
_TRACE_REGION = 400_000_000
_L1_SMALL = 32768
_DEFAULT_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_DEFAULT_PREFILL_BUCKET = 128
_DEFAULT_DECODE_POS = 128
_MAX_NEW_TOKENS = 256
_CP_MAX_SEQ = 32
_ICL_EMBED_SEQ = 32


def _hf_id() -> str:
    return os.getenv("HF_MODEL") or os.getenv("QWEN3_TTS_HF_ID", _DEFAULT_HF_ID)


def _padded_max_seq(prefill_len: int, max_new_tokens: int = _MAX_NEW_TOKENS) -> int:
    raw = prefill_len + max_new_tokens + 16
    return ((raw + _TILE - 1) // _TILE) * _TILE


def _prefill_bucket() -> int:
    seq = int(os.getenv("QWEN3_TTS_PREFILL_PERF_SEQ_LEN", str(_DEFAULT_PREFILL_BUCKET)))
    if seq not in SUPPORTED_PREFILL_LENS:
        raise ValueError(
            f"QWEN3_TTS_PREFILL_PERF_SEQ_LEN={seq} is not a production bucket. " f"Use one of {SUPPORTED_PREFILL_LENS}."
        )
    return seq


@pytest.fixture(scope="module")
def state_dict():
    from models.demos.qwen3_tts.tt.server import load_weights

    main_dict, _ = load_weights(_hf_id())
    return main_dict


def _build_model(device, state_dict):
    device.enable_program_cache()
    talker_config = talker_config_for_hf_id(_hf_id())
    return Qwen3TTS(device=device, state_dict=state_dict, talker_config=talker_config)


def _drop(outs):
    if outs is None:
        return
    if isinstance(outs, (list, tuple)):
        for o in outs:
            if o is not None:
                ttnn.deallocate(o)
    else:
        ttnn.deallocate(outs)


def _capture_trace(device, fn):
    """Warmup (compile) then capture. Returns (trace_id, captured_outputs)."""
    warm = fn()
    ttnn.synchronize_device(device)
    _drop(warm)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        captured = fn()
    finally:
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    return trace_id, captured


def _execute(device, trace_id):
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)


def _se_trace_ids(model):
    ids = []
    se = model.speaker_encoder
    traces = getattr(se, "_se_traces", None) or {}
    for block in sorted(traces):
        ids.append(traces[block]["trace_id"])
    fc = getattr(se, "_fc_trace", None)
    if fc is not None:
        ids.append(fc["trace_id"])
    return ids


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": _L1_SMALL, "trace_region_size": _TRACE_REGION}],
    indirect=True,
)
def test_single_decode_perf(device, state_dict):
    """One AR-frame device path: CP prefill + CP residual decodes + Talker decode.

    Signposts: start → cp_prefill → cp_decode → talker_decode → stop.
    """
    model = _build_model(device, state_dict)
    talker = model.talker
    cp = model.code_predictor
    tcfg = model.talker_config
    cpcfg = model.code_predictor_config
    decode_pos = int(os.getenv("QWEN3_TTS_DECODE_POS", str(_DEFAULT_DECODE_POS)))
    max_talker_seq = _padded_max_seq(decode_pos)
    n_cp_decode = cpcfg.num_code_groups - 2  # generation_step 2 .. num_code_groups-1

    talker_trans = get_transformation_mat(tcfg.head_dim, device)
    cp_trans = get_transformation_mat(cpcfg.head_dim, device)

    # ── Talker decode buffers ──────────────────────────────────────────────
    talker_embed = ttnn.from_torch(
        torch.zeros(1, 1, 1, tcfg.hidden_size, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    talker_cos, talker_sin = get_rope_tensors(device, tcfg.head_dim, 1, torch.tensor([decode_pos]), tcfg.rope_theta)
    talker_cur_pos = ttnn.from_torch(
        torch.tensor([decode_pos], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    talker_mask_h = torch.full((1, tcfg.num_attention_heads, 1, max_talker_seq), float("-inf"))
    talker_mask_h[0, :, 0, : decode_pos + 1] = 0.0
    talker_mask = ttnn.from_torch(
        talker_mask_h, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    talker_kv = create_kv_cache_list(device, tcfg, max_batch_size=1, max_seq_len=max_talker_seq)

    def talker_decode_fn():
        hidden, _ = talker.forward_from_hidden(
            talker_embed,
            talker_cos,
            talker_sin,
            talker_trans,
            kv_caches=talker_kv,
            start_pos=decode_pos,
            mode="decode",
            cur_pos_tensor=talker_cur_pos,
            decode_attn_mask=talker_mask,
        )
        return hidden, talker.get_codec_logits(hidden)

    # ── CP prefill buffers (seq=2, talker-hidden + code0 embed) ────────────
    cp_pf_embed = ttnn.from_torch(
        torch.zeros(1, 1, 2, tcfg.hidden_size, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_pf_cos, cp_pf_sin = get_rope_tensors(device, cpcfg.head_dim, 2, torch.arange(2), cpcfg.rope_theta)
    cp_kv = create_kv_cache_list(device, cpcfg, max_batch_size=1, max_seq_len=_CP_MAX_SEQ)
    cp_pf_mask_h = torch.full((1, cpcfg.num_attention_heads, 2, _CP_MAX_SEQ), float("-inf"))
    cp_pf_mask_h[0, :, 0, 0] = 0.0
    cp_pf_mask_h[0, :, 1, 0:2] = 0.0
    cp_pf_mask = ttnn.from_torch(
        cp_pf_mask_h, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def cp_prefill_fn():
        logits, _ = cp.forward_single_step(
            cp_pf_embed,
            cp_pf_cos,
            cp_pf_sin,
            cp_trans,
            generation_step=1,
            kv_caches=cp_kv,
            start_pos=0,
            mode="prefill",
            cp_prefill_mask=cp_pf_mask,
            return_hidden_state=False,
        )
        return logits

    # ── CP decode buffers (one persistent embed, per-step RoPE/mask) ───────
    cp_dc_embed = ttnn.from_torch(
        torch.zeros(1, 1, 1, tcfg.hidden_size, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_dc_steps = []
    for step_code_idx in range(2, cpcfg.num_code_groups):
        cos_i, sin_i = get_rope_tensors(device, cpcfg.head_dim, 1, torch.tensor([step_code_idx]), cpcfg.rope_theta)
        mask_h = torch.full((1, cpcfg.num_attention_heads, 1, _CP_MAX_SEQ), float("-inf"))
        mask_h[0, :, 0, : step_code_idx + 1] = 0.0
        mask_i = ttnn.from_torch(
            mask_h, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cp_dc_steps.append((step_code_idx, cos_i, sin_i, mask_i))

    def _make_cp_decode_fn(step_code_idx, cos_i, sin_i, mask_i):
        def _fn():
            logits, _ = cp.forward_single_step(
                cp_dc_embed,
                cos_i,
                sin_i,
                cp_trans,
                generation_step=step_code_idx,
                kv_caches=cp_kv,
                start_pos=step_code_idx,
                mode="decode",
                cur_pos_tensor=None,
                decode_attn_mask=mask_i,
                return_hidden_state=False,
            )
            return logits

        return _fn

    print(f"  Capturing CP prefill + {n_cp_decode} CP decode + Talker decode traces...")
    cp_pf_id, _ = _capture_trace(device, cp_prefill_fn)
    cp_dc_ids = []
    for step_code_idx, cos_i, sin_i, mask_i in cp_dc_steps:
        tid, _ = _capture_trace(device, _make_cp_decode_fn(step_code_idx, cos_i, sin_i, mask_i))
        cp_dc_ids.append(tid)
    talker_id, talker_outs = _capture_trace(device, talker_decode_fn)
    print(f"  Captured CP prefill=1, CP decode={len(cp_dc_ids)}, Talker decode=1")

    # Warm replay (dispatch-path fill, not signposted).
    _execute(device, cp_pf_id)
    for tid in cp_dc_ids:
        _execute(device, tid)
    _execute(device, talker_id)

    signpost("start")
    signpost("cp_prefill")
    _execute(device, cp_pf_id)
    signpost("cp_decode")
    for tid in cp_dc_ids:
        _execute(device, tid)
    signpost("talker_decode")
    _execute(device, talker_id)
    signpost("stop")

    _, logits = talker_outs
    out = ttnn.to_torch(logits).float()
    assert torch.isfinite(out).all(), "traced AR-step Talker decode produced non-finite logits"
    print(
        f"Traced AR-step decode complete (pos={decode_pos}, max_talker_seq={max_talker_seq}, "
        f"cp_decodes={len(cp_dc_ids)}); logits {tuple(out.shape)}"
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": _L1_SMALL, "trace_region_size": _TRACE_REGION}],
    indirect=True,
)
def test_single_prefill_perf(device, state_dict):
    """ICL device ops (embed/proj + speaker-encoder traces) + Talker prefill.

    Signposts: start → icl_device → talker_prefill → stop.
    """
    model = _build_model(device, state_dict)
    talker = model.talker
    tcfg = model.talker_config
    seq_len = _prefill_bucket()
    max_seq_len = _padded_max_seq(seq_len)
    trans_mat = get_transformation_mat(tcfg.head_dim, device)

    # ── ICL device: text embed + projection + codec embed ──────────────────
    text_ids = ttnn.from_torch(
        torch.ones(1, _ICL_EMBED_SEQ, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    codec_ids = ttnn.from_torch(
        torch.ones(1, _ICL_EMBED_SEQ, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def icl_fn():
        text_emb = talker.get_text_embedding(text_ids)
        text_proj = talker.project_text(text_emb)
        codec_emb = talker.get_codec_embedding(codec_ids)
        return text_emb, text_proj, codec_emb

    print("  Capturing speaker-encoder block + FC traces...")
    model.speaker_encoder.capture_se_block_traces()
    model.speaker_encoder.capture_fc_trace()
    se_ids = _se_trace_ids(model)
    print(f"  Speaker-encoder traces: {len(se_ids)}")

    print("  Capturing ICL embed/proj + Talker prefill traces...")
    icl_id, _ = _capture_trace(device, icl_fn)

    embed_tt = ttnn.from_torch(
        torch.zeros(1, 1, seq_len, tcfg.hidden_size, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt, sin_tt = get_rope_tensors(device, tcfg.head_dim, seq_len, torch.arange(seq_len), tcfg.rope_theta)
    kv_caches = create_kv_cache_list(device, tcfg, max_batch_size=1, max_seq_len=max_seq_len)

    def prefill_fn():
        hidden, _ = talker.forward_from_hidden(
            embed_tt,
            cos_tt,
            sin_tt,
            trans_mat,
            kv_caches=kv_caches,
            start_pos=0,
            mode="prefill",
        )
        return hidden, talker.get_codec_logits(hidden)

    talker_id, talker_outs = _capture_trace(device, prefill_fn)

    # Warm replay.
    for tid in se_ids:
        _execute(device, tid)
    _execute(device, icl_id)
    _execute(device, talker_id)

    signpost("start")
    signpost("icl_device")
    for tid in se_ids:
        _execute(device, tid)
    _execute(device, icl_id)
    signpost("talker_prefill")
    _execute(device, talker_id)
    signpost("stop")

    _, logits = talker_outs
    out = ttnn.to_torch(logits).float()
    assert torch.isfinite(out).all(), "traced Talker prefill produced non-finite logits"
    print(
        f"Traced ICL+Talker prefill complete (bucket={seq_len}, max_seq={max_seq_len}, "
        f"se_traces={len(se_ids)}); logits {tuple(out.shape)}"
    )
