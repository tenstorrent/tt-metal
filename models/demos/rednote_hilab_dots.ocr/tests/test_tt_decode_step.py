# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decode-step PCC guardrail for the dots.ocr sharded-decode rewrite.

This is the FAST inner-loop gate for the Phase 1-4 sharded-decode work (see
``docs/superpowers/plans/2026-05-30-dots-ocr-sharded-decode.md``). It does NOT
run the full 28L e2e -- it builds a small (2-layer) real-weight LM + KV cache,
prefills a short fixed prompt to populate the cache, then runs ONE cached decode
step at a fixed position and checks the decode-step logits against:

  1. a pure-PyTorch reference (``reference.functional.language_model_forward``
     run over ``prompt + next_token`` with logits taken at the last position --
     which is exactly what a cached decode step at ``pos = prompt_len`` produces),
     at PCC > 0.99; AND
  2. the trace-captured decode path (``decode_step_traced``) vs the untraced
     decode path (``decode_step``) at the same position, at PCC > 0.99.

Real OCR checkpoint weights are loaded via ``weight_loader.load_language_model_weights``
(the same loader the other real-weight tests use), at a REDUCED layer count
(2) to keep the gate quick. The decode step is invoked in isolation via the
``TtLanguageModel.decode_step`` / ``decode_step_traced`` / ``write_decode_pos``
API and a ``SelfAttentionKVCache`` -- the LM's decode entry points are clean and
do NOT require the generator.

The model dir name (rednote_hilab_dots.ocr) contains a dot, so siblings are
imported by file path with importlib (the project convention).

Run::

    pytest models/demos/rednote_hilab_dots.ocr/tests/test_tt_decode_step.py -v -m device
"""
import importlib.util
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

_TT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tt"))
_REF_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "reference"))

CHECKPOINT_PATH = os.environ.get(
    "DOTS_OCR_CHECKPOINT",
    "/local/ttuser/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/"
    "c0111ce6bc07803dbc267932ffef0ae3a51dc951",
)

# Small fast gate: 2 layers, a short fixed prompt, one decode step.
NUM_LAYERS = 2
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
ROPE_THETA = 1000000.0
EPS = 1e-6
BIAS = True
HIDDEN = NUM_HEADS * HEAD_DIM

# A short fixed prompt of token ids (arbitrary in-vocab ids; the maths is
# weight-driven, not semantics-driven, so any valid ids exercise the path).
# Length is a tile multiple (32) -- the prefill path's fused head-split /
# materialized-mask ops expect a tile-aligned prompt seq.
PROMPT_IDS = [(100 + 7 * i) % 151000 for i in range(32)]
# The token decoded at step 0 (fed into the single decode step at pos=prompt_len).
NEXT_ID = 5678


def _load_by_path(name, filename, directory):
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_loader = _load_by_path("dots_decode_step_loader", "weight_loader.py", _TT_DIR)
_lm_mod = _load_by_path("dots_decode_step_lm", "language_model.py", _TT_DIR)
_kvc_mod = _load_by_path("dots_decode_step_kv_cache", "kv_cache.py", _TT_DIR)
_functional = _load_by_path("dots_decode_step_functional", "functional.py", _REF_DIR)

TtLanguageModel = _lm_mod.TtLanguageModel
SelfAttentionKVCache = _kvc_mod.SelfAttentionKVCache


def _torch_decode_golden(state_dict):
    """Pure-torch reference logits for the decode step at pos=prompt_len.

    A cached decode step at position ``prompt_len`` (after prefilling PROMPT_IDS)
    produces the next-token logits for the sequence ``PROMPT_IDS + [NEXT_ID]``.
    The no-cache full-causal reference over that full sequence, sliced to its LAST
    position, is the exact same quantity -- so it is a valid golden for the cached
    decode step.

    Returns logits [1, vocab] (fp32).
    """
    full_ids = torch.tensor([PROMPT_IDS + [NEXT_ID]], dtype=torch.long)  # [1, prompt_len+1]
    logits = _functional.language_model_forward(
        full_ids,
        state_dict,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=ROPE_THETA,
        eps=EPS,
        bias=BIAS,
    )  # [1, seq, vocab]
    return logits[:, -1, :].to(torch.float32).reshape(1, -1)


SHARDED_DECODE = os.environ.get("DOTS_SHARDED_DECODE", "0") == "1"


def _build_lm_and_cache(device, state_dict, prompt_len, max_seq_len):
    lm = TtLanguageModel(
        device=device,
        state_dict=state_dict,
        num_layers=NUM_LAYERS,
        seq_len=prompt_len,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=ROPE_THETA,
        eps=EPS,
        bias=BIAS,
        max_seq_len=max_seq_len,
        sharded_decode=SHARDED_DECODE,
    )
    cache = SelfAttentionKVCache(
        device=device,
        num_layers=NUM_LAYERS,
        batch=1,
        num_kv_heads=NUM_KV_HEADS,
        max_seq_len=max_seq_len,
        head_dim=HEAD_DIM,
        dtype=ttnn.bfloat16,
    )
    return lm, cache


def _embed(state_dict, ids):
    """Host gather: ids list -> [len(ids), hidden] fp32 from embed_tokens.weight."""
    table = state_dict["embed_tokens.weight"].to(torch.float32)
    return table[torch.tensor(ids, dtype=torch.long)]  # [n, hidden]


def _to_device_embed(device, embeds):
    return ttnn.from_torch(
        embeds.to(torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run(device):
    torch.manual_seed(0)
    state_dict = _loader.load_language_model_weights(CHECKPOINT_PATH, num_layers=NUM_LAYERS)
    state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}

    prompt_len = len(PROMPT_IDS)
    pos = prompt_len  # the single decode step's position
    max_seq_len = prompt_len + 2  # cache capacity (rounded to 512 internally)

    golden = _torch_decode_golden(state_dict)  # [1, vocab]

    lm, cache = _build_lm_and_cache(device, state_dict, prompt_len, max_seq_len)

    # ---- Prefill the fixed prompt to populate the KV cache. -------------- #
    prompt_embeds = _embed(state_dict, PROMPT_IDS)  # [prompt_len, hidden]
    prefill_in = _to_device_embed(device, prompt_embeds)
    cache.reset()
    _ = lm.prefill_from_embeds(prefill_in, cache)
    ttnn.synchronize_device(device)

    # ---- ONE untraced decode step at pos=prompt_len. --------------------- #
    next_embed = _embed(state_dict, [NEXT_ID])  # [1, hidden]
    decode_in = _to_device_embed(device, next_embed)
    untraced_logits_tt = lm.decode_step(decode_in, pos, cache)
    untraced_logits = ttnn.to_torch(untraced_logits_tt).to(torch.float32).reshape(1, -1)
    ttnn.synchronize_device(device)

    # Gate 1: untraced decode logits vs the torch reference.
    passing_ref, msg_ref = comp_pcc(golden, untraced_logits, 0.99)
    print(comp_allclose(golden, untraced_logits))
    print(f"comp_pcc(decode_step vs torch reference): {msg_ref}")
    pcc_ref = _parse_pcc(msg_ref)

    # ---- Trace-capture the decode step and replay at the SAME pos. ------- #
    # Persistent decode-embed buffer (stable address for trace replay).
    persistent_embed = _to_device_embed(device, next_embed)

    # WARMUP (compile) the traced decode kernels BEFORE capture: a fresh trace
    # capture cannot tolerate the device writes/reads that JIT compilation does,
    # so the path must be run once untraced first (mirrors profile_ocr_traced.py).
    cache.reset()
    _ = lm.prefill_from_embeds(prefill_in, cache)
    ttnn.synchronize_device(device)
    lm.write_decode_pos(pos, cache)
    _ = lm.decode_step_traced(persistent_embed, cache)
    ttnn.synchronize_device(device)

    # Reset + re-prefill so the cache state matches the untraced run, then write
    # the decode position into the persistent pos / RoPE buffers before capture.
    cache.reset()
    _ = lm.prefill_from_embeds(prefill_in, cache)
    ttnn.synchronize_device(device)
    lm.write_decode_pos(pos, cache)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_logits_tt = lm.decode_step_traced(persistent_embed, cache)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Replay: stream the embed + pos into the persistent buffers, execute trace.
    host_embed = ttnn.from_torch(next_embed.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(host_embed, persistent_embed)
    lm.write_decode_pos(pos, cache)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    traced_logits = ttnn.to_torch(traced_logits_tt).to(torch.float32).reshape(1, -1)

    # Gate 2: traced decode logits vs the untraced (verified) decode logits.
    passing_tr, msg_tr = comp_pcc(untraced_logits, traced_logits, 0.99)
    print(comp_allclose(untraced_logits, traced_logits))
    print(f"comp_pcc(decode_step_traced vs decode_step): {msg_tr}")
    pcc_tr = _parse_pcc(msg_tr)

    # Also report traced-vs-torch-reference for completeness.
    _, msg_tr_ref = comp_pcc(golden, traced_logits, 0.99)
    print(f"comp_pcc(decode_step_traced vs torch reference): {msg_tr_ref}")

    assert passing_ref, f"untraced decode step below PCC 0.99 vs torch reference: {msg_ref}"
    assert passing_tr, f"traced decode step below PCC 0.99 vs untraced decode step: {msg_tr}"
    return pcc_ref, pcc_tr


def _parse_pcc(message) -> float:
    msg = str(message)
    return float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(message)


@pytest.mark.device
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 60_000_000}],
    indirect=True,
)
def test_tt_decode_step(device):
    pcc_ref, pcc_tr = _run(device)
    print(f"decode-step guardrail: PCC(decode vs reference)={pcc_ref:.6f}  PCC(traced vs untraced)={pcc_tr:.6f}")


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=60_000_000)
    try:
        _run(dev)
    finally:
        ttnn.close_device(dev)
