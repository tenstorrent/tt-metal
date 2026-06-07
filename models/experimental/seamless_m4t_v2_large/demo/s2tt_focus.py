# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused S2TT (Speech-to-Text Translation) demo + correctness/latency analysis.

Runs ONLY task 3 (Hindi speech -> English text) from the full demo, reusing the
Hindi WAV that the full demo's T2ST step writes to ``outputs/t2st_hindi_speech.wav``.
If that WAV is missing, run the full demo once first to produce it.

Adds two things the full demo does not do for S2TT:

  1. **Correctness** — compares the TT greedy output against a CPU HuggingFace
     reference run (token-id prefix match + decoded-text equality).
  2. **Latency breakdown** — wraps ``_encode_speech`` to split total generate time
     into speech-encoder prefill vs. text-decoder loop (prefill + N decode steps),
     so we can confirm the README claim that S2TT is speech-encoder dominated.

Run from repo root:

  python models/experimental/seamless_m4t_v2_large/demo/s2tt_focus.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the helper functions already written for the full demo.
from models.experimental.seamless_m4t_v2_large.demo.demo import (
    T2ST_WAV,
    _decode,
    _hf_gen_kwargs,
    _load_mono_wav,
    _text_tokens_generated,
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import (
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    DEVICE_PARAMS_P150_E2E_2CQ_GENERATE,
    MESH_SHAPE_P150,
    open_seamless_mesh_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GreedySearchOutput,
)

# One warmup (untimed) + N timed iters; min() over iters drops host jitter.
_WARMUP_ITERS = 1
_MEASURE_ITERS = 3

SRC_LANG_AUDIO = "hin"  # the WAV is Hindi speech
TGT_TEXT = "eng"  # translate to English text


def _weights_dir() -> Path:
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


def main() -> None:
    if not T2ST_WAV.exists():
        raise SystemExit(
            f"Input WAV not found: {T2ST_WAV}\n"
            "Run the full demo once (python .../demo/demo.py) to generate the Hindi T2ST audio first."
        )

    weights_dir = _weights_dir()
    path = os.fspath(weights_dir)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    sample_rate = int(getattr(cfg, "sampling_rate", 16000))

    # ---- Audio input (Hindi speech) -> mel features ----
    hindi_wav = _load_mono_wav(T2ST_WAV)
    audio_inputs = processor(audios=hindi_wav, sampling_rate=sample_rate, return_tensors="pt")
    input_features = audio_inputs["input_features"]
    input_speech_attn = audio_inputs["attention_mask"]
    mel_frames = int(input_speech_attn.sum().item())
    feat_seq = int(input_features.shape[1])
    print("=" * 78)
    print("  S2TT — Speech-to-Text Translation (Hindi speech -> English text)")
    print("=" * 78)
    print(
        f"  Input audio: {T2ST_WAV.name}  {hindi_wav.size} samples "
        f"({hindi_wav.size / sample_rate:.2f}s @ {sample_rate} Hz), "
        f"mel_frames={mel_frames}, feat_seq={feat_seq}"
    )

    gen_common = hf_aligned_generation_kwargs(
        model.generation_config,
        use_kv_cache=True,
        use_decode_trace=True,
        use_2cq=True,
    )
    max_new_tokens = int(gen_common["max_new_tokens"])
    eos_token_id = int(gen_common["eos_token_id"])

    # ---- CPU HuggingFace reference (correctness baseline) ----
    print("\n  Running CPU HF reference (this is slow for long audio)...")
    t_ref0 = time.perf_counter()
    with torch.no_grad():
        hf_out = model.generate(
            input_features=input_features.float(),
            attention_mask=input_speech_attn,
            generate_speech=False,
            tgt_lang=TGT_TEXT,
            **_hf_gen_kwargs(gen_common),
        )
    hf_secs = time.perf_counter() - t_ref0
    hf_ids = (hf_out.sequences[0] if hasattr(hf_out, "sequences") else hf_out[0]).cpu().long().tolist()
    hf_text = tokenizer.batch_decode([hf_ids], skip_special_tokens=True)[0]
    print(f"  HF reference ({len(hf_ids)} tokens, {hf_secs:.1f}s CPU): {hf_text}")

    # ---- Open device + build TT model ----
    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        original_default = None
    # SEAMLESS_FORCE_1x1=1 opens a single Blackhole chip (MeshShape(1,1), no fabric) directly,
    # bypassing the >=4-device BH-QB path. Use this when the 1x4 FABRIC_1D ethernet handshake
    # is unhealthy, or to measure single-chip (TP=1) latency. Otherwise auto-select per host.
    if os.environ.get("SEAMLESS_FORCE_1x1") == "1":
        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*MESH_SHAPE_P150),
            **dict(DEVICE_PARAMS_P150_E2E_2CQ_GENERATE),
        )
        mesh_shape = MESH_SHAPE_P150
    else:
        device, mesh_shape = open_seamless_mesh_device(enable_decode_trace=True, enable_2cq=True)
    ttnn.SetDefaultDevice(device)
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    print(f"\n  Device: MeshShape({rows}, {cols}) — TP={rows * cols} — decode: trace+2CQ")

    tt_model = None
    try:
        tt_model = make_tt_model(device, model, cfg, t2u_cfg)

        # Warm the speech-encoder program cache for this mel length, then clear so the
        # decode path JITs cleanly (mirrors the full demo's _warm_speech_enc).
        tt_model.prewarm_speech_encoder([feat_seq])
        tt_model.clear_runtime_program_cache()
        ttnn.synchronize_device(device)

        # Wrap _encode_speech to split encoder time out of the total generate time.
        enc_times: list[float] = []
        _orig_encode = tt_model._encode_speech

        def _timed_encode(*a, **k):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            r = _orig_encode(*a, **k)
            ttnn.synchronize_device(device)
            enc_times.append(time.perf_counter() - t0)
            return r

        tt_model._encode_speech = _timed_encode  # type: ignore[assignment]

        feats_tt = torch_feats_to_ttnn(device, input_features)
        attn_tt = torch_ids_to_ttnn(device, input_speech_attn)

        def _run():
            return tt_model.generate(
                input_features=feats_tt,
                attention_mask=attn_tt,
                generate_speech=False,
                tgt_lang=TGT_TEXT,
                **gen_common,
            )

        # Warmup (untimed)
        for _ in range(_WARMUP_ITERS):
            warm = _run()
            ttnn.synchronize_device(device)
            ttnn.deallocate(warm.sequences)

        # Timed iters: record (total, encoder) per iter; keep the iter with min total.
        best = None  # (total_s, encoder_s, out)
        for _ in range(_MEASURE_ITERS):
            enc_times.clear()
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = _run()
            ttnn.synchronize_device(device)
            total_s = time.perf_counter() - t0
            enc_s = enc_times[-1] if enc_times else 0.0
            # Keep only the fastest iter's output alive; free every other (incl. the prior best)
            # so we never deallocate the tensor we later read back.
            if best is None or total_s < best[0]:
                if best is not None:
                    ttnn.deallocate(best[2].sequences)
                best = (total_s, enc_s, out)
            else:
                ttnn.deallocate(out.sequences)

        total_s, enc_s, out = best
        if not isinstance(out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"S2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(out)}")

        tt_ids = to_torch_replicated_first_shard(out.sequences).long().reshape(-1).tolist()
        tt_text = _decode(tokenizer, out.sequences)
        n_new = _text_tokens_generated(out.sequences)

        # ---- Correctness: token-prefix match + text equality vs HF ----
        lcp = 0
        for a, b in zip(hf_ids, tt_ids):
            if a != b:
                break
            lcp += 1
        full_match = tt_ids == hf_ids
        last_id = tt_ids[-1] if tt_ids else -1
        if last_id == eos_token_id:
            stop = f"EOS({eos_token_id})"
        elif n_new >= max_new_tokens:
            stop = f"max_new_tokens={max_new_tokens}"
        else:
            stop = "ended"

        # ---- Latency breakdown ----
        dec_s = max(0.0, total_s - enc_s)
        ms_per_tok = (total_s * 1e3 / n_new) if n_new else 0.0
        tps = (n_new / total_s) if total_s else 0.0
        dec_ms_per_tok = (dec_s * 1e3 / n_new) if n_new else 0.0

        print("\n" + "-" * 78)
        print("  CORRECTNESS (TT greedy vs CPU HF reference)")
        print("-" * 78)
        print(f"  HF text : {hf_text}")
        print(f"  TT text : {tt_text}")
        print(f"  token-id prefix match : {lcp}/{len(hf_ids)} (seed 2 + {max(0, lcp - 2)} content)")
        print(f"  full token-id match   : {full_match}")
        print(f"  decoded text match    : {tt_text.strip() == hf_text.strip()}")
        print(f"  TT new tokens         : {n_new} (budget {max_new_tokens}, stopped at {stop})")

        print("\n" + "-" * 78)
        print(f"  LATENCY (min over {_MEASURE_ITERS} timed iters, {_WARMUP_ITERS} warmup; trace+2CQ)")
        print("-" * 78)
        print(f"  total generate        : {total_s * 1e3:10.1f} ms")
        print(f"   ├─ speech encoder     : {enc_s * 1e3:10.1f} ms  ({100 * enc_s / total_s:4.1f}%)")
        print(f"   └─ decoder loop       : {dec_s * 1e3:10.1f} ms  ({100 * dec_s / total_s:4.1f}%)   ({n_new} tokens)")
        print(f"  throughput            : {tps:10.2f} tokens/s")
        print(f"  per-token (total)     : {ms_per_tok:10.1f} ms/tok")
        print(f"  per-token (decode-only): {dec_ms_per_tok:9.1f} ms/tok")
        print("-" * 78)
        print(
            "  Note: 'speech encoder' is the one-shot prefill over all mel frames; the decoder\n"
            "  loop is amortized over the generated tokens. S2TT is speech-encoder dominated when\n"
            "  the encoder slice is a large fraction of the total."
        )

        ttnn.deallocate(out.sequences)
        tt_model.release_generation_runtime()
    finally:
        if tt_model is not None:
            try:
                tt_model.release_generation_runtime()
            except Exception:
                pass
        if original_default is not None:
            ttnn.SetDefaultDevice(original_default)
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
