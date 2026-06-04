# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Seamless M4T v2 — TTNN demo of all five inference tasks.

Tasks demonstrated (every one runs through ``TTSeamlessM4Tv2Model.generate``):

  1. **T2TT** Text-to-Text Translation        (English text → Hindi text)
  2. **T2ST** Text-to-Speech Translation      (English text → Hindi speech)
  3. **S2TT** Speech-to-Text Translation      (Hindi speech → English text)
  4. **S2ST** Speech-to-Speech Translation    (Hindi speech → Spanish speech)
  5. **ASR**  Automatic Speech Recognition    (Hindi speech → Hindi text)

The Hindi speech used as the input to tasks 3–5 is produced by task 2, so the demo is fully
self-contained — no external audio files needed. Output audio is written next to this file:

  * ``outputs/t2st_hindi_speech.wav``  — task 2 output (re-used as input for tasks 3-5)
  * ``outputs/s2st_spanish_speech.wav`` — task 4 output

Run from repo root:

  python models/experimental/seamless_m4t_v2_large/demo/demo.py

Optional: ``SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large`` if not using the default tree.
"""

from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.common import (
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
T2ST_WAV = OUTPUT_DIR / "t2st_hindi_speech.wav"
# The speech encoder uses a DRAM residual/LN path above ``_LONG_AUDIO_RES_DRAM_THRESHOLD``
# (1024 mel frames ≈ 20 s) and falls back to uncached relative-position tables above 32 MB —
# both unlock the full ~43 s T2ST audio for the chain tasks (matches HF semantics, no trim).
MAX_CHAIN_AUDIO_SEC: Optional[float] = None
S2ST_WAV = OUTPUT_DIR / "s2st_spanish_speech.wav"

# Per-task warmup + measurement iteration counts.
#
# Background: the first ``generate()`` for a fresh task hits trace capture + program-cache compile
# + ``_conv1d_prepared_cache`` build, which is non-trivial and makes the single-call wall-clock
# vary by tens of percent run-to-run. qwen3-tts solves this by timing each AR step separately and
# averaging steps[1:] (excluding the compile spike on step 0). Our analogous lever for a single
# ``generate()`` call would be one untimed warmup, but at ``gen_max_new=128`` the L1 state after a
# back-to-back replay sometimes fragments enough that a downstream ``slice`` op clashes with a
# static CB region (seen empirically on BH QB). The e2e perf test works around this by using
# ``gen_max_new=10``, but the demo's larger workload doesn't tolerate it cleanly.
#
# Default behaviour here: 0 in-task warmups (avoids the L1 clash) but ``_DEMO_MEASURE_ITERS`` is
# still a knob — bump it to read ``min(times)`` across N runs (qwen3's "steady" reading), which is
# the closest the host can get to the device-limited floor for a single ``generate()`` call.
_DEMO_WARMUP_ITERS = 0
_DEMO_MEASURE_ITERS = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights_dir() -> Path:
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


def torch_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def torch_feats_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def make_tt_model(device: ttnn.Device, model: torch.nn.Module, cfg, t2u_cfg) -> TTSeamlessM4Tv2Model:
    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    return TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        pad_token_id=cfg.pad_token_id,
        decoder_start_token_id=cfg.decoder_start_token_id,
        vocab_size=cfg.vocab_size,
        adaptor_kernel_size=cfg.adaptor_kernel_size,
        adaptor_stride=cfg.adaptor_stride,
        t2u_eos_token_id=cfg.t2u_eos_token_id,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        vocoder_offset=cfg.vocoder_offset,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
        generation_config=model.generation_config,
        hf_config=cfg,
    )


def _readback_first_shard(t: ttnn.Tensor) -> torch.Tensor:
    """Read replicated mesh tensor; delegate to ``to_torch_replicated_first_shard`` in ``tt/common.py``."""
    return to_torch_replicated_first_shard(t)


def _waveform_to_mono_fp32(waveform_tt: ttnn.Tensor, lengths_tt: ttnn.Tensor) -> np.ndarray:
    """Read a TT vocoder waveform back to host as a 1-D fp32 numpy array, trimmed to valid length.

    TT vocoder output shape: ``[B, T_max, 1]`` (right-padded with zeros to the batch max). The valid
    sample count per row is in ``lengths_tt`` — we trim to that to drop trailing silence padding.
    """
    arr = _readback_first_shard(waveform_tt).float().squeeze().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    valid_len = int(_readback_first_shard(lengths_tt).long().reshape(-1)[0].item())
    if 0 < valid_len <= arr.size:
        arr = arr[:valid_len]
    return arr


def _save_wav(path: Path, waveform_np: np.ndarray, sample_rate: int) -> None:
    """Save a mono fp32 waveform to ``path`` as a 16-bit PCM WAV (stdlib ``wave`` only)."""
    arr = np.clip(waveform_np, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _load_mono_wav(path: Path) -> np.ndarray:
    """Load a mono fp32 waveform from a PCM WAV (inverse of :func:`_save_wav`)."""
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sw} in {path}")
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return pcm.astype(np.float32)


_TT_ONLY_GEN_KEYS = frozenset({"use_kv_cache", "use_decode_trace", "use_2cq"})


def _hf_gen_kwargs(gen_common: dict) -> dict:
    """HF ``generate()`` kwargs — strip TT-only perf flags."""
    return {k: v for k, v in gen_common.items() if k not in _TT_ONLY_GEN_KEYS}


def _decode(tokenizer: Any, sequences_tt: ttnn.Tensor) -> str:
    """Read a TT decoder sequence back to host and decode to a single string (special tokens skipped)."""
    ids = _readback_first_shard(sequences_tt).to(torch.int64).cpu()
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def _tt_row_length(t: ttnn.Tensor) -> int:
    """Logical length of a 1-D or ``[1, L]`` int sequence on device."""
    return int(_readback_first_shard(t).long().reshape(-1).numel())


def _valid_unit_frames(unit_tt: ttnn.Tensor, *, pad_id: int) -> int:
    """Count non-pad unit ids in the vocoder input timeline."""
    u = _readback_first_shard(unit_tt).long().reshape(-1)
    return int((u != int(pad_id)).sum().item())


def _print_header(idx: int, name: str, abbrev: str, src: str, tgt: str) -> None:
    print()
    print("=" * 78)
    print(f"  {idx}. {abbrev}  —  {name}  ({src} → {tgt})")
    print("=" * 78)


def _time_generate(device: ttnn.Device, generate_fn):
    """Time a single ``tt_model.generate(...)`` call with explicit synchronize before/after.

    Returns ``(output, elapsed_seconds)``. The ``synchronize_device`` calls bracket *only* the
    model runtime — input tensors must already be uploaded to device, and any host post-processing
    (token decode, waveform readback, WAV write) must happen *outside* this window so it's
    excluded from the throughput metric.
    """
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    out = generate_fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - t0


def _warmup_and_time(
    device: ttnn.Device,
    generate_fn,
    release_fn,
    *,
    warmup_iters: int = _DEMO_WARMUP_ITERS,
    measure_iters: int = _DEMO_MEASURE_ITERS,
):
    """Run untimed warmup ``generate()`` calls, then return the last timed call's output + min
    elapsed across ``measure_iters`` runs.

    Why min over avg/median: with greedy decoding the workload is fixed (same prompt → same token
    counts → same trace replays), so any inter-run variance is host noise (GC, IO, dispatch
    jitter). The minimum is the closest the host gets to the device-limited steady-state floor.

    The final timed output is kept (caller post-processes it for text/WAV); warmup outputs and
    earlier timed outputs are released via ``release_fn``.
    """
    for _ in range(warmup_iters):
        warm_out = generate_fn()
        ttnn.synchronize_device(device)
        release_fn(warm_out)

    times = []
    out = None
    for _ in range(measure_iters):
        if out is not None:
            release_fn(out)
        out, elapsed = _time_generate(device, generate_fn)
        times.append(elapsed)
    return out, min(times) if times else 0.0


def _text_tokens_generated(sequences_tt: ttnn.Tensor, *, seed_len: int = 2) -> int:
    """Number of new tokens emitted by the decoder, excluding the 2-token seed
    ``[decoder_start, lang]`` that ``generate`` always prepends."""
    return max(0, _tt_row_length(sequences_tt) - seed_len)


def _samples_generated(lengths_tt: ttnn.Tensor) -> int:
    """Valid audio-sample count from the TT vocoder ``waveform_lengths`` tensor."""
    return int(_readback_first_shard(lengths_tt).long().reshape(-1)[0].item())


def _record_text_perf(
    perf: list,
    task: str,
    sequences_tt: ttnn.Tensor,
    elapsed_s: float,
    *,
    eos_token_id: int,
    max_new_tokens: int,
) -> None:
    n_tokens = _text_tokens_generated(sequences_tt)
    tps = n_tokens / elapsed_s if elapsed_s > 0 else 0.0
    perf.append((task, "tokens/s", tps, n_tokens, elapsed_s))
    ids = _readback_first_shard(sequences_tt).long().reshape(-1).tolist()
    last_id = int(ids[-1]) if ids else -1
    if last_id == int(eos_token_id):
        stop = f"EOS (id {eos_token_id})"
    elif n_tokens >= max_new_tokens:
        stop = f"max_new_tokens={max_new_tokens}"
    else:
        stop = "ended"
    print(
        f"  {task} runtime: {elapsed_s * 1000:.1f} ms  →  {tps:.2f} tokens/s "
        f"({n_tokens} new tokens, budget {max_new_tokens}, stopped at {stop})"
    )


def _record_speech_perf(perf: list, task: str, lengths_tt: ttnn.Tensor, elapsed_s: float) -> None:
    n_samples = _samples_generated(lengths_tt)
    sps = n_samples / elapsed_s if elapsed_s > 0 else 0.0
    perf.append((task, "samples/s", sps, n_samples, elapsed_s))
    print(f"  {task} runtime: {elapsed_s * 1000:.1f} ms  →  {sps:.2f} samples/s ({n_samples} audio samples)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    weights_dir = _weights_dir()
    path = os.fspath(weights_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    sample_rate = int(getattr(cfg, "sampling_rate", 16000))

    # ---- Single English prompt drives the entire demo chain ----
    src_text = """Maya lived in a small coastal town where every morning began with the sound of fishing boats leaving the harbor. She worked at her grandfather’s old bookstore, a narrow shop filled with dusty shelves, handwritten notes, and the smell of paper that had aged for decades. Most customers came looking for schoolbooks or travel guides, but Maya loved recommending forgotten stories hidden in the back corners of the store.

One rainy evening, while organizing a stack of returned books, she discovered a small blue journal tucked between two novels. The cover had no title, only a silver compass symbol that shimmered faintly under the light. Curious, she opened it and found detailed sketches of places around the town along with cryptic messages about a hidden lighthouse path that only appeared during storms.

At first, Maya thought someone was playing a prank. But the next night, as heavy clouds gathered over the sea, she noticed something unusual from the bookstore window. A narrow trail of lantern lights stretched along the cliffs where no road existed before. Holding the journal tightly, she followed the glowing path through the rain until she reached an abandoned lighthouse overlooking the crashing waves.

Inside the lighthouse, she found old maps, letters, and photographs belonging to sailors who had once protected ships during dangerous storms. Among the papers was a letter written by her grandfather many years earlier. He explained that the journal was meant for the next person curious enough to search for the truth hidden in ordinary places. Maya smiled as thunder echoed outside. For the first time, she realized the bookstore had never only been about selling books. It had always been about discovering stories waiting quietly for someone brave enough to follow them.
"""
    src_lang = "eng"
    tgt_translate = "hin"  # task 1, 2: translate eng → hin
    tgt_back_text = "eng"  # task 3: speech in hin → text in eng (back-translation)
    tgt_speech_other = "spa"  # task 4: speech in hin → speech in spa
    tgt_asr = "hin"  # task 5: transcribe the hin audio in hin

    text_inputs = processor(text=src_text, src_lang=src_lang, return_tensors="pt")
    input_ids = text_inputs["input_ids"]
    input_text_attn = text_inputs["attention_mask"]

    # KV-decode Metal trace (single-capture, valid for all decode positions).
    # Requires ``trace_region_size`` in device params — ``open_seamless_mesh_device`` with
    # ``enable_decode_trace=True`` sets ``trace_region_size=450_000_000`` automatically.
    use_decode_trace = True
    # 2CQ: CQ1 stages next-step H2D while CQ0 executes the trace.
    # Requires ``num_command_queues=2`` — set when ``enable_2cq=True`` in ``open_seamless_mesh_device``.
    use_2cq = True
    # HF ``GenerationConfig`` defaults for text decode; TT-only perf flags appended below.
    gen_common = hf_aligned_generation_kwargs(
        model.generation_config,
        use_kv_cache=True,
        use_decode_trace=use_decode_trace,
        use_2cq=use_2cq,
    )

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        original_default = None
    # 65536 B L1_SMALL is the value used by the speech-generation PCC tests
    # (``test_code_hifigan.py`` and ``test_seamless_m4t_v2_model.py``). 32768 B works for the
    # text-only path but is not enough for S2ST: speech-encoder + T2U + vocoder chained back-to-back
    # exhausts L1_SMALL before the vocoder's ``_resblock`` conv1d can allocate.
    #
    # Open a 1×N mesh over every visible P150 (N=1 on P150, N=4 on BH QB). All ``ttnn.from_torch``
    # uploads without an explicit mesh_mapper take the auto-replicate path (1×1 host tensor →
    # 1×N device mesh via ``h2d_as_replicate_tensor_on_1x1_mesh``), so weights, inputs, masks and
    # control tensors are replicated on every device. Every host readback inside ``tt/`` now
    # goes through ``to_torch_replicated_first_shard`` (in ``tt/common.py``), which uses
    # ``ConcatMeshToTensor(dim=0)`` + a leading slice to pull just one device's copy — without
    # that wiring ``ttnn.to_torch`` would TT_FATAL on the multi-shard replicated tensor.
    # All N devices run the same generate loop in lock-step; the demo's audio/text outputs are
    # the device-0 result.
    from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import open_seamless_mesh_device

    device, mesh_shape = open_seamless_mesh_device(
        enable_decode_trace=bool(gen_common.get("use_decode_trace")),
        enable_2cq=bool(gen_common.get("use_2cq")),
    )
    ttnn.SetDefaultDevice(device)
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    tp = rows * cols
    trace_info = "trace+2CQ" if (use_decode_trace and use_2cq) else ("trace" if use_decode_trace else "eager")
    print(f"  Demo device: MeshShape({rows}, {cols}) — TP={tp} — decode: {trace_info}")
    print(
        f"  HF-aligned greedy: max_new_tokens={gen_common['max_new_tokens']} "
        f"(cap), eos_token_id={gen_common['eos_token_id']}, "
        f"repetition_penalty={gen_common['repetition_penalty']}, "
        f"decode=trace+2CQ+ttnn_argmax"
    )

    tt_model = None
    try:
        tt_model = make_tt_model(device, model, cfg, t2u_cfg)
        # Per-task throughput log — populated by ``_record_text_perf`` / ``_record_speech_perf``
        # inside each task block, printed as a summary table at the end of the run.
        perf_log: list = []

        # =========================================================================
        # 1. T2TT — Text-to-Text Translation (English → Hindi)
        # =========================================================================
        _print_header(1, "Text-to-Text Translation", "T2TT", "eng", tgt_translate)
        print(f"  Input text  ({src_lang}): {src_text}")
        # Pre-upload inputs (preprocessing, not timed).
        t2tt_ids_tt = torch_ids_to_ttnn(device, input_ids)
        t2tt_attn_tt = torch_ids_to_ttnn(device, input_text_attn)
        t2tt_out, t2tt_elapsed = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_ids=t2tt_ids_tt,
                attention_mask=t2tt_attn_tt,
                generate_speech=False,
                tgt_lang=tgt_translate,
                **gen_common,
            ),
            release_fn=lambda o: ttnn.deallocate(o.sequences),
        )
        if not isinstance(t2tt_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"T2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(t2tt_out)}")
        _record_text_perf(
            perf_log,
            "T2TT",
            t2tt_out.sequences,
            t2tt_elapsed,
            eos_token_id=gen_common["eos_token_id"],
            max_new_tokens=gen_common["max_new_tokens"],
        )
        print(f"  Output text ({tgt_translate}): {_decode(tokenizer, t2tt_out.sequences)}")
        ttnn.deallocate(t2tt_out.sequences)

        # T2TT compiles many text-decoder programs; clear before T2ST so vocoder conv1d fits in L1.
        tt_model.clear_runtime_program_cache()
        ttnn.synchronize_device(device)

        # =========================================================================
        # 2. T2ST — Text-to-Speech Translation (English text → Hindi speech)
        # =========================================================================
        _print_header(2, "Text-to-Speech Translation", "T2ST", "eng", tgt_translate)
        print(f"  Input text  ({src_lang}): {src_text}")
        t2st_ids_tt = torch_ids_to_ttnn(device, input_ids)
        t2st_attn_tt = torch_ids_to_ttnn(device, input_text_attn)

        def _release_speech_out(o):
            ttnn.deallocate(o.waveform)
            ttnn.deallocate(o.waveform_lengths)
            if getattr(o, "sequences", None) is not None:
                ttnn.deallocate(o.sequences)
            if getattr(o, "unit_sequences", None) is not None:
                ttnn.deallocate(o.unit_sequences)

        t2st_out, t2st_elapsed = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_ids=t2st_ids_tt,
                attention_mask=t2st_attn_tt,
                generate_speech=True,
                return_intermediate_token_ids=True,
                tgt_lang=tgt_translate,
                speaker_id=0,
                **gen_common,
            ),
            release_fn=_release_speech_out,
        )
        if not isinstance(t2st_out, TTSeamlessM4Tv2GenerationOutput):
            raise TypeError(f"T2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(t2st_out)}")
        _record_speech_perf(perf_log, "T2ST", t2st_out.waveform_lengths, t2st_elapsed)
        print(f"  Intermediate text ({tgt_translate}): {_decode(tokenizer, t2st_out.sequences)}")
        hindi_wav_np = _waveform_to_mono_fp32(t2st_out.waveform, t2st_out.waveform_lengths)
        _save_wav(T2ST_WAV, hindi_wav_np, sample_rate=sample_rate)
        t2u_pad = int(t2u_cfg.pad_token_id)
        n_units = _valid_unit_frames(t2st_out.unit_sequences, pad_id=t2u_pad)
        print(
            f"  T2ST stats: text_tokens={_tt_row_length(t2st_out.sequences)}, "
            f"unit_frames={n_units}, "
            f"audio={hindi_wav_np.size} samples ({hindi_wav_np.size / sample_rate:.2f}s)"
        )
        print(f"  Saved to: {T2ST_WAV}")

        tt_model.clear_runtime_program_cache()
        ttnn.synchronize_device(device)

        # Tasks 3–5 consume the saved T2ST WAV (same path as ``compare_speech_asr_hf_tt.py``).
        # Reload from disk so the processor sees exactly what the printed path refers to, not a
        # stale in-memory buffer from the TT vocoder readback.
        hindi_wav_chain = _load_mono_wav(T2ST_WAV)
        if MAX_CHAIN_AUDIO_SEC is not None:
            max_chain_samples = int(sample_rate * MAX_CHAIN_AUDIO_SEC)
            if hindi_wav_chain.size > max_chain_samples:
                hindi_wav_chain = hindi_wav_chain[:max_chain_samples]
                print(
                    f"  Note: S2TT/S2ST/ASR use first {MAX_CHAIN_AUDIO_SEC:.0f}s of T2ST audio "
                    f"({max_chain_samples} samples)."
                )

        audio_inputs = processor(audios=hindi_wav_chain, sampling_rate=sample_rate, return_tensors="pt")
        input_features = audio_inputs["input_features"]
        input_speech_attn = audio_inputs["attention_mask"]
        mel_frames = int(input_speech_attn.sum().item())
        print(
            f"  Chain audio: {hindi_wav_chain.size} samples "
            f"({hindi_wav_chain.size / sample_rate:.2f}s), mel_frames={mel_frames}"
        )

        # Warm the speech-encoder JIT/disk cache for a given audio's mel-length bucket *before* its
        # timed task. The encoder kernels are shape-specialized, so the first encode of a bucket
        # otherwise pays a cold ~7-20 s recompile; after this the task rebuilds from the warm cache
        # (~1-2 s). Called per speech task so each task is self-contained — run a single task in
        # isolation and it warms its own bucket and reproduces the steady-state number; in this chained
        # run ``prewarm_speech_encoder`` is idempotent so the shared bucket is warmed only once. The
        # length is derived from the actual audio (not input-specific); only the live bucket is warmed
        # (a larger neighbour bucket compiles an oversized program that clashes with the decode CB
        # budget — see ``prewarm_speech_encoder``). The clear resets to a clean pre-task device state.
        def _warm_speech_enc(feats_torch: torch.Tensor) -> None:
            tt_model.prewarm_speech_encoder([int(feats_torch.shape[1])])
            tt_model.clear_runtime_program_cache()
            ttnn.synchronize_device(device)

        # =========================================================================
        # 3. S2TT — Speech-to-Text Translation (Hindi speech → English text)
        # =========================================================================
        _warm_speech_enc(input_features)
        _print_header(3, "Speech-to-Text Translation", "S2TT", tgt_translate, tgt_back_text)
        print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
        s2tt_feats_tt = torch_feats_to_ttnn(device, input_features)
        s2tt_attn_tt = torch_ids_to_ttnn(device, input_speech_attn)
        s2tt_out, s2tt_elapsed = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_features=s2tt_feats_tt,
                attention_mask=s2tt_attn_tt,
                generate_speech=False,
                tgt_lang=tgt_back_text,
                **gen_common,
            ),
            release_fn=lambda o: ttnn.deallocate(o.sequences),
        )
        if not isinstance(s2tt_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"S2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(s2tt_out)}")
        _record_text_perf(
            perf_log,
            "S2TT",
            s2tt_out.sequences,
            s2tt_elapsed,
            eos_token_id=gen_common["eos_token_id"],
            max_new_tokens=gen_common["max_new_tokens"],
        )
        print(f"  Output text ({tgt_back_text}): {_decode(tokenizer, s2tt_out.sequences)}")

        tt_model.clear_runtime_program_cache()
        ttnn.synchronize_device(device)

        # =========================================================================
        # 4. S2ST — Speech-to-Speech Translation (Hindi speech → Spanish speech)
        # =========================================================================
        _warm_speech_enc(input_features)
        _print_header(4, "Speech-to-Speech Translation", "S2ST", tgt_translate, tgt_speech_other)
        print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
        s2st_feats_tt = torch_feats_to_ttnn(device, input_features)
        s2st_attn_tt = torch_ids_to_ttnn(device, input_speech_attn)
        s2st_out, s2st_elapsed = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_features=s2st_feats_tt,
                attention_mask=s2st_attn_tt,
                generate_speech=True,
                return_intermediate_token_ids=True,
                tgt_lang=tgt_speech_other,
                speaker_id=0,
                **gen_common,
            ),
            release_fn=_release_speech_out,
        )
        if not isinstance(s2st_out, TTSeamlessM4Tv2GenerationOutput):
            raise TypeError(f"S2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(s2st_out)}")
        _record_speech_perf(perf_log, "S2ST", s2st_out.waveform_lengths, s2st_elapsed)
        print(f"  Intermediate text ({tgt_speech_other}): {_decode(tokenizer, s2st_out.sequences)}")
        spanish_wav_np = _waveform_to_mono_fp32(s2st_out.waveform, s2st_out.waveform_lengths)
        _save_wav(S2ST_WAV, spanish_wav_np, sample_rate=sample_rate)
        print(f"  Output audio ({tgt_speech_other}, {sample_rate} Hz, {spanish_wav_np.size} samples)")
        print(f"  Saved to: {S2ST_WAV}")

        tt_model.clear_runtime_program_cache()
        ttnn.synchronize_device(device)

        # =========================================================================
        # 5. ASR — Automatic Speech Recognition (Hindi speech → Hindi text)
        # =========================================================================
        # S2ST leaves speech-gen + decode-trace state that can skew the hin ASR decode if we
        # reuse the eng-target trace from S2TT without a full reset (see compare_speech_asr_hf_tt.py).
        tt_model.release_generation_runtime()
        _warm_speech_enc(input_features)
        _print_header(5, "Automatic Speech Recognition", "ASR", tgt_translate, tgt_asr)
        print(f"  Input audio ({tgt_translate}): {T2ST_WAV} ({sample_rate} Hz)")
        print("  Note: ASR transcribes the WAV (speech→text), not the T2ST intermediate text string.")
        with torch.no_grad():
            hf_asr_out = model.generate(
                input_features=input_features.float(),
                attention_mask=input_speech_attn,
                generate_speech=False,
                tgt_lang=tgt_asr,
                **_hf_gen_kwargs(gen_common),
            )
        hf_asr_ids = (
            hf_asr_out.sequences[0].cpu().tolist() if hasattr(hf_asr_out, "sequences") else hf_asr_out[0].cpu().tolist()
        )
        print(
            f"  HF reference ({tgt_asr}, {len(hf_asr_ids)} tokens): {tokenizer.batch_decode([hf_asr_ids], skip_special_tokens=True)[0]}"
        )
        asr_feats_tt = torch_feats_to_ttnn(device, input_features)
        asr_attn_tt = torch_ids_to_ttnn(device, input_speech_attn)
        asr_out, asr_elapsed = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_features=asr_feats_tt,
                attention_mask=asr_attn_tt,
                generate_speech=False,
                tgt_lang=tgt_asr,
                **gen_common,
            ),
            release_fn=lambda o: ttnn.deallocate(o.sequences),
        )
        if not isinstance(asr_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"ASR expected TTSeamlessM4Tv2GreedySearchOutput, got {type(asr_out)}")
        tt_asr_ids = _readback_first_shard(asr_out.sequences).long().reshape(-1).tolist()
        lcp = 0
        for a, b in zip(hf_asr_ids, tt_asr_ids):
            if a != b:
                break
            lcp += 1
        print(f"  HF/TT token prefix match: {lcp} (seed + {max(0, lcp - 2)} content tokens)")
        _record_text_perf(
            perf_log,
            "ASR",
            asr_out.sequences,
            asr_elapsed,
            eos_token_id=gen_common["eos_token_id"],
            max_new_tokens=gen_common["max_new_tokens"],
        )
        print(f"  TT output ({tgt_asr}): {_decode(tokenizer, asr_out.sequences)}")

        # Free trace + pinned host readback buffers before mesh teardown (avoids abort on exit).
        tt_model.release_generation_runtime()

        print()
        print("=" * 78)
        print("  ok — all five tasks completed")
        print("=" * 78)
        print(f"  Audio outputs saved under: {OUTPUT_DIR}")
        # Per-task throughput summary — wall-clock around ``tt_model.generate(...)`` only;
        # input upload, output readback, tokenizer decode, and WAV write are excluded.
        #
        # Two metrics per task:
        #   * Throughput  — tokens/s or samples/s (user-facing, but for tasks with EOS-dependent
        #                  output length this can vary run-to-run as bf16 noise shifts where the
        #                  decoder emits EOS — affects S2TT in particular).
        #   * Per-work-unit latency — ms per token (text out) or μs per sample (speech out).
        #                  Decouples the device cost from output-length variance; this is the
        #                  more stable number for cross-run comparisons.
        if perf_log:
            print()
            print("-" * 78)
            print("  TT model runtime summary (excludes pre/post-processing)")
            print("-" * 78)
            print(f"  {'Task':<6} {'Runtime':>11} {'Throughput':>22} {'Workload':>20} {'Per-unit':>14}")
            for task_name, unit, value, count, elapsed_s in perf_log:
                workload = f"{count} {'samples' if unit == 'samples/s' else 'tokens'}"
                if unit == "samples/s":
                    # μs/sample is more readable than ms/sample for 16 kHz audio.
                    per_unit = f"{(elapsed_s * 1e6 / count) if count else 0.0:.2f} μs/smp"
                else:
                    per_unit = f"{(elapsed_s * 1e3 / count) if count else 0.0:.1f} ms/tok"
                print(
                    f"  {task_name:<6} {elapsed_s * 1000:>9.1f} ms  "
                    f"{value:>15.2f} {unit:<6} {workload:>20} {per_unit:>14}"
                )
            print("-" * 78)
            print(
                "  Note: throughput depends on output length (variable for EOS-terminated tasks).\n"
                "  For run-to-run comparison the per-unit latency column is more stable."
            )

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
