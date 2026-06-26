# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized (trace + dual command queue) OpenAI-compatible TTS server for Qwen3-TTS on a
SINGLE Tenstorrent chip (P150a-equivalent / one P300 chip).

Uses the trace-based runtime (tt.server.init_server_context + run_inference): all kernels
compiled and ~41 traces captured ONCE at startup, then each request replays traces with H2D
on CQ1 overlapping compute on CQ0. Measured RTF ~0.59 steady-state on one P300 chip (vs
~1.3-1.5 for the eager TTSGenerator path) — i.e. faster than real-time.

Same OpenAI Audio Speech contract as qwen3_tts_server.py:
  POST /v1/audio/speech  {"model","input","voice","response_format","speed","language"}
Voices are zero-shot clones built from reference audio (ICL: ref_codes + speaker embedding +
ref_text) via VOICE_<NAME>_REF_AUDIO / VOICE_<NAME>_REF_TEXT env (tt-home: jim=GUEST,
riata=HOST). wav/pcm/flac native (soundfile); mp3/opus/aac and `speed` via ffmpeg.

Device is opened single-chip with num_command_queues=2 (unchanged P150a-equiv config).
"""
import asyncio
import io
import os
import re
import struct
import subprocess
import threading
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

MODEL_ID = os.environ.get("QWEN3TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEFAULT_LANGUAGE = os.environ.get("QWEN3TTS_LANGUAGE", "japanese")
MAX_NEW_TOKENS = int(os.environ.get("QWEN3TTS_MAX_NEW_TOKENS", "1024"))
DEFAULT_VOICE = os.environ.get("QWEN3TTS_DEFAULT_VOICE", "jim")
SR = 24000  # Qwen3-TTS output sample rate

# format -> (ffmpeg container/muxer, ffmpeg codec args, HTTP content-type)
_FFMPEG_FORMATS = {
    "mp3": ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"], "audio/mpeg"),
    "aac": ("adts", ["-c:a", "aac", "-b:a", "128k"], "audio/aac"),
    "opus": ("ogg", ["-c:a", "libopus", "-b:a", "96k"], "audio/ogg"),
    "flac": ("flac", ["-c:a", "flac"], "audio/flac"),
    "wav": ("wav", ["-c:a", "pcm_s16le"], "audio/wav"),
    "pcm": ("s16le", ["-c:a", "pcm_s16le"], "audio/L16"),
}
_SOUNDFILE_FORMATS = {"wav": ("WAV", "audio/wav"), "flac": ("FLAC", "audio/flac")}

app = FastAPI(title="Qwen3-TTS fast (trace+2cq) OpenAI-compatible server")
STATE = {}
_LOCK = threading.Lock()


class SpeechRequest(BaseModel, extra="allow"):
    input: str
    model: str = "qwen3-tts"
    voice: str = "default"
    response_format: str = "wav"  # tt-home contract default; OpenAI clients may send mp3 etc.
    speed: float = 1.0
    language: str | None = None  # Qwen3-TTS language (default japanese)
    stream: bool = False  # progressive (per-sentence) audio streaming -> lower time-to-first-audio


def _voice_specs_from_env():
    """name -> (ref_audio_path, ref_text) from VOICE_<NAME>_REF_AUDIO (+ optional _REF_TEXT)."""
    specs = {}
    for key, val in os.environ.items():
        if key.startswith("VOICE_") and key.endswith("_REF_AUDIO") and val and os.path.exists(val):
            name = key[len("VOICE_") : -len("_REF_AUDIO")].lower()
            txt_path = os.environ.get(f"VOICE_{name.upper()}_REF_TEXT", "")
            ref_text = ""
            if txt_path and os.path.exists(txt_path):
                with open(txt_path) as fh:
                    ref_text = fh.read().strip()
            specs[name] = (val, ref_text)
    return specs


def _resolve_default(voices):
    return DEFAULT_VOICE if DEFAULT_VOICE in voices else (sorted(voices)[0] if voices else None)


@app.on_event("startup")
def _load():
    import ttnn
    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
    from models.demos.qwen3_tts.tt.server import (
        TTSConfig,
        create_icl_embedding_ttnn,
        decode_audio,
        encode_reference_audio,
        init_server_context,
        load_weights,
        run_inference,
    )

    t0 = time.time()
    main_weights, decoder_weights = load_weights()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Single chip, dual command queue (H2D on CQ1 overlaps trace exec on CQ0).
    dev = ttnn.open_device(
        device_id=0, l1_small_size=32768, trace_region_size=200000000, num_command_queues=2
    )
    dev.enable_program_cache()

    model = Qwen3TTS(device=dev, state_dict=main_weights)
    cfg = TTSConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS

    # Build zero-shot voices (ICL): reference codes + speaker embedding + reference text.
    voices = {}
    for name, (ref_audio, ref_text) in _voice_specs_from_env().items():
        try:
            cache = os.path.splitext(ref_audio)[0] + ".refcache.pt"
            ref_codes, audio_data = encode_reference_audio(ref_audio, main_weights, cache_path=cache)
            spk = model.extract_speaker_embedding(audio_data)
            voices[name] = {"ref_codes": ref_codes, "spk": spk, "ref_text": ref_text}
            print(f"[fast] voice '{name}' ready (ref_text={'set' if ref_text else 'empty'})", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad ref shouldn't sink startup
            print(f"[fast] voice '{name}' failed: {e}", flush=True)
    if not voices:
        raise RuntimeError("no reference voices configured — set VOICE_<NAME>_REF_AUDIO")

    # Capture all traces / warm all kernels once (the slow, one-time step).
    ctx = init_server_context(dev, model, cfg, main_weights)

    STATE.update(
        dev=dev, model=model, cfg=cfg, ctx=ctx, tok=tokenizer, voices=voices,
        mw=main_weights, dw=decoder_weights,
        icl=create_icl_embedding_ttnn, run=run_inference, dec=decode_audio,
    )
    print(
        f"[fast] ready in {time.time()-t0:.1f}s; voices={['default']+sorted(voices)}; "
        f"default->{_resolve_default(voices)}; single chip + 2cq",
        flush=True,
    )


def _synth(text, voice, language):
    """Trace-replay synth under the device lock. Returns (waveform float32 mono, sr)."""
    name = voice if voice in STATE["voices"] else _resolve_default(STATE["voices"])
    v = STATE["voices"][name]
    inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _cpe = STATE["icl"](
        target_text=text,
        ref_text=v["ref_text"],
        ref_codes=v["ref_codes"],
        speaker_embedding=v["spk"],
        tokenizer=STATE["tok"],
        model=STATE["model"],
        device=STATE["dev"],
        config=STATE["cfg"],
        main_weights=STATE["mw"],
        language=language or DEFAULT_LANGUAGE,
    )
    codes, _timings, _ = STATE["run"](
        ctx=STATE["ctx"],
        model=STATE["model"],
        device=STATE["dev"],
        inputs_embeds_tt=inputs_embeds_tt,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        config=STATE["cfg"],
        use_2cq=True,
    )
    audio = STATE["dec"](codes, STATE["dw"])
    wav = audio.squeeze().detach().cpu().float().numpy().astype("float32").reshape(-1)
    return wav, SR


def _atempo_chain(speed):
    """ffmpeg atempo handles 0.5..2.0 per stage; chain for the OpenAI 0.25..4.0 range."""
    speed = float(np.clip(speed, 0.25, 4.0))
    factors = []
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    factors.append(speed)
    return ",".join(f"atempo={f:.4f}" for f in factors)


def _ffmpeg_encode(wav, sr, fmt, speed):
    container, codec_args, media = _FFMPEG_FORMATS[fmt]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0"]
    if abs(speed - 1.0) > 1e-3:
        cmd += ["-af", _atempo_chain(speed)]
    cmd += codec_args + ["-f", container, "pipe:1"]
    proc = subprocess.run(cmd, input=wav.astype("<f4").tobytes(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {fmt}: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout, media


def _encode(wav, sr, fmt, speed):
    fmt = (fmt or "wav").lower()
    fast = abs(speed - 1.0) <= 1e-3
    if fast and fmt in _SOUNDFILE_FORMATS:
        sub, media = _SOUNDFILE_FORMATS[fmt]
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format=sub)
        return buf.getvalue(), media
    if fast and fmt == "pcm":
        return (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes(), "audio/L16"
    if fmt in _FFMPEG_FORMATS:
        return _ffmpeg_encode(wav, sr, fmt, speed)
    raise ValueError(f"unsupported response_format: {fmt}")


# --- segment-level streaming (progressive playback -> lower time-to-first-audio) ---
_SENT_SPLIT = re.compile(r"(?<=[。．！？!?\n])")


def _segment_text(text, max_chars=48):
    """Split into sentence-ish chunks (JA/Latin enders) packed to <=max_chars, so the first
    chunk's audio streams out while later chunks still synthesize."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


def _wav_stream_header(sr, ch=1, bits=16):
    """Streaming WAV header (unknown length: sizes 0xFFFFFFFF); raw PCM16 frames follow."""
    byte_rate = sr * ch * bits // 8
    block_align = ch * bits // 8
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, byte_rate, block_align, bits)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def _pcm16(wav):
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


@app.get("/health")
def health():
    return {"status": "ok" if STATE else "loading"}


@app.get("/v1/audio/voices")
def voices():
    return {"voices": ["default"] + sorted(STATE.get("voices", {}))}


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if not req.input or not req.input.strip():
        return JSONResponse({"error": "input text is required"}, status_code=400)
    fmt = (req.response_format or "wav").lower()
    if fmt not in _FFMPEG_FORMATS:
        return JSONResponse({"error": f"unsupported response_format: {fmt}"}, status_code=400)
    loop = asyncio.get_event_loop()

    def _synth_locked(text):
        with _LOCK:
            return _synth(text, req.voice, req.language)

    # Streaming: synth per sentence-chunk and emit audio as each chunk completes, so the
    # client hears the first sentence after only the first chunk's synth (low TTFB) instead
    # of waiting for the whole utterance. Best with wav/pcm; mp3/opus/aac stream per-chunk.
    if req.stream:
        media = _FFMPEG_FORMATS[fmt][2]
        segs = _segment_text(req.input)
        native = abs(req.speed - 1.0) <= 1e-3 and fmt in ("wav", "pcm")

        async def gen():
            if fmt == "wav":
                yield _wav_stream_header(SR)
            for seg in segs:
                wav, sr = await loop.run_in_executor(None, _synth_locked, seg)
                if native:
                    yield _pcm16(wav)
                else:
                    body, _ = await loop.run_in_executor(None, _ffmpeg_encode, wav, sr, fmt, req.speed)
                    yield body

        return StreamingResponse(gen(), media_type=media, headers={"X-Stream-Segments": str(len(segs))})

    # Non-streaming: whole utterance in one response.
    def run():
        with _LOCK:
            t0 = time.time()
            wav, sr = _synth(req.input, req.voice, req.language)
            dt = time.time() - t0
        body, media = _encode(wav, sr, fmt, req.speed)
        dur = len(wav) / sr
        return body, media, dt / max(dur, 1e-3), dur

    try:
        body, media, rtf, dur = await loop.run_in_executor(None, run)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)
    return Response(content=body, media_type=media,
                    headers={"X-Audio-Duration": f"{dur:.2f}", "X-RTF": f"{rtf:.3f}"})
