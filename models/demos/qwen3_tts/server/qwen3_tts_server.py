# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible Text-to-Speech server for TT Qwen3-TTS-12Hz-1.7B (Blackhole P300).

Exposes the OpenAI Audio Speech contract so any OpenAI TTS client can synthesize speech on
the Tenstorrent device. Mirrors the Qwen3-ASR server's structure (single device behind one
lock, first-request warmup). The full pipeline (Talker + Code Predictor on device, Speaker
Encoder + Vocoder on host) is built once at startup.

Request : POST /v1/audio/speech  JSON
    {"model": <str>, "input": <text>, "voice": <str>,
     "response_format": "mp3"|"opus"|"aac"|"flac"|"wav"|"pcm",
     "speed": <0.25..4.0>, "language": <optional, default "japanese">}
Response: raw audio bytes (Content-Type per format), 24 kHz mono.

Run (P300 single chip — uses the p150 mesh-graph descriptor; tt-smi -r if eth-core init times out):
  docker run -d --device /dev/tenstorrent/0 -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v /home/ttuser/ttwork/tt-metal-qwen3-tts:/work \
    -v /home/ttuser/.cache/huggingface:/root/.cache/huggingface \
    -v /home/ttuser/ttwork/qwen3_tts_voices:/models/qwen3_tts_voices \
    -p 8003:8003 --cap-add ALL qwen3-tts-server:latest
"""
import asyncio
import glob
import io
import os
import subprocess
import threading
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

MODEL_PATH = os.environ.get("QWEN3TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICES_DIR = os.environ.get("QWEN3TTS_VOICES_DIR", "/models/qwen3_tts_voices")
DEFAULT_LANGUAGE = os.environ.get("QWEN3TTS_LANGUAGE", "japanese")
MAX_NEW_TOKENS = int(os.environ.get("QWEN3TTS_MAX_NEW_TOKENS", "2048"))
WARMUP_TEXT = os.environ.get("QWEN3TTS_WARMUP_TEXT", "こんにちは")
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

app = FastAPI(title="Qwen3-TTS OpenAI-compatible server")
STATE = {}
_LOCK = threading.Lock()
_WARMED = False


def _ref_voices_from_env():
    """Map of voice-name -> reference-audio path from VOICE_<NAME>_REF_AUDIO env vars.
    Matches the tt-home compose contract (VOICE_JIM_REF_AUDIO, VOICE_RIATA_REF_AUDIO).
    The companion VOICE_<NAME>_REF_TEXT is accepted but unused by the TTSGenerator voice-
    cloning path (timbre comes from the speaker embedding, not an ICL transcript)."""
    out = {}
    for key, val in os.environ.items():
        if key.startswith("VOICE_") and key.endswith("_REF_AUDIO") and val:
            name = key[len("VOICE_") : -len("_REF_AUDIO")].lower()
            if name and os.path.exists(val):
                out[name] = val
    return out


class SpeechRequest(BaseModel, extra="allow"):
    input: str
    model: str = "qwen3-tts"
    voice: str = "default"
    response_format: str = "mp3"  # OpenAI default
    speed: float = 1.0
    language: str | None = None  # extension: Qwen3-TTS language (default japanese)
    # TTS sampling controls (extensions; not in the OpenAI schema)
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0


@app.on_event("startup")
def _load():
    import ttnn
    from models.demos.qwen3_tts.tt.generator import TTSGenerator
    from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

    os.environ["HF_MODEL"] = MODEL_PATH  # generator.build reads HF_MODEL internally
    t0 = time.time()
    device_ids = ttnn.get_device_ids()
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(device_ids)),
        dispatch_core_config=ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.ETH if len(device_ids) > 1 else ttnn.DispatchCoreType.WORKER
        ),
    )
    try:
        mesh_device.enable_program_cache()
    except AttributeError:
        ttnn.enable_program_cache(mesh_device)

    generator = TTSGenerator.build(MODEL_PATH, mesh_device, max_seq_len=MAX_NEW_TOKENS + 512)

    # Two voice sources (the tt-home contract + a generic fallback), merged into one dict:
    #   1. Reference-audio voices via env VOICE_<NAME>_REF_AUDIO (tt-home: jim=GUEST, riata=HOST).
    #      Built in-memory by the speaker encoder from a reference WAV.
    #   2. Pre-saved <name>.safetensors embeddings from VOICES_DIR (generic; made with
    #      demo_tts.py --ref_audio <ref.wav> --save_speaker <name>.safetensors).
    voices = {}
    for name, ref_audio in _ref_voices_from_env().items():
        try:
            wav, sr = sf.read(ref_audio, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 24000:  # speaker_encoder.encode requires 24 kHz
                import librosa

                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
            voices[name] = generator.speaker_encoder.encode(wav.astype("float32"), 24000)
            print(f"[server] voice '{name}' from reference audio {ref_audio}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad ref shouldn't sink startup
            print(f"[server] failed to build voice '{name}' from {ref_audio}: {e}", flush=True)
    if os.path.isdir(VOICES_DIR):
        for path in sorted(glob.glob(os.path.join(VOICES_DIR, "*.safetensors"))):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                voices[name] = SpeakerEncoder.load_embedding(path, mesh_device)
            except Exception as e:  # noqa: BLE001
                print(f"[server] failed to load voice '{name}': {e}", flush=True)
    STATE.update(dev=mesh_device, gen=generator, voices=voices)
    print(
        f"[server] loaded in {time.time()-t0:.1f}s; voices={['default']+sorted(voices)}; " f"ready on P300",
        flush=True,
    )


def _ensure_warm():
    """Compile kernels on the FIRST request (request context), like the ASR server: warming
    inside @startup corrupted that pipeline. One short synth compiles the common shapes so
    the first real request isn't a cold-JIT burst."""
    global _WARMED
    if _WARMED:
        return
    _WARMED = True
    try:
        STATE["gen"].generate(text=WARMUP_TEXT, language=DEFAULT_LANGUAGE, max_new_tokens=64)
    except Exception as e:  # noqa: BLE001
        print(f"[server] warmup skipped: {e}", flush=True)


def _synth(text, voice, language, max_new_tokens, temperature, top_k, top_p):
    """Run the TT pipeline under the device lock. Returns (waveform float32 mono, sr)."""
    speaker_emb_tt = STATE["voices"].get(voice)  # None for "default"/unknown -> model default
    wav, sr = STATE["gen"].generate(
        text=text,
        language=language or DEFAULT_LANGUAGE,
        speaker_emb_tt=speaker_emb_tt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return wav, sr


def _atempo_chain(speed):
    """ffmpeg atempo handles 0.5..2.0 per stage; chain stages for the OpenAI 0.25..4.0 range."""
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
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0"]
    if abs(speed - 1.0) > 1e-3:
        cmd += ["-af", _atempo_chain(speed)]
    cmd += codec_args + ["-f", container, "pipe:1"]
    proc = subprocess.run(cmd, input=wav.astype("<f4").tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {fmt}: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout, media


def _encode(wav, sr, fmt, speed):
    """Encode float32 mono waveform to the requested OpenAI response_format."""
    fmt = (fmt or "mp3").lower()
    fast = abs(speed - 1.0) <= 1e-3  # no tempo change -> use the dependency-free fast path
    if fast and fmt in _SOUNDFILE_FORMATS:
        sub, media = _SOUNDFILE_FORMATS[fmt]
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format=sub)
        return buf.getvalue(), media
    if fast and fmt == "pcm":  # OpenAI pcm = 24kHz signed 16-bit LE mono
        return (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes(), "audio/L16"
    if fmt in _FFMPEG_FORMATS:
        return _ffmpeg_encode(wav, sr, fmt, speed)
    raise ValueError(f"unsupported response_format: {fmt}")


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
    fmt = (req.response_format or "mp3").lower()
    if fmt not in _FFMPEG_FORMATS:
        return JSONResponse({"error": f"unsupported response_format: {fmt}"}, status_code=400)
    loop = asyncio.get_event_loop()

    def run():
        with _LOCK:
            _ensure_warm()
            t0 = time.time()
            wav, sr = _synth(req.input, req.voice, req.language, MAX_NEW_TOKENS, req.temperature, req.top_k, req.top_p)
            dt = time.time() - t0
        body, media = _encode(wav, sr, fmt, req.speed)
        rtf = dt / max(len(wav) / sr, 1e-3)
        return body, media, rtf, len(wav) / sr

    try:
        body, media, rtf, dur = await loop.run_in_executor(None, run)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)
    return Response(content=body, media_type=media, headers={"X-Audio-Duration": f"{dur:.2f}", "X-RTF": f"{rtf:.3f}"})
