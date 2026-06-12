"""OpenAI-compatible ASR server for TT Qwen3-ASR-1.7B (same /v1/audio/transcriptions
contract as the tt-media-server whisper endpoint, so the qwen3-asr-eval pipeline can
target it transparently via --asr-api).

Request : multipart  file=<wav>, model=<str>, language=<optional>
Response: {"text": ..., "language": ..., "duration": ...}

Auto-detects language when `language` is absent/auto (Qwen3-ASR is multilingual).

Run in the dev container (chip 3 = fake P150):
  docker exec -e TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
    -e HF_MODEL=/ttwork/qwen3_asr_text_decoder qwen3asr-dev bash -lc \
    'source /opt/venv/bin/activate && cd /work && \
     uvicorn models.demos.audio.qwen3_asr.server.qwen3_asr_server:app --host 0.0.0.0 --port 8002'
"""
import asyncio, io, os, re, sys, threading, time
import numpy as np
import soundfile as sf
import torch
import ttnn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from safetensors import safe_open
from transformers import AutoTokenizer
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from models.tt_transformers.tt.model_config import ModelArgs

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "tt"))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import audio_encoder as tt_enc                # noqa: E402
import audio_encoder_ref as ref               # noqa: E402  (weights loader only)
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa: E402

CKPT = os.environ.get("HF_MODEL", "/ttwork/qwen3_asr_text_decoder")
SNAP_BASE = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"
AUDIO_TOKEN_ID = 151676
_LOCK = threading.Lock()

app = FastAPI()
STATE = {}


def _build_prompt(processor, force_language):
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}<asr_text>"
    return base


def _parse(text):
    m = re.search(r"language\s*(.*?)<asr_text>(.*)", text, flags=re.DOTALL)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", text.strip())


@app.on_event("startup")
def _load():
    snap = os.path.join(SNAP_BASE, os.listdir(SNAP_BASE)[0])
    t0 = time.time()
    dev = ttnn.open_device(device_id=0, trace_region_size=200000000, l1_small_size=65536)
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
    enc_params = tt_enc.preprocess_weights(w, dev)
    args = ModelArgs(dev, max_batch_size=1, max_seq_len=2048)
    sd = args.load_state_dict()
    model = Qwen3ASRDecoder(args, ttnn.bfloat16, dev, sd,
                            args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False)
    with safe_open(os.path.join(CKPT, "model.safetensors"), "pt") as h:
        embed = h.get_tensor("model.embed_tokens.weight").float()
    STATE.update(dev=dev, enc=enc_params, model=model, embed=embed,
                 tok=AutoTokenizer.from_pretrained(CKPT),
                 proc=Qwen3ASRProcessor.from_pretrained(snap, fix_mistral_regex=True))
    print(f"[server] loaded in {time.time()-t0:.1f}s; ready on chip 3 (P150)", flush=True)


_WARMED = False


def _ensure_warm():
    """Lazily warm the common prefill shapes on the FIRST request (request context).
    NOTE: running this in the @startup handler instead persistently corrupted the model
    (every later request returned empty) — warming in request context is safe. Uses a
    real-speech wav (QWEN3ASR_WARMUP_WAV); skips if absent."""
    global _WARMED
    if _WARMED:
        return
    _WARMED = True
    wpath = os.environ.get("QWEN3ASR_WARMUP_WAV", "/warmup.wav")
    if not os.path.exists(wpath):
        return
    tw = time.time()
    ww, _ = sf.read(wpath, dtype="float32")
    if ww.ndim > 1:
        ww = ww.mean(1)
    # Now that EVERY _infer runs at the fixed FIXED_INFER_SEC length, warmup is safe (single
    # prefill shape — the earlier warmup corruption came from VARYING lengths) and it compiles
    # the exact kernel all requests use, so the first real request / stream segment is warm
    # (no cold-JIT burst). Two passes to also compile the decode loop.
    try:
        _infer(ww, None)
        _infer(ww, None)
    except Exception as e:
        print(f"[server] warmup skipped: {e}", flush=True)
    print(f"[server] warm in {time.time()-tw:.1f}s", flush=True)


FIXED_INFER_SEC = 14.0  # every _infer runs at EXACTLY this audio length (see below)


def _infer(wav, force, max_new_tokens=200):
    """Run the full TT pipeline on a 16k mono float32 waveform. Returns (text, lang, secs).

    ROOT-CAUSE FIX (variable-prefill-length corruption): in the long-lived encoder+decoder
    pipeline, mixing requests of DIFFERENT prefill lengths corrupts the decoder — it "locks"
    to the first request's length and emits only the language tag + EOS (empty transcript) for
    other lengths (encoder-alone and decoder-alone are each stable; the interleaving with a
    varying real sequence length is the trigger — a length-keyed program/state issue in
    tt-metal). Pinning every request to ONE fixed audio length (pad short with silence, cap
    long) makes the prefill length constant -> all requests transcribe fully. Verified: varied
    14s clips and 6s-padded-to-14s clips all transcribe; the old 6s/30s mix truncated."""
    t0 = time.time()
    n = int(FIXED_INFER_SEC * SR)
    wav = wav[:n] if len(wav) >= n else np.concatenate([wav, np.zeros(n - len(wav), dtype=np.float32)])
    prompt = _build_prompt(STATE["proc"], force)
    inputs = STATE["proc"](text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"][0].long()
    mel = inputs["input_features"][0].float() if inputs["input_features"].dim() == 3 \
        else inputs["input_features"].float()
    audio_embeds = tt_enc.encode_mel(mel, STATE["enc"], STATE["dev"]).float()
    inp = STATE["embed"][input_ids].clone()
    mask = (input_ids == AUDIO_TOKEN_ID)
    n_mask = int(mask.sum())
    # Encoder output length should equal the audio-token count; it is >= it by construction
    # (last partial chunk padded to 13 then masked, matching the reference). Guard BOTH
    # directions so a future mismatch degrades instead of 500-ing.
    if audio_embeds.shape[0] > n_mask:
        audio_embeds = audio_embeds[:n_mask]
    elif audio_embeds.shape[0] < n_mask:
        pad = torch.zeros(n_mask - audio_embeds.shape[0], audio_embeds.shape[1])
        audio_embeds = torch.cat([audio_embeds, pad], 0)
    inp[mask] = audio_embeds
    # use_trace=False: the persistent decode trace gives ~2x but HANGS the long-lived server
    # after a few mixed-shape requests (tt-metal trace-stability issue; see qwen3_asr_decoder).
    # Decode speed instead comes from on-device argmax (no 151936-vocab host transfer/token).
    ids = STATE["model"].generate(inp.unsqueeze(0), max_new_tokens=max_new_tokens, use_trace=False)
    lang, text = _parse(STATE["tok"].decode(ids, skip_special_tokens=False))
    return text, lang, time.time() - t0


# --- Tier-2 long-form: silence-aware chunking + hallucination/non-speech gating ---
SR = 16000
# CRITICAL: keep EVERY segment <= ~32s so its prefill pads to exactly 512 tokens (one MLP
# reshape bucket, S//512==1). The decoder MLP reshapes prefill x to [1, S_pad//512, 512, -1];
# 512-pad (bucket 1) and 1024-pad (bucket 2, for >~37s) take different program shapes, and
# MIXING buckets across requests in the long-lived server corrupts later requests (they return
# empty) — the same failure class as the old 256-vs-512 mix. Capping all segments to one bucket
# (512) makes the server stable. Longer audio is chunked; nothing is single-shot above the cap.
SINGLE_CAP = FIXED_INFER_SEC   # <= this -> single-shot; longer -> chunk. Segments are <=FIXED_INFER_SEC
# so _infer's fixed-length pin never drops real audio. Constant prefill length across all requests
# is what keeps the pipeline stable (see _infer). Longer audio is silence-chunked into <=14s windows.
SEG_TARGET, SEG_MAX, SEG_MIN = 12.0, 14.0, 7.0
SIL_RMS = 0.005          # per-frame silence detection for segment CUTTING (not speech gating)
SPEECH_MIN = 0.008       # speech-presence gate on the 90th-pctile of short-window RMS


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2) + 1e-12))


def _has_speech(a, win_s=0.2):
    """True if the segment contains speech. Gates on the 90th-percentile of short-window
    RMS (the loud parts), NOT the whole-segment mean — a short window with a little speech
    + lots of silence has a low mean but a high peak, so mean-gating wrongly dropped it.
    Near-silence: p90 ~ noise floor (<<SPEECH_MIN); speech windows: p90 ~0.015-0.05."""
    w = int(win_s * SR)
    if len(a) < w:
        return _rms(a) >= SPEECH_MIN
    n = len(a) // w
    r = np.sqrt(np.maximum(1e-12, (a[:n * w].reshape(n, w) ** 2).mean(1)))
    return float(np.percentile(r, 90)) >= SPEECH_MIN


def _segment(w):
    """Silence-aware segmentation (port of asr_core.segment): cut at the longest silent
    run near SEG_TARGET, hard cap at SEG_MAX."""
    fl = int(0.02 * SR)
    n = len(w) // fl
    if n == 0:
        return [(0, len(w))]
    rms = np.sqrt(np.maximum(1e-12, (w[:n * fl].reshape(n, fl) ** 2).mean(1)))
    thr = max(rms.mean() * 0.15, np.percentile(rms, 20) * 0.5)
    sil = rms < thr
    segs, start, N = [], 0, len(w)
    while start < N:
        ideal = start + int(SEG_TARGET * SR)
        if ideal >= N:
            segs.append((start, N)); break
        lo = start + int(SEG_MIN * SR)
        hi = min(start + int(SEG_MAX * SR), N)
        f_lo, f_hi = lo // fl, min(hi // fl, len(sil))
        best_len, cut, run_s = 0, None, None
        for f in range(f_lo, f_hi):
            if sil[f]:
                run_s = f if run_s is None else run_s
            elif run_s is not None:
                if f - run_s > best_len:
                    best_len, cut = f - run_s, (run_s + f) // 2
                run_s = None
        if run_s is not None and (f_hi - run_s) > best_len:
            cut = (run_s + f_hi) // 2
        end = cut * fl if cut else hi
        if end <= start:
            end = hi
        segs.append((start, end)); start = end
    return segs


def _hallucinated(text):
    """Repetition hallucination (e.g. '厉害厉害…' on non-speech): one char dominating."""
    import re
    s = re.sub(r"[\s\W_]+", "", text)
    if len(s) < 10:
        return False
    from collections import Counter
    return Counter(s).most_common(1)[0][1] / len(s) > 0.35


@app.get("/health")
def health():
    return {"status": "ok" if STATE else "loading"}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: str = Form("qwen3-asr"),
                     language: str = Form(None)):
    from collections import Counter
    raw = await file.read()
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    force = None if (not language or language.strip().lower() in ("auto", "", "none")) else language
    dur = len(wav) / SR
    with _LOCK:
        _ensure_warm()
        t0 = time.time()
        if dur <= SINGLE_CAP:
            if not _has_speech(wav):              # non-speech upload -> empty (peak-energy gate)
                text, langs, nseg = "", [], 1
            else:
                text, lang, _ = _infer(wav, force)
                text = "" if _hallucinated(text) else text
                langs = [lang] if text else []
                nseg = 1
        else:
            segs = _segment(wav)
            nseg = len(segs)
            parts, langs = [], []
            for a, b in segs:
                win = wav[a:b]
                if not _has_speech(win):           # skip non-speech windows (peak-energy gate)
                    continue
                tx, lg, _ = _infer(win, force)
                if tx and not _hallucinated(tx):
                    parts.append(tx); langs.append(lg)
            text = " ".join(parts)
        dt = time.time() - t0
    # forced language: the tag lives in the prompt, not the generated tokens, so _parse
    # returns "" — echo the requested language instead. Auto-detect: majority vote.
    if force:
        lang = force
    else:
        lang = Counter(langs).most_common(1)[0][0] if langs else ""
    return {"text": text, "language": lang, "duration": round(dur, 2),
            "rtf": round(dt / max(dur, 1e-3), 3), "segments": nseg, "model": model}


# --- P2-4 streaming: WebSocket online ASR (near-real-time, silence-segmented) ---
# Client streams raw PCM16 mono 16k chunks; server emits committed segments as JSON
# {"text","language","seg_end_s"} as soon as each phrase ends (silence) or hits max_seg.
# Single device -> inference runs in a thread executor so audio keeps being received.
# Fine segmentation for LOW LATENCY: cut at the first pause after MIN_SEG, force at MAX_SEG.
# Each segment is padded to FIXED_INFER_SEC(14s) inside _infer anyway (constant prefill -> stable),
# and 14s-padded inference of a short segment is cheap (~0.5s << segment length), so shrinking the
# window directly cuts time-to-caption (~3-6s here vs the old 6-14s) while staying real-time.
STREAM_MIN_SEG, STREAM_MAX_SEG = 3.0, 6.0
STREAM_SIL_SEC, STREAM_SIL_RMS = 0.35, 0.01


@app.websocket("/v1/audio/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    q = ws.query_params
    language = q.get("language")
    force = None if (not language or language.strip().lower() in ("auto", "", "none")) else language
    min_seg = float(q.get("min_seg", STREAM_MIN_SEG))
    max_seg = float(q.get("max_seg", STREAM_MAX_SEG))
    loop = asyncio.get_event_loop()

    def _warm():
        with _LOCK:
            _ensure_warm()
    await loop.run_in_executor(None, _warm)   # compile kernels before the stream's first segment

    buf, nsamp, sil_run, consumed = [], 0, 0.0, 0

    def run_infer(seg):
        with _LOCK:
            tx, lg, _ = _infer(seg, force)
        return ("" if _hallucinated(tx) else tx), lg

    async def flush(seg, final=False):
        if not _has_speech(seg):
            return
        tx, lg = await loop.run_in_executor(None, run_infer, seg)
        if tx:
            await ws.send_json({"text": tx, "language": (force or lg),
                                "seg_end_s": round(consumed / SR, 2), "final": final})

    try:
        while True:
            data = await ws.receive_bytes()
            frame = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
            buf.append(frame); nsamp += len(frame); consumed += len(frame)
            sil_run = sil_run + len(frame) / SR if _rms(frame) < STREAM_SIL_RMS else 0.0
            dur = nsamp / SR
            if (dur >= min_seg and sil_run >= STREAM_SIL_SEC) or dur >= max_seg:
                seg = np.concatenate(buf); buf, nsamp, sil_run = [], 0, 0.0
                await flush(seg)
    except WebSocketDisconnect:
        if buf:
            await flush(np.concatenate(buf), final=True)
