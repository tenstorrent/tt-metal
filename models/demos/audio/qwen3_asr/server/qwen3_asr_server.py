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
import io, os, re, sys, threading, time
import numpy as np
import soundfile as sf
import torch
import ttnn
from fastapi import FastAPI, File, Form, UploadFile
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
    # NOTE: no silence warmup — warming on all-zero audio left the reused KV cache in a
    # state that produced empty output on subsequent requests. The first real request of
    # each new prefill-length pays a one-time JIT compile (~1.5x RTF); later ones are warm.


def _infer(wav, force, max_new_tokens=200):
    """Run the full TT pipeline on a 16k mono float32 waveform. Returns (text, lang, secs)."""
    t0 = time.time()
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
    # use_trace=False in the long-lived server: per-request trace capture/release is
    # not yet robust across many calls (TODO: capture the decode trace once and reuse).
    # Decode is still ~30 tok/s -> RTF well under the 0.30 target.
    ids = STATE["model"].generate(inp.unsqueeze(0), max_new_tokens=max_new_tokens, use_trace=False)
    lang, text = _parse(STATE["tok"].decode(ids, skip_special_tokens=False))
    return text, lang, time.time() - t0


# --- Tier-2 long-form: silence-aware chunking + hallucination/non-speech gating ---
SR = 16000
SINGLE_CAP = 45.0          # <= this -> single-shot; longer -> chunk
SEG_TARGET, SEG_MAX, SEG_SEARCH = 38.0, 45.0, 6.0   # window (s); ~45s -> ~1024 prefill (needs Tier-1)
SIL_RMS = 0.005


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2) + 1e-12))


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
        lo = max(start + int((SEG_TARGET - SEG_SEARCH) * SR), start + int(8 * SR))
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
        t0 = time.time()
        if dur <= SINGLE_CAP:
            if _rms(wav) < SIL_RMS:               # non-speech upload -> empty (matches chunk path)
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
                if _rms(win) < SIL_RMS:            # skip non-speech windows
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
