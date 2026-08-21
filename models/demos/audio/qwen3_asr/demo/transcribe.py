# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Simple audio-in / text-out demo for Qwen3-ASR-1.7B on Tenstorrent (n150/n300).

Give it one or more audio files, get the transcription (language auto-detected):

    python models/demos/audio/qwen3_asr/demo/transcribe.py clip.wav
    python models/demos/audio/qwen3_asr/demo/transcribe.py a.wav b.wav c.flac

Audio of any length works: clips longer than 30s (the feature-extractor cap) are
split into contiguous ~28s windows, transcribed one by one, and the text is joined
(tune with --chunk-sec).

It is fully self-contained: preprocessing (mel + prompt) is done here with the
model's own Whisper feature extractor + chat template, so ONLY tt-metal + its
python_env are needed (no extra `qwen_asr` package).

One-time setup (all inside the tt-metal python_env):
  1. Build tt-metal.
  2. Download the model once:
       huggingface-cli download Qwen/Qwen3-ASR-1.7B

Then just run this script. On first run it auto-extracts the text-decoder checkpoint
from the download (one-time, cached to ~/qwen3_asr_text_decoder). Paths are
auto-detected from the HF cache; override with:
  QWEN3ASR_SNAP_BASE   audio-tower snapshot base (…/models--Qwen--Qwen3-ASR-1.7B/snapshots)
  QWEN3ASR_TEXT_DECODER / HF_MODEL   extracted text-decoder checkpoint dir
  MESH_DEVICE=N300     run on 2-chip n300 (default: single-chip n150)
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, WhisperFeatureExtractor

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs, ModelOptimizations, parse_decoder_json

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "tt"))
import audio_encoder as tt_enc  # noqa: E402
import audio_encoder_ref as ref  # noqa: E402
from extract_text_decoder import TEXT_CFG, TOK_FILES  # noqa: E402
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa: E402

SR = 16000
AUDIO_TOKEN_ID = 151676


def find_snap():
    """Locate the downloaded Qwen3-ASR-1.7B snapshot dir (audio tower + tokenizer)."""
    base = os.environ.get("QWEN3ASR_SNAP_BASE")
    candidates = [base] if base else []
    candidates += [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"),
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            snaps = [os.path.join(c, d) for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            snaps = [s for s in snaps if os.path.exists(os.path.join(s, "config.json"))]
            if snaps:
                return sorted(snaps)[0]
    raise SystemExit(
        "Could not find the Qwen3-ASR-1.7B snapshot. Run `huggingface-cli download "
        "Qwen/Qwen3-ASR-1.7B` or set QWEN3ASR_SNAP_BASE."
    )


def extract_text_decoder(snap, out_dir):
    """Reshuffle the raw qwen3_asr snapshot into a plain Qwen3 checkpoint tt_transformers can load.

    One-time: strips the `thinker.`/`thinker.model.` prefixes, writes a Qwen3 config.json,
    and copies the tokenizer files. Cheap and self-contained (safetensors only, no qwen_asr).
    """
    import json
    import shutil

    from safetensors.torch import save_file

    print(f"[setup] extracting text decoder (one-time) -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    sd = {}
    for f in sorted(glob.glob(snap + "/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.startswith("thinker.model."):
                    sd["model." + k[len("thinker.model.") :]] = h.get_tensor(k)
                elif k == "thinker.lm_head.weight":
                    sd["lm_head.weight"] = h.get_tensor(k)
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump(TEXT_CFG, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for fn in TOK_FILES:
        src = os.path.join(snap, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fn))
    print(f"[setup] extracted {len(sd)} tensors")
    return out_dir


def find_text_decoder(snap):
    d = os.environ.get("QWEN3ASR_TEXT_DECODER") or os.environ.get("HF_MODEL")
    candidates = [d] if d else []
    candidates += [os.path.expanduser("~/qwen3_asr_text_decoder"), "/ttwork/qwen3_asr_text_decoder"]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "model.safetensors")):
            return c
    # Not found -> auto-extract once from the snapshot into the first writable default.
    out_dir = d or os.path.expanduser("~/qwen3_asr_text_decoder")
    return extract_text_decoder(snap, out_dir)


def feat_out_len(frames: int) -> int:
    """Number of audio embeddings the encoder emits for `frames` mel frames."""
    leave = int(frames) % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (int(frames) // 100) * 13


def load_wav(path):
    """Load -> 16k mono float32 (no length padding; chunking handles length)."""
    w, sr = sf.read(path, dtype="float32")
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=-1).astype(np.float32)
    if sr != SR:
        from scipy.signal import resample_poly

        w = resample_poly(w, SR, sr).astype(np.float32)
    return w


def pad_to_second(w):
    """Pad to a whole second so mel frames are a multiple of 100 (matches encoder token count)."""
    nsec = int(np.ceil(len(w) / SR))
    return np.concatenate([w, np.zeros(max(0, nsec * SR - len(w)), dtype=np.float32)])


def chunk_wav(w, chunk_sec):
    """Split a (16k mono) waveform into contiguous <= chunk_sec windows."""
    step = int(chunk_sec * SR)
    if len(w) <= step:
        return [w]
    return [w[i : i + step] for i in range(0, len(w), step)]


def build_inputs(wav, fe, tok, chat_template):
    """wav (16k mono) -> (input_ids, mel) exactly as the Qwen3-ASR processor would."""
    feats = fe(wav, sampling_rate=SR, return_attention_mask=True, return_tensors="pt", padding=True)
    flen = int(feats["attention_mask"].sum())
    mel = feats["input_features"][0][:, :flen].float().contiguous().numpy()
    n_audio = feat_out_len(flen)
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}]},
        ],
        add_generation_prompt=True,
        tokenize=False,
        chat_template=chat_template,
    )
    scaffold = tok(text, return_tensors="pt")["input_ids"][0].tolist()
    cut = scaffold.index(AUDIO_TOKEN_ID)
    ids = np.array(scaffold[:cut] + [AUDIO_TOKEN_ID] * n_audio + scaffold[cut + 1 :], dtype=np.int64)
    return torch.from_numpy(ids).long(), torch.from_numpy(mel).float()


def parse_asr(text):
    """Raw decode -> (language, transcription). Format: 'language <Lang><asr_text><text>'."""
    m = re.search(r"language\s*(.*?)<asr_text>(.*)", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


def main():
    ap = argparse.ArgumentParser(description="Qwen3-ASR-1.7B audio -> text on Tenstorrent")
    ap.add_argument("audio", nargs="+", help="audio file(s): wav/flac/etc @ any sample rate")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--chunk-sec",
        type=float,
        default=28.0,
        help="window length for long audio (must stay <=30s, the feature-extractor cap)",
    )
    args = ap.parse_args()
    chunk_sec = min(args.chunk_sec, 30.0)

    snap = find_snap()
    ckpt = find_text_decoder(snap)
    # tt_transformers' ModelArgs loads the decoder weights/tokenizer from HF_MODEL.
    os.environ["HF_MODEL"] = ckpt
    os.environ["QWEN3ASR_TEXT_DECODER"] = ckpt
    print(f"[setup] snapshot     = {snap}")
    print(f"[setup] text decoder = {ckpt}")

    tok = AutoTokenizer.from_pretrained(ckpt)
    import json

    with open(os.path.join(snap, "chat_template.json")) as fh:
        chat_template = json.load(fh)["chat_template"]
    fe = WhisperFeatureExtractor.from_pretrained(snap)
    with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as h:
        embed = h.get_tensor("model.embed_tokens.weight").float()
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)

    mesh = os.environ.get("MESH_DEVICE", "").upper()
    open_kwargs = dict(trace_region_size=200000000, l1_small_size=32768, num_command_queues=2)
    if mesh == "N300":
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
        )
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), **open_kwargs)
    else:
        dev = ttnn.open_device(device_id=0, **open_kwargs)
    try:
        enc_params = tt_enc.preprocess_weights(w, dev)
        decoder_cfg = os.environ.get("QWEN3ASR_DECODER_CONFIG")
        opt = parse_decoder_json(decoder_cfg, default_optimization=ModelOptimizations.accuracy) if decoder_cfg else None
        margs = ModelArgs(dev, max_batch_size=1, max_seq_len=2048, optimizations=opt)
        sd = margs.load_state_dict()
        model = Qwen3ASRDecoder(
            margs, ttnn.bfloat16, dev, sd, margs.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False
        )

        def transcribe_chunk(wav):
            input_ids, mel = build_inputs(pad_to_second(wav), fe, tok, chat_template)
            t0 = time.time()
            audio_embeds = tt_enc.encode_mel(mel, enc_params, dev).float()  # (N,2048)
            inp = embed[input_ids].clone()
            mask = input_ids == AUDIO_TOKEN_ID
            assert int(mask.sum()) == audio_embeds.shape[0], (int(mask.sum()), audio_embeds.shape)
            inp[mask] = audio_embeds
            t_enc = time.time() - t0
            t0 = time.time()
            ids = model.generate(inp.unsqueeze(0), max_new_tokens=args.max_new_tokens)
            t_dec = time.time() - t0
            lang, text = parse_asr(tok.decode(ids, skip_special_tokens=False))
            return lang, text, len(ids), t_enc, t_dec

        for path in args.audio:
            if not os.path.isfile(path):
                print(f"[skip] not found: {path}")
                continue
            wav = load_wav(path)
            audio_sec = len(wav) / SR
            chunks = chunk_wav(wav, chunk_sec)

            lang = ""
            texts = []
            ntok = t_enc = t_dec = 0
            for ci, cw in enumerate(chunks):
                cl, ct, nt, te, td = transcribe_chunk(cw)
                lang = lang or cl
                if ct:
                    texts.append(ct)
                ntok += nt
                t_enc += te
                t_dec += td
                if len(chunks) > 1:
                    print(f"  [chunk {ci + 1}/{len(chunks)}] {cl}: {ct}")
            text = " ".join(texts)
            rtf = (t_enc + t_dec) / max(audio_sec, 1e-6)
            print(
                f"\n===== {os.path.basename(path)}  ({audio_sec:.0f}s, {len(chunks)} chunk(s), "
                f"{ntok} tok, {ntok / max(t_dec, 1e-6):.0f} tok/s, RTF {rtf:.3f}) ====="
            )
            print(f"  language : {lang}")
            print(f"  text     : {text}")
    finally:
        if mesh == "N300":
            ttnn.close_mesh_device(dev)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        else:
            ttnn.close_device(dev)


if __name__ == "__main__":
    main()
