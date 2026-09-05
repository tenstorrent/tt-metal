# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase-1 reference dump for the Qwen3-ASR-1.7B TT port.

Runs the HF CPU model on a short clip and captures per-stage intermediate
tensors as PCC golden data for the ttnn bring-up (encoder conv frontend,
encoder layers, projector, decoder logits, end-to-end token ids).

Run in the CPU-reference virtualenv (``requirements-reference.txt``: full qwen_asr +
its transformers pin + torch — deliberately NOT the tt-metal env):

    python3 -m venv /tmp/qwen3-asr-ref
    /tmp/qwen3-asr-ref/bin/pip install -r models/demos/audio/qwen3_asr/requirements-reference.txt
    /tmp/qwen3-asr-ref/bin/python models/demos/audio/qwen3_asr/reference/dump_reference.py \
        --out /tmp/qwen3_asr_golden

The defaults need nothing outside a clean checkout: the input clip is an in-repo 16 kHz
mono wav and the model comes from the HF cache. Pass ``--wav/--start/--dur`` to use your
own audio (the PCC goldens are content-independent — they only have to match what the
ttnn port is fed).

Golden tensors are written outside the repo (large) with a small manifest
(shapes/dtypes) printed and saved next to them.
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# Default clip: an in-repo 16 kHz mono speech wav, so golden generation works from a clean
# checkout with no external assets (previously a machine-local path).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DEFAULT_WAV = os.path.join(
    REPO_ROOT,
    "models",
    "demos",
    "audio",
    "whisper",
    "demo",
    "dataset",
    "conditional_generation",
    "17646385371758249908.wav",
)
DEFAULT_GOLDEN_DIR = os.environ.get("QWEN3ASR_GOLDEN_DIR", os.environ.get("GOLDEN_DIR", "/tmp/qwen3_asr_golden"))

# Keep the clip length a whole number of seconds. The AuT front-end consumes mel in 1 s
# chunks (100 frames at the 10 ms hop) and emits 13 encoder rows per chunk. A whole-second
# clip therefore has no partial final chunk, and the CPU reference's audio_tower output
# (masked to feature_lens) has exactly the same row count as the ttnn encoder's chunk-aligned
# output — which is what the PCC tests compare. A 7.62 s clip, say, gives the reference
# 7*13 + ceil(62/8) = 99 rows against the port's 8*13 = 104, and the differing attention
# blocks drop the encoder PCC to ~0.96. See "Known limitations" in the README.
CHUNK_SEC = 1.0
DEFAULT_DUR = 7.0  # <= the in-repo sample clip (7.62 s), whole seconds


def load_slice(path, start, dur, sr=16000):
    w, file_sr = sf.read(path, dtype="float32")
    assert file_sr == sr, file_sr
    if w.ndim > 1:
        w = w.mean(axis=1)
    a = int(start * sr)
    b = len(w) if dur is None else min(len(w), a + int(dur * sr))
    return w[a:b].copy()


# capture: module-name -> list of (args, output) for each forward call
CAP = {}


def make_hook(name):
    def hook(mod, inp, out):
        CAP.setdefault(name, []).append((inp, out))

    return hook


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu()
    return x


def first_tensor(out):
    """Pull the primary tensor out of a hook output (Tensor / tuple / ModelOutput)."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        return first_tensor(out[0])
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "logits"):
        return out.logits
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=DEFAULT_WAV)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=DEFAULT_DUR)
    ap.add_argument("--language", default="English")
    ap.add_argument("--out", default=DEFAULT_GOLDEN_DIR)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from qwen_asr import Qwen3ASRModel

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    print(f"[load] {MODEL_ID} dtype={args.dtype} ...", flush=True)
    t0 = time.time()
    wrap = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="cpu",
        max_inference_batch_size=1,
        max_new_tokens=64,
    )
    model = wrap.model
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    # ---- locate submodules by class-name suffix (robust to attr nesting) ----
    targets = {}
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        leaf = name.split(".")[-1]
        if cls == "Qwen3ASRAudioEncoder":
            targets["audio_tower"] = (name, mod)
        elif leaf == "conv2d1" and "conv2d1" not in [t[0].split(".")[-1] for t in targets.values()]:
            targets["conv2d1"] = (name, mod)
        elif leaf == "conv_out":
            targets["conv_out"] = (name, mod)
        elif leaf == "ln_post":
            targets["ln_post"] = (name, mod)
        elif leaf == "proj2":
            targets["proj2"] = (name, mod)
        elif cls == "Qwen3ASRAudioEncoderLayer" and "enc_layer0" not in targets:
            targets["enc_layer0"] = (name, mod)
        elif leaf == "lm_head":
            targets["lm_head"] = (name, mod)

    print("[hooks] attaching to:")
    for k, (n, _) in targets.items():
        print(f"   {k:12s} <- {n}")
    handles = [m.register_forward_hook(make_hook(k)) for k, (_, m) in targets.items()]

    # pre-hook with kwargs to capture audio_tower's feature_lens (passed as kwarg)
    ENC_KW = {}

    def enc_pre(mod, a, kw):
        ENC_KW["args"] = a
        ENC_KW["kwargs"] = kw

    if "audio_tower" in targets:
        handles.append(targets["audio_tower"][1].register_forward_pre_hook(enc_pre, with_kwargs=True))

    # ---- run one short transcription ----
    wav = load_slice(args.wav, args.start, args.dur)
    print(f"[run] {args.wav} [{args.start},{args.start+args.dur}]s  {len(wav)/16000:.1f}s", flush=True)
    if abs(len(wav) / 16000 / CHUNK_SEC - round(len(wav) / 16000 / CHUNK_SEC)) > 1e-6:
        print(
            f"[warn] clip is {len(wav)/16000:.2f}s — not a whole number of {CHUNK_SEC}s chunks. The "
            "encoder PCC test will see a shorter (feature_lens-masked) golden than the port's "
            "chunk-aligned output; use a whole-second --dur.",
            flush=True,
        )
    t0 = time.time()
    res = wrap.transcribe(audio=[(wav, 16000)], language=args.language)
    txt = res[0].text.strip()
    print(f"[run] done in {time.time()-t0:.1f}s", flush=True)
    print(f"[text] {txt!r}")

    for h in handles:
        h.remove()

    # ---- save audio-encoder INPUTS (drive the ttnn encoder + validate windowing) ----
    enc_calls = CAP.get("audio_tower", [])
    if enc_calls:
        inp, _ = enc_calls[0]
        # forward(input_features, feature_lens=None, aftercnn_lens=None)
        input_features = to_cpu(inp[0])
        np.save(os.path.join(args.out, "input_features.npy"), input_features.numpy())
        print(f"[save] input_features shape={tuple(input_features.shape)}")
        feature_lens = None
        kw = ENC_KW.get("kwargs", {}) or {}
        a = ENC_KW.get("args", ()) or ()
        if "feature_lens" in kw and isinstance(kw["feature_lens"], torch.Tensor):
            feature_lens = kw["feature_lens"]
        elif len(a) > 1 and isinstance(a[1], torch.Tensor):
            feature_lens = a[1]
        elif len(inp) > 1 and isinstance(inp[1], torch.Tensor):
            feature_lens = inp[1]
        if feature_lens is not None:
            fl = feature_lens.detach().cpu().long()
            np.save(os.path.join(args.out, "feature_lens.npy"), fl.numpy())
            print(f"[save] feature_lens={fl.tolist()}")

    # ---- save golden (first forward call per module) ----
    manifest = {
        "model": MODEL_ID,
        "dtype": args.dtype,
        "wav": args.wav,
        "start": args.start,
        "dur": args.dur,
        "language": args.language,
        "text": txt,
        "tensors": {},
    }
    np.save(os.path.join(args.out, "input_wav.npy"), wav)
    for k in targets:
        calls = CAP.get(k, [])
        if not calls:
            print(f"[warn] no capture for {k}")
            continue
        inp, out = calls[0]  # first forward
        t = first_tensor(out)
        if t is None:
            print(f"[warn] no tensor for {k}")
            continue
        t = to_cpu(t)
        fn = os.path.join(args.out, f"{k}.npy")
        np.save(fn, t.numpy())
        manifest["tensors"][k] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "n_calls": len(calls),
            "file": os.path.basename(fn),
        }
        print(f"[save] {k:12s} shape={tuple(t.shape)} calls={len(calls)}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] golden -> {args.out}")


if __name__ == "__main__":
    main()
