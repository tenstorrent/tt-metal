# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Synthesise the mode x language sweep on device and lay it out for scoring.

    python models/demos/cosyvoice/demo/sweep.py --inputs /tmp/cosy_sweep --out-dir /out/run

`--inputs` is what `scripts/prepare_inputs.py` wrote in the CosyVoice venv. The
output is a `results.json` + wavs in exactly `run_reference.py`'s layout, so
`scripts/eval_wer_sim.py --run-dir` scores TTNN audio through the identical code
path that scored the PyTorch reference. That identity is the point: a separate
scoring path would make any WER difference ambiguous between "the model is worse"
and "the harness differs".

All three stages run on device. The LLM's sampled token IDs come back to the host
because RAS needs the full distribution and the emission history; nothing else
does.

**Randomness is drawn here, on the host, and injected.** The CFM's initial `z`,
and `SineGen`'s phase offsets and noise. A device RNG cannot be aligned with
torch's stream, and the noise in particular is not optional: for unvoiced frames
`uv` zeroes the sine bank, so the Gaussian noise is the entire excitation there.
Synthesising without it silences every fricative.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave

import numpy as np
import torch

sys.path.insert(0, os.environ.get("TT_METAL_HOME", "."))

import ttnn  # noqa: E402
from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec  # noqa: E402
from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator  # noqa: E402
from models.demos.cosyvoice.tt.llm.model import TtTransformerLM  # noqa: E402
from models.demos.cosyvoice.tt.weights import WeightBag  # noqa: E402

SAMPLE_RATE = 22050
HARMONICS = 9  # nb_harmonics 8, plus the fundamental


def write_wav(path, wav, sample_rate=SAMPLE_RATE):
    data = wav.flatten().clamp(-1.0, 1.0).mul(32767).to(torch.int16).numpy()
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(data.tobytes())


def load_case(path):
    with np.load(path) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        arrays = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in z.files if k != "__meta__"}
    return arrays, meta


def synth(device, models, case, meta, seed, max_tokens):
    """One utterance, all three stages. Returns (waveform, stats)."""
    llm, flow, hift = models

    def dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

    def ids(v):
        return ttnn.from_torch(v.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # ---- LLM: the prompt text (or the instruct description) is a prefix on the
    # text, exactly as TransformerLM.inference concatenates it.
    text = case["text"]
    n_text = int(case["text_len"][0])
    if "prompt_text" in case:
        text = torch.cat([case["prompt_text"], text], dim=1)
    # instruct deletes llm_embedding entirely -- the LLM runs with no speaker vector
    spk = None
    if "llm_embedding" in case:
        spk = llm.speaker_embedding(dev(case["llm_embedding"].reshape(1, 1, -1)))
    prompt_tok = case.get("llm_prompt_speech_token")

    t0 = time.perf_counter()
    tokens = llm.generate(
        ids(text),
        spk_emb=spk,
        prompt_speech_tokens=ids(prompt_tok) if prompt_tok is not None else None,
        text_len=n_text,
        sampler="ras",
        seed=seed,
        max_tokens=max_tokens,
    )
    llm_s = time.perf_counter() - t0
    if not tokens:
        raise RuntimeError("LLM emitted no tokens")

    # ---- flow
    flow_prompt = case.get("flow_prompt_speech_token")
    gen = torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)
    all_tokens = torch.cat([flow_prompt.to(torch.int32), gen], dim=1) if flow_prompt is not None else gen
    token_len1 = int(flow_prompt.shape[1]) if flow_prompt is not None else 0
    feat = case.get("prompt_speech_feat")
    mel_len1 = int(feat.shape[1]) if feat is not None else 0
    mel_len2 = TtMaskedDiffWithXvec.mel_len_for(len(tokens))
    if feat is None:
        feat = torch.zeros(1, 0, flow.output_size)

    torch.manual_seed(seed)
    z = torch.randn(1, mel_len1 + mel_len2, flow.output_size)
    t0 = time.perf_counter()
    mel = flow.inference(
        ids(all_tokens),
        token_len1,
        mel_len1,
        mel_len2,
        dev(feat.float()),
        dev(case["flow_embedding"].reshape(1, 1, -1)),
        dev(z),
    )
    flow_s = time.perf_counter() - t0

    # ---- vocoder. phase and unit noise drawn here; see the module docstring.
    audio_len = mel_len2 * 256  # total upsample: 8 x 8 x hop 4
    phase = torch.empty(1, 1, HARMONICS).uniform_(-np.pi, np.pi)
    phase[0, 0, 0] = 0.0  # the fundamental is unshifted, as upstream
    noise_unit = torch.randn(1, audio_len, HARMONICS)
    t0 = time.perf_counter()
    wav, n_samples, src = hift.inference(
        mel,
        mel_len2,
        phase_vec=dev(phase, dtype=ttnn.float32),
        sine_noise_unit=dev(noise_unit, dtype=ttnn.float32),
    )
    hift_s = time.perf_counter() - t0
    out = ttnn.to_torch(wav).float().reshape(1, -1)
    for _t in (mel, wav, src):
        ttnn.deallocate(_t)

    seconds = n_samples / SAMPLE_RATE
    return out, {
        "tokens": len(tokens),
        "seconds": round(seconds, 3),
        "llm_s": round(llm_s, 2),
        "flow_s": round(flow_s, 2),
        "hift_s": round(hift_s, 2),
        "rtf": round((llm_s + flow_s + hift_s) / max(seconds, 1e-6), 3),
        "tokens_per_second": round(len(tokens) / max(llm_s, 1e-6), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="directory written by scripts/prepare_inputs.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--modes", default="zero_shot,cross_lingual")
    ap.add_argument("--langs", default="zh,en,ja,yue,ko")
    ap.add_argument("--seed", type=int, default=1986)
    ap.add_argument("--max-tokens", type=int, default=600)
    # 32768 is what a single-model test uses and it is not enough here. ttnn.conv1d
    # allocates its prepared weights from the L1_SMALL bank and keeps them, so the
    # bank fills in proportion to the number of distinct conv CONFIGURATIONS live in
    # the process -- not to any tensor size. Three models at once (LLM, flow,
    # vocoder) is ~80 convs and overflows a 32 KB bank part-way through the second
    # utterance, reported as "Not enough space to allocate 480 B L1_SMALL buffer".
    ap.add_argument("--l1-small", type=int, default=131072)
    args = ap.parse_args()

    wdir = args.weights_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
    with open(os.path.join(args.inputs, "index.json")) as fh:
        index = json.load(fh)["cases"]
    want_modes, want_langs = set(args.modes.split(",")), set(args.langs.split(","))
    cases = [c for c in index if c["mode"] in want_modes and c["lang"] in want_langs]
    if not cases:
        raise SystemExit(f"no cases match modes={args.modes} langs={args.langs}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = ttnn.open_device(device_id=0, l1_small_size=args.l1_small)
    results = []
    try:
        print("loading weights ...", flush=True)
        llm_bag = WeightBag.load(os.path.join(wdir, "llm_weights.npz"))
        flow_bag = WeightBag.load(os.path.join(wdir, "flow_weights.npz"))
        models = (
            TtTransformerLM(device, llm_bag, llm_bag.meta),
            TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta),
            TtHiFTGenerator(device, WeightBag.load(os.path.join(wdir, "hift_weights.npz"))),
        )
        for c in cases:
            case, meta = load_case(os.path.join(args.inputs, c["file"]))
            name = f"ttnn_{c['mode']}_{c['lang']}.wav"
            t0 = time.perf_counter()
            try:
                wav, stats = synth(device, models, case, meta, args.seed, args.max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  {c['mode']:<14} {c['lang']:<4} FAILED: {str(e)[:120]}", flush=True)
                results.append({**c, "error": str(e)[:400]})
                models[0].release_caches()
                continue
            models[0].release_caches()
            write_wav(os.path.join(args.out_dir, name), wav)
            entry = {k: c[k] for k in ("mode", "lang", "text", "checkpoint")}
            entry.update({"wav": name, **stats, "wall_s": round(time.perf_counter() - t0, 1)})
            results.append(entry)
            print(
                f"  {c['mode']:<14} {c['lang']:<4} {stats['tokens']:4d} tok  {stats['seconds']:5.2f}s audio"
                f"  RTF {stats['rtf']:5.2f}  ({stats['tokens_per_second']:.1f} tok/s)  wall {entry['wall_s']}s",
                flush=True,
            )
    finally:
        ttnn.close_device(device)

    payload = {"engine": "ttnn", "device": "blackhole-p150a", "seed": args.seed, "results": results}
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    ok = [r for r in results if "wav" in r]
    print(f"\n{len(ok)}/{len(results)} synthesised -> {args.out_dir}")
    if ok:
        print(f"  median RTF {sorted(r['rtf'] for r in ok)[len(ok)//2]:.2f}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
