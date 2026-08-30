# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Synthesise speech on a Tenstorrent device and write a .wav.

    export PYTHONPATH=$TT_METAL_HOME
    python models/demos/cosyvoice/demo/demo.py --inputs /tmp/sweep --out /tmp/out

`/tmp/sweep` is a directory from `scripts/prepare_inputs.py`, which runs the
CosyVoice front-end once in its own venv -- see that file for why the boundary
sits there. With `--inputs`, the demo runs all four modes (sft, zero_shot,
cross_lingual, instruct) through `tt.pipeline.CosyVoiceTTNN.synthesize`,
generating fresh semantic tokens and fresh vocoder excitation for each -- this
is real synthesis, not golden reproduction, so no two runs sound identical.

Without `--inputs`, the demo runs on the captured golden utterance instead,
which needs no front-end at all and is the quickest way to hear the pipeline
work, and additionally scores itself against the reference waveform.

All three stages run on device. What crosses back to the host is the sampled
token IDs (RAS needs the full distribution and the emission history) and the
final waveform.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("TT_METAL_HOME", "."))

import ttnn  # noqa: E402
from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden  # noqa: E402
from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec  # noqa: E402
from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator  # noqa: E402
from models.demos.cosyvoice.tt.llm.model import TtTransformerLM  # noqa: E402
from models.demos.cosyvoice.tt.pipeline import MODES, CosyVoiceTTNN, PromptContext, RandomSources  # noqa: E402
from models.demos.cosyvoice.tt.weights import WeightBag, default_weights_path  # noqa: E402

SAMPLE_RATE = 22050


def write_wav(path: str, wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    """16-bit PCM via the stdlib, so the demo needs no audio library."""
    import wave

    data = wav.flatten().clamp(-1.0, 1.0).mul(32767).to(torch.int16).numpy()
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(data.tobytes())


def write_run_dir(run_dir: str, wav: torch.Tensor, seconds: float) -> None:
    """Lay the output out exactly as `run_reference.py` does, so
    `scripts/eval_wer_sim.py --run-dir <this>` scores TTNN audio through the
    identical code path that scored the PyTorch reference.

    That identity is the point. A separate scoring path for the port would make
    any WER difference ambiguous between "the model is worse" and "the harness
    differs", which is precisely the ambiguity the scoring path exists to remove.
    """
    import json

    from models.demos.cosyvoice.tt.common import golden_manifest

    manifest = golden_manifest()
    os.makedirs(run_dir, exist_ok=True)
    name = "ttnn_zero_shot_zh.wav"
    write_wav(os.path.join(run_dir, name), wav)
    payload = {
        "engine": "ttnn",
        "device": "blackhole-p150a",
        "checkpoint": manifest.get("model_dir", "CosyVoice-300M"),
        "results": [
            {
                "mode": "zero_shot",
                "lang": "zh",
                "text": manifest["text"],
                "wav": name,
                "seconds": round(seconds, 3),
                # The excitation is the device's own, not the reference's. WER and
                # SIM are perceptual metrics, so this is the right variant to score
                # -- the phase difference documented in tt/hifigan/source.py is
                # inaudible, and scoring the injected-excitation variant would be
                # measuring the reference's vocoder rather than the port's.
                "excitation": "self-computed",
            }
        ],
    }
    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"wrote {run_dir}/results.json + {name} -- score with scripts/eval_wer_sim.py --run-dir")


def golden_inputs():
    """The captured utterance, so the demo runs with no front-end."""
    emb_g = load_golden("flow.input_embedding")
    lr_g = load_golden("flow.length_regulator")
    cfm_g = load_golden("flow.cfm")
    spk_g = load_golden("flow.spk_embed_affine")
    sine_g = load_golden("hift.sinegen")
    mel_len1 = int(lr_g["call0.in_mel_len1"])
    return {
        "tokens": torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32),
        "token_len1": as_torch(lr_g["call0.in_x1"]).shape[1],
        "mel_len1": mel_len1,
        "mel_len2": int(lr_g["call0.in_mel_len2"]),
        "prompt_feat": as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous(),
        "embedding": as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1),
        "z": as_torch(cfm_g["call0.rng_z"]).permute(0, 2, 1).contiguous(),
        "phase_vec": as_torch(sine_g["call0.in_phase_vec"]),
        "sine_noise": as_torch(sine_g["call0.out_noise"]).permute(0, 2, 1).contiguous(),
    }


def run_multi_mode(args) -> int:
    """`--inputs <dir>`: real synthesis, all four modes, through `CosyVoiceTTNN`.

    Unlike the golden path below, this generates fresh LLM tokens and a fresh
    vocoder excitation per mode -- there is no reference to score against, only
    a wav per mode to listen to. One utterance per mode, not the full
    mode x language sweep `prepare_inputs.py`/`run_reference.py` already cover.
    """
    # hift is bit-identical across CosyVoice-300M / -SFT / -Instruct (checksums
    # compared directly), so one export covers all four modes. llm differs across
    # all three checkpoints; flow differs only for -SFT (-Instruct's flow.pt is
    # byte-identical to the base checkpoint's). scripts/export_weights.py
    # --checkpoint CosyVoice-300M-SFT/-Instruct produced the per-mode files this
    # reads -- see tests/golden/per_mode/.
    hift_path = os.path.join(args.weights_dir, "hift_weights.npz")
    per_mode_dir = os.path.join(args.weights_dir, "per_mode")
    llm_path_for = {
        "sft": os.path.join(per_mode_dir, "llm_weights_sft.npz"),
        "instruct": os.path.join(per_mode_dir, "llm_weights_instruct.npz"),
    }
    flow_path_for = {"sft": os.path.join(per_mode_dir, "flow_weights_sft.npz")}
    base_llm_path = os.path.join(args.weights_dir, "llm_weights.npz")
    base_flow_path = os.path.join(args.weights_dir, "flow_weights.npz")

    modes = args.modes.split(",") if args.modes else list(MODES)
    # `--out`'s default is a filename sized for the single-wav golden path; a
    # directory literally named "cosyvoice_ttnn.wav" would work but reads oddly.
    out_dir = "cosyvoice_ttnn_out" if args.out == "cosyvoice_ttnn.wav" else args.out
    os.makedirs(out_dir, exist_ok=True)

    # A fresh device per mode, not one shared across the loop. Measured on
    # silicon: `model.llm.release_caches()` alone is not enough here -- the four
    # modes' mel lengths are all different (this is real synthesis, not one
    # repeated golden geometry), and something in the conv/halo sliding-window
    # path accumulates L1_SMALL state per *distinct* geometry rather than per
    # utterance. Two modes in a row completely exhausted a 32 KB L1_SMALL bank
    # (`0 B free`) even with the LLM caches released every time. Nobody has
    # exercised this pipeline across varying-length real inputs before, so
    # reopening the device is the correct scope for a quickstart demo -- root
    # -causing the leak itself belongs to whichever stage turns out to own it,
    # not to this script.
    for mode in modes:
        npz_path = os.path.join(args.inputs, f"{mode}_{args.lang}.npz")
        if not os.path.exists(npz_path):
            print(f"  skip {mode}: {npz_path} not found (did prepare_inputs.py write --langs including {args.lang!r}?)")
            continue
        ctx, meta = PromptContext.from_npz(npz_path)
        llm_path = llm_path_for.get(mode, base_llm_path)
        flow_path = flow_path_for.get(mode, base_flow_path)
        for p in (llm_path, flow_path, hift_path):
            if not os.path.exists(p):
                raise SystemExit(f"missing {p} -- run scripts/export_weights.py --checkpoint ... first")
        device = ttnn.open_device(device_id=0, l1_small_size=32768)
        try:
            model = CosyVoiceTTNN(
                device, WeightBag.load(llm_path), WeightBag.load(flow_path), WeightBag.load(hift_path)
            )
            t0 = time.perf_counter()
            if args.stream:
                # Interleaved: the flow decoder and the vocoder run on each finished
                # chunk while the LLM is still decoding later tokens, so audio starts
                # before generation does. `first_audio_s` is the number that
                # distinguishes this from chunked vocoding after the fact.
                res = model.synthesize_streaming(ctx, sampler=args.sampler, seed=args.seed)
                dt = time.perf_counter() - t0
                out = torch.cat([ttnn.to_torch(c).float().reshape(1, -1) for c in res.chunks], dim=1)
                tokens, first_s, n_chunks = res.tokens, res.first_audio_s, res.n_chunks
                res.free()
            else:
                wav, tokens = model.synthesize(ctx, rng=RandomSources(), sampler=args.sampler, seed=args.seed)
                dt = time.perf_counter() - t0
                out = ttnn.to_torch(wav).float().reshape(1, -1)
                ttnn.deallocate(wav)
                first_s, n_chunks = dt, 1  # nothing can be emitted before the end
        except Exception as e:  # noqa: BLE001
            print(f"  {mode:<14} FAILED: {str(e)[:160]}")
            continue
        finally:
            ttnn.close_device(device)
        out_path = os.path.join(out_dir, f"{mode}_{args.lang}.wav")
        write_wav(out_path, out)
        secs = out.shape[1] / SAMPLE_RATE
        tail = f"  first audio {first_s:5.2f}s in {n_chunks} chunks" if args.stream else ""
        print(
            f"  {mode:<14} {meta['text'][:40]!r:<44} {len(tokens):>3} tokens  "
            f"{secs:5.2f}s audio  {dt:5.2f}s wall{tail}  -> {out_path}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", default=None, help="a scripts/prepare_inputs.py --out-dir; omit to use the golden utterance"
    )
    ap.add_argument("--lang", default="en", help="which --inputs language to pick per mode (multi-mode path only)")
    ap.add_argument("--modes", default=None, help=f"comma-separated subset of {','.join(MODES)} (default: all four)")
    ap.add_argument("--out", default="cosyvoice_ttnn.wav", help="a .wav path, or with --inputs, an output directory")
    ap.add_argument("--weights-dir", default=GOLDEN_DIR)
    ap.add_argument("--sampler", default="ras", choices=["ras", "greedy"])
    ap.add_argument("--seed", type=int, default=1986)
    ap.add_argument("--skip-llm", action="store_true", help="use the captured tokens instead of generating")
    ap.add_argument(
        "--stream",
        action="store_true",
        help="interleave the stages (--inputs path only): emit audio chunks as tokens are generated "
        "instead of after the last one, and report time to first audio. NOTE: generation is correct "
        "but the assembled audio has a known amplitude defect -- see CosyVoiceTTNN.synthesize_streaming",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="also write a results.json + wav laid out for scripts/eval_wer_sim.py, so TTNN audio "
        "is scored by the identical code path as the PyTorch reference",
    )
    args = ap.parse_args()

    if args.inputs:
        return run_multi_mode(args)

    hift_path = os.path.join(args.weights_dir, "hift_weights.npz")
    flow_path = os.path.join(args.weights_dir, "flow_weights.npz")
    llm_path = os.path.join(args.weights_dir, "llm_weights.npz")
    for p in (hift_path, flow_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run scripts/export_weights.py first")

    src = golden_inputs()

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:

        def dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

        tokens = src["tokens"]
        if not args.skip_llm and os.path.exists(llm_path):
            t0 = time.perf_counter()
            llm_bag = WeightBag.load(llm_path)
            llm = TtTransformerLM(device, llm_bag, llm_bag.meta)
            n_prompt = src["token_len1"]
            generated = llm.generate(
                dev(tokens[:, :1].repeat(1, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                prompt_speech_tokens=dev(tokens[:, :n_prompt], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                sampler=args.sampler,
                seed=args.seed,
                max_tokens=64,
            )
            print(f"  LLM: {len(generated)} tokens in {time.perf_counter() - t0:.2f} s")
            print("  (the demo then continues with the captured tokens, so the audio is comparable)")

        t0 = time.perf_counter()
        flow_bag = WeightBag.load(flow_path)
        flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)
        mel = flow.inference(
            dev(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            src["token_len1"],
            src["mel_len1"],
            src["mel_len2"],
            dev(src["prompt_feat"]),
            dev(src["embedding"]),
            dev(src["z"]),
        )
        print(f"  flow: {src['mel_len2']} mel frames in {time.perf_counter() - t0:.2f} s")

        t0 = time.perf_counter()
        hift = TtHiFTGenerator(device, WeightBag.load(hift_path))
        wav, audio_len, src = hift.inference(
            mel,
            src["mel_len2"],
            phase_vec=dev(src["phase_vec"], dtype=ttnn.float32),
            sine_noise=dev(src["sine_noise"], dtype=ttnn.float32),
        )
        out = ttnn.to_torch(wav).float().reshape(1, -1)
        print(f"  vocoder: {audio_len} samples in {time.perf_counter() - t0:.2f} s")
        for _t in (mel, wav, src):
            ttnn.deallocate(_t)
    finally:
        ttnn.close_device(device)

    write_wav(args.out, out)
    secs = out.shape[1] / SAMPLE_RATE
    print(f"wrote {args.out}  ({secs:.2f} s, {SAMPLE_RATE} Hz, peak {float(out.abs().max()):.3f})")

    if args.run_dir:
        write_run_dir(args.run_dir, out, secs)
    ref = os.path.join(GOLDEN_DIR, "e2e.npz")
    if os.path.exists(ref):
        want = as_torch(np.load(ref)["waveform"])
        n = min(out.shape[1], want.shape[1])

        def corr(a, b):
            a, b = a.flatten().double(), b.flatten().double()
            a, b = a - a.mean(), b - b.mean()
            return float((a * b).sum() / (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()))

        win = 256
        env = [x[:, :n][:, : n // win * win].reshape(-1, win).pow(2).mean(1).sqrt() for x in (out, want)]
        print(
            f"vs the PyTorch reference audio:  envelope {corr(*env):.6f}   samples {corr(out[:, :n], want[:, :n]):.6f}"
        )
        print(
            "  The envelope is the meaningful figure here. This run builds its own excitation, and\n"
            "  NSF phase is chaotically sensitive to f0 -- drift is sum(dF0)/sr over samples, so\n"
            "  holding it under a tenth of a cycle across 72192 samples needs a 0.03 Hz f0 error,\n"
            "  finer than Tensix arithmetic delivers. The audio is right; its phase is a different\n"
            "  valid realisation. tests/e2e/ gates samples with the reference excitation injected\n"
            "  (PCC 0.9951) and the envelope without it (0.9975). See tt/hifigan/source.py."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
