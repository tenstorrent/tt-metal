#!/usr/bin/env python3
"""Fish S2 Pro on Tenstorrent — command-line TTS (Phase A: TT slow tower, CPU fast decoder + codec).

  python -m models.autoports.fishaudio_s2_pro.demo.demo --text "Hello" --out out_dir [--ref-audio ref.wav --ref-text "..."]
      [--greedy] [--seed 0] [--temperature 0.8 --top-p 0.8] [--max-new-tokens 1024] [--mesh 1x1] [--max-seq-len 8192]

Writes out_dir/{audio.wav, codes.pt, frames.pt, prompt.pt, meta.json} (the audio_gates.py artifact layout).
"""
import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("FISH_S2_MAX_SEQ_LEN", 8192)))
    ap.add_argument("--mesh", default=None, help="RxC (default: FISH_S2_MESH_SHAPE / MESH_DEVICE / all devices)")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--prompt-id", default="cli")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--weights-dtype", default="bfp8", choices=["bfp8", "bf16"])
    args = ap.parse_args()

    import ttnn
    from models.autoports.fishaudio_s2_pro.config import SAMPLE_RATE
    from models.autoports.fishaudio_s2_pro.tt.codec.codec_decoder import CPUCodec
    from models.autoports.fishaudio_s2_pro.tt.device import open_mesh, parse_shape
    from models.autoports.fishaudio_s2_pro.tt.generator import S2Generator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    handle = open_mesh(parse_shape(args.mesh))
    try:
        gen = S2Generator(
            handle.mesh,
            args.snapshot,
            max_seq_len=args.max_seq_len,
            dtype=ttnn.bfloat8_b if args.weights_dtype == "bfp8" else ttnn.bfloat16,
        )
        codec = CPUCodec(gen.snapshot)
        ref_codes = ref_texts = None
        if args.ref_audio:
            wav, sr = sf.read(args.ref_audio, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != SAMPLE_RATE:
                import soxr

                wav = soxr.resample(wav, sr, SAMPLE_RATE)
            ref_codes = [codec.encode(wav)]
            ref_texts = [args.ref_text or ""]
            print(f"reference: {ref_codes[0].shape[1]} frames")
        t0 = time.time()
        codes, st = gen.generate(
            args.text,
            ref_codes=ref_codes,
            ref_texts=ref_texts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
            seed=args.seed,
            on_frame=lambda i, f: print(f"  frame {i} tok={f[0]}", flush=True) if i % 25 == 0 else None,
        )
        t_lm = time.time() - t0
        t1 = time.time()
        wav = codec.decode(codes)
        t_codec = time.time() - t1
        sf.write(out / "audio.wav", wav, SAMPLE_RATE, subtype="PCM_16")
        torch.save(codes, out / "codes.pt")
        meta = {
            "text": args.text,
            "prompt_id": args.prompt_id,
            "variant": args.variant or ("ref" if args.ref_audio else "no_ref"),
            "mode": "greedy" if args.greedy else "sampled",
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stopped_on_im_end": st.stopped_on_im_end,
            "prompt_len": st.prompt_len,
            "frames": st.frames,
            "duration_s": len(wav) / SAMPLE_RATE,
            "mesh": "x".join(map(str, handle.shape)),
            "device_name": gen.args.device_name,
            "weights_dtype": args.weights_dtype,
            "prefill_s": st.prefill_s,
            "decode_s": st.decode_s,
            "slow_s": st.slow_s,
            "fast_s": st.fast_s,
            "codec_s": t_codec,
            "frames_per_s": st.frames_per_s,
            "rtf_lm": st.rtf_lm,
            "rtf_total": (t_lm + t_codec) / max(st.audio_s, 1e-6),
            "impl": {
                "slow": "ttnn/tt_transformers",
                "fast": "ttnn" if getattr(gen, "fast_device", "cpu") == "tt" else "torch-cpu",
                "codec": "torch-cpu",
            },
        }
        json.dump(meta, open(out / "meta.json", "w"), indent=2)
        print(
            json.dumps(
                {
                    k: meta[k]
                    for k in (
                        "frames",
                        "duration_s",
                        "stopped_on_im_end",
                        "prefill_s",
                        "decode_s",
                        "frames_per_s",
                        "rtf_lm",
                        "rtf_total",
                    )
                },
                indent=2,
            )
        )
    finally:
        handle.close()


if __name__ == "__main__":
    main()
