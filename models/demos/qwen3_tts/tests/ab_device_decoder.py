# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
A/B of device speech-decoder variants sharing the chip with the traced talker.

One process, one model load: every variant's decoder is built and warmed BEFORE
``init_server_context`` captures the talker traces (as tt-media-server must), then
real ICL requests run through ``run_inference`` and each variant decodes the same
codes. Every decode is scored against the CPU reference (``decode_icl_audio``) and
timed. ``--negative-control`` also builds one decoder AFTER capture, which must
come out corrupted; if it does not, this harness cannot see the bug.

Run (production device config; --l1-small 131072 for the conv1d variants):
    python models/demos/qwen3_tts/tests/ab_device_decoder.py --device-id 0 --negative-control
    python models/demos/qwen3_tts/tests/ab_device_decoder.py --device-id 0 --l1-small 131072 \\
        --variants cont_hifi4 cont_conv1d_hifi4 cont_conv1d_hifi3
"""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TT_QWEN3_CP_FP32", "1")  # as tt-media-server

import soundfile as sf
import torch

import ttnn
from models.demos.qwen3_tts.tt import server as api

HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
PROMPTS = [
    ("english", "Hello, this is a test."),
    ("english", "Could we push the meeting back to Thursday? I have a conflict in the afternoon."),
    ("japanese", "こんにちは。今日はいい天気ですね。"),
]

# name -> (decode mode, decoder numerics variant)
VARIANTS = {
    # Teja's branch as it was: whole decoder on device, default fidelity, approx GELU.
    "full_legacy": ("full", {"numerics": "legacy", "conv": "matmul"}),
    "full_hifi4": ("full", {"numerics": "hifi", "conv": "matmul", "fidelity": "HiFi4"}),
    "cont_legacy": ("continue", {"numerics": "legacy", "conv": "matmul"}),
    "cont_hifi4": ("continue", {"numerics": "hifi", "conv": "matmul", "fidelity": "HiFi4"}),
    "cont_hifi3": ("continue", {"numerics": "hifi", "conv": "matmul", "fidelity": "HiFi3"}),
    "cont_conv1d_hifi4": ("continue", {"numerics": "hifi", "conv": "conv1d", "fidelity": "HiFi4"}),
    "cont_conv1d_hifi3": ("continue", {"numerics": "hifi", "conv": "conv1d", "fidelity": "HiFi3"}),
}
DEFAULT_VARIANTS = ["full_legacy", "full_hifi4", "cont_legacy", "cont_hifi4", "cont_hifi3"]
NEGATIVE_CONTROL = "cont_hifi4"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snr_db(test, ref):
    test, ref = test.flatten().float(), ref.flatten().float()
    n = min(test.numel(), ref.numel())
    noise = (ref[:n] - test[:n]).pow(2).mean().item()
    return 10.0 * math.log10(ref[:n].pow(2).mean().item() / noise) if noise > 0 else math.inf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--l1-small", type=int, default=32768)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=sorted(VARIANTS))
    p.add_argument("--negative-control", action="store_true")
    # Decoder bucket sizing only (keeps several decoders' DRAM small); the prompts stay well under it.
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--out", default="generated/qwen3_tts_decoder_ab")
    args = p.parse_args()
    out = Path(args.out) / f"l1_{args.l1_small}"
    out.mkdir(parents=True, exist_ok=True)

    log(f"opening device {args.device_id}: l1_small={args.l1_small} trace=512MB cqs=2")
    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=args.l1_small, trace_region_size=512_000_000, num_command_queues=2
    )
    device.enable_program_cache()
    report = {"args": vars(args), "variants": {}, "prompts": []}
    try:
        main_weights, decoder_weights = api.load_weights(HF_ID)
        wav = DEMO_DIR / "jim_reference.wav"
        ref_text = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        ref_codes, audio_data = api.encode_reference_audio(str(wav), main_weights=None)
        ref_state = api.prepare_icl_decoder_state(ref_codes, decoder_weights)
        # Fixed real-speech check input: 20 reference frames + 31 "generated".
        chk_ref, chk_gen = ref_codes[:20], ref_codes[20:]
        chk_cpu = api.decode_icl_audio(chk_ref, chk_gen, decoder_weights)

        decoders = {}

        def build(name):
            mode, variant = VARIANTS[name]
            log(f"build {name} ({mode}, {variant})")
            t0 = time.perf_counter()
            try:
                dec, buckets = api.prepare_device_decoder(
                    device,
                    decoder_weights,
                    int(ref_codes.shape[0]),
                    args.max_new_tokens,
                    icl_continue=mode == "continue",
                    variant=variant,
                )
            except Exception as e:  # noqa: BLE001 -- record and keep the other variants
                log(f"  BUILD FAILED {name}: {type(e).__name__}: {e}")
                return None, {"build_error": f"{type(e).__name__}: {e}"}
            log(f"  {name}: buckets {buckets}, warm {time.perf_counter() - t0:.1f}s")
            return dec, {"buckets": buckets, "warm_s": time.perf_counter() - t0}

        def decode(name, dec, ref, gen, state=None):
            mode, _ = VARIANTS[name]
            if mode == "continue":
                return api.decode_icl_audio(ref, gen, decoder_weights, ref_state=state, device_decoder=dec)
            return api.decode_audio_device(ref, gen, dec)

        def check(tag):
            for name, dec in decoders.items():
                try:
                    s = snr_db(decode(name, dec, chk_ref, chk_gen), chk_cpu)
                except Exception as e:  # noqa: BLE001
                    s = f"{type(e).__name__}: {e}"
                report["variants"][name][f"check_{tag}"] = s
                log(f"  [{tag}] {name}: {s if isinstance(s, str) else f'{s:.2f} dB'}")

        # ---- 1. every variant BEFORE any trace exists ----
        for name in args.variants:
            dec, info = build(name)
            report["variants"][name] = info
            if dec is not None:
                decoders[name] = dec
        log("pre-capture check")
        check("pre_capture")

        # ---- 2. talker + server context (all traces) ----
        from transformers import AutoTokenizer

        from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        talker_config = talker_config_for_hf_id(HF_ID)
        model = Qwen3TTS(device=device, state_dict=main_weights, talker_config=talker_config)
        config = api.TTSConfig(max_new_tokens=256)  # production talker; decoders are sized on --max-new-tokens
        config.greedy = False
        config.repetition_penalty = 1.15
        config.hidden_size = talker_config.hidden_size
        log("init_server_context (captures traces)")
        ctx = api.init_server_context(device, model, config, main_weights)

        if args.negative_control:
            name = f"AFTER_CAPTURE_{NEGATIVE_CONTROL}"
            VARIANTS[name] = VARIANTS[NEGATIVE_CONTROL]
            dec, info = build(name)
            report["variants"][name] = info
            if dec is not None:
                decoders[name] = dec

        tokenizer = AutoTokenizer.from_pretrained(HF_ID, trust_remote_code=True)
        speaker_embedding = model.extract_speaker_embedding(audio_data)

        # ---- 3. real requests; every variant decodes the same codes ----
        for i, (language, text) in enumerate(PROMPTS):
            torch.manual_seed(1234 + i)
            inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _ = api.create_icl_embedding_ttnn(
                target_text=text,
                ref_text=ref_text,
                ref_codes=ref_codes,
                speaker_embedding=speaker_embedding,
                tokenizer=tokenizer,
                model=model,
                device=device,
                config=config,
                main_weights=main_weights,
                language=language,
            )
            t0 = time.perf_counter()
            codes, _, _ = api.run_inference(
                ctx=ctx,
                model=model,
                device=device,
                inputs_embeds_tt=inputs_embeds_tt,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                config=config,
                use_2cq=True,
            )
            gen_ms = (time.perf_counter() - t0) * 1e3
            frames = int(codes.shape[0])
            torch.set_num_threads(1)  # tt-media-server's per-worker budget
            t0 = time.perf_counter()
            cpu = api.decode_icl_audio(ref_codes, codes, decoder_weights, ref_state=ref_state)
            cpu_ms = (time.perf_counter() - t0) * 1e3
            torch.set_num_threads(os.cpu_count() or 1)
            sf.write(out / f"p{i}_cpu.wav", cpu.squeeze().numpy(), 24000)
            row = {"text": text, "frames": frames, "gen_ms": gen_ms, "cpu_1thread_ms": cpu_ms, "variants": {}}
            log(f"prompt {i}: {frames} frames, generation {gen_ms:.0f} ms, CPU decode @1 thread {cpu_ms:.0f} ms")
            for name, dec in decoders.items():
                try:
                    state = ref_state if VARIANTS[name][0] == "continue" else None
                    times = []
                    for _ in range(2):  # 2nd call is the steady-state number
                        torch.set_num_threads(1)
                        t0 = time.perf_counter()
                        dev = decode(name, dec, ref_codes, codes, state)
                        times.append((time.perf_counter() - t0) * 1e3)
                        torch.set_num_threads(os.cpu_count() or 1)
                    s = snr_db(dev, cpu)
                    rms = dev.pow(2).mean().sqrt().item()
                    row["variants"][name] = {"snr_db": s, "decode_ms": times[-1], "first_ms": times[0], "rms": rms}
                    sf.write(out / f"p{i}_{name}.wav", dev.squeeze().numpy(), 24000)
                    log(f"  {name}: {s:.2f} dB, decode {times[-1]:.0f} ms (first {times[0]:.0f}), rms {rms:.4f}")
                except Exception as e:  # noqa: BLE001
                    row["variants"][name] = {"error": f"{type(e).__name__}: {e}"}
                    log(f"  {name}: ERROR {type(e).__name__}: {e}")
            row["cpu_rms"] = cpu.pow(2).mean().sqrt().item()
            report["prompts"].append(row)

        log("post-request check")
        check("post_requests")
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        ttnn.close_device(device)

    # ---- summary ----
    print("\n| variant | build | check pre-capture | check post | SNR min / mean (dB) | decode ms (per prompt) |")
    print("|---|---|---|---|---|---|")
    for name, info in report["variants"].items():
        snrs = [r["variants"][name]["snr_db"] for r in report["prompts"] if "snr_db" in r["variants"].get(name, {})]
        ms = [
            f"{r['variants'][name]['decode_ms']:.0f}"
            for r in report["prompts"]
            if "decode_ms" in r["variants"].get(name, {})
        ]
        fmt = lambda v: v if isinstance(v, str) or v is None else f"{v:.2f}"  # noqa: E731
        print(
            f"| {name} | {'FAIL' if 'build_error' in info else 'ok'} | {fmt(info.get('check_pre_capture'))} | "
            f"{fmt(info.get('check_post_requests'))} | "
            f"{(f'{min(snrs):.2f} / {statistics.mean(snrs):.2f}' if snrs else '-')} | {' / '.join(ms) or '-'} |"
        )
    print(
        "| CPU @1 thread | | | | reference | "
        + " / ".join(f"{r['cpu_1thread_ms']:.0f}" for r in report["prompts"])
        + " |"
    )
    print(f"\nframes per prompt: {[r['frames'] for r in report['prompts']]}; report + WAVs in {out}")


if __name__ == "__main__":
    main()
