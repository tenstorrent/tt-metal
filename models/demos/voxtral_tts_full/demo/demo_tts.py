# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Call 1 -- text-to-speech demo for `voxtral-tts-full`.

    python -m models.demos.voxtral_tts_full.demo.demo_tts \
        --text "It took me quite a long time to develop a voice." \
        --voice neutral_male --max-frames 24 --out /tmp/voxtral_tt.wav

Runs the SAME `tt/pipeline.py::run_tts` the e2e test asserts on -- there is exactly one copy of
the chain, so a green test is a working demo by construction.  Input is built by the HF
tokenizer plus a voice preset; output is a real 24 kHz WAV written to disk.

`--compare-reference` additionally runs the HF reference for the same prompt and prints the PCC,
which is the demo-side view of Gate 3.
"""

from __future__ import annotations

import argparse
import sys
import time

import ttnn

from models.demos.voxtral_tts_full.tt import pipeline as P
from models.demos.voxtral_tts_full.tt import reference as ref


def parse_args(argv=None):
    cfg = ref.load_config()
    p = argparse.ArgumentParser(description="Voxtral-TTS on Tenstorrent: text -> 24 kHz speech")
    p.add_argument("--text", default=None,
                   help="text to speak; omitted = the prompt shipped in config.default_prompt_ids")
    p.add_argument("--voice", default=cfg["default_voice"], help="one of the presets in assets/voice_embedding")
    p.add_argument("--max-frames", type=int, default=24,
                   help=f"safety cap on decode length ({cfg['frame_rate']} frames = 1 s of audio); "
                        "generation still stops early on [END_AUDIO]")
    p.add_argument("--layers", type=int, default=None,
                   help="cap the depth of every repeated stack (profiling builds); default = all")
    p.add_argument("--out", default="generated/voxtral_tts_tt.wav", help="output WAV path")
    p.add_argument("--compare-reference", action="store_true",
                   help="also run the HF reference and print the end-to-end PCC")
    p.add_argument("--save-reference", default=None, help="write the reference WAV here too (A/B listening)")
    p.add_argument("--device-id", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    inputs = ref.encode_inputs(text=args.text, voice=args.voice)
    print(f"[demo] text  : {inputs['text']!r}")
    print(f"[demo] voice : {inputs['voice_name']} ({inputs['voice'].shape[0]} preset rows) | "
          f"prompt {inputs['input_ids'].shape[0]} ids | max_frames {args.max_frames}")

    device = ttnn.open_device(device_id=args.device_id,
                              trace_region_size=P.DEFAULT_TRACE_REGION_SIZE)
    try:
        t0 = time.time()
        model = ref.load_hf_model()
        pipe = P.build_pipeline(device, model=model, layers=args.layers)
        print(f"[demo] pipeline built in {time.time() - t0:.0f}s "
              f"(backbone {len(pipe.backbone_layers)}/{pipe.depths['backbone_total']} layers)")

        out = pipe.run_tts(inputs, max_frames=args.max_frames, verbose=True)
        t = out["timings"]
        secs = out["waveform"].shape[-1] / P.SAMPLING_RATE
        print(f"[demo] {out['n_frames']} frames -> {tuple(out['waveform'].shape)} = {secs:.2f}s "
              f"@ {P.SAMPLING_RATE} Hz | peak |x| {out['waveform'].abs().max():.3f}"
              f"{' | stopped on [END_AUDIO]' if out['stopped'] else ''}")
        print(f"[demo] prefill {t['prefill_s']:.1f}s | decode {t['decode_s']:.1f}s "
              f"({t['decode_s'] / max(out['n_frames'], 1):.2f}s/frame) | codec {t['codec_s']:.1f}s")
        print(f"[demo] graduated modules invoked: {out['invoked']}")
        print(f"[demo] wrote {ref.save_wav(out['waveform'], args.out, P.SAMPLING_RATE)}")

        if args.compare_reference or args.save_reference:
            golden = ref.cached_reference_tts(inputs, max_frames=args.max_frames, model=model)
            print(f"e2e PCC={ref.pcc(golden['waveform'], out['waveform'])}")
            flips = int((out["frames"].long() != golden["frames"].long()).sum())
            print(f"[demo] audio-code flips vs reference: {flips}")
            if args.save_reference:
                print(f"[demo] wrote {ref.save_wav(golden['waveform'], args.save_reference, P.SAMPLING_RATE)}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
