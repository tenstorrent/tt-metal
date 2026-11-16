# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The upstream PyTorch CosyVoice2 over the fixed corpus: the baseline every TTNN number is compared against.

RUN IN THE REFERENCE VENV (see requirements-reference*.txt and scripts/reference_env.py):

    COSYVOICE2_REPO=<upstream checkout> LIBRISPEECH_ROOT=<dir containing LibriSpeech/> \\
        $COSYVOICE2_REF_ENV/bin/python run_reference.py --out-dir <dir> [--parity | --extension] [--seed 1986] \\
            [--noise-seed N]

Each case runs `CosyVoice2.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)` -- the same
normalization, splitting and frontend as scripts/prepare_inputs.py -- on CPU in fp32, after
`set_all_random_seed(seed)`. It writes `<case_id>.wav` (24 kHz) and one `results.json` in the schema
scripts/eval_wer_sim.py scores, so the identical command scores this run and a TTNN run.

The reference venv pins the same torch as python_env (2.11.0+cpu), but the TTNN port's logits are not
bit-identical to upstream's. So a seeded run is a baseline for WER/SIM, not a token-for-token target; token
agreement is measured teacher-forced. Upstream's generated speech tokens are captured per segment, by wrapping
`token2wav`, and recorded for that comparison.

`--noise-seed N`: one draw of the vocoder's noise over the same tokens. torch is re-seeded (N x 1000 + call) just
before each `token2wav`, after the LLM has finished, and its state restored after, so the tokens stay those of
`--seed`. WER/SIM over several draws (scripts/eval_draws.py) keeps a claim off one lucky draw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus  # noqa: E402
import reference_env  # noqa: E402

SEED = 1986  # CosyVoice1's, for parity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extension", action="store_true", help="the token-accuracy extension instead (corpus.py)")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--noise-seed", type=int, default=None, help="one vocoder noise draw over the same tokens")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = reference_env.load_upstream()
    from cosyvoice.utils.common import set_all_random_seed

    captured = []
    token2wav = model.model.token2wav

    def capture(*a, **kw):
        captured.append(kw["token"].reshape(-1).tolist())
        if args.noise_seed is None:
            return token2wav(*a, **kw)
        import torch

        state = torch.get_rng_state()  # the LLM's stream: the next segment must sample the same tokens
        torch.manual_seed(args.noise_seed * 1000 + len(captured) - 1)
        try:
            return token2wav(*a, **kw)
        finally:
            torch.set_rng_state(state)

    model.model.token2wav = capture

    results = []
    for case in corpus.extension_cases() if args.extension else corpus.cases(include_parity=args.parity):
        base = reference_env.upstream_repo() if case["set"] == "cosyvoice1_parity" else reference_env.librispeech_root()
        prompt_wav = os.path.join(base, case["prompt_wav"])
        set_all_random_seed(args.seed)
        captured.clear()
        t0 = time.perf_counter()
        pieces = [
            out["tts_speech"].reshape(-1).numpy()
            for out in model.inference_zero_shot(case["text"], case["prompt_text"], prompt_wav, stream=False)
        ]
        wall = time.perf_counter() - t0
        wav = np.concatenate(pieces).astype(np.float32)
        name = f"{case['case_id']}.wav"
        soundfile.write(os.path.join(args.out_dir, name), wav, model.sample_rate)
        audio_s = len(wav) / model.sample_rate
        results.append(
            {
                **case,
                "wav": name,
                "prompt_wav_abs": prompt_wav,
                "seed": args.seed,
                "noise_seed": args.noise_seed,
                "audio_s": round(audio_s, 3),
                "wall_s": round(wall, 3),
                "rtf": round(wall / audio_s, 3),
                "segment_tokens": [list(t) for t in captured],
            }
        )
        print(
            f"  {case['case_id']:<40} audio {audio_s:6.2f} s  wall {wall:6.2f} s  "
            f"tokens {[len(t) for t in captured]}",
            flush=True,
        )
    run = {
        "backend": "pytorch-reference",
        "device": "cpu",
        "meta": {"corpus_version": corpus.CORPUS_VERSION, **reference_env.versions()},
        "results": results,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(results)} wavs and results.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
