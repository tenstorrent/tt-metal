# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate a committed golden code fixture for the teacher-forced PCC tests.

Runs the CPU reference once, offline, for a given text/voice and saves the discrete codes.
Tests only load the fixture; they never run the reference. Saved as ``.refpt`` (committable;
``*.pt`` is gitignored). Codes are the only thing stored — tests regenerate hiddens/waveforms live.

    ./python_env/bin/python models/experimental/voxtraltts/tests/generate_voxtral_golden_codes.py \
        --text "..." --voice casual_male --max-tokens 64 --out <path>.refpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.tests.common import VOXTRAL_STANDARD_CHAR_TEXT, resolve_voxtral_model_name_or_skip

_OUT_DIR = Path(__file__).resolve().parent / "reference_outputs"
_DEFAULT_OUT = _OUT_DIR / "voxtral_golden_codes.refpt"


@torch.no_grad()
def generate(*, text: str, voice: str, max_tokens: int, seed: int, out_path: Path) -> None:
    name = resolve_voxtral_model_name_or_skip()
    logger.info(f"Loading CPU reference {name!r} …")
    cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")

    logger.info(f"Generating golden codes (voice={voice}, max_tokens={max_tokens}, seed={seed}) …")
    _ref_wav, ref_codes = cpu.generate(
        text=text, voice=voice, max_tokens=max_tokens, seed=seed, return_tokenizer_codes=True
    )
    codes = torch.as_tensor(ref_codes, dtype=torch.long).cpu()
    assert codes.dim() == 3 and int(codes.shape[1]) == 37, f"expected [1,37,T], got {tuple(codes.shape)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"codes_b37t": codes, "text": text, "voice": voice, "seed": seed, "model_name": name},
        out_path,
    )
    logger.info(f"Saved golden fixture → {out_path}  codes={tuple(codes.shape)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a committed golden codes fixture for any text/voice.")
    p.add_argument("--text", type=str, default=VOXTRAL_STANDARD_CHAR_TEXT)
    p.add_argument("--voice", type=str, default="casual_male")
    p.add_argument("--max-tokens", type=int, default=64, help="Acoustic frames to generate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=str(_DEFAULT_OUT))
    args = p.parse_args()
    generate(text=args.text, voice=args.voice, max_tokens=args.max_tokens, seed=args.seed, out_path=Path(args.out))


if __name__ == "__main__":
    main()
