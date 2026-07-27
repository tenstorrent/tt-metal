# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Dump Voxtral-TTS prompt token ids to JSON, for `reference/voxtral_pipeline_ref.py --prompt-ids`.

THIS IS THE ONE SCRIPT THAT IS NOT TORCH-ONLY. It needs `mistral-common` for the tekken
tokenizer and the TTS chat template, which is exactly the dependency the reference deliberately
keeps out. Keeping tokenization here draws the boundary in the same place as the XTTS-v2
reference (whose tokenizer also lived outside the ported blocks) and means the reference
pipeline itself stays importable with nothing but torch.

Run it in a THROWAWAY venv, not the tt-metal one:

    python3 -m venv /tmp/mc_venv && /tmp/mc_venv/bin/pip install mistral-common
    /tmp/mc_venv/bin/python models/experimental/voxtral_tts/scripts/dump_prompt_ids.py \
        --text "It took me quite a long time to develop a voice." \
        --voice neutral_male --out prompt_ids.json

Then, back in the tt-metal venv:

    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_pipeline_ref.py \
        --prompt-ids prompt_ids.json --threads $(nproc)

Emitted layout (200 ids for a 20-word sentence with a 169-frame preset):

    [1]  BOS
    [25] begin_audio
    [24] x N_ref        <- audio placeholders; the pipeline replaces these IN ORDER with the
                           voice preset's rows, so N_ref MUST equal the preset's row count
    [35] <text ids> [35]
    [25] begin_audio    <- generation starts after this

`mistral-common` 1.11.7 is known to have `SpeechRequest`; older releases may not (upstream's
example warns about this and tells you to install from git).
"""

import argparse
import json
import os

from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEKKEN = os.path.join(_HERE, "..", "reference", "weights", "tekken.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="neutral_male", help="one of the 20 shipped presets")
    ap.add_argument("--tekken", default=DEFAULT_TEKKEN)
    ap.add_argument("--out", default="prompt_ids.json")
    args = ap.parse_args()

    it = MistralTokenizer.from_file(args.tekken).instruct_tokenizer
    tok = it.encode_speech_request(SpeechRequest(input=args.text, voice=args.voice))
    ids = list(tok.tokens)
    ae = it.audio_encoder

    n_audio = sum(1 for t in ids if t == ae.special_ids.audio)
    payload = {
        "ids": ids,
        "text": args.text,
        "voice": args.voice,
        "audio_token_id": ae.special_ids.audio,
        "begin_audio_token_id": ae.special_ids.begin_audio,
        "n_audio_placeholders": n_audio,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"[dump] {len(ids)} ids ({n_audio} audio placeholders), voice {args.voice!r} -> {args.out}")
    print(f"[dump] audio_token_id={ae.special_ids.audio} begin_audio={ae.special_ids.begin_audio}")
    print("[dump] NOTE the preset must have exactly this many rows; the pipeline asserts it.")


if __name__ == "__main__":
    main()
