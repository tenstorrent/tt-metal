# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 — `text_generation`: prompt -> continuation, on device.

    python -m models.demos.voxtral_tts_backbone.demo.demo_text_generation \
        --prompt 'The quick brown fox' --max-new-tokens 48

The wiring lives in `tt/pipeline.py`; this file only parses argv, opens the one
device, prints the task output and the PCC against the HF golden. The chain it
runs is the same `build_pipeline` + `run_generate` the e2e test gates.
"""
from __future__ import annotations

import argparse
import sys

from models.common.utility_functions import comp_pcc
from models.demos.voxtral_tts_backbone.selftest_device import close_selftest_device, open_selftest_device
from models.demos.voxtral_tts_backbone.tt.pipeline import (
    DEFAULT_PROMPT,
    _hf_reference_generate,
    build_pipeline,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="cap on the decode horizon; the stop rule is the model's eos_token_id",
    )
    parser.add_argument("--layers", type=int, default=None, help="cap the decoder depth (default: all)")
    args = parser.parse_args(argv)

    device = open_selftest_device()
    try:
        pipeline = build_pipeline(device, layers=args.layers, prompt=args.prompt)
        staged = pipeline.stage_inputs(args.prompt)
        generated = pipeline.run_generate(staged, max_new_tokens=args.max_new_tokens)

        tt_token_ids = pipeline.token_ids(generated["tokens"])
        tt_scores = pipeline.stacked_logits(generated["step_logits"])
        hf_token_ids, hf_scores = _hf_reference_generate(pipeline, staged, generated["max_new_tokens"])
        common = pipeline.common_stop_length(tt_token_ids, hf_token_ids)
        _, pcc = comp_pcc(hf_scores[:common], tt_scores[:common], 0.95)
        divergence = next((i for i in range(common) if tt_token_ids[i] != hf_token_ids[i]), -1)

        print("==USER 0 - OUTPUT")
        print("prompt:          %r" % args.prompt)
        print("tt continuation: %r" % pipeline.tokenizer.decode(tt_token_ids[:common]))
        print("hf continuation: %r" % pipeline.tokenizer.decode(hf_token_ids[:common]))
        print("tt tokens:       %s" % tt_token_ids[:common])
        print("hf tokens:       %s" % hf_token_ids[:common])
        print("depth=%d horizon=%d compared=%d first_divergence=%d" % (
            pipeline.depth,
            generated["max_new_tokens"],
            common,
            divergence,
        ))
        print("e2e PCC=%s" % pcc)
        return 0
    finally:
        close_selftest_device(device)


if __name__ == "__main__":
    sys.exit(main())
