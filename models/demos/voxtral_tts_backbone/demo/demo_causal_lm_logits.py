# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 — `causal_lm_logits`: prompt -> next-token logits for every position.

    python -m models.demos.voxtral_tts_backbone.demo.demo_causal_lm_logits \
        --prompt 'The quick brown fox'

The wiring lives in `tt/pipeline.py`; this file only parses argv, opens the one
device, prints the task output and the PCC against the HF golden. The chain it
runs is the same `build_pipeline` + `run_prefill_logits` the e2e test gates.
"""
from __future__ import annotations

import argparse
import sys

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.voxtral_tts_backbone.selftest_device import close_selftest_device, open_selftest_device
from models.demos.voxtral_tts_backbone.tt.pipeline import DEFAULT_PROMPT, _hf_reference_logits, build_pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--layers", type=int, default=None, help="cap the decoder depth (default: all)")
    args = parser.parse_args(argv)

    device = open_selftest_device()
    try:
        pipeline = build_pipeline(device, layers=args.layers, prompt=args.prompt)
        staged = pipeline.stage_inputs(args.prompt)
        logits = pipeline.run_prefill_logits(staged)
        tt_logits = ttnn.to_torch(pipeline.unpadded_logits(logits, staged["prompt_len"])).float()

        golden = _hf_reference_logits(pipeline, staged)
        _, pcc = comp_pcc(golden, tt_logits, 0.95)
        tt_top5 = [int(i) for i in tt_logits[0, -1].topk(5).indices]
        hf_top5 = [int(i) for i in golden[0, -1].topk(5).indices]

        print("==USER 0 - OUTPUT")
        print("prompt:      %r" % args.prompt)
        print("logits:      %s" % (tuple(tt_logits.shape),))
        print("next token:  tt=%d %r | hf=%d %r" % (
            tt_top5[0],
            pipeline.tokenizer.decode([tt_top5[0]]),
            hf_top5[0],
            pipeline.tokenizer.decode([hf_top5[0]]),
        ))
        print("top5:        tt=%s hf=%s (overlap %d/5)" % (tt_top5, hf_top5, len(set(tt_top5) & set(hf_top5))))
        print("depth=%d prompt_len=%d" % (pipeline.depth, staged["prompt_len"]))
        print("e2e PCC=%s" % pcc)
        return 0
    finally:
        close_selftest_device(device)


if __name__ == "__main__":
    sys.exit(main())
