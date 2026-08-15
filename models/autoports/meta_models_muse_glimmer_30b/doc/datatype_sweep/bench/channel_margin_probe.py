# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Why one qualitative prompt picks a different chat channel under the selected policy.

On prompt `p1` the selected config emits ` to=user<|message|>` where the HF control
and the previous stage's TT output emit ` to=self<|message|>` -- a divergence at
**token 1**, which is the position an early-divergence rule would flag as a wrapper
bug rather than as numerics.

This probe measures the thing that decides it: the logit margin between the two
recipient tokens at exactly that position, under a named precision config. If the
two are separated by a hair, a precision change flipping the branch is ordinary
numerics on a near-tie and both continuations are legal template completions; if
the margin is wide, something is wrong and the divergence is a defect.

It scores **both numeric paths**, and that distinction is the point rather than
thoroughness. The qualitative run generates on the traced *decode* path, and
several policy fields -- ``decode_ccl_dtype`` most obviously -- are consumed only
there (``MultichipDecoder._row_parallel_dtype(role, prefill=False)``). A
prefill-only probe therefore cannot attribute anything to those fields: it would
report them as having no effect by construction. Round 4 of the stage review
caught exactly that.

The prompt is the *pinned* one from ``qualitative/qualitative_prompts.json`` -- the
same token ids the HF control ran -- plus the control's own first generated token,
so the probe scores the identical position both arms scored.

Usage::

    python doc/datatype_sweep/bench/channel_margin_probe.py \\
        --configs c14-attn4-cclbfp8-kv8,c00-baseline-attn8-mlp4-kv8-lofi --prompt p1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from models.autoports.meta_models_muse_glimmer_30b.tt import precision_config as pc  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

D = ROOT / "doc/datatype_sweep"
CONFIG_DIR = D / "configs"
QUAL = D / "qualitative"


def say(*args) -> None:
    print(*args, flush=True)


def score(generator, prefix: list[int], *, path: str) -> "torch.Tensor":
    """Host logits for the position after ``prefix``, on one numeric path.

    ``prefill`` runs the whole prefix through the prefill kernels and reads the
    last position's logits. ``decode`` prefills everything but the last token and
    then takes **one traced decode step** on it, which is the path the qualitative
    run actually generates on -- and, for anything gated on ``prefill=False``
    (the decode CCL payload dtype, for one), the only path that can show an
    effect at all.
    """
    import torch

    generator.reset()
    if path == "prefill":
        return generator.prefill_forward(torch.tensor([prefix], dtype=torch.int32), prompt_lens=[len(prefix)])
    if path != "decode":
        raise SystemExit(f"unknown path {path!r}; choose prefill or decode")
    head, last = prefix[:-1], prefix[-1]
    generator.prefill_forward(torch.tensor([head], dtype=torch.int32), prompt_lens=[len(head)])
    return generator.decode_forward(
        torch.tensor([[last]], dtype=torch.int32),
        torch.tensor([len(head)], dtype=torch.int32),
        sample_on_device=False,
        enable_trace=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="c14-attn4-cclbfp8-kv8,c00-baseline-attn8-mlp4-kv8-lofi")
    parser.add_argument("--prompt", default="p1")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--paths",
        default="prefill,decode",
        help="which numeric path to score the position on; the flip happens on decode",
    )
    parser.add_argument("--out", default="channel_margin_probe.json")
    args = parser.parse_args()

    prompts = {item["id"]: item for item in json.loads((QUAL / "qualitative_prompts.json").read_text())}
    hf = {item["id"]: item for item in json.loads((QUAL / "qualitative_hf_chat.json").read_text())}
    tt = {item["id"]: item for item in json.loads((QUAL / "qualitative_tt_chat.json").read_text())}
    entry = prompts[args.prompt]
    # The control's first generated token, appended so the scored position is the
    # one at which the two arms disagree.
    prefix = [int(t) for t in entry["token_ids"]] + [int(hf[args.prompt]["token_ids"][0])]

    from transformers import AutoTokenizer

    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import HF_MODEL_ID, weights_snapshot_dir

    tokenizer = AutoTokenizer.from_pretrained(str(weights_snapshot_dir(HF_MODEL_ID)), local_files_only=True)

    summary: dict = {
        "prompt_id": args.prompt,
        "prompt_source": "doc/datatype_sweep/qualitative/qualitative_prompts.json (pinned to the HF control)",
        "scored_position": len(prefix),
        "prefix_note": (
            "the pinned prompt plus the HF control's own first generated token, so both arms' "
            "disagreement position is the one scored"
        ),
        "hf_token_at_position": int(hf[args.prompt]["token_ids"][1]),
        "hf_token_text": tokenizer.decode([int(hf[args.prompt]["token_ids"][1])]),
        "tt_token_at_position": int(tt[args.prompt]["token_ids"][1]),
        "tt_token_text": tokenizer.decode([int(tt[args.prompt]["token_ids"][1])]),
        "configs": {},
    }

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        for config_id in [c.strip() for c in args.configs.split(",") if c.strip()]:
            config_path = CONFIG_DIR / f"{config_id}.json"
            pc.SELECTED_PRECISION_CONFIG_PATH = config_path
            generator = build_generator(ROOT, mesh, max_seq_len=131072, max_batch_size=1, reuse=False)
            try:
                realised = generator.capability_report()["precision_policy"]
                assert realised["selected_config_id"] == config_id, realised["selected_config_id"]
                summary["configs"][config_id] = {}
                for path in [p.strip() for p in args.paths.split(",") if p.strip()]:
                    logits = score(generator, prefix, path=path)
                    logits = logits.reshape(-1)[: generator.model.config.vocab_size].float()
                    values, indices = torch.topk(logits, args.top)
                    rows = [
                        {
                            "rank": rank,
                            "token": int(token),
                            "text": tokenizer.decode([int(token)]),
                            "logit": float(value),
                        }
                        for rank, (value, token) in enumerate(zip(values.tolist(), indices.tolist()))
                    ]
                    margin = float(values[0] - values[1])
                    summary["configs"][config_id][path] = {
                        "top": rows,
                        "top1_minus_top2": margin,
                        "argmax_token": int(indices[0]),
                        "argmax_text": tokenizer.decode([int(indices[0])]),
                    }
                    say(
                        f"MARGIN {config_id} [{path}]: argmax={int(indices[0])} "
                        f"{tokenizer.decode([int(indices[0])])!r} top1-top2={margin:.4f}  "
                        + " ".join(f"{r['text']!r}={r['logit']:.3f}" for r in rows)
                    )
            finally:
                generator.teardown()
                clear_generator_cache()
    finally:
        path = D / args.out
        path.write_text(json.dumps(summary, indent=2) + "\n")
        say(f"MARGIN summary -> {path}")
        close_multichip_mesh(mesh)
    say("MARGIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
