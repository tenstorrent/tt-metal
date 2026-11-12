# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF golden of ONE long-context decision (the 262144-token needle prompt), streamed on CPU.

Same pipeline as ``hf_decision_golden`` (embed -> 64 x ``Qwen3_5DecoderLayer`` -> final norm ->
last token -> readout -> mask -> / temperature -> softmax), for the single prompt that
``demo/long_context.py`` runs: the needle prompt of one color at a target length. One layer is
resident at a time; the activations are checkpointed every ``--save-every`` layers (2.5 GiB at
262144 tokens, bf16) so ``--resume`` continues after an interruption.

Writes ``$PPLX_DECIDER_ARTIFACTS/long_context/hf_<color>_<target>_<dtype>.pt``: input_ids, count,
logits [255], probs [count], final_hidden (final-normed last token) and the last token's residual
after every layer [64, 5120]. ``demo/long_context.py --compare-hf`` compares the device run with it.

Usage::

    python -m models.demos.pplx_decider_v1_27b.reference.hf_long_golden            # red, 262144, bf16
    python -m models.demos.pplx_decider_v1_27b.reference.hf_long_golden --resume
"""

from __future__ import annotations

import argparse
import time

import torch
from loguru import logger

from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer
from models.demos.pplx_decider_v1_27b.reference.hf_decision_golden import (
    DTYPES,
    decision_head,
    peak_rss_gb,
    position_embeddings,
)
from models.demos.pplx_decider_v1_27b.reference.hf_reference import (
    SnapshotReader,
    build_decoder_layer,
    build_final_norm,
    build_readout,
    layer_forward,
)
from models.demos.pplx_decider_v1_27b.tt import paths
from models.demos.pplx_decider_v1_27b.tt.model_config import LONG_MAX_LENGTH

OUT_DIR = paths.ARTIFACTS / "long_context"


class _TokenizerDecider:
    """The ``TTDecider`` prompt API (``prepare`` / ``row`` / ``tokenizer``) without a device."""

    def __init__(self, tokenizer: AppTokenizer):
        self.tokenizer = tokenizer

    def row(self, state, question):
        return {"state": state, "question": question}

    def prepare(self, state, question):
        row = self.row(state, question)
        return row, self.tokenizer.input_ids(row), self.tokenizer.count(row)


def golden_path(color: str, target: int, dtype: str):
    return OUT_DIR / f"hf_{color}_{target}_{dtype}.pt"


@torch.no_grad()
def main() -> None:
    from models.demos.pplx_decider_v1_27b.demo.long_context import COLORS, needle_prompt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--color", default="red", choices=COLORS)
    parser.add_argument("--target", type=int, default=LONG_MAX_LENGTH)
    parser.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=64, help="< 64: timing estimate only, nothing saved")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dtype = DTYPES[args.dtype]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = OUT_DIR / f"hf_state_{args.color}_{args.target}_{args.dtype}.pt"

    reader = SnapshotReader()
    cfg = reader.text_config
    tokenizer = AppTokenizer(reader.path)
    ids, count = needle_prompt(_TokenizerDecider(tokenizer), args.color, args.target)
    seq = len(ids)
    logger.info(f"needle {args.color}: {seq} tokens, {count} options, dtype {args.dtype}")

    trace = torch.zeros(cfg.num_hidden_layers, cfg.hidden_size, dtype=torch.float32)
    start_layer = 0
    if args.resume and state_path.exists():
        state = torch.load(state_path)
        if state["input_ids"] != ids:
            raise ValueError(f"{state_path} holds another prompt")
        h, trace, start_layer = state["hidden"], state["trace"], state["next_layer"]
        logger.info(f"resumed at layer {start_layer}")
    else:
        h = reader.embedding_weight()[torch.tensor(ids)][None].to(dtype)
    pe = position_embeddings(cfg, seq, dtype)

    n_layers = min(args.max_layers, cfg.num_hidden_layers)
    t0 = time.time()
    for i in range(start_layer, n_layers):
        start = time.time()
        layer = build_decoder_layer(reader, i, dtype)
        h = layer_forward(layer, h, pe)
        del layer
        trace[i] = h[0, -1].float()
        logger.info(
            f"layer {i:2d} ({cfg.layer_types[i]}) S={seq} {time.time() - start:.1f}s "
            f"(total {time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.1f} GB)"
        )
        if n_layers == cfg.num_hidden_layers and (i + 1) % args.save_every == 0 and i + 1 < n_layers:
            torch.save(dict(input_ids=ids, hidden=h, trace=trace, next_layer=i + 1), state_path)
    if n_layers < cfg.num_hidden_layers:
        logger.info(f"timing estimate only ({n_layers} layers); nothing saved")
        return

    last = build_final_norm(reader, dtype)(h[:, -1])
    logits, probs = decision_head(last, build_readout(reader, dtype), count, reader.temperature)
    out = dict(
        input_ids=ids,
        count=count,
        color=args.color,
        dtype=args.dtype,
        logits=logits.float(),
        probs=probs.float(),
        final_hidden=last[0].float(),
        layer_trace=trace,
    )
    torch.save(out, golden_path(args.color, args.target, args.dtype))
    state_path.unlink(missing_ok=True)
    logger.info(
        f"HF {args.color} at {seq} tokens: probs {probs.tolist()}; saved {golden_path(args.color, args.target, args.dtype)}"
    )


if __name__ == "__main__":
    main()
