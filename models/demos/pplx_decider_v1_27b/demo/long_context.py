# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context mode on one device: prompts up to 262144 tokens (HF ``max_position_embeddings``).

Loads the text model with ``max_seq_len=262144`` (shared K/V cache, chunked residual stream; see
``PplxDeciderModel.forward_long``) and runs two checks:

1. parity: one prompt of about 8000 tokens through the app path (``forward``, bucket 8192, one
   full-attention chunk) and through ``forward_long`` (four 2048-token pieces, chunked SDPA over the
   shared paged cache). Same math, different chunking: the probabilities must agree.
2. needle: a fact at the very start of the prompt ("the vault code color is <C>"), then filler log
   lines up to the target length, then a choice question about the fact. The decision must follow
   the needle for every color.

Usage::

    python models/demos/pplx_decider_v1_27b/demo/long_context.py                       # 262144 tokens
    python models/demos/pplx_decider_v1_27b/demo/long_context.py --lengths 32768 262144 --colors red blue

Results: ``$PPLX_DECIDER_ARTIFACTS/long_context/results.json``.
"""

from __future__ import annotations

import argparse
import json
import time

from loguru import logger

import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider
from models.demos.pplx_decider_v1_27b.tt import paths
from models.demos.pplx_decider_v1_27b.tt.model import LONG_CHUNK, dram_view, long_length
from models.demos.pplx_decider_v1_27b.tt.model_config import LONG_MAX_LENGTH

OUT_DIR = paths.ARTIFACTS / "long_context"
COLORS = ("red", "green", "blue")
QUESTION = {
    "type": "choice",
    "instructions": "According to the note at the very beginning of the log, what is the vault code color?",
    "criteria": {"red": "Red", "green": "Green", "blue": "Blue"},
}


def needle_state(color: str, lines: int) -> str:
    head = f"NOTE: the vault code color is {color.upper()}. Keep this in mind.\n"
    body = "".join(
        f"Log entry {i:06d}: routine status check completed; services reported nominal values.\n" for i in range(lines)
    )
    return head + body + "End of log."


def needle_prompt(decider: TTDecider, color: str, target: int) -> tuple[list[int], int]:
    """Token ids of the needle prompt with the most filler lines that keep it at or under ``target`` tokens."""

    def ids_for(lines):
        return decider.prepare(needle_state(color, lines), QUESTION)[1]

    probe = 1000
    per_line = (len(ids_for(probe)) - len(ids_for(0))) / probe
    lines = max(0, int((target - len(ids_for(0))) / per_line))
    ids = ids_for(lines)
    while len(ids) > target:
        lines -= max(1, int((len(ids) - target) / per_line))
        ids = ids_for(lines)
    return ids, decider.tokenizer.count(decider.row(needle_state(color, lines), QUESTION))


def run_long(model, ids: list[int], count: int, *, outputs: dict | None = None) -> tuple[list[float], float]:
    """``forward_long`` on any length (also <= 8192, for the parity check).

    ``outputs``: also fill it with the host logits [255] and final-normed last-token hidden [5120]."""
    tokens, last_index = model.upload_tokens(ids, bucket=long_length(len(ids)))
    start = time.perf_counter()
    probs, logits, extras = model.forward_long(tokens, last_index, count, return_hidden=outputs is not None)
    out = ttnn.to_torch(probs).reshape(-1)[:count].tolist()
    seconds = time.perf_counter() - start
    if outputs is not None:
        outputs["logits"] = ttnn.to_torch(logits).float().reshape(-1)[:255]
        outputs["final_hidden"] = ttnn.to_torch(extras["final_hidden"]).float().reshape(-1)
        ttnn.deallocate(extras["final_hidden"])
    for t in (tokens, probs, logits):
        ttnn.deallocate(t)
    return out, seconds


def pcc(a, b) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def compare_hf(model, color: str, target: int, decider) -> dict:
    """One prompt on device against the HF golden of ``reference/hf_long_golden.py``."""
    import torch

    from models.demos.pplx_decider_v1_27b.reference.hf_long_golden import golden_path

    path = golden_path(color, target, "bf16")
    if not path.exists():
        raise SystemExit(f"Missing {path}; run python -m models.demos.pplx_decider_v1_27b.reference.hf_long_golden")
    hf = torch.load(path)
    ids, count = needle_prompt(decider, color, target)
    if ids != hf["input_ids"]:
        raise SystemExit("The device prompt differs from the HF golden prompt")
    outputs = {}
    probs, seconds = run_long(model, ids, count, outputs=outputs)
    options = list(QUESTION["criteria"])
    hf_probs = hf["probs"].tolist()
    row = dict(
        tokens=len(ids),
        needle=color,
        tt_answer=options[max(range(count), key=probs.__getitem__)],
        hf_answer=options[max(range(count), key=hf_probs.__getitem__)],
        tt_probs=dict(zip(options, probs)),
        hf_probs=dict(zip(options, hf_probs)),
        max_abs_prob_diff=max(abs(a - b) for a, b in zip(probs, hf_probs)),
        logit_pcc_valid=pcc(outputs["logits"][:count], hf["logits"][:count]),
        logit_pcc_all=pcc(outputs["logits"], hf["logits"]),
        final_hidden_pcc=pcc(outputs["final_hidden"], hf["final_hidden"]),
        seconds=seconds,
    )
    row["same_decision"] = row["tt_answer"] == row["hf_answer"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=int, nargs="+", default=[LONG_MAX_LENGTH], help="needle prompt lengths")
    parser.add_argument("--colors", nargs="+", default=list(COLORS), choices=COLORS)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument(
        "--compare-hf",
        metavar="COLOR",
        choices=COLORS,
        help="only run the needle prompt of COLOR at the first --lengths value against its HF golden",
    )
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    if max(args.lengths) > LONG_MAX_LENGTH:
        parser.error(f"lengths must be <= {LONG_MAX_LENGTH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"max_seq_len": LONG_MAX_LENGTH, "chunk": LONG_CHUNK, "parity": None, "needle": []}

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        start = time.perf_counter()
        decider = TTDecider.from_pretrained(device, max_seq_len=LONG_MAX_LENGTH, vision=False)
        model = decider.model
        results["load_s"] = time.perf_counter() - start
        results["dram_after_load"] = dram_view(device)
        logger.info(f"long-context model loaded in {results['load_s']:.1f} s; DRAM {results['dram_after_load']}")

        if args.compare_hf:
            row = compare_hf(model, args.compare_hf, args.lengths[0], decider)
            (OUT_DIR / f"compare_hf_{args.compare_hf}_{args.lengths[0]}.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row, indent=2))
            return

        if not args.skip_parity:
            ids, count = needle_prompt(decider, "green", 8000)
            app = model.decide(ids, count)
            long, _ = run_long(model, ids, count)
            diff = max(abs(a - b) for a, b in zip(app, long))
            results["parity"] = dict(tokens=len(ids), app=app, long=long, max_abs_diff=diff)
            logger.info(f"parity at {len(ids)} tokens: app {app} long {long} max |diff| {diff:.4g}")

        for target in args.lengths:
            for color in args.colors:
                ids, count = needle_prompt(decider, color, target)
                probs, seconds = run_long(model, ids, count)
                answer = list(QUESTION["criteria"])[max(range(count), key=probs.__getitem__)]
                row = dict(
                    target=target,
                    tokens=len(ids),
                    needle=color,
                    answer=answer,
                    correct=answer == color,
                    probs=dict(zip(QUESTION["criteria"], probs)),
                    seconds=seconds,
                    dram_after=dram_view(device),
                )
                results["needle"].append(row)
                logger.info(
                    f"needle {color} at {len(ids)} tokens: answer {answer} ({'OK' if row['correct'] else 'WRONG'}), "
                    f"probs {row['probs']}, {seconds:.1f} s"
                )
                (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        ttnn.close_device(device)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "dram_after_load"}, indent=2))


if __name__ == "__main__":
    main()
