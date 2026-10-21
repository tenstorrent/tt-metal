# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Store a GPU trace's side of the likelihood comparison, so the accuracy test scores only TT's logits.

For every position of the trace's final-layer stream, the HF final norm and tied LM head give the reference's
argmax, gold-token log-probability, top-k tokens and log-probabilities and tail mass. They are written to
``likelihood_reference.safetensors`` in the trace directory, about 50 MB for 262,144 positions, and are used while
they match the checkpoint and the document.

    python -m models.demos.gemma4_d_p.tests.scripts.prepare_likelihood_reference TRACE_DIR [--output PATH]
"""

import argparse
import time
from pathlib import Path

import torch
from loguru import logger

from models.demos.gemma4_d_p.tests.scripts.likelihood import (
    REFERENCE_SCORES,
    FinalHead,
    ReferenceScores,
    gpu_final_layer_stream,
    load_gpu_samples,
    reference_scores,
    save_reference_scores,
)


def prepare_likelihood_reference(trace_dir, output=None, head=None):
    trace_dir = Path(trace_dir)
    output = Path(output) if output else trace_dir / REFERENCE_SCORES
    _, metadata, stream = gpu_final_layer_stream(trace_dir)
    head = head or FinalHead()
    started = time.perf_counter()
    parts = []
    # One stream shard at a time keeps the hidden states in memory to a shard's worth.
    for chunk in stream["chunks"]:
        positions = torch.arange(chunk["row_start"], chunk["row_end"])
        parts.append(reference_scores(head, load_gpu_samples(trace_dir, positions)))
        logger.info(
            f"Reference scores for rows [{chunk['row_start']}, {chunk['row_end']}) "
            f"after {time.perf_counter() - started:.0f} s"
        )
    scores = ReferenceScores(*(torch.cat(field) for field in zip(*parts)))
    save_reference_scores(output, scores, head.identity, metadata["token_ids"])
    logger.info(f"Likelihood reference for {len(scores.positions)} positions: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path, help="Prepared GPU trace directory (or its source capture)")
    parser.add_argument(
        "--output", type=Path, help=f"Where to write the scores (default: TRACE_DIR/{REFERENCE_SCORES})"
    )
    args = parser.parse_args()
    prepare_likelihood_reference(args.trace_dir, args.output)


if __name__ == "__main__":
    main()
