# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Step information analysis script for tt-train logs.
Parses step loss and step time.
"""

import argparse
import re
from typing import Dict, List, Optional
import numpy as np


def find_step_summaries(content: str) -> List[Dict[str, int | float]]:
    """Parse tt-train log content and extract step number, loss, and step time per step.

    Uses regex to find "Step: N", "Loss: X.Y", and "Full step time X.Y ms" in order.
    All three match lists must have the same length (one entry per step).

    Args:
        content: Raw log file text.

    Returns:
        List of dicts, each with keys "step", "loss", "step_time" (int/float).

    Raises:
        ValueError: If the counts of step/loss/step_time matches differ.
    """
    step_pattern = r"Step:\s*(\d+)"
    loss_pattern = r"Loss:\s*([\d.]+)"
    step_time_pattern = r"Full step time\s+([\d.]+)\s*ms"

    step_matches = re.findall(step_pattern, content, re.DOTALL)
    loss_matches = re.findall(loss_pattern, content, re.DOTALL)
    step_time_matches = re.findall(step_time_pattern, content, re.DOTALL)

    step_len, loss_len, step_time_len = (
        len(step_matches),
        len(loss_matches),
        len(step_time_matches),
    )
    if not (step_len == loss_len == step_time_len):
        raise ValueError(
            f"Length of pattern matches not equal. step: {step_len}, loss: {loss_len}, step_time: {step_time_len}"
        )

    step_summary = []
    for step, loss, step_time in zip(step_matches, loss_matches, step_time_matches):
        step_summary.append(
            {"step": int(step), "loss": float(loss), "step_time": float(step_time)}
        )

    return step_summary


def analyze_step_summary(summary: List[Dict[str, int | float]]) -> Dict[str, float]:
    """Compute and print step metrics from a list of step records.

    Metrics: last loss, and average step time (excluding the first two steps
    to avoid warmup). Prints a short summary to stdout.

    Args:
        summary: List of step dicts from find_step_summaries (keys: step, loss, step_time).

    Returns:
        Dict with "last_loss" and "average_iteration_time_ms".
    """
    last_loss = summary[-1]["loss"]
    # Exclude first two steps to avoid warmup/cold-start bias in timing.
    average_step_time = np.mean([step_info["step_time"] for step_info in summary[2:]])

    print("\n--- Step Information ---")
    print(f"  Total steps:   {len(summary)}")
    print(f"  Last loss:   {last_loss:,.2f}")
    print(
        f"  Average iteration time excluding first two steps:   {average_step_time:,.2f} ms"
    )

    breakdown = {"last_loss": last_loss, "average_iteration_time_ms": average_step_time}

    return breakdown


def main(raw_args: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
    parser = argparse.ArgumentParser(
        description="Analyze step loss and step time from tt-train logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--logs",
        required=True,
        help="Path to log file containing step information",
    )

    args = parser.parse_args(raw_args)

    with open(args.logs, "r") as f:
        content = f.read()

    summaries = find_step_summaries(content)

    if not summaries:
        raise ValueError("No step information found in the log file")

    print(f"Found {len(summaries)} step information summary/summaries")

    breakdown = analyze_step_summary(summaries)

    print(f"\n{'='*80}")
    print("Analysis complete")
    print(f"{'='*80}\n")

    return breakdown if breakdown else None


if __name__ == "__main__":
    main()
