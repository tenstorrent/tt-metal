# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity check of two ``e2e_outputs.safetensors`` dumps (``tests/e2e/test_model.py``).

Used to show a code change leaves the text-only path unchanged: every tensor (per row probs,
logits, final hidden, last-token hidden after every layer) must be bit-identical.

Usage::

    python -m models.demos.pplx_decider_v1_27b.tests.e2e.compare_outputs BEFORE.safetensors AFTER.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from safetensors.torch import load_file


def compare(before: str, after: str) -> dict:
    a, b = load_file(before), load_file(after)
    differing = sorted(k for k in set(a) & set(b) if not torch.equal(a[k], b[k]))
    return {
        "tensors_before": len(a),
        "tensors_after": len(b),
        "missing_after": sorted(set(a) - set(b)),
        "extra_after": sorted(set(b) - set(a)),
        "bit_identical": sum(1 for k in set(a) & set(b) if torch.equal(a[k], b[k])),
        "differing": differing,
        "max_abs_diff": {k: float((a[k].float() - b[k].float()).abs().max()) for k in differing},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    result = compare(args.before, args.after)
    print(json.dumps(result, indent=2))
    ok = not (result["differing"] or result["missing_after"] or result["extra_after"])
    print("BIT_IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
