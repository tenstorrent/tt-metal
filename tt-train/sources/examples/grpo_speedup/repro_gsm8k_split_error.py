#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproducer for the GSM8K NonMatchingSplitsSizesError.

Just calls ``datasets.load_dataset("gsm8k", "main", split="train")`` with
default verification and prints what comes back.

Expected failure (with current `datasets` defaults):

    datasets.utils.info_utils.NonMatchingSplitsSizesError:
      [{'expected': SplitInfo(name='train', num_examples=7473, ...),
        'recorded': SplitInfo(name='train', num_examples=14946, ...,
                              dataset_name='parquet')},
       {'expected': SplitInfo(name='test', num_examples=1319, ...),
        'recorded': SplitInfo(name='test',  num_examples=2638, ...,
                              dataset_name='parquet')}]

Run it with:

    python tt-train/sources/examples/grpo_speedup/repro_gsm8k_split_error.py
"""

from __future__ import annotations

import sys

import datasets


def main() -> int:
    print(f"datasets version: {datasets.__version__}")
    print('calling: datasets.load_dataset("gsm8k", "main", split="train")')
    try:
        ds = datasets.load_dataset("gsm8k", "main", split="train")
    except Exception as e:
        print(f"FAILED with {type(e).__name__}:")
        print(e)
        return 1

    print(f"OK  num_rows={len(ds)}  features={list(ds.features)}")
    print(f"first question: {ds[0]['question']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
