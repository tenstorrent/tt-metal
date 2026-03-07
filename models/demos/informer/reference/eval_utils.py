# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class SplitConfig:
    train_len: int
    val_len: int
    test_len: int


def default_etth1_splits() -> SplitConfig:
    return SplitConfig(
        train_len=12 * 30 * 24,
        val_len=4 * 30 * 24,
        test_len=4 * 30 * 24,
    )


def iter_windows(
    total_length: int,
    *,
    seq_len: int,
    pred_len: int,
    stride: int,
    max_windows: int | None = None,
) -> Iterable[int]:
    total_window = seq_len + pred_len
    count = 0
    for start in range(0, total_length - total_window + 1, stride):
        yield start
        count += 1
        if max_windows is not None and count >= max_windows:
            break


def resolve_eval_range(
    total_length: int,
    *,
    split: str,
    split_cfg: SplitConfig,
    start: int | None = None,
    end: int | None = None,
) -> tuple[int, int]:
    if start is not None or end is not None:
        eval_start = 0 if start is None else start
        eval_end = total_length if end is None else end
    else:
        split = split.lower()
        if split == "full":
            eval_start = 0
            eval_end = total_length
        elif split == "train":
            eval_start = 0
            eval_end = min(split_cfg.train_len, total_length)
        elif split == "val":
            eval_start = min(split_cfg.train_len, total_length)
            eval_end = min(eval_start + split_cfg.val_len, total_length)
        elif split == "test":
            eval_start = min(split_cfg.train_len + split_cfg.val_len, total_length)
            eval_end = min(eval_start + split_cfg.test_len, total_length)
        else:
            raise ValueError(f"Unsupported split: {split}.")

    if eval_start < 0 or eval_end > total_length or eval_start >= eval_end:
        raise ValueError(f"Invalid eval range [{eval_start}, {eval_end}) for length {total_length}.")
    return eval_start, eval_end
