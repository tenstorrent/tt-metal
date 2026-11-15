# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real-prompt workloads for the model-level host / perf probes (P.63 / P.64 / P.66 tests): the novel prompt (the
ladder's ids) and a code prompt, SEQ tokens each, and the max |a - b| of two device tensors. Moved here from the
removed MoE-replica tests (FIX.33: tests/test_moe_counts.py _ids, tests/test_moe_prev_replicas.py _maxdiff)."""

from __future__ import annotations

import copy

import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.reference import prompt as P
from models.demos.common.bringup.testing.harness import spec

S = spec()
SEQ = 56320
CODE_TEXT = "/scratch/dnijemcevic/moeplace/code.txt"
CODE_REQUEST = "Print the C++ source of tt-metal TTNN operations, file by file."
NOVEL_REQUEST = "Recite the Project Gutenberg text of A Tale of Two Cities by Charles Dickens, from the start."


def prompt_ids(wl: str) -> torch.Tensor:
    """SEQ token ids of workload ``wl`` ("novel" | "code")."""
    if wl == "novel":  # (the ladder's ids: its goldens are made from this file; the .bz2 source is not in the tree)
        rec = P.load(S)
        assert rec is not None, f"no prompt file {P.prompt_path(S)}"
        assert rec.get("request") == NOVEL_REQUEST and rec.get("wrap") == "model_turn", (
            rec.get("wrap"),
            rec["request"],
        )
        ids = rec["token_ids"]
    else:
        from models.demos.common.bringup.reference.golden import tokenizer

        sc = copy.copy(S)
        sc.data = copy.deepcopy(S.data)
        sc.data["text"].update(source=CODE_TEXT, request=CODE_REQUEST, wrap="model_turn")
        ids, info = P.build_ids(sc, tokenizer(S), SEQ)
        logger.info(f"code prompt: {info['source']} request {info['request']!r}, {info['prefix_tokens']} prefix tokens")
    assert len(ids) >= SEQ, (wl, len(ids))
    return torch.tensor(list(ids[:SEQ]))


def maxdiff(a: ttnn.Tensor, b: ttnn.Tensor) -> float:
    """max |a - b| over every chip (a device reduction, one scalar a chip read back)."""
    d = ttnn.max(ttnn.abs(ttnn.sub(a, b)))
    v = max(float(ttnn.to_torch(t).float().abs().max()) for t in ttnn.get_device_tensors(d))
    ttnn.deallocate(d)
    return v
