# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Smallest known reproducer of the Blackhole topk_xl split failure that needs instruction gathering.

One case of test_topk_xl_split.py: K=512, 2 chunks, an 8-element tail, 1 row, ascending last-chunk rebuild.
On Blackhole silicon it fails with gathering on (the default) and passes with TT_METAL_DISABLE_GATHERING=1;
both builds are the same kernel source. Only the split kernel runs (no single-thread reference run), so a
simulator run stays short. The failure is sensitive to PACK code layout: see
tests/repro/topk_xl_split_gathering/RUNBOOK.md before changing any source this kernel compiles.
"""

import os

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.golden_generators import TopKXLGolden, get_golden_generator
from test_topk_xl import _extract_hw_topk
from test_topk_xl_split import _config

pytestmark = [skip_for_wormhole, skip_for_quasar]

# (K, num_chunks, tail_elements, num_rows, mode, fp32 input, seg_base, ascending)
CASE = (512, 2, 8, 1, "positive", False, 0, True)


def test_topk_xl_split_gathering_repro():
    K = CASE[0]
    gathering = os.environ.get("TT_METAL_DISABLE_GATHERING") != "1"
    config, rows, _ = _config("sources/topk_xl_split_test.cpp", CASE)
    result = config.run().result

    gold = set(int(x) for x in get_golden_generator(TopKXLGolden)(rows, K)[0].tolist())
    ((hw_idx, hw_val),) = _extract_hw_topk(result, K, rows.shape[0])
    got = [int(x) for x in hw_idx.tolist()]
    missing, extra = sorted(gold - set(got)), sorted(set(got) - gold)
    pair_errors = sum(float(rows[0][i]) != float(v) for i, v in zip(got, hw_val.tolist()))
    # The repro signature on silicon with gathering on: missing [516, 517], extra [3, 15], 0 pair errors.
    print(
        f"gathering={'on' if gathering else 'off'} lanes={len(got)} missing={missing} extra={extra} "
        f"pair_errors={pair_errors}"
    )
    assert len(got) == K and not missing and not extra and pair_errors == 0, (
        f"top-K wrong with gathering {'on' if gathering else 'off'}: missing {missing}, extra {extra}, "
        f"{pair_errors} value/index pair errors"
    )
