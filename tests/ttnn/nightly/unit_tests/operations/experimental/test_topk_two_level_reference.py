# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the C1 oracle and the input generators (no device)."""

import pytest
import torch

from tests.ttnn.nightly.unit_tests.operations.experimental.topk_two_level_reference import (
    GENERATORS,
    assert_topk_exact,
    exact_topk,
    generate,
    normalize,
)

LENGTHS = [65_536, 65_537, 524_304, 1_048_576]
SEED = 0
ROWS = 4


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("kind", GENERATORS)
def test_oracle_accepts_the_exact_topk(kind, n):
    rows = generate(kind, ROWS, n, SEED, n=n)
    for r in range(ROWS):
        assert_topk_exact(rows[r], exact_topk(rows[r], n), n)


def test_normalize_matches_device_zero_class():
    # The device ranks every subnormal of either sign and -0 equal to +0, between -min_normal and +min_normal;
    # test_single_level_load_order_zero_class checks that on device.
    words = torch.tensor([0x0001, 0x8001, 0x007F, 0x8000, 0x0000, 0x0080, 0x8080], dtype=torch.int32)
    values = (words << 16).view(torch.float32).to(torch.bfloat16)
    x = normalize(values)
    assert x[:5].eq(0).all()
    assert x[5] > 0 and x[6] < 0


def test_oracle_rejects_a_wrong_answer(expect_error):
    row = generate("iid", 1, 70_000, 0)[0]
    good = exact_topk(row, 70_000)
    bad = good.clone()
    bad[5] = int(torch.argmin(row.float()))
    with expect_error(AssertionError, "not descending|value multiset differs"):
        assert_topk_exact(row, bad, 70_000)
