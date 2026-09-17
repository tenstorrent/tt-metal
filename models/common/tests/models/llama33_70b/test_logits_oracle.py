# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from models.common.tests.models.llama33_70b.logits_oracle import assert_rowwise_logits_parity


def _logits(rows: int = 15, vocab: int = 4096) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(rows, 1, vocab, generator=generator)
    logits[:, :, 0] = 10.0
    return logits


def test_accepts_correlated_logits_with_exact_top1_and_bounded_error():
    expected = _logits()
    generator = torch.Generator().manual_seed(23)
    actual = expected + 0.005 * torch.randn(expected.shape, generator=generator)

    assert_rowwise_logits_parity(actual, expected, min_row_pcc=0.9999, max_abs=1.0)


def test_rejects_one_corrupted_row_even_when_global_pcc_is_high(expect_error):
    expected = _logits()
    actual = expected.clone()
    generator = torch.Generator().manual_seed(29)
    actual[7] += 0.1 * torch.randn(actual[7].shape, generator=generator)

    global_pcc = torch.corrcoef(torch.stack((actual.flatten(), expected.flatten())))[0, 1]
    assert global_pcc > 0.999
    with expect_error(AssertionError, r"row PCC below 0.9999: row 7"):
        assert_rowwise_logits_parity(actual, expected, min_row_pcc=0.9999, max_abs=1.0)


def test_rejects_sparse_large_error_that_pcc_can_hide(expect_error):
    expected = _logits(vocab=131072)
    actual = expected.clone()
    actual[3, 0, 100] += 1.125

    with expect_error(AssertionError, r"row max-abs above 1.0: row 3"):
        assert_rowwise_logits_parity(actual, expected, min_row_pcc=0.9999, max_abs=1.0)


def test_rejects_top1_change_with_small_numeric_error(expect_error):
    expected = _logits()
    expected[2, 0, 0] = 4.0
    expected[2, 0, 1] = 3.9
    actual = expected.clone()
    actual[2, 0, 1] = 4.1

    with expect_error(AssertionError, r"top-1 mismatch"):
        assert_rowwise_logits_parity(actual, expected, min_row_pcc=0.9999, max_abs=1.0)


def test_geometry_policy_accepts_near_tie_top1_flip_with_topk_preserved():
    expected = _logits()
    expected[2, 0, :5] = torch.tensor([4.0, 3.9, 3.8, 3.7, 3.6])
    actual = expected.clone()
    actual[2, 0, 1] = 4.1

    assert_rowwise_logits_parity(
        actual,
        expected,
        min_row_pcc=0.999,
        max_abs=1.0,
        require_exact_top1=False,
        max_top1_mismatches=1,
        expected_top1_in_actual_topk=5,
        min_topk_overlap=4,
        isclose_atol=0.25,
        isclose_rtol=0.05,
        max_isclose_failure_fraction=0.005,
    )


def test_geometry_policy_rejects_lost_reference_top1(expect_error):
    expected = _logits()
    actual = expected.clone()
    actual[4, 0, :6] = torch.tensor([4.0, 4.1, 4.2, 4.3, 4.4, 4.5])

    with expect_error(AssertionError, r"expected top-1 missing from actual top-5 at rows \[4\]"):
        assert_rowwise_logits_parity(
            actual,
            expected,
            min_row_pcc=0.99,
            max_abs=10.0,
            require_exact_top1=False,
            max_top1_mismatches=1,
            expected_top1_in_actual_topk=5,
            min_topk_overlap=4,
            isclose_atol=0.25,
            isclose_rtol=0.05,
            max_isclose_failure_fraction=0.005,
        )
