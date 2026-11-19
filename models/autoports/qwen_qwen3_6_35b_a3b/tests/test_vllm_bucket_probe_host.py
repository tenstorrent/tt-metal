# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate exact bucket-probe comparisons without importing device code."""

import ast
from pathlib import Path

import pytest
import torch


def comparator():
    path = Path(__file__).with_name("run_vllm_bucket_probe.py")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compare_tree"]
    scope = {"torch": torch}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["compare_tree"]


def test_exact_comparison_detects_one_bit_on_last_chip_of_dormant_slot():
    reference = {"linear": {"23": {"recurrent": [torch.zeros(4, dtype=torch.int16) for _ in range(4)]}}}
    candidate = {"linear": {"23": {"recurrent": [tensor.clone() for tensor in reference["linear"]["23"]["recurrent"]]}}}
    compare = comparator()
    compare(reference, candidate)
    candidate["linear"]["23"]["recurrent"][3][-1] = 1
    with pytest.raises(  # allow-pytest.raises: CPU-only validation
        AssertionError, match=r"state.linear.23.recurrent\[3\].*first_flat_index=3"
    ):
        compare(reference, candidate)


def test_full_vocabulary_comparison_checks_last_logit_without_tolerance():
    reference = torch.zeros((2, 248320))
    candidate = reference.clone()
    candidate[-1, -1] = torch.finfo(torch.float32).eps
    with pytest.raises(AssertionError, match="1 differing elements"):  # allow-pytest.raises: CPU-only validation
        comparator()(reference, candidate, "logits")


@pytest.mark.parametrize("candidate", [torch.zeros(2, dtype=torch.int32), torch.zeros(1, 2)])
def test_comparison_rejects_changed_tensor_metadata(candidate):
    with pytest.raises(AssertionError, match="tensor metadata differs"):  # allow-pytest.raises: CPU-only validation
        comparator()(torch.zeros(2), candidate)
