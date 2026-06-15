# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared PCC (Pearson Correlation Coefficient) assertion utilities.

CRITICAL: The framework's compare_fn_outputs() at core/utils.py only prints
warnings when PCC < 0.999 -- it NEVER raises or asserts. Tests using only
compare_fn_outputs will silently pass even with catastrophically wrong outputs.
Use assert_pcc() instead.
"""

import torch

# Self-contained: the framework's TorchTTNNTensor is NOT a dependency of this
# vendored bench suite. The harnesses pass plain torch.Tensors to assert_pcc /
# compute_pcc, so the TorchTTNNTensor branch is unreachable here. A sentinel
# class (never instantiated) keeps the isinstance() check below well-defined
# without importing tt_symbiote.
class TorchTTNNTensor:  # pragma: no cover - sentinel, never produced here
    pass


def _extract_tensors(output, force_readback=False):
    """Flatten an output into a list of torch.Tensors.

    Handles single tensors, TorchTTNNTensor instances, and arbitrarily
    nested lists/tuples. ``force_readback`` clears the cached ``elem``
    on TorchTTNNTensors so the next ``to_torch`` access reads from the
    TTNN device rather than returning a stale CPU copy.
    """
    tensors = []
    if isinstance(output, TorchTTNNTensor):
        if force_readback:
            output.elem = None
        tensors.append(output.to_torch)  # to_torch is a @property, not a method
    elif isinstance(output, torch.Tensor):
        tensors.append(output)
    elif isinstance(output, (list, tuple)):
        for item in output:
            tensors.extend(_extract_tensors(item, force_readback=force_readback))
    elif output is None:
        pass
    return tensors


def compute_pcc(actual, expected):
    """Compute PCC between actual (TTNN) and expected (PyTorch) outputs.

    Returns a list of ``(pcc_value, max_abs_diff)`` pairs, one per output
    tensor pair. Mismatched output counts raise immediately.
    """
    actual_tensors = _extract_tensors(actual, force_readback=True)
    expected_tensors = _extract_tensors(expected, force_readback=False)

    assert len(actual_tensors) == len(expected_tensors), (
        f"Mismatched output count: {len(actual_tensors)} (actual) " f"vs {len(expected_tensors)} (expected)"
    )

    results = []
    for a, e in zip(actual_tensors, expected_tensors):
        a = a.to(torch.float32).flatten()
        e = e.to(torch.float32).flatten()
        assert a.shape == e.shape, f"Shape mismatch: {a.shape} vs {e.shape}"
        if a.numel() <= 1:
            pcc = torch.tensor(1.0) if torch.allclose(a, e) else torch.tensor(0.0)
        else:
            pcc = torch.corrcoef(torch.stack([a, e]))[0, 1]
        max_diff = torch.max(torch.abs(a - e)).item()
        results.append((pcc.item(), max_diff))
    return results


def assert_pcc(actual, expected, threshold=0.99, msg=""):
    """Assert PCC >= threshold between actual (TTNN) and expected (PyTorch) outputs.

    Args:
        actual: TTNN output (TorchTTNNTensor, torch.Tensor, or nested collection).
        expected: PyTorch reference output (same structure).
        threshold: Minimum acceptable PCC (default 0.99; pass 0.999 for bring-up).
        msg: Optional message prefix for assertion errors.
    """
    results = compute_pcc(actual, expected)
    prefix = f"{msg}: " if msg else ""
    assert len(results) > 0, f"{prefix}No output tensors to compare"
    for i, (pcc, max_diff) in enumerate(results):
        assert not torch.tensor(pcc).isnan(), f"{prefix}output[{i}]: PCC is NaN (max_abs_diff={max_diff:.6f})"
        assert pcc >= threshold, f"{prefix}output[{i}]: PCC {pcc:.6f} < {threshold} " f"(max_abs_diff={max_diff:.6f})"
