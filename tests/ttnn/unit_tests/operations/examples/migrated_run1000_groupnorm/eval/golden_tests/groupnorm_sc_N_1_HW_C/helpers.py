# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for groupnorm_sc_N_1_HW_C golden tests (registry model).

Provides:
- pytorch_groupnorm_sc_N_1_HW_C: fp32 reference, result returned in input dtype.
- create_ttnn_input_tensor: single chokepoint for torch → ttnn conversion.
  Every test goes through this; do NOT call ttnn.from_torch directly.
- TOLERANCES: per-dtype thresholds keyed by ttnn dtype enum. Passed into
  the shared `check_output(..., tolerance=TOLERANCES[dtype])`.
- run_groupnorm_sc_N_1_HW_C: canonical entry point for test_golden.py.
  Keyword-only individual axes; **_ swallows tagger outputs and inputs-extras
  the runner doesn't act on. Output layout is ALWAYS TILE_LAYOUT —
  baked in at the runner level, not threaded through axes. Regression
  tests may pass a per-call `tolerance=(pcc, rms)` to widen the band
  for distributions where the default is too tight.

Shape/dtype/layout + PCC/RMS checking lives in `eval/metrics.py` —
`check_output` and `CheckOutputError` are imported from there.
"""

from __future__ import annotations

import torch
import ttnn

from eval.metrics import CheckOutputError, check_output  # noqa: F401  (re-export)
from ttnn import migrated_run1000_groupnorm as groupnorm_sc_N_1_HW_C  # type: ignore


# --- Reference ------------------------------------------------------------


def pytorch_groupnorm_sc_N_1_HW_C(
    input_tensor,
    num_groups,
    *,
    gamma=None,
    beta=None,
    eps=1e-5,
):
    """Reference GroupNorm for (N, 1, H*W, C) layout.

    Reshapes to PyTorch's (N, C, ...) expected by F.group_norm, applies,
    then reshapes back. Computed in fp32; result returned in input dtype.
    """
    original_dtype = input_tensor.dtype
    x = input_tensor.to(torch.float32)
    N, one, HW, C = x.shape

    # (N, 1, HW, C) → (N, C, HW)
    x_nchw = x.squeeze(1).permute(0, 2, 1)
    weight = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    bias = beta.to(torch.float32).reshape(C) if beta is not None else None
    result = torch.nn.functional.group_norm(
        x_nchw,
        num_groups,
        weight=weight,
        bias=bias,
        eps=eps,
    )
    # (N, C, HW) → (N, 1, HW, C)
    result = result.permute(0, 2, 1).unsqueeze(1)
    return result.to(original_dtype)


# --- ttnn dtype → torch dtype map ----------------------------------------
#
# dtype/layout flow as ttnn enums end-to-end. Only torch needs translation
# because it has no native bfloat8_b.

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}


def create_ttnn_input_tensor(tensor, device, *, dtype, layout, memory_config=None):
    """Single chokepoint for `torch.Tensor → ttnn.Tensor`. Used everywhere
    a golden test puts a tensor on device.

    Keeping this centralized matters once sharding lands: memory_config
    becomes axis-driven (DRAM_INTERLEAVED vs L1_HEIGHT_SHARDED vs
    L1_WIDTH_SHARDED, with shard spec coming from a input tagger), and
    every callsite needs the same translation. Don't bypass this with
    `ttnn.from_torch(...)` in test code.
    """
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


# --- Tolerances ----------------------------------------------------------

# Keyed by ttnn dtype enum. pcc = Pearson correlation; rms = RMS error
# normalized by the reference output's stddev (relative). Carried from
# old THRESHOLDS in test_golden_cross_product.py; bf8b inherits the
# layer_norm_rm bf8b tolerance as a starting point.
TOLERANCES = {
    ttnn.float32: (0.999, 0.01),
    ttnn.bfloat16: (0.995, 0.02),
    ttnn.bfloat8_b: (0.99, 0.10),
}


# --- run_groupnorm_sc_N_1_HW_C -------------------------------------------


def run_groupnorm_sc_N_1_HW_C(
    inputs,
    *,
    dtype,
    layout,
    affine,
    affine_dtype,
    affine_layout,
    num_groups,
    device,
    eps=1e-5,
    extras=None,
    **_,
):
    """Build tensors and dispatch groupnorm_sc_N_1_HW_C. Raises
    CheckOutputError on miss.

    Individual axes are keyword-only — call as
    `run_groupnorm_sc_N_1_HW_C(inputs, device=device, **axes)` from
    test_golden.py. The **_ swallows tagger outputs (`alignment`) and any
    extras the runner doesn't act on.

    `extras` is the loose-case override dict (precision thresholds, input
    distribution, etc.). Cartesian-generated cases pass `extras=None`.
    Currently unused — placeholder so loose tests can carry runner overrides
    through the parametrize without breaking the call signature.

    `affine_dtype` / `affine_layout` apply to both gamma and beta. They
    are independent of input dtype/layout — supports mixed-precision
    (bf16 input + fp32 weights). For `affine=no_affine`, the op file's
    INVALID block canonicalizes (affine_dtype, affine_layout) to a single
    cell, so we never see redundant no_affine variants here.

    `num_groups` is the per-shape coupled value — it lives in inputs[1]
    and the op's tag_num_groups tagger projects it onto the num_groups
    axis. Arrives here via `**axes` like any other tagged axis.

    `inputs` is `(shape, num_groups)` — shape is the (N, 1, H*W, C) input
    tensor shape; the runner unpacks `shape = inputs[0]` for tensor build.
    """
    shape = inputs[0]
    N, one, HW, C = shape
    torch_dtype = _TORCH_DTYPE[dtype]
    # affine_dtype is the "none" sentinel when affine=no_affine — no weight
    # tensor is built, so the dtype lookup must be skipped (not a real dtype).
    torch_affine_dtype = None if affine == "no_affine" else _TORCH_DTYPE[affine_dtype]

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)

    if affine in ("gamma_beta", "gamma_only"):
        torch_gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch_affine_dtype)
        ttnn_gamma = create_ttnn_input_tensor(
            torch_gamma,
            device,
            dtype=affine_dtype,
            layout=affine_layout,
        )
    else:
        torch_gamma = None
        ttnn_gamma = None

    if affine == "gamma_beta":
        torch_beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch_affine_dtype)
        ttnn_beta = create_ttnn_input_tensor(
            torch_beta,
            device,
            dtype=affine_dtype,
            layout=affine_layout,
        )
    else:
        torch_beta = None
        ttnn_beta = None

    expected = pytorch_groupnorm_sc_N_1_HW_C(
        torch_input,
        num_groups,
        gamma=torch_gamma,
        beta=torch_beta,
        eps=eps,
    )

    ttnn_input = create_ttnn_input_tensor(torch_input, device, dtype=dtype, layout=layout)
    ttnn_output = groupnorm_sc_N_1_HW_C(
        ttnn_input,
        num_groups,
        gamma=ttnn_gamma,
        beta=ttnn_beta,
        eps=eps,
    )

    # Output dtype/layout: dtype follows input; layout is always TILE.
    check_output(
        ttnn_output,
        expected,
        shape=shape,
        dtype=dtype,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[dtype],
    )
