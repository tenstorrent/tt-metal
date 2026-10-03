# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar fork of tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py.

Runs the same windowed (block-diagonal) SDPA cases against
ttnn.experimental.quasar.transformer.scaled_dot_product_attention by swapping the op entry point
for each test in this module (see _quasar_sdpa). Those reused tests upload tensors with a device-side tilize, which
Quasar does not run; test_windowed_sdpa_full_chunk_masked below runs on Quasar as well as WH/BH.
"""

import pytest
import torch

import ttnn
from ttnn.experimental.quasar.transformer import scaled_dot_product_attention

from tests.ttnn.unit_tests.operations.sdpa.test_windowed_sdpa import *  # noqa: F401,F403
from tests.ttnn.unit_tests.operations.sdpa.test_windowed_sdpa import windowed_mask
from tests.ttnn.unit_tests.operations.sdpa import test_windowed_sdpa as _mainline
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.test_sdpa_attention_sink import (
    _check,
    _compute_kernel_config,
    _to_device,
)


@pytest.fixture(autouse=True)
def _quasar_sdpa(monkeypatch):
    # Point the reused tests at the Quasar fork for each test in this module only; monkeypatch restores the
    # public op afterwards, so other modules in the same pytest process are unaffected.
    monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", scaled_dot_product_attention)


# Quasar does not support bfloat8_b, so the reused smoke test's bf8 parametrizations run in bfloat16
# (same ids and shapes, bf16 PCC threshold), as the quasar sdpa_decode fork does for its bfp8 cases.
def test_windowed_sdpa_smoke(
    device, dtype, pcc_threshold, num_heads, seq_len, chunk, cu_window_seqlens, fp32_dest_acc_en
):
    if dtype == ttnn.bfloat8_b:
        dtype, pcc_threshold = ttnn.bfloat16, 0.99
    _mainline.test_windowed_sdpa_smoke(
        device, dtype, pcc_threshold, num_heads, seq_len, chunk, cu_window_seqlens, fp32_dest_acc_en
    )


test_windowed_sdpa_smoke.pytestmark = _mainline.test_windowed_sdpa_smoke.pytestmark


@pytest.mark.parametrize(
    "seq_len, q_chunk, k_chunk, cu_window_seqlens",
    [
        # Each 32-row Q chunk lies inside one window: no row is masked across a whole K chunk.
        (128, 32, 32, [0, 64, 128]),
        # The 128-row Q chunk spans both windows: rows 64-127 are masked across all of K chunk 0.
        (128, 128, 32, [0, 64, 128]),
        # The 64-row Q chunk straddles an unaligned boundary: rows 48-63 are masked across K chunk 0.
        (128, 64, 32, [0, 48, 128]),
    ],
    ids=["control_q32", "q128_k32_aligned", "q64_k32_straddle"],
)
def test_windowed_sdpa_full_chunk_masked(device, seq_len, q_chunk, k_chunk, cu_window_seqlens):
    """A row masked across an entire K chunk must not turn into NaN.

    With -inf in the mask, such a row's running max is -inf and exp(-inf - -inf) is NaN on Quasar;
    the Quasar mask uses a large finite negative instead.
    """
    torch.manual_seed(42)
    num_heads, head_dim = 1, 64
    scale = head_dim**-0.5
    q, k, v = (torch.randn(1, num_heads, seq_len, head_dim).bfloat16() for _ in range(3))

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    cu_tt = _to_device(
        torch.tensor(cu_window_seqlens, dtype=torch.int32), device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        _to_device(q, device),
        _to_device(k, device),
        _to_device(v, device),
        is_causal=False,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=_compute_kernel_config(),
        cu_window_seqlens=cu_tt,
    )
    out = ttnn.to_torch(out)[:, :, :seq_len, :].float()

    nan_rows = torch.isnan(out).any(dim=-1)[0, 0].nonzero().flatten().tolist()
    assert not nan_rows, f"NaN in rows {nan_rows}"
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=windowed_mask(seq_len, cu_window_seqlens), scale=scale
    )
    _check(gt, out)
