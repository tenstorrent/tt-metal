# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verifier-added checks for chunk_gated_delta_rule_fwd (gaps found in review, Phase 0).

* Sharded output placement is a documented mechanism cap (interleaved I/O only): it must be refused
  with ValueError before any dispatch, not allocated and scattered through an interleaved address map.
* The float32 + fp32_dest_acc_en=False hard refusal (op_design.md -> Parameters).
"""

from __future__ import annotations

import torch

import ttnn
from ttnn.operations.chunk_gated_delta_rule_fwd import chunk_gated_delta_rule_fwd

from eval.golden_tests.chunk_gated_delta_rule_fwd.helpers import make_reference_inputs, quantize


def _small(device, dtype=ttnn.float32):
    ref = quantize(make_reference_inputs((1, 64, 2, 32, 32), "no_h0", seed=0), dtype)
    return {
        n: (None if t is None else ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))
        for n, t in ref.items()
    }


def _run(dev, **kw):
    return chunk_gated_delta_rule_fwd(dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"], chunk_size=32, **kw)


def test_rejects_sharded_output_memory_config(device, expect_error):
    dev = _small(device)
    sharded = ttnn.create_sharded_memory_config(
        (32, 32),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    with expect_error(ValueError, ""):
        _run(dev, memory_config=sharded)


def test_rejects_fp32_without_fp32_dest_acc(device, expect_error):
    dev = _small(device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    with expect_error(ValueError, ""):
        _run(dev, compute_kernel_config=cfg)


def test_bf16_without_fp32_dest_acc_runs(device):
    """bfloat16 with a caller-chosen 16-bit DEST is honored (not refused) and stays finite."""
    dev = _small(device, ttnn.bfloat16)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    o = ttnn.to_torch(_run(dev, compute_kernel_config=cfg)[0]).float()
    assert torch.isfinite(o).all()
