# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-weights forward pytest test for ``RMSNorm.update``.

RMSNorm computes ``y = (x / sqrt(mean(x^2) + eps)) * gamma``. After
``update`` zeros out ``gamma`` the multiplication collapses the entire
output to 0 regardless of ``x`` or ``eps``. We still set a small ``eps``
explicitly so a pathological zero-mean-zero-input wouldn't blow up the
intermediate division (paranoia; the random ``x`` we feed has nonzero
RMS in practice).

We exercise ``DistributedNorm.update`` -- the one-line passthrough into
``RMSNorm.update`` that the rest of the model actually calls -- so this
also covers the wrapper.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import as_update_input, open_completer

SEQ_LEN = 32  # one decode tile

# Small but nonzero eps so the 1/sqrt(...) in RMSNorm never sees a
# literal zero denominator even if the input happens to be all-zeros.
SMALL_EPS = 1e-12


@pytest.fixture(scope="module")
def completer_and_norm():
    with open_completer(dummy_weights=True) as completer:
        dn = completer.model.layers[0].attention_norm
        yield completer, dn, dn.norm


def _build_random_rms_input(completer):
    """Construct a synthetic ``(1, 1, SEQ_LEN, dim)`` RMSNorm input."""
    import ttnn

    dim = completer.model_args.dim
    return ttnn.from_torch(
        torch.randn(1, 1, SEQ_LEN, dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=completer.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(completer.mesh_device),
    )


def test_rmsnorm_forward_is_zero_when_gamma_is_zero(completer_and_norm):
    """Zero gamma via the ``DistributedNorm`` passthrough and check the
    forward output is elementwise zero (within bf16 noise)."""
    import ttnn

    from models.tt_transformers.tt.common import Mode

    completer, distributed_norm, rms = completer_and_norm

    gamma_hf = torch.zeros(completer.model_args.dim, dtype=torch.bfloat16)
    distributed_norm.update(weight=as_update_input(gamma_hf, completer.mesh_device))
    rms.eps = SMALL_EPS

    out = ttnn.to_torch(rms.forward(_build_random_rms_input(completer), Mode.PREFILL))

    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
        "RMSNorm.forward != 0 after zeroing gamma: "
        f"max|out|={float(out.abs().max()):.6g}, mean|out|={float(out.abs().mean()):.6g}"
    )
