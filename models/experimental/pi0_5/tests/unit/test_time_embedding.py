# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
No-device CPU smoke tests for the Pi0.5 flow-matching time embedding and the
gated-residual helper.

``create_sinusoidal_pos_embedding`` turns a scalar flow-matching timestep into
the sinusoidal feature vector that feeds the suffix time MLP; ``_gated_residual``
implements the adaRMS gated skip connection (Pi0.5) vs. the plain residual
(Pi0). Both are pure functions and run on CPU.
"""

import pytest
import torch

from models.experimental.pi0_5.reference.torch_gemma import _gated_residual
from models.experimental.pi0_5.reference.torch_suffix import create_sinusoidal_pos_embedding


@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 256)])
def test_sinusoidal_embedding_shape_and_determinism(batch, dim):
    t = torch.full((batch,), 0.3)
    emb1 = create_sinusoidal_pos_embedding(t, dim)
    emb2 = create_sinusoidal_pos_embedding(t, dim)
    assert emb1.shape == (batch, dim)
    assert torch.equal(emb1, emb2)  # deterministic
    assert torch.isfinite(emb1).all()
    assert emb1.dtype == t.dtype


def test_sinusoidal_embedding_is_time_dependent():
    dim = 128
    early = create_sinusoidal_pos_embedding(torch.tensor([0.0]), dim)
    late = create_sinusoidal_pos_embedding(torch.tensor([1.0]), dim)
    assert not torch.allclose(early, late)


def test_sinusoidal_embedding_odd_dimension_raises():
    with pytest.raises(ValueError):
        create_sinusoidal_pos_embedding(torch.tensor([0.5]), dimension=63)


def test_gated_residual_zero_gate_is_identity():
    x = torch.randn(1, 4, 16)
    y = torch.randn(1, 4, 16)
    gate = torch.zeros(1, 1, 16)
    assert torch.allclose(_gated_residual(x, y, gate), x)


def test_gated_residual_none_gate_is_plain_sum():
    x = torch.randn(1, 4, 16)
    y = torch.randn(1, 4, 16)
    assert torch.allclose(_gated_residual(x, y, None), x + y)
