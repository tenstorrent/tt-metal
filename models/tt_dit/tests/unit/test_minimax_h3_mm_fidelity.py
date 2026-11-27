# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the matmul fidelities an H3 block and its attention pick. Each is built with its
sub-layers replaced by recorders, so only the module's own choices run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ttnn
from models.tt_dit.models.transformers.minimax_h3 import attention_minimax_h3 as am
from models.tt_dit.models.transformers.minimax_h3 import transformer_block_minimax_h3 as tb

SDPA_FLAG = "MINIMAX_H3_SDPA_MM_FIDELITY"

LOFI, HIFI2 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2


def _stub(monkeypatch, module, *names):
    for name in names:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda *args, **kwargs: None)
    axis = SimpleNamespace(mesh_axis=1, factor=4)
    return dict(
        mesh_device=SimpleNamespace(arch=lambda: None, compute_with_storage_grid_size=lambda: ttnn.CoreCoord(11, 10)),
        ccl_manager=SimpleNamespace(topology=ttnn.Topology.Ring),
        parallel_config=SimpleNamespace(tensor_parallel=axis, sequence_parallel=axis),
    )


def _block(monkeypatch, *, adaln_curve: bool = False):
    """A block for a pruned (`adaln_curve`) or a full checkpoint, and the kwargs it gave its attention."""
    seen = {}
    mesh = _stub(monkeypatch, tb, "DistributedRMSNorm", "ParallelFeedForward", "ColParallelLinear")
    monkeypatch.setattr(tb, "MiniMaxH3Attention", lambda **kwargs: seen.update(kwargs))
    block = tb.MiniMaxH3TransformerBlock(
        hidden_size=8, num_heads=4, head_dim=2, ffn_dim=16, time_embed_dim=8, adaln_curve=adaln_curve, **mesh
    )
    return block, seen


def _attention(monkeypatch, **kwargs):
    mesh = _stub(monkeypatch, am, "ColParallelLinear", "DistributedRMSNorm", "bf16_tensor")
    monkeypatch.setattr(am, "is_blackhole", lambda: False)
    return am.MiniMaxH3Attention(hidden_size=8, num_heads=4, head_dim=2, **mesh, **kwargs)


@pytest.mark.parametrize(("adaln_curve", "expected"), [(True, LOFI), (False, None)])
def test_ring_sdpa_default(monkeypatch, adaln_curve, expected):
    """Pruned (10Eros) checkpoints default the ring SDPA matmuls to LoFi; the others leave the op's HiFi2."""
    assert _block(monkeypatch, adaln_curve=adaln_curve)[1]["sdpa_matmul_fidelity"] == expected


def test_ring_sdpa_flag_beats_default(monkeypatch):
    """hifi2 in the flag is the way back from the block's LoFi default."""
    monkeypatch.delenv(SDPA_FLAG, raising=False)
    assert _attention(monkeypatch, sdpa_matmul_fidelity=LOFI).sdpa_matmul_fidelity == LOFI
    monkeypatch.setenv(SDPA_FLAG, "hifi2")
    assert _attention(monkeypatch, sdpa_matmul_fidelity=LOFI).sdpa_matmul_fidelity == HIFI2
