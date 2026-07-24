# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ttml.modules.VocabParallelEmbedding``.

The oracle is a plain replicated ``Embedding`` holding the *identical* weight
and fed the *identical* ids. Vocab-parallelism only adds exact operations on top
of the same gather — vocabulary sharding, a 1.0/0.0 ownership mask, and a
sum all-reduce over zeros — so the two must agree.

Mesh: ``[1, 2]`` with axes ``("dp", "tp")``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pytest

import ttnn
import ttml
from ttml.modules import Embedding, VocabParallelEmbedding


pytestmark = pytest.mark.requires_device

TP_AXIS_SIZE = 2
MESH_SHAPE = (1, TP_AXIS_SIZE)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", MESH_SHAPE): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_1_2_line_line.textproto"),
    ("wormhole_b0", MESH_SHAPE): os.path.join(_REPO_ROOT, "configs", "mgd", "n300_1_2_line_line.textproto"),
}


# ---------------------------------------------------------------------------
# Multi-device mesh fixture (same shape/skip conventions as test_fsdp.py)
# ---------------------------------------------------------------------------
def _detect_arch() -> Optional[str]:
    try:
        name = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    if "blackhole" in name:
        return "blackhole"
    if "wormhole_b0" in name:
        return "wormhole_b0"
    return None


def _close_device_mesh_quietly() -> None:
    """Reverse ``open_device_mesh`` (close device, disable fabric, clear the global mesh),
    swallowing errors so it is safe on the pre-open and teardown paths."""
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


def _ensure_mgd_path(shape: tuple[int, ...]) -> Optional[str]:
    """Point ``TT_MESH_GRAPH_DESC_PATH`` at a bundled MGD if unset; return the old value."""
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if previous:
        return previous
    arch = _detect_arch()
    if arch is None:
        return previous
    candidate = _MGD_FOR_ARCH_AND_SHAPE.get((arch, shape))
    if candidate and os.path.isfile(candidate):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = candidate
    return previous


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


@pytest.fixture(scope="module")
def tp_mesh():
    """Open a ``[1, TP_AXIS_SIZE]`` mesh with axes ``("dp", "tp")``; skip if unavailable."""
    previous_mgd = _ensure_mgd_path(MESH_SHAPE)
    _close_device_mesh_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(MESH_SHAPE, ("dp", "tp")))
    except Exception as e:  # noqa: BLE001
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"VocabParallelEmbedding tests need {TP_AXIS_SIZE} devices on the 'tp' axis: {e}")

    yield ttml.mesh()

    _close_device_mesh_quietly()
    _restore_mgd_path(previous_mgd)


# ---------------------------------------------------------------------------
# Host <-> distributed tensor helpers
#
# Reads impose the layout via an explicit composer rather than trusting the
# tensor's own topology (a grad produced by embedding_bw need not carry the
# weight's shard placement), which is what makes the gather robust.
# ---------------------------------------------------------------------------
def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _replicated_mapper():
    placements = [ttnn.PlacementReplicate() for _ in ttml.mesh().shape]
    return ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig(placements))


def _tp_shard_mapper(tdim: int):
    return ttml.mesh().axis_mapper("tp", tdim=tdim)


def _read_replicated(t) -> np.ndarray:
    """Read a tensor that is identical on every device into one host copy (fp32)."""
    mesh = ttml.mesh()
    n_axes = len(mesh.shape)
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig(list(range(n_axes))))
    full = t.to_numpy(ttnn.DataType.FLOAT32, composer=composer)
    slicer = [slice(None)] * full.ndim
    for axis in range(n_axes):
        slicer[axis] = slice(0, full.shape[axis] // mesh.shape[axis])
    return full[tuple(slicer)]


def _read_tp_sharded(t, shard_dim: int) -> np.ndarray:
    """Gather a tensor sharded along ``shard_dim`` on the 'tp' axis into the full table (fp32)."""
    mesh = ttml.mesh()
    tp_axis = mesh.axis_index("tp")
    scratch = iter(d for d in range(t.get_rank()) if d != shard_dim)
    composer_dims, scratch_axes = [], []
    for axis in range(len(mesh.shape)):
        if axis == tp_axis:
            composer_dims.append(shard_dim)
        else:
            d = next(scratch)
            composer_dims.append(d)
            scratch_axes.append((axis, d))
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig(composer_dims))
    full = t.to_numpy(ttnn.DataType.FLOAT32, composer=composer)
    slicer = [slice(None)] * full.ndim
    for axis, d in scratch_axes:
        slicer[d] = slice(0, full.shape[d] // mesh.shape[axis])
    return full[tuple(slicer)]


def _set_weight(module, weight_np: np.ndarray, mapper) -> None:
    module.weight.tensor.set_value(
        ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestVocabParallelEmbedding:
    # seq_len and embedding_dim are multiples of 32: the embedding backward
    # kernel requires it, and it keeps the reference and module on equal footing.
    DIM = 64
    BATCH = 2
    SEQ = 32

    @pytest.fixture(autouse=True)
    def _bind(self, tp_mesh):
        self.mesh = tp_mesh
        self.tp = tp_mesh.axis_size("tp")
        self.tp_axis = tp_mesh.axis_index("tp")

    def _weight(self, vocab: int, seed: int = 42) -> np.ndarray:
        return (np.random.default_rng(seed).standard_normal((1, 1, vocab, self.DIM)) * 0.02).astype(np.float32)

    def _ids(self, vocab: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).integers(0, vocab, size=(self.BATCH, 1, 1, self.SEQ)).astype(np.uint32)

    def _pair(self, vocab: int):
        """A replicated ``Embedding`` and a ``VocabParallelEmbedding`` sharing identical bf16 weights."""
        weight = self._weight(vocab)
        ref = Embedding(vocab, self.DIM)
        _set_weight(ref, weight, _replicated_mapper())
        vpe = VocabParallelEmbedding(vocab, self.DIM, axis_name="tp")
        _set_weight(vpe, weight, _tp_shard_mapper(tdim=2))
        return ref, vpe

    def _ids_tensor(self, ids_np: np.ndarray, mapper=None):
        return ttml.autograd.Tensor.from_numpy(
            ids_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper or _replicated_mapper()
        )

    def test_weight_is_vocab_sharded(self):
        vocab = 128
        vpe = VocabParallelEmbedding(vocab, self.DIM, axis_name="tp")

        assert vpe.num_embeddings_per_partition == vocab // self.tp

        placements = ttml.Sharding.from_tensor(vpe.weight.tensor).placements
        assert isinstance(placements[self.tp_axis], ttnn.PlacementShard)
        assert placements[self.tp_axis].dim == 2

    def test_num_embeddings_must_divide_tp(self, expect_error):
        with expect_error(ValueError, "must be divisible by the tensor-parallel"):
            VocabParallelEmbedding(self.tp * 32 + 1, self.DIM, axis_name="tp")

    def test_forward_rejects_tp_sharded_ids(self, expect_error):
        vpe = VocabParallelEmbedding(128, self.DIM, axis_name="tp")
        # ids split along seq_len across the tp axis => not replicated on tp.
        ids = self._ids_tensor(self._ids(128), mapper=_tp_shard_mapper(tdim=3))
        with expect_error(ValueError, "expects ids replicated across TP axis"):
            vpe(ids)

    @pytest.mark.parametrize("vocab", [64, 128, 256])
    def test_forward_matches_replicated_embedding(self, vocab):
        ctx = ttml.autograd.AutoContext.get_instance()
        ref, vpe = self._pair(vocab)
        ids = self._ids_tensor(self._ids(vocab))

        out_ref = _read_replicated(ref(ids))
        ctx.reset_graph()
        out_vpe = _read_replicated(vpe(ids))
        ctx.reset_graph()

        assert out_vpe.shape == out_ref.shape
        np.testing.assert_array_equal(out_vpe, out_ref)

    @pytest.mark.parametrize("vocab", [64, 128, 256])
    def test_backward_matches_replicated_embedding(self, vocab):
        ctx = ttml.autograd.AutoContext.get_instance()
        ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        ref, vpe = self._pair(vocab)
        ids = self._ids_tensor(self._ids(vocab, seed=1))

        ttml.ops.unary.mean(ref(ids)).backward(False)
        assert ref.weight.tensor.is_grad_initialized()
        ref_grad = _read_replicated(ref.weight.tensor.get_grad_tensor())
        ctx.reset_graph()

        ttml.ops.unary.mean(vpe(ids)).backward(False)
        assert vpe.weight.tensor.is_grad_initialized()
        vpe_grad = _read_tp_sharded(vpe.weight.tensor.get_grad_tensor(), shard_dim=2)
        ctx.reset_graph()

        assert vpe_grad.shape == ref_grad.shape
        np.testing.assert_array_equal(vpe_grad, ref_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
