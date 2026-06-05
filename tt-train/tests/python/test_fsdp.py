# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ttml.fsdp.fully_shard`` on a multi-device mesh.

Covered:
  * Shard mechanics on a single ``LinearLayer`` — local-shape change,
    dtype preservation, ``_fsdp_managed`` marker.
  * ``shard -> gather`` matches the source tensor (proves the shards are
    the correct slices, not arbitrary garbage).
  * ``shard -> unshard`` round-trip preserves the full tensor's values.
  * ``fully_shard`` guard rails: double-wrap raises, unknown axis raises.
  * Root-wrapping with nested blocks (FSDP2-root semantics: the root
    wrapper claims only parameters NOT already owned by a nested
    wrapper).
  * Forward and backward equivalence against a replicated reference
    (every device runs the same computation, every device sees the same
    weights). With identical inputs on every device the reduce-scatter
    mean inside FSDP's backward_post collapses to the per-device full
    grad sliced to the local shard, so the gathered FSDP grad must
    match the full reference grad (modulo bf16 rounding).

These tests need at least ``FSDP_AXIS_SIZE`` devices on the host (default
2 — the smallest viable layout, e.g. an N300 board). The module-scoped
fixture skips the whole file when fewer devices are available.

NOTE: placements on the underlying ttnn tensors are NOT a reliable
indicator of shard state in TTML right now — several CCL ops drop or
fail to update them. The tests below intentionally do not assert on
``tensor_topology().placements()``; they rely on the ``_fsdp_managed``
marker and on actual data-level checks via host-side numpy gathers.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pytest

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList


pytestmark = pytest.mark.requires_device


# How many devices we need on the FSDP axis. 2 is the smallest viable
# layout (N300 / a single Blackhole tray). Bump this and re-export
# ``TT_MESH_GRAPH_DESC_PATH`` (or extend ``_MGD_FOR_SHAPE`` below) to
# exercise larger mesh sizes.
FSDP_AXIS_SIZE = 2

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", (1, 2)): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_1_2_line_line.textproto"),
    ("wormhole_b0", (1, 2)): os.path.join(_REPO_ROOT, "configs", "mgd", "n300_1_2_line_line.textproto"),
}


def _detect_arch() -> Optional[str]:
    """Return ``"blackhole"`` or ``"wormhole_b0"`` for the host, or ``None``.

    Uses ``ttnn.get_arch_name()`` which reads the cluster yaml at
    process start and does not require any device to be open. Returns
    ``None`` on any failure so the caller can fall back to whatever the
    user supplied via ``TT_MESH_GRAPH_DESC_PATH``.
    """
    try:
        name = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    if "blackhole" in name:
        return "blackhole"
    if "wormhole_b0" in name:
        return "wormhole_b0"
    return None


# ---------------------------------------------------------------------------
# Multi-device mesh fixture
# ---------------------------------------------------------------------------


def _close_device_quietly() -> None:
    try:
        ttml.autograd.AutoContext.get_instance().close_device()
    except Exception:  # noqa: BLE001
        pass


def _ensure_mgd_path(shape: tuple[int, ...]) -> Optional[str]:
    """If ``TT_MESH_GRAPH_DESC_PATH`` isn't set, point it at a bundled MGD.

    ``ttml.open_device_mesh`` calls ``_validate_mgd``, which only does a
    soft warning when the env var is unset — but the underlying fabric
    layer relies on the MGD too, so on a Blackhole galaxy host the open
    can hang or fail without one. We pick a bundled MGD that matches the
    host arch + requested mesh shape; if no match exists we leave the
    env alone so the open path can still succeed on hosts that don't
    need an MGD.

    Returns the previous value of the env var (``None`` if unset), so the
    caller can restore it on teardown.
    """
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
def fsdp_mesh():
    """Open a 2D mesh ``[1, FSDP_AXIS_SIZE]`` with axes ``("dp", "fsdp")``.

    A 2D layout with ``dp=1`` keeps the same fixture re-usable for HSDP
    tests later (those need a real ``dp`` axis without re-shaping the
    mesh). Skips the module if the system can't host the requested mesh.

    If ``TT_MESH_GRAPH_DESC_PATH`` isn't set in the environment, we point
    it at a bundled MGD that matches the host arch + requested shape (see
    ``_MGD_FOR_ARCH_AND_SHAPE``) so the fabric layer can come up cleanly.
    The original value is restored at teardown.
    """
    shape = (1, FSDP_AXIS_SIZE)
    previous_mgd = _ensure_mgd_path(shape)

    _close_device_quietly()
    try:
        m = ttml.Mesh(shape, ("dp", "fsdp"))
        ttml.open_device_mesh(m)
    except Exception as e:  # noqa: BLE001
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"FSDP tests need {FSDP_AXIS_SIZE} devices on the 'fsdp' axis: {e}")

    yield ttml.mesh()

    # Close at teardown so later test modules can lazily reopen a fresh
    # single-device handle if they need to. The global ``ttml._mesh._mesh``
    # is reset so ``ttml.mesh()`` doesn't return a handle to a closed
    # device for any test that runs after this module.
    _close_device_quietly()
    try:
        import ttml._mesh as _mesh_mod  # type: ignore[import-not-found]

        _mesh_mod._mesh = None
    except Exception:  # noqa: BLE001
        pass
    _restore_mgd_path(previous_mgd)


# ---------------------------------------------------------------------------
# Numpy <-> distributed tensor helpers
# ---------------------------------------------------------------------------


def _replicated_mapper():
    """A mesh mapper that replicates a tensor on every mesh axis."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mesh = ttml.mesh()
    placements = [ttnn.PlacementReplicate() for _ in mesh.shape]
    return ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))


def _read_replicated_to_numpy(autograd_tensor, mesh, *, dtype=ttnn.DataType.FLOAT32):
    """Read a tensor identical on every mesh device into canonical numpy.

    Compose along an unused tensor dim per mesh axis (the first ``n_axes``
    dims), then slice each stacking dim back to the per-device size.
    Works for any tensor of rank >= ``len(mesh.shape)``.
    """
    device = ttml.autograd.AutoContext.get_instance().get_device()
    n_axes = len(mesh.shape)
    composer_dims = list(range(n_axes))
    composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(composer_dims))
    full_np = autograd_tensor.to_numpy(dtype, composer=composer)
    slicer = [slice(None)] * full_np.ndim
    for d in range(n_axes):
        per_dev_size = full_np.shape[d] // mesh.shape[d]
        slicer[d] = slice(0, per_dev_size)
    return full_np[tuple(slicer)]


def _read_fsdp_sharded_to_numpy(autograd_tensor, mesh, shard_dim, *, dtype=ttnn.DataType.FLOAT32):
    """Gather an FSDP-sharded tensor across the ``"fsdp"`` axis into numpy.

    The "fsdp" axis composes along ``shard_dim`` (concat the shards);
    every other axis (replicated) gets a scratch tensor dim that we slice
    back to its per-device size afterwards.
    """
    device = ttml.autograd.AutoContext.get_instance().get_device()
    fsdp_axis = mesh.axis_index("fsdp")
    rank = autograd_tensor.get_rank()

    composer_dims = []
    scratch_iter = iter(d for d in range(rank) if d != shard_dim)
    scratch_axes = []
    for ax_idx in range(len(mesh.shape)):
        if ax_idx == fsdp_axis:
            composer_dims.append(shard_dim)
        else:
            d = next(scratch_iter)
            composer_dims.append(d)
            scratch_axes.append((ax_idx, d))

    composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(composer_dims))
    full_np = autograd_tensor.to_numpy(dtype, composer=composer)

    if scratch_axes:
        slicer = [slice(None)] * full_np.ndim
        for ax_idx, d in scratch_axes:
            per_dev_size = full_np.shape[d] // mesh.shape[ax_idx]
            slicer[d] = slice(0, per_dev_size)
        full_np = full_np[tuple(slicer)]
    return full_np


def _set_param_replicated(autograd_tensor, full_np, *, dtype=ttnn.DataType.BFLOAT16) -> None:
    """Overwrite a parameter's value with a freshly built replicated tensor."""
    mapper = _replicated_mapper()
    new_t = ttml.autograd.Tensor.from_numpy(full_np.astype(np.float32), ttnn.Layout.TILE, dtype, mapper)
    autograd_tensor.set_value(new_t.get_value())


# ---------------------------------------------------------------------------
# Tiny test models
# ---------------------------------------------------------------------------


class _TinyBlock(AbstractModuleBase):
    """Two-linear-layer block with biases. All shapes are tile-aligned (32-mult)."""

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = LinearLayer(in_features, hidden, has_bias=True)
        self.fc2 = LinearLayer(hidden, out_features, has_bias=True)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _TinyModel(AbstractModuleBase):
    """A few blocks + a head, used to exercise FSDP2-root wrapping semantics."""

    def __init__(self, in_features: int, hidden: int, out_features: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = ModuleList([_TinyBlock(in_features, hidden, out_features) for _ in range(num_blocks)])
        self.head = LinearLayer(out_features, out_features, has_bias=False)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def _build_block_with_known_weights(in_features: int, hidden: int, out_features: int, *, seed: int = 42) -> _TinyBlock:
    """Build a ``_TinyBlock`` whose parameters are deterministic and replicated.

    LinearLayer's default uniform init is non-deterministic across processes;
    we overwrite every param with seeded random data so the reference run
    and the FSDP run start from the exact same numbers on every device.
    """
    rng = np.random.default_rng(seed)
    block = _TinyBlock(in_features, hidden, out_features)
    for _, t in block.named_parameters():
        shape = tuple(t.shape())
        data = rng.standard_normal(shape).astype(np.float32) * 0.1
        _set_param_replicated(t, data)
    return block


# ---------------------------------------------------------------------------
# fully_shard mechanics on a single LinearLayer
# ---------------------------------------------------------------------------


class TestFullyShardLinear:
    """``fully_shard`` mechanics on a single ``LinearLayer``."""

    @pytest.fixture(autouse=True)
    def _setup(self, fsdp_mesh):
        self.mesh = fsdp_mesh
        self.axis_size = FSDP_AXIS_SIZE
        ttml.autograd.AutoContext.get_instance().set_seed(42)
        np.random.seed(42)

    def test_shard_changes_local_shape(self):
        """After ``fully_shard``, the per-device shape collapses along the chosen dim.

        Auto picks ``rank-2`` for weight (the typical "O" dim in
        ``[1,1,O,I]``). Bias has ``shape[rank-2] == 1`` so auto falls
        through to ``rank-1``.
        """
        in_features, out_features = 64, 128
        linear = LinearLayer(in_features, out_features, has_bias=True)

        assert linear.weight.tensor.shape() == [1, 1, out_features, in_features]
        assert linear.bias.tensor.shape() == [1, 1, 1, out_features]

        ttml.fsdp.fully_shard(linear)

        assert linear.weight.tensor.shape() == [1, 1, out_features // self.axis_size, in_features]
        assert linear.bias.tensor.shape() == [1, 1, 1, out_features // self.axis_size]

        assert ttml.fsdp.is_fsdp_managed(linear.weight.tensor)
        assert ttml.fsdp.is_fsdp_managed(linear.bias.tensor)
        assert ttml.fsdp.fsdp_axis_of(linear.weight.tensor) == self.mesh.axis_index("fsdp")

    def test_shard_preserves_dtype(self):
        """``fully_shard`` does not change the parameter dtype (bf16 stays bf16)."""
        linear = LinearLayer(64, 128, has_bias=True)
        weight_dtype_before = linear.weight.tensor.get_value().dtype
        bias_dtype_before = linear.bias.tensor.get_value().dtype

        ttml.fsdp.fully_shard(linear)

        assert linear.weight.tensor.get_value().dtype == weight_dtype_before
        assert linear.bias.tensor.get_value().dtype == bias_dtype_before

    def test_shard_gather_matches_original(self):
        """Set known random weights, shard, gather across mesh, compare to source.

        The gather is bit-exact against the bf16-quantised pre-shard
        snapshot: FSDP's shard step is a redistribute through host, so it
        must not perturb values, only their layout.
        """
        in_features, out_features = 64, 128
        linear = LinearLayer(in_features, out_features, has_bias=True)

        rng = np.random.default_rng(0)
        weight_np = rng.standard_normal((1, 1, out_features, in_features)).astype(np.float32) * 0.1
        bias_np = rng.standard_normal((1, 1, 1, out_features)).astype(np.float32) * 0.1
        _set_param_replicated(linear.weight.tensor, weight_np)
        _set_param_replicated(linear.bias.tensor, bias_np)

        # Snapshot the bf16-quantised baseline AFTER set_value, so the
        # comparison below isn't fighting bf16 rounding from the
        # fp32 -> bf16 host -> device hop.
        weight_bf16 = _read_replicated_to_numpy(linear.weight.tensor, self.mesh)
        bias_bf16 = _read_replicated_to_numpy(linear.bias.tensor, self.mesh)

        ttml.fsdp.fully_shard(linear)

        weight_shard_dim = int(linear.weight.tensor._fsdp_shard_dim)
        bias_shard_dim = int(linear.bias.tensor._fsdp_shard_dim)
        gathered_w = _read_fsdp_sharded_to_numpy(linear.weight.tensor, self.mesh, weight_shard_dim)
        gathered_b = _read_fsdp_sharded_to_numpy(linear.bias.tensor, self.mesh, bias_shard_dim)

        assert gathered_w.shape == weight_bf16.shape
        assert gathered_b.shape == bias_bf16.shape
        np.testing.assert_array_equal(gathered_w, weight_bf16)
        np.testing.assert_array_equal(gathered_b, bias_bf16)

    def test_shard_unshard_round_trip_preserves_values(self):
        """``shard -> unshard`` returns the parameter to its full shape and values."""
        in_features, out_features = 64, 128
        linear = LinearLayer(in_features, out_features, has_bias=True)

        rng = np.random.default_rng(123)
        weight_np = rng.standard_normal((1, 1, out_features, in_features)).astype(np.float32) * 0.1
        _set_param_replicated(linear.weight.tensor, weight_np)
        weight_before = _read_replicated_to_numpy(linear.weight.tensor, self.mesh)
        dtype_before = linear.weight.tensor.get_value().dtype

        ttml.fsdp.fully_shard(linear)
        assert linear.weight.tensor.shape() == [1, 1, out_features // self.axis_size, in_features]

        linear.unshard()
        assert linear.weight.tensor.shape() == [1, 1, out_features, in_features]
        assert linear.weight.tensor.get_value().dtype == dtype_before
        weight_after = _read_replicated_to_numpy(linear.weight.tensor, self.mesh)
        np.testing.assert_array_equal(weight_before, weight_after)

    def test_double_wrap_raises(self):
        """Calling ``fully_shard`` twice on the same module raises."""
        linear = LinearLayer(64, 128, has_bias=False)
        ttml.fsdp.fully_shard(linear)
        with pytest.raises(RuntimeError):
            ttml.fsdp.fully_shard(linear)

    def test_invalid_axis_raises(self):
        """Sharding on an axis the mesh doesn't have raises before any state change."""
        linear = LinearLayer(64, 128, has_bias=False)
        with pytest.raises(RuntimeError):
            ttml.fsdp.fully_shard(linear, mesh_axis="this_axis_does_not_exist")


# ---------------------------------------------------------------------------
# Root-wrapping semantics
# ---------------------------------------------------------------------------


class TestFullyShardRoot:
    """``fully_shard`` on a multi-block model in FSDP2-root style."""

    @pytest.fixture(autouse=True)
    def _setup(self, fsdp_mesh):
        self.mesh = fsdp_mesh
        ttml.autograd.AutoContext.get_instance().set_seed(7)
        np.random.seed(7)

    def test_root_wrapper_excludes_block_params(self):
        """Per-block wrappers manage their own params; the root manages only ``head``."""
        model = _TinyModel(64, 128, 64, num_blocks=2)

        for blk in model.blocks:
            ttml.fsdp.fully_shard(blk)
        ttml.fsdp.fully_shard(model)

        # Every block's parameters are FSDP-managed (by the block wrapper).
        for blk in model.blocks:
            assert ttml.fsdp.is_fsdp_managed(blk.fc1.weight.tensor)
            assert ttml.fsdp.is_fsdp_managed(blk.fc1.bias.tensor)
            assert ttml.fsdp.is_fsdp_managed(blk.fc2.weight.tensor)
            assert ttml.fsdp.is_fsdp_managed(blk.fc2.bias.tensor)

        # The head, owned only by the root, is FSDP-managed too.
        assert ttml.fsdp.is_fsdp_managed(model.head.weight.tensor)


# ---------------------------------------------------------------------------
# Forward / backward equivalence vs replicated reference
# ---------------------------------------------------------------------------


class TestFSDPEquivalence:
    """Equivalence vs a replicated (non-FSDP) reference run.

    Both runs use the *same* multi-device mesh and the *same* replicated
    input on every device, so for the reference the forward output and
    every parameter's grad are identical on every rank — which means we
    can compare a single device's view to the full FSDP gather.

    With identical replicated inputs the reduce-scatter mean inside
    ``backward_post`` collapses to the per-device full grad sliced to
    the local shard, so the gathered FSDP grad equals the full reference
    grad.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, fsdp_mesh):
        self.mesh = fsdp_mesh
        ttml.autograd.AutoContext.get_instance().set_seed(0)
        np.random.seed(0)

    @staticmethod
    def _make_input(batch_size: int, seq_len: int, features: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((batch_size, 1, seq_len, features)).astype(np.float32) * 0.1

    def test_forward_matches_reference(self):
        """FSDP forward output ≈ replicated-reference forward output."""
        in_features, hidden, out_features = 64, 128, 64
        batch_size, seq_len = 2, 32
        input_np = self._make_input(batch_size, seq_len, in_features, seed=0)
        mapper = _replicated_mapper()

        # ---- Reference: replicated model, no fully_shard.
        ref_model = _build_block_with_known_weights(in_features, hidden, out_features, seed=42)
        ref_model.eval()
        x_ref = ttml.autograd.Tensor.from_numpy(input_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
        out_ref = ref_model(x_ref)
        out_ref_np = _read_replicated_to_numpy(out_ref, self.mesh)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        # ---- FSDP: same initial weights, fully_shard, same input.
        fsdp_model = _build_block_with_known_weights(in_features, hidden, out_features, seed=42)
        fsdp_model.eval()
        ttml.fsdp.fully_shard(fsdp_model)
        x_fsdp = ttml.autograd.Tensor.from_numpy(input_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
        out_fsdp = fsdp_model(x_fsdp)
        out_fsdp_np = _read_replicated_to_numpy(out_fsdp, self.mesh)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        assert out_fsdp_np.shape == out_ref_np.shape
        np.testing.assert_array_equal(out_fsdp_np, out_ref_np)

    def test_backward_matches_reference(self):
        """FSDP gathered gradients ≈ replicated-reference gradients."""
        in_features, hidden, out_features = 64, 128, 64
        batch_size, seq_len = 2, 32
        input_np = self._make_input(batch_size, seq_len, in_features, seed=1)
        mapper = _replicated_mapper()

        # ---- Reference backward
        ref_model = _build_block_with_known_weights(in_features, hidden, out_features, seed=42)
        ref_model.train()
        x_ref = ttml.autograd.Tensor.from_numpy(input_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
        out_ref = ref_model(x_ref)
        loss_ref = ttml.ops.unary.mean(out_ref)
        loss_ref.backward(False)

        ref_grads: Dict[str, np.ndarray] = {}
        for name, t in ref_model.named_parameters():
            if not t.is_grad_initialized():
                continue
            grad_t = t.get_grad_tensor()
            ref_grads[name] = _read_replicated_to_numpy(grad_t, self.mesh)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        assert ref_grads, "Reference run should have produced gradients for at least one parameter"

        # ---- FSDP backward
        fsdp_model = _build_block_with_known_weights(in_features, hidden, out_features, seed=42)
        fsdp_model.train()
        ttml.fsdp.fully_shard(fsdp_model)
        x_fsdp = ttml.autograd.Tensor.from_numpy(input_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
        out_fsdp = fsdp_model(x_fsdp)
        loss_fsdp = ttml.ops.unary.mean(out_fsdp)
        loss_fsdp.backward(False)

        fsdp_grads: Dict[str, np.ndarray] = {}
        for name, t in fsdp_model.named_parameters():
            if not t.is_grad_initialized():
                continue
            grad_t = t.get_grad_tensor()
            shard_dim = int(t._fsdp_shard_dim)
            fsdp_grads[name] = _read_fsdp_sharded_to_numpy(grad_t, self.mesh, shard_dim)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        assert set(fsdp_grads.keys()) == set(
            ref_grads.keys()
        ), f"FSDP grad keys {sorted(fsdp_grads)} != reference keys {sorted(ref_grads)}"
        for name in ref_grads:
            assert fsdp_grads[name].shape == ref_grads[name].shape, (
                f"Grad shape mismatch for {name}: "
                f"FSDP gathered {fsdp_grads[name].shape} vs reference {ref_grads[name].shape}"
            )
            np.testing.assert_array_equal(fsdp_grads[name], ref_grads[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
