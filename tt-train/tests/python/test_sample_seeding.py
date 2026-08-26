# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-device seeding behavior of ttml.ops.sample.sample_op on a multi-device mesh.

The sample op adds Gumbel noise to the logits before a local argmax. On a
multi-device mesh the noise must MIRROR the logits' data distribution:

  * devices that hold DISTINCT data (batch-sharded data-parallel axes: dp / fsdp)
    must draw INDEPENDENT noise -> they should sample DIFFERENT tokens, and
  * devices that hold REPLICATED data (tensor-parallel axis: tp) must draw
    IDENTICAL noise -> they must sample the SAME tokens (else a tp group would
    disagree on its single shared token).

`sample_op` takes an optional ``seed_axes`` list naming the mesh axes to seed
uniquely; the GRPO completers pass ONLY their dp/fsdp axes (never tp). These
tests exercise the op directly with that same policy across DDP / FSDP / TP /
DP+TP / DP+FSDP layouts and assert:

  * every pair of devices differing along a SEEDED (dp/fsdp) axis samples
    differently, and
  * every pair differing ONLY along an un-seeded (tp) axis samples identically.

To make the argmax a pure function of the noise, the logits are REPLICATED and
all-zero: every vocab class is tied, so the sampled index is decided entirely by
each device's Gumbel draw. With hundreds of independent draws per device, two
devices with distinct noise matching by chance is astronomically unlikely, while
two with identical noise match exactly.

Modeled on ``test_fsdp.py``: a single module-scoped fixture opens ONE mesh (so
the whole file shares one fabric bring-up), and the seeding scenarios are
parametrized against it. On CI (N150/N300/single BH) only the 2-device layouts
run; the 2D (DP+TP, DP+FSDP) layouts self-skip for want of a second non-trivial
axis. The opened shape is controlled by ``SAMPLE_SEEDING_MESH`` (e.g. "8,4"),
letting a galaxy run cover the 2D cases; set ``TT_MESH_GRAPH_DESC_PATH`` to a
descriptor matching that shape (the fixture only auto-fills a bundled MGD when
the env var is unset, so your value always wins).

The per-row ``positions`` tests live here too (rather than in the single-device
gtest suite) because what they check only EXISTS on a mesh: that a positions
tensor sharded with the batch mapper hands each device its own rows, and that
the op's topology equality check actually fires on a mismatch. On the default
single mesh both sides of that comparison are the trivial replicated topology,
so a single-device test of the check is vacuous.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

# Replicated logits shape [B, 1, T, V]. Kept tile-aligned; V>1 and rows all-tied
# so the argmax is driven purely by the per-device noise. B*T (~128) independent
# draws per device makes an accidental cross-device match effectively impossible.
B, T, V = 4, 32, 64
SEED = 1234
TEMPERATURE = 1.0

# Default opened mesh. [1, 2] is the smallest viable layout (N300 / one BH tray)
# and is what CI exercises. Override with SAMPLE_SEEDING_MESH="R,C" (e.g. "8,4"
# on a galaxy) to also cover the 2D DP+TP / DP+FSDP scenarios.
_DEFAULT_MESH = "1,2"


# --- MGD selection (mirrors test_fsdp.py): only fill in a bundled descriptor
#     when TT_MESH_GRAPH_DESC_PATH is UNSET, so a user-provided value always
#     wins (e.g. the galaxy descriptor set in the launch script). ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_DIR = os.path.join(_REPO_ROOT, "configs", "mgd")
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", (1, 2)): os.path.join(_MGD_DIR, "bh_galaxy_1_2_line_line.textproto"),
    ("blackhole", (2, 2)): os.path.join(_MGD_DIR, "bh_galaxy_2_2_line_line.textproto"),
    ("wormhole_b0", (1, 2)): os.path.join(_MGD_DIR, "n300_1_2_line_line.textproto"),
}


def _detect_arch() -> Optional[str]:
    """Return "blackhole"/"wormhole_b0" for the host (no device open needed), else None."""
    try:
        name = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    if "blackhole" in name:
        return "blackhole"
    if "wormhole_b0" in name:
        return "wormhole_b0"
    return None


def _ensure_mgd_path(shape: Tuple[int, ...]) -> Optional[str]:
    """Point TT_MESH_GRAPH_DESC_PATH at a bundled descriptor IFF it is unset.

    Returns the previous env value so the caller can restore it. Respects any
    user-provided value (never overrides it) and leaves the env alone when no
    bundled descriptor matches the host arch + shape.
    """
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if previous:
        return previous
    arch = _detect_arch()
    candidate = _MGD_FOR_ARCH_AND_SHAPE.get((arch, tuple(shape))) if arch else None
    if candidate and os.path.isfile(candidate):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = candidate
    return previous


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


def _close_device_quietly() -> None:
    try:
        ttml.autograd.AutoContext.get_instance().close_device()
    except Exception:  # noqa: BLE001
        pass


def _close_device_mesh_quietly() -> None:
    """Reverse ``open_device_mesh`` (close device, disable fabric, clear the global mesh),
    swallowing errors so teardown never masks a real failure."""
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


def _mesh_shape_from_env() -> Tuple[int, ...]:
    raw = os.environ.get("SAMPLE_SEEDING_MESH", _DEFAULT_MESH)
    return tuple(int(x) for x in raw.replace(" ", "").split(","))


@pytest.fixture(scope="module")
def seeding_mesh():
    """Open ONE mesh (shape from ``SAMPLE_SEEDING_MESH``) for all seeding scenarios.

    Skips the whole module if the host has too few devices or the mesh can't be
    opened. Closes the device and restores the MGD env var on teardown.
    """
    shape = _mesh_shape_from_env()
    required = math.prod(shape)
    num_devices = ttnn.get_num_devices()
    if num_devices < required:
        pytest.skip(f"mesh {shape} needs {required} devices, have {num_devices}")

    previous_mgd = _ensure_mgd_path(shape)
    _close_device_quietly()
    try:
        ttml.open_device_mesh(shape)
    except BaseException as e:  # noqa: BLE001 - mesh unopenable on this topology
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"could not open mesh {shape}: {e}")

    ttml.autograd.AutoContext.get_instance().set_seed(SEED)
    yield shape

    _close_device_mesh_quietly()
    _restore_mgd_path(previous_mgd)


# --- Seeding scenarios. `seed_axes` is evaluated against the opened shape; a
#     scenario self-skips when the opened mesh lacks the axes it needs.
#       need_seeded            : every axis in seed_axes must have size > 1
#       need_unseeded_nontrivial: some axis NOT in seed_axes must have size > 1
#                                 (so the "identical across the un-seeded/tp axis"
#                                  half of the invariant is actually exercised)
_SCENARIOS = [
    # name,        seed_axes, need_unseeded_nontrivial
    ("TP", [], True),  # seed nothing -> every device identical
    ("DDP", [0], False),  # unique along axis 0
    ("FSDP", [1], False),  # unique along axis 1
    ("DP_TP", [0], True),  # unique along axis 0 (dp), identical along axis 1 (tp)
    ("DP_FSDP", [0, 1], False),  # unique along both axes
]


def _row_major_coord(idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Mesh coordinate of linear device ``idx`` (row-major, matching get_device_tensors)."""
    coord = []
    for dim in reversed(shape):
        coord.append(idx % dim)
        idx //= dim
    return tuple(reversed(coord))


def _seed_key(idx: int, shape: Tuple[int, ...], seed_axes: List[int]) -> Tuple[int, ...]:
    """Devices sharing this key hold the same seeded-axis coords -> must get identical noise."""
    coord = _row_major_coord(idx, shape)
    return tuple(coord[a] for a in seed_axes)


def _run_sample(seed_axes: List[int]) -> List[np.ndarray]:
    """Replicated all-zero logits -> per-device sampled token indices (one array per device)."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)

    logits_np = np.zeros((B, 1, T, V), dtype=np.float32)
    logits = ttml.autograd.Tensor.from_numpy(logits_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, replicate)

    sampled = ttml.ops.sample.sample_op(logits, TEMPERATURE, SEED, None, seed_axes)

    shards = ttnn.get_device_tensors(sampled.get_value())
    per_device = [ttnn.to_torch(s).flatten().to(torch.int64).cpu().numpy() for s in shards]

    ttml.autograd.AutoContext.get_instance().reset_graph()
    return per_device


def _assert_seeding(per_device: List[np.ndarray], shape: Tuple[int, ...], seed_axes: List[int]) -> None:
    n = math.prod(shape)
    assert len(per_device) == n, f"expected {n} device shards, got {len(per_device)}"
    for arr in per_device:
        assert arr.size > 0
        assert np.all(arr < V), "sampled index outside [0, V)"

    # Group devices by seeded-axis coordinate. Within a group (differ only along
    # an un-seeded/tp axis) noise must be identical; across groups it must differ.
    groups: dict = {}
    for idx in range(n):
        groups.setdefault(_seed_key(idx, shape, seed_axes), []).append(idx)

    for key, members in groups.items():
        first = per_device[members[0]]
        for other in members[1:]:
            np.testing.assert_array_equal(
                per_device[other],
                first,
                err_msg=(
                    f"devices {members[0]} and {other} share seeded coords {key} "
                    f"(differ only on an un-seeded/tp axis) but sampled differently -- "
                    f"replicated data must get identical noise"
                ),
            )

    reps = {key: per_device[members[0]] for key, members in groups.items()}
    keys = list(reps)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.array_equal(reps[keys[i]], reps[keys[j]]), (
                f"device groups {keys[i]} and {keys[j]} differ on a seeded (dp/fsdp) axis "
                f"but sampled identically -- distinct data must get unique noise"
            )


@pytest.mark.parametrize("name,seed_axes,need_unseeded_nontrivial", _SCENARIOS, ids=[c[0] for c in _SCENARIOS])
def test_sample_seeding(seeding_mesh, name, seed_axes, need_unseeded_nontrivial):
    shape = seeding_mesh

    if any(a >= len(shape) or shape[a] <= 1 for a in seed_axes):
        pytest.skip(f"{name}: mesh {shape} has no non-trivial axis for seed_axes={seed_axes}")
    if need_unseeded_nontrivial and not any(i not in seed_axes and shape[i] > 1 for i in range(len(shape))):
        pytest.skip(f"{name}: mesh {shape} has no un-seeded non-trivial axis to exercise")
    if not seed_axes and math.prod(shape) <= 1:
        pytest.skip(f"{name}: single-device mesh, nothing to compare")

    per_device = _run_sample(seed_axes)
    _assert_seeding(per_device, shape, seed_axes)


# --- Per-row positions on a mesh. Greedy (temperature 0) so the result is a pure
#     function of the logits and positions: no noise, no seed sensitivity.


def _positions_winners(B_total: int, T_pos: int, V_pos: int):
    """Distinct winner per (row, token position) so a wrong row or wrong device shard
    lands on a DIFFERENT id rather than coincidentally matching."""
    logits_np = np.full((B_total, 1, T_pos, V_pos), -1.0, dtype=np.float32)
    winner = np.zeros((B_total, T_pos), dtype=np.int64)
    for b in range(B_total):
        for t in range(T_pos):
            w = ((b * T_pos + t) * 7) % V_pos
            logits_np[b, 0, t, w] = -0.5
            winner[b, t] = w
    return logits_np, winner


def test_sample_positions_sharded_selects_each_devices_rows(seeding_mesh):
    """Sharded logits + positions sharded with the SAME mapper: every global row must be
    sampled at ITS OWN position. This is the one configuration production uses and the one
    no single-device test can exercise -- it covers the per-device row slice, the op's
    positions/logits topology equality on real Shard{0} topologies (non-vacuous only on a
    mesh), and the [B, 1, 1, 1] output composing back in batch order."""
    from ttml.common.sampling import positions_to_tensor

    shape = seeding_mesh
    n = math.prod(shape)
    if n <= 1:
        pytest.skip("positions sharding needs a multi-device mesh")

    device = ttml.autograd.AutoContext.get_instance().get_device()
    dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    B_total, T_pos, V_pos = 4 * n, 64, 64  # T spans two tile rows so positions cross a tile boundary
    logits_np, winner = _positions_winners(B_total, T_pos, V_pos)
    # Positions differ per row and cover first/last tile rows, so a stale or mis-sliced
    # device shard cannot pass by luck.
    positions = [(b * 13) % T_pos for b in range(B_total)]

    logits = ttml.autograd.Tensor.from_numpy(logits_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)
    positions_tt = positions_to_tensor(positions, B_total, T_pos, dp_mapper)

    sampled = ttml.ops.sample.sample_op(logits, 0.0, SEED, None, [0], positions_tt)
    got = ttnn.to_torch(sampled.get_value(), mesh_composer=composer).flatten().to(torch.int64).cpu().numpy()
    ttml.autograd.AutoContext.get_instance().reset_graph()

    assert got.size == B_total, f"expected one token per global row, got {got.size}"
    expected = np.array([winner[b, positions[b]] for b in range(B_total)], dtype=np.int64)
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg="a row sampled at another row's position -- the positions shard does not match the batch shard",
    )


def test_sample_positions_topology_mismatch_rejected(seeding_mesh, expect_error):
    """REPLICATED positions against SHARDED logits must be rejected loudly. On a mesh the
    two topologies genuinely differ, so this is the first configuration in which the op's
    topology equality check can fire at all; if it silently passed, page e of the replicated
    tensor would be read as device-local row e -- valid-looking tokens for the wrong rows."""
    shape = seeding_mesh
    n = math.prod(shape)
    if n <= 1:
        pytest.skip("topologies coincide on a single-device mesh; the check is vacuous")

    device = ttml.autograd.AutoContext.get_instance().get_device()
    dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)

    B_total, T_pos, V_pos = 4 * n, 64, 64
    logits_np, _ = _positions_winners(B_total, T_pos, V_pos)
    logits = ttml.autograd.Tensor.from_numpy(logits_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)

    # Deliberately the WRONG mapper: replicated [B_local, 1, 1, 1] per device (B_local rows so
    # the shape check passes and the failure is attributable to the topology check alone).
    B_local = B_total // n
    bad_np = np.zeros((B_local, 1, 1, 1), dtype=np.uint32)
    bad_positions = ttml.autograd.Tensor.from_numpy(bad_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, replicate)

    with expect_error(RuntimeError, "distributed across the mesh exactly as the logits"):
        ttml.ops.sample.sample_op(logits, 0.0, SEED, None, [0], bad_positions)
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_sample_per_row_mask_topology_mismatch_rejected(seeding_mesh, expect_error):
    """A per-row [B, 1, 1, V] logits_mask is per-device-row data exactly like positions: page e must
    be the bias for the logits' local entry e. A REPLICATED per-row mask against SHARDED logits must
    be rejected loudly -- on a mesh the topologies genuinely differ, which no single-device test can
    exercise. (A shared [1, 1, 1, V] mask stays exempt: one row for everyone is distribution-free.)"""
    shape = seeding_mesh
    n = math.prod(shape)
    if n <= 1:
        pytest.skip("topologies coincide on a single-device mesh; the check is vacuous")

    device = ttml.autograd.AutoContext.get_instance().get_device()
    dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)

    B_total, T_pos, V_pos = 4 * n, 64, 64
    logits_np, _ = _positions_winners(B_total, T_pos, V_pos)
    logits = ttml.autograd.Tensor.from_numpy(logits_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)

    # Wrong mapper on purpose: replicated, sized to the LOCAL batch so the shape check passes and
    # the failure is attributable to the topology check alone.
    B_local = B_total // n
    bad_np = np.zeros((B_local, 1, 1, V_pos), dtype=np.float32)
    bad_mask = ttml.autograd.Tensor.from_numpy(bad_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, replicate)

    with expect_error(RuntimeError, "mask must be distributed across the mesh exactly as the logits"):
        ttml.ops.sample.sample_op(logits, 0.0, SEED, bad_mask, [0])
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_sample_per_row_mask_sharded_applies_each_rows_bias(seeding_mesh):
    """The ACCEPTANCE path for a per-row mask on a mesh: a [B, 1, 1, V] mask sharded with the SAME
    mapper as the logits must hand every device its own rows' bias. Greedy (temperature 0) so the
    result is a pure function of logits and mask. Every global row b has a UNIQUE winner w(b) and
    its mask row bans exactly w(b): the correct outcome is the shared runner-up (column 0) for every
    row, while a device that received another device's mask shard leaves its own rows' winners
    un-banned -- w(b) is injective across the GLOBAL batch, so any shard swap or offset surfaces as
    a non-zero token. This is the configuration the single-device gtests structurally cannot reach."""
    shape = seeding_mesh
    n = math.prod(shape)
    if n <= 1:
        pytest.skip("mask sharding needs a multi-device mesh")

    device = ttml.autograd.AutoContext.get_instance().get_device()
    dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    B_total, T_tok, V_tok = 2 * n, 32, 96  # w(b) = 1 + b must stay injective within the vocab
    assert B_total + 1 < V_tok
    logits_np = np.full((B_total, 1, T_tok, V_tok), -1.0, dtype=np.float32)
    mask_np = np.zeros((B_total, 1, 1, V_tok), dtype=np.float32)
    for b in range(B_total):
        logits_np[b, 0, :, 1 + b] = -0.25  # unique winner per global row
        logits_np[b, 0, :, 0] = -0.5  # shared runner-up
        mask_np[b, 0, 0, 1 + b] = 1e4  # ban exactly this row's winner

    logits = ttml.autograd.Tensor.from_numpy(logits_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)
    mask = ttml.autograd.Tensor.from_numpy(mask_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)

    # Sanity without the mask: each row's own winner (also proves the winners are distinguishable).
    unmasked = ttml.ops.sample.sample_op(logits, 0.0, SEED, None, [0])
    got = ttnn.to_torch(unmasked.get_value(), mesh_composer=composer).flatten().to(torch.int64).cpu().numpy()
    got = got.reshape(B_total, T_tok)
    for b in range(B_total):
        assert np.all(got[b] == 1 + b), f"unmasked winner wrong for row {b}: {got[b][:4]}"

    masked = ttml.ops.sample.sample_op(logits, 0.0, SEED, mask, [0])
    got = ttnn.to_torch(masked.get_value(), mesh_composer=composer).flatten().to(torch.int64).cpu().numpy()
    got = got.reshape(B_total, T_tok)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    for b in range(B_total):
        assert np.all(got[b] == 0), (
            f"row {b} sampled {got[b][:4]} -- its own winner {1 + b} was not banned, so this device "
            f"applied another row's mask shard"
        )
