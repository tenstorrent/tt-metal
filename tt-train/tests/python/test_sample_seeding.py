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

    _close_device_quietly()
    try:
        import ttml._mesh as _mesh_mod  # type: ignore[import-not-found]

        _mesh_mod._mesh = None
    except Exception:  # noqa: BLE001
        pass
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
