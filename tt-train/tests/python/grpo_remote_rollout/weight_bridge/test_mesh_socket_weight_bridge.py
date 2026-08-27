# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Transport-level unit test for :class:`MeshSocketWeightBridge` (1->1).

Rank 0 sends synthetic bf16/TILE/DRAM tensors on a ``[1, 1]`` mesh; rank 1
receives them onto a matching ``[1, 1]`` mesh (single-submesh) and asserts the
device tensor holds the exact sent values. Runs under tt-run with world_size ==
2; self-skips otherwise (see ``runner_mesh_socket_weight_bridge.sh``).

Mirrors :mod:`test_weight_bridge` (which exercises the MPI-only
:class:`HostWeightBridge` at 4->4). This test focuses on the fabric transport
alone: no model, no completer, no TTT worker.
"""

from __future__ import annotations

import gc
import math
import os

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_mesh_socket_weight_bridge must run under tt-run with world_size == 2 "
        "(use tests/weight_bridge/runner_mesh_socket_weight_bridge.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

# Fabric pinned FABRIC_2D by conftest's autouse fixture (both ranks must match).
import torch  # noqa: E402
import ttnn  # noqa: E402

from utils.mesh_socket_bridge import MeshSocketWeightBridge  # noqa: E402
from utils.weight_bridge import (  # noqa: E402
    RECEIVER_RANK,
    SENDER_RANK,
)

# 1->1 socket on top of [1, 4] parent meshes.
# Both ranks open a [1, 4] mesh so the fabric topology mapper can resolve the
# inter-mesh routes over the BH loudbox's physical 2x4 wiring (see
# configurations/1_1/mgd.textproto). Each side then carves a [1, 1] submesh at
# offset (0, 0) and hands that to the bridge -- the MeshSocket itself is
# strictly single-device on both ends (chip 0 <-> chip 4 over the vertical
# inter-mesh cable).
PARENT_SHAPE = (1, 4)
SUBMESH_SHAPE = (1, 1)
SUBMESH_OFFSET = (0, 0)

# A few tile-aligned tensor specs (rows/cols multiples of 32 -> no tile padding).
_SPECS = {
    "weight.alpha": (1, 1, 32, 64),
    "weight.beta": (1, 1, 64, 96),
    "weight.gamma": (1, 1, 128, 32),
}


def _synthetic_torch() -> dict:
    """Deterministic per-key torch bf16 tensors (both ranks compute identically)."""
    out = {}
    for i, key in enumerate(sorted(_SPECS)):
        shape = _SPECS[key]
        n = math.prod(shape)
        # Both ranks round to bf16 the same way, so the compare is exact.
        out[key] = (torch.arange(n, dtype=torch.float32).reshape(shape) + i * 10_000).to(torch.bfloat16)
    return out


def _upload_replicated(host_tensor, mesh) -> "ttnn.Tensor":
    return ttnn.from_torch(
        host_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )


def _ensure_distributed_context() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()


def _sender_side(mesh) -> None:
    weights = {k: _upload_replicated(v, mesh) for k, v in _synthetic_torch().items()}
    bridge = MeshSocketWeightBridge.init_sender(mesh=mesh, peer_rank=RECEIVER_RANK)
    bridge.connect()
    print(f"[sender] MeshSocketWeightBridge: sending {len(weights)} tensors", flush=True)
    bridge.send_weights(weights)
    bridge.barrier()
    print("[sender] MeshSocketWeightBridge: done", flush=True)


def _receiver_side(mesh) -> None:
    expected = _synthetic_torch()
    # Single-device receiver: submeshes is just [mesh] (validated in the bridge).
    bridge = MeshSocketWeightBridge.init_receiver(mesh=mesh, peer_rank=SENDER_RANK, submeshes=[mesh])
    bridge.connect()
    per_submesh = bridge.receive_weights()
    bridge.barrier()

    assert isinstance(per_submesh, list), f"receive_weights must return a list, got {type(per_submesh)}"
    assert len(per_submesh) == 1, f"expected 1 dict (single-device receiver), got {len(per_submesh)}"
    got_dict = per_submesh[0]

    # Compare per-key. Collect all mismatches instead of aborting on the first.
    failures = []  # (key, reason)
    if sorted(got_dict.keys()) != sorted(expected.keys()):
        print(
            f"[receiver] KEY MISMATCH {sorted(got_dict.keys())} != {sorted(expected.keys())}",
            flush=True,
        )
        failures.append(("<keys>", "key set mismatch"))
    else:
        for key in sorted(expected.keys()):
            tensor = got_dict[key]
            meta_bad = []
            if tensor.dtype != ttnn.bfloat16:
                meta_bad.append(f"dtype={tensor.dtype}")
            if tensor.layout != ttnn.TILE_LAYOUT:
                meta_bad.append(f"layout={tensor.layout}")
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                meta_bad.append(f"memcfg={tensor.memory_config()}")

            got = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
            exp = expected[key]
            if got.shape != exp.shape:
                print(
                    f"[receiver] key {key!r}: SHAPE {tuple(got.shape)} != {tuple(exp.shape)}",
                    flush=True,
                )
                failures.append((key, "shape mismatch"))
                continue

            match = torch.equal(got, exp)
            n_bad = int((got != exp).sum().item())
            status = "OK" if (match and not meta_bad) else "MISMATCH"
            detail = ""
            if not match:
                detail = (
                    f"  bad_elems={n_bad}/{exp.numel()}"
                    f"  got[:4]={got.flatten()[:4].tolist()}"
                    f"  exp[:4]={exp.flatten()[:4].tolist()}"
                )
            if meta_bad:
                detail += "  meta[" + ",".join(meta_bad) + "]"
            print(f"[receiver] key {key!r}: {status}{detail}", flush=True)
            if status != "OK":
                failures.append((key, "value mismatch" if not match else "metadata"))

    status = "PASS" if not failures else "FAIL"
    print(f"[receiver] MeshSocketWeightBridge: {status}", flush=True)
    assert not failures, f"MeshSocketWeightBridge: mismatches on (key, reason): {failures}"


def _open_parent_and_submesh():
    """Open the [1, 4] parent mesh and carve a [1, 1] submesh at offset (0, 0).

    Returns ``(parent, submesh)``; the caller must release the submesh reference
    before closing the parent (the parent can't close while a submesh still
    holds a command queue).
    """
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*PARENT_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    submesh = parent.create_submesh(
        ttnn.MeshShape(*SUBMESH_SHAPE),
        ttnn.MeshCoordinate(*SUBMESH_OFFSET),
    )
    return parent, submesh


def test_mesh_socket_weight_bridge() -> None:
    """MeshSocketWeightBridge: fabric transfer over one (0,0)->(0,0) socket (1->1)."""
    _ensure_distributed_context()
    if _MPI_RANK == SENDER_RANK:
        parent, submesh = _open_parent_and_submesh()
        try:
            _sender_side(submesh)
        finally:
            submesh = None
            gc.collect()
            ttnn.close_mesh_device(parent)
    elif _MPI_RANK == RECEIVER_RANK:
        parent, submesh = _open_parent_and_submesh()
        try:
            _receiver_side(submesh)
        finally:
            submesh = None
            gc.collect()
            ttnn.close_mesh_device(parent)
    else:
        raise RuntimeError(f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}).")
