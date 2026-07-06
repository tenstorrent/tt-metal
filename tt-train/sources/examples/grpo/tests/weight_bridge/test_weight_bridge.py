# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Transport-level unit test for :class:`HostWeightBridge`.

Exercises the sender -> per-submesh fan-out with a small *synthetic* replicated
weight dict -- no HF model, no generate. The shape is 4->4 (4 sender chips ->
4 receiver submeshes): rank 0 (sender) opens a ``[1, 4]`` mesh and sends a
handful of deterministic bf16/TILE/DRAM tensors replicated across it; rank 1
(receiver) opens a ``[1, 4]`` parent, splits it into four ``[1, 1]`` submeshes,
receives, and asserts every submesh holds the exact sent values.

Launched under tt-run with world_size == 2 (see ``runner_weight_bridge.sh``,
``configurations/4_4``). Self-skips otherwise.
"""

from __future__ import annotations

import gc
import math
import os

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_weight_bridge must run under tt-run with world_size == 2 "
        "(use tests/weight_bridge/runner_weight_bridge.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

# Fabric is pinned FABRIC_2D by the autouse _set_fabric_2d fixture in
# tests/conftest.py before any device opens (both ranks must match).
import ttnn  # noqa: E402

from utils.weight_bridge import (  # noqa: E402
    SENDER_RANK,
    RECEIVER_RANK,
    HostWeightBridge,
)

# 4->4: 4 sender chips -> 4 receiver submeshes.
SENDER_SHAPE = (1, 4)
NUM_SUBMESHES = 4

# A few tile-aligned tensor specs (rows/cols multiples of 32 -> no tile padding).
_SPECS = {
    "weight.alpha": (1, 1, 32, 64),
    "weight.beta": (1, 1, 64, 96),
    "weight.gamma": (1, 1, 128, 32),
}


def _synthetic_torch() -> dict:
    """Deterministic per-key torch bf16 tensors (both ranks compute identically)."""
    import torch

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
    bridge = HostWeightBridge.init_sender(mesh=mesh, peer_rank=RECEIVER_RANK)
    bridge.connect()
    print(f"[sender] HostWeightBridge: sending {len(weights)} tensors", flush=True)
    bridge.send_weights(weights)
    bridge.barrier()
    print("[sender] HostWeightBridge: done", flush=True)


def _receiver_side(parent, submeshes, num_submeshes) -> None:
    import torch

    expected = _synthetic_torch()
    bridge = HostWeightBridge.init_receiver(mesh=parent, peer_rank=SENDER_RANK, submeshes=submeshes)
    bridge.connect()
    per_submesh = bridge.receive_weights()
    bridge.barrier()

    assert isinstance(per_submesh, list), f"receive_weights must return a list, got {type(per_submesh)}"
    assert len(per_submesh) == num_submeshes, f"expected {num_submeshes} dicts, got {len(per_submesh)}"

    # Report every (submesh == physical device, key) instead of aborting on the
    # first mismatch, so a single run shows which devices hold good data.
    failures = []  # (submesh_index, key, reason)
    for i, got_dict in enumerate(per_submesh):
        if sorted(got_dict.keys()) != sorted(expected.keys()):
            print(
                f"[receiver] submesh {i} (device {i}): KEY MISMATCH "
                f"{sorted(got_dict.keys())} != {sorted(expected.keys())}",
                flush=True,
            )
            failures.append((i, "<keys>", "key set mismatch"))
            continue
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
                    f"[receiver] submesh {i} (device {i}) key {key!r}: SHAPE "
                    f"{tuple(got.shape)} != {tuple(exp.shape)}",
                    flush=True,
                )
                failures.append((i, key, "shape mismatch"))
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
            print(f"[receiver] submesh {i} (device {i}) key {key!r}: {status}{detail}", flush=True)
            if status != "OK":
                failures.append((i, key, "value mismatch" if not match else "metadata"))

    ok_submeshes = sorted(set(range(num_submeshes)) - {i for i, _, _ in failures})
    bad_submeshes = sorted({i for i, _, _ in failures})
    print(f"[receiver] HostWeightBridge: PASS submeshes={ok_submeshes}  FAIL submeshes={bad_submeshes}", flush=True)
    assert not failures, f"HostWeightBridge: mismatches on (submesh, key, reason): {failures}"


def test_host_weight_bridge() -> None:
    """HostWeightBridge: MPI byte transfer, host-upload to each submesh (4->4)."""
    _ensure_distributed_context()
    if _MPI_RANK == SENDER_RANK:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*SENDER_SHAPE), offset=ttnn.MeshCoordinate(0, 0))
        try:
            _sender_side(mesh)
        finally:
            ttnn.close_mesh_device(mesh)
    elif _MPI_RANK == RECEIVER_RANK:
        parent = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, NUM_SUBMESHES),
            offset=ttnn.MeshCoordinate(0, 0),
        )
        submeshes = parent.create_submeshes(ttnn.MeshShape(1, 1))
        assert len(submeshes) == NUM_SUBMESHES, f"expected {NUM_SUBMESHES} submeshes, got {len(submeshes)}"
        try:
            _receiver_side(parent, submeshes, NUM_SUBMESHES)
        finally:
            # Release the child submeshes before closing the parent: a parent
            # mesh cannot close while a submesh still holds its command queue.
            submeshes = None
            gc.collect()
            ttnn.close_mesh_device(parent)
    else:
        raise RuntimeError(f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}).")
