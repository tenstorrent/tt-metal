# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-submesh weight transport: one TTML DDP mesh -> N TTT submeshes.

This is the multi-socket sibling of :mod:`utils.inference_bridge`'s
``WeightBridge``. It implements the "DDP on TTML, one submesh per device
on TTT" topology discussed in the grpo speedup design:

* **TTML rank (sender)** owns a single ``[1, N]`` ``MeshDevice`` running
  DDP. Every weight is *replicated* across the N devices (DDP only shards
  the input batch and all-reduces gradients; ordinary modules keep
  replicated weights -- see ``llama_overrides.export_to_hf_dict``).

* **TTT rank (receiver)** owns a ``[1, N]`` parent ``MeshDevice`` that it
  splits into N independent ``[1, 1]`` submeshes via
  ``parent.create_submeshes(MeshShape(1, 1))`` -- one model per submesh,
  the layout tt-transformers wants for independent data-parallel
  rollouts.

Transport is **point-to-point per device**: we open N ``MeshSocket``s,
where ``socket_i`` carries a single connection ``TTML(0, i) -> submesh_i(0, 0)``.
``send_async(tensor, socket_i)`` reads the replicated tensor's copy at
TTML device ``i`` and delivers it to submesh ``i``. This is the
``ttml_i -> ttt_i`` mapping: N parallel fabric links instead of funnelling
the whole model through one ``(0,0) -> (0,0)`` link.

Why N sockets instead of one cartesian socket onto the parent mesh? A
tensor received onto the parent ``[1, N]`` mesh carries *parent*-mesh
topology, and feeding it to a per-submesh model's ``update_weights``
(``ttnn.copy``) would cross ``MeshDevice`` objects -- not a supported
path without a host round-trip. Receiving straight onto each submesh
keeps the source and destination on the same device.

Wire protocol (mirrors ``WeightBridge``)::

    1. MPI handshake before each socket open (pins both ranks so the
       MeshSocket descriptor exchange does not trip its 10s timeout).
    2. Length-prefixed JSON manifest over MPI: per-key shape / dtype /
       layout, plus the byte length of each reference blob.
    3. The sender's *reference tensors* for every key over MPI, one
       replicated copy each, serialized with ``torch.save``. The receiver
       reloads them with ``torch.load`` and compares with ``torch.equal``
       against what landed on each submesh. This is a verification aid for
       the test; a production path would drop it.
    4. send_async / recv_async per (key, socket).
    5. Post-transfer MPI barrier so the sender keeps its source tensors
       alive until the receiver has drained them.

Both ranks must call the symmetric entry points the same number of
times and in the same order. The sender calls :func:`open_sender_sockets`
+ :func:`send_weights`; the receiver calls :func:`open_receiver_sockets`
+ :func:`recv_weights` + :func:`verify_received`.
"""

from __future__ import annotations

import io
import json
import math
import struct
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

from ttnn._ttnn.multi_device import recv_bytes as _mpi_recv_bytes
from ttnn._ttnn.multi_device import send_bytes as _mpi_send_bytes

import ttnn

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


# Roles / ranks -- shared with utils.inference_bridge.
TTML_RANK: int = 0
TTT_RANK: int = 1

_ROLE_TTML: str = "ttml"
_ROLE_TTT: str = "ttt"

# MPI tag namespace. Disjoint from the inference RPC tags (10..13) in
# inference_bridge so the two protocols could in principle coexist; reuses
# the WeightBridge manifest/handshake tags (1..4) because this module is
# never run alongside a live WeightBridge.
_MANIFEST_LEN_TAG: int = 1
_MANIFEST_BODY_TAG: int = 2
_HANDSHAKE_TAG_FROM_TTML: int = 3
_HANDSHAKE_TAG_FROM_TTT: int = 4
_REF_BLOB_TAG: int = 5

_HANDSHAKE_PAYLOAD: bytes = b"ready"

# 80 MB DRAM fifo, identical to WeightBridge / ttml's socket_manager.cpp.
_SOCKET_FIFO_BYTES: int = 10 * 1024 * 1024 * 2 * 4


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _handshake(role: str, peer_rank: int) -> None:
    """Two-rank barrier: each side eager-sends, then blocks on recv."""
    if role == _ROLE_TTML:
        _mpi_send_bytes(_HANDSHAKE_PAYLOAD, peer_rank, _HANDSHAKE_TAG_FROM_TTML)
        _mpi_recv_bytes(len(_HANDSHAKE_PAYLOAD), peer_rank, _HANDSHAKE_TAG_FROM_TTT)
    else:
        _mpi_send_bytes(_HANDSHAKE_PAYLOAD, peer_rank, _HANDSHAKE_TAG_FROM_TTT)
        _mpi_recv_bytes(len(_HANDSHAKE_PAYLOAD), peer_rank, _HANDSHAKE_TAG_FROM_TTML)


def barrier(role: str, peer_rank: int) -> None:
    """Public post-transfer fence (same primitive as the handshake)."""
    _handshake(role, peer_rank)


def _shape_to_list(shape) -> List[int]:
    return [int(d) for d in shape]


def _make_socket_mem_config() -> "ttnn.SocketMemoryConfig":
    return ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, _SOCKET_FIFO_BYTES)


def _socket_config_for_device(i: int) -> "ttnn.SocketConfig":
    """SocketConfig for the i-th link: TTML(0, i) -> submesh_i(0, 0).

    The sender coordinate ``(0, i)`` is a coordinate in the TTML ``[1, N]``
    mesh; the receiver coordinate ``(0, 0)`` is the sole coordinate of the
    i-th ``[1, 1]`` submesh. Both ranks build the *same* SocketConfig --
    only the ``MeshDevice`` the socket is opened on differs (the full TTML
    mesh on the sender, ``submesh_i`` on the receiver).
    """
    sender = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, i), ttnn.CoreCoord(0, 0))
    receiver = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    return ttnn.SocketConfig(
        [ttnn.SocketConnection(sender, receiver)],
        _make_socket_mem_config(),
        sender_rank=TTML_RANK,
        receiver_rank=TTT_RANK,
    )


def _is_fully_replicated(tensor: "ttnn.Tensor") -> bool:
    placements = tensor.tensor_topology().placements()
    return all(isinstance(p, ttnn.PlacementReplicate) for p in placements)


def _validate_source_tensor(key: str, tensor: "ttnn.Tensor") -> None:
    """Enforce the Transformer.update_weights contract before the wire."""
    if tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"submesh transfer: tensor {key!r} dtype={tensor.dtype}, expected bfloat16.")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"submesh transfer: tensor {key!r} layout={tensor.layout}, expected TILE_LAYOUT.")
    if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(f"submesh transfer: tensor {key!r} not DRAM_MEMORY_CONFIG ({tensor.memory_config()}).")
    if not _is_fully_replicated(tensor):
        placements = tensor.tensor_topology().placements()
        raise ValueError(
            f"submesh transfer: tensor {key!r} is not fully replicated (placements={placements}). "
            "DDP on the ttml side must keep weights replicated; TP/CP is not supported here."
        )


def device0_to_torch(tensor: "ttnn.Tensor") -> "torch.Tensor":
    """Bring device 0's buffer of ``tensor`` to host as a torch tensor.

    ``get_device_tensors(...)[0]`` selects the first per-device shard so
    this works uniformly for a replicated ``[1, N]`` tensor and a
    ``[1, 1]`` submesh tensor.
    """
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _all_devices_equal(tensor: "ttnn.Tensor") -> bool:
    """True iff every per-device buffer of ``tensor`` is torch-equal."""
    import torch

    per_device = [ttnn.to_torch(dev) for dev in ttnn.get_device_tensors(tensor)]
    return all(torch.equal(t, per_device[0]) for t in per_device[1:])


def _torch_save_bytes(t: "torch.Tensor") -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _torch_load_bytes(blob: bytes) -> "torch.Tensor":
    import torch

    return torch.load(io.BytesIO(blob), weights_only=True)


# ---------------------------------------------------------------------------
# Socket opening
# ---------------------------------------------------------------------------


def open_sender_sockets(ttml_mesh: "ttnn.MeshDevice", num_submeshes: int) -> List["ttnn.MeshSocket"]:
    """Open the N sender-side sockets on the TTML rank.

    ``num_submeshes`` must equal the number of TTT submeshes and the
    TTML mesh's device count (one link per device).
    """
    if int(ttml_mesh.shape[0]) != 1 or int(ttml_mesh.shape[1]) != num_submeshes:
        raise ValueError(
            f"open_sender_sockets: expected a [1, {num_submeshes}] TTML mesh, "
            f"got shape={_shape_to_list(ttml_mesh.shape)}."
        )
    sockets: List[ttnn.MeshSocket] = []
    for i in range(num_submeshes):
        _handshake(_ROLE_TTML, TTT_RANK)
        sockets.append(ttnn.MeshSocket(ttml_mesh, _socket_config_for_device(i)))
    return sockets


def open_receiver_sockets(submeshes: List["ttnn.MeshDevice"]) -> List["ttnn.MeshSocket"]:
    """Open the N receiver-side sockets on the TTT rank, one per submesh."""
    sockets: List[ttnn.MeshSocket] = []
    for i, submesh in enumerate(submeshes):
        _handshake(_ROLE_TTT, TTML_RANK)
        sockets.append(ttnn.MeshSocket(submesh, _socket_config_for_device(i)))
    return sockets


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


def send_weights(
    ttml_mesh: "ttnn.MeshDevice",
    hf_dict: Dict[str, "ttnn.Tensor"],
    sockets: List["ttnn.MeshSocket"],
) -> dict:
    """Send the full HF-keyed weight dict to every submesh.

    Also ships, over MPI, one reference copy of each weight (taken from
    device 0, ``torch.save``-serialized) so the receiver can prove
    equality with ``torch.equal``. Each tensor is validated (replicated /
    DRAM / TILE / bf16) and its N device copies are asserted torch-equal --
    a direct check that DDP really replicated the weight.

    Returns the manifest dict that was put on the wire.
    """
    keys = sorted(hf_dict.keys())
    print(
        f"[submesh-transfer] send: validating + serializing {len(keys)} tensors "
        "to host (replication check brings every device copy to host -- can be slow)...",
        flush=True,
    )
    entries = []
    blobs: Dict[str, bytes] = {}
    for k in keys:
        tensor = hf_dict[k]
        _validate_source_tensor(k, tensor)
        if not _all_devices_equal(tensor):
            raise ValueError(
                f"send_weights: tensor {k!r} differs across DDP devices. Weights must be replicated for this transport."
            )
        blob = _torch_save_bytes(device0_to_torch(tensor))
        blobs[k] = blob
        entries.append(
            {
                "key": k,
                "shape": _shape_to_list(tensor.shape),
                "dtype": tensor.dtype.name,
                "layout": tensor.layout.name,
                "ref_len": len(blob),
            }
        )

    manifest = {"version": 1, "num_submeshes": len(sockets), "entries": entries}
    body = json.dumps(manifest).encode("utf-8")
    print(f"[submesh-transfer] send: sending manifest ({len(body)} B, {len(entries)} entries)", flush=True)
    _mpi_send_bytes(struct.pack("<Q", len(body)), TTT_RANK, _MANIFEST_LEN_TAG)
    _mpi_send_bytes(body, TTT_RANK, _MANIFEST_BODY_TAG)

    # Reference blobs, sorted-key order (matches the receiver's recv).
    total_ref = sum(len(b) for b in blobs.values())
    print(
        f"[submesh-transfer] send: sending {len(keys)} reference blobs (~{total_ref / 1e6:.0f} MB) over MPI", flush=True
    )
    for k in keys:
        _mpi_send_bytes(blobs[k], TTT_RANK, _REF_BLOB_TAG)

    # Per-socket FIFO: for a fixed socket the send order is the sorted key
    # order, matching the receiver's recv order below.
    print(f"[submesh-transfer] send: streaming {len(keys)} tensors x {len(sockets)} sockets (send_async)", flush=True)
    send_start = time.perf_counter()
    for k in keys:
        for socket in sockets:
            ttnn.experimental.send_async(hf_dict[k], socket)
        # Per-key sync: keep <=1 send_async outstanding per socket, in lockstep with the
        # receiver's per-entry recv sync. Issuing all sends before a single sync floods the
        # concurrent submesh sockets (>2 outstanding/socket) and deadlocks the fabric.
        ttnn.synchronize_device(ttml_mesh)
    send_elapsed = time.perf_counter() - send_start
    # bf16 => 2 bytes/elem; each socket gets the full (replicated) weight set.
    gib_per_socket = sum(math.prod(_shape_to_list(hf_dict[k].shape)) for k in keys) * 2 / 1024**3
    gib_total = gib_per_socket * len(sockets)
    rate = gib_total * 1024 / send_elapsed if send_elapsed > 0 else 0.0
    print(
        f"[submesh-transfer] send: done -- weight transfer took {send_elapsed:.2f}s for "
        f"{len(keys)} tensors x {len(sockets)} sockets "
        f"({gib_total:.2f} GiB total, {gib_per_socket:.2f} GiB/socket, {rate:.0f} MiB/s aggregate)",
        flush=True,
    )
    return manifest


def recv_weights(
    submeshes: List["ttnn.MeshDevice"],
    sockets: List["ttnn.MeshSocket"],
) -> Tuple[List[Dict[str, "ttnn.Tensor"]], Dict[str, "torch.Tensor"], dict]:
    """Receive the weight dict onto each submesh.

    Returns ``(per_submesh_dicts, reference, manifest)``:

    * ``per_submesh_dicts[i]`` -- HF-keyed dict of on-device tensors
      living on ``submeshes[i]`` (ready for that submesh's
      ``Transformer.update_weights``).
    * ``reference`` -- ``key -> sender torch tensor`` for ``torch.equal``.
    * ``manifest`` -- the full manifest dict.
    """
    print("[submesh-transfer] recv: waiting for manifest...", flush=True)
    header = _mpi_recv_bytes(8, TTML_RANK, _MANIFEST_LEN_TAG)
    (n,) = struct.unpack("<Q", header)
    body = _mpi_recv_bytes(int(n), TTML_RANK, _MANIFEST_BODY_TAG)
    manifest = json.loads(body.decode("utf-8"))
    n_entries = len(manifest["entries"])
    print(
        f"[submesh-transfer] recv: got manifest ({n_entries} entries); receiving reference blobs over MPI...",
        flush=True,
    )

    reference: Dict[str, "torch.Tensor"] = {}
    for entry in manifest["entries"]:
        blob = _mpi_recv_bytes(int(entry["ref_len"]), TTML_RANK, _REF_BLOB_TAG)
        reference[entry["key"]] = _torch_load_bytes(blob)
    print(
        f"[submesh-transfer] recv: {len(reference)} reference blobs received; "
        f"receiving tensors over {len(sockets)} sockets (recv_async)...",
        flush=True,
    )

    per_submesh: List[Dict[str, ttnn.Tensor]] = [{} for _ in submeshes]
    recv_start = time.perf_counter()
    for j, entry in enumerate(manifest["entries"]):
        spec = ttnn.TensorSpec(
            entry["shape"],
            getattr(ttnn.DataType, entry["dtype"]),
            getattr(ttnn.Layout, entry["layout"]),
        )
        for i, socket in enumerate(sockets):
            # Verbose on the first entry (so a hang on socket 1's untested
            # (0,1)->(0,1) routing is obvious) and then periodically.
            if j == 0 or (j + 1) % 25 == 0:
                print(
                    f"[submesh-transfer] recv: recv_async entry {j + 1}/{n_entries} socket {i} key={entry['key']}",
                    flush=True,
                )
            template = ttnn.allocate_tensor_on_device(spec, submeshes[i])
            received = ttnn.experimental.recv_async(template, socket)
            tensor = received[0] if isinstance(received, (list, tuple)) and received else template
            per_submesh[i][entry["key"]] = tensor

        for submesh in submeshes:
            ttnn.synchronize_device(submesh)

        print("[submesh-transfer] submeshes syncronized, next entry")

    recv_elapsed = time.perf_counter() - recv_start
    # bf16 => 2 bytes/elem; each socket receives the full (replicated) weight set.
    gib_per_socket = sum(math.prod(e["shape"]) for e in manifest["entries"]) * 2 / 1024**3
    gib_total = gib_per_socket * len(sockets)
    rate = gib_total * 1024 / recv_elapsed if recv_elapsed > 0 else 0.0
    print(
        f"[submesh-transfer] recv: weight transfer took {recv_elapsed:.2f}s for "
        f"{n_entries} tensors x {len(sockets)} sockets "
        f"({gib_total:.2f} GiB total, {gib_per_socket:.2f} GiB/socket, {rate:.0f} MiB/s aggregate)",
        flush=True,
    )
    print("[submesh-transfer] recv: done", flush=True)
    return per_submesh, reference, manifest


def verify_received(
    per_submesh: List[Dict[str, "ttnn.Tensor"]],
    reference: Dict[str, "torch.Tensor"],
) -> List[dict]:
    """Bring each received tensor to host and ``torch.equal`` it with the
    sender's reference.

    Returns a flat list of ``{"submesh", "key", "equal"}`` rows, one per
    (submesh, key). ``equal`` is exact tensor equality between the tensor
    that landed on the submesh and the sender's reference copy.
    """
    import torch

    report: List[dict] = []
    for i, submesh_dict in enumerate(per_submesh):
        for key, tensor in submesh_dict.items():
            got = device0_to_torch(tensor)
            report.append(
                {
                    "submesh": i,
                    "key": key,
                    "equal": torch.equal(got, reference[key]),
                }
            )
    return report
