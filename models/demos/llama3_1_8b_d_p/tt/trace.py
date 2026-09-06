# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-mode support for the Llama 3.1 8B prefill runtime.

A ttnn trace is recorded ONCE and replayed per chunk, so anything the recorded program reads from a
host Python value is frozen at capture time. Per-chunk values — the cache write offset and the user
slot — must therefore live in **persistent device tensors** that the host updates in place between
replays, and every op that consumes them must have a metadata-tensor form.

Two of the three do::

    ttnn.experimental.deepseek_prefill.update_padded_kv_cache   scalar | metadata-tensor overload
    ttnn.experimental.deepseek_prefill.rotary_embedding_indexed scalar | metadata-tensor overload
    ttnn.transformer.ring_joint_scaled_dot_product_attention    SCALAR ONLY

.. _trace-blocker:

Why chunked prefill cannot be traced today
------------------------------------------

``ring_joint_scaled_dot_product_attention`` — the GQA cache-read attention this model uses — binds
``kv_actual_isl`` and ``kv_cache_batch_idx`` as ``std::optional<uint32_t>``, and ``logical_n`` as a
plain ``std::size_t``. There is no tensor form for any of them. The MLA ring op **does** have one
(``ring_mla`` takes ``slot_id`` and ``kv_actual_isl_tensor``), which is why the MLA packages can
trace chunked prefill and this one cannot: the disaggregated-prefill substrate was built MLA-first
and the GQA path has not caught up.

Capturing anyway would bake chunk 0's ``kv_actual_isl`` and ``logical_n`` into the trace. Chunk 1
would then rotate the KV cache by the wrong offset and mask against the wrong causal bound — wrong
KV, no error. So :func:`assert_traceable` refuses, loudly, rather than let that happen.

What it would take to lift this: add ``slot_id`` / ``kv_actual_isl_tensor`` optionals to
``ring_joint_scaled_dot_product_attention``, mirroring ``ring_mla``'s existing pair, and read them in
the reader kernel where the scalars are read now. The plumbing on this side — persistent buffers,
in-place update, capture/replay — is already here and is exercised by
``tests/unit/test_trace_metadata_vs_ref.py``.

What DOES work today
--------------------

The metadata-tensor path itself: RoPE and the KV-cache write reading this chunk's offset from device
tensors instead of host ints. That is the prerequisite for tracing and is validated to produce
bit-identical KV to the scalar path, so when the op gains its tensor form the remaining change is
small.
"""

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


@dataclass
class TraceMetadata:
    """Persistent per-chunk metadata at fixed device addresses.

    Both are 1-element uint32 ROW_MAJOR DRAM tensors replicated across the mesh — the shape the
    metadata overloads of ``update_padded_kv_cache`` / ``rotary_embedding_indexed`` require. They are
    allocated once and updated **in place**, so the addresses a capture records stay valid.
    """

    slot_idx: ttnn.Tensor
    kv_actual: ttnn.Tensor

    def update(self, *, slot_idx: int, kv_actual: int) -> None:
        """Overwrite both words in place. Must not reallocate — a capture recorded these addresses."""
        _fill(self.slot_idx, slot_idx)
        _fill(self.kv_actual, kv_actual)


def _meta_tensor(mesh_device, value: int) -> ttnn.Tensor:
    """One persistent 1-element uint32 replicated-DRAM metadata scalar."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int32).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _fill(tensor: ttnn.Tensor, value: int) -> None:
    """In-place host write of a single uint32 word, preserving the buffer address."""
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        tensor,
    )


def make_metadata(mesh_device, *, slot_idx: int = 0, kv_actual: int = 0) -> TraceMetadata:
    """Allocate the persistent metadata pair, seeded for the first chunk."""
    return TraceMetadata(slot_idx=_meta_tensor(mesh_device, slot_idx), kv_actual=_meta_tensor(mesh_device, kv_actual))


class TraceUnsupported(NotImplementedError):
    """Raised when a trace capture would silently produce wrong results. See the module docstring."""


def assert_traceable(*, uses_cache_backed_ring: bool, num_users: int) -> None:
    """Refuse to capture a trace that would be silently wrong.

    ``uses_cache_backed_ring``: the chunked path, where the ring SDPA reads the accumulated prefix out
    of the KV cache. Its offsets are host scalars with no tensor form, so a capture freezes chunk 0's
    values — see :ref:`the module docstring <trace-blocker>`.

    ``num_users > 1``: ``kv_cache_batch_idx`` (``slot * num_layers + layer``) is likewise a scalar on
    the ring op, so one capture cannot serve a second user's slot even on the one-shot path.
    """
    if uses_cache_backed_ring:
        raise TraceUnsupported(
            "trace capture refused: chunked prefill uses ring_joint_scaled_dot_product_attention, "
            "whose kv_actual_isl / kv_cache_batch_idx / logical_n are host scalars with no tensor "
            "form. A capture would freeze chunk 0's cache offset and causal bound, and every later "
            "chunk would read the cache at the wrong offset with NO error. Lifting this needs "
            "slot_id / kv_actual_isl_tensor added to the joint ring op, mirroring ring_mla. "
            "Run with use_trace=False (PREFILL_USE_TRACE=0) until then; see tt/trace.py."
        )
    if num_users > 1:
        raise TraceUnsupported(
            f"trace capture refused: num_users={num_users}, but the ring op's kv_cache_batch_idx is a "
            f"host scalar, so one capture cannot serve more than one slot. Use num_users=1 or "
            f"use_trace=False; see tt/trace.py."
        )


def capture(mesh_device, fn, *, cq_id: int = 0) -> int:
    """Record ``fn()`` into a ttnn trace and return its id. The mesh needs a trace_region_size > 0."""
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=cq_id)
    fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=cq_id)
    return tid


def replay(mesh_device, trace_id: int, *, cq_id: int = 0, blocking: bool = False) -> None:
    """Replay a recorded trace. The caller updates the persistent metadata first."""
    ttnn.execute_trace(mesh_device, trace_id, cq_id=cq_id, blocking=blocking)


def release(mesh_device, trace_id: Optional[int], *, cq_id: int = 0) -> None:
    if trace_id is not None:
        ttnn.release_trace(mesh_device, trace_id)
