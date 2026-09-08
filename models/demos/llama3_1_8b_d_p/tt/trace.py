# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-mode support for the Llama 3.1 8B prefill runtime.

A ttnn trace is recorded ONCE and replayed per chunk, so anything the recorded program reads from a
host Python value is frozen at capture time. Per-chunk values — the cache write offset and the user
slot — must therefore live in **persistent device tensors** that the host updates in place between
replays, and every op that consumes them must have a metadata-tensor form.

All three do::

    ttnn.experimental.deepseek_prefill.update_padded_kv_cache   scalar | metadata-tensor overload
    ttnn.experimental.deepseek_prefill.rotary_embedding_indexed scalar | metadata-tensor overload
    ttnn.transformer.ring_joint_scaled_dot_product_attention    scalar | metadata-tensor overload

.. _trace-blocker:

Correction: the GQA ring op is NOT the blocker it was documented as
-------------------------------------------------------------------

Earlier revisions of this module (and ``docs/SPEC_NOTES.md`` §8d) stated that
``ring_joint_scaled_dot_product_attention`` had no metadata-tensor form and that chunked GQA prefill
therefore could not be traced. **That was wrong**, and the error came from reading the model-side
wrapper rather than the op.

``ring_joint_scaled_dot_product_attention`` and ``ring_mla`` are two front-ends over the *same*
primitive (``ttnn::prim::ring_joint_scaled_dot_product_attention``); MLA is simply the latent-V
configuration. The primitive takes ``slot_id`` and ``kv_actual_isl_tensor`` as optional 1-element
uint32 tensors and reads **both on-device** — ``kv_cache_batch_idx`` is folded as
``slot_id[0] * kv_cache_num_layers + kv_cache_layer_idx`` in the all-gather reader, and
``logical_nt`` / q-mapping / ring masks are derived from ``kv_actual_isl`` in the SDPA reader. So one
capture replays across chunks. Upstream's own coverage asserts this: the metadata path is bit-exact
against the host-scalar path, with one capture and fifteen replays in forward, reverse and
out-of-order sequence.

Using it correctly requires two things together, or the capture is silently wrong:

* **Withhold the host scalars.** A supplied ``kv_actual_isl`` re-enables the host-side valid-pages
  patch, which a trace freezes at chunk 0's value.
* **Make ``logical_n`` a constant.** It remains a host argument, so it must not carry per-chunk
  information; the cache's global capacity is the natural choice, identical for every chunk. See
  ``tt/attention/dense_sp.py``, which switches both on the presence of the metadata tensors.

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

    MEASURED, not assumed. With the ring_joint front-end patched to forward slot_id /
    kv_actual_isl_tensor (see the module docstring), a chunked capture now *succeeds* and replays
    13.9x faster than eager dispatch (285 ms vs 3962 ms for 4 x 4096). It is also **wrong**:
    ``tests/galaxy_prefill_kv_pcc.py`` with ``PREFILL_USE_TRACE=1`` gives per-layer KV PCC of
    0.10-0.35 against a 0.99 gate, on all 32 layers.

    Layer 0 is wrong too (K 0.352 / V 0.073), which rules out per-chunk offset drift -- a frozen
    offset would still leave chunk 0 correct. The corruption is total from the first layer, pointing
    at the causal geometry rather than the cache index: on the metadata path ``logical_n`` must be a
    per-chunk constant (the cache capacity) and the op is supposed to derive ``logical_nt`` on-device
    from ``kv_actual_isl``. Upstream validates that derivation only for ``ring_mla`` -- the latent-V
    configuration -- and the GQA/explicit-V configuration evidently is not covered.

    So the original conclusion in ``docs/SPEC_NOTES.md`` §8d stands: chunked GQA prefill is not
    traceable today. The *reason* recorded there was wrong (the kwargs were missing from the
    ring_joint front-end, not the primitive), and lifting that is necessary but not sufficient.
    """
    if uses_cache_backed_ring:
        raise TraceUnsupported(
            "trace capture refused: chunked GQA prefill replays 13.9x faster but produces WRONG KV "
            "(per-layer PCC 0.10-0.35 vs a 0.99 gate, all 32 layers, layer 0 included). The "
            "ring_joint front-end now forwards slot_id / kv_actual_isl_tensor, so the capture "
            "succeeds -- but the on-device geometry derivation is validated upstream only for "
            "ring_mla (latent-V), and the explicit-V GQA path does not reproduce the scalar result. "
            "Reproduce with: PREFILL_USE_TRACE=1 PREFILL_CHUNKED=1 pytest "
            "models/demos/llama3_1_8b_d_p/tests/galaxy_prefill_kv_pcc.py -k 8x4. "
            "Run with use_trace=False (PREFILL_USE_TRACE=0); see tt/trace.py."
        )
    if num_users > 1:
        raise TraceUnsupported(
            f"trace capture refused: num_users={num_users}. The slot does live in a metadata tensor, "
            f"but multi-slot replay has never been validated in this package. Use num_users=1."
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
