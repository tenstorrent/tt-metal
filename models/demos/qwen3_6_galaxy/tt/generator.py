# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen36Generator — generator wrapper for TtQwen36Transformer — Task T14.

Public API
----------

    gen = Qwen36Generator(tt_model, mesh_device, args)
    logits = gen.prefill_forward(input_ids, page_table=None, enable_trace=False)
    logits, kv, dn, cv = gen.prefill_forward_with_caches(...)
    logits, kv, dn, cv = gen.decode_forward_with_caches(input_ids, current_pos, ...)

The generator provides a stable surface that callers (vLLM plugin, demos,
tt-inference-server) can adopt today and that will transparently gain trace
capture once the forward path is refactored to be host-write-free.

Trace capture status (BLOCKED — precondition not yet met)
---------------------------------------------------------
The intent of T14 is to wrap ``TtQwen36Transformer.forward_prefill`` and
``forward_decode`` with ``ttnn.begin_trace_capture`` / ``ttnn.execute_trace`` so
that the host launch overhead is amortised across many decode steps.

The current forward path contains multiple ``ttnn.from_torch`` /
``ttnn.to_torch`` calls per layer (token-id embedding upload, distributed-norm
shard/gather around ``DistributedNorm`` in ``llama_decoder._shard_across_cols``
and ``_gather_from_cols``, attention mask creation, per-decode ``current_pos``
upload, final logits gather).  Every such call is a host-write to a device
buffer, which TTNN forbids inside an active trace.  Empirical evidence on the
8×4 BH GLX mesh::

    TT_FATAL: Writes are not supported during trace capture. trace id: 0

Worse, when this fires from inside ``begin_trace_capture`` / ``end_trace_capture``
the device state is left half-committed: subsequent ``ttnn.synchronize_device``
calls hang.  As a result we cannot even safely *attempt* capture and fall back —
the attempt itself leaves the mesh in a bad state requiring ``tt-smi -r``.

Concrete call sites that must become trace-safe before
``begin_trace_capture`` can wrap the forward:

  - ``models/demos/qwen3_6_galaxy/tt/llama_decoder.py:_shard_across_cols`` and
    ``_gather_from_cols`` — used twice per layer around DistributedNorm.
  - ``models/demos/qwen3_6_galaxy/tt/llama_model.py:_embed`` — uploads
    input_ids per call.
  - ``models/demos/qwen3_6_galaxy/tt/llama_model.py:_norm_and_lm_head`` —
    final ``ttnn.to_torch`` for logits gather.  **Moved outside the trace
    boundary in T14b.1**: pass ``output_as_ttnn=True`` to
    ``forward_prefill`` / ``forward_decode`` and the generator's execute
    path will gather to CPU AFTER ``ttnn.execute_trace`` returns.  The
    capture path now uses this kwarg.
  - ``models/demos/qwen3_6_galaxy/tt/llama_attention.py`` — causal mask,
    ``current_pos`` SDPA tensor, update-index tensor (all via ``from_torch``).
  - ``models/demos/qwen3_6_galaxy/tt/qwen36_deltanet.py:_causal_conv1d_fir_mesh``
    — zero pad allocation on first call.

These call sites were all introduced for correctness during T7..T13 and the
existing model achieves PCC > 0.99 with them in place.  Replacing them
requires:

  1. On-device sharding for the residual stream (hold ``x`` sharded across
     cols throughout the decoder, eliminate the shard/gather around the norm).
  2. Pre-allocated, persistent device buffers for input_ids / page_table /
     current_pos / mask, refreshed each iteration via
     ``ttnn.copy_host_to_device_tensor`` (this primitive IS trace-safe — it
     records a metadata-only buffer rebinding, not a host write).
  3. Logits gather moved outside the trace boundary.

This refactor is tracked as the T14b follow-on.  Until it lands the
generator's ``_TRACE_SUPPORTED`` class attribute is ``False`` and trace
attempts are skipped entirely — we run the eager forward in all cases.

When the refactor lands, flip ``_TRACE_SUPPORTED`` to ``True`` and the
existing capture / execute paths in this file will activate without any
other changes.

Behaviour
---------
``enable_trace=False`` calls ``model.forward_prefill`` / ``model.forward_decode``
directly.

``enable_trace=True`` currently also runs the eager forward (because
``_TRACE_SUPPORTED=False``), so callers can adopt the generator API today
without behavioural change.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

import ttnn

logger = logging.getLogger(__name__)


# Sentinel marking a trace-capture key that previously failed.  We never retry
# capture for that key (would just fail again).
_NO_TRACE_FALLBACK = "no-trace-fallback"


class Qwen36Generator:
    """Generator wrapper around TtQwen36Transformer.

    Parameters
    ----------
    model : TtQwen36Transformer
        The transformer model whose ``forward_prefill`` / ``forward_decode``
        methods provide the no-trace baseline.
    mesh_device : ttnn.MeshDevice
        The full 8×4 BH GLX mesh device.
    args : TtQwen36ModelArgs
        Model configuration (cluster_shape, max_batch_size, etc.).

    Attributes
    ----------
    model, mesh_device, args
        As above.
    _prefill_traces, _decode_traces : dict
        Trace caches keyed by ``(B, T, has_page_table)`` / ``(B, has_page_table)``.
        Values are dicts with ``trace_id`` / ``captured``, or
        ``_NO_TRACE_FALLBACK`` marking a key whose capture failed.
    """

    # When True, attempt trace capture for ``enable_trace=True`` callers.
    # Currently False because the forward path contains host-writes that the
    # TTNN trace machinery forbids (see module docstring).  Flip to True after
    # the forward path is refactored to be host-write-free.
    _TRACE_SUPPORTED: bool = False

    def __init__(self, model, mesh_device, args):
        self.model = model
        self.mesh_device = mesh_device
        self.args = args
        self._prefill_traces: dict = {}
        self._decode_traces: dict = {}

    # ------------------------------------------------------------------
    # Private: gather a captured on-device logits tensor to CPU.  Used by
    # the trace execute path AFTER ``ttnn.execute_trace`` returns, so this
    # ``ttnn.to_torch`` call is OUTSIDE the trace boundary.  Mirrors the
    # eager gather in ``TtQwen36Transformer._norm_and_lm_head``.
    # ------------------------------------------------------------------

    def _gather_logits_to_cpu(self, logits_tt) -> torch.Tensor:
        """Gather an on-device logits tensor (per-chip [B, T, V/cols]) to a
        CPU [B, T, padded_vocab] float32 torch tensor.  Caller-side companion
        to ``output_as_ttnn=True`` on the model forward.
        """
        cluster_shape = list(self.args.cluster_shape)
        logits_cpu_concat = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device,
                dims=(0, -1),
                mesh_shape=cluster_shape,
            ),
        )  # [n_rows*B, T, padded_vocab]
        n_rows = cluster_shape[0]
        B = logits_cpu_concat.shape[0] // n_rows
        return logits_cpu_concat[:B].float()

    # ------------------------------------------------------------------
    # Public: prefill
    # ------------------------------------------------------------------

    def prefill_forward(
        self,
        input_ids: torch.Tensor,
        page_table=None,
        enable_trace: bool = True,
    ) -> torch.Tensor:
        """Run prefill and return logits (CPU torch tensor [B, T, padded_vocab])."""
        logits, _, _, _ = self.prefill_forward_with_caches(input_ids, page_table=page_table, enable_trace=enable_trace)
        return logits

    def prefill_forward_with_caches(
        self,
        input_ids: torch.Tensor,
        page_table=None,
        enable_trace: bool = True,
    ) -> Tuple[torch.Tensor, List, List, List]:
        """Run prefill and return ``(logits, kv_caches, dn_states, conv_states)``.

        Always runs the eager forward to produce the result.  When
        ``enable_trace=True`` and the generator's ``_TRACE_SUPPORTED`` flag is
        True, additionally attempts trace capture for that key.  See module
        docstring for the host-write precondition that currently disables
        trace capture entirely.
        """
        # Always run eager forward for the actual result.
        result = self.model.forward_prefill(
            input_ids,
            return_caches=True,
            page_table=page_table,
        )

        if not enable_trace or not self._TRACE_SUPPORTED:
            return result

        B, T = input_ids.shape
        has_pt = page_table is not None
        key = (B, T, has_pt)
        cache = self._prefill_traces.get(key)
        if cache is None:
            cache = self._try_capture_prefill(input_ids, page_table)
            self._prefill_traces[key] = cache
        # If capture succeeded, optionally execute the trace and use its output.
        if cache != _NO_TRACE_FALLBACK and isinstance(cache, dict):
            try:
                ttnn.execute_trace(self.mesh_device, cache["trace_id"], cq_id=0, blocking=True)
                # captured = (logits_tt, kv_caches, dn_states, conv_states).
                # logits_tt is an on-device tensor (output_as_ttnn=True at
                # capture time); gather it to CPU AFTER execute_trace so the
                # host-write is outside the trace boundary.
                logits_tt, kv_c, dn_c, cv_c = cache["captured"]
                logits_cpu = self._gather_logits_to_cpu(logits_tt)
                return logits_cpu, kv_c, dn_c, cv_c
            except Exception as exc:
                logger.warning(f"[Qwen36Generator] execute_trace (prefill) failed: {exc}")
        return result

    # ------------------------------------------------------------------
    # Public: decode
    # ------------------------------------------------------------------

    def decode_forward(
        self,
        input_ids: torch.Tensor,
        current_pos: int,
        page_table=None,
        kv_caches: Optional[List] = None,
        dn_states: Optional[List] = None,
        conv_states: Optional[List] = None,
        enable_trace: bool = True,
    ) -> torch.Tensor:
        """Run a single decode step and return logits (CPU torch tensor)."""
        logits, _, _, _ = self.decode_forward_with_caches(
            input_ids,
            current_pos=current_pos,
            page_table=page_table,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
            enable_trace=enable_trace,
        )
        return logits

    def decode_forward_with_caches(
        self,
        input_ids: torch.Tensor,
        current_pos: int,
        page_table=None,
        kv_caches: Optional[List] = None,
        dn_states: Optional[List] = None,
        conv_states: Optional[List] = None,
        enable_trace: bool = True,
    ) -> Tuple[torch.Tensor, List, List, List]:
        """Run a decode step and return ``(logits, kv, dn, conv)``.

        Always runs the eager forward.  ``enable_trace=True`` additionally
        attempts trace capture per ``(B, has_page_table)`` key.  See module
        docstring for the host-write precondition.
        """
        B, T = input_ids.shape
        assert T == 1, f"decode_forward_with_caches expects T=1, got T={T}"

        # Always run eager forward for the actual result.
        result = self.model.forward_decode(
            input_ids,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
        )

        if not enable_trace or not self._TRACE_SUPPORTED:
            return result

        has_pt = page_table is not None
        key = (B, has_pt)
        cache = self._decode_traces.get(key)
        if cache is None:
            cache = self._try_capture_decode(
                input_ids,
                current_pos=current_pos,
                page_table=page_table,
                kv_caches=kv_caches,
                dn_states=dn_states,
                conv_states=conv_states,
            )
            self._decode_traces[key] = cache
        if cache != _NO_TRACE_FALLBACK and isinstance(cache, dict):
            try:
                ttnn.execute_trace(self.mesh_device, cache["trace_id"], cq_id=0, blocking=True)
                # captured = (logits_tt, kv_caches, dn_states, conv_states).
                # Gather logits AFTER execute_trace so the host-write is
                # outside the trace boundary.
                logits_tt, kv_c, dn_c, cv_c = cache["captured"]
                logits_cpu = self._gather_logits_to_cpu(logits_tt)
                return logits_cpu, kv_c, dn_c, cv_c
            except Exception as exc:
                logger.warning(f"[Qwen36Generator] execute_trace (decode) failed: {exc}")
        return result

    # ------------------------------------------------------------------
    # Private: capture (best-effort)
    # ------------------------------------------------------------------

    def _try_capture_prefill(self, input_ids, page_table):
        """Attempt prefill trace capture.  Returns trace dict or sentinel."""
        try:
            return self._capture_prefill_trace(input_ids, page_table)
        except Exception as exc:
            logger.warning(
                f"[Qwen36Generator] Prefill trace capture failed for "
                f"B={input_ids.shape[0]}, T={input_ids.shape[1]}, "
                f"has_page_table={page_table is not None}: {exc}. "
                "Falling back to eager forward (see generator.py module docstring)."
            )
            return _NO_TRACE_FALLBACK

    def _try_capture_decode(
        self,
        input_ids,
        current_pos,
        page_table,
        kv_caches,
        dn_states,
        conv_states,
    ):
        """Attempt decode trace capture.  Returns trace dict or sentinel."""
        try:
            return self._capture_decode_trace(
                input_ids,
                current_pos=current_pos,
                page_table=page_table,
                kv_caches=kv_caches,
                dn_states=dn_states,
                conv_states=conv_states,
            )
        except Exception as exc:
            logger.warning(
                f"[Qwen36Generator] Decode trace capture failed for "
                f"B={input_ids.shape[0]}, has_page_table={page_table is not None}: "
                f"{exc}. Falling back to eager forward."
            )
            return _NO_TRACE_FALLBACK

    def _capture_prefill_trace(self, input_ids, page_table):
        """Compile + capture trace for prefill.  Currently fails at capture
        because of host-writes inside ``forward_prefill`` (see module docstring).
        Kept as the future entry point — will succeed once the forward path
        is refactored to be host-write-free.

        ``forward_prefill`` is called with ``output_as_ttnn=True`` so the LM-head
        ``ttnn.to_torch`` gather is OUTSIDE the captured trace.  The on-device
        logits tensor is returned in ``captured`` and the execute path gathers
        it to CPU AFTER ``ttnn.execute_trace`` returns.
        """
        # Compile run.  Run as output_as_ttnn=True so the compile run matches
        # the captured op sequence (no LM-head host gather).
        _ = self.model.forward_prefill(input_ids, return_caches=True, page_table=page_table, output_as_ttnn=True)
        ttnn.synchronize_device(self.mesh_device)
        # Capture run.
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        captured = self.model.forward_prefill(input_ids, return_caches=True, page_table=page_table, output_as_ttnn=True)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        return {"trace_id": trace_id, "captured": captured}

    def _capture_decode_trace(
        self,
        input_ids,
        current_pos,
        page_table,
        kv_caches,
        dn_states,
        conv_states,
    ):
        """Compile + capture trace for one decode step.

        ``forward_decode`` is called with ``output_as_ttnn=True`` so the LM-head
        ``ttnn.to_torch`` gather happens OUTSIDE the captured trace, in the
        execute path after ``ttnn.execute_trace`` returns.
        """
        _ = self.model.forward_decode(
            input_ids,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
            output_as_ttnn=True,
        )
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        captured = self.model.forward_decode(
            input_ids,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
            output_as_ttnn=True,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        return {"trace_id": trace_id, "captured": captured}
