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
    # T14b.9 added the Generator/model split required for trace:
    #   - model.prepare_decode_inputs_host / prepare_inputs_decode
    #     hoist all host->device copies out of model.forward_*
    #   - model.ttnn_decode_forward takes pre-uploaded device tensors and
    #     is the body recorded into the trace
    #   - Generator._capture_decode_trace runs the compile pass, calls
    #     prepare_inputs_decode OUTSIDE begin_trace_capture, then captures
    #     ttnn_decode_forward against those tensor addresses
    #   - decode_forward_with_caches per-step replay uses
    #     prepare_decode_inputs_host + copy_host_to_device_tensor against
    #     the captured device tensors, then execute_trace
    #
    # Trace requires the paged decode path — page_table=None falls back to
    # eager because the non-paged KV cache slice uses cur_pos as a Python
    # literal that would bake into the trace.
    #
    # _TRACE_SUPPORTED stays False until the residual "Writes are not
    # supported during trace capture" warnings emitted inside
    # ttnn_decode_forward are eliminated. The remaining host-write site
    # is suspected to live in llama_attention.forward_decode's paged
    # branch (the cur_pos_tensor / page_table interaction with
    # paged_update_cache and paged_scaled_dot_product_attention_decode) or
    # in DeltaNet's recurrent_gated_delta_rule_ttnn — see
    # tests/test_paged_decode_trace.py for the diagnostic test that
    # surfaces the warnings.
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

        T14b.9 trace path (only when ``page_table is not None``):
          - First call per ``(B, has_pt)`` key compiles + captures.
          - Subsequent calls refresh the captured device input tensors via
            ``ttnn.copy_host_to_device_tensor`` and call ``ttnn.execute_trace``.
            The captured forward body is pure device — no host writes inside.
          - Non-paged decode (page_table=None) is currently NOT trace-safe
            because ``llama_attention.forward_decode``'s non-paged branch
            slices the KV cache using ``cur_pos`` as a Python int (the slice
            extents would bake into the trace). For page_table=None we fall
            back to eager regardless of ``enable_trace``.
        """
        B, T = input_ids.shape
        assert T == 1, f"decode_forward_with_caches expects T=1, got T={T}"

        has_pt = page_table is not None
        use_trace = enable_trace and self._TRACE_SUPPORTED and has_pt

        if not use_trace:
            # Eager fallback — handles both enable_trace=False and the
            # non-paged path (see docstring).
            return self.model.forward_decode(
                input_ids,
                current_pos=current_pos,
                kv_caches=kv_caches,
                dn_states=dn_states,
                conv_states=conv_states,
            )

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
                # The capture path already executed once with these exact
                # inputs (the compile run + the trace run). Return that
                # result directly without an extra refresh+replay.
                logits_tt, kv_c, dn_c, cv_c = cache["captured"]
                logits_cpu = self._gather_logits_to_cpu(logits_tt)
                return logits_cpu, kv_c, dn_c, cv_c

        if cache == _NO_TRACE_FALLBACK or not isinstance(cache, dict):
            return self.model.forward_decode(
                input_ids,
                current_pos=current_pos,
                kv_caches=kv_caches,
                dn_states=dn_states,
                conv_states=conv_states,
            )

        # Per-step trace replay: refresh captured device input tensors
        # OUTSIDE the trace boundary, then execute_trace. The page_table is
        # passed through unchanged (already on device; doesn't change per
        # decode step), so we only copy the three refreshable inputs.
        host_inputs = self.model.prepare_decode_inputs_host(input_ids, current_pos, page_table)
        captured_device_inputs = cache["device_inputs"]
        for key_name in ("tokens", "current_pos", "rope_idxs"):
            host_tensor = host_inputs.get(key_name)
            device_tensor = captured_device_inputs.get(key_name)
            if host_tensor is None or device_tensor is None:
                continue
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

        ttnn.execute_trace(self.mesh_device, cache["trace_id"], cq_id=0, blocking=True)
        logits_tt, kv_c, dn_c, cv_c = cache["captured"]
        logits_cpu = self._gather_logits_to_cpu(logits_tt)
        return logits_cpu, kv_c, dn_c, cv_c

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
        """Compile + capture trace for one decode step (T14b.9).

        Mirrors ``llama3_70b_galaxy/tt/generator.py::_capture_trace_text``:
          1. Compile run — eager ``forward_decode`` to populate the program
             cache. Discarded (its outputs are not used for replay).
          2. ``prepare_inputs_decode(...)`` — creates fresh DEVICE tensors
             at concrete addresses; these are the addresses the trace will
             record reads against. Returns the dict before ``begin_trace_capture``.
          3. ``begin_trace_capture`` → ``ttnn_decode_forward(device tensors)``
             → ``end_trace_capture``. The body is pure device — no host writes.
          4. Returns a cache dict carrying ``trace_id``, ``device_inputs``
             (the per-key handles to refresh between replays), and the
             ``captured`` output ``(logits_tt, kv_caches, dn_states,
             conv_states)`` so the replay path can gather logits afterwards.

        Requires ``page_table is not None`` — the non-paged decode branch in
        ``llama_attention.forward_decode`` slices the KV cache using
        ``cur_pos`` as a Python literal, which bakes the slice extents into
        the captured trace.
        """
        assert page_table is not None, (
            "Decode trace capture requires page_table — the non-paged path "
            "slices the KV cache using a Python int cur_pos, which is not "
            "trace-safe. Use the paged attention path for trace capture."
        )

        # 1. Compile run: populates the program cache for every op the
        #    captured forward will issue. Uses the eager forward_decode so we
        #    don't have to wire up fresh device tensors before the cache is
        #    warm. The result is discarded.
        _ = self.model.forward_decode(
            input_ids,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
            output_as_ttnn=True,
        )
        ttnn.synchronize_device(self.mesh_device)

        # 2. Build the persistent device input tensors OUTSIDE the trace.
        device_inputs = self.model.prepare_inputs_decode(input_ids, current_pos, page_table=page_table)

        # 3. Capture the pure-device forward against those tensor addresses.
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        captured = self.model.ttnn_decode_forward(
            device_inputs["tokens"],
            device_inputs["current_pos"],
            device_inputs["rope_idxs"],
            device_inputs["page_table"],
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

        return {
            "trace_id": trace_id,
            "device_inputs": device_inputs,
            "captured": captured,
        }
