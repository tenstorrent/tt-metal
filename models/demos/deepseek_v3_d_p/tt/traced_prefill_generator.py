# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fast token generation on the prefill-only stack, by trace replay.

This folder is disaggregated *prefill* and has no decode path, so a demo that wants text generates
by re-running a whole prefill per token. Done naively that is dominated by HOST DISPATCH, not by the
device: measured on Mistral Small 4 (36 layers, 32 chips, window 512), one forward is **2316 op
dispatches at ~0.55 ms each against ~80 ms of device kernel time** -- the mesh sits idle ~95% of the
wall clock. Capturing the block stack in a ttnn trace collapses those 2316 dispatches into a replay:

    eager   1715 ms/token          traced   80 ms/token      21x, same sampled token

This class packages that so other models in this stack can reuse it rather than re-deriving it.
It is deliberately model-agnostic: it takes an already-built ``TtPrefillTransformer`` and the caches
``run_model`` produced, and knows nothing about any particular variant.

HOW IT WORKS, AND WHY EACH PIECE IS NECESSARY
---------------------------------------------
* **Segmented capture.** The MoE swaps sub-device managers mid-forward, which a single flat capture
  cannot survive. ``SubDeviceTraceController`` splits the capture at those boundaries (73 segments
  for a 36-layer model) and replays them in order. ``TtPrefillTransformer.set_trace_controller``
  already wires it to every layer, so nothing new is needed.
* **Constant ``actual_isl``.** A trace records a fixed op sequence, but ``actual_isl`` normally grows
  by one per generated token and selects a different memoized MoE padding config each step. Holding
  it constant -- at ``isl_total``, the whole window marked real -- keeps the captured sequence and
  that config invariant, so ONE capture serves every request regardless of prompt length.
  Safe by causality: the LM head reads row ``n-1``, and no row attends to positions after it, so
  whatever occupies later positions (pad, stale KV, not-yet-generated slots) cannot change those
  logits. ``TtPrefillRuntime`` captures with ``actual_isl=chunk_size`` for exactly this reason.
* **Eager tail.** ``norm -> lm_head -> logit_to_host -> sample`` ends in a blocking device->host read,
  which cannot live inside a trace. It is excluded with ``stop_after_blocks=True`` (~36 of 2316 ops,
  so ~98% of dispatch is still captured) and run eagerly on the replay's output -- which is also what
  lets each step select the CORRECT row ``n-1`` while the captured stack stays fixed.
* **Fixed input address.** The capture records the address of the token tensor, so each step must
  write into THAT tensor (``copy_host_to_device_tensor``). A fresh ``from_torch`` would allocate
  elsewhere and the replay would keep reading stale tokens -- silently, with plausible output.

APPLICABILITY
-------------
* **Dense-MLA variants only.** ``set_trace_controller`` rejects sparse/DSA (indexer) models --
  GLM-5.1/5.2 -- because the captured forward never threads ``index_kv_cache`` and would replay
  without its indexer cache, producing wrong KV rather than failing. deepseek_v3, kimi_k2_6,
  kimi_k2_7 and mistral_small4 are eligible.
* **Right padding only**, since the LM head must read row ``actual_isl-1``.
* Requires ``TtPrefillTransformer`` built WITH its sampling tail (``kv_only_last_layer=False``).

ALWAYS ``release()``
--------------------
Leaving trace buffers or MoE-created sub-device managers registered makes ``close_mesh_device``
**segfault** during teardown. A failed capture once stranded a 32-chip galaxy for ~3 hours. Call
``release()`` from a ``finally``, and run harnesses under ``timeout``.
"""

import torch
from loguru import logger

import ttnn


class TracedPrefillGenerator:
    """Generate tokens from a prefill-only transformer via trace replay.

    Usage::

        gen = TracedPrefillGenerator(transformer=..., mesh_device=..., kvpe_cache=..., ...)
        try:
            gen.capture()                       # once, at startup
            tok = gen.forward_token(window, n, temperature)   # per generated token
        finally:
            gen.release()
    """

    def __init__(
        self,
        *,
        transformer,
        mesh_device,
        kvpe_cache,
        index_kv_cache=None,
        isl_total: int,
        sp_factor: int,
        isl_per_chip: int,
        pad_id: int = 0,
        chunk_order=None,
        padding_side: str = "right",
    ):
        assert padding_side == "right", (
            f"traced generation requires right padding (the LM head reads row actual_isl-1); "
            f"got {padding_side!r}"
        )
        self.transformer = transformer
        self.mesh_device = mesh_device
        self.kvpe_cache = kvpe_cache
        self.index_kv_cache = index_kv_cache
        self.isl_total = isl_total
        self.sp_factor = sp_factor
        self.isl_per_chip = isl_per_chip
        self.pad_id = pad_id
        self.chunk_order = chunk_order

        self._controller = None
        self._trace_input = None
        self._trace_hidden = None

    # ------------------------------------------------------------------ helpers

    def _shard(self, window: torch.Tensor, to_device: bool):
        """Shard the [1, isl_total] host window across the mesh, matching run_model's layout."""
        ids = window
        if self.chunk_order is not None:
            from models.demos.deepseek_v3_d_p.tt.mla.utils import reorder_tensor_chunks

            ids = reorder_tensor_chunks(ids.unsqueeze(1).unsqueeze(-1), self.chunk_order, seq_dim=2)
            ids = ids.squeeze(1).squeeze(-1)
        kwargs = dict(
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=(0, None)
            ),
        )
        if to_device:
            kwargs.update(device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.from_torch(ids.reshape(self.sp_factor, 1, self.isl_per_chip), **kwargs)

    def _blocks(self, tokens):
        return self.transformer(
            tokens,
            self.kvpe_cache,
            actual_isl=self.isl_total,  # CONSTANT -- see module docstring
            return_intermediates=False,
            read_profiler=False,
            temperature=0.0,
            index_kv_cache=self.index_kv_cache,
            stop_after_blocks=True,
        )

    # ------------------------------------------------------------------ lifecycle

    @property
    def is_traced(self) -> bool:
        return self._controller is not None

    def capture(self):
        """Warm-compile, then capture the block stack. Idempotent."""
        if self._controller is not None:
            return
        from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

        self._trace_input = self._shard(
            torch.full((1, self.isl_total), self.pad_id, dtype=torch.int64), to_device=True
        )
        # Compile before capturing: a capture records dispatch, not compilation, so an uncompiled
        # program would otherwise be compiled inside the capture.
        self._blocks(self._trace_input)
        ttnn.synchronize_device(self.mesh_device)

        controller = SubDeviceTraceController(self.mesh_device)
        self.transformer.set_trace_controller(controller)
        try:
            controller.begin_capture()
            self._trace_hidden = self._blocks(self._trace_input)
            controller.end_capture()
        except Exception:
            self.transformer.set_trace_controller(None)
            self.transformer.release_sub_device_managers()
            raise
        self._controller = controller
        logger.success(
            f"traced prefill: captured {controller.num_segments} segments "
            f"({controller.trace_bytes() / (1024 * 1024):.1f} MB); expect ~20x faster tokens"
        )

    def release(self):
        """Release traces and sub-device managers. Safe to call repeatedly; call from a finally."""
        if self._controller is None:
            return
        try:
            self._controller.release()
        finally:
            self._controller = None
            self._trace_hidden = None
            self.transformer.set_trace_controller(None)
            self.transformer.release_sub_device_managers()

    # ------------------------------------------------------------------ generation

    def forward_token(self, window: torch.Tensor, n: int, temperature: float = 0.0) -> int:
        """Next token for a window holding ``n`` real tokens. Traced when captured, else eager."""
        if self._controller is None:
            tokens = self._shard(window, to_device=True)
            token_id, _prob, _ = self.transformer(
                tokens,
                self.kvpe_cache,
                actual_isl=n,
                return_intermediates=False,
                read_profiler=False,
                temperature=temperature,
                index_kv_cache=self.index_kv_cache,
            )
            ttnn.synchronize_device(self.mesh_device)
            ttnn.deallocate(tokens)
            return int(token_id)

        # Into the tensor the capture recorded — a fresh allocation would be read as stale tokens.
        ttnn.copy_host_to_device_tensor(self._shard(window, to_device=False), self._trace_input)
        self._controller.replay()
        # `n` (not isl_total) picks the row, so one capture serves every step of every request.
        h = self.transformer.norm(self._trace_hidden)
        _logits, first_token_logits = self.transformer._lm_head_and_extract(h, n)
        token_id, _prob, _sweep = self.transformer._sample(first_token_logits, n, temperature)
        return int(token_id)
