# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reusable LoRA mixin for the Linear family.

LoRA-aware Linears keep a *host bank* of registered adapters and apply
the active adapter via one of two execution paths, chosen at construction
with ``lora_mode``:

    fuse     — merge the delta into the base weight at bind time:
                   W += scale * A.T @ B.T   on bind
                   W -= scale * A.T @ B.T   on unbind
               Forward runs the un-modified base path with every fused
               kernel intact (matmul+gelu, matmul+addcmul, etc.) and
               forward-time LoRA overhead is zero. Swap cost is bounded
               to ~one small matmul + an in-place add per Linear.

    runtime  — keep A, B on device and add their effect to every forward:
                   y = base(x) + scale * (x @ A) @ B
               Swap cost is microseconds (a host-side index assignment +
               re-uploading A/B). Per-forward cost is two extra small
               matmuls. Useful when the per-forward overhead is cheaper
               than the bind-time matmul (very tall stacks of cheap adapters,
               or quick A/B experiments).

Both A and B are uploaded with sharding inferred from W's own ``mesh_axes``
so the matmul lands the right shard:
  - A is [in, rank] → shard A's dim-0 the same way W's dim-0 is sharded.
  - B is [rank, out] → shard B's dim-1 the same way W's dim-1 is sharded.

Runtime mode supports replicated outputs only. Chunked-output Linears
(``chunks`` set on ColParallel) must use fuse mode — splitting the LoRA
delta across the same chunk boundaries is not implemented and would
defeat the runtime path's main advantage (zero CCL).

Runtime mode is also incompatible with the fused-addcmul paths
(``RowParallelLinear.forward_fused_addcmul``,
``WanAttention._to_out_fused_addcmul``): those kernels combine matmul,
reduce-scatter, and addcmul in a single op and read ``weight.data``
directly, so the runtime A/B tensors have nowhere to participate. Fuse
mode works on these paths automatically because the delta lives in
``weight.data``.

TODO: ``lora_enabled`` is currently threaded through 5 constructors
(``WanPipeline`` → ``WanTransformer3DModel`` → ``WanTransformerBlock`` →
``WanAttention`` / ``ParallelFeedForward``) to pick between the LoRA
class and its base. The LoRA classes could instead be drop-in
replacements with a zero-overhead empty-bank fast path in
``LoRAMixin.forward``, removing the flag entirely. Deferred to a
follow-up — needs perf validation that the empty-bank forward is
truly identical to the base.

Note: bind/unbind in fuse mode adds and subtracts the delta in bf16,
which is not bit-exactly inverse. W drifts by ~O(N · eps_bf16) over N
swap cycles. In production this is bounded automatically by
``dynamic_load=True`` (``cache.load_model`` restores the cached base
on every page-in). For static pipelines (``dynamic_load=False``) that
need bit-exact restore across many swaps, snapshot the state dict once
at construction and call ``load_torch_state_dict`` to re-seed W when
drift becomes material. (With the A/B cache enabled — ``cache_capacity>0``
— unbind subtracts the *same* device delta that bind added for a cached
adapter, so those swap cycles are exact and do not drift.)

Fuse-mode A/B cache (``cache_capacity``)
----------------------------------------
Each bind/unbind/reload otherwise re-uploads A and B from host and
re-lays them out for W's sharding. With ``cache_capacity>0`` the uploaded
device factors are kept in a per-module LRU (keyed by bank index), so a
re-bind of an already-seen adapter — and, under ``dynamic_load``, the
re-merge after every page-in — skips the host upload. The cache holds
only the rank-sized A/B (never the full delta), survives page-out (the
factors are plain tensors, not tracked ``Parameters``, so
``deallocate_weights`` leaves them alone), and is bounded to
``cache_capacity`` adapters per Linear. ``cache_capacity=0`` disables it
and restores the upload-then-free behavior.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

import ttnn

from ..utils.tensor import from_torch as _from_torch


@dataclass
class LoRAAdapter:
    """Host-resident LoRA adapter record. ``A`` and ``B`` are torch tensors
    in PyTorch LoRA layout (A: [rank, in], B: [out, rank])."""

    name: str
    A: torch.Tensor
    B: torch.Tensor
    rank: int
    scale: float


class LoRAMixin:
    """Adds a LoRA bank + active-adapter state to a Linear-family Module.

    Subclasses call ``_init_lora_state(mode=...)`` at the end of their
    ``__init__``. The mixin handles bind/unbind and (for ``runtime`` mode)
    overrides ``forward`` to add the delta.
    """

    # Layout-neutral (adds no __dict__/__weakref__ of its own) so a plain
    # Linear instance can be promoted in place via ``__class__`` assignment
    # (see experimental/lora/promote.py); state still lives in Module's __dict__.
    __slots__ = ()

    LORA_MODES = ("fuse", "runtime")

    # ---- one-time state init ----
    def _init_lora_state(self, mode: str = "fuse", cache_capacity: int = 0) -> None:
        if mode not in self.LORA_MODES:
            raise ValueError(f"lora_mode must be one of {self.LORA_MODES}; got {mode!r}")
        self.lora_mode = mode
        self.lora_bank: list[LoRAAdapter | None] = []
        self.active_idx: int | None = None
        self.active_scale: float = 1.0
        # Fuse-mode LRU of device-resident A/B factors, keyed by bank index:
        # {idx: (A_dev, B_dev, scale)}, MRU last. Bounded to ``lora_cache_capacity``
        # adapters; 0 disables caching (upload-then-free per bind).
        self.lora_cache_capacity: int = int(cache_capacity)
        self._ab_cache: OrderedDict[int, tuple[ttnn.Tensor, ttnn.Tensor, float]] = OrderedDict()
        # ``active_idx`` records intent (which adapter should be used).
        # ``_delta_applied`` (fuse mode) records whether the merge is
        # currently present on the on-device weight. These can diverge
        # under dynamic_load: when the transformer is paged out the
        # device weight is gone; when it pages back in the cached base
        # weight is restored and the delta must be re-applied.
        self._delta_applied: bool = False
        # Device-resident A/B for runtime mode (None when no adapter
        # is bound or when the layer is paged out).
        self._runtime_A: ttnn.Tensor | None = None
        self._runtime_B: ttnn.Tensor | None = None

    # ---- introspection ----
    @property
    def is_lora_active(self) -> bool:
        """True if any LoRA is currently bound (whether or not the on-device
        side state is currently realized — e.g. paged out under dynamic_load)."""
        return self.active_idx is not None

    # ---- bank management ----
    def register_lora(
        self,
        A_torch: torch.Tensor,
        B_torch: torch.Tensor,
        scale: float = 1.0,
        name: str = "",
    ) -> int:
        # swiglu doubles out_features AND head-permutes W in _prepare_torch_state;
        # the LoRA delta would not undergo the same permutation, so the merged
        # weight's gate/value split would be wrong. Refuse rather than silently
        # corrupt — and avoid the confusing 2*out validation error.
        if getattr(self, "fused_activation_fn", None) is None and getattr(self, "activation_fn", None) == "swiglu":
            raise ValueError(
                "LoRA adapters are not supported on swiglu layers; the merged "
                "delta would not match the swiglu head-permuted weight layout"
            )

        rank = A_torch.shape[0]
        if A_torch.shape != (rank, self.in_features):
            raise ValueError(
                f"A must be [rank, in_features]; got {tuple(A_torch.shape)} " f"expected (*, {self.in_features})"
            )
        if B_torch.shape != (self.out_features, rank):
            raise ValueError(
                f"B must be [out_features, rank]; got {tuple(B_torch.shape)} " f"expected ({self.out_features}, {rank})"
            )

        adapter = LoRAAdapter(
            name=name,
            A=A_torch.detach().clone(),
            B=B_torch.detach().clone(),
            rank=rank,
            scale=float(scale),
        )
        self.lora_bank.append(adapter)
        return len(self.lora_bank) - 1

    def unregister_lora(self, idx: int) -> None:
        if not (0 <= idx < len(self.lora_bank)):
            return
        if self.active_idx == idx:
            self.unbind_active()
        self._drop_cached_ab(idx)
        self.lora_bank[idx] = None

    def set_lora_cache_capacity(self, n: int) -> None:
        """Set how many adapters keep their device A/B factors resident (LRU).
        Shrinking evicts the least-recently-used down to ``n``; ``n<=0`` frees
        the whole cache and disables further caching."""
        self.lora_cache_capacity = int(n)
        if self.lora_cache_capacity <= 0:
            self._free_ab_cache()
        else:
            self._evict_ab_cache()

    # ---- bind / unbind ----
    def bind_active(self, idx: int, scale: float | None = None) -> None:
        """Make bank[idx] the active LoRA.

        If a different adapter is currently bound, it is removed first.
        When the layer's weights are not currently on device (dynamic_load
        page-out), the merge is deferred — ``active_idx`` and ``active_scale``
        record the intent and ``reapply_after_load`` will pick it up on
        the next load.
        """
        # Validate the target slot BEFORE touching the current active adapter,
        # so a bad idx doesn't leave W with the prior delta half-undone.
        if not (0 <= idx < len(self.lora_bank)) or self.lora_bank[idx] is None:
            raise IndexError(f"invalid lora slot {idx}")

        adapter = self.lora_bank[idx]
        use_scale = adapter.scale if scale is None else float(scale)

        # Idempotent: don't subtract-and-re-add the same delta — that round-trip
        # accumulates bf16 quantization drift in W.
        if self.active_idx == idx and self.active_scale == use_scale:
            return

        if self.active_idx is not None:
            self._undo_active()

        self.active_idx = idx
        self.active_scale = use_scale
        if self.is_loaded():
            self._realize_active()

    def unbind_active(self) -> None:
        if self.active_idx is None:
            return
        self._undo_active()

    def _realize_active(self) -> None:
        """Materialize the on-device effect of the active adapter.

        fuse → add the delta into ``W``. runtime → upload ``A`` and ``B``
        for the forward path to consume.
        """
        if self.lora_mode == "fuse":
            self._apply_delta(self.active_idx, self.active_scale, sign=+1)
            self._delta_applied = True
        else:  # runtime
            adapter = self.lora_bank[self.active_idx]
            self._upload_runtime_ab(adapter.A, adapter.B)

    def _undo_active(self) -> None:
        if self.is_loaded():
            if self.lora_mode == "fuse" and self._delta_applied:
                if self.lora_bank[self.active_idx] is not None:
                    self._apply_delta(self.active_idx, self.active_scale, sign=-1)
            elif self.lora_mode == "runtime":
                self._free_runtime_ab()
        self._delta_applied = False
        self.active_idx = None
        self.active_scale = 1.0

    def reapply_after_load(self) -> None:
        """Re-materialize the active adapter after a dynamic_load reload.

        Called by the pipeline after a reload restores the cached base
        weights and (for runtime mode) wipes the device-resident A/B.
        No-op when no adapter is active or the effect is already in place.
        """
        if self.active_idx is None or not self.is_loaded():
            return
        if self.lora_mode == "fuse" and self._delta_applied:
            return
        if self.lora_mode == "runtime" and self._runtime_A is not None:
            return
        self._realize_active()

    def deallocate_weights(self) -> None:  # type: ignore[override]
        # Weights about to be freed → any merged delta goes with them
        # and runtime A/B are tied to the layer being resident.
        self._delta_applied = False
        if self._runtime_A is not None or self._runtime_B is not None:
            self._free_runtime_ab()
        super().deallocate_weights()

    def deallocate_lora(self) -> None:
        if self.active_idx is not None:
            self.unbind_active()
        self._free_ab_cache()
        self.lora_bank = []

    # ---- forward override (runtime mode) ----
    def forward(self, x: ttnn.Tensor, *args, **kwargs):  # type: ignore[override]
        if self.lora_mode != "runtime" or self._runtime_A is None:
            return super().forward(x, *args, **kwargs)

        # Reference math: y = activation(linear(x, W, b) + delta). The base
        # Linear may fuse the activation into the matmul; if so, disable it
        # for this call so we can add the delta BEFORE the activation runs.
        saved_fused = getattr(self, "fused_activation_fn", None)
        saved_activation = getattr(self, "activation_fn", None)
        try:
            if saved_fused is not None:
                self.fused_activation_fn = None
            if saved_activation is not None:
                self.activation_fn = None
            base = super().forward(x, *args, **kwargs)
        finally:
            self.fused_activation_fn = saved_fused
            self.activation_fn = saved_activation

        if isinstance(base, list):
            raise RuntimeError(
                "runtime LoRA delta does not support chunked-output Linears; "
                "construct with lora_mode='fuse' for chunks>1"
            )
        # Honor a caller-passed compute_kernel_config override (the base
        # Linear path does), so runtime LoRA and the base matmul use the
        # same precision settings within one forward.
        delta_compute_config = kwargs.get("compute_kernel_config", self.compute_config)
        ax = ttnn.matmul(x, self._runtime_A, compute_kernel_config=delta_compute_config)
        delta = ttnn.matmul(ax, self._runtime_B, compute_kernel_config=delta_compute_config)
        ttnn.deallocate(ax)
        out = ttnn.add(base, delta)
        ttnn.deallocate(base)
        ttnn.deallocate(delta)

        if saved_fused is not None:
            # Only gelu is currently hoisted into fused_activation_fn.
            if saved_fused[0] == ttnn.UnaryOpType.GELU:
                out = ttnn.gelu(out)
            else:
                raise NotImplementedError(f"runtime LoRA cannot replay fused activation {saved_fused}")
        if saved_activation is not None:
            from .linear import _apply_activation_fn

            out = _apply_activation_fn(out, saved_activation)
        return out

    def forward_fused_addcmul(self, *args, **kwargs):  # type: ignore[override]
        # The fused matmul + reduce-scatter + addcmul kernel reads weight.data
        # directly and has nowhere to add a separate LoRA delta — runtime mode
        # would silently no-op. Fuse mode is fine because the delta is already
        # merged into weight.data.
        if self.lora_mode == "runtime" and self.is_lora_active:
            raise RuntimeError(
                "runtime LoRA mode is incompatible with forward_fused_addcmul; "
                "construct this layer with lora_mode='fuse'"
            )
        return super().forward_fused_addcmul(*args, **kwargs)

    # ---- internals ----
    def _apply_delta(self, idx: int, scale: float, sign: int) -> None:
        """Add (``sign>0``) or subtract (``sign<0``) bank[idx]'s ``+scale`` delta
        into ``self.weight.data`` in place (fuse mode). The device A/B factors
        come from the LRU cache when enabled, so a re-bind or a post-reload
        re-merge skips the host upload; the same cached ``+scale`` delta serves
        both bind and unbind, making the pair an exact negation. The full-size
        delta itself is never cached (that would cost a whole weight per adapter)."""
        A_dev, B_dev, owned = self._acquire_delta_ab(idx, float(scale))
        delta = ttnn.matmul(A_dev, B_dev, compute_kernel_config=self.compute_config)
        if owned:
            ttnn.deallocate(A_dev)
            ttnn.deallocate(B_dev)
        if sign > 0:
            ttnn.add(self.weight.data, delta, output_tensor=self.weight.data)
        else:
            ttnn.subtract(self.weight.data, delta, output_tensor=self.weight.data)
        ttnn.deallocate(delta)

    def _acquire_delta_ab(self, idx: int, scale: float) -> tuple[ttnn.Tensor, ttnn.Tensor, bool]:
        """Return ``(A_dev, B_dev, owned)`` for bank[idx]'s ``+scale`` delta
        factors, uploading + laying them out for W's sharding on a miss.
        ``owned=True`` means the factors are uncached and the caller must
        deallocate them; cached factors are owned by the LRU (``owned=False``)."""
        adapter = self.lora_bank[idx]
        if self.lora_cache_capacity <= 0:
            A_dev, B_dev = self._upload_ab_for_w_sharding(adapter.A, adapter.B, scale=scale)
            return A_dev, B_dev, True
        cached = self._ab_cache.get(idx)
        if cached is not None and cached[2] == scale:
            self._ab_cache.move_to_end(idx)  # mark most-recently-used
            return cached[0], cached[1], False
        if cached is not None:  # same slot bound at a new scale → rebuild
            self._drop_cached_ab(idx)
        A_dev, B_dev = self._upload_ab_for_w_sharding(adapter.A, adapter.B, scale=scale)
        self._ab_cache[idx] = (A_dev, B_dev, scale)  # inserted at the MRU end
        self._evict_ab_cache()
        return A_dev, B_dev, False

    def _evict_ab_cache(self) -> None:
        # Trim from the LRU end. The active adapter is always the MRU (just
        # inserted or touched), so for capacity >= 1 it is never evicted.
        while len(self._ab_cache) > self.lora_cache_capacity:
            _, (A_old, B_old, _) = self._ab_cache.popitem(last=False)
            ttnn.deallocate(A_old)
            ttnn.deallocate(B_old)

    def _drop_cached_ab(self, idx: int) -> None:
        entry = self._ab_cache.pop(idx, None)
        if entry is not None:
            ttnn.deallocate(entry[0])
            ttnn.deallocate(entry[1])

    def _free_ab_cache(self) -> None:
        for A_dev, B_dev, _ in self._ab_cache.values():
            ttnn.deallocate(A_dev)
            ttnn.deallocate(B_dev)
        self._ab_cache.clear()

    def _upload_runtime_ab(self, A_torch: torch.Tensor, B_torch: torch.Tensor) -> None:
        """Upload A and B for the runtime forward path. The active scale
        is baked into B so forward stays as two pure matmuls + one add."""
        self._runtime_A, self._runtime_B = self._upload_ab_for_w_sharding(A_torch, B_torch, scale=self.active_scale)

    def _free_runtime_ab(self) -> None:
        if self._runtime_A is not None:
            ttnn.deallocate(self._runtime_A)
            self._runtime_A = None
        if self._runtime_B is not None:
            ttnn.deallocate(self._runtime_B)
            self._runtime_B = None

    def _upload_ab_for_w_sharding(
        self,
        A_torch: torch.Tensor,
        B_torch: torch.Tensor,
        *,
        scale: float,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Upload ``A`` ([rank,in] → [in,rank]) and ``scale*B`` ([out,rank] → [rank,out])
        with the sharding of ``W``:
          - A's dim 0 sharded like W's dim 0
          - B's dim 1 sharded like W's dim 1
        So that matmul(A, B) yields a tensor with W's combined sharding."""
        w_axes = self.weight.mesh_axes
        # Always take the multiply path so bind (+scale) and unbind (-scale)
        # exercise the same code with the same fp32→bf16 rounding boundaries;
        # the on-device deltas then cleanly negate each other.
        B_eff = B_torch * scale
        A_dev = _from_torch(
            A_torch.transpose(0, 1).contiguous().to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.bfloat16,
            mesh_axes=(w_axes[0], None),
        )
        B_dev = _from_torch(
            B_eff.transpose(0, 1).contiguous().to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.bfloat16,
            mesh_axes=(None, w_axes[1]),
        )
        return A_dev, B_dev
