# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 pipeline with runtime-swappable LoRA adapters.

Constructs the transformer with ``lora_enabled=True`` so every attention
and FFN Linear is a LoRA-aware variant. Exposes a small API for
registering and switching adapters at runtime:

  pipeline.register_lora(name, high_path, low_path, scale)  → AdapterHandle
  pipeline.set_active_lora(handle | None)
  pipeline.unregister_lora(handle)
  pipeline.list_loras()

Adapter swap merges the active LoRA delta directly into the base weight:
``W += scale * A.T @ B.T`` per LoRA Linear, on device, with W's sharding.
Forward-time LoRA overhead is zero — the captured trace runs the
un-modified base path. Swap cost is bounded to ~one small matmul + an
in-place add per LoRA Linear; sub-second end to end. See
``models/tt_dit/layers/lora.py`` for the merge details.

Out of scope for v0:
  - Stacking multiple adapters into a single bound slot (callers wanting
    stacked LoRAs should pre-merge offline, or use the older CPU-fusion
    pipeline at ``pipeline_wan_lora.py``).
  - ``.diff`` / ``.diff_b`` direct deltas (the loader warns and skips).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from models.tt_dit.experimental.lora.adapter_loader import AdapterHandle, load_adapter_into
from models.tt_dit.layers.lora import LoRAMixin
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V


@dataclass
class _RegisteredLoRA:
    """Per-pipeline record of a registered adapter pair."""

    name: str
    handle_high: AdapterHandle | None
    handle_low: AdapterHandle | None


@dataclass
class CombinedAdapterHandle:
    """Returned by `register_lora`. Knows about both experts."""

    name: str
    high: AdapterHandle | None
    low: AdapterHandle | None
    rank: int = field(init=False)

    def __post_init__(self):
        ranks = [h.rank for h in (self.high, self.low) if h is not None]
        if not ranks:
            raise ValueError(f"adapter '{self.name}' loaded for neither expert")
        self.rank = max(ranks)


def _iter_lora_modules(module):
    """Yield every LoRAMixin-bearing submodule of `module`, recursively."""
    if isinstance(module, LoRAMixin):
        yield module
    for _, child in module.named_children():
        yield from _iter_lora_modules(child)


class _LoRAPipelineMixin:
    """Implements the LoRA swap API on top of a WanPipeline subclass.

    Assumes the host pipeline was constructed with ``lora_enabled=True``
    so both ``self.transformer`` and ``self.transformer_2`` have LoRA-aware
    Linear modules throughout.
    """

    def __init__(self, *args, lora_enabled: bool = True, **kwargs):
        # Initialize LoRA state *before* super().__init__() because the base
        # pipeline init calls self._prepare_transformer() during warmup, and
        # our override below reads self._active_adapter_name.
        self._lora_registry: dict[str, _RegisteredLoRA] = {}
        self._active_adapter_name: str | None = None
        super().__init__(*args, lora_enabled=lora_enabled, **kwargs)

        # Fuse-mode bind/unbind adds and subtracts the delta in bf16; under
        # dynamic_load=True the cache restore re-seeds the base every reload
        # and drift is bounded. With dynamic_load=False the base weight is
        # never refreshed, so W drifts by ~O(N · eps_bf16) over N swap cycles.
        # See models/tt_dit/layers/lora.py for the snapshot+reseed escape hatch.
        if not getattr(self, "dynamic_load", True):
            logger.warning(
                "LoRA fuse-mode pipeline constructed with dynamic_load=False — "
                "repeated bind/unbind cycles will accumulate bf16 quantization "
                "drift in the base weight. For long-lived sessions, snapshot "
                "the transformer state dict at construction and re-seed via "
                "load_torch_state_dict when drift becomes material."
            )

    def _prepare_transformer(self, idx: int):
        """Re-apply any active LoRA delta after the transformer is reloaded.

        Under dynamic_load the two transformers swap in and out of DRAM
        between uses; each reload restores the cached *base* weights, so
        the merged LoRA delta is wiped. We walk this transformer's LoRA
        modules and re-merge their active adapter (no-op for modules that
        already have the delta applied, e.g., the first load).
        """
        super()._prepare_transformer(idx)
        if self._active_adapter_name is None:
            return
        transformer = self.transformer_states[idx].model
        for mod in _iter_lora_modules(transformer):
            mod.reapply_after_load()

    # ---- public API ----
    def register_lora(
        self,
        name: str,
        *,
        high_path: str | Path | None = None,
        low_path: str | Path | None = None,
        scale: float = 1.0,
    ) -> CombinedAdapterHandle:
        """Load a LoRA adapter for one or both experts.

        Both ``high_path`` and ``low_path`` are independently optional — pass
        either or both. Adapter weights are parsed from disk and stored on the
        host LoRA bank of every applicable Linear in the respective transformer;
        device-side merge happens later in ``set_active_lora`` (fuse mode) or
        in the layer's forward (runtime mode). Device-OOM therefore surfaces
        at ``set_active_lora`` time, not here.
        """
        if not high_path and not low_path:
            raise ValueError("register_lora requires high_path and/or low_path")
        if name in self._lora_registry:
            raise ValueError(f"LoRA name '{name}' already registered")

        high_handle = None
        low_handle = None

        if high_path is not None:
            logger.info(f"loading LoRA '{name}' high-noise from {high_path} (scale={scale})")
            high_handle = load_adapter_into(self.transformer, str(high_path), scale=scale, name=name)
        if low_path is not None:
            logger.info(f"loading LoRA '{name}' low-noise from {low_path} (scale={scale})")
            low_handle = load_adapter_into(self.transformer_2, str(low_path), scale=scale, name=name)

        combined = CombinedAdapterHandle(name=name, high=high_handle, low=low_handle)
        self._lora_registry[name] = _RegisteredLoRA(name=name, handle_high=high_handle, handle_low=low_handle)
        return combined

    def unregister_lora(self, handle: CombinedAdapterHandle | str) -> None:
        name = handle if isinstance(handle, str) else handle.name
        rec = self._lora_registry.pop(name, None)
        if rec is None:
            return
        if self._active_adapter_name == name:
            self.set_active_lora(None)
        # Walk and unregister bank entries
        for transformer, handle_ in (
            (self.transformer, rec.handle_high),
            (self.transformer_2, rec.handle_low),
        ):
            if handle_ is None:
                continue
            self._unregister_handle(transformer, handle_)

    def set_active_lora(self, handle: CombinedAdapterHandle | str | None) -> None:
        """Bind the named adapter as the active LoRA for both experts.

        Each LoRA Linear's bind merges the new delta into the base weight
        in-place. If a different adapter is currently active, its delta is
        subtracted first. Passing ``None`` removes the active delta (W is
        restored to its base value).
        """
        if handle is None:
            self._unbind_all()
            self._active_adapter_name = None
            return

        name = handle if isinstance(handle, str) else handle.name
        rec = self._lora_registry.get(name)
        if rec is None:
            raise KeyError(f"no LoRA registered with name '{name}'")

        self._bind_per_module(self.transformer, rec.handle_high)
        self._bind_per_module(self.transformer_2, rec.handle_low)
        self._active_adapter_name = name

    def list_loras(self) -> list[str]:
        return list(self._lora_registry)

    def deallocate_all_loras(self) -> None:
        """Unbind every active adapter and clear the LoRA bank on every layer.

        Releases host-side adapter records across both transformers. Use this
        on a long-running server to reset state between unrelated jobs.
        """
        self._unbind_all()
        self._active_adapter_name = None
        self._lora_registry.clear()
        for transformer in (self.transformer, self.transformer_2):
            for mod in _iter_lora_modules(transformer):
                mod.deallocate_lora()

    # ---- internals ----
    def _path_to_module(self, transformer, dotted: str):
        """Resolve a dotted path like ``blocks.0.attn1.to_qkv`` to a child module."""
        cur = transformer
        for part in dotted.split("."):
            if part.isdigit():
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
        return cur

    def _bind_per_module(self, transformer, handle: AdapterHandle | None):
        addressed: set[int] = set()
        if handle is not None:
            for dotted, bank_idx in handle.target_indices.items():
                mod = self._path_to_module(transformer, dotted)
                mod.bind_active(bank_idx)
                addressed.add(id(mod))

        # Any LoRA module not touched by this adapter (including the all-orphan
        # case where handle is None, e.g. switching to an adapter that only
        # covers the other expert) but currently active from a previous bind
        # must have its delta removed.
        for mod in _iter_lora_modules(transformer):
            if id(mod) in addressed:
                continue
            if mod.is_lora_active:
                mod.unbind_active()

    def _unbind_all(self):
        for transformer in (self.transformer, self.transformer_2):
            for mod in _iter_lora_modules(transformer):
                if mod.is_lora_active:
                    mod.unbind_active()

    def _unregister_handle(self, transformer, handle: AdapterHandle):
        for dotted, bank_idx in handle.target_indices.items():
            mod = self._path_to_module(transformer, dotted)
            mod.unregister_lora(bank_idx)


# --------------------------------------------------------------------
# Public pipeline classes
# --------------------------------------------------------------------
class WanPipelineRuntimeLoRA(_LoRAPipelineMixin, WanPipeline):
    """T2V pipeline with runtime LoRA support."""

    @staticmethod
    def create_pipeline(*args, **kwargs):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        return WanPipeline.create_pipeline(*args, pipeline_class=WanPipelineRuntimeLoRA, **kwargs)


class WanPipelineI2VRuntimeLoRA(_LoRAPipelineMixin, WanPipelineI2V):
    """I2V pipeline with runtime LoRA support."""

    @staticmethod
    def create_pipeline(*args, **kwargs):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        return WanPipeline.create_pipeline(*args, pipeline_class=WanPipelineI2VRuntimeLoRA, **kwargs)
