# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Penalties1D: Presence/frequency/repetition penalty transforms for 1D mesh topologies.

TTTv2 module — declarative config, lazy buffer allocation, no mutable module state.
Penalty state (PenaltyParams, PenaltyAccumulator) is caller-constructed and passed
as arguments to forward methods.

See also: models/common/sampling/tt_penalties.py (TTTv1 source)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_buffer import LazyBuffer, resolve_lazy_buffer

# ---------------------------------------------------------------------------
# Caller-constructed penalty state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PenaltyParams:
    """Per-request penalty constants. Set before the decode loop, read-only during it.

    Caller allocates tensors directly (via LazyBuffer, ttnn.from_torch, or any other means)
    and constructs this dataclass. The module does NOT provide a convenience creation method.
    """

    prompt_mask: ttnn.Tensor  # [max_batch_size, vocab_per_device], int32, sharded
    presence_penalties: ttnn.Tensor  # [max_batch_size, 1], bfloat16
    frequency_penalties: ttnn.Tensor  # [max_batch_size, 1], bfloat16
    repetition_penalties: ttnn.Tensor  # [max_batch_size, 1], bfloat16
    inverse_repetition_penalties: ttnn.Tensor  # [max_batch_size, 1], bfloat16 (precomputed 1/rep)


@dataclass
class PenaltyAccumulator:
    """Per-step accumulator state. Mutated by update_output_tokens() after each sampled token.

    Caller allocates tensors directly and constructs this dataclass.
    """

    output_mask: ttnn.Tensor  # [max_batch_size, vocab_per_device], int32, sharded
    output_counts: ttnn.Tensor  # [max_batch_size, vocab_per_device], int32, sharded
    output_counts_gathered: ttnn.Tensor  # [max_batch_size, vocab_size], int32, replicated


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Penalties1DConfig:
    """Declarative config for Penalties1D.

    Buffer fields use the triple-type pattern: ``LazyBuffer | ttnn.Tensor | None``.

    - ``None`` → auto-filled by ``_resolve_penalties1d_config()`` with topology-aware defaults.
    - ``LazyBuffer`` → declarative spec, materialized lazily in ``load_device_buffers()``.
    - ``ttnn.Tensor`` → pre-allocated device tensor, used directly (power user bypass).
    """

    vocab_size: int  # Required. Caller pre-pads to be divisible by num_devices.
    mesh_device: Optional[ttnn.MeshDevice] = None  # None → GetDefaultDevice()
    max_batch_size: int = 32
    # todo)) sharding should be configurable! --> the defaults currently do not work in existing code
    sub_core_grids: Any = None  # From args.sub_core_grids

    # --- Persistent buffer specs (LazyBuffer | ttnn.Tensor | None) ---
    # Sharded vocab buffers: [max_batch_size, vocab_size], int32, TILE, sharded across devices
    prompt_mask: LazyBuffer | ttnn.Tensor | None = None
    output_mask: LazyBuffer | ttnn.Tensor | None = None
    output_counts: LazyBuffer | ttnn.Tensor | None = None
    # Replicated vocab buffers: [max_batch_size, vocab_size], int32, replicated
    output_counts_gathered: LazyBuffer | ttnn.Tensor | None = None
    zeros: LazyBuffer | ttnn.Tensor | None = None
    # Utility buffers
    decode_src: LazyBuffer | ttnn.Tensor | None = None  # [max_batch_size, 1], int32, ROW_MAJOR, ones
    # BF16 penalty param buffers: [max_batch_size, 1], bfloat16, TILE, replicated
    presence_penalties: LazyBuffer | ttnn.Tensor | None = None
    frequency_penalties: LazyBuffer | ttnn.Tensor | None = None
    repetition_penalties: LazyBuffer | ttnn.Tensor | None = None
    inverse_repetition_penalties: LazyBuffer | ttnn.Tensor | None = None

    @staticmethod
    def _buf_resolved(buf) -> bool:
        if buf is None:
            return False
        if isinstance(buf, ttnn.Tensor):
            return True
        return buf.is_resolved()

    def is_resolved(self) -> bool:
        return self.mesh_device is not None and all(
            self._buf_resolved(getattr(self, f))
            for f in (
                "prompt_mask",
                "output_mask",
                "output_counts",
                "output_counts_gathered",
                "zeros",
                "decode_src",
                "presence_penalties",
                "frequency_penalties",
                "repetition_penalties",
                "inverse_repetition_penalties",
            )
        )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class Penalties1D(LightweightModule):
    """Presence/frequency/repetition penalty transforms for 1D mesh topologies.

    Pure compute pipeline — all penalty state (PenaltyParams, PenaltyAccumulator) is
    caller-constructed and passed as arguments to forward methods.
    """

    def __init__(self, vocab_size: int, mesh_device: ttnn.MeshDevice | None = None, **kwargs):
        """Happy path — minimal required args, config auto-resolved."""
        super().__init__()
        self.config = _resolve_penalties1d_config(
            Penalties1DConfig(vocab_size=vocab_size, mesh_device=mesh_device, **kwargs)
        )
        self._device_buffers_loaded = False

    @classmethod
    def from_config(cls, config: Penalties1DConfig) -> Penalties1D:
        """Power path — fully custom config."""
        instance = object.__new__(cls)
        super(Penalties1D, instance).__init__()
        instance.config = _resolve_penalties1d_config(config)
        instance._device_buffers_loaded = False
        return instance

    @classmethod
    def from_model_args(cls, mesh_device, args) -> Penalties1D:
        """Backward compat factory for TTTv1 model args."""
        cluster_shape = mesh_device.shape
        if min(cluster_shape) > 1:
            raise ValueError(
                f"Penalties1D only supports 1D mesh topologies, got shape {cluster_shape}. "
                "Use TTPenalties for Galaxy (2D) topologies."
            )
        padded_vocab_size = getattr(args, "padded_vocab_size", None)
        vocab_size = padded_vocab_size if padded_vocab_size is not None else args.vocab_size
        sub_core_grids = getattr(args, "sub_core_grids", None)
        return cls(vocab_size=vocab_size, mesh_device=mesh_device, sub_core_grids=sub_core_grids)

    # -- Device buffers (idempotent) ------------------------------------------

    def load_device_buffers(self):
        """Materialize module-owned buffers. Called on first forward; idempotent."""
        if self._device_buffers_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device buffers!"
        cfg = self.config

        # Module-owned buffers
        self._decode_src = _materialize(cfg.decode_src)
        self._zeros = _materialize(cfg.zeros)

        # Derived topology fields
        self._cluster_shape = cfg.mesh_device.shape
        self._num_devices = max(self._cluster_shape[-1], self._cluster_shape[-2])
        self._op_kwargs = {"sub_core_grids": cfg.sub_core_grids} if cfg.sub_core_grids else {}

        # Slice tensors for scatter → slice (port from tt_penalties.py:117-139)
        self._slice_start, self._slice_end = self._build_slice_tensors()

        self._device_buffers_loaded = True

    # -- Lifecycle methods (per-request setup) ---------------------------------

    def init_prompt_penalties(
        self,
        params: PenaltyParams,
        accum: PenaltyAccumulator,
        prompt_tokens: "torch.Tensor",
    ) -> None:
        """Record prompt token positions into params.prompt_mask via scatter_add.

        Called once per request before the decode loop.
        Port of TTPenalties.reset_prompt_tokens (tt_penalties.py:194-212).
        """
        # todo)) can we get rid of the torch import here? --> will rethink the boundaries of the module when active development is done on the TTTv1 side
        import torch

        self.load_device_buffers()

        prompt_tokens_2d = prompt_tokens.reshape(-1, prompt_tokens.shape[-1])
        prompt_tokens_2d = self._pad_batch_to_max(prompt_tokens_2d, pad_value=-1)

        src_host = (prompt_tokens_2d != -1).to(torch.int32)
        idx_host = torch.where(prompt_tokens_2d == -1, torch.zeros_like(prompt_tokens_2d), prompt_tokens_2d)

        prompt_tokens_tt = ttnn.from_torch(
            idx_host,
            device=self.config.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        src_tt = ttnn.from_torch(
            src_host,
            device=self.config.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._token_bin_counts_and_mask(new_tokens=prompt_tokens_tt, src=src_tt, mask=params.prompt_mask)

    # -- Forward methods ------------------------------------------------------

    def decode_forward(
        self,
        logits: ttnn.Tensor,
        params: PenaltyParams,
        accum: PenaltyAccumulator,
    ) -> ttnn.Tensor:
        """Apply presence/frequency/repetition penalties to logits.

        Port of apply_penalties() at tt_penalties.py:32-75.
        """
        self.load_device_buffers()
        op = self._op_kwargs

        original_shape = logits.shape
        logits = ttnn.reshape(logits, (-1, original_shape[-1]))

        # Presence: logits -= typecast(output_mask, bf16) * presence
        presence_term = ttnn.multiply(
            ttnn.typecast(accum.output_mask, ttnn.bfloat16, **op), params.presence_penalties, **op
        )
        presence_term_bf16 = ttnn.typecast(presence_term, ttnn.bfloat16, **op)
        logits = ttnn.subtract(logits, presence_term_bf16, output_tensor=logits, **op)
        presence_term_bf16.deallocate()

        # Frequency: logits -= typecast(output_counts, bf16) * frequency
        output_counts_bf16 = ttnn.typecast(accum.output_counts, ttnn.bfloat16, **op)
        freq_term = ttnn.multiply(output_counts_bf16, params.frequency_penalties, **op)
        freq_term_bf16 = ttnn.typecast(freq_term, ttnn.bfloat16, **op)
        logits = ttnn.subtract(logits, freq_term_bf16, output_tensor=logits, **op)
        freq_term_bf16.deallocate()

        # Repetition: sign-dependent scaling
        combined_mask_int32 = ttnn.add(params.prompt_mask, accum.output_mask, **op)
        combined_mask = ttnn.typecast(combined_mask_int32, ttnn.bfloat16, **op)
        combined_mask_int32.deallocate()
        penalties = ttnn.where(combined_mask, params.repetition_penalties, 1.0, **op)
        inverse_penalties = ttnn.where(combined_mask, params.inverse_repetition_penalties, 1.0, **op)
        combined_mask.deallocate()

        logits_bf16 = ttnn.typecast(logits, ttnn.bfloat16, **op)
        logits_gt1 = ttnn.gt(logits_bf16, 0, **op)
        scaling = ttnn.where(logits_gt1, inverse_penalties, penalties, **op)
        logits_gt1.deallocate()
        penalties.deallocate()
        inverse_penalties.deallocate()
        logits = ttnn.multiply(logits, scaling, output_tensor=logits, **op)
        scaling.deallocate()

        return ttnn.reshape(logits, original_shape)

    def forward(self, logits, params=None, accum=None, **kwargs) -> ttnn.Tensor:
        """Dispatcher. If params/accum are None, returns logits unchanged."""
        if params is None or accum is None:
            return logits
        if "prompt_tokens" in kwargs:
            self.init_prompt_penalties(params, accum, kwargs["prompt_tokens"])
            return logits
        return self.decode_forward(logits, params, accum)

    # -- Accumulator operations (called AFTER sampling) -----------------------

    def update_output_tokens(self, accum: PenaltyAccumulator, new_tokens: ttnn.Tensor) -> None:
        """Update accum with newly sampled tokens.

        Port of TTPenalties.update_output_tokens (tt_penalties.py:247-264).
        Called after each decode step, between decode_forward and the next step.
        """
        self.load_device_buffers()

        if new_tokens.shape[-1] == self.config.max_batch_size and new_tokens.shape[-2] == 1:
            new_tokens = ttnn.reshape(new_tokens, [self.config.max_batch_size, 1], **self._op_kwargs)
            src = self._decode_src
        else:
            # todo)) can we get rid of the torch import here? --> will rethink the boundaries of the module when active development is done on the TTTv1 side
            import torch

            src = ttnn.from_torch(
                torch.ones(self.config.max_batch_size, new_tokens.shape[-1], dtype=torch.int32),
                device=self.config.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        self._token_bin_counts_and_mask(
            new_tokens=new_tokens,
            counts=accum.output_counts_gathered,
            src=src,
            counts_sliced=accum.output_counts,
            mask=accum.output_mask,
        )

    def reset_output_tokens(self, accum: PenaltyAccumulator, tokens: "torch.Tensor | None" = None) -> None:
        """Zero out accumulator buffers. Optionally re-initialize from provided tokens.

        Port of TTPenalties.reset_output_tokens (tt_penalties.py:214-245).
        """
        self.load_device_buffers()
        op = self._op_kwargs

        accum.output_mask = ttnn.mul(accum.output_mask, 0, output_tensor=accum.output_mask, **op)
        accum.output_counts = ttnn.mul(accum.output_counts, 0, output_tensor=accum.output_counts, **op)
        accum.output_counts_gathered = ttnn.mul(
            accum.output_counts_gathered, 0, output_tensor=accum.output_counts_gathered, **op
        )

        if tokens is not None:
            # todo)) can we get rid of the torch import here? --> will rethink the boundaries of the module when active development is done on the TTTv1 side
            import torch

            tokens_2d = tokens.reshape(-1, tokens.shape[-1])
            tokens_2d = self._pad_batch_to_max(tokens_2d, pad_value=-1)
            src_host = (tokens_2d != -1).to(torch.int32)
            idx_host = torch.where(tokens_2d == -1, torch.zeros_like(tokens_2d), tokens_2d)

            tokens_tt = ttnn.from_torch(
                idx_host,
                device=self.config.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            src_tt = ttnn.from_torch(
                src_host,
                device=self.config.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self._token_bin_counts_and_mask(
                new_tokens=tokens_tt,
                counts=accum.output_counts_gathered,
                src=src_tt,
                counts_sliced=accum.output_counts,
                mask=accum.output_mask,
            )

    # -- Private helpers ------------------------------------------------------

    def _build_slice_tensors(self):
        """Derive slice_start/slice_end from vocab_size and num_devices.

        Port of tt_penalties.py:117-139.
        """
        # todo)) can we get rid of the torch import here? --> will rethink the boundaries of the module when active development is done on the TTTv1 side
        import torch

        cfg = self.config
        vocab_per_dev = cfg.vocab_size // self._num_devices
        d = torch.arange(self._num_devices, dtype=torch.int32)

        if self._cluster_shape[-1] == self._num_devices:
            shard_dims_slice = (None, 0)
        else:
            shard_dims_slice = (0, None)

        start_1d = torch.empty(2 * self._num_devices, dtype=torch.int32)
        start_1d[0::2] = 0
        start_1d[1::2] = d * vocab_per_dev

        end_1d = torch.empty(2 * self._num_devices, dtype=torch.int32)
        end_1d[0::2] = cfg.max_batch_size
        end_1d[1::2] = (d + 1) * vocab_per_dev

        slice_start = ttnn.from_torch(
            start_1d,
            device=cfg.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(cfg.mesh_device, dims=shard_dims_slice, mesh_shape=self._cluster_shape),
        )
        slice_end = ttnn.from_torch(
            end_1d,
            device=cfg.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(cfg.mesh_device, dims=shard_dims_slice, mesh_shape=self._cluster_shape),
        )
        return slice_start, slice_end

    def _pad_batch_to_max(self, tokens_2d: "torch.Tensor", pad_value: int) -> "torch.Tensor":
        """Pad/truncate first dim to max_batch_size."""
        # todo)) can we get rid of the torch import here? --> will rethink the boundaries of the module when active development is done on the TTTv1 side
        import torch

        if tokens_2d.dim() != 2:
            raise ValueError(f"Expected 2D tensor [B, S], got {tokens_2d.shape}")
        B, S = tokens_2d.shape
        if B < self.config.max_batch_size:
            pad = torch.full((self.config.max_batch_size - B, S), pad_value, dtype=tokens_2d.dtype)
            return torch.cat([tokens_2d, pad], dim=0)
        if B > self.config.max_batch_size:
            return tokens_2d[: self.config.max_batch_size]
        return tokens_2d

    def _token_bin_counts_and_mask(self, new_tokens, src, counts=None, mask=None, counts_sliced=None):
        """Scatter tokens into histogram, slice per-device, compute mask.

        Port of TTPenalties.token_bin_counts_and_mask (tt_penalties.py:266-289).
        """
        op = self._op_kwargs
        counts_new = ttnn.scatter_add(self._zeros, 1, new_tokens, src, **op)

        new_tokens.deallocate()
        counts_new = ttnn.tilize(
            counts_new, **op, use_low_perf=True if self.config.sub_core_grids is not None else False
        )
        if counts is not None:
            counts = ttnn.add(counts, counts_new, output_tensor=counts, **op)
        else:
            counts = counts_new
        counts_sliced = ttnn.slice(
            counts,
            self._slice_start,
            self._slice_end,
            output_tensor=counts_sliced,
            slice_dim=1,
            num_devices=self._num_devices,
            **op,
        )

        mask = ttnn.gt(counts_sliced, 0, output_tensor=mask, **op)
        return counts, mask


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_penalties1d_config(config: Penalties1DConfig) -> Penalties1DConfig:
    """Fill None fields in config with topology-aware defaults.

    Power users who set fields explicitly will NOT have them overwritten.
    Mirrors the ``_resolve_mlp1d_config()`` pattern.
    """
    import torch  # lazy import — only needed for source tensor construction

    to_set: dict = {}

    # Phase 1: Device
    mesh_device = config.mesh_device or ttnn.GetDefaultDevice()
    to_set["mesh_device"] = mesh_device

    # Phase 2: Topology → shard_dims (port from tt_penalties.py:97-103)
    cluster_shape = mesh_device.shape
    num_devices = max(cluster_shape[-1], cluster_shape[-2])
    if cluster_shape[-1] == num_devices:
        shard_dims = (None, 1)
    else:
        shard_dims = (1, None)

    B = config.max_batch_size
    V = config.vocab_size

    # Build mesh mappers
    shard_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=cluster_shape)
    replicate_mapper = None  # None → replicate_tensor_to_mesh_mapper in LazyBuffer

    def _resolve_buf(field_val, defaults, source_factory):
        """Resolve a single buffer field: None → LazyBuffer, LazyBuffer → fill, ttnn.Tensor → passthrough."""
        if field_val is None:
            return LazyBuffer(source=source_factory(), **defaults)
        if isinstance(field_val, ttnn.Tensor):
            return field_val
        return resolve_lazy_buffer(field_val, **defaults)

    # Phase 3: Sharded vocab buffers — [B, V], int32, TILE, sharded
    sharded_vocab_defaults = dict(
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=shard_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros_BV = lambda: torch.zeros(B, V, dtype=torch.int32)
    to_set["prompt_mask"] = _resolve_buf(config.prompt_mask, sharded_vocab_defaults, zeros_BV)
    to_set["output_mask"] = _resolve_buf(config.output_mask, sharded_vocab_defaults, zeros_BV)
    to_set["output_counts"] = _resolve_buf(config.output_counts, sharded_vocab_defaults, zeros_BV)

    # Phase 4: Replicated vocab buffers — [B, V], int32, replicated
    replicated_vocab_defaults = dict(
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["output_counts_gathered"] = _resolve_buf(config.output_counts_gathered, replicated_vocab_defaults, zeros_BV)

    replicated_vocab_rm_defaults = dict(
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["zeros"] = _resolve_buf(config.zeros, replicated_vocab_rm_defaults, zeros_BV)

    # Phase 5: Utility buffers
    decode_src_defaults = dict(
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["decode_src"] = _resolve_buf(
        config.decode_src, decode_src_defaults, lambda: torch.ones(B, 1, dtype=torch.int32)
    )

    # Phase 6: BF16 penalty param buffers — [B, 1], bfloat16, TILE, replicated
    bf16_param_defaults = dict(
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate_mapper,
        memory_config=None,
    )
    zeros_B1 = lambda: torch.zeros(B, 1, dtype=torch.float32)
    for field_name in (
        "presence_penalties",
        "frequency_penalties",
        "repetition_penalties",
        "inverse_repetition_penalties",
    ):
        to_set[field_name] = _resolve_buf(getattr(config, field_name), bf16_param_defaults, zeros_B1)

    resolved = replace(config, **to_set)
    assert resolved.is_resolved(), "Config not fully resolved after _resolve_penalties1d_config"
    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _materialize(buf):
    """Materialize a buffer field: ttnn.Tensor passthrough, LazyBuffer → get_device_buffer()."""
    if isinstance(buf, ttnn.Tensor):
        return buf
    return buf.get_device_buffer()
