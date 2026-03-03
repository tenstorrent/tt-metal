# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sampling1D: Top-k/top-p/temperature sampling for 1D mesh topologies.

TTTv2 module — declarative config, lazy buffer allocation, k/p/temp as per-call args.
No mutable sampling state stored on the module.

See also: models/common/sampling/tt_sampling.py (TTTv1 source)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any, Optional

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_buffer import LazyBuffer, resolve_lazy_buffer
from models.common.modules.tt_ccl import get_tt_ccl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Sampling1DConfig:
    """Declarative config for Sampling1D.

    Buffer fields use the triple-type pattern: ``LazyBuffer | ttnn.Tensor | None``.
    k/p/temp are NOT stored here — they are per-call forward args.
    """

    vocab_size: int  # Required. Caller pre-pads to be divisible by num_devices.
    mesh_device: Optional[ttnn.MeshDevice] = None  # None → GetDefaultDevice()
    tt_ccl: Any = None  # None → get_tt_ccl(mesh_device) if multi-device
    max_batch_size: int = 32
    max_top_k: int = 32
    sub_core_grids: Any = None
    sub_core_grid_topk: Any = None
    start_core: Optional[ttnn.CoreCoord] = None  # None → CoreCoord(0,0)
    num_gather_links: int = 1
    sampling_memory_config: Optional[ttnn.MemoryConfig] = None  # None → DRAM_MEMORY_CONFIG
    allow_force_argmax: bool = False
    num_argmax_gather_links: Optional[int] = None  # None → same as num_gather_links
    ag_topology: Optional[ttnn.Topology] = None  # None → Topology.Linear

    # --- Persistent buffer specs (LazyBuffer | ttnn.Tensor | None) ---
    # Static index buffers (computed from vocab_size + num_devices, never mutated)
    index_offsets: LazyBuffer | ttnn.Tensor | None = None  # [1,1,32,max_top_k*num_devices], int32, TILE
    local_indices: LazyBuffer | ttnn.Tensor | None = (
        None  # [1,1,32,W], uint16, TILE (W=vocab for 1x1, per_dev_vocab otherwise)
    )
    # Seed/ID buffers (seeds mutable via LazyBuffer.update(), user_ids static)
    seeds: LazyBuffer | ttnn.Tensor | None = None  # [32], uint32, ROW_MAJOR
    user_ids: LazyBuffer | ttnn.Tensor | None = None  # [32], uint32, ROW_MAJOR

    @staticmethod
    def _buf_resolved(buf) -> bool:
        if buf is None:
            return False
        if isinstance(buf, ttnn.Tensor):
            return True
        return buf.is_resolved()

    def is_resolved(self) -> bool:
        if self.mesh_device is None:
            return False
        if self.mesh_device.get_num_devices() > 1 and self.tt_ccl is None:
            return False
        return all(
            self._buf_resolved(getattr(self, f)) for f in ("index_offsets", "local_indices", "seeds", "user_ids")
        )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class Sampling1D(LightweightModule):
    """Top-k/top-p/temperature sampling for 1D mesh topologies.

    k/p/temp are per-call forward args — NOT stored on the module. This eliminates
    mutable sampling state.
    """

    def __init__(self, vocab_size: int, mesh_device: ttnn.MeshDevice | None = None, **kwargs):
        """Happy path — minimal required args."""
        super().__init__()
        self.config = _resolve_sampling1d_config(
            Sampling1DConfig(vocab_size=vocab_size, mesh_device=mesh_device, **kwargs)
        )
        self._device_buffers_loaded = False
        self._bind_strategy()

    @classmethod
    def from_config(cls, config: Sampling1DConfig) -> Sampling1D:
        """Power path — fully custom config."""
        instance = object.__new__(cls)
        super(Sampling1D, instance).__init__()
        instance.config = _resolve_sampling1d_config(config)
        instance._device_buffers_loaded = False
        instance._bind_strategy()
        return instance

    # -- Strategy binding (TTTv2: no if-else in forward) ----------------------

    def _bind_strategy(self):
        """Bind self._topk to the correct strategy based on mesh topology."""
        cluster_shape = self.config.mesh_device.shape
        self._multi_step_reduction = list(cluster_shape) == [1, 1]
        if self._multi_step_reduction:
            self._topk = self._topk_single_device
        else:
            self._topk = self._topk_multi_device

        # Argmax strategy: single vs multi-device
        num_devices = self.config.mesh_device.get_num_devices()
        if num_devices > 1:
            self._pre_argmax_gather = self._argmax_all_gather
        else:
            self._pre_argmax_gather = self._argmax_noop

        # Memory config strategy for top-k post-processing
        cfg = self.config
        if cfg.sampling_memory_config is not None and cfg.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            self._prepare_topk_memory = self._topk_memory_sharded_roundtrip
        else:
            self._prepare_topk_memory = self._topk_memory_noop

        # CCL introspection (port from TTSampling.__init__ lines 77-91)
        self._line_all_gather = getattr(self.config.tt_ccl, "line_all_gather", None) if self.config.tt_ccl else None
        self._line_all_gather_supports_buffer_key = False
        self._line_all_gather_supports_dtype = False
        if callable(self._line_all_gather):
            try:
                sig = inspect.signature(self._line_all_gather)
                params = sig.parameters
                self._line_all_gather_supports_buffer_key = "buffer_key" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                self._line_all_gather_supports_dtype = "dtype" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                logger.warning("Unable to inspect line_all_gather signature")

    # -- Device buffers (idempotent) ------------------------------------------

    def load_device_buffers(self):
        """Materialize all LazyBuffer fields from resolved config."""
        if self._device_buffers_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device buffers!"
        cfg = self.config

        self._index_offsets = _materialize(cfg.index_offsets)
        self._local_indices = _materialize(cfg.local_indices)
        self._seeds = _materialize(cfg.seeds)
        self._user_ids = _materialize(cfg.user_ids)
        from models.common.utils import LogProbsCalculator  # lazy: transitively imports torch

        self._log_probs_calculator = LogProbsCalculator(cfg.mesh_device, cfg.sub_core_grids, cfg.tt_ccl)

        # Pre-compute static sub_core_grids for ttnn.sampling()
        self._sampling_sub_core_grids = (
            ttnn.num_cores_to_corerangeset_in_subcoregrids(
                cfg.start_core, cfg.max_batch_size, cfg.sub_core_grids, row_wise=True
            )
            if cfg.sub_core_grids is not None
            else None
        )

        self._device_buffers_loaded = True

    # -- Forward methods ------------------------------------------------------

    def decode_forward(
        self,
        logits: ttnn.Tensor,
        *,
        k: ttnn.Tensor | None = None,
        p: ttnn.Tensor | None = None,
        temp: ttnn.Tensor | None = None,
        seeds: ttnn.Tensor | None = None,
        tt_out_tok: ttnn.Tensor | None = None,
    ):
        """Sample tokens from logits.

        Args:
            logits: Input logits tensor (sharded across devices)
            k, p, temp: Per-call sampling parameters. All None + allow_force_argmax → argmax path.
                All provided → top-k sampling path.
            seeds: Optional per-call seed override. If None, uses config seeds buffer.
            tt_out_tok: Optional output tensor to write results to.

        Returns:
            (token_ids, log_probs_or_none)
        """
        self.load_device_buffers()
        cfg = self.config

        # Route: argmax or top-k
        if k is None and p is None and temp is None:
            if cfg.allow_force_argmax:
                return self._sample_argmax(logits, tt_out_tok)
            else:
                raise ValueError("k/p/temp are all None but allow_force_argmax is False")
        if k is None or p is None or temp is None:
            raise ValueError("k, p, temp must all be provided, or all be None (for argmax)")

        return self._sample_topk(logits, k, p, temp, seeds, tt_out_tok)

    def forward(self, logits, **kwargs):
        """Dispatcher."""
        return self.decode_forward(logits, **kwargs)

    # -- Argmax path (port from tt_sampling.py:310-341) -----------------------

    def _sample_argmax(self, logits, tt_out_tok):
        logits = self._pre_argmax_gather(logits)
        x_untilized = ttnn.untilize(logits, use_multicore=True)
        tt_out_tok = ttnn.argmax(
            x_untilized,
            dim=-1,
            output_tensor=tt_out_tok,
            keepdim=False,
            use_multicore=True,
        )
        log_probs = self._log_probs_calculator.calculate_log_probs(logits, tt_out_tok)
        return tt_out_tok, log_probs

    def _argmax_all_gather(self, logits):
        """Multi-device: all-gather logits before argmax."""
        cfg = self.config
        cluster_axis = 1
        return ttnn.experimental.all_gather_async(
            logits,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=cfg.num_argmax_gather_links,
            memory_config=logits.memory_config(),
            cluster_axis=cluster_axis,
            topology=cfg.ag_topology,
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=1,
            num_buffers_per_channel=2,
        )

    @staticmethod
    def _argmax_noop(logits):
        """Single-device: no gather needed."""
        return logits

    # -- Top-k memory strategies (bound at init, no if-else in forward) -------

    def _topk_memory_sharded_roundtrip(self, topk_values, topk_indices_int32):
        """Non-DRAM sampling_memory_config: round-trip through sharded memory."""
        cfg = self.config
        topk_values_sharded = ttnn.to_memory_config(
            topk_values, memory_config=cfg.sampling_memory_config, dtype=ttnn.bfloat16
        )
        topk_values = ttnn.to_memory_config(topk_values_sharded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(topk_values_sharded)
        topk_indices_int32 = ttnn.to_memory_config(topk_indices_int32, cfg.sampling_memory_config)
        return topk_values, topk_indices_int32

    @staticmethod
    def _topk_memory_noop(topk_values, topk_indices_int32):
        """DRAM memory config: no extra round-trip needed."""
        return topk_values, topk_indices_int32

    # -- Top-k sampling (port from tt_sampling.py:343-481) --------------------

    def _sample_topk(self, logits, k, p, temp, seeds, tt_out_tok):
        cfg = self.config

        x_bf16 = ttnn.typecast(logits, dtype=ttnn.bfloat16, sub_core_grids=cfg.sub_core_grids)

        # Strategy-dispatched top-k
        topk_values, topk_indices = self._topk(x_bf16)

        # Convert indices to int32
        topk_indices_int32 = ttnn.typecast(topk_indices, dtype=ttnn.int32, sub_core_grids=cfg.sub_core_grids)

        topk_values, topk_indices_int32 = self._prepare_topk_memory(topk_values, topk_indices_int32)

        # Add device offsets for global vocabulary indices
        topk_global_indices = ttnn.add(
            self._index_offsets,
            topk_indices_int32,
            dtype=ttnn.int32,
            memory_config=cfg.sampling_memory_config,
        )
        ttnn.deallocate(topk_indices_int32)

        topk_global_indices = ttnn.to_memory_config(topk_global_indices, ttnn.DRAM_MEMORY_CONFIG)
        topk_global_indices = ttnn.untilize(topk_global_indices, use_multicore=True, sub_core_grids=cfg.sub_core_grids)

        # Seed the RNG
        seeds_tensor = seeds if seeds is not None else self._seeds
        ttnn.manual_seed(
            seeds=seeds_tensor,
            user_ids=self._user_ids,
            sub_core_grids=cfg.sub_core_grids,
        )

        # Sample
        tt_out_tok = ttnn.sampling(
            topk_values,
            topk_global_indices,
            k=k,
            p=p,
            temp=temp,
            sub_core_grids=self._sampling_sub_core_grids,
            output_tensor=tt_out_tok,
        )

        ttnn.deallocate(topk_values)
        ttnn.deallocate(topk_global_indices)

        log_probs = self._log_probs_calculator.calculate_log_probs(logits, tt_out_tok)
        return tt_out_tok, log_probs

    # -- Top-k strategies (bound at init, no if-else in forward) --------------

    def _topk_single_device(self, x_bf16):
        """Split vocab in half → two topk → concat. Port of tt_sampling.py:346-371."""
        cfg = self.config
        x_list = ttnn.split(x_bf16, x_bf16.shape[-1] // 2, dim=3)
        indices_list = ttnn.split(self._local_indices, self._local_indices.shape[-1] // 2, dim=3)

        values_parts = []
        indices_parts = []
        for i in range(len(x_list)):
            vals, idxs = ttnn.topk(
                x_list[i],
                k=cfg.max_top_k,
                dim=-1,
                sub_core_grids=cfg.sub_core_grid_topk,
                indices_tensor=indices_list[i],
            )
            values_parts.append(vals)
            indices_parts.append(idxs)
            x_list[i].deallocate()
            indices_list[i].deallocate()

        gathered_values = ttnn.concat(values_parts, dim=3)
        gathered_indices = ttnn.concat(indices_parts, dim=3)

        for v, i in zip(values_parts, indices_parts):
            ttnn.deallocate(v)
            ttnn.deallocate(i)

        return gathered_values, gathered_indices

    def _topk_multi_device(self, x_bf16):
        """Local topk → all_gather across devices. Port of tt_sampling.py:372-421."""
        cfg = self.config
        cluster_shape = cfg.mesh_device.shape

        topk_values, topk_indices = ttnn.topk(
            x_bf16,
            k=cfg.max_top_k,
            dim=-1,
            sub_core_grids=cfg.sub_core_grid_topk,
            indices_tensor=self._local_indices,
        )

        # For 1D meshes use cluster_axis=None
        sampling_cluster_axis = None if 1 in cluster_shape else 0

        # Gather values
        gathered_values = self._perform_all_gather(
            topk_values,
            dim=3,
            cluster_axis=sampling_cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=cfg.num_gather_links,
            buffer_key="SAMPLING_VALUES",
        )
        ttnn.deallocate(topk_values)

        # Gather indices
        gathered_indices = self._perform_all_gather(
            topk_indices,
            dim=3,
            cluster_axis=sampling_cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=cfg.num_gather_links,
            buffer_key="SAMPLING_INDICES",
            dtype=ttnn.uint16,
        )
        ttnn.deallocate(topk_indices)

        return gathered_values, gathered_indices

    # -- CCL helper -----------------------------------------------------------

    def _perform_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links, buffer_key=None, dtype=None):
        """Flexible all-gather: prefer line_all_gather if available, else ttnn.all_gather.

        Port of TTSampling._perform_all_gather (tt_sampling.py:231-259).
        """
        if callable(self._line_all_gather):
            kwargs = {
                "dim": dim,
                "cluster_axis": cluster_axis,
                "memory_config": memory_config,
                "num_links": num_links,
            }
            if self._line_all_gather_supports_buffer_key and buffer_key is not None:
                kwargs["buffer_key"] = buffer_key
            if self._line_all_gather_supports_dtype and dtype is not None:
                kwargs["dtype"] = dtype
            return self._line_all_gather(tensor, **kwargs)

        return ttnn.all_gather(
            tensor,
            dim=dim,
            num_links=num_links,
            memory_config=memory_config,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
        )

    # -- (Backward compat) Model args factory ----------------------------------

    @classmethod
    def from_model_args(cls, mesh_device, tt_ccl, args, model_config=None) -> Sampling1D:
        """Backward compat factory for TTTv1 model args."""
        cluster_shape = mesh_device.shape
        if min(cluster_shape) > 1:
            raise ValueError(
                f"Sampling1D only supports 1D mesh topologies, got shape {cluster_shape}. "
                "Use TTSampling for Galaxy (2D) topologies."
            )
        padded_vocab_size = getattr(args, "padded_vocab_size", None)
        vocab_size = padded_vocab_size if padded_vocab_size is not None else args.vocab_size

        # Extract config from model_config dict
        mc = model_config or getattr(args, "model_config", {})
        num_gather_links = 1
        if "GALAXY_NUM_LINKS" in mc:
            max_links = mc["GALAXY_NUM_LINKS"]
            max_top_k = getattr(args, "max_top_k", 32)
            num_gather_links = min(max_top_k // 32, max_links) if max_top_k // 32 <= max_links else max_links

        sampling_memory_config = mc.get("DECODE_SAMPLING_INPUT_MEMCFG", ttnn.DRAM_MEMORY_CONFIG)

        allow_force_argmax = False
        num_argmax_gather_links = num_gather_links
        ag_topology = ttnn.Topology.Linear
        if "SAMPLING_AG_CONFIG" in mc:
            ag_cfg = mc["SAMPLING_AG_CONFIG"]
            allow_force_argmax = ag_cfg.get("allow_force_argmax", False)
            num_argmax_gather_links = ag_cfg.get("num_links", num_gather_links)
            ag_topology = ag_cfg.get("topology", ttnn.Topology.Linear)

        config = Sampling1DConfig(
            vocab_size=vocab_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            max_batch_size=getattr(args, "max_batch_size", 32),
            max_top_k=getattr(args, "max_top_k", 32),
            sub_core_grids=getattr(args, "sub_core_grids", None),
            sub_core_grid_topk=getattr(args, "sub_core_grid_topk", None),
            start_core=getattr(args, "start_core", ttnn.CoreCoord(0, 0)),
            num_gather_links=num_gather_links,
            sampling_memory_config=sampling_memory_config,
            allow_force_argmax=allow_force_argmax,
            num_argmax_gather_links=num_argmax_gather_links,
            ag_topology=ag_topology,
        )
        return cls.from_config(config)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_sampling1d_config(config: Sampling1DConfig) -> Sampling1DConfig:
    """Fill None fields with topology-aware defaults."""
    import torch

    to_set: dict = {}

    # Phase 1: Device and CCL
    mesh_device = config.mesh_device or ttnn.GetDefaultDevice()
    to_set["mesh_device"] = mesh_device

    cluster_shape = mesh_device.shape
    num_devices = mesh_device.get_num_devices()
    multi_step_reduction = list(cluster_shape) == [1, 1]

    if num_devices > 1 and config.tt_ccl is None:
        to_set["tt_ccl"] = get_tt_ccl(mesh_device)

    # Phase 2: Scalar config defaults
    if config.start_core is None:
        to_set["start_core"] = ttnn.CoreCoord(0, 0)
    if config.sampling_memory_config is None:
        to_set["sampling_memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    if config.num_argmax_gather_links is None:
        to_set["num_argmax_gather_links"] = config.num_gather_links
    if config.ag_topology is None:
        to_set["ag_topology"] = ttnn.Topology.Linear

    # Phase 3: Buffer specs
    B = config.max_batch_size
    K = config.max_top_k
    V = config.vocab_size
    replicate_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=cluster_shape)

    # num_devices_in_mesh for index computation
    if multi_step_reduction:
        num_devices_in_mesh = 2
    else:
        num_devices_in_mesh = max(cluster_shape[0], cluster_shape[1])
    per_device_vocab = V // num_devices_in_mesh

    def _resolve_buf(field_val, defaults, source_factory):
        if field_val is None:
            return LazyBuffer(source=source_factory(), **defaults)
        if isinstance(field_val, ttnn.Tensor):
            return field_val
        return resolve_lazy_buffer(field_val, **defaults)

    # index_offsets: [1, 1, B, K * num_devices_in_mesh]
    def _make_index_offsets():
        offsets = torch.ones(1, 1, B, K * num_devices_in_mesh, dtype=torch.int64)
        for device_id in range(num_devices_in_mesh):
            offsets[:, :, :, device_id * K : (device_id + 1) * K] = device_id * per_device_vocab
        return offsets

    idx_defaults = dict(
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["index_offsets"] = _resolve_buf(config.index_offsets, idx_defaults, _make_index_offsets)

    # local_indices: [1, 1, B, local_indices_width]
    # For multi-device: width = per_device_vocab (each device's shard)
    # For single-device split (multi_step_reduction): width = V (full vocab), so that
    #   after ttnn.split(..., V//2, dim=3) each half has width V//2 = per_device_vocab,
    #   matching the logits half width. Each half contains a 0-based range [0..V//2-1].
    #   Bug fix: TTTv1 used per_device_vocab here too, causing a 2x width mismatch
    #   between indices_tensor and logits in ttnn.topk on single-device.
    local_indices_width = V if multi_step_reduction else per_device_vocab

    def _make_local_indices():
        indices = torch.zeros(1, 1, B, local_indices_width, dtype=torch.int32)
        if multi_step_reduction:
            half = local_indices_width // 2
            for i in range(half):
                indices[:, :, :, i] = i
                indices[:, :, :, half + i] = i
        else:
            for i in range(local_indices_width):
                indices[:, :, :, i] = i
        return indices

    local_idx_defaults = dict(
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["local_indices"] = _resolve_buf(config.local_indices, local_idx_defaults, _make_local_indices)

    # seeds and user_ids: [B], uint32, ROW_MAJOR
    seed_defaults = dict(
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    to_set["seeds"] = _resolve_buf(
        config.seeds, seed_defaults, lambda: torch.arange(B, dtype=torch.int64).to(torch.int32)
    )
    to_set["user_ids"] = _resolve_buf(
        config.user_ids, seed_defaults, lambda: torch.arange(B, dtype=torch.int64).to(torch.int32)
    )

    resolved = replace(config, **to_set)
    assert resolved.is_resolved(), "Config not fully resolved after _resolve_sampling1d_config"
    return resolved


# -- Helper functions -------------------------------------------------------


def _materialize(buf):
    """Materialize a buffer field: ttnn.Tensor passthrough, LazyBuffer → get_device_buffer()."""
    if isinstance(buf, ttnn.Tensor):
        return buf
    return buf.get_device_buffer()
