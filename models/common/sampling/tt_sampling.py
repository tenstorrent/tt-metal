# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import os
import sys

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.sampling._utils import compact_debug_list as _compact_debug_list
from models.common.sampling._utils import is_default_value, is_llama33_70b_model, is_power_of_2
from models.common.sampling._utils import log_sampling_debug as _log_sampling_debug
from models.common.sampling._utils import upper_power_of_2
from models.common.sampling.tt_log_probs import LogProbsCalculator
from models.common.sampling.vocab_padding import (
    build_invalid_vocab_mask,
    build_tail_invalid_vocab_mask,
    get_vocab_shard_dims,
)


class TTSampling(LightweightModule):
    """
    On-device sampling module supporting top-k, top-p, and temperature-based sampling.

    This class implements high-performance on-device sampling that can work across different
    model implementations by accepting configuration parameters rather than assuming specific
    args structures.

    Multi-device sampling works by partitioning the vocabulary across devices. Each device
    computes top-k locally on its vocabulary partition, then all-gather operations combine
    the results across devices to perform global top-k selection before final sampling.

    Args:
        mesh_device: The device or MeshDevice for computations
        tt_ccl: CCL object for distributed operations (supports both line_all_gather and tt_all_gather)
        vocab_size: Vocabulary size of the model
        padded_vocab_size: Padded vocabulary size (must be divisible by num devices)
        max_batch_size: Maximum batch size supported
        max_top_k: Maximum number of top-k tokens to consider
        cluster_shape: Shape of the device cluster (rows, cols)
        sampling_all_gather_axis: Axis to all-gather over in 2D meshes (0=rows, 1=cols, default: 0)
        sub_core_grids: Sub-core grid configuration for operations
        sub_core_grid_topk: Sub-core grid configuration specifically for top-k operations
        start_core: Starting core coordinate for sampling operations
        num_gather_links: Number of links to use for all-gather operations (optional)
        sampling_memory_config: Memory configuration for sampling tensors (optional)
        k, p, temp: Initial sampling parameters (tensors of size max_batch_size)

    Note:
        Uses persistent buffers when CCL supports line_all_gather (llama3_70b_galaxy),
        otherwise uses standard all_gather where the CCL API handles memory allocation (tt-transformers).
    """

    def _is_force_argmax_sampling(self, k, p, temp):
        """Detect whether all users request deterministic greedy decoding.

        When every user in the batch has k=1 (top-1), p=0.0 or p=1.0 (no top-p filter),
        and temp=1.0 (no temperature scaling), we can skip the full top-k / top-p /
        temperature / RNG pipeline and use a single all-gather + argmax instead.
        This is significantly faster because argmax needs only one all-gather of the
        full logits tensor vs. three gathers (values, indices, sampled tokens) in the
        normal path.

        Note: callers may represent greedy rows with p=1.0, while the
        device argmax-style representation uses p=0.0.
        The model config must also set allow_force_argmax=True for this to activate.

        Changing this state between decode steps invalidates captured traces, so
        SamplingGenerator maintains separate trace slots keyed by force_argmax.
        """
        return (
            self._allow_force_argmax_sampling
            and is_default_value(k, 1)
            and (is_default_value(p, 1.0) or is_default_value(p, 0.0))
            and is_default_value(temp, 1.0)
        )

    def _select_topk_indices_dtype(self, per_device_vocab_size: int, multi_step_reduction: bool):
        # if vocab is larger than uint16 max, return uint32 for indices
        if per_device_vocab_size > torch.iinfo(torch.uint16).max:
            return ttnn.uint32

        # if vocab size is missaligned with tile size and multi-step reduction is used, we need uint32 because of slice op compatibility
        if multi_step_reduction and (per_device_vocab_size // 2) % ttnn.TILE_SIZE != 0:
            return ttnn.uint32

        return ttnn.uint16

    @property
    def force_argmax_sampling(self) -> bool:
        return self._force_argmax_sampling

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        k=None,
        p=None,
        temp=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self._sampling_debug_enabled = is_llama33_70b_model(args)
        # Multi-step reduction is supported only on single device
        self.multi_step_reduction = list(mesh_device.shape) == [1, 1]
        self.tt_ccl = tt_ccl
        self._line_all_gather = getattr(self.tt_ccl, "line_all_gather", None)
        self._line_all_gather_supports_buffer_key = False
        self._line_all_gather_supports_dtype = False
        self.pad_to_power_of_2 = getattr(args, "pad_logits_to_power_of_2", False)
        if callable(self._line_all_gather):
            try:
                line_all_gather_sig = inspect.signature(self._line_all_gather)
                line_all_gather_params = line_all_gather_sig.parameters
                self._line_all_gather_supports_buffer_key = "buffer_key" in line_all_gather_params or any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in line_all_gather_params.values()
                )
                self._line_all_gather_supports_dtype = "dtype" in line_all_gather_params or any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in line_all_gather_params.values()
                )
            except (TypeError, ValueError):
                logger.warning("Unable to inspect line_all_gather signature; assuming no buffer_key or dtype support.")

        padded_vocab_size = getattr(args, "padded_vocab_size", None)
        self.padded_vocab_size = padded_vocab_size if padded_vocab_size is not None else args.vocab_size
        self.vocab_size = args.vocab_size
        # Round up to the next tile boundary (32) — device tensors must be tile-aligned.
        raw_batch = getattr(args, "max_batch_size", 32)
        self.max_batch_size = max(32, ((raw_batch + 31) // 32) * 32)
        self.max_top_k = getattr(args, "max_top_k", 32)
        self.cluster_shape = args.cluster_shape

        self.sampling_all_gather_axis = getattr(args, "sampling_all_gather_axis", 0)
        self.sub_core_grids = getattr(args, "sub_core_grids", None)
        self.sub_core_grid_topk = getattr(args, "sub_core_grid_topk", None)
        self.start_core = getattr(args, "start_core", ttnn.CoreCoord(0, 0))
        self._sampling_sub_core_grids = (
            ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, self.max_batch_size, self.sub_core_grids, row_wise=True
            )
            if self.sub_core_grids is not None
            else None
        )
        # Blackhole galaxy prefetcher (unfused-CCL) keeps a split senders/worker sub-device manager
        # loaded during prefill sampling (prefill samples in decode mode). Auto-multicore ops grab the
        # full 12-wide compute grid, which includes the dispatch column that no loaded sub-device covers
        # ("kernel group cores do not match sub device cores"). Pin the force-argmax untilize/argmax to
        # the worker sub-core grids so they stay inside the worker sub-device. Left None elsewhere so no
        # other arch's behaviour changes.
        # force_argmax_sub_core_grids (optional): a wider grid than sub_core_grids for the
        # force-argmax untilize/argmax. Argmax is a scalar-RISC compare loop over the full gathered
        # vocab, so its runtime scales ~1/num_cores; the BH galaxy passes the full 100-core worker
        # sub-device here instead of the 40-core sub_core_grids (2.1 ms -> ~0.85 ms per token).
        self._force_argmax_sub_core_grids = (
            (getattr(args, "force_argmax_sub_core_grids", None) or self.sub_core_grids)
            if getattr(args, "use_unfused_ccl", False)
            else None
        )
        # Distributed force-argmax (see _distributed_force_argmax): lazily created constants.
        self._dist_argmax_iota = None
        self._dist_argmax_fp32_ckc = None

        # sampling_dp > 1 when multiple mesh groups each sample users independently
        # (e.g. GPT-OSS on [4,8]: 4 rows × 32 users; Llama Galaxy on [8,4]: 4 cols × 8 users)
        self._sampling_dp = getattr(args, "sampling_dp", 1)
        # Shard params along the non-all-gather axis; replicate along the all-gather axis
        if self._sampling_dp > 1:
            if self.sampling_all_gather_axis == 0:
                self._param_dims = (None, 0)  # shard along cols
            else:
                self._param_dims = (0, None)  # shard along rows
        else:
            self._param_dims = (None, None)

        if hasattr(args, "model_config") and "GALAXY_NUM_LINKS" in args.model_config:
            # Calculate num_gather_links based on model config
            max_num_gather_links = args.model_config["GALAXY_NUM_LINKS"]
            self.num_gather_links = (
                args.max_top_k // 32 if args.max_top_k // 32 <= max_num_gather_links else max_num_gather_links
            )
        else:
            self.num_gather_links = 1
        if hasattr(args, "model_config") and "DECODE_SAMPLING_INPUT_MEMCFG" in args.model_config:
            self.sampling_memory_config = args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"]
        else:
            self.sampling_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Force argmax sampling
        if hasattr(args, "model_config") and "SAMPLING_AG_CONFIG" in args.model_config:
            # The model config may describe the fastest full-size Galaxy path, but
            # the actual CCL shape is resolved from the runtime mesh below.
            sampling_ag_config = args.model_config["SAMPLING_AG_CONFIG"]
            self._allow_force_argmax_sampling = sampling_ag_config["allow_force_argmax"]
            self.num_argmax_gather_links = sampling_ag_config["num_links"]
            self.argmax_chunks_per_sync = sampling_ag_config.get("chunks_per_sync", 10)
            self.argmax_num_workers_per_link = 1
            self.ag_topology = sampling_ag_config["topology"]
        else:
            self._allow_force_argmax_sampling = False
            self.num_argmax_gather_links = self.num_gather_links
            self.argmax_chunks_per_sync = 10
            self.argmax_num_workers_per_link = 1
            self.ag_topology = ttnn.Topology.Linear

        # Set defaults for sampling parameters if not provided
        # Default: k=1 (top-1), p=0 (effectively argmax), temp=1 (no temperature scaling)
        # When p=0, the sampling operation will select the token with highest probability (argmax)
        total_param_size = self.max_batch_size * self._sampling_dp
        if k is None:
            k = torch.ones(total_param_size)
        if p is None:
            p = torch.zeros(total_param_size)
        if temp is None:
            temp = torch.ones(total_param_size)

        self._force_argmax_sampling = self._is_force_argmax_sampling(k, p, temp)

        # Create sampling parameter tensors on device
        # When _sampling_dp > 1, dims=(0, None) shards the [128] tensor across 4 rows → [32] per row
        self.k_tensor = ttnn.from_torch(
            k,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._param_dims, mesh_shape=self.cluster_shape),
        )
        self.p_tensor = ttnn.from_torch(
            p,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._param_dims, mesh_shape=self.cluster_shape),
        )
        self.temp_tensor = ttnn.from_torch(
            temp,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._param_dims, mesh_shape=self.cluster_shape),
        )

        # Create device offset indices for global indexing
        self._create_indices_tensors()
        self._create_invalid_vocab_mask()
        # Log-probs tensor to store the log-probs for the batch
        self.tt_log_probs = None
        self.log_probs_calculator = LogProbsCalculator(
            self.mesh_device,
            self.sub_core_grids,
            self.tt_ccl,
            batch_size=self.max_batch_size,
            use_topk_logprobs=getattr(args, "use_topk_logprobs", False),
        )

        # Seeds tensor: one RNG slot per user across all rows.
        # When sampling_dp > 1, shard across rows so each row gets its own slice.
        # user_ids tensor: core routing only (32 per row, replicated).
        self.seeds_tt_tensor = ttnn.from_torch(
            torch.arange(total_param_size).to(torch.uint32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._param_dims, mesh_shape=self.cluster_shape)
            if self._sampling_dp > 1
            else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.user_ids_tt_tensor = ttnn.as_tensor(
            torch.arange(self.max_batch_size).to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _get_num_sampling_shards(self):
        if self.multi_step_reduction:
            return 2
        if 1 in self.cluster_shape:
            return max(self.cluster_shape[0], self.cluster_shape[1])

        if self.sampling_all_gather_axis not in (0, 1):
            raise ValueError(
                f"sampling_all_gather_axis must be 0 or 1 for 2D meshes, got {self.sampling_all_gather_axis}"
            )
        return self.cluster_shape[self.sampling_all_gather_axis]

    def _create_indices_tensors(self):
        """Create the indices tensors needed for distributed top-k operations."""
        num_devices_in_mesh = self._get_num_sampling_shards()
        indices_device_offsets = torch.ones(
            1, 1, self.max_batch_size, self.max_top_k * num_devices_in_mesh, dtype=torch.int64
        )
        # padded_per_device: tile-aligned width matching actual logit tensors (for indices tensor)
        padded_per_device = self.padded_vocab_size // num_devices_in_mesh

        for device_id in range(num_devices_in_mesh):
            indices_device_offsets[:, :, :, device_id * self.max_top_k : (device_id + 1) * self.max_top_k] = (
                device_id * padded_per_device
            )
        self.tt_indices_device_offsets = ttnn.from_torch(
            indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Create local indices tensor for top-k operations (must match logit width)
        indices_tensor_torch = torch.zeros(1, 1, self.max_batch_size, padded_per_device, dtype=torch.int32)
        for i in range(padded_per_device):
            indices_tensor_torch[:, :, :, i] = i

        # pad to power of 2 if needed
        if self.pad_to_power_of_2 and not is_power_of_2(indices_tensor_torch.shape[-1]):
            padded_value = upper_power_of_2(indices_tensor_torch.shape[-1])
            indices_tensor_torch = torch.nn.functional.pad(
                indices_tensor_torch,
                (0, padded_value - indices_tensor_torch.shape[-1]),  # pad only last dim
                mode="constant",
                value=-1,  # invalid index to ensure that the padding values are not used
            )

        indices_dtype = self._select_topk_indices_dtype(padded_per_device, self.multi_step_reduction)
        self.tt_indices_tensor = ttnn.from_torch(
            indices_tensor_torch,
            dtype=indices_dtype,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _create_invalid_vocab_mask(self):
        self.tt_invalid_vocab_mask = None
        self.tt_invalid_vocab_tail_mask = None
        self._invalid_vocab_tail_width = 0

        vocab_shard_dims = get_vocab_shard_dims(self.cluster_shape, self.sampling_all_gather_axis)
        tail_mask = build_tail_invalid_vocab_mask(
            self.vocab_size,
            self.padded_vocab_size,
            self.max_batch_size,
            self.cluster_shape,
            self.sampling_all_gather_axis,
            tile_size=ttnn.TILE_SIZE,
        )
        if tail_mask is not None:
            self._invalid_vocab_tail_width = tail_mask.tail_width
            self.tt_invalid_vocab_tail_mask = ttnn.from_torch(
                tail_mask.mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=vocab_shard_dims,
                    mesh_shape=self.cluster_shape,
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return

        invalid_vocab_mask = build_invalid_vocab_mask(
            self.vocab_size,
            self.padded_vocab_size,
            self.max_batch_size,
        )
        if invalid_vocab_mask is None:
            return

        self.tt_invalid_vocab_mask = ttnn.from_torch(
            invalid_vocab_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=vocab_shard_dims,
                mesh_shape=self.cluster_shape,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mask_invalid_vocab_logits(self, logits):
        if self.tt_invalid_vocab_tail_mask is not None:
            return self._mask_invalid_vocab_tail_logits(logits)
        if self.tt_invalid_vocab_mask is None:
            return logits
        return ttnn.add(
            logits,
            self.tt_invalid_vocab_mask,
            memory_config=logits.memory_config(),
            sub_core_grids=self.sub_core_grids,
        )

    def _mask_invalid_vocab_tail_logits(self, logits):
        tail_width = self._invalid_vocab_tail_width
        local_width = logits.shape[-1]
        valid_width = local_width - tail_width
        if tail_width <= 0 or valid_width < 0:
            return self._mask_invalid_vocab_logits_fallback(logits)
        if valid_width == 0:
            return ttnn.add(
                logits,
                self.tt_invalid_vocab_tail_mask,
                memory_config=logits.memory_config(),
                sub_core_grids=self.sub_core_grids,
            )

        valid_logits = ttnn.slice(
            logits,
            [0, 0, 0, 0],
            [logits.shape[0], logits.shape[1], logits.shape[2], valid_width],
            memory_config=logits.memory_config(),
            sub_core_grids=self.sub_core_grids,
        )
        tail_logits = ttnn.slice(
            logits,
            [0, 0, 0, valid_width],
            [logits.shape[0], logits.shape[1], logits.shape[2], local_width],
            memory_config=logits.memory_config(),
            sub_core_grids=self.sub_core_grids,
        )
        masked_tail_logits = ttnn.add(
            tail_logits,
            self.tt_invalid_vocab_tail_mask,
            memory_config=logits.memory_config(),
            sub_core_grids=self.sub_core_grids,
        )
        masked_logits = ttnn.concat(
            [valid_logits, masked_tail_logits],
            dim=3,
            memory_config=logits.memory_config(),
            sub_core_grids=self.sub_core_grids,
        )
        ttnn.deallocate(valid_logits)
        ttnn.deallocate(tail_logits)
        ttnn.deallocate(masked_tail_logits)
        return masked_logits

    def _mask_invalid_vocab_logits_fallback(self, logits):
        if self.tt_invalid_vocab_mask is None:
            return logits
        return ttnn.add(
            logits,
            self.tt_invalid_vocab_mask,
            memory_config=logits.memory_config(),
            sub_core_grids=self.sub_core_grids,
        )

    def _can_slice_valid_vocab_for_argmax(self):
        return self.vocab_size < self.padded_vocab_size and self.vocab_size % ttnn.TILE_SIZE == 0

    def _slice_valid_vocab_for_argmax(self, logits):
        if not self._can_slice_valid_vocab_for_argmax() or logits.shape[-1] != self.padded_vocab_size:
            return logits
        return ttnn.slice(
            logits,
            [0, 0, 0, 0],
            [logits.shape[0], logits.shape[1], logits.shape[2], self.vocab_size],
            memory_config=logits.memory_config(),
            sub_core_grids=self.sub_core_grids,
        )

    def _perform_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links, buffer_key=None, dtype=None):
        """
        Flexible all-gather that works across different CCL implementations.

        - If `tt_ccl` exposes `line_all_gather`, prefer it (enables persistent buffer usage on some stacks).
        - Otherwise fall back to `ttnn.all_gather`.
        """
        if callable(self._line_all_gather):
            # Some implementations accept `buffer_key` (for persistent buffers), others may not.
            line_all_gather_kwargs = {
                "dim": dim,
                "cluster_axis": cluster_axis,
                "memory_config": memory_config,
                "num_links": num_links,
            }
            if self._line_all_gather_supports_buffer_key and buffer_key is not None:
                line_all_gather_kwargs["buffer_key"] = buffer_key
            if self._line_all_gather_supports_dtype and dtype is not None:
                line_all_gather_kwargs["dtype"] = dtype
            return self._line_all_gather(tensor, **line_all_gather_kwargs)

        return ttnn.all_gather(
            tensor,
            dim=dim,
            num_links=num_links,
            memory_config=memory_config,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
        )

    def _get_sampling_cluster_axis(self):
        if self.mesh_device.get_num_devices() <= 1:
            return None
        # 1D submeshes should use the default CCL axis; forcing axis 1 can make
        # smaller Galaxy DP groups request routes outside the submesh.
        if 1 in self.cluster_shape:
            return None
        return self.sampling_all_gather_axis

    def _get_force_argmax_all_gather_config(self, cluster_axis):
        num_links = self.num_argmax_gather_links
        if hasattr(self.tt_ccl, "get_num_links"):
            # Clamp the tuned config to the links available on the actual submesh.
            num_links = min(num_links, self.tt_ccl.get_num_links(cluster_axis))

        topology = self.ag_topology
        # Ring is available for T3K-like 8-device groups; smaller DP groups need
        # linear routing to avoid wraparound routes such as D0 -> D12.
        if self.mesh_device.get_num_devices() < 8:
            topology = ttnn.Topology.Linear

        return max(1, num_links), topology

    def _use_distributed_argmax(self, x, tt_out_tok, cluster_axis):
        """Whether the distributed (local-argmax + tiny gather) greedy path applies.

        Only used on the BH galaxy prefetcher path (where _force_argmax_sub_core_grids is set):
        the full-logits gather + full-width argmax there costs ~1.15 ms/token, dominated by the
        scalar-RISC argmax over the 32 x 155648 gathered tensor. The distributed variant argmaxes
        each device's 32 x (vocab/ncols) shard locally and combines per-column (value, index)
        candidates, ~5x cheaper. Kept off elsewhere so no other model's behaviour changes.
        """
        if os.environ.get("QWEN_BH_DIST_ARGMAX", "1") != "1":
            return False
        if self._force_argmax_sub_core_grids is None or self._sampling_dp != 1:
            return False
        if cluster_axis is None:
            return False
        # A caller-provided output buffer (the decode trace's token input in the galaxy demo flow,
        # RM uint32 [1,1,1,B]) is written back via a worker-grid copy; only a single-page RM
        # uint32 buffer with B in the last dim is supported.
        if tt_out_tok is not None:
            shp = list(tt_out_tok.shape)
            if (
                tt_out_tok.dtype != ttnn.uint32
                or tt_out_tok.layout != ttnn.ROW_MAJOR_LAYOUT
                or shp[-1] != self.max_batch_size
                or math.prod(shp) != self.max_batch_size
            ):
                return False
        w = x.shape[-1]
        return x.shape[-2] == self.max_batch_size and self.max_batch_size == 32 and w % ttnn.TILE_SIZE == 0

    def _distributed_force_argmax(self, x, cluster_axis, topology, tt_out_tok=None):
        """Greedy sampling via per-device argmax + tiny cross-column combine.

        Instead of all-gathering the full padded-vocab logits (32 x 155648) and running one huge
        argmax, each column device computes the argmax/max over its local 32 x W shard, then only
        the per-column candidate (max value, argmax index) pairs are gathered (one tile per
        device). The winner column per user is picked by a small argmax over the gathered values,
        and the final token id is reconstructed as local_idx[winner] + W * winner_col via a
        one-hot select (byte-split fp32 select + exact int32 recombination; see the TF32 note
        below).

        Tie-break matches the single-op argmax: within a column argmax returns the first max, and
        the cross-column argmax picks the lowest winning column, i.e. the global first occurrence.

        Everything is pinned to the worker sub-core grid / sub-device (same reasons as the
        non-distributed path: split senders/worker sub-device manager on the BH galaxy).
        """
        grids = self._force_argmax_sub_core_grids
        w = x.shape[-1]
        batch = self.max_batch_size
        ncols = self.cluster_shape[cluster_axis]
        if self._dist_argmax_iota is None:
            # Row-index iota over the gathered-candidates height (ncols tiles of 32 rows): the
            # gathered index tensor holds column c's candidates at row block 32*c, and the winner
            # position from argmax over the gathered (tile-padded) values is exactly 32*c.
            rows = ncols * ttnn.TILE_SIZE
            iota = torch.arange(rows, dtype=torch.float32).reshape(1, 1, rows, 1).expand(1, 1, rows, batch)
            self._dist_argmax_iota = ttnn.from_torch(
                iota.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            # fp32 accumulation: bf16 logits and index sums up to padded_vocab must stay exact
            # (fp16 dest would round integers > 2048 and saturate logits > 65504).
            self._dist_argmax_fp32_ckc = ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
            )

        barrier_accessor = getattr(
            self.tt_ccl,
            "get_sampling_barrier_semaphore_handle",
            self.tt_ccl.get_and_cycle_barrier_semaphore_handle,
        )
        ag_sub_device_id = getattr(self.tt_ccl, "worker_sub_device_id", None)

        _debug_sync = os.environ.get("QWEN_DIST_ARGMAX_SYNC") == "1"

        def ck(step):
            # Debug: block after each step so a device hang is attributable to one op.
            if _debug_sync:
                ttnn.synchronize_device(self.mesh_device)
                logger.info(f"dist-argmax step ok: {step}")

        def argmax_grid(width):
            # The multicore argmax factory gives every core ceil(blocks/num_cores) reduction units
            # and only lets the LAST core be partial. If (num_cores-1)*units_per_core >= width the
            # last core's unit count underflows (huge uint32) and trailing cores read out of bounds
            # -> device hang. Cap the core count so every core lands inside the row. Granularity 64
            # is a safe multiple of the factory's alignment-derived minimum (16/32/64 elements).
            gran = 64
            blocks = -(-width // gran)
            max_cores = grids.num_cores()
            units_per_core = -(-blocks // max_cores) * gran
            n = -(-width // units_per_core)
            if n >= max_cores:
                return grids
            return ttnn.num_cores_to_corerangeset_in_subcoregrids(self.start_core, n, grids, row_wise=True)

        def tiny_gather(t, dim):
            # 1 link: each device contributes a single tile page, nothing to split across links.
            return ttnn.experimental.all_gather_async(
                t,
                persistent_output_buffer=None,
                dim=dim,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=cluster_axis,
                topology=topology,
                barrier_semaphore=barrier_accessor(cluster_axis),
                chunks_per_sync=self.argmax_chunks_per_sync,
                num_workers_per_link=1,
                num_buffers_per_channel=2,
                subdevice_id=ag_sub_device_id,
            )

        # Local shard argmax + max. Typecast to bf16 first so the reduce max value is exactly the
        # value the argmax saw (a bfp8_b-quantized reduce output could disagree with the bf16
        # untilize output near cross-column ties).
        x_bf16 = ttnn.typecast(x, ttnn.bfloat16, sub_core_grids=grids) if x.dtype != ttnn.bfloat16 else x
        ck("typecast x bf16")
        x_unt = ttnn.untilize(x_bf16, use_multicore=True, sub_core_grids=grids)
        ck("untilize local")
        local_idx = ttnn.argmax(x_unt, dim=-1, keepdim=False, sub_core_grids=argmax_grid(w))  # [1,1,B] RM u32
        ck("argmax local")
        # Rank-4 view (same single RM page) so the pad/tilize/gather below address H and W plainly.
        local_idx = ttnn.reshape(local_idx, [1, 1, 1, batch], sub_core_grids=grids)
        ck("reshape local_idx")
        x_unt.deallocate(True)
        local_max = ttnn.max(
            x_bf16, dim=3, keepdim=True, sub_core_grids=grids, compute_kernel_config=self._dist_argmax_fp32_ckc
        )  # [1,1,B,1] tile bf16
        ck("max local")
        if x_bf16 is not x:
            x_bf16.deallocate(True)

        # Values: pad the [B,1] column to a full logical tile width with -inf so the gathered
        # [B, 32*ncols] tensor has column c's max at width 32*c and -inf in the pad lanes (the
        # physical tile pad lanes of the reduce output are undefined, so they must be overwritten).
        # Sub-tile-width pad on TILE layout is unproven here, so round-trip through row-major.
        local_max_rm = ttnn.untilize(local_max, use_multicore=True, sub_core_grids=grids)
        ck("untilize local_max")
        local_max.deallocate(True)
        vals_rm = ttnn.pad(
            local_max_rm, [(0, 0), (0, 0), (0, 0), (0, ttnn.TILE_SIZE - 1)], value=-3.38e38, sub_core_grids=grids
        )
        ck("pad vals")
        local_max_rm.deallocate(True)
        vals_p = ttnn.tilize(vals_rm, sub_core_grids=grids)  # [1,1,B,32] tile bf16
        ck("tilize vals")
        vals_rm.deallocate(True)
        vals_g = tiny_gather(vals_p, dim=3)  # [1,1,B,32*ncols] tile bf16
        ck("gather vals")
        vals_p.deallocate(True)

        # Indices: pad the [1,B] row to a full logical tile height (zeros), tilize, and typecast
        # to int32 (exact) for the gather. Gather on dim 2 stacks column c's candidates at row
        # block 32*c.
        idx_p = ttnn.pad(local_idx, [(0, 0), (0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0)], value=0, sub_core_grids=grids)
        ck("pad idx")
        local_idx.deallocate(True)
        idx_t = ttnn.tilize(idx_p, sub_core_grids=grids)
        ck("tilize idx")
        idx_p.deallocate(True)
        idx_i = ttnn.typecast(idx_t, ttnn.int32, sub_core_grids=grids)
        ck("typecast idx i32")
        idx_t.deallocate(True)
        idx_g = tiny_gather(idx_i, dim=2)  # [1,1,32*ncols,B] tile int32
        ck("gather idx")
        idx_i.deallocate(True)

        # Winner position per user: 32 * winner_col (pad lanes are -inf; ties pick lowest column).
        vals_gu = ttnn.untilize(vals_g, use_multicore=True, sub_core_grids=grids)
        ck("untilize vals_g")
        vals_g.deallocate(True)
        g_pos = ttnn.argmax(
            vals_gu, dim=-1, keepdim=False, sub_core_grids=argmax_grid(vals_gu.shape[-1])
        )  # [1,1,B] RM u32
        ck("argmax g_pos")
        vals_gu.deallocate(True)
        g_pos = ttnn.reshape(g_pos, [1, 1, 1, batch], sub_core_grids=grids)
        g_pos_t = ttnn.tilize_with_val_padding(
            g_pos, [1, 1, ttnn.TILE_SIZE, batch], 0, sub_core_grids=grids
        )  # logical [1,1,1,B] tile
        ck("tilize g_pos")
        g_pos.deallocate(True)
        g_pos_f = ttnn.typecast(g_pos_t, ttnn.float32, sub_core_grids=grids)
        ck("typecast g_pos f32")
        g_pos_t.deallocate(True)

        # One-hot select of the winning column's local index, then add the column offset:
        # final = idx_g[32*c, u] + W * c, with W * c == (W / 32) * g_pos.
        #
        # TF32 hazard: the FPU eltwise mul/add unpack fp32 operands as TF32 (11-bit mantissa), so
        # any fp32 value > 2048 that isn't on the TF32 grid gets truncated (observed: index 3031
        # -> 3028). All values that flow through FPU mul/sum are therefore kept <= 2048: the index
        # is split into hi/lo bytes for the one-hot select, and the recombination (<< 8 | lo, plus
        # the column offset) is done with exact SFPU int32 shifts/adds. The colterm product
        # (W/32) * g_pos = W * c is exactly representable in TF32 for tile-aligned W (multiple of
        # 128 with a small mantissa), so that one multiply is safe.
        onehot = ttnn.eq(self._dist_argmax_iota, g_pos_f, sub_core_grids=grids)  # bcast H: [rows,B] vs [1,B]
        ck("eq onehot")
        hi_i = ttnn.bitwise_right_shift(idx_g, 8, sub_core_grids=grids)
        lo_i = ttnn.bitwise_and(idx_g, 255, sub_core_grids=grids)
        ck("split idx hi/lo")
        idx_g.deallocate(True)
        hi_f = ttnn.typecast(hi_i, ttnn.float32, sub_core_grids=grids)
        lo_f = ttnn.typecast(lo_i, ttnn.float32, sub_core_grids=grids)
        hi_i.deallocate(True)
        lo_i.deallocate(True)
        picked_hi = ttnn.multiply(onehot, hi_f, sub_core_grids=grids)
        picked_lo = ttnn.multiply(onehot, lo_f, sub_core_grids=grids)
        ck("mul picked")
        onehot.deallocate(True)
        hi_f.deallocate(True)
        lo_f.deallocate(True)
        sel_hi = ttnn.sum(
            picked_hi, dim=2, keepdim=True, compute_kernel_config=self._dist_argmax_fp32_ckc, sub_core_grids=grids
        )  # [1,1,1,B] fp32, < 2048
        sel_lo = ttnn.sum(
            picked_lo, dim=2, keepdim=True, compute_kernel_config=self._dist_argmax_fp32_ckc, sub_core_grids=grids
        )  # [1,1,1,B] fp32, < 256
        ck("sum sel")
        picked_hi.deallocate(True)
        picked_lo.deallocate(True)
        colterm = ttnn.multiply(g_pos_f, float(w // ttnn.TILE_SIZE), sub_core_grids=grids)
        ck("mul colterm")
        g_pos_f.deallocate(True)
        sel_hi_i = ttnn.typecast(sel_hi, ttnn.int32, sub_core_grids=grids)
        sel_lo_i = ttnn.typecast(sel_lo, ttnn.int32, sub_core_grids=grids)
        col_i = ttnn.typecast(colterm, ttnn.int32, sub_core_grids=grids)
        ck("typecast pieces i32")
        sel_hi.deallocate(True)
        sel_lo.deallocate(True)
        colterm.deallocate(True)
        hi_shifted = ttnn.bitwise_left_shift(sel_hi_i, 8, sub_core_grids=grids)
        sel_hi_i.deallocate(True)
        local_part = ttnn.add(hi_shifted, sel_lo_i, sub_core_grids=grids)  # int32 add: exact
        hi_shifted.deallocate(True)
        sel_lo_i.deallocate(True)
        final_i = ttnn.add(local_part, col_i, sub_core_grids=grids)
        ck("add final")
        local_part.deallocate(True)
        col_i.deallocate(True)
        final_u = ttnn.typecast(final_i, ttnn.uint32, sub_core_grids=grids)
        ck("typecast final u32")
        final_i.deallocate(True)
        tok = ttnn.untilize(final_u, use_multicore=True, sub_core_grids=grids)  # [1,1,1,B] RM u32
        ck("untilize final")
        final_u.deallocate(True)
        if tt_out_tok is None:
            # Match the single-op argmax output shape ([1,1,B], keepdim=False) expected downstream.
            return ttnn.reshape(tok, [1, 1, batch], sub_core_grids=grids)
        # Reshape to the caller buffer's logical shape (a view: same single RM page).
        tok = ttnn.reshape(tok, list(tt_out_tok.shape), sub_core_grids=grids)
        # Write the token into the caller's persistent buffer (the decode trace's frozen token
        # input in the galaxy demo flow). ttnn.copy can't be used here: its factory grids from
        # core (0,0), which under the split senders/worker sub-device manager lands the copy on
        # the senders sub-device, where dispatch runs it concurrently with the worker-sub-device
        # producers under trace replay (stale-token race, same failure mode as the unpinned
        # all_gather). A no-op bitwise_or pinned to the worker grid keeps ordering correct.
        out = ttnn.bitwise_or(tok, 0, output_tensor=tt_out_tok, sub_core_grids=grids)
        ck("copy to tt_out_tok")
        tok.deallocate(True)
        return out

    def reset_params(
        self,
        k,
        p,
        temp,
        enable_log_probs: bool | list[bool] = None,
        num_logprobs: int | list[int] = None,
        empty_slots: list[int] | None = None,
    ):
        """Update sampling parameters (k, p, temperature, logprobs) dynamically."""
        self._force_argmax_sampling = self._is_force_argmax_sampling(k, p, temp)
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "TTSampling reset params",
            force_argmax=self._force_argmax_sampling,
            empty_slots=_compact_debug_list(empty_slots),
            top_k=_compact_debug_list(k),
            top_p=_compact_debug_list(p),
            temperature=_compact_debug_list(temp),
            enable_log_probs=_compact_debug_list(enable_log_probs),
            num_logprobs=_compact_debug_list(num_logprobs),
            sampling_dp=self._sampling_dp,
        )
        if not self._force_argmax_sampling:
            # When _sampling_dp > 1, create multi-device host tensors so
            # copy_host_to_device_tensor writes per-row shards correctly.
            if self._sampling_dp > 1:
                mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._param_dims, mesh_shape=self.cluster_shape)
            else:
                mapper = None

            self.k_tensor_new = ttnn.from_torch(
                torch.tensor(k),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            self.p_tensor_new = ttnn.from_torch(
                torch.tensor(p),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            self.temp_tensor_new = ttnn.from_torch(
                torch.tensor(temp),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )

            ttnn.copy_host_to_device_tensor(self.k_tensor_new, self.k_tensor)
            ttnn.copy_host_to_device_tensor(self.p_tensor_new, self.p_tensor)
            ttnn.copy_host_to_device_tensor(self.temp_tensor_new, self.temp_tensor)

        self.log_probs_calculator.set_log_probs_mode(
            enable_log_probs, num_logprobs=num_logprobs, empty_slots=empty_slots
        )

    def forward(
        self,
        x: ttnn.Tensor,
        tt_out_tok: ttnn.Tensor = None,
    ):
        """
        Perform on-device sampling on logits tensor.
        The logits are sharded over the devices in the cluster.
        We perform local top-k on each device, then all-gather the top-k values and indices across all devices.
        We then convert the gathered values and indices to the appropriate format, add the device offsets to get the global vocabulary indices,
        and perform the actual sampling with top-k, top-p, and temperature.

        Args:
            x: Input logits tensor
            tt_out_tok: Optional output tensor to write results to

        Returns:
            Sampled token indices tensor
        """
        _log_sampling_debug(
            self._sampling_debug_enabled,
            "TTSampling forward",
            force_argmax=self._force_argmax_sampling,
            logits_shape=list(x.shape),
            tt_out_tok_shape=list(tt_out_tok.shape) if tt_out_tok is not None else None,
            max_top_k=self.max_top_k,
            multi_step_reduction=self.multi_step_reduction,
            sampling_dp=self._sampling_dp,
        )
        if self._force_argmax_sampling:
            logger.info("Forcing argmax sampling")
            # BH galaxy prefetcher (unfused-CCL) keeps a split senders/worker sub-device manager
            # loaded during decode. The vocab-trim ttnn.slice and the tail-mask slices auto-grid a
            # 32-core block from origin (0,0) (they don't honor sub_core_grids on the DRAM-interleaved
            # path), which spills into the uncovered senders-column tail -> "kernel group cores do not
            # match sub device cores" fatal. Skip the vocab trim on this path: the padded-vocab logits
            # are exactly 0 (the lm_head weight is zero-padded), so argmax over the full padded width
            # still selects the top valid token, and the following untilize/argmax are pinned to the
            # worker sub-core grid via _force_argmax_sub_core_grids.
            force_argmax_skip_vocab_trim = self._force_argmax_sub_core_grids is not None
            slice_valid_vocab = self._can_slice_valid_vocab_for_argmax() and not force_argmax_skip_vocab_trim
            if not slice_valid_vocab and not force_argmax_skip_vocab_trim:
                x = self._mask_invalid_vocab_logits(x)
            # Gather the output across all devices and untilize the tensor (for argmax)
            num_devices = self.mesh_device.get_num_devices()
            if num_devices > 1:
                cluster_axis = self._get_sampling_cluster_axis()
                num_links, topology = self._get_force_argmax_all_gather_config(cluster_axis)
                if self._use_distributed_argmax(x, tt_out_tok, cluster_axis):
                    tok = self._distributed_force_argmax(x, cluster_axis, topology, tt_out_tok=tt_out_tok)
                    self.tt_log_probs = None
                    return tok, self.tt_log_probs
                logger.debug(
                    f"Force argmax sampling all-gather: cluster_axis={cluster_axis}, "
                    f"num_links={num_links}, topology={topology}"
                )
                # NOTE: do NOT pass a persistent_output_buffer here. The all_gather_async factory
                # disables its barrier semaphore when persistent buffers are used, and the barrier
                # is what bounds cross-invocation skew: without it a fast device's next-invocation
                # writes/increments can reach a peer before that peer's reader has reset its
                # out_ready semaphore, leaving residual counts that make later waits pass early
                # (argmax then reads the previous step's logits). Under trace the fresh output
                # allocation is frozen at capture, so the address is stable anyway.
                barrier_accessor = getattr(
                    self.tt_ccl,
                    "get_sampling_barrier_semaphore_handle",
                    self.tt_ccl.get_and_cycle_barrier_semaphore_handle,
                )
                # Pin the gather's worker cores to the same sub-device as the downstream
                # untilize/argmax. Without this, all_gather_async defaults to sub-device 0, which
                # under the galaxy prefetcher decode manager is the prefetcher/senders sub-device.
                # Dispatch only serializes programs within a sub-device, so a gather on sub-device 0
                # runs concurrently with the untilize on the worker sub-device: under trace replay
                # (back-to-back go signals) the untilize/argmax read the gather output buffer before
                # this step's writes land and return the previous step's argmax (stale tokens).
                # Eager mode masks this because per-op host dispatch latency exceeds the gather time.
                ag_sub_device_id = getattr(self.tt_ccl, "worker_sub_device_id", None)
                x = ttnn.experimental.all_gather_async(
                    x,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                    num_links=num_links,
                    memory_config=x.memory_config(),
                    cluster_axis=cluster_axis,
                    topology=topology,
                    barrier_semaphore=barrier_accessor(cluster_axis),
                    chunks_per_sync=self.argmax_chunks_per_sync,
                    num_workers_per_link=self.argmax_num_workers_per_link,
                    num_buffers_per_channel=2,
                    subdevice_id=ag_sub_device_id,
                )
                if os.environ.get("TT_SAMPLING_DEBUG_ADDR") == "1":
                    logger.info(f"force-argmax gather out addr {x.buffer_address():#x}")
                if os.environ.get("QWEN_SAMPLING_KEEP_GATHERED") == "1":
                    # Keep a handle to the gathered logits so tests can read the trace-resident
                    # buffer back after replay and diff its slots against host-composed logits.
                    self.debug_gathered = x
            if slice_valid_vocab:
                x = self._slice_valid_vocab_for_argmax(x)
            x_untilized = ttnn.untilize(x, use_multicore=True, sub_core_grids=self._force_argmax_sub_core_grids)
            if os.environ.get("QWEN_SAMPLING_KEEP_GATHERED") == "1":
                self.debug_untilized = x_untilized
            tt_out_tok = ttnn.argmax(
                x_untilized,
                dim=-1,
                output_tensor=tt_out_tok,
                keepdim=False,
                sub_core_grids=self._force_argmax_sub_core_grids,
            )
            # Argmax path: logprobs not supported (force-argmax is disabled
            # when logprobs are enabled via format_sampling_params guard).
            self.tt_log_probs = None
            return tt_out_tok, self.tt_log_probs

        # Convert to bfloat16 for top-k operations (typecast is no-op if already bfloat16)
        x_bf16 = ttnn.typecast(x, dtype=ttnn.bfloat16, sub_core_grids=self.sub_core_grids)
        x_bf16 = self._mask_invalid_vocab_logits(x_bf16)

        if self.multi_step_reduction:
            x_bf16_list = ttnn.split(x_bf16, x_bf16.shape[-1] // 2, dim=3)
            indices_tensor_list = ttnn.split(self.tt_indices_tensor, self.tt_indices_tensor.shape[-1] // 2, dim=3)
            topk_values_list = []
            topk_indices_list = []

            for i in range(len(x_bf16_list)):
                topk_values, topk_indices = ttnn.topk(
                    x_bf16_list[i],
                    k=self.max_top_k,
                    dim=-1,
                    sub_core_grids=self.sub_core_grid_topk,
                    indices_tensor=indices_tensor_list[i],
                )
                topk_values_list.append(topk_values)
                topk_indices_list.append(topk_indices)
                x_bf16_list[i].deallocate()
                indices_tensor_list[i].deallocate()

            topk_values_gathered_bf16_interleaved = ttnn.concat(topk_values_list, dim=3)
            topk_indices_gathered = ttnn.concat(topk_indices_list, dim=3)

            for i in range(len(topk_indices_list)):
                ttnn.deallocate(topk_values_list[i])
                ttnn.deallocate(topk_indices_list[i])

        else:
            # apply padding to the input tensor if needed
            # if number is not power of 2, pad to upper power of 2
            # pad only last dimension with float::min value to upper_power_of_2
            # This is necessary to use full optimization in the topk operation.
            if self.pad_to_power_of_2 and not is_power_of_2(x_bf16.shape[-1]):
                padded_value = upper_power_of_2(x_bf16.shape[-1])
                x_bf16 = ttnn.pad(
                    x_bf16,
                    [(0, 0), (0, 0), (0, 0), (0, padded_value - x_bf16.shape[-1])],
                    value=-sys.float_info.max,
                    sub_core_grids=self.sub_core_grids,
                )
            # Perform local top-k on each device
            topk_values, topk_indices = ttnn.topk(
                x_bf16,
                k=self.max_top_k,
                dim=-1,
                sub_core_grids=self.sub_core_grid_topk,
                indices_tensor=self.tt_indices_tensor,
            )

            # For 1D meshes use `cluster_axis=None`. For 2D meshes, use the configured gather axis.
            sampling_cluster_axis = self._get_sampling_cluster_axis()

            # Gather top-k values across all devices
            topk_values_gathered = self._perform_all_gather(
                topk_values,
                dim=3,
                cluster_axis=sampling_cluster_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=self.num_gather_links,
                buffer_key="SAMPLING_VALUES",
            )

            ttnn.deallocate(topk_values)

            # Convert gathered values to appropriate format
            if self.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
                topk_values_gathered_bf16 = ttnn.to_memory_config(
                    topk_values_gathered,
                    memory_config=self.sampling_memory_config,
                    dtype=ttnn.bfloat16,
                )
                topk_values_gathered_bf16_interleaved = ttnn.to_memory_config(
                    topk_values_gathered_bf16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(topk_values_gathered_bf16)
            else:
                topk_values_gathered_bf16_interleaved = topk_values_gathered

            # Gather top-k indices across all devices
            topk_indices_gathered = self._perform_all_gather(
                topk_indices,
                dim=3,
                cluster_axis=sampling_cluster_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=self.num_gather_links,
                buffer_key="SAMPLING_INDICES",
                dtype=ttnn.uint16,
            )
            ttnn.deallocate(topk_indices)

        # Convert indices to appropriate data types

        topk_indices_gathered_int32 = ttnn.typecast(
            topk_indices_gathered, dtype=ttnn.int32, sub_core_grids=self.sub_core_grids
        )

        if self.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            topk_indices_gathered_int32_sharded = ttnn.to_memory_config(
                topk_indices_gathered_int32, self.sampling_memory_config
            )
            ttnn.deallocate(topk_indices_gathered_int32)
        else:
            topk_indices_gathered_int32_sharded = topk_indices_gathered_int32

        # Add device offsets to get global vocabulary indices
        topk_global_indices = ttnn.add(
            self.tt_indices_device_offsets,
            topk_indices_gathered_int32_sharded,
            dtype=ttnn.uint32,
            memory_config=self.sampling_memory_config,
        )

        ttnn.deallocate(topk_indices_gathered_int32_sharded)

        topk_global_indices_interleaved = ttnn.to_memory_config(topk_global_indices, ttnn.DRAM_MEMORY_CONFIG)

        # Untilize indices for sampling operation
        topk_global_indices_interleaved_untilised = ttnn.untilize(
            topk_global_indices_interleaved, use_multicore=True, sub_core_grids=self.sub_core_grids
        )
        ttnn.manual_seed(
            seeds=self.seeds_tt_tensor,
            user_ids=self.user_ids_tt_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
        )
        # Perform the actual sampling with top-k, top-p, and temperature
        tt_out_tok = ttnn.sampling(
            topk_values_gathered_bf16_interleaved,
            topk_global_indices_interleaved_untilised,
            k=self.k_tensor,
            p=self.p_tensor,
            temp=self.temp_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
            output_tensor=tt_out_tok,
        )

        # Compute logprobs if enabled
        if self.log_probs_calculator.enable_log_probs and self.log_probs_calculator._use_topk_logprobs:
            # New path: top-K logprobs for gpt-oss-120b
            self.tt_log_probs = self.log_probs_calculator.calculate_topk_log_probs(
                logits_tensor=x,
                topk_values=topk_values_gathered_bf16_interleaved,
                topk_global_indices=topk_global_indices_interleaved,
                sub_core_grid_topk=self.sub_core_grid_topk,
            )
        elif self.log_probs_calculator.enable_log_probs:
            # Old path: single sampled-token logprob
            self.tt_log_probs = self.log_probs_calculator.calculate_log_probs(x, tt_out_tok)
        else:
            self.tt_log_probs = None

        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok, self.tt_log_probs
