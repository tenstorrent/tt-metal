# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
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
        # Persistent per-user ARGMAX mask [1,1,N,1] (1.0 where k==1), distributed like k_tensor. Used by
        # _adjust_values_for_tiebreak to boost the lowest-index tied-max for greedy users only. Built
        # host-side and kept in sync in reset_sampling_params (an on-device reshape of the [N] k_tensor
        # to [1,1,N,1] is not sub-device-safe). float32 TILE so it broadcasts over the candidate width.
        self._greedy_col = ttnn.from_torch(
            (torch.as_tensor(k).reshape(1, 1, -1, 1) == 1).to(torch.float32),
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=self._greedy_col_dims(), mesh_shape=self.cluster_shape
            ),
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
        # The compact tail-mask path slices off the valid region, masks the tail,
        # and concats back. That reassembly is only safe when sampling runs on the
        # full compute grid: on a sampling sub-core grid (e.g. Llama TG) the slice
        # and concat are placed on different cores and the columns are stitched back
        # incorrectly, so padded-vocab tokens (id >= vocab_size) survive into top-k
        # and get sampled -> garbage / non-deterministic output. Fall back to the
        # plain full-width additive mask (a single elementwise add, no reassembly)
        # whenever a sub-core grid is in use.
        tail_mask = (
            build_tail_invalid_vocab_mask(
                self.vocab_size,
                self.padded_vocab_size,
                self.max_batch_size,
                self.cluster_shape,
                self.sampling_all_gather_axis,
                tile_size=ttnn.TILE_SIZE,
            )
            if self.sub_core_grids is None
            else None
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
            # Match the slices above: run concat on the sampling sub-device cores,
            # otherwise the concat program is placed on the full Tensix grid and
            # fails with "Kernel group cores do not match sub device cores"
            # (TT_FATAL num_intersections == num_cores).
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

            # Keep the greedy tie-break mask (1.0 where k==1) in sync with k, distributed like k_tensor.
            self._greedy_col_new = ttnn.from_torch(
                (torch.tensor(k).reshape(1, 1, -1, 1) == 1).to(torch.float32),
                device=None,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=(
                    ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=self._greedy_col_dims(), mesh_shape=self.cluster_shape
                    )
                    if self._sampling_dp > 1
                    else None
                ),
            )
            ttnn.copy_host_to_device_tensor(self._greedy_col_new, self._greedy_col)

        self.log_probs_calculator.set_log_probs_mode(
            enable_log_probs, num_logprobs=num_logprobs, empty_slots=empty_slots
        )

    def _greedy_col_dims(self):
        """Map the 1-D k_tensor shard dims (self._param_dims, batch on dim0) to the [1,1,N,1] greedy
        mask's dims (batch on dim2), so self._greedy_col is distributed exactly like self.k_tensor."""
        return tuple(2 if d == 0 else d for d in self._param_dims)

    def _adjust_values_for_tiebreak(self, gathered_values, gathered_global_indices):
        """Return gathered_values with, for ARGMAX users (k==1) ONLY, the single lowest-GLOBAL-INDEX
        candidate among the tied maxima boosted by DELTA, so ttnn.sampling's argmax selects it
        deterministically. This fixes ttnn.sampling's array-position tie-break (it breaks exact value
        ties by all_gather/device order, which varies run-to-run/slot-to-slot and flips the greedy
        token) by correcting the sampling INPUT in the TILE domain -- avoiding an in-place write into
        the ROW_MAJOR output buffer, which NO ttnn op supports on a restricted sub-device. Random
        users (k>1) get boost==0 => their values are bit-identical => their sampling is byte-for-byte
        unchanged. All ops honor self.sub_core_grids.

        is_winner = (value == rowmax) AND (global_index == lowest_index_among_maxima)  # exactly one candidate
            lowest_index_among_maxima = min(global_index + (rowmax - value)*LARGE)     # == idx at maxima, huge else
        Validated on a restricted active sub-device by
        tests/ttnn/unit_tests/operations/reduce/test_tiebreak_input_adjust.py.
        """
        if getattr(self, "_greedy_col", None) is None:
            return gathered_values
        try:
            scg = self.sub_core_grids
            BIG = 1.0e9  # >> max vocab index; EXACT binary offset (no bf16 (maxv-value) magnitude dependence)
            DELTA = 1.0  # >> bf16 tie granularity => the chosen tied-max becomes the strict argmax
            maxv = ttnn.max(gathered_values, dim=3, keepdim=True, sub_core_grids=scg)  # [1,1,B,1] bf16
            idx_f = ttnn.typecast(gathered_global_indices, ttnn.float32, sub_core_grids=scg)
            is_max = ttnn.eq(gathered_values, maxv, sub_core_grids=scg)  # 1.0 at the (tied) maxima, exact
            not_max = ttnn.lt(gathered_values, maxv, sub_core_grids=scg)  # 1.0 strictly below max
            # lowest global index among the maxima: push non-maxima up by BIG (exact, robust), then min.
            masked_idx = ttnn.add(idx_f, ttnn.multiply(not_max, BIG, sub_core_grids=scg), sub_core_grids=scg)
            greedy_i = ttnn.min(masked_idx, dim=3, keepdim=True, sub_core_grids=scg)  # [1,1,B,1] f32
            is_lowidx = ttnn.eq(idx_f, greedy_i, sub_core_grids=scg)  # broadcast over W
            is_winner = ttnn.multiply(is_max, is_lowidx, sub_core_grids=scg)  # 1.0 at exactly one candidate
            # gate by k==1 (self._greedy_col [1,1,B,1]); random users get boost 0 => values unchanged
            boost = ttnn.multiply(
                ttnn.multiply(is_winner, self._greedy_col, sub_core_grids=scg), DELTA, sub_core_grids=scg
            )
            return ttnn.add(
                gathered_values, ttnn.typecast(boost, ttnn.bfloat16, sub_core_grids=scg), sub_core_grids=scg
            )
        except Exception as e:
            if not getattr(self, "_tiebreak_logged", False):
                self._tiebreak_logged = True
                logger.error(f"[TIEBREAK_FIX] disabled (using original values) due to: {e}")
            return gathered_values

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
            slice_valid_vocab = self._can_slice_valid_vocab_for_argmax()
            if not slice_valid_vocab:
                x = self._mask_invalid_vocab_logits(x)
            # Gather the output across all devices and untilize the tensor (for argmax)
            num_devices = self.mesh_device.get_num_devices()
            if num_devices > 1:
                cluster_axis = self._get_sampling_cluster_axis()
                num_links, topology = self._get_force_argmax_all_gather_config(cluster_axis)
                logger.debug(
                    f"Force argmax sampling all-gather: cluster_axis={cluster_axis}, "
                    f"num_links={num_links}, topology={topology}"
                )
                x = ttnn.experimental.all_gather_async(
                    x,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                    num_links=num_links,
                    memory_config=x.memory_config(),
                    cluster_axis=cluster_axis,
                    topology=topology,
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    chunks_per_sync=self.argmax_chunks_per_sync,
                    num_workers_per_link=self.argmax_num_workers_per_link,
                    num_buffers_per_channel=2,
                )
            if slice_valid_vocab:
                x = self._slice_valid_vocab_for_argmax(x)
            x_untilized = ttnn.untilize(x, use_multicore=True)
            tt_out_tok = ttnn.argmax(
                x_untilized,
                dim=-1,
                output_tensor=tt_out_tok,
                keepdim=False,
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
        # Perform the actual sampling with top-k, top-p, and temperature.
        # Fix ttnn.sampling's ARRAY-POSITION tie-break (it flips greedy tokens on exact bf16 value ties
        # because the tie is broken by all_gather/device order): for argmax users (k==1) only, boost the
        # single lowest-GLOBAL-INDEX tied-max in the sampling INPUT so argmax picks it deterministically.
        # Random users are byte-for-byte unchanged. Correcting the INPUT (not the RM output buffer) is
        # required: no ttnn op writes an interleaved ROW_MAJOR tensor in-place on a restricted sub-device.
        sampling_values = self._adjust_values_for_tiebreak(
            topk_values_gathered_bf16_interleaved, topk_global_indices_interleaved
        )
        tt_out_tok = ttnn.sampling(
            sampling_values,
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

        if sampling_values is not topk_values_gathered_bf16_interleaved:
            ttnn.deallocate(sampling_values)
        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok, self.tt_log_probs
