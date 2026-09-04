# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
import sys

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.sampling._utils import is_default_value, is_power_of_2, upper_power_of_2
from models.common.sampling.tt_log_probs import LogProbsCalculator
from models.common.sampling.vocab_padding import (
    build_invalid_vocab_mask,
    build_tail_invalid_vocab_mask,
    get_vocab_shard_dims,
)

# The multi-core ttnn.topk factory's admission rules, mirrored from
# ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp
# (select_program_factory) and topk_constants.hpp. A reduced width that fails any of
# them falls to a single-core kernel whose cost scales with the width, which on a large
# vocab shard can cost more than an entire transformer decode step. See
# TTSampling._topk_multicore_split.
_TOPK_MULTICORE_MAX_WIDTH = 65535  # multi-core indices are UInt16
_TOPK_MULTICORE_MIN_WIDTH = 8192  # ttnn::prim::constants::multi_core_min_width

# Greedy tie-break boost (see TTSampling._adjust_values_for_tiebreak).
# bfloat16 keeps an 8-bit mantissa, so the gap to the next representable value at magnitude |x| is
# between |x| * 2^-8 and |x| * 2^-7. Scaling the tied maximum by 2^-6 is therefore at least 2 ULP at
# every magnitude, whereas a fixed constant is not: 1.0 is below one ULP once |logit| >= 256 (bf16
# spacing there is 2.0), so 256 + 1 rounds straight back to 256 and the tie survives. Multiplying by
# a power of two is exact in bf16, so the boost is reproducible.
TIEBREAK_DELTA_SCALE = 2**-6
# Floor so the boost stays strictly positive when the tied maximum is exactly 0.0. Still a normal
# bf16 number (smallest normal ~1.18e-38) and orders of magnitude below one ULP of any real logit,
# so it never perturbs a non-zero maximum.
TIEBREAK_DELTA_FLOOR = 1e-30
# Added to the global index of every non-maximum candidate so the row min over the masked indices can
# only ever land on a tied maximum. A power of two, so the mask that carries it (built in bfloat16)
# holds it exactly; larger than any padded vocabulary size, which __init__ asserts; and small enough
# that sentinel + index cannot overflow the int32 the min reduce runs in.
TIEBREAK_INDEX_SENTINEL = 2**24


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
        # ttnn.topk rejects stable=True outright on any architecture whose LLK lacks the stable bitonic
        # network -- only Wormhole B0 and Blackhole implement it -- so ask for it just where it exists
        # instead of taking a TT_FATAL everywhere else. Requesting it is best effort regardless
        # (tenstorrent/tt-metal#33492); _adjust_values_for_tiebreak is what guarantees the greedy pick,
        # so falling back to the default network costs correctness nothing.
        self._topk_stable = ttnn.device.is_wormhole_b0(mesh_device) or ttnn.device.is_blackhole(mesh_device)
        # Multi-step reduction is supported only on single device
        self.multi_step_reduction = list(mesh_device.shape) == [1, 1]
        self.tt_ccl = tt_ccl
        self._line_all_gather = getattr(self.tt_ccl, "line_all_gather", None)
        self._line_all_gather_supports_buffer_key = False
        self._line_all_gather_supports_dtype = False
        self.pad_to_power_of_2 = getattr(args, "pad_logits_to_power_of_2", False)
        # Opt-in, and off by default so no existing model changes behaviour: pad the
        # local vocab shard to a power of two and then *split* it, so each ttnn.topk
        # call reaches the multi-core factory instead of a single-core kernel whose
        # cost scales with vocab width. See _topk_multicore_split for the measurements
        # and for why padding alone is not enough. Implies pad_to_power_of_2.
        self.topk_split_to_power_of_2 = getattr(args, "topk_split_to_power_of_2", False)
        if self.topk_split_to_power_of_2:
            self.pad_to_power_of_2 = True
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
        # The tie-break sentinel has to outrank every real global index. Vocabularies this large are
        # far beyond anything shipped, but exceeding it would corrupt the greedy token silently rather
        # than fail, so check it at construction. Raise rather than assert: this guards a correctness
        # invariant and must survive python -O.
        if self.padded_vocab_size > TIEBREAK_INDEX_SENTINEL:
            raise ValueError(
                f"padded_vocab_size {self.padded_vocab_size} exceeds the greedy tie-break sentinel "
                f"{TIEBREAK_INDEX_SENTINEL}; raise TIEBREAK_INDEX_SENTINEL (keeping it a power of two)"
            )
        # Persistent per-user ARGMAX mask [1,1,N,1] (1.0 where k==1), distributed like k_tensor. Used by
        # _adjust_values_for_tiebreak to boost the lowest-index tied-max for greedy users only. Built
        # host-side and kept in sync in reset_sampling_params (an on-device reshape of the [N] k_tensor
        # to [1,1,N,1] is not sub-device-safe). bfloat16 TILE so it broadcasts over the candidate width
        # and multiplies the (bfloat16) winner mask without a dtype mix; 0.0/1.0 are exact in bf16.
        self._greedy_col = ttnn.from_torch(
            (torch.as_tensor(k).reshape(1, 1, -1, 1) == 1).to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
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
            mesh_mapper=(
                ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._param_dims, mesh_shape=self.cluster_shape)
                if self._sampling_dp > 1
                else None
            ),
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
        indices_device_offsets, indices_tensor_torch, padded_per_device = self._indices_host_tensors()
        self.tt_indices_device_offsets = ttnn.from_torch(
            indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

        # The multi-core top-k factory uses 16-bit indices.  A split local shard can
        # itself be wider than uint16 (P150x2 pads 101056 logits to 131072), so feeding
        # slices of the full uint32 identity tensor to the otherwise-legal 32768-wide
        # calls corrupts the returned indices. Rank every piece with the same uint16
        # piece-relative identity instead. ``tt_indices_device_offsets`` already folds
        # the piece base into each candidate block, so the ordinary post-gather add
        # restores exact global vocabulary ids without another decode-time operation.
        self.tt_topk_piece_indices = None
        if self.topk_pieces > 1:
            piece_indices_torch = self._topk_piece_indices_host(indices_tensor_torch.shape[-1])
            self.tt_topk_piece_indices = ttnn.from_torch(
                piece_indices_torch,
                dtype=ttnn.uint16,
                layout=ttnn.Layout.TILE,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def _indices_host_tensors(self):
        """Build shape-matched local indices and global-shard offsets on host.

        Single-device sampling performs two top-k reductions over the two halves
        of one logits tensor. Those halves are logical sampling shards, but they
        are not the per-device shards used by ``_topk_multicore_split``. Each
        half therefore contributes exactly ``max_top_k`` candidates and needs a
        full-width repeated local-index vector before the second half's offset
        is added.
        """
        num_devices_in_mesh = self._get_num_sampling_shards()
        if self.padded_vocab_size % num_devices_in_mesh:
            raise ValueError(
                f"padded_vocab_size {self.padded_vocab_size} is not divisible by "
                f"{num_devices_in_mesh} sampling shards"
            )
        # padded_per_device: tile-aligned width matching actual logit tensors (for indices tensor)
        padded_per_device = self.padded_vocab_size // num_devices_in_mesh
        # How many candidates each device contributes to the gather. Normally max_top_k;
        # the multi-core topk split contributes max_top_k *per piece* (see
        # _topk_multicore_split for why they cannot be reduced back on device), so the
        # offset stride has to follow it or the gathered indices get the wrong device's
        # offset added.
        self.topk_pieces = 1 if self.multi_step_reduction else self._topk_piece_count(padded_per_device)
        self.candidates_per_device = self.max_top_k * self.topk_pieces
        # Create local indices tensor for top-k operations (must match logit width)
        local_indices = torch.arange(padded_per_device, dtype=torch.int32)
        if self.multi_step_reduction:
            local_indices = local_indices.repeat(num_devices_in_mesh)

        # pad to power of 2 if needed
        if not self.multi_step_reduction and self.pad_to_power_of_2 and not is_power_of_2(local_indices.shape[-1]):
            padded_value = upper_power_of_2(local_indices.shape[-1])
            local_indices = torch.nn.functional.pad(
                local_indices,
                (0, padded_value - local_indices.shape[-1]),
                mode="constant",
                value=-1,
            )

        # Candidate order is device-major after all-gather and piece-major within
        # each device after concat. Fold both bases into one persistent int32 tensor:
        #
        #   global_id = device * unpadded_shard_width
        #             + piece * padded_piece_width
        #             + uint16_piece_relative_id
        #
        # For the single-device two-half reduction, ``num_devices_in_mesh`` is the
        # number of logical halves and ``topk_pieces`` is forced to one, preserving
        # its existing [0, half_width] offsets.
        padded_topk_width = local_indices.shape[-1] // (num_devices_in_mesh if self.multi_step_reduction else 1)
        if padded_topk_width % self.topk_pieces:
            raise ValueError(f"top-k width {padded_topk_width} is not divisible by {self.topk_pieces} pieces")
        per_piece = padded_topk_width // self.topk_pieces
        piece_offsets = (torch.arange(self.topk_pieces, dtype=torch.int64) * per_piece).repeat_interleave(
            self.max_top_k
        )
        device_bases = torch.arange(num_devices_in_mesh, dtype=torch.int64) * padded_per_device
        offsets = (device_bases[:, None] + piece_offsets[None, :]).reshape(-1)
        indices_device_offsets = offsets.reshape(1, 1, 1, -1).expand(1, 1, self.max_batch_size, -1)

        indices_tensor_torch = local_indices.reshape(1, 1, 1, -1).expand(1, 1, self.max_batch_size, -1)
        return indices_device_offsets.contiguous(), indices_tensor_torch.contiguous(), padded_per_device

    def _topk_piece_indices_host(self, padded_topk_width: int):
        """Build the uint16-safe identity tensor shared by every split top-k call."""
        if self.topk_pieces <= 1:
            raise ValueError("piece-relative top-k indices require more than one piece")
        if padded_topk_width % self.topk_pieces:
            raise ValueError(f"top-k width {padded_topk_width} is not divisible by {self.topk_pieces} pieces")
        per_piece = padded_topk_width // self.topk_pieces
        if per_piece > torch.iinfo(torch.uint16).max:
            raise ValueError(f"top-k piece width {per_piece} exceeds the uint16 index limit")
        piece_indices = torch.arange(per_piece, dtype=torch.int32)
        return piece_indices.reshape(1, 1, 1, -1).expand(1, 1, self.max_batch_size, -1).contiguous()

    def _create_invalid_vocab_mask(self):
        self.tt_invalid_vocab_mask = None
        self.tt_invalid_vocab_tail_mask = None
        self._invalid_vocab_tail_width = 0

        vocab_shard_dims = get_vocab_shard_dims(self.cluster_shape, self.sampling_all_gather_axis)
        # The compact tail-mask path masks only the padded tail, but it has to slice the
        # logits and concat them back. ttnn.concat only honours sub_core_grids when the
        # input is unsharded and the output is interleaved; otherwise it falls through to
        # the "massaged" untilize/transpose path, which is invoked without sub_core_grids
        # and so runs on the full Tensix grid. The sampling logits are width-sharded
        # (DECODE_LOGITS_MEMCFG), so on a sampling sub-core grid the concat escapes the
        # sub-device. Use the plain full-width additive mask (one elementwise add, no
        # reassembly) whenever a sub-core grid is in use.
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

    def _topk_piece_count(self, padded_per_device: int) -> int:
        """How many pieces the local shard is split into for ``ttnn.topk``.

        1 means no split. Mirrors the width the split actually operates on: the padded
        shard when ``pad_to_power_of_2`` applies, otherwise the shard itself.
        """
        if not self.topk_split_to_power_of_2:
            return 1
        width = padded_per_device
        if not is_power_of_2(width):
            width = upper_power_of_2(width)
        pieces = 1
        while width // pieces >= _TOPK_MULTICORE_MAX_WIDTH:
            pieces *= 2
        per_piece = width // pieces
        if pieces == 1 or not is_power_of_2(per_piece) or per_piece < _TOPK_MULTICORE_MIN_WIDTH:
            return 1
        return pieces

    def _topk_multicore_split(self, x_bf16):
        """Local top-k, split so each ``ttnn.topk`` call reaches the multi-core factory.

        ``TopKDeviceOperation::select_program_factory`` takes the multi-core factory
        only when the reduced width is **a power of two**, **below the uint16 index
        bound** (65535) and **at least** ``multi_core_min_width`` (8192), with
        ``k <= 64``.  A vocab shard that is none of the first three falls to a
        single-core kernel whose cost scales with width, and on a large vocab that
        kernel can dominate an entire decode step.

        Measured on 4 x Blackhole at the real decode payload ``[1, 1, 32, W]``, k=32
        (``doc/full_model/bench/topk_geometry_probe.py`` in the Muse-Glimmer-30B
        autoport, ``topk_geometry_probe.json``):

        The measured widths, and which reach the multi-core factory, are in the
        Muse-Glimmer-30B autoport's ``doc/full_model/topk_geometry_probe.json``; they are
        not repeated here, because a table of timings in a docstring goes stale silently.

        So padding a 50688-wide shard to the next power of two is *not* enough --
        65536 is over the uint16 bound and stays single-core.  Padding and then
        splitting into halves of 32768 satisfies every condition, for a 33x
        reduction on the op.

        This returns ``pieces * max_top_k`` candidates per device rather than
        ``max_top_k``, which is why :attr:`candidates_per_device` exists and why the
        device-offset tensor is built from it.  Reducing the pieces back to
        ``max_top_k`` on device with a second ``ttnn.topk`` was tried first and is
        **wrong**: a 64-wide reduction is below ``multi_core_min_width``, and the
        single-core factory *ignores* ``indices_tensor`` and returns positions into its
        input.  The unsplit path only survives that because its indices tensor is the
        identity map, where position and index coincide; a second stage over
        already-permuted indices does not have that property, and it silently returned
        candidate positions 0..63 as if they were vocab ids
        (``doc/full_model/bench/topk_split_correctness.py``,
        ``topk_split_correctness.json``: values correct, indices ``[0, 32, 2, 1, ...]``).
        Widening the candidate set keeps every ``ttnn.topk`` call in the multi-core
        regime and keeps the index mapping exact.
        """
        width = int(x_bf16.shape[-1])
        pieces = self.topk_pieces
        if pieces <= 1:
            # The shard cannot be split into the multi-core regime, so take the op as it
            # is. ``candidates_per_device`` is ``max_top_k * 1`` in this case, so the
            # device-offset stride already matches this branch's output.
            return ttnn.topk(
                x_bf16,
                k=self.max_top_k,
                dim=-1,
                sub_core_grids=self.sub_core_grid_topk,
                indices_tensor=self.tt_indices_tensor,
                stable=self._topk_stable,
            )
        per_piece = width // pieces
        if width % pieces or not is_power_of_2(per_piece):
            # ``topk_pieces`` was derived at build time from the padded shard width, and
            # ``candidates_per_device`` -- which sizes the device-offset tensor -- was
            # derived from it. If the runtime width disagrees, silently falling back to a
            # single call would return max_top_k candidates where the offsets expect
            # pieces * max_top_k, and the offset add would either fail on shape or, worse,
            # apply the wrong device's offset. Say so instead.
            raise ValueError(
                f"topk split expected a width divisible into {pieces} power-of-two pieces, "
                f"got {width} (per piece {per_piece}); candidates_per_device="
                f"{self.candidates_per_device} was built for the split"
            )

        if self.tt_topk_piece_indices is None:
            raise RuntimeError(f"missing piece-relative indices for {pieces}-piece top-k")

        x_list = ttnn.split(x_bf16, per_piece, dim=3)
        values_parts = []
        indices_parts = []
        for x_piece in x_list:
            values, indices = ttnn.topk(
                x_piece,
                k=self.max_top_k,
                dim=-1,
                sub_core_grids=self.sub_core_grid_topk,
                indices_tensor=self.tt_topk_piece_indices,
                stable=self._topk_stable,
            )
            values_parts.append(values)
            indices_parts.append(indices)
            ttnn.deallocate(x_piece)

        topk_values = ttnn.concat(values_parts, dim=3)
        topk_indices = ttnn.concat(indices_parts, dim=3)
        for values, indices in zip(values_parts, indices_parts):
            ttnn.deallocate(values)
            ttnn.deallocate(indices)
        return topk_values, topk_indices

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
                (torch.tensor(k).reshape(1, 1, -1, 1) == 1).to(torch.bfloat16),
                device=None,
                dtype=ttnn.bfloat16,
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
        candidate among the tied maxima boosted just past the tie, so ttnn.sampling's argmax selects
        it deterministically. This fixes ttnn.sampling's array-position tie-break (it breaks exact
        value ties by all_gather/device order, which varies run-to-run/slot-to-slot and flips the
        greedy token) by correcting the sampling INPUT in the TILE domain -- avoiding an in-place
        write into the ROW_MAJOR output buffer, which NO ttnn op supports on a restricted
        sub-device. Random users (k>1) get boost==0 => their values are bit-identical => their
        sampling is byte-for-byte unchanged. All ops honor self.sub_core_grids.

        WORKAROUND for tenstorrent/tt-metal#33492 (stable top-k is unreliable). Remove this method,
        `_greedy_col` and `_greedy_col_dims` once that issue is fixed and validated on device.

        With a working stable top-k this pass is redundant, because candidate position and global
        token id are ordered the same way BY CONSTRUCTION: the gathered buffer is laid out as one
        contiguous block per device, and `_create_indices_tensors` derives each global id as
        `local_topk_index + device_id * padded_per_device` from that same layout. Across blocks the
        two orders therefore agree unconditionally. WITHIN a block they agree only if the per-device
        top-k emitted its tied maxima in ascending local-index order -- i.e. only if `stable=True`
        actually works. It currently may not (#33492: `ttnn.sort` rejects `stable=True` outright, the
        LLK top-k test skips every stable case, and this tree still carries the double-SFPSWAP
        scheme rather than the index-aware comparator from tt-llk#1340), which is why this exists.

        NOTE, and it matters if #33492 ever closes and someone deletes this pass: the
        "ordered the same way BY CONSTRUCTION" argument above assumes each device block is one
        top-k over one contiguous shard. With `topk_split_to_power_of_2` a block is
        `[piece0 top-k | piece1 top-k]`, so position order does NOT track global-id order
        within a block even with a perfectly stable top-k -- piece1's ids are all larger than
        piece0's, but the two runs interleave by value. This pass stays correct either way,
        because it takes an explicit `min` over the *global* indices rather than relying on
        layout; the redundancy argument is what stops being true.

        KNOWN LIMITATION: this picks the lowest global id among the GATHERED candidates. Note the
        unit this is measured in changes under `topk_split_to_power_of_2`: the gathered set is then
        `pieces * max_top_k` per device, and the "more maxima than we keep" condition applies per
        *piece* (32768 columns here) rather than per whole shard. If a single
        device shard holds more than `max_top_k` (32) maxima tied at the same value, its top-k drops
        all but 32 of them by the same unreliable network, so the true lowest-id token may never
        reach the gathered set and this pass cannot recover it. Fixing #33492 is the real fix; this
        only narrows the window.

        is_winner = (value == rowmax) AND (global_index == lowest_index_among_maxima)  # exactly one candidate
            lowest_index_among_maxima = min(global_index + not_max * SENTINEL)          # == idx at maxima, huge else

        The value half runs in bfloat16 and the index half in int32, and neither dtype is incidental --
        see the comments on each block. Validated on a restricted active sub-device by
        tests/ttnn/unit_tests/operations/reduce/test_tiebreak_input_adjust.py.
        """
        scg = self.sub_core_grids
        # Every intermediate is deallocated as soon as its last reader is issued: this runs once per
        # decode step, so leaving them to Python refcounting would hold ~9 extra buffers per step.

        # Value domain, bfloat16: gathered_values is bf16, so the max reduce and the comparisons
        # against its result are exact -- bf16 survives the FPU's source-register truncation intact.
        maxv = ttnn.max(gathered_values, dim=3, keepdim=True, sub_core_grids=scg)  # [1,1,B,1] bf16
        is_max = ttnn.eq(gathered_values, maxv, sub_core_grids=scg)  # 1.0 at the (tied) maxima, exact
        not_max = ttnn.lt(gathered_values, maxv, sub_core_grids=scg)  # 1.0 strictly below max

        # Per-row boost, >= 2 bf16 ULP of that row's maximum whatever its magnitude or sign.
        abs_max = ttnn.abs(maxv, sub_core_grids=scg)
        ttnn.deallocate(maxv)
        delta_scaled = ttnn.multiply(abs_max, TIEBREAK_DELTA_SCALE, sub_core_grids=scg)
        ttnn.deallocate(abs_max)
        delta = ttnn.add(delta_scaled, TIEBREAK_DELTA_FLOOR, sub_core_grids=scg)  # [1,1,B,1] bf16
        ttnn.deallocate(delta_scaled)

        # Index domain, int32: lowest global index among the maxima. Push the non-maxima above every
        # real index by TIEBREAK_INDEX_SENTINEL, then take the row min.
        #
        # This min MUST run in int32. ttnn.min/ttnn.max on a FLOAT32 tensor go to the FPU -- only
        # int32 and the opt-in accurate fp32 mean take the SFPU reduce path (see
        # ttnn/cpp/ttnn/operations/reduction/generic/device/common.hpp:use_sfpu_reduce_path) -- and the
        # FPU truncates its source registers to a 10-bit mantissa. Every index above 2**11 therefore
        # comes back rounded to a value that equals NO actual index, the winner mask comes out empty,
        # and the boost silently degrades to a no-op: the tie survives and the greedy token keeps
        # flipping. That is exactly how the first version of this pass failed on device. Int32 min/max
        # are integer-exact at every index. Keep the reduce, the mask arithmetic and the equality all
        # in int32 -- routing any of them through float32 (or uint32, which is on the FPU path too)
        # reintroduces the same rounding. forward() hands us uint32, so cast once here.
        idx = ttnn.typecast(gathered_global_indices, ttnn.int32, sub_core_grids=scg)
        offset = ttnn.multiply(not_max, TIEBREAK_INDEX_SENTINEL, sub_core_grids=scg)  # bf16, exact (2**24)
        ttnn.deallocate(not_max)
        offset_i32 = ttnn.typecast(offset, ttnn.int32, sub_core_grids=scg)
        ttnn.deallocate(offset)
        masked_idx = ttnn.add(idx, offset_i32, sub_core_grids=scg)  # int32
        ttnn.deallocate(offset_i32)
        greedy_i = ttnn.min(masked_idx, dim=3, keepdim=True, sub_core_grids=scg)  # [1,1,B,1] int32
        ttnn.deallocate(masked_idx)

        is_lowidx_i32 = ttnn.eq(idx, greedy_i, sub_core_grids=scg)  # broadcast over W
        ttnn.deallocate(idx)
        ttnn.deallocate(greedy_i)
        is_lowidx = ttnn.typecast(is_lowidx_i32, ttnn.bfloat16, sub_core_grids=scg)  # 1/0 -> 1.0/0.0
        ttnn.deallocate(is_lowidx_i32)
        is_winner = ttnn.multiply(is_max, is_lowidx, sub_core_grids=scg)  # 1.0 at exactly one candidate
        ttnn.deallocate(is_max)
        ttnn.deallocate(is_lowidx)

        # gate by k==1 (self._greedy_col [1,1,B,1] bf16); random users get boost 0 => values unchanged
        winner_gated = ttnn.multiply(is_winner, self._greedy_col, sub_core_grids=scg)
        ttnn.deallocate(is_winner)
        boost = ttnn.multiply(winner_gated, delta, sub_core_grids=scg)  # delta broadcasts over W
        ttnn.deallocate(winner_gated)
        ttnn.deallocate(delta)

        adjusted = ttnn.add(gathered_values, boost, sub_core_grids=scg)
        ttnn.deallocate(boost)
        return adjusted

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
            # Argmax fast-path does not compute logprobs (it never runs a softmax over
            # the vocab). On single-chip, on-device logprobs are unsupported anyway
            # (LogProbsCalculator._is_supported requires num_devices in (8, 32)).
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
                    # Break exact-value ties by lowest index instead of array position, so which
                    # of a set of tied candidates enters the top-k does not depend on placement.
                    # Best effort only, and only where the LLK has the network at all (see
                    # self._topk_stable) -- the stable bitonic network is an open LLK issue
                    # (tenstorrent/tt-metal#33492); _adjust_values_for_tiebreak is what actually
                    # guarantees the greedy pick.
                    stable=self._topk_stable,
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
            if self.topk_split_to_power_of_2:
                topk_values, topk_indices = self._topk_multicore_split(x_bf16)
            else:
                topk_values, topk_indices = ttnn.topk(
                    x_bf16,
                    k=self.max_top_k,
                    dim=-1,
                    sub_core_grids=self.sub_core_grid_topk,
                    indices_tensor=self.tt_indices_tensor,
                    # Break exact-value ties by lowest index instead of array position, so which
                    # of a set of tied candidates enters the top-k does not depend on placement.
                    # Best effort only, and only where the LLK has the network at all (see
                    # self._topk_stable) -- the stable bitonic network is an open LLK issue
                    # (tenstorrent/tt-metal#33492); _adjust_values_for_tiebreak is what actually
                    # guarantees the greedy pick.
                    stable=self._topk_stable,
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
        # Perform the actual sampling with top-k, top-p, and temperature.
        # WORKAROUND for tenstorrent/tt-metal#33492 (stable top-k unreliable), to be removed with it:
        # for argmax users (k==1) only, boost the single lowest-GLOBAL-INDEX tied maximum in the
        # sampling INPUT so ttnn.sampling's argmax picks it regardless of how the top-k network
        # ordered the tied candidates. Random users are byte-for-byte unchanged. Correcting the INPUT
        # (not the RM output buffer) is required: no ttnn op writes an interleaved ROW_MAJOR tensor
        # in-place on a restricted sub-device. See _adjust_values_for_tiebreak for the full rationale
        # and its known limitation (>max_top_k maxima tied within one device shard).
        sampling_values = self._adjust_values_for_tiebreak(
            topk_values_gathered_bf16_interleaved, topk_global_indices_interleaved
        )
        # ``manual_seed`` must be the LAST op before ``ttnn.sampling``: it installs the per-core
        # PRNG state that ``sampling``'s ``generate_rand_tile`` then advances, and that state is a
        # register on the core, not a tensor.  Any compute kernel dispatched to the same core in
        # between destroys it, and the affected users silently fall back to whatever RNG state that
        # kernel left -- so two users given the *same* seed draw different random numbers, purely as
        # a function of which core they sit on.  Measured, not assumed: with the seed call placed
        # before ``_adjust_values_for_tiebreak`` (which typecasts int32 -> bfloat16), the users
        # mapped to the cores that typecast ran on disagreed with the other 30 on 19 of 20 seeds;
        # moving the call here made all 32 agree on every seed.  ``ttnn.typecast(-> bfloat16)`` and
        # ``ttnn.exp`` both reproduce it; ``add``, ``abs`` and ``typecast(-> int32)`` do not, so it
        # is a property of the intervening kernel, not of dispatch in general, and ordering is the
        # only defence this level has.
        ttnn.manual_seed(
            seeds=self.seeds_tt_tensor,
            user_ids=self.user_ids_tt_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
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

        ttnn.deallocate(sampling_values)
        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok, self.tt_log_probs
