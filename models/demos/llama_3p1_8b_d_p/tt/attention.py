# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-causal GQA and attention output projection for Llama-3.1 Galaxy prefill."""

import math
from collections.abc import Mapping

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import (
    SUPPORTED_CACHE_DTYPES,
    LlamaKVCache,
    _cache_memory_config,
    _validate_cache_tensor,
)
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import DEFAULT_MAX_SEQ_LEN, DEFAULT_NUM_USERS
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PrefillGeometry, validate_mesh

_O_WEIGHT_SHAPE = (Model.EMB_SIZE, Model.EMB_SIZE)


def _validate_scalar(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an eager Python int, got {type(value).__name__}")


def _validate_device_tensor(tensor, mesh_device, *, name, shape, dtype):
    if not isinstance(tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"{name} must be a device ttnn.Tensor")
    if tensor.device() != mesh_device:
        raise ValueError(f"{name} must reside on the constructor mesh")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have local shape {shape}, got {tuple(tensor.shape)}")
    if len(ttnn.get_device_tensors(tensor)) != layout.num_devices:
        raise ValueError(f"{name} must cover {layout.num_devices} mesh devices")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} must use TILE_LAYOUT, got {tensor.layout}")
    if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(f"{name} must use interleaved DRAM, got {tensor.memory_config()}")


def _forwarding_links(mesh_device, *, axis, required_links, owner):
    links = []
    axis_size = layout.mesh_shape[axis]
    other_axis = 1 - axis
    other_size = layout.mesh_shape[other_axis]
    for other_coord in range(other_size):
        for axis_coord in range(axis_size):
            src = [0, 0]
            src[axis] = axis_coord
            src[other_axis] = other_coord
            src_coord = ttnn.MeshCoordinate(src)
            src_node = mesh_device.get_fabric_node_id(src_coord)
            for neighbor in ((axis_coord - 1) % axis_size, (axis_coord + 1) % axis_size):
                dst = list(src)
                dst[axis] = neighbor
                dst_coord = ttnn.MeshCoordinate(dst)
                dst_node = mesh_device.get_fabric_node_id(dst_coord)
                available = tuple(ttnn.get_forwarding_link_indices(src_node, dst_node))
                if any(link not in available for link in required_links):
                    raise RuntimeError(
                        f"{owner} requires forwarding links {required_links} on every ring edge; "
                        f"src={src_coord}/{src_node}, dst={dst_coord}/{dst_node}, available={available}"
                    )
                links.append((tuple(src), tuple(dst), available))
    return tuple(links)


class FullCausalAttention:
    """Attend TP-local Q heads over one selected packed K/V prefix with supported FP32 SDPA.

    Persistent gather outputs are reused sequentially. Concurrent calls through one instance are
    unsupported.
    """

    # Q128/K512 explicit-mask standard SDPA uses 1,241,088 B/core of CBs at the BF16 worst
    # case. Reserve one additional 32,768-byte Q buffer as a conservative scheduling margin.
    _SDPA_L1_BYTES = 1_273_856

    def __init__(
        self,
        mesh_device,
        mesh_config,
        *,
        cache_dtype=ttnn.bfloat8_b,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        num_users=DEFAULT_NUM_USERS,
    ):
        self.geometry = PrefillGeometry(max_seq_len, num_users)
        self.max_seq_len = self.geometry.max_seq_len
        self.num_users = self.geometry.num_users
        # The current chunk's mask and query-validity column, reused across its layers (see _chunk_mask).
        self._mask = None
        self._query_valid = None
        self._mask_key = None
        validate_mesh(mesh_device, mesh_config, "FullCausalAttention")
        if cache_dtype not in SUPPORTED_CACHE_DTYPES:
            raise ValueError(f"attention cache_dtype must be bfloat16 or bfloat8_b, got {cache_dtype}")
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.cache_dtype = cache_dtype
        self.fabric_links = _forwarding_links(
            mesh_device,
            axis=layout.sp_axis,
            required_links=(0,),
            owner="FullCausalAttention",
        )

        grid = mesh_device.compute_with_storage_grid_size()
        if grid.x < 9 or grid.y < 2:
            raise ValueError(f"attention requires a compute grid at least 9x2, got {grid}")
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # The packed cache has one KV head per TP column. Select one slot/layer plane and gather its
        # capacity/256 SP blocks into these persistent output buffers before restoring natural order.
        gather_shape = (1, 1, self.max_seq_len, Model.HEAD_DIM)
        self.gathered_k = ttnn.empty(
            gather_shape,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.gathered_v = ttnn.empty(
            gather_shape,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Each SP row receives one exact FP32 absolute-position stream. Row-major storage keeps the
        # persistent logical local payload to 32*max_seq_len bytes; only the selected slice is tiled.
        query_positions = torch.empty(self.max_seq_len // ttnn.TILE_SIZE, layout.sp, layout.local_sequence, 1)
        for start_index, actual_start in enumerate(range(0, self.max_seq_len, ttnn.TILE_SIZE)):
            owned = [[] for _ in range(layout.sp)]
            for position in range(actual_start, actual_start + layout.chunk_size):
                owned[(position % layout.chunk_size) // layout.local_sequence].append(position)
            for sp_coord, positions in enumerate(owned):
                query_positions[start_index, sp_coord, :, 0] = torch.tensor(positions, dtype=torch.float32)
        self.query_position_table = ttnn.from_torch(
            query_positions,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=layout.mesh_shape, dims=(1, None)),
        )
        self.key_positions = ttnn.from_torch(
            torch.arange(self.max_seq_len, dtype=torch.float32).reshape(1, 1, 1, self.max_seq_len),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def _validate_cache(self, kv_cache):
        if not isinstance(kv_cache, LlamaKVCache):
            raise ValueError(f"kv_cache must be LlamaKVCache, got {type(kv_cache).__name__}")
        metadata = (kv_cache.num_users, kv_cache.num_layers, kv_cache.max_seq_len, kv_cache.sp)
        expected = (self.num_users, Model.NUM_LAYERS, self.max_seq_len, layout.sp)
        if metadata != expected:
            raise ValueError(f"attention cache metadata must be {expected}, got {metadata}")
        for name, tensor in (("k", kv_cache.k), ("v", kv_cache.v)):
            _validate_cache_tensor(
                name,
                tensor,
                self.mesh_device,
                max_seq_len=self.max_seq_len,
                num_users=self.num_users,
            )
            if tensor.dtype != self.cache_dtype:
                raise ValueError(
                    f"attention cache {name} dtype must match constructor cache_dtype "
                    f"{self.cache_dtype}, got {tensor.dtype}"
                )
            if tensor.memory_config() != _cache_memory_config(self.mesh_device):
                raise ValueError(f"attention cache {name} must use the fixed packed NdShard DRAM layout")
        if kv_cache.k.dtype != kv_cache.v.dtype:
            raise ValueError(f"attention K/V cache dtypes must match, got {kv_cache.k.dtype} and {kv_cache.v.dtype}")

    def _validate_request_fields(self, *, slot_idx, layer_idx, actual_start, actual_end):
        for name, value in (
            ("slot_idx", slot_idx),
            ("layer_idx", layer_idx),
            ("actual_start", actual_start),
            ("actual_end", actual_end),
        ):
            _validate_scalar(name, value)
        if not 0 <= slot_idx < self.num_users:
            raise ValueError(f"slot_idx {slot_idx} out of range [0, {self.num_users})")
        if not 0 <= layer_idx < Model.NUM_LAYERS:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {Model.NUM_LAYERS})")
        if actual_start < 0 or actual_start % ttnn.TILE_SIZE:
            raise ValueError(f"actual_start must be nonnegative and tile-aligned, got {actual_start}")
        if not 0 <= actual_start < actual_end <= self.max_seq_len:
            raise ValueError(
                f"actual range must satisfy 0 <= start < end <= {self.max_seq_len}, "
                f"got [{actual_start}, {actual_end})"
            )
        if actual_end - actual_start > layout.chunk_size:
            raise ValueError(
                f"actual range may contain at most {layout.chunk_size} tokens, " f"got [{actual_start}, {actual_end})"
            )

    def _validate_request_resources(self, kv_cache):
        self._validate_cache(kv_cache)
        topology = ttnn.get_usable_topology(kv_cache.k, topology=ttnn.Topology.Ring, cluster_axis=layout.sp_axis)
        if topology != ttnn.Topology.Ring:
            raise RuntimeError(f"attention requires a live SP ring, but TTNN selected {topology}")
        self._require_sdpa_l1()

    def validate_request(self, kv_cache, *, slot_idx, layer_idx, actual_start, actual_end):
        """Validate all request and cache preconditions before a caller writes the selected KV plane."""

        self._validate_request_fields(
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        self._validate_request_resources(kv_cache)

    def _validate_call(self, q, kv_cache, *, slot_idx, layer_idx, actual_start, actual_end):
        # Preserve eager error precedence: scalar/range fields, then Q, then cache/topology/resources.
        self._validate_request_fields(
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        _validate_device_tensor(
            q,
            self.mesh_device,
            name="attention Q",
            shape=(1, layout.local_q_heads, layout.local_sequence, Model.HEAD_DIM),
            dtype=ttnn.bfloat16,
        )
        self._validate_request_resources(kv_cache)

    def _require_sdpa_l1(self):
        memory = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1)
        if (
            memory.total_bytes_free_per_bank < self._SDPA_L1_BYTES
            or memory.largest_contiguous_bytes_free_per_bank < self._SDPA_L1_BYTES
        ):
            raise RuntimeError(
                "attention requires at least "
                f"{self._SDPA_L1_BYTES} free contiguous L1 bytes per bank; "
                f"got total={memory.total_bytes_free_per_bank}, "
                f"largest={memory.largest_contiguous_bytes_free_per_bank}"
            )

    def _reorder_natural(self, gathered, extent):
        """Restore natural token order from the rank-major gathered buffer.

        The buffer is already (rank, chunk, block, head_dim) once reshaped, and natural order is that
        same view with rank and chunk swapped, so a fixed five shape ops replace one slice per
        256-token block plus a concat of the same arity. The per-block route cost 2.838 ms at an
        8192-token extent against 0.128 ms here (22x), and it grew with the extent while this does
        not -- at 32 layers x K and V that reorder was a large share of the per-chunk time.
        tests/unit/test_prefix_reorder_probe.py grades both routes against ground truth at four
        extents.
        """
        stride = self.max_seq_len // layout.sp
        chunks = extent // layout.chunk_size
        ranked = ttnn.reshape(gathered, (layout.sp, stride, Model.HEAD_DIM))
        active = ttnn.slice(ranked, [0, 0, 0], [layout.sp, chunks * layout.local_sequence, Model.HEAD_DIM])
        split = ttnn.reshape(active, (layout.sp, chunks, layout.local_sequence, Model.HEAD_DIM))
        chunk_major = ttnn.permute(split, (1, 0, 2, 3))
        natural = ttnn.reshape(chunk_major, (1, 1, extent, Model.HEAD_DIM))
        # Every reshape here is a view, so `ranked` aliases the caller's persistent gather buffer and
        # must not be freed; at a full prefix the slice spans every row and hands back that same
        # buffer rather than a copy, which makes `active` an alias too. Freeing either one leaves the
        # next gather reporting that its input and output are on different mesh devices.
        if chunks * layout.local_sequence != stride:
            active.deallocate(True)
        return natural

    def _gather_and_reorder(self, cache_tensor, output_tensor, *, batch_index, logical_n):
        # Move only the chunks the prefix actually populates, not the whole allocation: a chunk-1 read
        # of a 128K cache is 1/128th of the bytes. The active extent is excluded from the op's program
        # hash, so every prefix length still shares ONE cached program. The reorder below then works
        # on the active extent only, so neither the transfer nor the copy scales with capacity.
        extent = self.geometry.gathered_prefix_extent(logical_n)
        gathered = ttnn.experimental.high_bw_all_gather(
            cache_tensor,
            dim=2,
            output_tensor=output_tensor,
            cluster_axis=layout.sp_axis,
            num_links=1,
            input_batch_index=batch_index,
            gathered_dim_size=extent,
        )
        natural = self._reorder_natural(gathered, extent)
        if logical_n < extent:
            prefix = ttnn.slice(natural, [0, 0, 0, 0], [1, 1, logical_n, Model.HEAD_DIM])
            natural.deallocate(True)
            natural = prefix
        return natural

    def _build_mask(self, *, actual_start, actual_end, logical_n):
        start_index = actual_start // ttnn.TILE_SIZE
        query_positions_rm = ttnn.slice(
            self.query_position_table,
            [start_index, 0, 0, 0],
            [start_index + 1, 1, layout.local_sequence, 1],
        )
        query_positions = ttnn.to_layout(query_positions_rm, ttnn.TILE_LAYOUT)
        query_positions_rm.deallocate(True)

        owns_key_positions = logical_n < self.max_seq_len
        key_positions = (
            ttnn.slice(self.key_positions, [0, 0, 0, 0], [1, 1, 1, logical_n])
            if owns_key_positions
            else self.key_positions
        )
        query_valid = ttnn.lt(query_positions, float(actual_end))
        safe_query_positions = ttnn.where(query_valid, query_positions, 0.0)
        allowed = ttnn.le(key_positions, safe_query_positions)
        mask = ttnn.typecast(ttnn.log(allowed), ttnn.bfloat16)
        query_valid_bf16 = ttnn.typecast(query_valid, ttnn.bfloat16)
        temporaries = [query_positions, query_valid, safe_query_positions, allowed]
        if owns_key_positions:
            temporaries.append(key_positions)
        for tensor in temporaries:
            tensor.deallocate(True)
        return mask, query_valid_bf16

    def _chunk_mask(self, *, actual_start, actual_end, logical_n):
        """One chunk's mask, reused by every layer that attends over it.

        The mask is a function of the chunk's position range alone -- not of the layer or of the cache
        contents -- so the 32 layers of a chunk all want the same (local_q x prefix) tensor. Building it
        per layer meant 32 sets of full-extent elementwise passes per chunk to recompute the same bits.
        """
        key = (actual_start, actual_end, logical_n)
        if self._mask_key != key:
            self._release_chunk_mask()
            self._mask, self._query_valid = self._build_mask(
                actual_start=actual_start,
                actual_end=actual_end,
                logical_n=logical_n,
            )
            self._mask_key = key
        return self._mask, self._query_valid

    def _release_chunk_mask(self):
        for tensor in (self._mask, self._query_valid):
            if tensor is not None:
                tensor.deallocate(True)
        self._mask = None
        self._query_valid = None
        self._mask_key = None

    def __call__(self, q, kv_cache, *, slot_idx, layer_idx, actual_start, actual_end):
        self._validate_call(
            q,
            kv_cache,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        logical_n = math.ceil(actual_end / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        batch_index = slot_idx * Model.NUM_LAYERS + layer_idx
        natural_k = self._gather_and_reorder(
            kv_cache.k,
            self.gathered_k,
            batch_index=batch_index,
            logical_n=logical_n,
        )
        natural_v = self._gather_and_reorder(
            kv_cache.v,
            self.gathered_v,
            batch_index=batch_index,
            logical_n=logical_n,
        )
        mask, query_valid = self._chunk_mask(
            actual_start=actual_start,
            actual_end=actual_end,
            logical_n=logical_n,
        )
        self._require_sdpa_l1()
        output = ttnn.transformer.scaled_dot_product_attention(
            q,
            natural_k,
            natural_v,
            attn_mask=mask,
            is_causal=False,
            scale=Model.HEAD_DIM**-0.5,
            program_config=self.program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        masked_output = ttnn.multiply(output, query_valid)
        output.deallocate(True)
        # mask/query_valid stay alive: they belong to the chunk, not to this call.
        for tensor in (natural_k, natural_v):
            tensor.deallocate(True)
        return masked_output


class AttentionOutputProjection:
    """Concatenate TP-local heads, apply the row-parallel O weight, and TP all-reduce."""

    def __init__(self, mesh_device, mesh_config, state_dict):
        validate_mesh(mesh_device, mesh_config, "AttentionOutputProjection")
        weight = self._validate_weight(state_dict)
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.fabric_links = _forwarding_links(
            mesh_device,
            axis=layout.tp_axis,
            required_links=(0, 1),
            owner="AttentionOutputProjection",
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        self.o_weight = ttnn.from_torch(
            weight.to(torch.bfloat16).transpose(-2, -1).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.row_parallel(mesh_device),
        )

    @staticmethod
    def _validate_weight(state_dict):
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"attention output state_dict must be a mapping, got {type(state_dict).__name__}")
        if "o_proj.weight" not in state_dict:
            raise ValueError("attention output state_dict is missing required weight: o_proj.weight")
        weight = state_dict["o_proj.weight"]
        if not isinstance(weight, torch.Tensor):
            raise ValueError(f"attention output o_proj.weight must be a host torch.Tensor, got {type(weight).__name__}")
        if weight.device.type != "cpu":
            raise ValueError(f"attention output o_proj.weight must be a host CPU tensor, got {weight.device}")
        if tuple(weight.shape) != _O_WEIGHT_SHAPE:
            raise ValueError(
                f"attention output o_proj.weight must have shape {_O_WEIGHT_SHAPE}, got {tuple(weight.shape)}"
            )
        return weight

    def _validate_input(self, heads):
        _validate_device_tensor(
            heads,
            self.mesh_device,
            name="attention output heads",
            shape=(1, layout.local_q_heads, layout.local_sequence, Model.HEAD_DIM),
            dtype=ttnn.bfloat16,
        )
        topology = ttnn.get_usable_topology(heads, topology=ttnn.Topology.Ring, cluster_axis=layout.tp_axis)
        if topology != ttnn.Topology.Ring:
            raise RuntimeError(f"attention output requires a live TP ring, but TTNN selected {topology}")

    def __call__(self, heads):
        self._validate_input(heads)
        concatenated = ttnn.experimental.nlp_concat_heads(heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        partial = ttnn.matmul(
            concatenated,
            self.o_weight,
            program_config=self.program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        concatenated.deallocate(True)
        output = ttnn.all_reduce(
            partial,
            cluster_axis=layout.tp_axis,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        partial.deallocate(True)
        return output
