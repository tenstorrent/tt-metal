# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-causal GQA and attention output projection for Llama-3.1 Galaxy prefill."""

import math
from collections.abc import Mapping

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import LlamaKVCache, _cache_memory_config, _validate_cache_tensor

_MESH_SHAPE = (4, 8)
_SP = 4
_TP = 8
_SP_AXIS = 0
_TP_AXIS = 1
_NUM_USERS = 2
_NUM_LAYERS = Llama31_8BConfig.NUM_LAYERS
_MAX_SEQ_LEN = 2048
_GLOBAL_CHUNK = 1024
_LOCAL_SEQUENCE = _GLOBAL_CHUNK // _SP
_HEAD_DIM = Llama31_8BConfig.HEAD_DIM
_NUM_Q_HEADS = Llama31_8BConfig.NUM_ATTENTION_HEADS
_NUM_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
_LOCAL_Q_HEADS = _NUM_Q_HEADS // _TP
_HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
_SUPPORTED_CACHE_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)
_O_WEIGHT_SHAPE = (_HIDDEN_SIZE, _HIDDEN_SIZE)


def _validate_mesh(mesh_device, mesh_config, owner):
    required = ("mesh_shape", "tp", "tp_axis", "sp_axis", "sp")
    missing = [name for name in required if not hasattr(mesh_config, name)]
    if missing:
        raise ValueError(f"{owner} mesh_config is missing: {', '.join(missing)}")
    if tuple(mesh_config.mesh_shape) != _MESH_SHAPE:
        raise ValueError(f"{owner} requires mesh_shape={_MESH_SHAPE}, got {tuple(mesh_config.mesh_shape)}")
    if (mesh_config.sp, mesh_config.tp, mesh_config.sp_axis, mesh_config.tp_axis) != (
        _SP,
        _TP,
        _SP_AXIS,
        _TP_AXIS,
    ):
        raise ValueError(
            f"{owner} requires SP=4 on mesh axis 0 and TP=8 on mesh axis 1; "
            f"got SP={mesh_config.sp}, TP={mesh_config.tp}, "
            f"sp_axis={mesh_config.sp_axis}, tp_axis={mesh_config.tp_axis}"
        )
    if tuple(mesh_device.shape) != _MESH_SHAPE or mesh_device.get_num_devices() != _SP * _TP:
        raise ValueError(
            f"{owner} device requires {_MESH_SHAPE} with {_SP * _TP} chips; "
            f"got shape={tuple(mesh_device.shape)}, devices={mesh_device.get_num_devices()}"
        )


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
    if len(ttnn.get_device_tensors(tensor)) != _SP * _TP:
        raise ValueError(f"{name} must cover {_SP * _TP} mesh devices")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} must use TILE_LAYOUT, got {tensor.layout}")
    if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(f"{name} must use interleaved DRAM, got {tensor.memory_config()}")


def _forwarding_links(mesh_device, *, axis, required_links, owner):
    links = []
    axis_size = _MESH_SHAPE[axis]
    other_axis = 1 - axis
    other_size = _MESH_SHAPE[other_axis]
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

    # SP gather order groups two 256-row blocks per source rank:
    # [sp0c0, sp0c1, sp1c0, sp1c1, ...]. Natural order groups ranks per 1024-token chunk.
    _NATURAL_BLOCK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
    # Q128/K512 explicit-mask standard SDPA uses 1,241,088 B/core of CBs at the BF16 worst
    # case. Reserve one additional 32,768-byte Q buffer as a conservative scheduling margin.
    _SDPA_L1_BYTES = 1_273_856

    def __init__(self, mesh_device, mesh_config, *, cache_dtype=ttnn.bfloat8_b):
        _validate_mesh(mesh_device, mesh_config, "FullCausalAttention")
        if cache_dtype not in _SUPPORTED_CACHE_DTYPES:
            raise ValueError(f"attention cache_dtype must be bfloat16 or bfloat8_b, got {cache_dtype}")
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.cache_dtype = cache_dtype
        self.fabric_links = _forwarding_links(
            mesh_device,
            axis=_SP_AXIS,
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
        # eight 256-row SP blocks into these persistent output buffers before restoring natural order.
        gather_shape = (1, 1, _MAX_SEQ_LEN, _HEAD_DIM)
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
        # persistent logical local payload to 64 KiB; only the selected 256-position slice is tiled.
        query_positions = torch.empty(_MAX_SEQ_LEN // ttnn.TILE_SIZE, _SP, _LOCAL_SEQUENCE, 1)
        for start_index, actual_start in enumerate(range(0, _MAX_SEQ_LEN, ttnn.TILE_SIZE)):
            owned = [[] for _ in range(_SP)]
            for position in range(actual_start, actual_start + _GLOBAL_CHUNK):
                owned[(position % _GLOBAL_CHUNK) // _LOCAL_SEQUENCE].append(position)
            for sp_coord, positions in enumerate(owned):
                query_positions[start_index, sp_coord, :, 0] = torch.tensor(positions, dtype=torch.float32)
        self.query_position_table = ttnn.from_torch(
            query_positions,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=_MESH_SHAPE, dims=(1, None)),
        )
        self.key_positions = ttnn.from_torch(
            torch.arange(_MAX_SEQ_LEN, dtype=torch.float32).reshape(1, 1, 1, _MAX_SEQ_LEN),
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
        expected = (_NUM_USERS, _NUM_LAYERS, _MAX_SEQ_LEN, _SP)
        if metadata != expected:
            raise ValueError(f"attention cache metadata must be {expected}, got {metadata}")
        for name, tensor in (("k", kv_cache.k), ("v", kv_cache.v)):
            _validate_cache_tensor(name, tensor, self.mesh_device)
            if tensor.dtype != self.cache_dtype:
                raise ValueError(
                    f"attention cache {name} dtype must match constructor cache_dtype "
                    f"{self.cache_dtype}, got {tensor.dtype}"
                )
            if tensor.memory_config() != _cache_memory_config(self.mesh_device):
                raise ValueError(f"attention cache {name} must use the fixed packed NdShard DRAM layout")
        if kv_cache.k.dtype != kv_cache.v.dtype:
            raise ValueError(f"attention K/V cache dtypes must match, got {kv_cache.k.dtype} and {kv_cache.v.dtype}")

    @staticmethod
    def _validate_request_fields(*, slot_idx, layer_idx, actual_start, actual_end):
        for name, value in (
            ("slot_idx", slot_idx),
            ("layer_idx", layer_idx),
            ("actual_start", actual_start),
            ("actual_end", actual_end),
        ):
            _validate_scalar(name, value)
        if not 0 <= slot_idx < _NUM_USERS:
            raise ValueError(f"slot_idx {slot_idx} out of range [0, {_NUM_USERS})")
        if not 0 <= layer_idx < _NUM_LAYERS:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {_NUM_LAYERS})")
        if actual_start < 0 or actual_start % ttnn.TILE_SIZE:
            raise ValueError(f"actual_start must be nonnegative and tile-aligned, got {actual_start}")
        if not 0 <= actual_start < actual_end <= _MAX_SEQ_LEN:
            raise ValueError(
                f"actual range must satisfy 0 <= start < end <= {_MAX_SEQ_LEN}, " f"got [{actual_start}, {actual_end})"
            )
        if actual_end - actual_start > _GLOBAL_CHUNK:
            raise ValueError(
                f"actual range may contain at most {_GLOBAL_CHUNK} tokens, " f"got [{actual_start}, {actual_end})"
            )

    def _validate_request_resources(self, kv_cache):
        self._validate_cache(kv_cache)
        topology = ttnn.get_usable_topology(kv_cache.k, topology=ttnn.Topology.Ring, cluster_axis=_SP_AXIS)
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
            shape=(1, _LOCAL_Q_HEADS, _LOCAL_SEQUENCE, _HEAD_DIM),
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

    def _gather_and_reorder(self, cache_tensor, output_tensor, *, batch_index, logical_n):
        gathered = ttnn.experimental.high_bw_all_gather(
            cache_tensor,
            dim=2,
            output_tensor=output_tensor,
            cluster_axis=_SP_AXIS,
            num_links=1,
            input_batch_index=batch_index,
            gathered_dim_size=_MAX_SEQ_LEN,
        )
        blocks = [
            ttnn.slice(
                gathered,
                [0, 0, block * _LOCAL_SEQUENCE, 0],
                [1, 1, (block + 1) * _LOCAL_SEQUENCE, _HEAD_DIM],
            )
            for block in range(_SP * (_MAX_SEQ_LEN // _GLOBAL_CHUNK))
        ]
        natural = ttnn.concat([blocks[index] for index in self._NATURAL_BLOCK_ORDER], dim=2)
        for block in blocks:
            block.deallocate(True)
        if logical_n < _MAX_SEQ_LEN:
            prefix = ttnn.slice(natural, [0, 0, 0, 0], [1, 1, logical_n, _HEAD_DIM])
            natural.deallocate(True)
            natural = prefix
        return natural

    def _build_mask(self, *, actual_start, actual_end, logical_n):
        start_index = actual_start // ttnn.TILE_SIZE
        query_positions_rm = ttnn.slice(
            self.query_position_table,
            [start_index, 0, 0, 0],
            [start_index + 1, 1, _LOCAL_SEQUENCE, 1],
        )
        query_positions = ttnn.to_layout(query_positions_rm, ttnn.TILE_LAYOUT)
        query_positions_rm.deallocate(True)

        owns_key_positions = logical_n < _MAX_SEQ_LEN
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
        batch_index = slot_idx * _NUM_LAYERS + layer_idx
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
        mask, query_valid = self._build_mask(
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
            scale=_HEAD_DIM**-0.5,
            program_config=self.program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        masked_output = ttnn.multiply(output, query_valid)
        output.deallocate(True)
        for tensor in (natural_k, natural_v, mask, query_valid):
            tensor.deallocate(True)
        return masked_output


class AttentionOutputProjection:
    """Concatenate TP-local heads, apply the row-parallel O weight, and TP all-reduce."""

    def __init__(self, mesh_device, mesh_config, state_dict):
        _validate_mesh(mesh_device, mesh_config, "AttentionOutputProjection")
        weight = self._validate_weight(state_dict)
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.fabric_links = _forwarding_links(
            mesh_device,
            axis=_TP_AXIS,
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
            shape=(1, _LOCAL_Q_HEADS, _LOCAL_SEQUENCE, _HEAD_DIM),
            dtype=ttnn.bfloat16,
        )
        topology = ttnn.get_usable_topology(heads, topology=ttnn.Topology.Ring, cluster_axis=_TP_AXIS)
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
            cluster_axis=_TP_AXIS,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        partial.deallocate(True)
        return output
