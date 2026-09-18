# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-causal GQA and attention output projection for Llama-3.1 Galaxy prefill."""

import math
from collections.abc import Mapping

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import LlamaKVCache, _cache_memory_config, _validate_cache_tensor
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import DEFAULT_MAX_SEQ_LEN, DEFAULT_NUM_USERS, PrefillGeometry

_MESH_SHAPE = (4, 8)
_SP = 4
_TP = 8
_SP_AXIS = 0
_TP_AXIS = 1
_NUM_LAYERS = Llama31_8BConfig.NUM_LAYERS
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
    """Attend TP-local Q heads over one selected packed K/V prefix with ring SDPA.

    Production reads the cache through ``ring_joint_scaled_dot_product_attention``: the prefix stays
    sequence-parallel, each SP rank streams its own K/V shard around the ring, and the device op
    merges the partial attentions with a running log-sum-exp. Nothing assembles the whole prefix on
    one chip, so the per-chunk cost no longer carries a gather of the prefix or an explicit
    (local_q x prefix) mask.

    The older route -- gather the prefix into ``gathered_k``/``gathered_v``, build that mask, and run
    standard SDPA -- is retained below as a reference implementation. It is built from independent
    coordinate math, so the tests use it as an oracle for the ring path rather than as a second
    production path; see ``_gather_and_reorder`` and ``_build_mask``.

    Persistent ring and gather buffers are reused sequentially. Concurrent calls through one instance
    are unsupported.
    """

    # Q128/K512 explicit-mask standard SDPA uses 1,241,088 B/core of CBs at the BF16 worst
    # case. Reserve one additional 32,768-byte Q buffer as a conservative scheduling margin. The ring
    # op tiles the same Q128/K512 but carries no mask CB, so this stays the bound for both routes.
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
        # The current chunk's query-validity column, reused across its layers (see _chunk_query_valid).
        self._query_valid = None
        self._query_valid_key = None
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

        # Production ring configuration. The ring op merges each rank's partial attention with a
        # running log-sum-exp, and that streaming accumulation runs in the BF16 destination registers,
        # so fp32_dest_acc_en must be off here -- unlike the reference route's standard SDPA above,
        # which keeps FP32 accumulation. Both keep exp_approx_mode off.
        # Q128/K512 measured fastest across prefixes on a 256-row local Q (1.04 ms/call at an 8192
        # prefix against 1.22 for K128), and the chunk shape does not move the numerics; see
        # tests/unit/test_ring_attention_perf.py and tests/unit/test_ring_attention_accuracy.py.
        self.ring_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        self.ring_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # The ring op gathers each rank's K/V shard into these buffers as it walks the ring. They hold
        # one KV head per TP column, matching the packed cache, and are shared by every slot and layer
        # because one call owns the ring for its duration.
        ring_buffers = torch.zeros(1, _NUM_KV_HEADS, self.max_seq_len, _HEAD_DIM)
        ring_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=_MESH_SHAPE, dims=(None, _TP_AXIS))
        self.ring_k, self.ring_v = (
            ttnn.from_torch(
                ring_buffers,
                device=mesh_device,
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ring_mapper,
            )
            for _ in range(2)
        )
        # CCL workers live in the column the SDPA grid above gives up, and the ring op signals its
        # own completion, so these semaphores need no reset between calls.
        self.ring_ccl_offset = ttnn.CoreCoord(grid.x - 1, 0)
        ring_ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(grid.x - 1, 0), ttnn.CoreCoord(grid.x - 1, 1))}
        )
        self.ring_semaphores = [ttnn.create_global_semaphore(mesh_device, ring_ccl_cores, 0) for _ in range(3)]

        # The packed cache has one KV head per TP column. Select one slot/layer plane and gather its
        # capacity/256 SP blocks into these persistent output buffers before restoring natural order.
        gather_shape = (1, 1, self.max_seq_len, _HEAD_DIM)
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
        query_positions = torch.empty(self.max_seq_len // ttnn.TILE_SIZE, _SP, _LOCAL_SEQUENCE, 1)
        for start_index, actual_start in enumerate(range(0, self.max_seq_len, ttnn.TILE_SIZE)):
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
        expected = (self.num_users, _NUM_LAYERS, self.max_seq_len, _SP)
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
        if not 0 <= layer_idx < _NUM_LAYERS:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {_NUM_LAYERS})")
        if actual_start < 0 or actual_start % ttnn.TILE_SIZE:
            raise ValueError(f"actual_start must be nonnegative and tile-aligned, got {actual_start}")
        if not 0 <= actual_start < actual_end <= self.max_seq_len:
            raise ValueError(
                f"actual range must satisfy 0 <= start < end <= {self.max_seq_len}, "
                f"got [{actual_start}, {actual_end})"
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

    def _reorder_natural(self, gathered, extent):
        """Restore natural token order from the rank-major gathered buffer.

        The buffer is already (rank, chunk, block, head_dim) once reshaped, and natural order is that
        same view with rank and chunk swapped, so a fixed five shape ops replace one slice per
        256-token block plus a concat of the same arity. The per-block route cost 2.8 ms at an
        8192-token extent against 0.13 ms here (22x), and it grew with the prefix while this does
        not -- at 32 layers x K and V that reorder was a large share of the per-chunk time.
        tests/unit/test_prefix_reorder_probe.py grades both routes against ground truth at four
        extents.
        """
        stride = self.max_seq_len // _SP
        chunks = extent // _GLOBAL_CHUNK
        ranked = ttnn.reshape(gathered, (_SP, stride, _HEAD_DIM))
        active = ttnn.slice(ranked, [0, 0, 0], [_SP, chunks * _LOCAL_SEQUENCE, _HEAD_DIM])
        split = ttnn.reshape(active, (_SP, chunks, _LOCAL_SEQUENCE, _HEAD_DIM))
        chunk_major = ttnn.permute(split, (1, 0, 2, 3))
        natural = ttnn.reshape(chunk_major, (1, 1, extent, _HEAD_DIM))
        # Every reshape here is a view, so `ranked` aliases the caller's persistent gather buffer and
        # must not be freed; at a full prefix the slice spans every row and hands back that same
        # buffer rather than a copy, which makes `active` an alias too. Freeing either one leaves the
        # next gather reporting that its input and output are on different mesh devices.
        if chunks * _LOCAL_SEQUENCE != stride:
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
            cluster_axis=_SP_AXIS,
            num_links=1,
            input_batch_index=batch_index,
            gathered_dim_size=extent,
        )
        natural = self._reorder_natural(gathered, extent)
        if logical_n < extent:
            prefix = ttnn.slice(natural, [0, 0, 0, 0], [1, 1, logical_n, _HEAD_DIM])
            natural.deallocate(True)
            natural = prefix
        return natural

    def _query_positions(self, actual_start):
        """This chunk's absolute query positions, one exact FP32 column per SP rank."""
        start_index = actual_start // ttnn.TILE_SIZE
        query_positions_rm = ttnn.slice(
            self.query_position_table,
            [start_index, 0, 0, 0],
            [start_index + 1, 1, _LOCAL_SEQUENCE, 1],
        )
        query_positions = ttnn.to_layout(query_positions_rm, ttnn.TILE_LAYOUT)
        query_positions_rm.deallocate(True)
        return query_positions

    def _build_query_valid(self, *, actual_start, actual_end):
        """Which of this rank's query rows carry a real token, as a bf16 0/1 column.

        A chunk is always a full 256 rows per rank, so a short interval leaves padded rows whose
        attention output is undefined: their query positions sit past ``actual_end``, and the keys
        they would read are whatever the tile-rounded prefix holds. Multiplying by this column is what
        makes those rows exact zeros.
        """
        query_positions = self._query_positions(actual_start)
        query_valid = ttnn.lt(query_positions, float(actual_end))
        query_valid_bf16 = ttnn.typecast(query_valid, ttnn.bfloat16)
        for tensor in (query_positions, query_valid):
            tensor.deallocate(True)
        return query_valid_bf16

    def _build_mask(self, *, actual_start, actual_end, logical_n):
        """Reference route: the explicit (local_q x prefix) causal mask for standard SDPA.

        Production does not build this -- the ring op applies causality from ``actual_start`` and
        ``logical_n`` instead. It stays because it derives allowed positions from absolute query and
        key coordinates, independently of the cache layout, which is what makes it an oracle.
        """
        query_positions = self._query_positions(actual_start)

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

    def _chunk_query_valid(self, *, actual_start, actual_end):
        """One chunk's query-validity column, reused by every layer that attends over it.

        Validity depends on the chunk's position range alone -- not on the layer or the cache contents
        -- so the 32 layers of a chunk all want the same column.
        """
        key = (actual_start, actual_end)
        if self._query_valid_key != key:
            self._release_chunk_query_valid()
            self._query_valid = self._build_query_valid(actual_start=actual_start, actual_end=actual_end)
            self._query_valid_key = key
        return self._query_valid

    def _release_chunk_query_valid(self):
        if self._query_valid is not None:
            self._query_valid.deallocate(True)
        self._query_valid = None
        self._query_valid_key = None

    def __call__(self, q, kv_cache, *, slot_idx, layer_idx, actual_start, actual_end):
        self._validate_call(
            q,
            kv_cache,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        # The ring op reads the cache in place. `logical_n` is the total valid prefix rounded up to a
        # tile, and `kv_actual_isl` is what was valid before this chunk, which is how the op recovers
        # the block-cyclic rotation and applies causality. Rounding up only admits keys at positions
        # at or past `actual_end`, and those sit strictly above every valid query position, so
        # causality excludes them and the rounding cannot leak padding into a real row.
        logical_n = math.ceil(actual_end / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        batch_index = slot_idx * _NUM_LAYERS + layer_idx
        query_valid = self._chunk_query_valid(actual_start=actual_start, actual_end=actual_end)
        self._require_sdpa_l1()
        output, joint_output, statistics = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            kv_cache.k,
            kv_cache.v,
            None,
            None,
            None,
            persistent_output_buffer_k=self.ring_k,
            persistent_output_buffer_v=self.ring_v,
            joint_strategy="rear",
            logical_n=logical_n,
            program_config=self.ring_program_config,
            compute_kernel_config=self.ring_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=self.ring_semaphores,
            num_links=1,
            cluster_axis=_SP_AXIS,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Ring,
            ccl_core_grid_offset=self.ring_ccl_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=_HEAD_DIM**-0.5,
            is_balanced=False,
            kv_cache_batch_idx=batch_index,
            kv_actual_isl=actual_start,
        )
        # Llama attends one prefix, so there is no joint sequence, and the merged log-sum-exp is not
        # needed once the op has folded it into the output.
        for auxiliary in (joint_output, statistics):
            if isinstance(auxiliary, ttnn.Tensor) and auxiliary is not output:
                auxiliary.deallocate(True)
        masked_output = ttnn.multiply(output, query_valid)
        output.deallocate(True)
        # query_valid stays alive: it belongs to the chunk, not to this call.
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
