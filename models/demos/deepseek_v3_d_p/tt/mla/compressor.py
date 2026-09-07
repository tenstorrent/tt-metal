# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable TT compressor infrastructure and the HCA compressor."""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, resolve_per_axis_topology


def rope_table_tokens(max_seq_len: int, chunk_tokens: int) -> int:
    """Cover the context rounded to whole chunks plus one padded slab."""
    return -(-int(max_seq_len) // chunk_tokens) * chunk_tokens + chunk_tokens


# Rows in one Blaze-compatible CSA overlap state. The state carries 8 rows of real overlap in a
# parity-mapped 64-row slab, which is the layout decode consumes, so prefill keeps the same shape.
CSA_STATE_ROWS = 64


def csa_state_rows(position: int, compress_rate: int) -> tuple[int, int]:
    """The two rows of a CSA overlap state that token ``position`` writes, as ``(ca_row, cb_row)``.

    This is the host-side statement of the ``csa_compressor`` state layout, which Blaze decode also
    consumes. Nothing in the model path calls it -- the op owns the write -- but the op's torch
    reference in ``tests/op_unit_tests/test_csa_compressor.py`` needs the mapping, so it belongs here
    next to the constant that sizes the slab rather than restated there.

    The slab is two tiles of ``TILE_SIZE`` rows, indexed by the window's parity. A window writes its Ca
    half into the tile of the OPPOSITE parity, which is where its successor -- the next window, of
    opposite parity -- looks for a predecessor Ca. Cb stays in the window's own tile, past the
    ``compress_rate`` Ca slots. Consecutive windows therefore never write the same tile, which is what
    lets one read a predecessor while the other is still being filled.

    A corollary the SP hand-off depends on: the row a token writes is a function of its position alone,
    and positions repeat every ``2 * compress_rate`` rows, so a slab filled in position order retains
    only the last ``2 * compress_rate`` tokens -- everything earlier has been overwritten. Any chip
    whose local slab is at least that wide therefore ends a chunk holding exactly the slab that packing
    the WHOLE prefix would produce. That is why ``terminal_state`` can take the last SP chip's outgoing
    state as-is and hand it to the next chunk, and why the same tensor is what Blaze decode consumes,
    with no fix-up for a partially filled final window."""
    slot = position % compress_rate
    parity = (position // compress_rate) & 1
    ca_row = (parity ^ 1) * ttnn.TILE_SIZE + slot
    cb_row = parity * ttnn.TILE_SIZE + compress_rate + slot
    return ca_row, cb_row


def csa_state_predecessor_ca(position: int, compress_rate: int) -> int:
    """First row of the predecessor Ca block that the window containing ``position`` reads; the block is
    ``[result, result + compress_rate)``.

    The same parity trick as ``csa_state_rows``, from the reading side: a window of parity p finds its
    predecessor's Ca in tile p, because that predecessor had parity p^1 and wrote its Ca into the
    opposite tile."""
    parity = (position // compress_rate) & 1
    return parity * ttnn.TILE_SIZE


def csa_slab_align(compress_rate: int, sp_factor: int, tp_factor: int = 1) -> int:
    """The token granularity a CSA slab has to be a multiple of, which is everything CSA asks of a
    slab's width at once:

    - ``compress_rate * TILE_SIZE * sp_factor``: each CHIP's share of the entries one slab produces is a
      whole number of tiles. The block-cyclic indexer score op is handed the local slab in tokens,
      divides it by ``compress_rate`` to get its per-shard chunk in compressed rows, and rejects that
      chunk unless it is tile-aligned. This subsumes the two constraints the compressor alone needs --
      ``compress_rate * sp_factor`` (every SP shard owns whole compression windows) and
      ``compress_rate * TILE_SIZE`` (the entries one slab produces are a whole number of tiles, which
      keeps the compressed append offset on a tile boundary),
    - ``TILE_SIZE * sp_factor * tp_factor``: each chip's share of the slab is a whole number of tiles,
      which the indexer's TP gathers need.

    The first term binds only when ``tp_factor < compress_rate``; at ``tp_factor == compress_rate`` the
    second is exactly equal, which is why an 8x4 mesh at rate 4 never sees it.

    ``max`` stands in for the LCM because every factor here is a power of two."""
    # The max-as-LCM shortcut above silently returns too small an alignment the moment a factor is not a
    # power of two (rate 3 with sp 2 needs 6, but max gives 3), and callers only assert against the value
    # this returns -- so the assumption is checked here, where the arithmetic is.
    for name, factor in (("compress_rate", compress_rate), ("sp_factor", sp_factor), ("tp_factor", tp_factor)):
        assert factor > 0 and factor & (factor - 1) == 0, (
            f"csa_slab_align uses max as an LCM, which needs every factor to be a power of two; " f"{name} is {factor}"
        )
    return max(compress_rate * ttnn.TILE_SIZE * sp_factor, ttnn.TILE_SIZE * sp_factor * tp_factor)


class TtCompressorUtils:
    """Mesh, tensor, and indexed-RoPE operations used by compressors and their consumers."""

    def __init__(
        self,
        device,
        *,
        rotary_emb,
        sp_axis: int,
        tp_axis: int,
        dtype,
        weights_dtype,
        memory_config,
    ):
        self.device = device
        self.rotary_emb = rotary_emb
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.memory_config = memory_config
        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1

    def to_tt_linear_weight(self, weight: torch.Tensor, tp_shard_dim: int | None = None):
        torch_weight = weight.detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
        return self.from_torch(
            torch_weight, mesh_mapper=self.mesh_mapper(tp_dim=tp_shard_dim), dtype=self.weights_dtype
        )

    def from_torch(self, x: torch.Tensor, mesh_mapper=None, dtype=None, layout=ttnn.TILE_LAYOUT, on_device=True):
        """Convert a host tensor, replicating it across a mesh by default."""
        if self.is_mesh and mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        tiled = on_device and layout == ttnn.TILE_LAYOUT
        return ttnn.from_torch(
            x,
            device=self.device if on_device else None,
            dtype=dtype or self.dtype,
            layout=layout,
            memory_config=self.memory_config if tiled else None,
            mesh_mapper=mesh_mapper,
        )

    def mesh_mapper(self, sp_dim: int | None = None, tp_dim: int | None = None):
        """Map host dimensions onto the configured SP and TP mesh axes."""
        if not self.is_mesh:
            return None
        dims = [None, None]
        if sp_dim is not None and self.sp_factor > 1:
            dims[self.sp_axis] = sp_dim
        if tp_dim is not None and self.tp_factor > 1:
            dims[self.tp_axis] = tp_dim
        if dims == [None, None]:
            return ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)

    def scalar_buffer(self, dtype, shape=(1, 1, 1, 1), layout=ttnn.TILE_LAYOUT):
        """Allocate a device scalar buffer that can be overwritten without reallocating."""
        return self.from_torch(torch.zeros(*shape, dtype=torch.int32), dtype=dtype, layout=layout)

    def push_scalar(self, buf, value):
        """Overwrite a device buffer with one host scalar."""
        host_dtype = torch.float32 if buf.dtype == ttnn.float32 else torch.int32
        host = self.from_torch(
            torch.full(tuple(buf.shape), value, dtype=host_dtype),
            dtype=buf.dtype,
            layout=buf.layout,
            on_device=False,
        )
        ttnn.copy_host_to_device_tensor(host, buf)
        return buf

    def build_rope_table(self, count: int, stride: int, layer_type: str = "compress"):
        """Build replicated cos/sin tables for compressed or token positions.

        ``layer_type`` picks the reference's rope variant: "compress" is the YaRN-scaled table the
        HCA/CSA layers share with their compressor, "main" the plain theta=10000 one a sliding-only
        layer uses (``DeepseekV4Attention.rope_layer_type``)."""
        positions = (torch.arange(count) * stride).unsqueeze(0)
        cos, sin = self.rotary_emb(torch.zeros(1), position_ids=positions.to(torch.long), layer_type=layer_type)
        return tuple(self.from_torch(t.repeat_interleave(2, dim=-1)) for t in (cos, sin))

    def rope_index_base(self, rows: int):
        """Build the constant and mutable halves of an indexed-RoPE gather index."""
        const = self.from_torch(
            torch.arange(self.sp_factor * rows, dtype=torch.int32).view(self.sp_factor, rows),
            mesh_mapper=self.mesh_mapper(sp_dim=0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return const, self.scalar_buffer(ttnn.uint32, shape=(1, 1), layout=ttnn.ROW_MAJOR_LAYOUT)

    def rope_index(self, index_base, base: int):
        """Build this slab's gather index from its constant and scalar halves."""
        const, buf = index_base
        return ttnn.add(const, self.push_scalar(buf, base))

    def rope_gather(self, table, index):
        """Gather indexed rows from a replicated RoPE table."""
        out = []
        for tensor in table:
            gathered = ttnn.embedding(index, tensor, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            out.append(ttnn.reshape(gathered, [1, 1, gathered.shape[-2], gathered.shape[-1]]))
        return tuple(out)


class TtCompressorBase(LightweightModule):
    """Common device, mesh, and input setup for TT compressor implementations."""

    def __init__(
        self,
        device,
        *,
        rotary_emb,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.memory_config = memory_config
        self.rotary_emb = rotary_emb

        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        # _project's reduce-scatter/all-gather ride the TP axis; the compressed-row gather, the terminal
        # state hand-off, and the csa_compressor op ride SP. See resolve_per_axis_topology.
        self.sp_ccl_topology, self.tp_ccl_topology = resolve_per_axis_topology(topology, sp_axis, tp_axis)
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = 2 if is_blackhole() else 1
        self.ops = TtCompressorUtils(
            device,
            rotary_emb=rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            dtype=dtype,
            weights_dtype=weights_dtype,
            memory_config=memory_config,
        )
        self._mask_consts = None  # built by alloc_tables, and only when the caller wants a mask block

    @staticmethod
    def prepare_input(hidden: torch.Tensor, sp_factor: int, compress_rate: int):
        """Pad sequence rows so every SP shard owns whole compression windows."""
        seq_len_actual = hidden.shape[1]
        align = compress_rate * sp_factor
        pad = (-seq_len_actual) % align
        if pad:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
        return hidden, seq_len_actual

    def _init_compression_weights(
        self,
        *,
        kv_proj_weight: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        position_bias: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        projection_dim: int,
    ):
        self.wkv = self.ops.to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.wgate = self.ops.to_tt_linear_weight(gate_proj_weight, tp_shard_dim=2)
        self.position_bias = self.ops.from_torch(
            position_bias.detach().reshape(1, 1, self.compress_rate, projection_dim)
        )
        self.kv_norm_weight = self.ops.from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))
        self.trans_mat = self.ops.from_torch(get_rot_transformation_mat())
        self._entry_rope = None
        self._entry_index = None

    def _alloc_rope_tables(self, max_seq_len: int, chunk_tokens: int):
        tokens = rope_table_tokens(max_seq_len, chunk_tokens)
        self._entry_rope = self.ops.build_rope_table(-(-tokens // self.compress_rate), self.compress_rate)
        self._entry_index = self.ops.rope_index_base(chunk_tokens // self.sp_factor // self.compress_rate)

    def _project(self, hidden_states):
        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)
        gate = ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config)

        if self.tp_factor > 1:
            kv = self._tp_all_reduce(kv)
            gate = self._tp_all_reduce(gate)
        return kv, gate

    def _tp_all_reduce(self, tensor):
        tensor = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.tp_axis,
        )
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.tp_axis,
        )

    def _normalize_rotate_and_gather(self, pooled, first_window_position: int, *, gather_sp: bool = True):
        """Normalize and rotate local compressed rows, optionally gathering them across SP."""
        batch, n_windows = pooled.shape[0], pooled.shape[1]
        compressed = ttnn.reshape(pooled, [batch, 1, n_windows, self.head_dim])
        compressed = ttnn.rms_norm(compressed, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(compressed, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
        rope = ttnn.slice(compressed, [0, 0, 0, nope_dim], [batch, 1, n_windows, self.head_dim])
        idx = self.ops.rope_index(self._entry_index, first_window_position // self.compress_rate)
        cos, sin = self.ops.rope_gather(self._entry_rope, idx)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        compressed_kv = ttnn.concat([nope, rope], dim=-1)

        if gather_sp and self.sp_factor > 1:
            compressed_kv = ttnn.experimental.all_gather_async(
                compressed_kv,
                dim=2,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.sp_ccl_topology,
                cluster_axis=self.sp_axis,
            )
        return compressed_kv

    def _build_mask_consts(self, seq_global: int, width: int):
        """Build the constant vectors and mutable scalars for the compressed mask."""
        sp_mapper = self.ops.mesh_mapper(sp_dim=2)
        rate = self.compress_rate
        return {
            "seq": seq_global,
            "thr": self.ops.from_torch(
                ((torch.arange(seq_global) + 1) // rate).float().view(1, 1, seq_global, 1),
                sp_mapper,
                dtype=ttnn.float32,
            ),
            "ic": self.ops.from_torch(
                torch.arange(seq_global).float().view(1, 1, seq_global, 1),
                sp_mapper,
                dtype=ttnn.float32,
            ),
            "w": self.ops.from_torch(torch.arange(width).float().view(1, 1, 1, width), dtype=ttnn.float32),
            "ec": self.ops.scalar_buffer(ttnn.float32),
            "rl": self.ops.scalar_buffer(ttnn.float32),
        }

    def _mask_block(self, seq: int, first_window_position: int, seq_len_actual: int):
        """Build additive compressed-cache mask columns on device.

        Entry ``w`` is visible to query ``i`` iff ``w < (position_i + 1) // compress_rate``, which is the
        reference's ``causal_threshold``; the offset ``kv_actual`` splits off as ``first_window_position /
        rate`` because a chunk only ever starts on a window boundary. Rows past the chunk's real length
        are pad and get -inf outright."""
        rate = self.compress_rate
        seq_global = seq * self.sp_factor
        consts = self._mask_consts
        assert consts is not None and consts["seq"] == seq_global, (
            f"mask constants cover {None if consts is None else consts['seq']} query rows but this call has "
            f"{seq_global}; alloc_tables has to be given the slab forward is called with"
        )
        within = ttnn.lt(
            consts["w"],
            ttnn.add(
                consts["thr"],
                self.ops.push_scalar(consts["ec"], first_window_position // rate),
            ),
        )
        live = ttnn.lt(consts["ic"], self.ops.push_scalar(consts["rl"], seq_len_actual))
        return ttnn.typecast(ttnn.log(ttnn.multiply(within, live)), self.dtype)


class TtHCACompressor(TtCompressorBase):
    def __init__(
        self,
        device,
        *,
        kv_proj_weight: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        position_bias: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        head_dim: int,
        compress_rate: int,
        rope_head_dim: int,
        rotary_emb,
        rms_norm_eps: float = 1e-6,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__(
            device,
            rotary_emb=rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            dtype=dtype,
            weights_dtype=weights_dtype,
            memory_config=memory_config,
        )
        self.head_dim = int(head_dim)
        self.compress_rate = int(compress_rate)
        self.rope_head_dim = int(rope_head_dim)
        self.rms_norm_eps = float(rms_norm_eps)

        self._init_compression_weights(
            kv_proj_weight=kv_proj_weight,
            gate_proj_weight=gate_proj_weight,
            position_bias=position_bias,
            kv_norm_weight=kv_norm_weight,
            projection_dim=self.head_dim,
        )

    def alloc_tables(self, max_seq_len: int, chunk_tokens: int, mask_width: int):
        """Build the indexed-RoPE and mask constants needed by forward."""
        self._alloc_rope_tables(max_seq_len, chunk_tokens)
        self._mask_consts = self._build_mask_consts(chunk_tokens, mask_width)

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCACompressor":
        return cls(
            device,
            kv_proj_weight=reference.kv_proj.weight,
            gate_proj_weight=reference.gate_proj.weight,
            position_bias=reference.position_bias,
            kv_norm_weight=reference.kv_norm.weight,
            head_dim=config.head_dim,
            compress_rate=config.compress_rates["heavily_compressed_attention"],
            rope_head_dim=config.qk_rope_head_dim,
            rotary_emb=reference.rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def forward(
        self,
        hidden_states,
        seq_len_actual: int | None = None,
        first_window_position: int = 0,
    ):
        """Compress an SP/TP-sharded hidden-state slab and build its mask columns."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        if seq_len_actual is None:
            seq_len_actual = seq_len * self.sp_factor

        kv, gate = self._project(hidden_states)

        n_windows = seq_len // self.compress_rate
        t_real = seq_len_actual // self.compress_rate
        assert n_windows > 0, (
            f"each chip needs at least one whole compression window: {seq_len} rows is under "
            f"compress_rate {self.compress_rate}; run prepare_input on the hidden states first"
        )
        gate = ttnn.reshape(gate, [batch, n_windows, self.compress_rate, self.head_dim])
        gate = ttnn.add(gate, self.position_bias)
        weights = ttnn.softmax(gate, dim=2, numeric_stable=True)

        kv = ttnn.reshape(kv, [batch, n_windows, self.compress_rate, self.head_dim])
        pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)
        compressed_kv = self._normalize_rotate_and_gather(pooled, first_window_position)

        mask_block = None
        if seq_len_actual > 1 and t_real > 0:
            mask_block = self._mask_block(seq_len, first_window_position, seq_len_actual)

        return compressed_kv, mask_block


class TtCSACompressor(TtCompressorBase):
    """CSA compressor with Blaze-compatible overlap state."""

    @staticmethod
    def prepare_input(hidden: torch.Tensor, sp_factor: int, compress_rate: int, tp_factor: int = 1):
        """Pad sequence rows up to what ``csa_slab_align`` asks for. The TP factor has to be passed for a
        slab the indexer will see; the compressor alone does not care about it."""
        seq_len_actual = hidden.shape[1]
        pad = (-seq_len_actual) % csa_slab_align(compress_rate, sp_factor, tp_factor)
        if pad:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
        return hidden, seq_len_actual

    def __init__(
        self,
        device,
        *,
        kv_proj_weight: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        position_bias: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        head_dim: int,
        compress_rate: int,
        rope_head_dim: int,
        rotary_emb,
        rms_norm_eps: float = 1e-6,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preloaded_weights: dict[str, ttnn.Tensor] | None = None,
    ):
        super().__init__(
            device,
            rotary_emb=rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            dtype=dtype,
            weights_dtype=weights_dtype,
            memory_config=memory_config,
        )
        self.head_dim = int(head_dim)
        self.compress_rate = int(compress_rate)
        self.rope_head_dim = int(rope_head_dim)
        self.rms_norm_eps = float(rms_norm_eps)
        if preloaded_weights is None:
            self._init_compression_weights(
                kv_proj_weight=kv_proj_weight,
                gate_proj_weight=gate_proj_weight,
                position_bias=position_bias,
                kv_norm_weight=kv_norm_weight,
                projection_dim=2 * self.head_dim,
            )
        else:
            self.wkv = preloaded_weights["kv_proj"]
            self.wgate = preloaded_weights["gate_proj"]
            self.position_bias = preloaded_weights["position_bias"]
            self.kv_norm_weight = preloaded_weights["kv_norm"]
            self.trans_mat = self.ops.from_torch(get_rot_transformation_mat())
            self._entry_rope = None
            self._entry_index = None

    def alloc_overlap_state(self, batch: int = 1):
        """The zero/-inf overlap state a first chunk starts from: no predecessor window, so its slot
        carries a zero KV and a -inf score, which is softmax weight 0.

        One state per SP chip, each covering that chip's own slab, so the host tensor is sp_factor
        states tall and sharded on the sequence axis."""
        rows = CSA_STATE_ROWS * self.sp_factor
        mapper = self.ops.mesh_mapper(sp_dim=2)
        kv_state = self.ops.from_torch(torch.zeros(batch, 1, rows, self.head_dim), mapper)
        score_state = self.ops.from_torch(
            torch.full((batch, 1, rows, self.head_dim), float("-inf")),
            mapper,
        )
        return kv_state, score_state

    def terminal_state(self, state):
        """The state the NEXT chunk starts from, given this chunk's outgoing per-chip states.

        Every chip emits the state for its own slab, but the chunk's last window lives on the last SP
        chip, so that is the only one the next chunk may start from. Gather and take it; the rest are
        freed here.

        An all-gather to keep one chip's rows is a broadcast written the long way, and it is deliberate
        for now. The payload is small: at sp 8 each call moves 64 rows a chip, so 512 KB gathered at
        ``head_dim`` 512 and 128 KB at ``index_head_dim`` 128, and a CSA layer makes four calls a chunk
        (this compressor's KV and score states, plus the indexer's two) for about 1.3 MB against the 160
        KB it keeps. Replacing it needs a broadcast primitive this module does not have today, so the
        gate is measurement: the four programs have to show up in a per-program breakdown of a chunked
        CSA forward first. HCA does not call this, so ``test_ttnn_hca_perf`` says nothing about it."""
        if self.sp_factor == 1:
            return state
        rows = state.shape[2]
        gathered = ttnn.all_gather(state, dim=2, cluster_axis=self.sp_axis, topology=self.sp_ccl_topology)
        start = (self.sp_factor - 1) * rows
        terminal = ttnn.slice(gathered, [0, 0, start, 0], [state.shape[0], 1, start + rows, self.head_dim])
        ttnn.deallocate(state)
        ttnn.deallocate(gathered)
        return terminal

    def alloc_tables(self, max_seq_len: int, chunk_tokens: int, mask_width: int | None = None):
        """``mask_width`` is the compressed-cache capacity the mask columns have to span. The indexer's
        inner compressor leaves it out: its keys go to a block-cyclic cache the score op masks itself."""
        self._alloc_rope_tables(max_seq_len, chunk_tokens)
        if mask_width is not None:
            self._mask_consts = self._build_mask_consts(chunk_tokens, mask_width)

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtCSACompressor":
        return cls(
            device,
            kv_proj_weight=reference.kv_proj.weight,
            gate_proj_weight=reference.gate_proj.weight,
            position_bias=reference.position_bias,
            kv_norm_weight=reference.kv_norm.weight,
            head_dim=config.head_dim,
            compress_rate=config.compress_rates["compressed_sparse_attention"],
            rope_head_dim=config.qk_rope_head_dim,
            rotary_emb=reference.rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def forward(
        self,
        hidden_states,
        initial_kv_state,
        initial_score_state,
        seq_len_actual: int | None = None,
        first_window_position: int = 0,
        gather_sp: bool = True,
    ):
        """Compress one SP slab and return its decode-compatible outgoing state.

        ``gather_sp=False`` keeps the transformed compressed rows local. The CSA
        indexer uses this path to write directly into its block-cyclic key cache.

        Returns ``(compressed_kv, mask_block, kv_state, score_state)``. ``mask_block`` holds the causal
        compressed-cache mask columns and is None unless ``alloc_tables`` was given a mask width.
        """
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        if seq_len_actual is None:
            seq_len_actual = seq_len * self.sp_factor
        assert 0 <= seq_len_actual <= seq_len * self.sp_factor, (
            f"seq_len_actual {seq_len_actual} must fit in the padded global sequence " f"{seq_len * self.sp_factor}"
        )

        kv, gate = self._project(hidden_states)
        n_windows = seq_len // self.compress_rate
        assert n_windows > 0, (
            f"each chip needs at least one whole compression window: {seq_len} rows is under "
            f"compress_rate {self.compress_rate}; run prepare_input on the hidden states first"
        )

        pooled, local_kv_state, local_score_state = ttnn.experimental.deepseek_prefill.csa_compressor(
            kv,
            gate,
            self.position_bias,
            initial_kv_state,
            initial_score_state,
            seq_len_actual=seq_len_actual,
            first_token_position=first_window_position,
            cluster_axis=self.sp_axis,
            topology=self.sp_ccl_topology,
        )
        pooled = ttnn.reshape(pooled, [batch, n_windows, self.head_dim])
        compressed_kv = self._normalize_rotate_and_gather(pooled, first_window_position, gather_sp=gather_sp)

        mask_block = None
        if self._mask_consts is not None and seq_len_actual > 1 and seq_len_actual // self.compress_rate > 0:
            mask_block = self._mask_block(seq_len, first_window_position, seq_len_actual)

        return compressed_kv, mask_block, local_kv_state, local_score_state
