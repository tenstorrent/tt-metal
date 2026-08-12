# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Heavily Compressed Attention (TTNN prefill).
Mirrors ``DeepseekV4Attention`` in ``reference/deepseek_v4/modeling_deepseek_v4.py``."""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl


def _buf_for_width(width: int) -> int:
    """Rows the tail_tile write assembles at once: the up to TILE_SIZE-1 entries already in the open tile
    plus one chunk's worth, rounded to whole tiles. 64 for a 4096-token chunk, 96 for 5120."""
    return -(-(ttnn.TILE_SIZE - 1 + width) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _tail_tile_buf(chunk_tokens: int, compress_rate: int) -> int:
    return _buf_for_width(-(-int(chunk_tokens) // compress_rate))


def hca_block_bias(position_ids: torch.Tensor, compressed_len: int, compress_rate: int) -> torch.Tensor:
    """Per-query causal mask: query ``t`` may attend entry ``w`` only if ``t >= (w+1)*compress_rate``."""
    batch, seq_len = position_ids.shape
    entry_indices = torch.arange(compressed_len)
    causal_threshold = (position_ids + 1) // compress_rate
    block_bias = torch.zeros(batch, 1, seq_len, compressed_len)
    return block_bias.masked_fill(
        entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
        float("-inf"),
    )


class _TtHCABase(LightweightModule):
    """Helpers shared by the compressor and the block. Subclasses must set ``device`` / ``dtype`` /
    ``weights_dtype`` / ``memory_config`` / ``rotary_emb`` and the mesh attributes before calling these."""

    @staticmethod
    def prepare_input(hidden: torch.Tensor, sp_factor: int, compress_rate: int):
        """Pad seq up to a multiple of ``compress_rate * sp_factor`` so each SP shard holds whole
        compression windows. Must run before sharding: appending per-chip would land mid-sequence.
        Pad rows are causally masked downstream and trimmed via the returned ``seq_len_actual``."""
        seq_len_actual = hidden.shape[1]
        align = compress_rate * sp_factor
        pad = (-seq_len_actual) % align
        if pad:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
        return hidden, seq_len_actual

    def _to_tt_linear_weight(self, weight: torch.Tensor, tp_shard_dim: int | None = None):
        # tp_shard_dim indexes the transposed 4D weight [1, 1, in, out]: 2 = contraction (in), 3 = output.
        # Handed over at its source dtype: from_torch quantizes straight to weights_dtype, and a bf16
        # stop on the way to bfloat8_b would pre-round away what the block exponent could still keep.
        torch_weight = weight.detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
        mesh_mapper = None
        if self.is_mesh:
            if tp_shard_dim is not None and self.tp_factor > 1:
                dims = [None, None]
                dims[self.tp_axis] = tp_shard_dim
                mesh_mapper = ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            torch_weight,
            device=self.device,
            dtype=self.weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _from_torch(self, x: torch.Tensor, mesh_mapper=None, dtype=None):
        if self.is_mesh and mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            x,
            device=self.device,
            dtype=dtype or self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _sp_shard_mapper(self, dim: int = 2):
        if not (self.is_mesh and self.sp_factor > 1):
            return ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None
        dims = [None, None]
        dims[self.sp_axis] = dim
        return ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)

    def _f32(self, x: torch.Tensor, mesh_mapper=None):
        """float32 on device; replicated unless a mapper is given. The mask comparisons run in float32
        because bf16 is only exact on integers to 256."""
        if mesh_mapper is None and self.is_mesh:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            x,
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=mesh_mapper,
        )

    def _build_mask_consts(self, seq_global: int, width: int):
        """Index vectors for the compressed mask block. ``thr`` and ``ic`` are indexed by the GLOBAL query
        row, so they are SP-sharded exactly like the mask they feed; ``w`` is replicated.

        float32, not the block dtype: the comparison is on integers up to width-1 and bf16 is only exact to
        256 -- measured, bf16 misses 640 of 328K elements at entry_count 300 and 4480 of 5M at 7863."""
        shard = self._sp_shard_mapper(dim=2)
        rate = self.compress_rate if hasattr(self, "compress_rate") else self.compressor.compress_rate
        return {
            "seq": seq_global,
            "width": width,
            "thr": self._f32(((torch.arange(seq_global) + 1) // rate).float().view(1, 1, seq_global, 1), shard),
            "ic": self._f32(torch.arange(seq_global).float().view(1, 1, seq_global, 1), shard),
            "w": self._f32(torch.arange(width).float().view(1, 1, 1, width)),
            "ec": self._scalar_buffer(ttnn.float32),
            "rl": self._scalar_buffer(ttnn.float32),
        }

    def _scalar_buffer(self, dtype, shape=(1, 1, 1, 1), layout=ttnn.TILE_LAYOUT):
        """Persistent 1-element buffer for a per-chunk scalar. Refreshed in place by ``_push_scalar``, so
        forward allocates nothing -- the same shape MLA's per-chunk metadata uses
        (tt_prefill_runtime.py:481), and the only form that could later be trace-safe."""
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.int32),
            device=self.device,
            dtype=dtype,
            layout=layout,
            memory_config=self.memory_config if layout == ttnn.TILE_LAYOUT else None,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None,
        )

    def _push_scalar(self, buf, v):
        """Write one value into an existing device buffer. 4 bytes, and the only thing forward still sends
        from host: the chunk's real length is known nowhere else."""
        host_dtype = torch.float32 if buf.dtype == ttnn.float32 else torch.int32
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.full(tuple(buf.shape), v, dtype=host_dtype),
                dtype=buf.dtype,
                layout=buf.layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None,
            ),
            buf,
        )
        return buf

    def _mask_block(self, seq: int, width: int, first_window_position: int, seq_len_actual: int):
        """The mask's compressed columns, built on device: 0 where a query may attend an entry, -inf else.
        Replaces the host ``hca_block_bias``, which now only serves the tests as a reference.

        A query in global row j may attend entry w while ``w < entry_count + (j+1)//compress_rate``, so with
        ``thr[j] = (j+1)//compress_rate`` the whole block is ONE broadcast comparison against a 1-element
        tensor. The zeros over older entries, the staircase over this chunk's, and the -inf tail over
        entries not written yet all fall out of that one condition -- which is what lets the width stay at
        the cache capacity and the shape stay constant across chunks.

        ``log`` turns the 1/0 a comparison returns into the 0/-inf an additive mask wants: ttnn gives
        log(1) = 0 and log(0) = -inf exactly, in both float32 and bfloat16."""
        rate = self.compress_rate if hasattr(self, "compress_rate") else self.compressor.compress_rate
        # seq arrives LOCAL (the chip's slab rows); the index vectors carry GLOBAL rows and are sharded,
        # so a chip compares its own rows against the global entry_count and real_len.
        seq_global = seq * self.sp_factor
        c = self._mask_consts
        if c is None or c["seq"] != seq_global or c["width"] != width:
            c = self._mask_consts = self._build_mask_consts(seq_global, width)
        within = ttnn.lt(c["w"], ttnn.add(c["thr"], self._push_scalar(c["ec"], first_window_position // rate)))
        live = ttnn.lt(c["ic"], self._push_scalar(c["rl"], seq_len_actual))  # pad query rows attend nothing
        return ttnn.typecast(ttnn.log(ttnn.multiply(within, live)), self.dtype)

    def _build_rope_table(self, count: int, stride: int):
        """cos/sin for every position this layer can ever rotate by, REPLICATED so each chip can reach any
        row. Built once; forward only gathers from it.

        Replicated and not SP-sharded because the rows a chip needs move with ``kv_actual``: chip c wants
        ``kv_actual + c*local + [0, local)``, which walks past any fixed contiguous shard. MLA avoids the
        replication by rotating which token lands on which chip (mla/utils.py:83) so its shard is
        position-invariant, but that rotation would break the compressor's contiguous 128-token windows.
        At 64 rope columns in bf16 a table is 128 B a row: 14 MB for a 56K token cache, 246 MB at 1M
        against ~32 GiB of DRAM per chip."""
        positions = (torch.arange(count) * stride).unsqueeze(0)
        cos, sin = self.rotary_emb(torch.zeros(1), position_ids=positions.to(torch.long), layer_type="compress")
        pair = []
        for t in (cos, sin):
            t = t.repeat_interleave(2, dim=-1)  # [1, count, rope_head_dim]
            pair.append(
                ttnn.from_torch(
                    t,
                    device=self.device,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self.memory_config,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None,
                )
            )
        return tuple(pair)

    def _rope_index_base(self, rows: int):
        """The gather index splits into a constant and a scalar: index[c][r] = base + c*rows + r. This is the
        constant half, SP-sharded so chip c holds its own row, plus a 1-element buffer for the base.

        Only ``base`` moves between chunks, so forward pushes 4 bytes rather than computing sp*rows indices
        on host -- the same shape MLA's per-chunk metadata uses (tt_prefill_runtime.py:481)."""
        mapper = None
        if self.is_mesh:
            dims = [None, None]
            dims[self.sp_axis] = 0
            mapper = ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)
        const = ttnn.from_torch(
            torch.arange(self.sp_factor * rows, dtype=torch.int32).view(self.sp_factor, rows),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        return const, self._scalar_buffer(ttnn.uint32, shape=(1, 1), layout=ttnn.ROW_MAJOR_LAYOUT)

    def _rope_index(self, index_base, base: int):
        """This chunk's gather index, built on device from the constant half plus a 4-byte base."""
        const, buf = index_base
        return ttnn.add(const, self._push_scalar(buf, base))

    def _rope_gather(self, table, index):
        """Pick this chunk's rows out of the replicated table. ttnn.embedding addresses rows BY INDEX, so
        unlike ttnn.slice it carries no tile-alignment constraint -- which matters for the compressor,
        whose per-chip start ``entry_count + c*n_windows`` is never a multiple of TILE_SIZE. Index values
        stay out of the program hash, so every chunk reuses one program."""
        out = []
        for t in table:
            g = ttnn.embedding(index, t, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            out.append(ttnn.reshape(g, [1, 1, g.shape[-2], g.shape[-1]]))
        return tuple(out)

    def _cos_sin(self, positions: torch.Tensor, negate_sin: bool = False):
        """``negate_sin`` gives the conjugate rotation used to undo RoPE on the attention output."""
        positions = positions[:1].to(torch.long)
        cos, sin = self.rotary_emb(torch.zeros(1), position_ids=positions, layer_type="compress")
        if negate_sin:
            sin = -sin
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
        mesh_mapper = None
        if self.is_mesh and self.sp_factor > 1:
            dims = [None, None]
            dims[self.sp_axis] = 2
            mesh_mapper = ttnn.ShardTensor2dMesh(self.device, mesh_shape=tuple(self.device.shape), dims=dims)
        return self._from_torch(cos, mesh_mapper=mesh_mapper), self._from_torch(sin, mesh_mapper=mesh_mapper)


class TtHCACompressor(_TtHCABase):
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
        self.device = device
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.memory_config = memory_config
        self.head_dim = int(head_dim)
        self.compress_rate = int(compress_rate)
        self.rope_head_dim = int(rope_head_dim)
        self.rotary_emb = rotary_emb
        self.rms_norm_eps = float(rms_norm_eps)

        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        self.ccl_topology = topology
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = 2 if is_blackhole() else 1

        self.wkv = self._to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.wgate = self._to_tt_linear_weight(gate_proj_weight, tp_shard_dim=2)
        self.position_bias = self._from_torch(position_bias.detach().reshape(1, 1, self.compress_rate, self.head_dim))
        self.kv_norm_weight = self._from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))
        self.trans_mat = self._from_torch(get_rot_transformation_mat())
        self._entry_rope = None  # set by TtHCA.alloc_state; None means build on host (standalone use)
        self._entry_index = None
        self._mask_width = None  # set by TtHCA.alloc_state to the cache capacity; None = this call's entries
        self._mask_consts = None

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
        position_ids: torch.Tensor,
        seq_len_actual: int | None = None,
        first_window_position: int = 0,
        total_entries: int | None = None,
    ):
        """[B, 1, S_pad/sp, hidden/tp] -> compressed_kv [B, 1, S_pad/compress_rate, head_dim] (replicated),
        plus a host block_bias [B, 1, S_real, entries].

        ``seq_len_actual`` is the real pre-pad length (``None`` = unpadded single-shot).
        ``first_window_position`` is the global token position of this call's first window; without it a
        later chunk would rotate its entries as if the sequence restarted.

        The output width follows the padded tensor, not the real length -- trailing pad-derived entries
        are left in so the shape stays fixed across chunks. The caller tracks how many are real and the
        mask -infs the rest."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        if seq_len_actual is None:
            seq_len_actual = seq_len * self.sp_factor

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)
        gate = ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config)

        # Row-parallel weights leave a partial sum on every chip -> TP all-reduce (RS + AG).
        if self.tp_factor > 1:
            for name in ("kv", "gate"):
                t = kv if name == "kv" else gate
                t = ttnn.experimental.reduce_scatter_minimal_async(
                    t,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(
                        cluster_axis=self.tp_axis
                    ),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                    num_links=self.ccl_num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl_topology,
                    cluster_axis=self.tp_axis,
                )
                t = ttnn.experimental.all_gather_async(
                    t,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(
                        cluster_axis=self.tp_axis
                    ),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                    num_links=self.ccl_num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl_topology,
                    cluster_axis=self.tp_axis,
                )
                if name == "kv":
                    kv = t
                else:
                    gate = t

        usable = (seq_len // self.compress_rate) * self.compress_rate
        n_windows = usable // self.compress_rate  # windows this chip owns
        t_real = seq_len_actual // self.compress_rate
        if n_windows > 0:
            gate = ttnn.slice(gate, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            gate = ttnn.reshape(gate, [batch, n_windows, self.compress_rate, self.head_dim])
            gate = ttnn.add(gate, self.position_bias)
            weights = ttnn.softmax(gate, dim=2, numeric_stable=True)

            kv = ttnn.slice(kv, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            kv = ttnn.reshape(kv, [batch, n_windows, self.compress_rate, self.head_dim])
            pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)

            compressed = ttnn.reshape(pooled, [batch, 1, n_windows, self.head_dim])
            compressed = ttnn.rms_norm(compressed, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

            nope_dim = self.head_dim - self.rope_head_dim
            nope = ttnn.slice(compressed, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
            rope = ttnn.slice(compressed, [0, 0, 0, nope_dim], [batch, 1, n_windows, self.head_dim])
            if self._entry_rope is None:  # standalone compressor use: no state, so no table
                positions = (
                    torch.arange(n_windows * self.sp_factor) * self.compress_rate + first_window_position
                ).unsqueeze(0)
                cos, sin = self._cos_sin(positions)
            else:
                # Index is in ENTRIES: table row k carries position k*compress_rate, so the window's
                # position has to be divided back down.
                idx = self._rope_index(self._entry_index, first_window_position // self.compress_rate)
                cos, sin = self._rope_gather(self._entry_rope, idx)
            rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
            compressed_kv = ttnn.concat([nope, rope], dim=-1)

            if self.sp_factor > 1:
                compressed_kv = ttnn.experimental.all_gather_async(
                    compressed_kv,
                    dim=2,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(
                        cluster_axis=self.sp_axis
                    ),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                    num_links=self.ccl_num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.ccl_topology,
                    cluster_axis=self.sp_axis,
                )
        else:
            compressed_kv = self._from_torch(torch.zeros(batch, 1, 0, self.head_dim))

        # Must span every entry the queries can see -- under chunked prefill that includes earlier
        # chunks', not just this call's. total_entries=None -> single-shot, where the two coincide.
        bias_entries = t_real if total_entries is None else total_entries
        mask_block = None
        if seq_len_actual > 1 and bias_entries > 0:
            mask_block = self._mask_block(
                seq_len, self._mask_width or bias_entries, first_window_position, seq_len_actual
            )

        return compressed_kv, mask_block


class TtHCAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtHCA.forward``.

    Both device tensors keep a FIXED shape for the whole prefill and only their contents advance --
    that is what lets one compiled program serve every chunk. The counters say how much is real;
    the attention mask -infs the rest."""

    def __init__(self, compressed_kv, sliding_carry, tail=None):
        self.compressed_kv = compressed_kv  # [B, 1, compressed_capacity, head_dim]
        self.sliding_carry = sliding_carry  # [B, 1, sliding_window, head_dim]
        # Entries of the cache tile that is only partly filled, right-aligned so one uniform shift places
        # them and the new entries together. Only the tail_tile write uses it.
        self.tail = tail  # [B, 1, TILE_SIZE, head_dim] or None
        self.entry_count = 0
        self.kv_actual = 0


class TtHCA(_TtHCABase):
    """HCA block: query/kv stems + compressor + attention core + grouped output projection.

    Block I/O is ``[B, 1, S/sp, hidden/tp]``, so layers chain without a reshard."""

    def __init__(
        self,
        device,
        *,
        compressor: TtHCACompressor,
        q_a_proj_weight: torch.Tensor,
        q_a_norm_weight: torch.Tensor,
        q_b_proj_weight: torch.Tensor,
        kv_proj_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        sinks: torch.Tensor,
        o_a_proj_weight: torch.Tensor,
        o_b_proj_weight: torch.Tensor,
        rotary_emb,
        num_heads: int,
        head_dim: int,
        rope_head_dim: int,
        sliding_window: int,
        o_groups: int,
        rms_norm_eps: float = 1e-6,
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
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.sliding_window = int(sliding_window)
        self.scaling = self.head_dim**-0.5
        self.rotary_emb = rotary_emb
        self.rms_norm_eps = float(rms_norm_eps)
        self.compressor = compressor

        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        self.tp_ccl_topology = topology
        self.tt_ccl = get_tt_ccl(device) if (self.is_mesh and (self.sp_factor > 1 or self.tp_factor > 1)) else None
        self.ccl_num_links = 2 if is_blackhole() else 1

        # Pre-divided by scale: SDPA scales BOTH QK and the sink internally, the reference scales only
        # QK -- dividing here cancels the kernel's extra multiply. TP-sharded to match the query heads.
        sinks_host = sinks.detach().reshape(1, self.num_heads, 1, 1) / self.scaling
        if self.is_mesh and self.tp_factor > 1:
            sink_dims = [None, None]
            sink_dims[self.tp_axis] = 1
            sink_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=sink_dims)
            self.sinks_sdpa = self._from_torch(sinks_host, mesh_mapper=sink_mapper)
        else:
            self.sinks_sdpa = self._from_torch(sinks_host)

        self.wq_a = self._to_tt_linear_weight(q_a_proj_weight, tp_shard_dim=2)
        self.wq_b = self._to_tt_linear_weight(q_b_proj_weight, tp_shard_dim=3)
        self.q_a_norm_weight = self._from_torch(q_a_norm_weight.detach().reshape(1, 1, 1, -1))
        self.q_b_norm_weight = self._from_torch(torch.ones(1, 1, 1, self.head_dim))
        self.wkv = self._to_tt_linear_weight(kv_proj_weight, tp_shard_dim=2)
        self.kv_norm_weight = self._from_torch(kv_norm_weight.detach().reshape(1, 1, 1, self.head_dim))

        # o_a_proj is block-diagonal over o_groups. Groups partition the heads, so a TP chip owns whole
        # groups: keep it as ONE batched weight sharded on the group axis and run a single batched
        # matmul -- each chip applies only its own groups, no collective
        self.o_groups = int(o_groups)
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        o_a_grouped = o_a_proj_weight.detach().view(self.o_groups, -1, in_per_group).transpose(1, 2).unsqueeze(0)
        o_a_mapper = None
        if self.is_mesh and self.tp_factor > 1:
            o_a_dims = [None, None]
            o_a_dims[self.tp_axis] = 1
            o_a_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=o_a_dims)

        self.wo_a = self._from_torch(o_a_grouped, mesh_mapper=o_a_mapper, dtype=self.weights_dtype)
        self.wo_b = self._to_tt_linear_weight(o_b_proj_weight, tp_shard_dim=2)

        self.trans_mat = self._from_torch(get_rot_transformation_mat())
        self._shift_cache = {}
        self._take_cache = {}
        self._carry_index = {}
        self._slab_rope = None  # set by alloc_state for chunked use; None means build per call
        self._slab_index = None
        self._mask = None  # persistent additive mask; forward overwrites only the moving columns
        self._mask_col = None
        self._carry_cols = None
        self._kv_pad = None  # zero rows that bring Sk up to a tile multiple; constant, so built once

    def _build_masks(self, seq_global: int, cap: int):
        """Two persistent additive masks over the key layout ``[carry | chunk | compressed | pad]``. The
        carry, chunk and pad columns are constant across chunks and written here; forward only overwrites
        the compressed columns, at a FIXED offset, so nothing recompiles.

        Two of them because chunk 0's carry holds zeros for positions that do not exist yet. In
        ``allowed = (j >= 0) & (j <= i) & (i - j < sw)`` the global offset kv_actual CANCELS in the last two
        -- verified bit-exact against the old host mask -- and survives only in ``j >= 0``, which reduces to
        ``jc >= carry`` at kv_actual = 0 and holds trivially after. So the first chunk masks its carry
        columns and every later chunk does not; nothing else differs, and the shape is the same for both.

        Built from four small index vectors, so no large host tensor is created anywhere."""
        carry, sw = self.sliding_window, self.sliding_window
        raw = carry + seq_global
        sk_pad = -(-(raw + cap) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        shard = self._sp_shard_mapper(dim=2)

        ic = self._f32(torch.arange(seq_global).float().view(1, 1, seq_global, 1), shard)
        ic_lo = self._f32((torch.arange(seq_global) - sw).float().view(1, 1, seq_global, 1), shard)
        jc = self._f32((torch.arange(raw) - carry).float().view(1, 1, 1, raw))

        # j <= i  and  i - j < sw, both with kv_actual cancelled
        sliding = ttnn.typecast(ttnn.log(ttnn.multiply(ttnn.le(jc, ic), ttnn.gt(jc, ic_lo))), self.dtype)

        zero_seq = ttnn.multiply(ic, 0.0)
        blank = ttnn.typecast(ttnn.add(zero_seq, self._f32(torch.zeros(1, 1, 1, cap))), self.dtype)
        parts = [sliding, blank]
        pad_w = sk_pad - raw - cap
        if pad_w:
            parts.append(
                ttnn.typecast(ttnn.log(ttnn.add(zero_seq, self._f32(torch.zeros(1, 1, 1, pad_w)))), self.dtype)
            )
        self._mask = ttnn.concat(parts, dim=3)
        self._mask_col = raw
        # SDPA tile-pads a non-aligned Sk with zeros and a provided mask reads those columns as 0 (= attend),
        # so the kv side gets explicit zero rows and the mask -infs their columns. Constant, hence built here.
        sk = raw + cap
        self._kv_pad = self._from_torch(torch.zeros(1, 1, sk_pad - sk, self.head_dim)) if sk_pad > sk else None

        # ONE mask, not two: the first chunk differs only in its carry columns, whose positions do not exist
        # yet, so both variants of that [seq, carry] slab are kept and forward writes the right one. Building
        # a second full mask instead cost 1.1 ms of BinaryNg on device for the same information.
        self._carry_cols = {
            False: ttnn.slice(sliding, [0, 0, 0, 0], [1, 1, sliding.shape[2], carry]),
            True: ttnn.typecast(ttnn.log(ttnn.multiply(zero_seq, self._f32(torch.zeros(1, 1, 1, carry)))), self.dtype),
        }

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None) -> TtHCAState:
        """Size the state once for the longest context this layer will serve, so its shape is fixed for
        every chunk. ``max_seq_len`` is the same serving-capacity knob the runner resolves for the KV
        cache. Contents start zeroed and nothing is read before it is written -- the mask -infs
        everything past ``entry_count`` / ``kv_actual``, including chunk 0's empty carry."""
        entries = -(-int(max_seq_len) // self.compressor.compress_rate)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE  # cache writes land on tile boundaries
        # The write rewrites whole tiles from the tile boundary below entry_count, so the last write can
        # reach past the entries themselves. ``chunk_tokens`` sizes that headroom; single-shot is one chunk.
        chunk = chunk_tokens or max_seq_len
        capacity += _tail_tile_buf(chunk, self.compressor.compress_rate)
        # forward must build no host tensors, so every one-hot the write can ever need is built here. The
        # slab width is fixed -- it is part of the program shape, so the no-recompile contract already
        # forbids it moving -- but r_e reaches every value once chunks carry differing real lengths, so a
        # chunked state needs the whole TILE_SIZE set. Single-shot writes once, at r_e = 0, and keeps no
        # tail, so it needs no take matrices at all.
        width = -(-int(chunk) // self.compressor.compress_rate)
        self._build_tail_tile_matrices(
            width,
            range(ttnn.TILE_SIZE) if chunk_tokens is not None else (0,),
            range(ttnn.TILE_SIZE + width) if chunk_tokens is not None else (),
        )
        if chunk_tokens is not None:
            self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self.compressor._mask_width = capacity

        # Rope tables for every position this state can serve, so forward gathers instead of building.
        # The slab needs one row per TOKEN and the compressor one per ENTRY; both are sized on chunk-padded
        # capacity because pos_padded covers [kv_actual, kv_actual + chunk) even for the last chunk.
        if chunk_tokens is not None:
            rate, sl = self.compressor.compress_rate, chunk // self.sp_factor
            tokens = -(-int(max_seq_len) // chunk) * chunk + chunk
            self._slab_rope = self._build_rope_table(tokens, 1)
            self._slab_index = self._rope_index_base(sl)
            self.compressor._entry_rope = self.compressor._build_rope_table(-(-tokens // rate), rate)
            self.compressor._entry_index = self.compressor._rope_index_base(sl // rate)
        return TtHCAState(
            compressed_kv=self._from_torch(torch.zeros(batch, 1, capacity, self.head_dim)),
            sliding_carry=self._from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
            tail=self._from_torch(torch.zeros(batch, 1, ttnn.TILE_SIZE, self.head_dim)),
        )

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCA":
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype")
        compressor = TtHCACompressor.from_reference(
            device, reference.compressor, config, **{k: kwargs[k] for k in compressor_keys if k in kwargs}
        )
        return cls(
            device,
            compressor=compressor,
            q_a_proj_weight=reference.q_a_proj.weight,
            q_a_norm_weight=reference.q_a_norm.weight,
            q_b_proj_weight=reference.q_b_proj.weight,
            kv_proj_weight=reference.kv_proj.weight,
            kv_norm_weight=reference.kv_norm.weight,
            sinks=reference.sinks,
            o_a_proj_weight=reference.o_a_proj.weight,
            o_b_proj_weight=reference.o_b_proj.weight,
            rotary_emb=reference.compressor.rotary_emb,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            sliding_window=config.sliding_window,
            o_groups=config.o_groups,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def _q_stem(self, hidden_states, cos, sin):
        """[B, 1, S/sp, hidden/tp] -> q [B, num_heads/tp, S/sp, head_dim]. ``cos``/``sin`` cover the padded
        slab and are built once per call -- the q stem, the kv stem and the output un-rope all want the same
        rotation, and rebuilding it costs ~2.9 ms of host time each (rotary_emb plus two uploads)."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        num_heads_local = self.num_heads // self.tp_factor

        q = ttnn.linear(hidden_states, self.wq_a, memory_config=self.memory_config)

        # Row-parallel -> partial sums; all-reduce rebuilds the full q_lora, replicated across TP.
        if self.tp_factor > 1:
            q = ttnn.experimental.reduce_scatter_minimal_async(
                q,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
            q = ttnn.experimental.all_gather_async(
                q,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )

        q = ttnn.rms_norm(q, weight=self.q_a_norm_weight, epsilon=self.rms_norm_eps)
        q = ttnn.linear(q, self.wq_b, memory_config=self.memory_config)

        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=num_heads_local,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=self.memory_config,
        )
        q = ttnn.rms_norm(q, weight=self.q_b_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(q, [0, 0, 0, 0], [batch, num_heads_local, seq_len, nope_dim])
        rope = ttnn.slice(q, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_len, self.head_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _kv_stem(self, hidden_states, cos, sin):
        """[B, 1, S/sp, hidden/tp] -> single-head sliding_kv [B, 1, S/sp, head_dim], TP-replicated.
        K == V in V4. Returns the full S; the sliding-window truncation is a chunked-prefill concern."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)

        # kv_proj is contraction(row)-parallel like wq_a -> partial-sum single-head KV; TP all-reduce
        # (reduce_scatter + all_gather) rebuilds the full head_dim, replicated across TP.
        if self.tp_factor > 1:
            kv = ttnn.experimental.reduce_scatter_minimal_async(
                kv,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
            kv = ttnn.experimental.all_gather_async(
                kv,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )

        kv = ttnn.rms_norm(kv, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(kv, [0, 0, 0, 0], [batch, 1, seq_len, nope_dim])
        rope = ttnn.slice(kv, [0, 0, 0, nope_dim], [batch, 1, seq_len, self.head_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    def _attention(
        self,
        q,
        sliding_kv,
        compressed_kv,
        mask_block,
        cos,
        sin,
        carry=None,
        kv_actual: int = 0,
        real_len: int | None = None,
        need_carry: bool = True,
    ):
        """SP-gathers sliding_kv to full S, concats ``[carry | sliding | compressed | pad]``, runs
        per-chip SDPA, then undoes V's RoPE. ``carry`` is the previous chunk's raw KV tail (None in
        single-shot); ``kv_actual`` the global position of this chunk's first query.

        Returns ``(attn, next_carry)``. The carry can only be taken here: before the gather those rows
        live on the last SP chip alone."""
        batch, seq_local = q.shape[0], q.shape[2]
        seq_len = seq_local * self.sp_factor  # global query/main-key length
        num_heads_local = self.num_heads // self.tp_factor
        carry_len = 0 if carry is None else carry.shape[2]

        # Per-chip SDPA needs every key on every chip; compressed_kv is already replicated.
        if self.sp_factor > 1:
            sliding_kv = ttnn.experimental.all_gather_async(
                sliding_kv,
                dim=2,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.sp_axis,
            )

        # The next chunk's sliding window reaches back into this one, so the carry has to be this chunk's
        # last REAL keys -- rows [real_len - sliding_window, real_len) -- not the last rows of the padded
        # slab. Those differ whenever real_len < seq_len, and a plain slice cannot express it: slice_start
        # is hashed (slice_device_operation.cpp:306), so a new real_len would compile a new program.
        # The tensor-args overload takes the start from a device tensor instead, and hashes only its shape
        # (:327), so every offset reuses one program. Verified on 8x4: +1 program on the first call, +0 for
        # every offset after.
        #
        # Skipped entirely when no chunk follows: single-shot's state is thrown away, so the slice and the
        # index table it needs would be pure cost -- and alloc_state runs inside forward there, which put
        # the table's tilize/typecast straight into the measured region.
        next_carry = None
        if need_carry:
            start, end = self._carry_index[self._carry_key(real_len if real_len is not None else seq_len)]
            next_carry = ttnn.slice(sliding_kv, start, end, slice_dim=2, num_devices=seq_len // self.sliding_window)

        # Pad Sk to a multiple of 32 explicitly: SDPA tile-pads a non-aligned Sk with zeros, and a
        # provided mask's pad columns default to 0 (= attend), which pollutes the softmax. The mask
        # -infs the columns we add here.
        sk = carry_len + seq_len + compressed_kv.shape[2]
        sk_pad = ((sk + 31) // 32) * 32
        parts = [sliding_kv, compressed_kv] if carry is None else [carry, sliding_kv, compressed_kv]
        if self._kv_pad is not None:
            parts.append(self._kv_pad)
        kv = ttnn.concat(parts, dim=2)

        # Only the compressed columns move between chunks, and their column range is fixed, so the mask is
        # persistent and this overwrites that range in place. A slice offset lands in the program hash, but
        # this one never changes, so one program serves every chunk.
        mask = self._mask
        rows = mask_block.shape[2]
        carry_cols = self._carry_cols[kv_actual == 0]
        ttnn.experimental.slice_write(
            carry_cols, mask, start=[0, 0, 0, 0], end=[batch, 1, rows, carry_cols.shape[3]], step=[1, 1, 1, 1]
        )
        ttnn.experimental.slice_write(
            mask_block,
            mask,
            start=[0, 0, 0, self._mask_col],
            end=[batch, 1, rows, self._mask_col + mask_block.shape[3]],
            step=[1, 1, 1, 1],
        )

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv,
            kv,
            attn_mask=mask,
            is_causal=False,
            scale=self.scaling,
            attention_sink=self.sinks_sdpa,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
        )

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(attn, [0, 0, 0, 0], [batch, num_heads_local, seq_local, nope_dim])
        rope = ttnn.slice(attn, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_local, self.head_dim])
        # Undoing V's RoPE is the conjugate rotation, so the same cos serves and only sin flips. Negating
        # on device costs one elementwise op instead of a second host build and upload.
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1), next_carry

    def _write_compressed(self, state, new_entries, n_new, keep_tail=True):
        """Append this call's entries to the cache at row ``state.entry_count``.

        One entry is one row, so the append offset advances by ``chunk/compress_rate`` per chunk --
        tile-aligned only when that is a multiple of TILE_SIZE, i.e. chunks of 4096 tokens. That rules
        out ``fill_cache_for_user_``, which requires the alignment
        (update_cache_device_operation.cpp:96), for the runner's default chunk_size of 5120.

        The whole padded width is written, not just the real entries: the width is then constant across
        chunks, and the attention mask -infs everything past ``total_entries`` anyway.

        Writing at the tile boundary below the append offset and supplying that tile's whole content is
        what keeps the offset tile-aligned; ``state.tail`` carries the open tile's entries across chunks."""
        width = new_entries.shape[2]
        assert state.entry_count + width <= state.compressed_kv.shape[2], (
            f"compressed cache full: writing {width} entries at {state.entry_count} exceeds capacity "
            f"{state.compressed_kv.shape[2]}; allocate the state with a larger max_seq_len"
        )
        self._write_tail_tile(state, new_entries, width, n_new, keep_tail)

    def _write_tail_tile(self, state, new_entries, width, n_new, keep_tail):
        """fill_cache keeps update_idx out of its program hash but needs it tile-aligned, so this writes at
        the tile boundary below entry_count and supplies that tile's whole content -- the entries already
        in it come from ``state.tail``, which carries them across chunks right-aligned.

        Right-aligning the tail is what makes one uniform shift enough: both it and ``new_entries`` then sit
        in ``src`` under the same rule, src row m holding entry ``entry_count - TILE_SIZE + m``, so placing
        the merged block is a single shift by ``TILE_SIZE - r_e``.

        The shift is a matmul against a one-hot matrix rather than a slice: a slice folds its offset into
        its program hash, a matmul carries it as data. A matmul also never addresses a row -- it sums over
        all of them -- so a shift that crosses a tile boundary is not a special case."""
        tile = ttnn.TILE_SIZE
        f, r_e = divmod(state.entry_count, tile)
        shift, take = self._tail_tile_matrices(r_e, width, n_new if keep_tail else None)

        src = ttnn.concat([state.tail, new_entries], dim=2)  # [B, 1, tile + width, head_dim]
        merged = ttnn.matmul(shift, src, memory_config=self.memory_config)
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, merged, 0, update_idx=f * tile)
        # The tail is only ever read by the NEXT chunk's write, so single-shot skips it -- and with it
        # the one-hot table, which alloc_state would otherwise build inside forward.
        if keep_tail:
            state.tail = ttnn.matmul(take, merged, memory_config=self.memory_config)

    def _build_tail_tile_matrices(self, width, r_es, take_steps):
        """Every one-hot pair the write can need, for one slab width. Called from alloc_state, never from
        forward: building a pair costs ~0.9 ms of host time that device perf does not measure, and forward
        is required to touch no host tensors.

        A pair is 24 KB for chunk 5120 (18 KB shift + 6 KB take, both independent of head_dim), so the
        whole TILE_SIZE set is 768 KB per chip -- negligible beside the compressed cache itself."""
        tile, buf = ttnn.TILE_SIZE, _buf_for_width(width)
        for r_e in r_es:
            # merged row i takes src row i + (tile - r_e), for the r_e + width rows that carry entries;
            # the rest stay zero, so nothing reads past src.
            rows = torch.arange(r_e + width)
            shift = torch.zeros(1, 1, buf, tile + width)
            shift[0, 0, rows, rows + (tile - r_e)] = 1.0
            self._shift_cache[(r_e, width)] = self._from_torch(shift)

        # Lifting the next open tile back out of merged is keyed on r_e + n_new, NOT r_e + width: the
        # whole padded width is written, but entry_count only advances by the real entries, so a chunk
        # with real_len < chunk_size leaves the tail right-aligned on a count the state never reaches.
        # Rows before the open tile's first live entry stay zero -- that is what lets the next chunk's
        # shift skip them by construction instead of masking them.
        for s in take_steps:
            rows = torch.arange(tile - s % tile, tile)
            take = torch.zeros(1, 1, tile, buf)
            take[0, 0, rows, rows + s - tile] = 1.0
            self._take_cache[s] = self._from_torch(take)

    def _carry_key(self, real_len):
        """The carry index is tabulated per whole compression window. A ragged chunk rounds down, which is
        sound because it can only be the final one -- appending after it is rejected -- so its carry is
        never read."""
        rate = self.compressor.compress_rate
        return max(rate, (int(real_len) // rate) * rate)

    def _build_carry_index(self, chunk_tokens):
        """start/end index tensors for the carry slice, one pair per real_len the chunk can carry. Built
        here rather than in forward, which must touch no host tensors; each pair is 8 uint32s."""
        rate, sw = self.compressor.compress_rate, self.sliding_window
        assert sw % ttnn.TILE_SIZE == 0 and rate % sw == 0, (
            f"the carry slice needs a tile-aligned start and one whole window per step: sliding_window "
            f"{sw} must be a multiple of {ttnn.TILE_SIZE} and divide compress_rate {rate}"
        )

        def idx(vals):
            return ttnn.from_torch(
                torch.tensor(vals, dtype=torch.int32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh else None,
            )

        for real_len in range(rate, int(chunk_tokens) + 1, rate):
            self._carry_index[real_len] = (idx([0, 0, real_len - sw, 0]), idx([1, 1, real_len, self.head_dim]))

    def _tail_tile_matrices(self, r_e, width, n_new):
        """Lookup only. A miss means alloc_state was given a chunk width this call does not match, and
        building here would put host work back in forward -- so fail instead. ``n_new`` is None when the
        caller does not keep the tail, in which case no take matrix exists to look up."""
        shift = self._shift_cache.get((r_e, width))
        take = self._take_cache.get(r_e + n_new) if n_new is not None else None
        assert shift is not None and (take is not None or n_new is None), (
            f"no tail-tile one-hot for (r_e={r_e}, width={width}, n_new={n_new}); alloc_state built widths "
            f"{sorted({w for _, w in self._shift_cache})} -- pass chunk_tokens matching the slab width"
        )
        return shift, take

    def _o_proj(self, attn):
        """[B, num_heads/tp, S/sp, head_dim] -> [B, 1, S/sp, hidden/tp], the block's own input layout."""
        batch, _, seq_len, _ = attn.shape
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        groups_local = self.o_groups // self.tp_factor

        x = ttnn.reshape(attn, [groups_local, attn.shape[1] // groups_local, seq_len, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x, memory_config=self.memory_config)
        x = ttnn.reshape(x, [batch, groups_local, seq_len, in_per_group])

        grouped = ttnn.linear(x, self.wo_a, memory_config=self.memory_config)  # [B, groups_local, S, o_lora_rank]
        o_lora_rank = grouped.shape[-1]
        grouped = ttnn.concat(
            [ttnn.slice(grouped, [0, g, 0, 0], [batch, g + 1, seq_len, o_lora_rank]) for g in range(groups_local)],
            dim=-1,
        )  # [B, 1, S, groups_local*o_lora_rank]

        out = ttnn.linear(grouped, self.wo_b, memory_config=self.memory_config)  # partial-sum [B,1,S,hidden]

        # Reduce-scatter, not a full all-reduce: it both sums the partials and slices to hidden/tp,
        # which is already the layout the next block wants.
        if self.tp_factor > 1:
            out = ttnn.experimental.reduce_scatter_minimal_async(
                out,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
        return out

    def forward(
        self,
        hidden_states,
        position_ids: torch.Tensor,
        seq_len_actual: int | None = None,
        state: TtHCAState | None = None,
    ):
        """One chunk: [B, 1, S_pad/sp, hidden/tp] in and out; the caller keeps the first S_real rows.

        ``position_ids`` holds this chunk's GLOBAL positions and ``seq_len_actual`` its real pre-pad
        length. ``state`` is advanced in place. ``state=None`` is single-shot, which is just a one-chunk
        prefill -- a throwaway state is allocated here so there is no separate compute path for it."""
        batch = hidden_states.shape[0]
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        compress_rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        assert batch == 1, f"HCA prefill expects batch 1, got {batch}"

        assert real_len >= compress_rate, (
            f"HCA prefill needs at least one full compression window: got seq_len {real_len} < "
            f"compress_rate {compress_rate}"
        )

        single_shot = state is None
        if single_shot:
            # Sized on the padded slab, not real_len: the write covers the whole padded width, so that is
            # what the tail-tile one-hot has to be built for.
            state = self.alloc_state(seq_pad_global, batch=batch)

        n_new = real_len // compress_rate
        total_entries = state.entry_count + n_new
        capacity = state.compressed_kv.shape[2]
        assert total_entries <= capacity, (
            f"compressed cache full: {total_entries} entries > capacity {capacity}; allocate the state "
            f"with a larger max_seq_len"
        )
        # Checked on tokens, not entry_count: a dropped partial window is invisible there (4097 tokens
        # still lands on 32 entries) and the next chunk would start at the wrong window position.
        #
        # A non-final chunk that ends mid-window strands the leftover tokens: they never join a
        # compression window, and the next chunk's windows would start off the global grid. So every
        # chunk but the last has to carry whole windows, which keeps kv_actual a multiple of
        # compress_rate and every chunk's slab aligned to the window grid the compressor assumes.
        assert state.kv_actual % compress_rate == 0, (
            f"cannot append after a chunk with {state.kv_actual % compress_rate} leftover tokens; only "
            f"the final chunk may be ragged, non-final chunks must be a multiple of {compress_rate}"
        )

        # One rotation for the whole padded slab: the q stem, the kv stem and the output un-rope all want
        # it, and a rebuild is ~2.9 ms of host time. Chunked runs gather it from the table alloc_state
        # built; single-shot rotates once, so a table it would have to build inside forward is pure cost.
        if self._slab_rope is not None:
            cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
        else:
            pos_padded = position_ids
            if position_ids.shape[1] < seq_pad_global:
                tail = torch.arange(1, seq_pad_global - position_ids.shape[1] + 1).unsqueeze(0)
                pos_padded = torch.cat([position_ids, position_ids[:, -1:] + tail], dim=1)
            cos, sin = self._cos_sin(pos_padded)
        q = self._q_stem(hidden_states, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)
        new_entries, mask_block = self.compressor(
            hidden_states,
            position_ids,
            seq_len_actual=seq_len_actual,
            first_window_position=state.entry_count * compress_rate,
            total_entries=total_entries,
        )
        # Attention then reads the WHOLE cache every chunk, so its shape stays constant and the mask
        # -infs everything past total_entries.
        self._write_compressed(state, new_entries, n_new, keep_tail=not single_shot)

        attn, next_carry = self._attention(
            q,
            sliding_kv,
            state.compressed_kv,
            mask_block,
            cos,
            sin,
            carry=state.sliding_carry,
            kv_actual=state.kv_actual,
            real_len=real_len,
            need_carry=not single_shot,
        )

        state.entry_count = total_entries
        state.kv_actual += real_len
        state.sliding_carry = next_carry
        return self._o_proj(attn)
