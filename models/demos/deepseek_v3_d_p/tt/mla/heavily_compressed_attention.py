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


def _cache_write_rows(chunk_entries: int) -> int:
    """How many cache rows one write covers: this chunk's new entries plus the 0..31 rows already sitting in
    the last tile. The cache is written a whole tile at a time, so this rounds up to whole tiles -- 64 rows
    for a 4096-token chunk, 96 for 5120."""
    return -(-(ttnn.TILE_SIZE - 1 + chunk_entries) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _rope_table_tokens(max_seq_len: int, chunk_tokens: int) -> int:
    """How many token positions a rope table has to cover: the context rounded up to whole chunks, plus one
    chunk more, because the last chunk's padded slab runs past the end of the real context."""
    return -(-int(max_seq_len) // chunk_tokens) * chunk_tokens + chunk_tokens


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
        torch_weight = weight.detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
        return self._from_torch(
            torch_weight, mesh_mapper=self._mesh_mapper(tp_dim=tp_shard_dim), dtype=self.weights_dtype
        )

    def _from_torch(self, x: torch.Tensor, mesh_mapper=None, dtype=None, layout=ttnn.TILE_LAYOUT, on_device=True):
        """Replicated across the mesh unless a mapper is given. ``on_device=False`` leaves it on host, which
        is what ``copy_host_to_device_tensor`` takes as its source."""
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

    def _mesh_mapper(self, sp_dim: int | None = None, tp_dim: int | None = None):
        """How one host tensor lands on the mesh: split ``sp_dim`` across the SP axis, ``tp_dim`` across the
        TP axis, replicate along whichever axis is left. Both None means fully replicated; a single device
        needs no mapper."""
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

    def _scalar_buffer(self, dtype, shape=(1, 1, 1, 1), layout=ttnn.TILE_LAYOUT):
        """A one-element device tensor, allocated once and then overwritten by ``_push_scalar``. Values
        that change every chunk live here so forward never has to allocate."""
        return self._from_torch(torch.zeros(*shape, dtype=torch.int32), dtype=dtype, layout=layout)

    def _push_scalar(self, buf, v):
        """Overwrite an existing device buffer with one value -- the only thing forward still sends from
        host."""
        host_dtype = torch.float32 if buf.dtype == ttnn.float32 else torch.int32
        host = self._from_torch(
            torch.full(tuple(buf.shape), v, dtype=host_dtype), dtype=buf.dtype, layout=buf.layout, on_device=False
        )
        ttnn.copy_host_to_device_tensor(host, buf)
        return buf

    def _build_rope_table(self, count: int, stride: int):
        """cos/sin for every position this layer can ever rotate by, built once. forward only gathers rows
        out of it.

        Kept whole on every chip rather than split, because which rows a chip needs moves with the chunk:
        chip c reads from ``kv_actual + c*local``, which walks out of any fixed slice. A row is 128 B, so
        the table is 14 MB for a 56K-token context -- cheap enough to keep whole."""
        positions = (torch.arange(count) * stride).unsqueeze(0)
        cos, sin = self.rotary_emb(torch.zeros(1), position_ids=positions.to(torch.long), layer_type="compress")
        pair = []
        for t in (cos, sin):
            t = t.repeat_interleave(2, dim=-1)  # [1, count, rope_head_dim]
            pair.append(self._from_torch(t))
        return tuple(pair)

    def _rope_index_base(self, rows: int):
        """The gather index is ``base + c*rows + r``. This builds the constant ``c*rows + r`` half, split so
        chip c holds its own row, plus an empty buffer for ``base``. Only ``base`` moves between chunks, so
        forward pushes 4 bytes instead of computing the whole index on host."""
        const = self._from_torch(
            torch.arange(self.sp_factor * rows, dtype=torch.int32).view(self.sp_factor, rows),
            mesh_mapper=self._mesh_mapper(sp_dim=0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return const, self._scalar_buffer(ttnn.uint32, shape=(1, 1), layout=ttnn.ROW_MAJOR_LAYOUT)

    def _rope_index(self, index_base, base: int):
        """This chunk's gather index, built on device from the constant half plus a 4-byte base."""
        const, buf = index_base
        return ttnn.add(const, self._push_scalar(buf, base))

    def _rope_gather(self, table, index):
        """Pick this chunk's rows out of the table. ttnn.embedding looks rows up by index, so unlike
        ttnn.slice it needs no tile-aligned start -- the compressor's start ``entry_count + c*n_windows``
        rarely is one. The index is data, not part of the program, so every chunk reuses one program."""
        out = []
        for t in table:
            g = ttnn.embedding(index, t, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            out.append(ttnn.reshape(g, [1, 1, g.shape[-2], g.shape[-1]]))
        return tuple(out)


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
        # All three come from alloc_tables, which every caller has to run before forward.
        self._entry_rope = None
        self._entry_index = None
        self._mask_consts = None

    def alloc_tables(self, max_seq_len: int, chunk_tokens: int, mask_width: int):
        """Build every host tensor forward would otherwise need: the rope table it gathers from, the
        constant half of the gather index, and the mask block's index vectors.

        ``mask_width`` is the compressed cache's capacity, which only the block knows -- taking it as an
        argument keeps that formula in one place. It becomes the width of the mask block forward returns."""
        tokens = _rope_table_tokens(max_seq_len, chunk_tokens)
        self._entry_rope = self._build_rope_table(-(-tokens // self.compress_rate), self.compress_rate)
        self._entry_index = self._rope_index_base(chunk_tokens // self.sp_factor // self.compress_rate)
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

    def _build_mask_consts(self, seq_global: int, width: int):
        """The parts of the compressed mask block that never change, built once by ``alloc_tables``.

        ``thr[j] = (j+1)//rate`` counts the entries this chunk has finished by query row j; ``ic[j] = j``
        and ``w[k] = k`` are plain index vectors; ``ec`` and ``rl`` are the two numbers that change every
        chunk (entries already in the cache, real query rows). ``seq`` is not a tensor -- it records how
        many query rows these cover, which ``_mask_block`` checks its call against.

        ``thr`` and ``ic`` count GLOBAL query rows, so they are SP-sharded like the mask they feed. ``w``
        counts cache columns, which are the same on every chip, so it is replicated.

        float32 and not bfloat16: these are whole numbers up to width-1, and bfloat16 stops being exact
        above 256 -- measured, it misses 640 of 328K mask elements at 300 entries."""
        sp_mapper = self._mesh_mapper(sp_dim=2)
        rate = self.compress_rate
        return {
            "seq": seq_global,
            "thr": self._from_torch(
                ((torch.arange(seq_global) + 1) // rate).float().view(1, 1, seq_global, 1),
                sp_mapper,
                dtype=ttnn.float32,
            ),
            "ic": self._from_torch(
                torch.arange(seq_global).float().view(1, 1, seq_global, 1), sp_mapper, dtype=ttnn.float32
            ),
            "w": self._from_torch(torch.arange(width).float().view(1, 1, 1, width), dtype=ttnn.float32),
            "ec": self._scalar_buffer(ttnn.float32),
            "rl": self._scalar_buffer(ttnn.float32),
        }

    def _mask_block(self, seq: int, first_window_position: int, seq_len_actual: int):
        """The mask's compressed columns, built on device: 0 where a query may attend an entry, -inf else.

        Query row j may attend entry w while ``w < ec + thr[j]`` (both from ``_build_mask_consts``), so the
        whole block is one broadcast comparison. The zeros over older entries, the staircase over this
        chunk's, and the -inf tail over entries not written yet all come out of that single condition --
        which is what lets the width stay at the full cache capacity and the shape never change.

        ``log`` turns the 1/0 of a comparison into the 0/-inf an additive mask needs: ttnn gives
        log(1) = 0 and log(0) = -inf exactly."""
        rate = self.compress_rate
        # seq is this chip's row count; the index vectors are global and sharded, so each chip ends up
        # comparing its own rows.
        seq_global = seq * self.sp_factor
        c = self._mask_consts
        assert c is not None and c["seq"] == seq_global, (
            f"mask constants cover {None if c is None else c['seq']} query rows but this call has "
            f"{seq_global}; alloc_tables has to be given the slab forward is called with"
        )
        within = ttnn.lt(c["w"], ttnn.add(c["thr"], self._push_scalar(c["ec"], first_window_position // rate)))
        live = ttnn.lt(c["ic"], self._push_scalar(c["rl"], seq_len_actual))  # pad query rows attend nothing
        return ttnn.typecast(ttnn.log(ttnn.multiply(within, live)), self.dtype)

    def forward(
        self,
        hidden_states,
        seq_len_actual: int | None = None,
        first_window_position: int = 0,
    ):
        """[B, 1, S_pad/sp, hidden/tp] -> compressed_kv [B, 1, S_pad/compress_rate, head_dim] (replicated),
        plus the mask's compressed columns [B, 1, S_pad, mask_width].

        ``seq_len_actual`` is the real pre-pad length (``None`` = the whole padded slab).
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

        n_windows = seq_len // self.compress_rate  # windows this chip owns
        t_real = seq_len_actual // self.compress_rate
        # prepare_input pads the sequence to a multiple of compress_rate * sp_factor, so every chip is
        # left with at least one whole window. Asserted and not handled: the empty-slab branch this
        # replaces was the last thing building a host tensor inside forward.
        assert n_windows > 0, (
            f"each chip needs at least one whole compression window: {seq_len} rows is under "
            f"compress_rate {self.compress_rate}; run prepare_input on the hidden states first"
        )
        gate = ttnn.reshape(gate, [batch, n_windows, self.compress_rate, self.head_dim])
        gate = ttnn.add(gate, self.position_bias)
        weights = ttnn.softmax(gate, dim=2, numeric_stable=True)

        kv = ttnn.reshape(kv, [batch, n_windows, self.compress_rate, self.head_dim])
        pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)

        compressed = ttnn.reshape(pooled, [batch, 1, n_windows, self.head_dim])
        compressed = ttnn.rms_norm(compressed, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)

        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(compressed, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
        rope = ttnn.slice(compressed, [0, 0, 0, nope_dim], [batch, 1, n_windows, self.head_dim])
        # Index is in ENTRIES: table row k carries position k*compress_rate, so the window's position
        # has to be divided back down.
        idx = self._rope_index(self._entry_index, first_window_position // self.compress_rate)
        cos, sin = self._rope_gather(self._entry_rope, idx)
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        compressed_kv = ttnn.concat([nope, rope], dim=-1)

        if self.sp_factor > 1:
            compressed_kv = ttnn.experimental.all_gather_async(
                compressed_kv,
                dim=2,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.sp_axis,
            )

        # Spans the whole cache, not just this call's entries: the queries can see earlier chunks' too,
        # and a fixed width is what keeps the shape constant.
        mask_block = None
        if seq_len_actual > 1 and t_real > 0:
            mask_block = self._mask_block(seq_len, first_window_position, seq_len_actual)

        return compressed_kv, mask_block


class TtHCAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtHCA.forward``.

    The device tensors keep a FIXED shape for the whole prefill and only their contents advance --
    that is what lets one compiled program serve every chunk. The counters say how much is real;
    the attention mask -infs the rest."""

    def __init__(self, compressed_kv, sliding_carry, tail, max_seq_len):
        self.compressed_kv = compressed_kv  # [B, 1, compressed_capacity, head_dim]
        self.sliding_carry = sliding_carry  # [B, 1, sliding_window, head_dim]
        # The cache's last tile is usually only partly filled. Its entries sit here, right-aligned, so the
        # next write can place them and the new entries with a single shift.
        self.tail = tail  # [B, 1, TILE_SIZE, head_dim]
        self.max_seq_len = int(max_seq_len)
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
        self.sinks_sdpa = self._from_torch(sinks_host, mesh_mapper=self._mesh_mapper(tp_dim=1))

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
        self.wo_a = self._from_torch(o_a_grouped, mesh_mapper=self._mesh_mapper(tp_dim=1), dtype=self.weights_dtype)
        self.wo_b = self._to_tt_linear_weight(o_b_proj_weight, tp_shard_dim=2)

        self.trans_mat = self._from_torch(get_rot_transformation_mat())
        # Everything below comes from alloc_state, which every caller has to run before forward.
        self._shift_cache = {}
        self._take_cache = {}
        self._carry_index = {}
        self._slab_rope = None
        self._slab_index = None
        self._mask = None  # persistent additive mask; forward overwrites only the moving columns
        self._mask_col = None
        self._carry_cols = None
        self._kv_pad = None  # zero rows that bring Sk up to a tile multiple

    def _build_masks(self, seq_global: int, cap: int):
        """The persistent additive mask over the key layout ``[carry | chunk | compressed | pad]``. Carry,
        chunk and pad columns are the same for every chunk and are written here; forward overwrites only
        the compressed columns, always at the same offset, so nothing recompiles.

        The carry columns come in two versions because the first chunk has no history: its carry holds
        zeros for positions that do not exist yet, so those columns must be -inf. Nothing else differs --
        in ``allowed = (j >= 0) & (j <= i) & (i - j < sw)`` the offset kv_actual cancels out of the last
        two conditions.

        Built from four small index vectors, so no large host tensor is created. They are float32 for the
        reason ``_build_mask_consts`` gives."""
        carry, sw = self.sliding_window, self.sliding_window
        raw = carry + seq_global
        sk_pad = -(-(raw + cap) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        sp_mapper = self._mesh_mapper(sp_dim=2)

        ic = self._from_torch(torch.arange(seq_global).float().view(1, 1, seq_global, 1), sp_mapper, dtype=ttnn.float32)
        ic_lo = self._from_torch(
            (torch.arange(seq_global) - sw).float().view(1, 1, seq_global, 1), sp_mapper, dtype=ttnn.float32
        )
        jc = self._from_torch((torch.arange(raw) - carry).float().view(1, 1, 1, raw), dtype=ttnn.float32)

        # j <= i  and  i - j < sw, both with kv_actual cancelled
        sliding = ttnn.typecast(ttnn.log(ttnn.multiply(ttnn.le(jc, ic), ttnn.gt(jc, ic_lo))), self.dtype)

        zero_seq = ttnn.multiply(ic, 0.0)
        blank = ttnn.typecast(
            ttnn.add(zero_seq, self._from_torch(torch.zeros(1, 1, 1, cap), dtype=ttnn.float32)), self.dtype
        )
        parts = [sliding, blank]
        pad_w = sk_pad - raw - cap
        if pad_w:
            parts.append(
                ttnn.typecast(
                    ttnn.log(ttnn.add(zero_seq, self._from_torch(torch.zeros(1, 1, 1, pad_w), dtype=ttnn.float32))),
                    self.dtype,
                )
            )
        self._mask = ttnn.concat(parts, dim=3)
        self._mask_col = raw
        # SDPA pads a non-tile-multiple Sk with zeros, and a supplied mask reads those columns as 0, which
        # means "attend". So the kv side gets real zero rows and the mask -infs their columns.
        sk = raw + cap
        self._kv_pad = self._from_torch(torch.zeros(1, 1, sk_pad - sk, self.head_dim)) if sk_pad > sk else None

        # Both versions of the [seq, carry] slab are kept and forward writes the right one. Keeping a
        # second full mask instead cost 1.1 ms on device for the same information.
        self._carry_cols = {
            False: ttnn.slice(sliding, [0, 0, 0, 0], [1, 1, sliding.shape[2], carry]),
            True: ttnn.typecast(
                ttnn.log(ttnn.multiply(zero_seq, self._from_torch(torch.zeros(1, 1, 1, carry), dtype=ttnn.float32))),
                self.dtype,
            ),
        }

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None) -> TtHCAState:
        """Size the state once for the longest context this layer will serve, so its shape is fixed for
        every chunk. ``max_seq_len`` is the longest context to serve, ``chunk_tokens`` the slab width
        forward will be called with (defaults to one chunk).
        Contents start zeroed and nothing is read before it is written -- the mask -infs everything past
        ``entry_count`` / ``kv_actual``, including chunk 0's empty carry.

        Every host tensor this layer will ever need is built here, which is what leaves forward with none.
        So the caller has to own the state: a prefill of one chunk allocates the same way a long one
        does."""
        entries = -(-int(max_seq_len) // self.compressor.compress_rate)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE  # cache writes land on tile boundaries
        # A write always rewrites whole tiles, so the last one can reach past the entries themselves.
        # ``chunk_tokens`` sizes that headroom.
        chunk = chunk_tokens or max_seq_len
        align = self.compressor.compress_rate * self.sp_factor
        assert chunk % align == 0, (
            f"the slab is {chunk} wide, which is not a multiple of compress_rate * sp_factor "
            f"({self.compressor.compress_rate} * {self.sp_factor} = {align}); every chip's share of it has "
            f"to end on a compression-window boundary. prepare_input rounds a raw prompt up to that."
        )
        width = -(-int(chunk) // self.compressor.compress_rate)
        capacity += _cache_write_rows(width)
        # Every one-hot the write can ever need: with chunks of differing real length r_e reaches every
        # value, so the whole TILE_SIZE set is built.
        self._build_tail_tile_matrices(width)
        self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self.compressor.alloc_tables(max_seq_len, chunk, capacity)

        # One rope table per state, so forward only gathers. This one has a row per TOKEN; the compressor
        # builds its own, with a row per ENTRY.
        self._slab_rope = self._build_rope_table(_rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self._rope_index_base(chunk // self.sp_factor)
        return TtHCAState(
            compressed_kv=self._from_torch(torch.zeros(batch, 1, capacity, self.head_dim)),
            sliding_carry=self._from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
            tail=self._from_torch(torch.zeros(batch, 1, ttnn.TILE_SIZE, self.head_dim)),
            max_seq_len=max_seq_len,
        )

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCA":
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype", "memory_config")
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
        slab and are built once per call: both stems and the output un-rope want the same rotation, and
        building it again costs ~2.9 ms of host time."""
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
        carry,
        kv_actual: int,
        real_len: int,
    ):
        """SP-gathers sliding_kv to full S, concats ``[carry | sliding | compressed | pad]``, runs
        per-chip SDPA, then undoes V's RoPE. ``carry`` is the previous chunk's raw KV tail, zeros on the
        first chunk; ``kv_actual`` the global position of this chunk's first query.

        Returns ``(attn, next_carry)``. The carry can only be taken here: before the gather those rows
        live on the last SP chip alone."""
        batch, seq_local = q.shape[0], q.shape[2]
        seq_len = seq_local * self.sp_factor  # global query/main-key length
        num_heads_local = self.num_heads // self.tp_factor

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
        # slab, which differ as soon as real_len < seq_len. A plain slice cannot do it: its start value is
        # part of the program, so every new real_len would compile another
        # one. Taking the start from a device tensor keeps only its shape in the program, so one program
        # serves every offset -- measured on 8x4 as +1 program on the first call and +0 after.
        #
        # Taken on the last chunk too, whose carry nobody reads: skipping it saves one op out of ~100.
        start, end = self._carry_index[self._carry_key(real_len)]
        next_carry = ttnn.slice(sliding_kv, start, end, slice_dim=2, num_devices=seq_len // self.sliding_window)

        # Pad Sk to a multiple of 32 by hand: SDPA would pad it with zeros, and the mask reads its own pad
        # columns as "attend", which would pollute the softmax. The mask -infs the columns added here.
        parts = [carry, sliding_kv, compressed_kv]
        if self._kv_pad is not None:
            parts.append(self._kv_pad)
        kv = ttnn.concat(parts, dim=2)

        # Only the compressed columns move between chunks, and always over the same column range, so the
        # mask is built once and this overwrites that range in place. The offset is part of the program,
        # but it never changes, so one program serves every chunk.
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
        # Undoing V's RoPE is the same rotation with the sign of sin flipped, so cos is reused and the
        # negation is one op on device instead of another host build.
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1), next_carry

    def _write_compressed(self, state, new_entries, n_new):
        """Append this call's entries to the cache at row ``state.entry_count``.

        One entry is one row, so the offset advances by ``chunk/compress_rate`` per chunk -- a multiple of
        TILE_SIZE only for chunks of 4096 tokens. ``fill_cache_for_user_`` needs that alignment
        which a 5120-token chunk does not give it, so the write goes through ``_write_tail_tile`` instead.

        The whole padded width is written, not just the real entries, so the width is the same for every
        chunk; the mask -infs everything past ``total_entries`` anyway."""
        width = new_entries.shape[2]
        tile_start = (state.entry_count // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        write_end = tile_start + _cache_write_rows(width)
        assert write_end <= state.compressed_kv.shape[2], (
            f"compressed cache full: writing rows [{tile_start}, {write_end}) exceeds capacity "
            f"{state.compressed_kv.shape[2]}; allocate the state with a larger max_seq_len"
        )
        self._write_tail_tile(state, new_entries, width, n_new)

    def _write_tail_tile(self, state, new_entries, width, n_new):
        """fill_cache leaves update_idx out of its program but needs it tile-aligned, so this writes at the
        tile boundary below entry_count and supplies that whole tile -- the entries already in it come from
        ``state.tail``.

        The tail being right-aligned is what makes one shift enough: it and ``new_entries`` then sit in
        ``src`` under the same rule, so placing them is a single shift by ``TILE_SIZE - r_e``.

        The shift is a matmul against a one-hot matrix and not a slice: a slice's offset becomes part of
        its program, a matmul carries it as data."""
        tile = ttnn.TILE_SIZE
        f, r_e = divmod(state.entry_count, tile)
        shift, take = self._tail_tile_matrices(r_e, width, n_new)

        src = ttnn.concat([state.tail, new_entries], dim=2)  # [B, 1, tile + width, head_dim]
        merged = ttnn.matmul(shift, src, memory_config=self.memory_config)
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, merged, 0, update_idx=f * tile)
        state.tail = ttnn.matmul(take, merged, memory_config=self.memory_config)

    def _build_tail_tile_matrices(self, width):
        """Every one-hot pair the write can need, for one slab width. Called from alloc_state and never
        from forward, because a pair costs ~0.9 ms of host time to build. The whole set is 768 KB a chip."""
        tile, buf = ttnn.TILE_SIZE, _cache_write_rows(width)
        for r_e in range(tile):
            # merged row i takes src row i + (tile - r_e); rows past the entries stay zero, so nothing
            # reads past src.
            rows = torch.arange(r_e + width)
            shift = torch.zeros(1, 1, buf, tile + width)
            shift[0, 0, rows, rows + (tile - r_e)] = 1.0
            self._shift_cache[(r_e, width)] = self._from_torch(shift)

        # The take matrix is keyed on r_e + n_new and not r_e + width: the whole padded width is written,
        # but entry_count only advances by the real entries. Rows before the tile's first live entry stay
        # zero, so the next chunk's shift skips them by construction instead of having to mask them.
        for s in range(tile + width):
            rows = torch.arange(tile - s % tile, tile)
            take = torch.zeros(1, 1, tile, buf)
            take[0, 0, rows, rows + s - tile] = 1.0
            self._take_cache[s] = self._from_torch(take)

    def _carry_key(self, real_len):
        """The carry index is tabulated per whole compression window. A ragged chunk rounds down, which is
        safe because only the final chunk may be ragged and nothing reads its carry."""
        rate = self.compressor.compress_rate
        return max(rate, (int(real_len) // rate) * rate)

    def _build_carry_index(self, chunk_tokens):
        """start/end index tensors for the carry slice, one pair per real_len a chunk can have. Built here
        because forward must build no host tensors; each pair is 8 uint32s."""
        rate, sw = self.compressor.compress_rate, self.sliding_window
        assert sw % ttnn.TILE_SIZE == 0 and rate % sw == 0, (
            f"the carry slice needs a tile-aligned start and one whole window per step: sliding_window "
            f"{sw} must be a multiple of {ttnn.TILE_SIZE} and divide compress_rate {rate}"
        )

        def idx(vals):
            return self._from_torch(
                torch.tensor(vals, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        for real_len in range(rate, int(chunk_tokens) + 1, rate):
            self._carry_index[real_len] = (idx([0, 0, real_len - sw, 0]), idx([1, 1, real_len, self.head_dim]))

    def _tail_tile_matrices(self, r_e, width, n_new):
        """Lookup only. A miss means alloc_state was given a different chunk width, and building one here
        would put host work back into forward."""
        shift = self._shift_cache.get((r_e, width))
        take = self._take_cache.get(r_e + n_new)
        assert shift is not None and take is not None, (
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
        seq_len_actual: int | None = None,
        *,
        state: TtHCAState,
    ):
        """One chunk: [B, 1, S_pad/sp, hidden/tp] in and out; the caller keeps the first S_real rows.

        ``seq_len_actual`` is the chunk's real pre-pad length. Where the chunk sits in the sequence comes
        from ``state``, which ``alloc_state`` builds and this advances in place -- a prefill of one chunk
        passes a state too, so there is no second path through here."""
        batch = hidden_states.shape[0]
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        compress_rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        assert batch == 1, f"HCA prefill expects batch 1, got {batch}"

        assert real_len >= compress_rate, (
            f"HCA prefill needs at least one full compression window: got seq_len {real_len} < "
            f"compress_rate {compress_rate}"
        )

        n_new = real_len // compress_rate
        total_entries = state.entry_count + n_new
        assert state.kv_actual + real_len <= state.max_seq_len, (
            f"context longer than the state was allocated for: {state.kv_actual + real_len} tokens > "
            f"max_seq_len {state.max_seq_len}"
        )
        # Checked on tokens and not on entry_count, where a dropped partial window is invisible: 4097
        # tokens still gives 32 entries, and the next chunk would start at the wrong position.
        #
        # A non-final chunk ending mid-window would strand those leftover tokens -- they never join a
        # compression window -- and push the next chunk off the 128-token grid the compressor assumes.
        assert state.kv_actual % compress_rate == 0, (
            f"cannot append after a chunk with {state.kv_actual % compress_rate} leftover tokens; only "
            f"the final chunk may be ragged, non-final chunks must be a multiple of {compress_rate}"
        )

        # One rotation for the whole padded slab, shared by both stems and the output un-rope.
        cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
        q = self._q_stem(hidden_states, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)
        new_entries, mask_block = self.compressor(
            hidden_states,
            seq_len_actual=seq_len_actual,
            first_window_position=state.entry_count * compress_rate,
        )
        # Attention then reads the WHOLE cache every chunk, so its shape stays constant and the mask
        # -infs everything past total_entries.
        self._write_compressed(state, new_entries, n_new)

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
        )

        state.entry_count = total_entries
        state.kv_actual += real_len
        state.sliding_carry = next_carry
        return self._o_proj(attn)
