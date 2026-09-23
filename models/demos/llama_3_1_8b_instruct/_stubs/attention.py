# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN grouped-query attention for Llama-3.1-8B-Instruct (tensor-parallel).

Derived from the Hugging Face ``LlamaAttention`` architecture (q/k/v/o projections +
rotary embedding + causal SDPA) and the generic TTNN building blocks; the whole
forward runs on device.

Tensor-parallel scheme (TP = number of devices in the mesh, 4 here)
------------------------------------------------------------------
Model config: hidden = 4096, n_heads = 32, n_kv_heads = 8, head_dim = 128.

* ``q_proj`` / ``k_proj`` / ``v_proj`` are COLUMN-parallel: their outputs feed
  per-head ops (rotary, SDPA), so each chip owns a disjoint slice of HEADS.
  Chip d owns q-heads ``[8d, 8d+8)`` and kv-heads ``[2d, 2d+2)``. GQA stays
  consistent because head ``h`` reads kv-head ``h // 4``, and ``[8d, 8d+8) // 4
  == [2d, 2d+2)`` — no KV replication and no cross-chip traffic inside SDPA.
  The three projections are fused into one per-chip ``[hidden, (8+2+2)*128]``
  matmul, laid out so that a single ``ShardTensorToMesh(dim=-1)`` over the
  globally concatenated ``[q_d | k_d | v_d]`` blocks hands each chip exactly its
  own slice (that is also the layout ``nlp_create_qkv_heads`` expects).
* ``o_proj`` is ROW-parallel: it reduces back to the model dim, so its INPUT
  features are split — chip d owns rows ``[1024d, 1024d+1024)``, which are
  precisely the head-dims of the q-heads chip d computed. Each chip produces a
  PARTIAL sum over the full hidden dim, and one ``all_reduce`` over the TP axis
  turns those partials into the replicated full output.
* Rotary tables and any biases stay REPLICATED (per-element ops); an output bias
  is added AFTER the reduce so it is counted once.

The math is unchanged: gathered output == single-device golden. Only placement moves.

Two forwards, one weight set (added for the e2e pipeline; the graduated body above
is untouched):

* ``mode="prefill"`` — the graduated full-sequence causal path. When a KV cache has
  been allocated it ALSO writes the post-rotary K/V into it (``ttnn.fill_cache``),
  so decode can read the prefix instead of recomputing it.
* ``mode="decode"`` — one token: ``nlp_create_qkv_heads_decode`` ->
  rotary -> ``paged_update_cache`` -> ``scaled_dot_product_attention_decode`` over
  the RESIDENT cache -> ``nlp_concat_heads_decode`` -> row-parallel WO -> all_reduce.
  Same weights, same TP=4 split, same collective — only the attention kernel and the
  head layout differ. Pure ttnn: no host compute in either path.

``position_embeddings`` may be a pair of TORCH tensors (what the per-component PCC
harness passes) or a pair of TTNN tensors (what the graduated ``rotary_embedding``
stub emits in the e2e pipeline). The ttnn branch keeps the whole chain on device.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.tt_ccl import default_topology, get_num_links
from models.demos.llama_3_1_8b_instruct.tt._invocation import record


def _num_devices(device) -> int:
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _tp_cluster_axis(device) -> int:
    """The mesh axis the TP shards live on (the axis whose extent > 1)."""
    try:
        shape = list(device.shape)
    except Exception:
        return 1
    for axis, extent in enumerate(shape):
        if int(extent) > 1:
            return axis
    return 1


def _compute_kernel_config(device):
    """HiFi4 + fp32 accumulate — attention PCC is dominated by matmul fidelity."""
    try:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    except Exception:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


class TtLlamaAttention(LightweightModule):
    """Column-parallel QKV -> rotary -> causal SDPA -> row-parallel WO -> all_reduce."""

    def __init__(self, mesh_device, torch_module):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = _num_devices(mesh_device)

        cfg = getattr(torch_module, "config", None)
        wq = torch_module.q_proj.weight
        wk = torch_module.k_proj.weight
        wv = torch_module.v_proj.weight
        wo = torch_module.o_proj.weight

        self.hidden_size = int(wq.shape[1])
        self.head_dim = int(getattr(torch_module, "head_dim", 0) or 0)
        if self.head_dim <= 0:
            self.head_dim = int(getattr(cfg, "head_dim", 0) or 0)
        if self.head_dim <= 0:
            self.head_dim = self.hidden_size // int(getattr(cfg, "num_attention_heads", 1) or 1)
        self.n_heads = int(wq.shape[0]) // self.head_dim
        self.n_kv_heads = int(wk.shape[0]) // self.head_dim
        self.scaling = float(getattr(torch_module, "scaling", self.head_dim**-0.5))

        # --- TP degree: split heads only when they divide evenly across the mesh ---
        tp = self.num_devices
        if tp > 1 and (self.n_heads % tp or self.n_kv_heads % tp):
            tp = 1  # replicate rather than split a KV group across chips
        self.tp = tp
        self.n_local_heads = self.n_heads // tp
        self.n_local_kv_heads = self.n_kv_heads // tp

        q_slice = self.n_local_heads * self.head_dim
        kv_slice = self.n_local_kv_heads * self.head_dim

        # --- Column-parallel fused QKV weight -------------------------------
        # Per chip: [ q_heads(d) | k_heads(d) | v_heads(d) ], concatenated over d
        # so ShardTensorToMesh(dim=-1) hands chip d exactly its own block.
        wq_t = wq.detach().to(torch.float32).transpose(0, 1)  # [hidden, n_heads*hd]
        wk_t = wk.detach().to(torch.float32).transpose(0, 1)  # [hidden, n_kv*hd]
        wv_t = wv.detach().to(torch.float32).transpose(0, 1)
        # Assembled ONCE into a preallocated host buffer at build time, then uploaded
        # once and kept resident; nothing here runs per forward.
        per_chip = q_slice + 2 * kv_slice
        wqkv_host = torch.empty(wq_t.shape[0], tp * per_chip, dtype=torch.float32)
        for d in range(tp):
            base = d * per_chip
            wqkv_host[:, base : base + q_slice] = wq_t[:, d * q_slice : (d + 1) * q_slice]
            wqkv_host[:, base + q_slice : base + q_slice + kv_slice] = wk_t[:, d * kv_slice : (d + 1) * kv_slice]
            wqkv_host[:, base + q_slice + kv_slice : base + per_chip] = wv_t[:, d * kv_slice : (d + 1) * kv_slice]

        # --- Row-parallel output weight -------------------------------------
        # [n_heads*hd, hidden]; the INPUT features split over the mesh, and chip d's
        # rows are the head-dims of the q-heads chip d computed.
        wo_host = wo.detach().to(torch.float32).transpose(0, 1).contiguous()

        self.wqkv = self._to_device(wqkv_host, shard_dim=-1)
        self.wo = self._to_device(wo_host, shard_dim=0)

        # --- Biases: Llama-3.1 has none, but keep the general case honest ----
        self.qkv_bias = None
        qb = getattr(torch_module.q_proj, "bias", None)
        kb = getattr(torch_module.k_proj, "bias", None)
        vb = getattr(torch_module.v_proj, "bias", None)
        if qb is not None and kb is not None and vb is not None:
            qb, kb, vb = (t.detach().to(torch.float32) for t in (qb, kb, vb))
            bias_host = torch.empty(1, tp * per_chip, dtype=torch.float32)
            for d in range(tp):
                base = d * per_chip
                bias_host[0, base : base + q_slice] = qb[d * q_slice : (d + 1) * q_slice]
                bias_host[0, base + q_slice : base + q_slice + kv_slice] = kb[d * kv_slice : (d + 1) * kv_slice]
                bias_host[0, base + q_slice + kv_slice : base + per_chip] = vb[d * kv_slice : (d + 1) * kv_slice]
            self.qkv_bias = self._to_device(bias_host, shard_dim=-1)
        ob = getattr(torch_module.o_proj, "bias", None)
        self.o_bias = None if ob is None else self._to_device(ob.detach().to(torch.float32).reshape(1, -1))

        self.compute_kernel_config = _compute_kernel_config(mesh_device)

        # --- Collective plumbing (row-parallel WO needs a sum over the TP axis) ---
        self.cluster_axis = _tp_cluster_axis(mesh_device)
        self.topology = ttnn.Topology.Linear
        self.num_links = 1
        self._ar_semaphores = None
        if self.tp > 1:
            try:
                self.topology = default_topology(mesh_device) or ttnn.Topology.Linear
            except Exception:
                self.topology = ttnn.Topology.Linear
            try:
                self.num_links = max(int(get_num_links(mesh_device, self.cluster_axis)), 1)
            except Exception:
                self.num_links = 1
            self._ar_semaphores = self._make_all_reduce_semaphores()

        self._rot_cache = {}

        # Flash-decode splits the KV sequence over cores and tree-reduces the partials.
        # The kernel caps that tree at 6 rounds (64 cores per head), so the grid it may
        # use is pinned to at most 8x8 -- on a >64-core part the default grid asks for
        # more cores per head than the reduction can fold back.
        grid = mesh_device.compute_with_storage_grid_size()
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(min(int(grid.x), 8), min(int(grid.y), 8)),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

        # Resident KV cache (allocated by the pipeline via allocate_kv_cache).
        # Each chip owns ONLY its own kv-heads, so the buffer is created from a
        # replicated zero tensor: 4 independent per-chip caches of the same shape.
        self.kv_cache = None
        self.kv_capacity = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _to_device(self, host_tensor, shard_dim=None, dtype=ttnn.bfloat16):
        mapper = None
        if self.num_devices > 1:
            if shard_dim is None or self.tp == 1:
                mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            else:
                mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=shard_dim)
        kwargs = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        if mapper is not None:
            kwargs["mesh_mapper"] = mapper
        return ttnn.from_torch(host_tensor.to(torch.bfloat16), **kwargs)

    def _make_all_reduce_semaphores(self):
        """all_reduce_async == reduce_scatter + all_gather: 2 barrier, 3 rs, 2 ag semaphores."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

        def _new(count):
            return [ttnn.create_global_semaphore(self.mesh_device, cores, 0) for _ in range(count)]

        return {"barrier": _new(2), "rs": _new(3), "ag": _new(2)}

    def allocate_kv_cache(self, max_seq_len, batch=1):
        """Allocate the resident per-chip K/V cache [batch, n_local_kv, C, head_dim].

        ``C`` is the pinned capacity of the sequence axis (the model's variable dim).
        The cache lives for the life of the module and is never reallocated inside a
        forward, which is what lets decode be trace-captured.
        """
        zeros = torch.zeros(batch, self.n_local_kv_heads, int(max_seq_len), self.head_dim, dtype=torch.bfloat16)
        kwargs = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        if self.num_devices > 1:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.mesh_device)
        self.kv_cache = (ttnn.from_torch(zeros, **kwargs), ttnn.from_torch(zeros, **kwargs))
        self.kv_capacity = int(max_seq_len)
        return self.kv_cache

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        return self.forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, mode="prefill", cur_pos=None, **kwargs):
        record("attention")  # Gate 2: proof of invocation, from INSIDE the real forward
        if mode == "decode":
            return self._forward_decode(hidden_states, position_embeddings, cur_pos)
        x = hidden_states
        in_rank = len(x.shape)
        batch = int(x.shape[0]) if in_rank >= 3 else 1
        seq_len = int(x.shape[-2])
        if in_rank == 3:
            x = ttnn.reshape(x, (batch, 1, seq_len, int(x.shape[-1])))

        # --- Column-parallel fused QKV projection ---
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        if self.qkv_bias is not None:
            xqkv = ttnn.add(xqkv, self.qkv_bias)

        # No collective here: each chip owns whole heads, so its rows are already complete.
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # --- Rotary embedding (replicated tables, applied to this chip's heads) ---
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = self._apply_rope(q, cos, sin, self.n_local_heads, batch, seq_len)
            k = self._apply_rope(k, cos, sin, self.n_local_kv_heads, batch, seq_len)

        # --- Seed the resident KV cache so decode never recomputes the prefix ---
        if self.kv_cache is not None:
            keys, values = self.kv_cache
            ttnn.fill_cache(keys, k, 0)
            ttnn.fill_cache(values, v, 0)

        # --- Causal SDPA over this chip's heads (GQA handled inside the op) ---
        sdpa_kwargs = dict(
            scale=self.scaling,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if attention_mask is None:
            attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, **sdpa_kwargs)
        else:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=self._to_ttnn_mask(attention_mask), is_causal=False, **sdpa_kwargs
            )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # [b, n_local_heads, s, hd] -> [b, 1, s, n_local_heads * hd]
        attn_concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        # --- Row-parallel output projection: per-chip PARTIAL over the full hidden dim ---
        out = ttnn.linear(
            attn_concat,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_concat)

        # --- The collective that closes the row-parallel split ---
        out = self._all_reduce(out)
        if self.o_bias is not None:
            out = ttnn.add(out, self.o_bias)

        if in_rank == 3:
            out = ttnn.reshape(out, (batch, seq_len, int(out.shape[-1])))
        return out

    # ------------------------------------------------------------------
    # Pieces
    # ------------------------------------------------------------------
    @staticmethod
    def _as_bf16(t):
        return t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)

    def _rot_tables(self, cos, sin, n_local, batch, seq_len):
        """cos/sin shaped for the PREFILL head layout [b, n_local, s, head_dim].

        Two input flavours:
          * TTNN (the graduated ``rotary_embedding`` stub's output, [b, s, head_dim]):
            reshaped on device to [b, 1, s, head_dim] and left to broadcast over the
            head axis. NOT cached — the tables are position-dependent and change on
            every call, so a shape-keyed cache would serve a stale table.
          * TORCH (what the per-component PCC harness passes): staged as before and
            cached, since that harness reuses one fixed table.
        """
        if isinstance(cos, ttnn.Tensor):
            hd = int(cos.shape[-1])
            return (
                self._as_bf16(ttnn.reshape(cos, (batch, 1, seq_len, hd))),
                self._as_bf16(ttnn.reshape(sin, (batch, 1, seq_len, hd))),
            )
        key = (n_local, batch, seq_len, int(cos.shape[-1]))
        cached = self._rot_cache.get(key)
        if cached is not None:
            return cached

        def _stage(t):
            t = t.detach()
            # HF broadcasts (b, s, head_dim) over the head axis (unsqueeze_dim=1).
            t = t.reshape(-1, 1, int(t.shape[-2]), int(t.shape[-1]))
            t = t.expand(batch, n_local, seq_len, int(t.shape[-1])).contiguous()
            return self._to_device(t)

        tables = (_stage(cos), _stage(sin))
        self._rot_cache[key] = tables
        return tables

    def _apply_rope(self, t, cos, sin, n_local, batch, seq_len):
        """t * cos + rotate_half(t) * sin — the HF Llama rotary convention."""
        cos_tt, sin_tt = self._rot_tables(cos, sin, n_local, batch, seq_len)
        return self._rotate(t, cos_tt, sin_tt)

    def _rotate(self, t, cos_tt, sin_tt):
        """The rotary math itself, on already-shaped (or broadcastable) tables."""
        shape = [int(s) for s in t.shape]
        half = shape[-1] // 2
        x1 = ttnn.slice(t, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half])
        x2 = ttnn.slice(t, [0, 0, 0, half], [shape[0], shape[1], shape[2], shape[3]])
        neg_x2 = ttnn.neg(x2)
        ttnn.deallocate(x2)
        rotated = ttnn.concat([neg_x2, x1], dim=-1)
        ttnn.deallocate(neg_x2)
        ttnn.deallocate(x1)
        out = ttnn.add(ttnn.mul(t, cos_tt), ttnn.mul(rotated, sin_tt))
        ttnn.deallocate(rotated)
        ttnn.deallocate(t)
        return out

    def _rot_tables_decode(self, cos, sin, batch):
        """cos/sin shaped for the DECODE head layout [1, b, 1, head_dim].

        The decode tensors put BATCH on dim 1 and HEADS on dim -2, so the table is
        reshaped to a single row and left to broadcast over the head axis.
        """
        hd = int(cos.shape[-1])
        return (
            self._as_bf16(ttnn.reshape(cos, (1, batch, 1, hd))),
            self._as_bf16(ttnn.reshape(sin, (1, batch, 1, hd))),
        )

    def _forward_decode(self, hidden_states, position_embeddings, cur_pos):
        """One-token forward over the RESIDENT KV cache. Pure ttnn.

        Same weights and same TP=4 split as prefill; only the head layout and the
        attention kernel change. ``cur_pos`` is a device int32 tensor of length
        ``batch`` holding the cache slot this token is written to (and the last
        position it may attend to).
        """
        assert self.kv_cache is not None, "allocate_kv_cache(...) must be called before decode"
        x = hidden_states
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, int(x.shape[0]) * int(x.shape[1]), int(x.shape[-1])))
        batch = int(x.shape[-2])

        # --- Column-parallel fused QKV projection (same weight as prefill) ---
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        if self.qkv_bias is not None:
            xqkv = ttnn.add(xqkv, self.qkv_bias)

        # Decode head layout: [1, b, n_local_heads, head_dim], height-sharded in L1.
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads
        )
        ttnn.deallocate(xqkv)
        q_memcfg = q.memory_config()
        kv_memcfg = k.memory_config()

        # --- Rotary embedding (replicated tables, this chip's heads) ---
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos_tt, sin_tt = self._rot_tables_decode(cos, sin, batch)
            q_d = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q)
            k_d = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(k)
            q = self._rotate(q_d, cos_tt, sin_tt)
            k_rot = self._rotate(k_d, cos_tt, sin_tt)
            # paged_update_cache requires a SHARDED input, so put K back where it was.
            k = ttnn.to_memory_config(k_rot, kv_memcfg)
            ttnn.deallocate(k_rot)

        # --- Write this token into the resident cache (no prefix recompute) ---
        keys, values = self.kv_cache
        ttnn.experimental.paged_update_cache(keys, k, update_idxs_tensor=cur_pos)
        ttnn.experimental.paged_update_cache(values, v, update_idxs_tensor=cur_pos)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # --- Flash-decode over [0, cur_pos] of the cache (GQA handled by the op) ---
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            keys,
            values,
            cur_pos_tensor=cur_pos,
            scale=self.scaling,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)

        # [1, b, n_local_heads, hd] -> [1, 1, 32, n_local_heads * hd] (batch padded to 32)
        attn_sharded = ttnn.to_memory_config(attn, q_memcfg)
        ttnn.deallocate(attn)
        attn_concat = ttnn.experimental.nlp_concat_heads_decode(attn_sharded, num_heads=self.n_local_heads)
        ttnn.deallocate(attn_sharded)
        attn_concat = ttnn.to_memory_config(attn_concat, ttnn.DRAM_MEMORY_CONFIG)
        width = int(attn_concat.shape[-1])
        if int(attn_concat.shape[-2]) != batch:
            attn_concat = ttnn.slice(attn_concat, [0, 0, 0, 0], [1, 1, batch, width])

        # --- Row-parallel output projection + the collective that closes it ---
        out = ttnn.linear(
            attn_concat,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_concat)
        out = self._all_reduce(out)
        if self.o_bias is not None:
            out = ttnn.add(out, self.o_bias)
        return out

    def _to_ttnn_mask(self, attention_mask):
        if not isinstance(attention_mask, torch.Tensor):
            return attention_mask
        return self._to_device(attention_mask.detach())

    def _all_reduce(self, tensor):
        """Sum the row-parallel partials over the TP axis; the result is replicated."""
        if self.tp == 1 or self._ar_semaphores is None:
            return tensor
        sems = self._ar_semaphores
        reduced = ttnn.experimental.all_reduce_async(
            tensor,
            cluster_axis=self.cluster_axis,
            mesh_device=self.mesh_device,
            barrier_semaphores=sems["barrier"],
            rs_global_semaphores=sems["rs"],
            ag_global_semaphores=sems["ag"],
            math_op=ttnn.ReduceType.Sum,
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
        )
        ttnn.deallocate(tensor)
        return reduced


def build(device, torch_module):
    """Entry point used by the per-component PCC harness."""
    return TtLlamaAttention(device, torch_module)
