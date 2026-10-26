# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer attention: fused QKVG projection, per-head q/k RMSNorm, optional RoPE, paged KV, output gate.

Op sequences adapted from models/demos/gemma4/tt/attention/{decode,prefill}.py.
"""

import functools
import math

import torch

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_MAX_SEQ,
    _paged_fill_cache,
    apply_rope,
    chunked_prefill_sdpa,
    chunked_prefill_sdpa_sliding,
    prefill_sdpa_program_config,
)
from models.demos.gemma4.tt.dram_sharded import _PREFILL_CUTOFF, TILE_SIZE, DramShardedLinear, can_dram_shard
from models.demos.muse_glimmer_30b.tt.common import (
    as_tt,
    decode_memcfg,
    is_decode,
    matmul_compute_config,
    norm_compute_config,
    sdpa_compute_config,
    shard_mapper,
)
from models.demos.muse_glimmer_30b.tt.rms_norm import per_head_rms_norm
from models.demos.muse_glimmer_30b.tt.weight_packing import pack_o, pack_qkvg


@functools.lru_cache(maxsize=None)
def heads_memcfg(batch, head_dim):
    """[1, batch, heads, head_dim] height-sharded L1, one user per core (first `batch` cores of an 8x8 grid)."""
    cores = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
    )


@functools.lru_cache(maxsize=None)
def kv_memcfg(batch, head_dim):
    """K layout for the fused cache update: like heads_memcfg on disjoint cores (rows 4+ of an 8x8 grid)."""
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(i % 8, 4 + i // 8), ttnn.CoreCoord(i % 8, 4 + i // 8)) for i in range(batch)}
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
    )


def dram_sharded_enabled(k: int, n: int, dtype) -> bool:
    """DRAM-width-sharded decode matmul is used whenever the shapes qualify."""
    return can_dram_shard(k, n, dtype)


class MuseLinear:
    """Per-rank weight [k, n] served by DramShardedLinear (decode + prefill) when eligible, else ttnn.linear.

    Decode compute config (group "attn" -> HiFi2 config; group "mlp" -> None, i.e. LoFi without fp32 acc):
      * DRAM-sharded decode (M <= 32): the group's decode config.
      * DRAM-sharded prefill (M > 32) and the non-sharded fallback: the HiFi2/fp32-acc config.
    Output is DRAM interleaved in all paths.
    """

    def __init__(
        self,
        weight_torch,
        mesh_device,
        *,
        shard_dim,
        k,
        n,
        dtype,
        cache_file_name,
        sharded=None,
        group="attn",
        readers=2,
        block=2,
    ):
        self.k, self.n = k, n
        self.hifi_cfg = matmul_compute_config(mesh_device)
        self.decode_cfg = self.hifi_cfg if group == "attn" else None
        self.sharded = dram_sharded_enabled(k, n, dtype) if sharded is None else sharded
        mapper = shard_mapper(mesh_device, shard_dim)
        if self.sharded:
            self.linear = DramShardedLinear(weight_torch, mesh_device, mapper, k, n, dtype, cache_file_name)
            self.weight = self.linear.weight
            self._decode_mem = decode_memcfg(k, mesh_device)
            # Multi-core DRAM-sharded pipeline; readers/block tuned per shape on QB2.
            self._decode_pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=math.ceil((n // 32) / 52),
                fused_activation=None,
                num_workers_per_dram_bank=readers,
                use_multicore_pipeline=True,
            )
        else:
            self.linear = None
            self.weight = as_tt(
                weight_torch, mesh_device, dtype=dtype, mesh_mapper=mapper, cache_file_name=cache_file_name
            )

    def __call__(self, x):
        if not self.sharded:
            return ttnn.linear(
                x, self.weight, compute_kernel_config=self.hifi_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        m = 1
        for d in range(len(x.shape) - 1):
            m *= int(x.shape[d])
        cfg = self.decode_cfg if m <= TILE_SIZE else self.hifi_cfg
        # DramShardedLinear's chunk+concat fallback (M > cutoff and M % cutoff != 0) returns an already-deallocated
        # buffer when there is a single chunk. Pad rows to a multiple of the cutoff and slice back
        # (matmul rows are independent, so numerics are unchanged).
        if (
            m > _PREFILL_CUTOFF
            and m % _PREFILL_CUTOFF != 0
            and len(x.shape) == 4
            and int(x.shape[0]) == 1
            and int(x.shape[1]) == 1
        ):
            mp = -(-m // _PREFILL_CUTOFF) * _PREFILL_CUTOFF
            xp = ttnn.pad(x, [(0, 0), (0, 0), (0, mp - m), (0, 0)], value=0.0)
            outp = self.linear(xp, compute_kernel_config=cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG)
            xp.deallocate(True)
            out = ttnn.slice(outp, [0, 0, 0, 0], [1, 1, m, int(outp.shape[-1])])
            outp.deallocate(True)
            return out
        return self.linear(x, compute_kernel_config=cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode(self, x):
        """Decode matmul (M = 32): L1 width-sharded output; falls back to __call__ when not DRAM-sharded."""
        if not self.sharded:
            return self(x)
        own = not (x.is_sharded() and x.memory_config() == self._decode_mem)
        x_sh = ttnn.to_memory_config(x, self._decode_mem) if own else x
        out = ttnn.linear(
            x_sh,
            self.weight,
            program_config=self._decode_pc,
            compute_kernel_config=self.decode_cfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if own:
            x_sh.deallocate(True)
        return out


def allocate_kv_cache(args, mesh_device, num_blocks: int | None = None) -> tuple:
    """Paged K/V cache for one layer: each [num_blocks, kv_heads_local, block_size, head_dim], zeros, replicated."""
    n = num_blocks if num_blocks is not None else args.max_num_blocks
    shape = [n, args.kv_heads_local, args.block_size, args.head_dim]
    caches = []
    for _ in range(2):
        caches.append(
            ttnn.as_tensor(
                torch.zeros(shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=args.kv_cache_dtype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    return tuple(caches)


class MuseGlimmerAttention:
    def __init__(self, args, mesh_device, ccl, layer_idx: int, state_dict: dict, use_weight_cache: bool = True):
        self.args = args
        self.mesh_device = mesh_device
        self.ccl = ccl
        self.layer_idx = layer_idx
        self.is_sliding = args.is_sliding(layer_idx)
        self.uses_rope = args.uses_rope(layer_idx)
        self.sliding_window = args.sliding_window if self.is_sliding else None
        self.matmul_cfg = matmul_compute_config(mesh_device)
        self.sdpa_cfg = sdpa_compute_config(mesh_device)
        self.rope_cfg = norm_compute_config(mesh_device)

        g = lambda k: state_dict[f"self_attn.{k}_proj.weight"]  # noqa: E731
        i = layer_idx
        dt = args.attn_weight_dtype
        qkvg = pack_qkvg(g("q"), g("k"), g("v"), g("gate"), args.tp, args.head_dim)
        tag = "v1ws" if dram_sharded_enabled(args.hidden_size, args.qkvg_width_local, dt) else "v1"
        self.wqkvg = MuseLinear(
            qkvg,
            mesh_device,
            shard_dim=-1,
            k=args.hidden_size,
            n=args.qkvg_width_local,
            dtype=dt,
            cache_file_name=args.cache_file(f"layers.{i}.self_attn.wqkvg", dt, tag=tag) if use_weight_cache else None,
            readers=2,
            block=2,
        )
        tag_o = "v1ws" if dram_sharded_enabled(args.q_width_local, args.hidden_size, dt) else "v1"
        self.wo = MuseLinear(
            pack_o(g("o"), args.tp),
            mesh_device,
            shard_dim=-2,
            k=args.q_width_local,
            n=args.hidden_size,
            dtype=dt,
            cache_file_name=args.cache_file(f"layers.{i}.self_attn.wo", dt, tag=tag_o) if use_weight_cache else None,
            readers=3,
            block=2,
        )

    def _split_qkv_gate(self, xqkvg):
        """Fused [.., qkvg_w] -> (qkv [.., qkv_w], gate [.., q_w])."""
        a = self.args
        s = xqkvg.shape
        qkv = ttnn.slice(xqkvg, [0, 0, 0, 0], [s[0], s[1], s[2], a.qkv_width_local])
        gate = ttnn.slice(xqkvg, [0, 0, 0, a.qkv_width_local], [s[0], s[1], s[2], a.qkvg_width_local])
        xqkvg.deallocate(True)
        return qkv, gate

    def _gate_oproj(self, attn, gate, out_mem=ttnn.DRAM_MEMORY_CONFIG):
        """attn * sigmoid(gate) -> o_proj (partial sums) -> all_reduce."""
        # Fused SIGMOID is the accurate one: a bare UnaryOpType::SIGMOID emits sigmoid_tile_init() (unary_op_utils.cpp:815),
        # whose fast_and_approx template arg defaults to false (compute_kernel_api.h:123).
        sig = [ttnn.UnaryOpType.SIGMOID]
        if is_decode(attn):
            attn_m = ttnn.to_memory_config(attn, gate.memory_config())
            gated = ttnn.multiply(gate, attn_m, input_tensor_a_activations=sig)
            attn_m.deallocate(True)
            attn.deallocate(True)
            gate.deallocate(True)
            out = self.wo.decode(gated)
            gated.deallocate(True)
            reduced = self.ccl.all_reduce(out, memory_config=out_mem)
        else:
            gated = ttnn.multiply(gate, attn, input_tensor_a_activations=sig)
            attn.deallocate(True)
            gate.deallocate(True)
            out = self.wo(gated)
            gated.deallocate(True)
            reduced = self.ccl.all_reduce(out)
        if reduced is not out:
            out.deallocate(True)
        return reduced

    def forward_prefill(self, x, rope, kv_cache, page_table, user_id: int):
        a = self.args
        k_cache, v_cache = kv_cache

        xqkvg = self.wqkvg(x)
        qkv, gate = self._split_qkv_gate(xqkvg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=a.q_heads_local,
            num_kv_heads=a.kv_heads_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)

        # q/k weightless per-head RMSNorm (HF does it in fp32); no v norm.
        q = per_head_rms_norm(q, a.rms_norm_eps, self.mesh_device, fp32_acc=True)
        k = per_head_rms_norm(k, a.rms_norm_eps, self.mesh_device, fp32_acc=True)

        if self.uses_rope and rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if page_table is not None:
            _paged_fill_cache(k_cache, k, page_table, batch_idx=user_id)
            _paged_fill_cache(v_cache, v, page_table, batch_idx=user_id)
        else:
            kf = k if k.dtype == k_cache.dtype else ttnn.typecast(k, k_cache.dtype)
            vf = v if v.dtype == v_cache.dtype else ttnn.typecast(v, v_cache.dtype)
            ttnn.fill_cache(k_cache, kf, batch_idx=user_id)
            ttnn.fill_cache(v_cache, vf, batch_idx=user_id)

        seq_len = x.shape[-2]
        if seq_len > PREFILL_SDPA_MAX_SEQ:
            # Non-chunked prefill SDPA is wrong past 2^15 query rows: use gemma4's chunked paths instead.
            if self.is_sliding:
                sdpa = chunked_prefill_sdpa_sliding(q, k, v, self.sliding_window, a.head_dim, scale=a.sdpa_scale)
            else:
                assert page_table is not None, "prefill longer than 32768 tokens needs a paged KV cache"
                sdpa = chunked_prefill_sdpa(q, k_cache, v_cache, page_table, user_id, a.head_dim, scale=a.sdpa_scale)
        else:
            sdpa = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=a.sdpa_scale,
                sliding_window_size=self.sliding_window,
                program_config=prefill_sdpa_program_config(a.head_dim, seq_len, self.sliding_window),
                compute_kernel_config=self.sdpa_cfg,
            )
        for t in (q, k, v):
            t.deallocate(True)

        attn = ttnn.experimental.nlp_concat_heads(sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,S,ql*128]
        sdpa.deallocate(True)
        return self._gate_oproj(attn, gate)

    def forward_decode(self, x, current_pos, rope, kv_cache, page_table, sdpa_inputs=None):
        a = self.args
        k_cache, v_cache = kv_cache
        heads_mem = heads_memcfg(x.shape[-2], a.head_dim)
        batch = x.shape[-2]

        xqkvg = self.wqkvg.decode(x)
        qkv, gate = self._split_qkv_gate(xqkvg)
        qkv = ttnn.to_memory_config(qkv, decode_memcfg(a.qkv_width_local, self.mesh_device))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=a.q_heads_local, num_kv_heads=a.kv_heads_local, memory_config=heads_mem
        )
        qkv.deallocate(True)

        # Per-head weightless norm in L1. No fp32 acc for decode (the fp32 config lowered paged-decode PCC).
        q = ttnn.rms_norm(
            ttnn.to_memory_config(q, ttnn.L1_MEMORY_CONFIG), epsilon=a.rms_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        k = ttnn.rms_norm(
            ttnn.to_memory_config(k, ttnn.L1_MEMORY_CONFIG), epsilon=a.rms_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        if self.uses_rope and rope is not None:
            cos_b, sin_b = rope
            if cos_b.shape[1] != batch:  # tables may be padded to 32 users
                cos_b = cos_b[:, :batch, :, :]
                sin_b = sin_b[:, :batch, :, :]
            moved = cos_b.memory_config() != heads_mem
            if moved:  # temporaries, freed below
                cos_b = ttnn.to_memory_config(cos_b, heads_mem)
                sin_b = ttnn.to_memory_config(sin_b, heads_mem)
            q_i, k_i = q, k
            q = ttnn.experimental.rotary_embedding_hf(
                ttnn.to_memory_config(q_i, heads_mem),
                cos_b,
                sin_b,
                is_decode_mode=True,
                compute_kernel_config=self.rope_cfg,
            )
            k = ttnn.experimental.rotary_embedding_hf(
                ttnn.to_memory_config(k_i, heads_mem),
                cos_b,
                sin_b,
                is_decode_mode=True,
                compute_kernel_config=self.rope_cfg,
            )
            q_i.deallocate(True)
            k_i.deallocate(True)
            if moved:
                cos_b.deallocate(True)
                sin_b.deallocate(True)

        k = ttnn.to_memory_config(k, kv_memcfg(batch, a.head_dim))
        ttnn.experimental.paged_fused_update_cache(
            k_cache,
            k,
            v_cache,
            v,
            update_idxs_tensor=current_pos,
            **({} if page_table is None else {"page_table": page_table}),
        )
        k.deallocate(True)
        v.deallocate(True)

        # SDPA decode splits its cores over all user slots, so run it on the first `sb` only.
        sb = min(a.max_batch_size, batch)
        sdpa_pos, sdpa_pt, own = current_pos, page_table, ()
        if sb < batch:
            if sdpa_inputs is not None:
                sdpa_pos, sdpa_pt = sdpa_inputs
            else:
                sdpa_pos = ttnn.slice(current_pos, [0], [sb])
                sdpa_pt = None if page_table is None else ttnn.slice(page_table, [0, 0], [sb, page_table.shape[1]])
                own = (sdpa_pos,) if sdpa_pt is None else (sdpa_pos, sdpa_pt)
            q_full = q
            q = ttnn.slice(
                q_full, [0, 0, 0, 0], [1, sb, q_full.shape[2], q_full.shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            q_full.deallocate(True)
        else:
            q = ttnn.to_memory_config(
                q, ttnn.DRAM_MEMORY_CONFIG
            )  # paged SDPA decode mis-reads height-sharded q past the first core row
        grid = self.mesh_device.compute_with_storage_grid_size()
        # k_chunk measured on QB2: 256 for the 2048-token window, 512 for full-context layers.
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=256 if self.is_sliding else 512,
            exp_approx_mode=False,
        )
        sdpa_kwargs = dict(
            cur_pos_tensor=sdpa_pos,
            scale=a.sdpa_scale,
            sliding_window_size=self.sliding_window,
            program_config=program_config,
            compute_kernel_config=self.sdpa_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if page_table is not None:
            sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, k_cache, v_cache, page_table_tensor=sdpa_pt, **sdpa_kwargs
            )
        else:
            sdpa = ttnn.transformer.scaled_dot_product_attention_decode(q, k_cache, v_cache, **sdpa_kwargs)
        q.deallocate(True)
        for t in own:
            t.deallocate(True)

        attn = ttnn.experimental.nlp_concat_heads_decode(
            ttnn.to_memory_config(sdpa, heads_memcfg(sb, a.head_dim)), num_heads=a.q_heads_local
        )  # [1,1,32,ql*128] L1 width-sharded
        if attn.shape[2] != batch:  # the concat output has 32 rows; rows past the SDPA users are unused
            w = attn.shape[-1]
            attn = ttnn.reshape(attn, ttnn.Shape([1, 1, batch, w]), ttnn.Shape([1, 1, 32, w]))
        sdpa.deallocate(True)
        return self._gate_oproj(attn, gate, out_mem=decode_memcfg(a.hidden_size, self.mesh_device))
