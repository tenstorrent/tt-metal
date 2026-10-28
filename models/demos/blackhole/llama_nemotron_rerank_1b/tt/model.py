# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""llama-nemotron-rerank-1b-v2 as a bidirectional encoder on one Blackhole chip.

One forward scores B (query, passage) sequences padded to a common length S:

    tokens [B, S] -> embedding -> 16 x (RMSNorm -> QKV -> heads -> RoPE ->
    bidirectional SDPA with padding mask -> concat heads -> WO -> +residual ->
    RMSNorm -> SiLU(W1 x) * W3 x -> W2 -> +residual) -> final RMSNorm ->
    per-token score (x . w_score) -> sum over real tokens -> [B] scores

The score head is applied per token before pooling: mean_t(x_t) . w equals
mean_t(x_t . w), and the per-token product is a [B*S, 2048] x [2048, 32]
matmul instead of a pooled hidden state that would need a second pass. The
division by the real-token count happens on the host.
"""

import math

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.fused_add_rmsnorm import fused_add_rmsnorm
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.fused_heads_rope import fused_heads_rope
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_add_norm import (
    embed_rmsnorm,
    matmul_add_norm,
    resident_rmsnorm,
    to_resident,
)
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_qkv_rope import (
    make_rope_table,
    matmul_qkv_rope,
)
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import (
    MM_ADD_NORM_CONFIGS,
    MM_ADD_NORM_IN1_DEPTH,
    MM_ADD_NORM_Y_FIDELITY,
    MM_CONFIGS,
    PAD_TOKEN_ID,
    QKV_ROPE_CONFIGS,
    QKV_ROPE_FIDELITY,
    QKV_ROPE_IN1_DEPTH,
    SEQ_BUCKETS,
    RerankConfig,
    mm_block_config,
    sdpa_chunks,
)
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.weights import WeightLoader
from models.tt_transformers.tt.common import gather_cos_sin, get_rot_transformation_mat, precompute_freqs

# Matmuls with more rows than this are split into row blocks so each fits on device.
MAX_MM_ROWS = 2048
MASK_NEG = -1e9


class Layer:
    def __init__(self, cfg: RerankConfig, loader: WeightLoader, i: int):
        p = f"model.layers.{i}."
        self.cfg = cfg
        self.attn_norm = loader.norm(f"l{i}_attn_norm", p + "input_layernorm.weight")
        self.ffn_norm = loader.norm(f"l{i}_ffn_norm", p + "post_attention_layernorm.weight")
        if cfg.fuse_add_norm:
            self.attn_norm_consts = loader.norm_consts(p + "input_layernorm.weight")
            self.ffn_norm_consts = loader.norm_consts(p + "post_attention_layernorm.weight")
        # matmul_qkv_rope rotates HF's half-split head layout; the stock RoPE ops want Meta's
        # interleaved pairs, so Q / K rows are permuted for them.
        self.wqkv = loader.linear(
            f"l{i}_wqkv_hf_qscaled" if cfg.fuse_qkv_rope else f"l{i}_wqkv_qscaled",
            [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"],
            cfg.wqkv_dtype,
            permute_heads=None if cfg.fuse_qkv_rope else [cfg.n_heads, cfg.n_kv_heads, None],
            # softmax(Q K^T / sqrt(d)): 1/sqrt(64) = 1/8 is a power of two, so folding it into Wq is exact
            # and SDPA runs at scale 1. At any other scale ttnn's SDPA wrapper multiplies the mask by
            # 1/scale on every call (one extra [B, 1, S, S] pass per layer).
            scales=[1.0 / math.sqrt(cfg.head_dim), None, None],
            gamma_key=p + "input_layernorm.weight",
        )
        self.wo = loader.linear(f"l{i}_wo", [p + "self_attn.o_proj.weight"], cfg.wo_dtype)
        if cfg.fuse_swiglu:
            self.w13 = loader.swiglu(
                f"l{i}_w13s",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                cfg.w13_dtype,
                gamma_key=p + "post_attention_layernorm.weight",
            )
        else:
            self.w1 = loader.linear(f"l{i}_w1", [p + "mlp.gate_proj.weight"], cfg.w13_dtype)
            self.w3 = loader.linear(f"l{i}_w3", [p + "mlp.up_proj.weight"], cfg.w13_dtype)
        self.w2 = loader.linear(f"l{i}_w2", [p + "mlp.down_proj.weight"], cfg.w2_dtype)


class NemotronReranker:
    def __init__(self, device, cfg: RerankConfig, state_dict, cache_dir=None):
        self.device = device
        self.cfg = cfg
        loader = WeightLoader(device, cfg, state_dict, cache_dir)
        self.embed = loader.embedding()
        self.layers = [Layer(cfg, loader, i) for i in range(cfg.n_layers)]
        self.final_norm = loader.norm("final_norm", "model.norm.weight")
        self.final_norm_consts = loader.norm_consts("model.norm.weight") if cfg.fuse_add_norm else None
        self.score_w = loader.score_head(gamma_key="model.norm.weight")

        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.grid = grid
        self.mm_ckc = cfg.compute_config(cfg.mm_fidelity)
        self.mm_ckc_approx = cfg.compute_config(cfg.mm_fidelity, approx=True)
        self.mm_ckc_lofi = cfg.compute_config(ttnn.MathFidelity.LoFi)
        self.lofi_kinds = {k for k in cfg.lofi_kinds.split(",") if k}
        self.bf16_acc_kinds = {k for k in cfg.bf16_acc_kinds.split(",") if k}
        self.mm_ckc_lofi_bf16acc = cfg.compute_config(ttnn.MathFidelity.LoFi, fp32_acc=False)
        self.mm_ckc_bf16acc = cfg.compute_config(cfg.mm_fidelity, fp32_acc=False)
        # packer_l1_acc off: SDPA accumulates in its own CBs; with it on, 32x512 SDPA took 2.5 vs 1.6 M cycles.
        self.sdpa_ckc = cfg.compute_config(
            cfg.sdpa_softmax_fidelity or cfg.sdpa_fidelity, fp32_acc=False, packer_l1_acc=False
        )
        self.norm_ckc = cfg.compute_config(ttnn.MathFidelity.HiFi4)
        self.one_tile = ttnn.from_torch(
            torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(cfg.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        cos, sin = precompute_freqs(
            cfg.head_dim, cfg.max_seq_len * 2, cfg.rope_theta, cfg.rope_scale_factor, cfg.rope_orig_context_len
        )
        self._freq_cos, self._freq_sin = cos[: cfg.max_seq_len].float(), sin[: cfg.max_seq_len].float()
        self._cos_host, self._sin_host = gather_cos_sin(torch.arange(cfg.max_seq_len), cos, sin)
        self._rot = {}
        self._norm_cfgs = {}

    # ------------------------------------------------------------------ inputs
    def rot_mats(self, seq_len):
        """Per sequence length: one [1, 1, S, 64] table for matmul_qkv_rope (cos and sin tiles of the
        32 frequencies), or [cos, sin] in rotary_embedding_llama's layout."""
        if seq_len not in self._rot:
            if self.cfg.fuse_qkv_rope:
                self._rot[seq_len] = make_rope_table(self._freq_cos[:seq_len], self._freq_sin[:seq_len], self.device)
            else:
                self._rot[seq_len] = [
                    ttnn.from_torch(t[:, :, :seq_len], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                    for t in (self._cos_host, self._sin_host)
                ]
        return self._rot[seq_len]

    def sdpa_chunks(self, B, S):
        qc, kc = sdpa_chunks(B, S)
        return min(S, self.cfg.sdpa_q_chunk or qc), min(S, self.cfg.sdpa_k_chunk or kc)

    @staticmethod
    def bucket(n):
        return next(b for b in SEQ_BUCKETS if b >= n)

    def prepare_host_inputs(self, batch_ids, seq_len=None):
        """Token id lists -> host tensors (tokens, valid, block map) plus real lengths.

        ``valid`` [B, 1, 32, S] is 1 on real tokens and 0 on padding (all 32 rows equal). The
        device turns it into the [B, 1, S, S] padding mask and uses it as the pooling vector,
        so the host never builds or converts an S x S tensor per request.
        """
        lens = [len(x) for x in batch_ids]
        S = seq_len or self.bucket(max(lens))
        B = len(batch_ids)
        qc, kc = self.sdpa_chunks(B, S)
        tokens = torch.full((B, S), PAD_TOKEN_ID, dtype=torch.int32)
        valid = torch.zeros(B, 1, 32, S, dtype=torch.bfloat16)
        # (q chunk, k chunk) blocks: 0 = no real query row or no real key (skipped), 1 = partly padded keys,
        # 2 = all keys real.
        block_map = torch.zeros(B, 1, (S + qc - 1) // qc, (S + kc - 1) // kc, dtype=torch.int32)
        for b, ids in enumerate(batch_ids):
            n = len(ids)
            tokens[b, :n] = torch.tensor(ids, dtype=torch.int32)
            valid[b, :, :, :n] = 1.0
            block_map[b, 0, : (n + qc - 1) // qc, : (n + kc - 1) // kc] = 1
            # 2 = every key in the k chunk is real, so the block's mask is all zeros: the patched SDPA
            # reader zero-fills those mask tiles in L1 instead of reading them from DRAM.
            block_map[b, 0, : (n + qc - 1) // qc, : n // kc] = 2
        host = (
            ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.from_torch(valid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(block_map, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        )
        return host, lens

    def to_device(self, host_inputs):
        return tuple(t.to(self.device) for t in host_inputs)

    # ----------------------------------------------------------------- forward
    def _mem(self, rows, width, nbytes=2):
        """L1 for activations small enough to fit (``cfg.l1_act_bytes``), else DRAM."""
        if rows * width * nbytes <= self.cfg.l1_act_bytes:
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.DRAM_MEMORY_CONFIG

    def _linear(self, x, w, kind, activation=None, dtype=None, fuse_swiglu=False):
        rows = x.shape[-2] * x.shape[-3]
        if self.cfg.use_minimal_matmul:
            bf16_acc = (
                kind in self.bf16_acc_kinds
                and f"{kind}_bf16acc" in MM_CONFIGS
                and rows >= min(MM_CONFIGS[f"{kind}_bf16acc"])
            )
            mb, kb, nb, sh, sw = mm_block_config(f"{kind}_bf16acc" if bf16_acc else kind, rows, self.cfg.residual_l1)
            if (x.dtype == ttnn.bfloat16 or self.cfg.residual_l1) and kb > 16:
                # The table is swept with bfp8 activations; bf16 in0 blocks of 32 K tiles overflow L1,
                # and so do bfp8 ones next to an L1-resident residual stream.
                kb = 16
            return ttnn.experimental.minimal_matmul(
                x,
                w,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=self.grid,
                ),
                fused_activation=None if activation is None else ttnn.UnaryWithParam(activation),
                compute_kernel_config=(
                    (self.mm_ckc_lofi_bf16acc if bf16_acc else self.mm_ckc_lofi)
                    if kind in self.lofi_kinds
                    else self.mm_ckc_approx
                    if fuse_swiglu and self.cfg.swiglu_approx
                    else (self.mm_ckc_bf16acc if bf16_acc else self.mm_ckc)
                ),
                dtype=dtype or self.cfg.act_dtype,
                memory_config=self._mem(rows, w.shape[-1] // (2 if fuse_swiglu else 1)),
                fuse_swiglu=fuse_swiglu,
            )
        if rows > MAX_MM_ROWS and rows % MAX_MM_ROWS == 0:
            x = ttnn.reshape(x, [1, rows // MAX_MM_ROWS, MAX_MM_ROWS, x.shape[-1]])
        out = ttnn.linear(
            x,
            w,
            activation=None if activation is None else "silu",
            dtype=dtype or self.cfg.act_dtype,
            compute_kernel_config=self.mm_ckc,
            core_grid=self.core_grid,
            memory_config=self._mem(rows, w.shape[-1]),
        )
        return ttnn.reshape(out, [1, 1, rows, out.shape[-1]]) if out.shape[1] != 1 else out

    def _norm(self, x, w, dtype=None):
        rows, width = x.shape[-2] * x.shape[-3], x.shape[-1]
        mem = self._mem(rows, width)
        dtype = dtype or self.cfg.act_dtype
        if rows <= self.cfg.sharded_norm_max_rows:
            shard, pc = self._norm_shard(rows, width)
            xs = ttnn.interleaved_to_sharded(x, shard)
            y = ttnn.rms_norm(
                xs,
                epsilon=self.cfg.norm_eps,
                weight=w,
                program_config=pc,
                compute_kernel_config=self.norm_ckc,
                memory_config=shard,
            )
            ttnn.deallocate(xs)
            out = ttnn.sharded_to_interleaved(y, mem, output_dtype=dtype)
            ttnn.deallocate(y)
            return out
        # The interleaved rms_norm has no output dtype argument; this path stays bf16.
        return ttnn.rms_norm(
            x, epsilon=self.cfg.norm_eps, weight=w, compute_kernel_config=self.norm_ckc, memory_config=mem
        )

    def _norm_shard(self, rows, width):
        """Block-shard [rows, width] over 8 core columns x (one core row per tile-row, up to 8)."""
        key = (rows, width)
        if key not in self._norm_cfgs:
            mt, wt = rows // 32, width // 32
            gx = 8
            gy = max(d for d in range(1, 9) if mt % d == 0)
            shard = ttnn.create_sharded_memory_config(
                shape=(rows // gy, width // gx),
                core_grid=ttnn.CoreGrid(y=gy, x=gx),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                subblock_w=4,
                block_h=mt // gy,
                block_w=wt // gx,
                inplace=False,
            )
            self._norm_cfgs[key] = (shard, pc)
        return self._norm_cfgs[key]

    def _add_norm(self, x, delta, consts, out_dtype=None):
        """(x + delta, rmsnorm(x + delta)) in one op; with ``residual_l1`` the sum overwrites x in L1."""
        rows, width = x.shape[-2] * x.shape[-3], x.shape[-1]
        if self.cfg.residual_l1:
            return fused_add_rmsnorm(
                x,
                delta,
                consts,
                out_dtype=out_dtype or self.cfg.norm_out_dtype,
                fidelity=self.cfg.add_norm_fidelity,
                sum_dtype=x.dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                out_memory_config=self._mem(rows, width),
                inplace=True,
                # Single-buffered: the residual's L1 share leaves no room for a second row of CBs.
                max_buffers=1,
            )
        return fused_add_rmsnorm(
            x,
            delta,
            consts,
            out_dtype=out_dtype or self.cfg.norm_out_dtype,
            fidelity=self.cfg.add_norm_fidelity,
            sum_dtype=self.cfg.residual_dtype,
            memory_config=self._mem(rows, width),
        )

    def _attention(self, layer, xn, B, S, mask, block_map, rope, wo=True):
        """xn: the attention-normed residual [1, 1, B * S, dim]. ``wo=False`` returns the concatenated
        heads before WO (for matmul_add_norm)."""
        cfg = self.cfg
        qc, kc = self.sdpa_chunks(B, S)
        if cfg.fuse_qkv_rope:
            rows = B * S
            keys = [m for m in sorted(QKV_ROPE_CONFIGS) if m <= rows] or [min(QKV_ROPE_CONFIGS)]
            q_r, k_r, v = matmul_qkv_rope(
                xn,
                layer.wqkv,
                rope,
                batch=B,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                block=QKV_ROPE_CONFIGS[keys[-1]],
                grid=self.grid,
                fidelity=cfg.mm_fidelity,
                rope_fidelity=QKV_ROPE_FIDELITY,
                in1_depth=QKV_ROPE_IN1_DEPTH,
                out_dtype=cfg.heads_out_dtype,
                memory_config=self._mem(rows, (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim),
            )
        else:
            q_r, k_r, v = self._qkv_heads(layer, xn, B, S, *rope)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v,
            is_causal=False,
            attn_mask=None if cfg.sdpa_no_mask else mask,
            attn_mask_block_map=block_map if (self.cfg.sdpa_block_map and not cfg.sdpa_no_mask) else None,
            output_concat_heads=cfg.sdpa_concat_heads,
            scale=1.0,
            compute_kernel_config=self.sdpa_ckc,
            memory_config=self._mem(B * S, cfg.dim),
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.grid,
                q_chunk_size=qc,
                k_chunk_size=kc,
                exp_approx_mode=cfg.sdpa_exp_approx,
                matmul_math_fidelity=cfg.sdpa_fidelity,
                max_cores_per_head_batch=cfg.sdpa_max_cores_per_head_batch,
                k_splits=cfg.sdpa_k_splits,
                segmented_accumulation=cfg.sdpa_segmented_accumulation,
            ),
        )
        for t in (q_r, k_r, v):
            ttnn.deallocate(t)
        if cfg.sdpa_concat_heads:
            o = attn
        else:
            o = ttnn.experimental.nlp_concat_heads(attn, memory_config=self._mem(B * S, cfg.dim))
            ttnn.deallocate(attn)
        o = ttnn.reshape(o, [1, 1, B * S, o.shape[-1]])
        if not wo:
            return o
        out = self._linear(o, layer.wo, "wo", dtype=cfg.wo_out_dtype)
        ttnn.deallocate(o)
        return out

    def _qkv_heads(self, layer, xn, B, S, cos, sin):
        """QKV matmul, then head split + RoPE as separate ops (the path before matmul_qkv_rope)."""
        cfg = self.cfg
        qkv = self._linear(xn, layer.wqkv, "qkv", dtype=cfg.qkv_out_dtype)
        if cfg.fuse_heads_rope:
            q_r, k_r, v = fused_heads_rope(
                qkv,
                cos,
                sin,
                self.trans_mat,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                batch=B,
                memory_config=self._mem(B * S, qkv.shape[-1]),
                out_dtype=cfg.heads_out_dtype,
            )
            ttnn.deallocate(qkv)
        else:
            qkv = ttnn.reshape(qkv, [B, 1, S, qkv.shape[-1]])
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=cfg.n_heads,
                num_kv_heads=cfg.n_kv_heads,
                transpose_k_heads=False,
                memory_config=self._mem(B * S, qkv.shape[-1]),
            )
            ttnn.deallocate(qkv)
            q_r = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans_mat, is_decode_mode=False)
            k_r = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self.trans_mat, is_decode_mode=False)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
        return q_r, k_r, v

    def _mlp(self, layer, h, ff2=True):
        """h: the ffn-normed residual [1, 1, rows, dim]. ``ff2=False`` returns the SwiGLU product before FF2."""
        if self.cfg.fuse_swiglu:
            a = self._linear(h, layer.w13, "ff13s", fuse_swiglu=True, dtype=self.cfg.ff2_in_dtype)
        else:
            g = self._linear(h, layer.w1, "ff13")
            u = self._linear(h, layer.w3, "ff13")
            # silu(g) * u in one eltwise op: SiLU is applied to g as it is read.
            a = ttnn.mul(
                g,
                u,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)],
                memory_config=self._mem(g.shape[-2], g.shape[-1]),
            )
            ttnn.deallocate(g)
            ttnn.deallocate(u)
        if not ff2:
            return a
        out = self._linear(a, layer.w2, "ff2", dtype=self.cfg.ff2_out_dtype)
        ttnn.deallocate(a)
        return out

    def _mm_add_norm(self, x, w, res, consts, kind, y_dtype=None):
        """res += x @ w (res: the L1-resident residual stream), returns rmsnorm(res) * gamma."""
        rows = x.shape[-2] * x.shape[-3]
        mpc = -(-(rows // 32) // self.grid.x)
        mb, kb, nb, sh, sw = MM_ADD_NORM_CONFIGS[kind]
        mb = max(d for d in range(1, mb + 1) if mpc % d == 0)
        sh = max(d for d in range(1, sh + 1) if mb % d == 0)
        gamma, inv_w, _eps = consts
        return matmul_add_norm(
            x,
            w,
            res,
            (gamma, inv_w, self.one_tile),
            eps=self.cfg.norm_eps,
            block=(mb, kb, nb, sh, sw),
            grid=self.grid,
            fidelity=ttnn.MathFidelity.LoFi if kind in self.lofi_kinds else self.cfg.mm_fidelity,
            y_dtype=y_dtype or self.cfg.norm_out_dtype,
            memory_config=self._mem(rows, self.cfg.dim),
            y_fidelity="HiFi2" if kind in self.cfg.y_hifi2_kinds.split(",") else MM_ADD_NORM_Y_FIDELITY,
            skip_gamma=self.cfg.fold_norm_gamma,
            in0_mcast=kind in self.cfg.in0_mcast_kinds.split(","),
            in1_depth=MM_ADD_NORM_IN1_DEPTH[kind],
        )

    def _padding_mask(self, valid, S):
        """valid [B, 1, 32, S] (1 real / 0 pad) -> additive mask [B, 1, S, S]: 0 on real keys, MASK_NEG on padding.
        The arithmetic runs on the 32-row tensor (its rows are identical); only the bfp4 result is
        repeated to S rows."""
        row = ttnn.multiply(
            ttnn.subtract(valid, 1.0), -MASK_NEG, dtype=self.cfg.mask_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        mask = ttnn.repeat(row, ttnn.Shape([1, 1, S // 32, 1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(row)
        return mask

    def forward(self, tokens, valid, block_map):
        """Device inputs from ``prepare_host_inputs`` -> [B, 1, 32, 32] fp32 tile, score sum at [b, 0, 0, 0]."""
        B, S = tokens.shape[0], tokens.shape[1]
        rope = self.rot_mats(S)
        mask = self._padding_mask(valid, S)
        fused = self.cfg.fuse_add_norm and B * S >= self.cfg.fuse_add_norm_min_rows
        resid_l1 = fused and self.cfg.residual_l1
        resident = fused and self.cfg.resident_residual
        if resident and self.cfg.embed_gather and S % 32 == 0:
            # The residual stream stays in L1 (sharded like matmul_add_norm's grid); WO and FF2 add
            # into it in place and write the following RMSNorm's output. Each core gathers its slice
            # of the token embeddings straight into its shard, then normalizes it.
            gamma, inv_w, _eps = self.layers[0].attn_norm_consts
            res, xn = embed_rmsnorm(
                tokens,
                self.embed,
                (gamma, inv_w, self.one_tile),
                rows=B * S,
                width=self.cfg.dim,
                eps=self.cfg.norm_eps,
                grid=self.grid,
                y_dtype=self.cfg.norm_out_dtype,
                memory_config=self._mem(B * S, self.cfg.dim),
                y_fidelity=MM_ADD_NORM_Y_FIDELITY,
                skip_gamma=self.cfg.fold_norm_gamma,
            )
        else:
            x = ttnn.embedding(
                tokens,
                self.embed,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.cfg.act_dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG if resid_l1 else ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.reshape(x, [1, 1, B * S, self.cfg.dim])
        if resident:
            if not (self.cfg.embed_gather and S % 32 == 0):
                res = to_resident(x, self.grid)
                ttnn.deallocate(x)
                gamma, inv_w, _eps = self.layers[0].attn_norm_consts
                xn = resident_rmsnorm(
                    res,
                    (gamma, inv_w, self.one_tile),
                    rows=B * S,
                    width=self.cfg.dim,
                    eps=self.cfg.norm_eps,
                    grid=self.grid,
                    y_dtype=self.cfg.norm_out_dtype,
                    memory_config=self._mem(B * S, self.cfg.dim),
                    y_fidelity=MM_ADD_NORM_Y_FIDELITY,
                    skip_gamma=self.cfg.fold_norm_gamma,
                )
            for i, layer in enumerate(self.layers):
                o = self._attention(layer, xn, B, S, mask, block_map, rope, wo=False)
                ttnn.deallocate(xn)
                h = self._mm_add_norm(o, layer.wo, res, layer.ffn_norm_consts, "wo")
                ttnn.deallocate(o)
                a = self._mlp(layer, h, ff2=False)
                ttnn.deallocate(h)
                last = i + 1 == len(self.layers)
                nxt = self.final_norm_consts if last else self.layers[i + 1].attn_norm_consts
                xn = self._mm_add_norm(a, layer.w2, res, nxt, "ff2", self.cfg.final_norm_dtype if last else None)
                ttnn.deallocate(a)
            ttnn.deallocate(res)
        elif fused:
            xn = self._norm(x, self.layers[0].attn_norm, self.cfg.norm_out_dtype)
            for i, layer in enumerate(self.layers):
                a = self._attention(layer, xn, B, S, mask, block_map, rope)
                ttnn.deallocate(xn)
                x_new, h = self._add_norm(x, a, layer.ffn_norm_consts)
                if not resid_l1:
                    ttnn.deallocate(x)
                ttnn.deallocate(a)
                m = self._mlp(layer, h)
                ttnn.deallocate(h)
                nxt = self.layers[i + 1].attn_norm_consts if i + 1 < len(self.layers) else self.final_norm_consts
                last = i + 1 == len(self.layers)
                x, xn = self._add_norm(x_new, m, nxt, out_dtype=self.cfg.act_dtype if last else None)
                if not resid_l1:
                    ttnn.deallocate(x_new)
                ttnn.deallocate(m)
            ttnn.deallocate(x)
        else:
            for layer in self.layers:
                xn = self._norm(x, layer.attn_norm, self.cfg.norm_out_dtype)
                a = self._attention(layer, xn, B, S, mask, block_map, rope)
                ttnn.deallocate(xn)
                x_new = ttnn.add(x, a, memory_config=self._mem(B * S, self.cfg.dim))
                ttnn.deallocate(x)
                ttnn.deallocate(a)
                h = self._norm(x_new, layer.ffn_norm, self.cfg.norm_out_dtype)
                m = self._mlp(layer, h)
                ttnn.deallocate(h)
                x = ttnn.add(x_new, m, memory_config=self._mem(B * S, self.cfg.dim))
                ttnn.deallocate(x_new)
                ttnn.deallocate(m)
            xn = self._norm(x, self.final_norm)
            ttnn.deallocate(x)
        t = self._linear(xn, self.score_w, "score", dtype=ttnn.float32)
        ttnn.deallocate(xn)
        t = ttnn.reshape(t, [B, 1, S, 32])
        # valid's rows are identical, so every output row holds the sum; row 0 is read.
        pooled = ttnn.matmul(valid, t, dtype=ttnn.float32, compute_kernel_config=self.mm_ckc)
        ttnn.deallocate(t)
        ttnn.deallocate(mask)
        return pooled

    def scores_from_output(self, out, lens):
        sums = ttnn.to_torch(out).float()[:, 0, 0, 0]
        return [s / n for s, n in zip(sums[: len(lens)].tolist(), lens)]

    def score(self, batch_ids, seq_len=None):
        """Convenience path (no trace): token id lists -> list of float scores."""
        host, lens = self.prepare_host_inputs(batch_ids, seq_len)
        dev = self.to_device(host)
        out = self.forward(*dev)
        for t in dev:
            ttnn.deallocate(t)
        return self.scores_from_output(out, lens)
