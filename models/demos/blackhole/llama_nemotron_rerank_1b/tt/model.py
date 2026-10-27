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
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import (
    PAD_TOKEN_ID,
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
        self.wqkv = loader.linear(
            f"l{i}_wqkv",
            [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"],
            cfg.wqkv_dtype,
            permute_heads=[cfg.n_heads, cfg.n_kv_heads, None],
        )
        self.wo = loader.linear(f"l{i}_wo", [p + "self_attn.o_proj.weight"], cfg.wo_dtype)
        if cfg.fuse_swiglu:
            self.w13 = loader.swiglu(f"l{i}_w13s", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", cfg.w13_dtype)
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
        self.score_w = loader.score_head()

        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.grid = grid
        self.mm_ckc = cfg.compute_config(cfg.mm_fidelity)
        self.mm_ckc_lofi = cfg.compute_config(ttnn.MathFidelity.LoFi)
        self.lofi_kinds = {k for k in cfg.lofi_kinds.split(",") if k}
        self.sdpa_ckc = cfg.compute_config(cfg.sdpa_fidelity, fp32_acc=False)
        self.norm_ckc = cfg.compute_config(ttnn.MathFidelity.HiFi4)
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(cfg.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        cos, sin = precompute_freqs(
            cfg.head_dim, cfg.max_seq_len * 2, cfg.rope_theta, cfg.rope_scale_factor, cfg.rope_orig_context_len
        )
        self._cos_host, self._sin_host = gather_cos_sin(torch.arange(cfg.max_seq_len), cos, sin)
        self._rot = {}
        self._norm_cfgs = {}

    # ------------------------------------------------------------------ inputs
    def rot_mats(self, seq_len):
        if seq_len not in self._rot:
            self._rot[seq_len] = [
                ttnn.from_torch(t[:, :, :seq_len], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                for t in (self._cos_host, self._sin_host)
            ]
        return self._rot[seq_len]

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
        qc, kc = sdpa_chunks(B, S)
        tokens = torch.full((B, S), PAD_TOKEN_ID, dtype=torch.int32)
        valid = torch.zeros(B, 1, 32, S, dtype=torch.bfloat16)
        # (q chunk, k chunk) blocks holding at least one real query row and one real key.
        block_map = torch.zeros(B, 1, (S + qc - 1) // qc, (S + kc - 1) // kc, dtype=torch.int32)
        for b, ids in enumerate(batch_ids):
            n = len(ids)
            tokens[b, :n] = torch.tensor(ids, dtype=torch.int32)
            valid[b, :, :, :n] = 1.0
            block_map[b, 0, : (n + qc - 1) // qc, : (n + kc - 1) // kc] = 1
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
            mb, kb, nb, sh, sw = mm_block_config(kind, rows)
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
                compute_kernel_config=self.mm_ckc_lofi if kind in self.lofi_kinds else self.mm_ckc,
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

    def _norm(self, x, w):
        rows, width = x.shape[-2] * x.shape[-3], x.shape[-1]
        mem = self._mem(rows, width)
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
            out = ttnn.sharded_to_interleaved(y, mem)
            ttnn.deallocate(y)
            return out
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

    def _add_norm(self, x, delta, consts):
        """(x + delta, rmsnorm(x + delta)) in one op."""
        return fused_add_rmsnorm(x, delta, consts, memory_config=self._mem(x.shape[-2] * x.shape[-3], x.shape[-1]))

    def _attention(self, layer, xn, B, S, mask, block_map, cos, sin):
        """xn: the attention-normed residual [1, 1, B * S, dim]."""
        cfg = self.cfg
        qc, kc = sdpa_chunks(B, S)
        qkv = self._linear(xn, layer.wqkv, "qkv")
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
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v,
            is_causal=False,
            attn_mask=mask,
            attn_mask_block_map=block_map if self.cfg.sdpa_block_map else None,
            output_concat_heads=cfg.sdpa_concat_heads,
            scale=1.0 / math.sqrt(cfg.head_dim),
            compute_kernel_config=self.sdpa_ckc,
            memory_config=self._mem(B * S, cfg.dim),
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.grid,
                q_chunk_size=qc,
                k_chunk_size=kc,
                exp_approx_mode=False,
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
        out = self._linear(o, layer.wo, "wo")
        ttnn.deallocate(o)
        return out

    def _mlp(self, layer, h):
        """h: the ffn-normed residual [1, 1, rows, dim]."""
        if self.cfg.fuse_swiglu:
            a = self._linear(h, layer.w13, "ff13s", fuse_swiglu=True)
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
        out = self._linear(a, layer.w2, "ff2")
        ttnn.deallocate(a)
        return out

    def _padding_mask(self, valid, S):
        """valid [B, 1, 32, S] (1 real / 0 pad) -> additive mask [B, 1, S, S]: 0 on real keys, MASK_NEG on padding."""
        rows = ttnn.repeat(valid, ttnn.Shape([1, 1, S // 32, 1]))
        mask = ttnn.multiply(
            ttnn.subtract(rows, 1.0), -MASK_NEG, dtype=self.cfg.mask_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(rows)
        return mask

    def forward(self, tokens, valid, block_map):
        """Device inputs from ``prepare_host_inputs`` -> [B, 1, 32, 32] fp32 tile, score sum at [b, 0, 0, 0]."""
        B, S = tokens.shape[0], tokens.shape[1]
        cos, sin = self.rot_mats(S)
        mask = self._padding_mask(valid, S)
        x = ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.act_dtype)
        x = ttnn.reshape(x, [1, 1, B * S, self.cfg.dim])
        if self.cfg.fuse_add_norm and B * S >= self.cfg.fuse_add_norm_min_rows:
            xn = self._norm(x, self.layers[0].attn_norm)
            for i, layer in enumerate(self.layers):
                a = self._attention(layer, xn, B, S, mask, block_map, cos, sin)
                ttnn.deallocate(xn)
                x_new, h = self._add_norm(x, a, layer.ffn_norm_consts)
                ttnn.deallocate(x)
                ttnn.deallocate(a)
                m = self._mlp(layer, h)
                ttnn.deallocate(h)
                nxt = self.layers[i + 1].attn_norm_consts if i + 1 < len(self.layers) else self.final_norm_consts
                x, xn = self._add_norm(x_new, m, nxt)
                ttnn.deallocate(x_new)
                ttnn.deallocate(m)
            ttnn.deallocate(x)
        else:
            for layer in self.layers:
                xn = self._norm(x, layer.attn_norm)
                a = self._attention(layer, xn, B, S, mask, block_map, cos, sin)
                ttnn.deallocate(xn)
                x_new = ttnn.add(x, a, memory_config=self._mem(B * S, self.cfg.dim))
                ttnn.deallocate(x)
                ttnn.deallocate(a)
                h = self._norm(x_new, layer.ffn_norm)
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
