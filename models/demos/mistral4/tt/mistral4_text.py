# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-local TTNN modules for the Mistral-Small-4 (mistral4) text core: MLA + MoE + decoder layer.

Built from generic ttnn ops (no new kernels, no deepseek framework). The recipes here are the
ones PCC-verified bottom-up in tests/multimodal/mistral_24b/test_m4_mla.py and test_m4_moe.py:
  - MLA: q/kv low-rank projections + RMSNorms, interleaved RoPE applied as a permutation-matrix
    matmul (de-interleave) then standard rotate_half, MQA k_rot expand, SDPA, o_proj.
  - MoE: router (linear -> softmax -> top-4/128 kth-threshold mask -> normalize), dense weighted
    SwiGLU experts, plus a shared SwiGLU expert.
Correctness-first: weights replicated across the mesh; experts computed densely. Sharding /
sparse dispatch / flash-MLA / paged / trace are follow-up optimizations.
"""
import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _lin(weight, mesh):
    # HF nn.Linear weight [out, in] -> ttnn [in, out] for x @ W; replicated across the mesh.
    return ttnn.as_tensor(
        weight.transpose(0, 1).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _norm(weight, mesh):
    return ttnn.as_tensor(
        weight.reshape(1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _rotate_half(x):
    d = x.shape[-1]
    x1 = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], d // 2])
    x2 = ttnn.slice(x, [0, 0, 0, d // 2], [x.shape[0], x.shape[1], x.shape[2], d])
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


class TtMistral4MLA(LightweightModule):
    """Multi-head Latent Attention. forward(x, cos, sin) -> [B, S, hidden]."""

    def __init__(self, mesh, sd, cfg, eps):
        super().__init__()
        self.mesh = mesh
        self.eps = eps
        self.H = cfg.num_attention_heads
        self.qk = cfg.qk_head_dim
        self.nope = cfg.qk_nope_head_dim
        self.rope = cfg.qk_rope_head_dim
        self.vd = cfg.v_head_dim
        self.kvl = cfg.kv_lora_rank
        self.scale = self.qk ** (-0.5)

        self.w_qa = _lin(sd["self_attn.q_a_proj.weight"], mesh)
        self.w_qan = _norm(sd["self_attn.q_a_layernorm.weight"], mesh)
        self.w_qb = _lin(sd["self_attn.q_b_proj.weight"], mesh)
        self.w_kva = _lin(sd["self_attn.kv_a_proj_with_mqa.weight"], mesh)
        self.w_kvan = _norm(sd["self_attn.kv_a_layernorm.weight"], mesh)
        self.w_kvb = _lin(sd["self_attn.kv_b_proj.weight"], mesh)
        self.w_o = _lin(sd["self_attn.o_proj.weight"], mesh)

        # de-interleave permutation matrix for interleaved RoPE: out[i]=x[2i] (i<d/2) else x[2(i-d/2)+1]
        perm = [2 * i for i in range(self.rope // 2)] + [2 * i + 1 for i in range(self.rope // 2)]
        P = torch.zeros(self.rope, self.rope)
        for i, p in enumerate(perm):
            P[p, i] = 1.0
        self.P = _lin(P.T, mesh)  # _lin transposes; we want the matrix itself, so pass P.T

    def _rope(self, x, cos, sin):
        xd = ttnn.matmul(x, self.P)  # de-interleave
        return ttnn.add(ttnn.mul(xd, cos), ttnn.mul(_rotate_half(xd), sin))

    def _qkv(self, x, cos, sin):
        """x [B,S,hidden] -> q_states/k_states [B,H,S,qk_head_dim], value [B,H,S,v_head_dim]."""
        B, S = x.shape[0], x.shape[1]
        q = ttnn.linear(ttnn.rms_norm(ttnn.linear(x, self.w_qa), epsilon=self.eps, weight=self.w_qan), self.w_qb)
        qh = ttnn.transpose(ttnn.reshape(q, (B, S, self.H, self.qk)), 1, 2)
        q_nope = ttnn.slice(qh, [0, 0, 0, 0], [B, self.H, S, self.nope])
        q_rot = ttnn.slice(qh, [0, 0, 0, self.nope], [B, self.H, S, self.qk])
        q_states = ttnn.concat([q_nope, self._rope(q_rot, cos, sin)], dim=-1)

        kv_a = ttnn.linear(x, self.w_kva)
        kv_pass = ttnn.rms_norm(ttnn.slice(kv_a, [0, 0, 0], [B, S, self.kvl]), epsilon=self.eps, weight=self.w_kvan)
        kv_b = ttnn.linear(kv_pass, self.w_kvb)
        kh = ttnn.transpose(ttnn.reshape(kv_b, (B, S, self.H, self.nope + self.vd)), 1, 2)
        k_nope = ttnn.slice(kh, [0, 0, 0, 0], [B, self.H, S, self.nope])
        value = ttnn.slice(kh, [0, 0, 0, self.nope], [B, self.H, S, self.nope + self.vd])
        k_rot = ttnn.reshape(ttnn.slice(kv_a, [0, 0, self.kvl], [B, S, self.kvl + self.rope]), (B, 1, S, self.rope))
        k_rot = ttnn.repeat(self._rope(k_rot, cos, sin), ttnn.Shape([1, self.H, 1, 1]))
        k_states = ttnn.concat([k_nope, k_rot], dim=-1)
        return q_states, k_states, value

    def forward(self, x, cos, sin):
        B, S = x.shape[0], x.shape[1]
        q_states, k_states, value = self._qkv(x, cos, sin)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_states, k_states, value, is_causal=True, scale=self.scale
        )
        attn = ttnn.reshape(ttnn.transpose(attn, 1, 2), (B, S, -1))
        return ttnn.linear(attn, self.w_o)

    def init_kv_cache(self, batch, max_seq):
        """Allocate the standard expanded-k/v KV cache [batch, n_heads, max_seq, qk_head_dim]."""
        z = torch.zeros(batch, self.H, max_seq, self.qk, dtype=torch.bfloat16)
        mk = lambda: ttnn.from_torch(
            z, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        return [mk(), mk()]  # [k_cache, v_cache]

    def forward_prefill(self, x, cos, sin, kv_cache):
        """Prefill: full causal attention over the prompt AND populate the KV cache (positions
        0..S-1) so decode can continue. Same math as forward(); adds the cache fill per user."""
        B, S = x.shape[0], x.shape[1]
        q_states, k_states, value = self._qkv(x, cos, sin)  # [B,H,S,qk]
        for b in range(B):
            kf = k_states if B == 1 else ttnn.slice(k_states, [b, 0, 0, 0], [b + 1, self.H, S, self.qk])
            vf = value if B == 1 else ttnn.slice(value, [b, 0, 0, 0], [b + 1, self.H, S, self.qk])
            ttnn.fill_cache(kv_cache[0], kf, b)
            ttnn.fill_cache(kv_cache[1], vf, b)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_states, k_states, value, is_causal=True, scale=self.scale
        )
        attn = ttnn.reshape(ttnn.transpose(attn, 1, 2), (B, S, -1))
        return ttnn.linear(attn, self.w_o)

    def forward_decode(self, x, cur_pos, cos, sin, kv_cache):
        """Single-token decode: x [B,1,hidden], cur_pos int. Writes k/v to kv_cache at cur_pos and
        runs flash-decode over the cached sequence. Expanded-k/v cache => standard decode op."""
        B = x.shape[0]
        q_states, k_states, value = self._qkv(x, cos, sin)  # [B,H,1,*]
        # write the new token's k/v at cur_pos (input [B,H,1,dh])
        ttnn.update_cache(kv_cache[0], k_states, cur_pos)
        ttnn.update_cache(kv_cache[1], value, cur_pos)
        q_dec = ttnn.permute(q_states, (2, 0, 1, 3))  # [B,H,1,qk] -> [1,B,H,qk]
        cur = ttnn.from_torch(
            torch.tensor([cur_pos] * B, dtype=torch.int32),
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_dec, kv_cache[0], kv_cache[1], cur_pos_tensor=cur, scale=self.scale
        )
        attn = ttnn.reshape(ttnn.permute(attn, (1, 0, 2, 3)), (B, 1, self.H * self.qk))
        return ttnn.linear(attn, self.w_o)


class TtMistral4MoE(LightweightModule):
    """Ungrouped top-4/128 MoE + shared SwiGLU expert. forward(x) -> [B, S, hidden].

    shard_experts=True shards the 128 experts across the mesh (E/n_dev per device) so the full
    36-layer model fits in DRAM; partial sums are combined with ttnn.all_reduce(Sum). The expert
    compute (the heavy weights) stays on device; only the tiny routing matrix W is round-tripped
    to host to expert-shard it (negligible; an all-to-all dispatch is the perf path — see #14).
    """

    def __init__(self, mesh, sd, cfg, shard_experts=False, expert_dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh = mesh
        self.E = cfg.n_routed_experts
        self.k = cfg.num_experts_per_tok
        self.interm = cfg.moe_intermediate_size
        self.hidden = cfg.hidden_size
        self.shard = shard_experts
        self.w_gate = _lin(sd["mlp.gate.weight"], mesh)
        gup, down = sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"]  # [E,2I,H],[E,H,I]
        if shard_experts:
            self.per = self.E // mesh.get_num_devices()
            self.gup_sh = ttnn.as_tensor(
                gup.transpose(1, 2).contiguous().to(torch.bfloat16),
                dtype=expert_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )  # [per,H,2I]/device
            self.down_sh = ttnn.as_tensor(
                down.transpose(1, 2).contiguous().to(torch.bfloat16),
                dtype=expert_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )  # [per,I,H]/device
        else:
            self.w_gup = [_lin(gup[e], mesh) for e in range(self.E)]
            self.w_down = [_lin(down[e], mesh) for e in range(self.E)]
        self.w_sg = _lin(sd["mlp.shared_experts.gate_proj.weight"], mesh)
        self.w_su = _lin(sd["mlp.shared_experts.up_proj.weight"], mesh)
        self.w_sd = _lin(sd["mlp.shared_experts.down_proj.weight"], mesh)

    def _route(self, x, B, S):
        # softmax -> kth-largest threshold mask -> normalize (no scatter); W replicated [B,S,E].
        # NB: routing precision is bf16-bounded — ttnn.topk requires bf16/bfp8 input, so top-k
        # selection can't be done in fp32 on device (a borderline expert can flip vs the fp32
        # reference; this is the main source of the full-depth logit-PCC gap, ~0.982). An exact
        # selection would need a host fp32 top-k round-trip per layer (not worth the perf cost).
        probs = ttnn.softmax(ttnn.linear(x, self.w_gate), dim=-1)
        kth = ttnn.slice(ttnn.topk(probs, self.k, dim=-1)[0], [0, 0, self.k - 1], [B, S, self.k])
        masked = ttnn.mul(probs, ttnn.ge(probs, kth))
        return ttnn.div(masked, ttnn.sum(masked, dim=-1, keepdim=True))

    def forward(self, x):
        B, S, I, H = x.shape[0], x.shape[1], self.interm, self.hidden
        W = self._route(x, B, S)
        if self.shard:
            # expert-shard the tiny routing matrix (host round-trip; see class docstring / #14)
            W_host = ttnn.to_torch(W, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))[:B]
            W_sh = ttnn.from_torch(
                W_host, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1)
            )
            # BATCHED experts: the per local experts in 2 batched matmuls (not `per` sequential
            # linears) — the dominant decode/forward speedup. Stacked weights gup_sh [per,H,2I],
            # down_sh [per,I,H]; broadcast x over the expert (batch) dim.
            xb = ttnn.repeat(x, ttnn.Shape([self.per, 1, 1]))  # [per,S,H]
            gu = ttnn.matmul(xb, self.gup_sh)  # [per,S,2I]
            h = ttnn.mul(
                ttnn.silu(ttnn.slice(gu, [0, 0, 0], [self.per, S, I])), ttnn.slice(gu, [0, 0, I], [self.per, S, 2 * I])
            )
            y = ttnn.matmul(h, self.down_sh)  # [per,S,H]
            w = ttnn.permute(W_sh, (2, 1, 0))  # [B,S,per] -> [per,S,B]
            acc = ttnn.reshape(ttnn.sum(ttnn.mul(y, w), dim=0), (B, S, H))  # weighted sum over local experts
            experts = ttnn.all_reduce(acc)  # sum across devices
        else:
            acc = None
            for e in range(self.E):
                gu = ttnn.linear(x, self.w_gup[e])
                g = ttnn.slice(gu, [0, 0, 0], [B, S, I])
                u = ttnn.slice(gu, [0, 0, I], [B, S, 2 * I])
                y = ttnn.linear(ttnn.mul(ttnn.silu(g), u), self.w_down[e])
                contrib = ttnn.mul(y, ttnn.slice(W, [0, 0, e], [B, S, e + 1]))
                acc = contrib if acc is None else ttnn.add(acc, contrib)
            experts = acc
        sh = ttnn.linear(ttnn.mul(ttnn.silu(ttnn.linear(x, self.w_sg)), ttnn.linear(x, self.w_su)), self.w_sd)
        return ttnn.add(experts, sh)


class TtMistral4DecoderLayer(LightweightModule):
    """input_layernorm -> MLA -> +residual -> post_attention_layernorm -> MoE -> +residual."""

    def __init__(self, mesh, sd, cfg, eps, shard_experts=False, expert_dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        self.w_in = _norm(sd["input_layernorm.weight"], mesh)
        self.w_post = _norm(sd["post_attention_layernorm.weight"], mesh)
        self.mla = TtMistral4MLA(mesh, sd, cfg, eps)
        self.moe = TtMistral4MoE(mesh, sd, cfg, shard_experts=shard_experts, expert_dtype=expert_dtype)

    def forward(self, x, cos, sin):
        h = ttnn.add(x, self.mla(ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), cos, sin))
        return ttnn.add(h, self.moe(ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)))

    def forward_decode(self, x, cur_pos, cos, sin, kv_cache):
        h = ttnn.add(
            x,
            self.mla.forward_decode(ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), cur_pos, cos, sin, kv_cache),
        )
        return ttnn.add(h, self.moe(ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)))

    def forward_prefill(self, x, cos, sin, kv_cache):
        h = ttnn.add(
            x, self.mla.forward_prefill(ttnn.rms_norm(x, epsilon=self.eps, weight=self.w_in), cos, sin, kv_cache)
        )
        return ttnn.add(h, self.moe(ttnn.rms_norm(h, epsilon=self.eps, weight=self.w_post)))


class TtMistral4TextModel(LightweightModule):
    """Stacked decoder layers + final norm + LM head. forward(embedded_hidden, cos, sin) -> logits.

    Embedding is the caller's responsibility (a trivial row gather); this keeps the module a pure
    device forward over the decoder stack. `sd` is the model's named_parameters dict with HF keys
    (``model.layers.{i}.*``, ``model.norm.weight``, ``lm_head.weight``).
    """

    def __init__(self, mesh, sd, cfg, n_layers, eps, shard_experts=False, expert_dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        self.layers = []
        for i in range(n_layers):
            pfx = f"model.layers.{i}."
            layer_sd = {k[len(pfx) :]: v for k, v in sd.items() if k.startswith(pfx)}
            self.layers.append(
                TtMistral4DecoderLayer(mesh, layer_sd, cfg, eps, shard_experts=shard_experts, expert_dtype=expert_dtype)
            )
        self.w_norm = _norm(sd["model.norm.weight"], mesh)
        self.w_lm = _lin(sd["lm_head.weight"], mesh)

    def forward(self, hidden, cos, sin):
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        hidden = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm)
        return ttnn.linear(hidden, self.w_lm)

    def init_kv_caches(self, batch, max_seq):
        return [layer.mla.init_kv_cache(batch, max_seq) for layer in self.layers]

    def forward_prefill(self, hidden, cos, sin, kv_caches):
        for layer, kv in zip(self.layers, kv_caches):
            hidden = layer.forward_prefill(hidden, cos, sin, kv)
        hidden = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm)
        return ttnn.linear(hidden, self.w_lm)

    def forward_decode(self, hidden, cur_pos, cos, sin, kv_caches):
        for layer, kv in zip(self.layers, kv_caches):
            hidden = layer.forward_decode(hidden, cur_pos, cos, sin, kv)
        hidden = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.w_norm)
        return ttnn.linear(hidden, self.w_lm)


class TtMistral4Projector(LightweightModule):
    """Mistral3 multi-modal projector: vision features -> text-hidden image embeds.

    norm(RMSNorm) -> 2x2 spatial patch-merge -> merging_layer -> linear_1 -> gelu -> linear_2.
    The 2x2 merge (channel-major, == HF F.unfold) is a one-time-per-image spatial rearrange done
    host-side here (negligible vs decode; an on-device reshape/permute is a perf follow-up); the
    norm + all linears run on device. `sd` keys are relative to multi_modal_projector. (norm.weight,
    patch_merger.merging_layer.weight, linear_1.weight, linear_2.weight).
    """

    def __init__(self, mesh, sd, cfg):
        super().__init__()
        self.mesh = mesh
        self.eps = cfg.text_config.rms_norm_eps
        self.sm = cfg.spatial_merge_size
        self.patch = cfg.vision_config.patch_size
        self.vh = cfg.vision_config.hidden_size
        self.w_norm = _norm(sd["norm.weight"], mesh)
        self.w_merge = _lin(sd["patch_merger.merging_layer.weight"], mesh)  # Linear(vh*sm^2 -> vh)
        self.w_l1 = _lin(sd["linear_1.weight"], mesh)  # Linear(vh -> text_hidden)
        self.w_l2 = _lin(sd["linear_2.weight"], mesh)  # Linear(text_hidden -> text_hidden)

    def _unfold_host(self, x, image_sizes):
        # x [n, vh] (single image) -> [n/sm^2, vh*sm^2], channel-major (== HF F.unfold). Host-side.
        sm, d = self.sm, x.shape[-1]
        h, w = image_sizes[0][0] // self.patch, image_sizes[0][1] // self.patch
        g = x.view(h, w, d).permute(2, 0, 1)  # [d,h,w]
        g = g.view(d, h // sm, sm, w // sm, sm).permute(0, 2, 4, 1, 3).reshape(d * sm * sm, -1).t()
        return g.contiguous()

    def forward(self, feats, image_sizes):
        n = feats.shape[0]
        x = ttnn.rms_norm(feats, epsilon=self.eps, weight=self.w_norm)  # [n, vh]
        x_host = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))[:n].float()
        merged = ttnn.from_torch(
            self._unfold_host(x_host, image_sizes).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )  # [n/sm^2, vh*sm^2]
        m = ttnn.linear(merged, self.w_merge)  # [n/sm^2, vh]
        h = ttnn.gelu(ttnn.linear(m, self.w_l1))  # [n/sm^2, text_hidden]
        return ttnn.linear(h, self.w_l2)  # [n/sm^2, text_hidden]
