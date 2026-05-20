# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Code Predictor — fp32 activation path, fully on-device, trace-compatible.

Mirrors production CodePredictor API but every op is ttnn (no host roundtrips
inside forward_single_step). Activations and KV cache are fp32 throughout;
weights stay bf16 (PCC neutral). RoPE casts Q/K to bf16 (kernel is bf16-only;
rotation only — small precision cost).

Lifetime contract: forward_single_step does NOT deallocate the caller's
`inputs_embeds`; per-layer residuals owned by forward_single_step are freed
after each layer. ttnn.slice was avoided in favor of running lm_head over
the full hidden state — caller indexes the last position.
"""
from typing import List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    dram_sharded_program_config,
    width_sharded_l1_memcfg,
)


def _pick_grid_cp_block_w2(k_tiles: int, n_tiles: int, max_rows: int = 10, max_cols: int = 13):
    """Pick (rows, cols) so num_cores divides both K_tiles and N_tiles and
    in0_block_w = K_tiles / num_cores ≥ 2."""
    max_cores = max_rows * max_cols
    candidates = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0 and (k_tiles // c) >= 2]
    if not candidates:
        # Fallback: largest grid that divides both (any block_w).
        candidates = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
    candidates.sort(reverse=True)
    for cores in candidates:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and (cores // rows) <= max_cols:
                return rows, cores // rows
    return 1, 1


class CodePredictor(LightweightModule):
    def __init__(
        self,
        device,
        config,
        talker_hidden_size: int,
        state_dict: dict,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.hidden_size = config.hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_code_groups = config.num_code_groups
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.rms_norm_eps = config.rms_norm_eps
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / (self.head_dim**0.5)

        DRAM = ttnn.DRAM_MEMORY_CONFIG
        L1 = ttnn.L1_MEMORY_CONFIG
        TILE = ttnn.TILE_LAYOUT
        ROW = ttnn.ROW_MAJOR_LAYOUT

        # bf16 activations (was float32) — drops ~50 TypecastDeviceOperations across
        # the 5-layer CP block (≈100 µs saving per CP_Prefill / CP_Decode replay).
        # RoPE / KV cache / SDPA all run native bf16 now.
        self.act_dtype = ttnn.bfloat16
        self.kcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # --- weight format helpers ---
        def _perm_rope_rows(w_2d: torch.Tensor, head_dim: int) -> torch.Tensor:
            out_dim = w_2d.shape[0]
            assert out_dim % head_dim == 0
            nh = out_dim // head_dim
            half = head_dim // 2
            idx = torch.arange(head_dim).view(2, half).t().reshape(-1)
            w = w_2d.view(nh, head_dim, w_2d.shape[1])
            w = w[:, idx, :].contiguous()
            return w.view(out_dim, w_2d.shape[1])

        def _perm_rope_vec(v_1d: torch.Tensor, head_dim: int) -> torch.Tensor:
            half = head_dim // 2
            idx = torch.arange(head_dim).view(2, half).t().reshape(-1)
            return v_1d.view(head_dim)[idx].contiguous()

        def w_to_tt(w_2d, dt=ttnn.bfloat16):
            w_host = w_2d.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
            return ttnn.from_torch(w_host, device=device, dtype=dt, layout=TILE, memory_config=DRAM)

        def norm_w_1d_to_tt(w_1d, dim, *, permute_rope=False):
            w = _perm_rope_vec(w_1d, dim) if permute_rope else w_1d
            w_host = w.to(torch.bfloat16).view(1, 1, dim // 32, 32).contiguous()
            return ttnn.from_torch(w_host, device=device, dtype=ttnn.bfloat16, layout=ROW, memory_config=DRAM)

        # input projection
        proj_key = "talker.code_predictor.small_to_mtp_projection.weight"
        bias_key = "talker.code_predictor.small_to_mtp_projection.bias"
        self.needs_projection = talker_hidden_size != config.hidden_size
        if self.needs_projection and proj_key in state_dict:
            self.input_proj = w_to_tt(state_dict[proj_key])
            if bias_key in state_dict:
                b = state_dict[bias_key]
                bias_tt = ttnn.from_torch(
                    b.to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ROW,
                    memory_config=DRAM,
                )
                self.input_proj_bias = ttnn.reshape(bias_tt, [1, 1, 1, int(b.shape[0])], memory_config=DRAM)
            else:
                self.input_proj_bias = None
        else:
            self.needs_projection = False
            self.input_proj = None
            self.input_proj_bias = None

        # Helper: build DRAM-sharded weight + matching width-sharded in0/out memcfgs
        # + DRAM-sharded matmul progcfg for a single linear projection. Picks grid
        # so in0_block_w = K_tiles / num_cores ≥ 2 (halves inner-loop iterations).
        def _build_sharded(weight_torch_2d):
            # weight_torch_2d is [N_out, K_in] from state_dict; transpose for K×N.
            w_kn = weight_torch_2d.transpose(-2, -1).contiguous()
            ws, k, n_padded = build_dram_sharded_weight(w_kn, device, dtype=ttnn.bfloat16)
            k_tiles, n_tiles = k // 32, n_padded // 32
            rows, cols = _pick_grid_cp_block_w2(k_tiles, n_tiles)
            num_cores = rows * cols
            progcfg = dram_sharded_program_config(m=32, k=k, n=n_padded, num_cores=num_cores)
            in0_mc = width_sharded_l1_memcfg(m_tiles=1, k_tiles=k_tiles, num_cores_x=cols, num_cores_y=rows)
            out_mc = width_sharded_l1_memcfg(m_tiles=1, k_tiles=n_tiles, num_cores_x=cols, num_cores_y=rows)
            return {"w": ws, "progcfg": progcfg, "in0_mc": in0_mc, "out_mc": out_mc, "n_padded": n_padded}

        # per-layer weights — fused QKV plain stack [Q | K | V] for the regular
        # (non-DRAM-sharded) nlp_create_qkv_heads kernel. Q/K rows are RoPE-permuted
        # so rotary_embedding_llama's interleaved format works directly.
        H = self.hidden_size
        NH, NKV, HD = self.num_heads, self.num_kv_heads, self.head_dim
        self.layers_w = []
        for li in range(self.num_layers):
            pfx = f"talker.code_predictor.model.layers.{li}."
            lw_torch = {k.replace(pfx, ""): v for k, v in state_dict.items() if k.startswith(pfx)}
            q_w = _perm_rope_rows(lw_torch["self_attn.q_proj.weight"], HD)
            k_w = _perm_rope_rows(lw_torch["self_attn.k_proj.weight"], HD)
            v_w = lw_torch["self_attn.v_proj.weight"]
            wqkv_plain = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
            self.layers_w.append(
                {
                    "input_ln_w": norm_w_1d_to_tt(lw_torch["input_layernorm.weight"], H),
                    "post_ln_w": norm_w_1d_to_tt(lw_torch["post_attention_layernorm.weight"], H),
                    "wqkv": w_to_tt(wqkv_plain),
                    "o_proj": w_to_tt(lw_torch["self_attn.o_proj.weight"]),
                    "gate": w_to_tt(lw_torch["mlp.gate_proj.weight"]),
                    "up": w_to_tt(lw_torch["mlp.up_proj.weight"]),
                    "down": w_to_tt(lw_torch["mlp.down_proj.weight"]),
                    "q_norm_w": norm_w_1d_to_tt(lw_torch["self_attn.q_norm.weight"], HD, permute_rope=True),
                    "k_norm_w": norm_w_1d_to_tt(lw_torch["self_attn.k_norm.weight"], HD, permute_rope=True),
                    # DRAM-sharded variants for the 5 large linear projections.
                    "wqkv_sh": _build_sharded(wqkv_plain),
                    "o_proj_sh": _build_sharded(lw_torch["self_attn.o_proj.weight"]),
                    "gate_sh": _build_sharded(lw_torch["mlp.gate_proj.weight"]),
                    "up_sh": _build_sharded(lw_torch["mlp.up_proj.weight"]),
                    "down_sh": _build_sharded(lw_torch["mlp.down_proj.weight"]),
                }
            )

        self.final_norm_w = norm_w_1d_to_tt(state_dict["talker.code_predictor.model.norm.weight"], H)

        self.lm_heads = []
        for g in range(self.num_code_groups - 1):
            k = f"talker.code_predictor.lm_head.{g}.weight"
            self.lm_heads.append(w_to_tt(state_dict[k]))

        self.codec_embeddings_tt: List[Optional[ttnn.Tensor]] = []
        for i in range(self.num_code_groups - 1):
            k = f"talker.code_predictor.model.codec_embedding.{i}.weight"
            if k in state_dict:
                w = state_dict[k]
                vocab_size, emb_dim = int(w.shape[0]), int(w.shape[1])
                e_tt = ttnn.from_torch(
                    w.to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ROW,
                    memory_config=DRAM,
                )
                self.codec_embeddings_tt.append(ttnn.reshape(e_tt, [1, 1, vocab_size, emb_dim], memory_config=DRAM))
            else:
                self.codec_embeddings_tt.append(None)

    # ─── Per-layer forward — caller owns input h_tt; we do NOT deallocate it. ───
    def _layer_forward(
        self,
        h_tt: ttnn.Tensor,
        lw: dict,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
        start_pos: int,
        mode: str,
        cur_pos_tensor: Optional[ttnn.Tensor],
        decode_attn_mask: Optional[ttnn.Tensor],
        cp_prefill_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        # residual aliases h_tt (caller-owned). Do NOT deallocate it.
        residual = h_tt
        x = ttnn.rms_norm(
            h_tt,
            epsilon=self.rms_norm_eps,
            weight=lw["input_ln_w"],
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # QKV (K=1024) — default L1_INTERLEAVED beats DRAM-sharded by ~17% on this
        # shape (128 cores vs 80, and in0_block_w=2 setup cost doesn't pay off here).
        xqkv = ttnn.matmul(
            x,
            lw["wqkv"],
            dtype=self.act_dtype,
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        q_n = ttnn.rms_norm(
            q,
            epsilon=self.rms_norm_eps,
            weight=lw["q_norm_w"],
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        q = q_n
        k_n = ttnn.rms_norm(
            k,
            epsilon=self.rms_norm_eps,
            weight=lw["k_norm_w"],
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(k)
        k = k_n

        # RoPE expects bf16 — cast in/out.
        if q.dtype != ttnn.bfloat16:
            q_b = ttnn.typecast(q, dtype=ttnn.bfloat16)
            ttnn.deallocate(q)
            q = q_b
        if k.dtype != ttnn.bfloat16:
            k_b = ttnn.typecast(k, dtype=ttnn.bfloat16)
            ttnn.deallocate(k)
            k = k_b
        q_r = ttnn.experimental.rotary_embedding_llama(
            q,
            cos,
            sin,
            transformation_mat,
            is_decode_mode=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.kcfg,
        )
        ttnn.deallocate(q)
        q = q_r
        k_r = ttnn.experimental.rotary_embedding_llama(
            k,
            cos,
            sin,
            transformation_mat,
            is_decode_mode=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.kcfg,
        )
        ttnn.deallocate(k)
        k = k_r

        # KV cache write/read.
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache.dtype != k.dtype:
                k_w = ttnn.typecast(k, dtype=k_cache.dtype)
                ttnn.deallocate(k)
                k = k_w
            if v_cache.dtype != v.dtype:
                v_w = ttnn.typecast(v, dtype=v_cache.dtype)
                ttnn.deallocate(v)
                v = v_w
            if mode == "prefill":
                ttnn.fill_cache(k_cache, k, 0)
                ttnn.fill_cache(v_cache, v, 0)
            else:
                ttnn.update_cache(k_cache, k, update_idx=start_pos)
                ttnn.update_cache(v_cache, v, update_idx=start_pos)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if k_cache.dtype == self.act_dtype:
                k_for_attn = k_cache
                v_for_attn = v_cache
                k_cache_alias = True
            else:
                k_for_attn = ttnn.typecast(k_cache, dtype=self.act_dtype)
                v_for_attn = ttnn.typecast(v_cache, dtype=self.act_dtype)
                k_cache_alias = False
            updated_kv = (k_cache, v_cache)
        else:
            if k.dtype != self.act_dtype:
                k_f = ttnn.typecast(k, dtype=self.act_dtype)
                ttnn.deallocate(k)
                k = k_f
            if v.dtype != self.act_dtype:
                v_f = ttnn.typecast(v, dtype=self.act_dtype)
                ttnn.deallocate(v)
                v = v_f
            k_for_attn = k
            v_for_attn = v
            k_cache_alias = False
            updated_kv = None

        if q.dtype != self.act_dtype:
            q_f = ttnn.typecast(q, dtype=self.act_dtype)
            ttnn.deallocate(q)
            q = q_f

        # GQA: repeat KV heads.
        if self.num_kv_groups > 1:
            k_exp = ttnn.repeat_interleave(k_for_attn, self.num_kv_groups, dim=1)
            v_exp = ttnn.repeat_interleave(v_for_attn, self.num_kv_groups, dim=1)
            if not k_cache_alias and kv_cache is not None:
                ttnn.deallocate(k_for_attn)
                ttnn.deallocate(v_for_attn)
            elif kv_cache is None:
                ttnn.deallocate(k_for_attn)
                ttnn.deallocate(v_for_attn)
        else:
            k_exp = k_for_attn
            v_exp = v_for_attn

        # Manual fp32 SDPA chain.
        scores = ttnn.matmul(
            q,
            k_exp,
            transpose_b=True,
            dtype=self.act_dtype,
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        if self.num_kv_groups > 1:
            ttnn.deallocate(k_exp)
        scores = ttnn.mul(scores, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Mask: caller masks override; otherwise build causal mask if seq>1, or
        # decode-position mask if cache is set (so attention ignores stale slots).
        q_seq = int(scores.shape[-2])
        k_seq_eff = int(scores.shape[-1])
        if decode_attn_mask is not None:
            scores = ttnn.add(scores, decode_attn_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
        elif cp_prefill_mask is not None:
            scores = ttnn.add(scores, cp_prefill_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
        elif q_seq > 1:
            mask_cpu = torch.full((q_seq, k_seq_eff), float("-inf"), dtype=torch.float32)
            for i in range(q_seq):
                mask_cpu[i, : i + 1] = 0.0
            mask_tt = ttnn.from_torch(
                mask_cpu.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            scores = ttnn.add(scores, mask_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(mask_tt)
        elif kv_cache is not None:
            valid = start_pos + 1
            if k_seq_eff > valid:
                mask_cpu = torch.full((1, k_seq_eff), float("-inf"), dtype=torch.float32)
                mask_cpu[0, :valid] = 0.0
                mask_tt = ttnn.from_torch(
                    mask_cpu.unsqueeze(0).unsqueeze(0),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                scores = ttnn.add(scores, mask_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(mask_tt)

        attn_weights = ttnn.softmax(scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(scores)
        attn_out = ttnn.matmul(
            attn_weights,
            v_exp,
            dtype=self.act_dtype,
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_weights)
        if self.num_kv_groups > 1:
            ttnn.deallocate(v_exp)

        attn_concat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        attn_sh = ttnn.to_memory_config(attn_concat, lw["o_proj_sh"]["in0_mc"])
        ttnn.deallocate(attn_concat)
        o_sh = ttnn.linear(
            attn_sh,
            lw["o_proj_sh"]["w"],
            dtype=self.act_dtype,
            compute_kernel_config=self.kcfg,
            program_config=lw["o_proj_sh"]["progcfg"],
            memory_config=lw["o_proj_sh"]["out_mc"],
        )
        ttnn.deallocate(attn_sh)
        o_padded = ttnn.to_memory_config(o_sh, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(o_sh)
        if lw["o_proj_sh"]["n_padded"] != self.hidden_size:
            o = ttnn.slice(
                o_padded,
                [0, 0, 0, 0],
                [o_padded.shape[0], o_padded.shape[1], o_padded.shape[2], self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(o_padded)
        else:
            o = o_padded

        # Residual + post-norm. residual = caller's h_tt — DO NOT deallocate.
        h_post = ttnn.add(residual, o, dtype=self.act_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(o)

        residual2 = h_post  # we own h_post → free after MLP residual.
        h2 = ttnn.rms_norm(
            h_post,
            epsilon=self.rms_norm_eps,
            weight=lw["post_ln_w"],
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # gate/up (K=1024) — default L1_INTERLEAVED beats DRAM-sharded (same as QKV).
        gate_o = ttnn.matmul(
            h2, lw["gate"], dtype=self.act_dtype, compute_kernel_config=self.kcfg, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        up_o = ttnn.matmul(
            h2, lw["up"], dtype=self.act_dtype, compute_kernel_config=self.kcfg, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(h2)
        gate_silu = ttnn.silu(gate_o, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gate_o)
        gated = ttnn.mul(gate_silu, up_o, dtype=self.act_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gate_silu)
        ttnn.deallocate(up_o)
        # down (K=3072 = K_tiles=96) wins big with DRAM-sharded.
        gated_sh = ttnn.to_memory_config(gated, lw["down_sh"]["in0_mc"])
        ttnn.deallocate(gated)
        mlp_sh = ttnn.linear(
            gated_sh,
            lw["down_sh"]["w"],
            dtype=self.act_dtype,
            compute_kernel_config=self.kcfg,
            program_config=lw["down_sh"]["progcfg"],
            memory_config=lw["down_sh"]["out_mc"],
        )
        ttnn.deallocate(gated_sh)
        mlp_padded = ttnn.to_memory_config(mlp_sh, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mlp_sh)
        if lw["down_sh"]["n_padded"] != self.hidden_size:
            mlp_o = ttnn.slice(
                mlp_padded,
                [0, 0, 0, 0],
                [mlp_padded.shape[0], mlp_padded.shape[1], mlp_padded.shape[2], self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_padded)
        else:
            mlp_o = mlp_padded
        out = ttnn.add(residual2, mlp_o, dtype=self.act_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual2)
        ttnn.deallocate(mlp_o)
        return out, updated_kv

    # ─── Public API matching production CodePredictor.forward_single_step ──
    def forward_single_step(
        self,
        inputs_embeds,
        cos,
        sin,
        transformation_mat,
        generation_step: int = 1,
        attention_mask=None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_attn_mask: Optional[ttnn.Tensor] = None,
        cp_prefill_mask: Optional[ttnn.Tensor] = None,
        return_hidden_state: bool = False,
    ):
        """Trace-compatible fp32 CP forward. Caller retains ownership of inputs_embeds."""
        if self.needs_projection:
            h = ttnn.linear(
                inputs_embeds,
                self.input_proj,
                bias=self.input_proj_bias if self.input_proj_bias is not None else None,
                dtype=self.act_dtype,
                compute_kernel_config=self.kcfg,
            )
            own_h = True
        else:
            h = inputs_embeds
            own_h = False

        updated_kvs = [] if kv_caches is not None else None
        for li, lw in enumerate(self.layers_w):
            layer_kv = kv_caches[li] if kv_caches is not None else None
            h_new, updated_kv = self._layer_forward(
                h,
                lw,
                cos,
                sin,
                transformation_mat,
                kv_cache=layer_kv,
                start_pos=start_pos,
                mode=mode,
                cur_pos_tensor=cur_pos_tensor,
                decode_attn_mask=decode_attn_mask,
                cp_prefill_mask=cp_prefill_mask,
            )
            if own_h:
                ttnn.deallocate(h)
            h = h_new
            own_h = True
            if updated_kvs is not None:
                updated_kvs.append(updated_kv)

        h_norm = ttnn.rms_norm(
            h,
            epsilon=self.rms_norm_eps,
            weight=self.final_norm_w,
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if own_h:
            ttnn.deallocate(h)

        if return_hidden_state:
            return h_norm, updated_kvs

        # Apply lm_head over full hidden (caller indexes last position).
        lm_idx = generation_step - 1
        logits = ttnn.matmul(
            h_norm,
            self.lm_heads[lm_idx],
            dtype=self.act_dtype,
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(h_norm)
        return logits, updated_kvs

    def get_codec_embedding(self, code_idx: int, token_ids_tt, *, mode: str = "decode"):
        if code_idx < len(self.codec_embeddings_tt) and self.codec_embeddings_tt[code_idx] is not None:
            return ttnn.embedding(
                token_ids_tt,
                self.codec_embeddings_tt[code_idx],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        raise ValueError(f"Missing TTNN codec embedding for index {code_idx}")
