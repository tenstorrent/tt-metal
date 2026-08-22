# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder for poolside/Laguna-XS-2.1.

Laguna-XS-2.1 is Poolside's MoE decoder. Each layer is one of two attention kinds
combined with either a dense MLP (layer 0 only) or a sigmoid-routed MoE block:

  * ``full_attention``  – 48 query heads, YARN RoPE (theta 5e5, factor 32,
    partial_rotary_factor 0.5 -> rotary_dim 64), no sliding window.
  * ``sliding_attention`` – 64 query heads, default RoPE (theta 1e4,
    partial_rotary_factor 1.0 -> rotary_dim 128), sliding_window 512.

Both kinds share: head_dim 128, 8 KV heads (GQA), per-head RMS QK-norm applied
*before* RoPE, and a per-head softplus output gate ``attn = attn * softplus(g_proj(x))``
applied *before* o_proj. The MoE block uses a sigmoid router with an
auxiliary-loss-free ``e_score_correction_bias`` (added for top-k *selection* only),
``norm_topk_prob`` normalisation, ``moe_routed_scaling_factor`` 2.5, 256 experts /
top-8, and an always-on shared expert.

Prefill/decode contract
------------------------
``FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx,
mesh_device, max_seq_len, ...)`` builds the device module. Weight conversion,
RoPE-table construction and paged-cache allocation all happen at setup; the
runtime prefill/decode paths stay on device (no torch / from_torch / to_torch).

* ``prefill_forward(x_BSH, kv_cache, page_table, *, fill_page_table=None, user_id=0, start_pos=0)``
  processes one user's prompt ``x`` of shape ``[1, seq, hidden]`` (any logical
  ``seq``; internal chunking owns padding/masking), fills that user's paged KV
  slot and returns ``[1, seq, hidden]``. ``fill_page_table`` can carry ``-1``
  write-skip padding while ``page_table`` remains valid for attention reads.
* ``decode_forward(x_1BH, cur_pos, rope_idx, page_table, kv_cache)`` processes one
  token for a batch of ``B`` users (``x`` shape ``[1, 1, B, hidden]``), updates the
  cache at the per-user ``cur_pos`` (int32 device tensor, traceable) and returns
  ``[1, 1, B, hidden]``.

The paged KV cache is a pair of tensors shaped
``[max_num_blocks, num_kv_heads, block_size, head_dim]``; ``page_table`` is
``[num_users, blocks_per_user]`` int32.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from .prefill_page_table import single_shot_fill_page_table

TILE = 32


# --------------------------------------------------------------------------- #
# Per-layer configuration
# --------------------------------------------------------------------------- #
@dataclass
class LayerConfig:
    hidden: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    num_kv_groups: int
    scaling: float
    attention_type: str
    is_sliding: bool
    sliding_window: int | None
    rotary_dim: int
    eps: float
    is_moe: bool
    # dense
    intermediate: int
    # moe
    num_experts: int
    top_k: int
    moe_intermediate: int
    shared_intermediate: int
    routed_scaling: float
    norm_topk_prob: bool

    @classmethod
    def from_hf(cls, hf_config, layer_idx: int) -> "LayerConfig":
        layer_types = getattr(hf_config, "layer_types", None)
        attention_type = layer_types[layer_idx] if layer_types else "full_attention"
        is_sliding = attention_type == "sliding_attention"
        per_layer_heads = getattr(hf_config, "num_attention_heads_per_layer", None)
        num_heads = per_layer_heads[layer_idx] if per_layer_heads else hf_config.num_attention_heads
        head_dim = hf_config.head_dim
        nkv = hf_config.num_key_value_heads
        # rope: pick sub-dict and derive rotary_dim from partial_rotary_factor
        rp = hf_config.rope_parameters
        sub = rp["sliding_attention"] if is_sliding else rp["full_attention"]
        partial = sub.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial)
        is_moe = (layer_idx not in hf_config.mlp_only_layers) and (
            hf_config.num_experts > 0 and (layer_idx + 1) % hf_config.decoder_sparse_step == 0
        )
        return cls(
            hidden=hf_config.hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=nkv,
            num_kv_groups=num_heads // nkv,
            scaling=head_dim**-0.5,
            attention_type=attention_type,
            is_sliding=is_sliding,
            sliding_window=hf_config.sliding_window if is_sliding else None,
            rotary_dim=rotary_dim,
            eps=hf_config.rms_norm_eps,
            is_moe=is_moe,
            intermediate=hf_config.intermediate_size,
            num_experts=hf_config.num_experts,
            top_k=hf_config.num_experts_per_tok,
            moe_intermediate=hf_config.moe_intermediate_size,
            shared_intermediate=hf_config.shared_expert_intermediate_size,
            routed_scaling=float(getattr(hf_config, "moe_routed_scaling_factor", 1.0)),
            norm_topk_prob=hf_config.norm_topk_prob,
        )


# --------------------------------------------------------------------------- #
# Setup-time helpers
# --------------------------------------------------------------------------- #
def _rectangular_grid(n: int) -> tuple[int, int]:
    """Rectangular (x, y) core grid whose product equals the number of 32-wide
    N-tiles ``ceil(n/32)`` (so per_core_N=1 and num_blocks==num_cores), bounded to
    the 11x10 Blackhole compute grid. Falls back to the largest fitting divisor."""
    nt = int(math.ceil(n / TILE))
    max_x, max_y = 8, 8
    best = None
    for cores in range(min(nt, max_x * max_y), 0, -1):
        if nt % cores != 0:
            continue
        for x in range(min(cores, max_x), 0, -1):
            if cores % x == 0 and cores // x <= max_y:
                return (x, cores // x)
    return best or (1, 1)


def _sparse_pc(n: int, m: int, k: int, in0_block_w: int = 4):
    core_x, core_y = _rectangular_grid(n)
    num_cores = core_x * core_y
    nt = int(math.ceil(n / TILE))
    per_core_N = (nt + num_cores - 1) // num_cores
    kt = int(math.ceil(k / TILE))
    if kt % in0_block_w != 0:
        divs = [d for d in range(2, in0_block_w + 1) if kt % d == 0]
        in0_block_w = max(divs) if divs else kt
    # mcast_in0: in0 (activations) is multicast to every core, each core computes all
    # M tiles for its N-slice, so per_core_M must span ALL M tiles (num_blocks_M == 1).
    per_core_M = max(1, (m + TILE - 1) // TILE)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _hf_rope_tables(hf_config, attention_type: str, max_seq_len: int):
    """Build cos/sin tables [max_seq_len, rotary_dim] using the exact HF Laguna
    rotary embedding (YARN attention_scaling baked in)."""
    import copy

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_id = hf_config._name_or_path or "poolside/Laguna-XS-2.1"
    RE = get_class_from_dynamic_module("modeling_laguna.LagunaRotaryEmbedding", model_id)
    rp = hf_config.rope_parameters
    cfg = copy.deepcopy(hf_config)
    if attention_type == "sliding_attention":
        cfg.rope_parameters = dict(hf_config.swa_rope_parameters)
        cfg.partial_rotary_factor = cfg.rope_parameters.get("partial_rotary_factor")
    else:
        cfg.rope_parameters = dict(rp["full_attention"])
    re = RE(config=cfg)
    pos = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.zeros(1, max_seq_len, 1)
    cos, sin = re(dummy, pos)  # [1, max_seq_len, rotary_dim]
    return cos[0].float(), sin[0].float()


def _as_tt(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _linear_w(t, device, dtype=ttnn.bfloat16):
    """HF linear weight [out, in] -> ttnn [in, out] for ttnn.linear."""
    return _as_tt(t.t().contiguous(), device, dtype)


@dataclass
class PagedConfig:
    block_size: int = 32
    max_num_blocks: int = 0  # set from max_seq_len * max_users at alloc time


# --------------------------------------------------------------------------- #
# Decoder module
# --------------------------------------------------------------------------- #
class FunctionalDecoder(LightweightModule):
    def __init__(self, cfg: LayerConfig, weights: dict, cos_table, sin_table, mesh_device):
        self.cfg = cfg
        self.w = weights
        self.cos_2d = cos_table  # [max_seq, rotary_dim] ROW_MAJOR bf16 (decode gather)
        self.sin_2d = sin_table
        self.device = mesh_device
        self._compute_kernel = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Precise SDPA: exp_approx_mode=False + fp32 accumulation matter for long-context
        # attention (bf16 approximate-exp over O(1e5) keys loses accuracy).
        grid = mesh_device.compute_with_storage_grid_size()
        self._sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len, **kwargs):
        cfg = LayerConfig.from_hf(hf_config, layer_idx)
        dev = mesh_device
        w = {}

        def g(name):
            return state_dict[name].float()

        # attention projections
        w["wq"] = _linear_w(g("self_attn.q_proj.weight"), dev)
        w["wk"] = _linear_w(g("self_attn.k_proj.weight"), dev)
        w["wv"] = _linear_w(g("self_attn.v_proj.weight"), dev)
        w["wo"] = _linear_w(g("self_attn.o_proj.weight"), dev)
        w["wg"] = _linear_w(g("self_attn.g_proj.weight"), dev)
        # qk norm weights (over head_dim); reshape for rms_norm broadcast
        w["q_norm"] = _as_tt(g("self_attn.q_norm.weight").reshape(1, 1, 1, cfg.head_dim), dev)
        w["k_norm"] = _as_tt(g("self_attn.k_norm.weight").reshape(1, 1, 1, cfg.head_dim), dev)
        w["input_ln"] = _as_tt(g("input_layernorm.weight").reshape(1, 1, 1, cfg.hidden), dev)
        w["post_ln"] = _as_tt(g("post_attention_layernorm.weight").reshape(1, 1, 1, cfg.hidden), dev)

        if cfg.is_moe:
            E, H, I = cfg.num_experts, cfg.hidden, cfg.moe_intermediate
            w["gate_w"] = _linear_w(g("mlp.gate.weight"), dev)  # [H, E]
            bias = g("mlp.experts.e_score_correction_bias").reshape(1, 1, 1, E)
            w["e_bias"] = _as_tt(bias, dev)
            # per-expert weights stacked to sparse_matmul layout [1,E,K,N]
            gate = torch.stack([g(f"mlp.experts.{i}.gate_proj.weight") for i in range(E)])  # [E,I,H]
            up = torch.stack([g(f"mlp.experts.{i}.up_proj.weight") for i in range(E)])  # [E,I,H]
            down = torch.stack([g(f"mlp.experts.{i}.down_proj.weight") for i in range(E)])  # [E,H,I]
            w["exp_gate"] = _as_tt(gate.transpose(1, 2).reshape(1, E, H, I), dev)  # [1,E,H,I]
            w["exp_up"] = _as_tt(up.transpose(1, 2).reshape(1, E, H, I), dev)
            w["exp_down"] = _as_tt(down.transpose(1, 2).reshape(1, E, I, H), dev)  # [1,E,I,H]
            # shared expert (dense SiLU-GLU, intermediate = shared_intermediate)
            w["sh_gate"] = _linear_w(g("mlp.shared_expert.gate_proj.weight"), dev)
            w["sh_up"] = _linear_w(g("mlp.shared_expert.up_proj.weight"), dev)
            w["sh_down"] = _linear_w(g("mlp.shared_expert.down_proj.weight"), dev)
        else:
            w["mlp_gate"] = _linear_w(g("mlp.gate_proj.weight"), dev)
            w["mlp_up"] = _linear_w(g("mlp.up_proj.weight"), dev)
            w["mlp_down"] = _linear_w(g("mlp.down_proj.weight"), dev)

        cos, sin = _hf_rope_tables(hf_config, cfg.attention_type, max_seq_len)
        cos_2d = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        sin_2d = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        return cls(cfg, w, cos_2d, sin_2d, dev)

    # ---- KV cache allocation ---------------------------------------------- #
    def alloc_kv_cache(self, max_users, max_seq_len, block_size=32, dtype=ttnn.bfloat16):
        blocks_per_user = int(math.ceil(max_seq_len / block_size))
        max_num_blocks = blocks_per_user * max_users
        shape = (max_num_blocks, self.cfg.num_kv_heads, block_size, self.cfg.head_dim)
        k = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return {"k": k, "v": v, "block_size": block_size, "blocks_per_user": blocks_per_user}

    def make_page_table(self, num_users, blocks_per_user):
        pt = torch.arange(num_users * blocks_per_user, dtype=torch.int32).reshape(num_users, blocks_per_user)
        return ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    # ---- shared ops -------------------------------------------------------- #
    def _rms(self, x, weight):
        return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.eps, compute_kernel_config=self._compute_kernel)

    def _per_head_norm(self, x, weight):
        # x: [1, heads, S, head_dim] -> flatten so rms reduces over head_dim only
        b, h, s, d = x.shape
        flat = ttnn.reshape(x, (1, 1, b * h * s, d))
        normed = ttnn.rms_norm(flat, weight=weight, epsilon=self.cfg.eps, compute_kernel_config=self._compute_kernel)
        return ttnn.reshape(normed, (b, h, s, d))

    def _apply_rope(self, x, cos, sin):
        """x: [.., head_dim]; cos/sin: [.., rotary_dim] broadcastable. Reproduces HF
        partial rotary with NeoX rotate_half over the first ``rotary_dim`` dims."""
        rd = self.cfg.rotary_dim
        hd = self.cfg.head_dim
        if rd == hd:
            x_rot, x_pass = x, None
        else:
            x_rot = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], rd])
            x_pass = ttnn.slice(x, [0, 0, 0, rd], list(x.shape))
        half = rd // 2
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [x_rot.shape[0], x_rot.shape[1], x_rot.shape[2], half])
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [x_rot.shape[0], x_rot.shape[1], x_rot.shape[2], rd])
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        embed = ttnn.add(ttnn.mul(x_rot, cos), ttnn.mul(rot, sin))
        if x_pass is None:
            return embed
        return ttnn.concat([embed, x_pass], dim=-1)

    def _gate(self, attn, ln, seq_or_batch, layout4d):
        """Apply per-head softplus output gate. attn: heads-concatenated [.., nh*hd]."""
        cfg = self.cfg
        g = ttnn.linear(ln, self.w["wg"], compute_kernel_config=self._compute_kernel)  # [.., nh]
        g = ttnn.softplus(g)
        # reshape attn to per-head and broadcast gate over head_dim
        shp = list(attn.shape)
        attn_h = ttnn.reshape(attn, shp[:-1] + [cfg.num_heads, cfg.head_dim])
        g_h = ttnn.reshape(g, shp[:-1] + [cfg.num_heads, 1])
        gated = ttnn.mul(attn_h, g_h)
        return ttnn.reshape(gated, shp)

    def _dense_mlp(self, x, gate_w, up_w, down_w):
        gate = ttnn.silu(ttnn.linear(x, gate_w, compute_kernel_config=self._compute_kernel))
        up = ttnn.linear(x, up_w, compute_kernel_config=self._compute_kernel)
        return ttnn.linear(ttnn.mul(gate, up), down_w, compute_kernel_config=self._compute_kernel)

    # ---- MoE --------------------------------------------------------------- #
    def _moe(self, ln_flat):
        """ln_flat: [1, 1, T, H] -> [1, 1, T, H]."""
        cfg = self.cfg
        E, H, I, K = cfg.num_experts, cfg.hidden, cfg.moe_intermediate, cfg.top_k
        T = ln_flat.shape[2]
        # router
        logits = ttnn.linear(ln_flat, self.w["gate_w"], compute_kernel_config=self._compute_kernel)  # [1,1,T,E]
        scores = ttnn.sigmoid(logits)
        sel = ttnn.add(scores, self.w["e_bias"])
        _, idx = ttnn.topk(ttnn.typecast(sel, ttnn.bfloat16), k=K, dim=-1, sorted=True)  # [1,1,T,K]
        wsel = ttnn.gather(scores, dim=3, index=idx)
        if cfg.norm_topk_prob:
            wsum = ttnn.sum(wsel, dim=3, keepdim=True)
            wsel = ttnn.div(wsel, wsum)
        if cfg.routed_scaling != 1.0:
            wsel = ttnn.multiply(wsel, cfg.routed_scaling)
        dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=idx, src=wsel)  # [1,1,T,E]
        # experts (union sparsity)
        union = ttnn.sum(dense, dim=2, keepdim=True)  # [1,1,1,E]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        a = ttnn.reshape(ln_flat, (1, 1, T, H))
        gu_pc = _sparse_pc(I, T, H)
        gate_o = ttnn.sparse_matmul(
            a, self.w["exp_gate"], sparsity=sparsity, program_config=gu_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,1,1,E,T,I]
        up_o = ttnn.sparse_matmul(
            a, self.w["exp_up"], sparsity=sparsity, program_config=gu_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate_o = ttnn.reshape(gate_o, (1, E, T, I))
        up_o = ttnn.reshape(up_o, (1, E, T, I))
        glu = ttnn.mul(ttnn.silu(gate_o), up_o)  # [1,E,T,I]
        dn_pc = _sparse_pc(H, T, I)
        down_o = ttnn.sparse_matmul(
            glu,
            self.w["exp_down"],
            sparsity=sparsity,
            is_input_a_sparse=True,
            program_config=dn_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,E,T,H]
        # per-token per-expert weighting then reduce over experts
        w = ttnn.reshape(dense, (1, T, E))
        w = ttnn.permute(w, (0, 2, 1))  # [1,E,T]
        w = ttnn.reshape(w, (1, E, T, 1))
        weighted = ttnn.mul(down_o, w)
        routed = ttnn.sum(weighted, dim=1)  # [1,T,H]
        routed = ttnn.reshape(routed, (1, 1, T, H))
        shared = self._dense_mlp(ln_flat, self.w["sh_gate"], self.w["sh_up"], self.w["sh_down"])
        return ttnn.add(routed, shared)

    # Bound the MoE expert output tensor [1, E, chunk, *] in DRAM by processing the
    # sequence in chunks; the union-sparsity expert matmul materialises E*chunk*I.
    MOE_PREFILL_CHUNK = 256

    def _mlp(self, ln, T):
        cfg = self.cfg
        ln_flat = ttnn.reshape(ln, (1, 1, T, cfg.hidden))
        if not cfg.is_moe:
            return self._dense_mlp(ln_flat, self.w["mlp_gate"], self.w["mlp_up"], self.w["mlp_down"])
        if T <= self.MOE_PREFILL_CHUNK:
            return self._moe(ln_flat)
        outs = []
        for s in range(0, T, self.MOE_PREFILL_CHUNK):
            e = min(s + self.MOE_PREFILL_CHUNK, T)
            chunk = ttnn.slice(ln_flat, [0, 0, s, 0], [1, 1, e, cfg.hidden])
            outs.append(self._moe(chunk))
        return ttnn.concat(outs, dim=2)

    # ---- prefill ----------------------------------------------------------- #
    # Beyond this prompt length the full-sequence Q/K/V activations (e.g. 64*128*seq*2 B
    # for a sliding layer) exceed device DRAM, so long prompts are processed as a
    # sequence-pipelined stream of block-aligned chunks (production prefill pattern):
    # each chunk projects, fills its paged-cache slot, attends (over the paged cache for
    # full layers, a rolling window for sliding), and runs the MLP, keeping peak activation
    # bounded to one chunk. The public API accepts any seq_len up to the advertised context.
    PIPE_CHUNK = 8192

    def _qkv_roped(self, ln, seq, start_pos):
        cfg = self.cfg
        q = ttnn.linear(ln, self.w["wq"], compute_kernel_config=self._compute_kernel)
        k = ttnn.linear(ln, self.w["wk"], compute_kernel_config=self._compute_kernel)
        v = ttnn.linear(ln, self.w["wv"], compute_kernel_config=self._compute_kernel)
        q = ttnn.permute(ttnn.reshape(q, (1, seq, cfg.num_heads, cfg.head_dim)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (1, seq, cfg.num_kv_heads, cfg.head_dim)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (1, seq, cfg.num_kv_heads, cfg.head_dim)), (0, 2, 1, 3))
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        cos = self._rope_prefill(start_pos, seq)
        sin = self._rope_prefill(start_pos, seq, sin=True)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        return q, k, v

    def prefill_forward(
        self,
        x_BSH,
        kv_cache,
        page_table,
        *,
        fill_page_table=None,
        fill_page_table_base_pos=0,
        user_id=0,
        start_pos=0,
        rope_mats=None,
        runtime_offsets=None,
    ):
        if runtime_offsets is not None:
            raise ValueError("runtime-offset prefill is supported only by the P150x2 multichip decoder")
        fill_page_table = page_table if fill_page_table is None else fill_page_table
        seq = x_BSH.shape[-2]
        if seq > self.PIPE_CHUNK:
            return self._prefill_pipelined(
                x_BSH,
                kv_cache,
                page_table,
                fill_page_table,
                user_id,
                start_pos,
                fill_page_table_base_pos=fill_page_table_base_pos,
            )
        cfg = self.cfg
        residual = x_BSH
        ln = self._rms(x_BSH, self.w["input_ln"])  # [1,seq,H]
        q, k, v = self._qkv_roped(ln, seq, start_pos)
        # fill paged cache for decode continuation
        fill_pt = single_shot_fill_page_table(
            fill_page_table,
            start_pos=start_pos,
            seq_len=seq,
            block_size=kv_cache["block_size"],
            fill_page_table_base_pos=fill_page_table_base_pos,
        )
        ttnn.experimental.paged_fill_cache(kv_cache["k"], k, fill_pt, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(kv_cache["v"], v, fill_pt, batch_idx=user_id)
        # attention (chunk Q for lengths beyond the single-shot SDPA op limit)
        attn = self._prefill_attention(q, k, v, kv_cache, page_table, user_id, start_pos, seq)  # [1,nh,seq,hd]
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (1, seq, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln, seq, "prefill")
        o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._compute_kernel)
        h = ttnn.add(residual, o)
        # mlp
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = self._mlp(ln2, seq)
        mlp_out = ttnn.reshape(mlp_out, (1, seq, cfg.hidden))
        return ttnn.add(h, mlp_out)

    def _prefill_pipelined(
        self,
        x_BSH,
        kv_cache,
        page_table,
        fill_page_table,
        user_id,
        start_pos,
        *,
        fill_page_table_base_pos=0,
    ):
        cfg = self.cfg
        seq = x_BSH.shape[-2]
        bs = kv_cache["block_size"]
        CH = (self.PIPE_CHUNK // bs) * bs  # block-aligned chunk
        win = cfg.sliding_window
        k_tail = v_tail = None  # rolling window K/V for sliding layers
        outs = []
        fill_base = int(fill_page_table_base_pos)
        if int(start_pos) < fill_base or (int(start_pos) - fill_base) % bs:
            raise ValueError(
                f"prefill start {start_pos} and fill page-table base {fill_base} are not block aligned"
            )
        for c in range(0, seq, CH):
            ch = min(CH, seq - c)
            gpos = start_pos + c
            xc = ttnn.slice(x_BSH, [0, c, 0], [1, c + ch, cfg.hidden])
            residual = xc
            ln = self._rms(xc, self.w["input_ln"])
            q, k, v = self._qkv_roped(ln, ch, gpos)
            # fill this chunk's cache slot: page-table columns for blocks [gpos/bs : (gpos+ch)/bs]
            col0 = (gpos - fill_base) // bs
            ncol = (ch + bs - 1) // bs
            chunk_pt = ttnn.slice(
                fill_page_table,
                [0, col0],
                [fill_page_table.shape[0], col0 + ncol],
            )
            ttnn.experimental.paged_fill_cache(kv_cache["k"], k, chunk_pt, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(kv_cache["v"], v, chunk_pt, batch_idx=user_id)
            # attention for this chunk's queries
            if not cfg.is_sliding:
                user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    chunk_start_idx=gpos,
                    compute_kernel_config=self._sdpa_compute,
                )
            else:
                if k_tail is None:
                    k_loc, v_loc, pad = k, v, 0
                else:
                    k_loc = ttnn.concat([k_tail, k], dim=2)
                    v_loc = ttnn.concat([v_tail, v], dim=2)
                    pad = k_tail.shape[2]
                    # front-pad queries with zeros so q/k share a start position for the
                    # causal+window mask; the padded rows' outputs are sliced off below.
                    qpad = ttnn.zeros(
                        [1, cfg.num_heads, pad, cfg.head_dim],
                        dtype=q.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                    )
                    q = ttnn.concat([qpad, q], dim=2)
                out_loc = ttnn.transformer.scaled_dot_product_attention(
                    q,
                    k_loc,
                    v_loc,
                    is_causal=True,
                    sliding_window_size=win,
                    scale=cfg.scaling,
                    compute_kernel_config=self._sdpa_compute,
                )
                attn = ttnn.slice(out_loc, [0, 0, pad, 0], [1, cfg.num_heads, pad + ch, cfg.head_dim])
                tail = min(win - 1, ch)
                k_tail = ttnn.slice(k, [0, 0, ch - tail, 0], [1, cfg.num_kv_heads, ch, cfg.head_dim])
                v_tail = ttnn.slice(v, [0, 0, ch - tail, 0], [1, cfg.num_kv_heads, ch, cfg.head_dim])
            attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (1, ch, cfg.num_heads * cfg.head_dim))
            attn = self._gate(attn, ln, ch, "prefill")
            o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._compute_kernel)
            h = ttnn.add(residual, o)
            ln2 = self._rms(h, self.w["post_ln"])
            mlp_out = ttnn.reshape(self._mlp(ln2, ch), (1, ch, cfg.hidden))
            outs.append(ttnn.add(h, mlp_out))
        return ttnn.concat(outs, dim=1)

    # Single-shot SDPA is correct up to this Q length; beyond it we chunk Q. The op is
    # documented to return wrong results for Q seq_len > 32768, so keep a safety margin.
    PREFILL_SDPA_CHUNK = 8192

    def _prefill_attention(self, q, k, v, kv_cache, page_table, user_id, start_pos, seq):
        """Q/K/V: [1, heads, seq, hd] (post rope). Returns [1, nh, seq, hd].

        For seq <= PREFILL_SDPA_CHUNK use one SDPA call. For longer prompts chunk Q:
          * full attention  -> ttnn chunked SDPA over the paged cache (causal, any length);
          * sliding window   -> overlapping local slices (each Q chunk plus its window of
            preceding keys, start-aligned so the causal+window mask is correct), keeping
            only the chunk's own outputs. Both keep every SDPA call within the op's limit.
        """
        cfg = self.cfg
        base = {"scale": cfg.scaling, "compute_kernel_config": self._sdpa_compute}
        if seq <= self.PREFILL_SDPA_CHUNK:
            kw = {"is_causal": True, "program_config": self._sdpa_pc, **base}
            if cfg.is_sliding:
                kw["sliding_window_size"] = cfg.sliding_window
            return ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)

        CH = self.PREFILL_SDPA_CHUNK
        outs = []
        if not cfg.is_sliding:
            user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            for c in range(start_pos, start_pos + seq, CH):
                ch = min(CH, start_pos + seq - c)
                q_c = ttnn.slice(q, [0, 0, c - start_pos, 0], [1, cfg.num_heads, c - start_pos + ch, cfg.head_dim])
                out_c = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_c,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    chunk_start_idx=c,
                    compute_kernel_config=self._sdpa_compute,
                )
                outs.append(out_c)
            return ttnn.concat(outs, dim=2)
        # sliding: overlapping local slices
        win = cfg.sliding_window
        for c in range(0, seq, CH):
            ch = min(CH, seq - c)
            s0 = max(0, c - (win - 1))
            q_sl = ttnn.slice(q, [0, 0, s0, 0], [1, cfg.num_heads, c + ch, cfg.head_dim])
            k_sl = ttnn.slice(k, [0, 0, s0, 0], [1, cfg.num_kv_heads, c + ch, cfg.head_dim])
            v_sl = ttnn.slice(v, [0, 0, s0, 0], [1, cfg.num_kv_heads, c + ch, cfg.head_dim])
            out_sl = ttnn.transformer.scaled_dot_product_attention(
                q_sl, k_sl, v_sl, is_causal=True, sliding_window_size=win, **base
            )
            outs.append(ttnn.slice(out_sl, [0, 0, c - s0, 0], [1, cfg.num_heads, ch + (c - s0), cfg.head_dim]))
        return ttnn.concat(outs, dim=2)

    def _rope_prefill(self, start_pos, seq, sin=False):
        table = self.sin_2d if sin else self.cos_2d
        rd = self.cfg.rotary_dim
        sliced = ttnn.slice(table, [start_pos, 0], [start_pos + seq, rd])  # [seq, rd] ROW_MAJOR
        t = ttnn.to_layout(ttnn.reshape(sliced, (1, 1, seq, rd)), ttnn.TILE_LAYOUT)
        return t

    # ---- decode ------------------------------------------------------------ #
    def decode_forward(self, x_1BH, cur_pos, rope_idx, page_table, kv_cache):
        cfg = self.cfg
        B = x_1BH.shape[-2]
        residual = x_1BH
        ln = self._rms(x_1BH, self.w["input_ln"])  # [1,1,B,H]
        q = ttnn.linear(ln, self.w["wq"], compute_kernel_config=self._compute_kernel)
        k = ttnn.linear(ln, self.w["wk"], compute_kernel_config=self._compute_kernel)
        v = ttnn.linear(ln, self.w["wv"], compute_kernel_config=self._compute_kernel)
        q = ttnn.reshape(q, (1, B, cfg.num_heads, cfg.head_dim))
        k = ttnn.reshape(k, (1, B, cfg.num_kv_heads, cfg.head_dim))
        v = ttnn.reshape(v, (1, B, cfg.num_kv_heads, cfg.head_dim))
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        # rope: gather cos/sin per position [B, rd] -> [1,B,1,rd]
        cos = self._rope_decode(rope_idx, B)
        sin = self._rope_decode(rope_idx, B, sin=True)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        # paged_update_cache requires height-sharded K/V (one batch element per core)
        k_sh = self._shard_kv(k, B)
        v_sh = self._shard_kv(v, B)
        ttnn.experimental.paged_update_cache(kv_cache["k"], k_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache["v"], v_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        # NOTE: on this ttnn build the decode SDPA is correct with the default program
        # config but breaks with an explicit full-grid one; pass only the fp32 compute
        # config (improves long-context accumulation) and let the op pick its grid.
        sdpa_kwargs = {
            "cur_pos_tensor": cur_pos,
            "page_table_tensor": page_table,
            "scale": cfg.scaling,
            "compute_kernel_config": self._sdpa_compute,
        }
        if cfg.is_sliding:
            sdpa_kwargs["sliding_window_size"] = cfg.sliding_window
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, kv_cache["k"], kv_cache["v"], **sdpa_kwargs
        )  # [1,B,nh,hd]
        attn = ttnn.reshape(attn, (1, 1, B, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln, B, "decode")
        o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._compute_kernel)
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = self._mlp(ln2, B)  # [1,1,B,H]
        return ttnn.add(h, mlp_out)

    def _shard_kv(self, kv, B):
        """Height-shard [1, B, nkv, head_dim] onto B cores for paged_update_cache."""
        nkv = self.cfg.num_kv_heads
        nkv32 = ((nkv + TILE - 1) // TILE) * TILE
        row = 8
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord((B - 1) % row, (B - 1) // row))}
        )
        mem = ttnn.create_sharded_memory_config(
            shape=(nkv32, self.cfg.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.to_memory_config(kv, mem)

    def _rope_decode(self, rope_idx, B, sin=False):
        table = self.sin_2d if sin else self.cos_2d
        rd = self.cfg.rotary_dim
        gathered = ttnn.embedding(rope_idx, table, layout=ttnn.TILE_LAYOUT)  # [1,B,rd] or [B,rd]
        g = ttnn.reshape(gathered, (1, B, 1, rd))
        return g
