# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo decoder layer: RMSNorm -> attention -> residual -> RMSNorm -> (dense MLP | MoE) -> residual.

The residual stream is replicated on every chip of the row (each sub-block ends in an all-reduce).
"""

import os

import ttnn

from .attention import TtMiMoAttention
from .common import TILE, DramShardedMatmul, as_weight, cache_name, compute_config
from .moe import TtMiMoMoE


class TtRMSNorm:
    def __init__(self, plan, weight, eps, cache_file):
        w = None
        if weight is not None:
            w = weight.float().reshape(1, 1, -1, TILE).bfloat16()
        self.weight = as_weight(w, plan, plan.replicate(), ttnn.bfloat16, cache_file, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.eps = eps
        self.config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)
        # Decode (one tile row): an interleaved rms_norm runs on a single core (67 us); width-shard the
        # hidden dim over 8 x 4 cores instead.
        grid = ttnn.CoreGrid(y=4, x=8)
        self.decode_grid = grid

    def _decode_configs(self, hidden):
        cores = self.decode_grid.x * self.decode_grid.y
        shard_w = hidden // cores
        mem = ttnn.create_sharded_memory_config(
            shape=(TILE, shard_w),
            core_grid=self.decode_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(self.decode_grid.x, self.decode_grid.y),
            subblock_w=shard_w // TILE,
            block_h=1,
            block_w=shard_w // TILE,
            inplace=False,
        )
        return mem, program

    def torch_weight(self):
        """The weight as a torch [hidden] tensor (read back from chip 0; it is replicated)."""
        return ttnn.to_torch(ttnn.get_device_tensors(self.weight)[0]).float().reshape(-1)

    def __call__(self, x, memory_config=None):
        """``memory_config``: where the decode (one tile row) output goes; DRAM interleaved by default."""
        if x.shape[-2] == TILE:
            mem, program = self._decode_configs(x.shape[-1])
            x_sharded = ttnn.to_memory_config(x, mem)
            y = ttnn.rms_norm(
                x_sharded,
                epsilon=self.eps,
                weight=self.weight,
                program_config=program,
                memory_config=mem,
                compute_kernel_config=self.config,
            )
            x_sharded.deallocate(True)
            out = ttnn.to_memory_config(y, memory_config or ttnn.DRAM_MEMORY_CONFIG)
            y.deallocate(True)
            return out
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.config,
        )


class TtMiMoDenseMLP:
    """SwiGLU MLP (layer 0): gate/up column-parallel, down row-parallel, all-reduce."""

    def __init__(self, plan, layer_idx, weights=None, cache_dir=None, weight_dtype=ttnn.bfloat8_b):
        self.plan = plan
        prefix = f"layer{layer_idx}_mlp_tp{plan.tp}"
        g = u = d = None
        if weights is not None:
            g = weights["gate_proj"].T.contiguous()
            u = weights["up_proj"].T.contiguous()
            d = weights["down_proj"].T.contiguous()
        w_gate = as_weight(g, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, "w_gate"))
        w_up = as_weight(u, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, "w_up"))
        w_down = as_weight(d, plan, plan.shard(-2), weight_dtype, cache_name(cache_dir, prefix, "w_down"))
        self.config = compute_config(ttnn.MathFidelity.HiFi2)
        # Decode reads each weight once per step: DRAM-sharded matmuls (one reader core per bank) run at ~2x the
        # bandwidth of the interleaved linears here (gate / up 74 us -> ~34 us each). The weights are resharded once
        # on device; prefill uses interleaved copies made per call.
        hidden, inter = w_gate.shape[-2], w_gate.shape[-1]
        self.up_mm = DramShardedMatmul(plan, hidden, inter, compute_kernel_config=self.config)
        self.down_mm = DramShardedMatmul(plan, inter, hidden, compute_kernel_config=self.config)
        assert self.up_mm.n_pad == inter and self.down_mm.n_pad == hidden
        self.w_gate = ttnn.to_memory_config(w_gate, self.up_mm.weight_memory_config)
        self.w_up = ttnn.to_memory_config(w_up, self.up_mm.weight_memory_config)
        self.w_down = ttnn.to_memory_config(w_down, self.down_mm.weight_memory_config)
        for w in (w_gate, w_up, w_down):
            w.deallocate(True)

    def forward(self, x, token_mask=None):
        if x.shape[-2] == TILE:
            gate = self.up_mm(x, self.w_gate)
            up = self.up_mm(x, self.w_up)
            act = ttnn.mul(
                gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            gate.deallocate(True)
            up.deallocate(True)
            out = self.down_mm(act, self.w_down)
            act.deallocate(True)
            return self.plan.all_reduce(out)
        kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.config)
        w_gate, w_up, w_down = (
            ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG) for w in (self.w_gate, self.w_up, self.w_down)
        )
        gate = ttnn.linear(x, w_gate, **kw)
        up = ttnn.linear(x, w_up, **kw)
        act = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(act, w_down, **kw)
        act.deallocate(True)
        for w in (w_gate, w_up, w_down):
            w.deallocate(True)
        return self.plan.all_reduce(out)


class TtMiMoDecoderLayer:
    def __init__(
        self,
        plan,
        cfg,
        layer_idx,
        weights,
        rope_setups,
        cache_dir=None,
        attn_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
    ):
        """``weights``: dict from reference.model.load_layer_weights, or None to load from a warm cache.
        ``rope_setups``: {"swa": MiMoRotarySetup, "full": MiMoRotarySetup}."""
        self.layer_idx = layer_idx
        self.is_swa = cfg.is_swa(layer_idx)
        self.is_moe = cfg.is_moe(layer_idx)
        w = weights or {}
        prefix = f"layer{layer_idx}"
        eps = cfg.layernorm_epsilon
        self.input_norm = TtRMSNorm(plan, w.get("input_layernorm"), eps, cache_name(cache_dir, prefix, "input_norm"))
        self.post_norm = TtRMSNorm(
            plan, w.get("post_attention_layernorm"), eps, cache_name(cache_dir, prefix, "post_attn_norm")
        )
        self.rope = rope_setups["swa" if self.is_swa else "full"]
        self.attn = TtMiMoAttention(
            plan, cfg, layer_idx, w.get("attn"), self.rope, cache_dir=cache_dir, weight_dtype=attn_dtype
        )
        if self.is_moe:
            self.mlp = TtMiMoMoE(
                plan, cfg, layer_idx, w.get("router"), w.get("experts"), cache_dir=cache_dir, expert_dtype=expert_dtype
            )
        else:
            self.mlp = TtMiMoDenseMLP(plan, layer_idx, w.get("mlp"), cache_dir=cache_dir, weight_dtype=mlp_dtype)
        self.fused_norm = None

    def can_fuse_norms(self):
        return self._compact_decode() and self.mlp.reduce_expand.can_norm

    def set_fused_norms(self, plan, next_norm, buffers):
        """Compact decode with both block reductions also computing the following RMSNorm: the attention block's
        reduction writes post_norm(x2) into ``buffers[1]`` and the MoE block's writes ``next_norm`` (the next layer's
        input norm, or the final norm) of the layer output into ``buffers[0]``; forward_decode(h=..., return_norm=True)
        passes the latter on. ``buffers``: persistent zero [1, 1, 32, hidden] bf16 DRAM tensors shared by all layers."""
        red = self.mlp.reduce_expand
        # Both block reductions also all-gather the compact rows over the fabric (MIMO_FABRIC_GATHER=0: all_gather).
        fabric = os.environ.get("MIMO_FABRIC_GATHER", "1") == "1" and tuple(plan.mesh_device.shape) == (1, plan.tp)
        self.attn.fabric_gather = self.mlp.fabric_gather = fabric
        self.fused_norm = {
            "post": (red.compact_gamma(plan, self.post_norm.torch_weight()), buffers[1], self.post_norm.eps),
            "next": (red.compact_gamma(plan, next_norm.torch_weight()), buffers[0], next_norm.eps),
        }

    def _ffn(self, x, token_mask, decode=False):
        h = self.post_norm(x)
        if decode and self.is_moe and self.mlp.decode_users is not None:
            out = self.mlp.forward_decode(h, token_mask)
        else:
            out = self.mlp.forward(h, token_mask)
        h.deallocate(True)
        y = ttnn.add(x, out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return y

    def _page_table(self, page_table):
        """Accepts one table or {is_swa: table} (the SWA and the global layers have their own tables, kv.py)."""
        return page_table[self.is_swa] if isinstance(page_table, dict) else page_table

    def forward_prefill(
        self, x, user_id, page_table, kv_cache, token_mask=None, real_len=None, start_pos=0, window_blocks=None
    ):
        """One prefill chunk at ``start_pos``; ``window_blocks``: see TtMiMoAttention.forward_prefill (SWA only)."""
        h = self.input_norm(x)
        attn = self.attn.forward_prefill(
            h,
            user_id,
            self._page_table(page_table),
            kv_cache,
            start_pos=start_pos,
            real_len=real_len,
            window_blocks=window_blocks if self.is_swa else None,
        )
        h.deallocate(True)
        x2 = ttnn.add(x, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        y = self._ffn(x2, token_mask)
        x2.deallocate(True)
        return y

    def attention_decode(self, x, rot_mats, cur_pos, page_table, kv_cache):
        """Decode attention block: input norm, attention (with its row all-reduce), residual add."""
        # The normed row only feeds the qkv matmul's L1 reshard: keep it in L1.
        h = self.input_norm(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = self.attn.forward_decode(h, rot_mats[self.is_swa], cur_pos, self._page_table(page_table), kv_cache)
        h.deallocate(True)
        x2 = ttnn.add(x, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        return x2

    def _compact_decode(self):
        """The MoE streaming decode serves users in rows 0-3 only: both block reductions then move only those rows."""
        return self.is_moe and self.mlp.gate_up_stream is not None

    def forward_decode(self, x, rot_mats, cur_pos, page_table, kv_cache, token_mask=None, h=None, return_norm=False):
        """``h``: input_norm(x) when the previous layer's reduction already computed it (see set_fused_norms).
        ``return_norm``: return (y, the next norm of y or None)."""
        if not self._compact_decode():
            x2 = self.attention_decode(x, rot_mats, cur_pos, page_table, kv_cache)
            y = self._ffn(x2, token_mask, decode=True)
            x2.deallocate(True)
            return (y, None) if return_norm else y
        # Each block's row all-reduce is a compact all-gather; ReduceExpand sums the chips and adds the residual
        # (with fused norms, it also computes the following RMSNorm).
        fused = self.fused_norm
        own_h = h is None
        if own_h:
            h = self.input_norm(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        gathered = self.attn.forward_decode(
            h, rot_mats[self.is_swa], cur_pos, self._page_table(page_table), kv_cache, compact=True
        )
        if own_h:
            h.deallocate(True)
        if fused is not None and self.attn.fabric_gather:
            x2, h2 = self.mlp.reduce_residual(None, x, norm=fused["post"], compact=gathered)
        elif fused is not None:
            x2, h2 = self.mlp.reduce_residual(gathered, x, norm=fused["post"])
        else:
            x2 = self.mlp.reduce_residual(gathered, x)
        if gathered is not self.attn.compact_buf and hasattr(gathered, "deallocate"):
            gathered.deallocate(True)
        if fused is None:
            h2 = self.post_norm(x2)
        nxt = fused["next"] if fused is not None and return_norm else None
        out = self.mlp.forward_decode(h2, token_mask, residual=x2, norm=nxt)
        if fused is None:
            h2.deallocate(True)
        x2.deallocate(True)
        y, h_next = out if nxt is not None else (out, None)
        return (y, h_next) if return_norm else y
