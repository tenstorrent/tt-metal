# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN Gemma building blocks for the pi0.5 action expert (streamed-denoise port).

VENDORED from ``tt_symbiote.models.pi05.modeling_pi05_gemma`` into the standalone port's
native TTNNModule lifecycle. Relative to source:
  * VLM-prefill self-attention store (``init_vlm_prefix_store`` / ``forward_vlm`` /
    ``_vlm_prefix_*``) is TRIMMED (denoise uses only the static / concat cross-attention KV).
  * All ``PI05_VLM_*`` / ``LADDER_*`` / ``KV_LADDER_*`` env branches are DROPPED (clean
    deterministic surface; weights stay bf8_b, outputs bf16 where the source default emits bf16).
  * The plain ``TTNNPi05GemmaBlock`` (VLM block) is DROPPED -- denoise uses only the AdaRMS block.
  * Config import is rewired to the target ``common/configs.py`` (fields verified compatible).

ZERO tt_symbiote imports.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import ttnn

from .._module import DeviceArch, StatefulTTNNModule, StatelessTTNNModule, run_on_devices
from .._trace import trace_enabled
from .bs import matmul_pcfg, sdpa_program_config, sharded_rms_norm
from .common import get_sdpa_compute_kernel_config

from models.experimental.pi0_5.common.configs import GemmaConfig

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

__all__ = [
    "TTNNPi05GemmaMLP",
    "TTNNPi05GemmaAttention",
    "TTNNPi05AdaRMSGemmaBlock",
    "_linear_weight_to_tt",
    "_norm_weight_to_tt",
    "_rms_norm",
]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


# ---------------------------------------------------------------------------
# Weight upload helpers (host-side; allowed to use torch).
# ---------------------------------------------------------------------------
def _linear_weight_to_tt(w: torch.Tensor, dtype: "ttnn.DataType" = ttnn.bfloat8_b) -> ttnn.Tensor:
    """Transpose a torch ``[out, in]`` linear weight to ttnn ``[in, out]`` host tensor."""
    return ttnn.from_torch(w.t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _norm_weight_to_tt(w: torch.Tensor) -> ttnn.Tensor:
    """Gemma RMSNorm weight with the ``+1.0`` offset folded in, shape ``[1, dim]``."""
    folded = (w + 1.0).reshape(1, w.shape[0]).contiguous()
    return ttnn.from_torch(folded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _rms_norm(x: ttnn.Tensor, weight: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Plain RMSNorm using a pre-offset weight (``weight`` already holds ``w+1``)."""
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=_L1)
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=_L1)


# ---------------------------------------------------------------------------
# GeGLU MLP
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05GemmaMLP(StatelessTTNNModule):
    """Gemma GeGLU MLP: ``down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))``."""

    @classmethod
    def from_torch(cls, mlp, config: Optional[GemmaConfig] = None, weight_dtype=ttnn.bfloat8_b) -> "TTNNPi05GemmaMLP":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = mlp
        new._config = config
        new._wdtype = weight_dtype
        new._gate_w = mlp.gate_proj
        new._up_w = mlp.up_proj
        new._down_w = mlp.down_proj
        return new

    def preprocess_weights_impl(self):
        self.tt_gate = _linear_weight_to_tt(self._gate_w, dtype=self._wdtype)
        self.tt_up = _linear_weight_to_tt(self._up_w, dtype=self._wdtype)
        self.tt_down = _linear_weight_to_tt(self._down_w, dtype=self._wdtype)

    def move_weights_to_device_impl(self):
        self.tt_gate = ttnn.to_device(self.tt_gate, self.device, memory_config=_DRAM)
        self.tt_up = ttnn.to_device(self.tt_up, self.device, memory_config=_DRAM)
        self.tt_down = ttnn.to_device(self.tt_down, self.device, memory_config=_DRAM)
        _g = self.device.compute_with_storage_grid_size()
        self._row_cg = ttnn.CoreGrid(y=1, x=_g.x)
        self._full_cg = ttnn.CoreGrid(y=_g.y, x=_g.x)

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        seq = x.shape[-2]
        # EXPERT path (small-M, seq<=96): tuned 2D block-shard program configs on the 8x8 grid
        # with FUSED gelu. Falls back to the legacy path when matmul_pcfg declines the shape.
        if seq <= 96:
            mt = seq // 32
            k_gu, n_gu = self.tt_gate.shape[-2] // 32, self.tt_gate.shape[-1] // 32
            k_dn, n_dn = self.tt_down.shape[-2] // 32, self.tt_down.shape[-1] // 32
            gate_pc = matmul_pcfg(mt, k_gu, n_gu, 8, 8, activation=(ttnn.UnaryOpType.GELU, True))
            up_pc = matmul_pcfg(mt, k_gu, n_gu, 8, 8)
            down_pc = matmul_pcfg(mt, k_dn, n_dn, 8, 8)
            if gate_pc is not None and up_pc is not None and down_pc is not None:
                gate = ttnn.linear(x, self.tt_gate, memory_config=_L1, program_config=gate_pc)
                up = ttnn.linear(x, self.tt_up, memory_config=_L1, program_config=up_pc)
                hidden = ttnn.multiply(gate, up, memory_config=_L1)
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
                out = ttnn.linear(hidden, self.tt_down, memory_config=_L1, program_config=down_pc)
                ttnn.deallocate(hidden)
                return out
        # gate/up: large-M VLM wins on the FULL 2D grid; small-M expert keeps default.
        gu_cg = self._full_cg if seq > 96 else None
        gate = ttnn.linear(x, self.tt_gate, memory_config=_L1, core_grid=gu_cg)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=_L1)
        up = ttnn.linear(x, self.tt_up, memory_config=_L1, core_grid=gu_cg)
        hidden = ttnn.multiply(gate, up, memory_config=_L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        down_cg = self._row_cg if seq <= 96 else self._full_cg
        out = ttnn.linear(hidden, self.tt_down, memory_config=_L1, core_grid=down_cg)
        ttnn.deallocate(hidden)
        return out


# ---------------------------------------------------------------------------
# Multi-Query Attention (8 Q heads, 1 KV head, head_dim 256)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05GemmaAttention(StatefulTTNNModule):
    """Gemma MQA with fused QKV, meta-format RoPE, SDPA and KV cache (static + concat)."""

    @classmethod
    def from_torch(cls, attn, config: GemmaConfig) -> "TTNNPi05GemmaAttention":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = attn
        new._config = config
        new._q_w = attn.q_proj
        new._k_w = attn.k_proj
        new._v_w = attn.v_proj
        new._o_w = attn.o_proj
        new.num_heads = config.num_heads
        new.num_kv_heads = config.num_kv_heads
        new.head_dim = config.head_dim
        new.scale = 1.0 / math.sqrt(config.head_dim)
        new._eps = config.rms_norm_eps
        # Static KV buffer state (trace-safe cross-attention path).
        new._static_k = None
        new._static_v = None
        new._static_prefix_len = 0
        return new

    def preprocess_weights_impl(self):
        wq = self._q_w.t().contiguous()
        wk = self._k_w.t().contiguous()
        wv = self._v_w.t().contiguous()
        wqkv = torch.cat([wq, wk, wv], dim=-1)  # [in, (Q+K+V)_out]
        self.tt_wqkv = ttnn.from_torch(wqkv, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.tt_o = _linear_weight_to_tt(self._o_w)

    def move_weights_to_device_impl(self):
        self.tt_wqkv = ttnn.to_device(self.tt_wqkv, self.device, memory_config=_DRAM)
        self.tt_o = ttnn.to_device(self.tt_o, self.device, memory_config=_DRAM)
        self._row_cg = ttnn.CoreGrid(y=1, x=self.device.compute_with_storage_grid_size().x)

    # ------------------------------------------------------------------ static KV
    def init_static_kv(self, prefix_len: int, suffix_len: int) -> None:
        """Pre-allocate the static cross-attention K/V buffers (once per inference, outside trace)."""
        if prefix_len % 32 != 0:
            raise ValueError(f"static KV requires a tile-aligned prefix_len; got {prefix_len}")
        total = prefix_len + suffix_len
        if self._static_k is not None and self._static_k.shape[2] == total and self._static_prefix_len == prefix_len:
            return
        if self._static_k is not None:
            ttnn.deallocate(self._static_k)
            ttnn.deallocate(self._static_v)
        zeros = torch.zeros(1, self.num_kv_heads, total, self.head_dim)
        # BAKED bf8_b (PI05_VLM_KV_BF16 dropped); bf16 PCC-recovery flip documented in PORT_NOTES.
        _static_kv_dtype = ttnn.bfloat8_b
        self._static_k = ttnn.from_torch(
            zeros, dtype=_static_kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM
        )
        self._static_v = ttnn.from_torch(
            zeros, dtype=_static_kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM
        )
        self._static_prefix_len = prefix_len

    def fill_static_prefix(self, past_k: ttnn.Tensor, past_v: ttnn.Tensor) -> None:
        """Copy the (already RoPE'd) prefix K/V into the static buffer prefix region (in-place)."""
        ttnn.fill_cache(self._static_k, past_k, 0, update_idx=0)
        ttnn.fill_cache(self._static_v, past_v, 0, update_idx=0)

    def clear_static_kv(self) -> None:
        if self._static_k is not None:
            ttnn.deallocate(self._static_k)
            ttnn.deallocate(self._static_v)
        self._static_k = None
        self._static_v = None
        self._static_prefix_len = 0

    def reset_trace_state(self) -> None:
        # STATEFUL: forward writes self._static_k/_v via ttnn.fill_cache at a FIXED update_idx
        # (in-place OVERWRITE, NOT an advancing append) -> the warm-up+capture double-run is
        # idempotent and there is nothing to revert. Justified no-op.
        return None

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        if len(hidden_states.shape) == 3:
            b, s, _ = hidden_states.shape
            hidden_states = ttnn.reshape(hidden_states, (b, 1, s, hidden_states.shape[-1]))
        else:
            b, _, s, _ = hidden_states.shape

        qkv_cg = self._row_cg if s <= 96 else None
        _g = self.device.compute_with_storage_grid_size()
        _qkv_pc = matmul_pcfg(
            s // 32, self.tt_wqkv.shape[-2] // 32, self.tt_wqkv.shape[-1] // 32, _g.x, _g.y, in0_block_w=8
        )
        if _qkv_pc is not None:
            qkv = ttnn.linear(
                hidden_states, self.tt_wqkv, dtype=ttnn.bfloat8_b, memory_config=_L1, program_config=_qkv_pc
            )
        else:
            qkv = ttnn.linear(hidden_states, self.tt_wqkv, dtype=ttnn.bfloat8_b, memory_config=_L1, core_grid=qkv_cg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=_L1,
        )
        ttnn.deallocate(qkv)

        q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=_L1)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=_L1)

        if self._static_k is not None:
            ttnn.fill_cache(self._static_k, k, 0, update_idx=self._static_prefix_len)
            ttnn.fill_cache(self._static_v, v, 0, update_idx=self._static_prefix_len)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k = self._static_k
            v = self._static_v
            new_cache = None
        else:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = ttnn.concat([past_k, k], dim=2, memory_config=_L1)
                v = ttnn.concat([past_v, v], dim=2, memory_config=_L1)
            new_cache = (k, v) if use_cache else None

        _sdpa_kwargs = {}
        if s <= 96:
            _sdpa_kwargs["memory_config"] = _L1
            _spc = sdpa_program_config(q.shape[-2], k.shape[-2], min(_g.x, 8), min(_g.y, 2))
            if _spc is not None:
                _sdpa_kwargs["program_config"] = _spc
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=get_sdpa_compute_kernel_config(),
            **_sdpa_kwargs,
        )
        ttnn.deallocate(q)

        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=_L1)
        _o_pc = matmul_pcfg(s // 32, self.tt_o.shape[-2] // 32, self.tt_o.shape[-1] // 32, _g.x, _g.y, in0_block_w=8)
        if _o_pc is not None:
            out = ttnn.linear(attn_out, self.tt_o, dtype=ttnn.bfloat16, memory_config=_L1, program_config=_o_pc)
        else:
            out = ttnn.linear(attn_out, self.tt_o, dtype=ttnn.bfloat16, memory_config=_L1, core_grid=qkv_cg)
        ttnn.deallocate(attn_out)
        if len(out.shape) == 4:
            out = ttnn.reshape(out, (b, s, out.shape[-1]))
        return out, new_cache


# ---------------------------------------------------------------------------
# AdaRMS Gemma block (action expert)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05AdaRMSGemmaBlock(StatefulTTNNModule):
    """Action-expert block with adaptive RMSNorm + gated residuals (parent of the denoise expert)."""

    @classmethod
    def from_torch(cls, block, config: GemmaConfig) -> "TTNNPi05AdaRMSGemmaBlock":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = block
        new._config = config
        new._eps = config.rms_norm_eps
        new._width = config.width
        new._pre_attn_mod_w = block.pre_attn_mod_weight
        new._pre_attn_mod_b = getattr(block, "pre_attn_mod_bias", None)
        new._pre_ffw_mod_w = block.pre_ffw_mod_weight
        new._pre_ffw_mod_b = getattr(block, "pre_ffw_mod_bias", None)
        new.attention = TTNNPi05GemmaAttention.from_torch(block.attention, config)
        new.mlp = TTNNPi05GemmaMLP.from_torch(block.mlp, config)
        return new

    def reset_trace_state(self) -> None:
        # No OWN trace state; the stateful descendant (self.attention) is reset by the
        # framework's trace tree-reset. Stateful only because it owns that descendant.
        return None

    def preprocess_weights_impl(self):
        self.tt_pre_attn_mod_w = _linear_weight_to_tt(self._pre_attn_mod_w, dtype=ttnn.bfloat16)
        self.tt_pre_ffw_mod_w = _linear_weight_to_tt(self._pre_ffw_mod_w, dtype=ttnn.bfloat16)
        self.tt_pre_attn_mod_b = (
            ttnn.from_torch(
                self._pre_attn_mod_b.reshape(1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            if self._pre_attn_mod_b is not None
            else None
        )
        self.tt_pre_ffw_mod_b = (
            ttnn.from_torch(
                self._pre_ffw_mod_b.reshape(1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            if self._pre_ffw_mod_b is not None
            else None
        )
        self.attention.preprocess_weights()
        self.mlp.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.tt_pre_attn_mod_w = ttnn.to_device(self.tt_pre_attn_mod_w, self.device, memory_config=_DRAM)
        self.tt_pre_ffw_mod_w = ttnn.to_device(self.tt_pre_ffw_mod_w, self.device, memory_config=_DRAM)
        if self.tt_pre_attn_mod_b is not None:
            self.tt_pre_attn_mod_b = ttnn.to_device(self.tt_pre_attn_mod_b, self.device, memory_config=_DRAM)
        if self.tt_pre_ffw_mod_b is not None:
            self.tt_pre_ffw_mod_b = ttnn.to_device(self.tt_pre_ffw_mod_b, self.device, memory_config=_DRAM)
        self.attention.move_weights_to_device()
        self.mlp.move_weights_to_device()

    # Static-KV delegation to the attention sub-module (trace-safe cross-attn).
    def init_static_kv(self, prefix_len: int, suffix_len: int) -> None:
        self.attention.init_static_kv(prefix_len, suffix_len)

    def fill_static_prefix(self, past_k: ttnn.Tensor, past_v: ttnn.Tensor) -> None:
        self.attention.fill_static_prefix(past_k, past_v)

    def clear_static_kv(self) -> None:
        self.attention.clear_static_kv()

    def _compute_modulation(
        self, cond: ttnn.Tensor, mod_w: ttnn.Tensor, mod_b: Optional[ttnn.Tensor]
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """cond [B, width] -> (scale1, shift, gate) each [B, 1, width], scale1 = 1+scale."""
        mod = ttnn.linear(cond, mod_w, bias=mod_b, memory_config=_L1)
        w = self._width
        b = mod.shape[0]
        scale = ttnn.reshape(ttnn.slice(mod, [0, 0], [b, w]), (b, 1, w))
        shift = ttnn.reshape(ttnn.slice(mod, [0, w], [b, 2 * w]), (b, 1, w))
        gate = ttnn.reshape(ttnn.slice(mod, [0, 2 * w], [b, 3 * w]), (b, 1, w))
        ttnn.deallocate(mod)
        scale1 = ttnn.add(scale, 1.0, memory_config=_L1)
        ttnn.deallocate(scale)
        return scale1, shift, gate

    def precompute_mods(self, adarms_cond: ttnn.Tensor) -> Tuple[ttnn.Tensor, ...]:
        """Precompute the 6 modulation tensors for a fixed conditioning signal."""
        return (
            *self._compute_modulation(adarms_cond, self.tt_pre_attn_mod_w, self.tt_pre_attn_mod_b),
            *self._compute_modulation(adarms_cond, self.tt_pre_ffw_mod_w, self.tt_pre_ffw_mod_b),
        )

    def _apply_ada(
        self,
        x: ttnn.Tensor,
        scale1: ttnn.Tensor,
        shift: ttnn.Tensor,
        eps: float,
        *,
        out_block_sharded: bool = False,
    ) -> ttnn.Tensor:
        # FUSED adaRMS: fold the modulation into the norm kernel (weight=(1+scale), bias=shift).
        # out_block_sharded: emit the block-sharded norm directly (skip trailing S2I) for a
        # downstream matmul_decode(reshard_input=True) consumer.
        return sharded_rms_norm(
            x, scale1, eps, x.shape[-2], x.shape[-1], bias=shift, out_block_sharded=out_block_sharded
        )

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        adarms_cond: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
        precomputed_mod: Optional[Tuple[ttnn.Tensor, ...]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        owned = precomputed_mod is None
        if owned:
            sa1, sha, ga, sf1, shf, gf = self.precompute_mods(adarms_cond)
        else:
            sa1, sha, ga, sf1, shf, gf = precomputed_mod

        normed = self._apply_ada(hidden_states, sa1, sha, self._eps)
        attn_out, new_cache = self.attention(normed, cos, sin, attention_mask, past_key_value, use_cache)
        ttnn.deallocate(normed)
        gated_attn = ttnn.multiply(ga, attn_out, memory_config=_L1)
        ttnn.deallocate(attn_out)
        hidden_states = ttnn.add(hidden_states, gated_attn, memory_config=_L1)
        ttnn.deallocate(gated_attn)

        normed = self._apply_ada(hidden_states, sf1, shf, self._eps)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        gated_ffw = ttnn.multiply(gf, mlp_out, memory_config=_L1)
        ttnn.deallocate(mlp_out)
        hidden_states = ttnn.add(hidden_states, gated_ffw, memory_config=_L1)
        ttnn.deallocate(gated_ffw)

        if owned:
            for ten in (sa1, sha, ga, sf1, shf, gf):
                ttnn.deallocate(ten)
        return hidden_states, new_cache
