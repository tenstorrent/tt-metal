# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tuned denoise expert block (attention/MLP/block) for the pi0.5 streamed-denoise port.

VENDORED from ``tt_symbiote.models.pipelined_pi05.denoise_block`` with env flags BAKED
(plan §9):
  * PI0_DENOISE_MM_TUNE -> ON  (tuned per-matmul pcfg is the perf default)
  * PI0_MM_SWEEP_V2     -> ON  (V2 table confirmed superset of BASE: adds (32,80)->(64,8))
  * PI0_EXPERT_MM_LOFI  -> OFF (HiFi; PCC-safe default. LoFi flip documented in PORT_NOTES)
  * PI0_DENOISE_MLP_WIDE-> OFF (base tuned MLP; the wide variant is dropped)
  * PI0_SDPA_DENOISE_K_FORCE -> DROPPED (auto-pick first divisor of kv_aligned)
``@run_on_devices(P150, BHGLX)`` on every forward (arch-guard widening, plan §8.2).
ZERO tt_symbiote imports, no os.environ reads.
"""
from __future__ import annotations

from typing import Optional, Tuple

import ttnn

from models.experimental.pi0_5.tt.tile_config import TILE_HEIGHT, from_torch_pi05
from models.experimental.pi0_5.tt._ttnn_compat import (
    concat_heads_matmul,
    decode_all_supported,
    kv_sdpa,
    nlp_create_qkv_heads_rope,
)

from ._module import DeviceArch, run_on_devices
from ._trace import trace_enabled
from .modeling.bs import matmul_pcfg, sdpa_program_config
from .modeling.common import get_sdpa_compute_kernel_config, width_sharded_l1_config
from .modeling.gemma import TTNNPi05AdaRMSGemmaBlock, TTNNPi05GemmaAttention, TTNNPi05GemmaMLP

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"
_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG

# BAKED V2 table (PI0_MM_SWEEP_V2 ON; V2 confirmed superset of BASE).
_DENOISE_TUNE_TABLE = {
    (64, 32): (120, 32),
    (128, 32): (24, 32),
    (32, 80): (64, 8),
}


def _denoise_tuned_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, *, activation=None):
    # PI0_DENOISE_MM_TUNE baked ON; tune only applies to m_tiles == 1.
    if m_tiles != 1:
        return None
    override = _DENOISE_TUNE_TABLE.get((k_tiles, n_tiles))
    if override is None:
        return None
    num_cores, in0_bw = override
    num_cores = min(num_cores, grid_x * grid_y)
    if k_tiles % in0_bw != 0:
        return None
    per_core_N = (n_tiles + num_cores - 1) // num_cores if n_tiles % num_cores else n_tiles // num_cores
    eff_budget = 4
    out_sw = min(per_core_N, eff_budget)
    while out_sw > 1 and per_core_N % out_sw != 0:
        out_sw -= 1
    out_sh = max(1, eff_budget // out_sw)
    out_sh = min(m_tiles, out_sh)
    while out_sh > 1 and m_tiles % out_sh != 0:
        out_sh -= 1
    cfg_gx = min(grid_x, num_cores)
    cfg_gy = min(grid_y, (num_cores + cfg_gx - 1) // cfg_gx)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cfg_gx, cfg_gy),
        in0_block_w=in0_bw,
        out_subblock_h=out_sh,
        out_subblock_w=out_sw,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=True,
    )


def _expert_lofi_ck():
    # PI0_EXPERT_MM_LOFI baked OFF (HiFi default). Returns None -> HiFi. LoFi flip = return
    # the WormholeComputeKernelConfig(LoFi, ...) below; gated on PCC >= 0.99 re-validation.
    return None


def _denoise_sdpa_pcfg(q_seq, kv_seq, grid_x, grid_y):
    q_chunk = 32
    kv_aligned = ((kv_seq + 31) // 32) * 32
    k_chunk = None
    for cand in (256, 128, 96, 64, 32):
        if kv_aligned % cand == 0:
            k_chunk = cand
            break
    k_chunk = k_chunk or 32
    return sdpa_program_config(q_seq, kv_seq, grid_x, grid_y, q_chunk=q_chunk, k_chunk=k_chunk)


# --------------------------------------------------------------------------- decode_all mode
# When DECODE_ALL is set (by the walltime test), the denoise block routes all five projection
# matmuls (QKV, o, MLP gate/up/down) through ttnn.matmul_decode (partial-width-sharded resident-L1
# weights + width-sharded input-A reshard) + concat_heads_matmul_decode for the o-proj. Numerics
# are PCC ~1.0 vs the linear path (validated by the single-layer test). Ported from the
# matmul_decode branch (Sankar Manoj matmul_decode op + alnah005 decode_all wiring).
DECODE_ALL = True


def _decode_all_active() -> bool:
    return True
    return DECODE_ALL and decode_all_supported()


# When True (default), the decode_all expert SDPA folds the resident prefix-KV into kv_sdpa's reader
# (two-range read) instead of pre-concatenating prefix+suffix -- eliminates the two per-layer KV concats.
_KV_FOLD = True

# decode_all: the adaRMS norm emits its result block-sharded; each projection matmul_decode reshards
# it internally (reshard_input=True), so the norm skips its trailing sharded_to_interleaved and there
# is no interleaved intermediate between the norm and the projections.

_K_BLOCKS = 2
_RESHARD_CORES = 2
_QKV_N_BLOCKS, _O_N_BLOCKS, _MLP_N_BLOCKS = 40, 32, 32

_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)


def _crs(device, n):
    return ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), True)


def _pws_B(device, w_kn, n_blocks):
    """Partial-width-sharded resident-L1 bf8_b weight for matmul_decode (w_kn is torch [K, N])."""
    k, n = w_kn.shape
    kc, nc = k // _K_BLOCKS, n // n_blocks
    br = w_kn.reshape(_K_BLOCKS, kc, n).permute(1, 0, 2).reshape(kc, n * _K_BLOCKS).contiguous()
    mc = ttnn.create_sharded_memory_config(
        (kc, nc),
        core_grid=_crs(device, _K_BLOCKS * n_blocks),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return from_torch_pi05(br, device=device, memory_config=mc, dtype=ttnn.bfloat8_b)


def _ws_in_mc(device, m, k):
    """Width-sharded input-A memory config (matmul_decode hard-requires WIDTH_SHARDED in0)."""
    return ttnn.create_sharded_memory_config(
        (m, k // _RESHARD_CORES),
        core_grid=_crs(device, _RESHARD_CORES),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _matmul_decode_pws(a, b_pws, n_blocks, device, out_dtype=ttnn.bfloat16, interleaved_output=False):
    """partial-width-sharded matmul_decode against a ``_pws_B`` weight (the only decode op this build
    exposes: ``ttnn.experimental.matmul_decode(a, b, partial_width_sharded, dtype, output_mem_config)``).
    Reshards A to WIDTH-SHARDED L1 over ``_RESHARD_CORES`` (matmul_decode hard-requires width-sharded
    in0), emits a WIDTH-SHARDED ``[padded_m, N/n_blocks]`` output over ``n_blocks`` base cores, and
    optionally converts the result back to interleaved L1."""
    print(
        f"matmul_decode_pws: a.shape={a.shape}, b_pws.shape={b_pws.shape}, n_blocks={n_blocks}, device={device}, out_dtype={out_dtype}, interleaved_output={interleaved_output}"
    )
    m, k = a.shape[-2], a.shape[-1]
    a_ws = ttnn.to_memory_config(a, width_sharded_l1_config(m, k, device, _RESHARD_CORES))
    n = b_pws.shape[-1] // _K_BLOCKS
    padded_m = ((m + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    out_mc = ttnn.create_sharded_memory_config(
        (padded_m, n // n_blocks),
        core_grid=_crs(device, n_blocks),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    out = ttnn.experimental.matmul_decode(
        a_ws, b_pws, partial_width_sharded=True, dtype=out_dtype, output_mem_config=out_mc
    )
    ttnn.deallocate(a_ws)
    if interleaved_output:
        out = ttnn.sharded_to_interleaved(out, _L1)
    return out


def _build_fused_gate_ws(device, gate, w, n_blocks=_MLP_N_BLOCKS):
    """Build the resident WIDTH-SHARDED per-channel gate for a matmul_decode fused-residual epilogue.
    The gate [1,1,W] (per-channel over W) is replicated down all 32 tile rows (so mul_tiles needs no
    broadcast) and width-sharded [32, W/n_blocks] across the n_blocks output base cores (ordering ==
    num_cores_to_corerangeset(n_blocks) -- the first n_blocks of the matmul's B grid). Built ONCE at
    precompute time and held RESIDENT on the block (NOT routed through _to_dram), so it adds ZERO
    per-replay dispatch. Used for both the MLP down (n_blocks=_MLP_N_BLOCKS) and the attention o-proj
    (n_blocks=_O_N_BLOCKS)."""
    g2d = ttnn.reshape(gate, (1, w))
    g_rep = ttnn.repeat(g2d, ttnn.Shape([32, 1]))  # [32, W], gate replicated down the rows
    ttnn.deallocate(g2d)
    mc = ttnn.create_sharded_memory_config(
        (32, w // n_blocks),
        core_grid=_crs(device, n_blocks),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    g_ws = ttnn.to_memory_config(g_rep, mc)
    ttnn.deallocate(g_rep)
    return g_ws


class TTNNPi05DenoiseExpertAttention(TTNNPi05GemmaAttention):
    def _proj(self, x, w, dtype, m_t, grid, ck):
        k_t, n_t = w.shape[-2] // 32, w.shape[-1] // 32
        pc = _denoise_tuned_pcfg(m_t, k_t, n_t, grid.x, grid.y) or matmul_pcfg(
            m_t, k_t, n_t, grid.x, grid.y, in0_block_w=8
        )
        if pc is not None:
            return ttnn.linear(x, w, dtype=dtype, memory_config=_L1, program_config=pc, compute_kernel_config=ck)
        cg = ttnn.CoreGrid(y=1, x=grid.x)
        return ttnn.linear(x, w, dtype=dtype, memory_config=_L1, core_grid=cg, compute_kernel_config=ck)

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
        residual: Optional[ttnn.Tensor] = None,
        gate_ws: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        if len(hidden_states.shape) == 3:
            b, s, _ = hidden_states.shape
            hidden_states = ttnn.reshape(hidden_states, (b, 1, s, hidden_states.shape[-1]))
        else:
            b, _, s, _ = hidden_states.shape

        _g = self.device.compute_with_storage_grid_size()
        _expert_ck = _expert_lofi_ck()

        m_t = s // 32
        if _decode_all_active():
            # adaRMS emits block-sharded (fast 8-core norm, no S2I); matmul_decode reshards in0 in
            # its reader. QKV output stays WIDTH-SHARDED (no interleaved scatter) so
            # nlp_create_qkv_heads_rope reads it directly via TensorAccessor. hidden_states
            # (== the block's `normed`) is owned + freed by the block forward.
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                width_sharded_l1_config(hidden_states.shape[-2], hidden_states.shape[-1], self.device, _RESHARD_CORES),
            )
            # partial_width_sharded matmul_decode reduces the K-partials onto _QKV_N_BLOCKS output
            # base cores; give it the WIDTH-SHARDED [padded_m, N/_QKV_N_BLOCKS] output config so the
            # result is laid out exactly like nlp_create_qkv_heads_rope expects to read it.
            _qkv_n = self.wqkv_b.shape[-1] // _K_BLOCKS
            _qkv_m = ((hidden_states.shape[-2] + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
            _qkv_out_mc = ttnn.create_sharded_memory_config(
                (_qkv_m, _qkv_n // _QKV_N_BLOCKS),
                core_grid=_crs(self.device, _QKV_N_BLOCKS),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            qkv = ttnn.experimental.matmul_decode(
                hidden_states,
                self.wqkv_b,
                partial_width_sharded=True,
                # reshard_input=True,
                # reshard_cores=_RESHARD_CORES,
                # compute_kernel_config=_LOFI,
                dtype=ttnn.bfloat8_b,
                output_mem_config=_qkv_out_mc,
            )
        else:
            qkv = self._proj(hidden_states, self.tt_wqkv, ttnn.bfloat8_b, m_t, _g, _expert_ck)
        # Fused create-qkv-heads + q/k RoPE in ONE dispatch (custom op; byte-identical to
        # nlp_create_qkv_heads + 2x rotary_embedding, PCC 1.0). Replaces 3 launches with 1.
        q, k, v = nlp_create_qkv_heads_rope(qkv, cos, sin, self.num_heads, self.num_kv_heads, memory_config=_L1)
        ttnn.deallocate(qkv)

        # decode_all default: route the small-query MQA expert SDPA through ttnn.kv_sdpa -- a specialized
        # fused-flash op (reuses the production transformer-SDPA sdpa_standard online-softmax, specialized
        # for Sq == 1 tile / single KV head / non-causal), ~17% faster per denoise layer at PCC ~0.9999.
        # Guarded to the supported shape (suffix_len == 32, single KV head); otherwise the tuned ttnn SDPA.
        # kv_sdpa treats attn_mask as a no-op (non-causal full attention -- validated PCC-equal on the mask).
        _use_kv_sdpa = (
            _decode_all_active() and hasattr(ttnn, "kv_sdpa") and int(q.shape[-2]) == 32 and int(self.num_kv_heads) == 1
        )
        # When not caching, kv_sdpa reads the resident prefix-KV (past_k/past_v) + the new suffix K/V as
        # two ranges in its reader, so we skip the two ttnn.concat ops (and the [prefix+suffix] tensor).
        _kv_fold = _KV_FOLD and _use_kv_sdpa and (past_key_value is not None) and (not use_cache)
        if _kv_fold:
            past_k, past_v = past_key_value
            attn_out = kv_sdpa(q, k, v, attn_mask=attention_mask, scale=self.scale, past_k=past_k, past_v=past_v)
            new_cache = None
        else:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = ttnn.concat([past_k, k], dim=2, memory_config=_L1)
                v = ttnn.concat([past_v, v], dim=2, memory_config=_L1)
            new_cache = (k, v) if use_cache else None
            if _use_kv_sdpa:
                attn_out = kv_sdpa(q, k, v, attn_mask=attention_mask, scale=self.scale)
            else:
                kv_seq = k.shape[-2]
                _sdpa_kwargs = {"memory_config": _L1}
                _sdpa_cores = min(_g.x, self.num_heads * ((q.shape[-2] + 31) // 32))
                _spc = _denoise_sdpa_pcfg(q.shape[-2], kv_seq, _sdpa_cores, 1)
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

        # Fused concat-heads + O-projection (custom op wrapping the tuned 1D-mcast matmul): attn_out
        # is consumed directly as in0 (concat = contiguous tiles for seq<=1 tile, PCC ~1.0); the
        # [.,32,K] view is build-time-only so it is trace-replay-safe. 2 launches -> 1. Pass the
        # SAME tuned program config _proj uses for the O-matmul so the matmul stays as fast.
        if _decode_all_active():
            # concat-heads + partial-width-sharded matmul_decode o-proj (bf16 out). The gated
            # residual is applied by the caller's explicit addcmul (fuse disabled on this build).
            heads = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=_L1)
            out = _matmul_decode_pws(heads, self.wo_b, _O_N_BLOCKS, self.device, interleaved_output=True)
            ttnn.deallocate(heads)
        else:
            _o_k, _o_n = self.tt_o.shape[-2] // 32, self.tt_o.shape[-1] // 32
            _o_pc = _denoise_tuned_pcfg(m_t, _o_k, _o_n, _g.x, _g.y) or matmul_pcfg(
                m_t, _o_k, _o_n, _g.x, _g.y, in0_block_w=8
            )
            out = concat_heads_matmul(attn_out, self.tt_o, memory_config=_L1, program_config=_o_pc)
        ttnn.deallocate(attn_out)
        if len(out.shape) == 4:
            out = ttnn.reshape(out, (b, s, out.shape[-1]))
        return out, new_cache


class TTNNPi05DenoiseExpertMLP(TTNNPi05GemmaMLP):
    """GeGLU MLP with an optional decode_all path: gate/up/down via matmul_decode (partial-width-
    sharded resident-L1 weights), gate fuses a tanh-approx gelu. Falls back to the linear path."""

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not _decode_all_active():
            return super().forward(x)
        # adaRMS emits block-sharded (fast 8-core norm, no S2I); matmul_decode reshards in0 in its reader.
        # x (== the block's `normed`) is owned + freed by the block forward.
        # Fused gate+up+GeGLU: ONE A-gather, two resident weights, one output hid = gelu(x@gate)*(x@up)
        # (gate gets the tanh-approx gelu, the multiply is folded into the op). Shares the x-gather
        # across both projections and emits the GeGLU activation directly -- no separate multiply.
        # GeGLU via two partial-width-sharded matmul_decodes (gate + up) sharing the input, the
        # tanh-approx gelu on the gate branch, then the down projection. This build's matmul_decode
        # has no fused gate/up or gelu epilogue, so the activation + multiply are explicit.
        gate = _matmul_decode_pws(x, self.gate_b, _MLP_N_BLOCKS, self.device)
        up = _matmul_decode_pws(x, self.up_b, _MLP_N_BLOCKS, self.device)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
        hid = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        hid = ttnn.sharded_to_interleaved(hid, _L1)
        out = _matmul_decode_pws(hid, self.down_b, _MLP_N_BLOCKS, self.device, interleaved_output=True)
        ttnn.deallocate(hid)
        return out

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward_gated_residual(self, x: ttnn.Tensor, residual: ttnn.Tensor, gate_ws: ttnn.Tensor) -> ttnn.Tensor:
        """decode_all MLP with the gated residual FOLDED into the down matmul_decode:
        returns residual + gate * (gelu(x@gate)*(x@up) @ down_w) -- no separate addcmul. gate_ws is the
        resident width-sharded (row-replicated) per-channel ada gate across the down base cores;
        residual is the interleaved [M,W] hidden."""
        hid = ttnn.gate_up_matmul_decode(
            x,
            self.gate_b,
            self.up_b,
            compute_kernel_config=_LOFI,
            fused_gelu_approx=True,
            reshard_input=True,
            reshard_cores=_RESHARD_CORES,
        )
        out = ttnn.matmul_decode(
            hid,
            self.down_b,
            partial_width_sharded=True,
            reshard_input=True,
            reshard_cores=_RESHARD_CORES,
            compute_kernel_config=_LOFI,
            interleaved_output=True,
            residual=residual,
            gate=gate_ws,
        )
        ttnn.deallocate(hid)
        return out


@trace_enabled
class TTNNPi05DenoiseExpertBlock(TTNNPi05AdaRMSGemmaBlock):
    @classmethod
    def from_torch(cls, block, config):
        new = super().from_torch(block, config)
        new.attention = TTNNPi05DenoiseExpertAttention.from_torch(block.attention, config)
        new.mlp = TTNNPi05DenoiseExpertMLP.from_torch(block.mlp, config)
        return new

    def _apply_ada(self, x, scale1, shift, eps):
        # decode_all: emit the norm block-sharded so the downstream matmul_decode(reshard_input)
        # reshards it internally (no trailing S2I / interleaved intermediate); linear path unchanged.
        return super()._apply_ada(x, scale1, shift, eps, out_block_sharded=_decode_all_active())

    # In the decode_all path both gated residuals are folded into their projection matmul_decode
    # epilogues (out = residual + gate * (A @ W)), eliminating the separate ttnn.addcmul. The linear
    # (DECODE_ALL=False) path keeps the explicit addcmul.
    def _fuse_mlp_residual(self) -> bool:
        # This build's matmul_decode has no fused residual/gate epilogue, so keep the explicit
        # addcmul path (numerically identical) instead of folding into the projection.
        return False

    def _fuse_attn_residual(self) -> bool:
        return False

    def precompute_mods(self, adarms_cond: ttnn.Tensor) -> Tuple[ttnn.Tensor, ...]:
        # When folding a gated residual into its projection matmul, additionally precompute the
        # resident width-sharded per-channel gate ONCE here -- outside the traced forward -- so it
        # adds zero per-replay dispatch. Appended after the 6 base mods in a fixed order: the
        # attention gate (from ga, over _O_N_BLOCKS o-proj base cores) then the MLP gate (from gf,
        # over _MLP_N_BLOCKS down base cores), each present only if its fold is enabled. The fused
        # epilogues consume them and the corresponding addcmul is skipped.
        mods = super().precompute_mods(adarms_cond)
        extra = []
        if self._fuse_attn_residual():
            extra.append(_build_fused_gate_ws(self.device, mods[2], self._width, n_blocks=_O_N_BLOCKS))
        if self._fuse_mlp_residual():
            extra.append(_build_fused_gate_ws(self.device, mods[5], self._width, n_blocks=_MLP_N_BLOCKS))
        return (*mods, *extra)

    def move_weights_to_device_impl(self):
        # L1-resident projection weights (each stage holds <=5 layers, fits in L1): removes the
        # per-matmul DRAM weight read.
        super().move_weights_to_device_impl()
        a, m = self.attention, self.mlp
        if _decode_all_active():
            # decode_all: build partial-width-sharded resident-L1 weights for matmul_decode and
            # free the now-unused interleaved weights (keeps L1 within stage budget).
            import torch

            dev = self.device
            a.wqkv_b = _pws_B(dev, torch.cat([a._q_w.t(), a._k_w.t(), a._v_w.t()], dim=-1).contiguous(), _QKV_N_BLOCKS)
            a.wo_b = _pws_B(dev, a._o_w.t().contiguous(), _O_N_BLOCKS)
            m.gate_b = _pws_B(dev, m._gate_w.t().contiguous(), _MLP_N_BLOCKS)
            m.up_b = _pws_B(dev, m._up_w.t().contiguous(), _MLP_N_BLOCKS)
            m.down_b = _pws_B(dev, m._down_w.t().contiguous(), _MLP_N_BLOCKS)
            for t in (a.tt_wqkv, a.tt_o, m.tt_gate, m.tt_up, m.tt_down):
                ttnn.deallocate(t)
            return
        a.tt_wqkv = ttnn.to_memory_config(a.tt_wqkv, _L1)
        a.tt_o = ttnn.to_memory_config(a.tt_o, _L1)
        m.tt_gate = ttnn.to_memory_config(m.tt_gate, _L1)
        m.tt_up = ttnn.to_memory_config(m.tt_up, _L1)
        m.tt_down = ttnn.to_memory_config(m.tt_down, _L1)

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
        fuse_attn = self._fuse_attn_residual()
        fuse_mlp = self._fuse_mlp_residual()
        owned = precomputed_mod is None
        if owned:
            mods = self.precompute_mods(adarms_cond)
        else:
            mods = precomputed_mod
        # precompute_mods appends the resident width-sharded gates after the 6 base mods, in order:
        # attn gate (if fuse_attn) then MLP gate (if fuse_mlp).
        sa1, sha, ga, sf1, shf, gf = mods[:6]
        _i = 6
        attn_gate_ws = mods[_i] if fuse_attn else None
        _i += 1 if fuse_attn else 0
        mlp_gate_ws = mods[_i] if fuse_mlp else None

        normed = self._apply_ada(hidden_states, sa1, sha, self._eps)
        if fuse_attn:
            # Fold the attention gated residual (hidden + ga*attn_out) INTO the o-proj
            # concat_heads_matmul_decode epilogue: out = hidden + ga * (attn @ Wo). The o-proj reads
            # the old hidden as `residual` internally; we reassign hidden_states to the result and do
            # NOT explicitly free the old hidden (it is the block input -- for block 0 the stage input,
            # held by the caller; for later blocks an intermediate freed on reassignment -- exactly as
            # the original addcmul path leaves its input). Eliminates the addcmul.
            hidden_states, new_cache = self.attention(
                normed,
                cos,
                sin,
                attention_mask,
                past_key_value,
                use_cache,
                residual=hidden_states,
                gate_ws=attn_gate_ws,
            )
            ttnn.deallocate(normed)
        else:
            attn_out, new_cache = self.attention(normed, cos, sin, attention_mask, past_key_value, use_cache)
            ttnn.deallocate(normed)
            # Fused gated residual: hidden + ga*attn_out in one addcmul (was multiply + add).
            hidden_states = ttnn.addcmul(hidden_states, ga, attn_out, memory_config=_L1)
            ttnn.deallocate(attn_out)

        normed = self._apply_ada(hidden_states, sf1, shf, self._eps)
        if fuse_mlp:
            # Fold the MLP gated residual (hidden + gf*mlp_out) INTO the down matmul_decode epilogue:
            # out = hidden + gf * (mlp(normed)). Eliminates the separate addcmul (its dispatch + the
            # [M,W] mlp_out materialization + re-read). The down matmul_decode reads `residual` (the
            # old hidden) internally; free it after.
            residual_in = hidden_states
            hidden_states = self.mlp.forward_gated_residual(normed, residual_in, mlp_gate_ws)
            ttnn.deallocate(normed)
            ttnn.deallocate(residual_in)
        else:
            mlp_out = self.mlp(normed)
            ttnn.deallocate(normed)
            hidden_states = ttnn.addcmul(hidden_states, gf, mlp_out, memory_config=_L1)
            ttnn.deallocate(mlp_out)

        if owned:
            for ten in mods:
                ttnn.deallocate(ten)
        return hidden_states, new_cache


class TTNNPi05DenoiseExpertBlockDRAM(TTNNPi05DenoiseExpertBlock):
    """Oracle-only (plan iter-3 §5): streamed-expert attention/SDPA kernels with DRAM weights
    (no L1 pinning), so all 18 layers fit on one chip for the single-chip drill-down oracle.

    Used ONLY by build_single_stage_reference for the R0.5 oracle; NEVER on the streamed
    4-chip path (build_denoise_loop_pipeline keeps TTNNPi05DenoiseExpertBlock). It INHERITS the
    streamed TTNNPi05DenoiseExpertAttention (swapped in via TTNNPi05DenoiseExpertBlock.from_torch)
    unchanged, so it runs the identical streamed compute -- only the projection-weight residency
    (DRAM vs L1) differs, which is numerically inert (to_memory_config is a copy, not a recompute).
    """

    def move_weights_to_device_impl(self):
        # Skip the expert L1-pin; use the grandparent DRAM placement (the residency the existing
        # parent-block 0.991 oracle already fit). TTNNPi05DenoiseExpertAttention does NOT override
        # move_weights_to_device_impl -> tt_wqkv/tt_o land in DRAM via the grandparent path.
        TTNNPi05AdaRMSGemmaBlock.move_weights_to_device_impl(self)
