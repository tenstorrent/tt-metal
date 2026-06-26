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

from ._module import DeviceArch, run_on_devices
from ._trace import trace_enabled
from .modeling.bs import matmul_pcfg, sdpa_program_config
from .modeling.common import get_sdpa_compute_kernel_config
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
    cfg_gy = (num_cores + cfg_gx - 1) // cfg_gx
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
DECODE_ALL = False

# decode_all keeps the activation WIDTH-SHARDED across consecutive ops: the adaRMS norm emits its
# output ALREADY in the 2-core width-sharded layout matmul_decode wants, so the norm's trailing
# ShardedToInterleaved AND the InterleavedToSharded before qkv / gate-up are eliminated (norm PCC
# 0.99996 vs golden). Genuine layout boundaries (SDPA out -> concat_heads, gate/up N-shard -> down
# K-shard) keep their conversion.

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
    return ttnn.from_torch(br, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc, dtype=ttnn.bfloat8_b)


def _ws_in_mc(device, m, k):
    """Width-sharded input-A memory config (matmul_decode hard-requires WIDTH_SHARDED in0)."""
    return ttnn.create_sharded_memory_config(
        (m, k // _RESHARD_CORES),
        core_grid=_crs(device, _RESHARD_CORES),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _wshard_rms_norm(x, weight, eps, bias, device):
    """adaRMS that emits the result ALREADY in the 2-core WIDTH_SHARDED layout matmul_decode wants
    (== _ws_in_mc). One I2S to shard the input, the norm computes width-sharded across _RESHARD_CORES
    cores, NO trailing sharded_to_interleaved -- so the downstream qkv/gate-up matmul_decode consumes
    it with ZERO reshard. Numerically identical to sharded_rms_norm (PCC 0.99996 vs golden)."""
    m, k = x.shape[-2], x.shape[-1]
    ws_mc = _ws_in_mc(device, m, k)
    x_sh = ttnn.to_memory_config(x, ws_mc)  # I2S (interleaved -> width-shard 2-core); norm needs sharded in
    m_t = m // 32
    block_w = (k // 32) // _RESHARD_CORES
    # subblock_w must be <= 8 tiles (kernel constraint at dst_full_sync_en=false); pick the largest divisor.
    subblock_w = block_w
    while subblock_w > 8 or block_w % subblock_w != 0:
        subblock_w -= 1
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(_RESHARD_CORES, 1),
        subblock_w=subblock_w,
        block_h=m_t,
        block_w=block_w,
        inplace=False,
    )
    normed = ttnn.rms_norm(x_sh, weight=weight, bias=bias, epsilon=eps, program_config=pc, memory_config=ws_mc)
    ttnn.deallocate(x_sh)
    return normed  # WIDTH_SHARDED on _RESHARD_CORES cores -- feed straight to matmul_decode


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
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        if len(hidden_states.shape) == 3:
            b, s, _ = hidden_states.shape
            hidden_states = ttnn.reshape(hidden_states, (b, 1, s, hidden_states.shape[-1]))
        else:
            b, _, s, _ = hidden_states.shape

        _g = self.device.compute_with_storage_grid_size()
        _expert_ck = _expert_lofi_ck()

        m_t = s // 32
        if DECODE_ALL:
            # adaRMS norm already emitted width-sharded (== _ws_in_mc); consume directly, no I2S.
            # hidden_states (== the block's `normed`) is owned + freed by the block forward.
            qkv = ttnn.matmul_decode(
                hidden_states,
                self.wqkv_b,
                partial_width_sharded=True,
                compute_kernel_config=_LOFI,
                interleaved_output=True,
                dtype=ttnn.bfloat8_b,
            )
        else:
            qkv = self._proj(hidden_states, self.tt_wqkv, ttnn.bfloat8_b, m_t, _g, _expert_ck)
        # Fused create-qkv-heads + q/k RoPE in ONE dispatch (custom op; byte-identical to
        # nlp_create_qkv_heads + 2x rotary_embedding, PCC 1.0). Replaces 3 launches with 1.
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_rope(
            qkv, cos, sin, self.num_heads, self.num_kv_heads, memory_config=_L1
        )
        ttnn.deallocate(qkv)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ttnn.concat([past_k, k], dim=2, memory_config=_L1)
            v = ttnn.concat([past_v, v], dim=2, memory_config=_L1)
        new_cache = (k, v) if use_cache else None

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
        if DECODE_ALL:
            # FREE-view concat-heads + matmul_decode o-proj (partial-width-sharded), bf16 out.
            out = ttnn.experimental.concat_heads_matmul_decode(
                attn_out,
                self.wo_b,
                output_dtype=ttnn.bfloat16,
                compute_kernel_config=_LOFI,
                reshard_cores=_RESHARD_CORES,
            )
        else:
            _o_k, _o_n = self.tt_o.shape[-2] // 32, self.tt_o.shape[-1] // 32
            _o_pc = _denoise_tuned_pcfg(m_t, _o_k, _o_n, _g.x, _g.y) or matmul_pcfg(
                m_t, _o_k, _o_n, _g.x, _g.y, in0_block_w=8
            )
            out = ttnn.experimental.concat_heads_matmul(attn_out, self.tt_o, memory_config=_L1, program_config=_o_pc)
        ttnn.deallocate(attn_out)
        if len(out.shape) == 4:
            out = ttnn.reshape(out, (b, s, out.shape[-1]))
        return out, new_cache


class TTNNPi05DenoiseExpertMLP(TTNNPi05GemmaMLP):
    """GeGLU MLP with an optional decode_all path: gate/up/down via matmul_decode (partial-width-
    sharded resident-L1 weights), gate fuses a tanh-approx gelu. Falls back to the linear path."""

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not DECODE_ALL:
            return super().forward(x)
        s = x.shape[-2]
        # adaRMS norm already emitted width-sharded (== _ws_in_mc); consume x directly, no I2S.
        # x (== the block's `normed`) is owned + freed by the block forward.
        gate = ttnn.matmul_decode(
            x,
            self.gate_b,
            partial_width_sharded=True,
            compute_kernel_config=_LOFI,
            fused_gelu=True,
            fused_gelu_approx=True,
        )
        up = ttnn.matmul_decode(x, self.up_b, partial_width_sharded=True, compute_kernel_config=_LOFI)
        hid = ttnn.multiply(gate, up, memory_config=gate.memory_config())
        hid2 = ttnn.to_memory_config(hid, _ws_in_mc(self.device, s, hid.shape[-1]))
        out = ttnn.matmul_decode(
            hid2, self.down_b, partial_width_sharded=True, compute_kernel_config=_LOFI, interleaved_output=True
        )
        for t in (gate, up, hid, hid2):
            ttnn.deallocate(t)
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
        # decode_all: emit width-sharded (2-core) so the next matmul_decode skips its input I2S and
        # the norm skips its trailing S2I. Linear path: parent's block-sharded + S2I-to-interleaved.
        if DECODE_ALL:
            return _wshard_rms_norm(x, scale1, eps, shift, self.device)
        return super()._apply_ada(x, scale1, shift, eps)

    def move_weights_to_device_impl(self):
        # L1-resident projection weights (each stage holds <=5 layers, fits in L1): removes the
        # per-matmul DRAM weight read.
        super().move_weights_to_device_impl()
        a, m = self.attention, self.mlp
        if DECODE_ALL:
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
        owned = precomputed_mod is None
        if owned:
            sa1, sha, ga, sf1, shf, gf = self.precompute_mods(adarms_cond)
        else:
            sa1, sha, ga, sf1, shf, gf = precomputed_mod

        normed = self._apply_ada(hidden_states, sa1, sha, self._eps)
        attn_out, new_cache = self.attention(normed, cos, sin, attention_mask, past_key_value, use_cache)
        ttnn.deallocate(normed)
        # Fused gated residual: hidden + ga*attn_out in one addcmul (was multiply + add).
        hidden_states = ttnn.addcmul(hidden_states, ga, attn_out, memory_config=_L1)
        ttnn.deallocate(attn_out)

        normed = self._apply_ada(hidden_states, sf1, shf, self._eps)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.addcmul(hidden_states, gf, mlp_out, memory_config=_L1)
        ttnn.deallocate(mlp_out)

        if owned:
            for ten in (sa1, sha, ga, sf1, shf, gf):
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
