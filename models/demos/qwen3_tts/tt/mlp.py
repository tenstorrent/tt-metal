# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP implementation for Qwen3-TTS.

Decode (M=1): down_proj is DRAM-sharded on every card. Gate/up is 1D mcast
on N150 (DRAM-sharded pins those two to the 12 WH banks with in0_block_w=1)
and DRAM-sharded elsewhere. Gate/up output stays sharded into the SiLU·mul
and feeds down_proj's sharded input — no L1_INTERLEAVED round-trip.

Prefill (M>1 tile): 1D-mcast. When the activation is already width-sharded
(piped from RMSNorm) the 1D kernel reads it in place — no S2I. DRAM-sharded
matmul is decode-only (M must be one tile).
"""

import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    build_dram_sharded_weight_tp,
    decode_hidden_width_memcfg,
    dram_sharded_program_config,
    find_grid_k_n,
    unpad_dram_sharded_out,
    width_sharded_l1_memcfg,
)
from models.demos.qwen3_tts.tt.linear_1d_program_config import find_1d_mcast_grid, make_linear_1d_program_config
from models.demos.qwen3_tts.tt.mesh_utils import is_n150
from models.demos.qwen3_tts.tt.model_config import PREFILL_SEQS, SHORT_SEQ_LIMIT

# Swept decode gate/up core grids, keyed by (K, per-chip N) so only the exact shapes
# measured can match — same discipline as _N300_GATE_UP below.
#
# `find_grid_k_n` MAXIMISES the core count, and for a DRAM-sharded matmul
# `in0_block_w = K_tiles / cores`. Maximising cores therefore MINIMISES in0_block_w,
# which is the wrong end for a weight-bandwidth-bound M=1-tile matmul at bfloat8_b:
# halving the bytes per tile stops hiding the per-read fixed cost. tt-perf-report says
# it outright on these ops — "in0_block_w=1 is small, try in0_block_w=2 or above".
#
# Swept in isolation at the model's shapes (bfloat8_b, LoFi + fp32 accumulate, median of
# 4 steady-state launches; tests/test_qwen3_tts_gate_up_sweep_n150.py):
#
#   (2048, 6144)  N150 / TP=1     shipped 1D  @64  in0_block_w=1   91.7 us  146 GB/s  50.6 %
#                                 ->  dram    @16  in0_block_w=4   58.0 us  230 GB/s  80.0 %
#                                 (dram @64, i.e. just flipping _n150_gate_up_1d off,
#                                  is 132.3 us / 101 GB/s — WORSE than shipping)
#   (2048, 3072)  N300 / TP=2     shipped dram @32 in0_block_w=2   45.2 us  148 GB/s  51.3 %
#                                 ->  dram    @8   in0_block_w=8   30.7 us  218 GB/s  75.6 %
#                                 (dram @16 is 30.9 us — tied within noise)
#
# Both SKUs land at 218-230 GB/s, i.e. the rate down_proj already gets. NOT bit-exact:
# a different core count splits the K reduction differently.
# QWEN3_TTS_DECODE_GATE_UP_CORES=<n> forces a grid; =0 restores find_grid_k_n.
#
# In the deployed traced-decode window (median of 3 captures each):
#
#   N150   443 -> 385 us/layer (-13.1 %), 28 -> 30 ops.  End to end 41.78 -> 39.95
#          ms/frame (-4.4 %). PCC mlp_decode 0.999690 -> 0.999693; attention_decode
#          and talker_chain digit-identical. Frame count 82 -> 83.  DEFAULT ON.
#
#   N300   352 -> 329 us/layer (-6.5 %), 32 -> 34 ops; matmul total 167.0 -> 138.1 us
#          and the four CCLs measured 19 us in BOTH arms, so this is not CCL luck.
#          End to end is only 39.39 -> 39.13 ms/frame (-0.7 %) -- the N300 layer
#          carries ~76 us of CCL this does not touch, each chip's matmul is half the
#          size, and the CodePredictor still dominates the frame. Take it for the
#          -29 us of matmul, not for the wall clock.
#
# TP=2 had no numerics gate at all -- test_qwen3_tts_pcc.py opens
# ttnn.open_device(device_id=0), so every test in it runs tp_size=1. Rather than treat
# that as a reason to withhold the N300 arm, test_qwen3_tts_mlp_tp2_pcc.py now covers
# it, and the two arms are indistinguishable on a real 2-chip mesh: PCC vs a
# full-precision torch reference 0.99965531 -> 0.99965531 (identical to 8 digits),
# relative RMS 4.693 -> 4.716 %, arm-to-arm PCC 0.99999285.
#
# Generation length does move (81 -> 94 frames on the perf-gate text, 81 -> 77 on the
# QA text, each re-measured, so it is the numerics and not sampler noise) -- but it
# moves in BOTH directions depending on the prompt, WER was 0.0 % in both arms and SIM
# 0.9016 -> 0.9280 sits inside the +/-0.03 noise band of PERF_NOTES 2.9. That is what a
# benign perturbation looks like, and it is a smaller move than the bfp8 promotion
# (85 -> 79) or the CP fused-SDPA promotion (87 -> 68) that both shipped default ON.
# PERF_NOTES 6.4's >=8-seed sweep is still the honest gate for all three.
#
# NOTE for whoever re-goldens the perf gate: N300 now measures 38.86 / 39.40 ms against
# EXPECTED_STEADY_MS_PER_FRAME=40.1 +/-5 % = [38.09, 42.11]. It passes, but it is ~0.3 ms
# nearer the LOWER bound, and that band breaks from below (see 2.8).
_DECODE_GATE_UP_CORES = {
    (2048, 6144): 16,  # N150 / TP=1
    (2048, 3072): 8,  # N300 / TP=2
}


def _swept_decode_gate_up_grid(hidden, local_inter, k_tiles, n_tiles, cg, rows, cols):
    """(rows, cols, swept_cores) for the decode gate/up matmul.

    Returns the find_grid_k_n choice unchanged (and swept_cores=None) when this shape
    has no swept entry or the requested count is not legal on this grid.
    """
    env = os.environ.get("QWEN3_TTS_DECODE_GATE_UP_CORES")
    if env == "0":
        return rows, cols, None
    want = int(env) if env else _DECODE_GATE_UP_CORES.get((hidden, local_inter))
    if not want or want == rows * cols:
        return rows, cols, None
    if k_tiles % want or n_tiles % want or want > cg.x * cg.y:
        return rows, cols, None
    for r in range(1, cg.y + 1):
        if want % r == 0 and want // r <= cg.x:
            return r, want // r, want
    return rows, cols, None


def _wide_intermediate_memcfg(n_tiles, swept_cores, cg):
    """Widest legal width-shard of the intermediate, for re-parallelising the SiLU-mul.

    A DRAM-sharded matmul writes ITS OWN grid and ignores the grid in the output
    memory_config (verified: ask for 64 cores, get the 16-core shard back). So after a
    narrow-grid gate/up the SiLU-mul inherits that narrow grid — and unlike RMSNorm the
    mul is purely parallelism-bound, so it costs proportionally more there (measured on
    N150: 9 us on 64 cores -> 33 us on 16). Resharding both operands wide first pays 2
    ops to get that back, which measured net -18 us/layer.

    None when gate/up kept its original grid, or nothing wider is legal.
    QWEN3_TTS_DECODE_GATE_UP_WIDEN=0 leaves the mul on the matmul's grid.
    """
    if swept_cores is None or os.environ.get("QWEN3_TTS_DECODE_GATE_UP_WIDEN", "1") == "0":
        return None
    wide = next((c for c in range(cg.x * cg.y, swept_cores, -1) if n_tiles % c == 0), None)
    if wide is None:
        return None
    for r in range(1, cg.y + 1):
        if wide % r == 0 and wide // r <= cg.x:
            return width_sharded_l1_memcfg(m_tiles=1, k_tiles=n_tiles, num_cores_x=wide // r, num_cores_y=r)
    return None


class MLP(LightweightModule):
    """SwiGLU MLP for Qwen3-TTS — down_proj(silu(gate(x)) * up(x))."""

    def __init__(
        self,
        device,
        hidden_size: int,
        intermediate_size: int,
        state_dict: dict,
        layer_prefix: str,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size

        self.tp_size = get_tp_size(device) if is_mesh_device else 1
        assert (
            intermediate_size % self.tp_size == 0
        ), f"intermediate_size={intermediate_size} must be divisible by tp_size={self.tp_size}"
        # Per-chip intermediate after column-parallel gate/up split.
        self.local_intermediate = intermediate_size // self.tp_size

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"{layer_prefix}_{name}".replace(".", "_")

        _mesh_mapper_replicate = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
        _dram = ttnn.DRAM_MEMORY_CONFIG

        def _build_proj_weight(weight_key: str, cache_name: str, mesh_mapper=None):
            # Host-side [out, in] -> [1, 1, in, out] for ttnn.linear; one upload at init.
            weight_host = state_dict[weight_key].transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
            cache_file = get_cache_name(cache_name)
            if cache_file is not None:
                return ttnn.as_tensor(
                    weight_host,
                    device=device,
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=_dram,
                    cache_file_name=cache_file,
                    mesh_mapper=mesh_mapper,
                )
            return ttnn.from_torch(
                weight_host,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=mesh_mapper,
            )

        def _build_colpar_weight(weight_key: str, cache_name: str):
            """Column-parallel: split N (output/intermediate) across TP chips."""
            import torch

            w_full = state_dict[weight_key]  # [out_features, in_features]
            if self.tp_size == 1:
                return _build_proj_weight(weight_key, cache_name, mesh_mapper=_mesh_mapper_replicate)
            chunks = list(torch.chunk(w_full, self.tp_size, dim=0))  # split out_features
            stacked = torch.stack(chunks, dim=0)  # [tp, local_out, in]
            host = stacked.transpose(-2, -1).unsqueeze(0).contiguous()  # [1, tp, in, local_out]
            return ttnn.from_torch(
                host,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
            )

        def _build_rowpar_weight(weight_key: str, cache_name: str):
            """Row-parallel: split K (input / local_intermediate) across TP chips."""
            import torch

            w_full = state_dict[weight_key]  # [out_features, in_features]
            if self.tp_size == 1:
                return _build_proj_weight(weight_key, cache_name, mesh_mapper=_mesh_mapper_replicate)
            # After transpose, K is in_features; split that.
            w_t = w_full.transpose(-2, -1).contiguous()  # [in, out]
            chunks = list(torch.chunk(w_t, self.tp_size, dim=0))  # split in (=local_intermediate per chip)
            stacked = torch.stack(chunks, dim=0).unsqueeze(0).contiguous()  # [1, tp, local_in, out]
            return ttnn.from_torch(
                stacked,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
            )

        # Prefill weights (DRAM_INTERLEAVED) — TP>1: column/row-parallel sharded.
        self.gate_proj = _build_colpar_weight(f"{layer_prefix}.mlp.gate_proj.weight", "gate_proj")
        self.up_proj = _build_colpar_weight(f"{layer_prefix}.mlp.up_proj.weight", "up_proj")
        self.down_proj = _build_rowpar_weight(f"{layer_prefix}.mlp.down_proj.weight", "down_proj")

        # Decode-only DRAM-sharded weights + program/memory configs (M=1 tile).
        # gate_proj and up_proj share K=hidden, N=intermediate so they share configs.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        grid = device.compute_with_storage_grid_size()
        self.short_seq_limit = SHORT_SEQ_LIMIT
        _fp32 = self.compute_kernel_config.fp32_dest_acc_en
        # 1D program configs: gate/up output = local_intermediate per chip; down input = local_intermediate.
        self._decode_gate_up_progcfg = make_linear_1d_program_config(
            m=1, k=hidden_size, n=self.local_intermediate, grid_x=grid.x, grid_y=grid.y, fp32_dest_acc_en=_fp32
        )
        self._decode_down_progcfg = make_linear_1d_program_config(
            m=1, k=self.local_intermediate, n=hidden_size, grid_x=grid.x, grid_y=grid.y, fp32_dest_acc_en=_fp32
        )
        self._short_seq_gate_up_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=hidden_size,
            n=self.local_intermediate,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )
        self._short_seq_down_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=self.local_intermediate,
            n=hidden_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )
        # Gate/up 1D uses the full grid to match width-sharded LN in0.
        # Down is interleaved after silu·mul — pick a 1D grid for blocking.
        _down_gx, _down_gy = find_1d_mcast_grid(self.local_intermediate, hidden_size, grid.x, grid.y)
        self._prefill_gate_up_progcfg = {
            m: make_linear_1d_program_config(m, hidden_size, self.local_intermediate, grid.x, grid.y, _fp32)
            for m in PREFILL_SEQS
            if m > self.short_seq_limit
        }
        # Swept prefill gate/up overrides, keyed by exact (seq, K, per-chip N) so only the
        # shapes actually measured can match. `_prefill_gate_up_progcfg` above uses the FULL
        # grid, which drives in0_block_w to 1 (K=64 tiles / 64 cores) — the same wrong end of
        # the tradeoff that cost 34 us/matmul in decode.
        #
        # `in0_sharded` asks for a width-sharded in0 on the matmul's own grid. It is not
        # optional for a narrower grid here: on N150 the post-attention RMSNorm hands gate/up
        # a 64-core width shard (2048/64 = 1 tile per core), and a 32-core config derives
        # in0_block_w=2, which the matmul refuses --
        #   "shard_shape[1] (32) / in0_tile width (32) must be divisible by in0_block_w (2)".
        # The isolated sweep fed an L1-interleaved in0 and so never saw this. Resharding to
        # the matmul's 32-core grid gives 2 tiles per core and in0_block_w=2 divides.
        #
        # N300 / TP=2, N=3072: 64 cores gives per_core_N=1.5 worth of work and 145 GB/s,
        # 19-26 us/matmul slower than the auto-routing it replaced; 32 cores with
        # in0_block_w=2 recovers 22-25 %, net of the in0 reshard.
        #
        # N150 / TP=1, N=6144: the full-grid choice was swept at **bf16** ("210 GB/s, 73 % of
        # DRAM peak"). At bfp8 it is 135 GB/s. Re-swept in isolation at the model's shapes
        # (bfloat8_b, LoFi + fp32 acc, median of 4 steady launches,
        # test_qwen3_tts_prefill_mm_sweep_n150.py); isolated time tracks the in-model window
        # to ~1 us on every shape:
        #
        #   m=64   c64 ibw=1 (shipped) 99.0 us 135 GB/s | c32 ibw=2 71.3 us 187 GB/s  <- best
        #          c32 sharded-in0 72.4 (tied, so not worth the reshard) | c16 86.0 | 2D 144.4
        #   m=128  c64 ibw=1 (shipped) 125.2 us        | c32 ibw=2 117.9 us           <- best
        #
        # The demo pads 61 tokens to bucket 64, so m=64 is the one on its critical path.
        _PREFILL_GATE_UP = {
            # (seq, K, N): (num_cores, out_subblock, in0_sharded)
            (64, 2048, 3072): (32, (2, 1), True),  # N300 / TP=2
            (128, 2048, 3072): (32, (1, 3), True),  # N300 / TP=2
            (64, 2048, 6144): (32, None, True),  # N150 / TP=1
            (128, 2048, 6144): (32, None, True),  # N150 / TP=1
        }
        self._prefill_gate_up_n300 = {}
        self._prefill_gate_up_in0_memcfg = {}
        _mm_override = os.environ.get("QWEN3_TTS_N300_MM_OVERRIDE", "1") != "0"
        for (_m, _k, _n), (_cores, _sb, _in0_sharded) in _PREFILL_GATE_UP.items() if _mm_override else ():
            if _k != hidden_size or _n != self.local_intermediate or _m not in self._prefill_gate_up_progcfg:
                continue
            self._prefill_gate_up_n300[_m] = make_linear_1d_program_config(
                _m, _k, _n, grid.x, grid.y, _fp32, num_cores=_cores, out_subblock=_sb
            )
            if _in0_sharded:
                self._prefill_gate_up_in0_memcfg[_m] = width_sharded_l1_memcfg(
                    _m // 32, _k // 32, min(_cores, grid.x), max(1, _cores // grid.x)
                )
        self._prefill_down_progcfg = {
            m: make_linear_1d_program_config(m, self.local_intermediate, hidden_size, _down_gx, _down_gy, _fp32)
            for m in PREFILL_SEQS
            if m > self.short_seq_limit
        }
        # Prefill down: the GRID is already right (find_1d_mcast_grid picks 32 cores, so
        # in0_block_w = 192/32 = 6). What is wrong is the in0 LAYOUT. `hidden` arrives
        # L1-INTERLEAVED from the SiLU-mul, so the mcast sender re-reads it out of
        # interleaved L1 for every K block; a width shard on the matmul's own grid hands
        # each core its 6 K-tiles up front. Same program config, only the in0 memcfg
        # changes (test_qwen3_tts_prefill_mm_sweep_n150.py, median of 4 steady launches):
        #
        #   m=64   c32 interleaved-in0  85.4 us 156 GB/s -> sharded-in0  72.1 us 185 GB/s
        #   m=128  c32 interleaved-in0 149.5 us  89 GB/s -> sharded-in0 110.7 us 121 GB/s
        #
        # K=6144 is TP=1 only; N300's down is 3072x2048 and was not swept, so it does not
        # match and keeps the interleaved in0. QWEN3_TTS_PREFILL_DOWN_SHARD_IN0=0 reverts.
        _PREFILL_DOWN_SHARD_IN0 = {
            # (seq, K, N): num_cores — must equal the program config's grid
            (64, 6144, 2048): 32,  # N150 / TP=1
            (128, 6144, 2048): 32,  # N150 / TP=1
        }
        self._prefill_down_in0_memcfg = {}
        if os.environ.get("QWEN3_TTS_PREFILL_DOWN_SHARD_IN0", "1") != "0":
            for (_m, _k, _n), _cores in _PREFILL_DOWN_SHARD_IN0.items():
                if _k != self.local_intermediate or _n != hidden_size or _m not in self._prefill_down_progcfg:
                    continue
                if _cores != _down_gx * _down_gy:
                    continue  # the shard grid must match the config the matmul actually runs
                self._prefill_down_in0_memcfg[_m] = width_sharded_l1_memcfg(
                    _m // 32, _k // 32, min(_cores, grid.x), max(1, _cores // grid.x)
                )

        # DRAM-sharded decode path — now supported for TP>1 too.
        # TP=2 benefit: each chip has smaller dimensions (local_intermediate = intermediate // tp)
        # so tensors fit L1 more easily and DRAM bandwidth splits across chips.
        _cg = device.compute_with_storage_grid_size()
        if self.tp_size > 1:
            # Column-parallel gate/up: full weight [hidden, intermediate] sharded along N (dim=1).
            gate_w_kn_full = state_dict[f"{layer_prefix}.mlp.gate_proj.weight"].transpose(-2, -1).contiguous()
            up_w_kn_full = state_dict[f"{layer_prefix}.mlp.up_proj.weight"].transpose(-2, -1).contiguous()
            self.gate_proj_dram_sharded, k_gu, n_padded_gu = build_dram_sharded_weight_tp(
                gate_w_kn_full, device, self.tp_size, split_dim=1, dtype=weight_dtype
            )
            self.up_proj_dram_sharded, _, _ = build_dram_sharded_weight_tp(
                up_w_kn_full, device, self.tp_size, split_dim=1, dtype=weight_dtype
            )
            self._decode_gate_up_n_padded = n_padded_gu
            k_tiles_gu, n_tiles_gu = k_gu // 32, n_padded_gu // 32
            rows_gu, cols_gu = find_grid_k_n(k_tiles_gu, n_tiles_gu, max_rows=_cg.y, max_cols=_cg.x)
            rows_gu, cols_gu, self._decode_gate_up_swept_cores = _swept_decode_gate_up_grid(
                hidden_size, self.local_intermediate, k_tiles_gu, n_tiles_gu, _cg, rows_gu, cols_gu
            )
            self._decode_gate_up_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=k_gu, n=n_padded_gu, num_cores=rows_gu * cols_gu
            )
            self._decode_gate_up_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=k_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
            )
            self._decode_gate_up_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=n_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
            )
            # Row-parallel down: full weight [intermediate, hidden] sharded along K (dim=0).
            down_w_kn_full = state_dict[f"{layer_prefix}.mlp.down_proj.weight"].transpose(-2, -1).contiguous()
            self.down_proj_dram_sharded, k_d, n_padded_d = build_dram_sharded_weight_tp(
                down_w_kn_full, device, self.tp_size, split_dim=0, dtype=weight_dtype
            )
            self._decode_down_n_padded = n_padded_d
            k_tiles_d, n_tiles_d = k_d // 32, n_padded_d // 32
            rows_d, cols_d = find_grid_k_n(k_tiles_d, n_tiles_d, max_rows=_cg.y, max_cols=_cg.x)
            self._decode_down_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=k_d, n=n_padded_d, num_cores=rows_d * cols_d
            )
            self._decode_down_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=k_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
            )
            self._decode_down_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=n_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
            )
            self._n150_gate_up_1d = False
            self._decode_gate_up_wide_memcfg = _wide_intermediate_memcfg(
                n_tiles_gu, self._decode_gate_up_swept_cores, _cg
            )
            self._decode_residual_memcfg = None
            return

        gate_w_kn = state_dict[f"{layer_prefix}.mlp.gate_proj.weight"].transpose(-2, -1).contiguous()
        up_w_kn = state_dict[f"{layer_prefix}.mlp.up_proj.weight"].transpose(-2, -1).contiguous()
        self.gate_proj_dram_sharded, k_gu, n_padded_gu = build_dram_sharded_weight(
            gate_w_kn, device, dtype=weight_dtype
        )
        self.up_proj_dram_sharded, _, _ = build_dram_sharded_weight(up_w_kn, device, dtype=weight_dtype)
        self._decode_gate_up_n_padded = n_padded_gu
        k_tiles_gu, n_tiles_gu = k_gu // 32, n_padded_gu // 32
        # find_grid_k_n must honor the actual compute grid (8x8 on WH N150, 13x10 on BH P150).
        _cg = device.compute_with_storage_grid_size()
        rows_gu, cols_gu = find_grid_k_n(k_tiles_gu, n_tiles_gu, max_rows=_cg.y, max_cols=_cg.x)
        rows_gu, cols_gu, self._decode_gate_up_swept_cores = _swept_decode_gate_up_grid(
            hidden_size, self.local_intermediate, k_tiles_gu, n_tiles_gu, _cg, rows_gu, cols_gu
        )
        self._decode_gate_up_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=k_gu, n=n_padded_gu, num_cores=rows_gu * cols_gu
        )
        self._decode_gate_up_in0_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=k_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
        )
        self._decode_gate_up_out_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=n_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
        )
        self._decode_gate_up_wide_memcfg = _wide_intermediate_memcfg(n_tiles_gu, self._decode_gate_up_swept_cores, _cg)

        down_w_kn = state_dict[f"{layer_prefix}.mlp.down_proj.weight"].transpose(-2, -1).contiguous()
        self.down_proj_dram_sharded, k_d, n_padded_d = build_dram_sharded_weight(down_w_kn, device, dtype=weight_dtype)
        self._decode_down_n_padded = n_padded_d
        k_tiles_d, n_tiles_d = k_d // 32, n_padded_d // 32
        rows_d, cols_d = find_grid_k_n(k_tiles_d, n_tiles_d, max_rows=_cg.y, max_cols=_cg.x)
        self._decode_down_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=k_d, n=n_padded_d, num_cores=rows_d * cols_d
        )
        self._decode_down_in0_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=k_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
        )
        self._decode_down_out_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=n_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
        )
        # N150 decode gate/up: DRAM-sharded on the swept narrow grid when the override
        # above found one, else 1D mcast on the full compute grid (the pre-sweep path).
        # Residual dest is the same N150-only choice: slice padded-N into the
        # decode RMSNorm shard spec so the next layer skips I2S.
        _n150 = is_n150(device)
        self._n150_gate_up_1d = _n150 and self._decode_gate_up_swept_cores is None
        self._decode_residual_memcfg = decode_hidden_width_memcfg(device, hidden_size) if _n150 else None
        if self._n150_gate_up_1d:
            self._decode_gate_up_1d_progcfg = make_linear_1d_program_config(
                1,
                hidden_size,
                self.local_intermediate,
                _cg.x,
                _cg.y,
                _fp32,
                num_cores=rows_gu * cols_gu,
            )

    def forward(self, x: ttnn.Tensor, mode: str = "prefill") -> ttnn.Tensor:
        """Apply SwiGLU MLP.

        Decode (mode=="decode" or seq_len==1): DRAM-sharded matmul chain.
        Prefill: standard 1D-mcast matmul; for seq>=1024 we reshape to fit on device.
        """
        seq_len = x.shape[-2]
        is_decode = mode == "decode" or seq_len == 1

        # Keep activations in L1 for all sequence lengths. Very long sequences
        # (seq >= 1024) are split into 1024-tile chunks via ttnn.reshape below, so
        # the per-chunk tensor is always ≤ local_intermediate × 1024 × 2 bytes ≈ 6 MB
        # on TP=2 (local_intermediate=3072) or 12 MB on TP=1 — both well within L1.
        mem_cfg = ttnn.L1_MEMORY_CONFIG
        if is_decode:
            gate_up_progcfg = self._decode_gate_up_progcfg
            down_progcfg = self._decode_down_progcfg
        elif seq_len in self._prefill_gate_up_progcfg:
            gate_up_progcfg = self._prefill_gate_up_n300.get(seq_len) or self._prefill_gate_up_progcfg[seq_len]
            down_progcfg = self._prefill_down_progcfg[seq_len]
        elif seq_len <= self.short_seq_limit:
            gate_up_progcfg = self._short_seq_gate_up_progcfg
            down_progcfg = self._short_seq_down_progcfg
        else:
            gate_up_progcfg = down_progcfg = None

        # Reshape for very large sequences to fit on device.
        if seq_len >= 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # seq <= SHORT_SEQ_LIMIT (32) is exactly ONE tile row, which is the only thing the
        # DRAM-sharded chain asks of M — and every decode config above is already built
        # with `m=32` / `m_tiles=1`, so this is a reuse, not a new config. Attention has
        # taken the DRAM-sharded path at this bucket all along
        # (`use_dram_shard_qkv = ... or seq_len <= short_seq_limit`, attention.py); the MLP
        # was left on `_short_seq_*_progcfg`, which is 1D mcast on the FULL 64-core grid
        # and therefore drives in0_block_w to K_tiles/64 = 1. Shipped bucket-32 window vs
        # the same shapes on the DRAM-sharded chain:
        #
        #   gate  32x2048x6144  1D c64 ibw=1  94 us (46.5 %) -> DRAM-sharded  58 us (75.4 %)
        #   up    32x2048x6144  1D c64 ibw=1  94 us          -> DRAM-sharded  58 us
        #   down  32x6144x2048  1D c64        98 us (44.6 %) -> DRAM-sharded  64 us (77.0 %)
        #
        # QWEN3_TTS_SHORT_SEQ_DRAM_MLP=0 keeps the 1D path.
        _dram_chain = is_decode or (
            seq_len <= self.short_seq_limit
            and seq_len == 32
            and os.environ.get("QWEN3_TTS_SHORT_SEQ_DRAM_MLP", "1") != "0"
        )

        # Decode path: DRAM-sharded chain. Now enabled for all tp_size values.
        # TP>1: each chip runs its per-chip (smaller) DRAM-sharded matmuls, then all_reduce
        # on the down output combines partial sums.
        if _dram_chain and seq_len < 1024:
            # Width-shard x once, reuse for both gate and up. Skip the I→S if the
            # caller already gave us a tensor in the matching shard config (e.g. piped
            # from a sharded layernorm in decoder_layer).
            if x.memory_config() == self._decode_gate_up_in0_memcfg:
                x_sharded = x
                _own_x_sharded = False
            else:
                x_sharded = ttnn.to_memory_config(x, self._decode_gate_up_in0_memcfg)
                _own_x_sharded = True
            # N150 gate/up: 1D mcast + interleaved weights. Down stays DRAM-sharded.
            if self._n150_gate_up_1d:
                gate_w, up_w = self.gate_proj, self.up_proj
                gate_up_pc = self._decode_gate_up_1d_progcfg
            else:
                gate_w, up_w = self.gate_proj_dram_sharded, self.up_proj_dram_sharded
                gate_up_pc = self._decode_gate_up_dramshard_progcfg
            gate_sharded = ttnn.linear(
                x_sharded,
                gate_w,
                compute_kernel_config=self.compute_kernel_config,
                program_config=gate_up_pc,
                memory_config=self._decode_gate_up_out_memcfg,
            )
            up_sharded = ttnn.linear(
                x_sharded,
                up_w,
                compute_kernel_config=self.compute_kernel_config,
                program_config=gate_up_pc,
                memory_config=self._decode_gate_up_out_memcfg,
            )
            if _own_x_sharded:
                ttnn.deallocate(x_sharded)
            # The narrow DRAM-sharded gate/up grid is right for the MATMULS and wrong
            # for the SiLU·mul that follows. A DRAM-sharded matmul writes its own grid
            # and ignores the output memory_config's grid (verified: asking for 64 cores
            # still returns a 16-core shard), so the mul inherits 16 cores and, being
            # purely parallelism-bound, costs 4x more there: 9 us on 64 cores -> 33 us
            # on 16. Widening both operands first pays 2 reshards to get that back.
            _mul_memcfg = self._decode_gate_up_out_memcfg
            if self._decode_gate_up_wide_memcfg is not None:
                gate_wide = ttnn.to_memory_config(gate_sharded, self._decode_gate_up_wide_memcfg)
                ttnn.deallocate(gate_sharded)
                gate_sharded = gate_wide
                up_wide = ttnn.to_memory_config(up_sharded, self._decode_gate_up_wide_memcfg)
                ttnn.deallocate(up_sharded)
                up_sharded = up_wide
                _mul_memcfg = self._decode_gate_up_wide_memcfg
            # gate/up sharding == down in0 sharding (same K). Mul preserves layout.
            hidden_sharded = ttnn.mul(
                gate_sharded,
                up_sharded,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=_mul_memcfg,
            )
            ttnn.deallocate(gate_sharded)
            ttnn.deallocate(up_sharded)
            # Gate-up output layout and down input layout match only when N for
            # down doesn't get DRAM-padded (i.e. hidden_size already multiple of
            # TILE*dram_cores). True on BH P150 (dram_cores=8, hidden=2048) but
            # not on Wormhole (dram_cores=12 → hidden pads 2048→2304, giving
            # down a different num_cores). Reshard when they differ.
            if _mul_memcfg != self._decode_down_in0_memcfg:
                hidden_for_down = ttnn.to_memory_config(hidden_sharded, self._decode_down_in0_memcfg)
                ttnn.deallocate(hidden_sharded)
            else:
                hidden_for_down = hidden_sharded
            out_sharded = ttnn.linear(
                hidden_for_down,
                self.down_proj_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._decode_down_dramshard_progcfg,
                memory_config=self._decode_down_out_memcfg,
            )
            ttnn.deallocate(hidden_for_down)
            # Decoder layer's residual add wants sharded input — return sharded (TP=1).
            # TP>1: all_reduce after unpadding — go via L1_INTERLEAVED for CCL compat.
            if self._decode_down_n_padded == self.hidden_size:
                if self.tp_size > 1:
                    from models.demos.qwen3_tts.tt.mesh_utils import tp_all_reduce

                    output_il = ttnn.to_memory_config(out_sharded, ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(out_sharded)
                    return tp_all_reduce(output_il, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
                return out_sharded
            # Weight N was padded; trim straight off the width-sharded output.
            output = unpad_dram_sharded_out(
                out_sharded,
                self.hidden_size,
                self._decode_residual_memcfg if self._decode_residual_memcfg is not None else ttnn.L1_MEMORY_CONFIG,
            )
            if self.tp_size > 1:
                from models.demos.qwen3_tts.tt.mesh_utils import tp_all_reduce

                output = tp_all_reduce(output, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
            return output

        # Prefill path: 1D-mcast. Width-sharded in0 (from RMSNorm) is consumed in place —
        # except on the N300 override, whose 32-core config needs a narrower in0 shard. That
        # reshard is done once and shared by gate and up.
        _gu_in0 = self._prefill_gate_up_in0_memcfg.get(seq_len) if not is_decode else None
        _own_x = False
        if _gu_in0 is not None and x.memory_config() != _gu_in0:
            x = ttnn.to_memory_config(x, _gu_in0)
            _own_x = True
        # gate/up's N is exactly down's K, so the width shard built for down's in0 is
        # also a legal output layout for gate/up — and writing into it is 3.8 us cheaper
        # per matmul than writing L1-interleaved. The SiLU-mul then READS sharded and
        # still writes interleaved at the same cost, so the win is free of any op change
        # (probed at m=64: gate/up 71.8 -> 68.0 us each, mul 19.7 -> 19.5 us):
        #
        #   gu sharded + mul interleaved   m=64  167.4 -> 159.5 us  (-7.8 us/layer)
        #                                  m=128 279.9 -> 266.8 us  (-13.1 us/layer)
        #
        # Asking the MUL to write sharded instead loses: it nearly doubles (19.7 -> 32.6),
        # which costs more than the reshard it would remove. So the reshard below stays.
        _gu_out = self._prefill_down_in0_memcfg.get(seq_len) if not is_decode else None
        _gu_mem = _gu_out if _gu_out is not None else mem_cfg
        gate_out = ttnn.linear(
            x,
            self.gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=_gu_mem,
            program_config=gate_up_progcfg,
        )
        up_out = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=_gu_mem,
            program_config=gate_up_progcfg,
        )
        if _own_x:
            ttnn.deallocate(x)
        hidden = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=mem_cfg,
        )
        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)
        _down_in0 = self._prefill_down_in0_memcfg.get(seq_len) if not is_decode else None
        if _down_in0 is not None and hidden.memory_config() != _down_in0:
            _hidden_sharded = ttnn.to_memory_config(hidden, _down_in0)
            ttnn.deallocate(hidden)
            hidden = _hidden_sharded
        output = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mem_cfg,
            program_config=down_progcfg,
        )
        ttnn.deallocate(hidden)
        if seq_len >= 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])
        # Row-parallel down-proj on TP>1: each chip has a partial sum; all_reduce gives full hidden.
        if self.tp_size > 1:
            from models.demos.qwen3_tts.tt.mesh_utils import tp_all_reduce

            output = tp_all_reduce(output, self.device, memory_config=mem_cfg)
        return output
