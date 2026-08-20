# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP: down(silu(gate(x)) * up(x)).

9B (single device): dense matmuls, full weights.
27B TP (1,4 mesh): w1/w3 column-parallel, w2 row-parallel; tt_all_reduce
reduce-scatters on meshes with a dim-1 shape (e.g. P150x4), fracturing hidden.
"""

import os
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj [in, out], bfloat8_b
    w3: ttnn.Tensor  # up_proj [in, out], bfloat4_b
    w_gate_up: ttnn.Tensor = None  # TP prefill: tile-pair-interleaved packed [gate|up] for fused-swiglu AGMM


# WORMHOLE MLP prefill: one K pass for gate/up/down.
#
# The unfused prefill arm below used halve_out_block=True because the full per_core_N-wide
# output/intermediate CB (gate AND up live at once) overflowed WH L1 by ~28KB. Halving out_block_w is
# what turns one K pass into two, and each extra pass re-reads the DRAM-resident activation. Dropping
# fp32 dest accumulation halves that CB and raises the output-subblock cap from 4 to 8, so the
# full-width block fits -- the same unlock that was worth -16% on the GDN in-projection. packer_l1_acc
# goes ON at the same time (prefill had it off while decode had it on, with no comment defending it).
#
# MEASURED IN THE REAL LAYER (Tracy, T=2048, N300, full decoder layer):
#     gate  2048x4096x6144  1,321us @ 33.5% of peak  ->  982us @ 45.1%   (-339us)
#     up    2048x4096x6144  1,183us @ 37.4%          ->  903us @ 49.0%   (-280us)
#     down  2048x6144x4096  1,110us @ 39.9%          -> 1,013us @ 43.7%  (-97us)
#                                                                        = -716us/layer
# The GDN in-proj and out-proj matmuls are byte-identical across the same capture, which is the check
# that this is scoped to the MLP.
#
# ACCURACY: negligible. MLP PCC at T=2048 goes 0.9867734 -> 0.9866364 (test_mlp_tp), because LoFi with
# bfp4 gate/up weights already dominates the error budget -- fp32 dest accumulation was never what was
# holding this matmul's precision up.
#
# CAUTION on measurement: the isolated sweep in tests/perf/test_mlp_matmul_sweep_prefill.py CANNOT
# decide this. It dispatches 8 back-to-back copies of the same matmul over an 18MB bfp4 weight and runs
# DRAM-saturated, which penalises packer_l1_acc=False far more than the real layer (whose gate matmul
# sits at 41 GB/s) -- it reported the current config at 2,060us against a real 1,321us. Trust the
# full-layer capture for this shape.
#
# Wormhole only. On Blackhole at TP>1 mlp_gateup_agmm_enabled is always True, so the branch below is
# unreachable there; the guard makes that explicit rather than relying on it.
_CKC_MLP_KPASS1 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
)


def _build_gate_up(gate_w, up_w, mesh, tp, cache_path):
    """Packed [gate|up] weight for all_gather_swiglu_prefill: prepare_for_fused_swiglu tile-pair
    interleave, then column-parallel shard on the 2N dim so each device holds its interleaved slice."""
    import torch

    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    gk = gate_w.to(torch.bfloat16).T.contiguous()  # [K=dim, N=hidden]
    uk = up_w.to(torch.bfloat16).T.contiguous()
    packed = torch.cat([gk, uk], dim=-1)  # [dim, 2*hidden], gate first
    il = prepare_for_fused_swiglu(packed, ndev=tp, gate_is_first=True)  # [dim, 2*hidden]
    return ttnn.as_tensor(
        il,
        dtype=ttnn.bfloat4_b,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # TP: w1/w3 column-parallel (shard out dim), w2 row-parallel (shard in dim).
        # down_proj (w2) stays bfloat8_b -- an explicit ACCURACY choice (see the non-TP loader's
        # comment below). bfp4 down_proj is faster (decode matmuls are DRAM-bandwidth-bound, so a
        # narrower dtype is a real -45% there) but costs measurable end-to-end quality
        # (tests/perf/test_decode_weight_dtype_sweep.py has the numbers); not worth it.
        # DRAM-sharded memcfgs from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # w1/w3 DRAM-WIDTH_SHARDED for decode (M=1 tile, ~+10% tok/s); w2 interleaved.
        # Cache uses `.dramshard` suffix — layout incompatible with interleaved cache
        # (as_tensor ignores requested memcfg on reload). Fallback if memcfgs absent.
        # 1D-decode (default) uses interleaved weights (its mcast decode matmul needs them).
        dram_sharded = (
            args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not getattr(args, "mlp_1d_decode", False)
        )

        def cache(name, tag=""):
            return str(tensor_cache_path / f"mlp.{name}.weight{tag}.tp") if tensor_cache_path else None

        # Prefill-only packed [gate|up] AGMM weight (decode keeps w1/w3; extra DRAM ~w1+w3/layer).
        wgu = (
            _build_gate_up(
                state_dict["gate_proj.weight"],
                state_dict["up_proj.weight"],
                mesh_device,
                tp,
                cache("gate_up", ".swiglu"),
            )
            if tpc.mlp_gateup_agmm_enabled(tp)
            else None
        )

        if dram_sharded:
            return MLPWeights(
                w1=tpc.shard_w(
                    state_dict["gate_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w1_weight_memcfg,
                    cache_path=cache("gate_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w3=tpc.shard_w(
                    state_dict["up_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w3_weight_memcfg,
                    cache_path=cache("up_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w2=tpc.shard_w(
                    state_dict["down_proj.weight"],
                    mesh_device,
                    dim=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_path=cache("down_proj"),
                    dtype=ttnn.bfloat8_b,
                ),
                w_gate_up=wgu,
            )

        # Default: INTERLEAVED DRAM shards; ttnn.linear works for decode and prefill.
        return MLPWeights(
            w1=tpc.shard_w(
                state_dict["gate_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("gate_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w3=tpc.shard_w(
                state_dict["up_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("up_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w2=tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("down_proj"),
                dtype=ttnn.bfloat8_b,
            ),
            w_gate_up=wgu,
        )

    def load(name, dtype):
        t = state_dict[f"{name}.weight"].T.contiguous()  # [in, out] for ttnn.linear
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mlp.{name}.weight") if tensor_cache_path else None,
        )

    # gate/up: bfloat4_b (bandwidth); down: bfloat8_b (accuracy).
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat4_b),
        w2=load("down_proj", ttnn.bfloat8_b),
        w3=load("up_proj", ttnn.bfloat4_b),
    )


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        # 1D-decode (default): small-grid 1D matmuls beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny decode MLP matmuls (see test_mlp_matmul_sweep). Forces interleaved weights.
        self._mlp_1d_decode = args is not None and getattr(args, "mlp_1d_decode", False)
        # Match load_mlp_weights dram_sharded condition for layout consistency.
        self._dram_sharded = (
            self.num_devices > 1
            and args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not self._mlp_1d_decode
        )
        # Prefill fused-swiglu AGMM (ff_norm skips its AG; layer.py sets _fuse_ff_agmm to match).
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path, args=args)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # fuse_swiglu AGMM: fp32 acc (subblock_w=4) to match GDN/attn in-proj.
        self.compute_kernel_config_agmm = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # WH-only, gate/up decode matmuls ONLY (not down): MEASURED (N300, M=32 K=4096 N=6144,
        # test_mlp_decode_matmul_sweep.py, 3 independent runs each) fp32_dest_acc_en=False beats
        # compute_kernel_config_decode's fp32_dest_acc_en=True by ~3-4% on both gate and up,
        # reproducibly -- paired with mlp_w1_decode_1d_progcfg/mlp_w3_decode_1d_progcfg's num_cores=56
        # (also re-tuned there). Down-proj showed no such win (already at its optimal core count, and
        # fp32_dest_acc_en made an inconsistent <0.5% difference there) so it keeps
        # compute_kernel_config_decode unchanged.
        # SCOPED TO WORMHOLE 9B ON N300 (tpc.wh_9b_n300) -- and this MUST match the fp32_acc argument
        # model_config.py passes to mlp_w1/w3_decode_1d_progcfg, which is gated on the same helper.
        # Outside the scope this is compute_kernel_config_decode, i.e. the previously shipped config.
        self.compute_kernel_config_gateup_decode = (
            ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=False
            )
            if (args is not None and tpc.wh_9b_n300(args))
            else self.compute_kernel_config_decode
        )

    def forward(self, x):
        if self.num_devices > 1:
            return self._forward_tp(x)
        w = self.weights
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        down_pc = None
        if (
            T > 1
            and getattr(self.args, "prefill_progcfg", None) is not None
            and os.environ.get("QWEN9B_MLP_DOWN_AUTO") != "1"
        ):
            down_pc = self.args.prefill_progcfg(T, hidden.shape[-1], w.w2.shape[-1])
        output = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc, program_config=down_pc)
        ttnn.deallocate(hidden)
        return output

    def _forward_tp(self, x):
        """TP forward: replicated input; reduce-scatter output fractured on hidden dim."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        mc = ttnn.DRAM_MEMORY_CONFIG
        _silu_fused = False
        # 27B-on-Wormhole prefill matmul blocking (gate/up AND down; see both call sites below).
        # Hoisted out of the gate/up branch because the down-proj block reads it too, and that block
        # is also reached from Blackhole's fused-AGMM arm, which never enters the gate/up branch.
        # Length-capped: see tp_common.PREFILL_FULL_GRID_MAX_M for why (CBs scale with per_core_M and
        # overflow L1 past the production chunk length).
        _mlp_full_grid = args.dim > 4096 and not tpc.is_blackhole() and x.shape[-2] <= tpc.PREFILL_FULL_GRID_MAX_M
        # Prefill: x is K-sharded (ff_norm skipped AG); fused AG + [gate|up] + SwiGLU
        _fused_gu = self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None
        if _fused_gu:
            hidden = tpc.all_gather_swiglu_prefill(
                x, w.w_gate_up, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()
            )
            _silu_fused = True
        elif getattr(self, "_dram_sharded", False) and x.shape[-2] <= ttnn.TILE_SIZE:
            # DRAM-WIDTH_SHARDED w1/w3 decode (M=1 tile). Prefill uses fused AGMM above.
            x_sh = ttnn.to_memory_config(x, args.act_shard_hidden)
            w1_out = ttnn.linear(
                x_sh,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_sh,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_sh)
            # Keep gate/up in L1 for mul → w2 (avoid L1→DRAM→L1).
            w1_out = ttnn.to_memory_config(w1_out, ttnn.L1_MEMORY_CONFIG)
            w3_out = ttnn.to_memory_config(w3_out, ttnn.L1_MEMORY_CONFIG)
        elif self._mlp_1d_decode and x.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.
            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            w1_out = ttnn.linear(
                x_il,
                w.w1,
                compute_kernel_config=self.compute_kernel_config_gateup_decode,
                program_config=args.mlp_w1_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_il,
                w.w3,
                compute_kernel_config=self.compute_kernel_config_gateup_decode,
                program_config=args.mlp_w3_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_il)
            _silu_fused = True
        elif x.shape[-2] > ttnn.TILE_SIZE:
            # Prefill (M>1 tile, compute-bound): FPU-tuned 2D config (grid width -> 1x4 subblock,
            # in0_block_w=4) beats ttnn-auto's 1x1 stall ~2.7x (test_mlp_matmul_sweep_prefill). SILU fused.
            seq = x.shape[-2]
            # max_cols = device worker-grid width (11 on BH): wide grid (gate/up -> 9x10) vs old 8-wide.
            _gw = getattr(args, "decode_grid_w", 8)
            # TP-selected prefill tuning; absent (single-device 9B) => frozen TP=4 behavior.
            _pt = getattr(args, "prefill_tuning", None)
            # This elif (inside _forward_tp, so always TP>1) is only reached when the fused AGMM path is
            # unavailable (see _fused_gu above) — on Blackhole that never happens at TP>1 (fusion is
            # always on there), so this combination was never tuned for BH's L1 budget and is new,
            # WH-only territory: the default full-per_core_N output/intermediate CB (both gate AND up
            # held at once) overflowed WH's smaller, already-at-max-grid L1 by ~28KB (measured). Halve it
            # (halve_out_block, see tp_common.py) and route the output to DRAM (below) instead of L1 —
            # blocking alone doesn't help if the final tensor must still fully reside in L1.
            # One K pass + _CKC_MLP_KPASS1 on Wormhole; see the constant's definition for the
            # measurements and why halve_out_block was needed before.
            _kpass1 = not tpc.is_blackhole()
            _half = not _kpass1
            # WORMHOLE ONLY: with _CKC_MLP_KPASS1's fp32_dest_acc_en=False the output-subblock ceiling
            # is DST_TILES (8), not DST_TILES_FP32_ACC (4) -- but the shared factory defaults to the
            # conservative 4 (correct for its fp32-acc callers: attention wo on BH, GDN out-proj).
            # Raising it here was left undone when the MLP moved to one K pass: per_core_N is 24
            # (gate/up) and 16 (down), both divisible by 8, so out_block_w == per_core_N (one K pass)
            # stays legal at sub_w=8. Same lever create_prefill_kpass1_matmul_program_config already
            # pulls explicitly for the GDN in-projection.
            #
            # MEASURED (device kernel duration, N300, M=2048; tests/perf/test_mlp_subblock_actdtype_sweep.py),
            # in0=bf16, one K pass, sub_w 4 -> 8:
            #     gate  2048x4096x6144   989.9us -> 864.6us   -12.7%
            #     up    2048x4096x6144   900.4us -> 815.0us    -9.5%
            #     down  2048x6144x4096  1015.0us -> 924.8us    -8.9%
            # PCC is UNCHANGED to 5 decimals on all three (gate 0.99013, up 0.99313, down 0.99983) --
            # this is pure blocking, not an accuracy trade.
            _sub_cap = tpc.DST_TILES if _kpass1 else None
            # MODEL-GATED (27B on Wormhole). The 27B's per-device gate/up width is
            # hidden_dim/tp = 17408/8 = 2176 = 68 tiles, and 68's only divisors <=8 are 1/2/4 -- so
            # _best_prefill_cols, which picks the grid width that maximises the output subblock,
            # settles on 6 columns = 48 of 64 cores. That trade is wrong here: the missing 16 cores
            # cost far more than the wider subblock earns. The 9B's 192 tiles already land on the
            # full 8 wide, so it never saw this and is left byte-identical (gate on dim, not
            # model_name: HF_MODEL is often a hashed snapshot directory).
            #
            # MEASURED (T3K TP=8, device kernel duration, M=2048, in0 bf8;
            # tests/perf/test_mlp_prefill_matmul_sweep.py -- and both baselines below reproduce the
            # real layer's numbers to within 1.5us, which is why this isolated sweep is trusted for
            # this shape where mlp.py's older caution applied to a compute-config axis):
            #     gate/up  2048x5120x2176  cols=6 in0_bw=4 sub=1x6  468.5us  (48 cores, 40.7% FPU)
            #                              cols=8 in0_bw=8 sub=1x3  352.3us  (64 cores, 54.7% FPU)  -24.8%
            #     down     2048x2176x5120  cols=8 in0_bw=4 sub=1x5  442.8us  (64 cores, 44.5% FPU)
            #                              cols=8 in0_bw=4 sub=2x4  390.9us  (64 cores, 49.6% FPU)  -11.7%
            # PCC is unchanged (0.9989485 either way, test_mlp_tp_prefill[in_bf8]) -- pure blocking.
            # out_subblock_h is passed explicitly because no simple rule predicts both winners; see
            # create_prefill_mlp_matmul_program_config_full_grid.
            if _mlp_full_grid:
                pc_gate = tpc.create_prefill_mlp_matmul_program_config_full_grid(
                    seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU, out_subblock_h=1
                )
                pc_up = tpc.create_prefill_mlp_matmul_program_config_full_grid(
                    seq, args.dim, w.w3.shape[-1], out_subblock_h=1
                )
            else:
                pc_gate = tpc.create_prefill_mlp_matmul_program_config(
                    seq,
                    args.dim,
                    w.w1.shape[-1],
                    fused_activation=ttnn.UnaryOpType.SILU,
                    max_cols=_gw,
                    tuning=_pt,
                    halve_out_block=_half,
                    max_subblock_hw=_sub_cap,
                )
                pc_up = tpc.create_prefill_mlp_matmul_program_config(
                    seq,
                    args.dim,
                    w.w3.shape[-1],
                    max_cols=_gw,
                    tuning=_pt,
                    halve_out_block=_half,
                    max_subblock_hw=_sub_cap,
                )
            if _kpass1:
                ckc = _CKC_MLP_KPASS1
            # MODEL-GATED (27B on Wormhole, M <= PREFILL_FULL_GRID_MAX_M via _mlp_full_grid).
            # gate/up outputs -- and therefore the SwiGLU product below -- stay in L1.
            #
            # The note further down says L1 for `hidden` is a verified dead end. That verdict was
            # measured at 9B/N300 shapes and does NOT carry to the 27B at TP=8, because the tensor is
            # 5x smaller per core: [1,1,2048,hidden/tp] is 6144 cols of bf16 = 25.2MB = 393KB/core on
            # the 9B, but 2176 cols of bf8 = 4.7MB = 74KB/core here (TP=8 narrows it, and the bf8
            # gather above already halved the dtype). Peak L1 is the 3 live tensors across the
            # multiply -- gate + up + product = ~222KB/core -- which leaves the down-proj room for
            # its CBs where the 9B had none.
            #
            # MEASURED (T3K TP=8, 27B, seq 2048, device kernel duration, 2 runs each, DRAM -> L1):
            #     up    2048x5120x2176   302/301us -> 250/250us   -17%  (FLOPs 65.1% -> 78.5%)
            #     mul   [1,1,2048,2176]   80/80us  ->  42/42us    -47%  (pure bandwidth op)
            #     gate  2048x5120x2176   352/352us -> 350/350us   unchanged
            #     down  2048x2176x5120   392/393us -> 392/391us   unchanged (reads in0 from L1)
            #                                                     = -89us/layer
            # The two collectives move +10/-6us and +31/+4us across the same runs, i.e. CCL noise;
            # they are not affected by this.
            #
            # Why only `up` gains: gate carries the fused SILU, which costs ~100us at this shape
            # (visible in the DRAM baseline too, 352 vs 301 for an identical matmul), and that is
            # what it stays pinned by. `down` is unchanged because at 19-23% DRAM utilisation none of
            # these matmuls is bandwidth-bound -- the win is the elementwise multiply, which is, plus
            # up no longer contending with it for DRAM.
            _gu_mc = ttnn.L1_MEMORY_CONFIG if _mlp_full_grid else ttnn.DRAM_MEMORY_CONFIG
            w1_out = ttnn.linear(x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=_gu_mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=_gu_mc)
            _silu_fused = True
        else:
            # Interleaved weights: auto matmul program for decode and prefill.
            w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
            _silu_fused = True

        # gated activation (down-proj INPUT): L1 in decode, DRAM in prefill. The L1 win is OUTPUT-only;
        # keeping both down input (hidden) and output (partial) in L1 at seq 2048 overflows L1.
        _prefill_tuned = x.shape[-2] > ttnn.TILE_SIZE and _silu_fused
        # gate * up (skipped when _fused_gu already produced `hidden` with SwiGLU in-kernel).
        if not _fused_gu:
            mc_out = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= ttnn.TILE_SIZE else mc
            if _mlp_full_grid:
                mc_out = ttnn.L1_MEMORY_CONFIG  # see _gu_mc above for why this is safe on the 27B
            # MEASURED (N300, 2026-08): putting `hidden` in L1 on WH prefill does NOT work, and the
            # reason is not the one the note above gives. On WH the two are never both L1 anyway --
            # prefill_out_memory_config's 8MB budget already sends the [2048,4096] down-proj OUTPUT to
            # DRAM -- but `hidden` ALONE is too big: [1,2048,6144] bf16 = 25.2MB = ~393KB/core, and the
            # down-proj then cannot place its own circular buffers ("clash with L1 buffers ... L1 buffer
            # allocated at 1080192 and static circular buffer region ends at 1162432",
            # test_mlp_tp_prefill at T=2048). No smaller budget helps: 2048 IS the production
            # chunk-outer length, so anything that excludes it wins nothing. Keep DRAM.
            #
            # WORMHOLE PREFILL: emit `hidden` as bf8 while KEEPING IT IN DRAM. This is deliberately
            # the dtype axis and NOT the placement axis -- L1 is the known-dead end above, and staying
            # in DRAM means none of that CB-clash risk applies. Two wins for one change:
            #   * the multiply writes half as many bytes (it is pure bandwidth, no FLOPs)
            #   * the down-proj's in0 becomes bf8, turning it into a BFP8 x BFP8 matmul
            # MEASURED (device kernel duration, N300, T=2048):
            #   SwiGLU mul  [1,1,2048,6144]  out DRAM/bf16 424.0us -> out DRAM/bf8 348.6us  -17.8%
            #                                (tests/perf/test_layer_elementwise_l1_sweep.py, pcc 0.99996)
            #   down-proj   2048x6144x4096   in0 bf16 924.8us -> in0 bf8 758.5us  -18.0%
            #                                (tests/perf/test_mlp_subblock_actdtype_sweep.py, pcc 0.99978)
            # It also halves `hidden`'s DRAM footprint (25.2MB -> 12.6MB), easing pressure for
            # everything downstream. Only on the one-K-pass (_kpass1 / Wormhole) prefill arm, whose
            # numerics are already the loosest in the layer (LoFi + bfp4 gate/up weights) -- Blackhole's
            # fused-AGMM path produces `hidden` inside all_gather_swiglu_prefill and never reaches here.
            _hidden_dt = {"dtype": ttnn.bfloat8_b} if (_prefill_tuned and not tpc.is_blackhole()) else {}
            # Standalone silu only on DRAM-sharded decode path (SILU not fused there).
            if _silu_fused:
                hidden = ttnn.mul(w1_out, w3_out, memory_config=mc_out, **_hidden_dt)
                ttnn.deallocate(w1_out)
            else:
                w1_act = ttnn.silu(w1_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
                hidden = ttnn.mul(w1_act, w3_out, memory_config=mc_out, **_hidden_dt)
                ttnn.deallocate(w1_act)
            ttnn.deallocate(w3_out)
        # Prefill w2: 2D progcfg on (8,10); decode (M<=32) keeps ttnn-auto.
        w2_pc = None
        if self._mlp_1d_decode and hidden.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode down-proj on a small explicit grid (~16 cores).
            w2_pc = args.mlp_w2_decode_1d_progcfg
        elif hidden.shape[-2] > ttnn.TILE_SIZE:
            # Prefill down-proj: subblock-tuned 2D config with the wide grid (max_cols=device width),
            # off the generic 8-wide prefill_progcfg. Output L1 via mc_w2_out below.
            # max_subblock_hw keyed off the ACTUAL compute config rather than an arch check: the
            # raised DST_TILES ceiling is only legal with fp32_dest_acc_en=False, and `ckc` here is
            # _CKC_MLP_KPASS1 only on the Wormhole non-fused arm above (Blackhole's fused-AGMM path
            # leaves ckc at self.compute_kernel_config, which has fp32 dest acc ON -> cap stays 4).
            # Testing the object directly means this cannot silently desync if that branch changes.
            # MEASURED down 2048x6144x4096: 1015.0us -> 924.8us (-8.9%), PCC 0.99983 unchanged.
            #
            # MODEL-GATED (27B on Wormhole), same factory and the same reason as gate/up above --
            # except that here the grid width was ALREADY the full 8 (dim = 5120 = 160 tiles divides
            # cleanly), so the whole win is out_subblock_h=2: 2x4 uses all 8 DST tiles where the
            # derived 1x5 uses 5. MEASURED 2048x2176x5120: 442.8us -> 390.9us (-11.7%), PCC
            # unchanged. Note in0_block_w stays 4 here, not 8 -- K is 68 tiles and 68 % 8 != 0, so 8
            # is not a legal blocking; the factory's _largest_divisor_le picks 4 on its own.
            if _mlp_full_grid:
                w2_pc = tpc.create_prefill_mlp_matmul_program_config_full_grid(
                    hidden.shape[-2], hidden.shape[-1], w.w2.shape[-1], out_subblock_h=2
                )
            else:
                w2_pc = tpc.create_prefill_mlp_matmul_program_config(
                    hidden.shape[-2],
                    hidden.shape[-1],
                    w.w2.shape[-1],
                    max_cols=getattr(args, "decode_grid_w", 8),
                    max_subblock_hw=tpc.DST_TILES if ckc is _CKC_MLP_KPASS1 else None,
                )
        # down-proj OUTPUT in L1 for the tuned prefill path (DRAM input `hidden` + L1 output = the
        # validated sweep outL1 config; tt_all_reduce already consumes an L1 partial) — but only while
        # the [1,T,dim] output actually fits; see tp_common.prefill_out_memory_config for why WH has to
        # spill the long-chunk case to DRAM.
        mc_w2_out = (
            tpc.prefill_out_memory_config(x.shape[-2], w.w2.shape[-1])
            if (x.shape[-2] <= ttnn.TILE_SIZE or _prefill_tuned)
            else mc
        )
        partial = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc_w2_out, program_config=w2_pc)
        ttnn.deallocate(hidden)

        # tt_all_reduce on (1,4) mesh reduce-scatters to hidden dim (dim=3).
        # Prefill passes tuned chunks_per_sync / num_workers_per_link (see tp_common
        # prefill_ccl_tuning); decode keeps tt_all_reduce's defaults, which its own path was tuned at.
        # Gate on x.shape[-2] (seq/M), not `T` above: T is x.shape[1] and the activation is
        # [1, 1, S, dim] in both modes, so T is always 1 and this branch never fired. Attention
        # and GDN already pass prefill_ccl_tuning() unconditionally on their prefill RS.
        _ccl_kw = {}
        if x.shape[-2] > ttnn.TILE_SIZE:
            _cps, _wpl = tpc.prefill_ccl_tuning()
            _ccl_kw = {"chunks_per_sync": _cps, "num_workers_per_link": _wpl}
        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **_ccl_kw,
        )
        return out
