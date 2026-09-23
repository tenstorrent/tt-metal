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
from models.demos.blackhole.qwen36.tt import tp_common as tpc


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj [in, out], bfloat8_b
    w3: ttnn.Tensor  # up_proj [in, out], bfloat4_b
    w_gate_up: ttnn.Tensor = None  # TP prefill: tile-pair-interleaved packed [gate|up] for fused-swiglu AGMM


def _build_gate_up(gate_w, up_w, mesh, tp, cache_path):
    """Packed [gate|up] weight: prepare_for_fused_swiglu tile-pair interleave, then (tp>1)
    column-parallel shard on the 2N dim so each device holds its interleaved slice.

    tp=1 (single-device prefill fused-swiglu minimal_matmul) skips the mesh_mapper: the
    single-device caller's `mesh` is a plain ttnn.Device (ttnn.CreateDevice), not a MeshDevice,
    and shard_tensor_to_mesh_mapper requires a MeshDevice."""
    import torch

    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    def pack(gate):
        # Runs only on a tensor-cache miss (as_tensor preprocess), so a cached load never reads
        # the checkpoint tensors.
        gk = gate.to(torch.bfloat16).T.contiguous()  # [K=dim, N=hidden]
        uk = up_w.to(torch.bfloat16).T.contiguous()
        packed = torch.cat([gk, uk], dim=-1)  # [dim, 2*hidden], gate first
        return prepare_for_fused_swiglu(packed, ndev=tp, gate_is_first=True)  # [dim, 2*hidden]

    return ttnn.as_tensor(
        gate_w,
        preprocess=pack,
        dtype=ttnn.bfloat4_b,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if tp > 1 else None,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None, use_gateup_agmm=True) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # TP: w1/w3 column-parallel (shard out dim), w2 row-parallel (shard in dim).
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
            if tpc.mlp_gateup_agmm_enabled(tp) and use_gateup_agmm
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
        return ttnn.as_tensor(
            state_dict[f"{name}.weight"],
            preprocess=lambda t: t.T.contiguous(),  # [in, out] for ttnn.linear; cache-miss only
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mlp.{name}.weight") if tensor_cache_path else None,
        )

    # Prefill-only packed [gate|up] weight for the single-device fused-swiglu minimal_matmul
    # (T>512; see Qwen36MLP.forward). w1/w3 stay for decode and T<=512 prefill. New cache name --
    # tile-pair-interleaved layout is incompatible with the plain w1/w3 cache.
    wgu = (
        _build_gate_up(
            state_dict["gate_proj.weight"],
            state_dict["up_proj.weight"],
            mesh_device,
            1,
            str(tensor_cache_path / "mlp.w_gate_up_swiglu_1dev.weight") if tensor_cache_path else None,
        )
        if use_gateup_agmm
        else None
    )

    # gate/up: bfloat4_b (bandwidth); down: bfloat8_b (accuracy).
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat4_b),
        w2=load("down_proj", ttnn.bfloat8_b),
        w3=load("up_proj", ttnn.bfloat4_b),
        w_gate_up=wgu,
    )


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None, use_gateup_agmm=True):
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
        self._prefill_progcfg_fn = tpc.make_prefill_progcfg_fn(mesh_device) if self.num_devices == 1 else None
        # Decode (T==1) 1D matmul progcfg (see tp_common measured table); single-device only.
        self._decode_progcfg_fn = tpc.make_decode_progcfg_fn(mesh_device) if self.num_devices == 1 else None
        # minimal_matmul worker grid (single device only; see _prefill_matmul).
        self._mm_grid = None
        if self.num_devices == 1:
            g = mesh_device.compute_with_storage_grid_size()
            self._mm_grid = ttnn.CoreCoord(g.x, g.y)
        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and use_gateup_agmm
        self.weights = load_mlp_weights(
            mesh_device, state_dict, tensor_cache_path, args=args, use_gateup_agmm=use_gateup_agmm
        )
        # step2 (2026-09-22): routed through tpc.prefill_matmul_ckc() -- see QWEN36_PREFILL_MM_*
        # flags in tp_common.py. Legacy values (packer_l1_acc=False, fp32_dest_acc_en=True) unless
        # overridden.
        self.compute_kernel_config = tpc.prefill_matmul_ckc()
        # fuse_swiglu AGMM: fp32 acc (subblock_w=4) to match GDN/attn in-proj.
        self.compute_kernel_config_agmm = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def _prefill_progcfg(self, T, x, w, env_flag):
        """Swept 2D progcfg for a DRAM-output prefill matmul (single device); None keeps auto-config."""
        if self._prefill_progcfg_fn is None or os.environ.get(env_flag) == "1":
            return None
        fp32_acc = getattr(self.compute_kernel_config, "fp32_dest_acc_en", True)
        return self._prefill_progcfg_fn(T, x.shape[-1], w.shape[-1], x.dtype, w.dtype, fp32_acc)

    # Measured on P150, bit-identical vs auto: w2 down (K=6144,N=2048) minimal default config wins,
    # 0.241/0.433ms (T=2048/4096) vs swept-2D 0.311/0.546ms. w1/w3 up (K=2048,N=6144): swept 2D wins
    # at T=2048 (0.308ms) but the picker falls back to in0_block_w=2 at T=4096 (1.167ms), where a
    # swept MinimalMatmulConfig(8,8,8) gets 0.577ms -> use minimal only when the 2D pick is weak.
    def _prefill_matmul(self, x, w, T, env_flag, activation=None, memory_config=None):
        """T>512 up/down-proj matmul: swept 2D progcfg, or minimal_matmul when it wins (see above).

        memory_config: output placement override (default None -> DRAM, prior behavior). Used by the
        QWEN36_MLP_L1_OUT A/B (see forward()) to try an L1-resident down-proj output at T<=1024.
        """
        out_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        if os.environ.get(env_flag) == "1":
            # Escape hatch: plain ttnn-auto config, bypassing both the picker and minimal_matmul.
            return ttnn.linear(
                x,
                w,
                activation=activation,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=out_mc,
            )
        pc = self._prefill_progcfg(T, x, w, env_flag)
        use_minimal = os.environ.get("QWEN36_MLP_MINIMAL_MM", "1") != "0" and (
            x.shape[-1] > w.shape[-1] or pc is None or pc.in0_block_w < 4
        )
        if use_minimal:
            config = (
                ttnn.MinimalMatmulConfig(
                    M_block_size=8, K_block_size=8, N_block_size=8, compute_with_storage_grid_size=self._mm_grid
                )
                if x.shape[-1] <= w.shape[-1]
                else None  # down-proj: default minimal_matmul config measured best
            )
            fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation == "silu" else None
            return ttnn.experimental.minimal_matmul(
                x,
                w,
                fused_activation=fused_activation,
                config=config,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=out_mc,
            )
        return ttnn.linear(
            x,
            w,
            activation=activation,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=out_mc,
            program_config=pc,
        )

    def forward(self, x, mode=None):
        # mode is unused (accepted only for a uniform signature with Qwen36MoE, which needs an
        # explicit decode/prefill mode); the dense MLP still infers its path from the input shape.
        if self.num_devices > 1:
            return self._forward_tp(x)
        w = self.weights
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        # Fused path (fused SwiGLU minimal_matmul + _prefill_matmul down-proj) is used for ALL
        # prefill T (T>1): at T<=512 the old else-branch's L1-interleaved outputs made ttnn's
        # matmul auto-config fall back to the naive MatmulMultiCoreProgramConfig kernel (930us/
        # layer down-proj measured at T=512 vs 43us on the fused path; matmul_program_config.cpp:1183).
        # QWEN36_MLP_LEGACY_SHORT=1 restores the old T<=512 else-branch for A/B testing.
        legacy_short = T <= 512 and os.environ.get("QWEN36_MLP_LEGACY_SHORT") == "1"
        # Step1 task F10A (G4): L1-resident fused-swiglu output + down-proj output, now default ON
        # and raised to T<=2048 (the production chunk size) -- was default OFF, capped at T<=1024.
        # QWEN36_MLP_L1_OUT=0 restores the pre-F10A default (off at every T) exactly.
        l1_out_ab = T <= 2048 and os.environ.get("QWEN36_MLP_L1_OUT", "1") != "0"
        if T > 1 and not legacy_short:
            # Fused single-device SwiGLU: one minimal_matmul(fuse_swiglu=True) replaces w1(silu)+w3+mul.
            # Weight pairing: prepare_for_fused_swiglu(cat([w1_KN, w3_KN], -1), ndev=1, gate_is_first=True)
            # (PCC 0.999995 vs silu(x@w1)*(x@w3)). P150: 0.615ms vs 0.70ms at T=2048 (config=None), and
            # 1.08ms vs 1.7ms at T=4096 (MinimalMatmulConfig(8,8,8) on the device grid) -- both measured.
            use_fused_swiglu = (
                w.w_gate_up is not None
                and os.environ.get("QWEN36_MLP_FUSED_SWIGLU", "1") != "0"
                and os.environ.get("QWEN9B_MLP_UP_AUTO") != "1"
            )
            if use_fused_swiglu:
                # step2 (2026-09-22): QWEN36_PREFILL_MINIMAL_CFG/_FP32_ACC (tp_common.py) gate the
                # explicit config on both branches. Legacy (MINIMAL_CFG=0) keeps the exact T>=4096
                # MinimalMatmulConfig(8,8,8) at its binding-default (1,1) subblock, and config=None
                # below 4096 -- both bit-identical to the pre-flag code.
                if T >= 4096:
                    if tpc.PREFILL_MINIMAL_CFG:
                        cfg = (
                            ttnn.MinimalMatmulConfig(
                                M_block_size=8,
                                K_block_size=8,
                                N_block_size=8,
                                subblock_h=2,
                                subblock_w=2,
                                compute_with_storage_grid_size=self._mm_grid,
                            )
                            if tpc.PREFILL_MM_FP32_ACC
                            else ttnn.MinimalMatmulConfig(
                                M_block_size=4,
                                K_block_size=8,
                                N_block_size=16,
                                subblock_h=2,
                                subblock_w=4,
                                compute_with_storage_grid_size=self._mm_grid,
                            )
                        )
                    else:
                        cfg = ttnn.MinimalMatmulConfig(
                            M_block_size=8, K_block_size=8, N_block_size=8, compute_with_storage_grid_size=self._mm_grid
                        )
                else:
                    cfg = tpc.prefill_minimal_matmul_config(T, x.shape[-1], w.w_gate_up.shape[-1], self._mm_grid)
                hidden = ttnn.experimental.minimal_matmul(
                    x,
                    w.w_gate_up,
                    fuse_swiglu=True,
                    config=cfg,
                    compute_kernel_config=self.compute_kernel_config,
                    memory_config=ttnn.L1_MEMORY_CONFIG if l1_out_ab else ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                w1_out = self._prefill_matmul(x, w.w1, T, "QWEN9B_MLP_UP_AUTO", activation="silu")
                w3_out = self._prefill_matmul(x, w.w3, T, "QWEN9B_MLP_UP_AUTO")
                hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
                ttnn.deallocate(w1_out)
                ttnn.deallocate(w3_out)
        else:
            up_pc = (
                self._decode_progcfg_fn(x.shape[-1], w.w1.shape[-1])
                if (T == 1 and self._decode_progcfg_fn is not None)
                else None
            )
            w1_out = ttnn.linear(
                x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc, program_config=up_pc
            )
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc, program_config=up_pc)
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
        elif T == 1 and self._decode_progcfg_fn is not None:
            down_pc = self._decode_progcfg_fn(hidden.shape[-1], w.w2.shape[-1])
        if down_pc is not None:
            output = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc, program_config=down_pc)
        elif T > 1 and not legacy_short:
            output = self._prefill_matmul(
                hidden,
                w.w2,
                T,
                "QWEN9B_MLP_DOWN_AUTO",
                memory_config=ttnn.L1_MEMORY_CONFIG if l1_out_ab else None,
            )
        else:
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
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_il,
                w.w3,
                compute_kernel_config=ckc,
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
            pc_gate = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU, max_cols=_gw, tuning=_pt
            )
            pc_up = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w3.shape[-1], max_cols=_gw, tuning=_pt
            )
            # L1 output (gate/up outputs; down output via mc_out below): +FPU, avoids the DRAM round-trip
            # (test_mlp_matmul_sweep_prefill *_outL1). The [seq,N] tensors fit L1 at the prefill chunk.
            w1_out = ttnn.linear(
                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            w3_out = ttnn.linear(
                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=ttnn.L1_MEMORY_CONFIG
            )
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
            # Standalone silu only on DRAM-sharded decode path (SILU not fused there).
            if _silu_fused:
                hidden = ttnn.mul(w1_out, w3_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
            else:
                w1_act = ttnn.silu(w1_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
                hidden = ttnn.mul(w1_act, w3_out, memory_config=mc_out)
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
            w2_pc = tpc.create_prefill_mlp_matmul_program_config(
                hidden.shape[-2],
                hidden.shape[-1],
                w.w2.shape[-1],
                max_cols=getattr(args, "decode_grid_w", 8),
                tuning=getattr(args, "prefill_tuning", None),
            )
        # down-proj OUTPUT in L1 for the tuned prefill path (DRAM input `hidden` + L1 output = the
        # validated sweep outL1 config; tt_all_reduce already consumes an L1 partial).
        mc_w2_out = ttnn.L1_MEMORY_CONFIG if (x.shape[-2] <= ttnn.TILE_SIZE or _prefill_tuned) else mc
        partial = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc_w2_out, program_config=w2_pc)
        ttnn.deallocate(hidden)

        # tt_all_reduce on (1,4) mesh reduce-scatters to hidden dim (dim=3).
        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out
