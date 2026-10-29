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
    w1: ttnn.Tensor  # gate_proj [in, out], bfloat8_b (bfloat4_b if QWEN36_MLP_GATEUP_BFP8=0, dense only)
    w2: ttnn.Tensor  # down_proj [in, out], bfloat8_b (bfloat4_b if QWEN36_BFP4_MLP_DOWN=1, dense only)
    w3: ttnn.Tensor  # up_proj [in, out], bfloat8_b (bfloat4_b if QWEN36_MLP_GATEUP_BFP8=0, dense only)
    w_gate_up: ttnn.Tensor = None  # TP prefill: tile-pair-interleaved packed [gate|up] for fused-swiglu AGMM
    w2_dec: ttnn.Tensor = None  # QWEN36_DSHARD_MC (down): DRAM-width-sharded decode copy of w2; None when off


def mlp_dsmc_targets(args, dense=True):
    """(gateup, down): which QWEN36_DSHARD_MC targets this MLP serves in decode. Dense TP MLP on the default 1D
    decode path only (the MoE shared expert and the legacy DRAM-sharded decode path keep their layouts)."""
    if not dense or args is None or getattr(args, "num_devices", 1) <= 1 or not getattr(args, "mlp_1d_decode", False):
        return False, False
    return (
        getattr(args, "mlp_w1_dsmc_progcfg", None) is not None,
        getattr(args, "mlp_w2_dsmc_progcfg", None) is not None,
    )


def _build_gate_up(gate_w, up_w, mesh, tp, cache_path, dtype=ttnn.bfloat4_b):
    """Packed [gate|up] weight for all_gather_swiglu_prefill: prepare_for_fused_swiglu tile-pair
    interleave, then column-parallel shard on the 2N dim so each device holds its interleaved slice."""
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
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def load_mlp_weights(
    mesh_device, state_dict, tensor_cache_path=None, args=None, use_gateup_agmm=True, down_dtype=None, gateup_dtype=None
) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights.

    down_dtype=None: dense MLP, w2 dtype from QWEN36_BFP4_MLP_DOWN (default bfloat8_b). The MoE shared expert
    passes ttnn.bfloat8_b explicitly."""
    from models.demos.blackhole.qwen36.tt import tp_common as _tpc

    _dsmc_gu, _dsmc_down = mlp_dsmc_targets(args, dense=down_dtype is None)
    down_dtype = _tpc.resolve_mlp_down_dtype(down_dtype)
    gateup_dtype = _tpc.resolve_mlp_gateup_dtype(gateup_dtype)
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
                gateup_dtype,
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
                    dtype=gateup_dtype,
                ),
                w3=tpc.shard_w(
                    state_dict["up_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w3_weight_memcfg,
                    cache_path=cache("up_proj", ".dramshard"),
                    dtype=gateup_dtype,
                ),
                w2=tpc.shard_w(
                    state_dict["down_proj.weight"],
                    mesh_device,
                    dim=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_path=cache("down_proj"),
                    dtype=down_dtype,
                ),
                w_gate_up=wgu,
            )

        # QWEN36_WEIGHT_PREFETCH (gateup): with the prefill AGMM weight present, w1/w3 are read by the 1D decode
        # matmul only, so they load receiver-contiguous for the Tensor prefetcher instead of interleaved (no extra
        # DRAM). None -> target off/unsupported -> the interleaved load below.
        from models.demos.blackhole.qwen36.tt import weight_prefetch

        _pf_gu = wgu is not None and getattr(args, "mlp_1d_decode", False)

        def gate_up(name, progcfg_name):
            w = (
                weight_prefetch.load_prefetched(
                    "gateup",
                    f"mlp.{name}",
                    mesh_device,
                    state_dict[f"{name}.weight"],
                    -1,
                    gateup_dtype,
                    cache(name),
                    getattr(args, progcfg_name, None),
                    tp,
                    replaces_primary=True,
                )
                if _pf_gu
                else None
            )
            if w is not None:
                return w
            return tpc.shard_w(
                state_dict[f"{name}.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache(name),
                dtype=gateup_dtype,
            )

        if _dsmc_gu:
            # QWEN36_DSHARD_MC (gateup): decode-only w1/w3 (prefill reads the packed AGMM weight), DRAM-width-sharded
            # for the multi-core DRAM-sharded pipeline. Distinct `.dsmc` cache (layout differs from the interleaved).
            tpc.dshard_mc_check_weight("gateup", gateup_dtype)
            if wgu is None:
                raise ValueError(
                    f"{tpc.DSHARD_MC_ENV} gateup needs the prefill fused-swiglu AGMM weight (prefill would otherwise "
                    "read the DRAM-sharded w1/w3)"
                )
            w1, w3 = (
                tpc.shard_w(
                    state_dict[f"{name}.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=memcfg,
                    cache_path=cache(name, ".dsmc"),
                    dtype=gateup_dtype,
                )
                for name, memcfg in (("gate_proj", args.mlp_w1_weight_memcfg), ("up_proj", args.mlp_w3_weight_memcfg))
            )
        else:
            # Default: INTERLEAVED DRAM shards (decode and prefill). Load order w1, w3, w2 as before.
            w1 = gate_up("gate_proj", "mlp_w1_decode_1d_progcfg")
            w3 = gate_up("up_proj", "mlp_w3_decode_1d_progcfg")
        if _dsmc_down:
            tpc.dshard_mc_check_weight("down", down_dtype)
        w2 = tpc.shard_w(
            state_dict["down_proj.weight"],
            mesh_device,
            dim=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=cache("down_proj"),
            dtype=down_dtype,
        )
        # QWEN36_WEIGHT_PREFETCH (down): prefill reads w2 too, so decode gets a receiver-contiguous copy.
        _w2_pc = getattr(args, "mlp_w2_decode_1d_progcfg", None) if getattr(args, "mlp_1d_decode", False) else None
        weight_prefetch.attach_twin(
            mesh_device,
            w2,
            weight_prefetch.load_prefetched(
                "down",
                "mlp.down_proj",
                mesh_device,
                state_dict["down_proj.weight"],
                0,
                down_dtype,
                cache("down_proj"),
                _w2_pc,
                tp,
            ),
            _w2_pc,
        )
        w2_dec = None
        if _dsmc_down:
            # QWEN36_DSHARD_MC (down): prefill keeps the interleaved w2; decode reads this DRAM-width-sharded copy.
            w2_dec = tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=args.mlp_w2_weight_memcfg,
                cache_path=cache("down_proj", ".dsmc"),
                dtype=down_dtype,
            )
        return MLPWeights(w1=w1, w2=w2, w3=w3, w_gate_up=wgu, w2_dec=w2_dec)

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

    # gate/up: bfloat8_b (QWEN36_MLP_GATEUP_BFP8=0 -> bfloat4_b); down: bfloat8_b (QWEN36_BFP4_MLP_DOWN=1 -> bfloat4_b).
    return MLPWeights(
        w1=load("gate_proj", gateup_dtype),
        w2=load("down_proj", down_dtype),
        w3=load("up_proj", gateup_dtype),
    )


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(
        self,
        mesh_device,
        state_dict,
        tensor_cache_path=None,
        args=None,
        tt_ccl=None,
        use_gateup_agmm=True,
        down_dtype=None,
        gateup_dtype=None,
    ):
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

        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and use_gateup_agmm
        # QWEN36_DSHARD_MC (gateup / down): decode on the multi-core DRAM-sharded pipeline (load_mlp_weights matches).
        self._dsmc_gu, self._dsmc_down = mlp_dsmc_targets(args, dense=down_dtype is None)
        self.weights = load_mlp_weights(
            mesh_device,
            state_dict,
            tensor_cache_path,
            args=args,
            use_gateup_agmm=use_gateup_agmm,
            down_dtype=down_dtype,
            gateup_dtype=gateup_dtype,
        )
        # Dense MLP: QWEN36_MLP_FIDELITY (default hifi2). MoE shared expert (down_dtype passed) stays LoFi.
        fid = tpc.mlp_fidelity(ttnn.MathFidelity.LoFi if down_dtype is not None else None)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # fuse_swiglu AGMM: fp32 acc (subblock_w=4) to match GDN/attn in-proj.
        self.compute_kernel_config_agmm = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x, mode=None, decode_ar=None):
        # mode is unused (accepted only for a uniform signature with Qwen36MoE, which needs an
        # explicit decode/prefill mode); the dense MLP still infers its path from the input shape.
        # decode_ar: tp_common.DecodeResidualAllReduce (QWEN36_DECODE_ALLREDUCE=1, decode only; TP only).
        if self.num_devices > 1:
            return self._forward_tp(x, decode_ar=decode_ar)
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

    def _forward_tp(self, x, decode_ar=None):
        """TP forward: replicated input; reduce-scatter output fractured on hidden dim
        (decode_ar: all-reduce -> replicated RES40 output instead)."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.demos.blackhole.qwen36.tt import weight_prefetch
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        if (self._dsmc_gu or self._dsmc_down) and x.shape[-2] <= ttnn.TILE_SIZE:
            return self._forward_tp_dsmc_decode(x, ckc, decode_ar)

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
            # weight_prefetch.linear == ttnn.linear unless QWEN36_WEIGHT_PREFETCH registered the weight.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            w1_out = weight_prefetch.linear(
                x_il,
                w.w1,
                args.mlp_w1_decode_1d_progcfg,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = weight_prefetch.linear(
                x_il,
                w.w3,
                args.mlp_w3_decode_1d_progcfg,
                compute_kernel_config=ckc,
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
                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=tpc.l1_or_dram(seq)
            )
            w3_out = ttnn.linear(
                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=tpc.l1_or_dram(seq)
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
        mc_w2_out = (
            ttnn.L1_MEMORY_CONFIG
            if (x.shape[-2] <= ttnn.TILE_SIZE or (_prefill_tuned and tpc.prefill_l1_rows_ok(x.shape[-2])))
            else mc
        )
        partial = weight_prefetch.linear(hidden, w.w2, w2_pc, compute_kernel_config=ckc, memory_config=mc_w2_out)
        ttnn.deallocate(hidden)

        if decode_ar is not None:
            return decode_ar.all_reduce(partial)

        # tt_all_reduce on (1,4) mesh reduce-scatters to hidden dim (dim=3).
        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=tpc.prefill_rs_links(partial.shape[-2]),
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def _forward_tp_dsmc_decode(self, x, ckc, decode_ar):
        """QWEN36_DSHARD_MC decode (M <= 1 tile): gate/up and/or down on the multi-core DRAM-sharded pipeline; a
        target that is off keeps today's 1D decode matmul. The pipeline's output is L1 width-sharded on
        ceil(N / per_core_N) cores (gate/up 34 x 2 tiles, down 40 x 4), which feeds the next matmul as is."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.demos.blackhole.qwen36.tt import weight_prefetch
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        ws = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

        def width_sharded(t):
            return t.is_sharded() and t.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED

        if self._dsmc_gu:
            # The decode ff_norm output is already act_shard_hidden (32 cores x 5 tiles): no interleave.
            x_sh = x if width_sharded(x) else ttnn.to_memory_config(x, args.act_shard_hidden)
            g = ttnn.linear(
                x_sh, w.w1, program_config=args.mlp_w1_dsmc_progcfg, compute_kernel_config=ckc, memory_config=ws
            )
            u = ttnn.linear(
                x_sh, w.w3, program_config=args.mlp_w3_dsmc_progcfg, compute_kernel_config=ckc, memory_config=ws
            )
            if x_sh is not x:
                ttnn.deallocate(x_sh)
            # SiLU on the mul input: a fused activation would drop the matmul back to the single-reader program.
            hidden = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=g.memory_config())
            ttnn.deallocate(g)
            ttnn.deallocate(u)
        else:
            # Today's 1D decode gate/up (silu fused in the w1 progcfg), as _forward_tp.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            w1_out = weight_prefetch.linear(
                x_il,
                w.w1,
                args.mlp_w1_decode_1d_progcfg,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = weight_prefetch.linear(
                x_il,
                w.w3,
                args.mlp_w3_decode_1d_progcfg,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_il)
            hidden = ttnn.mul(w1_out, w3_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(w1_out)
            ttnn.deallocate(w3_out)

        if self._dsmc_down:
            h_sh = hidden if width_sharded(hidden) else ttnn.to_memory_config(hidden, args.mlp_w2_dsmc_act_memcfg)
            partial = ttnn.linear(
                h_sh, w.w2_dec, program_config=args.mlp_w2_dsmc_progcfg, compute_kernel_config=ckc, memory_config=ws
            )
            if h_sh is not hidden:
                ttnn.deallocate(h_sh)
        else:
            # 1D mcast decode down-proj needs an interleaved in0.
            h_il = ttnn.to_memory_config(hidden, ttnn.L1_MEMORY_CONFIG) if hidden.is_sharded() else hidden
            partial = weight_prefetch.linear(
                h_il,
                w.w2,
                args.mlp_w2_decode_1d_progcfg,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            if h_il is not hidden:
                ttnn.deallocate(h_il)
        ttnn.deallocate(hidden)

        if decode_ar is not None:
            # all_reduce reshards the partial into RES40 (to_memory_config) as it does today.
            return decode_ar.all_reduce(partial)
        if partial.is_sharded():
            p_il = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(partial)
            partial = p_il
        return tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=tpc.prefill_rs_links(partial.shape[-2]),
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
