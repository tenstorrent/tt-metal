# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import os

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


def pg_batched_attn(pg):
    """Grouped prefill attention batched over the rows (cos_cat / sin_cat given), unless QWEN36_GROUP_ATTN_PER_ROW=1."""
    import os

    return pg.get("cos_cat") is not None and os.environ.get("QWEN36_GROUP_ATTN_PER_ROW", "0") != "1"


def tokpar_enabled():
    """QWEN36_TOKPAR=1: token-parallel residual stream for TP prefill (forward_tokpar)."""
    import os

    return os.environ.get("QWEN36_TOKPAR", "0") == "1"


def tokpar_out_enabled():
    """QWEN36_TOKPAR_OUT=1 (with QWEN36_TOKPAR=1): the mixer out-projection (GDN out_proj / attention o_proj) also runs
    token-parallel with the full weight: the gated pre-projection activation is all-to-all'ed instead of the AGMM /
    matmul + reduce-scatter."""
    import os

    return os.environ.get("QWEN36_TOKPAR_OUT", "0") == "1"


def tokpar_a2a_bf8():
    """QWEN36_TOKPAR_A2A_BF8=1: the two mixer all-to-alls of forward_tokpar carry bf8 (their consumers cast to bf8)."""
    import os

    return os.environ.get("QWEN36_TOKPAR_A2A_BF8", "0") == "1"


def tokpar_a2a(x, to_tokens):
    """Hidden-sharded [1,1,S,dim/tp] <-> token-sharded [1,1,S/tp,dim] over the (1, tp) TP axis (all_to_all)."""
    in_dim, out_dim = (3, 2) if to_tokens else (2, 3)
    return ttnn.experimental.all_to_all_async_generic(
        x,
        in_dim=in_dim,
        out_dim=out_dim,
        num_links=2,
        topology=ttnn.Topology.Ring,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _load_tokpar_mlp(mesh, mlp_state, cache_dir):
    """Full (unsharded) gate / up / down weights, bf8, replicated on every chip, [in, out] for ttnn.linear."""
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def load(name):
        return ttnn.as_tensor(
            mlp_state[f"{name}.weight"],
            preprocess=lambda t: t.T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(cache_dir / f"mlp.{name}.weight.tokpar_rep") if cache_dir else None,
        )

    return {"w1": load("gate_proj"), "w3": load("up_proj"), "w2": load("down_proj")}


def _tokpar_pc(m, k, n, bw, act=None):
    """2D mcast prefill config on the full 12x10 grid: per_core_M = M tiles / 10 rows, per_core_N = N tiles / 12."""
    import math

    gx, gy = 12, 10
    mt, kt, nt = m // 32, k // 32, n // 32
    pm, pn = math.ceil(mt / gy), math.ceil(nt / gx)
    sw = max(d for d in (4, 3, 2, 1) if pn % d == 0)
    while kt % bw:
        bw -= 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=sw,
        out_block_h=pm,
        out_block_w=pn,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=act,
        fuse_batch=False,
    )


class _FusedDistRMSNorm(RMSNorm):
    """RMSNorm whose PREFILL distributed path is the single fused dit_fused_distributed_rmsnorm op
    (args.prefill_fused_dist_norm, see tpc.fused_distributed_rmsnorm). Decode keeps the replicated-weight rms_norm."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The fused op wants gamma TILE [1,1,dim] sharded on dim 2 (weight_distributed is ROW_MAJOR [1,1,dim/32,32]).
        dim, device, cache = kwargs["dim"], kwargs["device"], kwargs.get("weight_cache_path")
        name = f"{kwargs['state_dict_prefix']}{kwargs['weight_key']}.weight"
        self.weight_fused = ttnn.as_tensor(
            kwargs["state_dict"][name].unsqueeze(0).view(1, 1, dim) + 1.0,  # same +1 as the base weight
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=None if cache is None else cache / (name + "_fused_tile"),
            mesh_mapper=tpc.tp_shard_mapper(device, 2),
        )

    def _distributed_rmsnorm(
        self,
        inp,
        epsilon=None,
        weight=None,
        program_config=None,
        memory_config=None,
        compute_kernel_config=None,
        output_memory_config=None,
    ):
        assert program_config is None and memory_config is None, "Distributed RMSNorm does not support sharded I/O"
        mc = output_memory_config if output_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        shape = tuple(inp.shape)
        x = inp if len(shape) == 4 else ttnn.reshape(inp, (1,) * (4 - len(shape)) + shape)  # the op needs rank 4
        out = tpc.fused_distributed_rmsnorm(
            x, self.weight_fused, epsilon, self.tt_ccl, self.ccl_topology, compute_kernel_config, mc
        )
        return out if len(shape) == 4 else ttnn.reshape(out, shape)


class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        #
        # Single device: plain RMSNorm on the full hidden state (validated path).
        # TP (27B on a (1,4) mesh): the residual stream is fractured along the
        # hidden dim, so each norm is wrapped in the framework DistributedNorm,
        # which all-gathers (PREFILL: distributed rmsnorm + gather; DECODE:
        # gather-then-norm) to hand the modules a replicated full-dim input —
        # exactly as models/demos/qwen35_27b does via the framework decoder.
        # Prefill fuses the norm all-gather into the in-proj matmul (all_gather_minimal_matmul_async):
        # GDN qkvzab and full-attn QKV. attention_norm then skips its post-norm AG (prefill only;
        # decode gathers pre-norm). Gates must match the module-side _fuse_agmm gates.
        self._fuse_norm_agmm = self.num_devices > 1 and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        # QWEN36_DECODE_ALLREDUCE=1 (decode only): replicated 40-core L1 residual + one all_reduce_async per
        # row-parallel output (see tp_common.DecodeResidualAllReduce). None when off/unsupported; the shared
        # persistent buffer is allocated here, at construction, never inside a trace. Prefill never uses it.
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        self._decode_ar = _tpc.get_decode_allreduce(args, tt_ccl) if self.num_devices > 1 else None
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # MoE layers gather in the norm (the sparse MoE + shared expert need full/replicated
        # hidden and do NOT run the fused gate/up AGMM), so only fuse for the dense MLP.
        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and not args.is_moe_layer(layer_num)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )

        if self.num_devices > 1:
            # Tensor-parallel modules (sharded weights from the raw substate).
            # Cache the sharded mesh weights to disk so re-runs skip the (slow,
            # single-threaded) reorder+shard of the full 27B.
            tp_cache = (tensor_cache_path / f"layers.{layer_num}" / "tp") if tensor_cache_path else None
            if self.is_full_attention:
                from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp

                tw = load_attention_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.self_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPAttention(mesh_device, args, tw, tt_ccl)
            else:
                from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp

                tw = load_gdn_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.linear_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        elif self.is_full_attention:
            attn_state = substate(state_dict, f"layers.{layer_num}.self_attn")
            attn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedAttention(mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache)
        else:
            gdn_state = substate(state_dict, f"layers.{layer_num}.linear_attn")
            gdn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedDeltaNet(mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache)

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        if args.is_moe_layer(layer_num):
            # Sparse MoE MLP (Qwen3.5-MoE). Qwen36MoE.forward(x) keeps the same
            # single-in/single-out signature + fractured-hidden output as Qwen36MLP,
            # so the forward below and the model/trace-capture loops are unchanged.
            from models.demos.blackhole.qwen36.tt.moe import MoEConfig, Qwen36MoE

            self.feed_forward = Qwen36MoE(
                mesh_device, MoEConfig.from_args(args), mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl
            )
        else:
            self.feed_forward = Qwen36MLP(mesh_device, mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl)
        # QWEN36_TOKPAR=1 (prefill, TP meshes, dense MLP): token-parallel residual stream. Full MLP weights replicated on
        # every chip; see forward_tokpar.
        self._tokpar = None
        if tokpar_enabled() and self.num_devices > 1 and not args.is_moe_layer(layer_num):
            self._tokpar = _load_tokpar_mlp(mesh_device, mlp_state, mlp_cache)
            if tokpar_out_enabled():
                key = "self_attn.o_proj.weight" if self.is_full_attention else "linear_attn.out_proj.weight"
                self._tokpar["wo"] = ttnn.as_tensor(
                    state_dict[f"layers.{layer_num}.{key}"],
                    preprocess=lambda t: t.T.contiguous(),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=(mlp_cache / f"{key}.tokpar_rep") if mlp_cache else None,
                )

    def _make_norm(
        self,
        mesh_device,
        args,
        state_dict,
        layer_num,
        weight_key,
        tensor_cache_path,
        tt_ccl,
        ag_key,
        enable_all_gather=True,
    ):
        """Build the per-layer RMSNorm; wrap in DistributedNorm when TP>1.

        On a single device this returns the same plain RMSNorm the validated 9B
        path used. The DistributedNorm wrapper (TP>1) mirrors tt_transformers
        decoder.py and handles the fractured->replicated transition.
        """
        fused = getattr(args, "prefill_fused_dist_norm", False) and self.num_devices > 1
        # RMSNorm's distributed path gathers its statistics with no cluster_axis, i.e. over the whole mesh; on a
        # 2D (rows, tp) mesh that mixes the rows' tokens, so only the fused norm (TP-axis gather) is correct there.
        if tpc.is_2d_mesh(mesh_device) and not fused:
            raise ValueError("2D meshes need the fused distributed RMSNorm (QWEN36_FUSED_DIST_NORM=1)")
        norm_cls = _FusedDistRMSNorm if fused else RMSNorm
        norm = norm_cls(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(
                norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key, enable_all_gather=enable_all_gather
            )
        return norm

    def _tokpar_norm(self, x, norm, dtype=None):
        """Local rms_norm of this chip's [1,1,S/tp,dim] rows -> DRAM interleaved (dtype: optional output dtype, e.g.
        bf8 for the consumers that cast anyway). The interleaved norm parallelizes over tile rows only (21 cores at
        S/tp=672); QWEN36_TOKPAR_NORM_SHARD=1 block-shards the rows over gy x 8 cores (gy = the largest divisor of the
        tile rows <= 10, >= 3), norms there and writes back interleaved in `dtype` (one op for the cast)."""
        import os

        M, D = int(x.shape[-2]), int(x.shape[-1])
        mt, dt_ = M // 32, D // 32
        gy = max((d for d in range(1, 11) if mt % d == 0), default=1)
        gx = 8
        if os.environ.get("QWEN36_TOKPAR_NORM_SHARD", "0") == "1" and gy >= 3 and dt_ % gx == 0:
            key = (M, D, gy, gx)
            cache = self.__dict__.setdefault("_tokpar_norm_cfg", {})
            if key not in cache:
                sh = ttnn.create_sharded_memory_config(
                    (M // gy, D // gx),
                    core_grid=ttnn.CoreGrid(y=gy, x=gx),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                bw = dt_ // gx
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    subblock_w=max(d for d in (4, 2, 1) if bw % d == 0),
                    block_h=mt // gy,
                    block_w=bw,
                    inplace=False,
                )
                cache[key] = (sh, pc)
            sh, pc = cache[key]
            xs = ttnn.to_memory_config(x, sh)
            n = ttnn.rms_norm(
                xs,
                epsilon=norm.norm.eps,
                weight=norm.norm.weight,
                program_config=pc,
                compute_kernel_config=norm.norm.compute_kernel_config_hifi2,
                memory_config=sh,
            )
            ttnn.deallocate(xs)
            out = ttnn.sharded_to_interleaved(
                n, ttnn.DRAM_MEMORY_CONFIG, **({"output_dtype": dtype} if dtype is not None else {})
            )
            ttnn.deallocate(n)
            return out
        out = ttnn.rms_norm(
            x,
            epsilon=norm.norm.eps,
            weight=norm.norm.weight,
            compute_kernel_config=norm.norm.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if dtype is not None and out.dtype != dtype:
            o2 = ttnn.typecast(out, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out)
            out = o2
        return out

    def _tokpar_mlp(self, x):
        """Token-parallel SwiGLU MLP on this chip's [1,1,S/tp,dim] rows with the full weights (no CCL)."""
        import os

        w = self._tokpar
        M, K = x.shape[-2], x.shape[-1]
        F = w["w1"].shape[-1]
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=tpc.mlp_fidelity(getattr(self.args, "mlp_math_fidelity", None)),
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        dram = ttnn.DRAM_MEMORY_CONFIG
        x8 = x
        if os.environ.get("QWEN36_TOKPAR_MLP_BF8", "1") == "1" and x.dtype != ttnn.bfloat8_b:
            x8 = ttnn.typecast(x, ttnn.bfloat8_b, memory_config=dram)
        # QWEN36_TOKPAR_MLP_L1=1: gate / up / hidden in L1 (interleaved); QWEN36_TOKPAR_SILU_MUL=1: SiLU applied by the
        # multiply (input activation) instead of the gate matmul's packer
        act_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("QWEN36_TOKPAR_MLP_L1", "0") == "1" else dram
        silu_mul = os.environ.get("QWEN36_TOKPAR_SILU_MUL", "0") == "1"
        gate = ttnn.linear(
            x8,
            w["w1"],
            compute_kernel_config=ck,
            memory_config=act_mc,
            program_config=_tokpar_pc(M, K, F, 8, act=None if silu_mul else ttnn.UnaryOpType.SILU),
        )
        up = ttnn.linear(
            x8, w["w3"], compute_kernel_config=ck, memory_config=act_mc, program_config=_tokpar_pc(M, K, F, 8)
        )
        if x8 is not x:
            ttnn.deallocate(x8)
        if silu_mul:
            hidden = ttnn.mul(
                gate,
                up,
                memory_config=act_mc,
                dtype=ttnn.bfloat8_b,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
        else:
            hidden = ttnn.mul(gate, up, memory_config=act_mc, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        pc_d = tpc.create_prefill_mlp_matmul_program_config(M, F, K, max_cols=12)
        out = ttnn.linear(hidden, w["w2"], compute_kernel_config=ck, memory_config=dram, program_config=pc_d)
        ttnn.deallocate(hidden)
        return out

    def forward_tokpar(self, x, prefill_group, chunk_size=128):
        """Grouped TP prefill with a TOKEN-parallel residual stream (QWEN36_TOKPAR=1).

        x: [1,1,S/tp,dim] this chip's rows of the group's S = Bg*Tb rows (chip d owns rows [d*S/tp, (d+1)*S/tp)).
        The mixer (GDN / full attention) runs unchanged, head-parallel: its K-sharded input is the local norm output
        all-to-all'ed to [1,1,S,dim/tp], its hidden-sharded output all-to-all'ed back to rows. Norms are local (no
        distributed-norm CCL); the MLP is token-parallel with the full weights (no all-gather / reduce-scatter)."""
        pg = prefill_group
        out_tp = "wo" in self._tokpar
        a2a_bf8 = tokpar_a2a_bf8()
        # a2a_bf8: the in-proj AGMM gathers bf8 anyway (QWEN36_AGMM_BF8): the norm emits bf8 before the all-to-all
        xn = self._tokpar_norm(x, self.attention_norm, dtype=ttnn.bfloat8_b if a2a_bf8 else None)
        xh = tokpar_a2a(xn, to_tokens=False)
        ttnn.deallocate(xn)
        if self.is_full_attention:
            if pg_batched_attn(pg):
                a = self.attention.forward_prefill_paged_group(
                    xh,
                    pg["cos_cat"],
                    pg["sin_cat"],
                    pg["Bg"],
                    pg["Tb"],
                    pg["full_pts"],
                    pg["chunk_pts"],
                    pg["csi"],
                    return_gated=out_tp,
                )
            else:
                raise NotImplementedError("forward_tokpar needs the batched group attention (cos_cat / sin_cat)")
        else:
            a = self.attention.forward_prefill_group(
                xh,
                pg["Bg"],
                pg["Tb"],
                pg["gdn_masks"],
                pg["gdn_scratch"][id(self.attention)],
                chunk_size=chunk_size,
                init=(pg.get("gdn_init") or {}).get(id(self.attention)),
                return_gated=out_tp,
            )
        ttnn.deallocate(xh)
        a = ttnn.reshape(a, (1, 1, a.shape[-2], a.shape[-1]))
        if out_tp and a2a_bf8 and a.dtype != ttnn.bfloat8_b:  # the out-proj consumes bf8: cast before the all-to-all
            a8_ = ttnn.typecast(a, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(a)
            a = a8_
        at = tokpar_a2a(a, to_tokens=True)
        ttnn.deallocate(a)
        if out_tp:
            # token-parallel out-projection with the full weight (bf8 activation, 12x10 grid, L1 accumulation)
            M, K = at.shape[-2], at.shape[-1]
            if at.dtype == ttnn.bfloat8_b:
                a8 = at
            else:
                a8 = ttnn.typecast(at, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(at)
            at = ttnn.linear(
                a8,
                self._tokpar["wo"],
                compute_kernel_config=(
                    tpc.COMPUTE_LOFI if os.environ.get("QWEN36_TOKPAR_OUT_LOFI", "0") == "1" else tpc.COMPUTE_HIFI2
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=_tokpar_pc(M, K, self._tokpar["wo"].shape[-1], 8),
            )
            ttnn.deallocate(a8)
        # QWEN36_TOKPAR_RES_L1=1: the [S/tp, dim] residual stream lives in L1 (interleaved, ~46 KB/core at S/tp=672)
        res_mc = (
            ttnn.L1_MEMORY_CONFIG if os.environ.get("QWEN36_TOKPAR_RES_L1", "0") == "1" else ttnn.DRAM_MEMORY_CONFIG
        )
        h = ttnn.add(x, at, memory_config=res_mc)
        ttnn.deallocate(at)
        _mlp_bf8 = os.environ.get("QWEN36_TOKPAR_MLP_BF8", "1") == "1"
        ff = self._tokpar_norm(h, self.ffn_norm, dtype=ttnn.bfloat8_b if _mlp_bf8 else None)
        f = self._tokpar_mlp(ff)
        ttnn.deallocate(ff)
        out = ttnn.add(h, f, memory_config=res_mc)
        ttnn.deallocate(h)
        ttnn.deallocate(f)
        return out

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
        gdn_masks=None,
        gdn_recurrent=False,
        gdn_seed=False,
        decode_cfg=False,
        alias_kv_write=False,
        spec_verify_mode=False,
        spec_page_table=None,
        spec_user_page_table=None,
        n_users=1,
        state_blk_idx=None,
        conv_sel=None,
        prefill_group=None,
        gdn_capture_state=True,
        gdn_carry_state=True,
        sp=None,
    ):
        # prefill_group (TP prefill only): dict for a GROUPED short prefill of Bg users x Tb bucket rows laid out
        # user-major in x [1,1,Bg*Tb,dim]: keys Bg, Tb, cos, sin, csi, full_pts / chunk_pts (per-user device page
        # tables), gdn_masks, gdn_scratch ({id(gdn layer): alloc_group_scratch(Bg)}). Full attention runs per user on
        # its row slice (own KV pages), GDN runs once for the whole group (forward_prefill_group).
        # Validate up front: attention/norm treat non-"prefill" as decode while the MoE experts
        # treat non-"decode" as prefill, so an unsupported mode would split the two down opposite
        # paths. Fail fast instead.
        assert mode in ("decode", "prefill"), f"mode must be 'decode' or 'prefill', got {mode!r}"
        # gdn_seed: the spec loop's seed step (one row per user, T = 1). Outside GDN it is the verify
        # body at T = 1; GDN takes forward_seed_recurrent because the ring and E_prev do not exist
        # until capture_verify_trace allocates them.
        # n_users / state_blk_idx / conv_sel: multi-user speculative verify. Bucket rows are n_users
        # users x T = rows // n_users candidates, user-major (row u*T + j). GDN reshapes them to
        # [n_users, T, C] and reads each user's initial state / conv window through the shared device
        # selectors state_blk_idx / conv_sel; full attention groups the SDPA and the aliased KV write
        # per user (spec_n_users, spec_user_page_table).
        # decode_cfg: run a SHORT prefill-mode forward (spec verify, <=TILE_SIZE rows) with the DECODE
        # matmul/norm configuration. Every matmul in the stack already selects decode-vs-prefill purely
        # on `x.shape[-2] <= TILE_SIZE`, so at a 32-row bucket they all pick the DRAM-sharded decode
        # kernels — which stream the weights ONCE per 32-row M-tile instead of re-blocking them for a
        # padded prefill tile. The only thing that does not follow automatically is the norm: PREFILL
        # norms leave the activation K-sharded for a fused AGMM that a <=TILE input never takes, so the
        # decode matmuls would receive a dim/tp tensor. Mode.DECODE gathers PRE-norm and hands them the
        # full-dim activation they expect. This is the weight-load amortization that makes verifying
        # K+1 tokens cost ~one decode step instead of ~K+1.
        _norm_mode = Mode.DECODE if (decode_cfg or mode != "prefill") else Mode.PREFILL
        # Replicated-residual decode (QWEN36_DECODE_ALLREDUCE=1): dispatch on the tensor itself — a full-width
        # input is the replicated RES40 residual; a fractured one (flag off, or a caller that bypassed the model
        # entry) takes the unchanged fractured path below.
        _dar = self._decode_ar
        if _dar is not None and not (mode == "decode" and x.shape[-1] == self.args.dim):
            _dar = None
        _dar_kw = {"decode_ar": _dar} if _dar is not None else {}
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM
            # (DRAM above S=2048: the norm output stays resident through the whole attention, clashing with its CBs).
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {
                    **_attn_norm_config,
                    "distributed_output_mem_config": tpc.prefill_act_mem(x.shape[-2], ttnn.L1_MEMORY_CONFIG),
                }
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        if _dar is not None:
            attn_input = _dar.rms_norm(x, self.attention_norm.norm)
        else:
            attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.num_devices > 1:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill" and prefill_group is not None and pg_batched_attn(prefill_group):
                    pg = prefill_group
                    attn_output = self.attention.forward_prefill_paged_group(
                        attn_input,
                        pg["cos_cat"],
                        pg["sin_cat"],
                        pg["Bg"],
                        pg["Tb"],
                        pg["full_pts"],
                        pg["chunk_pts"],
                        pg["csi"],
                    )
                elif mode == "prefill" and prefill_group is not None:
                    pg = prefill_group
                    Bg, Tb = pg["Bg"], pg["Tb"]
                    D = attn_input.shape[-1]
                    outs = []
                    for i in range(Bg):
                        xi = ttnn.slice(attn_input, (0, 0, i * Tb, 0), (1, 1, (i + 1) * Tb, D))
                        oi = self.attention.forward_prefill_paged(
                            xi,
                            pg["cos"][i] if isinstance(pg["cos"], (list, tuple)) else pg["cos"],
                            pg["sin"][i] if isinstance(pg["sin"], (list, tuple)) else pg["sin"],
                            pg["full_pts"][i],
                            chunk_page_table=pg["chunk_pts"][i],
                            chunk_start_idx=0,
                            chunk_start_idx_tensor=pg["csi"],
                        )
                        ttnn.deallocate(xi)
                        outs.append(oi)
                    if len(outs) == 1:  # Bg == 1: concat of one tensor would return it (then freed below)
                        attn_output = outs[0]
                    else:
                        attn_output = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        for oi in outs:
                            ttnn.deallocate(oi)
                elif mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                            sp=sp,
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    # alias_kv_write: the B rows are not independent users (spec verify's candidates
                    # as pseudo-users), so the KV write cannot be one batched call — see
                    # TPAttention._write_kv_aliased.
                    # spec_verify_mode/spec_page_table: additionally fold those rows into ONE SDPA
                    # batch row so the KV cache is read once per layer (same place).
                    attn_output = self.attention.forward_decode(
                        attn_input,
                        position_tensor,
                        cos,
                        sin,
                        page_table=page_table,
                        alias_kv_write=alias_kv_write,
                        spec_verify_mode=spec_verify_mode,
                        spec_page_table=spec_page_table,
                        **_dar_kw,
                        spec_n_users=n_users,
                        spec_user_page_table=spec_user_page_table,
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill" and prefill_group is not None:
                    pg = prefill_group
                    attn_output = self.attention.forward_prefill_group(
                        attn_input,
                        pg["Bg"],
                        pg["Tb"],
                        pg["gdn_masks"],
                        pg["gdn_scratch"][id(self.attention)],
                        chunk_size=chunk_size,
                        init=(pg.get("gdn_init") or {}).get(id(self.attention)),
                    )
                elif mode == "prefill":
                    if gdn_recurrent and gdn_seed:
                        # Spec-decode seed: the verify's conv1d + fused recurrent arithmetic, run
                        # against the durable state and without the deferred-commit selectors.
                        attn_output = self.attention.forward_seed_recurrent(
                            attn_input, valid_len, pre_gathered=decode_cfg, n_users=n_users
                        )
                    elif gdn_recurrent:
                        # Hybrid spec-decode verify: advance GDN recurrently (bit-exact to decode)
                        # over valid_len tokens while the rest of the stack runs batched.
                        attn_output = self.attention.forward_verify_recurrent(
                            attn_input,
                            valid_len,
                            pre_gathered=decode_cfg,
                            n_users=n_users,
                            state_blk_idx=state_blk_idx,
                            conv_sel=conv_sel,
                        )
                    elif gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        # gdn_masks: persistent device masks of the traced masked-bucket prefill (None: unchanged call).
                        _gm = {"prefill_masks": gdn_masks} if gdn_masks is not None else {}
                        attn_output = self.attention.forward_prefill(
                            attn_input,
                            chunk_size=chunk_size,
                            valid_len=valid_len,
                            capture_state=gdn_capture_state,
                            carry_state=gdn_carry_state,
                            sp=sp,
                            **_gm,
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input, row_pos=position_tensor, **_dar_kw)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        h = _dar.add(x, attn_output) if _dar is not None else ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        if _dar is not None:
            ff_input = _dar.rms_norm(h, self.ffn_norm.norm)
        else:
            ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input, mode=mode, **_dar_kw)
        ttnn.deallocate(ff_input)

        output = _dar.add(h, ff_output) if _dar is not None else ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
