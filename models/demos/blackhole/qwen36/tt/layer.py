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

_TILE = 32


def make_decode_norm_sharded_config(dim, num_cores=8):
    """I-1 D3 (QWEN36_I1_D3): WIDTH-sharded L1 memory config + sharded rms_norm program config for
    the single-device DECODE decoder norms ([1,1,dim] input = one tile row). The interleaved
    rms_norm runs that one tile row on 1 core (~33 us); 8 cores, shard (32, dim/8), run it in ~5 us
    (T5b D3: 16/32/64 cores are slower than 8). Returns (memory_config, program_config), or None if
    dim does not split into whole tiles on num_cores cores."""
    if dim % (num_cores * _TILE) != 0:
        return None
    shard_w = dim // num_cores
    mem_cfg = ttnn.create_sharded_memory_config(
        shape=(_TILE, shard_w),
        core_grid=ttnn.CoreGrid(y=1, x=num_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    block_w = shard_w // _TILE
    subblock_w = max(i for i in (1, 2, 3, 4) if block_w % i == 0)
    prog_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[num_cores, 1],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    return mem_cfg, prog_cfg


_NORM_WIDE_PC = {}
_NORM_WIDTH_SPLIT = 3


def _norm_wide_pc(x):
    """Width-split interleaved RMSNorm program config (LayerNormDefaultProgramConfig(width_split=3): every tile row on
    3 cores that exchange their partial mean of squares, one tile row per core) for a full-T single-device prefill
    norm at T in tp_common.R3_T_SET, or None (the default interleaved op) when it does not fit the grid. T=1024,
    dim 2048 on 11x10 (96 cores): fused add+norm 31.3 -> 13.9 us (bf16 residual), 30.3 -> 13.3 us (bf8), plain
    21.7 -> 12.6 us. Not bit-exact (the cross-core sum of squares changes the reduction order)."""
    shape = list(x.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] not in tpc.R3_T_SET or x.memory_config().is_sharded():
        return None
    split = _NORM_WIDTH_SPLIT
    mt, kt = shape[1] // _TILE, shape[2] // _TILE
    grid = x.device().compute_with_storage_grid_size()
    if shape[1] % _TILE or shape[2] % _TILE or kt < 2 * split or mt * split > grid.x * grid.y:
        return None
    if split not in _NORM_WIDE_PC:
        _NORM_WIDE_PC[split] = ttnn.LayerNormDefaultProgramConfig(width_split=split)
    return _NORM_WIDE_PC[split]


def _m5_add_norm(norm, a, b, h_mc, n_mc):
    """M5 ADDNORM (tp_common M5 table): h = a + b and n = rmsnorm(h) * gamma in ONE op (the R6 stage-1
    ttnn.rms_norm residual_output_tensor path). h is allocated here (bf16 TILE, a's shape, memory config h_mc =
    where the plain ttnn.add(a, b) writes); n goes to n_mc. Same eps / weight / HiFi2 compute config /
    program_config=None as RMSNorm.forward, so h and n are bit-identical to ttnn.add + ttnn.rms_norm (R6 T1/T3).
    Returns (h, n)."""
    h = ttnn.allocate_tensor_on_device(a.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, a.device(), h_mc)
    n = ttnn.rms_norm(
        a,
        epsilon=norm.eps,
        weight=norm.weight,
        residual_input_tensor=b,
        program_config=_norm_wide_pc(a),
        memory_config=n_mc,
        compute_kernel_config=norm.compute_kernel_config_hifi2,
        residual_output_tensor=h,
    )
    return h, n


def decode_norm_sharded_applies(x):
    """D3 applies to an interleaved TILE decode input with one logical row ([1,1,dim] or [1,1,1,dim])."""
    shape = list(x.shape)
    return (
        len(shape) in (3, 4)
        and all(d == 1 for d in shape[:-1])
        and x.layout == ttnn.TILE_LAYOUT
        and not x.memory_config().is_sharded()
    )


def decode_norm_sharded_out(norm, x, cfg):
    """D3 without the final S2I: I2S(x) -> sharded rms_norm; returns the 8-core WIDTH_SHARDED L1 output
    (M4: the I-3 A3 LM head reshards it to its own in0 layout)."""
    mem_cfg, prog_cfg = cfg
    xs = ttnn.to_memory_config(x, mem_cfg)
    y = ttnn.rms_norm(
        xs,
        epsilon=norm.eps,
        weight=norm.weight,
        program_config=prog_cfg,
        memory_config=mem_cfg,
        compute_kernel_config=norm.compute_kernel_config_hifi2,
    )
    ttnn.deallocate(xs)
    return y


def decode_norm_sharded(norm, x, cfg, out_memory_config):
    """D3: I2S(x) -> sharded rms_norm (same eps / weight / HiFi2 ckc as RMSNorm.forward) -> S2I to
    out_memory_config. Not routed through RMSNorm.forward(in_sharded=True): its sharded_to_interleaved
    takes no memory_config (models/common/rmsnorm.py, shared file)."""
    y = decode_norm_sharded_out(norm, x, cfg)
    out = ttnn.sharded_to_interleaved(y, out_memory_config)
    ttnn.deallocate(y)
    return out


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
        self.tp_enabled = self.num_devices > 1 or getattr(args, "sequence_parallel", False)
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
        # I-1 D3 (QWEN36_I1_D3, single device): 8-core width-sharded decode norms; None = pre-I-1 path.
        self._decode_norm_cfg = (
            make_decode_norm_sharded_config(args.dim) if not self.tp_enabled and tpc.i1_enabled("D3") else None
        )
        # M1 S3 / S4 (QWEN36_M1_S3 / _S4, single device, GDN layers): tp_common.m1_gdn_norm_l1 decides per
        # call whether the prefill attention_norm output goes to L1 (see forward()); None = never.
        self._m1_norm_l1_fn = None
        self._m1_grid = None
        if not self.tp_enabled and not self.is_full_attention:
            _g = mesh_device.compute_with_storage_grid_size()
            self._m1_grid = ttnn.CoreCoord(_g.x, _g.y)
            self._m1_norm_l1_fn = tpc.m1_gdn_norm_l1
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

        if self.tp_enabled:
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
        from models.demos.blackhole.qwen36.tt.tp_common import n_enabled

        norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            # N GAMMA_L1 (QWEN36_N_GAMMA_L1, default 0): gamma in L1 interleaved instead of DRAM
            # (placement only, bit-exact). Applies to attention_norm and ffn_norm only (the two
            # norms _m5_add_norm reads); the final norm and the FA q/k norms are built elsewhere.
            weight_memory_config=ttnn.L1_MEMORY_CONFIG if n_enabled("GAMMA_L1") else ttnn.DRAM_MEMORY_CONFIG,
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
        last_row_only=False,
        last_row_tile_slices=False,
        last_row_pos_tensor=None,
        m5_addnorm=False,
        pending=None,
        defer_out=False,
        post_mixer_hook=None,
    ):
        # last_row_tile_slices (M4 R4A) / last_row_pos_tensor (M4 R4B), with last_row_only only (tp_common M4
        # table): R4A = the one-row reads (residual row here; gate input + SDPA output in the attention) are a
        # tile-aligned [T-32:T] block slice + row 31 of the block (bit-exact); R4B = the attention's SDPA is the
        # paged decode SDPA for row T - 1 at the absolute position held by last_row_pos_tensor (int32 [1]).
        assert last_row_only or (
            not last_row_tile_slices and last_row_pos_tensor is None
        ), "last_row_tile_slices / last_row_pos_tensor (M4) need last_row_only (M2 LASTROW)"
        # M5 ADDNORM (tp_common M5 table; Qwen36Model._forward_prefill_chunk passes m5_addnorm=True for its
        # T in M5_ADDNORM_T_SET chunks): fused residual add + RMSNorm (_m5_add_norm) for full-T pairs.
        #   pending=(h_prev, mlp_prev): x is None; this layer makes its input x = h_prev + mlp_prev (the previous
        #     layer's MLP residual add, which that layer deferred) together with its attention_norm, frees
        #     h_prev and mlp_prev, owns x and frees it after its last reader (the attention residual add).
        #   defer_out=True: skip the MLP residual add and return (h, ff_output) for the next layer's pending.
        assert m5_addnorm or (pending is None and not defer_out), "pending / defer_out need m5_addnorm (M5 ADDNORM)"
        assert (x is None) == (pending is not None), "M5 ADDNORM: pass exactly one of x and pending"
        assert not (defer_out and last_row_only), "M5 ADDNORM: a last_row_only layer cannot defer its output"
        # post_mixer_hook (sequence-parallel prefill, tt/sp_prefill_sc.py): a no-arg callable run once, right
        # after the token mixer (attention / GDN) and its residual add, before ffn_norm (with the M5 ADDNORM
        # intra-layer fused add + ffn_norm, right before that fused op). SP uses it to enqueue the cross-die
        # send of the state the mixer just wrote (paged K/V or GDN recurrent + conv state) before the MLP.
        # None (default) = no call, identical behavior.
        own_x = pending is not None
        xs = x if x is not None else pending[0]  # shape / memory reference until x exists
        # last_row_only (M2 LASTROW, tp_common; single device, full-attention layer, paged prefill with a
        # chunk page table): keep the K/V cache fill and the full SDPA, then compute the rest of the layer
        # (gate, head concat, gate multiply, o_proj, residual, ffn_norm, MLP, residual) for the last row
        # only. Returns [1, 1, dim] = row T - 1 of the full output.
        if last_row_only:
            assert (
                self.num_devices == 1
                and self.is_full_attention
                and mode == "prefill"
                and chunk_page_table is not None
                and valid_len is None
                and not gdn_collect
                and len(xs.shape) == 3
                and xs.shape[0] == 1
            ), "last_row_only: single-device full-attention paged chunk prefill, B == 1, 3D input only"
        # Validate up front: attention/norm treat non-"prefill" as decode while the MoE experts
        # treat non-"decode" as prefill, so an unsupported mode would split the two down opposite
        # paths. Fail fast instead.
        assert mode in ("decode", "prefill"), f"mode must be 'decode' or 'prefill', got {mode!r}"
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        # F10A (G5) / F10B (item A): L1-resident residual add + norm outputs, single-device
        # prefill only (see below); TP and decode are untouched (memory_config=None below ==
        # omitting the kwarg, today's behavior).
        _residual_mc = None
        if self.tp_enabled:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # unchanged by F10A/G5/F10B (decode, T==1, is out of scope for these changes).
            #
            # F10A (G5): prefill norm output also goes to L1 when T <= QWEN36_LAYER_L1_MAX_T
            # (new flag, default 0/off -- see below). Above the threshold the framework RMSNorm
            # keeps returning interleaved DRAM (matches the old None default).
            #
            # Caution (see models/common/rmsnorm.py forward()): output_mem_config is applied via a
            # SEPARATE ttnn.to_memory_config after ttnn.rms_norm (rms_norm itself always writes
            # DRAM here, since neither in_sharded nor out_sharded is set) -- this is an *extra* copy
            # op per norm call, not a free reinterpretation.
            #
            # Measured (step1 F10A): at T=2048 this norm-output-in-L1 placement makes a GDN layer's
            # native conv1d (conv1d_native.py) throw "Statically allocated circular buffers ...
            # clash with L1 buffers" -- the L1-resident norm output is still alive when conv1d's own
            # CBs are sized, and the two don't both fit. Isolated A/B (QWEN36_LAYER_L1_DEBUG,
            # residual-only vs norm-only) confirmed the residual-add L1 placement below is NOT the
            # cause (it passes standalone); the norm output_mem_config alone reproduces the clash.
            # Since one flag drove both and the norm half breaks GDN layers at the production T,
            # QWEN36_LAYER_L1_MAX_T stays 0 by default (fully off, matching pre-F10A prefill
            # behavior) -- set it explicitly (e.g. =2048) only for further A/B experimentation.
            #
            # F10B (item A): the residual add is now split onto its OWN flag (QWEN36_LAYER_RESID_L1)
            # instead of being coupled to QWEN36_LAYER_L1_MAX_T above. F10A's isolated A/B claimed
            # the residual-add-in-L1 half is safe standalone at T=2048; step1-F10B re-measured it
            # end to end (full 24-layer model, real traced-chunk-outer capture) and found the
            # opposite: turning this on for EVERY layer keeps the whole inter-layer residual stream
            # L1-resident continuously (each layer's `output` is the next layer's `x`), and that
            # persistent L1 pressure clashes with a GDN layer's native conv1d CBs exactly like the
            # norm case above -- "Statically allocated circular buffers ... clash with L1 buffers",
            # reproduced in capture_prefill_trace_chunked's WARMUP call (the traced path this item
            # targets) at T=2048, and again in the eager multi-chunk path (prefill_layer_chunked) at
            # T=4096, and again in the T<=1024 single-shot short-prefill path. Isolating just the
            # model.py post-embedding placement (this same flag, one L1 hop from embd into layer 0,
            # reverting to DRAM after layer 0's own add) passed standalone -- so the risk is specific
            # to chaining the L1 placement across many layers, not to any single add. Given this,
            # QWEN36_LAYER_RESID_L1 now DEFAULTS TO "0" (off, byte-identical to pre-F10A/F10B) --
            # set it to "1" only to reproduce the clash or to continue investigating a narrower
            # (e.g. GDN-layer-aware, or freed-between-layers) placement.
            _residual_mc = None
            if mode == "decode":
                _attn_norm_config = _ff_norm_config = {"output_mem_config": ttnn.L1_MEMORY_CONFIG}
            else:
                _layer_l1_max_t = int(os.environ.get("QWEN36_LAYER_L1_MAX_T", "0"))
                _T = xs.shape[1] if len(xs.shape) >= 3 else 1
                if _T <= _layer_l1_max_t:
                    _attn_norm_config = _ff_norm_config = {"output_mem_config": ttnn.L1_MEMORY_CONFIG}
                else:
                    _attn_norm_config = _ff_norm_config = None
                if os.environ.get("QWEN36_LAYER_RESID_L1", "0") == "1" and _T <= 2048:
                    _residual_mc = ttnn.L1_MEMORY_CONFIG
        # I-1 D3: single-device decode norms run width-sharded on 8 cores; the output is the same
        # interleaved L1 tensor as the output_mem_config=L1 path above.
        _d3 = self._decode_norm_cfg is not None and mode == "decode"
        # M1 S3 / S4 (tp_common M1 table): GDN layer, unmasked T == 2048 prefill chunk -> the attention_norm
        # output is written to L1 interleaved by the rms_norm op itself (same eps / weight / HiFi2 ckc and
        # program_config=None as RMSNorm.forward; no extra copy op). The GDN in-proj frees it after its last
        # consumer (ttnn_gated_deltanet.py), so the deallocate below is then a no-op.
        _m1_norm_l1 = (
            self._m1_norm_l1_fn is not None
            and mode == "prefill"
            and valid_len is None
            and not gdn_collect
            and _attn_norm_config is None
            and len(xs.shape) == 3
            and self._m1_norm_l1_fn(xs.shape[1], self._m1_grid)
        )
        # M5 ADDNORM: the fused op applies to a full-T (T in M5_ADDNORM_T_SET rows) single-device prefill pair whose norm has
        # no extra output config (the norm output then lands where the plain norm writes it); else the plain ops.
        _m5 = (
            m5_addnorm
            and not self.tp_enabled
            and mode == "prefill"
            and valid_len is None
            and not gdn_collect
            and len(xs.shape) == 3
            and xs.shape[1] in tpc.M5_ADDNORM_T_SET
            and xs.dtype == ttnn.bfloat16
        )
        attn_input = None
        if pending is not None:
            # M5 ADDNORM cross-layer pair: x = h_prev + mlp_prev (the add the previous layer deferred, written where
            # that add writes today) + this layer's attention_norm (L1 on the M1 path, else x's placement).
            h_prev, mlp_prev = pending
            x_mc = _residual_mc if _residual_mc is not None else h_prev.memory_config()
            if _m5 and _attn_norm_config is None:
                x, attn_input = _m5_add_norm(
                    self.attention_norm, h_prev, mlp_prev, x_mc, ttnn.L1_MEMORY_CONFIG if _m1_norm_l1 else x_mc
                )
            else:
                x = ttnn.add(h_prev, mlp_prev, memory_config=_residual_mc)
            ttnn.deallocate(h_prev)
            ttnn.deallocate(mlp_prev)
        if attn_input is None:
            if _d3 and decode_norm_sharded_applies(x):
                attn_input = decode_norm_sharded(self.attention_norm, x, self._decode_norm_cfg, ttnn.L1_MEMORY_CONFIG)
            elif _m1_norm_l1:
                attn_input = ttnn.rms_norm(
                    x,
                    epsilon=self.attention_norm.eps,
                    weight=self.attention_norm.weight,
                    program_config=None,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=self.attention_norm.compute_kernel_config_hifi2,
                )
            elif (
                mode == "prefill"
                and not self.tp_enabled
                and _attn_norm_config is None
                and len(x.shape) == 3
                and _norm_wide_pc(x) is not None
            ):
                # The RMSNorm.forward call (norm_config None: output in x's placement) with the width-split
                # program config.
                attn_input = ttnn.rms_norm(
                    x,
                    epsilon=self.attention_norm.eps,
                    weight=self.attention_norm.weight,
                    program_config=_norm_wide_pc(x),
                    memory_config=x.memory_config(),
                    compute_kernel_config=self.attention_norm.compute_kernel_config_hifi2,
                )
            else:
                attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.tp_enabled:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
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
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input, position_tensor, cos, sin, page_table=page_table
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len, capture_state=True
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input)
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
                last_row_only=last_row_only,
                **(
                    dict(last_row_tile_slices=last_row_tile_slices, last_row_pos_tensor=last_row_pos_tensor)
                    if last_row_only
                    else {}
                ),
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        if last_row_only:
            # M2 LASTROW: attn_output is row T - 1 only ([1, 1, dim]); take the same row of the residual
            # (same slice + to_layout as Qwen36Model's exact-multiple last-row read). The 1-row ffn_norm
            # writes L1 interleaved, as in decode, so the MLP runs its T == 1 (decode) path.
            _T_full = x.shape[1]
            if last_row_tile_slices:
                # M4 R4A: tile-aligned 32-row block (TILE path), then its row 31 (small untilize path).
                _x_blk = x[:, _T_full - 32 : _T_full, :]
                x_res = ttnn.to_layout(_x_blk[:, 31:32, :], ttnn.TILE_LAYOUT)
                ttnn.deallocate(_x_blk)
            else:
                x_res = ttnn.to_layout(x[:, _T_full - 1 : _T_full, :], ttnn.TILE_LAYOUT)
            h = ttnn.add(x_res, attn_output, memory_config=_residual_mc)
            ttnn.deallocate(x_res)
            _ff_norm_config = {"output_mem_config": ttnn.L1_MEMORY_CONFIG}
            ff_input = None
        elif _m5 and _ff_norm_config is None:
            # M5 ADDNORM intra-layer pair: h = x + attn_output (where the residual add writes today) + ffn_norm
            # (the plain ffn_norm writes h's placement, as here).
            h_mc = _residual_mc if _residual_mc is not None else x.memory_config()
            if post_mixer_hook is not None:
                post_mixer_hook()  # before the fused add + ffn_norm (the mixer state is complete here)
                post_mixer_hook = None
            h, ff_input = _m5_add_norm(self.ffn_norm, x, attn_output, h_mc, h_mc)
        else:
            h = ttnn.add(x, attn_output, memory_config=_residual_mc)
            ff_input = None
        ttnn.deallocate(attn_output)
        if post_mixer_hook is not None:
            post_mixer_hook()
        if own_x:
            # M5 ADDNORM: this layer made x (pending); the residual add above was its last reader.
            ttnn.deallocate(x)

        if ff_input is None:  # (M5 ADDNORM: the fused op above made it)
            # M2 LASTROW: the 1-row ffn_norm also runs on the 8-core D3 decode layout.
            _lastrow_d3 = last_row_only and self._decode_norm_cfg is not None
            if (_d3 or _lastrow_d3) and decode_norm_sharded_applies(h):
                ff_input = decode_norm_sharded(self.ffn_norm, h, self._decode_norm_cfg, ttnn.L1_MEMORY_CONFIG)
            else:
                ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        if last_row_only and isinstance(self.feed_forward, Qwen36MLP):
            ff_output = self.feed_forward.forward(ff_input, mode=mode, last_row_only=True)
        else:
            ff_output = self.feed_forward.forward(ff_input, mode=mode)
        ttnn.deallocate(ff_input)

        if defer_out:
            # M5 ADDNORM: the next layer fuses this residual add with its attention_norm (pending).
            return (h, ff_output)
        output = ttnn.add(h, ff_output, memory_config=_residual_mc)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
