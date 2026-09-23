# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import os

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


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
        norm = RMSNorm(
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
    ):
        # Validate up front: attention/norm treat non-"prefill" as decode while the MoE experts
        # treat non-"decode" as prefill, so an unsupported mode would split the two down opposite
        # paths. Fail fast instead.
        assert mode in ("decode", "prefill"), f"mode must be 'decode' or 'prefill', got {mode!r}"
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        # F10A (G5) / F10B (item A): L1-resident residual add + norm outputs, single-device
        # prefill only (see below); TP and decode are untouched (memory_config=None below ==
        # omitting the kwarg, today's behavior).
        _residual_mc = None
        if self.num_devices > 1:
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
                _T = x.shape[1] if len(x.shape) >= 3 else 1
                if _T <= _layer_l1_max_t:
                    _attn_norm_config = _ff_norm_config = {"output_mem_config": ttnn.L1_MEMORY_CONFIG}
                else:
                    _attn_norm_config = _ff_norm_config = None
                if os.environ.get("QWEN36_LAYER_RESID_L1", "0") == "1" and _T <= 2048:
                    _residual_mc = ttnn.L1_MEMORY_CONFIG
        attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.num_devices > 1:
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
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        h = ttnn.add(x, attn_output, memory_config=_residual_mc)
        ttnn.deallocate(attn_output)

        ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input, mode=mode)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output, memory_config=_residual_mc)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
