# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import ttnn
from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import is_blackhole
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
        # BH-only: the fused all_gather_minimal_matmul_async grid assumes BH's taller (9-10 row)
        # compute grid (see tp_common.py all_gather_matmul_prefill); WH tops out at 8 rows, so this
        # fusion is unvalidated there. Falls back to unfused AG + matmul on WH.
        self._fuse_norm_agmm = (
            self.num_devices > 1
            and is_blackhole()
            and (
                (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
                or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
            )
        )
        # bf8 attention_norm prefill all-gather. DONE for GDN layers (see _attn_gather_dtype in
        # forward()); NOT DONE for full-attention layers.
        #
        # Two blockers were found (2026-08-17), one since dissolved:
        #   1. KV cache -- paged_fill_cache rejects bf8 K/V against a bf16 cache,
        #      impossible with today's paged_update_cache (see the contract note in attention/tp.py).
        #      Needs a C++ change. Still open, so full-attention layers keep bf16.
        #   2. GDN depthwise FIR -- ternary.cpp:268 needs all addcmul operands to
        #      share a dtype. NOT a property of the layer but of the CALL: the FIR
        #      runs only on masked/tail chunks; a full chunk takes native conv1d
        #      call on the GDN native-conv predicate; FIR chunks keep bf16.
        #
        # The real reason (2) looked unfixable: ttnn.linear defaults its out dtype to in0's
        # (matmul.cpp:72), so a bf8 in0 silently made the whole qkvzab in-proj bf8. The bf8 slice
        # ternary.cpp rejected came from THAT, not the gather. In-proj is now pinned to bf16
        # (gdn/tp.py _project_qkvzab), keeping the recurrent path bit-for-bit on its old dtypes.
        #
        # MEASURED (T3K TP=8, 27B, GDN layer, seq 2048, device kernel duration):
        #     attention_norm all-gather   1,144 -> 698 us   -446   (BF16 -> BFP8, 10 cores both)
        #     LayerNormPostAllGather         48 ->  39 us     -9
        #     qkvzab in-proj 2048x5120x2112 567 -> 553 us    -14   (BF16xBFP8 -> BFP8xBFP8)
        #     attention block             4,806 -> 4,336 us -470
        #     whole layer                 7,353 -> 6,889 us -464   (-6.3%)
        #     MLP block (control)         2,464 -> 2,470 us    +6   (noise)
        # = -21.4ms per 2048-token chunk over 48 GDN layers x 31/32 chunks; model-level attn_norm
        # gather 73.1ms (16.7% of prefill) -> ~51.8ms.
        # Cost: test_gdn_tp_prefill PCC 0.9992650 -> 0.9991681 (-1e-4), same order as ff_norm's.
        #
        # The in-proj won only 14us, not the qkv matmul's -149us: at 59% FPU it
        # is compute-bound (DRAM 71 -> 54 GB/s, FLOPs unchanged). That -149us was the output narrowing
        # cascading downstream, which the pin declines. Unpinning is a real lever but lands on
        # a/b -> sigmoid/softplus -> the recurrent decay, so it needs a demo-level accuracy gate.
        #
        # Blocker (1) leaves 16 full-attention layers x 1,144us = 18.3ms/chunk on the table.
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

        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        # MODEL-GATED, MLP-SCOPED. ff_norm's output is consumed by the MLP alone (forward() deallocates
        # it right after feed_forward.forward), so narrowing to bf8 cannot touch attention. It only
        # takes effect on the POST-norm gather, which on a 1D mesh is is_distributed_norm ==
        # (dim > 4096 and prefill) -- the 27B, never the 9B. The dim test is
        # never gathers here (mlp_gateup_agmm_enabled fuses the gather into the matmul).
        #
        # WHY: at TP=8 that gather moves 2.62MB/device over ONE ETH link, the
        # block. bf8 halves the payload and costs nothing -- rms_norm_post_all_gather takes a dtype, so
        # the norm itself also got faster (49 -> 39us). Accuracy trade matches the down-proj's
        # (mlp.py in0 bf8, PCC 0.99978) and lands on the loosest matmuls in the layer.
        #
        # TRIED bf8 -> bfp4 AND REJECTED (2026-08-20). MEASURED (T3K TP=8, 27B, seq 2048):
        #     ff_norm gather   bf16 2.62MB/dev 1,144us | bf8 1.31MB 693us | bfp4 0.66MB 680us
        # The second halving bought 13us, not the ~280 a bytes model predicts. Same capture:
        # attention_norm (bf8, 692us) and ff_norm (bfp4, 680us) sit 12us apart carrying 2x different
        # payloads -- both on a ~660-690us FLOOR. At 0.66MB over 7 hops in 680us
        # of ~12.5, so what remains is per-hop latency and 7 sequential hops at get_num_links()==1 --
        # not payload, and not chunks_per_sync (swept; tp_common.prefill_ccl_tuning). Whole change was
        # -78us/layer = -5.0ms per 2048-token chunk (1.1% of prefill) against a projected -19ms.
        #
        # And the precision step is NOT free the way bf8 was. test_mlp_tp_prefill vs fp32 torch:
        # bf16 0.9989521, bf8 0.9989442 (-8e-6), bfp4 0.9979378 (-1.0e-3) -- 130x the bf8 step, a
        # 3rd-decimal move on the MLP block alone before compounding over 64 layers. If you re-try
        # this, note mlp.py FLOORS gate/up's output at bf8 so the 4-bit mantissa cannot reach the
        # SwiGLU accumulation; without that floor ttnn.linear propagates in0's dtype and you measure a
        # different, worse trade.
        #
        # THE LEVER LEFT is links/topology, not dtype: fewer hops, more links, or
        # the matmul (all_gather_minimal_matmul_async -- Blackhole's path, needs the C++ NOC-assignment
        # fix for 8-row grids, see tp_common.mlp_gateup_agmm_enabled). Same conclusion for
        # attention_norm's gather: same op, same shape, same floor.
        _ff_gather_dtype = ttnn.bfloat8_b if (args.dim > 4096 and not is_blackhole()) else None
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
            prefill_gather_dtype=_ff_gather_dtype,
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
        prefill_gather_dtype=None,
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
            # PrefillTunedDistributedNorm == DistributedNorm except that the PREFILL all-gather gets
            # tuned chunks_per_sync / num_workers_per_link (and, for ff_norm, a narrowed dtype).
            # Upstream only honours the per-op CCL configs for mode == "decode" and hardcodes 10/2
            # otherwise, which left the 9B's pre-norm gather at ~1,245us/layer; tuned it is ~1,015us.
            # The 27B takes the other branch (post-norm gather) -- see tt/prefill_norm_tuned.py.
            # Decode and TG delegate to upstream unchanged.
            from models.demos.blackhole.qwen36.tt.prefill_norm_tuned import PrefillTunedDistributedNorm

            return PrefillTunedDistributedNorm(
                norm,
                args,
                tt_ccl=tt_ccl,
                TG=args.is_galaxy,
                ag_config_key=ag_key,
                enable_all_gather=enable_all_gather,
                prefill_gather_dtype=prefill_gather_dtype,
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
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            #
            # MODEL-GATED on dim. This full-width [S, dim] output stays L1-resident across the whole
            # layer, including the GDN chunk kernel. At S=2048 that is 262 KB/core for the 9B
            # (dim 4096) and 328 KB/core for the 27B (dim 5120) over a Wormhole's 64 cores; the 27B
            # figure leaves the chunk kernel ~20 KB short of placing its own CBs and it dies with
            # "circular buffers ... clash with L1 buffers". Blackhole has both the larger L1 and ~110
            # cores, so it keeps the tuned path. Gate on dim rather than model_name: HF_MODEL is often
            # a hashed snapshot directory.
            _norm_l1_fits = self.args.dim <= 4096 or is_blackhole()
            if _norm_mode == Mode.PREFILL and _norm_l1_fits:
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
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        # PER-CALL bf16 -> bf8 narrowing of attention_norm's prefill gather. This gather is the
        # largest single line item in whole-model prefill (73.1ms/chunk, 16.7%), it is bytes-bound at
        # TP=8, and bf8 is free inside the norm -- see prefill_norm_tuned.py and the __init__ block
        # above for the measurements and for the two blockers that kept it bf16 until now.
        #
        # WHY PER CALL AND NOT A CONSTRUCTOR ARG (which is how ff_norm's is wired): the FIR blocker is
        # a property of the CALL, not the layer. ttnn.addcmul needs all three operands to share a
        # dtype, but the MAC FIR only runs on MASKED chunks; a full chunk takes ttnn.conv1d, which has
        # no addcmul. At 64k that is 31 of 32 chunks. Same norm object, different answer per chunk.
        #
        # The gate is the GDN module's OWN predicate rather than a re-derivation: prefill_uses_native_
        # conv1d() IS the expression forward_prefill uses for _use_native_conv1d. Out of sync, this
        # crashes in ternary.cpp on exactly the masked tail chunk that no single-layer perf test
        # reaches -- test_demo_text is the gate that matters. The batched-group prefill path in
        # model.py calls attention_norm directly (no kwarg -> bf16), which is also correct: its GDN
        # always takes the FIR because valid_lens is a per-row list.
        #
        # Full-attention layers stay bf16 (blocker 1, the KV cache). is_distributed_norm keeps this to
        # the POST-norm gather -- the 27B, prefill, never the 9B and never decode -- because only that
        # gather sends the norm's OUTPUT, where the dtype is a norm kwarg. Blackhole never gathers
        # here at all (fused into the in-proj AGMM).
        _attn_gather_dtype = None
        if (
            self.num_devices > 1
            and not self.is_full_attention
            and self.args.is_distributed_norm(_norm_mode)
            and not is_blackhole()
            and self.attention.prefill_uses_native_conv1d(x.shape[-2], valid_len)
        ):
            _attn_gather_dtype = ttnn.bfloat8_b
        _attn_norm_kw = {"prefill_gather_dtype": _attn_gather_dtype} if self.num_devices > 1 else {}
        attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config, **_attn_norm_kw)

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

        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
