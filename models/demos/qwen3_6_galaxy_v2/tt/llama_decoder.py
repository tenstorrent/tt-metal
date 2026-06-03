# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.qwen3_6_galaxy_v2.tt.distributed_norm import DistributedNorm
from models.demos.qwen3_6_galaxy_v2.tt.llama_attention import TtLlamaAttention
from models.demos.qwen3_6_galaxy_v2.tt.llama_mlp import TtLlamaMLP


def _extract_layer_dn_weights(state_dict, layer_num):
    """Slice the per-layer DeltaNet weights out of the full state_dict.

    Returns a flat dict with the ``linear_attn.*`` prefix preserved — the
    DeltaNet ``_resolve_weight`` helper accepts that form directly.

    Looks for keys of the form ``layers.{layer_num}.linear_attn.<rest>``
    (as emitted by ``load_checkpoints.map_hf_to_meta_keys``) and rewrites
    them to ``linear_attn.<rest>``.
    """
    if state_dict is None:
        return {}
    prefix = f"layers.{layer_num}.linear_attn."
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out["linear_attn." + k[len(prefix) :]] = v
    return out


class TtTransformerBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        n_layers,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher_setup=None,
        tt_ccl=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.weight_cache_path = weight_cache_path
        self.current = 0
        self.model_config = args.get_model_config()

        self.layer_num = layer_num
        self.n_layers = n_layers

        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        self.unfuse_res_add = args.unfuse_res_add

        # --- Hybrid attention dispatch (qwen3.6 only) ---------------------
        # For Qwen3.6-27B, args.linear_attention_pattern is a per-layer list
        # of ``"linear_attention"`` / ``"full_attention"`` strings loaded
        # from HF ``config.layer_types``. Linear layers use the DeltaNet
        # (Gated DeltaNet) block; full-attention layers use the standard
        # gated/QK-norm attention path. For 70B / qwen3-32B / olmo, the
        # ``is_qwen36`` flag is False (or absent) and we instantiate
        # TtLlamaAttention unconditionally — preserving the 70B regression
        # surface.
        self.is_qwen36 = getattr(args, "is_qwen36", False)
        pattern = getattr(args, "linear_attention_pattern", None)
        self.is_linear_attention_layer = bool(
            self.is_qwen36 and pattern is not None and pattern[layer_num] == "linear_attention"
        )

        if self.is_linear_attention_layer:
            # Late import: keeps the 70B import surface decoupled from the
            # qwen36-specific DeltaNet module.
            from models.demos.qwen3_6_galaxy_v2.tt.qwen36_delta_attention import TtQwen36DeltaAttention

            dn_weights = _extract_layer_dn_weights(state_dict, layer_num)
            self.attention = TtQwen36DeltaAttention(
                mesh_device=mesh_device,
                args=args,
                layer_num=layer_num,
                weights_dict=dn_weights,
                tt_ccl=tt_ccl,
                dtype=dtype,
            )
        else:
            self.attention = TtLlamaAttention(
                mesh_device=mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
                transformation_mats=transformation_mats,
                configuration=args,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                prefetcher_setup=prefetcher_setup,
                tt_ccl=tt_ccl,
            )
        self.feed_forward = TtLlamaMLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        # Norm zero-centering (Qwen3NextRMSNorm: output = (1 + w) * normalize(x))
        # is enabled for qwen3.6; default False keeps 70B / qwen3-32B / olmo
        # paths unaffected.
        zero_centered = getattr(args, "zero_centered_norm", False)
        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                output_mem_config=self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
            ),
            args,
            tt_ccl=tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
            zero_centered=zero_centered,
        )
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="ffn_norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                output_mem_config=self.model_config["SHARDED_FF12_RING_MEMCFG"],
            ),
            args,
            tt_ccl=tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
            zero_centered=zero_centered,
        )

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        # DeltaNet layers don't expose a prefetch() hook — they don't use the
        # weight prefetcher. Only call prefetch() on attention blocks that
        # support it (e.g. TtLlamaAttention on the full_attention layers).
        if hasattr(self.attention, "prefetch"):
            self.attention.prefetch(prefetcher_setup, tt_ccl)
        else:
            self.attention.tt_ccl = tt_ccl
        self.feed_forward.prefetch(prefetcher_setup, tt_ccl)
        self.attention_norm.tt_ccl = tt_ccl
        self.ff_norm.tt_ccl = tt_ccl

    def _mlp_decode_qwen36(self, ff_in_sharded: ttnn.Tensor, batch_size: int = 1) -> ttnn.Tensor:
        """V2-decode: SwiGLU MLP for the qwen3.6 single-user decode path.

        The prefill MLP (``forward_prefill``) uses ``matmul_1d_config`` for
        ``seq_len <= 128`` which assumes L1-sharded mcast_in0 input plus
        ``fuse_batch=True`` over an M >= 1 tile. For T=1 (tile-padded to 32),
        the runtime tripped "Only L1 buffers can have an associated circular
        buffer" — the 1D MCAST mcast_in0 path requires L1 input. The decode
        MLP (``forward(mode='decode')``) on the 70B path uses ring-MMU prog
        configs sized for a packed batch=32 single token — also incompatible
        with single-user [B=1, 1, T=1, H/4] DRAM-residual.

        Inline a minimal DRAM-safe variant: plain ``ttnn.linear`` (no prog
        config), the same reduce-scatter / all-gather / all-reduce CCL
        pattern as ``forward_prefill``, and the same residual-stream dtype
        lock (typecast to bf16 at exit). Matches the GPU SwiGLU semantics
        verified at 64L prefill PCC 0.998.

        Input:  ``ff_in_sharded`` ttnn.Tensor [B, 1, T=1, H/4=1280] DRAM,
                col-sharded across cluster_axis=1.
        Output: ttnn.Tensor [B, 1, T=1, H/4=1280] DRAM, col-sharded.
        """
        mlp = self.feed_forward
        # V2-decode: mirror v1 ``TtLlamaMLP.forward`` and use HiFi4 + fp32 dest
        # accumulation throughout (the previous hifi2_fp16 had
        # fp32_dest_acc_en=False which accumulates the K-dim dot product in
        # bf16).  Empirically this did NOT fix the 64L compounding regression,
        # but it brings the decode MLP into structural parity with v1 and
        # matches our prefill MLP's effective HiFi4 path — kept as the safe
        # default to remove one variable from future bisection.
        compute_kernel = self.args.compute_kernel_config_hifi4
        # 1. gate_proj (w1) and up_proj (w3): [B, 1, T, H/4] × [H/4, hidden_per_tp]
        w1_out = ttnn.linear(
            ff_in_sharded,
            mlp.w1_interleaved,
            compute_kernel_config=compute_kernel,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            ff_in_sharded,
            mlp.w3_interleaved,
            compute_kernel_config=compute_kernel,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # 2. Reduce-scatter across cols (cluster_axis=1, partial sum on K dim).
        #    Skip the persistent FFx prefill buffer (seqlen=1 not in support).
        w1_red = mlp.tt_ccl.line_reduce_scatter(
            w1_out,
            cluster_axis=1,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dim=3,
            batch_size=batch_size,
        )
        w1_out.deallocate(True)
        w3_red = mlp.tt_ccl.line_reduce_scatter(
            w3_out,
            cluster_axis=1,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dim=3,
            batch_size=batch_size,
        )
        w3_out.deallocate(True)
        # 3. SwiGLU: silu(w1) * w3
        ff = ttnn.mul(
            w1_red,
            w3_red,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w1_red.deallocate(True)
        w3_red.deallocate(True)
        # 4. Gather across cols so w2 sees the full hidden_per_tp dim.
        ff_gathered = mlp.tt_ccl.line_all_gather(
            ff,
            cluster_axis=1,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dim=3,
        )
        ff.deallocate(True)
        # 5. down_proj (w2): [B, 1, T, hidden_per_tp] × [hidden_per_tp, dim_per_tp]
        w2_out = ttnn.linear(
            ff_gathered,
            mlp.w2_interleaved,
            compute_kernel_config=compute_kernel,  # HiFi4 (see comment above)
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ff_gathered.deallocate(True)
        # 6. All-reduce on rows (cluster_axis=0) so each col holds a full
        #    sum partial -> col-sharded dim_per_tp output.
        w2_red = mlp.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="FF2",
            batch_size=batch_size,
        )
        w2_out.deallocate(True)
        return w2_red

    def forward(
        self,
        x: ttnn.Tensor,
        h: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        # x contains input in layer 0 and ffout of previous layer thereafter, x should be dealocated
        # h contains 0 in layer 0 and h_prev+x_prev+attn_out_prev thereafter, h is persistent
        # V2-decode: qwen3.6 single-user decode uses the DRAM-residual contract
        # (mirrors v1's forward_decode and v2's qwen36 prefill path). The 70B
        # decode L1-sharded contract is structurally incompatible with the
        # qwen3.6 attention/DeltaNet blocks (they were written against
        # ``[B=1, 1, T=1, H]`` DRAM single-user). We re-use the existing
        # is_qwen36_prefill branch below for decode too; the skip_mem_cfg here
        # only governs the 70B decode path.
        is_qwen36_decode = self.is_qwen36 and mode == "decode"
        # Step 1 (QWEN36_DECODE_L1_RESIDUAL, default OFF): move the qwen3.6 decode
        # residual stream from DRAM to L1 (DECODE_RESIDUAL_MEMCFG) and run the norms
        # via the decode-mode sharded primitive (norm_mlp_mode="decode" below). Read
        # the flag ONCE here so both residual adds + the entry contract agree. Flag
        # off ⇒ byte-identical to the DRAM path.
        _l1_residual = is_qwen36_decode and os.environ.get("QWEN36_DECODE_L1_RESIDUAL", "0") == "1"
        # QWEN36_DECODE_32ROW (default OFF): carry the decode hidden as 32 tile-padded
        # rows ([1,1,32,H/4]) through the backbone — llama70b's BATCH-1 contract (1 real
        # user in row 0, 31 tile-padding rows) — so the decode-mode sharded RMSNorm
        # (rms_allgather, which requires logical (1,1,32,M)) is reachable at batch-1,
        # WITHOUT yet moving the residual to L1. Attention (full-attn + GDN) runs
        # SINGLE-USER on row 0 (sliced below) and its output is broadcast back to 32 rows
        # for the residual add — so no batch-N attention path is exercised. ``_l1_residual``
        # IMPLIES this 32-row carry (it additionally flips the norms + residual to L1 — Part
        # B). Flag off ⇒ byte-identical to the legacy 1-row DRAM decode.
        _carry32 = is_qwen36_decode and (os.environ.get("QWEN36_DECODE_32ROW", "0") == "1" or _l1_residual)
        skip_mem_cfg = (
            self.model_config["DECODE_RESIDUAL_MEMCFG"]
            if ((mode == "decode" and not is_qwen36_decode) or _l1_residual)
            else ttnn.DRAM_MEMORY_CONFIG
        )
        if _l1_residual:
            # Layer-0 entry: the embedding (or a test harness) may hand us a DRAM x.
            # Convert once at entry so the residual lives in L1 from here on; layers
            # 1..N-1 already receive the prior layer's L1 `out`, so this is a no-op there.
            if x.memory_config() != skip_mem_cfg:
                x = ttnn.to_memory_config(x, skip_mem_cfg)
        else:
            assert (
                x.memory_config() == skip_mem_cfg
            ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # qwen3.6 prefill sharding contract:
        #   - embedding produces col-sharded [B, 1, T, H/4] (see TtLlamaEmbedding)
        #   - DistributedNorm prefill (tt_distributed_rmsnorm) preserves the
        #     col-sharded layout (gamma is col-sharded H/4)
        #   - qwen3.6 attention (full + DeltaNet) needs full-H replicated input
        #     (wqkvg / w_q,k,v,z,a,b have dims=(None, 3) / dims=(1, None))
        #   - qwen3.6 attention OUTPUT is full-H replicated (per-col WO + all_gather/
        #     fast_reduce_nc inside TtLlamaAttention; per-row out_proj + all_gather/
        #     fast_reduce_nc inside TtQwen36DeltaAttention)
        #   - 70B-style MLP: w1/w3 dim_per_tp=H/4 input × intermediate_per_tp output;
        #     i.e. it wants col-sharded H/4 input and produces col-sharded H/4 output
        #
        # The cleanest fix (V2-7b): keep the **residual stream full-H replicated**
        # throughout the layer (mirrors v1 _shard_across_cols / _gather_from_cols)
        # so attention's add-with-residual lands in the same H representation
        # attention produces. Scatter just before each norm + MLP (which want
        # col-sharded), gather their output before re-joining the residual.
        # The decoder still receives + returns col-sharded H/4 (matching the
        # embedding contract and the inter-layer contract), with a single
        # entry gather + exit scatter.
        is_qwen36_prefill = self.is_qwen36 and mode == "prefill"
        # V2-decode: extend the qwen36 prefill DRAM-residual path to decode too.
        # The norms + MLP are run via the prefill primitives (tt_distributed_rmsnorm,
        # MLP forward_prefill) since those work on DRAM full-H/4 inputs and have
        # been verified at 64L. The attention dispatches to the decode forward
        # (_forward_decode_qwen36 for full-attention, forward_decode for DeltaNet).
        is_qwen36_path = self.is_qwen36 and mode in ("prefill", "decode")
        # Step 1: route the norms to the decode-mode L1 sharded primitive when the L1
        # residual is on; otherwise keep the verified prefill (DRAM) norm primitive.
        norm_mlp_mode = "decode" if _l1_residual else "prefill"
        attn_mode = mode  # "prefill" or "decode" dispatches to the right attention path

        if is_qwen36_path:
            # V2-TP: 2D tensor-parallel data flow.
            # Residual stream is COL-SHARDED H/cols throughout the decoder.
            # - Full-attention layers: pure col-sharded I/O (attention takes
            #   and returns col-sharded). No gather / no scatter in the
            #   decoder.
            # - DeltaNet layers: DeltaNet still requires full-H input/output
            #   (its w_qkvz / w_out are dims=(1, None), so input dim is not
            #   sharded). We gather just before DeltaNet and scatter just
            #   after — same op count as before for DeltaNet, but the
            #   surrounding norm / MLP / residual all stay col-sharded.
            # Net per-layer savings on full-attn: 2 all_gather + 2
            # mesh_partition removed; 16 full-attn layers × 4 CCL ops.

            # --- Attention norm (col-sharded → col-sharded) ---
            # x arrives col-sharded H/cols from the embedding (layer 0) or
            # from the prior layer's exit (layer > 0).  DistributedNorm
            # consumes col-sharded directly (gamma is col-sharded H/cols).
            #
            # DRAM 32-row baseline (_carry32 without L1): the attention sub-path must be
            # BYTE-IDENTICAL to the legacy batch-1 path so the GDN/full-attn decode blocks
            # see the exact 1-row sharded input they were built for (feeding them a
            # 32-row-derived/DRAM layout clashed the GDN seq op's static L1 CBs). RMSNorm is
            # per-row, so we slice the residual to row 0 FIRST and norm just that 1 row —
            # identical to baseline. The 32-row carry stays only on the residual/FF/MLP path
            # below. The L1 path (norm_mlp_mode=="decode") instead norms all 32 rows
            # (rms_allgather requires (1,1,32,M)) and slices AFTER the norm (block below).
            _x_for_attn_norm = x
            if _carry32 and not _l1_residual:
                _xb, _, _xt, _xh = list(x.shape)
                if _xt != 1:
                    _x_for_attn_norm = ttnn.slice(
                        x, [0, 0, 0, 0], [_xb, 1, 1, _xh], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
            attn_in_sharded, _ = self.attention_norm(_x_for_attn_norm, None, norm_mlp_mode)
            if _carry32 and not _l1_residual and _x_for_attn_norm is not x:
                _x_for_attn_norm.deallocate(True)

            # Step 1 (QWEN36_DECODE_L1_RESIDUAL): under the L1 residual the norm runs in
            # "decode" mode (tt_sharded_distributed_rmsnorm → rms_allgather), which REQUIRES
            # a 32-row input/output (logical [1,1,32,M]); the residual stream x is therefore
            # carried as [1,1,32,1280] (llama70b tile_padded_batch_rows convention — at
            # batch-1 all 32 rows hold the SAME token, see embedding). The attention blocks
            # (full-attn _forward_decode_qwen36 + GDN forward_decode), however, reshape
            # [B,1,T,H]→[B,T,H] treating dim-2 as the seq/time dim T and assert/slice to T=1.
            # So just before attention we drop to a single logical row in DRAM ([1,1,1,1280]):
            # row 0 is the correct token (all 32 rows identical). attn_out is then broadcast
            # back to 32 rows for the L1 residual add below. norm_mlp_mode=="prefill" (flag
            # off) already produces a DRAM 1-row tensor, so this is a no-op there.
            # The DRAM 32-row baseline already sliced BEFORE the norm (above), so this
            # post-norm slice is L1-only (norm ran at 32 rows for rms_allgather → drop to row 0).
            if _l1_residual:
                _ais = ttnn.to_memory_config(attn_in_sharded, ttnn.DRAM_MEMORY_CONFIG)
                attn_in_sharded.deallocate(True)
                _B_n, _, _T_n, _H_n = list(_ais.shape)
                if _T_n != 1:
                    _ais_r0 = ttnn.slice(_ais, [0, 0, 0, 0], [_B_n, 1, 1, _H_n], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    _ais.deallocate(True)
                    _ais = _ais_r0
                attn_in_sharded = _ais

            if self.is_linear_attention_layer:
                # V2-DN-TP: DeltaNet now consumes + produces COL-SHARDED H/4.
                # No surrounding gather/scatter needed.
                attn_out = self.attention.forward(
                    attn_in_sharded,
                    current_pos,
                    rot_mats,
                    user_id,
                    attn_mode,
                    page_table=page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx=chunk_start_idx,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    kv_cache=kv_cache,
                    batch_size=batch_size,
                )
                attn_in_sharded.deallocate(True)
                if len(list(attn_out.shape)) == 3:
                    _B_a, _T_a, _H_a = list(attn_out.shape)
                    attn_out = ttnn.reshape(attn_out, ttnn.Shape([_B_a, 1, _T_a, _H_a]))
                if mode == "decode":
                    _B_a, _, _T_a, _H_a = list(attn_out.shape)
                    if _T_a != 1:
                        attn_out_t1 = ttnn.slice(
                            attn_out, [0, 0, 0, 0], [_B_a, 1, 1, _H_a], memory_config=ttnn.DRAM_MEMORY_CONFIG
                        )
                        attn_out.deallocate(True)
                        attn_out = attn_out_t1
            else:
                # Full-attention 2D-TP: col-sharded I/O all the way through.
                attn_out = self.attention.forward(
                    attn_in_sharded,
                    current_pos,
                    rot_mats,
                    user_id,
                    attn_mode,
                    page_table=page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx=chunk_start_idx,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    kv_cache=kv_cache,
                    batch_size=batch_size,
                )
                attn_in_sharded.deallocate(True)
                if len(list(attn_out.shape)) == 3:
                    _B_a, _T_a, _H_a = list(attn_out.shape)
                    attn_out = ttnn.reshape(attn_out, ttnn.Shape([_B_a, 1, _T_a, _H_a]))
                # BATCH-32 full-attention decode: at batch>1 the N users live in
                # dim-2 and MUST survive to the residual add (the residual stream
                # x also carries N rows). ``_forward_decode_qwen36`` now threads
                # the N users through and returns dim-2=N, so only collapse dim-2
                # to 1 in the single-user path. At batch_size==1 attn_out is
                # already T=1 (the caller feeds a 1-row embedding), so guarding
                # this slice on batch_size==1 is byte-identical to the prior code.
                if mode == "decode" and batch_size == 1:
                    _B_a, _, _T_a, _H_a = list(attn_out.shape)
                    if _T_a != 1:
                        attn_out_t1 = ttnn.slice(
                            attn_out, [0, 0, 0, 0], [_B_a, 1, 1, _H_a], memory_config=ttnn.DRAM_MEMORY_CONFIG
                        )
                        attn_out.deallocate(True)
                        attn_out = attn_out_t1

            # --- Residual add (col-sharded) ---
            # x and attn_out are both col-sharded H/cols, bf16.  No dtype
            # kwarg (see prior comment about regression on layer-3 PCC).
            # Step 1 (L1 residual): attention returns a single logical row
            # ([1,1,1,1280] DRAM). The residual stream x lives at 32 rows in L1
            # (DECODE_RESIDUAL_MEMCFG), so broadcast attn_out's row 0 across all
            # 32 rows (ttnn.repeat on dim 2) — at batch-1 every row is the same
            # token — then add in the L1 layout. Flag off keeps the prior path
            # (DRAM 1-row add). _carry32 (DRAM baseline) does the SAME broadcast;
            # skip_mem_cfg is DRAM there (vs L1 for _l1_residual).
            if _carry32:
                _B_o, _, _T_o, _H_o = list(attn_out.shape)
                if _T_o != 32:
                    attn_out_b = ttnn.repeat(attn_out, ttnn.Shape((1, 1, 32 // _T_o, 1)))
                    attn_out.deallocate(True)
                    attn_out = attn_out_b
                attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
            h_new = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)
            x.deallocate(True)
            attn_out.deallocate(True)

            # --- FF norm (col-sharded → col-sharded) ---
            # Step 1 (L1 residual): ff_norm runs in "decode" mode → 32-row L1 output
            # (SHARDED_FF12_RING_MEMCFG). _mlp_decode_qwen36 uses plain DRAM ttnn.linear +
            # the prefill CCL pattern (no ring prog-config), so it accepts the 32 rows
            # cleanly; convert its 32-row L1 input to DRAM first. NOTE: the L1 ring MLP
            # (TtLlamaMLP.forward(mode="decode")) is NOT used here — it depends on the
            # prefetcher global_circular_buffer, which the qwen3.6 decode path explicitly
            # bypasses (llama_model.forward skips create_global_cb for is_qwen36_decode).
            # Routing the ring MLP is left as a follow-up (needs a prefetcher-less ring
            # path or a qwen36 global_cb). See task note.
            #
            # DRAM 32-row baseline: like the attention sub-path, run ff_norm + MLP SINGLE-USER
            # (slice h_new to row 0) so they are byte-identical to the legacy batch-1 decode —
            # _mlp_decode_qwen36 + the prefill-style MLP CCL were written for a 1-row decode
            # token (batch_size=1) and corrupt row 0 when fed 32 rows. The 32-row carry stays
            # only on the residual stream; the genuine 32-row norm/MLP materialize in the L1
            # path (rms_allgather + ring MLP, _l1_residual), not here.
            _h_for_ff_norm = h_new
            if _carry32 and not _l1_residual:
                _hb, _, _ht, _hh = list(h_new.shape)
                if _ht != 1:
                    _h_for_ff_norm = ttnn.slice(
                        h_new, [0, 0, 0, 0], [_hb, 1, 1, _hh], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
            ff_in_sharded, _ = self.ff_norm(_h_for_ff_norm, None, norm_mlp_mode)
            if _carry32 and not _l1_residual and _h_for_ff_norm is not h_new:
                _h_for_ff_norm.deallocate(True)
            if _l1_residual:
                _ffin_dram = ttnn.to_memory_config(ff_in_sharded, ttnn.DRAM_MEMORY_CONFIG)
                ff_in_sharded.deallocate(True)
                ff_in_sharded = _ffin_dram

            # --- MLP (col-sharded → col-sharded) ---
            if mode == "decode":
                ff_out_sharded = self._mlp_decode_qwen36(ff_in_sharded, batch_size=batch_size)
            else:
                ff_out_sharded = self.feed_forward.forward(ff_in_sharded, norm_mlp_mode, batch_size=batch_size)

            # --- Final residual (col-sharded) ---
            # _carry32: the MLP ran single-user (DRAM) or 32-row (L1); broadcast the row-0
            # result back to 32 rows and place it in the residual layout (DRAM or L1) for the
            # final add against the 32-row h_new.
            if _carry32:
                _B_f, _, _T_f, _H_f = list(ff_out_sharded.shape)
                if _T_f != 32:
                    ff_out_b = ttnn.repeat(ff_out_sharded, ttnn.Shape((1, 1, 32 // _T_f, 1)))
                    ff_out_sharded.deallocate(True)
                    ff_out_sharded = ff_out_b
                ff_out_sharded = ttnn.to_memory_config(ff_out_sharded, skip_mem_cfg)
            out_sharded = ttnn.add(ff_out_sharded, h_new, memory_config=skip_mem_cfg)
            ff_out_sharded.deallocate(True)
            h_new.deallocate(True)
            return out_sharded, None

        # --- Non-qwen36 / decode path (unchanged) -------------------------
        # Norms take fractured inputs and output replicated across devices
        # attn_in_sharded=norm(x+h), h = x+h happens implicitly
        if self.layer_num == 0 or mode == "prefill":
            # In the first layer we "make" the h tensor from the original x keeping it alive
            # Note this works because layer 0 has a bfloat16 input while other layers use bfloat8
            # since we want residual to be bfloat16
            attn_in_sharded, _ = self.attention_norm(x, None, mode)
            h = x

        else:
            # In subsequent Layers we take the h tensor from before and modify it in place
            if self.unfuse_res_add:
                h = ttnn.add(x, h)
                attn_in_sharded, _ = self.attention_norm(h, None, mode)
            else:
                attn_in_sharded, _ = self.attention_norm(x, h, mode)

        attn_out = self.attention.forward(
            attn_in_sharded,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )
        if mode == "prefill":
            h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)  # bfloat8_b
            x.deallocate(True)
            ff_in_sharded, _ = self.ff_norm(h, None, mode)
        if mode == "decode":
            if self.unfuse_res_add:
                h = ttnn.add(attn_out, h)
                ff_in_sharded, _ = self.ff_norm(h, None, mode)
            else:
                ff_in_sharded, _ = self.ff_norm(attn_out, h, mode)
            attn_out.deallocate(True)

        # MLP takes replicated inputs and produces fractured outputs
        ff_out = self.feed_forward.forward(ff_in_sharded, mode, batch_size=batch_size)
        if self.layer_num == self.n_layers - 1 or mode == "prefill":
            out = ttnn.add(ff_out, h, memory_config=skip_mem_cfg)  # , dtype=ttnn.bfloat16)
            if mode == "decode":
                ff_out.deallocate(True)
            if mode == "prefill":
                h.deallocate(True)
            return out, None
        else:
            return ff_out, h
