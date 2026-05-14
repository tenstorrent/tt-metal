# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
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
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
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

        if is_qwen36_prefill:
            # --- Entry gather: residual stream lifted to full-H ---
            # For layer 0, x and h-into-block both = the embedding output
            # (col-sharded). Gather x to full-H once; h is unused this branch.
            if self.layer_num == 0:
                x_full = ttnn.all_gather(
                    x,
                    dim=3,
                    num_links=1,
                    cluster_axis=1,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x.deallocate(True)
                x = x_full
                # h follows x (residual seed), full-H replicated.
                h = x
            else:
                # x and h came from prior layer's exit (still col-sharded).
                x_full = ttnn.all_gather(
                    x,
                    dim=3,
                    num_links=1,
                    cluster_axis=1,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x.deallocate(True)
                x = x_full
                if h is not None:
                    h_full = ttnn.all_gather(
                        h,
                        dim=3,
                        num_links=1,
                        cluster_axis=1,
                        topology=ttnn.Topology.Linear,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    h.deallocate(True)
                    h = h_full

            # --- Attention norm: scatter → norm (col-sharded) → gather ---
            x_norm_in = ttnn.mesh_partition(x, dim=-1, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_in_sharded, _ = self.attention_norm(x_norm_in, None, mode)
            x_norm_in.deallocate(True)
            attn_in_full = ttnn.all_gather(
                attn_in_sharded,
                dim=3,
                num_links=1,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            attn_in_sharded.deallocate(True)

            # --- Attention forward (full-H replicated in/out) ---
            attn_out = self.attention.forward(
                attn_in_full,
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
            attn_in_full.deallocate(True)
            # Some attention paths return rank-3 [B, T, H]; reshape to rank-4 for
            # the residual add to match x's [B, 1, T, H] layout.
            if len(list(attn_out.shape)) == 3:
                _B_a, _T_a, _H_a = list(attn_out.shape)
                attn_out_4d = ttnn.reshape(attn_out, ttnn.Shape([_B_a, 1, _T_a, _H_a]))
                attn_out.deallocate(True)
                attn_out = attn_out_4d

            # --- Residual add (full-H replicated) ---
            h_new = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)
            x.deallocate(True)
            attn_out.deallocate(True)

            # --- FF norm: scatter → norm (col-sharded) ---
            h_norm_in = ttnn.mesh_partition(h_new, dim=-1, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ff_in_sharded, _ = self.ff_norm(h_norm_in, None, mode)
            h_norm_in.deallocate(True)

            # --- MLP (col-sharded H/4 → col-sharded H/4) ---
            ff_out_sharded = self.feed_forward.forward(ff_in_sharded, mode, batch_size=batch_size)
            # Gather MLP output back to full-H for the residual add against h_new.
            ff_out_full = ttnn.all_gather(
                ff_out_sharded,
                dim=3,
                num_links=1,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ff_out_sharded.deallocate(True)

            # --- Final residual + exit scatter back to col-sharded H/4 ---
            out_full = ttnn.add(ff_out_full, h_new, memory_config=skip_mem_cfg)
            ff_out_full.deallocate(True)
            h_new.deallocate(True)
            # Decoder exit contract: col-sharded H/4 so the next layer / final
            # norm / lm_head sees the same layout as the embedding produces.
            out_sharded = ttnn.mesh_partition(out_full, dim=-1, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out_full.deallocate(True)
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
