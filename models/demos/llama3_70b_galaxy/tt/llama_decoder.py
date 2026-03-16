# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm


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
        self.enable_trace = False  # Set to True during trace capture to skip input deallocation
        self.is_olmo = getattr(args, "is_olmo", False)
        self.capture_intermediates = False  # Set to True to capture intermediates for PCC debug
        self.captured = {}  # Stores reconstructed torch tensors when capture_intermediates=True

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
        )

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        self.attention.prefetch(prefetcher_setup, tt_ccl)
        self.feed_forward.prefetch(prefetcher_setup, tt_ccl)
        self.attention_norm.tt_ccl = tt_ccl
        self.ff_norm.tt_ccl = tt_ccl

    def _capture(self, name, tensor):
        """Capture a tensor as CPU torch tensor for PCC comparison. Only active when capture_intermediates=True."""
        if not self.capture_intermediates:
            return
        try:
            t = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(3, 1), mesh_shape=self.args.cluster_shape
                ),
            ).float()
            self.captured[name] = t
        except Exception as e:
            from loguru import logger

            logger.warning(f"  [capture] {name}: failed to reconstruct — {e}")

    def _capture_prefill(self, name, tensor):
        """Capture a prefill tensor for PCC comparison. Only active when capture_intermediates=True.

        In prefill, tensors are sharded across mesh dim 1 (hidden dim) and replicated
        across mesh dim 0. Concat across mesh dim 1 along tensor dim 3 to reconstruct
        the full hidden dim; mesh dim 0 copies are stacked along tensor dim 1.
        """
        if not self.capture_intermediates:
            return
        try:
            t = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(1, 3), mesh_shape=self.args.cluster_shape
                ),
            ).float()
            self.captured[name] = t
        except Exception as e:
            from loguru import logger

            logger.warning(f"  [capture_prefill] {name}: failed to reconstruct — {e}")

    def _debug_check(self, name, tensor):
        """Check tensor for Inf/NaN and log stats."""
        import os

        if os.environ.get("DEBUG_DECODE", "0") != "1":
            return
        try:
            import torch
            from loguru import logger

            # Get tensor from mesh device
            torch_tensor = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(1, 3), mesh_shape=self.args.cluster_shape
                ),
            )
            has_inf = torch.isinf(torch_tensor).any().item()
            has_nan = torch.isnan(torch_tensor).any().item()
            max_val = torch_tensor.float().abs().max().item()
            status = "OK" if not (has_inf or has_nan) else "BAD"
            logger.info(f"  [{status}] {name}: max={max_val:.4e}, Inf={has_inf}, NaN={has_nan}")
        except Exception as e:
            import traceback
            from loguru import logger

            logger.error(f"  [ERROR] {name}: {e}\n{traceback.format_exc()}")

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
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        # x contains input in layer 0 and ffout of previous layer thereafter, x should be dealocated
        # h contains 0 in layer 0 and h_prev+x_prev+attn_out_prev thereafter, h is persistent
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        self._debug_check("input_x", x)

        # OLMo3 post-sublayer-norm prefill path:
        # Norm is applied AFTER each sublayer (attention/MLP), BEFORE the residual add.
        # Weight mapping note (from load_checkpoints.py):
        #   self.ff_norm        has post_attention_layernorm weights  → applied after attention
        #   self.attention_norm has post_feedforward_layernorm weights → applied after FFN
        if self.is_olmo and mode == "prefill":
            attn_out = self.attention.forward(
                x,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                batch_size=batch_size,
                skip_input_dealloc=True,  # keep x alive for residual add
            )
            self._debug_check("attn_out", attn_out)
            self._capture_prefill("attn_out", attn_out)
            attn_out_normed, _ = self.ff_norm(attn_out, None, mode)  # post-attention norm
            self._capture_prefill("attn_normed", attn_out_normed)
            h = ttnn.add(x, attn_out_normed, memory_config=skip_mem_cfg)
            self._capture_prefill("h_attn", h)
            if not self.enable_trace:
                x.deallocate(True)
            ff_out = self.feed_forward.forward(h, mode, batch_size=batch_size, skip_input_dealloc=True)
            self._debug_check("ff_out", ff_out)
            self._capture_prefill("ff_out", ff_out)
            ff_out_normed, _ = self.attention_norm(ff_out, None, mode)  # post-FFN norm
            self._capture_prefill("ff_normed", ff_out_normed)
            out = ttnn.add(h, ff_out_normed, memory_config=skip_mem_cfg)
            self._capture_prefill("layer_out", out)
            h.deallocate(True)
            return out, None

        elif self.is_olmo and mode == "decode":
            if h is not None:
                x_combined = ttnn.add(x, h)
                if not self.enable_trace:
                    x.deallocate(True)
            else:
                x_combined = x

            x_res_dram = ttnn.to_memory_config(x_combined, ttnn.DRAM_MEMORY_CONFIG)
            if not self.enable_trace and h is not None:
                x_combined.deallocate(True)
            attn_in = ttnn.to_memory_config(x_res_dram, self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"])

            attn_out = self.attention.forward(
                attn_in,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
            self._capture("attn_out", attn_out)

            attn_normed, _ = self.ff_norm(attn_out, None, mode)
            self._capture("attn_normed", attn_normed)
            attn_normed_dram = ttnn.to_memory_config(attn_normed, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_normed)

            h_attn_dram = ttnn.add(x_res_dram, attn_normed_dram)
            self._capture("h_attn", h_attn_dram)
            ttnn.deallocate(attn_normed_dram)
            ttnn.deallocate(x_res_dram)

            ff_in = ttnn.to_memory_config(h_attn_dram, self.model_config["SHARDED_FF12_RING_MEMCFG"])

            ff_out = self.feed_forward.forward(ff_in, mode, batch_size=batch_size)
            self._capture("ff_out", ff_out)

            ff_normed, _ = self.attention_norm(ff_out, None, mode)
            self._capture("ff_normed", ff_normed)
            ff_normed_dram = ttnn.to_memory_config(ff_normed, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ff_normed)

            out_dram = ttnn.add(h_attn_dram, ff_normed_dram)
            ttnn.deallocate(h_attn_dram)
            ttnn.deallocate(ff_normed_dram)

            out = ttnn.to_memory_config(out_dram, skip_mem_cfg)
            ttnn.deallocate(out_dram)
            self._capture("layer_out", out)

            return out, None

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

        self._debug_check("attn_in_sharded", attn_in_sharded)

        attn_out = self.attention.forward(
            attn_in_sharded,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )

        self._debug_check("attn_out", attn_out)

        if mode == "prefill":
            h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)  # bfloat8_b
            if not self.enable_trace:  # Skip deallocation during trace capture
                x.deallocate(True)
            ff_in_sharded, _ = self.ff_norm(h, None, mode)
        if mode == "decode":
            if self.unfuse_res_add:
                h = ttnn.add(attn_out, h)
                ff_in_sharded, _ = self.ff_norm(h, None, mode)
            else:
                ff_in_sharded, _ = self.ff_norm(attn_out, h, mode)
            attn_out.deallocate(True)

        self._debug_check("ff_in_sharded", ff_in_sharded)

        # MLP takes replicated inputs and produces fractured outputs
        ff_out = self.feed_forward.forward(ff_in_sharded, mode, batch_size=batch_size)

        self._debug_check("ff_out", ff_out)
        if self.layer_num == self.n_layers - 1 or mode == "prefill":
            out = ttnn.add(ff_out, h, memory_config=skip_mem_cfg)  # , dtype=ttnn.bfloat16)
            if mode == "decode":
                ff_out.deallocate(True)
            if mode == "prefill":
                h.deallocate(True)
            return out, None
        else:
            return ff_out, h
