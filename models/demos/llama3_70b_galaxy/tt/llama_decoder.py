# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os
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
        # Optional decode stage capture for PCC localization (attn_out / ff_in / ff_out / out).
        self.stage_debug = os.getenv("QWEN_DECODER_STAGE_DEBUG", "0") == "1"
        self.stage_debug_tensors = {}
        self.unfuse_res_add = args.unfuse_res_add
        self.blackhole_no_prefetcher = (not getattr(args, "use_prefetcher", True)) and getattr(
            args, "is_blackhole", False
        )

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
        if self.blackhole_no_prefetcher:
            attn_norm_output_memcfg = (
                self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"]
                if layer_num == 0
                else self.model_config["DECODE_RESIDUAL_MEMCFG"]
            )
        else:
            attn_norm_output_memcfg = self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"]
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
                output_mem_config=attn_norm_output_memcfg,
            ),
            args,
            tt_ccl=tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
            use_sharded_decode=not self.blackhole_no_prefetcher,
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
            use_sharded_decode=not self.blackhole_no_prefetcher,
        )

    def _stage_mag(self, name, tensor):
        """Log mean/max abs of a gathered decode tensor (any width) for zero-output debugging."""
        if not self.stage_debug:
            return
        try:
            pass

            full = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(1, 3), mesh_shape=self.args.cluster_shape
                ),
            ).float()
            from loguru import logger as _logger

            _logger.info(
                f"[decoder-stage-mag] {name}: shape={tuple(full.shape)} "
                f"mean_abs={float(full.abs().mean()):.6e} max_abs={float(full.abs().max()):.6e}"
            )
        except Exception as exc:
            from loguru import logger as _logger

            _logger.info(f"[decoder-stage-mag] {name}: failed {exc}")

    def _stage_capture(self, name, tensor):
        """Gather a [1,1,32,dim/4] column-fractured decode tensor to full [32, dim] on host."""
        if not self.stage_debug:
            return
        try:
            full = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(1, 3), mesh_shape=self.args.cluster_shape
                ),
            )[:, 0:1, : self.max_batch_size, : self.dim].reshape(-1, 1, self.dim)
            self.stage_debug_tensors[name] = full.float()
        except Exception as exc:  # diagnostics must never break the forward
            self.stage_debug_tensors[name] = f"capture failed: {exc}"

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        self.attention.prefetch(prefetcher_setup, tt_ccl)
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
        if mode == "decode":
            # In no-prefetcher Blackhole demo runs, layer-0 decode input can remain DRAM/interleaved.
            # Let downstream norm path handle this instead of hard-failing at the boundary check.
            if getattr(self.args, "use_prefetcher", True):
                assert (
                    x.memory_config() == skip_mem_cfg
                ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"
        else:
            assert (
                x.memory_config() == skip_mem_cfg
            ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"
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
                if getattr(self.args, "use_prefetcher", True):
                    h = ttnn.add(attn_out, h)
                else:
                    # No-prefetcher decode may mix DRAM/interleaved residuals with sharded attn output.
                    # Align inputs for binary add; if sharded alignment fails, fall back to DRAM add.
                    try:
                        if h.memory_config() != attn_out.memory_config():
                            h = ttnn.to_memory_config(h, attn_out.memory_config())
                        h = ttnn.add(attn_out, h)
                    except RuntimeError:
                        if attn_out.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                            attn_out = ttnn.to_memory_config(attn_out, ttnn.DRAM_MEMORY_CONFIG)
                        if h.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                            h = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
                        h = ttnn.add(attn_out, h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ff_in_sharded, _ = self.ff_norm(h, None, mode)
            elif not getattr(self.args, "use_prefetcher", True) and getattr(self.args, "is_blackhole", False):
                # BH no-prefetch decode: the residual stream is column-fractured (dim/4 per
                # device). The distributed ff_norm normalizes (attn_out + h) over the full
                # hidden dim but does not return the summed residual, so build the post-attention
                # residual h = h + attn_out explicitly for the final residual add.
                self._stage_capture("attn_out", attn_out)
                ff_in_sharded, _ = self.ff_norm(attn_out, h, mode)
                self._stage_capture("ff_in", ff_in_sharded)
                attn_res = (
                    attn_out
                    if attn_out.memory_config() == h.memory_config()
                    else ttnn.to_memory_config(attn_out, h.memory_config())
                )
                h = ttnn.add(h, attn_res)
                self._stage_capture("h_residual", h)
            else:
                ff_in_sharded, _ = self.ff_norm(attn_out, h, mode)
            if h is not attn_out:
                attn_out.deallocate(True)

        # MLP takes replicated inputs and produces fractured outputs
        if self.stage_debug and mode == "decode":
            self._stage_mag("mlp_in", ff_in_sharded)
            ff_out, _mlp_interm = self.feed_forward.forward(
                ff_in_sharded, mode, batch_size=batch_size, return_intermediates=True
            )
            for _k in ("ff1_reduced", "ff3_reduced", "activation", "ff2_input", "ff2_pre_allreduce", "ff2_output"):
                if _k in _mlp_interm:
                    self._stage_mag(f"mlp_{_k}", _mlp_interm[_k])
        else:
            ff_out = self.feed_forward.forward(ff_in_sharded, mode, batch_size=batch_size)
        self._stage_capture("ff_out", ff_out)
        if self.layer_num == self.n_layers - 1 or mode == "prefill":
            out = ttnn.add(ff_out, h, memory_config=skip_mem_cfg)  # , dtype=ttnn.bfloat16)
            if mode == "decode":
                ff_out.deallocate(True)
            if mode == "prefill":
                h.deallocate(True)
            return out, None
        else:
            return ff_out, h
