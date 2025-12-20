# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm

import torch
import torch.nn as nn
import os


class TorchRMSNorm(torch.nn.Module):
    """Torch implementation of RMSNorm for debugging purposes."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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
        reference_model=None,
    ):
        super().__init__()

        self.reference_model = reference_model

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
            reference_model=reference_model,
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

    def torch_rms_norm(self, x, res, norm_module, output_mem_config, mode):
        """
        Apply torch RMSNorm instead of ttnn distributed norm.
        Converts ttnn tensor to torch, applies torch RMSNorm, converts back to ttnn.

        Args:
            x: Input ttnn tensor
            res: Residual ttnn tensor (can be None)
            norm_module: The DistributedNorm module containing weights and eps
            output_mem_config: Memory config for output tensor
            mode: "decode" or "prefill"

        Returns:
            Tuple of (normalized output, updated residual h)
        """
        mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])

        # Convert x to torch
        x_torch = ttnn.to_torch(x, mesh_composer=mesh_composer)[:1, :, :, :]
        x_torch = torch.permute(x_torch, (0, 2, 1, 3))  # [1, seq, batch, dim] -> [1, batch, seq, dim]
        x_torch = x_torch.squeeze(0)  # [batch, seq, dim]

        # If residual is provided, add it first (fused residual add)
        if res is not None:
            res_torch = ttnn.to_torch(res, mesh_composer=mesh_composer)[:1, :, :, :]
            res_torch = torch.permute(res_torch, (0, 2, 1, 3))
            res_torch = res_torch.squeeze(0)
            x_torch = x_torch + res_torch
            # Update h (residual) with the sum
            h_updated_torch = x_torch.clone()
        else:
            h_updated_torch = None

        # Get weights from the state_dict directly (more reliable than extracting from ttnn tensor)
        # Determine the weight key based on which norm we're using
        state_dict_prefix = self.args.get_state_dict_prefix("", self.layer_num)
        if norm_module is self.attention_norm:
            weight_key = "attention_norm"
        else:
            weight_key = "ffn_norm"
        weight_name = f"{state_dict_prefix}{weight_key}.weight"
        weight_torch = self.state_dict[weight_name].clone()  # Shape: [dim]

        eps = norm_module.norm.eps

        # Create torch RMSNorm and set weights
        torch_norm = TorchRMSNorm(dim=weight_torch.shape[0], eps=eps)
        with torch.no_grad():
            torch_norm.weight.copy_(weight_torch)

        # Apply torch RMSNorm
        output_torch = torch_norm(x_torch)

        # Convert back to ttnn format: [batch, seq, dim] -> [seq, 1, batch, dim]
        output_torch = output_torch.unsqueeze(0)  # [1, batch, seq, dim]
        output_torch = torch.permute(output_torch, (0, 2, 1, 3))  # [1, seq, batch, dim]

        # Shard back to mesh devices
        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=list(self.mesh_device.shape))

        output_ttnn = ttnn.from_torch(
            output_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=output_mem_config,
        )

        # Also convert h_updated back to ttnn if it was updated
        if h_updated_torch is not None:
            h_updated_torch = h_updated_torch.unsqueeze(0)
            h_updated_torch = torch.permute(h_updated_torch, (0, 2, 1, 3))
            h_updated = ttnn.from_torch(
                h_updated_torch,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"]
                if mode == "decode"
                else ttnn.DRAM_MEMORY_CONFIG,
            )
            return output_ttnn, h_updated
        else:
            return output_ttnn, None

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
        # Norms take fractured inputs and output replicated across devices
        # attn_in_sharded=norm(x+h), h = x+h happens implicitly
        # breakpoint()

        if os.environ.get("DEBUG_PCC") == "1":
            print(f"Layer nr: {self.layer_num}")

        # Check if we should use torch norm instead of ttnn norm
        use_torch_norm = os.environ.get("USE_TORCH_NORM") == "1"

        if self.layer_num == 0 or mode == "prefill":
            # In the first layer we "make" the h tensor from the original x keeping it alive
            # Note this works because layer 0 has a bfloat16 input while other layers use bfloat8
            # since we want residual to be bfloat16
            # if use_torch_norm:
            #     attn_in_sharded, _ = self.torch_rms_norm(
            #         x, None, self.attention_norm,
            #         self.attention_norm.norm.output_mem_config, mode
            #     )
            # else:
            attn_in_sharded, _ = self.attention_norm(x, None, mode)
            h = x

        else:
            # In subsequent Layers we take the h tensor from before and modify it in place
            if self.unfuse_res_add:
                h = ttnn.add(x, h, dtype=ttnn.bfloat16)

                # debug pcc
                # if self.reference_model is not None and os.environ.get("DEBUG_PCC") == "1":
                #     ref_after_ffn_add = self.reference_model.layers[self.layer_num - 1].ref_after_ffn_add
                #     # convert to torch
                #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
                #     comp_out = ttnn.to_torch(h, mesh_composer=mesh_composer)[:1, :, :, :]
                #     comp_out = torch.permute(comp_out, (0, 2, 1, 3))
                #     comp_out = comp_out.squeeze(0)
                #     passing, pcc_message = comp_pcc(ref_after_ffn_add, comp_out)
                #     print(f"MLP add PCC: {pcc_message}")
                #     print(comp_allclose(ref_after_ffn_add, comp_out))
                #     print()

                # if use_torch_norm:
                #     attn_in_sharded, _ = self.torch_rms_norm(
                #         h, None, self.attention_norm,
                #         self.attention_norm.norm.output_mem_config, mode
                #     )
                # else:
                attn_in_sharded, _ = self.attention_norm(h, None, mode)
            else:
                if use_torch_norm:
                    attn_in_sharded, h = self.torch_rms_norm(
                        x, h, self.attention_norm, self.attention_norm.norm.output_mem_config, mode
                    )
                else:
                    attn_in_sharded, _ = self.attention_norm(x, h, mode)

        # debug pcc
        # if self.reference_model is not None and os.environ.get("DEBUG_PCC") == "1":
        #     ref_after_attention_norm = self.reference_model.layers[self.layer_num].ref_after_attention_norm
        #     # convert to torch
        #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
        #     comp_out = ttnn.to_torch(attn_in_sharded, mesh_composer=mesh_composer)[:1, :, :, :]
        #     comp_out = torch.permute(comp_out, (0, 2, 1, 3))
        #     comp_out = comp_out.squeeze(0)
        #     passing, pcc_message = comp_pcc(ref_after_attention_norm, comp_out)
        #     print(f"Attention norm PCC: {pcc_message}")
        #     print(comp_allclose(ref_after_attention_norm, comp_out))
        #     print()

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

        # debug pcc
        # if self.reference_model is not None and os.environ.get("DEBUG_PCC") == "1":
        #     ref_after_attention = self.reference_model.layers[self.layer_num].ref_after_attention
        #     # convert to torch
        #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
        #     comp_out = ttnn.to_torch(attn_out, mesh_composer=mesh_composer)[:1, :, :, :]
        #     comp_out = torch.permute(comp_out, (0, 2, 1, 3))
        #     comp_out = comp_out.squeeze(0)
        #     passing, pcc_message = comp_pcc(ref_after_attention, comp_out)
        #     print(f"Attention out PCC: {pcc_message}")
        #     print(comp_allclose(ref_after_attention, comp_out))
        #     print()

        if mode == "prefill":
            h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)  # bfloat8_b
            x.deallocate(True)
            # if use_torch_norm:
            #     ff_in_sharded, _ = self.torch_rms_norm(
            #         h, None, self.ff_norm,
            #         self.ff_norm.norm.output_mem_config, mode
            #     )
            # else:
            ff_in_sharded, _ = self.ff_norm(h, None, mode)

        if mode == "decode":
            if self.unfuse_res_add:
                h = ttnn.add(attn_out, h, dtype=ttnn.bfloat16)

            # debug pcc
            # if self.reference_model is not None and os.environ.get("DEBUG_PCC") == "1":
            #     # ref_after_attention_add = self.reference_model.layers[self.layer_num].ref_after_attention_add
            #     # # convert to torch
            #     # mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
            #     # comp_out = ttnn.to_torch(h, mesh_composer=mesh_composer)[:1, :, :, :]
            #     # comp_out = torch.permute(comp_out, (0, 2, 1, 3))
            #     # comp_out = comp_out.squeeze(0)
            #     # passing, pcc_message = comp_pcc(ref_after_attention_add, comp_out)
            #     # print(f"Attention add PCC: {pcc_message}")
            #     # print(comp_allclose(ref_after_attention_add, comp_out))
            #     # print()

            #     if use_torch_norm:
            #         ff_in_sharded, _ = self.torch_rms_norm(
            #             h, None, self.ff_norm,
            #             self.ff_norm.norm.output_mem_config, mode
            #         )
            #     else:
            #         ff_in_sharded, _ = self.ff_norm(h, None, mode)
            # else:
            #     if use_torch_norm:
            #         ff_in_sharded, h = self.torch_rms_norm(
            #             attn_out, h, self.ff_norm,
            #             self.ff_norm.norm.output_mem_config, mode
            #         )
            #     else:
            ff_in_sharded, _ = self.ff_norm(h, None, mode)
            attn_out.deallocate(True)

        # debug pcc
        # if self.reference_model is not None and os.environ.get("DEBUG_PCC") == "1":
        #     ref_after_ffn_norm = self.reference_model.layers[self.layer_num].ref_after_ffn_norm
        #     # convert to torch
        #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
        #     comp_out = ttnn.to_torch(ff_in_sharded, mesh_composer=mesh_composer)[:1, :, :, :]
        #     comp_out = torch.permute(comp_out, (0, 2, 1, 3))
        #     comp_out = comp_out.squeeze(0)
        #     passing, pcc_message = comp_pcc(ref_after_ffn_norm, comp_out)
        #     print(f"MLP norm PCC: {pcc_message}")
        #     print(comp_allclose(ref_after_ffn_norm, comp_out))
        #     print()

        # MLP takes replicated inputs and produces fractured outputs
        ff_out = self.feed_forward.forward(ff_in_sharded, mode)

        # debug pcc
        # if self.reference_model is not None and os.environ.get("DEBUG_PCC") == "1":
        #     ref_after_feed_forward = self.reference_model.layers[self.layer_num].ref_after_feed_forward
        #     # convert to torch
        #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
        #     comp_out = ttnn.to_torch(ff_out, mesh_composer=mesh_composer)[:1, :, :, :]
        #     comp_out = torch.permute(comp_out, (0, 2, 1, 3))
        #     comp_out = comp_out.squeeze(0)
        #     passing, pcc_message = comp_pcc(ref_after_feed_forward, comp_out)
        #     print(f"MLP out PCC: {pcc_message}")
        #     print(comp_allclose(ref_after_feed_forward, comp_out))
        #     print()

        if self.layer_num == self.n_layers - 1 or mode == "prefill":
            out = ttnn.add(ff_out, h, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16)

            # debug pcc
            # if self.reference_model is not None and os.environ.get("DEBUG_PCC") == "1":
            #     ref_after_ffn_add = self.reference_model.layers[self.layer_num].ref_after_ffn_add
            #     # convert to torch
            #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
            #     comp_out = ttnn.to_torch(out, mesh_composer=mesh_composer)[:1, :, :, :]
            #     comp_out = torch.permute(comp_out, (0, 2, 1, 3))
            #     comp_out = comp_out.squeeze(0)
            #     print()

            if mode == "decode":
                ff_out.deallocate(True)
            if mode == "prefill":
                h.deallocate(True)
            return out, None
        else:
            return ff_out, h
