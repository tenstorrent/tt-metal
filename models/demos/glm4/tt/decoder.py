# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.decoder import TransformerBlock as BaseTransformerBlock
from models.demos.glm4.tt.attention import Glm4Attention

# Import necessary norm classes
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm

# Import MLP, might need a Glm4MLP later if MLP logic differs
from models.tt_transformers.tt.mlp import MLP

# Import Glm4ModelArgs for type hint
from models.demos.glm4.tt.model_config import Glm4ModelArgs


class Glm4TransformerBlock(BaseTransformerBlock):
    """
    glm4 specific transformer block, inheriting from base block
    implements post-norm architecture and uses glm4 specific components
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        args: Glm4ModelArgs,
        layer_num,
        weight_cache_path,
        transformation_mats,
        dtype,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        # Initialize base class WITHOUT passing args directly, to avoid base __init__ logic we override
        # Instead, pass individual necessary attributes if super().__init__ requires them
        # Or, more simply, call LightweightModule.__init__() directly if BaseTransformerBlock only adds components
        # super().__init__(...) # Avoid calling base __init__ for now, initialize components directly
        # Initialize parent LightweightModule
        super(BaseTransformerBlock, self).__init__()

        # Store args and basic config
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.model_config = args.get_model_config()
        self.layer_num = layer_num

        # Initialize standard norms (attention_norm, ff_norm)
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
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
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
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )

        # Initialize GLM-4 specific post-attention and post-mlp norms
        self.post_attention_layernorm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="post_attention_layernorm",  # GLM-4 specific key
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )
        self.post_mlp_layernorm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="post_mlp_layernorm",  # GLM-4 specific key
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )

        # Use Glm4Attention instead of base Attention
        self.attention = Glm4Attention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,  # Pass Glm4ModelArgs here
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        # Use base MLP for now, create Glm4MLP if needed later
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # GLM-4 post-norm architecture:
        # 1. Apply input norm
        # 2. Apply attention (Glm4Attention)
        # 3. Apply post-attention norm
        # 4. Add residual (after norm)
        # Repeat similar pattern for MLP

        # Store residual for later
        residual = x

        # Input layernorm
        attn_in = self.attention_norm(x, mode)

        # Attention module (using self.attention which is Glm4Attention)
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
        )

        # Apply post-attention norm to the attention output
        attn_norm_out = self.post_attention_layernorm(attn_out, mode)
        ttnn.deallocate(attn_out)

        # Add residual after norm
        h = ttnn.add(residual, attn_norm_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None)
        if mode == "prefill":
            # Only deallocate original input in prefill if it's truly not needed
            # In post-norm, the original residual 'x' is used in the first add
            # Let's keep residual alive until after the first add
            pass  # residual = x is still needed
        ttnn.deallocate(attn_norm_out)

        # Store new residual for MLP path
        residual = h  # Note: h holds x + norm(attn(norm(x)))

        # Pre-MLP norm
        ff_in = self.ff_norm(h, mode)
        if TG and mode == "decode":
            ff_in = ttnn.to_memory_config(ff_in, memory_config=self.model_config["MLP_ACT_MEMCFG"])

        # MLP
        ff_out = self.feed_forward.forward(ff_in, mode)

        # Apply post-MLP norm
        ff_norm_out = self.post_mlp_layernorm(ff_out, mode)
        ttnn.deallocate(ff_out)

        # Add residual after norm
        out = ttnn.add(
            residual,  # h = x + norm(attn(norm(x)))
            ff_norm_out,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else self.model_config["ACTIVATION_DTYPE"] or ttnn.bfloat16,
        )
        ttnn.deallocate(ff_norm_out)
        ttnn.deallocate(residual)  # Now we can deallocate h

        return out
