# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniCPM Transformer class that extends tt_transformers Transformer with cross-attention layers.
"""

from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding, ScaledEmbedding
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.rope import RotarySetup
from ttnn_cross_attention import TtnnCrossAttention


class MiniCPMTransformer(LightweightModule):
    """
    MiniCPM Transformer that extends tt_transformers Transformer with cross-attention layers
    at specific positions (8, 16, 24) for multimodal fusion.
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
        cross_attention_layers=None,
    ):
        super().__init__()

        # Initialize parent class first
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        state_dict_prefix = args.get_state_dict_prefix("", None)

        self.tt_ccl = TT_CCL(self.mesh_device)

        embd_kwargs = {
            "mesh_device": mesh_device,
            "args": args,
            "weight_cache_path": args.weight_cache_path(dtype),
            "state_dict": state_dict,
            "dtype": ttnn.bfloat16,  # Row major layout requires bfloat16
        }
        if self.args.embed_scale is not None:
            embd_cls = ScaledEmbedding
            embd_kwargs["embed_scale"] = self.args.embed_scale
        else:
            embd_cls = Embedding
        self.embd = embd_cls(**embd_kwargs)

        ActualRopeSetupClass = rope_setup_class if rope_setup_class is not None else RotarySetup
        self.rope_setup = ActualRopeSetupClass(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )

        if args.rope_theta_local:
            self.rope_local_setup = RotarySetup(
                mesh_device,
                args.max_batch_size,
                args.head_dim,
                args.max_seq_len,
                args.rope_theta_local,
            )

        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        # Initialize transformer layers (same as parent class)
        self.layers = [
            TransformerBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                attention_class=attention_class,
            )
            for i in tqdm(range(self.n_layers))
        ]

        # Initialize cross-attention layers (MiniCPM specific)
        self.cross_attention_layers = cross_attention_layers or [8, 16, 24]
        self.cross_attn_modules = {}

        for layer_idx in self.cross_attention_layers:
            if layer_idx < self.n_layers:
                self.cross_attn_modules[layer_idx] = TtnnCrossAttention(
                    device=mesh_device,
                    hidden_size=args.hidden_size,
                    num_attention_heads=args.n_heads,
                    num_key_value_heads=args.n_kv_heads,
                    layer_idx=layer_idx,
                    model_args=args,  # Pass model args for proper configs
                )

        # Initialize output norm and lm_head (same as parent class)
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                args=args,
                weight_cache_path=args.weight_cache_path(dtype),
                state_dict=state_dict,
                layer_num=None,
                dtype=dtype,
            ),
            args,
        )

        self.lm_head = LMHead(
            device=mesh_device,
            args=args,
            weight_cache_path=args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=dtype,
        )

        # Initialize other attributes needed for forward pass
        self._initialize_other_attributes(args, mesh_device, state_dict)

    def _initialize_other_attributes(self, args, mesh_device, state_dict):
        """Initialize additional attributes needed for the model to work properly."""
        # These are copied from the parent Transformer class initialization
        # We need them for the forward pass to work correctly
        pass  # For now, we'll rely on the parent class attributes

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        encoder_hidden_states=None,  # MiniCPM specific: multimodal embeddings
    ):
        """
        Forward pass with cross-attention insertion at specific layers and full RoPE support.

        Args:
            encoder_hidden_states: Optional multimodal embeddings for cross-attention
                                  Shape: [batch_size, seq_len_enc, hidden_size]
        """
        for i, layer in enumerate(self.layers):
            # No-op if callers already provide the right memory config
            activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if mode == "decode" and not self.args.is_galaxy:
                x = ttnn.to_memory_config(x, self.model_config["DECODE_RESIDUAL_MEMCFG"], activation_dtype)
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            # Run normal transformer layer with full RoPE support
            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )

            # Insert cross-attention at specific layers (MiniCPM specific)
            if i in self.cross_attention_layers and encoder_hidden_states is not None:
                x = self.cross_attn_modules[i](x, encoder_hidden_states)

        if mode == "prefill" and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        x = self.norm(x, mode=mode)

        if mode == "prefill" and self.model_config["LM_HEAD_INPUT_MEMCFG"].is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.model_config["LM_HEAD_INPUT_MEMCFG"])

        x = self.lm_head(x)

        if mode == "prefill":
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x
