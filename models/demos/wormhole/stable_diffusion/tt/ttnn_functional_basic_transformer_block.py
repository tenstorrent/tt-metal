# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_cross_attention import cross_attention
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_feedforward import feedforward
import torch


def compare(tensor, name, permute=False):
    from models.utility_functions import comp_pcc

    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    tensor = ttnn.from_device(tensor)
    tensor = ttnn.to_torch(tensor)

    golden = torch.load(name)
    if permute:
        golden = golden.permute(0, 2, 3, 1)
        golden = golden.reshape(tensor.shape)

    while len(tensor.shape) > len(golden.shape):
        golden = golden.unsqueeze(0)
    while len(golden.shape) > len(tensor.shape):
        tensor = tensor.unsqueeze(0)

    passed, message = comp_pcc(tensor, golden, 0.95)
    print(f"Maches on {name}: {passed} with message {message}, tensor shape: {tensor.shape}")


class basic_transformer_block:
    def __init__(self, device, parameters, seq_len):
        self.device = device
        self.parameters = parameters
        self.cross_attention_1 = cross_attention(device, self.parameters.attn1, seq_len=seq_len)
        self.cross_attention_2 = cross_attention(device, self.parameters.attn2, seq_len=seq_len)
        self.ff = feedforward(device, parameters=self.parameters.ff)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
        config=None,
        num_embeds_ada_norm=False,
        norm_type: str = "layer_norm",
        cross_attention_dim: int = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        attention_bias: bool = False,
        attention_head_dim=None,
    ):
        use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if use_ada_layer_norm:
            assert False, "AdaLayerNorm not supported and not used in stable diffusion"
        elif use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        if hidden_states.memory_config().shard_spec.num_cores() == 64:
            end_grid = ttnn.experimental.tensor.CoreCoord(7, 7)
        elif hidden_states.memory_config().shard_spec.num_cores() == 40:
            end_grid = ttnn.experimental.tensor.CoreCoord(4, 7)
        elif hidden_states.memory_config().shard_spec.num_cores() == 32:
            end_grid = ttnn.experimental.tensor.CoreCoord(7, 3)

        sharded_mem_cfg = ttnn.get_memory_config(hidden_states)
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=1,
            block_h=sharded_mem_cfg.shard_spec.shape[0] // 32,
            block_w=sharded_mem_cfg.shard_spec.shape[1] // 32,
            inplace=False,
        )

        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=self.parameters.norm1.weight,
            bias=self.parameters.norm1.bias,
            memory_config=sharded_mem_cfg,
            program_config=program_config,
        )

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        cross_attention_dim = config.cross_attention_dim if cross_attention_dim is None else cross_attention_dim

        attn_output = self.cross_attention_1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if only_cross_attention else None,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            cross_attention_dim=cross_attention_dim,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        if use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        if attn_output.memory_config() != hidden_states.memory_config():
            if attn_output.memory_config().is_sharded():
                attn_output = ttnn.reshard(attn_output, hidden_states.memory_config())
            else:
                attn_output = ttnn.to_memory_config(attn_output, hidden_states.memory_config())
        sum = ttnn.add(attn_output, hidden_states, memory_config=hidden_states.memory_config())
        ttnn.deallocate(attn_output)
        ttnn.deallocate(hidden_states)
        if hidden_states.shape[-2] == 8192:
            hidden_states = ttnn.reallocate(sum)
        else:
            hidden_states = sum
        if cross_attention_dim is not None:
            norm_hidden_states = ttnn.layer_norm(
                hidden_states,
                epsilon=1e-05,
                weight=self.parameters.norm2.weight,
                bias=self.parameters.norm2.bias,
                memory_config=sharded_mem_cfg,
                program_config=program_config,
            )

            # 2. Cross-Attention
            attn_output = self.cross_attention_2(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                cross_attention_dim=cross_attention_dim,
                dim_head=attention_head_dim,
                upcast_attention=upcast_attention,
            )
            if attn_output.memory_config() != hidden_states.memory_config():
                if attn_output.memory_config().is_sharded():
                    attn_output = ttnn.reshard(attn_output, hidden_states.memory_config())
                else:
                    attn_output = ttnn.to_memory_config(attn_output, hidden_states.memory_config())
            sum = ttnn.add(attn_output, hidden_states, memory_config=hidden_states.memory_config())
            ttnn.deallocate(attn_output)
            ttnn.deallocate(hidden_states)
            if hidden_states.shape[-2] == 8192:
                hidden_states = ttnn.reallocate(sum)
            else:
                hidden_states = sum

        # 3. Feed-forward
        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=self.parameters.norm3.weight,
            bias=self.parameters.norm3.bias,
            memory_config=sharded_mem_cfg,
            program_config=program_config,
        )
        if use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        mem_cfg = hidden_states.memory_config()
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        if hidden_states.shape[-2] == 8192:
            hidden_states = ttnn.reallocate(hidden_states)
        norm_hidden_states = self.ff(config=config, hidden_states=norm_hidden_states)

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)
        norm_hidden_states = ttnn.to_memory_config(norm_hidden_states, mem_cfg)
        hidden_states = ttnn.add(norm_hidden_states, hidden_states, memory_config=hidden_states.memory_config())

        if hidden_states.memory_config().shard_spec.num_cores() == 64:
            end_grid = ttnn.experimental.tensor.CoreCoord(7, 7)
        elif hidden_states.memory_config().shard_spec.num_cores() == 40:
            end_grid = ttnn.experimental.tensor.CoreCoord(7, 4)
        elif hidden_states.memory_config().shard_spec.num_cores() == 32:
            end_grid = ttnn.experimental.tensor.CoreCoord(3, 7)
        else:
            assert False, f"Unsupported number of cores: {hidden_states.memory_config().shard_spec.num_cores()}"

        return hidden_states
