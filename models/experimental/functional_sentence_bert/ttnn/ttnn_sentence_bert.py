# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional

SDPAProgramConfig = ttnn._ttnn.operations.transformer.SDPAProgramConfig


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class ttnn_BertEmbeddings:
    def __init__(self, parameters):
        self.parameters = parameters
        self.word_embeddings = ttnn.embedding
        self.position_embeddings = ttnn.embedding
        self.token_type_embeddings = ttnn.embedding
        self.LayerNorm = ttnn.layer_norm
        self.add = ttnn.add

    def __call__(self, input_ids: ttnn.Tensor, token_type_ids: ttnn.Tensor, position_ids: ttnn.Tensor, device):
        inputs_embeds = self.word_embeddings(
            input_ids,
            weight=self.parameters.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        p(inputs_embeds, "after word")
        token_type_embeddings = self.token_type_embeddings(
            token_type_ids, self.parameters.token_type_embeddings.weight, layout=ttnn.TILE_LAYOUT
        )
        p(token_type_embeddings, "after token")
        position_embeddings = self.position_embeddings(
            position_ids, self.parameters.position_embeddings.weight, layout=ttnn.TILE_LAYOUT
        )
        p(position_embeddings, "after position")
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings

        embeddings = self.LayerNorm(
            embeddings, weight=self.parameters.LayerNorm.weight, bias=self.parameters.LayerNorm.bias
        )
        return embeddings


class ttnn_BertOutput:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.dense = ttnn.linear
        self.LayerNorm = ttnn.layer_norm
        self.add = ttnn.add

    def __call__(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor):
        hidden_states = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = self.add(hidden_states, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = self.LayerNorm(
            hidden_states,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        )
        # print("ttnn out of BertOutput is ",hidden_states.shape)
        return hidden_states


class ttnn_BertIntermediate:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.intermediate_act_fn = ttnn.gelu
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        hidden_states = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ttnn_BertSelfOutput:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.dense = ttnn.linear
        self.LayerNorm = ttnn.layer_norm
        self.add = ttnn.add

    def __call__(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor):
        hidden_states = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = self.add(hidden_states, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = self.LayerNorm(
            hidden_states,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        )
        # print("ttnn out ofBertSelfOutput is ",hidden_states.shape)
        return hidden_states


class ttnn_BertPooler:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.activation = ttnn.tanh
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        print("pooler forward is called", hidden_states.shape)
        first_token_tensor = hidden_states[:, 0, :]
        print("pooler forward first token is called", first_token_tensor.shape)
        pooled_output = self.dense(
            first_token_tensor,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ttnn_BertSelfAttention:
    # init - 3 linear's,
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.query = ttnn.linear
        self.key = ttnn.linear
        self.value = ttnn.linear
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        device=None,
    ):
        query_layer = self.query(
            hidden_states,
            self.parameters.query.weight,
            bias=self.parameters.query.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        query_layer = ttnn.reshape(
            query_layer,
            (query_layer.shape[0], query_layer.shape[1], self.num_attention_heads, self.attention_head_size),
        )
        query_layer = ttnn.permute(query_layer, (0, 2, 1, 3))
        # query_layer = ttnn.reshape(
        #     query_layer,
        #     (1, query_layer.shape[0]*query_layer.shape[1], query_layer.shape[2], query_layer.shape[3]),
        # )
        key_layer = self.key(
            hidden_states,
            self.parameters.key.weight,
            bias=self.parameters.key.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        key_layer = ttnn.reshape(
            key_layer, (key_layer.shape[0], key_layer.shape[1], self.num_attention_heads, self.attention_head_size)
        )
        key_layer = ttnn.permute(key_layer, (0, 2, 1, 3))
        # key_layer = ttnn.reshape(
        #     key_layer, (1,key_layer.shape[0]*key_layer.shape[1], key_layer.shape[2], key_layer.shape[3])
        # )
        value_layer = self.value(
            hidden_states,
            self.parameters.value.weight,
            bias=self.parameters.value.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        value_layer = ttnn.reshape(
            value_layer,
            (value_layer.shape[0], value_layer.shape[1], self.num_attention_heads, self.attention_head_size),
        )
        value_layer = ttnn.permute(value_layer, (0, 2, 1, 3))
        # value_layer = ttnn.reshape(
        #     value_layer,
        #     (1,value_layer.shape[0]*value_layer.shape[1], value_layer.shape[2], value_layer.shape[3]),
        # )
        query_layer = ttnn.to_memory_config(query_layer, ttnn.DRAM_MEMORY_CONFIG)
        key_layer = ttnn.to_memory_config(key_layer, ttnn.DRAM_MEMORY_CONFIG)
        value_layer = ttnn.to_memory_config(value_layer, ttnn.DRAM_MEMORY_CONFIG)
        p(query_layer, "query")
        p(key_layer, "key")
        p(value_layer, "value")
        p(attention_mask, "mask")
        # query_layer = ttnn.to_torch(query_layer)
        # key_layer = ttnn.to_torch(key_layer)
        # value_layer = ttnn.to_torch(value_layer)
        # attention_mask = ttnn.to_torch(attention_mask)
        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_layer,
        #     key_layer,
        #     value_layer,
        #     attn_mask=attention_mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        # )
        # program_config = ttnn.SDPAProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     q_chunk_size=32,
        #     k_chunk_size=32,
        #     exp_approx_mode=True,
        # )
        # compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        #     math_fidelity=ttnn.MathFidelity.HiFi4,
        #     math_approx_mode=False,
        #     fp32_dest_acc_en=True,
        #     packer_l1_acc=False,
        # )
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            is_causal=False,
            # program_config =program_config,
            # compute_kernel_config=compute_kernel_config
        )
        #
        # attn_output = ttnn.from_torch(attn_output,device=device,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG)
        p(attn_output, "aftr scaled_dot_product_attention")
        # attn_output = ttnn.reshape(
        #     attn_output, (2,12,8,64)
        # )
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(
            attn_output, (attn_output.shape[0], attn_output.shape[1], attn_output.shape[2] * attn_output.shape[3])
        )
        p(attn_output, "attnn output")
        return (attn_output,)


class ttnn_BertAttention:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.self = ttnn_BertSelfAttention(parameters.self, config)
        self.output = ttnn_BertSelfOutput(parameters.output, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        # head_mask: Optional[ttnn.Tensor] = None,
        # encoder_hidden_states: Optional[ttnn.Tensor] = None,
        # encoder_attention_mask: Optional[ttnn.Tensor] = None,
        device=None,
    ):
        self_outputs = self.self(hidden_states, attention_mask, device=device)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ttnn_BertLayer:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.attention = ttnn_BertAttention(parameters.attention, config)
        self.intermediate = ttnn_BertIntermediate(parameters.intermediate)
        self.output = ttnn_BertOutput(parameters.output, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        # head_mask: Optional[ttnn.Tensor] = None,
        # encoder_hidden_states: Optional[ttnn.Tensor] = None,
        # encoder_attention_mask: Optional[ttnn.Tensor] = None,
        device=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, device=device)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class ttnn_BertEncoder:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.layers = {}
        for i in range(config.num_hidden_layers):
            self.layers[i] = ttnn_BertLayer(parameters.layer[i], config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        # head_mask: Optional[ttnn.Tensor] = None,
        # encoder_hidden_states: Optional[ttnn.Tensor] = None,
        # encoder_attention_mask: Optional[ttnn.Tensor] = None,
        # return_dict: Optional[bool] = True,
        device=None,
    ):
        for i in range(len(self.layers)):
            layer_outputs = self.layers[i](hidden_states, attention_mask, device=device)
            print("tttnn layer out", layer_outputs[0].shape)
            hidden_states = layer_outputs[0]
            # torch.save(ttnn.to_torch(hidden_states),f"/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_sentence_bert/dumps/ttnn_out_{i}.pth")
        print("tt out of encoder", hidden_states.shape)
        return hidden_states


class ttnn_BertModel:
    def __init__(self, parameters, config):
        self.embeddings = ttnn_BertEmbeddings(parameters.embeddings)
        self.encoder = ttnn_BertEncoder(parameters.encoder, config)
        self.pooler = ttnn_BertPooler(parameters.pooler)

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        token_type_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        device=None,
    ):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, device=device)
        sequence_output = self.encoder(embedding_output, attention_mask, device=device)
        pooled_output = self.pooler(sequence_output)
        return (sequence_output, pooled_output)


def preprocess_inputs(
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device,
):
    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    attention_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return input_ids, token_type_ids, position_ids, attention_mask
