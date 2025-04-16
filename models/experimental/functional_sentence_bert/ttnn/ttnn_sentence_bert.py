# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional

SDPAProgramConfig = ttnn._ttnn.operations.transformer.SDPAProgramConfig

layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    subblock_w=3,
    block_h=12,
    block_w=3,
    inplace=True,
)
ff1_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=3,
    out_subblock_h=1,
    out_subblock_w=8,
    per_core_M=12,
    per_core_N=16,
    transpose_mcast=True,
    fused_activation=(ttnn.UnaryOpType.GELU, True),
)
f_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=True,
    fused_activation=None,
)
query_key_value_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=3,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=True,
    fused_activation=None,
)


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class ttnn_BertEmbeddings:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.word_embeddings = ttnn.embedding
        self.position_embeddings = ttnn.embedding
        self.token_type_embeddings = ttnn.embedding
        self.LayerNorm = ttnn.layer_norm
        self.add = ttnn.add

    def __call__(self, input_ids: ttnn.Tensor, token_type_ids: ttnn.Tensor, position_ids: ttnn.Tensor, device):
        p(input_ids, "un sharded input_ids")
        # if input_ids.is_sharded():
        input_ids_interleaved = ttnn.sharded_to_interleaved(input_ids, ttnn.L1_MEMORY_CONFIG)
        print("after conv")
        ttnn.deallocate(input_ids)
        print("after dealloc")
        shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
        shard_spec = ttnn.ShardSpec(
            shard_grid,
            (input_ids_interleaved.shape[-1], ((self.parameters.word_embeddings.weight.shape[-1]) // 8)),
            ttnn.ShardOrientation.COL_MAJOR,
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        print("before word emb")
        p(input_ids, "input tensor")
        # ss
        inputs_embeds = self.word_embeddings(
            input_ids_interleaved,
            weight=self.parameters.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # output_mem_config,
            padding_idx=self.config.pad_token_id,
        )
        # torch.save(
        #     ttnn.to_torch(inputs_embeds),
        #     "/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_sentence_bert/dumps/ttnn_out.pth",
        # )
        # ttnn.deallocate(input_ids)
        p(inputs_embeds, "after word")
        token_type_embeddings = self.token_type_embeddings(
            token_type_ids,
            self.parameters.token_type_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        embeddings = inputs_embeds + token_type_embeddings
        # ttnn.deallocate(inputs_embeds)
        # ttnn.deallocate(token_type_ids)
        p(token_type_embeddings, "after token")
        position_embeddings = self.position_embeddings(
            position_ids,
            self.parameters.position_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # ttnn.deallocate(position_ids)
        p(position_embeddings, "BEFORE position")
        p(embeddings, "BEFORE add1")
        # position_embeddings = ttnn.sharded_to_interleaved(position_embeddings,ttnn.L1_MEMORY_CONFIG)
        # embeddings = ttnn.sharded_to_interleaved(embeddings,ttnn.L1_MEMORY_CONFIG)
        p(position_embeddings, "after position")
        p(embeddings, "after add1")
        # p(position_embeddings,"before repeat")
        # position_embeddings = ttnn.repeat_interleave(position_embeddings,8,dim=0)
        # p(position_embeddings,"after repeat")
        embeddings = embeddings + position_embeddings
        # a = position_embeddings.memory_config()
        # b = embeddings.memory_config()
        # embeddings = ttnn.from_device(embeddings)
        # embeddings = ttnn.to_device(embeddings, device=device)
        # embeddings = ttnn.to_memory_config(embeddings, b)
        # position_embeddings = ttnn.from_device(position_embeddings)
        # position_embeddings = ttnn.to_device(position_embeddings, device=device)
        # position_embeddings = ttnn.to_memory_config(position_embeddings, a)
        # embeddings_d = ttnn.add(position_embeddings, embeddings, memory_config=embeddings.memory_config())
        # torch.save(
        #     ttnn.to_torch(embeddings_d).squeeze(),
        #     "/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_sentence_bert/dumps/ttnn_out.pth",
        # )
        # # p(embeddings, "after add2")
        p(embeddings, "after add")

        # ttnn.deallocate(token_type_embeddings)
        # ttnn.deallocate(position_embeddings)
        embeddings = self.LayerNorm(
            embeddings,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            # program_config=layernorm_program_config,
        )
        p(embeddings, "out is")
        return embeddings


class ttnn_BertOutput:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.dense = ttnn.linear
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor):
        p(hidden_states, "infor-out")
        bert_output_lin = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=f_program_config,
        )
        p(bert_output_lin, "outfor-out")
        p(input_tensor, "in2")
        bert_output_lin = ttnn.reshard(bert_output_lin, input_tensor.memory_config())
        print("after resahrd")
        p(bert_output_lin, "outfor-out")
        p(input_tensor, "in2")
        bert_out = self.LayerNorm(
            bert_output_lin,
            residual_input_tensor=input_tensor,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            program_config=layernorm_program_config,
        )
        # ttnn.deallocate(bert_output_lin)
        # ttnn.deallocate(input_tensor)
        return bert_out


class ttnn_BertIntermediate:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        p(hidden_states, "in-intermediate")
        out_intermediate = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            # activation="gelu",
            program_config=ff1_matmul_program_config,
            # dtype=ttnn.bfloat8_b,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                packer_l1_acc=False,
            ),
        )
        p(out_intermediate, "out-intermediate")
        return out_intermediate


class ttnn_BertSelfOutput:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.dense = ttnn.linear
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor):
        p(hidden_states, "before self-out linear")
        output = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=query_key_value_matmul_program_config,
        )
        p(output, "after self-out linear")
        output_sh = ttnn.reshard(output, input_tensor.memory_config())
        ttnn.deallocate(output)
        p(output, "self_linear_out")
        p(input_tensor, "input2")
        # ttnn.deallocate(hidden_states)
        out_norm = self.LayerNorm(
            output_sh,
            residual_input_tensor=input_tensor,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            # core_grid=ttnn.CoreGrid(y=8,x=8)
            program_config=layernorm_program_config,
        )
        p(out_norm, "after self-out layernorm")
        return out_norm


class ttnn_BertPooler:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.activation = ttnn.tanh
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        p(hidden_states, "in-pool")
        first_token_tensor = hidden_states[:, 0, :]
        print("after slice")
        # ttnn.deallocate(hidden_states)
        pooled_output = self.dense(
            first_token_tensor,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=first_token_tensor.shape[0], x=8),
        )
        print("after linear")
        pooled_output = self.activation(pooled_output)
        print("after act")
        return pooled_output


class ttnn_BertSelfAttention:
    # init - 3 linear's,
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
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
        p(attention_mask, "first mask")
        # p(hidden_states,"before qkv linear")
        query_key_value_output = ttnn.linear(  # 40 cores used
            hidden_states,
            self.parameters.query_key_value.weight,
            bias=self.parameters.query_key_value.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=query_key_value_matmul_program_config
            # dtype=ttnn.bfloat,
        )
        # p(query_key_value_output,"after qkv linear")
        query_key_value_output_d = ttnn.to_memory_config(query_key_value_output, ttnn.DRAM_MEMORY_CONFIG)
        (
            query_layer,
            key_layer,
            value_layer,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            query_key_value_output_d,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.config.num_attention_heads,
        )
        ttnn.deallocate(query_key_value_output)
        key_layer = ttnn.permute(key_layer, (0, 1, 3, 2))
        p(query_layer, "query")
        p(key_layer, "key")
        p(value_layer, "value")
        p(attention_mask, "mask")
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            is_causal=False,
        )
        ttnn.deallocate(query_layer)
        ttnn.deallocate(key_layer)
        ttnn.deallocate(value_layer)
        # ttnn.deallocate(attention_mask)
        p(attn_output, "aftr scaled_dot_product_attention")
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(
            attn_output, (attn_output.shape[0], attn_output.shape[1], attn_output.shape[2] * attn_output.shape[3])
        )
        return attn_output


class ttnn_BertAttention:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.self = ttnn_BertSelfAttention(parameters.self, config)
        self.output = ttnn_BertSelfOutput(parameters.output, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        device=None,
    ):
        self_outputs = self.self(hidden_states, attention_mask, device=device)
        self_outputs_sharded = ttnn.to_memory_config(
            self_outputs,
            memory_config=ttnn.create_sharded_memory_config(
                self_outputs.shape,
                core_grid=device.core_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
        )
        ttnn.deallocate(self_outputs)  # dram
        attention_output = self.output(self_outputs_sharded, hidden_states)
        ttnn.deallocate(self_outputs_sharded)
        return attention_output


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
        ttnn.deallocate(hidden_states)
        self_attention_outputs = ttnn.reallocate(self_attention_outputs)
        intermediate_output = self.intermediate(self_attention_outputs)
        layer_output = self.output(intermediate_output, self_attention_outputs)
        return layer_output


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
        device=None,
    ):
        for i in range(len(self.layers)):  #
            print("iteration is ", i)
            layer_outputs = self.layers[i](hidden_states, attention_mask, device=device)
            print("tttnn layer out", layer_outputs.shape)
            # torch.save(
            #     ttnn.to_torch(layer_outputs),
            #     f"/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_sentence_bert/dumps/ttnn_out_{i}.pth",
            # )
            hidden_states = layer_outputs
        print("tt out of encoder", hidden_states.shape)
        return hidden_states


class ttnn_BertModel:
    def __init__(self, parameters, config):
        self.embeddings = ttnn_BertEmbeddings(parameters.embeddings, config)
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
        # ttnn.deallocate(input_ids)
        # embedding_output  = torch.load("/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_sentence_bert/dumps/torch_emb_out.pth")
        p(embedding_output, "before sharding 1st input")
        # embedding_output = ttnn.from_torch(embedding_output,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device,memory_config=ttnn.L1_MEMORY_CONFIG)
        encoder_input = ttnn.to_memory_config(
            embedding_output,
            memory_config=ttnn.create_sharded_memory_config(
                embedding_output.shape,
                core_grid=device.core_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
        )
        ttnn.deallocate(embedding_output)
        encoder_input = ttnn.reallocate(encoder_input)
        p(embedding_output, "after sharding 1st input")
        sequence_output = self.encoder(encoder_input, attention_mask, device=device)
        ttnn.deallocate(encoder_input)
        sequence_output = ttnn.to_memory_config(sequence_output, ttnn.L1_MEMORY_CONFIG)
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
