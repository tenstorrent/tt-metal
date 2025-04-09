import ttnn 
import torch

def rotate_half(x):
    x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], x.shape[-1] // 2))
    x2 = ttnn.slice(x, (0, 0, 0, x.shape[-1] // 2), (x.shape[0], x.shape[1], x.shape[2], x.shape[-1]))
    return ttnn.concat([ttnn.multiply(x2,-1), x1], dim=-1)

class TT_LlamaRMSNorm():
    def __init__(self, parameters, path = None, layer_name = None, eps=1e-6):
        super().__init__()
        self.variance_epsilon = eps
        
        if layer_name is not None:
            self.weight = parameters[f"{path}.{layer_name}.weight"]
        else:
            self.weight = parameters["model.norm.weight"]
        
    def __call__(self, hidden_states): 
        variance = ttnn.pow(hidden_states, 2)
        variance = ttnn.mean(variance, dim = -1)
        hidden_states = hidden_states * ttnn.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
    
class TT_LlamaMLP():
    def __init__(self, parameters, path):
        super().__init__()
        
        self.gate_proj = parameters[f"{path}.mlp.gate_proj.weight"]
        self.up_proj = parameters[f"{path}.mlp.up_proj.weight"]
        self.down_proj = parameters[f"{path}.mlp.down_proj.weight"]
        
        self.bias = None
        self.act_fn = ttnn.silu

    def __call__(self, x, device):
        
        # ttnn.linear produces low pc in GS

        x = ttnn.to_torch(x)
        self.gate_proj = ttnn.to_torch(self.gate_proj)
        gate = torch.nn.functional.linear(x, self.gate_proj.T.contiguous(), bias = self.bias)
        gate = ttnn.from_torch(gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        gate = self.act_fn(gate)
        gate = ttnn.to_torch(gate)
        self.up_proj = ttnn.to_torch(self.up_proj)
        up = torch.nn.functional.linear(x, self.up_proj.T.contiguous(), bias = self.bias)
        product = gate * up
        
        self.down_proj = ttnn.to_torch(self.down_proj)
        down_proj = torch.nn.functional.linear(product, self.down_proj.T.contiguous(), bias = self.bias)
        
        down_proj = ttnn.from_torch(down_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        
        self.gate_proj = ttnn.from_torch(self.gate_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.up_proj = ttnn.from_torch(self.up_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.down_proj = ttnn.from_torch(self.down_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        
        return down_proj
    
class TT_LlamaRotaryEmbedding():
    def __init__(self, hidden_size, num_heads, device, base = 10000):
        super().__init__()
        dim = hidden_size // num_heads
        self.device = device

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(None) / dim)) # pow op not supported for scalar^tensor in GS  
        inv_freq = ttnn.from_torch(inv_freq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        self.inv_freq = ttnn.reshape(inv_freq, (1,inv_freq.shape[-1]))

    def __call__(self, position_ids):
        
        inv_freq_expanded = ttnn.expand(self.inv_freq,[position_ids.shape[0], self.inv_freq.shape[1]])
        inv_freq_expanded = ttnn.reshape(inv_freq_expanded, (1,  self.inv_freq.shape[1], 1))
        
        position_ids_expanded = ttnn.reshape(position_ids, (position_ids.shape[0], 1, position_ids.shape[1]))
    
        freqs = ttnn.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = ttnn.transpose(freqs, 1, 2)
        
        emb = ttnn.concat([freqs, freqs], dim=-1)
        
        emb = ttnn.to_torch(emb)
        cos = emb.cos()    # cos and sin op produce low pcc in GS
        sin = emb.sin()
        
        cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = self.device)
        sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = self.device)
        
        return cos, sin

class TT_LlamaAttention():
    def __init__(self, parameters, path, hidden_size, num_attention_heads):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.bias = None
        
        self.q_proj = parameters[f"{path}.self_attn.q_proj.weight"]
        self.k_proj = parameters[f"{path}.self_attn.k_proj.weight"]
        self.v_proj = parameters[f"{path}.self_attn.v_proj.weight"]
        self.o_proj = parameters[f"{path}.self_attn.o_proj.weight"]

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value = None,
        device = None
    ):
        
        input_shape = (hidden_states.shape[0],hidden_states.shape[1])
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        hidden_states = ttnn.to_torch(hidden_states)
        self.q_proj = ttnn.to_torch(self.q_proj)
        self.k_proj = ttnn.to_torch(self.k_proj)
        self.v_proj = ttnn.to_torch(self.v_proj)
        self.o_proj = ttnn.to_torch(self.o_proj)
        
        # ttnn.linear produces low pc in GS
        
        query_states = torch.nn.functional.linear(hidden_states, self.q_proj.T.contiguous(), bias = self.bias)
        key_states = torch.nn.functional.linear(hidden_states, self.k_proj.T.contiguous(), bias = self.bias)
        value_states = torch.nn.functional.linear(hidden_states, self.v_proj.T.contiguous(), bias = self.bias)
        
        query_states = ttnn.from_torch(query_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        key_states = ttnn.from_torch(key_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        value_states = ttnn.from_torch(value_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)

        query_states = ttnn.transpose(ttnn.reshape(query_states, hidden_shape), 1, 2)
        key_states = ttnn.transpose(ttnn.reshape(key_states, hidden_shape), 1, 2)
        value_states = ttnn.transpose(ttnn.reshape(value_states, hidden_shape), 1, 2)
        
        cos, sin = position_embeddings
        
        cos = ttnn.unsqueeze(cos, 1)
        sin = ttnn.unsqueeze(sin, 1)
        
        query_states = ttnn.add(ttnn.mul(query_states, cos), ttnn.mul(rotate_half(query_states), sin))
        key_states = ttnn.add(ttnn.mul(key_states, cos), ttnn.mul(rotate_half(key_states), sin))
        
        attn_output = ttnn.transformer.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=None,
        is_causal=True,
        scale=self.scaling,
        )
    
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output,(*input_shape, -1))
        attn_output = ttnn.to_torch(attn_output)
        
        attn_output = torch.nn.functional.linear(attn_output, self.o_proj.T.contiguous(), bias = self.bias)
        attn_output = ttnn.from_torch(attn_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        attn_weights = None
        
        self.q_proj = ttnn.from_torch(self.q_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.k_proj = ttnn.from_torch(self.k_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.v_proj = ttnn.from_torch(self.v_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.o_proj = ttnn.from_torch(self.o_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        return attn_output, attn_weights
    
class TT_LlamaDecoderLayer():
    def __init__(self, parameters, path, hidden_size, num_attention_heads):
        super().__init__()

        self.self_attn = TT_LlamaAttention(parameters, path, hidden_size, num_attention_heads)

        self.mlp = TT_LlamaMLP(parameters, path)
        self.input_layernorm = TT_LlamaRMSNorm(parameters, path, layer_name = "input_layernorm")
        self.post_attention_layernorm = TT_LlamaRMSNorm(parameters, path, layer_name = "post_attention_layernorm")

    def __call__(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,
        device = None
    ):
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            device=device
        )
        
        hidden_states = ttnn.add(residual, hidden_states)

        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states, device)
        
        hidden_states = ttnn.add(residual, hidden_states)

        outputs = (hidden_states,)
        
        return outputs
    
class TT_LlamaModel():
    def __init__(self, config, parameters, hidden_size, num_attention_heads, num_hidden_layers, device):
        super().__init__()
        self.config = config
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.padding_idx = config.pad_token_id
        self.embed_tokens_weight = ttnn.to_layout(parameters["model.embed_tokens.weight"], layout=ttnn.ROW_MAJOR_LAYOUT)
        self.layers = list(
            [TT_LlamaDecoderLayer(parameters, f"model.layers.{layer_idx}", hidden_size, num_attention_heads) for layer_idx in range(num_hidden_layers)]
        )
        
        self.norm = TT_LlamaRMSNorm(parameters)
        self.rotary_emb = TT_LlamaRotaryEmbedding(hidden_size, num_attention_heads, device)
        self.gradient_checkpointing = False

    def __call__(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
        device = None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = ttnn.embedding(input_ids, self.embed_tokens_weight, layout=ttnn.ROW_MAJOR_LAYOUT)
        
        inputs_embeds = ttnn.to_layout(inputs_embeds, layout=ttnn.TILE_LAYOUT)
        
        causal_mask = None

        hidden_states = inputs_embeds
        
        position_embeddings = self.rotary_emb(position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[ : self.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                device = device
            )

            hidden_states = layer_outputs[0]
        
        hidden_states = self.norm(hidden_states)

        output = (
            hidden_states,
            past_key_values if use_cache else None,
            all_hidden_states,
            all_self_attns,
            )
        return output if return_dict else output.to_tuple()
    
class TT_LlamaForCausalLM():
    
    def __init__(self, config, parameters, hidden_size, num_attention_heads, num_hidden_layers, device):
        super().__init__()
        self.config = config
        self.model = TT_LlamaModel(config, parameters, hidden_size, num_attention_heads, num_hidden_layers, device)
        self.lm_head = parameters["lm_head.weight"]

    def __call__(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
        device = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            device=device
        )

        hidden_states = outputs[0]

        hidden_states = ttnn.to_torch(hidden_states)

        self.lm_head = ttnn.to_torch(self.lm_head)
        
        logits = torch.nn.functional.linear(hidden_states, self.lm_head.T.contiguous(), bias=None)
        logits = ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        
        self.lm_head = ttnn.from_torch(self.lm_head, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        
        loss = None
        
        return(
            loss,
            logits,
        )