import torch
import ttnn
import math

class TtLfm2RMSNorm:
    def __init__(self, device, dim, eps, parameters):
        self.device = device
        self.eps = eps
        self.weight = parameters.weight

    def __call__(self, x):
        return ttnn.rms_norm(x, self.weight, epsilon=self.eps)

class TtSigLip2VisionEncoder:
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.num_layers = config["num_hidden_layers"]
        
    def __call__(self, pixel_values):
        # Tilize input
        pixel_values = ttnn.to_layout(pixel_values, ttnn.TILE_LAYOUT)
        x = ttnn.linear(pixel_values, self.parameters.patch_embed.weight)
        # Vision transformer layers should go here
        for i in range(self.num_layers):
            # Placeholder for transformer block
            # In a real implementation, this would call a TtSigLip2Block
            x = x 
        return x

class TtLfm2VlProjector:
    def __init__(self, device, config, parameters):
        self.device = device
        self.parameters = parameters
        
    def __call__(self, vision_embeddings):
        x = ttnn.linear(vision_embeddings, self.parameters.gate_proj.weight)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.parameters.down_proj.weight)
        return x

class TtLfm2ConvBlock:
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        
    def __call__(self, x):
        projected = ttnn.linear(x, self.parameters.input_projection.weight)
        B, C, x_proj = ttnn.split(projected, self.config["hidden_size"], dim=-1)
        x_gated = ttnn.mul(B, x_proj)
        x_conv_in = ttnn.permute(x_gated, (0, 2, 1))
        x_conv = ttnn.conv1d(x_conv_in, self.parameters.conv.weight, groups=self.config["hidden_size"])
        x_conv = ttnn.permute(x_conv, (0, 2, 1))
        x_gated_2 = ttnn.mul(C, x_conv)
        return ttnn.linear(x_gated_2, self.parameters.output_projection.weight)

class TtLfm2AttentionBlock:
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.input_layernorm = TtLfm2RMSNorm(device, config["hidden_size"], config["norm_eps"], parameters.input_layernorm)
        self.post_attention_layernorm = TtLfm2RMSNorm(device, config["hidden_size"], config["norm_eps"], parameters.post_attention_layernorm)

    def __call__(self, x, layer_past=None, use_cache=False):
        residual = x
        x = self.input_layernorm(x)
        query_states = ttnn.linear(x, self.parameters.self_attn.q_proj.weight)
        key_states = ttnn.linear(x, self.parameters.self_attn.k_proj.weight)
        value_states = ttnn.linear(x, self.parameters.self_attn.v_proj.weight)
        
        if use_cache and layer_past is not None:
            # Infrastructure for KV caching
            # k_cache, v_cache = layer_past
            # ttnn.fill_cache or equivalent would be implemented here
            pass

        attn_output = ttnn.linear(query_states, self.parameters.self_attn.o_proj.weight)
        x = ttnn.add(residual, attn_output)
        residual = x
        x = self.post_attention_layernorm(x)
        gate = ttnn.linear(x, self.parameters.mlp.gate_proj.weight)
        up = ttnn.linear(x, self.parameters.mlp.up_proj.weight)
        x = ttnn.mul(ttnn.silu(gate), up)
        x = ttnn.linear(x, self.parameters.mlp.down_proj.weight)
        return ttnn.add(residual, x)

class TtLfm2VlModel:
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.vision_encoder = TtSigLip2VisionEncoder(device, config["vision_config"], parameters.vision)
        self.projector = TtLfm2VlProjector(device, config, parameters.projector)
        self.layers = []
        for i, layer_type in enumerate(config["layer_types"]):
            if layer_type == "conv":
                self.layers.append(TtLfm2ConvBlock(device, config, parameters.layers[i]))
            else:
                self.layers.append(TtLfm2AttentionBlock(device, config, parameters.layers[i]))

    def __call__(self, pixel_values, input_ids, layer_past=None, use_cache=False):
        img_embs = self.vision_encoder(pixel_values)
        img_tokens = self.projector(img_embs)
        text_tokens = ttnn.embedding(input_ids, self.parameters.embed_tokens.weight)
        
        # Interleaving logic: Replace placeholder tokens with image tokens
        IMAGE_TOKEN_ID = 32000
        input_ids_cpu = ttnn.to_torch(input_ids).to(torch.int32)
        
        mask = (input_ids_cpu == IMAGE_TOKEN_ID)
        if mask.any():
            text_tokens_cpu = ttnn.to_torch(text_tokens)
            img_tokens_cpu = ttnn.to_torch(img_tokens)
            # Replace placeholders in the text sequence with projected vision tokens
            text_tokens_cpu[mask] = img_tokens_cpu.view(-1, img_tokens_cpu.shape[-1])
            x = ttnn.from_torch(text_tokens_cpu, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            # Fallback: simple concatenation if no placeholders found
            x = ttnn.concat([img_tokens, text_tokens], dim=1)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, TtLfm2AttentionBlock):
                x = layer(x, layer_past=layer_past[i] if layer_past else None, use_cache=use_cache)
            else:
                x = layer(x)
        return x
