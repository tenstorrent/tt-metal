import torch
import ttnn
import math

# ----------------------------------------------------------------------------
# DPT Components (Neck & Head)
# ----------------------------------------------------------------------------

class TtDPTReassembleLayer:
    def __init__(self, parameters, read_idx, config):
        self.parameters = parameters
        self.read_idx = read_idx
        self.config = config

    def __call__(self, x):
        # x is (B, Seq, Hidden)
        batch_size, seq_len, hidden_size = x.shape
        grid_h, grid_w = 37, 37 # for 518x518 input with patch 14
        
        # 1. Remove CLS token and reshape
        # x = ttnn.slice(x, (0, 1, 0), (batch_size, seq_len, hidden_size)) # seq_len becomes 1369
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch_size, grid_h, grid_w, hidden_size))
        x = ttnn.permute(x, (0, 3, 1, 2)) # (B, Hidden, H, W)
        
        # 2. Projection (1x1 conv)
        # We assume parameters.projection.weight/bias are available
        # For now, return as is if weights are missing in current mapping
        if hasattr(self.parameters, "projection"):
            # x = ttnn.conv2d(x, self.parameters.projection.weight, bias=self.parameters.projection.bias, ...)
            pass
            
        # 3. Resample (Spatial manipulation - resize)
        # 0: stride=4 (down) -> up 4? No, DPT reassembles to high res
        # Based on ref: 
        # Layer 0: ConvTranspose2d stride 4
        # Layer 1: ConvTranspose2d stride 2
        # Layer 2: Identity
        # Layer 3: Conv2d stride 2
        return x

class TtDPTFusionStage:
    def __init__(self, parameters):
        self.parameters = parameters
    
    def __call__(self, features):
        # Fusion logic: Add + Upsample + Conv
        # features[0] corresponds to layer 5, [1] to 11, [2] to 17, [3] to 23
        return features[-1] # Placeholder for full fusion logic

class TtDPTHead:
    def __init__(self, parameters):
        self.parameters = parameters
        
    def __call__(self, x):
        # Final prediction head
        # x = ttnn.conv2d(x, self.parameters.conv1.weight, ...)
        # x = ttnn.relu(x)
        # x = ttnn.conv2d(x, self.parameters.conv2.weight, ...)
        # x = ttnn.relu(x)
        # x = ttnn.conv2d(x, self.parameters.conv3.weight, ...)
        return x


# ----------------------------------------------------------------------------
# ViT Backbone (ttnn Implementation)
# ----------------------------------------------------------------------------

def vit_patch_embeddings(config, pixel_values, parameters):
    # pixel_values: (B, C, H, W)
    batch_size, img_c, img_h, img_w = pixel_values.shape
    patch_size = 14
    
    # 1. Patchify: (B, C, H, W) -> (B, H/14, W/14, 14*14*C)
    # This is a simplification. In real ttnn, we'd use fold or conv2d.
    # We use ttnn.to_layout(..., ROW_MAJOR) for reshape
    x = ttnn.to_layout(pixel_values, ttnn.ROW_MAJOR_LAYOUT)
    
    # Simple patchification via reshape
    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    
    # (B, C, grid_h, patch_size, grid_w, patch_size)
    x = ttnn.reshape(x, (batch_size, img_c, grid_h, patch_size, grid_w, patch_size))
    # permute to (B, grid_h, grid_w, patch_size, patch_size, C)
    x = ttnn.permute(x, (0, 2, 4, 3, 5, 1))
    # flatten patches: (B, grid_h * grid_w, patch_size * patch_size * C)
    x = ttnn.reshape(x, (batch_size, grid_h * grid_w, patch_size * patch_size * img_c))
    
    # 2. Project
    # parameters.projection.weight is (patch_size*patch_size*C, hidden_size)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = x @ parameters.projection.weight + parameters.projection.bias
    
    return x

def vit_embeddings(config, pixel_values, parameters):
    # 1. Patch Embeddings
    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters.patch_embeddings)
    
    # 2. Concatenate CLS Token
    # cls_token is (1, 1, Hidden)
    # We need to broadcast it to (B, 1, Hidden)
    batch_size = patch_embeddings.shape[0]
    cls_token = ttnn.to_layout(parameters.cls_token, ttnn.TILE_LAYOUT)
    # Concatenate along sequence dimension (dim 1)
    embedding_output = ttnn.concat([cls_token, patch_embeddings], dim=1)
    
    # 3. Add Position Embeddings
    # position_embeddings is (1, Seq+1, Hidden)
    pos_embeds = ttnn.to_layout(parameters.position_embeddings, ttnn.TILE_LAYOUT)
    embedding_output = embedding_output + pos_embeds
    
    return embedding_output

def vit_layer(hidden_states, parameters, config):
    # Layernorm 1
    ln1 = ttnn.layer_norm(hidden_states, weight=parameters.layernorm_before.weight, bias=parameters.layernorm_before.bias)
    
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # Self Attention (Multi-head)
    q = ln1 @ parameters.attention.query.weight + parameters.attention.query.bias
    k = ln1 @ parameters.attention.key.weight + parameters.attention.key.bias
    v = ln1 @ parameters.attention.value.weight + parameters.attention.value.bias
    
    # Reshape and Permute for Multi-head Attention
    q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT)
    q = ttnn.reshape(q, (batch_size, sequence_size, num_heads, head_size))
    q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
    q = ttnn.permute(q, (0, 2, 1, 3))

    k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT)
    k = ttnn.reshape(k, (batch_size, sequence_size, num_heads, head_size))
    k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
    k = ttnn.permute(k, (0, 2, 3, 1))

    v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT)
    v = ttnn.reshape(v, (batch_size, sequence_size, num_heads, head_size))
    v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
    v = ttnn.permute(v, (0, 2, 1, 3))

    # Q * K^T
    attn_scores = q @ k
    attn_scores = attn_scores * (1 / (head_size ** 0.5))
    attn_probs = ttnn.softmax(attn_scores, dim=-1)
    
    # Attn * V
    context_layer = attn_probs @ v
    
    # Merge heads
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer_shape = (batch_size, sequence_size, hidden_size)
    context_layer = ttnn.reshape(context_layer, context_layer_shape)
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)
    
    # Output dense
    attn_out = context_layer @ parameters.attention.output.dense.weight + parameters.attention.output.dense.bias
    
    # Residual 1
    hidden_states = hidden_states + attn_out
    
    # Layernorm 2
    ln2 = ttnn.layer_norm(hidden_states, weight=parameters.layernorm_after.weight, bias=parameters.layernorm_after.bias)
    
    # MLP
    mlp_out = ln2 @ parameters.intermediate.dense.weight + parameters.intermediate.dense.bias
    mlp_out = ttnn.gelu(mlp_out)
    mlp_out = mlp_out @ parameters.output.dense.weight + parameters.output.dense.bias
    
    # Residual 2
    hidden_states = hidden_states + mlp_out
    
    return hidden_states


# ----------------------------------------------------------------------------
# Main Model: Depth Anything V2
# ----------------------------------------------------------------------------

class TtDepthAnythingV2:
    def __init__(self, config, parameters):
        self.config = config
        self.parameters = parameters
        self.reassemble = [TtDPTReassembleLayer(parameters.neck.reassemble[i], i, config) for i in range(4)]
        self.fusion = TtDPTFusionStage(parameters.neck.fusion)
        self.head = TtDPTHead(parameters.head)
        
    def __call__(self, pixel_values):
        # 1. Embeddings
        # pixel_values shape: (B, 3, 518, 518)
        embeddings = vit_embeddings(self.config, pixel_values, self.parameters.backbone.embeddings)
        
        # 2. Encoder (ViT-Large)
        hidden_states = embeddings
        features = []
        out_indices = [5, 11, 17, 23] 
        
        for i in range(24):
            layer_params = self.parameters.backbone.encoder.layer[i]
            hidden_states = vit_layer(hidden_states, layer_params, self.config)
            
            if i in out_indices:
                features.append(hidden_states)
        
        # Final backbone norm
        hidden_states = ttnn.layer_norm(hidden_states, weight=self.parameters.backbone.layernorm.weight, bias=self.parameters.backbone.layernorm.bias)
        
        # 3. Neck (DPT Reassemble & Fusion)
        # Note: Reassemble layers expect (B, Seq, Hidden) and handle token removal
        reassembled_features = [self.reassemble[i](features[i]) for i in range(4)]
        fused_feature = self.fusion(reassembled_features)
        
        # 4. Head
        depth = self.head(fused_feature)
        
        return depth

def custom_preprocessor(torch_model, name):
    parameters = {}
    
    # Helper to convert a single weight
    def convert(tensor):
        return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    
    # helper for ROW_MAJOR tensors (bias, tokens, etc)
    def convert_rm(tensor):
        return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # 1. Backbone
    # Reshape patch projection weight: (1024, 3, 14, 14) -> (3*14*14, 1024) = (588, 1024)
    patch_weight = torch_model.backbone.embeddings.patch_embeddings.projection.weight
    patch_weight = patch_weight.permute(1, 2, 3, 0).reshape(-1, 1024)
    
    parameters["backbone"] = {
        "embeddings": {
            "patch_embeddings": {
                "projection": {"weight": convert(patch_weight), 
                               "bias": convert_rm(torch_model.backbone.embeddings.patch_embeddings.projection.bias)}
            },
            "cls_token": convert_rm(torch_model.backbone.embeddings.cls_token),
            "position_embeddings": convert_rm(torch_model.backbone.embeddings.position_embeddings)
        },
        "encoder": {"layer": []},
        "layernorm": {
            "weight": convert_rm(torch_model.backbone.layernorm.weight),
            "bias": convert_rm(torch_model.backbone.layernorm.bias)
        }
    }
    
    # Encoder Layers
    for i, layer in enumerate(torch_model.backbone.encoder.layer):
        layer_params = {
            "layernorm_before": {
                "weight": convert_rm(layer.norm1.weight),
                "bias": convert_rm(layer.norm1.bias)
            },
            "attention": {
                "query": {"weight": convert(layer.attention.attention.query.weight), "bias": convert_rm(layer.attention.attention.query.bias)},
                "key": {"weight": convert(layer.attention.attention.key.weight), "bias": convert_rm(layer.attention.attention.key.bias)},
                "value": {"weight": convert(layer.attention.attention.value.weight), "bias": convert_rm(layer.attention.attention.value.bias)},
                "output": {"dense": {"weight": convert(layer.attention.output.dense.weight), "bias": convert_rm(layer.attention.output.dense.bias)}}
            },
            "layernorm_after": {
                "weight": convert_rm(layer.norm2.weight),
                "bias": convert_rm(layer.norm2.bias)
            },
            "intermediate": {
                "dense": {"weight": convert(layer.mlp.fc1.weight), "bias": convert_rm(layer.mlp.fc1.bias)}
            },
            "output": {
                "dense": {"weight": convert(layer.mlp.fc2.weight), "bias": convert_rm(layer.mlp.fc2.bias)}
            }
        }
        parameters["backbone"]["encoder"]["layer"].append(layer_params)

    # 2. Neck (DPT Reassemble & Fusion)
    parameters["neck"] = {
        "reassemble": [],
        "fusion": {"layers": []}
    }
    
    for i, layer in enumerate(torch_model.neck.reassemble_stage.layers):
        p = {"projection": {"weight": convert(layer.projection.weight), "bias": convert_rm(layer.projection.bias)}}
        # Optional: handle resize weights if they exist (Conv2d or ConvTranspose2d)
        if hasattr(layer, "resize") and hasattr(layer.resize, "weight"):
             p["resize"] = {"weight": convert(layer.resize.weight), "bias": convert_rm(layer.resize.bias)}
        parameters["neck"]["reassemble"].append(p)
        
    for i, layer in enumerate(torch_model.neck.fusion_stage.layers):
        l = {
            "projection": {"weight": convert(layer.projection.weight), "bias": convert_rm(layer.projection.bias)},
            "residual_layer1": {
                "convolution1": {"weight": convert(layer.residual_layer1.convolution1.weight), "bias": convert_rm(layer.residual_layer1.convolution1.bias)},
                "convolution2": {"weight": convert(layer.residual_layer1.convolution2.weight), "bias": convert_rm(layer.residual_layer1.convolution2.bias)}
            },
            "residual_layer2": {
                "convolution1": {"weight": convert(layer.residual_layer2.convolution1.weight), "bias": convert_rm(layer.residual_layer2.convolution1.bias)},
                "convolution2": {"weight": convert(layer.residual_layer2.convolution2.weight), "bias": convert_rm(layer.residual_layer2.convolution2.bias)}
            }
        }
        parameters["neck"]["fusion"]["layers"].append(l)

    # 3. Head
    parameters["head"] = {
        "conv1": {"weight": convert(torch_model.head.conv1.weight), "bias": convert_rm(torch_model.head.conv1.bias)},
        "conv2": {"weight": convert(torch_model.head.conv2.weight), "bias": convert_rm(torch_model.head.conv2.bias)},
        "conv3": {"weight": convert(torch_model.head.conv3.weight), "bias": convert_rm(torch_model.head.conv3.bias)}
    }
        
    return parameters
