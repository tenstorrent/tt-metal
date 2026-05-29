import torch
import ttnn
import math
from typing import Optional, Tuple, List

class TtLfm2RMSNorm:
    """RMS Normalization layer used in LFM2.5-VL."""
    def __init__(self, device, dim, eps, parameters):
        self.device = device
        self.eps = eps
        self.weight = parameters.weight

    def __call__(self, x):
        return ttnn.rms_norm(x, self.weight, epsilon=self.eps)


class TtSigLip2VisionAttention:
    """Multi-head attention for SigLIP2 vision encoder."""
    def __init__(self, device, config, parameters):
        self.device = device
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.parameters = parameters
        self.has_fused_qkv = hasattr(parameters.attn, "qkv")
        
    def __call__(self, x):
        # QKV projection
        if self.has_fused_qkv:
            # Fused QKV: single projection then split
            qkv = ttnn.linear(x, self.parameters.attn.qkv.weight)
            # Split into Q, K, V (each [B, S, D])
            q, k, v = ttnn.split(qkv, 3, dim=-1)
        else:
            q = ttnn.linear(x, self.parameters.attn.q.weight)
            k = ttnn.linear(x, self.parameters.attn.k.weight)
            v = ttnn.linear(x, self.parameters.attn.v.weight)
        
        # Reshape to multi-head format: [B, num_heads, S, head_dim]
        batch, seq_len, _ = x.shape
        q = ttnn.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch, seq_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (batch, seq_len, self.num_heads, self.head_dim))
        
        # Permute to [B, num_heads, S, head_dim]
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        # scores = Q * K^T / sqrt(d)
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        attn_weights = ttnn.matmul(q, k_t)
        attn_weights = ttnn.mul(attn_weights, self.scale)
        attn_weights = ttnn.softmax(attn_weights, dim=-1)
        
        # attention_output = scores * V
        attn_output = ttnn.matmul(attn_weights, v)
        
        # Permute back: [B, S, num_heads, head_dim] -> [B, S, D]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch, seq_len, self.hidden_size))
        
        # Output projection
        output = ttnn.linear(attn_output, self.parameters.attn.proj.weight)
        
        return output


class TtSigLip2VisionMLP:
    """MLP for SigLIP2 vision encoder."""
    def __init__(self, device, config, parameters):
        self.device = device
        self.parameters = parameters
        
    def __call__(self, x):
        x = ttnn.linear(x, self.parameters.mlp.fc1.weight)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.parameters.mlp.fc2.weight)
        return x


class TtSigLip2VisionBlock:
    """A single SigLIP2 vision transformer block with pre-norm architecture."""
    def __init__(self, device, config, parameters):
        self.device = device
        self.hidden_size = config["hidden_size"]
        self.eps = 1e-6  # SigLIP2 uses 1e-6 for layer norms
        
        self.norm1 = TtLfm2RMSNorm(device, config["hidden_size"], self.eps, parameters.norm1)
        self.norm2 = TtLfm2RMSNorm(device, config["hidden_size"], self.eps, parameters.norm2)
        self.attention = TtSigLip2VisionAttention(device, config, parameters)
        self.mlp = TtSigLip2VisionMLP(device, config, parameters)
        
    def __call__(self, x):
        # Pre-norm attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = ttnn.add(residual, x)
        
        # Pre-norm MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = ttnn.add(residual, x)
        
        return x


class TtSigLip2VisionEncoder:
    """SigLIP2 Vision Encoder with full transformer blocks.
    
    Processes pixel_values through:
    1. Patch embedding (linear projection of patches)
    2. Optional position embedding
    3. N transformer blocks (pre-norm attention + MLP)
    """
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.num_layers = config["num_hidden_layers"]
        
        # Build vision blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = TtSigLip2VisionBlock(device, config, parameters.layers[i])
            self.blocks.append(block)
        
    def __call__(self, pixel_values):
        # Tilize input (required by ttnn linear for TILE_LAYOUT)
        pixel_values = ttnn.to_layout(pixel_values, ttnn.TILE_LAYOUT)
        
        # Patch embedding
        x = ttnn.linear(pixel_values, self.parameters.patch_embed.weight)
        
        # Add position embedding if present
        if hasattr(self.parameters, "pos_embed"):
            x = ttnn.add(x, self.parameters.pos_embed.weight)
        
        # Vision transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return x


class TtLfm2VlProjector:
    """MLP projector bridging vision features to text embedding space."""
    def __init__(self, device, config, parameters):
        self.device = device
        self.parameters = parameters
        
    def __call__(self, vision_embeddings):
        x = ttnn.linear(vision_embeddings, self.parameters.gate_proj.weight)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.parameters.down_proj.weight)
        return x


class TtLfm2ConvBlock:
    """LIV (Linear Input-Varying) Convolution Block.
    
    Implements the double-gating convolution pattern:
    1. Input projection with split into B, C, x_proj
    2. Gate: B * x_proj
    3. Depthwise conv1d
    4. Gate: C * conv_output
    5. Output projection
    """
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


class TtLfm2Attention:
    """Grouped-Query Attention with RoPE.
    
    Implements the full attention mechanism:
    1. QKV projections
    2. Head creation with GQA (num_heads=32, num_kv_heads=8)
    3. RoPE on Q and K
    4. Scaled dot-product attention (via SDPA or manual)
    5. Head concat
    6. Output projection
    """
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config.get("num_key_value_heads", 8)
        self.hidden_size = config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
    def __call__(self, x, cos_sin_cache=None, layer_past=None, use_cache=False):
        batch, seq_len, _ = x.shape
        
        # QKV projections
        q = ttnn.linear(x, self.parameters.self_attn.q_proj.weight)
        k = ttnn.linear(x, self.parameters.self_attn.k_proj.weight)
        v = ttnn.linear(x, self.parameters.self_attn.v_proj.weight)
        
        # Reshape to multi-head format
        q = ttnn.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch, seq_len, self.num_kv_heads, self.head_dim))
        v = ttnn.reshape(v, (batch, seq_len, self.num_kv_heads, self.head_dim))
        
        # Permute to [B, heads, S, head_dim]
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        
        # Apply RoPE to Q and K if cos_sin_cache provided
        if cos_sin_cache is not None:
            cos, sin = cos_sin_cache
            # Apply rotary embeddings
            # Using ttnn.experimental.rotary_embedding_llama or manual
            try:
                q = ttnn.experimental.rotary_embedding_llama(
                    q, cos, sin, None, is_decode_mode=False
                )
                k = ttnn.experimental.rotary_embedding_llama(
                    k, cos, sin, None, is_decode_mode=False
                )
            except Exception:
                # Fallback: manual rotary embedding
                q = self._apply_rotary(q, cos, sin)
                k = self._apply_rotary(k, cos, sin)
        
        # KV cache update for decode mode
        if use_cache and layer_past is not None:
            k_cache, v_cache = layer_past
            # Concatenate past and current K/V
            k = ttnn.concat([k_cache, k], dim=2)
            v = ttnn.concat([v_cache, v], dim=2)
        
        # GQA: repeat KV heads to match Q heads
        n_repeat = self.num_heads // self.num_kv_heads
        if n_repeat > 1:
            k = ttnn.repeat(k, (1, n_repeat, 1, 1))
            v = ttnn.repeat(v, (1, n_repeat, 1, 1))
        
        # Scaled dot-product attention
        # scores = Q * K^T / sqrt(d)
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        attn_weights = ttnn.matmul(q, k_t)
        attn_weights = ttnn.mul(attn_weights, self.scale)
        
        # Causal mask (optional - applied for decode/autoregressive)
        # For prefill with prompt, this may not be needed if SDPA handles it
        attn_weights = ttnn.softmax(attn_weights, dim=-1)
        
        # attention_output = scores * V
        attn_output = ttnn.matmul(attn_weights, v)
        
        # Prepare KV cache output
        present_kv = None
        if use_cache:
            # K/V are now [B, heads, S_total, head_dim] after concat
            present_kv = (k, v)
        
        # Permute back: [B, S, D]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch, -1, self.hidden_size))
        
        # Output projection
        output = ttnn.linear(attn_output, self.parameters.self_attn.o_proj.weight)
        
        return output, present_kv
    
    def _apply_rotary(self, x, cos, sin):
        """Apply rotary position embeddings manually.
        
        x: [B, heads, S, head_dim]
        cos, sin: [1, 1, S, head_dim]
        """
        # Split into pairs
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        
        # RoPE: rotate pairs
        cos_even = cos[:, :, :, 0::2]
        cos_odd = cos[:, :, :, 1::2]
        sin_even = sin[:, :, :, 0::2]
        sin_odd = sin[:, :, :, 1::2]
        
        rotated_even = ttnn.sub(ttnn.mul(x_even, cos_even), ttnn.mul(x_odd, sin_even))
        rotated_odd = ttnn.add(ttnn.mul(x_even, sin_odd), ttnn.mul(x_odd, cos_odd))
        
        # Interleave back
        result = ttnn.concat([rotated_even, rotated_odd], dim=-1)
        return result


class TtLfm2AttentionBlock:
    """Transformer block with Grouped-Query Attention and MLP.
    
    Pre-norm architecture:
    1. Input norm
    2. Attention
    3. Residual add
    4. Post-attention norm
    5. MLP (SwiGLU: silu(gate) * up)
    6. Residual add
    """
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.input_layernorm = TtLfm2RMSNorm(device, config["hidden_size"], config["norm_eps"], parameters.input_layernorm)
        self.post_attention_layernorm = TtLfm2RMSNorm(device, config["hidden_size"], config["norm_eps"], parameters.post_attention_layernorm)
        self.attention = TtLfm2Attention(device, config, parameters)

    def __call__(self, x, cos_sin_cache=None, layer_past=None, use_cache=False):
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        attn_out, present_kv = self.attention(x, cos_sin_cache, layer_past, use_cache)
        x = ttnn.add(residual, attn_out)
        
        # Pre-norm MLP (SwiGLU)
        residual = x
        x = self.post_attention_layernorm(x)
        gate = ttnn.linear(x, self.parameters.mlp.gate_proj.weight)
        up = ttnn.linear(x, self.parameters.mlp.up_proj.weight)
        x = ttnn.mul(ttnn.silu(gate), up)
        x = ttnn.linear(x, self.parameters.mlp.down_proj.weight)
        x = ttnn.add(residual, x)
        
        return x, present_kv


class TtLfm2VlModel:
    """LFM2.5-VL Multimodal Model.
    
    Architecture:
    1. SigLIP2 Vision Encoder -> Projector for image input
    2. Text embedding for token input
    3. Token interleaving (placeholder-based image token insertion)
    4. Hybrid backbone (interleaved LIV Conv blocks + GQA Attention blocks)
    """
    def __init__(self, device, config, parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        
        # Vision encoder and projector
        self.vision_encoder = TtSigLip2VisionEncoder(
            device, config["vision_config"], parameters.vision
        )
        self.projector = TtLfm2VlProjector(
            device, config, parameters.projector
        )
        
        # Hybrid backbone layers
        self.layers = []
        for i, layer_type in enumerate(config["layer_types"]):
            if layer_type == "conv":
                self.layers.append(TtLfm2ConvBlock(
                    device, config, parameters.layers[i]
                ))
            else:
                self.layers.append(TtLfm2AttentionBlock(
                    device, config, parameters.layers[i]
                ))
        
        # RoPE cache (optional - can also be passed in)
        self.cos_sin_cache = None

    def set_rope_cache(self, cos_sin_cache):
        """Set the RoPE cos/sin cache for attention layers."""
        self.cos_sin_cache = cos_sin_cache

    def __call__(
        self,
        pixel_values,
        input_ids,
        layer_past=None,
        use_cache=False,
    ):
        batch_size = input_ids.shape[0]
        
        # === Vision path ===
        img_embs = self.vision_encoder(pixel_values)
        img_tokens = self.projector(img_embs)
        
        # === Text embedding ===
        text_tokens = ttnn.embedding(input_ids, self.parameters.embed_tokens.weight)
        
        # === Token interleaving ===
        # Replace placeholder token IDs with projected vision features
        IMAGE_TOKEN_ID = 32000
        input_ids_cpu = ttnn.to_torch(input_ids).to(torch.int32)
        
        mask = (input_ids_cpu == IMAGE_TOKEN_ID)
        if mask.any():
            text_tokens_cpu = ttnn.to_torch(text_tokens)
            img_tokens_cpu = ttnn.to_torch(img_tokens)
            # Insert image tokens at placeholder positions
            text_tokens_cpu[mask] = img_tokens_cpu.view(-1, img_tokens_cpu.shape[-1])
            x = ttnn.from_torch(
                text_tokens_cpu,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            # Fallback: prefix concatenation
            x = ttnn.concat([img_tokens, text_tokens], dim=1)
        
        # === Hybrid backbone ===
        present_kv = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TtLfm2AttentionBlock):
                x, kv = layer(
                    x,
                    cos_sin_cache=self.cos_sin_cache,
                    layer_past=layer_past[i] if layer_past else None,
                    use_cache=use_cache,
                )
                if use_cache:
                    present_kv.append(kv)
            else:
                x = layer(x)
        
        # Apply final norm if present
        if hasattr(self.parameters, "norm") and hasattr(self.parameters.norm, "weight"):
            x = ttnn.rms_norm(x, self.parameters.norm.weight, epsilon=self.config["norm_eps"])
        
        if use_cache:
            return x, present_kv
        return x