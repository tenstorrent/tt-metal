# models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""
import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_attention import Qwen35GatedAttention
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet import Qwen35GatedDeltaNet
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_mlp import Qwen35MLP


def rms_norm_ttnn(x, weight, eps=1e-6, memory_config=None):
    """Zero-centered RMSNorm using fused ttnn.rms_norm.

    Qwen3.5 uses zero-centered RMSNorm for ALL layer norms:
      output = x * rsqrt(mean(x^2) + eps) * (1 + weight)

    The weight should be pre-offset by +1 so we can use the standard fused op.
    Verified: fused ttnn.rms_norm works for all decode shapes on Blackhole P150:
      [1, 1, 4096], [1, 1, 16, 256], [1, 1, 32, 128]
    """
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)


class Qwen35TransformerBlock:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
        self.layer_num = layer_num
        self.device = device
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        def load_norm(name):
            """Load norm weight with +1 offset for zero-centered RMSNorm.

            Qwen3.5 uses zero-centered RMSNorm: output = x_normed * (1 + weight).
            We pre-add 1 to the weight so the fused ttnn.rms_norm can be used directly.
            The +1 offset is baked into the cached tensor — safe on reload.
            """
            t = state_dict[f"{prefix}.{name}"]
            t_offset = t + 1.0  # Pre-offset for zero-centered norm
            return ttnn.as_tensor(
                t_offset,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
            )

        self.attention_norm_weight = load_norm("input_layernorm.weight")
        self.ff_norm_weight = load_norm("post_attention_layernorm.weight")
        self.norm_eps = args.norm_eps

        if self.is_full_attention:
            self.attention = Qwen35GatedAttention(args, state_dict, layer_num, device, weight_cache_path)
        else:
            self.attention = Qwen35GatedDeltaNet(args, state_dict, layer_num, device, weight_cache_path)

        self.feed_forward = Qwen35MLP(args, state_dict, layer_num, device, weight_cache_path)

    def forward(self, x, cos=None, sin=None, mode="decode", chunk_size=64):
        mc = ttnn.L1_MEMORY_CONFIG if mode == "decode" else None
        # "prefill_segmented" acts like prefill but forces recurrent for DeltaNet
        attn_input = rms_norm_ttnn(x, self.attention_norm_weight, eps=self.norm_eps, memory_config=mc)

        if self.is_full_attention:
            attn_output = self.attention.forward(attn_input, cos, sin)
        else:
            # Use chunked mode for direct prefill (parallel within chunks, faster).
            # Segmented prefill uses recurrent — chunked error compounds through 24 layers
            # × many segments, producing garbage output at 4K+ tokens.
            # Decode always uses recurrent mode (T=1).
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(attn_input, mode=deltanet_mode, chunk_size=chunk_size)
        ttnn.deallocate(attn_input)

        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        ff_input = rms_norm_ttnn(h, self.ff_norm_weight, eps=self.norm_eps, memory_config=mc)
        ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)
        return output
