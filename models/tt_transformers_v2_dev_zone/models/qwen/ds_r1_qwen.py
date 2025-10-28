#!/usr/bin/env python3
"""Pure TTNN implementation of Qwen model (DeepSeek-R1-Distill-Qwen-1.5B)"""

import math
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn

# todo)) work on ds_r1_qwen model validation using new decorator

# TODO)) accumulate prompts that instructs AI to use testing tools to validate the model
# -- where, how, what to add to validate_against decorator
# -- get us to the debug iteration starting point!
# -- almost automatically debugging a model by probing tensors and checking their pcc against reference tensors, through one prompt?!
# -- TTTv2 could be agent-first toolkit! -- a set of prompts and libraries that users who comes to us with TTNN model can use to debug their model, add demo, add vllm, add perf checks, etc.


# ============================================================================
# Model Implementation by claude-4.5-sonnet thinking
# ============================================================================


class RMSNorm:
    """RMS Normalization in TTNN"""

    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def __call__(self, x):
        # x shape: [1, seq_len, hidden_size]
        # Compute RMS: sqrt(mean(x^2) + eps)
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        # Normalize and scale
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)


class RotaryEmbedding:
    """Rotary Position Embedding"""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0, device=None):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        # Cache cos and sin
        self.cos_cached = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        self.sin_cached = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

    def apply_rotary_emb(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Apply rotary embeddings to input tensor (on CPU/torch)"""
        # x: [batch, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = x.shape

        # Get cos/sin for the current position range
        cos = self.cos_cached[:, :, position : position + seq_len, :].to(x.device)
        sin = self.sin_cached[:, :, position : position + seq_len, :].to(x.device)

        # Split into even and odd dimensions
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rotated.flatten(-2)

        return rotated


class Attention:
    """Multi-head attention with GQA support"""

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        device,
        max_seq_len: int = 2048,
    ):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.hidden_size = hidden_size
        self.device = device

        # Convert weights to TTNN
        self.wq = ttnn.from_torch(
            wq.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wk = ttnn.from_torch(
            wk.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wv = ttnn.from_torch(
            wv.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wo = ttnn.from_torch(
            wo.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len)

        # KV cache storage (on CPU as torch tensors for now)
        self.cache_k = torch.zeros((1, num_kv_heads, max_seq_len, head_dim))
        self.cache_v = torch.zeros((1, num_kv_heads, max_seq_len, head_dim))

    def __call__(self, x, start_pos: int, mask: Optional[torch.Tensor] = None):  # TTNN tensor [1, seq_len, hidden_size]
        # Project to Q, K, V
        xq = ttnn.matmul(x, self.wq)
        xk = ttnn.matmul(x, self.wk)
        xv = ttnn.matmul(x, self.wv)

        # Convert to torch for reshaping and RoPE
        xq_torch = ttnn.to_torch(xq).squeeze(0)  # [seq_len, hidden_size]
        xk_torch = ttnn.to_torch(xk).squeeze(0)
        xv_torch = ttnn.to_torch(xv).squeeze(0)

        batch_size, seq_len, _ = xq_torch.shape

        # Reshape to [batch, num_heads, seq_len, head_dim]
        xq_torch = xq_torch.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk_torch = xk_torch.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv_torch = xv_torch.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        xq_torch = self.rotary_emb.apply_rotary_emb(xq_torch, start_pos)
        xk_torch = self.rotary_emb.apply_rotary_emb(xk_torch, start_pos)

        # Update KV cache
        self.cache_k[:, :, start_pos : start_pos + seq_len] = xk_torch
        self.cache_v[:, :, start_pos : start_pos + seq_len] = xv_torch

        # Get keys and values up to current position
        keys = self.cache_k[:, :, : start_pos + seq_len]
        values = self.cache_v[:, :, : start_pos + seq_len]

        # Repeat KV heads for GQA
        if self.num_queries_per_kv > 1:
            keys = keys.repeat_interleave(self.num_queries_per_kv, dim=1)
            values = values.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute attention scores
        scores = torch.matmul(xq_torch, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        scores = torch.nn.functional.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(scores, values)

        # Reshape back to [batch, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Convert back to TTNN and apply output projection
        output_tt = ttnn.from_torch(
            output.unsqueeze(0), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        output_tt = ttnn.matmul(output_tt, self.wo)

        return output_tt


class MLP:
    """Feed-forward network"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        device,
    ):
        self.gate_proj = ttnn.from_torch(
            gate_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.up_proj = ttnn.from_torch(
            up_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.down_proj = ttnn.from_torch(
            down_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def __call__(self, x):
        # SwiGLU activation: gate(x) * up(x) then down projection
        gate = ttnn.matmul(x, self.gate_proj)
        gate = ttnn.silu(gate)

        up = ttnn.matmul(x, self.up_proj)

        hidden = ttnn.mul(gate, up)
        output = ttnn.matmul(hidden, self.down_proj)

        return output


class TransformerBlock:
    """Single transformer layer"""

    def __init__(self, layer_id: int, config, layer_weights, device, max_seq_len: int = 2048):
        self.layer_id = layer_id

        # Attention
        self.attention = Attention(
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            wq=layer_weights["self_attn.q_proj.weight"],
            wk=layer_weights["self_attn.k_proj.weight"],
            wv=layer_weights["self_attn.v_proj.weight"],
            wo=layer_weights["self_attn.o_proj.weight"],
            device=device,
            max_seq_len=max_seq_len,
        )

        # MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_proj=layer_weights["mlp.gate_proj.weight"],
            up_proj=layer_weights["mlp.up_proj.weight"],
            down_proj=layer_weights["mlp.down_proj.weight"],
            device=device,
        )

        # Norms
        self.input_layernorm = RMSNorm(layer_weights["input_layernorm.weight"], config.rms_norm_eps, device)
        self.post_attention_layernorm = RMSNorm(
            layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps, device
        )

    def __call__(self, x, start_pos: int, mask: Optional[torch.Tensor] = None):
        # Attention with residual
        h = self.input_layernorm(x)
        h = self.attention(h, start_pos, mask)
        x = ttnn.add(x, h)

        # MLP with residual
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = ttnn.add(x, h)

        return x


class QwenModel:
    """Complete Qwen transformer model"""

    def __init__(self, model_name: str, device, max_seq_len: int = 2048):
        print(f"Loading {model_name}...")

        # Load HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.config = hf_model.config
        self.device = device
        self.max_seq_len = max_seq_len

        print(f"Model config:")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Num layers: {self.config.num_hidden_layers}")
        print(f"  Num attention heads: {self.config.num_attention_heads}")
        print(f"  Num KV heads: {self.config.num_key_value_heads}")
        print(f"  Intermediate size: {self.config.intermediate_size}")
        print(f"  Vocab size: {self.config.vocab_size}")

        # Extract weights
        state_dict = hf_model.state_dict()

        # Embedding (keep on CPU for now, will convert per-batch)
        self.embed_tokens = state_dict["model.embed_tokens.weight"]

        # Build transformer layers
        print("Building transformer layers...")
        self.layers = []
        for layer_id in range(self.config.num_hidden_layers):
            layer_weights = {
                key.replace(f"model.layers.{layer_id}.", ""): value
                for key, value in state_dict.items()
                if f"model.layers.{layer_id}." in key
            }

            layer = TransformerBlock(
                layer_id=layer_id,
                config=self.config,
                layer_weights=layer_weights,
                device=device,
                max_seq_len=max_seq_len,
            )
            self.layers.append(layer)

            if (layer_id + 1) % 4 == 0:
                print(f"  Loaded {layer_id + 1}/{self.config.num_hidden_layers} layers")

        # Final norm
        self.norm = RMSNorm(state_dict["model.norm.weight"], self.config.rms_norm_eps, device)

        # LM head (output projection)
        self.lm_head = ttnn.from_torch(
            state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        print("Model loaded successfully!")

        # Free HF model
        del hf_model
        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):  # [batch, seq_len]
        batch_size, seq_len = tokens.shape

        # Embed tokens
        h = self.embed_tokens[tokens]  # [batch, seq_len, hidden_size]

        # Create causal mask
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        else:
            mask = None

        # Convert to TTNN
        h_tt = ttnn.from_torch(h.unsqueeze(0), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Apply transformer layers
        for layer in self.layers:
            h_tt = layer(h_tt, start_pos, mask)

        # Final norm
        h_tt = self.norm(h_tt)

        # LM head
        logits = ttnn.matmul(h_tt, self.lm_head)

        return logits

    def reset_kv_cache(self):
        """Clear KV cache for all layers"""
        for layer in self.layers:
            layer.attention.cache_k.zero_()
            layer.attention.cache_v.zero_()


def generate(model: QwenModel, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0):
    """Generate text from prompt"""

    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="pt")

    # Reset cache
    model.reset_kv_cache()

    # Prefill phase
    print(f"Prefill: {tokens.shape[1]} tokens")
    logits = model.forward(tokens, start_pos=0)

    # Get last token logits
    logits_torch = ttnn.to_torch(logits).squeeze(0)  # [batch, seq_len, vocab_size]
    next_token_id = torch.argmax(logits_torch[:, -1, :], dim=-1).item()  # Take last position

    generated = [next_token_id]
    current_pos = tokens.shape[1]

    # Decode phase
    print("Generating...")
    for i in range(max_new_tokens - 1):
        next_token = torch.tensor([[next_token_id]], dtype=torch.long)  # [1, 1]
        logits = model.forward(next_token, start_pos=current_pos)
        logits_torch = ttnn.to_torch(logits).squeeze(0)  # [batch, 1, vocab_size]

        # Sample next token
        if temperature > 0:
            probs = torch.nn.functional.softmax(logits_torch[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs.squeeze(0), num_samples=1).item()
        else:
            next_token_id = torch.argmax(logits_torch[:, -1, :], dim=-1).item()

        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated.append(next_token_id)
        current_pos += 1

        # Print token as it's generated
        if i % 5 == 0:
            print(".", end="", flush=True)

    print()
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    model_name = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    print(f"Pure TTNN implementation of {model_name}")
    print("=" * 80)

    # Setup device
    if os.environ.get("MESH_DEVICE") == "N150":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
    elif os.environ.get("MESH_DEVICE") == "N300":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 2]))
    else:
        device_ids = ttnn.get_device_ids()
        num_devices = len(device_ids)
        if num_devices >= 1:
            mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
        else:
            raise RuntimeError("No devices found")

    print(f"Using {mesh_device.get_num_devices()} device(s)")
    print()

    # Load model
    model = QwenModel(model_name, mesh_device, max_seq_len=2048)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare prompt
    messages = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print()
    print("Prompt:", prompt)
    print()

    # Generate
    response = generate(model, tokenizer, prompt, max_new_tokens=50)

    print()
    print("Response:", response)
    print()

    # Print validation report if any validations were run
    registry = get_validation_registry()
    if registry.results:
        registry.print_report()

    # Cleanup
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
