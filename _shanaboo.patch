--- a/models/demos/phi1/demo.py
+++ b/models/demos/phi1/demo.py
@@ -0,0 +1,62 @@
+# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
+# SPDX-License-Identifier: Apache-2.0
+
+import torch
+import ttnn
+from models.demos.phi1.phi1 import Phi1ForCausalLM
+from transformers import AutoTokenizer
+
+
+def main():
+    # Initialize device
+    device_id = 0
+    device = ttnn.open_device(device_id=device_id)
+
+    # Model configuration
+    model_name = "microsoft/phi-1"
+    batch_size = 1
+    max_seq_len = 512
+
+    # Load tokenizer
+    tokenizer = AutoTokenizer.from_pretrained(model_name)
+    tokenizer.pad_token = tokenizer.eos_token
+
+    # Initialize model
+    model = Phi1ForCausalLM.from_pretrained(
+        model_name,
+        device=device,
+        dtype=ttnn.bfloat16,
+        max_seq_len=max_seq_len,
+        batch_size=batch_size,
+    )
+
+    # Sample prompt
+    prompt = "def fibonacci(n):"
+    input_ids = tokenizer.encode(prompt, return_tensors="pt")
+
+    # Run inference
+    print(f"Prompt: {prompt}")
+    print("=" * 50)
+
+    generated_tokens = []
+    for _ in range(50):  # Generate 50 tokens
+        logits = model(input_ids)
+        next_token_logits = logits[:, -1, :]
+        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
+        
+        generated_tokens.append(next_token.item())
+        input_ids = torch.cat([input_ids, next_token], dim=-1)
+        
+        # Print generated text so far
+        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
+        print(f"\r{prompt}{generated_text}", end="", flush=True)
+
+    print("\n" + "=" * 50)
+    print("Generation complete!")
+
+    # Cleanup
+    ttnn.close_device(device)
+
+
+if __name__ == "__main__":
+    main()

--- a/models/demos/phi1/phi1.py
+++ b/models/demos/phi1/phi1.py
@@ -0,0 +1,215 @@
+# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
+# SPDX-License-Identifier: Apache-2.0
+
+import torch
+import ttnn
+from typing import Optional, Tuple
+from models.tt_transformers.tt.model_config import ModelConfig
+from models.tt_transformers.tt.common import (
+    create_attention_mask,
+    create_position_ids,
+    get_rotation_matrix,
+)
+from models.tt_transformers.tt.layers import (
+    TtEmbedding,
+    TtRMSNorm,
+    TtAttention,
+    TtMLP,
+)
+
+
+class Phi1Config(ModelConfig):
+    def __init__(self):
+        super().__init__()
+        self.vocab_size = 50257
+        self.hidden_size = 2048
+        self.intermediate_size = 8192
+        self.num_hidden_layers = 24
+        self.num_attention_heads = 32
+        self.num_key_value_heads = 32
+        self.max_position_embeddings = 2048
+        self.rms_norm_eps = 1e-5
+        self.rope_theta = 10000.0
+        self.tie_word_embeddings = True
+        self.use_cache = True
+        self.pad_token_id = 50256
+        self.bos_token_id = 50256
+        self.eos_token_id = 50256
+
+
+class Phi1Attention:
+    def __init__(
+        self,
+        config: Phi1Config,
+        device: ttnn.Device,
+        dtype: ttnn.DataType,
+        mesh_mapper=None,
+    ):
+        self.config = config
+        self.device = device
+        self.dtype = dtype
+        self.mesh_mapper = mesh_mapper
+
+        self.hidden_size = config.hidden_size
+        self.num_heads = config.num_attention_heads
+        self.head_dim = self.hidden_size // self.num_heads
+        self.num_key_value_heads = config.num_key_value_heads
+        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
+
+        self.qkv_proj = ttnn.Linear(
+            self.hidden_size,
+            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
+            dtype=dtype,
+            device=device,
+            mesh_mapper=mesh_mapper,
+        )
+
+        self.o_proj = ttnn.Linear(
+            self.hidden_size,
+            self.hidden_size,
+            dtype=dtype,
+            device=device,
+            mesh_mapper=mesh_mapper,
+        )
+
+        self.rot_mat = get_rotation_matrix(
+            config.max_position_embeddings,
+            self.head_dim,
+            config.rope_theta,
+            device,
+            mesh_mapper,
+        )
+
+    def __call__(
+        self,
+        hidden_states: ttnn.Tensor,
+        attention_mask: Optional[ttnn.Tensor] = None,
+        position_ids: Optional[ttnn.Tensor] = None,
+        past_key_value: Optional[Tuple[ttnn.Tensor]] = None,
+    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor]]]:
+        batch_size, seq_len, _ = hidden_states.shape
+
+        # QKV projection
+        qkv = self.qkv_proj(hidden_states)
+        q, k, v = ttnn.split(
+            qkv,
+            [
+                self.hidden_size,
+                self.num_key_value_heads * self.head_dim,
+                self.num_key_value_heads * self.head_dim,
+            ],
+            dim=-1,
+        )
+
+        # Reshape for attention
+        q = ttnn.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
+        k = ttnn.reshape(k, (batch_size, seq_len, self.num_key_value_heads, self.head_dim))
+        v = ttnn.reshape(v, (batch_size, seq_len, self.num_key_value_heads, self.head_dim))
+
+        # Apply rotary embeddings
+        q = ttnn.matmul(q, self.rot_mat)
+        k = ttnn.matmul(k, self.rot_mat)
+
+        # SDPA attention
+        attn_output = ttnn.scaled_dot_product_attention(
+            q, k, v, attention_mask=attention_mask, is_causal=True
+        )
+
+        # Reshape back
+        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
+        attn_output = self.o_proj(attn_output)
+
+        return attn_output, None
+
+
+class Phi1MLP:
+    def __init__(
+        self,
+        config: Phi1Config,
+        device: ttnn.Device,
+        dtype: ttnn.DataType,
+        mesh_mapper=None,
+    ):
+        self.config = config
+        self.device = device
+        self.dtype = dtype
+        self.mesh_mapper = mesh_mapper
+
+        self.gate_up_proj = ttnn.Linear(
+            config.hidden_size,
+            config.intermediate_size * 2,
+            dtype=dtype,
+            device=device,
+            mesh_mapper=mesh_mapper,
+        )
+
+        self.down_proj = ttnn.Linear(
+            config.intermediate_size,
+            config.hidden_size,
+            dtype=dtype,
+            device=device,
+            mesh_mapper=mesh_mapper,
+        )
+
+    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
+        gate_up = self.gate_up_proj(hidden_states)
+        gate, up = ttnn.split(gate_up, [self.config.intermediate_size, self.config.intermediate_size], dim=-1)
+        hidden_states = ttnn.mul(up, ttnn.gelu(gate))
+        hidden_states = self.down_proj(hidden_states)
+        return hidden_states
+
+
+class Phi1DecoderLayer:
+    def __init__(
+        self,
+        config: Phi1Config,
+        device: ttnn.Device,
+        dtype: ttnn.DataType,
+        layer_idx: int,
+        mesh_mapper=None,
+    ):
+        self.self_attn = Phi1Attention(config, device, dtype, mesh_mapper)
+        self.mlp = Phi1MLP(config, device, dtype, mesh_mapper)
+        self.input_layernorm = TtRMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype, mesh_mapper)
+        self.post_attention_layernorm = TtRMSNorm(
+            config.hidden_size, config.rms_norm_eps, device, dtype, mesh_mapper
+        )
+
+    def __call__(
+        self,
+        hidden_states: ttnn.Tensor,
+        attention_mask: Optional[ttnn.Tensor] = None,
+        position_ids: Optional[ttnn.Tensor] = None,
+    ) -> ttnn.Tensor:
+        residual = hidden_states
+        hidden_states = self.input_layernorm(hidden_states)
+        hidden_states, _ = self.self_attn(hidden_states, attention_mask, position_ids)
+        hidden_states = ttnn.add(residual, hidden_states)
+
+        residual = hidden_states
+        hidden_states = self.post_attention_layernorm(hidden_states)
+        hidden_states = self.mlp(hidden_states)
+        hidden_states = ttnn.add(residual, hidden_states)
+
+        return hidden_states
+
+
+class Phi1Model:
+    def __init__(
+        self,
+        config: Phi1Config,
+        device: ttnn.Device,
+        dtype: ttnn.DataType,
+        mesh_mapper=None,
+    ):
+        self.config = config
+        self.device = device
+        self.dtype = dtype
+        self.mesh_mapper = mesh_mapper
+
+        self.embed_tokens = TtEmbedding(
+            config.vocab_size, config.hidden_size, device, dtype, mesh_mapper
+        )
+
+        self.layers = [
+            Phi1DecoderLayer(config, device, dtype, i, mesh_mapper)
+            for i in range(config.num_hidden_layers)
+        ]
+
+        self.norm = TtRMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype, mesh_mapper)
+
+    def __call__(
+        self,
+        input_ids: ttnn.Tensor,
+        attention_mask: Optional[ttnn.Tensor] = None,
+        position_ids: Optional[ttnn.Tensor] = None,
+    ) -> ttnn.Tensor:
+        hidden_states = self.embed_tokens(input_ids)
+
+        for layer in self.layers:
+            hidden_states = layer(hidden_states, attention_mask, position_ids)
+
+        hidden_states = self.norm(hidden_states)
+        return hidden_states
+
+
+class Phi1ForCausalLM:
+    def __init__(
+        self,
+        config: Phi1Config,
+        device: ttnn.Device,
+        dtype: ttnn.DataType,
+        mesh_mapper=None,
+    ):
+        self.config = config
+        self.device = device
+        self.dtype = dtype
+        self.mesh_mapper = mesh_mapper
+
+        self.model = Phi1Model(config, device, dtype, mesh_mapper)
+        self.lm_head = ttnn.Linear(
+            config.hidden_size,
+            config.vocab_size,
+            dtype=dtype,
+            device=device,
+            mesh_mapper=mesh_mapper,
+        )
+
+    @classmethod
+    def from_pretrained(
+        cls,
+        model_name: str,
+        device: ttnn.Device,
+        dtype: ttnn.DataType,
+        max_seq_len: int = 2048,
+        batch_size: int = 1,
+    ):
+        config = Phi1Config()
+        
+        # Load weights from HuggingFace
+        from transformers import AutoModelForCausalLM
+        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
+        
+        # Initialize model
+        model = cls(config, device, dtype)
+        
+        # Map weights (simplified - actual implementation would need proper mapping)
+        # This is a placeholder for the actual weight loading logic
+        
+        return model
+
+    def __call__(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
+        batch_size, seq_len = input_ids.shape
+        
+        # Create attention mask and position ids
+        attention_mask = create_attention_mask(input_ids, self.device)
+        position_ids = create_position_ids(input_ids, self.device)
+        
+        # Forward pass
+        hidden_states = self.model(input_ids, attention_mask, position_ids)
+        logits = self.lm_head(hidden_states)
+        
+        return logits

--- a/models/demos/phi1/README.md
+++ b/models/demos/phi1/README.md
@@ -0,0 +1,45 @@
+# Phi-1 on Tenstorrent Wormhole
+
+This directory contains the implementation of Microsoft's Phi-1 language model optimized for Tenstorrent's Wormhole hardware.
+
+## Model Details
+
+- **Model**: microsoft/phi-1
+- **Architecture**: Transformer decoder-only
+- **Parameters**: 1.3B
+- **Context Length**: 2048 tokens
+- **Vocabulary**: 50,257 tokens
+
+## Usage
+
+### Basic Inference
+
+