# Reference Model

Model: [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

The `modeling_deepseek.py` and `configuration_deepseek.py` files allow the creation and use of reference model objects without installing flash_attn and all its CUDA dependencies.

The other files here are experimental extractions / implementations by LLMs, so do not trust them blindly.

Loading the model structure without loading weights or CUDA dependencies:

```python
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM

config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)

with no_init_weights():
    model = DeepseekV3ForCausalLM._from_config(config)
```

Model structure:

```python
DeepseekV3ForCausalLM(
  (model): DeepseekV3Model(
    (embed_tokens): Embedding(129280, 7168)
    (layers): ModuleList(
      (0-2): 3 x DeepseekV3DecoderLayer(
        (self_attn): DeepseekV3Attention(
          (q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
          (q_a_layernorm): DeepseekV3RMSNorm()
          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
          (kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
          (kv_a_layernorm): DeepseekV3RMSNorm()
          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
          (o_proj): Linear(in_features=16384, out_features=7168, bias=False)
          (rotary_emb): DeepseekV3YarnRotaryEmbedding()
        )
        (mlp): DeepseekV3MLP(
          (gate_proj): Linear(in_features=7168, out_features=18432, bias=False)
          (up_proj): Linear(in_features=7168, out_features=18432, bias=False)
          (down_proj): Linear(in_features=18432, out_features=7168, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): DeepseekV3RMSNorm()
        (post_attention_layernorm): DeepseekV3RMSNorm()
      )
      (3-60): 58 x DeepseekV3DecoderLayer(
        (self_attn): DeepseekV3Attention(
          (q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
          (q_a_layernorm): DeepseekV3RMSNorm()
          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
          (kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
          (kv_a_layernorm): DeepseekV3RMSNorm()
          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
          (o_proj): Linear(in_features=16384, out_features=7168, bias=False)
          (rotary_emb): DeepseekV3YarnRotaryEmbedding()
        )
        (mlp): DeepseekV3MoE(
          (experts): ModuleList(
            (0-255): 256 x DeepseekV3MLP(
              (gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
              (up_proj): Linear(in_features=7168, out_features=2048, bias=False)
              (down_proj): Linear(in_features=2048, out_features=7168, bias=False)
              (act_fn): SiLU()
            )
          )
          (gate): MoEGate()
          (shared_experts): DeepseekV3MLP(
            (gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
            (up_proj): Linear(in_features=7168, out_features=2048, bias=False)
            (down_proj): Linear(in_features=2048, out_features=7168, bias=False)
            (act_fn): SiLU()
          )
        )
        (input_layernorm): DeepseekV3RMSNorm()
        (post_attention_layernorm): DeepseekV3RMSNorm()
      )
    )
    (norm): DeepseekV3RMSNorm()
  )
  (lm_head): Linear(in_features=7168, out_features=129280, bias=False)
)
```
