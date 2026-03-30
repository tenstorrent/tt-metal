# Reference Model (EAGLE3 DeepSeek R1)

Model target: [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

The `modeling_deepseek_r1.py` and `configuration_deepseek_r1.py` files provide a
comprehensive reference implementation of the DeepSeek-R1-0528 architecture without
requiring flash_attn or CUDA dependencies.

## Architecture Overview

DeepSeek-R1-0528 shares the DeepSeek-V3 architecture:

| Component | Details |
|---|---|
| Parameters | 671B total, ~37B active per token |
| Layers | 61 (3 dense + 58 MoE) |
| Hidden size | 7168 |
| Attention | Multi-Head Latent Attention (MLA) with 128 heads |
| Q LoRA rank | 1536 |
| KV LoRA rank | 512 |
| QK nope head dim | 128 |
| QK rope head dim | 64 |
| V head dim | 128 |
| Experts | 256 routed (8 active per token) + 1 shared |
| Expert groups | 8 groups, top-4 selected |
| MoE intermediate | 2048 |
| Dense intermediate | 18432 |
| RoPE | YaRN scaling (factor=40, 163K context) |
| Vocab size | 129,280 |

## Key Components

- `configuration_deepseek_r1.py` — Full `PretrainedConfig` subclass (`DeepseekR1Config`)
  with all architecture hyperparameters, plus the backward-compatible `DeepSeekR1ReferenceConfig`
  summary dataclass.
- `modeling_deepseek_r1.py` — Complete model implementation:
  - `DeepseekR1RMSNorm` — RMS normalization
  - `DeepseekR1RotaryEmbedding` / `DeepseekR1YarnRotaryEmbedding` — RoPE variants
  - `DeepseekR1MLP` — SwiGLU feedforward
  - `MoEGate` — Grouped routing with noaux_tc / bitonic sort
  - `DeepseekR1MoE` — Mixture-of-Experts with shared experts
  - `DeepseekR1Attention` — MLA with compressed KV via LoRA projections
  - `DeepseekR1DecoderLayer` — Full transformer block
  - `DeepseekR1Model` — Stack of decoder layers with embeddings
  - `DeepseekR1ForCausalLM` — Causal LM head with generation support
  - `DeepseekR1ForSequenceClassification` — Classification head
  - `DeepSeekR1ReferenceForCausalLM` — Backward-compatible wrapper
- `reference_utils.py` — Common loading utilities, bitonic sort, model introspection.
- `test_run_model.py` — Standalone smoke runner.
- `deepseek/` — Standalone inference-oriented implementation with `ModelArgs`,
  `Transformer`, FP8 kernels, and RoPE helpers.

## Loading the model structure without weights

```python
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from models.demos.speculative_deepseek_r1_broad.reference.modeling_deepseek_r1 import DeepseekR1ForCausalLM
from models.demos.speculative_deepseek_r1_broad.reference.configuration_deepseek_r1 import DeepseekR1Config

config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
r1_config = DeepseekR1Config(**{k: v for k, v in config.to_dict().items() if k != "model_type"})

with no_init_weights():
    model = DeepseekR1ForCausalLM(r1_config)
```

## Quick run (smoke test)

```bash
/proj_sw/user_dev/dchrysostomou/tt-metal/python_env/bin/python \
  models/demos/speculative_deepseek_r1_broad/reference/test_run_model.py \
  --model-id sshleifer/tiny-gpt2 \
  --prompt "Speculative decoding is" \
  --max-new-tokens 16 \
  --device cpu
```

## Structure-only inspection

```bash
/proj_sw/user_dev/dchrysostomou/tt-metal/python_env/bin/python \
  models/demos/speculative_deepseek_r1_broad/reference/test_run_model.py \
  --model-id deepseek-ai/DeepSeek-R1-0528 \
  --prompt "Hello" \
  --structure-only --trust-remote-code
```

## Model structure

```
DeepseekR1ForCausalLM(
  (model): DeepseekR1Model(
    (embed_tokens): Embedding(129280, 7168)
    (layers): ModuleList(
      (0-2): 3 x DeepseekR1DecoderLayer(
        (self_attn): DeepseekR1Attention(
          (q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
          (q_a_layernorm): DeepseekR1RMSNorm()
          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
          (kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
          (kv_a_layernorm): DeepseekR1RMSNorm()
          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
          (o_proj): Linear(in_features=16384, out_features=7168, bias=False)
          (rotary_emb): DeepseekR1YarnRotaryEmbedding()
        )
        (mlp): DeepseekR1MLP(
          (gate_proj): Linear(in_features=7168, out_features=18432, bias=False)
          (up_proj): Linear(in_features=7168, out_features=18432, bias=False)
          (down_proj): Linear(in_features=18432, out_features=7168, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): DeepseekR1RMSNorm()
        (post_attention_layernorm): DeepseekR1RMSNorm()
      )
      (3-60): 58 x DeepseekR1DecoderLayer(
        (self_attn): DeepseekR1Attention(...)
        (mlp): DeepseekR1MoE(
          (experts): ModuleList(
            (0-255): 256 x DeepseekR1MLP(
              (gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
              (up_proj): Linear(in_features=7168, out_features=2048, bias=False)
              (down_proj): Linear(in_features=2048, out_features=7168, bias=False)
              (act_fn): SiLU()
            )
          )
          (gate): MoEGate()
          (shared_experts): DeepseekR1MLP(
            (gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
            (up_proj): Linear(in_features=7168, out_features=2048, bias=False)
            (down_proj): Linear(in_features=2048, out_features=7168, bias=False)
            (act_fn): SiLU()
          )
        )
        (input_layernorm): DeepseekR1RMSNorm()
        (post_attention_layernorm): DeepseekR1RMSNorm()
      )
    )
    (norm): DeepseekR1RMSNorm()
  )
  (lm_head): Linear(in_features=7168, out_features=129280, bias=False)
)
```
