# MiniMax Reference Model

Model: [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)

Model family: MiniMax M2.7 (`minimax_m2`)

The `modeling_minimax_m2.py` and `configuration_minimax_m2.py` files allow the creation and use of reference model objects without installing flash_attn and all its CUDA dependencies.

This directory contains local reference code for MiniMax-style model components used by TT module tests and bring-up:

- `configuration_minimax_m2.py`
- `modeling_minimax_m2.py`
- `reference_utils.py`
- `config.json` (MiniMax M2.7 config snapshot)

Loading the model structure without loading weights or CUDA dependencies:

```python
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from models.demos.minimax_m27.reference.modeling_minimax_m2 import MiniMaxM2ForCausalLM

config = AutoConfig.from_pretrained("models/demos/minimax_m27/reference", trust_remote_code=True)

with no_init_weights():
    model = MiniMaxM2ForCausalLM._from_config(config)
```

Example usage with weights:

```python
from transformers import AutoConfig
from models.demos.minimax_m27.reference.modeling_minimax_m2 import MiniMaxM2ForCausalLM

cfg = AutoConfig.from_pretrained("/data/minimax_m27", trust_remote_code=True)
model = MiniMaxM2ForCausalLM(cfg).eval()
```

Model structure:

```python
MiniMaxM2ForCausalLM(
  (model): MiniMaxM2Model(
    (embed_tokens): Embedding(200064, 3072)
    (layers): ModuleList(
      (0-61): 62 x MiniMaxM2DecoderLayer(
        (self_attn): MiniMaxM2Attention(
          (q_proj): Linear(in_features=3072, out_features=6144, bias=False)
          (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (o_proj): Linear(in_features=6144, out_features=3072, bias=False)
          (q_norm): MiniMaxM2RMSNorm()
          (k_norm): MiniMaxM2RMSNorm()
        )
        (block_sparse_moe): MiniMaxM2SparseMoeBlock(
          (gate): Linear(in_features=3072, out_features=256, bias=False)
          (experts): MiniMaxM2Experts(
            (0-255): 256 x MiniMaxM2MLP(
              (w1): Linear(in_features=3072, out_features=1536, bias=False)
              (w2): Linear(in_features=1536, out_features=3072, bias=False)
              (w3): Linear(in_features=3072, out_features=1536, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): MiniMaxM2RMSNorm()
        (post_attention_layernorm): MiniMaxM2RMSNorm()
      )
    )
    (norm): MiniMaxM2RMSNorm()
    (rotary_emb): MiniMaxM2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=200064, bias=False)
)
```
