# Reference Files

Model: [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)

This directory holds *small* reference artifacts that let us load the HF
config / tokenizer / modeling code without pulling the full 230 GB FP8
checkpoint. Mirrors the layout of `models/demos/deepseek_v3/reference/`.

## Files to pull

The MiniMax repo ships its own modeling code via `trust_remote_code=True`.
Grab just the small files (the `*.safetensors` shards are intentionally
excluded):

```bash
huggingface-cli download MiniMaxAI/MiniMax-M2.7 \
    config.json \
    configuration_minimax_m2.py \
    modeling_minimax_m2.py \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    --local-dir models/demos/minimax2.7/reference \
    --local-dir-use-symlinks False
```

(If `modeling_minimax_m2.py` isn't present in the repo because MiniMax has
upstreamed it into `transformers`, install
`transformers>=4.57.0` and rely on the upstream `MiniMaxM2ForCausalLM`
class instead — both M2.5 and M2.7 ship the file in the repo today.)

## Loading the reference module without weights

```python
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights

config = AutoConfig.from_pretrained(
    "models/demos/minimax2.7/reference",
    trust_remote_code=True,
)

with no_init_weights():
    from transformers import AutoModelForCausalLM
    ref_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

Use `ref_model` for golden-tensor comparisons in `tests/unit/`.

## Pulling actual FP8 weights (large)

Only do this on a host with ~230 GB free in your HF cache:

```bash
huggingface-cli download MiniMaxAI/MiniMax-M2.7 \
    --local-dir /mnt/MLPerf/tt_dnn-models/MiniMaxAI/MiniMax-M2.7 \
    --local-dir-use-symlinks False
export HF_MODEL=/mnt/MLPerf/tt_dnn-models/MiniMaxAI/MiniMax-M2.7
```
