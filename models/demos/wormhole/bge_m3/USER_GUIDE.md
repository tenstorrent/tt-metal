# BGE-M3 User Guide

## Create a TT model

```python
import ttnn

from models.demos.wormhole.bge_m3.tt.common import create_tt_model

device = ttnn.open_device(device_id=0)

model_args, tt_model, state_dict = create_tt_model(
    mesh_device=device,
    max_batch_size=1,
    max_seq_len=128,
    dtype=ttnn.bfloat16,
    hf_model_name="BAAI/bge-m3",
)
```

## Encode prompts and run the TT model

```python
import torch
import ttnn

sentences = ["Artificial intelligence is transforming search."]

encoded = model_args.encode_prompts(sentences)

input_ids = ttnn.from_torch(
    encoded["input_ids"].to(torch.int32),
    device=device,
    dtype=ttnn.uint32,
)
attention_mask = ttnn.from_torch(
    encoded["attention_mask"].to(torch.int32),
    device=device,
    dtype=ttnn.uint32,
)
token_type_ids = ttnn.from_torch(
    encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).to(torch.int32),
    device=device,
    dtype=ttnn.uint32,
)

tt_output = tt_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
)
```

## Current status

`tests/pcc/test_model.py` passes for most sequence lengths, but `test_model` still fails at:

- `seq_len=2048` with PCC `0.9396697736415722`
- `seq_len=4096` with PCC `0.9197336036560791`

This still needs more work.

## Next tasks

- Reference this FlagEmbedding section for scoring logic: [BGE-M3 scoring reference](https://github.com/FlagOpen/FlagEmbedding/blob/dbc600560b2dadcc1514989092f7b849673bb67d/FlagEmbedding/inference/embedder/encoder_only/m3.py#L482)
- Implement sparse score support
- Implement ColBERT score support
- Optimize program configuration and memory configuration


### TTT-v2 style reference

For tt-transformer v2 implementation please reference : models/common/modules
