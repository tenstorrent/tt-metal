# BGE-M3 User Guide

## Low-level model creation

Use `create_tt_model()` when you want the raw TT encoder model.

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

You can then tokenize with `model_args.encode_prompts(...)` and pass `input_ids`, `attention_mask`, and `token_type_ids` to `tt_model`.

## Embedding API

For dense, sparse, and ColBERT-style embeddings, use `BgeM3ForEmbedding`.

```python
import torch
import torch.nn.functional as F
import ttnn

from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding
from models.demos.wormhole.bge_m3.demo.m3_scores import (
    compute_colbert_score_torch,
    compute_dense_score_torch,
    compute_sparse_score_torch,
)

device = ttnn.open_device(device_id=0)

sentences_1 = ["What is BGE M3?", "Definition of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks documents based on matching query terms.",
]

model = BgeM3ForEmbedding(
    device=device,
    max_batch_size=2,
    max_seq_len=512,
    tt_data_parallel=1,
    dtype=ttnn.bfloat8_b,
    model_name="BAAI/bge-m3",
    sentence_pooling_method="cls",
    return_dense=True,
    return_sparse=True,
    return_colbert=True,
)
model._initialize_model()
model_args = model.model_args
```

Notes:

- The current generator path is single-device.
- `sentence_pooling_method` controls how `dense_vecs` are produced from the last hidden state.
- The default is `"mean"`, which averages token embeddings across the non-padded tokens in the prompt.
- `"cls"` pools from the first token and matches the reference setup used in `tests/pcc/test_generator_vllm.py`.
- `"last_token"` pools from the last valid token in each prompt.
- The returned tensors are padded to `max_batch_size`, so slice back to your real batch size.

## Dense pooling modes

`BgeM3ForEmbedding` currently supports these `sentence_pooling_method` values:

- `"mean"`: (default) averages token embeddings using the attention mask.
- `"cls"`: returns the embedding from the first token position.
- `"last_token"`: returns the embedding from the last non-padding token.

Example with the default behavior:

```python
model = BgeM3ForEmbedding(
    device=device,
    max_batch_size=2,
    max_seq_len=512,
    model_name="BAAI/bge-m3",
    return_dense=True,
)
```

## Run inference (Example)

```python
def encode(sentences,model_args,model):
    encoded = model_args.encode_prompts(sentences)
    outputs = model.forward(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        token_type_ids=encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])),
    )

    seq_len = encoded["input_ids"].shape[1]
    batch_size = len(sentences)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "dense_vecs": outputs["dense_vecs"][:batch_size].to(torch.float32),
        "dense_vecs_norm": F.normalize(outputs["dense_vecs"][:batch_size].to(torch.float32), dim=-1),
        "sparse_vecs": outputs["sparse_vecs"][:batch_size].to(torch.float32),
        "colbert_vecs": outputs["colbert_vecs"][:batch_size, : seq_len - 1].to(torch.float32),
        "colbert_vecs_norm": F.normalize(outputs["colbert_vecs"][:batch_size, : seq_len - 1].to(torch.float32), dim=-1),
    }

embeddings_1 = encode(sentences_1)
embeddings_2 = encode(sentences_2)
```

## Dense retrieval

`dense_vecs` are sentence embeddings. Normalize them before computing similarity.

```python
similarity = compute_dense_score_torch(
    embeddings_1["dense_vecs_norm"],
    embeddings_2["dense_vecs_norm"],
)
print(similarity)
```

## Sparse retrieval

`sparse_vecs` are lexical-weight vectors over the vocabulary. Use them for sparse matching.

```python
sparse_scores = compute_sparse_score_torch(
    embeddings_1["sparse_vecs"],
    embeddings_2["sparse_vecs"],
)
print(sparse_scores)
```

## ColBERT / multi-vector retrieval

`colbert_vecs` are token-level multi-vector embeddings. Normalize them before scoring.

```python
colbert_scores = compute_colbert_score_torch(
    embeddings_1["colbert_vecs_norm"],
    embeddings_2["colbert_vecs_norm"],
    q_mask=embeddings_1["attention_mask"],
)
print(colbert_scores)
```

The ColBERT path skips the first token internally, which is why the examples slice token vectors to `: seq_len - 1`.

## Reference examples

- `models/demos/wormhole/bge_m3/tests/pcc/test_generator_vllm.py`
