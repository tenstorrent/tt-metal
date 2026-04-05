# all-MiniLM-L6-v2

## Platforms:
    Wormhole (n150)

## Introduction
**all-MiniLM-L6-v2** is a sentence-transformers model that maps sentences to a 384-dimensional dense vector space. It uses a 6-layer BERT encoder (384 hidden, 12 heads, 1536 intermediate) fine-tuned for semantic textual similarity. The model is widely used for semantic search, clustering, and sentence embedding tasks.

Resource link - [source](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

### Baseline Model (DRAM, bfloat16)
```
pytest --disable-warnings models/demos/minilm/tests/test_minilm.py -v
```

### Optimized Model (L1 sharded, fused QKV, bfloat8_b)
```
pytest --disable-warnings models/demos/minilm/tests/test_minilm_optimized.py -v
```

### Demo (Semantic Similarity)
```
pytest --disable-warnings models/demos/minilm/demo/demo.py::test_demo_minilm -v
```

## Accuracy (PCC vs HuggingFace)

| Variant | Hidden States PCC | Sentence Embedding PCC |
|---------|-------------------|------------------------|
| Baseline (bfloat16, DRAM) | 0.9997 | 0.9998 |
| Optimized (bfloat8_b, L1 sharded) | 0.9914 | 0.9910 |

## Details

### Model Architecture
- **Parameters**: 22.7M
- **Layers**: 6
- **Hidden size**: 384
- **Attention heads**: 12
- **Intermediate size**: 1536
- **Vocabulary**: 30,522

### Implementation
- **Baseline** (`models/demos/minilm/tt/minilm_model.py`): Single-file implementation using 4D tensors `[batch, 1, seq, hidden]` and DRAM memory for all intermediates. Highest accuracy.
- **Optimized** (`models/demos/minilm/tt/minilm_optimized.py`): L1 block-sharded memory, fused QKV projection, bfloat8_b weight quantization, Wormhole compute kernel configs (HiFi2 matmul, HiFi4 layernorm). Higher throughput with slightly lower accuracy.

### Configuration
- Batch size: 8
- Sequence length: 128
- Grid: (6, 8) for optimized variant
