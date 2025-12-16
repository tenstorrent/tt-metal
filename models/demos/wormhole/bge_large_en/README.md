# BGE-Large-EN-v1.5

## Platforms:
    Wormhole (n150, n300), T3K, Galaxy

## Introduction
**BAAI/bge-large-en-v1.5** (BAAI General Embedding) is a state-of-the-art sentence embedding model developed by Beijing Academy of Artificial Intelligence. It achieves top performance on the MTEB English benchmark. Built on BERT-large architecture, it leverages mean pooling to generate dense sentence embeddings of 1024 dimensions, enabling highly accurate semantic search, text clustering, and information retrieval tasks.

Resource link - [source](https://huggingface.co/BAAI/bge-large-en-v1.5)

## Model Architecture
- **Hidden Size**: 1024 (vs 768 in BERT-base)
- **Layers**: 24 (vs 12 in BERT-base)
- **Attention Heads**: 16 (vs 12 in BERT-base)
- **Intermediate Size**: 4096 (vs 3072 in BERT-base)
- **Max Sequence Length**: 512
- **Vocabulary Size**: 30522
- **Embedding Dimension**: 1024

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- Use the following command to run the model:
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_model.py::test_ttnn_bge_model
```

###  Performant Model with Trace+2CQ
> **Note:** BGE-Large uses BERT-large as its backbone model. The model supports both 384 and 512 sequence lengths.

#### Single Device (BS=8):
- End-to-end performance with mean-pooling post-processing:
  - **Sequence Length 384**: **230 sentences per second**
  - **Sequence Length 512**: **146 sentences per second**
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/perf/test_bge_e2e_performant.py::test_e2e_performant_bge
```

#### Multi Device (DP=2, n300):
- End-to-end performance with mean-pooling post-processing
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/perf/test_bge_e2e_performant.py::test_e2e_performant_bge_dp
```

### Performant Demo with Trace+2CQ
#### Single Device (BS=8):
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/demo.py::test_bge_demo_inference
```

#### Multi Device (DP=2, n300):
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/demo.py::test_bge_demo_inference_dp
```

### Performant Interactive Demo with Trace+2CQ
- This script demonstrates semantic search using the BGE-large-en-v1.5 model. It loads a knowledge base from a text file (`knowledge_base.txt`), encodes all entries, and waits for user input. For each query, it returns the most semantically similar entry from the knowledge base using cosine similarity.
- Run the interactive demo using the command below. For every user input, the top-matching knowledge base entry along with its similarity score will be displayed.
- Type `exit` to quit the interactive demo.
- Modify the `knowledge_base.txt` file to customize the knowledge base with your own input sentences.
- **Note:** For retrieval tasks, BGE models work best with the instruction prefix: "Represent this sentence for searching relevant passages: "

#### Single Device (BS=8):
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/interactive_demo.py::test_interactive_demo_inference
```

#### Multi Device (DP=2, n300):
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/interactive_demo.py::test_interactive_demo_inference_dp
```

## Testing
### Performant Dataset evaluation with Trace+2CQ
- The `dataset_evaluation.py` script supports evaluation on MTEB datasets including:
  - **Retrieval tasks** (e.g., ArguAna): Evaluates using Recall@K and nDCG@K metrics
  - **Semantic Textual Similarity tasks** (e.g., STSBenchmark): Evaluates using Spearman's correlation
- Datasets are automatically downloaded from Hugging Face
- Adjust the `max_samples` parameter to control the number of dataset samples used during evaluation

#### Single Device (BS=8):
- Retrieval task evaluation:
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/dataset_evaluation.py::test_bge_retrieval_evaluation
```
- Semantic Textual Similarity task evaluation:
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/dataset_evaluation.py::test_bge_sts_evaluation
```

#### Multi Device (DP=2, n300):
- Retrieval task evaluation:
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/dataset_evaluation.py::test_bge_retrieval_evaluation_dp
```
- Semantic Textual Similarity task evaluation:
```
pytest --disable-warnings models/demos/wormhole/bge_large_en/demo/dataset_evaluation.py::test_bge_sts_evaluation_dp
```

##  Details
- The entry point to the BGE model is located at: `models/demos/bge_large_en/ttnn/ttnn_bge_model.py`
- Batch size: 8
- Sequence Length: Supports both **384** and **512** sequence lengths

### Changing Sequence Length
To change the sequence length from 384 to 512 (or vice versa), modify the `BGE_SEQ_LENGTH` constant in `models/demos/wormhole/bge_large_en/ttnn/common.py`:

```python
BGE_SEQ_LENGTH = 512  # Change from 384 to 512
```

**Note:** This change affects:
- `demo.py` - Demo inference scripts
- `dataset_evaluation.py` - Dataset evaluation scripts
- `test_bge_e2e_performant.py` - End-to-end performance tests

All these files import `BGE_SEQ_LENGTH` from `common.py`, so modifying it in one place will update the sequence length across all demos, evaluations, and performance tests.

## Performance
BGE-Large-EN-v1.5 achieves state-of-the-art results on:
- MTEB English benchmark (Rank #1)
- Semantic textual similarity tasks
- Retrieval and information extraction
- Text classification and clustering
