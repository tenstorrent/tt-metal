# SentenceBERT Model

### Platforms:
    WH N300,N150

### Introduction

**bert-base-turkish-cased-mean-nli-stsb-tr** is a SentenceBERT-based model fine-tuned for semantic textual similarity and natural language inference tasks in Turkish. Built on a cased BERT architecture, it leverages mean pooling to generate dense sentence embeddings, enabling efficient and accurate sentence-level understanding. Optimized for performance in real-world NLP applications such as semantic search, clustering, and question answering.

Resource link - [source](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)

### Model Details

- The entry point to the SentenceBERT is located at:`models/experimental/sentence_bert/ttnn/ttnn_sentence_bert.py`

Export the following command before running pytests:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

Use the following command to run the model :

`pytest tests/ttnn/integration_tests/sentence_bert/test_ttnn_sentencebert_model.py:test_ttnn_sentence_bert_model`

Note : The model currently supports a batch size of 8 for a sequence length of 384.
