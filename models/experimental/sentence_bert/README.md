# SentenceBERT Model

## Platforms:
    WH N300,N150

## Introduction

**bert-base-turkish-cased-mean-nli-stsb-tr** is a SentenceBERT-based model fine-tuned for semantic textual similarity and natural language inference tasks in Turkish. Built on a cased BERT architecture, it leverages mean pooling to generate dense sentence embeddings, enabling efficient and accurate sentence-level understanding. Optimized for performance in real-world NLP applications such as semantic search, clustering, and question answering.

Resource link - [source](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)

## Model Details
The entry point to the SentenceBERT model is located at:`models/experimental/sentence_bert/ttnn/ttnn_sentence_bert.py`
- Sequence Length: 384
-  Batch size: 8

## How to Run:
If running on Wormhole N300 (not required for N150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:
```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

### Build Command to Use:
To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

Use the following command to run the model :

```
pytest --disable-warnings tests/ttnn/integration_tests/sentence_bert/test_ttnn_sentencebert_model.py::test_ttnn_sentence_bert_model
```

##  Performant Model with Trace+2CQ
- end-2-end perf is 403 sentences per second

Use the following command to run the performant Model with Trace+2CQ:

```
pytest --disable-warnings models/experimental/sentence_bert/tests/test_sentence_bert_e2e_performant.py
```
