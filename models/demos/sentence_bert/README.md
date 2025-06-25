# SentenceBERT

### Platforms:

Wormhole N150, N300

**Note:** On N300, make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction

**bert-base-turkish-cased-mean-nli-stsb-tr** is a SentenceBERT-based model fine-tuned for semantic textual similarity and natural language inference tasks in Turkish. Built on a cased BERT architecture, it leverages mean pooling to generate dense sentence embeddings, enabling efficient and accurate sentence-level understanding. Optimized for performance in real-world NLP applications such as semantic search, clustering, and question answering.

Resource link - [source](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)

###  Details

- The entry point to the SentenceBERT model is located at:`models/demos/sentence_bert/ttnn/ttnn_sentence_bert.py`
-  Batch size: 8
- Sequence Length: 384

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings tests/ttnn/integration_tests/sentence_bert/test_ttnn_sentencebert_model.py::test_ttnn_sentence_bert_model
```

###  Performant Model with Trace+2CQ
- end-2-end perf is 419 sentences per second

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/demos/sentence_bert/tests/test_sentence_bert_e2e_performant.py
```

### Performant Demo with Trace+2CQ

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/sentence_bert/demo/demo.py::test_sentence_bert_demo_inference
```
