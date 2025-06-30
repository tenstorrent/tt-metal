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
> **Note:** SentenceBERT uses BERT-base as its backbone model.
- End-to-end performance with mean-pooling post-processing is **408 sentences per second**


Use the following command to run the performant Model with Trace+2CQs (without mean-pooling):

```
pytest --disable-warnings models/demos/sentence_bert/tests/test_sentence_bert_e2e_performant.py
```

### Performant Demo with Trace+2CQ

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/sentence_bert/demo/demo.py::test_sentence_bert_demo_inference
```

### Performant Interactive Demo with Trace+2CQ
- This script demonstrates a simple semantic search using a Turkish Sentence-BERT model. It loads a knowledge base from a text file (`knowledge_base.txt`), encodes all entries, and waits for user input. For each query, it returns the most semantically similar entry from the knowledge base using cosine similarity.
- Run the interactive demo using the command below. For every user input, the top-matching knowledge base entry along with its similarity score will be displayed.
- Type `exit` to quit the interactive demo.
- Modify the `knowledge_base.txt` file to customize the knowledge base with your own turkish input sentences.

Use the following command to run the performant interactive Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/sentence_bert/demo/interactive_demo.py::test_interactive_demo_inference
```

### Performant Dataset evaluation with Trace+2CQ

- dataset source: [STSb Turkish](https://github.com/emrecncelik/sts-benchmark-tr) (Semantic textual similarity dataset for the Turkish language)
- Adjust the `num_samples` parameter to control the number of dataset samples used during evaluation.

Use the following command to run the performant dataset evaluation with Trace+2CQs:

```
pytest --disable-warnings models/demos/sentence_bert/demo/dataset_evaluation.py::test_sentence_bert_eval
```
