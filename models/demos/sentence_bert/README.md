# SentenceBERT

## Platforms:
    Wormhole (n150, n300)

## Introduction
**bert-base-turkish-cased-mean-nli-stsb-tr** is a SentenceBERT-based model fine-tuned for semantic textual similarity and natural language inference tasks in Turkish. Built on a cased BERT architecture, it leverages mean pooling to generate dense sentence embeddings, enabling efficient and accurate sentence-level understanding. Optimized for performance in real-world NLP applications such as semantic search, clustering, and question answering.

Resource link - [source](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
    - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
- Use the following command to run the model:
```
pytest --disable-warnings models/demos/sentence_bert/tests/pcc/test_ttnn_sentencebert_model.py::test_ttnn_sentence_bert_model
```

###  Performant Model with Trace+2CQ
```
pytest --disable-warnings models/demos/sentence_bert/tests/perf/test_sentence_bert_e2e_performant.py::test_e2e_performant_sentencebert
```

###  Performant Model with Trace+2CQ
> **Note:** SentenceBERT uses BERT-base as its backbone model.

#### Single Device (BS=8):
- End-to-end performance with mean-pooling post-processing is **408 sentences per second**
```
pytest --disable-warnings models/demos/sentence_bert/tests/perf/test_sentence_bert_e2e_performant.py::test_e2e_performant_sentencebert
```

#### Multi Device (DP=2, n300):
- End-to-end performance with mean-pooling post-processing is **757 sentences per second**
```
pytest --disable-warnings models/demos/sentence_bert/tests/perf/test_sentence_bert_e2e_performant.py::test_e2e_performant_sentencebert_dp
```

### Performant Demo with Trace+2CQ
#### Single Device (BS=8):
```
pytest --disable-warnings models/demos/sentence_bert/demo/demo.py::test_sentence_bert_demo_inference
```

#### Multi Device (DP=2, n300):
```
pytest --disable-warnings models/demos/sentence_bert/demo/demo.py::test_sentence_bert_demo_inference_dp
```

### Performant Interactive Demo with Trace+2CQ
- This script demonstrates a simple semantic search using a Turkish Sentence-BERT model. It loads a knowledge base from a text file (`knowledge_base.txt`), encodes all entries, and waits for user input. For each query, it returns the most semantically similar entry from the knowledge base using cosine similarity.
- Run the interactive demo using the command below. For every user input, the top-matching knowledge base entry along with its similarity score will be displayed.
- Type `exit` to quit the interactive demo.
- Modify the `knowledge_base.txt` file to customize the knowledge base with your own turkish input sentences.

#### Single Device (BS=8):
```
pytest --disable-warnings models/demos/sentence_bert/demo/interactive_demo.py::test_interactive_demo_inference
```

#### Multi Device (DP=2, n300):
```
pytest --disable-warnings models/demos/sentence_bert/demo/interactive_demo.py::test_interactive_demo_inference_dp
```

## Testing
### Performant Dataset evaluation with Trace+2CQ
- dataset source: [STSb Turkish](https://github.com/emrecncelik/sts-benchmark-tr) (Semantic textual similarity dataset for the Turkish language)
- Adjust the `num_samples` parameter to control the number of dataset samples used during evaluation.

#### Single Device (BS=8):
```
pytest --disable-warnings models/demos/sentence_bert/demo/dataset_evaluation.py::test_sentence_bert_eval
```

#### Multi Device (DP=2, n300):
```
pytest --disable-warnings models/demos/sentence_bert/demo/dataset_evaluation.py::test_sentence_bert_eval_dp
```

##  Details
- The entry point to the SentenceBERT model is located at:`models/demos/sentence_bert/ttnn/ttnn_sentence_bert.py`
-  Batch size: 8
- Sequence Length: 384
