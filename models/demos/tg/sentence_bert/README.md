# SentenceBERT

## Platforms:
    Galaxy

## Introduction
**bert-base-turkish-cased-mean-nli-stsb-tr** is a SentenceBERT-based model fine-tuned for semantic textual similarity and natural language inference tasks in Turkish. Built on a cased BERT architecture, it leverages mean pooling to generate dense sentence embeddings, enabling efficient and accurate sentence-level understanding. Optimized for performance in real-world NLP applications such as semantic search, clustering, and question answering.

Resource link - [source](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)

This TG variant is a scaled version tailored for Tenstorrent Galaxy hardware.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
    - To obtain perf reports through the profiler, build with: `./build_metal.sh -p`

## How to Run
- Use the following command to run the model:
```
pytest --disable-warnings models/demos/sentence_bert/tests/pcc/test_ttnn_sentencebert_model.py::test_ttnn_sentence_bert_model
```

### Performant Model with Trace+2CQ
> **Note:** SentenceBERT uses BERT-base as its backbone model.
- End-to-end performance with mean-pooling post-processing is **10064 sentences per second**

Run the performant model with Trace+2CQs (with mean-pooling and random inputs):
```
pytest --disable-warnings models/demos/tg/sentence_bert/tests/test_sentence_bert_e2e_performant.py::test_e2e_performant_sentencebert_data_parallel
```

### Performant Demo with Trace+2CQ
Run the performant demo with Trace+2CQs:
```
pytest --disable-warnings models/demos/tg/sentence_bert/demo/demo.py::test_sentence_bert_demo_inference
```

### Performant Interactive Demo with Trace+2CQ
- This script demonstrates a simple semantic search using a Turkish Sentence-BERT model. It loads a knowledge base from a text file (`knowledge_base.txt`), encodes all entries, and waits for user input. For each query, it returns the most semantically similar entry from the knowledge base using cosine similarity.
- Run the interactive demo using the command below. For every user input, the top-matching knowledge base entry along with its similarity score will be displayed.
- Type `exit` to quit the interactive demo.
- Modify the `knowledge_base.txt` file to customize the knowledge base with your own Turkish input sentences.

Run the interactive demo:
```
pytest --disable-warnings models/demos/tg/sentence_bert/demo/interactive_demo.py::test_interactive_demo_inference
```

## Testing
### Performant Dataset evaluation with Trace+2CQ
- End-to-end performance with mean-pooling post-processing is **9980 sentences per second**
- Dataset source: [STSb Turkish](https://github.com/emrecncelik/sts-benchmark-tr) (Semantic textual similarity dataset for the Turkish language)
- Adjust the `num_samples` parameter to control the number of dataset samples used during evaluation.

Run the dataset evaluation:
```
pytest --disable-warnings models/demos/tg/sentence_bert/demo/dataset_evaluation.py::test_sentence_bert_eval_data_parallel
```

## Details
- The entry point to the SentenceBERT model is located at: `models/demos/sentence_bert/ttnn/ttnn_sentence_bert_model.py`
- Batch size: 8 (Single Device)
- Sequence Length: 384
