# Distilbert

## Platforms:
    Wormhole (n150, n300)

## Introduction
DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. The DistilBERT Question Answering model is fine-tuned specifically for the task of extracting answers from a given context, making it highly efficient for question-answering applications.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- Run the ttnn_optimized_distilbert demo.
```
pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py::test_demo
```

## Details
- The entry point to  distilebert model is distilbert_for_question_answering in `models/demos/wormhole/distilbert/tt/ttnn_optimized_distilbert.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `distilbert-base-uncased-distilled-squad` version from huggingface as our reference.
