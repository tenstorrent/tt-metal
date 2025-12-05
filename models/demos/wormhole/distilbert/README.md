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

- Our second demo is designed to run SQuADV2 dataset, run this with:
```
pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py::test_demo_squadv2
```

## Details
- The entry point to  distilebert model is distilbert_for_question_answering in `models/demos/wormhole/distilbert/tt/ttnn_optimized_distilbert.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `distilbert-base-uncased-distilled-squad` version from huggingface as our reference.

**Sequence Size: 384**
- Sequence size determines the maximum length of input sequences processed by the model, optimizing performance and compatibility. It's recommended to set the `sequence_size` to 384

**Batch size: 8**
- Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 8

### Inputs
- The demo receives inputs from respective input_data.json by default. To modify the inputs or specify a different path, adjust the input_path parameter in the command accordingly. It's recommended to avoid direct modifications to the input_data.json file.

**Model Owner:** Sudharsan Vijayaraghavan
