# metal_BERT_large 11

## Platforms:
    Grayskull (e150), Wormhole (n150, n300)

## Introduction
BERT stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
### Batch Support Chart
Replace `BATCH_SIZE` with the appropriate size for your device according to this chart:
| Batch size | Grayskull (e150)   | Wormhole (n150)    | Wormhole (n300)    |
|------------|--------------------|--------------------|--------------------|
| 7          | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 8          | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 12         | :white_check_mark: | :x:                | :x:                |

### Run the Demo
```
pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo -k BATCH_SIZE
```

- If you wish to run the demo with a different input use:
```
pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo[address_to_your_json_file.json-1-BATCH_SIZE]
```
This file is expected to have exactly `BATCH_SIZE` inputs.

- Our second demo is designed to run SQuADV2 dataset, run this with:
```
pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo_squadv2 -k BATCH_SIZE
```

## Details
- The optimized demos will parallelize batch on one of the device grid dimensions. The grid size used is `batch x 8` or `8 x batch` depending on your device grid.
- The entry point to metal bert model is `TtBertBatchDram` in `bert_model.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `phiyodr/bert-large-finetuned-squad2` version from huggingface as our reference.
- For fast model loading, we have cached preprocessed weights for TT tensors on Weka. These weights are directly read in and loaded to device.
- If your machine does not have access to Weka, during model loading it will preprocess and convert the pytorch weights from huggingface to TT tensors before placing on device.

### Inputs
- Inputs by default are provided from `input_data.json`. If you wish you to change the inputs or provide a different path to `test_demo`.
- We do not recommend modifying `input_data.json` file.
