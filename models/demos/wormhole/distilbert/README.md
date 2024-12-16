## Distilbert Model

# Platforms:
    WH N300, N150

## Introduction
DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. The DistilBERT Question Answering model is fine-tuned specifically for the task of extracting answers from a given context, making it highly efficient for question-answering applications.

# Details
The entry point to  distilebert model is distilbert_for_question_answering in `models/demos/wormhole/distilbert/tt/ttnn_optimized_distilbert.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `distilbert-base-uncased-distilled-squad` version from huggingface as our reference.

This model, located in `models/demos/wormhole`, supports functionality on both N150 and N300 devices, depending on availability. If the device is N300, the weights and inputs are distributed across the device, allowing the model to run in parallel.

## Sequence Size: 384

Sequence size determines the maximum length of input sequences processed by the model, optimizing performance and compatibility. It's recommended to set the `sequence_size` to 384

## Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 8

Use `pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py::test_demo[wormhole_b0-True-models.demos.wormhole.distilbert.tt.ttnn_optimized_distilbert-8-distilbert-base-uncased-distilled-squad-models/demos/wormhole/distilbert/demo/input_data.json]` to run the ttnn_optimized_distilbert demo.


If you wish to run the demo with a different input, change the pytest fixture input_loc to the desired location and use  `pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py::test_demo[wormhole_b0-True-models.demos.wormhole.distilbert.tt.ttnn_optimized_distilbert-8-distilbert-base-uncased-distilled-squad-<path to input file>]`. This file is expected to have exactly 8 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py::test_demo_squadv2[wormhole_b0-True-3-8-models.demos.wormhole.distilbert.tt.ttnn_optimized_distilbert-distilbert-base-uncased-distilled-squad]`.

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py::test_demo_squadv2[wormhole_b0-True-<n_iterations>-8-models.demos.wormhole.distilbert.tt.ttnn_optimized_distilbert-distilbert-base-uncased-distilled-squad]`

## Inputs

The demo receives inputs from respective input_data.json by default. To modify the inputs or specify a different path, adjust the input_path parameter in the command accordingly. It's recommended to avoid direct modifications to the input_data.json file.

# Owner Sudharsan Vijayaraghavan
