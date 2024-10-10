# SqueezeBERT demo

Demo showcasing SqueezeBERT running on Grayskull - e150 and Wormhole - n150, n300 using ttnn.

## Introduction
SqueezeBERT is a bidirectional transformer similar to the BERT model. The key difference between the BERT architecture and the SqueezeBERT architecture is that SqueezeBERT uses grouped convolutions instead of fully-connected layers for the Q, K, V and FFN layers.


## Details
The entry point to  functional_squeezebert model is squeezebert_for_question_answering in `models/demos/squeezebert/tt/ttnn_functional_squeezebert.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `squeezebert/squeezebert-uncased` version from huggingface as our reference.

### Sequence Size: 384
Sequence size determines the maximum length of input sequences processed by the model, optimizing performance and compatibility. It's recommended to set the sequence_size to 384

### Batch size: 16
Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. On each device, the batch size will be 8, as the operations run in parallel. It's recommended to set the batch_size to 16

## How to Run

Use `pytest --disable-warnings models/demos/wormhole/squeezebert/demo/demo.py::test_demo[wormhole_b0-True-models.demos.wormhole.squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased-models/demos/wormhole/squeezebert/demo/input_data.json-device_params0]` to run the demo.

If you wish to run the demo with a different input use `pytest --disable-warnings --input-path="<address_to_your_json_file.json>" models/demos/wormhole/squeezebert/demo/demo.py::test_demo[wormhole_b0-True-models.demos.wormhole.squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased]`. This file is expected to have exactly 16 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/wormhole/squeezebert/demo/demo.py::test_demo_squadv2[wormhole_b0-True-3-models.demos.wormhole.squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased-device_params0]`.

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/demos/wormhole/squeezebert/demo/demo.py::test_demo_squadv2[wormhole_b0-True-<n_iterations>-models.demos.wormhole.squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased-device_params0]`


## Inputs
The demo receives inputs from respective `input_data.json` by default. To modify the inputs or specify a different path, adjust the input_path parameter in the command accordingly. It's recommended to avoid direct modifications to the input_data.json file.


#### Owner: [kkeerthana0573](https://github.com/kkeerthana0573)
