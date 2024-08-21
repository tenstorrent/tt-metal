# metal_BERT_large 11 Demo

>[!WARNING]
>
> This model demo does not work on N150 Wormhole cards.

## How to Run

### Batch support for all architectures

The optimized demos will parallelize batch on one of the device grid dimensions. The grid size used is `batch x 8` or `8 x batch` depending on your device grid.

For E150 (unharvested) Grayskull, the model demo supports batch 2 - 12, so you can use `batch_12` for `BATCH_SIZE` for the following commands.

For Wormhole N300, the model demo supports batch 2 - 7, so you can use `batch_7` for `BATCH_SIZE` for the following commands.

Replace `BATCH_SIZE` with the appropriate size depending on your device.
Use `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo -k BATCH_SIZE` to run the demo for Grayskull.

If you wish to run the demo with a different input use `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo[address_to_your_json_file.json-1-BATCH_SIZE]`. This file is expected to have exactly `BATCH_SIZE` inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo_squadv2 -k BATCH_SIZE`.

The table below summarizes the information above.

| Batch size | Supported on Grayskull (E150) | Supported on Wormhole (N300)         |
|------------|-------------------------------|--------------------------------------|
| 7          | :x:                           | :white_check_mark:                   |
| 8          | :x:                           | See under construction section below |
| 12         | :white_check_mark:            | :x:                                  |

## Inputs

Inputs by default are provided from `input_data.json`. If you wish you to change the inputs or provide a different path to `test_demo`.

We do not recommend modifying `input_data.json` file.

## Details

The entry point to metal bert model is `TtBertBatchDram` in `bert_model.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `phiyodr/bert-large-finetuned-squad2` version from huggingface as our reference.

For fast model loading, we have cached preprocessed weights for TT tensors on Weka. These weights are directly read in and loaded to device.

If your machine does not have access to Weka, during model loading it will preprocess and convert the pytorch weights from huggingface to TT tensors before placing on device.

## Under construction

> [!NOTE]
>
> This section is under construction and is not guaranteed to work under all conditions.
>
> If you are using Wormhole, you must set the `WH_ARCH_YAML` environment variable to use  the following batch sizes:
>
> - `batch_8`
>
> ```
> export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
> ```

We currently do not have demos that show batch sizes other than 7 or 12.

N300 can also theoretically support batch 8, if `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` is added to the environment variables, `batch_8` can be added to the command.