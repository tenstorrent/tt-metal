# TinyTimeMixer for TTNN

This directory contains an implementation of the IBM Granite Timeseries TTM-R1 model using TTNN APIs for Tenstorrent hardware.

## Model Description

TinyTimeMixer (TTM) is a compact (<1M parameters) pre-trained model for multivariate time-series forecasting. It is designed for efficient zero-shot and few-shot forecasting. This implementation ports the official Hugging Face model `ibm-granite/granite-timeseries-ttm-r1` to TTNN.

## Setup

1.  **Environment**: Ensure you are in the `tt-metal` environment with all dependencies installed.
2.  **Install Transformers**: The model loader and evaluation scripts require the Hugging Face `transformers` library.
    ```bash
    pip install transformers
    ```
3.  **Install Datasets**: For evaluation on benchmark datasets.
    ```bash
    pip install datasets
    ```

## Verification

To verify the correctness of the TTNN model against the original Hugging Face implementation, run the following test. This test loads the pre-trained weights, runs both models with the same input, and asserts that their outputs are numerically close.

```bash
pytest models/tt_tinytimemixer/test_ttnn.py
```

## Evaluation

To evaluate the zero-shot performance of the TTNN model on a benchmark dataset, use the `evaluate.py` script.

**Note**: The current script uses placeholder data. To run on a real dataset like ETT, you will need to download the data and modify the `get_ett_data` function in the script.

```bash
python models/tt_tinytimemixer/evaluate.py
```

## Current Status & Limitations

This implementation is a work in progress towards completing the bounty requirements.

### Stage 1: Bring-Up (Partially Complete)
- **Architecture**: The core mixer blocks, normalization, and MLP layers have been ported to TTNN.
- **Data Flow**: The model is now almost fully end-to-end on device, with scaling and patching operations ported to TTNN to minimize host-device data transfers.
- **Verification**: A test is in place (`test_ttnn.py`) to compare outputs with the official Hugging Face model.

### Known Limitations:
- **Adaptive Patching**: The official model uses an "adaptive patching" mechanism. This implementation currently uses a **fixed-size patcher** (`ttnn.experimental.tensor.sliding_window`). The implementation of the learnable adaptive patching is a major remaining item for full architectural compliance.
- **Performance**: Stage 2 and Stage 3 optimizations (sharding, op fusion, etc.) have not been implemented. The current focus is on correctness and basic bring-up.