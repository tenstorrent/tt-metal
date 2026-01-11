# PatchTSMixer

PatchTSMixer is a lightweight MLP-Mixer based architecture designed for multivariate time series forecasting. The model uses patching to convert time series into sequences of patches, then applies mixer layers to capture temporal patterns and cross-variate dependencies efficiently.

### Architecture

The PatchTSMixer architecture consists of:
- **Patchify Layer**: Converts input time series into fixed-length patches using a sliding window approach
- **Linear Projection**: Projects each patch from patch_length dimensions to d_model hidden dimensions
- **Positional Encoding**: Adds sinusoidal positional embeddings to preserve temporal ordering information
- **Mixer Blocks**: Stack of mixer layers, each containing:
  - **Patch Mixer**: MLP applied across the patch (time) dimension to capture temporal dependencies
  - **Feature Mixer**: MLP applied across the feature dimension to mix hidden representations
  - **Channel Mixer** (optional): MLP applied across channels for cross-variate modeling in mix_channel mode
- **Forecasting Head**: Linear projection that flattens patch representations and projects to prediction horizon

### Model Details

- **Input**: Multivariate time series of shape (batch_size, context_length, num_channels)
- **Output**: Future predictions of shape (batch_size, prediction_length, num_channels)
- **Task**: Multivariate time series forecasting
- **Modes**:
  - `common_channel`: Channel-independent forecasting (simpler, faster)
  - `mix_channel`: Cross-channel modeling for capturing dependencies between variables
- **Precision**: FP32 for reference implementation, with BFloat16 support for TT-NN

## Getting Started

### Reference Implementation

We provide PyTorch and HuggingFace reference implementations for validation:

#### Train PyTorch Model

Train the custom PyTorch implementation on ETTh2 dataset:

```bash
cd models/demos/patchtsmixer/reference
python main.py --num_epochs 10 --batch_size 64 --d_model 16 --num_layers 4
```

#### Train HuggingFace Model

Train using HuggingFace Transformers library:

```bash
cd models/demos/patchtsmixer/reference
python train_patchtsmixer_etth2.py
```

#### Compare Implementations

Validate that PyTorch and HuggingFace implementations match:

```bash
cd models/demos/patchtsmixer/reference
python compare_implementations.py \
    --pytorch_checkpoint simple_patchtsmixer_etth2/best_model.pt \
    --hf_model patchtsmixer/etth2/simple_model/
```

This will:
1. Load both trained models
2. Run inference on the same test data
3. Compare:
   - Model parameter counts (should be identical)
   - Prediction outputs
   - Performance metrics (MSE, MAE, RMSE)
4. Generate a detailed comparison report

### Configuration Options

Key hyperparameters for training:

- `--context_length`: Length of input sequence (default: 512)
- `--prediction_length`: Length of forecast horizon (default: 96)
- `--patch_length`: Size of each patch (default: 8)
- `--patch_stride`: Stride between patches (default: 8)
- `--d_model`: Hidden dimension size (default: 16)
- `--num_layers`: Number of mixer layers (default: 4)
- `--mode`: Mixing mode - `common_channel` or `mix_channel` (default: common_channel)
- `--use_gated_attn`: Enable gated attention mechanism
- `--dropout`: Dropout rate for regularization (default: 0.1)
- `--batch_size`: Training batch size (default: 64)
- `--num_epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)

### Dataset

The reference implementation uses the ETTh2 (Electricity Transformer Temperature - Hourly) dataset:
- **Source**: [ETDataset](https://github.com/zhouhaoyi/ETDataset)
- **Variables**: 7 multivariate features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- **Splits**:
  - Train: First 12 months (8,640 hours)
  - Validation: Months 12-16 (2,880 hours)
  - Test: Months 16-20 (2,880 hours)

## Model Visualization

Generate architecture diagrams:

```python
from torchview import draw_graph
from pytorch_patchtsmixer import PatchTSMixerModelForForecasting

model = PatchTSMixerModelForForecasting(
    context_length=512,
    prediction_length=96,
    patch_length=8,
    patch_stride=8,
    num_channels=7,
    d_model=16,
    num_layers=4,
)

model_graph = draw_graph(
    model,
    input_size=(1, 512, 7),
    expand_nested=True,
    save_graph=True,
    filename='patchtsmixer_architecture',
    format='pdf'
)
```

## Performance Metrics

### Stage 1: Bring-Up (Correctness Validation)

Performance collected from [benchmark_datasets.py](benchmark_datasets.py) on ETTh2 test set with trained checkpoint.

**Configuration:**
- Context Length: 512 timesteps
- Prediction Length: 96 timesteps
- Model Dimension (d_model): 16
- Number of Layers: 4
- Patch Length: 8
- Batch Size: 1
- Dataset: ETTh2 (7 channels)
- Device: Wormhole (single device, no sharding)

#### Accuracy Metrics (TTNN vs PyTorch)

| Metric | PyTorch | TTNN | Difference | Target | Status |
|--------|---------|------|------------|--------|--------|
| MSE (vs ground truth) | 0.2579 | 0.2579 | +0.02% | <5% | ✅ PASS |
| MAE (vs ground truth) | 0.3550 | 0.3550 | +0.01% | <5% | ✅ PASS |
| RMSE (vs ground truth) | 0.5078 | 0.5079 | +0.01% | <5% | ✅ PASS |
| Correlation (vs ground truth) | 0.9009 | 0.9009 | -0.004% | >0.90 | ✅ PASS |
| TTNN-PyTorch Correlation | - | 0.9999 | - | >0.99 | ✅ PASS |

#### Performance Metrics (Stage 1 Baseline - Unoptimized)

| Metric | PyTorch (CPU) | TTNN (Wormhole) | Ratio | Target (Stage 2/3) |
|--------|---------------|-----------------|-------|-------------------|
| Throughput (samples/sec) | 821.75 | 1.39 | 591x slower | 200+ samples/sec |
| Latency per sample (ms) | 1.2 | 718 | 598x slower | <30ms |
| Total time (100 samples) | 0.12s | 71.83s | 598x slower | <0.5s |

**Notes:**
- ✅ **Stage 1 Complete:** All correctness criteria met (MSE/MAE <5%, correlation >0.90)
- ⚠️ **Performance optimization is Stage 2/3 work** - current numbers are expected baseline
- No memory sharding, operation fusion, or core parallelization applied yet
- Single device utilization (Wormhole device 0)

### Training Results

PyTorch reference model training on ETTh2:

| Metric | Value |
|--------|-------|
| Best Validation MSE | 0.3496 (Epoch 6) |
| Test MSE | 0.3270 |
| Training Time | ~2 minutes (10 epochs) |
| Checkpoint | `checkpoints/etth2_512_96/best_model.pt` |

*See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed results and [BOUNTY_PROGRESS.md](BOUNTY_PROGRESS.md) for Stage 2/3 roadmap.*

## References

- [PatchTSMixer Paper](https://arxiv.org/abs/2303.14304)
- [IBM Time Series Foundation Models](https://github.com/ibm/tsfm)
- [HuggingFace PatchTSMixer](https://huggingface.co/docs/transformers/model_doc/patchtsmixer)
