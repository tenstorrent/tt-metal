# PatchTSMixer Setup and Usage Guide

## 📦 Installation

### Prerequisites
- TT-Metal environment already set up and built
- Python 3.10+
- Wormhole or Blackhole hardware

### Step 1: Install Core Dependencies

```bash
cd /root/workspace/tt-metal/models/demos/patchtsmixer
pip install -r requirements.txt
```

This installs:
- `torch>=2.0.0` - PyTorch for reference implementation
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data handling
- `transformers>=4.30.0` - Optional, for HuggingFace training scripts

### Step 2: Install Time Series Foundation Models (Optional)

For using the HuggingFace-style training scripts (`reference/main.py`, `reference/train_patchtsmixer_etth2.py`):

```bash
pip install git+https://github.com/IBM/tsfm.git
```

**Note:** This is optional if you use `benchmark_datasets.py` which has built-in dataset handling.

### Step 3: Verify Installation

```bash
python -c "import torch; import numpy; import pandas; import ttnn; print('All packages installed successfully')"
```

---

## 🚀 Running the Model

### Option 1: Quick Validation (10 samples)

Test TTNN implementation against PyTorch reference with a small sample:

```bash
cd /root/workspace/tt-metal/models/demos/patchtsmixer
python quick_validation.py
```

**Output:**
- PCC (Pearson Correlation Coefficient)
- MSE difference percentage
- MAE difference percentage
- Correlation coefficient

**Expected:** PCC > 0.99, metrics within 5%, correlation > 0.90

### Option 2: Full Dataset Benchmarking

Compare TTNN vs PyTorch on complete ETTh2 test set:

```bash
python benchmark_datasets.py --model_path <path_to_checkpoint.pt>
```

**Arguments:**
- `--model_path`: Path to trained PyTorch checkpoint (optional, will use random weights if not provided)
- `--dataset`: Dataset to use (default: "etth2")
- `--context_length`: Input sequence length (default: 512)
- `--prediction_length`: Forecast horizon (default: 96)
- `--batch_size`: Batch size (default: 1)
- `--num_samples`: Number of test samples to evaluate (default: 100)

**Example with trained model:**
```bash
python benchmark_datasets.py \
    --model_path checkpoints/etth2_512_96/best_model.pt \
    --dataset etth2 \
    --num_samples 100
```

**Output:**
- Per-sample and aggregate metrics (MSE, MAE, RMSE)
- Correlation coefficient
- Throughput (sequences/second)
- Average latency
- Pass/fail validation (within 5%, correlation > 0.90)

### Option 3: Unit Tests

Test individual components:

```bash
cd tests/pcc

# Test all modules (gated attention, normalization, MLP, etc.)
pytest test_modules.py -v

# Test end-to-end model
pytest test_patchtsmixer_end_to_end.py -v

# Run specific test
pytest test_modules.py::test_gated_attention -v
```

---

## 🎓 Training PyTorch Reference Model

You need a trained PyTorch model to validate the TTNN implementation against. Two options:

### Option A: Using Built-in Dataset Handler (Recommended)

Uses `benchmark_datasets.py` which has built-in ETT dataset support:

```bash
# This feature needs to be added to benchmark_datasets.py
# Currently, benchmark_datasets.py only does evaluation
```

**Status:** Training mode not yet implemented in `benchmark_datasets.py`. Use Option B.

### Option B: Using IBM TSFM Toolkit

Requires `tsfm_public` package (install via Step 2 above):

```bash
cd reference

# Train vanilla PyTorch implementation
python main.py \
    --context_length 512 \
    --prediction_length 96 \
    --patch_length 8 \
    --patch_stride 8 \
    --d_model 16 \
    --num_layers 4 \
    --batch_size 64 \
    --num_epochs 10 \
    --lr 1e-3 \
    --mode common_channel \
    --output_dir ../checkpoints/etth2_512_96
```

**Or use HuggingFace-style training:**
```bash
python train_patchtsmixer_etth2.py
```

**Training Output:**
- Model saved to: `checkpoints/etth2_512_96/best_model.pt`
- Final model: `checkpoints/etth2_512_96/final_model.pt`
- Preprocessor config: `checkpoints/etth2_512_96/preprocessor/`

**Training Time:** ~10-20 minutes on CPU, ~5 minutes on GPU

---

## 📊 Benchmarking Workflow

### Complete Validation Pipeline

1. **Train PyTorch reference model:**
```bash
cd /root/workspace/tt-metal/models/demos/patchtsmixer
python reference/main.py \
    --context_length 512 \
    --prediction_length 96 \
    --patch_length 8 \
    --patch_stride 8 \
    --d_model 16 \
    --num_layers 4 \
    --batch_size 64 \
    --num_epochs 10 \
    --lr 1e-3 \
    --mode common_channel \
    --output_dir checkpoints/etth2_512_96
```

2. **Run full benchmark validation:**
```bash
python benchmark_datasets.py \
    --model_path checkpoints/etth2_512_96/best_model.pt \
    --num_samples 100
```

3. **Check results:**
   - ✅ MSE/MAE within 5% of PyTorch
   - ✅ Correlation > 0.90
   - ✅ No runtime errors on TT hardware
   - 📊 Record throughput and latency (for Stage 2/3 baseline)

---

## 🔧 Configuration Options

### Model Hyperparameters

| Parameter | Description | Default | Tested Values |
|-----------|-------------|---------|---------------|
| `context_length` | Input sequence length | 512 | 512 |
| `prediction_length` | Forecast horizon | 96 | 96, 192, 336, 720 |
| `patch_length` | Length of each patch | 8 | 8, 16 |
| `patch_stride` | Stride for patch extraction | 8 | 8 (non-overlapping) |
| `d_model` | Model hidden dimension | 16 | 8, 16, 32, 64 |
| `num_layers` | Number of PatchTSMixer blocks | 4 | 4, 8 |
| `mode` | Channel modeling mode | "common_channel" | "common_channel", "mix_channel" |
| `use_gated_attn` | Enable gated attention | False | True, False |
| `dropout` | Dropout rate | 0.1 | 0.0, 0.1, 0.2 |
| `head_dropout` | Head dropout rate | 0.1 | 0.0, 0.1, 0.2 |

### Datasets Supported

| Dataset | Channels | Frequency | Train Size | Test Size | Features |
|---------|----------|-----------|------------|-----------|----------|
| **ETTh1** | 7 | Hourly | 8640 | 2880 | Electricity transformer temperature |
| **ETTh2** | 7 | Hourly | 8640 | 2880 | Electricity transformer temperature |
| **ETTm1** | 7 | 15-min | 34560 | 11520 | Electricity transformer temperature |
| **ETTm2** | 7 | 15-min | 34560 | 11520 | Electricity transformer temperature |
| **Weather** | 21 | 10-min | 36792 | 5271 | Meteorological indicators |
| **Traffic** | 862 | Hourly | 12185 | 3509 | Road occupancy rates |

**Currently Tested:** ETTh2

---

## 📈 Performance Expectations

### Stage 1 (Current Status)

**Accuracy (TTNN vs PyTorch):**
- ✅ PCC: > 0.99
- ✅ MSE difference: < 5%
- ✅ MAE difference: < 5%
- ✅ Correlation: > 0.90

**Performance (Baseline, not optimized):**
- Throughput: ~0.66 sequences/second
- Latency: ~1500ms per sequence

**Hardware:**
- Wormhole devices 0, 1
- No sharding/optimization yet

### Stage 2 Target (After Basic Optimizations)

**Performance Goals:**
- Throughput: 50-100 sequences/second
- Latency: 50-100ms per sequence

**Optimizations:**
- Memory sharding
- Operation fusion
- L1 cache utilization

### Stage 3 Target (After Deep Optimizations)

**Performance Goals:**
- Throughput: 200+ sequences/second (required), 1000+ (stretch)
- Latency: < 30ms (required), < 10ms (stretch)

**Optimizations:**
- Multi-core parallelization
- Pipeline stages
- Advanced fusion

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'tsfm_public'`

**Solution 1 (Recommended):** Install from GitHub
```bash
pip install git+https://github.com/IBM/tsfm.git
```

**Solution 2:** Use `benchmark_datasets.py` instead of `reference/main.py`
- Built-in dataset handling without external dependencies
- No training support yet (evaluation only)

### Issue: `Device not found` or `Metal device initialization failed`

**Check:**
1. TT-Metal built correctly: `./build_metal.sh --build-all`
2. Devices detected: `tt-smi` or check device status
3. Firmware version compatible: Currently using 19.3.0

### Issue: Low PCC or accuracy mismatch

**Debug steps:**
1. Check model configuration matches PyTorch reference
2. Verify checkpoint loaded correctly
3. Test individual components: `pytest tests/pcc/test_modules.py -v`
4. Check data preprocessing (normalization)
5. Verify parameter conversion in `tt/model_processing.py`

### Issue: Out of memory on device

**Current constraints:**
- Batch size: 1 (tested)
- Sequence length: 512 (tested)
- No sharding/optimization yet (Stage 1)

**For Stage 2/3:**
- Implement memory sharding
- Optimize tensor layouts
- Use L1 cache

---

## 📁 File Structure

```
models/demos/patchtsmixer/
├── README.md                          # Overview
├── requirements.txt                   # Python dependencies
├── SETUP_AND_USAGE.md                 # This file
├── BOUNTY_PROGRESS.md                 # Stage tracking
├── STAGE1_STATUS.md                   # Stage 1 details
├── BENCHMARKING_GUIDE.md              # Benchmarking instructions
├── DATASET_BENCHMARK_SUMMARY.md       # Benchmark results
│
├── tt/                                # TTNN implementation
│   ├── patchtsmixer.py                # Main TTNN model (724 lines)
│   └── model_processing.py            # PyTorch → TTNN conversion (198 lines)
│
├── reference/                         # PyTorch reference
│   ├── pytorch_patchtsmixer.py        # Custom PyTorch implementation (512 lines)
│   ├── main.py                        # Training script (vanilla)
│   ├── train_patchtsmixer_etth2.py    # Training script (HuggingFace)
│   └── patchtsmixer_huggingface_eval.py  # HF evaluation
│
├── tests/pcc/                         # Correctness tests
│   ├── test_modules.py                # Unit tests (841 lines)
│   └── test_patchtsmixer_end_to_end.py  # End-to-end test (352 lines)
│
├── benchmark_datasets.py              # Dataset benchmarking (465 lines)
├── quick_validation.py                # Quick 10-sample test
├── test.py                            # Simple test script
│
├── checkpoints/                       # Trained models (created during training)
│   └── etth2_512_96/
│       ├── best_model.pt
│       ├── final_model.pt
│       └── preprocessor/
│
└── ETT-small/                         # Datasets (auto-downloaded)
    ├── ETTh1.csv
    ├── ETTh2.csv
    ├── ETTm1.csv
    └── ETTm2.csv
```

---

## 🎯 Quick Start Commands

```bash
# 1. Install dependencies
cd /root/workspace/tt-metal/models/demos/patchtsmixer
pip install -r requirements.txt
pip install git+https://github.com/IBM/tsfm.git

# 2. Quick validation (10 samples, random weights)
python quick_validation.py

# 3. Train PyTorch model
python reference/main.py \
    --context_length 512 \
    --prediction_length 96 \
    --batch_size 64 \
    --num_epochs 10 \
    --output_dir checkpoints/etth2_512_96

# 4. Full benchmark with trained model
python benchmark_datasets.py \
    --model_path checkpoints/etth2_512_96/best_model.pt \
    --num_samples 100

# 5. Run unit tests
cd tests/pcc
pytest test_modules.py -v
pytest test_patchtsmixer_end_to_end.py -v
```

---

## 📞 Support

For issues or questions:
1. Check [BOUNTY_PROGRESS.md](BOUNTY_PROGRESS.md) for current status
2. Review test outputs in `tests/pcc/`
3. Check TT-Metal documentation
4. Refer to HuggingFace PatchTSMixer docs: https://huggingface.co/docs/transformers/en/model_doc/patchtsmixer

---

## 📝 Notes

- **Stage 1 Focus:** Correctness and functionality, not performance
- **Performance optimization:** Deferred to Stage 2 and 3
- **Dataset download:** Automatic on first run of `benchmark_datasets.py`
- **Checkpoint format:** PyTorch state_dict, compatible with both PyTorch and TTNN
- **Hardware tested:** Wormhole devices 0 and 1, firmware 19.3.0
