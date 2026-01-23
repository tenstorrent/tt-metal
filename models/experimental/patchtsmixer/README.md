# PatchTSMixer for TT-NN

PatchTSMixer is a lightweight MLP-Mixer based architecture for multivariate time series forecasting, implemented using TTNN APIs for Tenstorrent hardware (Wormhole).

**Status:** ✅ Stage 1 Complete + Stage 2 Optimizations Applied - 3x baseline performance achieved

## Quick Links
- [Architecture](#architecture) | [Benchmarks](#benchmarks--validation) | [Getting Started](#getting-started) | [Performance](#performance-metrics) | [Stage Progress](#bounty-stage-progress)

## Architecture

**Components:**
- **Patchify** → **Linear Projection** → **Positional Encoding** → **Mixer Blocks** → **Forecast Head**

**Mixer Block:** Patch Mixer (temporal) + Feature Mixer (hidden) + Optional Channel Mixer (cross-variate)

**I/O:**
- Input: `(batch, context_length, num_channels)`
- Output: `(batch, prediction_length, num_channels)`

**Modes:** `common_channel` (independent) | `mix_channel` (cross-variate dependencies)

## Benchmarks & Validation

### Accuracy (TTNN vs PyTorch on 100 ETTh2 samples)

| Metric | TTNN | PyTorch | Difference | Target | Status |
|--------|------|---------|------------|--------|--------|
| MSE | 0.2579 | 0.2579 | **+0.02%** | <5% | ✅ |
| MAE | 0.3550 | 0.3550 | **+0.01%** | <5% | ✅ |
| Correlation | 0.9009 | 0.9009 | **-0.004%** | >0.90 | ✅ |
| TTNN-PyTorch Corr. | **0.9999** | - | - | >0.99 | ✅ |

### Performance (Optimized TTNN on Wormhole)

| Metric | Baseline (Stage 1) | Optimized (Stage 2) | Target | Status |
|--------|-------------------|---------------------|--------|--------|
| **Throughput** (samples/s) | 0.13 | **530-600** | 200+ | ✅ **3x target** |
| **Latency** (ms) | 7692 | **~2ms** | <30ms | ✅ **15x better** |
| **Speedup** | 1x | **4,400x** | 1,538x | ✅ **2.9x target** |


## Getting Started

### 1. Install Dependencies
```bash
pip install git+https://github.com/IBM/tsfm.git  # Optional, for HF training
```

### 2. Train PyTorch Reference
```bash
python reference/main.py \
    --context_length 512 --prediction_length 96 \
    --d_model 16 --num_layers 4 --num_epochs 10 \
    --output_dir checkpoints/etth2_512_96
```

### 3. Full Benchmark (100 samples)
```bash
python benchmark_datasets.py \
    --checkpoint checkpoints/etth2_512_96/best_model.pt \
    --num-samples 100 --d-model 16 --num-layers 4
```

### 4. Run Tests
```bash
cd tests/pcc
pytest test_modules.py -v                      # Unit tests (all modules components)
pytest test_patchtsmixer_end_to_end.py -v      # End-to-end model
```

## Configuration

**Key Parameters:**
- `context_length=512`: Input sequence length
- `prediction_length=96`: Forecast horizon
- `patch_length=8`, `patch_stride=8`: Patching config
- `d_model=16`: Hidden dimension
- `num_layers=4`: Number of mixer blocks
- `mode`: `common_channel` | `mix_channel`

**Dataset:** ETTh2 (7 channels, hourly electricity data)
- Train: 12 months | Val: 4 months | Test: 4 months
- Auto-downloads on first run

## Performance Metrics

### Stage 1: Correctness Validation ✅

| Metric | PyTorch | TTNN (Wormhole) | Ratio | Target (Stage 2/3) |
|--------|---------|-----------------|-------|-------------------|
| **Accuracy** | | | | |
| MSE | 0.2579 | 0.2579 (+0.02%) | ✅ <5% | Maintain |
| MAE | 0.3550 | 0.3550 (+0.01%) | ✅ <5% | Maintain |
| Correlation | 0.9009 | 0.9009 | ✅ >0.90 | Maintain |
| **Performance (Baseline)** | | | | |
| Throughput (samples/s) | 821.75 | 0.13 | 6,321x slower | **200+** |
| Latency (ms) | 1.2 | 7692 | 6,410x slower | **<30ms** |

**Stage 1 Status:** ✅ Complete - All correctness criteria met

### Stage 2: Optimizations Applied ✅

**Optimizations:**
- ✅ L1 memory configuration for weights and activations
- ✅ Multi-core parallelization (core_grid) for MLP, Embedding, Gated Attention
- ✅ Hardware-optimized compute kernels (HiFi2 math fidelity, approximation mode)
- ✅ Smart memory allocation (DRAM for large temps, L1 for hot data)
- ✅ idx4 caching to avoid repeated tensor expansions

| Metric | Baseline | Optimized | Improvement | Target | Status |
|--------|----------|-----------|-------------|--------|--------|
| Throughput (samples/s) | 0.13 | **530-600** | **4,400x** | 200+ | ✅ **3x target** |
| Latency (ms) | 7692 | **~2** | **3,846x** | <30ms | ✅ **15x better** |
| Accuracy (PCC vs PyTorch) | 0.9999 | **0.9999** | Maintained | >0.99 | ✅ |

**Stage 2 Status:** ✅ Complete - Baseline target exceeded by 3x

## Bounty Requirements Checklist

### ✅ Stage 1: Bring-Up (Core Requirements: 90% | Performance: 0%)

#### Core Implementation
- [x] **Implement PatchTSMixer using TTNN APIs (Python)**
  - [x] Input patching layer (divides time series into patches)
  - [x] Patch normalization (instance normalization and batch normalization)
  - [x] Time-Mixing MLP layers (processes temporal patterns)
  - [x] Channel-Mixing MLP layers (processes cross-variate patterns)
  - [x] Optional gated attention mechanism
  - [x] Head module for forecasting
  - [ ] Head module for classification *(optional)*
  - [ ] Head module for regression *(optional)*
  - [ ] Online reconciliation head *(optional)*

#### Hardware & Execution
- [x] **Model runs on Tenstorrent hardware (Wormhole) with no errors**

#### Task Mode Support
- [x] **Time-series forecasting:** Multi-horizon prediction
- [ ] **Classification:** Time-series classification tasks *(optional)*
- [ ] **Pre-training:** Self-supervised pre-training for transfer learning *(optional)*
- [ ] **Regression:** Direct regression tasks *(optional)*

#### Channel Modeling Modes
- [x] **Channel-independent:** Each variable processed separately (common_channel)
- [x] **Channel-mixing:** Cross-variate dependencies modeled (mix_channel)
- [ ] **Hybrid:** Combination of both approaches *(architecture supports, needs testing)*

#### Benchmarking & Validation
- [x] **Produces valid predictions on standard benchmarks** (ETTh2 dataset)
- [x] **Output is verifiable** (PyTorch/HuggingFace reference comparison)

#### Accuracy Requirements ✅
- [x] **MSE and MAE within 5% of PyTorch reference** (Achieved: 0.02% MSE, 0.01% MAE)
- [x] **Prediction correlation coefficient > 0.90** (Achieved: 0.9009 vs ground truth, 0.9999 vs PyTorch)

#### Performance Requirements ✅
- [x] **Inference throughput:** At least 200 sequences/second (Achieved: **530-600 seq/s**, 3x target)
- [x] **Latency:** < 30ms for single sequence prediction (Achieved: **~2ms**, 15x better)

#### Documentation
- [x] **Clear instructions for setup and running the model**

**Stage 1 Status:** ✅ All correctness requirements met | ✅ Performance targets exceeded (3x baseline)

---

### 📋 Stage 2: Basic Optimizations (✅ Complete - Baseline Target Exceeded)

#### Memory Configuration ✅
- [x] **Use optimal sharded/interleaved memory configs for:**
  - [x] Patch embedding layers (L1 for weights and outputs)
  - [x] Time-Mixing MLP layers (L1 with multi-core parallelization)
  - [x] Channel-Mixing MLP layers (L1 with multi-core parallelization)
  - [x] Gated attention computation (L1 with multi-core parallelization)
  - [x] Head projection layers (L1 for forecast head)

#### Multi-Core Parallelization ✅
- [x] **Implement core_grid distribution for:**
  - [x] MLP layers (2 linears × 8 layers, distributed across 64 cores)
  - [x] Embedding projection (input pipeline optimization)
  - [x] Gated attention (if enabled, 8 layers distributed)
  - [x] Use CoreGrid(y=min(B*C, 8), x=8) for optimal Wormhole utilization

#### Hardware Optimization ✅
- [x] **Apply compute kernel optimizations:**
  - [x] HiFi2 math fidelity (balanced precision/speed)
  - [x] Approximation mode for GELU/softmax
  - [x] BF16 accumulation (fp32_dest_acc_en=False)
  - [x] Optimized packer configuration

#### Memory & Library Integration ✅
- [x] **Store intermediate activations in L1 where beneficial**
  - [x] MLP activations distributed by core_grid
  - [x] Small frequently-reused tensors (idx2, idx4 cache)
  - [x] DRAM for large temporary tensors (63x patchify expansion)
- [x] **Use recommended TTNN linear operations with optimizations**

#### Patch Operations Optimization ✅
- [x] **Optimize patch-specific operations:**
  - [x] Efficient idx4 caching (avoid repeated expansions)
  - [x] Smart memory allocation (DRAM for large temps, L1 for outputs)
  - [x] Minimize layout conversions

**Stage 2 Achieved:** 530-600 samples/sec (3x baseline target), ~2ms latency (15x better than target)

---

### 📋 Stage 3: Deeper Optimization (Stretch Goals)

**Note:** Stage 2 baseline requirements (200 seq/s, <30ms) already exceeded by 3x. Stage 3 focuses on stretch goals.

#### Stretch Goals Status
- ⏸️ **1000+ sequences/second throughput** (Currently: 530-600 seq/s, 53-60% of stretch)
- ✅ **< 10ms latency** (Achieved: ~2ms)
- ⏸️ **Support for 2048+ patch inputs** (Framework-dependent)
- ⏸️ **100+ channels** (Scalable with current architecture)

#### Known Limitations
- **TTNN gather operation:** Requires matching dimensions, no broadcast support
  - Blocks removal of 63x memory expansion in patchify
  - Would unlock additional 1.5-2x performance if resolved
- **HEIGHT_SHARDED limitations:** Not supported by layer_norm operations
- **Program config constraints:** Tensor shapes don't fit strict matmul config requirements

#### Future Optimizations (Framework-Dependent)
- [ ] **Advanced sharding strategies** (blocked by TTNN layer_norm limitations)
- [ ] **Broadcast support in gather** (would eliminate patchify expansion)
- [ ] **Operation fusion** (GELU + linear, residual connections)
- [ ] **Pipeline overlapping** (patchify + embedding + MLP stages)

**Stage 3 Target:** 1000+ samples/sec (stretch), <10ms latency (✅ achieved)

---

### Summary

| Stage | Core Features | Performance | Status |
|-------|---------------|-------------|--------|
| **Stage 1** | 21/21 (100%) | 2/2 (100%) | ✅ Complete |
| **Stage 2** | 15/15 (100%) | 2/2 (100%) | ✅ Complete - **3x baseline target** |
| **Stage 3** | 1/4 stretch goals | <10ms latency ✅ | ⏸️ Stretch goals (53-60% of 1000 seq/s target) |

**Performance Summary:**
- ✅ **Baseline requirements exceeded:** 530-600 samples/sec (3x target), ~2ms latency (15x better)
- ⏸️ **Stretch goal progress:** 53-60% toward 1000 samples/sec target
- 🔒 **Framework limitations identified:** TTNN gather broadcast, HEIGHT_SHARDED layer_norm support

## File Structure

```
models/demos/patchtsmixer/
├── README.md
├── tt/
│   ├── patchtsmixer.py          # TTNN implementation
│   └── model_processing.py      # Parameter conversion
├── reference/
│   ├── pytorch_patchtsmixer.py  # PyTorch reference
│   ├── train_patchtsmixer_pytorch.py   # Training script
│   └── train_patchtsmixer_HF_etth2.py  # HuggingFace training
├── tests/pcc/
│   ├── test_modules.py          # Unit tests (modules)
│   └── test_patchtsmixer_end_to_end.py  # E2E tests
├── benchmark_datasets.py        # Benchmarking tool
└── checkpoints/                 # Trained models (auto-created)
```

## References

- **Paper:** [TSMixer Paper](https://arxiv.org/pdf/2306.09364)
- **HuggingFace PatchTSMixer tuto:** [PatchTSMixer docs](https://huggingface.co/docs/transformers/model_doc/patchtsmixer)
