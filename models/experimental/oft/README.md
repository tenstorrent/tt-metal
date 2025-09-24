# OFT (Orthographic Feature Transform) Model

OFT is 3d object detection model that uses orthographic feature transforms to detect objects in 3D space. The model combines a ResNet-based frontend with specialized orthographic feature transformation layers and a topdown refinement network.

## Model Architecture

The OFT model consists of several key components:

- **Frontend**: ResNet-18/34 backbone for feature extraction at multiple scales (8x, 16x, 32x downsampling)
- **Lateral Layers**: Convert ResNet outputs to a common 256-channel feature representation
- **OFT Layers**: Orthographic Feature Transform modules that project features into bird's-eye view
- **Topdown Network**: 8-layer refinement network using BasicBlock modules
- **Detection Head**: Final convolutional layer that outputs object scores, positions, dimensions, and angles
- **Decoder** Additional module that is used to decode encoded outputs into objects

The model outputs:
- **Scores**: Object detection confidence scores
- **Position Offsets**: 3D position predictions (x, y, z)
- **Dimension Offsets**: Object size predictions (width, height, length)
- **Angle Offsets**: Object orientation predictions (sin, cos components)
- **Objects**: Decoded outputs into list of detected objects.

## Project Structure

```
models/experimental/oft/
├── demo/              # Demo scripts and visualization
├── reference/         # PyTorch reference implementation
├── resources/         # Test images and calibration files
├── tests/            # Unit tests for individual components
└── tt/               # TenstorrentNN (TTNN) optimized implementation
```

## Section 1: Demo Scripts

**Input Requirements:**
Both demos require:
- env variable CHECKPOINTS_PATH with pre-trained checkpoint file (e.g., `export CHECKPOINTS_PATH="/home/mbezulj/checkpoint-0600.pth"`)
- Input images in JPG format (located in `resources/`)
- Corresponding calibration files in TXT format (camera intrinsic parameters)

### demo.py
Full end-to-end inference demo that runs both PyTorch reference and TTNN implementations, comparing their outputs and generating visualizations.

**Features:**
- Loads pre-trained model weights from checkpoint
- Processes input images with calibration data
- Runs full OFT inference pipeline on both CPU (PyTorch) and device (TTNN)
- **Executes complete pipeline on TTNN**: OFTNet model inference + object decoder/encoder
- Compares intermediate outputs and final predictions
- Generates detection visualizations and heatmaps
- Supports various precision modes (float32, bfloat16)
- Configurable fallback modes for debugging

**Usage:**
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/oft/demo/demo.py
```

### host_demo.py
Host-only demo that compares float32 and bfloat16 precision using only PyTorch reference implementation.

**Features:**
- Precision comparison between fp32 and bfp16
- Object detection visualization
- Performance and accuracy analysis
- No device execution required - pure CPU inference
- Useful for baseline validation

**Usage:**
```bash
pytest models/experimental/oft/demo/host_demo.py
```

## Section 2: Test Files

The test suite validates individual components of the OFT model, ensuring correctness of both reference and TTNN implementations.

### test_basicblock.py
Tests the fundamental building block of the ResNet backbone and topdown network.

**What it tests:**
- TTBasicBlock forward pass correctness against PyTorch reference
- Memory layout conversions (NCHW ↔ NHWC)
- Sharding configurations for device execution
- Sequential execution of multiple BasicBlocks (topdown layers)

**Key test cases:**
- Single BasicBlock with various input dimensions
- 8 sequential BasicBlocks (mimicking topdown network)
- Different sharding strategies ("HS" - Height Sharding)

**Usage:**
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/oft/tests/test_basicblock.py
```

### test_encoder.py
Tests the object detection decoder/encoder that converts model outputs to final object detections.

**What it tests:**
- Peak detection in score heatmaps
- Non-maximum suppression (NMS)
- Object position, dimension, and angle decoding
- Score smoothing and filtering operations
- Object creation from decoded parameters

**Key features:**
- Loads pre-computed OFT outputs for consistent testing
- Validates intermediate processing steps
- Generates debug visualizations for manual inspection
- Tests both host and device implementations

**Usage:**
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/oft/tests/test_encoder.py
```

### test_oft.py
Tests the core Orthographic Feature Transform modules at different scales.

**What it tests:**
- OFT forward pass at 8x, 16x, and 32x scales
- Integral image computation
- Bounding box corner calculations
- Grid-based feature sampling
- Precision modes (float32, bfloat16)
- Pre-computed vs. on-demand grid calculation

**Key test parameters:**
- Different input resolutions corresponding to feature scales
- Various precision and grid computation modes
- Expected PCC (Pearson Correlation Coefficient) thresholds for each scale

**Usage:**
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/oft/tests/test_oft.py
```

### test_oftnet.py
Tests the OFTNet model (without decoder).

**What it tests:**
- Full model inference pipeline
- Integration of all components (ResNet + OFT + Topdown + Head)
- Host fallback mechanisms for debugging
- Multiple precision modes
- Real image processing with pre-trained weights

**Key features:**
- Uses real checkpoint weights for realistic testing
- Tests with actual images from the resources directory
- Configurable fallback modes (feedforward, lateral, OFT)
- Comprehensive intermediate output validation
- Output serialization for debugging encoder issues

**Usage:**
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/oft/tests/test_oftnet.py
```

### test_resnet.py
Tests the ResNet backbone feature extractor.

**What it tests:**
- ResNet-18 frontend implementation
- Multi-scale feature extraction (feats8, feats16, feats32)
- Memory layout handling for TTNN compatibility
- All intermediate activations in the ResNet pipeline

**Key features:**
- Tests with real images
- Validates all ResNet layers and operations
- Memory layout conversions for device execution
- Feature extraction at multiple downsampling rates

**Usage:**
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/oft/tests/test_resnet.py
```

## Running All Tests

To run the complete test suite:

```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/oft/tests
```

## Environment Setup

The tests require:
- Pre-trained model checkpoint
- Test images and calibration files

## Expected Outputs

All tests generate:
- **Console logs**: Detailed PCC comparisons and validation results
- **Visualizations**: Debug plots and comparison images (saved to `outputs/` directories)
