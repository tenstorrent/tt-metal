# BEVFormer Encoder Model

BEVFormer Encoder is a transformer-based 3D object detection model that creates Bird's-Eye-View (BEV) representations from multi-camera images. The encoder uses spatiotemporal transformers to learn unified BEV representations by combining spatial cross-attention for feature extraction from camera views and temporal self-attention for modeling temporal dependencies.

## Model Architecture

The BEVFormer Encoder consists of several key components:

- **Spatial Cross-Attention**: Projects multi-camera features into BEV space using multi-scale deformable attention
- **Temporal Self-Attention**: Models temporal dependencies between consecutive frames using deformable attention
- **Multi-Scale Deformable Attention**: Core attention mechanism that samples features at multiple scales and locations
- **Point Sampling (3D to 2D)**: Projects 3D reference points to 2D camera coordinates for spatial attention
- **BEVFormer Layer**: Single transformer layer combining spatial and temporal attention with feed-forward network
- **BEVFormer Encoder**: Multi-layer encoder processing BEV queries through transformer layers

The model processes:
- **Multi-camera Features**: Feature maps from multiple cameras at different scales
- **BEV Queries**: Initial bird's-eye view query features
- **Previous BEV Features**: Temporal context from previous timesteps
- **Camera Metadata**: Intrinsic/extrinsic parameters for 3D-2D projection

The model outputs:
- **BEV Features**: Unified bird's-eye view representations combining spatial and temporal information

## Project Structure

```
models/experimental/bevformer/
├── config/             # Configuration files and model parameters
│   └── encoder_config/ # Encoder-specific configurations
├── reference/          # PyTorch reference implementation
├── tests/              # All tests together
│   └── pcc/            # Unit tests for individual components
└── tt/                 # TTNN optimized implementation
```

## Section 1: Test Files

The test suite validates individual components of the BEVFormer encoder, ensuring correctness of both reference and TTNN implementations.

### PCC (Pearson Correlation Coefficient) Tests

Located in `models/experimental/bevformer/tests/pcc/`, these tests validate the accuracy of TTNN implementations against PyTorch reference models using PCC metrics.

#### test_encoder.py
Tests the complete BEVFormer encoder implementation.

**What it tests:**
- Full BEVFormer encoder forward pass correctness against PyTorch reference
- Multi-layer transformer processing with spatial and temporal attention
- BEV query processing through transformer layers
- Integration of all encoder components

**Key test cases:**
- Single and multi-layer encoder configurations
- Different BEV grid sizes and feature dimensions

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_encoder.py
```

#### test_spatial_cross_attention.py
Tests the spatial cross-attention mechanism for projecting camera features to BEV space.

**What it tests:**
- Spatial cross-attention forward pass correctness
- Multi-scale deformable attention for spatial feature extraction
- 3D-2D point projection and sampling
- Camera mask handling and feature aggregation

**Key features:**
- Validates camera coordinate transformations
- Tests different numbers of cameras and feature levels
- Precision and memory layout compatibility

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_spatial_cross_attention.py
```

#### test_temporal_self_attention.py
Tests the temporal self-attention mechanism for modeling frame-to-frame dependencies.

**What it tests:**
- Temporal self-attention forward pass correctness
- Deformable attention for temporal feature aggregation
- Previous BEV feature integration
- Temporal shift handling for camera motion

**Key test parameters:**
- Different sequence lengths and temporal configurations
- Various BEV grid resolutions
- Memory length and temporal context settings

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_temporal_self_attention.py
```

#### test_ms_deformable_attention.py
Tests the core multi-scale deformable attention mechanism.

**What it tests:**
- Multi-scale deformable attention forward pass
- Attention weight computation and normalization
- Sampling point generation and feature aggregation
- Different scales and sampling point configurations

**Key features:**
- Tests with various input resolutions and scales
- Validates offset and mask generation
- Tests attention weight distributions
- Memory layout and precision handling

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_ms_deformable_attention.py
```

#### test_point_sampling_3d_2d.py
Tests the 3D to 2D point projection functionality.

**What it tests:**
- 3D reference point generation for BEV grid
- Point projection from 3D space to camera coordinates
- Camera intrinsic/extrinsic matrix handling
- Visibility mask computation for projected points

**Key test cases:**
- Different BEV grid sizes and point cloud ranges
- Various camera configurations and projection matrices
- Edge cases for points outside camera field of view

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_point_sampling_3d_2d.py
```

## Running All Tests
To run the complete test suite:

**Usage**
```bash
# Run all PCC tests
pytest models/experimental/bevformer/tests/pcc/
```

## Expected Outputs

All tests generate:
- **Console logs**: Detailed PCC comparisons and validation results
- **Tensor comparisons**: Element-wise accuracy analysis


## Configuration

The model supports flexible configuration through dataclass-based config objects:

- **AttentionConfig**: Base configuration for attention modules
- **DeformableAttentionConfig**: Multi-scale deformable attention parameters
- **SpatialCrossAttentionConfig**: Spatial attention with camera setup
- **TemporalSelfAttentionConfig**: Temporal attention with memory configuration

Default configurations are provided for common datasets like nuScenes with typical parameters:
- embed_dims: 256
- num_heads: 8
- num_levels: 4
- num_points: 4
- num_cams: 6
- pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] (nuScenes default)
