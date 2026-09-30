# BEVFormer

This directory holds the BEVFormer-base image backbone (ResNet101-DCN) and FPN neck, the BEVFormer encoder, and the detection decoder.

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

### Image Backbone and FPN

The multi-camera features the encoder reads come from BEVFormer-base's image backbone and neck:

- **ResNet101-DCN** (`tt/tt_resnet.py`): caffe-style ResNet101 with DCNv2 (modulated deformable convolution, `tt/tt_modulated_deform_conv.py`) in layer3 and layer4. It emits C3, C4 and C5 at strides 8, 16 and 32.
- **FPN** (`tt/tt_fpn.py`): maps C3-C5 to four 256-channel levels, the fourth from an extra stride-2 conv on the last output.

It runs 6 cameras at 1600x900, padded to 1600x928. Weights are prepared in the constructors (`tt/model_preprocessing_backbone.py` preprocesses them), so the forward runs on device only.

### Detection Decoder

`tt/tt_decoder.py` ports BEVFormer's `DetectionTransformerDecoder` (shared by tiny and base):
six DETR layers of self-attention, single-level deformable cross-attention over the BEV map
(`TTMSDeformableAttention`) and an FFN, with 900 object queries. After each layer the
`reg_branches` output refines the (x, y, z) reference points.

- The reference points and the sampling grid built from them are float32. In bfloat16 a
  point moves in steps of up to 0.8 px on the 200x200 base BEV grid, and the error compounds
  through the refinement.
- The BEV size is folded into the sampling-offset Linear when the module is built, so the
  decoder is built for one `(bev_h, bev_w)` and its forward runs on device only.
- Parameters come from `tt/model_preprocessing_decoder.py`.

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

The test suite validates the image backbone, the FPN, the individual components of the BEVFormer encoder and the detection decoder, ensuring correctness of both reference and TTNN implementations.

### PCC (Pearson Correlation Coefficient) Tests

Located in `models/experimental/bevformer/tests/pcc/`, these tests validate the accuracy of TTNN implementations against PyTorch reference models using PCC metrics.

#### test_resnet.py
Tests the ResNet101-DCN backbone block by block.

**What it tests:**
- The first bottleneck of layer1, layer3 (DCN) and layer4 (DCN, two 256-channel sampling chunks)
- The whole of layer1 and layer2
- The dtype each block emits

The whole backbone is checked in `test_backbone_fpn.py`.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_resnet.py
```

#### test_fpn.py
Tests the FPN neck on its own, fed random C3-C5 in the dtypes the backbone emits.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_fpn.py
```

#### test_backbone_fpn.py
Runs camera images through the backbone and the FPN, end to end.

**What it tests:**
- The backbone's C3-C5 against the reference
- The FPN's four outputs against the reference
- The hand-off between the two (layout, dtype, memory)

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_backbone_fpn.py
```

The backbone and FPN tests use seeded random weights (`tests/backbone_weights.py`), tuned to the output statistics of the trained backbone, and assert PCC 0.99.

#### test_decoder.py
Tests the six-layer detection decoder, layer by layer.

**What it tests:**
- The tiny (50x50) and base (200x200) BEV grids, a non-square 50x100 grid and batch size 2
- PCC 0.99 on the stacked six-layer outputs and refined reference points; per-layer PCC and the
  refined points' error in BEV pixels are logged
- Traced runs: capture proves the forward has no host reads or writes, and the replay runs on new
  inputs
- That a second eager run adds no programs to the program cache

It uses seeded random weights (`tests/decoder_common.py`): BEVFormer's sampling-offset grid init
with random weights on top, so offsets spread over a few pixels, and peaked cross- and
self-attention. The BEV features are random but spatially smooth, as the encoder's are, and the
reference points include the grid edges. bfloat16 error grows through the reference-point
refinement, faster on the 200x200 grid.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_decoder.py
```

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
