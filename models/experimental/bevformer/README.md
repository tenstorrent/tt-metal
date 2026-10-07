# BEVFormer

This directory holds BEVFormer-base end to end (`tt/tt_bevformer.py`): the image backbone (ResNet101-DCN) and FPN neck, the perception transformer around the BEVFormer encoder, the detection decoder, and the detection head with its NMS-free box coder.

BEVFormer Encoder is a transformer-based 3D object detection model that creates Bird's-Eye-View (BEV) representations from multi-camera images. The encoder uses spatiotemporal transformers to learn unified BEV representations by combining spatial cross-attention for feature extraction from camera views and temporal self-attention for modeling temporal dependencies.

## Model Architecture

The BEVFormer Encoder consists of several key components:

- **Spatial Cross-Attention**: Projects multi-camera features into BEV space using multi-scale deformable attention
- **Temporal Self-Attention**: Models temporal dependencies between consecutive frames using deformable attention
- **Multi-Scale Deformable Attention**: Core attention mechanism that samples features at multiple scales and locations
- **Point Sampling (3D to 2D)**: Projects 3D reference points to 2D camera coordinates for spatial attention; the encoder does it on the host in float32, once per frame
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

It runs 6 cameras at 1600x900, padded to 1600x928. Weights are prepared in the constructors (`tt/model_preprocessing.py` preprocesses them), so the forward runs on device only.

### Encoder

`tt/tt_encoder.py` ports BEVFormer-base's `BEVFormerEncoder`: six layers of temporal
self-attention (`tt/tt_temporal_self_attention.py`), spatial cross-attention into the cameras
(`tt/tt_spatial_cross_attention.py`) and an FFN, each followed by a LayerNorm, over one query per
cell of the 200x200 BEV grid. The reference (`reference/encoder.py`) follows upstream's modules
(`projects/mmdet3d_plugin/bevformer/modules/` in fundamentalvision/BEVFormer) and parameter names,
so the checkpoint's `pts_bbox_head.transformer.encoder` loads into it unchanged; it was checked
against upstream's modules, run on CPU with the checkpoint's weights.

- The temporal self-attention samples both the previous BEV, aligned to the current frame and
  shifted by the ego motion, and the current queries, from offsets that read both, and averages
  the two. On the first frame it samples the queries twice.
- The spatial cross-attention gathers each query into the cameras its pillar projects into and
  attends to the four FPN levels there with 8 points split over the pillar's 4 heights.
- `prepare_frame(img_metas)` projects the pillars into the cameras and builds the
  cross-attention's rebatch plan once per frame, on the host in float32 as upstream does: the
  geometry depends on the cameras only, and in bfloat16 the projection's homogeneous divide loses
  the points' precision. The forward then runs on device only and can be traced.
- The plan's device buffers have a fixed capacity per camera. `prepare_frame(img_metas, plan=plan)`
  refills them in place for the next frame, so a trace captured with the plan replays on every
  later frame. By default the first `prepare_frame` sizes the plan for its own frame only; a plan
  that later frames refill needs a `capacity` covering every rig it will see: a bound for the rig
  (with nuScenes' rig the busiest camera sees about a quarter of the base grid), or
  `full_capacity(bev_h * bev_w)` on smaller grids. `full_capacity` runs the deformable attention on
  every query per camera and does not fit in DRAM on the 200x200 base grid.
  `tests/perf/test_encoder_perf.py` captures the forward as a trace, which fails on any host read
  or write, and replays it on the next frame.
- As upstream, with batch size above 1 every sample gathers the queries the first sample's cameras
  see, and the temporal self-attention reads `value[:bs]`; both are exact at batch size 1, which
  BEVFormer runs.
- The sampling grids are float32, as in the decoder.
- The camera features are `(bs * num_cams, num_keys, 256)`: each sample's cameras in turn, the
  FPN's order, with the levels concatenated, as the perception transformer builds them.
- Parameters come from `tt/model_preprocessing.py` (`create_bevformer_encoder_parameters`) and are
  single use: building an encoder consumes the cross-attentions' sampling-offset weights.

### Perception Transformer

`tt/tt_perception_transformer.py` ports the BEV half of upstream's `PerceptionTransformer`,
`get_bev_features` (`reference/perception_transformer.py`): the glue around the encoder.

- The BEV queries get the CAN-bus MLP of the frame's ego motion added; each FPN level gets the
  camera and level embeddings, and the levels are concatenated into the encoder's camera features.
- The previous frame's BEV is rotated by the heading change about cell (100, 100), upstream's
  default for every grid size, as torchvision's nearest-neighbour `rotate` does, and the encoder
  shifts its reference points by the ego translation in cells of the point-cloud range over the
  grid, as upstream's head passes it.
- `prepare_frame(img_metas)` turns the frame's CAN bus, ego shift, rotation and cameras into device
  buffers, refilled in place for later frames, so the forward runs on device only and can be
  traced. The rotation moves whole cells: the host rotates an image of cell indices with the
  reference's own `rotate`, and the device gathers the previous BEV's rows by them, which picks
  exactly the cells the reference picks.
- `img_metas[b]["can_bus"]` is relative to the previous frame, as upstream's `forward_test` makes
  it (`reference/bevformer.py`'s `relative_can_bus`).
- The reference matches upstream's `get_bev_features` exactly with the BEVFormer-base checkpoint,
  over two frames, at batch sizes 1 and 2.

### Detector

`tt/tt_bevformer.py` runs the detector end to end: backbone and FPN, the perception transformer,
and the head, for one image shape and batch size (the backbone's convs are pinned to it).

- `prepare_frame(img_metas, frame)` prepares or refills the frame's buffers; the caller carries the
  previous BEV between frames, as upstream does.
- The BEV queries and their learned positional encoding depend on the weights only and are
  uploaded once (`tt/model_preprocessing.py`).
- `reference/bevformer.py`'s `load_bevformer_checkpoint` loads the BEVFormer-base checkpoint
  strictly, every key but the loss's `code_weights`.

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
- Parameters come from `tt/model_preprocessing.py` and are single use: building a decoder
  consumes them, so each decoder instance needs its own `create_decoder_parameters` call.
- Inputs and outputs are sequence-first by default, as in the reference; `batch_first=True` takes
  and returns batch-first tensors and skips the permutes.
- Besides the layer outputs and refined points, it returns each layer's box codes, the
  `reg_branches` raw output, which the head builds its box predictions from. The branches come
  from `create_reg_branch_parameters`, three Linears each.

### Detection Head and Box Coder

`tt/tt_head.py` ports the part of BEVFormer's `BEVFormerHead` on top of the encoder's
`(bs, bev_h * bev_w, 256)` BEV features: it runs the decoder batch-first over the object queries
and, per decoder layer, the classification branch.

- The object queries, their positional embeddings and the initial reference points depend on
  the weights only, so `tt/model_preprocessing.py` computes them once. The parameters also
  carry the head's BEV shape and `pc_range`, and are single use, like the decoder's.
- The box predictions are the decoder's box codes with cx, cy and cz replaced by its refined
  float32 points, scaled from [0, 1] to `pc_range` metres. The class logits are float32, the
  dtype the coder ranks.
- The encoder side of BEVFormer's `PerceptionTransformer` (BEV queries, positional encoding, can
  bus, previous BEV) is not part of the head; see the perception transformer and the detector.

`tt/tt_nms_free_coder.py` ports `NMSFreeCoder`: the top 300 (query, class) pairs of the last
layer by score, their box predictions decoded to `(cx, cy, cz, w, l, h, yaw, vx, vy)`, and a
range filter.

- The top-k and the decoding run on device. A float32 top-k over all 9000 scores is slow, so the
  logits are ranked in bfloat16 against a pivot, the bfloat16 k-th logit, and the 512 candidates
  are re-ranked exactly in float32.
- Only the final range filter runs on host: it keeps a data-dependent number of boxes and ends
  the pipeline.
- cz stays at the box's gravity center; upstream `get_bboxes` moves it to the bottom face after
  the coder.

## Project Structure

```
models/experimental/bevformer/
├── model_config.py     # BEVFormer-base's configuration and the port's precision and memory choices
├── reference/          # PyTorch reference implementation
├── tests/
│   ├── common.py       # Test helpers: camera rigs, dataset presets, dummy weights, random inputs
│   ├── pcc/            # Unit tests for individual components
│   └── perf/           # Traced device-perf harnesses (backbone and FPN, encoder)
└── tt/                 # TTNN implementation; model_preprocessing.py prepares every part's parameters
```

## Section 1: Test Files

The test suite validates the image backbone, the FPN, the individual components of the BEVFormer encoder, the detection decoder, and the detection head and box coder, ensuring correctness of both reference and TTNN implementations.

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

The backbone and FPN tests use seeded random weights (`tests/common.py`), tuned to the output statistics of the trained backbone, and assert PCC 0.99.

#### test_decoder.py
Tests the six-layer detection decoder.

**What it tests:**
- The tiny (50x50) and base (200x200) BEV grids, a non-square 50x100 grid and batch size 2
- PCC 0.99 on every layer's output and refined reference points, and per channel on the box
  codes; per-layer PCC of the refinement steps and the refined points' error in BEV pixels are
  logged
- Traced runs: capture proves the forward has no host reads or writes, and the replay runs on new
  inputs
- That a second eager run adds no programs to the program cache

It uses seeded random weights (`tests/common.py`): BEVFormer's sampling-offset grid init
with random weights on top, spread as in the BEVFormer-base checkpoint for the sampling offsets,
the cross- and self-attention logits and the reg branches' refinement rows. The BEV features are
random but spatially smooth, as the encoder's are, and the reference points include the grid
edges. bfloat16 error grows through the reference-point refinement, faster on the 200x200 grid.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_decoder.py
```

#### test_head.py
Tests the detection head, and the coder on the head's outputs.

**What it tests:**
- The tiny grid, a traced run on new BEV features, batch size 2, and the base 200x200 grid
- PCC 0.99 on the class logits and, channel by channel, on the box predictions, over all layers
  and on the last one; a joint PCC would be carried by the metre-scale centers alone
- That the coder, on the head's outputs, selects pairs within twice the score error of the reference
  head's top 300, with the reference's scores and boxes at those pairs

The dummy decoder and reg-branch weights are scaled to the BEVFormer-base checkpoint's
statistics (sampling offsets, attention logits, reference-point refinements, box channel spread);
`tests/common.py` lists them.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_head.py
```

#### test_nms_free_coder.py
Tests the box coder on head-shaped inputs.

**What it tests:**
- Distinct logits, so the top 300 labels and queries must match the reference exactly, spread
  evenly and in a background-dominated shape whose dense band defeats a bfloat16 top-k without
  the pivot
- The decoded boxes channel by channel, yaw through its sine and cosine, and the scores to a few
  float32 ulps
- The range filter's kept boxes, and that a second call only hits the program cache

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_nms_free_coder.py
```

#### test_encoder.py
Tests the six-layer encoder over two consecutive frames.

**What it tests:**
- The base (200x200) BEV grid with six layers and with one, the tiny (50x50) grid with batch sizes
  1 and 2 (per-sample shifts, the second sample's rig turned), and a non-square 50x100 grid
- Two frames: the first without a previous BEV, the second with each side's own first-frame output
  as its previous BEV and an ego shift, so the device's error is carried forward as in the
  detector; PCC 0.997 on both frames' outputs

It uses seeded random weights (`tests/common.py`): upstream's offset-grid init with random
weights on top, scaled to the offset spread and attention-logit spread the BEVFormer-base
checkpoint's encoder shows, except the self-attention offsets: at the checkpoint's spread, random
offset weights make the six layers amplify a bfloat16-sized input perturbation on their own,
which the trained encoder does not. The camera features are random but spatially smooth, as the FPN's
are, at the FPN's four level sizes for 928x1600 images; the camera geometry is nuScenes' rig
(`tests/common.py`).

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_encoder.py
```

#### test_perception_transformer.py
Tests the perception transformer over two consecutive frames: the first without a previous BEV, the
second with each side's own first-frame BEV, rotated and shifted by the ego motion. The base grid
at batch size 1 and the tiny grid at batch size 2, each sample with its own CAN bus, rotation, shift
and turned rig, which also runs a third frame and checks it compiles no new programs. PCC 0.999 on
every frame.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_perception_transformer.py
```

#### test_bevformer.py
Tests the detector end to end on random camera images over two consecutive frames: the BEV, the
last decoder layer's class logits and, channel by channel, its box predictions, at PCC 0.95.

It uses dummy weights by default (the backbone's, encoder's and head's tests' weights). With
`BEVFORMER_CHECKPOINT` set to the BEVFormer-base checkpoint
([bevformer_r101_dcn_24ep.pth](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth),
from upstream's model zoo) it loads the trained weights instead;
the inputs stay random. With the trained weights the box velocities and yaw are the least precise
channels, at PCC 0.97 to 0.99, from the backbone's bfloat8_b weights and bfloat16 accumulation
carried through the encoder and decoder; the other outputs stay above 0.99. The reference runs the
backbone on the CPU three times, so the test takes several minutes.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_bevformer.py
BEVFORMER_CHECKPOINT=/path/to/bevformer_r101_dcn_24ep.pth pytest models/experimental/bevformer/tests/pcc/test_bevformer.py
```

#### test_temporal_self_attention.py
Tests the temporal self-attention alone: the first frame (the queries stacked with themselves) and
a later one (a smooth previous BEV stacked with the queries, shifted reference points), on the
tiny, base and non-square grids and with batch size 2. PCC 0.999 on the output and on the attended
part alone, without the residual.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_temporal_self_attention.py
```

#### test_spatial_cross_attention.py
Tests the spatial cross-attention alone, with a rebatch plan from `build_rebatch_plan`, on the
tiny, base and non-square grids, CARLA's rig, batch size 2 with the second sample's rig turned, a
frame no camera sees (an empty plan) and a frame that fills the plan exactly. PCC 0.999 on the
output and on the attended part alone, except for the empty plan, whose attended part is the output
projection's bias alone. `test_rebatch_plan_update` refills a plan in place for
another rig.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_spatial_cross_attention.py
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

`model_config.py` holds BEVFormer-base's configuration in one place, shared by the reference and
the port:

- The architecture: six cameras at 1600x928, the ResNet101-DCN and FPN arguments and the FPN levels'
  sizes, the 200x200 BEV grid, `embed_dims` 256, the encoder's six layers (8 heads, the spatial
  cross-attention's 8 points over 4 pillar heights, the temporal self-attention's 4 points, FFN 512),
  the decoder's six layers and 900 queries, the CAN-bus and rotation constants. The reference modules
  take these as their defaults, and `reference/bevformer.py`'s `build_bevformer_base` builds the
  detector from them.
- `pc_range`, [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]: the nuScenes range the BEV grid, the encoder's
  pillars and the head's boxes share.
- The box code and the coder's box layout, ranges and top-k size.
- The port's precision and memory choices: the float32 sampling grids and class logits, and the
  backbone and FPN's per-layer DRAM, sharding and fp32-accumulation settings.

The dataset presets (camera rigs, image sizes, point-cloud ranges) that the deformable-attention,
point-sampling and encoder tests use are test fixtures, in `tests/common.py`.
