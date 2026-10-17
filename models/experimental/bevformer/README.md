# BEVFormer

This directory holds BEVFormer-base end to end (`tt/tt_bevformer.py`): the image backbone (ResNet101-DCN) and FPN neck, the perception transformer around the BEVFormer encoder, the detection decoder, and the detection head with its NMS-free box coder.

BEVFormer detects 3D objects from multi-camera images through a Bird's-Eye-View (BEV) representation. Per frame:

1. **Backbone and FPN**: the six camera images go through ResNet101-DCN and the FPN, which emit four 256-channel feature levels per camera.
2. **Perception transformer and encoder**: BEV queries, one per cell of a 200x200 grid, get the frame's ego motion (CAN bus) added, then six encoder layers refine them. Each layer's temporal self-attention samples the previous frame's BEV, aligned to the ego motion, and the queries themselves; its spatial cross-attention projects each cell's pillar into the cameras and samples their features there. The output is the frame's BEV, which the next frame reads as its previous BEV.
3. **Decoder and head**: 900 object queries go through six decoder layers of self-attention and deformable cross-attention into the BEV, each layer refining the queries' 3D reference points. Per layer, a classification branch emits class logits and a regression branch the box codes.
4. **Box coder**: the last layer's top 300 (query, class) pairs by score become 3D boxes `(cx, cy, cz, w, l, h, yaw, vx, vy)`, without NMS.

Every stage runs on device; per frame the host only prepares the camera geometry and the ego motion, into buffers a trace replays, and drops the boxes the coder marks invalid.

## Model Architecture

### Image Backbone and FPN

The multi-camera features the encoder reads come from BEVFormer-base's image backbone and neck:

- **ResNet101-DCN** (`tt/tt_resnet.py`): caffe-style ResNet101 with DCNv2 (modulated deformable convolution, `tt/tt_modulated_deform_conv.py`) in layer3 and layer4. It emits C3, C4 and C5 at strides 8, 16 and 32.
- **FPN** (`tt/tt_fpn.py`): maps C3-C5 to four 256-channel levels, the fourth from an extra stride-2 conv on the last output.

It runs 6 cameras at 1600x900, padded to 1600x928. `tt/model_preprocessing.py` prepares the weights: every BatchNorm folded into its conv, the 1x1 convs whose activations are in DRAM as Linear weights, and each DCN's matmul weight chunks and an offset conv that emits the offsets in the device DCN's (x, y) order and grid units. conv2d lays out its own weights on its first call, in the constructors, so the forward runs on device only.

### Encoder

`tt/tt_encoder.py` ports BEVFormer-base's `BEVFormerEncoder`: six layers of temporal
self-attention (`tt/tt_temporal_self_attention.py`), spatial cross-attention into the cameras
(`tt/tt_spatial_cross_attention.py`) and an FFN, each followed by a LayerNorm, over one query per
cell of the 200x200 BEV grid. The reference (`reference/encoder.py`) follows upstream's modules
(`projects/mmdet3d_plugin/bevformer/modules/` in fundamentalvision/BEVFormer) and parameter names,
so the checkpoint's `pts_bbox_head.transformer.encoder` loads into it unchanged.

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
- The temporal self-attention, the spatial cross-attention and the decoder's cross-attention all
  run on the fused MSDA op
  (`ttnn.experimental.fused_msda_from_offsets`), through `TTMSDeformableAttention`
  (`tt/tt_ms_deformable_attention.py`), with float32 reference points: in bfloat16 a point moves
  in steps of up to 0.8 px on the 200x200 grid. The self-attention calls the op directly, as its
  Linears read both stacked maps.
- The camera features are `(bs * num_cams, num_keys, 256)`: each sample's cameras in turn, the
  FPN's order, with the levels concatenated, as the perception transformer builds them.
- Parameters come from `tt/model_preprocessing.py` (`create_bevformer_encoder_parameters`).

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

### Detector

`tt/tt_bevformer.py` runs the detector end to end: backbone and FPN, the perception transformer,
and the head, for one image shape and batch size (the backbone's convs are pinned to it). Its
output ends, as upstream's `simple_test` does, in 3D boxes from the last decoder layer.

- `detect(cls_scores, bbox_preds)` runs upstream's `BEVFormerHead.get_bboxes` on device, in the
  forward's trace: the NMS-free coder's top-k and box decoding, the post-center range test as a
  validity mask, and each box's z moved to its bottom face, on fixed `(bs, 300)` tensors.
  `get_bboxes(scores, labels, boxes, valid)` only drops the invalid boxes on the host: the count
  that remains depends on the data, which no fixed-shape device tensor can hold. For BEVFormer-base
  it drops none, as the head scales the box centers into `pc_range`, inside the post-center range.

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

- The reference points stay float32 through the refinement and into the fused MSDA op, which
  forms the sampling location from them in fp32. In bfloat16 a point moves in steps of up to
  0.8 px on the 200x200 base BEV grid, and the error compounds through the refinement. The
  forward runs on device only.
- Parameters come from `tt/model_preprocessing.py` (`create_decoder_parameters`).
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
  carry the head's BEV shape and its `pc_range` as a corner and a size.
- The box predictions are the decoder's box codes with cx, cy and cz replaced by its refined
  float32 points, scaled from [0, 1] to `pc_range` metres. The class logits are float32, as the
  coder's exact re-rank needs.
- The encoder side of BEVFormer's `PerceptionTransformer` (BEV queries, positional encoding, can
  bus, previous BEV) is not part of the head; see the perception transformer and the detector.

`tt/tt_nms_free_coder.py` ports `NMSFreeCoder`: the top 300 (query, class) pairs of the last
layer by score, their box predictions decoded to `(cx, cy, cz, w, l, h, yaw, vx, vy)`, and the
range and score filters.

- The top-k and the decoding run on device, and the top-k is exact. A float32 top-k over all
  9000 scores is slow, so the exact k-th logit is found first: a bfloat16 top-k gives it rounded,
  and a bisection on counts (`CODER_BISECTION_STEPS`) narrows that to the float32 value. A
  bfloat16 top-k of the logits minus it then selects exactly the top 300, and a float32 top-k
  sorts them.
- The range test and the optional score threshold run on device too, as a validity mask on the
  300 boxes; only dropping the invalid ones runs on host (`compact`), as the count left depends on
  the data and device tensors have fixed shapes. Upstream's fallback when no score passes the
  threshold, lowering it by 10% at a time, uses thresholds that depend on the threshold only:
  they are computed once on the host, and the device picks the first one the top score reaches.
- `topk`'s boxes keep cz at the box's gravity center, as upstream's coder does; `finalize` moves it
  to the bottom face on device, as upstream `get_bboxes` does after the coder.

## Model Structure

BEVFormer-base's PyTorch reference (`reference/bevformer.py`, `build_bevformer_base(200, 200)`), the
repeated bottlenecks folded; each module points to its TTNN implementation.

```
BEVFormer(                                                            ----> tt/tt_bevformer.py (TTBEVFormer)
  (img_backbone): ResNet(                                             ----> tt/tt_resnet.py (TTResNet)
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64)                                            ----> folded into conv1 (tt/model_preprocessing.py)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
    (layer1): ResLayer(                                               ----> tt/tt_resnet.py (TTResLayer)
      (0-2): 3 x Bottleneck(                                          ----> tt/tt_resnet.py (TTBottleneck)
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), bias=False)      (64 in block 0)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), bias=False)
        (bn1-3): BatchNorm2d                                          ----> folded into conv1-3
        (downsample): Conv2d(64, 256, kernel_size=(1, 1)) + BatchNorm2d   (block 0)
      )
    )
    (layer2): ResLayer(                                               ----> tt/tt_resnet.py (TTResLayer)
      (0-3): 4 x Bottleneck(                                          ----> tt/tt_resnet.py (TTBottleneck)
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), bias=False)     (256 in, stride 2 in block 0)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), bias=False)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), bias=False)
        (downsample): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2)) + BatchNorm2d   (block 0)
      )
    )
    (layer3): ResLayer(                                               ----> tt/tt_resnet.py (TTResLayer)
      (0-22): 23 x Bottleneck(                                        ----> tt/tt_resnet.py (TTBottleneck)
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), bias=False)    (512 in, stride 2 in block 0)
        (conv2): ModulatedDeformConv2dPack(                           ----> tt/tt_resnet.py (TTModulatedDeformConv2dPack),
                                                                            tt/tt_modulated_deform_conv.py
          (conv_offset): Conv2d(256, 27, kernel_size=(3, 3), padding=(1, 1))
        )
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), bias=False)
        (downsample): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2)) + BatchNorm2d   (block 0)
      )
    )
    (layer4): ResLayer(                                               ----> tt/tt_resnet.py (TTResLayer)
      (0-2): 3 x Bottleneck(                                          ----> tt/tt_resnet.py (TTBottleneck)
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), bias=False)    (1024 in, stride 2 in block 0)
        (conv2): ModulatedDeformConv2dPack(                           ----> tt/tt_resnet.py (TTModulatedDeformConv2dPack),
                                                                            tt/tt_modulated_deform_conv.py
          (conv_offset): Conv2d(512, 27, kernel_size=(3, 3), padding=(1, 1))
        )
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), bias=False)
        (downsample): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2)) + BatchNorm2d   (block 0)
      )
    )
  )
  (img_neck): FPN(                                                    ----> tt/tt_fpn.py (TTFPN)
    (lateral_convs): ModuleList(                                      ----> tt/tt_fpn.py (TTConvModule)
      (0): ConvModule((conv): Conv2d(512, 256, kernel_size=(1, 1)))
      (1): ConvModule((conv): Conv2d(1024, 256, kernel_size=(1, 1)))
      (2): ConvModule((conv): Conv2d(2048, 256, kernel_size=(1, 1)))
    )
    (fpn_convs): ModuleList(                                          ----> tt/tt_fpn.py (TTConvModule)
      (0-2): 3 x ConvModule((conv): Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)))
      (3): ConvModule((conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
    )
  )
  (transformer): PerceptionTransformer(                               ----> tt/tt_perception_transformer.py (TTPerceptionTransformer)
    (encoder): BEVFormerEncoder(                                      ----> tt/tt_encoder.py (TTBEVFormerEncoder)
      (layers): ModuleList(
        (0-5): 6 x BEVFormerLayer(                                    ----> tt/tt_encoder.py (TTBEVFormerLayer)
          (attentions): ModuleList(
            (0): TemporalSelfAttention(                               ----> tt/tt_temporal_self_attention.py (TTTemporalSelfAttention)
              (sampling_offsets): Linear(in_features=512, out_features=128, bias=True)
              (attention_weights): Linear(in_features=512, out_features=64, bias=True)
              (value_proj): Linear(in_features=256, out_features=256, bias=True)
              (output_proj): Linear(in_features=256, out_features=256, bias=True)
            )
            (1): SpatialCrossAttention(                               ----> tt/tt_spatial_cross_attention.py (TTSpatialCrossAttention)
              (deformable_attention): MSDeformableAttention3D(        ----> tt/tt_ms_deformable_attention.py (TTMSDeformableAttention)
                (sampling_offsets): Linear(in_features=256, out_features=512, bias=True)
                (attention_weights): Linear(in_features=256, out_features=256, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
              )
              (output_proj): Linear(in_features=256, out_features=256, bias=True)
            )
          )
          (ffns): ModuleList(
            (0): FFN(                                                 ----> tt/tt_common.py (TTFFN)
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=256, out_features=512, bias=True)
                  (1): ReLU(inplace=True)
                )
                (1): Linear(in_features=512, out_features=256, bias=True)
              )
            )
          )
          (norms): ModuleList(
            (0-2): 3 x LayerNorm((256,), eps=1e-05)                   ----> tt/tt_common.py (layer_norm)
          )
        )
      )
    )
    (can_bus_mlp): Sequential(                                        ----> tt/tt_perception_transformer.py (TTPerceptionTransformer)
      (0): Linear(in_features=18, out_features=128, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=128, out_features=256, bias=True)
      (3): ReLU(inplace=True)
      (norm): LayerNorm((256,), eps=1e-05)
    )
  )
  (head): BEVFormerHead(                                              ----> tt/tt_head.py (TTBEVFormerHead)
    (query_embedding): Embedding(900, 512)                            ----> uploaded once (tt/model_preprocessing.py)
    (reference_points): Linear(in_features=256, out_features=3, bias=True)  ----> computed once (tt/model_preprocessing.py)
    (decoder): DetectionTransformerDecoder(                           ----> tt/tt_decoder.py (TTDetectionTransformerDecoder)
      (layers): ModuleList(
        (0-5): 6 x DetrTransformerDecoderLayer(                       ----> tt/tt_decoder.py (TTDetrTransformerDecoderLayer)
          (attentions): ModuleList(
            (0): MultiheadAttention(                                  ----> tt/tt_decoder.py (TTMultiheadAttention)
              (attn): MultiheadAttention(
                (out_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (1): MSDeformableAttention(                               ----> tt/tt_ms_deformable_attention.py (TTMSDeformableAttention)
              embed_dims=256, num_heads=8, num_levels=1, num_points=4
              (value_proj): Linear(in_features=256, out_features=256, bias=True)
              (sampling_offsets): Linear(in_features=256, out_features=64, bias=True)
              (attention_weights): Linear(in_features=256, out_features=32, bias=True)
              (output_proj): Linear(in_features=256, out_features=256, bias=True)
            )
          )
          (ffns): ModuleList(
            (0): FFN(...)                                             ----> tt/tt_common.py (TTFFN)
          )
          (norms): ModuleList(
            (0-2): 3 x LayerNorm((256,), eps=1e-05)                   ----> tt/tt_common.py (layer_norm)
          )
        )
      )
    )
    (cls_branches): ModuleList(                                       ----> tt/tt_head.py (TTBEVFormerHead)
      (0-5): 6 x Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): LayerNorm((256,), eps=1e-05)
        (2): ReLU(inplace=True)
        (3): Linear(in_features=256, out_features=256, bias=True)
        (4): LayerNorm((256,), eps=1e-05)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=256, out_features=10, bias=True)
      )
    )
    (reg_branches): ModuleList(                                       ----> tt/tt_decoder.py (TTDetectionTransformerDecoder._reg_branch)
      (0-5): 6 x Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=10, bias=True)
      )
    )
  )                                                                   (box coder: tt/tt_nms_free_coder.py (TTNMSFreeCoder))
  (bev_embedding): Embedding(40000, 256)                              ----> uploaded once (tt/model_preprocessing.py)
  (positional_encoding): LearnedPositionalEncoding(                   ----> computed once (tt/model_preprocessing.py)
    (row_embed): Embedding(200, 128)
    (col_embed): Embedding(200, 128)
  )
)
```

## Project Structure

```
models/experimental/bevformer/
├── model_config.py     # BEVFormer-base's configuration and the port's precision and memory choices
├── reference/          # PyTorch reference implementation
├── tests/
│   ├── common.py       # Test helpers: camera rigs, dataset presets, dummy weights, random inputs
│   ├── pcc/            # Unit tests for individual components
│   └── perf/           # Device-perf harnesses, one per stage and one for the detector
└── tt/                 # TTNN implementation; model_preprocessing.py prepares every part's parameters
```

## Test Files

The test suite validates the image backbone, the FPN, the individual components of the BEVFormer encoder, the detection decoder, and the detection head and box coder, each TTNN part against its PyTorch reference.

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
- The tiny (50x50) and base (200x200) BEV grids, a non-square 50x100 grid, batch size 2, and
  batch size 2 with `batch_first=True`
- PCC 0.99 on every layer's output and refined reference points, and per channel on the box
  codes; per-layer PCC of the refinement steps and the refined points' error in BEV pixels are
  logged
- Traced runs: capture proves the forward has no host reads or writes, and the replay runs on new
  inputs
- That a second eager run adds no programs to the program cache

It uses seeded random weights (`tests/common.py`): BEVFormer's sampling-offset grid init
with random weights on top, scaled to what the BEVFormer-base checkpoint's decoder shows on the
same BEV features: the sampling offsets and the cross- and self-attention logits, and in the reg
branches' last Linear the reference-point refinements and the other box channels' spread. The BEV
features are random but spatially smooth, as the encoder's are, and the reference points include
the grid edges. bfloat16 error grows through the reference-point refinement, faster on the 200x200
grid.

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
- The box centers within a mean and a max error in metres (`tests/common.py`), which PCC
  over the 102.4 m axes does not bound
- That the coder, on the head's outputs, selects pairs within twice the score error of the reference
  head's top 300 covering at least 90% of it, with the reference's scores at those pairs within
  0.04 at the 99th percentile and 0.2 at worst (`MAX_SCORE_ERROR`, `MAX_WORST_SCORE_ERROR`), and
  that its device box decoding matches the host's on the same predictions

The decoder and reg branches use `tests/common.py`'s dummy-weight init, and the reference
points BEVFormer's xavier init (`tests/common.py`).

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_head.py
```

#### test_nms_free_coder.py
Tests the box coder on head-shaped inputs.

**What it tests:**
- Distinct logits, so the top 300 labels and queries must match the reference exactly, spread
  evenly, in a background-dominated shape whose dense band defeats a bfloat16 top-k without
  the pivot, and with the 300th logit alone just above a band that rounds to its bfloat16 value
- The decoded boxes channel by channel, yaw through its sine and cosine, and the scores to a few
  float32 ulps
- The kept boxes, with no score threshold, one some scores pass, and one none passes so the
  threshold is lowered, and that a second call only hits the program cache
- The top-k and the validity mask captured as a trace, which fails on any host operation

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
last decoder layer's class logits and, channel by channel, its box predictions, at PCC 0.99, and
the 3D boxes: the device detections are checked as in `test_head.py`, and the host compaction
runs on them. The yaw sine and the velocities gate at 0.94: they are read off small differences
between frames, so they carry the backbone's and the encoder's bfloat16 error the most.

It uses dummy weights (the backbone's, encoder's and head's tests' weights). The reference runs
the backbone on the CPU three times, so the test takes several minutes.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_bevformer.py
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
tiny, base and non-square grids, the CARLA preset (a synthetic ring at CARLA's image size and range), batch size 2 with the second sample's rig turned, a
frame no camera sees (an empty plan) and a frame that fills the plan exactly. PCC 0.999 on the
output and on the attended part alone, except for the empty plan, whose attended part is the output
projection's bias alone. `test_rebatch_plan_update` refills a plan in place for
another rig.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_spatial_cross_attention.py
```

#### test_ms_deformable_attention.py and test_fused_msda.py
Test `TTMSDeformableAttention` against the reference module: on the tiny, base and CARLA presets
with bfloat16 and float32 reference points, and over several seeds, at PCC 0.999; and that two
identical calls agree exactly.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_ms_deformable_attention.py
pytest models/experimental/bevformer/tests/pcc/test_fused_msda.py
```

#### test_point_sampling_3d_2d.py
Tests the bfloat16 TTNN port of the pillar-point projection, which the encoder does not use (it
projects on the host in float32): the BEV grid's 3D reference points exactly, and the projection
into nuScenes' and a synthetic rig's cameras at PCC 0.999 on the points both sides find valid,
the validity masks at PCC 0.994.

**Usage:**
```bash
pytest models/experimental/bevformer/tests/pcc/test_point_sampling_3d_2d.py
```

### Perf harnesses

Located in `models/experimental/bevformer/tests/perf/`. Each runs a stage, on the base 200x200 grid
unless noted, behind a PCC gate that doubles as the warmup, captures its forward as a trace, which fails on any
host read or write, and replays it once between Tracy's `start` and `stop` signposts.
`test_fused_msda_perf.py` instead runs its forward eagerly between the signposts, without a trace,
and its report takes no `--tracing-mode`:

- `test_backbone_fpn_perf.py`: the backbone and the FPN on six 928x1600 images.
- `test_fused_msda_perf.py`: the fused MSDA module and the fused op alone, on the nuScenes tiny
  and base attention shapes.
- `test_encoder_layer_perf.py`: one encoder layer on a 100x100 grid, without the per-frame setup.
- `test_encoder_perf.py`: the six-layer encoder, and one layer alone, on a second frame; it then replays the trace on the
  next frame, with the plan refilled in place.
- `test_decoder_perf.py`: the six-layer decoder.
- `test_head_perf.py`: the head and the coder's top-k.
- `test_bevformer_perf.py`: the detector end to end, images to the 3D detections, then a replay on
  a third frame as a deployment would run it.

**Usage:**
```bash
python -m tracy -p -r -v -m "pytest models/experimental/bevformer/tests/perf/test_bevformer_perf.py"
tt-perf-report <ops_perf_results.csv> --start-signpost start --end-signpost stop --tracing-mode
```

## Running All Tests
To run the complete test suite:

**Usage**
```bash
# Run all PCC tests
pytest models/experimental/bevformer/tests/pcc/
```

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
- The image size, its padding and normalization.
- The port's precision and memory choices: the weight dtypes, the float32 reference points and
  class logits, the coder's bisection steps (`CODER_BISECTION_STEPS`), and the backbone and FPN's per-layer DRAM, sharding,
  act_block_h and fp32-accumulation settings. `tt_resnet_kwargs()` and `tt_fpn_kwargs()` collect
  them; `create_resnet_parameters` and `create_fpn_parameters` prepare the weights for them and
  record them in the parameters, which `TTResNet` and `TTFPN` read them from.

The dataset presets (camera rigs, image sizes, point-cloud ranges) that the deformable-attention,
point-sampling and encoder tests use are test fixtures, in `tests/common.py`.
