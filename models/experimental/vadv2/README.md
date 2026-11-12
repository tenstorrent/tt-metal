# VAD

## Platforms:
    Blackhole (P100)

## Introduction
This directory runs VAD-Tiny end to end on device: six camera images through the ResNet50 backbone
and FPN, the perception transformer around the BEVFormer encoder, the detection and map decoders,
the motion and planning decoders, and the device-side box and map coders, ending in 3D boxes with
agent trajectories, vectorized map elements and the ego plan for each driving command.

VAD represents the driving scene as vectors: agents with their future motion and map elements as
polylines, both learned from a Bird's-Eye-View built from the cameras, and plans the ego trajectory
by attending to them.
https://arxiv.org/abs/2303.12077

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- The PCC tests need the reference weights, which are not checked into the repository:

```sh
bash models/experimental/vadv2/weights_download.sh
```

## Demo

The sequence demo takes the checkpoint and the dataset independently. It runs one preparation frame
followed by 10 output frames by default. From the repository root, a synthetic run, which measures
the model's FPS, is:

```sh
python -m models.experimental.vadv2.demo.demo \
  --checkpoint random \
  --dataset-format synthetic \
  --num-frames 10
```

For a real sequence, pass the checkpoint and a sequence manifest:

```sh
python -m models.experimental.vadv2.demo.demo \
  --checkpoint /path/to/checkpoint.pth \
  --dataset-format manifest \
  --dataset /path/to/sequence.json \
  --dataset-root /path/to/dataset \
  --num-frames 10
```

See [`demo/README.md`](demo/README.md) for the manifest schema, scene selection, the output format
and visualization.

## Performance

- See the demo's [recorded smoke results](demo/README.md#recorded-smoke-results) for ten-frame
  end-to-end measurements.
- `test_tt_vad.py` reads `TT_VADV2_WARM_ITERS=N` for extra forwards after the first and
  `TT_VADV2_TIMING=1` to log each forward's wall time:

```sh
TT_VADV2_TIMING=1 TT_VADV2_WARM_ITERS=2 pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py -s
```

## Testing

- Use the following command to run every PCC test:

```sh
pytest --disable-warnings models/experimental/vadv2/tests/pcc/
```
- Every test compares against the PyTorch reference in `reference/` with a PCC floor of 0.99, the
  coders' outputs with 0.999.
- `test_tt_backbone.py`: camera images through the ResNet50 backbone and the FPN.
- `test_tt_temporal_self_attention.py`: the temporal self-attention.
- `test_tt_spatial_cross_attention.py`: the spatial cross-attention with its per-camera rebatch.
- `test_tt_transformer.py`: the perception transformer: the BEV encoder and the detection and map
  decoders.
- `test_tt_mha.py`: the multi-head attention with a key padding mask.
- `test_tt_head.py`: the head end to end, at two map score thresholds for the motion decoder's map
  queries.
- `test_tt_nms_free_coder.py`: the detection and map coders' top-k, decoding and filters.
- `test_tt_vad.py`: the model end to end over two frames, all nine outputs against the reference.
  Run it independently with:

```sh
pytest --disable-warnings models/experimental/vadv2/tests/pcc/test_tt_vad.py
```

### Reduced grid

The model sizes its memory from the device's grid. On the full grid every activation stays where it
is today; on a grid with fewer cores, an activation whose L1 share would crowd out the tensors live
beside it goes to DRAM instead (`tt/common.py`). A backbone conv whose input and output do not fit L1
runs over bands of output rows, each through the same L1 conv with only the input rows it reads,
and a conv whose weights alone leave a band no room runs from DRAM. A banded block-sharded conv is
width sharded with the full grid's channel blocks, since the block width decides how often its
partial sums are rounded. The fused multi-scale deformable attention takes the per-query counts from
a device tensor rather than its runtime arguments. The end-to-end test passes on two worker cores:

```sh
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,0 pytest --disable-warnings models/experimental/vadv2/tests/pcc/test_tt_vad.py
```

The override is the inclusive end coordinate of the worker grid, so `1,0` is a 2x1 grid.

## Project Structure

```
models/experimental/vadv2/
├── common.py           # Loads the reference weights for the tests
├── demo/               # Sequence CLI, manifest adapter and visualizer
├── reference/          # PyTorch reference implementation
├── tests/
│   └── pcc/            # Unit tests for individual components and the model end to end
├── tt/                 # TTNN implementation; model_preprocessing.py prepares every part's parameters
└── weights_download.sh # Fetches the reference weights the tests use
```

## Model Structure

VAD-Tiny's PyTorch reference (`reference/vad.py`), the repeated layers folded.

```
VAD(
  (img_backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
    (layer1): Sequential(
      (0-2): 3 x Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), bias=False)  (64 in block 0)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), bias=False)
        (bn1-3): BatchNorm2d
        (downsample): Conv2d(64, 256, kernel_size=(1, 1), bias=False) + BatchNorm2d  (block 0)
      )
    )
    (layer2): Sequential(0-3): 4 x Bottleneck(256 -> 128 -> 512), stride 2 in block 0
    (layer3): Sequential(0-5): 6 x Bottleneck(512 -> 256 -> 1024), stride 2 in block 0
    (layer4): Sequential(0-2): 3 x Bottleneck(1024 -> 512 -> 2048), stride 2 in block 0
  )
  (img_neck): FPN(
    (lateral_convs): ConvModule((conv): Conv2d(2048, 256, kernel_size=(1, 1)))
    (fpn_convs): ConvModule((conv): Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)))
  )
  (pts_bbox_head): VADHead(
    (positional_encoding): LearnedPositionalEncoding(
      (row_embed): Embedding(100, 128)
      (col_embed): Embedding(100, 128)
    )
    (transformer): VADPerceptionTransformer(
      (encoder): BEVFormerEncoder(
        (layers): ModuleList(
          (0-2): 3 x BEVFormerLayer(
            (attentions): ModuleList(
              (0): TemporalSelfAttention(
                (sampling_offsets): Linear(in_features=512, out_features=128)
                (attention_weights): Linear(in_features=512, out_features=64)
                (value_proj): Linear(in_features=256, out_features=256)
                (output_proj): Linear(in_features=256, out_features=256)
              )
              (1): SpatialCrossAttention(
                (deformable_attention): MSDeformableAttention3D(
                  (sampling_offsets): Linear(in_features=256, out_features=128)
                  (attention_weights): Linear(in_features=256, out_features=64)
                  (value_proj): Linear(in_features=256, out_features=256)
                )
                (output_proj): Linear(in_features=256, out_features=256)
              )
            )
            (ffns): FFN(Linear(256, 512), ReLU, Linear(512, 256))
            (norms): 3 x LayerNorm((256,))
          )
        )
      )
      (decoder): DetectionTransformerDecoder(
        (layers): ModuleList(
          (0-2): 3 x BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(embed_dims=256, num_heads=8)
              (1): CustomMSDeformableAttention(
                (sampling_offsets): Linear(in_features=256, out_features=64)
                (attention_weights): Linear(in_features=256, out_features=32)
                (value_proj): Linear(in_features=256, out_features=256)
                (output_proj): Linear(in_features=256, out_features=256)
              )
            )
            (ffns): FFN(Linear(256, 512), ReLU, Linear(512, 256))
            (norms): 3 x LayerNorm((256,))
          )
        )
      )
      (map_decoder): MapDetectionTransformerDecoder(
        (layers): 3 x BaseTransformerLayer, as the decoder's
      )
      (reference_points): Linear(in_features=256, out_features=3)
      (map_reference_points): Linear(in_features=256, out_features=2)
      (can_bus_mlp): Sequential(Linear(18, 128), ReLU, Linear(128, 256), ReLU, LayerNorm((256,)))
    )
    (cls_branches): 3 x Sequential(Linear(256, 256), LayerNorm, ReLU, Linear(256, 256), LayerNorm, ReLU, Linear(256, 10))
    (reg_branches): 3 x Sequential(Linear(256, 256), ReLU, Linear(256, 256), ReLU, Linear(256, 10))
    (traj_branches): 1 x Sequential(Linear(512, 512), ReLU, Linear(512, 512), ReLU, Linear(512, 12))
    (traj_cls_branches): 1 x Sequential(Linear(512, 512), LayerNorm, ReLU, Linear(512, 512), LayerNorm, ReLU, Linear(512, 1))
    (map_cls_branches): 3 x Sequential(Linear(256, 256), LayerNorm, ReLU, Linear(256, 256), LayerNorm, ReLU, Linear(256, 3))
    (map_reg_branches): 3 x Sequential(Linear(256, 256), ReLU, Linear(256, 256), ReLU, Linear(256, 2))
    (bev_embedding): Embedding(10000, 256)
    (query_embedding): Embedding(300, 512)
    (map_instance_embedding): Embedding(100, 512)
    (map_pts_embedding): Embedding(20, 512)
    (motion_decoder): CustomTransformerDecoder(
      (layers): 1 x BaseTransformerLayer(
        (attentions): MultiheadAttention(embed_dims=256, num_heads=8)
        (ffns): FFN(Linear(256, 512), ReLU, Linear(512, 256))
        (norms): 2 x LayerNorm((256,))
      )
    )
    (motion_mode_query): Embedding(6, 256)
    (pos_mlp_sa): Linear(in_features=2, out_features=256)
    (lane_encoder): LaneNet(
      (layer_seq): 3 x MLP(Linear(256, 128), LayerNorm((128,)), ReLU)
    )
    (motion_map_decoder): CustomTransformerDecoder, as the motion decoder
    (pos_mlp): Linear(in_features=2, out_features=256)
    (ego_query): Embedding(1, 256)
    (ego_agent_decoder): CustomTransformerDecoder, as the motion decoder
    (ego_agent_pos_mlp): Linear(in_features=2, out_features=256)
    (ego_map_decoder): CustomTransformerDecoder, as the motion decoder
    (ego_map_pos_mlp): Linear(in_features=2, out_features=256)
    (ego_fut_decoder): Sequential(Linear(512, 512), ReLU, Linear(512, 512), ReLU, Linear(512, 36))
    (agent_fus_mlp): Sequential(Linear(3072, 256), LayerNorm((256,)), ReLU, Linear(256, 256))
  )
)
```

Indented under each file are the modules from the tree it implements, with the class in parentheses.

```
tt/tt_vad.py
    VAD (TtVAD)

tt/tt_backbone.py
    img_backbone (TtResnet50)

tt/tt_bottleneck.py
    img_backbone.layer1-4 Bottleneck (TtBottleneck)

tt/tt_fpn.py
    img_neck (TtFPN)
    img_neck.lateral_convs, img_neck.fpn_convs (TtConvModule)

tt/common.py
    every Conv2d (TtConv2D)

tt/tt_transformer.py
    transformer (TtVADPerceptionTransformer)
    transformer.can_bus_mlp, transformer.reference_points (TtVADPerceptionTransformer)

tt/tt_encoder.py
    transformer.encoder (TtBEVFormerEncoder)
    transformer.encoder layers (TtBEVFormerLayer)

tt/tt_temporal_self_attention.py
    TemporalSelfAttention (TtTemporalSelfAttention)

tt/tt_spatial_cross_attention.py
    SpatialCrossAttention (TtSpatialCrossAttention)
    MSDeformableAttention3D (TtMSDeformableAttention3D)

tt/tt_point_sampling.py, tt/tt_sca_rebatch_plan.py
    the BEV pillars' projection into the cameras and the spatial cross-attention's rebatch plan

tt/tt_decoder.py
    transformer.decoder (TtDetectionTransformerDecoder)
    transformer.map_decoder (TtMapDetectionTransformerDecoder)
    motion_decoder, motion_map_decoder, ego_agent_decoder, ego_map_decoder (TtCustomTransformerDecoder)

tt/tt_base_transformer_layer.py
    decoder layers (TtBaseTransformerLayer)

tt/tt_deformable_attention.py
    CustomMSDeformableAttention (TtCustomMSDeformableAttention)

tt/tt_mha.py
    MultiheadAttention (TtMultiheadAttention)

tt/tt_ffn.py
    FFN (TtFFN)

tt/tt_head.py
    pts_bbox_head (TtVADHead)
    positional_encoding (TtLearnedPositionalEncoding)
    cls, reg, traj, traj_cls, map_cls and map_reg branches, ego_fut_decoder, agent_fus_mlp,
    pos_mlp, pos_mlp_sa, ego_agent_pos_mlp, ego_map_pos_mlp (TtVADHead)

tt/tt_lanenet.py, tt/tt_mlp.py
    lane_encoder (TtLaneNet), its MLPs (TtMLP)

tt/tt_nms_free_coder.py
    box coder (TtVADDetectionCoder), map coder (TtVADMapCoder)

tt/model_preprocessing.py
    img_backbone.bn1, folded into conv1
    Bottleneck bn1-3 and downsample BatchNorm2d, folded into their convs
    bev_embedding, query_embedding and the positional encoding's tables, uploaded once
```

## Details
- Entry point for the model is `TtVAD` in `models/experimental/vadv2/tt/tt_vad.py`;
  `tt/model_preprocessing.py` prepares its parameters from the reference.
- Batch size: 1
- Input: 6 cameras at 1600x900, scaled by 0.4 to 640x360 and padded to 640x384
- BEV grid: 100x100 over 30 m x 60 m; 300 object queries, 100 map queries of 20 points each;
  6 trajectory modes over 6 future steps for agents and ego
- Preparing a frame projects the cameras on device and reads back only their visible counts, which
  size the spatial cross-attention's rebatch plan; the plan's shapes follow a capacity that only
  grows.
