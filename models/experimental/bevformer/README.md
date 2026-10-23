# BEVFormer

## Platforms:
    Blackhole (p100)

## Introduction
This directory runs BEVFormer-base end to end on device: six camera images through the
ResNet101-DCN backbone and FPN, the perception transformer around the BEVFormer encoder, the
detection decoder and the head with its NMS-free box coder, ending in 3D boxes.

BEVFormer learns a Bird's-Eye-View representation from multi-camera images with spatiotemporal
transformers: a spatial cross-attention into the cameras and a temporal self-attention into the
previous frame's BEV.
https://arxiv.org/abs/2203.17270

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- Use the following command to run the detector end to end over two frames against the reference:
```sh
pytest --disable-warnings models/experimental/bevformer/tests/pcc/test_bevformer.py
```

### Performance
- For the detector's device ops, one live frame with a previous BEV between Tracy's `start` and
  `stop` signposts:
```sh
python -m tracy -p -r -v -m "pytest models/experimental/bevformer/tests/perf/test_bevformer_perf.py"
tt-perf-report <ops_perf_results.csv> --start-signpost start --end-signpost stop
```
- The per-stage harnesses in `tests/perf/` (`test_backbone_fpn_perf.py`, `test_decoder_perf.py`,
  `test_head_perf.py`) capture a trace and replay it once; add `--tracing-mode` to `tt-perf-report`.
  `test_encoder_perf.py`, `test_encoder_layer_perf.py` and `test_fused_msda_perf.py` run live.

## Testing
- Use the following command to run every PCC test:
```sh
pytest --disable-warnings models/experimental/bevformer/tests/pcc/
```
- `test_resnet.py`: the ResNet101-DCN backbone's bottlenecks and layers against the reference.
- `test_fpn.py`: the FPN neck on random C3-C5 in the dtypes the backbone emits.
- `test_backbone_fpn.py`: camera images through the backbone and the FPN end to end.
- `test_ms_deformable_attention.py`, `test_fused_msda.py`: the deformable attention module and the fused MSDA op.
- `test_point_sampling_3d_2d.py`: the BEV pillars' projection into the cameras and the visibility mask.
- `test_temporal_self_attention.py`: the temporal self-attention, batch-first and sequence-first.
- `test_spatial_cross_attention.py`: the spatial cross-attention with its per-camera rebatch.
- `test_encoder.py`: the encoder on several presets and layer counts.
- `test_perception_transformer.py`: the encoder's glue over consecutive frames with ego motion.
- `test_decoder.py`: the six-layer detection decoder and its reference-point refinement.
- `test_head.py`: the detection head, and the coder on its outputs.
- `test_nms_free_coder.py`: the box coder's exact top-k, decoding and filters.
- `test_bevformer.py`: the detector end to end over two frames.

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

## Details
- Entry point for the model is `TTBEVFormer` in `models/experimental/bevformer/tt/tt_bevformer.py`;
  `tt/model_preprocessing.py` prepares its parameters from the reference.
- Batch size: 1
- Input: 6 cameras at 1600x900, padded to 1600x928
- BEV grid: 200x200; 900 object queries; up to 300 boxes per frame
- The encoder reads the cameras' visibility on the host inside its forward, so the detector runs
  live, not under a trace.
