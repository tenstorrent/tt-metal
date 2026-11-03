# RT-DETR

## Platforms:
    Blackhole (P100, P150)

## Introduction
This directory runs RT-DETR and RT-DETRv2 with a ResNet-50-vd backbone end to end on device: the
backbone, the hybrid encoder (one AIFI transformer layer and the CSP-RepVGG FPN/PAN), query
selection over the encoder's proposals, and the six-layer deformable-attention decoder that
predicts boxes and class logits. It supports the `PekingU/rtdetr_r50vd` and
`PekingU/rtdetr_v2_r50vd` checkpoints at batch size 1.

RT-DETR is a real-time end-to-end object detector: a hybrid encoder decouples intra-scale
attention from cross-scale fusion, and IoU-aware query selection picks the decoder's initial
queries, so detection needs no NMS.
https://arxiv.org/abs/2304.08069

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Demo

From the repository root, run inference on a sample COCO image (`--sample_image 0-9`):

```sh
python models/demos/vision/detection/rtdetr/demo.py
```

Run an image or a direct image URL, or up to 20 seconds of a video or direct video URL:

```sh
python models/demos/vision/detection/rtdetr/demo.py --image <path-or-url> --model_version 1
python models/demos/vision/detection/rtdetr/demo.py --video <path-or-url> --model_version 2
```

`--threshold` sets the detection threshold and `--device-id` selects the device. Annotated results
are written to a new directory under `models/demos/vision/detection/rtdetr/outputs/`.

## Performance

- Trace latency at batch 1, one warmup and eight trace replays on different images:

```sh
pytest models/demos/vision/detection/rtdetr/tests/latency.py::test_rtdetr_batch_1_trace_latency
pytest models/demos/vision/detection/rtdetr/tests/latency.py::test_rtdetr_v2_batch_1_trace_latency
```
- `test_rtdetr_batch_1_e2e_latency` also counts input transfer, output transfer and
  postprocessing.
- [VALIDATION.md](VALIDATION.md) records P150a latency, video throughput and detection
  agreement, with its base commit.

## Testing

- Use the following command to run every PCC test:

```sh
pytest --disable-warnings models/demos/vision/detection/rtdetr/tests/pcc.py
```
- Each module test loads the Hugging Face checkpoint as its reference:
  - Backbone: `test_rtdetr_resnet_conv_layer`, `_embeddings`, `_shortcut`, `_bottleneck`, `_stage`,
    `_encoder` and `_backbone`, and `test_rtdetr_conv_encoder` for the backbone's three feature maps.
  - Hybrid encoder: `test_rtdetr_conv_norm_layer`, `test_rtdetr_rep_vgg_block`,
    `test_rtdetr_csp_rep_layer`, `test_rtdetr_encoder_layer`, `test_rtdetr_aifi_layer` and
    `test_rtdetr_hybrid_encoder`.
  - Decoder: `test_rtdetr_mlp`, `test_rtdetr_self_attention`, `test_rtdetr_decoder_mlp`,
    `test_rtdetr_mlp_prediction_head`, `test_rtdetr_multiscale_deformable_attention` (and `_v2_`),
    `test_rtdetr_decoder_layer` and `test_rtdetr_decoder`.
- `test_rtdetr_model` and `test_rtdetr_v2_model` run the whole model on a COCO image. Decoder queries
  are an unordered set, so the outputs are compared on the top-k proposals both implementations
  selected. Run them independently with:

```sh
pytest --disable-warnings models/demos/vision/detection/rtdetr/tests/pcc.py::test_rtdetr_model
```
- `tests/validate_detection.py` compares detections against PyTorch on the first 500 COCO
  `val2017` images; set `COCO_VAL2017_IMAGES` to the image directory (see
  [VALIDATION.md](VALIDATION.md)).

### Reduced grid

The model derives every grid, shard and program config from the device, and places activations
that do not fit the grid's L1 in DRAM. The whole PCC suite passes on two worker cores:

```sh
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,0 pytest models/demos/vision/detection/rtdetr/tests/pcc.py
```

The override is the inclusive end coordinate of the worker grid, so `1,0` is a 2x1 grid.

### Reduced layer count

`RTDETR_NUM_LAYERS=N` keeps `N` repeats of every stack of identical layers: the decoder layers, the
AIFI encoder layers, the RepVGG blocks of each CSP layer, and the repeated blocks of each ResNet
stage (each stage also keeps its first, strided block). The reference is reduced the same way, so
the PCC thresholds still apply, but the reduced model is not the trained detector. The E2E test and
the demo (`--num-layers`, which defaults to the variable) support it:

```sh
RTDETR_NUM_LAYERS=1 pytest models/demos/vision/detection/rtdetr/tests/pcc.py::test_rtdetr_model
python models/demos/vision/detection/rtdetr/demo.py --num-layers 1
```

With `N=1` the decoder runs one layer, so the `ttnn.stack` of its per-layer outputs launches no
concat kernel; every other kernel of the full model still runs.

## Project Structure

```
models/demos/vision/detection/rtdetr/
├── common/
│   ├── arch_50vd.txt      # print() of the v1 Hugging Face model
│   ├── config_50vd.txt    # its Hugging Face config
│   ├── num_layers.py      # RTDETR_NUM_LAYERS: reduces the reference model
│   └── preprocessing.py   # Parameter extraction from the Hugging Face modules
├── demo.py                # Image and video CLI
├── tests/
│   ├── pcc.py             # Module and E2E PCC tests against Hugging Face
│   ├── latency.py         # Trace and end-to-end latency
│   └── validate_detection.py  # Detection agreement on COCO val2017
├── tt/                    # TTNN implementation
└── VALIDATION.md          # Recorded P150a results
```

## Model Structure

RT-DETR v1 (`RTDetrForObjectDetection`, `PekingU/rtdetr_r50vd`), the repeated blocks folded.
RT-DETRv2 (`PekingU/rtdetr_v2_r50vd`) has the same modules and shapes and differs in how the
decoder's deformable attention samples its levels. The full print is `common/arch_50vd.txt`.

```
RTDetrForObjectDetection(
  (model): RTDetrModel(
    (backbone): RTDetrConvEncoder(
      (model): RTDetrResNetBackbone(
        (embedder): RTDetrResNetEmbeddings(
          (embedder): Sequential(
            (0): RTDetrResNetConvLayer(Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) + FrozenBatchNorm2d + ReLU)
            (1): RTDetrResNetConvLayer(Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)) + FrozenBatchNorm2d + ReLU)
            (2): RTDetrResNetConvLayer(Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)) + FrozenBatchNorm2d + ReLU)
          )
          (pooler): MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        (encoder): RTDetrResNetEncoder(
          (stages): ModuleList(
            (0): RTDetrResNetStage(
              (0-2): 3 x RTDetrResNetBottleNeckLayer(
                (layer): Sequential(
                  (0): RTDetrResNetConvLayer(Conv2d(256, 64, kernel_size=(1, 1)) + ReLU)  (64 in, block 0)
                  (1): RTDetrResNetConvLayer(Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)) + ReLU)
                  (2): RTDetrResNetConvLayer(Conv2d(64, 256, kernel_size=(1, 1)))
                )
                (shortcut): RTDetrResNetShortCut(Conv2d(64, 256, kernel_size=(1, 1)))  (block 0, Identity otherwise)
                (activation): ReLU()
              )
            )
            (1): RTDetrResNetStage(
              (0-3): 4 x RTDetrResNetBottleNeckLayer(
                (layer): Sequential(
                  (0): RTDetrResNetConvLayer(Conv2d(512, 128, kernel_size=(1, 1)) + ReLU)  (256 in, block 0)
                  (1): RTDetrResNetConvLayer(Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)) + ReLU)  (stride 2 in block 0)
                  (2): RTDetrResNetConvLayer(Conv2d(128, 512, kernel_size=(1, 1)))
                )
                (shortcut): Sequential(  (block 0, Identity otherwise)
                  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
                  (1): RTDetrResNetShortCut(Conv2d(256, 512, kernel_size=(1, 1)))
                )
              )
            )
            (2): RTDetrResNetStage(
              (0-5): 6 x RTDetrResNetBottleNeckLayer(
                1x1 512/1024 -> 256, 3x3 256 -> 256 (stride 2 in block 0), 1x1 256 -> 1024
                (shortcut): AvgPool2d + Conv2d(512, 1024, kernel_size=(1, 1))  (block 0)
              )
            )
            (3): RTDetrResNetStage(
              (0-2): 3 x RTDetrResNetBottleNeckLayer(
                1x1 1024/2048 -> 512, 3x3 512 -> 512 (stride 2 in block 0), 1x1 512 -> 2048
                (shortcut): AvgPool2d + Conv2d(1024, 2048, kernel_size=(1, 1))  (block 0)
              )
            )
          )
        )
      )
    )
    (encoder_input_proj): ModuleList(
      (0): Conv2d(512, 256, kernel_size=(1, 1)) + BatchNorm2d
      (1): Conv2d(1024, 256, kernel_size=(1, 1)) + BatchNorm2d
      (2): Conv2d(2048, 256, kernel_size=(1, 1)) + BatchNorm2d
    )
    (encoder): RTDetrHybridEncoder(
      (aifi): ModuleList(
        (0): RTDetrAIFILayer(
          (position_embedding): RTDetrSinePositionEmbedding()
          (layers): ModuleList(
            (0): RTDetrEncoderLayer(
              (self_attn): RTDetrSelfAttention(
                (q_proj, k_proj, v_proj, o_proj): 4 x Linear(in_features=256, out_features=256, bias=True)
              )
              (self_attn_layer_norm): LayerNorm((256,), eps=1e-05)
              (mlp): RTDetrMLP(
                (fc1): Linear(in_features=256, out_features=1024, bias=True)
                (fc2): Linear(in_features=1024, out_features=256, bias=True)
                (activation_fn): GELUActivation()
              )
              (final_layer_norm): LayerNorm((256,), eps=1e-05)
            )
          )
        )
      )
      (lateral_convs): ModuleList(
        (0-1): 2 x RTDetrConvNormLayer(Conv2d(256, 256, kernel_size=(1, 1)) + BatchNorm2d + SiLU)
      )
      (fpn_blocks): ModuleList(
        (0-1): 2 x RTDetrCSPRepLayer(
          (conv1): RTDetrConvNormLayer(Conv2d(512, 256, kernel_size=(1, 1)) + BatchNorm2d + SiLU)
          (conv2): RTDetrConvNormLayer(Conv2d(512, 256, kernel_size=(1, 1)) + BatchNorm2d + SiLU)
          (bottlenecks): Sequential(
            (0-2): 3 x RTDetrRepVggBlock(
              (conv1): RTDetrConvNormLayer(Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)) + BatchNorm2d)
              (conv2): RTDetrConvNormLayer(Conv2d(256, 256, kernel_size=(1, 1)) + BatchNorm2d)
              (activation): SiLUActivation()
            )
          )
          (conv3): Identity()
        )
      )
      (downsample_convs): ModuleList(
        (0-1): 2 x RTDetrConvNormLayer(Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) + BatchNorm2d + SiLU)
      )
      (pan_blocks): ModuleList(
        (0-1): 2 x RTDetrCSPRepLayer(...)  (as fpn_blocks)
      )
    )
    (denoising_class_embed): Embedding(81, 256, padding_idx=80)  (training only)
    (enc_output): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): LayerNorm((256,), eps=1e-05)
    )
    (enc_score_head): Linear(in_features=256, out_features=80, bias=True)
    (enc_bbox_head): RTDetrMLPPredictionHead(
      (layers): ModuleList(
        (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
        (2): Linear(in_features=256, out_features=4, bias=True)
      )
    )
    (decoder_input_proj): ModuleList(
      (0-2): 3 x Conv2d(256, 256, kernel_size=(1, 1)) + BatchNorm2d
    )
    (decoder): RTDetrDecoder(
      (layers): ModuleList(
        (0-5): 6 x RTDetrDecoderLayer(
          (self_attn): RTDetrSelfAttention(
            (q_proj, k_proj, v_proj, o_proj): 4 x Linear(in_features=256, out_features=256, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((256,), eps=1e-05)
          (encoder_attn): RTDetrMultiscaleDeformableAttention(
            (attn): MultiScaleDeformableAttention()
            (sampling_offsets): Linear(in_features=256, out_features=192, bias=True)
            (attention_weights): Linear(in_features=256, out_features=96, bias=True)
            (value_proj): Linear(in_features=256, out_features=256, bias=True)
            (output_proj): Linear(in_features=256, out_features=256, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((256,), eps=1e-05)
          (mlp): RTDetrMLP(
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (activation_fn): ReLU()
          )
          (final_layer_norm): LayerNorm((256,), eps=1e-05)
        )
      )
      (query_pos_head): RTDetrMLPPredictionHead(
        (0): Linear(in_features=4, out_features=512, bias=True)
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (class_embed): ModuleList(
        (0-5): 6 x Linear(in_features=256, out_features=80, bias=True)
      )
      (bbox_embed): ModuleList(
        (0-5): 6 x RTDetrMLPPredictionHead(
          (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
          (2): Linear(in_features=256, out_features=4, bias=True)
        )
      )
    )
  )
)
```

Indented under each file are the modules from the tree it implements, with the class in parentheses.

```
tt/model.py
    model (TtRTDetrModel)
    model.encoder_input_proj, model.decoder_input_proj (TtRTDetrResNetConvLayer)
    model.enc_output, enc_score_head, enc_bbox_head (TtRTDetrModel)
    query selection: anchors, computed once at init; top-k proposals and their gather (TtRTDetrModel)

tt/backbone.py
    model.backbone (TtRTDetrConvEncoder)
    model.backbone.model (TtRTDetrResNetBackBone)
    model.backbone.model.embedder (TtRTDetrResNetEmbeddings)
    model.backbone.model.encoder (TtRTDetrResNetEncoder)
    RTDetrResNetStage (TtRTDetrResNetStage)
    RTDetrResNetBottleNeckLayer (TtRTDetrResNetBottleNeckLayer)
    RTDetrResNetShortCut, with its AvgPool2d (TtRTDetrResNetShortcut)
    RTDetrResNetConvLayer, every conv with its norm folded in at init (TtRTDetrResNetConvLayer)

tt/encoder.py
    model.encoder (TtRTDetrHybridEncoder)
    model.encoder.aifi, with its position embedding computed once at init (TtRTDetrAIFILayer)
    AIFI layers (TtRTDetrEncoderLayer)
    RTDetrSelfAttention (TtRTDetrSelfAttention)
    RTDetrMLP (TtRTDetrMLP)
    RTDetrConvNormLayer (TtRTDetrConvNormLayer)
    fpn_blocks, pan_blocks (TtRTDetrCSPRepLayer)
    RTDetrRepVggBlock (TtRTDetrRepVggBlock)

tt/decoder.py
    model.decoder (TtRTDetrDecoder)
    decoder layers (TtRTDetrDecoderLayer)
    decoder self_attn (TtRTDetrDecoderSelfAttention)
    encoder_attn (TtRTDetrMultiscaleDeformableAttention)
    decoder mlp (TtRTDetrDecoderMLP)
    query_pos_head, bbox_embed (TtRTDetrMLPPredictionHead)
    class_embed (TtRTDetrDecoder)

common/preprocessing.py
    every module's parameters, taken from the Hugging Face model
```

## Details
- Entry point for the model is `TtRTDetrModel` in `models/demos/vision/detection/rtdetr/tt/model.py`;
  `common/preprocessing.py` prepares its parameters from the Hugging Face model.
- Batch size: 1
- Input: 3x640x640, bfloat16; the input resolution and batch size are fixed when the model is built.
- Feature maps: strides 8, 16 and 32 (80x80, 40x40, 20x20); 300 decoder queries; 80 COCO classes.
