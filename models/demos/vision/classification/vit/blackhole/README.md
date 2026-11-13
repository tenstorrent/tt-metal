# ViT

## Platforms:
    Blackhole (P100, P150)

## Introduction
This directory runs ViT-Base (`google/vit-base-patch16-224`) end to end on Blackhole: the patch
embedding, twelve transformer encoder layers and the classifier over 1000 classes, at batch
size 10 and 224x224 input. A high-resolution variant runs single encoder layers at longer sequences
and wider hidden sizes, at batch size 1.

The Vision Transformer splits an image into 16x16 patches and runs them through a standard
Transformer encoder, reaching image-classification accuracy comparable to convolutional networks.
https://arxiv.org/abs/2010.11929

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Demo

From the repository root, run inference with trace and two command queues:

```sh
pytest --disable-warnings models/demos/vision/classification/vit/blackhole/demo/demo_vit_performant_inference.py::test_run_vit_trace_2cqs_inference
```

`VIT_IMAGE_DIR` names a labelled image folder, one subfolder per class named by the classifier's
class index (`0` to `999`) holding that class's images. Without it the demo and tests use
synthetic images, which measure throughput only; with it they also report the top-1 accuracy.
`test_run_vit_trace_2cqs_accuracy` runs every image in the folder and is skipped without it.

## Performance

- End-to-end throughput with trace and two command queues, batch 10:

```sh
pytest --disable-warnings models/demos/vision/classification/vit/blackhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py
```
- The same for the high-resolution variant, over sequence sizes 1024, 2048 and 3072 and hidden sizes
  512, 1024, 1536 and 2304:

```sh
pytest --disable-warnings models/demos/vision/classification/vit/blackhole/tests/test_demo_vit_hiRes_ttnn_inference_perf_e2e_2cq_trace.py
```
- Device ops for a profile:

```sh
./tools/tracy/profile_this.py -n vit -c "pytest --disable-warnings models/demos/vision/classification/vit/blackhole/tests/test_vit_device_perf.py::test_vit_device_ops"
```

## Testing

- Use the following command to run every PCC test:

```sh
pytest --disable-warnings models/demos/vision/classification/vit/blackhole/tests/test_ttnn_optimized_sharded_vit_bh.py models/demos/vision/classification/vit/blackhole/tests/test_ttnn_optimized_sharded_vit_hiRes_bh.py
```
- `test_ttnn_optimized_sharded_vit_bh.py` checks each block against Hugging Face: `test_vit_patch_embeddings`,
  `test_vit_embeddings`, `test_vit_attention`, `test_vit_mlp`, `test_vit_layer`, `test_vit_encoder`, and the
  whole model in `test_vit`. Run the end-to-end test independently with:

```sh
pytest --disable-warnings models/demos/vision/classification/vit/blackhole/tests/test_ttnn_optimized_sharded_vit_bh.py::test_vit
```
- `test_ttnn_optimized_sharded_vit_hiRes_bh.py` checks one high-resolution encoder layer over the same sweep as
  its performance test.

### Reduced grid

The model derives every grid, shard spec and program config from the device. When a core's share of the
activations does not fit its L1, they stay interleaved in DRAM, so the model also runs on two worker cores:

```sh
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,0 pytest models/demos/vision/classification/vit/blackhole/tests/test_ttnn_optimized_sharded_vit_bh.py
```

The override is the inclusive end coordinate of the worker grid, so `1,0` is a 2x1 grid.

### Reduced layer count

`VIT_NUM_LAYERS=N` keeps the first `N` of the twelve encoder layers. The reference is reduced the same way, so
the PCC thresholds still apply, but the reduced model is not the trained classifier. Every test and demo that
loads the pretrained model supports it; the high-resolution tests build single layers and do not:

```sh
VIT_NUM_LAYERS=1 pytest models/demos/vision/classification/vit/blackhole/tests/test_ttnn_optimized_sharded_vit_bh.py::test_vit
```

## Project Structure

```
models/demos/vision/classification/vit/
├── blackhole/
│   ├── demo/      # Inference, accuracy and throughput demos
│   ├── tests/     # PCC tests, end-to-end and device performance, hi-res test infra
│   └── tt/        # TTNN implementation: ttnn_optimized_sharded_vit_bh.py and ttnn_optimized_sharded_vit_hiRes_bh.py
└── common/
    ├── common.py  # Loads the Hugging Face model, honors VIT_NUM_LAYERS
    └── tests/     # Test infra shared with Wormhole: inputs, trace with two command queues, input loading
```

## Model Structure

ViT-Base (`ViTForImageClassification`, `google/vit-base-patch16-224`).

```
ViTForImageClassification(
  (vit): ViTModel(
    (embeddings): ViTEmbeddings(
      (patch_embeddings): ViTPatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (layers): ModuleList(
      (0-11): 12 x ViTLayer(
        (attention): ViTAttention(
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (o_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (mlp): ViTMLP(
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  )
  (classifier): Linear(in_features=768, out_features=1000, bias=True)
)
```

Indented under each file are the modules from the tree it implements, with the function in parentheses.

```
tt/ttnn_optimized_sharded_vit_bh.py
    ViTForImageClassification (vit)
    vit.embeddings (vit_embeddings)
    vit.embeddings.patch_embeddings, the conv as a fold and a matmul (vit_patch_embeddings)
    vit.layers (vit_encoder)
    ViTLayer (vit_layer)
    ViTAttention, q/k/v fused into one projection (vit_attention)
    ViTMLP (vit_mlp)
    vit.layernorm, classifier (vit)
    every module's parameters, taken from the Hugging Face model (custom_preprocessor)

tt/ttnn_optimized_sharded_vit_hiRes_bh.py
    the same modules for the high-resolution variant; ViTAttention runs as scaled dot-product attention
```

## Details
- Entry point for the model is `vit` in `models/demos/vision/classification/vit/blackhole/tt/ttnn_optimized_sharded_vit_bh.py`;
  `update_model_config` derives its grids and program configs from the device, and `custom_preprocessor` prepares its
  parameters from the Hugging Face model.
- Batch size: 10
- Input: 3x224x224, 196 patches plus the class token, padded to 224 tokens; the encoder runs in bfloat8_b.
- Classes: 1000, padded to 1152 for the classifier.
- High-resolution variant (`ttnn_optimized_sharded_vit_hiRes_bh.py`): batch size 1, sequence length 1024, 2048 or
  3072 and hidden size 512, 1024, 1536 or 2304; the standard model trades latency for images per second, this one
  handles more patches per image.
- For the sharding strategies, matmul configs and the encoder layer breakdown, see the
  [ViT Tech Report](../../../../../../tech_reports/ViT-TTNN/vit.md).
