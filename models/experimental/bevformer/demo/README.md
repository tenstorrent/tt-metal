# BEVFormer sequence demo

This demo runs one unmeasured preparation frame followed by a configurable number of output
frames. The first output frame is also executed once as an unmeasured temporal-path warmup without
advancing the retained BEV state. Model architecture, weights, and input sequence are independent
command-line arguments.

## Synthetic smoke run

The synthetic mode uses dummy weights and newly generated random camera images for every frame.
Its boxes have no semantic meaning.

```sh
python -m models.experimental.bevformer.demo.demo \
  --config models/experimental/bevformer/config/bevformer_tiny.json \
  --weights random \
  --dataset-format synthetic \
  --num-frames 10
```

`--variant tiny|small|base` is a shorthand for the corresponding bundled config.

## Why a PyTorch model is constructed

The PyTorch BEVFormer object is a checkpoint-compatible parameter and shape container. The demo
loads the checkpoint into it, converts its weights to TTNN parameters, and performs one CPU
backbone/FPN shape-inference pass while preparing convolution configurations. It is not executed
for every frame and its detections are not compared with the TTNN detections. This parameter-source
pattern is also used by other TT-Metal demos; numerical comparison belongs in PCC tests.

## Checkpoint and real sequence

```sh
python -m models.experimental.bevformer.demo.demo \
  --config /path/to/model.json \
  --weights /path/to/checkpoint.pth \
  --checkpoint-format mmdet3d \
  --dataset-format manifest \
  --dataset /path/to/sequence.json \
  --dataset-root /path/to/dataset \
  --scene-index 0 \
  --start-frame 0 \
  --num-frames 10 \
  --score-threshold 0.25 \
  --output-dir bevformer_demo_output
```

The normalized model config contains the architecture differences, preprocessing and ordered
camera contract required for inference. Bundled examples live in `config/bevformer_*.json`:

```json
{
  "schema_version": 1,
  "name": "bevformer_base",
  "variant": "base",
  "class_names": [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
  ],
  "input": {
    "camera_order": [
      "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
      "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ],
    "can_bus_dims": 18,
    "camera_image_size": [1600, 900],
    "image_scale": 1.0,
    "color_order": "BGR",
    "mean": [103.53, 116.28, 123.675],
    "std": [1.0, 1.0, 1.0],
    "size_divisor": 32
  },
  "backbone": {"depth": 101, "style": "caffe", "with_dcn": true, "out_indices": [1, 2, 3]},
  "fpn": {"num_levels": 4},
  "transformer": {
    "bev_size": 200,
    "embed_dims": 256,
    "num_heads": 8,
    "feedforward_channels": 512,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "spatial_num_points": 8,
    "temporal_num_points": 4,
    "decoder_num_points": 4,
    "num_points_in_pillar": 4,
    "num_query": 900,
    "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    "rotate_center": [100, 100]
  },
  "box_coder": {
    "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    "max_detections": 300
  }
}
```

The number of class names, when supplied, must match the model head. The checkpoint is loaded
strictly; a variant, class-count, or parameter-shape mismatch stops before inference. The current
bundled architectures and their checkpoints use six learned camera embeddings and an 18-value CAN
bus MLP input. The selected config supplies those contracts to the model, manifest adapter and
converter, avoiding duplicated constants; incompatible checkpoint shapes fail strict loading.

`--config` accepts a local path or an HTTPS URL. GitHub `blob` links are converted to raw links;
directory links are rejected. Remote files are cached under `~/.cache/tt-metal/bevformer` by
default. Use `--config-sha256` to pin their contents and `--refresh-config` to refetch them.
Normalized JSON is validated directly. A `.py` path or URL is treated as an official-style config
and interpreted by a restricted AST adapter; it is never imported or executed. Use
`--config-format json|upstream` only when the filename does not identify the syntax.

An upstream URL can therefore be passed directly:

```sh
python -m models.experimental.bevformer.demo.demo \
  --config https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py \
  --weights /path/to/bevformer_tiny.pth \
  --dataset-format manifest \
  --dataset /path/to/sequence.json
```

To save and inspect its normalized form, use the same non-executing adapter as a converter:

```sh
python -m models.experimental.bevformer.config.convert_upstream \
  --input https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py \
  --output /tmp/bevformer_tiny.json
```

For a normalized remote config pinned by content:

```sh
python -m models.experimental.bevformer.demo.demo \
  --config https://raw.githubusercontent.com/ORG/REPO/COMMIT/bevformer_tiny.json \
  --config-sha256 EXPECTED_HEX_DIGEST \
  --weights /path/to/checkpoint.pth \
  --dataset-format manifest \
  --dataset /path/to/sequence.json
```

## Sequence manifest

The manifest is source-neutral. Matrices are calibrated for each source image's original pixel
size; the loader rescales them with the images and pads normalized images at the bottom and right.
`can_bus` is the absolute 18-value vector for the frame. The runner makes translation and heading
relative to the preceding frame.

```json
{
  "camera_order": [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
  ],
  "frames": [
    {
      "token": "frame-0000",
      "scene_token": "scene-0",
      "timestamp": 0,
      "cameras": {
        "CAM_FRONT": {
          "image": "samples/CAM_FRONT/0000.jpg",
          "lidar2img": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        },
        "CAM_FRONT_RIGHT": {
          "image": "samples/CAM_FRONT_RIGHT/0000.jpg",
          "lidar2img": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        },
        "CAM_FRONT_LEFT": {
          "image": "samples/CAM_FRONT_LEFT/0000.jpg",
          "lidar2img": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        },
        "CAM_BACK": {
          "image": "samples/CAM_BACK/0000.jpg",
          "lidar2img": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        },
        "CAM_BACK_LEFT": {
          "image": "samples/CAM_BACK_LEFT/0000.jpg",
          "lidar2img": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        },
        "CAM_BACK_RIGHT": {
          "image": "samples/CAM_BACK_RIGHT/0000.jpg",
          "lidar2img": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        }
      },
      "can_bus": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
  ]
}
```

All frames selected for one run must share `scene_token`. At least `--num-frames + 1` frames are
required because the first establishes the temporal BEV and is not included in the results.

An existing trusted MMDetection3D camera info pickle can be converted without dataset-specific
inference code:

```sh
python -m models.experimental.bevformer.demo.convert_info \
  --config models/experimental/bevformer/config/bevformer_base.json \
  --input /path/to/infos_val.pkl \
  --output /path/to/sequence.json \
  --strip-prefix data/dataset
```

The converter looks cameras up by name and writes them in the config's `camera_order`, which must
match the order used while training the camera embeddings. Use `--dataset-root` at inference time
to resolve the converted relative paths. Because pickle loading can execute code, only convert
metadata from a trusted source.

The demo writes `detections.json`, containing per-frame boxes, scores, numeric labels, optional
class names, device latency, end-to-end latency, and a run summary.
