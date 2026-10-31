# BEVFormer sequence demo

This demo runs one unmeasured preparation frame followed by a configurable number of output
frames. The first output position is also executed once as an unmeasured temporal-path warmup
without advancing the retained BEV state. Model architecture, checkpoint, and input sequence are
independent command-line arguments.

## Synthetic smoke run

The synthetic mode uses dummy weights and newly generated random camera images for every frame.
Its boxes have no semantic meaning.

Run the synthetic demos from the tt-metal repository root:

```sh
python -m models.experimental.bevformer.demo.demo \
  --config models/experimental/bevformer/config/bevformer_tiny.json \
  --checkpoint random \
  --dataset-format synthetic \
  --num-frames 10 \
  --output-dir generated/demo/bevformer/synthetic_tiny

python -m models.experimental.bevformer.demo.demo \
  --config models/experimental/bevformer/config/bevformer_small.json \
  --checkpoint random \
  --dataset-format synthetic \
  --num-frames 10 \
  --output-dir generated/demo/bevformer/synthetic_small

python -m models.experimental.bevformer.demo.demo \
  --config models/experimental/bevformer/config/bevformer_base.json \
  --checkpoint random \
  --dataset-format synthetic \
  --num-frames 10 \
  --output-dir generated/demo/bevformer/synthetic_base
```

`--variant tiny|small|base` is a shorthand for the corresponding bundled config.

## Recorded smoke results

These runs were performed on a BH P100 with tt-metal commit
`326b4e8aa4b28500e875d2503f3bb88cc6ecf904`. The generated `detections.json` artifacts contain
these recorded results for ten measured frames. Each row used the corresponding bundled local
`config/bevformer_<variant>.json` file shown in the commands above:

| Variant | BEV size | Encoder layers | Device-path mean (ms) | Device-path p50 (ms) | E2E mean (ms) | E2E p50 (ms) | Device FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 50 | 3 | 138.6 | 138.2 | 306.9 | 305.9 | 7.21 |
| small | 150 | 3 | 563.0 | 514.4 | 1145.8 | 1103.1 | 1.78 |
| base | 200 | 6 | 1227.4 | 1222.5 | 2148.0 | 2124.6 | 0.81 |

These values come from one ten-frame smoke run per variant and validate end-to-end execution only.
No repeated-run statistics were collected, so they are not statistically characterized benchmark
results. The device-path columns report host wall-clock `device_latency_ms`, not profiler device
duration. The artifacts record random weights, synthetic inputs, `synthetic-0` as the unmeasured
preparation frame, ten measured frames, the TT architecture and card, and the full tt-metal commit
SHA. They do not record the firmware version or original command line.

## Model configuration

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
GitHub `tree` links are rejected. Remote files are cached under `~/.cache/tt-metal/bevformer` by
default. Use `--config-sha256` to pin their contents and `--refresh-config` to refetch them.
Normalized JSON is validated directly. A `.py` path or URL is treated as an official-style config
and interpreted by a restricted AST adapter; it is never imported or executed. Use
`--config-format json|upstream` only when the filename does not identify the syntax.

The real-sequence command below can use an upstream URL directly by replacing its `--config`
argument with:

```text
--config https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py
```

To save and inspect its normalized form, use the same non-executing adapter as a converter:

```sh
python -m models.experimental.bevformer.config.convert_upstream \
  --input https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py \
  --output /path/to/bevformer_tiny.json
```

For a normalized remote config pinned by content, replace the same argument with:

```text
--config https://raw.githubusercontent.com/ORG/REPO/COMMIT/bevformer_tiny.json
--config-sha256 EXPECTED_HEX_DIGEST
```

## Sequence manifest

The manifest is source-neutral. Matrices are calibrated for each source image's original pixel
size; the loader rescales them with the images and pads normalized images at the bottom and right.
`can_bus` is the absolute 18-value vector for the frame. The runner makes translation and heading
relative to the preceding frame.

The synthetic smoke runs above reuse one calibrated `lidar2img` matrix per camera for every frame.
Real datasets may provide frame-dependent matrices, which change the visibility mask, spatial
cross-attention rebatching, and `max_len`. The recorded synthetic timings therefore may not
represent real-sequence performance. Deterministic synthetic sequences with variable, valid
projection geometry are tracked in [#55190](https://github.com/tenstorrent/tt-metal/issues/55190).

The following schema fragment shows one representative frame. Identity `lidar2img` matrices are
used only to keep the structure readable; real inference requires calibrated projection matrices.

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

### Prepare a manifest locally

The demo accepts JSON manifests only. It never loads pickle metadata.

Prepare the manifest outside the tt-metal repository:

1. Generate the dataset metadata locally.
2. Use only metadata you generated or verified.
3. Never load a downloaded, untrusted pickle.
4. Run conversion inside the dataset environment.
5. Write JSON matching the schema above.
6. Pass JSON through `--dataset`.
7. Pass the image root through `--dataset-root`.

Example local conversion command:

```sh
python /path/to/local/convert_info.py \
  --input /path/to/infos_val.pkl \
  --output /path/to/sequence.json \
  --strip-prefix /path/to/dataset/root \
  --camera-order CAM_FRONT CAM_FRONT_RIGHT CAM_FRONT_LEFT \
                 CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT \
  --can-bus-dims 18
```

The local converter must preserve these contracts:

- `camera_order` exactly matches the selected model config.
- Every frame contains exactly those camera names.
- Every camera provides one readable image path.
- Every camera provides one numeric `4x4` `lidar2img` matrix.
- Matrices target each source image's original dimensions.
- `can_bus` contains the configured number of values.
- `can_bus` stores absolute frame state.
- `scene_token` groups frames into independent sequences.
- `timestamp` defines chronological order within each scene.
- Relative image paths resolve below `--dataset-root`.
- Absolute image paths remain valid without `--dataset-root`.
- Each run needs `--num-frames + 1` frames.

The demo selects one scene and sorts its timestamps. It reserves the first selected frame for
temporal preparation. The remaining frames produce outputs and timing samples.

## Checkpoint-backed real sequence

Replace every `/path/to/...` placeholder before running this command from the tt-metal repository
root:

```sh
python -m models.experimental.bevformer.demo.demo \
  --config /path/to/model.json \
  --checkpoint /path/to/checkpoint.pth \
  --checkpoint-format mmdet3d \
  --dataset-format manifest \
  --dataset /path/to/sequence.json \
  --dataset-root /path/to/dataset \
  --scene-index 0 \
  --start-frame 0 \
  --num-frames 10 \
  --score-threshold 0.25 \
  --output-dir /path/to/demo-output
```

Use `--scene-token` instead of `--scene-index` when the manifest scene token is known.

## Output

The demo writes `detections.json` inside `--output-dir`, containing per-frame boxes, scores, numeric
labels, optional class names, device latency, end-to-end latency, and a run summary. The preparation
invocation and the initial temporal cold-start/warmup invocation are excluded from all reported
metrics. The first output frame position is loaded again as measured frame 0. Manifest mode reloads
the same stored images, while synthetic mode generates fresh deterministic random images. In both
modes, all reported latencies describe the already-compiled temporal execution path.

`device_latency_ms` measures host wall-clock time around the TT model forward, device-side
detection, synchronization, and input deallocation. `end_to_end_latency_ms` additionally includes
image loading, preprocessing, transfer preparation, and host box readback.

The `summary` object contains:

| Field | Meaning |
| --- | --- |
| `variant` | Selected model variant. |
| `architecture` | TT architecture reported by the runtime, for example `blackhole`. |
| `device_card` | TT card reported by device discovery, for example `P100`. |
| `hardware` | Compact architecture and card label, for example `BH P100`. |
| `tt_metal_commit` | Full tt-metal Git commit SHA, or `unknown` when it cannot be resolved. |
| `checkpoint` | Checkpoint path, or `random`. |
| `dataset_format` | `synthetic` or `manifest`. |
| `score_threshold` | Device-side minimum detection score. |
| `preparation_frame` | Unmeasured temporal-state frame token. |
| `num_frames` | Number of measured output frames. |
| `mean_device_latency_ms` | Arithmetic mean of device latency samples. |
| `p50_device_latency_ms` | Median device latency sample. |
| `mean_end_to_end_latency_ms` | Arithmetic mean of end-to-end samples. |
| `p50_end_to_end_latency_ms` | Median end-to-end latency sample. |
| `device_fps` | Estimated steady-state device-path throughput: `1000 / mean_device_latency_ms`. |

The p50 fields use the standard median. `device_fps` remains mean-based, so its value accounts for
every measured frame. It excludes image loading, preprocessing, host box readback, preparation,
and cold-start warmup.

## Visualize detections

The visualization CLI renders manifest-backed demo results without using a Tenstorrent device.
Synthetic results cannot be visualized because they do not reference real camera images or camera
calibration. Run the following command from the tt-metal repository root:

```sh
python -m models.experimental.bevformer.demo.visualize_detections \
  /path/to/demo-output \
  --manifest /path/to/sequence.json \
  --dataset-root /path/to/dataset \
  --score-threshold 0.25
```

The positional input may be either the demo output directory or its `detections.json` file. The
manifest must contain every frame token present in the results. Relative camera paths from the
manifest are resolved below `--dataset-root`. Pass both paths explicitly for portable use. The
visualizer currently expects the standard six BEVFormer camera names shown in the manifest schema.

The command writes two images per result frame beside `detections.json`:

- `frame_<index>_<token>.png`: six-camera mosaic.
- `frame_<index>_<token>_bev.png`: top-down boxes, headings, velocities, and ego position.

Existing images with the same names are overwritten. `--score-threshold` applies only to the
rendered images and does not modify `detections.json`.

| Option | Portable usage | Meaning |
| --- | ---: | --- |
| `--manifest` | Pass explicitly | Manifest used for the demo run. |
| `--dataset-root` | Pass explicitly | Root for relative manifest image paths. |
| `--camera-width` | Default: `800` | Width of each camera tile in pixels. |
| `--score-threshold` | Default: `0.0` | Minimum displayed detection score. |
| `--bev-size` | Default: `1000` | Width and height of the BEV image in pixels. |
| `--bev-range` | Default: `51.2` | Displayed distance in meters around ego. |

## PyTorch usage during setup

Frame inference and detections are produced by TTNN. The demo creates a PyTorch BEVFormer model
only during setup to:

- Load the checkpoint and validate parameter shapes.
- Supply weights for conversion into TTNN parameters.
- Infer backbone and FPN output shapes once.

The PyTorch model is not run for every frame and does not produce the demo detections. Numerical
comparison between PyTorch and TTNN belongs in the PCC tests, not this demo.
