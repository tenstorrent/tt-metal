# VAD sequence demo

This demo runs one unmeasured preparation frame followed by a configurable number of output frames.
The first output position is also executed once as an unmeasured temporal-path warmup without
advancing the retained BEV state. Checkpoint and input sequence are independent command-line
arguments.

## Synthetic smoke run

The synthetic mode uses dummy weights, newly generated random camera images for every frame, a
fixed six-camera rig and an ego driving a gentle curve. It measures the model's latency and FPS; its
outputs have no semantic meaning.

Run the synthetic demo from the tt-metal repository root:

```sh
python -m models.experimental.vadv2.demo.demo \
  --checkpoint random \
  --dataset-format synthetic \
  --num-frames 10 \
  --output-dir generated/demo/vad/synthetic
```

Every op is dispatched from Python on each frame (`execution: eager`).

## Recorded smoke results

This run used the command above:

| Model | BEV size | Encoder layers | p100a Device FPS |
| --- | ---: | ---: | ---: |
| VAD-Tiny | 100 | 3 | 5.38 |

| Card | Commit |
| --- | --- |
| p100a | [`4e154aed67b`](https://github.com/tenstorrent/tt-metal/commit/4e154aed67b96a7f0828ce1c8a6cbb08c06edd6d) |

Compare runs by Device FPS (see [Output](#output)). This comes from one ten-frame run with random
weights and validates end-to-end execution only; it is not a statistically characterized benchmark
result.

## Sequence manifest

The manifest is source-neutral. Matrices are calibrated for each source image's original pixel
size; the loader rescales them with the images and pads normalized images at the bottom. `can_bus`
is the absolute 18-value vector for the frame. The model makes translation and heading relative to
the preceding frame.

The synthetic run above reuses one `lidar2img` matrix per camera for every frame. Real datasets may
provide frame-dependent matrices, which change the visibility mask and the spatial
cross-attention's rebatch, and can grow its capacity, which reallocates the plan. The recorded
synthetic timings therefore may not represent real-sequence performance.

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
      "can_bus": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "command": [0, 0, 1]
    }
  ]
}
```

`command` is optional: the driving command as a one-hot triple in the order turn right, turn left,
go straight. It selects which of the model's three ego plans the output reports as `ego_plan`; all
three are reported in `ego_plans` either way.

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

The local converter must preserve these contracts:

- `camera_order` is exactly the order shown above; the model's camera embeddings are learned in it.
- Every frame contains exactly those camera names.
- Every camera provides one readable image path.
- Every camera provides one numeric `4x4` `lidar2img` matrix, in a lidar frame with x to the right,
  y forward and z up.
- Matrices target each source image's original dimensions.
- `can_bus` contains 18 values and stores absolute frame state: translation in `[0:3]`, rotation
  quaternion in `[3:7]`, heading in radians in `[16]` and in degrees in `[17]`.
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
python -m models.experimental.vadv2.demo.demo \
  --checkpoint /path/to/checkpoint.pth \
  --dataset-format manifest \
  --dataset /path/to/sequence.json \
  --dataset-root /path/to/dataset \
  --scene-index 0 \
  --start-frame 0 \
  --num-frames 10 \
  --output-dir /path/to/demo-output
```

Use `--scene-token` instead of `--scene-index` when the manifest scene token is known.

The camera images are scaled by 0.4, padded to 640x384 and normalized as the released VAD-Tiny
weights were trained: BGR, with the mean `(103.53, 116.28, 123.675)` subtracted and no division by
a standard deviation. This differs from the RGB mean and standard deviation in the published
VAD-Tiny config; images normalized that way yield near-random detections.

## Output

The demo writes `detections.json` inside `--output-dir`, containing per-frame detections, map
elements, ego plans, device latency, end-to-end latency, and a run summary. The preparation
invocation and the temporal warmup invocation are excluded from all reported metrics.

Each frame records:

| Field | Meaning |
| --- | --- |
| `token`, `frame_index` | The frame's manifest token and its position among the output frames. |
| `device_latency_ms` | The frame's preparation, forward and on-device decoding, until the device finishes. |
| `end_to_end_latency_ms` | Adds the frame's host work: image loading, preprocessing, the images' upload and the results' readback. |
| `command` | The manifest's driving command, or `null`. |
| `ego_plan` | The ego plan for `command`, or `null` without one. |
| `ego_plans` | The ego plans for `turn_right`, `turn_left` and `go_straight`: six future waypoints each, in meters. |
| `detections` | `box` (`cx, cy, cz, dx, dy, dz, yaw, vx, vy` in the lidar frame), `score`, `label`, `class_name`, and six predicted `trajectories` of six waypoints each. |
| `map_elements` | `points` (a 20-point polyline), `score`, `label`, `class_name` (`divider`, `ped_crossing`, `boundary`). |

The `summary` object contains:

| Field | Meaning |
| --- | --- |
| `model` | `VAD-Tiny`. |
| `seed` | Random seed for synthetic inputs and random checkpoint weights. |
| `hardware` | TT architecture and card, for example `blackhole P100`. |
| `tt_metal_commit` | Full tt-metal Git commit SHA, or `unknown` when it cannot be resolved. |
| `checkpoint` | Checkpoint path, or `random`. |
| `dataset_format` | `synthetic` or `manifest`. |
| `preparation_frame` | Unmeasured temporal-state frame token. |
| `num_frames` | Number of measured output frames. |
| `execution` | `eager`. |
| `mean_device_latency_ms`, `p50_device_latency_ms` | Mean and median of the frames' `device_latency_ms`. |
| `mean_end_to_end_latency_ms`, `p50_end_to_end_latency_ms` | Mean and median of the frames' `end_to_end_latency_ms`. |
| `device_fps` | `1000 / mean_device_latency_ms`. |
| `end_to_end_fps` | `1000 / mean_end_to_end_latency_ms`. |

## Visualize results

The visualization CLI renders manifest-backed demo results without using a Tenstorrent device. Run
the following command from the tt-metal repository root:

```sh
python -m models.experimental.vadv2.demo.visualize \
  --results /path/to/demo-output/detections.json \
  --manifest /path/to/sequence.json \
  --dataset-root /path/to/dataset \
  --output-dir /path/to/demo-output/png
```

It writes `frame_<index>.png` per result frame: the six camera images above a bird's-eye view in
the lidar frame with the map elements, the detections and their trajectories, and the ego plan.
Without `--manifest` only the bird's-eye view is drawn, which is all synthetic results have.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--score-threshold` | `0.3` | Minimum displayed detection score. |
| `--map-score-threshold` | `0.3` | Minimum displayed map element score. |

The thresholds apply only to the rendered images and do not modify `detections.json`.

## PyTorch usage during setup

Frame inference and its outputs are produced by TTNN. The demo creates a PyTorch VAD model only
during setup to load the checkpoint and supply weights for conversion into TTNN parameters. The
PyTorch model is not run for any frame and does not produce the demo outputs. Numerical comparison
between PyTorch and TTNN belongs in the PCC tests, not this demo.
