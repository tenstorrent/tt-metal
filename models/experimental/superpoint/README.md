# SuperPoint

SuperPoint keypoint-detection and descriptor-extraction inference on a single Tenstorrent Blackhole (p150a/p150b) accelerator, implemented with tt-nn.

Reference model: `magic-leap-community/superpoint` on Hugging Face.
Input: single-image, 480×640, batch size 1.

## How to Run

To run the model, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer to the [installation and build guide](https://docs.tenstorrent.com/tt-metalium/latest/get_started/get_started.html#install-and-build).

Run the benchmark and accuracy test (PCC + keypoint-set F1):

```sh
pytest -s -q models/experimental/superpoint/tests/test_superpoint.py::test_superpoint_benchmark
```

Set `SP_TRACE_NMS=1` to close the NMS loop on device via the fused `ttnn.experimental.sp_eq_mul_mask` kernel (recommended — the fast e2e path):

```sh
SP_TRACE_NMS=1 pytest -s -q models/experimental/superpoint/tests/test_superpoint.py::test_superpoint_benchmark
```

## Sample Output

Top-500 tt-nn keypoints overlaid on the resized (480×640) sample image.
Circle radius is proportional to keypoint score.

![tt-nn SuperPoint keypoints on the sample image](media/sample.png)

## Supported Hardware

- Blackhole (p150a / p150b)

## Performance (Blackhole p150b, 480×640, batch 1, natural image)

### Accuracy vs Hugging Face reference (fp32 CPU torch)

| Tensor | PCC |
|---|---:|
| Pre-NMS score map | 0.9971 |
| Descriptor map (pre grid-sample, post L2-norm) | 0.9991 |

Keypoint-set evaluation (top-500, matching radius 2 px, real photograph):

| Metric | tt-nn vs reference |
|---|---:|
| Recall | 98.80% |
| Precision | 98.80% |
| F1 | 98.80% |

### Throughput

**Forward-only trace (`SP_TRACE_NMS=0`) — host-side NMS**

| Metric | Natural image | Paper (Titan X, 2018) |
|---|---:|---:|
| Device forward (input pre-resident) | 355.37 fps | 90 fps |
| Traced forward incl. per-frame H2D | 73.55 fps | — |
| Full e2e (incl. host NMS) | 17.00 fps | not reported |

**Forward + device NMS (`SP_TRACE_NMS=1`) — fused `sp_eq_mul_mask` kernel**

| Metric | Natural image | Paper (Titan X, 2018) |
|---|---:|---:|
| Device forward + device NMS (pre-resident) | 85.60 fps | 90 fps |
| Traced forward+NMS incl. per-frame H2D | 44.56 fps | — |
| Full e2e (no host NMS) | **40.73 fps** | not reported |

Moving NMS on-device via the fused `sp_eq_mul_mask` C++ Tensix kernel pushes end-to-end throughput from 17.0 → 40.73 fps (+140%) while preserving PCC (0.9971) and F1 (98.80%).

## Implementation Notes

- Uses `ttnn.trace` to capture the full device forward pass; Python dispatch overhead dropped from 97% of wall-clock to negligible after tracing.
- Two command queues: H2D on CQ1 overlapped with the traced compute on CQ0.
- The fused `ttnn.experimental.sp_eq_mul_mask` kernel (C++ Tensix, ~450 LoC) fuses `eq + multiply` into a single JIT-compiled program, replacing a 36 ms host NMS with ~7 ms of on-device trace work.
- Weights in bfloat16, HiFi2 math, fp32 accumulator — the minimum precision that keeps score-map PCC ≥ 0.99.
- D2H dispatch cost (~6–9 ms per `ttnn.to_torch` call, payload-independent) is the current bottleneck; a batched D2H API or a 1-channel NMS kernel would reduce it further.
