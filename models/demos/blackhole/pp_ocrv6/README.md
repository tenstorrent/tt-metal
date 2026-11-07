# PP-OCRv6 small on Blackhole (P150)

[PP-OCRv6_small](https://arxiv.org/abs/2606.13108) text detector (2.48M params) and
text recognizer (5.2M params, 50 languages, 18,710 characters) on a single Blackhole
die with ttnn, plus an end-to-end OCR pipeline: crop -> detector -> text boxes ->
recognizer -> text.

Target use: reading trailer / unit numbers from gate-camera crops for a yard
management system (customer: Blackrock Holdings). In the customer's design,
YOLO11-small runs first on the camera frame and hands PP-OCRv6 the text region.

Weights (Apache 2.0):
[PP-OCRv6_small_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_safetensors),
[PP-OCRv6_small_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_safetensors).

## Supported HW

| Device | Mesh |
|--------|------|
| **P150** | 1x1 |

## Customer requirements

- Detector on the YOLO text-region crop, recognizer on each text line; the number is a
  short trailer or unit ID.
- Load: 2-3 gates x ~4 cameras, 2-5 frames/s per view while a truck is unread, all on
  **one chip**. Worst case ~60 requests/s, i.e. ~16 ms per request for YOLO + det + rec.
- Latency: < 500 ms from first readable frame to the YMS event, goal 150-400 ms.
- Numbers to report: ms per frame for YOLO11-small, ms per crop for PP-OCRv6 det + rec.
- Fully local, no cloud runtime.

## Layout

| Path | Contents |
|------|----------|
| `reference/det.py`, `reference/rec.py` | Standalone PyTorch models; parameter names match the safetensors |
| `tt/common.py` | Shared ttnn blocks: conv/linear with folded BatchNorm, squeeze-excitation, LCNetV4 layer, stem, backbone |
| `tt/ttnn_det.py` | Detector: LCNetV4 backbone, RepLKFPN neck, DB head (transposed convs as matmuls) |
| `tt/ttnn_rec.py` | Recognizer: LCNetV4 backbone, LightSVTR encoder (heads padded 15 -> 32), CTC head, optional on-device argmax |
| `tt/runner.py` | Trace capture and replay with two command queues |
| `demo/pipeline.py` | End-to-end pipeline with torch and ttnn backends and per-stage timings |
| `demo/demo.py` | Runs one image, prints text, saves a result picture |
| `tests/pcc/` | Accuracy tests: detector, recognizer, pipeline |
| `tests/perf/` | Detector and recognizer timings |

## How to run

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
# optional: local weights instead of the Hugging Face cache
# export PPOCRV6_WEIGHTS_DIR=/path/containing/PP-OCRv6_small_{det,rec}_safetensors

pytest models/demos/blackhole/pp_ocrv6/tests/pcc/
pytest -s models/demos/blackhole/pp_ocrv6/tests/perf/
python models/demos/blackhole/pp_ocrv6/demo/demo.py --image crop.jpg --out result.jpg
```

`onnxruntime` is only needed to compare the reference against the official ONNX files.

## Measured accuracy

- **Reference vs official ONNX:** PCC 1.0 (max abs diff 4e-6 detector, 7e-5 recognizer).
- **Detector on device:** PCC 0.9996 on a text image; 0.015% of pixels flip at the 0.2
  box threshold.
- **Recognizer on device:** 8 synthetic trailer-number crops read exactly as torch,
  PCC >= 0.99995, 100% per-step argmax match at batch 1 and 8.
- **Pipeline:** one line, two lines and a 6-degree rotated plate read the expected text
  and match torch.
- **Real photos** (Wikimedia: car plates, highway truck, shipping containers): plates and
  large unit numbers read correctly and match torch (`AE 6133 CT`, `CA 388-632`,
  `MRKU 3158293`, cab number `364547` from a crop). Device and torch agree on every
  read above ~0.85 confidence; they differ only on low-confidence text both get wrong.

## Measured performance

bf16, HiFi4, activations in DRAM. Traced with two command queues; the recognizer
returns argmax indices and confidences instead of probabilities. Per call, including
host <-> device transfers.

| Stage | Untraced | Traced |
|-------|---------:|-------:|
| Detector, 640x640 | 16.9 ms | 20.3 ms (16.6 device, 3.0 input conversion, 1.0 readback, 0.3 unshuffle) |
| Recognizer, 1 line | 8.3 ms | 3.1 ms |
| Recognizer, batch 2 | - | 4.4 ms (2.2 per line) |
| Recognizer, batch 4 | - | 6.8 ms (1.7 per line) |
| Recognizer, batch 8 | 24.1 ms (3.0 per line) | 12.5 ms (1.6 per line) |
| Full pipeline, 1 line (host pre/post included) | ~33 ms | ~26 ms |
| Same pipeline, torch on 24-core CPU | ~77 ms | |

The untraced detector figure is device time. The detector is kernel-bound, so tracing helps
little there; the recognizer was dispatch-bound and gets ~2.7x. Still above the ~16 ms
per request budget; the detector kernels are the next target (Tracy, then tt-opt).

Traced mode captures one trace per shape (detector 640x640, recognizer batch 1/2/4/8)
when `TtnnBackend` is created. All shapes run once untraced before any capture, so
prepared conv weights never land in another trace's scratch memory; otherwise replays
corrupt them and the device hangs.

## Known limitations

- **Small text in a full frame** is missed: a 1280-pixel frame letterboxed to 640 leaves
  small numbers a few pixels tall. It reads once cropped, which is the YOLO stage's job.
- **Vertical text** (container corner codes, vertical trailer numbers) is not read yet.
  The test crops were chosen by eye, so the cause is not established; stacked upright
  letters would also not read as one line.
- **Low-confidence text:** where the model is unsure, bf16 rounding can tip the device
  result differently from torch.
- Validation is on synthetic and public photos only, not customer gate images.

## Next steps

Updated at each milestone.

### 1. Performance (to ~16 ms per request on one chip)
- [ ] Tracy profile of detector and recognizer: per-op kernel time, find the top ops
- [ ] tt-optimization-loop (`tt-opt`), detector and recognizer as separate runs:
  - profile_command: one traced forward under the device profiler
  - gate_command: `tests/pcc` for that model (PCC + exact text)
  - latency_command: traced replay time (it optimizes kernel time; op-to-op gaps are already gone with trace)
  - Levers it can try: sharding/layout, bfp8 / LoFi, matmul and softmax configs, fusion. No conv2d config sweeps, so conv sharding below is manual
- [ ] L1-sharded activations for the detector convs (manual; detector is kernel-bound at 16.6 ms)
- [ ] bfp8 weights, lower math fidelity, approximate GELU, gated on the accuracy tests and the torch agreement rate
- [ ] Detector input: build the bf16 NHWC tensor in preprocessing (saves ~3 ms of host conversion)
- [ ] Pipelining across requests (overlap det of frame N+1 with rec of frame N) for the throughput target

### 2. YOLO11-small
- [ ] Confirm scope with Dalar / Michael
- [ ] Bring up on device; report ms per frame at 640x640

### 3. Real-world accuracy
- [ ] Customer sample gate images (request via Dalar / Michael)
- [ ] Vertical text: redo the test with careful crops; try both 90-degree rotations on tall boxes and keep the more confident read
- [ ] Decide handling for stacked upright letters (per-character split or another camera angle)
- [ ] Track device vs torch text agreement during optimization so precision changes cannot regress it
- [ ] Discuss recognizer fine-tuning or a reduced character set for trailer-number formats

### 4. Load test and customer numbers
- [ ] Simulate 2-3 gates / 8-12 cameras at 2-5 frames/s per view; measure throughput and p50 / p99 latency against 500 ms
- [ ] Report ms per frame (YOLO) and ms per crop (PP-OCRv6 det + rec) on one chip

### 5. Non-code follow-ups
- [ ] Confirm the customer accepts a ttnn implementation (their email says "compiled through TT-Forge")
- [ ] Open a pull request

### Done
- [x] PyTorch reference for detector and recognizer, verified against ONNX
- [x] Detector and recognizer on device, verified against torch
- [x] End-to-end pipeline, demo and tests
- [x] Real-photo test; fixed conv weights being reused across batch sizes
- [x] On-device argmax + confidence for the recognizer
- [x] Recognizer batch buckets 1/2/4/8
- [x] Trace capture + two command queues for detector and recognizer (recognizer ~2.7x faster)
