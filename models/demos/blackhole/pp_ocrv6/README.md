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
| `tt/ttnn_rec.py` | Recognizer: LCNetV4 backbone, LightSVTR encoder (heads padded 15 -> 32), CTC head |
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

Unoptimized: bf16, HiFi4, activations in DRAM, no trace. Includes host <-> device transfers.

| Stage | Time |
|-------|-----:|
| Detector, 640x640 | 16.9 ms |
| Recognizer, 1 line | 8.3 ms |
| Recognizer, batch 8 | 24.1 ms (3.0 ms per line) |
| Full pipeline, 1 line (host pre/post included) | ~33 ms |
| Same pipeline, torch on 24-core CPU | ~77 ms |

Not yet inside the ~16 ms per request budget; see next steps.

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
- [ ] On-device argmax + confidence for the recognizer (return 40 indices, not ~1.5 MB of probabilities per line)
- [ ] Recognizer batch buckets 2 and 4 (two lines are padded to 8 today)
- [ ] Trace capture + two command queues for detector and recognizer
- [ ] L1-sharded activations instead of DRAM
- [ ] bfp8 weights, lower math fidelity, approximate GELU, gated on the accuracy tests and the torch agreement rate

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
