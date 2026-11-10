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

bf16, HiFi4, activations in DRAM. Traced with two command queues; packed input copy;
the recognizer decodes on device (argmax index + confidence). Per call, including
host <-> device transfers. "Kernel" is the Tracy kernel sum from the tt-opt baselines.

| Stage | Untraced, first bring-up | Traced | + packed input, tiled decode | + stem concat split, `where` decode | + neck split, input fold | Kernel now |
|-------|------:|------:|------:|------:|------:|------:|
| Detector, 640x640 | 16.9 ms | 20.3 ms | 9.3 ms | 8.0 ms | 7.3 ms | 4.91 ms (was 7.02) |
| Recognizer, 1 line | 8.3 ms | 3.1 ms | 2.6 ms | 2.4 ms | 2.5 ms | |
| Recognizer, batch 2 | - | 4.4 ms | 3.5 ms | 3.2 ms | 3.3 ms | |
| Recognizer, batch 4 | - | 6.8 ms | 4.6 ms | 3.8 ms | 4.0 ms | |
| Recognizer, batch 8 | 24.1 ms | 12.5 ms | 8.3 ms | 6.9 ms | 6.9 ms (0.9 per line) | 4.82 ms (was 6.55) |
| Full pipeline, 1 line (host pre/post included) | ~33 ms | ~26 ms | ~14 ms | ~13 ms | **~12 ms** | |
| Same pipeline, torch on 24-core CPU | ~77 ms | | | | | |

Detector at batch 2 / 4 (640x640, per image): 8.8 / 7.8 ms per call with transfers, 6.5 / 6.1 ms
replay, against 9.3 / 7.3 ms at batch 1. Batching helps the detector far less than the recognizer.

What moved the numbers:
- **Packed input copy.** NHWC row-major with 3 channels makes every page one pixel (6 bytes);
  the 640x640 copy took 9.7 ms. The host now sends `[1, 1, B*H, W*3]` (one image row per page,
  0.16 ms) and a reshape inside the trace restores NHWC (0.43 ms). Casting to bf16 in torch
  before `from_torch` cut host conversion from 2.7 to 0.25 ms.
- **Tiled decode.** max, compare, then min over the matching 1-based class indices (fp32, exact)
  instead of softmax + row-major argmax over 18,710 classes: 7.53 -> 6.59 ms traced at batch 8,
  same indices. Confidence is 1 / sum(exp(logit - max)). One `where` instead of
  multiply/rsub/add saves another 0.4 ms.
- **Stem concat split.** stem3 ran on a row-major DRAM concat of the pool and stem2b
  outputs (0.90 ms in the detector, 0.37 ms in the recognizer, plus a 0.64 ms reshard in the
  recognizer). Its weights are split by input channel instead:
  conv(cat(p, a)) = conv_p(p) + conv_a(a). About -1.1 ms kernel time in each model.
- **Neck concat split.** The DB head's conv_down ran on a concat of the four neck outputs; split
  the same way into four convs and a sum: detector -0.25 ms.
- **Input fold (space-to-depth).** The host folds each 2x2 pixel block into 12 channels and stem1
  becomes the equivalent 2x2 stride-1 conv (padded one block top/left; exact). The first conv's
  3-channel input reshard (0.72 ms) is gone and the unpack reshape drops 0.37 -> 0.14 ms:
  detector 5.69 -> 4.91 ms, recognizer batch 8 5.04 -> 4.82 ms. The model still accepts plain
  NHWC 3-channel input.

With YOLO11-small (3.05 ms per frame, measured standalone) a one-line request is ~15 ms,
just inside the ~16 ms budget; the two models have not run together on one device yet.

Smaller detector inputs (measurement only; the default stays 640x640), batch 1, traced:

| Detector input | Full call | Device replay |
|---|---|---|
| 640x640 | 7.0 ms | 5.2 ms |
| 480x480 | 5.6 ms | 4.4 ms |
| 320x320 | 3.5 ms | 2.8 ms |
| 256x256 | 3.0 ms | 2.3 ms |

A tight crop around the number reads at every size down to 256x256. Wide container photos
lose their smaller text at 320 and below, so the size depends on how tight the YOLO crop is
and is chosen on customer images.

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
Full list with estimates: [optimization_strategies.md](optimization_strategies.md).
- [ ] tt-opt detector run: harness, gate and baseline done (6.7 ms kernel); experiments blocked while outbound internet is closed (security incident)
- [ ] Review tt-opt recognizer keeps (branch `tt-opt/pp_ocrv6_rec/decode-20261107-181455`, 7.46 -> 7.13 ms kernel at batch 8). Batch-8 shape guards only; port the general ones (SiLU fused into fc1, HiFi2) by hand
- [ ] Keep conv outputs sharded in L1 between consecutive convs (DRAM -> L1 reshards are still ~1.2 ms in the detector)
- [ ] Smaller detector input on YOLO crops: measured (320x320 saves ~3.5 ms per crop); pick the size on customer crops
- [ ] Normalization folded into the first detector conv
- [ ] Recognizer width bucket 160 for short lines
- [ ] L1-sharded activations for the detector convs (manual; tt-opt has no conv2d sweeps)
- [ ] bfp8 weights, lower math fidelity, approximate GELU, gated on the accuracy tests and the torch agreement rate
- [x] Pipelining across requests: `TtnnBackend.lock` with worker threads (see `gate_ocr`)
- [x] Several crops per call: `PPOCRv6Pipeline.read_many` batches the detector (`det_batches`) and the recognizer across crops
- [ ] Detector batch 4 reads wrong once an L1 buffer is allocated after its first run (no error raised); report with a repro

### 2. YOLO11-small
- [ ] Confirm with the customer how the 4K frame becomes the 640x640 input (full-frame resize, fixed lane region, larger input or tiles)
- [x] Bring up on device (`models/demos/yolov11s`, from `sdawle/yolov8_bh`): PCC 0.99991, 3.05 ms per frame (328 fps) at 640x640
- [ ] One device for both models: YOLO -> crop from the 4K frame -> det -> rec, all traced; ms per frame and per crop
- [ ] Fine-tuned customer weights (their classes) once available

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
- [x] tt-opt recognizer run, 20 experiments, 11 keeps (on the tool's branch, not merged)
- [x] Packed host-to-device input copy and bf16 host cast (pipeline ~26 -> ~17 ms)
- [x] Tiled on-device decode instead of softmax + argmax (pipeline ~14 ms, recognizer batch 8 8.3 ms)
- [x] Tracy op breakdown of both models (data movement dominates: concats, reshards, untilizes)
- [x] Stem concat split into two convs; `where`-based decode (pipeline ~13 ms, kernel det 5.94 / rec b8 5.05 ms)
- [x] Neck concat split; space-to-depth input fold (pipeline ~12 ms, kernel det 4.91 / rec b8 4.82 ms)
- [x] Detector input size sweep 640 / 480 / 320 / 256 (measurement only)
- [x] YOLO11-small on device: PCC 0.99991, 3.05 ms per frame
