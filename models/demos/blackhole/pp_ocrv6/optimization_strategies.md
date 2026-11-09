# PP-OCRv6 small: optimization strategies

Where the time goes today and what can cut it, toward the customer budget of
~16 ms per request on one p150a (YOLO11-small + detector + recognizer, ~60 requests/s).

## Where the time goes

Traced pipeline, one text line, 640x640 detector input: **~26 ms** before 1.0 and 3.1 below; **~14 ms** after them (done).

| Part | Time |
|------|-----:|
| Detector: input copy to device + trace replay | 16.6 ms (kernel time only 6.7 ms, replay alone 8.4 ms, per the tt-opt baseline; the rest is likely the input copy, see 1.0) |
| Detector input conversion on host (torch -> bf16 NHWC) | 3.0 ms |
| Detector readback + sub-pixel unshuffle | 1.3 ms |
| DB box finding + crop warp + recognizer preprocessing | 1.6 ms |
| Recognizer, batch 1 | 3.1 ms |
| Letterbox + normalize | 0.9 ms |

The detector's kernels take only 6.7 ms (tt-opt baseline); most of its 16.6 ms looks like
the host-to-device input copy (1.0 below). The recognizer was dispatch-bound and is
already ~2.7x faster with trace. YOLO11-small is not counted yet.

Gains below are estimates until measured; a Tracy profile ranks the on-device items.

## 1. Pipeline (no model change)

| # | Strategy | Expected gain | Notes |
|---|----------|---------------|-------|
| 1.0 | **Done: packed input copy** (9.7 -> 0.6 ms with the on-device reshape). Was: The input goes over row-major `[1, 640, 640, 3]`, so each page is one pixel row of 3 bf16 values (6 bytes) and the copy is 409,600 tiny pages. Send it as `[1, 640, 1920]` (or channel-padded / uint8) and reshape on device | likely ~8 ms | Measure copy alone vs replay alone first |
| 1.1 | **Smaller detector input.** After YOLO the detector sees a trailer or plate crop, not a full frame. Run it at 320x320 (or a few size buckets, e.g. 320 / 480 / 640, or non-square like 640x320) | ~16 -> ~5 ms at 320x320 (cost ~ pixels) | Biggest single win. Check box recall on real crops; one trace per size bucket |
| 1.2 | **Normalization on device.** Fold `(x/255 - mean) / std` into the first conv's weights and bias; send the raw uint8 / bf16 image | ~3 ms host | Border pixels differ slightly (padding pads raw zeros, not normalized zeros); gate with the PCC tests |
| 1.3 | **Recognizer width buckets.** Short codes padded to 320 wide waste half the work; add a 160-wide bucket | up to ~2x for short lines | One trace per (batch, width); CTC length 20 instead of 40 |
| 1.4 | **Overlap requests.** Detector of frame N+1 on device while host does DB post / crops / CTC for frame N; two command queues are already in place | throughput, not single-request latency | Needed for 60 requests/s |
| 1.5 | **Faster host post-processing.** Vectorize / move DB box finding and the crop warp off the critical path (thread pool) | ~1 ms | Small but on every frame |

## 2. Detector on device (manual; tt-opt has no conv2d config sweeps)

Tracy (current code): detector 5.94 ms kernel, recognizer batch 8 5.05 ms. Data movement dominates:
concats, DRAM -> L1 reshards before convs, untilizes; convs + matmuls are ~30%.
Done: stem concat split (conv(cat(p, a)) = conv_p(p) + conv_a(a)), -1.1 ms in each model.
Tried, no gain: padding the 3-channel input to 4 / 8 / 16 channels; reshape output in L1 breaks a
downstream fused-ReLU matmul. Next: 2x2 space-to-depth fold of the input on the host.

| # | Strategy | Notes |
|---|----------|-------|
| 2.1 | **L1-sharded activations** instead of DRAM, height-sharded for the early high-resolution layers (stem at 320x320, stage 1 at 160x160) | Largest expected on-device win; tune `act_block_h`, enable activation / weight double buffering |
| 2.2 | **Lower precision:** bfp8 weights, bfp8 activations where PCC allows, LoFi / HiFi2 instead of HiFi4, fp32 dest accumulation off | Gate on detector PCC and pipeline text match |
| 2.3 | **Fuse activations into convs** (ReLU / Hardswish via the conv config) instead of separate eltwise ops | Fewer ops and DRAM round trips |
| 2.4 | **Squeeze-excitation blocks:** many tiny ops (global pool, two 1x1, hardsigmoid, multiply); keep them in L1, fuse the gate math | Small ops add up across LCNetV4 layers |
| 2.5 | **Neck:** upsample + add + concat in DRAM; shard them and avoid layout conversions between them | |
| 2.6 | **Depthwise 7x7 convs** in RepLKFPN: check their kernel time in Tracy; depthwise convs are often slow relative to their FLOPs | |
| 2.7 | **Output:** keep the DB head output on device and threshold there, read back only the binary mask (or uint8) | Cuts readback; box finding stays on host |

## 3. Recognizer on device (partly what tt-opt tries)

| # | Strategy | Notes |
|---|----------|-------|
| 3.1 | **Done: tiled decode** (7.53 -> 6.59 ms traced at batch 8, same indices). Was: The tt-opt baseline (batch 8, 7.46 ms kernel time) shows `ArgMax` alone at 1.89 ms, ~25%, the costliest op, plus the untilize before it and the softmax. Argmax works on logits, so the softmax can go; confidence = 1 / sum(exp(logit - max)) needs one reduction. Try a tiled max + compare instead of the row-major argmax | Clearest recognizer win |
| 3.2 | **Head matmul 120 -> 18,710:** bfp8 weights, matmul program config sweep | tt-opt matmul config lever |
| 3.3 | **Attention head packing.** 8 heads of width 15 are zero-padded to 32 each (~half the attention work is padding); pack to 128 total or run attention in fewer, wider tiles | |
| 3.4 | **LayerNorm / SiLU / MLP:** sharded, fused where ttnn allows | tt-opt layout / fusion levers |
| 3.5 | **Backbone convs:** same as detector 2.1 - 2.3, at the 48x320 input | Manual |

## 4. Tools

- **Tracy** (`python -m tracy -r -p ...`): per-op kernel time for detector and recognizer; decides
  which of sections 2 and 3 matter.
- **tt-optimization-loop (`tt-opt`)**: one measured change at a time, kept only if the PCC gate
  passes and the traced window drops by at least 5 us. Good at matmul / softmax / layout / dtype
  levers (section 3); not at conv configs (section 2). Runs on the isolated worktree
  `tt-metal-ppocr-opt` (branch `tvardhineni/ppocrv6_small_opt`); its keeps are commits on its own
  `tt-opt/pp_ocrv6_rec/<run>` branch, to cherry-pick after review.

## Suggested order

1. 1.1 smaller detector input, 1.2 normalization on device, 3.1 drop the softmax: simplest, together
   roughly ~26 ms -> ~10-12 ms per crop, leaving room for YOLO.
2. Tracy profile of both models, then 2.1 / 2.2 for the detector by hand.
3. Merge reviewed tt-opt keeps for the recognizer.
4. 1.3 and 1.4 for the throughput target, then the load test.
