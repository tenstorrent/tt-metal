# YOLO26 (yolo26n) bring-up — postmortem

Source: tenstorrent/tt-metal PR #37882 "[DRAFT] YOLOv26 model bringup" (author tvardhineniTT), single commit
`d86a0f7c51b5a91545838ed5c3109b811a365849` (2026-02-13). The PR was auto-closed for inactivity and never merged; its
branch `tvardhineni/yolov26_bringup` is deleted, so the code was fetched from `refs/pull/37882/head`. It runs here on
`/home/ttuser/sdawle/tt-metal` at `985c23b1bdb` (main, 2026-10-22), ~8 months newer than the PR. Hardware: QB2,
4× Blackhole p300c; all runs below are on ONE chip (chip 0) unless stated.

Rules:

- record every defect, fix and refuted hypothesis in the same session;
- corrections are appended, never silently edited;
- numbers carry their method and units;
- MEASURED / READ / INFERRED are kept apart.

PCC = Pearson correlation of the TTNN output vs the Ultralytics PyTorch model (fp32, CPU), 640×640 input, batch 1,
yolo26n weights `yolo26n.pt` (ultralytics 8.4.184), computed with `models.common.utility_functions.comp_pcc`.

## Index

| ID | Symptom / question | Date | Status |
|---|---|---|---|
| YOLO26-0001 | Do we have everything from the draft PR? | 2026-10-22 | CLOSED: 29/29 files byte-identical to the PR head; PR touches nothing outside `models/experimental/yolo26/`; no other YOLO26 PR in tt-metal |
| YOLO26-0002 | Which variant are we running — n or s? | 2026-10-22 | CLOSED: n only. All 14 test call sites load `yolo26n.pt` and the channel counts are hardcoded for n; the `variant` argument is ignored. `yolo26s.pt` is downloaded but unsupported |
| YOLO26-0003 | Installing `ultralytics` would upgrade torch 2.11.0+cpu → 2.12.1 (+ CUDA/triton); `pip` on PATH points at a different venv | 2026-10-22 | WORKED AROUND: `uv pip install ultralytics "torch==2.11.0+cpu"`; torch/torchvision/numpy unchanged |
| YOLO26-0004 | First PCC run: "Sysmem mapped at unexpected NOC address" | 2026-10-22 | ROOT-CAUSED & RESOLVED: container `tt-bh-qb2-qwen3` held all 4 chips; stopped (user-approved). Container is still stopped |
| YOLO26-0005 | After stopping the container: "Timed out while waiting for active ethernet core 29-25" | 2026-10-22 | FIXED: `tt-smi -r` board reset |
| YOLO26-0006 | The run opened all 4 chips even though the model uses one | 2026-10-22 | FIXED: `TT_VISIBLE_DEVICES=0` + `TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto` |
| YOLO26-0007 | `TtYOLO26` (used by demo, runner, perf tests) does not match the real yolo26n graph | 2026-10-22 | OPEN: `test_backbone_only` fails with a weight-shape error; the demo/perf path is untested for correctness. **FIXED → YOLO26-0012** |
| YOLO26-0008 | `test_pcc.py` passes but does not exercise `TtYOLO26`; head PCC is not asserted in `test_full_model` / `test_detailed_e2e` | 2026-10-22 | OPEN: no end-to-end assert on the class that ships. **FIXED → YOLO26-0012** (`test_tt_yolo26.py`) |
| YOLO26-0009 | Low cls PCC: box 0.9719, cls 0.9353 (threshold 0.93; docstring expects > 0.99) | 2026-10-22 | ROOT-CAUSED (mostly), FIX NOT APPLIED: conv precision (LoFi, no fp32 acc, bfp8 weights). bf16 + HiFi4 + fp32 acc → box 0.9973, cls 0.9754. Residual cls gap to 0.99 still OPEN. **CORRECTED → YOLO26-0011**: real root cause is BN eps |
| YOLO26-0010 | README references things that don't exist | 2026-10-22 | OPEN (docs) |
| YOLO26-0011 | Where is the residual cls PCC loss (0.975 after HiFi4)? | 2026-10-22 | ROOT-CAUSED & FIXED: BN folded with eps=1e-5, model uses eps=1e-3 (46/114 BN layers off by >1%, worst ×1.90). Fix → TtYOLO26 box 0.9997, cls 0.9986 (n) |
| YOLO26-0012 | Fix `TtYOLO26` graph (0007) and add an end-to-end test (0008) | 2026-10-22 | FIXED: model.6/8 → TtC3k2, model.22 → TtC3k2PSA; new `tests/pcc/test_tt_yolo26.py` (threshold 0.99, n + s) |
| YOLO26-0013 | Support yolo26s | 2026-10-22 | FIXED: width multiplier (s = 2× n), PSA heads = c // 64, head c2/c3 as Ultralytics. yolo26s box 0.9996, cls 0.9989 |
| YOLO26-0014 | yolo26s: "circular buffers grow to 1930688 B, beyond max L1 1572864 B" at model.7 | 2026-10-22 | FIXED: `BlockShardedStrategyConfiguration(reshard_if_not_optimal=True)` |
| YOLO26-0015 | Trace + 2CQ perf test skipped on Blackhole and broken for TtYOLO26 | 2026-10-22 | FIXED: runs on BH; n 11.64 ms / 85.9 FPS, s 11.77 ms / 85.0 FPS e2e (input-transfer bound); model-only n 5.36 ms, s 6.21 ms |
| YOLO26-0016 | Demo produced wrong/no boxes (wrong decode, BGR input, dict outputs) | 2026-10-22 | FIXED: NMS-free one2one decode; 16/16 detections match Ultralytics end2end on bus.jpg + zidane.jpg, n and s |
| YOLO26-0017 | Other perf tests (`test_yolo26_device_perf.py`, `test_e2e_performant.py`, `test_perf.py`) | 2026-10-22 | OPEN: not updated for dict outputs / BH; not run |
| YOLO26-0018 | E2E bound by input upload (9.63 ms of 11.6 ms); move padding + normalization to device | 2026-10-22 | FIXED: raw uint8 image as [B, 1, H, W·3], typecast/reshape on device, 1/255 folded into stem. E2E n 6.16 ms / 162.3 FPS, s 7.00 ms / 142.9 FPS |
| YOLO26-0019 | Items observed this session but not yet recorded (L1_SMALL OOM, tilize warning spam, API additions) | 2026-10-22 | RECORDED (appended): see entry; OOM worked around, warning not addressed |
| YOLO26-0020 | Which findings are ttnn / tt-metal behaviour rather than model bugs? | 2026-10-22 | OPEN: summary table in entry; nothing reported upstream yet |
| YOLO26-0021 | dp=4 on the QB2 (1×4 mesh): does it scale, and what is the 10-run e2e FPS? | 2026-10-23 | DONE: n 643.7 FPS, s 566.9 FPS (10-run mean, trace+2CQ e2e) = 3.97× single chip; intermittent hang at trace capture logged, not root-caused |

## Entries

### YOLO26-0001 — PR completeness check

- MEASURED: GitHub API `pulls/37882/files` lists 29 files; `git hash-object` of each local file equals the PR blob SHA
  (29/29 match, 0 missing, 0 diff). Only local-only file: `SOURCE.txt` (provenance note added by us).
- READ: PR has 1 commit (`d86a0f7c51`); all 29 files are under `models/experimental/yolo26/` (no edits to shared
  tt-metal code). Only comment is the github-actions auto-close notice.
- READ: GitHub issue/PR search for `yolo26` and `yolov26` in tenstorrent/tt-metal returns only #37882; current main has
  no YOLO26 code.

### YOLO26-0002 — variant

- READ: `YOLO("yolo26n.pt")` at all 14 call sites. `TtYOLO26Backbone.__init__` hardcodes n channels
  ("YOLO26n exact channels from weights analysis"), likewise the neck and `test_pcc.py`.
- MEASURED: `yolo26n.pt` = 2,572,280 params (scale n); `yolo26s.pt` = 10,009,784 params (scale s). Both files are in
  the tt-metal root.
- INFERRED: running s needs channel/depth scaling derived from the model yaml; not attempted.

### YOLO26-0003 — environment

- MEASURED: `pip install --dry-run ultralytics` would install torch-2.12.1, torchvision-0.27.1, triton-3.7.1.
- READ: `which pip` → `~/.tenstorrent-venv/bin/pip`; `python_env` has no pip module, it is managed by `uv`.
- Fix: `uv pip install ultralytics "torch==2.11.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
  --index-strategy unsafe-best-match` → adds ultralytics 8.4.184, opencv-python 4.11.0.86, polars, thop only.

### YOLO26-0004 — devices busy

- MEASURED: `/proc/*/fd` holders: chip 0 `main.py`, chip 1 `whisper_server.py`, chip 2/3 `qwen3_tts_server.py`, all in
  container `tt-bh-qb2-qwen3` (`tt-voice-assistant:push-button-v2`, up 4 weeks, restart policy `no`).
- Resolution: `sudo docker stop tt-bh-qb2-qwen3`. To restore: `sudo docker start tt-bh-qb2-qwen3`
  (may need `tt-smi -r` first, see 0005).

### YOLO26-0005 — ethernet core timeout

- MEASURED: `TT_THROW llrt.cpp:684 — Device 0: Timed out while waiting for active ethernet core 29-25 to become active
  again. Try resetting the board.` Heartbeat start == end (0xabcd451f), i.e. ERISC not running.
- INFERRED: ERISC state left behind by the container's (older) tt-metal.
- Fix: `~/.tenstorrent-venv/bin/tt-smi -r`, then the run passed.

### YOLO26-0006 — single-device run

- MEASURED: without restriction UMD logs `Opening local chip ids/PCIe ids: {0, 1, 2, 3}` and warns "Opening subset of
  mmio devices"; compute is on device 0 only.
- MEASURED: `TT_VISIBLE_DEVICES=0` alone → `TT_FATAL tt_cluster.cpp:304: Custom fabric mesh graph descriptor path must
  be specified for CUSTOM cluster type` (1 of 4 chips visible is not a known cluster type).
- Fix: also set `TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`
  (BLACKHOLE, 1×1 mesh). UMD then logs `{0}/[0]`; PCC identical to the 4-chip run (box 0.971947, cls 0.935308).
- Command:

  ```
  cd /home/ttuser/sdawle/tt-metal && source python_env/bin/activate
  export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
  pytest models/experimental/yolo26/tests/pcc/test_pcc.py -v
  ```

- Next (dp=4): drop `TT_VISIBLE_DEVICES`, open all 4 chips in one process (`MeshDevice::create_unit_meshes()` per the
  tt-metal warning); the runner needs a data-parallel path.

### YOLO26-0007 — `TtYOLO26` graph mismatch

- READ (Ultralytics graph for yolo26n, `model.model[i]`):

  | Layer | Real module | `TtYOLO26` builds | `test_pcc.py` builds |
  |---|---|---|---|
  | model.2, model.4 | C3k2 (inner Bottleneck) | TtC2f | TtC2f — equivalent |
  | model.6, model.8 | C3k2 (inner **C3k**) | **TtC2f** | TtC3k2 |
  | model.22 | C3k2 (inner **PSA** Sequential) | **TtC3k2** | TtC3k2PSA |

- MEASURED: `test_backbone_only.py` (uses `TtYOLO26Backbone`) FAILS: `ValueError: Weight shape Shape([32, 64, 1, 1])
  doesn't match expected shape (32, 64, 3, 3)`.
- Impact: `demo/demo.py`, `runner/yolo26_test_infra.py`, `tests/perf/test_yolo26_device_perf.py` all build `TtYOLO26`.
- Fix (proposed): build the backbone and neck from the same modules as `test_pcc.py` (TtC3k2 for model.6/8, TtC3k2PSA
  for model.22), then add a `TtYOLO26` end-to-end PCC test.

### YOLO26-0008 — test coverage

- READ: `test_pcc.py` builds its own layer list; it never instantiates `TtYOLO26`. `test_full_model.py` asserts only
  N3/N4/N5 neck PCC > 0.99 and only logs head PCC; `test_detailed_e2e.py` has no assert. `test_pcc.py` threshold 0.93
  vs its docstring's "Expected PCC > 0.99".

### YOLO26-0009 — cls PCC isolation

Per-stage PCC, baseline PR code (MEASURED, 1 chip, all pcc tests):

| Stage | PCC | Test |
|---|---|---|
| Backbone model.0 … model.9 | 0.9996, 0.9991, 0.9977, 0.9990, 0.9906, 0.9971, 0.9929, 0.9943, 0.9958, 0.9996 | test_backbone_continuous |
| C2PSA model.10 (isolated input) | 0.9986 | test_c2psa |
| Neck model.10 / 13 / 16 (N3) / 19 (N4) / 22 (N5) | 0.9931 / 0.9936 / 0.9966 / 0.9933 / 0.9922 | test_full_neck |
| Head N3 80×80: bbox / cls | 0.9595 / 0.9661 | test_full_model |
| Head N4 40×40: bbox / cls | 0.9687 / 0.9002 | test_full_model |
| Head N5 20×20: bbox / cls | 0.9940 / 0.8730 | test_full_model |
| **PyTorch** head fed **TTNN** neck output: boxes / scores | 0.9776 / 0.9404 | test_end_to_end |
| TTNN head, all scales concatenated: box / cls | 0.9719 / 0.9353 | test_pcc |

- INFERRED: every backbone/neck stage is ≥ 0.99, yet a PyTorch head on the TTNN neck output already drops to cls 0.940.
  The TTNN head implementation adds little (0.940 → 0.935); the cls branch amplifies small upstream errors.
  REFUTED (hypothesis): "the TTNN detection head is broken".
- READ: `TtConvBNSiLU._get_conv` uses `weights_dtype=ttnn.bfloat8_b` and the `tt_cnn` builder defaults
  `math_fidelity=LoFi`, `fp32_dest_acc_en=False`, `packer_l1_acc=False` for every conv.

Precision sweep, applied to all `TtConvBNSiLU` convs (MEASURED; test_pcc + test_full_model; file restored to the PR
version afterwards):

| Config | weights | fidelity | fp32 acc | test_pcc box / cls | Neck avg | Head box avg / cls avg |
|---|---|---|---|---|---|---|
| Baseline (PR) | bfp8 | LoFi | off | 0.9719 / 0.9353 | 0.9940 | 0.9740 / 0.9131 |
| A | bf16 | LoFi | off | 0.9796 / 0.9323 | 0.9912 | 0.9529 / 0.8704 |
| D | bf16 | LoFi | on | 0.9746 / 0.9158 (FAIL) | 0.9842 | 0.9331 / 0.8192 |
| E | bf16 | HiFi4 | off | 0.9917 / 0.9605 | 0.9891 | 0.9782 / 0.9256 |
| C | bfp8 | HiFi4 | on | 0.9829 / 0.9608 | 0.9955 | 0.9832 / 0.9540 |
| **B** | **bf16** | **HiFi4** | **on** | **0.9973 / 0.9754** | **0.9966** | **0.9937 / 0.9719** |

- INFERRED: math fidelity is the main lever (LoFi → HiFi4); fp32 accumulation only helps together with HiFi4
  (with LoFi it is worse, D); bf16 weights add on top (C → B).
- Status: fix not applied to the code; perf cost of B not measured.
- OPEN: B still leaves cls at 0.975 (N4 0.961, N5 0.971). Candidates not yet tested: non-conv ops (attention
  matmuls/softmax in C2PSA and PSA, upsample, concat) still at default precision; a per-layer sweep to see whether
  HiFi4 is only needed in a few layers (e.g. the head and the PSA blocks).

### YOLO26-0010 — README

- READ: references `test_pcc.py::test_yolo26_backbone_pcc` (only `test_yolo26_pcc` exists), `reference/` and
  `runner/performant_runner_infra.py` (actual: `runner/yolo26_test_infra.py`); its diagram says C2f, the network uses
  C3k2/C2PSA/PSA.

## Entries (appended 2026-10-22, later session)

All MEASURED numbers below: 1 chip (`TT_VISIBLE_DEVICES=0` + p150 mesh descriptor), 640×640, batch 1, PCC vs
Ultralytics fp32 CPU.

### YOLO26-0011 — residual cls loss is BatchNorm eps (corrects 0009)

- MEASURED (bf16 + HiFi4 + fp32 acc applied, `TtYOLO26` continuous on device, bus.jpg): backbone ≥ 0.9993 per layer,
  but model.10 0.9975 and model.22 0.9886; head cls N4 0.966.
- REFUTED (hypothesis): attention matmul/softmax precision. HiFi4 + fp32 acc + fp32 QK scores on the attention matmuls
  and softmax changed model.22 PCC by < 0.0001 (kept anyway; cheap).
- REFUTED (hypothesis): bf16 floor. PyTorch bf16 vs fp32 on the same model: cls 0.9985 (randn), 0.9977 (bus.jpg).
- MEASURED (model.22 fed the exact PyTorch input, bus.jpg): cv1 0.99996, bottleneck 0.99992, qkv 0.99989,
  V (input to `pe`) 0.99988, **`pe` output 0.98564**. A depthwise 3×3 conv through `TtConvBNSiLU` with random weights is
  exact (PCC 1.00000 for C = 64…256), so the defect is in the weights.
- READ: `fold_bn_to_conv_weights_bias(..., eps=1e-5)` and `get_folded_conv_weights(..., eps=1e-5)`; every
  `BatchNorm2d` in yolo26n/s has `eps=0.001` (114 layers each).
- MEASURED: per-channel scale error sqrt(var+1e-3)/sqrt(var+1e-5): 46/114 layers > 1%; worst
  `model.23.one2one_cv3.2.0.0.bn` ×1.90 (the N5 cls depthwise conv — N5 cls was the worst output),
  `model.22.m.0.1.attn.pe.bn` ×1.81, `model.10.m.0.attn.pe.bn` ×1.47.
- Fix: default eps → 1e-3 in `common.py` and `tt/model_preprocessing.py`.
- MEASURED after fix, precision sweep on `test_tt_yolo26` (n):

  | weights | fidelity | fp32 acc | box | cls | N5 cls |
  |---|---|---|---|---|---|
  | bfp8 | LoFi | off (PR) | 0.9790 | 0.9246 | 0.772 |
  | bfp8 | HiFi2 | on | 0.9876 | 0.9528 | 0.800 |
  | bfp8 | HiFi4 | on | 0.9866 | 0.9554 | 0.820 |
  | bf16 | HiFi2 | on | 0.9991 | 0.9957 | 0.995 |
  | **bf16** | **HiFi4** | **on (kept)** | **0.9997** | **0.9987** | **0.999** |

- INFERRED: with correct folding, bfp8 weights are the limiting factor (the corrected folded weights have wider
  per-channel range); bf16 + HiFi2 is a viable faster option (≥ 0.995) if perf requires.
- Correction to 0009: the precision sweep there was measured on wrongly folded weights; "conv precision is the root
  cause" is withdrawn — eps was the root cause, precision is a secondary requirement.

### YOLO26-0012 — `TtYOLO26` fixed and tested

- Change: `TtYOLO26Backbone` model.6/8 `TtC2f` → `TtC3k2`; `TtYOLO26Neck` model.22 `TtC3k2` → `TtC3k2PSA`.
- New test `tests/pcc/test_tt_yolo26.py`: runs `TtYOLO26` end to end with activations on device, compares one2one
  box/cls per scale and concatenated, asserts ≥ 0.99, parametrized over yolo26n/yolo26s.
- MEASURED: all 10 PCC test files pass (15 tests incl. layer-by-layer), `test_backbone_only` now passes.

  | Test | box | cls |
  |---|---|---|
  | test_pcc (n, host round-trip per layer) | 0.99968 | 0.99876 |
  | test_tt_yolo26 n (on-device) | 0.99967 | 0.99858 |
  | test_tt_yolo26 s (on-device) | 0.99959 | 0.99885 |

### YOLO26-0013 — yolo26s

- READ (Ultralytics graphs): s = n with every channel ×2 (width 0.5 vs 0.25, same depth, every C3k2 n=1). Not
  linear: PSA `num_heads = max(c // 64, 1)` (n: 2, s: 4); Detect `c2 = max(16, ch0 // 4, 4)` (n: 16, s: 32),
  `c3 = max(ch0, min(nc, 100))` (n: 80, s: 128). The PR used `num_classes` as the cls hidden width, which only holds for n.
- Change: `YOLO26_WIDTH_MULTIPLIER = {"yolo26n": 1, "yolo26s": 2}` applied to backbone/neck/head; PSA heads and head
  widths computed as above; other variants raise `ValueError`.
- MEASURED (bus.jpg, s, on-device per layer): all backbone/neck layers ≥ 0.9987; head box ≥ 0.9985, cls ≥ 0.9945.

### YOLO26-0014 — yolo26s L1 overflow

- MEASURED: `TT_THROW program.cpp:2568: Statically allocated circular buffers on core range [0-0 - 10-4] grow to
  1930688 B which is beyond max L1 size of 1572864 B` at model.7 (256→512, 3×3 s2, 40×40, block sharded).
- MEASURED: the same conv in isolation (interleaved input) fits in every block-sharded setting tried.
- INFERRED: in-model the conv inherits the previous layer's 55-core shard grid (`reshard_if_not_optimal=False`).
- Fix: block-sharded convs use `reshard_if_not_optimal=True`. n PCC unchanged within 0.001.

### YOLO26-0015 — trace + 2CQ perf

- READ: `test_yolo26_trace_2cq.py` decorated `@run_for_wormhole_b0()` (skips on Blackhole); harness called
  `ttnn.to_memory_config` on the dict output; runner fed a 3D `[1, H·W, 16]` tensor (conv2d asserts 4D) to a 3-channel
  stem; trace input-address assert failed because the head outputs reuse the freed input region.
- Fix: removed the WH-only decorator; `outputs_to_dram()` for the dict; `TtYOLO26(..., input_channels=16)` zero-pads
  the stem weights and `__call__` takes explicit `batch_size/height/width`; runner uses a persistent DRAM input that the
  trace reads (host → DRAM staging on CQ1, `ttnn.copy` staging → input on CQ0), so no address assert is needed.
- MEASURED (`test_yolo26_trace_2cq_perf`, 10 warmup + 50 timed, trace + 2CQ):

  | Variant | E2E latency | E2E FPS | Model only (trace replay ×100) | Input H2D (13.1 MB) |
  |---|---|---|---|---|
  | yolo26n | 11.64 ms | 85.9 | 5.36 ms (187 FPS) | 9.63 ms |
  | yolo26s | 11.77 ms | 85.0 | 6.21 ms (161 FPS) | 9.63 ms |

- INFERRED: E2E is bound by the host→device input copy (640×640×16 bf16, channels zero-padded 3 → 16).
  Next: send 3-channel or uint8 input and pad/normalize on device; then dp=4 across the QB2.

### YOLO26-0016 — demo

- READ: PR demo assumed a per-scale `[cx, cy, w, h, cls]` tensor, no sigmoid, BGR input, `ttnn.open_device` without
  `l1_small_size`.
- READ: Ultralytics 8.4.184 `predict` defaults to `end2end=False` (one2many head + NMS); the PR implements only the
  one2one (NMS-free) head, so the reference is Ultralytics with `Detect.end2end = True`.
- Fix: letterbox + BGR→RGB; decode ltrb distances from cell-centre anchors × stride, sigmoid scores, top-k (300)
  without NMS, as `Detect.postprocess`; `--compare` runs the reference on the same letterboxed input and writes a
  side-by-side image.
- MEASURED (conf 0.25): 16/16 reference detections matched (same class, IoU ≥ 0.990), confidence within 0.021:

  | Image | Variant | Detections | min IoU | max |Δconf| |
  |---|---|---|---|---|
  | bus.jpg | n | bus, 4× person | 0.990 | 0.007 |
  | zidane.jpg | n | 2× person, tie | 0.991 | 0.021 |
  | bus.jpg | s | bus, 4× person | 0.995 | 0.014 |
  | zidane.jpg | s | 2× person, tie | 0.995 | 0.008 |

- Outputs: `/home/ttuser/sdawle/yolo26_demo_outputs/{yolo26n,yolo26s}_{bus,zidane}[_vs_reference].jpg`.
- Command: `python models/experimental/yolo26/demo/demo.py -i <image> -o out.jpg --variant yolo26s --compare`
  (with the env from 0006). The demo is a single eager run; it does not use the trace runner.

### YOLO26-0018 — input padding + normalization moved to device

- MEASURED (host → DRAM upload, 1 chip, 30 iterations): the cost is set by the number of row pages, not bytes.

  | Host tensor | Size | Pages | Upload |
  |---|---|---|---|
  | bf16 [1,1,409600,16] (previous: padded + normalized on host) | 13.11 MB | 409600 | 9.63 ms |
  | bf16 [1,1,409600,3] | 2.46 MB | 409600 | 9.63 ms |
  | uint8 [1,1,409600,3] | 1.23 MB | 409600 | 9.63 ms |
  | **uint8 [1,1,640,1920] (HWC rows of the raw image)** | **1.23 MB** | **640** | **0.09 ms** |

- MEASURED (on-device prep from the wide uint8 tensor): typecast → bf16 + reshape to [1,1,H·W,3] 0.656 ms (bit-exact);
  + `ttnn.pad` 3 → 16 channels 2.346 ms. Tilize of the [.,.,H·W,3/4] uint8 layout fails (TT_FATAL in
  tilize_with_val_padding); the wide layout tilizes fine.
- Change: `TtYOLO26(..., uint8_input=True)` takes the raw uint8 RGB image [B, 1, H, W·3]; `preprocess()` runs
  typecast + reshape (+ zero-pad if `input_channels > 3`) inside the model/trace; normalization is folded into the stem
  weights (conv(x/255, W) = conv(x, W/255)), so it costs nothing at runtime. uint8 → bf16 is exact for 0..255.
  Runner uploads the raw image (`setup_uint8_input`); demo uses the same path.
- MEASURED (model-only = trace replay ×100; upload = host → DRAM staging):

  | Variant | input_channels | Model only | Upload |
  |---|---|---|---|
  | yolo26n | 3 (default) | 6.01 ms (166 FPS) | 0.09 ms |
  | yolo26n | 16 | 7.71 ms (130 FPS) | 0.09 ms |
  | yolo26s | 3 (default) | 6.85 ms (146 FPS) | 0.09 ms |
  | yolo26s | 16 | 8.55 ms (117 FPS) | 0.09 ms |

  Model-only rose 5.36 → 6.01 ms (n) / 6.21 → 6.85 ms (s): the on-device prep (~0.65 ms) is now inside the trace.
- MEASURED (`test_yolo26_trace_2cq_perf`, trace + 2CQ, 10 warmup + 50 timed):

  | Variant | E2E before | E2E after |
  |---|---|---|
  | yolo26n | 11.64 ms / 85.9 FPS | **6.16 ms / 162.3 FPS** |
  | yolo26s | 11.77 ms / 85.0 FPS | **7.00 ms / 142.9 FPS** |

  E2E is now ~0.15 ms above model-only: the upload is fully overlapped and compute-bound.
- MEASURED accuracy, new `test_tt_yolo26_uint8_input_pcc` (bus.jpg resized to 640, vs Ultralytics fp32 on x/255):
  n box 0.99959 / cls 0.99850; s box 0.99903 / cls 0.99755. Demo (uint8 path): 16/16 detections matched,
  IoU ≥ 0.988. Full suite: 19 passed (PCC + trace perf).
- Next: the remaining 0.65 ms prep could shrink by fusing typecast into the stem conv input or sharding the reshape;
  dp=4 across the QB2.

### YOLO26-0019 — late-recorded items (appended; observed earlier in the session)

- MEASURED: loading yolo26n and then yolo26s `YOLO26PerformantRunner` in one device session (perf split script) →
  `TT_FATAL bank_manager.cpp:514: Out of Memory: Not enough space to allocate 47200 B L1_SMALL buffer across 50 banks
  (allocated: 24256 B, free: 320 B)`. Each model alone fits (`YOLO26_L1_SMALL_SIZE`). Worked around by one process per
  variant; INFERRED: conv state of the first model (weights/halo configs held in `TtConv2d` caches) is never released.
  Not fixed — matters for multi-model serving.
- MEASURED: every `TtYOLO26` forward logs dozens of `warning | Op | ttnn::tilize: Using input shard spec for output tensor
  because the legacy sharded optimized program factory is being used (tilize_device_operation.cpp:275)`. Functionally
  harmless (PCC unaffected); not addressed. Noise hides real warnings in logs.
- API additions to `TtYOLO26` not listed elsewhere:
  - `__call__(x, batch_size, height, width, deallocate_input)` — explicit dims for flattened input; `deallocate_input`
    frees the input after the stem. Added for the first trace attempt (YOLO26-0015); the trace runner no longer uses it,
    the uint8 path sets it internally to free its own preprocessed copy.
  - `ATTN_COMPUTE_KERNEL_CONFIG` (HiFi4, fp32 dest acc, packer L1 acc) on the attention QKᵀ/AV matmuls and softmax, QKᵀ
    output in fp32. Measured no PCC change (YOLO26-0011, REFUTED as root cause); kept. Perf effect not isolated.
- Test-infra change: `YOLO26TestInfra` default input `torch.randn` → `torch.rand` so it is a valid [0, 1] image for
  the uint8 path (`setup_uint8_input` quantizes to 0..255).

### YOLO26-0020 — ttnn / tt-metal behaviour vs model bugs

Model-code bugs (in `models/experimental/yolo26`, all FIXED): BN eps (0011), wrong blocks in `TtYOLO26` (0007/0012),
n-only channels/heads (0013), demo decode (0016), trace harness (0015), bfp8/LoFi precision choice (0011).

ttnn / tt-metal / environment behaviour we worked around (candidates for upstream issues; none filed):

| ID | Behaviour | Workaround here | Upstream? |
|---|---|---|---|
| 0006 | 1 of 4 visible chips on a QB2 → `ClusterType::CUSTOM` → TT_FATAL unless `TT_MESH_GRAPH_DESC_PATH` set | p150 1×1 mesh descriptor | Yes: QB2 single-chip subset should resolve automatically |
| 0005 | Ethernet core timeout after another process (older tt-metal) used the board | `tt-smi -r` | Maybe: error text already says "reset the board" |
| 0014 | Block-sharded conv inheriting a previous layer's shard grid overflows L1 (1.93 MB > 1.57 MB) with no fallback | `reshard_if_not_optimal=True` | Yes: conv could auto-reshard or report the grid it chose |
| 0018 | Host → device copy time scales with row-page count, not bytes: 1.23 MB as 409600 pages 9.63 ms vs 640 pages 0.09 ms | upload wide rows, reshape on device | Yes: large perf cliff for narrow row-major tensors |
| 0018 | `to_layout(TILE)` on uint8 row-major [1,1,409600,3/4] → TT_FATAL in tilize_with_val_padding | avoid; typecast in row-major | Yes |
| 0019 | L1_SMALL not released between two models in one session | one process per model | Maybe: confirm with a minimal repro first |
| 0019 | `ttnn::tilize` legacy-factory warning on every call | none | Yes: downgrade to debug or emit once |
| 0003 | (environment) `pip` on PATH points at `~/.tenstorrent-venv`; tt-metal venv is uv-managed | `uv pip` | No (local setup) |
| 0016 | (Ultralytics) `predict` default `end2end=False` uses one2many + NMS for YOLO26 | reference with `Detect.end2end = True` | No (by design) |

### YOLO26-0021 — dp=4 across the QB2 (appended 2026-10-23)

- Code (uncommitted at time of writing):
  - `YOLO26TestInfra.setup_uint8_input` uses `ttnn.ShardTensorToMesh(device, dim=0)` when `num_devices > 1`, so each
    chip receives one image of the [4, 1, H, W·3] uint8 batch. Weights replicate through conv2d unchanged.
  - `test_yolo26_trace_2cq.py`: new `test_yolo26_trace_2cq_perf_dp` (`mesh_device` = 4, indirect); FPS =
    batch_size × num_devices / time; model name tagged `_dp_{num_devices}`.
- Correctness (prototype `dp4_proto.py`, 4 distinct images: bus, zidane, and their mirrors): MEASURED per-chip PCC
  vs Ultralytics one2one ≥ 0.9976 cls for both variants.
- Perf, METHOD: `pytest ...::test_yolo26_trace_2cq_perf_dp -q`, 10 separate processes, both variants per process,
  trace + 2CQ e2e (upload on CQ1, trace on CQ0, outputs read back), batch 1 per chip × 4 chips, 640×640.
  `TT_VISIBLE_DEVICES` / `TT_MESH_GRAPH_DESC_PATH` unset. MEASURED:

| Variant | ms per batch of 4 (range) | FPS, 10 runs (min / mean / max) | Single-chip 10-run mean | Scaling |
|---|---|---|---|---|
| yolo26n | 6.21–6.22 | 642.6 / 643.7 / 644.3 | 162.28 FPS (6.16 ms) | 3.97× |
| yolo26s | 7.05–7.06 | 566.4 / 566.9 / 567.4 | 142.88 FPS (7.00 ms) | 3.97× |

  Per-run FPS n: 643.8, 643.7, 643.5, 644.3, 643.5, 643.9, 643.7, 642.6, 644.2, 643.6;
  s: 567.2, 566.4, 566.9, 567.2, 567.0, 567.4, 566.8, 566.6, 566.9, 567.0.
  INFERRED: the ~0.05 ms per-batch overhead over single chip is mesh dispatch / 4-way read-back; no inter-chip traffic.
- Defect, MEASURED: in the first 10-run loop, runs 1–5 passed, then run 6 hung inside `capture_trace_2cq` (yolo26n,
  first variant) until the 900 s timeout, although run 5's log ends with a clean cluster close. Runs 7–10 hung at the
  same point (state carried over). After `tt-smi -r`, runs 6–10 were redone with a reset before each run and all
  passed with the numbers above. Not reproduced since; root cause unknown (INFERRED candidates: device or dispatch state
  left over across mesh open/close, not the model code, since the identical code passed 10/10 after resets).
  Status OPEN: if it recurs, capture `tt-triage` / watcher output before resetting.
- Next: commit the dp changes; consider larger per-chip batch to amortize dispatch.
