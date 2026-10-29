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
| YOLO26-0022 | Single-chip perf optimization from a tracy profile (ops, fusion, sharding, L1, matmul/conv config sweeps) | 2026-10-23 | DONE: model-only n 6.01 → 1.81 ms, s 6.85 → 2.34 ms; e2e n 522 FPS, s 412 FPS; dp=4 n 1772 FPS, s 1567 FPS; PCC unchanged, 17/17 PCC tests, demo 16/16 |
| YOLO26-0023 | 600 FPS on one chip for yolo26s with PCC intact | 2026-10-23 | DONE: batch 2 per chip, e2e trace+2CQ 603.6 / 603.9 / 604.4 FPS (10 runs); PCC unchanged; batched demo 8/8 matched, checked visually |
| YOLO26-0024 | Batch > 2 per chip | 2026-10-23 | DONE: max batch 4 (batch-aware conv blocking); e2e batch 4 s 659.7 FPS, n 1092.6 FPS; PCC intact; batch 5 still fails |
| YOLO26-0025 | dp=4 with batch 2 / 4 per chip | 2026-10-23 | DONE: batch 4/chip n 2561–2571 FPS, s 2347–2404 FPS; batch 2/chip n ~1868, s ~1805; one trace-capture hang (batch 2, n) |
| YOLO26-0026 | Can one chip serve 20 camera streams at 30 FPS (600 images/s)? Live multi-stream demo | 2026-10-26 | DONE: yolo26s batch 4, 20 CAVIAR shopping-centre streams, 30.0 FPS per stream sustained 60 s with recording (599.4 images/s); ceiling 661 images/s (33 FPS per stream) |
| YOLO26-0027 | Demo says batch 4: are the 20 streams run 4 at a time, in sequence? | 2026-10-27 | ANSWERED: yes, each 33.3 ms tick runs 5 batch-4 traces back to back (~30 ms of compute); panel changes and v2/v3 recordings |
| YOLO26-0028 | 80 streams × 30 FPS on all 4 QB2 chips (same 20 clips on each chip), live demo + recording | 2026-10-27 | DONE: dp=4 mesh, batch 4/chip, outputs packed on device, host prep in worker processes: 2399.2 images/s = 30.0 FPS per stream for 60 s with recording; ceiling 2598.5 images/s (32.5 FPS/stream); capture hang hit 2× (OPEN, see YOLO26-0025) |
| YOLO26-0029 | Per-camera-group latency of the multiplexed demo; cut it | 2026-10-27 | DONE (1 chip): CAM 17-20 46.9 → 39.6 ms, on screen 50.9 → 43.4 ms at 600 images/s; 4 chips: on screen 57-61 ms, unchanged within noise (device at ~92 % of its ceiling) |
| YOLO26-0030 | Batch > 4 on one chip using DRAM (8 / 16 / 20); does 20 streams x 30 FPS hold at batch 20? | 2026-10-27 | DONE (branch yolov26_dram_batch): hybrid grouped forward, model-only b8 731.3 / b16 729.8 / b20 716.4 FPS vs b4 683.1, PCC intact; demo at b20 29.9 FPS but 108 ms latency (b4: 30.0 FPS, 39 ms) |
| YOLO26-0031 | Cut end-to-end latency of the 1-chip 20-stream demo further | 2026-10-27 | DONE: capture-to-results 10.9 ms for every camera (was 17-40 ms by group), on screen 23.2 ms (was 43.4), 30.0 FPS kept; staggered capture, earlier hand-over, event-driven render |
| YOLO26-0032 | Batch 20 (all 20 cameras in one pass) on 1 chip: minimize its latency | 2026-10-27 | DONE: capture-to-results 61.7 → 44.0 ms (all cameras alike), on screen 66.0 → 52.4 ms with live view, 30.0 FPS; batch 4 also 10.9 → 9.4 ms / 23.2 → 20.7 ms |
| YOLO26-0033 | How did batch 4 reach ~9.4 ms per camera group? Is it async? (summary of 0026-0032) | 2026-10-27 | ANSWERED: pipelined across processes and both command queues; no group queues behind another; chip work unchanged (~6 ms per batch of 4) |
| YOLO26-0034 | Batch 20 before / after table; how batch 20 combines L1 and DRAM | 2026-10-27 | ANSWERED: 108.3 → 44.6 ms (all cameras), on screen 110.7 → 53.0 ms; every layer computes in L1 in groups (4 above 20×20, 8 at 20×20), DRAM only holds full-batch stage tensors |
| YOLO26-0035 | Why on-screen latency exceeds capture-to-results, and by different amounts per configuration; synchronized batch 4 | 2026-10-27 | ANSWERED: ~9 ms fixed display work (panel ~2 ms, JPEG ~6 ms) plus up to one compose of waiting when the renderer is busy; synchronized batch 4 11.5 → 35.3 ms by set |
| YOLO26-0036 | Cut the on-screen (display) overhead: parallel strip JPEG + panel at 10 Hz | 2026-10-27 | DONE: on screen bs4 staggered 21.4 → 11.3 ms, bs4 synchronized 34.2 → 24.6, bs20 53.0 → 48.0; live-view image identical; panel may lag tiles ≤ 100 ms |
| YOLO26-0037 | Defect from 0036: panel at 10 Hz disagreed with the tiles in the recording | 2026-10-27 | FIXED: only changed panel rows redrawn on every update; 60/60 rows match in 3 checked frames; on screen 11.5 → 12.8 ms |
| YOLO26-0038 | Like-for-like latency table: bs=4 vs bs=20, all 20 cameras captured together, with display breakdown | 2026-10-28 | RECORDED: bs=4 11.2 → 35.2 ms by set (on screen ~14.7 → ~38.7), bs=20 44.6 ms all (on screen ~50.3); display +3.5 / +5.7 ms |
| YOLO26-0039 | dp=4 batch 4 reaches ~91 % of 4 × one chip (2,406 vs 2,638 FPS): where does the rest go? | 2026-10-29 | ROOT-CAUSED: output read-back (reads from the 4 chips run one after another; 8.60 MB per chip, a third of it tile padding). Copies outside the trace refuted (≤ 0.05 ms); a host thread per chip refuted (2× slower); untilizing the outputs in the trace 2,450 → 2,533 / 2,554 FPS on 4 chips but 662 → 633 on 1 chip; not applied to the test |

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

### YOLO26-0022 — single-chip performance optimization (appended 2026-10-23)

Goal: cut latency on one chip (dp scales it), guided by a tracy device profile. All code in `tt/ttnn_yolo26.py`
unless noted; uncommitted at time of writing.

Methods:
- Device profile: `python -m tracy -r -p --no-web-server <script>`; script (`scratchpad/prof.py`) builds `TtYOLO26`
  (uint8 input), runs 2 warm forwards and 1 profiled forward, `ttnn.ReadDeviceProfiler` after each (reading only once
  overflowed the 12000-marker profiler DRAM buffers: "markers were dropped" + report assert). Numbers below are
  DEVICE KERNEL DURATION of the last forward.
- Model-only latency: trace of one forward replayed 200× back to back (`scratchpad/quick.py`), plus PCC on bus.jpg
  (uint8 path) vs Ultralytics one2one output. "e2e" = `test_yolo26_trace_2cq_perf` (trace + 2CQ, upload + read-back).
- All runs on chip 0 (`TT_VISIBLE_DEVICES=0` + p150 mesh descriptor) unless dp=4.

Starting point (MEASURED, yolo26n, 624 ops, kernel sum 5.37 ms; trace 6.01 ms):

| Bucket | Device time | Cause |
|---|---|---|
| Input reshape [640,1920]→[409600,3] + I2S of 6-byte rows | 0.637 + 0.791 ms | one NoC transfer per pixel |
| untilize / tilize / slice / concat / reshape / S2I / I2S / permute / binary in DRAM row-major | ~2.6 ms | every add, split and concat went DRAM + row-major |
| standalone SiLU (87 unary ops) | 0.34 ms | SiLU not fused into conv |
| conv + halo + matmul | ~0.85 ms | |

Changes and their effect (MEASURED, model-only ms, PCC box/cls on bus.jpg):

| # | Change | n | s | PCC n | PCC s |
|---|---|---|---|---|---|
| 0 | start | 6.01 | 6.85 | 0.99959/0.99850 | 0.99903/0.99755 |
| 1 | uint8 input padded to 8 ch on host ([B,1,H,W·8], `uint8_input_from_torch`); rows sharded into L1 (640 wide pages), typecast in L1, `ttnn.experimental.view` to [HW, 8] (zero copy: shard width == last dim); stem consumes that sharding | | | | |
| 2 | SiLU fused into conv (`activation=SILU`) | 4.34 | 5.29 | 0.99960/0.99819 | 0.99919/0.99788 |
| 3 | Activations stay [1,1,NHW,C] tile in L1: residual `add` and channel `concat` shard-local (height-sharded inputs aligned to the first, else L1 interleaved); C2f/C3k2/C2PSA chunk replaced by a second 1×1 conv on half the cv1 weights (`cv1_b`); block-interface channels zero-padded to 32 (n: 16-ch hidden of model.2/model.16) so concats are tile-aligned | 2.40 | 3.12 | 0.99963/0.99837 | 0.99919/0.99788 |
| 4 | Attention token-major: qkv weights reordered to [q|k|v] heads at load, scale folded into q; no permutes | 2.01 | 2.66 | 0.99964/0.99857 | 0.99918/0.99779 |
| 5 | Heads batched: q/k zero-padded to head_dim, `nlp_create_qkv_heads` → one Q·Kᵀ, one softmax, one ·V, `nlp_concat_heads` | 1.951 | 2.506 | same | same |
| 6 | SPPF pools row-major height-sharded (n only, hidden ≤ 128) | 1.928 | 2.505 | same | same |
| 7 | bf16 attention scores (was fp32) | 1.918 | 2.488 | 0.99964/0.99845 | 0.99918/0.99792 |
| 8 | explicit `MatmulMultiCoreReuseProgramConfig` for Q·Kᵀ / ·V (one 32-token block of one head per core) | 1.865 | 2.388 | 0.99964/0.99844 | 0.99919/0.99796 |
| 9 | conv height/block threshold NHW/C_out > 4 → > 8 | 1.835 | 2.351 | 0.99962/0.99844 | 0.99921/0.99802 |
| 10 | softmax without fp32 dest acc (`SOFTMAX_COMPUTE_KERNEL_CONFIG`) | 1.814 | 2.331 | 0.99963/0.99830 | 0.99919/0.99802 |

Step 1 alone (stem micro-benchmark, host-timed incl. dispatch): 1.537 → 0.142 ms, bit-identical conv output.
End state (MEASURED, yolo26n profile): 272 ops, kernel sum 1.434 ms (conv 576, matmul 476, halo 140, softmax 43 µs);
yolo26s: 277 ops, 2.158 ms. Trace minus kernel sum ≈ 0.4 ms on n = dispatch gaps (~1.5 µs/op).

Sweeps (MEASURED) — kept configuration is HiFi4, fp32 dest acc, packer L1 acc, bf16 weights:

| Conv/matmul config (all layers) | n ms | n PCC box/cls | s ms | s PCC box/cls |
|---|---|---|---|---|
| HiFi4, fp32 acc, l1acc, bf16 (kept) | 2.014 | 0.99964/0.99857 | 2.655 | 0.99918/0.99779 |
| HiFi2 | 2.000 | 0.99879/0.99336 | hung, see below | |
| fp32 acc off | 1.951 | 0.99815/0.99348 | 2.508 | 0.99470/0.99111 |
| packer l1 acc off | 2.018 | 0.99957/0.99854 | 2.676 | 0.99918/0.99775 |
| fp32 + l1 acc off | 1.945 | 0.99717/0.98968 | 2.534 | 0.98976/0.98433 |
| bfp8 weights | 2.003 | 0.99241/0.96691 | 2.598 | 0.95598/0.94312 |
| LoFi | 1.997 | 0.96428/0.84376 | 2.542 | 0.93236/0.90340 |

(run at step 4; at most 3–6 % faster for a clear accuracy loss.) Per-op sweep under tracy (6 conv shapes × fidelity ×
fp32 × l1acc × SiLU): fidelity changes nothing (ops are not FPU-bound); the fused SiLU dominates, e.g. 1×1
160×160 32→64: 3.7 µs without, 23.9 µs with; stem 29 vs 67 µs (~1.25 µs per output tile per core). Model-level with
activations removed (timing only): 2.398 → 2.205 ms, i.e. SiLU ≈ 0.19 ms total.

| SiLU variant on a sharded 160×160×64 conv output | device time | accuracy |
|---|---|---|
| fused in conv (kept) | 23.9 µs | PCC 0.99995 |
| conv + `ttnn.silu` | 3.7 + 16.2 µs | same |
| conv + `sigmoid(FastApproximate)` + mul | 3.7 + 3.4 + 9.8 µs | PCC 0.9974, max rel err 2.7 % (rejected) |
| conv + `sigmoid(AccurateWithFastExp)` + mul | 3.7 + 9.6 + 9.8 µs | exact |
| mul with sigmoid as `input_tensor_b_activations` | binding rejected `UnaryWithParam` list | — |

READ: `ckernel_sfpu_silu.h` always uses the non-approximate sigmoid (exp + reciprocal), ignoring approx mode.

| Height/block threshold (height if NHW/C_out > t) | n ms | s ms |
|---|---|---|
| 4 (original) | 1.865 | 2.388 |
| 6 | 1.863 | 2.387 |
| 8 / 10 / 12 (kept 8) | 1.835 | 2.350 |
| 16 | 1.898 | 2.555 |
| 32 | 2.008 | 2.608 |
| 64 | 2.088 | FAIL (TT_THROW program.cpp:2625) |

| SPPF maxpool 5×5 on 20×20 (tracy, pool + halo µs) | C=128 | C=256 |
|---|---|---|
| tile in/out, auto = block (was) | 26.2 + 2.0 (28 cores) | 26.3 + 2.0 (56 cores) |
| row-major in/out, height | 8.5 + 2.7 (100 cores) | 15.3 + 4.9 |
| tile out, height | 48.4 (13 cores) | 94.0 |
| width | 145–170 (4–16 cores) | 145–170 |

In-model, row-major pools need an untilize and a tilize of the 400-row concat that run on 13 cores; net n 1.951 →
1.928 ms, s 2.506 → 2.520 ms, so s keeps tile pools (`row_major_pools = hidden ≤ 128`).

| Softmax variant (PSA) | n ms | s ms | PCC n | PCC s |
|---|---|---|---|---|
| numeric_stable, fp32 dest | 1.835 | 2.351 | 0.99962/0.99844 | 0.99921/0.99802 |
| not stable, fp32 dest | 1.822 | 2.346 | 0.99962/0.99836 | 0.99919/0.99799 |
| numeric_stable, bf16 dest (kept) | 1.814 | 2.331 | 0.99963/0.99830 | 0.99919/0.99802 |
| not stable, bf16 dest | 1.805 | 2.332 | 0.99960/0.99826 | 0.99919/0.99808 |

Stable kept: guards exp overflow on other images for ≤ 0.01 ms.

Refuted / rejected (MEASURED): `core_grid=` on the batched attention matmuls (n 1.918 → 2.027 ms; replaced by explicit
program configs); lower conv precision (tables above); SiLU alternatives (table above); thresholds > 12.

Defects hit and fixed during the work (MEASURED):
- concat helper copied the first tensor's full memory config incl. shard width → `TT_FATAL Shard width 64 must match
  physical width 32` in reshard; fixed with a per-tensor config (same grid / rows per core, own width).
- `max_pool2d(applied_shard_scheme=...)` on an already sharded input → `TT_FATAL A sharding scheme should not be
  specified for a sharded input tensor`; pass the scheme only for the first (interleaved) pool.
- conv given a row-major tensor height-sharded 4 rows per core → `TT_FATAL shard_align_error` (tensor_layout.cpp:171);
  SPPF cv2 now gets it L1 interleaved.
- 8 of 17 tests in `tests/pcc` (block-by-block / partial-model tests that re-upload each block input as row-major NHWC
  and call `ttnn.concat` on `TtUpsample` output) failed after step 3: (a) SPPF residual `add` got a row-major [1,H,W,C]
  identity → tilize shard error; (b) the tests concat `TtUpsample` output with NHWC tensors. Fixed by `_as_flat_tile`
  (add/concat bring any input to [1,1,NHW,C] tile; no-op inside the full model) and restoring `TtUpsample`'s NHWC
  row-major output contract. 17/17 pass after; model-only time unchanged (n 1.814, s 2.339 ms).
- Device hang: the HiFi2 yolo26s sweep run sat at 100 % CPU for 11 min (normal run ~2 min) inside the forward. Run
  killed, `tt-smi -r`, all later runs used a reset per run and a 420 s timeout; not reproduced; root cause unknown
  (INFERRED: same family as the YOLO26-0021 trace-capture hang; HiFi2 itself not suspected, n HiFi2 ran fine).

Validation of the final code (MEASURED):
- `tests/pcc` 17/17 pass. `test_tt_yolo26.py`: random input n box 0.999686 / cls 0.998921, s 0.999620 / 0.998932;
  uint8 bus.jpg n 0.999628 / 0.998305, s 0.999190 / 0.998020.
- e2e trace + 2CQ, single chip, 3 runs: n 1.91 ms, 522.1–522.7 FPS (was 162.28); s 2.43 ms, 412.1–412.3 FPS (was
  142.88).
- dp=4 trace + 2CQ, 3 runs (reset before each): n 2.24–2.27 ms per batch of 4, 1759.6 / 1772.1 / 1789.2 FPS
  (min/mean/max; was 643.7); s 2.55–2.56 ms, 1564.6 / 1566.6 / 1569.3 FPS (was 566.9). n scales 3.4× over one chip
  (vs 3.97× before): INFERRED per-iteration mesh overhead (2.26 vs 1.91 ms) now matters at ~1.9 ms per forward.
- Demo (`demo.py --compare`, bus + zidane, n + s): 16/16 reference detections matched (same class, IoU ≥ 0.992,
  conf within 0.02); images in `/home/ttuser/sdawle/yolo26_demo_outputs/optimized/`.
- Log noise: some 1×1 convs log `Mismatch between computed MemoryConfig ... Using computed config`
  (matmul_device_operation.cpp:278) — matmul picks its own output core grid; harmless, not addressed.

API changes: uint8 input is now [B, 1, H, W·8] (`UINT8_INPUT_CHANNELS`, `uint8_input_from_torch`; callers in test
infra, demo and PCC test updated); `TtAttention` / `TtBottleneck` / `TtC2f` / `TtC3k` / `TtC3k2` internals changed
(padded interfaces, `cv1_b`); `TtConvBNSiLU.load_weights` takes optional `in_segments` / `out_segments`.

Open: SiLU SFPU cost (~0.19 ms) needs a tt-metal kernel change; dispatch gaps (~0.4 ms on n) need fewer ops; stem conv
67 µs of which ~38 µs SiLU on 16→32-padded output tiles.

### YOLO26-0020 addendum (2026-10-23) — further ttnn behaviour found in YOLO26-0022 (none filed)

| Behaviour | Workaround here | Upstream? |
|---|---|---|
| SFPU SiLU ignores approx mode, ~1.25 µs per tile per core; dominates small fused convs | none | Yes: approximate SiLU option |
| Batched matmul [1,h,416,64]×[1,h,64,416] default config uses 1 core per head (44 µs); `core_grid=` made it slower | explicit `MatmulMultiCoreReuseProgramConfig` | Yes: default config |
| `max_pool2d` tile output: 26 µs on 28 cores at 20×20×128 vs 8.5 µs row-major height-sharded | row-major pools for n | Maybe |
| `reshape` of a sharded row-major tensor changing the last dim goes S2I → reshape → I2S; `experimental.view` does it for free when shard width == last dim | use `view` | Maybe: reshape could take the view path |
| Profiler: one `ReadDeviceProfiler` after weight-prep + 2 forwards overflows 12000-marker buffers, then the ops report asserts | read after every forward | Maybe: clearer error |

### YOLO26-0023 — 600 FPS single chip for yolo26s (appended 2026-10-23)

Target: ≥ 600 FPS end-to-end (trace + 2CQ, upload + read-back) on ONE chip for yolo26s, PCC intact, detections
checked visually. Start: batch 1, e2e 2.43 ms = 412 FPS, model-only 2.33 ms.

Where batch-1 time goes (MEASURED, s): model-only without any activation 1.918 ms vs 2.339 ms ⇒ SiLU ≈ 0.42 ms; even a
free SiLU leaves batch 1 at ~1.9 ms + e2e overhead, i.e. < 600 FPS. So the target needs throughput: several 20×20
/ 40×40 layers run on 13–56 cores, so a second image costs much less than a second forward.

Changes (code):
- `TtAttention` supports batch > 1: fused qkv → row-major → [B, 1, N, C] → tile, `nlp_create_qkv_heads` batches
  heads × images; heads concat back and return to [1, 1, B·N, C] (N = 400 is not tile-aligned, hence row-major).
- Intermediates freed as soon as consumed (`_free` in every block, backbone, neck, head, attention; temporary copies in
  `add` / `concat`). Without it batch 2 failed: `Statically allocated circular buffers ... clash with L1 buffers`
  (head cv3_0_0). Batch 3/4 still fail the same way in the 160×160 stage (not pursued).
- Persistent trace input lives in L1, sharded by image rows (`TtYOLO26.input_memory_config`); per iteration
  `ttnn.copy(staging DRAM → L1 sharded)` (0.020 ms) replaces DRAM→DRAM copy (0.039 ms) + in-trace sharding copy;
  `preprocess` skips the copy when the input is already sharded.
- SPPF pools row-major also from batch 2 on (s: 3.191 → 3.178 ms).
- Height/block rule on output NHW (stride-2 convs), threshold 4 (s 2.339 → 2.305 ms, n 1.814 → 1.806 ms).
- perf test parametrized over batch 1 and 2; demo takes several `-i` images as one batch.

MEASURED, yolo26s batch 2: model-only 3.178 ms per 2 images (629 FPS); PCC per image (bus, mirrored bus) box
0.99920 / 0.99897, cls 0.99800 / 0.99788 (batch-1 values 0.99920 / 0.99800). e2e (`test_yolo26_trace_2cq_perf
[yolo26s-640-2]`, 10 separate processes, chip reset before each): 3.31 ms per batch, 603.6 / 603.93 / 604.4 FPS
(min / mean / max). Loop split before the input change (batch 2): trace 3.190, + copy & upload 3.243, + outputs to
DRAM 3.279, + reads 3.383 ms.

Regression (MEASURED): `tests/pcc` 17/17; single chip n b1 527.0, n b2 795.2, s b1 418.4, s b2 604.2 FPS; dp=4 (b1
per chip) n 1772.6, s 1582.5 FPS. Batched demo (bus + zidane in one forward, yolo26s): 8/8 reference detections
matched, IoU ≥ 0.995, conf within 0.021; images in `/home/ttuser/sdawle/yolo26_demo_outputs/batch2/`, checked
visually (identical objects and boxes to the CPU reference).

Tried and rejected (MEASURED):
- Per-layer conv sweep (each conv alone, tracy, height/block/width × act_block_h 0/32/64/128): best-per-layer sum
  1600 vs default 1680 µs, but applying the 11 winning overrides made the model slower (2.305 → 2.428 ms; reshards
  between neighbours). Mechanism removed.
- Packed single output tensor (one DRAM read per iteration): e2e n 1.91 → 2.07 ms, s 2.43 → 2.59 ms (in-trace concats
  of unaligned 80-ch scores cost more than the saved reads on one chip). Removed.
- bfloat8_b conv outputs: s 2.314 ms (−1 %), cls 0.98573. fp32 dest acc off globally: s 2.197 ms, cls 0.99232.
- Stem conv only with bf16 dest: s b2 3.178 → 3.159 ms but cls 0.99800 → 0.99691 (PCC not intact). Reverted.
- Depthwise convs: READ `prepare_conv2d_weights.cpp` expands grouped weights to dense, but with block sharding each
  core only multiplies its 32-channel block (act_block_w 3 tiles); their time is readers + SiLU, not wasted FLOPs.

Process note: an edit script computed an empty slice and `str.replace("", ...)` prepended a block to the top of
`ttnn_yolo26.py` (IndentationError in the next test run); repaired by moving the block into the class and checked
against the last checkpoint. The intermittent trace-capture hang (YOLO26-0021/0022) recurred once (packed-output run);
all later runs reset the chip first.

Margin: 0.6 %. Remaining levers: SiLU kernel (needs tt-metal change, ~0.8 ms of the 3.2 ms batch-2 forward is SFPU
SiLU by the batch-1 ratio, INFERRED), attention batch round-trip (~38 µs per PSA block), e2e reads (0.10 ms).

### YOLO26-0024 — batch > 2 on one chip (appended 2026-10-23)

MEASURED failures before the change (all `Statically allocated circular buffers ... clash with L1 buffers`): batch 3
and 4 at the stem conv (640×640, default act_block_h 30 tiles); after shrinking the stem blocks, batch 3 at model.5
(3×3 s2 80×80×256→256: full 3×3×256×256 bf16 weights on every height-sharded core, double-buffered); batch 4 then at
model.1.

Fix (`TtConvBNSiLU._large_batch_settings`, chosen per batch size since the conv cache is keyed by batch):
- batch ≤ 2: conv defaults (unchanged; the large-batch settings measured 2.305 → 2.568 ms at batch 1, 3.178 → 3.667
  ms at batch 2 for s, so they are not applied there);
- batch 3: weights single-buffered, stem act_block_h 256;
- batch 4: weights single-buffered, stem act_block_h 128, other height-sharded convs act_block_h 64.
- `preprocess` frees the model's own sharded uint8 copy right after the typecast.

MEASURED (yolo26s, model-only, quickb.py, PCC per image vs Ultralytics): batch 1 2.309 ms (433 FPS), 2 3.179 ms (629),
3 4.661 ms (644), 4 5.855 ms (683); min cls PCC 0.99788–0.99800 at every batch. yolo26n batch 4: 3.430 ms (1166 FPS),
cls ≥ 0.99822. e2e trace + 2CQ batch 4: s 6.06 ms = 659.7 FPS, n 3.66 ms = 1092.6 FPS. Batched demo (4 images, s):
16/16 reference detections matched, IoU ≥ 0.995, checked visually (`/home/ttuser/sdawle/yolo26_demo_outputs/batch4/`).
`tests/pcc` 17/17.

Batch 5 still fails at model.2.cv2 (1×1 conv on the 160×160×192 concat, ~12 KB short; act_block_h does not apply to
the matmul path); act_block_h 32 / stem 64 did not help. Would need the block input freed before cv2 inside C2f
(ownership change) or DRAM slicing of the 160×160 stage. Not pursued: batch 3 → 4 already gained only 6 %.

### YOLO26-0025 — dp=4 with batch 2 / 4 per chip (appended 2026-10-23)

METHOD: `test_yolo26_trace_2cq_perf_dp[blackhole-<variant>-640-<batch>-4-device_params0]` (dp test now parametrized
over batch 1, 2, 4), trace + 2CQ e2e, 3 separate processes per point, chip reset before each. MEASURED:

| dp=4 | ms per pass | FPS (runs) |
|---|---|---|
| n, batch 1/chip (YOLO26-0022) | 2.26 | 1772.6 |
| s, batch 1/chip (YOLO26-0022) | 2.53 | 1582.5 |
| n, batch 2/chip (8 images) | 4.28–4.29 | 1870.2, 1866.2 (run 1 hung, see below) |
| s, batch 2/chip (8 images) | 4.41–4.46 | 1812.8, 1797.3, 1795.4 |
| n, batch 4/chip (16 images) | 6.22–6.25 | 2561.9, 2570.5, 2560.7 |
| s, batch 4/chip (16 images) | 6.66–6.82 | 2403.6, 2379.0, 2347.4 |

Scaling vs one chip at the same per-chip batch (e2e): s batch 2 604 → 1805 (3.0×), s batch 4 660 → ~2377 (3.6×).
INFERRED: the gap to 4× is per-iteration host work on the mesh (8 / 16-image upload, 6 outputs × 4 devices read back),
which scales with batch; the traced compute itself shows no mesh overhead (YOLO26-0022: 1.876 vs 1.861 ms).

Defect, MEASURED: dp=4 batch 2 yolo26n run 1 hung in `capture_trace_2cq` until the 600 s timeout; the next runs
(after `tt-smi -r`) passed. Same intermittent hang as YOLO26-0021/0022; root cause still unknown.

### YOLO26-0026 — 20 streams × 30 FPS on one chip, live demo (appended 2026-10-26)

Question: batch 4 gives 660 FPS e2e (YOLO26-0024), i.e. 165 forwards/s of 4 images. Does that carry 20 streams × 30
FPS = 600 images/s once real frames, letterboxing, decode, drawing and display are in the loop?

Setup:
- Videos: 20 clips of the CAVIAR shopping-centre set (Lisbon, corridor + shop-front views, 384×288, 25 FPS MPEG-2;
  homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/, CC BY-SA, credit EC-funded CAVIAR project IST 2001 37540), stored in
  `/home/ttuser/sdawle/yolo26_store_videos/`. The server serves ~35 KB/s per connection; downloads needed resume
  (`curl -C -`) in a retry loop.
- `demo/multistream_demo.py`: clips decoded into memory at start (≤ 1800 frames each, looped; video decode is NOT in
  the measured path). A pacing thread emits a tick every 1/30 s and builds 5 host batches of 4 letterboxed uint8
  frames; the main thread runs the YOLO26PerformantRunner trace with the same 2CQ pattern as the perf test (upload of
  batch k+1 on CQ1 overlaps trace k on CQ0, outputs to DRAM and read back on CQ1), waits on the read event of batch
  k only after trace k+1 is queued; a post thread decodes class 0 (person, conf > 0.35, one2one head, no NMS), draws
  a 5×4 mosaic + side panel (per-camera count, 10 s average, total, stream FPS, latency), serves MJPEG on :8090 and
  optionally records.

MEASURED (yolo26s, batch 4, chip 0, reset before each run; FPS = images processed / wall time; latency = tick
capture → mosaic published, median of the last 200 ticks):

| Run | Target | Achieved | Latency |
|---|---|---|---|
| first version, 300 frames/clip, no MJPEG | 600 images/s | 421.3 images/s = 21.1 FPS/stream | 74 ms |
| host path optimized (below), MJPEG on, 30 s | 600 | 599.5 = 30.0 FPS/stream | 48 ms |
| same, target 35 FPS/stream (above capacity), 30 s | 700 | **661.2 = 33.1 FPS/stream** (ceiling) | 112 ms |
| full clips (≤1800 frames), no recording, 30 s | 600 | 599.6 = 30.0 | 48 ms |
| + recording with cv2 mp4v in the post thread, 60 s | 600 | 504.9 = 25.2 | 289 ms |
| + recording via ffmpeg libx264 veryfast subprocess, 60 s | 600 | 548.7 = 27.4 | 273 ms |
| + ffmpeg ultrafast, 4 threads, nice 10, 60 s | 600 | **599.4 = 30.0** (36000 images, 1800/1800 frames recorded) | 90 ms |

The 661 images/s ceiling equals the perf test's 659.7 FPS at batch 4: with the host path fixed, the demo pipeline
costs no throughput; 20 × 30 FPS runs with ~10 % headroom.

Host-path defects found and fixed (MEASURED, `scratchpad/host_bench.py`, per batch of 4 unless noted):
- Letterbox wrote RGB per channel into the 8-channel input (strided numpy writes): 3.53 ms. Frames are converted once
  at load to RGB + zero 4th byte, resized as RGBA and written as one uint32 per pixel: 0.46 ms. (The 4th byte is 0;
  the stem weights for channels 3..7 are zero-padded anyway.)
- Decode cast all 80 class channels to fp32 before slicing: 2.61 ms → slice first: 1.87 ms.
- Per tick the host work (5 × prep + 5 × decode + tiles 0.7 + panel 2.0 + JPEG 6.0 ms) shares the Python GIL with the
  device loop; before the fixes it was ~35 ms per 33.3 ms tick, hence 21 FPS.
- Snapshot stored a reference to the live canvas: tiles of the next, partial tick overwrote it after the panel was
  drawn, so panel and tile counts disagreed in the saved image (live MJPEG unaffected). Now copies.
- Recording: cv2 mp4v encode of 2340×1152 in the post thread cost ~5 FPS/stream; x264 in a subprocess at default
  threading took all 16 cores; capped to 4 threads at low priority it no longer affects throughput.
- A "device capacity" readout (batch / loop time minus queue wait) read 1588 images/s while paced: the trace is
  asynchronous, so host loop time is not device time. Removed rather than shown wrong.

Visual check: `yolo26_demo_outputs/multistream/snapshot.jpg` (end of run) and `frame_25s.jpg` (from the recording):
people boxed in corridor and shop-front views, tile counts equal panel counts, mannequins (CAM 20) not counted.
Recording: `yolo26_demo_outputs/multistream/yolo26s_20streams_1chip.mp4` (60 s, 2340×1152, H.264).

Limits: per-frame detection counts, no tracking (a person is counted in every frame they appear in); clips are
25 FPS sources played at 30 FPS; decode of 20 live camera streams is not included (would take additional host cores
or hardware decode).

### YOLO26-0026 addendum (2026-10-27) — inference decoupled from rendering; per-tile FPS

- MEASURED, after adding a per-tile FPS readout: recording + live MJPEG, 60 s, two runs: 28.6 and 28.3 FPS per
  stream, latency 264–266 ms; same build without recording 30.0 FPS, 48 ms; recording without MJPEG 30.0 FPS, 49 ms.
  The earlier 30.0 FPS with recording + MJPEG (entry above) was at the edge of the host budget.
- ROOT CAUSE (READ, code): one post thread both decoded and rendered (draw 20 tiles + panel, JPEG encode, 8 MB copy
  to the recorder) and `post_q` was bounded, so when rendering took longer than a 33 ms tick the device loop blocked
  on `post_q.put`: the display throttled inference, all streams together (they are batched in lockstep). The latency
  climbs because frames wait in the queues. The 33 FPS/stream headroom (661 images/s) is chip-side; it does not
  cover extra host work.
- FIX: three threads. `post` decodes every batch (~2 ms) into a shared latest-result state (unbounded queue, never
  blocks the device loop); `render` draws only the newest completed tick, so a slow display skips display frames
  instead of slowing inference. Panel shows inference stream FPS and display FPS separately; each tile shows that
  stream's measured result rate over the last second (top right).
- MEASURED (recording + live MJPEG, 60 s): 599.3 and 599.4 images/s = 30.0 FPS per stream, display 30.0 FPS,
  latency 53–54 ms, 1799/1800 frames recorded. Per-tile readouts 30.0–30.1 FPS.

### YOLO26-0027 — How 20 streams map onto batch 4 (appended 2026-10-27)

Question: the panel says "batch 4". Are the 20 streams run 4 at a time, in sequence?

Answer: yes. The chip runs one batch-4 forward at a time; each tick is served by five of them back to back.

- READ (`demo/multistream_demo.py`, pacing thread, lines 321–342): every 1/30 s tick takes the current frame of each
  of the 20 streams and builds `nb = streams / B = 5` batches grouped by camera number: CAM 01–04, 05–08, 09–12,
  13–16, 17–20. Each batch is uploaded and run as one execution of the captured batch-4 trace; the 5 executions go
  through the chip in order. Within a batch the 4 images share one forward pass; across batches it is time-sharing.
- INFERRED from YOLO26-0024 (e2e batch 4, trace+2CQ, 659.7 FPS): one batch-4 forward ≈ 4 / 660 ≈ 6.1 ms; five per
  tick ≈ 30 ms of the 33.3 ms tick, i.e. ~10 % headroom. With two command queues, the upload of the next batch (CQ1)
  and the readback of the previous one overlap the current trace (CQ0), so those ~30 ms are almost all compute.
- INFERRED: within a tick, CAM 01–04 results complete first and CAM 17–20 about 4 × 6.1 ≈ 24 ms later. The median
  capture-to-display latency (MEASURED 53–54 ms, YOLO26-0026 addendum) includes this queueing.
- Why batch 4 (MEASURED, YOLO26-0024 / perf table, yolo26s e2e trace+2CQ, 1 chip): batch 1 418 FPS, batch 2 604 FPS,
  batch 4 660 FPS. At batch 1, 600 images/s does not fit; batch 2 has ~0.7 % headroom; batch 8 does not fit in
  device memory (batch 5 already fails, YOLO26-0024).

Panel changes in this period (requested by the user; READ, commits `d6b963ffa4d`, `4d788ee864d`):
- FPS readouts (tile top right and panel headline) are always green; the red/orange colour below 0.97 × target was
  removed. The "(target 600)" text was removed.
- The "avg 10 s" column (mean per-camera count over the last 300 results) and its history buffer were removed; the
  bars were widened into the freed space.

MEASURED (60 s, recording + live MJPEG, yolo26s batch 4, 1 chip, from each run's log):
- v2 (`yolo26_demo_outputs/multistream/yolo26s_20streams_1chip_v2.mp4`): 599.6 images/s = 30.0 FPS per stream,
  latency 54 ms median, 1799 frames, 59.97 s, 34.7 MB.
- v3 (`yolo26s_20streams_1chip_v3.mp4`, without the avg column): 36000 images in 60.0 s = 599.6 images/s = 30.0 FPS
  per stream, latency 54 ms median, display 30.5 FPS, 1799 frames, 59.97 s. Frame at 30 s: tile counts equal panel
  counts (43 total), per-tile 29.9–30.0 FPS.
- OPEN: v3 is 114.9 MB against 34.7 MB for v2 with the same encoder settings in code, length and frame count. Cause
  not investigated.

### YOLO26-0028 — 80 streams × 30 FPS on 4 chips (appended 2026-10-27)

Question: dp=4 batch 4/chip measured 2347–2404 FPS e2e (YOLO26-0025). Can the QB2 serve the same 20 clips on each of
the 4 chips, 80 streams × 30 FPS = 2400 images/s, as a live demo? Note the requirement sits at the top of that
measured range, and that test re-uploaded one pre-built input and did no decode.

Refuted approach — 4 independent processes, one per chip (MEASURED, `multistream_demo.py` unchanged, 30 s each,
`TT_VISIBLE_DEVICES=k` + p150 descriptor, logs confirm PCIe ids 0/1/2/3, i.e. 4 different chips):
- 4 processes: 166.9 / 165.6 / 166.0 / 175.7 images/s = 8.1–8.8 FPS per stream (one process alone: 599.6).
- 4 processes pinned to 4 cores each (`taskset`, `OMP_NUM_THREADS=2`): 161–168 images/s each — so not CPU
  oversubscription.
- 2 processes: 346.2 + 344.4 images/s.
- 1 process pinned to 4 cores: 599.6 images/s.
- INFERRED: separate processes share a host-side limit of ~660–690 images/s in total (≈ one chip's batch-4 rate);
  cause not investigated (candidates: per-process dispatch / host-memory locking in UMD). One of the 4 processes also
  aborted at exit (`terminate called without an active exception`, pthread_exit during interpreter shutdown) after
  logging its result.

dp=4 mesh, first version (READ: new `demo/multichip_demo.py`; one process, `open_mesh_device(1×4)`, input sharded with
`ShardTensorToMesh(dim=0)`, outputs gathered with `ConcatMeshToTensor(dim=0)`; prep / decode / device loop as in the
1-chip demo, render in a separate process): MEASURED 768.6 images/s = 9.6 FPS per stream (20 s).

Host cost per 16-image mesh pass (MEASURED, `scratchpad/bench/mc_host_bench.py`, mean of 10, ms): `pad.copy` of the
[16, 640, 640, 8] input 4.26; letterbox 16 frames 2.96; `from_torch` + mapper 0.02 (no copy); `to_torch` + composer
of the 6 tile outputs 6.91; full decode 7.03; blocking `from_device` of the 6 outputs 4.63; trace alone 5.86.
At 5 passes per 33.3 ms tick the host work (~15–19 ms per pass) is ~3× the budget, all under one Python lock.
Threads do not fix it (MEASURED letterbox 16: 2.73 ms with 1 thread, 1.77 with 2, 1.87 with 4; the threaded decode
test stalled and was abandoned).

Fixes (READ, `demo/multichip_demo.py`; `demo/multistream_demo.py` refactored to share them):
1. Output packing inside the trace (`pack_outputs`): per level copy box / score outputs to DRAM, slice the person
   logit, concat with the 4 box values, concat the levels, convert to row-major → one [1, 1, 33600, 5] bf16 DRAM
   tensor per chip (336 KB instead of ~8.6 MB of tile outputs). MEASURED (`mc_pack_bench.py`): trace + pack 6.083 ms
   vs 5.86 ms; `to_torch` of the packed result 0.070 ms vs 6.91 ms; max abs difference packed vs full outputs 0
   (random uint8 input, all 16 images, all anchors). Defects hit on the way: packing directly on the L1 outputs →
   `TT_FATAL bank_manager.cpp:514` (L1 allocation), then with DRAM outputs of the slice / concat only →
   `TT_THROW: Statically allocated dataflow buffers ... clash with L1 buffers` (circular buffers vs the model's L1
   outputs); fixed by copying the 6 outputs to DRAM first (the device loop did this copy anyway).
2. Host prep in `--prep-workers` (2) processes writing into a shared-memory ring of 16 mesh input buffers
   (2 × passes per tick + 6). The padding is written once at creation, so the per-pass `pad.copy` is gone. A slot
   goes back to the free list when its pass reaches decode: by then its read-back event, recorded on CQ1 after its
   upload, has completed.
3. Decode of the packed tensor (`decode_packed`) in the main process; render, MJPEG and recording in their own
   process (newest tick only).
4. `multistream_demo.py`: device loop moved into `run_device_loop` (shared), `load_clips(rgba=False)`,
   `decode_people(mesh_composer=)`. MEASURED regression, 1 chip, 20 s: 599.6 images/s = 30.0 FPS per stream.

Results (MEASURED, yolo26s, batch 4 per chip, 4 chips, 640×640, CAVIAR 20 clips × 4 at start offsets of 0, ¼, ½, ¾
clip; one run each):
- 20 s, no recording: 2397.8 images/s = 30.0 FPS per stream, median capture-to-display latency 57 ms, 600 frames
  displayed.
- 20 s, `--fps 40` (ceiling): 2598.5 images/s = 32.5 FPS per stream (~8 % above 2400). That is above the 2377 of the
  YOLO26-0025 perf test; INFERRED cause: the packed read-back (336 KB vs ~8.6 MB per chip per pass).
- 60 s with recording + live MJPEG: 143984 images in 60.0 s = 2399.2 images/s = 30.0 FPS per stream, latency 61 ms
  median, 1799 frames, 59.97 s, 2340×1152 H.264, 107.5 MB
  (`yolo26_demo_outputs/multistream/yolo26s_80streams_4chips.mp4`).
- Visual check (frame at 30 s and end snapshot): tile counts equal panel cells, per-chip sums equal "People now"
  (43 + 58 + 58 + 50 = 209), per-tile 29.9–30.3 FPS.

Defect, MEASURED: the recording run hung in `capture_trace_2cq` twice in a row (attempt 1 03:31:57, after 4 clean
runs of the same code; attempt 2 03:42:49, after `tt-smi -r` with its output discarded). Attempt 3 passed after a
reset whose output showed "Re-initializing boards after reset". tt-triage was run on the attempt-1 hang
(`yolo26_demo_outputs/multistream/triage_capture_hang.txt`, last 60 lines): it reports cores on all 4 devices as
"halted by triage but no longer halted - core was broken during triage", so it is inconclusive. Same symptom as
YOLO26-0021/0022/0025; status stays OPEN.

Limits: as YOLO26-0026 (per-frame counts, no tracking, 25 FPS clips played at 30). 80 streams are 20 clips × 4
offsets, not 80 cameras. The 4 chips run in lockstep (one mesh trace), so every chip shows the same FPS.

### YOLO26-0029 — latency per camera group, and cutting it (appended 2026-10-27)

Question: with 5 batches of 4 per tick (YOLO26-0027), what is the latency of CAM 05-08 vs CAM 17-20?

METHOD: instrumented scratch copy of the demo (`scratchpad/bench/make_latency_copy.py`): per pass, time from the
tick's capture to results handed to decode and to decoded; on-screen = capture to the tick's mosaic drawn. 1 chip,
20 streams, 30 FPS, 30 s, first 2 s skipped, 840 ticks.

MEASURED baseline (code as committed in 2c7aba4e2f2), decoded median: CAM 01-04 19.7 ms, 05-08 25.4, 09-12 29.0,
13-16 41.1, 17-20 46.9; on screen 50.9 ms (p95 52.3). All cameras appear on screen together: the renderer draws a
tick only when its 5 batches are in, so on-screen latency is that of the last group for every camera.

READ (device loop): results of a batch were handed over one iteration later, after the next batch's input was
queued. At the end of a tick the next input is the next tick's first batch, so batch 4 waited for the next tick
(the 12 ms step between batches 3 and 4).

Attempts:
1. REFUTED: a waiter thread calling `event_synchronize` per read, plus reading first when the next input is not
   ready. MEASURED: every group 107-136 ms, on screen 138 ms (throughput kept, 597.9 images/s); the input queue
   stayed full. READ: `event_synchronize` releases the GIL (nanobind `gil_scoped_release`), so the GIL is not the
   reason; cause not found, dropped.
2. Tick-boundary flush, no thread: when the next input is not ready, read the current batch first, wait for both
   outstanding reads, hand both over, then wait for input (the device would idle anyway). MEASURED: groups 23.9 /
   28.3 / 33.9 / 41.0 / 48.7 ms, on screen 51.6. Little gain. Timeline (MEASURED, per batch, median ms after capture)
   showed why: the prep thread produced batches only at 3.4 / 7.7 / 12.1 / 18.7 / 24.6 ms (≈ 5 ms per batch of 4
   under the GIL against ~1.5 ms alone), so the device waited for input.
3. Prep in worker processes over a shared-memory ring (the 4-chip design, YOLO26-0028). MEASURED: groups 20.2 /
   25.0 / 31.7 / 40.5 / 45.0, on screen 48.7. Timeline: inputs ready at 1.7 ms but handed to the device loop at
   8.7 ms; the main process was busy decoding the previous tick (~2 ms per batch) and rendering it (tiles, panel,
   JPEG, recorder ~10 ms) under the same GIL.
4. Both demos on one shared pipeline (`multistream_demo.run_pipeline`): prep workers, on-device output packing
   (`pack_outputs`, decode ~0.1 ms), decode in the main process, render in its own process. `multichip_demo.py`
   keeps only its mesh setup and mosaic layout.

MEASURED result, 1 chip (60 s recording + live MJPEG, `yolo26s_20streams_1chip_v4.mp4`): 599.9 images/s =
30.0 FPS per stream; capture-to-results median CAM 01-04 17.3 ms, 05-08 19.7, 09-12 28.8, 13-16 32.3, 17-20 39.6;
on screen 43.4 ms; 1800 frames.

4 chips (MEASURED, 30 s timelines then the 60 s recording `yolo26s_80streams_4chips_v2.mp4`): 2399.1 images/s =
30.0 FPS per stream; capture-to-results CAM 01-04 32.7, 05-08 34.3, 09-12 49.4, 13-16 51.3, 17-20 55.4 ms (chip 1
cameras; the 4 chips run in lockstep); on screen 61.0 ms (57.6 and 57.8 in two 30 s runs; 57-61 ms before the change).
Timeline: 16-image prep passes take ~6-7 ms per worker in the demo (2.7 ms alone) and the device is still busy
with the previous tick when the next tick's first pass is ready (start ~11 ms after it is ready).
- REFUTED: 5 prep workers: all passes ready only at ~15 ms, on screen 61.1 ms.
- REFUTED: `cv2.setNumThreads(1)` in the workers (2 workers: on screen 56.8 ms; 3 workers: 57.4 ms): no change;
  reverted.
- INFERRED: at 2400 of ~2600 images/s (~92 %) any start delay carries over to the next tick, so latency is set by
  queueing; reducing device time per pass (e.g. the 8-channel uint8 input, 5/8 padding) is the remaining lever.

Also fixed: the end-of-run summary measured elapsed time after the prep workers had shut down (585 images/s
reported for 18000 images in 30 s); elapsed is now taken when the device loop ends. The per-pass latency medians
are now part of the demos' summary log.

### YOLO26-0030 — batch 8 / 16 / 20 on one chip with DRAM (appended 2026-10-27, branch `sdawle/tvardhineni/yolov26_dram_batch`)

Question: batch 4 is the L1 limit (YOLO26-0024). Moving the large activations to DRAM, can batch 16 and 20 run, and
does 20 streams × 30 FPS still hold?

METHOD unless noted: `scratchpad/bench/quickb.py <variant> <batch>` (PCC per image vs Ultralytics one2one on bus /
zidane and mirrors, then traced model-only time, mean of 200), yolo26s, 1 chip, chip reset before each run.

MEASURED baseline: batch 8 with the batch-4 code fails at the stem conv (`TT_FATAL Out of Memory: ... 52459520 B L1
buffer`, the 8 × 320 × 320 × 32 bf16 stem output).

Attempt 1 — REFUTED, layer-level DRAM (saved as `scratchpad/bench/ttnn_yolo26_layer_dram_v7.py`): above batch 4,
every layer whose input exceeds a pixel threshold keeps activations in DRAM: input typecast / re-page in DRAM, 3×3
convs with `Conv2dDRAMSliceHeight` (auto slices), adds / concats / upsample DRAM interleaved. Defects on the way:
- 1×1 convs ignore the DRAM slice config (they run as a matmul) and wrote their whole output to L1 (`dataflow buffers
  ... clash with L1 buffers`, model.2.cv2) → ran them in groups of 4 images.
- model.5 (80×80) and the 40×40 head depthwise conv: circular buffers 1668544 B > 1572864 B → threshold lowered to
  batch-4-at-40×40.
- model.9.cv2: at batch 8 the rows / channels ratio flips the sharding heuristic to height sharding, which puts the
  whole 1×1 1024 → 512 weight (1 MB) on every core → layout now chosen as at batch 4 (`min(batch, 4)`; kept).
- The neck's 80×80 upsample / concat in L1 → upsample in DRAM at large sizes.
MEASURED: batch 8 runs, PCC cls 0.99780–0.99848, but 37.397 ms per batch = 213.9 FPS (batch 4: 683.1). Device profile
(tracy, kernel sum 37.2 ms): reshape 10.9 ms (4.1 ms re-paging the DRAM input to 8-channel sticks; ~0.16–0.57 ms per
DRAM-sliced conv, which reshapes [1, 1, NHW, C] tile ↔ [1, B·H, W, C] with W padded to tiles), interleaved→sharded
8.4 ms (6.2 ms loading the stem input as 16-byte sticks), conv 5.0 ms, matmul 3.5, concat 2.8, slice 2.6.

Where batching can gain (MEASURED, tracy, kernel time by output size per image):
| stage | batch 1 µs/img | batch 4 µs/img | share at batch 4 |
|---|---|---|---|
| 320×320 | 80 | 67 | 4.7 % |
| 160×160 | 254 | 170 | 12.1 % |
| 80×80 | 476 | 340 | 24.1 % |
| 40×40 | 810 | 338 | 24.0 % |
| 20×20 | 713 | 496 | 35.2 % |

Attempt 2 — hybrid grouped forward (`TtYOLO26._forward_grouped`, used when the uint8 batch > `MAX_L1_BATCH` = 4;
batch ≤ 4 unchanged): every image is independent, so the stages above 20×20 (input prep, backbone to model.7, the
neck's 40×40 / 80×80 path, the 80×80 / 40×40 heads) run in groups of 4 with the batch-4 programs entirely in L1, and
the 20×20 stages (model.8, SPPF, C2PSA, model.22, 20×20 head) in groups of `LOW_RES_BATCH` = 8. Stage outputs that
cross between the two (P3, P4, model.7 / model.20 outputs, the C2PSA output, head outputs) wait in DRAM. The model's
input stays in DRAM (`input_memory_config` returns DRAM above batch 4); each group is sharded on its way in.
- Full-batch 20×20 at 16 or 20: SPPF's first conv `dataflow buffers ... clash with L1 buffers` → groups of 8.
- Batch 20 trace: 8404992 B > the 8 MB trace region → demos' `trace_region_size` 8 MB → 16 MB.

MEASURED (model-only, traced; PCC min over all images in the batch):
| batch | ms per batch | FPS | vs batch 4 | min cls / box PCC |
|---|---|---|---|---|
| 4 (unchanged path) | 5.856 | 683.1 | — | 0.99790 / 0.99898 |
| 8 | 10.939 (10.927) | 731.3 (732.1) | +7.1 % | 0.99780 / 0.99896 |
| 16 | 21.924 | 729.8 | +6.8 % | 0.99780 / 0.99896 |
| 20 | 27.917 | 716.4 | +4.9 % | 0.99780 / 0.99896 |
Batch 20 is lower because its third 20×20 group has only 4 images. `tests/pcc` 17/17.

20 streams × 30 FPS, 1-chip demo (`multistream_demo.py --batch-size B`, 30 s, no recording; MEASURED):
- batch 20 (one pass per tick): 17960 images = 598.4 images/s = 29.9 FPS per stream; capture-to-results 108.3 ms,
  on screen 110.7 ms.
- batch 4 (5 passes per tick): 18000 images = 599.9 images/s = 30.0 FPS; CAM 01-04 14.3 ms … CAM 17-20 38.7 ms,
  on screen 42.6 ms.
- Ceiling (`--fps 40`, 20 s): batch 20 682.4 images/s (34.1 FPS per stream), batch 4 650.4 (32.5).
INFERRED: at a paced 30 FPS a tick's 20 frames exist only when the tick starts, so the 20-image upload (~65 MB of the
8-channel input) cannot overlap the previous pass; upload + a ~28 ms pass fill the 33.3 ms tick, the queue fills and
latency rises to ~3 ticks. The +5 % ceiling only shows when inputs queue ahead (as with `--fps 40`). Batch 4 keeps
30 FPS with less than half the latency; batch 8 (if the stream count is a multiple of 8) would be the next to try.

### YOLO26-0031 — 1-chip demo latency, second round (appended 2026-10-27)

Scope: 1 chip only (user decision during this entry). METHOD: `multistream_demo.py`, yolo26s, batch 4, 20 streams ×
30 FPS, 30 s, chip reset before each run; per-pass capture-to-results medians from the demo's summary log; recording
on unless noted.

Changes and MEASURED effect (capture-to-results by camera group; on screen):
1. Staggered capture (`prep_worker`): pass b of a tick is captured b / 5 of a tick after the tick starts, like
   free-running cameras; `--synchronized-capture` keeps the old behaviour. With an unchanged device loop: no effect
   on throughput.
2. REFUTED: device loop with uploads alone on CQ1 and trace + read on CQ0 (read the packed output directly). 586.4
   images/s = 29.3 FPS, every group 118-121 ms in both capture modes: a read on CQ0 holds CQ0 until the host has
   taken the data (~0.7 ms per pass more), which saturates the chip. Reverted.
3. Bounded wait for the next batch (`wait_next` = one capture interval, 6.7 ms, when staggered): the read stays
   queued after the next upload, so uploads keep overlapping traces. Staggered: every group 16.3-16.6 ms, 30.0 FPS.
   Synchronized: 14.4 → 35.8 ms by group (as before).
4. Hand over the previous batch as soon as the current trace is queued, not after the next input arrives (READ: a
   result ready ~9 ms after capture waited in the loop until the next batch had arrived and been uploaded).
   Staggered: CAM 01-04 10.8, 05-08 10.6, 09-12 10.8, 13-16 11.4, 17-20 11.1 ms; synchronized 16.9 → 36.3 ms.
5. Renderer: tiles update per pass as results arrive (no waiting for a whole tick); the mosaic is composed and
   published whenever new results are in (merging those queued meanwhile) and recorded on a fixed 30 FPS clock.
   On-screen latency is now measured per result, from its capture to the publication (after JPEG encode) of the
   first mosaic showing it. Earlier on-screen numbers (YOLO26-0026..0030) were taken per tick before the encode and
   from the last group's capture, so they are not comparable one-to-one.
   - Intermediate, fixed 60 Hz composition, measured before encode: 19.6 ms (not comparable, superseded).
   - Final (`scratchpad/lat_v6.*`): 18000 images in 30.0 s = 599.9 images/s = 30.0 FPS; capture-to-results
     10.7-11.5 ms by group; on screen 23.2 ms; 924 frames recorded in 30.8 s (real time); live view composed
     ~121 times/s.
6. Prep split per image across workers (every worker letterboxes part of every pass): 1 chip, 4 workers 16.5 /
   24.8 ms, 2 workers 16.6 / 24.1 ms vs 16.7 / 25.9 ms with whole passes per worker (before change 4): no gain on
   1 chip; reverted to whole passes per worker (kept as `scratchpad/bench/ms_split_prep.py`).

Also: start-up now raises if the render process dies before it is ready (a renderer that failed to bind port 8090,
right after a killed run, left the main process waiting until the 540 s timeout).

Summary (1 chip, 20 × 30 FPS, MEASURED): before this entry CAM 01-04 17.3 … CAM 17-20 39.6 ms, on screen 43.4 ms
(old method); after, 10.7-11.5 ms for every camera, on screen 23.2 ms (per-result method). INFERRED remaining
budget: ~6.1 ms trace + upload / copy / read / decode ≈ 4.8 ms to results; on screen + ~12 ms for transfer to the
renderer, tile + panel drawing (~2 ms) and the 2340×1152 JPEG encode (~6 ms).
Not re-measured on 4 chips with these changes (`multichip_demo.py` shares the pipeline; earlier this entry, 4 chips
with changes 1-3: 26.5 ms to results, 36.4 ms on screen).

### YOLO26-0032 — batch 20, one pass per tick: latency (appended 2026-10-27)

Goal (user): capture and run all 20 cameras at once (batch 20, grouped DRAM path of YOLO26-0030) with the least
latency; not to beat batch 4. METHOD: `multistream_demo.py --batch-size 20`, 1 chip, 20 × 30 FPS, chip reset before
each run; timelines from instrumented scratch copies (`scratchpad/bench/ms_tl3.py`, `ms_tl4.py`).

MEASURED start (code of YOLO26-0031): 599.5 images/s = 30.0 FPS, capture-to-results 61.7 ms, on screen 66.0 ms
(108 ms in YOLO26-0030 before the 0031 changes). Lower load made it worse (25 FPS 68.6 ms, 28 FPS 63.9 ms): not
saturation. Timeline: frames letterboxed at 4.1 ms, picked up by the device loop at 16.9 ms, trace queued at 22.3 ms,
handed over at 59.3 ms.

1. `from_torch` copied the 65.5 MB ring slot on a single device (10.85 ms, MEASURED `ft_bench.py`; 12.7 ms in the
   demo under the GIL), while with a mesh mapper it aliases the buffer (`ShardTensorToMesh` on the 1×1 device:
   0.10 ms, later writes visible). The demo now always passes a mesh mapper (safe: a slot is reused only after its
   upload completed). MEASURED: batch 20 49.6 / 53.9 ms; batch 4 9.5 / 12.6 ms (port off).
2. REFUTED: uploader thread on CQ1 + main thread on CQ0 (events passed through queues): 445.6 images/s =
   22.3 FPS, 217 ms. Issuing ttnn commands from two threads serializes / contends (cf. YOLO26-0029 attempt 1).
3. Single-pass device loop (`run_device_loop_single`, used when there is one pass per tick): outputs read on CQ0
   right after the trace (one ~0.7 ms read per tick), so CQ1 carries only uploads; per tick: queue copy + trace +
   read, wait for the next batch and upload it (overlapping the end of the pass), then hand this batch over.
   (Before, the read sat on CQ1 behind the next tick's upload and the hand-over came after the next trace.)
   MEASURED: 46.0 / 50.4 ms.
4. Frame prep split per image over the workers (YOLO26-0031 item 6, re-applied; default 4 workers): with one pass
   per tick one worker letterboxed all 20 frames (4.2 ms). MEASURED: batch 20 44.4 / 48.7 ms; batch 4 9.3 / 12.5 ms.
5. Defect found: with 4 prep workers the live view failed to bind port 8090/8091 ("Address already in use").
   READ / MEASURED: every process importing the Tracy-enabled ttnn opens a Tracy listener on the first free port from
   8086 (`TracyProfiler.cpp`), so main + 4 workers + renderer took 8086-8091 (a spawned child seen listening on
   *:8090 with Tracy threads). Fix: each spawned process gets its own `TRACY_PORT` (18086+) at start.

Timeline after the changes (MEASURED, median after capture): letterboxed 2.8 ms, loop has it 3.6, upload written
10.0 (6.4 ms host write of 65.5 MB; 4.63 ms = 14.2 GB/s alone, `up_bench.py`), copy + trace + read queued 10.5, read
done 43.4 (≈ 33 ms on device vs 27.9 ms model-only; staging → input copy 0.33 ms, `copy_bench.py`; INFERRED: the
next tick's upload overlapping the pass), decoded 44.5.

Final (MEASURED, 60 s, live MJPEG on, no recording):
| | capture → results | on screen | throughput |
|---|---|---|---|
| batch 20 (1 pass per tick) | 44.0 ms, all 20 cameras | 52.4 ms | 599.9 images/s = 30.0 FPS |
| batch 4 (5 staggered passes) | 9.4 ms, every group | 20.7 ms | 600.0 images/s = 30.0 FPS |
Snapshot (batch 20) checked: tile counts equal panel rows, 30.0 FPS on every tile.
Remaining for batch 20 (INFERRED): upload (5/8 of the 8-channel input is padding) and the ~31 ms pass; going lower
would need a smaller input or splitting the trace so groups start before the whole batch has been uploaded.

### YOLO26-0033 — how batch 4 reached ~9.4 ms per camera group (appended 2026-10-27; summary of YOLO26-0026..0032)

Question: how did capture-to-results for every camera group at batch 4 get to ~9.4 ms, and are the groups run
asynchronously?

Answer: the host side is an asynchronous pipeline, but the chip still runs one batch of 4 at a time. Device work is
unchanged (~6 ms per batch of 4, ~30 ms of each 33.3 ms tick, YOLO26-0027). The latency fell because a group no
longer waits behind other groups or behind host work.

What one group's ~9.4 ms consists of (INFERRED from the timelines of YOLO26-0029 / 0031 / 0032): a group of 4
cameras captures at its own phase of the tick; letterboxing in 4 worker processes ~1 ms; upload ~1 ms (on CQ1,
overlapping the previous group's trace); trace ~6 ms; read-back + decode ~1 ms. The next group arrives 6.7 ms later,
about when the chip finishes, so nothing queues and every group sees the same latency.

What runs concurrently (READ, `demo/multistream_demo.py`):
| unit | does |
|---|---|
| 4 prep worker processes | letterbox frames into a shared-memory ring, each a share of every pass |
| main process | feeds ring slots (aliased, no copy) to the device loop; device loop; decode of the packed outputs |
| CQ1 | uploads the next batch |
| CQ0 | copy to the trace input, trace, while CQ1 uploads |
| render process | draws tiles per pass, composes / publishes the live view, records on a fixed 30 FPS clock |

Steps, in order (MEASURED, 1 chip, 20 × 30 FPS, capture-to-results by camera group; on screen):
| step | change | entry | CAM 01-04 | CAM 17-20 | on screen |
|---|---|---|---|---|---|
| start | one process: prep, decode, drawing and device loop share the GIL | 0029 baseline | 19.7 | 46.9 | 50.9 |
| 1 | prep in worker processes over a shared-memory ring; outputs packed on device (decode ~7 → ~0.1 ms); render process | 0029 | 17.3 | 39.6 | 43.4 |
| 2 | staggered capture: each group at its own phase (0 / 6.7 / 13.3 / 20 / 26.7 ms) instead of all 20 at once | 0031 | 16.4 | 16.4 | — |
| 3 | bounded wait for the next batch, so its upload is queued before the current read (keeps 30.0 FPS) | 0031 | 16.4 | 16.4 | — |
| 4 | early hand-over: a group is handed over once read, not after the next group arrived and was uploaded | 0031 | 10.8 | 11.1 | — |
| 5 | tiles update per group as results arrive; mosaic composed on arrival; recording on a 30 FPS clock | 0031 | — | — | 23.2 |
| 6 | `from_torch` aliases the ring slot (mesh mapper) instead of copying 13 MB per batch under the GIL | 0032 | 9.5 | 9.4 | 12.6 (no live view) |
| 7 | each batch's letterboxing split across 4 workers | 0032 | 9.4 | 9.6 | 21.4 (live view on) |
Throughput stayed at 600 images/s (30.0 FPS per stream) throughout. Latest figures are from the 60 s recording
`yolo26_demo_outputs/multistream/yolo26s_20streams_1chip_batch4_lowlatency.mp4` (CAM 01-04 9.5, 05-08 9.4, 09-12
9.8, 13-16 9.9, 17-20 9.6 ms; on screen 21.4 ms; 600.0 images/s).

The two largest effects: staggered capture (step 2) removed queueing between groups (all 20 captured together put
5 batches on the chip at once, the fifth waiting ~24 ms); early hand-over (step 4) stopped finished results waiting
in the loop for the next group's input.

Caveats:
- Staggered capture changes the scenario to free-running cameras (the realistic case). With truly synchronized
  cameras (`--synchronized-capture`) groups queue again: 16.9 → 36.3 ms by group (MEASURED in 0031 after step 4,
  before steps 6-7).
- On-screen latency was measured differently before step 5 (before the JPEG encode, from the last group's capture),
  which made the old figure lower than it would be now, so 50.9 → 21.4 ms is not overstated.
- The ~12 ms between results and screen are the transfer to the render process, tile / panel drawing and the ~6 ms
  JPEG encode of the 2340×1152 mosaic.

### YOLO26-0034 — batch 20: before / after, and how it combines L1 and DRAM (appended 2026-10-27)

Before / after (MEASURED, 1 chip, 20 streams × 30 FPS, batch 20 = all 20 cameras in one pass; capture-to-results
is the same for every camera):
| cameras (batch in tick) | before (YOLO26-0030) | after (60 s recording, YOLO26-0032 code) |
|---|---|---|
| CAM 01-20 (one pass) | 108.3 ms | 44.6 ms |
| on screen (all cameras) | 110.7 ms | 53.0 ms |
| throughput | 598.4 images/s (29.9 FPS) | 599.9 images/s (30.0 FPS) |
Recording: `yolo26_demo_outputs/multistream/yolo26s_20streams_1chip_batch20.mp4` (1830 frames, 61.0 s).
Path: 108 → 61.7 ms from the shared changes of YOLO26-0031 (early hand-over, per-pass rendering); 61.7 → 44.6 ms
from YOLO26-0032 (no 65 MB `from_torch` copy, one-pass-per-tick device loop, prep split over 4 workers). The before
run was saturated (29.9 FPS, frames queued); the after run holds 30.0 FPS without queueing. On-screen method differs
as in YOLO26-0033 (before: per tick, before the JPEG encode; after: per result, after it), so the drop is not
overstated.

How batch 20 combines L1 and DRAM (READ, `tt/ttnn_yolo26.py`, `TtYOLO26._forward_grouped`, used when the uint8
batch > `MAX_L1_BATCH` = 4): no layer computes from DRAM. Every layer runs in L1 with a program sized for a group;
DRAM only holds full-batch tensors between stages.
- Input: `input_memory_config` returns DRAM above batch 4, so the 20-image uint8 input (65.5 MB) stays in DRAM; each
  group of 4 is sliced out and sharded into L1 by the batch-4 `preprocess` on its way in.
- Stages above 20×20, in 5 groups of 4 with the batch-4 L1 programs: backbone stem .. model.7, neck model.13 /
  model.16 / model.17 / model.19 / model.20 (40×40 and 80×80 path), heads at 80×80 and 40×40. A group's L1 use is
  that of batch 4.
- 20×20 stages, in groups of `LOW_RES_BATCH` = 8 (8, 8, 4 for batch 20): model.8, SPPF, C2PSA (model.10),
  model.22, 20×20 head. Larger groups keep the cores busier here (20×20 was 35 % of batch-4 time, YOLO26-0030);
  8 is the largest that fits (at 16 SPPF's first conv clashes with L1).
- Parked in DRAM between the two: P3 (80×80), P4 (40×40), model.7 output, C2PSA output, model.20 output, and the
  per-group head outputs; each group writes its slice and the slices are concatenated in DRAM when a stage needs
  the batch (or at the end).
- Order of one pass: groups 1-5 backbone → 20×20 in groups of 8 → groups 1-5 neck + 80×80 / 40×40 heads → 20×20
  model.22 + head in groups of 8 → outputs joined.
- The layer sharding is chosen as at batch 4 (`min(batch, 4)` rows) so larger groups keep the validated layouts.
- REFUTED earlier (YOLO26-0030 attempt 1): computing the large layers from DRAM (DRAM-sliced convs, DRAM adds /
  concats): batch 8 at 213.9 FPS, dominated by reshapes and DRAM ↔ L1 copies. Grouping: batch 8 731.3, batch 20
  716.4 FPS model-only (batch 4: 683.1), PCC unchanged.

### YOLO26-0035 — the on-screen gap, and batch 4 with synchronized capture (appended 2026-10-27)

Synchronized capture with the current code (MEASURED, `multistream_demo.py --batch-size 4 --synchronized-capture`,
1 chip, 20 × 30 FPS, 60 s, live view on): 600.0 images/s = 30.0 FPS; capture-to-results CAM 01-04 11.5, 05-08 17.0,
09-12 22.6, 13-16 28.3, 17-20 35.3 ms; on screen 34.2 ms (median over all cameras).
| cameras (set in tick) | own capture time (staggered, default) | same capture time (synchronized) | batch 20 (one pass) |
|---|---|---|---|
| CAM 01-04 | 9.5 ms | 11.5 ms | 44.6 ms |
| CAM 05-08 | 9.4 ms | 17.0 ms | 44.6 ms |
| CAM 09-12 | 9.8 ms | 22.6 ms | 44.6 ms |
| CAM 13-16 | 9.9 ms | 28.3 ms | 44.6 ms |
| CAM 17-20 | 9.6 ms | 35.3 ms | 44.6 ms |
| on screen (median) | 21.4 ms | 34.2 ms | 53.0 ms |
| throughput | 600.0 images/s | 600.0 images/s | 599.9 images/s |
INFERRED: with synchronized capture each set waits for the sets before it, ~5.7 ms (one pass) per set; set 1 is
~2 ms above staggered because the 5 sets' prep and uploads coincide. The on-screen median sits near the middle sets
because it is taken over all cameras' results.

Why on screen is later than capture-to-results, and by different amounts. After a set's counts are decoded, the
render process must receive them, draw them and encode a new live-view image (this is where "on screen" is taken):
| step after results | time | source |
|---|---|---|
| hand-over to the render process (multiprocessing queue) | ~0.5 ms | INFERRED |
| draw the updated tiles | ~0.2 ms per 4 tiles, ~0.7 ms for 20 | MEASURED, YOLO26-0026 host bench |
| redraw the side panel | ~2 ms | MEASURED, YOLO26-0026 |
| JPEG-encode the 2340×1152 mosaic | ~6 ms | MEASURED, YOLO26-0026 |
| wait while the renderer finishes a previous image | 0 to ~8 ms | INFERRED |
The first four are ~9 ms every time; the wait depends on how often results arrive:
| configuration | results | on screen | gap | why (INFERRED) |
|---|---|---|---|---|
| batch 20 | 44.6 ms | 53.0 ms | ~8.4 ms | one update per tick; the renderer is idle when results arrive: one draw + encode, no wait |
| batch 4, staggered | ~9.6 ms | 21.4 ms | ~12 ms | a set every 6.7 ms vs ~8 ms per draw + encode: the renderer is nearly always busy (~125 images/s composed), a set waits on average half a compose (~4 ms) and then its own |
| batch 4, synchronized | 11.5 → 35.3 ms | 34.2 ms | ~12 ms vs the middle sets | as staggered; the 34.2 ms median over all cameras sits near the middle sets (22.6 + ~12) |
So the gap is ~9 ms of fixed display work plus up to one compose of waiting when the renderer is busy; batch 20 has
the smaller gap only because it updates the screen 5× less often.
- "On screen" ends when the JPEG is ready to send; network transfer, the browser's decode and the monitor refresh are
  not included.
- To shrink it: the JPEG encode first (smaller live-view image or lower quality), then the panel (redraw it less
  often than the tiles); both cut the fixed part and the waiting.

### YOLO26-0036 — display overhead: parallel strip JPEG and panel at 10 Hz (appended 2026-10-27)

Chosen from the options in YOLO26-0035 (user): 3a (encode in parallel) and 2 (panel less often). The recording is
unaffected by both (it takes the raw full-resolution mosaic and encodes with ffmpeg).

1. `StripJpeg` (`demo/multistream_demo.py`): the live-view JPEG is encoded as 4 horizontal strips on 4 threads
   (`cv2.imencode` releases the GIL) and joined into one baseline JPEG: each strip is a whole number of 16-row MCU
   rows (4:2:0), all strips share quality 80 and the standard Huffman tables (no optimize), restart markers between
   strips reset the DC prediction, the first strip's header gets the full height and a DRI segment. MEASURED
   (`scratchpad/bench/strip_check.py`, the 2340×1152 demo snapshot): the merged JPEG decodes to the same pixels as a
   single-pass encode (max difference 0), 495 vs 495 KB; single-thread 3.85 ms, 2 strips 2.03 ms, 4 strips 1.04 ms,
   6 strips 0.75 ms → 4 strips used.
2. Side panel redrawn at most every 100 ms (was every update); tiles still update on every result. Trade-off seen in
   the snapshot: a panel row can lag its tile by up to 100 ms (e.g. CAM 16 tile 1, panel 0).

MEASURED (1 chip, 20 × 30 FPS, 60 s, live view on; capture → results unchanged):
| | bs=4, 20 streams (staggered) | bs=4, 20 streams (synchronized) | bs=20, 20 streams (one pass) |
|---|---|---|---|
| capture → results | 9.4 ms every set | 11.2 / 16.8 / 22.5 / 28.1 / 35.0 ms by set | 44.4 ms |
| on screen before (YOLO26-0035) | 21.4 ms | 34.2 ms | 53.0 ms |
| on screen now | 11.3 ms | 24.6 ms | 48.0 ms |
| throughput | 600.0 images/s | 600.0 images/s | 599.9 images/s |
(on screen = median over all cameras' results, capture → first published live-view frame showing them.)
INFERRED: staggered bs4 gap 12 → ~2 ms: with ~1 ms encode and ~0.2 ms tiles a compose is shorter than the 6.7 ms
between sets, so results no longer wait for a previous compose; bs20 gap 8.4 → 3.6 ms (one compose per tick, encode
~6 → ~1 ms, panel skipped on most ticks... at 30 updates/s every third tick redraws it).

### YOLO26-0037 — correction to YOLO26-0036: panel throttle (appended 2026-10-27)

Defect (MEASURED): with the panel capped at 10 redraws per second (YOLO26-0036 item 2), the 60 s batch-4 recording
`yolo26s_20streams_1chip_batch4_v2.mp4` showed 6 of 20 panel rows different from their tiles in the frame at 30 s
(CAM 02 tile 0 / panel 1, 06 0/1, 08 1/0, 13 3/2, 19 2/4, 20 0/1). The recording captures 30 frames a second, so
about 2 of every 3 recorded frames carried a panel up to 100 ms old. Before 0036 the panel was redrawn on every
update (30/s originally, ~120-150/s after YOLO26-0031) and always matched. The cap was chosen for readability and
the recording was not considered; it saved only ~2 ms per update (most of 0036's gain is the strip JPEG).

Fix (`Display.panel(stats, counts, rows)`): the header (rates, latency, total) is redrawn on every update; of the
20 camera rows only those whose streams were updated since the last call, or all rows when the bar scale (largest
count) changes. `multichip_demo.Display.panel` takes `rows` and still redraws everything.

MEASURED (1 chip, batch 4, staggered, 20 × 30 FPS, 60 s, live view + recording,
`yolo26_demo_outputs/multistream/yolo26s_20streams_1chip_batch4_v3.mp4`, 1831 frames, 61.0 s): 600.0 images/s;
capture-to-results 9.4 / 9.4 / 10.0 / 9.8 / 9.5 ms by group; on screen 12.8 ms (11.5 ms with the 10 Hz panel, v2
clip; 21.4 ms before YOLO26-0036). Frames at 15 / 30 / 45 s: 20/20 panel rows equal their tiles in each, totals
52 / 42 / 62 equal the tile sums.

### YOLO26-0038 — bs=4 vs bs=20 with all cameras captured together, display breakdown (appended 2026-10-28)

Like-for-like comparison: in both columns all 20 cameras are captured at the same instant (bs=4 with
`--synchronized-capture`, bs=20 by construction). Code of YOLO26-0037; 1 chip, 20 streams × 30 FPS, 60 s, live
view on; 600.0 / 599.9 images/s.
| cameras (set in tick) | bs=4, 20 streams latency | bs=20, 20 streams (one pass) latency |
|---|---|---|
| **capture → results** (MEASURED, median) | | |
| CAM 01-04 (1st) | 11.2 ms | 44.6 ms |
| CAM 05-08 (2nd) | 16.9 ms | 44.6 ms |
| CAM 09-12 (3rd) | 22.5 ms | 44.6 ms |
| CAM 13-16 (4th) | 28.2 ms | 44.6 ms |
| CAM 17-20 (5th) | 35.2 ms | 44.6 ms |
| **+ display** (draw tiles and panel, encode, publish) | +3.5 ms | +5.7 ms |
| **= on screen** | | |
| CAM 01-04 (1st) | ~14.7 ms | ~50.3 ms |
| CAM 17-20 (5th) | ~38.7 ms | ~50.3 ms |
Method:
- capture → results: MEASURED per set (bs=4: run log `scratchpad/sync_b4_v3.log`, not recorded; bs=20: the 60 s
  recording `yolo26_demo_outputs/multistream/yolo26s_20streams_1chip_batch20_v2.mp4`).
- display: INFERRED as measured on-screen median minus results median (bs=4: 26.0 − 22.5 = 3.5 ms over all
  cameras; bs=20: 50.3 − 44.6 = 5.7 ms). bs=20's step is longer because each update redraws all 20 tiles and panel
  rows; a bs=4 update touches 4.
- on screen per set: INFERRED as results + display; the demo measures only the on-screen median over all cameras.
- For reference, staggered bs=4 (default, cameras on their own phases; recording `..._batch4_v3.mp4`): 9.4-10.0 ms
  per set to results, 12.8 ms on screen (median). A full round of all 20 staggered cameras completes ~36 ms after
  its first capture (26.7 ms capture spread + ~9.5 ms; INFERRED), about the same as synchronized (35.2 ms).

### YOLO26-0039 — dp=4 batch 4: where the last ~9 % goes (appended 2026-10-29)

Question: one chip runs yolo26s batch 4 at ~660 FPS, so four should give ~2,640; dp=4 gave 2,347-2,404 (YOLO26-0025).
Is something slow, and does the demo's > 2,400 images/s mean the test is understating?

Code: branch `sdawle/tvardhineni/yolov26_bringup` at `fb1856e06ca` (the batch-4 L1 path is the same on this branch).
yolo26s, 640×640, batch 4 per chip, trace + 2CQ; chips reset (`tt-smi -r`) before every process.

1. Re-measure (MEASURED, `test_yolo26_trace_2cq.py`, `-k yolo26s-640-4`):
   - each chip alone (`test_yolo26_trace_2cq_perf`, `TT_VISIBLE_DEVICES=0..3`): 659.6 / 653.8 / 659.1 / 665.4 FPS
     (6.06 / 6.12 / 6.07 / 6.01 ms per iteration), sum 2,638;
   - dp=4 (`test_yolo26_trace_2cq_perf_dp`, 50 iterations): 2,363.1 / 2,402.6 / 2,404.0 / 2,420.6 / 2,438.0 FPS,
     mean 2,406 (6.56-6.77 ms per iteration) = 91 % of the sum. One further attempt hung at trace capture
     (YOLO26-0021, still OPEN) and was killed after 15 min; not counted.
   - The demo's 2,400 images/s (YOLO26-0028) is the offered load (80 × 30 FPS), not a ceiling; its measured
     ceiling 2,598.5 is explained in step 3.

2. Loop breakdown (MEASURED, scratch script `scratchpad/dpcheck/breakdown.py`: the perf-test loop with parts
   removed, 200 iterations after 20 warm-up, ms per iteration after `synchronize_device`; 2 processes each, values
   within 0.01 ms unless shown; one 4-chip process hung at capture):

   | step | 1 chip | 4 chips | 4 chips, output copies captured in the trace |
   |---|---|---|---|
   | trace only | 5.82 | 5.84 | 5.87 |
   | + upload (CQ1) + input copy | +0.05 | +0.05 | +0.05 |
   | + 6 outputs → DRAM (`to_memory_config`, outside the trace) | +0.05 | +0.05 | +0.00 (in trace) |
   | + `from_device` of the 6 outputs (CQ1) | +0.14 | +0.65 | +0.57 / +0.64 |
   | = perf-test loop | 6.05 (660 FPS) | 6.58 (2,431 FPS) | 6.49 / 6.56 (2,466 / 2,439 FPS) |

   - The device scales perfectly: the trace takes 5.84 ms on the 1×4 mesh vs 5.82 ms on one chip (2,740 FPS of
     compute).
   - Host time spent issuing commands (perf_counter before synchronize): 5.4-5.6 ms per iteration on 1 chip, 6.55 of
     6.58 ms on 4 chips, i.e. the 4-chip loop is host bound.
   - Hypothesis "the 6 output copies issued from the host, outside the trace, cost the 4-chip loop": REFUTED. They
     cost 0.05 ms on 1 and 4 chips alike; capturing them in the trace saves that but the trace grows 0.03-0.04 ms.
   - Hypothesis "reading the full outputs back is the cost": CONFIRMED, see step 3.

3. Read-back variants (MEASURED, scratch script `scratchpad/dpcheck/readback.py`; output copies in the trace for all
   variants; "reads alone" = 20 reads of the outputs with the device idle, ms each):

   | read-back | data per chip | reads alone | perf-test loop | FPS |
   |---|---|---|---|---|
   | 1 chip, 6 tile outputs (`from_device`) | 8.60 MB | 0.64 ms | 6.04 ms | 662.0 |
   | 4 chips, 6 tile outputs (`from_device` on the mesh tensors, as the test) | 8.60 MB | 3.89 ms | 6.53 ms | 2,449.9 |
   | 4 chips, per-chip shards (`get_device_tensors`) read from 4 host threads | 8.60 MB | 5.26 ms | 13.81 ms | 1,158.6 |
   | 4 chips, 6 outputs untilized in the trace (`to_layout` ROW_MAJOR, DRAM) | 5.64 MB | 3.36 / 3.17 ms | 6.32 / 6.27 ms | 2,533.3 / 2,553.9 |
   | 1 chip, same untilized outputs | 5.64 MB | 1.50 ms | 6.32 ms | 633.0 |
   | 4 chips, demo packing (`pack_outputs`: boxes + person logit only, row-major) | 0.34 MB | 1.03 ms | 6.14 ms | 2,606.1 |

   - Output sizes (READ from the tensors): boxes [1,1,N,4] are tile-padded to 32 columns (8× the data), scores
     [1,1,N,80] to 96; N = 25,600 / 6,400 / 1,600 for batch 4. Of the 8.60 MB per chip, 2.96 MB is padding.
   - 4 chips take ~6× as long as 1 chip to read the same per-chip data with the device idle (3.89 vs 0.64 ms), so the
     mesh reads run one chip after another (INFERRED from the timing; ~0.97 ms per chip). In the loop ~0.5 ms of it is
     not hidden behind compute: that is the 6.06 → 6.53-6.58 ms gap.
   - A host thread per chip: REFUTED, the iteration more than doubles (not investigated further; INFERRED to be
     contention on the shared CQ1 / blocking reads).
   - Untilizing in the trace removes the padding (5.64 MB per chip; lossless layout change, PCC not re-run): +3.4 to
     +4.2 % on 4 chips, but −4.4 % on one chip, where the device is the bottleneck and the untilize adds ~0.28 ms of
     device time (INFERRED: 6.32 − 6.04 ms). Row-major reads are also slower per byte (1.50 ms for 5.64 MB vs 0.64 ms
     for 8.60 MB on one chip).
   - Demo packing reaches 2,606 FPS on 4 chips, matching the demo's measured ceiling (2,598.5, YOLO26-0028), but it
     keeps 1 of 80 classes, so it is not the same test.

Status: ROOT-CAUSED. The chips scale; the shortfall is host read-back of tile-padded outputs from 4 chips in turn.
Not applied: the perf test still reads the 6 tile outputs, so its dp=4 batch-4 number stays 2,406 FPS (mean). Options
left: untilize only when `num_devices > 1` (+~4 %), or read a packed full-output tensor (boxes + 80 scores, row-major,
~5.6 MB per chip; the packed single-output attempt in YOLO26-0023 was slower on one chip, not tried on four).

### YOLO26-0039 addendum (2026-10-29) — why the demo is faster than the perf test on 4 chips

Question: is the demo faster because it uses 1 of the 80 classes while the perf test reads all 80?

Answer: mostly yes; precisely, the demo reads back far less data. The chips do the same work in both: the full model
runs and all 80 class scores are computed on the chip. Only what is copied back to the host differs:

| | perf test | demo (`pack_outputs`) |
|---|---|---|
| read per anchor | 4 box values + 80 class scores | 4 box values + 1 person score |
| layout | tile, padded (boxes 4 → 32 columns, scores 80 → 96) | row-major, no padding |
| data per chip per iteration | 8.60 MB | 0.34 MB |

- Of the 8.60 MB, the scores are 6.45 MB (READ, tensor sizes in step 3), so dropping 79 classes is the bulk of the
  saving; the rest is tile padding (boxes are 2.15 MB padded vs 0.27 MB of data).
- MEASURED (step 3, same script, only the read-back changed, 4 chips): all outputs 2,449.9 FPS, demo packing
  2,606.1 FPS. The demo's own measured ceiling, 2,598.5 images/s (YOLO26-0028), matches the packed figure, so the
  read-back difference accounts for the demo's higher number (INFERRED from the match; the demo was not re-run).
- It matters only on 4 chips. On 1 chip the read-back is hidden behind compute and the chip is the limit: demo ceiling
  661 images/s (YOLO26-0026) vs perf test 660 FPS. On 4 chips the reads from each chip run in turn (step 3), so the
  extra data costs throughput.
