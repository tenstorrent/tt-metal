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
