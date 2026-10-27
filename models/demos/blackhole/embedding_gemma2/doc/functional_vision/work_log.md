# E2 functional vision tower: work log

Host sjc-snva-tp100, one Blackhole P150a (device 0), branch `gtobarTT/embedding_gemma2_bringup`
on top of E1 (634d9b6f11). Date 2026-10-27. Environment for every command:

```bash
cd /local/ttuser/gtobar/tt-metal
source /local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e1/env.sh   # TT_METAL_HOME, overlay PYTHONPATH, TT_VISIBLE_DEVICES=0
E2=/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e2
```

## 1. Reading

- HF source: `tf_overlay/transformers/models/gemma4/modeling_gemma4.py` (patch embedder 585-627,
  pooler 630-694, MLP 697-710, axial rope 713-769 + 847-900, attention 903-969, layer 972-1013,
  encoder 1016-1057, vision model 1981-2046) and `embedding_gemma2/modeling_embedding_gemma2.py`
  (embedder 565-591, `get_image_features` 663-692, `get_video_features` 882-927).
- Recon "Vision arch" checked line by line. One detail the recon did not state: HF calls the
  first rope half `freq_h` but feeds it `position_ids[..., 0]`, which is the x (column) coordinate
  (`recomposition_frequencies`, :766-769). The implementation follows the code: channels 0-31 by x.
- The checkpoint stores `Gemma4ClippableLinear` weights as `<name>.linear.weight`; with
  `use_clipped_linears: false` there are no clip buffers (listed with `safe_open`).
- The Gemma4 PR branch (`origin/arg/gemma4_multimodal` @ 71162a738c) was read for orientation:
  its 2D RoPE splits head_dim and runs `rotary_embedding_llama` per half. Nothing was imported or
  copied. The 64-channel head made one signed-permutation matmul simpler here.

## 2. Op smoke (scratch, not committed)

One device script checked the ops the design needs at S = 2560, B = 2 (PCC vs torch): fp32-input x
bf16-weight `ttnn.linear` 0.999999; head-broadcast multiply `[B,12,S,64] * [B,1,S,64]` 0.999996;
batch-broadcast matmul `[B,12,S,64] @ [1,1,64,64]` 0.999996; `ttnn.embedding` on a 10241-row
table 0.999999; batched matmul `[B,1,288,S] @ [B,1,S,768]` 0.999997; SDPA with a `[B,1,S,S]` mask
0.999986. A `[B,1,1,S]` mask is rejected (`sdpa_device_operation.cpp:178`, mask rows must equal
Q rows), so the mask is repeated to `[B,1,S,S]` on device as in E1.

## 3. First full-path smoke (scratch)

Landscape + small image, batch 2: teacher-forced PCC >= 0.99997 for every module and all 16
layers; free-running soft-token PCC 0.99878 / 0.99972. No numerical fix was needed.

## 4. Free-running drift analysis (scratch scripts)

- Per-layer free-running probe on landscape: PCC >= 0.99999 through layer 10, 0.99942 at 14,
  0.99736 at 15. (The per-patch figures from this scratch probe were removed during E2 review:
  no committed log reproduces them.)
- HF itself in bf16 vs HF fp32, from the committed `$E2/final_results.jsonl`: soft PCC 0.99882
  (landscape) / 0.99843 (portrait) / 0.99922 (square) / 0.99984 (small). Worst per-token cosine:
  TT is below HF-bf16 on landscape (0.92698 vs 0.94131), square, small, tiny and 5 of 8 video
  frames, and above it on portrait (0.94289 vs 0.86620) and 3 of 8 frames. So the worst-token
  error of the port is not "no worse than HF bf16"; it is worse on most cases.
- fp32 residual stream variant (all residual adds and norm inputs in fp32): soft PCC landscape
  0.99963, portrait 0.99910, video frames 0.99833-0.99923. Not a consistent win, so not kept.

## 5. PCC suite

```bash
tmux new-session -d -s tt-bringup_embeddinggemma2_e2_pcc1 ... \
  EMBEDDING_GEMMA2_RESULTS=$E2/pcc_run1.jsonl \
  python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision -q -p no:cacheprovider \
  -k 'not trace and not audit and not perf'
```
66 passed in 113.97 s (`$E2/logs/pcc_run1.log`). Tables in `README.md` of this directory.

## 6. Trace, allocation tracking, audit, perf

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 EMBEDDING_GEMMA2_RESULTS=$E2/trace_run1.jsonl \
python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_trace.py \
  models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_fallback_audit.py \
  models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_perf.py -q -p no:cacheprovider
```
14 passed in 54.21 s (`$E2/logs/trace_run1.log`): replay == eager bit-exact on 7 cases (15
images / frames), no live-buffer report from the allocation tracker, determinism 5 replays + 2
eager bit-identical (b2, video8), audit 0 torch calls, latency b1 55.60 ms, b4 294.47 ms, b8 x 1280
199.61 ms device per forward.

## 7. Tracy + tt-perf-report

```bash
PATH=$PWD/python_env/bin:$PATH python_env/bin/python -m tracy -r -p -v -m pytest \
  models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_perf.py -k tracy -q
CSV=generated/profiler/reports/2026_10_27_13_05_21/ops_perf_results_2026_10_27_13_05_21.csv
D=models/demos/blackhole/embedding_gemma2/doc/functional_vision/tracy/b1_s2560
tt-perf-report $CSV --start-signpost PERF_VISION --end-signpost PERF_VISION_END --csv $D/forward_perf_report.csv --no-advice \
  > $D/forward_perf_report.console.log
tt-perf-report $CSV --start-signpost PERF_VISION --end-signpost PERF_VISION_END --no-advice --no-color > $D/forward_perf_report.txt
```
448 device ops, 0 host ops, 55,493 us device time + 361 us gaps (`$E2/logs/tracy_run1.log`).
The stacked PNG that tt-perf-report also writes was deleted (not committed, as in E1).

## 8. Watcher (includes the trace tests, unlike E1)

```bash
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$E2/watcher EMBEDDING_GEMMA2_RESULTS=$E2/watcher_results.jsonl \
python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision -q -p no:cacheprovider \
  -k 'not perf and not reference_matches'
```
73 passed in 146.18 s. `$E2/watcher/generated/watcher/watcher.log`: 986 lines, 0 lines matching
`error|assert|exception|overflow|sanitiz|fault` (`grep -ciE`). Deselected: the perf tests
(timing under watcher is meaningless) and the CPU-only reference-vs-full-model checks.

## 9. Final run on the committed code (18cd32291c, after pre-commit black)

```bash
EMBEDDING_GEMMA2_RESULTS=$E2/final_results.jsonl \
python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision -q -p no:cacheprovider
```
80 passed in 150.76 s (`$E2/logs/final_run.log`, tmux `tt-bringup_embeddinggemma2_e2_final`).
All 199 PCC metric values shared with run 1 are bit-identical. Latency b1 55.597 ms, b4 294.414 ms,
b8 x 1280 199.614 ms (run 1: 55.604 / 294.470 / 199.609).

No E1 file was modified (`git diff 634d9b6f11 --stat` lists only new vision files, README.md and
context_contract.json), so the E1 suite was not rerun. The vision code imports the E1 `MLP`,
`RMSNorm` and weight helpers unchanged.
