# E1 functional text encoder: work log

Host sjc-snva-tp100, one Blackhole P150a (device 0), branch `gtobarTT/embedding_gemma2_bringup`
(base main be66b32587, Release build with ENABLE_TRACY=ON). Date 2026-10-27.

Environment for every device command:

```bash
cd /local/ttuser/gtobar/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=/local/ttuser/gtobar/venvs/tf_overlay:$PWD TT_VISIBLE_DEVICES=0 \
       HF_MODEL=google/embeddinggemma-2 OMP_NUM_THREADS=16
```

## 1. Reference overlay

python_env has no pip module; uv installs into a target dir without touching python_env:

```bash
uv pip install --python python_env/bin/python --no-deps --target /local/ttuser/gtobar/venvs/tf_overlay transformers==5.20.2
# transformers 5.20.2 needs huggingface_hub>=1.31 and tokenizers>=0.23.1 (python_env: 1.16.1 / 0.22.2)
uv pip install --python python_env/bin/python --no-deps --target /local/ttuser/gtobar/venvs/tf_overlay \
    "huggingface_hub>=1.31,<2" tokenizers==0.23.3      # -> huggingface_hub 1.33.0
```

`huggingface_hub==2.3.0` (the hf_ref version) was tried first and fails on py3.10 python_env
(`ModuleNotFoundError: httpx2`); 1.33.0 works.

Checks (measured):
- `EmbeddingGemma2Model` (full model, fp32, default attention) on a 2-prompt padded batch:
  overlay (py3.10, transformers 5.20.2, torch 2.11) vs hf_ref (py3.11, transformers 5.21.0,
  torch 2.14): max abs diff of `last_hidden_state` **0.0**; `modeling_embedding_gemma2.py` and the
  bidirectional sliding mask code are byte-identical between the two versions (`diff`, no output).
- `reference/hf_reference.py` (`EmbeddingGemma2TextModel` from the `language_model.*` tensors,
  sdpa) vs the full model on the same batch: max abs diff over valid tokens **0.0**.

Goldens are not stored; the tests run the HF fp32 reference live.

## 2. First device smoke (scratch script, not committed)

Teacher-forced per-layer PCC 0.99998-0.99999 for all 24 layers at L=20, then at L=(1025,600,7)
and L=8192; end-to-end last-hidden PCC 0.9983 (L=20) / 0.9995 / 0.9994, embedding cosine >= 0.9993.
The first implementation passed without numerical fixes.

Long-context probe, L=32768 (bucket 32768): all layers >= 0.99980 teacher-forced, last-hidden PCC
0.99882, cosine 0.9999. `/usr/bin/time -v`: wall 2:33, max RSS 23,982,524 kB (HF fp32 reference).

## 3. PCC suite

```bash
EMBEDDING_GEMMA2_RESULTS=.../e1/results.jsonl \
pytest models/demos/embedding_gemma2/tests/pcc -q -p no:cacheprovider
```
Result: **99 passed in 271.72 s** (tmux `tt-bringup_embeddinggemma2_e1_pcc`, log
`tt-ops/runs/embeddinggemma-2/e1/logs/pcc_full.log`, metrics `e1/pcc_results.jsonl`).
Table in `README.md` of this directory.

## 4. Fallback audit + demo

```bash
pytest models/demos/embedding_gemma2/tests/pcc/test_fallback_audit.py \
       models/demos/embedding_gemma2/demo/demo.py -q -s
```
2 passed. Audit: `torch calls inside forward: 0 []`; no ttnn host-transfer entry point called;
`throw_exception_on_fallback=True` did not raise. Demo ranks the right document first for all
three queries (scores 0.865 / 0.752 / 0.818).

## 5. Watcher

```bash
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<e1>/watcher \
pytest models/demos/embedding_gemma2/tests/pcc/test_text_model.py -k "test_text_model and not s32768" -q
```
15 passed in 122.24 s (3 teacher-forced runs + 12 free-running cases). `watcher.log` (3859
lines, 12 dumps) has no line matching `error|assert|exception|overflow|sanitiz|fault`.
The 32768 case was deselected for watcher runtime only; it passed in the non-watcher suite.

## 6. Perf (traced)

```bash
pytest models/demos/embedding_gemma2/tests/perf/perf.py -k traced -q -s
```
b1x128 8.751 ms, b1x512 18.310 ms, b1x2048 58.982 ms, b8x512 74.481 ms device per forward
(20 replays after 3 warmups). A first run before the capture_all change gave 8.729 / 18.315 /
58.974 / 74.534 ms.

## 7. Tracy + tt-perf-report

```bash
python -m tracy -r -p -v -m pytest models/demos/embedding_gemma2/tests/perf/perf.py -k tracy -q -s
tt-perf-report <ops_perf_results.csv> --start-signpost PERF_TEXT --end-signpost PERF_TEXT_END \
    --csv doc/functional_text/tracy/b1_s512/forward_perf_report.csv --no-advice
tt-perf-report <ops_perf_results.csv> --start-signpost PERF_TEXT --end-signpost PERF_TEXT_END \
    --no-advice --no-color > doc/functional_text/tracy/b1_s512/forward_perf_report.txt
```
743 device ops, 0 host ops, 18,134 us device time, 855 us op-to-op gaps.
Raw CSV: `generated/profiler/reports/2026_10_27_12_36_08/ops_perf_results_2026_10_27_12_36_08.csv`.

## 8. Trace allocation tracking (finding and fix)

`TT_METAL_TRACE_ALLOC_TRACKING=1 pytest tests/pcc/test_trace.py` first failed:
`Found 63 device buffer(s) still alive before trace replay` in `test_determinism[b3_pad]`.
The PCC and bit-exact checks had passed, so the hazard was latent.

Root cause (scratch bisect script, measured): an eager forward of a shape not yet compiled leaves
program-cache buffers alive (masks only: +3 buffers, full forward: 61 live including the 2 inputs;
the count stays 61 over three repeats, so this is per-program caching, not a leak). The test ran
the b8 x 512 eager pass after the b3 x 2048 trace was captured, so those buffers could sit in the
b3 trace's intermediate region.

Fix: `TextEncoder.capture_all(shapes)` uploads every bucket constant, allocates every persistent
input and compiles every shape before the first capture, and acknowledges trace outputs as
corruptible (each replay rewrites them). Lazy capture in `replay` was removed; eager runs after
capture are allowed only for compiled shapes. Rerun with tracking: test_trace.py + demo, 7 passed.

## 9. Final suite

```bash
pytest models/demos/embedding_gemma2/tests/pcc models/demos/embedding_gemma2/demo/demo.py -q
```
101 passed in 249.22 s (`e1/logs/pcc_full.log`; first run in `pcc_full_run1.log`). All 230 logged
metric values equal those of the first run exactly.
