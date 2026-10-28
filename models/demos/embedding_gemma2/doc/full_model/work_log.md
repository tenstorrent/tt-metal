# E4 work log (full multimodal model)

Host sjc-snva-tp100, one Blackhole P150a (device 0), branch `gtobarTT/embedding_gemma2_bringup`,
on top of E3 (245fd0da17). Date 2026-10-27. Environment for every device command:
`source tt-ops/runs/embeddinggemma-2/e1/env.sh` (python_env + transformers 5.20.2 overlay).
`$E4` = `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e4`. Logs: `$E4/logs/`.

## 0. E3 review doc fixes (commit 24e5485886, docs only)

- `doc/functional_vision/README.md`: landscape worst-token gap 0.0146 -> 0.0143
  (0.94130939 - 0.92697722 from `e2/final_results.jsonl`).
- `doc/functional_audio/README.md`: "ill-conditioned in the model itself" labelled [guess];
  sensitivity check scheduled for the datatype-sweep stage.
- `README.md`: worst audio soft-token cosine (0.764 / 0.773 vs HF-bf16 0.851 / 0.820) next to the
  soft PCC range.

## 1. HF semantics and ST formatting

Read `modeling_embedding_gemma2.py` (5.20.2) `:510-562`, `:694-880`, `masking_utils.py:79-170`,
the processor, `chat_template.jinja`, and ST 6.2.0 `base/modality.py`, `base/modules/transformer.py`.
Findings with file:line: `README.md` section 1-2.

```bash
# ST 6.2.0 (hf_ref venv, py3.11) on all inputs, fp32 and bf16
$E4/st/st_dump.py <tt-metal> $E4/st/st_features_fp32.pt fp32     # and bf16
# wrapper formatting + HF overlay reference vs ST (python_env)
python_env/bin/python $E4/st/check_formatting.py                 # -> $E4/st/check_formatting.json
```
28/28 inputs: identical input ids; HF overlay fp32 vs ST fp32 cosine 1.0000 (+-2e-7); ST bf16 vs
fp32 0.99962-0.99998. Summary committed as `st_reference.json`. The first ST run loaded bf16 by
default ("This model was loaded in torch.bfloat16 on CPU"), so fp32 was forced with
`model_kwargs={"dtype": torch.float32}`. The first reference load failed with
`attn_implementation="sdpa"` (Gemma4AudioModel has eager only); the reference now uses the
per-sub-model default, as ST does.

## 2. Smoke (`$E4/smoke.py`, `logs/smoke1.log`)

First device run of the merge, eager: text_query 0.99997, image_dog_caption 0.99993,
audio_speech2s_text 0.99984, mixed_image_audio 0.99975 cosine vs HF fp32.

## 3. PCC / retrieval / batching / audit (`logs/pcc_run1.log`, `pcc_run1.jsonl`)

```bash
EMBEDDING_GEMMA2_RESULTS=$E4/pcc_run1.jsonl EMBEDDING_GEMMA2_ARTIFACTS=$E4/artifacts \
python_env/bin/python -m pytest models/demos/embedding_gemma2/tests/pcc/test_full_model.py -q -p no:cacheprovider
```
27 passed in 62 s. 21 inputs: cosine min 0.99957 (video_text), mean 0.99990; MRL 256 min 0.99963.
Retrieval argmax TT == HF on 4/4 image rows and 3/3 audio rows. `retrieval_*.json` copied from
`$E4/artifacts/`.

## 4. Trace (`logs/trace_run1.log`, `trace_run1.jsonl`)

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 EMBEDDING_GEMMA2_RESULTS=$E4/trace_run1.jsonl \
python_env/bin/python -m pytest models/demos/embedding_gemma2/tests/pcc/test_full_model_trace.py -q -p no:cacheprovider
```
11 passed in 31 s: 8 traces, replay == eager bit-exact, determinism 5 replays + 2 eager.

## 5. Latency (`logs/perf_run1.log`, `perf_run1.jsonl`)

```bash
EMBEDDING_GEMMA2_RESULTS=$E4/perf_run1.jsonl \
python_env/bin/python -m pytest models/demos/embedding_gemma2/tests/pcc/test_full_model_perf.py -k traced -q -s -p no:cacheprovider
```
6 passed. Device ms: text 1x128 8.73, 1x512 18.30, 8x512 74.48, image + caption 73.90,
10 s audio + text 41.99, 8-frame video 230.11. One "Allocating device buffers is potentially
unsafe due to the existence of an active trace" warning appears after capture (also in the E2/E3
final runs); the trace module runs the same readback path under `TT_METAL_TRACE_ALLOC_TRACKING=1`
without a failure.

## 6. Tracy + tt-perf-report (`logs/tracy_run1.log`, `logs/tracy_run2.log`)

Run 1 failed in post-processing (`Device data missing: Op 1213440`, 250 "Profiler DRAM buffers
were full" warnings): same cause as E3 (1000-program flush default). Run 2 with
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000`: 1194 device ops, 0 host ops, 73,590 us + 949 us
gaps. Commands in `README.md` section 8. Stacked PNG deleted (not committed).

## 7. Watcher (`logs/watcher_run1.log`)

```bash
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$E4/watcher EMBEDDING_GEMMA2_RESULTS=$E4/watcher_results.jsonl \
python_env/bin/python -m pytest models/demos/embedding_gemma2/tests/pcc/test_full_model.py \
  models/demos/embedding_gemma2/tests/pcc/test_full_model_trace.py -q -p no:cacheprovider
```
38 passed in 140 s. `$E4/watcher/generated/watcher/watcher.log`: 1293 lines, 0 lines matching
`error|assert|exception|overflow|sanitiz|fault` (`grep -ciE`).

## 8. Merge placement test + demo (`logs/merge_demo_run1.log`, `logs/demo_run2.log`, `logs/demo_script_run1.log`)

`test_merge_matches_hf` was added after the watcher run: 3 passed (`$E4/merge_run1.jsonl`). The
first demo run failed: `KeyError: no trace for b3_s32`. The demo captured `shape_keys` of all
inputs at once, which groups the 3 queries with the 4 documents (`b8_s32`), while `encode` runs
the queries and documents in separate calls. Fix: capture the keys of each `encode` call. Then
`pytest demo/demo.py -s`: 2 passed (TT argmax == HF argmax in all 4 tasks, every TT vs HF
embedding cosine >= 0.99961); `python demo/demo.py --no-hf`: exit 0.

E1/E2/E3 modules were not modified (`git diff --stat` shows only README.md, demo/demo.py,
doc/context_contract.json plus new files), so the E1-E3 PCC suite was not rerun.

## 9. Final run on the committed code (f07e1f1d82, after pre-commit black / isort)

```bash
export EMBEDDING_GEMMA2_RESULTS=$E4/final_results.jsonl
pytest $P/tests/pcc/test_full_model.py -q                                       # 30 passed in 51 s
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest $P/tests/pcc/test_full_model_trace.py -q # 11 passed in 29 s
pytest $P/tests/pcc/test_full_model_perf.py -k traced -q -s                     # 6 passed in 29 s
pytest $P/demo/demo.py -q -s                                                    # 2 passed in 26 s
```
(`logs/final_run.log`). Same numbers as above: 21 inputs cosine min 0.99957 / mean 0.99990,
MRL 256 min 0.99963, batched min 0.99975, merge row cosine min 0.99868, retrieval agree 2/2,
trace vs eager max abs diff 0.0, device ms 8.73 / 18.30 / 74.49 / 73.89 / 41.99 / 230.16.
