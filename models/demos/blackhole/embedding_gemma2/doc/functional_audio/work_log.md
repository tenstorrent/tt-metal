# E3 functional audio tower: work log

Host sjc-snva-tp100, one Blackhole P150a (device 0), branch `gtobarTT/embedding_gemma2_bringup`
on top of E2 (037da96434) and the E2 doc-fix commit (a65a03e002). Date 2026-10-27. Environment for
every command:

```bash
cd /local/ttuser/gtobar/tt-metal
source /local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e1/env.sh   # TT_METAL_HOME, overlay PYTHONPATH, TT_VISIBLE_DEVICES=0
E3=/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e3
```

## 0. E2 doc fixes (commit a65a03e002, docs only)

- `functional_vision/README.md` and `work_log.md`: the worst per-token cosine comparison is now
  restated from the committed `e2/final_results.jsonl`. TT is below HF-bf16 on landscape
  (0.92698 vs 0.94131), square, small, tiny and 5 of 8 video frames, and above it on portrait
  and 3 of 8 frames. The "patch 1243, cosine 0.290 vs 0.294" figures came from uncommitted
  scratch scripts and were **removed**, not reproduced.
- `functional_vision/README.md` now states that the b1_s2560 tt-perf-report is the
  representative report and that batch 4 has a traced device time only (294.41 ms).

## 1. Reading

- HF source: `tf_overlay/transformers/models/gemma4/modeling_gemma4.py` (ClippableLinear 176-202,
  RMSNorm 205-223, rel-pos 226-254, attention 257-360, subsampling 363-418, FFN 421-453, causal
  conv 457-487, LightConv1d 490-528, layer 531-579, mask function 1874-1885, audio model
  1889-1978), `feature_extraction_gemma4.py` (semicausal pad, frame mask 169-224) and
  `embedding_gemma2/modeling_embedding_gemma2.py` (embedder 565-591, merge 818-837,
  `get_audio_features` 856-878).
- Recon "Audio arch" checked line by line. All of it held. Details the recon did not state:
  - The HF mask path is a float additive mask (`masked_fill(mask != 0, -1e9)` after the
    softcap). Under that mask the 12 x 24 blocking is exactly the band `q-11 .. q`.
  - The feature-extractor mask can mark trailing frames invalid even for one clip
    (`pad_to_multiple_of=128` samples), so the conv masks matter at batch 1 too.
  - LayerNorm eps in the subsampling stack is `rms_norm_eps`.
  - The clip bounds are 4 bf16 scalars per linear, 84 linears.
- Read `models/experimental/parakeet/tt/ttnn_parakeet.py` (rel_shift 536-545, shift-matrix depthwise
  conv 417-431, conv2d 477-501): the rel-shift row-major view trick and the 0/1 shift-matrix idea
  were adapted (re-derived for a 32 x 96 block layout); nothing is imported.

## 2. Op smoke (scratch, not committed)

One device script checked the ops the design needs. Against torch:

| op | result |
|---|---|
| `ttnn.conv2d` with 1 input channel on fp32 RM `[2,3072,128,1]` | PCC 0.999995 |
| same, with weights prepared by a first call (`return_weights_and_bias=True`) | identical |
| `layer_norm` over 128 channels | 0.999991 |
| conv 128->32 | 0.999995 |
| NHWC -> `[B,1,t,1024]` reshape | matches HF flatten order |
| fp32 `tanh(x/50)*50` | max abs err 7.6e-6 |
| parakeet-style rel shift | exact |
| in0-broadcast matmul `[1,1,M,K] @ [B,1,K,N]` | exact |
| tile-aligned block reshape | exact |
| fp32 `bmm` with `transpose_b` | rel err 1.2e-3 |

Two findings changed the design:
- `ttnn.pad` cannot front-pad on device in TILE layout (`pad.cpp:553`). The previous-block
  context is therefore built by slice + concat with a device-made zero block.
- The fused `ttnn.softmax` on fp32 logits is off by up to 7.9e-3 in probability; a manual
  max / exp / sum / reciprocal softmax is within 3.6e-7. The attention uses the manual form.

## 3. First full-path smoke (scratch)

s2, b2 (s10 + s2), s30 and short. Teacher-forced PCC was at least 0.99992 for every module, and
the host band mask equals the HF 5D mask on valid queries. Two bugs:
- `block_context` deallocated `slice(x)`, which is x itself when S = 32 (short clip): "Input
  Tensor is not allocated". Fixed: the slice is not deallocated.
- The first `test_reference_matches_full_model` failed with max abs 2.7e-3. `AutoModel.from_config`
  builds in the config dtype (bf16), so the reference's non-persistent
  `rel_pos_enc.inv_timescales` was bf16-rounded (0.9805 vs 0.9821). The full
  `from_pretrained(dtype=float32)` model and the TT table use exact fp32. Fixed in the
  reference loader (`from_config(acfg, dtype=torch.float32)`); after that, max abs diff 0.0.
  `pcc_run1.jsonl` predates this fix and is superseded by `pcc_run2.jsonl`.

## 4. PCC suite

```bash
tmux new-session -d -s tt-bringup_embeddinggemma2_e3_pcc2 ... \
  EMBEDDING_GEMMA2_RESULTS=$E3/pcc_run2.jsonl \
  python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/audio -q -p no:cacheprovider \
  -k 'not trace and not audit and not perf'
```
92 passed (`$E3/logs/pcc_run2.log`).

## 5. Trace, allocation tracking, audit, perf

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 EMBEDDING_GEMMA2_RESULTS=$E3/trace_run1.jsonl \
python_env/bin/python -m pytest $A/test_audio_trace.py $A/test_audio_fallback_audit.py $A/test_audio_perf.py -q -k 'not tracy'
```
14 passed (`$E3/logs/trace_run1.log`):
- replay == eager bit-exact on 8 cases / 9 clips
- no live-buffer report from the tracker
- determinism: 5 replays + 2 eager runs bit-identical on s10 and b2
- audit: 0 torch calls
- device latency: 23.67 ms (1 x 10 s), 47.80 ms (1 x 30 s), 80.48 ms (4 x 10 s)

## 6. Tracy + tt-perf-report

```bash
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 PATH=$PWD/python_env/bin:$PATH python_env/bin/python -m tracy -r -p -v -m pytest \
  models/demos/blackhole/embedding_gemma2/tests/pcc/audio/test_audio_perf.py -k tracy -q
CSV=generated/profiler/reports/2026_10_27_13_48_12/ops_perf_results_2026_10_27_13_48_12.csv
D=models/demos/blackhole/embedding_gemma2/doc/functional_audio/tracy/b1_s768
tt-perf-report $CSV --start-signpost PERF_AUDIO --end-signpost PERF_AUDIO_END --csv $D/forward_perf_report.csv --no-advice \
  > $D/forward_perf_report.console.log
tt-perf-report $CSV --start-signpost PERF_AUDIO --end-signpost PERF_AUDIO_END --no-advice --no-color > $D/forward_perf_report.txt
```
The first attempt (`tracy_run1.log`) failed in post-processing: `Device data missing: Op 1114112
not present in cpp_device_perf_report.csv`. 73 of 2734 ops had no device record, spread over the
run including the measured window. The cause is the profiler's default of 1000 programs between
`ReadDeviceProfiler` flushes (`tt_metal/impl/profiler/profiler_state_manager.cpp:25`), while one
audio forward is about 1350 programs. With `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000` the run
is complete: 1357 device ops, 0 host ops, 47,482 us device + 746 us gaps (`tracy_run2.log`). The
stacked PNG was deleted (not committed, as in E1/E2).

## 7. Watcher (includes the trace tests)

```bash
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$E3/watcher EMBEDDING_GEMMA2_RESULTS=$E3/watcher_results.jsonl \
python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/audio -q -p no:cacheprovider \
  -k 'not perf and not reference_matches'
```
101 passed in 188.81 s. `$E3/watcher/generated/watcher/watcher.log` has 1003 lines and 0 lines
matching `error|assert|exception|overflow|sanitiz|fault` (`grep -ciE`). Deselected: the perf
tests (timing under watcher is meaningless) and the CPU-only reference-vs-full-model checks.

## 8. Precision probe -> committed test

A scratch probe (`$E3/logs/exp_fp32_act.log`) switched every activation to fp32 and showed higher
soft PCC on all cases. Its figures are not quoted from the scratch run. The option was made a real
constructor argument (`AudioModel.from_state_dict(..., act_dtype=ttnn.float32)`, default bf16), and
the evidence comes from committed tests:
- `test_audio_model.py::test_soft_tokens_fp32_activations`
- `test_audio_perf_fp32act.py`

## 9. Full suite before commit

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 EMBEDDING_GEMMA2_RESULTS=$E3/full_run1.jsonl \
python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/audio -q -p no:cacheprovider -k 'not tracy'
```
114 passed in 63.63 s, no tracker report (`$E3/logs/full_run1.log`). The tables in `README.md`
come from this run.

## 10. Shared files

No E1 / E2 code file was modified. `git status` before the audio commit listed only new paths
(`tt/audio/`, `tests/pcc/audio/`, `reference/hf_audio_reference.py`, `doc/functional_audio/`) and
the additive audio sections in `README.md` and `doc/context_contract.json`. The audio modules
import the E1 `RMSNorm` and weight helpers unchanged. Those two shared files were touched, so the
E1 / E2 suite was rerun after the commit (section 11).
