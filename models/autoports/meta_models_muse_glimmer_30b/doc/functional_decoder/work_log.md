# Functional decoder work log — `meta-models/Muse-Glimmer-30B`

Date: 2026-08-11. Host `tt-quietbox`, 4 x Blackhole visible, stage run on a
1x1 mesh. Repo: `/home/ttuser/dev/muse-glimmer/tt-metal`, branch
`agentic-research/hous/muse-glimmer-30b`. Python env
`/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv`.

## 1. Environment

`config.json` declares `model_type: muse_glimmer` and
`transformers_version: 5.15.0.dev0`.  The repo pins `transformers == 5.12.1`,
which has no `muse_glimmer` module at all — `AutoConfig.from_pretrained` fails
there, so no reference is possible.  Upgraded the (model-specific) venv:

```bash
pip download transformers==5.15.0        # into /tmp/tfw
/usr/bin/pip --python /home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv/bin/python \
    install --no-deps --upgrade /tmp/tfw/transformers-5.15.0-py3-none-any.whl
python -c "import transformers; print(transformers.__version__)"   # 5.15.0
python -m pip install tt-perf-report                                # 1.2.8
```

Device health / mesh smoke (`tt-smi` is not installed on this host; used the
TTNN open/close probe instead):

```bash
python -c "import ttnn; m=ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0); \
           print('MESH_SMOKE_OK', m.arch(), m.compute_with_storage_grid_size()); ttnn.close_mesh_device(m)"
# MESH_SMOKE_OK Arch.BLACKHOLE 11-10
```

No hardware faults, resets or hangs occurred during this stage.

## 2. Reading the HF model

Read `transformers/models/muse_glimmer/{configuration,modeling}_muse_glimmer.py`
line by line.  Findings that drove the implementation:

* Two layer kinds only: `sliding_attention` + RoPE(θ=500000) and
  `full_attention` + **NoPE** (`layer_rope_theta[i] == 0`, and
  `MuseGlimmerTextModel.forward` passes `position_embeddings=None` for those).
  Pattern `[s, s, s, f]` x13 over 52 layers.
* `MuseGlimmerTextCenteredRMSNorm` = `rms_norm(x) * (1 + w)` — four per layer,
  with **two different epsilons** (`rms_norm_eps=1e-5` on the pre-norms,
  `post_norm_eps=1e-8` on the post-norms).  Getting the post-norm eps wrong is
  a silent-ish accuracy bug, so it is explicit in the config dataclass.
* Attention has a **sigmoid output gate** (`self_attn.gate_proj`) applied to the
  concatenated heads *before* `o_proj`, driven by the same normed hidden states
  that feed Q/K/V.  This is why the layer does not reuse
  `models/common/modules/attention/attention_1d.py` (no gate hook there).
* `qk_norm` is a **scale-less** RMSNorm over `head_dim`, applied to Q and K but
  **not V**; Q is then multiplied by `qk_scale_factor = 3.87`.
* Sliding mask semantics: `transformers.masking_utils.sliding_window_overlay` is
  `kv_idx > q_idx - W` and-ed with causal — i.e. W tokens including self.
  Confirmed `ttnn`'s `sliding_window_size` matches at PCC 0.9999.
* `max_position_embeddings = 131072`.

Reference implementations read for patterns: `models/demos/gemma4/tt/attention/`
(sliding vs full layers, paged fill/update, chunked prefill, decode head split),
`models/autoports/.../functional_decoder.py` from earlier ports (module shape,
`from_state_dict` contract).

## 3. Implementation decisions

* Self-contained layer using direct TTNN ops rather than `Attention1D`: the
  output gate, scale-less per-head QK norm and per-layer NoPE toggle do not fit
  that module's contract, and the functional stage values a readable, explicit
  layer.
* `qk_scale_factor` folded into the SDPA `scale` (`3.87/sqrt(128)`), which is
  exact because RoPE is a rotation and Q only feeds `q @ k^T`.
* Fused `wqkv = [q | k | v]` -> `ttnn.experimental.nlp_create_qkv_heads(_decode)`.
* `1 + w` folded into the four norm weights at setup.
* KV cache owned by the layer, paged, full `max_seq_len` per user, BF16.
* Prefill chunked internally at 8192 tokens through the whole layer to bound
  DRAM (see README).
* Decode RoPE gathers per-user cos/sin rows on device with `ttnn.embedding` from
  a 2D `[max_seq_len, head_dim]` table, so a single captured trace replays
  across positions.

## 4. Bugs found and fixed

| # | symptom | root cause | fix |
| --- | --- | --- | --- |
| 1 | sliding prefill PCC 0.9766 at `seq_len=3000, chunk=1024` | the carried sliding tail was only the previous *chunk's* last rows, so a chunk shorter than the 2048 window handed over too little history | carry the tail from `[previous tail | this chunk]`, always the last `window` rows |
| 2 | `TypeError: incompatible function arguments` on `chunked_scaled_dot_product_attention(..., chunk_start_idx=...)` | `chunk_start_idx` is positional-only in the nanobind signature | pass positionally |
| 3 | same op rejects `scale=0.342062905…` while accepting `scale=1.0` | nanobind `nb::arg("scale").noconvert()` on `std::optional<float>` refuses any Python double that is not exactly a float32 | `_as_float32()` rounds the scale to the nearest float32 (relative error ~1e-8) |
| 4 | `RuntimeError: bad optional access` from `ttnn.slice` on the page table, only on the 3rd prefill chunk | a **full-range** `ttnn.slice` returns the input tensor itself; the full-attention path then `deallocate`d it, freeing the caller's page table | `_page_table_row()` returns `(tensor, owned)` and only the owned case is deallocated |
| 5 | sliding prefill PCC 0.9843 at `seq_len=2049` (single chunk) | **tt-metal bug**: sliding-window prefill SDPA is wrong when `q_chunk_size == 2 * k_chunk_size` at certain lengths past the window | always use `q_chunk_size == k_chunk_size`; standalone reproducer committed |
| 6 | `ttnn.pad: on device tile padding does not support front padding` | front-padding a tiled tensor is unsupported | build filler Q rows with `ttnn.zeros` + `ttnn.concat` |

Bug 5 reproducer (kept as stage evidence, runs standalone in ~1 min):

```bash
python models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/sdpa_sliding_window_chunk_repro.py
# S=2080  q256/k128=0.97796  q128/k64=0.97319  q128/k128=0.99987  q256/k256=0.99988
# S=2304  q256/k128=0.99983  q128/k64=0.99983  q128/k128=0.99982  q256/k256=0.99982
# S=4128  q256/k128=0.97495  q128/k64=0.96852  q128/k128=0.99980  q256/k256=0.99981
# S=8224  q256/k128=0.97613  q128/k64=0.97084  q128/k128=0.99985  q256/k256=0.99986
```

Bugs 1–6 were all localised by ordinary narrowing (isolate the op, compare to a
PyTorch reference, bisect chunk sizes), so `$autofix` was not needed.

## 4b. Stage-review round 1 findings and remediation

`$stage-review` (fresh xhigh subagent, read-only) returned **more-work-needed**
on the first pass.  All findings were fixed in this same stage; verbatim
findings and the remediation:

| finding | verdict | what was done |
| --- | --- | --- |
| **P1** `prefill_forward(start_pos > 0)` silently wrong on `sliding` layers: `sliding_tail` was reset to `None` before the chunk loop, so a continuation attended only the new chunk's K/V instead of the previous window. No test passed a non-zero `start_pos`. | real bug | `prefill_forward` now takes `sliding_kv_tail` / `return_sliding_kv_tail` (the gemma4 generator-level hand-off contract) and **raises** if a sliding continuation is attempted without the window. New `test_continuation_prefill_pcc` compares 4096+3000 two-call prefill against a single-shot HF prefill for both kinds (0.99849 / 0.99829) plus a decode at 7096; `test_continuation_prefill_requires_sliding_tail` covers the error path. |
| **P1** `_prefill_rope_tables` repeated the full-range `ttnn.slice` aliasing hazard and would free the layer's persistent cos/sin tables when `start_pos == 0 and length == max_seq_len`. | real latent bug | ownership check added, mirroring `_page_table_row`. Regression test `test_prefill_seq_len_equals_max_and_chunk` (`max_seq_len == chunk == seq_len == 4096`, two prefills back to back). |
| **P2** the 32-replay decode Tracy captures logged `Profiler DRAM buffers were full, markers were dropped!` (360x sliding, 5x full) and under-counted ops, making decode look ~13 % faster than reality. | real | re-profiled all four windows with `MG_PERF_DECODE_ITERS=8`; all four logs now have **0** drop warnings and every op-code count is an exact multiple of the replay count. Corrected numbers: sliding 3.165 ms/token (was 2.766), full 3.081 ms/token (was 2.974). README perf section rewritten with the integrity check and the real op breakdown. |
| BF16 `(1 + w)` fold vs HF's FP32 fold | measured, not changed | `norm_weight_dtype_probe.py`: FP32 device weight moves the norm PCC by ~1e-6 (0.99994260 -> 0.99994183) because `ttnn.rms_norm` emits BF16 regardless. Recorded as limitation 10 with the log. |
| decode RMSNorm is a 14 % single-core hotspot, hidden by the "weight-bandwidth bound" narrative | accepted | README perf section now shows the per-op split with core counts and names it an optimized-decoder target. |
| sliding 131072 tail PCC is window-bounded | accepted | disclosed as limitation 7 and in the capability table. |
| `start_pos` vs previous logical length under-specified | accepted | spelled out in the module docstring and README contract. |
| SDPA sliding-window bug not filed upstream | accepted | limitation 9: this autonomous stage does not open GitHub issues; reproducer + log committed. |
| repro docstring numbers stale vs the committed log | fixed | docstring table replaced with the committed log's numbers. |
| `_chunk_page_table` return annotation wrong | fixed | now `tuple[ttnn.Tensor, bool]`. |
| `num_to_corerange` imported inside the decode hot path | fixed | moved to module scope. |
| fallback audit only covered single-chunk prefill | fixed | `test_no_host_fallback_in_forward` parameterised over `seq_len ∈ {3000, 12345}`. |
| `resolve_layer_kind` rejection untested | fixed | `test_resolve_layer_kind_rejects_unsupported_pairings`. |
| all PCC measured against a BF16 HF reference | fixed | `test_prefill_decode_pcc_vs_fp32_reference` adds an FP32 HF control for both kinds. |
| `__pycache__` inside the stage tree | fixed | excluded from the checkpoint commit via `.gitignore`. |

## 4c. Stage-review round 2 findings and remediation

`$stage-review` round 2 (fresh subagent) verified all 15 round-1 remediations as
real in code + test + artifact and found no new bug introduced by them, but
returned **more-work-needed** on three further findings:

| finding | verdict | what was done |
| --- | --- | --- |
| **P1** `decode_forward` raised `ValueError: max() iterable argument is empty` for any batch with no `batch`-core rectangle on the 11x10 grid (13, 17, 19, 23, 26, 29, 31, …). `nlp_concat_heads_decode` needs a rectangular height-sharded core set; primes above `grid.x` have none. | real bug | `_decode_concat_grid_width()` now returns `None` for those batches and `_concat_heads_decode` falls back to `transpose -> nlp_concat_heads` (shape agnostic). `ttnn.num_cores_to_corerangeset` was evaluated first and rejected: it yields a non-rectangular set that `create_sharded_memory_config(HEIGHT, use_height_and_width_as_shard_shape)` refuses with `bad optional access` for batch 13/17/31/32/63 (probed on device). `test_batched_prefill_decode_pcc` now covers batch 13 and asserts the fallback is the path taken. |
| **P2** headline decode latency was measured only at a 2048-token context on a model advertising 131072. | real | `test_perf_decode_traced` is parameterised over `context ∈ {2048, 131071}` and all four decode windows are profiled. The gap is real and now recorded: `full` goes 3.082 -> 3.575 ms/token, `sliding` is unchanged (window-capped SDPA). |
| **P2** the sliding continuation contract had no coverage for `tail_len < sliding_window` — the exact regime of work-log bug 1. | real gap | `CONTINUATION_SPLITS` now includes `1024+1024` and `64+100`. `64+100` immediately exposed a second real defect: `chunked_scaled_dot_product_attention` requires `chunk_start_idx % q_chunk_size == 0`, which a `full`-layer continuation at `start_pos=64` violated (`TT_FATAL … attrs.chunk_start_idx.value() % q_chunk_size == 0`). `_prefill_sdpa_full` now halves the SDPA chunk until it divides `start_pos`. |
| README/work-log drift: "16 periodic dumps" (8), "12 torch entry points" (13), "Full test suite (49 tests)" above `59 passed`, and an over-general claim that all SDPA `scale` args use `.noconvert()` | fixed | all four corrected; the `.noconvert()` claim now names the two ops that actually use it (`chunked_scaled_dot_product_attention`, `joint_scaled_dot_product_attention`) and notes the plain op does not. |
| sliding prefill always cloned the K/V tail even when nobody consumes it | fixed | `need_tail` is threaded through `_prefill_chunk`/`_prefill_attention`/`_prefill_sdpa_sliding`; the last chunk of a non-continuation prefill no longer builds it (sliding prefill window is 42 ops now, was 46). |
| tile padding could push a prefill past a non-tile-aligned `max_seq_len` | fixed | `from_state_dict` rejects `max_seq_len % 32 != 0`. |
| sliding full-context evidence only covered the final tail carry | fixed | `test_full_context_prefill_tail_pcc` now also checks an **interior** 32-row block at 65536 (the first row of internal chunk 9, entirely dependent on the 8th carried window): 0.998490 sliding / 0.997775 full. |
| no CI assertion tying perf artifacts to the code (op-count-per-iteration) | accepted | recorded as a hard-check gap; the integrity check is documented as a command in the README perf section and was run for all six windows. |

## 4d. Stage-review round 3 findings and remediation

Round 3 verified all round-2 remediations as real and found **no new bug**; it
returned **more-work-needed** on two evidence-accuracy items plus concerns:

| finding | what was done |
| --- | --- |
| **P2** `context_contract.json` `layer_weight_bytes` (968,884,224) did not match its own formula, which evaluates to 967,835,648 (a 1 MiB arithmetic slip). | corrected in `context_contract.json`; the README now states the exact byte count and `≈ 968 MB`. |
| **P2** `logs/fallback_audit.log` and `logs/full_context_tests.log` were generated by an earlier 49-test revision and under-covered the claims citing them (no `12345` fallback params, no interior-row PCC lines). | both re-run against the current code and overwritten (4 passed / 6 passed). |
| batched decode never crossed the 2048 window (prompts were 512..1659) | `test_batched_prefill_decode_pcc` prompts are now `2000 + 37*user` (2000..3147), so per-user decode positions straddle the window on both sides. |
| decode kernel's window semantics were only inferred from end-to-end PCC | new `test_decode_sdpa_sliding_window_semantics` probes `paged_scaled_dot_product_attention_decode` directly against an explicit torch mask at `cur_pos ∈ {2047, 2048, 2049, 5000}` (PCC >= 0.99959, bar 0.999). |
| multi-chunk prefill was only exercised for `user_id == 0` | new `test_multi_chunk_prefill_nonzero_user`: 12345-token (two-chunk) prefill + decode off cache slot 2 of a 4-slot cache. |
| `test_real_weights_*` silently `skip`ped when the checkpoint was absent | now raises unless `MG_ALLOW_MISSING_WEIGHTS=1` is set, so the contract's real-weight requirement cannot degrade to a skip unnoticed. |
| README/work-log still said "batch 4 and 32" in two prose spots | corrected to 4/13/32. |
| no CI assertion ties perf artifacts to the op graph; the fallback guard is Python-level only; long-context `full` decode is measured over an unfilled cache | accepted and left as recorded hard-check gaps / disclosed limitations — see README limitations and the perf section's integrity-check command. |

## 5. Commands

Weight statistics (one-off, from the real checkpoint) →
`tests/layer_weight_stats.json`:

```bash
python - <<'PY'
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
# safe_open over the shards named by model.safetensors.index.json for layers 0 and 3
PY
```

Capacity probes at the advertised context (both layer kinds):

```bash
python models/autoports/meta_models_muse_glimmer_30b/tests/functional_decoder_capacity_probe.py \
    --seq-len 131072 --layer 0 --decode      # logs/capacity_probe_131072_layer0.log
python models/autoports/meta_models_muse_glimmer_30b/tests/functional_decoder_capacity_probe.py \
    --seq-len 131072 --layer 3 --decode      # logs/capacity_probe_131072_layer3.log
# CAPACITY_PROBE_PASS mode=prefill seq_len=131072 output_shape=(1, 1, 131072, 6656) tail_finite=True
# CAPACITY_PROBE_PASS mode=decode  cur_pos=131071
# (both probes, the SDPA repro and the norm-dtype probe were re-run after the final
#  implementation edit so every evidence log postdates the code it exercises)
```

Both pass, so **no capability reduction was taken** and
`doc/context_contract.json` records `current_supported_context = 131072`.

Full test suite:

```bash
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
    -q --no-header --junitxml=models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/test_results.xml
# 73 passed in 396.92s   -> logs/full_test_run.log, test_results.xml
```

Full-context subset (also runs inside the suite above):

```bash
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
    -k full_context -q --no-header          # logs/full_context_tests.log, 6 passed
```

Fallback audit:

```bash
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
    -k fallback -q --no-header              # logs/fallback_audit.log, 4 passed
```

Watcher (separate run from any profiling):

```bash
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=$D/watcher \
python -m pytest "$T::test_prefill_pcc[12345-sliding]" "$T::test_prefill_pcc[12345-full]" \
  "$T::test_decode_pcc[3000-sliding]" "$T::test_decode_pcc[3000-full]" \
  "$T::test_continuation_prefill_pcc[64-100-sliding]" \
  "$T::test_continuation_prefill_pcc[4096-3000-sliding]" "$T::test_continuation_prefill_pcc[4096-3000-full]" \
  "$T::test_traced_decode_advances_positions[sliding]" "$T::test_traced_decode_advances_positions[full]" \
  "$T::test_batched_prefill_decode_pcc[13-sliding]" "$T::test_batched_prefill_decode_pcc[4-full]" \
  "$T::test_multi_chunk_prefill_nonzero_user[sliding]" "$T::test_decode_sdpa_sliding_window_semantics[2049]" \
  "$T::test_prefill_seq_len_equals_max_and_chunk[sliding]" -q --no-header
# 14 passed in 104.67s  -> logs/watcher_run.log, watcher/watcher.log.gz
# watcher.log: 11867 lines, 11 dumps, 0 x {Watcher detected, tripped, sanitize, TT_ASSERT, DEBUG_ASSERT, fault}

# Watcher writes $D/watcher/generated/watcher/. The repo-root .gitignore excludes any
# path component named "generated", so watcher.log and kernel_names.txt were moved up
# to $D/watcher/ (and the fabric/inspector YAML noise dropped) before committing.
mv $D/watcher/generated/watcher/{watcher.log,kernel_names.txt} $D/watcher/ && rm -rf $D/watcher/generated
# then `gzip -9` watcher.log, kernel_names.txt and the two decode ops CSVs: the repo's
# check-large-files pre-commit hook rejects anything over 500 KB.
```

Profiling (4 sequential runs; `python -m tracy` loses quoted `-k` expressions,
so node ids are used):

```bash
for kind in sliding full; do
  MG_PERF_DECODE_ITERS=8 python -m tracy -r -p -v -m pytest "$T::test_perf_prefill[$kind]"
  cp $(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1) $D/tracy/$kind/prefill_ops.csv
  tt-perf-report $D/tracy/$kind/prefill_ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
      --no-summary --no-advice > $D/tracy/$kind/prefill_perf_report.txt
  tt-perf-report $D/tracy/$kind/prefill_ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
      --csv $D/tracy/$kind/prefill_perf_report.csv --no-advice > $D/tracy/$kind/prefill_perf_report.console.log
done
# same shape with test_perf_decode_traced / PERF_DECODE / decode_*
```

Logs: `logs/tracy_{prefill,decode}_{sliding,full}.log`.  Integrity check after
each capture (must be 0 everywhere, and every op-code count in the filtered CSV
must divide by the replay count):

```bash
grep -c "markers were dropped" $D/logs/tracy_*_*.log
```

Norm-weight dtype probe:

```bash
python models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/norm_weight_dtype_probe.py
# weight dtype bfloat16 (shipped)   -> PCC 0.99994260, output dtype DataType.BFLOAT16
# weight dtype float32              -> PCC 0.99994183, output dtype DataType.BFLOAT16
```

## 5b. Stage review outcome

`$stage-review` round 4 returned **clean-pass** with no required work, after
verifying every round-3 remediation in code + test + artifact, re-deriving every
number in README / work_log / context_contract from the committed artifacts
(194 PCC lines byte-identical to the run log, all six perf windows recomputed
from the CSVs, watcher line/dump counts, byte budgets, weight stats re-read from
the real safetensors shards), and independently re-deriving the layer semantics
against the HuggingFace source.  Two non-blocking suggestions from that round
were also applied: limitation 6 now names HF-reference tractability (not DRAM)
as the reason batch 32 x 131072 was not run, and the three op-level probe logs
were regenerated so none predates the final implementation file.

### Artifact regeneration after the pre-commit hooks

`black`/`isort` reformatted the four Python sources when the stage was first
committed (whitespace and import order only; no statement changed).  To keep the
evidence strictly newer than the code it exercises, the **entire** evidence set —
full suite, fallback and full-context subsets, watcher run, all six Tracy
windows, and the three op-level probes — was regenerated against the committed,
formatted sources and the numbers below updated accordingly.

## 6. Results

* 73/73 tests pass; 194 PCC checks, minimum **0.99742** (bar 0.995).
* Full advertised context 131072 validated in both prefill and decode, both
  layer kinds; non-aligned 130073 also validated.
* Batch 4, 13 and 32 with ragged prompt lengths (straddling the 2048 sliding window)
  and ragged decode positions. 13 is prime and > the 11-wide grid, so it exercises
  the decode head-concat's shape-agnostic fallback.
* Real-checkpoint weights pass at PCC 0.9974–0.9992.
* Determinism: bit-identical over 3 repeats, prefill and decode, both kinds.
* Watcher clean.
* Caller-chunked (`start_pos > 0`) prefill validated against a single-shot HF
  prefill for both kinds, plus a decode past the continuation.
* FP32 HF control matches the BF16 control to ~1e-3.
* Warmed prefill (8192 tok): 101.23 ms sliding / 99.38 ms full device time.
  Warmed traced decode (drop-free 8-replay captures, both ends of the context):
  sliding **3.163 / 3.160 ms/token** at 2048 / 131071 (its SDPA is window-capped);
  full **3.080 ms/token** at 2048 and **3.575 ms/token** at 131071.

## 7. Checkpoint commits

Local only — nothing was pushed. Only `tt-metal` was touched, and only under
`models/autoports/meta_models_muse_glimmer_30b/` (`git status --porcelain` was
clean apart from that tree before each commit).

| repo | branch | commit | contents |
| --- | --- | --- | --- |
| `/home/ttuser/dev/muse-glimmer/tt-metal` | `agentic-research/hous/muse-glimmer-30b` | `2e2acc13f960200541da67de286223293542f5e4` | implementation, tests, reference harness, capacity probe, docs and the first evidence set |
| `/home/ttuser/dev/muse-glimmer/tt-metal` | `agentic-research/hous/muse-glimmer-30b` | `6363a7c9badad9d03250ea1a4539eb29a85f26bc` | evidence regenerated against the black/isort-formatted sources, docs updated to those numbers |

A third, doc-only commit on the same repo and branch carries this table itself;
its own SHA cannot be recorded inside it, so read it with
`git log --oneline -1` (subject: "record stage checkpoint SHAs in the work log").

No changes were made outside the repo except installing `transformers==5.15.0`
and `tt-perf-report==1.2.8` into the model-specific venv
`/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv` (see section 1).

## 8. Not done in this stage (by scope)

Optimized decoder (dtype/sharding/fusion), multichip, full model, vLLM.  No
files outside `models/autoports/meta_models_muse_glimmer_30b/` were modified.
