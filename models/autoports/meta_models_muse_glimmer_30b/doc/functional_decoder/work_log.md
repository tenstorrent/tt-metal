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
| **P2** the 32-replay decode Tracy captures logged `Profiler DRAM buffers were full, markers were dropped!` (360x sliding, 5x full) and under-counted ops, making decode look ~13 % faster than reality. | real | re-profiled all four windows with `MG_PERF_DECODE_ITERS=8`; all four logs now have **0** drop warnings and every op-code count is an exact multiple of the replay count. Corrected numbers at the time: sliding 3.165 ms/token (was 2.766), full 3.081 ms/token (was 2.974) — superseded by the final committed captures (3.163 / 3.080; the authoritative table is the README perf section, re-derived by `bench/summarize_perf.py`). README perf section rewritten with the integrity check and the real op breakdown. |
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
| **P2** headline decode latency was measured only at a 2048-token context on a model advertising 131072. | real | `test_perf_decode_traced` is parameterised over `context ∈ {2048, 131071}` and all four decode windows are profiled. The gap is real and now recorded: `full` goes 3.082 -> 3.575 ms/token at the time of that round (3.080 -> 3.575 in the final committed captures), `sliding` is unchanged (window-capped SDPA). |
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

## 4e. Stage re-run 2026-08-12: live re-verification and contract restoration

The functional-decoder stage goal was re-issued after the fused-decoder stage had
already landed on this branch.  The implementation and tests were *not* changed —
`git rev-parse HEAD:<path>` gives blob `0f949dc99d23e1148cb94da4bb39be9600b015cd`
for `tt/functional_decoder.py` and `175b59cbeee9bd29fd51c2b0ac4322f43bf5bddc` for
`tests/test_functional_decoder.py`, identical to commit `2e2acc13f96` — so this
round re-verified the committed evidence on live hardware, closed the one real
gap the later stage had opened in the shared context contract, and made the
remaining hand-transcribed numbers re-derivable.

Device health first (`tt-smi` is still not installed on this host):

```bash
python -c "import ttnn; m=ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0); \
           print('MESH_SMOKE_OK', m.arch(), m.compute_with_storage_grid_size()); ttnn.close_mesh_device(m)"
# MESH_SMOKE_OK Arch.BLACKHOLE 11-10
```

### What was re-run

| run | command | result |
| --- | --- | --- |
| full suite | `python -m pytest .../tests/test_functional_decoder.py -q --no-header --junitxml=logs/reverify_2026_08_12_test_results.xml` | **73 passed in 394.88s**; `logs/reverify_2026_08_12_full_test_run.log` |
| PCC comparison | `python bench/summarize_pcc.py --compare logs/reverify_2026_08_12_full_test_run.log` | `RERUN_IDENTICAL 194 checks bit-identical to the committed run` — 0 missing, 0 extra, 0 differing values |
| watcher | the same 14-node-id command as section 5, with `TT_METAL_LOGS_PATH=$D/watcher_reverify` | **14 passed in 105.47s**; `bench/check_watcher.py watcher_reverify/watcher.log.gz` → `WATCHER_CLEAN` (11868 lines, 22 `Dump #` boundaries = 11 dumps, 5809 `k_ids:`, 4 attach + 4 detach, **0** fatal messages) |
| perf | not re-captured | see below |

Tracy captures were deliberately **not** re-taken.  The code they exercise is
byte-identical to the code they were captured against, the captures are committed
with their raw ops CSVs, and re-capturing would move the numbers by measurement
noise — which would also invalidate the fused stage's committed `baseline`
columns, since those quote exactly these six windows.  Instead every quoted perf
number is now *re-derived* from the committed CSVs:

```bash
python bench/summarize_perf.py            # writes logs/perf_summary.txt
python bench/summarize_perf.py --check    # exit 1 on drift or capture-integrity failure
```

It reproduces the README table exactly (101.229 / 99.375 ms prefill; 3.163 /
3.160 / 3.080 / 3.575 ms per traced decode token; 42 / 24 / 64 / 32 ops per
iteration) and re-runs both integrity checks per window: zero
`markers were dropped` in the Tracy log, and every `OP Code` row count an exact
multiple of the replay count read back out of that same log (`iters=8`).

### Contract gap that was fixed

The fused-decoder stage rewrote the shared `doc/context_contract.json` top level
for itself and reduced the functional stage's record to a four-field
`previous_stage` stub (test totals and `min_pcc` only).  The stage contract
requires the contract to record the *functional*-decoder tested prefill/decode
context and capability reduction, so the stub was replaced by a full
`functional_decoder` block — advertised/supported context, `capability_reduction:
none`, tested prefill (131072, non-aligned 130073) and decode (position 131071)
contexts with per-check PCC, batch/non-aligned/continuation/sliding-window/
non-zero-slot coverage, real weights, FP32 control, determinism, fallback audit,
watcher, capacity probes, byte budget, perf and test counts.

The only fused-stage-authored thing this touches is that `previous_stage` stub,
which is *deleted*: the new `functional_decoder` block is a strict superset of
its four fields, nothing in the repo or in `.agents/` reads `previous_stage`
(repo-wide grep), and the fused stage's own generator still agrees with its own
committed run afterwards.  Every fused-owned value is byte-identical.

```bash
python doc/functional_decoder/bench/refresh_context_contract.py --check
# functional_decoder block matches the committed run   <- the gate for this stage's record
python doc/fused_decoder/bench/refresh_context_contract.py --check
# context_contract.json matches the committed suite run <- fused block undisturbed
python .agents/scripts/check_context_contract.py --hf-model meta-models/Muse-Glimmer-30B \
    --stage functional_decoder --require-contract
# Context contract OK ...: target=131072, supported=131072 (full HF context).
# (that runner script validates the *top-level* context fields and the absence of
#  smaller caps under tt/; its --stage argument is parsed but unused, so it is not
#  evidence about the functional block itself.)
```

### New evidence-regeneration scripts (`doc/functional_decoder/bench/`)

Round 3 of the original review found stale hand-transcribed numbers, and the
later stages answered that with generator scripts.  The functional stage now has
the same:

| script | owns |
| --- | --- |
| `summarize_pcc.py` | `logs/pcc_summary.txt` (194 rows, ascending) from the suite log; `--check`; `--compare <log>` for rerun diffing |
| `summarize_perf.py` | `logs/perf_summary.txt` from the six committed `*_perf_report.csv` windows + their Tracy logs, including the two capture-integrity checks; `--check` |
| `check_watcher.py` | the watcher-clean verdict (fatal-pattern count vs benign structure) for any committed watcher log |
| `refresh_context_contract.py` | the measured fields of the `functional_decoder` block of `doc/context_contract.json`; `--check` |

`summarize_pcc.py` regenerated `logs/pcc_summary.txt` in the generated format;
all 194 values are identical to the previous hand-made file (verified by parsing
both and diffing the label→value maps), and the README's eight-lowest table is
now literally the first eight rows of that file.

## 4f. Stage-review round 5 (2026-08-12) findings and remediation

`$stage-review` round 5 (fresh independent subagent, read-only, no device) was
given the re-issued goal contract, both skills, the live worktree and the staged
diff.  Verdict: **clean-pass, no required work**.  It re-derived every headline
number independently rather than trusting the prose: 74-way test decomposition
and 194-way PCC decomposition (so no check is silently absent), worst PCC
0.997422, all six perf windows including the raw-nanosecond reconciliation, both
watcher logs, the weight stats against the real safetensors, the byte budgets,
and the HF layer semantics line by line (norm epsilons, `(1+w)` fold, scale-less
QK norm on Q/K but not V, `qk_scale_factor` folding into the SDPA scale, sigmoid
output gate on the normed hidden states, NoPE on `full` layers, NeoX
`rotate_half`, `sliding_window_overlay = kv_idx > q_idx - W`).

It also raised non-blocking concerns.  All of the actionable ones were fixed in
this round even though none gated the pass:

| finding | class | what was done |
| --- | --- | --- |
| **RoPE base read from the wrong HF field.** `rope_theta` came from `layer_rope_theta[layer_idx]`, but HF uses that list *only* as a boolean NoPE gate (`modeling_muse_glimmer.py`: `position_embeddings if config.layer_rope_theta[i] else None`) and takes the rotary base from the model-level `rope_parameters["rope_theta"]`.  Identical (`500000.0`) in this checkpoint, so PCC could never see it; latent for any revision that moved one and not the other, and `_require_muse_glimmer_text_config` did not pin `rope_parameters`. | real latent gap | new `_rope_theta()` reads `rope_parameters["rope_theta"]`; `resolve_layer_kind` now documents `layer_rope_theta[i]` as the gate it is; `rope_parameters` added to the pinned-config guard; new `test_rope_base_is_the_model_level_rope_parameters` asserts the built layer's base equals `rope_parameters` and that a revision moving the base without the gate raises.  Suite re-run post-change: **74 passed**, and `bench/summarize_pcc.py --compare` shows all **194 PCC values bit-identical** to the pre-change run, which is what proves the edit is numerically a no-op on this checkpoint. |
| **`check_watcher.py` certified an empty log as clean** — it only counted fatal patterns, and a truncated or stubbed log has none of those either. | real gate weakness | the benign-structure counters are now *minimums* (>= 1000 lines, >= 2 `Dump #` boundaries, >= 100 `k_ids:`, >= 10 stack-usage rows, >= 1 attach and >= 1 detach); a log that misses any of them exits 1 with `WATCHER_LOG_NOT_A_REAL_RUN`.  Probed: empty log and a 100-line truncation both now fail, both committed logs still pass. |
| **`summarize_perf.py` never tied the filtered `*_perf_report.csv` back to the raw ops CSV** it was produced from, so a hand-edited filtered CSV would have passed the gate. | real gate weakness | the script now sums `DEVICE KERNEL DURATION [ns]` between the signposts of the committed raw capture (gz-aware) and requires the op count to match the filtered row count and the totals to agree within 0.5 us.  All six windows reconcile at **0.000 us** (`logs/perf_summary.txt`).  Implementation note worth keeping: the raw rows must be taken in *file order*, because in a traced decode window every replayed op carries the host timestamp of trace **capture** (ops end at 6491019686 while `PERF_DECODE` is at 6504387979), so sorting by `HOST START TS` would move every op outside its own window. |
| **tt-metal #16667 workaround undisclosed in this stage's limitations** (decode stages the fused QKV in L1 because the op's interleaved-DRAM reader zeroes odd Q rows on Blackhole). | doc gap | README limitation 13, with the measured cost re-derived from the committed capture: one `CopyDeviceOperation`, 2.13 us/token, 0.07 % of a decode step. |
| **The multimodal / logit-softcapping parts of the checkpoint were nowhere flagged** for later stages (`MuseGlimmerForConditionalGeneration`, vision tower + adapter, `image_token_id`/`video_token_id`, `final_logit_softcapping=20.0`, `output_multiplier=0.196`). | doc gap | folded into README limitation 11 (Scope) as an explicit warning to the full-model/serving stages. |
| **`layer_rope_theta` described as "the theta"** in README/work log. | doc gap | the architecture table and the module docstring now call it the gate and point at `rope_parameters`. |
| **"Nothing owned by the fused stage was touched" was imprecise** — the staged diff does delete the fused stage's `previous_stage` stub. | wording | section 4e now says exactly that, and why it is safe (no consumer anywhere in the repo or `.agents/`; the new block is a strict superset; `doc/fused_decoder/bench/refresh_context_contract.py --check` still passes). |
| **`.agents/scripts/check_context_contract.py --stage` is parsed and never used**, so citing it as evidence for the *functional* record overstated what it validates (it only reads the top-level context fields). | wording | section 4e now names `bench/refresh_context_contract.py --check` as the gate for the functional block and the runner script as the top-level context guard only. |
| `doc/fused_decoder/bench/summarize_pcc.py` has no `argparse`, so `--check` is silently ignored and it *writes*. | cross-stage | not this stage's file and not changed here.  Recorded for the fused-stage owner: that script must not be treated as a read-only gate.  The functional stage's four scripts all honour `--check`. |
| Fallback guard is attribute-level only; no runtime host-op counter. | accepted (already recorded) | unchanged — still a disclosed hard-check gap. |
| Residual risks it listed (window-bounded sliding evidence at 131072, reduced 32-row full-context harness, batch 32 x 131072 untested for HF-reference tractability, prefill SDPA pinned to 128/128 by the sliding-window chunk bug, no upstream issue filed, model-specific `transformers` venv) | accepted | all already disclosed as README limitations 1, 5, 6, 7, 9 and 12; limitation 12 now also spells out that a CI job on the repo pin fails at import. |

Because the code changed, the primary evidence was regenerated against the
edited sources: `logs/full_test_run.log` and `test_results.xml` (74 passed),
`logs/pcc_summary.txt`, the `functional_decoder` contract block, and a fresh
watcher run into `watcher/` (the pre-change reproduction is kept as
`watcher_reverify/`).

The six Tracy captures were **not** re-taken, deliberately.  The edit is confined
to config resolution (`_rope_theta`, the pinned-config guard), docstrings and one
new non-device-graph test — no runtime op was added, removed or reordered — and
the post-change suite reproduces all 194 PCC values bit-identically, which also
proves the cos/sin tables are unchanged.  Re-capturing would only add measurement
noise and would desynchronise the fused stage's committed `baseline` columns,
which quote exactly these six windows.  `bench/summarize_perf.py` now proves each
committed filtered CSV against its own raw capture, so the captures are verifiable
in place.

One cross-stage consequence of that edit, beyond the `previous_stage` stub:
`tt/fused_decoder.py` imports `_require_muse_glimmer_text_config` from this module
and calls it, so the new `rope_parameters` pin now also gates the committed fused
decoder.  Verified benign — the released config's `rope_parameters` is exactly
`{"rope_theta": 500000.0, "rope_type": "default"}` (checked from `config.json` and
through `AutoConfig`), i.e. the only config either stage supports, and
`doc/fused_decoder/bench/refresh_context_contract.py --check` still passes.

## 4g. Stage-review round 6 (2026-08-12) findings and remediation

Round 5's clean-pass was on the pre-code-change worktree, so the round-5
remediations were re-reviewed by a fresh independent subagent.  Round 6 verified
every remediation as real in code + test + artifact (including re-deriving the six
perf windows, all 194 PCC values, the 74-test decomposition, both watcher logs and
the layer semantics, and independently confirming the file-order argument in the
new raw-capture reconciliation with tamper tests), and accepted the
Tracy-not-re-taken justification after diffing the code delta itself.  It returned
**more-work-needed** on three documentation-accuracy findings, all of which came
from this round's own regeneration:

| finding | what was done |
| --- | --- |
| **P2** `context_contract.json` `functional_decoder.tested.watcher.tests` said **14** while the watcher artifacts it cites are the post-change **15**-test run (`logs/watcher_run.log`: `15 passed`, 107.86s in the capture round 6 read).  `refresh_context_contract.py --check` could not see it because the watcher block was hand-maintained. | the count is now *derived*: `watcher_tests()` parses the pytest summary out of `logs/watcher_run.log`, refuses to refresh if that log reports failures or has an ambiguous number of summary lines, and `--check` fails on drift.  Contract now records 15. |
| **P2** work-log section 5 documented a 14-node-id watcher command and `14 passed in 104.67s`, which does not reproduce the committed 15-test artifact, and the canonical command appeared nowhere in full. | section 5 now carries the exact 15-node-id command that produced `watcher/`, its real result (`15 passed`, 109.81s in the final committed capture), the `check_watcher.py` verdict, and a note that the same command minus the last node id is the `watcher_reverify/` pre-change reproduction (104.67s originally, 105.47s on 2026-08-12). |
| **P2** "every evidence log postdates the code it exercises" and "the **entire** evidence set was regenerated" became false for the 2026-08-11 subset/probe logs once `tt/functional_decoder.py` was edited on 2026-08-12. | both sentences are now scoped to the 2026-08-11 revision and point here.  Deliberately kept from 2026-08-11: the six Tracy captures (justified in 4f), `logs/fallback_audit.log` and `logs/full_context_tests.log` (subsets of the suite; the same params run green inside the post-change 74-test suite, see `logs/full_test_run.log`), `logs/capacity_probe_131072_layer{0,3}.log`, `logs/norm_weight_dtype_probe.log` and `logs/sdpa_sliding_window_chunk_repro.log` (the last one does not import this module at all — it probes `ttnn.transformer.scaled_dot_product_attention` directly).  None of them exercises a line the 2026-08-12 edit touched. |
| **Concern** `test_rope_base_is_the_model_level_rope_parameters` did not discriminate the read site: with the gate and the base both `500000.0`, its assertions also passed under the old implementation. | strengthened: it now builds a decoder from a config whose `layer_rope_theta[layer_idx]` is `12345.0` (still truthy, so still a `sliding` layer) and asserts the base is `500000.0`.  Mutation-tested — reverting `from_state_dict` to `layer_rope_theta[layer_idx]` makes it fail with `assert 12345.0 == 500000.0`. |
| **Concern** the same wrong read survives in `tt/fused_decoder.py:647` and the untracked `tt/optimized_decoder.py`, which are the modules that will ship. | out of scope for this goal (functional decoder only) and **not** changed here.  Recorded for those stages' owners: both still do `rope_theta=float(text_config.layer_rope_theta[layer_idx])`; invisible on this checkpoint, wrong on any revision that moves the base without the gate.  The `rope_parameters` pin they inherit from this module at least makes such a revision fail loudly rather than silently mis-rotate. |
| **Concern** the `check_watcher.py` structural minimums are calibrated to this suite's ~11.8k-line logs, so a legitimately shorter run would be rejected. | disclosed in the README watcher section and in the script's docstring; it fails conservatively (it never certifies a bad log). |
| **Concern** historical perf figures inside the round-2/3 remediation tables (3.165 / 3.081 / 3.082 ms) predate the final captures. | annotated in place as the then-current numbers, pointing at the authoritative README perf table (3.163 / 3.080 / 3.575). |
| **Concern** README limitation 11 rounded `output_multiplier` to `0.196`. | quoted exactly: `0.19611613513818404`. |
| Hard-check gaps it recorded (no automated tie between a Tracy capture and the code revision; prose/coverage contract fields still hand-maintained apart from the watcher count; attribute-level fallback guard; no in-suite assertion that watcher was enabled) | accepted and left recorded; the watcher-count one is now closed by `watcher_tests()`. |

Because the test file changed again, the suite, watcher run, PCC summary and
contract block were regenerated once more against the final sources: **74 passed
in 395.38s** (the same 194 PCC values, bit-identical for the third time) and a
**15 passed in 109.81s** watcher run, still `WATCHER_CLEAN`.  All four `--check`
gates and the runner's context-contract check pass on the final tree.

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
#  implementation edit of the 2026-08-11 revision, so every evidence log postdated
#  the code it exercised at that point.  Section 4g lists which of those logs are
#  deliberately kept after the 2026-08-12 config-resolution edit and why.)
```

Both pass, so **no capability reduction was taken** and
`doc/context_contract.json` records `current_supported_context = 131072`.

Full test suite:

```bash
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
    -q --no-header --junitxml=models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/test_results.xml
# 74 passed in 395.38s   -> logs/full_test_run.log, test_results.xml
#   (73 before section 4f added test_rope_base_is_the_model_level_rope_parameters)
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

Watcher (separate run from any profiling).  This is the **canonical** command —
the 15 node ids that produced the committed `watcher/watcher.log.gz` and
`logs/watcher_run.log`:

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
  "$T::test_prefill_seq_len_equals_max_and_chunk[sliding]" \
  "$T::test_rope_base_is_the_model_level_rope_parameters" -q --no-header
# 15 passed in 109.81s  -> logs/watcher_run.log, watcher/watcher.log.gz
# watcher.log: 11867 lines, 11 dumps, 0 x {Watcher detected, tripped, sanitize, TT_ASSERT, DEBUG_ASSERT, fault}
# python $D/bench/check_watcher.py  ->  WATCHER_CLEAN  (and the contract's
#   tested.watcher.tests is derived from logs/watcher_run.log by
#   bench/refresh_context_contract.py, so it cannot drift from this run again)
#
# The same command without the last node id (14 tests, 104.67s originally and
# 105.47s when it was reproduced on 2026-08-12 into $D/watcher_reverify with
# TT_METAL_LOGS_PATH=$D/watcher_reverify) is what produced
# watcher_reverify/watcher.log.gz and logs/reverify_2026_08_12_watcher_run.log —
# the pre-code-change reproduction described in section 4e.

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
evidence strictly newer than the code it exercises, the entire evidence set of
that revision — full suite, fallback and full-context subsets, watcher run, all
six Tracy windows, and the three op-level probes — was regenerated against the
committed, formatted sources and the numbers below updated accordingly.

That statement is scoped to the 2026-08-11 revision.  The 2026-08-12
config-resolution edit (section 4f) regenerated the full suite, the PCC summary,
the contract block and the watcher run; section 4g lists the six Tracy captures
and five subset/probe logs that are intentionally kept from 2026-08-11, and why
keeping them is sound.

## 6. Results

* 74/74 tests pass; 194 PCC checks, minimum **0.99742** (bar 0.995).
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
