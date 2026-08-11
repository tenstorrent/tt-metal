# Functional decoder work log — meta-models/Muse-Glimmer-30B

Stage 01 (`$functional-decoder` + `$tt-device-usage`). Chronological record of what was
run, what broke, and what the evidence is. All commands were run from the repo root with
`source python_env/bin/activate`.

## Environment

| item | value |
|---|---|
| host | Ryzen 9700X (8C/16T), 249 GB RAM |
| devices | 4x Blackhole `p300c` (two p300 boards), `tt-smi -ls --local` clean |
| mesh used | `(1, 1)` — functional bringup is single-device by design; multichip is stage 03 |
| Blackhole compute grid | 11 x 10 (`compute_with_storage_grid_size`) |
| device DRAM | 8 banks x 4,272,341,376 B = **34.18 GB** per chip |
| device L1 | 1,461,504 B per core |
| transformers | 5.15.0 (`muse_glimmer` is upstream, no `trust_remote_code` shim needed) |
| torch | 2.11.0+cpu |
| ttnn | source build at `/home/jashan/tt-metal/ttnn/ttnn/__init__.py` |

Device health / mesh smoke before starting (`$tt-device-usage`):

```bash
timeout 90 tt-smi -ls --local            # 4x Blackhole p300c listed, resettable
python - <<'PY'                          # mesh open/close smoke
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
print(mesh.arch(), mesh.compute_with_storage_grid_size())
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: `Arch.BLACKHOLE`, grid `11-10`, `MESH_SMOKE_OK`.

**Five** device incidents happened later in the stage; all are recorded in full, with
artifacts, and none was a model correctness or performance result:

| # | what | where |
|---|---|---|
| 1 | a killed pytest left the devices undiscoverable (`tt-smi -ls --local` aborted in `TopologyDiscovery::init_device`) | section 4 |
| 2 | a genuine op hang at prefill length 2080, captured with `tools/tt-triage.py` before the process was killed | section 3, bug 7 |
| 3 | `Read 0xffffffff over PCIe ID 1: the board should be reset` during perf re-collection | section 8 |
| 4 | the out-of-bounds page-table reproducer hung the device, and left it in a state where the *next* `open_mesh_device` also hung even though `tt-smi -ls --local` still listed all four boards | section 9, "Device incidents 4 and 5" |
| 5 | two suite runs collided (a chained `fast; long` command kept going after the fast suite failed, so a second invocation ran against a device the first still held) — 46 device tests errored at fixture setup | section 9, "Device incidents 4 and 5" |

Each was recovered with one bounded list / `tt-smi -r` / list + mesh-smoke cycle; no second
reset was ever needed, no lock clearing beyond one orphaned `/dev/shm/tt_device_*` segment
(which the next tt-metal run reclaims itself), and no operator intervention. Incident 4 is
the one that shows why a bare `tt-smi -ls --local` is not a health check: it listed all four
boards while the next `open_mesh_device` still hung, and only the reset + mesh smoke made the
device usable again.

Environment change made by this stage: `uv pip install tt-perf-report` (1.2.8) into
`python_env`. Its dependency resolution also moved `matplotlib` 3.11.1 -> 3.10.9.

## 1. Reading the HF model

* `config.json` -> `text_config` (`MuseGlimmerTextConfig`), `architectures =
  [MuseGlimmerForConditionalGeneration]`. Only the text decoder is in scope for this stage.
* `transformers/models/muse_glimmer/modeling_muse_glimmer.py` read line by line:
  * `MuseGlimmerTextDecoderLayer.forward`: sandwich norms
    (`input_layernorm` eps 1e-5, `post_attention_layernorm` eps 1e-8, residual,
    `pre_feedforward_layernorm` eps 1e-5, SwiGLU MLP, `post_feedforward_layernorm` eps
    1e-8, residual).
  * `MuseGlimmerTextCenteredRMSNorm` is `normed * (1 + w)` (weights stored around 0).
  * `MuseGlimmerTextAttention.forward`: q/k/v (no bias) -> `qk_norm` (weight-less RMSNorm
    over `head_dim`, eps `rms_norm_eps`) -> `q *= qk_scale_factor` (3.87) -> RoPE (only if
    the layer gets `position_embeddings`) -> SDPA with `scaling = head_dim ** -0.5` ->
    `attn * sigmoid(gate_proj(xn))` (gated attention) -> `o_proj`.
  * **NoPE**: `MuseGlimmerTextModel.forward` passes
    `position_embeddings=position_embeddings if self.config.layer_rope_theta[i] else None`.
    `layer_rope_theta` is 0 on every `full_attention` layer, so those layers apply *no*
    rotary embedding. There is a single rotary embedding module built from
    `rope_parameters["rope_theta"] = 500000.0`; `layer_rope_theta` is used only as this
    on/off marker (verified in `modeling_muse_glimmer.py:513-524`).
  * `layer_types` = `[sliding, sliding, sliding, full] * 13` (52 layers), so exactly two
    decoder-layer kinds exist. Representative layers: 0 (`sliding_rope`) and 3
    (`full_nope`).
  * `final_logit_softcapping` (20.0) and `output_multiplier` (0.196...) are applied in
    `MuseGlimmerForConditionalGeneration.forward` to the **logits**
    (`modeling_muse_glimmer.py:1255-1260`), not inside the decoder layer — recorded here
    for the full-model stage, out of scope for this file.
* Sliding-window semantics cross-checked between HF and TTNN:
  * HF `masking_utils.sliding_window_overlay`: `kv_idx > q_idx - sliding_window` (plus
    causal), i.e. a window of exactly `sliding_window` tokens including the current one.
  * TTNN prefill SDPA `sliding_window_geometry.hpp`: "causal window: left = window - 1
    tokens behind the diagonal".
  * TTNN decode SDPA `rt_args_common.hpp`: `window_start = cur_pos + 1 - window`.
  * All three agree; `test_reference_matches_hf[3000-*]` and
    `test_sliding_window_is_enforced` cover the boundary empirically.

## 2. Host reference

`reference/hf_reference.py` re-drives the HF submodules in HF's order so the golden path
can (a) evaluate one *block* of query positions against a K/V prefix, (b) fill a long K/V
prefix cheaply, and (c) run a single decode step — none of which
`MuseGlimmerTextDecoderLayer.forward` can do at a 131072-token context (the eager
attention matrix alone would be `[1, 32, 131072, 131072]`).

Fidelity is pinned by `test_reference_matches_hf`: the re-drive is compared against the
untouched `layer.forward` with HF's own `create_causal_mask` /
`create_sliding_window_causal_mask` at seq 512 and 3000 (past the 2048 window), for both
layer kinds. Measured PCC `1.0` (eager) and `> 1 - 1e-10` (torch-SDPA backend).

Real weights are read straight out of the safetensors shards for one layer at a time
(`load_real_layer_state_dict`), so no test ever materialises the 60 GB checkpoint.
`scripts/dump_weight_stats.py` records per-tensor shape/dtype/mean/std/absmax into
`weight_stats.json`; the fast suite generates synthetic weights from those stats, with the
**real** shapes and scales.

## 3. Bugs found and fixed during bringup

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | `chunked_scaled_dot_product_attention(..., scale=0.34)` -> "incompatible function arguments" | the op's nanobind binding declares `nb::arg("scale").noconvert()` on a `std::optional<float>`, so no Python float can bind (`sdpa_nanobind.cpp:381`) | apply `qk_scale_factor` to Q (where HF applies it) and let the op use its default `1/sqrt(head_dim)`; the plain and decode SDPA ops still take an explicit `scale` |
| 2 | `TT_FATAL: K's sequence length must be >= Q's sequence length + chunk_start_idx` at seq 32 / 2080 / 3000 / 12345 | the tail Q chunk was zero-padded up to a fixed `q_chunk_size`; the op then thinks the causal prefix runs past the page table (`kv_length = blocks * block_size`) | `_chunked_sdpa_chunk_sizes`: pick the largest candidate that is `<= prefill_sdpa_q_chunk` **and** *divides* the real chunk length, and pass Q unpadded |
| 3 | `TT_FATAL: Page table batch size must match input batch size. Got Page table: 4, Input: 1` | per-user prefill into a shared 4-slot page table: chunked SDPA requires `page_table.shape[0] == q.shape[0]` | `_rows_page_table` gathers exactly the rows of the input batch, in batch order; `user_ids` became a host `list[int]` of cache slots |
| 4 | `TT_FATAL: Input Tensor is not allocated` on *every* prefill | `ttnn.slice` returns the **input tensor** for a full-range slice (`slice.cpp:181`) and the returned handle shares its buffer, so deallocating the "slice" freed the caller's page table (and would have freed the RoPE cos/sin cache at full context) | `_slice_view` returns `(tensor, owned)` and callers only free real copies |
| 5 | `RuntimeError: bad optional access` in `nlp_concat_heads_decode` at batch 32 | `ttnn.num_cores_to_corerangeset` fills row-major and produces a non-rectangular set (11+11+10 on this grid); the op needs a rectangle | `_decode_core_range_set` picks a rectangle whose width divides the batch (32 -> 8x4) |
| 6 | batch-32 sliding decode PCC 0.9044 (batch 4 fine, batch-32 full-attention fine) | `nlp_create_qkv_heads_decode` shards Q over its *own* non-rectangular grid (`{[0-0 - 10-1], [0-2 - 9-2]}`); the cos/sin were laid out on a tidy 8x4 rectangle, so the sharded `rotary_embedding_hf` paired each user's Q with another user's position | `_rope_decode_mats` derives the cos/sin shard config from Q's own `shard_spec` (grid + orientation) |

Bug 6 is the interesting one: it is invisible at batch 1 and 4, invisible on
`full_attention` (NoPE) layers, and only shows up as a mid-0.9 PCC — exactly the class of
defect the batch-32 + both-kinds test matrix exists to catch.

Two more were found after the Blackhole core-grid sweep (section 6) changed the SDPA
program configs away from their initial values:

| # | symptom | cause | fix |
|---|---|---|---|
| 7 | **device hang** in prefill at seq 2080 (host spinning at 100% in one thread); `tools/tt-triage.py` shows `SDPAOperation` with Q `[1, 32, 2080, 128]` stuck on 32 cores of device 3, previous op `PagedFillCacheDeviceOperation` | `ttnn.transformer.scaled_dot_product_attention` hangs when `q_chunk_size` does not divide the Q sequence length: the sweep-optimal `q_chunk_size=512` against 2080 = 4*512 + 32 | `_sdpa_chunk_for_length` picks the largest candidate that is `<= cap` **and divides** the length (32 always works, since lengths are tile-aligned) |
| 8 | batch-32 decode PCC **0.7176** on both layer kinds after switching the decode SDPA to the sweep-optimal 8x4 grid | `sdpa_decode_program_factory.cpp` only validates `cores >= batch`; with `cores < batch * num_kv_heads` it silently folds both KV heads onto one core (`num_heads_per_core = 2`) and returns wrong results. 8x4 = 32 cores < 32 * 2 | `_decode_sdpa_program_config(batch)` keeps the measured grid only while `cores >= batch * num_kv_heads`, else uses the full compute grid, and raises if even that is too small |

Triage evidence for bug 7 is preserved in
[`triage/tt-triage.txt`](triage/tt-triage.txt) and
[`triage/triage-summary.txt`](triage/triage-summary.txt). The
`check_binary_integrity` / "core was broken during triage" entries in that report are
artifacts of triage halting cores on an already-hung device, not independent findings.
After capturing triage the run was killed and the bounded
list/reset/list + mesh-smoke sequence from section 4 was repeated (one reset was enough).

Both bugs are silent-wrong-answer or hang behaviour in TTNN ops that pass their own
validation, so both are worth upstream reports; both are worked around here in a way that
keeps the measured fast configuration wherever it is legal.

## 4. Long-context run: a false stall, and a device recovery

The first attempt at the 131072-token test was launched with `-s` and its stdout redirected
to a file. `-s` disables pytest's capture, so pytest's own progress lines go through
block-buffered Python stdout while the tt-metal C++ logger writes straight to the file —
the log therefore showed device-open messages and *no* test progress for half an hour. That
looked like a stall, so the run was killed. It was not a stall: measured afterwards, each
reference query block costs ~5 s and the whole test is ~20 min.

Killing pytest mid-run left the devices undiscoverable (`tt-smi -ls --local` aborted inside
`TopologyDiscovery::init_device`, `tt_device.cpp:144`) and one orphaned
`/dev/shm/tt_device_*_memory` segment. Recovery followed `$tt-device-usage`:

| step | command | result |
|---|---|---|
| confirm no live owner | `ps aux \| grep -E "[p]ytest\|[t]racy"` | none (the killed pytest was gone) |
| list | `timeout 120 tt-smi -ls --local` | **failed** — `TopologyDiscovery::init_device` abort |
| reset | `timeout 180 tt-smi -r` | exit 0, "Resetting all PCI devices: [0, 1, 2, 3]" |
| list | `timeout 120 tt-smi -ls --local` | all 4 Blackhole `p300c` back (8 table rows) |
| mesh smoke | `ttnn.open_mesh_device(MeshShape(1,1), trace_region_size=0)` | `Arch.BLACKHOLE`, grid `11-10`, `MESH_SMOKE_OK` |

One reset was enough; no second reset, no lock clearing beyond the orphaned shm segment
(which the next tt-metal run reclaims itself), no `tt-triage`, no `$autofix`. This is
recorded as infrastructure recovery, not a model result.

Measurements taken while diagnosing (all on one Blackhole chip / this host):

| what | time |
|---|---|
| TTNN prefill, 131072 tokens, sliding kind | **1.8 s** |
| TTNN prefill, 131072 tokens, full-attention kind | **13.3 s** |
| `ttnn.from_torch` upload of the 131072-token input | 0.5 s |
| `ttnn.slice` of a 4096-token block out of the `[1,1,131072,6656]` output + `to_torch` | 0.00 s + 0.03 s |
| host reference, one 4096-token query block (sliding) | ~5.0 s |
| of which streaming-PCC accumulation | 0.2 s |
| host reference at 32768 tokens, with vs without a TTNN device open | 40.9 s vs 40.9 s |

So the device side of the full-context test is seconds; the host reference is the cost. The
long-context test is now marked `@pytest.mark.timeout(0)` (pytest.ini caps tests at 300 s)
and is run **without** `-s` so progress is visible.

## 5. Commands run

Fast correctness suite (host reference + device, both layer kinds):

```bash
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
  -q -k "not long_context"
```

Long-context suite (full 131072 context, ~18 min: the host reference is the cost):

```bash
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
  -q -k "long_context" -s
```

Performance artifacts (one Tracy session per measured window):

```bash
models/autoports/meta_models_muse_glimmer_30b/scripts/collect_perf.sh prefill sliding_rope 8192
models/autoports/meta_models_muse_glimmer_30b/scripts/collect_perf.sh prefill full_nope   8192
models/autoports/meta_models_muse_glimmer_30b/scripts/collect_perf.sh decode  sliding_rope 1
models/autoports/meta_models_muse_glimmer_30b/scripts/collect_perf.sh decode  full_nope    1
models/autoports/meta_models_muse_glimmer_30b/scripts/collect_perf.sh decode  sliding_rope 32
models/autoports/meta_models_muse_glimmer_30b/scripts/collect_perf.sh decode  full_nope    32
```

Watcher run (separate from any profiler run, per `$tt-device-usage`):

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/watcher \
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
  -q -k "(paged_prefill_decode_pcc and (8256 or 100)) or traced_decode or (batched_prefill_and_decode and 32) or ragged or batched_multichunk or short_prefill or continued_prefill or page_block_sizes or real_weights"
```

That selects 26 of the 103 collected tests; `watcher/WATCHER_SUMMARY.md` records the result and
the code fingerprint it certifies.

Blackhole core-grid sweep:

```bash
python models/autoports/meta_models_muse_glimmer_30b/scripts/sweep_core_grids.py
```

## 6. Blackhole core-grid sweep

This is Blackhole (11 x 10 compute grid), so the 8x8 / 8x4 program configs that Wormhole
model code uses are only *one* legal option here. `scripts/sweep_core_grids.py` measures
every legal grid x chunk-size combination for the three SDPA call sites at the layer's real
shapes; results in [`perf/core_grid_sweep.md`](perf/core_grid_sweep.md) /
[`perf/core_grid_sweep.csv`](perf/core_grid_sweep.csv): 232 rows, of which 24 are
`RuntimeError` (chunk sizes that exceed L1) and 24 are
`illegal_cores_lt_batch_times_kv_heads` (grids that would silently mis-compute at batch 32,
recorded rather than timed).

What it changed:

**The first sweep was invalid and was re-run.** It called each SDPA op without a
`compute_kernel_config`, i.e. at the op default, while the layer passes HiFi4 +
`fp32_dest_acc_en` — which roughly triples SDPA math time and changes the ranking (the
first sweep's per-call SDPA numbers also disagreed with the profiled `tt-perf-report` rows
by ~3x, which is what exposed the mistake). The committed sweep measures every row under
the layer's own compute-kernel config, so the geometry choice and the precision policy are
the same in the evidence and in the model.

| call site | before | after (measured under HiFi4 + fp32 acc) | effect |
|---|---|---|---|
| sliding prefill SDPA | 11x10, q=128, k=128 | 11x10, q<=256, k<=256 (largest divisor of the length) | 12.02 ms -> 7.46 ms per 8192-token call |
| chunked prefill SDPA | 11x10, q<=128 | 11x10, q<=256, k<=256 | 12.16 ms -> 7.55 ms per 8192-token call |
| decode SDPA | 11x10, k=64 | 8x4 while `cores >= batch * kv_heads`, k=64 everywhere, with the page table sized to a whole number of K chunks (section 9) | batch 1: 52.29 us -> 43.30 us (sliding), 67.32 us -> 50.90 us (full). k=64 is the batch-**32** optimum for both kinds; at batch 1 it costs a few us against 8x4/k=256 |

`q_chunk=512` is *slower* than 256 under this policy (10.31-10.34 ms) and 128 is slower
still, so 256 is the measured optimum rather than "as large as possible". Bigger is not even
always legal: `q_chunk=1024` (swept at `k_chunk=128`) is rejected on **both** prefill call sites
and `q_chunk=512, k_chunk=256` is rejected on the sliding one, with
`TT_THROW: Statically allocated circular buffers on core range [0-0 - 10-9] grow to 2294528 B
which is beyond max L1 size of 1572864 B` — recorded as `RuntimeError` rows in the sweep CSV.
(`q_chunk=512, k_chunk=256` does run on the chunked call site, at 8.51 ms: legal, still slower
than 256/256's 7.55 ms.)
The k-chunk axis was extended after a review noted it was under-swept: `k_chunk=256` with
`q_chunk=256` is a further ~12% (8.44-8.59 ms -> 7.46-7.55 ms) and is the shipped default. The full 11x10
grid and the 11x8 sub-grid are within ~1% of each other for prefill and both beat the
Wormhole-shaped 8x8 by ~30%; the full grid is kept because it is derived from
`compute_with_storage_grid_size()` rather than hard-coded. For decode the *smaller* 8x4
grid wins at batch 1 — with only 2 KV heads a wider grid deepens the flash-decode reduction
tree without adding useful parallelism — but it is only legal while every (user, KV head)
pair still gets its own core (bug 8).

Every change here was re-validated by rerunning the whole correctness suite and the
full-context suite; see the PCC artifact for the final numbers.

## 7. Performance collection notes

`collect_perf.sh` runs one Tracy session per (mode, kind, size) because `tt-perf-report`
keys off the *first* start/end signpost pair in a CSV. Two profiler-tooling problems came
up and are handled in the script/tests rather than papered over:

1. `test_decode_perf[32-*]` fills 32 users' caches with 32 sequential prefills before the
   measured window. That overflows the per-core profiler buffers, and post-processing dies
   with `Unexpected FW start ... Other cores might have their profiler buffer filled up ...
   run read device profiler more often`. Fixed by draining with
   `ttnn.ReadDeviceProfiler(mesh)` between setup and the measured window (`_drain_profiler`,
   a no-op when not profiling) — the measured ops are untouched.
2. Tracy's host-side post-processing can also fail with `Device data missing: Op N not
   present in cpp_device_perf_report.csv`. `collect_perf.sh` falls back to the
   `TT_METAL_DEVICE_PROFILER=1` + `tools/tracy/process_ops_logs.py --date` flow from
   `tech_reports/LLMs/llms.md`, same measured window and signposts.

All eight artifact sets were then re-collected with the final script, so every committed
report comes from the committed tooling. A separate **unprofiled** pass of the same tests
records plain wall clock; the two agree within a few percent, so the profiled numbers are
not distorted.

## 8. Stage-review round 1 and the fixes it forced

An independent `$stage-review` subagent returned `more-work-needed` with four P2 findings.
All four were real; each is fixed with a regression test or a documentation correction, and
every affected artifact was re-collected.

| # | finding | fix | how it is now pinned |
|---|---|---|---|
| R1 | the **full-attention** prefill SDPA ran at `q_chunk_size=512` — the size the sweep measured as *slower* — because `_chunked_sdpa_chunk_sizes` took `max(candidates)` with no cap, so the `prefill_sdpa_q_chunk` constructor argument was dead on that path and the recorded geometry did not match the executed geometry | cap both candidates with `self.prefill_sdpa_q_chunk` / `self.prefill_sdpa_k_chunk` | `test_chunked_sdpa_chunk_sizes_respect_the_configured_cap`; the re-collected prefill ops CSVs for **both** kinds show the configured geometry (`k_chunk_size` was 128 when this was fixed; section 9 later moved it to 256 after extending the sweep, and that is what the committed artifacts show today) |
| R2 | `start_pos > 0` (continued prefill) was documented and validated but silently wrong on sliding layers: the window prefix comes from the input tensor, so a continued segment's first 2048 positions attend to a truncated window | sliding layers raise with an explanation; full-attention layers keep the capability (they read the whole prefix from the paged cache) | `test_continued_prefill_contract`: asserts the raise for sliding, and PCC of a `[0,1024)` + `[1024,1536)` continued prefill against a single-shot `[0,1536)` reference for full-attention |
| R3 | 132 of 144 PCC records had no provenance, so a record from an older revision was indistinguishable from a fresh one, and no console log was kept for the definitive suites | every record now carries `code_sha256` (hash of implementation + reference + tests), `git_head` and `recorded_at`; the artifact records the current fingerprint and `render_evidence.py` names any stale record | `pcc/pcc_results.json` was deleted and rebuilt from scratch: **165 records, 0 stale**. Console logs kept in [`logs/`](logs) |
| R4 | the Environment section claimed "no reset, no hang, no recovery ... `tt-triage` was never required" while the same file documented a hang with captured triage and a device recovery | Environment section now points at both incidents | see the top of this file |

Reviewer "Other Concerns" also addressed: the decode core-grid sweep now measures **batch 32**
as well as batch 1 (and marks grids with `cores < batch * kv_heads` illegal rather than
timing them); `decode_sdpa_k_chunk` collapsed to the single measured-best value 64;
`PROFILED_DECODE_ITERS = 12` so every committed capture is complete (op counts are exact
multiples of the iteration count — previously three of four decode reports silently dropped
4-25% of rows); `num_to_corerange` hoisted to module import so no host import happens inside
the decode path; the `sigmoid` temporaries are deallocated; README wording corrected on
"every legal grid", on device-time-vs-wall-clock for prefill, and on the batch-1-only op
percentages; `WATCHER_SUMMARY.md` corrected from "85 summaries" to 17 blocks / 85 per-RISC
lines. New coverage: `test_short_prefill_lengths` (1/7/31 tokens),
`test_batched_multichunk_prefill_shared_pool` (batch 2 x 8256 tokens, `block_size=128`,
non-identity slots out of a 4-row pool — the first test to exercise `_rows_page_table`'s
multi-row gather, `first_block > 0` with batch > 1, and the sliding overlap trim with
batch > 1).

### Third device incident, and a script bug it exposed

While re-collecting the prefill artifacts, `test_prefill_perf[4096-sliding_rope]` died with
`RuntimeError: Read 0xffffffff over PCIe ID 1: the board should be reset.` (preceded by
`TENSTORRENT_IOCTL_SET_POWER_STATE failed on device 1`) — an ARC/PCIe fault that
`$tt-device-usage` classifies as recoverable. `tt-smi -ls --local` still listed all four
boards, `tt-smi -r` succeeded, the list and the mesh smoke were clean afterwards, and the
suite ran normally from then on.

The failure also exposed a real bug in `collect_perf.sh`: it took `ls -t ... | head -1` as
"this run's ops CSV", so when the pytest run produced no CSV it silently published the
*previous* configuration's data. Three prefill artifacts were briefly wrong because of it
(their `ATTRIBUTES` showed decode-shaped program configs). The script now records the newest
CSV before the run and fails with exit 4 if no newer one appears, and fails with exit 5 if
the selected test did not pass. All eight artifact sets were then collected again with the
fixed script; the committed `*_ops.csv.provenance` files and the `ATTRIBUTES` columns were
re-checked op by op:

* prefill (both kinds, 4096 and 8192): `q_chunk_size=256;k_chunk_size=256`;
* decode SDPA: `k_chunk_size=64`, `num_kv_heads_override=2`, HiFi4 + `fp32_dest_acc_en`,
  **32 cores at batch 1** (the measured 8x4 grid) and **110 cores at batch 32** (the bug-8
  fallback), exactly as documented.

## 9. Stage-review round 2, and the `$autofix` it triggered

The re-review confirmed the four round-1 findings were fixed but raised five more, all real:
a **stale watcher run** (it had been collected before the round-1 code changes, and covered 60
of the now-73 collected tests), README still documenting `prefill_sdpa_q_chunk=512`, the new
`start_pos` restriction undisclosed in the README and the context contract, README perf
percentages that disagreed with the committed reports, and an Environment section that still
contradicted the rest of this file. Fixes: watcher re-run on the shipped code with the new
tests included (24 tests, and `WATCHER_SUMMARY.md` now records the code fingerprint it
certifies); README default corrected; `continued_prefill` recorded in
`doc/context_contract.json` and in the README contract and Limitations; the op-family
percentages are now **generated** into `evidence_tables.md` from the committed CSVs instead of
hand-written; Environment corrected. Secondary items: `render_evidence.py` now recomputes the
code fingerprint itself (and gates the *executed* SDPA geometry against the module constants,
exiting non-zero on drift) instead of trusting the artifact; `_code_fingerprint` also covers
`conftest.py` and the perf tests; work-log §6's decode row corrected; the README test matrix
completed; `also_tested_seq_lens` extended; the k-chunk sweep axis extended (which found
`k_chunk=256` a further ~12% faster, now the default, and recorded `q_chunk=1024` /
`q_chunk=512,k_chunk=256` as L1-illegal); a decode-reference control against HF's own
`DynamicCache` added (`test_reference_decode_matches_hf`).

### The bug the new batched multi-chunk test found

The new `test_batched_multichunk_prefill_shared_pool[full_nope]` passed the 0.995 bar but at
**0.9971/0.9979**, an order of magnitude worse than every other prefill measurement. Rather
than accept it, I split it by position and found the tail chunk `[8192, 8256)` at PCC
**0.7345** while the head `[0, 8192)` was 0.9998, and reproduced the divergence purely
TTNN-vs-TTNN (batch 2 vs batch 1, same input: max abs difference 5.09, 56% relative). I ruled
out cross-user contamination, the K/V cache contents, the page-table gather, and every
individual op in isolation, then handed it to `$autofix` with that evidence.

`$autofix` verified the root cause: **an unvalidated page-table overrun in
`ttnn.transformer.chunked_scaled_dot_product_attention`.** The op checks only
`kv_length >= q_len + chunk_start_idx` (`sdpa_device_operation.cpp`), but its program factory
rounds the K extent up to the K chunk size (`padded_Sk = ceil(Sk / k_chunk) * k_chunk`) and the
reader consumes one page-table entry per `block_size` of that rounded extent with no bound
check (`sdpa/device/kernels/dataflow/dataflow_common.hpp`: `page_table_ptr[virtual_block]`).
With `Sk = 8256`, `k_chunk = 256` and a 65-block page table (8320 tokens) it reads entry 65 of a row that
has 65 valid entries (0..64) — inside the page-table stick's 32-byte alignment padding — and
uses whatever integer is there as a physical block id, reading K/V from outside the cache buffer.

Two orthogonal controls fixed it, which is what pins the mechanism: shrinking `padded_Sk`
(k_chunk 128/64/32 -> tail PCC 0.99977) or growing the page table to cover `padded_Sk`
(k_chunk still 256 -> 0.99978). `q_chunk` was irrelevant (bit-identical results). Batch was a
*correlate, not the cause*: batch 1 overran identically but got away with it because its page
table came from `ttnn.from_torch`, which writes the whole aligned page and so leaves the
padding as zeros -> block id 0 -> back inside the cache buffer. The batched path builds its
page table with `ttnn.slice` + `ttnn.concat`, whose padding holds stale DRAM.

The fix is one predicate in `_chunked_sdpa_chunk_sizes` — the k-chunk candidate must satisfy
`_round_up(chunk_start_idx + chunk_len, c) <= kv_length`, i.e. encode the op's *actual*
constraint rather than its documented one:

| metric | before | after |
|---|---|---|
| `full_nope` batched multi-chunk, row 0 / row 1 | 0.997942 / 0.997938 | **0.999808 / 0.999809** |
| `full_nope` tail `[8192, 8256)` only | 0.734472 / 0.733595 | 0.999772 / 0.999786 |

A logic sweep over 117 geometries (seq 128…131136 x block_size 32/64/128) shows the bug was
**wider than the symptom**: 14 configurations were overrunning, including `seq=3000` at block
size 32/64 on the *whole* prefill, and the 16448/32832/131136 tail chunks. Every
non-overrunning configuration keeps its previous chunk sizes.
`test_chunked_sdpa_k_chunk_stays_inside_the_page_table` walks the real chunking for a range of
lengths and block sizes and asserts the selected `k_chunk` never overruns.

The same unvalidated rounding exists on the **decode** call site — `rt_args_common.hpp` rounds
the K extent to `nearest_n(cur_pos + 1, k_chunk_size)` and `reader_decode_all.cpp` walks it
through the same unbounded `page_table_ptr[virtual_block]`, with `cur_pos` a device tensor that
nothing host-side bounds. A stage review caught this by reading the source: with a 25-block
page table at `block_size=32` (capacity 800) and `k_chunk=64`, decoding at position 777 rounds
to 832 and reads block-id entry 25 of a 25-entry row — a configuration `test_page_block_sizes`
was already producing, masked only by `ttnn.from_torch`'s zeroed padding.

The first attempt at a fix was to *shrink* the decode K chunk so it divides the capacity. That
is wrong on device: `k_chunk=32` for that geometry measured decode PCC **0.6558**, so the small
K chunk is itself broken here. The shipped fix goes the other way and keeps the K chunk at the
measured, correctness-verified 64: `blocks_per_seq` rounds the page-table capacity up to a whole
number of `lcm(block_size, k_chunk)` tokens (25 blocks -> 26 at `block_size=32`), and
`_check_decode_page_table_capacity` refuses a capacity that is not a whole number of K chunks,
so the rounding is safe for every position the table can address.
`test_decode_page_table_capacity_covers_the_k_chunk_rounding` pins it. Evidence for the
localization of the prefill-side bug is in [`logs/autofix/`](logs/autofix).

Upstream: `scripts/repro_chunked_sdpa_page_table_overrun.py` is a standalone reproducer with no
model weights. Its out-of-bounds calls are **opt-in** (`--include-hang-case`) because the
overrun does not always merely corrupt: on one set of padding contents it produced PCC -0.0018
with `max|out|` 2.4e37, and on another it **hung the device** — and left it in a state where
the next `open_mesh_device` hung too, while `tt-smi -ls --local` still listed all four boards
(incident 4). The in-range case it runs by default reproduces the fix at PCC 0.999776.

### Device incidents 4 and 5, in the form `$tt-device-usage` asks for

**Incident 4 — the out-of-bounds reproducer hung the device, twice, and the hang outlived the
process.**

| field | value |
|---|---|
| failure signature | no output and no progress for >900 s, then `Terminated`; on the *next* run, `open_mesh_device` itself hung after "Real-time profiler Device 3 sync complete" |
| exact command | `python models/autoports/meta_models_muse_glimmer_30b/scripts/repro_chunked_sdpa_page_table_overrun.py` (then the same command again, and once more with `PYTHONUNBUFFERED=1` to rule out buffered output) |
| processes killed | the reproducer process only (`timeout`/SIGTERM); no other job was running |
| `tt-smi` list / reset / list | list: **all four boards listed, exit 0** — i.e. listing did *not* reveal the problem; `timeout 180 tt-smi -r` exit 0 ("Resetting all PCI devices: [0, 1, 2, 3]"); list again: all four boards |
| second reset needed | no |
| locks cleared | none; the orphaned `/dev/shm/tt_device_*` segment is reclaimed by the next tt-metal run, which logs `ShmResourceTracker: removed orphaned shm` |
| mesh smoke | `open_mesh_device(MeshShape(1,1), trace_region_size=0)` -> `Arch.BLACKHOLE`, grid `11-10`, `MESH_SMOKE_OK` |
| `tt-triage` | **not captured.** The hang was deliberately provoked by a script whose whole purpose is to read out of bounds, and the mechanism was already proven from the op source plus the two orthogonal controls in section 9, so triage would have added nothing to the diagnosis. Triage *was* captured for the one hang whose cause was unknown (incident 2, `triage/tt-triage.txt`). |
| `$autofix` | not needed for the incident itself; `$autofix` had already produced the fix whose reproducer this is |
| resumed | the reproducer's in-range case, which then ran clean (`logs/autofix/repro_in_range.log`); the out-of-bounds calls are now opt-in behind `--include-hang-case` |

**Incident 5 — two suite runs collided.**

| field | value |
|---|---|
| failure signature | `46 errors` at fixture setup (`mg_mesh_device`) with `40 passed`, i.e. every device test errored while every host-only test passed |
| cause | my own chained command `fast; long` kept going after the fast suite failed on a stale test expectation, so its long-context run was still holding the device when I launched the next fast suite — operator error, not a hardware fault |
| exact command | the second `python -m pytest ... -k "not long_context"` while `... -k "long_context"` from the previous chain was still running |
| processes killed | the surviving `pytest ... -k long_context` (PID 1144220), confirmed gone with `ps` |
| `tt-smi` list / reset / list | `timeout 180 tt-smi -r` exit 0, then list: all four boards |
| second reset needed | no |
| locks cleared | none |
| mesh smoke | `Arch.BLACKHOLE`, grid `11-10`, `MESH_SMOKE_OK` |
| `tt-triage` | not applicable (no hang; a fixture-level open failure) |
| resumed | both suites re-run serially, one at a time; the failed run is superseded and its records discarded (the PCC artifact is deleted before every re-run). The fast suite was 86 tests at the time of this incident; the committed `logs/fast_suite.log` is the final rerun, **101 passed** of 103 collected, and `logs/long_context_suite.log` is **2 passed**. |

## 10. Final state

* Correctness suite: **101 passed** (`-k "not long_context"`, log in `logs/fast_suite.log`),
  full-context suite: **2 passed** (`logs/long_context_suite.log`). 103 tests collected.
* PCC: **165 recorded measurements, 0 stale** (every record stamped with the current code
  fingerprint `9997b1d8381fdb9a`, which is also the fingerprint the watcher summary certifies),
  global minimum **0.999723** against a 0.995 bar, including the full-context (131072) and
  real-weight runs. `scripts/render_evidence.py` exits 0, i.e. no stale record and no drift
  between the executed SDPA geometry and the module defaults.
* Watcher: clean on the shipped code — the summary records the same code fingerprint as the
  PCC artifact; see `watcher/WATCHER_SUMMARY.md`.
* Perf: 8 `tt-perf-report` artifact sets, all freshness- and attribute-verified, plus
  profiled and unprofiled wall clock (`logs/perf_unprofiled.log`).
* Device incidents: 5, all recoverable, all recorded (sections 4, 8 and 9).
* `$autofix` was used once, for the bug in section 9; the earlier failures were explained
  directly by op-validation source, triage output, or a script bug and fixed at the cause.

### One classified log oddity

Every committed pytest log ends with `nanobind: leaked N instances / types / functions` at
interpreter shutdown, after the pytest summary line. It appears identically in every run
regardless of which tests ran and every run still exits 0 — it is a ttnn Python-binding
teardown property of this build, not a stage result, and the watcher log shows no device-side
counterpart.

## 11. Checkpoint commit

Stage-owned changes are committed locally on `agentic-research/hous/multigoal-claude`; nothing
was pushed.

| repo | branch | commit | contents |
|---|---|---|---|
| tt-metal | `agentic-research/hous/multigoal-claude` | `c24bb9de468f` | everything under `models/autoports/meta_models_muse_glimmer_30b/` — implementation, host reference, tests, scripts and evidence |
| tt-metal | same | `6c3a6236d44` | this section |
| tt-metal | same | `6b9aa772d56` | the round-3/4 corrections: the decode page-table capacity guard, the `layer_rope_theta` assertion, the HF-snapshot fix, the tightened multi-chunk bar, the evidence-gate additions, and the re-run evidence |

The commit contains only this stage's files; the one other dirty path in the worktree
(`tt_metal/third_party/tt-cluster-descriptors/`, an untracked submodule checkout) was left
alone. `doc/.gitignore` re-includes `*.csv`, `*.log` and `generated/` for this subtree, which
the repo root `.gitignore` excludes, because those are the stage's evidence; the two artifacts
that exceed the repo's 500 KB pre-commit limit (the raw Tracy ops CSVs and the watcher log) are
committed gzipped, and `scripts/render_evidence.py` reads either form.

All 165 PCC records carry code fingerprint `9997b1d8381fdb9a` and `git_head` `6c3a6236d44`.
The **fingerprint** is what proves the evidence was produced by exactly the committed code: it
hashes the implementation, host reference and test files, and the runs happened after the
pre-commit formatting hooks had settled. The recorded `git_head` is necessarily the *previous*
HEAD, because the runs precede the commit that contains them.

See `README.md` for the results tables and the exact artifact paths.
