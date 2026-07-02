# RESTORE CONTEXT — per-element-tensor trace-safe metadata (Kimi/DeepSeek prefill)

**Last updated:** 2026-07-01, mid-Phase-4. Read this top-to-bottom to resume.

## 0. How to restore / resume (start here)

1. `cd /home/ppopovic/tt-metal`
2. The deliverable branch is **`ppopovic/per_element_metadata`** (HEAD `2609e892780`). It contains the
   finished per-element-tensor metadata work + the two correct bug fixes + Phase-3 results. It is
   **pushed** at commit `7fb1f6f9111` (the code fix); two later commits are docs-only and were not yet
   pushed (remote network was flaky — see §7).
3. `git checkout ppopovic/per_element_metadata` and pick up at **Phase 4** (§4). If a
   `ppopovic/trace_experiments_rebased` branch exists mid-merge, it can be discarded and redone (§4).
4. Original overnight plan (full text) lives in session transcript `ea247624-df70-4989-a0da-13dca0d9e850`
   under `/data/ppopovic/.claude/projects/-home-ppopovic-tt-metal/`; the running log is
   `models/demos/deepseek_v3_d_p/NIGHT2_REPORT.md`; Phase-3 numeric results are in
   `PER_ELEMENT_TENSOR_RESULTS.log` (gitignored — content mirrored in NIGHT2_REPORT.md).

## 1. Branch topology (IMPORTANT — do not re-pollute trace_experiments)

- **`ppopovic/trace_experiments`** — local+remote reset to commit 1 `326056b7ac8` (the pre-existing
  squashed *packed*-metadata trace work). The user does NOT want the per-element work committed here.
  Leave it alone.
- **`ppopovic/per_element_metadata`** — THE deliverable. Commits on top of `326056b7ac8`:
  - `7fb1f6f9111` `#0: Per-element-tensor trace-safe metadata for chunked prefill` (code + both fixes) — **pushed**
  - `f6e9c4f4e47` `#0: Phase 3 validation — trace KV-PCC L10/L61 + results log` (docs) — not pushed
  - `2609e892780` `#0: Phase 3 complete — runner standalone + request-loop` (docs) — not pushed
- **`ppopovic/trace_experiments_rebased`** — Phase-4 WIP (squash-merge onto main, conflicts partly
  resolved). Safe to `git checkout ppopovic/per_element_metadata && git branch -D
  ppopovic/trace_experiments_rebased` and redo Phase 4 from scratch per §4 (resolution decisions recorded there).

## 2. What the work IS

Convert the trace-safe chunked-prefill metadata from ONE packed DRAM tensor `[slot_id, actual_start,
actual_end]` to **one 1-element uint32 replicated-DRAM tensor per scalar** (`slot_id`, `kv_actual_isl`),
host-updated in place per chunk so a single captured ttnn trace replays across 11 chunks. Four ops were
converted (`update_padded_kv_cache`, `rotary_embedding_indexed`, `zero_padded_kv_cache`, `ring_mla`) plus
the mla.py / pipeline / transformer / runner plumbing and the op-equivalence tests. Each op keeps its
scalar (int) overload and gains a per-element-tensor overload.

## 3. The two bug fixes (CORRECT, NO hacks) — found during ring_mla bring-up

Both are on `ppopovic/per_element_metadata` @ `7fb1f6f9111`.

### Bug A — all-gather reader read slot_id=0 on ~10% of cores (indexed[slot1])
- File: `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp`
- ROOT CAUSE: the reader read the metadata scalars into the tiny dedicated `c_in3` CB (`cb_meta_id`) via
  `get_write_ptr()`. That 32-byte CB's L1 region is **transiently clobbered by concurrent fabric /
  packet-header activity** during the op-start read window → intermittent drop-to-zero (~6–15% of cores;
  clean-run diffs varied 1584/720/1584). DRAM stably held the true value (host readback confirmed).
- FIX (1-line, single read, NO retry): read the scalars into the **output CB (`cb_output_id` / c_in0)**
  L1 as scratch — a large real data CB not touched until the gather loop reserves+overwrites it — exactly
  as the proven SDPA `ring_joint_reader` does with `cb_q_in`.
- ❌ REJECTED earlier hack (do not reintroduce): a `for(r=0;r<8;r++)` re-read-and-take-max loop. The user
  correctly called this out as masking the bug. It is GONE.

### Bug B — ring_mla rotation (kv64/256/320) deterministic wrong output
- File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp` +
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
- ROOT CAUSE: the SDPA reader read `kv_actual_isl` into a **nonzero CB offset** (`kKvDstOffset = 16`).
  On this platform, reading a small metadata scalar into any nonzero CB offset **deterministically lands
  0**; only the CB page base (offset 0) works. → `logical_nt` / q-mapping derived from `kv_actual_isl=0`
  → wrong rotation output (byte-identical diffs 20.375/24.625/24.75 across runs).
- FIX: read kv into **offset 0** (slot is read + consumed into `kv_cache_batch_idx` before the kv read
  reuses offset 0), + gave the reader its **own kv accessor** (mirrors the writer; the factory appends
  `TensorAccessorArgs(kv_actual_isl->buffer())` and the kernel builds `kv_meta_args`). Offset-0 was the
  operative fix; the separate accessor is correct-and-tested.

## 4. Phase 4 (rebase onto main) — WHERE I STOPPED

Goal branch: `ppopovic/trace_experiments_rebased`. Approach chosen: **single squash-merge** onto main
(one clean commit, resolve conflicts once) rather than per-commit rebase (which would resolve the same
files twice: packed commit then per-element commit).

Facts:
- `main` = `origin/main` = `5a47eae1e13` (local, current; NO fetch needed — network flaky).
- merge-base(per_element_metadata, main) = `51bbd92b5f5`. main is **82 commits ahead** of it.
- Commit to bring over = the per-element NET (326056b7ac8 packed → 7fb1f6f9111 per-element → docs).

Redo command:
```
git checkout -b ppopovic/trace_experiments_rebased main
git merge --squash ppopovic/per_element_metadata     # exit 1, surfaces conflicts
```
**8 conflicts** surfaced (the original plan expected the C++ op files to be conflict-free; they are NOT —
main refactored ring_joint_sdpa since the plan was written):
- Python glue: `models/demos/deepseek_v3_d_p/tt/moe/tt_moe.py`, `.../runners/prefill_runner.py`,
  `.../runners/runner_utils.py`, `.../tt_prefill_block.py`, `.../tt_prefill_runtime.py`,
  `.../tt_prefill_transformer.py`
- C++: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp`,
  `.../ring_joint_sdpa_device_operation_types.hpp`

### Conflict resolutions — decisions already made (redo these):

1. **`ring_joint_sdpa_device_operation_types.hpp` — RESOLVED.** Main refactored reflection from an
   explicit `attributes()` method to `static constexpr attribute_names` + `attribute_values()`. My branch
   had the old `attributes()` method. RESOLUTION: DELETE my `attributes()` method (take main's empty HEAD
   side); main's `attribute_names`/`attribute_values` auto-merged and is kept. My new struct fields
   `kv_cache_num_layers` / `kv_cache_layer_idx` auto-merged into the ctor fine.
   → BUT see item 2: they must ALSO be added to main's `attribute_names`/`attribute_values`.

2. **`ring_joint_sdpa_device_operation.cpp` — DECIDED, NOT YET APPLIED.** Main REMOVED the explicit
   `compute_program_hash` method entirely and relies on **auto-hashing** (from `attribute_values` + the
   op's input tensors; op is new-style: `operation_attributes_t`/`tensor_args_t`, no `.hpp`
   `compute_program_hash` decl). My branch had an explicit `compute_program_hash` that additionally hashed
   `tensor_args.has_metadata()`, `args.kv_cache_num_layers`, `args.kv_cache_layer_idx` — REQUIRED so each
   layer's baked program is keyed distinctly (else deep-layer KV collision — this is a real, subtle
   correctness bug; only the L61 traced run catches it, NOT the single-invocation op tests).
   **CORRECT RESOLUTION:** take main's empty HEAD side (delete my explicit `compute_program_hash`), and
   instead ADD the two layer fields to main's reflection in types.hpp:
   - add `"kv_cache_num_layers", "kv_cache_layer_idx"` to `attribute_names`
   - add `std::cref(kv_cache_num_layers), std::cref(kv_cache_layer_idx)` to `attribute_values()` (as
     VALUES, not `has_value()` bools — they are the per-layer keying values).
   The metadata-vs-scalar distinction: verify `slot_id` / `kv_actual_isl_tensor` are in the op's
   auto-hashed **input tensor list** (they are fields of `RingJointSDPAInputs` — `slot_id` at
   types.hpp:145, `has_metadata()` at :148). If main's auto-hash includes optional input tensors, presence
   distinguishes metadata vs scalar automatically. **VERIFY THIS before trusting the hash** — if the
   metadata tensors are NOT in the hashed input list, the metadata program could collide with the scalar
   program. (Note main's comment at .cpp:167 "validate_runtime_patched_scalars ... NOT part of
   compute_program_hash: kv_cache_batch_idx and logical_n / kv_actual_isl" — those are runtime-patched, so
   fine to exclude.)
   ⚠️ This is the single most correctness-critical merge decision. Getting the program hash wrong is a
   silent deep-layer bug. Confirm with the L61 traced KV-PCC run after building (§6).

3. **6 Python glue conflicts — NOT YET EXAMINED.** main evolved tt_moe / prefill_runner / runner_utils /
   tt_prefill_block / tt_prefill_runtime / tt_prefill_transformer over 82 commits; my per-element changes
   thread the metadata tensors through them. Resolve as the UNION (main's evolution + my metadata-tensor
   threading). Examine each with `git diff --diff-filter=U` and the `<<<<<<<`/`=======`/`>>>>>>>` markers.

### After resolving all conflicts:
```
git add -A && git commit    # one squashed commit: per-element trace-safe metadata (rebased on main)
# FULL rebuild against main (82 commits ahead — likely a large/long build, NOT incremental):
cmake --build build_Release --target ttnncpp     # (+ --target ttnn if nanobind sigs changed)
cp build_Release/ttnn/_ttnncpp.so ttnn/ttnn/_ttnncpp.so   # <-- THE loaded copy (see §7 gotcha)
```
Then re-run the §6 validation suite. **The L61 traced KV-PCC run is the gate that confirms the program-hash
resolution (item 2).**

## 5. Phases 5 & 6 (NOT started)

- **Phase 5** — port the branch's trace machinery (`SubDeviceTraceController` + `capture_trace()`/replay)
  onto main's multi-host `TtPrefillRuntime` (per-rank; hidden state ships D2D over fabric). Keep the
  per-chunk D2D recv/send + `wait/release_fabric_links()` + `ttnn.distributed_context_barrier()` OUTSIDE
  the trace replay (they are host-side socket ops). Gate behind `PREFILL_USE_TRACE` (default off).
  ⚠️ **UNTESTABLE on this box** — the multi-host `tt-run` MPI pipeline is not available here. Implement +
  reverify only the single-host-testable tests; write `PER_ELEMENT_TENSOR_RESULTS_rebased.log`.
- **Phase 6** — off main, split into per-op cleanly-mergeable branches, each carrying only that op's
  per-element change + its test: `ppopovic/metadata_tensor_update_padded_kv_cache`,
  `..._rotary_embedding_indexed`, `..._zero_padded_kv_cache`, `..._ring_mla`, and
  `ppopovic/traced_prefill_glue` (mla/transformer/block/pipeline/runner/sub_device_trace + Phase-5
  multi-host). Build each in isolation, run its op test green, dry-run `git merge --no-commit --no-ff`
  clean into a fresh main.

## 6. Env + validation commands (all PASS on per_element_metadata @ 7fb1f6f9111)

```bash
cd /home/ppopovic/tt-metal && source python_env/bin/activate
export KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden
# ALWAYS wrap device runs: timeout --signal=KILL <s> ...   (see §7 hang recovery)

# 4-op equivalence (bit-exact):
pytest -q tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_metadata_matches_scalar_indexed \
          tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_metadata_matches_scalar_rotation
pytest -q models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_rotary_embedding_indexed.py::test_rotary_embedding_indexed_metadata_matches_scalar \
          models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py::test_update_padded_kv_cache_metadata_matches_scalar \
          models/demos/deepseek_v3_d_p/tests/test_zero_padded_kv_cache.py::test_zero_padded_kv_cache_tensor_matches_scalar
# transformer 11-chunk metadata trace KV-PCC (L61 is the program-hash gate):
pytest -q "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_trace_kv_pcc[blackhole-kimi-mesh-8x4-L10-chunks11]"
pytest -q "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_trace_kv_pcc[blackhole-kimi-mesh-8x4-L61-chunks11]"
# padded full-55k trace: [...L1-full55k] / [...L10-full55k]
# runner STANDALONE (exports then module; do NOT put 'prefill_runner' pkill in the same shell — see §7):
export PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320
export PREFILL_NUM_LAYERS=10 PREFILL_NUM_USERS=1 PREFILL_MAX_SEQ_LEN=56320
export PREFILL_STANDALONE=1 PREFILL_STANDALONE_NCHUNKS=11 PREFILL_STANDALONE_PCC=1 PREFILL_STANDALONE_ITERS=2 PREFILL_USE_TRACE=1 PYTHONPATH=/home/ppopovic/tt-metal
python_env/bin/python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner
# request-loop (2 proc): runner with PREFILL_REQUEST_LOOP_PCC=1 + DEEPSEEK_PREFILL_TRACE_DIR=<vllm subdir>,
#   wait for "entering request loop", then run prefill_h2d_producer. Full cmd in TRACED_PREFILL_HANDOFF.md §6(d).
```

### Phase-3 results (all GREEN, on per_element_metadata):
| Test | Result |
|---|---|
| op equivalence (4 ops) | 15/15 bit-exact |
| transformer trace KV-PCC | L10 0.994096, L61 0.966851 (min-asserted 0.993545) |
| padded full-55k (traced≡untraced) | kimi L1 0.00e+00, L10 4.75e-06 (ds unavailable: weight cache not on box) |
| runner standalone traced+PCC | 0.994096, ~1959 ms/iter steady |
| runner request-loop (producer+socket) | 0.994096 |
| ring_mla microperf | PASS (kv 256/1024/5120) |

## 7. GOTCHAS (these cost real time — heed them)

- **`.so` refresh:** the LOADED lib is `ttnn/ttnn/_ttnncpp.so`, NOT `build_Release/lib/_ttnncpp.so`.
  After `cmake --build ... --target ttnncpp` you MUST `cp build_Release/ttnn/_ttnncpp.so
  ttnn/ttnn/_ttnncpp.so`, else host C++ changes are silently ignored (tell-tale: byte-identical test
  output across genuinely different source). Kernel `.cpp` under `device/kernels/` are JIT-recompiled from
  source — NOT affected by the .so. (Memory: `ttnn-so-refresh-procedure`.)
- **`pkill -f prefill_runner` self-matches** any shell whose command line contains the string
  "prefill_runner" (heredoc/echo) → kills your own shell before later commands run. Kill by PID
  (`ps -eo pid,args | grep "[t]t.runners.prefill_runner"`), not `pkill -f`.
- **DPRINT:** this codebase requires format-style `DPRINT("fmt {}\n", args)`; streaming `DPRINT << ...`
  fails a static_assert ("Old style DPRINT is deprecated"). `TT_METAL_DPRINT_CORES=all` with MORE THAN ONE
  DPRINT statement overflows the buffer and HANGS the run — use at most one, or fewer cores.
- **`sleep` is blocked** in the Bash tool (even backgrounded). To poll/wait, use the Monitor tool (its
  shell can sleep) or a Bash `run_in_background` `until grep ...` (no bare sleep in a foreground Bash call).
- **Device hang recovery:** wrap every device run in `timeout --signal=KILL <s>`; on exit 124/137 →
  `pkill -9 -f pytest` (or by PID) then `tt-smi -glx_reset` (takes ~40s, re-inits 32 boards). Two hangs
  happened during this work (both DPRINT-buffer overflows); reset recovered cleanly each time.
- **git push to origin is currently FLAKY/slow** (pushes hang and get killed; produce no output). The
  CODE fix is pushed (`7fb1f6f9111`); the 2 doc commits are not. Retry later:
  `git push origin ppopovic/per_element_metadata` (and push `trace_experiments_rebased` when Phase 4 done).
- **DeepSeek (ds) tests skip** on this box: `DeepSeek-R1-0528` TTNN weight cache is not populated
  (`~/.cache/huggingface/models--deepseek-ai--DeepSeek-R1-0528/.../tensor_cache_bh_32dev/8x4` incomplete).
  Not a code failure. Kimi is the validatable variant here.

## 8. Current git state to expect on resume
- `ppopovic/per_element_metadata` @ `2609e892780` — clean, all validated.
- `ppopovic/trace_experiments_rebased` — may be mid-squash-merge (types.hpp resolved, .cpp + 6 Python
  conflicts open). Discard + redo per §4 (all decisions recorded there), OR finish resolving.
- untracked (unrelated, leave alone): `models/demos/deepseek_v3_d_p/reference/dflash_draft_prefill.py`,
  `.../reference/mtp_prefill.py` (from a separate dflash task).

## 8. Phase-5 (L10/L61 traced prefill on rebased-onto-main) — ROOT-CAUSED, BLOCKED (2026-07-02)

Status: rebased branch `trace_experiments_rebased` @ `21a3b24f2a9` builds clean, ring_mla equivalence
5/5, **L1 trace KV-PCC green**. **L10 trace capture fails** with `TT_FATAL: Writes are not supported
during trace capture`. All experimental debugging reverted — the branch is back to the clean rebased
commit (no uncommitted changes). Findings (keep — hard-won):

- **NOT padding_config host from_torch.** Memoizing `build_padding_config` per `actual_isl` + a Python
  monkeypatch instrumenting `from_torch/to_torch/copy_*_tensor/to_device` during capture proved **0
  Python host transfers** happen inside the capture window. padding_config is not the culprit.
- **NOT program-cache eviction.** `tt_metal/api/tt-metalium/program_cache.hpp` is a plain unbounded
  `std::unordered_map` — no LRU/eviction. A program compiled in warmup stays cached.
- **Root cause = tensor-address non-determinism across warmup→capture.** On a program-cache *hit*, an op
  re-patches its runtime args (tensor addresses) via WriteBuffer; that WriteBuffer is illegal during
  capture. It is only skipped when every input/output lands at the *same address* as in the warmup that
  compiled the program. The full 10-layer main-MoE forward does not allocate deterministically enough:
  the FIRST fatal op *moves* when you perturb allocations (clean branch: burst of 32 writes + 2 reads
  right after `forward_layer_9_start`; after a memoize+no-deallocate experiment: the burst moved to the
  very first op, `ttnn.embedding`, `tt_parallel_embedding.py:240`). "32 writes" = one op × 32 devices.
- Program factories that "defer tensor destruction until the cached workload is evicted" (seen across
  pad/pool/conv/sort/reshape factories) pin output tensors forever (unbounded cache), which is one source
  of allocator drift between the warmup that pins and the capture that doesn't.
- A 2nd warmup does NOT fix it (the earlier "2 warmups → writes:0" reading was an artifact: that run
  crashed in warmup2 on a freed memoized tensor *before* capture ran, so 0 writes was trivial).

What would actually fix it (not attempted — real trace-integration work, likely needs MoE-owner input):
make the whole L10 forward allocate deterministically so every op's I/O address matches warmup at capture
(free all warmup intermediates identically; avoid the per-layer pin drift), OR capture the embedding /
pre-MoE prefix in its own trace segment with pinned-address I/O. The per-element metadata deliverable
itself is unaffected and already validated on the original segmented-trace branch (L10/L61 KV-PCC green,
per project memory) — this blocker is specific to re-hosting that trace on top of current `main`'s MoE.
