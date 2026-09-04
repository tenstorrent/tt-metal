# GLM-4.7-Flash vLLM integration stage: work log

Branch `ttmodelmanager/glm47-flash-probe`, starting commit `8a53bd16b2f`
(datatype-sweep). Target: one Blackhole p150-class chip, device 0, 1x1 mesh
(vLLM mesh name `N150`). Repos touched: `tt-metal` (this repo) and
`/home/stisi/vllm-tt-plugin` (sibling checkout of `tenstorrent/vllm-tt-plugin`,
not nested inside `tt-metal`).

All commands below are literal and rerunnable from the repo root with
`./python_env/bin/python` (tt-metal's `python_env`, which already has the TT
fork of vLLM and `vllm_tt_plugin` installed).

---

## VS-001: minimum-surface bring-up loop, not the full 47-layer model first

Per `$vllm-integration`'s "Minimum-Surface Bring-Up Loop", the adapter's own
contract (kwarg shapes, row/slot bookkeeping, cache-dtype override, page-table
refresh, async decode split) was built and proven against a **reduced 2-layer
model** (`layer_indices=[0, 1]`: the model's one dense layer and one MoE
layer, one of each kind) before ever launching the full 47-layer model through
`run_vllm_server`. Test suite:
`models/autoports/zai_org_glm_4_7_flash/tests/test_generator_vllm_adapter.py`.
This is deliberately *not* final serving evidence (see the skill's "the
reduced target is only an inner-loop tool" warning) -- it exists to catch
adapter bugs in seconds instead of after a multi-minute full-model boot.

```
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_generator_vllm_adapter.py -x -q
```

Four real bugs were caught and fixed by this loop before the full model was
ever launched:

### VS-001a: page-table change detection was permanently defeated by aliasing

`GLM47FlashForCausalLM` maintains a persistent, slot-indexed torch mirror of
the page table (`self._pt_mirror`) and calls
`GLM47FlashGenerator.refresh_page_table(..., only_if_changed=True)` every
prefill/decode call so an unchanged table costs no host->device copy. The
first version passed `self._pt_mirror` **itself** (not a copy) both at the
initial `bind_decode_state` call and on every later `refresh_page_table` call.
`refresh_page_table`'s `only_if_changed` diff does
`torch.as_tensor(page_table_torch, dtype=torch.int32)`, which does not copy an
already-int32 tensor, so the generator's stored "previous value"
(`self._page_table_torch`) ended up being the *same object* as the adapter's
live, mutable mirror. Every later in-place scatter-write to the mirror
silently mutated what the diff considered "the previous value" too, so
`torch.equal(current, new)` was trivially true forever -- the diff never
fired again after the very first call, for the rest of the process. Fixed by
passing `self._pt_mirror.clone()` everywhere the mirror is handed to the
generator (both the initial `bind_decode_state` and every
`refresh_page_table`). Caught by
`test_page_table_refresh_changed_and_unchanged`, which asserts the refresh
counter increments on a genuinely different table and does not on an
unchanged one -- it failed silently-wrong (always "unchanged") until fixed.

### VS-001b: low-level `prefill_forward`/`decode_forward` index page_table by absolute slot, not call-local row

The host-sampling-fallback branches (used when `sampling_params is None`,
e.g. a logprobs request) called
`GLM47FlashGenerator.prefill_forward`/`decode_forward` (the caller-owned-cache
low-level contract) passing vLLM's own row-compacted page-table tensor
directly. That low-level API indexes `page_table` by absolute `user_id`
(physical slot), not by the row position within whatever tensor is passed --
`ttnn.experimental.paged_fill_cache(..., batch_idx=user_id)` asserts
`batch_idx < page_table.shape[0]`, so a 1-row table for `user_id=22` faults
immediately. Fixed by always passing the full, already-refreshed
`self.generator._page_table_dev` (shape `[max_batch_size, blocks_per_user]`)
to these calls instead of the caller's row-compacted tensor.

### VS-001c: `build_generator`'s standalone-cache allocation ran before vLLM's own cache existed

`initialize_vllm_model` originally called `build_generator(...)` with only
`capture_trace=False`. `build_generator` unconditionally calls
`generator._ensure_owned_state()`, which -- because no cache is bound yet --
allocates a full standalone KV cache sized for the model's whole
`max_seq_len`. On the full 47-layer model at 202752 context this OOMs
outright (weights alone leave little headroom for a second full-context
cache), and even when it fits it is exactly the "hidden standalone-cache
assumption" the goal forbids: vLLM's own `allocate_kv_cache` call, which
should be the *only* cache allocation, would then bind a second cache and
orphan the first. Reproduced on real hardware:

```
TT_FATAL: Out of Memory: Not enough space to allocate 3970695168 B DRAM
buffer across 8 banks ... (allocated: 3834686336 B, free: 393901568 B, ...)
```
(`initialize_vllm_model -> build_generator -> generator._ensure_owned_state
-> self.allocate_kv_cache() -> model.allocate_kv_cache -> ttnn.allocate_tensor_on_device`.)

Fixed by adding `build_generator(..., defer_cache_and_traces=True)`: builds
the model/generator/tokenizer only, and returns immediately without touching
cache/warmup/trace capture at all. `initialize_vllm_model` uses this mode;
`GLM47FlashForCausalLM.allocate_kv_cache` binds vLLM's cache via
`bind_decode_state` once it exists, and the plugin's own dedicated warmup
entry points (VS-001d) do the compile/capture work against that real cache.

### VS-001d: `warmup_model_prefill`/`warmup_model_decode` were missing entirely

`vllm_tt_plugin/model_runner.py`'s `warmup_model()` calls
`self.model.warmup_model_prefill(...)` and
`self.model.warmup_model_decode(...)` **unconditionally, with no `hasattr`
guard** -- a model class that does not define them gets a bare
`AttributeError` at server startup, after the (expensive) model load has
already completed. Neither `generator_vllm.py` nor
`models/common/readiness_check/contract_vllm.py`'s own `VllmGeneratorAdapter`
protocol were checked against this until a full grep of
`self.model.<name>(` call sites in `vllm-tt-plugin/src/vllm_tt_plugin/*.py`
turned up both methods. Added both:

- `warmup_model_prefill`: compiles every prefill-bucket program shape
  (`GLM47FlashGenerator.warmup_prefill()`). Called twice by the plugin
  (`enable_trace=False` then `True`); this model's prefill path is never
  traced (same as the DeepSeek-V3 TT adapter), so both calls run the same
  idempotent compile sweep.
- `warmup_model_decode`: a no-op on the `enable_trace=False` (phase 1,
  compile-only) call, since `capture_decode_trace()`'s own uncaptured warm
  pass already compiles everything phase 1 would; captures the decode +
  split-sampling traces on the `enable_trace=True` (phase 2) call.

A regression test (`test_implements_full_vllm_plugin_contract`) now asserts
every method the plugin's own source calls unconditionally is present, so a
future refactor cannot silently drop one again.

## VS-002: shared harness bug -- `run_vllm_server.py` used a renamed vLLM CLI flag

`models/common/readiness_check/run_vllm_server.py`'s `_launch_server` passed
`--plugin-config '{"tt": {...}}'`. The installed vLLM CLI (0.25.1) rejects
that flag (`error: unrecognized arguments: --plugin-config`); the TT plugin
config is now read from vLLM's generic `--additional-config`
(`vllm_tt_plugin/config.py`'s `get_tt_config`/`_extract_tt_config`, which
still expects the same `{"tt": {...}}` shape from
`vllm_config.additional_config`). This is shared-harness code used by every
model's vLLM stage, not GLM-4.7-Flash-specific; fixed directly per
`$vllm-integration`'s "if logs make the cause obvious, fix it directly"
guidance. One-line change: `--plugin-config` -> `--additional-config`.

## VS-003: server launch, full 47-layer model

Command (see `readiness_vllm/server.log` for the exact `vllm serve` argv the
runner builds):

```
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 \
  --max-num-seqs 32 \
  --max-model-len 202752 \
  --tt-config '{"trace_region_size": 350000000}'
```

vLLM auto-detects this model as MLA from HF's `glm4_moe_lite` model_type
(`vllm/transformers_utils/model_arch_config_convertor.py`'s `is_deepseek_mla`
list explicitly includes `glm4_moe_lite`), giving `num_kv_heads=1,
head_size=kv_lora_rank+qk_rope_head_dim=576` for KV-cache-shape purposes --
exactly this model's paged latent-cache entry, with no
`get_kv_cache_spec`/hybrid-attention plumbing needed (same precedent as
DeepSeek-V3). vLLM also auto-disables chunked prefill for this model_type
(`tt/platform.py:153`), consistent with this adapter's own rejection of a
non-zero prefill `start_pos`.

### VS-004: `get_max_tokens_all_users` under-reserved headroom for the cache-reset zero buffer

First full 47-layer boot got all the way through weight loading (17.9 GiB),
vLLM's own KV-cache sizing (`num_gpu_blocks=7956`, "GPU KV cache size:
509,184 tokens" -- matching this adapter's reported budget), and into
`allocate_kv_cache`, then OOM'd inside
`bind_decode_state -> model.prepare_cache_reset -> _cache_zeros ->
ttnn.zeros`:

```
TT_FATAL: Out of Memory: Not enough space to allocate 311620608 B DRAM buffer
across 8 banks, where each bank needs to store 38952576 B, but bank size is
4228587904 B (allocated: 4176451200 B, free: 52136704 B, largest free block:
23730880 B)
```

Root cause: `get_max_tokens_all_users`'s budget subtracted
`weights_plus_persistent_scratch`, which bakes in `cache_reset_zero_buffer`
at a *fixed* 0.116 GiB -- the size measured for the single-request,
202752-context cache (5.431 GiB / 47 layers). But the zero buffer is sized to
one full layer of *whatever pool vLLM actually allocated* (paged cache shared
across all users), and this adapter's own multi-user pool is ~2.5x bigger
(~13.6 GiB) than that single-request cache, so the real zero buffer is
~0.29 GiB -- about 0.18 GiB more than budgeted, which is why one DRAM bank
came up ~37 MiB short. Fixed by solving for `T` (total cache tokens) directly
against `bytes_per_token_all_layers + bytes_per_token_one_layer` (the second
term being the zero buffer, which scales 1:1 with the same `T`), plus a small
explicit 0.25 GiB safety margin for bank-level allocation/alignment overhead
that a whole-device average budget can't see. New reported budget: 487,379
tokens (was 507,082; test bound in
`test_get_max_tokens_all_users_matches_contract` left at
`400_000 < total < 600_000`, still satisfied).

### VS-005: 0.25 GiB margin was still too small -- MoE prefill-warmup scratch, not just bank rounding

Re-running the full 47-layer serve stage with the VS-004 fix got further
(weights loaded, `get_max_tokens_all_users=487379`, KV cache sized and bound)
but the engine core still died during `warmup_model_prefill`, not
`allocate_kv_cache` this time:

```
TT_FATAL: Out of Memory: ... unable to allocate a 402653184 B (384 MiB) DRAM
buffer at tt/fused_decoder.py:429
```

That line is `gu = ttnn.transpose(gu, 1, 3)` -- the post-sparse-matmul
transpose of the MoE gate_up projection's prefill output, a DRAM scratch
allocation that only exists transiently during prefill (compiled once per
prefill-bucket shape by `warmup_model_prefill` before any request is served)
and is not present at batch 1 in the non-serving readiness harness because
that harness's own headroom was never this tight. It is real activation
scratch, not a cache-sizing bug, but it still has to fit in whatever headroom
`get_max_tokens_all_users` leaves after weights+scratch+sampler+trace+cache --
and the 0.25 GiB margin (sized only for DRAM-bank rounding, VS-004) had
nothing left for it. Fixed by raising `safety_margin_gib` from 0.25 to 0.75
GiB (covers the observed ~0.38 GiB (384 MiB) scratch peak plus the original
~0.04 GiB bank-rounding term, with room to spare). New reported budget:
469,104 tokens (test bound `400_000 < total < 600_000` still satisfied;
reduced-model adapter test suite re-run clean, 9/9 pass).

Per the goal's execution-model instruction (full validation runs longer than
one turn), the full pipeline (serve+sampling(smoke)+qualitative+benchmark) was
then launched as a **detached** background process (`setsid`, stdin from
`/dev/null`, disowned) so it survives this session ending. See "Detached
validation run" below for the exact PID/log/command.

## Detached validation run (in flight)

Launched 2026-09-03 ~02:36 local, per the goal's execution-model instruction
(the full serve+sampling+qualitative+benchmark pipeline runs longer than one
turn budget):

```
cd /home/stisi/tt-metal
source python_env/bin/activate
setsid nohup python3 -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 \
  --max-num-seqs 32 \
  --max-model-len 202752 \
  --sampling-profile smoke \
  --tt-config '{"trace_region_size": 350000000}' \
  > /tmp/glm47_vllm_detached_run.log 2>&1 < /dev/null &
disown
```

- **PID: 35176** (`python3 -m models.common.readiness_check.run_vllm_server ...`).
  Confirmed reparented to init (`PPid: 1`), own session (`SID 35176`), no
  controlling TTY (`ps` shows `TT=?`, `STAT=Ssl`) -- survives this agent
  session ending.
- Default `--stages` (not passed): runs `serve,sampling,qualitative,benchmark`
  in one invocation, so a healthy finish produces every required artifact in
  one pass. `--sampling-profile smoke` for this first full-model pass per
  `$vllm-integration`'s "smoke first, then full" order; rerun with
  `--sampling-profile full` after smoke passes, for final evidence.
- Runner-side log: `/tmp/glm47_vllm_detached_run.log` (stdout+stderr of the
  `run_vllm_server` driver itself: launch/health-poll/stage-orchestration
  messages, not the vLLM server's own log).
- vLLM server log: `models/autoports/zai_org_glm_4_7_flash/readiness_vllm/server.log`
  (the actual `vllm.entrypoints.openai.api_server` subprocess output -- weight
  loading progress, warmup, request handling, any TT_FATAL/Traceback).
- Other artifacts this run should produce under `readiness_vllm/` if it
  completes: `sampling_tests.log`, `vllm_qualitative_outputs.json`,
  `vllm_result.json` + `vllm_benchmark.json` + `vllm_benchmark.log` (primary
  single-user), `vllm_ci_serving_result.json` + `vllm_ci_serving_benchmark.json`
  + `vllm_ci_serving_benchmark.log` (secondary CI serving-burst).

**To check status from a later session:** `ps -p 35176` (alive = still
running); tail both logs above; if the process has exited, check its exit
state from the tail of `/tmp/glm47_vllm_detached_run.log` and whether the
`readiness_vllm/` artifacts above exist and are non-empty/well-formed.

**Update: that PID 35176 run failed** (see VS-006) -- superseded by a new
detached run below.

### VS-006: 0.75 GiB margin fixed the 384 MiB peak but exposed a larger 768 MiB one -- wrong lever was margin, right lever is prefill chunk length

The detached run (PID 35176) reached the same op again --
`tt/fused_decoder.py`'s `FusedDecoder._moe_prefill`, the line-429 MoE gate_up
post-sparse-matmul transpose (`gu = ttnn.transpose(gu, 1, 3)`) -- now failing
to allocate 805,306,368 B (768 MiB), with DRAM essentially full (per-bank free
~48 MiB, largest free block 22.8 MiB). Traced why: this buffer's shape is
`[1, G, 1, E, 32, 2*inter]` where `G = S // TILE` and `S` is the **prefill
chunk length being processed** (`tt/fused_decoder.py`'s `_moe_prefill`), not
anything related to the KV-cache pool. `FusedDecoder.prefill_forward` already
splits any prompt into `self.prefill_chunk_size`-sized chunks internally
(`for start in range(0, S_pad, chunk): ...`), so the true worst case per call
is bounded by `prefill_chunk_size` alone, and `warmup_prefill` directly warms
each value in `prefill_buckets` (default `(128,256,512,1024,2048)`) as a
single-chunk call -- the largest bucket, 2048, equals the default
`prefill_chunk_size`, so warming it hits the worst case directly (`G=64`).
Measured scaling is exactly linear: 402,653,184 B (384 MiB) at chunk=1024
(`G=32`) vs. 805,306,368 B (768 MiB) at chunk=2048 (`G=64`) -- solving
`bytes = G * n_experts(64) * TILE(32) * 2*inter * 2(bf16)` for `inter` from
either point gives `moe_inter=1536` consistently.

Raising `safety_margin_gib` further would only shrink the KV pool to make
room for whatever the *largest* warmed chunk needs -- fighting the wrong
variable, and directly against the goal's "keep max_num_seqs=32,
max_model_len=202752 unchanged, don't just throw more margin at it" guidance.
The real fix is capping the chunk length itself, which is spec-preserving:
`prefill_physical_len`/`FusedDecoder.prefill_forward`'s existing chunking
already reaches the full 202752-token context via more, smaller chunks
regardless of `prefill_chunk_size`'s value -- nothing about served context or
concurrency changes.

Fix: `tt/generator_vllm.py` gained `VLLM_PREFILL_CHUNK_SIZE = 1024` and
`VLLM_PREFILL_BUCKETS = (128, 256, 512, 1024)` (drops the 2048 entry, which
would be dead weight as a tail-bucket once chunk=1024 anyway, since a tail
remainder is always `< prefill_chunk_size`), passed into `build_generator(...)`
from `initialize_vllm_model` only -- the readiness/full-model/datatype-sweep
stages keep the model's own default (`prefill_chunk_size=2048`) unchanged, so
none of their already-published perf/accuracy evidence is affected. This is a
compute/memory-layout knob, not a precision/fidelity one, so it does not
touch `doc/datatype_sweep/selected_precision_config.json`'s contract. 1024 is
not a guess: PID 35176's own run already proved every op for every warmup
bucket up to and including 1024 completes cleanly end-to-end (all 47 layers)
on this exact DRAM budget -- only the next bucket, 2048, failed. Reduced-model
adapter test suite re-ran clean, 9/9 pass, after this change.

Trade-off disclosed, not hidden: `warmup_terminal_shapes`'s per-bucket tile
offsets actually *decrease* slightly (dropping the 2048 bucket removes its 64
offsets: 4+8+16+32=60 vs. the previous 124), so terminal-path warmup is
cheaper, not more expensive. What does get more expensive is a prompt spanning
many chunk boundaries near the full 202752-token context: roughly 2x as many
distinct chunk-offset programs as at chunk=2048, compiled lazily on first use
of that exact prompt length per `warmup_terminal_shapes`'s own docstring (this
was already true at chunk=2048 -- only the constant changes). No correctness
impact; a first request at a new very-long length pays more one-time compile
cost inside its own TTFT.

## Detached validation run #2 (in flight)

Same command as before, relaunched after the VS-006 fix:

```
cd /home/stisi/tt-metal
source python_env/bin/activate
setsid nohup python3 -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 \
  --max-num-seqs 32 \
  --max-model-len 202752 \
  --sampling-profile smoke \
  --tt-config '{"trace_region_size": 350000000}' \
  > /tmp/glm47_vllm_detached_run2.log 2>&1 < /dev/null &
disown
```

- **PID: 35858** (`python3 -m models.common.readiness_check.run_vllm_server ...`).
  Confirmed reparented to init (`PPid: 1`), own session (`SID 35858`), no
  controlling TTY (`STAT=Ssl`, `TT=?`) -- survives this agent session ending.
  Launched 2026-09-03 ~02:47 local.
- Runner-side log: `/tmp/glm47_vllm_detached_run2.log`.
- vLLM server log: `models/autoports/zai_org_glm_4_7_flash/readiness_vllm/server.log`
  (overwritten from run #1; run #1's failure is preserved verbatim in this
  work log's VS-006 section above and in `/tmp/glm47_vllm_detached_run.log`
  if that file is still present).
- Same expected `readiness_vllm/` artifacts as listed for run #1 above.
- **To check status from a later session:** `ps -p 35858`; tail both logs;
  if exited, check `/tmp/glm47_vllm_detached_run2.log`'s tail and whether the
  `readiness_vllm/` artifacts exist and are well-formed. If it failed with a
  new DRAM peak at some OTHER op (not the VS-006 transpose), that would mean
  1024 is still too large for some other chunk-scaled op, or margin needs a
  measured (not guessed) adjustment -- do not just increase margin blindly.

**Closed out:** PID 35858 boot itself was healthy (the VS-006 chunk cap held),
but the full VS-007/VS-008/VS-009 investigation below (seed determinism,
sampling-lane state, the upstream-filed contamination defect) continued
across further sessions before the final evidence-collection pass. That final
pass is VS-010.

## VS-010: evidence collected at spec, smoke-gated

Final run at the real serving spec (one Blackhole p150, full 202752 context,
`max_num_seqs=32`, on-device sampling), after VS-007/VS-008/VS-009 landed.
Command (default stages, `--sampling-profile smoke`):

```
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/zai_org_glm_4_7_flash \
  --hf-model zai-org/GLM-4.7-Flash \
  --mesh-device N150 --max-num-seqs 32 --max-model-len 202752 \
  --sampling-profile smoke \
  --tt-config '{"trace_region_size": 350000000}'
```

produced the server boot, smoke sampling (3 passed/1 skipped/0 failed),
qualitative (6 prompts), and both benchmark profiles in one pass
(`server.log`'s `APIServer pid=89584` session, 19:20:03-19:36:47). The
recorded-but-not-gated `--sampling-profile full` run
(`sampling_tests_full_RECORD.log`, 11 failed/62 passed/1 skipped,
452.11 s) was a follow-up attach to that same live server using the
documented `--stages sampling --sampling-profile full --server-url
http://localhost:8000` pattern; its exact invocation was not preserved
verbatim (recorded as a provenance gap, README Known Limitations #4).

Headline: TTFT 273.8 ms (128-token prompt), decode 45.0 ms/token = 22.2 t/s/u
batch-1, serving burst 100/100/32 at 137.1 tok/s output / 274.2 tok/s total.
Both runner-side gates pass: `check_degenerate_output --scope all` exit 0
(no degenerate output across 12 completions), `check_context_contract
--stage vllm` exit 0 (target 202752 = supported 202752, two advisory-only
notes about an unrelated `--max-model-len 8192` debugging mention in VS-007/
VS-008, non-blocking).

Committed as `e95eb76d725` ("glm47-flash autoport: vLLM stage evidence at
spec, smoke-gated (VS-010)"). Full artifact list and headline numbers are in
`doc/vllm_integration/README.md`.

## VS-011: `$stage-review` findings, fixed

Independent review (`$stage-review`, fresh subagent, read-only, no hardware)
against the `e95eb76d725` evidence found one real correctness bug and several
documentation/attribution issues. Verdict was `more-work-needed`; findings and
fixes:

**P1, real bug -- `allocate_kv_cache` sized the per-request page-table width
wrong.** `model.blocks_per_user = num_blocks // self.max_batch_size` treats
vLLM's shared block pool as if it were divided into 32 equal, fixed
per-request shares -- it is not; vLLM's paged allocator lets one request use
far more than `num_blocks / max_batch_size` blocks as long as the sum across
concurrent requests fits `num_blocks`. At this stage's measured
`num_blocks=7362`, the wrong formula gave `blocks_per_user=230` (14,720
tokens) instead of the correct, `max_seq_len`-derived `cdiv(202752, 64)=3168`
(202,752 tokens) -- and `_write_page_table_rows` silently truncated any wider
table to that wrong width instead of raising, so a long request would have
been served with a partially-filled cache and wrong logits, not an error.
`GLM47FlashModel.max_seq_len_physical` (which clamps `prefill_physical_len`)
inherited the same wrong bound. Confirmed on real hardware to be the actual
number this stage's own run produced (`server.log`'s `num_gpu_blocks=7362`).

Fixed in `tt/generator_vllm.py`'s `allocate_kv_cache`: `blocks_per_user` is now
`cdiv(model.max_seq_len, block_size)`, computed independently of `num_blocks`
(with a loud `ValueError` if `num_blocks` is ever too small to hold even one
full-context request -- a real hard-physical-limit case, which this run's
pool is not: 7362 >> 3168). `_write_page_table_rows` now raises instead of
truncating if a table is ever wider than that. New hardware-verified
regression test, `test_blocks_per_user_is_max_seq_len_derived_not_pool_derived`
in `tests/test_generator_vllm_adapter.py`, uses a deliberately
non-equal-share pool (`NUM_BLOCKS_SHARED_POOL = BLOCKS_PER_USER + 8`, where
the old formula would have given `72 // 32 == 2`) and proves both the correct
width and the raise-on-overflow behavior. Full reduced-model suite re-ran
clean, 15/15 pass.

None of this stage's measured numbers are affected (every served request --
128/100-token benchmarks, 13-28-token qualitative prompts -- was always far
under both the old wrong cap and the corrected one). Re-verifying through a
live vLLM server request above the old 14,720-token cap was judged a new
hardware serving run, out of this review round's budget; the fix is proven by
the reduced-model hardware test only. See README's "Post-evidence-collection
correctness fix" note.

**P1, attribution overreach -- the #55408 write-up overstated its own
mechanism.** The original wording ("greedy requests lose determinism ...
seeded incidental", "monotone poisoning after a long host-sampled request")
does not match the committed `sampling_tests_full_RECORD.log`: several of the
11 failures are pure `temperature=1.0` seeded-reproducibility assertions with
no greedy row in the batch at all, and `test_specific_seed_reproducible`
alternates FAIL/PASS/FAIL/PASS across its four seed parametrizations at a
fixed batch size, which a clean "batch==32" or "poisoned-forever" rule cannot
explain either. Re-derived and corrected using only the already-committed log
(no new hardware run): README "Known limitations" #2 now states exactly what
the log shows (full-occupancy determinism breaks, both greedy-in-mixed-batch
and pure-seed forms, discriminator not fully identified), names the concrete
next step to disambiguate adapter-vs-upstream (an A/B reverting the VS-008
lane-broadcast), and stops short of asserting the upstream attribution as
settled fact. The upstream issue #55408 itself was not edited (out of scope,
no new evidence to add).

**P2s, fixed:** the degenerate-check number range in the README was
transcribed wrong (corrected from 0.0-0.015 to the actual 0.0-0.0273);
`generator_vllm.py`'s module docstring and `generator.py`'s
`apply_prefill_sampling_state` docstring both still said "seed drops on a
condense" after VS-007's position-anchoring fix already closed that specific
gap -- corrected to describe the current mechanism accurately, without
overclaiming that it also closes the separate open full-occupancy defect;
added the missing primary-profile aggregate-output-throughput number; added
the qualitative prompts' real tokenized lengths (13/20/28/14/19/15, all
non-aligned to tile/block/chunk sizes) as through-serving non-aligned-length
evidence, since none had been called out explicitly before.

**P2s, disclosed rather than fixed (would require new hardware evidence):**
the pre-VS-008 full-profile log and the SmolLM2 cross-check log that VS-009's
strongest claims depend on were never committed; the one
active-trace-during-warmup allocator warning was not run through the
`TT_METAL_TRACE_ALLOC_TRACKING=1` probe the codebase already has for this
exact hazard class. Both recorded in README Known Limitations rather than
chased further, per this review round's budget.

---

## VS-007: per-request seed determinism on the device-sampling decode path

**Symptom.** The plugin sampling suite's
`test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch`
failed: it re-runs a heterogeneous batch in shuffled order and asserts every
greedy-or-seeded request reproduces exactly. A `temperature=0.5, seed=42`
request did not.

**Root cause.** Per-request seeds were never applied on this path at all.
`generator_vllm.py::_slice_sampling_params_row` forced `seed=None`, and the
decode path (`apply_decode_sampling_state` -> `apply_decode_state`, and
`decode_step_traced` -> `sample`) never called `seed_manager.get_new_values()`,
which is the only place `_active_request_seed` is set. So
`has_active_request_seed()` was always False and every "seeded" request drew
from the shared global unseeded `rand_tile` stream, whose per-row output
depends on batch composition and global step count. This was the documented,
deferred FM-023 limitation, not an accident.

**Fix** (mirrors `models/tt_transformers/tt/generator.py::sample_decode_on_device`,
reusing the existing `models/common/sampling` SeedManager; no new seed hashing):

1. `tt/generator.py` `_SamplingArgs`: `salt_duplicate_seeds = False`. The
   contract shares `seed=42` across requests; salting duplicates apart makes
   the salt admission-order-dependent and breaks order-independence. Mirrors
   `models/demos/llama3_70b_galaxy/tt/model_config.py`.
2. `tt/generator_vllm.py` `_slice_sampling_params_row`: stop forcing
   `seed=None`; thread the per-row seed through.
3. `tt/generator_vllm.py` `decode_forward`: pass `start_pos=pos` into
   `apply_decode_sampling_state` on the `reset_batch` path.
4. `tt/generator.py` `apply_decode_sampling_state`: accept `start_pos`; after
   `apply_decode_state`, run `deactivate_slots_except` ->
   `reset_seed_from_slots` (reset) / `reset_seed_from_slots_if_needed` ->
   `align_seed_counters_to_positions`. Position anchoring is what keeps the
   stream row-independent across a mid-generation condense, which is why
   `slot_remap` stays unconsumed.
5. `tt/generator.py` `decode_step_traced`: call
   `seed_manager.get_new_values(active_rows)` before the traced-vs-eager
   decision and before `sample`. This is the missing per-token advance.

Unseeded batches are unaffected: `get_new_values` early-returns in its steady
state, `has_active_request_seed()` stays False, and the captured greedy/penalty
sampling trace still replays.

**Evidence (A/B on hardware, one p150, `--max-model-len 8192`, tests run in
isolation to avoid the cross-test contamination noted below).**

| test | no-fix baseline | with fix |
|---|---|---|
| `test_mixed_params_batch` | FAIL (Request 0) | **PASS** |
| `test_top1_is_greedy` | PASS | PASS (no regression) |
| `test_uniform_seed_deterministic` (6 params) | -- | PASS |
| `test_temperature_varied_in_batch` | FAIL (5) | FAIL (5), unchanged |
| `test_topk` | FAIL (3) | FAIL (3), unchanged |

Host-side: `models/common/tests/test_sampling.py` seed-isolation tests 9/9 pass.
Device regression: `tests/test_full_model.py` sampling/seed tests 7/7 pass,
including `test_request_seed_is_refused_rather_than_silently_ignored` (the
high-level `set_sampling_params` refusal is off the vLLM low-level path and is
deliberately left in place).

**Two findings this surfaced, NOT fixed here.**

* **A GLM-specific unseeded per-row variety defect.** `test_temperature_varied_in_batch`
  fails identically with and without this fix. Cross-checked against
  SmolLM2-135M served through `tt_transformers`' `LlamaForCausalLM` on the same
  chip with the same `sample_on_device_mode=all`: the reference model **passes**
  all 5, so this is not the shared sampler, the plugin, or the single-chip
  geometry. It is how this model drives the unseeded RNG path. `test_topk` fails
  on both models but for different assertions (GLM: top_k half lacks variety;
  SmolLM2: greedy half not deterministic), so it is not evidence of a shared
  cause. Tracked as the next item.
* **The reference path fails `test_mixed_params_batch` too.** SmolLM2 via
  `tt_transformers` fails it (Request 0) while this model with this fix passes,
  so per-request seed reproducibility appears broken on the reference path on
  this config. Worth filing upstream.

**Cross-test contamination in `--sampling-profile full`.** Several tests pass in
isolation but fail inside the 74-test sequence against one long-lived server
(`test_top1_is_greedy`, `test_mixed_params_batch`, `test_uniform_seed_deterministic[32-0]`).
The full-profile result is therefore not a clean per-test signal, and the gate
will stay red on that alone. Needs its own investigation.

**Commits.** tt-metal `9faa3d324dc` (branch `ttmodelmanager/glm47-flash-probe`);
vllm-tt-plugin `9f2ec5d` (branch `ttmodelmanager/glm47-flash-registration`,
branched off main rather than committing to the default branch). Neither pushed.

---

## VS-008: prefill sampling params reached a lane nothing reads

**Symptom.** 11 plugin sampling tests failed: `test_temperature_varied_in_batch`
(x5), `test_topk` (x3), and the three `*_penalty_mixed_batch`. Identical
unseeded requests at `temperature=2.0` all produced the *same* first token, and
the same token on re-runs. Whole outputs varied; only the first token did not.

**Root cause.** A request's sampling parameters were written to a sampler lane
that is never read.

1. The plugin hands per-row lists (`vllm_tt_plugin/model_runner.py`, `.tolist()`),
   so `_slice_sampling_params_row` yields row *i*'s **scalar**.
2. `format_sampling_params` wraps that scalar to a 1-element list
   (`models/common/sampling/generator.py:500`), so `active_len = 1` (`:528`).
3. Each per-user field is emitted as `[request value] + 31 * default`
   (`_pad_per_user`, `:537`). Those defaults are **greedy** (temp 1.0, k 1, p 1.0).
4. `TTSampling.reset_params` rewrites **all 32 rows** from those lists; its
   `empty_slots` argument is never used to merge (`tt_sampling.py:593-640`).
5. Prefill's sampler tile is indexed by prompt **position**, not by user: the
   first token is read from lane `(seq-1) % 32`
   (`tt/model.py::prefill_forward_last_logits_device`). For any prompt longer
   than one token that is a padded, greedy lane.

So the request's own temperature/top_k/top_p/penalties sat on lane 0, which
nothing reads, and **every first token was sampled greedily regardless of what
the client asked for**. A correctness defect, not just a variety one.

Hardware instrumentation (7 identical requests, `seq=4`) confirmed it exactly:
`row=3` for every request (slot-independent), `row_eq_slot` true only for
`user_id=3`, every request returning token 50 (`'S'`), lane 0 varying across
calls while lanes 1-7 stayed frozen.

The shared formatter states the assumption this model breaks
(`generator.py:539`: "a one-user batch has always meant lane 0"); tt_transformers
satisfies it by placing request *i* at row `empty_slots[i]`.

**Fix.** `_broadcast_per_user_fields` (`tt/generator.py`) expands a single
request's scalar per-user fields to all 32 lanes before `format_sampling_params`,
so whichever lane prefill reads carries that request's params. Applied in
`apply_prefill_sampling_state` (vLLM path) and `set_sampling_params` (the
high-level `generate()` path, which had the same latent bug). `seed` is
deliberately excluded (lane-scoped by design); `enable_log_probs`/`num_logprobs`
are already broadcast by the formatter. Decode is untouched: its params arrive
as per-lane lists and its row index is the slot, which is correct.

Host-only change: no device ops, no new programs. No blast radius, because
`reset_params` already rewrote all 32 rows on every prefill; only the contents
of lanes 1-31 change. `force_argmax` cannot flip as a result: it is gated on
`_allow_force_argmax_sampling` (`tt_sampling.py:137`), set only from
`args.model_config["SAMPLING_AG_CONFIG"]` (`:255`) and otherwise False (`:261`),
and this model's `_SamplingArgs` defines no `model_config`.

Also **R2**: `_slice_sampling_params_row`'s `_at` now indexes `torch.Tensor`
params. `TTSamplingParams` types every per-user field as `Tensor | list`
(`model_input.py:29-37`); a tensor previously passed through whole, which would
have handed one request the whole batch's values. Latent today only because the
plugin `.tolist()`s.

**Evidence (one p150, `--max-model-len 8192`, tests in isolation).**

| test | before | after |
|---|---|---|
| `test_temperature_varied_in_batch` | FAIL (5) | **PASS (5)** |
| `test_topk` | FAIL (3) | **PASS (3)** |
| `test_repetition/presence/frequency_penalty_mixed_batch` | FAIL (3) | **PASS (3)** |
| `test_mixed_params_batch` | PASS | PASS |
| `test_top1_is_greedy` | PASS | PASS |
| `test_uniform_seed_deterministic` | PASS (6) | PASS (6) |
| `test_specific_seed_reproducible` / `batch1_seed_reproducible` / `different_seeds` / `uniform_noseed_varied` / `min_p` | PASS | PASS |

Host: 5 new VS-008 regression tests pass; `models/common/tests/test_sampling.py -k seed` 9/9.
Full suite at 8192: **17 failed / 56 passed -> 13 failed / 60 passed**.

**What this unmasked (VS-009, next item).** Four seed-reproducibility tests
(`test_specific_seed_reproducible`, `test_uniform_seed_deterministic`,
`test_seeding`, `test_same_seeds_reproduce_across_batches`) now fail *in
isolation* where they previously passed. They were passing for the wrong reason:
the read lane was greedy, so the draw was deterministic and "reproducible"
trivially held. With the lane now honouring temperature, the seed must actually
control the draw, and it does not. Confirmed directly against the live server:

```
FIRST TOKEN ONLY (max_tokens=1, temperature=2.0)
  seed=0    run1='17'  run2='46'  DIFFER
  seed=42   run1='5'   run2='4'   DIFFER
  seed=123  run1='18'  run2='12'  DIFFER
```

The seed analogue of this same bug: the prefill seed does not reach the lane the
token is read from. `apply_prefill_state` does pass `replicate_seeds=True`
(which broadcasts the device seed across lanes) so the mechanism is not yet
established; it needs its own investigation. Not a VS-008 regression: the
defect predates it and VS-008 is a net improvement (17 -> 13 failures).

### Correction to VS-007: the suspected force_argmax trace thrash does not exist

An earlier note hypothesised that a greedy request's prefill would make all 32
lanes satisfy `_is_force_argmax_sampling`, flipping `force_argmax`, calling
`reset_trace()` and forcing a sampling-trace recapture on every admission.

**Disproven, on hardware.** `_is_force_argmax_sampling` is gated on
`_allow_force_argmax_sampling` (`models/common/sampling/tt_sampling.py:137`),
which is set only from `args.model_config["SAMPLING_AG_CONFIG"]` (`:255`) and is
otherwise `False` (`:261`). This model's `_SamplingArgs` defines no
`model_config`, so the gate is permanently off. Verified with the model's own
args on device (`probe/force_argmax_check.py` pattern):

```
_SamplingArgs has model_config attr : False
_allow_force_argmax_sampling        : False
  after all-greedy (32 lanes)    force_argmax_sampling = False
  after sampled  (32 lanes)      force_argmax_sampling = False
  after GREEDY const broadcast   force_argmax_sampling = False
```

This is consistent with the model's stated design (module docstring: greedy is
*semantically greedy split sampling*, not force-argmax, chosen deliberately so a
mixed greedy/sampled workload does not recapture on each mode change). No code
change. The two `reset_trace()` sites in `tt/generator.py` are the expected
ones: the post-compile recapture in `_maybe_recapture_after_compile` (counted by
`counters["trace_recaptures"]`) and the request-boundary reset.

---

## VS-009: the residual failures are cross-test contamination, not a seed defect

**Correction to the VS-008 note.** VS-008 recorded that four seed-reproducibility
tests "now fail in isolation" and inferred a distinct prefill-seed defect. That
inference was wrong, and the method behind it was flawed: those runs used a fresh
*pytest process* but a **server that had already served the full 74-test suite**.
Isolating the client does not isolate the server.

On a genuinely fresh server every one of them passes:

```
test_specific_seed_reproducible            4 passed
test_uniform_seed_deterministic            6 passed
test_seeding                               1 passed
test_same_seeds_reproduce_across_batches   1 passed
```

and identical seeded requests reproduce exactly at the first token, which is the
thing VS-008 was said to have broken:

```
seed=42  -> 'S'      seed=42  -> 'S'
seed=123 -> 'Given'  seed=123 -> 'Given'
```

with the params genuinely sampling rather than greedy (temperature 2.0 arrives
inverted as 0.5, top_k 32). **There is no separate prefill-seed defect, and
VS-008 regressed nothing.**

**What is real: server state accumulates and breaks later requests.**

One trigger is bisected and canary-confirmed. Using `test_seeding` as a canary
after each host-only-param test, on a fresh server:

| step | canary |
|---|---|
| baseline | PASS |
| after `test_min_p` | PASS |
| **after `test_bad_words`** | **FAIL** |
| after `test_logit_bias` / `test_allowed_token_ids` / `test_min_tokens` | FAIL (stays poisoned) |

`bad_words` forces vLLM to sample on the host, so the adapter takes its
`sampling_params is None` branch, which by design "does not touch the persistent
per-slot decode state" and bypasses the seed manager entirely. Meanwhile
`_seed_active` latches `True` after the first seeded request and never clears
(instrumented: 12 `seed_active=False` prefills then 372 consecutive `True`),
because `deactivate_slots_except` only runs on a decode `reset_batch`. The device
is left holding the seeded path's non-SKIP reinit values, the exact hazard that
method's own docstring warns about.

**Not the only trigger.** Deselecting `test_bad_words` from the full suite moves
it from 13 failed / 60 passed to **11 failed / 62 passed**, so at least one more
contributor remains unidentified. Ruled out as triggers by direct test (each ran,
then the canary still passed): `test_logprobs` (20 host-sampled requests),
`test_request_isolation::test_mixed_params_batch`, `test_temperature_varied_in_batch`,
`test_min_p`.

**Scope.** Every residual failure is order-dependent state, not a wrong answer to
a single request: a freshly started server serves seeded, unseeded, mixed-param
and penalty requests correctly. The defect surfaces only after a long-lived
server has served a particular mix. That makes it a real serving concern worth
fixing, but not a blocker on this model's single-request correctness, and it is
plausibly shared rather than GLM-specific (the SmolLM2/tt_transformers reference
on this same chip also fails `test_mixed_params_batch`).

**Not fixed here.** Remaining work: identify the second trigger the same way
(canary bisection over the suite), then decide whether the fix belongs in this
adapter's host-sampling branch (re-register seed-manager state after a
host-sampled request) or upstream in the shared sampler's seeded/unseeded device
handoff. Track as its own item; stage-7 evidence is collected on a fresh server
per stage and this is recorded under the stage README's known limitations.

### VS-009 addendum: the contamination is greedy determinism, not seeding

Instrumented boot (seed-manager state + the device seed tensor logged at every
prefill), comparing a fresh canary that PASSES against a poisoned canary that
FAILS on the same server.

**The seed path is provably innocent.** Every request seed derives an identical
device seed, counter and salt in both runs:

```
 req_seed |    FRESH dev_seed(ctr,salt) | POISONED dev_seed(ctr,salt) | match
        0 |        ('607536', '1', '0') |       ('607536', '1', '0')  | SAME
      100 |        ('749808', '1', '0') |       ('749808', '1', '0')  | SAME
      ...  (10 of 10 seeds identical)
```

**The failing assertion is about greedy requests, not seeded ones:**

```
AssertionError: Greedy requests should produce the same output across positions and runs.
Got 3 unique results out of 24.
Results: ['Squid\nSquid are marine moll', '1\n\nS = S\nS = S', 'Squid...', ...]
```

24 `temperature=0` requests, 3 distinct outputs. So the contamination breaks
**greedy (argmax) determinism in a mixed batch**, and the seeded requests in the
same test are incidental. This also explains the SmolLM2 cross-check, whose
`test_topk` failed on the identical assertion.

**It predates VS-008.** The same assertion text appears 12 times in the
pre-VS-008 sampling log, and `test_seeding` / `test_top1_is_greedy` both failed
in that run. VS-008 neither caused nor fixed it.

**Also confirmed:** a host-sampled request never reaches
`apply_prefill_sampling_state` at all (zero probe lines logged across the whole
`min_tokens` test), so it bypasses per-slot sampling state entirely, and
`_needs_skip` is left latched `True` because `_seed_active` keeps
`get_new_values` on the seeded branch, which never pushes the SKIP sentinel.
Neither of those changes the device seed, so they are not the failure mechanism,
but both are real state-hygiene gaps.

**Revised suspect.** Greedy rows losing argmax determinism points at the
per-lane `k`/`p`/`temp` a running request holds when another request's prefill
rewrites all 32 lanes (`TTSampling.reset_params` ignores `empty_slots` and
rewrites every row). The adapter relies on the next `reset_batch` to repair
that; any window where a greedy row decodes with another request's `k`/`temp`
before the repair produces exactly this symptom. Confirming that needs per-lane
`k`/`temp` logged at decode, not at prefill.

**Fix shape this implies** (not yet implemented): place a prefill request's
params only on the lane the token is actually read from, `(seq-1) % 32`, plus
its slot, rather than broadcasting to all 32. That requires tracking the live
per-lane params on the host so the untouched lanes can be preserved through
`reset_params`' all-rows write. Strictly narrower than the current broadcast and
removes the clobber window rather than racing the repair.

### VS-009: #48222 ruled out, and the upstream filing

**`ttnn.sampling` is correct at this model's shapes.** Issue #48222 reports
`ttnn.sampling` disagreeing with argmax at `k=1` on a fraction of batch rows,
with the error growing as the gathered candidate buffer widens. That is the
closest existing match to the greedy-determinism symptom, so it was tested
directly: vocab 154880 -> 4 vocab splits, `max_top_k=32` (a 128-wide candidate
buffer), 32 lanes, greedy `k=1, p=0, temp=1`, each row given a distinct
unambiguous peak.

```
TOTAL mismatched rows: 0 / 256   (8 trials, peak margins 2.0 and 0.25)
```

So #48222 does not reproduce single-chip; it appears TP-width dependent. #50512
(batched paged-attention decode race) is likewise scoped to multi-device TP ops
and the TP all-gather, which a 1x1 mesh does not have.

**Filed upstream as tenstorrent/tt-metal#55408** with the canary bisection, both
triggers, the fresh-vs-poisoned device-seed comparison, the SmolLM2 reference
data, and the list of eliminated hypotheses.

**Cumulative eliminations for this defect** (all checked on hardware): the
shared sampler primitive and its per-row RNG; the vocab split; a prefill-seed
defect; per-request seed state (device seeds identical fresh vs poisoned); the
`force_argmax` trace thrash; a stale host position mirror; a stale plugin
`reset_batch` flag; a `reset_params` skip-if-unchanged path; and `ttnn.sampling`
greedy correctness (#48222).

**Remaining suspect**, unproven and needing decode-time per-lane logging:
`TTSampling.reset_params` rewrites all 32 lanes and ignores `empty_slots`, so a
prefill rewrites every concurrently-decoding request's `k`/`p`/`temp`, and the
adapter relies on the next `reset_batch` to repair it.

**Stage-7 posture.** This is a pre-existing, upstream-tracked serving-state
defect that also affects the `tt_transformers` reference on this chip (worse:
it fails the same canary at baseline). It does not affect single-request
correctness on a freshly started server, and it does not touch the runner-side
stage gate (degenerate output + context contract). Recorded here as a known
limitation rather than a blocker.
