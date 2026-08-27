# TTTv2 vLLM Sampling Parity Work Log

## Goal

Complete the eight-milestone goal in
`tttv2_vllm_sampling_parity_goal.md` for Llama-3.1-8B-Instruct,
Llama-3.3-70B-Instruct, and Qwen3-32B, including explicit TTTv1 oracle
evidence, TTTv2-native sampling state, exact `top_k` routing, focused tests,
and same-SHA warm-cache hardware/vLLM qualification.

## Checkpoint 0: kickoff and reservation preflight

- Date/time (UTC): 2026-08-26T21:29:24Z
- Coordinating checkout: `/localdev/gwang/tt-metal`
- Branch: `gongyu/tttv2_bh_support`
- Full SHA: `7c2e39c3c7d5020af52ac0d6d1d0dbd51e0e6f19`
- Upstream: `origin/gongyu/tttv2_bh_support`
- Upstream divergence (behind/ahead): `0 0`
- Tracked/staged state: clean
- Pre-existing state preserved: multiple untracked TTTv2 documents and assets,
  including the goal and hardware manual; none is part of the tracked-code gate.
- `wh-lb-42`: reservation time left `10:58:43`, SSH port `47679`.
- `bh-qb-05`: reservation time left `10:58:50`, SSH port `47679`.
- Reservation refresh deadline: 2026-08-27T02:58:45Z (half of the smaller
  remaining reservation interval).
- Hardware policy: at most one TT process per host; different hosts may run in
  parallel. `tt-smi -r` is reserved for confirmed device/lifecycle faults.
- Test gate: no host-only or hardware test may run until the mandatory
  participating-checkout gate confirms identical branch/SHA, clean tracked
  state, configured upstream, and `0 0` divergence locally and on both hosts.

## Checkpoint 1: mandatory participating-checkout gate

- Date/time (UTC): 2026-08-26T21:30Z
- Local: `gongyu/tttv2_bh_support`,
  `7c2e39c3c7d5020af52ac0d6d1d0dbd51e0e6f19`, tracked-clean,
  upstream `origin/gongyu/tttv2_bh_support`, divergence `0 0`.
- `wh-lb-42.yyz2.tenstorrent.com`:
  `gongyu/tttv2_bh_support`,
  `33bb4307e13d9d97558c37571bbe65d9ea1e99af`, tracked-clean,
  upstream `origin/gongyu/tttv2_bh_support`, divergence `0 0`.
- `bh-qb-05.yyz2.tenstorrent.com`:
  `gongyu/tttv2_bh_support`,
  `33bb4307e13d9d97558c37571bbe65d9ea1e99af`, tracked-clean,
  upstream `origin/gongyu/tttv2_bh_support`, divergence `0 0`.
- Result: **failed** because the coordinating checkout SHA differs from both
  participating remote SHAs. The mismatch predates this task; no tracked source
  edits had yet been made locally and no remote state was changed.
- Consequence: all host-only tests, remote copies/pulls/resets, hardware runs,
  and real-vLLM launches are stopped. Read-only evidence collection and local
  source preparation may continue. After reviewed changes are ready, the exact
  synchronization decision required is whether to commit and push the local
  change set as the authoritative candidate, then update both remote checkouts
  to that common pushed SHA. The full gate must be rerun afterward.

## Milestone 1: frozen TTTv1 parity oracle

- Date/time (UTC): 2026-08-26T21:42Z
- Result: **complete as an evidence freeze; no inferred passes**.
- Configuration evidence: `tests/pipeline_reorg/vllm_model_tests.yaml` supplies
  the required legacy selector, `sample_on_device_mode=all`,
  `sampling-tests: true`, and concrete hardware for:
  - Llama-3.1-8B-Instruct on T3K/WH (`tt_transformers`), lines 243-259;
  - Llama-3.3-70B-Instruct on Galaxy lane-DP4
    (`llama3_70b_galaxy`), lines 338-356; and
  - Qwen3-32B on Galaxy lane-DP4 (`qwen3_32b_galaxy`), lines 360-376.
- Result-artifact audit: no tracked model-named plugin sampling pytest report,
  XML, generated result, or log was found. An enabled job is not classified as
  a pass. The available baseline contains zero evidence-backed pass rows and
  zero observed pytest failure rows.
- Llama-3.1-8B explicit YAML filters cover the seeding/variety suite, mixed
  request isolation, and the mixed penalty rows. Its individual penalty and
  logprob tests remain selected but have unknown outcomes.
- Llama-3.3-70B and Qwen3-32B select the seeding/variety, isolation, penalty,
  and logprob serving tests, but every outcome is unknown without a checked-in
  run artifact.
- Code-proven shared failure: legacy
  `models/common/sampling/generator.py` clamps stochastic `top_k < 1` and
  `top_k > 32` to 32. Exact `top_k=33`, `50`, vocabulary-sized/unrestricted,
  unrestricted-plus-`top_p`, batch-wide fallback, and defensive rejection
  therefore fail the goal contract before the fix.
- Code-proven unsupported case: for these Llama/Qwen families, top-N device
  logprobs route to host; the selected serving test does not demonstrate mixed
  logprob/non-logprob rows inside one batch. The known greedy force-argmax
  sampled-token-logprob limitation is retained as unsupported TTTv1 behavior.
- All other required rows are classified **unknown**, including top-k boundary
  exactness, greedy IDs, temperature/top-p, seeds, penalties/history,
  sampled-token logprobs, reset/preemption/remap/compaction, and
  eager/trace/async/DP state refresh. Real TTTv1 serving qualification remains
  required in Milestone 8 before those rows may be promoted to pass/fail.

## Checkpoint 2: exact top-k and prepared-request implementation

- Date/time (UTC): 2026-08-26T22:05Z
- Added topology-neutral
  `models/common/modules/sampling/params.py::PreparedSamplingParams` and
  `prepare_sampling_params`.
- Native formatting now normalizes greedy rows first, preserves stochastic
  `top_k=1..32` exactly, and raises for stochastic `top_k<=0`, `33`, `50`, or
  vocabulary-sized/unrestricted values instead of silently clamping.
- Prepared state retains K/P/inverse-temperature, all three penalties, seeds,
  sampled-token/top-N logprob mode, per-row/batch path classification,
  `prompt_tokens`, `output_tokens`, and `slot_remap`; focused tests were added
  in `models/common/tests/modules/sampling/test_params.py`.
- Added `max_device_top_k=32` to the three target TTTv2 generators and to each
  legacy bridge class that advertises device sampling, including the selected
  TTTv1 Llama/Qwen/Galaxy oracle paths. This prevents the strict plugin
  capability contract from breaking the TTTv1 comparison servers.
- `VLLMAdapter` no longer discards prompt history, output history, or slot
  remaps. The fields are being transported through the target executors,
  eager/traced execution, and lane-DP row/slot slicing.
- Removed aggregate `models.common.sampling` value imports from the audited
  common runtime call sites; native decode/prefill preparation now reads the
  sampler's resolved `max_top_k` and `allow_force_argmax` capabilities.
- Added penalties/logprobs to decode and prefill program/trace identity and
  enabled DP logprob aggregation instead of rejecting non-null lane metadata.
- Prepared an apply-ready vLLM patch at
  `models/docs/tttv2/tttv2_vllm_topk_routing.patch`, pinned to clean vLLM SHA
  `607831803c938c9dc4d92ef0b02a76ba622315fd`. It changes only plugin
  `platform.py` and `model_runner.py`: validates the numeric capability and
  applies a greedy-aware active-row, batch-wide host fallback. The companion
  qualification matrix records boundary/unrestricted/mixed payloads and the
  public-route observability limitation.
- Execution status: source authored and read-only reviewed; **not tested**.
  No pytest, compile, format, remote edit, server, hardware process, copy,
  pull, reset, or `tt-smi -r` was run because Checkpoint 1's gate remains
  failed.

## Checkpoint 3: native state integration prepared; validation blocked

- Date/time (UTC): 2026-08-26T22:23Z
- Added native `SeedManager1D`, caller-owned `SeedState`, and focused lifecycle
  tests for admission, absolute-position refresh, equal-seed salts, unseeded
  diversity, remap, suspend/resume, cleanup, and stable LazyBuffer updates.
- Added per-lane `SamplingState1D`, composed only from `Sampling1D`,
  `Penalties1D`, and `SeedManager1D`; non-1D topology fails construction.
- Target executors now create one controller/state per lane and share it across
  prefill and decode. Penalties run before sampling and sampled-token history is
  updated exactly once; compile-only warmups suppress phantom seed/token state.
- Decode preparation derives active rows from `start_pos>=0`, normalizes
  vLLM's `seed=-1` and disabled-logprob sentinels, places every request field
  back into fixed slots, and extends remaps across sampler-only padding.
- Prefill preserves request ordering while attaching seed streams to their real
  decode slots. Single-request values/seeds/penalty history are replicated over
  physical sampling rows where sequence-tile extraction requires it. Device
  sampling forces sequential prefill planning for all three targets, while a
  public multi-request prefill may still execute as several independent rows.
- DP request histories are routed through lane-local slot grids; sampled-token
  logprobs are normalized and aggregated in global row order.
- Remaining source-review risk to qualify: a non-identity remap for a DP lane
  with no prompt rows in a prefill-only step relies on the fixed-batch decode
  call to deliver that remap; lane-DP normally uses identity remaps. Focused
  tests and real serving must confirm the lifecycle.
- Hard blocker: the mandatory participating-checkout gate remains failed.
  Local started at `7c2e39c3c7d5020af52ac0d6d1d0dbd51e0e6f19` and now has
  tracked implementation edits; both hardware checkouts remain clean at
  `33bb4307e13d9d97558c37571bbe65d9ea1e99af`. The vLLM routing patch is
  staged only as a workspace artifact against clean vLLM SHA
  `607831803c938c9dc4d92ef0b02a76ba622315fd`.
- Exact unblock decision: authorize committing/pushing the reviewed tt-metal
  change set as the authoritative candidate, synchronize both hardware
  checkouts to that pushed SHA, and authorize applying/committing the narrow
  vLLM plugin patch on its registration-only branch. The complete checkout
  gate must then be rerun before any host or hardware validation.

## Checkpoint 4: source/test migration closure and pinned plugin contradiction

- Date/time (UTC): 2026-08-26T22:46Z
- Converted `models.common.sampling` aggregate exports to lazy compatibility
  attributes. Direct imports of the neutral `sampling_params` value module no
  longer execute legacy `generator.py`, `tt_penalties.py`, or `tt_sampling.py`.
  Added both AST and fresh-process import-boundary tests.
- Migrated focused decode-runtime tests from legacy formatter/SeedManager
  expectations to native active-gap placement, exact params, equal-seed salts,
  unseeded RNG advancement, reset-gated admission, remap, static identity, and
  row-major sampled-token logprobs.
- Updated common execution, target executor, lane-group, and prefill fixtures
  for request history/remap arguments, native controller ownership, sequential
  device-sampled prefill, cleanup ordering, and signature separation.
- Compile-only decode/prefill suppress phantom RNG and penalty counts. Decode
  trace capture records the penalty output update but does not capture dynamic
  host seed/penalty writes; replay refreshes those inputs before execution.
- Single-request prefill derives a `-1`-masked prompt history from the supplied
  tokens when the plugin omits a separate history, broadcasts K/P/T, seed, and
  repetition history over physical sequence-tile sampling rows, and attaches
  the persistent seed stream to the provided decode destination when known.
- Read-only audit of the pinned vLLM plugin at
  `607831803c938c9dc4d92ef0b02a76ba622315fd` proved a remaining contract gap:
  - `slot_remap` has identity entries for both unchanged live rows and removed
    unmoved tail rows; there is no inactive sentinel;
  - standard prefill does not forward complete live slots, prompt history,
    output history, or slot remap;
  - lane-DP prefill supplies only scheduled destination slots and scheduled-row
    histories, not the complete post-update live set.
- Consequence: a removed tail request can remain in SeedState during a new
  prefill and perturb equal-seed salt allocation before the next fixed-batch
  decode cleans it. Resumed prefill cannot reconstruct presence/frequency
  history because prompt and generated tokens are not distinguishable from the
  current kwargs. No controller-only inference is sound.
- Required external contract to close this gap: forward a complete lane-local
  post-update live-slot set (for example `prefill_live_slots`, or an explicit
  inactive remap sentinel) and complete prefill prompt/output histories.
  Implementing that is an additional vLLM semantic/plumbing change, which
  conflicts with the goal document's explicit rule that the only vLLM source
  change is the narrow `top_k` routing guard.
- Validation remains prohibited by the unchanged participating-checkout gate:
  local base `7c2e39c...`, both hardware remotes `33bb4307...`; no tests,
  formatters, compile checks, copies, remote edits, servers, hardware runs, or
  resets have occurred.
