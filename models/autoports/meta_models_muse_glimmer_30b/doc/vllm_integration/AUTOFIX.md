# AutoFix — the vLLM serving decode hang

## Starting Evidence

* `doc/vllm_integration/AUTODEBUG.md` — symptom, four bisect matrices, one refuted fix.
* `doc/vllm_integration/triage/tt-triage.txt` — captured on the live hang:
  `PagedUpdateCacheDeviceOperation (trace id: 0)` RUNNING on 3 cores/device,
  previous op `RotaryEmbeddingHfDeviceOperation` complete, queued tail never
  starts; `check_noc_status` fails on the ethernet/fabric-router cores of all
  four devices with unretired NOC reads; `dump_callstacks` shows
  `fabric_erisc_router.cpp` stuck in `WriteTransactionIdTracker::transaction_flushed`.
* Failing command:
  `timeout 420 python doc/vllm_integration/bench/adapter_probe.py --layers 0,3 --kv-token-budget 262144 --out probe_x.json`
  — deterministic, 4 reproductions across device resets, and reproducing on the
  shipped 52-layer target as well (`logs/probe_full.log`).

**The starting report's headline localisation was wrong**, and correcting it was
the whole fix. AUTODEBUG said the hang was "on the first/second traced decode
step of a multi-slot section". The probe prints *nothing at all* between the last
`prompt_len=4097 -> [...]` line and the end of the run: the three-slot decode
loop, and the host-sampling compatibility step that follows it, are both silent.
"The log goes silent after the three all-gather pairs" therefore covered the
whole tail of the probe, not just the multi-slot decode.

## Hypothesis Experiments

### H1 — the async read records its event on a queue the read did not use, so the wait returns early and the next replay overwrites the token buffer mid-read

* **Experiment**: `--read-mode sync`, i.e. `read_from_device=True` (a blocking
  read inside `decode_forward`) instead of the `cpu(blocking=False)` +
  `record_event` + `event_synchronize` split.
  `timeout 300 python .../adapter_probe.py --layers 0,3 --kv-token-budget 262144 --read-mode sync --out probe_syncread.json`
* **Result**: `RC=124` — hangs at the identical point (`logs/probe_syncread.log`).
* **Verdict**: **refuted.** The read path is not involved. `cpu(blocking=False)`
  + `record_event(mesh, 0)` is exonerated, so the identical pattern in
  `models/tt_transformers/tt/generator.py` is not implicated either.

### H2 — instrument: which step actually hangs, and what do the device inputs hold going into it?

* **Experiment**: added `--verbose-steps` to `adapter_probe.py`: a flushed marker
  around every decode step, plus a readback of all four persistent trace inputs
  (`current_pos`, `rope_pos_ids`, `tokens`, `page_table`) before each one.
  `timeout 300 python .../adapter_probe.py --layers 0,3 --kv-token-budget 262144 --verbose-steps --out probe_verbose.json`
* **Result** (`logs/probe_verbose.log`): **all 16 multi-slot decode steps
  complete**, with entirely sane inputs throughout —
  `multi step=15 before: current_pos=[111, 145, 76, -1, -1, -1] rope=[111, 145, 76, 15, 15, 15] page_table[:4,:6]=[[74,75,76,77,78,0], [79,80,81,82,83,0], [84,85,86,87,88,0], [0,0,0,0,0,0]]`.
  The run then hung *after* `multi step=15 done`, in the host-sampling
  compatibility step.
* **Verdict**: **verified, and it relocates the bug.** The multi-slot traced
  decode section was never the failure. Everything the four bisect matrices
  measured was measuring the wrong region.

### H3 — the hang is an out-of-range `current_pos` reaching the paged cache ops at the host-sampling step

The probe deliberately corrupts its host position tensor to `-7` after each
steady decode step, to prove the stale-input contract. Every traced,
device-sampled step ignores it (`refresh_inputs=False`), which is why the
corruption is inert for the whole multi-slot loop. The final host-sampling step
passes `sampling_params=None`, so `sample_on_device=False`, so
`refresh_inputs=True` **by contract** — and `-7` is staged to the device.

`writer_paged_fused_update_cache_interleaved_start_id.cpp:82` skips an inactive
row by comparing the index against `(uint32_t)-1` *exactly*. `-7` becomes
`0xFFFFFFF9`, so `virtual_block_id = update_idx / block_size = 67108863` reads
far past the page-table circular buffer, and `physical_block_id` is whatever
garbage that lands on. The op then issues a NOC transaction to an arbitrary
address which never retires — precisely `check_noc_status`'s unretired NOC reads
and the fabric router stuck in `transaction_flushed`.

* **Experiment**: add the guard to `MuseGlimmerModel.positions_to_device` (the
  single funnel for `current_pos`/`rope_pos_ids`) and run the reproducer with the
  corruption *still* fed to the host-sampling step, so the guard has to name it:
  `timeout 300 python .../adapter_probe.py --layers 0,3 --kv-token-budget 262144 --decode-steps 2 --keep-stale-for-host-sampling` (evidence is `logs/probe_guard.log`; the run exits 1 at the guard *before* writing its JSON, by design, so no `probe_guard.json` is produced)
* **Result** (`logs/probe_guard.log`), `RC=1` in ~60 s with no hang and no reset:
  ```
  [probe] multi decode section done; host-sampling step submit
  ValueError: decode start_pos must be in [0, 131072) or exactly -1 for an inactive
  slot; row(s) [0, 1, 2] carry [-7, -7, -7]. ...
  ```
* **Verdict**: **verified.** The corrupted value reaches the device at exactly
  one step, and it is the host-sampling step.

This also retro-explains every earlier observation:

| observation | explanation |
|---|---|
| `--no-stale-inputs` passes | positions are always legal, so the host-sampling step gets legal positions |
| `--decode-steps 2` still hangs | the host-sampling step runs regardless of step count |
| all four `multi_slot_bisect.py` arms pass | that harness never writes `-7` and has no host-sampling step |
| the param-restaging fix changed nothing | unrelated to the cause (kept for its own reasons) |
| the 52-layer target hangs identically | same probe, same final step |
| deterministic across resets | it is a value bug, not a race |

* **Fix**:
  * `tt/model.py` `positions_to_device` — refuse any position that is neither
    `-1` nor in `[0, max_seq_len)`, with a message that names the offending rows
    and values and explains the device consequence. Also refuse more rows than
    the decode batch.
  * `doc/vllm_integration/bench/adapter_probe.py` — the host-sampling step is not
    covered by the stale-input rule (it samples on host, so the adapter restages
    from the caller by contract), so it is now handed the real positions and
    tokens. `--keep-stale-for-host-sampling` retains the old behaviour to
    reproduce the original hang and prove the guard.
  * `tests/test_full_model.py` —
    `test_out_of_range_start_pos_is_rejected_instead_of_hanging_the_mesh`, a
    durable regression next to the existing `-1`-sentinel test.

* **Verification**:
  * reduced target, full default arm (16 steps, stale inputs, async read):
    `timeout 300 python .../adapter_probe.py --layers 0,3 --kv-token-budget 262144 --out probe_fixed.json`
    → `RC=0`, `"status": "ok"`, `rows_are_distinct: true`, host-sampling logits
    finite. Single-slot tokens are **byte-identical** to the pre-fix logs
    (`56562, 125391, 164576, ...`), so nothing about what the model computes moved.
  * The stage contract is intact and measured, not asserted — over 16 multi-slot
    decode steps: `trace_replays: 16`, `token_refreshes: 1`,
    `position_refreshes: 1`, `page_table_refreshes: 1`, `synchronizations: 0`,
    `sampling_param_reuses: 15`. No drain was added anywhere, the traced decode
    path and on-device split sampling are untouched, `supports_async_decode`
    stays `True`, and a device-sampled step with an unchanged layout still reads
    no host token/position state.
  * shipped 52-layer target: `timeout 1500 python .../adapter_probe.py --out probe_full_fixed.json`
    → `RC=0`, `"status": "ok"` (`logs/probe_full_fixed.log`, `probe_full_fixed.json`).
    Built in 174.8 s, the real serving pool bound (52 x 2 x `(16416, 1, 64, 128)`
    BFLOAT8_B, 14.86 GB/device, 1,050,624 tokens, `binds_the_allocated_buffers`
    true for every layer), both traces captured, three concurrent slots distinct,
    `host_sampling` logits `[32, 1, 202048]` finite, and the same counters as the
    reduced target (`trace_replays: 16`, one refresh of each input,
    `synchronizations: 0`, `sampling_param_reuses: 15`) with
    `supports_async_decode: true`.
  * nearby correctness, the decode position/staging neighbourhood:
    ```
    timeout 900 python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_full_model.py \
      -k "minus_one_sentinel or out_of_range_start_pos or page_table_is_normalised or \
          page_table_out_of_range or caller_driven_decode or steady_state_decode or \
          split_sampling_feeds or host_sampling_mode_runs" -q
    ```
    → `8 passed` in 21.4 s.

## Final Status

**Fixed.** The hang was an illegal decode position (`-7`, the probe's deliberate
stale-input marker) reaching `paged_update_cache` /
`paged_scaled_dot_product_attention_decode` at the one step whose contract makes
host state authoritative — the host-sampling compatibility step — where it is
read as a huge unsigned index and hangs the mesh with an unretired NOC
transaction. The port now refuses it as a caller error; the probe no longer
feeds an illegal position to a path that must consume it.

Commands that prove the final state:

```
timeout 300 python models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/bench/adapter_probe.py \
    --layers 0,3 --kv-token-budget 262144 \
    --out models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/probe_fixed.json
timeout 1500 python models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/bench/adapter_probe.py \
    --out models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/probe_full_fixed.json
timeout 300 python models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/bench/adapter_probe.py \
    --layers 0,3 --kv-token-budget 262144 --decode-steps 2 --keep-stale-for-host-sampling \
    --out models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/probe_guard.json   # expects RC=1, not a hang
    # NB: the guard fires before the JSON is written, so the artifact is logs/probe_guard.log, not probe_guard.json.
```

## Remaining Risks / Follow-Up

* The `-1`-only skip sentinel is a property of the shared paged-cache kernels,
  not of this port. Any other model driving `paged_update_cache` with a
  caller-supplied position has the same mesh-hang failure mode with no in-band
  error. Worth raising upstream; this port's guard only protects this port.
* AUTODEBUG's bisect matrices 1-4 were all measuring the multi-slot decode
  region, which H2 shows was never failing. They remain valid evidence that that
  region is sound; they are not evidence about the actual bug.
* Not investigated, because it turned out not to be on the failing path: whether
  the eager prefill's async collectives need a drain before a traced decode
  replay reuses the seven shared `MultichipDecoder._ccl_semaphores`. Nothing
  observed requires it — the multi-slot decode ran clean immediately after a
  three-user eager prefill in every passing run — but the sharing is real and
  `_warm_persistent_buffers` already documents one case where it needed an
  explicit barrier.

---

# AutoFix round 2 — the three correctness-class sampling failures

Starting evidence: `readiness_vllm/sampling_tests.log` (62 passed, 10 failed,
1 skipped) from the `--sampling-profile full` run against the live server. Seven
failures are the reproducibility-only class; the three below are not, and each
needed either a fix or proof that it is not a serving defect.

## H1 — presence penalty has no effect on the served output

`TestPresencePenalty::test_different_presence_penalties` sweeps
`presence_penalty` over `[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]` across 8
concurrent greedy requests on the prompt `"a b c a b c a b c"` and asserts at
least 2 unique outputs. All 8 are identical.

**Discriminating evidence already in the same file:** `TestRepetitionPenalty`
(both tests) and `TestFrequencyPenalty` (both tests) **pass**. So the on-device
penalty path runs, `_penalties_active` is set, prompt/output token history is
tracked and `TTPenalties.apply` executes — only *presence* is invisible. Presence
is binary (one `-penalty` on any token that has appeared, clamped by vLLM to 2.0)
where frequency scales with the occurrence count, which on a 40-token repetition
of "a b c" reaches ~13x that.

**The measurement trap, and why the obvious probe is worthless here.** Asking for
logprobs to watch the shift does not work on this mesh: the plugin routes *any*
logprobs request to host sampling (`check_perform_device_sampling` requires
`num_devices in (8, 32)` for device logprobs; this mesh has 4), and vLLM's host
sampler returns `raw_logprobs`, computed *before* penalties by design. A first
probe did exactly this and measured a relative shift of `0.0` over 39 steps
(`sampling_failure_probe.json -> item1_presence_penalty.measured_logit_shift`).
That number says nothing about the device path and must not be read as evidence
of a defect.

**Experiment** (`bench/presence_flip_probe.py` -> `presence_flip_probe.json`),
two phases with different sampling paths on purpose:

* *Phase A — host path, `logprobs=20`, `presence_penalty=0`.* Reads the true
  pre-penalty logit margin at every step: the gap between the greedy winner and
  the best challenger that has already appeared (the only tokens a presence
  penalty can move). For the test's own prompt the minimum such gap over the 40 scored steps
  is **3.0**, and the winner-fresh-to-best-already-seen gap is **4.5** — both
  above the 2.0 clamp, and the second is what rules out the *negative* penalties
  the failing test also sweeps. **No legal presence penalty can flip that prompt's
  argmax.** (Both figures are in `sampling_failure_probe.json ->
  item1_presence_penalty.greedy_flip_margins`, not in `presence_flip_probe.json`,
  whose phase A records only the already-emitted-winner direction.)
  Four other prompts were swept to find one with a small margin.
* *Phase B — device path, greedy, no logprobs anywhere* (so the request really is
  sampled on device by the traced split sampler). Prompt `"She opened the door and
  saw"`, whose Phase-A margin at step 19 is **0.5**. Sweeping presence_penalty:

  ```text
  0.0, 0.125, 0.25, 0.375  -> identical to the zero-penalty output
  0.475                    -> output changes, first divergence at char 76
  0.525, 0.625, 0.75, 1.0, 2.0 -> same changed output
  ```

**Verdict: not a serving defect — proven, not asserted.** The on-device presence
penalty reaches the logits: the output flips at **0.475** against a measured
margin of **0.5**, i.e. the observed threshold matches the predicted one to within
one quantization step of the BFP8 logits, and stays flipped for every larger
penalty. `test_different_presence_penalties` fails because its prompt's margin
(3.0) exceeds the maximum penalty the API allows (2.0), not because the penalty is
missing. **No fix; no code changed.**

## H2 — `test_allowed_token_ids` returns empty text

Five concurrent requests with `allowed_token_ids` `[1,2,3] [4,5,6] [7,8,9]
[10,11,12] [13,14,15]`; request 0 returns `''` and the test asserts non-empty text.

**Experiment** (`bench/sampling_failure_probe.py` -> `sampling_failure_probe.json
-> item2_allowed_token_ids`): send those exact five requests and record
`completion_tokens`, `finish_reason` and the emitted token ids rather than only
the text.

```text
req 0  ids [1,2,3]     text ''          completion_tokens 10  emitted [1,2,2,1,2,2,1,3,1,3]   all allowed: true
req 1  ids [4,5,6]     text ''          completion_tokens 10  emitted [4,6,5,5,4,4,6,4,4,4]   all allowed: true
req 2  ids [7,8,9]     text ''          completion_tokens 10  emitted [8,8,7,8,8,8,7,7,8,7]   all allowed: true
req 3  ids [10,11,12]  text ''          completion_tokens 10  emitted [12,12,12,11,11,10,...] all allowed: true
req 4  ids [13,14,15]  text '#!#!#!#!#!' completion_tokens 10 emitted [15,13,15,13,...]       all allowed: true
```

**Verdict: not a serving defect — proven.** Every request generated its full 10
tokens (`finish_reason: length`) and **every emitted id is inside its allowed
set**, so the constraint itself works. Ids 1-12 are byte-fallback tokens that each
decode to U+FFFD; an incomplete UTF-8 byte sequence is buffered by the
detokenizer, so empty *text* is the correct output of forcing only those ids.
Request 4, whose ids `[13,14,15]` are the printable `!"#`, returns 10 characters
through the identical path. Note this also exercises the host-sampling
compatibility mode, since `allowed_token_ids` forces it. **No fix; no code changed.**

## H3 — the `--async-scheduling` overlap validation

`supports_async_decode=True` was implemented and unit-proven, but no server run
had actually passed `--async-scheduling`, so the overlap path — the one where
vLLM may build and submit decode step N+1 before sampled token N has reached host
scheduler state — was never exercised end to end. That is the case the adapter's
stale-input rule exists for, and the case whose failure mode is doubled subwords
and repeated control tokens.

**Experiment** (`bench/run_async_overlap.sh`): launch the server with
`--additional-server-args=--async-scheduling` and the identical TT config,
artifacts into `doc/vllm_integration/async_overlap/` so they cannot overwrite the
committed non-overlap evidence; assert the plugin did not refuse the capability;
re-run the prompt-correct qualitative arm on the same pinned token ids; run the
degenerate-output check over the overlap artifacts; and diff the overlap
completions against the non-overlap ones.

One shared-infra bug had to be fixed first: `serve.sh` passed
`--additional-server-args "$EXTRA_SERVER_ARGS"`, and argparse treats a separate
value beginning with `--` as another option
(`error: argument --additional-server-args: expected one argument`). It now uses
the `--flag=value` form, with a comment so it does not get "cleaned up" back.

**Result** (`logs/async_overlap.log`, `async_overlap/overlap_vs_non_overlap.json`):

```text
server ready 11:20:42
ASYNC_ACCEPTED: no 'Disabling async scheduling' in the server log
STEP async_qualitative rc=0
STEP async_degenerate  rc=0     (no degenerate output)

overlap vs non-overlap, 6 pinned prompts:
  identical completions          6 / 6
  max adjacent token duplication 0.0000   (critical threshold 0.10)
  control tokens leaked to text  none
```

**Verdict: verified.** The plugin accepted the capability rather than silently
disabling it, and under real overlap the served text is **byte-identical** to the
non-overlap arm on every prompt, with zero adjacent-token duplication and no
control-token repetition. If the adapter were reading stale host token or position
state on an overlapped step, this is exactly the comparison that would diverge.
**No fix; no code changed** beyond the `serve.sh` argparse quoting.


### P1 follow-up from the stage review — the phase-B verdict field was stale

The stage review found `presence_flip_probe.json` carrying
`presence_penalty_reaches_device_logits: false` while this report cited it as
proof of the opposite. The measurements were never in doubt — the ladder shows
`[0.0 … 0.375]` unchanged and `[0.475 … 2.0]` changed against a 0.5 margin — but
the file had been written by an earlier revision of the probe whose verdict rule
differed from the one it printed, and it lacked the `monotone_step` and
`largest_penalty_with_no_change` fields the shipped script emits. Cited-but-
contradicted is exactly the failure mode a review should catch.

Fixed by making the rule a single pure function, `phase_b_verdict(runs, margin)`,
used both by the live probe and by a new `--recompute-verdict` mode that
re-derives the fields from a committed ladder offline (no server, no device) and
rewrites **only** the derived fields:

```text
$ presence_flip_probe.py --recompute-verdict presence_flip_probe.json
recomputed: False -> True
  observed_flip_threshold = 0.475
  largest_penalty_with_no_change = 0.375
  monotone_step = True
  tolerance = 0.15
  within_tolerance = True
  presence_penalty_reaches_device_logits = True
```

The artifact now records `verdict_provenance` naming the recompute and the value
it replaced. The shipped script exits 0 on it.
