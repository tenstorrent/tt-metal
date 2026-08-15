# AutoDebug — vLLM serving decode hangs on the multi-request step

## Symptom

`doc/vllm_integration/bench/adapter_probe.py --layers 0,3` (the reduced 2-layer
serving target: layer 0 sliding + layer 3 full) completes:

* KV-cache allocation and adoption (vLLM-owned, 4128 blocks);
* prefill + decode warmup, both traces captured;
* three batch-1 requests (prompt lengths 128, 37, 4097), 16 traced decode steps
  each, byte-identical tokens across processes and across a device reset;

and then **hangs** in the multi-request section (three concurrent slots at prompt
lengths 96 / 130 / 61). The process never returns; it has to be killed and the
devices reset.

Reproduced twice, with a `tt-smi -r` and a passing mesh smoke test in between, at
the same point with byte-identical preceding output. Not a transient fabric
fault.

## Triage

`doc/vllm_integration/triage/tt-triage.txt` (captured on the live hang):

* `dump_running_operations`: `PagedUpdateCacheDeviceOperation (trace id: 0)`
  RUNNING on 12 cores (3 per device across all four devices), inputs
  `cache [4128, 1, 64, 128] BFLOAT8_B`, `k [1, 32, 1, 128] BFLOAT16 HEIGHT_SHARDED L1`,
  `update_idxs [32] INT32`, `page_table [32, 2048] INT32`. Previous op
  `RotaryEmbeddingHfDeviceOperation` completed; the queued tail (`SdpaDecode`,
  `NLPConcatHeadsDecode`, `Matmul`, `ReduceScatter`, ...) never starts.
  So the stall is inside the **decode trace**, in layer 0's attention, at the
  paged cache write.
* `check_noc_status`: fail on the **ethernet** cores of all four devices —
  `erisc1 NOC0 NIU_MST_RD_RESP_RECEIVED noc_reads_num_issued 40135 0` and
  similar, i.e. the fabric routers hold unretired NOC reads.
* `dump_callstacks`: `fabric_erisc_router.cpp` stuck in
  `WriteTransactionIdTracker::transaction_flushed` ->
  `run_receiver_channel_step_impl`.
* `check_binary_integrity`, `check_core_magic`: pass.

## Bisect 1 — is it three prefills, or three decode rows?

`doc/vllm_integration/bench/multi_slot_bisect.py`, one arm per process, reset
after any non-zero exit (`logs/bisect.log`, `bisect_*.json`):

| arm | prefill rows | decode rows | result |
|---|---|---|---|
| `prefill3_decode1` | 3 (one batched call) | 1 | **pass** |
| `prefill1_decode3` | 1 | 3 | **pass** |
| `prefill3_drain` | 3 (separate calls + drain) | 3 | **pass** |
| `prefill3_decode3` | 3 (one batched call) | 3 | **pass** |

All four pass, including the exact prefill/decode shape that hangs in the probe.
So neither batched multi-row prefill nor multi-row traced decode is sufficient on
its own, and the failing combination is reproducible only with the probe's
preceding batch-1 requests.

## Bisect 2 — what does the probe add?

`doc/vllm_integration/bench/run_probe_arms.sh` (`logs/probe_arms.log`):

| arm | change | result |
|---|---|---|
| `nostale` | `--no-stale-inputs`: the probe writes *correct* values into the host token/position tensors between steady steps instead of deliberately wrong ones | **pass**, end to end, including the 3-slot section, host-sampling fallback and all counters |
| `short2` | `--decode-steps 2` (stale inputs kept) | see `logs/probe_short2.log` |
| `onlyfirst` | `--prompt-lens 128` (stale inputs kept) | **never completed** — the arm was killed before it ran, so there is no result and no log. Superseded by the root cause below, which makes the prefix irrelevant. |

`nostale` is the important one: the same code path, the same
`refresh_inputs=False` decision on every steady step, the same trace replays, the
same three concurrent slots — and it completes.

## Why that is surprising, and where the hypothesis has to go

`MuseGlimmerGenerator.decode_forward(refresh_inputs=False)` does not read the
caller's `tokens` or `start_pos` at all: it computes `token_list`/`positions`,
uses only `len(token_list)` for the output row count, and stages nothing but the
page table (which is memoised and unchanged). The adapter passes `start_pos` on
to `apply_decode_sampling_state`, where it selects `active_slots` and, for
explicit request seeds, aligns seed counters — but this run has
`seed=[None] * rows`, so `SeedManager._seed_active` is False and
`get_new_values([])` and `get_new_values([0, 1, 2])` take the identical branch
and issue the identical device work.

So on inspection the `stale` and `nostale` arms should be **device-identical**,
and yet one hangs and one does not. Either

1. that reading is wrong and some path does consume the corrupted host values
   (candidates: the `self._staged_positions = positions` aliasing of the
   caller's tensor, the sampler's parameter/seed staging, or the page-table
   memo), or
2. the hang is a genuine race in the multi-row traced-decode / async-read path
   that the two arms hit with different probability because they do different
   amounts of host work between replays (the async split's
   `tensor.cpu(blocking=False)` reads the *persistent* token buffer that the
   next trace replay overwrites, and its ordering guarantee is the cq0
   enqueue order plus the recorded event).

Hypothesis 2 is the one the fabric-router evidence points at: an op stalled with
the ethernet routers mid-transaction is a CCL/fabric ordering symptom, not a
wrong-index symptom, and `PagedUpdateCache` is simply the first op that cannot
retire behind it.

## Bisect 3 — determinism, and the first refuted fix

* `stale_repeat1` (`logs/probe_stale_repeat1.log`): the unmodified stale arm, run a
  third time after another reset. **Hangs**, same place. The hang is deterministic,
  not probabilistic, so "the two arms differ only in host timing" is not by itself
  an explanation — but it does not refute a race with a deterministic trigger.
* `short2` (`--decode-steps 2`, `logs/probe_short2.log`): **hangs**. The failure is
  not step-count dependent; two decode steps in the three-slot section are enough.
  So it is the *first or second* multi-row step, not accumulation.
* **Refuted fix — per-step sampling-parameter restaging.** Serving called
  `SamplingGenerator.apply_decode_state` on every decode step, which re-stages
  five host-to-device copies (`top_k`, `top_p`, `temperature`, the greedy
  tie-break column, the seed row) plus `TTPenalties.reset_params`' four, into
  tensors the sampling trace reads, while two `blocking=False` replays are in
  flight. Hypothesis: that write races the live replay. `apply_decode_sampling_state`
  now compares the formatted parameters against the last applied set and skips the
  restage when nothing changed (`probe_paramfix`). **The hang is unchanged**, so the
  hypothesis is refuted as the *cause*. The change is kept anyway: it removes real
  per-token host work from the steady decode path, which the stage contract asks
  for, and it is now counted (`serving_counters.sampling_param_refreshes` /
  `sampling_param_reuses`) rather than asserted.

## Bisect 4 — is the reduced target itself the trigger?

The reduced target is two layers. Its traced decode step is ~0.5 ms against the
shipped 52-layer model's ~23 ms, so the host can submit the next step roughly
50x sooner relative to device completion than it ever can on the real model.
Every hypothesis left standing is an ordering/rate hypothesis, which makes
"reduced target only" a live and cheap possibility that has to be settled before
any of this is called a serving bug. `probe_full` runs the identical probe on all
52 layers (`logs/probe_full.log`). That arm hung, so it never reached its JSON
write and there is no `probe_full.json`; the log is the whole of its evidence. The
JSON that does exist, `probe_full_fixed.json`, is the same arm re-run after the
position guard landed.

**Result: the shipped 52-layer target hangs in exactly the same place**
(`logs/probe_full.log`). It builds in 161.6 s, allocates and adopts the real
serving pool (52 x 2 x `(16416, 1, 64, 128)` BFLOAT8_B, 14.86 GB/device,
1,050,624 tokens), warms up, runs the three batch-1 requests with sensible
in-vocabulary tokens, and then goes silent after the three-slot prefill's
all-gathers with no further output for >150 s. So this is a **serving bug, not a
reduced-target artifact**, and the reduced target is a faithful ~40-second
reproducer of it.

## Next steps

1. Re-run the `stale` arm two more times to establish whether the hang is
   deterministic or probabilistic. A probabilistic hang moves the search to (2).
2. Split the corruption: stale positions only vs stale tokens only. If neither
   alone reproduces, the host values are exonerated and (1) is refuted.
3. If (2): test the async read in isolation — `read_from_device=True`
   (synchronous read inside `decode_forward`) versus the split path, with three
   active rows, and with `ttnn.synchronize_device` between steps, to separate
   "multi-row decode" from "deferred read racing the next replay".
4. Check whether the reduced 2-layer target is a factor: with two layers the
   decode step is ~100x shorter than the 52-layer one, so any race between the
   host's next submission and the device's completion is enormously more likely
   here than on the shipped target. The full-model arm has to be run before this
   is called a serving bug.
