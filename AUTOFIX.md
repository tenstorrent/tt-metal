# AutoFix: Mistral Small 24B vLLM integration

## Scope

Repair the shared vLLM adapter and TT plugin path for
`mistralai/Mistral-Small-24B-Instruct-2501`, preserving the full-model
generator's split on-device sampling trace and vLLM-owned paged KV cache.

## Proven fixes

- Requests with sampling features outside the device sampler contract (for
  example `top_k > 32`, penalties, seeds, and full-logit logprobs) are routed
  through the explicit optional host-compatibility path. Production greedy and
  supported top-k/top-p sampling remain on the canonical device trace.
- Decode reset, slot-remap, current-position, page-table, and sampled-token
  state are forwarded through both device and compatibility paths. Fresh
  prefill destinations are treated as remap destinations, not old source rows.
- A prefill boundary drains and releases an existing decode trace before TTNN
  can reuse allocations. Active physical KV pages are checked for accidental
  request aliasing.
- Persistent decode collectives use alternating workspace/semaphore pairs.
  The optional host-compatibility path uses the default synchronous collective;
  it does not replace the production traced path.
- Fixed-shape prefill previously let inactive physical rows reach
  `paged_fill_cache`. The active request count is now threaded through the
  full-model prefill stack, and only active rows may mutate paged KV while the
  physical tensor shape remains fixed. This is a proven cache-isolation fix,
  but the long sampling failure still reproduced afterward, so it was not the
  sole history-dependent cause.
- Optional host-compatibility logits are materialized and sampled
  synchronously. Only production on-device sampling uses the async device
  output wrapper; this preserves async traced decode for performance while
  preventing compatibility-generator/layout state from crossing admission
  boundaries.

## Refuted hypotheses

- Merely fencing persistent collectives did not eliminate the long-suite
  failures.
- Alternating persistent collective workspaces fixed isolated reuse hazards
  but did not eliminate history-dependent serving failures by itself.
- Forcing the optional host decode path to the default collective did not
  eliminate the failures.
- Trace release at prefill boundaries was necessary for safe allocator reuse
  but was not sufficient; failures still reproduced afterward.
- Excluding inactive prefill KV writes alone did not eliminate the long-suite
  failures.
- Inactive decode cache updates are not the cause: the TT paged-update kernels
  explicitly treat an index of `-1` as a skipped user.
- Disabling global async scheduling did not change the eight-failure pattern.
- Forcing a device-wide fence before every admission prefill did not change the
  pattern either. The eight failed nodes passed 8/8 when immediately rerun on
  the same engine, proving a repeatable test-order/transition defect rather
  than durable engine corruption, but the full 73-test profile still failed.

## Final AutoFix status

Resolved in the follow-up loop. After the cache/trace lifecycle fixes, the only
remaining failure was unseeded stochastic top-k variety across otherwise
identical request batches: 7/8 request pairs repeated because the model's
traced `Sampling1D` deliberately replays fixed slot seeds. In the explicit
`MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT=1` mode only, stochastic batches
now use vLLM's persistent host RNG. The normal environment-off production path
continues to use traced on-device split sampling.

The original ordered full sampling gate was rerun, not replaced by isolated
retries: **72 passed, 1 expected skip in 504.80 seconds**. The focused final
adapter/tokenizer/routing suite passed 43/43. Production qualitative and
benchmark evidence remains from the environment-off on-device path.

## Evidence

- Static adapter and plugin tests cover exact cache identity, async stale-token
  behavior, slot remap, current position, page tables, prefill trace release,
  device-sampling limits, and host state forwarding.
- `test_prefill_threads_active_rows_to_prevent_padded_kv_writes` proves a
  three-request prefill keeps the 32-row physical tensor while passing
  `active_batch=3` into model prefill.
- Final server, sampling, qualitative, and benchmark artifacts are stored under
  `models/autoports/mistralai_mistral_small_24b_instruct_2501/readiness_vllm/`.
