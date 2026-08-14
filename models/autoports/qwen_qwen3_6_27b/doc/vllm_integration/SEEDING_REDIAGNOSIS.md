# Stage 09 blocker re-diagnosed: seeded sampling is a plumbing gap, not a missing device capability

Stage 09 blocked with "no compliant repair path is available without an external
Blackhole `Sampling1D` capability/test change". The capability exists and is
already plumbed. What is missing is binding each request's **requested seed**
into the per-user seeds tensor. This is fixable in this repo.

## What failed

`4 failed, 68 passed, 1 skipped` in the shared vLLM sampling suite:

| test | result |
|---|---|
| `test_specific_seed_reproducible[42]` | **FAILED** |
| `test_specific_seed_reproducible[123]` | PASSED |
| `test_specific_seed_reproducible[999]` | PASSED |
| `test_specific_seed_reproducible[0]` | **FAILED** |
| `test_batch1_seed_reproducible[0]`, `[1]` | PASSED |
| `test_same_seeds_reproduce_across_batches` | **FAILED** |
| `test_mixed_params_batch` | **FAILED** |

Every failure involves batching; every batch-1 variant passes.

## Why

`models/common/sampling/tt_sampling.py`:

```python
# Seeds tensor: one RNG slot per user across all rows.
self.seeds_tt_tensor = ttnn.from_torch(
    torch.arange(total_param_size).to(torch.uint32), ...)
```

and at sampling time:

```python
ttnn.manual_seed(seeds=self.seeds_tt_tensor, user_ids=self.user_ids_tt_tensor, ...)
```

The per-user seeds are `torch.arange(...)` — **the slot index**, fixed at
construction and never updated from the request. So a request asking for
`seed=42` is given the RNG stream belonging to whatever batch slot it occupies.

That predicts exactly what was observed:

- **batch 1** always occupies slot 0, so the stream is stable across runs and the
  batch-1 tests pass;
- **batched / across-batch** runs place the same request in different slots, so
  the stream changes and reproducibility fails;
- **seeds 123 and 999 passing is luck.** The assertion compares generated text;
  on a peaked distribution two different RNG streams frequently draw the same
  token. Passing here is not evidence that seeding works.

`ttnn.manual_seed` already accepts a **per-user seeds tensor**, so the device
side can express per-request seeds today. Nothing external is required.

## The fix, in outline

Bind the vLLM `SamplingParams.seed` of each scheduled request into
`seeds_tt_tensor` at the row its request occupies, refreshing when the batch
composition changes; fall back to the current arange behaviour only for requests
that specify no seed. The plugin already tracks per-request slot identity for
exactly this class of state — `TTModelRunner._req_state_slot` exists because
"evict/re-add and condense move a request's ROW, not its state", and its own
comment lists **seed RNG** alongside GDN recurrent/conv state.

Note this is shared code: the change affects every model using the shared
sampler, so it wants a targeted test at batch > 1 rather than reliance on the
suite's luck-sensitive text comparison.

## This is not Qwen-specific, and Falcon's green result should not be trusted

`tt_sampling.py` is shared. Falcon3-7B-Base's stage 09/10 reported the same
suite as "72 passed, 1 skipped" — the same 73 tests — on the same shared sampler
with the same arange seeds. Its `readiness_vllm/sampling_tests.log` was **never
committed**, so that claim cannot be checked, and given the mechanism above the
most likely explanation is that its more peaked distribution made all four
comparisons pass by luck. Falcon's sampling evidence should be treated as
unverified rather than as a contradiction of this finding.
