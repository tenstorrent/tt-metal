# AutoDebug: optimized-decoder review findings

## Scope and provenance

This is an inspection-only investigation of the four requested stage-review
findings. No implementation or test file was changed and no TT hardware command
was run by this investigation.

The required fresh-context AutoDebug runner was attempted first with both
supported backends. The Codex backend could not enter its nested workspace
sandbox because `bubblewrap` was unavailable, and the Claude backend could not
authenticate because its OAuth session had expired. This report therefore uses
the same source/evidence-only constraints in the current context. Existing
hardware logs are treated as observations, not rerun.

## Headline findings

### 1. Definite BFP8 candidate bug: a later prefill can erase earlier users

**Direct evidence**

- The optional BFP8 path allocates a full-size BF16 staging key cache and value
  cache on every call, calls the functional prefill into those initially-zero
  caches, then typecasts and copies both *entire* staging caches over the real
  BFP8 caches
  (`tt/optimized_decoder.py:150-177`).
- Functional prefill only fills the users selected by `user_id + batch_idx`
  (`tt/functional_decoder.py:292-316`). It does not preserve unrelated users in
  a newly allocated staging cache.
- Consequently, a call which fills user B after an earlier call filled user A
  copies zeros from every staging location except B and destroys A's prior
  entries. The existing BFP8 transition log exercises one prefill call followed
  by decode, so it cannot expose this state-lifetime bug
  (`candidates/bfp8_kv_cache_transition.log`).

**Causal chain**

1. A BFP8 cache contains a valid prefix for user A.
2. A later `prefill_forward(..., user_id=B)` allocates all-zero BF16 staging
   caches.
3. Functional prefill populates only B's page-table rows in staging.
4. Lines 175-176 copy the complete staging tensors over the complete persistent
   caches.
5. A's pages become zero. A subsequent decode for A consumes the overwritten
   pages through `paged_scaled_dot_product_attention_decode`
   (`tt/optimized_decoder.py:286-305`) and produces the wrong result.

The default remains BF16 (`tt/optimized_decoder.py:137`), so this is not a bug
in the selected runtime path. It makes the current adapted BFP8 implementation
invalid as a selectable candidate.

**Smallest intervention**

Do not stage the complete cache. Preserve the functional prefill operation
chain and cast each sliced `user_key` and `user_value` to the destination cache
dtype immediately before its corresponding `paged_fill_cache`. Attention must
continue to use the original BF16 `key` and `value` tensors. This confines the
conversion to the newly written logical prefix, preserves other users, avoids
two full-cache allocations and copies, and satisfies the dtype equality
validated in the original failure
(`candidates/bfp8_kv_correctness.log`).

### 2. BFP4 geometry selection reused BFP8 sweep evidence

**Direct evidence**

- The selected BFP4/LoFi path has end-to-end correctness at B1/B32 and real
  weights:
  `attention_bfp4_correctness.log`, `mlp_bfp4_correctness.log`,
  `down_bfp4_correctness.log`, and `all_bfp4_real_weight.log`.
- It has final selected-geometry performance at B1/B32:
  `all_bfp4_lofi_perf.log` reports 0.568368/0.751526 ms.
- The role-specific geometry logs precede the BFP4 selection and their captured
  tensor diagnostics identify `BFLOAT8_B` weights. This includes
  `ds_qkv_bw6.log`, `ds_o_proj_bw12.log`, `ds_gate_up_bw6.log`,
  `ds_down_bw8.log`, `ds_down_bw16.log`, and `ds_down_bw32.log`.
- The present defaults are QKV 4, output 12, gate/up 4, and down 8
  (`tt/optimized_decoder.py:28-35,89-97`).

**Inference**

The BFP8 sweep is a useful seed but is not complete evidence for BFP4. BFP4
changes weight tile storage and circular-buffer pressure. In particular, a
BFP8 L1 failure does not prove the same geometry is illegal with BFP4. Candidate
ranking can also change even when both geometries are legal. The final BFP4 PCC
tests validate only the selected cumulative configuration; they do not validate
each alternative geometry at both batches.

**Remaining matrix**

The legal `in0_block_w` divisors implied by the source are:

| Projection | K tiles/core | Selected | Unresolved BFP4 contrasts |
|---|---:|---:|---|
| QKV | 12 | 4 | 1, 2, 3, 6, 12 |
| output | 12 | 12 | 1, 2, 3, 4, 6 |
| gate/up | 12 | 4 | 1, 2, 3, 6, 12 |
| down | 32 | 8 | 1, 2, 4, 16, 32 |

An exhaustive rerun is not necessarily required if dominated widths are
screened with component timing, but every retained/rejected BFP4 candidate
needs B1 and B32 correctness plus performance or an exact BFP4 resource-limit
failure. A first API/L1 error should be followed by the next legal program or
memory configuration before rejection.

### 3. Packed versus split gate/up now has performance data, but not a decisive A/B

**Direct evidence**

- The implementation contains independently executable packed and split paths
  controlled by `PHI_OPT_SPLIT_GATE_UP`
  (`tt/optimized_decoder.py:40-48,201-255`).
- `bfp4_lofi_split_gate_up_perf.log` is complete for all four workloads:
  prefill B1/B32 is 1.459376/31.439613 ms and traced decode B1/B32 is
  0.567862/0.749801 ms.
- The packed BFP4/LoFi candidate in `all_bfp4_lofi_perf.log` reports decode
  B1/B32 0.568368/0.751526 ms. The final packed run reports
  0.5678/0.7484 ms in `README.md`.
- No split-path PCC result is present in the candidate logs, and the small
  differences between packed and split decode are comparable to ordinary
  run-to-run variation visible across the logs.

**Inference**

The earlier claim that packed gate/up was never measured is now obsolete.
However, the available measurements are separate process runs rather than a
paired alternating comparison, and they do not establish a statistically
credible winner. Split is nominally faster in one comparison while packed is
nominally faster in another. Packed remains the topology-minimal choice (one
same-input matmul rather than two), but that argument is not measured evidence.

The split tensors are slices of the already quantized packed BFP4 tensor
(`tt/optimized_decoder.py:40-48`), so a split PCC test also checks the actual
dtype/layout path rather than a different host-side quantization policy.

### 4. The BF16 prefill-to-decode consumption finding is closed in source

**Direct evidence**

- `test_optimized_decode_consumes_non_aligned_paged_prefill` now prefills a
  random 33-token prompt, decodes at logical position 33, and compares
  optimized output against the functional path for both batch 1 and batch 32
  (`tests/test_optimized_decoder.py:245-295`).
- It uses a permuted page table (`line 256`) and the decode path consumes the
  cache through paged SDPA (`tt/optimized_decoder.py:286-305`).
- `correctness/cache_transition_bf16.log` records the selected BF16 test result,
  while `candidates/bfp8_kv_cache_transition.log` records PCC
  0.99998846/0.99998821 for the optional BFP8 single-call transition.

**Conclusion**

For the selected BF16 path, this is meaningful evidence that non-aligned
prefill-populated cache state affects subsequent decode correctly at both
required batches. A zero-prefix decode test alone would not prove this, but the
new random-prefix transition does.

The BFP8 result proves only the single-call transition. It does not refute the
multi-call overwrite bug above, and the BFP8 mode is enabled only by an
environment variable rather than a permanently parametrized regression test.

## Focused experiments for an AutoFix pass

Run hardware experiments serially under the repository's device-usage rules.
Each experiment should save its exact environment and command with the result.

### EXP-1: prove and then fix BFP8 multi-call ownership

1. Construct one optimized decoder with `PHI_OPT_KV_DTYPE=bfp8`, `batch=1`, a
   permuted page table, and capacity for at least two users.
2. Prefill random non-aligned length 33 for user 0.
3. Snapshot user 0's referenced physical pages, or decode user 0 and retain the
   output.
4. Prefill a different random length 31 or 33 for `user_id=1`.
5. Assert user 0's physical pages are bitwise unchanged. Then decode both users
   and compare to independent functional BF16-cache decoders at PCC >= the
   functional bar.
6. Run before the fix to demonstrate the overwrite and after the per-slice-cast
   fix to prove it is removed.
7. Repeat with a batch-32 one-call prefill-to-decode transition at logical
   position 33, plus a batch-1 transition, so the stage matrix remains covered.
8. Run a watcher-clean repetition.

Also measure cache allocation bytes and peak live device memory. The current
full BF16 staging method temporarily adds two complete BF16 caches, so any
capacity claim based only on persistent BFP8 cache size is misleading.

### EXP-2: paired BFP8 cache performance

After EXP-1 passes, compare BF16 and corrected BFP8 caches with the same
pre-created decoder resources, input seeds, page table, warmups, and alternating
sample order:

- warmed prefill S=128 at B1 and B32;
- traced decode C=128 at B1 and B32;
- a cache-sensitive longer context supported by the test budget.

Report mean, median, minimum, and paired deltas. Reject BFP8 if it does not
improve the selected objective or if its conversion/memory cost defeats its
capacity benefit. If selected, update the context contract only from measured
capacity evidence.

### EXP-3: precision-locked BFP4 geometry sweep

Hold all weights BFP4, math fidelity LoFi, cache BF16, packed gate/up, input
seeds, and trace order fixed. Sweep the legal widths in the table above, one
role at a time from the selected cumulative configuration.

For every candidate:

- run optimized-versus-functional decode PCC at B1 and B32;
- run alternating paired traced decode at B1 and B32;
- record exact BFP4 L1 failures;
- retry a failed candidate with another applicable legal memory/program
  configuration before declaring the width illegal.

Then test the best role-local choices cumulatively because L1 lifetime and
whole-layer scheduling can differ from isolated ranking. This experiment can
falsify the current assumption that BFP8 geometry ranking transfers to BFP4.

### EXP-4: paired packed/split adjudication

Use the selected BFP4 geometry and BF16 cache. First run split-path
optimized-versus-functional PCC at B1 and B32. Then capture packed and split
traces in the same process and alternate replay order for at least 100 paired
samples at each batch. Ensure environment-dependent branch selection occurs
before each trace capture and does not change during replay.

Record per-projection device time from Tracy in addition to end-to-end time.
Select split only if the paired confidence interval shows a repeatable
end-to-end win at B1 without B32 regression; otherwise keep packed for its
smaller topology. The present unpaired ~0.1-0.2% differences are insufficient.

## Suggested AutoFix order

1. Replace the full-cache BFP8 staging/copy path with per-user-slice casts and
   add the multi-call preservation regression.
2. Run EXP-1 and EXP-2; either select BFP8 with complete evidence or remove the
   invalid optional path and document its measured rejection.
3. Run the precision-locked BFP4 geometry matrix and update selected defaults.
4. Run the paired packed/split adjudication.
5. Retain the existing BF16 non-aligned transition test permanently; add an
   equivalent BFP8 parametrization only if BFP8 remains a supported option.

## Claims explicitly not made

- Existing logs do not prove that BFP8 cache is unsafe in TTNN generally. They
  prove that this full-cache staging adaptation has an ownership bug.
- Existing logs do not prove that the current BFP4 geometry is slower; they
  show that its alternative-geometry evidence was collected with BFP8.
- Existing logs do not prove split gate/up is faster or slower; the nominal
  deltas are unpaired and inconsistent across runs.
- The selected BF16 cache path is not implicated by the BFP8 staging bug.
