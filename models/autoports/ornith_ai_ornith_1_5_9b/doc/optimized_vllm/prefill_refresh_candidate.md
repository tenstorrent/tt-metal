# Repeated prefill refresh candidate

Prepared CPU-only; runtime patch is `prefill_refresh_candidate.patch`. No
hardware result or latency benefit is claimed by this note.

The candidate retains all production 8192 arithmetic, projections, masks and
page semantics. It skips only copies whose destination already contains the
requested value. `NormalizedLinearPrefill` owns independent mask/index tensors
per layer and physical profile. Their inputs depend on consumed `valid_count`,
not token IDs, absolute positions or cache slots. Both writers—the legacy
`refresh` method and serving `refresh_profile`—now call the same refresh helper.
Constructor masks are all ones and history indices describe the full bucket,
so its initial cached count is exactly the allocated content. Changing buckets
does not invalidate another bucket's tensors. Each layer has its own cache.

`PrefillRuntime` owns a shared row-index tensor per physical height. Construction
fills it with `arange(height)`, hence cached absolute zero. `_read` is its only
post-construction writer. Different sources, layers and slots still perform
embedding reads; they can reuse the indices only when height/absolute match.
Boundary restoration and terminal heads also use `_read`, so those updates
participate in the same cache. No independent caller writes `row_indices`.

Request resets, slot remaps, cache admission and prefill warmup do not replace
or mutate these input tensors through another path. They therefore do not
invalidate the metadata. New runtime/adapter construction starts new metadata;
unload ends its lifetime. Decode traces use separate inputs. Warmup updates the
metadata through the same writers, with no preparation or trace retirement
change. Metadata is invalidated **before** changed transfers and committed only
after success, so a partially failed copy cannot cause an old value to be
incorrectly skipped on retry.

Eight CPU tests execute the actual candidate methods with fake owning buffers.
They cover initialized/warm inputs, changed/unchanged counts, both public
writers, multiple layers/buckets, changed/unchanged absolute indices, changing
source with unchanged indices, invalid inputs and partially failed transfers.
They passed against the unapplied candidate in `/tmp/ornith_prefill_refresh`.
After application run the same module without the source-directory override:

```bash
python -m unittest models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_prefill_refresh_test -v
```

The existing device probe has a separate `--refresh-cache-control` mode. Its
baseline forces both refresh writers; its candidate uses actual cached writers.
Both use production 8192 geometry. It verifies identical executed geometries,
strictly fewer mask/index transfers, exact logits and complete valid model
state at aligned/nonaligned fragments, completion and following traced decode,
and retained programs/traces. Default compact-height testing remains unchanged.

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
timeout 3600 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_compact_prefill_probe \
  --refresh-cache-control --all-layers \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/prefill_refresh_full_model.json
```

Then rerun the actual serving chunk discriminator with the retained responsive
1-prefill/2-decode configuration. Host entry gaps and this copy count establish
neither kernel attribution nor serving ITL improvement; only that serving run
can establish the benefit. Keep the existing 6144 candidate rejected.
