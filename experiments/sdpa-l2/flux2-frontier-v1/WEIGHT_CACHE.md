# Converted-weight caching

## Working runtime configuration

```bash
export TT_DIT_CACHE_DIR=/localdev/cglagovich/flux2-frontier-20260915/dit-cache-v2
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
```

The second setting disables the runtime's **pinned host-memory cache**, not
the converted-weight files on disk. It selects the existing non-pinned transfer
path. No attention arithmetic, input precision, block sizes or communication
schedule changes. Apply it consistently to every variant in the new cohort.

## Evidence and scope

On bh-lb-08, reservation 221619, the fresh-cache synthetic test succeeds at
cold conversion/save but stalls at ordinary warm loading. The same cached
files reload successfully in a fresh process with the pinned cache limit set
to zero. No reset or cache-file modification was necessary between those two
warm-load attempts.

The probe covers three 4096×4096 BF16 parameters: replicated, column-sharded
over four devices, and row-sharded over four devices, on the 2×4 mesh. All 24
device-shard SHA256 hashes match cold versus warm. The warm run forbids falling
back to Torch conversion. Measured load-plus-synchronization times are:

- Cold conversion and cache save: 2.6674 s.
- Warm converted-cache reload: 0.01588–0.01619 s across three fresh processes.

These are synthetic measurements, exclude process/device setup and
verification hashing, and are **not full-model startup speedups**. See
`cache-validation-01/{cold,warm}.json`, `cache-validation-repeat-02/warm.json`,
and `cache-validation-source-03/warm.json`. The third warm run also independently
regenerated the original Torch tensors and required exact equality for all 24
shards. The pinned-path stall reproduced twice and was terminated;
bounded debugger attempts did not yield a stack. The exact runtime defect
is not established. This is a validated workaround, not a low-level fix.

`qualify_weight_cache.py` performs the full-model cold/warm validation in
separate processes. It requires a fresh checkpoint cache namespace, generates
one two-step execution-smoke image per phase, requires every warm component
to avoid Torch conversion, and compares all eight shard hashes of
`time_guidance_embed.timestep_embedder.linear_2.weight` (the original stall site).
The two-step images are not image-quality measurements. Full-pipeline replay
discrepancies are recorded separately and are not disguised as cache failures.

The full evaluation records the cache root, pinned-cache limit and each model
component's measured load time and conversion/cache-hit status in its manifest.
Weight-load timings include cache writes on a cold miss; pipeline setup also
includes model construction and the existing CPU reference-model load.

## Full-model qualification result

Completed on bh-lb-08 at 2026-09-16 15:10 UTC, in two separate processes:

| Measurement | Cold conversion + cache write | Warm cache reload |
|---|---:|---:|
| VAE load | 0.397 s | 0.019 s |
| Transformer load | 139.573 s | 3.045 s |
| Text encoder load | 104.835 s | 2.151 s |
| Component load total | 244.805 s | 5.215 s |
| Pipeline setup total | 248.773 s | 9.068 s |

Warm pipeline setup is **27.4× faster** than the first cold load/write, a measured
reduction of 239.7 seconds. An uncached conversion without writing cache files
would cost less than that first-run baseline. This is a warm local-filesystem-cache
measurement, not an inference-kernel speedup or a cold-disk throughput benchmark.
The component total does not include synchronization immediately following each
load call; pipeline setup includes it, plus normal model construction.

Both two-step smoke tests passed; their output PNG and final-latent file hashes
also match across cold/warm processes, and both recorded model-replay checks
were exact in each process. All three warm components hit the converted
cache without calling the Torch state-dict provider. The validation weight's
SHA256 matches cold/warm on all eight device shards. Evidence:
`cache-model-qualification-01/report.json`, both phase manifests and logs.
This cache qualification does not replace the separate full-model repeatability
and attention-quality controls.

## Qualification command

After setting the standard repository/runtime environment:

The command below records the original qualification. Repeating the cold/warm
comparison requires a **fresh cache root and output directory**; the driver
refuses to overwrite an existing checkpoint cache. Normal evaluation should
reuse the populated `dit-cache-v2` with `FLUX2_REQUIRE_WEIGHT_CACHE=1`.

```bash
python experiments/sdpa-l2/flux2-frontier-v1/qualify_weight_cache.py \
  --output /localdev/cglagovich/flux2-frontier-20260915/cache-model-qualification-01 \
  --cache "$TT_DIT_CACHE_DIR" \
  --checkpoint "$FLUX2_CHECKPOINT" \
  --embeddings /localdev/cglagovich/flux2-frontier-20260915/suite-02/prompt-embeddings
```

The old `dit-cache` is preserved; it was not used to seed the fresh cache.
