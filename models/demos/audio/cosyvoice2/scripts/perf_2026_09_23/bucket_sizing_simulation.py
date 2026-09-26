"""Streaming design, item 1 of the user's two pre-code follow-ups: size the bucket set
concretely for the chunk-shape decision (bucketed padding + masks, chosen over a fixed
window on 2026-09-23 to preserve upstream's real growing-prefix + chunk-causal receptive
field).

Real parameters, not assumed (see this round's research + tt/flow/encoder.py):
  - speech token rate: 25 Hz (mel frame rate 50 Hz = 2x, `tt/hifigan/source.py`'s
    upsample_scale=480 at sampling_rate=24000 -> 50 Hz mel; scripts' `token_len =
    prompt_feat.shape[1] // 2` confirms token rate is half that, 25 Hz).
  - streaming hop: 25 new tokens per chunk, first chunk = 25 + prompt padding
    (BRINGUP_STATUS.md's streaming design note).
  - pre-lookahead: 3 tokens (`tt/flow/encoder.py`'s `PRE_LOOKAHEAD_LEN`).
  - chunking is over the WHOLE GROWING PREFIX (not a sliding window) -- so the flow
    encoder's geometry at chunk n is `prompt_len + n * hop` tokens, growing without bound
    over a session unless bucketed.

This script does NOT touch device or production code. It drives the REAL
`GeometryWeightCache` class (imported directly, not re-implemented) with a synthetic
bucket-access trace derived from the real parameters above, so the hit/eviction numbers
reported are the actual cache's real behavior against this access pattern, not a
hand-wavy estimate.

Run: /opt/venv/bin/python bucket_sizing_simulation.py   (no device, no PYTHONPATH needed
     beyond making tt/geometry_cache.py importable -- see sys.path insert below)
"""
import sys

sys.path.insert(0, "/home/user/tt-metal")

from models.demos.audio.cosyvoice2.tt.geometry_cache import GeometryWeightCache

TOKEN_RATE_HZ = 25
HOP_TOKENS = 25
PRE_LOOKAHEAD = 3
PROMPT_TOKENS = 100  # ~4s prompt clip -- representative, not universal; the real value
# varies by whatever reference clip a session uses. Flagged as a free parameter below.

# Real measured anchor: the 2026-09-21 regression run's non-streaming, whole-utterance
# conv caches grew 90.9 -> 134.1 MB/bank across 4 DISTINCT (length, batch) geometries
# touching the FULL pipeline (flow encoder + CFM estimator + all ~40 HiFT convs) --
# ~14.4 MB/bank per additional geometry, averaged across that whole set. The streaming
# encoder's bucketing only touches the flow encoder's own convs (pre_lookahead conv1/
# conv2, up_layer's conv -- 3 conv instances, vs. the ~40+ in the full pipeline that
# produced the 14.4 MB/geometry figure), so the PER-BUCKET footprint should be smaller.
# This is an ESTIMATE, clearly labeled: scale the full-pipeline figure down by the
# fraction of convs actually involved (3 of the ~43 conv instances touched across the
# whole pipeline in that measurement) as a rough lower bound, and try a few multiples of
# it to see how sensitive the conclusion is -- a real number would need an actual
# streaming-shaped measurement, which doesn't exist yet (no bucketed encoder path is
# built). Flagged, not hidden behind a single made-up constant.
FULL_PIPELINE_MB_PER_GEOMETRY = 14.4
CONV_INSTANCES_IN_FULL_MEASUREMENT = 43  # flow encoder (3) + CFM estimator convs + ~40 HiFT convs
CONV_INSTANCES_IN_ENCODER_BUCKETING = 3
MB_PER_BUCKET_ESTIMATES = {
    "conservative (scaled by conv-instance-count fraction)": FULL_PIPELINE_MB_PER_GEOMETRY
    * CONV_INSTANCES_IN_ENCODER_BUCKETING
    / CONV_INSTANCES_IN_FULL_MEASUREMENT,
    "pessimistic (same per-geometry cost as the full pipeline)": FULL_PIPELINE_MB_PER_GEOMETRY,
}

DEFAULT_THRESHOLD_MB = 150  # GeometryWeightCache's real default (COSYVOICE2_DRAM_FREE_THRESHOLD_MB)


def linear_bucket(length: int, step: int) -> int:
    return ((length + step - 1) // step) * step


def geometric_bucket(length: int, growth: float = 1.25, floor: int = 64) -> int:
    """Bucket boundaries grow by `growth` each step starting from `floor` -- caps
    RELATIVE padding waste (roughly constant % overhead) instead of absolute waste,
    unlike a fixed linear step (huge % overhead early, tiny % overhead late)."""
    b = float(floor)
    while b < length:
        b *= growth
    return int(round(b))


def simulate(total_seconds: int, bucket_fn, label: str):
    total_new_tokens = total_seconds * TOKEN_RATE_HZ
    n_chunks = total_new_tokens // HOP_TOKENS
    lengths = [PROMPT_TOKENS + n * HOP_TOKENS for n in range(1, n_chunks + 1)]
    buckets = [bucket_fn(l) for l in lengths]
    waste = [b - l for b, l in zip(buckets, lengths)]
    waste_pct = [100 * w / l for w, l in zip(waste, lengths)]
    distinct = sorted(set(buckets))

    print(f"\n--- {label}, {total_seconds}s utterance ({n_chunks} chunks, prompt={PROMPT_TOKENS} tok) ---")
    print(f"true prefix length range: {lengths[0]} -> {lengths[-1]} tokens")
    print(f"distinct buckets hit: {len(distinct)}  ({distinct[0]} -> {distinct[-1]})")
    print(
        f"padding waste per chunk: min {min(waste)} / avg {sum(waste)/len(waste):.1f} / max {max(waste)} tokens  "
        f"({min(waste_pct):.1f}% / {sum(waste_pct)/len(waste_pct):.1f}% / {max(waste_pct):.1f}% of true length)"
    )
    return lengths, buckets, distinct


def simulate_cache(buckets_by_utterance: list[list[int]], threshold_mb: float, mb_per_bucket: float, label: str):
    """Drives the REAL GeometryWeightCache with a synthetic per-bucket byte cost,
    across several back-to-back utterances in one session (the realistic streaming
    scenario -- does bucket reuse survive from one utterance to the next, or does each
    utterance evict the last one's buckets before they can be reused?)."""

    class FakeDevice:
        def __init__(self, total_mb: float):
            self.total_bytes = int(total_mb * 1024 * 1024)
            self.free_bytes = self.total_bytes

    class SimGeometryWeightCache(GeometryWeightCache):
        def __init__(self, device, threshold_mb):
            self.device = device
            self._threshold_bytes = int(threshold_mb * 1024 * 1024)
            from collections import OrderedDict

            self._entries = OrderedDict()

        def _free_bytes_per_bank(self):
            return self.device.free_bytes

    entry_bytes = int(mb_per_bucket * 1024 * 1024)

    # Total device DRAM: matches this round's real measured device (1021 MB/bank).
    device = FakeDevice(total_mb=1021)
    cache = SimGeometryWeightCache(device, threshold_mb=threshold_mb)

    def put(key):
        device.free_bytes -= entry_bytes
        cache.put(key, [f"fake-tensor-{key}"], key)

    # Override pop() entirely -- the real GeometryWeightCache.pop() calls
    # ttnn.deallocate() on owned tensors, which this simulation has no real device/ttnn
    # for (deliberately, to keep this fast and dependency-free). Bookkeeping (entry
    # removal, LRU order via _entries.pop) is exercised via the REAL cache's get()/put()/
    # _evict_while_needed(); only the actual deallocate call is swapped for the
    # free-byte-accounting equivalent.
    def sim_pop(key):
        entry = cache._entries.pop(key, None)
        if entry is not None:
            device.free_bytes += entry_bytes

    cache.pop = sim_pop

    hits = misses = evictions_total = 0
    for utt_idx, buckets in enumerate(buckets_by_utterance):
        seen_this_utt = set()
        for b in buckets:
            key = b
            if cache.get(key) is not None:
                hits += 1
            else:
                misses += 1
                n_before = len(cache)
                put(key)
                n_after = len(cache)
                # put() may have evicted others to make room -- count net entries removed
                # (excluding the one we just added) as evictions from THIS put.
        seen_this_utt.update(buckets)

    print(f"\n--- cache simulation: {label} ---")
    print(f"threshold: {threshold_mb} MB/bank free floor, ~{mb_per_bucket:.2f} MB/bank per bucket entry")
    print(f"utterances simulated: {len(buckets_by_utterance)}, total bucket accesses: {hits + misses}")
    print(f"hits: {hits}  misses (fresh capture/prepare paid): {misses}  hit rate: {100*hits/(hits+misses):.1f}%")
    print(f"final cache size: {len(cache)} entries, final free DRAM: {device.free_bytes/1024/1024:.1f} MB/bank")
    return hits, misses


def find_breaking_point(buckets_by_utterance: list[list[int]], threshold_mb: float, total_dram_mb: float = 1021):
    """Sweep per-bucket MB cost upward until eviction actually starts firing (hit rate
    drops below what pure first-time-miss accounting predicts), to bound how wrong the
    MB-per-bucket estimate can be before it matters."""
    n_distinct = len(set(b for buckets in buckets_by_utterance for b in buckets))
    ideal_misses = n_distinct  # one miss per distinct bucket, ever, if nothing evicts
    budget_mb = total_dram_mb - threshold_mb
    naive_breaking_point = budget_mb / n_distinct
    print(
        f"\n--- breaking-point sweep: {n_distinct} distinct buckets across this session, "
        f"naive budget/bucket = {naive_breaking_point:.1f} MB ---"
    )
    for mb in [naive_breaking_point * f for f in (0.5, 0.8, 1.0, 1.2, 2.0, 5.0)]:
        hits, misses = simulate_cache(buckets_by_utterance, threshold_mb, mb, f"sweep probe @ {mb:.1f} MB/bucket")
        extra_misses = misses - ideal_misses
        print(f"  -> {extra_misses} eviction-caused re-misses beyond the {ideal_misses} unavoidable first-time ones")


if __name__ == "__main__":
    print("=" * 78)
    print("PART 1: how many distinct buckets, how much padding waste")
    print("=" * 78)
    for seconds in (30, 60):
        simulate(seconds, lambda l: linear_bucket(l, 32), "linear step=32")
        simulate(seconds, lambda l: linear_bucket(l, 64), "linear step=64")
        simulate(seconds, lambda l: linear_bucket(l, 128), "linear step=128")
        simulate(seconds, lambda l: geometric_bucket(l, growth=1.25), "geometric growth=1.25x")
        simulate(seconds, lambda l: geometric_bucket(l, growth=1.5), "geometric growth=1.5x")

    print("\n" + "=" * 78)
    print("PART 2: real GeometryWeightCache behavior against a realistic multi-utterance session")
    print("=" * 78)
    # 5 back-to-back 30s utterances in one session, sharing the SAME bucket scheme --
    # this is the realistic access pattern: does an utterance's early (small) buckets
    # survive to be reused by the NEXT utterance, or does the growing frontier evict them
    # before that happens?
    N_UTTERANCES = 5
    for scheme_name, bucket_fn in [
        ("linear step=64", lambda l: linear_bucket(l, 64)),
        ("geometric growth=1.25x", lambda l: geometric_bucket(l, growth=1.25)),
    ]:
        _, buckets, _ = simulate(30, bucket_fn, scheme_name)
        buckets_by_utterance = [buckets for _ in range(N_UTTERANCES)]
        for est_name, mb_per_bucket in MB_PER_BUCKET_ESTIMATES.items():
            simulate_cache(
                buckets_by_utterance,
                DEFAULT_THRESHOLD_MB,
                mb_per_bucket,
                f"{scheme_name}, {N_UTTERANCES}x 30s utterances, {est_name} ({mb_per_bucket:.2f} MB/bucket)",
            )
        find_breaking_point(buckets_by_utterance, DEFAULT_THRESHOLD_MB)
