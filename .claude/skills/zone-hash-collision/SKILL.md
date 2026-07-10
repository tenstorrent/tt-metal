---
name: zone-hash-collision
description: Diagnose and explain tt-metal's 16-bit device-profiler zone-source hash collision — the "Source location hashes are colliding" TT_THROW at close_mesh_device that aborts the whole device-profiler read. Use when a profiling run crashes in populateZoneSrcLocations, when a large fused program (e.g. blaze GPT-OSS) has hundreds of DeviceZoneScopedN zones, or to understand why per-op device zones collide and what dropping a zone means. Pairs with the profiler-zone-name-only skill (the fix).
---

# tt-metal device-profiler 16-bit zone-hash collision

## Symptom

At `close_mesh_device()` the device-profiler read aborts:

```
critical | Always | TT_THROW: Source location hashes are colliding, two different locations are having the same hash
RuntimeError: TT_THROW @ tt_metal/impl/profiler/profiler.cpp:237: tt::exception   # ~line 281 on origin/main
  --- tt::tt_metal::populateZoneSrcLocations(...)
  --- tt::tt_metal::generateZoneSourceLocationsHashes()
  --- tt::tt_metal::DeviceProfiler::readResults(...)
  --- ReadMeshDeviceProfilerResults(...) --- MeshDevice::close() --- close_mesh_device(...)
```

No `profile_log_device.csv` is produced → all per-op profiling is lost.

## What the hash is, and what a collision means

Every profiler sample the **device** writes carries a compact **16-bit `timer_id`** =
`Hash16_CT(zone_source_string)`, where the source string is:

```
GPTOSS_GLOBAL_LAYER__ATTN_Q , <kernel_file.cpp> , <line> , KERNEL_PROFILER
     └ field[0]: zone name       └ field[1]          └[2]     └[3]
```

`Hash16_CT` = FNV-1a-32 folded to 16 bits (`(lo^hi) & 0xFFFF`), 65536 slots. A **collision** =
two *different* source strings producing the *same* 16-bit number. It lives in two coupled places:

1. **Device / compile time (info is lost here).** `Hash16_CT` is `constexpr`; the id is baked into
   the kernel binary, so two colliding zones stamp the *same* id on every sample — indistinguishable
   in the raw stream.
2. **Host / read time (detected here).** `populateZoneSrcLocations()` rebuilds `id → (name,file,line)`
   from the zone log; when a second distinct string maps to an already-taken id
   (`hash_to_zone_src_locations.contains(hash)`), that's the collision the code throws on.

## Why it hits large fused layers (and not normal ops)

Birthday bound on 65536 slots vs. number of distinct source strings N:

| N distinct strings | p(≥1 collision) |
|---|---|
| ~30 (a normal model run) | <1% |
| 100 | ~7% |
| 200 | ~26% |
| **327 (blaze GPT-OSS global layer)** | **~56%** |

- **Standard tt-metal impls** emit ~30 distinct device zones: the model adds ~0 custom zones, only
  ~16 ttnn ops add short **shared** names (`"EXP"`, `"Q@KT MM+Pack"`), and zones live in **fixed**
  kernel source paths → few distinct strings → essentially never collide. Per-op timing comes from
  **host** op-to-op profiling, not hundreds of device zones.
- **Blaze fuses the whole layer into one program** and names every sub-op uniquely, and each name is
  compiled into ~3.4 **content-hashed** generated kernel files (`gptoss_global_layer_59b1….cpp`), so
  97 distinct names inflate to **327 distinct strings** → ~56% collision. Real example: id `0x99ad`
  is produced by both `ATTN_Q@…59b1…:289` and an `…ATTN_O__O_BROADCAST` fabric-barrier zone.

## Diagnose

Run the profiling once (device writes all zones to the log even on a patched build), then analyze the
compiler-emitted zone log offline with the bundled bit-exact `hash16CT` replica:

```bash
# zone log lives at <TT_METAL_PROFILER_DIR>/.logs/new_zone_src_locations.log
python3 scripts/hash16_collide.py <new_zone_src_locations.log>   # distinct strings + name-only, collisions listed
python3 scripts/report_collisions.py <log> out.txt               # human-readable report file
```

`hash16_collide.py` reports collisions at two granularities: full string (current default) and
name-only (the fix). It's the ground truth for "would an unpatched build throw?" — the collision is
**intra-run**, so a fresh/flushed profiler dir does **not** avoid it.

## What "dropping a colliding zone" costs (the non-fatal skip)

The pragmatic fix (PR tenstorrent/tt-metal#49467) replaces the `TT_THROW` with
`zone_src_locations.erase(...); continue;` — skip the second colliding string instead of aborting:

- **Only the id→name label is dropped, not timing samples.** The dropped zone's samples remain in the
  CSV but are attributed to the zone that claimed the id first.
- Impact is negligible: each op is scoped in several kernels, so it still appears (verified: `ATTN_Q`
  keeps 100k+ rows even when its `@289` label is the dropped one), and the collided partner here is a
  cheap fabric barrier. 326 of 327 zones are pristine; timestamps/durations untouched.

The deterministic fix (no dropped labels) is the **name-only hashing option** — see the
`profiler-zone-name-only` skill. The only true way to keep full strings *and* 16 bits is a perfect
hash with a global zone registry (large change); widening to 32-bit ids also works (2× per-sample cost).
