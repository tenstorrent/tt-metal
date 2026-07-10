---
name: profiler-zone-name-only
description: Use tt-metal's opt-in name-only zone-hash mode (TT_METAL_PROFILER_ZONE_NAME_ONLY=1) to eliminate 16-bit device-profiler zone-hash collisions in large fused programs, deterministically and without adding bits. Use when profiling a blaze fused layer (e.g. GPT-OSS) that hits "Source location hashes are colliding", or when you want stable-across-builds zone ids. Pairs with the zone-hash-collision skill (the problem).
---

# Opt-in name-only zone hashing (`TT_METAL_PROFILER_ZONE_NAME_ONLY`)

The deterministic fix for the 16-bit zone-hash collision (see `zone-hash-collision` for the problem).
Default **off** — behavior unchanged unless you set the flag.

## How to use

```bash
export TT_METAL_PROFILER_ZONE_NAME_ONLY=1        # set before the profiling run
# ...then run profiling exactly as usual (python -m tracy -r -p / TT_METAL_DEVICE_PROFILER=1)
```

That's the whole interface: one env var, read at kernel-compile time (device) and at profiler-read
time (host), so device and host stay consistent.

## What it does

Instead of hashing the full source string `name,file,line,KERNEL_PROFILER`, it hashes **only the zone
name** (field[0]). The `#pragma message` still emits the full source location, so the host log keeps
file/line for display. Effect on the blaze GPT-OSS global layer:

| | distinct ids | 16-bit collisions | stable across builds |
|---|---|---|---|
| default (full string) | 327 | 1 | ❌ (filename has a content hash) |
| **name-only** | **97** | **0** | ✅ |

The 327→97 drop is because the same op is compiled into ~3.4 content-hashed kernel files; those
same-name duplicates collapse to one id (and were the source of the collision + the per-build churn).

## Why it's correct here (and opt-in, not default)

- **Correct for blaze:** the code-gen gives every fused sub-op a unique name, so collapsing by name
  loses nothing — the per-op report already aggregates by name. Verified on hardware: 0 collisions,
  all op names resolve correctly, per-op times match the baseline (timing is untouched — hashing only
  relabels ids).
- **Opt-in because** name-only merges *genuinely different* zones that share a name (standard tt-metal
  keeps file/line precisely to disambiguate those). Blaze never does this; other models might → flag,
  not default.
- **Residual:** 97 names still has ~6% birthday chance for a *future* zone set, so keep the non-fatal
  skip in as a safety net. Not a global default; a perfect-hash registry would be the only way to make
  full-string 16-bit ids collision-free.
- **Do NOT** instead hash the *filename* only — it changes every build (content hash) and lumps many
  different zones into one file → far worse.

## Implementation (tt-metal, PR tenstorrent/tt-metal#49467)

Driven by one env var via `getenv` — no rtoptions enum edits. Three files:

- `tt_metal/tools/profiler/kernel_profiler.hpp` — new `PROFILER_ZONE_HASH_SRC(name)` seam; under
  `-DPROFILER_ZONE_NAME_ONLY` the `constexpr Hash16_CT` hashes `name` only (all `DeviceZoneScopedN*`
  variants). `#pragma message` still emits the full location.
- `tt_metal/impl/profiler/profiler.cpp` — host matches it: hashes the name field when the env var is
  set. Composes with the non-fatal skip (same-name/different-file strings collapse and are harmlessly
  skipped → exactly one map entry per name).
- `tt_metal/jit_build/build.cpp` — injects `-DPROFILER_ZONE_NAME_ONLY` into JIT kernel compilation
  when the env var is set, keeping device and host consistent.

Rebuild the host lib (`ninja -C build`) after; device kernels recompile via JIT on the next run.
