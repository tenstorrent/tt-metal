# Kernel prewarm

Move JIT kernel compilation **out of the device-held window**. On a shared,
serialized device the reservation should hold the device only for device work;
kernel compilation is host work and can happen before the reservation (offline)
or overlapped with host-idle device init (in-run).

Default-on. Set `TT_METAL_KERNEL_PREWARM=0` to disable entirely (byte-for-byte
the pre-prewarm behavior).

## How it works

- **Capture.** `ProgramImpl::compile` appends each newly-seen kernel's portable
  compile recipe to a manifest at `<cache_root>/kernel_prewarm.manifest`
  (`cache_root` = `TT_METAL_CACHE`, else `$HOME/.cache`). Dedup against the
  on-disk manifest means a cold cache records the full pipeline and later runs
  append only new kernels (new op/shape/config). The manifest is
  `build_key`-tagged and holds every build_key seen on the box.
- **In-run prewarm.** At device init (firmware linked, before the host issues
  any op), a background batch rebuilds the current build_key's manifest recipes
  into the JIT cache, overlapping the host-idle `weight_load` window, so the
  op-by-op compiles become cache HITs.
- **Offline prewarm.** The `kernel_prewarm` tool compiles every manifest recipe
  into the JIT cache **without opening a device**. Run it before reserving the
  device so the reservation does no compilation at all. It also builds the
  device-init (cq_/fabric) kernels, which the in-run batch skips.

## Using it from a run (the 3-stage wrapper)

```bash
# Cold build_key (fresh cache, or dprint/watcher/compiler change): 3 stages, device freed between each.
tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh \
    -e <env.yaml> -w <workspace> -c -- <command...>
```

`prewarm_and_submit.sh` orchestrates the split flow, releasing the device between stages:

1. **capture** (device, brief) — only when cold (no manifest, or `-c`): runs the command once with
   `TT_METAL_KERNEL_CAPTURE_ONLY=1` to record the full manifest. Because this is a *separate process*
   from the real run, it captures every kernel — including the audio path the in-process cold-start
   can't (see below) — with no shared state to corrupt.
2. **compile** (host, device FREE): `build_Release/tools/kernel_prewarm` batch-compiles the whole
   manifest off-device.
3. **run** (device): submits the real job, now warm — zero compilation in the reservation.

`-c` forces the capture stage when the cache already holds *other* build_keys (e.g. after toggling
dprint) — the manifest exists but not for this run's build_key. Omit it on a fresh cache (auto-detected)
or a warm one (capture skipped; the offline stage is a sub-second dephash no-op, safe to prepend to
every run). Reads `TT_METAL_CACHE`/`TT_METAL_HOME` from the env yaml so every stage hits the run's cache.

**Measured device-held (LTX distilled bh_2x4sp1tp0), lower is better:**

| Path | cold | dprint (new build_key) | output |
|---|---:|---:|---|
| all-on-device (no prewarm) | 531s | 578s | — |
| in-process cold-start (automatic fallback) | ~235s | ~310s | byte-identical |
| **3-stage wrapper (this script)** | **~166s** | **~172s** | byte-identical |

The 3-stage wins because the off-device compile runs while the device is unreserved, and its separate
capture process warms the whole kernel set (incl. audio) that the in-process path leaves to compile
in-window. Use it for a known cold/new build_key; if an agent forgets, the automatic in-process
cold-start still lands ~235–310s. (Watcher is a separate story — it crashes fabric bring-up on this
box, so no run completes under it, prewarmed or not.)

Or invoke the tool directly (warms the cache for a later run):

```bash
TT_METAL_CACHE=<dir> TT_METAL_HOME=<root> build_Release/tools/kernel_prewarm
```

## Keeping the reservation off the compile (broker recipe)

The broker (`tt-device-mcp run`/`run-bg`) holds the device for the whole
command, so any kernel compile inside it is host work blocking a shared device.
Warm off-device first, then submit the reserved run — both device-init
(fabric/dispatch/mux) and model kernels are built off-device, so the reservation
does zero compilation:

```bash
TT_METAL_CACHE=<cache> TT_METAL_HOME=<root> build_Release/tools/kernel_prewarm   # host, no reservation
tt-device-mcp run-bg '<pytest ...>' -e <env-with-that-TT_METAL_CACHE>.yaml       # reserved: device work only
```

Off-device warmup of device-init kernels is replay-safe: every topology/runtime
input (fabric handshake role, dispatch NoC coords, semaphore ids) is folded into
the kernel hash, so replay is content-addressed — a mismatch is a cache miss →
correct recompile, never a wrong binary.

**Measured (LTX 22B girl-e2e, bh_2x4sp1tp0, 2×4 Blackhole):**

| Cache state | Kernel compiles inside the reservation | Setup | Result |
|---|---:|---:|---|
| Cold, in-run | 1355 | — | **timed out (>300s)** |
| Full off-device warm | **0** | 13.2s | pass, 90s |

The 2436 kernels compile off-device in ~1–60s of host CPU (cold vs incremental);
the reserved run's in-run prewarm is then a ~0.5s warm no-op.

## Capture-only mode: warm even the FIRST run of a new build_key

The manifest is normally a byproduct of a real run, so the first run on a fresh
cache (or after a `build_key` change — see below) has nothing to prewarm from and
compiles on-device. `TT_METAL_KERNEL_CAPTURE_ONLY=1` breaks that: the pipeline
run generates each model kernel's genfiles and records its manifest recipe but
**skips the gcc compile and skips dispatch** (device-init cq_/fabric kernels still
compile — the device must come up). A fixed-schedule pipeline traverses on garbage
tensors and records the full manifest with only a brief device touch; then the
off-device tool compiles it, and the real run is warm.

```bash
# 1. capture pass: full manifest, no model gcc, no dispatch (device held briefly)
TT_METAL_KERNEL_CAPTURE_ONLY=1  <run the pipeline once, e.g. via the broker>
# 2. off-device compile (host, device free)
TT_METAL_CACHE=<cache> TT_METAL_HOME=<root> build_Release/tools/kernel_prewarm
# 3. real run: warm
```

Measured on the LTX first run of a new build_key (device-held): **531s all-on-device
→ ~166s** (75.6s capture + 90.2s warm; 52s compile off-device), output byte-identical.
Works for a freshly-toggled dprint build_key too (577.9s → ~176s), with no manifest
shipped ahead of time. Correctness is unchanged: recipes are content-addressed, and
the model gcc is byte-identical whether run in-line or off-device. The capture pass
still holds the device for its traversal (host genfile-gen while the device idles) —
it removes the ~500s compile from the reservation, not the traversal.

## In-process cold-start (automatic, no wrapper)

The capture-only + off-device-compile flow above is a three-step recipe an agent
must remember. The LTX pipeline does it **automatically in one run**: on a cold
build_key (detected via `KernelPrewarmColdStartNeeded()` — capture armed, no in-run
batch launched) it runs one capture-only warmup, batch-compiles the manifest
off-device **in-process** (`KernelPrewarmOfflineCompile()`), resets the poisoned
in-memory state (program cache + tracers), then runs warm. No wrapper, no manual
capture pass, nothing to forget. Controls are bound to `ttnn._ttnn.device`
(`kernel_prewarm_cold_start_needed` / `_set_capture_only` / `_offline_compile`),
declared in `<tt-metalium/kernel_prewarm_control.hpp>`.

Measured (LTX distilled `bh_2x4sp1tp0`): **531s → 235s device-held, output
byte-identical**. One reservation held continuously (the off-device compile runs
while the device sits idle-but-reserved), so it trades ~40-70s more device-held
than the split three-phase for being foolproof and automatic.

**Limitation — the audio path.** Components that lazily initialize device statics
on first dispatch and capture traces with `prep_run=False` (the LTX vocoder) can't
be captured under capture-only: dispatch is skipped, so their statics are left
corrupted (wrong output) or half-warmed (crash). The pipeline **skips audio decode
in the capture pass** and lets it compile op-by-op in the warm real warmup. On a
warm ccache that's fast (the 235s above); on a **new** build_key (e.g. toggling
dprint) that audio compile is cold-ccache serial and pushes device-held to ~310s.
For a strict <250s on a new build_key, use the split three-phase (offloads the
audio compile off-device too).

## What it costs / when it does NOT help

The manifest is a byproduct of a real run. So the first run on a **fresh cache
with no manifest** still compiles on-device — unless you run the capture-only pass
above. Otherwise prewarm only helps run #2 onward for a given cache.

A prewarm miss (full on-device recompile) happens when the **`build_key`
changes**, because prewarm only warms the current build_key. `build_key` folds
in: the compiler/sfpi version (the nuke — invalidates every build_key), arch,
kernel cflags/defines/includes, dispatch config (num CQs, dispatch core,
harvesting mask), and device debug toggles (**watcher, dprint, sanitizer**,
erisc modes). It is one-time per distinct config: toggling dprint off↔on pays
cold only the first time each side is seen, as long as `TT_METAL_CACHE` is not
wiped. A new op/shape/config appends one kernel (one on-device compile), not a
full miss.

## Correctness: it never serves stale code

The kernel body is compiled from its on-disk path (`#include`d by the generated
wrapper), never from the manifest's generated-file snapshot, and the on-disk
`.dephash` gate is re-verified by the op-by-op path on every run: any change to
a source or header content forces a recompile. Verified on device by
`tests/tt_metal/tt_metal/api/test_offline_kernel_compile.cpp`:

- `OfflinePrewarmReflectsEditedKernelBody` — edit a kernel body, run the offline
  prewarm, assert the loaded binary (`brisc.elf`) reflects the edit, never the
  captured snapshot.
- `EditedKernelBodyForcesRecompileNotStaleCacheHit` — the dephash backstop
  through the op-by-op path.

Run them (device required, slow dispatch):

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
  build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter='*OfflinePrewarm*:*EditedKernelBodyForcesRecompile*'
```

Note: `<name>.elf.xip.elf` next to a kernel ELF is a tt-triage debug
disassembly dump written at load time (`tt_memory.cpp`), not a loaded binary;
prewarm does not regenerate it, and it self-heals on the next real load.
