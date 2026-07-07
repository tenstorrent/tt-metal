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

## Using it from a run

```bash
# Warm the cache off-device, then submit the real job to the broker.
tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh \
    -e <env.yaml> -w <workspace> -- <command...>
```

`prewarm_and_submit.sh` runs `build_Release/tools/kernel_prewarm` (reading
`TT_METAL_CACHE`/`TT_METAL_HOME` from the env yaml) before the broker
reservation. Warm cache => sub-second no-op. After a kernel/header edit =>
rebuilds the working set off-device (~20s here) instead of on-device.

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

## What it costs / when it does NOT help

The manifest is a byproduct of a real run. So the first run on a **fresh cache
with no manifest** still compiles on-device — nothing to prewarm from yet.
Prewarm only helps run #2 onward for a given cache.

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
