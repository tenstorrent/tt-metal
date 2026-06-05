# `--precompile`: one-command JIT warm for `run_safe_pytest.sh`

Make a test run compile its kernels **up-front, in parallel, hardware-free**, instead of inline and
serial — then the real run just reuses them. One flag, no env vars, no second command.

```bash
scripts/run_safe_pytest.sh --precompile <tests>          # that's it
scripts/run_safe_pytest.sh --precompile -- -k my_filter <tests>
scripts/run_safe_pytest.sh --precompile --precompile-workers 16 <tests>
```

## What it does (all internal)

1. **Fingerprint** your device once per container: cluster descriptor (UMD) + resolved 2-erisc mode.
2. **Pre-flight build_key check**: compute the build_key your real run will use and the one the
   hardware-free path produces. If they differ, warming can't help — so it's **skipped** and you get
   a normal cold run plus a one-line explanation.
3. **Warm**: a hardware-free, `xdist`-parallel meta-collect compiles every distinct kernel into
   **whatever cache you already have** (`TT_METAL_CACHE`, or tt-metal's default). No device used.
4. **Run**: the normal test run hits the warm cache.

## It can only make a run slower, never broken or wrong

Every failure path degrades to a normal cold run. The warm cache holds **content-hashed** kernels —
byte-identical to what a cold run compiles — so results are identical (verified: see
`scripts/precompile_verify.sh`). If the device is unhealthy, the build is stale, the descriptor is
from another machine, or the build_key doesn't match, you get a cold run + a clear message.

## Reading the diagnostic

```
PRECOMPILE: ✓ fingerprint matches your device (build_key 8475…) — the warm cache WILL be reused.
PRECOMPILE: ✓ warm pass complete in 16s (build_key 8475…) — the real run below reuses it.
```
→ you got the speedup.

```
PRECOMPILE: ✗ build_key MISMATCH — your device uses 8475…, the fingerprint produces 5390…
PRECOMPILE:   => warm pass SKIPPED (no wasted work); running COLD. Results stay CORRECT.
PRECOMPILE:   Cause: • stale descriptor from another machine/docker -> rm -f /tmp/tt_precompile_cluster_desc.yaml
PRECOMPILE:          • a multi-device / Blackhole config the (1,1) fingerprint didn't reproduce
```
→ no speedup, results still correct, and it tells you why. Most common fix: delete the cached
descriptor (it's keyed to the machine you first ran on) and re-run.

## Multi-device / Blackhole

The whole fingerprint is **device-derived** (descriptor, 2-erisc, dispatch core type all come from
your real device), so it is arch- and topology-agnostic by construction. The pre-flight is the safety
net: on a homogeneous mesh the `(1,1)` warm key matches and you get the speedup; if a config isn't
reproduced by the hardware-free path, the pre-flight says so and you run cold. **It never silently
does the wrong thing** — `git checkout` it into a multi-device/Blackhole docker and the diagnostic
tells you exactly what happened.

## Known limitations (all degrade to cold, never wrong)

- Tests that drive **trace/graph capture** can't be warmed (NO_DISPATCH blocks recording) — they
  cold-compile in the real run.
- Tests using a **non-standard config** (different `num_hw_cqs`, dispatch core, mesh shape than the
  `(1,1)` warm pass) produce a different build_key for those programs → those cold-compile inline.
- Host-side tensor prep that `from_torch` does (e.g. `fill_pad` tilize) is partially skipped by the
  meta collect → a small % of programs cold-compile in the real run (still correct).

## Verifying equivalence

```bash
scripts/precompile_verify.sh <tests>   # runs cold + --precompile, asserts identical results
```
