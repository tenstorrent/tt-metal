# SFPI Silicon Validation Agent Handoff

Use these three Tenstorrent branches together:

```text
tenstorrent/tt-metal   nkapre/sfpi
tenstorrent/sfpi       nkapre/sfpi
tenstorrent/sfpi-gcc   nkapre/sfpi
```

The detailed source of truth is
[`WELFORD_SILICON_VALIDATION.md`](https://github.com/tenstorrent/sfpi/blob/nkapre/sfpi/WELFORD_SILICON_VALIDATION.md).
This file is the short machine handoff.

## 1. Clone and verify

```bash
git clone --branch nkapre/sfpi \
  git@github.com:tenstorrent/tt-metal.git tt-metal-sfpi

git clone --recursive --branch nkapre/sfpi \
  git@github.com:tenstorrent/sfpi.git sfpi-scheduler

cd sfpi-scheduler
git submodule update --init --recursive

git rev-parse HEAD
git -C gcc rev-parse HEAD
```

Known-good commits when this handoff was created:

```text
sfpi:     dd260b7205730e143f9ca1c4e8a754106a2b4985
sfpi-gcc: 8bea8aba4945485f32307212d28ca7dc6107f18d
```

Later commits on the same branches are acceptable only if their validation
notes explain the delta. Record the actual commits in every result bundle.

## 2. Build and gate the compiler

On Ubuntu, install the ordinary SFPI build dependencies plus
`liblpsolve55-dev` and `libsuitesparse-dev`, then run:

```bash
cd /path/to/sfpi-scheduler

SFPI_WITH_LP_SOLVE=yes \
  ./scripts/build.sh --dir="$PWD/../sfpi-silicon-build" --checking

./scripts/validate-sfpu-pressure-scheduler.sh \
  "$PWD/../sfpi-silicon-build/sfpi" \
  "$PWD/../sfpi-silicon-compiler-validation"

SFPI_WITH_LP_SOLVE=yes \
  ./scripts/build.sh --dir="$PWD/../sfpi-silicon-build" --test-tt
```

Stop on any unexpected test result, zero-test false green, nondeterministic
assembly, or compiler spill/fill attempt.

## 3. Make this TT-Metal worktree use that compiler

TT-Metal's JIT prefers `runtime/sfpi` over `/opt/tenstorrent/sfpi`. The
standalone TT-LLK harness uses `tt_metal/tt-llk/tests/sfpi`. In the clean
validation clone, point both paths at the same installation:

```bash
export TT_METAL_HOME=/path/to/tt-metal-sfpi
export CUSTOM_SFPI=/path/to/sfpi-silicon-build/sfpi

test -x "$CUSTOM_SFPI/compiler/bin/riscv-tt-elf-g++"
mkdir -p "$TT_METAL_HOME/runtime"
ln -sfn "$CUSTOM_SFPI" "$TT_METAL_HOME/runtime/sfpi"

LLK_SFPI="$TT_METAL_HOME/tt_metal/tt-llk/tests/sfpi"
if [[ -e "$LLK_SFPI" && ! -L "$LLK_SFPI" ]]; then
  mv "$LLK_SFPI" "${LLK_SFPI}.released-backup"
fi
ln -sfn "$CUSTOM_SFPI" "$LLK_SFPI"

readlink -f "$TT_METAL_HOME/runtime/sfpi"
readlink -f "$LLK_SFPI"
"$TT_METAL_HOME/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" --version
```

Use a fresh JIT cache after changing compiler commit, architecture, scheduler
mode, or implementation selector. Do not use build-map mode for measurements;
it deliberately omits the compiler version from the JIT cache key.

This validation branch intentionally does not change `tt_metal/sfpi-version`.
A product/CI repin requires immutable SFPI packages and SHA-256 hashes. The
local override is the correct mechanism until those artifacts are published.

## 4. Execute the silicon mission

Follow the full SFPI runbook and produce these controlled Welford variants:

```text
HANDWRITTEN_DIRECT
HANDWRITTEN_REPLAY
VFLOAT_DIRECT
VFLOAT_RESCUE
VFLOAT_MANUAL_EARLY_FOLD
```

Keep loads, transpose, reciprocal lookup, initialization, stores, and
finalization identical. Only the row recurrence may differ. Validate raw mean
and M2 before final variance on partial rows, multi-tile state carry, poisoned
padding, high-offset inputs, NaN/Inf/signed-zero, and repeated invocations.

Then enable the device profiler and measure paired, randomized runs on every
available Wormhole and Blackhole card:

```bash
export TT_METAL_DEVICE_PROFILER=1
# Run the Welford perf binary with >=20 warmups and >=100 measurements.
python tools/tracy/process_ops_logs.py
```

Report device cycles per row/block/tile, not pytest or simulator wall time.
Archive ELF/assembly hashes, physical LREG maps, MAD/MOV/NOP/load/store counts,
replay footprint, static text size, correctness statistics, raw cycles, and
the exact commands and environment.

After Welford, run the four broad-performance A/B controls:

1. serial versus interleaved Horner/addcmul;
2. automatic replay off versus on for the same unrolled body;
3. `DISABLE_SFPLOADMACRO` off versus on for `mul_int` and `where`; and
4. existing MOP versus the no-MOP matmul control.

Lead the report with `GO`, `GO-WH-ONLY`, `GO-BH-ONLY`, or `NO-GO`. Generated
Welford may replace handwritten replay only if correctness is clean and its
paired median hardware cycles are non-regressing within the full runbook's
confidence and code-size gates.
