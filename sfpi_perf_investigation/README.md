# sfpi codegen investigation: why the Blackhole comparison kernels regressed

Supporting evidence for [ISSUE.md](ISSUE.md), the umbrella writeup, and for the seven
individually fileable drafts in [issues/](issues/README.md). This branch is not intended
to be merged — it exists so those issues can link to a runnable harness.

## Background

[tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932) converted the
Blackhole SFPU comparison kernels (`ckernel_sfpu_comp.h`,
`ckernel_sfpu_binary_comp.h`) from hand-written TTI to sfpi. All eight kernel
families regressed, by +11% to +98% cycles, so the PR was closed.

These probes isolate where the instructions go. The short answer: **raw TTI is not
inherently cheaper than sfpi.** Four of the gaps are missed optimizations that a
source rewrite can already work around — four of the eight families reach exact
instruction parity — and one is a genuinely missing primitive (a total-order float
compare that `v_if` can consume).

## Running it

```sh
./run.sh                       # finds sfpi in $SFPI, ./runtime/sfpi, or /opt/tenstorrent/sfpi
SFPI=/path/to/sfpi ./run.sh    # or point it explicitly
./verify_quotes.py             # check the code quoted in issues/ still matches the probes
```

Requires the sfpi version pinned in `tt_metal/sfpi-version` (7.69.0 at the time of
writing); `run.sh` warns on a mismatch. It compiles each probe for
`-mcpu=tt-bh-tensix -O2` and prints Tensix instructions executed per DEST row.

`count_instructions.py` does the counting. sfpi drives the SFPU through the replay
buffer, so the emitted instruction count is not the executed count; the script
models `TTREPLAY` record/replay and divides by the 8 unrolled iterations. That per-row
figure is what `PerfRunType.MATH_ISOLATE` measures, and it tracks the CI cycle deltas
closely — e.g. int32 counts +11.1% against a measured +11.5%.

## Files

| file | what it establishes |
|---|---|
| `ISSUE.md` | The umbrella writeup — the whole argument in one place |
| `issues/` | The same seven asks as individually fileable issue drafts |
| `repro.cc` | The four claims that carry the argument. **Start here.** |
| `01_idiomatic_baseline.cc` | Cost of the kernels as PR #52932 shipped them, plus the naive int32 compare (cheap but wrong) |
| `02_single_workarounds.cc` | Each workaround in isolation: `addr_mode` fold, hoisted constant, predicated store, raw `SFPGT`, constant registers |
| `03_compare_costs.cc` | Isolated cost of each relational operator; `vSMag`/`SM32` compares; the `vUInt` overflow |
| `04_and_lowering.cc` | Why `&&` collapses into `SFPPUSHC`/`SFPCOMPC`/`SFPPOPC`, and that nested `v_if` does not |
| `05_shape_rewrite.cc` | Every family rewritten in the raw-TTI shape |
| `06_commuted_compares.cc` | The commute (`x <= k` → `k >= x`) that turns 4 instructions into 1 |
| `07_all_workarounds.cc` | Every family with all workarounds stacked |
| `08_best_and_ideal.cc` | Best achievable today, and what a total-order compare would buy |
| `shim.h` | Declares `ckernel::instrn_buffer` so `sfpi.h` compiles standalone |
| `count_instructions.py` | Replay-buffer model behind every number quoted here |
| `verify_quotes.py` | Guards the `issues/` drafts against drifting from the probes |

Naming inside the probes is deliberately mechanical (`a_`…`z_`, `best_`, `ideal_`)
because `ISSUE.md` refers to individual functions by name.

## Headline result

Tensix instructions per DEST row.

| kernel family | raw TTI | PR #52932 | best sfpi today | with a total-order compare |
|---|---|---|---|---|
| float `eqz`/`nez` | 6 | 8 (+33%) | **6 (par)** | 6 |
| float `ltz`/`gtz` | 8 | 14 (+75%) | **8 (par)** | 8 |
| float `lez`/`gez` | 10 | 14 (+40%) | **10 (par)** | 10 |
| int32 `lt`/`gt`/`le`/`ge` | 9 | 10 (+11%) | **9 (par)** | 9 |
| fp32 `lt`/`gt` | 11 | 19 (+73%) | 12 (+9%) | **11 (par)** |
| fp32 `eq`/`ne` | 14 | 20 (+43%) | 16 (+14%) | **14 (par)** |
| fp32 `le`/`ge` | 13 | 23 (+77%) | 19 (+46%) | **13 (par)** |

The raw-TTI column is a source count: each `TTI_*`/`TT_*` macro is exactly one
Tensix instruction. Cross-check on the counting method — the int32 fold compiles to
`SFPLOAD, SFPLOAD, SFPSETSGN, SFPIADD, SFPXOR, SFPOR, SFPXOR, SFPSHFT, SFPSTORE`,
the identical nine-instruction sequence in the identical order as the raw TTI it
replaced.

The last column is not directly measurable, because the primitive does not exist. The
probes approximate it with the `__builtin_rvtt_sfpgt` mask plus an `SFPSETCC` to get back
into a condition code, which measures one higher on the two fp32 ordering rows
(`ideal_fp32_le` at 14, `ideal_fp32_lt` at 12). A compare that sets the CC directly drops
that round-trip instruction. `ISSUE.md` §4 spells this out.
