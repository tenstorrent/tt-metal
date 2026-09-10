# KDA direct-ND performance analysis

Date: 2026-09-10
Branch: `momcilo/kda-nd-cache`
Implementation revision: `787803edef3`
Latest-main comparison base: `3f254861838`
Bead: `tt-metal_tracker-cpn`

## Verdict

**Do not retain the current direct-ND implementation as-is.** It is neutral at
SP1 but slower than latest main by 238 microseconds at SP2 and 457 microseconds
at SP4. More importantly, it is slower at SP2 and SP4 than latest main plus
even the original, unoptimized explicit adapter. Compared with the optimized
adapter, direct ND loses by 196 microseconds per SP2 round trip and 400
microseconds per SP4 round trip.

The native ND accesses added to the KDA kernels are not the problem. A causal
SP4 ablation with the two surrounding bridge operations removed runs in 9.953
ms versus 9.962 ms on main. The regression is fully attributable to:

1. an ND-to-interleaved convolution-history staging copy before SP halo
   assembly; and
2. an SP collective used to write a recurrent result that is already identical
   on every SP rank.

At SP4 those operations measure 131 and 379 microseconds respectively in the
device profile. Trace-wall ablations attribute 122 and 345 microseconds. The
small difference is expected because device-profile programs can overlap and
the profiler uses an eager forward rather than the gated trace.

The direct approach is worth revisiting only if both bridge operations are
replaced by KDA-native local ND paths. Until then, the explicit adapter is both
faster and materially simpler.

## Measurement method

All new measurements use eight Blackhole devices, firmware 19.5.0,
`FABRIC_1D`, real Kimi-K3 layer-1 weights, batch 1, and sequence 5120. Layer
latency is the median of five synchronized warm trace-wall samples with ten
trace replays per sample. Accuracy is checked before timing and on the first
trace replay against the independent pure-Torch FP32 reference.

The latest-main and direct-ND layer measurements use the same production test
and checkpoint. Device attribution uses a warm eager forward with per-program
duration defined as the maximum across all eight chip records. Program
durations must not be summed as a wall-time model because programs can overlap.

The adapter numbers are the synchronized trace-wall measurements in the
supplied `kda-cache-adapter-experiment-report (1).html`. They were captured on
the older PR7 base (`4de69738505`) rather than the latest-main base, so the
end-to-end adapter totals below are additive estimates: latest-main layer time
plus independently measured adapter overhead. This is adequate for the design
decision because the margins are large, and the current direct-ND internal
convolution staging program independently reproduces the original adapter's
SP4 convolution-import time within 3.1 microseconds.

## Top-level result

| Layout | Latest main (ms) | Direct ND (ms) | Direct delta | Existing gate |
| --- | ---: | ---: | ---: | --- |
| SP1xTP8 | 9.600972 | 9.595328 | -5.644 us (-0.06%) | pass |
| SP2xTP4 | 9.547209 | 9.785333 | +238.124 us (+2.49%) | pass, narrowly |
| SP4xTP2 | 9.962375 | 10.419397 | +457.022 us (+4.59%) | **fail**; 128.667 us above ceiling |

Five available direct-ND SP4 sessions have medians from 10.406877 to
10.421207 ms, with a median of medians of 10.419397 ms. The 9.991 ms calibrated
SP4 reference is itself a median across five sessions. Against that calibrated
reference, direct ND is 428.397 microseconds (4.29%) slower, so the result is
not a single-session outlier.

Accuracy is unchanged. SP4 output, recurrent-state, and convolution-state PCCs
are 0.999896542, 0.999921593, and 0.999999437 on every applicable SP rank, the
same values recorded on latest main.

## SP4 causal decomposition

The following temporary ablations changed placement only; both preserved the
same PCCs. They were reverted after measurement and are not part of the branch.

| SP4 variant | Recurrent output | Convolution input | Median (ms) | Change from preceding row |
| --- | --- | --- | ---: | ---: |
| Current direct ND | ND via SP broadcast | ND, staged to interleaved | 10.419397 | — |
| No recurrent broadcast | interleaved; invalid final contract | ND, staged to interleaved | 10.074791 | -344.607 us |
| Native-input hybrid | interleaved; invalid final contract | interleaved; no staging | 9.952923 | -121.868 us |
| Latest main | interleaved | interleaved | 9.962375 | +9.452 us versus hybrid |

The two bridge removals save 466.475 microseconds. The hybrid is 9.452
microseconds faster than main, leaving 457.022 microseconds as the observed net
regression. This closes the attribution: within normal measurement noise, the
new KDA-native ND reads and fused state writes have no standalone penalty.

### 1. Recurrent final-state broadcast: primary cause

`models/demos/deepseek_v3_d_p/tt/kda/recurrence.py:301` invokes
`ttnn.all_broadcast` to materialize the final recurrent carry in ND DRAM. This
is logically redundant communication. `_distributed_affine_prefix` first
all-gathers every partition's affine transform and then every rank executes the
same ordered composition (`recurrence.py:248-291`), so `carry` already contains
the same final value on every SP rank.

The collective exists only because the generic final `ttnn.add` cannot emit the
required ND DRAM layout on this harvested device; the attempted direct output
resolved an invalid ninth DRAM bank. Broadcasting from one rank happens to
produce the required ND placement, but transports a replicated result that is
already local.

| Layout | Broadcast device time | Direct-vs-main wall regression | Broadcast / wall delta |
| --- | ---: | ---: | ---: |
| SP2xTP4 | 171.919 us | 238.124 us | 72.2% |
| SP4xTP2 | 379.018 us median (354.341-382.899) | 457.022 us | 82.9% |

The causal SP4 trace-wall ablation assigns 344.607 microseconds, or 75.4% of
the observed regression, to this one collective. Its cost grows from 172 to
379 microseconds because SP4 both doubles the TP-local recurrent payload from
1.5 MiB to 3 MiB and expands the SP group from two ranks to four.

For comparison, the explicit adapter's local recurrent export costs only
23.631 microseconds at SP4. The current collective is about 16 times slower.

### 2. Convolution ND-to-interleaved staging: secondary cause

`models/demos/deepseek_v3_d_p/tt/kda/convolution.py:50` slices the caller's ND
state into interleaved DRAM before concatenating it with gathered neighbor
tails. The mixed-layout SP halo assembly cannot consume the ND tensor directly.
This is an adapter conversion inside the layer even though there is no adapter
API at the layer boundary.

| Layout | Staging device time | Local convolution state | Optimized adapter import | Original adapter import |
| --- | ---: | ---: | ---: | ---: |
| SP2xTP4 | 67.339 us | 54 KiB | 9.657 us | 69.749 us |
| SP4xTP2 | 130.706 us median (129.822-131.074) | 108 KiB | 11.349 us | 133.946 us |

The SP4 staging operation is within 2.4% of the original adapter's convolution
import time. Both use the default row-major redistribution path, whose work is
partitioned by the three logical rows. Doubling TP-local heads doubles the
width and almost doubles latency while active parallelism remains limited.

The causal SP4 trace-wall ablation assigns 121.868 microseconds, or 26.7% of
the observed regression, to this staging copy.

### Native kernel changes are not responsible

At SP2 the fused QKV convolution, including its new direct ND state write,
measures 1,148.564 microseconds versus 1,150.323 on main: 1.759 microseconds
faster. The first and second recurrent scans change from 301.205/366.724
microseconds on main to 302.641/360.997 microseconds with direct ND. These are
noise-level movements, not a regression source.

SP1 requires neither distributed bridge. Its -0.06% result confirms that the
kernel-native ND paths themselves are essentially free.

## Comparison with explicit adapters

Two adapter variants are relevant:

- **Original/simple adapter:** four direct `ttnn.to_memory_config(...,
  output_tensor=...)` calls. It uses existing generic operations and requires
  no KDA kernel signature or dataflow changes.
- **Optimized adapter prototype:** the same API plus a generic row-major copy
  path that distributes aligned page units over 110 cores. It reduces the SP4
  convolution import/export from 133.946/94.331 microseconds to
  11.349/11.974 microseconds.

### Export-only handoff

This model applies when a zero/native prefill state is created locally and only
the final state is exported to the migration contract.

| Layout | Direct ND (ms) | Main + simple export (ms) | Direct vs simple | Main + optimized export (ms) | Direct vs optimized |
| --- | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | 9.595328 | 9.636374 | -41.046 us | 9.615933 | -20.605 us |
| SP2xTP4 | 9.785333 | 9.608928 | **+176.405 us** | 9.571388 | **+213.945 us** |
| SP4xTP2 | 10.419397 | 10.077200 | **+342.197 us** | 9.990957 | **+428.440 us** |

### Import + forward + export round trip

This model applies when the persistent state is in contract layout, is imported
for native KDA execution, and the replacement state is exported afterward.

| Layout | Direct ND (ms) | Main + simple round trip (ms) | Direct vs simple | Main + optimized round trip (ms) | Direct vs optimized |
| --- | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | 9.595328 | 9.681297 | -85.969 us | 9.630768 | -35.440 us |
| SP2xTP4 | 9.785333 | 9.691510 | **+93.823 us** | 9.589580 | **+195.753 us** |
| SP4xTP2 | 10.419397 | 10.232092 | **+187.305 us** | 10.019002 | **+400.395 us** |

Direct ND wins only at SP1, where no bridge is required. At distributed
topologies it loses even to the simple adapter. The optimized adapter uses at
most 56.627 microseconds for the complete SP4 round trip, while the two hidden
direct-ND bridges add roughly 466 microseconds.

## Complexity comparison

The direct-ND implementation changes 28 code/test files: 18 KDA C++/kernel
files, six tests, and four KDA Python files, with 433 insertions and 114
deletions excluding specifications and reports. It adds independent state
memory configs, rank/group output rules, relaxed sharded-input validators, a
fourth fused-convolution output, additional tensor/runtime bindings, and SP
layout workarounds.

The simple adapter's actual conversion mechanism is four one-line calls in one
Python module (`cache_adapters.py:198-215` on
`momcilo/kda-cache-adapter-ablation`). Its 240-line investigation module is
mostly geometry, allocation, and validation. The optimized prototype does add
three generic C++ data-movement files plus tests, but it accelerates a reusable
primitive and still leaves KDA kernels and their APIs unchanged.

The present complexity is therefore not justified by performance. It also
makes the physical state contract part of KDA operation APIs, increasing
program-cache and maintenance surface.

## Recommendation and required break-even work

### Immediate recommendation

Keep `momcilo/kda-nd-cache` as an experiment; do not merge the current form.
Prefer the explicit adapter unless direct ND is reworked and revalidated. Do
not weaken the SP4 performance gate.

### If direct ND remains strategically desirable

Both of these changes are required before reconsideration:

1. **Local recurrent ND store.** Add a KDA-owned layout-aware writer for the
   already-local final affine carry. A natural home is the final affine-prefix
   operation or a small KDA state-store operation. It must write each rank's
   identical carry locally; it must not use a collective.
2. **Native convolution halo assembly.** Teach KDA's halo/QKV path to read rank
   zero's initial history directly from ND pages while sourcing later-rank
   histories from gathered interleaved tails. Do not concatenate the ND input
   through the generic three-row redistribution path.

The SP4 acceptance target should be end-to-end and stricter than merely passing
the old layer-only gate:

- direct ND must beat latest main plus the optimized adapter;
- use the appropriate export-only or round-trip lifecycle;
- require a material margin beyond run-to-run noise, not a sub-10-microsecond
  tie; and
- keep all current PCC and exact physical-layout checks.

The hybrid ablation indicates that removing both bridges can reach parity with
main, but parity alone still does not automatically justify 28 touched files.
The final decision should include maintenance value (one canonical state and no
caller adapter) in addition to a demonstrated end-to-end latency win.

## Reproduction and evidence

Production/profile command, with `-k SP4xTP2` for the focused runs:

```bash
KIMI_K3_CKPT=/localdev/mvasilijevic/.cache/Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
KDA_PERF_SKU=bh_loudbox PYTHONPATH=/tmp \
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py::test_kimi_k3_layer_1_perf \
  --tt-arch blackhole -p kda_force_blackhole -k SP4xTP2 -q -s
```

Local raw logs, intentionally ignored by Git:

- `artifacts/kda-nd-cache/main-perf.log`
- `artifacts/kda-nd-cache/nd-perf-final.log`
- `artifacts/kda-nd-cache/sp4-attribution.log`
- `artifacts/kda-nd-cache/sp4-attribution-repeat.log`
- `artifacts/kda-nd-cache/sp4-no-recurrent-broadcast.log`
- `artifacts/kda-nd-cache/sp4-native-input-no-broadcast.log`

Committed supporting artifacts:

- `artifacts/kda-nd-cache/kda-nd-cache-report.html`
- `specs/kda-nd-cache-design.md`
- `specs/kda-nd-cache-dev-spec.md`

Temporary profiling and placement ablations were restored. The only source
change introduced by this investigation is this report.
