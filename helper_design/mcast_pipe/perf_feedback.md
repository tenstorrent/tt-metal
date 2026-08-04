# `mcast_pipe` performance feedback

This is the intake log for performance risks introduced or exposed by
`mcast_pipe` migrations. An entry records a required comparison; it is not
evidence of a regression until the relevant kernels are profiled against their
pre-migration baseline.

## Status values

- **Open** — the performance comparison has not been completed.
- **No regression** — the comparison found no material regression; preserve the
  measurements and configurations.
- **Regression** — a material regression was measured and needs follow-up.
- **Resolved** — a measured regression was addressed and the fix was verified.

## PERF-001 — Profile GroupNorm's fixed three-rectangle multicast path

- **Date:** 2026-08-04
- **Status:** Open — no regression for the measured rectangular SDXL shape;
  wrapped-shape coverage remains
- **Migration commit:** `09afe010fb3` (`Apply mcast host helpers to sharded groupnorm v2`)
- **Host file:**
  `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp`
- **Sender kernels:** `reader_mcast_sender_unary_sharded_gn_v2.cpp` and
  `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp`

### Performance risk

Before the migration, the sender always constructed and sent the middle
rectangle, but constructed and sent the first and last edge rectangles only
when `has_mcast_first_group` or `has_mcast_last_group` was true.

The migrated factory emits one fixed wire containing three `Mcast2D` blocks for
every group: middle, first edge, and last edge. Missing edge rectangles are
represented as sender-only singleton rectangles. Consequently, both sender
kernels now construct three `SenderPipe`s and call all three `send()` methods on
every reduction iteration:

```cpp
mid_pipe.send(l1_read_addr_ex, l1_read_addr_ex, num_bytes_read);
first_pipe.send(l1_read_addr_ex, l1_read_addr_ex, num_bytes_read);
last_pipe.send(l1_read_addr_ex, l1_read_addr_ex, num_bytes_read);
```

For an absent edge, `SenderPipe::send()` recognizes the singleton as
degenerate, enters the local-copy path, observes `src_l1 == dst_l1`, and returns
without issuing NoC traffic. It is therefore behaviorally a no-op, but it still
adds pipe construction and a call/branch path. The common rectangular-group
case changed from one send call per reduction value to three, with two
degenerate calls. A group with one edge rectangle changed from two calls to
three. A group requiring all three rectangles does not add send calls, and may
benefit from pipe construction having moved out of the hot loop.

This needs a performance check; correctness coverage alone is not sufficient
to close the migration.

### Required comparison

Profile the migration against its parent using identical device, build, input,
and profiler settings. Record at least:

- legacy and Welford GroupNorm v2;
- a rectangular group, where both edge rectangles are absent;
- a wrapped group with one edge rectangle, if a supported configuration reaches
  it;
- a wrapped group with both edge rectangles, if a supported configuration
  reaches it;
- the sender data-movement kernel's `Kernel duration (ns)` and end-to-end
  operation duration.

Start with the exact legacy and Welford 8x4 nodes used to validate the migration,
then select configurations that deliberately cover each rectangle shape. Do not
average together shape classes, because the number of newly introduced
degenerate calls differs between them.

No acceptable regression threshold has been agreed yet. Preserve raw results
and variance so a threshold can be chosen from evidence.

### 2026-08-04 measurement

The first model-shape comparison used the real-time device profiler on a single
Blackhole p100a at AICLK 800. Each snapshot was rebuilt, each case had three
warm-up executions, and the reported value is the median of 20 device-program
duration records. The comparison was between baseline `4a1d6a97ca9` and
migrated production snapshot `28356d43846`.

| Case | Baseline median (ns) | Migrated median (ns) | Delta |
| --- | ---: | ---: | ---: |
| SDXL `(1, 1920, 32, 32)`, legacy | 48,593.704 | 48,714.444 | +0.248% |
| SDXL `(1, 1920, 32, 32)`, Welford | 261,695.556 | 260,426.667 | -0.485% |

Both cases use an 8x8 block-sharded layout with rectangular column groups. The
new three-pipe path therefore executes two degenerate sender calls. No material
regression was observed for this shape. This closes the immediate model-shape
risk but not the requested wrapped-group coverage above.

### Follow-up if a regression is measured

Retain the helper-owned coordinate conversion and fixed, chainable argument
layout, but investigate a way for the sender to skip absent rectangle sends.
Possible formulations include per-core rectangle-presence metadata or a helper
operation that makes optional destinations explicit. Choose the formulation
only after measuring where the cost occurs; do not assume that host argument
size, pipe construction, and degenerate calls contribute equally.

## PERF-002 — Investigate the Conv height-sharded sender/receiver regression

- **Date:** 2026-08-04
- **Status:** Regression
- **Affected case:** SDXL VAE height-sharded Conv (`vae_sdxl_hs`)
- **Migrated kernels:**
  `reader_writer_tiled_out_1d_mcast_{sender,receiver}_conv_weights_tiled_col_to_rm_blocks.cpp`

The baseline-to-migrated comparison measured 28,005.649 ns versus 28,719.126
ns, a +2.548% change. A reverse-order baseline rerun measured 27,955.899 ns,
making the repeated delta +2.730%. Within-run standard deviation for the
migrated result was 47.139 ns, so the shift is substantially larger than the
observed run variance.

Commit isolation found:

| Snapshot | Median (ns) | Interpretation |
| --- | ---: | --- |
| `4a1d6a97ca9` reverse rerun | 27,955.899 | pre-migration baseline |
| `59e75d6fc3a` | 28,748.347 | first complete sender/receiver kernel conversion |
| `9faa809ea5a` | 28,716.956 | loopback acknowledgement safety fix |
| `7aa395aac56` | 28,722.470 | immediately before host-helper conversion |
| `28356d43846` | 28,719.126 | final migrated production snapshot |

The full regression is present at the first complete kernel conversion. The
loopback fix and host-helper conversion are neutral within noise. The earlier
partial checkpoint `acd84d7f3fc` hung while its migrated receiver waited for a
peer from the not-yet-complete conversion; the safe test runner captured triage
and reset the device. It is not evidence of a hang in either complete snapshot.

### Root-cause ablation

The ops-codegen implementer/performance-measure workflow was applied to the
exact SDXL VAE case. Each temporary variant preserved the multicast payload,
receiver handshake, CB synchronization, and loop counts. Only one helper
component was changed at a time. Results are medians of 20 real-time-profiler
device-program records after three warmups.

| Variant | Median (ns) | Delta from reverse baseline |
| --- | ---: | ---: |
| Raw pre-migration path, reverse rerun | 27,955.899 | baseline |
| Migrated `SenderPipe` | 28,719.126 | +763.227 ns / +2.730% |
| Skip per-send fence | 28,207.679 | +251.780 ns / +0.901% |
| Skip fence and inline the complete send hot path | 28,022.482 | +66.583 ns / +0.238% |
| Equivalent straight-line sender, no fence | 28,004.016 | +48.117 ns / +0.172% |
| Inline hot path, retain per-send fence | 28,561.738 | +605.839 ns / +2.167% |
| Inline hot path, defer one barrier to kernel exit | 28,155.429 | +199.530 ns / +0.714% |

The regression comes from two concrete parts of `SenderPipe::send()`:

1. `fence_()` calls `noc_.async_writes_flushed()` after every weight or bias
   multicast. In the controlled inlined comparison this costs 539.256 ns,
   approximately 71% of the original regression. The raw Conv path linked data
   and signal but issued one `async_write_barrier()` only at kernel exit.
2. The compiler does not keep the complete generic send path inline. ELF symbol
   inspection showed separate hot-path helper bodies; forcing `send()`,
   `send_data_()`, and `signal_ready_()` inline recovered another 157--185 ns,
   depending on whether the fence was present. Forcing only `send()` inline was
   neutral because the compiler still outlined the two leaf operations.

Moving the receiver's flag reset, setting the sender's `VALID` flag only once,
and specializing away the degenerate/loopback branches were all neutral. The
receiver helper and those branches are not the source of this regression.

The deferred-fence plus inlined-hot-path candidate passed the exact nightly
SDXL VAE PCC case `(1, 4, 128, 128) -> 512`, with PCC 0.999932 against a 0.985
threshold. Its median was within +0.714% of the reverse baseline. This was an
ablation only; no production helper change was retained.

### Required fix design

Do not simply remove `fence_()` from `send()`: its current contract promises
that source L1 is safe to reuse when the call returns. Add an explicit batched
or deferred-completion formulation that keeps data-before-signal ordering but
lets a caller whose CB lifetime already protects the source issue one final
flush at the true reuse boundary. Possible shapes are `send_deferred()` plus
`flush()`, or a completion policy represented in the host wire and decoded by
`McastArgs` so it does not become an ad-hoc call-site knob.

Also make the complete hot send path reliably inline, or restructure it into a
single small inline implementation, then inspect the generated ELF to verify
that no `send_data_()` or `signal_ready_()` call boundaries remain. Re-profile
the SDXL VAE case and at least one loopback/rotating case before resolving this
entry.

## PERF-003 — Investigate the Conv width-sharded activation regression

- **Date:** 2026-08-04
- **Status:** Regression
- **Affected case:** SegFormer width-sharded Conv (`segformer_ws`)
- **Migrated kernel:** `activation_reader_width_sharded.cpp`
- **Migration commit:** `28356d43846`

The baseline-to-migrated comparison measured 38,080.741 ns versus 38,682.593
ns, a +1.580% change. A reverse-order baseline rerun produced +1.598%. The
immediate pre-migration parent `6ff7a20e67e` measured 38,013.031 ns, isolating
the change at `28356d43846` to +1.761%.

Next, compare the raw and helper activation-multicast instruction paths and use
a focused width-sharded case to determine which helper operation accounts for
the added device time.

## 2026-08-04 migration performance matrix

The same real-time-profiler protocol was applied to every migrated operation
selected for this check. Every baseline and migrated case completed and
produced exactly 20 matching device-program records. Kernel-source metadata was
checked to ensure the intended migrated kernels were in each program. The
focused deferred-fence candidate additionally ran the exact nightly PCC test,
as recorded under PERF-002.

| Operation and model/configuration | Baseline median (ns) | Migrated median (ns) | Delta |
| --- | ---: | ---: | ---: |
| GroupNorm SDXL 1920, legacy | 48,593.704 | 48,714.444 | +0.248% |
| GroupNorm SDXL 1920, Welford | 261,695.556 | 260,426.667 | -0.485% |
| Matmul 1D, SDXL ResNet 960x320 | 76,941.111 | 77,369.630 | +0.557% |
| Matmul 2D, SDXL FF GELU | 164,126.296 | 164,168.519 | +0.026% |
| Matmul 2D transpose multicast | 12,453.704 | 12,272.593 | -1.454% |
| Conv SDXL block-sharded | 53,855.185 | 53,846.667 | -0.016% |
| Conv SegFormer width-sharded | 38,080.741 | 38,682.593 | +1.580% |
| Sort single row, 524288 elements | 143,486,262.222 | 144,156,231.852 | +0.467% |
| Conv UNet height-sharded | 31,829.259 | 31,877.407 | +0.151% |
| Conv SDXL VAE height-sharded | 28,005.649 | 28,719.126 | +2.548% |

The reusable runner is
`tests/ttnn/perf_tests/operations/mcast/test_mcast_migration_realtime_perf.py`.
Raw JSON records are written under `generated/mcast_migration_rt/`; that
generated directory is not intended for source control.
