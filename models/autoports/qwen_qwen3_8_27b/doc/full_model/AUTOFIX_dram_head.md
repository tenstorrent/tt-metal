# AutoFix: split-reader rows larger than a NoC packet

## Starting evidence

`watcher_final.log` aborts134 in the DRAM in1 reader at `NKFW`, with pending
NoC reads at kernel completion. The fresh xhigh diagnosis and independent
hypothesis check are recorded in `AUTOTRIAGE_dram_head.md`. The preserved
compiled reader submits17408bytes and increments the global software read
counter once. Blackhole's packet limit is16384bytes.

## Hypothesis and repair

The second template parameter to `Noc::async_read` is a maximum-size promise,
not a requested chunk size. Supplying `NOC_MAX_BURST_SIZE` forces the one-packet
helper even for a larger runtime row. The existing per-transaction barriers
drain each issued block, but cannot repair the erroneous global packet accounting.
Independent source review checked the block/tag ledger for1..256 blocks and
found no missing final barrier.

The minimal fix makes row byte size constexpr and passes that actual size as
the template maximum. Large rows use the existing any-length helper, emitting
16384+1024-byte reads under the existing transaction tag. Small rows retain
the one-packet path. No precision, shape, placement, trace or barrier fallback
was added, and no watcher check was disabled.

## Verification

- Native target build and runtime installation pass; exact commands and
  Docker-wrapper limitation are in `work_log.md`. JIT compilation of the changed
  RISC-V reader is exercised by the watcher tests, not inferred from the host build.
- `native_wide_watcher.log`:3 passed,16.81s. The added BF8 N8192/K5120 row
  covers 1/2/3 readers on TP4 and repeated address rebinding in eager mode.
  The separate `watcher_fixed` generator run covers traced replay.
- `watcher_fixed.json` plus `.exit_status`0: original reduced full-generator
  mixed-slot trigger passes with all watcher and Ethernet checks enabled.
  It also passes exact logits under physical page remapping, stable feedback,
  inactive state, greedy/sample alternation, reset and host compatibility.
- `native_fixed.sha256` identifies the installed extensions and changed kernel.
- One bounded reset after the original abort restored the devices; explicit
  post-abort health evidence and clean post-reset ring open/close are in `triage/`.

The source hypothesis is verified and the original watcher defect is fixed.
Subsequent complete64-layer correctness, capacity and performance remain separate
stage gates, recorded in the final README/work log rather than inferred here.
