---
name: perf-counter-family
description: Add, extend, or review a hardware performance counter family (Tensix banks, RISC-V performance monitor, NoC interface or router, DRAM, ethernet, or a new one) across the tt-llk counter headers, the tt_llk_perf decoders, the tt-metal profiler hooks, the LLK harness and docs/performance_counters. Use when introducing a counter capture, a marker payload, a decode or metric, or a run-time option for counters, or when reviewing such a PR.
user_invocable: true
---

# Perf Counter Family

## Goal

A counter family should reuse what the repository already defines, capture
exactly the window it claims, survive its own counter width, and ship with a
consumer that uses it. This skill is a checklist for the author and the
reviewer; the family's doc page follows [docs-writing](../docs-writing/SKILL.md)
and its comments follow [code-comments](../code-comments/SKILL.md).

## Where a family lives

| Part | Location |
|------|----------|
| Inventory, registers, capture steps | `tools/include/perf_counters/` (`inventory.h`, `registers.h`, `hw.h` and family `*_hw.h` headers) |
| Host decode and metrics | `tools/python/tt_llk_perf/` |
| tt-metal emission | `tt_metal/hw/inc/internal/profiler/` and the firmware hooks that call it |
| LLK harness capture | `tests/helpers/include/counters.h`, read back by `tests/python_tests/helpers/counters.py` |
| Docs | `docs/performance_counters/` |

## 1. Reuse before adding

- Use the existing register and address definitions: `noc_parameters.h` for
  NIU registers, `eth_fw_api.h` and `internal/ethernet/tt_eth_ss_regs.h` for
  ethernet, the L1 address maps for firmware data. Do not redeclare them in
  a counter header and then add a test that they match.
- When architectures share a table and differ only in a few constants (base
  address, number of NoCs, word size), keep one table and per-architecture
  constants. Compose a superset table from the base one instead of copying it.
- Put the capture sequence (arm, reference, sample, delta) in the shared
  `*_hw.h` header so tt-metal and the harness call the same code.
- One helper for writing a profiler marker (`TS_DATA` header plus payload),
  one helper for parsing a comma-separated option of 32-bit values. Do not
  copy either per family.
- Generate repeated accessors (CSR readers, select and zero pairs) from one
  template instead of copies.
- Delete helpers nothing calls.

## 2. Host mirrors the device without copies

- Python parses layout constants (word offsets, block sizes, flags) from the
  device header, as `counters.py` already does for some constants. Do not
  hand-copy them and add a drift test.
- Describe the marker payload once, in the emitting header, and link to it
  from the decoder docstring and the doc page.
- Drop hardware-free tests that only restate header constants through a regex;
  they fail on every change and catch nothing else.

## 3. Correctness checklist

- **Width and wrap.** State each counter's width and the time it takes to wrap
  at the clock or line rate it counts. Use the units the ISA gives (a byte
  counter wraps far sooner than a word counter). Either detect the wrap, refuse
  windows that can wrap, or document the limit where the option is described.
- **Window bounds.**
  - The window covers exactly what its label says, for example the kernel call
    and not the idle wait, dispatch writes or bookkeeping around it.
  - It covers the same thing on every architecture and core type.
  - A launch-wide capture is not reported under a zone name.
- **Profiler buffer.** Reserve room for the markers once, at `begin()`. Do not
  add headroom that `bufferHasRoom` already accounts for, and do not write
  markers after the kernel without a reservation.
- **Option conflicts.** Reject incompatible profiler modes (streaming,
  trace-only) on the host instead of compiling the capture out silently.
- **Keys.** Decode keys include the device, so a mesh run does not merge chips.
- **Metrics.** Never add traffic in opposite directions into one number; name
  metrics by direction (in versus out).
- **Labels.** Two counters armed with the same select or label keep separate
  results; they must not overwrite each other.
- **Arithmetic.** Watch unsigned underflow in counts and indices for empty
  selections.

## 4. Perturbation

- State what the capture itself adds to the counts: host polls, the reading
  core's own register reads, instruction fetches from L1.
- If enabling one family changes another family's results (for example, extra
  BRISC reads moving Tensix L1 counters), warn when both are enabled, not only
  in the doc.
- Changes to the shared L1 counter layout affect every counters build; call
  them out in the PR description.

## 5. Scope

- A family ships with a consumer that is actually wired (ops report, Tracy
  flow, or harness CSV) and at least one test that exercises real traffic.
  A decoder with no caller, or a capture whose only signal is the test's own
  polling, is a candidate to defer.
- Keep adjacent work (a new core type, an unrelated test fixture) in its own
  change.

## 6. ISA cross-check

- Check register semantics against tt-isa-documentation: units, base
  addresses, numbering of virtual channels, queues or ports, and what one
  increment means.
- When the code and the ISA disagree, or the ISA is silent (for example a CSR
  list for a core type), record it as an open question with a follow-up to the
  ISA owners. Do not encode one side as fact without saying so.

## Review checklist

- [ ] Existing register headers reused; no duplicated per-arch tables.
- [ ] Capture steps shared through `*_hw.h`; one marker writer, one option parser.
- [ ] Host parses device constants; no constant-restating tests.
- [ ] Payload layout described in one place.
- [ ] Counter width and wrap time stated and handled.
- [ ] Window matches its label on every architecture.
- [ ] Buffer room reserved once; incompatible options rejected.
- [ ] Keys include the device; metrics are directional; duplicate selects kept apart.
- [ ] Perturbation stated; cross-family effects warned.
- [ ] A wired consumer and a real-traffic test exist.
- [ ] ISA conflicts and gaps raised as follow-ups.
