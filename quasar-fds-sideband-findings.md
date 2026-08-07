# Quasar fast dispatch signals: the sideband carries nothing

Findings from bring-up debugging of the Quasar FDS go/done sideband. Written so that someone with
no prior context can pick this up, understand what is already known, and continue without
repeating work.

Status as of the last run: **the register interface is fully working and characterised; no signal
has ever crossed between the dispatch-engine tile and a Tensix tile, in either direction, from any
processor that software can reach.**

---

## 1. Summary

Quasar has a sideband of dedicated wires between dispatch-engine tiles and Tensix tiles, intended
to replace a NOC atomic in the worker-completion path. Software drives it through a small register
block reached by custom processor instructions.

A bring-up test drives the handshake end to end: the dispatch engine sends a "go", a worker sends
back a "done". Neither ever arrives.

What the debugging established:

- The register block is real, correctly mapped, correctly addressed, and fully configurable. This
  is not an instrumentation problem.
- Both sides' receive logic is instantiated and actively evaluating its lanes — it reports every
  lane as idle, which is a different thing from reporting nothing.
- Both sides' transmit registers hold the values written to them.
- Nothing passes between the two, under any configuration, from any reachable processor, on either
  worker tile, in either direction.

The most likely remaining explanation is that the lanes are not connected in this simulator
configuration. That is a hardware-configuration question, not a software one. Every placement and
configuration avenue reachable from software has been tried; section 8 lists the few minor ones
that remain, none of them promising.

---

## 2. How to reproduce

```bash
export TT_METAL_SIMULATOR=<path to the Quasar simulator>

TT_METAL_DPRINT_CORES=all \
TT_METAL_DPRINT_DISPATCH_CORES=all \
TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build/test/tt_metal/unit_tests_legacy \
  --gtest_filter='QuasarMeshDeviceSingleCardFixture.DispatchEngineSingleWorker'
```

The test skips itself unless the simulator is enabled, slow dispatch is on, and native
dispatch-engine cores are in use. A run takes roughly 100 seconds, of which about 35 is simulator
startup.

Emulation will not work: it compiles kernels for the host, where the custom instructions the FDS
accessors use do not exist. The test checks `get_simulator_enabled()` rather than
`is_simulator_or_emulated()` for exactly this reason.

### Files

| Path | Role |
|---|---|
| `tests/tt_metal/tt_metal/test_quasar_dispatch_engines.cpp` | Host test |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_signal.cpp` | Dispatch-engine kernel: sends go, waits for done |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_signal.cpp` | Data-movement worker: waits for go, drives done |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_tensix_engine_signal.cpp` | Same, for the Tensix engine processors |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_probes.h` | Shared register-interface probes |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_signal_status.h` | Status-word layout shared with the host |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/fds_functions.hpp` | Vendored FDS accessor shim |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/rocc_instructions.hpp` | The custom-instruction macros |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/meta/fds_registers/` | Generated register headers |

---

## 3. Background

### What the hardware is meant to do

A dispatch engine writes a group id into an outbound register, which drives a "go" onto lanes
reaching every Tensix engine. A Tensix engine writes the same group id into its own outbound
register, driving a "done" back. The dispatch side counts how many enabled engines have signalled
a given group.

Both directions are **held levels**, not pulses. A value written to an outbound register stays
asserted until overwritten. This matters for debugging: there is no edge to miss, so a receiver
can be reconfigured and retried repeatedly against a signal that is still being driven.

### How software reaches it

`FDS_INTF_READ(addr)` and `FDS_INTF_WRITE(addr, val)` in `rocc_instructions.hpp:46-57` are custom
coprocessor instructions — custom-2 opcode, function code 36 — carrying a register address and a
value.

**Read the following carefully, because it invalidates naive testing.** The read macro declares
its destination register *uninitialised* and constrains it as an output. If the instruction does
nothing, the read returns whatever the compiler last left in that register. That value is very
often the one just written, so a write-then-read that appears to succeed proves nothing on its
own. Several early conclusions in this investigation were wrong because of this. Section 7
describes the probes that defeat it.

### Simulator topology

From `tt_metal/third_party/umd/tests/soc_descs/quasar_simulation_2x3.yaml`:

```
grid: 2 x 3
functional_workers: [0-1, 1-1]     # two Tensix tiles
dispatch:           [1-2]          # one dispatch-engine tile
dram:               [1-0, 0-0]
```

Both Tensix tiles run firmware and both are usable under slow dispatch. (The *fast*-dispatch
descriptor has only one usable worker; that constraint does not apply here.)

Each Tensix cluster has 8 data-movement cores and 4 Tensix engines of 4 TRISCs each. Hardware
thread indices, from `tt_metal/hw/inc/internal/hw_thread.h`: 0-7 are the data-movement cores, 8-23
are the TRISCs, four per engine.

`temp_quasar_api.hpp:34-38` reserves data-movement cores 0 and 1 on worker clusters, so user
kernels land on 2 through 7. Dispatch-engine cores have no such reservation.

---

## 4. Established facts about the register interface

Every item here is a measurement, not an inference, and each was reproduced across processors and
both tiles.

**Only nine address bits are decoded** — a 512-byte window. Everything above `0x1FF` aliases back
down. The dispatch map's documented `0x200` base is *stripped* by the decode, so on a
dispatch-engine core, probe address `0x000` reaches documented address `0x200`.

Consequence: `_REG_ADDR` and `_REG_OFFSET` forms of a dispatch-side address are interchangeable in
practice. A long-standing claim that `fds_clear_neo_status` is broken for using the offset form is
**false** — both reach the same register.

**There is one register block per tile, shared by every data-movement core on it**, and the block
matches the tile type: a dispatch-engine tile has the dispatch map, a Tensix tile the engine map.

This was measured directly rather than inferred. Every processor stamped its own index into one
otherwise-unused register early in the run and read that register back at the end, by which time
all had written. On the dispatch tile, all eight processors read `0xa4` — processor 4's stamp. On
each worker tile, all six read `0xa2` — processor 2's stamp. In every case the processor that
wrote last read its own value and every other processor read that same foreign value.

Two consequences. Concurrent access from two processors on a tile is a hardware hazard, not a
policy question: they will overwrite each other's configuration and consume each other's status.
And any experiment that sweeps processors on one tile is measuring the same registers repeatedly —
see section 7.

**There is no per-processor banking.** Addresses at `0x1000` and `0x2000` alias rather than
selecting a bank. The `CORE_OFFSET 0x1000` constant in `fds_functions.hpp`, commented "offset
between mhartid cores", is *not* a stride for this interface. There is no other processor's block
to reach: every processor on a tile addresses the same one.

**Field widths match the generated headers exactly.** Four bits for inbox and outbox registers,
three for the engine-side group enable and status, eight for count thresholds and counts,
thirty-two for the filter and the dispatch-side enable and status.

**Status and count registers are read-only.** Writes to them are ignored.

**Status is not masked by the enable register.** The dispatch-side enable for group 0 held zero
while its status read all ones.

**Group 0 is the idle value.** A lane presenting nothing reads as group 0, which makes the
group-0 status register a live map of *quiet* lanes. The dispatch engine reports all 32 done lanes
quiet; every engine-side block reports all 3 go lanes quiet. This is the most useful diagnostic
register available — any lane that starts carrying a real value drops out of that map, which would
identify the wire index at the same time.

**The interface is reachable only from data-movement cores.** Data-movement cores build as
coprocessor-equipped parts and Tensix engines do not:

```
tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp:348
  processor_class == DM ? "-mcpu=tt-qsr64-rocc " : "-mcpu=tt-qsr32-tensix "
```

On all four TRISC roles across both tiles, two registers with *different* declared widths returned
an *identical* value, and that value advanced by exactly 20 from one processor's report to the
next — an untouched destination register still holding a print-buffer offset. The same probe on
every data-movement core returns the declared field widths.

Consequence for any future design: an FDS owner must be a data-movement core, and worker
completion cannot be signalled from a compute kernel.

---

## 5. Established facts about the sideband

- The dispatch engine wrote group 1 to its outbox and read back 1, held for entire runs.
- Twelve data-movement blocks across both worker tiles drove done and held it.
- The dispatch engine's quiet-lane map never changed from all 32.
- All 32 of its raw inbox registers stayed zero; its group count stayed zero.
- Every worker block's 3 raw inbox registers stayed zero; group status stayed zero.

The raw inbox registers sit *before* all aggregation, so no group, enable or threshold setting can
explain them staying at zero.

Both sides' receive logic is instantiated and evaluating lanes — that is what the quiet-lane maps
mean. Both sides' transmit registers hold what is written. Nothing passes between them.

---

## 6. What has been ruled out

| Candidate | How it was eliminated |
|---|---|
| Threshold misconfiguration | The test originally required 8 done signals with only one worker present — a real bug, now fixed to 1. Fixing it changed nothing, and the raw inboxes bypass counting entirely. |
| Timing races, lost edges | Both signals are held rather than pulsed, with tens of seconds of simulated overlap between transmitter and receiver. |
| The test clobbering its own inbox | An early probe wrote zero into the register the go arrives in. Removed. Behaviour unchanged. |
| Enable masks, group id choice | Raw inbox registers sit before aggregation, so masks cannot gate them. |
| Deglitch filter | Swept 0, 1, 2, 8 and 64 on both sides against held signals. Reset value is 0. No effect at any setting. |
| Stale auto-dispatch or interrupt state | Both read zero before being explicitly zeroed. |
| Wrong processor, worker side | All 12 user data-movement cores across both worker tiles, simultaneously. |
| Wrong processor, dispatch side | All 8 data-movement cores on the dispatch tile, each sending its own go. |
| Wrong kind of processor | Tensix engine processors cannot reach the interface at all (section 4). |
| "The registers are plain storage" | Disproved by field-width truncation — real registers drop out-of-field bits. |
| "The read instruction is being ignored" | Disproved on data-movement cores by the same truncation probe, and by a cross-address probe. |
| Misaddressed reads | Cross-address probe: write two values to two registers, read the first back. Returns the first. |
| Instrument error generally | 34 probe points across both core types fit one address-decode model exactly. |

---

## 7. False leads and corrections

**Read this section before forming a hypothesis.** Each of these looked convincing and was wrong.

**"The inbox is plain storage with no logic behind it."** The first probe wrote a sentinel to a
hardware-driven input register and read it back unchanged, concluding the registers were dumb
storage. Wrong: the sentinel fitted inside the register's four-bit field, so nothing distinguished
storage from a real register. The fix is to write a value *wider* than the field.

**"Address `0x400` is unmapped and correctly rejected the write."** It kept four bits of it.
`0x400` aliases to offset `0x000`. The probe's verdict line was backwards.

**"`fds_clear_neo_status` writes into the wrong register map."** It does not — see section 4.

**"The Tensix engine experiment tested whether engines are the endpoint."** It did not. Those
processors cannot reach the interface, so every value they reported was stale register contents.
Any experiment involving TRISCs and FDS is measuring nothing until that changes.

**"The deglitch filter is the likely culprit."** It sits exactly on the receive path, its reset
value is 0, and the test had been overriding it with 1 from the beginning — a filter rejecting an
assertion is indistinguishable from an idle lane. It was a good hypothesis and it is dead: swept
across five settings on both sides with no effect.

**"Each processor has its own register block."** Stated as established for several runs, and
wrong. The evidence behind it was only that there is no banking stride at `0x1000`, which shows a
processor cannot *address* a different block — equally true if there is only one. The stamp test
in section 4 settled it: one block per tile.

This matters beyond bookkeeping, because it means **the processor sweeps were uninformative by
construction.** Six worker processors, then eight dispatch processors, all returned byte-identical
results — not because the lanes are wired identically to each, but because every processor was
reading and writing the same registers. Do not re-run a placement sweep on one tile expecting it
to distinguish anything; it cannot.

---

## 8. Coverage gaps — things not yet tried

Note that two gaps listed in earlier versions of this document are now closed, and one of them
turned out to be unanswerable in the form proposed:

- The dispatch side has been swept across all eight of its data-movement cores. Negative, and — as
  the stamp test then showed — it could not have been anything else, because those eight
  processors share one register block.
- Data-movement cores 0 and 1 on the worker tiles, reserved by the metal API and therefore never
  running a test kernel, no longer matter. They address the same block as cores 2 through 7, which
  has been exercised thoroughly.

**Auto-dispatch mode was never enabled.** The block has `AUTO_DISPATCH_EN`, `CYCLE_COUNT`,
`OUTBOX_ADDRESS` and `FIFO_FULL` registers, and the shim's `fds_go`/`fds_done` take an `ad_enable`
argument. The test has only ever used the direct path with auto-dispatch off. Reading the shim,
auto-dispatch appears to add flow control around the same outbox write rather than being a
different transport, so this is low-probability — but it is untried.

**Only group id 1 was ever used.** Group 0 is the idle value and effectively reserved. Groups 2
through 15 were never tried. Very unlikely to matter, trivial to sweep.

**Persistence across device reset was never checked** — whether configuration survives, and
whether a stale state from a previous run can affect a later one.

---

## 9. Recommended next steps

In order.

**1. Put the configuration questions to whoever owns the RTL and simulator build.** This is now the
first step, and the one most likely to actually resolve the bug. Every placement and configuration
avenue reachable from software has been exhausted:

- Does the 2x3 simulation configuration instantiate and connect the FDS lanes between the dispatch
  tile at `1-2` and the Tensix tiles at `0-1` and `1-1`, or are they tied off?
- If not, is there a configuration or build option that does?
- If they are connected, is there a block-level enable *outside* the two register maps — a
  tile-level or interface-level register we have no visibility into? Neither map contains anything
  resembling one, and this is the one possibility software cannot exclude by itself.
- What does each of the 32 done lanes correspond to? The block is named for the Tensix engine,
  which suggested one lane per engine across eight tiles. But there is one block per *tile* with a
  single outbox register in it, so a tile appears to drive one lane, not four — which would make 32
  lanes mean up to 32 tiles. Someone with the specification should say which reading is right,
  because it determines the worker-to-bit mapping the dispatch design depends on.

**2. Ask for the authoritative register specification.** The generated headers give addresses,
widths and reset values, but no access types and no behavioural description. Several questions
that took multiple simulator runs to answer would have been immediate with the source description:
whether status latches, what the deglitch threshold units are, whether the outbox drives a level
or a pulse, and whether a block enable exists.

**3. If a configuration with connected lanes becomes available**, the existing test should show it
immediately — the dispatch engine's quiet-lane map is the thing to watch, and a cleared bit names
the wire index as well as proving transport.

**4. Do not invest further in TRISC-side experiments** unless documentation shows a different
instruction encoding reaches FDS from those processors.

---

## 10. Reference

### Register maps

Both maps live in the same address space reached by the same instructions. Only bits `[8:0]` are
decoded.

`tt_fds_tensixneo` — present on Tensix data-movement cores. Base `0x000`, size `0x128`.

| Offset | Register | Count | Field width |
|---|---|---|---|
| `0x000` | `DISPATCH_TO_TENSIX_n` (go inbox) | 3 | 4 bits |
| `0x00C` | `TENSIX_TO_DISPATCH` (done outbox) | 1 | 4 bits |
| `0x010` | `FILTER_COUNT_THRESHOLD` | 1 | 32 bits |
| `0x014` | `GROUPID_STATUS_n` | 16 | 3 bits, read-only |
| `0x054` | `GROUPID_ENABLE_n` | 16 | 3 bits |
| `0x094` | `GROUPID_COUNT_THRESHOLD_n` | 16 | 8 bits |
| `0x0D4` | `GROUPID_COUNT_n` | 16 | 8 bits, read-only |
| `0x114`+ | `INTERRUPT_ENABLE`, `AUTO_DISPATCH_*` | | |

`tt_fds_dispatch` — present on dispatch-engine cores. Base `0x200`, size `0x19C`. The `0x200` base
is stripped by the decode.

| Address | Register | Count | Field width |
|---|---|---|---|
| `0x200` | `DISPATCH_TO_TENSIX` (go outbox) | 1 | 4 bits |
| `0x204` | `TENSIX_TO_DISPATCH_n` (done inbox) | 32 | 4 bits |
| `0x284` | `FILTER_COUNT_THRESHOLD` | 1 | 32 bits |
| `0x288` | `GROUPID_STATUS_n` | 16 | 32 bits, read-only |
| `0x2C8` | `GROUPID_ENABLE_n` | 16 | 32 bits |
| `0x308` | `GROUPID_COUNT_THRESHOLD_n` | 16 | 8 bits |
| `0x348` | `GROUPID_COUNT_n` | 16 | 8 bits, read-only |
| `0x388`+ | `INTERRUPT_ENABLE`, `AUTO_DISPATCH_*` | | |

All reset values are zero.

### Probes worth reusing

These are in `quasar_fds_probes.h` and are the techniques that made the difference.

**Field-width truncation.** Write `0xFFFFFFFF`, read back, restore. A real register returns its
field mask; storage or a dead instruction returns something else. Choose registers with *different*
widths so the answers are distinguishable from each other.

**Cross-address.** Write two different values to two different registers, read the first back.
A real read returns the first value; a dead instruction returns the second or unrelated data.
This is what defeats the uninitialised-destination-register problem.

**Address-map sweep.** Truncation across a spread of addresses at once. The pattern of widths
reveals how many address bits are decoded, which maps a processor hosts, and whether the file
repeats at a banking stride.

**Per-processor stamps.** Every processor writes a value carrying its own index into one otherwise
unused register early, and reads that register back at the end, once all have certainly written.
A processor reading a value that is not its own is reading a register a neighbour wrote, which is
what proved the block is shared per tile. Note the asymmetry when interpreting it: reading a
foreign stamp is proof of sharing, but reading your own proves nothing, since that is equally what
a private block gives you and what a shared block gives to whichever processor wrote last.

**Decoupling the directions.** The worker drives done part way through its wait whether or not a
go arrived, so a missing go cannot hide the state of the done direction. Before this change, every
run tested only one direction.

### Two implementation details that will bite

**Status blocks must be padded to cache lines.** Data-movement kernels write status through the
cached path and flush; TRISCs write through the uncached alias. With tightly packed blocks, two
processors share a line and one can write back over the other's slots — which looks exactly like
"that processor never ran". The shared header pads each processor's block to 32 words.

**Print output from many processors is not chronologically ordered.** Each processor has its own
buffer and the host drains them independently, so the interleaving in a log is not a timeline.
Several early hypotheses about ordering were unfalsifiable for this reason. Design experiments so
the conclusion does not depend on the relative order of lines from different processors.
