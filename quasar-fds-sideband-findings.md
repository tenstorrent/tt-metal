# Quasar fast dispatch signals: the sideband carries

Findings from bring-up of the Quasar FDS go/done sideband. Written so that someone with no prior
context can pick this up, understand what is already known, and continue without repeating work.

Status as of the run of 2026-08-20: **the register interface is fully working and characterised, and
both directions of the sideband now carry a signal.** A go crosses from a dispatch-engine tile to
every Tensix tile, a done crosses back, and the dispatch side aggregates dones across tiles into its
group counter.

This inverts the previous conclusion in this document, which was that nothing crossed in either
direction from any reachable processor. **What changed was the simulator build, not the software.**
The same tests against the previous build reported every lane idle throughout. Nothing about the
register interface findings changed; they were correct then and remain correct now.

---

## 1. Summary

Quasar has a sideband of dedicated wires between dispatch-engine tiles and Tensix tiles, intended to
replace a NOC atomic in the worker-completion path. Software drives it through a small register
block reached by custom processor instructions.

What the bring-up established:

- The register block is real, correctly mapped, correctly addressed, and fully configurable.
- A go sent by a dispatch engine reaches every worker tile and latches into the addressed group.
- A done sent by a worker reaches the dispatch engine and increments that group's counter.
- Dones aggregate: two worker tiles signalling the same group produce a count of two.
- The go wire is **shared across groups**. A worker in a group that was not signalled sees the
  signalled group's value in its raw inbox register and does not latch it. Groups are separated by
  the group filter, not by separate wires.
- A group that was configured but never signalled keeps a done count of zero.

What is not yet established is listed in section 8, and the largest items are the physical lane
mapping, the same-group two-epoch re-arm protocol, and a completion fence ordering worker writes
before a done.

---

## 2. How to reproduce

Runs so far have been on the **`emu-quasar-2x3_DISPATCH`** simulator config: two worker tiles and
native dispatch-engine cores. Its core descriptor is `tt_metal/core_descriptors/quasar_simulation_2x3_arch.yaml`
(compute grid `[0,1]`–`[1,1]`, no tensix dispatch cores) and its soc descriptor is
`tt_metal/third_party/umd/tests/soc_descs/quasar_simulation_2x3.yaml`. Record the config name with any
result: the lane mapping and worker counts below are properties of this config, not of Quasar.

```bash
export TT_METAL_SIMULATOR=<path to the emu-quasar-2x3_DISPATCH build>

TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build/test/tt_metal/unit_tests_legacy \
  --gtest_filter='QuasarMeshDeviceSingleCardFixture.DispatchEngine*'
```

Add `TT_METAL_DPRINT_CORES=all` and `TT_METAL_DPRINT_DISPATCH_CORES=all` if the kernels are carrying
instrumentation; the reduced kernels in the tree print nothing and report through L1 status words
that the host logs.

Six tests run: three that exercise the go/done handshake, one write-ordering harness for Gate 0b
(`DispatchEngineWriteOrdering`, which has not been run yet), and two that were the open Gate 0
experiments — `DispatchEngineSameGroupReArm` for the two-epoch re-arm cycle and `DispatchEngineLaneMap`
for the physical lane mapping. All five pass. The tests skip themselves unless the simulator is
enabled, slow dispatch is on, and native dispatch-engine cores are in use, and a skipped test still
exits zero — check for `SKIPPED` before believing a green run.

Timings from the 2026-08-20 run, which matter for interpreting a suite that looks slow:

| Phase | Wall clock |
|---|---|
| Simulator spawn and handshake, once per process | ~102 s |
| Device open and firmware init, per test | ~2 s |
| The handshake itself, when it succeeds | ~1 s |
| `SubDeviceGroupIsolation` | ~50 s |

The isolation test is slow because it asserts a negative: the unsignalled group's workers spin their
full poll budget waiting for a go that correctly never arrives. Nothing is wrong when that test is
the slowest one.

A note on the word "emulation", because the config name invites confusion. The `emu-quasar-*` names
are this repository's convention for **simulator** grid configs — see the header of
`tests/scripts/quasar/quasar_regression_tests.yaml`, which spells them out as "Simulator grid
(emu-quasar-<config>)" — and the UMD log lines for a run of one read `Instantiating RTL simulation
device`. Running on `emu-quasar-2x3_DISPATCH` is running the RTL simulation device, and these tests
work there.

What does *not* work is tt-metal's host-emulation mode, a separate thing that compiles kernels for the
host, where the custom instructions the FDS accessors use do not exist. That is why the tests check
`get_simulator_enabled()` rather than `is_simulator_or_emulated()`: the second would let host
emulation through, and every FDS register read there would return whatever the compiler left in the
destination register.

### Files

| Path | Role |
|---|---|
| `tests/tt_metal/tt_metal/test_quasar_dispatch_engines.cpp` | All five host tests |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_signal.cpp` | Dispatch engine: sends go, waits for dones |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_signal.cpp` | Worker: waits for go, drives done |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_rearm.cpp` | Dispatch engine: two epochs of one group |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_rearm.cpp` | Worker: answers twice, keyed on the go de-asserting |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_lane_map.cpp` | Dispatch engine: scans the raw done inboxes |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_drive_done.cpp` | Worker: drives its own group's done, unprompted |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_ordered_read.cpp` | Dispatch engine: reads a NOC payload on seeing done |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_ordered_write.cpp` | Worker: writes over the NOC, then signals |
| `tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_signal_status.h` | Status-word layout shared with the host |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/fds_functions.hpp` | Vendored FDS accessor shim |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/rocc_instructions.hpp` | The custom-instruction macros |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/meta/fds_registers/` | Generated register headers |

The instrumented version of the test — register probes, deglitch sweeps, raw-inbox dumps,
per-processor stamps, and a Tensix-engine worker variant — was removed once its questions were
answered. It is recoverable from commit `156af976a12`, which contains `quasar_fds_probes.h` and
`quasar_fds_tensix_engine_signal.cpp`. Section 10 describes which of those techniques are worth
restoring and why.

---

## 3. Background

### What the hardware does

A dispatch engine writes a group id into an outbound register, which drives a "go" onto lanes
reaching the Tensix tiles. A Tensix tile writes the same group id into its own outbound register,
driving a "done" back. The dispatch side counts how many enabled tiles have signalled a given group.

Both directions are **held levels**, not pulses. A value written to an outbound register stays
asserted until overwritten. This matters for debugging: there is no edge to miss, so a receiver can
be reconfigured and retried repeatedly against a signal that is still being driven. It also means a
stale value from a previous epoch is indistinguishable from a fresh one, which is what makes the
re-arm question in section 8 a real design problem rather than a formality.

### How software reaches it

`FDS_INTF_READ(addr)` and `FDS_INTF_WRITE(addr, val)` in `rocc_instructions.hpp:46-57` are custom
coprocessor instructions — custom-2 opcode, function code 36 — carrying a register address and a
value.

**Read the following carefully, because it invalidates naive testing.** The read macro declares its
destination register *uninitialised* and constrains it as an output. If the instruction does nothing,
the read returns whatever the compiler last left in that register. That value is very often the one
just written, so a write-then-read that appears to succeed proves nothing on its own. Several early
conclusions in this investigation were wrong because of this. Section 7 describes the probes that
defeat it, and section 5 describes which of the current results are and are not vulnerable to it.

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

Each Tensix cluster has 8 data-movement cores and 4 Tensix engines of 4 TRISCs each. Hardware thread
indices, from `tt_metal/hw/inc/internal/hw_thread.h`: 0-7 are the data-movement cores, 8-23 are the
TRISCs, four per engine.

`temp_quasar_api.hpp:34-38` reserves data-movement cores 0 and 1 on worker clusters, so user kernels
land on 2 through 7. Dispatch-engine cores have no such reservation.

---

## 4. Established facts about the register interface

Every item here is a measurement, not an inference, and each was reproduced across processors and
both tiles. All of it was established against the previous simulator build and none of it was
contradicted by the new one.

**Only nine address bits are decoded** — a 512-byte window. Everything above `0x1FF` aliases back
down. The dispatch map's documented `0x200` base is *stripped* by the decode, so on a
dispatch-engine core, probe address `0x000` reaches documented address `0x200`.

Consequence: `_REG_ADDR` and `_REG_OFFSET` forms of a dispatch-side address are interchangeable in
practice. A long-standing claim that `fds_clear_neo_status` is broken for using the offset form is
**false** — both reach the same register.

**There is one register block per tile, shared by every data-movement core on it**, and the block
matches the tile type: a dispatch-engine tile has the dispatch map, a Tensix tile the engine map.

This was measured directly rather than inferred. Every processor stamped its own index into one
otherwise-unused register early in the run and read that register back at the end, by which time all
had written. On the dispatch tile, all eight processors read `0xa4` — processor 4's stamp. On each
worker tile, all six read `0xa2` — processor 2's stamp. In every case the processor that wrote last
read its own value and every other processor read that same foreign value.

Two consequences. Concurrent access from two processors on a tile is a hardware hazard, not a policy
question: they will overwrite each other's configuration and consume each other's status. And any
experiment that sweeps processors on one tile is measuring the same registers repeatedly — see
section 7. The tests in the tree therefore run one data-movement core per tile and scale by adding
tiles.

**There is no per-processor banking.** Addresses at `0x1000` and `0x2000` alias rather than selecting
a bank. The `CORE_OFFSET 0x1000` constant in `fds_functions.hpp`, commented "offset between mhartid
cores", is *not* a stride for this interface. There is no other processor's block to reach: every
processor on a tile addresses the same one.

**Field widths match the generated headers exactly.** Four bits for inbox and outbox registers,
three for the engine-side group enable and status, eight for count thresholds and counts, thirty-two
for the filter and the dispatch-side enable and status.

**Status and count registers are read-only.** Writes to them are ignored.

**Status is not masked by the enable register.** The dispatch-side enable for group 0 held zero while
its status read all ones.

**Group 0 is the idle value.** A lane presenting nothing reads as group 0, which makes the group-0
status register a live map of *quiet* lanes. Any lane that starts carrying a real value drops out of
that map, which identifies the wire index at the same time — see section 9, where this is the basis
of the lane-mapping experiment.

**The interface is reachable only from data-movement cores.** Data-movement cores build as
coprocessor-equipped parts and Tensix engines do not:

```
tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp:348
  processor_class == DM ? "-mcpu=tt-qsr64-rocc " : "-mcpu=tt-qsr32-tensix "
```

On all four TRISC roles across both tiles, two registers with *different* declared widths returned an
*identical* value, and that value advanced by exactly 20 from one processor's report to the next — an
untouched destination register still holding a print-buffer offset. The same probe on every
data-movement core returns the declared field widths.

Consequence for any future design: an FDS owner must be a data-movement core, and worker completion
cannot be signalled from a compute kernel.

---

## 5. Established facts about the sideband

All from the 2026-08-20 run. `0x5a5a0002` is the completion marker in the status words and
`0x5a5a0003` the timeout marker.

**One worker, one dispatch engine.** The worker latched the go and answered; the dispatch engine
counted the done.

```
worker core 0-0 group 1: result=0x5a5a0002 observed_go=1 group_status=1
dispatch core 0-0:       result=0x5a5a0002 done count=1 quiet group count=0
```

**Two workers.** Both tiles latched the same go, both answered, and the dispatch side accumulated
both.

```
worker core 0-0 group 1: result=0x5a5a0002 observed_go=1 group_status=1
worker core 1-0 group 1: result=0x5a5a0002 observed_go=1 group_status=1
dispatch core 0-0:       done count=2 quiet group count=0
```

**A done count of two is the load-bearing measurement.** It is the one result that cannot be produced
by the uninitialised-destination-register problem described in section 3. A dead read instruction can
return a plausible 1 — the group id and the last value written to a register both happen to be 1 —
but it cannot return a 2 arising from two separate tiles. Any future result that rests on a count of
1 should be treated as weaker evidence than it looks.

**The go value on the wire is the group id.** Every worker reported `observed_go=1` for group 1.

**Group status asserts, and names the source instance.** The engine-side `GROUPID_STATUS` field is
three bits, one per dispatch instance. Workers read `1`, so dispatch instance 0 drove the go. This is
the first observation of that register under a real assertion; previously only idle lanes had ever
been seen.

**The go wire is shared across groups; the group filter is what separates them.** In the isolation
test, half the tiles took group 1 and half took group 2, and only group 1 was signalled:

```
worker core 0-0 group 1: result=0x5a5a0002 observed_go=1 group_status=1
worker core 1-0 group 2: result=0x5a5a0003 observed_go=1 group_status=0
dispatch core 0-0:       result=0x5a5a0002 done count=1 quiet group count=0
```

The group-2 tile saw group 1's value in a raw inbox register and did **not** latch it: its own group
status stayed zero and it timed out. So a foreign group's value appearing in a raw inbox is normal
and is not evidence of a leak; only the group status says whether a go was accepted. Any future test
of group isolation must distinguish those two things or it will report false failures.

**A configured but unsignalled group accumulates nothing.** Group 2 was configured on the dispatch
side with the same lane mask and a threshold of one, so a single stray done would have satisfied it.
Its count stayed at zero.

**No deglitch filtering is needed.** Both sides set the filter threshold to zero, its reset value,
and the handshake works. Other filter values are untested against a real assertion.

**Consecutive epochs of one group are distinguishable.** The full re-arm cycle works:

```
re-arm dispatch: round1 count=1 count after inbox clear=0 after settle=0 round2 count=1
re-arm worker:   round1 go=1 go de-assert seen=1 group status after de-assert=0 round2 go=1
```

Four things in one line. A sink-side inbox clear **holds against a live source** — the count went to
zero and stayed there while the worker was still driving that same done, which is the step the whole
protocol depends on. The go de-asserts when the dispatch engine writes the idle group, and the
worker's group status follows the wire back to zero, so a receiver can key an epoch boundary on
either. A second done for the same group produces **exactly one** new credit, not two and not zero.

The round-two count is 1, and results resting on a count of 1 are normally weak — but here the count
was read as zero twice immediately before, so the 1 is an observed transition rather than a value
that might always have been there.

**Each tile drives one lane, and the lanes are four apart.**

```
lane map: 2 of 2 lanes driving, group 0 status (idle lanes)=0xffffffee
  lane 0 carries group 1 -> core 0-0
  lane 4 carries group 2 -> core 1-0
```

`0xffffffee` has exactly bits 0 and 4 clear, so the group-0 status map and the raw inbox scan name
the same two lanes. Those are two readings taken on opposite sides of the aggregation logic, and
their agreement is what makes the mapping trustworthy rather than an artifact of one register.

Each tile appeared on exactly one lane, so a tile drives one lane rather than one per engine. The
per-group counts agreed with the lane scan and every group no tile drove counted zero, which is the
**done-direction isolation** result: a done for group 1 credits group 1 and nothing else.

**A worker's NOC writes are not visible when its done is observed.** The write-ordering harness
demonstrates the hazard directly, on this platform, at 32 KB:

```
barrier=true:  tail word=0xdef1fff (expected 0xdef1fff)  mismatched words=0 of 8192
barrier=false: tail word=0xbaadf00d (expected 0xdef1fff)  mismatched words=16 of 8192, first at index 8176
```

Without a barrier the final **16 words — exactly 64 bytes, one cache line** — of the payload still held
the host's pre-fill when the dispatch engine read them on seeing the done. With a barrier the payload
was intact. So a completion fence is not a theoretical requirement on this build; it is mandatory, and
`noc_async_write_barrier()` on the issuing hart is sufficient for that hart's own writes.

**A NOC atomic in the same position is ordered behind the write; the sideband is not.** The control
arm repeats the unbarriered case with completion announced by `noc_semaphore_inc` on
`NOC_UNICAST_WRITE_VC` — the mechanism and virtual channel the current path uses — and changes nothing
else:

```
barrier=false signal=fds:         tail word=0xbaadf00d  mismatched 16 of 8192
barrier=false signal=noc-atomic:  tail word=0xdef1fff   mismatched  0 of 8192
```

**Read this result narrowly.** It compares two *contract-violating* cases, not two real ones. Real
kernels do drain: `noc.async_write_barrier()` at the end of the writer is the norm — see
`test_kernels/dataflow/l1_to_dram.cpp` and `writer_unary.cpp` — and `wait_subordinates()` means DM0
signals only after every subordinate kernel, and therefore every subordinate's own barrier, has
finished. So for conforming kernels the drain is already paid today and FDS adds no new cost.

What the comparison does show is that FDS removes a safety net. When a kernel *fails* to drain, NOC
ordering currently hides it — the atomic cannot overtake the data it shares a road with — whereas the
sideband turns the same omission into reliable corruption. That is a robustness argument, not a
performance one, and it is sharpened by the fact that the firmware checks which would catch a
non-draining kernel are commented out at `dmk.cc:113-123`.

Note also an asymmetry in where barriers actually live. The **go** direction is covered on the
dispatcher side: `cq_dispatch.cpp:1094` barriers under the wait command's `barrier` flag, and
`dm.cc:289` records a barrier before multicasting the launch message. The **done** direction has no
firmware barrier at all — `dm.cc:410` signals straight after `wait_subordinates()` — so it rests
entirely on the kernel-level contract composing across harts.

One alternative explanation has not been excluded. The atomic arm polls an uncached L1 word while the
FDS arm reads a coprocessor register, and if the former is materially slower the signal is simply
noticed later, giving the write more time to land. Recording how many poll iterations each arm took
before its signal appeared would settle it: similar counts mean ordering is doing the work, a much
larger count in the atomic arm means latency is.

**The window is narrow, which is the part to be careful about.** Sixteen words of 8192 is 0.2% of the
transfer: the race is roughly the last packet's flight time. At 4 KB it was invisible. That does not
mean the hazard is small — it means a test has to be tuned to see it, and that congestion, larger
transfers, or different silicon timing will widen it. Never read a clean result from this harness as
safety without checking the tail word specifically.

**The stride of four is suggestive and not yet proven.** These are the only two worker tiles
`emu-quasar-2x3_DISPATCH` has, so the sample is as large as this config allows. Two adjacent tiles
landing on lanes 0 and 4 fits a lane space of four per tile — four Tensix engines per cluster, eight tiles, thirty-two lanes —
with software able to drive only the first of each group of four because the outbox register is per
tile. If that reading is right it caps a dispatch tile at **eight** worker tiles, not thirty-two,
which is a real constraint on the completion design. But two data points admit many rules, and the
logical-to-physical mapping is in play as well: logical `0-0` and `1-0` are physical `0-1` and `1-1`
in this descriptor. A descriptor with more worker tiles would settle it.

---

## 6. Why nothing crossed on the previous build

Kept because the elimination work is what made the diagnosis trustworthy, and because it says which
software-side explanations are already excluded if the symptom ever returns.

On the previous simulator build, the dispatch engine held group 1 in its outbox for entire runs with
the value read back, twelve data-movement blocks across both worker tiles drove done and held it,
and throughout: the dispatch engine's quiet-lane map never changed from all thirty-two, all
thirty-two of its raw inbox registers stayed at zero with its group count at zero, and every worker
block's three raw inbox registers stayed at zero with group status at zero.

Every software-side candidate was eliminated by measurement: timing races and lost edges (both
signals are held, with tens of seconds of simulated overlap), count thresholds (set to one, and the
raw inboxes bypass counting entirely), enable masks and group id choice (the raw inboxes sit before
aggregation), the deglitch filter (swept 0, 1, 2, 8 and 64 on both sides), stale auto-dispatch or
interrupt state (both read zero before being explicitly zeroed), wrong processor (all twelve user
data-movement cores across both worker tiles, then all eight on the dispatch tile), and instrument
error generally (field-width truncation and cross-address probes proving real registers and correct
addressing, with 34 probe points fitting one address-decode model exactly).

The conclusion drawn at the time was that the lanes were not connected in that configuration, and
that this was a hardware-configuration question rather than a software one. A new simulator build
confirmed it. The elimination table above is therefore still the right first response if the
sideband ever goes quiet again: none of those causes is a live suspect, so suspect the build.

---

## 7. False leads and corrections

**Read this section before forming a hypothesis.** Each of these looked convincing and was wrong.

**"The inbox is plain storage with no logic behind it."** The first probe wrote a sentinel to a
hardware-driven input register and read it back unchanged, concluding the registers were dumb
storage. Wrong: the sentinel fitted inside the register's four-bit field, so nothing distinguished
storage from a real register. The fix is to write a value *wider* than the field.

**"Address `0x400` is unmapped and correctly rejected the write."** It kept four bits of it. `0x400`
aliases to offset `0x000`. The probe's verdict line was backwards.

**"`fds_clear_neo_status` writes into the wrong register map."** It does not — see section 4.

**"The Tensix engine experiment tested whether engines are the endpoint."** It did not. Those
processors cannot reach the interface, so every value they reported was stale register contents. Any
experiment involving TRISCs and FDS is measuring nothing until that changes.

**"The deglitch filter is the likely culprit."** It sits exactly on the receive path, its reset value
is 0, and the test had been overriding it with 1 from the beginning — a filter rejecting an assertion
is indistinguishable from an idle lane. It was a good hypothesis and it is dead: swept across five
settings on both sides with no effect, and the working handshake now runs with the filter at 0.

**"Each processor has its own register block."** Stated as established for several runs, and wrong.
The evidence behind it was only that there is no banking stride at `0x1000`, which shows a processor
cannot *address* a different block — equally true if there is only one. The stamp test in section 4
settled it: one block per tile.

This matters beyond bookkeeping, because it means **the processor sweeps were uninformative by
construction.** Six worker processors, then eight dispatch processors, all returned byte-identical
results — not because the lanes are wired identically to each, but because every processor was
reading and writing the same registers. Do not re-run a placement sweep on one tile expecting it to
distinguish anything; it cannot.

**"A clean write-ordering result means writes are ordered."** Only if nothing stood between the
signal and the read. The first version of this harness came back clean on both arms, and both sides
were at fault: the worker wrote and flushed its status block between issuing the NOC write and driving
the done, and the dispatch engine invalidated 4 KB from L2 — dozens of fenced cache-line operations —
after observing the done and before reading. Each delay was ample for the transfer to land. Removing
both, and raising the payload to 32 KB, turned the same test from a false negative into a clear
positive. A negative result here is only as strong as the gap between signal and read; measure that
gap before believing it.

**"A test that passes proves the transport works."** Not on its own. Two of the three tests can be
satisfied by a done count of 1, which is also what a dead read instruction can fabricate, and the
isolation test's two negative assertions pass vacuously when nothing crosses at all. It was the count
of 2 across two tiles that settled it. Design each new test so that at least one of its assertions
cannot be satisfied by a single plausible stale value.

---

## 8. What is not yet established

In rough order of how much the dispatch design depends on it.

**The lane-assignment rule, as opposed to two measured lanes.** Two tiles map to lanes 0 and 4. The
design needs a function from physical worker coordinates to lane bits that is right for every
supported topology and rejects the ones it does not know. Two data points cannot pin that rule, and
harvested layouts have not been tried at all. This needs either a descriptor with more worker tiles
or the specification.

**Whether a dispatch tile can address more than eight workers.** Follows directly from the stride: if
lanes are four apart because the lane space is per-engine, thirty-two lanes cover eight tiles. The
completion design assumes it can enumerate all workers of a sub-device in one group mask, so this is
a cap worth confirming before building on it.

**The scope of the fence, now that its necessity is settled.** The issuing hart's own barrier is
sufficient for its own writes. Untested: whether a barrier on DM0 drains traffic issued by
*subordinate* DMs, or whether every hart must drain before signalling subordinate completion;
behaviour under heavy NOC congestion; and a sweep of transaction types.

**Whether the NOC ordering that protects today's path is architectural.** Measured to hold on this
build; unknown whether it is guaranteed. It may not extend to different virtual channels, multicast,
other destinations, or traffic issued by subordinate DMs. If it is not guaranteed, today's completion
path is latently broken too and the drain contract is all that stands behind it.

**Whether the per-hart barriers compose to cover every hart.** This is what the done direction rests
on: each kernel drains its own writes, and `wait_subordinates()` orders DM0's signal after every
subordinate kernel has returned. Whether a subordinate's barrier guarantees its writes are *committed*
at the destination rather than merely sent, and whether `wait_subordinates()` can return before that,
is the remaining question — and it applies to today's NOC-atomic path equally.

**Whether the atomic arm's clean result is ordering or poll latency.** The two arms observe their
signals by different means — an uncached L1 load versus a coprocessor register read. Recording the
poll-iteration count per arm would separate the explanations.

**More than two groups, and more than two tiles.** Two groups and two tiles have been exercised. The
design calls for eight groups, and the count fields are eight bits wide.

**Routing across dispatch instances.** Only one dispatch tile exists in this descriptor and instance
0 drove every go observed, so the three-instance dimension is untouched.

**Deassertion.** Nothing has watched a lane go from asserted back to idle.

**Auto-dispatch mode.** The block has `AUTO_DISPATCH_EN`, `CYCLE_COUNT`, `OUTBOX_ADDRESS` and
`FIFO_FULL` registers, and the shim's `fds_go`/`fds_done` take an `ad_enable` argument. Only the
direct path with auto-dispatch off has ever run. Reading the shim, auto-dispatch appears to add flow
control around the same outbox write rather than being a different transport, so this is
low-probability — but it is untried.

**Persistence across device reset.** Whether configuration survives, and whether stale state from a
previous run can affect a later one. Now more pressing than before, because held levels that survive
a reset would corrupt the first epoch after it.

---

## 9. Recommended next steps

In order.

**1. Pin down the lane-assignment rule.** Lanes 0 and 4 for two adjacent tiles is a measurement, not
a rule, and `emu-quasar-2x3_DISPATCH` has no third tile to add. A larger config is needed —
`quasar_simulation_8x4_arch.yaml` and `quasar_32_arch.yaml` exist as descriptors, so the question is
whether a simulator build for one of them is available. `DispatchEngineLaneMap` needs no changes to
map up to fifteen tiles in one run, and ask the specification owners whether the lane space
is four per tile and therefore caps a dispatch tile at eight workers. Until the rule is known, any
worker-coordinate-to-bit function should reject topologies it has not been measured on rather than
extrapolate from a stride of four.

**1a. Re-enable the post-kernel NOC flush assertions at `dmk.cc:113-123`.** They are what turns the
drain contract from a convention into something checked. Worth doing whether or not FDS proceeds, and
more so if it does, because the sideband removes the NOC ordering that currently hides a violation.

**2. Gate 0b, the completion fence — necessity settled, scope open.** The hazard is demonstrated on
this build and a barrier fixes it for the issuing hart. What remains is scope: subordinate-issued
traffic, congestion, and transaction types. Also run the NOC-atomic variant described in section 8,
which says whether today's path is safe by contract or by accident; it is cheap and it changes how
urgent the rest is. Silicon confirmation is still required before any of this becomes a contract, but
the central question no longer needs silicon: the answer is yes, a fence is required.

**3. Harvested layouts.** Never exercised. The lane rule has to survive them.

**4. Ask for the authoritative register specification.** Still outstanding, and still worth having.
The generated headers give addresses, widths and reset values, but no access types and no behavioural
description. Several questions that took multiple simulator runs to answer would have been immediate:
whether status latches, what the deglitch threshold units are, whether the outbox drives a level or a
pulse, and whether a block-level enable exists.

**5. Ask what the 32 done lanes correspond to.** The block is named for the Tensix engine, which
suggested one lane per engine across eight tiles. But there is one block per *tile* with a single
outbox register in it, so a tile appears to drive one lane, not four — which would make 32 lanes mean
up to 32 tiles. The lane-mapping experiment in step 1 will show which reading holds on this build;
the specification should confirm it, because a design cannot rest on a mapping observed on one
simulator configuration.

**6. Do not invest in TRISC-side experiments** unless documentation shows a different instruction
encoding reaches FDS from those processors.

**7. Re-confirm on silicon or a certified model before any of this becomes a contract.** A passing
simulator test is necessary and is not evidence that silicon has the same ordering, clear and
ownership semantics. The transport working on this build is a starting point, not a specification.

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

`tt_fds_dispatch` — present on dispatch-engine cores. Base `0x200`, size `0x19C`. The `0x200` base is
stripped by the decode.

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

### Probes worth restoring

These are the techniques that made the difference. They are not in the tree; they are in commit
`156af976a12` as `quasar_fds_probes.h`.

**Field-width truncation.** Write `0xFFFFFFFF`, read back, restore. A real register returns its field
mask; storage or a dead instruction returns something else. Choose registers with *different* widths
so the answers are distinguishable from each other.

**Cross-address.** Write two different values to two different registers, read the first back. A real
read returns the first value; a dead instruction returns the second or unrelated data. This is what
defeats the uninitialised-destination-register problem, and it is worth re-running on any new build
before trusting a negative result from it.

**Address-map sweep.** Truncation across a spread of addresses at once. The pattern of widths reveals
how many address bits are decoded, which maps a processor hosts, and whether the file repeats at a
banking stride.

**Per-processor stamps.** Every processor writes a value carrying its own index into one otherwise
unused register early, and reads that register back at the end, once all have certainly written. A
processor reading a value that is not its own is reading a register a neighbour wrote, which is what
proved the block is shared per tile. Note the asymmetry when interpreting it: reading a foreign stamp
is proof of sharing, but reading your own proves nothing, since that is equally what a private block
gives you and what a shared block gives to whichever processor wrote last.

**Decoupling the directions.** The worker drives done part way through its wait whether or not a go
arrived, so a missing go cannot hide the state of the done direction. Needed again for the
done-isolation test in section 9.

**Raw inbox dumps.** Reading every inbox register rather than only the aggregated count separates
"the signal arrived but was not counted" from "the signal never arrived". This is also what showed
that a foreign group's value appears in a raw inbox without being latched.

### Two implementation details that will bite

**Status blocks must be padded to cache lines** if more than one processor per tile writes them.
Data-movement kernels write status through the cached path and flush; TRISCs write through the
uncached alias. With tightly packed blocks, two processors share a line and one can write back over
the other's slots — which looks exactly like "that processor never ran". The tests in the tree now
run one processor per tile and do not need the padding, but any restored multi-processor variant does.

**Print output from many processors is not chronologically ordered.** Each processor has its own
buffer and the host drains them independently, so the interleaving in a log is not a timeline.
Several early hypotheses about ordering were unfalsifiable for this reason. Design experiments so the
conclusion does not depend on the relative order of lines from different processors.
