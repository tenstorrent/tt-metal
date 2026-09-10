<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First-step triage

Answers "is this chip alive, and if not, how is it dead?" on a host where the
runtime may not load at all. Nothing here depends on `tt_metal` or UMD:
`host_side.sh` is bash over sysfs, and `device_side.sh` drives `kmd_triage`, a
standalone binary that talks to tt-kmd over its ioctl interface directly. That
is the whole point — these are the tools you reach for when the stack above them
is what is broken.

The exabox health check runs both automatically on its `medium` and `deploy`
tiers; see
[`HEALTH_CHECK.md`](../exabox/health_check_test_suite/HEALTH_CHECK.md) for how
the findings are ingested. This file is about running them by hand.

## Quick start

Run `host_side.sh` first — it is safe on a wedged machine and needs no device
access. Run `device_side.sh` second — it opens the chips.

```bash
sudo ./host_side.sh   -o hostside-$(hostname -s)-$(date +%Y%m%d-%H%M%S).txt
     ./device_side.sh -o deviceside-$(hostname -s)-$(date +%Y%m%d-%H%M%S).txt
```

Both write one self-contained report and echo the verdict and problem list to
the terminal, so you can hand the file to someone else and it stands alone.
Without `-o` the report goes to stdout.

Exit status is the same for both: `0` nothing wrong at this level, `1` degraded,
`2` hopeless, `3` the script itself failed.

`host_side.sh` wants root for the full `lspci -vvv` capability blocks, the
driver's debugfs mappings and the kernel-log scan. It runs fine without, and
says which parts it could not read rather than reporting them clean.

### Getting the binary

`device_side.sh` needs the `kmd_triage` binary. It is a normal CMake target, so
a plain build produces it:

```bash
./build_metal.sh            # or: ninja -C build kmd_triage
```

It lands in `build/tools/scaleout/kmd_triage`, and the script finds it via
`$KMD_TRIAGE_BIN`, then `$TT_METAL_HOME`'s build tree, then a build tree above
this directory, then `$PATH`.

On a machine with no checkout, `kmd_triage.cpp` builds on its own — the source
is deliberately self-contained, with the tt-kmd ioctl structs inlined rather
than included, so copying the single file is enough:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o kmd_triage kmd_triage.cpp -lpthread
export KMD_TRIAGE_BIN=$PWD/kmd_triage
```

Keep it that way. Reaching for anything in this repo would mean the tool stops
working on exactly the hosts it exists for. Either script can be copied over on
its own — `host_side.sh` needs nothing but bash — and each only needs
`triage_json.sh` next to it if you want `--json`; the text report works without
it.

### Machine-readable output

Both scripts take `--json FILE` alongside `-o`, which is how the health check
consumes them. Useful by hand too, if you are diffing runs or scripting over
the result:

```bash
./device_side.sh --json /tmp/ds.json -o /tmp/ds.txt
jq -r '.checks[] | "\(.status)\t\(.name)\t\(.details)"' /tmp/ds.json
```

Each check is `{"name", "status", "details", "ip", "data"}`, with `status` one
of `PASS`/`WARN`/`FAIL`/`SKIP`. The emitter is `triage_json.sh`, sourced by both
scripts so the shape cannot drift between them.

## The two halves

The boundary is **does it open the device**, which used to also mean **does it
need a compiler**, because there is no shell path to an ioctl.

| | `host_side.sh` | `device_side.sh` |
|---|---|---|
| Language | bash only | bash + the `kmd_triage` binary |
| Opens `/dev/tenstorrent` | never | yes |
| Touches the NOC | never | yes |
| Needs root | yes, for full `lspci`, debugfs and dmesg | no |
| Works when the driver is unloaded | yes | no |
| Works when the chip is off the bus | yes | partly: host-side facts still print |

`host_side.sh` is the one that still works when everything else does not, which
is why it stays in bash and depends on nothing.

## What each one checks

### `host_side.sh` — host, PCIe and driver state

Chips present on the PCI bus versus chips bound to the driver, PCIe link
generation and width against what the link could actually train to, unassigned
BARs, AER counters on endpoints *and* upstream bridges, driver sysfs telemetry,
heartbeat progress, open file descriptors, driver debugfs mappings, IOMMU and
host configuration, and a deliberately narrow scan of the kernel log for PCIe,
IOMMU and machine-check faults.

Errors on a link frequently land on the bridge rather than the endpoint, which
is why the bridge AER counters are read separately — a chip can look clean while
the port feeding it is counting failures.

Options beyond `-o` / `--json`:

| Flag | Purpose |
|---|---|
| `--expect N` | Expect N chips on the bus (default: 32 on a Galaxy, otherwise no expectation) |
| `--no-device-reads` | Skip the telemetry sysfs attributes — the only part of this script that contacts the device |
| `--no-lspci` | Skip the full `lspci -vvv` dumps, which dominate the runtime |

### `device_side.sh` — liveness and NOC integrity

Five probes per device, cheapest and most passive first:

1. **`hung`** — driver sysfs ladder: PCI config space, every `tt_*` attribute, a
   double heartbeat sample, then one NOC read to ARC. Needs no TLB, so it still
   works when another process holds them all.
2. **`info`** — PCI link, IOMMU mode, board type and id, DRAM training judged
   against the enabled-GDDR mask, firmware version, vitals. Host-side facts
   print even when the chip is hung.
3. **`scratch`** — ARC reset-unit scratch registers, annotated with their
   firmware-assigned roles: postcodes, boot status, telemetry pointers.
4. **`telemetry`** — walks the firmware telemetry table and decodes every tag.
   Its structural reads are the liveness check: all ones means the chip is not
   answering.
5. **`test noc_sanity`** — asks all 204 (Blackhole) or 120 (Wormhole) NOC0 nodes
   who they are, across every tile type. The only probe that touches nodes other
   than ARC, so it runs last and is skipped for a chip telemetry already found
   silent.

With no arguments every `/dev/tenstorrent/*` is probed; pass device paths to
narrow it. `--no-noc` skips probe 5.

Device ordinals come from driver probe order and mean nothing physical. On a
Galaxy both scripts decode each chip's physical position from its PCI bus number
and report it as `u<ubb>c<chip>` — that is what tells you which tray to pull.
`host_side.sh` additionally knows that exactly one chip per UBB is wired x8 and
the rest x1, and judges link width per chip accordingly.

## Driving the binary directly

For interactive digging, `kmd_triage` is useful on its own. `<DEVICE>` is a path
or an ordinal, and defaults to the first device where it is optional.

```
kmd_triage <subcommand> [options]
```

| Subcommand | What it does |
|---|---|
| `discover` | Enumerate every device, one line each, with a liveness verdict |
| `info` | One-screen inventory: PCI link, IOMMU, board, DRAM, firmware, vitals |
| `scratch` | Dump the ARC scratch register banks |
| `telemetry` | Dump every telemetry tag ARC firmware publishes (`-r` for raw) |
| `hung` | Is the chip hung? Config space, sysfs telemetry, heartbeat, NOC read |
| `read32` | Read one 32-bit word from a NOC endpoint |
| `write32` | Write one 32-bit word to a NOC endpoint |
| `test` | Run one or all chip tests: `noc_sanity`, `dma_loopback` |
| `reset` | Reset the chip in place and check it came back |
| `nuke` | Kill every process holding the device open, naming each one |

`kmd_triage <subcommand> -h` gives each subcommand's own usage.

```bash
kmd_triage discover                        # what is on this host, and is it alive
kmd_triage hung 3                          # why is device 3 not working
kmd_triage telemetry -d /dev/tenstorrent/0
kmd_triage test --all 0
kmd_triage read32 -d /dev/tenstorrent/0 0 10 0x80030060
```

`reset`, `nuke` and `write32` are **destructive and never invoked by the
scripts** — they exist for interactive use:

```bash
kmd_triage nuke 3            # something is holding the device open
kmd_triage reset 3 --sbr     # link reset, no ASIC reset
kmd_triage reset --all       # every chip at once
kmd_triage reset --glx       # Galaxy tray reset over IPMI, needs ipmitool
```

`reset` takes no default device, unlike the read-only subcommands — you have to
name one or pass `--all`. It needs tt-kmd 2.10.0 or newer.

### Exit codes

`0` and `1` mean the same thing everywhere; `2` and `3` are per-subcommand.

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Usage error, or could not reach the point of testing — **the tool or its environment is broken, not the chip** |
| `2` | `telemetry`/`hung`/`reset`/`discover`: chip not answering · `test`: found a chip failure · `nuke`: processes still hold the device |
| `3` | `telemetry`: unpublished or malformed · `hung`: firmware looks sick · `discover`: some chip degraded or unknown · `reset`: a reset step failed · `test noc_sanity`: a node did not answer at all |

`hung`, `reset` and `nuke` end on exactly one `[PASS]` or `[FAIL]` line, so a
wrapper can judge a completed run by its last line alone.

Note that `1` means something different here than in the scripts, which use it
for "degraded". A wrapper running both has to know which it called.

## Safety

Neither script writes to the device. Every device is opened `O_RDWR | O_APPEND`,
which marks the client power-aware, and no `SET_POWER_STATE` is issued, so
device power state is left exactly as found.

Containment, in layers:

- Every sysfs read in `host_side.sh` runs under `timeout 5`; dmesg and each
  `lspci -vvv` under `timeout 30`.
- Every probe in `device_side.sh` runs under `timeout 60`; exit 124 or 137 is
  reported as `HUNG`.
- The NOC sweep is **gated on telemetry**: a chip whose structural telemetry
  reads came back all ones is not answering, and sweeping 200+ nodes of a dead
  chip can only make things worse.
- `noc_sanity` stops at the first node that does not answer, because continuing
  to poke a device that stopped listening is the documented hazard. It does
  *not* stop at a node that answers with the wrong coordinates: that node is
  alive, the next read costs a microsecond, and the failure pattern across the
  grid is what names the root cause. Both defaults are overridable when digging
  by hand: `-k` walks past silent nodes, `--stop-first` stops at any failure at
  all, `-s X,Y` skips a coordinate, and `-l` prints the whole address plan
  without issuing a single NOC read.

## Known gaps

- **All ones is ambiguous.** A read of `0xFFFFFFFF` can mean the chip answered
  and the value really is all ones, or that nobody answered and the root port's
  completion timeout synthesised it. Blackhole firmware stores exactly this in
  `TAG_FAN_SPEED` and `TAG_FAN_RPM` when fan control is disabled, which is every
  UBB tray in a Galaxy, on healthy silicon. The tools tell the two apart by
  *position* — a structural read (a pointer register, a table header, a
  directory entry) that is all ones is treated as fatal, a tag value that is all
  ones is reported and the dump continues. That is right in the common cases and
  wrong in one: a chip that is alive but has garbage in its telemetry pointer
  register is reported as "not answering". Measuring read latency would settle it
  directly, since the two differ by three orders of magnitude (~1 µs versus the
  65–210 ms completion timeout measured so far). Not implemented; if this
  ambiguity ever produces a misdiagnosis in the field, that is the fix.
- **The summary table is ordinal-sorted, `discover` is loc-sorted.** On a Galaxy
  the `loc` is on every row either way, so the grouping stays readable, but the
  two tables can list the same 32 chips in different orders. Device ordinals come
  from driver probe order, so neither ordering is physical.
- **Exit code 1 means different things** in the scripts and in the binary — see
  the exit-code table above.
- **No read deadline inside the binary.** Containment is the script's
  `timeout 60`. That is sufficient because every subcommand's worst case (reads ×
  the 210 ms completion timeout) fits inside it, and because the one true hang —
  a blocked ioctl in D state — is immune to `alarm()`, SIGTERM and SIGKILL alike.
- **No SIGBUS handler.** tt-kmd zaps every mapping when a device is reset
  (`memory.c`, `zap_special_vma_range`) and the TLB VMA ops have no `.fault`
  handler, so an access after that is a SIGBUS. A concurrent reset from another
  process therefore kills a probe outright instead of it reporting "mapping
  revoked mid-sweep". Do not reset a chip while a probe is running on it; the
  health check enforces this by making its post-test reset a discrete step that
  completes before the triage phase starts.

## Files

| File | What it is |
|---|---|
| `kmd_triage.cpp` | The multitool. Built as the `kmd_triage` target; libc, pthreads and the tt-kmd ioctl interface, nothing else |
| `host_side.sh` | Host, PCIe and driver state. Reads sysfs only |
| `device_side.sh` | Drives `kmd_triage` across every chip and assembles the report |
| `triage_json.sh` | Shared `--json` emitter, sourced by both scripts |

The build wiring is in [`../CMakeLists.txt`](../CMakeLists.txt) and
[`../sources.cmake`](../sources.cmake), alongside the other scaleout tools.
