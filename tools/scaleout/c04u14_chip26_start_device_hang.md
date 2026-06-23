# Device bring-up hang on `bh-glx-120-c04u14` — logical chip 26 stuck in `Cluster::start_device()`

**Status:** Root-caused to a single ASIC. Reproduces in isolation on one host → hardware/firmware fault, not a fabric/MGD/test problem.

**Date:** 2026-06-23
**Cluster:** SC36 / 120c Blackhole galaxy (revAB, Aisle C), hosts `bh-glx-120-c0*`
**Failing host:** `bh-glx-120-c04u14` (MPI rank 6 in the 12-host run)
**Failing device:** **logical/UMD chip id 26** (PCIe enumeration id **18**; tt-smi "UMD Chip ID" 26 → PCI BDF `0000:c3:…` — confirm against the tt-smi snapshot)

---

## TL;DR

`bh-glx-120-c04u14` has one ASIC (**logical chip 26**) whose ARC/firmware does **not complete device bring-up**. The chip is PCIe-enumerable (tt-smi lists all 32 chips) and topology discovery sees it, but UMD `Cluster::start_device()` **hangs indefinitely** on `get_chip(26)->start_device(...)`. There is no timeout on that path, so:

- **Single host:** the process hangs after `[PSD-DEBUG] start_device: bringing up chip 26` (chips 0–25 came up in ~1 ms each).
- **Multi-host (12 hosts):** rank 6 (`c04u14`) hangs in `start_device` **before** reaching Physical System Descriptor (PSD) discovery; the other 11 ranks finish device init, reach the PSD discovery entry `barrier()`, and **block forever** waiting for rank 6 → the launcher eventually kills the whole job.

Mock-cluster-descriptor runs pass because they never open real silicon (no `start_device`).

---

## Symptom (production, multi-host)

Running PSD discovery across the 12 SC36 hosts:

```
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --tcp-interface ens5f0np0 --hosts $HOSTS \
  --mesh-graph-descriptor models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_single_pod_mesh_graph_descriptor.textproto true
```

hangs and is killed. With the `[PSD-DEBUG]` instrumentation, **11 of 12 ranks** log:

```
[PSD-DEBUG] rank N host '…': entering PSD discovery, waiting on entry barrier (all 12 ranks must reach it) ...
```

The only rank that never reaches it is **rank 6 = `bh-glx-120-c04u14`**. Its UMD log shows it ~1 s behind every peer and ending at:

```
UMD | Starting devices in cluster (cluster.cpp:1236)
```

with **no** `Starting devices in cluster completed.` — i.e. stuck inside the per-chip `start_device` loop, before PSD discovery even begins.

## Isolation + pinpoint (single host) — the smoking gun

`tools/scaleout/isolate_host_device_init.sh` run on `c04u14` (single rank, `single_bh_galaxy` MGD):

1. **tt-smi:** all 32 chips (UMD Chip ID 0–31) enumerate and are listed as resettable. No PCIe/enumeration error. → the chip is *visible*, the fault is in *bring-up*.
2. **Device bring-up** (`Cluster::start_device`, per-chip `[PSD-DEBUG]` logging):

```
[PSD-DEBUG] start_device: bringing up chip 0  ... chip 0 brought up OK
...                                            (chips 0–25 each OK, ~1 ms apart)
[PSD-DEBUG] start_device: bringing up chip 25 ... chip 25 brought up OK
[PSD-DEBUG] start_device: bringing up chip 26   ←  LAST LINE. No "chip 26 brought up OK". Hang.
```

Chips 0–25 complete in ~13 ms total; **chip 26 never returns** from `get_chip(26)->start_device(...)`. This reproduces standalone on `c04u14`, confirming it is a property of that ASIC, not of the multi-host run.

## Chip identification

From `Opening local chip ids/PCIe ids: {0..31}/[0..15,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23]`:

| logical chip | 24 | 25 | **26** | 27 |
|---|---|---|---|---|
| PCIe id | 16 | 17 | **18** | 19 |

So **logical chip 26 → PCIe id 18**. Cross-reference tt-smi (UMD Chip ID 26 → PCI BDF `0000:c3:…`) in the saved snapshot for the physical board/tray slot.

---

## Root cause

`get_chip(26)->start_device()` performs the ASIC's ARC/firmware bring-up (deassert resets, ARC handshake, power-state setup). Logical chip 26 on `c04u14` accepts PCIe enumeration but its ARC does not complete this handshake, so the call blocks with no timeout.

This is **not**:
- a fabric / MGD / ethernet-link issue (the hang is before any fabric/ethernet code),
- a cross-host issue (each host opens only its own local chips; `remote chip ids {}`),
- a test/CI logic problem (mock passes precisely because it never starts silicon).

It **is** a per-ASIC bring-up fault on one chip of one host. The earlier `Waiting for AICLK value to settle … possible overheating … AICLK clamped` warnings on several chips suggest marginal power/clock bring-up on this system generally; chip 26 is the one that crosses from "slow" into "stuck."

## Why the whole 12-host job dies

`run_physical_system_discovery()` opens with `distributed_context->barrier()`. The UMD cluster (and `start_device`) runs during `MetalContext`/cluster construction **before** that barrier. So rank 6 never arrives at the barrier; ranks 0–5 and 7–11 block in it indefinitely. The collective has no timeout, so the run only ends when the MPI launcher / tt-run watchdog tears it down.

---

## Remediation

1. **Reset and retry** on `c04u14`:
   - `tt-smi -r <board>` (board-level) or galaxy reset (`tt-smi -glx_reset`), then re-run `isolate_host_device_init.sh`.
   - If chip 26 then brings up `OK`, the fault was a recoverable wedged ARC state.
2. **If it persists after reset:** hardware/firmware fault on that ASIC — flag the board (board number `00000471…`, BDF `0000:c3:…`) for service; check chip 26 temperature/power rails and firmware (`19.11.0`, newer than the tested `19.5.0`).
3. **Workaround to keep testing meanwhile:** exclude `c04u14` from `$HOSTS` (use a different 12-host set) — the rest pass discovery.

## How to reproduce / verify

```bash
ssh bh-glx-120-c04u14
source /data/rsong/tt-blaze-5/env.sh
bash tools/scaleout/isolate_host_device_init.sh ens5f0np0
# Expect: hang after "[PSD-DEBUG] start_device: bringing up chip 26"
# Healthy host (e.g. c01u02) completes all 32 chips with "brought up OK".
```

> Caveat: if a future run of the script completes all 32 chips (paste was merely truncated / the ARC self-recovered), then `c04u14` is currently healthy in isolation and the production hang was transient — rerun a few times to confirm chip 26 reproducibly hangs before condemning the board.

---

## Appendix — diagnostic instrumentation added (temporary, tag `[PSD-DEBUG]`)

- `tt_metal/third_party/umd/device/cluster.cpp` — `Cluster::start_device()`: per-chip "bringing up chip N" / "chip N brought up OK" logging (pinpoints the stuck chip).
- `tt_metal/fabric/physical_system_discovery.cpp` — `run_physical_system_discovery` / `exchange_metadata` / `resolve_hostname_uniqueness`: per-stage + per-peer logging (pinpoints the stuck rank/host in the multi-host gather).
- `tools/scaleout/isolate_host_device_init.sh` — single-host tt-smi + device-bring-up reproduction.

All `[PSD-DEBUG]` logging is temporary and should be removed once the hardware issue is resolved.
