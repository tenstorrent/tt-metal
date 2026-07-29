# X280 (L2CPU) L3 LIM ECC priming — findings & trials

Consolidated record of everything we learned about priming the Blackhole X280's L3
LIM ECC: the problem, the working prime, why it needs a chip reset, every attempt to
avoid/streamline that reset, the detection/induction tooling, and the metal-init
auto-recovery we shipped. Companion to `FINDINGS.md` (throughput/drain) and `PRIMER.md`.

---

## 1. The problem

The X280 boots and runs its drainer firmware out of **LIM** (L3 SRAM, `0x08000000`). On a
**fresh / cold-power-cycled** board the SRAM's ECC check bits are uninitialized, so the
first read of a line computes a syndrome against garbage → **double-bit (uncorrectable)
detection** → the tile latches `halt_from_tile` "until reset" (SiFive manual §18.3). The
hart never runs, the drainer never publishes its heartbeat, and (because it also never
drains the per-RISC SPSC profiler rings) a profiled workload can deadlock.

Symptom in metal: `realtime_profiler_manager` boots the X280, polls the main()-entry
heartbeat (`0xB007` at `params+0x70`), and times out.

---

## 2. The working prime (cache-controller method)

Route LIM's SRAM through the L3 cache controller so writes stamp valid data **and** ECC
into the physical SRAM, without ever *reading* the uninitialized ECC.

L3 Cache Controller base `0x02010000` (reached via the L2CPU tile NoC path):

| Offset | Reg | Use |
|---|---|---|
| `+0x000` | Config | reads `0x06091004` (health probe) |
| `+0x008` | WayEnable | `0xF` = all ways as cache. **Increase-only in SW.** |
| `+0x040` | ECCInjectError | `[7:0]`=bit index, `[16]`=type (0=data,1=dir) |
| `+0x148` | DatECCFixCount | correctable (1-bit) events |
| `+0x160/+0x164` | DatECCFailLow/High | address of last uncorrectable |
| `+0x168` | DatECCFailCount | uncorrectable (2-bit) events |
| `+0x800` | WayMask0..37 | stride 8, low 16 bits = way mask, 38 masters |

Prime sequence (`test_x280_profcons --primeecc`, and `X280Driver::prime_lim_ecc()`):

1. `WayEnable (0x02010008) = 0xF` — put L3 into cache mode (this is the irreversible step).
2. For all 38 masters, `WayMaskN = 0x8000` — pin every allocation to Way 15.
3. For `off` in `0 .. 0x60000` step 64: write `0` to the **L3 Zero Device** (`0x0A000000+off`).
   The zero device reads-as-zero and aliases the same SRAM ways as LIM; each cacheable
   touch allocates + writes back a zero-filled line **with valid ECC** into the SRAM.
4. **Warm reset the chip** (`tt-smi -r` or UMD `WarmReset`). This clears cache mode
   (WayEnable→default, LIM = plain SRAM again). The ECC bits written in step 3 **persist
   across the reset** — that's the whole trick — so LIM is now valid SRAM.

`0x60000` (384 KiB) covers firmware + mailboxes + profzone stacks/SP.

Why the zero device and not LIM directly: a direct partial write to LIM does a
read-modify-write, and the *read* faults on the uninitialized ECC. The zero-device path
never reads the old line, so no fault. (§18.1.1 recommends internal init, not NoC init.)

---

## 3. Why a full chip reset is unavoidable (for this method)

- `WayEnable=0xF` makes **all** of LIM's SRAM cache → nothing backs `0x08000000` → LIM is
  unusable until WayEnable is cleared, and WayEnable is **increase-only in software**.
- **`--wayprobe` result:** the L2CPU-*local* reset (ARC `L2CPU_RESET` bit, `0x80030014`
  bit `l2cpu+4`) does **NOT** clear WayEnable (`0x0 → 0xf → 0xf` across assert/release).
  Only a full ASIC/warm reset clears it.
- Therefore the prime is intrinsically **prime → reset** (reset must come *after*, to undo
  the cache mode the prime had to enter). It is not the reset that "activates" the ECC — the
  reset *undoes the cache mode*; the ECC was already written and simply survives.

---

## 4. Trials to avoid / streamline the reset

Goal was a prime that stamps ECC **without** entering cache mode, so no reset-to-undo is
needed (→ "reset once, everything works", or no reset at all). Three internal-master paths;
two are now closed.

### 4a. DMA-to-zero (§18.1.1's recommended init) — **BLOCKED**
Host-drive the L2CPU **Synopsys DesignWare DMAC** (NoC `0xFFFFF7FEFFF80000`, x280-phys
`0x20080000`, hart `0x2FF80000`) to do a MEM→MEM zero-scrub into LIM.
- The DMAC register block reads **all-ones** in every state we can reach from a fresh
  reset: in reset, after release, after release+suppress+scratch-spin park, and even after
  the L3 clock-gate ungate (`0x02010008=0xf`). The L2CPU *register* block (`0xFFF10xxx`:
  reset/suppress/vectors) responds — only the **DMA-engine sub-block is dark** (clock
  domain down).
- `--dmaprime` brings the complex up with harts parked two ways (reset-vector → `j .` spin
  in scratch `0x20010100`; suppress-fetch bits `[19:16]` of NoC `0xFFFFF7FEFFF10400`, set
  before release) — DMAC still all-ones.
- tt-llm-engine's probe *did* see real DMAC values, but only because their device had the
  L2CPU more fully up from prior activity. The DMAC needs a fuller bring-up than
  "released with harts parked" → **chicken-and-egg**: the DMA engine we'd use to prime ECC
  isn't clocked until the L2CPU is running, and the L2CPU can't run until ECC is primed.
- ISA docs have **no** DMA clock/reset/power-enable register. Open handoff question for
  exabox/SiFive-integration: *what enables the L2CPU DMA-engine clock without a running hart?*

### 4b. Hart-driven scrub via `cbo.zero` (Zicboz) — **CLOSED**
A hart booted from ECC-safe memory could zero LIM lines with no read. `cbo.zero` is the
clean primitive (establishes a zeroed block, no fetch → no RMW fault).
- `--eccscrub` probe FW (boots from primed LIM, runs `.word 0x0045200f` = `cbo.zero`): the
  hart **stalled** at the instruction (never completed, no trap recorded, mcause=0).
- Toolchain is `-march=rv64gc` (no Zicboz) and X280 cache management is via **MMIO Flush64
  registers**, not `cbo` instructions. **`cbo.zero` is not usable on the X280.**

### 4c. Hart-driven scrub via uncached full-ECC-granule stores — **UNTRIED**
The remaining lever: configure LIM (or an alias) uncached and do full-granule stores. Needs
the ECC granule size + an uncached PMA/alias for LIM. Not yet attempted; deeper + more
device-risk. The hart normally accesses LIM **cached**, so a plain store to an unprimed line
write-allocates → reads uninit ECC → faults (same RMW problem).

**Net:** DMA and cbo.zero are closed; only the uncached-store scrubber remains untried. The
cache-controller prime + warm reset stands as the working method.

---

## 5. Detection & induction tooling (`test_x280_profcons` modes)

| Mode | What it does | Result |
|---|---|---|
| `--eccprobe` | read-only dump of L3 ECC state (Config + fix/fail counters) | Config `0x06091004`; baseline counters 0 |
| `--eccinject [--injbit N]` | single-bit (correctable) inject on next cache op | DatECCFixCount 0→2, **no halt** — induce/detect loop proven |
| `--injdouble [--injaddr A]` | attempt persistent 2-bit (uncorrectable) inject | DatECCFailCount ticks (transient uncorrectable event) but the **persistent stored state is only correctable** — "injector can't make a persistent 2-bit line" |
| `--eccread [--injaddr A]` | read a line N times, report fix/fail counter deltas | scan / persistence probe |
| `--eccpoke [--injaddr A]` | boot probe FW; hart reads target line — does it HALT? | see §6 |
| `--wayprobe` | does L2CPU-local reset clear WayEnable? | **No** (§3) |
| `--dmaprime` | park harts + release, probe DMAC accessibility | DMAC all-ones (§4a) |
| `--dmascrub` | host-program the DMAC MEM→MEM zero-scrub | blocked (DMAC dark) |
| `--eccscrub` | probe cbo.zero (Zicboz) support | not usable (§4b) |

Probe FWs live in `tools/x280_bm/src/`: `eccscrub.c` (cbo.zero), `eccpoke.c` (hart read of a
host-settable target). Both boot via `boot/entry.S` + `ld/x280-lim.ld`.

---

## 6. Inducing a faithful "bad state" on bh-38 — **not possible**

To confirm a fix end-to-end we wanted a genuine unprimed→halt state. On bh-38 that can't be
reproduced:

1. **Can't cold-power-cycle remotely** — the only true source of unprimed SRAM.
2. **Can't un-prime already-primed SRAM** — bh-38's boot region is primed and stays so.
3. **Injector makes only correctable/transient errors** — no persistent uncorrectable, so no
   boot halt (§5 `--injdouble`).
4. **`--eccpoke`: a hart reading uninitialized high-LIM (`0x08100000`, `0x081C0000`) SURVIVES**
   (returns garbage, `mcause=0`, no halt) while primed `0x08000000` reads clean. → **bh-38's
   entire LIM is ECC-valid** (data beyond the `0x60000` prime is garbage, but the ECC is
   valid). That's *why* bh-38 "has no prime issues".
5. Positive control: an injected defect **survives a warm reset** — confirming a reset alone
   never heals ECC (only the prime does).

**Consequence:** the faithful "broken boot → auto-recover" proof must run on a board that
actually exhibits the fault (a freshly cold-booted problem machine). On bh-38 we can only
unit-test the recovery *logic*.

---

## 7. Shipping streamline: metal-init auto-recovery (Option A)

Requirement: streamline the prime for **normal metal users** — no `tt` cockpit CLI, no manual
`tt-smi`. Key enabler: **UMD exposes the same warm reset `tt-smi -r` performs**, callable from
metal (which links UMD):

```cpp
tt::umd::WarmReset::warm_reset_chip_id({chip_id});  // chip_id = Cluster::get_target_mmio_device_ids()
```

**Built** (`x280_driver.hpp` + `realtime_profiler_manager.cpp`):

- `X280Driver::prime_lim_ecc(bytes=0x60000)` — the §2 cache-controller prime as a driver method.
- At the no-heartbeat detection, **ON by default** (opt out `TT_METAL_X280_NO_AUTOPRIME`;
  `TT_METAL_X280_FORCE_PRIME` forces on a healthy board):
  `assert_reset → prime_lim_ecc() → WarmReset::warm_reset_chip_id({device})` → **direct-to-stderr
  banner** → `std::_Exit(75)` (EX_TEMPFAIL = "transient, rerun"; `1` on failure).
- Fires only when X280 kernel-zone profiling was requested **and** the drainer didn't start.
- **Loop guard:** `/tmp/tt_x280_autoprime_dev<N>` marker — written before the reset, removed on
  a clean boot; if it's still there next run, we don't re-prime (avoids a reset loop on a truly
  faulty chip).

**Not seamless within one run.** A warm reset re-enumerates the PCIe device, so metal can't
continue in-process — the run **exits and the user reruns once**:

- Run 1 (cold board): heartbeat fails → auto prime + warm reset → banner "RERUN" → exit 75.
- Run 2: X280 drainer boots clean; profiling works. No env vars, no `tt`, no `tt-smi`, no
  manual prime.

A seamless no-rerun version would need metal to tear down + reopen the device in-process
after the reset — bigger, and untestable without a bad card. **Deferred.**

### Unverified assumption
`tt-smi -r` does its warm reset from a *separate* process; Option A calls `WarmReset` from a
process that **already holds the device open**. Expected to work (same UMD/driver path) but
not yet validated on hardware. If it doesn't fire, the prime still succeeded, so a one-time
manual `tt-smi -r` completes recovery (degraded, not broken).

### How to test on a genuine cold-booted bad card
```
<your profiled run>       # run 1: auto prime + warm reset; prints RERUN banner; exits 75
<your profiled run>       # run 2: X280 kernel-zone drainer boots clean
```
`TT_METAL_X280_FORCE_PRIME=1` exercises the prime+reset mechanics on any board, but needs a
real RT-profiler workload (the `test_x280_profcons` harness uses its own boot path, not the
manager's).

---

## 8. Key addresses (Blackhole X280)

| Name | Address |
|---|---|
| LIM base (FW load / reset vector) | `0x08000000` |
| L3 Zero Device | `0x0A000000` |
| L3 Cache Controller base | `0x02010000` (WayEnable `+0x008`) |
| L2CPU register block (NoC) | `0xFFFFF7FEFFF10000` |
| — reset-vector regs | `+0x8*hart` |
| — general scratch (executable) | `+0x100` (`0x20010100`) |
| — hart status + suppress-fetch | `+0x400`, suppress `[19:16]` |
| DMA engine (NoC / x280 / hart) | `0xFFFFF7FEFFF80000` / `0x20080000` / `0x2FF80000` |
| ARC reset unit / L2CPU_RESET | `0x80030000` / `0x80030014` (bit `l2cpu+4`) |
| ARC PLL4 base | `0x80020500` |
| DRAM boot aperture (L2CPU-local) | `0x400030000000` |
| drainer heartbeat magic | `0xB007` @ `params+0x70` |
| L2CPU tiles (NOC0) | idx0=(8,3) 1=(8,9) 2=(8,5) 3=(8,7); ARC=(8,0) |
