# Blackhole device/host hang runbook

The **device- and host-level** hangs seen on Blackhole IRD boxes: how to tell them apart, how to
reproduce each, and what actually recovers them. Scope is card/link/host only — profiler-pipeline
wedges (full SPSC rings, two consumers on one ring, drainer boot) and tooling wedges (tracy-capture,
leftover processes) are deliberately out.

## TL;DR (read these first)

- **Classify by CARD STATE, never by exit code.** Pooling these failures produced *and then destroyed*
  four findings in a single day. A run can exit 0 having hit a hanging condition.
- **There are three card/link failures plus one host failure.** They have different signatures,
  different causes, and different recoveries — `tt-smi -r` fixes one of them and nothing else.
- **Read the ROOT PORT (`0000:00:01.1`), not the endpoint.** A wedged endpoint's own sysfs reads
  all-ones and cannot tell you anything. `Unknown|63` is an all-ones artifact, not downtraining.
- **Read `DevSta`, not just AER.** The endpoint carries a real `UnsupReq+` while the AER capability
  reads clean. Every earlier "AER is all zero, so no PCIe errors" conclusion missed this.
- **The knee is irrelevant to the wedge.** Rate is flat across producer delays.

| failure | signature | recovery |
|---|---|---|
| **WEDGE** | card `Unknown\|63`, endpoint config space all-ones, **root port still linked 32 GT/s x16** | `tt-smi -r` — seconds, box survives, full health |
| **TEARDOWN** | `wait_until_cores_done` never completes, card perfectly **healthy** | process dies or is killed; card is fine |
| **DEGRADED** | ack-write ~13× slower (~185 ns → ~2300 ns) | **cold power cycle**; a warm reboot can make it *worse* |
| **VM FREEZE** | whole reservation VM stops, IRD watchdog reboots it; physical host never reboots | reboot is automatic; often leaves the card DEGRADED |

---

## 1. WEDGE — PCIe endpoint stops completing TLPs

**Signature.** Card reports `Unknown|63`; endpoint config space reads all-ones; the root port is still
linked at 32 GT/s x16. Host hangs forever with no bus traffic and no AER.

**Why it hangs silently.** The completion-queue predicate compares the host read pointer against a
*device-written* pointer fetched via `read_cq_host_ptr<true>` → `read_sysmem` — that is **host DRAM**,
not device MMIO. A dead endpoint can never update it, and the loop has no timeout, no yield, and no
device access. Silent by construction.

**Repro**

```bash
tools/drisc_drain/drisc_hang_harness.sh    # scores card state + duration + masked-signature + rc
tools/drisc_drain/drisc_hang_compare.sh
tools/drisc_drain/drisc_reclassify.py      # re-derives classes from logs: a scoring bug costs no re-runs
```

- Rate is **~2–3% per run** with **no delay dependence** (125/150/500 alike), so budget ~50–100 runs
  for a couple of hits.
- **Randomize arm order, never alternate.** The wedge straddles run boundaries — the failure surfaces
  in the run *after* the one that seeded it, so alternation turns that into systematic bias. It voided
  two comparisons.

**Per-run PCIe probe.** `DevSta` bits are RW1C, so clearing them before each run turns the register
into a per-run probe:

```bash
sudo setpci -s 01:00.0 CAP_EXP+0A.w=0x000f   # clear, then read back after the run
# 0x8 = UnsupReq, 0x1 = CorrErr.  DevSta is PCIe cap + 0x0A — a DIFFERENT register from AER.
```

**The 220 ms MMIO stall is the root-port completion timeout firing** (`DevCtl2: Completion Timeout:
65ms to 210ms, TimeoutDis-` on `00:01.1`). The failing access is not slow — the endpoint never
completes it and the root port abandons it at ~210 ms. No 2 ms UMD retry budget can ever cover that.

**Not a fix:** arming `TT_METAL_OPERATION_TIMEOUT_SECONDS` gave 4 wedges vs 0 over 400 classified
randomized runs — p ≈ 0.12, not significant. It also fails to reliably rescue teardown.

---

## 2. TEARDOWN — core-wait never completes, card healthy

**Signature.** `wait_until_cores_done` never returns while the card is perfectly healthy.

**Slow dispatch: SOLVED (2026-08-07) — it was a harness grid bug, not a device fault.** `--gx 0` gave a
12×10 producer grid against an 11-column poll list. Always pass `--gx 11 --gy 10`. Slow cells are now
10/10 clean, and every previously recorded slow-dispatch cell is void.

Two corrections worth keeping:

- `14-2..14-11` are **Tensix workers** (one full worker column), **not** the DRAM/DRISC column. That
  misidentification pointed the investigation at the drainer instead of the producers.
- The Tensix arm did not hang on ten cores — it hung on **all 120**, because its drainer sits in the
  very column the stray producers landed on.

**Fast dispatch.** Still rare, and hits a single core (`14-3`) where the throw *is* caught, so the run
**exits 0**. This is why armed runs can look clean while having hit the hanging condition.

---

## 3. DEGRADED — ~13× MMIO latency

**Cannot be triggered directly.** It follows a **box freeze + watchdog reboot** (§4), not a card hang.
~510 runs and 9 hangs in one day produced zero degradation, because every reboot was clean.

**Confirm the freeze happened:**

```bash
last -x | head      # a session ending in "crash" + a boot with NO preceding shutdown = it froze
cat /proc/uptime    # small value = it went down
```

**A reboot does not fix it, and a warm reboot can make it worse.** Measured on bh-26 across two
reboots:

| | link | ack-write (posted) | device-read (non-posted) |
|---|---|---|---|
| healthy | 32 GT/s | ~185 ns | ~790 ns |
| after watchdog reboot | 32 GT/s | **2306 ns** | 2940 ns |
| after a clean warm reboot | **2.5 GT/s (downgraded)** | **183 ns** | **2738 ns** |

So there are at least **two distinct bad states**, and a warm reboot converted one into the other
rather than clearing anything. The Gen1 downtrain splits cleanly along posted vs non-posted: writes are
fire-and-forget so link rate barely matters, reads pay a round trip.

- **Include `current_link_speed` in the health check**, not just latency probes: `LnkSta: Speed 2.5GT/s
  (downgraded)` against `LnkCap: 32GT/s`, on **both** endpoint and root port. Without it, this state
  reads as "same degradation" when it is a different one.
- **`sock-read` GB/s is not a PCIe health indicator** — it measures the host reading the D2H socket out
  of host DRAM. It showed 17.28 GB/s on a Gen1 x16 link whose ceiling is ~4 GB/s. Check the arithmetic
  against the link rate first.
- Needs a **cold power cycle**; a warm reboot will not retrain the link.

---

## 4. VM FREEZE → IRD watchdog reboot

**Signature.** Presents as "the box rebooted" with SSH keys gone, but the **physical host never
reboots** — the reservation VM hard-freezes and the IRD/hypervisor watchdog restarts it. Kernel-silent:
two independent processes stop writing at the *same instant*, there are zero
`tenstorrent`/AER/DPC/MCE/hang/panic lines in the pre-reboot ring buffer, and `efi_pstore` is empty.

**Mechanism** (confirmed by stack-sampling): a host CPU stalls on an **uncompleted MMIO read or ioctl
to the card**, which cannot be software-timed-out → VM lockup → watchdog reboot.

**Repro.** Not deterministic. Reproduced on **bh-05 (2026-08-06)** during a DRISC
streaming profiler sweep: box went down mid-sweep, `last -x` showed `crash` with no `shutdown` in the
following boot. Run a long DRISC sweep and watch for the signature.

**Catch the mechanism instead of guessing:**

```bash
# ptrace_scope=0; gdb at /usr/local/bin/gdb
# sample the pid holding /dev/tenstorrent every 0.4 s, writing to /localdev (NOT /tmp — see below)
```

The last completed backtrace before the freeze names the stuck op. Identical stacks across consecutive
samples = genuinely hung, not transient.

**It costs more than the run in flight:** the bh-05 freeze came back **DEGRADED and stayed degraded**
through two later clean reboots (§3).

**Recovery gotchas after the VM reboot / container recreate**

- `/home/$USER` comes back **root-owned and unwritable**, which breaks both the venv (its uv-python
  interpreter lived in ephemeral `/home/.local`) and the tt-metal JIT firmware cache
  (`~/.cache/tt-metal-cache`). Repoint the venv `python` symlink at `/usr/bin/python3.10` and fix
  `pyvenv.cfg home=/usr/bin`, then `export HOME=/localdev/$USER` for runs.
- **`/localdev` persists; `/tmp` and `/home` do not.** Put diagnostics on `/localdev`.
- Password auth survives the recreate; the **injected pubkey does not**, so `tt run` / `tt git` (key
  auth) break until it is re-added.
- Repeated freezes can put the box into a reboot cycle. **Stop hammering it and let it settle.**

---

## Ruled out — do not resurrect

Egress bandwidth (saturates ~17 GB/s) · ingest/producer delay · cumulative runs · config churn · NoC
choice · host poll pressure (yield injected directly: 4/100 vs 3/100 over 300 randomized runs) · the
periodic device read (that comparison was masked teardowns) · static-TLB immunity to degradation ·
degradation-follows-a-hang · knee-as-safety-limit · `TT_METAL_OPERATION_TIMEOUT_SECONDS` as a fix.

**IOMMU page faults — DEAD (2026-08-07).** Decorrelated in both directions on one day: the last
`AMD-Vi IO_PAGE_FAULT` burst landed during a fully clean block, and two wedges logged no faults at all.
Faults without wedges, wedges without faults. The four near-zero IOVAs are real and unexplained but are
**not** the wedge.

## Method rules that would have prevented every wrong conclusion

- Every error came from a **coarse observable standing in for the real one** — exit code for failure
  mode, wall-clock for card health, arm label for causation, one block for a rate. The fix was never
  more runs; it was always a finer discriminator.
- **Randomize arm order, never alternate** (see §1).
- **Never `pkill -f <pattern>` on an IRD box.** `tt run` wraps commands in a `bash -c` whose command
  line contains your pattern, so pkill matches and kills your own ssh session. Kill by PID, filtering
  the list with `grep -av "bash -c"`.
- A "killed" run can survive as a `tt run` process tree and keep holding the device. The next *build*
  then deadlocks in `precompile_fw` (it opens the device) at 0% CPU, looking like a hung build. Check
  `ps -eo pid,args` before blaming the build.

## Source of record

`FINDINGS.md`, next to this file — read §N+21 first, it carries a status banner. That file holds the
full audit trail, including five claims retracted in one day with the reasoning. This runbook is the
short version: if the two ever disagree, FINDINGS.md wins.
