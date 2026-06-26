# X280 bare-metal — primer (names, terms, and why ILP works)

Orientation for the X280 bare-metal work: what each firmware/example is, the key
terms, and the reasoning behind the headline result (why ILP broke the "530 MB/s
wall"). Companion to `FINDINGS.md` (the measured results). Covers
`tools/x280_bm/` and `tt_metal/programming_examples/profiler/test_x280_*`.

## A note on the names

The firmware/example names below are **descriptive labels coined while building
each one** — they are *not* official tt-metal terminology. Rename freely if you'd
prefer something house-style; they're just file/target names. (The earlier
Linux-era tools under `tools/tracy/x280/` — `pollall`, `pollalln`, `pollmt` — were
named separately.)

Each item is a pair: a bare-metal FW source `tools/x280_bm/src/<name>.c` (the code
that runs on the X280 harts) and a host driver
`tt_metal/programming_examples/profiler/test_x280_<name>/` (a C++ program that boots
the FW via the low-level `Cluster` and reports results).

| Name | Mnemonic | What it does |
|---|---|---|
| `counter` | — | §1 single-hart heartbeat: write an incrementing counter to LIM; host polls it. First proof the FW boots. |
| `poller` | — | §2 single-counter poll-rate: one hart reads one Tensix L1 word over the NoC in a tight loop; measures read latency. |
| `dma_probe` | — | §3 DMA NoC→LIM copy + re-trigger benchmark (the X280's Synopsys DMA engine). |
| `grid_drain` | — | §4 1-DMA-channel drain of all 110 cores' L1. |
| `grid_drain4` | — | §5 multi-channel DMA drain (found the DMAC has 2 channels). |
| `poll4` | 4-hart **poll** | §6 all 4 harts busy-poll 1/4 of the 110-core grid each, 1 flit/core. The original 530 MB/s result. |
| `noc1_probe` | **NoC1** probe | §8 how the X280 reaches a Tensix tile on the 2nd NoC (NOC1). |
| `poll4n1` | poll4 + **N**oC**1** | §9 split the 4 harts across NOC0/NOC1; also a per-read burst-depth knob. |
| `poll6n1` | "6 pipes" on **N**oC**1** | §10 4 harts + 2 DMA channels concurrently — the NIU-vs-mesh decider. |
| `pollmp` | poll via **m**emory-**p**ort | §11 sequential-stream experiment: cached Memory Port (#3) + static VC (#4) + the ILP sweep that revealed the real lever. |
| `gridilp` | **grid** drain + **ILP** | §12 the real profiler read: each hart drains its slice of all 110 cores with multiple NoC reads in flight (ILP). 1.5 GB/s. |
| `d2hbw` | **d**evice-**2**-**h**ost **b**and**w**idth | §13 X280 fabricates packets and posted-writes them to host memory through the PCIe tile. ~3 GB/s. |
| `profrelay` | **prof**iler **relay** | §14 close-the-loop snapshot: drains each core's real device-profiler L1 timestamps, relays to host D2H. |
| `profcons` | **prof**iler **cons**umer | §15 continuous SPSC consumer: drains all rings live, advances heads (unblocks producers), relays to host. Lossless, flow-controlled. |
| `profcons_split` | profcons + reader/relay **split** | §15 fast consumer: reader harts → LIM staging ring → dedicated relay hart; two-sided ILP-4, reads NOC0 / writes NOC1 → 1097 MB/s. |
| `kernel_profiler_push.hpp` / `kernel_profiler.hpp` | — | the device profiler producer: original DRAM-push backend (kept as `_push`) vs the new SPSC-ring backend (`kernel_profiler.hpp`) that the X280 drains. |

(§ numbers refer to `FINDINGS.md`.)

## Key terms

- **LIM — Loosely Integrated Memory.** A SiFive term for the block of addressable
  scratchpad SRAM in the core complex (on the L2CPU, the L2 region used as
  directly-addressed memory rather than cache; distinct from the per-hart
  Tightly-Integrated ITIM/DTIM). On the Blackhole **X280 L2CPU tile**, LIM is the
  X280's local on-tile SRAM at **`0x08000000`** — the X280's private working RAM. It
  holds: the **firmware** (linked + loaded there; reset vectors point the harts at
  it), the **per-hart stacks**, the **mailboxes** (params in at `0x08011000`,
  results out at `0x08011040`, coord table at `0x08011200`), and the **drained data**
  in the read examples. Boot flow: the host NoC-writes the FW + params into LIM,
  releases the harts (which boot from LIM), and the harts read/write the LIM
  mailbox. LIM is distinct from the Tensix cores' L1 and from host memory.

- **X280 / L2CPU.** The SiFive X280 RISC-V core complex that occupies one Blackhole
  NoC tile (the "L2CPU" tile). 4 harts, in-order, with the RVV vector extension.
  We run bare-metal firmware on it (no OS).

- **hart.** A RISC-V hardware thread (CPU core). The X280 tile has 4.

- **flit.** The 64-byte NoC transfer unit. One `vle64.v`/`vse64.v` vector
  load/store moves exactly one flit in a single NoC transaction.

- **NoC.** The on-chip Network-on-Chip mesh that connects all tiles. Blackhole has
  two: **NOC0** and **NOC1**, each reached through its own NIU.

- **NIU.** NoC Interface Unit — a tile's port onto a NoC (the X280 tile has one per
  NoC).

- **TLB window.** A 2 MiB memory-mapped window the X280 programs to address a remote
  tile's NoC address space. Loads/stores through the window become NoC
  reads/writes. We pre-map one window per target core.

- **System Port vs Memory Port.** Two ways the X280 reaches a TLB window: the
  **System Port** (uncached, what we use) and the **Memory Port** (cached/coherent
  alias). §11 found the cached port is *slower* for this access pattern.

- **ILP — Instruction-Level Parallelism.** Issuing several independent NoC reads
  (to different windows) *before* consuming any result, so multiple transactions
  are in flight at once instead of one-at-a-time. The lever that lifted the
  per-hart rate past the latency bound and broke the "530 MB/s wall."

- **D2H — Device-to-Host.** The X280 writing into host pinned memory (sysmem),
  routed through the **PCIe tile** (a NoC tile that bridges NoC writes to host over
  PCIe). Write-only: a NoC *read* through the PCIe tile hangs the in-order hart.

- **profiler_msg_t.** The per-Tensix-core device-profiler buffer in L1
  (`control_vector[32]` + 5 per-RISC buffers of 2048 B = 10368 B), holding the
  FW/kernel start/end timestamps that `profrelay` drains.

## How the X280 reaches the NoC — TLB windows, not NIU commands

This is the foundation the throughput results rest on, and it differs from Tensix.

**The TLB does the NoC addressing; the hart only issues plain loads/stores.** A hart
**never names a NoC coordinate in an instruction**. The only way it reaches the NoC:

1. Software pre-programs a **2 MiB TLB window** descriptor (config regs at
   `0x2FF00000`) with the destination — NoC **x/y coord**, remote **address**
   (`addr>>21`), **NoC selector** (NOC0/1), and the **posted / ordering / VC** bits.
2. The hart does an ordinary **load/store to a CPU address inside that window's
   aperture** (System Port base `0x430000000 + window*2MB + offset`).
3. The tile's NoC-access hardware matches the address to the descriptor, **forms the
   NoC packet** (read request or write), and the **NIU injects it** onto the mesh;
   the read response comes back and completes the load.

So: **TLB = addressing/translation + routing/ordering attributes** (turns "a memory
access in this aperture" into "a NoC transaction to (x,y,addr)"); **NIU = the
physical NoC port** that transports it. The hart only ever sees memory loads/stores.
Even the **DMA engine** addresses the NoC *through* TLB windows — there is no
TLB-less NoC path.

**Harts cannot software-drive the NIU to post async reads.** On **Tensix**, the RISC
cores program the NIU **command buffers** directly (`noc_async_read` writes
src/dst/size into NIU registers, fires, polls a counter later — explicit
software-issued async). On the **X280 that path does not exist** — proven
empirically (`noc1_probe`/`x280_niu_read` lineage): programming a cmd buffer like a
Tensix `ncrisc_noc_fast_read` and firing it gave `CMD_ACCEPTED += 0`,
`RD_REQ_SENT += 0` — the NIU front-end never accepts a software-issued command. The
TLB-window memory-mapped load/store is the **only** NoC mechanism for the hart.

**So what is the "async"/overlap we exploit?** (Important: "ILP helped" does *not*
mean we issued async NIU reads — we can't.)
- A TLB-window read is a **synchronous CPU load**. The in-order hart can't
  "issue-and-poll-later" at the NIU. But it stalls at the instruction that *uses*
  the result, not at issue — so issuing several independent loads to *different*
  windows before consuming any keeps multiple loads outstanding **in the CPU
  LSU/memory pipeline**, and they overlap in the NoC. That is **load-level
  concurrency**, not NIU-command async — and it's the entire ILP trick (`gridilp`).
- **Writes are non-blocking**: a store to a *posted* TLB window is fire-and-forget
  (accepted into the store buffer/NIU; the hart moves on) — why one hart saturates
  the D2H write path.
- **True core-free async** comes only from the **DMA engine** (Synopsys DMAC, 2
  channels): a separate engine that runs transfers without the hart, but still
  TLB-addressed and slower per byte — its value is offload, not peak throughput.

## Why ILP helped — the "530 MB/s wall" was an artifact

**The wall.** The early grid poll (`poll4`) measured ~530 MB/s aggregate (≈135
MB/s × 4 harts) and that number looked like a hardware ceiling. It wasn't — it was
an artifact of *how the loop was written*, not of the NoC or the tile.

**Why one hart was stuck at ~135 MB/s.** poll4's inner loop was
**read a core → use the value (store it) → move to the next core**. The X280 is an
**in-order** core: it stalls at the instruction that *uses* a load result until
that result arrives. Because each iteration used the value immediately, the hart
could only ever have **one NoC read outstanding** — it issued a read, stalled the
full ~250 ns NoC round-trip, then issued the next. Throughput = flit ÷ round-trip =
latency-bound, independent of how wide the link is. Four harts each doing this →
4 × 135 ≈ 530. The link was nowhere near full; the harts were just *waiting*.

**The fix — ILP (multiple reads in flight).** A NoC read's latency is mostly
*transit time*, and the NoC/NIU can carry many requests concurrently. The in-order
core stalls at the *use*, not at the *issue* — so if you **issue several independent
reads before using any of them**, they all enter the network and overlap; you pay
~one round-trip for the whole batch instead of one per read. `gridilp` does exactly
this: each hart issues `ilp` `vle64` loads to `ilp` *different* cores' TLB windows,
then drains all of them:

```
vle64 v0,(core c+0)   ┐ issued back-to-back, no dependency between them,
vle64 v1,(core c+1)   │ so all `ilp` reads are in flight at once
vle64 v2,(core c+2)   │ (the core only stalls at the first vmv/store below)
vle64 v3,(core c+3)   ┘
store v0..v3          ← consume them after they've overlapped in the network
```

The reads must target **different** windows and be **independent** (no result feeds
the next address) — that's what lets them overlap. The `pollmp` sweep made the
mechanism unmistakable: single hart, sequential, ILP 1→2→4→8 = 246→447→793→914 MB/s.
Throughput tracks *reads-in-flight* almost linearly.

**Why a wider single load did NOT help (ILP ≠ bigger reads).** An obvious-looking
alternative — read 8 flits in one fat `vle64` — was *worse* (274 MB/s). The vector
LSU walks a multi-flit load as serial back-to-back NoC round-trips with no overlap,
so per-byte time just doubles. The win comes specifically from **multiple
independent in-flight transactions**, not from moving more bytes per instruction.
Likewise the **cached Memory Port** and **static-VC** tricks did nothing — the lever
is concurrency, not the port or the virtual channel.

**Scaling and the real ceiling.** With ILP, throughput scales by (harts × per-hart
ILP) until a shared resource saturates: the true read ceiling is **~1.8 GB/s**
(synthetic sequential stream), and the real 110-core scatter profiler hits **~1.5
GB/s at 2 harts × ILP 4** — 2.9× the old "wall." Push too hard and it *collapses*:
**≥3 harts at ILP ≥4** (≳12–16 outstanding from many issuers) thrashes the read port
back down to ~275 MB/s. Sweet spot: **1–2 harts, ILP 4.**

**On the write side it's automatic.** Posted NoC writes (the `d2hbw` D2H path) don't
stall the issuing hart at all — they're fire-and-forget into the NIU queue — so a
*single* hart already streams many writes in flight and saturates the PCIe-tile
egress at ~3 GB/s. There, more harts don't help (the egress is the bottleneck), so
the read-side ILP trick isn't needed.

**Takeaway.** On an in-order NoC client, the first-order throughput lever is
**number of outstanding transactions**, not bytes-per-instruction, port type, or VC.
The "530 MB/s wall" was a 1-outstanding-read-per-hart measurement artifact; ILP
exposes the ~3–4× of headroom that was always there.

## Why the DMA engine didn't help throughput (even in re-trigger mode)

The same "outstanding transactions" lesson explains why the Synopsys DMAC lost. A
single-channel DMA transfer fits `cycles ≈ 7,600 + ~571 × flits` — two parts:

- **Fixed setup ≈ 7,305 cycles** — channel reset + CTL/CFG init + TLB program +
  word-size descent + completion poll. *One-time per fresh setup.*
- **Steady-state ≈ 571 cycles per 64 B flit** — the actual streaming work.

**Re-trigger killed the setup term** (reuse a configured channel: restore
`block_ts` + SAR/DAR, clear int, kick — skip the ~7,305 cyc), which is why 2 KB went
80 → 112 MB/s (1.4×) and tiny transfers improved ~10×. **But it can't touch the
~571 cyc/flit steady-state term, and that is the wall** — re-trigger amortizes fixed
overhead, it doesn't make the transfer itself faster.

**Why ~571 cyc/flit is slow** (two compounding, structural reasons):
1. **One channel keeps only ONE NoC read outstanding** — it issues a burst, *waits*
   for the completion handshake, then the next. Same latency-bound trap as a single
   hart with one outstanding load, with **no overlap**. It can't do the ILP trick: a
   channel is inherently one-transaction-at-a-time.
2. **Per-burst engine/handshake overhead on top of NoC latency** — making the DMA
   channel actually *slower per byte than a single hart load* (~571 vs ~475 ns/flit).

**Only 2 channels, sharing one ingress.** Two channels gave 111 vs 96 MB/s —
**1.15×, not 2×** — because both funnel through the same NIU/mesh ingress into the
tile *and* the host must serially program+kick both before they overlap.

| Path | Rate | Why |
|---|---|---|
| 1 DMA channel, re-triggered | ~112 MB/s | 1 outstanding read + per-burst overhead |
| 2 DMA channels | ~111 MB/s | shared ingress + serial kicks |
| 1 hart, ILP 4 | 860 MB/s | multiple reads in flight |
| 2 harts, ILP 4 | 1534 MB/s | — |

**The DMA's value is offload, not throughput:** it drains the grid at ~111 MB/s with
the harts completely free. Beating the hart-ILP rate would need many channels each
deeply pipelining multiple outstanding NoC reads; the X280 DMAC (2 channels,
one-outstanding, software-handshake) doesn't have that. Harts win because they can
cheaply hold many loads in flight in the LSU; one DMA channel cannot.

## How would a normal Tensix RISC compare? (estimate — not yet measured)

Reasoned guess for the same two patterns on a Tensix data-movement RISC
(BRISC/NCRISC), to frame what the X280 numbers mean. **These are estimates from the
architecture, not measurements** — a quick `noc_async_read` / `noc_async_write`
kernel benchmark would give real figures.

| Path | X280 (measured) | Tensix RISC (estimate) | Why |
|---|---|---|---|
| Read from grid (scatter, 64 B/core) | ~1.5 GB/s | **~5–15+ GB/s (1 NCRISC)** | Tensix has the async NIU path the X280 lacks |
| Write to host (D2H) | ~3 GB/s | **~3 GB/s (similar)** | bottleneck is downstream of the issuer |

- **Reads: Tensix should far exceed the X280.** The X280's ~1.5–1.8 GB/s ceiling is
  *its own* handicap — no software NIU path, so reads are synchronous TLB-window
  loads overlapped only by the LSU-pipeline ILP trick. A Tensix RISC is a
  purpose-built NoC client: `noc_async_read` programs the NIU command buffers and
  the core continues, natively holding **many** outstanding NoC transactions (a pool
  of transaction IDs) with no LSU trickery. Scatter of 64 B × 110 still pays
  per-transaction overhead, so not full link BW, but a single NCRISC should reach
  several-to-~10+ GB/s, scaling with more issuers and climbing toward the NoC link
  rate for larger per-core reads — easily **5–10×+ the X280**.
- **Writes: roughly the same (~3 GB/s).** The X280's D2H ceiling is downstream of
  the issuer — the PCIe tile's NoC→PCIe→host egress (one hart already saturated it;
  more harts didn't help). A Tensix core writing posted flits to the same PCIe tile
  hits the same wall. Writes are posted (fire-and-forget) for both, so neither is
  issue-limited; the PCIe/host path is. (Tensix could edge higher only if part of the
  X280's 3 GB/s was its own injection rate, but D2H is fundamentally PCIe-bound.)

**The asymmetry is the point.** X280 = read 1.5 / write 3; Tensix would flip the
shape — reads jump way up (real async NoC engine), writes stay ~flat
(downstream-bound). This is exactly why the X280 is interesting as an *observability*
engine: a poor bulk-NoC client, but a free CPU that can run logic while draining at
"good enough" rates **without stealing Tensix cycles**.
