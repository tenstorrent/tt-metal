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
| `profrelay` | **prof**iler **relay** | §14 the close-the-loop: drains each core's real device-profiler L1 timestamps and relays them to the host D2H socket. |

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
