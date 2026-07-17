# Dedicated tiny-shape path for regime-A matmul — design

Motivation from the Mt≤8 campaign + the (rejected) reduction-tree experiment: the low-AI Pk=4 tiny shapes
(256×2048×512, 128×2048×512, 256×2048×1024, 256×2048×1536) top out at **~40–53% of 512 GB/s**, and neither
the picker (best config already chosen) nor the reduction tree (~1.6%, rejected) closes the gap. The gap is
the **fixed regime-A dataflow overhead**, not compute or DRAM bandwidth.

## Characterization (the floor)

`regime_a_campaign` ablation, full (mask 0) vs "all delivery+reduction skipped" (mask 15 = skip in1-read +
in0-read + in0-ring-forward + reduce), median of interleaved relaunches:

| shape (Pk4) | full µs | floor µs (mask15) | floor/full | full %512 | floor %512 |
|---|---|---|---|---|---|
| 256×2048×512 (Sm3) | 16.3 | 7.2 | 44% | 41% | 92% |
| 128×2048×512 (Sm2) | 11.3 | 5.0 | 44% | 47% | 108% |
| 256×2048×1024 (Sm2) | 22.4 | 11.9 | 53% | 50% | 95% |
| 256×2048×1536 (Sm3) | 30.1 | 17.2 | 57% | 53% | 92% |
| 256×6144×4608 (Sm2, big control) | 141.6 | 74.8 | 53% | 86% | 162% |

**Reading:** on the tiny shapes **~50% of the wall is the delivery+reduction machinery** (in0 ring
all-gather forwarding + split-K reduction chain + M-split in1 forwarding + their CB/semaphore sync). The
compute+structure floor is ~92% %512 — i.e. compute is cheap; the shape is *overhead*-bound. (The big
control's floor is 162% %512 = DRAM-read-bound, a different regime — it does NOT want this path.) Per-RISC
spans on these shapes are co-terminal (~15/15/15), the load-imbalance/overhead signature.

## What the overhead is, and what a tiny-shape path removes

The regime-A dataflow is built for DRAM-bandwidth-bound shapes: 8 banks each own an N-band; the **in0 ring
all-gather** rotates in0 shards so every core holds the full k-slice (amortising in0 DRAM traffic), and
**split-K + M-split** add core parallelism with a **reduction chain** + **in1 forwarding**. For a *tiny*
low-AI shape this machinery costs more (latency + sync of the ring/reduction/forward, largely serial at
small N_bpc) than the DRAM traffic it saves.

The obvious candidate — a "direct/local" dataflow (each core reads its in1 shard + its in0 rows directly, no
ring, Pk=1 no-reduction, Sm=1 no-forwarding) — was **checked against the evidence and does not hold up:**

1. **The 92% floor is not achievable.** mask 15 skips the in1/in0 DRAM reads, which a real path cannot — the
   reads are irreducible. Subtracting the unavoidable reads, the *removable* overhead is only in0-ring-forward
   (skip_in0_forward ≈ −10..−18%) + reduction (no_reduce ≈ −9%), i.e. ~25% upper bound, not ~50%.
2. **Direct per-core in0 read is worse, not better.** in0 is shared by all 8 banks' N-bands; the ring reads
   it ONCE and forwards. Reading it directly per core = **8× the in0 DRAM traffic**. in0-read is only ~3% of
   the wall (skip_in0_read −3%), so 8× ≈ ~24% > the ring's ~18% forward. The existing in0-delivery
   alternatives that cut ring rounds are **measured NEGATIVE here** (DIAG_IN0_SCATTER +21..+37%, IN0_REPL4
   +20..+30% on the tiny shapes) — consistent with the prototype record ("in0 delivery SOLVED; ring
   all-gather optimal; read-dedup/scatter/broadcast all NEGATIVE").
3. **The reduction + M-split are the *cost of parallelism the picker already chose as net-positive.*** The
   picker evaluated Pk=1 (no reduction) and Sm=1 (no forwarding) for these shapes and they were slower —
   the ring+split-K+M-split parallelism nets out ahead despite its overhead. Forcing Pk=1/Sm=1 to "remove
   overhead" throws away the parallelism and regresses.

**Conclusion: a dedicated direct/local tiny-shape path is NOT expected to help** — the "overhead" the floor
exposes is mostly (a) irreducible reads and (b) the intrinsic price of parallelising a tiny low-AI matmul
across 64–96 cores, which the picker has already balanced. The low-AI Mt≤8 ceiling (~40–53% %512) is close
to the practical floor for this architecture on this class of shape.

## Where the remaining gap actually is, and the real recommendation

The residual is **fixed per-launch overhead + inter-core sync on tiny work**, not a bad dataflow choice:
- These shapes are tiny in absolute terms (11–30 µs); a large constant fraction is kernel dispatch / CB &
  semaphore setup / ring+reduction fill-drain latency that does not amortise over so little compute.
- The levers that would move it are **outside the regime-A kernel**: (a) reducing fixed program-launch /
  dispatch overhead (a runtime/trace lever — e.g. these ops run inside a captured trace in production, which
  already amortises dispatch), and (b) **fusing these tiny matmuls with adjacent ops** at the model level so
  the fixed overhead is shared. Neither is a new regime-A dataflow.
- Within regime-A, the only untested micro-lever is a **fewer-cores variant** (trade parallelism for less
  ring/reduction fill-drain), but the picker already sweeps core count via (Pk,Ns,Sm) and lands here, so the
  expected headroom is small and would need a per-shape exhaustive core-count sweep to confirm — low priority.

**Recommendation: do NOT build the direct/local tiny-shape path** (the mechanism is measurably negative).
Treat the low-AI Mt≤8 shapes as near their practical floor; if these shapes matter for a model, pursue the
gain at the **trace/dispatch or op-fusion level**, not in a new regime-A kernel. Delivery, reduction, picker,
and placement tuning for this regime are now exhausted.
