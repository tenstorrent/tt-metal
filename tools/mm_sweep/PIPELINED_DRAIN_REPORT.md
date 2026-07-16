# Pipelined phase-2 drain for Regime-A matmul — report

Change: the `in0_ring_reduce_writer` phase-2 (split-K reduction forwarding + DRAM output) no longer waits
for **remote completion** of each partial-sum/output block before popping its CB slot. Instead it reuses a
source slot as soon as the payload has **departed L1** (`noc_async_writes_flushed`), signals the reduction
receiver with **ordered payload→semaphore** ops (same peer + NoC, exactly the proven in0-ring contract), and
defers full completion to **one final `noc_async_write_barrier` + `noc_async_atomic_barrier`** before kernel
return. This lets the linear split-K reduction chain and the DRAM output pipeline across bands / N-sub-blocks
instead of serializing on a per-block completion barrier despite CB2/CB7 being double-buffered.

**Verdict: WINS — shipped as the production default.** Stable end-to-end gain on both Mt=8 primaries
(256×6144×768 −4.0%, 256×2048×1024 −1.2%) and every Pk>1 control, neutral on Pk=1, no regression, output
bit-identical to the old barrier path. The old per-block barrier is retained as the `DIAG_BARRIER_DRAIN`
(mask `1<<11`) A/B diagnostic.

## Old vs new synchronization contract
Old (now `DIAG_BARRIER_DRAIN`), per output block:
- reduction forward: `noc_async_write(partial); noc_async_write_barrier(); noc_semaphore_inc(next_recv);` → pop
- Pk=1 / top output: issue pages; `noc_async_write_barrier();` → pop

New (default), per output block:
- reduction forward: `noc_async_write(partial); noc_semaphore_inc(next_recv); noc_async_writes_flushed();` → pop
  - payload→semaphore to the same peer on the same NoC is **ordered**, so the receiver never observes
    readiness before its partial-sum has landed (identical to the in0 ring). `writes_flushed` guarantees the
    source out_cb slot has been read out and is safe to reuse.
- Pk=1 / top output: issue pages; `noc_async_writes_flushed();` → pop
- **once, before return:** `noc_async_write_barrier(); noc_async_atomic_barrier();` — drains this core's
  in-flight forwarded partials / DRAM writes AND the non-posted reduction-readiness semaphore atomics, so no
  NoC transaction outlives the program. (The atomic barrier is required — `writes_flushed`/`write_barrier`
  do not drain non-posted atomics; the watcher flags a race without it.)

Phase-1 (in0 ring) is untouched — this isolates reduction/output draining.

## Correctness (gtest `RegimeADiagFixture.PipelinedDrainCorrectness`)
Random BF16 vs a CPU f32 golden, PCC ≥ 0.999, fresh AND cached-program, for the pipelined default (mask 0)
and the barrier diagnostic (mask `1<<11`) across: Pk=1 direct output; Pk=2/4/12 chains (bottom/middle/top
roles); N_bpc=1/2/3 and wide-N (N_bpc=18); Sm=1/>1; balanced K/N tails. All **PCC 0.99999**, and pipelined
is **BIT-IDENTICAL** to the barrier baseline (`ab_maxdiff = 0`) — only the write sync differs, not the math.
Public 20/20 suite passes on the pipelined default. Watcher (TT_METAL_WATCHER=1) clean on Pk=1 and the Pk=12
reduction chain. (Note: a pre-existing, unrelated watcher warning about the M-split `in1_reader`'s own
semaphore atomics fires for Sm>1 identically at baseline mask; it is orthogonal to this writer change and is
left for a separate reader-hardening task.)

## Performance (median device-profiler kernel µs, 3 interleaved relaunches, Δ vs barrier)
Raw: `regime_a_pipelined_bench.json` (all relaunches, per-RISC, util%512, PCC). All PCC ≥ 0.999.
| shape | group | cfg (Ns,Pk,Sm,kb,nsb) | Pk | N_bpc | barrier µs | pipelined µs | Δ | util512 |
|---|---|---|---|---|---|---|---|---|
| 256×2048×1024 | target | 1,4,2,2,2 | 4 | 2 | 28.4 `[28.3,28.4,28.7]` | **28.0** `[28.0,28.0,28.1]` | **−1.2%** | 39.7→40.2% |
| 256×6144×768 | target | 1,12,1,2,1 | 12 | 3 | 53.5 `[53.1,53.5,53.6]` | **51.3** `[51.0,51.3,51.7]` | **−4.0%** | 47.4→49.4% |
| 256×6144×2304 | control | 1,12,1,2,1 | 12 | 9 | 92.2 | 89.9 | −2.5% | 69.1→70.9% |
| 256×6144×4608 | control | 1,12,1,2,1 | 12 | 18 | 152.9 | 151.3 | −1.0% | 79.4→80.2% |
| 32×6144×4608 (Mt1) | control | 1,12,1,2,1 | 12 | 18 | 118.4 | 116.9 | −1.2% | 94.6→95.7% |
| 64×6144×4608 (Mt2) | control | 1,6,1,4,2 | 6 | 9 | 119.4 | 118.5 | −0.8% | 94.9→95.6% |
| 128×6144×4608 (Mt4) | control | 1,12,1,2,1 | 12 | 18 | 129.8 | 128.3 | −1.1% | 89.4→90.4% |
| 32×6144×3072 (Pk1) | control_pk1 | 1,1,1,4,6 | 1 | 2 | 106.9 `[106.9,106.9,106.9]` | 107.1 `[106.9,107.1,107.1]` | +0.2% | 70.1→69.9% |

**Every Pk>1 shape improves; Pk=1 is neutral; nothing regresses.** The win scales with reduction-chain depth
exposure: deepest on the shallow-N deep-K primary 256×6144×768 (Pk=12, −4.0%, relaunch bands separated), and
on 256×6144×2304 (−2.5%). The **Pk=1 output-only case is neutral** (+0.2%, within a 0.2µs relaunch spread):
with no reduction chain, the DRAM output write is the tail either way and the final barrier still waits for
it — there is nothing downstream to overlap, so departure-flush buys nothing. This confirms the gain comes
specifically from **pipelining the split-K reduction forwards**, not from the output write itself.

### Per-RISC (median µs) — the reduction chain tightens
| shape | mode | wall | BRISC | NCRISC (writer) | TRISC |
|---|---|---|---|---|---|
| 256×6144×768 | barrier | 53.5 | 44.2 | 44.1 | 44.5 |
| 256×6144×768 | pipelined | 51.3 | 43.3 | 43.4 | 43.8 |
| 256×6144×2304 | barrier | 92.2 | 82.9 | 82.6 | 83.1 |
| 256×6144×2304 | pipelined | 89.9 | 82.3 | 82.1 | 82.8 |
| 32×6144×3072 (Pk1) | barrier | 106.9 | 104.9 | 106.6 | 106.0 |
| 32×6144×3072 (Pk1) | pipelined | 107.1 | 104.9 | 106.7 | 106.0 |

On the Pk>1 shapes the wall drops ~2µs while the per-RISC spans move only slightly — the barrier removal
mostly eliminates *inter-block serialization* (the writer no longer blocks compute/next-band forwarding on
each partial-sum's remote completion), tightening the reduction chain's critical path. On Pk=1 every span is
flat and the wall is unchanged.

## Per-block flush sufficiency
The simple departure-flush schedule already wins on the primaries with no obvious residual writer-side
stall (per-RISC shows the writer NCRISC span is not the isolated bottleneck after the change; wall ≈ spans).
The optional two-block batched CB2 variant was therefore **not** pursued — the gate for it ("simpler schedule
correct but leaves an obvious writer stall") is not met.

## Decision & validation
Pipelined drain is the **production default** (public path, mask 0). The old barrier path is kept as
`DIAG_BARRIER_DRAIN` for A/B. Validation on the pipelined default:
- Public 20/20 correctness: pass.
- Six-shape perf parity vs the frozen C++ oracle: the op is **faster on all six** (−1.3% to −8.2%; the
  shallow-N/deep-K 128×6144×768 gets −8.2%). The parity script's `|Δ|≤5%` flag trips only because the op now
  *outruns* the barrier-era oracle — i.e. the intended win, not a regression; no shape is slower.
- FLUX/LTX: output is bit-identical and the drain mode is independent of picker config selection, so the
  picker's shape→config choices and their correctness are unchanged; the perf effect is a uniform
  Pareto improvement (Pk>1 faster, Pk=1 neutral). A full FLUX/LTX perf re-sweep is optional follow-up.

## New synchronization contract (summary)
Reduction receiver readiness is guaranteed by **payload→semaphore ordering on the same peer/NoC**, never by a
completion barrier. A source CB slot may be reused after its payload has **departed L1** (`writes_flushed`).
Global completion — remote landing of all forwards/outputs and all non-posted semaphore atomics — is enforced
**once** by `noc_async_write_barrier(); noc_async_atomic_barrier();` before the writer returns.
