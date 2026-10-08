# r06-b02-a01: uneven two-wave split: wave A takes 9 of 20 tile-rows (18 workers), wave B 11 (22 workers), so A's drain starts earlier and finishes as B's drain begins

## Motivation
r05-b01-a01 (best, 1.5696) splits the 20 tile-rows 10/10 into two AG waves. Its `waves.py` timeline at h7168
(µs from chip start): A read end 3.59, A drain 9.05-14.21, B read end 6.29, B drain 11.31-15.59, kernel end 15.74.
- The drains overlap from 11.31 to 14.21, and while both run the write side is DRAM-aggregate-bound (~350 GB/s,
  write window 6.5 µs for the whole output).
- B's drain start (~11.3) is pinned by B's chain: the whole-input read end (~6.3, aggregate-bound, the same no
  matter how rows split) + push + AG + go + gather (~5 µs).
- A's drain start is A read end + ~5.5 µs (A's chain), and A read end scales with A's share of the rows.

A simple bandwidth model (write at ~6.5 µs per full output, A drain start = 6.29*f + 5.46, B drain start 11.31)
reproduces the measured end for f = 0.5 (15.5 vs 15.59). The same model says f = 0.45 (9/20 rows) ends at about
14.9 µs: A's smaller drain starts ~0.3 µs earlier and is nearly done when B's starts, so less write volume is left
after 11.3 µs. The narrow shapes behave the same way (h3584: ~10.9 -> ~10.3 µs if not skew-bound).

## Mechanism
Make the wave split uneven: wave A gets a = floor(9R/20) rows (9 of 20), wave B gets the other b = R - a (11).
Every row is still split across 2 column-half workers, so there are still 40 workers.
- Host factory (`dit_fused_distributed_rmsnorm_program_factory.cpp`):
  - worker -> (wave, slot) uses a Bresenham interleave (2a A workers spread over 2R grid positions). It reproduces
    the old even/odd mapping when a = R/2.
  - Rows: wave A slot j -> row j/2, wave B slot j -> a + j/2.
  - The k-th wave-B worker gets its start-sem signal from A worker rank floor(k*a/b). So some A workers signal 2
    B workers (new reader role 3, second partner coords as RT args).
  - Wave span and page size are sized for the larger wave (22 slots -> 3456 B span, under the 4352 B payload).
  - The forwarder gets per-wave slot counts (CT), and its go-release table puts wave B at offset 2a.
- Forwarder (`dit_rmsnorm_wave_forwarder.cpp`): per-wave slot counts for the arrival threshold and the go loop.
- Reader (`dit_rmsnorm_fused_reader.cpp`): role 3 ups two partners' start sems at the same block.
- Writer and compute are unchanged. The stick slot layout, pair read and 8-partial combine don't depend on the
  wave sizes.

## Why this is not a repeat
- r05-b01-a01 / r05-b03-a01 use 10/10 splits. r05-b03-a01 #3 suggested the opposite skew (a smaller B) to shorten
  B's exposed drain. My model says B's drain start is pinned by the full-read end, so the lever is making *A*
  smaller so A's drain clears before B's begins.
- The 4-wave nodes (r05-b01-a02/a03, r05-b03-a02/a03) add AG rounds. This keeps 2 rounds and only moves the boundary.
- Orthogonal to the writer un-gate fix (r05-b01-a03 #1) and the wave-lead tuning (r05-b01-a01 #1). Both stack on
  top of this.

## Expected effect and risk
- Expected: -0.3..-0.7 µs on h6144/h7168, smaller and possibly noise on h3584/h4096, where cross-chip launch skew
  bounds A's go.
- Risks:
  - an 18-core wave A might be per-core read-capped (20-core waves were not; 10-core waves were), so A's read
    end wouldn't move;
  - B's larger drain could become the tail;
  - a slot/partner mapping bug would hang (a B worker never released) or fail PCC (wrong row).
- Check: `waves.py` A drain end ≈ B drain start; A read end ≈ 0.45 × total.
