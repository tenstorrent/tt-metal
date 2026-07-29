# Warm-solve / minimal-host mode comparison

Per (host-size mock, shape, stage): **hosts landed / intermesh-solve time**. `⚠` = did not reach k_min. `TMO`=timeout(150s), `fail`=no solution/greedy dead-end. k_min = arithmetic minimum hosts.

Modes: **baseline**(warm+softdescent+hardcap) · **skipdescent**(warm+hardcap) · **greedy**(construct+verify, no SAT) · **hardcap-only**(cold all-or-nothing, 1 solve) · **atmostk**(cold ≤k counter, 1 solve).

| mock | shape | stage | k_min | baseline | skipdescent | greedy | hardcap-only | atmostk |
|---|---|---|---|---|---|---|---|---|
| SC16 | 2x4 | 16 | 4 | 4h/3ms | 4h/3ms | 4h/3ms | 4h/3ms | 4h/3ms |
| SC16 | 2x4 | 32 | 8 | 9h⚠/744ms | 9h⚠/336ms | 0h⚠/40ms | 0h⚠/183ms | 0h⚠/597ms |
| SC16 | 2x4 | 64 | 16 | 16h/59ms | 16h/59ms | 16h/11ms | 16h/59ms | 16h/59ms |
| SC16 | 4x4 | 8 | ? | TMO | TMO | TMO | TMO | TMO |
| SC16 | 4x4 | 16 | ? | TMO | TMO | TMO | TMO | TMO |
| SC16 | 4x4 | 32 | ? | TMO | TMO | TMO | TMO | TMO |
| SC20 | 2x4 | 16 | 4 | 4h/5ms | 4h/4ms | 0h⚠/8ms | 4h/4ms | 4h/4ms |
| SC20 | 2x4 | 32 | 8 | 8h/1196ms | 8h/35ms | 0h⚠/16ms | 8h/99ms | 8h/244ms |
| SC20 | 2x4 | 64 | 16 | 16h/9986ms | 16h/2281ms | 0h⚠/1074ms | 16h/10794ms | TMO |
| SC20 | 2x4 | 80 | 20 | 20h/19323ms | 20h/19402ms | 0h⚠/9717ms | 20h/19066ms | 20h/19370ms |
| SC20 | 4x4 | 8 | 8 | 8h/2ms | 8h/2ms | 0h⚠/2ms | 8h/2ms | 8h/2ms |
| SC20 | 4x4 | 16 | 16 | 16h/3ms | 16h/3ms | 0h⚠/4ms | 16h/3ms | 16h/3ms |
| SC20 | 4x4 | 32 | 32 | 32h/4ms | 32h/5ms | 0h⚠/6ms | 32h/4ms | 32h/5ms |
| SC20 | 4x4 | 40 | 40 | 40h/54ms | 40h/55ms | 0h⚠/475ms | 40h/54ms | 40h/53ms |
| SC36 | 2x4 | 16 | 4 | 4h/24ms | 4h/17ms | 0h⚠/18ms | 4h/43ms | 4h/43ms |
| SC36 | 2x4 | 32 | 8 | 8h/3230ms | 8h/480ms | 0h⚠/61ms | 8h/552ms | 8h/3003ms |
| SC36 | 2x4 | 64 | 16 | TMO | 16h/18544ms | 0h⚠/2239ms | 16h/11316ms | TMO |
| SC36 | 2x4 | 80 | 20 | 20h/111303ms | 20h/25736ms | 0h⚠/2015ms | TMO | TMO |
| SC36 | 2x4 | 96 | 24 | TMO | TMO | 0h⚠/1868ms | 24h/36786ms | TMO |
| SC36 | 2x4 | 112 | 28 | TMO | TMO | 0h⚠/1447ms | TMO | TMO |
| SC36 | 2x4 | 128 | 32 | TMO | TMO | TMO | TMO | TMO |
| SC36 | 2x4 | 144 | 36 | TMO | TMO | TMO | TMO | TMO |
| SC36 | 4x4 | 8 | 8 | 8h/4ms | 8h/4ms | 0h⚠/4ms | 8h/5ms | 8h/4ms |
| SC36 | 4x4 | 16 | 16 | 16h/6ms | 16h/6ms | 0h⚠/12ms | 16h/6ms | 16h/6ms |
| SC36 | 4x4 | 32 | 32 | 32h/10ms | 32h/10ms | 0h⚠/13ms | 32h/10ms | 32h/9ms |
| SC36 | 4x4 | 40 | 40 | 40h/12ms | 40h/11ms | 0h⚠/971ms | 40h/12ms | 40h/12ms |
| SC36 | 4x4 | 48 | 48 | 48h/16ms | 48h/14ms | 0h⚠/1348ms | 48h/13ms | 48h/13ms |
| SC36 | 4x4 | 56 | 56 | 56h/29ms | 56h/15ms | 0h⚠/670ms | 56h/19ms | 56h/29ms |
| SC36 | 4x4 | 64 | 64 | 64h/20ms | 64h/17ms | 0h⚠/530ms | 64h/21ms | 64h/23ms |
| SC36 | 4x4 | 72 | 72 | 72h/190ms | 72h/186ms | 0h⚠/626ms | 72h/188ms | 72h/227ms |

## Summary per mode

| mode | runs | success | landed on k_min | median intermesh_ms (successes) |
|---|---|---|---|---|
| baseline | 30 | 22 | 21 | 22 |
| skipdescent | 30 | 23 | 22 | 17 |
| greedy | 30 | 25 | 2 | 61 |
| hardcap-only | 30 | 23 | 22 | 21 |
| atmostk | 30 | 20 | 19 | 18 |

_Generated from RESULTS.raw (150 rows). Landed-on-k_min = optimal host packing achieved._

---

## Final conclusions (150/150 runs complete)

**4x4 shape is trivial** — k_min = n_target (each mesh its own host group, no packing), so every mode lands
optimal in ≤230 ms. The real minimal-host test is the **2x4 shape** (k_min < n_target).

**2x4 packing results (SC36):**

| stage (k_min) | baseline | skip-descent | hardcap-only | atmostk | greedy |
|---|---|---|---|---|---|
| 64 (16) | TMO | 16h/18.5s | **16h/11.3s** | TMO | fail |
| 80 (20) | 20h/111s | 20h/25.7s | TMO | TMO | fail |
| 96 (24) | TMO | TMO | **24h/36.8s** | TMO | fail |
| 112–144 (28–36) | TMO | TMO | TMO | TMO | fail |

**Key takeaways**
1. **The soft-descent loop is the villain** — baseline is the slowest and times out most; it re-proves
   feasibility one host at a time. Dropping it is a strict win.
2. **hardcap-only (cold all-or-nothing, one solve) is fastest when it works** (64: 11.3s vs baseline TMO), and
   **complementary** to skip-descent (each cracks packings the other times out on). Neither dominates → run
   hardcap-only first, fall back to skip-descent.
3. **atmostk (plain ≤k counter, one solve) is weak** — times out on every hard 2x4; the all-or-nothing
   tightening's unit propagation is what makes hardcap-only fast.
4. **greedy (constructive) reaches k_min on only 2/30** — its BFS traversal can't fill hosts with contiguous
   ring stages on larger rings (a DFS/linear-walk order fixes this; identified, not applied). Where it does
   construct, it's ms-fast.
5. **The largest 2x4 packings (112–144 stages) are intractable for every mode within 150 s** — SAT minimal-host
   is genuinely hard there; a structural/greedy method (with the traversal fix) is the only path.
6. **Smaller mocks (SC16/SC20):** hardcap-only can return UNSAT and fall back — the exact-fill packing isn't
   always present in the physical adjacency.
7. **Enumeration (--max-solutions 20):** works on tractable cases (4x4-32 → 20 sols / 57 s); slow on hard
   packings (2x4-64 → 1 sol, 300 s timeout) even though a single solution takes 11 s.

**Recommendation:** eliminate the soft-descent loop; use **hardcap-only → skip-descent fallback**; fix greedy's
traversal order (BFS→DFS) to unlock the ms-fast path on regular rings; treat 2x4 ≥112 as needing a
non-SAT/structural approach.
