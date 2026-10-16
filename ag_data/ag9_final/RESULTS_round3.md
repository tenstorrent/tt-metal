# Round 3 results: factory crossover with the chosen policies (measurement only)

Same build and harness as rounds 1-2: commit 99932452e22, build_Release unchanged, BH 1x8 mesh, 2 links, packet 15232,
`test_ag_knobs`, AG_TOPO=both, AG_ITERS=20, tracy (`python -m tracy -r -p -v -m pytest`), TT_METAL_OPERATION_TIMEOUT_SECONDS=45.
Knob file `round3.txt`: `default` / `factory=unicast,k=auto,entry=one_packet` / `factory=multicast,k=auto,run_cap=0`.

| run | shapes | cells | pytest | report dir | analysis files |
|---|---|---|---|---|---|
| R3a | 8K,32K,65K,131K,512K,884K,901K,917K,1M,2M,2.6M,5M,10M,25M,50M,100M | 96 | **96 passed** (13:19) | `generated/profiler/reports/2026_10_16_01_35_27` | r3a.txt/.csv, r3a_vsdef.txt/.csv, r3a_vsuc.txt/.csv, tab_r3a.md, log r3a.log |
| R3b | p64,p128,p256,p512,p1K,p2K,p4K,p8K | 48 | **48 passed** (5:18) | `generated/profiler/reports/2026_10_16_01_43_23` | r3b.txt/.csv, r3b_vsdef.txt/.csv, r3b_vsuc.txt/.csv, tab_r3b.md, log r3b.log |

- Every config has n=20, and the analyzer printed no desync or non-contiguous-block warnings. Multicast fit in L1 and did not hang at 50M or 100M, so no shape was dropped.
- **Missing shape ids.** `884K`, `901K`, `917K` and `2.6M` are in `test_ag_tensor_size`'s list but not in `_SHAPES`. I did not edit tests/. Instead, a pytest plugin, `perf/plugin/ag_extra_shapes.py` loaded with `-p ag_extra_shapes`, injects those 4 shapes into the module's `_SHAPES`. It copies them verbatim from `test_ag_tensor_size`. The runner is `perf/run3.sh <tag> <shapes>`.
- **Labels.** The analyzer prints the `default` tuning as `-`, so the default reference uses `--vs-default`; `--vs "default"` matches nothing. The pinned rows print as `factory=unicast,k=auto,entry=scatter_aware` (which is `entry=one_packet`) and `factory=multicast,k=auto` (which is `k=auto,run_cap=0`).
- **The default's factory was read from the ops CSV.** For each default row I read the `DATA MOVEMENT KERNEL SOURCE` column: `multicast_*` kernels or `unicast*` kernels. In every cell this matched what `heuristic_prefers_unicast` predicts from per-link bytes. Note that the default row runs the **default policies**: unicast k=1/legacy entry, or multicast k=1/run_cap 8192. The pinned rows run the chosen policies. "Default us" is therefore not the same thing as the pinned row of the same factory.
- **Verdict rule as in rounds 1-2.** A cell is decided when |delta| >= 5 % and the bootstrap CIs don't overlap; otherwise it is a tie. Shapes below 2 MB/link with |delta| < 10 % are inconclusive. Margin = (multicast - unicast) / unicast, so + means unicast is faster.
- **Heuristic** (all_gather_geometry.cpp `heuristic_prefers_unicast`, BH): `link_bytes = packed output bytes / num_links`. It picks unicast when `link_bytes >= threshold_kb * 1024`, with `threshold_kb = max(0, ring ? 320*(dev/2 - 2) : 384 + 544*(dev - 4))`. At 8 devices that is **ring 640 KiB = 655,360 B/link** and **line 2560 KiB = 2,621,440 B/link**.

## R3a: tensor-size ladder (tile, dim 3; stripe = W/8/32 chunks)

**ring** (heuristic threshold 640 KiB/link = 655360 B)

| shape | out bytes | per-link bytes | stripe | unicast us | multicast us | faster | margin % (mc vs uc) | verdict | default us | default factory | heuristic says | default picks faster? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8K | 16,384 | 8,192 | 1 | 15.6 | 17.1 | unicast | +9.3 | inconcl. (<10% small) | 16.8 | multicast | multicast | n/a (tie) |
| 32K | 65,536 | 32,768 | 4 | 16.6 | 17.4 | unicast | +4.8 | tie (noise) | 17.8 | multicast | multicast | n/a (tie) |
| 65K | 131,072 | 65,536 | 2 | 19.4 | 19.5 | unicast | +0.4 | tie (noise) | 18.9 | multicast | multicast | n/a (tie) |
| 131K | 262,144 | 131,072 | 4 | 21.7 | 20.3 | multicast | -6.7 | inconcl. (<10% small) | 20.5 | multicast | multicast | n/a (tie) |
| 512K | 1,048,576 | 524,288 | 8 | 28.3 | 29.2 | unicast | +3.2 | tie (noise) | 29.4 | multicast | multicast | n/a (tie) |
| 884K | 1,769,472 | 884,736 | 12 | 38.0 | 35.0 | multicast | -7.9 | inconcl. (<10% small) | 36.7 | unicast | unicast | n/a (tie) |
| 901K | 1,802,240 | 901,120 | 10 | 37.8 | 36.5 | multicast | -3.5 | tie (noise) | 37.7 | unicast | unicast | n/a (tie) |
| 917K | 1,835,008 | 917,504 | 14 | 38.0 | 35.1 | multicast | -7.6 | inconcl. (<10% small) | 38.5 | unicast | unicast | n/a (tie) |
| 1M | 2,097,152 | 1,048,576 | 2 | 38.7 | 40.7 | unicast | +5.4 | inconcl. (<10% small) | 40.9 | unicast | unicast | n/a (tie) |
| 2M | 4,194,304 | 2,097,152 | 2 | 60.4 | 62.6 | unicast | +3.8 | tie (noise) | 65.2 | unicast | unicast | n/a (tie) |
| 2.6M | 5,242,880 | 2,621,440 | 10 | 71.7 | 69.8 | multicast | -2.7 | tie (noise) | 67.1 | unicast | unicast | n/a (tie) |
| 5M | 10,485,760 | 5,242,880 | 20 | 94.7 | 118.2 | unicast | +24.8 | **unicast** | 94.3 | unicast | unicast | yes |
| 10M | 20,971,520 | 10,485,760 | 10 | 226.0 | 226.7 | unicast | +0.3 | tie (noise) | 209.3 | unicast | unicast | n/a (tie) |
| 25M | 50,331,648 | 25,165,824 | 32 | 376.8 | 512.4 | unicast | +36.0 | **unicast** | 377.5 | unicast | unicast | yes |
| 50M | 100,663,296 | 50,331,648 | 32 | 736.6 | 1006.6 | unicast | +36.7 | **unicast** | 732.1 | unicast | unicast | yes |
| 100M | 201,326,592 | 100,663,296 | 32 | 1449.0 | 2000.7 | unicast | +38.1 | **unicast** | 1444.2 | unicast | unicast | yes |

**linear** (heuristic threshold 2560 KiB/link = 2621440 B)

| shape | out bytes | per-link bytes | stripe | unicast us | multicast us | faster | margin % (mc vs uc) | verdict | default us | default factory | heuristic says | default picks faster? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8K | 16,384 | 8,192 | 1 | 22.1 | 19.1 | multicast | -13.6 | **multicast** | 19.2 | multicast | multicast | yes |
| 32K | 65,536 | 32,768 | 4 | 24.4 | 20.1 | multicast | -17.6 | **multicast** | 20.3 | multicast | multicast | yes |
| 65K | 131,072 | 65,536 | 2 | 29.3 | 22.1 | multicast | -24.7 | **multicast** | 21.7 | multicast | multicast | yes |
| 131K | 262,144 | 131,072 | 4 | 33.0 | 25.0 | multicast | -24.2 | **multicast** | 24.9 | multicast | multicast | yes |
| 512K | 1,048,576 | 524,288 | 8 | 42.7 | 41.8 | multicast | -2.1 | tie (noise) | 42.4 | multicast | multicast | n/a (tie) |
| 884K | 1,769,472 | 884,736 | 12 | 56.0 | 53.0 | multicast | -5.4 | inconcl. (<10% small) | 57.1 | multicast | multicast | n/a (tie) |
| 901K | 1,802,240 | 901,120 | 10 | 57.4 | 55.1 | multicast | -4.0 | tie (noise) | 58.2 | multicast | multicast | n/a (tie) |
| 917K | 1,835,008 | 917,504 | 14 | 56.6 | 50.9 | multicast | -10.1 | **multicast** | 53.5 | multicast | multicast | yes |
| 1M | 2,097,152 | 1,048,576 | 2 | 62.6 | 64.6 | unicast | +3.2 | tie (noise) | 64.4 | multicast | multicast | n/a (tie) |
| 2M | 4,194,304 | 2,097,152 | 2 | 107.1 | 107.5 | unicast | +0.3 | tie (noise) | 108.4 | multicast | multicast | n/a (tie) |
| 2.6M | 5,242,880 | 2,621,440 | 10 | 129.0 | 123.5 | multicast | -4.3 | tie (noise) | 121.3 | unicast | unicast | n/a (tie) |
| 5M | 10,485,760 | 5,242,880 | 20 | 175.6 | 187.5 | unicast | +6.7 | **unicast** | 175.7 | unicast | unicast | yes |
| 10M | 20,971,520 | 10,485,760 | 10 | 480.7 | 427.4 | multicast | -11.1 | **multicast** | 429.8 | unicast | unicast | **no** |
| 25M | 50,331,648 | 25,165,824 | 32 | 744.4 | 787.2 | unicast | +5.7 | **unicast** | 745.5 | unicast | unicast | yes |
| 50M | 100,663,296 | 50,331,648 | 32 | 1475.2 | 1556.6 | unicast | +5.5 | **unicast** | 1477.1 | unicast | unicast | yes |
| 100M | 201,326,592 | 100,663,296 | 32 | 2930.7 | 3093.6 | unicast | +5.6 | **unicast** | 2926.8 | unicast | unicast | yes |

## R3b: page-size ladder (RM, dim 0, 16.8 MB output = 8.4 MB/link)

**ring** (heuristic threshold 640 KiB/link = 655360 B)

| shape | out bytes | per-link bytes | page B | unicast us | multicast us | faster | margin % (mc vs uc) | verdict | default us | default factory | heuristic says | default picks faster? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p64 | 16,777,216 | 8,388,608 | 64 | 145.1 | 176.4 | unicast | +21.6 | **unicast** | 145.6 | unicast | unicast | yes |
| p128 | 16,777,216 | 8,388,608 | 128 | 144.9 | 177.0 | unicast | +22.2 | **unicast** | 144.9 | unicast | unicast | yes |
| p256 | 16,777,216 | 8,388,608 | 256 | 143.6 | 176.8 | unicast | +23.1 | **unicast** | 144.2 | unicast | unicast | yes |
| p512 | 16,777,216 | 8,388,608 | 512 | 145.7 | 176.3 | unicast | +21.1 | **unicast** | 143.5 | unicast | unicast | yes |
| p1K | 16,777,216 | 8,388,608 | 1024 | 142.8 | 175.6 | unicast | +23.0 | **unicast** | 142.5 | unicast | unicast | yes |
| p2K | 16,777,216 | 8,388,608 | 2048 | 142.8 | 176.0 | unicast | +23.3 | **unicast** | 143.1 | unicast | unicast | yes |
| p4K | 16,777,216 | 8,388,608 | 4096 | 143.3 | 177.0 | unicast | +23.5 | **unicast** | 152.0 | unicast | unicast | yes |
| p8K | 16,777,216 | 8,388,608 | 8192 | 142.4 | 175.4 | unicast | +23.2 | **unicast** | 185.0 | unicast | unicast | yes |

**linear** (heuristic threshold 2560 KiB/link = 2621440 B)

| shape | out bytes | per-link bytes | page B | unicast us | multicast us | faster | margin % (mc vs uc) | verdict | default us | default factory | heuristic says | default picks faster? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p64 | 16,777,216 | 8,388,608 | 64 | 246.1 | 274.9 | unicast | +11.7 | **unicast** | 246.7 | unicast | unicast | yes |
| p128 | 16,777,216 | 8,388,608 | 128 | 246.2 | 274.1 | unicast | +11.3 | **unicast** | 246.8 | unicast | unicast | yes |
| p256 | 16,777,216 | 8,388,608 | 256 | 245.7 | 274.9 | unicast | +11.9 | **unicast** | 246.7 | unicast | unicast | yes |
| p512 | 16,777,216 | 8,388,608 | 512 | 249.6 | 275.6 | unicast | +10.4 | **unicast** | 249.6 | unicast | unicast | yes |
| p1K | 16,777,216 | 8,388,608 | 1024 | 252.0 | 279.8 | unicast | +11.0 | **unicast** | 252.3 | unicast | unicast | yes |
| p2K | 16,777,216 | 8,388,608 | 2048 | 252.1 | 279.6 | unicast | +10.9 | **unicast** | 252.2 | unicast | unicast | yes |
| p4K | 16,777,216 | 8,388,608 | 4096 | 253.6 | 280.0 | unicast | +10.4 | **unicast** | 277.1 | unicast | unicast | yes |
| p8K | 16,777,216 | 8,388,608 | 8192 | 252.6 | 281.1 | unicast | +11.3 | **unicast** | 362.4 | unicast | unicast | yes |

## Measured crossover

### Ring (8 dev, current threshold 640 KiB/link)
- **Multicast never wins decisively anywhere on the ring.** Its best margins are -7.9 % (884K), -7.6 % (917K) and -6.7 % (131K), and all three are inconclusive under the small-shape rule.
- **Unicast wins decisively at 5.2 MB/link and above:**

  | shape | per-link bytes | unicast margin |
  |---|---|---|
  | 5M | 5.2 MB | +24.8 % |
  | 25M | 25 MB | +36.0 % |
  | 50M | 50 MB | +36.7 % |
  | 100M | 100 MB | +38.1 % |
  | all 8 page-size shapes | 8.4 MB | +21 to +24 % (flat across 64 B to 8 KB pages) |

  The exception is 10M (10.5 MB/link, stripe 10), which ties at +0.3 %.
- **From 8 KB to 2.6 MB/link the result is a tie band with no consistent sign:**
  - 8K-65K lean unicast (+0.4 to +9 %).
  - 131K leans multicast.
  - 512K leans unicast.
  - 884K-917K (0.88-0.92 MB/link, stripes 10-14) lean multicast (-3.5 to -7.9 %).
  - 1M and 2M (stripe 2) lean unicast (+5.4 / +3.8 %).
  - 2.6M (stripe 10) leans multicast (-2.7 %).
- **Measured crossover:** unicast becomes reliably better somewhere between 2.6 and 5.2 MB/link. Below that, the factory choice is worth at most ~8 %, and none of those differences clears the noise bar.

### Linear (8 dev, current threshold 2560 KiB/link)
- **Multicast wins decisively at the small end:**

  | shape | per-link bytes | multicast margin |
  |---|---|---|
  | 8K | 8 KB | -13.6 % |
  | 32K | 33 KB | -17.6 % |
  | 65K | 66 KB | -24.7 % |
  | 131K | 131 KB | -24.2 % |
  | 917K | 0.92 MB | -10.1 % |

  Also leaning multicast: 512K (-2.1 %, noise), 884K (-5.4 %, inconclusive) and 901K (-4.0 %, noise).
- **Tie band from 1.0 to 2.6 MB/link:**
  - 1M: +3.2 %.
  - 2M: +0.3 %.
  - 2.6M (stripe 10): -4.3 %.
- **Unicast wins decisively at 5.2 MB/link and above:**
  - 5M: +6.7 %.
  - 25M, 50M and 100M: +5.5 to +5.7 %.
  - All 8 page-size shapes (8.4 MB/link): +10.4 to +11.9 %.
- **The exception is 10M** (10.5 MB/link, stripe 10): multicast is 11.1 % faster than *pinned* unicast. But the **default** unicast (k=1, legacy entry) runs 429.8 us, which ties multicast's 427.4 us. So this is a defect of the chosen unicast policy at stripe 10, not a factory crossover. See the policy note below.
- **Measured crossover:** multicast is better up to about 0.9 MB/link, there is a tie band from 1.0 to 2.6 MB/link, and unicast is better from 5.2 MB/link (excluding the stripe-10 policy outlier).

### Comparison with the current thresholds, and recommendation
- **Ring 640 KiB/link.** It sits in the tie band, so it is not contradicted. Moving it anywhere between about 0.9 and 2.6 MB/link is not contradicted either.
  - **Recommendation:** keep a single per-link threshold. 640 KiB is acceptable.
  - **Optional:** 1 MiB/link (1,048,576 B) would hand 884K/901K/917K to multicast, worth about 4-8 %, inconclusive. It would keep 1M on unicast (+5.4 %).
  - The data does not support a decisive move. A threshold of 0 (always unicast) would cost only noise-level amounts at the small end on the ring.
- **Line 2560 KiB/link.** It sits inside the measured tie band (1.0-2.6 MB/link), below the first decisive unicast win (5.2 MB/link) and above the last decisive multicast win (0.92 MB/link).
  - **Recommendation: keep 2560 KiB/link.** Anything from about 1 MiB to 5 MB/link fits the data equally well.
- **Finer rules (stripe length, page size).**
  - **Page size:** no effect. The margin is flat from 64 B to 8 KB pages in both topologies.
  - **Stripe length:** there is a hint. Stripe-2 shapes (1M, 2M) lean unicast and stripe 10-14 shapes lean multicast in the tie band. But every one of those differences is under 8 %, and the stripe-10 cases are confounded by the unicast-policy regression below. **The data does not support a stripe-dependent threshold.** One per-link threshold per topology is enough.

### Does the heuristic default already pick the faster factory?
- **Yes in every decided cell, with one exception: linear 10M.** Counting cells where the pinned A/B is decided (|delta| >= 5 % and CIs separate, not inconclusive), the default's factory is the faster one in:
  - ring: 4/4 tensor cells and 8/8 page cells.
  - linear: 9/10 tensor cells and 8/8 page cells.
- **Linear 10M is not a real factory miss.** The default runs unicast with its *default* policies at 429.8 us, which ties pinned multicast (427.4 us). The miss only appears against the pinned unicast row.
- **Every other cell is a tie or inconclusive,** so either factory is acceptable there.

### Side finding: the chosen unicast policy regresses at stripe 10 (outside the brief, flagged for round 4)
The chosen unicast policy is `k=auto,entry=one_packet`. Default unicast is k=1 with legacy entry. Default vs pinned unicast, same report:

| shape | stripe | ring: pinned vs default | linear: pinned vs default |
|---|---|---|---|
| 10M | 10 | +8.0 % slower (226.0 vs 209.3) | +11.9 % slower (480.7 vs 429.8) |
| 2.6M | 10 | +6.8 % slower (71.7 vs 67.1) | +6.3 % slower (129.0 vs 121.3) |

On the ring at 10M, the default (209.3 us) also beats pinned multicast (226.7 us) by 8 %. Stripe 10 is where the `q_s10` regressions were recorded earlier. The full-sweep comparison of the new defaults against the baseline should include a stripe-10 shape.

Elsewhere the chosen unicast policy ties or beats the default unicast. Pinned unicast is much faster than default unicast at large pages:
- p8K ring: 142.4 vs 185.0 us (-23 %).
- p8K linear: 252.6 vs 362.4 us (-30 %).
- p4K ring: -5.7 %.
- p4K linear: -8.5 %.
