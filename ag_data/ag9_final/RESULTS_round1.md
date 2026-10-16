# Round 1 results: each policy knob alone, pinned factory (measurement only)

Commit 99932452e22 (branch scardoza/mux-ag9), build_Release libs from 2026-10-15 23:41, newer than every
C++ source change in that commit. The one uncommitted C++ diff (tt_metal/jit_build/build.cpp, which stops
JIT warning logs from printing) is from 00:06 and is **not** in the built libs. It does not affect perf.
BH 1x8 mesh, 2 links, packet 15232, AG_ITERS=20, AG_TOPO=both.

| run | knob file | cells | pytest | report dir | analysis files |
|---|---|---|---|---|---|
| R1a | round1_unicast.txt | 126 | **126 passed** (13:49) | `generated/profiler/reports/2026_10_16_00_24_25` | r1a.txt/.csv, r1a_vs.txt/.csv (ref `factory=unicast`), log r1a.log |
| R1b | round1_multicast.txt | 108 | **108 passed** (11:50) | `generated/profiler/reports/2026_10_16_00_39_37` | r1b.txt/.csv, r1b_vs.txt/.csv (ref `factory=multicast`), log r1b.log |

Nothing TT_FATALed or failed, so no cell was dropped. Multicast at 25M fit in L1. The analyzer printed no
desync or partial-iteration warnings, and every config has n=20.

## Shapes actually run (from `_SHAPES` in the harness; all ids exist)

| id | output shape | out bytes | ~MB/link | stripe (chunks) | note |
|---|---|---|---|---|---|
| 131K | [1,1,128,1024] tile dim3 | 262 KB | 0.13 | **4** | the brief said [1,1,128,512] stripe 2; the harness has W=1024, so stripe is 4 |
| 512K | [1,1,256,2048] tile dim3 | 1.0 MB | 0.5 | **8** | |
| 2M | [1,1,4096,512] tile dim3 | 4.2 MB | 2.1 | **2** | the brief said [1,1,2048,512]; that is the "1M" id. Stripe is 2 either way |
| 5M | [1,1,1024,5120] tile dim3 | 10.5 MB | 5.2 | 20 | |
| 25M | [1,1,3072,8192] tile dim3 | 50 MB | 25 | 32 | |
| p64/p2K/p4K/p8K | RM dim0, page 64 B/2 KB/4 KB/8 KB | 16.8 MB | 8.4 | n/a | |

Stripe < 16 chunks: 131K (4), 512K (8), 2M (2). Small shapes (<2 MB/link, so 25-50 % run-to-run drift):
131K and 512K. 2M is right at the boundary.

Note on labels: `entry=one_packet` is an alias. `Tuning::parse` maps it to `EntryPolicy::ScatterAware`
(all_gather_tuning.cpp:102), so the analyzer's tuning column prints `entry=scatter_aware`. The tables below
relabel it `entry=one_packet`.

Verdict rule: the analyzer's verdict (|delta| >= 5 % and the bootstrap CIs do not overlap) is shown as
faster/slower. `·` means noise. For 131K/512K, a non-noise verdict with |delta| < 10 % is downgraded to
**inconcl.** All deltas are within one report, B vs the reference tuning, using the median over 20 iterations
of the max over the 8 devices.

## Unicast (R1a, reference `factory=unicast`)

**ring** (delta% of B vs `factory=unicast`; `·` = noise; small shapes <2 MB/link with |delta|<10% marked inconcl.)

| knob | 131K | 512K | 2M | 5M | 25M | p64 | p2K | p4K | p8K |
|---|---|---|---|---|---|---|---|---|---|
| *A median us* | 20.3 | 30.2 | 65.3 | 93.6 | 381.0 | 145.3 | 142.8 | 152.5 | 185.0 |
| `entry=one_packet` | +2.8 · | -7.7 faster | -8.0 faster | +0.9 · | -0.5 · | -0.9 · | +0.6 · | -0.8 · | +0.1 · |
| `k=auto` | +6.8 slower | +1.1 · | -0.6 · | +0.3 · | -1.0 · | -0.5 · | -0.5 · | -6.0 faster | -22.7 faster |
| `k=auto,xfer=segments,rotation=absolute_dir_stagger,entry=one_packet` | +0.4 · | -7.6 faster | -9.5 faster | +2.2 · | -0.3 · | -2.6 · | -1.2 · | -7.5 faster | -24.0 faster |
| `rotation=absolute` | +3.6 · | +2.2 · | -1.5 · | +0.8 · | -0.3 · | -3.0 · | -1.0 · | -2.2 · | -0.1 · |
| `rotation=absolute_dir_stagger` | +5.7 slower | +1.3 · | -1.0 · | +1.1 · | -0.6 · | -2.6 · | -0.6 · | -1.1 · | -0.4 · |
| `xfer=segments` | +6.5 slower | -1.2 · | -1.5 · | +0.5 · | -0.9 · | -0.0 · | -0.4 · | -0.2 · | +0.1 · |

**linear** (delta% of B vs `factory=unicast`; `·` = noise; small shapes <2 MB/link with |delta|<10% marked inconcl.)

| knob | 131K | 512K | 2M | 5M | 25M | p64 | p2K | p4K | p8K |
|---|---|---|---|---|---|---|---|---|---|
| *A median us* | 32.8 | 50.9 | 122.1 | 174.9 | 744.2 | 246.9 | 252.5 | 277.3 | 362.9 |
| `entry=one_packet` | +0.3 · | -15.8 faster | -12.5 faster | +0.4 · | -0.1 · | +0.1 · | -0.1 · | -0.0 · | -0.2 · |
| `k=auto` | -2.0 · | -0.9 · | +0.1 · | +0.5 · | +0.0 · | +0.0 · | -0.1 · | -9.2 faster | -30.5 faster |
| `k=auto,xfer=segments,rotation=absolute_dir_stagger,entry=one_packet` | -0.2 · | -16.9 faster | -12.4 faster | -0.5 · | -0.6 · | -1.1 · | -0.9 · | -9.7 faster | -31.0 faster |
| `rotation=absolute` | +0.8 · | +2.9 · | -0.1 · | -0.7 · | -0.4 · | -0.8 · | -0.6 · | -0.2 · | -0.3 · |
| `rotation=absolute_dir_stagger` | +0.2 · | -0.6 · | +0.1 · | -0.6 · | -0.6 · | -1.0 · | -0.6 · | -0.3 · | -0.3 · |
| `xfer=segments` | +0.1 · | -0.1 · | -0.2 · | +0.2 · | -0.2 · | -0.1 · | -0.1 · | -0.1 · | -0.2 · |

**per-knob summary** (both topologies, 18 configs; geomean_sig over non-noise rows as analyzer; geomean_all over all 18)

| knob | faster | slower | noise | geomean_sig% | geomean_all% | faster/slower with small shapes <10% treated as noise |
|---|---|---|---|---|---|---|
| `entry=one_packet` | 4 | 0 | 14 | -11.1 | -2.4 | 3/0 |
| `k=auto` | 4 | 1 | 13 | -13.3 | -4.1 | 4/0 |
| `k=auto,xfer=segments,rotation=absolute_dir_stagger,entry=one_packet` | 8 | 0 | 10 | -15.2 | -7.3 | 7/0 |
| `rotation=absolute` | 0 | 0 | 18 | — | -0.1 | 0/0 |
| `rotation=absolute_dir_stagger` | 0 | 1 | 17 | +5.7 | -0.1 | 0/0 |
| `xfer=segments` | 0 | 1 | 17 | +6.5 | +0.1 | 0/0 |

## Multicast (R1b, reference `factory=multicast`)

**ring** (delta% of B vs `factory=multicast`; `·` = noise; small shapes <2 MB/link with |delta|<10% marked inconcl.)

| knob | 131K | 512K | 2M | 5M | 25M | p64 | p2K | p4K | p8K |
|---|---|---|---|---|---|---|---|---|---|
| *A median us* | 19.8 | 28.6 | 63.4 | 116.8 | 517.3 | 177.0 | 183.3 | 182.7 | 181.8 |
| `entry=one_packet` | -0.6 · | -3.6 · | -2.6 · | +1.8 · | -0.4 · | -0.4 · | -1.3 · | -0.7 · | -1.2 · |
| `k=auto` | +0.5 · | -0.0 · | -1.1 · | -0.3 · | -0.0 · | -0.3 · | -0.1 · | -0.1 · | -0.3 · |
| `k=auto,xfer=segments,rotation=absolute,entry=one_packet` | +0.8 · | -7.3 faster | -2.5 · | +1.8 · | -0.7 · | -0.4 · | -1.4 · | -1.1 · | -0.8 · |
| `rotation=absolute` | +0.6 · | -1.0 · | -1.5 · | +0.3 · | -0.4 · | -0.3 · | -1.0 · | -0.5 · | -0.7 · |
| `xfer=segments` | +1.0 · | +0.7 · | -1.3 · | +0.8 · | -0.1 · | -0.2 · | +0.1 · | -0.1 · | -0.4 · |

**linear** (delta% of B vs `factory=multicast`; `·` = noise; small shapes <2 MB/link with |delta|<10% marked inconcl.)

| knob | 131K | 512K | 2M | 5M | 25M | p64 | p2K | p4K | p8K |
|---|---|---|---|---|---|---|---|---|---|
| *A median us* | 25.0 | 42.3 | 108.6 | 205.3 | 780.3 | 274.1 | 274.1 | 287.3 | 346.9 |
| `entry=one_packet` | -4.5 · | -1.8 · | -2.4 · | +10.7 slower | +28.0 slower | +0.3 · | +26.4 slower | +20.5 slower | -0.3 · |
| `k=auto` | +0.4 · | +0.0 · | -0.2 · | -0.5 · | -0.2 · | +0.1 · | -0.1 · | -0.1 · | -0.3 · |
| `k=auto,xfer=segments,rotation=absolute,entry=one_packet` | -1.8 · | -1.8 · | -2.9 · | +10.5 slower | +27.8 slower | -0.2 · | +25.9 slower | +20.5 slower | -0.3 · |
| `rotation=absolute` | -1.1 · | +1.2 · | -0.3 · | -0.4 · | -0.2 · | +0.2 · | +0.1 · | +0.1 · | -0.3 · |
| `xfer=segments` | +1.3 · | +0.4 · | -0.3 · | -0.4 · | -0.1 · | +0.4 · | +0.1 · | +0.0 · | -0.2 · |

**per-knob summary** (both topologies, 18 configs; geomean_sig over non-noise rows as analyzer; geomean_all over all 18)

| knob | faster | slower | noise | geomean_sig% | geomean_all% | faster/slower with small shapes <10% treated as noise |
|---|---|---|---|---|---|---|
| `entry=one_packet` | 0 | 4 | 14 | +21.2 | +3.3 | 0/4 |
| `k=auto` | 0 | 0 | 18 | — | -0.2 | 0/0 |
| `k=auto,xfer=segments,rotation=absolute,entry=one_packet` | 1 | 4 | 13 | +14.7 | +3.2 | 0/4 |
| `rotation=absolute` | 0 | 0 | 18 | — | -0.3 | 0/0 |
| `xfer=segments` | 0 | 0 | 18 | — | +0.1 | 0/0 |

## Reading

**k=auto**
- Unicast: it moves only the RM rows with page > 2 KB, as predicted. p8K drops 30.5 % on linear
  (363 to 252 us) and 22.7 % on ring (185 to 143 us). p4K drops 9.2 % on linear and 6.0 % on ring. p64 and
  p2K and every tile shape are noise.
- The only other flag is 131K ring at +6.8 %. That is a 0.13 MB/link shape, under the 10 % bar, so it is
  inconclusive. It is probably not a k effect, because a tile page is 2 KB and should get k=1.
- Multicast: it changes nothing, with all 18 rows within ±1.1 %. Either multicast ignores k or auto resolves
  to 1 on that path. Worth a quick code check before round 2, but no perf reason to carry it.
- Verdict: keep for unicast. It is page-dependent and only acts on p4K and p8K. Irrelevant for multicast as
  it stands.

**entry=one_packet (= scatter_aware)**
- Unicast: it helps exactly the short-stripe tile shapes 512K (stripe 8) and 2M (stripe 2).
  - 512K: -15.8 % linear, -7.7 % ring. The ring number sits under the small-shape bar, but it agrees in
    direction with linear and with the combo row (-7.6 %).
  - 2M: -12.5 % linear, -8.0 % ring.
  - 131K (stripe 4, tiny) shows nothing: +0.3 % linear, +2.8 % ring.
  - The long-stripe shapes (5M stripe 20, 25M stripe 32) and all RM shapes are noise. So "helps short
    stripes" holds on unicast for stripe 2-8 at 0.5-2 MB/link.
- Multicast: it **hurts on linear, badly**: 5M +10.7 %, 25M +28.0 %, p2K +26.4 %, p4K +20.5 %. Ring is
  neutral everywhere (-3.6 % to +1.8 %, all noise), and p64 and p8K on linear are untouched.
  - Odd detail: linear p2K and p4K both land at ~346 us, the same as the multicast linear p8K baseline
    (346.9 us). That looks like the entry size clamping to one fixed transfer cadence on that path.
  - The short-stripe tile shapes on multicast are -1.8 % to -4.5 %, all noise.
- Verdict: for unicast, keep it, scoped to short stripes; it is neutral elsewhere, so making it the unicast
  default costs nothing measurable here. For multicast, drop it (or gate it off on linear).

**rotation (absolute / absolute_dir_stagger)**
- It does **not** help the large shapes.
  - 25M: -0.3 % / -0.6 % on ring and -0.4 % / -0.6 % on linear (unicast), and -0.4 % ring / -0.2 % linear
    (multicast absolute).
  - 5M: within ±1.1 %.
  - The only small consistent lean is -2.6 % to -3.0 % on unicast ring p64, which is below 5 %, so noise.
  - absolute_dir_stagger is flagged +5.7 % slower on 131K ring, which is inconclusive.
- So at 2 links with the current worker counts there is no worker-collision cost for rotation to recover, or
  rotation does not address it.
- Verdict: no effect; drop it as a default candidate on both factories.

**xfer=segments**
- Noise on every row for both factories: geomean over all rows is +0.1 % for each. The single flag is unicast
  131K ring +6.5 %, which is inconclusive.
- Verdict: no perf effect. Keep or drop it on code-simplicity grounds only.

**Combos**
- Unicast (k=auto + segments + dir_stagger + one_packet) is the best line: 7 rows faster, 0 slower,
  geomean_all -7.3 %. Its deltas are the sum of the k=auto rows (p4K, p8K) and the one_packet rows
  (512K, 2M). It adds nothing beyond those two knobs: p8K linear -31.0 % vs -30.5 %, 2M ring -9.5 % vs -8.0 %.
- Multicast (k=auto + segments + absolute + one_packet) inherits the one_packet linear regressions
  (+10.5 %, +27.8 %, +25.9 %, +20.5 %). Its one "faster" row (512K ring, -7.3 %) is a small shape and
  inconclusive.

**Per factory, in short**
- Unicast: keep k=auto (p4K/p8K) and one_packet (short-stripe tile). rotation and segments are neutral.
- Multicast: none of the five knobs helps. one_packet is harmful on linear above ~5 MB and on RM 2-4 KB
  pages. Leave multicast on its legacy policies.

## Side observation: unicast vs multicast baselines (cross-report, so ±a few % of drift)

Ratio unicast/multicast - 1 (negative means unicast is faster):

| shape | ring | linear |
|---|---|---|
| 131K | +2.5 % | +31.2 % |
| 512K | +5.9 % | +20.4 % |
| 2M | +3.0 % | +12.4 % |
| 5M | -19.9 % | -14.8 % |
| 25M | -26.3 % | -4.6 % |
| p64 | -17.9 % | -9.9 % |
| p2K | -22.1 % | -7.9 % |
| p4K | -16.5 % | -3.5 % |
| p8K | +1.8 % | +4.6 % |

With unicast k=auto, p8K becomes 143 us ring and 252 us linear, against multicast's 182 us and 347 us.
Multicast wins on linear for <= 2 MB/link. With one_packet on, unicast linear 2M is 106.8 us vs 108.6 us
for multicast. That is input for round 3's crossover, not a conclusion.
