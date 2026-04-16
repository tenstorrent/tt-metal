# BH Fast-Tilize vs Regular Tilize — Performance Comparison

Measured on Blackhole silicon. All numbers are **cycles per tile** from the
L1\_TO\_L1 profiler zone (bottleneck thread), with LOOP\_FACTOR=4.

## Float16\_b → Float16\_b

Regular tilize: `perf_unpack_tilize.py` (`sources/unpack_tilize_perf.cpp`)
Fast tilize: `perf_fast_tilize_full.py` (`sources/fast_tilize_bh_test.cpp`)

Format: **Regular / Fast**

| | ct=1 | ct=2 | ct=3 | ct=4 | ct=5 | ct=6 | ct=7 | ct=8 |
|---|---|---|---|---|---|---|---|---|
| **rt=1** | 185/- | 124/**62** | 107/**41** | 97/**32** | 92/**47** | 89/**39** | 87/**34** | 86/**25** |
| **rt=2** | 134/- | 100/**51** | 95/**34** | 90/**26** | 84/**44** | 80/**36** | 78/**32** | 78/**26** |
| **rt=3** | 116/- | 93/**46** | 90/**31** | 82/**26** | 79/**42** | 77/**35** | 75/**31** | 75/**26** |
| **rt=4** | 107/- | 91/**46** | 84/**31** | 81/**26** | 77/**41** | 75/**34** | 73/**31** | 73/**26** |
| **rt=5** | 103/- | 86/**46** | 83/**31** | 78/**26** | 76/**41** | 74/**34** | 73/**31** | 72/**26** |
| **rt=6** | 100/- | 83/**46** | 81/**31** | 78/**26** | 74/**40** | 73/**34** | 72/**31** | 71/**26** |
| **rt=7** | 99/- | 81/**46** | 79/**31** | 76/**26** | 74/**40** | 73/**33** | 72/**31** | 71/**26** |
| **rt=8** | 98/- | 81/**46** | 80/**31** | 76/**26** | 74/**40** | 72/**33** | 71/**31** | 71/**26** |

- ct=1: fast-tilize uses standard tilize fallback (no fast path)
- Steady-state (4-wide): **~26 cyc/tile** vs ~71–98 cyc/tile regular (**2.7–3.5x**)
- Non-4-wide widths (ct=2,3,5,6,7) have higher cost due to 2/3-wide tail units

## Float16\_b → Bfp8\_b

| | ct=2 | ct=4 | ct=8 |
|---|---|---|---|
| **rt=1** | 124/**72** | 97/**38** | 86/**36** |
| **rt=2** | 100/**61** | 90/**36** | 78/**36** |
| **rt=4** | 91/**55** | 81/**36** | 73/**37** |
| **rt=8** | 81/**52** | 76/**37** | 71/**37** |

BFP output adds ~10 cyc/tile overhead vs Float16\_b due to per-tile
PACK|THCON stall and L1 address update (BFP tiles need explicit closure).

## Float32 → Float16\_b

| | ct=2 | ct=4 | ct=8 |
|---|---|---|---|
| **rt=1** | 129/**62** | 101/**31** | 90/**33** |
| **rt=2** | 104/**51** | 94/**33** | 82/**37** |
| **rt=4** | 95/**48** | 85/**37** | 77/**38** |
| **rt=8** | 85/**48** | 80/**38** | 75/**39** |

## Summary

| Format | Regular (steady-state) | Fast (steady-state) | Speedup |
|--------|----------------------|-------------------|---------|
| Float16\_b → Float16\_b | ~71 cyc/tile | ~26 cyc/tile | **2.7x** |
| Float16\_b → Bfp8\_b | ~71 cyc/tile | ~36 cyc/tile | **2.0x** |
| Float32 → Float16\_b | ~75 cyc/tile | ~31–38 cyc/tile | **2.0–2.5x** |

### Bottleneck analysis

- **Regular tilize**: math is the bottleneck (A2D datacopy is the slowest stage)
- **Fast tilize Float16\_b**: pack is the bottleneck (~26 cyc/tile from MOP replay)
- **Fast tilize BFP**: pack is the bottleneck (~36 cyc/tile, includes per-tile stall)
- Unpack and math are well-hidden behind pack in the fast path

### Non-4-wide width overhead

Widths not divisible by 4 decompose as `{4, 2, 3}` units. The 2-wide and
3-wide tail units are less efficient than 4-wide because:

- Fewer tiles amortize MOP start/end overhead
- Stride-preserving approach keeps the same CH1\_Z stride (designed for 4-wide)

| Width | Decomposition | Approx cyc/tile |
|-------|--------------|----------------|
| 2 | 2 | ~46–51 |
| 3 | 3 | ~31 |
| 4 | 4 | ~26 |
| 5 | 2+3 | ~40–47 |
| 6 | 4+2 | ~33–39 |
| 7 | 4+3 | ~31–34 |
| 8 | 4+4 | ~25–26 |
