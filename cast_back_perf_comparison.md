# per_token_cast_back — device-time comparison: main vs branch

**Branch:** `nostojic/comp_decomp_perf_improvement` (working tree) vs `main`
**Op:** `PerTokenCastBackDeviceOperation`
**Metric:** `DEVICE FW DURATION [ns]`, 110 cores, Blackhole
**Method:** Tracy (`python -m tracy -p -r`), 3 runs per state, program-cache warm, **medians** reported
**Test:** `tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_deepseek_prefill_per_token_cast.py -k test_cast_back_dequant`

## Results (all 12 SHAPE × dtype combinations)

| Shape | Out dtype | main (median) | branch (median) | Δ |
|---|---|---:|---:|---:|
| (1, 1024) | bf16 | 4,668 | 3,308 | **−29.1%** |
| (1, 1024) | fp32 | 4,691 | 3,313 | **−29.4%** |
| (30, 1152) | bf16 | 5,072 | 3,716 | **−26.7%** |
| (30, 1152) | fp32 | 5,421 | 4,059 | **−25.1%** |
| (2, 3, 30, 1152) | bf16 | 7,678 | 6,156 | **−19.8%** |
| (2, 3, 30, 1152) | fp32 | 10,126 | 8,752 | **−13.6%** |
| (640, 7168) | bf16 | 74,200 | 66,222 | **−10.8%** |
| (640, 7168) | fp32 | 132,332 | 123,884 | **−6.4%** |
| (3200, 7168) | bf16 | 343,253 | 320,763 | **−6.6%** |
| (3200, 7168) | fp32 | 635,552 | 593,167 | **−6.7%** |
| (6400, 7168) | bf16 | 738,843 | 635,475 | **−14.0%** |
| (6400, 7168) | fp32 | 1,251,805 | 1,184,428 | **−5.4%** |

Times in ns. Branch is faster in all 12 cases; no regressions. Correctness held — 12/12 passed on every run.

## Interpretation

- **Small shapes (−25 to −29%):** tiny ops dominated by pipeline startup / fixed overhead; CB double-buffering hides most of it.
- **Large shapes (−5 to −7% typical):** DRAM/NoC-bound regime; steady-state win is smaller but consistent.
- **(6400, 7168) bf16 (−14%)** is the least trustworthy cell: baseline runs were jittery (671k / 770k / 739k, ~14% spread) while branch runs were tight (635k / 625k / 640k). Direction solid; treat that exact % as ±several points.

## Caveats

1. **This is not "the fusion."** The `pack_untilize_dest` fusion from PR #48553 is **not** in this build — the compute-helper step un-fused the untilize (`compute_kernel_lib::untilize` reads from a tiled CB via `pack_untilize_block`, reintroducing the `cb_out_tile` round-trip). This table measures the branch's *other* changes: **CB double-buffering** (`2×` on `cb_in_rm` / `cb_in_tile` / `cb_out_tile`) + **`UnpackToDestFp32`** on the e4m3 decode + the helper restructure. The double-buffering is almost certainly the dominant contributor (reader/compute/writer pipeline overlap).
2. Single-device, program-cache-warm, medians of 3. Baseline large-shape runs show more run-to-run jitter than the branch.

## Raw runs (main / branch), ns

```
(1,1024)      bf16 [4660,4668,4744]/[3308,3302,3322]        fp32 [4670,4738,4691]/[3313,3304,3313]
(30,1152)     bf16 [5071,5087,5072]/[3716,3681,3756]        fp32 [5421,5421,5479]/[4004,4067,4059]
(2,3,30,1152) bf16 [7678,7450,7705]/[6156,6190,6150]        fp32 [10126,9936,10144]/[8752,8576,8822]
(640,7168)    bf16 [74879,74200,73310]/[66275,64823,66222]  fp32 [132332,132688,129804]/[123695,123884,123890]
(3200,7168)   bf16 [336706,343253,346009]/[320763,319284,329123]  fp32 [635552,623348,642973]/[593167,592728,595278]
(6400,7168)   bf16 [671112,769665,738843]/[635475,624761,640420]  fp32 [1237855,1253347,1251805]/[1179470,1184428,1197819]
```
