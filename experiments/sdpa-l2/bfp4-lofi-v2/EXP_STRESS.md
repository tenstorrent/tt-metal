# Native versus LUT exp: 1K, 32K and 256K stress comparisons

All three plans and their separate-process repetitions are complete: **304 result records, 152 exp-pair checks, and 152 process-repeat checks pass**. The [small plan](exp-stress-plan.json) contributes 80 configurations / 160 records, the [32K plan](exp-stress-long-plan.json) 48 / 96, and the [256K plan](exp-stress-256k-plan.json) 24 / 48. Every complete-output hash and every recorded result field other than the fresh label match across process repetitions. All 304 runtime manifests match current source without principal omissions or drift. Independent fresh audits exactly match the parsed [small audit](exp-stress-small-audit-v1.json), [32K audit](exp-stress-long-audit-v1.json), and [256K audit](exp-stress-256k-audit-v1.json). The tables are direct measurements, not cross-length extrapolations. Use [the read-only audit](EXP_STRESS_AUDIT.md) for live status. This study remains separate from the earlier 85-output post-format qualification; its LoFi evidence limitations remain explicit below. Numerical producers and their kernel configurations are unchanged.

## Measurement contract

All tables use square noncausal H2/D128, Q256/K512, FP32 destination/numerator/denominator, one existing K/V input slot, BF16 output, and seeds 1240/1241. The N1024 tables use four cores and reference every query row; N32768 and N262144 use 22 cores and 128 explicit sampled Q rows per head. All references use FP64 attention on the **original BF16 Q/K/V**, including all KV. Finiteness and output hashes cover every device output element, but the 32K/256K numerical metrics are not all-query bounds. Each cell is the minimum–maximum across the two seeds, not a confidence interval. There are no timing iterations or new performance claims.

LoFi uses Q7 and per-value RNE5 K/V stored as BFP8. HiFi2 uses Q7 with BF16 K/V preserved bit-for-bit. Within each table native/LUT changes exp refinement, not inputs, chunk sizes, state formats or the attention matmul fidelity. Comparing between tables changes both multiplication fidelity and K/V representation, so it is not an exp-only ablation.

Inputs come from the pinned `repro_sdpa_l2.py` generator. `outliers` adds sparse (0.1%) Gaussian perturbations scaled by 10; `scaled_qk` multiplies Q and K by 2; common-Q/K/V adds 32 to that operand before BF16 conversion. `biased_v` adds 1. `uniform` means Q=0 with random K/V, **not** uniformly distributed inputs. Constant-V uses V=1, and uniform-constant-V also sets Q=0.

## N1024: LoFi FP32, BFP8 K/V

| Input | Native L2 % | Native PCC | LUT L2 % | LUT PCC |
|---|---:|---:|---:|---:|
| normal | 2.904–2.941 | 0.999572–0.999581 | 2.200–2.251 | 0.999747–0.999758 |
| outliers | 3.656–5.013 | 0.998857–0.999351 | 3.449–4.394 | 0.999062–0.999406 |
| scaled QK | 4.004–4.101 | 0.999175–0.999215 | 3.793–3.939 | 0.999224–0.999280 |
| common Q | 7.855–9.313 | 0.995837–0.996872 | 7.843–9.322 | 0.995829–0.996881 |
| common K | 45.045–45.760 | 0.900652–0.903108 | 45.240–45.962 | 0.900167–0.902644 |
| common V | 0.343–0.348 | 0.731781–0.742925 | 0.343–0.348 | 0.732617–0.743428 |
| constant V | 0.000–0.000 | undefined | 0.000–0.000 | undefined |
| uniform attention | 1.499–1.661 | 0.999862–0.999888 | 1.496–1.662 | 0.999862–0.999889 |
| uniform attention, constant V | 0.000–0.000 | undefined | 0.000–0.000 | undefined |
| biased V | 0.243–0.244 | 0.998853–0.998885 | 0.223–0.223 | 0.999033–0.999064 |

## N1024: HiFi2 FP32, BF16 K/V

| Input | Native L2 % | Native PCC | LUT L2 % | LUT PCC |
|---|---:|---:|---:|---:|
| normal | 2.048–2.065 | 0.999792–0.999795 | 0.728–0.733 | 0.999973–0.999974 |
| outliers | 1.737–1.854 | 0.999864–0.999884 | 0.879–0.977 | 0.999953–0.999962 |
| scaled QK | 1.634–1.651 | 0.999891–0.999891 | 0.995–1.002 | 0.999950–0.999951 |
| common Q | 1.208–3.645 | 0.999327–0.999927 | 1.186–3.649 | 0.999325–0.999929 |
| common K | 2.245–2.258 | 0.999752–0.999754 | 1.239–1.247 | 0.999923–0.999924 |
| common V | 0.134–0.136 | 0.631063–0.637114 | 0.134–0.136 | 0.632540–0.638607 |
| constant V | 0.000–0.000 | undefined | 0.000–0.000 | undefined |
| uniform attention | 0.162–0.179 | 0.999998–0.999999 | 0.162–0.180 | 0.999998–0.999999 |
| uniform attention, constant V | 0.000–0.000 | undefined | 0.000–0.000 | undefined |
| biased V | 0.207–0.208 | 0.999158–0.999195 | 0.182–0.183 | 0.999345–0.999377 |

Evidence filenames are `exp-stress-{lofi_native,lofi_lut,hi2_native,hi2_lut}-{distribution}-s{1240,1241}-v1.json`. The explicit plan lists every command and path. Metrics in the records retain more precision than these tables; displayed zero is rounded, not a general exactness assertion. PCC is undefined for the constant-reference cases and is not treated as zero or failure.

## N32768: LoFi FP32, BFP8 K/V

First-process results shown; every result field other than the label is identical in the completed independent-process repeats.

| Input | Native L2 % | Native PCC | LUT L2 % | LUT PCC |
|---|---:|---:|---:|---:|
| normal | 2.820–2.835 | 0.999598–0.999602 | 2.198–2.212 | 0.999755–0.999758 |
| outliers | 7.002–7.491 | 0.997199–0.997548 | 6.879–7.521 | 0.997214–0.997639 |
| scaled QK | 4.531–4.647 | 0.998948–0.998993 | 4.384–4.459 | 0.999006–0.999040 |
| common Q | 1.542–4.552 | 0.998986–0.999882 | 1.542–4.545 | 0.998989–0.999882 |
| common K | 45.849–46.039 | 0.899391–0.899972 | 45.894–46.080 | 0.899316–0.899907 |
| common V | 0.388–0.389 | 0.021296–0.038507 | 0.388–0.389 | 0.031421–0.038507 |

## N32768: HiFi2 FP32, BF16 K/V

| Input | Native L2 % | Native PCC | LUT L2 % | LUT PCC |
|---|---:|---:|---:|---:|
| normal | 1.914–1.928 | 0.999814–0.999817 | 0.748–0.752 | 0.999972–0.999972 |
| outliers | 0.980–1.780 | 0.999874–0.999954 | 0.719–1.114 | 0.999942–0.999974 |
| scaled QK | 1.559–1.703 | 0.999898–0.999909 | 1.006–1.014 | 0.999949–0.999950 |
| common Q | 0.038–3.588 | 0.999356–1.000000 | 0.038–3.541 | 0.999373–1.000000 |
| common K | 2.175–2.204 | 0.999758–0.999765 | 1.274–1.320 | 0.999914–0.999920 |
| common V | 0.029–0.029 | undefined | 0.029–0.029 | undefined |

Long evidence filenames are `exp-stress-long-{lofi_native,lofi_lut,hi2_native,hi2_lut}-{distribution}-s{1240,1241}-v1.json`. Six distributions were deliberately selected; constant-V, uniform-attention and biased-V were not run at 32K in this plan. An undefined common-V PCC is not perfect correlation: the metric is undefined when the reference has negligible centered variation or the sampled output has none. The hardware records alone do not give a centered-residual bound or computed BF16 floor; the separate CPU-derived comparison below supplies that additional scale diagnostic.

## N262144: LoFi FP32, BFP8 K/V

First-process results shown; every result field other than the label is identical in the completed independent-process repeats.

| Input | Native L2 % | Native PCC | LUT L2 % | LUT PCC |
|---|---:|---:|---:|---:|
| normal | 2.774–2.803 | 0.999607–0.999615 | 2.200–2.219 | 0.999754–0.999758 |
| outliers | 10.349–15.660 | 0.987754–0.994641 | 10.378–15.948 | 0.987336–0.994630 |
| common K | 44.824–45.438 | 0.902410–0.904129 | 44.832–45.458 | 0.902345–0.904139 |

## N262144: HiFi2 FP32, BF16 K/V

| Input | Native L2 % | Native PCC | LUT L2 % | LUT PCC |
|---|---:|---:|---:|---:|
| normal | 1.864–1.870 | 0.999825–0.999826 | 0.750–0.752 | 0.999972–0.999972 |
| outliers | 1.137–1.617 | 0.999872–0.999939 | 0.785–1.371 | 0.999906–0.999969 |
| common K | 2.098–2.098 | 0.999781–0.999781 | 1.232–1.276 | 0.999920–0.999925 |

Evidence filenames are `exp-stress-256k-{lofi_native,lofi_lut,hi2_native,hi2_lut}-{distribution}-s{1240,1241}-v1.json`. Only normal, outliers and common-K were tested at 256K in this plan. The normal error bands remain close to their 32K values; this is not true of every distribution.

The LoFi outlier result is not merely an unstable maximum-element relative error. With LUT, seed 1240 has **15.948% aggregate L2**, sampled-row median/p95 L2 of **6.857%/17.990%**, and gain **0.990924**. Seed 1241 has 10.378% aggregate, 7.757%/18.544% row median/p95, and gain 0.996966. Thus substantial row errors persist even with gain near one. HiFi2/BF16 LUT row medians are 1.958–2.019% and p95 values 4.343–4.362% in these two runs. These statistics cover the 256 sampled query rows (128 per head), not every 256K query; row-relative and aggregate L2 use different normalizations and need not have the same ordering.

## Interpretation and limitations

LUT refinement improves normal-input accuracy clearly in both paths and helps the small-suite outlier/scaled-QK cases. It does **not** establish a universal accuracy band: common-Q changes little, LoFi common-K remains about 45–46% and slightly worsens, while HiFi2/BF16 common-K improves but remains around 1.23–1.32%. The 32K LoFi outlier error is about 7%, and LUT slightly worsens one seed rather than repairing both; at 256K LoFi outlier aggregate L2 reaches 10–16% and LUT worsens both seeds slightly. Common-Q is strongly seed-dependent. The unchanged-input exp pair isolates that refinement's effect, not the complete error mechanism. Cross-length inputs and numerical reference sampling differ, so these ranges are not a controlled monotonic-accumulation experiment. Differences between LoFi and HiFi2 tables cannot be uniquely attributed to fidelity because K/V storage and operand preparation also change.

Common-V aggregate L2 is particularly misleading: the large coherent value dominates the reference norm. Small-suite PCC is only roughly 0.63–0.74; at 32K LoFi PCC is roughly 0.02–0.04 and HiFi2 PCC is undefined despite its 0.029% aggregate L2. The separately audited [common-V derived-metrics study](COMMON_V_DERIVED_METRICS.md) now puts these errors on a BF16-output-floor scale: at 32K the recorded HiFi2 native/LUT **error norms equal the regenerated nearest-BF16-reference error norm**, while LoFi is 13.41–13.56 times that floor. This is equality of norms over 128 sampled Q rows per head, not an actual-output/floor tensor-hash match or new full-output device check.

Subtracting the same original-V mean from output and reference preserves the error numerator but exposes the small residual scale. At 32K the BF16-floor centered L2 is already 125.69–125.96%; LoFi's derived centered L2 is 1685–1708%. At 1K, HiFi2 is within about 0.035% of the nearest-output error norm, while LoFi is 2.52–2.60 times it despite having higher PCC. Thus PCC alone can reverse the useful comparison, and a residual error above 100% is not necessarily avoidable with fixed BF16 output. The CPU derivation retains the original error, does not derive centered PCC or identify the cause of excess LoFi error, and does not replace missing LoFi input-hash evidence. See also [value smoothing](VALUE_SMOOTHING.md). Uniform attention suppresses logit-selection effects; its nearly unchanged native/LUT error shows that exp refinement is not the main remaining error there.

HiFi2 records explicitly check original/prepared hashes, CPU/device input immutability, finite output and two bitwise correctness trace replays. LoFi records exact value-preparation counts and output hashes, with finite/source-stability assertions in matching-current producer code, but `iters=0` executes no trace and records no original/prepared hashes or input-immutability gate. Its paired-input contract therefore uses the same pinned seeded generator and exact value oracles, not measured pairwise input-bit hashes. The completed separate-process repetitions establish full-output reproducibility for the tested runs; they do not fill these omitted fields or become trace replays.

The `max_l2=1e9` command option is an abort-safety limit, not acceptance. No real model activations, model scores, timing, untested distributions, or numerical bound beyond the recorded scope is inferred. The CPU exp model and block-order intervention remain separate attribution studies, not extra rows or hidden numerical controls in these tables.
