# Residual K/V error budget: two/three BFP4 versus 4+8

Status: exploratory CPU evidence, September 15, 2026. **Three RNE BFP4 components improve K/V reconstruction, but Q/P7 already dominate normal-input output error. The extra matmul is unlikely to be a good trade against unbiased 4+8.**

## Contract and selected results

[Driver](residual_rne_models.py); [regular 4K/32K sweep](residual-rne-models-regular-v1.jsonl), [selected 256K cases](residual-rne-models-long-v1.jsonl), [centered cases](residual-rne-models-centered-v1.jsonl). Provenance and imported-source hashes are embedded.

Q128/K512/D128, noncausal distinct keys, seed 1242, original BF16 inputs and BF16 output. Q/P use per-value RNE7. K/V components are decomposed against what LoFi consumes, using the existing native shared-16 host-RNE BFP4 model. The 4+8 residual is RNE5 stored in BFP8. Subsequent math is FP64: these are **representation/operand-bit simulations, not device accuracy or throughput**. No new device jobs were run. Three four-thread CPU sweeps took 53.7, 16.2, and 127.3 seconds.

L2 is 100×||output−reference||₂/||reference||₂. Tables were generated from JSON; L2 values are displayed to three decimal places. The linked full-precision JSON remains authoritative. All use the **original, unquantized-P denominator**, no preprocessing unless stated.

### 32K

|Input|2×BFP4 L2%|3×BFP4 L2%|4+8 L2%|Q/P7 with exact K/V L2%|
|---|---:|---:|---:|---:|
|normal|1.543|0.578|0.591|0.565|
|outliers|2.295|0.650|0.728|0.656|
|scaled_qk|2.981|1.016|0.992|1.000|
|scaled_down|1.069|0.402|0.412|0.396|
|common_q|3.745|2.934|3.617|3.016|
|common_k|14.335|0.569|0.578|0.567|
|common_v|0.028|0.028|0.028|0.028|

The common-K two-component entry corrects an earlier conversational transcription. Common-V outputs are constant BF16 at 32K, making PCC undefined: tiny aggregate L2 does not demonstrate preservation of the small attention-dependent signal.

### 256K selected checks

|Input|3×BFP4 L2%|4+8 L2%|Q/P7 with exact K/V L2%|
|---|---:|---:|---:|
|normal|0.601|0.613|0.593|
|outliers|0.348|0.377|0.307|
|scaled_qk|1.120|1.099|1.083|

For 256K outliers, row-p95 L2 is 4.111% for 444 and 4.212% for 48, despite low aggregate L2. Scaled QK means standard deviation 2 in Q/K; scaled-down means 0.5. The 3×BFP4 output is not uniformly better than 4+8 because errors can cancel.

## Representation and theoretical budget

Normal 32K K/V reconstruction errors, before attention:

|Variant|K L2%|V L2%|
|---|---:|---:|
|rne44_qp7|0.993|0.993|
|rne444_qp7|0.080|0.079|
|rne48_qp7|0.118|0.119|

The Q/P7-only ablation already produces 0.565% normal 32K output L2. Assuming roughly independent errors, an illustrative root-sum-square budget gives:

- 4+8: sqrt(0.5654²+2×0.1185²) ≈ 0.590%.
- 444: sqrt(0.5654²+2×0.0795²) ≈ 0.576%.

Those estimates closely match the measured 0.591% and 0.578%. This is a heuristic decomposition, **not an error theorem**; “exact K/V floor” is an ablation, not a strict lower bound.

For ordinary-range native groups, the BFP4 magnitude step is Δ=2^(e−2), where e is the group's maximum exponent. Saturation can make error approach Δ rather than Δ/2. Re-encoding residuals shrinks their scale, explaining the large reconstruction improvement. The model does not establish exceptional-value or subnormal behavior. See the [quantizer](../bfp4-lofi-v1/probe.py) and [operand-bit model](numerics.py).

## Centering and represented-V mean matching

The following 32K results use `center_qkv_vmatch`: center Q/K/V, restore the accurate Q correction, and match the **represented** V mean after quantization.

|Input|3×BFP4 L2%|4+8 L2%|Q/P7 with exact K/V L2%|
|---|---:|---:|---:|
|normal|0.524|0.542|0.514|
|common_q|0.206|0.245|0.193|
|common_k|0.525|0.541|0.511|

Q means here cover the 128 tested queries; do not extrapolate normal-input gains to a mean over an entire long query sequence. Centering Q must restore its key-dependent correction. Latest Sage2 v7 Algorithm 1 uses per-Q-block centering, unlike the global-Q description in older revisions. [Sage2 v7](https://arxiv.org/html/2411.10958v7)

## Costs and recommendation

|K/V scheme|LoFi products per QK and per PV|Bytes per K/V tile|
|---|---:|---:|
|2×BFP4|2|1152|
|4+8|2|1664|
|3×BFP4|3|1728|

Three components require 50% more **bulk matmul work** than 4+8, not a measured 50% kernel slowdown; K/V bytes rise 3.85%. Preprocessing adds dependent residual formation, shared-group reductions, and packing. These costs are not measured in this CPU model.

Prioritize unbiased 4+8. If paying for a third product, a more targeted hypothesis is QhKh+QhKl+QlKh and PhVh+PhVl+PlVh, dropping second-order residual×residual terms. That would spend work on the dominant Q/P rounding error; it was **not tested by this driver**. The four frozen SDPA options remain unchanged.
