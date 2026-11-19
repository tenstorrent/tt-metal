# BFP8 column scaling: CPU attribution and candidate

Power-of-two output-column equalization is **not a uniform precision improvement** on these actual attention weights. Nearest-power-of-two RMS scaling helps V and some low-amplitude columns but worsens aggregate Q/K and isolated A/B quantization. Continuous RMS scaling gives a modest, broader improvement and is a possible controlled device candidate; no sequence-pass or latency improvement is claimed.

The seven-mode CPU sweep uses pinned layer0 real QKV/Z/B/A weights, their existing packed order, and exact captured input RMSNorm outputs at step19/user20,58/user30,99/user27. Both HF and TT input-norm outputs are exercised. Results: `column_scaling_cpu.json`. Reproducer: `candidates/column_scaling_probe.py`. Setup helper: `candidates/scaled_projection_setup.py`. Main decoder source is unchanged.

## Numeric results

Each candidate uploads mathematically scaled weights `W'[k,n]=W[k,n]*s[n]` in BFP8 and restores `y[n]=y'[n]/s[n]`. The CPU probe explicitly rounds the intermediate matmul output to BF16, then divides and rounds to BF16 again. Thus continuous-scaling rows include their additional output rounding boundary rather than assuming ideal unscaling. Scaling is derived from weights only, not fitted to these three activations.

The table reports isolated quantization RMSE on the three captured **TT** input-norm tensors, versus unquantized BF16 checkpoint weights on those same inputs. These are CPU FP32 matmuls, not device HiFi2.

| Scaling | Weight relative RMS error | Q RMSE | K RMSE | V RMSE | Z RMSE | B RMSE | A RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | .007775 | .008066 | .009113 | .008646 | .008711 | .005280 | .010208 |
| Power2 RMS nearest | .007708 | .008493 | .009149 | .008295 | .008714 | .005656 | .010539 |
| Power2 RMS floor | .008470 | .008661 | .009986 | .008896 | .009853 | .005705 | .011112 |
| Power2 max nearest | .008230 | .008803 | .009922 | .008940 | .009316 | .006063 | .010539 |
| Power2 max floor | .009196 | .009181 | .010613 | .009436 | .010805 | .005903 | .012137 |
| Continuous RMS | .007242 | .007795 | .008797 | .008088 | .008434 | .004687 | .010502 |
| Continuous max | .007585 | .008224 | .009190 | .008129 | .008831 | .005979 | .010142 |

Continuous RMS reduces total weight error6.86%, V projection RMSE6.45%, B11.24%, Q/K/Z approximately3–3.5%. A worsens2.88% for isolated quantization. Continuous RMS output errors versus actual captured HF projections, including the already-different TT norm input, improve Q/K/V/Z/B/A by approximately2.4%/2.8%/5.5%/2.7%/15.1%/1.8%. These are modest changes; three selected tokens cannot establish a long-sequence repair.

Nearest-power2 RMS specifically improves the lowest-RMS quarter of output columns: Q relative quantization error .010397→.008986, V .009713→.008666, B .009083→.007343. But improving small columns changes which channel sets the shared exponent and may coarsen larger columns; aggregate errors can increase. Power-of-two scaling's exactness does not guarantee a better block quantizer.

The unquantized CPU reference matches actual HF projected values at PCC .9999999992, relative RMS .0000405. The unscaled emulated BFP8 projection differs from captured TT projection by relative RMS roughly .0047–.0050, because this probe does not emulate device matrix fidelity/accumulation. Candidate performance and final PCC require a real device run.

## Packing contract and quantizer

The existing setup correctly transposes checkpoint `[N,K]` to TT matrix `[K,N]` **before** BFP8 packing. Standard32x32 tiles have16x16 faces. `blockfloat_common.cpp::pack_as_bfp_tiles` and `bfp_simd.hpp::row_scalar` share one exponent across each **16-value face row**, therefore across16 adjacent output columns at a fixed K, not across16 K values and not across a whole32x32 tile. The independent C++ quantizer in the probe reproduces the scalar path's maximum exponent, mantissa alignment, ties-to-even rounding and saturation. It is source-derived quantization evidence, not a linked-runtime/device equivalence claim.

All packed group boundaries are multiples of16 (and the explicit B/A pad16 keeps their matrix layout aligned). QKV/Z/B/A packing does not mix a group with unrelated columns inside an exponent block. Packing them separately with the same column order would produce the same BFP8 grouping for every live channel. Reinterpreting tilized storage or quantizing `[N,K]` before transpose changes the quantizer axis and is not a free precision fix for the existing matmul ABI.

A uniform power-of-two scale applied to all16 columns in a block cannot improve their quantization: it shifts the common exponent and is undone exactly, absent overflow/underflow. Only relative scales within a block change information allocation. Power-of-two per-column unscale is exactly representable in BF16 in the normal range, so it adds no new rounding beyond the matmul's BF16 pack. Continuous scales require a real FP32 inverse-scale multiply and final BF16 pack, both modeled here.

A tiny attempted TTNN host-only32x32 pack/unpack requested no device argument and returned `StorageType.HOST`. However, this checkout's packer queries `MetalContext::hal()` and initialized UMD topology/drivers, briefly waiting for the caller's device lock. The child reported this immediately and made **no further TTNN calls**. No mesh or device tensor was created and no compute kernel was requested. The complete weight sweep uses only Torch and an independent C++ CPU helper, avoiding this hidden driver initialization.

## Candidate integration and rejection boundaries

`scaled_projection_setup.py` returns scaled FP32 host weights for the normal BFP8 upload and an inverse-scale vector. It preserves BFP8 as the weight dtype. The runtime candidate would apply **one** device broadcast multiply after the packed projection and before QKV/Z/B/A slicing, using the inverse scale and BF16 output. Keep physical-N padding scales at1 and preserve projection output memory config. The extra operation's legality, sharding and cost remain unverified; do not introduce a host runtime path.

Always restore scale before convolution/history publication, beta/g nonlinearities, or Z gating. Folding arbitrary inverse scales into nonlinearities is invalid. Folding QKV scales into conv taps while retaining scaled history would alter the visible cache representation and the fused-state comparison; this helper deliberately avoids that contract change.

An evidence-backed narrow candidate is continuous RMS for QKV/Z/B with A left unscaled. A more conservative power2 candidate can isolate V/B, but QKV's shared packed projection would need an appropriately concatenated inverse scale and separate Q/K/V setup statistics. Neither candidate is selected as a final policy here. The broad power2 max and power2 RMS-floor policies have measured worse aggregate quantization and should not be prioritized without a different objective/evidence.

Do not add an empirical output correction or tune scales to the failed token's desired output. Scale/unscale is a mathematical reparameterization; all resulting accuracy changes must come from the explicitly measured quantization. Maintain the original .995 checks and stage8 precision-sweep decision. The next device candidate must show both sequential correctness and traced latency before adoption.

CPU reproduction:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/column_scaling_probe.py
```

The probe and setup helper compile. Seven complete real-weight CPU candidate rows are retained; no device latency or quality-gate pass is claimed.
