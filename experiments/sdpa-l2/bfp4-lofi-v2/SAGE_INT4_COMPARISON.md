# SageAttention2 INT4 versus native BFP4: QK-only model

Status: exploratory CPU evidence, September 15, 2026. **Unbiased BFP4 QK is in a comparable intrinsic normal-input error band to simplified Sage-style INT4, but neither establishes sub-0.5% operator L2.** This is not an executed SageAttention benchmark or a model-quality comparison.

## Contract and selected results

[Driver](sage_int4_qk_models.py); [all 72 records and source hashes](sage-int4-qk-models-v1.jsonl).
Q128, D128, distinct K/V lengths 4096 and 32768, seed 1243; original BF16 inputs/output. Normal Q/K have standard deviation 1; scaled-down Q/K have standard deviation 0.5. K is centered over keys; Q is centered within its 128-row block with an accurate correction. FP64 score arithmetic, softmax, and PV isolate QK representation loss. No PV quantization, GPU accumulation, or device timing is modeled. Four CPU threads; completed in 3.9 seconds.

L2 is 100×||output−reference||₂/||reference||₂. Tables were generated from JSON with `preprocessing=center_qk`; L2 values are displayed to three decimal places. The linked full-precision JSON remains authoritative.

|Q/K encoding|Normal4K L2%|Normal 32K L2%|Scaled-down4K L2%|Scaled-down32K L2%|
|---|---:|---:|---:|---:|
|Exact QK (BF16-output rounding only)|0.165|0.165|0.167|0.166|
|INT8 thread groups Q32/K64|1.068|1.078|0.304|0.298|
|Ideal per-row INT4|16.241|16.362|3.791|3.709|
|INT4 thread groups Q32/K64|19.843|19.482|4.643|4.524|
|Ideal per-block INT4 Q128/K64|22.335|22.789|5.373|5.251|
|Ideal per-block INT4 Q128/K512|23.760|24.355|5.682|5.609|
|Ideal arbitrary-scale INT4 per-16 channels|11.711|11.705|2.778|2.693|
|Native shared-16 BFP4 host-RNE|16.314|16.239|3.854|3.734|
|Native BFP4 biased device-pack model|33.539|33.579|7.596|7.423|

The JSON also includes K-centering-only controls, PCC, row-p95 L2, operand reconstruction error, and row-centered score-error RMS. Similar K-only results show that small-mean removal is not the main explanation on IID normal inputs.

## What is, and is not, SageAttention2

The latest inspected paper is **v7** (October 2025), not the older v3. Algorithm 1 now centers Q per block and restores its score correction. Its typical Q128/K64/four-warp configuration uses four complete Q rows per scale and 16 complete K rows per scale. P/V use E4M3, with per-block P and per-channel V scales; local PV accumulation is buffered into FP32. [Paper §§3.1–3.4 and A.6](https://arxiv.org/html/2411.10958v7)

The official quantizer kernels group Q rows `i,i+8,i+16,i+24` within 32 rows, and K rows `2i,2i+1,2i+8,2i+9,…` within 64 rows. They use FP32 `max(abs(x))/7 + 1e-7` and nearest-away-from-zero rounding. Our thread-group variants reproduce these patterns and scalar quantizer arithmetic, **not the complete CUDA attention algorithm**. Other INT4 rows are idealized RNE scale-granularity ablations. [Pinned quantizer code](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/triton/quant_per_thread.py)

Current public standard dispatch calls INT8 QK routines; the name SageAttention2 does not alone establish INT4. [Pinned dispatch](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/core.py)

The paper reports cosine similarity, relative L1 and RMSE, not this normalized L2. Table 2 shows Llama3.1 MMLU 0.635→0.607 for INT4 versus 0.634 for INT8; WikiText 6.013→6.256 for INT4. Its 262K H100 experiment uses INT8. Thus “nearly lossless SageAttention” must not be generalized to every INT4 configuration. [Paper Tables2/6 and A.9](https://arxiv.org/html/2411.10958v7)

## Interpretation

Native BFP4 shares a power-of-two exponent across 16 adjacent channels; the Sage-style groups above cover 512 Q values or 2048 K values at D128. Finer native grouping offsets some loss from restricted scale choice. At the **same 16-value grouping**, ideal arbitrary-scale INT4 is better: its normal 32K L2 is 11.705% versus 16.239% for native BFP4. That ideal grouping is not the published Sage kernel and does not establish free arbitrary inner-dimension scaling on Blackhole.

The old biased pack model approximately doubles QK-only error here. Unbiased rounding is therefore important before comparing formats. The BFP4 pack model is the existing validated ordinary-range [v1 quantizer](../bfp4-lofi-v1/probe.py); exceptional/subnormal values are outside this experiment.

Recommendation: describe the work as **SageAttention-inspired BFP4 attention with measured errors**. Do not promise Sage model quality, a 0.5% L2 threshold, or NVIDIA performance equivalence from these simulations. Model-derived QKV and end-to-end checks remain necessary. The four frozen SDPA options are unchanged.
