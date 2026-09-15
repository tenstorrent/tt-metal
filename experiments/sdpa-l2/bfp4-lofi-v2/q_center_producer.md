# Q-block centering producer (isolated, not yet device qualified)

For a single Q128 block let `b = BF16(device_mean(Q))`. Produce
`Qc = RNE7(FP32(Q)-FP32(b))` and `C = HiFi4_FP32(b @ original_K.T)`.
The eventual unscaled score is `LoFi(Qc @ quantized_K.T) + C`.
The same actual BF16 `b` is used on both branches, so its rounding does not
itself break the exact decomposition `(Q-b)K^T + bK^T = QK^T`. Rounding Qc,
quantizing K, finite subtraction, and correction matmul still contribute error.
Do not compute C against quantized K: common-Q times K-quantization error is
precisely the term this scheme should avoid. Original Q/K are never overwritten.

## Producer and contract

`q_center_producer.build(device, q, k, ncores=4, mean_mode='bf16_fpu')`
returns `(centered_q, correction, invoke, info)`. Q is BF16 `[1,H,128,128]`;
K is original BF16 `[1,H,N,128]`, with N a positive multiple of 32. Correction
is unscaled FP32 `[1,H,32,N]` with 32 repeated rows; it is O(HN), not O(HN²).
The device mean, fused center/RNE7 kernel, and correction are recomputed by
each invoke. No host-computed bias or correction is uploaded.

The new center kernel uses BF16 DST: four input tiles plus four bias tiles.
It loads matching values into SFPU, subtracts in FP32, rounds the live bits
to seven significant bits, and only then stores BF16. This avoids a centered
BF16 spill before RNE7. Copy zero-source state is explicitly cleared after
copy initialization, preserving the previously qualified tiny-value behavior.
All input/bias values and centered results must be finite normal or zero;
subnormals and rounding overflow are outside the prototype contract.

The default mean is existing `center_mean.py` BF16/FPU + HiFi4/FP32 DST.
Its BF16 reciprocal is exact for Q128 (`1/128`), unlike non-power-of-two
sequence lengths. The alternative FP32-SFPU mean includes the source cast
and final BF16 bias conversion in timing. Both use the same rounded bias
for correction and centering. No need to compute the mean exactly for the
algebra to hold; a better mean merely makes the centered values smaller.

The correction uses the supported `ttnn.matmul(..., transpose_b=True,
dtype=ttnn.float32, compute_kernel_config=HiFi4/FP32,
optional_output_tensor=correction)` interface. See
`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:825` and
`ttnn/cpp/ttnn/operations/matmul/matmul.cpp:268`: auto-selected programs
may require a materialized transpose; any such work is inside the timing.

## Integration recommendation: correctness before seeding optimization

The private streaming QK helper is `p8_streaming/compute_streaming.hpp:466`.
It completes QK into FP32 DST and packs raw unscaled scores. Row maxima are
then reduced from those scores (`:614`); scaling is deferred to subtraction
and exp. Therefore add C **before row-max reduction**, not merely inside
sub-exp. An incorrect max changes online rescaling and can overflow exp even
if a later subtraction nominally includes C.

Safest first experiment is an explicit FP32-SFPU add of correction to score
CB6 after QK packing but before publication to the max reducer. Process two
score tiles and two correction tiles per FP32-DST acquire (four usable slots),
add using full-FP32 SFPU arithmetic, and repack scores. Correction tiles
repeat rows, so no FPU broadcast/ELWADD or BF16 truncation is required. Restore
Q/K unpack formats and width4 score pack state afterward. Gate the max-reducer
semaphore on **corrected** score readiness; existing half-row overlap signals
must not expose uncorrected scores. This costs another score read/write pass
and is deliberately a correctness-first implementation, not a speed claim.

Seeding DST with FP32 C before LoFi QK is potentially cheaper: load the four
correction tiles directly to FP32 DST, restore Q/K unpack and matmul state,
then accumulate the four reduction tiles. But first verify the exact LLK
initial-accumulation/zero-accumulator behavior; a first-MM clear or a seed
loaded through SrcA/SrcB narrowing silently loses the correction. Do not use
an FPU BF16 elementwise seed. Also large C may worsen low-bit preservation
when added before the QK sum, whereas post-sum SFPU addition gives a cleaner
numerical baseline. No shared header is changed in this producer task.

If K is centered later, use the same centered-K representation consistently
in QK and high-precision correction. Algebraically removing a common K vector
only adds a query-specific constant to all logits and leaves softmax invariant,
but quantization and cancellation require separate accounting.

## Honest cost and next optimization

Dense Tensix matmul still computes 32 physical rows for one unique mean row.
The HiFi4 correction therefore costs `4*32/128 = 1` times Q128 LoFi QK's
matmul phase work (about 50% extra versus equal-cost LoFi QK+PV), ignoring
utilization, preprocessing, and score-add traffic. The logical arithmetic is
32 times smaller than that physical matmul. Producer correction storage is
`128*H*N` bytes; each 512-key chunk is 64 KiB per head. A later custom
dot/reduce or compact correction layout could reduce physical redundancy,
but must be measured. Reuse C for every row of this Q block, not every head's
entire Q sequence; block means differ. Producer scratch can be reused block
by block rather than storing all Q-block corrections at once.

## Probe

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/q_center_producer.py \
  --label q-center-normal-smoke-v1 --length 1024 --heads 1 --iters 0
python experiments/sdpa-l2/bfp4-lofi-v2/q_center_producer.py \
  --label q-center-common-smoke-v1 --length 1024 --heads 3 --cores 4 \
  --distribution common_q --iters 0
```

Use `--iters 5 --trace-repeats 10` for individual mean, center, correction,
and total producer timings. The probe requires exact centered-RNE7 output,
repeated correction rows, and correction relative L2 below 0.01% against
FP64 multiplication of the actual BF16 bias and original BF16 K. It records
mean error separately. Its final attention diagnostic runs **host exact QK**
on device-centered Q plus device correction, compared to original Q/K/V; this
is not a LoFi attention result or evidence that integration is correct.
