# V-axis BFP4: standalone Blackhole experiment

Static audit: feasible with full 32×32 tiles and the existing no-MOP LoFi matmul. No device result or performance claim yet. The four locked variants remain unchanged.

## Mapping

For original BF16 `V[1,H,N,128]`, make an exact device BF16 transpose `VT[1,H,128,N]`, then run the existing native-group RNE BFP4 preprocessor. Shared exponents now span 16 consecutive tokens at one value channel, rather than 16 value channels at one token. The quantized mathematical operand is `decode(quantize(VT)).transpose(-2,-1)`; the acceptance reference remains attention on the **original BF16 Q/K/V**.

Let `Nt=N/32`, chunk `c`, local token tile `k∈[0,16)`, channel tile `d∈[0,4)`. Gather global VT page `head*4*Nt + d*Nt + c*16 + k` into V CB tile `k*4+d`. This deliberately preserves the original **N-major CB tile grid**, while each individual tile contains transposed values. A D-major CB would be wrong with the unchanged PV block loop: the LLK advances the second operand by one page across output-column tiles, regardless of its transpose flag.

Q256/K512/D128, two independent 64-tile K slots and two 64-tile V slots, chain forwarding, barriers, PV subblocks and all CB data formats remain unchanged. V DRAM reads become four contiguous 16-page runs, scattered into the existing CB grid, rather than one contiguous 64-page run. That access-order change and the real transpose/quantization preprocessing must be measured; equal byte counts do not establish equal DM performance.

## Compute change and evidence

- [`matmul_custom.h`](../../../tt_metal/hw/inc/api/compute/experimental/matmul_custom.h): `mm_no_mop_init_short` / `mm_no_mop_reinit_short` pass transpose to **both** unpack and math. The execute function `matmul_block_no_mop` does not consume its transpose argument. Changing only that argument does nothing.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`: init sets within-face transpose in `THCON_SEC0_REG2_Haloize_mode`; execute addresses logical operand B with consecutive output-column tile offsets, independent of transpose. Logical in1 is physical SrcA.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h`: transpose changes face-order address modifiers; full-tile LoFi replay already supports this, as used by QK.
- Frozen MAIN and FAST streaming materialized-PV call sites all use `mm_no_mop_reinit_short(P, V, false, ...)`. An isolated wrapper can force `transpose=true` only when `in1_cb_id==2`, retaining QK and every existing index/stride. Include the API and selected frozen common header **before** defining call-renaming macros; include the selected streaming header under those macros, then undefine them.
- Existing mixed K/V format reconfiguration remains mandatory and unchanged: transpose does not replace unpacker format reconfiguration. FAST's separate direct math restoration is under `SDPA_FP32_STREAMING`, excluded by this BF16-only prototype. Do not extend the wrapper to FP32 without also auditing that restoration.

## Prototype and validation plan

Use a new private full-chip driver, reader and compute wrapper. Compare ordinary-D-group V against token-group V behind an explicit flag; keep native/grid7 exp and MAIN/FAST/denominator-only choices orthogonal. A standalone stable-address BF16 transpose uses the standard TTNN transpose compute kernel, a source-linear reader and transposed-page writer. This avoids temporary-output allocations during trace capture. Preprocessing recomputes transpose then Q/K/V quantization on every combined replay; report transpose and quantization separately without adding them twice.

First smoke: N1024/H2, original-reference all Q rows, normal and constant V, independent K8/V4 and K4/V4 controls, then common V/outliers. Check exact BF16 transpose against the original input and exact quantizer output against the **actual** transposed BF16 tensor. Verify all output finite, two trace replays bit-identical, original inputs immutable, source hashes unchanged before/after. Only then time N32K/256K with fixed Q256/K512 and two slots. Include both mathematical quantized-V error and total operator L2/PCC; no model-quality claims.

This changes grouping direction, not BFP4's three magnitude bits, power-of-two scale restriction, clipping behavior, LoFi arithmetic or BF16 recurrence. It does not make the format identical to NVFP4. Any Sage/NVFP4 precision comparison still needs matched quantizers and accumulation semantics.

## Implemented, awaiting device qualification

`Vtransposed_fullchip.py` and `vtransposed/{compute.cpp,pv_transpose.hpp,reader_chain.cpp,transpose_writer.cpp}` implement the plan without copied or modified frozen headers. `--v-transposed` enables token grouping; omit it for ordinary channel grouping. `--grid7-exp` is independent. MAIN BF16, full FAST BF16, and FAST `--denom-only` retain exactly two KV slots; K8/V4 is the default. V8 is also supported as an axis/control ablation.

The optional `channel_v` distribution multiplies every sixteenth original BF16 V channel by 32, matching the CPU codec study. It reports quiet-channel operator error separately so the large channels do not hide small-channel loss. `--check-preprocess` verifies exact transpose and quantizer equality and reports decoded packed-V L2 against original BF16 V. Source hashes include the selected frozen `.h`/`.hpp`, reference, new files and critical transpose/no-MOP LLK implementations.

Static validation: Python compilation and three stdlib unit tests passed. Tests compare all 24 format/compensation/exp configurations with the prior builder, with/without V transpose, and confirm identical CB allocation, two slots, job assignment, Q256/K512/D128, and only the intended compute define. Symbolic page/index tests cover N1024/32768/262144 and multiple heads/chunks. These are **not** C++ compilation or hardware correctness/performance results.

Suggested first device command (then repeat without `--v-transposed` using a fresh label):

```sh
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v2/Vtransposed_fullchip.py \
  --label vt-k8v4-main-1024-v1 --destination main_bf16 --kv-formats b8_b4 \
  --length 1024 --heads 2 --cores 4 --v-transposed --check-preprocess \
  --distributions normal constant_v channel_v --iters 0
```

N1024 checks every output row against original-input FP64 attention and performs two bit-exact combined trace replays even with `--iters 0`. No prototype device job has been launched by this agent.

Audit follow-up: the source ledger also pins the transitively imported `frontier-accuracy-v1/run.py`. Transpose, original-input immutability and replay checks compare BF16 `uint16` storage bits, including signed zero. Quantizer checks deliberately compare exact decoded numeric values, not encoded BFP bytes, and label that distinction.
