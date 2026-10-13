# topk-accuracy

## What the code does
`ttnn.experimental.topk_large_indices(input, k=..., valid_length=...)` (Blackhole only) returns, for every row of
a ROW_MAJOR tensor, the indices of the `k` largest values (16 <= k <= 2048, a multiple of 16), sorted by
descending value, as UINT32 ROW_MAJOR `[.., k]`. Exact -inf values yield the 0xFFFFFFFF sentinel. The sparse-SDPA
(DSA) indexer in `models/demos/deepseek_v3_d_p/tt/mla/indexer.py` (`select_local`) uses it to pick the
key positions each query attends to: rows are the chip's queries, the last dimension is the key length (up to
tens of thousands), and k is the index top-k capacity (1536-2048).

The input must be BFLOAT16 today (`TT_FATAL` in `device/topk_large_indices_device_operation.cpp`). The indexer
logits are FP32 sums of `w_h * relu(q_h . k_t)` over heads, so rounding them to BF16 (8 mantissa bits) turns many
near-ties at the k-th score into exact ties. The op then keeps some keys that the FP32 ranking would drop. The
kernel ranks BF16 exactly. Internally each candidate is a 32-bit dest word: the BF16 value in the upper 16 bits
(fp32 dest, `cb_in` is Float16_b) and the index stamped into the low 16 bits (chunk id in bits [15:11], see the
`topk_xl_add_lsb_indices*` / `separate_indices` LLKs). The TopK XL SFPU sort/merge works on those words.
So FP32 input cannot just be copied in: its low mantissa bits overlap the index field.

**Goal: improve selection accuracy against the FP32 ranking** while keeping device kernel time within 5% of
the baseline on every case. One route is to accept FP32 input and rank at more than BF16 precision, for
example by refining only the candidates near the k-th threshold, or with a different value/index packing.
Any approach is fine if it is general, keeps the output contract, and keeps BF16 input exactly as it is.

## Where the code is
- `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/`: op, validation, nanobind docs (host, needs a
  rebuild), `device/topk_large_indices_program_factory.cpp` (CBs, data formats, core split, compute body mode:
  fused wide-row vs classic), `device/kernels/{reader,compute,writer}.cpp` (JIT-compiled).
- TopK XL LLK stack (Blackhole):
  `tt_metal/hw/inc/api/compute/experimental/topk_xl.h`,
  `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/*topk_xl*`,
  `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h`,
  `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/*topk_xl*` (unpack/datacopy into dest).
  LLK tests (`tt_metal/tt-llk/tests/.../topk_xl*`) are reference material, not part of the eval.

## What is measured
`agent_orch/campaigns/topk-accuracy/eval.py` runs
`tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices_accuracy.py` under Tracy, 4 cases
(1 chip): 256x8192 k2048, 128x32768 k2048, 640x51200 k1536, 128x65536 with valid_length 40960 k2048, 3 warmup +
10 measured calls each. The test builds FP32 indexer-like logits and **offers them to the op as an FP32 tensor**,
falling back to BF16 if the op raises. Per case:
- `value` = `miss_pct` + 0.001: the % of returned indices whose FP32 score is below the row's FP32 k-th score
  (ties at the threshold count as correct). Lower is better. Baseline (BF16): 0.096 / 0.26 / 0.38 / 0.29 %.
- `duration_us`: mean `TopkLargeIndicesDeviceOperation` DEVICE KERNEL DURATION. Baseline 111 / 278 / 1285 /
  347 us. `slowdown` = duration / baseline must be <= 1.05 (gate). Note that FP32 input doubles the bytes read.
- Also logged: `mass_gap_ppm` (lost score mass) and `fp32_input` (whether the FP32 path was taken).
- Then `tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py` (154 BF16 exactness /
  argument tests) must pass. Output indices must stay in range and distinct per row.

## Constraints
See the campaign rules: fixed output contract, the BF16 path unchanged, no shape- or data-specific tricks, perf within 5%.
