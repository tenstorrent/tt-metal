# V5 terminal-row TILE gather diagnostic

The native gather source admits FP32 TILE input. A diagnostic candidate is prepared against qualified V5 `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f`; it is **rejected after exact component checks and measured whole-decoder slowdown**. Production, native source and existing probes are unchanged. This is a source-based experiment, not a claim that a transpose/gather adapter is faster.

The reported selected-profile opportunity is the terminal-row extraction in ordered projection: approximately 35.706 microseconds Untilize plus 21.634 microseconds Tilize. The replacement retains the original aligned last-32-row slice, then transposes FP32 H/W, gathers column 31 using preallocated full-tile indices, transposes back, and changes only logical H to 1. The original FP32 block reduction and final output cast follow unchanged. Thus both K512 decode and K1024 serial prefill use exactly the existing product, cumsum and block-sum order.

## Completed device decision

The parent-owned [complete projection probe](v5_terminal_gather_device.json) passed all seven cases for FP32 and BF16 outputs, including eager/trace agreement and both K policies. Nevertheless, complete traced projection latency increased: singleton K512 **0.280329→0.463937 ms**, singleton K1024 **0.253288→0.344606 ms**, and S128 **30.592375→42.260292 ms**. Every measured sequence length regressed.

The [whole S256 candidate](v5_terminal_gather_s256.json) also regressed against the [unchanged V5 control](v5_prefill_pair_blocks_baseline_s256.json): warmed prefill **107.8132→127.394980 ms**, traced warmed decode **1.285217→1.470923 ms**. Short PCC remained identical (prefill0.999962395060449, decode0.9999819053980984). [Decision and raw hashes](v5_terminal_gather.decision.json) preserve the exact values and source identities. The complete adapter costs outweigh the eliminated Untilize/Tilize pair. This is a measured performance rejection, not a dtype or correctness rejection; no8192 endurance run is needed or claimed for this slower candidate.

## Native contracts and costs

- [Native validation](../../../../../../ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp) accepts matching TILE input/index layouts and UINT16 or UINT32 indices. It does not restrict input to BF16. [Codegen eligibility](../../../../../../ttnn/cpp/ttnn/operations/data_movement/gather/codegen/gather_codegen_supported.cpp) does restrict BF16; FP32 therefore selects the native route in [gather.cpp](../../../../../../ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp).
- The [native factory](../../../../../../ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp) derives CB formats and tile bytes from actual operand dtypes. The [dataflow helpers](../../../../../../ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_common.hpp) read/write a four-byte payload through UINT32 pointers, with no floating-point arithmetic. This supports FP32 payload transport; the complete device probe still must check transposes and the consumer.
- The [H/W transpose factory](../../../../../../ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_wh_program_factory.cpp) derives formats and enables 32-bit destination for FP32 via `requires_32bit_dest`. Both runtime transposes and gather explicitly request interleaved L1. No transposition cost is omitted from timing.
- Each pretransposed index tensor has shape `(blocks, 1, 6208, 32)`, contains 31 in every lane, and resides in DRAM. Both last dimensions are tile aligned, so [implicit padding fill](../../../../../../ttnn/cpp/ttnn/operations/data_movement/fill_pad/fill_pad.cpp) returns immediately. Calling gather on its last dimension avoids gather's automatic index transpose and its non-last-axis output-placement rule.
- Setup indices for blocks 2 and 4 add exactly **2,383,872 DRAM bytes**, no additional weight copies and no persistent L1. One terminal FP32 buffer is 3,178,496 bytes for K512 or 1,589,248 bytes for K1024. The adapter temporarily holds several such buffers and writes all 32 replicated physical rows; those costs can outweigh the saved layout conversion. The candidate does not claim optimality or a capacity pass.
- After the back-transpose, all physical rows contain the original terminal row. Only logical row zero is consumed. The CPU mapping proof checks both this logical row and every replicated physical row using distinct finite FP32 bit payloads, for both block counts. There is no CPU forward fallback in the device path.

## Artifacts and parent-only device command

- [Candidate](v5_terminal_gather.py.snapshot), SHA256 `87bed62a409b7c08d71e8d134ca2514e76f37b3573be2073386efbe6858d63d4`.
- [Source/probe provenance](v5_terminal_gather.py.provenance.json): complete module AST is identical to the V5 baseline after removing the one setup allocation and restoring the one terminal slice. It also hashes the relevant native source files.
- [CPU proof](v5_terminal_gather_cpu_final.json) and [run provenance](v5_terminal_gather_cpu_final.json.provenance.json): both mappings exact; no TTNN import. The earlier `v5_terminal_gather_cpu.json` is retained as the pre-final-probe CPU run, with its own source hash.
- [Complete projection probe](../../../tests/v5_terminal_gather_probe.py): real layer-0 checkpoint K/V/A/B weights, synthetic normalized BF16 activation rows, FP32 and BF16 output comparisons over all 6208 columns and separate K/V versus A/B ranges. Cases are singleton K512/K1024 and S2/32/33/128/129. Complete eager and traced timings include all runtime adapters, serial row concatenation and final tilize. It records source hashes, setup/readback time, replay correctness, and closes mesh 0 in `finally`. Formatting and pre-commit passed.

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_terminal_gather_probe \
  --baseline-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/selected_fused_decoder_v5.py.snapshot \
  --candidate-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_terminal_gather.py.snapshot \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_terminal_gather_device.json
```

This comparison is diagnostic. The measured slowdown rejects this candidate; it is not selected and does not alter production qualification or quality gates.

## Reverse cumsum and other extraction routes

The reverse-cumsum idea is not prepared. In [accumulation_device_operation.cpp](../../../../../../ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_device_operation.cpp), `is_tile_axis_accumulation` explicitly returns false when `flip` is true. [accumulation_common.cpp](../../../../../../ttnn/cpp/ttnn/operations/reduction/accumulation/accumulation_common.cpp) consequently swaps H with dim 0 before scanning and reverses that permutation afterward. On the actual product shapes, `(4,1,512,6208)` becomes logical `(512,1,4,6208)` with physical H32, and `(2,1,1024,6208)` becomes `(1024,1,2,6208)` with physical H32. Each FP32 product/prefix expands from 101,711,872 bytes to 813,694,976 bytes (K512) or 1,627,389,952 bytes (K1024). This cannot preserve the current bounded interleaved-L1 path. Moving the expanded products to DRAM introduces a different, much larger adapter; no performance rejection is claimed without measuring it, but the existing-L1 proposal has a concrete source/geometry blocker.

A sub-tile `roll` is not a conversion-free extractor: [roll.cpp](../../../../../../ttnn/cpp/ttnn/operations/data_movement/roll/roll.cpp) routes non-tile-aligned H/W shifts through untilize/roll/tilize. An ordinary row-31 slice is the measured baseline. This audit found no other existing native direct row extractor with a demonstrated cheaper, FP32-preserving complete adapter; it does not assert that none exists.
