# AutoDebug: BF16 KV prefill geometry exceeds available L1

## Finding

The BF16 KV candidate retains the 128/128 prefill Q/K chunk geometry selected for smaller KV tiles. For the full-attention layer (head dimension 512), its SDPA static circular buffers end at **1,408,256 bytes**, exactly matching the failure. The live L1 tensor frontier is **1,406,208**, so the overlap is **2,048 bytes**. The allocator correctly rejects the program; this is an admission/geometry problem in the autoport, not evidence of a faulty SDPA allocator or a decode kernel failure.

**Minimal recommended adaptation:** in `tt/multichip_decoder.py`'s geometry construction, select `prefill_q=64` for BF16-cache full-attention layers, retaining `prefill_k=128`. Keep the existing 32/32 fallback for attention starts not divisible by 128. This changes one geometry field for the affected policy/layer kind, keeps BF16 KV precision, and preserves the 262144-token context and arbitrary logical prompt lengths. No source changes were made in this investigation.

Confidence: high for the resource diagnosis and static capacity reduction; hardware confirmation is still required for the complete warmup, numerical results, and performance. The precise identity of the tensor setting the recorded L1 frontier is not present in the log.

## Evidence and scope

Inspected the supplied run contract and eval inputs, the current autoport, generic SDPA implementation, allocator, and relevant tests. No hardware commands, reproductions, resets, removed Gemma implementations, or external sources were used. Only this report was written.

Prepared HEAD: `c673ec5a40de93afe05c3383a359eb6df0cb71d1`, with existing uncommitted stage08 changes. The traceback's Python line numbers/source excerpts do not perfectly match current files. Therefore historical execution details are separated from current-source deductions. The exact static allocation match substantially strengthens correspondence but does not establish a complete historical source snapshot.

Direct observations:

- `doc/datatype_sweep/smoke_kv_bf16.log:22-47` loads layers 0 and 5, prints warmup buckets 32 through 704, then fails at `WARM_PREFILL 736`.
- Its runtime policy records BF16 KV, BFP8 prefill activations, BF16 residuals, and max sequence length 262144 for both loaded layers.
- The native stack names `SDPAOperation`, `compile_and_allocate`, and `validate_circular_buffer_region`; the conflicting range is `[0-0 - 7-7]`.
- `baseline.log` reaches `WARM_PREFILL 1024`. This is useful supporting evidence, but is not a fully controlled same-layer/same-allocation experiment.

Current-source causal interpretation:

- `tt/model.py:46-60` forwards the KV policy into `MultichipDecoder`.
- `tt/multichip_decoder.py:79-83,187-204` selects local TP4 geometry: 8 Q heads, full-attention 1 KV head with D=512; sliding attention has 4 KV heads with D=256. Cache dtype changes independently of the fixed 128/128 prefill geometry.
- `tt/multichip_decoder.py:230-232` uses HiFi2, FP32 destination accumulation disabled for prefill SDPA. This is distinct from the decode kernel's FP32 setting.
- `tt/optimized_decoder.py:599-635` allows L1 residual storage for up to 1024 physical token rows, generates Q/K/V, fills the paged cache, then invokes chunked SDPA with 128/128 at start zero. It has no dtype-aware capacity adjustment.

## Lowered path and exact resource ledger

All paths below are relative to the checkout. SDPA paths abbreviate `ttnn/cpp/ttnn/operations/transformer/sdpa/`.

`tt/generator.py:154-180` warms every 32-token bucket before decode or trace capture. `tt/prefill.py:12-32,43-94` supplies a persistent full-attention page table with 8192 INT32 entries (32768 bytes), 32-token cache pages, and a device offset tensor containing zero. At bucket 736, Q is locally `[1,8,736,512]`; K/V are paged BF16 DRAM cache tensors. Q and output are BF16. The supplied BFP8 activation policy controls other model intermediates; it does not make attention Q BFP8.

The tensor-offset overload at `sdpa.cpp:145-182` preserves Q/K/V order, forces causal attention, forwards the explicit geometry and kernel configuration, and defaults output memory to DRAM. `device/sdpa_device_operation.cpp:580-596` makes the output shape/dtype follow Q. The operation uses its ordinary `SDPAProgramFactory`, not a ring/joint/decode factory.

`device/sdpa_program_factory.cpp:594-638` rounds Q's effective loop extent to 768 (6 Q chunks), while preserving `valid_Sqt=23` for 736 rows. The flexible K compile-time extent is 736+262144; runtime reader bounds restrict actual reads to the prefix plus this Q interval. At start zero this is 736, not 262880 tokens of live cache reads.

The factory distributes 8*6=48 Q chunks over 64 cores in pairs, so its maximum per-core assignment is 2 and Q is double buffered (`:736-754`). Paged mode disables KV forwarding chains (`:785-795`, helper `:245-248`); K and V each use two slots (`:188-198`). FP32-destination false selects streaming compute (`:81,766`). The full-attention mask uses two BF16 tiles. Output streaming uses 64 tiles, not the smaller count one might assume from halving Q (`:864-874`; `sdpa_subblock_utils.hpp:73-79`).

Per-core CB inventory from factory `:835-845,1101-1127,1159-1224`:

| Buffer group | Q128/K128 tiles | Q64/K128 tiles |
| --- | ---: | ---: |
| Q input | 128 | 64 |
| K input | 128 | 128 |
| V input | 128 | 128 |
| Causal mask | 2 | 2 |
| Identity/column scalars | 2 | 2 |
| Reciprocal scratch | 1 | 1 |
| QK intermediate | 16 | 8 |
| Two output intermediates | 128 | 64 |
| Five statistics buffers | 20 | 10 |
| Streaming output | 64 | 64 |
| **Total BF16 tiles** | **617** | **471** |

Each tile is 2048 bytes. Add 32768 bytes for the page table, two 32-byte offset CBs, and 64 bytes of inter-CB alignment gaps. Blackhole's default CB base is 111744: the memory-map end is 41088 plus 69 KiB kernel configuration (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:111-182`, `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp:43,68-69`). Each CB starts at DRAM alignment, 64 bytes (`tt_metal/impl/program/program.cpp:2400`); each 32-byte offset CB leaves a 32-byte gap before its successor.

```
Current end = 111744 + 617*2048 + 32768 + 64 + 64 = 1408256
Q64 end     = 111744 + 471*2048 + 32768 + 64 + 64 = 1109248
Savings     = 299008 bytes/core
Margin against the recorded frontier = 1406208 - 1109248 = 296960
```

At 704 and 736, the current Q chunk count is the same (6), so SDPA static sizing does **not** jump at 736. The observed boundary reflects available live L1 capacity as physical model tensors grow, rather than a new SDPA chunk-count branch. A tensor-allocation trace is required to attribute the exact frontier to a named live buffer. Earlier printed buckets establish progress through host calls, not independently synchronized numerical correctness.

For comparison, replacing only the 256 double-buffered K/V tiles with BFP8 tiles (1088 bytes each) saves 245760 bytes of static storage. This explains why geometry safe for the baseline can fail for BF16, even before considering any dtype-related changes to live tensors.

The allocator compares the actual occupied L1 frontier against the static region at compile/allocation time (`tt_metal/impl/program/program.cpp:2560-2624`). This happens before the failing SDPA launches. The first explicit warmup synchronization in `prepare_decode` is later, before trace capture; the exception is not merely a delayed device failure reported there.

## Why the proposed geometry is legal

1. **Keep prompt handling unchanged.** Generator prefill iterates 1024-token outer chunks and rounds the last physical bucket to 32 (`tt/generator.py:223-238`). Examples: 129 -> 160; 1023 -> 1024; 1025 -> 1024+32. Selecting Q64 does not change those buckets or logical selection positions.
2. **Retain offset alignment.** Scalar-offset validation requires starts divisible by both Q and K chunks (`device/sdpa_device_operation.cpp:373-402`). The tensor-offset reader derives the Q chunk offset by integer division (`kernels/dataflow/reader_interleaved.cpp:344-363`), so the same semantic constraint applies. Every 128-aligned start is divisible by 64 and 128; existing non-128-aligned starts continue using 32/32. Do not replace 128 with 96 merely because 96 is a legal tile multiple: 1024 is not divisible by 96.
3. **Partial chunks are supported.** Factory rounding explicitly supports lengths not divisible by Q/K chunks (`:609-638`). The reader clamps Q tile rows and pads the remainder (`reader_interleaved.cpp:549-560`); paged K/V reads use the runtime-valid bound (`:401-403,679-693,897-913`). The writer drains complete internal output chunks but discards padded rows (`writer_interleaved.cpp:284-304`). At 736 with Q64, there are 12 chunks; the final chunk has one real 32-token tile and one padded tile.
4. **Preserve context and cache ownership.** Full attention still uses the full page table, the same absolute RoPE positions/cache writes, and the same 262144 admission limit. Sliding attention keeps its existing rebased/cyclic cache handling. This L1 failure does not justify shortening context or reducing KV precision.

Changing Q only on BF16 full-attention layers is the narrowest recommended policy intervention. Applying Q64 to all BF16 layers is a broader conservative option, but the observed end identifies the D512 path and does not require changing sliding attention. Q128/K64 is another legal one-field alternative, saving 278528 bytes/core; choosing between them for speed requires measurement. Neither is claimed to be optimal.

The capacity estimate uses the recorded frontier as a fixed comparison. It is not proof that every future allocation history, full-layer batch, or decode configuration fits. Keeping Q/K/V/output shapes and memory placement unchanged makes the comparison useful, but hardware must verify all warmup buckets and subsequent execution.

## Other potential issues and excluded explanations

- The generic factory's `l1_budget_bytes` uses total allocator-region capacity rather than current free space (`:763-765`). Its budget-dependent two/three-slot selector only affects KV chain mode 3. Paged prefill selects mode 0, so that capacity heuristic cannot explain this failure. No generic selector rewrite is needed for the proposed fix.
- No evidence here establishes a numerical BF16-KV bug, NoC/CB synchronization failure, trace capture bug, or context-capacity limit. Those claims would overreach an allocation exception.
- Policy unit tests in `tests/test_precision_policy.py` validate configuration schemas, not geometry feasibility. Generic `test_sdpa_prefill.py` includes divisibility-filtered cases (`:676-707`); those tests alone do not establish this persistent tensor-offset, partial-tail model path.

## Required follow-up verification (not run)

- Run the same layers0/5 BF16 candidate through **all 32..1024 warm buckets**, especially 704/736/768/1024, then its existing 129/1023/1025 generation smokes.
- Exercise scalar and tensor offset paths at starts 0,32,64,96,128,1024 and near the context boundary, with partial final buckets; preserve correct causal/sliding masks and page ownership.
- Validate BF16 versus reference outputs with the stage's existing accuracy criteria; do not relax precision or accuracy to clear an allocation failure.
- Verify full-layer/default-batch runs and advertised-context boundary execution, recording actual allocator headroom. This reduced-layer diagnosis does not certify whole-model DRAM capacity or decode resources.
- Measure warmed prefill/serving performance for Q64/K128 versus Q128/K64 if choosing a final sweep configuration. Rebuild warmup/traces after changing geometry.
