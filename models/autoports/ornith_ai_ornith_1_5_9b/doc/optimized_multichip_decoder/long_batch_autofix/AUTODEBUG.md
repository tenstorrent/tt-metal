# AutoDebug: long-history TP4 batch-32 decoder corruption

The strongest diagnosis is an unsafe page-table tail in paged SDPA decode, followed by a misleading PCC assertion. The newly saved failing tensors are **exactly equal on all four ranks**: this is common output corruption, not ordinary rank-dependent BF16 reduction error. The caller's padding correction now has a recorded 128-step batch-32 pass with the original reference and strict thresholds.

This investigation was inspection-only. I read source, archived logs, command metadata and saved tensors; CPU-only checks verified hashes, dimensions, tensor equality and numerically stable PCC. I did not run the decoder, access devices, launch AutoDebug again, or edit implementation/test source. The caller continued working in the shared tree and produced additional diagnostics and a padding fix during this investigation; those changes and runs are identified separately below.

## Findings and confidence

1. **Confirmed correctness bug in the failing revision:** decode passes 133 logical page entries to a kernel that reads 136 entries. The three extra entries are unspecified physical alignment padding, not initialized page IDs. This is the leading explanation of the corruption: the source contract is violated exactly in the long fixture, and the caller's attention-table-only padding change passes the same long test. The exact bogus page IDs and the first corrupted intermediate were not captured, so the numerical propagation from those reads to the enormous final values remains unverified.
2. **Confirmed diagnostic bug:** float32 PCC overflows on the saved finite values around `1e34`. It returns NaN for identical ranks and zero against the reference. The assertion text therefore misidentifies this recorded failure as rank disagreement. Float64 comparison establishes both exact rank agreement and severe reference disagreement.
3. **The proposed ordinary BF16 accumulation-order explanation is contradicted by the selected source path.** The minimal persistent-buffer all-reduce gathers into canonical source-rank slots and executes the same ordered pair additions on every destination. No source-supported collective count, ping-pong lifetime, or cache-hit address-rebinding defect was found for this geometry.

## Evidence and revision ledger

All model evidence below is under `models/autoports/ornith_ai_ornith_1_5_9b/`.

| Artifact | What it establishes |
|---|---|
| `doc/optimized_multichip_decoder/integrated_long_stack_32_command.json`, original `.log` | Original requested B32/prefix4095/128-step failure. Decoder SHA `0386f8b6c817…`; harness SHA `508f77fb4743…`. The original log identifies only a decode readback assertion, not its step/row or error magnitude. |
| `doc/optimized_multichip_decoder/source_snapshots/multichip_decoder_0386f8b6c817.py.txt` | Exact failing decoder. Its SHA matches command metadata. Use this snapshot for historical line references; the live decoder changed during investigation. |
| `doc/optimized_multichip_decoder/source_snapshots/multichip_stack_508f77fb4743.py.txt` | Exact original harness, also hash-verified. Lines 80–94 perform the rank gate before the `.995` reference gate; lines 228–237 replay with `blocking=True` before reading output. |
| `doc/optimized_multichip_decoder/integrated_long_stack_32_diagnostic_command.json`, `.log`, `.json.failure.json`, `.json.failure.pt` | Caller-added diagnostics retain original decoder `0386…`, with harness `9cfcc235ec42…`. First reported failure is `decode_0_8`; 136 checks completed. The saved four tensors permit exact comparison. |
| `doc/optimized_multichip_decoder/integrated_long_stack_padded_32_command.json`, `.json` | Same long command, reference, diagnostic harness, tracking settings and capacity; decoder SHA `6235412425cb…`. Completes 128 trace replays and all 4416 reference checks. Minimum reference PCC **0.9987144470**, at `decode_27_13`. Rank threshold remains `.99999`. |
| `doc/optimized_multichip_decoder/long_stack_stage4.json` | Caller-run serial/private-state Stage4 control also completes 4416 checks, minimum reference PCC `0.9987227917`. This changes several implementation choices and is weaker isolation than the attention-table padding contrast. |

The live decoder diff from `0386…` to observed `623541…` changes attention-table preparation in single-request and packed decode. The single-request helper also handles prefill: moving padding before the pool-offset addition changes prefill's padded tail mapping from global physical page zero to each request's physical page-zero offset. Thus the contrast includes that prefill mapping correction as well as decode padding. Cache-update tables remain separate. It does not change all-reduce precision or relax thresholds. This report does not attribute that caller-authored patch or its hardware runs to the investigator.

### What the saved failure actually means

CPU inspection of `integrated_long_stack_32_diagnostic.json.failure.pt` found:

| Measurement | Result |
|---|---:|
| Each rank's shape/dtype | `[1,1,1,4096]`, BF16 |
| Tensor elements finite | All four ranks: yes |
| Exact equality with rank 0 | All three comparisons: yes |
| Actual minimum / maximum | `-1.1195890101e34` / `4.8028745941e34` |
| Reference minimum / maximum | `-3.40625` / `15.375` |
| Inter-rank PCC in float64 | `1.0`, `1.0`, `1.0` |
| Reference PCC in float64 | **0.1337192438** |
| Existing float32 inter-rank PCC | NaN, NaN, NaN |
| Existing float32 reference PCC | `0.0` |

The helper at `tests/run_decoder.py:61–64` casts to float32, checks input finiteness, then calls `torch.corrcoef`. Finite inputs do not guarantee finite covariance: the actual variance overflows to infinity. Self-correlation becomes `inf / inf`, while reference correlation has an infinite denominator. The saved JSON's zero maximum differences are consistent with exact equality, not evidence of small BF16 differences.

`decode_0_8` means step 0, request row 8, position **4114**. Its 136 completed records are 128 prefill checks plus the first eight decode rows. This location is established for the diagnostic rerun, not retrospectively for the original unspecific log. The helper should measure PCC in float64 or normalize safely before covariance, and distinguish nonfinite metrics from unequal tensors. Correcting the metric will expose the real `.995` failure; it will not fix the decoder output.

The stack's reference is single-chip TTNN `OptimizedDecoder`, not a direct Hugging Face full-model reference (`multichip_stack` snapshot:96–97,164–188).

## Runtime geometry and why the passing cases differ

The failing stack uses real layers `[0,3]`, linear attention followed by full attention, on mesh `(1,4)`. Each device owns different attention/MLP channels; residual activations are replicated. Packed decode is BF16 `[1,1,32,4096]`, with one physical 32-row tile across both layers.

The original harness derives capacity rather than using the nominal prefix length directly (`multichip_stack` snapshot:164–189):

```text
capacity = max(1152, ceil_to_32(4095 + 2 + 18 + 128)) = 4256 tokens/request
pages/request = 4256 / 32 = 133
initial lengths = 4095 + row % 3
continuation lengths = 17 + row % 2
starts repeat [4112, 4114, 4114, 4113, 4113, 4115]
requested positions over 128 steps = 4112 ... 4242
```

KV pool shape per device is `[32*133,1,32,256] = [4256,1,32,256]`, BFP8. Request `i` owns pages `[133*i,133*(i+1))`; the maximum physical page is 4255. Linear state is convolution BF16 `[32,1,3,2048]` and recurrence FP32 `[32,8,128,128]`. These allocations and offsets are internally consistent; no request-region overlap was found (`multichip_decoder` snapshot:437–478,1050–1068).

| Passing observation | Coverage and relevant limitation |
|---|---|
| `integrated_track_stack_32.json` | Same failing decoder SHA, B32 pooled stack, prefix31/32/33, capacity1152/36 pages, 16 evolving replays, positions48–66. SDPA reads only virtual pages0–7, safely within the 36 logical entries. 768 reference checks pass; minimum PCC `.999560058`. |
| `integrated_stress_{0,3}.json` | B1, each layer separately, 128 evolving steps at positions128–255 with capacity256. Checks each rank against baseline at `.995`, including contribution PCC; it does not enforce the stack's pairwise `.99999` gate. |
| `integrated_long_{0,3}.json` | B1, each layer separately, capacity262144/8192 pages. Samples very long prefills and replays the terminal decode position three times with restoration. It is not 128 advancing long-context steps. Its page count is divisible by eight. |

Before the diagnostic failure, all sampled initial prefills, all continuation outputs, two warm forwards/restores, capture, and a blocking replay completed. Original final-state checks were not reached. Blocking replay/readback rules out the simple interpretation that the host read an unfinished nonblocking replay; it does not by itself validate any device buffer contents.

## Leading bug: SDPA consumes page IDs the caller never initialized

### Lowered contract

The failing decoder snapshot's `full_attention_decode_batch` concatenates page tables at lines 1060–1068 and passes the resulting `[32,133]` table directly to paged SDPA at lines 1070–1080. It fixes `k_chunk_size=256`, with 32-token cache pages. Single-request decode also passes an unpadded table (snapshot:767–780). In contrast, its prefill path explicitly pads the attention table to a multiple of eight entries (snapshot:791–804), documenting the missing preparation through a sibling path.

The lowered implementation needs complete chunks:

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:60–69` rounds `cur_pos+1` up to the configured chunk size. Lines 94–106 retain the explicit eight-tile chunk size; dynamic shrinking applies only when configured size is zero.
- At position4114, `ceil_to_256(4115)=4352`: 17 chunks, 136 cache pages. The last chunk walks virtual page indices **128 through 135**. Actual initialized table entries end at index132. This mismatch exists from the first replay and throughout the requested position range.
- `sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:754–781` and `:833–860` read all eight K/V tile rows in that chunk. The shared mapper `sdpa/device/kernels/dataflow/dataflow_common.hpp:53–68` dereferences `page_table_ptr[virtual_block]` without a logical-length guard.
- The caller omits `cache_position_modulo`; the wrapper defaults it to absent and the factory lowers it to zero, so the mapper's optional circular-cache wrap does not rescue these indices (`sdpa_decode.hpp:44`; `sdpa_decode_program_factory.cpp:55–62`).
- `sdpa_decode/device/sdpa_decode_device_operation.cpp:454–466` checks page-table capacity divisibility for noncausal mode only. Causal admission therefore does not reject this undersized logical descriptor.

Masking future attention positions is not a bounds guard on these reads. It cannot make arbitrary physical page IDs valid. With arbitrary padding, the kernel can issue KV reads outside the allocated pool and interpret unrelated memory as BFP8. However, the compute path does apply a real negative-infinity causal mask before softmax (`sdpa_flash_decode.cpp:449–453`), and its approximate exponentiation uses packer ReLU (`sdpa/device/kernels/compute/compute_common.hpp:365–373`). Correctly masked finite V values should contribute zero. I therefore do not claim that ordinary zero-times-large-finite arithmetic explains the corruption. The exact invalid IDs, arbitrary-data conversion behavior and subsequent arithmetic responsible for the saved `1e34` values were not recorded; that numerical link remains unverified despite the direct unsafe-read bug and padding/pass contrast.

### Alignment padding is not initialized descriptor data

On Blackhole DRAM, the 133-entry int32 row has 532 logical bytes and a 576-byte aligned physical stride: `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379–394` and `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:383–385`. SDPA explicitly uses the aligned stride (`sdpa_decode_program_factory.cpp:233–237`). The required extra three entries fit inside physical padding, which explains why a tensor shape validator need not notice the problem.

That padding is not guaranteed page zero. For the 32-input native concat, `data_movement/concat/device/concat_program_factory.cpp:199,402` uses the logical `page_size()`; its reader copies that byte count, and `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp:22–42` writes exactly 532 bytes per row. The remaining 44 physical bytes are unspecified. Host upload also copies logical bytes while advancing by aligned bytes (`tt_metal/impl/buffers/dispatch.cpp:109–110`; `tt_metal/impl/dispatch/device_command.cpp:889–893`). Physical allocation therefore does not prove valid tail page IDs.

### Intervention and observed contrast

The smallest repair is to pad **the attention-read table** to 136 actual logical entries with a valid local page-zero mapping, then add each request's pool offset. Keep cache-update addressing separate. This preserves the existing state capacities, random page permutations, prefixes, inputs and reference fixture. The caller's observed `623541…` revision implements that preparation in both decode paths and the corresponding long run passes all 4416 checks.

This is substantially stronger evidence than a precision-only workaround: the intervention directly repairs the proven descriptor/read mismatch and the exact long fixture passes with BF16 all-reduce unchanged. It remains one recorded fixed-run contrast; attention-table preparation also changes the prefill tail mapping described above, allocations and timing, so the run does not isolate those effects or mathematically exclude an independent latent race. The defect also occurs during some serial nonaligned continuation calls before capture, whose checks passed. Unspecified padding can be harmless on one allocation/use and harmful on another; the source does not promise a deterministic failure on every access.

## Why ordinary async all-reduce numerical order is not the cause

Production `decode_reduce` calls the persistent-buffer overload (`multichip_decoder` snapshot:402–416). `all_reduce_async.cpp:541–572` dispatches the minimal primitive directly. This is not the composite reduce-scatter/all-gather overload elsewhere in that file.

The source-rank index is the cluster-axis coordinate (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:277–292`). The writer places source `s` into its fixed slot at `s * num_tiles_per_core * page_size` on every destination (`all_reduce_async/device/kernels/dataflow/worker_writer.cpp:117–145`). The receiver waits for all four contributions before publishing the complete input (`reduction_receiver.cpp:31–36`). The compute kernel then reads the same slots in this order (`kernels/compute/reduction.cpp:29–60`):

```text
D = ELWADD(source0, source1, initially-zero DST)
D = ELWADD(source2, source3, D)
pack D to BF16
```

`add_init(..., true)` means accumulate `A+B+DST` (`tt_metal/hw/inc/api/compute/eltwise_binary.h:74–89`). There is no intermediate pack between source pairs. Ethernet arrival order cannot permute these operands. Ordinary rounding can change agreement with a higher-precision reference, but identical gathered bytes do not acquire different source-level addition orders on different ranks. The saved diagnostic tensors in fact satisfy exact rank equality.

The default binding sets `fp32_dest_acc=False` (`all_reduce_async_nanobind.cpp:129–136`). Setting it true enables **both FP32 destination accumulation and full destination synchronization** (`all_reduce_async_program_factory.cpp:469–477`). A pass with `--precise-ccl` would therefore not isolate arithmetic from synchronization/scheduling. The factory comment about independence from ETH arrival order should not be treated as evidence that the default path sums in arrival order.

Target-specific audit: Blackhole's selected `add_tiles` uses the standard, nonbroadcast, non-destination-reuse FPU path, not an SFPU approximation. Startup configures unpack, math and pack (`api/compute/compute_kernel_hw_startup.h:63–79`); Blackhole kernel entry clears DST (`hw/ckernels/blackhole/metal/common/chlkc_list.h:29–38`), and pack release clears it again (`tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:65–83`). Unpack configuration defaults stochastic rounding off (`common/inc/cunpack_common.h:761,839–847`). No concrete stale-mode or accumulator-initialization violation was identified here. Two additions cannot ordinarily amplify modest inputs to `1e34`; such a result requires corrupt/huge inputs or an actual implementation/hardware fault.

### Buffer/count and asynchronous completion audit

Passing short B32 and failing long B32 have the same all-reduce physical work:

| Per output core | Concrete value |
|---|---:|
| Output shard | `(32,128)`, BF16, four 32x32 tiles |
| Receive shard | `(32,512)`, sixteen tiles, 32768 bytes |
| Contributions / semaphore wait | Four |
| Compute loop | One pack iteration, four destination tiles, two source-pair iterations |
| Output cores | 32 |

The receive capacity is exactly four source blocks (`factory:401–432,480`). Each source sends one notification per completed output shard, plus its local notification (`worker_writer.cpp:134–145,166–171`); the receiver publishes sixteen tiles and resets its semaphore. No count/capacity discrepancy is exposed by the long prefix.

Both layers issue exactly two reductions and return their local `ar_index` to zero, giving shared-buffer order `0,1,0,1` per stack call (decoder snapshot:402–405,600–606,1150). Under normal in-order local program completion, ping-pong reuse is protected: a rank cannot start reduction `k+2` until it has received every rank's `k+1`; every rank can send `k+1` only after finishing local `k`. This rules out a simple unsupported assertion that two shared buffers necessarily overwrite a slower rank's active input.

Payload and notification address the same destination core. The existing fabric ordering contract permits that write-before-atomic path without an extra flush (`tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_kernels_utils.hpp:1292–1294`), and the writer closes connections and ends with a write barrier (`worker_writer.cpp:172–179`). `flush=False` alone is not a demonstrated bug.

Cache-hit rebinding refreshes the input base, both semaphore arguments, receive CB and output CB (`all_reduce_async_program_factory.cpp:678–700`). No missing per-call address refresh was found. Link count is dynamically discovered; nearby saved runtime evidence records two links, but the failing log itself does not print that value. Rank-slot order and per-output-core counts above do not depend on assuming two links.

## Other potential issues and test gaps

- **Trace/cache corruption remains a fallback hypothesis, not a finding.** Pooled row ownership, unique region checks, deduplicated snapshots and restoration into persistent allocations are consistent in the failing path. The full-attention batch uses a single shared KV pool, so its multi-pool concatenate/scatter-back branch is not exercised. No source-supported trace ownership contradiction was found.
- **PCC is an inadequate rank-equality diagnostic for extreme/constant data.** Preserve the strict gate but report exact equality/max differences separately, and distinguish metric overflow from actual nonfinite tensors. The current failure is a concrete example.
- **Standalone CCL coverage is weaker than this model's rank contract.** `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:103–104` uses eight buffers; lines 278–293 compare each rank to a common golden with PCC, not exact rank equality. This is a coverage gap, not proof of a two-buffer defect.
- **Changing `--steps` changes the fixture.** Reducing 128 to 1 without pinning context changes capacity 4256→4128, page count 133→129, random permutations, allocation geometry and subsequent random inputs. Preserve `--context 4256` when isolating replay count; omit final state checks when shortening, because their expected labels/pages depend on final position.

## Follow-ups if further confirmation is needed

1. Retain the caller's same-fixture padding pass and record repeated runs if nondeterminism is a concern. For stronger isolation, sanitize only the three consumed attention-table tail entries while preserving original capacity, inputs, reference and strict thresholds. Do not infer an arithmetic fix merely from a passing FP32/full-sync variant.
2. At the first bad step, save pooled table contents including consumed tail entries, and capture ranges/finite counts after attention and before each of the four stack all-reduces. This locates the first enormous value and can complete the remaining numerical causal link.
3. If collective behavior remains suspect, freeze one selected reduction before its receive buffer is reused. Save each rank's projected input, gathered receive buffer, output and surrounding residual. The host receive width can be viewed as `[32 output cores,4 source ranks,128 columns]`; selecting a source slice reconstructs that source's input. Different gathered bits indicate transfer/address/lifetime trouble; identical gathered bits with different outputs indicate compute/configuration trouble. Instrumentation can change timing, so a passing instrumented run alone is not exonerating.
4. Compare eager and traced execution from identical snapshots only if the padding-fixed fixture still fails. A synchronization-before-capture variant is useful as a controlled experiment; absence of an extra host synchronization is not itself a proven bug.

## Report review

I revisited the headline claims against the failing source snapshot and the new artifacts, with an independent subagent review of the draft. The original rank-divergence narrative was rejected when saved tensor correspondence proved exact equality. The padding mismatch was traced through runtime chunk rounding, unchecked page lookup, physical stride and the actual 32-input concat writer; allocated alignment was not mistaken for initialized page IDs. BF16 arrival order, automatic pool overlap, missing cache rebinding, unconditional ping-pong overwrite and missing-flush stories were demoted or rejected because the selected source contradicts their required preconditions. The leading padding diagnosis has a direct source violation and a revision-matched repair/pass contrast; the precise corrupted-page-to-output arithmetic is explicitly left unproven, and the repair contrast's additional prefill tail mapping effect is disclosed.
