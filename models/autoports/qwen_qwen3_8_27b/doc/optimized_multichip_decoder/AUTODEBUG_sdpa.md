# AutoDebug: TP4 Qwen3.8-27B decode SDPA chunk 512

## Finding

**The leading diagnosis is a physical-capacity / decode-chunk contract mismatch.** The model supplies 136 valid page-table entries (4352 tokens), but fixed `k_chunk_size=512` makes SDPA consume 144 entries (4608 tokens) at every tested position, 4096–4223. Entries 136–143 are not caller-provided page IDs. The default model chunk 256 consumes exactly the 136 supplied entries. Causal masking happens after the K reads and QK multiplication and does not prevent either K or V tail reads.

**Blackhole-specific qualification:** these eight words occupy the 32-byte transfer-alignment padding of the 576-byte table stick. This is an out-of-range access to the logical table, **not a demonstrated overrun of its allocated L1 circular buffer** on the reported architecture. That padding has no valid-page-ID initialization contract. Interpreting it as addresses can read unrelated cache/DRAM contents.

An additional caller-generated artifact became available during inspection: `sdpa_pad_aligned512_l3.json` uses 144 valid pages with the same chunk-512 policy and passes all 128 decode correlations, minimum **0.9999394903707336**. This strongly supports fixing the capacity contract first. The exact invalid page IDs and first corrupted numerical intermediate were not captured, so the final conversion from bad reads to the observed degenerate outputs remains unmeasured. This is not normal precision loss.

**Evidence discrepancy:** the original artifacts establish 128 **NaN PCC values**, not 128 tensors proven to contain NaN elements. The runner's successful replica and replay `torch.equal` checks would ordinarily reject tensor NaNs. Constant output or infinities can also yield NaN correlation. The report preserves this distinction rather than inventing a tensor-level observation.

## Scope and evidence provenance

Investigation was source/artifact only. No target code, accelerator commands, device queries, tests using TTNN, resets, or implementation edits were performed. Three independent xhigh investigators audited the reader contract, numerical/CB path, and model ownership. Only this report was written by this investigation.

Checkout HEAD: `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`, with pre-existing model changes. Original artifacts under `models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/`:

| Artifact | Decode chunk / capacity | Decode PCC | Other observations |
|---|---|---|---|
| `sdpa_control_l3.json` | 256 / 4352 | 128 finite; minimum 0.9999434865886184 | Prefill 0.9999760875900713 |
| `sdpa_chunk512_l3.json` | 512 / 4352 | 128 NaN | Identical prefill and K/V PCC to control |
| `sdpa_chunk128_l3.json` | 128 / 4352 | 128 finite; minimum 0.9999420780686875 | Additional safe-extent contrast |
| `sdpa_pad_aligned512_l3.json` | 512 / 4608 | 128 finite; minimum 0.9999394903707336 | Caller-generated padding experiment, read only |
| `sdpa_chunk1024_l3.log` | 1024 / original pool | No comparable completed output | Lines 39, 65: static CBs grow to 1,690,432 B, beyond 1,572,864 B L1 |

Original control and failing artifacts have identical four-file provenance and policies differing only by `sdpa_chunk: 512`. Their K/V PCC values are respectively `0.9998951419924735` and `0.9998742528117408`. Both report restored replay equality, replica equality, no page-remap test, and no asynchronous stress. Their logs identify **Blackhole**, e.g. failing log line 5.

Initially all four recorded hashes matched the inspected source. During this investigation the caller independently added `--capacity-alignment` to the runner; its hash changed from `a1432743910336e4a9df402f7cfb67905f8ebcf50135632060fbe9b506d0d6a7` to `881fee05b8462205718d12e5d91347cd6e2bdca99f162fef56cad3669f145c74`. The aligned512 artifact records the latter. The three model files remained identical across these artifacts and inspection:

- `tt/multichip_decoder.py`: `a1dc704b24892a35f5d9ad4307accb5ff0d9b1b79f6b764c10438304cc1a3a14`
- `tt/optimized_decoder.py`: `ed3f7efd9f0e743a0863f264c7c966e9af0677b588d7e90f015a656db368fadf`
- `tt/decode_workspace.py`: `b0973635a484da7ce9734edd83bb6046cd96ec1e2e2a9943cf23e3f5b735aab2`

Source line citations below use the inspected tree, including the updated runner. Kernel source was unchanged in the working tree. Artifacts do not independently fingerprint the loaded C++ binary. The padding experiment also changes randomized page placement, cache allocations, and table contents; it is strong supporting evidence, not a dump of the original bad IDs.

## Lowered path and earliest divergence

1. The runner prepares 4096 prefill inputs and 128 independent recorded decode inputs. Decode positions are 4096 through 4223; this is a layer comparison, not autoregressive feedback of a bad decoder output. Original capacity was `ceil(4224/256)*256 = 4352`. The current generalized equivalent and random page table construction are at [run_multichip_decoder.py:76–79](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:76). The table is replicated INT32 row-major, `[1,136]`; its entries permute physical IDs 0–135.
2. [Multichip allocation:316–331](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:316) also rounds to 256 tokens, independently of `sdpa_chunk`. Each chip owns zero-initialized K/V `[136,1,32,256]`, BFP8 here. The artifact's `[136,4,32,256]` is the host reconstruction of four local KV heads, not four heads per chip. The allocation comment concerns eight INT32 page IDs / 32-byte sticks; it does not establish chunk-512 read safety. This existing rule is real: the [flexible prefill factory:162–165](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:162) requires raw table-row bytes to be divisible by32. Both136 and144 entries satisfy it (544 and576 bytes).
3. [Attention:448–479](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:448) projects with BFP4 weights, splits Q6/KV1, normalizes Q/K, applies RoPE, permutes to decode layout, and updates both caches through the supplied table. SDPA receives tiled BF16 Q logically `[1,1,6,256]`, physically padded to 32 heads; independent tiled BFP8 K/V in DRAM; row-major INT32 positions and table; causal mode; scale `1/16`; no explicit mask, sliding window, geometry override, or cache modulo.
4. The `sdpa_chunk` override reaches only decode SDPA at line 475. Prefill has a separate `prefill_sdpa_k=256` at lines 485–501. The SDPA config selects **HiFi4, FP32 destination accumulation, math approximation false**, from lines 41–47; projection LoFi is a different config. The caller also supplies `packer_l1_acc=True`, but the factory extracts that field at line 219 without forwarding it in its compute descriptor at lines 946–951; kernel helpers control accumulation locally. `exp_approx_mode=False` is explicitly passed. Effective policy supplies 32 cores/head.
5. [Paged wrapper:133–172](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:133) passes fixed 256 or 512 without clamping to the table extent; its explicit-chunk validation checks power-of-two and divisibility by 32. “Default 256” here is the **model's explicit program config**; the no-program-config paged API default is 32 at lines 114–121.
6. [Factory:122–171](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:122) computes `S=table.padded_shape[-1]*block_size=4352`, `St=136`, `block_size_t=1`, `PNHt=1`, `DHt=vDHt=8`, `Sk_chunk_t=8 or 16`. [Core assignment:202–216](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:202) assigns 32 cores/head for B1/KV1 on the available grid.
7. **The incompatible runtime extent appears at [rt_args_common.hpp:60–69](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:60):** `valid_seq_len=nearest_n(cur_pos+1,k_chunk_size)`. This is not bounded by the factory's `S`. The earliest model-side mismatch is supplying chunk512 while its allocator/table preparation still guarantees only 256-token alignment. The callee then permits and executes that mismatch.

Paged validation at [device operation:278–375](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:278) checks table presence, dtype/layout, batch, and K/V block agreement, but does not enforce a chunk-compatible table extent or bounds-check each page ID for this causal path. Lines 454–466 explicitly restrict the table-capacity divisibility check to non-causal mode; the unpaged path enforces sequence/chunk divisibility at lines 498–508. Thus acceptance is not proof of safe geometry. Sibling tests encode the missing preparation rule: [sdpa_test_utils.py:48–60](/home/mvasiljev/tt-metal/tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py:48) caps the selected chunk to a power-of-two divisor of cache sequence length; paged tests use that helper at lines 550–563. For 4352 it selects at most 256.

## Concrete work and buffer ledger

These values hold for **all 128 positions**, not just the first decode:

| Quantity | Passing 256 | Failing 512 |
|---|---:|---:|
| Logical attended tokens `p+1` | 4097–4224 | 4097–4224 |
| Runtime rounded tokens | 4352 | 4608 |
| Sequence tiles / page entries read | 136 | 144 |
| Number of chunks | 17 | 9 |
| Active cores out of 32 | 17 | 9 |
| Core 0's last chunk index | 16 | 8 |
| Core 0's virtual page indices | 128–135 | 128–143 |
| K tiles and V tiles per local chunk | 64 each | 128 each |
| Whole invocation K tiles and V tiles | 1088 each | 1152 each |
| Table indices beyond valid extent | none | 136–143 |

The reversed assignment at `rt_args_common.hpp:74–89` gives core `c` chunk `16-c` for c=0..16 or `8-c` for c=0..8. Every active core handles exactly one chunk. Core 0's chunk always contains at least one live token, so no active chunk is entirely causally masked. Cores outside those ranges return consistently in reader lines 199–201, compute lines 211–218, and writer lines 174–180. Both compute and writer filter reduction children by `child_id < k_num_chunks` (compute lines 220–239; writer lines 182–197). Root receives live subtrees at rounds corresponding to child IDs 1,2,4,8,16 versus 1,2,4,8.

The [reader:312–362](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:312) passes the entire chunk to both read helpers. [K helper:755–781 and V helper:837–861](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:755) iterate every sequence tile without a current-position or table-extent guard. The [shared address mapper:53–68](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:53) reads `page_table_ptr[virtual_block]` directly. Here `virtual_block=virtual_seq_tile_row`, and each entry `e` yields physical tile IDs `8*e + [0..7]`. Invalid entries therefore alter addresses before any numerical mask exists.

Reader reserve/read/barrier/push counts match the compute matmul's full K/V consumption; no truncated-tail CB count mismatch was found. The [factory:479–508,637–720](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:479) produces these per-core capacities:

| Buffer | 256 | 512 |
|---|---:|---:|
| K c1, double buffered | 128 tiles / 139264 B | 256 tiles / 278528 B |
| V c2, double buffered | 128 tiles / 139264 B | 256 tiles / 278528 B |
| QK c24 | 8 half tiles / 8192 B | 16 half tiles / 16384 B |
| Causal mask c3 | 8 half tiles / 8192 B | 16 half tiles / 16384 B |
| Tree c19, `31*(8+2)` half tiles | 317440 B | 317440 B |
| Other tiled CBs | 80896 B | 80896 B |
| Total, excluding raw position/table buffers | 693248 B | 988160 B |

Q is retained as 8 half tiles; PV produces 8 half tiles; each max/sum is one half tile. The legacy causal mask is retained for reuse, while the generated fixed-size QK block is consumed and popped by the PV path. Position delivery uses separate c8/c15 mailboxes for writer and compute. The substantial numerical input defect is the page-address source, not a proven reserve/push/pop imbalance.

## Blackhole table padding is not valid page storage

The failing log's Blackhole architecture matters:

- [Row-major layout:49–58,107–141](/home/mvasiljev/tt-metal/tt_metal/impl/tensor/spec/layout/page_config.cpp:49) gives an unsharded row default element alignment 1 and raw row size `136*4=544` B. [Host encoding:249–259](/home/mvasiljev/tt-metal/tt_metal/impl/tensor/tensor_impl.cpp:249) does not manufacture eight extra tensor elements when logical and physical tensor shapes already match.
- [Blackhole NoC alignment:374–394](/home/mvasiljev/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374) requires DRAM alignment 64. [Buffer:842–845](/home/mvasiljev/tt-metal/tt_metal/impl/buffers/buffer.cpp:842) rounds the raw page to 576 B. Factory line 237 uses this aligned size; lines 666–674 allocate c9 to exactly that stick size. The shared page-table reader at lines 71–88 copies the full stick into c9.
- Normal host dispatch distinguishes `data_size_to_copy=544` from `page_size_to_write=576` ([dispatch.cpp:109–110,599–615](/home/mvasiljev/tt-metal/tt_metal/impl/buffers/dispatch.cpp:109)). [DeviceCommand::add_data:907–911](/home/mvasiljev/tt-metal/tt_metal/impl/dispatch/device_command.cpp:907) copies the former bytes and advances by the latter size, without initializing the difference. The normal command uses a reserved hugepage region at dispatch lines 982–986. There is no API guarantee that the extra bytes encode zero or any owned physical page.

Thus offsets 544–575, interpreted as table entries 136–143, are padding bytes rather than supplied IDs. Their actual values are absent from the artifacts. Even if an allocation happens to supply zeros, that would merely alias physical page 0 and could hide the defect for finite K/V; it does not establish correctness. With an explicit 144-entry table, raw and aligned sizes both equal 576 B: the successful padding experiment defines those words **without changing the c9 allocation size**. Cache placement and other allocations still change.

On a 32-byte DRAM-alignment architecture a 544-byte c9 would instead end before these accesses. That is a portability hazard, not the reported Blackhole memory-layout claim.

## Masking, arithmetic, and symptom propagation

[Writer:254–256](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp:254) generates the causal mask; [mask helper:251–304](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:251) allows offsets 0 through `p-4096` inclusive in the final chunk. The mask is logically appropriate. It is not a memory-access bound.

Both fixed chunks take the non-fused mask path: [compute:403–454](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:403) first computes QK, then adds the mask. The shared `add_block_inplace` at `sdpa/device/kernels/compute/compute_common.hpp:636–662` uses arithmetic addition, not assignment/select. Row-max subtraction, exponentiation and sum follow at compute lines 487–507. PV at lines 512–527 multiplies **every** V tile in the chunk.

For finite K/V, masked future elements can contribute zero safely. Exceptional values from invalid addressing need not disappear: `+inf + -inf`, a pre-existing NaN in QK, or `0 * NaN/inf` in PV are possible contaminating operations. These are mechanisms, not observed values from this run. Target arithmetic and subsequent packing can alter the final manifestation. The report does not claim that every invalid page ID must produce NaN.

Any affected final-chunk partial is on core 0, directly entering the tree root's max/sum/output merge ([compute:593–696](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:593)). Model consumers then reshape/gate/project/reduce attention and run residual, normalization, and MLP (model lines 479,505–506,333–364). These provide routes to a degenerate whole decoder output. K/V updates precede SDPA and do not consume its output, explaining why cache correlations can remain high while output correlation fails. The same extra-page range is touched at every tested step, explaining immediate and persistent failure without an accumulation story.

The actual numeric config also argues against ordinary precision drift. [Factory:428–470,497–551](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:428) uses four-destination subblocks with FP32 accumulation, but BF16 intermediate/statistic CBs. QK has two versus four `1x4` subblocks; PV has two `1x4` output subblocks and reduction width 8 versus 16. No truncated division occurs. Half tiles deliberately force an approximate exp path and packer-ReLU underflow handling despite the false exposed exp setting ([shared compute:366–399,454](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:366)); this is shared by passing and failing cases. Scale `1/16` is exactly representable. The 144-page pass exercises the same chunk-512 arithmetic and active-core pattern.

## What the artifacts prove about completion and NaNs

[PCC helper:23–27](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_optimized_decoder.py:23) is bare FP64 `torch.corrcoef`. No finite-count, infinity-count, variance, or raw SDPA tensor dump is recorded. [Runner:127–133](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:127) asserts replica equality on each host read; lines 245–249 assert restored replay equality. Subject to the recorded runner executing normally with assertions enabled, literal NaN elements in those read outputs would fail these checks. Stable infinities or constant finite vectors remain compatible with the available evidence.

Before each measured decode, input copies are followed by `synchronize_device`, then `execute_trace(...,blocking=True)`, then host read ([runner:222–229](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:222)). There is no evidence that a host read outran completion. Internal asynchronous collectives are present in both policies, so this does not prove every internal race impossible. Successful replay equality establishes repeatability, not numerical correctness or defined page-table padding.

## Corrections that preserve logical lengths and page ownership

### Smallest safe model-owned correction for existing external tables

Resolve an **effective** supported chunk from the actual mapped row capacity, before dispatch, instead of passing the requested chunk unchecked. For the current block32 contract, with a row containing `W` valid page IDs, let `C=W*32`; choose the largest supported power-of-two chunk no greater than the requested chunk that divides `C`. For `C=4352`, requested512 resolves to256. If the table has 144 valid entries,512 remains512. Record requested and effective values in performance evidence.

The proof is simple: for every valid `0 <= p < C`, `ceil((p+1)/K_effective)*K_effective <= C` whenever `K_effective` divides `C`. This uses shape/capacity metadata, not a runtime host read of positions, so it can be prepared before capture and survives content-only page remapping. Require the pool owner to supply or validate logical capacity, per-row mapped capacity, valid positions, and page ownership; the current model does not retain or enforce this metadata. Do not count implicit tensor/transfer padding or invalid/sentinel entries as mapped capacity. Ragged callers must define valid mapped capacity for each row; merely knowing a rectangular tensor width is insufficient if some rows contain invalid IDs in that width.

This accepts arbitrary logical lengths and preserves all caller mappings. It also preserves explicit512 when the actual pool/table supports it. It is a safe fallback, not a performance validation of512 on the original capacity.

### To keep512: own both physical padding and its mapping

For a model-owned pool and table, maintain separate logical length/capacity and physical capacity. With page32 and the existing eight-ID alignment rule, round physical capacity to `ceil(L/A)*A`, where `A=lcm(256,requested_decode_chunk,other_required_read_alignments)`. For this case, `A=512`, physical capacity4608,144 pages. Initialize the additional K/V storage and provide explicit valid mappings for every rounded-read page. Preserve logical positions, causal bounds, maximum-context semantics and output slicing; never require user sequence lengths to be multiples of512.

The current [PagedState:72–75](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:72) contains only K/V. The caller supplies the page table and may remap it under a live trace (runner lines 188–205,250–255). Therefore **changing `allocate_state` alone is insufficient**, and padding only the test runner is an experiment, not a production correction.

An external-table adapter would need an explicitly owned padded table and additional pages supplied by the actual pool owner. Preserve every existing ID, map added virtual slots only to reserved zero-filled pages, and define device-side refresh/rebinding whenever the caller changes mappings. Keep those buffers alive through captured traces. Never append guessed sequential IDs, hijack another caller's pages, silently replace its mapping, or cache a stale padded copy. If ownership/refresh cannot be guaranteed, use the effective-chunk fallback.

### Callee-owned robust correction

SDPA can independently prevent undefined-address consumption: carry the mapped-table/physical-cache bounds to the reader, avoid page-table dereferences for fully masked tail tile rows, and synthesize zero K/V tiles while preserving full chunk CB counts. Guard live accesses as well. Preserve the partial final tile's causal mask. This supports arbitrary capacities without relying on caller over-allocation. A static conservative validation of chunk-compatible mapped capacity would at least reject this unsafe configuration; a properly bounded reader is more general. Truncating pushes without updating compute/writer counts would introduce a separate deadlock.

Do not use `cache_position_modulo` as an incidental padding workaround: it encodes circular-cache semantics, and using it to wrap arbitrary future reads does not establish owned, initialized padding for full attention.

## Ranked hypotheses and discriminating follow-ups

| Rank | Candidate and evidence | Unverified prerequisite / smallest useful discriminator |
|---|---|---|
| 1 | **Undefined page IDs from chunk-rounded tail beyond valid table.** Direct source contradiction; exactly separates256 and512 at every observed position; explicit144-page512 passes. | Actual eight padding words and first bad numerical tensor are unknown. Capture them or use an explicitly mapped/zero-filled144-page table while preserving original136 mappings and live K/V. |
| 2 | Width512, half-tile arithmetic, or nine-active-core reduction defect independent of the tail. | No direct violation found; same geometry passes at144 pages. Revisit only if512 still fails after all accessed pages are defined, using a standalone SDPA replay of the same Q/K/V. |
| 3 | Allocation-sensitive downstream corruption, including collective behavior. | Padding changes cache addresses/placement too. Compare finite/range statistics immediately after SDPA and after gate/projection/collective if ownership-preserving padding does not settle the issue. |

Minimal controls for the caller, **not executed by this investigation**:

1. Record `isfinite`, `isnan`, `isinf`, min/max and variance on both decoder outputs and on Q, SDPA output, gated attention and projected output at the first step. Preserve rank/head/index correspondence. This resolves NaN PCC versus tensor NaNs and localizes first corruption.
2. Retain original136 page IDs and K/V contents; append eight explicitly reserved zero pages/IDs without reshuffling existing ownership. Compare256 and512 using this same padded setup and logical4096+128 inputs. The existing144-page artifact already supplies a strong less-isolated version of this control.
3. Cover positions4095,4096,4223 and nearby non-divisible logical lengths. The predicted unsafe boundary is `round_up(p+1,512)>mapped_capacity`, not merely “long sequence.” At4095 both chunks read4096 tokens. Extend ownership-preserving remap/ragged checks to the padded/fallback path and verify subsequent live trace updates.
4. If a callee fix is chosen, verify K/V tile counts remain64/128 per chunk and only masked source reads are replaced by zero fills; keep a minimal page-tail regression at136 versus144 mapped entries. Separately test a requested chunk larger than a tiny mapped context.

## Other Potential Issues / demoted explanations

- **1024 L1 exhaustion is separate.** Its artifact explicitly fails static CB placement; it is not evidence that512 ran out of L1. A1024 read would also require160 mapped pages for these positions if resource admission succeeded. Padding alone does not cure its L1 failure.
- **Fully masked softmax / `-inf - -inf`:** the final active chunk always includes token4096, and empty cores return. This does not fit the current bounds.
- **Local softmax accumulation drift:** each participating core handles one local chunk for256 and512; the multi-local-chunk update is not exercised. Tree arithmetic remains, and succeeds for padded512.
- **Q-head count6 and GQA power-of-two assumptions:** local KV1 takes the MQA writer path, so the multi-KV head-gather concern does not explain this case.
- **BFP4/LoFi projection precision:** upstream operations and real weights are identical across the controlled pair, and cache/prefill PCC is unchanged. Treating catastrophic output degeneration as normal quantization variation is unsupported.
- **Implicit alignment zero fill:** not a supported page-mapping contract. It can make invalid inputs appear safe on some runs without correcting them.
- **Broad correctness gates:** NaN PCC should be rejected explicitly, and replay equality alone should not mark numerical correctness. This reporting gap does not cause the page-address defect.

## Review and verification

The first-draft diagnosis was checked against the concrete reader bounds, factory dimensions, target DRAM alignment, dispatch copy sizes, mask operation, numerical phases, CB counts, caller ownership, and source hashes. Three independent review passes confirmed the headline and corrected remaining precision issues: caller-supplied packer configuration is not a forwarded kernel flag, and logical-capacity/ownership metadata must be supplied rather than presumed to exist. Review also retained two essential qualifications: Blackhole accesses alignment padding rather than necessarily overrunning c9; the artifacts show NaN correlations rather than proven NaN tensor elements. Caller-generated padding evidence is separated from work performed here.

Cheap standard-library-only checks parsed the artifacts, verified the original policy/provenance contrast and same-policy padded512 control, instantiated every position4096–4223, and checked the effective-chunk bound for every valid position in4352- and4608-token capacities. No TTNN/model imports or hardware execution were used. The leading contract defect is established by source; the exact invalid-data-to-output manifestation and production correction remain for the caller's controlled tests.
