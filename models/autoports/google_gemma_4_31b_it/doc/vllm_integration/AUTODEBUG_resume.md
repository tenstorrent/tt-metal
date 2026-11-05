# Stage 09 prefill resume investigation

**Finding: resumed prefill changes SDPA's numerical partition; an aligned cache-indexing defect was not found.** This is the leading source-backed mechanism, not a proven explanation of the full error magnitude. Physical-row-dependent prefix production remains an independent candidate requiring a small controlled test. Neither “head_bfp4 precision” nor a fixed-1024 bucket is an established diagnosis/fix.

Inspection only; no hardware commands or source edits. Read the run contract and evaluation inputs. Compared the current autoport against HEAD `9df04b7d`; inspected generic TTNN lowering, not removed/stock Gemma implementations. Only this report was written.

## Observations and revision boundary

`readiness_vllm/prefill_resume_before_strict.json` and `prefill_resume_fabric.log` record all 60 layers, natural text, B1, context allocation 16384, with these results:

| Start | L=2560 PCC mean/min; top1 | L=16384 PCC mean/min; top1 |
|---|---|---|
| 32 | .9514/.5520; .8521 | .9514/.5520; .8482 |
| 64 | .9161/.4700; .8132 | .9160/.4700; .8093 |
| 132 | .9005/.6606; .8249 | .9005/.6606; .8210 |
| 1024, 1029 | approximately 1/1; 1 | approximately 1/1; 1 |
| L−64 | .9926/.9646; .9531 | .9465/.5983; .6875 |
| L−1 | .9963/.9963; 1 | .9943/.9943; 0 |

The runner compares TT fresh versus TT resumed, not HF. It collects the first 256 resumed positions plus the last position; negative controls fail as intended. It does not save per-position errors or intermediate tensors. Thus the aggregate minimum cannot be assigned to a particular token/layer.

The continuation method is uncommitted and absent from HEAD. The JSON's commit is not a snapshot of the executed dirty tree. Current `generator.py:256–295` teacher-forces a leading partial page, whereas the supplied evidence predates that repair. Current 132 executes decode positions132–159, then prefill160; 1029 executes1029–1055, then1056. **Do not use the old 1029 pass as evidence that this new path passes.** The optimized/multichip decoder numerical paths are unchanged in the diff; aligned starts32,64,1024,2496,16320 avoid the new leading-partial branch. Historical unaligned execution cannot be fully reconstructed from HEAD/current diff alone.

## Leading mechanism: partition-dependent attention

Actual dispatch is `model.py:49` → `MultichipDecoder` → `OptimizedDecoder`, not `FusedDecoder`.

`optimized_decoder.py:625–637` selects Q/K chunks32/32 when **attention-view** offset is not divisible by128, otherwise128/128. `prefill.py:67–77` rebases sliding attention to offset1024 after the first window; global attention retains the absolute offset.

| Aligned resume | Sliding Q/K | Global Q/K | Fresh reference |
|---|---|---|---|
| 32,64 | 32/32 | 32/32 | 128/128 |
| 1024 | 128/128 | 128/128 | 128/128, same chunk boundary |
| 2496,16320 | 128/128 | 32/32 | 128/128 |

This survives lowering: `transformer/sdpa/sdpa.cpp:145–181` forwards the tensor-offset causal call without swapping Q/K/V. `device/sdpa_program_factory.cpp:557–638` takes the supplied chunks unchanged and pads sequence length; **prefix32 at start0 still uses Q/K128**, with valid one Q tile and padded four, rather than silently selecting32.

The prefill config disables FP32 destination accumulation (`multichip_decoder.py:235–236`). The factory selects streaming compute and BF16 QK/sum/intermediate formats (`sdpa_program_factory.cpp:81,766,1108–1112`). Omitted `exp_approx_mode` defaults true (`:105–112`), separately from `math_approx_mode=False`. `kernels/compute/compute_streaming.hpp:672–712,1932–1952` implements max correction and accumulated output/sum updates. K32 versus K128 changes the number and boundaries of separately rounded softmax updates. That is an actual calculation difference, not just a different core assignment.

Even unchanged sliding Q/K128 does not guarantee identical partitions: at16320 the resumed sliding view starts at15296, whereas the fresh final chunk's view starts at14336. Their K partition origins differ by960 = 7×128+64. Fixed32 would align both origins, since all paged-prefill starts are multiples of32. This makes a fixed32 control relevant to late failures too.

This mechanism predicts divergence for32/64 and late resumes, with an exact aligned1024 contrast. It does **not** prove PCC minima near.47, nor fully explain the historical132/1029 results or last-token cases. A hardware control is needed before calling it the root cause or changing precision.

## Earlier possible divergence: prefix production

Current prefix execution ends at the requested start (`generator.py:586–588`), rounding its last physical chunk to32 rows (`:282–285`). Prefix32/64 therefore executes different shapes than the first1024 reference rows; prefix16320 ends with960 rows at15360 instead of1024.

The earliest changed configuration is input RMSNorm (`multichip_decoder.py:315–344`): physical32/64/160/1024 gives `(shard height, grid y)` of `(32,1)/(32,2)/(32,5)/(128,8)`. However, width reduction remains8×672, `block_w=21`, `subblock_w=3`; the sharded norm kernel reduces rows independently. This is **not a proven norm bug**.

Dense prefill likewise retains explicit M/K/N blocks4/8/16 and subblock1×8 (`optimized_decoder.py:254–286`); only M work allocation changes. Current lowering uses `minimal_matmul_program_descriptor.cpp` and its `compute_metal2.cpp`, not the legacy factory's compute file. No operand-format, reduction-count, or queue contradiction was established. Fixed1024 may help, but its rationale must be tested before blaming dense kernels.

The selected head cannot explain the split by itself: this readiness driver selects32 hidden rows before terminal projection (`generator.py:574–583`, `prefill.py:109–118`) on both paths. Head quantization could amplify upstream differences; it is not evidence of their source.

## Cache mapping and the proposed fixed1024 control

For aligned calls, fill uses absolute pages `[start/32,(start+size)/32)` (`prefill.py:64`). Sliding attention uses base `max(0,start−1024)`, offset `min(start,1024)`, over a2048-token ring (`model.py:225–228`). A cheap host arithmetic check confirmed no overlap between required preceding-window pages and current writes for32,64,128,1024,2496,16320.

Concrete late case: prefix final fill pages480–509 → ring32–61; resume64 fills510–511 → ring62–63. Attention base15296/page478 starts at ring30, preserving history478–509. Current absolute-table indexing and HEAD's modulo64 indexing are equivalent for these generator-owned cyclic tables; scheduler ownership differs, but this repro owns its cache. No missing prefix fill or offset-unit mismatch was found.

`paged_fill_cache` supports INT32 `−1`: `writer_fill_cache_interleaved.cpp:271–275` drains each skipped row's Wt tiles; successful writes barrier then pop (`:286–287`). Thus a32-entry fill table with real entries through `ceil(valid/32)` and−1 thereafter is supported. **Never put−1 in the SDPA attention table:** its address helper multiplies physical block IDs without this sentinel handling (`sdpa/device/kernels/dataflow/dataflow_common.hpp:65–68`).

Simply forcing `size=1024` is incomplete: current fill slicing can return fewer than32 entries near capacity; attention coverage validation rejects `offset+1024` beyond its table (`optimized_decoder.py:599`); padded RoPE positions may exceed the advertised table (`prefill.py:61`); and `hidden()` currently passes `valid_tokens=size` (`:96`). A general implementation must preserve logical validity and provide safe physical attention/position padding. Keep this control away from capacity boundaries initially.

Execution order is input copies → QKV → fill → attention → downstream layers → logits readback. Readback via `_host_logits`/`ttnn.to_torch` observes completion; merely enqueueing fill is not a correctness check. The inspected path does not replay decode traces during aligned continuation, and the logs show numerical failure rather than a hang. No specific asynchronous lifetime defect was established.

## Smallest discriminating follow-up (not executed)

1. **Prefix producer probe, one real sliding layer0:** same natural first32/64 tokens at physical32/64 versus1024, same selected policy and start0. Compare corresponding input-norm rows, raw QKV, normalized/rotated QKV, filled cache pages, then attention and layer contribution. Hold Q/K128 as the code already does at start0. Stop at the first divergence. Also compare128 versus1024 to separate short padded-Q handling from row geometry. Do not start by loading60 layers.
2. **Attention consumer probe with identical supplied Q/K/V and cache:** compare the same absolute query rows32–63 using fresh Q/K128 against resumed Q/K32, then run **both** paths with Q/K32. Match visible keys, absolute positions, dtype, and page ownership; compare raw attention before the output projection. This isolates attention partitioning from prefix production. For the late contrast, repeat on global layer5 with16320 offset, then sliding layer0 with the two rebased origins.
3. If these establish which stage diverges, test the full selected model with separate arms: fixed physical1024 only; fixed Q/K32 only; both. Use L2560/start32 first, include start128 as a partition-preserving control and1024 as the known pass, then16320 and repaired unaligned cases. Keep gates unchanged. A combined-arm pass alone cannot identify the cause.

Claims were rechecked after drafting: cache corruption, a proven dense/norm defect, historical unaligned behavior, and an already-correct fixed1024 patch are deliberately **not** headline findings. The exact first divergent value remains unmeasured.
