# AutoDebug: causal indexer scorer warmup stall

## Finding

**No concrete CB/compute/dataflow violation was established for the reported `Q=1, K=65536, position=0` invocation.** The inspected native zero-live path has balanced metadata exchange and no dot-tile producer/consumer dependency. Its SFPU arithmetic cannot execute in this invocation. An implementation patch presented as the stall fix would therefore be speculative.

**The available evidence does not establish that the native scorer launched.** The probe warms the full baseline scorer and its selector first, synchronizes, then warms the candidate and its selector, and only later prints a case. No intermediate phase markers or usable device/host stack were saved for the original run. The stall remains unlocalized.

A separate, concrete **validation defect was present in the initially inspected probe**: it recorded exact elementwise equality but did not require it to pass. The probe changed during report review to require that equality and add upload/warm/capture markers. Those changes were not made by this investigator. The current probe fixes the tolerance-based acceptance defect; neither snapshot supplies a stall diagnosis. A literal bitwise check still requires integer views of the FP32 data.

Investigation was source-only. No TTNN/device execution, reset, process termination, implementation edit, or AutoDebug relaunch was performed. Only this report was written.

## Observations versus interpretations

Direct artifact/source observations:

- [The saved log][log] identifies **Blackhole**, local chips 0–3, and successful four-device fabric initialization. It has no case output, exception, warmup marker, or blocked kernel PC. During review it acquired a final `Killed` line; that establishes termination text, not a particular stalled phase or timeout mechanism.
- At the final artifact check, [triage text][triage], its sibling `causal_q1_v1.log`, and [host stack][stack] were all **0 bytes**. They supply no stack/CB/waypoint evidence yet.
- [The probe's upload/warmup sequence][warmup] uploads keys, query, and initial positions, then executes `old()` before `new()`, with `synchronize_device(mesh)` after each complete graph. Both graphs include score selection. Initial positions are `arange(count)`, so the first `count=1` invocation uses FP32 `+0.0`. This ordering is unchanged between the initial and revised snapshots; current lines 78–104 add markers absent from the original saved log.
- [Trace capture and case output][capture] happen after those warmups. The `--starts` sweep is applied only during trace replay; it does not control initial warmup positions.

Requester observations retained as reported, rather than independently proven by these files: a warmup stall under a 600-second timeout, with no completed case. In particular, “the new native compute kernel is stalled” is an interpretation that these artifacts do not yet support. Host operation return generally means enqueue, whereas the explicit synchronization is a completion boundary; a wait there could belong to any preceding operation in that graph.

## Concrete native invocation and CB ledger

The host validates tiled interleaved DRAM geometry: query BF16 `[1,4,1,128]` padded to `[1,4,32,128]`; keys BF16 `[1,1,65536,128]`; positions FP32 `[1,1,1,1]` padded to `[1,1,32,32]`; output FP32 `[1,4,1,65536]` padded to `[1,4,32,65536]`. See [host validation and work assignment][host].

`kt=65536/32=2048`, `rows=1`, total jobs `kt*4=8192`. Let `N=min(grid.x*grid.y,8192)` (the log does not record the numeric grid). Core `i` receives reader arguments `[query_addr, keys_addr, positions_addr, i, N, 2048, 1]`, compute `[i,N,2048,1]`, and writer `[output_addr,positions_addr,i,N,2048,1]`. Every kernel visits exactly `job=i+m*N<8192`, with `head=job/2048` and `key_tile=job%2048`. [Generic op dispatch][spmd] applies the same program to all four replicated devices; this scorer has no interchip collective handshake.

| CB | Capacity / format | Producer and consumer at position 0 |
|---|---|---|
| 0 | 1 × 4096 B, FP32 descriptor; raw integer metadata | Reader reserves/pushes 1; UNPACK waits/pops 1. MATH/PACK receive the metadata through mailboxes. |
| 1 | 4 × 2048 B, BF16 | Reader reserves 4 as private query scratch; no push/pop or consumer is needed. |
| 2 | 8 × 2048 B, BF16 | Zero query-broadcast pushes and zero compute pops. |
| 3 | 8 × 2048 B, BF16 | Zero key-tile pushes and zero compute pops. |
| 4 | 2 × 4096 B, FP32 | Zero compute-result pushes and zero writer pops. |
| 5 | 1 × 4096 B, FP32 | Writer reserves 1 as private output scratch. |
| 6 | 1 × 4096 B, FP32 | Writer reserves 1 as private positions scratch. |

Capacities: [host CB descriptors][cbs]. Total allocation is **61,440 B per worker**. A reserve-only private scratch CB is valid: [dataflow `cb_reserve_back`][reserve] polls free capacity without advancing a pointer or publishing pages. Adding artificial pushes/pops to “balance” those scratch reservations is not a justified fix.

Exact zero-live sequence:

1. [Reader][reader] NoC-reads positions, completes the read barrier, returns `live[0]=0` immediately for zero bits, and writes raw integer zero at metadata word 0 before publishing CB0. Its `key_tile*32 < live[0]` condition is false for every job, so it never enters the key/query streaming loop.
2. [Compute][compute] waits for CB0 and reads its raw integer through `read_tile_value(0,0,0)`. It skips every job, then pops CB0. **No `tile_regs_acquire`, copy, MUL, ADD, row reduction, output reserve, pack, or CB2/3/4 wait runs.** Startup and `copy_init` do run.
3. [Writer][writer] independently derives `(0+1)/4=0`; it never waits for CB4. It zeroes all 1024 words of each assigned output tile, writes 4096 bytes, and waits for that write before reusing scratch. Across workers this writes exactly **8192 tiles = 32 MiB per chip**; all logical invalid cells and padded cells are overwritten.

[The actual `read_tile_value` implementation][metadata] performs a raw volatile `uint32_t` L1 load and sends that same value to both MATH and PACK. It does not interpret the word as FP32, unpack it numerically, or return zero only on UNPACK. [Blackhole mailbox reads/writes][mailbox] use FIFO status separately from the payload, so zero is a valid message. There is one write/read per directed channel in this `rows=1` case. Thus the Float32 metadata descriptor does **not** establish a type-conversion or thread-divergence bug in this checkout. A sibling comment claiming float descriptors misread raw words is not the executable contract.

Remaining possible waits on this zero path are startup, positions NoC reads, CB0 publication/mailbox delivery, and output NoC writes. [Startup math/pack synchronization][startup] waits for outstanding pack state before initializing the semaphore. No evidence shows that this particular wait is stuck, and the native zero path itself does not create an unmatched tile-register acquire/commit/release. [Firmware completion][firmware] drains Tensix and signals done; it does not require private scratch buffers to have consumers.

## Nonzero live paths and boundary checks

For each job, all reader/compute/writer decisions use `live[q]=(position[q]+1)//4`. For an active job, reader pushes exactly 4 key tiles to CB3; compute waits/pops those 4 after processing its active query rows. For each active query row, reader pushes 4 one-tile broadcasts to CB2; compute consumes all 4, then pushes one result tile to CB4; writer consumes one. The indexed key reads `w=0..3` follow `cb_wait_front(3,4)` and fit at either half of the eight-tile ring. CB2 and CB4 advance one page at a time. No short-live-tail mismatch changes these transaction sizes. Writer copies only `k` satisfying `key_tile*32+k<live[q]` and leaves the rest zero.

A cheap, standalone Python integer calculation (stdlib only, no TTNN import) checked position-bit decoding and work counts for 40 combinations of `Q∈{1,3,17,32}` and starts `{0,2,3,127,128,2047,2048,32767,205504,262112}`. It also checked output-job coverage for illustrative worker counts 1, 64, and 140; these are arithmetic controls, **not measured device grids or passing hardware cases**.

| Q / start | Live-count range | Active jobs | CB2 pushes = pops | CB3 pushes = pops | CB4 pushes = pops |
|---|---:|---:|---:|---:|---:|
| 1 / 0 | 0 | 0 | 0 | 0 | 0 |
| 1 / 3 | 1 | 4 | 16 | 16 | 4 |
| 3 / 0 | 0 | 0 | 0 | 0 | 0 |
| 32 / 0 | 0–8 | 4 | 464 | 16 | 116 |
| 32 / 3 | 1–8 | 4 | 512 | 16 | 128 |

These are source/count checks, not proof of hardware liveness. There is no completed native passing/failing contrast in the supplied artifacts from which to infer a shape-dependent deadlock.

## Exact selected arithmetic

[Reference `fused_indexer_scores`][reference] converts both inputs to FP32, reshapes query to `[4,1,1,128]`, multiplies by FP32 keys `[1,1,65536,128]`, and requests `sum(dim=-1, keepdim=True, fast_and_approximate_mode=False)` with FP32 destination enabled. On Blackhole:

- FP32×FP32 selects [binary SFPU multiplication][binary-select]. The original operand order selects `ROW_A`; the Blackhole FP32 path uses [software row broadcast][binary-bcast], followed by `eltwise_binary_sfpu_no_bcast.cpp`, not FPU/TF32 multiplication or a TF32 dot. The native `mul_binary_tile` selects the same Blackhole SFPU multiply primitive. BF16 operands widened into FP32 have exact ordinary finite products when representable.
- [Accurate sum selection][sum-select] requires FP32 input, non-Quasar architecture, `fp32_dest_acc_en=True`, and fast/approximate mode disabled. Those conditions hold. `ReduceMultiCoreWProgramFactory` selects tiled `reduce.cpp` with accurate SFPU mode.
- The width reduction first folds feature tiles **sequentially**, then reduces within the remaining tile. If `p[w,c]` is the separately computed product at feature `32*w+c`, both paths compute `a[c]=((p[0,c]+p[1,c])+p[2,c])+p[3,c]` with separate FP32 ADDs, followed by `sfpu_reduce<SUM,Float32,REDUCE_ROW>(dst,1,1)`. See [reference seed/fold][fold], [reference width loop][width-loop], and [native compute lines 27–43][compute]. It is not the older half-width 128-element tree or fused multiply-add accumulation.
- The exact intra-tile implementation is [Blackhole `perform_reduce_row_sum_avg`][row-reduce] → `perform_reduce_row_tile` → `perform_reduce_row_face_pair`. It loads even/odd columns from each left/right face and first forms `u[j]=a[2*j]+(a[2*j+1]+(a[16+2*j]+a[17+2*j]))`, then performs the merged masked-swap/rotate-by-1/2/4 FP32 butterfly. The recorded fold and inline C/D folds preserve the same association; both scorer paths call this very implementation. Its [output contract][row-contract] guarantees column 0 of each row and leaves other columns unspecified. Writer correctly selects `dots[face(k,0)]`; it does not rely on those unspecified columns or a broadcast result.
- Initialization ownership is coherent: [binary SFPU init][sfpu-init] uses address modifier 7, separate from A2D modifiers 0/2; native reinitializes MUL/ADD before their uses and reduction immediately before reduction. Reduce init restores masks/replay state. No demonstrated need exists to re-run `copy_init` for every copy or to reconfigure between CB2/CB3, which have identical BF16 formats.

Both public scorers then use the identical TTNN expression `multiply(sum(relu(dots),dim=1,keepdim=True),128**-0.5)`. For this FP32 input, [generic reduction excludes `fast_reduce_nc`][head-sum]: [it transposes the head axis to H, reduces, and transposes back][head-transpose], using accurate SFPU reduction. The omitted compute config takes [the Blackhole FP32 defaults][reduce-defaults]. The candidate's explicit HiFi4 versus the reference owner's HiFi2 does not change the selected SFPU MUL/ADD schedule. These source correspondences are not a substitute for the requested bitwise device comparison.

## Baseline and downstream completion audit

The baseline computes all allocated keys even at position 0; it has no causal bound in its score graph. Before any native invocation, it can therefore execute a **128 MiB FP32 product, 32,768 product tiles**, followed by **8192 width-reduced tiles** and the reshape/score tail.

For a core assigned `r` tile rows, [the width-sum factory][width-factory] gives reader count `4*r`, compute `Ht=r,Wt=4,NC=1`, and writer count `r`. Its input depth is eight, reader batches four, compute consumes one tile at a time, and output depth two carries one tile per row. Uneven worker partitioning does not make a partial four-tile input batch: work splits by complete rows. No mismatch was found here.

The final accurate head sum has physical `Ht=1`, full width `Wt=2048`; its compute column chunks can be three tiles. [The selected H reader][h-reader] explicitly disables batch mode when that chunk differs from host batch four, then uses single-tile transactions. Thus an apparent three-page/eight-page ring-wrap mismatch is not a reachable violation on this branch.

At position 0, [the common selector][selector] derives `complete=0`, skips all score reads and heap work, sets the uncompressed tail slot for token 0, then writes the 2080-element result. Its 130 aligned 64-byte writes end within the 32 KiB private scratch allocation. It has no dot-CB wait. This clears a concrete zero-count heap/CB hypothesis, but does not prove its NoC transfers completed in the reported run.

The width-sum-to-dots reshape changes tiled width and is not necessarily a metadata-only view. If phase evidence points into the baseline, its casts, broadcast product, width sum, reshape, and score tail remain separate localization targets; the absence of a width-sum CB contradiction does not clear that entire graph.

## Minimum intervention and validation defect

**No stall-fix implementation change is currently justified.** The current probe now has `INPUT_UPLOAD_BEGIN`, `KEY_UPLOAD_DONE`, `ROW_INPUT_BEGIN`, `WARM_BEGIN`, `WARM_DONE`, and `CAPTURE_BEGIN` markers. `WARM_DONE` follows synchronization and is completion evidence if reached; upload-call return markers do not independently establish queued upload completion. First inspect the marked run. If it stops inside a warmup, the smallest further diagnostic change is flushed markers and synchronization separating baseline scorer/selector, native dots, candidate score tail, and candidate selector, before trace capture. Record actual grid/core count and runtime inputs. If the baseline scorer is implicated, subdivide casts/product/width-sum/reshape/head-sum. This is a suggested follow-up, not a device run performed here.

A nonempty preserved stack/triage capture identifying the first blocked kernel, PC/waypoint, CB counters, or NoC barrier would discriminate among the remaining waits. Capture evidence from the already-failed run before any recovery; no reset is proposed by this investigation. Phase markers should distinguish reaching a call from completing its queued work.

**Historical tolerance-gate bug, now corrected, independent of the stall:** the initial snapshot (SHA256 `8d222bfea739eb120b2e72a0247b4a2b2f3ed58d85a96bac1b49696f94b7cbd5`, original lines 131–155) computed `exact=torch.equal(a,b)` and saved it as `live_scores_exact`, but accepted only `max_abs<1e-4 and ids_equal`. A difference of, for example, `1e-5` with unchanged IDs could pass. [Current lines 137–161][gate] correct that gate with `passed=exact and ids_equal`, retaining error magnitude as diagnostic data. `torch.equal` on floating tensors checks exact elementwise values, not bits: opposite signed zeros compare equal. For the literal bitwise contract, compare contiguous FP32 values through `view(torch.int32)` as well. Also compare valid per-head dots before ReLU/head reduction; final scores and IDs can hide negative-dot differences and compensating head differences. The `complete=0` score comparison is empty and therefore vacuous; explicitly verify the native all-zero output in that case. None of these acceptance changes explains or repairs an earlier warmup stall.

## Report review / rejected headline candidates

| Candidate | Code-level adjudication | Status |
|---|---|---|
| Zero-live compute waits for missing key/query/result tiles | All relevant waits are inside branches skipped when `live=0`; writer also skips its result wait. | Refuted for stated invocation. |
| `read_tile_value` returns different values across TRISCs, converts FP32 metadata, or cannot transport zero | Direct raw-word load and balanced mailbox broadcast; occupancy is independent of payload. | Refuted in inspected implementation. |
| Query/output/positions scratch must be pushed and consumed | Reserve only polls capacity; scratch owners directly read/write reserved storage. | Not a violation. |
| Candidate row-reduction layout/association differs from accurate sum | Same sequential tile fold and same Blackhole row primitive; writer reads guaranteed column 0. | Not established. |
| SFPU init clobbers the copy configuration | Address modifiers are explicitly separated; reduction is reinitialized before use. | Not established. |
| Baseline width reader leaves a final short four-tile batch | Factory partitions complete width rows, so per-core count is `4*r`. | Refuted. |
| Final head-sum reader batches three into an eight-page ring | Selected reader disables batching for this mismatch. | Refuted. |
| Tolerance gate accepts unequal numerical values | True in the initial snapshot; current `passed=exact and ids_equal` corrects it. Literal bit comparison remains a test gap. | Historical tolerance bug, now fixed; unrelated to hang. |
| Native startup or NoC transfer hangs due to unobserved prior state | Possible wait sites exist, but no phase/PC/state artifact selects one. | Unresolved; no patch supported. |

After drafting, headline claims were rechecked against the concrete zero-work branch, helper implementations, baseline-before-candidate ordering, and asynchronous completion boundaries. No active-SFPU hypothesis was promoted to explain a zero-live native invocation. No missing log correspondence was invented to assign the stall to a specific graph.

## Provenance

Prepared checkout HEAD: `de432e04c21c3ea53ffe13d9c6e78a96f7acb704`. The five focus/probe files are untracked additions in this checkout. The runtime log has no source hashes, so its exact correspondence to these bytes is not independently pinned. The four native files retained their hashes throughout inspection; the probe changed during review. The table records the final inspected probe. Initial-snapshot findings are explicitly labeled above and are not attributed to the revised probe.

| Inspected file | SHA256 |
|---|---|
| `tt/causal_indexer_scores.py` | `9be09fbbc48aba1e080f4e85f4455bfe0696b7178c8c5ffeafabfd6b2a7a04a3` |
| `tt/kernels/causal_score_compute.cpp` | `bcdbe4456db20203d59f452b62ed08a384bf2e4b9e47786d39766f3748dacd19` |
| `tt/kernels/causal_score_reader.cpp` | `6e9051ca951894e23c4b7a86618ca206c424c5b709af9060bf32191cee47a46d` |
| `tt/kernels/causal_score_writer.cpp` | `425481f7c410bc35602bc3dc0c26c14957be279523f7849a519387ef1d3d32bf` |
| `tests/probe_causal_indexer_scores.py` | `4493e5af38024a8a04af60bf6b82d437701a6c8871819de76c02b9ea0a4b526a` |

[log]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/causal_q1_v1.log
[triage]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/triage/causal_q1_v1.txt
[stack]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/causal_q1_v1_host_stack.log
[warmup]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_causal_indexer_scores.py:78
[capture]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_causal_indexer_scores.py:102
[host]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/causal_indexer_scores.py:39
[spmd]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/generic/generic_op.cpp:12
[cbs]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/causal_indexer_scores.py:131
[reader]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/kernels/causal_score_reader.cpp:8
[compute]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/kernels/causal_score_compute.cpp:11
[writer]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/kernels/causal_score_writer.cpp:15
[reserve]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h:406
[metadata]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/inc/api/compute/cb_api.h:202
[mailbox]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h:547
[startup]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:152
[firmware]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc:219
[reference]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:1170
[binary-select]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:63
[binary-bcast]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:347
[sum-select]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:198
[fold]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:108
[width-loop]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:520
[row-reduce]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h:1152
[row-contract]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h:1981
[sfpu-init]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu.h:23
[head-sum]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp:500
[head-transpose]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp:209
[reduce-defaults]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:148
[width-factory]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp:726
[h-reader]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:71
[selector]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:1242
[gate]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_causal_indexer_scores.py:137
