# AutoDebug: full-attention decode near 262144 tokens

Date: 2026-10-23. Checkout HEAD: `0611d6e393882ce28e811b023b5f7d23bba1aced`.

## Finding and confidence

**For the reported baseline that omitted `program_config`, the leading diagnosis is loss of precision in decode's serial BF16 online-softmax state: 32 tokens per chunk and one core per KV head execute 8192 chunks, with 8191 rounded online corrections per head.** The caller's `fp32_dest_acc_en=True` does not make the running state FP32. In contrast, the passing chunked-prefill implementation stores its softmax sums and QK intermediates in FP32 with that flag.

This is a concrete source-level precision bottleneck and a strong explanation to test, **not a hardware-confirmed root cause of the reported .690/.609 PCCs**. No exported failing attention intermediates or controlled program-configuration comparison are available. Static inspection found no max-context position truncation, out-of-bounds page index, unsupported 32 KiB table transfer, or producer/consumer count mismatch in the selected path.

The smallest discriminating intervention is an explicit decode program configuration with `k_chunk_size=512`, retaining `max_cores_per_head_batch=1` and the current exponent mode. This reduces serial chunks to 512, essentially the 513 chunks in the passing shorter case, while preserving context, pages, position, model math and inputs. It is a proposed diagnostic, not a verified workaround.

This investigation only read source/evidence and ran small host-only scalar arithmetic checks. It did not import TTNN, run a model or hardware, reset devices, or edit implementation. Three xhigh subagents separately audited indexing, compute/buffer behavior, and the model/reference chain. Installed HF source was read through `docker exec qwen38-functional-build`; unrelated Qwen TT implementations were not inspected. Existing diagnosis reports were preserved.

**Revision boundary:** the shared workspace changed during this investigation, outside this investigator's edits. The decoder now explicitly requests k256/core16/exp-false, and the runner has added auditing/provenance instrumentation. The detailed failing-case ledgers below concern the earlier no-config call observed at the start of inspection. The newer override is not validated by this report. See the source-version note below.

## Evidence: observations versus interpretation

The artifacts under `doc/functional_decoder/` establish:

| Run | Prefill final-128 PCC | First decode PCC | Changed-input second decode PCC | Other observation |
|---|---:|---:|---:|---|
| `full_16385_tensor_aligned.json` | .9999552965 | .9998331070 | .9998925328 | Three replays bitwise identical |
| `full_262142_decode_max.json` | .9963822961 | .6901835799 | .6092750430 | Three replays bitwise identical |
| `full_262144.json` | .9964427948 | Not run | Not run | Prefill only |

`full_262142_decode_max.log:22-24` additionally records eager decode PCC .6901835799, matching traced decode. Its final failure is the accuracy assertion, not a hang or allocator failure. The log identifies Blackhole. `work_log.md:21-23` records an 11x10 compute grid and the use of `WormholeComputeKernelConfig` on this checkout's Blackhole API.

Important limits on the interpretation:

- These are **complete-layer output** correlations after attention gating, output projection, residual addition and MLP. They do not identify the first divergent operator or measure attention-output magnitude error.
- The second decode changes `xd` to independently generated `xd2`, position, RoPE, and valid cache extent. Its lower PCC is not a same-query boundary experiment.
- Different sequence-length runs also generate different decode inputs and page permutations. They constrain a length-dependent explanation but are not a frozen-input configuration A/B test.
- Prefill compares 128 output tokens while using the entire valid prefix. It does not establish all returned outputs or every cached value independently.
- The work log records the checkout/build, but these three JSON artifacts have no `provenance` key; neither the exact Python producer snapshot nor loaded-library hash is established by them. The logs also lack effective SDPA argument dumps. The lowered values below combine the baseline Python call inspected at investigation start with the unchanged tracked C++ source, not runtime telemetry. Later Python edits are separated explicitly below.

## Source map

Paths below are relative to the repository unless explicitly identified as installed HF paths. Abbreviations keep subsequent line references compact.

| Name | Path |
|---|---|
| Model | `models/autoports/qwen_qwen3_8_27b/tt/functional_decoder.py` |
| Runner | `models/autoports/qwen_qwen3_8_27b/tests/run_decoder.py` |
| Wrapper | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp` |
| Decode factory | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` |
| Decode compute | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` |
| Decode reader/writer | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/{reader_decode_all,writer_decode_all}.cpp` |
| Decode dataflow helpers | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` |
| Work planner | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp` |
| Shared compute | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` |
| Shared dataflow | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp` |
| Prefill factory | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` |

### Source-version note

At the first read, Model:202-205 called paged decode without `program_config`; Runner was the shorter variant without the new auditing/provenance instrumentation. During review, Model:205-212 acquired an explicit configuration with `k_chunk_size=256`, `max_cores_per_head_batch=16`, `exp_approx_mode=False`, and Runner acquired new instrumentation. No investigation agent made those implementation edits. The newer configuration would derive 1024 total chunks, 64 local chunks/core on 16 cores/head, and four tree rounds at the long positions. It changes three factors at once, so a future pass with it would be useful repair evidence but would not distinguish their individual contributions.

Model and Runner line references below, except the explicitly identified original no-config call, use the final-review snapshots:

| Source | SHA-256 at final review |
|---|---|
| Model | `e843748bd41908b43565470155d21c784669f28cef42d2976361a4193ee193db` |
| Runner | `7da30c7c187fb690bb7c09fb06595ce593dd2f0492e1f1d922018b1058430dd2` |
| Wrapper | `6aa630bdf0c2db2731c9fa3ea501b10c9cffe99002f20fde96d83be9faeac31f` |
| Decode factory | `c53d7dd356dc30c13a2622c2f1e5f4c1f8850256c79971fef507f816387e0c96` |

These hashes identify inspected files, not the historical binary or Python files that produced the reported results. Relevant B1 reference/cache/trace lifecycle calculations remained equivalent across the two inspected Runner versions.

## 1. Effective baseline configuration and source-level precision difference

Model:182-205 produces BF16 Q/K with full-head normalization and partial RoPE, and BF16 projected V. Q reaches decode as tiled, DRAM-interleaved `[1,1,24,256]`, padded to `[1,1,32,256]`. K/V are tiled, DRAM-interleaved paged caches `[8192,4,32,256]`. Only the single-token K/V update inputs are height-sharded; the decode Q and cache remain interleaved.

The baseline call passed scale `256**-0.5 = 0.0625` and the explicit compute configuration at Model:52-55: HiFi4, `math_approx_mode=False`, FP32 destination accumulation, `packer_l1_acc=True`. It omitted `program_config`; the newer override described above must not be used when reconstructing that baseline.

Wrapper:114-121 creates a configuration with full device grid, `q_chunk_size=32`, `k_chunk_size=32`, and **`max_cores_per_head_batch=1`**. Wrapper:133-143 consequently takes the fixed-chunk path; this is not the dynamic-chunk path. Decode factory:201-216 assigns four active cores, one per KV head, regardless of the remaining available cores. Each computes the Q-head tile against its own KV head; the writer selects the corresponding six logical Q heads.

The compute flag is honored for DST arithmetic (Decode factory:946-950), but the factory independently hardcodes both `im_df` and `stats_df` to `Float16_b` (BF16), lines 507-508. Lines 694-712 use those formats for QK, output accumulators, maxima, sums, correction factors and reduction outputs. No public compute flag makes these buffers FP32 in this path.

| Property | Failing decode | Passing maximum-context prefill |
|---|---|---|
| Model compute request | HiFi4, FP32 DST | Same |
| Default K chunk | 32 tokens | 32 tokens |
| QK intermediate | BF16 | FP32 |
| Running sum | BF16 | FP32 |
| Output accumulator storage | BF16 | BF16 |
| Kernel | `sdpa_flash_decode.cpp` | Regular SDPA compute path |

Prefill factory:176-177,1108-1112,1216-1223 explicitly derives QK and sum formats from `fp32_dest_acc_en`. Lines 557-560 establish its default chunk sizes. This is a material precision difference despite the identical model-level compute configuration; successful prefill does not exercise decode's BF16 denominator recurrence.

The independent exponent setting also defaults to `exp_approx_mode=True` (Decode factory:221-223,896). `math_approx_mode=False` does not override it. Preserve this setting during the first chunk-size A/B comparison so two variables do not change together.

## 2. Why the serial BF16 recurrence is a substantive candidate

Decode compute:487-507 calculates the row maximum, exponentiated scores and a chunk sum. Lines 537-574 then implement, conceptually:

```text
m_i = max(m_previous, max(raw_QK_i))
alpha = exp((m_previous - m_i) * 0.0625)
l_i = sum(exp((raw_QK_i - m_i) * 0.0625)) + alpha * l_previous
o_i = exp((raw_QK_i - m_i) * 0.0625) @ V_i + alpha * o_previous
output = o_final / l_final
```

The implementation packs intermediates back to BF16 after the multiply and add, not only once after the final division. `cb_sum_1/cb_sum_2` are c29/c30; `cb_out_accumulate_im` is c26. Shared compute:636-662 packs every `add_block_inplace` result into its destination CB, and lines 697-718 similarly pack the multiplication result. Decode compute:687-696 takes the reciprocal of the already-rounded sum and normalizes the already-rounded numerator. FP32 arithmetic within a tile cannot recover contributions discarded at earlier packs.

A cheap host-only scalar check demonstrates the magnitude of the representational problem. For equal valid logits, the ideal correction factor is one and each full chunk adds 32 to the denominator. With BF16 round-to-nearest-even, the sum reaches 8192 after 256 chunks; `8192 + 32` is a half-ULP tie that rounds back to 8192. Subsequent additions of at most 32 cannot advance it. Default RNE is documented in `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h:150-160`.

| Actual decode position used in scalar check | Valid keys | Chunks | BF16 denominator | Exact denominator |
|---|---:|---:|---:|---:|
| 16385 | 16386 | 513 | 8192 | 16386 |
| 262142 | 262143 | 8192 | 8192 | 262143 |
| 262143 | 262144 | 8192 | 8192 | 262144 |

The check used Python's standard library, no model/tensor runtime:

```python
import struct

def bf16_rne(x):
    bits = struct.unpack('>I', struct.pack('>f', x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('>f', struct.pack('>I', bits))[0]

for pos in (16385, 262142, 262143):
    total = 0.0
    for start in range(0, pos + 1, 32):
        total = bf16_rne(total + min(32, pos + 1 - start))
    print(pos, total, pos + 1)
```

**This proves a numerical failure mode, not that the real attention rows have equal logits or that their sums reach this exact plateau.** The toy also has error at 16385, where the real model passes. The observed distinction therefore cannot be explained as a universal hard threshold at 262144. Real logits determine how diffuse attention is, when maxima change, and how much each chunk contributes. Inspecting actual per-head sums/norms is needed to establish the causal chain quantitatively.

A separate subagent's abstract signed Gaussian numerator-sum check gave PCC about .99968 after 513 chunks and .99483 after 8192 chunks. It did not reproduce .69. Thus ordinary unbiased numerator rounding alone is insufficient evidence. Denominator distortion, head-dependent gains, correlated value vectors and the downstream residual/MLP remain relevant. A common scale error can leave raw attention PCC deceptively high while changing the full layer after residual addition and nonlinear processing. Measure norm ratios and relative error as well as PCC.

### Match to the reported observations

- **Length dependence:** serial corrections increase from 512 to 8191 with unchanged storage precision. This predicts greater error for diffuse long-context attention, without requiring a page-index limit.
- **Prefill versus decode:** prefill's sum and QK buffers are FP32. It avoids the particular BF16 denominator accumulation despite visiting the long prefix.
- **Eager and trace agreement:** deterministic arithmetic loss occurs in both, consistent with the observed identical first-decode PCC and repeatability.
- **Both final positions:** they execute the same 8192 chunks; only the last causal column and the new query/state differ. Both can be inaccurate without an off-by-one error.
- **Remaining uncertainty:** no attention-output norm, per-head error, sum statistic, or same-input chunk-size control exists in the supplied artifacts. The exact PCCs are unexplained quantitatively.

## 3. Concrete shape, page and workload audit

Runner:152 rounds `(seq+128)` up to 256 then caps capacity at 262144. Model:111-115 allocates eight 32-token pages per 256-token capacity unit. Runner:182-184 uploads one INT32 row containing a permutation of all physical page IDs and an INT32 position vector `[B]`.

| Derived quantity | Short run, first/second decode | Long run, first/second decode |
|---|---|---|
| Prefill logical length | 16385 | 262142 |
| Capacity | 16640 | 262144 |
| Cache shape | `[520,4,32,256]` | `[8192,4,32,256]` |
| Page-table row bytes | 2080 | 32768 |
| Position | 16385 / 16386 | 262142 / 262143 |
| Valid keys, inclusive position | 16386 / 16387 | 262143 / 262144 |
| Padded scanned keys | 16416 | 262144 |
| Capacity tiles `St` | 520 | 8192 |
| Runtime scanned tiles `PSt` | 513 | 8192 |
| K chunks per active core | 513 | 8192 |
| Virtual final page, row | 512, 1 / 2 | 8191, 30 / 31 |
| Valid columns in final tile | 2 / 3 | 31 / 32 |
| K tiles consumed per core; V same | 4104 | 65536 |

Decode factory:123-171 gives `page_block_size_t=1`, `PNHt=1`, `DHt=vDHt=8`, `Sk_chunk_t=1`. QK matmul tile dimensions are `(M,N,K)=(1,1,8)`; probability-times-V is `(1,8,1)`. Their subblocks are 1x1 and 1x4 respectively (factory:438-470). Half tiles, MLA, speculative positions, sliding windows, geometry overrides, position modulo and dynamic chunking are all inactive. On the reported 11x10 grid, four cores are active and 106 idle.

The work planner uses signed/unsigned 32-bit arithmetic; it rounds **`cur_pos+1`** to the chunk size (Work planner:35-91). With one core per head the interval is `[0,513)` or `[0,8192)`. No division remainder distributes work to other sequence cores; there are no cross-core softmax tree rounds.

### Page addresses and final-page update

The unsharded table is INT32; the UINT16 table branch is exclusive to sharded tables (decode validation:348 onward; Decode reader:270 onward). Shared dataflow:53 and Decode dataflow helpers:754-834 reduce the K/V mapping here to:

```text
physical_page = page_table[virtual_page]
tile_id = physical_page * 32 + kv_head * 8 + dimension_tile
```

The maximum tile ID is `8191*32 + 3*8 + 7 = 262143`. Each cache contains 536870912 bytes (512 MiB), and these indices/offset extents fit uint32. This does not prove runtime cache contents, but refutes an obvious 16-bit position or tile-ID truncation in the selected code.

The last long prefill chunk starts at 262016, has 126 real tokens, and pads to 128 (Model:252-265). Its table slice starts at page 8188 and fills four pages. Rows 30 and 31 of virtual page 8191 are padding. The decode source computes those row targets in order; their actual stored values were not read in this investigation.

The update kernels under `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/` read `uint32_t` positions and select `page_table[position/32]`: `reader_update_cache_interleaved_start_id.cpp:86-128`; the matching writer:98-115 uses `position%32` for the row, and writer:135-197 writes all four heads. The update factory `device/update_cache/paged_update_cache_program_factory.cpp:174-180,223,282-288` supplies the full 32768-byte table row and no position modulo. No max-context branch disagrees with SDPA's mapping.

Decode's final causal mask excludes column 31 at position 262142 and allows all columns at 262143 (Decode dataflow helpers:47,251 onward). Neither request exceeds the logical capacity.

### Buffer state ledger

For the selected full-tile, fixed-32-token configuration:

| Buffer/state | Producer and consumer transition | Count per chunk |
|---|---|---:|
| Q, c0 | Reader supplies one Q head tile row; compute retains through the chunk loop, pops after head work | 8 retained tiles |
| K, c1 | Reader reserves/reads/pushes; QK matmul waits/consumes | 8 tiles; capacity 16 |
| V, c2 | Reader reserves/reads/pushes; probability-times-V matmul waits/consumes | 8 tiles; capacity 16 |
| QK, c24 | Matmul produces; mask/exp replace in place; row reductions inspect; V matmul uses; explicit final pop | 1 tile |
| New maximum/sum | Reduction produces c27/c29; correction consumes old maximum/sum; moves establish c28/c30 for next chunk | 1 tile each |
| Output accumulator, c26 | First V matmul initializes 8 tiles; later iterations replace after multiply/add; final reciprocal scaling and move consume | 8 live tiles |
| Causal mask | Writer supplies final-chunk mask; compute uses only at last local chunk | 1 tile |
| Tree inputs | No sequence-tree children with one core/head | 0 |

Counts are identical for ordinary and last chunks; causality masks values rather than shortening K/V tile transfers. Decode compute:372-575,687-734 and factory:479-495 establish these counts. The long scan transfers exactly 65536 K tiles and 65536 V tiles per core, but this is not itself an overflow defect: Blackhole `tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_unpack.h:18-33` uses uint16 modular received-minus-acked counts, paired with the dataflow producer's uint16 modular accounting in `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406-425`. The live occupancy remains at most 16 tiles, far below the modulus. No inconsistent wait/reserve/pop/push count was identified for this geometry.

The final tile is reached from position 262112 onward. If a future sweep finds a sharp failure precisely there, revisit counter/address boundaries; current evidence only samples 262142/262143 and does not establish such a step.

## 4. Model, reference and completion audit

Installed source inspected: `/opt/venv/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` in `qwen38-functional-build` (the work log records Transformers 5.12.1).

- HF:683-686 splits each head's 512 query-projection outputs into query256 and gate256. Model:87-89 splits weights the same way.
- HF:669-670,688-693 normalizes the full 256-wide Q/K head before RoPE. Model:184-188 agrees. HF:570-605 and Model:177-180 rotate only the first 64 dimensions and retain the remainder.
- HF:713-716 flattens heads, applies sigmoid gate and output projection. Model:214-228 agrees. No full-attention gate substitution is indicated.
- Runner:52-60 constructs normalized/rotated K and projected V for **all** prefix tokens. Runner:65-68 evaluates the tail with absolute-position masking. Installed `transformers/cache_utils.py:149-155` concatenates rather than discarding that prefix.
- Runner:127,131 makes two sequential HF decode calls, retaining the first decode token for the second. The query inputs are teacher-supplied activations, not generated tokens.

Device/lifetime sequence:

1. TT prefill is enqueued; `to_torch(actual)` at Runner:187 obtains a completed host result before its PCC is printed.
2. Runner:213 snapshots prefill cache. Eager decode is enqueued and its result read with `to_torch(warm)` at 274-275, so the failure is already observable before capture.
3. Cache is restored, then `synchronize_device` at 277 completes prior work before trace capture.
4. Each of three replays restores the prefill snapshot, executes with `blocking=True`, and reads the result (282-285).
5. Runner:288-291 updates input, RoPE and position, retains the first decoded token, and blocks on the second replay before reading its result.

Thus a host enqueue being mistaken for successful completion is not the reported accuracy evidence. A trace-only stale-position explanation cannot explain eager failure. No source evidence identifies an incorrect snapshot/restore sequence.

## 5. Ranked competing explanations and rejected suspects

| Candidate | Source evidence and necessary condition | Fit to observations | Smallest falsifying/discriminating check |
|---|---|---|---|
| **Leading: BF16 online-state loss under default k32/core1** | Verified storage and 8192-loop recurrence. Real rows must accumulate enough diffuse mass or numerator rounding for material error. | Explains length dependence, prefill/decode precision difference and deterministic eager/trace failure. Exact PCC magnitude remains unverified. | Same Q/K/V and position, explicit k512/core1; measure attention scale/error, not only PCC. |
| Upstream Q/K/V numerical difference amplified by long context | BF16 projections/norm weights differ from HF arithmetic; no wrong head/RoPE/gate structure found. | Could contribute; both prefill and decode share most producers. No intermediate evidence. | Run Torch attention on exported **TT-produced** Q/K/V and compare to TT decode. |
| Last-page fill/update corruption | Boundary has padded rows30/31 followed by updates. Source indices and head strides are consistent. | Possible runtime alternative, but no concrete contradiction; cannot explain via an obvious capacity overrun. | Compare mapped final page before/after updates and all prior rows in that page. |
| Exponent/helper behavior | SDPA exponent flag defaults true independently of math flag; BF16 QK and helper paths differ from prefill. No demonstrated target-specific contract violation. | Secondary until a one-factor exponent control or identical-input reference implicates it. | Explicit exp false while retaining k32/core1; use per-head errors and norms. |
| Page/position integer overflow, packet truncation, GQA power-of-two requirement, trace-only corruption | Audited contradictions do not hold in the selected code. | Do not account for the evidence as presently formulated. | Revisit only if direct intermediate/counter/permutation evidence contradicts the source-derived mapping. |

Specific rejected or demoted claims:

- **32 KiB page-table row exceeds NoC burst size:** not a supported diagnosis. The whole row is allocated (Decode factory:666-674), and `tt_metal/hw/inc/api/dataflow/noc.h:237-262` routes the ordinary call through arbitrary-length reads in `internal/dataflow/dataflow_api_raw.h:62-73`. Blackhole `internal/tt-1xx/blackhole/noc_nonblocking_api.h:956-971` splits it into bursts.
- **GQA 24/4=6 is invalid because of the writer's power-of-two comment:** Decode writer:435-436 has that assumption in a comment, but the selected unsharded writer helper at Decode dataflow helpers:496-523 computes each logical head's tile-row/face offsets separately. The group crossing row16 is handled. A fixed head-placement defect would also need to explain the passing short run.
- **262143 wraps a 16-bit position:** the selected position and unsharded table data are uint32/int32, not uint16. The separate sharded-table uint16 branch is inactive.
- **Exactly 65536 streamed tiles necessarily overflows CB accounting:** the counter protocol intentionally uses modular differences and small live occupancy. No mismatch found.
- **A loose exponential approximation alone is proven:** it is not. The selected Blackhole first-column correction with `EXP_APPROX_MODE=true` calls `_ckernel_sfpu_exp_accurate_<...,true>` (`tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h:197-215`), which routes to FP32 accurate exponential (`llk_sfpu/ckernel_sfpu_exp.h:439-475`). The QK exponent helper is separate (Shared compute:352-407). Flag names alone are insufficient to assign blame. Both paths still pack to the configured BF16 CBs.

## 6. Smallest follow-up checks, in order

These are recommendations for a separately authorized execution phase; none were run here.

### A. Reduce serial chunk count with fixed core cap

Capture/reuse a single failing Q, completed K/V cache and page table. Compare explicit configurations that preserve the baseline exponent choice:

```python
# Use the same grid and existing compute_kernel_config in every case.
ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
    q_chunk_size=32,
    k_chunk_size=512,                 # baseline control: 32
    max_cores_per_head_batch=1,        # retain the implicit baseline cap
    exp_approx_mode=True,             # retain baseline exponent mode
)
```

At position262142 the baseline has 8192 chunks; k512 has 512. First compare the raw attention result, then full-layer output, with per-head PCC, norm ratio and relative L2 error. A large accuracy recovery favors serial numerical accumulation, though changing chunk size also changes matmul/reduction geometry and is not by itself exclusive proof.

Do not omit `max_cores_per_head_batch` in this A/B: the `SDPAProgramConfig` type defaults it to 16 (`ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp:21`), unlike the wrapper's no-config default of 1. Accidentally changing both parameters would confound the experiment.

If needed, a second independent control retains k32 but explicitly uses 16 cores/head: 64 active cores and 512 local chunks per core, followed by four tree rounds. Recovery from both controls strongly implicates serial depth. The combined k512/core16 case has 32 local chunks and is useful only after the separate controls. No claim is made here that any override has passed allocation or accuracy on hardware.

### B. Isolate the operator with identical inputs

Export the post-norm/post-RoPE TT Q and post-update TT K/V. Unpage K/V with exactly the recorded table, retain keys `0..current_pos`, expand each KV head over its six Q heads, and evaluate one-query attention in Torch FP32. Compare this to the TT attention result before gate/projection/MLP. This avoids attributing HF-versus-TT producer differences to SDPA and requires only one query's reference attention, not a quadratic full prefill reference.

Also compare TT Q and the two updated K/V rows directly to HF. If Torch attention on the TT tensors is already far from the HF result, upstream producers/cache contents remain necessary to the explanation. If it matches HF while TT decode diverges, the attention operator is isolated.

For a controlled synthetic operator check, use equal logits and values whose contributions do **not** identically mirror the denominator (for example, nonzero V only in the first page). All-constant V can hide paired numerator/denominator saturation. Report absolute magnitude error; PCC alone can miss a uniform gain error.

### C. Verify the final page and permutation independently

Record `physical_page = page_table[0,8191]`. Read it before first decode, after row30 update and after row31 update. Verify all four heads, projected K/V parity, preservation of rows0:30 during the first update, and preservation of row30 during the second. To test page mapping separately, construct identity and randomized physical layouts for the same logical cache and Q. Comparing different logical data would not isolate permutation handling.

### D. Distinguish gradual accumulation from a hard boundary

With fixed allocated capacity and one fixed cache/Q fixture, compare each result against its own correctly masked Torch reference at several positions, including 16385, 32767, 65535, 131071, **262111 and 262112**, 262142 and 262143. The bold pair enters the 8192nd chunk and the 65536-tile total without changing table width. Progressive norm/error drift favors numerical state loss; a sharp step tied to the final chunk would justify renewed counter/address inspection. This sweep is an operator-level follow-up, not a request to rerun every full-model prefill.

## Other potential issues and test gaps

- **Norm-weight rounding:** Model:63-65,92-93 uploads BF16-rounded `1 + weight.float()`, whereas installed HF RMSNorm:745-750 multiplies FP32-normalized activations by FP32 `1 + weight` before casting back. This is a real rounding difference without a context-specific branch; it is not the headline cause absent Q/K evidence.
- **Default-path coverage is short and different:** `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode.py:274-308` exercises omitted `program_config` at capacity1024 with eight Q heads, one KV head, dimension512 and BF8 K/V. General paged tests use explicit configurations: `sdpa_test_utils.py:48-60,552-564` raises chunk size up to512, sets exp false, and inherits the configuration type's 16-core cap. Those tests do not establish the default 8192-serial-chunk BF16 behavior for this model. Their helper also uses FP32 DST false and looser PCC thresholds (`513-526`).
- **The layer harness cannot localize first divergence:** current artifacts lack Q/K/V, raw attention, cache-row, per-head magnitude and sum checks. Its deterministic flag checks repeatability, not mathematical accuracy.
- **A precision repair needs more than toggling the existing compute flag:** it is already true. If the kernel is repaired instead of configuring shorter reductions, format/size contracts must be audited together. In particular, blindly changing all `stats_df` buffers to FP32 affects writer/reduction transfers; promoting only sums requires checking reconfiguration in `mul_block_inplace` and surrounding callers, which currently rely on matching BF16 formats. This report proposes no implementation patch.

## Report adjudication

After the initial investigation, headline claims were rechecked against wrapper defaulting, host work assignment, selected compute/dataflow kernels, shared helpers, installed HF and the test lifecycle. The report intentionally does not promote the attractive 16-bit-index, 32 KiB-transfer, GQA-comment, CB-wrap or trace stories: their required source contradictions were not found. The leading precision explanation remains qualified by the lack of a frozen-input runtime control and by the scalar toy's failure to predict the real short-context pass quantitatively.

No implementation was changed. This report is the requested model-specific `full_context_debug/AUTODEBUG.md`; the existing `functional_decoder/AUTODEBUG.md` and `functional_decoder/long_linear_debug/AUTODEBUG.md` were not replaced.
