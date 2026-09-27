# AutoDebug: BF16 paged prefill L1 overflow

## Verdict and scope

Source-proven allocation error: the full-attention paged-prefill configuration
uses a 256-token K chunk tuned for BFP8 caches. BF16 K/V double buffers exceed
Blackhole L1 when Q needs double buffering. This is an allocation failure before
device execution, not evidence of BF16 numerical instability. No implementation
edits or hardware execution were performed during this investigation.

Existing evidence: `smoke_kv_bf16.log` fails at
`tt/optimized_decoder.py:1108`, reporting CB end **1,590,848 B** above L1 limit
**1,572,864 B** on `(8,4)` cores. `smoke_kv_bf16.json` records passes for prompt
lengths 31, 32, 33, 1023, 1024 and 1025; the next requested length is 4097.
The working tree already contained datatype-sweep changes; these were left intact.

## Lowered configuration and first failing shape

- `configs/kv_bf16.json` selects BF16 caches for both layer kinds. Full-attention
  prefill retains HiFi2, BF16 Q/output, FP32 destination accumulation,
  `math_approx_mode=False`, `packer_l1_acc=False`, and full destination sync.
  FP32 accumulation selects the factory's legacy compute path.
- `tests/config.json` gives 16 Q heads, two global KV heads, global head dimension
  512, and context 262144. `MultichipDecoder` partitions Q heads four ways and
  replicates the global KV heads as necessary: each device uses four Q heads and
  one KV head. Cache pages contain 32 tokens.
- The outer decoder prefill chunk is **1024**, distinct from the attention
  helper's maximum internal chunk of 8192. The initial chunk uses non-paged
  attention over projected BF16 K/V. Later full-attention chunks use
  `ConfiguredChunkedPrefillAttention` and the paged cache.
- A 4097-token prompt with three generated tokens allocates context 4100,
  rounded to 4224 tokens: local K/V shapes `[132,1,32,512]`, page table `[1,132]`.
  The first failing paged call is the second full 1024-token chunk, with Q
  `[1,4,1024,512]` and `chunk_start_idx=1024`.
- Defaults are grid `(8,4)`, Q chunk 64, K chunk 256. There are 16 Q chunks per
  head, 64 total over 32 cores, so `q_buffer_factor=2`. The read end 2048 fits
  capacity 4224, so the existing page-boundary fallback does not activate.

For the previously passing 1025-token prompt, the only paged call has 32 physical
Q rows. Its table has 36 pages / 1152-token capacity. Q padding gives end 1088;
K256 would read through 1280, so the existing fallback already selects Q64/K128.
That short call also has only one Q chunk per participating core. Thus its pass
does not validate Q64/K256 with a full later chunk.

## Exact source allocation arithmetic

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
lines 474–515 and 769–889 determine the following per-core allocations.
Here `Sq_chunk_t=2`, `Sk_chunk_t=8`, `DHt=vDHt=16`. BF16 tiles are 2048 B,
FP32 tiles 4096 B, BFP8 tiles 1088 B (`tt_backend_api_types.hpp:121`).

| Buffer | Tiles / bytes | Bytes |
| --- | --- | ---: |
| Q | `2*16*2` BF16 tiles | 131072 |
| K | `8*16*2` BF16 tiles | 524288 |
| V | `8*16*2` BF16 tiles | 524288 |
| Causal mask | 2 BF16 tiles | 4096 |
| Identity scale and column identity | 2 BF16 tiles | 4096 |
| Page table | `align64(132*4)` | 576 |
| QK intermediate | `2*8` FP32 tiles | 65536 |
| Output intermediates A and B | `2*2*16` BF16 tiles | 131072 |
| Max A, max B, exp max difference | `3*2` BF16 tiles | 12288 |
| Sum A and B | `2*2` FP32 tiles | 16384 |
| Output | `2*16` BF16 tiles | 65536 |
| **Total circular buffers** | | **1479232** |

The page-table CB uses `buffer()->aligned_page_size()` (factory lines 162–168),
so its size is **576**, not its logical 528 bytes. Blackhole DRAM alignment is
64 bytes (`noc/noc_parameters.h:378–394`); all CB starts are aligned to that
boundary (`tt_metal/impl/program/program.cpp:1735`). All listed sizes are already
64-byte multiples, so there are no additional inter-buffer gaps.

Blackhole `dev_mem_map.h` expands `MEM_MAP_END` to **40960** with two DM processors.
`bh_hal_tensix.cpp:44,68–69` adds the 69 KiB default kernel-config reservation,
giving aligned `DEFAULT_UNRESERVED = 40960 + 70656 = 111616`.
`ProgramImpl::allocate_circular_buffers` starts at the allocator's L1 base.
Therefore **111616 + 1479232 = 1590848**, exactly the logged failure, exceeding
the limit by **17984 B**.

Keeping BFP8 K/V changes only those two rows: each becomes 278528 B. This saves
491520 B and produces end **1099328 B**, explaining why the same program fits
the baseline cache dtype.

## Smallest bounded repair and pending verification

For BF16 caches with full-attention head dimension 512, choose the already
constructed `boundary_program` before calculating query/read padding. Its
default Q64/K128 halves K/V buffering and the FP32 QK intermediate, saving
**557056 B**. A narrow condition such as
`head_dim >= 512 and k_cache.dtype == ttnn.bfloat16` preserves every BFP8 program
selection and avoids changing the sliding-attention configuration. Retain the
existing read-capacity check, padding, output slicing, geometry override,
cache allocation, and compute policy.

| Configuration | CB end, 132 pages | CB end, 8192 pages / context 262144 |
| --- | ---: | ---: |
| Current BF16 Q64/K256 | 1590848 | 1623040 |
| Proposed BF16 Q64/K128 | 1033792 | 1065984 |
| Existing BFP8 Q64/K256 | 1099328 | 1131520 |

The full-context table is 32768 B; proposed BF16 headroom remains **506880 B**.
Even the fallback's maximum Q128/K128 uses end 1455104 B at full context.
These are source allocation bounds, not measured performance or correctness.
The 128-token K chunk also agrees with the public cache-capacity rounding, so
the fix needs no context reduction or additional cache pages.

Pending device validation: focused Q64/K256 versus Q64/K128 reproduction using
the actual model cache geometry; rerun the original BF16 smoke including 4097;
exercise full later chunks, nonaligned final chunks and the logical page-capacity
boundary; compare BFP8 program selection and existing correctness evidence.
Source reasoning establishes the allocation cause and predicted L1 reduction;
numerical equivalence and complete model smoke remain to be verified.
