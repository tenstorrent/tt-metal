# AutoFix decoder address hypothesis audit

## Conclusion

**No decoder page-table, paged-fill, chunked-SDPA, or earlier slice index violation was found for the captured run.** The concrete address formulas refute an out-of-range page or a padded-head/cache-stride mismatch at any aligned chunk preceding offset 93696, assuming the tensors/page-table values satisfy the source contract. The source also bounds all aligned chunks through context 262144. This does not prove the device executed these formulas correctly or rule out corrupted runtime metadata, allocator/descriptor faults, or a distinct kernel bug. It supplies no basis for a decoder bounds workaround or a claim that the sampled RoPE concat caused corruption.

Scope: read-only source and preserved `AUTOTRIAGE.md`; independent host integer arithmetic, with no TTNN/model import or target execution. No device/process access, resets, debugger attachment, or implementation changes. Checkout HEAD is `822d04806f292e2cf7e92de7f7be79fa8d83e5aa`. Native source was checked in that checkout; loaded binary equivalence is not asserted.

## Captured path and geometry

`tests/long_context.py` allocates context `n=262144`, page order `randperm(8192)`, and two BF16 tiled caches of `[8192,2,32,256]`. `new_kv_state` validates distinct physical page IDs in `[0,8191]`. It uploads one row-major INT32 table `[1,8192]`. Each KV cache has 131072 BF16 tiles of 2048 bytes, or 268435456 bytes.

The first test pass has logical length 262143. Before offset 93696 every plan entry is a full 256-token aligned prefill chunk: entries 0–365 were submitted, entry 366 stopped in Q RoPE. No one-token/decode path, terminal partial chunk, second pass, or trace replay has been reached. “Submitted” does not imply known device completion; the capture lacks previous-operation completion history.

For chunk index `j`, `s=256*j`, and logical table `P`, the fill receives suffix `P[8*j:]`, while SDPA receives the original table `P` and scalar absolute start `s`. This difference is intentional: fill's local sequence starts at zero, while SDPA reads absolute history.

| Case | Start | Fill logical pages | Suffix INT32 count / bytes | SDPA maximum logical page |
|---|---:|---|---|---:|
| First chunk | 0 | 0–7 | 8192 / 32768 | 7 |
| Last earlier submitted chunk, j=365 | 93440 | 2920–2927 | 5272 / 21088 | 2927 |
| Stopped chunk, j=366, if reached | 93696 | 2928–2935 | 5264 / 21056 | 2935 |
| Last full-context aligned chunk | 261888 | 8184–8191 | 8 / 32 | 8191 |

The first pass's terminal plan instead uses 224 tokens at 261888 followed by 31 single-token steps. None can explain the earlier stop.

## Paged-fill address proof

Sources: `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_{device_operation,program_factory}.cpp` and `device/kernels/dataflow/{reader,writer}_fill_cache_interleaved.cpp`.

After `(0,2,1,3)` permute, each K/V input is `[1,2,256,256]`, with the same padded shape. Native factory values are `num_heads=2`, `input_seq_len_t=8`, `Wt=8`, `block_size_t=1`, `num_blocks_of_work=16`. Work rows 0–15 decode to `head=row/8` in `{0,1}` and `seq_tile=row%8` in `[0,7]`. Reader source tiles are precisely 0–127.

The writer's page lookup is `suffix[seq_tile] = P[8*j+seq_tile]`. For width tile `w` in `[0,7]`, its destination tile is:

```
physical_tile = 16 * P[8*j+seq_tile] + 8 * head + w
```

Thus every write is in `[0,131071]`, the allocated tile range; physical IDs need not be contiguous or increasing. Page uniqueness prevents different logical pages from aliasing, and the two head ranges occupy distinct halves of each physical page. `capacity_t=0`, no block-size override, no batch-index tensor, and no valid-length tensor are passed, so circular-cache and batched special cases do not apply. The raw-copy dtype contract holds: projected/normalized/rotated K and projected V are BF16, matching the BF16 cache.

For all earlier chunks, the suffix contains at least 5272 entries, but fill reads only its first eight. The factory allocates one metadata entry using `page_table_tensor.buffer()->aligned_page_size()` and the writer reads exactly that amount into the same-sized DFB. Although a floor-log2 is computed for the variable stick length, the current writer does not use it to construct an address; it uses the tensor accessor and explicit byte count. Non-power-of-two suffix lengths therefore do not introduce a log2-based addressing error here.

## Chunked SDPA address proof

Sources: `operations/transformer/sdpa/sdpa.cpp:105–141`, `device/sdpa_program_factory.cpp:145–174,1189–1270`, `device/kernels/dataflow/reader_interleaved.cpp:399–450,560–713,922–935`, and `device/kernels/dataflow/dataflow_common.hpp:53–94,255–319` (under `ttnn/cpp/ttnn`).

The wrapper sets causal attention. This is scalar-offset legacy chunked mode, not tensor-offset flexible mode. Values are `B=1`, `NQH=16`, `NKH=NVH=2`, `DHt=8`, `Sq=256`, `Sq_chunk_t=Sk_chunk_t=1`, `q_num_chunks=8`, and `Sk=s+256`. There is no sequence padding for these aligned chunks. GQA maps `q_head/8` to KV head 0 or 1. Paged mode excludes the separate dense GQA-crossing optimization.

The reader addresses Q using its **local** query tile 0–7, before adding `chunked_q_chunk_offset=s/32`. The resulting absolute query tile is `s/32+q`. Its causal K loop visits virtual rows `0..s/32+q`, inclusive. In particular, it cannot visit beyond `(s+256)/32-1`, and every such page has been filled by the current or earlier chunk under the intended queue ordering. The reader's `valid_Skt_bound` is `valid_Skt=(s+256)/32` for scalar-offset mode: it does not add the prefix a second time.

`read_paged_chunk_with_padding` resolves virtual row `r` through the full table and uses the same physical tile formula as fill:

```
physical_tile = 16 * P[r] + 8 * kv_head + w
```

All page indices remain below 8192, all physical tile indices below 131072. Its per-K/V read writes eight tiles into an eight-tile CB region (`dst_rows=1`, `dst_cols=8`); K's transpose changes L1 tile order, not the byte bound. The full table is 32768 bytes and the factory allocates its metadata CB using that same aligned stick size. For this caller that table size never varies between chunks.

## Earlier slices, padding, and cache identity

Sources: decoder `reshape`, `_full_attention`, and `prepare_prefill`; `operations/data_movement/slice/{slice.cpp,device/slice_program_factory_rm.cpp,device/slice_program_factory_tile.cpp,device/slice_device_operation.cpp}`; RM slice reader/writer; `operations/data_movement/permute/permute.cpp:140–149`; `operations/data_movement/transpose/device/transpose_device_operation.cpp:25–61`.

- The full input is logically `[1,262143,2048]`, padded to `[1,262144,2048]`. Every prior input slice starts at a multiple of 256 and selects 256 rows; its end is at most 93696. Its source tile start is `(s/32)*64`; its 512 selected tiles remain below the full input's 524288 tiles. No tail-padding row is selected before the stop.
- The decoder explicitly converts to row-major around reshape, repacking logical rows before returning to tiles. Q/gate starts as logical `[1,256,16,512]`, padded `[1,256,32,512]`. Width slices `[0:256]` and `[256:512]` stay within the 512-feature row. Q/K RoPE slices `[0:32]`, `[32:64]`, `[0:64]`, and `[64:256]` have tile-aligned feature boundaries. The heads dimension retains 32 padded rows while logical heads remain 16 or 2; the width slices do not read beyond those padded rows.
- Permute `(0,2,1,3)` routes to HC transpose. Its output-shape code explicitly sets output dimension 1 to the input's **logical** head count and tiles the new sequence dimension. Consequently K/V become `[1,2,256,256]`, not `[1,32,256,256]`; there is no factor-of-16 head-stride mismatch with cache storage. Q becomes `[1,16,256,256]`.
- The row-major table suffix has begin byte `32*j` and length `32768-32*j`. RM slice reads from `begin-misalignment`, for `length+misalignment` bytes, then trims the prefix; its end is exactly byte 32768 for either 32- or 64-byte alignment. The writer emits only the requested length into the suffix output. The DFB sizing reserves space for alignment; any output row alignment padding is irrelevant to fill's first eight integer lookups. These suffixes remain multiples of 32 bytes throughout the earlier 256-token chunks.
- Slice program hashing includes starts, ends, steps, logical/padded input shape, dtype/layout, and memory config; a new absolute slice does not intentionally reuse an old start. Paged-fill hashing includes its tensor arguments (thus table specs), and cache-hit bindings refresh input/cache/page-table tensors. SDPA's scalar `chunk_start_idx` is an operation attribute, explicitly part of its default cache key. These are source contracts; no preserved cache-key collision, stale binding, or actual runtime value is available to establish a violation.

An independent Python integer-only audit enumerated all 1024 possible aligned 256-token chunks. It verified suffix spans ending at byte 32768 for 32/64-byte alignment, eight fill pages within the table, every causal query-tile upper bound within 8192 pages, and physical tile maximum 131071 of 131072. This is arithmetic validation, not a device test or numerical result.

## Remaining discriminating evidence

The corruption hypothesis would need an actual wrong page-table value, a discrepant tensor padded shape/buffer extent, a bad runtime binding, or an observed out-of-range NoC write before the stop. Those facts are not in the preserved evidence. A later authorized instrumented run could check the last completed fill's page values and runtime arguments, but no such run was performed here. The known captured dependency remains host FetchQ reservation/MMIO credit access, with device prefetch awaiting host input; neither the current chunk's fill nor SDPA has been called at that host stop.
