# AutoDebug: continuation SDPA block alignment and mapped read bounds

Status: source diagnosis; no hardware operations or implementation edits by
this investigator. The proposed alignment repair is correct for the selected
power-of-two chunk sizes. It should also cap rounded reads to existing
page-table capacity, as proposed by the main agent during this investigation.
Hardware verification of the combined repair remains pending.

## Observed failure

`stress.log` records full-attention continuation failures:

- start32 with Q chunk32 / K chunk128: `chunk_start_idx must be a multiple
  of k_chunk_size`;
- start64 with Q chunk128 / K chunk128: `chunk_start_idx must be a multiple
  of q_chunk_size`.

The later `cq_id 0 is out of range` messages follow the original failures and
device teardown; they do not explain the SDPA validation errors. The log
also contains a separate B32 linear-attention L1 circular-buffer collision,
outside this bounded continuation diagnosis.

## Native contracts and the prefill loop

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:331`
requires Q and K chunk sizes divisible by32. For scalar `chunk_start_idx`,
lines346-361 additionally require the absolute start to be divisible by
**both** chunks. Lines365-374 check only logical end coverage:
`page_table.shape[1] * page_size >= start + Q_length`.

`sdpa_program_factory.cpp:153` divides the start by Q chunk size to form the
query-chunk offset, so changing the start or simply suppressing validation
would change position semantics. Lines330-360 explicitly support lengths
that are not divisible by Q/K chunks, padding internally and masking tails.
Consequently the real tail length need not be rounded or truncated by the
model.

In `OptimizedDecoder.prefill_forward`, `absolute = start_pos + offset` drives
both routing and the SDPA start. A start inside a 32-token page takes one
token through `_full_decode`; once the next page boundary is reached,
`_full_prefill` sees a page-aligned absolute start. Reducing Q and K chunks
independently until each divides that start preserves all positions. For the
selected configured128 blocks, starts32/64/96/128 select32/64/32/128,
respectively, before any length/capacity restriction. Start0 retains the
configured size. Short-Q selection32 is also valid, though native code does
not require Q chunk <= logical Q length.

Recompute these choices at every `_full_prefill` invocation; a single choice
at the outer request start misses the transition from inside-page decode to
paged prefill. The selected model chunk2048 is page-aligned, so every complete
internal chunk preserves page alignment. An arbitrary policy `chunk_size`
not divisible by32 would send later internal chunks into the inside-page
path, requiring positions even for a fresh prefill; that policy is outside
the selected configuration.

Cache fill remains correct: it slices virtual pages from `start//32` through
`ceil((start+t)/32)`, writes the actual K/V chunk, and only touches future
padded rows in its last page. Inside-page continuation never calls that fill
path. Lowering SDPA blocks changes the reader's processing geometry, not
cache page indices, positions, request ownership, or output lengths.

## Separate rounded-window access risk

The existing caller-owned mapping covers logical pages. It is not necessarily
large enough for a selected SDPA block's rounded read window:

1. **Decode:**
   `sdpa_decode/device/kernels/rt_args_common.hpp:61` rounds `current_pos+1`
   up to K chunk size. The reader passes whole chunks into K/V reads
   (`reader_decode_all.cpp:294`), and `dataflow_common.hpp:704` and line786
   read every tile row. The shared paged mapper indexes
   `page_table_ptr[virtual_block]` directly
   (`sdpa/device/kernels/dataflow/dataflow_common.hpp:64`). Causal masking
   does not prevent those earlier reads.
2. **Legacy scalar prefill:** the factory sets
   `valid_Skt = ceil((start+t)/32)` and
   `padded_Sk = ceil((start+t)/k_chunk)*k_chunk`
   (`sdpa_program_factory.cpp:325-348`). The reader's legacy branch then adds
   `start/32` again to its valid bound (`reader_interleaved.cpp:321-327`).
   Its comment describes a chunk0-based bound, inconsistent with the
   current factory expression. The K-loop remains capped by `padded_Sk`,
   but its row count can therefore include rounded tail pages rather than
   zero-padding them. `read_paged_chunk_with_padding` maps every `src_rows`
   entry before zero-filling the remainder (`dataflow_common.hpp:281`).

Concrete source-derived cases with the original minimal mapping:

| Case | Mapped pages | Potential pages read |
| --- | ---: | ---: |
| length1, decode K64 | 1 | 2 |
| continuation start64, t193, K64, total length257 | 9 | 10 |
| final chunk start2048, t1, K128, total length2049 | 65 | 68 |

`run_optimized_decoder.py` maps `ceil((length+4)/32)` pages per user; the
three extra physical sentinel pages do not extend those virtual mappings.
An aligned page-table row buffer can contain padding beyond its logical
entries, but such bytes are not caller-owned mappings. A passing PCC from
masked reads is insufficient to certify these accesses. This is a mapping
and reader-bound issue for any dtype, not evidence of intrinsic BFP8
numerical instability. Keep the selected BFP8 cache while testing the fix.

## Preferred fix: adapt blocks to existing mapped capacity

The main agent proposed preserving the caller page contract instead of
requiring extra allocation. Source review supports that approach.

Let `C = page_table.shape[-1] * PAGE_SIZE`, using logical mapped entries.
For each full-prefill call with logical end `E = start+t`:

```text
q = selected Q block (including existing short-Q policy)
while start % q != 0: q /= 2
k = configured prefill K block
while start % k != 0 OR ceil(E/k)*k > C: k /= 2
```

This satisfies both scalar-start checks and bounds every native prefill
page access, including the legacy reader's overly generous valid bound:
the outer K-loop cannot read beyond `ceil(E/k)*k`. The native logical-end
contract supplies `E <= C`; start and C are multiples32, so a legal choice32
always exists for in-contract calls.

For decode, start from the effective configured K block after any existing
policy selection and halve until `C % k == 0`. For every in-contract device
position `0 <= p < C`, `ceil((p+1)/k)*k <= C` then follows. This choice depends
only on static page-table shape, preserving trace replay with changed
device-side positions and avoiding host reads of `current_pos`.

The halving proof assumes positive configured blocks of the form `32*2^n`,
which includes the selected32/64/128 values. An arbitrary tile multiple96
could halve to48 and then below32 without becoming a legal tile multiple;
do not claim arbitrary configured chunk values are covered by this rule.

Host-only arithmetic checks exercised configured32/64/128/256, every
page-aligned start through8192, and then capacities1..129 pages with all page
starts and singleton/33-token/end-of-capacity tails. Every selected block
remained >=32, met start divisibility, and stayed within mapped capacity.
These checks validate the arithmetic, not device execution or accuracy.

## Required follow-through

- Rerun the original full-attention continuation cases at starts32 and64,
  including minimally mapped total lengths33/129/257. Assert selected Q/K
  blocks and public shapes, then apply existing per-user/PCC/cache checks.
- Exercise fresh and continued chunk tails at2049 and4097 using the existing
  minimal page mapping, plus singleton decode. Those cases distinguish
  logical-page coverage from rounded-window coverage.
- In a watcher-enabled exact-shape test, retain BFP8, PAGE_SIZE32, actual KV
  head count, shuffled/disjoint pages, and the exact update/reader ops. Check
  both B1 and higher batch, refreshed trace positions, and page-table swaps.
  Extra physical sentinel pages must remain outside each logical mapping.
- Rerun aligned normal prefixes to establish correctness and measure any
  effect of block reduction. Do not infer that the repair retains previous
  timing at odd page counts: decode will deliberately select a smaller K
  block when the larger block does not divide mapped capacity.
- Keep the separate linear-attention L1 failure distinct. Passing the SDPA
  alignment checks cannot establish that the full stress suite now passes.
