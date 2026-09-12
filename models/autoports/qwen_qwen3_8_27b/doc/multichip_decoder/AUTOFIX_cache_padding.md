# AutoFix diagnosis: one-token cache comparison

Source-only follow-up, 2026-09-11. No hardware commands or implementation
changes were made by this agent. The parent owns fixture corrections and
runtime verification.

## Observed failure

`final_l3_b1_s1_c0.log` reports prefill/decode/changed-decode output PCCs
0.9999998808, 0.9999999404, and 0.9999993443. Whole prefix-cache PCCs are
K 0.8709033 and V 0.5387563. Raw eager/replay state equality and the
changed-update write-ownership check pass; unowned pages stay zero.
The baseline report also records finite outputs and successful trace checks.

The parent separately inspected baseline raw cache values: valid row 0 has
maximum magnitude K 11.25 / V 3.6875, while future rows contain nonzero values
up to 6.5. This is parent-supplied runtime evidence, not a measurement by this
investigation. It refutes the assumption that the frozen baseline guarantees
zero cache padding.

## Source contract and causal chain

The future rows are **physically written padding**, not unwritten cache memory.

- `optimized_decoder.py:766-790` explicitly allows padded writes in the
  request's final page. It obtains K/V from `_qkv`, casts them to cache dtype,
  and calls `paged_fill_cache` without a token-granular valid-row mask.
- For S1, `_qkv` uses decode head creation, whose logical K/V output is
  `[1,B,Hkv,256]`, then reshapes to `[B,Hkv,1,256]`
  (`optimized_decoder.py:599-622`). Baseline Hkv=4; TP-local Hkv=1. Both have
  32 physical sequence rows after the reshape.
- Native decode-head specs round head storage to 32 rows
  (`experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.cpp:127-156`).
  The interleaved reader writes only `num_kv_heads` head rows
  (`device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:228-279`).
  Logical tensor semantics do not require identical remaining head padding.
- BF16 tiled reshape likewise does not unconditionally clear implicit
  padding: `data_movement/reshape_view/reshape.cpp:536-548` fills it only when
  explicitly requested. `_qkv` supplies no padding value. Different head
  counts and reshape paths can therefore leave different future-row bytes
  while preserving every logical K/V element. The exact producer of each
  observed padding value is not established here.
- `experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp:95-104`
  derives sequence work from `input_tensor.padded_shape()[2]`, which is 32
  for S1. Its writer copies complete tiles to the mapped physical page
  (`device/kernels/dataflow/writer_fill_cache_interleaved.cpp:226-244`).
  All 32 rows are copied, but only row 0 belongs to the logical prefix.
  The optional valid-sequence feature also rounds to whole tiles/blocks
  (writer lines 129-136); it is not a one-row write mask.
- Chunked prefill attention is always causal
  (`transformer/sdpa/sdpa.cpp:103-130`). Decode builds a mask after `cur_pos`
  (`transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:247-280`).
  A contiguous decode writes the next row before using it, so finite future
  padding is outside the logical attention prefix until overwritten.

Native paths above are relative to `ttnn/cpp/ttnn/operations/`.

This explains high output agreement alongside low whole-cache PCC. It is a
strong source-supported fixture-scope diagnosis; acceptance still requires
the parent's comparison of actual valid rows. Do not dismiss any mismatch in
those rows or unexpected writes to another page as padding.

## Required cache comparison

For each user `u`, logical token position `p` in an initialized prefix of
length `n` maps to:

```text
physical_page = page_table[u, p // 32]
physical_row  = p % 32
compare cache[physical_page, every_real_head, physical_row, every_head_dim]
for 0 <= p < n
```

S1 therefore compares only row 0 of the owned page across all four global
heads. After an ordinary contiguous decode at position n, include the newly
written row n as well. Stateful recurrent/conv buffers have no corresponding
future-token padding exclusion and still require their full comparison.

Keep independent checks on raw cache buffers: unowned pages unchanged,
only authorized physical rows changed by updates, and eager/replay state
bitwise equality. Excluding semantically invalid padding from cross-layout
PCC does not justify weakening those ownership/replay checks.

## Changed-input fixture correction

The original changed replay restored the prefix and jumped to n+1, leaving
position n uninitialized. It also rolled the page table without moving its
cache contents. Such a test can establish replay equivalence but does not
represent valid contiguous history: causal masking retains all earlier rows,
including the hole.

The parent's updated runner now chooses changed positions inside the existing
prefix, remaps cache pages with the changed page table for short cases, and
compares logical prefix rows. Source review confirms the short-case mapping
and raw untouched-row/eager-replay checks remain present. Runtime reruns are
in progress; no final pass is claimed by this report.

For long sampled-cache cases, mask by logical page number rather than assuming
the last allocated page is the last prefix page: capacity reserves n+2 tokens.
For example, n8191 can allocate an extra wholly future page; the last valid
page is `table[:,(n-1)//32]`, not necessarily `table[:,-1]`. Exclude future
rows in that last valid page and wholly future sampled pages. This is a
source-level generality check, not an observed additional failing run.

## Verdict

Whole physical-page PCC imposes an unsupported equality requirement on
padding. Comparing logical initialized cache rows is the correct model
contract. The frozen implementation need not be changed to make padding match.
The fixture fix must pass valid-row PCC, output PCC, raw write ownership,
and eager/refreshed-input replay checks before this regression is closed.

## Parent hardware follow-up

Resolution: controlled native padding, corrected fixture. `regressions.xml`
records28 passed cases with valid-row cache PCC, exact raw-state replay,
unchanged-row ownership, and unowned-page checks. Changed-table fixtures remap
physical cache contents and use initialized positions; they no longer skip an
uncommitted token. All four max-context layer probes and the retained-input
stack also pass (`capacity_*.json`). Decoder source was unchanged by this fix.
