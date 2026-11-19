# Bounded prefill: certify actual cache writes

The previous bulk-fill eligibility check required every entry in an external
virtual page table to be globally unique. A bounded physical pool can legitimately
use repeated valid addresses in unused virtual entries. For example, three rows
of 96 virtual pages backed by 64 physical pages can each own the two pages needed
for a 33-token prefill. Rejecting bulk fill because of unused tail aliases loses
the accepted batched fill strategy without identifying a write conflict.

`tt/bounded_prefill.py` now certifies only the current aligned chunk's write
footprint. For absolute start `s` and logical count `n`, it checks the entries
`table[:, s//32 : s//32 + ceil(n/32)]`. Every checked address must be nonnegative
and below the minimum physical extent of all bound K/V allocations. Duplicate
active destinations retain the existing ordered-fill path; disjoint destinations
retain batched fill. The `bulk_fill_chunks` and `ordered_fill_chunks` counters
record aligned runtime decisions, excluding preparation.

## Native write-range evidence

The proof follows the current native implementation:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp`
  computes work from `input_batch * num_heads * input_seq_len_t`, where
  `input_seq_len_t` is the padded input sequence length divided by 32 (lines
  136–153). Cache and page-table capacity do not extend the work range.
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp`
  derives each sequence tile from that work index and accesses
  `page_table_ptr[seq_tile_idx / block_size_t]` (lines 23–38, 220–226).
  Here the accepted cache block size is 32, so `block_size_t=1`.
- Bounded prefill pads the final logical sequence tile before filling. Thus
  `n=33` writes two complete physical pages, while `n=32` writes one.
- The same writer explicitly treats `0xffffffff` (`int32(-1)`) as a skip entry.
  It consumes/discards the corresponding input tiles and issues no cache write
  (lines 278–283). The native validation/program hash does not specialize on
  page-table contents, so synthetic skip values do not change the prepared
  runtime signature.

## Safe preparation and restored mappings

Preparation no longer assumes caller-supplied page mappings are disjoint. It
installs a temporary private mapping with distinct real physical destinations
for both the batched and ordered fill warmups. Page IDs are interleaved across
rows, making each row's first page distinct for the decode-style head warmup.
When the physical pool is smaller than `batch * 16`, excess synthetic fill pages
use the native skip sentinel. The full attention table always contains valid
physical addresses; unused synthetic entries read valid page zero instead of
the skip sentinel. This prepares both fill signatures without allocating a
larger cache or excluding valid short prefills from small pools.

The synthetic head warmup requires at least one physical page per simultaneously
active prepared row. This is the minimum physical requirement for that fixed
batch, not a reduction in per-request context capability. The ordinary
64-page / three-row case meets it and needs no skip entries in its first 16
pages per row.

A `finally` block restores both full and fill device tables, their matching host
mirrors, and the prior bulk eligibility flag. Runtime page-table comparisons
therefore remain coherent even when the first real request has the same mapping
as construction. No optimized decoder or generator code was changed by this
repair.

## CPU verification

Command:

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/full_model/check_active_fill_pages_cpu.py > models/autoports/qwen_qwen3_6_27b/doc/full_model/active_fill_pages_cpu.json
python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/bounded_prefill.py models/autoports/qwen_qwen3_6_27b/doc/full_model/check_active_fill_pages_cpu.py
```

The CPU script extracts and executes the actual two proof helpers from the
source AST; it does not import TTNN or open devices. **6,152 checks passed**:
all logical counts 1–512, virtual-tail aliases, nonzero aligned chunk offsets,
active duplicates, negative/out-of-bounds active addresses, and disjoint real
warmup writes for batches 1/3/32 with small and large physical pools.

Artifact: `doc/full_model/active_fill_pages_cpu.json`. Hardware mixed-prefill
parity and allocation tracking are owned and reported separately by the parent
agent; this document makes no hardware-pass claim.
