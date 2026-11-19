# QSA / PLE source audit

CPU-only independent audit of the live `tt/multichip_decoder.py`, 2026-11-13.
No TTNN import, device open, hardware execution, reset, or implementation edit
was performed. This is a source audit, not stage acceptance or device evidence.

## Required repair

### P1: local selected-KV validator rejects every actual QSA output

`LocalSelectedPagedKV.__init__` tests `selected_ids.shape[1] != 1080` (line
1062 at audit time), while `fused_selected_token_ids`, the error text, output
allocation, SDPA mask, and kernel all require 2080. Every valid QSA forward
therefore raises before launching the gather. Merely changing the selector to
1080 would be unsafe: the kernel unconditionally reads 65 UINT32 tiles and
produces 2080 slots.

Exact repair: change the one validator constant from 1080 to 2080. Keep all
other widths at 2080. Validate the local gather first at Q=1 and an awkward
Q such as 3 or 31, with a nonidentity page permutation, invalid sentinel
slots, and selections crossing page boundaries, then run complete QSA parity.

A CPU AST evaluation of the actual condition reported:

```text
selected_width=1080: rejects=False
selected_width=2080: rejects=True
```

## Concrete public-contract regressions

`MultichipDecoder.prefill_forward` accepts only tuple batches, whereas the
optimized baseline inherits `(tuple, list)` support and `_validate_batch`.
Passing a valid baseline list batch now reaches `hidden_states.shape` and
raises `AttributeError`. The tuple branch also drops explicit request-count
and token-tuple length validation. Exact repair: recognize `(tuple, list)` and
call the inherited `_validate_batch` before iterating, retaining `zip(...,
strict=True)` if desired. This is host-only shape validation and adds no
runtime tensor read.

The public prefill path drops the baseline's `length <=
config.max_position_embeddings` check. Restore this check so oversized input
fails before state mutation. Decode delegates directly to prefill and thus
accepts arbitrary T, whereas the baseline's documented decode contract is
T=1. Either restore the one-row validation or document an intentional extended
decode API; this is not itself a valid-input accuracy failure.

## Memory / performance concern to resolve in the target evidence

The new prefill list retains every chunk output and then concatenates them,
reversing the baseline's bounded output assembly. For the local BF16 width
2560, retained chunks occupy 160 MiB at 32K, 320 MiB at 64K, and 1280 MiB at
262144 tokens, per device. The final concat needs an equally large output;
recursive concat grouping can add temporary storage. The input is additional.
This is not a proved OOM on the target board, but it must be included in the
activation/context budget rather than budgeting one <=32-row output.

Exact lower-memory repair: allocate one `[1,1,T,2560]` destination and use
`chunk_output.write_token_chunk(chunk, destination, start)`, which already
supports arbitrary tile-aligned widths and logical 1..32-row chunks, releasing
each chunk after its write. Unlike `chunk_input.read_token_chunk`, that writer
does not hardcode width 10240. Preserve the mesh's per-stream ownership.
The native slice path itself supports these aligned starts and unaligned ends;
I found no source basis to claim it exposes an aligned-only public contract.

## Contracts checked without a discovered arithmetic/address defect

- QSA query ownership is six adjacent query heads per chip. The projection
  partitions include each head's interleaved 256 query + 256 gate features.
  Chips 0/1 own KV head 0 and chips 2/3 own KV head 1, matching 12:1 GQA.
  Local projected QKV width is 3072+256+256=3584. Local attention output is
  6*256=1536; WO splits its input dimension by the same head ownership.
- The QSA gather's cache tile expression is correct for the enforced one-head
  cache: `((physical_page * (page_size/32) + token/32) * 8 + channel)`.
  Output tile order is `[query,1,slot_tile,channel_tile]`; the UINT32 selector
  reads use query as tile row and slot as tile column. Sentinel slots are
  zeroed. Page-table remapping is applied before cache addressing.
- Current positions remain replicated device INT32 state. QSA updates each
  valid row, preserves the prior three raw indexer keys, and advances once
  by the logical chunk size. No aligned-length requirement was introduced
  in that source path.
- PLE projections split output streams, giving each chip a full 2560-wide
  stream and full-width local RMS statistics. The local nine-row convolution
  state and norm/conv weights follow the same stream ownership. Embeddings
  are replicated after lookup reduction, as required by those projections.
- The packed embedding ownership `owner=id&3`, `local=id>>2` matches the
  setup partition `linear[r::4]`. A small integer check of boundary IDs
  0,1,2,3,4,31,32,rows-4,rows-1 and word lanes 0/19 confirmed
  `owner+4*local=id` and `column*32+stripe=local*20+lane`. There are exactly
  80,000,384 local rows. Scale storage is four words per row; the called
  `quantization_spec` enforces group10 for scale-LUT encoding. I found no
  UINT32 overflow in local word/scale addressing.
- Masked local embedding values reduce to a single contributing chip per
  ID. The FP32 collective sum does not introduce accumulation error for one
  finite BF16 contributor plus zeros. Other ranks nevertheless perform an
  unnecessary local lookup before masking; whether to skip those reads is a
  measured optimization question, not an address correctness repair.

## Remaining runtime questions

Source inspection cannot establish PCC after row-parallel WO accumulation,
PLE unfused versus optimized fused nonlinear rounding, mesh generic-op address
rebinding, warmed trace lifetime safety, or watcher cleanliness. Those need
the stage's actual target-mesh checks. The QSA constructor first materializes
the full baseline projections and `make_state` first allocates two local KV
heads before replacing them with one; account for this transient construction
peak, or construct the final local tensors directly if capacity requires it.

## Inspected sources

`tt/multichip_decoder.py`, `optimized_decoder.py`, `fused_decoder.py`,
`functional_decoder.py`, `qsa.py`, `qsa_chunked.py`, `qsa_selected_cache.py`,
`ple.py`, `packed_embedding.py`, `packed_embedding_cpu.py`, `striped_gather.py`,
`ngram_hash.py`, `decoder_components.py`, `chunk_input.py`, `chunk_output.py`;
native slice and concat operation sources; optimized-decoder README.

Tools: capped `sed`/`rg` source reads; Python `ast` validation-condition
evaluation and integer-only address/size arithmetic. No external sources.
