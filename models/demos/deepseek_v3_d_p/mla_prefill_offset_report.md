# MLA prefill at a non-chunk-aligned offset

## Verdict

For a Galaxy stage configured as **SP8 x TP4**, the production prefill chunk is
5120 tokens, or 640 tokens per SP rank. A full 5120-token chunk beginning at
absolute offset 960 covers logical positions `[960, 6080)`.

MLA's intended and test-validated representation is an **offset-rotated,
block-cyclic, chip-major** activation. The placement is:

| SP rank | Logical token positions in its 640 local rows |
| ---: | --- |
| 0 | `5120..5759` |
| 1 | `960..1279`, then `5760..6079` |
| 2 | `1280..1919` |
| 3 | `1920..2559` |
| 4 | `2560..3199` |
| 5 | `3200..3839` |
| 6 | `3840..4479` |
| 7 | `4480..5119` |

Every row is repeated across the four TP columns, but each TP device owns only
one quarter of the hidden dimension. TP does not further divide the 640 token
rows.

There is an important integration gap at the PR head examined here. The model
test explicitly gathers tokens in the table's rotated order before upload, but
the production producer slices `[actual_start:actual_start + 5120]` in natural
order and only reshapes it. Therefore, the MLA core's offset machinery is
present, while the current producer-to-runtime path does not construct the
required input order for offset 960. Chunk-aligned offsets hide this mismatch.

## Configuration and vocabulary

The production constants are defined as SP=8, global chunk=5*1024, and
per-chip chunk=640 in
[`utils/chunk_config.py`](utils/chunk_config.py#L4-L10).

This report uses:

| Symbol | Meaning | Value |
| --- | --- | ---: |
| `P` | SP factor | 8 |
| `G` | global prefill chunk | 5120 tokens |
| `C = G/P` | local rows per SP rank | 640 tokens |
| `S` | absolute start / prior valid KV length | 960 tokens |
| `E = S+G` | end of this full chunk | 6080 tokens |

`actual_start` is an absolute global KV position, not a per-chip row and not a
chunk index. The runtime describes `[actual_start, actual_end)` this way in
[`tt_prefill_runtime.py`](tt/tt_prefill_runtime.py#L548-L578). Offset 960 is
tile-aligned (`960 / 32 = 30`), which satisfies MLA's required 32-token
alignment ([`mla.py`](tt/mla/mla.py#L887-L896)).

## Why offset 960 rotates the chunk

One cache slab contains `G=5120` global positions and contributes `C=640`
contiguous positions to each SP rank. For any start `S`, MLA computes:

```text
boundary_slab   = floor(S / G)
boundary_chip   = floor(S / C) mod P
boundary_offset = S mod C
```

For `S=960`:

```text
boundary_slab   = 0
boundary_chip   = 1
boundary_offset = 320
```

The first 960 cached tokens have already consumed:

```text
SP0 slab 0: 640 / 640 rows  (global 0..639)
SP1 slab 0: 320 / 640 rows  (global 640..959)
SP2..SP7:      0 / 640 rows
```

The next free cache cell is therefore SP1's local row 320. Filling 5120 new
positions continues from there, wraps around SP7 to SP0's next slab, and ends
after another 320 rows on SP1's next slab.

### Natural-time view

Read left to right to see the logical sequence:

```text
global:  960       1280       1920       2560       3200       3840       4480       5120       5760       6080
         |--320--| |---640---| |---640---| |---640---| |---640---| |---640---| |---640---| |---640---| |--320--|
SP:         1           2           3           4           5           6           7           0          1
```

### Galaxy SP8 x TP4 view

Each cell names the token positions; `H0..H3` are quarters of the hidden
dimension:

| SP row | TP0 (`H0`) | TP1 (`H1`) | TP2 (`H2`) | TP3 (`H3`) |
| ---: | --- | --- | --- | --- |
| 0 | `5120..5759` | `5120..5759` | `5120..5759` | `5120..5759` |
| 1 | `960..1279; 5760..6079` | same | same | same |
| 2 | `1280..1919` | same | same | same |
| 3 | `1920..2559` | same | same | same |
| 4 | `2560..3199` | same | same | same |
| 5 | `3200..3839` | same | same | same |
| 6 | `3840..4479` | same | same | same |
| 7 | `4480..5119` | same | same | same |

"Same" means the same token rows with a different hidden-width shard, not a
full hidden-state replica.

## Exact local-row and cache-row mapping

The cache's inverse block-cyclic mapping is:

```text
global_position(local_cache_row lr, SP rank c)
    = floor(lr / C) * G + c*C + (lr mod C)
```

For offset 960, the writer chooses these local cache starts:

| SP | Input local rows | Local cache rows written | Global positions represented |
| ---: | --- | --- | --- |
| 0 | `0..639` | `640..1279` | `5120..5759` |
| 1 | `0..319` | `320..639` | `960..1279` |
| 1 | `320..639` | `640..959` | `5760..6079` |
| 2 | `0..639` | `0..639` | `1280..1919` |
| 3 | `0..639` | `0..639` | `1920..2559` |
| 4 | `0..639` | `0..639` | `2560..3199` |
| 5 | `0..639` | `0..639` | `3200..3839` |
| 6 | `0..639` | `0..639` | `3840..4479` |
| 7 | `0..639` | `0..639` | `4480..5119` |

The Python oracle `rotated_chip_positions` implements this mapping in
[`utils.py`](tt/mla/utils.py#L83-L111). The device writer independently uses
the same boundary-slab/chip/offset equations in
[`writer_update_padded_kv_cache.cpp`](../../../ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp#L100-L122).

For this full chunk, every SP rank has 640 real rows. A partial tail would still
use the same position map, but rows with positions at or beyond `actual_end`
would be padding; the per-rank real counts are derived from the position map in
[`utils.py`](tt/mla/utils.py#L114-L133).

## What MLA does with the layout

1. The caller gathers the logical token IDs at the positions in the first
   table, flattens them chip-major, reshapes to `[8, 1, 640]`, and shards tensor
   dimension 0 across SP. The chunked transformer correctness test does exactly
   this in
   [`test_prefill_transformer_chunked.py`](tests/test_prefill_transformer_chunked.py#L558-L578).
2. Embedding and the transformer preserve these local token rows. Across TP,
   the activation has per-device shape conceptually
   `[1, 1, 640, hidden_size/4]`.
3. Dense MLA applies RoPE using an indexed whole-cache table. The RoPE reader
   derives the same per-SP local offset as the cache writer, so each rotated
   row gets the correct absolute position
   ([`mla.py`](tt/mla/mla.py#L804-L831)). Kimi-K3 MLA is NoPE, so this specific
   transform is the identity; the row-placement contract still applies to the
   cache and attention.
4. `update_padded_kv_cache` writes every rank's 640 input rows at its computed
   local start, overwriting old pad before spilling to the next slab
   ([`mla.py`](tt/mla/mla.py#L898-L908) and
   [`mla.py`](tt/mla/mla.py#L1175-L1196)).
5. `ring_mla` receives the same `actual_start`; its gather and causal-mask
   readers interpret the block-cyclic cache without first restoring a natural
   host layout ([`mla.py`](tt/mla/mla.py#L910-L935)).

The important invariant is that **activation row, positional metadata, cache
destination, and attention interpretation must all name the same global token**.
Sharing the boundary equations is how the MLA core maintains that invariant.

## Observed production integration gap

The runtime API says its first-rank input must already be "block-cyclic,
chip-major" ([`tt_prefill_runtime.py`](tt/tt_prefill_runtime.py#L569-L578)). Its
helper, however, calls `prepare_prefill_input_tensor(..., is_balanced=False)`
([`tt_prefill_runtime.py`](tt/tt_prefill_runtime.py#L378-L392)), and that path
only reshapes the supplied list into SP shards; it does not use `actual_start`
or gather rotated positions
([`input_prep.py`](tt/runners/input_prep.py#L23-L51)).

The producer likewise:

1. loads natural prompt token IDs from `metadata.json`
   ([`runner_utils.py`](../common/prefill/runners/runner_utils.py#L175-L182));
2. slices the natural interval
   `pool[actual_start:actual_start + CHUNK_SIZE]`
   ([`prefill_producer.py`](../common/prefill/runners/prefill_producer.py#L1453-L1462));
3. only reshapes that interval to `[SP, 1, chunk_local]`
   ([`prefill_producer.py`](../common/prefill/runners/prefill_producer.py#L205-L215)).

If that current path sends natural `[960,6080)`, the values and the positions
that MLA assigns to them diverge immediately. For example:

| Current input shard | Values received | Positions used by RoPE/cache writer |
| ---: | --- | --- |
| SP0 | `960..1599` | `5120..5759` |
| SP1 rows `0..319` | `1600..1919` | `960..1279` |
| SP1 rows `320..639` | `1920..2239` | `5760..6079` |
| SP2 | `2240..2879` | `1280..1919` |

The remaining ranks are shifted similarly. Offset 0 or any multiple of 5120
does not expose the issue because the rotation degenerates to a plain reshape.

This is a source-level diagnosis, not a Galaxy execution result. The device
correctness tests demonstrate the MLA core with correctly reordered input; an
end-to-end production test at offset 960 is still needed to prove the producer
fix and hardware behavior.

## Consequence for KDA offset work

MLA can tolerate this physical wrap because attention names cache positions
explicitly and its readers know the block-cyclic inverse. KDA recurrence is
order-sensitive: its distributed prefix composes SP partitions in rank order,
and its convolution carry also flows between temporal partitions. At offset
960, chronological order is:

```text
SP1(first 320) -> SP2 -> SP3 -> SP4 -> SP5 -> SP6 -> SP7 -> SP0 -> SP1(last 320)
```

That is not the simple `SP0 -> SP1 -> ... -> SP7` order assumed by a normal
contiguous sequence partition, and one physical SP rank appears twice. KDA
therefore cannot gain correct offset support by accepting `actual_start` as
metadata alone. Its input/carry contract must explicitly handle the boundary
rank split or repack the current turn into chronological SP partitions before
the recurrence. This is the central design constraint for the exploration
branch.

## Host-side validation performed

The repository's `rotated_chip_positions` oracle was executed for
`sp=8`, `chunk_local=640`, and `actual_start=960`. The check asserted that the
flattened result, when sorted, equals every integer in `[960,6080)` exactly
once. It also returned real-token counts `[640,640,640,640,640,640,640,640]`.
No device test was run; this workspace has no Galaxy hardware attached.
