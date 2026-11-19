# QSA / PLE setup follow-up audit

CPU-only source audit of `_repack_ple_table`, `MeshPackedEmbedding.from_cache`,
the corrected `MeshPLE` projections, shared RoPE/index arrays, and direct local
QSA state allocation. No TTNN import, device access, reset, implementation edit,
or execution of the parent's full repack test was performed. This is not a
stage-review verdict or a replacement for pending PLE device evidence.

## Findings

No additional blocking defect was identified in these exact changes for the
fixed target model and 1x4 mesh.

The PLE value repair is necessary and correct. The checkpoint inventory records
`ple.key_proj.weight=[10240,2560]` and
`ple.value_proj.weight=[2560,2560]`. Four-way output splitting is appropriate
only for the key projection. Replicating the complete value projection gives
each chip the 2560 features consumed by its stream-specific gate and norm.
The first audit's general statement that PLE projections followed correct
stream ownership was too broad: it missed this value-width defect. This
follow-up supersedes that statement.

## Repack address and padding checks

`local_columns = ceil(local_rows*words_per_row/1024)*32` correctly pads each
rank to whole 32-column tiles before concatenating rank stripes along axis 1.
The target has 80,000,384 local rows:

| Payload | Words/row | Valid columns/rank | Padded columns/rank | Padding columns | Bytes/rank |
|---|---:|---:|---:|---:|---:|
| Codes | 20 | 50,000,240 | 50,000,256 | 16 | 6,400,032,768 |
| Scales | 4 | 10,000,048 | 10,000,064 | 16 | 1,280,008,192 |

Thus the previously rejected `C%32=16` becomes `C%32=0` for both tables.
The 16 padding columns are per rank, not appended only to the global tensor;
`ShardTensorToMesh(dim=1)` consequently yields the intended local tile shape.

For local row `j`, owner `r`, and word `k`, the source reads global linear
word `(4*j+r)*words_per_row+k` via its original 32-stripe coordinates.
The destination stores that word at local coordinates
`((j*words_per_row+k)%32, (j*words_per_row+k)//32)`. The rank's padded column
offset is used only for storage assembly and disappears on mesh sharding.
This matches `PackedEmbedding`'s local integer addressing.

The 262144-row chunks and the final 46464-row tail are divisible by 8, so
both 20-word and 4-word rows form whole groups of 32 words. Every assignment's
column span equals `block.reshape(-1,32).T` exactly. No final valid row is
discarded, and no rank crosses the next rank's padded region. This helper is
model-specific: those divisibility conclusions rely on the fixed row count
enforced by `from_cache`; they are not a claim of arbitrary-table support.

An integer-only script checked rank/stripe/chunk bounds at first, second, and
last chunks and the first/last word of boundary rows for all four ranks and
both payload types. Result: `Integer chunk/rank/stripe bounds: PASS`.

The completed file is atomically installed only after the memmap flush and
close; an interrupted temporary file is not accepted as the final cache.
The v2 name prevents accepting the earlier unpadded representation. Padding
is outside every valid lookup address. Source identity is the original
manifest plus filename, consistent with treating that original completed
cache as immutable. Full file validation remains the parent's CPU test.

## Shared state and allocation checks

- RoPE reuse is keyed by context, rotary dimension, and theta. `QSA` already
  enforces default RoPE and rotary dimension 64, so this covers all varying
  frequency/shape inputs in the supported implementation. The shared tensors
  are only read in the inspected QSA paths.
- The index arrays are keyed by capacity and contain only immutable FP32
  ranges `[0,capacity/4)` and `[0,capacity)`. Their values are exact at the
  advertised context, and their contents do not depend on layer weights or
  request position. Request-owned caches, position, and raw history remain
  newly allocated.
- Direct state construction preserves the baseline capacity/permutation
  checks and fixes page size at the existing 32-token public default.
  It allocates `[capacity/32,1,32,256]` local K/V directly, matching the
  selected local head, page-table interpretation, and local gather kernel.
  The wasteful temporary two-head cache is gone.
- Externally supplied device page tables remain caller-owned, as in the
  baseline. They must be valid, replicated permutations with the documented
  cache remapping; this change does not introduce a host readback.

One memory qualification remains: `MeshQSA.build` still constructs `QSA`
before replacing its fresh cosine/sine tensors with the shared pair. Persistent
storage is shared, but each additional constructor temporarily allocates a new
pair (64 MiB/device at 262144 context). This is a setup peak, not a persistent
per-layer cost or a discovered wrong-output defect. The same factory still
creates then replaces ordinary full QKV/WO weights. Preserve those transient
costs in any claim about construction peak capacity.

## Evidence and remaining checks

Read: the named live implementation sections; `QSA`, `QSAState`, `PLE`,
`PackedEmbedding`, `StripedGather` contracts; functional-decoder
`checkpoint_inventory.json`; searches for writes to shared RoPE/index tensors.
Only capped source reads and integer arithmetic were executed.

Pending target-mesh PLE evidence must still establish correct reconstructed
embeddings, whole-decoder PCC, continuation/history, awkward logical tails,
trace replay, and watcher cleanliness. The successful GDN/QSA device evidence
reported by the parent was not independently rerun or re-audited here.
