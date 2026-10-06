# V6 source delta and evidence boundary

Current runtime: `b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846`.
Exact archived v5 source: [runtime_v5_before_review.py.txt](runtime_v5_before_review.py.txt),
SHA256 `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`.
This audit reads source and existing artifacts only. It executes no accelerator
code and changes no runtime or tests.

Only four of 40 class methods differ. [The JSON proof](source_delta_v6.json)
contains the exact text diff, hashes of all 36 unchanged method ASTs and an
inverse transformation that removes only the specified delta from each changed
method. Every resulting method AST exactly matches the archived v5 version.
Comments and line numbers are excluded from AST equality; the full text diff
remains available separately.

| Changed method | Exact change | Retained behavior |
| --- | --- | --- |
| `OptimizedDecoder.prefill_forward` | Pass `retain_prefill_tail = start + valid < length` to each fresh-prefill chunk. | Chunk construction, cache allocation/validation, valid lengths, outputs, prefix continuation and decode calls are unchanged. |
| `OptimizedAttention.prefill` | Create the K/V tail pair only when another chunk consumes it; otherwise assign `self.tail=None`. Direct callers omitting the keyword retain the old default. | Current chunk's projection, cache fill, sliding attention, history inputs and output projection are unchanged. Every nonfinal tail is retained as before. |
| `ConfiguredChunkedPrefillAttention.__init__` | Construct a second SDPA config at setup, with Q/K capped at 128 and the same grid/exp-approx policy. | Existing primary program and compute config remain identical. Default full attention is primary Q64/K256, boundary Q64/K128. |
| `ConfiguredChunkedPrefillAttention.__call__` | Compute the primary program's rounded read extent and select the boundary config only when it exceeds logical page-table capacity. | All sufficient-capacity cases use the identical primary program, query padding, compute object, scale, cache tensors and geometry. |

The factory, its defaults/auto resolution, direct/minimal QKV, output projection,
router, experts, shared MLP, normalization, RoPE, cache validation and public
decode methods are AST-identical. No weight dtype, fidelity, activation cast or
projection arithmetic changes. The full-attention boundary changes SDPA block
geometry only when the previous rounded read would exceed capacity; it retains
HiFi2, FP32 destination, full destination synchronization and the existing dtype
boundaries. Equality of unaffected source is evidence about the change scope,
not a claim that old hardware results executed the new source.

## Logical page-table boundary

For each paged-prefill slice, the source calculates:

- `query_end = base_offset + start + length + (-length) % primary_Q`;
- `read_end = query_end + (-query_end) % primary_K`;
- `capacity = page_table.shape[-1] * k_cache.shape[-2]`.

The public contract uses 32-token cache pages and 128-token capacity rounding.
At S=1025 with 36 pages, capacity is 1152. The final query is physically 32 rows:
Q64 padding gives `query_end=1088`; primary K256 would read through 1280.
Boundary K128 instead reads through 1152. The primary program is retained when
its rounded extent fits; the change does not globally replace K256 by K128.

CPU-only execution of the exact new scalar source statements covers 6,144 cases:
all valid final lengths 1..1024 at offsets 1024, 2048 and 261120, under both minimum
128-rounded and larger 1024-rounded capacities. All selected read extents fit;
1,536 cases select the boundary config. Every sufficient-capacity case retains
the original program and padding. This proves the arithmetic for default
Q64/K256 geometry with 32-token pages, not arbitrary environment overrides or
hardware behavior.

The parent-run [tight-cache test](tight_cache_v6_layer5_1025.json) records this
new source, real weights and the actual-input fixture hash. Both observed
paged-prefill calls use Q64/K128, padded query 64, read end 1152 and capacity 1152.
Prefill PCC is 0.99904124467498; decode PCC is 0.99916185200719. The runtime audit,
program-cache guard, capacity assertions and repeated traced decode all pass.
Seven additional current controls cover lengths1088,1089,1152(prefill-only and decode),1153,1281 and2049. All pass HF checks, program-cache guards and instrumented capacity checks; both K128 and K256 branches execute. The1281 case also passes Watcher. Their reports, fixture content hashes and completed commands are checked by the [v6 manifest](validated_v6_validation_summary.json).

## Storage effect

Checkpoint, expert, shared-prefill, QKV/output, router, KV-cache and RoPE tensor
payloads are unchanged. The boundary config adds one small host config object;
there is no new setup tensor. A new SDPA program may be cached when that branch
is exercised; its code/CB storage is outside the resident tensor accounting.

For sliding attention, each removed final clone has BF16 shape
`[1,8,T,256]`, where `T=min(1024, physical_K_rows)`. The eliminated pair contains
`2*8*T*256*2 = 8192*T` bytes, at most **8,388,608 bytes (8 MiB)** per request.
Each clone is 4 MiB at T=1024. A single short chunk uses its 32-rounded physical
length; later short sliding chunks retain the existing physical window padding,
so their final pair would also have been 8 MiB. Full attention has no sliding-tail
pair and therefore no corresponding tensor-payload change.

These are two skipped `ttnn.clone` calls and their private per-request tail
allocations. Nonfinal tails are still needed. This is not a measured allocator
peak or a change to maximum-context cache sizing; it does not imply a specific
native-operation count or latency saving before profiling.

## Validation and profile attribution

The complete 18-command v5 correctness summary, candidate measurements and both
v5 native profiles remain attributed to `169c0d97…`. The JSON records their
original report/raw-file hashes. They are not relabeled as v6 measurements,
even though projection/decode code and primary full-chunk math are unchanged.
The [v6 manifest](validated_v6_validation_summary.json) has status
`current_targeted_and_inherited_gates_passed`. It binds 14 completed journal
commands plus the separately recorded initial1025 control to `b585a21f…`:
eight tight-cache cases total, both nine-request reuse catalogs with full trace
allocation tracking and unchanged live-trace program-cache counts, both
4096-prefill/128-decode headline and Watcher runs, and four raw pytest cases.
All current recorded accuracy gates pass at the unchanged0.995 threshold.

B32 heterogeneous requests, prefix/BF16-cache compatibility, the262144/262143
sampled maximum-context checks and1025/512 stress remain attributed to v5.
Their broad-suite inheritance is supported by the exact unchanged-source map
and current tests of both changed behaviors; they are not mandatory outstanding
v6 reruns. The CPU manifest rechecks all16 archived public report hashes and
the archived stress gate. This accepts scoped evidence inheritance without
claiming an18-command v6 campaign. Current native profiles remain separate;
old native rows and timings are never relabeled with the new hash.
