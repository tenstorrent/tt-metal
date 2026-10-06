# Runtime fallback and decoder interface source audit

Result: no host tensor conversion, Torch computation, or CPU fallback found
on the inspected selected decoder forward path. No concrete source bug requiring
a fix was found in continuation assembly, slot-loop decode, or the selected
replicated stack interface. This is a bounded source audit, not final stage
review or a claim that pending selected-policy tests have passed.

Inspected runtime SHA256:
`49cd4b3e47aeca24389fd58c29262556d05ec35cf0f0d80c8077c02af51f88e5`.
Inspected unapplied `selected_policy.patch` SHA256:
`6eaebf837567450b625e10e1e8bc24ee7913d9b97e0412218b6e222b05f10686`.
No runtime/default edits or hardware calls were made for this audit.

## Setup versus runtime

| Surface | Host work / ownership | Runtime behavior |
| --- | --- | --- |
| `MultichipDecoder.from_state_dict` | HF state/config inspection, weight conversion/padding, normalization weights, position tables, CCL manager, per-layer objects | Constructor completes before warmup/capture |
| `_SharedMLP.configure_decode` and optional DRAM variant | Torch padding/packing and `ttnn.from_torch`; decode weight copies and configs | `_SharedMLP.__call__` uses TTNN linear, slices, GELU/mul and collective only |
| `_ExpertParallelExperts.__init__` | Torch checkpoint transpose/packing and128 global ownership IDs; upload rank-local32 IDs for row1/32 | `_chunk` uses device gather/layout conversion/sparse matmul/mix; no expert ID/count host read |
| `_HybridExperts` | Retains EP and TP weight layouts loaded at construction | Selects by logical row count; prefill EP, one-row decode indexed TP |
| `_Projection` / optional `_DramAttentionProjection` | Program config and optional setup-owned bank-sharded weight copy | TTNN projection/layout conversion; no weight reload from host |
| Optional `_GatherProjection` | Persistent FP32 gathered buffer and global semaphores allocated at construction | TTNN AGMM plus logical-row slicing; not selected by default |
| Generalized router | Bias, index tables and persistent output buffers initialized in `optimized_decoder.py:822` onward | Device logits, centered softmax/top8, scatter, retained device index handle |
| Native paged attention | Program/compute config created at construction | TTNN paged cache update and native paged SDPA; positions/page IDs remain device tensors |

All direct `torch` imports/operations and `ttnn.from_torch` calls in
`tt/multichip_decoder.py` occur in constructors or configuration methods.
No `ttnn.to_torch`, `.cpu()`, `.numpy()`, `.item()` or `.tolist()` call exists
in that file. Python shape/dtype checks, static metadata branches, loops over
logical lengths/batch slots, and TTNN program configuration do not read tensor
payloads. `ttnn.zeros_like` in routing is a native device op, not a Torch zero
construction. Forward TTNN outputs allocate device temporaries during eager
execution/capture; this is distinct from allocating host tensors or allocating
new buffers while replaying an already captured trace.

The selected patch moves runner-only attention dtype/fidelity overrides into
constructor parameters and `_reduce_attention`. That method performs a device
typecast followed by the existing allreduce binding. It does not introduce a
host branch on tensor values. The BF16 default applies to both layer kinds;
an explicit full-layer dtype override remains optional. Prefill projection
policies remain independently configured. The patch keeps the selected
replicated residual and grouped MoE contract.

## Active experts and persistent state

`OptimizedExperts.enable_indexed_decode` and `_chunk`
(`tt/optimized_decoder.py:172–241`) consume router-retained top8 indices and
invoke indexed sparse projections. Scores and indices are never converted to
Python. EP prefill uses `_active_prefill` at line243: the device routing mask
selects the union of active experts across each32-token block. That union can
be large for a block, but the implementation does not replace gate selection
with unconditional dense128-expert execution. EP decode's variable0–8 count
per rank is handled by native sparse-mask inference, not a hardcoded nnz8.

Each layer constructs its own router buffers, CCL semaphores, and optional AG
persistent buffer. Within a layer the router runs before indexed experts, so
`last_decode_indices` references that invocation's device indices. The
inherited batch loop serializes slots on the same command queue. Sharing one
decoder object between concurrent independent queues is not established by
this code or this stage's one-request contract; use serialized execution or
separate stateful instances. Caller-owned K/V tensors are passed explicitly,
and different layers must receive distinct cache pairs.

## Cache, positions, continuation and stacking

- `OptimizedDecoder._validate_kv_cache` (`optimized_decoder.py:648`) accepts
  matching BF16/BFP8 K/V dtypes with32-token pages. Local head geometry is
  established by the multichip constructor: sliding Q4/KV2; full Q4/KV1 with
  duplicated global KV ownership across rank pairs. Page tables and positions
  are replicated so each chip addresses the same logical token in its local
  head cache. The lightweight validator does not prove caller page IDs or
  position contents are valid; allocation/coverage is an explicit caller
  contract, exercised separately by tests.
- `decode_forward` is inherited from **OptimizedDecoder**, line737, not the
  older BF16-only FunctionalDecoder method. It slices one hidden row, page-table
  row, RoPE position and cache position per slot, then concatenates outputs.
  The slot loop uses static shape metadata; the actual position values are
  read by device embedding/cache kernels. Warmup/capture must use the same
  logical batch and signatures; host input refresh occurs outside replay.
- `_prefill_continuation` (`multichip_decoder.py:776`) selects the requested
  page-table row and absolute device-owned position slices, then performs
  per-token decode. It preserves partially filled pages. Output assembly
  groups32 rows,32 tiles,32 chunks, then at most8 large groups. A final partial
  row group is padded by referencing the last output solely for concatenation
  and sliced back to logical length; no duplicate cache update occurs.
  Checked boundary reasoning for lengths1,32,1024,32768 and nonzero remainders
  shows no empty merge or output-order inversion.
- Fresh long prefill (`multichip_decoder.py:826`) chunks at1024, pads only
  physical computation, and slices logical output after bounded concatenation.
  Chunk page tables are sliced by page index; `user_id` remains passed to
  paged fill. Short sliding tails retain an internal window-sized physical
  buffer and tensor chunk offset. Fresh requests release prior sliding tails;
  continuation and decode use paged cache instead of that tail.
- Selected output is BF16 `[1,1,S,2816]`, replicated on the target1x4 mesh,
  matching the next layer's input directly. The grouped reduction concatenates
  local shared/routed tensors along dimension1, reduces hidden dimension3,
  then splits dimension1; this preserves separate post-normalization operands.
  Incompatible grouped+sharded residual configuration is rejected at setup.

RoPE/page coverage requirements remain those documented in
`OptimizedDecoder.prefill_forward`: absolute RoPE tables covering internal
padding, cache pages covering kernel read padding, and valid request-owned
padding rows. Arbitrary valid logical lengths are accepted; internal32/1024
alignment is not a public restriction.

## Dynamic guard evidence and remaining gates

`tests/runtime_audit.py::device_only` combines `TorchDispatchMode` rejection
with patched TTNN host boundaries (`from_torch`, `to_torch`, `as_tensor`,
`to_device`, `zeros`, `ones`, `full`, `arange`). Paired layer tests and the
stack/contract harnesses wrap eager forward, warmup and capture in this guard.
Host fixture creation, input refresh and result/cache conversion intentionally
happen outside those regions. Trace replay runs the already captured device
commands; the guard is not a claim that test harness I/O is device-only.
The guard is useful executed-path evidence, not proof of every possible TTNN
internal branch; source inspection covers the selected calls above.

Required source fixes from this audit: **none identified**. Remaining execution
gates are the selected patch's actual stack, batch/continuation, long-context,
changed-position/page-table and Watcher runs. Old candidate results do not
substitute for selected-policy validation. Sliding BFP8 replay failure remains
an independent numerical/state investigation (`AUTOFIX_bfp8_ccl.md`); the
selected patch's BF16 default does not silently accept that failing candidate.
