# Current runtime fallback, precision, trace and ownership audit

Result: no concrete host fallback, trace-safety defect, or buffer-ownership defect
was found in the selected replicated decoder paths inspected below. This is a
bounded source audit of the final policy, not independent final stage approval
or a proof of native kernel correctness. No TTNN import, hardware access,
runtime edit, or test edit was performed for this audit.

Audited runtime SHA256:
`a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255`.
The superseded pre-integration audit is preserved verbatim in
[runtime_fallback_audit_49cd4b3e.md](runtime_fallback_audit_49cd4b3e.md).
Its BF16-both policy and pending-failure statements are historical.

## Effective selected policy

The constructor, not test-only flags, selects the following policy.
`None` means per-kind selection for `shared_geometry`, `sharded_decode_rope`,
and `moe_ccl_bfp8`; it does not mean those features are disabled.
Sources are [multichip_decoder.py](../../tt/multichip_decoder.py), particularly
`from_state_dict` lines 583–884, `_LocalAttention.rotary` lines 173–233,
`_SharedMLP` lines 248–336 and `_reduce_moe_pair` lines 1006–1017.

| Selected surface | Sliding attention | Full attention |
| --- | --- | --- |
| Mesh / residual / output | 1x4 Linear, one link; replicated BF16 `[1,1,S,2816]` | Same |
| Experts / tail | Hybrid EP prefill + indexed TP decode; grouped MoE reduction; fused tail | Same |
| Local Q / KV heads | 4 / 2, head dimension 256 | 4 / 1, head dimension 512; KV ownership replicated across rank pairs |
| Attention QKV / WO weights | BFP8 | BFP8 |
| Decode QKV / WO compute | LoFi, FP32 destination and output, approximation off, packer accumulation off | Same |
| Decode QKV program | Grid 8x4, K22, N2 | Grid 8x6, K22, N2 |
| Decode WO program | Grid 11x8, K32, N1 | Grid 11x8, K8, N1 |
| Prefill QKV | Minimal matmul, K8/M4, HiFi4, FP32 destination/output | Minimal matmul, K16/M2, HiFi4, FP32 destination/output |
| Prefill WO | Minimal matmul K8, LoFi, FP32 destination/output | Same |
| Attention CCL payload | BF16 | BFP8 |
| Grouped shared+routed CCL payload | BFP8, cast back to BF16 before tail | BF16 |
| Decode RoPE | Native sharded D256; activation, cosine and sine cast to BF16; BF16 destination; cast result back to original FP32 dtype | Inherited interleaved fused RoPE, FP32 operands/tables and destination |
| Prefill RoPE | FP32 device multiply/add rotate-half path | Inherited interleaved fused FP32 path |
| Native decode SDPA | BF16 Q; HiFi4, FP32 destination, full sync, approximation off | BF16 Q; LoFi, FP32 destination, full sync, approximation off |
| Prefill attention | BF16 Q/K/V; LoFi, FP32 destination, full sync | BF16 Q/K/V; HiFi2, FP32 destination, full sync |
| Cache default | BFP8 K/V, 32-token pages; BF16 is also accepted | Same |
| Decode router projection | Original BF16 weight, FP32 activation/output and destination, HiFi4, K22, grid 4x1 | Same formats, LoFi, K44, grid 4x1 |
| Decode gate | Center FP32 logits, round to BF16, native top8/softmax on core (10,9) | Same |
| TP decode expert GU / down weights | BFP4 / BFP4; GU packed directly from raw checkpoint during setup | BFP4 / BFP4 via setup device conversion of imported BF16 weights |
| TP decode expert activation / compute | Input cast to BFP8; GU/down LoFi, BF16 destination/output; indexed top8 | Same |
| TP decode expert geometry | GU 6x2/K44/N1, down 11x8/K6/N1; mix K1 after indexed setup | Same |
| TP decode weighted mix | HiFi4 with FP32 destination; BF16 output | HiFi4 with BF16 destination/output |
| EP prefill expert GU / down | BFP8 / BFP4, LoFi, BF16 activation and output | BFP4 / BFP4, LoFi, BF16 activation and output |
| Shared decode GU / down weights | BFP4 / BFP8 | BFP4 / BFP4 |
| Shared decode compute / geometry | LoFi, BF16 destination/output, packer accumulation on; geometry 2: GU 9x2/K88/N2 | Same compute; geometry 1: GU 11x4/K44/N1 |
| Shared decode down geometry | 11x4/K17/N2 | Same |
| Shared prefill | Imported BF16 interleaved weights; native linear defaults | Same |
| Input / post-attention / common normalization | FP32 values and destination, HiFi4, approximation off; sharded for one-row hidden-width input | Same |
| Head normalization | FP32 SFPU variance/reduction and fused rsqrt | Native FP32 RMSNorm; common K/V normalization for global tied KV |
| Fused tail normalization | BF16 input/output, native RMSNorm default HiFi4, approximation on, FP32 destination off | Same |

The attention and grouped-MoE choices apply to both prefill and decode in their
respective layer. `full_attention_ccl_dtype=None` explicitly inherits
`attention_ccl_dtype`; the actual constructor default is BFP8, so the default
full layer does not inherit sliding's BF16 choice. Optional sharded residuals,
AGMM, attention DRAM projections, EP-only execution, shared DRAM decode, and
runner geometry/projection/split overrides are not selected defaults.

Precision qualifications:

- Sliding sharded RoPE changes arithmetic and table precision. Returning an
  FP32 tensor does not restore the discarded precision. See
  [AUTOFIX_sharded_rope.md](AUTOFIX_sharded_rope.md) for the homogeneous-BF16
  and native DST-capacity controls; full-layer D512 adaptation remains opt-in.
- Sliding raw GU packing at lines 810–828 is setup-only: checkpoint gate/up
  transpose, zero-pad 704 to 768 features, pack `[gate_i,up_i]` per TP rank,
  then upload directly as BFP4. This preserves the measured host packing
  route; substituting BF16-to-BFP4 device conversion is not equivalent.
  The EP prefill object retains its separate BFP8 gate weights. The full
  layer does not take this raw-GU branch.
- Attention CCL output is promoted by the existing FP32 normalization path;
  promotion does not reverse BF16/BFP8 communication rounding. Grouped MoE
  explicitly restores BF16 before separate shared/routed normalization.
- The constructor's main HiFi4 compute object does not override every call.
  Shared prefill uses imported linear callables without a compute override,
  and the fused tail uses native RMSNorm defaults. The latter is explicit in
  `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:16–19,62`.

## Selected forward calls and host-boundary inspection

An AST/text scan of the current multichip source places every Torch operation,
`from_torch` call, and raw checkpoint transformation in a constructor or
setup helper. It contains no `to_torch`, `.cpu()`, `.numpy()`, `.item()` or
`.tolist()` call. The imported layer's `layer_scalar.item()` is setup-only
(`models/demos/gemma4/tt/layer.py:121–125`). The following inherited calls were
also read; the conclusion is not based only on the top-level file.

| Forward surface | Inspected implementation and result |
| --- | --- |
| Main residual graph | `MultichipDecoder._forward:1074–1099`: TTNN normalize, attention, FP32 residual, device router, shared/experts, grouped reduction and fused tail. Output is directly consumable by the next layer. |
| QKV / WO | `_Projection:80–120`, `_LocalAttention.project:234`, `OptimizedAttention.project:1319–1367`, `MinimalPrefillProjection:1447–1499`: setup-owned weights/configs, device projections and movement. Head split/concat helpers in `models/demos/gemma4/tt/attention/operations.py:80–128,513–578` call native TTNN operations; selected decode projection supplies its own concat path. |
| Heads / normalization / RoPE | `FusedAttention.heads:566–594`, `FusedAttention.rotary:528–536`, `OptimizedDecoder.normalize:786–816`, `FusedDecoder.normalize:155–167`, `precision_ops.py`: device arithmetic, static shape/format dispatch, no payload readback. Sharded RoPE constructs compute/memory metadata during forward but uploads no host tensor. |
| Decode attention / cache | `FusedAttention.decode:609–643`, `OptimizedAttention.cache_cast:1369–1371`, `NativePagedAttention:1135–1166`: embedding reads device position tensors, fused cache writer updates caller-owned K/V, then native paged SDPA. BF16 cache update rows are repacked into the destination cache format by the writer. |
| Prefill attention | `OptimizedAttention.prefill:1373–1444`, `ConfiguredChunkedPrefillAttention:1055–1132`: device paged fill, native sliding SDPA with cloned history, or native chunked full SDPA. The selected sliding override does not call the older imported sliding helper's host-zero path. Full chunks use scalar offsets; nonzero public prefix continuation uses decode instead. |
| Decode router | `GeneralizedRouter:917–981`: TTNN linear, max/subtract, BF16 cast/pad, native gate, device gather/scatter. No Python top-k decisions or expert-ID extraction. One-row gate buffers were moved to core (10,9) during setup. |
| Prefill router | `GeneralizedRouter:918–919` delegates to `BroadcastRouter:407–430`: TTNN FP32 linear, topk, softmax, BF16 routing/scatter. `routing_precision.Router:53–64` supplies setup-owned scale/projection config. |
| Experts | `_HybridExperts:567–574`, `PackedExperts.__call__:344–354`, `_ExpertParallelExperts._chunk:529–539`, `OptimizedExperts._chunk:193–241` and `_active_prefill:243–271`: device routing indices/masks and sparse projections. EP prefill computes each 32-token block's union of active experts, with native nnz inference; indexed decode uses eight selected IDs, without a host route/count read. |
| Shared MLP | `_SharedMLP.__call__:307–336`: device linear/slices/fused GELU-mul/linear. Prefill delegates to imported `SharedMLP` callables at `models/demos/gemma4/tt/shared_mlp.py:146–201`; MoE disables that loader's DRAM-sharded branch, so selected prefill is native linear over setup-owned BF16 interleaved weights. |
| Collectives | `_reduce_attention:1019`, `allreduce:1022`, `_reduce_moe_pair:1006`, actual `models/demos/gemma4/config.py:96–133`: DRAM input, native reduce-scatter on dim3, native all-gather on dim3, cluster axis1. No CPU summation; no external/legacy helper substitution. No pad_size is supplied, so the helper's optional forced-free padding branch is inactive. |
| Tail | `_fused_tail:1101–1134` and imported `RMSNorm.forward`: native RMSNorm/add/scalar activation only. Three hidden-width sharded norm configurations are initialized at model setup, before warmup/capture. |

Python comparisons on tensor shape/dtype/layout, program selection, fixed loops,
and reading `device().arch()` are metadata operations. `ttnn.zeros_like` used
by routing is a device operation. New device temporaries are expected during
eager execution and capture; replay executes recorded commands without
re-running Python configuration or allocating new host/device input buffers.

## Trace inputs, mutable state and ownership

- `OptimizedDecoder.decode_forward:737–774` is the inherited public decode
  method. Batch >1 records a static slot loop, slicing one hidden row,
  page-table row, uint32 RoPE position and int32 cache position per slot.
  It never extracts those position/page values to Python. Warmup/capture must
  keep the same logical batch, tensor signatures, topology and static config.
- Caller-owned hidden input, position tensors, page table, RoPE tables and K/V
  remain alive through replay. `run_multichip_decoder.py:532–557` allocates
  these inputs before two warmups and capture. It refreshes input/positions
  into the same addresses before `execute_trace`, not during captured forward.
  The stack harness similarly records both layers and keeps distinct cache,
  position and page-table objects per layer (`test_multichip_stack.py:104–196`).
- Every decoder constructs its own `CCLManager` and router buffers. The CCL
  manager creates six RS, four AG and two barrier global semaphores at setup
  (`models/demos/gpt_oss/tt/ccl.py:8–86`). Selected slot forward performs two
  allreduces; their ping-pong semaphore choices and barrier addresses are
  recorded during capture. Replay does not need Python counters to advance.
  No model code resets or reallocates these semaphores during replay.
- `GeneralizedRouter` owns persistent bias/index/output/output-index tensors.
  It converts gate outputs to interleaved device tensors, slices indices and
  stores `last_decode_indices`; the immediately following indexed expert call
  consumes that handle. The next slot is serialized on the same queue.
  Trace replay records these device dependencies; it does not re-enter the
  Python `last_decode_indices` setter. No forced deallocation of these
  persistent buffers appears on the selected forward path.
- Sparse decode raw GU remains a local reference through the chunk return;
  its reshape aliases the same buffer. Gate/up slices are native outputs.
  Prefill's explicit deallocations release private GU/gate/up/hidden buffers
  after their consumers are enqueued; its in-place `weighted` output aliases
  the private down projection, not model weights or caller cache/input.
  Full-prefill chunk helpers likewise release their private query/output
  slices after enqueued consumers. No reviewed path force-frees caller K/V,
  RoPE tables, page-table inputs, persistent weights or router buffers.
- Sliding `tail` holds cloned BF16 K/V only between chunks in one fresh
  prefill. Fresh requests clear it; the final chunk stops retaining it, and
  continuation/decode read paged cache. These references are not an
  unbounded per-step history. Bounded output assembly clears lists after
  merging; it avoids force-freeing a result when a one-input merge aliases
  an element (`multichip_decoder.py:886–1004`).
- This ownership reasoning covers serialized execution of a decoder instance.
  Concurrent independent queues sharing its router buffers, cache or CCL
  state are not established by this implementation or stage. Separate
  requests/layers must retain their own intended cache ownership.

The (10,9) placement is an integrated, measured model-local workaround. It is
outside the identified expert and hidden-width normalization math grids;
this audit does not claim the core is unused by every other op or that native
LLK state corruption has been proved. Historical failures and placement
controls remain in [AUTOFIX_full_router1_bfp8.md](AUTOFIX_full_router1_bfp8.md).

## Public lengths, cache coverage and continuation

The cache validator (`optimized_decoder.py:648–654`) accepts matching BF16 or
BFP8 K/V dtypes and 32-token pages. It does not inspect page-ID values or
prove cache/RoPE coverage; those remain caller contracts. Prefill documents
request-owned pages covering attention read padding rounded to 128 tokens,
and absolute RoPE tables covering internal physical padding
(`optimized_decoder.py:656–669`). The selected advertised cache dtype is BFP8.

Fresh prefill uses physical tile/window padding and internal 1024-token
chunks without imposing that alignment on logical lengths. Nonzero prefix
continuation performs per-token cache updates, preserving occupied partial
pages. Continuation assembly groups 32 rows, 32 tiles and 32 chunks before
at most eight large groups at the 262144-token context limit. Repeating the
last output handle to pad a final concat group does not repeat a cache write;
the assembled result is sliced back to logical length. Checked boundary
reasoning covers lengths 1, 32, 1024, 32768 and nonzero remainders. Output
remains replicated BF16 `[1,1,S,2816]` for direct layer chaining.

## Existing execution evidence and audit limits

The evidence below was read from completed artifacts with runtime hash
`a12a913c`; these runs were performed by root, not rerun by this audit.
`tests/runtime_audit.py::device_only` rejects Torch execution through
`TorchDispatchMode` and patches TTNN host boundaries (`from_torch`, `to_torch`,
`as_tensor`, `to_device`, `zeros`, `ones`, `full`, `arange`). The ordinary,
stack and contract harnesses apply it to forward/warmup/capture. Fixture
creation, refresh and result inspection intentionally occur outside it.
This is a decoder-interface test, not a device-autoregressive token loop.

| Artifact | Recorded result relevant to this audit |
| --- | --- |
| [sliding_final_policy_raw_stress.json](sliding_final_policy_raw_stress.json) | Default 4096-token prefill +128 changing decode positions; eight duplicate replays per position; pass, replicas equal, minimum output PCC 0.9976112404 and cache PCC 0.9999966584. |
| [full_final_policy_stress.json](full_final_policy_stress.json) | Same 4096/128/eight-duplicate contract; pass, replicas equal, minimum output PCC 0.9994477563 and cache PCC 0.9999715022. |
| [stack_final_policy.json](stack_final_policy.json) | Sliding layer0 + full layer5, direct device handoff, independent caches; prefill33 and decode positions33/34 in one trace; exact duplicate outputs and replicas; runtime guard clean. |
| [sliding_final_batch32.json](sliding_final_batch32.json), [full_final_batch32.json](full_final_batch32.json) | B32 contract passes; per-slot lengths/continuations, refreshed request page ownership, prefix/other-slot preservation, exact replay/replicas and clean runtime guard. |
| [sliding_final_long_prefix.json](sliding_final_long_prefix.json), [full_final_long_prefix.json](full_final_long_prefix.json) | Prefix1056 plus continuation1025; cache ownership/preservation and repeat checks pass, runtime guard clean. |
| [sliding_final_max_262144.json](sliding_final_max_262144.json), [sliding_final_max_262143.json](sliding_final_max_262143.json) | Sliding context limit and unaligned limit-minus-one plus decode pass; anonymous DRAM reservations exercise capacity only, not a full-model stack. The 262144/steps0 case has no decode trace despite the requested trace flag. |

Eight duplicate replays mean 1024 comparisons after 128 first replays for
each ordinary TP run. Finite output/replica/repeat checks do not prove all
physical padding or arbitrary input values are correct. Current numerical
passes supersede the historical candidate failures for this selected policy;
the failing artifacts remain evidence and are not waived. Other pending
hardware gates and final stage review remain root's responsibility.

CPU validation for this audit: AST parsing and host-boundary enumeration,
manual selected caller/callee inspection, runtime hash verification, and
completed-artifact hash/result checks. Only Markdown changed, so no build
was required. No additional on-device result or performance improvement is
claimed by this source audit.

Inherited source hashes:

| File | SHA256 |
| --- | --- |
| `tt/optimized_decoder.py` | `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898` |
| `tt/fused_decoder.py` | `0c8892be32e04202fdbddd2b06850848c63ae57660c49d03c4d149a5a88b041c` |
| `tt/decode_attention.py` | `fd67ca281493097d4c7da60cbe52d5fe29581370b9fc62c4a9829baf69044b96` |
| `tt/precision_ops.py` | `52f0bad533c4d421b5c807914ad0d101533da3eb012b734e3e9c55bceefe23ef` |
| `tt/routing_precision.py` | `1237d3eb7b90563b4032b1ba1fa42606aca5cb2db93ba557f7603d5355ff6672` |
| `tests/runtime_audit.py` | `d3e6538f798a55172a03b93dbc319ecf20180e9794af2dc2811bda7fc71358c3` |
