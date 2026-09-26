# Multichip memory audit

Status: **calculated_unvalidated**. Source inspection and CPU arithmetic only;
no full-stack allocator measurement, hardware run, or runtime change was made
for this audit. Context target remains 262144. A dual TP-decode/EP-prefill
runtime is **hypothetical and not selected or implemented**.

The current TP4 conservative peak is **24,287,543,296 bytes per device
(22.620 GiB)**. A hypothetical dual expert layout increases it to
**29,085,106,176 bytes (27.088 GiB)**. Both are below even a decimal 32 GB
comparison, subject to the ownership and scheduling assumptions below. This
arithmetic does not validate actual usable DRAM or fragmentation.

Audited runtime SHA256:
`a7c8370500079afd74548628186a565c758a2fd4b37b07993988938ea4479efb`.
Source names below are repository-relative or relative to this model directory.

## Retained object graph and aliases

| Root reachable after construction | Retained tensor payload | Aliases and released setup values |
| --- | --- | --- |
| `decoder.layer.moe.experts` (TP4) | Packed quantized `gate_up`, quantized `down`, tiny original prefill sparsity | `prefill_gate is gate_up` and `prefill_down is down` under current policies. No retained BF16 expert source container. |
| `decoder.layer.moe.experts` (EP4 option) | Packed quantized 32-expert/full-width gate/up and down; two ownership-index tensors | Same phase aliases. Its local `weights`, `source` and `packed` construction variables are not saved on the result. |
| `decoder.layer.self_attn.source.weights` | One BFP8 QKV tensor, one BFP8 output-projection tensor, BF16 Q/K norm weights | `_Projection.weight` and `_Projection.prefill.weight` are the same QKV tensor. `prefill_minimal_output.weight` aliases the existing output weight. |
| `decoder.layer.shared_mlp.gate_up` and `.down` | BF16 packed shared gate/up and BF16 shared down | Imported callables are lambdas closing over these tensors. Replacing the `SharedMLP` Python wrapper does **not** release its weights. |
| `decoder.layer.moe.router` | BF16 router projection, FP32 transposed projection rows, BF16 and FP32 input scales, BF16 expert scales, small gate buffers | `GeneralizedRouter.projection_weight` aliases its original router projection. Its `original -> BroadcastRouter.source -> Router.source` chain retains both original BF16 tensors and FP32 routing preparation. |
| Decoder norms and position state | Seven BF16 norm vectors, seven tiled FP32 copies, Q/K norm copies, two full-context int32/uint32 position tables | The two position uploads are distinct allocations. Optional hidden-sharded residual adds five small local norm vectors. |

Proof of expert lifetime:

- `models/demos/gemma4/tt/experts/weights.py` returns an `ExpertWeights`
  container with separate gate/up/down BF16 tensors when called by the current
  BF16 layer loader.
- `tt/fused_decoder.py:259-267` (`PackedExperts.__init__`) stores `config`,
  `width`, a newly concatenated gate/up tensor, an alias of the original down
  tensor, and the small sparsity tensor. It stores neither `source` nor
  `source.weights`.
- `tt/optimized_decoder.py:53-57,105-116` shallow-copies that dictionary, sets
  `prefill_source=None` for `active_prefill=True`, replaces gate/up and down
  with converted tensors, and aliases those conversions for prefill when the
  phase dtypes match. Both TP and EP construction enable active prefill and
  matching phase dtypes. `expert_split=False` creates no extra weight slices.
- `tt/multichip_decoder.py:352-405` first builds the TP optimized experts. In
  the EP branch, Python evaluates the EP constructor before replacing
  `self.layer.moe.experts`; the imported original BF16 experts, local TP packed
  BF16 tensor, and TP optimized tensors therefore coexist during EP setup.
  After the assignment and factory return, no decoder path references those
  old expert objects. The TP expert's router reference does not introduce a
  reverse reference that would keep the replaced expert alive.
- `_ExpertParallelExperts.__init__` at lines 93-140 keeps its `ExpertWeights`
  and `PackedExperts` variables local. Only converted tensors and the small
  configuration survive on the new instance.

Consequently the BF16 copies are eligible for ordinary tensor release when
the corresponding constructor references disappear and queued device uses
finish. The code does not explicitly measure deallocation or allocator reuse;
this is a source ownership proof, not a measured memory plateau.

Other ownership evidence:

- `tt/multichip_decoder.py:310-317` replaces the attention weight container
  with BFP8 conversions. The original BF16 QKV/O matrices have no surviving
  reference in this path. `MinimalPrefillProjection.__init__`
  (`tt/optimized_decoder.py:1458`) assigns `self.weight = weight`; it does not
  copy the tensor. `OptimizedAttention.from_existing` copies the existing
  wrapper dictionary, and `FusedAttention.__init__` retains only `source.source`.
- `models/demos/gemma4/tt/shared_mlp.py:144-201` disables the DRAM-sharded
  special path for this MoE model and creates the two BF16 closure-backed
  linear projections. `_SharedMLP` retains those functions at runtime lines
  74-80. There is no extra optimized shared-MLP weight copy in this candidate.
- `tt/routing_precision.py:54-57` allocates FP32 scale/projection rows;
  `GeneralizedRouter` retains the original wrapper and aliases its BF16
  projection (`tt/optimized_decoder.py:839-845`). Direct decode projection
  does not remove the retained FP32 rows.
- `Gemma4Attention` defaults to `create_kv_cache=False`
  (`models/demos/gemma4/tt/attention/__init__.py:75-125`). The layer constructor
  does not override it, so there is no hidden original cache in addition to
  caller-owned paged caches. RoPE is also caller-owned. The optimized sliding
  prefill drops its K/V tail after the final chunk
  (`tt/optimized_decoder.py:1414-1421`).

## Exact large tensor payloads

Configuration: H=2816, expert I=704, E=128, TP=4; 25 sliding and 5 full-attention
layers. `tests/config.json` supplies these dimensions. TP expert padding produces
local I=192; EP uses 32 complete experts with I=704. Shared MLP local I=544.

For 32x32 tiles, `tt_metal/impl/data_format/tile.cpp:70-79` gives BF16=2048,
BFP8=1088, BFP4=576 bytes, including BFP exponents. All are already divisible
by Blackhole's 64-byte DRAM alignment; these matrices need no extra per-tile
round-up. Tile count for one TP expert matrix is `128*88*6=67584`; for EP it is
`32*88*22=61952`.

| Payload per device per layer | Sliding bytes | Full bytes |
| --- | ---: | ---: |
| TP gate/up | 147,062,784 | 77,856,768 |
| TP down | 38,928,384 | 38,928,384 |
| TP expert total | 185,991,168 | 116,785,152 |
| EP gate/up | 134,807,552 | 71,368,704 |
| EP down | 35,684,352 | 35,684,352 |
| EP expert total | 170,491,904 | 107,053,056 |
| Existing BFP8 QKV + O | 9,191,424 | 15,319,040 |
| Existing BF16 shared MLP | 9,191,424 | 9,191,424 |
| Existing small-state allowance | 8,388,608 | 8,388,608 |
| Existing TP layer weight/state bound | 212,762,624 | 149,684,224 |
| K+V at context 262144 | 285,212,672 | 285,212,672 |

The small retained DRAM tensors enumerated above total about 7,438,080 bytes
for sliding and 7,504,640 for full before small alignment/page-table allowances.
Optional sharded norm vectors add 450,560 bytes per layer. Router gate buffers,
EP ownership IDs and semaphores are small L1 allocations; allocator and trace
overheads remain separately reserved. The 8 MiB allowance is therefore retained
as a conservative per-layer planning bound, not replaced with a precision claim
about allocator sizes.

Both layer kinds have the same per-device cache payload:
`2*262144*2*256/1024*1088 = 285212672` for sliding and
`2*262144*1*512/1024*1088` for full. Full KV head duplication across rank pairs
is included. Sliding storage is not truncated to its attention window.

## Full-stack totals and maximum-length prefill

| Per-device allocation family | Current TP bytes | Hypothetical dual bytes |
| --- | ---: | ---: |
| Decoder weights and small-state bound | 6,067,486,720 | 10,865,049,600 |
| Absolute paged K+V, all 30 layers | 8,556,380,160 | 8,556,380,160 |
| One tied replicated BF16 embedding/LM head | 1,476,395,008 | 1,476,395,008 |
| Shared per-kind prefill/decode RoPE | 1,610,612,736 | 1,610,612,736 |
| Independent trace/workspace/allocator reserve | 2,147,483,648 | 2,147,483,648 |
| Subtotal before full-length activation buffers | 19,858,358,272 | 24,655,921,152 |
| Full prefill BF16 input | 1,476,395,008 | 1,476,395,008 |
| Retained full-length chunk outputs | 1,476,395,008 | 1,476,395,008 |
| Final concatenated BF16 output | 1,476,395,008 | 1,476,395,008 |
| **Conservative prefill peak bound** | **24,287,543,296** | **29,085,106,176** |
| Headroom below decimal 32,000,000,000 | 7,712,456,704 | 2,914,893,824 |

`OptimizedDecoder.prefill_forward` keeps `hidden_states` alive, appends every
1024-token result to `outputs`, and then concatenates the whole list
(`tt/optimized_decoder.py:700-735`). At 262144 tokens, one full BF16 hidden
tensor is `262144*2816*2 = 1476395008` bytes = 1.375 GiB. The three full-length
buffers therefore total **4.125 GiB**, independent of bounded internal chunking.
This audit adds them **in addition to** the 2 GiB reserve. The previous plan's
2 GiB reserve by itself was not a complete long-prefill live-buffer bound.

Dual experts add exactly
`25*170491904 + 5*107053056 = 4797562880` bytes = **4.4680786 GiB**.
Only the expert layouts are duplicated; attention, shared MLP, router, norms,
positions, RoPE and KV cache are shared. A wrapper that instead constructs two
whole decoders would not satisfy this accounting.

The RoPE subtotal requires sharing tables across layers of each attention
kind: `262144*(256+512)*2 bytes*2(cos/sin)*2(layouts) = 1610612736`.
If every layer independently owns both full-context table layouts, RoPE alone
becomes **17.5 GiB** and the current plan no longer fits. The decoder API permits
sharing because tables are caller-owned; the future stack owner must do so.
Likewise, embedding and LM head must share their tied tensor. Full-model
logits-for-every-prompt-position and retained diagnostic layer outputs are not
included; ordinary autoregressive inference and release of previous-layer
outputs are assumed.

## Setup peak versus retained weights

| Constructor temporary expert allocation | Bytes per device |
| --- | ---: |
| TP separate BF16 gate/up/down | 415,236,096 |
| TP additional packed BF16 gate/up | 276,824,064 |
| TP temporary BF16 total | 692,060,160 |
| EP separate BF16 gate/up/down | 380,633,088 |
| EP additional packed BF16 gate/up | 253,755,392 |
| EP temporary BF16 total | 634,388,480 |
| Combined TP+EP constructor temporary upper bound | 1,326,448,640 |

The current EP option constructs TP first, so this combined temporary upper
bound applies while EP is being built, in addition to live TP/EP quantized
weights. For a sliding layer, the full simultaneous expert-only setup bound is
`1326448640 + 185991168 + 170491904 = 1682931712` bytes. This is a per-layer
setup peak, not thirty retained copies. An eventual dual implementation that
keeps both converted layouts has a whole-stack setup bound, including the same
2 GiB reserve, of **25,982,369,792 bytes**; this is below the conservative
long-prefill bound above. Setup and maximum-length execution are separate phases.

No explicit deallocation change is needed to justify the current source graph,
but full-stack allocator validation must confirm that queued setup operations
and caller references actually release their temporary buffers. Allocating
duplicate per-layer RoPE, keeping BF16 source wrappers, retaining two entire
decoders, retaining all intermediate layer outputs, or using multiple concurrent
maximum-context requests invalidates the bound.

The machine-readable plan and context-contract memory section now state
`calculated_unvalidated`; all target-context and capability fields are unchanged.
Neither the arithmetic fit nor the component EP tests closes the stage's
maximum-context, allocator-peak, or watcher gates.
