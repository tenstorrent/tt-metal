# Multichip memory audit

The selected hybrid policy has a source-derived conservative peak of
**27,451,657,216 bytes per device**, leaving **4,548,342,784 bytes** below a
comparison capacity of decimal 32 GB. Resident weights/state, caches, shared
RoPE, tied embedding/LM head, and the independent reserve total
**23,022,472,192 bytes** before the three long-prefill buffers.

Audited final runtime SHA256:
`a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255`.
Status: **final policy calculated; matching maximum-context acceptance pending**.
This audit uses source inspection and CPU arithmetic. It makes no measured
full-model allocator, usable-capacity, or fragmentation claim. Context target
remains 262144 with no capability reduction.

The previous machine-readable plan is preserved unchanged in
`memory_capacity_plan_before_final_policy.json`. Earlier hybrid/shared
maximum-context reservation tests are historical evidence; their byte counts
and status are retained separately in the current plan. They are not relabeled
as acceptance of this final runtime.

## Final policy delta

The model has 25 sliding and five full-attention layers. Only the sliding
resident payload changes relative to the previous plan:

| Per-device sliding payload | Previous bytes/layer | Final bytes/layer | Saving/layer |
| --- | ---: | ---: | ---: |
| Indexed TP expert gate/up plus down | 185,991,168 | 116,785,152 | 69,206,016 |
| Additional shared decode weights | 4,882,944 | 3,351,040 | 1,531,904 |
| **Total** | **190,874,112** | **120,136,192** | **70,737,920** |

The full-stack saving is `25*70737920 = 1768448000` bytes per device. EP prefill
weights, full-layer resident weights, and all KV cache sizes are unchanged.
Sliding TP gate/up now use raw host BFP4 packing, down remains BFP4. Shared
decode gate/up now use BFP4 for both kinds; down remains BFP8 sliding/BFP4 full.
See `AUTODEBUG_final_policy_packing.md` for the numerical reason to preserve the
host packing path.

The additive **TP-decode base belongs to the selected hybrid policy**. It is
not actual standalone TP-only prefill memory: a nonhybrid sliding configuration
retains a separate BF8 prefill gate tensor. The selected default routes prefill
to the independent EP object, so its indexed TP object needs no such copy.

## Retained object graph and setup lifetime

Source names are repository-relative or relative to this model directory.

| Root reachable after construction | Retained tensor payload and ownership |
| --- | --- |
| `_HybridExperts.decode` | TP BFP4 gate/up and down; indexed routing references. `prefill_gate is gate_up` and `prefill_down is down` in the selected default. |
| `_HybridExperts.prefill` | EP weights for 32 complete experts/device: BF8 gate/up plus BFP4 down sliding, all BFP4 full; phase aliases and small ownership IDs. |
| `decoder.layer.self_attn.source.weights` | Shared prefill/decode BFP8 QKV and output weights plus small Q/K norm tensors. Projection wrappers alias these tensors. |
| `decoder.layer.shared_mlp` | Original BF16 prefill projections remain in imported closures; `decode_weights` adds separately packed gate/up and down. |
| `decoder.layer.moe.router` | Original BF16 projection/scales, retained FP32 routing preparation, expert scales, and four persistent single-core gate tensors. |
| Norms, position state, caller-owned cache/RoPE | Norm tensors and two position tables are covered by the small-state allowance; cache/RoPE are counted separately. |

`models/demos/gemma4/tt/experts/weights.py` creates original BF16 gate/up/down.
`PackedExperts.__init__` in `tt/fused_decoder.py` retains a new packed gate/up,
an alias of down, configuration and sparsity; it does not retain the original
source container. `OptimizedExperts.__init__` copies that dictionary, converts
the weight fields, and uses `prefill_source=None` with active prefill.
`_ExpertParallelExperts` similarly keeps its BF16 `weights`, `source`, and
`packed` containers local to construction.

In `MultichipDecoder.from_state_dict`, the sliding raw host upload replaces the
initial device-cast BFP4 `experts.gate_up`. The hybrid branch also assigns
`experts.prefill_gate = experts.gate_up`, removing the last phase alias to the
superseded quantized gate. The EP prefill constructor is independent and retains
its BF8 sliding gate/up. `_HybridExperts.__call__` dispatches one-row decode to
the TP object and prefill to the EP object.

Thus the original BF16 expert containers and superseded BFP4 gate are eligible
for release after their constructor references and queued uses finish. This is
an ownership proof, not a measured allocator plateau. Diagnostic references,
caller-retained setup objects, or unfinished queued uses can extend lifetimes.

`_SharedMLP.configure_decode` uploads raw host BFP4 gate/up and BFP8/BFP4 down;
its original callable fields still close over BF16 prefill matrices. The shared
MLP therefore requires both sets in the bound. Attention projection wrappers
reuse converted weights; no original KV cache is allocated by the imported
attention constructor (`create_kv_cache=False`). RoPE and paged cache remain
caller-owned. The final policy changes dtype/geometry/placement, not the cache
contract or the number of persistent position tables.

## Exact large tensor payloads

Dimensions: H=2816, expert I=704, E=128, TP=4. TP padding gives local I=192;
EP has 32 whole experts with I=704. Shared MLP local I=544. A 32x32 tile occupies
2048 bytes for BF16, 1088 for BFP8, or 576 for BFP4, including exponent storage;
these are already 64-byte aligned.

One TP expert matrix has `128*88*6=67584` tiles; one EP matrix has
`32*88*22=61952`. One shared matrix has `88*17=1496` tiles.

| Payload per device per layer | Sliding bytes | Full bytes |
| --- | ---: | ---: |
| TP decode gate/up, BFP4 | 77,856,768 | 77,856,768 |
| TP decode down, BFP4 | 38,928,384 | 38,928,384 |
| **TP decode expert total** | **116,785,152** | **116,785,152** |
| EP prefill gate/up | 134,807,552 | 71,368,704 |
| EP prefill down | 35,684,352 | 35,684,352 |
| **EP prefill expert total** | **170,491,904** | **107,053,056** |
| BFP8 QKV + output projection | 9,191,424 | 15,319,040 |
| Retained BF16 shared prefill MLP | 9,191,424 | 9,191,424 |
| Additional shared decode weights | 3,351,040 | 2,585,088 |
| Router/norm/position small-state allowance | 8,388,608 | 8,388,608 |
| Hybrid TP-decode base, excluding EP/additional shared decode | 143,556,608 | 149,684,224 |
| **Complete selected layer weight/state bound** | **317,399,552** | **259,322,368** |
| K+V at context 262144 | 285,212,672 | 285,212,672 |

Shared decode storage is `2*1496*576 + 1496*1088 = 3351040` bytes sliding,
and `3*1496*576 = 2585088` full. Across all layers this is **96,701,440 bytes**.
EP experts add `25*170491904 + 5*107053056 = 4797562880` bytes per device.

The retained 8 MiB/layer allowance covers router, norms, position tables and
small metadata. Earlier source enumeration put their large DRAM pieces around
7.44/7.50 MB per layer, before small alignment allowances. Router gate placement
moves existing small tensors; it does not duplicate them. Trace, semaphore and
allocator overhead remains separately reserved.

Each layer's per-device cache is 285,212,672 bytes:
`2*262144*2*256/1024*1088` sliding and
`2*262144*1*512/1024*1088` full. Full KV duplication across rank pairs is included.
Sliding storage retains absolute pages rather than truncating to its window.

## Full-stack totals and long-prefill lifetime

| Per-device allocation family | Selected hybrid bytes |
| --- | ---: |
| TP-decode base weights and small state | 4,337,336,320 |
| Additional EP prefill experts | 4,797,562,880 |
| Additional shared decode weights | 96,701,440 |
| **Complete decoder weight/state bound** | **9,231,600,640** |
| Absolute paged K+V, all 30 layers | 8,556,380,160 |
| One tied replicated BF16 embedding/LM head | 1,476,395,008 |
| Shared per-kind prefill/decode RoPE | 1,610,612,736 |
| Independent trace/workspace/allocator reserve | 2,147,483,648 |
| **Resident plus reserve** | **23,022,472,192** |
| Full prefill BF16 input | 1,476,395,008 |
| Retained full-length chunk outputs | 1,476,395,008 |
| Final concatenated BF16 output | 1,476,395,008 |
| **Conservative long-prefill peak** | **27,451,657,216** |
| **Headroom below decimal 32,000,000,000** | **4,548,342,784** |

For additive accounting, the TP-decode base's resident-plus-reserve subtotal is
18,128,207,872 bytes and its activation-inclusive subtotal is 22,557,392,896.
Adding EP but excluding the extra shared decode weights gives resident
22,925,770,752 and peak 27,354,955,776. Neither subtotal represents a separately
validated executable prefill policy.

The full-length buffers total `3*262144*2816*2 = 4429185024` bytes, independently
of bounded internal chunking. This bound assumes grouped concatenation releases
old groups, previous-layer outputs are released, and diagnostic outputs are not
retained. The independent 2 GiB reserve remains in addition to these buffers.

RoPE must be shared across layers by attention kind/layout:
`262144*(256+512)*2 bytes*2(cos/sin)*2(layouts) = 1610612736`. Per-layer copies of
both full-context layouts would alone require 17.5 GiB and invalidate the plan.
Embedding and LM head must share one tied tensor. Full prompt-position logits,
multiple simultaneous maximum contexts, and duplicated whole decoders are not
included. B1 maximum context and B32 short-context capacity are distinct.

## Setup peak

| Constructor temporary allocation | Bytes per device |
| --- | ---: |
| TP separate BF16 gate/up/down | 415,236,096 |
| TP additional packed BF16 gate/up | 276,824,064 |
| EP separate BF16 gate/up/down | 380,633,088 |
| EP additional packed BF16 gate/up | 253,755,392 |
| Prior combined BF16 temporary upper bound | 1,326,448,640 |
| Superseded TP BFP4 gate during raw host replacement | 77,856,768 |
| **Conservative combined temporary allowance** | **1,404,305,408** |

The new allowance conservatively permits the old device-cast BFP4 gate to coexist
with the replacement and the previous BF16 setup bound. It is counted once per
active layer constructor, not retained for all sliding layers. CPU packing
buffers are outside this per-device DRAM accounting. The source alias update
prevents the superseded gate from becoming an extra resident phase weight.

Adding the setup allowance to the selected full-stack resident-plus-reserve
bound gives **24,426,777,600 bytes**, below the long-prefill bound. Setup and
maximum-length execution are separate phases. Queued-use completion and actual
allocator reuse remain a full-model validation responsibility.

## Validation scope

`sliding_max_capacity.json` and `full_max_capacity_verified.json` are historical
hybrid/shared single-layer runs at 262143 prefill plus one decode, using anonymous
reservations for other resident payloads. Their old reservation/peak numbers
remain in `memory_capacity_plan.json:hybrid_shared_candidate`; the previous
entire plan is archived unchanged. They establish earlier layer/workspace
capacity, not execution or fragmentation of a complete 30-layer model.

The final a12a913c source reduces retained payload. Matching maximum-context
acceptance now passes both262143+1 and262144+0 for both attention kinds; see
sliding_final_max_262143.json, sliding_final_max_262144.json,
full_final_max_262143.json and full_final_max_262144.json. These fresh results
use the revised reservation plan. The arithmetic/source audit itself was CPU-only;
hardware commands and normal exits are recorded beside the final result JSONs.
