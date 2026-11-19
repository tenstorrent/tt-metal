# TP4 Ornith decode head-layout investigation

## Findings

**The existing `nlp_create_qkv_heads_decode` can replace the three packed Q/K/V slices and their head reshapes, using existing operations. It cannot directly replace the entire unpack → RMSNorm → partial RoPE → placement sequence.** Its outputs are height sharded, which the ordinary RMSNorm implementation explicitly rejects. The existing fused rotary operations also do not directly implement Ornith's first-64-of-256 rotation.

The smallest candidate does **not** require another weight permutation: current setup already packs each chip's projection as `[Q1024, gate1024, K256, V256]`. Call the native op with **8 query heads, 1 KV head**, then reduce Q's *logical* head count from 8 to 4 while retaining its padded 32-head storage. Keep the gate as one slice from the original packed projection. Q/K still require conversion to interleaved memory for the existing norms and partial RoPE; V still needs placement on the disjoint cache-update grid.

A second, explicit decode-only weight contract `[Q1024, K256, V256, gate1024]` is also feasible: slice the first1536 columns and call with Q4/K1. It trades the compatibility view and fake gate-head reads for a QKV prefix slice and a second packing contract. Preserve the existing prefill weights and handle the B1 decode branch as well as batched decode.

**No current model correctness defect or performance improvement is established here.** This is an optimization contract investigation. The hard discrepancies are between plausible replacement recipes and the actual op contracts: wrong inferred head width, unsupported height-sharded RMSNorm, incompatible partial rotary semantics, and overlapping K/V grids. Additional low-level validation gaps are recorded separately below because the minimal candidate avoids them.

## Scope and evidence

Inspected the working tree based on commit `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`, including the already modified model file. Line references describe this prepared tree, not a clean checkout of that commit. No model/source changes, device access, resets, target execution, or hardware-dependent tests were performed. This report is the only added artifact from this investigation. Three independent investigators checked native QKV contracts, model layouts, and fused operations; their important claims were checked against source again before finalizing this report.

Source abbreviations used below:

| Name | Repository path |
|---|---|
| `D` | `models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py` |
| `CFG` | `models/autoports/ornith_ai_ornith_1_5_9b/tests/target_config.json` |
| `QKV/` | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/` |
| `ROT/` | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/` |
| `RS/` | `ttnn/cpp/ttnn/operations/data_movement/reshape_view/` |
| `CACHE/` | `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/` |
| `SDPA/` | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/` |
| `GEMMA/` | `models/demos/gemma4/custom_ops/` |

“Proven” below means a source-level contract or arithmetic result. Hardware correctness, program compilation, latency, and numerical tolerances for the proposed composition remain unverified.

## Current model contract

### Projection, gate order, and precision

The real head dimension is **256**, not128. `D:117` asserts it; `CFG:19,69–74` specifies global Q16/KV4, partial rotary factor0.25, and RMSNorm epsilon1e-6. TP4 therefore has **Q4/K1/V1 per chip**.

Setup reads the HF query/gate weight as `[16,2,256,4096]`, selects four heads for each rank, and permutes the Q/gate axis before the head axis (`D:229–233`). The local output columns are:

```text
Q0 Q1 Q2 Q3 | G0 G1 G2 G3 | K0 | V0
each head/group block is 256 columns
Q:0..1023     G:1024..2047  K:2048..2303  V:2304..2559
```

This already preserves the gate associated with each query head. Moving gate columns ahead of or behind K/V must preserve their internal `[G0,G1,G2,G3]` order. Attention concatenation later uses `num_heads=4`, and multiplication applies sigmoid to the gate (`D:1131–1144`), so a gate permutation would mispair features unless corrected there.

The projection weights use BFLOAT8_B by default (`D:81,89–90`), but the activations and projected result are BF16. Decode does not request a different output dtype (`D:545–550`), and matmul defaults to input-A dtype (`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:3428–3455`). The native QKV op accepts BF16/F32, **not** BFLOAT8_B; do not infer its input dtype from the weight dtype.

### Layout ledger

| Stage | Logical shape per chip | Padded shape / memory | Evidence |
|---|---|---|---|
| Decode residual | `[1,1,B,4096]` | Height32; local width-sharded L1 | `D:5–6,1147–1167` |
| Packed projection | `[1,1,B,2560]` | `[1,1,32,2560]`; currently L1 interleaved after `decode_linear` | `D:542–556` |
| Real Q | `[1,B,4,256]` | `[1,B,32,256]` | `D:1032` |
| Real K/V | `[1,B,1,256]` | `[1,B,32,256]` | `D:1034–1035` |
| Gate | `[1,1,B,1024]` | `[1,1,32,1024]` | `D:1033` |
| Q/K norm | Same real head shapes | Independent RMS reduction over256 columns, separate learned weights | `D:174–183,575–580,1032–1034` |
| Rotary | First64 columns only | Rotate at midpoint32; concatenate untouched tail192 | `D:348–352,738–748,1040–1045` |
| Attention/cache inputs | Real Q4/K1/V1 shapes | HEIGHT_SHARDED L1, one `[32,256]` shard/user | `D:1047–1067` |

Q/K norm weights are uploaded as HF weight **plus 1**, unlike `linear_attn.norm` (`D:174–183`). Any future fused norm must use these already adjusted weights, epsilon 1e-6, and the full 256-element RMS denominator. Normalizing the 64 rotary prefix separately changes the operation.

The current rotation is explicitly NeoX half-split within the 64-wide prefix: `models/autoports/ornith_ai_ornith_1_5_9b/tt/functional_decoder.py:165–168` builds `[-rot[...,32:], rot[...,:32]]`. The generic rotary factory sets `Wt=2, half_Wt=1` for width 64 (`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp:498–499`); its reader swaps the two width tiles (`device/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp:63,126–130`) and compute negates the first half (`device/kernels/compute/rotary_embedding.cpp:129–167`). The config's `mrope_interleaved` label does not establish adjacent-channel pairing for this implementation.

With default full-attention decode geometry, setup selects64 nominal output cores, block2 and two readers (`D:271–310`). Width2560 gives `per_core_N=ceil(2560/(32*64))=2` tiles, so the movement-optimized actual output grid has **40 cores**, each `[32,64]` (`D:259–290`). It is selected from the device's actual compute grid, **not** necessarily an8-wide rectangle. On an11-wide grid it is three full rows plus a7-core tail. Input K4096 uses the shared32-core `[32,128]` residual layout (`D:292–294`). These are source-derived defaults; caller overrides can change the geometry.

### B1 and prefill have distinct call paths

`full_attention_decode_batch` currently runs only for `B>1` (`D:1179–1185`). B1 goes through `full_attention(..., decode=True)` and its separate unpack sequence (`D:1186–1195,762–765`). Changing only the batch function therefore does not optimize B1.

Prefill uses `self.weights["packed_input.weight"]` through `minimal_matmul` (`D:534–540`) and explicitly interprets its query block as eight heads, then separates Q and gate (`D:767–775`). Both ordinary weights and DRAM-sharded decode weights are currently uploaded from the same pieces (`D:246–257`). A new decode permutation must be applied only to the decode upload. Prefill continuation can enter a decode path (`D:857–868`), so keep both contracts explicit rather than selecting weights merely from the outer “prefill” label.

## Native QKV contract ledger

### Wrapper and validation

| Contract | Exact source | Consequence for this candidate |
|---|---|---|
| Input allocated device TILE, BF16/F32; logical `[1,1,B,W]`, B≤32, W multiple32 | `QKV/device/nlp_create_qkv_heads_decode_device_operation.cpp:36–53` | Use ordinary32×32 tiles, positive B1..32, physical input height32. There is no power-of-two batch restriction. |
| Heads positive; padded W divisible by `Hq+2*Hkv` | `QKV/nlp_create_qkv_heads_decode.cpp:47–65` | D is **inferred**, not an independent argument. |
| Q-head count at least KV-head count | `QKV/device/nlp_create_qkv_heads_decode_device_operation.cpp:93–97` | K and V have equal counts. No separate V-head count exists. |
| Sharded input must be WIDTH_SHARDED, ROW_MAJOR; shard height equals physical input rows | Same file`:55–66` | For normal decode use L1 `[32,s]` shards. |
| Non-overlap requires `D % shard_width == 0` | Same file`:68–72` | A head may span several narrower shards. The validator's “partial heads” wording should not be read as the opposite divisibility condition. |
| Supplied `memory_config` contributes only its grid | `QKV/nlp_create_qkv_heads_decode.cpp:34–46` | Output layout is rebuilt as HEIGHT_SHARDED L1; requested buffer/layout/shard dimensions are not honored. |
| Optional output tensors are ignored | Same file`:17`; device operation`:185–193` | The op allocates three outputs; this interface does not provide preallocated output reuse. |
| Interleaved input forces overlap=True | Wrapper`:23`; factory selection device operation`:18–24` | Interleaved L1 is supported despite the stale binding text claiming sharded-only. Passing False on interleaved input does not separate Q/K grids. |

The factory computes `D/32` head tiles and the kernels use fixed32×32/16×16 geometry. Use D multiple32 and standard tiles. Validation checks fused width alignment, not all of these implied restrictions. D256 is sound for the proposed counts.

### Output memory and placement

Output spec construction is at `QKV/device/nlp_create_qkv_heads_decode_device_operation.cpp:120–182`:

```text
Q logical [1,B,Hq,D]     padded [1,B,ceil32(Hq),D]
K logical [1,B,Hkv,D]    padded [1,B,ceil32(Hkv),D]
V logical [1,B,Hkv,D]    padded [1,B,ceil32(Hkv),D]
```

Exactly **B output cores** are used per tensor, one user/core. The B dimension itself is not rounded to32 on the output. For Q8/K1/D256 all three shard shapes are `[32,256]`; each reserves16384 bytes/core in BF16. Together they reserve49152 bytes/user over their respective grids, before other live tensors. Three tensors with those padded head shapes are needed by the current path as well; allocation/lifetime differences still need profiling.

| Mode | Q | K | V | Available output grid capacity |
|---|---|---|---|---|
| overlap=True | First B cores | Same B cores | Same B cores | At least B |
| overlap=False | First B cores | Next B cores | Same cores as Q | At least2B |

Evidence: output-grid construction`:134–156`, capacity checks`:99–110`. **False separates Q and K, not Q and V.**

For the first candidate, provide an **origin-based8×4 available grid** and overlap=True. The op selects its B-core prefix, matching current Q/K placement. A full8×4 grid is only the available container; output shard grids still contain exactly B cores. Do not supply extra logical users to fill a rectangle.

### Kernel arithmetic, batch boundaries, and padding

The kernels traverse contiguous projection sections in order Q → K → V, then scatter each source batch row into a destination head row. Standard-sharded Q/K/V loops are at `QKV/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:95–144,159–208,225–274`; interleaved loops are at `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:197` onward. The two data-movement processors handle the two16-column face phases. The kernel waits for its NoC reads before returning; the replacement remains queued device work and Python return does not prove hardware completion.

For BF16, source-row offsets are:

```text
b<16:  b*32 bytes
b>=16: (b-16)*32 +1024 bytes
phase2 adds512 bytes within each tile
```

The interleaved factory supplies these offsets for `i=0..B-1` (`QKV/device/nlp_create_qkv_heads_decode_interleaved_program_factory.cpp:263–278`). Concrete rows 0/15/16/31 start at 0/480/1024/1504 bytes. B15→16 adds row 15 within the upper face; B16→17 adds row 16 in the lower face. Neither requires B to divide a grid width. A partial last output row of cores is permitted.

Only the requested logical head rows are written; padding rows are **not zeroed**. The fake-Q candidate deliberately turns gate rows4..7 into padding. Downstream operations must use logical Q-head count4. SDPA does read the logical head count (`SDPA/sdpa_decode_device_operation.cpp:612–627`); final concatenation explicitly receives4 (`D:1131–1134`). Parent validation should include nonzero padding, rather than assuming it is zero.

### Sharded input must exactly cover the packed width

For native sharded consumption, require `N*s=W`, where N is actual input cores and s is shard width. Both must describe the real physical layout. The kernels derive tiles/core from total fused head tiles divided by core count, **without using the actual input shard width**:

- Origin-rectangle kernel: `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:76–82` uses `num_x*num_y`.
- Subgrid kernel: `reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp:74–78` uses actual N.

For W2560=80 tiles, these are concrete legal input choices at D256:

| N | Shard `[H,W]` | Example origin rectangle | overlap=True | overlap=False |
|---:|---|---|---|---|
|40|`[32,64]`|8×5|Yes|Yes|
|20|`[32,128]`|5×4|Yes|Yes|
|10|`[32,256]`|5×2|Yes|Yes|
|16|`[32,160]`|8×2|Yes|Rejected:256%160≠0|

The current movement-optimized40-core×64 projection also has exact coverage. When its grid has multiple ranges, wrapper dispatch selects the subgrid factory (`QKV/nlp_create_qkv_heads_decode.cpp:29–32`), which enumerates actual input coordinates (`device/nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:137–149`). An origin nonsquare single rectangle selects the standard factory; its x-coordinate increments carry into y (`reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:127–141,191–205,257–271`). **Rectangular does not mean square.**

A32-core `[32,96]` input is unsafe for logical W2560: it covers3072 columns, but the kernel derives `floor(80/32)=2` tiles/core rather than the actual3. This can pass the op's validation. It is not the current default40-core case.

Use L1 for sharded input: the kernels form raw source addresses using worker-core NoC coordinates. Validation does not establish that a DRAM-sharded input would have the required address interpretation. Interleaved DRAM has a separate TensorAccessor path, including alignment staging, and should not be conflated with DRAM sharding.

## Concrete minimal candidate

### A. Existing weights, fake Q8, no new setup permutation

Keep projection output interleaved initially, as `decode_linear` already returns it. Suggested code structure, not an implementation or hardware-validated patch:

```python
# Setup: available output grid; native create synthesizes exact per-B shard grids.
head_output_mem = ttnn.create_sharded_memory_config(
    (32, 256),
    ttnn.CoreGrid(x=8, y=4),
    ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)

# Decode: packed logical [1,1,B,2560], padded [1,1,32,2560], BF16 TILE L1.
gate = packed[:, :, :, 1024:2048]
q8, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    packed,
    num_heads=8,
    num_kv_heads=1,
    overlap_qk_coregrid=True,
    memory_config=head_output_mem,
)
q = ttnn.reshape(
    q8,
    ttnn.Shape((1, B, 4, 256)),
    ttnn.Shape((1, B, 32, 256)),
)
q = self.norm(ttnn.to_memory_config(q, ttnn.L1_MEMORY_CONFIG), "self_attn.q_norm")
k = self.norm(ttnn.to_memory_config(k, ttnn.L1_MEMORY_CONFIG), "self_attn.k_norm")

# Keep current position lookup and batch permute/partial-RoPE/permute-back.
# Then keep current q->shard(0), k->shard(0), v->shard(4).
# Keep cache update, SDPA, concat(num_heads=4), and sigmoid(gate)*out.
```

The inferred width is `2560/(8+2*1)=256`. The first output contains real Q0..Q3 followed by G0..G3, while K and V are real single-head outputs. Calling Q4/K1 on the full 2560 width fails wrapper divisibility: 2560%6≠0. Do not use D128 or normalize pairs of half-heads independently.

**The two-Shape reshape above is an important part of the recipe.** Python exposes that exact logical+padded overload (`RS/reshape_nanobind.cpp:117–136`). Since only logical Q-head count changes while padded shape remains `[1,B,32,256]`, the current compatibility branch permits a metadata view (`RS/reshape.cpp:778–787`). The preceding codegen dispatcher cannot intercept this case: `RS/codegen/reshape_codegen_supported.cpp:397–403` rejects explicit padding or a logical-volume change. `ttnn/core/tensor/tensor_ops.cpp:456–478` retains the existing buffer/address/configuration. This aliases the first four rows of every user's existing 32-row shard; it does not flatten users or compact padded rows.

An ordinary tuple/list reshape checks equal logical volume (`RS/reshape_common.cpp:16–61`) and is not a substitute. The source explicitly labels the differing-volume compatibility branch as historical; it is legal in this checkout but merits a focused regression if parent chooses it. Do not generalize it to different physical extents. The generic comparison-mode reshape golden (`ttnn/ttnn/operations/core.py:194–195`) accepts only a normal equal-volume reshape, so check this alias against explicit Q4 reference outputs instead of treating that golden as support for the two-Shape call. Reading gate from `q8[:,:,4:8,:]` would start at an unaligned head row4: `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:224–235,377–382` routes that through ROW_MAJOR conversion. Keeping the original aligned width gate slice avoids that conversion.

The required Q/K unshards are not optional: `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:70–83` forwards to layernorm, and `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:223–231` rejects HEIGHT_SHARDED input/output. For B>1, this recipe removes three packed slices and three material head reshapes, adds one native create, one metadata alias and two Q/K unshards, and retains the gate slice and existing norm/RoPE/final placement operations. For B1, the existing K/V reshapes already request the slices' identical shapes and are no-ops (`D:762–765`; `RS/reshape.cpp:636–641`); only the Q reshape moves data. That is an operation/layout comparison, **not a speed claim**.

To cover B1..32, use a shared unpack helper in both decode branches or explicitly route B1 through the candidate batch implementation after checking its whole contract. The native op itself accepts B1; changing only `full_attention_decode_batch` leaves B1 on the old path.

### B. Decode-only reordered projection, true Q4/K1

If parent prefers to avoid fake gate-head reads and the compatibility alias, reorder only the full-attention **DRAM decode** weight columns:

```text
new = old[:,0:1024] | old[:,2048:2304] | old[:,2304:2560] | old[:,1024:2048]
      Q1024          K256                V256                gate1024
```

Keep the per-rank K4096×N2560 matrix shape and mesh rank concatenation unchanged. Preserve `self.weights["packed_input.weight"]` in its existing Q/G/K/V order for prefill. Only `self.dram_weights["packed_input"]` receives this new decode order (`D:246–257` is the upload boundary).

From the existing interleaved decode output, take:

```python
qkv = packed[:, :, :, :1536]       # logical [1,1,B,1536], padded B32
gate = packed[:, :, :, 1536:2560]  # original gate order
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv, num_heads=4, num_kv_heads=1, memory_config=head_output_mem
)
```

All column boundaries are tile aligned, and `1536/(4+2)=256`. Native outputs now have real logical Q4/K1/V1 directly. The QKV prefix slice still moves data; neither a bare view nor changing the width metadata of the original strided packed tensor is a safe way to remove it. Retain the same Q/K unshards, norm, partial RoPE and final placement as A. Both B1 and B>1 decode unpack consumers must switch together, including gate offsets; leave prefill consumers on the old contract.

If subsequently consuming a sharded1536-column prefix, use exact coverage such as24×64 or12×128; do not retain the full2560 projection's40 input shards under narrower metadata. This is a separate sharding experiment from the interleaved recipe above.

### C. Optional direct-native projection follow-up

The40-core×64 default projection is a source-compatible input for QKV creation before `decode_linear` converts it to interleaved. Avoiding that conversion is a separate experiment: gate extraction still needs a correct access path, and Q/K later need interleaving for ordinary norms. Simply omitting the conversion globally also changes linear-attention consumers. Scope any native return to this full-attention candidate and inspect/assert its actual memory config.

With WIDTH_SHARDED input, overlap=False can put K on the next B cores and V with Q on the first B, potentially avoiding a V relocation **if the downstream cache layout is deliberately changed to use that disjoint K/V arrangement**. It does not retain current `shard(0)` K plus `shard(4)` V automatically. Arbitrary B may make the K grid start mid-row, so its exact range enumeration/user order must be checked through cache-update setup. This is not part of candidate A's validated source contract, and no speed benefit is asserted.

## Downstream topology and rectangular head grids

The current8-wide helper forms one full-row rectangle plus an optional tail range (`D:1047–1063`). Q and K use y-offset0; V uses y-offset4 (`D:1065–1067`). For all B1..32, the two grids are disjoint and fit inside the8×8 attention program grid. Examples:

| B | Q/K grid | V grid |
|---:|---|---|
|1|`(0,0)`|`(0,4)`|
|4|`x0..3,y0`|`x0..3,y4`|
|8|`x0..7,y0`|`x0..7,y4`|
|13|`x0..7,y0` + `x0..4,y1`|same shape starting y4|
|16|8×2 rectangle, y0..1|8×2 rectangle, y4..5|
|32|8×4 rectangle, y0..3|8×4 rectangle, y4..7|

Avoid replacing these ranges with individual singleton ranges: current code warns that range enumeration can reorder independent users (`D:1056–1057`). Preserve the same row/user order through projection, trig lookup, Q/K/V placement, positions and page-table rows.

Fused cache update explicitly rejects intersecting K/V grids and requires equal core counts (`CACHE/paged_fused_update_cache_device_operation.cpp:237–245`). Native overlap=True puts K and V together, so feeding both native outputs straight to fused cache update fails. Candidate A retains current final V placement and therefore satisfies this contract. Cache/table/pool processing at `D:1068–1118` can stay intact once the original real head shapes and placements are restored.

SDPA's scale stays **1/16=1/sqrt(256)** (`D:1125`). The final native concat takes four heads, creates a batch32 packed output, and the model trims back to logical B (`D:1131–1136`; `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp:79–107`). Do not reinterpret its batch32 output as32 live requests.

There is a separate native-create output-grid hazard: the interleaved and standard-sharded factories construct Q runtime argument locations with `grid_to_cores(B,bbox.end.x+1,bbox.end.y+1,true)`, starting at the origin (`QKV/device/nlp_create_qkv_heads_decode_interleaved_program_factory.cpp:263–278`; standard factory`:124–128,321–335`; `tt_metal/common/core_coord.cpp:496–508`). A shifted/non-origin output grid can validate but receive runtime arguments on the wrong cores. Origin rectangles/prefixes used by A are safe by construction; do not try to directly target the y4 V region with an interleaved call. The subgrid factory enumerates actual Q/K core sets and avoids this particular hazard (`device/nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:129–149,326–340`). Factory selection depends on **input** grid shape, so a special output grid alone does not select that safer factory.

## Fused normalization / rotary alternatives

### `experimental.rotary_embedding_llama_fused_qk`

This operation fuses **two RoPE calls**, not RMSNorm and RoPE: its API has no norm weights or epsilon (`ROT/rotary_embedding_llama_fused_qk.cpp:12–18`). Relevant exact contracts:

| Requirement | Source |
|---|---|
| BF16 HEIGHT_SHARDED Q/K, cos, sin and transformation matrix; matching device | `ROT/device/rotary_embedding_llama_fused_qk_device_operation.cpp:25–40` |
| Decode leading dimension1, matching Q/K width, width multiple32 | Same file`:51–74` |
| D256 requires **fp32_dest_acc_en=False**, unlike current `self.compute` | Same file`:60–62`; `D:152–154` |
| Equal batch≤32; Q/K grids disjoint; combined cores≤64 | Same file`:83–95` |
| Cos/sin batch dimension equals Q batch plus K batch; transformation matrix `[32,32]` per shard | Same file`:97–113` |
| Output memory configs preserve Q/K input configs | `ROT/rotary_embedding_llama_fused_qk.cpp:24–25` |

Its tiled kernel applies one32×32 transformation separately to **each width tile** (`ROT/device/kernels/compute/rotary_embedding_llama_sharded.cpp:90–96`), then multiplies cos/sin across every tile (`:122–139`). Ornith's natural first64 representation requires partner `j±32`, crossing two width tiles. No32×32 within-tile transformation can express that exchange in the natural input layout. Setting tail cos=1/sin=0 preserves the tail but does not fix the first64's partner mapping. There is no partial-rotary-dimension parameter.

The current factory targets the actual union of Q/K core sets and hardcodes one batch/core (`ROT/device/rotary_embedding_llama_fused_qk_program_factory.cpp:72–104`), so holes in that union are not by themselves a reason to reject it. The blockers are its rotation basis, absence of normalization, precision setting, and disjoint Q/K requirement. Applying a channel permutation to Q/K and gamma/trig could change the rotary basis, but then decode K must still match the prefill cache basis. Such a design needs a complete basis ledger and is outside the minimal candidate.

### Gemma `qkv_norm_rope` custom operation

`GEMMA/qkv_norm_rope/op.py:71–103,149–150` is a closer reference: per-user `[32,D]` shards, separate Q/K gamma and epsilon, and FP32 accumulation. It still is not a direct replacement:

- The partner offset is **D/2** (`GEMMA/kernels/norm_rope_stages.hpp:123–148`), so at D256 the first64 reads partners128 positions away. Ornith requires a midpoint32 inside its64-wide prefix.
- It unconditionally normalizes V (`GEMMA/qkv_norm_rope/kernels/compute_qkv_norm_rope.cpp:54–55`); Ornith does not. Out-of-place mode could discard that V result while preserving original V, but this would still do extra work and would not fix rotary semantics.
- Its test's `partial=64` sets cos/sin to neutral values after64 but still uses a full-width `rotate_half` (`models/demos/gemma4/tests/unit/custom_ops/test_qkv_norm_rope.py:21–30`; `GEMMA/qkv_norm_rope/op.py:61–63`). That test label does not establish Ornith's first64/midpoint32 behavior.

### Distributed/fused RMSNorm and other wrappers

The experimental distributed RMSNorm post-all-gather operation uses supplied statistics for the full hidden row before splitting heads; `num_heads` does not turn it into independent per-head RMSNorm (`ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp:93–130`). Its pre-op requires interleaved input (`device/fused_rmsnorm_pre_all_gather_device_operation.cpp:25–28`), and its rotary fusion uses within-tile32×32 matmuls (`rmsnorm_post_allgather.cpp:169–183`). It does not solve this contract.

Ordinary distributed RMSNorm and the DiT RMSNorm wrapper ultimately use the same layernorm machinery; the DiT wrapper delegates with type RMSNORM (`ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.cpp:51–65`). No directly applicable general height-sharded per-head norm was found. Keep the existing full-D RMSNorm and partial RoPE for the first experiment.

## Other potential issues and excluded stories

These are real source-level limitations or test gaps, **not diagnoses of an observed current failure**:

1. **Insufficient QKV validation for overpadded sharded width and shifted output grids.** Concrete mismatches are shown above. Candidate A avoids both by using interleaved input and an origin output grid; current native projection's40×64 width cover is also exact. No source fix is proposed in this inspection-only task.
2. **Slice parameters are not a general batch-padding mechanism.** Sharded input requires `batch_offset` and `slice_size` together, but offset+size bounds are not checked (`QKV/device/nlp_create_qkv_heads_decode_device_operation.cpp:74–80,120–123`). The interleaved factory never consumes batch_offset although slice_size affects output specs. Do not pass either for B1..32 here.
3. **Fake V is not a simpler unchanged-width solution.** With total width2560 and Q≥KV, V cannot occupy more than one third of the width, so the entire1024-wide gate cannot fit in V. At D256, Q4/K3/V3 could distribute gate halves into unused K/V heads, but then requires two extractions and gate reassembly. Unequal K1/V5 is not supported. Fake Q8 preserves natural K/V shapes directly.
4. **No mathematical need for a new Q/gate permutation in A.** Setup has already un-interleaved HF's per-head Q/gate output. Repeating that transformation would be wrong. The required permutation for B is only a movement of intact Q/G/K/V column blocks.
5. **No evidence that square grids are required.** Standard and subgrid kernels have explicit row carries/enumeration. Distinguish valid nonsquare origin rectangles from unsupported shifted-output addressing and from unsafe extra input width padding.
6. **No basis to call the current layout sequence incorrect.** The observed source uses separate slices/reshapes, then explicitly establishes the required norm/rotary/cache contracts. Its potential expense is an optimization hypothesis; no timings were collected here.

## Static validation and parent experiment boundary

Cheap pure-Python symbolic checks passed for all four TP ranks and every logical B1..32 (2112 user rows): the current HF-to-local Q/gate packing, fake-Q8 extraction/trimming, and decode-only Q/K/V/gate reorder all preserve real Q, K, V and gate correspondence. The same checks verified disjoint current8-wide Q/V placements for all32 batches, the exact-width sharding arithmetic in the table, and BF16 source-row byte offsets0/480/1024/1504 at users0/15/16/31. These checks model source arithmetic; they do **not** execute TTNN, validate the compiled kernel, prove the compatibility view on a live mesh, or measure performance.

Existing native tests (`tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_decode.py`) cover interleaved L1/DRAM B1/2/4/8/16/32 (`:23–103`), minimum-width sharding and both overlap modes (`:209–225,342–372`), a nonsquare carry regression (`:345–347`), slice offsets (`:375–413`), and non-origin/disjoint input grids (`:416–457`). They do not establish this exact Q8/K1/D256 composition, all non-power-of-two batches, or the metadata Q8→Q4 alias plus norm/partial RoPE/cache chain. Non-overlap B32 is skipped in that test helper for its CI hardware; it is not rejected by the op when64 output cores are available.

Suggested parent-owned controls, within the decoder optimization stage:

1. Compare the unpack stage directly against current Q/K/V and gate on every TP rank, with distinct values per user/head/channel. Cover B1..32; prioritize3/7/9/13/15/16/17/23/31/32 for face and grid boundaries. Assert logical/padded shapes, dtypes, shard specs and user order.
2. For A, confirm the Q8→Q4 result aliases the same device address and preserves `[1,B,32,256]`; include finite nonzero fake-gate padding. Check the first divergent stage after full256 norm, after first64 RoPE, and after cache update. Check tail192 unchanged by RoPE and original gate order after concat.
3. Preserve learned gamma+1, epsilon1e-6, existing fidelity, scale1/16, positions and page tables. Use positions that differ by user and have nontrivial sin; zero-position-only comparisons cannot adjudicate rotary pairing.
4. Include B1's separate path and prefill→decode continuation, with both ordered pooled and independently owned cache states. For B, specifically prove the prefill upload stays old-order and all decode consumers use new-order.
5. Benchmark A and B separately with parent-owned device synchronization and the same warmed decoder boundaries. Record remaining Q/K unshards, V movement, allocations, and whole attention/decode latency. Any direct-native projection or overlap=False variation is a separate controlled experiment. Static operation-count changes do not predict the net latency.

No full-model or vLLM integration is required or proposed by this report.
