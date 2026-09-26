# Graph inventory and fusion ledger

Shapes: H=2816, I=2112 shared, J=704 routed, E=128/top8. Sliding Q/KV heads
16/8,D256; full16/2,D512. Prefill logical S is padded inside physical chunks (default1024, inherited
configurable maximum16384);
decode has one active row per request-slot loop. BF16 weights/cache, FP32
attention/routing sensitive boundaries. All preparation is outside forwards.

Functional graph before fusion:

| Sequence | Inputs/output | Movement before fusion |
|---|---|---|
| Input RMS | x[S,H], learned gamma -> FP32 | typecast, square, mean, epsilon, rsqrt, two multiplies |
| QKV | normed[S,H], packed W[H,Q+K+V] | prefill linear; decode repeated x, product/reduce groups, transpose and concat |
| Heads | packed QKV -> Q/K/V | existing dedicated prefill/decode split; decode L1 input + height-sharded output, then DRAM norm inputs |
| Head norms | each head row, Q/K gamma | FP32 square/mean/rsqrt/multiply |
| RoPE | Q,K + supplied tables | decode embedding + repeat; neg/slice/concat, FP32 multiply/add |
| Cache update | K,V -> paged cache | explicit BF16 cast; decode height sharding required by paged_update_cache; prefill paged_fill_cache |
| Prefill attention | Q,K,V | native SDPA; sliding overlapping tail; full subsequent chunks paged SDPA |
| Decode attention | Q, dynamic page IDs, K/V | integer row addresses; flatten-cache embedding; head-major reshape; per-KV-head QK, max/sub/exp/sum/div, PV |
| O projection | head concat -> [S,H] | existing dedicated head concat, FP32 linear |
| Attention residual | attention norm + original x | FP32 post norm then FP32 residual add |
| Shared MLP | fused gate/up -> GELU(gate)*up -> down | two slices, standalone activation/multiply; existing pre/post norms |
| Router | residual norm, scale, W[H,E] | FP32 scale twice; decode repeated SFPU row products; topk logits, softmax, scatter, expert scale |
| Routed experts | normed residual, selected routes | separate sparse gate/up; repeated transpose/reshape; GELU/multiply; sparse down; route-weighted expert sum |
| Output | shared+routed norms and residual | add, layer scalar multiply, BF16 cast |

Source inventory: ttnn/cpp/ttnn/operations/{normalization,transformer,
experimental/transformer,eltwise,data_movement,matmul}; Python golden functions
in ttnn/ttnn/operations/{unary,binary}.py; tests/ttnn/unit_tests/operations;
models/demos/gemma4/tt, gpt_oss/tt/experts, models/common/modules.
Dedicated precision contracts and exact kernel paths are in AUTODEBUG_precision.md.

| Skill pattern | Applicability / evidence or experiment |
|---|---|
| Dedicated activation | GELU already dedicated. `ttnn.geglu` source unary_composite_op.cpp:279 expands split+approximate GELU+multiply, not a fused kernel. Accurate GELU folded into multiply is tested instead. Other activations not in graph. |
| Softmax recognition | Router already native softmax. Native stable and externally centered softmax tested: sliding fails the combined real headline; full passes but is slower. Exact SUB+EXP merged instead; sum/div retained. |
| RMSNorm | Native FP32 norm + external gamma selected except sliding head norms, where native fails S2049 reuse. Precise ADD+RSQRT merge preserves PCC. Head-row peer packing passes but is slower. |
| Distributed RMSNorm | No collective or distributed tensor on a1x1 mesh; not applicable. |
| SDPA | Prefill already native; decode native paged SDPA retried with explicit BF16 Q and HiFi4/FP32 accumulation: sliding/full PCC .9815/.9864. FP32 Q validator and BF16 statistics prevent exact contract; retain batched precise attention. |
| Split QKV / heads | Already nlp_create_qkv_heads for prefill and nlp_create_qkv_heads_decode for decode; packed QKV retained. |
| Decode head concat | Already nlp_concat_heads_decode; sharding follows its required contract. |
| Prefill head concat | Already dedicated concatenate_heads. |
| RoPE | Native and newer HF kernels retried with FP32 supplied tables/compute and interleaved decode-as-prefill layout; both partial and full rotary layer kinds tested. |
| TopK | Already ttnn.topk of logits. Sorting probabilities changes close expert ranks (functional controls); preserve selection order. |
| RepVGG conv-sum | No convolution in decoder. |
| Shared-LHS matmul | QKV/shared gate-up already packed. Routed gate/up packed and checked; QKV SFPU peer groups coalesced without changing arithmetic. |
| Spatial mean | No spatial dimensions. Hidden-axis RMS variance already a one-axis mean. |
| Permute-reshape-permute | Routed expert identity movement removed; independent KV heads batched as a candidate. |
| Conv+bias/scale/activation | No convolution or convolution parameters. |
| Matmul activation | Gate occupies only half packed output, so global matmul activation would incorrectly activate up. Binary input activation supplies the applicable merge. |
| Input activation + binary | Accurate GELU merged into shared/routed multiplication; direct same-input component evidence. |
| Matmul+bias | Model projections have no bias. |
| Transpose+matmul | Decode QK already uses transpose_b=True. Weight transposes are setup only. |
| Slice-after-matmul | QKV and gate/up slices all consumed; discarding an operand range would change outputs. Full-layer tied K/V projection duplication removed with narrower packed operand, reusing K for V. Tied K/V head norm shared; both pass real headline. |
| BatchNorm+conv | Neither operation exists. |
| Pad+pool/conv | Neither consumer exists; sequence padding is required by attention/cache/experts and is internal. |
| Stable softmax | Native stable and externally centered softmax tried; sliding misses .995 and full is slower. Exact subtract-exp merges without losing precise sum/div. |
| Reduction+reshape | QKV/router reductions already keepdim. Expert reduction preserves logical/padded row contract; identity transposes are removed in packed path. |
| Sum*1/N -> mean | RMS variance already mean. Router normalization is selected softmax, not constant-N scaled sum. |
| Decode RoPE reshape merge | Adapted dedicated rotary candidates evaluate existing decode geometry with repeated FP32 tables; source restrictions captured in diagnosis. |

Additional candidates: eliminate repeated QKV/router activations via binary
broadcast; combine scalar multiplies into binary output activation; output
residual add/scale/BF16 packing in one op; combine iterative prefill concatenation;
batched decode QK/PV; tiled gather to avoid full-cache untilization. Each has been measured; final device profiling and independent review are recorded separately.

## Additional graph boundaries

- Retain output norm sharding through shared+routed addition/combined norm,
  avoiding intermediate interleaved conversions. Fuse the addition with the
  combined RMSNorm residual argument. Real headline and broad cache checks pass.
- Fuse K/V cache writes with disjoint single-core input shards. Cache remains
  TILE/BF16/page32 and the public capacity is unchanged. Full and sliding headline
  candidates pass. Movement into height sharding is required by the cache op.
- Full-cache untilization is internal to embedding; no explicit host conversion
  occurs. Flattened tiled gather is exact but ~24ms versus~0.66ms embedding in the
  bounded same-input sliding probe. The page-axis adaptation in cache_gather_adapted.json is exact but ~3980us
  (~4107us including dynamic index expansion), versus139us embedding after full
  warmup; it is also rejected. The earlier659us embedding result had less warmup.
- Accurate GELU input activation speeds shared MLP and expert prefill. Expert
  decode keeps separate accurate GELU+multiply: the merged version measured
  slower on the same inputs. This phase-specific choice is intentional.
- Full native HF rotary passes but costs ~24us against precise rotary in the
  isolated full-boundary comparison. It is not selected merely to reduce ops.
  Sliding prefill native rotary fails .995, while sliding decode native passes.
- Every absent conv/pooling/BatchNorm/collective pattern above is structurally
  inapplicable on this single-device transformer layer. No model-semantic
  relaxation, dtype sweep, new kernel, or later pipeline stage is introduced.

## Final peer-merge controls

`rope_merge.json` compares the live sliding decode HF rotary calls against
concat-Q/K, one rotary, and split. Outputs are bitwise equal, but the merge is
102.86us versus100.44us, including all movement/table preparation; rejected.

`components_{sliding,full}_repro.json` explicitly forces the decode GELU merge
through `fuse_decode_gelu=True`, so the rejected path remains reproducible after
the phase policy was selected. Exact outputs; decode packed separate GELU is
930.71/930.45us versus932.47/933.15us merged. Prefill merge remains faster.

`expert_batch_{sliding,full}.json` and `expert_batch_512_{sliding,full}.json`
compare the identical recorded real-weight expert inputs/routes across M32,64,
128,256,512,1024. Every output is bitwise equal and replay-stable. M64 is fastest,
~695ms versus699ms for loop32 per1024 input rows. The selected internal expert
batch is64; partial physical tails use prepared32-row configs. No public sequence
alignment is imposed. Direct `[1,1,M,H]` multiplication produces expert-major
output, so the old grouped-tile grid-overflow warning does not apply. Only
`per_core_M=M/32` changes to express the merged work; precision and other matmul
settings stay fixed. Kernel source still rereads weights per output block; no
unmeasured DRAM reuse is claimed.

## Expert weighted reduction

`expert_mix_{sliding,full}.json` captures real S64 prefill/S1 decode sparse-down
outputs and routes. The copied multiply/reduce suffix exactly matches the
runtime output. HiFi4 native matmul with BF16 and FP32 accumulators passes PCC
above .99998, is replay-stable, and includes all required operand movement.
Prefill takes1480us versus397us, so it retains multiply/reduce. Decode takes48us
versus56us and is promoted to full-layer testing. The probe preserves the old
control explicitly after selection, while capturing unchanged live projections.
`mix_candidate_commands.json` records both complete4096/128 accumulator controls.
`mix_sharded_commands.json` additionally writes matmul output directly into the
next norm's width-sharded memory, eliminating its interleaved-to-sharded copy.

Complete-layer sharded-output candidates pass every headline output. Selected
sliding FP32-accumulation5129.59us versus BF165130.72us; full BF165609.52us
versus FP325610.54us. These close alternatives have no meaningful accuracy
regression. The selected graph additionally beats the previous best reduction
by~12us sliding and~13us full. These intermediate mix-only gates are in selected_validation_commands.json.
Final closure uses verified_validation_commands.json after the additional
producer/consumer controls below.

## Producer/consumer movement controls from independent review

Exact commands are in review_boundary_commands.json,
review_boundary_adapted_commands.json and review_boundary_combined_commands.json.
All whole-layer controls use real4096/128 inputs and the unchanged .995 bar.

| Boundary | Adaptation and result |
| --- | --- |
| Shared down → norm | Write native linear output into the norm's width shard. Captured live inputs/weights, copied baseline exactly equal, projection+movement+norm measured. K blocks2,3,6,11,22,33,66 pass;6 fastest44.75us versus62us. Headline passes both. |
| QKV → split heads | Final transpose/concat directly in L1, including tied-KV concat; both pass and save~6/10us. |
| Cache cast → height shard | Dedicated typecast rejects layout change. Adapted unary-chain TYPECAST passes both. I2S output-dtype conversion fails .991401/.983611; its copy/pack conversion uses different arithmetic, so rejected. |
| Combined norm → final residual add | Binary add consumes the existing shard and produces interleaved BF16 output with scalar activation; both pass. Tested in combination because isolated gain is small. |
| Concat heads → output projection | Retain dedicated concat's width-sharded output; reshape only logical batch padding and let native HiFi4/FP32 projection consume it. Both pass, saving~31/61us. |

The sharded shared-down program infers LoFi/BF16; the original interleaved
control infers HiFi2. The initial legal K-block sweep consistently used LoFi,
but its description incorrectly claimed HiFi2. Explicit matched controls are
recorded in shared_fidelity_commands.json; see the work log correction.
No residual-wide layout redesign or separate optimized-decoder stage is included.

Combined selection: boundary_combined_sliding_1 =5067.09us and
boundary_combined_full_1 =5520.47us, both full headline passing. Direct final
add improves both combined paths. Full unary cast remains2.4–2.9us slower
with the other accepted rewrites, so only sliding selects it. The final graph
retains the faster two-step full-cache cast/shard.

### Final decode movement inventory

| Producer → consumer | Selected movement contract |
| --- | --- |
| Broadcast QKV → head split | Producer writes L1; dedicated split owns required head sharding. |
| Head norms/rotary → paged update | Sliding TYPECAST unary directly produces disjoint update shards; full keeps measured faster explicit cast/shard. |
| Precise paged attention → concat/O projection | Height shard required by dedicated concat; its width shard feeds projection directly, with logical-padding reshape only. |
| Shared down → own norm | Native linear writes norm width shard, inferred LoFi/BF16, K block6. |
| Expert sparse down/routes → own norm | HiFi4 mixing matmul writes norm width shard; sliding FP32 accumulator, full BF16 accumulator. |
| Shared/routed norms → combined norm | Compatible width shards preserved; residual argument merges addition. |
| Combined norm/residual → layer output | Mixed-memory binary consumes norm shard directly, folds scalar, packs BF16 into interleaved DRAM. |

The remaining cache embedding's internal whole-pool untilization is retained
only after slower exact tiled-gather adaptations. Routing ROW_MAJOR sparsity,
head/cache sharding and selected precision boundaries follow consumer contracts.
No runtime torch/from_torch/to_torch or host fallback is introduced.

### Explicit shared-down fidelity control

The final profiler revealed inferred LoFi for the explicit sharded program, while the original interleaved baseline inferred HiFi2. The initial probe's HiFi2 description was wrong; it did not invalidate the measured outputs or times. Matmul program-config inference is the cause (matmul_device_operation.cpp:2808–2810). Four fresh real-activation controls explicitly set LoFi or HiFi2 with the same BF16 operands, FP32 destination disabled, packer L1 accumulation enabled, norm consumer, layout and K-divisor sweep. Every candidate passes PCC and repeated replay. K6 remains fastest under both policies:

| Kind | Interleaved HiFi2 baseline (us) | Sharded K6 LoFi (us / PCC) | Sharded K6 HiFi2 (us / PCC) |
| --- | ---: | ---: | ---: |
| sliding_attention | 62.031 | 44.718 / 0.9998896404 | 44.880 / 0.9999369429 |
| full_attention | 62.018 | 44.764 / 0.9999019428 | 45.018 / 0.9999455949 |

These are warmed traced host boundary timings, not whole-layer device times. The small fidelity difference is distinguished from the larger sharding improvement. LoFi remains the best measured correct configuration; no runtime change or headline reprofile is needed. Raw controls: shared_fidelity_{sliding,full}_{LoFi,HiFi2}.json; exact commands and exit codes: shared_fidelity_commands.json. Final whole_layer.json summaries and telemetry name the actual mixed-fidelity policy. Historical probe JSON is retained, with its stale attribution superseded here.
