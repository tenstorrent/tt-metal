# TTNN optimization reference

Supporting material split from `SKILL.md` to keep the invoked skill concise. DiffusionGemma-specific instructions in `SKILL.md` take precedence.

# Useful Optimization Knowledge

Use this reference while optimizing functional TTNN code. It captures repo-local optimization patterns and the strongest current LLM guidance. If you are not optimizing an LLM, use your best judgement about what applies in your case.

## Code Paths Worth Reading

- `tech_reports/LLMs/llms.md`: LLM memory configs, matmul variants, DRAM-sharded matmul guidance, and perf-report interpretation.
- `models/common/modules/attention/attention_1d.py`: reusable attention configs with BFP8 attention weights, BFP8 KV cache, DRAM-sharded decode matmuls, SDPA configs, and L1-sharded decode residual paths.
- `models/common/modules/mlp/mlp_1d.py`: decode/prefill MLP split, DRAM-sharded decode matmuls, sharded outputs, and precision knobs.
- `models/common/modules/lm_head/lm_head_1d.py`: reusable LM-head output projection with vocab splitting, LM-head dtype, DRAM-sharded weight memory config, input/output memory configs, and decode program config.
- `models/common/tests/modules/lm_head/test_lm_head_1d.py`: expected LM-head construction, weight splitting, memory config, and PCC checks.
- `models/common/tensor_utils.py`: helpers to serialize program and compute-kernel configs for artifact reporting.
- `models/common/sampling/generator.py` and `models/common/modules/sampling/sampling_1d.py`: common on-device sampling implementations to compare before optimizing token-out sampling.
- `models/demos/gpt_oss/tt/experts/README.md`, `models/demos/gpt_oss/tt/experts/decode.py`, `models/demos/gpt_oss/tt/experts/prefill.py`, `models/demos/gpt_oss/tt/experts/weights.py`, `models/demos/gpt_oss/tt/experts/config.py`, and `models/demos/gpt_oss/tt/topk.py`: default routed MoE active-expert path using `ttnn.sparse_matmul`.
- `models/demos/gpt_oss/tt/`, `models/demos/gemma4/tt/`, and `models/demos/deepseek_v3/tt/`: model-specific examples where common modules do not fully fit.

## Core Optimization Rules

- Your initial functional test suite remains the correctness floor. Rerun the same functional prefill, decode, PCC, paged KV-cache, determinism, stress, trace, and watcher checks against the optimized path before accepting performance wins.
- Avoid data movement before tuning math. A slightly smaller core grid can beat a faster individual op if it avoids resharding between ops.
- Optimize the operation topology before fine-tuning individual op configs. Count material matmuls, collectives, reshard/layout conversions, and host/device crossings in the measured path. A path with extra high-cost ops is not optimized just because each individual op has a reasonable program config.
- Decode activations should generally stay width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
- Prefill activations are usually large and often belong in DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
- Use SDPA/FlashDecode/FlashAttention ops instead of hand-built attention primitives when the target model fits their contracts.
- Explicitly configure `memory_config`, `program_config`, and `compute_kernel_config` for important ops. Defaults are often correct but suboptimal.
- Choose shard specs and core grids that divide tensor dimensions cleanly into tiles. Padding in sharded paths is a common source of bugs or wasted work.
- For DRAM-sharded decode matmul, weights should be width-sharded in DRAM and activations/outputs width-sharded in L1 on the matching core grid.
- Keep the primary optimization target single-user batch-1 prefill/decode. This is not permission to remove larger-batch support; preserve correct batch handling and verify up to batch/concurrency 32 for complete model and serving paths when hardware and memory allow it. For MoE decoders on non-Galaxy systems, preserve gate-selected active-expert execution and prefer the GPT-OSS `ttnn.sparse_matmul` path for sparse expert projections plus score weighting and expert reduction. Dense all-expert execution is a debug baseline, not the optimized target.

## Optimization Advice

### OPT-001: Treat split decode QKV as a topology defect until proven otherwise

When optimizing decoder attention and the measured path has separate Q, K, and V matmuls that consume the same post-norm activation, first test a packed QKV projection before spending time on local program-config tuning for the split projections. This applies when the model's Q, K, and V projections are ordinary linear projections of the same hidden state and there is no per-projection operation, dtype, bias, adapter, normalization, quantization, or device-placement contract that requires them to stay separate.

This is a special case of the rule that fewer larger matmuls are generally better than more smaller matmuls when the larger matmul preserves dtype, layout, program-config quality, and downstream consumption.

The packed candidate should pack or concatenate the Q, K, and V weights at load time, run one decode matmul for the combined output width, then use the target's legal on-device QKV-head creation path before rotary, KV-cache update, and SDPA or FlashDecode. For grouped-query attention, compute the packed width from the real head counts: `q_heads * head_dim + 2 * kv_heads * head_dim`. Preserve any required weight permutation or reverse-permutation convention from the reference implementation; a fast packed path with wrong Q/K ordering is a correctness bug, not an optimization.

Verify the topology in the final `tt-perf-report`, not only in code. A successful packed decode path should show one material QKV projection matmul in the attention block, not three Q/K/V matmuls, and it should not contain a cluster of reshapes, slices, and layout conversions whose only purpose is to repair the split projection outputs before rotary. If the path remains split, leave measured evidence or a precise blocker after adapting weight packing, output splitting/head creation, memory configs, rank, padding, and dtype/fidelity. "The functional implementation was already split" is not a blocker.

Compare packed and split candidates under the same traced warmed decode harness, context shape, precision policy, and correctness checks. Keep the faster correct path. If packing loses because it forces worse dtype, fidelity, memory layout, program config, or extra output movement, record that full-path comparison; do not infer the answer from op count alone.

### OPT-002: Preserve the SDPA and KV-cache contract when changing attention

When optimizing decoder attention, treat SDPA or FlashDecode time as part of the attention contract, not a downstream detail. Before accepting a change to QKV packing, head creation, rotary, cache update, page-table handling, cache dtype, cache layout, or attention input memory config, compare the candidate against the strongest prior correct attention path at the same traced decode context.

The comparison must include the `tt-perf-report` row for SDPA or FlashDecode, the KV-cache dtype, cache layout, page-table shape/distribution, Q/K/V head layout, attention op `program_config`, chunk sizes, block size, `num_kv_heads`, output memory config, and the final attention-block topology. Do not accept an attention candidate only because the projection matmul got faster if SDPA, cache update, or layout-conversion time moved the cost elsewhere. A lower total layer time can justify a local SDPA regression only when the report names the regression, quantifies it, and shows the whole traced decoder is still faster under the same context and correctness checks.

When the final candidate still uses a BF16 KV cache and SDPA or FlashDecode is a material cost, a reduced-cache candidate is mandatory before accepting the stage. Try the architecture-appropriate reduced cache dtype, usually BFP8/BFLOAT8_B, with the same context length, page table, head layout, trace path, and correctness checks. This requirement applies even when the functional baseline used BF16 cache; a functional baseline is not proof that BF16 is the optimized cache policy.

Preserve a lower-precision KV cache when the prior correct path used one and PCC still passes. If BFP8 or another reduced cache dtype fails, debug the fill/update contract before falling back to BF16: prefill fill-cache tensors may need an explicit cast to the cache dtype, while decode `paged_update_cache` inputs may need to remain BF16 or FLOAT32. Also compare cache shape, local/global head mapping, page-table distribution, and whether packing QKV forced a different Q/K/V head creation path. A BF16 cache fallback is acceptable only with a correctness failure, op-contract blocker, or measured same-context win that includes the SDPA/cache cost.

When SDPA or FlashDecode is material, do not rely on the op default configuration. Create an explicit architecture-appropriate decode attention program config and measure it against the default under the same cache dtype and head/page contract. For Wormhole paged decode, the usual first candidate is `ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)` when legal. If a default attention op is kept, record the explicit config tried, its SDPA row, and why it lost or failed.

The final report for an optimized decoder must list the SDPA or FlashDecode row, KV-cache dtype/layout, attention program config, reduced-cache candidate result, and kept/rejected decision. If SDPA or FlashDecode is one of the largest regressions versus the prior same-context correct candidate, if BF16 cache remains without a reduced-cache trial or precise blocker, or if a material SDPA path uses default program config without an explicit-config candidate, the stage is not optimized.

### OPT-003: Preserve the decode residual layout through norms and residual adds

For decoder optimization, the residual stream is a contract across the layer, not a temporary tensor between ops. Before tuning local matmul or attention knobs, choose the decode residual memory config and carry it through input RMSNorm, attention residual add, post-attention RMSNorm, MLP input, and final residual add when the ops legally support it. A residual path that repeatedly falls back to DRAM interleaved because it is convenient is not optimized.

When RMSNorm or LayerNorm is material, use a sharded norm program config with sharded L1 input and output. The common single-chip Wormhole starting point is a width-sharded L1 residual over a clean core grid, paired with `ttnn.LayerNormShardedMultiCoreProgramConfig`; adapt the grid, shard shape, and subblocks to the model shape and hardware. If the norm output is DRAM interleaved, or if a residual add writes DRAM interleaved before the next norm or matmul, measure a sharded-residual candidate before accepting the stage.

Do not let an attention or MLP helper force the whole layer back to DRAM. Pay required layout conversions only at the narrow API boundary, then restore the residual contract before the next residual add or norm. If a candidate keeps a slower residual layout because a downstream op rejects the sharded tensor, record the exact op, shape, memory config, error or PCC failure, and the measured cost of the fallback.

The final report for an optimized decoder must list both norm rows, norm program configs, residual-add output memory configs, residual memory config before attention and MLP, and any `InterleavedToSharded`, `ShardedToInterleaved`, or reshape/slice rows used to cross helper boundaries. If norm rows are one of the largest regressions versus the prior same-context correct path, or if norms/residual adds use DRAM interleaved without a measured sharded candidate or precise blocker, the stage is not optimized.

### OPT-004: Sweep DRAM-sharded decode matmul geometry, not only dtype

For dominant DRAM-sharded decode matmuls, do not assume the largest available compute-core count is fastest. The logical program core count, activation shard grid, `in0_block_w`, `per_core_N`, and output subblock fields are part of the performance contract. More cores can shrink the legal K block or per-core output block enough to lose to a smaller coherent grid. This is a block-geometry version of the rule that fewer larger work chunks can beat more smaller chunks.

When a DRAM-sharded matmul or repeated matmul role is material, measure at least one candidate tied to the residual or intermediate sharded layout grid, one larger-core candidate when legal, and any helper/default candidate already in the code. Keep dtype, math fidelity, weight memory config, input/output memory config, and surrounding residual layout fixed while sweeping geometry, unless the candidate requires a coherent layout change; if it does, measure the whole compatible layer path. A geometry sweep is invalid if it silently changes precision, fidelity, or weight layout from the best prior correct path or from the mandatory precision policy for that tensor group.

Treat the producer's activation shard width as part of the matmul program config. A residual or norm path that uses more cores can make each shard too narrow, forcing `in0_block_w=1` or `2` for the next DRAM-sharded matmul. Treat `in0_block_w=2` as a minimum floor, not a success condition, and do not encode an old expectation that `2` to `4` is enough. For dominant DRAM-sharded rows, especially BFP4/LoFi decode gate/up, QKV, output, down, expert, and LM-head matmuls, sweep larger legal divisors of the tiled K dimension and the input shard's K-tile width. Values do not need to be powers of two: try larger legal values such as `3`, `5`, `7`, `8`, `10`, `14`, `16`, or higher when L1 capacity, divisibility, and op contracts allow them. Experiment 22 Llama 3.1 8B showed `in0_block_w=16` beating `8` for material MLP rows, so `8` is not a default stopping point. If a dominant matmul is stuck at `in0_block_w<=2`, or a larger value fails with a divisibility error such as `(shard_shape[1] / tile_width) % in0_block_w != 0`, do not accept that as a matmul-only blocker until you have tried a coherent upstream shard grid with fewer cores and wider shards, or mathematically inert padding that makes larger divisors legal and slices/masks padded channels before they affect outputs. Measure the whole layer: a slightly slower norm or residual add can be the right choice if it unlocks a much faster QKV, output, gate/up, down, or LM-head matmul.

For dense MLP or expert gate/up/down geometry work, include the best legal reduced-weight policy in the geometry sweep, usually BFP4/LoFi for MLP weights. A BFP8/HiFi2 geometry sweep is useful as a baseline, but it cannot justify accepting the MLP geometry when BFP4/LoFi was never tried, failed without diagnosis, or was not measured under the same geometry candidates. If the final MLP path uses higher precision than the best prior correct path or the mandatory BFP4/LoFi trial, record the lower-precision candidate result and the exact correctness or op-contract blocker. A random-weight or synthetic-activation PCC failure is not an exact blocker when real target-model weights pass; reconcile that discrepancy with real-weight coverage before accepting the higher-precision fallback.

For each dominant matmul role, the final report must list shape, dtype/fidelity, program class, logical program core count or grid source, input shard shape in tiles, `in0_block_w`, `per_core_M`, `per_core_N`, output subblock fields when exposed, input and output memory configs, bandwidth classification, traced latency, correctness result, and kept/rejected decision. If `tt-perf-report` marks a dominant DRAM-sharded matmul `SLOW`, reports low DRAM utilization, reports missing output subblock information, or shows a small `in0_block_w` on a material row, do not accept the stage until a precision-locked block-geometry sweep has tried larger legal `in0_block_w` divisors, including non-powers-of-two and `16` when legal, or recorded a precise L1/divisibility/op-contract blocker.

### OPT-005: Keep logical decode batch separate from tile padding

Decoder kernels often use tile-padded activation rows, such as one logical token padded to 32 rows. Those padded rows are a tensor-layout contract, not extra active users. Do not change logical batch, active user count, page-table semantics, or KV-cache indexing just to satisfy an op shape check. A result with the right matmul shape but the wrong active batch is a different workload.

When reporting or comparing decode attention, list logical user batch, tile-padded activation rows, current-position tensor shape, page-table shape/distribution, KV-cache shape, page block size, active cache slots updated, and SDPA/FlashDecode row time. Compare SDPA or FlashDecode only against a prior path with the same logical batch and context length. If a helper requires padded RoPE, current-position, or head-layout metadata, create padded metadata that preserves the original logical batch semantics instead of treating padding rows as real users.

If an optimized decoder candidate changes logical batch from the target workload, the stage is not optimized for that target, even if PCC passes and the matmul rows look good. Such a run can be kept only as a separate throughput-scaling experiment, clearly labeled with active batch and excluded from single-user decoder signoff.

### OPT-006: Sign off cumulative optimized contracts, not isolated wins

Optimized decoder work is cumulative. Before a new focused experiment, write the strongest prior correct candidate's required contracts: projection topology, weight dtypes and math fidelity by tensor group, KV-cache dtype/layout, logical batch and tile padding, attention program config, residual/norm memory layout, dominant matmul geometry, and known required layout transitions. The new candidate must preserve those contracts unless the experiment is explicitly testing one of them.

A focused experiment may vary one contract to answer a local question, but it is not an optimized-decoder signoff if it regresses another previously validated material contract. For example, fixing SDPA while losing MLP precision, fixing MLP geometry while changing logical batch, or fixing logical batch while reverting attention weights to BF16 are local evidence, not final optimized evidence. Either integrate the old and new contracts in one candidate and remeasure, or record the conflict as a blocker with exact rows.

The final report for an optimized decoder must include a cumulative contract table: packed/fused projections and head path, attention weight dtype/fidelity, KV-cache dtype/layout, logical active batch versus tile-padded rows, SDPA/FlashDecode config and row, residual/norm layout and norm rows, MLP/expert dtype/fidelity and geometry rows, total device time, op-to-op gap, correctness, and any intentional deviations. Use the strongest cumulative correct candidate as the baseline for the next step, not the latest focused run that happened to pass its local target.

### OPT-007: Treat attention projection precision as its own decode search

Do not limit reduced-weight trials to MLP or expert projections. In decode, attention QKV and output projections are small-M, large-weight matmuls; if they are material or DRAM-bound, BFP4 attention-weight candidates are mandatory before accepting BFP8 or BF16 attention weights. This applies to packed QKV, separate Q/K/V, output projection, and fused CCL+matmul forms that consume attention weights.

Test attention weight dtype and math fidelity independently from activation dtype, KV-cache dtype, CCL payload dtype, and MLP weight dtype. At minimum compare the strongest prior attention path with the current/default attention weight dtype and fidelity, then with BFP4 attention weights and each legal fidelity that could plausibly win for that op. Keep residual/norm, cache, and topology contracts fixed unless the candidate deliberately tests a compatible topology family; if topology changes are required, compare the BFP4 attention policy inside that compatible family, not only in isolation. A BFP4 attention trial on a weaker split-Q/K/V or BF16-output topology is screening evidence only; it can prove feasibility, but it does not accept or reject BFP4 for the optimized stage when a packed, fused, or lower-movement attention topology is legal and material. After screening passes, remeasure the BFP4 policy on the lowest-op-count compatible topology before making the final optimized-stage decision. This is a special case of the rule that fewer larger matmuls are generally better than more small ones: precision and fidelity must be tested on the matmul topology the final decoder should use.

Do not reject BFP4 attention because BFP4 MLP failed, because BFP8 activations were selected, or because a prefill row got slower when the optimization target is decode. Do not reject it on synthetic weights or synthetic activations alone; synthetic runs can catch crashes and op-contract failures, but precision vetoes need real weights or recorded real activations. For attention-weight changes, raw K/V cache PCC against a higher-precision baseline is a diagnostic, not an automatic rejection, because changing K/V projection precision legitimately changes the cached intermediate. If layer output PCC passes but raw cache PCC fails, run a follow-on decode, multi-token replay, top-k/readiness check, or other cache-consuming real-weight test before rejecting the policy. Reject BFP4 attention when it causes model-visible correctness loss, trace replay failure, generation/readiness failure, unacceptable latency, extra typecasts or layout conversions that lose whole-layer traced decode, or a precise unsupported fused-op/shape/L1 blocker.

The final report must list QKV/output-projection dtype and fidelity, weight memory config, row times, SDPA/cache side effects, correctness, follow-on cache-use evidence when cache PCC is the concern, and kept/rejected decision. If attention projections remain BFP8 or BF16 while QKV, Q/K/V, output projection, or fused attention matmul rows are material, the stage is not optimized without a real-weight BFP4 attention trial or a precise op/correctness blocker.

### OPT-008: Compare row-parallel output projection decompositions

For tensor-parallel row-parallel output projections, choose the algebra before choosing the CCL primitive. Compare the two main legal decompositions: local-input/full-output matmul followed by reduce or reduce-scatter, and gathered-input/local-output matmul followed by all-gather when the residual boundary needs replicated hidden state. The second family often appears as fused all-gather plus matmul with output-sharded weights. Do not reject it because a matmul-reduce-scatter candidate failed, because a sharded-residual carry-forward probe failed, or because the functional implementation uses local full-output weights.

When testing fused all-gather plus output projection, repack or reshard the projection weight for local output width, adapt tensors to the fused op's rank and layout contract, and set the CCL axis, transfer count, input/output memory configs, and matmul program config explicitly. The expected perf report should show a fused CCL+matmul row whose matmul N is the local output width, followed by a gather only if the next residual/norm contract requires replicated hidden. A local-input/full-output matmul plus RS/AG is a different decomposition, not a failed version of this one.

The final report for a material row-parallel output projection must list which decomposition was selected, the weight sharding/layout for each candidate, fused op rank/layout requirements, CCL transfer count, program config, row times, whole-layer latency, correctness, and exact blocker for rejected decompositions. If the fused all-gather-matmul path fails with L1 circular-buffer allocation, report the requested and available bytes plus the tensor shapes and program config that produced them; then try the smallest legal local-output candidate before treating the fused family as blocked.

### OPT-009: Make repeated decode CCL buffers persistent when legal

For repeated traced decode collectives, async CCL is not complete by itself. If an all-gather, reduce-scatter, all-reduce, or fused CCL+matmul row is material and runs every token or every layer, try the API's persistent or preallocated output, intermediate, and ring buffers when available. This applies to hidden all-gathers after output projections, reduce-scatters after row-parallel matmuls, LM-head or logits collectives, expert collectives, and fused CCL+matmul rows.

Keep the CCL algorithm, payload dtype, collective dimension, topology, link count, worker count, buffers per channel, and memory configs fixed while testing persistence unless the API requires a coupled change. Compare the persistent-buffer candidate against the best non-persistent candidate in the same traced warmed decode window. If persistence cannot be used, report the exact API limitation, tensor shape, memory config, buffer size, trace constraint, L1/DRAM allocation failure, or correctness failure; do not leave the field implicit.

The final report must list every material decode CCL row with payload dtype, input/output memory configs, CCL axis/dim, topology, `chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`, persistent output buffer status, persistent intermediate buffer status, row time, and kept/rejected decision. A stage with repeated decode CCL rows using default non-persistent buffers is incomplete unless a persistent/preallocated candidate was measured or a precise blocker is recorded.

### OPT-010: Compare packed and split 3-matmul MLP gate/up in decode

For 3-matmul MLPs, such as gate/up/down or FF1/FF3/FF2 blocks, "pack the gate and up weights and run one larger matmul" is a strong first candidate, not a proof. If the gate/up group is material and both families are legal, compare packed gate-up against separate gate plus up while holding dtype/fidelity, input memory config, residual layout, downstream elementwise contract, and row-parallel output contract fixed. This is a special case of the "fewer larger matmuls are generally better than more small ones" rule: the usual bias is toward packing, but the measured whole-MLP winner is the optimization target.

A packed gate/up candidate must include the cost of splitting or reshaping the packed output before activation, plus the activation, binary elementwise combine, and any layout movement before the down matmul. A split candidate must include both matmul launches and any extra layout movement. Do not keep packed gate/up merely because it has fewer matmul launches or a faster packed matmul row if the doubled output width forces a worse `per_core_N`, `in0_block_w`, output subblock, L1 allocation, slice/layout row, or precision policy.

Also compare against the best legal fused-elementwise form of the split path. For SwiGLU-style MLPs, the activation may be fused into the gate matmul or into the following binary elementwise op; for other MLPs the analogous activation/add/multiply fusion can change the balance. Packed gate/up must beat that best separate-family implementation, not just an untuned separate baseline. If packed saves a few microseconds on the matmul but adds more split, unary, binary, or layout time, keep the separate path.

The final report for a material gate/up group must list packed and split candidate rows, program configs, input/output memory configs, weight memory configs, dtype/fidelity, split/slice/layout rows, activation rows, binary elementwise rows, downstream down-matmul and reduce-scatter or all-gather rows, whole-MLP and whole-layer traced latency, correctness, and kept/rejected decision. If one family is impossible, report the exact API, shape, memory, program-config, L1, trace, or correctness blocker.

### OPT-011: Use phase-specific activation shards when they unlock matmul geometry

Do not force one residual or norm shard grid to be the working input shard for every decoder sub-block. A layout that is good for a boundary, norm, or one projection can make another material matmul group slower by giving each core too few inner-dimension tiles for a larger `in0_block_w` or output subblock. When a material matmul is stuck at a small block size because the producer shard is too narrow, try a phase-specific working shard for that sub-block: usually fewer cores with wider shards, immediately before the projections that share that input.

Keep the model-visible boundary contract fixed while testing the working shard. Restore or preserve the residual layout only where it is actually required: residual adds, following norms, layer boundaries, cache/head helpers, or collective contracts. Compare the full traced sub-block, not the isolated matmul: include the added reshard/layout rows, all matmuls that consume the working shard, elementwise rows, downstream collectives, and any saved conversions. If the same working shard feeds multiple projections, account for the reshard cost once across the group.

The final report must list the boundary memory config, candidate working memory configs, shard shapes in tiles, legal and rejected `in0_block_w`/subblock values, exact divisibility or L1 errors, added layout rows, row times, whole-layer latency, correctness, and kept/rejected decision. A dominant matmul group blocked by an error like "shard inner tiles must be divisible by `in0_block_w`" is not optimized until a wider working-shard candidate has been measured or a precise blocker is recorded.

### OPT-012: Do not let synthetic precision stress veto real-weight wins

Random-weight and synthetic-activation precision cases are stress probes, not the definition of the optimized dtype/fidelity policy. When a lower-precision or lower-fidelity candidate is faster on real target-model weights plus recorded or real target activations, and it passes the stage's real-weight PCC, trace replay, and runtime checks, do not reject it only because a synthetic/random-weight "representative semantics" check falls below a PCC threshold against a higher-precision reference. That result may mean the synthetic distribution is adversarial or not representative of the target model.

If synthetic coverage fails while real target-model evidence passes, either keep the faster lower-precision policy or add a same-contract real-weight check that reproduces the failure under the disputed condition: non-aligned lengths, paged prefill/decode transition, traced replay, larger batch/concurrency, cache-consuming output, or generation/readiness evidence. Rejection requires one of: model-visible output PCC, generation, or readiness failure on target weights; real-weight trace/runtime failure; final default latency not winning; or an exact op-contract blocker.

The final report must list the real-weight PCCs, synthetic PCCs if run, traced latencies, and which evidence actually decided the policy. Do not create a passing test suite by marking the lower-precision synthetic case as an expected diagnostic failure while selecting the slower fallback. If such a test exists, its purpose is diagnostic coverage; it is not an acceptance gate.

### OPT-013: Prove dtype policy reached the measured ops

A dtype/fidelity policy is not implemented until the measured runtime rows show it. Policy names, dataclass defaults, JSON summaries, and constructor arguments are only intent. Before accepting a reduced-precision candidate, inspect `tt-perf-report` or an equivalent profiler artifact for each dominant matmul row and confirm the row's input dtype, weight dtype, output dtype, and math fidelity match the selected policy.

This is especially important when weights are lazy-loaded, cached, rewrapped, or passed through helper modules. A stale tensor cache, fallback constructor, convenience `weight_dtype` argument, replicated load path, or helper default can silently turn a claimed BFP4/LoFi policy into BF16 or BFP8 at the actual matmul. If the profiler row says `BF16 x BF16`, the matmul is not using BFP4 weights no matter what the policy object says.

When the row dtypes do not match the policy, debug the materialization path before timing or ranking that candidate: cache key and cache directory, `LazyWeight` dtype, mesh mapper, memory config, helper defaults, explicit `dtype=` arguments to `ttnn.matmul`, and whether a precomputed device tensor is reused across policies. The final report must include the dominant matmul rows that prove the kept policy was actually exercised.

### OPT-014: Cross precision with matmul geometry

For a dominant matmul group, dtype/fidelity and block geometry are not independent checkboxes. A sweep that measures one core grid, activation shard width, `in0_block_w`, or `per_core_N` under BFP8/HiFi2 does not prove the same geometry under BFP4/LoFi, and a BFP4/LoFi run on one geometry does not reject the other material geometries. Measure the best reduced-precision policy on the geometry candidates that matter.

This matters especially when more cores shrink the activation shard and reduce the legal K block. Fewer cores with wider shards can unlock a larger `in0_block_w` or output block and beat a larger-core run even though it uses fewer workers. When `tt-perf-report` marks a dominant row `SLOW`, reports low DRAM utilization, reports no output subblock, shows `in0_block_w<=2`, or the code records a blocker for a larger-core geometry, run a precision-locked smaller-core or residual-grid candidate before accepting the stage.

The final report must make the cross-product explicit: for each material geometry candidate, list dtype/fidelity, core/grid source, input shard shape, `in0_block_w`, `per_core_N`, row latency, whole-layer latency, correctness, and kept/rejected decision. If a candidate is impossible, record the exact TTNN/op-contract blocker. Do not mix evidence from different dtype/fidelity policies to declare a geometry optimized.

## Matmul Choices

- Decode matmuls with small activations and large weights are usually DRAM-bound. Use `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
- Prefill matmuls with large M and N are usually compute-bound. Use `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` over a large 2D grid.
- When several matmuls consume the same input activation, first ask whether they should be one packed/fused projection. Common examples are Q/K/V-style attention projections, paired gate/up MLP projections, and model-specific groups that are split only for code convenience. Try concatenating or packing weights at load time, running one matmul, then splitting or reshaping the result on device. Compare the packed candidate against a well-tuned separate candidate using the best legal dtype/fidelity for each. Do not keep a packed projection if it loses because it forces worse dtype, fidelity, memory layout, or program configs. Keep separate matmuls with measured evidence, incompatible dtype/fidelity requirements, or a minimal repro showing the fused contract cannot express the model.
- For row-parallel output projections on multiple devices, do not assume local matmul plus all-reduce is best. Compare it against legal all-gather-matmul fusion, matmul-reduce-scatter fusion, and reduce-scatter plus delayed-gather patterns. Measure whole-layer latency, not only the isolated op, because residual layout changes can move cost to the next module boundary.
- When a matmul or repeated matmul group dominates decode time, local program-config search is required. Do not rely on one global helper or one conservative config for every role. Tune QKV, output projection, dense MLP gate/up/down, router, shared expert, routed MoE gate/up/down, and LM head separately when they are material costs.
- For each dominant matmul role, enumerate legal candidates for core grid, `per_core_M`, `per_core_N`, `in0_block_w`, `out_subblock_h`, `out_subblock_w`, `out_block_h`, `out_block_w`, memory configs, and compute kernel config where the op supports them. Measure traced warmed decode with real model shapes, weights, activations, dtypes, sharding, and batch/sequence settings.
- Treat decode math fidelity as independent from dtype. For each dominant projection group, compare at least the legal LoFi and HiFi2 compute-kernel configs for the same dtype before accepting the policy. This matters on newer architectures too: BFP8+LoFi can be materially faster than BFP8+HiFi2 while still passing model-level accuracy, so a BFP8-only dtype sweep is not enough.
- For 3-matmul MLP gate/up/down groups, explicitly compare the two important families when both are legal: packed/fused gate-up and separate gate plus up. The packed family reduces launches and input reads, but it can lose if it prevents BFP4/LoFi weights, creates a very wide output, adds split/slice/layout overhead, blocks a better program config, or loses activation/binary fusion opportunities available to the separate path. Keep the measured winner for the full MLP group, not the locally faster gate/up matmul row.
- For non-DRAM-sharded matmuls, try to use the largest clean core grid the shape and hardware allow. If padding weights enables a better grid or block size, try it. Padding must be mathematically inert: padded hidden/intermediate channels must be sliced, masked, or reduced away before they can affect outputs.
- `in0_block_w>=2` is only a floor. For dominant DRAM-sharded matmuls, sweep upward through legal divisors of the tiled K dimension and the input shard's K-tile width. These candidates do not need to be powers of two; `3`, `5`, `7`, `10`, `14`, or other divisors can be legal and should be considered when larger than the current setting. Do not satisfy this item by trying `1` and keeping `2`, or by stopping at `4`/`8` because of old expectations. `in0_block_w` must divide the tiled K dimension; if the current shard spec makes only small values legal, try a different shard-spec core count or working activation shard that enables larger divisors, even if it uses fewer cores. For DRAM-sharded matmuls, the compute core count is fixed by the op, so the input/output shard grid may be more flexible than it first appears. Padding weights or hidden/intermediate channels can be worth trying when it enables a better block size, but the padding must be mathematically inert: padded channels must be sliced, masked, or reduced away before they affect outputs. Changing the shard spec is usually preferable when it preserves the surrounding layout contract.
- Output subblock size should usually be at least `2x1` or `1x2` when legal - again, larger is often better so check the matmul validation code and try larger valid sizes too.
- If any `in0_block_w<=2` or output subblock sizes are <2 for a matmul that is a non-trivial percentage of the runtime, call them out explicitly in your final output summary and list the exhaustive set of things you tried to enable a larger legal value and why they failed. There's probably a better matmul setting and you should find and use it.
- If an op runs out of L1, first try to increase the core count. If that's not possible, reduce `in0_block_w`, `out_subblock_h`, or `out_subblock_w` and see which combination preserves the most performance whilst avoiding the L1 OOM issue.
- If `tt-perf-report` says a matmul is DRAM-bound and it is not DRAM-sharded, trying DRAM-sharded matmul is mandatory. You can usually figure out a way to make it work with resharding if necessary and it's also usually worth it. If you find it is not a performance win, record that you tried and why it was not in your final output summary.
- For dominant `ttnn.sparse_matmul` groups, keep routed active-expert execution and sweep legal `in0_block_w` and output-subblock candidates separately for each sparse projection role. Dense all-expert execution is a debug baseline, not an optimized result. The down path often has a different K/N shape and may need `is_input_a_sparse=True` when it consumes sparse expert activations. Also try input/output memory placement. Sparse weights may live in DRAM, but decode outputs and elementwise/reduction intermediates should stay in L1 or sharded L1 unless a measured L1-capacity blocker prevents it. Explicit `ttnn.DRAM_MEMORY_CONFIG` on decode intermediates is a performance smell; keep it only with a failed L1 attempt and before/after perf evidence. If `tt-perf-report` shows slow `BinaryNg`, `Unary`, `Slice`, `Reduce`, or other non-matmul ops around MoE, first check whether tensors are round-tripping through DRAM before retuning matmul configs. If one candidate hangs, wedges hardware, or fails PCC, reject that candidate, recover the device, and continue with other legal candidates instead of abandoning sparse program-config search.
- Leave a table for every dominant matmul search: matmul role, percent of decode time, config tried, traced decode latency, PCC or correctness result, kept/rejected decision, and reason. If a dominant matmul group has no search table, the optimization pass is not complete.

## Sparse Matmul `nnz`

`ttnn.sparse_matmul(..., nnz=N)` selects a faster fixed-count path. The sparse sender skips zero sparsity entries, while receiver and compute kernels loop exactly `N` times. This is only correct when every invocation has exactly `N` non-zero entries in the sparsity tensor.

- Treat `nnz` as an exact invariant, not an upper bound. `top_k=4` does not prove `nnz=4`.
- Use explicit `nnz` when the sparsity tensor is constructed or validated so `count_nonzero(sparsity) == nnz` for every token and batch item. This avoids the runtime inference path and can matter: GPT-OSS Blackhole decode measured about 13-16% faster with a valid static `nnz` path than with runtime inference.
- Omit `nnz` when router weights, masking, dtype conversion, duplicate routes, expert remapping, or Blackhole zero-flush can change the actual non-zero count. The runtime inference path is slower but robust.
- A wrong `nnz` can hang or wedge the device. With watcher enabled it may fail loudly; without watcher it can look like a hardware hang.
- If you omit `nnz`, record the reason and the measured decode cost. If that cost is material, try to make the exact-count invariant true without changing model semantics, then remeasure.
- Background: [tt-metal #45943](https://github.com/tenstorrent/tt-metal/issues/45943) documents the deadlock contract, [#45958](https://github.com/tenstorrent/tt-metal/pull/45958) added asserts/docs, and [#45052](https://github.com/tenstorrent/tt-metal/issues/45052) / [#45969](https://github.com/tenstorrent/tt-metal/pull/45969) show the GPT-OSS Blackhole failure and runtime-inference workaround.

## Precision And Fidelity

- Start optimized decoder with BF16 activations and BFP8 weights. Keep norms BF16.
- Try BFP8 KV cache. Keep it if PCC remains above threshold and perf/memory improve. If BFP8 KV fails, inspect the prefill fill-cache dtype path before concluding the dtype is invalid: cache-fill tensors should be explicitly typecast to the cache dtype, while decode `paged_update_cache` inputs should remain BF16/FLOAT32.
- Try BFP4 for MLP FF1/FF3; these often tolerate BFP4 well.
- Try BFP4 for FF2/down-projection, but expect it to be more sensitive. Fall back based on PCC evidence, not preference.
- If dense MLP or expert matmuls dominate decode time, the BFP4/LoFi MLP trial is mandatory before accepting the decoder. Do not defer it to datatype sweep and do not spend long on lower-impact prefill-only advice first. Measure a BF16-activation/BFP8-weight baseline, a BFP4 FF1/FF3 or gate/up candidate, and a guarded FF2/down BFP4 candidate when the model can tolerate it.
- For BFP8 weights, HiFi2 is a safe starting point, not a default answer. Try LoFi for dominant decode projection groups and keep it when full-model accuracy and qualitative generation still pass. Record BFP8+LoFi versus BFP8+HiFi2 as separate candidates, because the speed difference can be large even when dtype is unchanged.
- For BFP4 weights, LoFi is expected.
- For BF16 weights or numerically sensitive operations, use HiFi4 or FP32 accumulation where PCC demands it.
- Evaluate precision changes one group at a time so regressions can be assigned to the right tensor group.
- A small or moderate PCC drop after lowering precision may be real precision loss and can reject that dtype with evidence. A catastrophic PCC collapse, such as PCC near zero, output magnitude explosion, NaNs, or a drop from passing to about half-scale or worse, is more likely an implementation, dtype propagation, layout, cache, scale, or validation-harness bug than proof that the lower precision is invalid. Immediately investigate that failure instead of recording "BFP8/BFP4 invalid." First reproduce it with a minimal same-contract test and compare intermediate tensors to the higher-precision path. Then use `autodebug` with the failing command, logs, tensor shapes, dtype policy, and first bad tensor; use `autofix` when the suspected bug is concrete. Reject the dtype only after the debug loop shows model-visible precision loss, an exact op-contract blocker, or an unfixed bug with preserved evidence.
- Activation size matters for CCLs. Try using BFP8 activations and see if PCC (and final top-1/top-5/benchmark eval scores if run) remain high enough. Do not treat one global BFP8-activation switch as a complete activation precision sweep. Attention activations, CCL payloads, MLP/FFN activations, MoE expert activations, residual/norm tensors, and logits can have different precision needs and different performance impact. Test BFP8 activations separately for the attention path and for the MLP/FFN or MoE path under the same topology before using a global BFP8 activation policy. If global BFP8 fails, isolate which path fails instead of falling back to BF16 everywhere. If global BFP8 passes but one path is slower or less accurate, keep a mixed BF16/BFP8 policy with evidence.
- Test precision and topology together when tensors cross devices. Activation dtype changes mainly affect CCL and layout traffic, so a BFP8-activation or BFP4-weight candidate tried under one residual/collective topology does not reject it for another topology. For dominant decode paths, keep a small combined matrix crossing activation/CCL dtype, material weight dtype, compute fidelity, and topology family. When any BFP8 activation, BFP4 weight, or LoFi candidate is rejected, state the residual/collective topology it used and whether the same dtype/fidelity was tested with the best topology candidate.
- Prefer fused CCL + matmuls where possible. If a collective output feeds a matmul, try to make the tensors satisfy fused all-gather-matmul or equivalent contracts. If a matmul output is immediately reduced, try the fused matmul-plus-reduce path where one exists. Adapt shape, rank, padding, sharding, and weight layout before rejecting the fused path; a first API validation failure is evidence about the current tensor contract, not proof that the optimization is impossible.
- Otherwise use async CCLs but be careful to ensure there are sufficient semaphores - other models have some good examples of CCL helper classes that track these.
- Always test with watcher when using async CCLs, it's easy to make mistakes that end up in data corruption or hangs.

## Compute Kernel Configs

Common Wormhole-style starting points:

```python
compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

Use the architecture-appropriate config class when optimizing non-Wormhole targets.

For larger-core devices, do not hard-code a Wormhole-shaped 8x4 or 32-core decode layout as the only optimized layout. Sweep legal core grids as a coherent layout: residual/intermediate memory configs, norm program grids, matmul program configs, and shard specs must agree. A first validation error such as a shard grid not fitting an op's program grid means adapt the layout/program-grid contract or use `autofix`; it is not enough evidence to reject the larger grid.

## `tt-perf-report`

Do not use this section to collect vLLM serving-stage evidence. vLLM and optimized-vLLM stages intentionally skip Tracy, `tt-perf-report`, and device-profiler collection to protect T3K hardware.

Install and verify in the active tt-metal environment:

```bash
python -m pip install tt-perf-report
tt-perf-report --help
```

Generate a Tracy ops CSV for a signposted window:

```bash
python -m tracy -r -p -v -m pytest <test-path> -k "<selector>"
```

If Tracy collection fails, use the device-profiler fallback and post-process:

```bash
TT_METAL_DEVICE_PROFILER=1 pytest <test-path> -k "<selector>"
python tools/tracy/process_ops_logs.py --date
```

Known tooling-failure signatures and prescribed actions - do not burn hours rediscovering these:

- Tracy enrichment failing with "too many source locations" or dropped device markers on large models: the profile was too broad. Stop it, preserve the log, and rerun a reduced-layer probe with one layer per kind instead of the full stack. Do this reduced probe up front for full-model profiling rather than waiting for the full-stack profile to fail.
- Device-profiler-enabled serving appearing in a run: this should only happen accidentally, because vLLM serving stages must not use profiler collection. Kill leftover `EngineCore`/server processes and run at most one bounded health check. If `tt-smi -ls --local` or reset hangs, or if logs show remote Ethernet/ARC/ERISC failures (`Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, ARC lock/readback waits), stop profiler collection and run the T3K reset recovery procedure. Do not run a full serving-adapter profile followed by `ttnn.ReadDeviceProfiler(mesh)`. Preserve the logs as `hardware-profiler-limited`; reboot/re-acquire is a fallback after reset fails, not the first blocker conclusion.
- Watcher overflowing the ACTIVE_ETH kernel config buffer: retry with `TT_METAL_WATCHER_DISABLE_ETH=1` and record the scoped limitation.
- Transient CCL/fabric link errors immediately after a failed multi-device run: reset devices and retry once before treating it as hardware evidence.

Copy the final CSV into the artifact directory and run:

```bash
export ARTIFACT_DIR="models/experimental/diffusion_gemma/doc/optimized_decoder"
cp <ops_perf_results_*.csv> "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --csv "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.csv" \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.console.log"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --no-summary \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.txt"
```

Use the same pattern for prefill with `PERF_PREFILL` signposts and `prefill_*` filenames. If your installed `tt-perf-report` version uses different flags, run `tt-perf-report --help`, use the equivalent flags, and record the exact command. You'll have to add these signposts to your code, of course.

The `*_perf_report.txt` file is for the human-readable table. Do not redirect the stdout from a `--csv` run into that filename; `--csv` mode prints command/status boilerplate such as "Writing CSV output..." rather than the rendered report table. Keep that chatter in `*_perf_report.console.log` if it is useful for provenance.

`tt-perf-report` runs should keep advice enabled. If you also need a compact no-advice table, run that as a secondary command with a distinct filename and keep the advice-backed table in your work log and final reports.

Check time units before computing latency. Filtered `tt-perf-report` CSVs may expose `Device Time` in microseconds; raw Tracy ops CSVs often expose `DEVICE KERNEL DURATION [ns]`.

Sanity-check profiler durations against the benchmark wall time before computing roofline percentages. If a filtered `tt-perf-report` table claims multi-second device ops inside a sub-millisecond traced decode window, the duration data is invalid. Preserve the raw CSV and report the profiler failure explicitly. When the raw ops CSV has sane performance-model bandwidth columns, you may compute a labeled modeled roofline fallback from `sum(PM BANDWIDTH [ns] for decode matmuls) / measured traced decode window`; otherwise leave DRAM utilization unreported rather than publishing a bogus percentage.

## Advice Policy

For every actionable `tt-perf-report` recommendation:

- try it. If there is a good reason to reject it, record the reason;
- record before/after latency, PCC, and any watcher or correctness issue;
- keep it if it improves the target metric without unacceptable PCC or complexity;
- reject it only with evidence, then continue optimizing the rest of the decoder.

Avoid suppressing advice in the report used to guide optimization. When applicable advice remains untried, call that out as remaining work rather than implying the optimization pass is complete.

## Final Audit Checks

- No unnecessary `InterleavedToSharded`, `ShardedToInterleaved`, `reshard`, `tilize`, `untilize`, `to_torch`, or `from_torch` in the optimized runtime path.
- Decode trace replay still measures the optimized path, not a fallback path.
- Program configs and compute-kernel configs are described in the final report or a compact structured summary.
- If `supports_async_decode=True` is advertised, the split vLLM path has been exercised: `decode_forward(read_from_device=False)`, `read_decode_output(async_read=True)`, and `process_decode_output_host`.
- On-device sampling returns device tokens/logprobs through decode; host top-1 or argmax fast paths are removed or proven unused by the measured benchmark.
- PCC covers prefill and decode for every representative layer kind.
- Optimized stress runs and passes for every representative layer kind and exercised mode; skipped stress is not a passing optimized result.
- Final perf reports cover warmed prefill and warmed decode separately, except vLLM serving stages where profiler collection is intentionally skipped.
- Watcher-clean evidence exists for the optimized correctness run.
