# Full-model optimization checklist

The inherited decoder policy is a constraint of this stage, not an invitation
to reopen the datatype frontier. Decoder implementation and its rejection
ledger remain byte-identical to the completed full-model starting commit.
The full-model boundary audit preceded tuning in work_log.md; mesh_plan.md
records complete ownership, collectives and layout contracts.

| Applicable item | Decision and evidence |
|---|---|
| Complete-model warm B1 and largest supported batch | baseline.json / baseline_resident.json vs optimized_final.json; B32 masked_control_final.json vs selective_batch32_perf.json; all_active_final.json. Real full stack, diverse per-slot tokens, 128 decode steps. |
| Correctness, context and non-aligned lengths | AIME24 top-k gates; prefill_resume_check.json (513, aligned and non-aligned starts); full-model capacity_final.json (262143/262144); mixed B32 logical lengths33..64; prefill_interleaved.json/prefill_dram.json at128/513/16385/32769. |
| Full sharding/topology audit | mesh_plan.md; embedding hidden TP, LM-head vocabulary TP, local final norm, candidate-only sampling gather; selected stack residual maintained. |
| Coherent collective families / fused CCL / row projection | Preserved measured separate/persistent/AG+MM/MM+RS/both families in ../optimized_multichip_decoder/topology_decisions.md and current_hidden_residual_audit. The selected local full-hidden residual avoids boundary gathers. Terminal head is column-parallel; row-parallel would introduce full-vocab partial reductions. |
| Persistent CCL | Existing shared MultichipCollectives, BF8 ping-pong receives and captured output addresses. No new per-token allocations or changed topology. Previous all-output-persistent distributed family was measured slower; no fallback to a rejected stream. |
| Projection packing | Existing packed attention projections and larger-prefill packed SwiGLU retained. Decode MLP gate/up stay separate after adapted split/packed experiments; stage5 topology ledger, packed_mlp_audit. No changed decoder role calls for retuning packing. |
| Dominant matmul geometry and readers | Prior role-specific decoder sweeps/rejections preserved. New head: head_split4_complete.json and head_unsplit_r34.json plus failed head_sweep/head_unsplit controls. Readers1/2/3/4, K2/4/8/16, output64/110, split adaptation; BF8/HiFi2 throughout. Three-reader K2/target64 selected. |
| Norm/residual layout | Final norm directly consumes existing origin8x4 WIDTH_SHARDED L1; DRAM-sharded head removes input conversion. Interlayer layout untouched. |
| Precision/fidelity / real inputs | Same BF8 weights, attention HiFi2, MLP LoFi, BF8 CCL/KV, FP32 recurrence. Prior separate attention/MLP fidelity/BFP4/KV experiments and rejection evidence remain authoritative. Selected head logits PCC0.99999994, max absolute difference0.03125 and same argmax on real reduced-model activations; full-model AIME gates independently pass. No broad full-model datatype search; stage8 owns Pareto selection. |
| Physical dtype proof | Final decode raw CSV contains matmul input/weight/output dtype and fidelity attributes; compare with preserved selected policy. LMHead1D output dtype BF16 is distinct from LazyWeight BF8. |
| SDPA/composite kernels | Existing adaptive paged SDPA, fused GDN and fused convolution/preparation retained. New inactive-copy kernel removes full-pool merge traffic while preserving exact bits, not model math. |
| Logical batch / tile padding | Native capacity1..32 distinct from physical32 sampler rows. Masked B32/31-active and all-active tests, reduced B4 lifecycle and isolated B1/B3/B32 copy. |
| LM head / sampler / feedback | Whole vocab shard legal without semantic padding; bank storage padding is internal and discarded by logical tensor shape. Split greedy0.571ms vs force-argmax2.609ms on identical real logits. No full-vocab gather, generic TopK or force-argmax in measured path. |
| Full-model gap | New resident no-readback API; lower-bound and terminal accounting in profile/layer_stack_accounting.json. Masked state overhead isolated and reduced with exact selective copy. |
| Prefill/TTFT and buckets | Before/after complete generator length sweep; existing phase2D512..1024, DRAM norm>4096, DRAM projections>8192 and linear32768/full16384 chunks preserved. Prepared buckets and cache counts recorded. Reduced prefill profile plus prior unchanged per-chunk decoder tables. |
| Trace / fallback / stress | runtime_audit.md; tracked B32, tracked resident replay, sampling transitions, positive-control tracker rejection, watcher, zero-transfer resident counters, host contract tests. No whole-capture exemptions added. |
| Qualitative | Six shared prompts on selected head/sampler exactly preserve previous-stage tokens/settings/prompt IDs; refreshed raw and AIME24 autoregressive output. qualitative_verdict.json and previous matched HF controls. |
| Profiler and advice | Reduced real layers0/3; separate prefill/decode tables and raw CSV windows; narrow B32 guard-only before/after trace. profile/README.md resolves advice and separates instrumentation from quiet-host rankings. |
| Dense/MoE | Dense model; expert routing, sparse matmul, NNZ and EP requirements do not apply. |
| vLLM | Excluded by the user. Serving-ready generator contracts are preserved and tested directly. |

No claim that each inherited decoder candidate was rerun in this stage. The
unchanged source and preserved selected contract make its measured rejection
ledger applicable. Historical stage5 stop-rule handoffs remain historical
handoffs; they are not newly measured rejections. This stage evaluates and
repairs complete-model costs absent from the decoder-only measurement.
