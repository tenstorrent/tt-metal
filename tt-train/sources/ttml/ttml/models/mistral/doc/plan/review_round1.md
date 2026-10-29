# Phase review, round 1 (train-plan), 2026-10-29

Verdict: more-work-needed

## Required Work
- P1: Saved-activation figure does not match ttml's retention model; the 16-layer depth claim is unsupported.
  Evidence: plan §4 counted 264 MiB/layer. Every ttml op captures its output `TensorPtr` in its grad closure (ops/linear_op.cpp:110, ops/binary_ops.cpp:91-158, ops/rmsnorm_op.cpp:57, ops/multi_head_utils.cpp:93, ops/rope_op.cpp:214, ops/swiglu_packed_op.cpp, ops/scaled_dot_product_attention.cpp:286-295); release only in autograd/tensor.cpp:77-86 and autograd/graph.cpp:70-72. Re-summed: 786 MiB/layer with main's 5-op `LoraLinear`, 466 with `LoraApply`; 16 layers = 34.7 / 29.7 GiB.
  Required: re-derive from the retention model for both LoRA paths, restate the depth that fits, change the profiling depths or state the deallocation/recompute policy; use 31.83 GiB in numbers.py.
- P1: Multi-device work (TP=4 full-depth run) planned inside a single-chip bring-up (train-bringup SKILL "Single chip"; stage-review SKILL:73).
  Required: remove it or keep it only behind explicit recorded user authorisation; state the single-chip substitutes (loss at the deepest fitting depth vs a truncated torch model; full-depth time as the fitted projection validated against the deepest depth's wall-clock), every number labelled "N of 40 layers".
- P2: The achievable target's LoRA term assumes a fused kernel the inventory does not have; the plan's own calibration measures the existing skinny-matmul classes at 2-5 % of peak (~6.6 ms/layer, +207 ms/step, ~1.44 s, outside the ±10 % margin).
  Required: price the LoRA term at the measured rates and restate the target, or add the missing efficient LoRA path to the inventory as new work and make the target conditional on it; state the same for the provisional 70 % bandwidth assumption.

## Other Concerns
- Dispatch gap: plan used 10 µs/op; the same diary measures a ~25 µs/op host floor and ~16 µs/op exposed at full scale. Cite and justify.
- Loss definition: HF `ForCausalLMLoss` shifts labels and averages over 1023 positions with fp32 logits; ttml CE has no ignore_index. State how targets are built and that the torch reference is fed the same shifted targets.
- Useful-FLOPs convention deviates (sensibly) from the skill's literal "6 x trainable params x tokens"; call it out so MFU runs do not "correct" it.
- Block 0: with `FrozenEmbedding` the block-0 input has no node, so pruning is automatic; the "mark requires_grad=false" instruction is redundant; roofline overcounts qkv dX by 0.43 ms.
- Units in §4 mix MB and MiB. transformers 5.12.1 is a release, acceptable. Do not promise a full-depth fp64 torch reference (188 GB).

## Hard-Check Gaps
- Retention estimate is from code reading; calibrate with the allocator peak after forward at 2 and 4 layers in the composite phase.
- No measured bandwidth backs 512 GB/s nominal and 70 %; acceptable if marked provisional.

## Anomaly Ledger
- Dispatch gap 10.7 vs 16/25 µs/op: documentation; recomputed targets stay inside the margin.
- 16-layer stack claimed to fit while ttml retains every forward intermediate: more-work-needed (fixed in round 2).

## Scope Inspected
- Skills: stage-review, train-plan, train-bringup, composite-bringup. Artefacts: doc/plan.md, doc/plan/work_log.md, numbers.py, mm_eff.py, HF config, Qwen3.8 Tracy CSV. Code: every op the inventory cites (paths confirmed), autograd/tensor.cpp, graph.cpp, graph_utils.hpp, modules/lora.py, common/performance.py, models/llama/*, umales/qwen38 diffs, transformers modeling_mistral.py and loss_utils.py. Commands: numbers.py and mm_eff.py reproduced; git show/diff; two re-derivation scripts. No devices opened.

## Residual Risk
- Inventory walk found no missing op; freeze map complete. Big-matmul calibration well founded; uncertainty in the memory-bound and LoRA terms and in the depth that fits.
