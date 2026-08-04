# Session notes — spec-decode → served + hybrid-KV diagnosis (autonomous run)

Goal: execute the approved plan `~/.claude/plans/woolly-wobbling-beacon.md` — ship eager spec-decode
into the served vLLM path (~1.5×), gated behind the hybrid-KV question + W1 stall. User away; take notes.

## Prior committed work (this session, before the plan)
- `6c46e8e` — spec_decode.py adaptive-K + runtime accept-guard + multi-trace warmup hang-fix + run_bg.sh
  full-tail launcher. Opt-in, adversarial-review-clean. 32k guard 24.4→29.1 t/s/u (~1.0×).

## KEY FINDING — the plan's hybrid-KV premise is WRONG / stale
The eval called hybrid-KV a "silently dead, one-line-fix bug." The actual code contradicts this:
- MODEL is fully hybrid-wired + ON: `generator_vllm.py:105 _HYBRID_KV_CACHE_GROUPS_ENABLED=True`,
  `get_kv_cache_spec:181` emits SlidingWindowSpec(512) for 30 sliding layers, grouped-PT plumbing present
  (`_decode_pt_grouped_alloc/refresh:382/396`), `decode_forward:1090 hybrid = page_tables_per_layer is not None`.
- `model.py:347 decode_layers` ALREADY routes per-layer: `per_layer = isinstance(page_table,(list,tuple))`
  → layer i uses `page_table[i]`. So a per-layer LIST just works.
- PLUGIN looks already-fixed: `platform.py:443 support_hybrid_kv_cache → True`; `worker.py:191-195`
  already prefers the TT-prefixed arch entry (the exact mismatch the eval flagged). Only `platform.py:392-395`
  has a STALE comment saying Laguna is "served as uniform full-attention".
=> Do NOT blind-edit the plugin. Diagnose on-device first.

## On-device diagnosis (boot serve_diag, max-model-len 131072, max-num-seqs 8)
- `kv_cache_utils.py:820` Overriding num_gpu_blocks=70368744177664 (=1<<46 sentinel) with override=2568
- `kv_cache_utils.py:1308` GPU KV cache size: 164,352 tokens  (2568 blocks × 64)
- CONFIRMED UNIFORM (1 KV group): single `num_gpu_blocks_override=2568` → one `GPU KV cache size: 164,352
  tokens` aggregate, `speculative_config=None`. No per-group breakdown. So all 40 layers carry FULL KV at
  serving DESPITE the model being hybrid-wired — the plugin hybrid hook is NOT firing. Matches the eval.
- IMPLICATION: serving passes uniform `page_table` (decode_forward hybrid=False, page_tables_per_layer=None).
  So (a) spec verify can use the plain page_table — SIMPLER; (b) Bug 2 grouped-PT is correct future-proofing
  but not exercised today; (c) hybrid-KV = a SEPARATE capacity project (model wired but vLLM still uniform →
  needs real plugin investigation into why the hook doesn't fire; NOT a one-line arch-scan fix, NOT a
  spec-serving prerequisite). De-scoped from the spec-serving critical path.
- The 1<<46 sentinel = `determine_available_memory` returns the policy cap; the real block count comes from
  num_gpu_blocks_override=2568. This is the "pool sizing" issue, SEPARATE from the arch hook.

## Eager spec integration DESIGN (decode_forward in-adapter) — the hard part
The served decode path returns tokens as a DEVICE tensor read ASYNC by the plugin
(`decode_forward` returns `[tok]` when read_from_device=False → `read_decode_output`/`process_decode_output_host`
in async_decode.py). So the buffered spec tokens (host-computed) can't just be `return`ed as ints — they must
land in the SAME device `tok` buffer the plugin reads, OR the interception happens where the plugin can accept
a host token. THIS return-form compatibility is the real integration risk (silent corruption if wrong), and it
needs served iteration to get right — do NOT ship unvalidated.

Design (B==1, greedy, TT_LAGUNA_SPEC_DECODE=1, not reset_batch):
- instance state: self._spec_buf (pending committed ids), self._spec_hist (running ids), reset on reset_batch/new-req.
- buffer non-empty: write the popped id into the persistent device `tok` buffer (ttnn.copy_host_to_device_tensor),
  advance cur/ridx by 1 on device or host, return `[tok]` — NO model forward. (Must match what the plugin's
  next-step feedback expects; the normal path reuses device-advanced cur/ridx.)
- buffer empty: propose K from _spec_hist, eager verify_greedy_decode(traced=False, page_tables_per_layer=...),
  accept greedy → commit m+1; KV for accepted positions already written by the verify batched-decode; push
  [t1..tm] to _spec_buf, write t0 into `tok`, return `[tok]`.
- guard/adaptive: reuse SpeculativeDecoder logic (guard=True prevents low-accept loss).
- POSITION/KV RECONCILIATION: vLLM advances +1/step & appends 1 tok/step; we return 1/step so vLLM stays
  consistent. KV at accepted positions written during verify; rejected-draft positions overwritten next round.
  The plugin's cur/ridx device-feedback (normal path) is BYPASSED on spec steps — must set them explicitly.
- GOOD NEWS (feasibility): decode_forward ALREADY receives `prompt_tokens` + `output_tokens`
  (async_decode.py:598-601) when device-sampling → full ngram history is available (history =
  prompt_tokens[0] + output_tokens[0] for B==1). No plugin plumbing needed for history.
- RETURN-FORM (clean approach): keep the device `tok` buffer as the single source of truth. Spec round or
  buffered-pop → write the token to return into `tok` (copy_host_to_device) and return `[tok]`; the plugin's
  read_decode_output/process_decode_output_host reads it exactly like a normal decode. No trace on spec steps.
- STATUS: feasible + plumbing understood; still NEEDS-SERVED-VALIDATION (tok-write + KV consistency +
  position handling can silently corrupt if wrong). Gate default OFF. Implement + smoke-test spec-on if time.

## Work items + status
- [Bug 2] verify_forward_decode grouped-PT — IMPLEMENTING (clean; decode_layers already routes lists).
- [W1] prefill warmup (N,w) shape coverage — root cause needs on-device shape confirmation; warmup already
  warms (1,w) single-row + serving width; the row-count gap is the hypothesis. Hold edit until diagnosis.
- [Eager spec integration] decode_forward in-adapter buffer — HIGH complexity/risk; gate OFF
  (`TT_LAGUNA_SPEC_DECODE`), implement carefully, mark NEEDS-SERVED-VALIDATION.
- [Plugin hybrid-KV / pool] — diagnose first; likely already working or a pool-sizing tweak, not the arch hook.

## Serve recipe (from scripts/stage_ce_serve.sh)
LOCAL=/home/ttuser/.local/lib/model-bringup/tt-metal
env TT_METAL_HOME=$LOCAL PYTHONPATH=/home/ttuser/dev/tt-metal:$LOCAL/vllm:$LOCAL/vllm/plugins/vllm-tt-plugin/src \
  TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1 python -m models.common.readiness_check.run_vllm_server \
  --model-dir $BASE --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
  --max-num-seqs 8 --block-size 64 --max-model-len 131072 --tt-config '{trace_region_size,fabric_config:FABRIC_1D_RING}'
Board recovery: tt-smi -r all after any hard-kill; truncate readiness_vllm/server.log before reboot.

## DONE this run
- Bug 2 (verify grouped-PT): generator_vllm.py verify_forward_decode builds a per-layer device PT list
  (each = user's group row replicated to K1) when page_tables_per_layer given; verify_greedy_decode(eager)
  forwards page_tables_per_layer. Uniform path behavior preserved. Syntax-clean. NOT yet on-device-validated
  in hybrid (serving is uniform today) — uniform path unchanged so low risk.
- Hybrid-KV diagnosis: UNIFORM at serving (1 group, 164352 tok, 1.25× ctx). De-scoped as capacity project.

## Decode-trace warmup (W1 context)
warmup_model_decode captures ONE trace at B=max_batch_size (gen:1328); serving pads decode to that B, so
decode-trace lazy capture should NOT fire (matches Explore: the [3.2] warning didn't appear). => W1 lives on
the PREFILL path. warmup_model_prefill (gen:1210) warms buckets at (1,w) single row + serving width serve_w;
hypothesis = a serving prefill shape (row-count or a width/pos combo) still first-compiles under the resident
decode trace. NEEDS on-device repro (C=16 then C=1 back-to-back; watch allocator.cpp:123 + mean ITL >> p99).

## HANDOFF — remaining work (do with mesh in front of you)
1. W1 fix: after the on-device repro pins the exact unseen prefill shape, extend warmup_model_prefill to warm
   that shape (row-count set and/or the width the stall recompiled). Validate: C=16→C=1 no 8-min outlier.
2. Eager spec-serving (task #40): add `self._spec_enabled = os.environ.get("TT_LAGUNA_SPEC_DECODE")=="1"` in
   __init__; at top of decode_forward, if _spec_enabled and B==1 and greedy and not reset_batch, call a NEW
   `_spec_decode_step(...)` that: builds history=prompt_tokens[0]+output_tokens[0]; pops _spec_buf if nonempty
   (write token→device `tok`, return [tok], NO trace); else runs one SpeculativeDecoder round (eager verify,
   guard+adaptive), writes t0→`tok`, buffers [t1..tm], returns [tok]. Reset _spec_buf/_hist on reset_batch.
   Keep the normal path byte-identical when the gate is off. RISK: tok-write/KV/position handling can silently
   corrupt — smoke-test spec-on (coherent output + speedup) before trusting; do NOT default on.
3. Then vllm bench spec-on vs off (batch-1, copy-heavy), and the HumanEval/quality gate if desired.
