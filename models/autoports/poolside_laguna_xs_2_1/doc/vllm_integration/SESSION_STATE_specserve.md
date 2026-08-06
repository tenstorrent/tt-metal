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

## On-device results (serve_diag, uniform, APC-on, 2026-08-03 21:08)
- BASELINE decode HEALTHY: ISL1024/OSL128/C1 = 27.4 tok/s, Mean TPOT 35.5ms => 28.2 t/s/u; Mean ITL 35.5 ≈
  P99 39.9. C16 = 100.3 agg, Mean ITL 38.1 ≈ P99 41.4. Trailing C1 (post-C16) = 28.0, ITL 35.0 ≈ P99 38.8.
- W1 STALL DID NOT REPRODUCE at ISL=1024 (C1→C16→C1). Mean ITL ≈ P99 throughout — no 8-min outlier. The
  original stall was ISL up to 130k / C16 then 1024/OSL1024/C1: it needs the LONG-CONTEXT / wide-page-table
  transition, not short ctx. => W1 is an intermittent long-ctx shape-transition issue; can't fix blindly.
  Next repro to try: ISL 32768 C8 (wide PT + concurrency) then ISL 1024 OSL512 C1, watch mean ITL >> p99.
- LONG-CTX REPRO (32k C8 then 1k C1): 32k/C8 = Mean ITL 131 >> P99 43.6 / Median 39 (LOOKS like the stall),
  BUT the server log shows NO unsafe-alloc / compile events in that window (the one unsafe warning was at
  21:08:41 during the HEALTHY baseline C1). => the 32k/C8 mean-ITL spike is PREFILL-SATURATION scheduling
  contention (8 concurrent 32k prefills block each other's decode — the known long-ISL-all-at-once artifact,
  STATUS §Concurrency), NOT the W1 single-request recompile stall. Trailing 1k/C1 was healthy (35ms).
- CONCLUSION: the ORIGINAL W1 (single-request 25→1.9, ISL1024/OSL1024/C1) did NOT reproduce this session in
  targeted tests. It is a RARE INTERMITTENT (fired once in a long full sweep). Cannot fix blindly without a
  deterministic repro. Handoff: instrument decode_forward/_prefill to log every shape-key miss + allocation,
  then run the ORIGINAL full sweep order to catch it; or add (1..max_num_seqs, w) prefill-warmup coverage
  defensively (cheap, matches the leading hypothesis) and re-run the sweep to see if it recurs.

## DECISIVE BLOCKER for in-adapter spec-serving (found reading _decode_state + KV write path)
The spec verify (verify_forward_decode) writes KV for ALL K+1 candidate positions P..P+K (look-ahead), via
paged_update_cache against the page table. But vLLM's scheduler allocates KV BLOCKS INCREMENTALLY based on
committed tokens — from its view, decode_forward returns 1 token/step, so it has only allocated blocks up to
the current position. So positions P+1..P+K may map to blocks vLLM has NOT allocated yet (or belong to another
request) -> the look-ahead KV writes go OOB / corrupt. This CANNOT be fixed inside decode_forward alone; it
needs the scheduler to allocate K blocks ahead for the speculative window (exactly what vLLM's native spec
machinery does — and which platform.py:562 hard-gates OFF for TT).
=> In-adapter eager spec-serving is NOT a contained change. Options: (a) bounded-K scheme that only speculates
within already-allocated block slack (limits/complicates K); (b) do the real work: un-gate vLLM spec + build
the multi-query-per-request decode path (large plugin project, the integration-scope agent's "out of scope").
The standalone driver avoids this entirely (whole context pre-allocated as an identity page table).
=> This is why spec-serving is deferred, not just "risky." Documented so the next session doesn't rediscover it.

### RECOMMENDED unblock (no scheduler surgery): bounded-K within the current block's slack
block_size=64 and vLLM allocates a FULL block at a time, so the current block already has allocated slack
(avg ~32 slots, = 64 - (pos % 64) - 1). If we cap the speculative window K <= slots_remaining_in_current_block
(derivable from the position + page table on the host), every look-ahead KV write P+1..P+K lands in an
ALREADY-ALLOCATED block -> no OOB, no scheduler change. Adaptive-K + the guard already handle a per-step K
cap, so this slots in cleanly: pass K_cap = 63 - (pos % 64) into the spec round each step (K shrinks to ~0
right before a boundary, where the next committed token allocates a fresh 64-slot block). This is the
concrete path to make in-adapter eager spec-serving correct. STILL needs the eager-verify-under-resident-
decode-trace interaction validated (may relate to W1) + greedy-parity smoke test. Implement with mesh present.

## PHASE-2 PROBE RESULT (2026-08-04, on device) — eager spec-serving FEASIBLE but MARGINAL
Ran a gated one-shot probe (`TT_LAGUNA_SPEC_DECODE=probe`) that executes ONE eager batched-decode verify
under the RESIDENT decode trace, logging to `_runs/spec_probe.txt` (MPI-worker stdout isn't captured in the
readiness log). Findings:
- **Eager verify under the resident decode trace does NOT hang/deadlock** — the core Phase-2 unknown: answered
  YES, feasible. (allocator.cpp:123 warns but the op completes and serving continues coherently.)
- **Served decode is padded to B=max_batch_size (B=8), NEVER B==1**, and reset_batch=True on the first step.
  So the plan's "B==1 greedy" gate is WRONG — spec must operate on ONE ROW of the padded B=8 batch. Design
  implication for the full loop.
- **Env passthrough:** `TT_LAGUNA_*` is NOT in the plugin's default worker allowlist (`launcher.py:261`
  default_env_patterns = VLLM_*/MESH_DEVICE). Pass `"env_passthrough": ["VLLM_*","MESH_DEVICE","TT_LAGUNA_*",
  "TT_METAL_*","PYTHONPATH"]` in `--tt-config` so model-side env flags reach the worker.
- **Warm eager verify ≈ 123 ms** (compile 219, warm 121/125) vs a **~35 ms traced decode step** = ~3.5×.
  Break-even: a spec round costs ~123 ms and commits m+1 tokens vs (m+1)×35 ms native → **wins only if mean
  accept ≥ ~3**. Real agentic accept ~2.5 → **break-even/marginal**, NOT the standalone traced ~2×. Decode is
  dispatch-bound; eager verify pays the full host dispatch tracing removes.
- **VERDICT: do NOT build the full eager in-adapter loop** — it's a large padded-batch change for a
  break-even result. The real win needs the TRACED verify served, which requires solving the two-resident-CCL-
  trace deadlock (decode trace + verify trace) — release/recapture on spec on/off, or decode-trace omission
  for the spec-eligible batch. That is the true Phase-2 follow-up, gated on the trace-coexistence fix.
- Probe code (gated OFF by default) is committed as the reusable feasibility harness.

## PHASE-3 HYBRID-KV ROOT CAUSE (2026-08-04, on device) — plugin never calls the model hook
Definitively pinned (always-on diagnostic in get_kv_cache_spec that writes to _runs/kv_spec.txt):
- The model's `get_kv_cache_spec` is **NEVER CALLED at serving** — kv_spec.txt stays empty across boots while
  KV init completes with `num_gpu_blocks_override=2568` / 164,352 tok / "concurrency 1.25x" (the single-group
  signature). So vLLM uses `_build_default_kv_cache_spec` (one FullAttentionSpec) → 1 group → uniform full-KV.
- RULED OUT: the model + HF config are CORRECT (host-side check: layer_types = 10 full + 30 sliding,
  sliding_window=512, 40 layers); `_is_sliding` would emit a proper hybrid spec IF called. And it is NOT a
  vLLM `unify_hybrid_kv_cache_specs` collapse (the hook never runs, so there's nothing to collapse).
- ROOT CAUSE = PLUGIN hook resolution: `worker.py:_try_get_spec_from_model_hook` (:177) does not reach
  `LagunaForCausalLM.get_kv_cache_spec`. Likely `ModelRegistry.resolve_model_cls(arch)` returns a
  class/wrapper on which `getattr(model_cls, "get_kv_cache_spec", None)` is None (or arch resolves to the
  non-TT class). NEXT: instrument worker.py:194-208 (log arch, model_cls, hasattr) — but that edits the
  uncommitted `.local` plugin, so FORK/BRANCH first. Capacity-only (no decode-speed gain): frees DRAM
  4.30→1.08 GB/dev + unblocks 262k, but decode t/s/u is batch-flat.

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

---

## FINAL — traced spec-serve device validation (2026-08-05): DEADLOCK FIXED, but TRACED VERIFY BLOCKED

Executed the traced-spec-serve plan (nested-hatching-corbato) on device (P150x4, --max-num-seqs 1, greedy).

**SOLVED — the plan's one hard blocker (two-CCL-trace deadlock).** warmup_model_decode, when
`TT_LAGUNA_SPEC_DECODE=1`, captures ONLY the verify traces (warmup_verify_decode_multi, K1=1..k_max+1) and
OMITS the normal decode trace. Boots HEALTHY every time, no mesh hang. Log line: "captured verify traces
K1=1..5; normal decode trace OMITTED". All batch-1 greedy decode routes through the verify traces (K1=1 =
native step, K1=K+1 = spec step). This infra is correct and reusable.

**Also fixed:** greedy served path gets NO prompt/output history from the plugin (penalty-gated,
model_runner.py:1051) → stash prompt at prefill + accumulate history in-adapter; detect new request by
position discontinuity (reset_batch fires every decode step). k_cap off-by-one → `63 - ((len-1)%64)`.

**BLOCKER — traced batched-verify is intermittently incorrect (device/kernel-level).** Bisection with
`TT_LAGUNA_SPEC_TRACED=0` (eager) + `TT_LAGUNA_SPEC_DEBUG=1` (shadow-eager per-row id compare):
- EAGER batched-verify (verify_forward_decode): bit-correct on P-fib (40 tok) → upstream logic (history,
  positions, accept, page-table refresh #49) is all CORRECT.
- TRACED batched-verify (K1>=2): ~1/38 rounds a GROSS per-row id error vs eager reading the same KV
  (e.g. K1=2 row0 traced=81865 vs eager=1915; K1=5 row2 268 vs 419; K1=4 row1 163 vs 86). The error occurs
  even on ROW 0 (the anchor — no cross-row data dependency), so it is NOT a sequential-KV-write ordering
  problem; it is an intermittent fault in the traced batched decode itself (on-device Sampling1D / reduce
  over the K1 batch, or a stale-buffer read after execute_trace). When the fault hits the anchor's own
  position, that KV is never rewritten (later anchors are at higher positions) → cascades to garbage output
  (Korean/`#+#` gibberish, not a valid-greedy near-tie). Full-accept copy-heavy rounds mask it (no rejected
  KV, and the model just repeats the prompt); partial-accept exposes it. Capping K (k_max=2 → K1<=3, or even
  k_max=1 → K1<=2) does NOT avoid it — the fault appears at K1=2.

**Verdict:** traced spec-serve cannot ship until the batched-decode-trace fault is fixed at the model/kernel
layer (reproduce with a standalone K1>=2 traced-verify vs eager-verify loop over many iters — the shadow
diagnostic is the repro). Eager spec-serve is correct but ~123ms/verify vs ~35ms native decode → net SLOWER,
not shippable. **Serve the spec-off baseline** (no TT_LAGUNA_SPEC_DECODE). Other throughput levers for
"agents work well" remain: cap per-step max_output_tokens (already helps), the shipped decode-perf wins
(k_chunk=**64** SDPA — NOT k128, which is lossy and unshipped — + fused rope/reduce), and trimming reasoning
on tool turns.

> **SUPERSEDED (2026-08-06).** The BLOCKED verdict above (written 14:58) was resolved the same day:
> commit `593942d378a` (17:49) fixed the traced batched-verify fault — the intermittent per-row error was the
> on-device Sampling1D; forcing argmax (k/p/temp=None) + pad-to-32 makes traced verify bit-correct. Commit
> `fa8d3c5a9b9` (18:26) then measured **~3.3× on a high-copy reproduce-block workload** (spec-off ~30–31 vs
> spec-on ~100–104 t/s/u, mean 4.94 committed/round; workload-dependent — unique text ≈1.0× = no speedup and
> no regression). Traced spec-serve is therefore VIABLE (opt-in, batch-1 greedy), not blocked. Caveat that
> remains: spec-decode + **hybrid KV** is incompatible with the traced path (uniform-only page-table refresh)
> and now fails loudly via a hard guard in `verify_greedy_decode` (audit item 4).

Repro/diagnostic env (all in tt/generator_vllm.py, committed 1d350285d30):
`TT_LAGUNA_SPEC_DECODE=1` (on), `TT_LAGUNA_SPEC_K=<k_max>`, `TT_LAGUNA_SPEC_TRACED=0|1`,
`TT_LAGUNA_SPEC_DEBUG=1` (logs `[laguna spec] TRACE-vs-EAGER MISMATCH ...` per divergent round).
