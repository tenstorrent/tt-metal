# GLM-4.7-Flash functional decoder — work log

Stage: functional-decoder (single Blackhole chip, 1x1 mesh, device 0).
Target: zai-org/GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`, transformers 5.12.1).
Snapshot: /home/stisi/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67
Started: 2026-08-31.

## Architecture facts (from installed modeling_glm4_moe_lite.py + config.json)

- 47 decoder layers; layer 0 dense MLP (intermediate 10240), layers 1..46 MoE.
  Layer 47 in the checkpoint is the MTP (`num_nextn_predict_layers=1`) head that HF
  itself drops (`_keys_to_ignore_on_load_unexpected = [r"model\.layers\.47.*"]`) —
  excluded from the decoder contract here, recorded in context_contract.json.
- Attention is DeepSeek-style MLA: q_a(2048->768, no bias) + RMSNorm(768, eps 1e-6)
  + q_b(768->20*256); kv_a(2048->576, no bias), kv_a_layernorm RMSNorm(512, eps 1e-6),
  kv_b(512->20*(192+256)); o_proj(20*256->2048). qk_head_dim=256 (192 nope + 64 rope),
  v_head_dim=256, 20 heads, scaling=256^-0.5.
- RoPE: rope_interleave=True (DeepSeek/meta-interleaved pairs), rotary dim 64
  (`attribute_map head_dim -> qk_rope_head_dim`), theta 1e6, no scaling,
  attention_scaling=1.0, max_position_embeddings=202752.
- input_layernorm / post_attention_layernorm: RMSNorm(2048, eps=config 1e-5).
  NOTE the q_a/kv_a layernorms use the HF class default eps=1e-6, not config eps.
- MoE: 64 routed experts (SwiGLU, moe_intermediate 1536), top-4, `noaux_tc` routing:
  fp32 router logits -> sigmoid -> +e_score_correction_bias (fp32 buffer) for
  selection only -> top-4 -> weights = sigmoid scores gathered at selected indices,
  normalized (sum + 1e-20), * routed_scaling_factor 1.8. n_group=topk_group=1 so
  group masking is a no-op. 1 shared expert (intermediate 1536) always added.
- HF checkpoint expert keys are per-expert `mlp.experts.{e}.{gate,up,down}_proj.weight`
  (the HF module fuses them at load; the checkpoint layout is the canonical contract).

## Key design decisions

1. **Compressed-latent MLA cache (576/token/layer, 1 KV head)**: an MHA-style cache
   would cost 20 heads x 256 x 2 (K+V) x 2B = 20.5 KB/token/layer -> ~962 KB/token for
   47 layers, capping a 32 GB single chip at ~1.2e4 tokens of context next stage. The
   absorbed-MLA latent cache costs 1.15 KB/token/layer (bf16), which is the only
   contract under which the advertised 202752 context can survive to the single-chip
   full model. Deepseek_v3 mla1d.py is the in-repo proof of this exact op pattern.
2. Prefill: absorbed form, `ttnn.experimental.paged_fill_cache` (block-aligned chunks)
   + `ttnn.transformer.chunked_flash_mla_prefill` per chunk with chunk_start_idx.
3. Decode: `ttnn.experimental.paged_update_cache` +
   `ttnn.transformer.paged_flash_multi_latent_attention_decode` (V reuses cache latent,
   head_dim_v=512), tensor current positions, traceable.
4. RoPE via `ttnn.experimental.rotary_embedding_llama` with meta-interleaved cos/sin
   tables (GLM's rope_interleave=True is exactly meta-style; no weight permutation),
   same as models/demos/deepseek_v3/tt/rope.py does for DeepSeek.
5. MoE via `ttnn.sparse_matmul` (GPT-OSS active-expert pattern):
   decode = per-step union sparsity over the batch from the dense routing-weights
   tensor (nnz=None per tt-metal#45943 flush-to-zero deadlock note);
   prefill = dense all-expert compute with all-ones sparsity (static nnz), routed
   weights applied post-hoc — same as models/demos/gpt_oss/tt/experts/prefill.py.
6. Router on device: linear (fp32 acc) -> sigmoid_accurate -> +bias -> ttnn.topk(bf16)
   -> scatter ones -> mask*scores -> normalize -> *1.8. bf16 topk can in principle flip
   near-tied expert selections vs HF fp32; measured, see README.
7. Functional dtype policy: bf16 weights/activations, HiFi4 + fp32 dest acc for
   correctness; routed experts default `bfloat8_b` (probe: PCC 0.9999 vs 0.9843 for
   bf4). `expert_dtype` is a knob; a bf4 arm is tested and recorded because the
   full-model stage mandates bf4 experts (doc/probe/README.md).

## Session log

- 2026-08-31 16:41 device smoke: `ttnn.open_device(0)` on Blackhole OK (4x p300c
  visible via tt-smi; single-chip stage uses device 0 only).
- Stats collection (CPU): tests/collect_stats.py -> tests/weight_stats.json
  (dense 12 tensors, moe 206 tensors; activation std 0.0142 dense-input /
  0.0161 moe-input from 1024 real README tokens through embed + real layer 0).
- Bring-up bugs found and fixed during dev_bringup iteration:
  1. sparse_matmul requires an explicit MatmulMultiCoreReuseMultiCast1DProgramConfig
     -> added `_sparse_pc` mirroring gpt_oss `_build_matmul_config`.
  2. ttnn.scatter rejects fp32 TILE input -> scatter runs in bf16, mask typecast to fp32.
  3. ttnn.slice can alias the input buffer (full-range/view fast path); explicit
     deallocate of a page-table slice freed the page table -> never explicitly
     deallocate page-table slices.
  4. paged_flash_multi_latent_attention_decode default program config exceeded
     MAX_TREE_REDUCTION_ROUNDS on the 13x10 Blackhole grid -> explicit
     SDPAProgramConfig(k_chunk=128, max_cores_per_head_batch=8).
  5. Router: bf16 logits+topk agreed with HF fp32 top-4 on only 29% of synthetic
     tokens (scores cluster at sigmoid~0.5 within bf16 ulp). Fix: fp32 router
     weight + fp32 linear/sigmoid/bias, then per-token mean-centering before the
     bf16 ttnn.topk (rank-preserving; resolution applies to the spread not the
     offset) -> 99.56% agreement, remaining flips proven sub-ulp ties.
  6. My torch latent-cache reference forgot the input_layernorm (reference bug,
     found via a 3.48x rope-part scale mismatch; TTNN path was correct).
- Validation snapshots (synthetic weights unless noted, bf8 experts):
  dense prefill S=128 PCC 0.999992, cache 0.999990, decode 0.999994
  moe   prefill S=128 PCC 0.999657, S=17 0.999953, S=3000 (3 chunks) 0.999416
  moe   decode PCC 0.99997x; traced decode replay 0.99997x, bit-identical repeat
  moe   REAL weights S=512: prefill 0.999188, decode 0.99997x except pos=514 at
        0.9766 -> proven router tie: HF 4th-5th biased-score gap 0.000246 <
        bf16 ulp 0.000488 at centered magnitude (probe output in README)
  dense REAL weights S=512: prefill 0.999992, decode 0.999993
  moe   REAL weights + bf4 experts S=512: prefill 0.997020, decode 0.9975-0.9982
        (plus the same pos-514 tie) - bf4 deployment arm beats 0.995 at layer level
  batch B=4 mixed non-aligned lens [96,130,64,200]: prefill 0.9983-0.9995,
        decode all users 0.99997x
- batch32: paged_update_cache static CBs scale as B*Wt per core (output CB) and
  at B=32, kvpe_dim=576, bf16 cache exceed Blackhole L1 together with the input
  shard. Fixes: (a) keep the decode rope transformation matrix DRAM-resident and
  shard transiently, (b) free cos/sin/trans before the update, (c) split the
  update into <=16-user groups (op maps one user per core; user/pos/page-table
  row slicing is exact). All three kept; batch32 passes.
- Tie-exemption window widened from 2 to 4 bf16 ulp after a measured 2.11-ulp
  flip (user 4 pos 61, batch32): budget = 0.5 ulp typecast rounding per
  candidate + bf16-input-boundary perturbation of the fp32 scores.
- 2026-08-31 17:33 full suite: **22 passed** (incl. real weights moe+dense,
  bf4 experts, batch 8/32, traced decode + bitwise determinism, cache content
  vs linear reference, prefill determinism, runtime fallback tripwire).
  Log: doc/functional_decoder/logs/pytest_functional_decoder.log
- Long-context ladder step 1+2: absorbed-MLA window reference certified vs HF
  at S=256 (PCC 1.00000000); prefill S=8191 vs full HF fp32 reference PCC
  0.999430, decode at 8191/8192 PCC 0.99998 (log: logs/pytest_long_small.log).
- tt-perf-report installed into python_env (pip via ensurepip).
- Perf (tracy) attempt 1: `python -m tracy -r -p -v -m pytest tests/test_perf.py`
  ran 4/4 tests but in-run report generation asserted twice:
  (a) modern path: host-recorded trace-capture op (FillPadDeviceOperation
  731136) had no device rows; (b) legacy path: device profiler buffer overflow.
  Fix: ttnn.ReadDeviceProfiler(device) flushes between compile/warm/measured
  phases in test_perf.py. Attempt 2 succeeded; ops CSV
  generated/profiler/reports/2026_08_31_17_37_57/ copied into doc tracy/ dirs;
  tt-perf-report tables per kind and mode with signpost windows.
  Warmed wall clock: moe prefill 2048 = 268.5 ms (7628 tok/s), dense 19.3 ms;
  traced decode @ctx1024: moe 1.576 ms/tok (device 1.167), dense 1.030 (0.923).
- Watcher run (TT_METAL_WATCHER=10, separate from profiling): 4 tests passed,
  watcher.log clean (6 dumps, no exceptions/asserts/sanitize, healthy stacks).
- 2026-08-31 17:45 launched the full-context 202751-token prefill + decode
  evidence run (tests/test_long_context.py::test_full_context_202k).
- INFRASTRUCTURE RECOVERY (not a model result): the runner session terminated
  while the 202k test was mid-run (chunk 91/99 enqueued), killing the pytest
  and leaving the board wedged: a fresh `ttnn.open_device(0)` threw
  `TT_THROW risc_firmware_initializer.cpp:1542`. Sequence per tt-device-usage:
  killed stale tracy viewer (serve_wasm.py); `timeout 60 tt-smi -ls --local`
  (8 Blackhole entries, OK); `timeout 180 tt-smi -r` (exit 0); list again (OK);
  open/compute/close smoke OK. Resumed the stage; 202k run relaunched.
- Long-context finding + verified fix (autofix loop, single hypothesis):
  SYMPTOM: 202751-token prefill windows degrade smoothly with attended K:
  dense control start/middle/end = 0.99999 / 0.99812 / 0.99360 (all 32 end
  rows uniformly ~0.9936, no routing involved), while the *decode* op over the
  IDENTICAL full 202k cache is exact (dense 0.9999946, moe 0.9999766). Cache
  content exact (0.99999). => localized to chunked_flash_mla_prefill
  accumulation. HYPOTHESIS: bf16 flash accumulator (fp32_dest_acc_en=False in
  ck_flash) drifts across ~1584 k-chunks. EXPERIMENT (/tmp/probe_drift.py,
  dense 202751, only fp32_dest_acc_en flipped): middle 0.99812 -> 0.999939,
  end 0.99360 -> 0.999704. VERDICT: verified; fix = split configs
  ck_flash_prefill (HiFi4 + fp32 acc) vs ck_flash_decode (HiFi4, no fp32 acc,
  proven exact). Prefill wall cost at 202k: 72s -> 98s (dense).
- moe 202k window analysis machinery added: every below-bar row must be either
  a sub-ulp router tie or exactly reproduced by an alternate top-4 subset of
  the reference top-6 experts (utils.explain_row_as_routing_flip) - proving
  routing-flip vs numerics. Dense 202k run acts as the no-routing control.
- FINAL 202k results (both kinds PASS, logs/pytest_long_202k.log,
  long_context_{moe,dense}.json):
  dense: cache 0.999988; windows 0.999989/0.999939/0.999704 with 32/32 rows at
  bar each; decode@202751 0.999995; prefill 70.8 s (2862 tok/s).
  moe: cache 0.999989; windows agg 0.999463/0.998914/0.994916 with 31/31/28 of
  32 rows at bar and every below-bar row proven tie or exact alternate-top-4
  routing (0 unexplained); decode@202751 0.999977; prefill 95.7 s (2119 tok/s).
- Final full suite on the fixed code: 22 passed
  (logs/pytest_functional_decoder.log). 8k anchor re-run: prefill 0.999458,
  decode 0.99998 (logs/pytest_long_small.log). Tracy perf re-run on final
  config: same numbers (moe decode 1.588 ms/tok wall / 1.167 ms device; moe
  prefill 7629 tok/s warmed at S=2048); tables regenerated under tracy/.
- context_contract.json finalized: supported context 202752, tested prefill
  202751 + decode at position 202751, capability_reduction: none.
- Stage review round 1: more-work-needed with two P2s (watcher evidence stale
  vs the fp32-acc fix; README numbers from pre-fix runs) + concerns. All
  addressed:
  1. Removed two latent slice-aliasing deallocs (prefill cos/sin slices of the
     persistent rope tables; sparsity_e slice of prefill_sparsity_ones) -
     same class as the page-table bug #3.
  2. utils "ulp" docstring corrected: the computed quantum is half the
     conventional bf16 spacing, so the factor-4 window = 2 bf16 ULPs.
  3. 202k window analysis strengthened: sub-ulp-tie rows are no longer a
     bypass; every below-bar moe row must pass the alternate-top-4
     reconstruction (tie recorded as annotation only).
  4. Added test_full_context_aligned_202752: prefill at exactly S=202752,
     cache 0.999989, final window rows 202720..202751 28/32 at bar,
     4 explained, 0 unexplained (long_context_aligned_202752.json).
  5. Preserved the fp32-acc A/B repro as tests/probe_fp32acc_drift.py.
  6. test_long_context docstring JSON names fixed.
  7. Reran on the final code: full suite 22 passed; 202k moe+dense+aligned
     3 passed (moe windows agg 0.999463/0.998914/0.994916, 0 unexplained under
     the strengthened proof; dense control 32/32 rows at bar everywhere).
  8. Watcher rerun on the final code at the CI-standard TT_METAL_WATCHER=2,
     now also covering the traced decode test: 5 passed, 20 dumps, 0
     exception/assert/fault/sanitize lines (logs/watcher, pytest_watcher.log).
  9. README refreshed from the final logs (bf4 0.997074 prefill, decode steps
     0.99746-0.99820 agg 0.99498; dense prefill 19.6 ms; watcher section;
     limitations: traced decode at batch 1, synthetic long-context caveat).
- Stage review round 2: **clean-pass** (review record: stage_review.md).
  Post-review non-gating cleanups applied: probe_fp32acc_drift.py knob renamed
  to ck_flash_prefill (live repro against the final code, arms
  baseline/fp32acc); decode-at-max-position branch in test_full_context_202k
  aligned with the window rule (tie = annotation, not bypass). Neither path
  was exercised by the recorded evidence runs.
- Commit: repo pre-commit hooks reformatted the stage files (black/isort/
  autoflake/whitespace; behavior-neutral, 4-test device smoke re-passed
  afterwards incl. traced decode), pytest.raises replaced with the repo
  expect_error fixture, and three raw artifacts exceed the 500 KB hook limit:
  tracy/{moe,dense}/ops_perf_results.csv (7.4 MB each, gzip still 859 KB) and
  logs/watcher/generated/watcher/watcher.log (775 KB). Those three stay
  disk-only (documented in README); watcher.log.gz (27 KB, bit-exact) is
  committed in their place.
- LOCAL CHECKPOINT COMMITS (never pushed), repo /home/stisi/tt-metal,
  branch ttmodelmanager/glm47-flash-probe, stage-owned paths only
  (models/autoports/zai_org_glm_4_7_flash), all pre-commit hooks passed:
  1. 11d5578c175cdbe63218363fb9168205e6b258e0 - functional decoder stage
     (78 files, ~43.6k insertions; originally 84282d24, amended once to fold
     in this work log's commit record).
  2. (follow-up) work-log SHA record - see `git log` on the branch.
