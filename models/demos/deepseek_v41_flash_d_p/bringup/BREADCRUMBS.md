# deepseek-ai/DeepSeek-V4.1-Flash bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference, attempt 1 (2026-10-14)

What: the CPU reference and the HF oracle.
- `reference/kernels.py`: CPU port of the checkpoint's tilelang `inference/kernel.py` (act_quant ue8m0, fp4 e2m1
  quantize-dequantize with ue8m0 / e4m3 scales, fp8 / fp4 GEMMs, sparse_attn with sink, hc_split_sinkhorn), plus the
  shared GEMM tiling (`linear`, `expert_linear`, `grouped_out`, `index_scores`, `einsum`).
- `reference/weights.py`: checkpoint by index + memory map (nothing loaded whole), fp8 32x32 / fp4 dequant,
  `EngramTable` (memory-mapped rows), `ExpertPrefetcher` (8 threads read + dequantize experts ahead).
- `reference/ngram.py`: engram n-gram hashing, chunked with a 3-id tail; equal to `inference/engram.py` (checked on
  1000 prompt tokens in 333-token chunks, both engram layers).
- `reference/ds41_ref.py`: the reference (`Config` roles, `State`, `block_steps`, `DS41Reference`); docstring has the
  block graph, boundary layouts and state layouts.
- `reference/hf_oracle.py`: `inference/model.py` imported unmodified from the checkpoint, kernels.py as its `kernel`,
  experts `LazyExpert` (model.py's own Expert.forward on the packed weights), engram memory-mapped, taps per block.
- hooks: `reference`, `hf_model` (bf16 whole model for sanity, fp32 first N for parity), `hf_layers` (taps),
  `state_sources`.

Decisions:
- Ground truth precision: the fp8 activation quantization of every fp8 / fp4 GEMM, the fp8 window KV, fp4 index q / k
  and fp4 compressed KV are part of the reference (owner rule: port the kernels exactly). Compute fp32.
- DS41-1: oracle and reference share one GEMM tiling, else the quantizers amplify last-bit GEMM differences (rel 1e-2
  at layer 2 before the fix). Result: the reference equals the oracle to the bit on 25 layers x 4096 tokens.
- DS41-3: single-pass mHC needs the pending pre in the boundary: in / x / out are [S, 4 * 5120 + 4]; h_mid [S, 4 * 5120].
- DS41-2: topk_idx [length, 512] and candidates [length, 2048] (block ids) are per position (the spec lists them under
  state.fixed; they grow): readers slice their chunk; check_reference needs equal shapes one-shot vs chunked. The
  device must report them the same way at the ladder's state check (or the plan changes the spec).
- window_kv is the inference ring: [128, 512], slot = position % 128.
- Top-k ties (indexer, candidate blocks) break toward the lowest position (torch.topk's tie order depends on row
  length); the attention's compressed width is min(512, end // ratio) as in the inference code.
- Engram: `engram_rows` is a host step with no inputs (token ids from ctx.extra["tokens"], tail in state engram_tail):
  hash + table rows [S, 24 * 256]; `engram_kv` (fp8 linear) and `engram_add` follow.

Gate (hand run, metrics in generated/bringup_adhoc/R.2.json, K-58): revision_ok 1; smoke "...Capital of France =
Paris...</think>Paris" ok; text_top1_acc 0.973 (2048 tokens, bf16, 40 layers); pcc_hidden_L00..L24 = 1.0 (max abs 0);
pcc_logits 1.0; top1_match_frac 1.0. Wall time 36.5 min (smoke decode ~13 min, 2048-token sanity ~7 min, parity
oracle + reference ~15 min).

Gotchas: transformers warns "model of type deepseek_v41 to instantiate a model of type ``" when AutoTokenizer reads
config.json: harmless. The smoke decodes 24 tokens one at a time through all 40 layers (each step dequantizes every
fp8 weight it touches): slow by design, no caching across layers (host RAM rule).

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.intake.check_hf_sanity && PYTHONPATH=$PWD python -m
models.demos.common.bringup.reference.check_hf --seq 4096`

R.3 pre-check (same code, not this step's gate; hand run tagged R.3, metrics in generated/bringup_adhoc/R.3.json):
`check_reference --seq 4096 --chunk 2048` first gave hidden pcc 0.984 / state min 0.0: the compressor's reset at
start 0 also cleared the engram tail layer 14's engram step had just written (DS41-4). Fixed (each step resets only
its own state); now hidden pcc 1.0, state min pcc 1.0, graph replay max abs 0 for all 8 block types. The one-shot
path (R.2 parity) is unaffected by that fix; the gate was re-run after it.
Next steps should know: the reference holds 25 layers' dense weights in fp32 (~18 GB) and reads experts per chunk
(8 prefetch threads); 4096 tokens through layers 0-24 take ~7 min on this host.

## SC.1 serving contract (attempt 1)

What: `bringup/serving_contract.md` (tt-d-gen @ a286b622 main; disagg_lb @ e001a53c, which pins tt-d-gen ba4c33e3
with chunk_aligned_start), `bringup/contract_tests.yaml`, tests in `tests/bringup/contract/`.
Decisions:
- Served caches (DS41-5): 0_window_kv every layer (per position, 32 x 512 per record), 1_compress_kv_r2 / 2_index_k_r2
  (layers 2, 8, 14; 16 rows per 32-token record), 3_compress_kv_r1 / 4_index_k_r1 (layer 20). bf16 ROW_MAJOR (owner:
  bf16 KV; a 16-row record cannot be a tile; bf16 is exact for the fp8 / fp4 values). ND shard [1, 1, 32/r, w]
  ROUND_ROBIN_1D (the KVM's one-bank record rule, noc_addr.cpp:12-18). Names sort in table order (DS41-6).
- Not served: topk / candidates (per chunk), compress_partial (never pending at a 32-aligned start or the 64-aligned
  decode boundary), engram tail (per-slot token history on the prefill host: a prefill node never copies a prefix;
  d2p off by default, owner question 5).
- Every layer acks (all own window_kv): layers_per_chunk = served layers. Host sink and D2H both required.
- Tests at chunk 2048 / max 4096 on the s4096 golden (every position dumped). Part test test_cache_starts gates
  `attention`; at C.<bt>.attention it runs that block type's representative + its sources (BRINGUP_TASK), by hand all
  spec layers. Runner tests serve the spec's contiguous prefix 0-3 (layers 14+ need golden inputs the runner cannot
  give). Runner smoke skips in this partial run (DS41-8, owner question 7).
- tt-d-gen has no engine build here (require_engine false): producer fallback. The framework's lookup does not expand
  $USER in serving.server_repo (DS41-7); the runner tests set BRINGUP_SERVER_REPO.
Ran (run_safe_pytest --no-precompile, 2x2 FABRIC_2D): test_cache_starts 4 x "not built: hooks.device_model";
test_kv_layout, test_adapter_acks, test_runner_contract "not built: adapter not registered" (12 s, no hang);
test_runner_smoke skipped (partial run). CPU checks: server_rules.self_check (reshuffle vs the server's ownership
rule), table_rules on synthetic good / misaligned / overlapping / wrong-size tables, record decode == kv_dump_compare.
Gate: `python -m models.demos.common.bringup.testing.serving` -> 5 tests, 0 problems.
Next steps should know: the device model needs the serving keywords (`embed/from_host(start=)`, `layer(end=)`,
`state.served/load_served/load_tokens`) from its first version; the adapter must be added to ADAPTER_PATHS at K.1.

## PL.1 plan (attempt 1)

What: `bringup/plan.yaml`, `plan.md`, `components.yaml` (136 entries, generated by a script kept in /tmp, no anchors),
`scaling.yaml`; known issues DS41-9, DS41-10; two model repo-map rows.
Decisions:
- Layout: SP rows (natural token blocks per row) x TP columns (heads / hidden), experts EP over all chips with the
  DeepSeek 2D dispatch (groups = columns), xing40's fp32 stream layout and mHC exchange, TtCSA-style sparse attention
  with a head->seq all_to_all whenever TP > 1, TtCsaIndexer-style indexer (heads replicated, queries split over TP).
- Experts: bfp8 is exact for the fp4 checkpoint (measured, CPU) but needs 26.9 GiB per chip for the run's 8 layers on
  2x2 and 33.6 GiB on 4x4 (DS41-10); bfp4 changes 16-17 % of the values. Plan: keep the packed e2m1 + e8m0 bytes
  (0.53 B / value) and decode in the expert op (fork extension of unified_routed_expert_ffn, with the routing weight +
  fp8 h quantization before the down projection); a composed per-chunk decode loop is the fallback.
- Deployment fit at 1M on 4x4 needed: one sparse table refilled per KV-source group per chunk (not a table per layer),
  window KV TP-dedup, activations 3.0 GiB, and bfp8 dense weights on galaxy_4x4 only (owner question 1; 3.5e-3 rel.
  error measured). Result 26.91 of 27.2 GiB; the run box 23.12 of 27.2 GiB.
- The model's fp8 / fp4 quantizers run on the device (tt/quant.py, composed bit ops), so device GEMM operands are
  bf16-exact; sums feeding quantizers use all_gather + add (K-30).
- Byte accounting: the routed experts' packed bytes are placed as uint8 (exact device bytes, K-60 without the flat
  extra); folded fp8 scales are counted as uint8 (skip is rejected for decoder layers).
Gotchas: the plan check rejects `skip` on any decoder-layer tensor, and the engram tables (host, owner rule) have no
other placement: 1 plan error remains (DS41-9). K-63: quote group names with commas. topk_idx / candidates history
for the ladder's state check is a run-only Setting (index_history), not in the deployment memory.
Gate (hand run, generated/bringup_adhoc/PL.1.json): plan_fits 1, unplaced_tensors 0, plan_errors 1 (DS41-9),
component_errors 0, ledger_errors 0, plan_approved 0 (needs the approval); deploy_fits 1, box_path_differences 0.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`

## SG.1 scale (attempt 1)

What: the reference reads the scaled checkpoint's counts. `reference/ds41_ref.py:inference_config` / `inference_dir`:
a dir without `inference/` but with `bringup_scale.json` takes the source's `inference/config.json` and applies every
text_config key the cut changed (HF_TO_INFERENCE; an unmapped change asserts) -> n_heads 32, o_groups 4,
n_routed_experts 96 (DS41-11). `Config.from_dir` and `hf_oracle.build_oracle` / `inference_module` use them.
scaling.yaml unchanged: scale.py reports 0 errors, 250 cut, 43200 dropped; layer 2 shapes checked (wq_b 16384 rows,
wo_a 4096 x 4096, wo_b 5120 x 4096, attn_sink 32, gate 96 rows, 576 expert tensors).
Checked on the CPU: layers 0-24 of the scaled checkpoint, 2 chunks of 512: finite, state shapes as expected (171 s).
Blocker (RESOLVED by the supervisor 2026-10-15: scaled.ladder 55296 / 2048; not in SG.1's paths): golden fails "the prompt has 56320 tokens < 65536" (DS41-12). input/prompt.json was
built at R.1 for the real ladder (56320); the scaled ladder (owner, 65536) needs spec `text.length: 65536` and a
prompt rebuild (prefix-identical ids, new sha; existing goldens record the old sha in their manifests), or a shorter
scaled.ladder. The 64k golden then runs ~2 h on the CPU (56k real took 5894 s).
Gotcha: hf_scaled/quietbox_2x2 is 237 GB (scale.py copies the engram tables whole).
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.scale && BRINGUP_SCALED=1 python -m models.demos.common.bringup.reference.generate_golden --rung scaled`
