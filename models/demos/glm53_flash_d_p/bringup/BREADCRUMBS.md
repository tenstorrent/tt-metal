# zai-org/GLM-5.3-Flash bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1 of the reference role; the previous run failed at the stock HF loader)

What was done:
- `reference/hf/`: the transformers glm5_next modeling + configuration files (commit 0291458166a6, 5.17 line),
  vendored with relative imports made absolute; five helpers 5.12.1 lacks come from `reference/hf/_compat.py`
  (kernel-hub / accelerate decorators as no-ops, `create_recurrent_attention_mask` = local 2D mask, vision stubs).
  The model code itself is unchanged.
- `reference/hf_oracle.py` (hook `hf_model`): text model only (`Glm5NextTextModel` + lm_head; no vision tower, no
  MTP), built on meta, weights assigned. Routed experts are `PackedExperts`: FP8 read from the safetensors (via the
  index) and dequantized per use, same forward as HF's experts. `num_layers=None` gives bf16 (intake sanity),
  otherwise fp32. fp32 kept for conv1d / dt_bias / A_log / e_score_correction_bias as `_keep_in_fp32_modules_strict`.
  Name mapping: `hc_{attn,ffn}_{fn,base,scale}` -> `{attn,ffn}_hc.*`, `{q,k,v}_conv1d` concatenated into `conv1d`,
  `f_a/f_b/dt_bias/A_log` -> `forget_gate.*`. `generate` is a greedy loop over `GlmCache` (see gotchas).
- `reference/glm_ref.py` (hook `reference`): standalone CPU reference, every block through `run_block`. Graph:
  attn_hc, attn_collapse, attn_norm, [KDA: attention] or [DSA: q_a, indexer, attention], attn_residual, ffn_hc,
  ffn_collapse, ffn_norm, [dense: mlp] or [MoE: router, experts, shared_expert, moe_add], ffn_residual (`out`).
  KDA runs HF's chunk-64 delta rule from the carried state (intra-chunk terms built 8 sub-chunks at a time). The
  indexer caches pooled keys and scores 128-query blocks against them. MLA uses absorbed kv_b and gathers the selected
  latent rows (topk [S, 2051] int32: 512 pools x 4 tokens + 3 tail, -1 = none). Experts are grouped per expert and
  padded to 32 rows (known issue on M-dependent sgemm).
- `reference/weights.py`: index-based loader, FP8 128x128 block dequant (fp32 product, exact).

Decisions:
- Residual-stream boundaries (`in`, `h_mid`, `out`) are `[S * 4, H]`, token-major (row 4t + n = stream n of token t):
  check_hf compares against HF's layer output flattened to `[-1, H]`, and `[S, 4, H]` fails its maxabs print.
- mHC coefficients are one boundary `[S, 24]` fp32 = pre 4 | post 4 | comb 16 (row-major, after Sinkhorn), shared
  by the collapse and residual steps.
- State: DSA `kv_latent [len, 512]`, `index_key [len // 4, 128]` (complete pools only); KDA `kda_recurrent
  [64, 128, 128]` fp32, `kda_conv [3, 24576]` (last 3 pre-conv q|k|v rows); a chunk at start 0 resets the KDA state.
  The indexer asserts chunk starts are multiples of 4. One-shot == chunked also needs starts that are multiples of 64
  (KDA sub-chunk grid); all ladder chunks are.
- Experts dequantized to fp32 at load when a subset has <= 2 MoE layers (58 GB for layers 3-4), else per use.

Gotchas:
- transformers 5.12.1 has linear-attention and indexed cache layers, but with other semantics than the 5.17 code
  expects (`update_conv_state` returns the stored state instead of [state | input]; `conv_states` is a tensor, the
  code indexes `[0]`). A 5.12 DynamicCache would silently corrupt KDA; `forward` therefore defaults to no cache and
  `generate` uses `GlmCache`.
- config.json names DSA layers `deepseek_sparse_attention`; the modeling code's mask table only has
  `indexed_attention` / `linear_attention` (KeyError). The oracle renames layer_types in the config object.
- The whole-model sanity streams ~300 GB of FP8 experts from disk; it took 10 min on this host (page cache warm).

Results (local runs): check_hf --seq 4096: L00-L04 pcc 1.0000000 (maxabs <= 5.7e-6), logits pcc 1.0000000, top-1
match 1.0000. check_hf_sanity: revision ok, smoke '</think>Paris<|user|>' ok, text_top1_acc 0.964.
Chunked (64) vs one-shot at 128 tokens: max abs 0.

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.intake.check_hf_sanity && PYTHONPATH=$PWD python -m
models.demos.common.bringup.reference.check_hf --seq 4096` (about 14 min).
Also run locally (R.3's check, not this gate): check_reference --seq 4096 --chunk 2048: hidden pcc 1.000000000,
state min pcc 1.000000000, graph replay max abs 0 for kda_dense (L0, 10 steps), dsa_moe (L3, 15), kda_moe (L4, 13);
4 min. Gate run: 13 min 23 s, all R.2 metrics pass.

## PL.1 plan (attempt 1)

What was done:
- `plan.yaml`: placements for every checkpoint tensor (76108, whole-model map from `bringup_trim.json`): vision, MTP
  layer 45 and layers 5-44 skipped; fp8 block scales folded at load. State: MLA latent [56320, 512] and pooled keys
  (seq_stride 4) replicated for layer 3; KDA recurrent (32 heads x 128 x 128 fp32) and conv tail per chip for 0-2, 4.
  Activations 7.0 GiB (8192-token chunk, 4-stream residual replicated). Gate: 13.59 of 27.20 GiB per chip.
- `components.yaml`: all 41 steps (kda_dense 10, dsa_moe 15, kda_moe 13, model 3), NATIVE or COMPOSED, none CPU or
  OPGEN.
- `plan.md`: per-layer tables, per-chip total, collectives, departures. `tasks.yaml` unchanged.

Decisions:
- KDA is TP=2 (axis 1) x SP=2 (axis 0), because ttKDA requires distinct SP / TP axes. Fallback: SP=1 on 1x2 row
  submeshes.
- DSA attention: replicated weights, query rows split 4 ways, all 64 heads per chip (sparse_sdpa needs H % 32 == 0).
  Caches replicated, every chip computes latent and pooled keys for all S rows, one all_gather after o_proj.
- Indexer is composed, not OPGEN: key pooling is reshape + 4 slices + an elementwise softmax. indexer_score_dsa runs
  unmasked (chunk_start_idx = kv_len - 1) plus a constant pool-causal mask on the chunk's own pools. Sentinel
  compaction: for q >= 2047 the pool part has no sentinels, so [pools | tail | pad] ends in sentinels; for q < 2047
  the row is the constant "all tokens 0..q".
- Experts EP=4 via the MiMo 2x2 dispatch (axis 0, groups = columns, 72 per chip), high_precision unified kernel with
  ClampedSiluGlu. Dense MLP and shared expert TP=4 + all_reduce(cluster_axis=None). Router fp32 (MiMo TtRouter).
- mHC: DeepSeek mhc_split_sinkhorn matches GLM's Sinkhorn order and pre/post forms (checked against
  reference/mhc/mhc_reference.py); RMS eps 1e-5 is a config field.

Gotchas / to verify at implement: ttnn.topk at width 288; unified_routed_expert_moe and the dispatch tables with 72
local / 288 global experts; ttKDA's SP segment order under a chunk-aligned actual_start; sparse_sdpa scale must be
passed explicitly as 1/16 (its default is 512^-0.5); KDA recurrence and o_proj default to HiFi2 in the Kimi config
(set HiFi4); core grids from the device (p300c 11x10). The gate builds the CPU reference for layers 0, 3 and 4 (14 s
here, page cache warm).

Result: plan_fits 1, unplaced_tensors 0, plan_errors 0, component_errors 0, ledger_errors 0; plan_approved 0 (waits
for the person's approval).

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`.

## C.kda_dense.attn_hc test (attempt 1)

Reviewed the rendered component test for attn_hc (layer 0, s4096 chunk 1). Kept the gated PCC and added per-part checks,
because whole-output PCC passes several real bugs (measured on the golden with a host script: comb transposed 0.9958,
softmax over the wrong axis 0.9987, 10 Sinkhorn iterations 0.9958, swapped pre/post scales 0.9958, a zeroed row 0.9997).
- Per part (pre / post / comb): rel L2 <= 0.03 / 0.03 / 0.02 and max abs <= 0.15 / 0.2 / 0.1.
- Comb column sums within 0.01 of 1; pre in (0, 1], post in [0, 2], comb >= 0.
- Headroom: bf16, bfp8 or tf32 projection operands give rel <= 0.002. A pessimistic random mix error at PCC 0.9989
  (the TT fp32-matmul ceiling from deepseek_v3_d_p test_mhc.py) gives pre rel 0.025, comb max abs 0.061.
- Not caught (small effect): 19 iterations, hc_eps 0, pre without +1e-6.
- The golden attn_hc is bf16-rounded values stored as fp32. Comb row sums are only 0.92..1.08 even after 20 iterations,
  so don't assert row sums.
- Implement: the output must be [2048, 24] (or reshapeable), comb row-major (comb[i, j] at column 8 + 4i + j).
Results: reference passes (PCC 0.999999, part rel ~0.0013, column sums 1.00000). Stub fails (PCC 0). The gate in
device mode fails with NotImplementedError until the implement step adds the module.
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_hc.py`

## C.kda_dense.attn_hc implement (attempt 1)

What was done:
- `tt/mhc.py:TtHcWeights` (+ `build_hc(mesh, loader, cfg, layer, "attn"|"ffn")`): replicated, no CCL. Input
  `[1, 1, S, 4H]` (streams packed along the last dim, the same memory as the reference's token-major `[S * 4, H]`),
  bf16 -> `ttnn.typecast` fp32 -> DeepSeek `tt_mhc._project` (fp32 matmul HiFi4 + fp32 acc, RMS rsqrt applied after
  the linear, eps 1e-5) -> `mhc_split_sinkhorn` (consts from `tt_mhc.build_consts`, n 4, 20 iters, eps 1e-6) ->
  `ttnn.concat` [pre | post | comb] -> `[1, 1, S, 24]` fp32. Weights and consts uploaded once at load.
- `tt/common.py`: `hifi4_config`, `replicate` / `replicated_to_host` (load time and harness boundary only).
- `hooks.py`: `device_component` handles `attn_hc` and `ffn_hc` (same module, other weights); `DEVICE_STEPS`
  (kda_dense: attn_hc) and a `HybridDeviceModel` (CPU reference + DEVICE_STEPS, host in / out per step) as
  `device_model` until assemble.

Decisions: the kernel's comb output is already row-major (entry (i, j) at column 4i + j), as DeepSeek's op test
reshapes it `[T, n, n]` against the row-major reference; no reorder needed. The harness uploads the residual as bf16
(the planned device residual dtype).

Result: pcc_attn_hc_L00 0.999999; rel L2 pre 0.0014 / post 0.0013 / comb 0.0016; max abs <= 4.9e-3; comb column
sums [0.9974, 1.0009].

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_hc.py`

## S.kda_dense.01 test (attempt 1)

Reviewed the rendered swap test (attn_hc on device, layer 0). The gated pcc_swap_out is kept. Asserted extra checks added,
because a CPU sensitivity run (attn_hc output perturbed, rest CPU) showed that block-out PCC passes real bugs: comb
transposed 0.9985, comb rows normalized instead of columns 0.9985, post x1.05 0.9991, pre x1.05 0.99986, last row zeroed
0.99993.
- Block out: rel L2 <= 0.01 (those bugs give 0.070 / 0.070 / 0.045 / 0.017 / 0.012) and per-row norm ratio in [0.97, 1.03].
- attn_hc's own output vs golden (its input is the golden `in`): the component test's per-part rel L2 and max abs, comb
  column sums, and pre/post ranges.
Results: reference passes (out PCC 0.999999, rel 0.0017). Stub fails (PCC 0). Device passes (out PCC 0.999998,
rel 0.0023, row ratio [0.9966, 1.0029]; attn_hc part rel <= 0.0016).
Gotcha: the run_safe_pytest precompile pass prints the whole metric set once with stub values (rel 1.0) before the real
pass. Ignore that block.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_01_attn_hc.py`
(prefix with `BRINGUP_IMPL=reference` or `BRINGUP_IMPL=stub` to check the other two modes).

## C.kda_dense.attn_collapse test (attempt 1)

Reviewed the rendered component test for attn_collapse (layer 0, s4096 chunk 1). Kept the gated PCC and added checks,
measured on the golden with a CPU-only host script:
- At layer 0 the four streams of `in` are identical (embedding copied), so pre-column or stream order is invisible,
  and PCC passes a dropped last stream (0.9998), output x1.02 (0.999998), and 32 zeroed rows (0.9934).
- Added: rel L2 <= 0.01 and per-token norm ratio in [0.985, 1.015] (those bugs: rel 0.021 / 0.020 / 0.115).
- Added: the same module (the step has no weights) on layer 1's golden `in` / `attn_hc` / `attn_in`, where the
  streams differ; pre reversed gives rel 1.27, pre 0/1 swapped 0.29, stream-major rows 1.28.
- Headroom: bf16 products accumulated in bf16 give rel 0.0036, ratio [0.994, 1.004].
Results: reference passes (PCC 0.999998; rel L2 0.0021 at L00, 0.0026 at L01). Stub fails (PCC 0). Device mode fails
with NotImplementedError until the implement step adds the module.
Implement: the device_component for attn_collapse is called with layer 0's ctx and fed layer 1's inputs too; the
input `in` is token-major [S * 4, H] (row 4s + n), the output is [S, H] (or reshapeable).
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_collapse.py`

## C.kda_dense.attn_collapse implement (attempt 1)

What was done:
- `tt/collapse.py:TtHcCollapse` (+ `build_collapse(cfg)`): replicated, no CCL, no weights. `x [1, 1, S, 4H]` (streams
  packed along the last dim = token-major `[S * 4, H]`), `hc [1, 1, S, K]` fp32 (pre = columns 0..3) -> DeepSeek
  `tt_mhc._streams` / `_cols` -> each stream `ttnn.typecast` to fp32 -> `_mix` (multiply + 3 x addcmul, column
  broadcast) in fp32 -> one `ttnn.typecast` to bf16 -> `[1, 1, S, H]`.
- `hooks.py`: `_collapse_host_fn` (harness boundary: x uploaded bf16, hc fp32, chip 0 read back) for `attn_collapse`
  and `ffn_collapse` (same module); `DEVICE_STEPS["kda_dense"]` now has `attn_collapse`.
Decisions: fp32 accumulate with a single bf16 round, matching `glm_ref.hc_collapse` (the component notes).
Result: pcc_attn_collapse_L00 0.999997; L00 rel L2 0.00238, ratio [0.9964, 1.0027]; L01 (distinct streams) rel L2
0.00298, ratio [0.9960, 1.0036].
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_collapse.py`
