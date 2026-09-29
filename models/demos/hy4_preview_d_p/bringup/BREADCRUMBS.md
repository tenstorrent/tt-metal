# tencent/Hy4-preview bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1)

What was done
- `reference/hf_hy_v4/`: transformers 5.17.0's `modeling_hy_v4.py` / `configuration_hy_v4.py`, vendored because
  python_env has 5.12.1. The imports were rewritten from `from ...x` to `from transformers.x`; 5.12.1 has all of them,
  `DynamicIndexedLayer` included, so `generate` works with the indexer cache. One semantic fix, marked "bring-up fix": RoPE (below).
- `reference/hf_oracle.py` (`hooks.hf_model`): builds HYV4ForCausalLM on meta with eager attention and eager experts, applies the
  conversion_mapping renames (rename-only), keeps HF's `_keep_in_fp32_modules_strict` modules in fp32 and never reads MTP.
  Routed experts run HF's own `HYV4Experts.forward`, but `gate_up_proj` / `down_proj` are `ExpertSlab` views that read
  expert e's byte range. A layer's hit experts are prefetched with 16 parallel 8 MB preads. Mask shim: 5.12 has no
  `allow_is_causal_skip`. Precision: bf16 for the full model (sanity), fp32 for a layer prefix (check_hf).
- `reference/weights.py`: `WeightLoader` (every tensor through model.safetensors.index.json, never a shard name, so R.4
  trimming is safe) and `ExpertSlab`.
- `reference/hy4_ref.py` (`hooks.reference`): standalone chunked sparse CPU reference. It owns explicit state: kv_latent
  [max_seq, 576] (kv_a_layernorm(latent) | RoPE'd k_rope) and index_key [max_seq, 128] (full layers; [0, 128] on shared
  layers). Attention runs absorbed (q_nope through W_uk to the 512-wide latent, scores over the 576-wide key) over the
  2048 gathered keys of each query, with the sink as an extra logit, the elementwise sigmoid output gate and o_proj.
  Indexer: ReLU scores weighted per head, top-2048 per row. A row that sees at most 2048 keys keeps all of them. Indices
  are sorted ascending, -1 padded.
  MoE: grouped per expert, padded to 32 rows. iHC and the router run in fp32, the LM head in fp32.
- `hooks.hf_layers`: identity taps, so check_hf sees HF's layer output as [S, 4H] like the reference.

Block graph (every block goes through run_block; boundaries are recorded as `L{i}.<output>`)
- attn_hc -> attn_hc_pre -> attn_norm -> q_a -> indexer (full) | topk_shared (shared) -> attention -> attn_residual ->
  ffn_hc -> ffn_hc_pre -> ffn_norm -> mlp (dense), or router -> experts -> shared_expert -> moe_combine (MoE) -> ffn_residual.
- Block in/out: the 4 streams flat, [S, 4H] (stream j = columns [jH, (j+1)H), HF's flatten(2) order). attn_hc / ffn_hc
  = [S, 8] fp32 (pre 4 | post 4). topk: int64 [S, 2048]. router: dense [S, 256] fp32. Extra model-level record:
  `hc_head` [S, H] before `final_norm`.
- Shared layers (2-4) get the top-k of the latest full layer's current chunk from the reference (`Hy4Reference._topk`,
  keyed (layer, start, length)). A component or swap test that runs a shared block without its source layer must set
  `ctx.extra["shared_topk"]` (the golden's `L{src}.topk`, src = cfg.topk_source(layer); for layers 2-4 that is L1).

Decisions and why
- RoPE is interleaved (GPT-J pairs) in the MLA and in the indexer's last 64 dims, not rotate-half as in the 5.17 port.
  Evidence: with the port as released, the full-model sanity scored accuracy 0.237 (smoke still "Paris") while
  reference == HF at PCC 1.0. SGLang's Hy4 config sets rope_interleave = indexer_rope_interleave = True
  (is_neox_style=False). With interleaved RoPE: accuracy 0.965, parity unchanged. See findings.yaml R2-hf-rope-interleaved and known issues.
  For the device: no host permutation of the rope columns for Meta-order kernels (this reverses the note in
  hy4_preview_findings.md).
- q_a_layernorm / kv_a_layernorm eps 1e-6 as in HF (HYV4RMSNorm default); SGLang uses 1e-5; negligible.
- The routed experts of layers 0-5 are held in fp32 in memory (5 x 38.6 GB, about 200 GB). More than 8 MoE layers switch
  to ExpertSlab reads.
- Query-row blocks of 64 are aligned to absolute positions, so chunked prefill reproduces one-shot exactly.

Results (hand runs, generated/bringup_adhoc)
- check_hf_sanity: revision ok, smoke "Paris" ok, text_top1_acc 0.965 (17.7 min, disk-bound expert reads).
- check_hf --seq 4096 (6 layers, fp32): pcc_hidden_L00..L05 1.0000000 (maxabs <= 6.2e-5), pcc_logits 1.0000000,
  top1_match 1.0. The whole gate command took 23.7 min.
- check_reference --seq 4096 --chunk 2048 (R.3's gate, run early): hidden and state PCC 1.000000000; graph replay max
  abs diff 0 for all 3 block types (6.5 min).

Gotchas
- check_hf's maxabs needs the recorded `L{i}.out` in HF's reshape(-1, H) layout, hence the flat [S, 4H] block
  boundary plus the taps.
- The full-model sanity reads about 1.5 TB of experts per 2048-token pass; do not run it alongside another heavy host job.

Re-run
    PYTHONPATH=$PWD python -m models.demos.common.bringup.intake.check_hf_sanity && \
    PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_hf --seq 4096

## PL.1 plan (attempt 1)

What was done
- `plan.yaml`: placements for every checkpoint tensor (2006 in the trim map). Layers 6-77 and `model.mtp_layers.*` are
  skipped. There are two state entries (the MLA latent cache on layers 0-5, the index-key cache on 0, 1 and 5) and
  extras (KV gather scratch and buffers 0.35 GiB, contract KV reserve 0.10 GiB). Activations are estimated at
  5.0 GiB for an 8192-token chunk.
- `plan.md`: scheme, one table per block type, model level, per-chip total, collectives, activation estimate,
  departures, open items.
- `components.yaml`: 45 entries (dense_full 12 steps, moe_full 15, moe_shared 15, plus embed, final_norm and
  lm_head). No CPU and no OPGEN steps.
- Gate (hand run): per chip 20.60 of 27.20 GiB, plan_errors 0, unplaced 0, component_errors 0, ledger_errors 0,
  plan_approved 0 (the overseer approves). tasks.yaml is unchanged: the orchestrator always allows `ttnn/ttnn/bringup`,
  so the implement tasks need no extra paths.

Decisions and why
- **Layout: SP=2 over rows (axis 0) x TP=2 over columns (axis 1)**, with the residual as `[S/2, 4 x 3072]` fp32 per
  chip. Why: the MLA cache is one 576-wide latent that cannot be split by head, `sparse_sdpa` needs 32 heads per chip
  (64 / 2), and ttMLA / TtIndexer / tests/sparse_mla are built for SP x TP on 2x2 FABRIC_2D. MiMo 2x2's flat TP=4
  would give 16 heads per chip and need the head->sequence all_to_all.
- iHC: fn is split by column; the only collective is one `[S/2, 32]` fp32 all_reduce over axis 1 per gate.
  mhc_split_sinkhorn is not used (Hy4 has no comb matrix).
- attn_norm is a distributed RMSNorm (the output stays K-split for q_a / kv_a / gate / indexer). ffn_norm
  all_gathers ffn_x over axis 1 once, because the dense MLP, the router, dispatch and the shared expert all need the
  full hidden.
- Experts are EP=4 bfp8 with the DeepSeek 2D dispatch along axis 0, as in mimo_v2_6_d_p_2x2. The group sum is a
  reduce_scatter over axis 1 (the residual is column-split), not MiMo's all_reduce + all_gather.
- bf16 attention and indexer weights (not ttMLA's bfp8), a bf16 index-key cache (not bfp8), fp32 router and iHC.
  Every matmul and sparse_sdpa runs at HiFi4 + fp32 acc.
- Sink: pass sink x 16 with an explicit scale of 1/16 to sparse_sdpa.

Gotchas for the component steps
- TtIndexer ropes dims 0-63 but Hy4 ropes 64-127: permute wq_b / wk rows and k_norm to [64..127 | 0..63] on the host.
  k_norm eps is 1e-5 (TtIndexer hard-codes 1e-6). q_a / kv_a norm eps is 1e-6 (ttMLA passes rms_norm_eps). See the
  known-issues proposal.
- No host RoPE permutation in the MLA (interleaved, R.2).
- The device top-k is unsorted with a 0xFFFFFFFF tail; the golden is ascending and -1 padded.
- ttMLA's persistent CCL buffers are sized for one local chunk length; the ladder uses 1024, 4096 and 2560 rows per
  chip.
- In plan.yaml flow mappings, quote any value that contains a comma (known-issues proposal).

Re-run
    PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan

## C.dense_full.attn_hc test (attempt 1)

What was done
- Reviewed the rendered component test for attn_hc (iHC gates [S, 8] fp32, pre 4 | post 4, layer 0, s4096 chunk 1).
  The golden is bf16. Kept the gated PCC and added checks against the golden: output finite, element count, rel L2
  <= 0.01, and max abs error per column <= 0.015. Also asserts that device_component is not a CPU bridge.
- Measured the mutations on the CPU (the table is in the test docstring). The fp32 reference scores PCC 1.000000,
  rel 0.00052, max abs 0.0019 (bf16 rounding of the golden). post x1, a missing TP partial, RMS over one stream,
  swapped pre/post fn or base rows: all pass PCC 0.99, and all fail rel L2 or per-column max abs.

Gotchas
- At layer 0 the four streams are identical and fn's four stream blocks are nearly equal, so a stream-order bug in
  the fn permutation cannot be seen at layer 0, even with synthetic distinct streams (<= 1e-4). The chip-major vs
  stream-major column order is caught (rel 0.11).
- The device module must return 8 columns (reshapeable to [2048, 8]). A [S, 32] padded tile row fails the element
  count check. Slice to 8 on the device before the read-back, or return the padded tensor through the hook's own
  read-back.
- Allowance: about 0.005 of sigmoid / rsqrt abs error on post, which 2 * sigmoid doubles. Use an accurate sigmoid.

Results
- BRINGUP_IMPL=reference: PASS (pcc 1.000000, rel 0.000518, max abs per column <= 0.00195).
- BRINGUP_IMPL=stub: FAIL (pcc 0.0).
- Gate (device): FAIL with NotImplementedError. No device module exists yet; that is the implement step.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_hc.py

## C.dense_full.attn_hc implement (attempt 1)

What was done
- `tt/layout.py`: the 2x2 stream layout (chip (r, c): rows r-half, columns [3072c, 3072(c+1)) of each of the 4 streams,
  packed stream-major, [1, 1, S/2, 12288] fp32). `streams_cols_to_chip_major` reorders HF (j, h) columns to (c, j, k)
  so ShardTensor2dMesh dims=(2, 3) hands each chip its block. Host helpers are the harness boundary only.
- `tt/ihc.py:TtHcGates`: matmul of the streams with this chip's fn^T [12288, 32] (fn permuted with the same helper,
  padded 8 -> 32, sharded over axis 1, replicated over axis 0), plus multiply + sum for the partial sum of squares,
  packed into column 8 via a [S/2, 1] x [1, 32] one-hot broadcast multiply. One `ttnn.all_reduce(cluster_axis=1)` of
  [S/2, 32] fp32. Then slice column 8 -> rsqrt(ss / 24576 + 1e-5) -> multiply, scale / base / magnitude / eps as
  [1, 32] fp32 row constants built at load, `ttnn.sigmoid` (default Accurate mode), slice to [S/2, 8] on the device.
  HiFi4 + fp32 dest for the matmul and the sum. No host work in the forward.
- `bringup/hooks.py`: `device_component` for `attn_hc` (`_hc_module` + `_hc_host_fn`: streams host -> device,
  gates read back from column 0's copy); `DEVICE_STEPS = {"dense_full": {"attn_hc"}}`; `device_model` returns a
  `HybridDeviceModel` (CPU reference, DEVICE_STEPS swapped in, embed repeats to 4 streams, final_norm = hc_head +
  RMSNorm) until the assemble step.

Decisions
- Kept the 32-wide tile row through the elementwise chain and sliced to 8 once at the end (fewer unaligned ops).
- `_HC_STEPS` maps only attn_hc -> hc_attn_layer; ffn_hc is the same module with `hc_mlp_layer` (add the entry in its
  own task).

Results
- Gate: PASS. pcc_attn_hc_L00 1.000000, rel L2 0.000539 (CPU reference 0.00052), max abs per column <= 0.0023
  (limit 0.015).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_hc.py

## S.dense_full.01 test (attempt 1)

What was done
- Reviewed the rendered swap test (dense_full layer 0, attn_hc on device, rest CPU). Kept the gated pcc_swap_out
  (0.98) and added asserted checks (informational metrics): the gates vs golden (8 columns, finite, rel L2 <= 0.01,
  per-column max abs <= 0.015, as the component test), attn_x rel L2 <= 0.01, h_mid rel L2 <= 0.01 and worst row
  <= 0.05, block out finite and rel L2 <= 0.01. Also asserts the device module is not a CPU bridge.
- Measured gate mutations through the block on the CPU (table in the test docstring). fn / base post-row swaps,
  post x 1.02 and a zeroed gate row pass the 0.98 out gate; pre-gate bugs change nothing downstream at layer 0
  (identical streams + RMSNorm), so only the gates and attn_x checks see them.

Gotchas
- The reference re-running the indexer on the golden input matches the golden topk at 0.79 (exact-position match),
  yet attn_out PCC is 0.999999: near-tie key choices at layer 0. Not gated here; relevant for the indexer test.
- Stream-order bugs (fn stream blocks permuted, RMS over one stream) stay invisible at layer 0.

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999999, rel 0.00167). BRINGUP_IMPL=stub: FAIL (out PCC 0.524, every check).
- Gate (device): PASS. pcc_swap_out 0.999998; gates rel 0.000539, col max 0.0023; attn_x 0.00103; h_mid 0.00174 /
  worst row 0.0034; out rel 0.00177.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_01_attn_hc.py

## C.dense_full.attn_hc_pre test (attempt 1)

What was done
- Reviewed the rendered component test (attn_x = sum_j pre_j x stream_j, layer 0, s4096 chunk 1, golden bf16).
  Kept the gated pcc_attn_hc_pre_L00 (0.99) and added asserted checks vs the golden: finite, element count, rel L2
  <= 0.004, per-row norm ratio in [0.995, 1.005], worst row rel L2 <= 0.01. Metrics rel_l2_*, worst_row_rel_l2_*,
  syn_rel_l2_* recorded (informational).
- Added a second call of the module on synthetic distinct streams (golden stream rows rolled by 7 j per stream, pre
  gates rotated by row mod 4), compared with the CPU hc_pre on the same inputs: rel L2 <= 0.008, worst row <= 0.02.
- Mutation table in the test docstring (CPU, on the golden).

Decisions and why
- Layer-0 streams are identical, so the golden check only sees sum_j pre_j per row; PCC passes no-gating, row-shift,
  zeroed row, x1.02. The synthetic call is the only layer-0 check that sees stream / gate order.
- Limits sit between bf16 accumulation (0.0013 rel, ratio [0.998, 1.004], row 0.005) and the smallest bug
  (pre x 1.01: 0.0099, ratio 1.008).

Gotchas
- The implement step's module must accept arbitrary [S, 4H] streams and [S, 8] gates of the golden shape (it is
  called twice, the second time on the synthetic inputs).
- Device gate currently fails with NotImplementedError (no device module for attn_hc_pre in hooks._HC_STEPS yet).

Results
- BRINGUP_IMPL=reference: PASS (pcc 1.000000, rel 0.000947, ratio [0.99805, 1.00167], row 0.00258; synthetic exact).
- BRINGUP_IMPL=stub: FAIL (pcc 0.0).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_hc_pre.py

## C.dense_full.attn_hc_pre implement (attempt 1)

What was done
- `tt/ihc.py:TtHcPre`: per chip, 4 `ttnn.slice` of the streams (local column blocks j*3072, stream-major) and 4
  `ttnn.slice` of gate columns 0-3 ([S/2, 1]), then `ttnn.multiply` + 3 x `ttnn.addcmul` (column broadcast), fp32
  throughout (deepseek_v3_d_p/tt/mhc/tt_mhc.py:_streams / _cols / _mix). Output [1, 1, S/2, 3072] fp32 per chip =
  this chip's hidden columns (rows split over axis 0, columns over axis 1). No collective, no weights, no host work.
  Optional `dtype=` typecasts at the end (default fp32; the device keeps fp32).
- `tt/layout.py`: `row_split_to_device` (host [S, W] -> rows over axis 0, replicated over axis 1) and
  `col_split_to_host` (rows over axis 0, columns over axis 1 -> host [S, W]). Harness boundary only.
- `bringup/hooks.py`: `_HC_PRE_STEPS = {"attn_hc_pre"}`, `_hc_pre_host_fn` (streams + gates host -> device ->
  attn_x host), `_device_step_fn` dispatches both iHC step kinds; `device_component` and `HybridDeviceModel` use it
  (the hybrid asserts every DEVICE_STEPS entry has a module). `DEVICE_STEPS["dense_full"] = {"attn_hc", "attn_hc_pre"}`.

Decisions
- Kept fp32 output (no bf16 cast) since the device residual is fp32; the test compares against fp32/bf16 golden fine.
- ffn_hc_pre is the same module (inputs h_mid, ffn_hc); add "ffn_hc_pre" to `_HC_PRE_STEPS` in its own task.

Results
- Gate: PASS. pcc_attn_hc_pre_L00 1.000000, rel L2 0.000947, row ratio [0.99805, 1.00167], worst row 0.00258
  (same as the fp32 CPU reference); synthetic distinct streams rel L2 0.000000.
- The "FAIL pcc ... 0.000000" line at the top of the log is the precompile collect pass (stubbed), not the real run.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_hc_pre.py

## S.dense_full.02 test (attempt 1)

What was done
- Reviewed the rendered swap test (dense_full layer 0, attn_hc + attn_hc_pre on device, rest CPU). Kept the gated
  pcc_swap_out (0.98); rewrote it on the swap-01 pattern with asserted checks (informational metrics): attn_hc gates
  (as swap 01), attn_x vs golden at the component limits (rel L2 <= 0.004, row norm ratio [0.995, 1.005], worst row
  <= 0.01), attn_x vs the CPU hc_pre on the same inputs (block input + device gates; rel <= 0.004, row <= 0.01), the
  attn_hc_pre module once more on synthetic distinct streams vs CPU (rel <= 0.008, row <= 0.02, as the component
  test), h_mid (rel <= 0.01, row <= 0.05), block out (rel <= 0.01). Asserts neither module is a CPU bridge.
- Mutation table (CPU, attn_hc_pre replaced by mutations, 8 s per run) in the test docstring.

Gotchas
- Every attn_hc_pre bug except a zeroed row, a post/pre mix-up and SP/TP layout swaps leaves attn_norm, h_mid and out
  unchanged at layer 0 (attn_norm removes the per-row scale); pre x 1.02, no gating, rows shifted, gate 3 dropped
  all score out PCC 0.999998+. Only the attn_x checks see them.
- With the zero stub, the "vs CPU on the same inputs" and synthetic checks compare 0 with 0 (gates are zero); the
  golden checks fail it instead.

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999999; attn_x rel 0.00103, ratio [0.99806, 1.00161]).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98, every check).
- Gate (device): PASS. pcc_swap_out 0.999998; gates rel 0.000539 / col max 0.0023; attn_x rel 0.00103, ratio
  [0.99805, 1.00159], worst row 0.0024; vs CPU same inputs 0.0; synthetic 0.0; h_mid 0.00174 / 0.0034; out rel 0.00177.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_02_attn_hc_pre.py

## C.dense_full.attn_norm test (attempt 1)

What was done
- Reviewed the rendered component test (attn_norm = input_layernorm, w * x * rsqrt(mean(x^2) + 1e-5), plain w).
  Kept the gated pcc_attn_norm_L00 (0.99) and added asserted checks vs the golden: output finite, element count,
  rel L2 <= 0.008, row norm ratio in [0.993, 1.007], worst row rel L2 <= 0.015. Added a second run on the golden
  input x 0.1 (bf16) vs the CPU step on the same input (rel <= 0.01, worst row <= 0.02) to catch a wrong eps.
  Asserts the device module is not a CPU bridge. Mutation tables (CPU) in the test docstring.

Gotchas
- Golden: w in [0.020, 0.225]; row rms of attn_x >= 0.0267, so eps 1e-6 vs 1e-5 is invisible there (rel 0.0031,
  like bf16 device noise 0.0032). The x 0.1 run makes it 0.158. q_a / kv_a norms use 1e-6, so a shared norm
  builder with the wrong eps is a plausible bug.
- Passing on PCC but caught by the extra checks: x 1.01, RMS over half / a quarter of the columns (missing
  all-reduce on a column-split input), LayerNorm instead of RMS, sum instead of mean, a zeroed last row.
- The pessimistic bf16 device estimate (bf16 input, rsqrt, product, output) is rel 0.0032 / ratio [0.996, 1.0035] /
  worst row 0.005, so the golden limits have about 2.5x margin.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999999, rel 0.00171, ratio [0.99985, 1.00013], worst row 0.0030; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Device (gate command): fails with "no device module for attn_norm yet", as expected before the implement step.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_norm.py

## C.dense_full.attn_norm implement (attempt 1)

What was done
- `tt/norm.py:TtDistributedRmsNorm`: input_layernorm on the column-split [1, 1, S/2, 3072] per-chip input (TtHcPre's
  fp32 output). `rms_norm_pre_all_gather` (fp32 stats) -> multiply by a [1, 32] one-hot column-0 mask ->
  `ttnn.all_gather(dim=3, cluster_axis=1, Linear)` -> `rms_norm_post_all_gather` (eps = cfg.rms_norm_eps 1e-5, w
  fp32 row-major [1, 1, 192, 32] per chip, split by mesh column). HiFi4 + fp32 dest on both ops. Output bf16,
  column-split [1, 1, S/2, 3072] (what q_a / kv_a / gate / indexer K-split matmuls take). No host work in __call__.
- `tt/layout.py:col_split_to_device` (harness boundary). hooks.py: `_NORM_STEPS`, `_norm_module`,
  `_col_split_host_fn`; `device_component` handles attn_norm; `attn_norm` added to DEVICE_STEPS["dense_full"].

Decisions and why
- Stats mask: with an fp32 input, rms_norm_pre_all_gather leaves junk (|v| up to 16.7) in stats columns 1-31
  (column 0 is exact); the post op row-reduces the whole tile, so the output was 1.6% low (rel 0.016, ratio
  [0.979, 0.989]). Masking fixes it with one tiny elementwise op on [S/2, 32]; no op edit or fork needed. (Typecasting
  the input to bf16 also works, rel 0.0027, but loses precision.) Proposed in known_issues.md.
- fp32 output of post_all_gather at width 3072 failed with a dataflow-buffer allocation error; bf16 output is kept.

Results
- Gate: PASS. pcc_attn_norm_L00 0.999999, rel L2 0.00170, row norm ratio [0.99905, 1.00117], worst row 0.0038;
  scaled x0.1: rel 0.00176, worst row 0.0020.
- The "FAIL pcc ... 0.000000" line at the top of the log is the precompile collect pass, not the real run.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_norm.py

## S.dense_full.03 test (attempt 1)

What was done
- Reviewed the rendered swap test (attn_hc, attn_hc_pre, attn_norm on device). Kept the gated pcc_swap_out (0.98)
  and added asserted checks, sized from a CPU mutation study (attn_norm replaced by mutations of the fp32 reference,
  whole block run; table in the test docstring):
  - attn_hc and attn_x vs golden at the swap 01 / 02 limits;
  - attn_norm vs golden at the component limits (rel 0.008, ratio [0.993, 1.007], worst row 0.015), vs the CPU
    attn_norm on the device attn_x (same limits), and the module again on attn_x x 0.1 vs CPU (rel 0.01, worst row
    0.02; the eps check);
  - q_resid rel <= 0.01; indexer top-k set overlap vs golden >= 0.995; attn_out rel <= 0.01 / worst row <= 0.05;
    h_mid rel <= 0.01 / worst row <= 0.05; block out rel <= 0.01.

Gotchas
- q_a_layernorm removes any per-row scale of attn_norm from q_resid (x 1.02 leaves q_resid at 0.0017), and the
  residual dominates out: attn_norm x 1.02 gives out rel 0.0128, x 1.01 only 0.0066; RMS over half the columns 0.0040,
  eps 1e-6 0.0019. All of these pass the 0.98 out gate (w halves swapped too, 0.9867); the attn_norm checks catch them.
- The trail's `pcc_swap_topk: match=0.79` is positional equality, not a bug; the set overlap is the real check
  (proposed in known_issues.md).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999999, attn_norm rel 0.00166, topk overlap 0.99926, out rel 0.00167).
- BRINGUP_IMPL=stub: FAIL (out PCC 0.524 and every extra check).
- Gate (device): PASS. pcc_swap_out 0.999998; attn_norm rel 0.00214 / ratio [0.99771, 1.00059] / worst row 0.0036;
  vs CPU same input 0.00195; x0.1 0.00176; q_resid 0.00182; topk overlap 0.99925; attn_out 0.00186; h_mid 0.00189 /
  0.0034; out rel 0.00201.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_03_attn_norm.py

## C.dense_full.q_a test (attempt 1)

What was done
- Reviewed the rendered component test (q_a = q_a_layernorm(q_a_proj(attn_norm)), eps 1e-6, plain w). Kept the
  gated pcc_q_a_L00 (0.99) and added asserted checks vs the golden: no CPU bridge, element count, finite output,
  rel L2 <= 0.008, row norm ratio in [0.994, 1.006], worst row rel L2 <= 0.015. Added a second run on the golden
  input x 0.01 (bf16) vs the CPU step on the same input (rel <= 0.01, worst row <= 0.02) to catch a wrong eps.
  Mutation tables (CPU, study script outside the repo) in the test docstring.

Gotchas
- Golden: pre-norm row rms 0.31-0.53, so eps is invisible on it (1e-5 and 0 score like the reference), and at x 0.1
  eps 1e-5 is still only rel 0.0025; x 0.01 gives 0.17 (eps 0: 0.027, 2e-6: 0.025) vs a bf16 estimate of 0.0023.
- Passing on PCC but caught by the extra checks: norm per K partial then sum (PCC 0.9996, rel 0.81), RMS over half
  the output columns (0.9995, rel 0.032), x 1.01, zeroed last row / last tile row.
- The test cannot see a bug in only one mesh column's copy of the replicated q_resid unless the host read-back uses
  that copy; the swap tests and the indexer / attention components consume q_resid on device.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999998, rel 0.00181, ratio [0.99964, 1.00038], worst row 0.0021; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Device (gate command): fails with "no device module for q_a yet", as expected before the implement step.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_q_a.py

## C.dense_full.q_a implement (attempt 1)

What was done
- `tt/q_a.py:TtQa`: q_a_proj^T [6144, 2048] bf16 K-split over mesh columns (chip column c holds rows
  c*3072..), `ttnn.linear` HiFi4 + fp32 dest with fp32 partial output -> `ttnn.all_reduce(cluster_axis=1)` (fp32
  [S/2, 2048]) -> `ttnn.bringup.rms_norm` (q_a_layernorm, eps 1e-6 = hy4_ref.LATENT_NORM_EPS, fp32 row-major weight)
  -> typecast bf16. Output q_resid [1, 1, S/2, 2048] per chip, replicated over the 2 columns of a row. No host work in
  `__call__`.
- hooks.py: `_QA_STEPS`, `_qa_module`, `_qa_host_fn` (harness boundary: attn_norm host -> column split bf16, read
  back column 0's copy with `row_split_to_host`); `device_component` serves q_a; "q_a" added to
  `DEVICE_STEPS["dense_full"]`, so the hybrid device_model runs it.
- `ttnn/ttnn/bringup/INDEX.md`: hy4_preview_d_p added to rms_norm_ttnn's "Used by" (fork unchanged).

Decisions
- all_reduce instead of ttMLA's reduce_scatter_minimal_async + high_bw_all_gather: no persistent buffers or
  semaphores to own, the same pair on a 2-chip axis; the components entry allows it. A perf step can switch.
- fp32 partials before the reduce (no bf16 rounding of the two K halves).
- rms_norm fork over native: native scaled every row ~0.1% low (probe vs fp32 CPU, fp32 out: ratio mean 0.99898,
  rel 0.00124; fork 0.99974 / 0.00062). `TtQa(norm_impl="native")` keeps the old op selectable.

Results (gate command)
- native (first run): PASS, pcc 0.999997, rel 0.002575, ratio [0.99798, 0.99984], worst row 0.00347; x0.01 rel 0.001996.
- fork (final): PASS, pcc 0.999998, rel 0.001984, ratio [0.99861, 1.00067], worst row 0.00296; x0.01 rel 0.001740,
  worst row 0.00210.

Gotchas
- tt-probe saves probes under tests/ttnn/unit_tests/operations/<name>/ (outside the allowed paths); deleted after.
- The fork's model-case test (optests role) needs a q_a call: fp32 TILE [1, 1, S/2, 2048] in, fp32 ROW_MAJOR
  [1, 1, 64, 32] weight, eps 1e-6, HiFi4 + fp32 dest.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_q_a.py

## S.dense_full.04 test (attempt 1)

What was done
- Reviewed the rendered swap test (attn_hc, attn_hc_pre, attn_norm, q_a on device). Kept the gated pcc_swap_out
  (0.98) and the trail; carried over swap 03's asserted checks (attn_hc gates, attn_x, attn_norm vs golden / vs CPU /
  x 0.1 eps check, topk set overlap, attn_out, h_mid, block out rel), and added for q_a: q_resid vs golden at the
  component limits (rel 0.008, row ratio [0.994, 1.006], worst row 0.015), vs the CPU q_a on the device attn_norm
  (same limits), and the q_a module on the device attn_norm x 0.01 vs CPU (rel 0.01, worst row 0.02; the eps check).
- Mutation table (CPU, study script /tmp/hy4_s04/study.py, outside the repo) in the test docstring.

Gotchas
- Nearly every q_a bug passes the 0.98 out gate: row halves swapped 0.9886, per-K norm then sum 0.9935, missing
  reduce 0.9990; only the zero stub (0.975) and no norm weight (0.971) fail it. The q_resid checks catch them all;
  attn_out / h_mid worst row catch zeroed rows (proposed in known_issues.md).
- run_safe_pytest's up-front collect pass prints a first block with pcc 0 / rel 1.0 (no device module run); only the
  second block is the real run.
- ttnn/ttnn/bringup/INDEX.md, known_issues.md and repo_map.md already had uncommitted changes from earlier steps when
  this step started.

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999999, q_resid rel 0.00165, topk overlap 0.99926, out rel 0.00167).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device): PASS. pcc_swap_out 0.999998; q_resid rel 0.00200 / ratio [0.99889, 1.00063] / worst row 0.00306;
  vs CPU same input 0.00176; x0.01 0.00175; attn_norm rel 0.00214; topk overlap 0.99925; attn_out 0.00186; h_mid
  0.00189 / 0.0034; out rel 0.00201.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_04_q_a.py

## C.dense_full.indexer test (attempt 1)

What was done
- Rewrote the rendered component test. Gated metric pcc_indexer_L00 is now the mean per-row set overlap
  (harness `topk_overlap` on the output with pads normalized to -1), not positional match: the fp32 CPU reference
  itself scores match 0.789 on the bf16 golden inputs, and the device's topk_large_indices output is unsorted.
- Extra asserted checks (informational metrics): integer output, S x 2048 elements; pads may be -1 or the uint32
  sentinel 0xFFFFFFFF; causal (no position > row position); no repeats per row; exactly min(pos + 1, 2048) valid
  positions per row; worst row overlap >= 0.97; every row selects its own position; and a second call on chunk 0
  (start 0, golden chunk-0 inputs, device ctx prefix_len 0) that must return exactly [0, pos] per row.
- Mutation table (CPU, study scripts /tmp/hy4_idx/, outside the repo) in the test docstring.

Gotchas
- Bugs that pass the 0.99 overlap: causal mask t <= s + 1, own key dropped (t < s), top-2047, a padded last row;
  the structural checks catch each (verified on the CPU with the test's own helpers). k_norm eps 1e-6 vs 1e-5 is
  invisible (and harmless).
- Device precision estimates: bf16 everything with fp32 scores 0.99896 (worst row 0.9961); bf16 scores 0.99699
  (0.9893); bfp8 q / k / cache as TtIndexer 0.99690; bfp8 + bf16 scores 0.99585 (0.9893). All pass.
- The implementation is called twice in one test (chunk 1 with the golden prefix, then chunk 0 with an empty
  prefix); it must take its prefix from each call's device ctx.

Results
- BRINGUP_IMPL=reference: PASS (overlap 0.999254, worst row 0.99707, self 1.0; chunk 0 exact).
- BRINGUP_IMPL=stub: FAIL (overlap below 0.99).
- Gate (device): fails with "no device module for indexer yet", as expected before the implement step.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_indexer.py

## C.dense_full.indexer implement (attempt 1)

What was done
- `tt/indexer.py:TtHy4Indexer`, adapted from deepseek_v3_d_p TtIndexer (same ops and cache layout):
  wk K-split (bf16, fp32 partials) -> all_reduce axis 1 -> layer_norm (eps 1e-5, weight + bias) -> bf16 ->
  rotary_embedding_indexed -> update_padded_kv_cache (bf16 block-cyclic cache striped over the 4 chips, tp_axis 1);
  q: mesh_partition(q_resid, axis 1) -> wq_b (replicated) -> nlp_create_qkv_heads -> rotary_embedding_indexed
  (seq_subshard_axis 1); weights_proj K-split fp32 (1/64 folded) -> all_reduce -> mesh_partition -> bf16;
  high_bw_all_gather (TP-inner slab rebuild) -> ttnn.bringup.ring_indexer_score_dsa (ring over axis 0, HiFi4 + fp32
  DEST, q 64 x k 32) -> topk_large_indices (k 2048, valid_length end) -> high_bw_all_gather over axis 1.
  Output [1, 1, S/2, 2048] uint32 per chip, replicated over axis 1. `setup(chunk, max_seq)` builds RoPE tables
  (block-cyclic), cache and scratch once per geometry; `__call__` has no host work.
- hooks.py: `_indexer_module`, `_IndexerHostFn` (harness boundary: with a device ctx it reloads the golden prefix
  each call; in the hybrid its cache persists, `_HybridState` resets / loads the prefix / reads index_key back from
  the device), "indexer" added to DEVICE_STEPS["dense_full"]; the hybrid also records the top-k in ref._topk for
  shared layers. `HY4_INDEXER_SCORE=native` selects the source score op (bf16 DEST).
- Forked `ttnn/cpp/ttnn/operations/experimental/indexer_score` -> `ttnn/ttnn/bringup/indexer_score`: opt-in fp32
  DEST for DSA scoring + the mask srcA reconfig it needs (CHANGELOG), source selection (tests/source.yaml, baseline
  128/129 on the fork, the one miss is the upstream "rejects fp32 DEST" test), tests/unit/test_fp32_dest.py.

Decisions
- rotary_offset=64 instead of the plan's host permutation of wq_b / wk / k_norm rows: the op ropes channels 64..127
  in place, so the cache is in checkpoint order and read-back needs no un-permute.
- Fork instead of composition: the source op's logits are rel 0.024 off (truncating bf16 MAC, LoFi-like blocked
  multiply), overlap 0.9853; k_chunk 32 alone reaches 0.9926 (thin margin); fp32 DEST 0.99708.
- k_chunk 32: the only kernel path whose gate multiply honours the fidelity; also avoids the L1 overflow of bf16 q/k.
- all_reduce + mesh_partition for the TP reduces (as TtQa), not reduce_scatter + persistent buffers.

Results (gate command)
- native, q 64 x k 256: FAIL 0.985320 (L1 overflow first at k 320).
- native, k 32 (probe): 0.99263. Fork fp32 DEST, k 64/128 before the mask fix: mask broken (0.74).
- final (fork fp32 DEST, q 64 x k 32): PASS, overlap 0.997077, worst row 0.99072, self 1.0, chunk 0 exact.

Gotchas
- ring_indexer_score_dsa rejects fp32 DEST upstream; the fork's IndexerScoreProgramConfig is its own type
  (ttnn.bringup.IndexerScoreProgramConfig), the source op does not accept it and vice versa.
- fork_op.py rewrote `ttnn::experimental::ccl::` into the fork namespace; fixed by hand (CHANGELOG).
- Upstream multi-device indexer_score tests open FABRIC_1D / torus: not carried (owner rule).
- Perf of the k 32 per-column path at 56k is unmeasured.
- Probes were saved under tests/ttnn/unit_tests/operations/hy4_indexer/ (deleted).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_indexer.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/indexer_score/tests/unit/test_fp32_dest.py
    PYTHONPATH=$PWD python -m models.demos.common.bringup.testing.fork_source --fork indexer_score

## S.dense_full.05 test (attempt 1)

What was done
- Rewrote the rendered swap test (attn_hc, attn_hc_pre, attn_norm, q_a, indexer on device), starting from swap 04.
  Kept the gated pcc_swap_out (0.98) and the trail, and every asserted check of swap 04 (gates, attn_x, attn_norm and
  q_resid vs golden / vs CPU / eps checks, attn_out, h_mid, block out rel <= 0.01). Replaced swap 04's topk overlap
  (>= 0.995 vs golden) with the component test's checks on the swapped topk: overlap >= 0.99 and worst row >= 0.97
  vs the golden and vs the CPU indexer on the device attn_norm / q_resid, causal, no repeats, exactly
  min(pos + 1, 2048) valid per row, own position selected; plus the indexer module on golden chunk 0 (start 0,
  device ctx prefix_len 0), which must return exactly [0, pos] per row.
- Mutation table (CPU, study script /tmp/hy4_s05/study.py, outside the repo) in the test docstring.

Decisions
- Overlap limit 0.99 (the component gate), not swap 04's 0.995: the device indexer scores 0.9970 (fp32 DEST,
  bf16 caches), below 0.995.
- Order is not checked: the device returns unsorted indices and shuffled columns leave attn_out unchanged.

Gotchas
- 18 of 24 indexer mutations pass the 0.98 out gate, including non-causal selection and RoPE from 0 (proposed in
  known_issues.md). The topk checks catch all of them.
- The trail's pcc_swap_topk is positional match: 0.79 for the reference, 0.0003 for the device (unsorted). Ignore it.
- run_safe_pytest's up-front collect pass prints a first block with pcc 0 / rel 1.0; only the second block is real.

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999999, topk overlap 0.99926 / worst 0.99756, chunk 0 exact).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device): PASS. pcc_swap_out 0.999998; topk vs golden 0.99704 / worst row 0.99121, vs CPU on the same input
  0.99713 / 0.99072, self 1.0, chunk 0 exact; attn_out rel 0.00187; h_mid 0.00190 / 0.00345; out rel 0.00203.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_05_indexer.py

## C.dense_full.attention test (attempt 1)

What was done
- Reviewed the rendered component test for attention (layer 0). Kept the gated pcc_attention_L00 (PCC, 0.99) on golden
  s4096 chunk 1. Added four asserted checks, each at rel L2 <= 0.01, row norm ratio in [0.99, 1.01], worst row <= 0.02:
  (1) golden chunk 1; (2) golden chunk 0 (start 0, device ctx prefix_len 0, rows with -1 pads); (3) probe: chunk 1
  inputs with a synthetic topk (per row 64 random causal positions, unsorted, then -1; seed 0) vs the CPU step on the
  same inputs; (4) attn_norm x 1e-3 (bf16) vs the CPU step, where the kv_a_layernorm eps 1e-6 matters.
- Mutation table (CPU, study script /tmp/hy4_c_attn/mut.py, outside the repo) in the test docstring.

Decisions
- Limits from a bf16 device estimate (bf16 W / q / kv / scores / P, fp32 acc): rel 0.0033, ratio [0.9988, 1.0010],
  worst row 0.0042 on the golden; probe 0.0030 / 0.0051; scaled 0.0030 / 0.0034. The limits give a 3-5x margin.
- The probe is what makes the module prove it attends to exactly the given positions in any order (the golden can
  not tell dense causal attention from top-2048: rel 0.0064). Pads at the tail only (the indexer's contract).
- eps 1e-5 (rms_norm_eps, ttMLA's default) fails the scaled check (rel 0.43): the reference follows HF's 1e-6.
- kv_latent is not read back here (no state API in the component contract); chunk 0 attends only to the chunk's own
  writes, and the ladder's state gate compares the cache.

Results
- BRINGUP_IMPL=reference: PASS (golden rel 0.00178, worst row 0.0019; chunk 0 0.00177; probe / scaled exact).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Device (gate): FAIL, NotImplementedError: no device module for attention yet (the implement step's job).

Gotchas
- The device module must reload the golden kv_latent prefix on every call with a device ctx (state_prefix,
  prefix_len; 2048 for chunk 1, 0 for chunk 0), and accept any order of valid positions with -1 pads at the tail.
- Four device calls per run (chunk 1, chunk 0, probe, scaled), plus two CPU reference calls (~10 s each).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attention.py

## C.dense_full.attention implement (attempt 1)

What was done
- `tt/attention.py:TtHy4Attention`, adapted from deepseek_v3_d_p ttMLA's sparse path (same ops, same BF16_RM latent
  cache striped block-cyclic over the 4 chips, tp_axis 1):
  kv stem: kv_a K-split (bf16, fp32 partials) -> all_reduce axis 1 -> slice 512 | 64 -> ttnn.bringup.rms_norm
  (kv_a_layernorm, eps 1e-6, fp32 in) and rotary_embedding_indexed (interleaved, block-cyclic tables) -> concat ->
  ROW_MAJOR -> update_padded_kv_cache. q stem: q_b_proj (32 heads per chip) -> nlp_create_q_heads_split (192 | 64)
  -> linear W_uk [1, 32, 192, 512] | RoPE -> concat 576. high_bw_all_gather (cluster_axis None, prefix [0, end))
  into a replicated [1, 1, max_seq, 576] scratch -> sparse_sdpa (scale 1/16, attention_sink = sink x 16 [1,1,1,32]
  bf16, k_chunk 128, block-cyclic remap, HiFi4 + fp32 dest) -> linear W_uv [1, 32, 512, 256] -> nlp_concat_heads
  -> gate: all_gather(attn_norm, axis 1) -> linear_gate (fp32 out) -> sigmoid -> multiply -> o_proj row-parallel
  (fp32 partials) -> reduce_scatter axis 1. Output attn_out [1, 1, S/2, 3072] fp32 per chip (column split).
  `setup(chunk, max_seq)` builds tables, cache and scratch once per geometry; `__call__` has no host work.
- hooks.py: `_attention_module`, `_AttentionHostFn` (harness boundary: topk -1 -> 0xFFFFFFFF uint32 ROW_MAJOR,
  row split; with a device ctx reloads the golden kv_latent prefix per call; in the hybrid the cache persists).
  "attention" added to DEVICE_STEPS["dense_full"]. `_HybridState` now holds a list of stateful device fns per layer,
  each owning one state tensor (`state_key`: index_key / kv_latent) for load_prefix / to_torch.

Decisions
- all_reduce / all_gather / reduce_scatter (sync, cluster_axis) instead of ttMLA's persistent-buffer CCLs sized for
  one chunk length: the ladder uses three chunk lengths.
- The gather's extent is ceil(end / chunk) * chunk, as ttMLA; sparse_sdpa remaps natural positions in-kernel.
- bf16 intermediates between ops (q_abs, sparse_sdpa out, W_uv out, gated product); fp32 only for the partials
  before a reduce and for the gate logits.

Results (gate command)
- PASS first run: pcc_attention_L00 0.999993; golden rel 0.00382, row ratio [0.99777, 1.00228], worst row 0.0052;
  chunk0 0.00384 / 0.0063; probe 0.00375 / 0.0061; scaled 0.00408 / 0.0058 (all limits rel 0.01, worst row 0.02).
- Probe (hybrid path, deleted): load_prefix + one chunk + read_state: prefix rows exact, chunk rows vs golden latent
  rel 0.0018, rope rel 0.0029.

Gotchas
- sparse_sdpa at HiFi4 + fp32 dest, 32 heads x 576, k128 fits L1 (plan open item 3).
- Device error (rel 0.0038) is a little above the test's bf16 estimate (0.0033); fp32 intermediates after
  sparse_sdpa are the knob if a later swap limit is tight.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attention.py

## S.dense_full.06 test (attempt 1)

What was done
- Rewrote the rendered swap test (attn_hc, attn_hc_pre, attn_norm, q_a, indexer, attention on device), starting from
  swap 05. Kept the gated pcc_swap_out (0.98), the trail and every asserted check of swap 05 (gates, attn_x,
  attn_norm / q_resid vs golden / vs CPU / eps, topk overlap + structure + chunk 0, h_mid rel <= 0.01 / worst row
  <= 0.05, block out rel <= 0.01). attn_out limits tightened to the component's (rel <= 0.01, row norm ratio
  [0.99, 1.01], worst row <= 0.02; swap 05 had worst row 0.05, no ratio), checked five ways: vs golden; vs the CPU
  attention on the device attn_norm / q_resid / topk; and the module again on golden chunk 0 (prefix_len 0, pads),
  on a probe topk (64 random causal positions per row, unsorted, seed 0) and on the device attn_norm x 1e-3 (eps),
  each vs the CPU step on the same inputs.
- Mutation table (CPU, study script /tmp/hy4_s06/study.py reusing /tmp/hy4_c_attn/mut.py, outside the repo) in the
  test docstring: 20 of 29 attention bugs pass the 0.98 out gate. Dense causal (rel 0.0064), prefix row halves
  swapped (0.0076) and pads read as key 0 (invisible on chunk 1) also pass every swap-05 limit; the probe and
  chunk 0 catch them.

Decisions
- kv_latent rows are not read back: module_under_test wraps the device fn in a lambda (no read_state), and the
  reference / stub modes have no state API. The ladder's state gate compares the cache.
- Probe and eps inputs are the device attn_norm / q_resid (the step's own error, as swap 05's eps checks).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999999, attn_out vs golden 0.00166, every vs-CPU check exact).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device): PASS. pcc_swap_out 0.999987; attn_out vs golden rel 0.00452 / ratio [0.99733, 1.00186] / worst row
  0.00648; vs CPU 0.00409 / 0.00593; chunk 0 0.00341 / 0.00605; probe 0.00375 / 0.00568; scaled 0.00451 / 0.00599;
  h_mid 0.00417 / 0.00785; out rel 0.00514. topk as swap 05 (0.99704 / 0.99121).

Gotchas
- Five device attention calls + five CPU attention calls per run; the run takes ~70 s on the device.
- The first result block printed by run_safe_pytest (collect pass) shows zeros; only the second is real.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_06_attention.py

## C.dense_full.attn_residual test (attempt 1)

What was done
- Reviewed the rendered test for h_mid_j = in_j + post_j * attn_out (4 iHC streams, post = attn_hc cols 4-7). Kept
  the gated PCC (0.99) and added, vs golden: not a CPU bridge, size, finite, rel L2 <= 0.01, per-token per-stream norm
  ratio [0.98, 1.02]; per stream on the addend (delta_j = out_j - in_j vs post_j * attn_out): coefficient in
  [0.97, 1.03], rel L2 <= 0.03, worst row rel <= 0.1.
- CPU mutation study (/tmp/hy4_c_res/study.py, outside the repo) in the test docstring. The addend is large here
  (||attn_out|| 510 vs ||in|| 245), unlike MiMo's sink-dominated residual; 1.1 x attn_out (PCC 0.9978), last row
  zeroed (0.9994) and last 32 columns zeroed (0.9977) pass PCC and are caught by rel L2 / ratio / addend checks.

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999996, rel 0.00267 = golden bf16 rounding, ratio [0.9959, 1.0043], addend exact).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device): FAIL, NotImplementedError: no device module for attn_residual yet (expected before implement).

Gotchas
- Layer 0 streams are identical, so an input-stream permutation is invisible; post-column permutations are caught.
- The first result line of run_safe_pytest (collect pass) prints pcc 0; only the second is real.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_residual.py

## C.dense_full.attn_residual implement (attempt 1)

What was done
- `tt/ihc.py:TtHcPost`: iHC post, h_j = stream_j + post_j * y, fp32. Per chip: 4 x (slice stream j's
  [S/2, 3072] block, slice gate column 4+j [S/2, 1], `ttnn.addcmul(stream_j, y, post_j)`) -> `ttnn.concat` back to
  [1, 1, S/2, 4 x 3072]. No collective, no weights, no host work (y is typecast to fp32 on the device if it is not).
  Reuse: deepseek_v3_d_p tt_mhc `TtMHCWrap.hc_post` without the comb terms.
- hooks.py: `_HC_POST_STEPS = {"attn_residual"}`, `_hc_post_host_fn` (streams / row-split gates / column-split
  attn_out in, streams out; harness boundary only), wired into `_device_step_fn` and `device_component`;
  "attn_residual" added to `DEVICE_STEPS["dense_full"]`.

Decisions
- Kept the output fp32 (the residual streams are fp32 on the device, as TtHcGates / TtHcPre expect).
- TtHcPost is generic for ffn_residual too (same layout: y column-split [S/2, H/2]); only the step set needs extending.

Results
- Gate (device): PASS. pcc 0.999996, rel L2 0.00267 (= golden bf16 rounding), stream norm ratio [0.9959, 1.0043],
  addend coef 1.0 / rel 0.0 on every stream (bit-identical to the fp32 CPU step).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_attn_residual.py

## S.dense_full.07 test (attempt 1)

What was done
- Rewrote the rendered swap test (steps 1-7 on device, last attn_residual) from swap 06: kept the gated pcc_swap_out
  (0.98), the trail and every swap-06 check (gates, attn_x, attn_norm, q_resid, topk, attn_out five ways, block out
  rel <= 0.01). Added for h_mid (the swapped step): per-token per-stream norm ratio vs golden [0.98, 1.02]; vs the
  CPU attn_residual on the same device inputs (golden in, device attn_hc, device attn_out) rel <= 5e-4 / worst row
  <= 1e-3, plus the component's per-stream addend checks (coef [0.97, 1.03], rel <= 0.03, worst row <= 0.1); and the
  module on distinct input streams (golden h_mid as streams) vs the CPU step at the same tight limits.
- CPU mutation study (/tmp/hy4_s07/study.py, outside the repo) in the docstring: 5 of 16 residual bugs pass the 0.98
  out gate (bf16 output, 1.01 x / 1.1 x attn_out, last row zeroed, last 32 columns zeroed); every one fails an added
  check.

Decisions
- vs-CPU limit 5e-4 (not the component's 0.01): the device step is fp32 in / fp32 out and bit-identical to the CPU
  step, so a bf16 output (0.0017) or a 1 % scale (0.0087) is caught; the golden-side limits stay loose because
  upstream device error (attn_out rel 0.0045) dominates h_mid vs golden.
- Distinct-stream probe: layer 0's four input streams are identical, so an input-stream permutation or stream layout
  bug is invisible on the block's own inputs.

Results
- BRINGUP_IMPL=reference: PASS (every vs-CPU check exact). BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999987 (rel 0.00514); h_mid vs golden 0.00417 / worst row 0.00785, stream ratio
  [0.99842, 1.00514]; h_mid vs CPU 0 / 0; addend coef 1.0, rel 0; distinct-stream probe 0.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_07_attn_residual.py

## C.dense_full.ffn_hc test (attempt 1)

What was done
- Reviewed the rendered component test for ffn_hc (iHC gates [S, 8] fp32 from h_mid, hc_mlp_layer weights, layer 0,
  s4096 chunk 1, bf16 golden). Kept the gated PCC; added, vs the golden: finite, element count, rel L2 <= 0.01,
  post columns (4-7) rel L2 <= 0.006 each and worst row <= 0.015, and the pre gates through the CPU ffn_hc_pre on the
  golden h_mid (ffn_x from device gates vs from golden gates: rel <= 0.005, worst row <= 0.02). Asserts
  device_component is not a CPU bridge.
- CPU mutation study (/tmp/hy4_ffnhc/study*.py, outside the repo); the table is in the test docstring.

Decisions
- Not a copy of the attn_hc checks: here the pre gates are small (col means 0.094, ~1e-5, 0.0009, 0.018), so a
  per-column max abs 0.015 sees nothing on pre, and per-column rel is meaningless on col 1 (~1e-5, hc_eps included).
  The downstream ffn_x measures what an absolute pre error does (1e-3 abs -> ffn_x rel 0.029; 3e-4 -> 0.0085).
- No synthetic distinct-stream probe: h_mid's streams are already distinct (18 % / 6 % / 111 % vs stream 0), and
  every tried stream permutation fails (ffn_x rel >= 0.019).

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, rel 0.00167, post col rel <= 0.0017, post row 0.0031, ffn_x 0.0012 /
  row 0.0035). BRINGUP_IMPL=stub: FAIL (pcc 0).
- Gate (device): FAIL, NotImplementedError: no device module for ffn_hc yet (expected before implement).

Gotchas for implement
- TtHcGates with hc_mlp_layer (add "ffn_hc" to `_HC_STEPS`) should pass; the sigmoid must be accurate in absolute
  terms on outputs near 0 (a few 1e-4), keep the default Accurate sigmoid and fp32.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_hc.py

## C.dense_full.ffn_hc implement (attempt 1)

- No new module: ffn_hc is the same iHC gate block as attn_hc (`tt/ihc.py:TtHcGates`), with the `hc_mlp_layer`
  fn / base / scale, applied to h_mid. hooks.py: `_HC_STEPS["ffn_hc"] = "hc_mlp_layer"` (so `device_component` and
  `_device_step_fn` route it through `_hc_module` / `_hc_host_fn`), and "ffn_hc" added to `DEVICE_STEPS["dense_full"]`
  for the hybrid ladder model.
- Device, layer 0, s4096 chunk 1: PCC 0.999997, rel L2 0.00173 (CPU fp32 reference 0.00167); post col rel <= 0.00177,
  post worst row 0.00327; ffn_x rel 0.00202, worst row 0.00463 (limits 0.005 / 0.02). Max abs error on the pre
  columns is 5.4e-4 / 1.6e-7 / 2.0e-5 / 1.6e-4. The fp32 SFPU sigmoid is accurate enough for the small pre gates, so
  no change was needed.
- The streams are distinct at h_mid, so this gate also confirms that the chip-major fn permutation
  (`streams_cols_to_chip_major`) matches the stream order.
- Gotcha: the log's first `FAIL pcc_ffn_hc_L00: pcc=0.000000` line comes from the precompile collect pass, not the
  real run.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_hc.py`

## S.dense_full.08 test (attempt 1)

What was done
- Rewrote the rendered swap test (steps 1-8 on device, last ffn_hc) from swap 07. It keeps the gated pcc_swap_out
  (0.98) and every swap-07 check, and adds these ffn_hc checks:
  - vs golden at the component's limits: rel <= 0.01, post column <= 0.006, post row <= 0.015. ffn_x vs golden:
    rel <= 0.01, row <= 0.05.
  - vs the CPU ffn_hc on the same device h_mid: rel <= 0.005, post column <= 0.004, post row <= 0.01. ffn_x from
    device gates vs from CPU gates: <= 0.005 / row 0.02.
  - block out vs the CPU tail (ffn_hc .. ffn_residual) run from the device h_mid: <= 0.005 / row 0.02.
  - the module once more on h_mid x 0.1 (the rms_norm_eps check), with the same vs-CPU limits.
- CPU mutation study in /tmp/hy4_s08/study{,2}.py (outside the repo); the table is in the docstring. 17 of 27
  mutations pass the out gate. Every real bug among them fails an added check. bf16 output and a dropped hc_eps
  pass (out change <= 0.0024).

Decisions
- The vs-CPU checks carry the tight limits. The golden-side limits stay at component level, because the device
  h_mid is already rel 0.004 off the golden.
- No synthetic distinct-stream probe: the streams at h_mid are already distinct, and stream-order mutations fail.
- The out-vs-CPU-tail check isolates what the post gates do to out, with the upstream error removed.

Results
- BRINGUP_IMPL=reference: PASS (every vs-CPU check is exactly 0). BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS.
  - pcc_swap_out 0.999985, out rel 0.00541.
  - ffn_hc: vs golden 0.0023 / post column 0.0028 / post row 0.0067; vs CPU 0.00062 / 0.00076 / 0.0015.
  - ffn_x: vs golden 0.0046 / row 0.0082; vs CPU 0.0022 / row 0.0030, row norm ratio [1.0005, 1.0029] (the device
    pre gates run slightly high).
  - out vs CPU tail 0.0023 / row 0.0047; scaled probe 0.00024 / ffn_x 0.0013.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_08_ffn_hc.py

## C.dense_full.ffn_hc_pre test (attempt 1)

What was done
- Reviewed the rendered component test (ffn_x = sum_j pre_j x h_mid stream j, layer 0, s4096 chunk 1, bf16 golden).
  Kept the gated pcc_ffn_hc_pre_L00 (0.99) and the CPU-bridge assert pattern; added asserted checks:
  - vs golden: finite, element count, rel L2 <= 0.005, row norm ratio in [0.994, 1.006], worst row <= 0.01.
  - vs the CPU step on the same golden inputs: rel L2 <= 0.003, worst row <= 0.006.
  - the module once more with each row's pre gates rotated by (row mod 4), vs the CPU step: rel <= 0.004, row <= 0.01.
  Metrics rel_l2_*, worst_row_rel_l2_*, cpu_rel_l2_*, rot_rel_l2_* recorded (informational).
- CPU mutation study in /tmp/hy4_ffnhcpre/study{,2}.py (outside the repo); tables in the test docstring.

Decisions and why
- h_mid's streams are distinct, so no synthetic streams (unlike attn_hc_pre); but stream 1's pre gate is ~4e-6, so
  dropping it is invisible on the golden. Rotating the gates per row gives every stream the large gate on a quarter
  of the rows (stream 1 dropped: rel 0.34).
- The fp32 CPU step is rel 0.0024 off the golden (golden from fp32 upstream, gates rounded to bf16), so the tight
  limits are vs the CPU step on the same inputs; the golden limits sit above bf16 accumulation (0.0032, ratio 1.0035).

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, rel 0.00241, ratio [0.99658, 1.00324], row 0.00405; vs CPU and
  rotated exact). BRINGUP_IMPL=stub: FAIL (pcc 0).
- Gate (device): FAIL, NotImplementedError (no device module for ffn_hc_pre yet; expected before implement).

Gotcha for implement
- Same module as attn_hc_pre (`tt/ihc.py:TtHcPre`): add "ffn_hc_pre" to `hooks._HC_PRE_STEPS` (and DEVICE_STEPS).
  It is called twice (golden gates, then rotated gates).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_hc_pre.py

## C.dense_full.ffn_hc_pre implement (attempt 1)

What was done
- No new module: ffn_hc_pre is the same math as attn_hc_pre, so it reuses `tt/ihc.py:TtHcPre` (4 slices of the
  streams, 4 slices of pre-gate columns, then multiply + 3 x addcmul in fp32; no collective, no weights, no host
  work). The inputs are h_mid and ffn_hc; the output ffn_x is [1, 1, S/2, 3072] fp32 per chip.
- `bringup/hooks.py`: "ffn_hc_pre" added to `_HC_PRE_STEPS` (so `device_component` / `_device_step_fn` route it
  through `_hc_pre_host_fn`) and to `DEVICE_STEPS["dense_full"]` (hybrid device_model). The DEVICE_STEPS dict is
  now one entry per line (black line length).

Results
- Gate: PASS. pcc_ffn_hc_pre_L00 0.999997. vs golden: rel L2 0.00241, row ratio [0.99658, 1.00324], worst row
  0.00405, the same as the fp32 CPU step. vs the CPU step: rel 0.0. Rotated gates: rel 0.0. The device fp32
  multiply/addcmul chain matches torch's fp32 4-term sum bit for bit, as it did for attn_hc_pre.
- The "FAIL pcc ... 0.000000" line at the top of the log comes from the stubbed precompile collect pass, not
  the real run.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_hc_pre.py

## S.dense_full.09 test (attempt 1)

What was done
- Rewrote the rendered swap test (steps 1-9 on device, the last one ffn_hc_pre) starting from swap 08. It keeps the
  gated pcc_swap_out (0.98) and every swap-08 check. Swap 08's ffn_x-vs-golden and out-vs-CPU-tail checks now see
  the device ffn_hc_pre. It adds these checks:
  - ffn_x (device) vs the CPU ffn_hc_pre on the same device h_mid and gates: rel <= 0.003, row <= 0.006.
  - the module again with each row's pre gates rotated by row mod 4, vs the CPU step: rel <= 0.004, row <= 0.01.
  - the module on the scaled probe (h_mid x 0.1, the device gates for it) vs the CPU step: the same limits as the
    first check.
- CPU mutation study in /tmp/hy4_s09/study.py (outside the repo); the table is in the test docstring. 8 of 20
  mutations pass the out gate. Each one fails an added check. Stream 1 dropped is caught only by the rotated probe.

Decisions
- The limits are the component test's. The device module is fp32 and matches the CPU step bit for bit, so the
  checks against the CPU step have large margins.

Results
- BRINGUP_IMPL=reference: PASS. BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999985, out rel 0.00541. ffn_x vs golden 0.0046 / row 0.0082. ffn_hc_pre vs
  CPU, rotated and scaled: all 0.0. out vs CPU tail 0.0023 / row 0.0047.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_09_ffn_hc_pre.py

## C.dense_full.ffn_norm test (attempt 1)

What was done
- Reviewed the rendered component test (ffn_norm = post_attention_layernorm, w * x * rsqrt(mean(x^2) + 1e-5), plain
  w, on ffn_x). Kept the gated pcc_ffn_norm_L00 (0.99) and added the attn_norm test's asserted checks vs the golden:
  finite, element count, rel L2 <= 0.008, row norm ratio in [0.993, 1.007], worst row <= 0.015. Asserts the module is
  not a CPU bridge. Added a second run on the golden input x 30 (bf16) vs the CPU step on the same input: rel <= 0.006,
  ratio in [0.993, 1.007], worst row <= 0.015.
- CPU mutation study in /tmp/hy4_ffnnorm/study{,2}.py (outside the repo); tables in the test docstring.

Decisions and why
- ffn_x is small (row rms 0.00083-0.0099; 76% of rows have mean(x^2) < eps), so eps dominates the golden. The
  opposite of attn_norm: here eps errors are large on the golden (eps 1.2e-5 rel 0.052, 1e-6 PCC 0.968). But the RMS
  reduction is damped (half columns rel 0.0044, LayerNorm 0.0075). So the synthetic probe scales the input up (x 30,
  mean(x^2) >= 60x eps) instead of down (x 0.1 would only test the eps-dominated regime again).
- Golden limits are the same as attn_norm (device noise estimate 0.0037 / [0.996, 1.0038] / 0.0053; attn_norm's
  device module measured 0.0017).

Results
- BRINGUP_IMPL=reference: PASS (pcc 1.0, rel 0.00234, ratio [0.99952, 1.00048], worst row 0.0028; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): FAIL, NotImplementedError "no device module for ffn_norm yet" (expected before implement).

Gotcha for implement
- Same module as attn_norm (`tt/norm.py:TtDistributedRmsNorm`, fp32 input, stats mask): add
  "ffn_norm": "post_attention_layernorm" to `hooks._NORM_STEPS` (and DEVICE_STEPS). It is called twice (golden,
  then x 30).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_norm.py

## C.dense_full.ffn_norm implement (attempt 1)

What was done
- `tt/norm.py:TtGatheredRmsNorm` (new): `ttnn.all_gather(x, dim=3, cluster_axis=1, Linear)` of the column-split
  ffn_x [1, 1, S/2, 3072] fp32 -> [S/2, 6144], then `ttnn.bringup.rms_norm` (post_attention_layernorm full weight as
  fp32 row-major [1, 1, 192, 32] replicated, eps rms_norm_eps 1e-5, HiFi4 + fp32 dest) -> typecast to bf16. Output
  [1, 1, S/2, 6144] bf16 per chip, replicated over the 2 columns of a row (components.yaml / plan.md: the dense MLP,
  router, dispatch and shared expert all take the full hidden). `norm_impl="native"` keeps ttnn.rms_norm selectable.
  No host work in __call__.
- `bringup/hooks.py`: `_GATHERED_NORM_STEPS = {"ffn_norm": "post_attention_layernorm"}`, `_gathered_norm_module`,
  harness boundary `_col_in_row_out_host_fn` (col_split_to_device in, row_split_to_host out, column 0's copy);
  routed in `_device_step_fn` / `device_component`; "ffn_norm" added to `DEVICE_STEPS["dense_full"]`.
- `ttnn/ttnn/bringup/INDEX.md`: rms_norm_ttnn "Used by" lists tt/norm.py:TtGatheredRmsNorm.

Decisions
- Followed components.yaml (gather + local fork norm), not the attn_norm distributed norm: the output must be the full
  hidden anyway, and the gathered form needs no stats mask (known issue: rms_norm_pre_all_gather junk on fp32 input).
- bf16 output (as attn_norm / q_a); the golden is bf16. The MLP implement step can ask for fp32 via `dtype=`.

Results
- Gate: PASS. pcc_ffn_norm_L00 0.999996. golden: rel 0.00284, row ratio [0.99874, 1.00071], worst row 0.00352.
  scaled x30: rel 0.00171, ratio [0.99933, 1.00100], worst row 0.00203.
- The "FAIL pcc ... 0.000000" line at the top of the log is the stubbed precompile collect pass.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_norm.py

## S.dense_full.10 test (attempt 1)

What was done
- Rewrote the rendered swap test (steps 1-10 on device, the last one ffn_norm) starting from swap 09. It keeps the
  gated pcc_swap_out (0.98) and every swap-09 check. Swap 08's out-vs-CPU-tail check now also sees the device
  ffn_norm. It adds these checks:
  - ffn_norm vs golden: rel <= 0.01, row norm ratio [0.99, 1.01], worst row <= 0.03 (upstream error included).
  - ffn_norm vs the CPU ffn_norm on the same device ffn_x: rel <= 0.008, ratio [0.993, 1.007], row <= 0.015.
  - the module again on device ffn_x x 30 (bf16) vs the CPU step: rel <= 0.006, ratio [0.993, 1.007], row <= 0.015.
- CPU mutation study in /tmp/hy4_s10/study.py (outside the repo); the table is in the test docstring. 8 of 27
  mutations pass the out gate. Each one fails an added check or an existing out check.

Decisions
- The limits vs the CPU step and on x 30 are the component test's. The limits vs golden are looser (swap style)
  because the upstream device ffn_x error (rel 0.0046) is included.
- RMS over half the columns and LayerNorm sit right at the existing out limits (0.0103 / 0.0102 on the CPU), so
  the added checks are what catches them (ratio and worst row).

Results
- BRINGUP_IMPL=reference: PASS. BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999984, out rel 0.00564. ffn_norm vs golden 0.0044 / [0.9976, 1.0017] /
  0.0074; vs CPU 0.0018 / [0.9983, 1.0003] / 0.0024; x 30 0.0017 / [0.9992, 1.0010] / 0.0021. out vs CPU tail
  0.0026 / row 0.0046 (swap 09: 0.0023).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_10_ffn_norm.py

## C.dense_full.mlp test (attempt 1)

What was done
- Reviewed the rendered component test for mlp (layer 0, HF HYV4MLP: down(silu(gate(x)) * up(x)), 18432, unclamped;
  swiglu_limit applies to the routed experts only). Kept the gated pcc_mlp_L00 (0.99, PCC on the bf16 golden) and
  added, vs the golden: finite, element count, rel L2 <= 0.008, row norm ratio in [0.993, 1.007], worst row
  <= 0.015. Asserts the module is not a CPU bridge. Added a second run on ffn_norm x 30 (bf16) vs the CPU mlp on the
  same input: rel <= 0.006, ratio in [0.993, 1.007], worst row <= 0.012.
- CPU mutation study in /tmp/hy4_mlp/study{,2}.py (outside the repo); tables in the test docstring.

Decisions and why
- Passing PCC 0.99 on the golden: gelu_tanh (0.9937), HiFi2-like truncation (0.99999, rel 0.024, ratio ~0.976),
  x 1.01 / 1.03, last row / last 32 rows zeroed (0.9912), input x 0.5 (0.996). The added golden checks catch all of
  them; device noise estimate (bf16 gate/up/h, fp32 acc) is 0.0029 / [0.998, 1.002] / 0.0038.
- On the golden gate is in [-1.11, 0.42]: a clamp at 10 (ClampedSiluGlu reused from the experts) is invisible. At
  x 30 gate reaches 12.5 and the clamp shows as worst row 0.021 vs noise 0.0029, hence the 0.012 limit.
- bfp8 weights would score rel 0.0061 on the golden and pass; the plan says bf16 weights anyway.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, rel 0.00238, ratio [0.99961, 1.00038], worst row 0.00254; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): FAIL, NotImplementedError "no device module for mlp yet" (expected before implement).

Gotcha for implement
- The module is called twice (golden, then x 30). Its input is ffn_norm [S, 6144] on the host boundary (device
  ffn_norm output is replicated over columns); output mlp_out [S, 6144] (reduce_scatter to [S/2, 3072] per chip on
  the device, per plan). No clamp. Scaled outputs reach row norms ~6300, fine in bf16/fp32.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_mlp.py

## C.dense_full.mlp implement (attempt 1)

What was done
- `tt/mlp.py:TtDenseMLP` (new, from `mimo_v2_6_d_p_2x2/tt/mlp.py:TtDenseMLP`): TP=2 over axis 1. Chip column c
  holds gate / up W^T columns and down W^T rows [9216c, 9216c + 9216), replicated over the rows. The weights are bf16
  as stored (`ShardTensor2dMesh dims=(None, 3)` / `(None, 2)`), 340 MB per chip. Forward: `ttnn.linear` gate and
  up (fp32 out) -> `ttnn.multiply(input_tensor_a_activations=[SILU], dtype=fp32)` -> `ttnn.linear` down (fp32
  partial [S/2, 6144]) -> `ttnn.reduce_scatter(dim=3, cluster_axis=1)` -> mlp_out [1, 1, S/2, 3072] fp32, the
  residual's column split (the same epilogue as the attention's o_proj). Every matmul runs HiFi4 + fp32 dest. There
  is no clamp and no host work in `__call__`.
- `bringup/hooks.py`: `_MLP_STEPS`, `_mlp_module`, and the harness boundary `_row_in_col_out_host_fn` (bf16
  `row_split_to_device` in, the TtGatheredRmsNorm output layout; `col_split_to_host` out). The step is routed in
  `_device_step_fn` / `device_component`, and "mlp" is added to `DEVICE_STEPS["dense_full"]`. `HY4_MLP_MID=bf16`
  selects bf16 gate / up / h for comparison.

Decisions
- fp32 intermediates and HiFi4, per components.yaml and known issues. With them the device matches the fp32 CPU
  reference on the golden (rel 0.00247 vs the reference's own 0.00238).

Results
- Gate: PASS. pcc_mlp_L00 0.999997. Golden: rel 0.00247, row ratio [0.99907, 0.99986], worst row 0.00267. Scaled
  x30 vs CPU: rel 0.00069, ratio [0.99944, 0.99950], worst row 0.00075.
- The "FAIL pcc ... 0.000000" line at the top of the log comes from the stubbed precompile collect pass.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_mlp.py

## S.dense_full.11 test (attempt 1)

What was done
- Rewrote the rendered swap test (steps 1-11 on device, the last one mlp) starting from swap 10. It keeps the gated
  pcc_swap_out (0.98) and every swap-10 check. The out-vs-CPU-tail check now also sees the device mlp. It adds these
  checks:
  - mlp_out vs golden: rel <= 0.01, row norm ratio [0.99, 1.01], worst row <= 0.03 (upstream error included).
  - mlp_out vs the CPU mlp on the same device ffn_norm: rel <= 0.008, ratio [0.993, 1.007], row <= 0.015.
  - the module again on device ffn_norm x 30 (bf16) vs the CPU step: rel <= 0.006, ratio [0.993, 1.007],
    row <= 0.012.
- CPU mutation study in /tmp/hy4_s11/study.py (outside the repo); the table is in the test docstring. 11 of 25
  mutations pass the out gate. The existing out checks catch all of them except a SwiGLU clamp at 10, which the
  golden cannot see. The x 30 check catches the clamp (worst row 0.021).

Decisions
- The limits vs the CPU step and on x 30 are the component test's. The limits vs golden are the same swap-style
  ones used for ffn_norm, because the upstream device ffn_norm error (rel 0.0044) is included.

Results
- BRINGUP_IMPL=reference: PASS. BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999984, out rel 0.00561. mlp_out vs golden 0.0043 / [0.9949, 1.0029] /
  0.0083; vs CPU 0.00065 / [0.9993, 0.9996] / 0.0008; x 30 0.00069 / [0.9994, 0.9995] / 0.00075. out vs CPU tail
  0.0024 / row 0.0041.
- The first block of printed metrics in each log (all zeros) comes from the stubbed precompile collect pass.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_11_mlp.py

## C.dense_full.ffn_residual test (attempt 1)

What was done
- Reviewed the rendered test for out_j = h_mid_j + post_j * mlp_out (4 iHC streams, post = ffn_hc cols 4-7, the
  same hc_post as attn_residual). Rewrote it from the frozen attn_residual test. It keeps the gated PCC (0.99) and
  adds, vs golden: not a CPU bridge, size, finite, rel L2 <= 0.01, per-token per-stream norm ratio [0.99, 1.01];
  per stream on the addend (delta_j = out_j - h_mid_j vs post_j * mlp_out): coefficient [0.97, 1.03], rel L2 <= 0.03,
  worst row rel <= 0.05.
- CPU mutation study (/tmp/hy4_c_ffnres/study.py, outside the repo). The table is in the test docstring. These pass
  PCC and are caught by the extra checks: 1.02 x / 1.05 x / 1.1 x mlp_out, last row zeroed, last 32 columns zeroed,
  mlp_out first or last tile row zeroed.

Decisions
- Tighter than attn_residual on the stream ratio ([0.99, 1.01] vs [0.98, 1.02]) and the worst row (0.05 vs 0.1).
  On this golden the reference gives [0.9985, 1.0016] and a bf16 output gives a worst row of 0.006.

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999990, rel 0.00437 = golden bf16 rounding, ratio [0.9985, 1.0016], addend exact).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device): FAIL, NotImplementedError: no device module for ffn_residual yet. This is expected before implement.

Gotchas
- Unlike attn_residual, the h_mid streams differ at layer 0, so a stream swap is visible (streams 0 / 1 swapped:
  PCC 0.984).
- ||out|| (167) is smaller than ||h_mid|| (285) because the addend partly cancels h_mid. That is why the golden's
  bf16 rounding costs rel 0.0044 here, against 0.0027 for attn_residual.
- The first pcc line of run_safe_pytest comes from the collect pass and prints 0. Only the second line is real.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_residual.py

## C.dense_full.ffn_residual implement (attempt 1)

What was done
- ffn_residual (out_j = h_mid_j + post_j * mlp_out) is the same iHC post as attn_residual, so it reuses
  `tt/ihc.py:TtHcPost` unchanged: 4 x (ttnn.slice stream_j, ttnn.slice post gate column 4+j, ttnn.addcmul) in fp32 ->
  ttnn.concat. No weights, no collective, no host work in the forward.
- hooks.py: "ffn_residual" is added to `_HC_POST_STEPS` (so `device_component` / `_device_step_fn` route it through
  `_hc_post_host_fn`) and to `DEVICE_STEPS["dense_full"]` (the hybrid device_model for the ladder). No new code in tt/.

Results
- Gate PASS: pcc_ffn_residual_L00 0.999990, rel L2 0.00437, stream norm ratio [0.9985, 1.0016], addend coef 1.0 and
  rel 0.0 on all 4 streams, worst row 0.0. These match the CPU reference on the bf16 golden.

Gotchas
- The step signature (streams, gates, y) and the gate column layout (post = columns 4-7) are the same as
  attn_residual. Only the inputs differ: h_mid / ffn_hc / mlp_out.
- The first pcc line (0.000000) comes from the precompile collect pass. The second line is the real one.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_dense_full_ffn_residual.py

## S.dense_full.12 test (attempt 1)

What was done
- Replaced the rendered 33-line swap test with swap 11's test (every earlier check kept) plus ffn_residual in
  SWAPPED and the ffn_residual checks: out per-token per-stream norm ratio vs golden [0.98, 1.02]; out vs the CPU
  ffn_residual on the same device h_mid / ffn_hc / mlp_out (rel <= 5e-4, worst row <= 1e-3); addend per stream
  (out_j - h_mid_j vs post_j * mlp_out) coef [0.97, 1.03], rel <= 0.03, worst row <= 0.05 (the component's limits).
- CPU mutation study (/tmp/hy4_s12/study.py, outside the repo); table in the test docstring. 9 of 23 mutations pass
  the 0.98 out gate (1.01-1.1 x mlp_out, last row / last 32 columns zeroed, post gates 1 / 2 swapped 0.985, h_mid
  streams 0 / 1 swapped 0.984, 2 x the output at PCC 0.999999); all fail the vs-CPU check.

Decisions
- No distinct-stream probe (swap 07 needed one because layer-0 input streams are identical); at h_mid the streams
  and the post gates already differ, so stream / gate-order bugs show on the block's own inputs.

Results
- BRINGUP_IMPL=reference: PASS (out rel 0.0017, vs CPU 0). BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999984, out rel 0.00561, stream ratio [0.9956, 1.0059], out vs CPU
  ffn_residual 0 / 0 (fp32 addcmul, bit-identical), addend coef 1.0 / rel 0, out vs CPU tail 0.0024 / row 0.0041.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_dense_full_12_ffn_residual.py

## C.moe_full.attn_hc test (attempt 1)

What was done
- Reviewed the rendered 22-line test for attn_hc at layer 1 (iHC gates [S, 8], hc_attn_layer of layer 1, s4096
  chunk 1, bf16 golden). Rewrote it from the layer-0 attn_hc / ffn_hc tests. It keeps the gated PCC (0.99) and adds
  these checks vs the golden: not a CPU bridge, element count, finite, rel L2 <= 0.01, per-column rel L2 (all 8
  columns) <= 0.01, post worst row <= 0.015. It also checks the pre gates through the CPU attn_hc_pre on the golden
  streams (attn_x rel <= 0.005, worst row <= 0.02) and the post gates through the CPU attn_residual with the golden
  attn_out (h_mid per stream <= 0.003, worst row <= 0.02).
- CPU mutation study in /tmp/hy4hc1/{an,an2,mut,mut2}.py (outside the repo). It needs only the golden and the hc
  weights. The table is in the test docstring.

Decisions
- Per-column rel L2 on every column, unlike layer 0's ffn_hc. The smallest column (post 4, ~3e-4) is 300x hc_eps, so
  its relative error is meaningful. It catches the bugs that the whole-matrix metrics and the downstream metrics miss:
  a pre base swap (col 0.24, attn_x only 0.0031) and SP row swaps (col 0.022).
- No synthetic distinct-stream probe and no eps probe. At layer 1 the streams are already distinct (fn streams 0 / 1
  swapped: col 0.098), and the row RMS (0.005-0.07) is small enough that eps 1e-6 fails (col 0.081).

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999999, rel 0.00143, max col rel 0.0019, attn_x 0.00138 / 0.0043, h_mid
  stream <= 0.00105 / row 0.0029).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device, the existing TtHcGates through device_component): PASS. PCC 0.999998, rel 0.00149, col rel
  [0.0023, 0.0024, 0.0014, 0.0017, 0.0054, 0.0030, 0.0025, 0.0018], post row 0.0065, attn_x 0.00144 / 0.0041,
  h_mid stream <= 0.00124 / row 0.0048.

Gotchas
- The tightest margin is post column 4 (device 0.0054 vs limit 0.01; max abs 4.8e-5 on gates of ~3e-4). It comes
  from the device sigmoid on small outputs. If a later change (bf16 sigmoid, approx exp) touches TtHcGates, watch
  this column.
- Only dropping hc_eps goes undetected (downstream change < 1e-5).
- The first pcc line (0.000000) comes from the precompile collect pass.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_hc.py

## S.moe_full.01 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_full layer 1, attn_hc on device, rest CPU). It keeps the gated
  pcc_swap_out (0.98) and the trail. It adds asserted checks (informational metrics): not a CPU bridge; the gates vs
  golden with the component test's limits (8 columns, finite, rel L2 <= 0.01, per-column rel L2 <= 0.01, post worst
  row <= 0.015); attn_x rel <= 0.005 / worst row <= 0.02; h_mid rel <= 0.005 / worst (row, stream) <= 0.02; router
  top-8 selection overlap >= 0.98; block out finite and rel L2 <= 0.01.
- CPU block-level mutation study in /tmp/hy4_sm1/study.py (outside the repo). The table is in the test docstring.
  Every gate mutation except the zero stub and "post = 1 x sigmoid" passes the 0.98 out gate.

Decisions
- Per-column rel L2 instead of layer 0's per-column max abs. At layer 1 every column is well above hc_eps (see
  C.moe_full.attn_hc).
- The out worst (row, stream) rel L2 is recorded but not asserted: it is 0.056 on the fp32 reference, because a few
  near-tie tokens switch experts in the CPU router. The router overlap check is only a gross check (reference 0.9987).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999997, rel 0.0024, gates col rel max 0.0019, router 0.9987).
- BRINGUP_IMPL=stub: FAIL on every check (out PCC 0.860).
- Gate (device TtHcGates): PASS. pcc_swap_out 0.999996, gates rel 0.00149, col rel max 0.00536 (post 4), post row
  0.0065, attn_x 0.00223 / 0.0028, h_mid 0.00218 / 0.0037, router 0.9978, out rel 0.00277.

Gotchas
- The tightest margin is still post column 4 (0.0054 vs 0.01), the same as in the component test.
- Not caught: dropping hc_eps, post x 1.005 (inside the bf16 golden noise).
- The first block of printed metrics (and "UP_FRONT_COLLECT_RESULT: status=failed reason=incomplete") comes from the
  precompile collect pass. The second block is the real run.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_01_attn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_01_attn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_01_attn_hc.py

## C.moe_full.attn_hc_pre test (attempt 1)

What
- Replaced the rendered 22-line component test (moe_full layer 1, attn_hc_pre). It keeps the gated
  pcc_attn_hc_pre_L01 (0.99) and the CPU-bridge assert. It adds asserted checks, copied from the dense_full
  ffn_hc_pre test with layer-1 limits: finite output and element count; vs golden rel L2 <= 0.005, row norm ratio
  in [0.994, 1.006], worst row <= 0.01; vs the CPU step on the same inputs rel <= 0.003, worst row <= 0.006; the
  module run again with each row's pre gates rotated by row mod 4, vs the CPU step: rel <= 0.004, worst row <= 0.01.
- CPU mutation study in /tmp/hcpre_l1.py (outside the repo). The table is in the test docstring.

Decisions
- Layer 1 streams are distinct, so no synthetic streams (unlike layer-0 attn_hc_pre). The pre gates are unequal
  (means 0.026 / 0.013 / 0.83 / 0.49), so streams 0 / 1 swapped is only just over the golden rel limit (0.00504).
  Its worst row (0.027) and the rotated-gates run (0.105) catch it.
- The golden limits sit above the golden's own rounding (fp32 CPU step rel 0.0026 / row 0.0050).

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999997, rel 0.0026, ratio [0.99687, 1.00314], row 0.00495).
- BRINGUP_IMPL=stub: FAIL (PCC 0.0).
- Gate (device): PASS already. hooks._HC_PRE_STEPS does not depend on the block type, so layer 1 runs the existing
  tt/ihc.py:TtHcPre. It is fp32 and matches the CPU step exactly (cpu rel 0.0, rotated rel 0.0). moe_full
  DEVICE_STEPS is still empty (the hybrid is not changed).

Gotchas
- Not caught: pre + 3e-4 on every gate (rel 0.0028 vs golden). The gates are an input, so the module cannot make
  that error by itself.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_hc_pre.py

## S.moe_full.02 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_full layer 1, attn_hc + attn_hc_pre on device, rest CPU). It keeps the
  gated pcc_swap_out (0.98) and the trail. It adds asserted checks (informational metrics): not a CPU bridge; the
  gates vs golden as in swap 01 (rel <= 0.01, per-column rel <= 0.01, post worst row <= 0.015); attn_x vs golden
  (rel <= 0.005, row norm ratio in [0.996, 1.004], worst row <= 0.01); attn_x vs the CPU hc_pre on the block input +
  device gates (0.003 / 0.006); the module again with the device gates' pre columns rotated by row mod 4 vs the CPU
  step (0.004 / 0.01); h_mid rel <= 0.005 / worst (row, stream) <= 0.02; router overlap >= 0.98; out rel <= 0.01.
- CPU block-level mutation study in /tmp/hy4_sm2/study.py (outside the repo). The table is in the test docstring.
  14 of 20 attn_hc_pre mutations pass the 0.98 out gate. Every mutation fails at least one of the added checks.

Decisions
- Limits for the step are the component test's. The row norm ratio is tighter than there ([0.996, 1.004] vs [0.994,
  1.006]) because here the fp32 CPU step is [0.9998, 1.0003] vs golden (the gates come from the fp32 attn_hc on the
  block input); this catches pre x 1.005 (1.0048).
- No synthetic streams: the layer-1 streams are distinct; the rotated-gates run covers stream order (0 / 1 swapped
  0.105).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999997, rel 0.0024, attn_x 0.0022 / ratio [0.9998, 1.0003]).
- BRINGUP_IMPL=stub: FAIL on every check (out PCC 0.871).
- Gate (device TtHcGates + TtHcPre): PASS. pcc_swap_out 0.999996, gates col rel max 0.00536 (post 4), attn_x
  0.00223 / ratio [0.9992, 1.0002] / row 0.0028, vs CPU 0 / 0 (fp32, bit-identical), rotated 0 / 0, h_mid 0.00218 /
  0.0037, router 0.9978, out rel 0.00277.

Gotchas
- The tightest margin is still post column 4 of the gates (0.0054 vs 0.01), from swap 01.
- The first block of printed metrics comes from the precompile collect pass; the second is the real run.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_02_attn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_02_attn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_02_attn_hc_pre.py

## C.moe_full.attn_norm test (attempt 1)

What was done
- Replaced the rendered 22-line component test (moe_full layer 1, attn_norm) with the reviewed dense_full attn_norm
  test at LAYER = 1, with the same limits: gated pcc_attn_norm_L01 (0.99); not a CPU bridge; finite output, element
  count; vs golden rel L2 <= 0.008, row norm ratio in [0.993, 1.007], worst row <= 0.015; module run again on
  golden x 0.1 (bf16) vs the CPU step, rel <= 0.01, worst row <= 0.02. The layer-1 mutation tables are in the docstring.
- CPU mutation study in /tmp/hy4_an1/study.py (outside the repo).

Decisions
- Kept the layer-0 limits. The pessimistic bf16 estimate on layer 1 is rel 0.0038 / ratio [0.9953, 1.0046] / row
  0.0058, so there is still about 1.5-2x margin. Every mutation that passes PCC fails at least one added check.
- Kept the x 0.1 run, although it is not needed here: on layer 1 eps is already visible on the golden (row rms from
  0.0045, the smallest mean(x^2) is 2.1x eps; eps 1e-6 scores rel 0.068 / worst row 0.19).

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999999, rel 0.00235, ratio [0.99987, 1.00013], row 0.0025; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): already PASS, because the dense_full attn_norm module also serves layer 1. pcc 0.999996, rel
  0.00287, ratio [0.99906, 1.00092], row 0.00315; scaled rel 0.00183, row 0.00252.

Gotchas
- The first "FAIL pcc_attn_norm_L01: pcc=0.0" line comes from the precompile collect pass. The real run follows it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_norm.py

## S.moe_full.03 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_full layer 1, attn_hc + attn_hc_pre + attn_norm on device, rest CPU).
  It keeps the gated pcc_swap_out (0.98) and the trail. It combines the swap 02 checks (gates, attn_x vs golden, vs
  CPU, rotated pre gates, h_mid, block out rel <= 0.01) with the dense_full swap 03 attn_norm checks at the component
  limits: attn_norm vs golden (rel 0.008, ratio [0.993, 1.007], worst row 0.015), vs the CPU attn_norm on the device
  attn_x (same limits), the module again on attn_x x 0.1 vs CPU (0.01 / 0.02). It also checks the downstream steps:
  q_resid rel <= 0.01, indexer top-k set overlap >= 0.995, attn_out rel <= 0.01 / worst row <= 0.05, and router
  top-8 overlap >= 0.99.
- CPU block-level mutation study in /tmp/hy4_sm3/study.py (outside the repo, 9 s per variant). The table is in the
  test docstring. 12 of 17 attn_norm mutations pass the 0.98 out gate, including eps 1e-6 / 0 / 2e-5, x 1.02, the
  RMS subset, LayerNorm, a zeroed row and TP-swapped weight halves. The attn_norm checks catch every one of them.

Decisions
- Router overlap raised from 0.98 (swaps 01 / 02) to 0.99. The fp32 reference gives 0.9987, bf16 0.9981 and the
  device 0.9979; x 1.02 gives 0.9898. The attn_norm checks still carry the load.
- h_mid limits stay at the swap 02 values (0.005 / 0.02); the device run gives 0.0022 / 0.0037.

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999997, attn_norm 0.00235, topk 0.99909, router 0.99866, out rel 0.00242).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98, and every extra check fails).
- Gate (device TtHcGates + TtHcPre + the attn_norm module): PASS. pcc_swap_out 0.999996. attn_norm 0.00298, ratio
  [0.99826, 1.00003], worst row 0.00346. vs CPU 0.00196. x0.1 0.00183. q_resid 0.00195. topk 0.99908. attn_out
  0.00190. h_mid 0.00222 / 0.0037. router 0.99786. out rel 0.00296.

Gotchas
- The smallest margin is still post column 4 of the gates (0.0054 vs 0.01), carried over from swap 01.
- The first block of printed metrics (a stub-like FAIL trail) comes from the precompile collect pass. The second
  block is the real run.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_03_attn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_03_attn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_03_attn_norm.py

## C.moe_full.q_a test (attempt 1)

What was done
- Replaced the rendered one-line test with the layer-0 q_a test's checks, at LAYER = 1 and with the same limits:
  gated pcc_q_a_L01 (0.99), no CPU bridge, element count, finite output, rel L2 <= 0.008, row norm ratio in
  [0.994, 1.006], worst row rel L2 <= 0.015 vs the golden; a second run on the golden input x 0.01 (bf16) vs the CPU
  step on the same input (rel <= 0.01, worst row <= 0.02) to catch a wrong eps. Layer-1 mutation tables (CPU study
  script in /tmp, outside the repo) are in the test docstring.

Decisions
- Kept the layer-0 limits: the layer-1 golden has the same statistics (pre-norm row rms 0.358-0.497; bf16 estimate
  rel 0.0029, ratio [0.9995, 1.0004], worst row 0.0035), and every mutation in the table fails at least one check.

Gotchas
- At layer 1, a dropped norm weight (PCC 0.9914) and 1 + w (0.9939) pass PCC; at layer 0 they did not. Rel L2
  catches both. Known-issues proposal added.
- The first "FAIL pcc_q_a_L01: pcc=0.000000" line in each run is the precompile collect pass, not the real run.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999998, rel 0.00180, ratio [0.99971, 1.00028], worst row 0.0020; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Device (gate command): PASS, because the layer-0 TtQa module already serves layer 1: pcc 0.999998, rel 0.00198, ratio
  [0.99885, 1.00041], worst row 0.0027; scaled x 0.01 rel 0.00175, worst row 0.0020.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_q_a.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_q_a.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_q_a.py

## S.moe_full.04 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_full layer 1, attn_hc + attn_hc_pre + attn_norm + q_a on device, rest
  CPU) with swap 03's checks plus the dense_full swap 04 q_a checks at the component limits: q_resid vs golden (rel
  0.008, row ratio [0.994, 1.006], worst row 0.015), vs the CPU q_a on the device attn_norm (same limits), and the q_a
  module on the device attn_norm x 0.01 vs CPU (0.01 / 0.02, the eps check). The gated pcc_swap_out (0.98) and the
  trail are unchanged. Swap 03's downstream q_resid rel <= 0.01 became the full q_a check.
- CPU mutation study at layer 1 (/tmp/hy4_sm4/study.py, outside the repo, 9 s per variant). The table is in the test
  docstring. 13 of 17 q_a mutations pass the 0.98 out gate, including the missing K reduce (0.99909), a norm per K
  partial (0.99517) and swapped norm-weight halves. Only no norm weight (0.9785), 1 + w (0.9770), swapped SP row
  halves (0.9698) and the zero stub (0.9315) fail it. The q_resid checks catch all of them except eps; the x 0.01
  check catches eps (1e-5 0.168, 0 0.026, 2e-6 0.024).

Decisions
- Kept swap 03's limits for gates, attn_x, attn_norm, topk (0.995), attn_out, h_mid (0.005 / 0.02) and router (0.99).
  x 1.02 on q_a gives router 0.9921, so the router check stays a backstop, not the q_a check.
- Out worst (row, stream) is recorded, not asserted: the fp32 reference already gives 0.056 (near-tie expert flips).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999997, q_resid 0.00180, topk 0.99909, router 0.99866, out rel 0.00242).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device TtHcGates + TtHcPre + attn_norm + TtQa): PASS. pcc_swap_out 0.999996. q_resid 0.00221, ratio
  [0.99892, 1.00052], worst row 0.0028. vs CPU 0.00177. x0.01 0.00175. attn_norm 0.00298. topk 0.99907. attn_out
  0.00192. h_mid 0.00222 / 0.0037. router 0.99774. out rel 0.00289.

Gotchas
- As before, the first block of printed metrics (pcc 0 / rel 1.0) comes from the precompile collect pass. Only the
  second block is the real run.
- The test file is untracked (rendered by the orchestrator), so `git diff` does not show it.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_04_q_a.py

## C.moe_full.indexer test (attempt 1)

What was done
- Replaced the rendered 22-line test with test_c_dense_full_indexer.py's checks at LAYER = 1: gated pcc_indexer_L01
  = mean per-row set overlap (`topk_overlap`, pads -1 / 0xFFFFFFFF dropped), threshold 0.99; asserted extras: integer
  S x 2048 output, causal, no repeats, exactly min(pos + 1, 2048) valid per row, worst row >= 0.97, own position
  selected, and a second call on chunk 0 that must return exactly [0, pos] per row.
- Re-ran the layer-0 CPU mutation study on the layer-1 golden (/tmp/hy4_idx1/study.py, check.py, outside the repo,
  1 s per variant). Table in the test docstring.

Gotchas
- Layer-1 margins are a bit thinner than layer 0: fp32 reference 0.99908 (worst row 0.9971), bf16 scores 0.99568
  (0.9863), bfp8 + bf16 scores 0.99395 (0.9854). Positional match of the fp32 reference is 0.761.
- As at layer 0, t <= s + 1, own key dropped, top-2047 and a padded last row pass the 0.99 overlap; the structural
  checks catch each (verified with the test's helpers). RoPE positions + 1 scores 0.98512 (fails the gate).
- The first "FAIL pcc_indexer_L01" line in each run is the precompile collect pass; the real run is the second block.
- The device gate already runs TtHy4Indexer at layer 1 (device_component builds _INDEXER_STEPS for any layer), even
  though DEVICE_STEPS["moe_full"] is still empty.

Results
- BRINGUP_IMPL=reference: PASS (overlap 0.999083, worst row 0.99707, self 1.0; chunk 0 exact).
- BRINGUP_IMPL=stub: FAIL (overlap 0.000488).
- Gate (device, existing TtHy4Indexer): PASS, overlap 0.995925, worst row 0.98877, 5 rows < 0.99, self 1.0, chunk 0
  exact (layer 0 device: 0.99708).

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_indexer.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_indexer.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_indexer.py

## S.moe_full.05 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_full layer 1, attn_hc + attn_hc_pre + attn_norm + q_a + indexer on
  device, rest CPU). It is swap 04 (moe_full) with its topk overlap check (>= 0.995) replaced by the dense_full swap 05
  topk checks: normalized integer output (pads -1 / 0xFFFFFFFF), overlap >= 0.99 and worst row >= 0.97 vs golden and
  vs the CPU indexer on the device attn_norm / q_resid, causal, no repeats, exactly min(pos + 1, 2048) valid per row,
  own position selected, and the indexer module on golden chunk 0 (must be exactly [0, pos] per row). The gated
  pcc_swap_out (0.98), the trail and every other swap 04 check (gates, attn_x, attn_norm, q_resid, attn_out, h_mid
  0.005 / 0.02, router 0.99, out rel 0.01) are unchanged.
- CPU mutation study at layer 1 (/tmp/hy4_sm5/study.py, the dense study at L = 1 plus router and per-stream columns,
  outside the repo, ~12 s per variant). Table in the test docstring.

Decisions
- Overlap limit 0.99 (the component gate), not swap 04's 0.995: the layer-1 device indexer scores 0.9959.
- Router 0.99 kept: the device run gives 0.99799. It also fails several indexer bugs (RoPE from 0 0.9870, t <= s + 1
  0.9873, non-causal 0.9688), but the topk checks already catch those.

Gotchas
- 14 of 21 indexer bugs pass the 0.98 out gate at layer 1 (as at layer 0), including non-causal selection, no k_norm,
  RoPE from 0 and prefix keys zero (0.98498). The topk checks catch all of them.
- The trail's pcc_swap_topk is positional match (0.758 for the reference). Ignore it.
- The first block of printed metrics (pcc 0 / rel 1.0, router 0.37) comes from the precompile collect pass.

Results
- BRINGUP_IMPL=reference: PASS (topk 0.99909 / worst 0.99658, chunk 0 exact, router 0.99866, out rel 0.00242).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device TtHcGates + TtHcPre + attn_norm + TtQa + TtHy4Indexer): PASS. pcc_swap_out 0.999995. topk vs golden
  0.99592 / worst 0.98926, vs CPU 0.99602 / 0.98828, self 1.0, chunk 0 exact. attn_out 0.00206 / 0.0046. h_mid
  0.00225 / 0.0039. router 0.99799. out rel 0.00316.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_05_indexer.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_05_indexer.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_05_indexer.py

## C.moe_full.attention test (attempt 1)

What was done
- Replaced the rendered 22-line test with test_c_dense_full_attention.py's four checks at LAYER = 1: gated
  pcc_attention_L01 (PCC, 0.99) on golden s4096 chunk 1, plus asserted checks on (1) golden chunk 1, (2) golden
  chunk 0 (start 0, -1 pads), (3) a probe topk (64 random causal positions per row, unsorted, seed 0) vs the CPU
  step, and (4) attn_norm x 1e-3 vs the CPU step (kv_a_layernorm eps).
- Re-ran the layer-0 CPU mutation study on the layer-1 golden (/tmp/hy4_c_attn1/mut.py c1 / c0 / probe / syn / pess,
  outside the repo). Tables are in the test docstring.

Decisions
- Limits loosened from layer 0's (0.01 / [0.99, 1.01] / 0.02) to rel 0.015, ratio [0.985, 1.015], worst row 0.04.
  The layer-1 device module scores 0.0066 / [0.9927, 1.0066] / 0.0165; the bf16 CPU estimate is 0.0045 / 0.013.
  Every mutation still fails at least one check. The closest calls are RoPE + 1 (golden rel 0.0103, caught by worst
  row 0.31), x 1.02 (caught by the ratio, 1.02) and dense causal attention (rel 0.0216).

Gotchas
- Layer 1 is noisier on the device than layer 0 (worst row 0.0165 vs 0.0052), but separates the bugs better (RoPE
  from 0 fails PCC here).
- device_component already builds TtHy4Attention for any layer, so the gate runs on the device even though
  DEVICE_STEPS["moe_full"] does not list attention yet.
- The first "FAIL pcc_attention_L01: pcc=0.000000" line and UP_FRONT_COLLECT_RESULT status=failed come from the
  precompile collect pass. Ignore them.

Results
- BRINGUP_IMPL=reference: PASS (golden rel 0.00173, worst row 0.0027; chunk 0 0.00171; probe / scaled exact).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): PASS. pcc_attention_L01 0.999978. golden 0.00663 / [0.99418, 1.00622] / 0.0155; chunk0 0.00672 /
  [0.99270, 1.00661] / 0.0158; probe 0.00670 / 0.0165; scaled 0.00385 / 0.0061.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attention.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attention.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attention.py

## S.moe_full.06 test (attempt 1)

What was done
- Replaced the rendered 27-line swap test (moe_full layer 1, attn_hc .. indexer + attention on device, rest CPU). It
  is swap 05 (moe_full) plus the attention checks from test_c_moe_full_attention.py at the layer-1 limits (rel 0.015,
  row norm ratio [0.985, 1.015], worst row 0.04): attn_out (a) vs the golden, (b) vs the CPU attention on the device
  attn_norm / q_resid / topk, then the module again vs the CPU step on (c) golden chunk 0, (d) a probe topk (64 random
  causal positions per row, seed 0), (e) the device attn_norm x 1e-3 (kv_a_layernorm eps). These replace swap 05's
  attn_out check (0.01 / 0.05). The gated pcc_swap_out (0.98), the trail, and every other swap 05 check (gates,
  attn_x, attn_norm, q_resid, topk, router 0.99, out rel 0.01) are unchanged.
- CPU swap mutation study at layer 1 (/tmp/hy4_sm6/study.py, reuses /tmp/hy4_c_attn1/mut.py's attention mutations,
  outside the repo, ~11 s per variant). The table is in the test docstring.

Decisions
- h_mid loosened from 0.005 / 0.02 to 0.007 / 0.025. The device scores 0.0040 / 0.0112: the device attention's own
  error (0.0067 vs the CPU step) now reaches h_mid. The attn_out checks carry the attention's detection, and the gates
  are checked per column.
- Router kept at 0.99 (device 0.99396, bf16 estimate 0.9963). Out rel kept at 0.01 (device 0.00576).

Gotchas
- 18 of 32 attention bugs pass the 0.98 out gate at layer 1, including scale 576^-0.5, sink raw, dense causal and
  RoPE from 0; no sink scores 0.97994. The attn_out checks catch every bug except x 1.01, which is at the tolerance.
- The trail's pcc_swap_topk is positional match (0.0003 on the device, unsorted indices). Ignore it.

Results
- BRINGUP_IMPL=reference: PASS (attn_out vs golden 0.00177 / 0.0037, vs CPU / chunk 0 / probe / scaled exact, h_mid
  0.0021, router 0.99866, out rel 0.00242).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device, attention = TtHy4Attention): PASS. pcc_swap_out 0.999984. attn_out vs golden 0.00713 /
  [0.99376, 1.00483] / 0.0165, vs CPU 0.00673 / 0.0164, chunk 0 0.00650 / 0.0157, probe 0.00672 / 0.0139, scaled
  0.00417 / 0.0062. h_mid 0.00405 / 0.0112. router 0.99396. out rel 0.00576.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_06_attention.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_06_attention.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_06_attention.py

## C.moe_full.attn_residual test (attempt 1)

What was done
- Replaced the rendered 22-line test with layer 0's attn_residual test at layer 1, with these changes. Gated PCC
  (0.99) kept. Not a CPU bridge, size and finite kept. rel L2 <= 0.01 kept. Per-token per-stream norm ratio tightened
  to [0.99, 1.01]. The addend checks (delta_j = out_j - in_j vs t_j = post_j * attn_out) are now rounding-aware:
  |coef_j - 1| <= 0.01 + 2 r_j / ||t_j||, ||delta_j - t_j|| <= 0.01 ||t_j|| + 2 r_j, per token <= 0.05 ||t|| + 2 r +
  1e-6. Here r is the bf16 rounding error of the exact fp32 result on the golden inputs.
- CPU mutation study on the layer-1 golden (/tmp/hy4_c_res1/study.py and bound.py, outside the repo). The table is in
  the test docstring.

Decisions
- Rounding-aware limits: at layer 1 the post gates of streams 0 / 1 are ~3e-4, so their addend (norm 0.29 / 0.47) is
  below the stream's bf16 resolution (norm ~70). Fixed limits would fail any bf16-output module; with this bound a
  bf16 output uses about 0.49 of the allowance.
- Addend statistics are computed in float64. With fp32 sums over 12.6M elements, the exact reference scored
  coefficient 1.02-1.03.
- The limits are tighter than layer 0's (coefficient 0.01, rel 0.01 + budget, norm ratio 0.99-1.01). 1.02 x attn_out
  now fails the addend (excess 1.34), rel L2 (0.0111) and the ratio. 1.01 x passes (blind spot, at bf16 tolerance).

Gotchas
- Layer-1 streams differ (norms 70 / 65 / 71 / 115), so input- or output-stream swaps are now visible (PCC 0.989,
  rel 0.149).
- The first "FAIL pcc_attn_residual_L01: pcc=0.000000" line comes from the precompile collect pass. Ignore it.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00243, ratio [0.9980, 1.0022], addend exact).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device, TtHcPost via device_component): PASS. pcc_attn_residual_L01 0.999997. rel 0.00243, ratio [0.9980,
  1.0022], addend coefficient 1.0 and excess 0 on every stream (bit-identical to the fp32 CPU step).

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_residual.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_residual.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_attn_residual.py

## S.moe_full.07 test (attempt 1)

What was done
- Replaced the rendered 28-line swap test (moe_full layer 1, attn_hc .. attention + attn_residual on device, rest
  CPU) with swap 06 (test_swap_moe_full_06_attention.py) plus attn_residual checks. The gated pcc_swap_out (0.98), the
  trail, and every swap-06 check at its limits stay as they were (gates, attn_x, attn_norm, q_resid, topk, attn_out
  five ways, h_mid vs golden 0.007 / 0.025, router 0.99, out rel 0.01).
  New h_mid checks: per-token per-stream norm ratio vs golden [0.98, 1.02]; vs the CPU attn_residual on the same
  device inputs rel <= 5e-4 / worst (row, stream) <= 1e-3; the component's rounding-aware addend checks (float64);
  and the module on rotated post gates (row mod 4) vs the CPU step at the same tight limits.
- CPU mutation study at layer 1 (/tmp/hy4_sm7/study.py, outside the repo, ~10 s per variant). The table is in the
  test docstring.

Decisions
- Tight vs-CPU limits, as layer 0's swap 07: the device TtHcPost is fp32 and bit-identical to the CPU step, so a bf16
  output (0.0016) or a 1 % scale (0.0054) fails there.
- Rotated post gates instead of layer 0's distinct-stream probe. The layer-1 streams are already distinct, but the
  post gates of streams 0 / 1 are ~3e-4. Dropping the addend on stream 0 or 1, or swapping post columns 0 / 1, passes
  every golden-side check; with rotated gates those bugs score rel >= 0.28.
- Stream norm ratio kept at layer 0's [0.98, 1.02], not the component's [0.99, 1.01]: device attention error reaches
  stream 3 (post ~0.23). The device scores [0.9972, 1.0038].

Gotchas
- 12 of 24 residual bugs pass the 0.98 out gate, including input streams 0 / 1 swapped (0.9966). Every bug fails the
  vs-CPU check.
- The trail's pcc_swap_topk is positional match (0.0003 on the device). Ignore it.

Results
- BRINGUP_IMPL=reference: PASS (every vs-CPU check exact, h_mid 0.0021, out rel 0.00242).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device): PASS. pcc_swap_out 0.999984; h_mid 0.00405 / 0.0112, ratio [0.99722, 1.00379]; h_mid vs CPU 0,
  addend exact, rotated 0; router 0.99396; out rel 0.00576.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_07_attn_residual.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_07_attn_residual.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_07_attn_residual.py

## C.moe_full.ffn_hc test (attempt 1)

What was done
- Replaced the rendered 22-line test for ffn_hc at layer 1 (iHC gates [S, 8] from h_mid, hc_mlp_layer of layer 1,
  s4096 chunk 1, bf16 golden). Built it from the C.moe_full.attn_hc test. It keeps the gated PCC (0.99) and the
  CPU-bridge assert, and adds these checks vs the golden: element count, finite, rel L2 <= 0.01, per-column rel L2
  <= 0.01 (<= 0.02 on columns 0, 1, 6), post worst row <= 0.015. The pre gates are checked through the CPU
  ffn_hc_pre on the golden h_mid (ffn_x rel <= 0.005, worst row <= 0.02). The post gates are checked through the CPU
  ffn_residual with the golden mlp_out (out per stream <= 0.005, worst row <= 0.02).
- CPU mutation study in /tmp/hy4_ffnhc1/{keys,mut}.py (outside the repo). The table is in the test docstring.

Decisions
- Columns 0, 1 and 6 get a limit of 0.02, not 0.01. Their means (1.05e-5, 1.45e-6, 1.6e-6) are within ~10x of
  hc_eps. The device reaches 0.0072 / 0.0035 / 0.0075 on them, which leaves little margin under 0.01. The check is
  still needed at 0.02: gates 0 / 1 / 6 zeroed, base 0 / 1 swapped and a dropped hc_eps move ffn_x and out by < 1e-4
  and fail only this check.
- The out per-stream limit is 0.005, not attn_hc's 0.003. Out stream 3 carries post 7 x mlp_out at 1.7x the stream
  norm, so it tracks post column 7: device 0.0026, CPU reference 0.0013. post x 1.01 still fails it (0.008).
- Every mutation in the study fails at least one check, except bf16 rounding of the input, fn or output.

Results
- BRINGUP_IMPL=reference: PASS. PCC 0.999999, rel 0.00131, col rel <= 0.0028, post row 0.0040, ffn_x 0.0013 /
  0.0048, out stream <= 0.0013 / row 0.0033.
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device, the existing TtHcGates via `_HC_STEPS["ffn_hc"]`): PASS. PCC 0.999999, rel 0.00133, col rel
  [0.0072, 0.0035, 0.0012, 0.0018, 0.0056, 0.0066, 0.0075, 0.0032], post row 0.0093, ffn_x 0.00135 / 0.0052, out
  stream <= 0.0026 / row 0.0063.

Gotchas
- The tightest margins are post columns 4 / 5 (0.0056 / 0.0066 vs 0.01) and the post worst row (0.0093 vs 0.015).
  All three come from the device sigmoid on gates of ~1e-4. Watch them if a change touches TtHcGates.
- The first `FAIL pcc ... 0.000000` line comes from the precompile collect pass.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_hc.py

## S.moe_full.08 test (attempt 1)

What was done
- Replaced the rendered 28-line swap test (moe_full layer 1, attn_hc .. attn_residual + ffn_hc on device, rest CPU)
  with swap 07 (test_swap_moe_full_07_attn_residual.py) plus ffn_hc checks. The gated pcc_swap_out (0.98), the trail
  and every swap-07 check stay at their limits.
  New checks:
  - ffn_hc vs golden: rel <= 0.01, per column <= 0.015 (small columns 0 / 1 / 6 <= 0.03), post row <= 0.02. ffn_x
    vs golden: <= 0.01 / row 0.05.
  - ffn_hc vs the CPU ffn_hc on the same device h_mid: rel <= 0.005, per column <= 0.015 (0 / 1 / 6 <= 0.02), post
    row <= 0.015. ffn_x from the device gates vs from the CPU gates: <= 0.005 / row 0.02.
  - post gates through out: block out vs ffn_residual(h_mid, CPU gates, the block's own mlp_out), per stream <= 0.005,
    row <= 0.02.
  - block out vs the whole CPU tail from the device h_mid: rel <= 0.01, worst row recorded only.
- CPU mutation study /tmp/hy4_sm8/study.py (outside the repo, ~4 s per variant on the tail). The table is in the test
  docstring. 23 of 35 mutations pass the out gate; every real bug among them fails an added check.

Decisions
- Per-column checks (not layer 0's post-column-only check): at layer 1, gates 0 / 1 / 6 are near hc_eps. Zeroing them,
  swapping base 0 / 1 or dropping hc_eps moves out by <= 1e-4, and only the column check catches it.
- Post gates are checked through out with the block's own mlp_out, not through the full CPU tail. In the full tail,
  near-tie expert flips give a worst row of 0.055 even for a bf16 output, so the full tail is gated on rel only.
- vs-CPU big-column limit is 0.015, not 0.01. The device reads 0.0089 on post column 5 (sigmoid on ~1e-4 gates). The
  only study bug between 0.01 and 0.015 (post x 1.01, 0.0100) fails the post-through-out check (0.0079 > 0.005).
- No scaled-input eps probe: at layer 1's own scale (row RMS 0.006 .. 0.08), rms_norm_eps 1e-6 already fails the
  column check (0.40).

Gotchas
- Golden-side column 0 reads 0.0146 on the device (limit 0.03): the device h_mid error reaches the near-hc_eps gates.
- Router overlap 0.99377 vs 0.99 is inherited from swap 07 (0.99396). Watch it if an upstream step changes.
- In the golden chunk, rows 1023 / 1024 have identical gates, so a swap of that pair is a no-op (left out of the table).

Results
- BRINGUP_IMPL=reference: PASS (every vs-CPU check 0). BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999983. ffn_hc vs golden 0.00155 / worst column 0.0146 (col 0) / post row 0.0139.
  vs CPU 0.00036 / big column 0.0089 / small column 0.0105 / post row 0.0095. ffn_x vs CPU 0.00048. Post through
  out 0.0032 / 0.0054. Tail 0.0033. Router 0.99377. Out rel 0.00608.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_08_ffn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_08_ffn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_08_ffn_hc.py

## C.moe_full.ffn_hc_pre test (attempt 1)

What
- Replaced the rendered 22-line component test (moe_full layer 1, ffn_hc_pre) with the moe_full attn_hc_pre test's
  structure (same limits). It keeps the gated pcc_ffn_hc_pre_L01 (0.99) and the CPU-bridge assert, and adds: finite
  output and element count; vs golden rel L2 <= 0.005, row norm ratio in [0.994, 1.006], worst row <= 0.01; vs the
  CPU step on the same inputs rel <= 0.003, worst row <= 0.006; the module run again with each row's pre gates
  rotated by row mod 4, vs the CPU step: rel <= 0.004, worst row <= 0.01.
- CPU mutation study in /tmp/hy4_moe_ffnhcpre/study.py (outside the repo; /tmp/hcpre_l1.py on h_mid / ffn_hc /
  ffn_x). The tables are in the test docstring.

Decisions
- The rotated-gates run is required here, not optional: pre gates 0 / 1 are ~1e-5 / 1.5e-6 (hc_eps), so a dropped
  stream 0 or 1, or a 0 / 1 swap, is bit-identical to the reference on the golden. With rotated gates they score rel
  0.33 / 0.30 / 0.12.
- Golden limits sit above the golden's own rounding (fp32 CPU step rel 0.0026, ratio [0.9965, 1.0038], row 0.0054).
  pre x 1.005 fails the ratio (1.0088) and the rel limit (0.00555).

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999997, rel 0.00260, ratio [0.99645, 1.00382], row 0.00542).
- BRINGUP_IMPL=stub: FAIL (PCC 0.0).
- Gate (device): PASS already, with the existing tt/ihc.py:TtHcPre (hooks._HC_PRE_STEPS is block-type independent):
  PCC 0.999997, vs CPU rel 0.0 / row 0.0, rotated rel 0.0.

Gotchas
- Not caught: pre + 3e-4 on every gate (golden rel 0.0028). The gates are an input, so the module cannot make it.
- The first `FAIL pcc ... 0.000000` line comes from the precompile collect pass.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_hc_pre.py

## S.moe_full.09 test (attempt 1)

What
- Replaced the rendered 25-line swap test (moe_full layer 1, steps 1-9, last ffn_hc_pre) with the reviewed swap 08
  test (test_swap_moe_full_08_ffn_hc.py): every check kept at its limits. Added the ffn_hc_pre checks from the layer-0
  swap 09 / layer-1 component test: device ffn_x vs the CPU ffn_hc_pre on the same device h_mid and gates (rel <= 0.003,
  worst row <= 0.006), and the module run again with each row's pre gates rotated by row mod 4 vs the CPU step
  (0.004 / 0.01). ffn_x vs golden (0.01 / 0.05) and out vs the CPU tail (rel 0.01) now see the device step too.
- CPU mutation study in /tmp/hy4_sm9/study.py (outside the repo; log study.log). Table in the test docstring.

Decisions
- Kept the component's vs-CPU limits (0.003 / 0.006), not fp32-exact ones. The device TtHcPre is bit-identical, but a
  bf16-output module (0.0017) is valid, and the one bug left between (pre + 3e-4, 0.0010) is below gate 2's bf16
  rounding. The gates are the step's input, so the module cannot make that error by itself.
- No scaled-input probe: ffn_hc_pre has no eps.

Results
- BRINGUP_IMPL=reference: PASS (every vs-CPU check 0). BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999983. ffn_hc_pre vs CPU rel 0 / row 0, rotated 0 / 0. ffn_x vs golden
  0.0043 / 0.0128. Tail 0.0033. Router 0.99377. Out rel 0.00608 (same as swap 08).

Gotchas
- pre x 1.02 moves the tail by only 0.0011: ffn_norm removes a row scale of ffn_x. Only the vs-CPU check sees it.
- Router overlap 0.99377 vs 0.99 is inherited from swap 07 / 08.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_09_ffn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_09_ffn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_09_ffn_hc_pre.py

## C.moe_full.ffn_norm test (attempt 1)

What
- Replaced the rendered 22-line component test (moe_full layer 1, ffn_norm) with the reviewed dense_full ffn_norm
  test at LAYER = 1, with the same limits. It keeps the gated pcc_ffn_norm_L01 (0.99) and asserts that the module is
  not a CPU bridge. Against the golden it checks: finite output, element count, rel L2 <= 0.008, row norm ratio in
  [0.993, 1.007], worst row <= 0.015. It also runs the module on the golden input x 30 (bf16) vs the CPU step:
  rel <= 0.006, ratio in [0.993, 1.007], worst row <= 0.015.
- Added the layer-1 attn_norm eps probe: the module on golden x 0.1 (bf16) vs the CPU step, rel <= 0.01, worst row
  <= 0.02. The layer-1 mutation tables are in the docstring.
- CPU mutation study in /tmp/hy4_ffnnorm1/study{,2}.py (outside the repo; log study.log). These are the layer-0
  scripts with the layer index changed.

Decisions
- Layer-1 ffn_x is not like layer 0. Row rms is in [0.0040, 0.072] (layer 0: from 0.00083, 76% of rows eps-dominated),
  and the smallest mean(x^2) is 1.6x eps. So both eps (1.2e-5 gives rel 0.013) and the RMS reduction (half columns:
  rel 0.011, row 0.040) already show on the golden. I kept both scaled probes anyway, so the module is checked in the
  eps-dominated and the pure-RMS regimes independently of this chunk's row-rms range.
- 10 of the mutations pass the 0.99 PCC gate, including 1 + w (0.9966), no weight and TP-permuted w halves (0.992).
  Each one fails a golden check. The pessimistic bf16 estimate (0.0037 / [0.9958, 1.0036] / 0.0054) leaves about 2x margin.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, rel 0.00234, ratio [0.99987, 1.00016], row 0.0025; both probes 0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): already PASS, because `hooks._GATHERED_NORM_STEPS` (tt/norm.py:TtGatheredRmsNorm) does not depend
  on the block type. pcc 0.999996, rel 0.00283, ratio [0.99938, 1.00084], row 0.00307. x0.1: rel 0.00168, row 0.00192.
  x30: rel 0.00169, ratio [0.99942, 1.00084], row 0.00189.

Gotchas
- The first "FAIL pcc_ffn_norm_L01: pcc=0.0" line comes from the precompile collect pass.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_norm.py

## S.moe_full.10 test (attempt 1)

What
- Replaced the rendered 31-line swap test (moe_full layer 1, steps 1-10, last ffn_norm) with the reviewed swap 09
  test (test_swap_moe_full_09_ffn_hc_pre.py), keeping every check at its limits. Added the ffn_norm checks from the
  layer-0 swap 10 and the layer-1 component test. vs golden: rel <= 0.01, ratio [0.99, 1.01], row <= 0.03. vs the
  CPU ffn_norm on the same device ffn_x: 0.008 / [0.993, 1.007] / 0.015. The module on that ffn_x x 0.1 vs the CPU step:
  rel <= 0.01, row <= 0.02, ratio not gated (the component's eps probe). The module on ffn_x x 30 vs the CPU step:
  0.006 / [0.993, 1.007] / 0.015.
- CPU mutation study in /tmp/hy4_sm10/study.py (outside the repo; log study.log). Table in the test docstring.

Decisions
- Kept the component's limits. 16 of 25 mutations pass the 0.98 out gate. At layer 1 the MoE output is large next to
  the residual (x 1.01 moves out by 0.018), so the swap-09 checks on out (rel vs golden 0.01, vs the CPU tail 0.01)
  already catch most of them. LayerNorm-instead-of-RMS (out 0.0075) and eps 1.2e-5 (out 0.0057) get past those, but
  fail the ffn_norm worst-row check (0.038) or the ratio check (0.963), and the eps probe (0.050).
- Added the x0.1 eps probe, which layer 0's swap 10 does not have, because layer-1 ffn_x is not eps-dominated.

Results
- BRINGUP_IMPL=reference: PASS (every vs-CPU ffn_norm check 0). BRINGUP_IMPL=stub: FAIL.
- Gate (device, tt/norm.py:TtGatheredRmsNorm; already on device through hooks._GATHERED_NORM_STEPS): PASS, run twice.
  pcc_swap_out 0.999984. ffn_norm vs golden 0.00572 [0.99866, 1.00066], row 0.0131. vs CPU 0.00174, row 0.0019.
  x0.1: 0.00168. x30: 0.00169. Tail 0.0043. Router 0.99347. Out rel 0.00599.

Gotchas
- Router overlap is now 0.99347 against the 0.99 limit, inherited from swaps 07-09 (a margin of 0.0035).

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_10_ffn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_10_ffn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_10_ffn_norm.py

## C.moe_full.router test (attempt 1)

What
- Replaced the rendered 22-line router test (moe_full layer 1) with a reviewed test based on
  mimo_v2_6_d_p/tests/bringup/test_c_full_moe_router.py. It keeps the gated pcc_router_L01 (0.99) and asserts that
  the module is not a CPU bridge. It also checks: finite output, element count, exactly 8 nonzeros per row,
  non-negative weights, selection overlap >= 0.995 vs golden and >= 0.996 vs the CPU step on the same input,
  matched-row weight rel L2 <= 0.005 vs golden and <= 0.004 vs the CPU step, and row sum / 2.827 within 0.004 of 1.
- CPU mutation study in /tmp/hy4_router1/{study,mut,mut2}.py (outside the repo; log mut.log). The table is in the test
  docstring.

Decisions
- The route scale is 2.827 (HF routed_scaling_factor, norm_topk_prob), so the row-sum check divides by it. MiMo's
  "sum within 0.01 of 1" check would fail the correct router. Dropping the scale passes PCC (0.99934).
- The layer-1 bias is small (-0.097..0.032), so a bf16 bias is harmless (overlap 0.99799), unlike MiMo. The overlap
  limit of 0.995 is set to catch bf16 logits (0.99384), bf16 sigmoid (0.99341), bf16 choice keys (0.99030) and all-bf16
  (0.98773). The reference scores 0.99811. The input is bf16 and the gate weight is stored bf16, so an fp32 HiFi4
  device matmul is exact on its operands (a TF32 cut of x or W changes nothing), and the device should land near 0.998.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999352, overlap 0.99811 / vs CPU 1.0, matched rel 0.00173 / 0, row sums 1.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): FAIL, as expected: hooks.device_component raises NotImplementedError (no device router yet). The
  implement step must add it (components.yaml: mimo_v2_6_d_p_2x2 TtRouter fp32 path, route_scale 2.827).

Gotchas
- The golden router comes from fp32 ffn_norm. The test feeds the bf16-dumped ffn_norm, so even the CPU reference loses
  0.0019 of overlap vs the golden. The vs-CPU checks see the device's own error without that loss.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_router.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_router.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_router.py

## C.moe_full.router implement (attempt 1)

What
- New `tt/router.py:TtHy4Router`, the fp32 path of mimo_v2_6_d_p_2x2 TtRouter. The gate weight [6144, 256] and the
  correction bias [1, 256] are fp32 and replicated. The steps: fp32 x -> ttnn.linear (HiFi4, fp32 acc, fp32 out) ->
  sigmoid -> add bias (SFPU) -> ttnn.topk 8 on fp32 keys -> gather the unbiased sigmoids -> sum ->
  div by (sum / 2.827). It runs on each row's S/2 tokens (the ffn_norm layout: row-split over axis 0, replicated over
  axis 1). No collective, no host work, no per-call constants.
- hooks.py: added `_ROUTER_STEPS`, `_router_module` and `_router_host_fn`, and routed them in `_device_step_fn` and
  `device_component`. "router" added to `DEVICE_STEPS["moe_full"]` (hybrid device_model).

Decisions
- There is no dense scatter on the device, unlike MiMo. The module returns (idx, wts) [1, 1, S/2, 8] for the future
  device experts. The host fn builds the dense [S, 256] fp32 matrix at the harness boundary (components.yaml notes).
  This also avoids the bf16-only ttnn.scatter.
- `norm_topk_prob` is asserted True, since the module always renormalizes. route_scale = cfg.routed_scaling_factor.

Results
- Gate: PASS. pcc_router_L01 0.999373. nnz 8 per row. vs golden: overlap 0.99817, matched rel 0.00173. vs the CPU step:
  overlap 0.99982 (2045/2048 rows), rel 0.00006. Row sums / 2.827 in [1.00000, 1.00000].

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_router.py

## S.moe_full.11 test (attempt 1)

What
- Replaced the rendered 32-line swap-11 test (moe_full layer 1, steps 1-11 on device, last: router) with swap 10's
  reviewed test plus router checks. Every swap-10 check is kept at its limits.
- New router checks on the swapped step: exactly 8 nonzeros per row, non-negative, row sum / 2.827 within 0.004;
  vs golden overlap >= 0.99 (kept) and matched-row rel L2 <= 0.01; vs the CPU router on the same device ffn_norm
  overlap >= 0.996, worst row overlap >= 0.75, matched-row rel L2 <= 0.004.
- CPU mutation study in /tmp/hy4_sm11/{study,rows}.py (outside the repo; logs study.log, rows.log). Table in the test
  docstring.

Decisions
- 20 of 26 router mutations pass the 0.98 out gate. Swap 10's out checks miss 9 of them: bf16 logits / sigmoid / choice
  keys / all-bf16, weights from the biased choice, weights x 1.005, logits x 1.01, rows 1023 / 1024 swapped.
  The vs-CPU limits of the component test catch all of them except the row swap: its mean overlap vs CPU is 0.99902.
  So a worst-row overlap check was added. A precision flip costs 1 of 8, and two adjacent rows share at most 4 of 8
  experts (measured). 0.75 splits them.
- The vs-golden overlap stays 0.99, not the component's 0.995: the device ffn_norm alone drops the CPU router to
  0.99347.
- No scaled-input probe. At x 3 the precision mutations separate no better than at x 1.

Results
- BRINGUP_IMPL=reference: PASS (router vs CPU 1.0 / 0). BRINGUP_IMPL=stub: FAIL.
- Gate (device, tt/router.py:TtHy4Router): PASS. pcc_swap_out 0.999984. Router vs golden overlap 0.99316, matched rel
  0.00233. vs CPU overlap 0.99933, worst row 0.875, rel 0.000062. Row sums [1.00000, 1.00000]. Tail 0.0045.
  Out rel 0.00594.

Gotchas
- Router overlap vs golden is 0.99316 against 0.99, a margin of 0.003. The device router is slightly below the CPU
  router on the same device ffn_norm (0.99347). Upstream device error causes most of the loss.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_11_router.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_11_router.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_11_router.py

## C.moe_full.experts test (attempt 1)

What
- Replaced the rendered 23-line experts test (layer 1) with a reviewed test. It keeps the gated PCC >= 0.99 and adds a
  no-CPU-bridge assert. Vs the golden: finite, element count, rel L2 <= 0.015, per-token norm ratio in [0.98, 1.02],
  worst token rel L2 <= 0.03, and the float64 global coefficient <got, want> / <want, want> within 0.004 of 1.
- A second module call on x * 2 (bf16-exact) with the golden routing, checked against the CPU experts on the same
  input at the same limits. The clamp at 10 barely fires on the golden (gate max 10.49).
- CPU mutation study in /tmp/hy4_exp1 (outside the repo): prep.py dumps the golden and the layer-1 expert weights,
  study.py / study2.py run the mutations (logs study.log, study2.log). The table is in the test docstring. Note that
  the PCC printed in study.log is fp32 and slightly > 1. study2.log uses float64.

Decisions
- Limits sit about 2x above the device estimate: bfp8 weights (blocks along the output dim) + bf16 h/out give 0.0072 /
  [0.995, 1.005] / 0.0093 / coef 0.99988. Adding bfp8 x and h gives 0.0104 / [0.993, 1.008] / 0.0153, which also passes.
- Worst row 0.03: dropping any single expert gives >= 0.046 (expert 98), so every single-expert drop is caught.
- Coefficient check: x 1.01 passes rel, ratio and worst row, and fails the coefficient (1.0099). x 1.005 fails it too
  (1.0049).
- Known gaps (in the docstring): x 1.003, and one token's smallest pair (half of the tokens are < 0.03). min(silu(g), 10)
  and a gate clamped on both sides are numerically harmless.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, golden rel 0.00231 / [0.9962, 1.0037] / 0.0043 / coef 0.99992; probe 0).
- BRINGUP_IMPL=stub: FAIL (pcc 0).
- Gate (device): FAIL with "no device module for experts yet". This is expected: the implement step comes next.

For implement
- Golden facts: tokens per expert 2..505 (hottest 187), so a capacity below 505 drops work (256 fails). Pairs per chip
  for experts 128c + 64r: 3986 / 4298 / 4429 / 3671. There are no outlier channels in x.
- The probe calls the module twice with the same routing, so keep no per-call state.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_experts.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_experts.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_experts.py

## C.moe_full.experts implement (attempt 1)

What
- tt/experts.py:TtHy4Experts, adapted from mimo_v2_6_d_p_2x2/tt/experts.py:TtExperts. It is the DeepSeek 2D EP
  pipeline: masked_bincount -> ttnn.bringup.offset_cumsum (axis 0) -> ttnn.bringup.dispatch (group size 2, axis 0,
  Topology.Linear on the FABRIC_2D mesh) -> ttnn.bringup.unified_routed_expert_moe (ClampedSiluGlu, limit 10 baked,
  high_precision=True, HiFi4 + fp32 dest, bf16 x, bfp8 weights) -> ttnn.bringup.combine -> post_combine_reduce
  (the dispatch table masks the other column's experts) -> typecast fp32 -> ttnn.reduce_scatter(dim 3, axis 1).
  Output: [1, 1, S/2, 3072] fp32 per chip, the residual's column split (same as tt/mlp.py).
- Input: x = ffn_norm [1, 1, S/2, H] (rows over axis 0, replicated over axis 1, the TtGatheredRmsNorm layout), plus the
  router's (idx, wts) [1, 1, S/2, 8] in the same placement. No host work, no per-call constants. The dispatch / combine
  modules are built once per chunk length.
- LazyExpertWeights reads gate_up_proj / down_proj one expert at a time (reference/weights.py:ExpertSlab). It splits
  gate = rows 0-2047 and up = rows 2048-4095. The bfp8 cache is in generated/hy4_preview_d_p/tt_cache/experts
  (layer_<i>.experts.BFLOAT8_B.*). Per-expert cap = the longest ladder chunk (8192, hooks._max_chunk).
- hooks.py: `_EXPERTS_STEPS`, `_max_chunk`, `_experts_module`, `_experts_host_fn`. The host fn is the harness boundary:
  it turns the dense [S, 256] routing back into topk (idx uint16, wts fp32) on the host and reads back col_split. Added
  "experts" to `DEVICE_STEPS["moe_full"]`.
- `HY4_EXPERTS_MODE=loop` selects the fallback: per local expert extract -> ttnn.linear gate / up in fp32 ->
  ttnn.clamp (gate max 10, up +-10) -> multiply with SILU -> linear down -> insert.
- ttnn/ttnn/bringup/INDEX.md: added hy4_preview_d_p to "Used by" for unified_routed_expert_ffn, dispatch, combine and
  offset_cumsum. No fork was changed.

Decisions
- ffn_norm per-channel max on the golden (s4096 chunk 1), layers 1-5: max |x| 0.85 / 0.77 / 0.41 / 0.54 / 0.68, so there
  are no outlier channels. x stays bf16 anyway (high_precision keeps it bf16).
- The axis-1 reduce_scatter runs in fp32: the partial is typecast before the CCL (`out_dtype`, default float32).
- Routing weights go to post_combine_reduce as bf16, as in MiMo 2x2.

Results (gate, default unified mode)
- PASS: pcc_experts_L01 0.999968. Golden: rel L2 0.00802, row norm ratio [0.99503, 1.00532], worst row 0.0103, coef
  1.00010. x*2 vs CPU: rel 0.00751, [0.99774, 1.00327], 0.00965, coef 1.00024. Call time 12.5 s, which includes both
  module calls and the CPU probe.
- Loop mode (HY4_EXPERTS_MODE=loop): PASS, pcc 0.999969, rel 0.00791, [0.99532, 1.00509], worst row 0.0099.
- First run: the precompile collect pass built the bfp8 cache (about 70 s). Later runs load it in about 2 s.

Gotchas
- The first "FAIL pcc_experts_L01: pcc=0.000000" line is the precompile collect pass. Ignore it.
- In routed_half, idx2 / ind / scores / w5 are reshape views of the caller's buffers, so they are not deallocated.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_experts.py
    HY4_EXPERTS_MODE=loop PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_experts.py

## S.moe_full.12 test (attempt 1)

What
- Replaced the rendered 32-line swap-12 test (moe_full layer 1, steps 1-12 on device, last: experts) with swap 11's
  reviewed test plus experts checks. Every swap-11 check is kept at its limits.
- New experts checks on the swapped step:
  - vs the CPU experts on the same device ffn_norm and device routing, at the component limits: rel 0.015, ratio
    [0.98, 1.02], worst row 0.03, coef 0.004.
  - The module again on the device ffn_norm x 2 with the device routing, vs the CPU experts, same limits.
  - vs golden: rel 0.03 and coef 0.004. On the rows whose top-8 set equals the golden's: ratio [0.97, 1.03] and
    worst row 0.04.
- CPU mutation study in /tmp/hy4_sm12/study.py (outside the repo; log study.log). It uses the device run's
  seen tensors, dumped once to /tmp/hy4_sm12/seen.pt by a temporary line that has since been removed. The table is in
  the test docstring.

Decisions
- 19 of 26 experts mutations pass the 0.98 out gate. Swap 11's out and tail checks miss x 1.005, x 1.01, the three
  clamp bugs, and dropping a small expert. The vs-CPU checks catch all of them.
- The vs-golden per-token limits are wider than the component's, and apply only to rows routed as in the golden. The
  exact CPU experts on the device inputs already score [0.986, 1.007] / 0.021 there, and the device scores
  [0.9827, 1.0079] / 0.0247.

Results
- BRINGUP_IMPL=reference: PASS (experts vs CPU 0; pcc_swap_out 0.999997). BRINGUP_IMPL=stub: FAIL.
- Gate (device, tt/experts.py unified mode): PASS. pcc_swap_out 0.999973. Experts vs CPU: rel 0.00785, ratio
  [0.99351, 1.00580], worst row 0.0110, coef 1.00016. x 2: rel 0.00768, worst row 0.0101. vs golden: rel 0.0139,
  1936 matched rows, ratio [0.98270, 1.00793], worst row 0.0247, coef 1.00043. Tail 0.0065, out rel 0.00757.

Gotchas
- Out rel vs golden is 0.00757 against a limit of 0.01, and the tail is 0.0065 against 0.01. Device experts add about
  0.0016 to each. There is less margin left for later steps (shared_expert, moe_combine, ffn_residual).
- The precompile collect pass prints experts vs CPU with coef nan and zeros. Ignore it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_12_experts.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_12_experts.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_12_experts.py

## C.moe_full.shared_expert test (attempt 1)

What was done
- Replaced the rendered 20-line test with the reviewed C.dense_full.mlp test, set for shared_expert (layer 1,
  mlp.shared_experts, SwiGLU 2048, unclamped). Kept the gated pcc_shared_expert_L01 (0.99, PCC on the bf16 golden).
  Added, vs the golden: finite, element count, rel L2 <= 0.008, row norm ratio in [0.99, 1.01], worst row <= 0.015.
  Also asserts the module is not a CPU bridge. Added a second run on ffn_norm x 2 (bf16) vs the CPU shared_expert on
  the same input: rel <= 0.006, ratio in [0.99, 1.01], worst row <= 0.012.
- CPU mutation study in /tmp/hy4_se/study{,2}.py (outside the repo). The tables are in the test docstring.

Decisions and why
- These pass PCC 0.99 on the golden: gelu_tanh (0.9952), HiFi2-like truncation (rel 0.027), x 1.01 / 1.03, last row
  or last 32 rows zeroed, input x 0.5. The added golden checks catch all of them.
- The ratio band is [0.99, 1.01], not the dense mlp's [0.993, 1.007]. Here bf16 gate / up / h already scores
  [0.9946, 1.0032] / worst row 0.0073, because the layer-1 input is wider. x 1.01 is still caught (ratio max 1.0105,
  rel 0.0102).
- Scale x 2, not x 30 as in the dense mlp. On the golden, gate reaches 8.07 and up 9.67, so a clamp at 10 does not
  show. At x 2 the clamp scores rel 0.085 / worst row 0.42, against bf16 noise of 0.0032 / 0.0056. It also keeps the
  outputs moderate (row norm up to about 1e3).
- bfp8 weights would score rel 0.0074 on the golden and pass. The plan says bf16 weights anyway.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999998, rel 0.00183, ratio [0.99950, 1.00047], worst row 0.00242; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): FAIL, NotImplementedError "no device module for shared_expert yet". This is expected before implement.

Gotcha for implement
- The module is called twice: on the golden, then on x 2. Its input is ffn_norm [S, 6144] at the host boundary. Its
  output is shared_out [S, 6144] (after the reduce_scatter, [S/2, 3072] per chip). tt/mlp.py:TtDenseMLP with
  intermediate 2048 (1024 per chip) and the mlp harness boundary should fit as is.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_shared_expert.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_shared_expert.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_shared_expert.py

## C.moe_full.shared_expert implement (attempt 1)

- Shared expert = `tt/mlp.py:TtDenseMLP` on `model.layers.<i>.mlp.shared_experts.{gate,up,down}_proj.weight`
  (intermediate 2048 -> 1024 per chip column, TP=2 over axis 1, bf16 weights, HiFi4 + fp32 dest acc, fp32
  gate/up/h, SILU input activation on multiply, no clamp, reduce_scatter over axis 1 -> [S/2, H/2] fp32).
  No new module: `hooks._mlp_module` gained a `prefix` argument; `_SHARED_STEPS = {"shared_expert": "mlp.shared_experts."}`
  routes through `_row_in_col_out_host_fn` like `mlp`.
- `DEVICE_STEPS["moe_full"]` now includes `shared_expert`. `moe_shared` left untouched (its own tasks).
- Gate: pcc 0.999998, rel_l2 0.00197, row ratio [0.99894, 0.99999], worst row 0.00254; x2 input rel_l2 0.00073,
  worst row 0.00096. The `pcc=0.000000` line in the log is the precompile collect pass (stubbed), not the real pass.
- Perf note (assemble): the shared partial can be added to the routed experts' partial before one reduce_scatter.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_shared_expert.py`

## S.moe_full.13 test (attempt 1)

What
- Replaced the rendered 34-line swap-13 test (moe_full layer 1, steps 1-13 on device, last: shared_expert) with swap
  12's reviewed test plus shared_expert checks. Every swap-12 check is kept at its limits; `shared_expert` added to
  SWAPPED.
- New checks on shared_out:
  - vs the CPU shared_expert on the same device ffn_norm, at the component limits: rel 0.008, ratio [0.99, 1.01],
    worst row 0.015, plus a float64 global coefficient within 0.003 of 1.
  - The module again on the device ffn_norm x 2 (bf16) vs the CPU step: rel 0.006, ratio [0.99, 1.01], worst row
    0.012, coef 0.003. This is the clamp probe, because the golden cannot see a clamp.
  - vs golden (backstop): rel 0.01, ratio [0.985, 1.015], worst row 0.02, coef 0.004.
- CPU mutation study in /tmp/hy4_sm13/study.py and coef.py (outside the repo; logs study.log, coef.log). It uses the
  swap-12 device run's seen tensors in /tmp/hy4_sm12/seen.pt. The table is in the test docstring.

Decisions
- These shared_expert bugs pass the 0.98 out gate and every swap-12 check: x 1.005, x 1.01, the clamp at 10, and rows
  1023 / 1024 swapped. x 1.01 moves the block out by only 0.0013. The vs-CPU and x 2 checks catch all of them.
- Added the coef check (not in the component test) so that x 1.005 is caught. Noise coefs are within 1.2e-4 of 1.

Results
- BRINGUP_IMPL=reference: PASS (shared vs CPU 0, pcc_swap_out 0.999997). BRINGUP_IMPL=stub: FAIL.
- Gate (device): PASS. pcc_swap_out 0.999973. Shared vs CPU: rel 0.00072, ratio [0.99915, 0.99961], worst row
  0.0010, coef 0.99945. x 2: rel 0.00072. vs golden: rel 0.0034, worst row 0.0112. Tail 0.0064, out rel 0.00753.

Gotchas
- The device shared expert has a steady -0.06 % scale (coef 0.99945, every row ratio below 1). That is 5x inside the
  0.003 limit. It is worth knowing if later steps tighten.
- Out rel vs golden is 0.00753 against 0.01, and the tail is 0.0064 against 0.01. moe_combine and ffn_residual remain.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_13_shared_expert.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_13_shared_expert.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_13_shared_expert.py

## C.moe_full.moe_combine test (attempt 1)

What
- Replaced the rendered test with a reviewed one for mlp_out = experts_out + shared_out (layer 1, [S, 6144], golden
  bf16, s4096 chunk 1). Norms: experts 6479, shared 3630, mlp_out 8216. Both addends are large, so no bf16 rounding
  budget is needed (unlike attn_residual).
- Checks vs golden: PCC >= 0.99 (gated), not a CPU bridge, size, finite, rel L2 <= 0.004, row norm ratio
  [0.995, 1.005], worst row <= 0.01.
- Per addend (float64): |coef - 1| <= 0.002, add rel <= 0.008 (shared) / 0.005 (experts), add worst row <= 0.03.
- Probes: the module again on (experts, -shared) and (experts, 0), vs the exact sums (rel <= 0.004, row <= 0.005).
- CPU study: /tmp/hy4_mc/study.py and mut.py (outside the repo). mut.py patches each variant in as the module and runs
  the test body. The table is in the test docstring.

Results
- fp32 reference: rel 0.00223, probes 0. A bf16 output: rel 0.00275, (experts, -shared) probe 0.0017.
- 1.01 x shared, 1.005 / 1.01 x experts, 2x, last row zeroed and last 32 columns zeroed fail rel L2.
- 0.995 x shared fails the addend worst-row check.
- A cached output and a + golden shared fail the probe. The zero stub fails PCC.
- Blind spot: a shared scale error below about 0.5 %.
- BRINGUP_IMPL=reference: PASS. BRINGUP_IMPL=stub: FAIL (PCC).
- Gate (device): FAIL with NotImplementedError "no device module for moe_combine yet". Expected before implement.

Gotcha for implement
- The module is called 3 times: golden inputs, then (experts, -shared), then (experts, zeros). All inputs are fp32
  at the host boundary (the bf16 golden values). Output is mlp_out [S, 6144]; a bf16 output fits the limits.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_moe_combine.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_moe_combine.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_moe_combine.py

## C.moe_full.moe_combine implement (attempt 1)

What
- `tt/mlp.py:TtMoeCombine`: mlp_out = ttnn.add(experts_out, shared_out), fp32, DRAM. It works on the column split
  [1, 1, S/2, H/2] per chip. There is no collective, because both inputs are already reduce-scattered over axis 1
  (TtHy4Experts and the shared TtDenseMLP both return fp32 in this layout). No host work in __call__.
- hooks.py: `_MOE_COMBINE_STEPS`, a two-input column-split boundary `_col_split2_host_fn` (fp32 in / out), the
  `_device_step_fn` / `device_component` branch, and "moe_combine" added to `DEVICE_STEPS["moe_full"]` (hybrid).

Results
- Gate: PASS. pcc 0.999997, rel_l2 0.002234 (the fp32 reference floor 0.00223), row ratio [0.99984, 1.00016],
  worst row 0.0024. Addend coefs 1.000000, addend rel 0. Both probes rel 0.
- The first "FAIL pcc=0.000000" line in the log comes from the precompile collect pass, not the real run.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_moe_combine.py

## S.moe_full.14 test (attempt 1)

What
- Replaced the rendered 34-line swap-14 test (moe_full layer 1, steps 1-14 on device, last: moe_combine) with swap
  13's reviewed test plus moe_combine checks. Every swap-13 check is kept at its limits; `moe_combine` added to
  SWAPPED.
- New checks on mlp_out (all on the block's own device experts_out / shared_out, which are the module's inputs):
  - vs the exact fp32 sum (CPU moe_combine): rel 0.003, ratio [0.997, 1.003], worst row 0.005.
  - per addend (float64): |coef - 1| <= 0.002, add rel 0.008 (shared) / 0.005 (experts), add worst row 0.03.
  - probes: the module on (experts, -shared) and (experts, 0) vs the exact sums, rel 0.004, row 0.005.
  - vs golden (backstop): rel 0.02, coef within 0.004 of 1, on rows routed as in the golden ratio [0.98, 1.02],
    worst row 0.03.
- `_errors` now returns failing values when the row selection is empty (the stub routes no row as the golden). Before
  this change the stub hit a torch `min()` RuntimeError there (a swap-13 check) and did not fail on an assertion.
- CPU mutation study: /tmp/hy4_sm14/study.py, log study.log (outside the repo). It uses the swap-12 device tensors in
  /tmp/hy4_sm12/seen.pt. The table is in the test docstring.

Decisions
- These bugs pass the 0.98 gate and every swap-13 check: 1.005 / 1.01 / 0.995 x shared, 1.003 / 1.005 x experts,
  1.005 x both, shared rows 1023 / 1024 swapped, golden shared_out in place of the device one, and a cached golden
  mlp_out (that one gives out rel 0.0036, better than the real run). The new checks catch every one of them.
- The vs-CPU limits are tight because the inputs are exact. A bf16 output (0.0017) still passes them.

Results
- BRINGUP_IMPL=reference: PASS. BRINGUP_IMPL=stub: FAIL (AssertionError).
- Gate (device): PASS. pcc_swap_out 0.999973. mlp_out vs CPU: rel 0, coefs 1.000000, probes 0. mlp_out vs golden:
  rel 0.0111, coef 1.00040, matched rows [0.99026, 1.00729], worst row 0.0176. Tail 0.0064, out rel 0.00753.

Gotchas
- Out rel vs golden is 0.00753 against 0.01, and the tail is 0.0064 against 0.01. Only ffn_residual remains.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_14_moe_combine.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_14_moe_combine.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_14_moe_combine.py

## C.moe_full.ffn_residual test (attempt 1)

What was done
- Replaced the rendered 22-line test with the layer-1 attn_residual test's checks, applied to out_j = h_mid_j +
  post_j * mlp_out (inputs h_mid, ffn_hc, mlp_out; post = ffn_hc cols 4-7). Gated PCC (0.99) kept. Against the
  golden: not a CPU bridge, size, finite, rel L2 <= 0.01, per-token per-stream norm ratio [0.99, 1.01], and the
  rounding-aware addend checks in float64 (|coef - 1| <= 0.01 + 2 r/||t||, excess, worst row).
- Added a second run with each row's post gates rotated by (row mod 4), compared with the CPU step
  (`ref.component`) on the same inputs: the same addend checks, plus rel L2 <= 0.005 and ratio [0.995, 1.005].
  The checks are in one helper, `_check(tag, ...)`. The rotated metrics are recorded with a `rot_` prefix.
- CPU mutation studies (/tmp/hy4_c_ffnres1/study.py and rot.py, outside the repo). The tables are in the test
  docstring.

Decisions
- Rotated second run: stream 2's post gate is ~1.6e-6 (addend norm 0.031 vs stream norm 73), so a dropped stream-2
  addend passes every golden check (excess 0.69). With rotated gates it scores rel 0.39 and excess 73.
- The golden limits are the same as attn_residual layer 1. 1.01 x mlp_out fails the stream ratio (1.0113). 1.005 x
  passes both runs; this is a blind spot, at the bf16 tolerance of stream 3's addend.

Gotchas
- ||mlp_out|| is 8216 at layer 1 (stream 3's addend norm 265 is larger than the stream, 153).
- Addend norms must be float64 (`tgt.double().norm()`). With fp32 norms the exact reference scores coefficient
  1.013-1.03.
- The first "FAIL pcc_ffn_residual_L01: pcc=0.000000" line comes from the precompile collect pass. Ignore it.

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999997, rel 0.00253, ratio [0.9973, 1.0030], addend exact; rotated exact).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device, TtHcPost via device_component; it is already registered for both hc_post steps): PASS.
  pcc_ffn_residual_L01 0.999997, rel 0.00253, addend rel 5.8e-5, rotated run within limits.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_residual.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_residual.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_full_ffn_residual.py

## S.moe_full.15 test (attempt 1)

What
- Replaced the rendered 36-line swap-15 test (moe_full layer 1, steps 1-15 on device, last: ffn_residual) with swap
  14's reviewed test plus ffn_residual checks. Every swap-14 check is kept at its limits; `ffn_residual` added to
  SWAPPED.
- New checks on block out (helper `post_check`):
  - vs the CPU ffn_residual on the block's own device h_mid / ffn_hc / mlp_out: rel <= 5e-4, worst (row, stream)
    <= 1e-3 (the h_mid limits from swap 07), plus the rounding-aware float64 addend checks per stream.
  - the module again with each row's post gates rotated by (row mod 4), vs the CPU step: same limits, plus stream
    norm ratio [0.995, 1.005] (`rot_` metrics).
  - vs golden: per-token per-stream norm ratio [0.98, 1.02] on rows routed as in the golden, [0.95, 1.05] on flipped
    rows.
- CPU mutation study: /tmp/hy4_sm15/study.py, log study.log (outside the repo). It uses the swap-12 device tensors in
  /tmp/hy4_sm12/seen.pt. The table is in the test docstring.

Decisions
- These bugs pass the 0.98 gate and every swap-14 check: 1.002 / 1.005 / 0.995 x mlp_out, 0.999 x h_mid, the
  addend dropped on stream 2, and a module that ignores the gates it is given (it only fails with rotated gates).
  The new checks catch every one of them.
- vs-CPU at 5e-4 would fail a bf16 hc_post (bf16 output 0.0017). The same TtHcPost already has to meet this limit at
  h_mid (swap 07), so this is not a new precision demand.

Results
- BRINGUP_IMPL=reference: PASS (all new checks 0; out stream ratio matched [0.99936, 1.00051], 22 flipped rows).
- BRINGUP_IMPL=stub: FAIL (AssertionError).
- Gate (device): PASS. pcc_swap_out 0.999973. out vs CPU ffn_residual rel 0 / row 0, addend coefs 1.0, rotated rel 0.
  Out stream ratio vs golden: 1936 matched rows [0.99037, 1.00875], 112 flipped rows [0.99069, 1.01949]. Tail 0.0064,
  out rel 0.00753.

Gotchas
- Out rel vs golden is 0.00753 (limit 0.01). This is now the whole block on the device.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_15_ffn_residual.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_15_ffn_residual.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_full_15_ffn_residual.py

## C.moe_shared.attn_hc test (attempt 1)

What was done
- Replaced the rendered 22-line test for attn_hc at layer 2 (moe_shared; iHC gates [S, 8], hc_attn_layer of layer 2,
  s4096 chunk 1, bf16 golden) with the layer-1 reviewed test (test_c_moe_full_attn_hc.py), LAYER = 2, with the same
  checks and limits: gated PCC 0.99; not a CPU bridge, element count, finite, rel L2 <= 0.01, per-column rel L2 (all
  8) <= 0.01, post worst row <= 0.015, attn_x rel <= 0.005 / row 0.02, h_mid per stream <= 0.003 / row 0.02.
- New docstring with the layer-2 CPU mutation table. Scripts are in /tmp/hy4hc2/{an,an2,mut,mut2,mut3,cols}.py
  (outside the repo; layer-2 copies of /tmp/hy4hc1).

Decisions
- Layer 2's gates differ from layer 1's. Pre gate 0 sits at hc_eps (1.03e-6 .. 1.6e-5), post gate 7 is small (mean
  0.0028), and stream 3 is 5x larger than the others. I first tried an abs-error limit on column 0 instead of rel L2.
  The device reached rel 0.0045 there, so all 8 columns keep rel L2 <= 0.01 as at layer 1. This is the only check
  that catches a dropped hc_eps (column 0 rel 0.47). Every mutation in the table fails at least one check.

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999999, rel 0.00132, max col rel 0.0018, post row 0.0030, attn_x 0.00117 /
  0.0027, h_mid stream <= 0.00176 / row 0.0045).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device, the existing TtHcGates through device_component; attn_hc is routed for any layer): PASS. PCC
  0.999999, rel 0.00136, col rel [0.0045, 0.0012, 0.0018, 0.0021, 0.0018, 0.0017, 0.0019, 0.0034], post row 0.0048,
  attn_x 0.00120 / 0.0032, h_mid stream <= 0.00181 / row 0.0058.

Gotchas
- Tightest margin: column 0 (device 0.0045 vs 0.01, max abs 1.05e-7 on gates of ~2e-6). Next is h_mid stream 1
  (0.00181 vs 0.003; the fp32 reference is already 0.00176, from bf16 golden rounding).
- DEVICE_STEPS["moe_shared"] in hooks.py is still empty. The implement step must add attn_hc to it (not this step's
  file).
- The first pcc line (0.000000) comes from the precompile collect pass.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_hc.py

## S.moe_shared.01 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_shared layer 2, attn_hc on device, rest CPU). It is the layer-1
  reviewed swap test (test_swap_moe_full_01_attn_hc.py) with BLOCK_TYPE = moe_shared and two changes:
  1. It sets `ctx.extra["shared_topk"]` to the golden's L1.topk (src = ref.cfg.topk_source(2)) in the reference and
     device contexts. The rendered test raised a KeyError in topk_shared, because layer 1 does not run in a
     one-block harness.
  2. It adds a per-stream h_mid rel L2 limit (<= 0.005).
- It keeps the gated pcc_swap_out (0.98), the trail, and the layer-1 asserted checks: not a CPU bridge; gates rel
  <= 0.01, per-column rel <= 0.01 on all 8 columns, post worst row <= 0.015; attn_x rel <= 0.005 / row 0.02; h_mid rel
  <= 0.005 / worst (row, stream) 0.02; router overlap >= 0.98; out rel <= 0.01.
- CPU block-level mutation study in /tmp/hy4_ss1/{study,s2}.py (outside the repo; the layer-1 /tmp/hy4_sm1 script
  with L = 2, the shared topk and extra layer-2 mutations). The table is in the test docstring.

Decisions
- Per-stream h_mid check: at layer 2, stream 3 (norm 340) dominates h_mid, so the whole-tensor rel L2 is 0.0009 on the
  reference and only 0.0042 for post x 1.02. The per-stream check (reference max 0.0029, bf16-rounded gates 0.0032)
  catches post x 1.005 (stream 1 at 0.0060), which every other check misses.
- The out worst (row, stream) rel L2 is recorded, not asserted (0.024 on the fp32 reference, 0.063 with bf16 gates:
  near-tie expert flips).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999995, rel 0.0030, gates col rel max 0.0018, h_mid stream max 0.0029,
  router 0.9973, topk match 1.0).
- BRINGUP_IMPL=stub: FAIL on every check (out PCC 0.852).
- Gate (device TtHcGates): PASS. pcc_swap_out 0.999994, gates rel 0.00136, col rel [0.0045, 0.0012, 0.0018, 0.0021,
  0.0018, 0.0017, 0.0019, 0.0034], post row 0.0048, attn_x 0.00226 / 0.0025, h_mid 0.00089 / row 0.0038 / stream max
  0.00295, router 0.9966, out rel 0.00345.

Gotchas
- Every later moe_shared swap test (02..) renders from the same template and needs the shared_topk fix too.
- Tightest margins: gate column 0 (device 0.0045 vs 0.01) and h_mid stream 0 (0.00295 vs 0.005).
- Not caught: nothing in the mutation table except bf16-rounded gates (not a bug).

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_01_attn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_01_attn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_01_attn_hc.py

## C.moe_shared.attn_hc_pre.test.1 (test review, layer 2)

What
- Rewrote the rendered test as the layer-1 attn_hc_pre test (test_c_moe_full_attn_hc_pre.py) with LAYER = 2. It has
  the same checks and limits: golden rel L2 <= 0.005, row norm ratio in [0.994, 1.006], worst row <= 0.01; vs the CPU
  step on the same inputs, rel <= 0.003 and row <= 0.006; rotated pre gates (row mod 4) vs the CPU step, rel <= 0.004
  and row <= 0.01. The docstring has the mutation tables measured on the layer-2 golden (CPU script, no device).

Decisions
- Kept the layer-1 limits. On the layer-2 golden, bf16 accumulation scores 0.00376 / [0.9964, 1.0023] / 0.0050
  (vs CPU 0.00278 / 0.0032), and pre x 1.005 is caught (rel 0.00558, ratio max 1.0073).
- Layer-2 pre gates: column means 1.7e-6, 0.96, 0.20, 0.032. Gate 0 sits at hc_eps, so stream 0 has no visible effect
  on the golden ("stream 0 dropped" passes every golden check). The rotated-gates run catches it (rel 0.19).

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, rel 0.00253, ratio [0.99766, 1.00227], row 0.0035).
- BRINGUP_IMPL=stub: FAIL (pcc 0).
- Gate (device): PASS. device_component already routes attn_hc_pre to TtHcPre for every block type. pcc 0.999997; vs
  CPU step rel 0.000000; rotated rel 0.000000. TtHcPre's fp32 result matches the CPU step to 6 decimals, as at layers 0
  and 1.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_hc_pre.py

## S.moe_shared.02 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_shared layer 2, attn_hc + attn_hc_pre on device, rest CPU). It is the
  layer-1 reviewed swap test (test_swap_moe_full_02_attn_hc_pre.py) with BLOCK_TYPE = moe_shared and the two layer-2
  changes from swap 01: `ctx.extra["shared_topk"]` = the golden's L1.topk in both contexts, and a per-stream h_mid rel
  L2 limit (<= 0.005).
- Kept the layer-1 checks and limits: gates (rel 0.01, per column 0.01, post row 0.015); attn_x vs golden (rel 0.005,
  row ratio [0.996, 1.004], row 0.01); attn_x vs the CPU hc_pre on the device gates (0.003 / 0.006); the module again
  with pre gates rotated by row mod 4 vs the CPU step (0.004 / 0.01); h_mid (0.005 / row 0.02 / stream 0.005); router
  overlap 0.98; out rel 0.01.
- CPU mutation study: /tmp/hy4_ss2/study.py (outside the repo; the layer-1 /tmp/hy4_sm2 script with L = 2, the shared
  topk, a per-stream h_mid column, and extra layer-2 mutations). The table is in the test docstring.

Decisions
- Kept the layer-1 limits. bf16 accumulation fits (ratio min 0.9967, CPU row 0.0039, rot row 0.0054), and pre x 1.005
  fails (ratio 1.0048, rel 0.0055).
- Stream 0 (pre gate ~hc_eps) is invisible on the golden. Only the rotated-gates run sees it dropped (rel 0.19).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999995, rel 0.0030; attn_x 0.00225; h_mid stream max 0.0029; router 0.9973).
- BRINGUP_IMPL=stub: FAIL on every check (out PCC 0.94).
- Gate (device TtHcGates + TtHcPre): PASS. pcc_swap_out 0.999994; gates rel 0.00136, col max 0.0045 (col 0); attn_x
  0.00226 / ratio [0.99957, 1.00024] / row 0.0025; vs CPU 0 / 0; rotated 0 / 0; h_mid 0.00089 / row 0.0038 / stream
  max 0.00295; router 0.9966; out rel 0.00345.

Gotchas
- Not caught: pre + 3e-4 (attn_x vs CPU worst row 0.0052 against 0.006), which is below the bf16 rounding of gate 1.
- The first block of output in each run is the precompile collect pass (stubbed), not the real run.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_02_attn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_02_attn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_02_attn_hc_pre.py

## C.moe_shared.attn_norm test (attempt 1)

What was done
- Replaced the rendered 22-line component test (moe_shared layer 2, attn_norm) with the reviewed layer-1 attn_norm
  test (test_c_moe_full_attn_norm.py) at LAYER = 2, with the same limits: gated pcc_attn_norm_L02 (0.99); not a CPU
  bridge; finite output, element count; vs golden rel L2 <= 0.008, row norm ratio in [0.993, 1.007], worst row
  <= 0.015; module run again on golden x 0.1 (bf16) vs the CPU step, rel <= 0.01, worst row <= 0.02. The docstring has
  the mutation tables measured on the layer-2 golden.
- CPU mutation study: /tmp/hy4_an2/study.py (outside the repo; the layer-1 /tmp/hy4_an1 script with L = 2).

Decisions
- Kept the layer-1 limits. Pessimistic bf16 on layer 2: rel 0.0037 / ratio [0.9954, 1.0043] / row 0.0057.
- Layer-2 w is flat ([0.084, 0.159]), so 1 + w, no weight and a permuted w (TP layout bug) all pass PCC (0.996-0.998).
  They fail rel L2 (0.087 to 7.5). Every mutation that passes PCC fails at least one added check.
- Eps matters more here: the smallest row's mean(x^2) is 0.94x eps (11 rows below eps). Eps 1e-6 scores rel 0.133.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, rel 0.00235, ratio [0.99988, 1.00012], row 0.0025; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): already PASS, because the dense_full attn_norm module serves every block type. pcc 0.999996, rel
  0.00288, ratio [0.99903, 1.00143], row 0.0032; scaled rel 0.00188, row 0.00253.

Gotchas
- The first "FAIL pcc_attn_norm_L02: pcc=0.0" line comes from the precompile collect pass. The real run follows it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_norm.py

## S.moe_shared.03 test (attempt 1)

What was done
- Replaced the rendered 22-line swap test (moe_shared layer 2, attn_hc + attn_hc_pre + attn_norm on device, rest
  CPU). It is the reviewed layer-1 test (test_swap_moe_full_03_attn_norm.py) with the layer-2 changes of swaps 01 / 02:
  ctx.extra["shared_topk"] = golden L1 topk in both contexts, no indexer top-k check (a shared layer has no indexer;
  golden L1 and L2 topk are identical), a per-stream h_mid limit (0.005) and router overlap >= 0.98. The other limits
  are unchanged: gates, attn_x (vs golden, vs CPU, rotated gates), attn_norm vs golden / vs CPU (0.008, ratio
  [0.993, 1.007], row 0.015), attn_norm on attn_x x 0.1 (0.01 / 0.02), q_resid 0.01, attn_out 0.01 / 0.05, h_mid
  0.005 / 0.02, out rel 0.01.
- CPU block-level mutation study: /tmp/hy4_ssh3/study.py (outside the repo, 9 s per variant). The table is in the
  test docstring.

Decisions
- Router limit 0.98, not the layer-1 0.99. Layer-2 reference gives 0.9973 and bf16 0.9964. At the swaps 01 / 02 value
  it is a gross check; the attn_norm checks catch the bugs.
- 12 of 17 attn_norm mutations pass the 0.98 out gate (all eps variants, x 1.01 / 1.02, RMS subsets, LayerNorm, a
  zeroed row, TP-swapped w). The attn_norm checks catch every one. bf16 everywhere fits: 0.0041 / [0.9939, 1.0058] /
  0.0071.

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999995, attn_norm 0.00222, router 0.99725, out rel 0.00304).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98, every extra check fails).
- Gate (device): PASS. pcc_swap_out 0.999994. attn_norm 0.00287, ratio [0.99823, 1.00034], row 0.00341. vs CPU
  0.00193. x0.1 0.00188. q_resid 0.00200. attn_out 0.00235. h_mid 0.00092, stream max 0.0031, row 0.0039. router
  0.99689. out rel 0.00347.

Gotchas
- The smallest margin is still gate column 0 (0.0045 vs 0.01), carried over from swap 01.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_03_attn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_03_attn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_03_attn_norm.py

## C.moe_shared.q_a test (attempt 1)

What was done
- Replaced the rendered one-line test with test_c_moe_full_q_a.py's checks at LAYER = 2, same limits: gated
  pcc_q_a_L02 (0.99), no CPU bridge, element count, finite output, rel L2 <= 0.008, row norm ratio in [0.994, 1.006],
  worst row rel L2 <= 0.015 vs the golden; second run on golden input x 0.01 (bf16) vs the CPU step (rel <= 0.01,
  worst row <= 0.02, the eps check). Layer-2 mutation tables are in the test docstring (CPU study script
  /tmp/hy4_qa_l2/study.py, outside the repo, 10 s).

Decisions
- Kept the layer-0/1 limits: layer-2 golden statistics match (pre-norm row rms 0.309-0.489, mean square >= 9.6e4 x
  eps; bf16 estimate rel 0.0030, ratio [0.9989, 1.0009], worst row 0.0039). Every mutation in the table fails at least
  one check. The shared layer has its own q_a (only the indexer is shared), so no shared_topk setup is needed here.

Gotchas
- Layer-2 norm w is smaller (mean 0.103, [0.021, 0.21]), so a dropped norm weight (PCC 0.952) and 1 + w (0.961) fail
  PCC again, unlike layer 1. Norm per K partial, RMS over half the columns, eps and a zeroed tile row still pass PCC;
  rel L2 / row ratio / x 0.01 catch them.
- As before, the first "FAIL pcc_q_a_L02: pcc=0.000000" line is the precompile collect pass.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999998, rel 0.00183, ratio [0.99952, 1.00051], worst row 0.0022; scaled 0.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device, existing TtQa serves layer 2): PASS. pcc 0.999998, rel 0.00201, ratio [0.99842, 1.00086], worst row
  0.00326; scaled x 0.01 rel 0.00176, worst row 0.00224.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_q_a.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_q_a.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_q_a.py

## S.moe_shared.04 test (attempt 1)

What was done
- Replaced the rendered one-line swap test (attn_hc, attn_hc_pre, attn_norm, q_a on device, layer 2) with
  test_swap_moe_shared_03_attn_norm.py (layer-2 setup: ctx.extra["shared_topk"] = golden L1 topk in both contexts, no
  indexer top-k check, per-stream h_mid limit 0.005, router >= 0.98) plus the q_a checks of
  test_swap_moe_full_04_q_a.py at the test_c_moe_shared_q_a.py limits: q_resid vs golden (rel 0.008, row ratio
  [0.994, 1.006], worst row 0.015), vs the CPU q_a on the device attn_norm (same limits), and the q_a module on the
  device attn_norm x 0.01 (bf16) vs the CPU step (0.01 / 0.02, the eps check). Every other limit is unchanged from
  swap 03. The q_resid check is now at component limits (swap 03 had a loose rel 0.01 only).
- CPU block-level q_a mutation study at layer 2: /tmp/hy4_ssh4/study.py (outside the repo, 9 s per variant). The table
  is in the test docstring.

Decisions
- At layer 2 every q_a mutation passes the 0.98 out gate, even a zeroed q_resid (out PCC 0.99977). The shared top-k
  comes from the golden, so q_a only feeds q_b, and the residual dominates. The q_resid checks catch every mutation
  except eps; the x 0.01 run catches eps (1e-5 0.181, 0 0.029, 2e-6 0.026 vs bf16 0.0023).

Results
- BRINGUP_IMPL=reference: PASS (out PCC 0.999995, q_resid 0.00180 / [0.99952, 1.00054] / 0.00209, router 0.99725).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98, every extra check fails).
- Gate (device): PASS. pcc_swap_out 0.999994. q_resid 0.00224, ratio [0.99860, 1.00094], row 0.00336; vs CPU
  0.00177 / 0.00222; x 0.01 0.00176 / 0.00216. attn_out 0.00235. h_mid 0.00092, stream max 0.0031. router 0.99689.
  out rel 0.00347.

Gotchas
- The smallest margin is still gate column 0 (0.0045 vs 0.01), carried over from swap 01.
- The first block of FAIL-looking lines in the reference / gate logs is the precompile collect pass, as before.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_04_q_a.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_04_q_a.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_04_q_a.py

## C.moe_shared.topk_shared.test.1 (test review)

What
- Rewrote test_c_moe_shared_topk_shared.py. The rendered test raised the reference's KeyError (layer 1's top-k not in
  the reference cache) under BRINGUP_IMPL=reference, and used positional match for the integer output.
- Now sets ctx.extra["shared_topk"] = golden L{topk_source(2)=1}.topk in both contexts (as the moe_shared swap tests).
- Gated metric pcc_topk_shared_L02 = order-free per-row set overlap on the normalized output (-1 / 0xFFFFFFFF pads
  as -1), 0.99. Extra asserts: integer S x 2048 output; exact per-row set equality with the golden (no repeats, no
  non-causal positions); pads a contiguous tail and >= 1 valid key per row (sparse_sdpa preconditions); a second call
  on chunk 0 (start 0, 2047 padded rows, its own shared_topk) must also match exactly.

Decisions
- Exactness, not a tolerance: topk_shared is an identity (components.yaml NATIVE, no op). Golden L2.topk == L1.topk
  exactly on chunks 0 and 1 (checked on the CPU).
- Order is not required: the all-device model hands on layer 1's unsorted topk_large_indices tensor. CPU check: a
  shuffled-valid, sentinel-tail output passes; pads moved to the front, a chunk-1 result on chunk 0, one changed
  entry, one dropped entry per row all fail.

Results
- BRINGUP_IMPL=reference: PASS (overlap 1.000000, chunk 0 exact, 2096128 pads).
- BRINGUP_IMPL=stub: FAIL (overlap 0.000488).
- Gate (device): FAIL, NotImplementedError "no device module for topk_shared yet". Expected: the implement step
  adds it.

For the implement step
- The device fn gets (dctx, attn_norm) and must read dctx.extra["shared_topk"] (int64 host, -1 pads). Return integer
  positions [S, 2048] (host), pads as -1 or 0xFFFFFFFF in a contiguous tail; the order within a row is free.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_topk_shared.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_topk_shared.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_topk_shared.py

## C.moe_shared.topk_shared implement (attempt 1)

What
- New `tt/topk_shared.py:TtTopkShared(mesh, source_layer)`: identity on the device top-k tensor (components.yaml
  NATIVE, no op; the ReuseIndexer idea). It returns the source full layer's tensor as it is: [1, 1, S/2, 2048] uint32
  ROW_MAJOR per chip, row-split over axis 0, replicated over axis 1, 0xFFFFFFFF tail. It never frees or copies it.
- `tt/layout.py`: `topk_to_device` / `topk_to_host` (host int64 with -1 pads <-> the TtHy4Indexer output layout).
  These are harness-boundary helpers only.
- hooks.py: `_TOPK_SHARED_STEPS`, `_TopkSharedHostFn` (source = ctx.extra["shared_topk"], else `source(ctx)`),
  `_topk_shared_module` (asserts that the layer is a shared-index layer), branches in `_device_step_fn` and
  `device_component`. `DEVICE_STEPS["moe_shared"] = {"topk_shared"}`. HybridDeviceModel sets the fn's
  `source = ref._shared_topk(i, ctx)`. That is the reference's per-chunk record of layer 1's top-k, which is the
  device indexer's output (through `_record_topk`).

Decisions
- There is no device op in the step. The per-call host transfer (upload + read-back) happens only at the harness
  boundary, because the component and hybrid contracts are host in / host out. In the all-device model the assemble
  step must keep layer 1's indexer output tensor alive and pass it to TtTopkShared for layers 2-4.
- `DEVICE_STEPS["moe_shared"]` lists only topk_shared. attn_hc, attn_hc_pre, attn_norm and q_a passed their moe_shared
  component and swap gates by reusing the existing modules, but no step added them to the hybrid. I left them out
  because they are outside this task.

Results
- Gate: PASS. pcc_topk_shared_L02 topk_overlap = 1.000000. Golden chunk: worst row 1.0, 0 pads, positional match
  1.0. Chunk 0: exact, 2096128 pads.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_topk_shared.py

## S.moe_shared.05.test.1 (test review)

What
- Rewrote test_swap_moe_shared_05_topk_shared.py from test_swap_moe_shared_04_q_a.py. It keeps the layer-2 setup
  (ctx.extra["shared_topk"] = golden L1.topk in both contexts) and every swap 04 check and limit, and adds the topk
  checks of test_c_moe_shared_topk_shared.py. The rendered test would have raised the reference's KeyError.
- topk checks: integer S x 2048 output, pads normalized; exact per-row set equality vs golden L2.topk and vs the
  shared input; causal; no repeats; contiguous pad tail; >= 1 valid key per row; the topk_shared module run again
  on chunk 0 (golden chunk-0 L1.topk, 2047 padded rows) must be exact too. Order within a row is free.

Decisions
- Exact checks, because every topk mutation passes the 0.98 out gate on the CPU (/tmp/hy4_ssh5/study.py). Even an
  all-zero top-k passes (0.99346), and so do the chunk-0 top-k (0.99951) and SP halves swapped (0.99985). Shuffled
  or reversed rows leave attn_out bit-identical. Table in the test docstring; known issue proposed.

Results
- BRINGUP_IMPL=reference: PASS (out rel 0.00304, topk exact, chunk 0 exact with 2096128 pads).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98, every check fails).
- Gate (device): PASS. pcc_swap_out 0.999994. topk exact on both chunks. attn_out 0.00235, h_mid 0.00092 (stream
  max 0.0031), router 0.99689, out rel 0.00347. The smallest margin is still gate column 0 (0.0045 vs 0.01).

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_05_topk_shared.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_05_topk_shared.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_05_topk_shared.py

## C.moe_shared.attention.test.1 (test review)

What
- Rewrote test_c_moe_shared_attention.py from test_c_moe_full_attention.py at LAYER = 2, with the same four checks:
  golden chunk 1, golden chunk 0 (empty prefix, -1 pads), a probe topk (64 random causal keys per row) vs the CPU
  step, and attn_norm x 1e-3 vs the CPU step (the eps probe). topk is a graph input (the topk_shared output), so the
  test does not need ctx.extra["shared_topk"].
- Added a float64 global scale coefficient check (<got, want> / <want, want> within 0.004 of 1) to every check.

Decisions
- Re-ran the layer-1 mutation study on layer 2 (/tmp/hy4_c_attn2/mut.py, CPU only). Sink passed raw to sparse_sdpa
  passes the layer-1 limits on chunk 1, so checks 1-3 are tightened to rel 0.012, ratio [0.99, 1.01], worst row
  0.025. The device has 2x / 5x / 3x margin on them.
- The device is noisier on the eps probe at layer 2 (rel 0.0119, worst row 0.0206), so that check has its own limits:
  rel 0.03, ratio [0.99, 1.01], worst row 0.05. The eps mutations score rel >= 0.127 there.
- Every mutation in the docstring tables fails at least one check.

Results
- BRINGUP_IMPL=reference: PASS (golden rel 0.0020, worst row 0.0022).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device, the existing TtHy4Attention with layer-2 weights): PASS. pcc_attention_L02 0.999981. golden rel
  0.00613 / [0.9980, 1.0012] / 0.0081 / coef 0.99970. chunk0 0.00592. probe 0.00538. scaled 0.0119 / 0.0206. The
  numbers are identical across three runs.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attention.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attention.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attention.py

## S.moe_shared.06.test.1 (test review)

What
- Replaced the rendered swap test (moe_shared layer 2, attn_hc .. topk_shared + attention on device). It is swap 05
  (moe_shared) with the attention checks from test_c_moe_shared_attention.py at its layer-2 limits: rel 0.012, row
  norm ratio [0.99, 1.01], worst row 0.025, float64 coef within 0.004. They apply to attn_out (a) vs golden and
  (b) vs the CPU attention on the device inputs, and to the module again on (c) golden chunk 0, (d) a probe topk
  (64 random causal keys per row, seed 0) and (e) the device attn_norm x 1e-3 (component check 4 limits: 0.03 /
  [0.99, 1.01] / 0.05). Swap 05's attn_out check (0.01 / 0.05) is gone. Every other swap 05 check and limit stays.
  That includes the layer-2 shared_topk setup and exact topk sets.
- CPU swap mutation study at layer 2: /tmp/hy4_ssh6/study.py on /tmp/hy4_c_attn2/mut.py (outside the repo, ~13 s
  per variant). The table is in the test docstring.

Decisions
- h_mid per-stream limit raised from 0.005 to 0.01. The device scores [0.0051, 0.0067, 0.0050, 0.0006] and the bf16
  estimate is 0.0049, because at layer 2 attn_out is a large part of streams 0-2. h_mid whole-tensor (0.005) and worst
  (row, stream) (0.02) are unchanged. Device: 0.0015 / 0.0084.
- Router kept at 0.98 (device 0.99353). Out rel kept at 0.01 (device 0.00683, the tightest remaining margin).

Gotchas
- 23 of 32 attention bugs pass the 0.98 out gate at layer 2. The attn_out checks catch every one of them.
- run_safe_pytest's log holds a precompile pass first (pcc 0, every check at 1.0). Read the second block.
- The study's "zero stub" row (scale_out 0.0) is a no-op because 0.0 is falsy; use 1e-30. Zeroed attn_out scores
  out PCC 0.852.

Results
- BRINGUP_IMPL=reference: PASS (pcc_swap_out 0.999995, attn_out vs golden 0.00193, vs CPU / chunk 0 / probe / scaled
  exact, h_mid stream max 0.0029, router 0.99725, out rel 0.00304).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device, TtHy4Attention): PASS, identical over two runs. pcc_swap_out 0.999977. attn_out vs golden 0.00628 /
  [0.99818, 1.00155] / 0.0088 / coef 0.99936, vs CPU 0.00582, chunk 0 0.00558, probe 0.00539, scaled 0.0119 /
  0.0212. topk exact. h_mid 0.00146 / 0.0067 / 0.0084. router 0.99353. out rel 0.00683.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_06_attention.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_06_attention.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_06_attention.py

## C.moe_shared.attn_residual.test.1 (test review, layer 2)

What
- Replaced the rendered test with the moe_full layer-1 attn_residual test (same step: iHC post, h_mid_j = in_j +
  post_j * attn_out) at LAYER = 2. It keeps the same checks with the fixed parts tightened: rel L2 <= 0.005 (was
  0.01), per-token per-stream norm ratio [0.995, 1.005] (was [0.99, 1.01]), addend coef tol 0.005 + 2 r/||t|| (was
  0.01), addend rel 0.005 ||t|| + 2 r (was 0.01). The per-row limit is unchanged (0.05 ||t|| + 2 r + 1e-6).
- CPU mutation study on the layer-2 golden: /tmp/hy4_c_res2/study.py (layer-1 limits) and full_tight.py / tight.py
  (new limits), outside the repo. The table is in the test docstring.

Decisions
- Layer 2 golden: stream norms 70 / 65 / 73 / 335, post column means 0.21 / 0.24 / 0.22 / 0.0028, addend norms
  37 / 45 / 40 / 0.64. The bf16 budget r/||t|| is 0.0023 / 0.0015 / 0.0023 on streams 0-2, so they can take tighter
  limits. Stream 3's addend is below bf16 resolution (0.12), and the rounding-aware term covers it.
- With the layer-1 limits, 1.01 x attn_out passed (coef 0.76 of tol). With the new limits it fails (1.24), and so
  does 0.99 x. A bf16-output module still uses only 0.49 of the allowance (worst row, stream 3). All other 33
  mutations fail either way.

Gotchas
- The bf16 golden h_mid fails the addend checks itself (stream 3 excess 1.18, row 10.6). It was rounded from
  unrounded fp32 inputs, so it is not a module output and not the bar. The fp32 reference on the golden inputs scores 0.
- The first "FAIL pcc ... 0.000000" line is the precompile pass. Ignore it.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00101, ratio [0.9989, 1.0015], addend exact, worst row 0.001).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device, TtHcPost via device_component; hooks already cover the step for any layer): PASS. Numbers are
  identical to the reference: rel 0.00101, ratio [0.9989, 1.0015], coef 1.0, excess 0.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_residual.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_residual.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_attn_residual.py

## S.moe_shared.07.test.1 (test review)

What
- Replaced the rendered swap test (moe_shared layer 2, attn_hc .. attention + attn_residual on device). It is swap 06
  (moe_shared) with every check and limit unchanged, including the layer-2 shared_topk setup, plus the attn_residual
  checks of test_swap_moe_full_07_attn_residual.py. Those are: h_mid per-token per-stream norm ratio vs golden; h_mid
  vs the CPU attn_residual on the same device inputs (rel 5e-4, worst (row, stream) 1e-3); the rounding-aware addend
  checks; and the module again with per-row rotated post gates vs the CPU step. The addend limits are the layer-2
  component's (coef 0.005, stream 0.005, row 0.05). The layer-1 limits were 0.01.
- CPU mutation study at layer 2: /tmp/hy4_ssh7/study.py (adapted from /tmp/hy4_sm7/study.py, outside the repo,
  ~10 s per variant). The table is in the test docstring.

Decisions
- Stream norm ratio limit is [0.99, 1.01] (moe_full 07 used [0.98, 1.02]). The device scores [0.99953, 1.00088].
- At layer 2 stream 3 has the tiny post gate (~0.003), not streams 0 / 1 as at layer 1. With the rotated gates,
  stream 3 meets a large gate on 3/4 of the rows, so attn_out dropped on stream 3 scores rot rel 0.10.

Gotchas
- 20 of 29 residual bugs pass the 0.98 out gate at layer 2, among them post columns 0 / 1 swapped (0.99940) and
  attn_out dropped on stream 0 (0.99294). The vs-CPU check catches every one of them.
- A bf16-output module has vs-CPU whole rel 0.00047, under the 5e-4 limit. It fails only on the worst row (0.0017).

Results
- BRINGUP_IMPL=reference: PASS (pcc_swap_out 0.999995, h_mid vs CPU exact, stream ratio [0.9998, 1.0002]).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device, TtHcPost + TtHy4Attention): PASS, identical over two runs. pcc_swap_out 0.999977; h_mid 0.00146 /
  stream max 0.0067 / 0.0084, ratio [0.99953, 1.00088]; h_mid vs CPU 0 / 0, addend coef 1.0 / excess 0, rotated
  0 / 0; attn_out vs golden 0.00628 / 0.0088; router 0.99353; out rel 0.00683.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_07_attn_residual.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_07_attn_residual.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_07_attn_residual.py

## C.moe_shared.ffn_hc test (attempt 1)

What was done
- Replaced the rendered 22-line test for ffn_hc at layer 2 (iHC gates [S, 8] from h_mid, hc_mlp_layer of layer 2,
  s4096 chunk 1, bf16 golden) with the layer-1 test (test_c_moe_full_ffn_hc.py) re-tuned for layer 2. Same checks:
  gated PCC 0.99, CPU-bridge assert, element count, finite, rel L2 <= 0.01, per-column rel L2 <= 0.01 (0.02 on
  column 1), post worst row <= 0.015, ffn_x via the CPU ffn_hc_pre (<= 0.005 / row 0.02), out via the CPU
  ffn_residual with golden mlp_out (per stream <= 0.003 / row 0.02).
- CPU mutation study: /tmp/hy4_ffnhc2/{keys,mut}.py (the layer-1 scripts with layer 2, outside the repo). Table in
  the test docstring.

Decisions
- SMALL_COLS = (1,) only: at layer 2 pre gate 1 sits at hc_eps (mean 1.4e-6); columns 0 and 6 (layer 1's small
  ones) are 1.1e-3 / 3.5e-4 here, so they get the 0.01 limit (device 0.0030 / 0.0050).
- Out per-stream limit 0.003 (layer 1: 0.005). Post x 1.01 scores 0.0044; the device scores 0.00099. At 0.005 that
  bug failed only the per-column check, by 0.0002.
- Every study mutation fails at least one check, except bf16 rounding and swapping the two gate scales (equal at
  layer 2, 0.0398, a no-op).

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00026, col rel <= 0.0019, post row 0.0040, out stream <= 0.00073).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device, the existing TtHcGates via `_HC_STEPS["ffn_hc"]`): PASS. PCC 1.000000, rel 0.00029, col rel [0.0030,
  0.0064, 0.0, 0.0018, 0.0042, 0.0041, 0.0050, 0.0023], post row 0.0081, ffn_x 0.00070 / 0.0052, out stream
  <= 0.00099 / row 0.0033.

Gotchas
- Tightest margins: post columns 4-6 (~0.0045 vs 0.01) and the post worst row (0.0081 vs 0.015), all from the device
  sigmoid on ~5e-4 gates. Column 1 (hc_eps) is at 0.0064 vs 0.02.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_hc.py

## S.moe_shared.08.test.1 (test review)

What
- Replaced the rendered swap test (moe_shared layer 2, attn_hc .. attn_residual + ffn_hc on device). It is swap 07
  (moe_shared) with every check and limit unchanged, including the layer-2 shared_topk setup. Added the ffn_hc checks
  of test_swap_moe_full_08_ffn_hc.py: ffn_hc vs golden; vs the CPU ffn_hc on the same device h_mid; the pre gates
  through ffn_x; the post gates through out with the block's own mlp_out; and out vs the whole CPU tail. The column
  set and out limit come from test_c_moe_shared_ffn_hc.py.
- CPU mutation study at layer 2: /tmp/hy4_ssh8/study.py + study2.py (adapted from /tmp/hy4_sm8/study.py, outside the
  repo, ~5 s per variant). The table is in the test docstring.

Decisions
- FHC_SMALL_COLS = (1,): only pre gate 1 sits at hc_eps at layer 2. Layer 1 used (0, 1, 6).
- The vs-CPU per-column limit is 0.01 (layer 1: 0.015) and column 1's is 0.02. The device's worst values are 0.0074
  (column 6) and 0.0100 (column 1).
- Post through out: 0.003 per stream (layer 1: 0.005). Post x 1.01 scores 0.0043; the device scores 0.0010.
- No scaled-input eps probe. At the block's own scale, eps 5e-6 already fails the per-column check (0.014).

Gotchas
- 36 of 49 ffn_hc mutations pass the 0.98 out gate, including post = 1 x sigmoid (0.986) and one chip's partial
  sumsq (0.985). The extra checks catch every one except post x 1.005.
- The layer-1 study's row swap (`o[[1023, 1024]].copy_(...)`) was a no-op, because advanced indexing copies. Fixed
  here with `__setitem__` (known_issues Proposed).

Results
- BRINGUP_IMPL=reference: PASS (pcc_swap_out 0.999995, ffn_hc vs CPU exact, router 0.99725).
- BRINGUP_IMPL=stub: FAIL (out PCC below 0.98 and every extra check).
- Gate (device: TtHcGates for ffn_hc, TtHcPost, TtHy4Attention, ...): PASS, identical over two runs. pcc_swap_out
  0.999977; ffn_hc vs golden 0.00043 / columns <= 0.0105 / post row 0.0088; vs CPU 0.00018 / columns <= 0.0074
  (column 1 0.0100) / post row 0.0067; ffn_x vs CPU 0.00038 / 0.0013; post through out <= 0.0010 / 0.0026; tail
  0.0016; router 0.99341; out rel 0.00686.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_08_ffn_hc.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_08_ffn_hc.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_08_ffn_hc.py

## C.moe_shared.ffn_hc_pre test (attempt 1)

What was done
- Replaced the rendered 22-line test for ffn_hc_pre at layer 2 (ffn_x = sum_j pre_j x h_mid stream j, s4096 chunk 1,
  bf16 golden) with the layer-1 test (test_c_moe_full_ffn_hc_pre.py), LAYER = 2, docstring tables re-measured on the
  layer-2 golden. Checks: gated PCC 0.99, CPU-bridge assert, element count, finite; vs golden rel <= 0.005, row ratio
  [0.994, 1.006], worst row <= 0.01; vs CPU step rel <= 0.003, row <= 0.006; rotated pre gates vs CPU 0.004 / 0.01.
- CPU mutation study: /tmp/hy4_ssh_ffnhcpre2/study.py (the layer-1 script with layer 2, outside the repo).

Decisions
- Limits unchanged from layer 1: every study mutation except pre + 3e-4 (a gate-input error) fails at least one
  check. Layer-2 gates: pre means 1.1e-3 / 1.4e-6 / 1.0 / 0.10; only stream 1 is invisible on the golden (the
  rotation catches it at rel 0.125); a dropped stream 0 now fails the golden worst row (0.049).

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00243, ratio [0.99712, 1.00294], worst row 0.00575; vs CPU 0; rotated 0).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device, TtHcPre via `_HC_PRE_STEPS`, blackhole-box-2x2): PASS, pcc_ffn_hc_pre_L02 0.999997, same numbers as
  the reference (the fp32 device module matches the CPU step exactly).

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_hc_pre.py

## S.moe_shared.09 test (attempt 1)

What
- Replaced the rendered swap test (moe_shared layer 2, attn_hc .. ffn_hc + ffn_hc_pre on device) with swap 08
  (test_swap_moe_shared_08_ffn_hc.py), keeping every check and limit, plus the ffn_hc_pre checks of
  test_swap_moe_full_09_ffn_hc_pre.py: device ffn_x vs the CPU ffn_hc_pre on the same device h_mid and gates
  (rel <= 0.003, row <= 0.006), and the module again on per-row rotated pre gates vs the CPU step (0.004 / 0.01).
  The ffn_x vs golden check (0.01 / 0.05) now sees the device step.

Decisions
- Limits are the component test's (test_c_moe_shared_ffn_hc_pre.py, the same as layer 1). No new CPU study: the
  component test's layer-2 mutation table (/tmp/hy4_ssh_ffnhcpre2/study.py) covers the step. The rotation is what
  catches a dropped stream 1 (gate 1 sits at hc_eps) and a 0 / 1 swap. ffn_norm hides these, so out cannot see them.

Results
- BRINGUP_IMPL=reference: PASS (pcc_swap_out 0.999995, ffn_hc_pre vs CPU and rotated 0).
- BRINGUP_IMPL=stub: FAIL (out PCC 0 and every extra check).
- Gate (device, TtHcPre): PASS. pcc_swap_out 0.999977; ffn_hc_pre vs CPU 0 / 0, rotated 0 / 0; ffn_x vs golden
  0.0047 / 0.0100; tail 0.0016; router 0.99341; out rel 0.00686.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_09_ffn_hc_pre.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_09_ffn_hc_pre.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_09_ffn_hc_pre.py

## C.moe_shared.ffn_norm test (attempt 1)

What
- Replaced the rendered 22-line component test (moe_shared layer 2, ffn_norm) with the reviewed layer-1 test
  (test_c_moe_full_ffn_norm.py) at LAYER = 2, with the same limits and checks. It keeps the gated pcc_ffn_norm_L02
  (0.99) and the CPU-bridge assert. vs golden: finite, element count, rel <= 0.008, row ratio [0.993, 1.007], worst row
  <= 0.015. Module on golden x 0.1 vs the CPU step: rel <= 0.01, row <= 0.02. On x 30: 0.006 / [0.993, 1.007] / 0.015.
  The docstring tables were re-measured on the layer-2 golden.
- CPU mutation study in /tmp/hy4_ffnnorm2/study{,2}{,_f64}.py (outside the repo; logs study.log, study_f64.log). These
  are the layer-1 scripts with the layer changed; the _f64 copies compute PCC in float64.

Decisions
- Limits unchanged. Layer-2 ffn_x has row rms [0.0025, 0.062], and 15% of rows have mean(x^2) < eps (layer 1 had
  none). The bf16 estimate (0.0037 / [0.9960, 1.0037] / 0.0051) leaves about 2x margin. Every mutation except eps 1e-2
  passes the 0.99 PCC gate, and each one fails a golden check. The closest is RMS over half the columns (rel 0.0080,
  but its ratio [0.9795, 1.0264] and worst row 0.027 fail).

Gotchas
- float32 torch.corrcoef underestimates PCC at this size (known_issues Proposed). Use float64 for the tables.
- The first "FAIL pcc_ffn_norm_L02: pcc=0.0" line comes from the precompile collect pass.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00234, ratio [0.99985, 1.00014], row 0.00246; both probes 0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): already PASS, because hooks._GATHERED_NORM_STEPS (TtGatheredRmsNorm) does not depend on the block
  type. pcc 0.999996, rel 0.00283, ratio [0.99936, 1.00081], row 0.00306. x0.1: rel 0.00169, row 0.00190. x30: rel
  0.00169, ratio [0.99951, 1.00081], row 0.00183.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_norm.py

## S.moe_shared.10 test (attempt 1)

What
- Replaced the rendered swap test (moe_shared layer 2, attn_hc .. ffn_hc_pre + ffn_norm on device) with swap 09
  (test_swap_moe_shared_09_ffn_hc_pre.py). Every check and limit is kept. Added the ffn_norm checks of
  test_swap_moe_full_10_ffn_norm.py (`_errors`, `fn_check`, FN_* limits). vs golden: rel <= 0.01, ratio [0.99, 1.01],
  row <= 0.03. vs the CPU ffn_norm on the device ffn_x: 0.008 / [0.993, 1.007] / 0.015. The module on ffn_x x 0.1 vs
  CPU: 0.01 / row 0.02. On x 30: 0.006 / [0.993, 1.007] / 0.015.

Decisions
- Limits unchanged; they are the layer-2 component test's (test_c_moe_shared_ffn_norm.py). A layer-2 CPU block study
  (/tmp/hy4_ssh10/study.py = /tmp/hy4_sm10/study.py with L = 2 and shared_topk from the golden; log study.log) gave
  these results. 15 of 24 ffn_norm mutations pass the 0.98 out gate. Swap 09's out / tail rel checks catch 11 of
  them. eps 1.2e-5, RMS over half the columns (out rel 0.0099), LayerNorm and SP rows 1023 / 1024 swapped pass every
  out check, and each one fails an ffn_norm check. The bf16-everywhere estimate passes with about 2x margin.

Results
- BRINGUP_IMPL=reference: PASS (pcc_swap_out 0.999995; ffn_norm vs CPU, x0.1 and x30 all 0).
- BRINGUP_IMPL=stub: FAIL (out PCC 0 and every extra check).
- Gate (device, TtGatheredRmsNorm; nothing to implement, the module already runs for this block type): PASS.
  pcc_swap_out 0.999978. ffn_norm vs golden 0.00518 [0.99703, 1.00073] row 0.0097; vs CPU 0.00174
  [0.99914, 1.00065] row 0.0019; x0.1 0.00169 / 0.0019; x30 0.00169 / 0.0019. Tail 0.0026, router 0.99335, out rel
  0.00670.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_10_ffn_norm.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_10_ffn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_10_ffn_norm.py

## C.moe_shared.router test (attempt 1)

What
- Replaced the rendered router test (moe_shared layer 2) with the reviewed layer-1 router test
  (test_c_moe_full_router.py) at LAYER = 2, with the same limits: gated pcc_router_L02 (0.99), not a CPU bridge,
  finite, 8 nonzeros per row, non-negative, overlap >= 0.995 vs golden and >= 0.996 vs the CPU step, matched-row rel L2
  <= 0.005 / 0.004, row sum / 2.827 within 0.004. Added the worst per-row overlap vs the CPU step >= 0.75 (from swap
  S.moe_full.11), which catches a row permutation that the mean checks miss.
- CPU mutation study on the layer-2 golden in /tmp/hy4_router2/{study,mut,mut2}.py (outside the repo; logs mut.log,
  mut2.log). The table is in the test docstring.

Decisions
- Kept the layer-1 limits after re-measuring on layer 2 (known issue: router mutations score differently per layer).
  The layer-2 bias is wider (-0.147..0.024) and the median 8th / 9th gap is smaller (0.0017), so the bf16 stages score
  lower than at layer 1 (choice keys bf16 overlap 0.98395, sigmoid bf16 0.98792, logits bf16 0.98969). All of them fail
  the overlap limits. A bf16 bias is harmless (0.99841).
- Worst-row limit 0.75: adjacent rows share up to 5 of 8 experts at layer 2, so a swapped row scores 0.625. The fp32
  device router scores 0.875 (a near-tie flip).

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.998961, overlap 0.99823 / vs CPU 1.0, matched rel 0.00172 / 0, row sums 1.0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device, TtHy4Router from the moe_full implement step already covers layer 2): PASS. pcc_router_L02 0.998978,
  overlap 0.99823 vs golden, 0.99988 vs CPU (2046/2048 rows matched), worst row 0.875, matched rel 0.00173 / 0.00006,
  row sums 1.00000.

Gotchas
- The first `FAIL pcc_router_L02: pcc=0.000000` line in the log comes from the precompile collect pass. Ignore it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_router.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_router.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_router.py

## S.moe_shared.11 test (attempt 1)

What
- Replaced the rendered swap test test_swap_moe_shared_11_router.py (moe_shared layer 2, steps 1-11, last router).
  It is now test_swap_moe_shared_10_ffn_norm.py with "router" added to SWAPPED and every swap-10 check kept at its
  limits. The router block is taken from test_swap_moe_full_11_router.py: 8 nonzeros per row, non-negative, row sum /
  2.827 within 0.004; vs golden overlap >= 0.98 and matched-row rel L2 <= 0.01; vs the CPU router on the same device
  ffn_norm overlap >= 0.996, matched rel <= 0.004, worst row overlap >= 0.75.
- CPU mutation study on the layer-2 golden: /tmp/hy4_ssh11/study.py (outside the repo). The table is in the docstring.

Decisions
- Kept swap 10's golden-overlap floor at 0.98, not layer 1's 0.99. Layer 2 routing is closer to ties, and with the
  device ffn_norm the CPU router scores only 0.99335. The vs-CPU checks catch the precision bugs instead: every bf16
  stage scores <= 0.98907 against 0.996.
- Every mutation in the table fails at least one check. bf16 output and bf16 bias pass, and both are harmless.

Results
- BRINGUP_IMPL=reference: PASS (out 0.999995; router vs golden 0.99725 / rel 0.00173, vs CPU 1.0).
- BRINGUP_IMPL=stub: FAIL (every check).
- Gate (device): PASS. pcc_swap_out 0.999977; router vs golden 0.99335 (1939 matched, rel 0.00206); vs CPU 0.99963,
  worst row 0.875, rel 0.000059; row sums 1.00000; tail 0.0025; out rel 0.00674.

Gotchas
- The first `FAIL pcc_swap_out: pcc=0.000000` in the log comes from the precompile collect pass. Ignore it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_11_router.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_11_router.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_11_router.py

## C.moe_shared.experts test (attempt 1)

What
- Replaced the rendered experts test (moe_shared, layer 2) with the reviewed layer-1 test (test_c_moe_full_experts.py)
  at LAYER = 2. It keeps the gated PCC >= 0.99, the no-CPU-bridge assert, the golden checks (finite, element count,
  rel L2 <= 0.015, per-token norm ratio in [0.98, 1.02], coef within 0.004) and the x * 2 probe vs the CPU experts.
  The worst per-token rel L2 limit is now 0.018 (layer 1: 0.03).
- CPU mutation study on the layer-2 golden and weights in /tmp/hy4_exp2 (outside the repo): prep.py, study.py,
  study2.py, study3.py (logs study.log, study2.log). The table is in the test docstring.

Decisions
- Worst row 0.018: dropping expert 180 (1 token) scores 0.0197 on the golden and 0.0299 on the probe, so it passes
  0.03. The next-smallest single-expert drop is 0.106. The device scores 0.0111 / 0.0114 and the all-bfp8 estimate
  0.0163 / 0.0167.
- The clamp never fires on the layer-2 golden (gate max 9.74), so the clamp bugs are detected only by the x * 2
  probe: no clamp 0.205, clamp up only 0.097, limit 9 0.054.
- The other limits are unchanged. The layer-2 device estimate (0.0075 / [0.994, 1.005]) is close to layer 1's.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, golden rel 0.00233 / [0.99637, 1.00349] / 0.00405 / coef 0.99994;
  probe 0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device, TtHy4Experts via _EXPERTS_STEPS, already wired for any layer): PASS. pcc_experts_L02 0.999965;
  golden rel 0.00839, ratio [0.99370, 1.00602], worst row 0.01114, coef 1.00018; x*2 vs CPU rel 0.00799,
  [0.99688, 1.00415], 0.01136, coef 1.00033.

Gotchas
- The first "FAIL pcc_experts_L02: pcc=0.000000" line is the precompile collect pass. Ignore it.
- Layer-2 golden facts for implement / swap: tokens per expert 1..390 (hottest 118), 4 experts with 1 token; pairs
  per chip 4514 / 4079 / 3894 / 3897.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_experts.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_experts.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_experts.py

## S.moe_shared.12 test (attempt 1)

What
- Replaced the rendered swap-12 test (moe_shared, layer 2, experts last) with swap 11's reviewed test plus the experts
  block of test_swap_moe_full_12_experts.py (built by script: SWAPPED + "experts", the EX_* constants, the ex_check
  block before "Block out"). Every swap-11 check is kept at its limits.
- Experts checks: vs the CPU experts on the device ffn_norm + device routing (rel <= 0.015, ratio [0.98, 1.02], worst
  row <= 0.018, coef 1 +- 0.004); the module on ffn_norm x 2 vs the CPU experts (same limits, the only clamp check at
  layer 2); vs golden rel <= 0.03, coef 1 +- 0.004, and on rows routed as in the golden ratio [0.97, 1.03], row <= 0.04.

Decisions
- Worst-row limit 0.018 (layer 1: 0.03), taken from test_c_moe_shared_experts.py: at layer 2 dropping the 1-token
  expert 180 scores 0.0197. The golden limits stay at layer 1's values; they are backstops.
- No new mutation study. The step is the same as at layer 1, and the component test already has the layer-2 table.
- Made ex_check robust: no rows routed as golden now fails cleanly (it used to raise in `min()` on an empty tensor
  under the stub), and a nan coefficient fails the coef check (`not abs(c - 1) <= tol`).

Results
- BRINGUP_IMPL=reference: PASS (out 0.999995; experts vs CPU 0; vs golden 0.0106, 2003 matched rows, row 0.0031).
- BRINGUP_IMPL=stub: FAIL (assertion, every check).
- Gate (device): PASS. pcc_swap_out 0.999974; experts vs CPU 0.00822 [0.99399, 1.00533] row 0.01091 coef 1.00018;
  x2 0.00814 row 0.01146; vs golden 0.0209, 1939 matched rows [0.98971, 1.00695] row 0.0164; out rel 0.00718.

Gotchas
- test_swap_moe_full_12_experts.py (frozen) has the same empty-rows `min()` crash under a broken router. It still
  fails, but with a RuntimeError instead of an assertion.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_12_experts.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_12_experts.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_12_experts.py

## C.moe_shared.shared_expert test (attempt 1)

What
- Replaced the rendered 20-line test with the reviewed test_c_moe_full_shared_expert.py, set for layer 2 (LAYER = 2).
  All checks and limits are the same: gated pcc_shared_expert_L02 >= 0.99; vs golden rel <= 0.008, row ratio in
  [0.99, 1.01], worst row <= 0.015; not a CPU bridge; a second run on scaled input vs the CPU step (rel <= 0.006,
  ratio [0.99, 1.01], row <= 0.012).
- Re-ran the layer-1 CPU mutation study on the layer-2 golden and weights (/tmp/hy4_se2/study{,2}.py, outside the
  repo). The tables are in the test docstring.

Decisions
- SYN_SCALE 3, not 2. The layer-2 input is narrower (gate [-2.91, 5.30], up [-5.65, 5.72]; layer 1: up to 8.07 /
  9.67). At x 2 the clamp at 10 scores only rel 0.0069 / worst row 0.088, just past the limits. At x 3 it scores
  0.0685 / 0.40, against bf16 noise of 0.0034 / 0.0057. That matches layer 1's x 2 (gate [-8.7, 15.9]).
- The golden limits hold at layer 2: bf16 gate/up/h scores 0.0036 / [0.9957, 1.0031] / 0.0062. x 1.01 is caught
  (ratio max 1.0106, rel 0.0102). bfp8 weights score 0.0080, right at the limit. The plan says bf16 weights.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999998, rel 0.00196, ratio [0.99956, 1.00061], row 0.00248; scaled 0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): already PASS. hooks.device_component returns the layer-1 TtDenseMLP shared-expert module for any
  layer: pcc 0.999998, rel 0.00209, ratio [0.99892, 0.99997], row 0.00263; x3 rel 0.00074, row 0.00098.
  The implement step still has to add shared_expert to DEVICE_STEPS["moe_shared"] (and whatever else it needs).
- The first "FAIL pcc_shared_expert_L02: pcc=0.000000" line is the precompile collect pass. Ignore it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_shared_expert.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_shared_expert.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_shared_expert.py

## S.moe_shared.13 test (attempt 1)

What
- Replaced the rendered 34-line test with test_swap_moe_shared_12_experts.py (all checks and limits kept), plus the
  shared_expert block and SE_* constants of test_swap_moe_full_13_shared_expert.py. "shared_expert" added to SWAPPED.
  New checks on shared_out: vs the CPU step on the device ffn_norm (rel 0.008, ratio [0.99, 1.01], row 0.015, coef
  within 0.003); the module on device ffn_norm x 3 vs CPU (0.006 / [0.99, 1.01] / 0.012 / 0.003); vs golden (0.01 /
  [0.985, 1.015] / 0.02 / 0.004).
- CPU mutation study on the layer-2 golden (/tmp/hy4_ss13/study.py, study.log, outside the repo). The table is in
  the test docstring. 20 of 27 mutations pass the 0.98 out gate. The vs-CPU checks catch all of them, except rounding
  and bfp8 weights, which should pass.

Decisions
- SE_SYN_SCALE 3, not 2. This matches the layer-2 component test: at x 2 the clamp barely fires (rel 0.0069). At x 3
  the clamp scores 0.068 / worst row 0.40.
- Limits are the same as layer 1. The tightest margin is bfp8 weights at x 3, worst row 0.0110 of 0.012. The plan
  says bf16 weights, so this does not affect the device module.

Results
- BRINGUP_IMPL=reference: PASS (out 0.999995; shared vs CPU 0, x3 0; vs golden 0.00217 row 0.0036).
- BRINGUP_IMPL=stub: FAIL (pcc_swap_out below 0.98 plus every check).
- Gate (device): PASS, pcc_swap_out 0.999974. Shared vs CPU 0.00073 [0.99916, 0.99963] row 0.0010 coef 0.99945.
  x3 0.00074. vs golden 0.00377 [0.99520, 1.00482] row 0.0115. Tail 0.0035, out rel 0.00717.
- The first "FAIL pcc_swap_out: pcc=0.000000" line is the precompile collect pass. Ignore it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_13_shared_expert.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_13_shared_expert.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_13_shared_expert.py

## C.moe_shared.moe_combine test (attempt 1)

What
- Replaced the rendered 20-line test with the reviewed test_c_moe_full_moe_combine.py, set for layer 2 (LAYER = 2).
  All checks and limits are the same: gated pcc_moe_combine_L02 >= 0.99; not a CPU bridge; vs golden rel L2 <= 0.004,
  row ratio [0.995, 1.005], worst row <= 0.01; per addend |coef - 1| <= 0.002, add rel <= 0.008 (shared) / 0.005
  (experts), add row <= 0.03; probes (experts, -shared) and (experts, 0) vs exact sums (rel 0.004, row 0.005).
- Re-ran the layer-1 CPU study on the layer-2 golden (/tmp/hy4_mc2/study.py and mut.py, outside the repo). The table
  is in the test docstring.

Decisions
- Kept the layer-1 limits. At layer 2 the addends are balanced (||experts|| 1769, ||shared|| 1669, ||mlp_out|| 2646;
  layer 1: 6479 / 3630). A bf16 output scores rel 0.00276, add rel 0.0028 / 0.0026, add row 0.013 / 0.019 (the
  tightest margin, 0.019 of 0.03).
- The blind spot shrinks: 0.997 x shared now fails the experts add-row check (0.033). 1.005 / 0.995 x experts, 1.01 x
  shared, the zeroed row / columns and 2x fail rel L2. A cached output and a + golden shared fail the probe. The stub
  fails PCC.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999997, rel 0.002252, probes 0).
- BRINGUP_IMPL=stub: FAIL (PCC below threshold).
- Gate (device): already PASS. device_component returns the layer-1 TtMoeCombine for any layer: pcc 0.999997, rel
  0.002252, row ratio [0.99982, 1.00015], worst row 0.0024, addend coefs 1.000000, probes 0.
  The implement step still has to add moe_combine to DEVICE_STEPS["moe_shared"] (the hybrid).
- The first "FAIL pcc_moe_combine_L02: pcc=0.000000" line is the precompile collect pass. Ignore it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_moe_combine.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_moe_combine.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_moe_combine.py

## S.moe_shared.14 test (attempt 1)

What
- Replaced the rendered 30-line swap test with test_swap_moe_shared_13_shared_expert.py plus "moe_combine" in SWAPPED
  and the moe_combine block of test_swap_moe_full_14_moe_combine.py (layer 1): the MC_* limits, the empty-rows guard
  in `_errors`, and the checks (mlp_out vs the exact sum of the block's own device addends, per-addend coefficient /
  error, probes (experts, -shared) and (experts, 0), vs golden on rows routed as the golden). Every swap-13 check and
  limit is unchanged.
- CPU mutation study on the layer-2 golden addends: /tmp/hy4_ms14/study.py, log study.log (outside the repo). The
  table is in the test docstring.

Decisions
- Kept the layer-1 MC limits (they match the layer-2 component test). Every mutation that passes the 0.98 out gate
  (0.997 / 1.003 addend scales, swapped rows, a cached mlp_out, the golden shared_out) fails at least one extra check.
  bf16 input or output rounding passes. Tightest margin: experts addend worst row for a bf16 output, 0.019 of 0.03.
- "golden experts in place of the input" is the identity in the study, because the addends there are the golden's.
  On the device addends (experts vs golden rel 0.021) the probes catch it.

Results
- BRINGUP_IMPL=reference: PASS (out 0.999995, mlp_out vs golden 0.00727, out rel 0.00304).
- BRINGUP_IMPL=stub: FAIL (every check).
- Gate (device, TtMoeCombine): PASS, pcc_swap_out 0.999974. mlp_out vs CPU 0, coefs 1, probes 0. vs golden
  0.01418, coef 0.99941, on 1939 matched rows [0.99408, 1.00491], row 0.0126. out rel 0.00717.
- The first "FAIL pcc_swap_out: pcc=0.000000" line in the log is the precompile collect pass. Ignore it.

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_14_moe_combine.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_14_moe_combine.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_14_moe_combine.py

## C.moe_shared.ffn_residual.test.1 (test review)
- Test `tests/bringup/test_c_moe_shared_ffn_residual.py` rebuilt from the reviewed moe_full layer-1 ffn_residual test
  (same hc_post step: out_j = h_mid_j + post_j * mlp_out, post = ffn_hc cols 4-7), LAYER = 2.
- Layer-2 golden (s4096 chunk 1): stream norms 51 / 42 / 56 / 335, addend norms 3.1 / 4.1 / 2.4 / 199, bf16 budget
  r_j/||t_j|| 0.027 / 0.017 / 0.038 / 0.0038. Every stream's addend is visible (layer 1 stream 2 was not), so the
  limits are tightened to 0.005 (rel L2, addend coef/excess), stream ratio [0.995, 1.005], as attn_residual layer 2.
  Rotated-gate second run kept (rel L2 <= 0.004 vs the CPU step on the same inputs). Variant table in the docstring.
- PCC 0.99 alone misses everything down to "post halved" (0.986); the extra checks catch 1.005 x mlp_out and up.
  Blind spot: uniform post / mlp_out scale <= ~1.004.
- reference: pass; stub: fail (PCC); gate on device: pass, pcc 0.999997, rel 0.0023, addend excess 0 (fp32 output).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_c_moe_shared_ffn_residual.py`

## S.moe_shared.15.test.1 (swap 15 test review, ffn_residual last, layer 2)
- Rendered test replaced by: the reviewed swap-14 file (every check at its limits) + "ffn_residual" in SWAPPED + the
  ffn_residual block and its 5 constants from the layer-1 test_swap_moe_full_15_ffn_residual.py (limits unchanged).
  Checks added: out vs the CPU ffn_residual on the block's own h_mid / ffn_hc / mlp_out (rel <= 5e-4, row <= 1e-3),
  per-stream addend (layer-2 ADD_* limits, 0.005), rotated-post-gate re-run (same limits + stream ratio
  [0.995, 1.005]), out per-stream norm ratio vs golden split by routing ([0.98, 1.02] matched / [0.95, 1.05] flipped).
- CPU study (golden layer-2 inputs, /tmp/hy4_ms15/study.py, outside the repo): the mutations that pass the gate and
  every swap-14 check (1.002 / 1.005 / 0.995 x mlp_out, 0.999 x h_mid, ignored rotated gates, cached golden out) all
  fail a new check. Known gap: uniform mlp_out / post scale error below ~0.1 %. A bf16 output would fail the 5e-4 c
  check (0.0017), as at layer 1; the device TtHcPost is fp32 and bit-identical.
- reference: PASS (out 0.999995); stub: FAIL (every check); gate on device: PASS, pcc_swap_out 0.999974, out vs CPU
  ffn_residual 0, addend coefs 1, rotated 0; out stream ratio vs golden 1939 matched [0.99739, 1.00380], 109 flipped
  [0.98956, 1.03257]; out rel 0.00717 (swap 14: 0.00717).

Re-run
    BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_15_ffn_residual.py
    BRINGUP_IMPL=stub scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_15_ffn_residual.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/hy4_preview_d_p/tests/bringup/test_swap_moe_shared_15_ffn_residual.py

## M.1 assemble (attempt 1)

What
- New `tt/model.py`: TtHy4Embedding (embed_tokens [V, 3072] bf16 per chip, hidden split over columns, tensorbin cache
  `generated/hy4_preview_d_p/tt_cache/embed_tokens_bf16_cols`; ids row-split over axis 0 -> typecast fp32 -> concat x4
  streams), TtHy4Block (the validated modules, built with the same hooks builders `_hc_module`, `_norm_module`, ... as
  the component / swap / hybrid paths; run_block over `Hy4Reference.block_graph`, every step an override, a missing
  step raises), TtHy4DeviceState (per-layer attention / indexer geometries + the latest full layers' device top-k),
  TtHy4FinalNorm (hc_head = TtHcGates with hc_head_fn zero-padded to 8 rows + TtHcPre, then TtGatheredRmsNorm fp32 out),
  TtHy4Model.
- `hooks.py`: Hy4DeviceModel (ladder / profile adapter) is the `device_model` hook; `BRINGUP_HYBRID=1` keeps the hybrid.

Decisions
- Hidden state resident as [1, 1, S/2, 4 x 3072] fp32 per chip (tt/layout.py) end to end; dtypes at every boundary
  match the hybrid's (bf16 attn_norm / q_resid / ffn_norm, fp32 streams / gates / sublayer outputs).
- The router boundary is the device (idx, wts) tuple; experts take it directly (no dense [S, E] round trip).
- Geometry (RoPE tables for max_seq, caches, scratch, dispatch/combine sizes) is built in `new_state` when the spec gives
  one chunk for that seq (`_state_chunk`), else at the first layer call (first chunk, not counted as warm). A golden
  prefix / zeros are written to the caches there too (harness boundary).
- Each boundary is freed after its last reader, except "in" (the caller's) and "topk": the indexer returns its
  geometry's persistent gather buffer (`g.idx_out`), which shared layers read; freeing it would break layers 2-4.
- LM head on the host (fp32, sampled rows only, `lm_head=True` only).

Result (gate, s4096): PASS. pcc_layer L00..L05 0.999984 / 0.999953 / 0.999921 / 0.999917 / 0.999900 / 0.999892,
pcc_state_min 0.999965, host_transfers_per_layer 0, device_model_hybrid 0. Chunks 2.11 s / 7.18 s (chunk 1 includes
program compiles for the new chunk offset); model load 159 s.

Gotchas
- The final norm path (TtHy4FinalNorm) is not exercised by the 0-5 subset ladder (ends_at_last is False); it is built
  and untested.
- `Ctx.length` is the full chunk (per-chip rows x mesh rows).

Re-run
    PYTHONPATH=$PWD BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py
    BRINGUP_HYBRID=1 ... (same command) for the hybrid harness
