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
