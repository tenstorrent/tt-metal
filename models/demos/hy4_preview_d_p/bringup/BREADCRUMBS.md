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
