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

## C.swa.attn_hc implement (attempt 1)

What: first device step of the model. New `tt/` package: `settings.py` (DS41_ table: MATMUL_FIDELITY HiFi4,
FP32_DEST_ACC, HC_TP_SUM gather_add | all_reduce), `common.py` (HiFi4 + fp32-acc compute config; `tp_sum`: exact
partial sum over axis 1), `layout.py` (xing40's chip-major stream layout, sizes from mesh.shape; host helpers are the
harness boundary only), `mhc.py:TtHcMixes` / `build_hc` (attn_hc and ffn_hc). hooks.py: `device_component` (attn_hc /
ffn_hc host fn), `HybridDeviceModel` = `device_model` (CPU reference + DEVICE_STEPS {"swa": {"attn_hc"}}).
Chain per chip: matmul(streams [S/SP, 4H/TP] fp32, chip rows of hc_attn_fn^T padded to 32, HiFi4 fp32 acc) ->
`ttnn.bringup.mhc_pre_xing_pack` (exact SFPU sum x^2 into column 24) -> all_gather dim 1 over axis 1 + ttnn.add in
mesh-column order (K-30: exact, bit-identical on every TP chip) -> x rsqrt(sum / 4H + norm_eps 1e-20) ->
`ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` (consts from deepseek_v3 `build_consts`, n 4, 20 iters,
eps 1e-6) -> concat [pre | post | comb] [S/SP, 24], replicated over TP.
Decisions: K-20 checked: the kernel's sequence (softmax, +eps, column norm, 19 (row, col) pairs, pre sigmoid + eps,
post 2 sigmoid; comb entry (j, k) at column 4j + k, row sums over k) is reference/kernels.py:sinkhorn_split's. The
streams-only input: attn_hc reads x's streams; the pending pre columns are dropped at the boundary (DS41-3).
Result: pcc 0.999999; vs CPU rel 6.5e-4 (limit 3.8e-3), worst column (21, a comb entry) 0.0144 (limit 0.044), median
row ratio -3.5e-4 (the Sinkhorn kernel's SEL / RB / CB matmuls read fp32 at about TF32, K-30); all second inputs pass;
determinism A,B,A bit-identical. Gate ~30 s.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attn_hc.py`

## C.swa.attn_collapse implement (attempt 1)
- `tt/collapse.py:TtHcCollapse(streams [1,1,S/SP,4 x H/TP], pre [1,1,S/SP,>=4]) -> [1,1,S/SP,H/TP]` fp32: 4 x ttnn.slice of
  the streams and of the pre columns, ttnn.multiply + 3 x ttnn.addcmul ([S/SP,1] column broadcast). Local, no CCL, no
  weight, no host work. Pre is the PENDING pre (DS41-3), passed as a separate row-split, TP-replicated tensor (plan.md
  "Residual"); ffn_collapse can reuse the module with attn_hc's columns 0..3.
- fp32 addcmul / multiply run on the SFPU (ternary_program_factory.cpp:1311: FPU only for non-fp32), so K-30's TF32
  read does not apply: device output equals the CPU step (rel 0 on golden, layer24, mixed, small, big).
- hooks.py: `_collapse_host_fn` (harness boundary: streams_to_device + row_split_to_device of x[:, 4H:], back via
  `tt/layout.py:hidden_split_to_host`); DEVICE_STEPS["swa"] += attn_collapse (hybrid device_model).
- Gate: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attn_collapse.py`
  -> pcc_attn_collapse_L00 = 1.000000, PASS.

## C.swa.attn_norm implement (attempt 1)
- `tt/norm.py:TtDistributedRmsNorm` / `build_norm` (attn_norm, weight `layers.<i>.attn_norm.weight`, eps cfg.eps 1e-20):
  per chip [1,1,S/SP,H/TP] fp32 -> `ttnn.bringup.mhc_pre_xing_pack(mix=x[:, :32], x, n=1)` (exact SFPU sum x^2, written
  over column 3 of the mix copy, so x's first tile column serves as mix: no per-call constant) -> slice col 3 ->
  `tp_sum` (all_gather axis 1 + SFPU add, K-30) -> x 1/H + eps -> rsqrt -> x * rstd ([R,1] bcast) * w ([1,H/TP] fp32
  TILE, split over axis 1, built at load). Same split out, fp32, no host work, no CCL besides the stats gather.
- K-12: xing40's rms_norm_pre/post_all_gather chain not used (fp32 stats junk, FPU stats at ~TF32).
- The fp8 act quant (blocks of 32) the components entry mentions is left to the fp8 linears (q_a / kv): the step's
  boundary is the fp32 norm output, and the reference's fp8_linear quantizes its own input. tt/quant.py does not exist yet.
- hooks.py: `_hidden_host_fn` (harness boundary: `tt/layout.py:hidden_split_to_device` in, `hidden_split_to_host` out);
  DEVICE_STEPS["swa"] += attn_norm.
- Gate: pcc 0.999999; vs CPU rel 7.9e-8 (limit 6.1e-3), vs golden rel 1.7e-3; all second inputs pass; A,B,A bit-identical.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attn_norm.py`

## C.swa.q_a implement (attempt 1)
- `tt/quant.py:fp8_qdq` (new; plan.md "Quantizer flips": C.swa.q_a builds it): act_quant_dequant(x, 32) on fp32 TILE
  [.., R, W]: reshape to [R W/32, 32] -> abs -> ttnn.max(dim=-1) -> max(., 1e-4) -> x fp32(1/448) -> scale bits
  (b + 0x7FFFFF) & 0x7F800000, 1/scale bits 0x7F000000 - b -> y = x / s, clamp +-448 -> e4m3 RNE: normals on the int
  bits ((b + 0x7FFFF + lsb20) & ~0xFFFFF), |y| < 2^-6: round(y 2^9) 2^-9 -> sign -> x s -> bf16 (exact).
  Probe (whole 2x2 mesh, FABRIC_2D; script kept out of the tree, K-56): ttnn.max fp32 exact, scalar multiply exact,
  ttnn.round half-to-even, fp8_qdq 0 mismatches vs kernels.act_quant_dequant on randn, wide-range, row-scaled,
  tiny + zero groups, e4m3 ties (normal + subnormal), amax = 448 x 2^k (512 x 2560 each). DS41-13.
- `tt/q_a.py`: `TtAttnInProj` = q(attn_norm) [S/SP, H/TP] bf16 @ the chip's K rows of [wq_a | wkv]^T (bf16, exact
  for e4m3 x 2^k; HiFi4 + fp32 acc, fp32 out) -> `tp_sum` (all_gather axis 1 + SFPU add, K-30) -> [S/SP, 1792] fp32
  replicated over TP (`cols` names the blocks; the kv step slices [1280, 1792)). `TtQa` = fp8_qdq -> proj -> slice
  q -> `TtDistributedRmsNorm(split=False)` q_norm (eps 1e-20, K-12: not ttnn.rms_norm). Out [S/SP, 1280] fp32.
- `tt/norm.py`: `split=False` option (replicated weight, no exchange); default unchanged (attn_norm gate re-run: PASS).
- hooks.py: `_hidden_to_rows_host_fn` (hidden-split in, row-split TP-replicated out), DEVICE_STEPS["swa"] += q_a.
- Gate: pcc 0.999974; vs CPU rel 1.0e-4 (limit 7.0e-3; summation order only, q(x) is bit-exact), vs golden 7.3e-3
  (limit 1.4e-2); second inputs rel <= 2.0e-4; A,B,A bit-identical. ~25 s.
- Perf later: fp8_qdq is ~25 eltwise ops + a TILE reshape per call; a fork of per_token_cast_to_fp8 (block 32, RNE)
  is the alternative (components.yaml header); linear uses the auto config (K-51).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_q_a.py`

## C.swa.kv test review (attempt 1)
- Previous freeze-without-review failed: the auto sweep let noise1e-2 slip (kv is fp8-quantized: the bf16 precision
  model already has rel 0.0108, so the calibrated rel limit is 0.0216 and 1 % noise, rel 0.010, passes). DS41-14, K-36.
- Added to the test (CHECKS="auto" kept): fp8 structural checks on every auto input (golden, layer24, mixed, small,
  big) vs the CPU step on the same inputs: off-grid fraction == 0 (act_quant_dequant(out, 32) == out) and exact
  mismatch fraction <= 0.15 (bf16 precision model <= 0.043). Metrics `fp8_kv_L00_<input>_{offgrid,mismatch}`.
- CPU probe (outside the tree) of structural mistakes vs auto + fp8 checks: RTZ e4m3 (mismatch 0.49), no output
  quant (off grid 0.94), non-pow2 scale (off grid 1.0), attn_norm not fp8-quantized (mismatch 0.30; auto rel catches
  only 3/5 inputs), neox / inverse / first-dims / no rope, positions 0 or +1, no kv_norm weight, eps 1e-3: all caught.
  Block 128 instead of 32 gives identical values on these inputs (not a distinguishable mistake here).
- BRINGUP_IMPL=mutations runs the test's own sweep (built-in checks + fp8 checks): every standard mistake caught,
  reference / bf16 controls pass. reference PASS, stub FAIL, mutate:noise1e-2 FAIL. Device gate: NotImplementedError
  (no kv device module yet; the implement step writes it).
- For the implementer: the device output must be bit-on the fp8 grid (use tt/quant.py:fp8_qdq, RNE; the stock e4m3
  packer rounds toward zero and fails the mismatch check), and the test builds the device module twice (auto + fp8).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_kv.py`

## C.swa.kv test review (attempt 2)
- Attempt 1 failed only on the knowledge check: DS41-12 (supervisor's RESOLVED note, committed) was 614 chars. Shortened
  the bullet to 448 chars; the resolution and the alternatives moved to `known_issues_detail.md` (section titled like
  the bullet, as knowledge/check.py:136 requires). `knowledge.check` now clean.
- Test unchanged from attempt 1 (built-in auto checks + fp8 grid / mismatch checks, DS41-14, K-36). Re-verified:
  reference PASS (pcc 0.999835, fp8 offgrid 0 / mismatch 0 on all 5 inputs), stub FAIL, mutations PASS (real pass:
  every mistake caught, noise1e-2 by the fp8 checks; the collect pass's "reference FAIL" lines are the precompile
  stash, K-52). Device gate: NotImplementedError (no kv device module yet).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_kv.py`

## C.swa.kv implement (attempt 1)
- `tt/kv.py:TtWindowKv(x attn_norm [1,1,S/SP,H/TP] fp32, start)` -> window KV [1,1,S/SP,512] bf16, rows split over
  axis 0 (row r = positions [start + r S/SP, ...)), replicated over TP: fp8_qdq -> `TtAttnInProj({"wkv"})` (HiFi4 +
  fp32 acc, tp_sum exact, K-30) -> slice -> `TtDistributedRmsNorm(split=False)` kv_norm (K-12) -> RoPE -> fp8_qdq.
  `from_proj(p, start)` takes the shared [wq_a | wkv] projection output (assemble: one exchange for q_a and kv).
- RoPE (`tt/rope.py`, DS41-15): not rotary_embedding_llama (components.yaml) - its FPU multiplies read fp32 at ~TF32
  (K-30) and the fp8 q-dq after it amplifies flips (DS41-1). wkv's rope rows and the kv_norm weight are permuted at
  load to [nope | even | odd] (`deinterleave_perm`), the rotation is fp32 SFPU mul / sub / add on two 32-wide blocks,
  then a ROW_MAJOR transpose [R,2,32] -> [R,32,2] restores the interleaved order before the quant (its blocks of 32
  are the reference's). cos / sin: `RopeTable` = the reference's own fp32 table (torch.polar) for [0, max_seq) on
  device, replicated; per chunk `ttnn.slice` at [start, start + S) + `ttnn.mesh_partition` over SP. No host work.
  max_seq = the spec's longest ladder / target seq (hooks `_max_seq`; 55296 here, 14 MB per rope type).
- Probe (2x2, FABRIC_2D, script outside the tree, dir deleted K-56): RM transpose / permute / TILE transpose of
  [1,R,2,32] fp32 bit-exact; fp32 tensor x tensor mul / sub / add equal torch; slice + mesh_partition gives row r the
  natural block.
- `tt/q_a.py:build_q_a` now builds its projection from `tt/kv.py:attn_in_weights` (wkv permuted; q_a slices only
  wq_a). q_a gate re-run: PASS, identical metrics (pcc 0.999974, rel vs CPU 1.0e-4).
- hooks.py: `_kv_host_fn` (hidden-split in, row-split TP-replicated out, positions from ctx.start), `_max_seq`,
  DEVICE_STEPS["swa"] += kv.
- Gate: pcc 0.999834; vs CPU rel 2.7e-3 (limit 2.2e-2); fp8 checks: off grid 0 on all 5 inputs, mismatch vs CPU
  0.0084 / 0.0093 / 0.0114 (limit 0.15; from the GEMM's fp32 sum order only: RoPE, norm and quant are exact);
  A,B,A bit-identical. ~37 s.
- Not in this step: the served 0_window_kv write (all_gather axis 0, owner-chip slice_write of [actual_start,
  actual_end), pad zeroing before the ack). The kv step's reference is stateless and the window state is still the
  CPU ring the CPU attention writes; per-chip write offsets differ for starts not a multiple of W (SPMD ops take one
  offset), so the write needs the device state / position tables the attention (or contract) step builds.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_kv.py`

## S.swa.05 implement (attempt 1)
- Previous attempt failed only the tail check: swap_kv_out rel 0.01085 > 0.01 (block out with the device kv vs with
  the CPU kv). The kv module already existed (C.swa.kv); hooks already registered kv in DEVICE_STEPS["swa"].
- Cause (DS41-16, corrects DS41-15's "sum order"): Blackhole's FPU rounds each in-tile partial dot (32 products) to
  ~TF32 before adding it to fp32 DEST, even with bf16 operands. Probes (2x2, FABRIC_2D): 1 + 2^-12 in one tile -> 1,
  1 + 2^-11 -> 1 + 2^-10; across K tiles exact fp32; ttnn.linear of e4m3-grid operands rel 1.8e-4 vs fp64 at
  LoFi/HiFi2/HiFi4, any K, any in0_block_w, with or without packer_l1_acc; one nonzero per tile-dot: 3.5e-9. That
  1.8e-4 flipped 0.84 % of the window KV's fp8 values (DS41-1), which the MoE after attention amplifies to 1 %.
- Fix: `tt/exact_mm.py:ExactMatmul` (new): x * mask_u (mask_u = [k % 32 == u], built at load, [1, 8, 1, K] per group)
  -> one batched matmul per group of 8 (explicit MatmulMultiCoreReuseMultiCastProgramConfig, out_block_h <= 8, fp32
  DEST subblock <= 4 tiles, cached per shape; grid from the mesh) -> fp32 SFPU adds (groups, then a pairwise tree).
  Bit-identical on every chip, rel 5e-8 vs fp64 (torch fp32: 6.6e-8). packer_l1_acc changes bits: not used.
  Cost per chip: [1024, 2560] x [2560, 1792] 7.9 ms vs 0.39 ms plain; [512, 1280] x [1280, 1792] 2.8 ms.
- `tt/q_a.py:TtAttnInProj` uses it when settings ATTN_IN_MM == "exact" (default and max_precision; "fpu" = the old
  ttnn.linear, a perf pick). Load-time assert: weights on the e4m3 x 2^k grid (low 20 fp32 mantissa bits zero), so
  products have <= 8 significant bits. q_a and kv share it (the assembled model's one projection).
- Gate: PASS. pcc_swap_out 0.999972; kv vs CPU rel 0.0 (worst row 2e-6); q_a vs CPU rel 0.0; swap_kv_out rel
  1.5e-4 (limit 0.01). Re-ran C.swa.kv (PASS, fp8 mismatch vs CPU 3e-6 on all 5 inputs, was 0.0084) and C.swa.q_a
  (PASS). Probe dir deleted (K-56).
- Perf note for later: the exact projection is ~20x the plain matmul; ATTN_IN_MM=fpu restores it (judge end to end).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_05_kv.py`

## C.swa.attention implement (attempt 1)
- `tt/attention.py` (new): `TtWindowAttention(q_resid [1,1,S/SP,1280] fp32, kv [1,1,S/SP,512] bf16, start, end, cache)
  -> attn_out [1,1,S/SP,H/TP] fp32`; `TtWindowCache` (served 0_window_kv per layer and slot: init_kvpe_cache bf16
  ROW_MAJOR, ND shard per 32-token record, TP-dedup); `WindowTables` (load-time, shared per model: server-order row
  index per (start mod S)/32, attention index lists per start class 0/32/64/96/>=128, pad-mask iota, zero carry).
  Chain: kv all_gather axis 0 -> rows >= end zeroed -> cache write (embedding into server order +
  update_padded_kv_cache(kv_actual_global=start, valid_global=end, tp_axis=1)) -> carry [start-128, start) read from
  the cache (same local slice on every chip, all_gather both axes, owner block) -> table [carry | chunk] RM, tail
  de-interleaved -> q: fp8_qdq -> wq_b linear (heads over TP, rope rows de-interleaved at load) -> heads -> rope_heads
  (fp32 SFPU) -> x 512^-0.5 -> bf16 -> all_to_all head->seq -> ttnn.bringup.sparse_sdpa(high_precision, sink, scale 1,
  128 ids per query) -> all_to_all back -> fp32 -> inverse RoPE + re-interleave -> fp8_qdq -> nlp_concat_heads per
  group -> batched wo_a -> fp8_qdq -> wo_b K-split -> tp_sum gather_add -> mesh_partition. No host work per chunk.
- Fork `ttnn/ttnn/bringup/sdpa`: sparse_sdpa high_precision now takes attention_sink (kernel: unpack reconfig to the
  bf16 max / sink before the subtraction; the Float32 row-sum format was still in srcB: rel 0.17-1.39 without it;
  sink exp scale rounded like the scores'). 4 new unit cases; regression: unit 32 passed / 10 skipped, fork_source
  216 tests 0 regressions, model cases 5 passed / 1 skipped (as recorded). CHANGELOG + INDEX updated. K-31 now out of
  date (reported).
- Settings: WINDOW_SDPA (high_precision | source), ATTN_O_SUM (gather_add | reduce_scatter); max precision = defaults.
- `tt/rope.py`: `rope_heads` (per-head [B,H,R,W], forward / inverse, optional re-interleave), `interleave_tail`.
- hooks.py: `_attention_host_fn` (component boundary: makes a cache for ctx.extra max_seq and loads the golden ring),
  DEVICE_STEPS["swa"] += attention; hybrid model: `_HybridState` holds a `TtWindowCache` per device-attention layer
  (to_torch / load_prefix map the reference's ring onto it; `served`, `load_served`, `load_tokens`, `formats`, `free`),
  `embed(tokens, start=)` clamps PAD ids (K-72), `from_host(h, start=)`, `layer(..., end=)` passes end and the cache.
- Decisions: scale folded into q (K-27); kv table in q's de-interleaved rope order (DS41-15) instead of interleaving
  q; plain linear for wq_b (DS41-16: q is cast to bf16 anyway); exact wo_b sum (K-30).
- Probes (2x2, FABRIC_2D; dir deleted, K-56): update_padded_kv_cache with tp_axis=1 from server-order rows equals
  the plan's position map at start 960 (bit-exact, pad rows zero); ttnn.slice on the ND-sharded RM cache works;
  ttnn.embedding with per-chip index rows; all_to_all head<->seq has the natural block semantics; fp32 TILE
  transpose(1,2) exact.
- Gate: component pcc 0.999752; vs CPU rel 0.0212 (limit 0.0292), worst row 0.0355, bias -0.0009; second inputs
  rel <= 0.022 (limit 0.03); A,B,A bit-identical. CPU emulation of the device roundings predicted 0.0206 (DS41-18:
  the bf16 sdpa output is most of it). test_cache_starts: 4 / 4 scenarios, 0 failures (~2.3 min).
- Next steps: the ladder's state check reads window_kv as the ring view of the device cache; the cache is sized by
  the spec's target.chunk (WindowTables asserts S == chunk).
- Re-run: `PYTHONPATH=$PWD BRINGUP_TASK=C.swa.attention scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attention.py && PYTHONPATH=$PWD BRINGUP_TASK=C.swa.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`

## S.swa.06 implement (attempt 1)
- Previous attempt (orchestrator gate run of the C.swa.attention module) failed two step checks: attention vs the CPU
  step rel 0.0212 > 0.02 and the tail (block out with the device vs the CPU attention output) 0.0163 > 0.01.
- CPU emulation of the device roundings (outside the tree; reference block on the golden chunk 1 of layer 0): the
  fork path reproduced (0.0207 / 0.0158). fp32 output alone: 0.0126 / 0.0132; fp32 operands but the FPU's in-tile
  dot rounding (DS41-16) in both matmuls: 0.0095 / 0.0106; exact products, q / probs as 3 bf16 pieces: 0.0004 /
  0.0015; plus the device's plain wq_b / wo_a / wo_b matmuls: 0.0039 / 0.0070 (exact wq_b would give 0.0059).
- Fix (DS41-19): `tt/window_exact.py:ExactWindow` (new), setting WINDOW_SDPA = "exact" (default and max precision;
  "high_precision" / "source" keep the sparse_sdpa paths), WINDOW_PIECES = 3 (max; 2 = 16 bits). Per chip, its own
  heads (TP) and rows (SP), no all_to_all: q fp32 -> blocks of 32 queries [1, nb, Hc 32, 512]; keys = 160 table rows
  per block (5 shifted 32-row slices of the replicated [carry | chunk] table, mesh_partition over SP); scores and
  P @ [V | ones] as lane matmuls (operand x0/1-masked to K columns k % 32 == u by one broadcast multiply, L lanes
  repeated along K so DEST sums them, G along N summed on the SFPU; L, G from a 256 MB transient cap); band bias
  (0 / -1e30) per start class 0/32/64/96/>=128 built at load, sharded over SP; fp32 max / exp / sink (fp32, was
  bf16 in the fork path: not bf16-exact) / divide. Output fp32 heads -> the existing inverse RoPE / fp8 / wo_a / wo_b.
- Probes (2x2 FABRIC_2D, dir deleted, K-56): lane-masked matmul rel 5e-9 vs fp64 (plain 3e-4); a single bf16 x e4m3
  product (12 bits) in a tile-dot is exact (DS41-16 rounds the in-tile sums, not single products); bit-round, broadcast
  add over dims 0 / 1, permute (1,0,2,3), fp32 exp (rel 7e-8, exp(-1e30) = 0), fp32 max: exact. ExactWindow vs fp64
  at starts 0/32/64/96/128/960: rel 4e-7 (3 pieces), 3.7e-6 (2). Time per call (S 2048, 32 heads / chip): 64 ms
  (3 pieces), 45 ms (2); was 101 ms with lanes only along N (32x fp32 output + slices).
- Gate: PASS. pcc_swap_out 0.999958; attention vs CPU rel 0.0039 (limit 0.02), row 0.013; tail 0.00798 (limit 0.01;
  the plain wq_b projection is most of the rest). C.swa.attention component: PASS (vs CPU 0.0039, second inputs
  <= 0.0051, A,B,A bit-identical). test_cache_starts with BRINGUP_TASK=C.swa.attention: 4 / 4, 0 failures (without
  BRINGUP_TASK it runs all layers and fails on layer 1, not on the device yet).
- Perf for later: 64 ms / layer on 2x2 (half on 4x4); levers: WINDOW_PIECES=2, matmul program configs (the batched
  lane matmuls run at ~25 TFLOP/s), or WINDOW_SDPA=high_precision (judge end to end).
- OV-1: DS41-12 shortened to one line (352 chars); the detail heading renamed to match.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_06_attention.py`

## C.swa.attn_residual implement (attempt 1)
- `tt/residual.py` (new): `TtHcResidual(x streams [1,1,S/SP,4 x H/TP] fp32, mixes [1,1,S/SP,24] fp32 row-split +
  TP-replicated, y [1,1,S/SP,H/TP] fp32) -> h_mid streams [1,1,S/SP,4 x H/TP] fp32`. Local per chip: no CCL, no
  weight, no host work. post = mixes[:, 4:8], comb = mixes[:, 8:24] (row-major (j input, k output)); DS41's
  out_k = post_k y + sum_j comb[j*4 + k] x_j is `ttnn.bringup.mhc_post`'s DEFAULT (comb_transposed=True), unlike
  xing40 (comb_transposed=False).
- Setting RESIDUAL_MIX: fused (default, mhc_post in one program) | addcmul (multiply + 4 x addcmul per stream).
  Both fp32 SFPU; both give identical numbers on the gate (the fused kernel keeps the chain's rounding order). Not a
  precision switch, so not in MAX_PRECISION.
- hooks.py: `_residual_host_fn` (harness boundary: streams_to_device(x) drops the pending pre, row_split_to_device(mix),
  hidden_split_to_device(y), streams_to_host back); DEVICE_STEPS["swa"] += attn_residual. ffn_residual can reuse the
  module (then pack mix[:, :4] as the pending pre).
- Gate: PASS. pcc 0.999998; vs CPU rel 1.7e-8 (limit 0.0049); vs golden 0.0022; second inputs rel <= 3.5e-8;
  A,B,A bit-identical. The FAIL lines with pcc 0 before the real pass are the precompile collect pass (K-52).
- OV-1: already done (DS41-12 is one line, 352 chars).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attn_residual.py`
  (DS41_RESIDUAL_MIX=addcmul for the op chain).

## C.swa.ffn_collapse implement (attempt 1)
- OV-3 first (owner 2026-10-15, accuracy against the noise floor): tt/settings.py defaults and MAX_PRECISION now
  ATTN_IN_MM = fpu (one ttnn.linear, HiFi4 + fp32 acc) and WINDOW_SDPA = high_precision
  (ttnn.bringup.sparse_sdpa(high_precision=True, attention_sink=...)); exact paths stay selectable (DS41_ATTN_IN_MM=exact,
  DS41_WINDOW_SDPA=exact), the decision noted in each Setting's why. Docstrings of tt/q_a.py / tt/attention.py updated.
  MAX_PRECISION had to change too: component / swap tests run at max precision (testing/model_precision.py), so with
  "exact" there the swap tests would never see the fast defaults.
- Re-runs with the fast defaults: S.swa.05 PASS (pcc_swap_out 0.999929; swap_kv_out rel 0.010850 <= limit 0.045857;
  swap_q_a step rel 1.0e-4, swap_kv step rel 0.00236). S.swa.06 FAIL on one step check only: swap_attention vs CPU
  rel 0.021227 > 0.02000 (fixed cap swap_step_rel; the component error is 0.02227); swap_attention_out rel 0.016893
  <= limit 0.039237 (F72 tail), pcc_swap_out 0.999864. DS41-20. No exact kernel added (OV-3).
- ffn_collapse: reuses tt/collapse.py:TtHcCollapse unchanged (slice + multiply + 3 x addcmul, fp32 SFPU, local per
  chip, no CCL) with attn_hc's mixes as ``pre`` (DS41-3: the FFN collapses with this layer's attention pre).
  hooks.py: `_ffn_collapse_host_fn` (streams_to_device(h_mid [S, 4H]), row_split_to_device(attn_mix [S, 24]) -> module
  reads columns 0..3 -> hidden_split_to_host); `_device_step_fn` "ffn_collapse"; DEVICE_STEPS["swa"] += ffn_hc
  (its C gate passed but it was never added), ffn_collapse.
- Gate: PASS. pcc 0.999997; vs CPU rel 0 (bit-exact), vs golden 0.00254 (limit 0.0070); second inputs rel 0.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_ffn_collapse.py`

## C.swa.ffn_norm implement (attempt 1)
- tt/norm.py: `TtDistributedRmsNorm(..., gather=True)` for ffn_norm (build_norm: NORM_WEIGHTS["ffn_norm"] =
  ffn_norm.weight, GATHERED_NORMS). Same exact fp32 SFPU chain as attn_norm, then one `ttnn.all_gather(dim=3,
  cluster_axis=1)` of the fp32 output -> [1, 1, S/SP, H] fp32 per chip, replicated over the mesh row (a data move:
  exact). No gather when TP == 1. `TtDistributedRmsNorm.quant(y)` = tt/quant.fp8_qdq (DS41-13) -> q(x) bf16 for the
  shared / routed experts; not called in the step (the boundary is the unquantized fp32 rows the router reads); the
  assembled model computes it once per layer from the gathered rows.
- hooks.py: `_device_step_fn` "ffn_norm" -> `_hidden_to_rows_host_fn` (hidden-split in, row-split TP-replicated out,
  read from mesh column 0); DEVICE_STEPS["swa"] += ffn_norm.
- Gate: PASS. pcc 0.999997; vs CPU rel 7.5e-8 (limit 0.0060); vs golden 0.00234 (limit 0.0083); second inputs rel
  <= 7.6e-8; A,B,A bit-identical.
- OV-4: DS41-20 marked resolved by F73 (component.py:350-356: limit = swap_step_calib x component error, no cap).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_ffn_norm.py`

## C.swa.router implement (attempt 1)
- tt/router.py:TtRouter (from xing40_a4b_d_p/tt/router.py, mimo's ttnn.topk path): input ffn_norm's gathered rows
  [1, 1, S/SP, H] fp32 (row split, TP-replicated) -> ttnn.linear with the bf16 gate as stored (HiFi4 + fp32 acc, fp32
  out) -> softplus -> sqrt -> + (bias - mean(bias)) fp32 SFPU (K-23; bias 9.59..9.88 on layer 0, recentred at load)
  -> ttnn.topk(6) on fp32 keys -> gather of the unbiased scores -> / ((sum + 1e-20) / 1.5) -> scatter into fp32
  ROW_MAJOR zeros (K-4). Returns (dense, idx, weights) for dispatch. No CCL; zeros built once for chunk / SP rows.
- The components entry's "x as bf16 hi + lo" fallback was tried and dropped: identical flips and error, 40 % slower
  (DS41-21; tt-probe ds41_router, not kept: TtRouter(x_mode fp32 / split) on the layer-0 golden ffn_norm vs a
  float64 _route, rows whose expert set differs and rel L2 of the weights on the rest, 10-call timing).
- hooks.py: `_router_host_fn` (row_split_to_device(x) -> dense -> row_split_to_host), `_device_step_fn` "router",
  DEVICE_STEPS["swa"] += router.
- Gate: PASS. pcc 0.999779; vs CPU overlap 0.99967 (4 of 2048 rows flipped, near ties), rel 4.5e-5 on matched rows;
  vs golden overlap 0.99902; layer24 / mixed overlap 0.99967 / 0.99943; A,B,A bit-identical.
- OV-4: already done by C.swa.ffn_norm (DS41-20 marked resolved by F73).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_router.py`

## C.swa.experts test review (attempt 1)
- Freeze without review failed: the CPU reference failed `auto_experts_L00_vs_golden` (worst row 0.0798 > 0.0776,
  row norm ratio [0.976, 1.058] outside 1 +- 0.0498); pcc 0.99986 and every other check ok.
- Cause (DS41-22): the golden is the fp32 reference output stored bf16, its inputs (ffn_norm, dense router weights) bf16.
  The fp8 act quants of x and h around the fp4 GEMMs turn the input rounding into whole e4m3 steps (DS41-1, DS41-14):
  fp32 step on golden inputs vs golden rel 0.0169 / row 0.0798; inputs jittered by half a bf16 ulp vs the step rel
  0.0168, row 0.054-0.075, ratio down to 0.945. Not a reference bug. The built-in checks add this golden error to the
  vs-golden rel and column limits only.
- Test change (CHECKS="auto" kept): `_Expect(CC.Expect)` also adds the CPU step's own worst-row error and ratio
  deviation vs the golden to the vs-golden row / ratio limits (now 0.157 / 0.108); installed as `CC.Expect` for the
  call only. vs_cpu and the second inputs keep the built-in limits (golden rel 0.0150, row 0.0776, ratio 0.0498).
- Mutation sweep: every mistake caught except noise1e-2. Not fixable by precision limits: the bf16 precision model is
  rel 0.0125 / row 0.065 on the golden (0.014-0.018 on the second inputs), 1 % noise is rel 0.010 / row 0.013-0.035,
  so it sits below a correct bf16 step's floor (owner rule: accuracy against the noise floor). The output has no grid
  (fp32 sum of GEMMs). Only the row-error-vs-row-norm slope separates them (model 0.65-0.96, noise ~0); not added, a
  statistic, not a structural property. The freeze after a review needs reference PASS + stub FAIL only.
- Results: reference PASS (pcc 0.999857, vs_golden rel 0.0169 / row 0.0798 within 0.0320 / 0.157), stub FAIL.
- For the implementer: a correct device is expected well below the model (gate/up at HiFi4 + fp32 DEST round ~1e-4,
  not 2e-3, so fewer fp8 flips in h); the vs_cpu rel limit is 0.015.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_experts.py`

## C.swa.experts implement (attempt 1)
- tt/experts.py:TtExperts, the components entry's interim loop path (setting EXPERTS_IMPL=loop; the
  unified_routed_expert_ffn fork option that decodes in the op is still to come). Pipeline as xing40's TtExperts:
  masked_bincount -> ttnn.bringup.offset_cumsum / dispatch / combine along axis 0, one dispatch group per mesh column
  (2x2: 192 experts per group, 96 per chip), cap = the whole chunk, dispatch buffer K = 6 x cap (all 6 on one chip).
- Weights: the checkpoint's e2m1 codes + e8m0 scales only (DS41-10, never bfp8 / bfp4). Each W [N, K] is held as
  W^T in 8 nibble planes (int32 [K/8, N], a lossless re-pack at load, `pack_planes`) + bf16 2^e scales [K/32, N].
  Per chunk and expert: (P >> 4i) & 15 -> ttnn.gather from the 16-value e2m1 table -> concat -> x (Rep @ scales).
  Probed bit-exact vs reference/weights.py:fp4_dequant (0 of 11.8M differ), 4.0 ms per matrix (arith variant 6.4 ms,
  setting EXPERTS_DECODE) (DS41-24). Load of layer 0: ~12 s (page-cached checkpoint), ~1.9 GB per chip.
- Routing weight before q(h) (reference _swiglu; q(r h) != r q(h)): dispatch metadata has only [src chip, token,
  slot], so the dispatched row is [q(x) | p0 | p1 | p2]: the column group's dense fp32 weights split into three
  truncated bf16 pieces (exact); per local expert a bf16 0/1 selector [3 Egp, 32] matmul sums its three pieces
  exactly (one product per tile-dot, DS41-16), max over the 32 equal columns -> [cap, 1] fp32 (DS41-23).
- Expert math: ttnn.linear q(x) @ W1^T / W3^T (HiFi4 + fp32 acc, fp32 out) -> min(g, 10), clamp(u, +-10) ->
  silu * u * r (fp32 SFPU) -> tt/quant.fp8_qdq -> @ W2^T fp32 -> bf16 (insert / combine are bf16-only) -> insert.
  combine (init_zeros) -> [s, 6, H] -> fp32 SFPU sum of the 6 slots -> group partial; groups summed over axis 1 by
  all_gather + SFPU add (EXPERTS_SUM=gather_add, K-30), the chip's hidden slice kept. `partial()` is exposed so the
  assembled model can add the shared expert's partial before the one exchange.
- hooks.py: `_experts_host_fn` (row_split_to_device(x), row_split_to_device(dense) -> hidden_split_to_host),
  `_device_step_fn` "experts", DEVICE_STEPS["swa"] += experts. The component path takes top-k of the dense routing
  on device; the assembled model passes the router's idx.
- Probe (not kept): 256 random rows vs ds41_ref._experts: rel 0.0028, warm 1.3 s.
- Gate: PASS. pcc 0.999855; vs CPU rel 0.00223 (limit 0.0150), worst row 0.039, ratio [0.983, 1.001]; vs golden rel
  0.0170 (limit 0.0320); layer24 / mixed / small / big rel <= 0.0028; A,B,A bit-identical; ~2.6 s per call (2048
  tokens). The gate takes ~12 min (precompile collect pass ~550 s, K-52).
- Gotchas: insert / extract take 2D slabs (ttnn.linear of a 2D x by a 4D weight returns 4D: reshape); the probe dir
  under tests/ttnn/unit_tests/operations/ds41_experts was deleted (K-56).
- OV-4: DS41-20 already states the F73 fix (done earlier); nothing left to change.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_experts.py`

## C.swa.shared_expert test (attempt 1)
- The rendered test (CHECKS="auto" only) failed the freeze sweep: noise1e-2 slipped (rel limit 0.015 = 2 x the bf16
  precision model's 0.0075, which the fp8 q-dq of x and h sets; DS41-14 / DS41-22, K-36).
- Added one structural check on every built-in input (golden, layer24, mixed, small, big): the output must lie in the
  column space of layer 0's shared w2 [5120, 2304] (the step ends with the w2 GEMM). Metric ||out - P out|| / ||out||,
  P from a float64 QR of w2, limit SPAN_REL 0.003 (DS41-25). Measured: fp32 reference 4e-8; bf16 inputs + output and
  the bf16 precision model 0.00105 (golden) .. 0.00121 (layer24); the bf16 golden itself 0.00105; noise1e-2 0.00742
  on every input; halfswap 0.75; sign 0.35-0.48. A device step that sums bf16 TP partials (instead of the planned
  fp32 partials) would raise it; keep the partials fp32 (components.yaml shared_expert).
- BRINGUP_IMPL=mutations runs the test's own sweep (built-in + span checks): every mistake caught, controls pass.
  reference: PASS (pcc 0.999899, vs golden rel 0.0142 / limit 0.0292). stub: FAIL. Device gate: NotImplementedError
  (no device module yet; the implement step writes it). The first "FAIL pcc ... 0.000000" block of each run is the
  precompile collect pass (K-52).
- OV-4: DS41-20 already states the F73 fix; nothing to change.
- Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=mutations scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_shared_expert.py`

## C.swa.shared_expert implement (attempt 1)
- `tt/shared_expert.py:TtSharedExpert`: q(x) (tt/quant.fp8_qdq, bf16 exact) -> one ttnn.linear by the chip's
  [W1^T_c | W3^T_c] [5120, 2 x I/TP] (bf16 weights, exact; HiFi4 + fp32 acc, fp32 out, K-25) -> slice g / u ->
  min / clamp +-10, silu, multiply (fp32 SFPU) -> fp8_qdq(h) over the chip's I/TP columns (1152 on 2x2, 576 on 4x4 /
  8x4 / 2x4: multiples of 32) -> ttnn.linear by W2^T_c [I/TP, 5120] fp32 partial -> sum over axis 1 + keep the chip's
  hidden columns (setting EXPERTS_SUM, gather_add default: all_gather + SFPU add, K-30) -> [S/SP, H/TP] fp32.
  TP read from mesh.shape; weights laid out once at load; no host work in the forward.
- `partial(x, xq=None)` / `reduce(part)` split so the assembled model can add it to TtExperts.partial before the one
  MoE exchange and reuse the experts' q(x) (plan.md). EXPERTS_SUM's description now names the shared expert (no new
  switch: in the assembled model it is the same exchange).
- hooks: `shared_expert` in DEVICE_STEPS["swa"]; `_rows_to_hidden_host_fn` (ffn_norm rows in, row-split +
  TP-replicated; hidden-split out).
- Gate: pcc 0.999899; vs CPU step rel 0.00152 (limit 0.015), worst row 0.0089; vs golden rel 0.0142 (limit 0.0292);
  second inputs rel 0.0015-0.0035; span outside w2 0.00014-0.00023 (limit 0.003; DS41-25: fp32 partials keep it low);
  determinism bit-identical. 14 s test body on 2x2.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_shared_expert.py`

## C.swa.moe_add implement (attempt 1)
- `tt/moe_add.py:TtMoeAdd` (from xing40 TtMoeAdd): `ttnn.add(experts, shared, dtype=fp32)` on the [S/SP, H/TP] fp32
  hidden-split layout TtExperts / TtSharedExpert return; fp32 + fp32 runs on the SFPU, so it equals torch's fp32 sum
  (K-30). Local, no CCL, no weights, no box constants. In the assembled model the add moves onto the fp32 partials
  before the one MoE exchange (components.yaml notes); this module is the standalone step.
- hooks: `moe_add` in DEVICE_STEPS["swa"]; `_two_hidden_host_fn` (both inputs and the output hidden-split, fp32).
- Gate: PASS. pcc 0.999998; vs CPU step rel 0 (bit-exact) on golden, layer24, mixed, small, big; vs golden rel 0.0022
  (limit 0.0058); determinism bit-identical. 23 s total.
- OV-4: known_issues.md not touched; DS41-20 already states the F73 fix.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_moe_add.py`

## C.swa.ffn_residual implement (attempt 1)
- `tt/residual.py:TtHcFfnResidual(TtHcResidual)`: the same local fp32 mix (ttnn.bringup.mhc_post default, setting
  RESIDUAL_MIX) with ffn_hc's mixes, plus the pending pre for the next layer (DS41-3: ffn_hc's pre, mixes[:, 0:4])
  sliced on the device into its own [1, 1, S/SP, 4] fp32 tensor. Returns (streams, pre); no CCL, no weight, no host
  work. `build_residual(cfg, step="ffn_residual")` picks it.
- hooks.py: `_ffn_residual_host_fn` (h_mid [S, 4H] via streams_to_device, mixes row_split_to_device, mlp_out
  hidden_split_to_device; back streams_to_host + row_split_to_host(pre), concatenated to [S, 4H + 4] only at the
  harness boundary); `_device_step_fn` "ffn_residual"; DEVICE_STEPS["swa"] += ffn_residual.
- Gate: PASS. pcc 0.999997; vs CPU rel 2.5e-8 (limit 0.0055); vs golden 0.00255 (limit 0.0081); second inputs rel
  <= 2.5e-8; determinism bit-identical. The pcc-0 FAIL lines before the real pass are the precompile collect pass (K-52).
- OV-4: known_issues.md not touched; DS41-20 already states the F73 fix.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_ffn_residual.py`

## C.swa_engram.engram_rows implement (attempt 1)
- Step tagged CPU (owner rule: hash + lookup on the host). `tt/engram.py:TtEngramRows`: `host_rows(tokens, tail,
  valid)` = EngramHasher.hashes + EngramTable.rows (memory-mapped, rows read on demand) -> [S, 6144] bf16 (exact:
  e4m3 x 2^k); `to_device` = one `ttnn.from_torch` [1, 1, S/SP, 6144] bf16 TILE DRAM, rows over axis 0, replicated
  over axis 1 (SP read from mesh.shape). `__call__` returns (device rows, new tail): the assembled model calls it in
  `begin_chunk`, never inside a layer; the tail is host state per slot (device keeps no engram state).
- Pads (DS41-26): only [start, actual_end) is hashed (explicit `valid`, else the first id outside the vocabulary);
  rows after it are zero, the tail is the last 3 real compressed ids. Host check: padded == unpadded prefix, chunked
  == one-shot (rows and tail bit-equal).
- hooks.py: `_engram_rows_host_fn` (tail from ctx.state.tail in the hybrid, updated there; from the golden
  state_prefix in component tests; DEAD at start 0; ctx.extra["end"] -> valid), `_device_step_fn("engram_rows")`,
  DEVICE_STEPS["swa_engram"] = {"engram_rows"}; the hybrid hands the reference's EngramHasher via shared (the
  compressed token map takes seconds to build; `engram_hasher` caches one per path otherwise).
- Gate: PASS. pcc 1.000000; vs CPU and vs golden rel 0 on golden and chunk0; determinism bit-identical. The pcc-0
  FAIL block before the real pass is the precompile collect pass (K-52).
- OV-4: DS41-20 already states the F73 fix; nothing to change.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_engram_engram_rows.py`

## C.swa_engram.engram_kv implement (attempt 1)
- `tt/engram_kv.py:TtEngramKv`: rows [1, 1, S/SP, 6144] (bf16 from TtEngramRows, fp32 from the harness) ->
  typecast fp32 -> `tt/quant.fp8_qdq` (DS41-13, bit-exact) -> one `ttnn.linear` with the chip's wkv columns
  [6144, 5 x H/TP] bf16 (fp8 x ue8m0 dequantized at load, asserted bf16-exact), HiFi4 + fp32 acc, fp32 out. wkv split
  by output, chip column c = its hidden slice of each of the 5 H-wide blocks (`layout.streams_cols_to_chip_major(...,
  n=5)`), so engram_add reads keys / value in the residual's hidden split, no exchange. TP read from mesh.shape
  (H/TP = 2560 on 2x2, 1280 on 4x4 / 8x4 / 2x4: all tile-aligned).
- fp8 act quant of the rows: identity on the layer-1 golden (host check: act_quant_dequant(rows, 32) == rows, 0 of
  12.6M values changed, no group with amax < 1e-4). Kept on by default because the auto checks' second inputs are
  not table rows; setting ENGRAM_KV_QDQ (qdq | skip, max precision qdq) lets the owner drop it as a perf pick.
- layout.py: `streams_cols_to_chip_major` / `chip_major_to_streams_cols` take `n` (default HC); `blocks_to_host`
  (harness boundary [S, 5H]). hooks.py: `_engram_kv_host_fn`, `_device_step_fn("engram_kv")`,
  DEVICE_STEPS["swa_engram"] += engram_kv.
- Gate: PASS. pcc 0.999999; vs CPU rel 1.0e-4 (limit 0.0033), vs golden 0.0017 (limit 0.0050); second inputs rel
  <= 1.9e-4; determinism bit-identical. ttnn.linear auto config (K-51 explicit config not tried; perf step).
- OV-4: DS41-20 already states the F73 fix; known_issues.md not touched.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_engram_engram_kv.py`

## C.swa_engram.engram_add implement (attempt 1)
- `tt/engram_add.py:TtEngramAdd`: streams [1, 1, S/SP, 4 x H/TP] fp32 + engram_kv [1, 1, S/SP, 5 x H/TP] fp32 (the
  chip's hidden slice of the 4 keys and the value, tt/engram_kv.py, no reshard) + pending pre -> (new streams, pre).
  Per stream: h * (q_w k_w) * key, h^2, key^2 (fp32 SFPU multiply) -> `ttnn.sum(dim=-1)` (fp32 + fp32 DEST = the
  accurate SFPU reduce, DS41-27) -> concat [S/SP, 12] -> `tp_sum` (all_gather axis 1 + SFPU add, K-30) ->
  rsqrt(ss / H + eps) products, / sqrt(H) -> copysign(sqrt(max(|d|, 1e-6)), d) via `where(ltz)` (ttnn.sign would
  give 0 at d = 0, torch.copysign +) -> sigmoid -> per-stream `addcmul(h_n, value, gate_n)` -> concat. q_w k_w is a
  fp32 [1, 4H] constant split chip-major over TP at load. TP from mesh.shape (H/TP 2560 / 1280, tile-aligned).
- layout.py: `blocks_to_device` (inverse of blocks_to_host). hooks.py: `_engram_add_host_fn` (x [S, 4H + 4], kv
  [S, 5H] in; streams + pre packed back), `_device_step_fn("engram_add")`, DEVICE_STEPS["swa_engram"] += engram_add.
- Gate: PASS. pcc 0.999998; vs CPU rel 1.0e-7 (limit 0.0072), vs golden 0.0020 (limit 0.0092); second inputs rel
  <= 1e-7; determinism bit-identical. Not tuned (12 slices / sums per call; a fused [R, 4, w] reduce is a perf item).
- OV-4: DS41-20 already states the F73 fix; unchanged.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_engram_engram_add.py`

## C.swa_engram.kv test (attempt 1)
- Attempt 0's built-in sweep let noise1e-2 slip (rel limits 0.0206 .. 0.0215 from the bf16 precision model's fp8
  re-roundings). The step is the same `_window_kv` as swa layer 0, so the test now carries test_c_swa_kv.py's fp8
  checks (DS41-14) with LAYER = 1: every value on the e4m3 x ue8m0 grid, <= 15 % of values != the CPU step, on all 5
  inputs, plus the test's own `BRINGUP_IMPL=mutations` sweep with those checks added.
- Layer 1 measured (CPU): bf16 precision model mismatch 0.040 .. 0.042, off-grid 0; noise1e-2 / scale +-2 % off-grid
  1.0; halfswap 0.99, rowshift 0.98, quarterzero 0.25 mismatch. The golden differs from the CPU step on the golden
  input in 0.20 of its values (bf16-stored upstream), so the mismatch check compares with the CPU step, not the golden.
- reference: PASS; stub: FAIL (all checks); mutations: every mistake caught, reference / bf16 rows pass.
- Gate (device, reused swa kv module): PASS, pcc 0.999835, vs CPU rel 0.0021, fp8 mismatch 0.0058 .. 0.0099.
- OV-4: DS41-20 already states the F73 fix; known_issues.md not touched.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_engram_kv.py`
  (add `BRINGUP_IMPL=reference|stub|mutations` for the freeze checks).

## C.swa_engram.attention implement (attempt 1)
- The window attention of layer 1 is the swa module (`tt/attention.py:TtWindowAttention` via `build_attention(...,
  layer)`, layer 1 weights and sink); device_component was already generic, so the frozen component test passed
  before any change (pcc 0.9997). The orchestrator's pre-run failed only the contract: the hybrid's
  `DEVICE_STEPS["swa_engram"]` held the engram steps alone, so layer 1 had no device window cache (DS41-28).
- hooks.py: `DEVICE_STEPS["swa_engram"]` += attn_hc, attn_collapse, attn_norm, q_a, kv, attention (all passed their
  component gates and the swap tests S.swa_engram.01-08 on the same swa modules). No tt/ change, no fork change.
- Gate: PASS. pcc_attention_L01 0.999704 (vs CPU rel 0.0222, limit 0.0301); test_cache_starts 4/4 (layer 1,
  0 failures each). The pcc-0 FAIL lines before the real pass are the precompile collect pass (K-52).
- OV-4: DS41-20 already states the F73 fix; unchanged.
- Re-run: `PYTHONPATH=$PWD BRINGUP_TASK=C.swa_engram.attention scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_engram_attention.py && PYTHONPATH=$PWD BRINGUP_TASK=C.swa_engram.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`

## C.swa.attention implement (attempt 1, scaled run)
- The swa window attention module from the real-size run (`tt/attention.py`, `tt/window_exact.py`) reads its dims
  from the scaled config and already passes the frozen component test (pcc 0.999704). The orchestrator's pre-run
  failed only the contract: `tests/bringup/contract/caches.py:53` reads `<hf>/inference/config.json`, which the scaled
  checkpoint lacks (DS41-11: scale.py writes only the top-level config.json; the reference derives it).
- Fix (outside the repo, no code change): `hf_scaled/quietbox_2x2/inference/config.json` = `ds41_ref.inference_config`
  of the scaled dir (n_heads 32, o_groups 4, n_routed_experts 96, the rest = the source's) plus symlinks to the
  source's `inference/` code files, so `inference_dir()` / hf_oracle still find model.py. A re-scale (scale.py rmtree)
  wipes it; the framework should write it (owner/overseer: scale.py or caches.py should use inference_config).
- No tt/, hooks.py or fork change. DEVICE_STEPS["swa"] already lists attention (DS41-28).
- Gate: PASS. pcc_attention_L00 0.999704; test_cache_starts 4/4, 0 failures each (layer 0, SP 2, W 1024).
- OV-4: DS41-20 already states the F73 fix; unchanged.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attention.py && PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`

## S.swa.06 implement (attempt 1, scaled run)
- Previous run (orchestrator, WINDOW_SDPA=high_precision per OV-3) failed one check: attention vs the CPU step worst
  row 0.0522 > 0.05 (the fixed swap_step_row limit; rel 0.0230 passed its F73 limit 0.0486; tail passed). Dumped the
  step's rows (probe, deleted): worst rows have ordinary norms, quantiles 0.022 / 0.027 / 0.037 / 0.047: the tail of
  the whole error, so the error had to come down, not one row.
- Fix: the bf16 output of sparse_sdpa (DS41-18) -> new fork option `ttnn.bringup.sparse_sdpa(fp32_output=True)`
  (needs high_precision; default off = previous programs). Setting WINDOW_SDPA_OUT = fp32 (default, max precision) |
  bf16. `tt/attention.py`: passes it, `_out` skips the typecast when the output is already fp32 (all_to_all back in
  fp32). No other step changed; no host work added.
- Gotcha (DS41-29): the first fp32 output was garbage: high_precision leaves the packer bf16 before the normalize
  and `sdpa_maybe_pack_reconfig_data_format<normalized_out, scratch>` skips at compile time once both are fp32;
  explicit `pack_reconfig_data_format(scratch)` under a new define SPARSE_SDPA_FP32_OUTPUT. cb_out_im unpacks with
  UnpackToDestFp32 for the untilize (srcA would be TF32).
- Gate: PASS. pcc_swap_out 0.999849; attention vs CPU rel 0.0189 (limit 0.0486), worst row 0.0458 (limit 0.05:
  ~9 % margin), ratio [0.9908, 1.0029]; tail 0.0168 (limit 0.0423). C.swa.attention 0.999777 and
  C.swa_engram.attention 0.999699 still pass. Fork: unit 37 passed / 10 skipped (32 + 5 new), fork_source 216 tests
  0 regressions, model cases 5 passed / 1 skipped. O.1 needs a case for the new call (fp32_output=True).
- Levers if the worst row tightens again: bf16 probabilities / q inside the op (DS41-18 emulation: small), or
  WINDOW_SDPA=exact (owner decision, OV-3).
- OV-4: DS41-20 already states the F73 fix; unchanged. DS41-18 updated (fp32 output now enough since F73).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_06_attention.py`;
  fork: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/sdpa/tests/unit/test_sparse_sdpa_high_precision.py -k fp32_output`

## S.swa.11 implement (attempt 1, scaled run) — gate FAIL, needs an owner / overseer decision
- Nothing to build: `tt/router.py:TtRouter` and `DEVICE_STEPS["swa"]` (router) were in place from C.swa.router.
  The gate fails one check only: `swap_router_vs_golden` pcc 0.988729 < 0.99 (the router component test's default
  PCC). Everything else passes: pcc_swap_out 0.999848, router vs the CPU step rel 9e-5, reselected 0.0015, tail rel
  0.0019 (limit 0.117).
- Cause (DS41-30): upstream bf16 noise, not the router. S.swa.10 (router on the CPU) already had router vs golden
  0.988758. Host-only CPU run, attention replaced by its bf16 precision model: router vs golden 0.984 (13 % of rows
  pick another expert set). The device is better than that floor. With DS41_WINDOW_SDPA=exact (comparison only) the
  gate passes: router vs golden 0.992686, attention vs CPU rel 0.0042.
- Not done: making WINDOW_SDPA=exact the default or the max-precision value. The owner rule and OV-3 forbid it
  without an owner decision ("never trade speed for exactness").
- Fix in `tt/window_exact.py:_lane_sum`: with one lane group (G == 1, the scaled 2x2 head count), the full-range slice
  aliased `o`, and `_lane_matmul` freed it ("Input Tensor B is not allocated"). G == 1 now returns o (or o + acc), and
  the caller frees o only when it is not the result. This only affects the exact (comparison) path.
- OV-4: DS41-20 already says F73 (2 x the component error, no cap); left as it is.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_11_router.py`
  (comparison: prefix `DS41_WINDOW_SDPA=exact`).

## S.swa.11 implement (attempt 2) — gate FAIL again, same single check; no model change
- Gate re-run unchanged: only `swap_router_vs_golden` 0.988729 < 0.99 fails (bit-identical to attempt 1); router vs
  the CPU step rel 9.1e-5, reselected 0.0015, pcc_swap_out 0.999848.
- DS41-30 / K-39: the router is right; the loss is top-6-of-96 near-tie flips from upstream bf16 noise. New CPU
  emulation (known_issues_detail.md, "Scaled router vs golden"): the attention alone with today's bf16 q / bf16
  probabilities gives router vs golden 0.9921; with fp32 (TF32-read) q and probabilities 0.9956. On device that
  would be roughly 0.991 (the in-tile rounding stays): a sparse_sdpa fork extension (fp32 q input, fp32 P CB) for a
  marginal pass, slower than the default. Not built: OV-3 / owner 2026-10-15 (fast path default, thresholds from the
  noise floor); the 0.99 vs-golden PCC is a fixed number below that floor (CC precision model 0.984).
- Needs: an owner / overseer decision (framework vs-golden limit for selection outputs from the precision model,
  or fp32-operand window attention as the max-precision value).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_11_router.py`

## S.swa.11 implement (attempt 3) — gate PASS (router vs golden 0.9918)
- Cause found (DS41-31): the window attention's sparse_sdpa kept the scores / probabilities in a bf16 CB, and the FPU
  reads fp32 operands at TF32 by truncation (K-30), a coherent bias. CPU emulation: bf16 scores alone cap router vs
  golden at 0.9910; TF32-nearest scores / q / probabilities 0.9953.
- Fork `ttnn.bringup.sparse_sdpa(fp32_scores=True)` (needs high_precision + fp32_output; default off = the previous
  program; define SPARSE_SDPA_FP32_SCORES): Float32 cb_qk_im, float32 q accepted, and an SFPU TF32 round-to-nearest in
  DEST on the pack thread before each pack the FPU later reads (scores, probabilities, running output, row-sum summed
  in DEST, 1/sum). Explicit unpack / pack format restores where the kernel assumed cb_qk_im was bf16. Gotcha: an SFPU
  op on the math thread next to the pack thread's SFPU work corrupts the other DEST half (one SFPU): pack thread only.
- Model: setting WINDOW_SDPA_SCORES (default and max precision fp32; bf16 = the previous path, a perf pick: 26.5 vs
  28.6 ms per attention call, 2x2, chunk 2048); q rounded by `tt/quant.round_tf32` (bit-exact vs torch) and kept fp32
  through the all_to_all. Setting FP32_WEIGHT_ROUND (rne): hc_fn rounded to TF32 at load (`tt/common.fpu_weight`),
  free; attn_hc rel 6.5e-4 -> 6.1e-4 (the rest is DS41-16's in-tile rounding).
- Gate: attention vs CPU rel 0.0189 -> 0.0143 (bias -7.5e-4 -> -3.0e-4), router vs golden 0.98873 -> 0.99182,
  pcc_swap_out 0.99987. Dropped: TF32 rounding of the mHC / router activations (5 ms per layer, router vs golden
  unchanged within flip noise).
- Regressions run: C.swa.attention / attn_hc / ffn_hc / router, C.swa_engram.attention / attn_hc / ffn_hc / router,
  S.swa.06 PASS. Fork: unit suite 41 passed / 10 skipped (37 + 4 new), source check 216 tests 0 regressions, model
  cases 5 passed / 1 skipped.
- Next: S.swa_engram.14 (layer 1, scaled) measures router vs golden 0.9847 with this (0.9839 without): it will fail
  the same fixed 0.99 check (DS41-30).
- Attempt 1's window_exact.py G == 1 fix stays in the tree (comparison path only).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_11_router.py`;
  fork: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/sdpa/tests/unit/test_sparse_sdpa_high_precision.py -k fp32_scores`

## C.swa.experts test review (scaled checkpoint, attempt 1)
- Freeze without review failed the mistake sweep: noise1e-2 slipped (rel 0.010 vs the bf16 precision model's
  0.016-0.018 on every input; DS41-22). Size limits cannot separate them; no grid (fp32 sum of six GEMMs) and no
  subspace (six experts' w2 span 6 x 2304 > 5120, so no DS41-25 span check).
- Added a row-scaling check (`_Expect.evaluate`, so the device run and `CC.sweep` under BRINGUP_IMPL=mutations both
  run it): least-squares slope b of log ||got_i - cpu_i|| on log ||cpu_i|| over rows, vs the CPU step, every input;
  b >= ROW_SLOPE_MIN 0.35; skipped when < 32 rows differ (the reference: rel 0). Float-point error is relative,
  added noise is not.
- Measured b: bf16 inputs + output 1.00; bf16 precision model golden 0.71, layer24 0.94, mixed 0.81, small 0.73,
  big 0.69; noise1e-2 -0.004..0.001; device (tt/experts.py) golden 0.72, layer24 0.87, mixed 0.67, small 0.66,
  big 0.78. Column slope separates less (model 0.13-0.25), not used.
- The previous review's `_Expect.check` (golden error added to vs-golden row / ratio limits) is kept; scaled golden:
  CPU step vs golden rel 0.0242, row 0.066, ratio [0.959, 1.025].
- Results: mutations PASS (every mistake caught, noise by rowscale_experts_L00_golden), reference PASS, stub FAIL,
  gate (device) PASS: pcc 0.999706, vs CPU rel 0.00225 (limit 0.0197), second inputs rel <= 0.0030, A,B,A identical.
- OV-4: DS41-20 already states the F73 fix; nothing to change.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_experts.py`
  (sweep: `BRINGUP_IMPL=mutations` in front).

## S.swa_engram.14 test review (layer 1, scaled, attempt 1)
- Freeze failed: with BRINGUP_IMPL=reference only `swap_router_vs_golden` failed, PCC 0.984586 < 0.99 (the router
  component test's default). The fp32 CPU chain from the bf16 golden block input drifts at the fp8 q-dq steps
  (DS41-1): ffn_norm 0.99985, then 12.4 % of rows pick another top-6 set (K-39). DS41-30 had predicted 0.9847.
- Test change (only the router's checks; everything else is run_swap_test as rendered): `_check_step` is patched
  inside the test (mock.patch.object) so the router's framework check runs without its golden PCC, and
  `_router_checks` adds (a) vs golden split by routing via `step_errors`: PCC >= 0.97, reselected <= 0.25, rel <=
  0.0103, worst row <= 0.05, ratio 1 +- 0.02, bias 0.01; (b) vs the CPU step, same-expert rows, rel <= 0.004.
- (b) because the framework's router rel limit is 2 x the component error 0.0493 = 0.0987 (that error is from the
  router's PCC on golden inputs, mostly flips) and noise1e-2 (rel 0.0097 vs CPU) passed every framework check
  (DS41-32). Golden rel alone caught it with only 7 % margin (0.01099 vs 0.0103).
- Measured: reference PASS (0.9846, reselected 0.1245, rel 0.00513); mutate:bf16 control PASS (cpu rel 0.00156);
  device gate PASS (pcc_swap_out 0.999781; router 0.9845 / 0.1284 / 0.00528, vs CPU rel 0.00012); stub FAIL;
  mutate:{noise1e-2, scale1.02, scale0.98, halfswap, rowshift, quarterzero, sign} with BRINGUP_MUTATE_STEP=router
  all FAIL. Table in the test docstring.
- BRINGUP_IMPL=mutations (the component sweep) does not apply to swap tests: run_swap_test takes the device path
  with mesh None (AttributeError in tt/engram_kv.py:42); the rendered test does the same.
- OV-4: DS41-20 already states the F73 fix; nothing to change.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_engram_14_router.py`
  (CPU: `BRINGUP_IMPL=reference` / `stub` / `mutate:<kind> BRINGUP_MUTATE_STEP=router` in front).

## C.swa_engram.experts test review (layer 1, scaled, attempt 1)
- Previous attempt failed the freeze sweep: noise1e-2 slipped. Its review had kept only `_Expect.check` (golden
  error added to the vs-golden row / ratio limits) and not layer 0's row-scaling check (DS41-22).
- Change: the layer-0 test's `ROW_SLOPE_MIN` 0.35, `ROW_SLOPE_ROWS` 32, `row_slope` and `_Expect.evaluate` copied
  verbatim (the tail of the file is identical to test_c_swa_experts.py from `ROW_SLOPE_MIN` on); LAYER = 1.
- Layer 1, CPU step vs golden: rel 0.0290, row 0.0690, ratio [0.964, 1.022] (limits with the golden error: rel
  0.0492, row 0.125, ratio 1 +- 0.079).
- Row slope b vs the CPU step: device (tt/experts.py) golden 0.685, layer24 0.663, mixed 0.535, small 0.614, big
  0.752; noise1e-2 (mutate) -0.002..0.005 on every input; stub 1.0 (fails the size checks).
- Results: mutations PASS (noise by rowscale_experts_L01_golden; bf16 inputs + output and bf16 everywhere pass),
  reference PASS (pcc 0.999579), stub FAIL, gate PASS: pcc_experts_L01 0.999577, vs CPU rel 0.00246 (limit 0.0202),
  second inputs rel <= 0.0029, A,B,A bit-identical.
- OV-4: DS41-20 already states the F73 fix; nothing to change.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_engram_experts.py`
  (sweep: `BRINGUP_IMPL=mutations` in front; noise slopes: `BRINGUP_IMPL=mutate:noise1e-2 BRINGUP_MUTATE_STEP=experts`).

## S.swa_engram.15 test review (layer 1, scaled, attempt 1)
- Freeze failed: with BRINGUP_IMPL=reference only `swap_router_vs_golden` failed (0.984586 < 0.99): the rendered test
  lacks test 14's router checks (DS41-32). Carried `_router_checks` over unchanged.
- Experts (DS41-33): vs golden PCC is 0.990793 on the CPU (device 0.990515), 0.0008 above 0.99, because 12.4 % of
  rows run other experts. `_experts_checks`: PCC >= 0.98 over all rows; on the rows whose routing (the swapped
  router's output, kept in `_ROUTING` by the router's check, which runs first) keeps the golden's support:
  reselected <= 0.25, rel <= 0.09, row <= 0.154, ratio 1 +- 0.092 (2 x the CPU chain's 0.045 / 0.077 / 0.046),
  bias 0.01.
- Experts vs the CPU step: mutate:bf16 control failed the framework's fixed row-ratio limit (1 +- 0.02; control
  [0.966, 1.013], worst row 0.047 vs 0.05). `_Limits` proxy gives `_check_step` row 0.094 / ratio 0.068 (2 x the
  control) for the experts only; rel (0.058) and bias (0.01) stay the framework's. noise1e-2 passed every check
  (rel 0.0100); added DS41-22's row slope >= 0.35 (device 0.709, control 0.61, noise 0.000).
- Results: reference PASS, stub FAIL, mutate:bf16 PASS, mutate:{noise1e-2, scale1.02, scale0.98, halfswap,
  rowshift, quarterzero, sign} (BRINGUP_MUTATE_STEP=experts) all FAIL; gate (device) PASS: pcc_swap_out 0.999781,
  experts vs CPU rel 0.0024, tail rel 0.0003. Table in the test docstring.
- OV-4: DS41-20 already states the F73 fix; nothing to change.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_engram_15_experts.py`
  (CPU: `BRINGUP_IMPL=reference` / `stub` / `mutate:<kind> BRINGUP_MUTATE_STEP=experts` in front).

## C.swa_engram.shared_expert test review (layer 1, scaled, attempt 1)
- Freeze failed: with BRINGUP_IMPL=reference only `auto_shared_expert_L01_vs_golden` failed: row norm ratio
  [0.982, 1.0126] outside 1 +- 0.0152. The fp32 CPU step on the bf16 golden input is itself rel 0.0277, worst row
  0.0539 from the golden (fp8 q-dq of x and h, DS41-1 / DS41-22); the built-in checks add that error to the
  vs-golden rel / column limits only.
- Change: the layer-0 test (test_c_swa_shared_expert.py: w2 span check, DS41-25, SPAN_REL 0.003, own mutation
  sweep) with LAYER = 1, plus layer 1 experts' `_Expect.check` (golden error added to the vs-golden worst-row and
  row-ratio limits: 0.058 -> 0.112, 1 +- 0.015 -> 1 +- 0.033), patched in as CC.Expect around run_component_test and
  used by `_expect()`. Tight vs-CPU and second-input checks unchanged.
- Measured: reference PASS (pcc 0.999617, span 5e-8); stub FAIL (built-ins); mutations PASS (bf16 inputs + output /
  bf16 everywhere pass, span 0.00122; scale1.02 caught by vs_cpu; noise1e-2 by span 0.0074 on every input; every
  other mistake caught); gate (device) PASS: pcc 0.999617, vs CPU rel 0.0031 (limit 0.0207), worst row 0.043, ratio
  [0.9956, 1.0035], second inputs rel <= 0.0036, span 0.00019-0.00029, A,B,A bit-identical.
- OV-4: DS41-20 already states the F73 fix; nothing to change. DS41-25 extended with the layer-1 note.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_engram_shared_expert.py`
  (CPU: `BRINGUP_IMPL=reference` / `stub` / `mutations` in front).

## S.swa_engram.16 test review (layer 1, scaled, attempt 1)
- Freeze failed: with BRINGUP_IMPL=reference only `swap_router_vs_golden` failed (0.984586 < 0.99, DS41-32): the
  rendered test lacks tests 14 / 15's checks. Rebuilt from test_swap_swa_engram_15_experts.py (router and experts
  checks unchanged, DS41-32 / DS41-33), SWAPPED + shared_expert.
- Shared expert (DS41-34): mutate:bf16 failed the framework's fixed worst row 0.05 vs the CPU step (0.0546); now
  SHARED_CPU_LIMITS row 0.109 / ratio 0.038 (2 x the control) via the `_Limits` proxy; rel (0.0553) and bias (0.01)
  stay the framework's. noise1e-2 passed every framework check, so DS41-25's w2 span check (<= 0.003) runs after
  the tail check (`component._check_tail` patched; `ref.w[1]["ffn.shared_experts.w2.weight"]`).
- Results: reference PASS, stub FAIL, mutate:bf16 PASS, mutate:{noise1e-2 (span 0.0074), scale1.02 / 0.98 (bias),
  halfswap, rowshift, quarterzero, sign} with BRINGUP_MUTATE_STEP=shared_expert all FAIL; gate (device) PASS:
  pcc_swap_out 0.999781, shared vs CPU rel 0.0030 row 0.027 ratio [0.9887, 1.0041], tail 0.00054, span 0.00023
  (the device also meets the framework's fixed row / ratio). Table in the test docstring.
- OV-4: DS41-20 already states the F73 fix; nothing to change.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_engram_16_shared_expert.py`
  (CPU: `BRINGUP_IMPL=reference` / `stub` / `mutate:<kind> BRINGUP_MUTATE_STEP=shared_expert` in front).

## S.swa_engram.17 test review (layer 1, scaled, attempt 1)
- Freeze failed: with BRINGUP_IMPL=reference only `swap_router_vs_golden` failed (0.984586 < 0.99, DS41-32): the
  rendered test lacks tests 14-16's checks. Rebuilt from test_swap_swa_engram_16_shared_expert.py (router, experts,
  shared-expert checks unchanged: DS41-32 / DS41-33 / DS41-34), SWAPPED + moe_add.
- Gotcha: the framework's tail check runs only for the LAST swapped step, so test 16's w2-span check (DS41-25,
  hooked into `_check_tail` under `last.name == "shared_expert"`) would silently stop running. It now reads the
  shared_expert step's output from the block trail (`seen[st.output]`, shared_out) after the moe_add tail check,
  and fails if that output is missing.
- moe_add keeps the framework's checks (rel <= 2 x component error 0.0046, tail <= 0.0033 from the precision
  model); they already catch every mutation, including noise1e-2 (rel 0.0100).
- Results: reference PASS, stub FAIL, mutate:bf16 PASS (rel 0.0024, tail 0.00055), mutate:{noise1e-2, scale1.02 /
  0.98, halfswap, rowshift, quarterzero, sign} with BRINGUP_MUTATE_STEP=moe_add all FAIL; gate (device) PASS:
  pcc_swap_out 0.999781, device moe_add bit-equal to the CPU step, span 0.00023. Table in the test docstring.
- OV-4: DS41-20 already states the F73 fix; known_issues.md not touched.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_engram_17_moe_add.py`
  (CPU: `BRINGUP_IMPL=reference` / `stub` / `mutate:<kind> BRINGUP_MUTATE_STEP=moe_add` in front).

## S.swa_engram.18 test review (layer 1, scaled, attempt 1)
- Freeze failed: with BRINGUP_IMPL=reference only `swap_router_vs_golden` failed (0.984586 < 0.99, DS41-32): the
  rendered test lacks tests 14-17's checks. Rebuilt from test_swap_swa_engram_17_moe_add.py (router, experts,
  shared-expert and w2-span checks unchanged: DS41-32 / DS41-33 / DS41-34 / DS41-25), SWAPPED + ffn_residual.
- The span check reads shared_out from the block trail (test 17's fix), so it still runs with ffn_residual last.
  ffn_residual's output is the block out: no tail check follows it.
- ffn_residual keeps the framework's checks (rel <= 2 x component error 0.0051); they catch every mutation,
  noise1e-2 by rel 0.0100.
- Results: reference PASS, stub FAIL, mutate:bf16 PASS (rel 0.0025), mutate:{noise1e-2, scale1.02 / 0.98, halfswap,
  rowshift, quarterzero, sign} with BRINGUP_MUTATE_STEP=ffn_residual all FAIL; gate (device) PASS: pcc_swap_out
  0.999781, device ffn_residual bit-equal to the CPU step, span 0.00023. Table in the test docstring.
- OV-4: DS41-20 already states the F73 fix; known_issues.md not touched.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_swa_engram_18_ffn_residual.py`
  (CPU: `BRINGUP_IMPL=reference` / `stub` / `mutate:<kind> BRINGUP_MUTATE_STEP=ffn_residual` in front).

## FIX.1 routed experts in bfp8 at load (implement, attempt 2; OV-6)
- What: `tt/experts.py` builds every routed expert's weights ONCE at load as bfloat8_b (EXPERTS_IMPL=bfp8, the new
  default): `fp4_dequant` (exact e2m1 x e8m0, bf16-exact) -> W13 [2I, H] (w1 rows | w3 rows) and W2 [H, I], kept in the
  stored [N, K] orientation, used with `ttnn.linear(..., transpose_b=True)`. The per-chunk decode is gone from the
  default path; the old path is EXPERTS_IMPL=fp4_decode (packed codes on device, `_decode` every chunk), comparison
  only. Rest of the expert math (dispatch, routing-weight pieces, ClampedSiluGlu, x r, fp8 q(h), combine, sum)
  unchanged.
- Orientation (DS41-35, contradicts DS41-10's "exact in both"): bfp8 groups 16 values along the last dim. As stored
  [N, K] a group sits inside one 32-wide e8m0 block -> always exact. As W^T [K, N] it spans 16 output rows with their
  own scales: 48 of the run's 2304 matrices (layers 0-3, 14, 20, 21, 24 x 96 experts x 3) would round (layer 0 expert
  3 w2: -0.5 at 2^-3 next to 3 x 2^-8). `_bfp8_exact` asserts exactness per tensor at load; it agrees with ttnn's own
  packing (`tt/perf_experts.py::test_bfp8_check_matches_ttnn`, 5.5 min host-only).
- transpose_b is done inside the matmul (mcast program configs; other configs would transpose B as a separate op and
  re-quantize bfp8): `test_transpose_b_exact` shows bfp8 [N, K] + transpose_b bit-identical to bf16 W^T, same time
  (w1 | w3 [2048, 5120] x [5120, 4608] 3.3 ms; w2 1.5 ms; HiFi4 + fp32 acc).
- Fused op: `ttnn.bringup.unified_routed_expert_moe` does not fit. DeepSeek's expert needs the token's routing weight
  multiplied into h BEFORE the fp8 act quant (q(r h) != r q(h), DS41-23) and q(h) between the activation and w2; the op
  runs gate/up/act/down in one program with neither (no per-row weight input, no in-kernel fp8 block-32 quant), and
  dropping q(h) is not a bf16 rounding of the model's step. Extending the fork = a per-row weight input + an SFPU
  block-32 amax / ue8m0 / e4m3 quant in the compute kernel: a kernel feature, left for a perf step. Kept the loop.
- Time (layer 0, 2x2, chunk 2048, random routing, `tt/perf_experts.py::test_time_experts`, mean of 3 synchronized calls
  after 2 warm-ups): before fp4_decode 610.7 ms, after bfp8 318.7 ms per layer per chunk; outputs bit-identical
  between the two. Load: 3.6 s -> 43.7 s (host dequant + bfp8 packing, 24 slots x 2 tensors). Matmuls are ~115 ms
  of the 319 (24 x (3.3 + 1.5)); the rest is the fp32 elementwise / fp8_qdq chain at cap = 2048 rows per expert (16x
  the mean 128 rows: the loop cannot see counts without a host read). Next lever: a count-aware expert op.
- Memory (plan.yaml / plan.md still say packed fp4 at 0.53 B / value; not in this step's paths): bfp8 1.0625 B / value,
  37.6 MB per expert, 0.84 GiB per layer per chip at 24 experts per chip (this scaled 2x2 and galaxy_4x4): 6.7 GiB for
  this run's 8 layers on 2x2 (fits). The deployment galaxy_4x4 at 40 layers would need 33.6 GiB of experts per chip
  (> 32 GB): there fp4_decode (or an in-op fp4 decode) stays the only exact option; owner / plan decision.
  galaxy_8x4 (12 per chip): 16.8 GiB.
- Gate PASS (all four): test_c_swa_experts pcc 0.999706 (vs CPU rel 0.0022), test_c_swa_engram_experts pcc 0.999577
  (rel 0.0025), test_swap_swa_15_ffn_residual pcc_swap_out 0.999874, test_swap_swa_engram_18_ffn_residual
  pcc_swap_out 0.999781 (experts vs CPU rel 0.0024, vs golden 0.990515: the same as before, the outputs are
  bit-identical). The chained gate outlives the agent tool's 600 s limit; run its four commands one at a time.
- Re-run: gate command of the brief; timing `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all
  models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py -s` (both impls + the exactness checks).

## FIX.2 overseer fix — window attention back to the reference kernel's precision
- Request: inference/kernel.py sparse_attn stage by stage (bf16 q / kv; fp32 scores, running max, sum; P cast to bf16
  before P.V; fp32 accumulation; bf16 output; fp32 sink) as the default AND the max-precision value; DS41-31's fp32
  path (TF32 q through the reshard, Float32 P, fp32 output) only as non-default values. FP32_WEIGHT_ROUND=rne kept.
- Fork `ttnn.bringup.sparse_sdpa` option `bf16_probs` (needs fp32_scores and a bf16 q; default off = the previous
  programs, define SPARSE_SDPA_BF16_PROBS): Float32 running max (cb_max_a/b; srcB reconfigured to it for score - max
  and for the flash correction), the row-sum from the TF32 probabilities (unchanged), then an in-place pass that rounds
  P to bf16 (RNE, SFPU on the pack thread, DS41-31's gotcha) before P.V; fp32_output may be off (bf16 output).
  Two wrong turns, measured: rounding P inside sub_exp made the row-sum use the rounded P (output moved 1.6x the
  kernel's bf16-P move); then the bf16 max put p_max = exp(0) off the bf16 grid (2.2x). Fixed by the order and the
  Float32 max.
- Model (tt/attention.py, tt/settings.py): WINDOW_SDPA_SCORES=fp32 (TF32-nearest scores), new WINDOW_SDPA_PROBS=bf16
  (bf16_probs, bf16 q through the reshard), WINDOW_SDPA_OUT=bf16: defaults and MAX_PRECISION. Comparison only:
  WINDOW_SDPA_PROBS=fp32 + WINDOW_SDPA_OUT=fp32 (DS41-31's program); WINDOW_SDPA=exact (DS41-19). The scale stays
  folded into q (K-27: one bf16 rounding of q, as the reference's one). typecast fp32 -> bf16 is RNE (probe: 100 %
  equal to torch on random values and ties).
- Time per layer (tt/perf_attention.py, wall clock, 5 calls after 2 warm-ups, layer 0, 2x2, chunk 2048, start 2048):
  before (DS41-31) 28.76 ms, after (reference dtypes) 26.13 ms, bf16 scores + bf16 out 26.17 ms.
- Accuracy at the reference's dtypes (F76 noise-floor limits): C.swa.attention pcc 0.999723 (limit 0.998523), vs CPU
  rel 0.0221 (limit 0.0526; DS41-31 was 0.0143), bias -5.0e-4; C.swa_engram.attention pcc 0.999621 (limit 0.998121),
  vs CPU rel 0.0253 (limit 0.0581); S.swa.11 router vs golden 0.9862 (limit 0.254; DS41-31 0.9918), router vs CPU
  0.99987, swap attention vs CPU rel 0.0222 (limit 0.0540), pcc_swap_out 0.999824 (limit 0.992326); S.swa.15
  pcc_swap_out 0.999824 (limit 0.976427); S.swa_engram.18 pcc_swap_out 0.999758 (limit 0.975779). Nothing failed at
  the reference's precision.
- Gate as run: 2 pre-existing failures, both independent of this change (same with the old dtypes):
  test_cache_starts under BRINGUP_TASK=FIX.2 runs every spec layer and fails on layer 2 (DS41-36; 4 passed under
  BRINGUP_TASK=C.swa.attention and C.swa_engram.attention); test_swap_swa_engram_18_ffn_residual.py's frozen `_checked`
  patch has F76's old 8-argument signature (DS41-37; TypeError after pcc_swap_out passed). Both are outside this
  step's paths (tests / framework).
- Op limits left (documented, DS41-31 detail): the FPU's score - max is TF32-precise (truncating); the flash
  correction stays bf16 (> 1 k_chunk only). Found: with fp32_scores alone P reaches P.V with ~9 bits.
- Fork regression (option off): unit 47 passed / 10 skipped (41 + 6 new), fork_source 216 tests 0 regressions, model
  cases 5 passed / 1 skipped. New tests: test_bf16_probs (x3, vs the kernel in torch), test_bf16_probs_values (P read
  back exactly through one-hot keys: 1.0 equal rne_bf16(rne_tf32(exp)); 0.26 with the rounding disabled, shown by
  hand), test_bf16_probs_program_and_refusal.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_attention.py`
  (and the other gate files; contract with `BRINGUP_TASK=C.swa.attention`); timing:
  `scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tt/perf_attention.py`; fork:
  `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/sdpa/tests/unit/test_sparse_sdpa_high_precision.py -k bf16_probs`

## R.5 reference, attempt 2: official taps (F80)
- Wrote (OV-7: no change to the reference) `reference/official_taps.py:capture`, the hook `hooks.py:official_taps`
  and `bringup/official_taps.yaml`. Official side = `hf_oracle.build_oracle(dtype=bf16)` (DS41-1's tiling), first
  max(layers) + 1 = 25 blocks, ONE prefill from position 0 over (chunk + 1) x chunk_size tokens (model.py has no chunked
  prefill: start_pos > 0 is a one-token decode), rows of chunk c sliced. Every step of every layer is mapped (module
  docstring has the map): block boundaries packed streams + pending pre (DS41-3); experts = MoE's fp32 `y` rebuilt
  from the expert outputs with the same `y[idx] += out` order (bit-equal); compressor = the rows written to
  compress_kv_cache + indexer.k_cache; indexer = output minus `offset`, padded with -1; kv = kv_norm's output object
  read after the in-place RoPE + fp8.
- Dtypes: packed boundaries marked bf16 (bf16 streams, fp32 pre in one tensor); mixes / router / experts fp32;
  kv / compressed `fixed` as attention / indexer inputs.
- Gate (run with `--spec $BRINGUP_SPEC`; the brief's command without it stops in Spec.load(None)): 129 steps,
  0 unmapped, 7 errors. 122 steps pass; every bf16-out step sits at 0.50 x its limit (one bf16 rounding).
  Failures and their cause (reference/official_taps_check.py):
  - compressor L2 / L14 / L20 rel 0.0373 / 0.0366 / 0.0371 > 0.0304 / 0.0351 / 0.0361: model.py's own bf16 points
    ahead of fp4; the fp32 step with them emulated equals the official output bit for bit (DS41-38).
  - attention L3 / L20 / L21 / L24 rel 0.057 / 0.115 / 0.152 / 0.173 > 0.055 / 0.070 / 0.068 / 0.074: the ctx is the
    golden's prefix state; with the official state 0.032 / 0.034 / 0.035 / 0.036 (all pass) (DS41-39).
  Both need a framework change (testing/official_floor.py); not fixable in this step's paths without misstating dtypes.
- official vs golden drift (bf16 chain vs fp32 chain), e.g. attn_norm rel 0.034 (L1), 0.090 (L3), 0.63 (L14), 0.29 (L24).
- Times (CPU): capture ~6 min, gate ~25 min.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.testing.official_floor --spec $BRINGUP_SPEC`;
  diagnosis: `PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.reference.official_taps_check`.

## R.5 reference, attempt 3 (run1 brief R.5.reference.1): official state in the taps (OV-8)
- Map and official_taps.yaml unchanged. `hooks.py:official_taps` now calls `capture(..., states=True)` and puts per
  layer `_state = {"start", "end"}` (DS41Reference.state_tensors format): window_kv from the official fp8 kv object,
  compress_kv / index_k from model.py's caches, compress_partial zeros (chunk ends are multiples of every ratio),
  topk_idx minus the indexer offset, candidates from shared_attn's mask; engram_tail from the token ids (EngramHasher,
  model.py keeps none).
- Gate (`python -m models.demos.common.bringup.testing.official_floor`, ~25 min CPU): 129 steps, 0 unmapped, 3 errors.
  Every attention now passes on the official state (DS41-39): L0-L24 rel 0.026-0.044 vs limits 0.037-0.075.
- Still failing, NOT state (same rel to 4 digits as with the golden's state; the compressor's output for a chunk reads
  only that chunk's x): compressor L2 / L14 / L20 rel 0.0373 / 0.0366 / 0.0371 > 0.0304 / 0.0351 / 0.0361. Re-verified
  DS41-38 with the hook's own taps: the fp32 step with model.py's bf16 points emulated equals the official output bit
  for bit (equal 1.000000) at all three. The map is right; the check's limit (2 x input nudge) does not hold the
  official code's internal bf16 roundings ahead of fp4. Needs a framework decision; dtypes left as they are (OV-8).

## R.5 reference, attempt 4 (run1 brief R.5.reference.1, OV-9): compressor `explained`
- The framework (F80, 014011cffcb) passes a step over its nudge limit when official_taps.yaml says why (`explained`).
- New `reference/official_checks.py` (the durable form of /tmp/r5chk/chk.py): runs the hook official_taps on the
  component golden's chunk (chunk 1, every golden layer) and recomputes each compressor with model.py's internal bf16
  points emulated (`compressor_bf16`). Result: official == emulated bit for bit (equal 1.000000) at L2 (ratio 2),
  L14 (ratio 2), L20 (ratio 1); official vs the plain fp32 step rel 0.0373 / 0.0366 / 0.0371. ~5 min CPU; exits 1 if
  any layer is not bit-equal.
- official_taps.yaml: `compressor` gets `explained` (why + that evidence); no dtype changed, nothing else explained.
  The bare step name covers every block type with a compressor (enc_source, enc_source_engram, dec_candidate).
- official_taps_check.py now imports compressor_bf16 from official_checks.py (one copy of the emulation).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.testing.official_floor` (~25 min CPU);
  evidence: `PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.reference.official_checks`.
- Gate: `ok   official taps: 129 steps, chunk 1` (0 errors, 0 unmapped); L2 / L14 / L20 compressor pass as explained
  (rel 0.0373 / 0.0366 / 0.0371 > 0.0304 / 0.0351 / 0.0361).

## FIX.4 overseer, attempt 3 (run1 brief FIX.4.overseer.3): hc mixes Sinkhorn in fp32 on the SFPU (OV-10/11/12)
- Reverted attempt 2 (OV-12): deleted ttnn/ttnn/bringup/mhc_split_sinkhorn/, restored INDEX.md and
  bringup_nanobind.cpp; tt/mhc.py and tt/settings.py back to HEAD before the change below.
- Fork `mhc_pre_ttnn`: `ttnn.bringup.mhc_pre_xing(..., deepseek=True)` (default False). DeepSeek's
  hc_split_sinkhorn_kernel math (K-20): pre = sigmoid + eps, no clamp, softmax + eps, column norm, then iters - 1
  (row, column) pairs; the same fp32 SFPU code as Xing's path. Kernel code behind the compute define
  MHC_XING_DEEPSEEK (absent when off: Xing's program and hash unchanged). CHANGELOG entry 7, INDEX row.
- tt/mhc.py: setting HC_SINKHORN (default and max_precision `sfpu_fp32`): after tp_sum, one mhc_pre_xing call on the
  exchanged row (it also does r = rsqrt(sum x^2 / 4H + eps), so the hand-composed slice / multiply / add / rsqrt chain
  is gone on this path); `fpu` keeps the old chain + ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn for A/B.
- Stage numbers, golden inputs, layer 0..2, component chunk 1 (probe_001.py in
  tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/):
  - sum x^2: rel 4-6e-7 (exact). Linear (device mixes vs fp64): 3.39e-4 / 3.67e-4 / 3.46e-4 / 3.87e-4 / 3.47e-4
    (L0 attn / L0 ffn / L1 attn / L1 ffn / L2 attn), multiplicative bias -3.3e-4..-3.75e-4; CPU fp32 mixes 2-4e-7.
  - Sinkhorn alone (device output vs a CPU fp32 Sinkhorn on the device's own mixes): fpu 5.9e-4 / 1.19e-3 / 6.5e-4 /
    7.7e-4 / 7.4e-4; sfpu_fp32 6.8e-8 / 7.6e-8 / 6.5e-8 / 7.4e-8 / 6.5e-8.
  - Whole step vs the CPU step: fpu 6.12e-4 / 1.27e-3 / 6.98e-4 / 8.21e-4 / 7.82e-4 (bias -3.3e-4..-4.4e-4, worst
    column L0 attn col 10 1.6 %, L2 attn col 15 3.5 %); sfpu_fp32 6.07e-5 / 1.08e-4 / 1.36e-4 / 1.50e-4 / 1.00e-4
    (bias +1.4e-5 / +3.8e-5 / +7e-6 / +4.5e-5 / -8e-6); rel limits 3.52e-5 / 1.47e-4 / 3.94e-4 / 2.84e-4 / 4.05e-4.
  - Times (wall, 20 synced calls, one call = 1024 rows per chip on 2x2): rsqrt + Sinkhorn fpu 0.45 ms, sfpu_fp32
    0.031 ms; whole mixes step 0.77 -> 0.41 ms; per layer per chunk (attn_hc + ffn_hc) 1.54 -> 0.82 ms.
  - Fork cases (random rows, real scale / base): rel 5.2e-8..1.0e-7 vs fp64.
- What remains (DS41-41): every column bias the gates flag equals the one the CPU fp32 Sinkhorn makes from the
  device's mixes (e.g. L0 attn col 20 -1.74e-3 of rms both): the linear's in-tile partial-dot truncation (DS41-16).
  Golden streams are bf16-valued and hc_fn is TF32-rounded, so the operands are exact; round_tf32(x) changed nothing
  here. With fp32 streams (swap / assembled model) the FPU also truncates x: mixes -6.9e-4; round_tf32(x) halves it,
  +1.94 ms per call (not taken: owner pick). No multi-pass (owner): the steps' column-bias checks need a waiver.
- RMS stage (owner note): exact as composed (REPORT: sum x^2 1e-6, rsqrt 5e-7; OV-10 said leave it); on the default
  path the rsqrt now runs inside mhc_pre_xing (Newton rsqrt, same fork), so no hand-composed chain remains there.
  rsqrt fast_and_approximate_mode: only on the `fpu` path (ttnn.rsqrt default, accurate).
- Not done: x as bf16 for the linear (OV-10 "optional"): in the assembled model the streams are fp32, not bf16-valued,
  so a bf16 cast would round them.
- Gate (each test with its own BRINGUP_TASK): err_* all pass (L0 attn 0.00124 <= 0.00247, L0 ffn 0.00125 <= 0.0036,
  L1 attn 0.00127 <= 0.0077, L1 ffn 0.00133 <= 0.0123, L2 attn 0.00111 <= 0.0077; swap err_swap_out 0.0187 <= 0.083,
  0.0218 <= 0.083). The tests still FAIL on the F83-F85 auto checks: vs CPU column bias 25-38 x (all five), L0 attn
  rel 6.07e-5 > 3.52e-5; vs golden column bias L0 attn col 20 16.7 x, L0 ffn 4.6 x, L1 attn 4.6 x, L2 attn 3.4 x;
  swaps also fail router / kv own-floor checks (as in the overseer's f83 log, not hc). Fork regression (option off):
  model cases 103 passed, unit + golden 321 passed / 1 skipped (= baseline).
- Re-run: the brief's gate; fork: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_pre_ttnn/tests/test_mhc_pre_xing.py -k deepseek`;
  A/B: `DS41_HC_SINKHORN=fpu`.

## FIX.4 overseer, attempt 4 (run1 brief FIX.4.overseer.4): no further change; residual is the device floor
- Tree = attempt 3's change (HC_SINKHORN=sfpu_fp32, mhc_pre_xing(deepseek=True); build current, 15:11). No code
  changed in this attempt: nothing left in tt/mhc.py is a bug or a lower-precision choice at normal settings.
- Gate as written: stops at test 1 (pytest asserts on the F83-F85 auto checks), numbers bit-identical to attempt 3.
  Each remaining test run alone with its BRINGUP_TASK: every err_* passes:
  err_attn_hc_L00 0.00124 <= 0.00247, err_ffn_hc_L00 0.00125 <= 0.00360, err_attn_hc_L01 0.00127 <= 0.00774,
  err_ffn_hc_L01 0.00133 <= 0.0123, err_attn_hc_L02 0.00111 <= 0.00773, err_swap_out 0.0187 <= 0.0830 (S.swa.15),
  0.0218 <= 0.0825 (S.swa_engram.18).
- What fails (all the linear's in-tile truncation, DS41-16 / DS41-41; the probe report's best is the same: L0 attn
  6.1e-5 = 3.3 x its 1.84e-5 floor): vs CPU rel L0 attn 6.07e-5 > 3.52e-5; column bias vs CPU 25-38 x (components),
  vs golden 3.4-16.7 x; own-floor row / column 2.1-3.2 x; swaps (fp32 inputs, x also truncated, K-30): S.swa.15
  attn_hc 6.1e-5 > 3.5e-5, ffn_hc 2.79e-4 > 1.5e-4; S.swa_engram.18 ffn_hc 3.14e-4 > 2.8e-4. Swaps also fail router
  and kv own-floor checks (not hc; as before this task).
- Left for the owner: waive the hc steps' vs-CPU / column-bias checks (DS41-41), or pick round_tf32(x) before the hc
  linear for fp32 inputs (halves the swap-side bias, +1.94 ms per call; not a "normal setting", not taken).
- Re-run: the brief's gate; per test `BRINGUP_TASK=<task> scripts/run_safe_pytest.sh --run-all <test>`.

## FIX.4 overseer, attempt 5 (run1 brief FIX.4.overseer.5): hc_fn rounded to the 9 bits HiFi4 multiplies in srcA
- Found the linear's bias (DS41-42, corrects DS41-41): at HiFi4 the FPU multiplies 9 explicit mantissa bits of matmul
  in1 (srcA), all 10 of TF32 in in0 (srcB). hc_fn^T sits in in1, so its TF32 10th bit was truncated: mixes bias
  -3.46e-4 with TF32-exact operands (probe_008.py). W in in0 or W rounded to 9 bits: 8e-5, bias +3e-5.
- Change: tt/common.fpu_weight rounds to nearest-even at 9 bits when FP32_WEIGHT_ROUND=srca (new value; default and
  max_precision); rne (TF32, attempts 3/4) and truncate stay selectable. Load time only. tt/mhc.py: comments only.
  No fork changed (mhc_pre_xing(deepseek=True) from attempt 3 stays).
- Stage numbers (golden inputs, chunk 1; L0 attn / L0 ffn / L1 attn / L1 ffn / L2 attn; probe_009.py), rne -> srca:
  device mixes vs fp64 3.4e-4..3.9e-4 (bias -2.8e-4..-3.4e-4) -> 7.5e-5..1.5e-4 (bias +1.9e-5..+6.6e-5); step vs
  the CPU step 6.07e-5 / 1.08e-4 / 1.36e-4 / 1.50e-4 / 1.00e-4 -> 1.07e-5 / 4.44e-5 / 6.02e-5 / 6.58e-5 / 5.88e-5
  (limits 3.52e-5 / 1.47e-4 / 3.94e-4 / 2.84e-4 / 4.05e-4: all pass now); Sinkhorn alone 6.5e-8..7.6e-8; fp32-mantissa
  streams 1.19e-4..3.04e-4 -> 5.9e-5..1.56e-4. Table in known_issues_detail.md (DS41-42).
- RMS statistic (owner note), per-chip sum of x^2 vs fp64, [1024, 10240] fp32 per chip (probe_007.py): today's
  mhc_pre_xing_pack rel 4.7e-7 (bf16-valued) / 1.0e-7 (fp32 mantissas), 0.119 ms incl. packing the mix row;
  ttnn.rms_norm_pre_all_gather does not fit L1 at this width in fp32 (dataflow_buffer.cpp:2695), on 4 column
  blocks 7.7e-8 / 2.3e-8, 0.247 ms; ttnn.bringup.rms_norm whole norm 0.29 ms, its statistic as implied by y rel
  7.6e-4 (incl. its y multiply). The statistic is already exact (its rsqrt error 2e-7 vs the tightest floor 1.8e-5)
  and on the default path the rsqrt is inside mhc_pre_xing (no multiply / add / ttnn.rsqrt chain): the fork was not
  extended (a second read of x per call, no accuracy to gain). ttnn.rsqrt is only on the `fpu` A/B path, accurate.
- Gate (each test run alone too): every err_* passes (attn L0 0.00123, ffn L0 0.00124, attn L1 0.00126, ffn L1
  0.00132, attn L2 0.00111; err_swap_out 0.0186 / 0.0218). Component vs-CPU rel all pass; they fail only the own-floor
  checks: column bias 5.0 / 6.9 / 7.7 / 5.5 x (L0 ffn / L1 attn / L1 ffn / L2 attn), L0 attn row 2.5 x, column 22
  2.7 x, its bias 23 x (+1.38e-4 of rms; host 9-bit rounding alone +1.6e-4), vs golden column 10 bias 2.6 x. Swaps:
  S.swa.15 ffn_hc rel 1.95e-4 > 1.5e-4 (was 2.79e-4; fp32 h_mid truncated to TF32 on unpack, K-30), own-floor
  col / row; S.swa_engram.18 attn_hc / ffn_hc rel pass (1.28e-4, 1.49e-4; ffn was 3.14e-4 > 2.8e-4), own-floor
  col / row fail; router and kv own-floor checks fail as before (not hc).
- What remains needs the owner's waiver: the weight's fixed per-column rounding (any FPU weight format; host alone
  TF32 9e-5..1.7e-3, 9 bits 1.9e-4..1.8e-3 of rms) needs an exact fp32 weight = a split matmul (owner: no).
- Re-run: the brief's gate; A/B `DS41_FP32_WEIGHT_ROUND=rne`; probes:
  `scripts/tt-probe.sh mhc_ds41_fix4 < tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/probe_00{7,8,9}.py`.

## FIX.4 fix, attempt 101 (run1 brief FIX.4.fix.101): no code change; what remains is the device floor (DS41-43)
- Tree = attempt 5 (FP32_WEIGHT_ROUND=srca, HC_SINKHORN=sfpu_fp32 = mhc_pre_xing(deepseek=True)). Nothing changed in
  tt/ or the forks. Gate test 1 re-run: bit-identical to the 16:27 log. err_attn_hc_L00 0.00123 <= 0.00247 ok; vs CPU
  rel 1.07e-5 <= 3.52e-5 ok. It fails only the own-floor checks: row 900 2.51 x, column 22 2.68 x, column 22 bias
  +1.38e-4 of rms 23.4 x; vs golden column 10 bias 2.57 x.
- Stage by stage, L0 attn_hc chunk 1, golden inputs (probe_010.py), device / host emulation of the 9-bit weight alone /
  device vs emulation:
  - sum x^2: exact (4.7e-7, attempt 5). Mixes vs fp64: 7.46e-5 / 4.99e-5 / 5.63e-5 (FPU accumulation, column means
    -1.2e-5..-4.0e-5 of rms).
  - Sinkhorn (SFPU fp32) vs a CPU fp32 Sinkhorn on the device's mixes: row 900 1.25e-7 (exact).
  - Step vs CPU: 1.07e-5 / 7.85e-6 / 1.04e-5. Column 22 bias +1.38e-4 / +1.57e-4 / -1.8e-5 (the weight's rounding).
    Row 900: 1.34e-4 / 3.96e-5 / 9.47e-5 (median row 1.6e-5). Its mixes error 1.47e-2 is below the median row's
    2.88e-2: ordinary FPU accumulation that this row's Sinkhorn amplifies.
- The test that settles it (CPU, the gate's own Expect and floor; cpu/weight_floor_cpu.py). A perfect device whose
  only error is hc_fn's rounding fails the vs-CPU column-bias check:
  - TF32 rne (the most an FPU source holds) at 4 of 5 steps: 5.3 / 2.9 / 1.65 ok / 4.1 / 7.4 x.
  - 9 bits (today) at 5 of 5: 26.5 / 11.3 / 5.9 / 5.2 / 3.6 x, plus the vs-golden check at L0 attn col 10, 2.8 x.
  - 11 bits at 2 of 5. 12 bits passes everything.
  So no single-pass FPU weight passes the checks; only a split / exact matmul would (owner: no). REPORT.md agrees:
  SFPU ops are fp32-exact, the FPU is the limit, and fidelity / fp32 acc / packer_l1_acc change nothing.
- Answer: disagree. Waiver requested for the hc steps' own-floor row / column / bias checks. Owner option, not taken:
  swap the operands so hc_fn sits in in0 / srcB at full TF32. Columns would improve on 4 of 5 steps but still fail,
  and fp32 x would then lose a bit in srcA (K-30).
- Re-run: `PYTHONPATH=$PWD python tests/ttnn/unit_tests/operations/mhc_ds41_fix4/cpu/weight_floor_cpu.py 0 5` (CPU,
  ~25 min); `scripts/tt-probe.sh mhc_ds41_fix4 < tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/probe_010.py`.

## FIX.4 fix, attempt 102 (run1 brief FIX.4.fix.102): no code change; bookkeeping of attempt 101 fixed; answer disagree
- Reverted the out-of-scope file: weight_floor_cpu.py moved out of the repo to
  /localdev/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/run1/evidence/FIX.4/ (its code is also in
  known_issues_detail.md, DS41-43 section). Re-run: `PYTHONPATH=$PWD python <that path> 0 5` (CPU, ~25 min).
- known_issues DS41-43 shortened to <= 600 chars; its detail heading now equals the entry title
  (`python3 -m models.demos.common.bringup.knowledge.check --model-dir models/demos/deepseek_v41_flash_d_p`: clean).
- Tree = attempt 5 (FP32_WEIGHT_ROUND=srca, HC_SINKHORN=sfpu_fp32). Each gate test run alone (logs
  /tmp/fix4_102/): numbers bit-identical to attempts 5 / 101. err_attn_hc_L00 0.00123 <= 0.00247, err_ffn_hc_L00
  0.00124 <= 0.00360, err_attn_hc_L01 0.00126 <= 0.00774, err_ffn_hc_L01 0.00132 <= 0.0123, err_attn_hc_L02 0.00111
  <= 0.00773, err_swap_out 0.0186 <= 0.0830 (S.swa.15). Every component vs-CPU rel passes (1.07e-5 / 4.44e-5 /
  6.02e-5 / 6.58e-5 / 5.88e-5 vs 3.52e-5 / 1.47e-4 / 3.94e-4 / 2.84e-4 / 4.05e-4). The pytest asserts fail only on
  own-floor checks: vs CPU column bias L0 attn col 22 23.4 x (+row 900 2.51 x, col 22 2.68 x), L0 ffn col 4 4.99 x,
  L1 attn col 11 6.91 x, L1 ffn col 0 7.73 x, L2 attn col 6 5.54 x; vs golden L0 attn col 10 2.57 x. Swaps (fp32
  streams, x truncated to TF32 on unpack, K-30): S.swa.15 ffn_hc rel 1.95e-4 > 1.5e-4, own-floor bias 60 x; attn_hc
  23.4 x; S.swa_engram.18 hc own-floor 22-43 x (rel pass); router / kv own-floor fails are not hc (as before).
- Fresh check of the floor's premise: the official hc_mixes is `F.linear(x, hc_fn) * rsqrt` in fp32
  (hf/inference/model.py:956; torch's default matmul is not TF32), so the floor (fp32 weights, input nudges only) is
  right and the device's 9-bit / TF32 weight is the residual. Nothing left is a bug or a lower-precision choice at
  HiFi4 / fp32 acc / approx off; the remaining options are owner picks (round_tf32(x) for fp32 streams, +1.94 ms
  per call; hc_fn in in0 / srcB still fails 4 of 5 steps). Waiver requested for the hc own-floor checks.

## FIX.4 fix, attempt 103 (run1 brief FIX.4.fix.103, ttnn-expert-debugger): no code change; independent check, answer disagree
- Tree = attempt 5 (FP32_WEIGHT_ROUND=srca, HC_SINKHORN=sfpu_fp32 = mhc_pre_xing(deepseek=True)); nothing changed in
  tt/ or the forks. Each gate test run alone (logs /tmp/fix4_103/): bit-identical to attempts 5 / 101 / 102. Every
  err_* passes (0.00123 / 0.00124 / 0.00126 / 0.00132 / 0.00111; err_swap_out 0.0186 / 0.0218) and every component
  vs-CPU rel passes; pytest fails only on own-floor checks (vs CPU column bias L0 attn col 22 23.4 x + row 900
  2.51 x + col 22 2.68 x, L0 ffn col 4 4.99 x, L1 attn col 11 6.91 x, L1 ffn col 0 7.73 x, L2 attn col 6 5.54 x; vs
  golden L0 attn col 10 2.57 x). Swaps: S.swa.15 ffn_hc rel 1.95e-4 > 1.5e-4 (fp32 h_mid, K-30) + own-floor; router /
  kv own-floor fails are not hc.
- The one thing not tried before: is the linear's remaining accumulation error a matmul configuration choice?
  probe_011 (chip 0, L0 attn_hc golden x [1024, 10240] @ the 9-bit / bf16 weight, exact products) vs fp64:
  default 8.08e-5 (bias +3.0e-5, worst column bias 1.75e-4 of rms); explicit 1D config in0_block_w 1 / 4 / 10 / 40:
  8.08e-5 bit-identical (the partial reload is exact fp32); packer_l1_acc on: 8.2-8.3e-5. A CPU model of exact
  tile dots + fp32 accumulation: 2.6e-7. probe_012: K = 32 (one tile) 2.1-3.1e-4; one nonzero product per dot:
  exact; two nonzero products: rel 1.5e-4, 44 % of outputs inexact. So the error is the FPU's in-tile sum (~TF32,
  DS41-16), not a setting: even an exact weight leaves a ~2e-4 column bias at the mixes and row 900's 2.5 x.
- Owner note on the RMS statistic (probe_013; DS41-44): ttnn.bringup.rms_norm's r = y / x is on the TF32 grid,
  bias +2.9e-4, rms 3.7e-4 vs fp64; today's mhc_pre_xing_pack sum 4.7e-7 (and the rsqrt inside mhc_pre_xing, SFPU
  fp32 Newton, 5e-7). The fork sums x^2 with an FPU reduce, so a return_sum_sq option would still be ~1e-4: not "at
  least as accurate"; not extended (as OV-10 decided). ttnn.rsqrt: only on the fpu A/B path, fast_and_approximate_mode
  defaults to False (accurate). No hand-composed multiply / add / rsqrt chain is left on the default path.
- Lead for the owner (not acted on: model-wide, outside this step): the official streams are bf16 (model.py:968
  hc_post `y.type_as(x)`, default dtype bf16; the golden streams are bf16-valued), while ds41_ref.hc_post and the
  device (tt/residual.py) keep them fp32. With bf16 streams the hc linear's x would be exact in srcB in the assembled
  model and the swaps' K-30 truncation (S.swa.15 ffn_hc rel 1.95e-4) would go; the own-floor column bias (weight) stays.
- Answer: disagree. Nothing left in tt/mhc.py is a bug or a lower-precision choice at HiFi4 / fp32 acc / approx off;
  the weight's fixed rounding (DS41-43) and the FPU's in-tile sum (DS41-16) need a split / exact matmul (owner: no).
  Waiver requested for the hc steps' own-floor row / column / bias checks (and S.swa.15 ffn_hc's rel on fp32 streams).
- Re-run: `scripts/tt-probe.sh mhc_ds41_fix4 < tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/probe_01{1,2,3}.py`.

## FIX.4 overseer, attempt 1 of brief FIX.4.overseer.1 (17:24, after the owner waiver OV-13): finish and report
- Tree = HEAD c50953498f6 (HC_SINKHORN=sfpu_fp32 = mhc_pre_xing(deepseek=True), FP32_WEIGHT_ROUND=srca). Rebuilt
  (`./build_metal.sh`; fork sources newer than the 15:11 build). No change to tt/ code or the fork's code: OV-13 waives
  the hc own-floor checks; nothing left in tt/mhc.py is a bug or a lower-precision choice (DS41-42, DS41-43).
- Changed: tt/settings.py HC_SINKHORN description = the table below (its step numbers were from before srca); fork
  CHANGELOG entry 7 = the regression re-run; known_issues DS41-45.
- HC_SINKHORN table, golden inputs, chunk 1, 2x2, 1024 rows per chip, srca (probe_014.py, 20 synced calls):

  | step | floor (limit/2) | sfpu_fp32 step vs CPU | x floor | fpu step vs CPU | x floor | Sinkhorn alone sfpu / fpu | rsqrt+Sinkhorn ms sfpu / fpu | whole step ms sfpu / fpu |
  |---|---|---|---|---|---|---|---|---|
  | L0 attn_hc | 1.76e-5 | 1.07e-5 | 0.61 | 5.94e-4 | 33.7 | 6.6e-8 / 5.9e-4 | 0.031 / 0.455 | 0.409 / 0.797 |
  | L0 ffn_hc | 7.37e-5 | 4.44e-5 | 0.60 | 1.19e-3 | 16.2 | 7.6e-8 / 1.2e-3 | 0.030 / 0.449 | 0.410 / 0.764 |
  | L1 attn_hc | 1.97e-4 | 6.02e-5 | 0.31 | 6.45e-4 | 3.28 | 6.5e-8 / 6.5e-4 | 0.035 / 0.448 | 0.411 / 0.805 |
  | L1 ffn_hc | 1.42e-4 | 6.58e-5 | 0.46 | 7.58e-4 | 5.35 | 7.5e-8 / 7.7e-4 | 0.031 / 0.451 | 0.413 / 0.771 |
  | L2 attn_hc | 2.03e-4 | 5.88e-5 | 0.29 | 7.44e-4 | 3.67 | 6.5e-8 / 7.4e-4 | 0.033 / 0.455 | 0.410 / 0.772 |

  Per layer per chunk (attn_hc + ffn_hc, L0): sfpu_fp32 0.82 ms, fpu 1.56 ms.
- Fork regression, option off, fresh build: model cases (test_mhc_pre_xing.py + test_mhc_pre_ttnn.py: xing40, glm53,
  the 5 DeepSeek cases, 2 CPU checks) 103 passed; unit + golden 321 passed / 1 skipped (= the recorded baseline).
- Gate, each test with its BRINGUP_TASK (logs /tmp/fix4_ov1/g1..g7.log): every threshold metric passes:
  err_attn_hc_L00 0.00123 <= 0.00247, err_ffn_hc_L00 0.00124 <= 0.00360, err_attn_hc_L01 0.00126 <= 0.00774,
  err_ffn_hc_L01 0.00132 <= 0.0123, err_attn_hc_L02 0.00111 <= 0.00773, err_swap_out 0.0186 <= 0.0830 (S.swa.15),
  0.0218 <= 0.0825 (S.swa_engram.18). The 5 component tests PASS (hc own-floor checks reported WAIVED). The 2 swap
  tests exit 1 on non-hc checks only (DS41-45): swap_router own-floor (S.swa.15 3.92e3 x, S.swa_engram.18 2.33e3 x;
  present before FIX.4, verify/f83) and S.swa.15 swap_kv column 183 bias 2.27 x. Their hc steps: S.swa.15 attn_hc rel
  1.1e-5 ok, ffn_hc rel 1.95e-4 > 1.5e-4 (fp32 h_mid truncated to TF32 on unpack, K-30) WAIVED; S.swa_engram.18
  attn_hc 1.28e-4 / ffn_hc 1.49e-4 rel ok, own-floor WAIVED.
- Re-run: the brief's gate; table `scripts/tt-probe.sh mhc_ds41_fix4 < tests/ttnn/unit_tests/operations/mhc_ds41_fix4/probes/probe_014.py`;
  fork `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_pre_ttnn/tests/test_mhc_pre_xing.py ttnn/ttnn/bringup/mhc_pre_ttnn/tests/test_mhc_pre_ttnn.py`
  and `... tests/unit tests/golden`; A/B `DS41_HC_SINKHORN=fpu`.

## C.enc_source.q_a test review (attempt 1)
- Freeze without review failed: the built-in sweep let noise1e-2 slip (rel limits 0.0158 golden / small / big, 0.0292
  layer24, 0.0325 mixed come from the input nudge's fp8 flips, DS41-1; 1 % noise is rel 0.010). K-36, DS41-14.
- q_a = q_norm(fp8_linear(attn_norm, wq_a)); official RMSNorm is fp32 with a bf16 output (hf/inference/model.py:281).
  Added a norm-energy check on every auto input: bias = mean over rows of (rms(out_row / q_norm.weight) - 1),
  |bias| <= NORM_BIAS 1.5e-5 (DS41-45). CPU probe (outside the tree): CPU / nudge / floor 2e-9; bf16 output
  +1.4e-6 / -1.0e-6 / -3.6e-6 (golden / layer24 / mixed); noise1e-2 +7.6e-5 / +5.7e-5 / +5.6e-5; noise1e-3 <= 2.1e-6.
  Rejected: column slope of log error on log |w| (q_norm.weight 1-99 % quantiles 0.84-1.25, too flat); per-row
  rms spread (bf16 output 1.2e-4 vs noise 3.5e-4, margin too thin).
- BRINGUP_IMPL=mutations runs a local sweep (controls reference, input nudge, bf16 output; every float mistake) with
  the built-in verdict (evaluate + error_gate) plus the norm check: all controls pass, all mistakes caught
  (noise1e-2 by norm_bias on all 5 inputs). Reference PASS, stub FAIL (bias -1, rel 1).
- Gate (device; the q_a module already serves enc_source): PASS, err_q_a_L02 0.00806 <= 0.0737, vs CPU rel 9.2e-5,
  norm_bias -8e-9 on every input, A,B,A bit-identical. Metrics `norm_bias_q_a_L02_<input>`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_source_q_a.py`
  (BRINGUP_IMPL=mutations | reference | stub for the freeze checks).

## C.enc_source.kv test review (attempt 1)
- Freeze without review failed: the built-in sweep let scale1.02, scale0.98 and noise1e-2 slip (rel limits 0.0464
  golden / small / big, 0.0589 layer24, 0.0619 mixed from the input nudge's fp8 flips, DS41-1; K-36).
- Same step as swa L0 (ds41_ref.py:_window_kv, ratio RoPE table), so DS41-14's fp8 checks were ported from
  test_c_swa_kv.py, with the sweep in the current form (q_a test: evaluate + error_gate, controls reference / input
  nudge / bf16 output). off grid == 0 catches scale and noise (1.0 off grid on all 5 inputs).
- DS41-46: the input-nudge control flips 0.260 / 0.298 / 0.315 of the values (golden-small-big / layer24 / mixed),
  so DS41-14's 0.15 mismatch limit fails the floor here; FP8_MISMATCH = 0.63 (2 x 0.315). CPU probe (deleted, K-56):
  RTZ e4m3 0.49 (caught by auto vs-CPU on 5 inputs), rope pos+1 0.038 (caught by auto), bf16 GEMM output 0.018,
  attn_norm not fp8-quantized 0.36-0.43 (passes auto and 2 x nudge: not separable at this floor).
- Results: mutations PASS (all mistakes caught), reference PASS (err 0.0232, mismatch 0), stub FAIL. Device gate
  (kv module from C.swa.kv already serves enc_source): PASS, err_kv_L02 0.0232 <= 0.112, vs CPU rel 2.6e-3,
  off grid 0, mismatch 0.0077-0.0102, A,B,A bit-identical. The FAIL lines in the log come from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_source_kv.py`
  (BRINGUP_IMPL=mutations | reference | stub for the freeze checks).

## C.enc_source.compressor test (attempt 1)
- Previous sweep: scale 1.02 / 0.98 slipped the auto checks (rel 0.020 vs limits 0.031-0.075; the golden's comes
  from the official arithmetic, DS41-38). Output [1024, 640] = cat(ckv fp4 x e4m3 scale per 16, k fp4 x ue8m0 per 32).
- Added (as DS41-14 / DS41-46 did for the fp8 kv), on all 6 inputs vs the CPU step on the same inputs:
  fp4 grid (re-quantizing either half changes nothing: limit exactly 0; scale mutations 0.86-0.93 off grid) and
  exact-match fraction <= 0.023 (2 x the nudge's worst half, ckv 0.0114; whole-output nudge 0.006-0.010).
  BRINGUP_IMPL=mutations runs the sweep with these checks (pattern of test_c_enc_source_kv.py).
- Measured mistakes (CPU): rope +1 position, ckv block 32, ckv ue8m0 scale, no fp4: all caught by the auto rel
  limits already. model.py's bf16 points ahead of fp4 (DS41-38): rel 0.036 > chunk0 / small / big limits 0.031-0.032
  -> the device step must keep its intermediates in fp32 (DS41-47).
- Results: mutations PASS (all 6 mistakes caught, 3 controls pass); reference PASS (err_compressor_L02 0.0155 <=
  0.1905); stub FAIL. Default (device) mode fails with "no device module for compressor yet" (implement step next).
- Re-run: `BRINGUP_IMPL={reference,stub,mutations} scripts/run_safe_pytest.sh --run-all
  models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_source_compressor.py`.

## C.enc_source.compressor implement (attempt 1)
- New `tt/compressor.py` (TtCompressor, CompressRope, TtServedRows, TtCompressCaches), `tt/quant.py:fp4_qdq`, setting
  COMPRESSOR_X, hooks `_compressor_host_fn` + device step "compressor", DEVICE_STEPS["enc_source"] = attn_hc .. kv
  (passed C.enc_source.* / S.enc_source.01-05 on the swa modules, DS41-28) + compressor; _HybridState holds the
  served compressed caches of compressor layers (state.served adds them when the layer's attention is on the device;
  formats() lists them, bf16).
- Chain per chip: x fp32 [S/SP, H/TP] (TF32-rounded, COMPRESSOR_X=round_tf32, DS41-31) @ the chip's K rows of
  [wkv | wgate]^T (bf16 as stored, HiFi4, fp32 acc) -> tp_sum (exact) -> reshape [S/SP/2, 2048] pairs -> softmax
  (max, exp, sum, divide; fp32 SFPU) -> kv0 w0 + kv1 w1 -> compressor.norm (fp32) -> RoPE at positions 2j from a
  load-time ROW_MAJOR table (sliced at start / 2, any 16-aligned offset) -> fp4_qdq(16, e4m3) = ckv; latent @ wk^T
  (bf16, HiFi4) -> k_norm -> RoPE -> fp4_qdq(32, ue8m0) = k. Rope dims de-interleaved at load in wkv / wgate rows,
  compressor.norm, wk columns (latent order) and rows, k_norm (DS41-15). Intermediates fp32 to fp4 (DS41-47).
- fp4_qdq (DS41-48): exact without dividing: e2m1 code = sum of 7 compares of |x| with midpoint x s (exact products),
  > or >= by the tie rule; e4m3 scale = round_e4m3(divide(amax, 6.0)) (the scalar divide matched torch on 524k values);
  0 mismatches vs kernels.fp4_quant_dequant on random / wide / tiny / zero / exact-tie sets (probe_002).
  ttnn.max over 16-wide groups (reshape [N, 16]) is exact (probe_001); tensor / tensor ttnn.divide is 1 ulp off 6.6 %.
- Served caches (DS41-48): token units: compressed row j = r token rows of width / r (ckv [.., 256], index [.., 64]),
  init_kvpe_cache record shard [1, 1, 32, 256] = the contract's [16, 512] record; write = TtWindowCache's path
  (all_gather over SP, rows of incomplete groups ((j + 1) r > end) zeroed by the load-time iota, embedding to server
  order, update_padded_kv_cache(kv_actual_global=start, valid_global=end)); ckv TP-dedup, index keys TP-replicated.
  Probe_003: starts (0, 2048)+(2048, 3017), 1000+(960, 3000), 1330+(1312, 3001), pulled back 1000+(960, 3008)+
  (2048, 4000): every row == expected (prefix untouched, incomplete / pad rows zero); load_host round trip exact.
- Not here: the sparse-attention table (all_gather of the source's served rows) belongs to enc_source attention; no
  device compress_partial (always zero: starts are 32-aligned, chunks even; the hybrid zeroes the reference's).
- Gate: PASS, err_compressor_L02 0.0159 <= 0.1905; vs CPU rel 0.0078 (limit 0.0746); chunk0 / layer24 / mixed /
  small / big rel 0.0080 / 0.0089 / 0.0099 / 0.0078 / 0.0082 (limits 0.031-0.042); fp4 off grid 0, mismatch
  0.0011-0.0019 (<= 0.023); A,B,A bit-identical. COMPRESSOR_X=truncate: err 0.0158, vs CPU 0.0077, mismatch
  0.0012-0.0019 (no measurable difference; round_tf32 kept as the unbiased read). The own-floor WARN lines
  (row / column inf x) are not gating (DS41-43). The FAIL lines at the top of the log are the precompile pass (K-52).
- Hybrid probe_004 (device_model(mesh, spec, [2]), golden chunks 0, 1): block out PCC 0.99964 / 0.99957; state
  compress_kv PCC 0.99988, index_k 0.99991, partial exact; served caches == the reference state bit for bit.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_source_compressor.py`;
  probes `scripts/tt-probe.sh ds41_compressor < tests/ttnn/unit_tests/operations/ds41_compressor/probes/probe_00{2,3,4}.py`.

## C.enc_source.indexer implement (attempt 1)
- New `tt/indexer.py` (TtIndexer, build_indexer), hooks `_indexer_host_fn` + device step "indexer" (only a layer with
  its own keys and no candidate pool: enc_source), DEVICE_STEPS["enc_source"] += indexer. Fork `indexer_score`
  extended (DS41-49): key_compression_ratio 2 accepted; ring opt-in `natural_query_blocks` (default off, hashed).
- Chain per chip (r, c), Sq = W / TP query rows = positions start + r W + c Sq + i: gates = round_tf32(attn_norm)
  (COMPRESSOR_X, the compressor's fused fp32 projection) @ (weights_proj / 64)^T K-split (bf16, exact) -> tp_sum
  (exact) -> mesh_partition (TP) -> bf16; q = mesh_partition(q_resid) -> fp8_qdq -> @ wq_b^T (replicated, every
  head's rope rows de-interleaved at load) HiFi4 fp32 acc -> [1, 32, Sq, 128] -> rope_heads(interleave=True, fp32
  SFPU) -> fp4_qdq(32, ue8m0) bf16; keys = the served 2_index_k_r2 cache (TtServedRows, RM token units [rows, 64])
  -> to_memory_config(interleaved) -> reshape [rows / 2, 128] -> TILE (block-cyclic W / 2 keys per SP rank per chunk:
  the ring's k_local as is); ring_indexer_score_dsa(cluster_axis 0, seq_subshard_axis 1, block_cyclic_chunk_local W,
  ratio 2, natural_query_blocks, kv_len = ceil32((start + S) / 2), HiFi4 + fp32 DEST, q 64 x k 32, K-13 / K-14) ->
  topk_large_indices(512, valid_length kv_len) -> all_gather axis 1 (uint32 RM [512, 512] per chip: no K-17 hang).
- Why the fork: the source refuses ratio 2, and its block-cyclic query geometry is rotation-exact (the server's order
  at a mid-slab start), while DS41 computes natural blocks at every 32-aligned start: at start 32, 16384 future
  columns of chip (0, 1) came out finite with the option off (fork test). chunk-aligned starts agree either way.
- Harness boundary: component tests build a TtServedRows index cache per call (golden prefix + compressed's last 128
  columns via load_host); the hybrid reads state.compress[layer].index_k (the device compressor wrote the chunk) and
  writes the reference state's topk_idx rows (reader layers, ladder state). Output pads 0xFFFFFFFF -> -1, int32.
- Gate: PASS, err_indexer_L02 0.0174 <= 0.0329 (2 x the nudge floor's miss 0.0165); vs CPU overlap 0.9944 (worst row
  0.9805), vs golden 0.9826 / 0.9434, chunk0 0.9985 / 0.9844, layer14 0.9957 / 0.9297; A,B,A bit-identical.
- Hybrid probe (tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_002.py: device_model [2], golden chunks
  0, 1): block out PCC 0.999643 / 0.999550 (compressor-only hybrid: 0.99964 / 0.99957), topk_idx state vs golden
  overlap 0.98778 (worst row 0.95117), valid counts equal on every row.
- Fork regression (option off): tests/unit 17 passed + 1 skipped (2x4), tests/test_indexer_score.py 2 passed,
  fork_source 134 tests, 0 regressions. New op test 6 starts exact masks, PCC 0.9999985.
- Perf notes for later: the key view converts the whole served index cache capacity per chunk (32 MiB per chip at 1M
  on 4x4), and weights_proj has its own exchange (the plan fuses it into the compressor's projection).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_source_indexer.py`;
  op test `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/indexer_score/tests/unit/test_natural_query_blocks.py`;
  hybrid `scripts/tt-probe.sh ds41_indexer < tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_002.py`.

## S.enc_source.07 implement (attempt 1)
- No code change. The indexer module (C.enc_source.indexer) is already a device step of enc_source
  (hooks.py DEVICE_STEPS, device_component "indexer"); the gate fails only `swap_indexer_out` rel 0.022992 <= 0.01
  ("fixed"); err_swap_out 0.0304 <= 0.0632 and every step check pass.
- Why fixed: `_check_tail` takes its limit from the nudged noise-floor outputs only when they are float; the top-k is
  int32, so it falls back to swap_out_rel 0.01 (DS41-50).
- Probe 003 (ds41_indexer): the same tail from the framework's own nudged CPU indexer runs is 0.0255-0.0263 (it fails
  0.01 too); bf16-rounded CPU scores (the official index_score dtype) 0.0234; device 0.0230, overlap vs CPU 0.9945.
  The device is at its noise floor; no truncation / bias / wrong dtype found (rule 14). Raising the scores to fp32
  would exceed the reference (bf16 index_score, model.py:557), so not done. Needs an owner waiver for
  swap_indexer_out (tests.waivers indexer) or a framework floor for int outputs. Detail: known_issues_detail.md DS41-50.
- Re-run: gate `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_enc_source_07_indexer.py`;
  probe `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_indexer < tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_003.py`
  (its random-tie variants crash on -inf + inf after the lowest-tie row; the rows above it are the evidence).

## S.enc_source.07 implement (attempt 2)
- Fixed the attempt-1 knowledge-check failure: the DS41-50 detail section heading in known_issues_detail.md now equals
  the bullet title ("The tail check after an index step uses a fixed 0.01; its noise floor is 0.026"); `knowledge.check`
  is clean (NEXT ID DS41-51).
- No code change (rule 14, DS41-50). Gate re-run: identical to attempt 1, deterministic: err_swap_out 0.030351 <= 0.063240,
  every step check ok, only `swap_indexer_out` FAIL rel 0.022992 <= 0.010000 (fixed), worst row 0.145.
  `_check_tail` (testing/component.py:552-580) uses the noise-floor limit only for float step outputs; the top-k is int,
  so it falls back to swap_out_rel 0.01. The bf16-score reference arithmetic (model.py:557) gives 0.0234 and the nudged
  CPU indexer gives 0.0255-0.0263, so the step cannot pass without an owner waiver (spec tests.waivers: enc_source indexer,
  read by NF.waive at component.py:597) or a framework floor for int outputs.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_enc_source_07_indexer.py`.

## S.enc_source.07 implement (attempt 3)
- No code change (rule 14, DS41-50). Checked `hf/inference/model.py:551-558` again: q, index_k, the einsum and
  weights_proj run in the default bf16, so bf16 scores are the reference's precision and fp32 scores would exceed it.
- Gate re-run, identical (deterministic, 3rd time): err_swap_out 0.030351 <= 0.063240, all step checks ok, only
  `swap_indexer_out` FAIL rel 0.022992 <= 0.010000 (fixed; `_check_tail` component.py:552/580 skips the noise model for
  the int top-k). The nudged CPU indexer gives 0.0255-0.0263 on this check, so it needs an owner waiver
  (spec tests.waivers indexer) or a framework floor for int outputs; nothing in tt/ can move it at the reference's precision.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_swap_enc_source_07_indexer.py`.

## S.enc_source.07 implement (attempt 101, expert debugger)
- Fresh check of DS41-50, no code change. Probe `tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_004.py`:
  device top-k has no duplicates, valid counts equal on all 2048 rows; the 101 same-set rows differ only in order
  (their tail rel goes 3.9e-4 -> 0 in the CPU order). The 0.023 is all near-tie flips, below the nudged CPU
  indexer's own 0.0255-0.0263 (probe_003). No device bug; `swap_indexer_out` needs an owner waiver
  (spec tests.waivers, enc_source indexer) or a framework noise floor for int step outputs in `_check_tail`.
- Re-run: `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_indexer < tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_004.py`.

## S.enc_source.07 implement (attempt 102)
- No code change (rule 14, DS41-50). Probe `tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_005.py` splits
  the device's two bf16 points: bf16 scores alone 0.0234, bf16 gates + bf16 scores 0.0224 (device 0.0230), and
  bf16 gates + fp32 scores (the device with an fp32 top-k) 0.0102, still over the fixed 0.01. No fp32 top-k exists
  (topk_large_indices / ttnn.topk are BFLOAT16-only). Passing would need fp32 gates + scores + a new fp32 top-k,
  which is the CPU's arithmetic and finer than model.py's bf16, and that is an owner decision. Otherwise the check
  needs a waiver (spec tests.waivers, enc_source indexer) or a framework int-output floor in `_check_tail`.
- Gate re-run (4th time, deterministic): err_swap_out 0.030351 <= 0.063240; only `swap_indexer_out` FAIL 0.022992 <= 0.01 (fixed).
- Re-run: `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_indexer < tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_005.py`.

## S.enc_source.07 implement (attempt 103, expert debugger)
- No code change (rule 14, DS41-50). Checked that the framework's nudged runs (`_capturing`, component.py:330-345)
  use the same device-chain inputs as the device indexer, so their 0.0255-0.0263 tail is a valid floor; model.py:551-558
  computes index_score in bf16, and FIX.2 forbids raising precision above the reference to pass a fixed bar.
- Probe `tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_006.py`: the frozen gate with only the
  `is_floating_point()` filter lifted in `_check_tail` (component.py:600) passes every check: swap_indexer_out
  0.022992 <= 0.052679 (2 x floor 0.026339), err_swap_out 0.030351 <= 0.063240, run_swap_test True. The step needs
  that framework change (int step outputs get the nudged tail floor) or the owner waiver (spec tests.waivers, indexer).
- Re-run: `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_indexer < tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_006.py`.

## C.enc_source.attention implement (attempt 1)
- New `tt/sparse_attention.py:TtSparseAttention` (subclass of TtWindowAttention) for a compressed layer: one
  `ttnn.bringup.sparse_sdpa` per chunk at the window step's dtypes (DS41-18: high_precision + sink / 1, scale 1 with
  512^-0.5 in q (K-27), fp32_scores, bf16_probs, bf16 out) over the table [window region 128 + S (TtWindowAttention._table,
  writes the served 0_window_kv) | compressed rows]. The compressed rows are the KV source's served cache
  (TtServedRows, TP-dedup, token units, DS41-48): the local rows of the first nb = ceil((start + S) / S) chunk periods,
  viewed as [nb stripe / 2, 512], tiled, all_gather axis 1 then axis 0 (gather order). Top-k group j -> table row
  128 + S + q nb hs + b hs + j mod hs (hs = stripe / r, s = j div hs, q = s mod N, b = s div N), fp32 integer ops on the
  device (floor((x + 1/2) / d), exact below 2^24; -1 kept via int32 -> bitcast uint32, K-18).
- Index list [window 128 | top-k 512] (640 = 5 k-chunks of 128). sparse_sdpa stops at the first sentinel
  (sparse_sdpa_reader.cpp:9), so for starts < 128 the list goes through `ttnn.gather` with a load-time permutation per
  start class (`SparseIndexTables`; TtCSA.index_tables' idea); start >= 128 is a plain concat.
- q and the table keep the rope tail interleaved (`INTERLEAVED = True`): the stored compressed rows are interleaved, so
  no per-chunk transform of the whole compressed table; q via rope_heads(interleave=True), the output de-interleaved
  (new `tt/rope.deinterleave_tail`) before the inverse RoPE. TtWindowAttention keeps INTERLEAVED = False (C.swa.attention
  re-run: PASS, err 0.0235, unchanged).
- hooks: `_sparse_attention_host_fn` (component tests: a window cache from the golden ring and a compressed cache from
  state_prefix compress_kv + this chunk's `compressed`; hybrid: ctx.extra window_cache / compress_cache.ckv), the
  `attention` step dispatches compressed layers to it (own KV and top-k source only so far), attention added to
  DEVICE_STEPS['enc_source'] (DS41-28). K-31 is out of date: the fork takes the sink under high_precision.
- Gate: component test FAILS on one check only: auto_attention_L02_small rel 0.0352 > 0.03 (the fixed second-input
  floor). Passing: err_attention_L02 0.0350 <= 0.1616, vs CPU 0.0316 <= 0.0668, chunk0 0.0292 <= 0.0545, layer14
  0.0287 <= 0.0529, mixed 0.0298 <= 0.0505, big 0.0230 <= 0.0491, determinism bit-identical. Contract
  test_cache_starts (BRINGUP_TASK=C.enc_source.attention): 4 passed.
- Why `small` fails (DS41-51, probe ds41_sparse_attn/probe_004): the official kernel's own bf16 o alone gives 0.0329 on
  those inputs (fp8 flips after it); the device 0.0352 (o vs CPU rel 0.0023, a flat-softmax shrink -8.9e-4, DS41-16).
  No device fix at the reference's precision (fp32 o is above it, FIX.2): needs an owner waiver (tests.waivers
  enc_source.attention) or a framework second-input floor that includes the official arithmetic.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_source_attention.py && BRINGUP_TASK=C.enc_source.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`;
  probe `CASE=small,golden PYTHONPATH=$PWD scripts/tt-probe.sh ds41_sparse_attn < tests/ttnn/unit_tests/operations/ds41_sparse_attn/probes/probe_004.py`.

## C.enc_source_engram.attention implement (attempt 1)
- No new module: layer 14's attention is `tt/sparse_attention.py:TtSparseAttention` (C.enc_source.attention), which
  the `attention` step already dispatches to for a layer that is its own KV and top-k source. The component test
  passed on it before any change (err_attention_L14 0.0457 <= 0.4871; vs CPU 0.0428 <= 0.0875; chunk0 / layer2 /
  mixed / small / big all within limits; determinism bit-identical).
- The previous run's only failure was the contract test (DS41-28): hooks `DEVICE_STEPS` had no `enc_source_engram`
  entry, so the hybrid ran layer 14 wholly on the CPU: `served()` raised NotImplementedError, and the CPU engram hash
  hit PAD_ID (IndexError 4294967295, DS41-26; the device `engram_rows` step hashes only [start, actual_end)).
  Fix: `DEVICE_STEPS["enc_source_engram"]` = every step that passed for the block type (engram_rows, engram_kv,
  engram_add, attn_hc, attn_collapse, attn_norm, q_a, kv, compressor, indexer, attention).
- The components entry's "searched" line saying sparse_sdpa(high_precision=True) refuses attention_sink is out of
  date (see C.enc_source.attention above; the fork takes the sink): no fork change needed.
- Gate: component PASS; test_cache_starts (BRINGUP_TASK=C.enc_source_engram.attention, layers [14]) 4 passed,
  0 failures each.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_source_engram_attention.py && BRINGUP_TASK=C.enc_source_engram.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`

## C.enc_reuse.attention implement (attempt 1)
- No new module: layer 3's attention is `tt/sparse_attention.py:TtSparseAttention` (C.enc_source.attention) run on its
  sources' state (layer 2: `cfg.kv_source_of` / `index_source_of`). The module already takes the KV source's served
  compressed cache and a top-k tensor; only the harness boundary and the hybrid wiring were new.
- hooks `_sparse_attention_host_fn` generalized: inputs (qr, kv[, compressed][, topk]) per the block graph (compressed
  only on its own KV source, topk only on its own index source). Component tests: a reader loads the compressed cache
  from `source_state[kv source]["compress_kv"][: (start + S) / r]` and the top-k from `source_state[index
  source]["topk_idx"][start, start + S)` (F65; no swap_context hook needed, K-43). Dispatch: every compressed layer
  except the decoder ones (candidates) and dec_reindex (own top-k over another layer's KV) goes to it.
- Hybrid model: `_indexer_host_fn` keeps its device top-k ([1, 1, S/SP, 512] uint32, TP-replicated) in
  `_HybridState.topk[layer] = (start, tensor)` when `ctx.extra["topk_store"]` is set (freed on the next chunk / state
  free); `HybridDeviceModel.layer` gives a reader `compress_cache` = its KV source's served caches and `topk_device` =
  the index source's device top-k of the same chunk start (else the reference state's rows). Probe
  (ds41_reuse_attn/probe_001): layer 3 output with the device top-k == the host top-k path, bit for bit, chunks 0
  and 2048.
- `_topk_rows` converts to TILE only when the top-k is not TILE already and frees only its own copy (K-1).
- `DEVICE_STEPS["enc_reuse"]` = attn_hc, attn_collapse, attn_norm, q_a, kv, attention (DS41-28: the earlier
  C.enc_reuse.* steps passed on the swa modules with no entry).
- Gate: component PASS, err_attention_L03 0.0331 <= 0.1817; vs CPU 0.0301 <= 0.0634; chunk0 0.0298 <= 0.0550;
  layer24 0.0380 <= 0.0670; mixed 0.0426 <= 0.0662; small 0.0385 <= 0.2; big 0.0235 <= 0.2; determinism
  bit-identical. Contract test_cache_starts (BRINGUP_TASK=C.enc_reuse.attention, layers [2, 3]): 4 passed, 0
  failures. It checks the served caches only, not layer 3's attention output (hence the probe).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_enc_reuse_attention.py && BRINGUP_TASK=C.enc_reuse.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`;
  probe `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_reuse_attn < tests/ttnn/unit_tests/operations/ds41_reuse_attn/probes/probe_001.py`.

## C.dec_candidate.indexer implement (attempt 1)
- New `tt/indexer.py:TtCandidateIndexer` (subclass of TtIndexer, picked by `build_indexer` for `cfg.is_cand_source`).
  TtIndexer's `__call__` split into `scores` (ring score, unchanged) + `topk_of` (top 512 + TP gather); C.enc_source.indexer
  re-run: PASS, same numbers (err 0.0174).
- Layer 20 = TtIndexer at ratio 1 (served 4_index_k_r1 cache, [rows, 128], ring_indexer_score_dsa key_compression_ratio 1 +
  natural_query_blocks, HiFi4 + fp32 DEST, k_chunk 32: K-13 / K-14; no fork change). The top-k is over every visible key
  (layer 20 is not restricted to candidates, reference `_indexer`).
- Candidate pool from the same scores, per chip: slice to kv_len, -inf pad to a multiple of 256 keys, block max over 8 by
  8 strided `ttnn.slice` (step 8) + `ttnn.maximum` (exact; probe ds41_cand/probe_001: 9 ms at [512, 55296] vs 18 ms
  reshape + ttnn.max, 17 ms permute + max), TILE; pin = + 1e30 at block (start + r W + c Sq + i) // 8 from a load-time
  fp32 row-position column + block iota (add start, x 1/8, floor, eq: exact below 2^24; -inf + 0 stays -inf); RM, -inf
  pad to 2048 blocks when fewer; `topk_large_indices(k 2048)` -> [1, 1, Sq, 2048] uint32 (0xFFFFFFFF = none). Not
  gathered over TP: dec_reindex scores the same chip rows (and K-17's wide uint32 gather hang is avoided).
- hooks: indexer step for the candidate source; `_keep_candidates` writes the reference state's candidates rows
  (ascending, -1 last: the reference's format) and keeps the device pool in `_HybridState.cands` (ctx.extra
  cand_store) for this chunk's readers; `DEVICE_STEPS["dec_candidate"]` = attn_hc .. kv, compressor, indexer (DS41-28).
- Gotcha DS41-52 (K-1): a full-range ttnn.slice returns a new wrapper over its input's buffer; freeing it freed the
  scores. Fixed with `_same` (buffer_address).
- Gate: PASS, err_indexer_L20 0.0260 <= 0.0525; vs CPU overlap 0.9974 (worst 0.9629), vs golden 0.9740 / 0.9258,
  chunk0 0.9987 / 0.9863; A,B,A bit-identical.
- Pool checks (probes ds41_cand/probe_002-004): golden chunks 0 / 1 (<= 512 blocks: all picked) equal to golden; at 53k
  (s55296 golden chunk 26, 6912 blocks): device pool vs `select_candidate_blocks` on the device's own scores overlap
  0.99958, the 1171 differing rows all pure ties; vs golden 0.9676 (worst 0.9302; top-k 0.9579 / 0.9023, bf16
  scores, K-42); own block in every row, no future block, no repeats; forward 22 ms per chunk at 53k on 2x2. Hybrid
  device_model [20], golden chunks 0, 1: out PCC 0.99973 / 0.99976, state candidates == golden, topk_idx 0.9777.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_dec_candidate_indexer.py`;
  probes `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_cand < tests/ttnn/unit_tests/operations/ds41_cand/probes/probe_00{3,4}.py`.

## C.dec_candidate.attention implement (attempt 1)
- No new module: layer 20 (ratio 1, its own KV and top-k source) runs `tt/sparse_attention.py:TtSparseAttention`, as
  layer 2 does. The candidate pool restricts only later indexers (dec_reindex), never layer 20's attention (reference
  `_attention` is the same code at r = 1). hooks: the attention dispatch now skips only `cfg.uses_candidates` (dec_reindex)
  instead of `uses_candidates or is_cand_source`; `"attention"` added to `DEVICE_STEPS["dec_candidate"]` (DS41-28).
  At r = 1: hs = stripe, table = [window 128 + S | N nb stripe compressed rows] (56 MB at 55k on 2x2), top-k always 512.
- Fix in `tt/compressor.py:TtCompressor._gathered` (DS41-52, K-1): at ratio 1 a chunk ending before its last row
  (actual_end < start + S) masks rows with a slice of the load-time `tables.iota` over its whole length; the full-range
  slice returned a wrapper over the iota's buffer and the deallocate freed the shared iota ("Input Tensor is not
  allocated" in the next slice). Now the iota itself is used when total == its rows and is never freed. Ratio 2
  (total = S / 2) and full chunks (no mask) are unchanged, so earlier compressor gates are unaffected.
- Components entry note (sparse_sdpa high_precision refuses the sink, cpp:132) is out of date: the fork takes the sink
  (lines 127-139 only require fp32 DEST / bf16 q, kv); no fork change.
- Gate: component PASS, err_attention_L20 0.0387 <= 0.4751; vs CPU 0.0335 <= 0.0709; chunk0 0.0340 <= 0.0722; mixed
  0.0415 <= 0.0823; small 0.0384 <= 0.2; big 0.0282 <= 0.2; determinism bit-identical. test_cache_starts
  (BRINGUP_TASK=C.dec_candidate.attention, layers [20]): 4 passed, 0 failures each.
- Re-run: `PYTHONPATH=$PWD BRINGUP_TASK=C.dec_candidate.attention scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_dec_candidate_attention.py && PYTHONPATH=$PWD BRINGUP_TASK=C.dec_candidate.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`

## C.dec_reindex.indexer implement (attempt 1)
- New `tt/indexer.py:TtReindexer(TtIndexer)` (picked by `build_indexer` for `cfg.uses_candidates`): TtIndexer.scores at
  ratio 1 on the KV source's (layer 20's) served 4_index_k_r1 cache (ring_indexer_score_dsa fork, fp32 DEST + HiFi4,
  k_chunk 32: K-13 / K-14; natural_query_blocks: DS41-49; no fork change), then the candidate restriction per chip:
  `ttnn.scatter` of layer 20's pool ids [1, 1, Sq, 2048] uint32 RM as 0 into a load-time -inf [1, 1, Sq, ceil16(T/8)]
  bf16 RM row (K-4: bf16, not fp32 TILE; DS41-53: 0xFFFFFFFF ids are skipped by scatter, input not written) ->
  `ttnn.repeat_interleave(8, dim=-1)` -> slice to kv_len -> RM `ttnn.add` to the scores (exact: s + 0, s + -inf) ->
  topk_large_indices(512) -> all_gather axis 1. Same chip rows as layer 20's pool (not gathered over TP), so no
  reorder. No query blocking (the components note's 128 rows per pass at 1M): the existing indexers score whole chip
  rows; left for perf.
- hooks: `_indexer_host_fn` takes the re-index case (keys: ctx.extra compress_cache = the KV source's, else
  source_state[20] index_k [0, end); pool: ctx.extra cand_device, else source_state[20] candidates rows uploaded per
  chip (r, c) = rows r W + c Sq); the indexer dispatch builds it for `uses_candidates`; `HybridDeviceModel.layer`
  passes `state.cands[20]` of the same chunk as cand_device; `DEVICE_STEPS["dec_reindex"]` = attn_hc .. kv + indexer
  (DS41-28: attn_hc .. kv had passed without an agent). The attention of layer 24 stays on the CPU (not built yet).
- Gate: PASS, err_indexer_L24 0.0306 <= 0.0577; vs CPU 0.9887 (worst 0.9688), vs golden 0.9694 / 0.9219, chunk0
  0.9955 / 0.9727, layer0 0.9902 / 0.9082; A,B,A bit-identical. The gate chunk (s4096 chunk 1, <= 512 blocks) never
  restricts the pool, so probe ds41_reindex/probe_002 checks it at 53k (s55296 golden chunk 26, 6912 blocks): every
  pick inside the row's candidate blocks (2048/2048 rows), no future pick, 512 valid per row, no repeats; vs the
  reference masking on the device's own bf16 scores 0.9806, all 2016 differing rows pure ties; unmasked control
  0.364; vs golden 0.948 (worst 0.902; layer 20's top-k there 0.958, bf16 scores K-42). Time per chunk at 53k on 2x2:
  16.4 ms (scores + top-k 12.5, bias 3.3).
- Hybrid probe_003 (device_model [20, 24], golden chunks 0, 1): layer 24 read cand_device and layer 20's device cache
  both chunks; block out PCC 0.99973 / 0.99979; topk_idx state overlap 0.9737 (worst 0.910).
- C.dec_candidate.indexer and C.enc_source.indexer re-run after the host fn change: PASS, same errors (0.0260, 0.0174).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_dec_reindex_indexer.py`;
  probes `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_reindex < tests/ttnn/unit_tests/operations/ds41_reindex/probes/probe_00{2,3}.py`.

## C.dec_reindex.attention implement (attempt 1)
- No new module: layer 24 (ratio 1, its own top-k over layer 20's KV) runs `tt/sparse_attention.py:TtSparseAttention`
  (reference `_attention` is the same code for every compressed layer). Table = [window 128 + S | layer 20's served
  3_compress_kv_r1 rows in gather order]; index list = window ids + layer 24's own top-k mapped to table rows on the
  device (`_topk_rows`), sentinel tail (SparseIndexTables for starts < 128).
- hooks: removed the two `return None` guards in `_device_step_fn` (uses_candidates; index source without own KV) and
  the matching assert in `_sparse_attention_host_fn`; its existing paths already cover the case (compressed rows from
  ctx.extra compress_cache = layer 20's in the hybrid, else source_state[20] compress_kv [0, end); top-k = the step's
  own `topk` input). `"attention"` added to `DEVICE_STEPS["dec_reindex"]` (DS41-28).
- Components note on K-31 (fork refuses the sink under high_precision) is out of date, as earlier steps found: no fork
  change.
- Gate: component PASS, err_attention_L24 0.0409 <= 0.5549; vs CPU 0.0352 <= 0.0748; chunk0 0.0370 <= 0.0762; layer2
  0.0485 <= 0.0840; mixed 0.0486 <= 0.0847; small 0.0347 <= 0.2; big 0.0303 <= 0.2. test_cache_starts
  (BRINGUP_TASK=C.dec_reindex.attention, layers [20, 24], DS41-36): 4 passed, 0 failures each.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_dec_reindex_attention.py && BRINGUP_TASK=C.dec_reindex.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`

## C.dec_reuse.attention implement (attempt 1)
- No new module: layer 21 (ratio 1, reader of layer 20's KV and top-k) runs `tt/sparse_attention.py:TtSparseAttention`
  through the existing reader path (`_sparse_attention_host_fn`; hybrid `HybridDeviceModel.layer` passes
  compress_cache = state.compress[20] and topk_device = state.topk[20] of the same chunk). The component gate already
  passed on it in the orchestrator's first run (err 0.0377); only the contract test failed ("layer 21: its attention
  (and served caches) is not on the device yet").
- Fix (DS41-28): `DEVICE_STEPS["dec_reuse"]` = attn_hc, attn_collapse, attn_norm, q_a, kv, attention in hooks.py
  (attn_hc .. kv had passed C.dec_reuse.* / S.dec_reuse.01-05 on the swa modules with no entry).
- Gate: component PASS, err_attention_L21 0.0377 <= 0.4881; vs CPU 0.0329 <= 0.0693; chunk0 0.0335 <= 0.0700; layer0
  0.0427 <= 0.0769; mixed 0.0416 <= 0.0800; small 0.0338 <= 0.2; big 0.0274 <= 0.2. test_cache_starts
  (BRINGUP_TASK=C.dec_reuse.attention, layers [20, 21], DS41-36): 4 passed.
- Probe ds41_dec_reuse/probe_001 (hybrid [20, 21], golden chunks 0, 2048): layer 21 used layer 20's device top-k and
  served compressed cache; output bit-identical to the host top-k path (max diff 0). (A block-out vs golden part was
  removed after the run: there is no layer_22 golden.)
- Re-run: `PYTHONPATH=$PWD BRINGUP_TASK=C.dec_reuse.attention scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_dec_reuse_attention.py && PYTHONPATH=$PWD BRINGUP_TASK=C.dec_reuse.attention scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/contract/test_cache_starts.py`;
  probe `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_dec_reuse < tests/ttnn/unit_tests/operations/ds41_dec_reuse/probes/probe_001.py`.

## M.1 assemble (attempt 2)
- Attempt 1 left no code in the tree (only the orchestrator's hybrid M.1 run: host_transfers_per_layer 40). Built the
  all-device model from scratch: `tt/model.py` (`TtDS41Model`, `TtDS41Block`, `DS41DeviceState`, `DevH`); hooks
  `device_model` returns it; setting `DEVICE_MODEL=hybrid` (env `DS41_DEVICE_MODEL=hybrid`) returns the hybrid
  (rule 10: no other env var).
- Resting layout (components.yaml header): block boundary `DevH` = streams [1, 1, S/SP, 4 x H/TP] fp32 + pending pre
  [1, 1, S/SP, 4] fp32 (DS41-3); every step is the validated module, no rewrite. Every layer runs every step on the
  device, including the steps hooks `DEVICE_STEPS` never listed (DS41-28: swa_engram's FFN side, the other types'
  FFN side passed their C.* gates on the swa modules).
- Shared work: q_a and kv share one fused [wq_a | wkv] projection (`TtQa.with_proj` added -> `TtWindowKv.from_proj`,
  kv.py's documented plan); experts and shared expert return fp32 partials, `moe_add` sums them and does the one TP
  exchange (`TtExperts.reduce`; components.yaml moe_add notes); experts get the router's top-k ids. RoPE tables (one
  per rope type), CompressRope (per ratio), WindowTables, SparseIndexTables: built once at load, shared.
- State (`DS41DeviceState`, one slot): served caches (0_window_kv every layer, compress / index per KV source); each
  index source's device top-k kept per chunk start (readers take their start's entry; `to_torch` reads them back for
  the ladder's state check), layer 20's candidate pool likewise; indexer `setup` (ring key buffer, -inf rows) in
  new_state, outside the forward. compress_partial reported as zeros (starts and S multiples of 32: never pending).
- Engram rows (host input): `begin_chunk` hashes + looks up + uploads per engram layer with the active state's tail
  (new_state registers itself as the model's active slot); the engram_rows step only hands the tensor on.
  `load_tokens` sets the tail from the last 3 ids. Embedding: [V, H/TP] bf16 ROW_MAJOR per chip, ttnn.embedding on
  host-clamped ids (K-72; uploaded in `embed`, outside `layer`).
- Freeing: each boundary is freed after its last reader; "in" never (the caller frees it), the caller's tensors and
  a step's own outputs protected (an engram layer's x shares in's pre); `Keep` marks state-owned outputs.
- Gate: PASS. host_transfers_per_layer 0; pcc_layer L00 0.99983, L01 0.99951, L02 0.99865, L03 0.99692, L14 0.99840,
  L20 0.99975, L21 0.99853, L24 0.99978; pcc_state_min 0.99588 (compress_kv L02); overlap_state_min 0.9672. Chunks
  8.3 s / 11.8 s (hybrid 61.5 / 57.8). model_load_s 288: ~34 s per layer, the routed experts' bfp8 conversion at
  load (DS41-24; the hybrid loaded experts for layer 0 only).
- test_cache_starts (all spec layers, no BRINGUP_TASK, `--no-precompile`) on the device model: 4 passed (cold_mid_end,
  follow_block, follow_tile, pulled-back last chunk), 19 min (each case reloads the model, ~5 min).
- Re-run: `PYTHONPATH=$PWD BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py`

## K.1 contract (attempt 2)
- Attempt 1 left no code (stopped on the framework's contiguous-subset ValueError, fixed by F100). Built from
  scratch: adapter `tt/runners/adapter.py` (`DS41PrefillAdapter`, registered as `deepseek_v41_flash_d_p` in
  `models/demos/common/prefill/adapter.py` ADAPTER_PATHS), `tt/runners/kv_contract.py` (engine caches, table,
  read-back), `tt/served.py` (cache storage); hooks `contract_state_pcc`.
- OV-14: engine layer k = model layer `layer_ids[k]` from `testing/contract.layer_ids_from_env` (PREFILL_LAYER_IDS in
  the framework test: 8 engine layers = 0,1,2,3,14,20,21,24; the model's own runner tests serve 0-3 with no env);
  `rt.layer_ids` exposed; acks `sink(first_layer_idx + k, request_id)`, table layer = engine index.
- Gaps (DS41-55): a run after a gap (14, 20, 24 in the framework test) starts from the golden block input of its
  first layer (`GoldenGapFeed`, the spec's rung with every chunk dumped), as testing/ladder.py; never active for a
  contiguous range. Without it layer 14+ KV cannot match (golden runs layers 4-13 etc. on the CPU).
- Caches (one format, bf16 ROW_MAJOR): one tensor per config, [slots x Lc, 1, local rows, width], ND shard
  [1, 1, 32 / r, width] ROUND_ROBIN_1D over all banks (`alloc_served`: init_kvpe_cache with a shard height; it fixes
  32). Window and compressed KV TP-dedup, index keys TP-replicated (plan / owner answer 2). Each slot's
  `DS41DeviceState` is bound to its `CacheSlab`s (`model.new_state(max_seq, slabs=..., keep_history=False)`); the
  ladder's state allocates its own slabs (slots = layers = 1), same classes, same layout.
- Compressed caches now stored in compressed rows (was token units [rows, 256] with 32-row shards: same bytes, but the
  layout test wants the [16, 512] shard). Writes: the token-order server index row as before, the slab viewed as
  [W / r, 512], then update_padded_kv_cache in compressed rows; ratio 2 through its metadata path (DS41-54; probe
  `tests/ttnn/unit_tests/operations/ds41_served/probes`, deleted after: bit-exact at 0 / 960 / 1312 / 2048).
  Readers (`TtIndexer._keys`, `TtSparseAttention._compressed`) slice the slab's compressed rows directly.
- Per chunk: the engine input is read back once (mesh column 0's rows), put in natural order with
  `WindowGeometry.server_rows(start)`; the ids are the engram host input (begin_chunk) and are embedded (host-clamped,
  K-72). Engram tail per slot from a host id history (`DS41DeviceState.note_tokens`): a follow-up at 960 after a
  1000-token turn needs ids 957-959, not the slot's last. Ack: host sink after `ttnn.synchronize_device` (K-71); D2H
  `outbound_socket_service_sync` on the same CQ. compile() runs one cold chunk in slot 0, acks off.
- Top-k / candidate stores keep only the latest chunk in serving (`keep_history=False`), so nothing grows.
- FABRIC_PAYLOAD_SIZE 4352 (K-79: the fabric default the bring-up tests ran with). Runner smoke skips (DS41-8).
- Results: test_contract PASS (contract_checks_failed 0, acks_early 0, pcc_producer_kv window 0.99777 / compress
  0.99589 / index 0.99742; fixed state window ring >= 0.99735, engram tail and compress partial exact);
  test_kv_layout, test_adapter_acks, test_runner_contract (producer fallback) PASS.
- Gotcha (DS41-52): with one slot (the ladder) the 1-row slot table's [0:1] slice is full-range and returned the
  table's own buffer, which `_put` then freed (ladder s4096 TT_FATAL input_tensor.is_allocated()); DeviceScalars
  tables now have >= 2 rows.
- Regression checks after the cache refactor: ladder s4096 PASS, identical to M.1 (worst layer 0.996916, pcc_state_min
  0.99588, overlap 0.9672, host_transfers_per_layer 0); test_cache_starts -k follow_tile (start 1312) PASS.
- The whole gate exceeds the tool's 10-minute foreground limit: run its five parts one by one (each 3-6 min, mostly
  model load); all five PASS after the last change (runner smoke skips, DS41-8).
- Re-run: the brief's gate command with `PYTHONPATH=$PWD`.

## P.1 perf (attempt 1): routed experts in bfp4 (owner pick)
- Change: EXPERTS_IMPL gains "bfp4" (tt/settings.py; choices bfp8 | bfp4 | fp4_decode). `tt/experts.py:_load_block`
  (was `_load_bfp8`) converts every expert's fused w1 | w3 [2I, H] and w2 [H, I] once at load to bfloat4_b in the
  stored [N, K] orientation, used with `ttnn.linear(transpose_b=True)` exactly like bfp8 (DS41-35); `_ffn` takes the
  same path for both block formats. Same HiFi4 + fp32 acc, same fp8 q(h). Default kept at bfp8 (owner text: "Keep
  bfp8 the default and the max precision"); MAX_PRECISION now names EXPERTS_IMPL=bfp8, so component / swap tests
  stay at bfp8. Off = EXPERTS_IMPL=bfp8 (the old path, unchanged).
- Load-time stats (CPU, every expert of the layer): `_bfp4_round` emulates ttnn's host packing
  (blockfloat_common.cpp convert_u32_to_bfp: shared exponent per 16 along the last dim, 3 magnitude bits, round to
  nearest even, saturate at 7); `_bfp4_change` -> logged as `EXPERTS_BFP4 layer=...` and kept in `bfp4_stats`.
  Checked: `tt/perf_experts.py::test_bfp4_round_matches_ttnn` (emulation == ttnn roundtrip on random, ties,
  saturation and expert matrices), `::test_transpose_b_bfp4` (device: linear(x, W bfp4 [N, K], transpose_b) ==
  linear(x, q4(W)^T bf16) bit for bit, w1 | w3 and w2; rel vs exact W 0.118 / 0.116).
- Fraction of expert values bfp4 changes / worst per-matrix ||q(W) - W|| / ||W||, per layer (96 experts each):
  L0 16.64 % / 0.1193 (e57 w1); L1 16.45 % / 0.1178 (e86 w1); L2 16.37 % / 0.1179 (e36 w1); L3 16.38 % / 0.1198
  (e20 w3); L14 16.59 % / 0.1214 (e22 w2); L20 16.48 % / 0.1192 (e12 w2); L21 16.50 % / 0.1226 (e43 w2);
  L24 16.56 % / 0.1242 (e7 w2).
- Expert weight memory per chip (0.5625 vs 1.0625 B per value, 3 x 5120 x 2304 per expert): bfp4 0.445 GiB per layer
  at 24 experts per chip: this run (8 layers) 3.56 GiB (bfp8 6.72); galaxy_4x4 40 layers 17.80 GiB (bfp8 33.62);
  galaxy_8x4 (12 per chip) 8.90 GiB (bfp8 16.81).
- Accuracy (DS41_EXPERTS_IMPL=bfp4, BRINGUP_RUNG=last ladder): PASS. pcc_layer L00 0.99965, L01 0.99921, L02 0.99803,
  L03 0.99535, L14 0.99660, L20 0.99938, L21 0.99730, L24 0.99949 (bfp8 at L.last: 0.99981 .. worst L03 0.99673);
  pcc_state_min 0.99726 (window_kv L03; bfp8 0.99801); overlap_state_min 0.99652; host_transfers_per_layer 0.
- Plain profile, same code, 2x2, chunk 53248-55296, 8 layers: experts section 2501.4 ms bfp4 vs 2522.5 ms bfp8
  (312.7 vs 315.3 ms per layer per chunk, -0.8 %; 43,264 programs both); device_ms_total 3004.1 vs 3025.3; chunk wall
  3212 vs 3241 ms; pcc_chunk_out 0.99949 vs 0.99981. No speed gain (DS41-56): bfp4 is a memory pick only.
- Load: model_load_s 427 s for bfp4 in the ladder run with the first stats code (bfp8 ~292 s); the stats were then
  made ~2x cheaper (one fp64 power-of-two scale per group instead of ldexp on every value); load ~34 s per layer
  in the profile run.
- Re-run: `PYTHONPATH=$PWD DS41_EXPERTS_IMPL=bfp4 BRINGUP_RUNG=last scripts/run_safe_pytest.sh --run-all
  --no-precompile models/demos/common/bringup/tests/test_ladder.py`; profile: the brief's second command with
  DS41_EXPERTS_IMPL=bfp4 / bfp8; checks: `scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py -k bfp4 -s`.

## P.2 perf (attempt 1): routed experts on the fused ttnn.bringup.unified_routed_expert_moe
- Change: setting EXPERTS_KERNEL {loop, unified} (tt/settings.py; default loop, owner text "default stays loop until
  the owner's table"; not a precision switch, so not in MAX_PRECISION: the component / swap tests take the env value).
  unified = `tt/experts.py:TtExperts._unified`: x = buf[:, :H] (RM slice), r_rows = pieces @ SelAll (bf16 0/1
  [3 Egp, 32] per mesh row, column le = the three piece columns of local expert r epc + le; exact, DS41-16 / DS41-23)
  -> fp32 RM [rows, 32]; then ONE unified_routed_expert_moe(x, ..., ClampedSiluGlu, high_precision, HiFi4 + fp32
  dest, routing_weights=r_rows, fp8_act_quant=True, weights_transposed=True) -> the same combine / sum / reduce.
  Weights: the same bfp8 / bfp4 [N, K] tensors as the loop (bfp8 exact, `_bfp8_exact`), w1 and w3 uploaded separately
  for unified (w13 for loop). Asserts: block-float EXPERTS_IMPL, swiglu_limit 10 (baked in the op), epc <= 32.
- Fork (ttnn/ttnn/bringup/unified_routed_expert_ffn, CHANGELOG "Opt-in DeepSeek-V4.1 expert numerics"): three
  options, default off = the old program. routing_weights: the reader builds one Float32 tile per tile-row (weight in
  every column of faces 0 / 2), compute copies it to DEST slot 2; the routing weight tensor is "the op's routing":
  row i column le of `routing_weights`. fp8_act_quant: custom SFPU `calculate_routed_h` on the fp32 DEST tile after
  ClampedSiluGlu (row amax by compare tree + SUBVEC_SHFLROR1 butterfly; ue8m0 scale from the bits; e4m3 RNE on the
  integer bits, subnormals by +2^14 - 2^14). weights_transposed: tiles read at page n K + k, matmuls transpose in1.
  Either h option also unpacks the partials straight to the fp32 DEST (DS41-58: through SrcA they are TF32).
- Gotchas: "too few lregs" with 4 live row registers (DS41-59: two passes, s / 1/s parked in the consumed up slot);
  random-weight exactness test showed 2.6 % fp8 differences that were the FPU's GEMM rounding, not the quant (DS41-60:
  selection weights -> 0 differences over 1528 x 1024); my `timeout 590` around fork_tests killed run_safe_pytest
  mid-run (cards marked dirty, next run reset them): run the model cases one case at a time.
- Fork tests: tests/unit/test_ds41_h_options.py 8 passed (exact quant + weight, weight-only, [N, K] == [K, N] bit for
  bit for bf16 and bfp8 e2m1, DS41 shapes vs float rel 0.0030-0.0058, refusals); by hand a kernel scale floor and a
  reader column off by one both failed. Regression with the options off: fork_source 75 tests, 0 regressions; model
  cases pass (mimo 1x4, gemma4 1x4, mimo 2x2, glm53 2x2, hy4 2x2; xing40 4x2 skips on 4 chips). Model case for this
  call: task O.1.
- Accuracy (DS41_EXPERTS_KERNEL=unified, bfp8 via MAX_PRECISION, each once, gate order): all 16 PASS. Component
  err_experts L00 0.0242, L01 0.0291, L02 0.0302, L14 0.0327, L03 0.0302, L20 0.0342, L24 0.0343, L21 0.0342; swap
  pcc_swap_out swa_12 0.999826, swa_engram_15 0.999762, enc_source_14 0.999556, enc_source_engram_17 0.998458,
  enc_reuse_12 0.999355, dec_candidate_14 0.999748, dec_reindex_13 0.999782, dec_reuse_12 0.999507 (swa_12's
  swap_ffn_hc noise-floor FAIL line is the owner waiver, reported WAIVED).
- Layer 0 A/B (tt/perf_experts.py::test_unified_vs_loop, 2x2, chunk 2048, random routing): bfp8 loop 317.7 vs
  unified 20.4 ms per call, bfp4 315.1 vs 20.0; unified vs loop rel 6e-5. Step times (sync-instrumented): the fused
  op 5.6 ms of ~20; the rest is the unchanged chain (fp8_qdq of x ~8 ms instrumented, slices 2.2, combine-sum adds
  2.1, dispatch 1.8, combine 1.4, pieces 1.3, reduce 1.2): next levers, not this step.
- Plain profile (test_profile.py, 2x2, chunk 53248-55296, 8 layers): experts section per chunk loop 2522.3 ms (bfp8)
  / 2501.4 (bfp4) [P.1, same loop code] -> unified 176.77 (bfp8) / 173.74 (bfp4); device_ms_total 679.6 (bfp8, was
  3025.1) / 676.4 (bfp4); chunk wall 987 / 938 ms (was 3281); pcc_chunk_out 0.99981 (bfp8) / 0.99949 (bfp4);
  host_transfers_per_layer 0.
- Re-run: `PYTHONPATH=$PWD DS41_EXPERTS_KERNEL=unified scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_experts.py` (each gate test likewise); fork tests
  `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/unified_routed_expert_ffn/tests/unit/test_ds41_h_options.py`;
  A/B `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py -k unified_vs_loop -s`;
  profile: the brief's last command with DS41_EXPERTS_KERNEL=unified [DS41_EXPERTS_IMPL=bfp4].

## P.3 perf (attempt 1): routed experts on Marko Staletovic's flat_routed_expert (EXPERTS_KERNEL=flat)
- Change: setting EXPERTS_KERNEL {loop, unified, flat}; on = flat, off = unified (the default is unified again,
  OV-16 / F104: accepting the pick flips it). `tt/experts.py`: `_load_flat` builds `FlatRoutedExpert(weights_nk=True, routed=True, x_bf16, h_bf16,
  act clamped_silu, wdtype bf8 / bf4)` once per layer from the same bf16-exact e2m1 x 2^e [N, K] matrices
  `_load_block` checks with `_bfp8_exact` (DS41-35; per device r cols + c its gtab[c, r] experts); `_flat` = the
  routing rows (pieces @ SelAll, Float32 TILE [rows, 32], `_routing_rows`, DS41-23 / DS41-16) + ONE
  `flat_routed_expert(buf, counts, region_offsets, routing_weights, fp8_act_quant=True, y_bf16, down_fp32 =
  FP32_DEST_ACC, math_fidelity = MATMUL_FIDELITY (HiFi4), hidden=H)` on the dispatched buffer in place (no x copy);
  dispatch / combine / K-slot sum / TP sum unchanged. Capacity = the dispatch cap (2048; the op needs >= 256).
- Fork (ttnn/ttnn/bringup/flat_routed_expert_ttnn, CHANGELOG "DeepSeek-V4.1 expert numerics"): routing_weights,
  fp8_act_quant, weights_nk, hidden; all off = the old program. Plan on this grid (11 x 10 p150) for bfp8 + bf16 x:
  16-column units, 3 a core, 48 gate/up + 32 down cores, mt 1 (the probe before any change: no other split fits the
  1400 KB gate/up budget with bfp8 weights). Every third h tile is split over two cores, so the quant block needed
  the half exchange (DS41-61); the routing-weight tile is read by the gate/up receiver (BRISC, NOC1: the NOC0 write
  wedge in the program factory's comment) and copied UnpackToDestFp32 into DEST tile 3 before the K loop.
- Gotchas: the official Expert.forward rounds h to bf16 before w2's act_quant (`self.w2(x.to(dtype))`); ds41_ref and
  the device quantize the fp32 r h (the device follows ds41_ref, as unified did). The fork's baseline unit run (before
  any change) already had 13 failures (TP2 / TP4 subgrid plans hit TT_FATAL `p.nsg == 1`; the MiMo perf band): not
  ours, unchanged after. The plain profile takes ~8-9.5 min (flat load ~45-54 s a layer, unified ~20 s): with a
  10-minute tool limit run it alone.
- Fork tests: tests/unit/test_ds41_h_options.py 7 passed (identity down: rw + q 1528 rows bit-exact but one near tie,
  q only, rw only, pinned schedule 2848 rows; [N, K] == [K, N] bit for bit; DS41 shapes vs float64 rel 0.0032-0.0053,
  PCC >= 0.99998; refusals); negative controls incl. per-half scales of the split tiles all fail the checker.
  Regression (options off): unit suite 19 passed / 13 failed = the baseline's ids.
- Accuracy (EXPERTS_KERNEL=flat, bfp8 via MAX_PRECISION, each frozen test once, gate order): all 16 PASS. experts vs
  the CPU step (component auto_*_vs_cpu) L00 0.00225, L01 0.00246, L02 0.00256, L14 0.00266, L03 0.00243, L20 0.00265,
  L24 0.00276, L21 0.00274; swap "swap_experts vs the CPU step" swa_12 0.00219, swa_engram_15 0.00255, enc_source_14
  0.00246, enc_source_engram_17 0.00263, enc_reuse_12 0.00248, dec_candidate_14 0.00276, dec_reindex_13 0.00277,
  dec_reuse_12 0.00277 (limits 0.070-0.100); determinism bit-identical; swa_12's swap_ffn_hc line is the owner waiver.
- Speed, layer 0, chunk 2048, same random inputs (tt/perf_experts.py::test_kernels; whole experts call / kernel part
  synced, ms): bfp8 flat 17.35 / 3.76, unified 20.35 / 6.37, loop 315.71 / 300.88; bfp4 flat 17.29 / 3.66, unified
  19.96 / 5.99, loop 314.70 / 299.75; flat vs unified rel < 1e-5, vs loop 6e-5. Flat is HiFi4 compute-bound (bfp4
  only 3 % faster; the gate/up FPU at HiFi4 ~22.7 us per 32-row sub-block on 48 cores); 13.6 of 17.4 ms is the chain
  around the op (DS41-62).
- Plain profile (test_profile.py, 2x2, chunk 53248-55296, 8 layers), experts section ms per chunk: flat 149.3 (bfp8)
  / 148.0 (bfp4); unified 176.8 / 173.7 (P.2); loop 2522.3 / 2501.4 (P.1 / X.1). Device total 652 (bfp8) / 651
  (bfp4) ms (P.2 unified 679.6 / 676.4). Owner (not done, owner rule HiFi4): with e2m1 weights srcA holds every weight
  bit in LoFi's 5, so HiFi2 may already give exact products (x's 8th bf16 bit is the open question): a test, not
  a pick for an agent.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/bringup/test_c_swa_experts.py` (each gate test likewise; DS41_EXPERTS_KERNEL
  =unified for the old path); fork tests `scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_ds41_h_options.py`; A/B `scripts/run_safe_pytest.sh
  --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::test_kernels -k "bfp8 and ful"`;
  profile: the brief's last command [DS41_EXPERTS_IMPL=bfp4].

## P.3 perf (finish, after the overseer stopped attempts 2-3 on the fork's own MIMO_FL_* knobs: OV-15 / OV-16)
- Nothing in the op or the model path changed after attempt 1; its accuracy (16 frozen expert tests PASS at bfp8) and
  profile (experts flat 149.3 / unified 176.8 / loop 2522.3 ms per chunk at bfp8; 148.0 / 173.7 / 2501.4 at bfp4;
  device total flat 652 vs unified 679.6 ms) stand and were not re-run. Marko's MIMO_FL_* / FLAT_CMB_LOG environment
  knobs left as imported (OV-16).
- EXPERTS_KERNEL default back to unified (OV-16; F104 flips it when the owner accepts); `DS41_EXPERTS_KERNEL=flat`
  selects the flat op. The orchestrator sets the switch for both sides.
- Fork regression, options off (final tree, build 06:28): unit suite (5 imported files) 19 passed / 13 failed vs 18 /
  14 at the import 9003c0fd2ca; failures all pre-existing: 12 TP2 / TP4 `p.nsg == 1` (same assert at the import) +
  the MiMo indexed_perf band (1710341 ns, faster than its 1720090 lower bound). GLM-5.3's case
  (tests/test_flat_bench.py, glm53flash preset, default options) passes. GLM's model path (tt/experts_ag.py) only runs
  at <= 64 experts per chip (2x4 / 4x4), never on this 2x2 (72 a chip), so the bench is its case here.
- Re-run: `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/
  <file>` per file (test_ds41_h_options.py is the new option tests); `scripts/run_safe_pytest.sh --run-all
  --no-precompile ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/test_flat_bench.py -s`.

## P.4 perf (attempt 1): routed experts at GLM-5.3's production precision (EXPERTS_PRECISION=glm)
- Change: setting EXPERTS_PRECISION {full (default), glm} in tt/settings.py; MAX_PRECISION gets EXPERTS_PRECISION
  "full" (EXPERTS_IMPL "bfp8" was already there), so the frozen component / swap tests keep checking full precision.
  `tt/experts.py`: `self.ecfg` = the expert matmuls' compute config (glm: LoFi, fp32 DEST = FP32_DEST_ACC; full: the
  model's `compute_config`); used by the loop's w1 | w3 / w2 linears and the unified op. The routing-weight selector
  matmuls (`_routing_rows`, `_weight_col`) stay on `self.cfg` (MATMUL_FIDELITY) in both: not expert matmuls.
  Flat (`_load_flat` / `_flat`): glm = x_bf16 = h_bf16 = False (plan args), y_bf16 = False (bfp8 TILE y; the combine
  fork accepts BFLOAT8_B TILE), down_fp32 = True, math_fidelity LoFi; full = unchanged (bf16 x / h / y,
  down_fp32 = FP32_DEST_ACC, MATMUL_FIDELITY). Unified with glm: LoFi only. V4.1 numerics (routing weight on h,
  fp8 q(h)) on in both. No kernel or fork change.
- Note vs GLM: GLM's experts_ag calls the op with y_row_major=True (bf16 row-major y, pack-untilized); the brief asked
  for y_bf16 = False, so y here is bfp8 TILE. x / h bfp8, down_fp32 1, LoFi (math_fidelity default 0) match GLM.
- Plan on 2x2 with bfp8 x / h: the same nu=3, 48 gate/up + 32 down cores, mt 1 as full (log EXPERTS_FLAT).
- Accuracy (frozen tests at defaults = full via MAX_PRECISION, gate order, each once): all 8 test_c_*_experts PASS.
  Diagnostic only (not a gate): test_c_swa_experts with glm + bfp4 fails its full-precision limits: vs CPU rel 0.131
  (limit 0.085), bias +0.034, layer24 rel 0.185 (limit 0.128): bfp4 weights + bfp8 packs (DS41-64).
- Ladder last, change on (flat, bfp4, glm): PASS. pcc_layer L00 0.99964, L01 0.99922, L02 0.99800, L03 0.99524,
  L14 0.99658, L20 0.99937, L21 0.99728, L24 0.99949; pcc_state_min 0.99710 (window_kv L03); overlap_state_min
  0.99652. (P.1 bfp4 at HiFi4: worst L03 0.99535, state 0.99726; bfp8: L03 0.99673, 0.99801.) Off side not re-run
  (the orchestrator runs it).
- Plain profile, 2x2, chunk 53248-55296, 8 layers, ms per chunk: experts on (flat bfp4 glm) 128.86 vs off (flat bfp8
  full) 149.24 (-20.4, -13.7 %; 16.1 vs 18.7 per layer; same 2624 programs); device total 631.76 vs 651.89; chunk wall
  774 vs 795; pcc_chunk_out 0.99949 vs 0.99981; host_transfers_per_layer 0 both. Every other section unchanged.
- Gotcha: a shell loop of the 8 frozen tests (~20 min) exceeds the 10-minute tool limit and is killed in the
  background; run them a few at a time.
- Re-run: on `DS41_EXPERTS_KERNEL=flat DS41_EXPERTS_IMPL=bfp4 DS41_EXPERTS_PRECISION=glm`, off `... bfp8 ... full`,
  in front of `BRINGUP_RUNG=last scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/common/bringup/tests/test_ladder.py` or the brief's profile command; frozen tests as in the gate.

## P.4 perf (attempt 2): EXPERTS_PRECISION=glm with EXPERTS_IMPL=bfp8 (owner: GLM precision without bfp4; OV-17)
- No code change: attempt 1's EXPERTS_PRECISION setting, MAX_PRECISION entries (EXPERTS_PRECISION "full",
  EXPERTS_IMPL "bfp8") and the glm path in tt/experts.py already cover glm + bfp8.
- Smoke (OV-17, not a gate): test_c_swa_experts with `DS41_EXPERTS_KERNEL=flat DS41_EXPERTS_PRECISION=glm
  DS41_EXPERTS_IMPL=bfp8` (env wins over MAX_PRECISION) PASSES the full-precision limits: vs CPU rel 0.0162 (limit
  0.0849), bias 0.0030; vs golden 0.0270 (limit 0.109); layer24 0.0208 (0.128); mixed 0.0200 (0.123). Log shows
  `EXPERTS_FLAT ... nu=3 mt=1 gate/up=48 down=32 precision=glm` (same plan as full). Compare glm + bfp4 (attempt 1):
  rel 0.131, so bfp4 weights, not LoFi / bfp8 packs, were what broke the limits there.
- Timing (tt/perf_experts.py::test_kernels, impl bfp8, layer 0, chunk 2048, 2x2, glm, ms): flat whole call 16.38 /
  op part 2.68 (second run 16.36 / 2.72); unified (LoFi only) 19.39 / 5.42; loop 296.7 / 281.9. Full at P.3 (same
  test): flat 17.35 / 3.76, unified 20.35 / 6.37. So glm saves ~1.0 ms a layer (-5.6 % of the call, -29 % of the op);
  over 8 layers ~7.8 ms a chunk (estimate, not profiled: experts ~141 vs 149.2 ms). glm flat vs unified-glm rel 0.021
  (the bfp8 x / h / y packs; unified keeps bf16 tiles).
- Not run here (orchestrator runs them, change on / off): the 8 frozen tests at defaults, the ladder, the plain profile.
- Re-run: `DS41_EXPERTS_PRECISION=glm PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::test_kernels" -k "bfp8 and fu" -s`.
