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
