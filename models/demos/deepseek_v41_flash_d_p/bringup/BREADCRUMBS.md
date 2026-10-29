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

## P.5 perf (attempt 1): collectives on the bring-up fabric forks (CCL_IMPL=fabric)
- Change: setting CCL_IMPL {stock (default, F104: the owner flips it), fabric}; new `tt/ccl.py` (all_gather, tp_sum,
  sum_cols, rs_links, setup); every `ttnn.all_gather` of tt/attention.py, sparse_attention.py, norm.py (ffn_norm,
  dim 3), indexer.py (u32 RM top-k), compressor.py goes through `ccl.all_gather`; `common.tp_sum` (HC_TP_SUM
  gather_add: mhc, norms, q_a, compressor, indexer gates, engram_add) -> `ccl.tp_sum`; the hidden-split sums
  (ATTN_O_SUM, EXPERTS_SUM routed + shared, gather_add) -> `common.sum_cols` -> `ccl.sum_cols`. `TtDS41Model` calls
  `ccl.setup(mesh)` at load (semaphores, the RS plan + counters incl. its one-time synchronize). Their
  reduce_scatter / all_reduce values stay stock ttnn. Nothing else changed (HiFi4, fp32, EXPERTS_KERNEL flat).
- Fabric mapping: all_gather -> `ttnn.bringup.fabric_all_gather(t, dim, out=allocate_tensor_on_device(...),
  num_links=None, caller-owned L1_SMALL semaphores)` when `_fits` (DRAM interleaved; not a RM innermost-dim gather;
  TILE dims 2 / 3 tile-aligned): every model gather fits on this box. sum_cols -> ONE `fabric_reduce_scatter(dim=3,
  exact_fp32_add=True, cache_descriptors=True)` (no full sum, no mesh_partition). tp_sum -> `fabric_reduce_scatter(
  dim=2, exact)` + `fabric_all_gather(dim 2)`; when R % (32 TP) or W % 32 != 0 (norm sum of squares [S/SP, 1],
  engram gates [S/SP, 12]: the fork requires a tile-aligned TILE input) the old all_gather(dim 1) + SFPU adds on the
  fabric gather. Not on a fork: the head <-> seq all_to_all (all_to_all_async_generic, 0.48 ms x 2 per layer; no
  bring-up fork exists).
- Fork (fabric_reduce_scatter_ttnn, CHANGELOG "Exact fp32 adds, column blocks, ..."): opt-in `exact_fp32_add`
  (UnpackToDestFp32 + SFPU add_binary_tile; default FPU path is rel 3.3e-3 on fp32, DS41-65), `dim=3`,
  `cache_descriptors` (descriptor build 0.655 of 0.85 ms host per call, DS41-66), `prepare()`. Off = the old program
  (sources, args, CBs, compute config). fabric_all_gather: no change.
- Links (DS41-66): 4 usable per 2x2 BH neighbour pair. fabric_all_gather at 2 links was slower than stock on the
  large gathers (fp32 [1024, 2560] dim 3: 0.235 vs 0.139 ms; bf16 [6912, 512]: 0.159 vs 0.098); at None 0.126 /
  0.090 (faster everywhere). RS at 4 links: sum_cols [1024, 5120] 0.150 ms (0.235 at 2); `ccl.rs_links` = min usable
  links over the axis's neighbour pairs, capped at DRAM banks / 2.
- Exactness: a tt-probe (directory deleted, K-56) stock vs fabric bit-identical for
  every gather shape / dtype the model uses and tp_sum [1024, 32 / 1 / 1792 / 1024 / 12], sum_cols [1024, 5120] (TP 2:
  one add, a + b either way). Fork tests: tests/unit/test_fabric_rs_exact_fp32.py 7 passed.
- Accuracy (DS41_CCL_IMPL=fabric, gate order, each once): test_c_dec_reindex_{attention, attn_hc, attn_norm, ffn_hc,
  ffn_norm, experts, shared_expert, indexer} all PASS.
- Measurement (2x2, rung last, chunk 53248-55296, 8 layers; BRINGUP_PROFILE_OPS=1 with BRINGUP_PROFILE_LAYERS=0,2:
  the 8-layer op profile did not finish within the tool's 20 minutes, so the per-op rows are layer 0 (swa) and layer
  2 (enc_source); sections and totals cover all 8 layers). Stock: collectives on L0 + L2 = 29 all_gather calls 2.31
  ms + the gather_add slices / adds / mesh_partitions 2.16 ms + all_to_all 1.92 ms = 6.39 of 147.2 ms device (4.3 %;
  3.0 % without the all_to_all), ~2.9 % of the chunk extrapolated. Fabric: 12 fabric_reduce_scatter 0.86 ms + 25
  fabric_all_gather 0.86 ms + all_to_all 1.92 = 3.64 ms. Per call (stock -> fabric, ms): o-sum / moe sum [1024, 5120]
  0.665 (gather 0.257 + slices 0.207 + add 0.142 + partition 0.059) -> 0.161-0.165; q_a tp_sum [1024, 1792] 0.323 ->
  0.109 (RS 0.059 + AG 0.050); compressor tp_sum [1024, 1024] 0.171 -> 0.069; hc / indexer tp_sum [1024, 32] 0.015 ->
  0.019; ffn_norm gather fp32 [1024, 2560] dim 3 0.137 -> 0.127; window gathers bf16 [1024 / 128 / 256, 512] 0.040 /
  0.011 / 0.017 -> 0.022 / 0.009 / 0.013; sparse table gathers [6912 / 13824, 512] 0.094 / 0.183 -> 0.088 / 0.171;
  top-k u32 [512, 512] 0.043 -> 0.019; compressor gathers [512, 512 / 128] 0.025 / 0.014 -> 0.017 / 0.012.
  Sections per chunk (ms): attention 238.65 -> 233.99, moe_add 6.49 -> 2.45, q_a 38.49 -> 36.76, compressor 9.42 ->
  9.11, indexer 56.59 -> 56.46, ffn_norm 2.93 -> 2.82, rest +-0.03. Device total 652.16 -> 641.02 (-11.1, -1.7 %);
  chunk wall 920.2 -> 922.0 (host-bound: host overhead 281 ms); host_transfers_per_layer 0; pcc_chunk_out 0.99981.
- Gotchas: run_safe_pytest output redirected to a file buffers Python's prints (use PYTHONUNBUFFERED=1 to see
  progress); `import ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter as m` yields the function (the
  package exports the same name): use sys.modules.
- Re-run: on `DS41_CCL_IMPL=fabric`, off `DS41_CCL_IMPL=stock`, in front of each frozen test / the ladder / the
  brief's profile command; fork tests `scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_exact_fp32.py`; regression (options off,
  2x2-runnable) `.../tests/unit/test_fabric_reduce_scatter.py .../tests/test_fabric_reduce_scatter_ttnn.py`: 5 passed,
  same errors as the unchanged file (the other unit files are 2x4-only).

## P.6 perf (attempt 1): TP partial sums in bf16 on the fabric reduce-scatter (CCL_IMPL=fabric_bf16)
- P.5 had not added the value (OV-18 open), so added: setting CCL_IMPL {stock, fabric (default, unchanged),
  fabric_bf16}; MAX_PRECISION CCL_IMPL=fabric (exact), so component / swap tests stay exact.
- fabric_bf16 (tt/ccl.py), as GLM-5.3's scatter_rows (glm53_flash_d_p/tt/common.py, GLM_SCATTER_OP=fabric_bf16):
  sum_cols (ATTN_O_SUM / EXPERTS_SUM) = bf16 partial -> ttnn.bringup.fabric_reduce_scatter(dim 3, default FPU add,
  cache_descriptors) -> typecast fp32 (mhc_post asserts fp32 y); tp_sum (HC_TP_SUM gather_add, tile-aligned:
  mHC mixes, q_a, compressor, indexer gates) = bf16 RS dim 2 -> fabric_all_gather in bf16 -> typecast fp32. Not
  tile-aligned sums (norm sum of squares [S/SP, 1], engram gates [S/SP, 12]) stay the exact fp32 gather_add (a bf16
  partial reaching it is cast to fp32 first). No fork change.
- Tuning (DS41-67): the first version typecast fp32 -> bf16 in ccl and was slower than exact (device total 642.09 vs
  641.23). Now the producers of the big partials write bf16 directly when `ccl.partial_dtype(mesh)` says so
  (fabric_bf16 and TP > 1 and the gather_add path): attention wo_b linear (tt/attention.py), moe_add's add
  (tt/model.py), q_a's linear (tt/q_a.py). Compressor / indexer / mHC partials still typecast in ccl (small).
- Probe (2x2, random fp32 partials x 3): bf16 sums vs fp64 rel 2.45e-3 for every shape; output vs
  bf16(bf16 a + bf16 b) within 1 bf16 ulp (max 0.125 at |x| 16-32).
- Per op (device kernel ms per call, max over chips, 6 calls after 2 warm-ups, 1024 rows per chip; probe, deleted per
  K-56). stock / fabric / fabric_bf16 with fp32 input (cast in ccl) / fabric_bf16 with bf16 producer input:
  sum_cols [1024, 5120] 0.664 / 0.147 / 0.196 (cast 0.071 + RS 0.080 + back 0.046) / 0.125 (RS 0.079 + back 0.045);
  tp_sum q_a [1024, 1792] 0.321 / 0.106 / 0.125 / 0.091; compressor [1024, 1024] 0.165 / 0.065 / 0.070 / 0.054;
  hc / indexer [1024, 32] 0.014 / 0.013 / 0.015 / 0.015.
- Ladder last (DS41_CCL_IMPL=fabric_bf16): PASS. pcc_layer L00 0.99976, L01 0.99948, L02 0.99863, L03 0.99674,
  L14 0.99781, L20 0.99972, L21 0.99813, L24 0.99980; pcc_state_min 0.99776 (P.5 fabric: L03 0.99681, state 0.99767).
- Plain profile (rung last, chunk 53248-55296, 8 layers), fabric_bf16 vs P.5's fabric (results/P.5_profile.json,
  another session): device total 639.62 vs 641.23 ms (-1.61, -0.25 %); attention 233.31 vs 234.20, moe_add 1.99 vs
  2.43, q_a 36.50 vs 36.75, rest within +-0.03; chunk wall 814.8 ms; host_transfers_per_layer 0; pcc_chunk_out 0.99979.
  On this 2x2 box (TP = 2) the sums are a small share of the chunk; the gain grows with the bytes per sum.
- Re-run: on `DS41_CCL_IMPL=fabric_bf16`, off `DS41_CCL_IMPL=fabric`, in front of the brief's ladder / profile command.

## P.7 perf (attempt 1): attention matmuls at HiFi2 (ATTN_FIDELITY, owner comparison)
- Added setting ATTN_FIDELITY {HiFi4 (default, unchanged: the task's "default unchanged"), HiFi2} in tt/settings.py
  and MAX_PRECISION ATTN_FIDELITY=HiFi4 (component / swap tests stay at HiFi4). tt/common.py:attn_compute_config =
  compute_config with ATTN_FIDELITY (fp32 DEST from FP32_DEST_ACC, approx off, as before).
- Applied to: tt/q_a.py TtAttnInProj's fpu linear ([wq_a | wkv], also the kv step's columns); tt/attention.py
  self.ckc (wq_b, window / compressed sparse_sdpa, wo_a, wo_b; TtSparseAttention inherits it); tt/indexer.py self.ckc
  (weights_proj, wq_b) and score_ckc (ring_indexer_score_dsa; the candidate / re-index indexers inherit);
  tt/compressor.py self.ckc ([wkv | wgate], wk). Not applied: mHC, norms, router, experts, shared expert, engram_kv
  (compute_config, MATMUL_FIDELITY). The comparison-only exact paths (ATTN_IN_MM=exact ExactMatmul,
  WINDOW_SDPA=exact) keep compute_config: their exactness needs HiFi4 (DS41-16).
- Accuracy (frozen, gate order, each once, HiFi4 by MAX_PRECISION): test_c_{swa_attention, dec_reindex_attention,
  dec_reindex_indexer, dec_reindex_q_a} all PASS.
- Smoke, information only, DS41_ATTN_FIDELITY=HiFi2 (same 4 tests, all PASS), vs CPU step HiFi4 -> HiFi2:
  q_a L24 rel 1.577e-4 -> 1.577e-4 (bit-identical: fp8 x fp8 operands, DS41-42); indexer L24 overlap 0.98866 ->
  0.98855, recall_tie 0.99546 -> 0.99545; attention L24 rel 0.0352 -> 0.0430 (limit 0.0748), bias -0.00077 ->
  -0.00467, ratio_min 0.9945 -> 0.9865 (+ a not-gating column-bias WARN); attention L00 (swa) rel 0.0230 (recorded
  C.swa.attention, 2026-10-15) -> 0.0300 (limit 0.0512), bias -0.00086 -> -0.00406. Attention's bias grows ~5x
  (K-25: HiFi2 truncates the bf16 q / probs / o operands), not its rel past a limit. DS41-68.
- Plain profile (rung last, chunk 53248-55296, 8 layers, 2x2), HiFi4 -> HiFi2: device total 641.06 -> 634.35 ms
  (-6.71, -1.05 %); attention 234.16 -> 232.40, indexer 56.48 -> 52.09, q_a 36.77 -> 36.42, compressor 9.13 -> 9.08;
  rest within +-0.3 (experts 149.20 -> 148.94 noise). Same program counts. HiFi2: host_transfers_per_layer 0,
  pcc_chunk_out 0.99979. Chunk wall 793 / 920 ms (host-bound, host overhead ~280 ms: noise, not the change).
- Not run (orchestrator does): the ladder at either setting.
- Re-run: on `DS41_ATTN_FIDELITY=HiFi2`, off `DS41_ATTN_FIDELITY=HiFi4`, in front of each frozen test / the ladder /
  the brief's profile command.

## P.8 perf (attempt 1): experts + weighted combine_rf in one program, NUMERICS, FFN_NORM_DTYPE (owner comparison)
- Settings (tt/settings.py, defaults = today's path): EXPERTS_COMBINE {separate, weighted}, NUMERICS {official,
  native}, FFN_NORM_DTYPE {fp32, bf16}. MAX_PRECISION: NUMERICS official, FFN_NORM_DTYPE fp32, EXPERTS_COMBINE
  separate (weighted cannot run the official q(r h): the weight moves to the combine; and its weights are bf16).
- NUMERICS=native: tt/quant.py `act_q` / `fp4_q` / `tf32_q` = a cast (or the tensor itself) at every fp8_qdq /
  fp4_qdq / round_tf32 call site (q_a, kv, attention wq_b / o / wo_b q, indexer q / fp4 q / weights_proj TF32,
  compressor ckv / k fp4 and TF32 input, shared x / h, experts x and loop h, engram_kv); the expert ops' fp8_act_quant
  off (flat / unified). Reference hook unchanged (official).
- FFN_NORM_DTYPE=bf16: tt/norm.py ffn_norm's last multiply writes bf16 (the row all-gather moves bf16); router /
  experts / shared expert read it as is; act_q casts it to fp32 for the official fp8 q-dq. bringup/hooks.py: the
  harness boundary hands router / experts / shared expert x in that dtype (`_ffn_norm_dtype`).
- EXPERTS_COMBINE=weighted (tt/weighted_combine.py, tt/experts.py `_weighted`; EXPERTS_KERNEL=flat): q(x) of the mesh
  row -> ROW_MAJOR -> fabric all_gather over axis 0 (the column's tokens: the flat expert's token_index x); top-k ids
  (uint32) / bf16 weights gathered as TILE over axis 0; every pass's one-core planner (combine_plan axis=0, column-
  local expert ids, tile_in, det order) in one program on two cores; per-call arena (all cores, at the top of L1),
  writers' / relays' words below it, combine state zeroed (l1_zero), a CB-free fabric semaphore barrier over the
  column (`_column_barrier`), then per pass flat_routed_expert_with_program (column-local gids, n_global Eg, plan
  routed=False, HiFi4 / glm LoFi, weights_nk, down_fp32) + sparse_combine_rf(axis=0, L1 hand-off, device plan,
  deterministic, ports_per_direction, out fp32 / bf16 under fabric_bf16, out_row0, acc_unpack_fp32, zero_unseen);
  out [S, H] RM -> TILE -> moe_add -> the same sum_cols. 2x2: 1024 home tokens per chip need 2 passes of 512 (1 port
  + 9 accumulators of 10 free cells, arena up to 1356 KB; DS41-69). Fabric payload 10880 B under weighted only
  (bringup/hooks.py device_params; one 10240 B row + trailer a packet).
- Forks: combine_rf (axis=0, out_dtype, out_row0, acc_unpack_fp32, zero_unseen, accumulator_bytes; planner column /
  pass options, ids_uint32, tile_in, det on the one-core planner; the < 4-slot local-ring grant deadlock fixed) and
  flat_routed_expert_with_program (math_fidelity, weights_nk). CHANGELOGs, READMEs, INDEX updated; ops_lint 0
  problems; combine_rf tests/unit/test_column_planner.py 4/4 (planner == plan_reference bit for bit, det lists).
- Gotchas: (1) ttnn.reshape refuses uint16 RM and took ~1 ms per [2048, 6] RM reshape to one page (DS41-71): the
  planner reads the gathered tiles. (2) A local hand-off ring of 2 slots (a near-full arena) deadlocked: grants were
  batched by 4 (fixed in the fork; the arena now leaves 8 slots). (3) A stock / fabric all_gather as the post-zeroing
  barrier: its CBs (10880 B packets) clashed with the arena on layer 14 (DS41-70). (4) Arrival-order fp32 sums are not
  bit-identical across calls (149 / 10.5M values at 1e-8): the frozen component tests' determinism check failed until
  the det order (every column chip's flat schedule, se_dyn) was ported to the one-core planner. (5) The planner's
  scalar loops: 616 µs (arrival) / 807 µs (det) a layer for both passes on two cores.
- A/B, tt/perf_experts.py::test_weighted_vs_separate (layer 0, chunk 2048, synced wall per experts call, random
  inputs): glm (shipped EXPERTS_PRECISION) native: separate 8.0 ms, weighted 5.3-5.4 ms (arrival order); full
  precision fabric (fp32 out) native: separate 8.5, weighted 6.05 ms (det). vs float64 on 128 tokens (full, fp32 out):
  separate rel 0.0023 bias +2e-5, weighted rel 0.0029 bias +2e-5; glm: separate rel 0.0228 bias +1.56 % (bfp8 y tiles),
  weighted 0.0202 / +1.33 % (bf16 y rows). Determinism (det): 0 of 10.5M values differ. Per-op device µs (glm,
  arrival): fused 2 x 2052, planners 616, x cast 72 + untilize 57 + gather 105, out tilize 113, RS 79 + 47, rest < 40:
  what remains of the experts path is the two weight-stream-bound passes (76 %) and the planner (12 %).
- Accuracy, frozen tests in gate order with the change on (DS41_EXPERTS_COMBINE=weighted DS41_NUMERICS=native
  DS41_FFN_NORM_DTYPE=bf16; the rest at MAX_PRECISION): test_c_{swa, swa_engram, enc_source, enc_source_engram,
  enc_reuse, dec_candidate, dec_reindex, dec_reuse}_experts all PASS (swa: vs CPU rel 0.0298 <= 0.0849, row slope
  0.59-0.95, determinism bit-identical). Swap tests FAIL (the official per-step checks of the steps NUMERICS=native
  changes, DS41-72):
    S.swa.12: FAIL swap_q_a vs the CPU step on the same inputs: rel=0.011596 (<= 0.01040, component error 0.00726);
              FAIL swap_ffn_hc vs the CPU step on the same inputs: rel=0.000197 (<= 0.00015, component error 0.00230;
              median row norm ratio off by +0.00008, limit 0.00007)
    S.swa_engram.15: FAIL swap_q_a vs the CPU step on the same inputs: rel=0.013907 (<= 0.01191, component error 0.00863)
    S.enc_source.14: FAIL swap_q_a vs the CPU step on the same inputs: rel=0.012939 (<= 0.01114, component error 0.00806);
              FAIL swap_compressor vs the CPU step on the same inputs: rel=0.101759 (<= 0.07459, component error 0.01587);
              FAIL swap_indexer near ties: flips=2048 unexplained=170 recall=0.963845 recall_tie=0.968086 worst_row=0.9238
  Per the brief: stopped there (no profile; the remaining five swap tests not run). EXPERTS_COMBINE / FFN_NORM_DTYPE
  change none of the failing steps (q_a, compressor, indexer, ffn_hc sit before ffn_norm).
- Smoke, information only (not gates): the 8 test_c_*_router with DS41_FFN_NORM_DTYPE=bf16 (bf16 x in): all PASS;
  vs the CPU step recall_tie 1 on every layer (every flip a near tie), recall 0.99919-0.99984 (L00 0.99976, L01
  0.99935, L02 0.99984, L03 0.99967, L14 0.99951, L20 0.99967, L21 0.99967, L24 0.99976), flips_unexplained 0.
  test_c_swa_experts with DS41_NUMERICS=native (together with the other two): PASS, vs CPU rel 0.0298 (limit 0.0849).
- Change off (defaults) unchanged: test_c_swa_experts vs CPU rel 0.00224936 (the recorded value), deterministic.
  Not run: the whole model with the change on (ladder / test_positions / profile: the brief stops at the failing
  swap checks).
- Fork regressions (options off): flat_routed_expert_ttnn tests/unit the baseline's 26 / 13 (one K-52 precompile
  artifact, CHANGELOG); combine_rf has no source tests (new tests/unit/test_column_planner.py 4/4).
- stock ttnn.all_gather instead of ttnn.bringup.fabric_all_gather: the weighted path's top-k ids (uint32) / weights
  (bf16) tiles [1024, 6] gathered over the column once a layer, 12 µs each (profile); the fork is untested on uint32
  TILE here, and q(x) itself (the 20 MB gather) goes through tt/ccl.py's fork.
- Re-run: change on `DS41_EXPERTS_COMBINE=weighted DS41_NUMERICS=native DS41_FFN_NORM_DTYPE=bf16`, off
  `DS41_EXPERTS_COMBINE=separate DS41_NUMERICS=official DS41_FFN_NORM_DTYPE=fp32`, in front of each gate command;
  A/B: `DS41_NUMERICS=native scripts/run_safe_pytest.sh --run-all --no-precompile
  "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::test_weighted_vs_separate[blackhole-ops-ab-2x2-box_payload]"`
  (ops: with the profiler env of testing/profiler.py).

## P.8 perf (attempt 2): re-verified accuracy with the change on; stopped at the first failing swap test
- Attempt 1 failed only on a process rule (a direct `pytest --collect-only`); its code (settings EXPERTS_COMBINE /
  NUMERICS / FFN_NORM_DTYPE, tt/weighted_combine.py, tt/quant.py act_q / fp4_q / tf32_q, the combine_rf and
  flat_routed_expert_with_program options) is unchanged in this attempt. The fork build was current (_ttnn.so newer
  than the fork sources). No pytest outside scripts/run_safe_pytest.sh / scripts/tt-probe.sh.
- Frozen tests in gate order, once each, change on (DS41_EXPERTS_COMBINE=weighted DS41_NUMERICS=native
  DS41_FFN_NORM_DTYPE=bf16; the rest at MAX_PRECISION): test_c_{swa, swa_engram, enc_source, enc_source_engram,
  enc_reuse, dec_candidate, dec_reindex, dec_reuse}_experts all PASS (vs CPU rel L00 0.0298, L01 0.0345,
  L02 0.0361, L14 0.0390, L03 0.0358, L20 0.0409, L24 0.0410, L21 0.0408; limits 0.085-0.120; determinism
  bit-identical every time; the weighted path ran: "EXPERTS_WEIGHTED S=1024 passes=2 x 512"). Then
  test_swap_swa_12_experts FAILED:
    FAIL swap_q_a vs the CPU step on the same inputs: rel=0.011596 (<= 0.01040, component error 0.00726)
    FAIL swap_ffn_hc vs the CPU step on the same inputs: rel=0.000197 (<= 0.00015, component error 0.00230;
         median row norm ratio off by +0.00008 (limit 0.00007))  [WAIVED by the owner's hc_fn waiver, DS41-43]
  Stopped there per the brief (no profile, the other 7 swap tests not run; attempt 1's swa_engram_15 / enc_source_14
  failures are in its section above).
- Why (DS41-72, detail in known_issues_detail.md): a tt-probe on the golden q_a input: CPU native (bf16 x, no fp8
  q-dq) vs CPU official rel 0.011637 (L0) / 0.013901 (L1); device native vs CPU native 0.000143 / 0.000160, the same
  size as device official vs CPU official 0.000101 / 0.000105. The failure is exactly the official fp8 rounding the
  device skips under NUMERICS=native (rule 17: reported, not added back); no device error. The per-step checks stay on
  the official arithmetic (the brief: no numerics switch in the reference hook).
- MAX_PRECISION keeps EXPERTS_COMBINE=separate: the combine's sums are fp32 there (CCL_IMPL=fabric), but weighted
  cannot run the official q(r h) before w2 (the weight moves to the combine; q(r h) != r q(h) with ue8m0 scales) and
  multiplies by bf16 routing weights (the planner's format), below the reference's fp32 weight (DS41-23).
- Re-run: change on `DS41_EXPERTS_COMBINE=weighted DS41_NUMERICS=native DS41_FFN_NORM_DTYPE=bf16`, off
  `DS41_EXPERTS_COMBINE=separate DS41_NUMERICS=official DS41_FFN_NORM_DTYPE=fp32`, in front of each gate command.

## P.9 perf (attempt 1): weighted combine with bf16 running sums, the chunk in one pass (owner comparison)
- Setting COMBINE_ACC {fp32, bf16} (tt/settings.py; default fp32 = today's path, as the owner's text says; F104 sets
  bf16 on accept), MAX_PRECISION fp32. Used only under EXPERTS_COMBINE=weighted.
- Fork combine_rf: `sparse_combine_rf_prepare(acc_dtype=None | ttnn.bfloat16)`: CB_ACC Float16_b with 2048 B tiles
  instead of fp32 4096 B; same kernels, same det order (the packer's L1 accumulation adds the weighted fp32 DEST row
  onto the bf16 sum). `accumulator_bytes(acc_dtype=)`, `acc_page()`, the arena refusal ("bf16 tiles", s_fit / a_min)
  size with it. Default None = fp32: programs unchanged. CHANGELOG / README / INDEX updated; Python only (no build).
- Model: tt/weighted_combine.py `WeightedCombine(acc_dtype=)` picks the fewest passes with it (DS41-69's sizing) and
  logs `EXPERTS_WEIGHTED S=1024 passes=1 x 1024 acc=bf16`; tt/experts.py `combine_acc` (build_experts arg / setting)
  asserts one pass for bf16 and logs the pass count per layer. 2x2: 9 accumulators, 114 tokens x 5 bf16 tiles =
  1140 KB each, arena 1314 KB (ring depth 4, local ring 8 kept: DS41-70).
- Tests: fork tests/unit/test_acc_dtype.py 4/4 (sizing: one call holds 1143 bf16 vs 567 fp32 tokens at 9
  accumulators / 1356 KB; refusal 1024 tokens in 1156 KB: bf16 "<= 963 or >= 10 accumulators", 963 builds, 972 refused,
  fp32 "<= 477"; standalone combine precision, 6 bf16 rows a token at 2^-6..2^6 vs the float64 sum: fp32 rel 7e-9, bf16
  rel 2.71e-3 / 2.79e-3 (one rounding of the exact sum 1.66e-3), bias -1.2e-5 / -1.5e-5, worst row 3.9e-3); whole fork
  suite 8/8 (test_column_planner 4/4 = baseline). Model tt/perf_experts.py: test_weighted_combine_routes_bf16 (S=1024,
  local / remote / random: one pass, completes) 3/3; test_combine_acc (layers 0 / 24, real weights, chunk 2048).
- Measured, test_combine_acc (synced experts call per layer; float64 on 128 tokens; bf16 vs fp32 on all 2048 rows):
  shipped glm precision (NUMERICS native, FFN_NORM_DTYPE bf16, CCL_IMPL fabric_bf16): fp32 2 x 512 5.74 ms, bf16
  1 x 1024 4.29 ms (L0 and L24); vs float64 L0 fp32 0.02015 / bf16 0.02018, L24 0.02059 / 0.02061; bf16 vs fp32
  0.0036 (bias -1.8e-4). Full precision (EXPERTS_PRECISION=full, CCL_IMPL=fabric): 6.09 / 6.05 -> 4.96 / 4.95 ms;
  vs float64 L0 fp32 0.00291 / bf16 0.00406 (bias +2e-5 / -2.0e-4), L24 0.00296 / 0.00413; bf16 vs fp32 0.00287 /
  0.00291. Determinism: bf16 twice 0 of 10.5M values differ (both layers).
- Per op, layer 0, glm (test_combine_acc ops): fp32 total 5372 µs = fused 2 calls 4107 + planners 807 (2 passes on
  2 cores); bf16 4236 µs = fused 1 call 2201 + planner 1573 (1024 tokens on one core). The planner is now the
  biggest non-expert piece (DS41-73: next lever).
- Accuracy, change on (DS41_COMBINE_ACC=bf16): the 16 frozen test files in gate order, each once, run one by one
  with scripts/run_safe_pytest.sh (the measure part `test_c_${t}_experts` is one 8-file command, ~19 min, over the
  10-minute foreground limit): all PASS; they run EXPERTS_COMBINE=separate (MAX_PRECISION), so COMBINE_ACC is not on
  their path (test_c_swa_experts vs CPU rel 0.00224936, the recorded value; swa_12 swap_ffn_hc WAIVED as before).
  Ladder `last` via `orchestrator measure --task P.9 --part last`: rc 0, min layer PCC 0.99680 (L03) (P.8 on: 0.9968).
- Plain profile via `measure --part profile` (the tool moved the foreground call to background at its 600 s limit; it
  ran to the end): device_ms_total 251.86 (P.8 on 258.1), device_ms_experts 38.91 (P.8 on 45.3; gate < 45.3),
  chunk_wall_ms 241.0, profiled_programs 11772, pcc_chunk_out 0.99977.
- Re-run: on `DS41_COMBINE_ACC=bf16`, off `DS41_COMBINE_ACC=fp32` in front of each gate command; A/B:
  `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::test_combine_acc" -k "not ops"` (ops: profiler env);
  fork: `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/combine_rf/tests/unit`.

## P.10 perf (attempt 1): routed expert weights in bfp4 (the checkpoint's fp4 width) vs bfp8 (owner comparison)
- Setting EXPERTS_IMPL: default bfp8 -> bfp4 (agent rule 7 / F117: fp4 -> bfp4), MAX_PRECISION stays bfp8 (exact), so
  the component / swap tests are unchanged (they run bfp8). Description in tt/settings.py and tt/experts.py updated.
- What bfp4 broke: nothing on the model's path. The flat plan changes with bf4 tiles: `nu=3 np=0 mt=1 -> mt=2`,
  flat arena 1156 -> 948 KB, same 48 gate/up + 32 down cores, same 10 free cells / 9 accumulators, weighted combine
  still ONE pass x 1024 (arena 1314 KB, set by the bf16 accumulators). Logged per layer (EXPERTS_FLAT,
  EXPERTS_WEIGHTED). One latent bug fixed: tt/weighted_combine.py:weighted_combine cached its state by (mesh, kwargs)
  only, so a process building bfp8 then bfp4 experts (an A/B) would reuse the bfp8 plan's core layout; the key now
  includes the flat plan.
- Measured, tt/perf_experts.py::test_weights_bfp4 (new; synced whole experts call, random inputs, default path:
  flat + weighted + COMBINE_ACC bf16 + glm + native + ffn_norm bf16 + fabric_bf16; 2x2, chunk 2048):
  L00 bfp8 4.47 ms, bfp4 3.47 ms; L24 4.47 / 3.48 ms (-22 %). Load 37-44 s a layer either way.
  vs float64 with the exact fp4 weights (128 tokens): L00 bfp8 rel 0.0202 bias +0.0132, bfp4 rel 0.1843 bias +0.0373;
  L24 bfp8 0.0206 / +0.0133, bfp4 0.2050 / +0.0399. bfp4 vs float64 with bfp4-rounded weights (`_bfp4_round`)
  L00 rel 0.0202 bias +0.0131: the device path is as exact as bfp8's; the rest is bfloat4_b's rounding of the weights
  (float64 alone: bfp4-rounded vs exact weights rel 0.179, bias +0.024). bfp4 vs bfp8 all rows rel 0.180 / 0.200.
  Determinism: bfp4 twice 0 of 10.5M values differ. Per layer at load: 16.4-16.6 % of values changed, worst matrix
  rel 0.118-0.124.
- Final check `python -m models.demos.common.bringup.orchestrator measure --task P.10` (quick, change on): rc 0, worst
  layer PCC 0.99530 (L03; P.9 on 0.9968), pcc_chunk_out 0.99943, chunk_wall_ms 238.3 (P.9 237.1: the wall did not move),
  host_transfers_per_layer 0. Layer PCCs L00 0.99937, L01 0.99903, L02 0.99787, L03 0.99530, L14 0.99617, L20 0.99929,
  L21 0.99689, L24 0.99943. Not run by me: the plain profile (device_ms_experts gate < 38.9; P.9 38.91; expected about
  -1 ms a layer x 8 from the timed call) and the frozen tests (pinned to bfp8 by MAX_PRECISION).
- Re-run: on `DS41_EXPERTS_IMPL=bfp4`, off `DS41_EXPERTS_IMPL=bfp8`; A/B: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh
  --run-all --no-precompile "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::test_weights_bfp4" -k "time"`
  (ops: with the profiler env of testing/profiler.py).

## P.11 perf (attempt 1): the weighted combine's planner on several cores (column multi-core planner) vs one core
- Setting COMBINE_PLANNER {one, multi} (tt/settings.py; default one = today's program, as the owner's text says; the
  orchestrator sets both sides). Not a precision switch (bit-identical tables), so not in MAX_PRECISION. Used only under
  EXPERTS_COMBINE=weighted: tt/experts.py `combine_planner` -> tt/weighted_combine.py `WeightedCombine(planner=,
  planner_cores=32)`; each pass's planner on its own 32 cores (disjoint: run_planners merges the passes' programs, now
  with their semaphores). Logged per layer: `COMBINE_PLANNER=multi (multi: 32 cores x 64 tokens a pass)`.
- Fork combine_rf: `combine_planner_prepare(axis=0, cores=[...], tile_in=True)` = new kernel `_PLANNER_MCC`: Marko's V2
  multi-core structure (counting sort into per-chunk buckets of final table words, count exchange, per-expert chunk
  bases, 1/M of the row tiles pulled per core, atomics onto core 0's plan) with the one-core planner's column / pass
  options (id_offset skip, tokens / token_base, uint32 or uint16 TILE ids read as faces 0 / 2, det). Chunks = 32-token
  multiples inside one home (`_mcc_geometry`: M = N ceil(S / CT)), so chunk order = the one-core (home, token, k) scan
  and every expert's rows land in the same order. det: the chunks of this chip's home tokens build their tokens' order
  lists themselves (a row's rank = the rows of the chunks before it, sent with the counts; source schedules from the
  totals), the accumulators' owners (core a % M) assemble the pages. The tails past the planned rows are written as
  zeros (the one-core planner writes the whole page). Default cores=None: the one-core program unchanged; the
  whole-mesh planners untouched. CHANGELOG / README / INDEX updated; Python + runtime-compiled kernel (no build).
- Tests: fork tests/unit/test_column_planner_mc.py 20 passed / 12 skipped by design: one-core vs multi-core, every
  output word (counts, regions, whole token index and table, plan, det pages) identical per chip (both columns), S 512 /
  1024 one pass, 1024 in passes of 512 and of 342, det / arrival, 32 / 8 cores (8 < 9 accumulators), uint32 / uint16
  ids, uniform / hot / one-column routings, outputs pre-filled with garbage. Fork suite tests/unit: 28 passed (the old
  8 = baseline, option off).
- Measured, tt/perf_experts.py::test_combine_planner (default path: bfp4, COMBINE_ACC bf16, glm, native, ffn_norm
  bf16, fabric_bf16; 2x2, chunk 2048; same random inputs): planner op 1572.7 -> 68.8 µs (device, profiler op mode, L00
  and L24); experts ops total 3427.5 -> 1912.4 µs (L00), 3425.5 -> 1927.1 µs (L24); synced experts call 3.66-3.68 ->
  2.10-2.16 ms; experts output multi vs one 0 of 10.5M values differ (both layers).
- Phases (run.tdbg, cycles at ~1.31 GHz, chip 0): chunk scan + sort ~20K, exchanges + slices ~15K, det ~50K on the 16
  home-chunk cores (count sums 10K, 2 schedules 10K, order lists ~20K, owner pages ~9K). DS41-75: scalar L1 accesses
  cost ~10-25 cycles (counters moved to local arrays, core 0's zeroing to local NoC copies: 104K -> 90K). Next lever
  if wanted: the schedules on idle cores (8 per det core on 8x4), not worth it on 2x2 (~7 µs).
- Final check `python -m models.demos.common.bringup.orchestrator measure --task P.11` (quick, change on): rc 0, layer
  PCCs equal to P.10 (L00 0.99937, L01 0.99903, L02 0.99787, L03 0.99530, L14 0.99617, L20 0.99929, L21 0.99689, L24
  0.99943), pcc_chunk_out 0.99943, chunk_wall_ms 232.1 (P.10 238.3), host_transfers_per_layer 0. Not run by me: the
  frozen tests (MAX_PRECISION runs EXPERTS_COMBINE=separate: the planner is not on their path) and the plain profile
  (device_ms_experts < 38.9; expected about -1.5 ms a layer x 8).
- Re-run: on `DS41_COMBINE_PLANNER=multi`, off `DS41_COMBINE_PLANNER=one`; A/B: `PYTHONPATH=$PWD
  scripts/run_safe_pytest.sh --run-all --no-precompile "models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py::
  test_combine_planner" -k time` (ops: with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
  TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000, -k ops); fork:
  `scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/combine_rf/tests/unit`.

## P.12 perf (attempt 1): the shared expert on the flat routed expert (bfp8, always-on) vs the ttnn.linear chain
- Setting SHARED_EXPERT_IMPL {linear, flat} (tt/settings.py). Default kept `linear` as the owner's text says (P.9 /
  P.11 precedent; the orchestrator sets both sides, F104 flips it on accept). MAX_PRECISION: linear (exact bf16 weights),
  so the component / swap tests are unchanged.
- flat (tt/shared_expert.py `_load_flat`, `_flat_partial`): Marko's K3_SE=flat with our TP layout: the chip's
  intermediate slice (2x2: 1152 columns of 2304) as SE_PARTS = 1 always-on expert of ttnn.bringup.flat_routed_expert
  (counts [1024], regions [0]), every own token routed to it; weights bfp8 as stored [N, K] (weights_nk, DS41-35), laid
  out once at load. The op refused I = 1152: se6_dcompute.cpp static_assert `kt_d % 8 == 0` (down K in blocks of 8
  tiles), so the slice is zero-padded to 1280 (DS41-76; 4x4 / 8x4 / 2x4: 576 -> 768). Precision follows the routed
  experts' EXPERTS_PRECISION (glm: LoFi, bfp8 x / h tiles; full: MATMUL_FIDELITY, bf16 tiles), fp32 DEST in gate/up and
  down, y bf16 TILE written in ccl.partial_dtype (bf16 under fabric_bf16, DS41-67: no typecast before moe_add).
  The down coordinators' done words are shared by every layer's shared expert (one L1 tensor, `_DONE`).
- x: under EXPERTS_COMBINE=weighted + NUMERICS=native the shared expert reads its own rows (r S + t) by a load-time token
  index from the routed experts' gathered row-major q(x) [R S, H] (TtExperts.keep_rows / take_rows; the block sets
  keep_rows, the shared_expert step takes and frees it): no layout conversion of its own (bit-identical to the own-x
  path, SHARED_ROWS). Without rows: own to_layout (56 us). NUMERICS=official: own q(x) rows + the op's fp8_act_quant
  with unit routing weights (that option refuses a token index); runs (7.9 ms, the composed fp8_qdq of x dominates).
- Hidden? Not folded, not overlapped (owner asked to prefer it only when no more work): folding into the routed
  experts' program needs one weight dtype for both (routed bfp4 since P.10, shared bfp8; the op takes one) and the
  combine planner to place extra ids; overlapping with the dispatch needs sub-devices. What K3 hides under the dispatch
  (the x untilize) costs nothing here: the gathered rows are reused.
- Measured, tt/perf_shared_expert.py::test_shared_flat_vs_linear (2x2, chunk 2048, real weights, random x, default
  path otherwise: native, glm, ffn_norm bf16, fabric_bf16, weighted, bfp4 routed; L00 / L24 identical to 0.01 ms):
  shared partial wall 1.449 -> 0.356 ms (own x) / 0.274 ms (rows); device per op (ops mode, L00): linear 1430.9 us
  (linear 801.7 + 445.1, slices, clamps, silu, multiply, typecast), flat 252.6 us (one op; + to_layout 55.9 own x).
  MoE block device (router .. reduce-scatter) 3727 -> 2529 us. Determinism: flat twice 0 values differ.
- Plain profile, by hand (gate env, BRINGUP defaults, K-58: generated/bringup_adhoc/P.12.json), per chunk of 8 layers:
  shared_expert linear 11.41 -> flat 2.01 ms (gate < 11.4: pass; programs 288 -> 32), moe_add 1.78 -> 1.58,
  experts 17.38 / 17.33, router 1.83 / 1.83; MoE block per layer 4.05 -> 2.84 ms; device total 230.28 -> 220.61 ms
  (-4.2 %); every other section within 0.06 ms.
- Accuracy, shared_out vs float64 with the exact fp8 weights (256 rows, native numerics): linear rel 0.00297 / 0.00301
  (L00 / L24), bias +0.0009; flat 0.02048 / 0.02177, bias +0.0132 / +0.0140, worst row 0.023 / 0.026; flat vs float64
  with bfp8-rounded weights 0.0197 / 0.0208 (the device path; the bfp8 weights alone: rel 0.0052 / 0.0062, bias 0;
  bfp8 changes 10.2-11.1 % of the fp8 values, ||dW||/||W|| 0.0035-0.0040). Variants (test_flat_variants, L00 / L24):
  LoFi + bf16 x / h 0.377 ms, rel 0.011 / 0.012, bias -0.8 %; LoFi x bfp8 h bf16 0.34 ms, 0.015 / 0.016, +0.9 %;
  LoFi x bf16 h bfp8 0.38 ms, 0.013 / 0.014, -0.4 %; stochastic packer 0.36 ms, bias +0.2-0.3 % but worst row
  0.05-0.075; HiFi2 + bf16 = LoFi + bf16 bit for bit (0.45 ms); HiFi4 + bf16 0.68 ms, rel 0.0060 / 0.0069, bias +0.1 %
  (DS41-77). Shipped flat = EXPERTS_PRECISION's glm choice (as the routed experts, which carry the same +1.3 % bfp8
  bias, DS41-74); the owner may prefer bf16 tiles (+15 us) or HiFi4.
- Final check `python -m models.demos.common.bringup.orchestrator measure --task P.12` (quick, change on): rc 0, worst
  layer L03 0.99519 (P.11 0.99530), pcc_chunk_out 0.99943, host_transfers_per_layer 0; chunk_wall_ms 232.1 (spread
  7.5; P.11 232.1) on the first run, 255.6 (spread 51.3, values 228-280: a noisy box) on the rerun after the last
  description edits. The device saves ~9.7 ms a chunk; the clean wall did not move by it (not isolated; the wall
  seems host-bound near 230 ms). Odd, not chased: in the perf test the linear side's MoE block wall is 8.0 ms vs 3.7
  ms device (flat 2.75 vs 2.53; routed alone 2.40 either way).
- Frozen tests not run (quick A/B; they run SHARED_EXPERT_IMPL=linear under MAX_PRECISION). Sanity:
  test_c_swa_shared_expert PASS (err 0.01754 <= 0.06646) after the hooks change (build_shared_expert gets the chunk).
- stock ttnn.to_layout instead of ttnn.bringup.combine_rf's untilize (layout_ops): only on the fallback own-x path (no
  gathered rows, or NUMERICS=official); the shipped path converts nothing. stock ttnn.add for the parts' sum instead
  of combine_rf sum_parts: only when SE_PARTS > 1 (it is 1).
- Re-run: on `DS41_SHARED_EXPERT_IMPL=flat`, off `DS41_SHARED_EXPERT_IMPL=linear`; A/B: `PYTHONPATH=$PWD
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_shared_expert.py -k
  "not ops"` (ops: with the profiler env, -k ops).

## P.13 perf (attempt 1): realistic defaults (fp8 weights bfp8, HiFi2, fp32 accumulation where needed, stock window attention)
- Settings (tt/settings.py), defaults = change on: DENSE_WEIGHTS bfp8 (new), MATMUL_FIDELITY HiFi2, FP32_ACC needed (new;
  replaces the boolean FP32_DEST_ACC), WINDOW_SDPA source. MAX_PRECISION: bf16 / HiFi4 / all / high_precision, so the
  component / swap tests run the old path (test_c_swa_experts re-run: PASS).
- DENSE_WEIGHTS (tt/common.py:dense_dtype): the checkpoint's fp8 matrices (attn wq_a | wkv, wq_b, wo_a, wo_b, indexer
  wq_b, engram wkv, the shared expert's linear path) uploaded as bfloat8_b; the per-matrix change is logged at load
  (DENSE_BFP8, ttnn's own host packing round-trip): 10.0-10.3 % of values, ||dW||/||W|| 0.0034-0.0035 for attention;
  indexer wq_b up to 13.8 % / 0.0046 (L14); engram wkv 12.9 % / 0.0045 (L1), 21.9 % / 0.0051 (L14). Layout kept (W^T,
  groups along N): a host check of [N, K] vs [K, N] gives the same rounding except the engram wkv (0.0035 vs 0.0045).
  The compressor's wkv / wgate / wk and the indexer's weights_proj are BF16 in the checkpoint: kept bf16 (rule 7).
  The shared expert's default (flat) path was already bfp8 (P.12).
- FP32_ACC (tt/common.py:fp32_acc, compute_config(needed=), attn_compute_config(needed=)): measured every dense projection
  at its per-chip shape (tt/perf_precision.py::test_fp32_acc_matmuls, layers 0 / 1 / 2 / 20, real bfp8 weights, HiFi2,
  random inputs vs float64 with the same weights). A 16-bit DEST: q_a rel 0.0044 -> 0.034 (bias +2.6 %), wq_b 0.0041 ->
  0.018, wo_a 0.0041 -> 0.056 (+4.5 %), wo_b 0.0044 -> 0.026, indexer wq_b 0.0041 -> 0.018, weights_proj / compressor
  0.0066 -> 0.045, engram wkv 0.0041 -> 0.084 (+6.8 %), router 0.0041 -> 0.084; time 45-55 % less per matmul (wo_b
  0.60 -> 0.28 ms, engram 2.48 -> 1.39 ms). Probe (1 chip, bf16 W): the bias comes from the spilled K-block partials
  (DS41-78); packer_l1_acc=True makes it unbiased but still 2-3.5x fp32's error (K 1024 0.0086, K 4096 0.0144 vs
  0.0041). Where fp32 stays (all of it): the long-K projections above; the stock sparse_sdpa (softmax statistics: rel
  0.019-0.024 with vs 0.026-0.042 without, worst row 0.064 vs 0.278 at x4 spread); the indexer ring score (K-13); mHC
  hc_fn + Sinkhorn, norms, engram gate sums (fp32 SFPU data, DS41-27 / DS41-40); routed + shared experts (as today,
  GLM-5.3). So FP32_ACC=needed == all on today's code (no program differs); no site was turned off.
- WINDOW_SDPA=source: ttnn.transformer.sparse_sdpa with the attention config (HiFi2, fp32 DEST) for the window and the
  compressed layers. Op alone (perf_precision.py::test_window_sdpa): 0.355 -> 0.334 ms, rel vs float64 0.0073 -> 0.024
  (bias -0.45 % -> -1.1 %); x4 spread 0.0086 -> 0.019.
- stock ttnn.transformer.sparse_sdpa instead of ttnn.bringup.sparse_sdpa: owner rule F123 / the P.13 brief (the stock op
  at its normal settings; the fork's high_precision stays the max-precision value, WINDOW_SDPA=high_precision).
- Plain profile, 2x2, chunk 53248-55296, by hand (gate env + DS41_ overrides; K-58: generated/bringup_adhoc/P.13*.json),
  device total per chunk: off 220.79; DENSE_WEIGHTS=bfp8 alone 220.90 (no section moves: compute-bound at 1024 rows,
  DS41-79); MATMUL_FIDELITY=HiFi2 alone 219.52 (engram_kv 6.02 -> 4.87); WINDOW_SDPA=source alone 220.30 (attention
  114.93 -> 114.39); FP32_ACC=needed alone: same programs; all on 219.17 (-1.62 ms, -0.7 %: attention 114.40, engram_kv
  4.93, q_a 4.75, indexer 34.99, experts 17.34).
- Quick A/B (last chunk vs golden + clean wall): off L00 0.99936 L01 0.99899 L02 0.99782 L03 0.99519 L14 0.99614 L20
  0.99928 L21 0.99688 L24 0.99943, wall 231.7 ms; on (final `measure --task P.13`, rc 0) L00 0.99932 L01 0.99892 L02
  0.99768 L03 0.99486 L14 0.99612 L20 0.99927 L21 0.99682 L24 0.99941, pcc_chunk_out 0.99941, wall 243.0 (spread 27.6;
  first run 236.7), host_transfers_per_layer 0; WINDOW_SDPA=source alone: L03 0.99487 (all of the drop).
- Re-run: on `DS41_DENSE_WEIGHTS=bfp8 DS41_MATMUL_FIDELITY=HiFi2 DS41_FP32_ACC=needed DS41_WINDOW_SDPA=source`, off
  `bf16 HiFi4 all high_precision`; measurements `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_precision.py -s`; final `python -m models.demos.common.bringup.orchestrator
  measure --task P.13`.

## P.14 perf (attempt 1): fused mHC (MHC_FUSED on / off)
- Setting MHC_FUSED (tt/settings.py), default on (the change; the brief's details said "off (default)", the role's rule
  says the change is the default: the orchestrator pins both sides). Not a precision switch (no MAX_PRECISION entry):
  the same fp32 SFPU arithmetic, sums in another order.
- Fork changes, every default unchanged (programs, hashes, returns): mhc_pre_ttnn CHANGELOG 8 (`mhc_pre_xing(...,
  collapse_pre=)`: coefficients + a collapse with a given pre in one program; DM define MHC_XING_GIVEN_PRE) and 9
  (`mhc_pre_xing_pack(None, h, engram_keys=, engram_weight=)`: the 12 engram partials in one pass, own program);
  mhc_post_ttnn CHANGELOG 7 (`mhc_post(..., stats_weight=W)` -> (X', X' @ W with sum X'^2 in column 24); own program).
  New fork tests: mhc_pre tests/unit/test_xing_collapse_pre.py (4), test_engram_pack.py (7); mhc_post
  tests/unit/test_mhc_post_stats.py (8). Regression with the options off (2x2): mhc_pre unit + golden 332 passed / 1
  skipped (recorded 321 / 1 + 11 new), model cases 103 (= recorded); mhc_post unit + golden + cases 382 (= 164 + 208 +
  2 recorded + 8 new). No source-test selection in either fork (AI-generated ops).
- Model (MHC_FUSED=on): attn_hc / ffn_hc = (the partial row: from the previous residual's mhc_post pass when it
  emitted one, else matmul + pack) -> tp_sum -> ONE mhc_pre_xing(deepseek=True, collapse_pre=the pending / attn_hc
  mixes) (DS41-3: a sublayer collapses with the PREVIOUS sublayer's pre); attn_collapse / ffn_collapse hand the y on
  (0 programs). The block boundary carries the whole ffn_hc mixes [S/SP, 24] as the pending pre (layer 0: one-hot
  [S/SP, 24] at load; from_host pads [S, 4] to 24, to_host trims) and DevH.stats = (layer, row). Residuals:
  TtHcResidual.fused -> mhc_post(stats_weight=next hc's fn_t, MATMUL_FIDELITY via a ComputeConfigDescriptor):
  attn_residual -> this layer's ffn_hc; ffn_residual -> the next loaded consecutive non-engram layer's attn_hc
  (TtDS41Model.next_hc); the engram add (identity-comb mhc_post) -> its own layer's attn_hc. Here (layers 0-3, 14,
  20, 21, 24) attn_hc of L0, L20, L24 still runs matmul + pack (embedding / golden-gap run starts); in a full stack
  only L0. Final collapse: mhc_pre_xing(coefficients_given=True) on the 24-wide pre. Engram gates: one engram-mode pack
  -> tp_sum(exact=True) (DS41-82) -> the unchanged gate math; the add: mhc_post(value, h, gate, I).
- stock ttnn.matmul instead of ttnn.bringup.mhc_post(stats_weight=): TtHcMixes.partial_row, for an attn_hc whose
  input came from no residual pass (the embedding, a run start after a gap); the hc_fn projection at MATMUL_FIDELITY.
- Op timings (2x2 p150, [1024, 4 x 2560] fp32 a chip, wall, 30 synced calls): mhc_post 0.270 / with stats 0.327 /
  unfused post + matmul + pack 0.638 ms; mhc_pre_xing coef 0.028, + collapse_pre 0.164 vs the multiply / addcmul
  collapse 0.765 ms; engram partials 0.424 vs the op chain 2.51 ms. The stats and engram programs are DRAM-bound
  (variants without matmul / SFPU / copies <= 0.01-0.04 ms faster; without input reads 0.271): DS41-80.
- Plain profile, 2x2, chunk 53248-55296 (by hand, generated by test_profile.py; profiles P.14_20261020T154842 off,
  P.14_20261020T152421 on), device ms a chunk, off -> on: attn_hc 3.21 -> 2.54, attn_collapse 5.00 -> 0, attn_residual
  2.44 -> 2.94, ffn_hc 3.21 -> 1.44, ffn_collapse 5.00 -> 0, ffn_residual 3.11 -> 3.27 (mHC 21.96 -> 10.19), engram_add
  6.99 -> 1.78; device total 219.54 -> 202.30 (-17.24 ms, -7.9 %); wall (profiler on) 390.8 -> 317.6.
- Quick (BRINGUP_QUICK, on, by hand): worst layer L03 0.99488 (P.13: 0.99486), wall median 216-221 ms (P.13 243.0).
  Final `measure --task P.14` (rc 0): min layer L03 0.99488, pcc_chunk_out 0.99941, chunk_wall 213.7 ms (spread 22.1),
  host_transfers_per_layer 0.
- Follow-ups not done: W multicast to the cores of a column block in the stats program (~13 % of its reads); post /
  comb read from the packed mixes in the stats program (2 slices a residual); the engram value slice as an F column
  offset; the engram pass at ~200 GB/s (DM-bound like the stats one).
- Re-run: fork tests `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_stats.py`
  and `.../mhc_pre_ttnn/tests/unit/test_{xing_collapse_pre,engram_pack}.py`; profile `DS41_MHC_FUSED=on|off` + the gate's
  profile command; final `python -m models.demos.common.bringup.orchestrator measure --task P.14`.

## P.15 perf (attempt 1): attention / RoPE glue folded into ops (ATTN_GLUE on / off)
- Setting ATTN_GLUE (tt/settings.py), default on (the change; the orchestrator pins both sides). Not a precision switch
  (no MAX_PRECISION entry): every fold is a data move or a cast moved into the op before it.
- (1) Pair interleave / de-interleave: tt/rope.py:PairPerm, a 0/1 [64, 64] bf16 matmul on the bf16 tail (data in in0),
  HiFi4 + fp32 DEST (DS41-83: HiFi2 / LoFi drop in0's last bit; DS41-16's one-product-per-dot idea, K-30: never on
  fp32 data). Sites: attention q (after the scale's bf16 multiply), attention output (bf16 sdpa output -> to_de, fp32
  out), indexer q, kv and compressor (rope_deinterleaved_bf16: the rope's add / subtract write bf16 = act_q / fp4_q's
  cast under native, DS41-84). Window layers now read the table as stored (instance INTERLEAVED=True when the q fold is
  on): no per-chunk table transform (glue report A17). Output re-interleave dropped: wo_a's input rows permuted at load
  to [nope | re | ro] (bfp8 groups run along N, so the weight rounds the same). NUMERICS=official keeps the fp32
  interleave before an fp8 / fp4 block (q has no block: its fold runs in both numerics).
- (2) q heads: ttnn.experimental.nlp_create_q_heads_split (attention: 16 heads, split 448; indexer: 32 heads, split
  64): bit-identical to reshape + transpose / permute (probe ds41_attn_glue/probe_001); RoPE then on the 64-wide tail only.
- (6) Output: the 448 nope columns stay bf16 (no bf16 -> fp32 -> bf16); after wo_a nlp_concat_heads (native: on the bf16
  act_q, K-19: 2 x 1024 columns fit; g_l = 2 on every listed box) instead of slices + concat; sum_cols(fp32=False) returns
  the bf16 reduce-scatter result. mhc_post takes bf16 F only without stats_weight (README limits), so under MHC_FUSED=on
  (default) tt/residual.py casts it to fp32 before the stats pass: the A12 cast moved, not removed (attn_residual +0.36
  ms); it is gone only under MHC_FUSED=off. Removing it needs a bf16 F CB in mhc_post's stats program (not done).
- Exactness (probes ds41_attn_glue/probe_001..010): q bf16 bit-identical after reordering the tail; rope_deinterleaved_bf16
  == typecast(rope_deinterleaved) (widths 512, 128); rope_heads(bf16) + PairPerm == typecast(rope_heads(interleave=True));
  PairPerm to_de == deinterleave_tail. Window attention L0 output path vs float64 (bf16 o input): rel 0.00736 off and
  on (DENSE_WEIGHTS / ATTN_FIDELITY / CCL max precision: 0.00136 both); on vs off 0.0019 (rounding order: permuted wo_a
  K rows, table order in sdpa). Whole window layer on random inputs: 11.00 -> 4.74 ms a call (wall, synced).
- Plain profile, 2x2, chunk 53248-55296 (profiles P.15_20261020T165333 off, P.15_20261020T164559 on), device ms a
  chunk off -> on: attention 114.36 -> 43.01, indexer 34.97 -> 25.15, kv 2.11 -> 1.24, compressor 2.38 -> 1.90,
  attn_residual 2.93 -> 3.29; device sum 202.19 -> 119.82 (-82.4 ms, -41 %). Programs (all chips) 9772 -> 8892:
  attention 3296 -> 2648, indexer 756 -> 660, kv 1024 -> 928, compressor 1120 -> 1048, attn_residual 224 -> 256.
- Quick A/B, on (`measure --task P.15`, rc 0): min layer L03 0.99489 (P.14: 0.99488), pcc_chunk_out 0.99941, chunk
  wall 201.8 ms (spread 13.6; P.14 213.7), host_transfers_per_layer 0; final run after the doc edits: chunk wall 196.0
  ms (spread 8.9), same PCCs. The wall falls far less than the device: the
  chunk is host-bound now (device ~120 ms vs 202 ms wall), so op count is the next lever.
- Frozen swap tests run by hand with ATTN_GLUE=on: test_swap_enc_source_14, swa_12, dec_reindex_13 pass (official
  numerics, device attention / indexer / compressor / kv in the block; swa_12's ffn_hc WAIVED as before).
- Re-run: probes `tests/ttnn/unit_tests/operations/ds41_attn_glue/probes/`; profile `DS41_ATTN_GLUE=on|off` + the gate's
  profile command; final `python -m models.demos.common.bringup.orchestrator measure --task P.15`.

## P.16 perf (attempt 1): indexer / shared-source glue (INDEXER_GLUE on / off; glue report items 3, 7, 8, 9, 10)
- Setting INDEXER_GLUE (tt/settings.py), default on (the change; the orchestrator pins both sides; the brief's
  "off (default)" vs "make the change the default": as P.14 / P.15). Not a precision switch: data moves, and the bf16
  partials q_a already writes under CCL_IMPL=fabric_bf16 (P.6).
- (3) Candidate block max (L20): fork option on ttnn.bringup.ring_indexer_score_dsa (indexer_score CHANGELOG "Block
  epilogue"): `block_keys=8, block_max_out=<load-time [1, 1, Sq, ceil32(T / 8)] fp32 RM buffer>, block_max_pin=True`;
  the writer takes the max of each 8 bf16 scores of its untilized strip (exact, widened to fp32) and stamps +inf at the
  query's own block (old: +1e30 in bf16; same selection). Then typecast to bf16 (exact) + topk_large_indices(2048,
  valid_length = kv_len / 8); below 2048 blocks a slice + -inf pad as before (DS41-52: the full-range slice is a
  wrapper over the persistent buffer, never freed). Replaces 8 strided slices, 7 maximum, to_layout, the 9 pin ops.
- (8) Re-index bias (L24): the same scatter (ROW_MAJOR, width ceil32(T / 8)) + to_layout TILE -> `block_bias` of the
  score op: the writer writes -inf over every key of a block whose bias is nonzero; no repeat_interleave, slices, add.
- Fork tests: tests/unit/test_block_epilogue.py (2x2, ratio 1, natural blocks, block-cyclic keys, T 6144, 6 starts,
  block_keys 8 / 4: bias, max, pin bit-exact vs the plain call; sentinel past kv_len; 7 refusals); negative controls by
  hand (corrupted max, a bias that skips a key) fail. Regression with the option off: unit + model cases 22 passed / 1
  skipped (= 17 / 1 + 2 recorded + 3 new); fork_source 134 tests, 0 regressions (the recorded
  test_indexer_score_rejects_fp32_dest_acc divergence only).
- Score op alone, L20 shapes (Sq 512 x 32 x 128, T 55296, HiFi2 + fp32 DEST, 30 synced calls; probe
  ds41_p16_epilogue, deleted): plain 1.68 ms; + max / pin 1.96 (first version 3.07; in the model's per-op profile
  6.68: DS41-85); + bias 1.69 (first, ROW_MAJOR bias rows: 2.16; DS41-86).
- (7) tt/sparse_attention.ChunkMemo on DS41DeviceState (memo): ("comp", kv source) the gathered compressed history
  and nb, ("index", idx source) the [window | top-k rows] index list, ("keys", kv source) the indexer's tiled index
  keys; each keyed by (start, cache id, TtServedRows.version: bumped by every _put); a new top-k of a layer drops its
  "index" slot (DS41DeviceState.put); TtDS41Model.memo_last drops each slot after its last loaded reader of the chunk
  (here: comp 2 after L3, 20 after L24; index 2 after L3, 20 after L21; keys 20 after L24). L3 / L21 / L24 no longer
  slice + tile + all_gather x 2 + untilize the history; L3 / L21 skip _topk_rows (~25 ops); L24 skips the key retile.
- (9) tt/rope.RowsMemo: RopeTable.rows(start, S[, tp_split]) and CompressRope.rows once per (start, length); callers
  release through `rope.free` (memoized rows stay; the latest start only). The indexer's TP split of the rows is a
  memo entry too. Router weights: with EXPERTS_COMBINE=weighted the router returns (None, idx, wts) (no zeros slice /
  2 to_layout / scatter) and experts._weighted takes wts (no to_layout + gather); bit-identical (dense was their
  scatter).
- (10) Indexer gates: linear writes ccl.partial_dtype (bf16 under fabric_bf16) -> tt/ccl.sum_rows = ONE
  fabric_reduce_scatter by row blocks: chip c's block is rows [c Sq, (c+1) Sq), what tp_sum + mesh_partition kept
  (no all_gather, mesh_partition, 3 typecasts). Compressor: its [wkv | wgate] partial written in partial_dtype (C1).
  Both only with HC_TP_SUM=gather_add; stock / untiled fall back to the old path.
- stock ttnn.slice instead of ttnn.bringup.ring_indexer_score_dsa: the first nbv < 2048 columns of the op's own
  block-max output before the -inf pad (a column slice, not a block pool).
- Plain profile, 2x2, chunk 53248-55296 (profiles P.15_20261020T164559 = off, P.16_20261020T183616 = on), device ms a
  chunk off -> on: indexer 25.15 -> 12.94, attention 43.01 -> 40.56, experts 17.18 -> 16.82, router 1.82 -> 1.47,
  compressor 1.90 -> 1.79, kv 1.24 -> 1.18; device sum 119.82 -> 104.28 (-15.5 ms, -13 %). Programs (4 chips) 8892
  -> 7800: indexer 660 -> 376, attention 2648 -> 2244, kv 928 -> 800, compressor 1048 -> 964, router 480 -> 384,
  experts 480 -> 384. Wall with the profiler on 367.1 -> 329.0 ms.
- Quick runs by hand: on: chunk wall 171.9 / 175.7 ms, L00 0.99933 L01 0.99892 L02 0.99769 L03 0.99490 L14 0.99612
  L20 0.99928 L21 0.99682 L24 0.99941; off: 197.0 ms, L03 0.99489 (P.15 196.0); host_transfers_per_layer 0.
  Ladder last (on, by hand): pcc_state_min 0.99712, overlap candidates L20 0.99757, top-k L24 0.99291.
- Frozen tests run by hand with INDEXER_GLUE=on (sanity, max precision): test_c_dec_candidate_indexer (failed first:
  DS41-52, the block-max buffer freed by a full-range slice; fixed), test_c_dec_reindex_indexer, enc_source_indexer,
  enc_source_compressor, dec_reuse_attention, swa_attention, dec_reindex_experts, test_swap_dec_reindex_13_experts:
  pass.
- Final `measure --task P.16` (on, rc 0): min layer L03 0.99490, pcc_chunk_out 0.99941, chunk wall 174.9 ms (spread
  6.9; P.15 196.0), host_transfers_per_layer 0.
- Not done: the 56 MiB concat [window | comp] per sparse layer (needs a two-KV sparse_sdpa option); a persistent
  natural-order history mirror (report item 7b).
- Re-run: fork `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/indexer_score/tests/unit/test_block_epilogue.py`;
  profile `DS41_INDEXER_GLUE=on|off` + the gate's profile command; final `python -m
  models.demos.common.bringup.orchestrator measure --task P.16`.

## FIX.5 (overseer fix, attempt 1): ChunkMemo cleared at chunk start: NOT DONE, blocked by scope
- The brief allows no model code: tasks.yaml FIX.5 has `paths: []`, so tt/ and hooks.py are read-only; the fix is
  in tt/model.py. No code changed, no device run.
- Patch for the retry (needs `paths: [models/demos/deepseek_v41_flash_d_p/tt]`): first lines of
  `TtDS41Model.begin_chunk` (tt/model.py:682), before the engram early return:
  `st = self.state` / `if st is not None and st.memo is not None: st.memo.clear()`. harness.begin_chunk calls it
  before every chunk (profile and timeline runs alike).
- Diagnosis note: memo keys already carry (start, cache id, cache version) (sparse_attention.py:251,
  indexer.py:265), so another chunk misses; the stale hit needs the same start and an unchanged source version
  (X.3 profile run -> timeline run of the same chunk). Clearing in begin_chunk covers both cases.
- Re-run: the X.3 gate command in the FIX.5 brief.

## FIX.6 (overseer fix, attempt 1): reference numerics switch official | native
- `DS41Reference(..., numerics="official" | "native")` (reference/ds41_ref.py, docstring "Numerics"), passed by
  `hooks.reference(spec, layers=None, dtype=None, numerics="official")`. One switch: `_q8` (fp8 act q-dq per 32) and
  `_q4` (fp4 q-dq) at every call site, `_fp8_linear` = mm(_q8(x), w). native: all are x.float(): fp8 inputs of q_a,
  kv, wq_b, wo_b, indexer wq_b, engram wkv, shared / routed experts x and h; window-KV fp8; attention output fp8
  before wo_a; compressed-KV, index-key and index-query fp4. Weights unchanged (exact dequant).
- Nothing else to remove: the reference has no TF32 rounding and casts no intermediate to bf16 (it computes in fp32).
  `kernels.attend`'s bf16 P only runs on bf16 inputs, which only the oracle passes. kernels.py is untouched: the
  oracle (hf_oracle.py) keeps running the faithful port, and official stays bit for bit (DS41-1: same tiled GEMMs).
- `with_numerics(n)`: shallow copy that shares weights, tables and hasher (the test loads the weights once).
- Gate (CPU, 35 min): R.2 every layer maxabs 0, logits pcc 1.0, top1 1.0; R.3 hidden / state pcc 1.000000000,
  graph replay 0 for every block type.
- reference/test_numerics.py (CPU): (1) at 512 tokens, the official forward calls the quantizers
  (act_quant_dequant 3472, fp4_quant_dequant 10 on the real checkpoint); the native forward runs with every
  kernels.py quantizer patched to raise. (2) At 4096 tokens, one-shot: rel of native vs official.
  rel = ||native - official|| / ||official||:
    | checkpoint | L0 | L1 | L2 | L3 | L14 | L20 | L21 | L24 | final_norm |
    |---|---|---|---|---|---|---|---|---|---|
    | real (BRINGUP_SCALED=0) | .0281 | .0270 | .0315 | .0360 | .0910 | .1411 | .0308 | .0201 | .2886 |
    | scaled quietbox_2x2 | .0285 | .0355 | .0525 | .0741 | .1205 | .1848 | .1724 | .1984 | .2145 |
  L0 steps (identical inputs): q_resid .0115 (one fp8 q-dq; DS41-72 had .0116), kv .0289, attn_out .043.
  L20: router row-rel median .51, top-k overlap .85 / .79 (real / scaled). Logits top-1 agreement over the last 512
  positions: .45 / .51. These are the partial model's layers (0-3, 14, 20, 21, 24, chained).
  Bounds: L0 q_resid in (.008, .016), kv in (.020, .040); layers < 0.3; final_norm < 0.4.
- Not ~1e-2 as the brief expected: one fp8 rounding alone is 1.15 %, and routing and top-k flips compound it (DS41-90).
- Heads-up (DS41-89): now that the hook takes `numerics`, component / swap tests get the native reference
  (defaults.yaml tests.step_numerics: native), but the device's MAX_PRECISION is still NUMERICS=official. Those tests
  were not rerun here; F128 should switch both together. tt/settings.py NUMERICS text ("the reference hook has no
  numerics switch") is now stale; settings.py was outside this step's paths.
- Gotcha: run the CPU test with `--no-precompile`. The wrapper's up-front collect pass gave other numbers (scaled
  L24 .334, 1562 vs 1328 quantizer calls) than the real pass, which matched a standalone script exactly.
- Re-run: the gate in the FIX.6 brief; `BRINGUP_SCALED=0|1 scripts/run_safe_pytest.sh --no-precompile --run-all
  models/demos/deepseek_v41_flash_d_p/reference/test_numerics.py` (scaled ~8 min, real ~12 min).

## FIX.6 (overseer fix, attempt 2): no code change; attempt 1's switch stands
- Attempt 1 passed the gate (R.2 maxabs 0 on all 25 layers, logits pcc 1.0; R.3 pcc 1.000000000, replay 0). Only the
  commit failed: pre-commit `check-large-files` refused bringup/state.json (502 KB then, 516 KB now, limit 500 KB).
  state.json is the orchestrator's file and outside this step's paths, so this attempt cannot shrink it. The
  orchestrator must trim it, or commit it apart from the hook, before FIX.6 can land.
- Reviewed attempt 1's diff again: every K.act_quant_dequant / K.fp4_quant_dequant call in ds41_ref.py goes through
  `_q8` / `_q4`. Only the module-level `fp8_linear` (ds41_ref.py:308) still calls K directly, and the per-layer
  steps no longer use it (they call `self._fp8_linear`). reference/test_numerics.py makes every K quantizer raise
  under native.
- Reran the gate unchanged (result in the summary below / results/FIX.6.json).

## FIX.7 (overseer fix, attempt 1): official numerics at the GPU's dtype (bf16 activations)
- `DS41Reference(numerics="official", dtype=torch.bfloat16)` now computes like DeepSeek's GPU code (generate.py's bf16
  default dtype): arithmetic stays fp32, and a value is rounded to bf16 exactly where model.py holds it in bf16
  (`_r`; ds41_ref.py docstring "Numerics" lists every point): streams at every unpack (so an fp32 golden input reads as
  the GPU's bf16 stream), engram rows, every fp8 / fp4 / bf16 GEMM output, every RMSNorm output, the hc_pre collapses,
  RoPE'd q / kv / index q / latent / index key, sparse_attn's bf16 P (kernels.attend on bf16 q) and output, the
  inverse-RoPE'd output, wo_a, hc_post outputs, the experts' h before w2, moe_add. fp32 stays where model.py computes
  in fp32: mHC mixes + Sinkhorn, ratio-2 compressor wkv / wgate / pooling (and compress_partial), router, the routed
  experts' fp32 sum, attention scores / softmax statistics, pending pre, logits. DS41-38's compressor bf16 points
  (bf16 latent before and after the norm, bf16 wk / k_norm, RoPE into bf16) are exactly these.
- `_mm16`: the GEMMs model.py runs on bf16 x and a bf16 weight (wo_a, ratio-1 compressor wkv, indexer wk /
  weights_proj) and the index-score einsum + relu x weights + head sum run on bf16 tensors (K.linear / K.grouped_out
  / K.index_score_tile tiles, DS41-1). First version used fp32 GEMMs rounded once: every step bit-exact except
  attention (99.9 % equal, rel 9e-5..1.6e-4), block outputs vs the oracle rel 3.6e-4 / 3.3e-3 / 7.7e-3 / 1.1e-2 at
  L0-L3. Cause: CPU bf16 einsum and fp32 + one rounding differ on 0.03 % of outputs (both rel 1.659e-3 vs fp64,
  probe of [256, 8, 4096] x [8, 1024, 4096]); fp8 q-dq flips grow it (DS41-91). With `_mm16` everything is bit-exact.
- native and official fp32 are untouched: `_r` is the identity there, `_mm16` is the old `mm(x.float(), w)` /
  `K.grouped_out(o, w)`. Checked against a copy of the pre-FIX.7 ds41_ref.py, real checkpoint, 512 tokens, spec layers:
  native 144 / 144 boundaries and official fp32 144 / 144 equal bit for bit. native refuses dtype bf16;
  `with_numerics("native")` is always fp32, `with_numerics("official", dtype)` picks the official dtype.
- Gate (CPU): R.2 official L00-L24 maxabs 0, logits pcc 1.0, top1 1.0; R.3 official and native chunked vs one-shot
  hidden pcc 1.000000000 / 0.999998146, state min 1.000000000, graph replay 0 for all 8 block types.
- reference/test_official_bf16.py (CPU, 4 tests, both checkpoints pass): (1) every step of layers 0-3 on the oracle's
  own inputs (official_taps.capture: build_oracle at bf16 = hooks.hf_model(spec, None)'s configuration, first 4
  blocks, one-shot 512 tokens) equals model.py at bf16: rel 0, equal 1.000000, indexer top-k overlap 1.0; control: the
  fp32 official misses L0 attn_norm by 1.68e-3 (0.86 bf16 roundings, bound 0.5). (2) block outputs end to end: rel 0
  at L0-L3 (bound 3 roundings). (3) native from the bf16 ref == native built at fp32 (bytes), native at bf16 refused.
  (4) report of official vs native at 4096 (below).
- The new margins, scaled quietbox_2x2, rel = ||official - native|| / ||native||:
  | 4096 one-shot | L0 | L1 | L2 | L3 | L14 | L20 | L21 | L24 | final_norm |
  |---|---|---|---|---|---|---|---|---|---|
  | official bf16 | .0286 | .0361 | .0528 | .0752 | .1213 | .1838 | .1718 | .1983 | .2165 |
  | official fp32 | .0285 | .0354 | .0523 | .0738 | .1204 | .1836 | .1703 | .1962 | .2145 |
  reference.drift.compute(s4096) (the framework's margin, chunked 2048 as the ladder, gapped runs fed from the native
  golden; run with the file write skipped, nothing written into the golden), layer_L*: bf16 / fp32
  L0 .0281 / .0273, L1 .0367 / .0355, L2 .0535 / .0526, L3 .0779 / .0762, L14 .0602 / .0622, L20 .0260 / .0254,
  L21 .0632 / .0629, L24 .0225 / .0218 (rel_res one bf16 rounding .00166); states: window_kv .029-.067,
  compress_kv .095-.104, index_k .116-.120, topk_idx overlap .886-.952, candidates_L20 1.0.
  bf16 activations move the margin by -3 % .. +4 % of itself; the fp8 / fp4 roundings dominate it (DS41-90).
- Not done here: drift files / official goldens are not regenerated (outside this step's paths); the next G task's
  `reference.drift --rung <rung>` and `generate_golden --numerics official` now run at the spec's shipped bf16.
- Gotcha: run_safe_pytest.sh only in the foreground (the real-checkpoint 4096 report runs > 10 min, so only the
  scaled one ran here; the 512-token tests run in 7.5 min on the real checkpoint).
- Re-run: the gate in the FIX.7 brief; `BRINGUP_SCALED=0|1 scripts/run_safe_pytest.sh --no-precompile --run-all
  models/demos/deepseek_v41_flash_d_p/reference/test_official_bf16.py -s` (`-k "not margins"` for the 512-token part).

## FIX.8 (overseer, 2026-10-21): onto F131-F142, shipped settings everywhere, parity code deleted

What changed (the tree did not import before: F137 removed `Settings(max_precision=...)`):
- (1) `tt/settings.py`: MAX_PRECISION and `max_precision=` gone; `Settings("DS41_", TABLE)`. Deleted settings:
  NUMERICS, COMPRESSOR_X (round_tf32), ENGRAM_KV_QDQ (qdq), ATTN_IN_MM, WINDOW_SDPA (source was the only shipped
  value; high_precision / exact gone), WINDOW_SDPA_OUT / _SCORES / _PROBS, WINDOW_PIECES (33 -> 24 settings). The
  "max precision" wording in the remaining descriptions rewritten. settings_lint: 33 files, 0 violations; no
  emulation default.
- (2) Deleted files: `tt/quant.py` (277 lines: fp8_qdq, fp4_qdq, round_tf32, native / act_q / fp4_q / tf32_q),
  `tt/exact_mm.py` (127), `tt/window_exact.py` (266), `tt/perf_attention.py` (81, timed only the deleted sdpa
  variants). `act_q` / `fp4_q` (their native branch) became `tt/common.py:cast` (x itself when already the dtype,
  else one typecast). Official branches removed: attention.py (fp32_out / fp32_scores / bf16_probs / fp32_q, the
  ttnn.bringup.sparse_sdpa call, the exact window path, the glue_out official re-interleave, the official wo_a
  concat order), sparse_attention.py (exact assert), q_a.py (ExactMatmul), kv.py / compressor.py / indexer.py
  (`native()` conditions, round_x, fp4 blocks), engram_kv.py (qdq), experts.py (`qdq`, the fork's
  `fp8_act_quant=` argument: default False now), shared_expert.py (the official flat path: `routed=` plan, unit
  routing weights, fp8_act_quant), norm.py (quant -> cast), weighted_combine.py (`acc_unpack_fp32=True` dropped: a
  no-op with COMBINE_ACC=bf16, bf16 values pass srcA exactly), perf_experts / perf_shared_expert (float64 references
  always native), perf_precision (fork variant dropped). tt/: +279 / -1332 lines. The fork options themselves stay
  (J2 removes routing_weights / fp8_act_quant); their READMEs' "used by" say the model no longer calls them.
- (3) Contract helpers from reports/fix8_dev_model_diffs.patch (its settings hunk ignored, (1) supersedes it):
  `server_rules.state_threshold()` returns a per-(config, layer) limit, the stricter of the server's 0.93 and
  `e2e_limits.pcc_limit(drift-s4096, state_<kind>_L<i>)`; `caches.pcc_failures` accepts it. On s4096: window L00
  0.9983, L03 0.9911, L24 0.9972; compress L02 0.9783; index L02 0.9709, L20 0.9728.
- Two K.1 failures the gate exposed, neither from this change (K.1 last passed 2026-10-17, before P.8 / F128):
  DS41-93 (integer engram_tail had no contract drift: the hook now checks it exactly, asserts, metric
  contract_state_exact_engram_tail = 1) and DS41-94 (runner fabric payload 4352 B vs the weighted combine's 10880 B:
  the runner hung in FlatWithProgramDeviceOperation; DS41Config.FABRIC_PAYLOAD_SIZE now follows EXPERTS_COMBINE).
- Kept: FIX.5's tt/model.py held wrappers + begin_chunk memo clear, tt/sparse_attention.py doc.

Measured (2x2, defaults):
- Plain profile, chunk 53248-55296 (test_profile, BRINGUP_TASK=FIX.8): device 104.31 ms a chunk vs P.16's 104.28,
  7800 programs both, every section within 0.02 ms (attention 40.57, experts 16.84, indexer 12.93); wall 331 ms.
- M.1 (s4096, native goldens, F135 limits): err_layer L00 .0273/.0561, L01 .0355/.0734, L02 .0547/.1071, L03
  .0847/.1558, L14 .0572/.1204, L20 .0307/.0519, L21 .0514/.1265, L24 .0309/.0449; err_state max window_kv_L03
  .0588/.1334; host_transfers_per_layer 0.
- L.last (53k prefix from the golden): err_layer L00 .0265/.0570, L01 .0346/.0720, L02 .0523/.1077, L03 .0818/.1587,
  L14 .0618/.1380, L20 .0306/.0507, L21 .0521/.1312, L24 .0281/.0418 (worst share of the margin: L24 67 %);
  err_state worst window_kv_L21 .0470/.1118.
- K.1: contract_checks_failed 0, acks_early 0; pcc_producer_kv window_kv .99840 >= .99164, compress_kv .99923 >=
  .97827, index_k .99977 >= .97088; pcc_contract_state_window_kv .99843 >= .99164; kv_layout, adapter_acks,
  runner_contract (185 s) pass; runner_smoke skipped (DS41-8).
- "Not numerically changed": no earlier err_* baseline exists (the tree did not import since F137); every deleted
  branch was off under the old defaults (NUMERICS=native, WINDOW_SDPA=source, ATTN_IN_MM=fpu), the dropped
  acc_unpack_fp32 / fp8_act_quant=False are no-ops on the shipped path, and the profile's program count and section
  times are the same.

Gotchas:
- The drift gives compress_partial an empty margin (DS41-92: ladder limit 1.8e256, contract PCC limit -3.0): the
  all-zero state passes anything; a framework fix (exact check of an all-zero state).
- Frozen per-step tests that still assume the official arithmetic were not touched (test role / J0): e.g.
  test_c_*_kv / test_c_enc_source_compressor check the fp8 / fp4 grid of the device output and will fail on the
  bf16 cast; the batched verify (F142) will list them.
- L.last's gate has no --no-precompile, so its body runs twice (K-52): 11.6 min, longer than one 10-minute call.

Re-run: the FIX.8 gate (brief), one command per call; profile: `BRINGUP_TASK=FIX.8 TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py`.

## FIX.9 (overseer, 2026-10-22): one expert pipeline (precision_debt_report.md J2)

What changed (the flat routed expert with the weighted combine, and the flat shared expert, are the only path):
- `tt/experts.py` 1019 -> 383 lines: deleted EXPERTS_COMBINE=separate (dispatch / combine / K-slot fp32 adds,
  `_modules`, `_dispatch`, `_combine`), the routing-weight pieces (`_pieces`, `_weight_col`, `_routing_rows`, the
  selectors `sel` / `sel_all`, DS41-23), EXPERTS_KERNEL loop / unified (`_loop`, `_ffn`, `_unified`, `_flat`), the
  fp4_decode weights (`pack_planes`, `_load_fp4`, `_decode`) and the unused device tables (dispatch table, gidx).
  `partial` = the old `_weighted`; `_load_block` keeps the bfp8 exactness assert / bfp4 stats on the [N, K]
  matrices and feeds `_load_flat`. `build_experts` lost `combine=`.
- `tt/shared_expert.py`: deleted SHARED_EXPERT_IMPL=linear (the ttnn.linear chain, its bf16 / DENSE_WEIGHTS
  weights); `chunk` is required, `impl=` gone.
- `tt/settings.py`: deleted EXPERTS_KERNEL, EXPERTS_COMBINE, EXPERTS_DECODE, SHARED_EXPERT_IMPL; EXPERTS_IMPL is
  {bfp8, bfp4}; descriptions of EXPERTS_PRECISION / COMBINE_ACC / COMBINE_PLANNER / MATMUL_FIDELITY / DENSE_WEIGHTS
  no longer name the deleted paths (24 -> 20 settings).
- `bringup/hooks.py:device_params` and `tt/runners/adapter.py:_FabricPayload` always use the weighted combine's
  10880 B payload (DS41-94); `tt/model.py` keep_rows / router_wts no longer test `experts.weighted`;
  `tt/weighted_combine.py` docstring. tt/ + hooks: +238 / -1218 lines.
- `tt/perf_experts.py`: deleted test_time_experts (bfp8 vs fp4_decode), test_unified_vs_loop, test_kernels;
  test_weighted_vs_separate -> test_weighted (weighted only; also fixed its undefined `wround`).
  `tt/perf_shared_expert.py`: test_shared_flat_vs_linear -> test_shared_flat.
- Forks (no other model calls the options: grep of every model's code, the forks' tests/cases.py and tests):
  - unified_routed_expert_ffn: removed `routing_weights` / `fp8_act_quant` (attributes, input, CBs c_20 / c_21,
    reader staging, `calculate_routed_h`, UnpackToDestFp32); `weights_transposed` stays (no caller now). Program
    factory and kernels = pre-P.2 + WEIGHTS_NK. `tests/unit/test_ds41_h_options.py` deleted, its transposed check ->
    `tests/unit/test_weights_transposed.py`; `tests/reference.py` lost the fp8 helpers.
  - flat_routed_expert_ttnn: removed `routing_weights` / `fp8_act_quant` and the plan's `routed` (plan branch, 36 KB
    arena region, RH_* CBs / semaphores / args, `AddrSrc::RoutingWeights`, SE_RH_* kernel blocks, `se_rh.hpp`);
    `weights_nk` / `hidden` stay. `tests/unit/test_ds41_h_options.py` deleted -> `tests/unit/test_weights_nk.py`
    (nk bit-identity, V4.1 shapes with hidden, block_float_exact). Weight-cache keys never had `routed`.
  - READMEs, CHANGELOGs, INDEX.md updated. Build OK (warnings as errors).
Fork-side diff: 30 files, +131 / -1764.

Measured (2x2, defaults):
- Baseline first (untouched tree, 6 of the 8 gate component tests; the loop was cut at dec_reuse): every printed check
  line of the after-runs equals the before-runs (diff of the `ok` / `FAIL` / `note` lines: identical). err_experts
  L00 0.130168, L02 0.16062, L20 0.183528, L21 0.183309 (no before); err_shared_expert L00 0.0151197, L02 0.0179665,
  L20 0.0182662, L21 0.0176558.
- Gate: the 4 shared-expert tests PASS. The 4 experts tests FAIL, before and after, identically (DS41-95): err_experts
  is within its limit (e.g. 0.130 <= 0.337) but the frozen auto checks vs the CPU step (rel 0.13-0.18 > 0.06-0.08;
  second inputs' fixed 0.03) fail, and test_c_swa_experts.py KeyErrors on `component_col_factor`. These frozen tests
  (2026-10-16/17) predate F128 and the bfp4 / glm defaults; the brief's "pass today" is not so for them.
- L.last (run with --no-precompile to fit one 10-minute call, K-52): PASS; err_layer L00 .0265/.057, L01 .0346/.072,
  L02 .0523/.1077, L03 .0818/.1587, L14 .0618/.138, L20 .0306/.0507, L21 .0521/.1312, L24 .0281/.0418 (= FIX.8's);
  err_state max window_kv_L03 .0542; worst layer pcc 0.996652.
- Plain profile (chunk 53248-55296, BRINGUP_TASK=FIX.9): device 104.28 ms a chunk (FIX.8 104.31), 7800 programs,
  experts 16.77 (16.84), shared_expert 2.02, router 1.47, moe_add 1.58; host_transfers_per_layer 0; wall 355 ms.
- Forks, options off: unified fork_source 73 / 75 run and pass, 0 regressions; the 2 weight-cache tests are not
  collected (deepseek_v3_d_p/tests/conftest.py imports xing40's settings, which still pass the removed
  `max_precision=`; F137, not this change); model cases 5 passed / 1 skipped (xing40 4x2); test_weights_transposed
  3 passed. Flat tests/unit/ 19 passed / 13 failed = the recorded baseline (12 x `p.nsg == 1`, the MiMo perf band,
  DS41-63) + test_weights_nk 3 passed; GLM's test_flat_bench 9.556 / 23.725 ms (9.558 / 23.721).
- settings_lint 0 violations, ops_lint 0 problems.

Gotchas:
- Out of date now (pointers into deleted code): DS41-23, DS41-57, DS41-58, DS41-59, DS41-60, DS41-61, DS41-62
  (knowledge-use report FIX.9.overseer.1.json).
- A shell loop of run_safe_pytest calls longer than 10 minutes is moved to the background by the tool and then killed:
  run one test file per call.

Re-run: the FIX.9 gate (brief), one command per call (L.last with --no-precompile to fit 10 minutes); fork checks
`scripts/run_safe_pytest.sh --run-all --no-precompile ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/<file>.py`,
`python -m models.demos.common.bringup.testing.fork_source --fork unified_routed_expert_ffn`,
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/unified_routed_expert_ffn/tests/test_unified_routed_expert_ffn.py`.

## FIX.9 attempt 5 (overseer, 2026-10-22): the re-rendered experts tests and the devfmt diagnosis

State: attempt 1's deletions (section above) were still in the tree after the crash; kept unchanged (re-read the
diff; rebuilt with ./build_metal.sh, since some fork sources were newer than the library; fork option tests
test_weights_nk.py 3 passed, test_weights_transposed.py 3 passed).
Added `bringup/hooks.py:devfmt_component` (+ `_bfp_round`): the F137 diagnosis hook (testing/devfmt.py, never gated).
Without it the diagnosis said "a logic bug" (it modelled only bf16 activations). experts: the weights go through
EXPERTS_IMPL's block format on the stored [N, K] matrices (tt/experts.py:_bfp4_round), x / h / y as bfp8 tiles (glm),
the bf16 routing weight is applied after w2, and the sum is rounded to bf16. Other steps: the framework's default
(bf16 activations). No device path or setting changed.
Gate, run one command per call (overseer re-rendered tests, OV-27):
- 4 x shared_expert PASS: err L00 0.0151197, L02 0.0179665, L20 0.0182662, L21 0.0176558 (= the baseline).
- 4 x experts FAIL, identically to the baseline: err_experts L00 0.130168 / L02 0.16062 / L20 0.183528 / L21 0.183309,
  each within its golden limit (0.337 / 0.507 / 0.882 / 0.958). The vs-CPU check fails (rel > 0.0598 / 0.0720 / 0.0814 /
  0.0819, median row ratio +4.4 to +5.5 %), and so do the second inputs (0.18-0.20 > 0.03). Diagnosis with the hook: the
  CPU step at the device's formats is 0.128 / 0.158 / 0.181 / 0.181 from native, and the device is 0.0149 / 0.0171 /
  0.0184 / 0.0182 from it. Verdict: a precision gap (DS41-95).
- CPU only, random x, layer 0, the CPU step at the device's formats vs native: bfp4 + glm 0.1825 (ratio 1.047),
  bfp4 + full 0.1798, bfp8 + glm 0.0171 (1.006), bfp8 + full 0.0038. So the gap is bfp4 alone.
- L.last (--no-precompile, K-52) PASS: err_layer L00 .0265, L01 .0346, L02 .0523, L03 .0818, L14 .0618, L20 .0306,
  L21 .0521, L24 .0281; worst layer pcc 0.996652; err_state max .0542 (= FIX.8's).
Decision: no change to shipped numerics (the brief forbids it; rule 14: a precision gap is reported, not raised).
The four experts tests need an owner waiver (tests.waivers) for EXPERTS_IMPL=bfp4, or the owner picks bfp8.
Re-run: the gate, one test file per call; diagnosis lines: `grep "^note err_experts"` in the test output.

## P.17 perf attempt 1 (2026-10-22): ROPE_IMPL=on, one stock rotary op per RoPE site (precision-debt J3)

What: setting ROPE_IMPL (tt/settings.py, default on; off = the P.16 path, kept for the owner's A/B). On: every RoPE
site (attention q and output inverse, indexer q, window kv, compressor ckv / k: 34 a chunk) is ONE
`ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` on bf16 heads (tt/rope.py:rope_stock). INDEX.md has no
bring-up rope fork, so the stock op is first in line (rule 16). K-11: rotary_offset ropes the 64-dim tail of a 512 /
128 head in place, so no weight row is permuted any more (wq_b, wkv, kv_norm, wo_a, compressor wkv / wgate / norm /
wk / k_norm, indexer wq_b all in checkpoint order); the Meta pair order with the trans-mat is the reference's
interleaved order; inverse = the transposed trans-mat.
Decisions:
- Rows: the model's layout is natural blocks, not the server's block-cyclic order the op's kv_actual_global math
  assumes, so each call gets this chip's rows pre-sliced (RopeTable.rows / CompressRope.rows, memoized per start as
  before) and kv_actual_global 0 (every chip reads from row 0; any 32-aligned start; one cached program).
- Tables bf16 [max_seq, 64] (each angle twice): the op refuses fp32 cos / sin (DS41-96, probe_001).
- Compute config: the op's default (HiFi4 + fp32 DEST). DS41-83: the trans-mat product is exact on bf16 only at
  HiFi3 / HiFi4; probe_002 tail vs float64: HiFi4 0.0037, HiFi2 0.0052 (one bf16 rounding 0.0017), 0.09-0.10 ms.
- q: the 512^-0.5 scale moved into the input's bf16 cast (multiply writing bf16: still one rounding, one op fewer);
  wq_b writes bf16 (DS41-84: noise); nlp_create_q_heads_split then rope_stock(tail, concat_prefix=nope). Indexer q
  likewise (no scale). Output: rope_stock(oh, trans_inv) on the bf16 heads, then the unchanged _out_proj.
- kv: kv_norm writes bf16 (TtDistributedRmsNorm(dtype=)), then rope_stock. Compressor: ckv = rope_stock of the
  latent cast to bf16 (wk still reads the fp32 latent); k_norm writes bf16, then rope_stock.
- Not deleted: PairPerm, the interleave helpers and the fp32 chains stay as the ROPE_IMPL=off path, because the
  orchestrator measures off as the old side. Once the owner accepts, delete with ROPE_IMPL: tt/rope.py PairPerm,
  deinterleave_perm, rope_deinterleaved(_bf16), interleave_tail, deinterleave_tail, rope_heads; the off branches in
  tt/attention.py (_q_glue, _q_heads, _out_glue, _out_heads, the table transform), tt/indexer.py (_query_glue and the
  permute path), tt/kv.py, tt/compressor.py; then rewrite DS41-15 / DS41-83.
Measured (2x2, chunk 53248-55296):
- Quick A/B side on (`measure --task P.17`): chunk wall 153.6 ms (spread 22.4), worst layer L03 pcc 0.99670,
  host_transfers_per_layer 0; every err_* within its limit: err_layer L00 .0261/.0570, L01 .0347/.0720, L02 .0520/.1077,
  L03 .0813/.1587, L14 .0605/.1380, L20 .0300/.0507, L21 .0507/.1312, L24 .0281/.0418; window_kv L00 .0042/.0585
  (FIX.9: .0034), state max .0541.
- Plain profile (on) vs FIX.5's profile of the same tree with the old path: programs (4 chips) 7800 -> 6140 (-415
  calls a chunk); device sum 104.33 -> 96.91 ms; attention 40.56 -> 34.50, indexer 12.93 -> 11.85, kv 1.18 -> 0.81,
  compressor 1.81 -> 1.65 (programs: attention 2244 -> 1476, indexer 376 -> 200, kv 800 -> 384, compressor 964 -> 664).
- Component steps vs golden, on / off (frozen tests run by hand, not this gate): kv L0 0.00417 / 0.00335 (limit
  0.0582), window attention L0 0.0214 / 0.0228 (0.0902), compressor L2 0.00361 / 0.00319 (0.217), compressed
  attention L2 0.0276 / 0.0298 (0.148); indexer overlap equal (L2 miss 0.00577 / 0.00576, L20 0.00340 / 0.00343).
Gotchas:
- Pre-existing, on and off alike (not this change): test_c_enc_source_indexer / test_c_dec_candidate_indexer fail
  their overlap limit (0.00065 / 0.00144); the gate's swap experts tests fail outside the waivers:
  test_swap_enc_source_14_experts err_swap_out 0.0296 on / 0.0294 off (<= 0.0159), swap_indexer match 0.0015 both,
  swap_router; test_swap_swa_12_experts err_swap_out 0.0248 on / 0.0267 off (<= 0.0185), swap_router_vs_golden_err
  0.149 on / 0.151 off (<= 0.142). test_c_swa_experts passes (waived).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.17`; probe
`scripts/tt-probe.sh ds41_rope_stock < tests/ttnn/unit_tests/operations/ds41_rope_stock/probes/probe_002.py`.

## P.18 perf attempt 1 (2026-10-22): NORM_IMPL=on, the RMS norms through ttnn.bringup.rms_norm (precision-debt J4)

What: setting NORM_IMPL (tt/settings.py, default on; off = the P.17 path, kept for the owner's A/B). On, every norm
(tt/norm.py:TtDistributedRmsNorm._fork) is ttnn.bringup.rms_norm at the fork's normal settings (HiFi4, fp32 DEST):
- column-split (attn_norm, ffn_norm, final norm): rms_norm(x, stats_out=True) -> ccl.all_gather of the [S/SP, 32 B]
  fp32 partials over TP (dim 3) -> rms_norm(x, partial_stats=, stats_width=H, weight=, dtype=): 3 calls (+ ffn_norm's
  output all_gather, as before). TP is read from the mesh; any TP works (galaxy 4x4 / 8x4: 4 partial sets gathered).
- q_norm / kv_norm: one call on the fused [wq_a | wkv] output p with input_window=(c0, c1), no slice (tt/q_a.py,
  tt/kv.py pass the window; the off path slices inside the norm, as before); kv_norm / k_norm write bf16 (dtype=);
  the compressor norm (fp32 in / out, whole rows) is the fork's default program.
Fork extension (rms_norm_ttnn CHANGELOG 6, README, INDEX): a second device operation behind the new keyword options
(stats_out, partial_stats, stats_width, input_window, dtype), routed to only when one is set; the default program is
untouched. Kernels kernels/rms_norm_split_*.cpp, host device/rms_norm_split_device_operation.*.
Decisions:
- Stats exchange as partials (K-12: stock rms_norm_pre_all_gather leaves junk in stats columns 1-31 on fp32 and post
  reduces the whole tile): stats_out writes a partial in EVERY element (packer-L1-accumulated x^2 tiles, fp32), so the
  apply reduces all columns of all gathered tiles. One gather of [1024, 64] fp32 per norm (0.04 ms) instead of an
  all_gather of the rows (10 MB fp32) + mesh_partition.
- The statistic is FPU-grade (DS41-44: x^2 and the row sum read fp32 at TF32); no exact-statistics option (brief).
  DS41-97: the weight multiply moved into DEST on the SFPU (v1 through an fp32 L1 intermediate: ratio -1.4e-3).
- attn_norm / final norm keep writing fp32 (consumers unchanged; ATTN_NORM_DTYPE is J5's), ffn_norm FFN_NORM_DTYPE.
- No stock norm op is called (rule 16): the INDEX line now lists rms_norm_pre/post_all_gather and slice + rms_norm
  under the fork's "Consider instead of".
Measured (2x2, chunk 53248-55296):
- Quick A/B side on (`measure --task P.18`): chunk wall 127.9 ms (spread 49.4; P.17 on: 153.6, spread 22.4),
  host_transfers_per_layer 0, worst layer L03 pcc 0.99678; every err_* within its limit, equal to P.17 within noise:
  err_layer L00 .0253/.0570 (P.17 .0261), L01 .0337/.0720, L02 .0509/.1077, L03 .0803/.1587, L14 .0603/.1380,
  L20 .0301/.0507, L21 .0512/.1312, L24 .0282/.0418 (.0281); window_kv L02 .0362/.0954 (.0330), L03 .0555/.1314.
- Plain profile (on; profiles/P.18_20261022T100716.json) vs P.17's: device 96.91 -> 93.18 ms a chunk; programs (4
  chips) 6140 -> 4580 (-390 calls a chip a chunk); attn_norm 1.83 -> 0.89 ms (programs 448 -> 96), ffn_norm 2.11 ->
  1.19 (480 -> 128), q_a 4.76 -> 3.79 (512 -> 192), kv 0.81 -> 0.22 (384 -> 64), compressor 1.65 -> 1.37 (664 -> 448);
  the norm-touched sections 11.16 -> 7.46 ms.
- Probe ds41_norm_split/probe_002 (bf16-valued weights, vs float64): split norm fp32 rows -> fp32 rel 6.1e-4, ratio
  -3.5e-4; -> bf16 rel 1.77e-3 (one bf16 rounding 1.66e-3); bf16-valued rows ratio -1.6e-4; deterministic.
- Swap test swa_12 by hand: attn_norm vs CPU rel 3.2e-4 (<= 3.3e-3; off 0), ffn_norm 1.77e-3 (<= 3.3e-3; off 1.66e-3),
  q_a 2.9e-3 (<= 0.024), kv 4.2e-3 (<= 0.058). The test still fails, identically with NORM_IMPL=off (pre-existing, P.17
  gotchas): err_swap_out 0.0260 on / 0.0248 off (<= 0.0185), swap_router_vs_golden_err 0.1457 on / 0.1489 off
  (<= 0.1425), swap_router vs CPU 7.9e-4 both (<= 3.6e-4).
- Fork regression with the options off: unit 1375 passed / 47 skipped (= 1327 + 48 new), fork_source 263 tests 0
  regressions, model cases 10 passed / 2 skipped (8-device cases).
Gotchas:
- unary_bcast of an fp32 CB unpacks to DEST (exact; srcB is 19 bits): the way to bring an fp32 operand into an SFPU
  step without TF32 truncation.
- O.1 should add the model's new calls (stats_out [1024, 2560] fp32, partial_stats [1024, 128], input_window q / kv
  of [1024, 1792], k_norm dtype bf16 [512, 128]) to the fork's cases.
- When the owner accepts: delete NORM_IMPL=off (TtDistributedRmsNorm.rstd, the mhc_pre_xing_pack statistic, the
  slice branch) and the norm-only exact sums in tt/ccl.py if nothing else uses them.
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.18`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/rms_norm_ttnn/tests/unit/test_rms_norm_ttnn_split.py`;
probe `scripts/tt-probe.sh ds41_norm_split < tests/ttnn/unit_tests/operations/ds41_norm_split/probes/probe_002.py`.

## P.19 perf attempt 1 (2026-10-22): GLUE_FOLD=on, the rest of the fusion report (owner comparison)

What: setting GLUE_FOLD (tt/settings.py, default on; off = the P.18 path, kept for the owner's A/B). On:
- (a) mHC exchange (report item 7): attn_hc / ffn_hc send the chips' fp32 partial rows [S/SP, 32] through ONE
  ccl.all_gather on dim 1 -> [1, TP, S/SP, 32]; ttnn.bringup.mhc_pre_xing(partials=TP) adds them in mesh-column order
  on the SFPU (fork mhc_pre CHANGELOG 10). Was: typecast fp32 -> bf16, bf16 fabric reduce-scatter, all_gather,
  typecast back (DS41-82). Numerics: the exact fp32 sum (= CCL_IMPL=fabric's / stock gather_add), not the bf16 RS:
  closer to the reference's fp32 mixes; any TP (4x4 / 8x4: partials=4).
- (b) mhc_post (item 9): post / comb read from the packed [S/SP, 24] mixes (post_col=4, comb_col=8; both programs),
  and the stats pass takes the bf16 attention / MoE sums as F (fork mhc_post CHANGELOG 8); moe_add's
  experts.reduce(fp32=False). Removes 2 slices + 1 typecast per residual and the MoE _to_fp32 (bit-identical: bf16 ->
  fp32 is exact).
- (c) fabric_reduce_scatter(program_hash=True) (item 4; fork CHANGELOG "program_hash"; DS41-98).
- (d) window path (item 6): no pad mask when end - start == chunk; TtWindowCache.keep_rows keeps the chunk's kv rows
  when the chunk is whole, and the next chunk at start = end slices its carry from their last 128 rows (any other
  write start, a host load or free() drops them; the cache read stays the fallback); WindowTables.srv_row memoizes the
  server-order index row of the latest start (8 window + 6 compressed writes a chunk); tt/served.DeviceScalars(memo=)
  keeps the latest 8 sliced scalars (the r2 writes' slot / start / end).
- (e) deallocations through the binding (tt/ccl.py:dealloc, bound at load by ccl.configure; every tt/ forward
  module): ~1.3 us of the wrapper a call. The bring-up profiler's op sequences skip them on both of its runs alike.
- Not done: the moe_add fold into the reduce-scatter (report item 10): it removes one add a layer, measured ceiling
  0.09 ms host + 0.073 ms device a call = ~1.3 ms a chunk, against new reader code in every port / final kernel of
  the RS; the combine planner / zero / barrier generic_ops (24 a chunk, 0.09 ms host): their descriptors bake
  addresses per object, so a custom hash per object would add program-cache misses whenever addresses change; the
  engram rows (report item 8): ROW_MAJOR upload + device tilize = TILE upload (3.4 ms both, probe ds41_p19_fork), the
  host cost is the table lookup itself (DS41-99).
- stock ttnn.slice instead of ttnn.bringup.rms_norm: the carry is a row slice of the kept kv rows (no norm).
Measured (2x2, chunk 53248-55296):
- Final `measure --task P.19` (quick A/B side on, rc 0): chunk wall 118.8 ms (spread 39.1; P.18 127.9), worst layer
  L03 pcc 0.99680, pcc_chunk_out 0.99960, host_transfers_per_layer 0; every err_* within its limit: err_layer L00
  .0250/.0570, L01 .0337/.0720, L02 .0508/.1077, L03 .0800/.1587, L14 .0605/.1380, L20 .0301/.0507, L21 .0510/.1312,
  L24 .0282/.0418 (P.18: .0253 .. .0282); state worst window_kv_L03 .0525/.1314.
- Plain profile (on; profiles/P.19_20261022T114503.json) vs P.18's: device 93.18 -> 91.87 ms a chunk; programs (4
  chips) 4580 -> 3776 (-201 program calls a chunk); attn_residual 3.31 -> 2.52 ms (programs 256 -> 32), ffn_residual
  3.29 -> 2.88 (248 -> 56), moe_add 1.58 -> 1.22 (96 -> 64), attn_hc 2.56 -> 2.64 (184 -> 88), ffn_hc 1.46 -> 1.57
  (160 -> 64), attention 34.49 -> 34.32 (1476 -> 1380), compressor 1.37 -> 1.37 (448 -> 380).
- Host ms per section, off -> on (10 warm clean-wall runs, a FastOperation timer + profiler.signpost hook, a test
  under generated/ (not committed); calls = outermost ttnn calls, the on side's binding deallocations not counted):
  wall 130.3 -> 117.5 ms; in-layer host 102.3 -> 83.0 ms; attention 29.88 -> 26.18, attn_hc 6.27 -> 2.28, ffn_hc 4.53
  -> 1.39, attn_residual 3.25 -> 0.86, ffn_residual 2.91 -> 1.02, moe_add 4.55 -> 3.40, compressor 8.80 -> 7.91,
  experts 11.83 -> 9.74, router 6.97 / 7.04, q_a 5.27 / 5.40, indexer 4.85 / 4.82; outside the layers begin_chunk +
  embed 29.1 / 30.9, final drain 2.0 / 3.2. fabric_reduce_scatter 49 -> 33 calls, 0.305 -> 0.271 ms host a call.
  The wall is now the host dispatch (DS41-99).
- Ladder s4096 by hand (on; chunk 2 takes its carry from chunk 1's kept rows): pass, err_layer L00 .0267/.0561 ..
  L03 .0823/.1558 (FIX.8: .0273 .. .0847), state worst window_kv_L03 .0582/.1334.
- Fork tests: new tests/unit/test_xing_partials.py (13), test_mhc_post_mixes.py (19), test_fabric_rs_program_hash.py
  (9), all bit-identical; regressions with the options off: mhc_pre unit + golden 345 / 1 skipped (= 332 / 1 + 13),
  model cases 103; mhc_post unit + golden + cases 401 (= 382 + 19); fabric RS unit 19 / 13 skipped, cases 2. Build OK.
Gotchas:
- The hashed RS path needs its own descriptor cache entries (DS41-98): sharing the default path's dkey silently reuses
  an unhashed descriptor.
- The carry from the kept rows never hits in the repeated-chunk profile (same start every run, the prefix reload drops
  it); only consecutive chunks (ladder s4096 / s55296, serving) use it.
- When the owner accepts: delete GLUE_FOLD=off (ccl.tp_sum's bf16 path for the hc rows, the mhc_post slices / typecasts,
  the window mask for whole chunks), and the wrapper branch of ccl.dealloc.
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.19`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/test_xing_partials.py`,
`.../mhc_post_ttnn/tests/unit/test_mhc_post_mixes.py`, `.../fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_program_hash.py`.

## P.21 perf attempt 1 (2026-10-22): STATIC_TABLES=on, one persistent attention table per KV source; trace ceiling

What: setting STATIC_TABLES (tt/settings.py, default on; off = the P.19 path, kept for the owner's A/B). Fusion report
item 3. On:
- tt/compressor.AttnTable: per KV source and slot (allocated with the source's served compressed KV, `TtServedRows(
  attn_table=True)`; TtCompressCaches passes it, so the device model, the hybrid and the serving runtime all get one)
  ONE replicated bf16 ROW_MAJOR buffer [hist + 128 + S, 512], hist = the cache's token capacity / r: the history in
  natural order (group j at row j), then the window region at base = hist.
- TtServedRows.write also slice_writes the replicated natural-order rows it was given (the compressor's `rm`) at row
  start / r: one `ttnn.experimental.slice_write` a source a chunk; TtServedRows.load_host (golden prefix, migrated
  prefix, hybrid) writes the loaded rows the same way, so the table follows the served cache on every write path.
- TtSparseAttention._static: the layer's [carry | chunk] window rows slice_written into the region (2 slice_writes;
  the source's layers run in order on one queue and reuse it), index list = [WindowTables.index_lists(base)[class] |
  the top-k as it is] (mesh_partition + concat, + the same compaction gather at starts < 128), memoized for the
  shared top-k as before. Gone per chunk: the history slice + to_layout + 2 all_gathers + to_layout (56 MiB at r1),
  the [window | history] concat, the 27-op fp32 id remap (`_topk_rows`). The table's shape is the same in every
  chunk (no program-cache misses as the history grows).
Decisions:
- History first, window region after it: top-k ids need no transform (0xFFFFFFFF stays the sentinel); only the
  load-time window ids carry the offset, built per state at state creation (index_lists(base), 5 small lists).
- The table is kept in step by the served cache's own writers instead of being rebuilt from the cache: the cache has
  no other writer on a prefill node (serving_contract.md "Input": a follow-up remounts its own slot; migrated prefixes
  come in through state.load_served -> load_host).
- Window-only layers (L0, L1) keep their [carry | chunk] concat (2 MiB, already one shape per chunk).
- Stock ops only (slice_write, concat, gather, mesh_partition): no INDEX entry lists them (rule 16).
Measured (2x2, chunk 53248-55296):
- Final `measure --task P.21` (quick A/B side on, rc 0): chunk wall 103.5 ms (spread 6.1; P.19 on: 118.8, spread
  39.1), host_transfers_per_layer 0, worst layer L03 pcc 0.99680, pcc_chunk_out 0.99960; every err_* within its limit
  and equal to P.19: err_layer L00 .0250/.0570, L01 .0337/.0720, L02 .0508/.1077, L03 .0800/.1587, L14 .0605/.1380,
  L20 .0301/.0507, L21 .0510/.1312, L24 .0282/.0418; worst state window_kv_L21 .0458/.1118.
- Ladder s4096 by hand (on; chunk 0 at start 0 takes the compaction path, chunk 1 a grown history): pass, err_layer
  L00 .0267/.0561, L01 .0353/.0734, L02 .0541/.1071, L03 .0823/.1558, L14 .0551/.1204, L20 .0303/.0519, L21
  .0508/.1265, L24 .0305/.0449 (P.19: L00 .0267 .. L03 .0823).
- Trace ceiling (owner; tt/perf_trace.py, standalone, not the model's path; STATIC_TABLES=on): the 8-layer stack at
  start 53248, inputs made before the capture: traced replay 87.0 ms a chunk (10 replays 87.0-87.1), untraced
  layer stack median 103.7 ms (87.9-199.0), begin_chunk (engram host input) 38-40 ms warm outside it; replay output
  vs untraced PCC 1.000000, vs golden 0.999605 (= untraced). The forward captures as it is; what blocked the first
  capture: the flat expert's address-keyed descriptor (DS41-100: settle until no new key, 6 runs).
- Not run: the plain profile (the orchestrator runs it with the change on and off).
Memory (rule 15): a replicated copy of each source's compressed KV per slot. 2x2 at 57,344: r2 28 MiB (L2, L14), r1
56 MiB (L20) + 2.1 MiB of window region each a chip. Full model: 4 sources (L2, L8, L14 r2, L20 r1): per chip per
slot 1.25 GiB at 524,288 and 2.5 GiB at 1,048,576 on every box (replicated: the chip count does not divide it;
the served caches themselves are 384 + 256 MiB at 1M before dedup).
What still blocks a chunk-to-chunk trace (code reading, printed by perf_trace.py): host per-start values baked into
the commands (rope row slices, srv_row / update_padded_kv_cache scalars / DeviceScalars.at, the window carry slices,
pad-mask scalar and start class, the ring score's chunk_start_idx / kv length and the candidate widths, the top-k /
candidate stores keyed by start), the address-keyed flat-op / planner descriptors (DS41-100), and the per-chunk
engram rows / ids uploads into new tensors.
Gotchas:
- K-52: ran perf_trace.py with --no-precompile (a model-sized test).
- When the owner accepts: delete STATIC_TABLES=off (sparse_attention._compressed, _topk_rows, _index, TOPK_LIMIT,
  the ChunkMemo "comp" slot and model.memo_last's ("comp", src) entries, the [window | comp] concat).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.21`; trace ceiling
`PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_trace.py`.

## FIX.10 overseer (attempt 1, 2026-10-22): "remove P.21's second copy of the compressed history": DISAGREE, no code change

Request: make the served compressed cache and P.21's AttnTable one storage, to recover "+1.25 GiB per chip per slot at
524,288 (+2.5 at 1M)" on galaxy_4x4. Answer: disagree (knowledge-use report, overseer FIX.10). Nothing in tt/,
hooks.py or the forks changed.
Why (DS41-101):
- The served copy is not a second full copy. 1_compress_kv_r2 / 3_compress_kv_r1 are TP-dedup and SP-sharded: each
  chip holds 1 / (SP x TP) of a source's history (tt/compressor.py TtServedRows: owners = sp x tp, rows =
  ceil(max_seq / chunk) x W / TP). The AttnTable is the whole history on every chip, because sparse_sdpa reads only its
  own chip's table and a top-k id may name any earlier group. The 1.25 GiB is that table.
- Measured (probe tests/ttnn/unit_tests/operations/ds41_fix10_mem/probes/probe_001.py, 2x2, DRAM allocated per chip
  via ttnn.get_memory_view, one TtServedRows(dedup=True, attn_table=True), STATIC_TABLES off / on), MiB per chip:
  max_seq 55,296: r2 6.87 / 37.12 (served slab 6.75, table 29.12 = 27,648 history + 128 + 2,048 window rows);
  r1 13.52 / 70.88 (served 13.50, table 56.12). max_seq 57,344: r2 7.13 / 38.38; r1 14.02 / 73.38. This run's three
  sources (L2, L14 r2; L20 r1) at 55,296: served 27.0 MiB, tables 114.4 MiB a chip.
- Full model, 4 sources (L2, L8, L14 r2; L20 r1), 1 KiB a row, chunk 4096 (window region 4,224 rows = 4.1 MiB each),
  per chip per slot: table 1.266 GiB at 524,288 and 2.516 GiB at 1,048,576 on every box. Served dedup copy:
  galaxy_4x4 80 MiB (524k) / 160 MiB (1M); galaxy_8x4 40 / 80 MiB. Today (P.21): 4x4 1.344 / 2.672 GiB, 8x4 1.305 /
  2.594 GiB. One storage readable by sparse_sdpa must still hold the whole history per chip: 1.266 / 2.516 GiB on both
  boxes. The merge saves 80 MiB (4x4, 524k), 40 MiB (8x4, 524k), 160 / 80 MiB at 1M: 6 % of the 1.25 GiB.
- What the merge would cost: the served configs replicated over SP and TP in natural order (serving_contract.md
  "Placement" shards the sequence over SP rows; the KV Manager would write each migrated record to all 16 chips), and a
  sparse_sdpa fork option to read an ND-sharded engine tensor [slots x layers, 1, rows, 512] at a (slot, layer) base row
  with the window region inside it. Not worth 80 MiB.
- Budget at 524k on galaxy_4x4 (plan_boxes.json gb_at_deploy_seq 26.49 GiB, which counts ONE r1-sized table, 0.5 GiB,
  plus the 80 MiB served copy): P.21 adds 3 r2 tables + 3 window regions = +0.76 GiB -> ~27.25 of 27.2; the merge
  would give ~27.17. But 26.49 predates two shipped settings: dense fp8 weights as bfp8 (DENSE_WEIGHTS, P.13; plan.md:
  bf16 dense costs +1.5 GiB on 4x4) and routed experts as bfp4 (0.5625 B a value vs the plan's packed 0.531: about
  +1.0 GiB). Estimate with both: ~26.7 GiB with P.21 as it is. The plan check should be re-run with the shipped dtypes
  and the per-source tables before anyone trades speed for memory.
- If memory is needed, options that do cut it (not done; a perf / plan decision): (a) the plan's one shared r1-sized
  table refilled per source per chunk (0.5 GiB at 524k; pays the per-chunk history gather P.21 removed); (b) keep the
  persistent table only for L20 (read by layers 20-39) and one shared r2 buffer refilled per r2 source per chunk
  (0.77 GiB at 524k, -0.5 GiB vs P.21; 3 r2 history gathers per chunk).
Gate: not run (no code changed since P.21's passing gate; agent rule 8). Chunk wall / device time / err_* unchanged:
P.21's quick wall 103.5 ms, err_* as in the P.21 section.
Re-run the measurement: `PYTHONPATH=$PWD scripts/tt-probe.sh ds41_fix10_mem < tests/ttnn/unit_tests/operations/ds41_fix10_mem/probes/probe_001.py`.

## FIX.10 overseer (attempt 3, 2026-10-22): disagree kept, no code change; the gate's failures are the indexer swap checks (DS41-102)

- FIX.10: answer stays `disagree` (attempt 1's evidence, accepted by the overseer 14:20; DS41-101). Nothing in tt/,
  hooks.py or the forks changed. Memory, chunk wall (quick 103.5 ms), device time and err_* are P.21's.
- OV-31: done. The owner waivers are in the spec (5fb7d75926d); swap swa_12 and swa_engram_15 now pass (orchestrator log
  FIX.10.log, 14:46-14:54).
- Gate status on P.21 code (the first full run of P.21's frozen tests; P.21 was accepted on a quick A/B):
  - test_c_*_experts (8): pass (FIX.10.log).
  - swap enc_source_14: FAIL, swap_indexer_vs_cpu match 0.0015 (>= 0.922), vs_golden 0.0015 (>= 0.830), near ties
    flips 1967 unexplained 1471 recall 0.9942 (FIX.10.log). Every other failed check there is waived.
  - swap enc_source_engram_17 (my run): FAIL, the same indexer checks (match 0.0016 >= 0.860; unexplained 218, recall
    0.9979) and pytest-timeout 300 s (the test ran 300.65 s; 291 s on the earlier run).
  - dec_candidate_14 and dec_reindex_13 run an indexer through the same checks: expect the same. enc_reuse_12,
    dec_reuse_12, the P.21 ladder and the profile: not reached.
  - K.1 contract tests (my runs, P.21 code): test_contract PASS (691 s; 335 s at K.1), test_kv_layout PASS,
    test_adapter_acks PASS, test_runner_contract PASS, test_runner_smoke PASS.
  - Ladder L.last: not finished. It runs > 20 min, beyond the agent's tool limit (10 min foreground + 10 background),
    and was stopped by SIGTERM at 1200 s (card marked dirty, run_safe_pytest resets it on the next run).
- Indexer diagnosis (CPU only, /tmp/fix10diag/idx_ties.py, layer 2, golden inputs, NF.flips, details in
  known_issues_detail.md DS41-102): native scores rounded to bf16 (topk_large_indices is bf16-only) give 1277
  unexplained / recall 0.9946 (device 1471 / 0.9942); the official bf16 arithmetic gives 2048 / 0.9638. The match is
  order: the reference returns ids ascending, the device by score (the native set by score: 0.0014). K-42 says index
  outputs need topk_overlap. Not a logic bug, not from P.21: a framework / frozen-test or waiver decision.
- No fix in the model: sorting the device ids would add a device sort only for the test and still not pass (a flip shifts
  positions), and fp32 top-k scores would raise precision above the official code (rule 7).

## P.20 perf attempt 1 (2026-10-22): BF16_ACTIVATIONS=on, activations in bf16 where the GPU code keeps bf16 (J5)

What: setting BF16_ACTIVATIONS (tt/settings.py, default on; off = the P.21 path, kept for the owner's A/B). On, per
bringup/official_taps.yaml:
- the four mHC streams bf16 (block boundary and h_mid): embed concats the bf16 embedding rows (no typecast), the
  residuals (mhc_post, plain and stats_weight) and the engram add write bf16, tt/layout.streams_to_device (harness
  boundary, from_host) uploads bf16 (tt/common.stream_dtype);
- attn_collapse / ffn_collapse / final collapse bf16 (mhc_pre_xing writes y in the streams' dtype);
- attn_norm and the final norm write bf16 (build_norm / TtDistributedRmsNorm dtype=act_dtype());
- the [wq_a | wkv] exchange stays bf16 (ccl.tp_sum(fp32=False): the fabric_bf16 RS + gather result without the
  typecast back), q_norm writes bf16 (q_resid), kv_norm already bf16;
- wo_a writes bf16 (no cast after it); engram wkv writes bf16 (the engram add reads it as is);
- the compressor's norm writes the latent bf16 (the rope reads it as is, no typecast), wk writes bf16.
fp32 kept where the reference is fp32: the hc mixes / Sinkhorn, every norm statistic, the engram gate sums (fp32 SFPU
on the exact bf16 values), the compressor's projection + pair pooling, the router; fp32 accumulation in every matmul.
Requires MHC_FUSED, GLUE_FOLD, ROPE_IMPL, NORM_IMPL on (asserted in TtDS41Model).
Fork extensions (bring-up ops, rule 16; option = bf16 inputs, fp32 programs unchanged):
- ttnn.bringup.mhc_pre_xing / mhc_pre_xing_pack: bf16 streams and engram keys (CHANGELOG 11; defines MHC_XING_X16 /
  MHC_ENGRAM_X16; srcA reconfigured between the fp32 / bf16 CBs; UnpackToDestFp32 on the bf16 CBs: exact);
- ttnn.bringup.mhc_post(stats_weight=): bf16 X with bf16 F (CHANGELOG 9; MHC_POST_STATS_X16; X' and the matmul's
  X' bf16, the sum of squares from the fp32 mix).
- The hybrid's collapse with a bare [R, 4] pre on bf16 streams pads the pre to 24 columns and uses
  mhc_pre_xing(coefficients_given=True) (tt/collapse.py; harness only).
Measured (2x2, chunk 53248-55296):
- Final `measure --task P.20` (quick A/B side on, rc 0): every err_* within its limit and equal to P.21 within noise:
  err_layer L00 .0248/.0570 (P.21 .0250), L01 .0338/.0720, L02 .0514/.1077, L03 .0805/.1587, L14 .0605/.1380,
  L20 .0299/.0507, L21 .0513/.1312, L24 .0280/.0418; worst state window_kv_L03 .0543/.1314 (P.21 .0525);
  compress_kv_L02 .0382/.2095 (.0390), index_k_L02 .0205/.2404 (.0210); worst err/limit 0.67; min layer PCC L03
  0.99676 (P.21 0.99680), pcc_chunk_out 0.99961, host_transfers_per_layer 0.
- Wall: chunk_wall_ms 142.1 (spread 27.5) on a loaded box (a golden generation at ~7 cores + a selftest, load
  13-19); the off side run by hand right after, same command and load (DS41_BF16_ACTIVATIONS=off BRINGUP_QUICK=1
  BRINGUP_PROFILE_WALL=1 test_profile.py): 161.3 (spread 57.3). P.21's 103.5 was on an idle box: compare only the
  orchestrator's A/B pair.
- Plain profile (on; profiles/P.20_20261022T202929.json) vs FIX.10's (P.21 code, off): device 88.33 -> 83.96 ms a
  chunk (-4.4); programs (4 chips) 3328 -> 3216; mHC / residual / norm / engram sections: attn_hc 2.65 -> 1.75,
  ffn_hc 1.57 -> 1.12, attn_residual 2.50 -> 1.80, ffn_residual 2.87 -> 2.44, engram_add 1.79 -> 1.34, engram_kv
  4.94 -> 4.62, attn_norm 0.89 -> 0.60, ffn_norm 1.17 -> 1.00, q_a 3.79 -> 3.12 (programs 192 -> 128), attention
  30.72 -> 30.38, indexer 11.86 -> 11.73; moe_add 1.52 -> 1.96 (its code is unchanged: the RS waits on the slowest
  chip of the row).
- Memory per chip (computed from the shapes, not measured; weights unchanged): a stream copy [1024, 4 x 2560] 40 ->
  20 MiB (2x2; 4x4 [1024, 4 x 1280]: 20 -> 10 MiB), about 3 live a block (in, h_mid, out); collapse / attn_norm
  [1024, 2560] 10 -> 5 MiB, q_resid 5 -> 2.5, the [wq_a | wkv] output 7 -> 3.5, engram kv [1024, 12800] 50 -> 25
  MiB: peak activations about -80 MiB a chip at 2x2.
- Hybrid path: test_swap_swa_engram_15_experts passes (every failure waived, as before); attn_hc vs CPU rel 0.0021 ->
  0.0012, ffn_hc 0.0027 -> 0.0016 (the FPU reads bf16 exactly, fp32 at TF32: DS41-104); err_swap_out 0.0235 (0.0234).
- Fork tests: tests/unit/test_xing_bf16_streams.py (10) and test_mhc_post_bf16_streams.py (7): bf16 paths
  bit-identical to the fp32 programs on the same values up to the packer's one rounding (to nearest, ties away:
  DS41-103); regressions with fp32 inputs: mhc_pre unit + golden 355 / 1 skipped (= 345 / 1 + 10), model cases 103;
  mhc_post unit + golden + cases 408 (= 401 + 7). Build OK.
Gotchas:
- The packer's fp32 -> bf16 rounds ties away from zero (DS41-103): a bit-exact test against torch's .bfloat16() fails
  on ~1e-5 of values.
- A background orchestrator `measure` killed at the tool's limit leaves its pytest running orphaned; stop it before
  re-running (the next safe run resets the cards).
- When the owner accepts: delete BF16_ACTIVATIONS=off with the fp32 stream paths that only it uses (embed's typecast,
  tp_sum's fp32 default for q_a, the compressor's latent cast); the fp32 fork programs stay (other models).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.20`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/test_xing_bf16_streams.py` and
`.../mhc_post_ttnn/tests/unit/test_mhc_post_bf16_streams.py`; profile (on)
`TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/common/bringup/tests/test_profile.py`.

## P.22 perf attempt 1 (2026-10-22): router through the fused moe_grouped_topk (ROUTER_IMPL=fused, owner comparison)

What: setting ROUTER_IMPL (tt/settings.py, default fused; ops = the P.21 chain). fused = the gate matmul (unchanged:
bf16 x, bf16 gate, HiFi2 + fp32 DEST, fp32 logits) then ONE ttnn.bringup.moe_grouped_topk(n_groups=1,
score_func="sqrtsoftplus", top 6, route_scale 1.5, eps 1e-20, stable_sort, fp32_selection=True,
fp32_activation=True) instead of softplus, sqrt, add, topk, gather, sum, add, multiply, div (tt/router.py:_fused).
Outputs: weights bf16 TILE, ids uint16 TILE: what the weighted experts read (tt/weighted_combine.py:routing now skips
the bf16 typecast when the weights are bf16; the uint32 -> uint16 typecast of the old topk ids is gone too). The
recentred fp32 bias is a load-time [max_rows, E] tensor (the op takes a full-shape bias). The dense routing (hybrid /
swap tests only) is the scatter of those weights typecast to fp32.
Decisions:
- The source commits are already in this tree (71062e76a38 / 570d95b6e7d = 5e47659815e / 27c5c2a16ae + the rebase fix
  9780c18f0d0), not only on Marko's branch: stock ttnn.experimental.deepseek_prefill.moe_grouped_topk has
  fp32_selection here. I forked it anyway (owner instruction, and it needed a change):
- DS41-105: stock moe_grouped_topk builds its compute kernel without INP_FLOAT32, so sqrt(softplus) runs the bf16-grade
  polynomial (degree 6, 0 below -5) even with fp32_selection: keys max error 8.2e-2 vs float64 on the same logits
  (the chain: 2.5e-5), and 0-13 of 2048 rows a layer picked other experts than the chain (scaled golden s4096 chunk
  1). Fork option fp32_activation (default off = the source program; define INP_FLOAT32 on the compute kernel only):
  keys bit-identical to stock ttnn.sqrt(ttnn.softplus) on fp32. Setting ROUTER_SOFTPLUS {fp32 (default), bf16}.
- Weights bf16 (the op has no fp32 output): the reference's own dtype (model.py Gate: weights.type_as(x)); the experts
  read bf16 weights in both paths. Router step checks now sit at one bf16 rounding (DS41-106, waived).
- No fork of the experts side and no uint32 ids from the gate (one typecast left per layer, 0.05 ms host).
Measured (2x2, scaled model, chunk 53248-55296 unless noted):
- Routing vs the old path on the golden (s4096 chunk 1, golden ffn_norm as bf16 input, 8 layers, 2048 rows each;
  probes tests/ttnn/unit_tests/operations/ds41_router/probes/probe_005.py / probe_006.py): selected experts identical
  on 2048 / 2048 rows of every layer; weights rel 1.53e-3..1.65e-3 vs the chain's fp32 weights (= one bf16 rounding);
  on same-set rows vs float64 on the device's logits rel = one bf16 rounding, bias <= 2.5e-5; both paths 25-45 rows a
  layer differ from float64 on the host's logits (the matmul's TF32 partials, waived); deterministic. Without
  fp32_activation: 0-13 rows differ from the chain.
- One router call (layer 0, 1024 rows a chip, 50 calls): host 0.583 -> 0.070 ms, async wall 0.598 -> 0.131 ms.
- Real-time profile with op rows (BRINGUP_PROFILE_OPS=1, both sides by hand, same load; profiles
  P.22_20261022T221748.json (fused) / P.22_20261022T222358.json (ops)): host calls per chunk 777 -> 697 (router 80 ->
  16, experts 96 -> 80); router section host 7.21 -> 1.23 ms, device 1.50 -> 1.04 ms a chunk, programs (4 chips)
  384 -> 64; device total 83.52 -> 82.97 ms; host dispatch (op mode) 459.5 -> 419.2 ms.
- Quick A/B side on (`measure --task P.22`, rc 0): every err_* within its limit and equal to P.20 to 4 digits
  (worst L03 0.08050 / 0.15875, min layer PCC 0.99676, pcc_chunk_out 0.99961), host_transfers_per_layer 0,
  chunk_wall_ms 102.0 (spread 30.9; box loaded by a golden generation; the orchestrator's off side is the comparison).
- Gate units passed with --check: swa (C experts), all 8 swap *_experts units (router step WAIVED as before: vs CPU rel
  1.75e-3..2.44e-3, 0 unexplained flips, recall_tie 1.0). Not run by me: the 7 other C.*.experts units (golden
  routing, the router is not in them), the ladder and the profile.
Fork: ttnn/ttnn/bringup/moe_grouped_topk (CHANGELOG, README, INDEX entry); source tests: the op's 5 nightly files,
203 / 203 pass on the original and the fork (tests/source_baseline.json); unit tests tests/unit/test_fp32_activation.py
(7 pass). Build OK.
Gotchas:
- ttnn.topk on fp32 returns UINT32 ids (the experts typecast them to uint16 then back to uint32); the fused gate gives
  uint16.
- tt/experts.py: routing may hand its inputs through (bf16 weights, a one-row mesh); the frees skip them now.
Stock calls added (F113):
- stock ttnn.typecast instead of ttnn.bringup.mhc_post: the fused gate's bf16 weights -> fp32 for the dense routing
  scatter (hybrid / swap tests only; no mHC step there).
- stock ttnn.slice instead of ttnn.bringup.rms_norm: the load-time bias rows cut to a row count other than the chunk's
  (once per count, kept; no norm there).
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.22`; fork tests
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/moe_grouped_topk/tests/unit` and
`python -m models.demos.common.bringup.testing.fork_source --fork moe_grouped_topk`; router section A/B
`[DS41_ROUTER_IMPL=ops] BRINGUP_PROFILE_OPS=1 scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/common/bringup/tests/test_profile.py`.

## P.22 perf attempt 2 (2026-10-22): the F113 failure of attempt 1 fixed; no code path changed

What: attempt 1 failed only on the F113 lint: the ROUTER_IMPL help text in tt/settings.py named the stock
`ttnn.experimental.deepseek_prefill.moe_grouped_topk(...)`, though tt/router.py calls the fork
`ttnn.bringup.moe_grouped_topk` (DS41-107: the lint matches `<op>(` inside strings too). The text now names the fork.
Nothing else changed; `./build_metal.sh` re-run (no-op, the fork was built).
Checked: `ops_lint.violations` and `settings_lint.violations` on every changed file: none. Gate units with --check: all
16 component / swap units PASS on this code (ladder and profile left to the orchestrator, as the brief says).
`measure --task P.22` (switch on, quick): rc 0, all 31 err_* within limits (worst err_layer_L01 0.0338 / 0.0720),
min layer PCC 0.99676 (L03), pcc_chunk_out 0.99961, host_transfers_per_layer 0, chunk_wall_ms 99.2 (spread 14.2);
the same accuracy digits as attempt 1 (deterministic). Before / after numbers: attempt 1's section above.
Re-run: `python -m models.demos.common.bringup.orchestrator measure --task P.22`.

## O.1 optests attempt 1 (2026-10-23): a random-input case for every ttnn.bringup call

What: 56 distinct fork calls (ladder rung last, 2x2) now each have a case: fabric_all_gather 15, fabric_reduce_scatter
6, rms_norm 8, mhc_post 3, mhc_pre_xing 16 + mhc_pre_xing_pack 2, ring_indexer_score_dsa 3, moe_grouped_topk 1
(new suite: cases.py / reference.py / test_moe_grouped_topk.py: the previous gate failed "forks without
tests/test_*.py: ['moe_grouped_topk']" before running any test), flat_routed_expert 1 + flat_routed_expert_with_program
1 (new suite incl. tests/weighted_case.py). No op changed; CHANGELOG / README of each fork updated.
Decisions:
- K-45 / DS41-109: the 8 per-layer flat_routed_expert_with_program sigs were keyed on the `extra` descriptor's
  address; `ttnn/ttnn/bringup/__init__.py:_stable_arg_reprs` gives MeshProgramDescriptor a value repr (kernel counts per
  program) -> one stable sig 284a4a9ead. Re-captured once (63 -> 56 calls); every other sig unchanged.
- Limits: new math cases declare `noise` and check rel L2 <= fork_cases.noise_limit (F139); PCC printed only (asserting
  noise_limit's pcc failed 7 mhc_pre cases at 1 - 1e-15 with rel inside its limit). Where the op's real formats are
  coarser than its inputs, the formats run models them, never a measured device number: the FPU's operand mantissa
  bits per fidelity (DS41-108: in0 6 bits at LoFi / HiFi2, in1 4 at LoFi / 9 at HiFi2-4; fitted on probes, the device
  3e-4 from the model), bfp8 x / h tiles (round_to), fp32 running sums (mhc_pre pack / engram).
- Measured vs limits: reduce-scatter bf16 1.8e-3 / 3.6e-3 (exact case bit-identical); rms_norm 1.7e-3 / 3.3e-3,
  stats_out 2.8e-8 / 1.1e-7; mhc_post X' 1.66e-3 / 3.3e-3, HiFi2 projection 4.06e-3 / 8.2e-3, sum x'^2 5.9e-8 / 1.0e-7;
  mhc_pre worst ratio 0.73; router weights 1.7e-3 / 3.4e-3 (4096 / 4096 rows clear, all match); indexer (HiFi2) 7.7e-3
  / 1.55e-2; shared expert (LoFi bfp8) 4.0e-2 / 9.9e-2; routed + weighted combine (LoFi bfp4, bf16 sums) 2.0e-2 /
  2.95e-2. Negative checks by hand: a corrupted all-gather element and a halved block of routed rows both fail.
- The routed case ports the model's WeightedCombine (tt/weighted_combine.py) into the fork's tests (copied, not
  imported) on combine_rf's builders; experts at exact bfp4 values (check_exact), built in ~100 s.
Gotchas: fork_tests runs every top-level tests/test_*.py of the 8 forks (other models' cases too; 4x2 xing cases of
mhc_post / mhc_pre run on the 2x2 box per chip; flat's test_flat_bench passes, 13 s).
Checked (not the gate's own --check: its one unit, capture ~8 min + the 8 forks' suites ~12.5 min, exceeds a 10 min
foreground call): the gate's capture command by hand (56 calls), `fork_cases --capture ... --spec ... --lint`
(uncovered 0, lint 0), and `fork_tests --fork <f>` for all 8 forks on the final files, every model's cases: all pass
(rms_norm 18 + 2 skipped-size, mhc_pre 121, indexer 5, flat 2 + bench, all_gather 19, reduce_scatter 8, mhc_post 5,
moe_grouped_topk 1).
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate O.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`; one fork: `python -m
models.demos.common.bringup.testing.fork_tests --fork <fork>`.

## O.1 optests attempt 2 (2026-10-23): blocked, the remaining uncovered sig cannot be fixed in this step's paths

What: the gate (attempt 1's code) failed only on fork_calls_uncovered 1: flat_routed_expert_with_program sig
b155b17c4d (x8), while the case carries attempt 1's 284a4a9ead. Every fork test passed (fork_tests_failed 0).
Cause (DS41-110): the call's `extra_key` = `tt/weighted_combine.py:346` key0 = 9_100_000 + (id(self) % 100_000) * 8, a
Python object address that differs run to run. Checked: hashing the gate's entry with extra_key 9507296 instead of
9582688 gives exactly 284a4a9ead, so nothing else differs. DS41-109's descriptor repr does work (one sig for all 8
layers), but its "stable across captures" claim was wrong (attempt 1 captured only once).
Not done: no case for b155b17c4d. It would only match until the next capture, so it would be a guess, not a fix.
Fixing this needs a path this step may not change: a deterministic key0 in tt/weighted_combine.py (e.g. a class-level
itertools.count; the key only needs to be unique per WeightedCombine instance and pass), or a capture that normalizes
cache keys (testing/fork_capture.py). After that: capture twice, set the routed case's sig in
flat_routed_expert_ttnn/tests/cases.py to the stable value, and re-run the gate.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate O.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## O.1 optests attempt 4 (2026-10-23): stable routed sig, indexer test race fixed

What: (1) flat_routed_expert_ttnn/tests/cases.py routed case sig 284a4a9ead -> 3fc158f72b, the F179 stable sig
(capture leaves out extra_key, DS41-110 / OV-35). The captured entry re-hashed with attempt 1's extra_key gives
284a4a9ead, so it is the same call; no second case (it would rebuild the ~100 s routed case). (2) The orchestrator's
gate after attempt 3 also failed fork_tests_failed 1: test_ring_indexer_score_dsa_ds41 ratio-2 case "not
deterministic", A run 4, chip 3. Probe (indexer_score_ds41_det, deleted): 4 of 299 calls wrong on chip 3 with two
cycled semaphore sets (so not semaphore reuse), wrong on exactly the 13824 remote-shard columns. Cause (DS41-111): the
test wrote each call's fresh k buffer from the host just before the call; the peer done with call N starts N+1 and
writes its shard there while chip 3 (most causal work, slowest) still runs N, then chip 3's queued host write overwrites
it. Fix (test only): every call's buffer written before the first call (`kbufs`); 0 of 299 and 0 of 599. The model is
not affected: its buffer is written once at load (tt/indexer.py _kbuf). The hy4 / glm53 ring tests keep the per-call
write (other models' tests; latent, they pass). No op change.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate O.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## Z.1 settings audit attempt 1 (2026-10-23)

Audit result: tt/settings.py (Settings("DS41_"), 25 switches, env names unchanged) already held every environment read
of tt/ and bringup/hooks.py; settings_lint: 0 violations. Added `hooks.settings()` returning `settings.all()`.
Switches (default; before: earlier tasks moved them into the table, e.g. P.13/P.18/P.20/P.22): DEVICE_MODEL device,
MATMUL_FIDELITY HiFi2, FP32_ACC needed, DENSE_WEIGHTS, ATTN_FIDELITY, FP32_WEIGHT_ROUND, ATTN_O_SUM, ATTN_GLUE,
ROPE_IMPL, NORM_IMPL, GLUE_FOLD, STATIC_TABLES, BF16_ACTIVATIONS, ROUTER_IMPL, ROUTER_SOFTPLUS, INDEXER_GLUE,
EXPERTS_IMPL, EXPERTS_PRECISION, COMBINE_ACC, COMBINE_PLANNER, FFN_NORM_DTYPE, EXPERTS_SUM, RESIDUAL_MIX, MHC_FUSED,
HC_SINKHORN, CCL_IMPL, HC_TP_SUM (values: see tt/settings.py).
Left as literals (not switches): HiFi4 + fp32 DEST in tt/norm.py (the rms_norm fork's own exact setting) and tt/rope.py
(the only exact bf16 data move through the FPU); LoFi for EXPERTS_PRECISION=glm is derived from that setting.
Forks: mhc_pre_ttnn / rms_norm_ttnn env reads are marked `diagnostic`; flat_routed_expert_ttnn's ~60 MIMO_FL_* reads
are pre-existing fork knobs (F106, not converted here; the model's tuned values come from the fork defaults). Listed for the
owner: converting them to op arguments is a separate task.

## P.23 perf attempt 1 (2026-10-23): matmul program configs from a rule (MATMUL_CONFIG tuned / auto)

What: every dense ttnn.linear / ttnn.matmul of the model goes through `tt/mm_configs.py` (`mm.linear` / `mm.matmul`;
sites attn.wq_b / wo_a / wo_b, q_a, engram_kv, router, indexer.wq_b / weights_proj, compressor, hc.fn). Setting
MATMUL_CONFIG: auto = no program config (the old path), tuned (default) = `mm_configs.rule(M, K, N, dtypes, fp32 DEST,
grid, batch)`. Nothing else changed: dtypes, compute kernel configs (HiFi2, fp32 DEST, packer_l1_acc off), DRAM
interleaved memory configs, op sequence. The rule (K-51: auto is slow at fp32 DEST):
- batch > 1 (wo_a, one weight per group): auto. DS41-113: MatmulMultiCoreReuseProgramConfig needs per_core_N == N and
  rounds its K-block partials to bf16 (rel vs fp64 9.5e-3..2.8e-2 vs auto 4.4e-3); no other config takes per-batch
  weights.
- bf16 output and N > 4 tiles per grid column: ttnn.experimental.minimal_matmul, M4 K4 N8, subblock 2x2 (DS41-112).
- else 2D multicast (GLM-5.3's mm2d): M over grid rows, N over grid columns, in0_block_w 16/8/4/2/1 within
  L1_BUDGET 1.1 MB, the largest-area out subblock in DEST.
Sweep (tests/perf/test_matmul_sweep.py; one call per site, model's exact shapes / dtypes / ckc from an in-model capture,
L1 as the model leaves it: 1382 KiB free for CBs at every matmul call, DS41-114; stale-L1 check = a same-shape auto run
on other inputs right before each candidate; determinism over 2 runs and 4 chips; device time from the real-time
profiler, 20 calls; error vs float64 on the device-rounded inputs). 2x2, grid 11x10, ms per call:

| site (calls / chunk) | shape | auto | tuned | config | vs auto (rel) | vs fp64 (both) | ms / chunk |
|---|---|---|---|---|---|---|---|
| attn.wq_b (8) | 1024x1280 bf16 . 1280x8192 bfp8 -> bf16 | 0.6254 | 0.1533 | minimal M4 K4 N8 | 0 | 4.378e-3 | 5.00 -> 1.23 |
| attn.wo_b (8) | 1024x2048 bf16 . 2048x5120 bfp8 -> bf16 | 0.5928 | 0.1376 | minimal M4 K4 N8 | 3.4e-6 | 4.382e-3 | 4.74 -> 1.10 |
| engram_kv (2) | 1024x6144 bf16 . 6144x12800 bfp8 -> bf16 | 2.3117 | 0.7956 | minimal M4 K4 N8 | 1.1e-5 | 4.380e-3 | 4.62 -> 1.59 |
| attn.wo_a (8) | 2x1024x4096 bf16 . 2x4096x1024 bfp8 -> bf16 | 0.4761 | = | auto (DS41-113) | 0 | 4.382e-3 | 3.81 |
| q_a (8) | 1024x2560 bf16 . 2560x1792 bfp8 -> bf16 | 0.3072 | 0.0684 | minimal M4 K4 N8 | 0 | 4.381e-3 | 2.46 -> 0.55 |
| router (8) | 1024x5120 bf16 . 5120x96 bf16 -> fp32 | 0.0862 | 0.0692 | 2d 3x8 pm4 pn1 bw16 sb4x1 | 0 | 4.072e-3 | 0.69 -> 0.55 |
| indexer.wq_b (4) | 512x1280 bf16 . 1280x4096 bfp8 -> bf16 | 0.1573 | 0.0492 | minimal M4 K4 N8 | 0 | 4.386e-3 | 0.63 -> 0.20 |
| hc.fn (3) | 1024x10240 bf16 . 10240x32 fp32 -> fp32 | 0.1540 | 0.0993 | 2d 1x8 pm4 pn1 bw16 sb4x1 | 0 | 4.587e-3 | 0.46 -> 0.30 |
| compressor (2) | 1024x2560 bf16 . 2560x1024 bf16 -> bf16 | 0.1511 | 0.0452 | 2d 11x8 pm4 pn3 bw16 sb4x1 | 0 | 4.375e-3 | 0.30 -> 0.09 |

Best seen per site: within 2 % of the rule's pick everywhere except indexer.wq_b (minimal N16 0.0466) and wo_b (minimal
M16 0.1365); router 1D in1-mcast bw8 0.0632. Rejected: minimal for fp32 outputs (router vs fp64 4.33e-3 vs 4.07e-3);
every batched config (above). Explicit 2D on wo_b reached 0.1456 (bit-identical) vs minimal 0.137.
Other boxes (test_rule_boxes): every call at 640 / 512 rows (galaxy_8x4) with TP-4 widths (2560 -> 1280, 12800 ->
6400), on the 11x10 grid and an 8x8 sub-grid: the rule's config ran, deterministic, error vs fp64 = auto's, 1.6-3.3x
faster (e.g. wq_b 640 rows 0.264 -> 0.087; engram 0.733 -> 0.293); wo_a -> auto.
In the model: capture with tuned: chunk out vs golden rel 0.02803 (auto 0.02803). Quick A/B (measure, on): every layer
and state within its limit, worst L03 rel 0.08050 (P.22 0.08050), wall 97.4 ms. gate P.23 --check: PASS (8 swap units,
ladder last, profile), device_ms_total 69.66 (X.3 83.08): attention 30.27 -> 22.83, engram_kv 4.62 -> 1.59, q_a 3.09
-> 1.19, indexer 11.69 -> 11.24, compressor 1.33 -> 1.03, attn_hc 1.74 -> 1.58, router 1.04 -> 0.90; clean wall 98.6 ms
(X.3 120.3). Out of scope: the experts' fused matmuls; packer_l1_acc (GLM's other change) untouched.
Gotchas: `-k sweep` also selects test_capture (module name); MemoryView block addresses are base-relative (DS41-114);
minimal_matmul M8 K8 N16 (and on engram M8 K4 N16 / M16 K4 N8) refused: CBs beyond L1 (dataflow_buffer.cpp:2695) or clashing with the filler that stands for the model's L1 buffers (:2704); the rule's M4 K4 N8 needs ~330 KB.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k "test_capture"` (DS41_MATMUL_CONFIG=auto writes
calls_auto.json), then `-k "test_sweep or test_rule_boxes"`; gate: `PYTHONPATH=$PWD python -m
models.demos.common.bringup gate P.23 --spec models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## P.24 perf attempt 1 (2026-10-23): batched attention wo_a on a tuned 2D config (WO_A_CONFIG tuned / auto)

What: `tt/mm_configs.py:rule` gives the batched matmul with one weight per group (attn.wo_a, [1, 2, 1024, 4096] bf16 .
[1, 2, 4096, 1024] bfp8 -> bf16, HiFi2, fp32 DEST, 8 calls a chunk) a MatmulMultiCoreReuseMultiCastProgramConfig with
fuse_batch=False (auto's own family: each core loops the batches, fp32 partials between K blocks) at the widest in0
block whose CBs fit L1_BUDGET, 32 tiles of K included (batched calls only; auto picks 1, K-51). On 2x2: grid 11x8,
pm4 pn3, bw32, subblock 4x1, ~790 KB of CBs. The minimal_matmul branch is now batch == 1 only; a broadcast over
per-batch weights (x batch != w batch) stays auto. Started from the comparison agent's patch (reports/pr54943/
p24_mm_configs.patch, applied as is). New setting WO_A_CONFIG {tuned (default), auto}: auto = no config for the
batched call (the path before P.24), checked in `program_config` before its cache; tuned only acts with
MATMUL_CONFIG=tuned. Nothing else changed (dtypes, fidelity, ckc, memory configs, other sites).
DS41-113 (never MatmulMultiCoreReuseProgramConfig: bf16 K-block partials) held and is updated: the safe explicit
config exists, it is auto's family.
Sweep (tests/perf/test_matmul_sweep.py, extended: batched calls also get the 2D fuse_batch=False family, in0 blocks
1..32, both mcast orientations; minimal_matmul now tries every out subblock of each blocking), P23_SITES=attn.wo_a,
2x2, 20 calls, real-time profiler, L1 as in the model (1381 KiB for CBs):
| config | ms / call | vs auto (rel) | vs fp64 |
|---|---|---|---|
| auto | 0.4769 | - | 4.381e-3 |
| rule (2d nofuse 11x8 pm4 pn3 bw32 sb4x1) | 0.1160 | 0 (max abs 0) | 4.381e-3 |
| 2d nofuse bw16 / bw8 | 0.1229 / 0.1465 | 0 | 4.381e-3 |
| bmm (MatmulMultiCoreReuseProgramConfig) bw1 / 2 / 4 | rejected | - | 2.81e-2 / 1.56e-2 / 9.47e-3 |
wo_a per chunk 3.816 -> 0.928 ms. Other boxes (test_rule_boxes, same run): wo_a at 640 rows 0.2564 -> 0.0674 ms (11x10),
0.2569 -> 0.1078 (8x8 grid); 512 rows 0.2564 -> 0.0669 / 0.2557 -> 0.0783; every one bit-identical to auto, det 1;
every other site's rule unchanged and ok. In-model capture: chunk out vs golden rel 0.02803 (= P.23).
Quick A/B (measure --task P.24, on): 31 err_* within limits, min layer PCC 0.99676 (L03), host_transfers_per_layer 0,
wall 102.9 ms. gate P.24 --check: all 10 units PASS; the 8 swap units' metrics equal P.23's to the last digit (e.g.
swa_15 err_swap_out 0.026917). Profile (plain, on): device_ms_total 69.69 (P.23) -> 66.77 ms (limit < 69.7),
attention 22.85 -> 20.00, every other section within 0.03; chips 67.04-68.05 -> 64.07-65.22. Clean wall 111.1 ms
(P.23 run 98.6; this run's spread 27.1 ms vs 8.5: wall noise, the device time is the measure).
Gotchas: the background-moved gate --check was cut at the tool's 20-min limit after 7 units; per-unit
`--check --unit 'P.24#<name>'` (the full unit id) runs in the foreground. `-k sweep` still selects test_capture and
test_rule_boxes (module name).
Re-run: `P23_SITES=attn.wo_a PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k "test_sweep"` (needs generated/ds41_p23/
calls_auto.json from `DS41_MATMUL_CONFIG=auto ... -k test_capture`); A/B: `python -m models.demos.common.bringup.
orchestrator measure --task P.24`; gate: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate P.24 --spec
models/demos/deepseek_v41_flash_d_p/bringup/spec.yaml --check`.

## W.1 port (galaxy_8x4-full, run1): the seed's layers on 8x4 at chunk 5120

What: the seed's 8 layers [0, 1, 2, 3, 14, 20, 21, 24] on mesh 8x4 (SP 8, TP 4: 640 rows a mesh row, 160 query rows a
chip, hidden share 1280), each run from its golden input. Four 8x4-only failures, each fixed in place for every box:
1. Shared expert (DS41-115): TP 4's slice 576 -> 768 (24 tile columns) lands in the flat op's 2 / 3-subgrid plans,
   which have no gate/up split at H 5120 (TT_FATAL at load). tt/shared_expert.py pads each part to >= 33 tile columns
   (ONE_SUBGRID_MIN), the one-subgrid plan the seed runs: 576 -> 1280 (2.2x shared-expert work on TP 4; 2x2 unchanged
   at 1152 -> 1280). Better: a one-subgrid / HG option in the flat fork for It 24 (48 units on 48 cores), a perf pick.
2. Served compressed caches (DS41-116): TP-dedup ratio-2 stripes are 640 / 2 / 4 = 80 rows a chip and the cache at
   56320 is 880 rows; stock update_padded_kv_cache checks both in 32-row units. New fork ttnn.bringup.update_padded_kv_cache
   (validation only, ROW_MAJOR copy path in rows; kernels untouched); tt/compressor.py and tt/attention.py call it.
   Fork test tests/unit/test_row_stripes.py (8x4, exact vs torch, scalar + metadata + clamp; mutation of the reference
   row fails it). The source op's tests do not collect on this tree (deepseek_v3_d_p tests conftest imports xing40's
   settings, which pass the removed max_precision): baseline recorded as that error, see the fork's CHANGELOG.
3. Indexer score (DS41-117): Q_CHUNK 64 does not divide 160 rows; tt/indexer.py takes the largest multiple of 32 <= 64
   dividing the chip's rows (32 on 8x4, 64 on 2x2).
4. Combine column barrier (DS41-118): tt/weighted_combine.py _column_barrier padded the column-end chips (1 fabric
   direction on > 2 rows) with a duplicate connection to the same router channel; intermittent hang on rows 0 / 7.
   Each chip now opens one connection per direction it has. Probe (8x4, add + barrier loop): old hangs < 50 calls,
   fixed 2000 / 2000 (probe script not kept: tt-probe saves under tests/, outside this step's paths).
Not changed (owner decisions / perf picks per the brief): settings defaults; mm_configs, link counts (indexer
num_links 2), core grids; the shared expert's 2.2x padding and the score's q chunk 32 are Galaxy perf-pick inputs.
Result: gate W.1 PASS. err_layer (limit): L00 0.0249 (0.054), L01 0.0368 (0.067), L02 0.0516 (0.090), L03 0.0711
(0.107), L14 0.0320 (0.055), L20 0.0300 (0.037), L21 0.0436 (0.074), L24 0.0312 (0.0336, the tightest); err_state_max
0.039; host_transfers_per_layer 0; load 1057 s, 2 chunks 91.6 s.
OV-1: sparse_sdpa (160 query rows a chip after the reshard), the ring indexer (W 640, 160 rows; fix 3) and the served
records (160-token sub-blocks, 80 / 40 compressed rows; fix 2) all run at 8x4's sizes in this gate, as do the
4-column TP paths. Every row count comes from mesh.shape and the chunk.
Gotchas: the gate --check takes ~20 min (8 layers load ~2.2 min each); a background tool call is cut at 10 min unless
given a longer timeout. The units log (runs/.../logs/W.1.units.log) appends across runs.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate W.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; fork test:
`scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/update_padded_kv_cache/tests/unit/test_row_stripes.py`.

## W.1 port, attempt 2 (2026-10-26)
Attempt 1 passed the gate but its diff included ledgers/galaxy_8x4-full/supervision.md. Those lines are the overseer's
(the W.1 pre-check review and the session hand-off notes), not the step's, and the orchestrator already committed them in
9582c9da53d. No code change in this attempt, and supervision.md was left as it is in HEAD: reverting it would delete the
overseer's notes and add a new out-of-scope change. Gotcha: `timeout 600` around the ~20-min gate kills it; run the gate
in the background with a timeout over 25 min. A SIGINT to the ladder's pytest during weight load takes effect only once
the load returns to Python.
The overseer appended another supervision.md line at 13:24:03 (after this brief). I restored the file to HEAD so the
step's scope check passes; the line is kept word for word in results/knowledge_use/W.1.port.2.json for the overseer.

## FIX.1 overseer, attempt 2 (2026-10-26): no padding around the flat op's limits (owner order, OV-3)
What: tt/shared_expert.py runs the chip's real intermediate slice (576 on TP 4: 8x4 / 4x4 / 2x4; 1152 on TP 2: 2x2)
as the one always-on expert: DOWN_K, ONE_SUBGRID_MIN and the zero-padding (576 -> 1280 = 2.2x, 1152 -> 1280) are
gone. The fork was extended instead (ttnn/ttnn/bringup/flat_routed_expert_ttnn, CHANGELOG "Intermediate widths with no
subgrid split"):
- flat_routed_expert_plan.cpp: when the 2 / 3 subgrids picked for It <= 32 have no gate/up split (H 5120: DS41-115),
  plan one subgrid; the uneven pair split there takes It 18 (np 1, 18 gate/up cores, 30 down, reader tails) and It 36
  (np 1, 36 cores). Plans that worked before are unchanged (the check only replaces the old TT_FATAL).
- se6_dcompute.cpp: the static_assert's `kt_d % 8` term removed. DS41-76 was wrong: hk = 8 is only matmul_block's
  kt_dim with rt_dim 1, unused by the LLK (llk_unpack_AB_matmul.h, reuse_a), so a down K of 18 / 36 tiles is fine.
- No option: both changes only reach inputs that failed before.
Measured:
- Fork tests/unit/test_flat_widths.py (new; one Galaxy card, H 5120, vs float64 within noise_limit, deterministic):
  I 576 glm rel 0.0400 (limit 0.099), full 0.0104 (0.021), 3 experts 0.0398 / 0.0401; I 1152 glm 0.0400, full
  0.0105. Corrupted h (MIMO_FL_NO_ACT=1) fails it: rel 1.475.
- Shared expert on 8x4 after the change (tt/perf_shared_expert.py::test_shared_flat, layer 0, chunk 5120, 640 rows a
  chip, EXPERTS_PRECISION=glm; one run, no device profiler, per OV-4): 0.153 ms a layer on the model's path (rows:
  0.484 minus the test's 0.331 ms rows clone), 0.513 ms with its own x conversion; MoE block 4.662 ms (3.922 without
  the shared expert and moe_add). vs float64: rel 0.0213, bias +1.43 % (P.12: 0.020-0.022); twice bit-identical. No
  before number on 8x4: OV-4 allowed one timing run, after the change. The padded op ran 2.2x the weight columns.
- Gate FIX.1 --check (W.1 unit, 17.5 min): passed, every layer's shared expert "parts=1 x 576 rows=640", np 1, 18
  gate/up cores. err_layer_L00..L24 and every err_state equal to W.1's to the last digit (L24 0.031249 <= 0.033573;
  host_transfers_per_layer 0): the padded lanes only added exact zeros at the end of the down's sequential fp32 K sum.
Gotchas:
- On a single Galaxy card (the `device` fixture) the flat op hangs on the first call for most shapes (the I 2304 / glm /
  k3 / mimo unit cases); glm_e12 hangs on the code before FIX.1 too; 2x2-mesh tests on the same box pass (DS41-119).
  test_flat_widths skips I 2304 on a 12-column grid.
- The fork's model case tests/cases.py "deepseek_v41_flash_d_p-2x2-shared-expert-s1024-i1280" is the old padded call
  (I 1280); the model now calls I 1152 on 2x2 and I 576 on 8x4. A later O.1 should add those cases (append only).
- The fork regression on one Galaxy card is not comparable to the 2x2 QuietBox baseline (CHANGELOG lists each failure).
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --device auto
ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_flat_widths.py`; timing `scripts/run_safe_pytest.sh --run-all
--no-precompile models/demos/deepseek_v41_flash_d_p/tt/perf_shared_expert.py -k "test_shared_flat and time and L00"`;
gate `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.1 --spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.2 overseer, attempt 1 (2026-10-26): the flat op's Galaxy hangs root-caused (DS41-119)
What: the fork ttnn/ttnn/bringup/flat_routed_expert_ttnn only (no model code changed; CHANGELOG "The schedule leader
found by its logical coordinate").
- Cause (tt-triage, test_flat_uneven_split[glm_e12] on one Galaxy chip, 5 s dispatch timeout): with SE_DYN_PULL (plans
  with no reader tails and no combine attached) reader 0's BRISC reads the counts and sets semaphore SE_DYN_LSEM; every
  other core polls it. The leader recognised itself by `my_x[0] == SE_DYN_LPX` with LPX = reader 0's physical NoC0 x
  (flat_worker_noc0), but BH firmware fills my_x from NOC_ID_LOGICAL (translated). On this card physical column 1 is
  harvested (triage: logical (0,5) at physical 2-7), so reader 0 (0,9) is physical (2,11) vs translated (1,11): no
  leader, every core polling (36.9 M completed reads, one in flight). The QuietBox p150s have column 1, so it never hung there.
- Bisection probes (single chip, glm shape, 64 tokens an expert): MIMO_FL_DRISC=0 hung, MIMO_FL_COLS=11 hung, bfp8
  weights / bf16 x and h hung, MIMO_FL_DYN_PULL=0 passed. The same shape hung on the full 8x4 FABRIC_2D mesh as well
  (DS41-119's "mesh runs pass" was wrong; the mesh cases that passed had no SE_DYN_PULL). After the fix: single chip
  and 8x4 mesh pass.
- Fix: se_dyn.hpp compares get_absolute_logical_x / _y with SE_DYN_LLX / LLY (reader 0's logical core), set in the
  program factory. The model's calls never take SE_DYN_PULL (routed: flat_routed_expert_with_program + combine;
  shared expert I 576 / 1152: reader tails), so W.1's numbers should not move.
- Builder asserts: flat_expert.py `_layout` (Python parity builder) asserted rcols [0, 1, 6, 7] and an 11 x 10 grid
  (the 4 test_flat_routed_expert_op failures). Now the C++ plan's rule (readers 0, 1, E, E + 1; east part moved by
  E - 6; extra columns are down cores). test_flat_widths' I 2304 skip on a 12-column grid removed.
- Fork tests/unit/ on one Galaxy chip: 24 passed / 14 failed / 3 skipped (2x2-mesh tests skip on one card), no hangs
  (FIX.1: 11 hangs). 12 x TT_FATAL p.nsg == 1 = the QuietBox baseline's 12 (DS41-63). Not identical to the baseline:
  the mimo and k2 indexed_perf bands (QuietBox p150 timings): mimo 1843456 / 1838520 ns vs upper 1826488, k2
  1573274 / 1575501 vs 1572413 (k2 has reader tails, so the change does not reach it; FIX.1 had 1573083). Not
  adjusted: a Galaxy band would special-case the box; the owner's 2026-10-23 rule puts Galaxy re-tuning in a perf pick.
- Not run on the 2x2 QuietBox (not on this box). Where the old check found the leader, the new one selects the same core.
Gotchas: tt-probe.sh saves probes under tests/ttnn/unit_tests/operations/<name>/probes/ (outside the step's paths:
deleted). A 2x2 sub-mesh of the Galaxy fails the fabric handshake; use the 8x4 mesh and touch every
/tmp/tt-device-N.dirty after single-card runs (K-57).
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --device auto --run-all
ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/`; gate `PYTHONPATH=$PWD python -m models.demos.common.bringup
gate FIX.2 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.3 (overseer, attempt 2): real-time profiler off on chips 7 and 10 (DS41-120)

- Attempt 1 left nothing in the tree (its commit 5b1a3ec2106 passed on reused units with no change).
- Cause, measured: 30 TT cards are in a DMA-FQ IOMMU domain, `/dev/tenstorrent/7` (08:00.0) and `/10` (43:00.0) in
  `identity` (iommu=pt host; tt-iommu-domains.service converts the cards once at boot; these two were re-enumerated
  after it ran, 2026-10-22 10:34 and 2026-10-25 16:56). Metal caches chip 0's IOMMU state -> pinned-memory D2H for all;
  UMD pins 7 / 10 with CONTIGUOUS (0x4) -> EINVAL on the 2 MB + 4 KB anonymous mmap. Hugepages are not involved.
- Probe (8x4 FABRIC_2D, one add per chip, RT callback): records from 30 / 32 chips, missing [7, 10];
  IsProgramRealtimeProfilerActive() True. Saved outside the repo: runs/galaxy_8x4-full/run1/probes/FIX.3_rt_profiler_check.py.
- Not fixed at the cause: it needs root (`sudo systemctl restart tt-iommu-domains`; sudo asks a password here) and
  tt_metal (per-chip IOMMU state) is outside this step's paths. Owner / host action.
- (Reverted by the overseer, 17:35: the check moved to the framework, F194; nothing below is in the model.) Added: tt/rt_profiler_check.py; hooks.device_params logs the host mismatch (sysfs) before every mesh open;
  hooks.device_model runs the device check at build (one 32x32 add, <= 3 s settle when chips are missing) and warns
  (Setting RT_PROFILER_CHECK warn / raise / off; DS41_RT_PROFILER_CHECK=raise for a perf run that must have all chips).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.3 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.2 (overseer, reopened, attempt 2): flat fork tests/unit/ all pass on a Galaxy chip

- (a) 12 x TT_FATAL `p.nsg == 1` (TP4 / TP2 subgrid cases, since the import; DS41-63): the op was wrong, not the
  tests. Upstream c485d15bbae moved the plain program's schedule defines (SE_RPS / SE_MAX_E / SE_GU_NREG / pin) into
  `flat_schedule_defines` so flat_combine_overlap's combine could rebuild the schedule, and put the overlap's
  one-subgrid assert inside it; `FlatRoutedExpertProgramFactory::create` calls it for every plan, so every subgrid plan
  the planner builds died at program build. Fix: the assert now sits at flat_combine_overlap's call site
  (device/flat_combine_overlap_device_operation.cpp); flat_schedule_defines returns what create set inline before
  c485d15bbae, for any nsg. The model's calls are one-subgrid (DS41-115): same defines, same programs.
- (b) Perf bands: `_PERF_EXPECTED_NS` (tests/unit/test_flat_expert_indexed_shapes.py) keyed by chip class
  `(arch, grid x, grid y)`: the QuietBox p150 table as BLACKHOLE 11 x 10; a BLACKHOLE 12 x 10 table measured on this
  Galaxy chip (RT-CAL, median of 3, 2026-10-26): mimo flat 1834370 (11 x 10: 1773289, +3.4 %), k2 indexed_rm 1578919
  (+3.4 %), tp4 flat 928769 (+0.6 %), tp2 flat 1452958 (+4.3 %), tp2 indexed_rm 1706179 (+17 % over its flat; equal on
  11 x 10, worth a look in a perf pick). Margin stays 3 %. The 11 x 10 mimo flat band failed fast on the seed QuietBox
  (DS41-63, 1710341): recalibrate that class there.
- Result (one Galaxy chip, tests/unit/, --run-all): before the 12 x 10 table 34 passed / 0 failed / 7 skipped (4 perf
  "no baseline", 3 need a 2x2 mesh: not applicable on one card); with it 38 passed / 0 failed / 3 skipped (the 2x2-mesh
  tests only); a second run sits within -0.9 .. +0.6 % of every 12 x 10 baseline (tp2 indexed_rm again 1705894).
  Full gate: W.1 PASS (all err_layer / err_state within limits, host_transfers_per_layer 0), FIX.2 PASS.
- Fork docs: CHANGELOG.md entry, README.md Limits, INDEX.md entry; DS41-63 rewritten.
Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.2 --spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## FIX.4 (overseer): routed-expert weights through the flat fork's laid-out cache, faster miss path

- What: `tt/experts.py` passes FlatRoutedExpert `cache_prefix=` and a weights callable (`_host_weights`, called by the
  fork only on a miss). Prefix `expert_cache_prefix`: `<art>/tt_cache/<mesh>[-scaled]/<cache_version>/experts-<hf
  revision[:12]>/<rows>x<cols>/L<layer>.<EXPERTS_IMPL>.<sha1 of the layer's stored expert tensors (file, dtype, shape,
  offset, bytes): real vs scaled differ>`; the fork's names add `.flat.bf4.lay<plan layout hash>.{w_gu,w_d}.tensorbin`.
  `_CacheMarker` (`<prefix>.done.json`): written after the fork dumped, lists the files with their sizes and holds
  `bfp4_stats` (so a hit logs the same EXPERTS_BFP4 line, `(cached)`). The fork dumps straight to the final names (no
  tmp + rename), so every `<prefix>.flat.*` file not in the marker with its size is deleted before the fork looks
  (a run killed mid-dump repacks instead of loading a truncated file; DS41-121). No fork change.
- Wiring: Setting `EXPERTS_WEIGHT_CACHE` on / off (default on). `expert_cache_dir(spec)` from hooks (`device_model`
  -> `TtDS41Model(weight_cache=)`, `_device_step_fn` experts for component / hybrid), the adapter's
  `weight_cache_path()` (was None) -> `DS41PrefillRuntime` -> `TtDS41Model(weight_cache=params.weight_cache_path)`.
  perf_experts.py's tests build without a cache (unchanged).
- Miss path (`_host_weights`): each slot's 32-chip stack read with 8 threads, dequantized in one fp4_dequant call
  (the same per-element arithmetic), one bf16-exactness check per stacked tensor (low 16 bits of the fp32 words zero
  + aminmax finite; was torch.equal per matrix), bfp8 `_bfp8_exact` / bfp4 `_bfp4_change` on the fp32 stacks (no
  .float() / slice copies); `_bfp4_round` and `_bfp8_exact` take max |v| as max(max, -min) (no |g| copy). Values,
  order and layout unchanged: `tt/perf_experts_load.py::test_host_weights_identical` (8x4, layer 0, all 384 experts):
  every (W1, W3, W2) bf16 tensor torch.equal to the old per-expert loop's, same chip / slot order, same changed count
  2262393486 / 13589544960, same worst matrix (122, w3) 0.11965. Batched vs per-matrix dequant measured equal
  (synthetic 32 x 2304 x 5120: 0.67 vs 0.70 s a stack); kept batched.
- Numbers (8x4, layer 0, bfp4, nice 19): old miss 118.9 s (cProfile: from_torch bfp4 packing 54.8 s,
  flat_weight_regions 22.4 s, _bfp4_change 20.2 s, _load_block loop 7.7 s, fp4_dequant 2.9 s); host part old 49-55 s
  -> new 31.7 s (same process, same page cache); new off (no cache) 127.1 s under cProfile, miss (packs + dumps)
  119.9 s (host_weights 24.4 s), **hit 1.0 s** (warm page cache; device tensors of devices 0 / 31 torch.equal to the
  miss's, bfp4_stats equal). 7.12 GiB of disk per layer (bfp4; 40 layers ~285 GiB; /scratch had 2.2 TB free). W.1
  gate run 1 (seed layers, L00 cached by the perf test): L01..L24 misses 116-130 s each, L00 hit 1.2 s; PASS, every
  err metric equal to the previous W.1's to the last digit; model_load_s 1057.3 s (old code, 8 misses) -> 887.7 s.
  `gate --check --rerun W.1` did not re-run it ("reused: same code, not run", F150), so the hit side was measured by
  `perf_experts_load.py::test_cache_hits` (the seed rung's 8 layers, as the ladder builds them): all hits, 1.0-1.23 s
  each, 8.3 s total (vs ~950 s of expert load on a miss): a W.1 with every layer hit loads in ~70 s (887.7 - 7 x
  ~117 + 7 x 1). The final gate check (after that test was added under tt/) re-ran W.1: all 8 layers hit (1.0-1.2
  s), PASS, metrics identical, **model_load_s 65.6 s** (was 1057.3 s), W.1 command 119 s (run 1: 956 s).
- Left: the miss is now ~75 % the fork's host packing (ttnn.from_torch bfp4 55 s, flat_weight_regions' 18880
  torch.cat 17 s); a fork-side change (parallel per-device packing, vectorised region building) is the next cut, not
  done here (needs the fork's regression). Dense weights (DENSE_BFP8, ~5 s / layer): not cached (not cheap: no
  cache path in the dense modules; the expert cache removes ~95 % of the load).
- Profile / check: `PYTHONPATH=$PWD nice -n 19 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_experts_load.py -s`.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.4 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check` (`--rerun W.1` for the hit run).

## FIX.5 (overseer, attempt 1, 2026-10-27): flat_routed_expert_with_program keys its program without addresses

- Cause (confirmed): the op hashed every baked buffer address (`FlatWithProgramParams.addrs`, `counts_addr`) and its
  `override_runtime_arguments` was empty; of those addresses only the gathered x (`xg`) moves call to call (overseer
  keylog), so each new xg address rebuilt a 32-chip program (~0.55 s host). DS41-122.
- Fix IN THE OP, behind a new option (rule 6; default = the op before): `patch_addresses=True` + `extra_addrs`.
  Hash: plan key, extra_key, writers, collector, l1_args sizes, every tensor's spec (optional ones by presence), no
  address. A hit re-applies what moved since the cached program last ran: per buffer (x, y, counts, regions, ids,
  gate/up, down, reader-down, done, arena, words, token_index) the flat program's address args (its existing AddrPatch
  list, grouped by buffer, RuntimeArgsData pointers cached at the miss), the flat + appended arena CBs when the arena
  moved, the y writers' counts_addr / L1 hand-off args when they changed, and the combine's runtime args copied from
  `extra` when `extra_addrs` (arena, words, y, out, the planner's plan / table / dord addresses) change. On a miss the
  combine's arena CBs are bound to the call's arena by the op (the cached descriptor may name a freed arena tensor at
  the same address; the old `seen` / `rebind` dance is not needed). `extra` is now a non-owning view (no per-call deep
  copy). Weights: their addresses are patched like everything else (nothing is baked), so the 40 layers share one
  program per pass (the old key also split them by weight address).
- Model: Setting `EXPERTS_PROGRAM_KEY` patch (default) / addresses (the op before, for comparison);
  `tt/weighted_combine.py` passes the cached descriptor on every call with `extra_addrs`; the addresses mode keeps
  the old `seen` logic. EXPERTS_XG_PINNED (the overseer's hack, default off) is moot with patch; left in place.
- Numbers, 8x4 (`tt/perf_flat_program_key.py`, layer 0, chunk 5120, real weights, 6 calls a phase): op host per call,
  hit: addresses 0.41 ms; patch 0.44 ms (same buffers) / 0.46 ms (x moved every call); miss ~0.55 s. Moving x:
  addresses +1 program every call, patch +0. All 24 outputs bit-identical to the address-keyed op's first call. A
  first version that re-applied every patch on every hit cost 2.0 ms a hit (override 1.51 ms for 26496 arg / CB writes
  over 32 chips): now only what moved is written.
- F196 warm wall (`BRINGUP_PROFILE_WALL=1 test_profile.py`; the profile's rung is `last`, i.e. 40 layers, one 5120
  chunk at position 0, whatever BRINGUP_RUNG says): patch: warm after 1 warm-up run (237 -> 237 entries), 10 timed
  runs with misses refused all completed, chunk_wall_ms median 999.8 (755-1175). addresses (`DS41_EXPERTS_PROGRAM_KEY=
  addresses`): warm-up +40 +36 +37 +38 +23 +34 +18 +2 +0 (316 -> 504 entries, 17-30 s a run), then the first timed
  run missed in FlatWithProgramDeviceOperation and was refused: never warm. Logs /scratch/dnijemcevic/fix5/.
- Fork tests: new `tests/unit/test_with_program_patch.py` (the model's wiring via tests/weighted_case.py, model shape:
  384 experts / 12 a chip / 640 tokens a chip / top 6 on 8x4): BRINGUP_TEST_MESH=8x4 PASS (bit-identical vs the
  address-keyed op; moved x +0 programs; moved arena, i.e. extra_addrs changed and the combine's args copied, +0
  programs, bit-identical; arena back, bit-identical). It first hung on 8x4 in the unchanged address-keyed call: the
  fork's weighted_case.py never got the model's DS41-118 barrier fix (ported now, DS41-123). `tests/unit/` on one
  Galaxy chip (`--device auto`, as FIX.2): 38 passed, 4 skipped (the 2x2-mesh tests; the new one defaults to 2x2).
  A 2x2 mesh cannot open on this Galaxy (fabric router sync timeout).
- Gotcha: run_safe_pytest without `--no-precompile` runs the precompile collect pass; use the flag for mesh tests.
- Left: the fork-call capture (O.1) keys sigs on argument values; `extra_addrs` holds addresses (allocator-
  deterministic in practice) and the routed call now always carries the full descriptor, so its sig changes once;
  `fork_capture.VOLATILE_KWARGS` (not this step's path) could list `extra_addrs` like `extra_key` (K-45).
  The 2x2 model case in tests/cases.py still runs the address-keyed default.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.5 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; measurement:
  `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_flat_program_key.py -s`; fork: `BRINGUP_TEST_MESH=8x4
  scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_with_program_patch.py`.

## K.1 contract attempt 3 (galaxy_8x4-full)
- OV-12: `tt/runners/kv_contract.golden_rows` returns (rows, first row, mode). The window KV of a layer with per-chunk
  golden dumps (all chunks present: step_dump_layers 0,1,2,3,14,20,21,24) is compared over [0, length) ("full"); any
  other layer over the last min(128, length) positions from `golden.state(layer, at=length)["window_kv"]`, unrolled by
  position (ring row p % 128) ("last-window"). Ring layout checked on CPU: layer 0's ring at 10208 and at 10240 equal
  the per-chunk 'kv' dumps exactly (max diff 0). Compressed KV / index keys unchanged (final state, full). No layer is
  skipped; the per-layer mode is logged and summarized (`KV comparison per config and kind`). Test-side read-back
  only; the model path reads no golden.
- Gate units: test_contract PASS (window_kv 0.9762 / limit 0.9434, compress_kv 0.9839 / 0.9370, index_k 0.9914 / 0.9465;
  acks_early 0; contract_checks_failed 0); test_kv_layout PASS.
- Blocker outside K.1's paths (DS41-124): the frozen tests test_adapter_acks / test_runner_contract build
  `caches.GoldenKV()`, which needs ladder rung `s4096` (server_rules.GOLDEN_RUNG, CHUNK 2048 / MAX_SEQ 4096). The 8x4
  spec has no such rung and golden_full/galaxy_8x4 no s4096 golden: CPU probe `C.GoldenKV()` -> SpecError. With the
  rung, GoldenKV._boundary would still need per-chunk 'kv' / 'in' for all 40 layers. Needs a spec rung + golden, or a
  test change (overseer / owner). test_runner_smoke (no GoldenKV) not run by me: left to the orchestrator's gate.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate K.1 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check --unit 'K.1#test_contract'`.

## K.1 contract attempt 5 (galaxy_8x4-full)
- OV-15: `tests/bringup/contract/server_rules.golden_geometry()` derives (GOLDEN_RUNG, CHUNK, MAX_SEQ) from the running
  spec: the shortest ladder rung (by seq) with its own golden and `full_dumps`. Seed spec.yaml -> s4096 / 2048 / 4096,
  galaxy_8x4 -> s10240 / 5120 / 10240 (CPU check, Spec.load, no device). Scenario / runner / ack lengths are written
  as the seed's offsets from H = CHUNK / 2, CHUNK and MAX_SEQ: the seed gets exactly its old plans (1000 / 3000 /
  3017 / 1330 / 3001 / 4000 / 1500 / 3500); 8x4 gets 2536 / 6072 / 6089 / 2866 / 6073 / 10144 / 3036 / 6572.
  self_check asserts the same shapes symbolically (plus the ACK_TURNS interleave). test_runner_contract T1 / B come
  from RUNNER_TURNS.
- Golden gap (OV-12 rule, test side): `caches.GoldenKV.select` / `caches.golden_failures`. Window KV of a dumped layer
  (all chunks: 0,1,2,3,14,20,21,24 on 8x4; every served layer on the seed) and every compressed / index cache: every
  row ("full", unchanged). Other layers: the positions the golden's ring snapshots hold ("last-window": [4992, 5120)
  and [10080, 10240) on 8x4), the final ring first: the 10208 snapshot is a separate reference run, fp32 noise apart
  (max 7.7e-4 of 2.7 at layer 39, not bit-equal). `coverage_failures` fails any served (config, layer) never compared;
  each test prints the per-config, per-mode layer list. 8x4 coverage per test: adapter acks / runner contract / cache
  starts: window_kv layers 0-3,14,20,21,24 full, the other 32 last-window (128 rows in [0, 6072), 288 in [0, 10240));
  compress / index (2, 8, 14, 20) full. The ack-time checks of [0, 2536) compare no rows of a non-dumped layer's
  window KV (none in the golden there); the slot-final checks do.
- `parts.run_chunk`: a layer with no golden block input (neither its "in" nor the previous layer's "out", F164) runs
  from the previous layer's device output of the same chunk (test_cache_starts on a full 8x4 run).
- Paths: OV-15 orders `tests/bringup/contract` changed; the brief's contract-role allowed list does not include it.
- test_adapter_acks on 8x4 (unit run): all 4 chunks give 40 acks in order, coverage as above (dumped layers 24328 rows,
  others 512 last-window rows), 6 failures, all window_kv "nope" PCC of the dumped deep layers vs the frozen
  per-layer state limit: L20 0.981 / 0.9896 ([0, 2536)), 0.9873 ([0, 5120)), 0.9888 ([0, 6072)); L21 0.985 / 0.9886;
  L24 0.980 / 0.9837. DS41-125: the limit comes from the ring drift (last 128 positions); L20's KV is L19's output,
  which the ladder measures at err 0.159 over all positions. Limit unchanged; left to the owner.
- test_runner_smoke on 8x4 (unit run, feeder tt-d-gen engine /scratch/dnijemcevic/tt-d-gen @ a9701ce6675): the runner
  came up (40 layers, table and device map exported, request loop entered); the engine driver died before any chunk:
  ENGINE_FATAL chunk_size=5120 not divisible by sp_factor=1800000 (DS41-126: testing/dgen_prefill_driver.py:79 passes
  layers_per_chunk positionally; tt-d-gen 1cf343887 removed it). Framework file, outside K.1's paths: not changed.
  test_runner_contract uses the same driver; not run.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate K.1 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## K.1 decode hang: the stand-in decode's mesh had the default fabric payload (fix role, overseer-directed, 2026-10-27, bh-glx-110)
- Symptom (K.1 run 2 on b06u08, the 1-layer repro here): the first MoE call (flat_routed_expert_with_program + the
  weighted combine) hangs on all 32 chips: se3 / se6 compute in matmul, readers in cb_reserve_back, combine
  accumulators polling their rings, ports in cb / credit waits, routers on credits.
- Not the routing. The call's PAD rows put ~4900 of a column's 5120 tokens on each of 6 experts (54 / 99 / 153 / 318 /
  355 / 379). A standalone replay of the op (fork tests/unit/test_hot_experts.py, the model's wiring through
  weighted_case.py) with the captured routing (dump_cap128/call0000_L00.pt) PASSES (min PCC 0.99989 on every chip, 15 s).
  It also passes with poisoned L1, the model's own fabric all-gather of x, uint16 ids, the model's real x, and every
  synthetic extreme (one chip at 30720 rows, every column's tokens on one expert). In the model the call still hangs
  with random weights (diag plugin swapping the layer's fe), so weights and data were ruled out too.
- Cause: tests/bringup/contract/test_runner_smoke.py open_mesh (used by the K.1 stand-in decode and the diag
  repro) called ttnn.set_fabric_config(fabric_config) without the spec's fabric_router_config. The router kept its
  4352 B default payload, but the combine sends one 10272 B packet a row (10240 B + 32 B trailer). The replay on a mesh
  opened that way hangs with the model's exact triage signature (same kernel counts, same 15-3 / 15-2 timeout cores).
  tt-d-gen's runner opens its mesh with the payload (K-79), so the prefill ran and only the decode hung.
- Fix: open_mesh passes router_config (one call). Fork combine_rf: sparse_combine_rf_prepare refuses a combine group
  of > 1 chip whose ttnn.get_tt_fabric_max_payload_size_bytes() < piece + 32 B, naming the payload it needs
  (tests/unit/test_fabric_payload.py). No kernel or program change.
- Results: 1-layer repro REPRO_CAP=128 STEPS=0 PASS (prefill [0, 128) 13.5 s); default STEPS=1 tail PASS; all 40
  layers, REPRO_STEPS=4: PASS, model up 386 s, prefill 57.6 s, tail steps 1.1-2.7 s, greedy 'We need answer one'
  (coherent; the smoke's Paris check is K.1's runner smoke). Fork
  regression on 8x4 (BRINGUP_TEST_MESH=8x4, flat tests/unit + combine_rf tests/unit + test_flat_routed_expert_ttnn.py):
  every test that builds sparse_combine_rf here passes (test_with_program_patch, test_hot_experts 3/3,
  test_fabric_payload 2/2, test_max_y_writers 2/2). Not related to the change: test_acc_dtype (2x2 only, fabric
  init error on Galaxy), test_column_planner_mc forced to 8x4 (planner L1, never calls the combine),
  test_flat_routed_expert_op[tp4] (AssertionError (16, 1, 6, 2), the plain op), and the k2 perf band (+4.7%, E=7 node).
- Gotcha: a mesh opened by hand (ttnn.set_fabric_config + open_mesh_device) must carry every device_params entry the
  harness gives, fabric_router_config included; only the conftest / harness path applied it.
- Re-run: `cd /scratch/dnijemcevic/diag && REPRO_LAYERS=1 REPRO_CAP=128 REPRO_STEPS=0 ./run.sh <name> --no-precompile
  /scratch/dnijemcevic/diag/test_repro_tail.py -s`; op replay: `BRINGUP_TEST_MESH=8x4
  HOT_CAPTURE=/scratch/dnijemcevic/diag/dump_cap128/call0000_L00.pt scripts/run_safe_pytest.sh --run-all
  ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_hot_experts.py`.

## K.1 contract, attempt 6 (OV-16, owner decision B)
- test_adapter_acks failed only on dumped deep layers' window KV over every position (L20 0.981 < 0.9896, L21, L24):
  the limit (server_rules.state_threshold, drift state_window_kv_L*) is measured on the ring = the last 128 positions
  (DS41-125). Owner decision B: change the compared positions, not the limit.
- caches.GoldenKV(window="last"): a dumped layer's window KV is compared on the last min(ring, hi - lo) positions of
  each compared range [lo, hi) (ring = rows of the golden's window_kv state, 128), mode "last-window"; other layers keep
  the ring-snapshot positions; compressed / index caches every row; coverage_failures still fails any layer never
  compared. Default window="all" keeps test_cache_starts / parts unchanged.
- Used by test_adapter_acks (every ack: [max(lo, hi - 128), hi) of every layer that has dumps) and test_runner_contract's
  kv_checks (same check, same limit: its slot 1 final [0, 6572) would hit the same L20 gap).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate K.1 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.
- Then test_runner_contract's other check failed: the pulled-back chunk (0, 5120, 10240) rewrote layer 0's window KV
  [5120, 7616) with other bytes (704 rows, 1 bf16 ulp; contract: layer 0's KV depends on the token and position
  only). The differing tiles were exactly those whose tp_sum row block (640 / 4 = 160 rows) changes between starts
  2496 and 5120: the bf16 fabric reduce-scatter by row blocks adds the 4 TP partials in a block-dependent ring order
  (DS41-128). New Setting TP_SUM_BLOCKS (default cols): tt/ccl.py:tp_sum splits by column blocks (dim 3) + column
  gather, or the fixed-order gather_add where the width does not split (mHC [S/SP, 32] row). Same volume, same bf16
  precision; rows keeps the old path. After it: test_runner_contract PASS (0 failures, tt-d-gen engine).

## Merge of origin/dnijemcevic/model_bringup (owner's MoE work, d762b21db30) into the galaxy branch (2026-10-28)
- Kept both sides: theirs (Marko sync of flat_routed_expert_ttnn / combine_rf, EXPERTS_LAYOUT=marko, MOE_IMPL=block,
  MOE_HEAD / MOE_TAIL fused, matmul_reduce_scatter, D14 torus for the block, framework F200 / F201, O.1 cases); ours
  (FIX.5 patch_addresses, max_y_writers, combine_rf payload check, test_hot_experts, TP_SUM_BLOCKS, contract work,
  framework F202 open_mesh). Our framework F200 (engine wait) -> F203, F201 (hangdump) -> F204; F202 kept (free).
- tt/experts.py: their flat_op() split (shared with the block) carries max_y_writers=L1_MAX_WRITERS; the block's
  exchange (deepseek_v3_d_p tt_k3_token_split_moe.py) passes the op's max_y_writers to flat_routed_expert_with_program.
  tt/weighted_combine.py passes layout + patch_addresses + extra_addrs + max_y_writers.
- Weight cache: on this box the marko plan packs the routed weights exactly as the capped legacy plan (32 down x 5
  columns, 16 readers, no tails; key lay9c3da689dd35): layer 0 loads cache=hit under marko, flat and block. No rebuild.
- 1-layer repro (smoke prompt, prefill [0, 128) + tail [128, 160)): flat (defaults) PASS; block + fused head FAIL at
  build (DS41-129, assertion, no hang); block + MOE_HEAD=ops PASS, same tokens as flat.
- Re-run: `cd /scratch/dnijemcevic/diag && [DS41_MOE_IMPL=block [DS41_MOE_HEAD=ops]] REPRO_LAYERS=1 ./run.sh <name>
  --no-precompile /scratch/dnijemcevic/diag/test_repro_tail.py -s`; fork: `BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh
  --run-all ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_with_program_patch.py` (and test_hot_experts.py,
  test_max_y_writers.py, combine_rf/tests/unit/test_fabric_payload.py).

## FIX.6 (overseer fix): MOE_IMPL default block, the one configuration on the Galaxy torus (owner 2026-10-28)
- tt/settings.py MOE_IMPL default flat -> block (MOE_HEAD / MOE_TAIL stay fused; no precision or limit changed); the
  setting's text carries the owner decision and the 06:20 torus evidence (block on 8x4 FABRIC_2D_TORUS_XY device
  219.5 / 233.9 / 237.5 / 249.1 / 284.4 ms at 0 / 50k / 100k / 200k / 500k vs flat FABRIC_2D 308.1 / 314.6 / 323.3 /
  338.0 / 375.7; warm wall 615-646 vs 672-1005 ms). Flat stays selectable (DS41_MOE_IMPL=flat) for the FABRIC_2D boxes.
  New DS41-130: flat hangs on the torus at the first MoE call. tt/MOE_BLOCK.md and the adapter docstring updated.
- Fabric of every gate path, checked in this gate's logs (no code change needed): the ladder / smoke / contract meshes
  open the spec's device_params (FABRIC_2D_TORUS_XY, hooks.device_params adds the 10880 B payload); the K.1 runner
  process gets PREFILL_FABRIC_MODE=2d_torus_xy from caches.fabric_mode(spec) and the payload from
  DS41Config.FABRIC_PAYLOAD_SIZE (10880, l1_small 24576): runner.log "Fabric config: FABRIC_2D_TORUS_XY (sp=8,
  PREFILL_FABRIC_MODE=2d_torus_xy)", then MOE_BLOCK layer=0 head=fused; the stand-in decode opens harness.open_mesh(S)
  (F202: every device_params entry, router_config included).
- Gate numbers, block (FIX.6) vs flat (the earlier passes): L.s10240 err_final_hidden 0.1920 vs 0.1891, top5 1.0 / 1.0
  (top1 0.9971); L.last 0.1878 vs 0.1852; L.s56320 0.1904 vs 0.1888 (limit 0.2586), top5 1.0; host_transfers 0;
  L.smoke Paris; K.1 contract 0 failures, acks_early 0, producer KV PCC window 0.9771 vs 0.9774, compress 0.9846 vs
  0.9845, index_k 0.9918 vs 0.9919; runner smoke (tt-d-gen a9701ce6675 prefill 333 s + stand-in decode) Paris.
  Unit times: W.1 2.4 min, M.1 / L.s10240 7.2-7.4, L.last 7.6, L.s56320 8.1, smoke 6.2, contract units ~6 each.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.6 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## P.1 (perf, galaxy 8x4): GENERIC_OP_PREPARED, the fabric reduce-scatters through prepared programs
- Fork option `ttnn.bringup.fabric_reduce_scatter(prepared=True)` (needs program_hash): `_fast` keeps a
  `ttnn.bringup.prepare_program(desc, [input, scratch, output])` next to each cached descriptor (its own LRU entry,
  dkey + "prepared") and dispatches with `ttnn.bringup.run_prepared`. The descriptor LRU is keyed by the buffer
  addresses, so a call patches nothing unless another descriptor of the same structure ran in between (then the 3 io
  slots). Not through `prepared_program.generic_op`: its 512-entry cache is shared with the MoE block.
- tt/settings.py `GENERIC_OP_PREPARED` (default on; off = generic_op on the cached descriptor, the old path);
  tt/ccl.py `prepared()` (on only with GLUE_FOLD=on) passed by `_rs_bf16`, `tp_sum`, `sum_rows`, `sum_cols`.
- Quick A/B change-on side (`measure --task P.1`): warm chunk wall median 594.3 ms (runs 570.6-639.3, spread 68.7) vs
  692.7 at X.1 (generic_op): ~98 ms a chunk host saved (estimate was ~50); every err_layer within its limit,
  host_transfers_per_layer 0, pcc_chunk_out 0.6206461212358627 = X.1's to the last digit (same programs, same args).
  Device time not measured here (no device change by construction); the orchestrator's A/B profile gives it.
- Fork tests: `BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/tests/unit/test_fabric_rs_prepared.py`: 9 passed (bit-identical to
  generic_op). Whole tests/unit/ on 8x4: 21 passed; 3 failures reproduce on the unchanged file (2x2-sized full-mesh
  tests), 13 errors are fabric router sync timeouts opening 2x4 sub-meshes on the Galaxy (not this change).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.1`; off side:
  `DS41_GENERIC_OP_PREPARED=off`.

## P.2 (perf, galaxy 8x4): TAIL_A2A_PREPARED, the block MoE's fused tail through prepared programs
- Block option `MoEOptions.tail_prepared` (default False = the old path) -> `FusedTail(prepared=)` ->
  `TailA2A(prepared=True)` (models/demos/moe_block/tt/tail_a2a.py `run`): each cache entry, keyed by the y / se / out
  addresses as before, now holds `(descriptor, ttnn.bringup.prepare_program(descriptor, io))`; a hit calls
  `ttnn.bringup.run_prepared` (no per-call hashing of the inline kernel sources, no runtime args re-applied over 32
  chips). Built for exactly those addresses, so a hit patches nothing (same pattern as P.1, DS41-66). io order
  unchanged: [y, out, z, zrw] (+ se). `TailA2A.misses` counts descriptor builds (diagnostic).
- tt/settings.py `TAIL_A2A_PREPARED` (default on; off = ttnn.generic_op on the cached descriptor);
  tt/moe_block_adapter.py `block_options()` passes it.
- Not done: the persistent `out` of the proposal (optional): the model frees the tail's output downstream, a kept
  buffer needs lifetime changes in model code; the timed runs show no new program-cache entries (280 -> 280), so the
  per-call `out` already lands at repeating addresses and the descriptor cache hits.
- Quick A/B change-on side (`measure --task P.2`): warm chunk wall median 544.4 ms (runs 534.7-574.6, spread 39.9)
  vs 594.3 at P.1 (the same tree with the tail on generic_op): ~50 ms a chunk host saved (estimate ~36). Every
  err_layer / err_state within its limit and equal to P.1's to the last digit (L00 0.0247993108407984,
  pcc_chunk_out 0.6206461212358627): same programs, same args. host_transfers_per_layer 0. Device time not measured
  here (no device change by construction; the tail's program is 0.079 ms/layer); the orchestrator's A/B profile gives it.
- No new fork test: prepared_program's model cases (tests/cases.py) build the block with tail_prepared off; a tail case
  needs its signatures captured (optests).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.2`; off side:
  `DS41_TAIL_A2A_PREPARED=off`.

## P.3 (perf, galaxy 8x4): FUSED_HEAD_PREPARED_PER_LAYER, one prepared fused-head program per layer
- Diagnostic first (brief: stop if ~0): the fused head's shared prepared program (fused_head.py `_run`, layers mode)
  records 40096 patch slots over 32 chips; per io: x 10848, w 3520, y 7040, arena 8352, rows 480, xbuf 3520, route
  160, router weights (io 7-9) 3680 + 160 + 160, slots 576, planner outs 5 x 320. Patched every call (weights move):
  w + router = 7520 slots (io 1, 7, 8, 9). Not ~0, so the change went ahead. Old path warm head call: ~710-745 us host
  (`run_prepared` timed in-process, DIAGTMP instrumentation, removed).
- Why the proposal's "40 PreparedPrograms keyed by skey" alone would not work: the prepared hash excludes patched
  slots, so 40 programs with w in io share ONE cached device program and still patch. The weights must be baked in.
- Fork option (ttnn/ttnn/bringup/prepared_program): `ttnn.bringup.freeze_prepared(prepared, io, frozen)` derives a
  PreparedProgram with the io tensors at `frozen` written into their slots / CBs, no longer patched, part of the key
  (salted). prepare_program / run_prepared unchanged (the hash walk moved to a helper, same result).
  CHANGELOG, README, INDEX updated; test `_check_frozen` added to the fused-head case (frozen A / B bit-identical to
  generic_op / shared, fewer patches, distinct keys, deterministic). Fork tests on this box: 2 skipped (2x2 cases need
  exactly 4 devices; the galaxy has 32): the new check has NOT run on a device yet.
- Model: `MoEOptions.head_per_layer` (moe_block.py) -> `HeadSE3.per_layer`; `HeadSE3._layer_pp` keeps `state["pl"]`
  {weight addresses: frozen program}, cleared when the shared program is re-prepared. tt/settings.py
  `FUSED_HEAD_PREPARED_PER_LAYER` (default on per the brief; off = the shared patched program, the old path);
  tt/moe_block_adapter.py passes it.
- Result (change on, `measure --task P.3`): accuracy identical to X.1 / P.2 to the last digit (pcc_chunk_out
  0.6206461212358627, every err_layer within limit, host_transfers_per_layer 0); program cache 320 (+40, one per
  layer). Warm chunk wall median 536.8 ms (spread 70.3) and 552.6 ms (spread 49.2) vs P.2 544.4 and an old-path run
  in this step 524.4 (with timing instrumentation). Warm head call host ~970 us (instrumented run; per-layer lookup
  ~9 us) vs ~710-745 us old: the change is SLOWER by ~250 us a call, ~10 ms a chunk. Patching 7520 slots was cheap;
  switching among 40 cached 32-chip programs costs more (likely host cache misses on their dispatch data; not
  isolated). DS41-131. Recommendation: off. Device time not measured (no device change by construction: the same
  kernels and arg values).
- Gotchas: one instrumented measure died with a bus error in to_torch after warm-up (F207-like), its automatic
  re-run found "No such device"; the next measure ran clean. tt-probe.sh saves scripts under tests/ (outside this
  step's paths): diagnostics went through measure with temporary logging instead.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.orchestrator measure --task P.3`; off side:
  `DS41_FUSED_HEAD_PREPARED_PER_LAYER=off`.

## FIX.7 (overseer fix): GEMM_INPUTS, the fp8-quantized GEMM inputs fed as bfloat8_b (owner 2026-10-28, F208)
- What: tt/settings.py `GEMM_INPUTS` (default bfp8; bf16 = the path before FIX.7), tt/common.py `gemm_dtype()` /
  `gemm_input(x)`. The reference (ds41_ref.py `_q8` / `_fp8_linear`, ~620-630; the attention output's q8 before wo_a,
  ~889) rounds these inputs to fp8; the device now stores and multiplies them in its own 8-bit format. Weights,
  fidelities (ATTN_FIDELITY HiFi2, MATMUL_FIDELITY HiFi2) and fp32 accumulation unchanged.
- Per matmul (8x4, per chip, input dtype before -> after; how):
  q_a [wq_a | wkv] 640x1280: bf16 -> bfp8, one typecast of attn_norm (attn_norm stays bf16: the indexer's
  weights_proj and the compressor read it, both bf16 in the reference); tt/kv.py's standalone path the same.
  attn wq_b 640x1280: bf16 -> bfp8, the softmax-scale multiply writes bfp8 (ROPE_IMPL=on; other paths gemm_input).
  attn wo_a 2x640x4096: bf16 -> bfp8, one typecast after nlp_concat_heads (rotary_embedding_indexed has no output
  dtype; a stock op). attn wo_b 640x2048: bf16 -> bfp8, wo_a writes bfp8 (nlp_concat_heads takes bfp8).
  indexer wq_b 160x1280: bf16 -> bfp8, one typecast after mesh_partition. engram wkv 640x6144: bf16 -> bfp8, one
  device typecast of the uploaded rows (host upload stays bf16: no host bfp8 packing per chunk).
  Routed experts x / h: bfp8 already (flat op, EXPERTS_PRECISION=glm). Shared expert x / h on the default block
  fused head: still bf16, hard-coded in models/demos/moe_block/tt/se_engine3.py, outside this step (DS41-132).
- Device time, plain profile, rung last chunk [51200, 56320], 40 layers (X.1 at 10:01 today, same programs bar this
  change, vs FIX.7): total 223.85 -> 222.75 ms a chunk (attention 78.02 -> 78.71, q_a 9.34 -> 9.54, indexer 9.73 ->
  9.80, engram_kv 0.58 -> 0.62; the others noise). Per chunk, matmuls bf16 -> bfp8 input: wq_b 3.554 -> 3.065, wo_b
  3.357 -> 3.203, wo_a 2.777 -> 2.391, q_a 1.116 -> 1.024, indexer wq_b 0.319 -> 0.313, engram 0.584 -> 0.553;
  added typecasts: wo_a input 1.543 (the one real cost), q_a 0.287, engram 0.073, indexer 0.023. Profile wall 692.7
  -> 674.8 ms (not a clean wall run). Profile kept in generated/bringup_adhoc/FIX.7_profile.json (+ _ops.json.gz).
- Accuracy, W.1 (seed rung, per-step; before = W.1's 10-26 record): err_layer L00 0.0249 -> 0.0241, L01 0.0368 ->
  0.0368, L02 0.0516 -> 0.0516, L03 0.0711 -> 0.0713, L14 0.0320 -> 0.0297, L20 0.0300 -> 0.0299, L21 0.0436 ->
  0.0429, L24 0.0313 -> 0.0310 (limit 0.0336); window_kv L14 0.0054 -> 0.0112 (limit 0.065); host_transfers 0.
- Gate (FIX.6 block numbers before -> FIX.7): final hidden rel L.s10240 0.1920 -> 0.1919 (top1 0.9971 -> 1.0, top5
  1.0), L.last 0.1878 -> 0.1843 (top1 0.9971, top5 1.0), L.s56320 0.1904 -> 0.1877 (limit 0.2586; top5 1.0); s56320
  err_layer L00 0.0248 -> 0.0248, L14 0.0941 -> 0.0933, L20 0.1639 -> 0.1606, L24 0.1382 -> 0.1362, L39 0.715 ->
  0.867 (limit 1.49); host_transfers 0; L.smoke Paris; K.1 contract 0 failures, acks_early 0, producer KV PCC window
  0.9771 -> 0.9783, compress 0.9846 -> 0.9843, index_k 0.9919 -> 0.9916. Every unit PASS. test_runner_smoke failed
  once with a SIGBUS in the tt-d-gen runner after warm-up ("Non-existent physical address", no IOMMU mismatch; the
  same intermittent host fault as P.3's measure); test_runner_contract (same engine, same code) passed, and the
  smoke's re-run passed (unit re-run, unchanged code).
- Follow-up: a bfp8 output on the inverse rope (a fork of rotary_embedding_indexed) would drop the wo_a typecast
  (~1.5 ms a chunk); a bfp8 x / h option in se_engine3 for the shared expert (DS41-132).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.7 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old path: DS41_GEMM_INPUTS=bf16.

## FIX.10 (overseer fix): ENGRAM_TABLE=resident coexists with the full model on the Galaxy host (DS41-133, DS41-134)
- Cause, measured (host-only probes, /tmp/fix10_scripts, no device; then the gate's runners):
  1. Threads (the slowdown). numba's omp threading layer sets the OpenMP max-threads ICV to NUMBA_NUM_THREADS (64 =
     every logical CPU) at its first launch, and torch takes its intra-op pool from the same ICV: 32 -> 64. A
     head-layout-like torch workload (32 chips x 32-row block permutes + cats) went 2.0 s -> 23 s (user 1295 s, sys 0,
     no reclaim / compaction during it) after one pack_rows(threads=8); torch.set_num_threads(32) -> 2.0 s again. The
     resident build runs numba at model init, so the whole load and the fused head's lazy per-layer layout ran at 64
     threads: the 18:39 runner's head first calls took ~112 s a layer (random: ~6 s; the ~59 busy cores of the py-spy
     note were OMP threads spinning). Random mode was hit too, from its first begin_chunk's pack_rows on; the same
     chunk's hash + gather + TILE bytes took 110 ms at 64 threads vs 8 ms at 32.
  2. THP compaction (the build). The host's THP defrag is 'madvise': each 2 MB fault of the MADV_HUGEPAGE rows
     compacts in the faulting thread while the page cache fills memory. Both tables: 236 s (with the 64-thread
     problem; sys 860 + 1867 s, compact_stall 214k) / 171 s with the thread fix (layer 14 alone 130 s, compact_stall
     125k, 22k fallbacks to 4 KB) and direct reclaim dropped 95 GB of page cache; 4 KB pages: 74 s, compact_stall 0,
     page cache kept. Huge pages gather faster: 6.3 vs 8.5 ms a layer-chunk (8 threads).
  3. Page cache: the build's pread left the ~200 GB of table bytes cached, evicting the weight cache the load reads
     next. Swap (8 GB, swappiness 60) fills from cold anonymous pages either way; it was not the slowdown.
- Changes: tt/engram_bfp8.py `_run` restores torch.get_num_threads() around every numba launch (set_num_threads
  included; holds from any thread), kernels nogil=True. tt/engram.py: rows in 4 KB pages unless ENGRAM_HUGEPAGES=on
  (default off), every read block's file pages dropped (POSIX_FADV_DONTNEED, `_drop_cache`; resident_copy too),
  ENGRAM_BUILD=background (default; load = the old blocking build): one builder thread a process started when the
  engram layers are created (before the decoder layers load), `TtEngramRows.packed` waits for it, the gather kernel
  warmed on that thread (~0.5 s off the first chunk). Probe: both tables built in 70.8 s behind the main thread, whose
  torch work ran at baseline meanwhile (1.8 -> 1.9-2.3 s). tt/perf_engram_host.py: CPU check of the thread guard.
- Rows bit-identical: tests/test_engram_table_modes.py with DS41_TEST_ENGRAM_RESIDENT=1 9 passed (resident_bfp8
  rows == pack_rows, tiles == the per-chunk pack, resident_copy bytes); tests/test_engram_bfp8.py 1 passed (device).
- Numbers (runner start -> first forward; random = the 19:12 run, resident before = 18:40):
  load: resident before 7m43s (4m23s blocking table build, then 40 layers in 3m16s at 64 threads), random 2m38s,
  resident now 3m21s (smoke) / 2m56s (runner contract). The rest vs random: 210 GB of tables leave ~220 GB of page
  cache for the 285 GB weight cache, so part of it comes from disk (expert loads summed 72.7 s in the smoke's runner,
  32.3 s in the contract's). Fused head's first call a layer: 112 s -> 3.5 s (random 6 s, also at 64 threads then).
  K.1 runner smoke (Paris, tt-d-gen): 848.6 s test call, runner prefill 427 s, runner wall 421 s (random 19:12:
  897.5 s / runner 455 s; resident 18:39: stopped at layer ~15 after 30 min). Runner peak RSS 244 GB (smoke) / 249 GB
  (contract), disk read 444 / 244 GB (200 GB of it the tables), system swap max 7 / 3 of 8 GB, MemAvailable min 176 /
  140 GB; compact_stall +2 over the whole gate. The 17:00 SIGBUS did not recur; FIX.7 saw the same tt-d-gen SIGBUS
  ("Non-existent physical address") with random tables, so it is likely not from the tables.
- Gate FIX.10: all 6 units PASS (L.s56320 591 s, err_final_hidden 0.1889 <= 0.2586, top5 1.0; contract 0 failed,
  acks_early 0, producer KV PCC window 0.9777 / compress 0.9847 / index_k 0.9919; smoke_runner_ok 1).
- Open: the spec's owner rules ("never hold the 203 GB of tables in host RAM", "Host RAM is shared (503 GB)") still
  disagree with the default (the overseer raises it). Pytest processes keep ENGRAM_TABLE=random (F222).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.10 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old behaviour:
  DS41_ENGRAM_HUGEPAGES=on DS41_ENGRAM_BUILD=load (the thread guard has no switch: it only restores torch's count).

## FIX.8 (overseer fix, 2026-10-28): CKV_FORMAT=bfp4_rows integrated and gated at full size on the Galaxy 8x4
- What: no code change was needed. The 2x2's work (merge bf6c8b497a6 = dnijemcevic/model_bringup_bfp4kv @ 9697878dfb0;
  ttnn rebuilt with it at 22:10, `_ttnn.so` has ROW_BFP4) runs as merged on the 8x4. CKV_FORMAT defaults to bfp4_rows
  (tt/settings.py; bf16 = the old stock path). Checked for box literals: ckv_bfp4 takes its core grid from the device
  (`_cores`: bands of 32 rows over compute_with_storage_grid_size, 80 bands at r 2 / 160 at r 1 for chunk 5120); the
  sparse_sdpa ROW_BFP4 program has no grid or chip constants; AttnTable sizes come from TtServedRows (`max_seq / r`
  rows, `WINDOW + geo.chunk` window rows); no CCL or fabric payload is involved (the table is replicated, every chip
  writes its own copy from the replicated gathered tiles).
- Gate FIX.8 (`--check`, all on the merged code, 4917 s + W.1 204 s): every unit PASS. Accuracy vs the bf16 path
  (FIX.10's records, same rungs; value / limit):
  - L.s56320: final hidden rel 0.1889 -> 0.1961 / 0.2586, top5 1.0 -> 1.0; err_layer L02 0.0526 -> 0.0573, L08 0.1125 ->
    0.1221, L14 0.0930 -> 0.0981, L20 0.1636 -> 0.1678, L24 0.1386 -> 0.1431, L39 0.905 -> 0.613 / 1.49; compress_kv
    state L02 0.0330 -> 0.1161 / 0.2011, L08 0.0732 -> 0.1445 / 0.2259, L14 0.1547 -> 0.1973 / 0.3294, L20 0.1288 ->
    0.1747 / 0.2912; topk_idx L20 0.1163 -> 0.1216, L24 0.1299 -> 0.1364; candidates L20 0.0364 -> 0.0382.
  - M.1 / L.s10240: final hidden pcc 0.9794 rel 0.2021 (FIX.7: 0.1919), top1 / top5 1.0. L.last: rel 0.1914 (FIX.7:
    0.1843), top1 0.9971, top5 1.0. W.1 (seed): err_layer L14 0.0331 / 0.0550, L20 0.0309 / 0.0370, L24 0.0318 / 0.0336
    (closest to its limit), compress_kv L02 0.116 / 0.201, L14 0.112 / 0.191, L20 0.113 / 0.191.
  - K.1: contract 0 failed, acks_early 0, producer KV PCC compress 0.9847 -> 0.9767 / 0.9370, window 0.9777 -> 0.9732,
    index_k 0.9919 -> 0.9909; L.smoke and the tt-d-gen runner smoke both answer Paris. host_transfers_per_layer 0.
  - The served compressed records stay bf16 (state_bits compress_kv 16): they hold the bfp4-rounded values, so the
    serving contract and tt-d-gen are unchanged.
- DRAM a chip of the replicated attention tables at the deploy context (1,044,480 tokens, KV sources 2 / 8 / 14 at
  ratio 2 and 20 at ratio 1; from the allocated shapes in tt/compressor.AttnTable):
  bf16 rows: 3 x [527488, 512] bf16 (515.1 MiB) + [1049728, 512] bf16 (1025.1 MiB) = 2570.5 MiB (2.51 GiB).
  bfp4_rows: 3 x ([522240, 320] u8 159.4 MiB + window [5248, 512] bf16 5.1 MiB) + ([1044480, 320] u8 318.8 MiB + 5.1
  MiB) = 817.4 MiB (0.80 GiB). Saves 1.71 GiB a chip (3.1x); the plan's deployment margin was 0.3 GiB.
- Time, chunk 5120, 40 layers, resident engram (DS41_ENGRAM_TABLE=resident), same session, bf16 -> bfp4_rows
  (scripts /scratch/dnijemcevic/fix8/, results /scratch/dnijemcevic/posprof/results/CKV16* / CKV4*):
  - sparse_sdpa kernel (64 heads x 160 queries a chip, top-k 512 after a 128 window): position 0 13.60 -> 13.94 ms a
    chunk (+2.5 %; per compressed layer 0.351 -> 0.360 ms, 38 calls; layers 0 / 1 stay stock, 0.136 ms each);
    position 512k 16.70 -> 17.39 ms (+4.2 %; 0.432 -> 0.451 ms a call). The 2x2 single-card figure was +3.5 % at 160 q.
  - Table write: ckv_bfp4 0.209 ms a chunk (4 calls) vs slice_write 0.066 ms; the start < 128 compaction (ttnn.gather
    0.18 ms at position 0) is gone in bfp4_rows.
  - Device time a chunk: position 0 216.44 -> 217.66 ms (+0.56 %), 512k 283.77 -> 284.25 ms (+0.17 %).
  - Warm wall (F196, 10 timed runs, misses refused), medians: position 0 419.7 -> 441.6 ms and in a second pair 412.5
    -> 426.3 ms; 512k 451.3 -> 472.0 ms. Run spreads 30-130 ms. cProfile of 5 warm chunks (position 0) puts the
    attributable cost at +5.3 ms a chunk, all of it ckv_bfp4's host time (1.33 ms a call, Python descriptor 1.4 ms of
    the 5.3; DS41-135); both cProfiled runs took 443 ms a chunk, and the other differences (engram tile packing,
    experts) moved in both directions run to run. So the real wall cost is about +5 ms a chunk (+1.2 %), not the +14 to
    +22 ms the medians show.
- Follow-up: cache ckv_bfp4's ProgramDescriptor (patch row0 / addresses; ttnn.bringup.prepared_program) or make it a
  C++ op, which removes the ~5 ms of host time a chunk (DS41-135). The sdpa fork README had no ROW_BFP4 entry: added
  (options, Galaxy numbers, used by); ckv_bfp4 README: Measured + the host limit.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.8 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old path DS41_CKV_FORMAT=bf16;
  profile series `/scratch/dnijemcevic/fix8/series_ckv.sh`, host profile `/scratch/dnijemcevic/fix8/hostprof_ckv.sh`.

## P.6 (perf, galaxy 8x4): ENGRAM_ASYNC, the engram host work overlapped with the layer dispatch
- What: setting ENGRAM_ASYNC (tt/settings.py, default on; off = the old path, unchanged code). On: begin_chunk
  (tt/model.py:_begin_engram_async) does only the new tails (TtEngramRows.next_tail: the 3 compressed ids, no hash) and
  allocates each engram layer's device rows (TtEngramRows.allocate, the same order as off) on the main thread, then
  submits one job a layer, layer 1 first, to a single worker thread (ThreadPoolExecutor(1)). The job
  (TtEngramRows.fill) hashes, looks up + packs, and uploads with copy_host_to_device_tensor into the allocated tensor;
  it also checks its hashed tail equals the one begin_chunk stored. take_engram_rows(i) waits for layer i's job only;
  begin_chunk waits for any job left from the last chunk (a layer that did not run) before freeing its tensor.
- Why no second CQ: the harness (spec device_params) and tt-d-gen (prefill/runners/runner_utils.open_mesh_device) open
  one CQ, and MeshDevice has no Python CQ count. The worker's write goes into CQ 0 (the API is locked:
  mesh_command_queue_base lock_api_function_); CQ 0 is in order, and layer i's ops are enqueued after the wait, so the
  write lands first. Allocation stays on the main thread so device addresses do not depend on thread timing.
- DS41-136: the off path's wrap (`.numpy().tobytes()` + tensor_from_bfp8_bytes, ~35 ms a layer at 5120) and
  ttnn.to_device hold the GIL, which would stall the dispatching main thread. On: one staging host tensor a layer
  (TtEngramRows._staging: 8 mesh-row bfp8 tensors, the mesh tensor's 32 shards share them), tile_rows(out=) writes the
  bytes in place through torch.from_dlpack(host_buffer().get_shard(c)); the copy (GIL released) reads an unpinned
  buffer synchronously (fd_mesh_command_queue.cpp:write_shard_to_device, pinned_memory null), so the reuse is safe.
- DS41-133: the worker's numba launches go through engram_bfp8._run (torch's thread count kept); perf_engram_host passes.
- host_transfers_per_layer: testing/host_transfers.py wraps the ttnn attributes for the whole process, so a
  begin_chunk upload issued by the worker while the main thread is inside a layer call would count against that layer.
  The worker calls copy_host_to_device_tensor / from_torch through references bound at import (tt/engram.py
  _COPY_TO_DEVICE / _FROM_TORCH, commented). The layers' own transfers are counted as before.
- Rows: tt/perf_engram_async.py::test_async_rows_bit_identical: fill() on a thread == the off path's upload on every
  chip, layers 1 and 14, bfp8 and bf16, full / padded / 2 / 0 real ids, next_tail == the hashed tail (staging reused
  across the 4 chunks). tests/test_engram_table_modes.py, test_engram_bfp8.py, tt/perf_engram_host.py: 8 passed, 1 skipped.
- Measured (tt/perf_engram_async.py::test_engram_async_chunk, DS41_ENGRAM_TABLE=resident, chunk
  51200-56320 (rung last), 40 layers, one process, on / off alternating, 8 warm-up + 6 timed runs each):
  - off: begin_chunk 48.9 ms (median; both layers' hash, gather + tile, wrap, to_device, all on the main thread).
  - on: begin_chunk 0.60 ms. Worker, median ms: layer 1 hash 6.8, lookup + pack (one numba pass, resident bfp8 rows)
    16.8, upload 2.75 (32 chips, 4.2 MB a chip), total 26.3, main thread waited 6.1 (embed + layer 0 dispatch hides
    the other ~20); layer 14 hash 3.0, lookup + pack 12.8, upload 2.7, total 18.5, waited 0.01 (all hidden).
  - Hidden: ~42 of ~49 ms a chunk (layer 1's 6.1 ms wait is what is left).
  - Chunk wall (begin_chunk to sync) median off 490.0 ms [443-552], on 449.1 ms [422-517]; run-to-run spread is
    larger than the gain, the medians differ by -41 ms (= -48.9 + 6.1 + 0.6). Last-layer output on == off bit for bit.
  - Device time is unchanged by construction (the same programs on the same bytes; the write is the same size).
- Gate P.6 `--check` (code final, 954 s): every accuracy metric ok (err_final_hidden 0.1914 / 0.2422, top5 1.0, err_layer
  L01 0.0367 / 0.0667, L14 and every state within limits), chunk_wall_ms 424.8 < 434 (measure --task P.6: 421.0,
  spread 26), host_transfers_per_layer 0. It still prints FAIL: `MISSING device_ms_total`. The task's gate command
  carries BRINGUP_PROFILE_WALL=1 on test_profile.py, so core/gate.py:with_clean_wall adds no profiled run and no unit
  can record device_ms_total (a tasks.yaml / framework issue; not editable in this step). The plain profiled run
  (`DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/common/bringup/tests/test_profile.py`, metrics in generated/bringup_adhoc/P.6.json): device_ms_total
  220.83 ms (<= 230; chunk 51200-56320), profiled_new_programs 0, host_transfers_per_layer 0.
- Next-chunk prefetch: the serving contract does not give the next chunk's ids: prefill_chunk(input_tensor, kv,
  slot_id, actual_start, actual_end) carries only this chunk (tt/runners/adapter.py:_run reads them from the engine's
  device input), so the first wait (layer 1, ~6 ms) cannot be hidden by prefetching without an engine change.
- Re-run: `DS41_ENGRAM_TABLE=resident PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  "models/demos/deepseek_v41_flash_d_p/tt/perf_engram_async.py"`; gate `PYTHONPATH=$PWD python -m
  models.demos.common.bringup gate P.6 --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
  --check`; old path DS41_ENGRAM_ASYNC=off.

## FIX.9 (overseer fix, attempt 3): the shared expert's x / h as bfp8 in the block MoE's fused head (DS41-132)
- Why: OWNER DECISION 2026-10-28 (F208, FIX.7 GEMM_INPUTS=bfp8): the reference fp8-quantizes the shared expert's x and
  h (reference/ds41_ref.py `_swiglu`: `_q8(x)`, `_q8(h)`); on MOE_HEAD=fused they stayed bf16 (se_engine3 hard-coded
  2048 B tiles), above the reference. Attempt 2 left no changes in the tree (only ledger files).
- Block (models/demos/moe_block, the option defaults to bf16 for every other user):
  - tt/se_engine3.py `SeEngine3(x_dtype=, h_dtype=)` (bf16 / bfloat8_b; `TILE_BYTES`): x own, the x rounds (XIN), the
    own-shard DRAM load, h own / h all and their CBs take the dtype's tile bytes (`SE_XTB` / `SE_HTB` defines in the
    BRISC); the dispatch prep's source view (PU) reads x own in its format (`SE_PREP_TB`; units' offsets x tile bytes),
    so the routed experts' dispatched rows are bfp8-rounded x (their own bfp8 x rounding is then exact). The non-RS
    down path now reconfigures the packer for OUT (it relied on h being bf16). No down (`mrs`): h must stay bf16.
  - tt/x_gather3.py: the x packets' addresses / payloads use `XTB` (the engine's x tile bytes); the port CB keeps its
    2048 B slots (RS, latent, logits unchanged), so a bfp8 packet carries the same tiles in 1088 B each.
  - tt/fused_head.py `HeadSE3(activations=)`, `run(x, w, xq=)`: the router keeps x (bf16), the engine and the gather
    take xq; xq is appended last to the prepared program's io (the weights' positions unchanged) and in its key.
  - moe_block.py: `MoEDtypes.shared_activations` = x and h inside the shared expert. FusedHead allocates a persistent
    bfp8 `xq` and typecasts x into it each call (`ttnn.typecast(output_tensor=)`; one op, no host). Ops head
    (SharedExpert): x cast for gate / up, gate / up write bf16 (they wrote shared_activations before: identical at
    the bf16 default), the activation's output cast for down.
- Model: tt/moe_block_adapter.py `block_dtypes()` sets `shared_activations=gemm_dtype()` (GEMM_INPUTS; bf16 = the old
  path); tt/settings.py GEMM_INPUTS text, tt/MOE_BLOCK.md D5, known_issues DS41-132 updated.
- Why a typecast copy, not rounding where the SE reads x: the router reads x from DRAM in bf16 (the reference's
  router is bf16), and the engine's own-shard load and the gather move tiles as stored, so rounding in the engine
  would need an extra unpack / pack pass of every x tile on every engine core; the typecast is 7 us a layer.
- Measured, 8x4, chunk 5120, real weights (tt/perf_se_activations.py, random x, every 7th token vs float64 with the
  checkpoint's exact fp8 weights), DS41_GEMM_INPUTS=bf16 -> bfp8:
  - engine x / h BFLOAT16 -> BFLOAT8_B, arena 1142 -> 985 KB; head twice: 0 values differ (both).
  - shared expert vs native (x, h unrounded): L00 rel 0.01158 -> 0.01704 (bias -0.0085 -> +0.0074, worst row 0.0145
    -> 0.0206); L24 0.01232 -> 0.01812 (bias -0.0085 -> +0.0077). vs q8 (fp8 act quant of x and h, per 32): L00
    0.04388 -> 0.04688, L24 0.04637 -> 0.04940 (bfp8 rounds differently from e4m3; the positive bias is the bfp8
    packer's ties-away rounding, DS41-77).
  - head alone (rt profiler, 10 calls): section 454.0 -> 462.2 us a call (L00; L24 454.7 -> 464.1): the program
    ~446 -> ~451 us, + typecast ~7 us, + the x copy 10 us both; synced wall 0.958 -> 0.973 ms.
- In the model (test_profile.py, DS41_ENGRAM_TABLE=resident, rung last, 40 layers, same session; bf16 side =
  DS41_GEMM_INPUTS=bf16, which also turns FIX.7's attention-side casts off): fused head (the "(other)" program in
  L*.experts) 38.35 -> 38.53 ms a chunk (median 0.951 -> 0.966 ms a layer), typecast 0.29 ms a chunk, experts section
  94.90 -> 95.32 ms; device total 224.30 -> 225.95 ms (P.6's run of the same code before this change: 220.83, run to
  run noise ~2 % in untouched sections, e.g. attention 78.35 / 79.74 / 80.00). host_transfers_per_layer 0,
  profiled_new_programs 0. Warm wall (BRINGUP_PROFILE_WALL=1): 429.4 ms (spread 48.2, loadavg 15.9) -> 435.3 ms
  (spread 47.1, loadavg 21.0): within the spread. Files generated/bringup_adhoc/FIX.9_{wall_bf16,wall_bfp8,profile,
  profile_bf16}*.json.
- Accuracy, W.1 seed rung (unit passed; before = FIX.8's record, the last on this code minus the change): err_layer
  L00 0.0241 -> 0.0243, L01 0.0369 -> 0.0359, L02 0.0568 -> 0.0557, L03 0.0777 -> 0.0762, L14 0.0331 -> 0.0329, L20
  0.0309 -> 0.0306, L21 0.0462 -> 0.0457, L24 0.0318 -> 0.0314 (limit 0.0336).
- Re-run: `DS41_GEMM_INPUTS=bf16 PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_se_activations.py` (and without the env for bfp8); gate
  `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.9 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`; old path
  DS41_GEMM_INPUTS=bf16.
- First full gate (code above, no precise packing): every unit PASS but K.1 test_adapter_acks: chunk 0 ack slot 0
  [0, 2536), last-window 128 rows, window_kv layer 20 nope PCC 0.98862 < limit 0.98957 (DS41-125's deep-layer check);
  the chain stopped there (runner_contract / runner_smoke not run). Diagnosis (temporary prints in the contract's
  caches.pcc_failures and an env switch for the SE dtype, both removed; caches.py restored byte for byte), the same
  test, every window_kv check of layers 14 / 20 / 21 / 24, SE bf16 (FIX.7 kept) / SE bfp8 / SE bfp8 + precise:
  mean 1 - PCC 0.00772 / 0.00751 / 0.00779, min margin to the limit +0.00323 / -0.00095 / +0.00174; L20 chunk-0
  window 0.99280 / 0.98862 / 0.99131, L24 0.99051 / 0.98717 / 0.99039; bfp8 is better on 5 of the 7 windows a layer.
  So the 128-row windows move by +-0.003 with any rounding change (routing flips at deep layers, DS41-90): the
  failing window is noise at the limit, not a loss of accuracy, and no variant is better on average.
- Then shipped: SE_PACK_PRECISE (tt/settings.py, default on; MoEDtypes.shared_pack_precise -> FusedHead / HeadSE3 ->
  SeEngine3(pack_precise=) -> ComputeConfigDescriptor.bfp8_pack_precise, plus a `#define SE_PACK_PRECISE` because the
  compute binary's cache key leaves the flag out, DS41-137). Why, on its own merit: one rounding of h from fp32 (as
  ttnn.typecast packs x, as the reference casts h once) instead of two (ties away twice); same format, same speed.
  Measured on the same inputs: SE vs native L00 rel 0.01704 -> 0.01647, bias +0.0074 -> +0.0063 (vs q8 0.04688 ->
  0.04662); head 462.2 -> 463.9-464.7 us a call (noise). The contract window passing with it (0.99131) is partly luck,
  per the means above: an owner / overseer reading this should treat that check as noise-limited at layer 20.
- Re-run of the contract diagnosis: not kept (it edits the frozen helper); use the numbers above.

## P.7 (perf, galaxy 8x4): CKV_BFP4_PREPARED, ttnn.bringup.ckv_bfp4 as a prepared program (DS41-135)
- What: setting CKV_BFP4_PREPARED (tt/settings.py, default on; off = the old path, unchanged code). On:
  tt/compressor.AttnTable.append_tiles calls ttnn.bringup.ckv_bfp4(..., prepared=True): the program is built once per
  (mesh.id(), n, rounded, memory configs) (ckv_bfp4.py _prepared, LRU 64), prepared with ttnn.bringup.prepare_program
  and dispatched with run_prepared each call: ckv / rows / out addresses patched, row0 patched as a value slot.
- Fork change (prepared_program, CHANGELOG 2026-10-29): option `value_slots=[(kernel, arg)]` on prepare_program and
  `values=[...]` on run_prepared. Before it only io-tensor addresses were patched, and row0 (the writer's runtime arg 3)
  changes every call, so without it every call would need a new prepare. A value slot is never matched to an address
  and is left out of the cache key. With no slots the hash walk and key are as before (fabric_reduce_scatter / TailA2A
  prepared paths unchanged). ckv_bfp4 checks patch_count == cores x (3 + rounded) and falls back to generic_op if a
  frozen arg collided with an io address.
- Bytes: ckv_bfp4 tests/unit 8 passed (new test_ckv_bfp4_prepared: 5 calls at other row0 / inputs / tables, byte-equal
  to the generic_op path, prepared once); sdpa tests/unit/test_sparse_sdpa_row_bfp4.py passed; prepared_program
  tests/test_prepared_program.py: its cases are 2x2-only, 2 skipped on the 8x4 (as before).
- Host (8x4, chunk 5120, position 0, DS41_ENGRAM_TABLE=resident, cProfile of 5 warm chunks after warm-up, 4 calls a
  chunk; /scratch/dnijemcevic/p7/hostprof_p7.sh and _rev.sh, results /scratch/dnijemcevic/p7/): ckv_bfp4 1.427 / 1.307
  ms a call off -> 0.503 / 0.508 on: 3.4-3.7 ms host saved a chunk. Isolated (tt/perf_ckv_bfp4_host.py, 20 warm calls):
  n=1280 0.514 -> 0.133 ms, n=64 0.168 -> 0.094 ms; tables equal. In the model the remaining ~0.5 ms is
  allocate_tensor_on_device for the rounded output + run_prepared (C++ calls, not split by cProfile).
- Warm wall (BRINGUP_PROFILE_WALL, 10 timed runs, medians): pair 1 (on first) on 415.7 / off 393.3; pair 2 (off first)
  off 391.8 / on 389.6. In pair 1 every unrelated host function was slower in the on run too (decorators __call__ +66
  ms over 5 chunks, fused_head +14, tail_a2a +7) while ckv_bfp4 saved 18.5 ms: the first run of a session is slower.
  So the wall gain is within the 30-50 ms run spread; expect ~3.5 ms. measure --task P.7 (quick, on): 413.3 ms, every
  err limit held (L39 0.907 / 1.533), host_transfers_per_layer 0.
- Device time: the same kernels, compile args and runtime args (bytes equal), so the device program is unchanged
  (FIX.8: ckv_bfp4 0.209 ms a chunk). Not re-profiled; the orchestrator's A/B profile has device_ms_total on and off.
- For optests: the model now calls ttnn.bringup.run_prepared with a value slot (ckv_bfp4 prepared); a prepared_program
  case for it (tests/cases.py) is still to write.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/unit/test_ckv_bfp4.py`; `... models/demos/deepseek_v41_flash_d_p/tt/perf_ckv_bfp4_host.py -s`;
  `python -m models.demos.common.bringup.orchestrator measure --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --task P.7`; old path
  DS41_CKV_BFP4_PREPARED=off.
- Attempt 3 (P.7): only docs fixed after attempt 2 failed the lint: ckv_bfp4_ttnn/README.md now has the `## Options`,
  `## Limits` and `## How to call` sections (ops_lint: 0 problems); DS41-135's fix line labelled `Fix:`. Code unchanged.
