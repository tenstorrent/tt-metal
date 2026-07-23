# Kimi Delta Attention bringup log

## Goals

- Build the Kimi Delta Attention (KDA) layer from the authoritative
  `moonshotai/Kimi-Linear-48B-A3B-Instruct` implementation.
- Use random initialization until checkpoint validation is explicitly needed.
- Reach PCC >= 0.98 on real Blackhole hardware with no CPU operations in the
  forward path.
- Characterize the production-mesh mapping and work toward approximately 60%
  compute-roofline utilization and 40% CCL-roofline utilization.

## Nomenclature

- **KDA**: Kimi Delta Attention, the per-key-dimension gated delta recurrence.
- **GDN**: Gated DeltaNet, the closest in-tree recurrence analog; it uses a
  scalar decay per value head rather than KDA's vector decay.
- **K**: query/key head dimension.
- **V**: value head dimension.
- **H**: number of query/key/value heads for Kimi Linear.
- **T**: sequence length.
- **State**: recurrent tensor with shape `[B, H, K, V]`.

## Decisions

### 2026-07-23 06:17:30 UTC — Trusted base and source boundary

- Branch `codex/kimi-linear-kda` starts from freshly fetched
  `origin/main@8ae1ef26e2fb963149d00f2a2cfe1725b8a9b3bc`.
- The existing `mvasilijevic/kda-bringup` branch and its worktree are
  untrusted and excluded from inspection, reuse, and comparison.
- Authoritative external inputs are:
  - Hugging Face model source/config:
    `moonshotai/Kimi-Linear-48B-A3B-Instruct`
  - MoonshotAI Kimi-Linear repository at
    `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`
  - Flash Linear Attention repository at
    `d1ce07369d581813553f30a750af3b6b5f9af6a9`
- User delegated gate decisions and requested uninterrupted autonomous
  execution. Each gate will still produce its required artifact and an
  evidence-backed decision in this ledger.

### 2026-07-23 06:17:30 UTC — Reuse shape

- **Fact:** current `origin/main` has no KDA implementation.
- **Fact:** current `origin/main` has a fully on-device Qwen3.6 Gated DeltaNet
  stack, including causal depthwise convolution, tensor-parallel prefill and
  decode, and the fused `ttnn.transformer.chunk_gated_delta_rule` operation.
- **Fact:** KDA and GDN share the same delta-rule state update, but KDA applies
  a distinct log-space decay to every key dimension.
- **Decision:** adapt the trusted GDN layer and fused-operation interfaces,
  adding only the per-key-dimension gate delta. Do not create an unrelated
  parallel model framework.

## Learnings

### 2026-07-23 06:21:44 UTC — Infrastructure map

- Mirror the public layer shape, weight/config separation, cache ownership,
  and tensor-parallel head ownership from
  `models/demos/blackhole/qwen36/tt/gdn/`.
- Use `models/experimental/gated_attention_gated_deltanet/` as the composed
  TTNN correctness oracle while the KDA primitive is being established.
- Reuse the launch/configuration structure of
  `ttnn.transformer.chunk_gated_delta_rule`, but not its scalar-gate
  contract.
- Existing fused GDN inputs carry decay as `[B, T, H]`; KDA requires
  `[B, T, H, K]`. The fused GDN kernels multiply the full `[K, V]` state by
  one scalar, so shape metadata alone cannot implement KDA.
- In recurrent decode, the isolated math delta is to reshape vector decay
  from `[B, H, K]` to `[B, H, K, 1]` before multiplying the state.
- Heads are recurrence-independent. The first production mapping will keep a
  complete `[K, V]` state for each locally owned head, use column-parallel
  input projections, and row-parallel output projection.

### 2026-07-23 06:17:30 UTC — Authoritative Kimi Linear contract

- Model dimensions: hidden size 2304, 32 KDA heads, K=V=128, causal
  convolution kernel size 4.
- The 27-layer model uses 20 KDA layers and 7 global MLA layers.
- Each of q, k, and v has an independent causal depthwise convolution followed
  by SiLU.
- Gate projection:
  `g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)`, with shape
  `[B, T, H, K]`.
- Delta strength:
  `beta = sigmoid(b_proj(x))`, with shape `[B, T, H]`.
- q and k are L2-normalized before the recurrence.
- Recurrent update:
  `S <- exp(g) * S + beta * k outer (v - S^T k)`;
  output is `q^T S / sqrt(K)`.
- Output applies a sigmoid-gated RMSNorm, flattens heads, then projects to the
  hidden size.
- Cache ownership is one convolution state per q/k/v path plus one recurrent
  `[B, H, K, V]` state per KDA layer.

### 2026-07-23 06:17:30 UTC — Source provenance

- `config.json` SHA-256:
  `a6ac3c2c4b5aa72370f9727f49ffa4432715d20061889acdb37c688be853096e`
- `configuration_kimi.py` SHA-256:
  `79422aca3ee6c89d201e0c15c4c9a6db517ba83d87ecdc4e41fa0f71297238d9`
- `modeling_kimi.py` SHA-256:
  `d79b365e37378881b9f1585007a56e236ca27a414920943cb85d1dacb75dda99`

### 2026-07-23 06:21:44 UTC — Known-good hardware baseline

- Hardware: one 8-device Blackhole LoudBox; 32 host CPUs; 755 GiB host RAM.
- `./build_metal.sh`: PASS, Release build and install completed.
- `./create_venv.sh`: PASS.
- Worktree-local import:
  `ttnn.__file__ = .../kimi-linear-kda/ttnn/ttnn/__init__.py`.
- Collection:
  `python -m pytest --collect-only -q
  models/demos/blackhole/qwen36/tests/unit/test_gdn.py` collected one test.
- Device command:
  `scripts/run_safe_pytest.sh
  models/demos/blackhole/qwen36/tests/unit/test_gdn.py -q -s`
- Result: `SAFE_PYTEST_RESULT: PASS`; one test passed in 61.20 s; GDN PCC
  `0.999183` on device 0.
- This proves the trusted worktree, build, Python environment, device access,
  and closest sibling implementation before KDA-specific changes.

### 2026-07-23 06:29:27 UTC — Disposable recurrence spike

- Scalar-degeneration invariant: PASS. Constant-over-K KDA decay reproduces
  the trusted torch GDN recurrence at `rtol=1e-5, atol=1e-6`.
- Device command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_recurrence_spike.py
  -q -s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; three tests passed in 13.45 s.
- T=1 output/state PCC: `0.999992` / `0.999997`.
- T=4 output/state PCC: `0.999991` / `0.999996`.
- Conclusion: `[B,H,K,1]` vector decay broadcasts over `[B,H,K,V]` state on
  Blackhole with existing device primitives. The multi-launch token loop is a
  correctness oracle only and is not a viable production performance path.

## Backlog

- Implement the proper independent torch reference and fully-on-device layer.
- Validate target-width depthwise convolution and recurrent-state dtype.
- Validate prefill, recurrent decode, cache continuity, and tensor-parallel
  behavior.
- Produce compute/CCL rooflines before committing to a distribution mapping.
- Profile warm steady state, classify compute/CCL/layout/elementwise costs,
  and optimize the measured dominant cost.

## Progress

### 2026-07-23 06:17:30 UTC

- Phase 0 framing complete.
- Phase 1 repository and upstream-source survey complete.
- Trusted Release build, venv, import, and sibling-device baseline complete.
- Phase 2 disposable vector-decay recurrence spike starting.
- Phase 2 feasibility proven on device.
- Prototype insights extracted to `tmp/design/kimi-kda-insights.md` before
  deletion.
- Phase 3 API and architecture design starting.

### 2026-07-23 06:35:00 UTC

- API contract written to `API_SPEC.md`.
- Design-extract insights and the complete design-review artifact written to
  `tmp/design/kimi-kda-insights.md` and `tmp/design/kimi-kda.md`.
- Alternatives reviewed: adapt trusted GDN boundaries with separate KDA ops;
  gate-rank-polymorphic GDN; permanent composed recurrence.
- Autonomous gate decision: approve the first alternative. It preserves the
  smallest existing public pattern while isolating vector-gate semantics.
- Hardest-to-change decision: partition whole heads and complete `[K,V]`
  states across devices, keeping recurrence free of collectives.
- Phase 3 production reference/config implementation starting.

### 2026-07-23 06:39:10 UTC — Independent specification

- Added immutable `KDAConfig` with authoritative model-config mapping and
  derived q/k/v widths.
- Added a pure-torch full-layer specification covering independent causal
  q/k/v convolution caches, vector gate, beta, recurrence, sigmoid-gated
  RMSNorm, output projection, and final state.
- The reference requires canonical Hugging Face weight names and exact shapes,
  including `A_log` shape `[1,1,H,1]`.
- Command:
  `python -m pytest -q
  models/experimental/kimi_delta_attention/tests/test_reference.py`.
- Result: 11 passed in 2.09 s.
- Covered target config mapping, invalid dimensions, causal-conv split
  equivalence, authoritative gate formula, scalar-GDN degeneration, sigmoid
  output gating, full-layer prefill/decode split equivalence, and exact weight
  validation errors.
- Command: `pre-commit run --files` over all four new Python files.
- Result: all applicable hooks passed.
- Phase 3 reference/config gate complete; composed full-device layer starting.

### 2026-07-23 06:42:11 UTC — Target-width native convolution

- Hypothesis: native depthwise `ttnn.conv1d` supports each Kimi q/k/v stream
  at D=4096 with kernel size 4 on Blackhole.
- Disposable spike used independent random BF16 input/history/weights and the
  pure-torch causal convolution as golden.
- Command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_conv_spike.py -q -s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; two tests passed in 11.36 s.
- T=1 output/state PCC: `0.999991` / `1.000000`.
- T=32 output/state PCC: `0.999992` / `1.000000`.
- Decision: native conv is a valid composed-prefill path. Keep the trusted
  explicit four-tap device FIR for decode until warm profiling compares both.
- The disposable spike was deleted after this evidence was recorded.

### 2026-07-23 06:50:46 UTC — Composed full-device layer

- Added fused q/k/v and auxiliary input projections, fused q/k/v causal FIR,
  exact vector-decay recurrence, sigmoid-gated RMSNorm, output projection, and
  persistent fused-convolution/recurrent state.
- Forward tests run under
  `ttnn.manage_config("throw_exception_on_fallback", True)`.
- First run failed before dispatch because TTNN `Shape` supports integer
  indexing but not `shape[:2]`; traceback pointed to `_validate_forward`.
- Fix: read batch and sequence with `shape[0]` and `shape[1]`. No device,
  layout, numerical, timeout, or resource hypothesis was involved.
- Passing command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; three tests passed in 28.41 s.
- T=1 output/recurrent/conv PCC:
  `0.999973` / `0.999952` / `0.999992`.
- T=4 output/recurrent/conv PCC:
  `0.999935` / `0.999966` / `0.999993`.
- Prefill output PCC: `0.999973`; following decode output PCC: `0.999882`.
- Final cache-continuity recurrent/conv PCC:
  `0.999961` / `0.999994`.
- Coverage: small production-aligned tile dimensions, nonuniform vector gate,
  full layer, output and both cache families, T=1/T=4, split execution, and
  fallback rejection. Not yet covered: target dimensions, long decode,
  chunk-parallel prefill, external trace-stable state, or mesh distribution.
- Exact post-dtype-enforcement rerun: all hooks passed and the same safe suite
  passed three tests in 4.20 s warm with identical PCC values.
- Persistent recurrent output is cast back to the configured FP32/BF16 dtype;
  external recurrent and convolution buffer dtypes are validated.
- `origin/main` advanced to `133e9563f37`; divergence was previewed, the two
  local commits were rebased cleanly, and the uncommitted layer was restored
  from a named stash. Upstream touched scalar remainder/unary code only.
- Post-rebase hooks passed; the full safe suite passed in 9.33 s with the same
  PCC values and fallback rejection enabled.
- Phase 3 composed correctness oracle complete; target-shape and state-dtype
  validation starting.

### 2026-07-23 07:12:31 UTC — Exact target-shape decode

- Hypothesis: the composed implementation scales without a layout, memory, or
  numerical failure to Kimi's exact decode geometry: hidden size 2304, 32
  heads, K/V head dimensions 128, and three 4096-wide convolution streams.
- Command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py::test_target_shape_decode_pcc
  -q -s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; one test passed in 14.43 s cold.
- Output PCC / max absolute error: `0.999955` / `2.698135e-02`.
- Recurrent-state PCC / max absolute error:
  `0.999964` / `1.624179e-02`.
- Convolution-state PCC / max absolute error:
  `0.999993` / `5.755138e-02`.
- This proves exact-shape one-token correctness with fallback rejection. It
  does not cover long-state accumulation, mesh distribution, or performance.

### 2026-07-23 07:06:04 UTC — Recurrent-state precision and ownership

- Source fact: Kimi initializes `A_log` from `log(uniform(1,16))`, but creates
  `dt_bias` with `torch.empty`; its generic initializer handles Linear and
  Embedding modules only. Without checkpoint weights, no checkpoint-like decay
  distribution can be inferred from random initialization.
- Controlled CPU experiment compared an FP32 cache with a cache quantized to
  BF16 after every token. Inputs were deterministic random tensors with
  B=1, H=2, K=V=32; decay was held constant to isolate retention sensitivity.
- At T=2048, output/state PCC for BF16 persistence was:
  `g=-1e-3`: `0.999945` / `0.999943`;
  `g=-1e-2`: `0.999969` / `0.999970`;
  `g=-1e-1`: `0.999994` / `0.999990`.
- At T=8192, output/state PCC and relative L2 error were:
  `g=-1e-4`: `0.999943` / `0.999937`, `1.087%` / `1.120%`;
  `g=-1e-3`: `0.999944` / `0.999937`, `1.054%` / `1.126%`.
- Decision: retain FP32 as the default recurrent-state dtype. BF16 remains an
  explicit memory/performance option, but it is not presumed accuracy-safe
  until real checkpoint activation statistics and end-to-end PCC are measured.
- Refactored deterministic config/weight construction into one shared test
  factory; this removes device-test dependence on private CPU-test helpers.
- Device command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py::test_external_state_is_updated_in_place
  -q -s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; FP32 and BF16 cases passed in 4.68 s.
- Both policies retained the exact recurrent and convolution buffer addresses
  across two decode calls and updated those external buffers in place.
- FP32 output/recurrent/convolution PCC:
  `0.999919` / `0.999975` / `0.999992`.
- BF16 output/recurrent/convolution PCC:
  `0.999919` / `0.999974` / `0.999992`.
- CPU reference command:
  `python -m pytest -q
  models/experimental/kimi_delta_attention/tests/test_reference.py`.
- Result: 11 passed in 2.10 s.
- Coverage now proves eager address stability and dtype behavior. Full TTNN
  trace capture/replay remains a separate graph-level validation.
- Full device regression command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; six cases passed in 7.25 s warm,
  including exact target decode, chunk/decode continuity, both external-state
  dtypes, address stability, and fallback rejection.

### 2026-07-23 07:22:46 UTC — Fused recurrent KDA

- Added a dedicated `ttnn.transformer.kda_recurrent_step` operation. It maps
  one complete head to one Tensix core, keeps the 128×128 FP32 state and
  scratch in local L1, broadcasts four decay-column tiles over state rows,
  and fuses decay, state read, beta-scaled rank-one write, and query.
- The private boundary accepts normalized/scaled q/k and exponentiated decay;
  those preprocessing operations remain device-side and separately profiled.
- `./build_metal.sh` completed successfully in Release mode, including TTNN
  Python binding compilation, link, and install.
- Direct operation command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_kda_recurrent.py -q -s`.
- The direct tests invoke each shape twice with distinct data, covering the
  program-cache runtime-address override as well as fallback rejection.
- Minimal H=2,K=V=32 output/state PCC across both seeds:
  `0.999985`/`0.999985` and `0.999998`/`0.999988`.
- Exact H=32,K=V=128 output/state PCC across both seeds:
  `0.999976`/`0.999983` and `0.999973`/`0.999985`.
- First full-layer integration attempt failed before kernel dispatch:
  validation reported `k_unit must be in DRAM`. Root cause was the trusted L2
  helper's deliberate L1 result for short sequences. Explicit DRAM
  materialization at the private fused boundary fixed that contract mismatch.
- Second integration attempt dispatched but failed output PCC (`0.103570`).
  Direct operation PCC remained correct, ruling out recurrence math. The layer
  had reshaped tiled `[B,T,H,K]` directly to `[BH,1,K]`; this reinterpreted one
  H×K tile matrix rather than materializing one matrix per head. Explicit
  T/H permutes before flattening and the inverse output permute fixed it.
- Exact target full-layer fused decode then passed with output/recurrent/conv
  PCC `0.999939` / `0.999950` / `0.999993`.
- Full regression command:
  `scripts/run_safe_pytest.sh
  models/experimental/kimi_delta_attention/tests/test_kda_recurrent.py
  models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; eight cases passed in 9.47 s warm.
- Coverage includes minimal and exact fused recurrence, program-cache replay,
  exact full-layer decode, composed T=4 chunk, prefill-to-fused-decode state
  continuity, FP32/BF16 external buffers, stable addresses, and no fallback.
- This proves correctness, not utilization. Warm recurrence/full-layer timing,
  graph trace replay, and preprocessing fusion remain performance gates.
- Post-format CPU reference command:
  `python -m pytest -q
  models/experimental/kimi_delta_attention/tests/test_reference.py`.
- Result: 11 passed in 1.96 s.
- Post-format device regression repeated the full command above without `-s`.
- Result: `SAFE_PYTEST_RESULT: PASS`; eight cases passed in 8.07 s.

### 2026-07-23 07:29:54 UTC — Recurrent-device roofline

- Profiled clean commit `03babc4b7fc` with 20 warm exact-shape calls:
  `python_env/bin/python3 -m tracy -p -r -o /tmp/kda_recurrent_profile
  --check-exit-code --op-support-count 1000 -t 5000
  -a device_kernel_duration -m "pytest
  models/experimental/kimi_delta_attention/tests/perf/test_kda_recurrent_perf.py
  -q -s"`.
- Tracy report result: mean `33.077 us`, median `32.991 us`, minimum
  `32.161 us`, maximum `34.803 us`, standard deviation `0.640 us`.
- The FP32 recurrent state is `32*128*128*4 = 2,097,152` bytes. Reading and
  writing it once moves at least 4 MiB per token, or `126.8 GB/s` at the
  measured mean. This is `24.8%` of the repository's 512 GB/s Blackhole DRAM
  ceiling (`ttnn/core/operation.cpp`).
- Counting decay, two state-vector products, the rank-one update, and the
  state add gives approximately 3,678,208 algorithmic FLOPs/token, or only
  `0.111 TFLOP/s` at the measured mean. The fused recurrent op is therefore
  state-traffic/dataflow bound, not compute bound.
- Decision: preserve this T=1 kernel as the correctness/decode primitive, then
  pursue the utilization target in a chunk-parallel KDA prefill op. The trusted


### 2026-07-23 08:01:07 UTC — Chunk-parallel KDA correctness

- Added `ttnn.transformer.chunk_kda`, reusing the phased GDN prep/scan scheduler with a vector-gate specialization. Prep factors `exp(G_i-G_j)` into per-key row scalings; scan carries the FP32 KxV recurrent state across 32-token chunks.
- First minimal hardware run hung. Triage showed the reader blocked reserving three WY-mask tiles while compute waited for those same three tiles. The shared `cb_u` alias had capacity `C*V=1`; sizing it to `max(C*V,3)` removed the reciprocal wait and also fixes that scalar-GDN edge case.
- The first completed run produced output PCC `0.949004`. FLA/source comparison proved the WY matrix must be `-strictly_lower(Akk)`; the vector path had inverted `diag(Akk)-Akk` without first masking the upper triangle. Adding the causal mask raised minimal output/state PCC to `0.999992` / `0.999995`.
- Direct hardware command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; two cases passed in 6.80 s. Exact two-chunk H=32,K=V=128,T=64 output/state PCC: `0.999993` / `0.999996`; max absolute errors: `6.757900e-04` / `8.301616e-03`.
- Chunk mode now routes through the fused primitive; recurrent mode retains `kda_recurrent_step`. The adapter restores TILE layout at its private boundary before RMSNorm.
- Full layer command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; six cases passed in 7.04 s, including chunk prefill, fused decode, cache continuity, exact target decode, and FP32/BF16 external state.
- CPU reference/factory command: `python_env/bin/python3 -m pytest models/experimental/kimi_delta_attention/tests/test_reference.py models/experimental/kimi_delta_attention/tests/test_factory.py -q`. Result: 11 passed in 2.45 s.
- Current coverage proves single-device functional correctness through two chunks. It does not yet establish T=640 latency, compute utilization, multi-device tensor parallelism, or CCL utilization.


### 2026-07-23 08:05:40 UTC — T=640 chunk baseline

- Added a direct exact-shape profiler harness at B=1,T=640,H=32,K=V=128 with BF16 q/k/v and FP32 vector gate, beta, and initial state.
- Smoke command: `PERF_REPS=2 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_chunk_kda_perf.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; one test passed in 4.16 s.
- Tracy command: `PERF_REPS=10 python_env/bin/python3 -m tracy -p -r -o /tmp/kda_chunk_profile --check-exit-code --op-support-count 1000 -t 5000 -a device_kernel_duration -m "pytest models/experimental/kimi_delta_attention/tests/perf/test_chunk_kda_perf.py -q -s"`.
- Report: `/tmp/kda_chunk_profile/reports/2026_07_23_08_05_40/ops_perf_results_2026_07_23_08_05_40.csv`. Ten warm calls averaged 1.295170 ms of serialized device-kernel time.
- Mean phased durations were 351.690 us for `ChunkGdnPrepOperation` and 285.298 us for `ChunkGdnScanOperation`; together they account for 49.2 percent of device time.
- Wrapper costs dominate the remaining 50.8 percent: five transposes total 466.276 us/call, output permute 79.266 us, untilize 56.953 us, scale 30.192 us, and reshape 25.497 us.
- Decision: preserve the validated prep/scan math and first remove token-major/head-major relayouts via a flat input/output path. Custom-op PM fields report zero/NaN utilization, so compute and CCL utilization require explicit work/traffic accounting after the layout path is reduced.


### 2026-07-23 08:12:08 UTC — Flat-value layout fast path

- Hypothesis: retaining the native flat value projection at the chunk boundary removes one T=640 token-major/head-major transpose without changing KDA numerics. The rank-4 compatibility path remains, and padded sequences retain it because the flat reader requires tile-aligned T.
- Direct hardware command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; three cases passed in 5.38 s. Rank-4 and flat exact H=32,K=V=128,T=64 paths both produced output/state PCC `0.999993` / `0.999996`.
- Full layer command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; six cases passed in 7.20 s, including the padded T=4 fallback and cache continuity.
- Profiler smoke: `PERF_REPS=2 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_chunk_kda_perf.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; one case passed in 2.92 s.
- Tracy report: `/tmp/kda_chunk_flat_v_profile/reports/2026_07_23_08_12_08/ops_perf_results_2026_07_23_08_12_08.csv`. Ten warm calls averaged `1.186095 ms` serialized device-kernel time, down `109.076 us` or `8.42%` from `1.295170 ms`.
- Mean prep/scan times remained stable at `353.214 us` / `284.964 us`. Four remaining transposes total `356.243 us`; output untilize/permute cost `56.997 us` / `78.941 us`. The measured delta matches the eliminated value transpose, validating the layout-cost diagnosis.


### 2026-07-23 08:18:47 UTC — Flat q/k with fused normalization

- Hypothesis: direct token-major q/k reads plus in-kernel L2 normalization will cost less than two head-split transposes and a standalone q scale. The implementation mirrors scalar GDN normalization, but uses KDA-safe scratch whose lifetimes end before WY inversion.
- Full TTNN build: `./build_metal.sh --build-ttnn`. Result: PASS.
- Direct hardware command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; four cases passed in 10.29 s. Exact flat q/k/v output/state PCC: `0.999994` / `0.999995`; max absolute errors: `8.402988e-04` / `9.960890e-03`.
- Full layer command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; six cases passed in 7.37 s.
- Profiler smoke: `PERF_REPS=2 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_chunk_kda_perf.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; one case passed in 2.88 s.
- Tracy report: `/tmp/kda_chunk_flat_qkv_profile/reports/2026_07_23_08_18_47/ops_perf_results_2026_07_23_08_18_47.csv`. Ten warm calls averaged `0.922861 ms`, down `263.233 us` or `22.19%` from flat-v and `372.309 us` or `28.75%` from the original baseline.
- Prep/scan averaged `341.005 us` / `284.850 us`. Two remaining transposes total `135.062 us`; output untilize/permute remain `56.833 us` / `79.474 us`.


### 2026-07-23 08:27:52 UTC — Stable realistic vector decay

- The first realistic aligned T=32 layer test failed despite synthetic direct tests passing: output/state PCC were `0.988784` / `0.773279`, and part of the state was exactly zero. A direct H=2 flat-q/k/v case passed at `0.999993` / `0.999994`, ruling out flat addressing and scan scheduling.
- Root cause: prep formed pairwise decay as `exp(G_i) * exp(-G_j)`. Real model gates accumulate to roughly -90 or below within a 32-token chunk, so the second factor overflowed even though the required causal difference `exp(G_i-G_j)` is finite. FLA likewise bounds exponent spans around an interior anchor.
- Fix: anchor the separable factors at `G_last/2`. `exp(G-anchor) * exp(anchor-G)` is algebraically identical, but each exponent spans at most half of the cumulative range. Scan-facing `exp(G)`, `exp(G_last)`, and `exp(G_last-G)` remain exact.
- A T=640 smoke initially hung because a raw Aqk intermediate was published through writer-facing `cb_intra`; with multiple work items per core, the writer became a competing consumer. Keeping raw Aqk private and publishing only the masked result removed the race.
- Realistic T=32 layer output/state PCC after the fix: `0.999933` / `0.999870`; convolution-state PCC: `0.999993`.
- Full device regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; 12 cases passed in 15.05 s.
