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


### 2026-07-23 08:30:02 UTC — Head-major output boundary

- Added an opt-in `output_head_major` KDA result `[B*H,T,V]` in TILE layout. The default token-major API remains unchanged. The aligned layer path applies per-head RMSNorm and output gating directly, then uses the existing TILE-native concat-heads primitive.
- Direct and composed regression is the 12-case command above; it covers default token-major, flat head-major, realistic T=32, padded T=4, decode, continuity, and both state dtypes.
- T=640 smoke: `PERF_REPS=2 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_chunk_kda_perf.py -q -s`. Result: `SAFE_PYTEST_RESULT: PASS`; one case passed in 4.21 s after the CB-race correction.
- Tracy report: `/tmp/kda_chunk_stable_headmajor_profile/reports/2026_07_23_08_30_02/ops_perf_results_2026_07_23_08_30_02.csv`. Ten warm calls averaged `0.825989 ms`, down `96.873 us` or `10.50%` from flat q/k/v token-major and `469.181 us` or `36.23%` from the original baseline.
- Stable prep/scan averaged `379.582 us` / `284.180 us`; the two remaining gate/beta transposes totaled `136.576 us`, and output untilize/permute were eliminated. Numerical stabilization adds about `38.6 us` to prep while the head-major boundary removes about `136.3 us` of output layout work.

### 2026-07-23 08:38:17 UTC — Flat vector-gate input

- Hypothesis: the remaining large wrapper transpose was the `[B,T,H,K]` vector gate; the prep reader can gather `[C,K]` directly from flat `[B,T,H*K]` without changing compute or scan tensors.
- Direct Blackhole matrix: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s` -> PASS, 5/5. Flat-gate output/state PCC was 0.999993/0.999994 at H=2,K=32,T=32 and 0.999994/0.999995 at H=32,K=128,T=64.
- Full regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> PASS, 12/12 in 11.73 s. Realistic T=32 layer output/state/conv PCC remained 0.999933/0.999870/0.999993.
- Tracy report: `/tmp/kda_chunk_flat_gate_profile/reports/2026_07_23_08_38_17/ops_perf_results_2026_07_23_08_38_17.csv`. Ten warm T=640 iterations total 688.8416 us/iteration: beta transpose 2.1607 us, reshape 25.6073 us, prep 376.6100 us, scan 284.4636 us. The gate transpose disappeared.
- Result: 137.1473 us (16.60%) faster than stabilized head-major 825.9889 us, and 606.3284 us (46.81%) faster than the original 1295.1700 us baseline.

### 2026-07-23 08:47:05 UTC — Exact doubling WY inverse

- Diagnosis: profiler RISC spans showed prep compute-active for about 369 of 377 us. Its masked 16x16 triangular solve spent 30 full 32x32 tile matmuls on quadrant data.
- Replaced it with the exact nilpotent identity `(I-N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)` for strictly-lower 32x32 `N`, requiring eight full-tile matmuls. Removed the superseded masked-quadrant helpers.
- Full Blackhole regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 12/12 in 15.94 s. Realistic T=32 output/state/conv PCC remained 0.999933/0.999870/0.999993.
- Tracy report: `/tmp/kda_chunk_doubling_paced_profile/reports/2026_07_23_08_47_05/ops_perf_results_2026_07_23_08_47_05.csv`. Ten warm T=640 calls averaged 640.8891 us: transpose 2.1554 us, reshape 25.7025 us, prep 328.8295 us, scan 284.2017 us.
- Result: 47.9525 us (6.96%) faster than flat-gate 688.8416 us, and 654.2809 us (50.52%) faster than the original 1295.1700 us baseline.
- A/B: removing the legacy three-tile startup read unexpectedly regressed prep to 359.6244 us (`/tmp/kda_chunk_doubling_clean_profile/reports/2026_07_23_08_46_09/ops_perf_results_2026_07_23_08_46_09.csv`). Restoring it recovered 328.8295 us, so it remains local and labeled as reader-burst pacing.


### 2026-07-23 09:04:04 UTC — Honor KDA compute fidelity

- Root cause: the phased prep/scan program factory hard-coded `HiFi4`, so the public `compute_kernel_config` and the layer's intended fidelity were silently ignored. The factory now maps the resolved config into both compute descriptors.
- Full build: `./build_metal.sh --build-ttnn` passed after correcting the architecture type to `tt::ARCH`.
- Hardware suite: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` passed 12/12 in 16.09 s. HiFi2 direct T=64 output/state PCC was 0.999928/0.999932.
- Controlled T=640 fidelity A/B: HiFi4 report `/tmp/kda_chunk_hifi4_control_profile/reports/2026_07_23_09_02_22/ops_perf_results_2026_07_23_09_02_22.csv` averaged 647.9074 us total (prep 335.1173 us, scan 285.0497 us); HiFi2 report `/tmp/kda_chunk_hifi2_profile/reports/2026_07_23_09_00_37/ops_perf_results_2026_07_23_09_00_37.csv` averaged 663.6048 us (prep 351.6653 us, scan 283.8808 us). HiFi4 is retained as the layer/perf default because it is 15.6974 us faster and more accurate.
- LoFi was rejected before profiling: T=64 output PCC was 0.998563, below the 0.999 acceptance floor.

### 2026-07-23 09:08:00 UTC — Keep complete value blocks on scan cores

- Hypothesis: splitting each head's four value tiles across two cores duplicated all value-independent reads and matmul setup, costing more than the extra column parallelism saved.
- Controlled T=640 A/B confirmed it. The 64-core value-split scan averaged 285.0497 us in `/tmp/kda_chunk_hifi4_control_profile/reports/2026_07_23_09_02_22/ops_perf_results_2026_07_23_09_02_22.csv`; the 32-core full-value scan averaged 182.1508 us in `/tmp/kda_chunk_scan_serial_profile/reports/2026_07_23_09_03_06/ops_perf_results_2026_07_23_09_03_06.csv`, 102.8989 us or 36.10% faster.
- Made one full-value core per head the default. `QWEN_GDN_SCAN_VALUE_SPLIT=1` retains the previous mapping solely as an explicit performance A/B knob until a larger-value crossover is measured.
- Full build and hardware suite: `./build_metal.sh --build-ttnn && scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 12/12 in 21.06 s.
- Confirmation report: `/tmp/kda_chunk_full_v_default_profile/reports/2026_07_23_09_07_51/ops_perf_results_2026_07_23_09_07_51.csv`. Ten warm T=640 calls averaged 553.9449 us: transpose 1.7821 us, reshape 25.4826 us, prep 343.9508 us, scan 182.7294 us.
- The confirmed default is 93.9625 us or 14.50% faster than the controlled 647.9074 us value-split baseline, and 741.2251 us or 57.23% faster than the original 1295.1700 us baseline.

### 2026-07-23 09:16:15 UTC — Move q/k squaring to SFPU

- Diagnosis: prep is compute-bound, and its in-kernel q/k normalization squared eight tiles per work item through binary matrix-FPU multiplies even though the elementwise operation does not require that unit. Existing RMSNorm kernels use SFPU for the same operation.
- Added a local SFPU destination-register multiply helper for q/k squares. The KDA algorithm and public API are unchanged.
- Direct Blackhole regression passed 5/5. Full regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 12/12 in 17.47 s. The HiFi2 T=64 flat-path output/state PCC remained 0.999926/0.999928.
- Tracy report: `/tmp/kda_chunk_sfpu_square_profile/reports/2026_07_23_09_13_38/ops_perf_results_2026_07_23_09_13_38.csv`. Ten warm T=640 calls averaged 530.8974 us: transpose 2.0228 us, reshape 25.7210 us, prep 321.2216 us, scan 181.9320 us. Prep improved 22.7292 us or 6.61% against the immediately preceding 343.9508 us profile; total improved 23.0475 us or 4.16%.
- Batching two fp32 tiles per destination-register acquisition was correct (5/5) but neutral at 321.2801 us prep, so the simpler single-tile helper was retained. The retained path is 764.2726 us or 59.01% faster than the original 1295.1700 us bringup baseline.

### 2026-07-23 09:22:00 UTC — Use the shared fp32 row reducer for q/k norms

- Diagnosis: q/k L2 normalization reduced each four-tile row by calling the generic matmul helper with an all-ones tile. The repository's shared reduction library already provides this exact operation with synchronization and destination-register handling specialized for reductions.
- Replaced the local matmul-based helper with `compute_kernel_lib::reduce` in accurate fp32 SFPU mode. Input lifetime remains caller-managed and the public algorithm/API are unchanged.
- Full Blackhole regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 12/12 in 17.65 s. HiFi2 T=64 output/state PCC was 0.999919/0.999918.
- Tracy report: `/tmp/kda_chunk_sfpu_reduce_profile/reports/2026_07_23_09_19_55/ops_perf_results_2026_07_23_09_19_55.csv`. Ten warm T=640 calls averaged 525.7882 us: transpose 1.8218 us, reshape 25.5303 us, prep 315.8257 us, scan 182.6104 us. Prep improved 5.3959 us or 1.68%; total improved 5.1092 us or 0.96% over the preceding SFPU-square profile.
- The documented fast tf32/FPU reduction mode passed 5/5 with slightly better PCC, but regressed prep to 325.8470 us (`/tmp/kda_chunk_fast_reduce_profile/reports/2026_07_23_09_21_06/ops_perf_results_2026_07_23_09_21_06.csv`), so accurate SFPU remains selected. The retained path is 769.3818 us or 59.40% faster than the original 1295.1700 us baseline.


### 2026-07-23 09:25:52 UTC — Target-shape full-layer profiler

- Added a trace-stable full-layer profiler at the Kimi target shape B=1,T=640,hidden=2304,H=32,K=V=128. It uses random initialization, external recurrent/convolution state, one warmup, and a configurable measured repetition count.
- Smoke command: `PERF_REPS=1 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_kda_layer_perf.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 12.18 s.
- Tracy report: `/tmp/kda_layer_t640_baseline_profile/reports/2026_07_23_09_25_52/ops_perf_results_2026_07_23_09_25_52.csv`. Three warm iterations averaged `5870.5100 us` of serialized device-kernel time.
- The largest groups were reshape/view `1835.592 us`, matmul `692.019 us`, slice `460.632 us`, untilize `442.856 us`, ternary `436.314 us`, tilize `425.282 us`, KDA prep `318.918 us`, and KDA scan `181.603 us`.
- Profiler-model aggregate utilization across rows with valid ideal-cycle data was `18.69%`. This is an operation-weighted diagnostic, not yet the formal layer roofline: custom KDA prep/scan rows have no ideal-cycle model and the report does not expose complete DRAM/NoC traffic.


### 2026-07-23 09:29:14 UTC — Keep aligned decay gate flat

- Diagnosis: the aligned chunk path reshaped the decay projection from `[B,T,H*K]` to `[B,T,H,K]`, applied pointwise bias/softplus/scale, then reshaped it back to the flat layout consumed by KDA prep. Each TILE-layout reshape cost about `604 us` at T=640.
- Added pre-expanded flat decay constants during weight loading and retained the projection flat only on the tile-aligned chunk path. Decode and padded prefill retain the original rank-4 compatibility path.
- Full Blackhole regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 12/12; reconfirmed immediately before commit in 12.55 s.
- Tracy report: `/tmp/kda_layer_flat_decay_profile/reports/2026_07_23_09_29_14/ops_perf_results_2026_07_23_09_29_14.csv`. Three warm T=640 iterations averaged `4641.3253 us`, down `1229.1847 us` or `20.94%` from the full-layer baseline.
- Profiler-model aggregate utilization increased from `18.69%` to `23.64%`. Reshape/view fell from `1835.592 us` to `627.498 us`; the remaining approximately `602 us` reshape is the output-gate flat-to-head boundary.


### 2026-07-23 09:51:10 UTC — Compute, DRAM, and CCL rooflines

- Committed analysis source is `ROOFLINE.md`; it uses the repository HiFi4 matrix ceiling (`152.064 TFLOP/s` at 110 cores, 1.35 GHz) and Blackhole DRAM ceiling (`512 GB/s`).
- Optimized B=1,T=640 full layer: `53.920 GFLOP` in `4641.325 us`, or `11.617 TFLOP/s` and `7.64%` whole-chip compute utilization. The five projections alone reach `47.86%`; QKV reaches `60.33%`.
- KDA prep moves `89.211 MB` and reaches `277.75 GB/s` (`54.25%` DRAM roofline); scan moves `72.352 MB` and reaches `398.00 GB/s` (`77.73%`). Both are below the `297 FLOP/byte` ridge and are data-movement dominated; scan is closest to its bandwidth ceiling.
- Added a real-time-profiler CCL benchmark mirroring sparse-MLA critical-path accounting. Command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_kda_ccl_perf.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- TP=8 BF16 `[1,1,640,2304]` all-reduce: payload `2.949 MB`, critical path `5.161 MB`, two-link LoudBox roofline `100 GB/s`, theoretical `51.610 us`, measured slowest-chip `219.169 us`, utilization `23.5%`. The standalone collective misses the `40%` aspiration; fused output-matmul + reduce-scatter is the next distributed path.


### 2026-07-23 09:47:36 UTC — Distribution crossover and TP=8 plan

- Committed plan: DISTRIBUTION_PLAN.md selects TP=8 whole-head sharding, no sequence parallelism, local complete states, row-parallel output projection, fused reduce-scatter preferred, and all-reduce fallback.
- Controlled T=640 scan A/B: H=4 four-way V split 96.132 us vs full-V 148.009 us (35.1% faster); H=8 split 138.286 us vs 149.206 us (7.3% faster); H=16 full-V 153.672 us vs split 260.952 us (41.1% faster); H=32 full-V remains 181.788 us.
- Production K=V=128 selection is four V blocks per head when local heads <=8, otherwise one complete V block per head. TP=8 maps 80 independent head-chunk prep items to 80 cores and four heads x four V blocks to 16 scan cores.
- Sequence parallelism is rejected for this phase because it inserts ordered state handoff on the scan dependency chain. The low-rank f_a and g_a projections remain replicated; beta and all head-width outputs are sharded.


### 2026-07-23 09:58:44 UTC — Apply the measured KDA scan crossover

- Localized the distribution rule to vector-gated KDA: split the four value tiles across four scan cores when `B*H <= 8`; retain one complete value block per head above the measured crossover. Scalar GDN keeps its established full-value mapping.
- `QWEN_GDN_SCAN_VALUE_SPLIT=0|1` remains an explicit A/B override. Without an override, H=4 now selects the TP=8 production mapping automatically.
- Full build: `./build_metal.sh --build-ttnn` passed.
- Hardware regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 12/12 in 10.77 s.
- Tracy confirmation: `/tmp/kda_chunk_h4_adaptive_profile/reports/2026_07_23_09_58_44/ops_perf_results_2026_07_23_09_58_44.csv`. Eleven H=4,T=640 scan calls averaged `95.940 us` (min `95.133 us`, max `97.238 us`), matching the forced-split crossover measurement (`96.132 us`) within `0.2%`.


### 2026-07-23 10:05:01 UTC — Whole-head TP=8 weight placement

- Mirrored the existing Qwen3.5 TP GDN contract but preserved KDA tensor semantics: each device receives corresponding Q/K/V head slices, replicated `f_a`/`g_a` low-rank factors plus its local beta slice, local decay/output-gate columns, local convolution taps, and a row shard of the output projection.
- A naïve shard of globally fused `[Q|K|V]` or `[f_a|g_a|beta]` is incorrect because it assigns projection families rather than corresponding heads. The loader now groups each device payload before applying `ShardTensorToMesh`.
- Eight-device layout test: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 7.15 s. It compares the physical tensor on every device against the exact expected host slice for fused QKV, fused auxiliary, output projection, and convolution taps.
- Single-device composed regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 7/7 in 8.19 s. Existing output/state PCC is unchanged; target decode output/state remained `0.999970` / `0.999968`.


### 2026-07-23 10:09:26 UTC — TP=8 local recurrence and output reduce-scatter

- The composed layer now derives a local config from the global head count, keeps Q/K/V, convolution, gates, recurrent state, norm, and output gating device-local, and requires caller-owned `TT_CCL` resources for TP execution.
- The row-parallel output projection produces one full-hidden partial per device. The current correctness path applies the existing minimal reduce-scatter and returns hidden-sharded `[B,T,hidden/TP]`; it is the unfused baseline for the planned matmul-reduce-scatter optimization.
- Eight-device command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 2/2 in 22.92 s. The distributed layer matched torch at output PCC `0.999955`, recurrent-state PCC `0.999892`, and convolution-state PCC `0.999997`.
- Single-device regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 7/7 in 8.02 s. Target decode output/state PCC remained `0.999970` / `0.999968`.


### 2026-07-23 10:19:04 UTC — Fused prefill output matmul + reduce-scatter

- Reused the production Qwen3.5 Blackhole `matmul_reduce_scatter_async` wrapper for TP prefill, including shared persistent buffers and disjoint matmul/CCL core rows. Decode retains the separate path.
- The first H=1,V=32 correctness case hung. Evidence rejected a generic timeout explanation: the same full layer passed with separate matmul + reduce-scatter, and the timeout appeared only when consuming the fused output. That case supplied one local K tile to an eight-column matmul grid, below the fused program mapping used by repository tests, so fusion is now gated on at least eight local K tiles.
- An eight-tile local-K retry still hung with `Topology.Linear`. Source inspection showed the validated P150x8 Qwen path and fused CCL tests use `Topology.Ring`. Changing only the topology to Ring made the identical shape pass, proving topology mismatch was the deadlock root cause. The safe test wrapper reset all eight devices after each hang.
- Command: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 4.79 s. Fused output PCC was `0.999949`; recurrent and convolution state PCC were `0.999914` and `0.999997`.


### 2026-07-23 10:24:46 UTC — Target-shape TP=8 fused profile

- Added an eight-device target-shape profiler for B=1,T=640,hidden=2304,H=32,K=V=128 with random initialization, caller-owned `TT_CCL`, one warmup, and three signposted repetitions.
- The first T=64 smoke failed before execution: Kimi's hidden shard gives `per_core_N=9`, while the shared Qwen fused helper selected `out_block_w=4`; the matmul validator requires exact divisibility. Selecting the largest divisor under the same half-width cap preserves Qwen's `20 -> 10` mapping and gives Kimi `9 -> 3`.
- Smoke command after the fix: `PERF_SEQ=64 PERF_REPS=1 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 7.31 s.
- Full profile command: `PERF_SEQ=640 PERF_REPS=3 python_env/bin/python3 -m tracy -p -r -o /tmp/kda_tp_layer_t640_fused_profile --check-exit-code --op-support-count 3000 -t 5000 -a device_kernel_duration -m "pytest models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s"` -> PASS, 1/1.
- Report: `/tmp/kda_tp_layer_t640_fused_profile/reports/2026_07_23_10_24_46/ops_perf_results_2026_07_23_10_24_46.csv`. Prep was 84.502 us slowest-chip median and scan was 96.336 us. The fused output matmul/reduce-scatter device medians span 140.160-176.089 us.
- The fused FP32 reduce-scatter has 5.160960 MB critical-path traffic and a 51.610 us two-link lower bound. Its 176.089 us slowest-device median is 29.3% effective fabric utilization, below the 40% aspiration and the 129.0 us target. Device imbalance, not recurrence mapping, is the next distribution sweep.
- Per-iteration device spans were 6.858, 5.496, and 5.605 ms; the signposted host interval averaged 6.401 ms. Only 1.20-1.27 ms/device was active kernel duration, proving that host dispatch gaps and unfused layout/pointwise boundaries dominate end-to-end latency.
- Combined hardware regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 9.71 s. TP output/recurrent/convolution PCC remained `0.999949`/`0.999914`/`0.999997`.
