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


### 2026-07-23 10:41:10 UTC — Reject fused core-placement changes

- Hypothesis: the 36 us device-median spread in fused output came from matmul tile imbalance or CCL worker placement. Held Ring topology, two links, FP32 output, and tensor ownership fixed.
- Three-sample sweeps rejected 9x8/offset `(0,8)` at 191.184 us and 8x6/offset `(0,6)` at 192.304 us slowest-chip median. An 8x7/offset `(0,7)` run initially appeared better at 172.227 us versus the original three-sample 176.089 us, so it was retested with ten samples.
- Matched ten-sample reports: original 8x8 `/tmp/kda_tp_layer_t640_grid8x8_r10/reports/2026_07_23_10_41_10/ops_perf_results_2026_07_23_10_41_10.csv`; 8x7 `/tmp/kda_tp_layer_t640_grid8x7_r10/reports/2026_07_23_10_39_29/ops_perf_results_2026_07_23_10_39_29.csv`.
- The larger sample rejected 8x7: slowest-chip median was 172.950 us versus 166.069 us for 8x8. Effective CCL utilization is therefore 31.1%, still below the 40% aspiration. The original 8x8/offset `(0,8)` mapping remains selected.
- Horizontal offset `(1,7)` aborted and produced no timing. A subsequent `PERF_SEQ=64 PERF_REPS=1 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` passed 1/1 and reset all eight devices.
- Original 8x8 ten-sample layer result: last-nine median device span 5.749 ms, host signpost average 6.116 ms, and 1.20-1.23 ms/device summed active kernels. The evidence keeps layout/pointwise fusion and device trace capture ahead of further core redistribution.


### 2026-07-23 10:49:42 UTC — Trace the full TP=8 layer

- Hypothesis: the approximately 4.5 ms difference between the eager device span and summed kernels is host dispatch, so trace replay should approach the active-kernel floor without changing the KDA work map.
- The first trace smoke failed with `TT_FATAL: Writes are not supported during trace capture` at `ttnn.transformer.chunk_kda`. Source inspection proved the op omitted `eye`/`tril`/`ones`/`masks` fallback performs host uploads and is explicitly eager-only (`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:106-127,486-494`). This ruled out trace capacity and CCL deadlock hypotheses.
- Reused Qwen established layer-owned `build_fused_const_tiles` implementation (`models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py:57-82`) and passed those persistent mesh tensors into chunk KDA. Identical smoke command `PERF_TRACE=1 PERF_SEQ=64 PERF_REPS=2 scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` then passed 1/1 in 3.74 s.
- An initial implementation used the logical test configuration chunk size to build constants. The T=32 composed test then preserved only the first rows and failed at PCC 0.332364. The hardware adapter executes physical 32-token chunks (`tt/recurrence.py`), while that fixture intentionally has `config.chunk_size=4`; using the fused op `_FUSED_CHUNK_SIZE=32` fixed the exact test to output/state PCC 0.999965/0.999883. This proves constant-shape mismatch, not single-device mesh behavior, caused the regression.
- Full command: `PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 python_env/bin/python3 -m tracy -p -r -o /tmp/kda_tp_layer_t640_trace_r10 --check-exit-code --op-support-count 10000 -t 5007 -a device_kernel_duration -m "pytest models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s"` -> PASS, 1/1.
- Report: `/tmp/kda_tp_layer_t640_trace_r10/reports/2026_07_23_10_49_42/ops_perf_results_2026_07_23_10_49_42.csv`. After one warm replay, ten measured replays had a 1.263 ms median slowest-device critical path; the signposted host interval was 12.999 ms, or 1.300 ms/layer. Median summed kernels were 1.213-1.216 ms/device. The 4.55x speedup over the 5.749 ms eager span validates dispatch as the root cause.
- At 59.205 GFLOP/layer, trace sustains 46.89 TFLOP/s or 3.85% of the eight-chip HiFi4 peak by device span; host-observed throughput is 45.55 TFLOP/s or 3.74%. The 60% aspiration remains unmet and is not renormalized to active cores.
- Slowest-device medians were prep 84.602 us, scan 96.252 us, and fused output matmul + reduce-scatter 148.023 us. The latter reaches 34.9% of the two-link fabric roofline, still below the 40% target. Retain the committed 80-core/16-core/8x8 distribution and next remove layout round trips.
- Final hardware regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 9.06 s. TP output/recurrent/convolution PCC remained 0.999949/0.999914/0.999997.

### 2026-07-23 11:02:40 UTC — Replace shifted FIR with native depthwise conv1d

- Hypothesis: three unaligned shifted FIR windows cause the dominant untilize/tilize traffic; the trace-safe native depthwise conv1d pattern already used by Qwen GDN should remove it without changing KDA ownership.
- Added a host-held, whole-head-grouped `[Q|K|V,1,K]` weight prepared once per input length. Native conv is local to aligned B=1 chunk prefill; decode, short, batched, and padded inputs retain the general FIR.
- Single-device T=32 passed at output/recurrent/convolution PCC 0.999965/0.999884/0.999997. TP=8 trace smoke with T=64 passed 1/1, validating weight sharding and capture safety.
- Matched Tracy report: `/tmp/kda_tp_layer_t640_native_conv_r10/reports/2026_07_23_11_02_40/ops_perf_results_2026_07_23_11_02_40.csv`. Ten replays reduced device span 1.263 -> 0.987 ms, host time 1.300 -> 1.023 ms/layer, and active kernels 1.213-1.216 -> 0.940-0.942 ms/device.
- Native conv measured 26.081 us. Removed shifted-window relayouts saved about 274 us active time/device; mesh throughput rose to 59.99 TFLOP/s or 4.93% of eight-chip peak.
- Full hardware regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 13.05 s.

### 2026-07-23 11:08:44 UTC — Emit output gate head-major

- Hypothesis: a batched `[local_heads,128,128]` gate weight can broadcast the rank activation and emit `[local_heads,T,128]` directly, avoiding the measured head-alignment reshape/transpose.
- Single-device T=32 passed at output/recurrent/convolution PCC 0.999965/0.999884/0.999997. TP=8 T=64 trace smoke passed 1/1.
- Report: `/tmp/kda_tp_layer_t640_batched_gate_r10/reports/2026_07_23_11_08_44/ops_perf_results_2026_07_23_11_08_44.csv`. Ten replays reduced device span 0.987 -> 0.890 ms (9.8%), host time 1.023 -> 0.924 ms/layer, and active kernels to 0.844-0.845 ms/device.
- The batched matmul adds about 13 us but removes an 84.7 us reshape and about 22 us transpose. Mesh throughput is 66.50 TFLOP/s or 5.47% of peak.
- Full hardware regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 9.36 s.

### 2026-07-23 11:13:49 UTC — Fuse QKV and auxiliary input projections

- Hypothesis: QKV and auxiliary are projections of the same replicated hidden input, so grouping `[Q_local|K_local|V_local|f_a|g_a|beta_local]` per device removes one matmul launch without communication.
- T=32 PCC and TP=8 T=64 trace smoke passed. The TP weight test now checks the exact fused physical payload; dead separate QKV/auxiliary tensors were removed.
- Report: `/tmp/kda_tp_layer_t640_fused_input_r10/reports/2026_07_23_11_13_49/ops_perf_results_2026_07_23_11_13_49.csv`. Ten replays reduced device span 0.890 -> 0.874 ms and host time 0.924 -> 0.913 ms/layer. Matmul active time fell 51.3 us while slicing added 12.6 us; active kernels are 0.806-0.807 ms/device.
- Mesh throughput is 67.71 TFLOP/s or 5.57% of peak. Full hardware regression passed 9/9 in 15.37 s.

### 2026-07-23 11:22:50 UTC — Reject wider fused-output subblock

- Hypothesis: KDA T=640, local K=512, and `per_core_N=9` may benefit from a legal 1x3 output subblock despite Qwen traced-8k evidence favoring 1x1.
- The first mechanical A/B did not wire the parameter into the intended program and was discarded; its numbers are intentionally absent. A function-scoped diff was inspected before rerunning.
- Corrected report: `/tmp/kda_tp_layer_t640_outsub3_actual_r10/reports/2026_07_23_11_22_50/ops_perf_results_2026_07_23_11_22_50.csv`. Against matched 1x1, median device span regressed 0.87433 -> 0.87484 ms, slowest-chip fused time regressed 146.778 -> 147.706 us, and active time was unchanged.
- Restored 1x1. Qwen overlap diagnosis transfers to KDA; closing the CCL gap requires a different fused dataflow, not a wider matmul subblock.

### 2026-07-23 11:26:27 UTC — Fuse aligned chunk decay bias

- Hypothesis: supplying the pre-expanded decay bias and softplus activation to `ttnn.linear` will eliminate both following pointwise programs.
- Report: `/tmp/kda_tp_layer_t640_fused_decay_r10/reports/2026_07_23_11_26_27/ops_perf_results_2026_07_23_11_26_27.csv`. Ten replays reduced median device span 0.87433 -> 0.85469 ms (2.25%), host time 0.91280 -> 0.89225 ms/layer, and active kernels to 0.799-0.802 ms/device.
- Program counts partially reject the hypothesis: binary programs fell 24 -> 16 per layer across the mesh, while unary programs remained 32. The bias add is absorbed; softplus remains a device program.
- Mesh throughput is 69.27 TFLOP/s or 5.69% of peak by device span; host-observed throughput is 66.36 TFLOP/s or 5.45%.
- Full Blackhole regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 10.88 s. TP output/recurrent/convolution PCC was 0.999953/0.999910/0.999997.

### 2026-07-23 11:39:49 UTC — Reject scan common-input sharing

- Hypothesis: one V worker/head can read the six V-independent FP32 inputs once per chunk and distribute them from L1 to its three sibling V workers faster than four independent DRAM reads.
- The implementation used ProgramDescriptor semaphores and one bundled readiness/valid handshake per chunk. Full build passed; isolated custom-kernel tests passed 5/5. The exact TP=8 sharing branch passed at output/recurrent/convolution PCC 0.999953/0.999910/0.999997.
- Matched reports: sharing `/tmp/kda_tp_layer_t640_scan_share_r10/reports/2026_07_23_11_38_34/ops_perf_results_2026_07_23_11_38_34.csv`; control `/tmp/kda_tp_layer_t640_scan_noshare_r10/reports/2026_07_23_11_39_49/ops_perf_results_2026_07_23_11_39_49.csv`.
- Sharing regressed slowest-device scan time 97.387 -> 145.942 us (+49.9%), median layer critical path 0.85484 -> 0.90400 ms (+5.75%), and active kernels from 0.800-0.801 to 0.847-0.850 ms/device. Synchronization and L1 fan-out cost more than the removed DRAM reads.
- Reverted the implementation and retained the 16-core, four-V-worker/head scan with independent reads.

### 2026-07-23 11:49:38 UTC — Fold output-gate conversion into multiply

- Hypothesis: BinaryNg can consume the BF16 sigmoid gate and produce the required FP32 result directly, removing a standalone typecast without changing output numerics.
- Matched report: `/tmp/kda_tp_layer_t640_mixed_gate_r10/reports/2026_07_23_11_49_38/ops_perf_results_2026_07_23_11_49_38.csv`; control: `/tmp/kda_tp_layer_t640_scan_noshare_r10/reports/2026_07_23_11_39_49/ops_perf_results_2026_07_23_11_39_49.csv`.
- Ten replays reduced median slowest-device span 0.85484 -> 0.84802 ms (0.80%) and median active time 0.80034 -> 0.79339 ms/device. Across the mesh and samples, Typecast count fell 240 -> 160 and BinaryNg aggregate time fell 1.294 -> 1.151 ms.
- Mesh throughput is 69.82 TFLOP/s or 5.74% of the eight-chip HiFi4 peak. The result retains the existing core/tensor distribution.
- Full hardware regression: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 9.98 s. TP output/recurrent/convolution PCC was 0.999953/0.999910/0.999997.

### 2026-07-23 11:55:48 UTC — Produce the scaled decay gate in FP32

- Hypothesis: the decay-scale multiply can produce the prep-required FP32 tensor directly and eliminate the remaining chunk-input typecast.
- Report: `/tmp/kda_tp_layer_t640_fp32_decay_r10/reports/2026_07_23_11_55_48/ops_perf_results_2026_07_23_11_55_48.csv`; matched control: `/tmp/kda_tp_layer_t640_mixed_gate_r10/reports/2026_07_23_11_49_38/ops_perf_results_2026_07_23_11_49_38.csv`.
- Ten replays reduced median device span 0.84802 -> 0.84038 ms (0.90%) and program count 42 -> 41/device/layer. Typecast count across the mesh fell 160 -> 80.
- The FP32 multiply is slower: summed per-op kernel maxima rose 0.79339 -> 0.80205 ms/device. Prep and fused collective remained 84.05 and 146.31 us, while the serialized device span improved from the removed program boundary.
- Mesh throughput is 70.45 TFLOP/s or 5.79% of peak. Full hardware regression passed 9/9 in 10.26 s; TP output/recurrent/convolution PCC was 0.999952/0.999903/0.999997.

### 2026-07-23 12:00:29 UTC — Make convolution layout dataflow explicit

- Hypothesis: explicitly untilizing QKV and carry once, then concatenating row-major inputs, avoids concat/native-conv internal layout round-trips without requiring a custom kernel.
- Report: `/tmp/kda_tp_layer_t640_conv_rm_r10/reports/2026_07_23_12_00_29/ops_perf_results_2026_07_23_12_00_29.csv`; control: `/tmp/kda_tp_layer_t640_fp32_decay_r10/reports/2026_07_23_11_55_48/ops_perf_results_2026_07_23_11_55_48.csv`.
- Ten replays reduced median device span 0.84038 -> 0.70788 ms (15.8%), active time 0.80205 -> 0.67051 ms/device, and programs 41 -> 38/device/layer.
- Untilize-with-unpadding fell from three programs and 81.35 us/device/layer to one 2.67 us carry conversion; tilize-with-padding fell from two programs and 56.56 us to one 4.24 us carry conversion. The remaining QKV untilize is 39.04 us.
- Mesh throughput is 83.64 TFLOP/s or 6.88% of peak. Full hardware regression passed 9/9 in 10.47 s; TP output/recurrent/convolution PCC was 0.999952/0.999903/0.999997.

### 2026-07-23 12:07:07 UTC — Keep the convolution cache row-major

- Hypothesis: the layer-owned 3-token cache can stay row-major across aligned prefill and eliminate its untilize/re-tilize pair.
- The first full suite failed at recurrent T=1 because the reused FIR helper concat requires tile inputs (`ttnn_gated_deltanet.py:104`). TP chunk tests had passed. Converting at the legacy FIR boundary localized that requirement; the rerun passed 9/9 in 13.96 s with TP PCC 0.999952/0.999903/0.999997.
- Report: `/tmp/kda_tp_layer_t640_rm_cache_r10/reports/2026_07_23_12_07_07/ops_perf_results_2026_07_23_12_07_07.csv`; control: `/tmp/kda_tp_layer_t640_conv_rm_r10/reports/2026_07_23_12_00_29/ops_perf_results_2026_07_23_12_00_29.csv`.
- Ten replays reduced median device span 0.70788 -> 0.69876 ms (1.29%), active time 0.67051 -> 0.66295 ms/device, and programs 38 -> 36/device/layer.
- Mesh throughput is 84.73 TFLOP/s or 6.97% of peak. Prep/scan/fused-output medians remained 84.30/96.99/146.36 us.


### 2026-07-23 12:13:23 UTC — Precompose the aligned-prefill output gate

- Hypothesis: precomposing `g_b @ g_a` into output-sharded direct gate columns will make the fused input GEMM wider but remove the poorly utilized batched rank-128 gate GEMM and one program.
- Exact shard-layout checks and full Blackhole regression passed: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 10.47 s. TP output/recurrent/convolution PCC was 0.999952/0.999903/0.999997.
- Matched report: `/tmp/kda_tp_layer_t640_precomposed_gate_r10/reports/2026_07_23_12_13_23/ops_perf_results_2026_07_23_12_13_23.csv`; control: `/tmp/kda_tp_layer_t640_rm_cache_r10/reports/2026_07_23_12_07_07/ops_perf_results_2026_07_23_12_07_07.csv`.
- Ten replays reduced median slowest-device span 698.758 -> 690.719 us (1.15%), active time 662.921 -> 657.757 us/device, and programs 36 -> 35/device/layer.
- The fused input matmul grew from logical width 1796 to 2180 and cost 81.004 -> 87.713 us; removing the batched `128 -> 128` gate matmul saved 20.450 us, while extra slicing cost 6.661 us.
- The retained path executes 67.594 GFLOP across the mesh and reaches 97.86 TFLOP/s or 8.04% of peak; conservative useful throughput from the original factorized 59.205 GFLOP is 85.71 TFLOP/s or 7.05%. Tensor ownership and core/CCL distribution are unchanged.


### 2026-07-23 12:21:02 UTC — Reject BF16 partials and two CCL workers/link

- BF16 output partial hypothesis: halving fused reduce-scatter traffic may cross the 40% CCL target. Focused TP=8 hardware correctness rejected it: output PCC collapsed to 0.004862 versus the required 0.98. FP32 partials remain load-bearing.
- Worker hypothesis: two workers/link may improve fabric throughput enough to offset reserving four CCL rows. The temporary experiment changed the fused op default to two workers/link and paired it with an 8x6 matmul plus CCL offset `(0,6)`; build and TP correctness passed at output/recurrent/convolution PCC 0.999952/0.999903/0.999997.
- Matched report: `/tmp/kda_tp_layer_t640_rs_workers2_r10/reports/2026_07_23_12_21_02/ops_perf_results_2026_07_23_12_21_02.csv`; control: `/tmp/kda_tp_layer_t640_precomposed_gate_r10/reports/2026_07_23_12_13_23/ops_perf_results_2026_07_23_12_13_23.csv`.
- Two workers regressed fused-program time 151.333 -> 166.374 us, median layer span 690.719 -> 706.001 us, and active time 657.757 -> 673.180 us/device. Restored and rebuilt the one-worker 8x8 implementation.


### 2026-07-23 12:25:10 UTC — Slice auxiliary outputs directly

- Hypothesis: decay, direct output gate, and beta can slice from the fused projection directly, removing the enclosing auxiliary slice without changing layout.
- Full hardware regression passed: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 11.83 s. TP output/recurrent/convolution PCC was 0.999952/0.999903/0.999997.
- Report: `/tmp/kda_tp_layer_t640_direct_aux_slices_r10/reports/2026_07_23_12_25_10/ops_perf_results_2026_07_23_12_25_10.csv`; control: `/tmp/kda_tp_layer_t640_precomposed_gate_r10/reports/2026_07_23_12_13_23/ops_perf_results_2026_07_23_12_13_23.csv`.
- Ten replays reduced median slowest-device span 690.719 -> 683.463 us (1.05%), active time 657.757 -> 650.942 us/device, programs 35 -> 34/device/layer, and slice time 36.944 -> 30.514 us.
- Executed-work throughput is 98.90 TFLOP/s or 8.13% of peak; conservative factorized-work throughput is 86.62 TFLOP/s or 7.12%. Distribution is unchanged.


### 2026-07-23 12:27:51 UTC — Reject Conv1d-fused SiLU

- Hypothesis: the existing Conv1d packer activation can absorb the standalone approximately 12.9 us SiLU without changing KDA numerics.
- Focused TP=8 hardware correctness rejected the hypothesis: output PCC fell to 0.884267 versus the required 0.98.
- Restoring the standalone `ttnn.silu` restored output/recurrent/convolution PCC to 0.999952/0.999903/0.999997; `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py::test_kimi_delta_attention_tensor_parallel -q -s` ended with `SAFE_PYTEST_RESULT: PASS`.
- No Tracy comparison was run because the fused variant failed correctness. A future fusion must preserve the standalone operation's math rather than use the generic Conv1d activation hook.


### 2026-07-23 12:35:26 UTC — Fuse output sigmoid into multiply

- Hypothesis: BinaryNg can apply sigmoid to the gate operand before multiplying by the normalized recurrence output, eliminating one serialized unary program without the unsafe `z * sigmoid(z)` intermediate in Qwen's rejected fused-SiLU path.
- Full hardware regression passed: `scripts/run_safe_pytest.sh models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `SAFE_PYTEST_RESULT: PASS`, 9/9 in 11.20 s. TP output/recurrent/convolution PCC was 0.999952/0.999903/0.999997.
- Report: `/tmp/kda_tp_layer_t640_fused_sigmoid_mul_r10/reports/2026_07_23_12_35_26/ops_perf_results_2026_07_23_12_35_26.csv`; control: `/tmp/kda_tp_layer_t640_direct_aux_slices_r10/reports/2026_07_23_12_25_10/ops_perf_results_2026_07_23_12_25_10.csv`.
- Ten replays reduced median slowest-device span 683.463 -> 679.336 us (0.60%), active time 650.942 -> 646.984 us/device, and programs 34 -> 33/device/layer. Unary time fell 25.447 -> 20.773 us while binary time rose 29.277 -> 30.110 us.
- Executed-work throughput is 99.50 TFLOP/s or 8.18% of peak; conservative factorized-work throughput is 87.15 TFLOP/s or 7.16%. Distribution is unchanged.


### 2026-07-23 12:49:34 UTC — Fuse the gated-RMS epilogue

- Hypothesis: a KDA-local kernel can combine per-head RMS normalization, norm-weight broadcast, gate sigmoid/multiply, and head-major to token-major conversion, replacing three serialized programs without changing ownership.
- Full build passed: `./build_metal.sh --build-tests --build-type Release` -> exit 0, 379 targets and install completed.
- Final incremental build passed, followed by the nine-test TP/layer hardware gate: `SAFE_PYTEST_RESULT: PASS`, 9/9 in 13.61 s.
- Focused TP=8 hardware test passed at output/recurrent/convolution PCC `0.999964/0.999903/0.999997`. All functional KDA tests passed: the safe-pytest command over the six non-perf test files ended `SAFE_PYTEST_RESULT: PASS`, 27/27 in 21.23 s.
- The broader tests-directory run passed five fixtures before the single-device layer perf fixture failed inside Conv1d before the new epilogue: it requested 1,572,864 B per L1 bank with 1,436,800 B available. This profiler-fixture resource limit is recorded separately from functional coverage.
- Report: `/tmp/kda_tp_layer_t640_fused_gated_rms_r10/reports/2026_07_23_12_49_34/ops_perf_results_2026_07_23_12_49_34.csv`; control: `/tmp/kda_tp_layer_t640_fused_sigmoid_mul_r10/reports/2026_07_23_12_35_26/ops_perf_results_2026_07_23_12_35_26.csv`.
- Ten replays reduced median slowest-device span `679.336 -> 667.330 us` (1.77%) and programs `33 -> 31/device/layer`. The fused kernel costs 19.213 us; it removes LayerNorm at 12.015 us and NLP head concat at 10.269 us, while aggregate BinaryNg time falls 30.110 -> 20.941 us.
- Executed-work throughput is 101.29 TFLOP/s or 8.33% of peak; conservative factorized-work throughput is 88.72 TFLOP/s or 7.29%.
- The target epilogue distributes 80 `(head, 32-token block)` work items/device, with each core processing all four value tiles. TP head ownership, prep/scan mapping, and fused-output collective placement are unchanged.


### 2026-07-23 12:57:57 UTC — Fuse decay Softplus into multiply

- Hypothesis: applying Softplus to the projected decay inside its consuming FP32 scale multiply removes one serialized unary program while preserving the tensor passed to prep.
- The first focused run proved that BinaryNg requires parameterized Softplus (`UnaryWithParam(SOFTPLUS, 1.0, 20.0)`), not the bare enum. The first full-suite run then proved that its binding requires an empty activation sequence rather than `None` on the unchanged recurrent path. Both API-contract failures occurred before numerical comparison and were corrected locally.
- Focused TP=8 hardware PCC passed at output/recurrent/convolution `0.999964/0.999903/0.999997`. The corrected complete functional suite ended `SAFE_PYTEST_RESULT: PASS`, 27/27 in 17.39 s.
- Report: `/tmp/kda_tp_layer_t640_fused_softplus_mul_r10/reports/2026_07_23_12_57_57/ops_perf_results_2026_07_23_12_57_57.csv`; control: `/tmp/kda_tp_layer_t640_fused_gated_rms_r10/reports/2026_07_23_12_49_34/ops_perf_results_2026_07_23_12_49_34.csv`.
- Ten replays reduced median slowest-device span `667.330 -> 658.960 us` (1.25%), active time `639.418 -> 633.656 us/device`, and programs `31 -> 30/device/layer`. Unary time fell `20.702 -> 15.403 us`; BinaryNg rose only `20.940 -> 21.976 us`.
- Executed-work throughput is 102.58 TFLOP/s or 8.43% of peak; conservative factorized-work throughput is 89.85 TFLOP/s or 7.39%. Prep/scan/collective medians are unchanged, so distribution remains unchanged.


### 2026-07-23 13:06:00 UTC — Reject fused beta sigmoid-typecast

- Hypothesis: scalar-one BinaryNg can apply sigmoid and produce FP32 in one program, replacing unary sigmoid plus typecast.
- Focused TP=8 hardware correctness passed at output/recurrent/convolution PCC `0.999963/0.999902/0.999997`.
- Report: `/tmp/kda_tp_layer_t640_fused_beta_sigmoid_cast_r10/reports/2026_07_23_13_06_00/ops_perf_results_2026_07_23_13_06_00.csv`; control: `/tmp/kda_tp_layer_t640_fused_softplus_mul_r10/reports/2026_07_23_12_57_57/ops_perf_results_2026_07_23_12_57_57.csv`.
- Ten replays regressed median slowest-device span `658.960 -> 690.497 us` (4.79%) despite programs falling `30 -> 29` and active time remaining flat at `633.656 -> 633.211 us/device`. The approximately 31 us gap is serialized scheduling overhead not represented by summed kernel duration.
- Restored unary sigmoid plus FP32 typecast. The beta head split still requires transpose because all heads occupy columns of the same source tile; removing it requires in-kernel column selection, not a flat-reader address change.


### 2026-07-23 13:20:54 UTC — Measure TP=8 at 5120 tokens

- “5k” is measured as `T=5120`, preserving the 32-token chunk boundary.
- The first warm forward failed in native Conv1d before trace capture: static
  circular buffers requested 2,470,848 B/core against 1,572,864 B available.
  This proves a sequence-dependent L1 overflow and rules out trace-region,
  DRAM-capacity, recurrence, and CCL hypotheses.
- Reused the repository tested long-sequence Conv1d route: keep the known
  L1-full path through `T=640`, then auto-slice width from DRAM. Eager and trace
  smoke each passed 1/1 at `T=5120`.
- Full functional command:
  `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_kda_recurrent.py models/experimental/kimi_delta_attention/tests/test_reference.py models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 27/27 in 17.90 s.
- Measurement command:
  `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 python_env/bin/python3 -m tracy -p -r -o /tmp/kda_tp_layer_t5120_dram_width_slice_r10 --check-exit-code --op-support-count 10000 -t 5007 -a device_kernel_duration -m "pytest models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s"`
  -> PASS, 1/1.
- Report:
  `/tmp/kda_tp_layer_t5120_dram_width_slice_r10/reports/2026_07_23_13_20_54/ops_perf_results_2026_07_23_13_20_54.csv`.
  After one warm replay, ten measured replays have a 3692.501 us median
  slowest-device span, 3665.305 us corresponding active kernels, and 35
  programs/device/layer.
- The 540.752 GFLOP executed path sustains 146.45 TFLOP/s, 12.04% of the
  eight-chip HiFi4 peak. Conservative factorized throughput is 128.27 TFLOP/s,
  10.54%. The 60% aspiration remains unmet.
- Prep/scan/fused-output medians are 335.954/677.885/1043.920 us. Prep and
  gated-RMS each distribute 640 independent items across all 110 cores
  (90 cores get six, 20 get five); scan keeps four V splits/head on 16 cores
  and serially processes 160 chunks/core.
- The FP32 output partial is 47.185920 MB/device and has 41.287680 MB
  reduce-scatter critical-path traffic. Its 412.877 us lower bound versus
  1043.920 us measured combined time gives 39.55% effective CCL utilization,
  11.728 us short of the 40% target. Retain the 8x8 matmul/two-row Ring
  placement and one worker/link.


### 2026-07-24 15:49:24 UTC — Intermediate-storage dtype control

- Hypothesis: establish a fresh all-FP32 control after rebasing onto
  `origin/main` so subsequent BF16 storage experiments differ only in the
  prep-to-scan intermediates.
- Direct Blackhole regression:
  `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 26.92 s. The four HiFi4 cases produced
  output PCC `0.999993-0.999994` and state PCC `0.999994-0.999996`; the
  all-flat HiFi2 case produced output/state PCC `0.999919/0.999918`.
- Measurement command:
  `PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_15_49_24/ops_perf_results_2026_07_24_15_49_24.csv`
  (SHA-256 `106c501e09f50fc185046b96d0263c7f2107a49fdf446fdc5669a54748293976`).
  Ten measured replays have a `659.024 us` median slowest-device span
  (`656.328-660.744 us`), `637.981 us` median collapsed active time, and
  `637.921 us` collapsed operation-ledger time.
- Median block times are projection `88.528 us`, convolution `130.005 us`,
  split `19.064 us`, decay `33.163 us`, layout `9.299 us`, recurrence
  `183.096 us`, epilogue `19.377 us`, output `152.330 us`, and state
  `3.059 us`. This is the control for every storage-only comparison below.


### 2026-07-24 15:59:40 UTC — Store q_decay in BF16

- Hypothesis: `q_decay` is read-only scan input with no role in the persistent
  state update; halving its prep write and four-way scan read traffic should
  reduce recurrence latency without exposing the state to BF16 rounding.
- Added a private `QWEN_KDA_PREP_BF16_MASK` experiment path. It changes only
  selected prep-to-scan DRAM tensors and matching CB formats/page sizes; scan
  still unpacks into FP32-destination math. Public APIs, `intra`, `t_inv`,
  recurrent state, recurrence output, and final output remain FP32. Mask bit 2
  (`0x4`) selects `q_decay`.
- Build: `./build_metal.sh --build-tests --build-type Release` -> exit 0, 52
  incremental targets and install completed.
- Direct correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x4 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 21.96 s. Every printed output/state PCC
  matches the FP32 control to six decimals.
- TP=8 layer correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x4 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 34.58 s; output/recurrent/convolution
  PCC is `0.999964/0.999903/0.999997`.
- Measurement command:
  `QWEN_KDA_PREP_BF16_MASK=0x4 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_15_59_40/ops_perf_results_2026_07_24_15_59_40.csv`
  (SHA-256 `cfbbdadab83d8d9c80e9911e8ff50042483ab853389f4a4a87ca6c8aef5471f1`).
  Against the fresh FP32 control, median wall latency improves
  `659.024 -> 653.435 us` (`-5.590 us`, `-0.85%`), active time improves
  `637.981 -> 631.362 us`, and recurrence improves `183.096 -> 176.869 us`.
  Retain `q_decay` as the first accepted BF16-storage candidate.


### 2026-07-24 16:03:09 UTC — Reject standalone BF16 v_beta storage

- Hypothesis: halving the V-dependent `v_beta` prep write and sliced scan read
  traffic will reduce recurrence latency enough to improve the layer critical
  path. Mask bit 0 (`0x1`) selects `v_beta` only.
- Direct correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 18.96 s. HiFi4 output PCC is
  `0.999992` and state PCC is `0.999993-0.999995`; the HiFi2 case is
  `0.999918/0.999917`.
- TP=8 layer correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 6.85 s; output/recurrent/convolution
  PCC is `0.999962/0.999902/0.999997`.
- Measurement command:
  `QWEN_KDA_PREP_BF16_MASK=0x1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_16_03_09/ops_perf_results_2026_07_24_16_03_09.csv`
  (SHA-256 `40d4fe1afb57a6b48f8531acc4fd8730183a4ba997f8b752bc00a5a28d2480ff`).
  Recurrence improves only `183.096 -> 181.887 us`, while median wall regresses
  `659.024 -> 660.214 us` (`+1.189 us`, `+0.18%`) and active time is unchanged
  (`637.981 -> 637.933 us`). The small local saving is not a layer-level win;
  do not retain standalone BF16 `v_beta`.


### 2026-07-24 16:05:53 UTC — Reject BF16 k_dec_t on state fidelity

- Hypothesis: `k_dec_t` has the same tile volume as `q_decay`, so BF16 storage
  should give a comparable bandwidth win, but its direct contribution to the
  persistent state update makes numerical drift the deciding risk. Mask bit 4
  (`0x10`) selects `k_dec_t` only.
- Direct correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x10 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 19.79 s. HiFi4 output PCC is
  `0.999993-0.999994` and state PCC is `0.999993-0.999995`; the HiFi2 case
  improves slightly versus control to `0.999921/0.999922`.
- TP=8 layer correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x10 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 6.22 s; output/recurrent/convolution
  PCC is `0.999964/0.999899/0.999997`. The recurrent-state result crosses below
  the `0.999900` retention guard despite passing the repository test threshold.
- Measurement command:
  `QWEN_KDA_PREP_BF16_MASK=0x10 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_16_05_53/ops_perf_results_2026_07_24_16_05_53.csv`
  (SHA-256 `6eb6e45c802a9da732244c4fafa323d60464e18b0c367da4ab84c4242e6519fd`).
  Median wall improves `659.024 -> 653.832 us` (`-5.192 us`, `-0.79%`), active
  time improves `637.981 -> 633.222 us`, and recurrence improves
  `183.096 -> 178.550 us`. Reject as a retained default because this bandwidth
  win changes the persistent state beyond the selected fidelity guard.


### 2026-07-24 16:08:43 UTC — Store kd in BF16

- Hypothesis: `kd` has the same tile volume as `q_decay`; despite participating
  in the state correction, its BF16 pack/unpack may preserve the effective
  recurrence numerics because its source products are already BF16-limited.
  Mask bit 1 (`0x2`) selects `kd` only.
- Direct correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x2 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 22.42 s. Every HiFi4 output/state PCC
  matches the FP32 control to six decimals; the HiFi2 case is
  `0.999920/0.999921`.
- TP=8 layer correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x2 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 7.17 s; output/recurrent/convolution
  PCC exactly matches control at `0.999964/0.999903/0.999997`.
- Measurement command:
  `QWEN_KDA_PREP_BF16_MASK=0x2 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_16_08_43/ops_perf_results_2026_07_24_16_08_43.csv`
  (SHA-256 `27b27fa049287b66bc91e831edf9e68e90d5f4f395d6e59aa75ac90658cb9d62`).
  Median wall improves `659.024 -> 652.986 us` (`-6.039 us`, `-0.92%`), active
  time improves `637.981 -> 631.827 us`, and recurrence improves
  `183.096 -> 177.339 us`. Retain `kd` as an accepted BF16-storage candidate.


### 2026-07-24 16:11:37 UTC — Store dl in BF16

- Hypothesis: vector `dl` is only one K-by-1 tile column per chunk, so BF16
  storage should be numerically safe but yield at most a small traffic win.
  Mask bit 5 (`0x20`) selects `dl` only.
- Direct correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x20 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 20.18 s. HiFi4 output/state PCC is
  `0.999993-0.999995`; the HiFi2 case is `0.999919/0.999917`.
- TP=8 layer correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x20 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 6.21 s; output/recurrent/convolution
  PCC exactly matches control at `0.999964/0.999903/0.999997`.
- Measurement command:
  `QWEN_KDA_PREP_BF16_MASK=0x20 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_16_11_37/ops_perf_results_2026_07_24_16_11_37.csv`
  (SHA-256 `b260736cf9c3aec4f12e12551b13b4ba56dafcacbc9a4b343f3722f97222f71f`).
  Median wall improves `659.024 -> 657.743 us` (`-1.281 us`, `-0.19%`), active
  time improves `637.981 -> 637.203 us`, and recurrence improves
  `183.096 -> 181.890 us`. Retain `dl` as a small, numerically clean candidate
  for combination testing; it is not meaningful enough to select alone.


### 2026-07-24 16:14:31 UTC — Combine BF16 kd and q_decay storage

- Hypothesis: the two individually clean, equal-volume tensors should combine
  additively because prep writes and four-way scan reads are independent. Mask
  `0x6` selects `kd` and `q_decay`.
- Direct correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x6 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 20.65 s. HiFi4 PCC matches the FP32
  control to six decimals; the HiFi2 case is `0.999921/0.999921`.
- TP=8 layer correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x6 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 6.75 s; output/recurrent/convolution
  PCC exactly matches control at `0.999964/0.999903/0.999997`.
- Measurement command:
  `QWEN_KDA_PREP_BF16_MASK=0x6 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_16_14_31/ops_perf_results_2026_07_24_16_14_31.csv`
  (SHA-256 `78b48a3d637defa1edd8228db431d92b05a09d1e9636ea60e5df5e3b4d3475f1`).
  Median wall improves `659.024 -> 646.784 us` (`-12.240 us`, `-1.86%`), active
  time improves `637.981 -> 626.043 us`, and recurrence improves
  `183.096 -> 171.047 us`. The result is approximately additive and becomes
  the leading retained candidate.


### 2026-07-24 16:17:28 UTC — Add BF16 dl to the retained pair

- Hypothesis: adding the numerically clean `dl` storage change to the leading
  `kd + q_decay` pair may recover its small standalone saving without a state
  penalty. Mask `0x26` selects `kd`, `q_decay`, and `dl`.
- Direct correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x26 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 20.07 s. HiFi4 output/state PCC is
  `0.999993-0.999996`; the HiFi2 case is `0.999921/0.999919`.
- TP=8 layer correctness:
  `QWEN_KDA_PREP_BF16_MASK=0x26 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1 in 6.58 s; a concise confirmation rerun
  reports output/recurrent/convolution PCC `0.999964/0.999903/0.999997`.
- Measurement command:
  `QWEN_KDA_PREP_BF16_MASK=0x26 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1.
- Report:
  `generated/profiler/reports/2026_07_24_16_17_28/ops_perf_results_2026_07_24_16_17_28.csv`
  (SHA-256 `09d6eebffdd8a38346e3c35bb65d4061f05d9a438aebc5d3052df42979adb873`).
  Median wall improves `659.024 -> 643.623 us` (`-15.401 us`, `-2.34%`), active
  time improves `637.981 -> 621.755 us`, and recurrence improves
  `183.096 -> 167.623 us`. This beats the `0x6` pair by `3.161 us` wall with
  unchanged TP PCC, so retain `0x26` as the selected storage policy.


### 2026-07-24 16:24:06 UTC — Select mixed storage as the KDA default

- Decision: make mask `0x26` the KDA default. `kd`, `q_decay`, and `dl` cross
  the prep/scan boundary in BF16; `v_beta`, `k_dec_t`, `intra`, `t_inv`,
  recurrent state, and public outputs remain FP32. The private
  `QWEN_KDA_PREP_BF16_MASK` override remains available for exact FP32 replay
  (`0`) and controlled experiments.
- Host build:
  `./build_metal.sh --build-tests --build-type Release`
  -> exit 0; 52 targets installed.
- Final correctness gate:
  `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_factory.py models/experimental/kimi_delta_attention/tests/test_kda_recurrent.py models/experimental/kimi_delta_attention/tests/test_reference.py models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 27/27 in 74.95 s. This covers direct KDA,
  program-cache reuse, TP=8 weight/layout and layer PCC, composed decode and
  prefill, cache continuity, and FP32/BF16 external-state updates.
- Matched long-context control:
  `QWEN_KDA_PREP_BF16_MASK=0 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1. CSV
  `generated/profiler/reports/2026_07_24_16_22_50/ops_perf_results_2026_07_24_16_22_50.csv`
  (SHA-256 `25759170be81e5186552a43dc7f531c13223b16c6307a29880c3886d1a3ac2c3`).
- Matched long-context selected default:
  `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 1/1. CSV
  `generated/profiler/reports/2026_07_24_16_24_06/ops_perf_results_2026_07_24_16_24_06.csv`
  (SHA-256 `b3fd2c8ff3df64fecdb5d2b977858fb56c12a8cc58271b0ce37e7fe687f98a02`).
- Ten-replay medians at T=5,120: wall latency improves
  `3681.529 -> 3584.949 us` (`-96.581 us`, `-2.62%`), active time improves
  `3679.696 -> 3573.748 us`, and recurrence improves
  `1023.411 -> 920.831 us` (`-102.581 us`, `-10.02%`). The fixed-work roofline
  utilization estimate consequently rises `12.07% -> 12.40%`.
- Attribution: projection is `524.704 -> 524.619 us`, convolution is
  `602.371 -> 601.621 us`, and fused output/CCL is
  `1069.479 -> 1067.788 us`; these are effectively unchanged. The measured
  gain is localized to recurrence storage traffic and does not alter the
  TP=8 distribution or CCL algorithm.


### 2026-07-24 17:22:41 UTC — Affine-prefix feasibility and cost experiment

- Hypothesis: each KDA chunk can be represented as
  `S_out = A_chunk @ S_in + B_chunk`, allowing chunk states to be computed by
  an associative parallel-prefix scan instead of the serial 160-chunk loop.
- Algebra:
  `(A2, B2) compose (A1, B1) = (A2 @ A1, A2 @ B1 + B2)`.
  Once the exclusive chunk prefixes produce each `S_in`, token outputs inside
  every chunk can be reconstructed independently.
- Reproducible CPU test:
  `python_env/bin/pytest models/experimental/kimi_delta_attention/tests/test_affine_prefix.py -q -s`
  -> 3/3 passed in 1.56 s. The tests prove associativity in FP64, reproduce
  chunk-entry states and outputs, and compare a balanced FP32 prefix with the
  serial token recurrence.
- Target-shape numerical study: K=V=128, C=32, 160 chunks, four heads/chip.
  The FP32 balanced prefix produced PCC `1.000000000`, maximum absolute error
  `3.874302e-7`, and RMSE `5.075655e-8` against the serial recurrence.
- Target-shape cost model: padding 160 chunks to 256 requires 510 Blelloch
  compositions. Each head performs `4.278 GFLOP` of prefix composition;
  four heads perform `17.11 GFLOP/chip`. Storing A+B in FP32 requires
  `80 MiB/chip`.
- DRAM verdict: reject. Reading four matrices and writing two for each
  composition produces approximately `780 MiB/chip` of level traffic, whose
  512 GB/s lower bound is at least `1.5 ms`. That already exceeds the measured
  `606.168 us` serial scan before constructing chunk outputs.
- Retained direction: a distributed-L1 prefix remains mathematically viable,
  but is deferred until a storage and level-traffic design demonstrates a
  lower bound below the serial scan. Direct prep-to-scan streaming and static
  common-input multicast now rank above it.


### 2026-07-24 17:36:05 UTC — Retain prepared tensors in distributed L1

- Hypothesis: prep/scan boundary traffic is a material recurrence bottleneck,
  and all seven prepared tensors fit in aggregate interleaved L1 even at
  T=5,120. Keeping the existing two-program boundary isolates this traffic
  effect from producer/consumer fusion.
- Implementation: `chunk_kda` now gives prep an interleaved L1 memory config;
  scan and public outputs retain their requested memory config. The private
  `QWEN_KDA_PREP_DRAM=1` override preserves a DRAM A/B control.
- Host build: `./build_metal.sh --build-tests --build-type Release`
  -> exit 0; 51 targets installed.
- Direct correctness:
  `QWEN_KDA_PREP_L1=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s`
  -> `SAFE_PYTEST_RESULT: PASS`, 5/5 in 14.59 s. Target-shape output/state PCC
  is `0.999993/0.999995`.
- T=640 DRAM control CSV:
  `generated/profiler/reports/2026_07_24_16_17_28/ops_perf_results_2026_07_24_16_17_28.csv`
  (SHA-256 `09d6eebffdd8a38346e3c35bb65d4061f05d9a438aebc5d3052df42979adb873`).
- T=640 L1 command:
  `QWEN_KDA_PREP_L1=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> PASS. CSV `generated/profiler/reports/2026_07_24_17_31_15/ops_perf_results_2026_07_24_17_31_15.csv`
  (SHA-256 `299f6b2df589c3e8c0fb1085512740e9e75db74be75edf373afb469edbe1afad`).
  Median wall improves `643.623 -> 622.564 us` (`-21.059 us`, `-3.27%`);
  recurrence improves `167.623 -> 144.945 us` (`-13.53%`).
- T=5,120 L1 command:
  `QWEN_KDA_PREP_L1=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> PASS. CSV `generated/profiler/reports/2026_07_24_17_33_14/ops_perf_results_2026_07_24_17_33_14.csv`
  (SHA-256 `904542228cf8cd9a67018a01156b54570f8bae24a96d3d7f07c1ae2a9f2ead6a`).
  Relative to the matched FP32 DRAM control, median wall improves
  `3681.529 -> 3474.029 us` (`-207.500 us`, `-5.64%`) and recurrence improves
  `1023.411 -> 809.704 us` (`-20.88%`).
- Attribution: T=5,120 projection is `524.704 -> 524.798 us`, convolution is
  `602.371 -> 601.588 us`, and output/CCL is `1069.479 -> 1069.975 us`.
  Their stability localizes the gain to recurrence and rejects a run-wide
  clock or scheduling shift as the explanation.
- TP=8 validation:
  `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> PASS; output/recurrent/convolution PCC `0.999964/0.999903/0.999997`.
- Final gate: the established six-file `scripts/run_safe_pytest.sh --run-all`
  suite -> `SAFE_PYTEST_RESULT: PASS`, 27/27 in 27.23 s.
- Verdict: accept and make L1 prep residency the default. The capacity
  hypothesis is confirmed through T=5,120; no numerical regression was
  observed. Static multicast of V-independent scan inputs is next.


### 2026-07-24 17:48:13 UTC — Reject one-tensor `k_dec_t` multicast

- Hypothesis: one sender per head can read and multicast the 16 KiB `k_dec_t`
  block to its three V-worker peers more cheaply than four independent reads.
  Four-worker groups were row-aligned; one ready/valid semaphore pair was
  reused once per chunk. The path was opt-in with
  `QWEN_KDA_SCAN_SHARE_KDEC=1`.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> exit 0.
- Target-shape correctness:
  `QWEN_KDA_SCAN_SHARE_KDEC=1 scripts/run_safe_pytest.sh --run-all test_chunk_kda.py::test_chunk_kda_pcc[...] -q -s`
  -> PASS; output/state PCC `0.999993/0.999995`.
- T=640 profile:
  `QWEN_KDA_SCAN_SHARE_KDEC=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> PASS. CSV `generated/profiler/reports/2026_07_24_17_48_10/ops_perf_results_2026_07_24_17_48_10.csv`
  (SHA-256 `c3a0a020eff19f618310c082d2f8199e43b18e1783ada94b0b4e86da6f5ac220`).
- Result: median wall regresses `622.564 -> 630.371 us` (`+7.807 us`,
  `+1.25%`) and recurrence regresses `144.945 -> 154.716 us` (`+6.74%`).
  Projection and convolution are unchanged (`88.462 -> 88.459 us`,
  `129.866 -> 129.866 us`), localizing the regression to scan sharing.
- Verdict: reject one-tensor sharing and remove its code. The evidence supports
  fixed per-chunk synchronization cost exceeding the saved `k_dec_t` reads.
  A single handshake amortized across all six common inputs remains a distinct
  hypothesis and is the next experiment.


### 2026-07-24 17:52:17 UTC — Reject row-aligned scan placement alone

- Alternative hypothesis: moving each four-worker head group into one NoC row,
  rather than synchronization, caused the one-tensor multicast regression.
- Command: `QWEN_KDA_SCAN_ROW_ALIGN=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS.
- CSV: `generated/profiler/reports/2026_07_24_17_52_13/ops_perf_results_2026_07_24_17_52_13.csv` (SHA-256 `f89d35b501583fe5f7eb673b0f30a53a26da52f79d3999a01701de93f9aa3996`).
- Result: wall is `622.564 -> 621.223 us` (`-0.22%`, within run noise);
  recurrence is `144.945 -> 145.140 us` (`+0.13%`).
- Verdict: reject placement-only change because it misses the 1% wall gate.
  It also rules out placement as the source of the prior `+9.77 us` recurrence
  regression, supporting per-chunk synchronization as the cause.


### 2026-07-24 17:56:15 UTC — Reject batched common-input multicast

- Hypothesis: one ready/valid handshake can be amortized across all six
  V-independent inputs (48 KiB per head/chunk), unlike the failed 16 KiB
  `k_dec_t`-only prototype. The sender issued all source reads, then six linked
  multicasts; receivers reserved all six CBs before one readiness signal.
- Build: `./build_metal.sh --build-tests --build-type Release` -> exit 0.
- Target correctness with `QWEN_KDA_SCAN_SHARE_COMMON=1` -> PASS; output/state
  PCC `0.999993/0.999995`.
- Profile command: `QWEN_KDA_SCAN_SHARE_COMMON=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS.
- CSV: `generated/profiler/reports/2026_07_24_17_56_12/ops_perf_results_2026_07_24_17_56_12.csv` (SHA-256 `2022582aa393c14ff8d417359bc09b66a08c913a801c4ae4d160c44b9062540a`).
- Result: wall regresses `622.564 -> 657.290 us` (`+34.726 us`, `+5.58%`);
  recurrence regresses `144.945 -> 181.216 us` (`+25.02%`).
- Verdict: reject and remove the prototype. Row alignment was previously shown
  neutral, so sender serialization plus runtime synchronization/fan-out is
  the supported cause. Static multicast leaves the ranked queue; barrier
  batching without inter-core synchronization is next.


### 2026-07-24 18:00:01 UTC — Reject global scan-read barrier collapse

- Hypothesis: issue all seven per-chunk input reads before one NoC barrier,
  replacing seven read barriers without inter-core synchronization.
- Build: `./build_metal.sh --build-tests --build-type Release` -> exit 0.
- Target correctness with `QWEN_KDA_SCAN_BATCH_READS=1` -> PASS; output/state
  PCC `0.999993/0.999995`.
- Profile command: `QWEN_KDA_SCAN_BATCH_READS=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS.
- CSV: `generated/profiler/reports/2026_07_24_17_59_57/ops_perf_results_2026_07_24_17_59_57.csv` (SHA-256 `6d1d105a6c4e802004c2e0436aeb80590ffb67d3a07777423295c4e123082b50`).
- Result: wall regresses `622.564 -> 638.924 us` (`+16.360 us`, `+2.63%`);
  recurrence regresses `144.945 -> 163.298 us` (`+12.66%`).
- Root cause: compute consumes `kd`, `v_beta`, then `t_inv`, `q_decay`,
  `intra`, `k_dec_t`, and `dl`. One global push withholds early `kd/v_beta`
  until every later read completes, eliminating reader/compute streaming.
- Verdict: reject and remove. Preserve per-input publication; test read ordering
  that matches compute consumption next.


### 2026-07-24 18:04:08 UTC — Reject compute-ordered scan reads

- Hypothesis: publish scan CBs in compute order (`kd`, `v_beta`, `t_inv`,
  `q_decay`, `intra`, `k_dec_t`, `dl`) so compute starts early and does not
  stall on `t_inv`, which the original reader published last.
- Target correctness -> PASS; output/state PCC `0.999993/0.999995`.
- T=640 CSV: `generated/profiler/reports/2026_07_24_18_02_33/ops_perf_results_2026_07_24_18_02_33.csv` (SHA-256 `dedb4fab18f4362fbbb80bd7a484c7089f7ab9a9f2c365a53c1d27172b6a3318`). Wall improves `622.564 -> 618.123 us` (`-0.71%`); recurrence improves `144.945 -> 141.990 us` (`-2.04%`).
- T=5,120 command: `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS.
- T=5,120 CSV: `generated/profiler/reports/2026_07_24_18_04_04/ops_perf_results_2026_07_24_18_04_04.csv` (SHA-256 `88385afab9ab560420ab195b90c297d6ad1e12202238f673c8080c50335fe217`). Wall regresses `3474.029 -> 3483.748 us` (`+0.28%`); recurrence improves only `809.704 -> 807.320 us` (`-0.29%`).
- Verdict: reject and remove. The short-context overlap benefit does not
  generalize to 160 chunks and misses both long-context retention gates.


### 2026-07-24 18:10:31 UTC — Reject double-buffered scan inputs

- Hypothesis: two CB slots for each of the seven streamed scan inputs let the
  reader prefetch chunk `c+1` while compute consumes chunk `c`; this costs
  about 38 KiB of additional L1 per scan core and adds no synchronization.
- Build: `./build_metal.sh --build-tests --build-type Release` -> exit 0.
- Target correctness with `QWEN_KDA_SCAN_DOUBLE_BUFFER=1` -> PASS; output/state
  PCC `0.999993/0.999995`.
- T=640 profile command:
  `QWEN_KDA_SCAN_DOUBLE_BUFFER=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> PASS. CSV
  `generated/profiler/reports/2026_07_24_18_08_06/ops_perf_results_2026_07_24_18_08_06.csv`
  (SHA-256 `f3429aa0752117a5c1276e1eb851b87e51e54c4919f1304fd44f48a1782627dd`).
  Wall improves `622.564 -> 620.369 us` (`-0.35%`); recurrence improves
  `144.945 -> 144.022 us` (`-0.64%`).
- T=5,120 profile command:
  `QWEN_KDA_SCAN_DOUBLE_BUFFER=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> PASS. CSV
  `generated/profiler/reports/2026_07_24_18_10_09/ops_perf_results_2026_07_24_18_10_09.csv`
  (SHA-256 `4ec515566b8e7b8ec6bea48172f75ceffe0ca1ff385806d79c0b02e04e9392fe`).
  Wall regresses `3474.029 -> 3489.888 us` (`+0.46%`); recurrence regresses
  `809.704 -> 812.764 us` (`+0.38%`).
- Diagnosis: because CB publication and consumption remain chunk-serial, the
  reader does not realize useful lookahead. The extra capacity only increases
  L1 occupancy and perturbs scheduling. The all-input experiment bounds the
  narrower BF16-only variant, so both are rejected.
- Verdict: reject and remove; no implementation change retained.


### 2026-07-24 18:23:20 UTC — Reject unbounded L1 gate residency

- Hypothesis: keep the transformed FP32 decay gate in distributed L1 so the
  110-core prep phase consumes it without a DRAM write/read handoff. At
  T=5,120 this occupies about 10 MiB/chip, or 95 KiB per L1 bank.
- Focused TP command:
  `QWEN_KDA_GATE_L1=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> PASS; output/recurrent/convolution PCC remained
  `0.999964/0.999903/0.999997`.
- T=640 profile command:
  `QWEN_KDA_GATE_L1=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> PASS. CSV
  `generated/profiler/reports/2026_07_24_18_19_44/ops_perf_results_2026_07_24_18_19_44.csv`
  (SHA-256 `f1faee3fd5712b2b429e7d1db303af23d27b3fb3833b48f40a6bbc723817eb8e`).
  Wall improves `622.564 -> 619.986 us` (`-0.41%`); recurrence improves
  `144.945 -> 142.713 us` (`-1.54%`). The decay op itself is unchanged
  (`33.106 -> 32.881 us`), confirming that the gain is the prep read path.
- T=5,120 was repeated twice with the identical trace command. Both tests
  passed, but neither report contained a traced replay: each had 488 rows with
  an empty `METAL TRACE REPLAY SESSION ID`, and the raw device log also had
  empty trace IDs. CSVs: `2026_07_24_18_20_53` (SHA-256
  `a620c95731bdb672b744cc22ca6247c1cb819dfaeac6ca0e8201c18bf9f08d36`)
  and `2026_07_24_18_22_24` (SHA-256
  `060f60093b891280d75e4160c922c244e75e038dce61d70c0533e7219f4b74d1`).
- Diagnosis: the 10 MiB/chip full-sequence L1 allocation is incompatible with
  the target trace capture/profiler path. The repeat rules out a transient
  instrumentation miss; without replay IDs there is no valid long wall metric.
- Verdict: reject and remove the unbounded allocation. The short-context gain
  supports only a bounded rolling L1 window in a prep/scan producer-consumer
  design.


### 2026-07-24 18:27:10 UTC — Reject BF8 fused input-projection weights

- Hypothesis: use the established Qwen `bfloat8_b` weight format for the fused
  KDA input projection to reduce its 524.798 us T=5,120 weight-traffic cost.
  Activations remain BF16 and the existing HiFi4/FP32 accumulation config is
  unchanged.
- Implementation was isolated behind `QWEN_KDA_INPUT_BF8=1`; cache names used
  a `.bf8` suffix so BF16 and BF8 artifacts could not collide.
- Focused command:
  `QWEN_KDA_INPUT_BF8=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s`
  -> PASS under the test threshold, but output/recurrent/convolution PCC was
  `0.999867/0.999878/0.999975` versus the selected endpoint
  `0.999964/0.999903/0.999997`.
- Diagnosis: the fused projection jointly produces Q/K/V, decay rank, beta,
  and output gate. BF8 weight error therefore enters the recurrent update and
  lowers recurrent-state PCC below the selected `0.999900` guard.
- Verdict: reject and remove without profiling; accuracy already disqualifies
  it. Test output-projection BF8 independently because it cannot affect state.


### 2026-07-24 18:31:25 UTC — Reject BF8 output-projection weights

- Hypothesis: use `bfloat8_b` only for the row-parallel output-projection
  weights. This cannot perturb recurrent state and may reduce weight traffic
  inside the fused matmul/reduce-scatter.
- Focused TP command with `QWEN_KDA_OUTPUT_BF8=1` -> PASS. Output PCC changed
  `0.999964 -> 0.999938`; recurrent/convolution PCC stayed exactly
  `0.999903/0.999997`.
- T=640 CSV:
  `generated/profiler/reports/2026_07_24_18_29_21/ops_perf_results_2026_07_24_18_29_21.csv`
  (SHA-256 `d4c644eef8e615400661e8281c6a71949ffa1ae2267817399635905b94fce16a`).
  Wall improves `622.564 -> 620.875 us` (`-0.27%`); output/CCL improves
  `152.946 -> 151.162 us` (`-1.17%`).
- T=5,120 CSV:
  `generated/profiler/reports/2026_07_24_18_31_00/ops_perf_results_2026_07_24_18_31_00.csv`
  (SHA-256 `4207beea2ecf51d9c1175cce24b3f6e11b3bdf7cdc5e21578c25d3b6e2ce08dd`).
  Wall regresses `3474.029 -> 3480.081 us` (`+0.17%`); output/CCL regresses
  `1069.975 -> 1071.173 us` (`+0.11%`).
- Diagnosis: the long fused MMRS is CCL-bound, matching the roofline result;
  reducing weight bytes does not shorten its critical path. The small short
  gain does not generalize, while output PCC is lower than the BF16 endpoint.
- Verdict: reject and remove.


### 2026-07-24 18:40:00 UTC — Measure and reject implicit post-MMRS clone removal

- Hypothesis: eliminate the clone of the persistent fused-MMRS output. The
  measured clone is 30.560 us at T=5,120, but ownership must remain safe when
  callers deallocate their returned tensor.
- Ceiling experiment returned the op `rs` alias and suppressed deallocation in
  the perf harness only. T=640 CSV
  `generated/profiler/reports/2026_07_24_18_33_57/ops_perf_results_2026_07_24_18_33_57.csv`
  (SHA-256 `56bb2c1b60b032b4e5ec7b0ec09251782c957b3b524334be9031a7b47364b56f`):
  29 calls/device/replay and wall `622.564 -> 616.505 us` (`-0.97%`).
- T=5,120 CSV
  `generated/profiler/reports/2026_07_24_18_35_37/ops_perf_results_2026_07_24_18_35_37.csv`
  (SHA-256 `d6da062ea7da7bda9491a48396deb71bea1da007c0f139506e8ab34aa810d1fa`):
  34 calls/device/replay and wall `3474.029 -> 3446.590 us` (`-0.79%`,
  `-27.439 us`). This is a theoretical/unsafe ceiling, not a retained result.
- Ownership hypothesis: return cached `out_buf` instead of `rs`; the cache
  should hold a shared tensor reference so default `deallocate(false)` cannot
  free persistent storage. The unmodified perf harness then deallocated the
  warm output normally.
- Result: traced T=640 hung, triggered the 5 s device timeout, and required the
  safe wrapper to reset all eight devices (`SAFE_PYTEST_RESULT: HANG`). This
  proves the MMRS output/cache aliases do not share ownership in the way
  required by `Tensor::deallocate`.
- Recovery: restored the clone, then
  `PERF_SEQ=64 PERF_REPS=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`
  -> PASS, confirming all eight devices recovered.
- Verdict: retain the clone. Its 0.79% long-context ceiling justifies future
  work only with an explicit pipeline-owned destination or explicit borrowed
  lifetime contract; implicit aliasing is unsafe.


### 2026-07-24 18:48:20 UTC — Retain BF16 transformed-gate storage for chunk prefill

- Hypothesis: the transformed vector decay gate does not need FP32 storage before chunk prep. Keeping its arithmetic consumers and recurrent state in FP32 should preserve accuracy while halving projection/Softplus output traffic. Recurrent/decode remains FP32 because it is not tile-aligned and does not amortize conversion the same way.
- Implementation: the layer emits BF16 transformed gates only for head-major chunk prefill. `chunk_kda` preserves an already-flat gate dtype; phased KDA validation and CB formats accept BF16 vector gates while retaining FP32-only scalar GDN. The prep reader uses separate BF16 gate and FP32 scalar tile addressing so beta and constants remain FP32.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> exit 0; all 52 requested targets/install steps completed.
- Focused TP command: `QWEN_KDA_GATE_BF16=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> PASS. Output/recurrent/convolution PCC was `0.999965/0.999910/0.999997`, versus retained `0.999964/0.999903/0.999997`.
- T=640 profile command: `QWEN_KDA_GATE_BF16=1 PERF_TRACE=1 PERF_SEQ=640 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_18_44_22/ops_perf_results_2026_07_24_18_44_22.csv` (SHA-256 `d00e2c212d8778b6fabc90dd80d3c583adaf8027f89aac38ea44b824d006f2eb`). Wall improved `622.564 -> 619.594 us` (`-0.48%`), decay transform `33.106 -> 20.528 us` (`-38.0%`), and recurrence `144.945 -> 143.634 us` (`-0.90%`).
- T=5,120 profile command: `QWEN_KDA_GATE_BF16=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_18_46_01/ops_perf_results_2026_07_24_18_46_01.csv` (SHA-256 `e4aced4133c9897beaffbb783051215141e5dcf2894bc815940ccda16cfa257c`). Wall improved `3474.029 -> 3385.003 us` (`-2.56%`, `-89.026 us`); decay transform improved `200.638 -> 108.106 us` (`-46.1%`); recurrence improved `809.704 -> 807.181 us` (`-0.31%`). Replay wall spans were `[3375.350, 3363.657, 3384.041, 3402.316, 3387.475, 3378.175, 3387.474, 3382.373, 3400.233, 3385.964] us`.
- Promoted full gate: `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py models/experimental/kimi_delta_attention/tests/test_factory.py models/experimental/kimi_delta_attention/tests/test_kda_recurrent.py models/experimental/kimi_delta_attention/tests/test_reference.py models/experimental/kimi_delta_attention/tests/test_tp_weights.py models/experimental/kimi_delta_attention/tests/test_ttnn_layer.py -q -s` -> `27 passed in 29.57s`; `SAFE_PYTEST_RESULT: PASS`.
- Verdict: retain. The long wall gain clears the 1% gate, short context does not regress, and correctness slightly improves. The localization rules out recurrence as the source: `92.532 us` of the `89.026 us` wall gain comes from the decay transform, with overlap accounting for the non-additivity.


### 2026-07-24 18:52:30 UTC — Reject configured fused SiLU for depthwise Conv1d

- Hypothesis: set `Conv1dConfig.activation` to SiLU and remove the standalone unary, eliminating the measured `106 us` T=5,120 pass.
- Source survey: the public Conv2d/Conv1d configuration documents SiLU support, and KDA uses the native height-sharded depthwise Conv1d path. This was the smallest existing API consistent with the intended fusion.
- Accuracy command with the configured epilogue and no standalone unary: `QWEN_KDA_CONV_FUSED_SILU=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> FAIL; output PCC was `0.884282`.
- Diagnostic control: restore standalone `ttnn.silu` while leaving the configured epilogue enabled, then repeat the same command -> PASS with output/recurrent/convolution PCC `0.999965/0.999910/0.999997`, exactly matching the retained endpoint.
- Diagnosis: this depthwise Conv1d implementation ignores `Conv1dConfig.activation`; the failed output was the raw convolution, not a lower-accuracy fused SiLU. The control rules out double activation because it is numerically identical to the single-activation endpoint.
- Verdict: reject the public configuration route and restore source. A custom SiLU in the convolution output kernel remains a distinct implementation, with a measured upper bound of `106 us` or `3.13%` of current T=5,120 wall latency.


### 2026-07-24 18:58:10 UTC — Reject finer DRAM width slicing for 5K convolution

- Hypothesis: the auto-selected two-slice width partition may leave insufficient parallelism or an oversized working set; explicit four or eight slices could reduce the `601.588 us` convolution block.
- Existing-pattern survey: Conv1d unit coverage explicitly supports manual `num_slices` values 4 and 8; the retained KDA report identifies auto routing as two DRAM width slices at T=5,120.
- Four-slice command: `QWEN_KDA_CONV_SLICES=4 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_18_55_45/ops_perf_results_2026_07_24_18_55_45.csv` (SHA-256 `9f9010da731e5728131d6d763775b36143520334f504cf2bc8c056cf1852ce65`). Median wall was `3467.099 us`, `+2.43%` versus retained `3385.003 us`.
- Eight-slice command: `QWEN_KDA_CONV_SLICES=8 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_18_57_05/ops_perf_results_2026_07_24_18_57_05.csv` (SHA-256 `1805cf58061585c1bf5e73d2c024e1fd40e12e47c14a33309b2aa3e241b1b157`). Median wall was `3712.531 us`, `+9.68%`.
- Diagnosis: calls per device/replay increase from 35 at two slices to 45 at four and 65 at eight, exactly five additional device ops per added slice. The monotonic regression supports fixed slicing/orchestration cost as dominant and rules out insufficient slice parallelism.
- Verdict: reject explicit four/eight-slice routing, restore `num_slices=0`, and retain the auto-selected two-slice path.


### 2026-07-24 19:00:20 UTC — Reject Conv1d activation reuse for KDA geometry

- Hypothesis: `enable_activation_reuse` could read each convolution activation slice once across output blocks and automatically use the split reader, reducing repeated movement without extra slice launches.
- T=32 correctness control: `QWEN_KDA_CONV_ACT_REUSE=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> FAIL before execution: activation block height 1 must be greater than output width 1 tile.
- Long-context localization enabled reuse only for `sequence > 640`. Command: `QWEN_KDA_CONV_ACT_REUSE=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s`. Pytest failed before trace capture: activation block height 2 must be greater than output width 160 tiles. The profiler wrapper produced only a partial failure CSV; no performance metric is valid.
- Diagnosis: activation reuse is designed for block-height-dominant convolution. KDA is width-dominant: satisfying the guard at T=5,120 requires an activation block of at least 161 tiles instead of 2, incompatible with the current L1 geometry. Both short and long controls rule out sequence-local applicability.
- Verdict: reject and restore source. Ordinary DRAM width-slice streaming remains the compatible native path.


### 2026-07-24 19:02:30 UTC — Reject forced convolution split reader

- Hypothesis: override the native convolution viability model and use both data-movement readers for activation transfer at T=5,120.
- Command: `QWEN_KDA_CONV_SPLIT_READER=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_19_02_09/ops_perf_results_2026_07_24_19_02_09.csv` (SHA-256 `08d451430f276b88d0e14452c0e2e890998ec9273779504fa8e7794d64103688`).
- Result: convolution active time improved `606.272 -> 603.237 us` (`-0.50%`), but median wall regressed `3385.003 -> 3387.903 us` (`+0.09%`). Replay spans were `[3372.809, 3378.543, 3394.839, 3393.488, 3402.440, 3380.552, 3374.595, 3382.587, 3393.219, 3393.310] us`.
- Diagnosis: the native source cost model compares one reader overlapping activation+tilize against two readers that delay weight transfer. The tiny active-time gain and wall regression empirically confirm that the forced schedule does not improve the critical path.
- Verdict: reject and restore auto selection; it misses both the 1% wall and 5% block gates.


### 2026-07-24 19:03:50 UTC — Reject convolution activation double buffering

- Hypothesis: double the activation CB so the ordinary single reader can overlap the next DRAM activation block with current convolution compute.
- Command: `QWEN_KDA_CONV_ACT_DB=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> pytest FAIL before execution. The outer profiler wrapper emitted a partial CSV and misleading PASS status; no replay metric is valid.
- Exact device error: DRAM auto slicing tried up to 160 width slices for output dimension 5120 but could not find a valid configuration with `1,434,496 bytes` available L1.
- Diagnosis: doubled activation CB capacity, not dispatch or timeout, is the proven blocker. Maximum slicing rules out recovering enough L1 by shrinking the sequence working set.
- Verdict: reject and restore source. The native Conv1d path has now rejected finer slicing, activation reuse, forced split reading, and activation double buffering; further gain requires a custom tiled depthwise reader/compute/writer path.


### 2026-07-24 19:10:20 UTC — Reject generic tiled FIR as the long-convolution replacement

- Hypothesis: reuse `_causal_conv1d_fir`, the existing tiled four-tap path used by non-native cases, to bypass native Conv1d row-major conversion and sliced-convolution plumbing.
- Focused TP command: `QWEN_KDA_TILED_FIR=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> PASS. Output/recurrent/convolution PCC was `0.999959/0.999913/0.999997`.
- T=5,120 command: `QWEN_KDA_TILED_FIR=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_19_10_08/ops_perf_results_2026_07_24_19_10_08.csv` (SHA-256 `0ac5b6b23ae37affdd03a4755b1c57a4617ca72218ec90fccfffa5ffa57d9844`).
- Result: median wall regressed `3385.003 -> 4837.313 us` (`+42.9%`). The convolution region expands from 15 calls in the native path to 23 calls, repeatedly running untilize, shifted slice, tilize, and ternary MAC operations per tap.
- Diagnosis: avoiding native Conv1d does not avoid layout traffic when expressed through generic TTNN slices; it multiplies that traffic by the four taps. Correctness rules out mathematical differences as the regression source.
- Verdict: reject and restore native Conv1d. A custom single device program remains the supported path: directly read shifted tiled rows, apply all four depthwise taps, fuse SiLU, and write tiled output/state.


### 2026-07-24 19:15:20 UTC — Reject fused QKV crop/untilize at target trace shape

- Hypothesis: replace the serialized QKV `SliceDeviceOperation` (`86.093 us`) plus `UntilizeDeviceOperation` (`95.591 us`) with existing `ttnn.untilize_with_unpadding` directly on the fused projection, cropping width 2208 to 1536 while producing row-major convolution input.
- Focused TP command: `QWEN_KDA_FUSED_QKV_UNTILIZE=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> PASS with the exact retained output/recurrent/convolution PCC tuple `0.999965/0.999910/0.999997`.
- Target command: `QWEN_KDA_FUSED_QKV_UNTILIZE=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> `SAFE_PYTEST_RESULT: HANG`. Warm/eager setup completed, but captured replay timed out after 5 seconds; all eight devices became unrecoverable and the safe wrapper reset them. No profiler CSV or latency is valid.
- Recovery: restored the retained slice+untilize path, then `PERF_SEQ=64 PERF_REPS=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS on all eight devices.
- Diagnosis: correctness at T=32 and successful warm setup rule out tensor semantics; failure is target-shape trace replay safety in the cropped-untilize program. This matches the prior full-sequence L1 gate case: a target path that cannot replay traces is not retainable.
- Verdict: reject the existing composite and restore source. The measured `181.684 us` combined slice/untilize ceiling still supports an offset-aware custom reader, but not this API route.


### 2026-07-24 19:21:20 UTC — Reject native equal-width Q/K/V split

- Hypothesis: target Kimi has equal TP-local Q/K/V widths (`512/512/512`), so the native tiled N-way `ttnn.split` fast path can replace three independent post-convolution slice programs with one device program. Unequal-width configurations retain the existing slices.
- Initial diagnostic: applying split-size `q_dim` unconditionally failed the focused test because its deliberate `32/32/256` Q/K/V geometry produced ten outputs. This rejected the assumption that every supported KDA configuration is equal-width and motivated a localized equal-width guard.
- Focused command with the guard: `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> PASS; output/recurrent/convolution PCC remained exactly `0.999965/0.999910/0.999997` through the unchanged unequal-width path.
- Target command: `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_19_20_59/ops_perf_results_2026_07_24_19_20_59.csv` (SHA-256 `e39287a74d6411e0295ae6ab33d13241ba536c45adc30d7b8fda3849d6772cc3`).
- Result: calls per device/replay fall `35 -> 33`; the three Q/K/V slices (`30.390 + 30.067 + 30.291 us`) become one `87.567 us` split. Median slowest-device wall improves `3385.003 -> 3381.006 us` (`-3.997 us`, `-0.118%`). Replay spans were `[3370.129, 3363.603, 3369.061, 3379.077, 3373.553, 3390.850, 3387.747, 3384.878, 3390.155, 3382.934] us`.
- Diagnosis: launch consolidation removes two programs but preserves nearly all tile movement; the replacement kernel costs `96.6%` of the three slice kernels. The approximately `2.9%` split-block improvement and `0.118%` wall improvement miss the established `5%` block and `1%` wall retention gates.
- Verdict: reject and restore independent slices. A useful projection/consumer fusion must eliminate data movement, not merely coalesce its launch.


### 2026-07-24 19:48:30 UTC — Reject row-major-input custom KDA convolution

- Hypothesis: one custom program distributed by 32-token blocks over up to 110 cores can replace native state concat, two sliced Conv1d pipelines, layout plumbing, and standalone SiLU. The scoped boundary retained QKV crop, TILE-to-row-major untilize, and the external carry-state slice.
- Implementation: the reader streams four shifted row-major windows and persistent tap tiles; compute performs four BF16 FPU products with row-broadcast tap weights, ping-pong partials, and fused SiLU; the writer emits tiled BF16 output. Initial `0.071281` full-layer PCC and the row-0-only identity pattern were diagnosed to plain multiplication of 1xC tap tiles, not tilization. Restoring symmetric tilize and using the established row-broadcast multiply fixed the root cause.
- Direct identity command: `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_kda_conv_debug.py -q -s` -> PASS, PCC `0.999998`.
- Focused TP command: `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> PASS; output/recurrent/convolution PCC `0.999955/0.999905/0.999997`.
- Target command: `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_24_19_47_52/ops_perf_results_2026_07_24_19_47_52.csv` (SHA-256 `38203f0212748d34049399b2f6494181d3f5adcc43d67117bbfb5cfed007ec02`).
- Result: calls per device/replay fell `35 -> 24`, but convolution active time regressed `606.271 -> 609.761 us` (`+0.58%`) and median slowest-device wall regressed `3385.003 -> 3390.084 us` (`+5.081 us`, `+0.150%`). Replay spans were `[3393.044, 3389.450, 3390.196, 3390.036, 3390.021, 3396.921, 3388.681, 3390.132, 3388.776, 3398.010] us`.
- Diagnosis: the custom kernel is `426.786 us`, but the retained QKV crop (`86.391 us`), external untilize (`95.561 us`), and carry slice (`1.023 us`) leave the new convolution region slightly slower than native. Launch removal is insufficient while projection-to-convolution data movement remains serialized.
- Verdict: reject this row-major-input boundary and restore source. Preserve the design evidence: the next custom version should consume tiled fused-projection output by offset and eliminate crop plus untilize before assessing compute micro-optimizations.


### 2026-07-24 20:42:15 UTC — Withdraw tiled-input KDA convolution and add a long PCC gate

- Finding: commit `83654eb5fc9` was fast but invalid. Its focused PCC test used T=32, while the implementation was guarded by `T > 640`; the test never executed the custom program.
- Real long-path command: `KDA_TP_TEST_SEQ=672 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` measured output/recurrent-state/convolution-state PCC `0.048428/-0.001323/-0.005458`.
- Root cause 1: the reader used floor division for the fused-projection tile stride. The focused logical width is 609, requiring 20 physical tiles, but the reader used 19; target width 2180 similarly requires 69, not 68. Ceiling division restored recurrent/convolution-state PCC to `0.999912/0.999997`, proving the stride error, but output PCC remained `0.951851`.
- Root cause 2 hypothesis: the residual error affects the first three rows of each 32-token block, which alone use cross-core previous-tile prefix extraction. A full-tile prefix prototype and a one-core ownership control both deadlocked, so this hypothesis remains unproven and neither change is retained.
- Native control: route the identical T=672 case through retained native Conv1d -> PASS with output/recurrent-state/convolution-state PCC `0.999967/0.999920/0.999997`. After rebuilding the exact reverted host library, the same command passed again with the identical tuple.
- Action: history-preserving revert `0c5beb02b0a`; the valid endpoint returns to T=5,120 wall `3385.003 us`. The TP test now accepts `KDA_TP_TEST_SEQ` so every future long-only optimization can be exercised by correctness CI.
- Verdict: reject and withdraw all performance claims from the tiled-input path. A future retry must pass T=672 before profiling T=5,120.


### 2026-07-24 21:08:00 UTC — Accept grouped affine-prefix device prototype

- Hypothesis: full per-chunk affine-prefix storage is unnecessary. A local
  group-summary pass, a prefix over summaries only, and a parallel short scan
  can expose 108-core sequence parallelism while retaining prepared tensors in
  distributed L1.
- Source evidence: the retained compute kernel executes one literal
  `NC`-iteration loop (`chunk_kda_scan.cpp:125-182`); its exact full-V CB set
  is declared at `chunk_gdn_phased_program_factory.cpp:398-417`.
- Distribution: 27 groups/head over four local heads = 108 workers. Groups are
  26x6 chunks plus 1x4 chunks per head. Phase 1 emits one 128 KiB FP32 `(A,B)`
  summary/group; phase 2 scans only those summaries; phase 3 reuses the
  current recurrence for at most six chunks/worker.
- Compute model: `2.013 GFLOP/chip` to construct transforms, `6.543 GFLOP/chip`
  for 532 local plus at most 248 summary compositions, and `2.349 GFLOP/chip`
  for final recurrence/output = `10.905 GFLOP/chip`. The `71.7 us` peak floor
  and `14.32%` chip-utilization break-even are below the current scan's
  approximately `21.2%` active-core efficiency.
- Capacity model: prepared tensors are `40 MiB/chip`; 108 summaries are
  `13.5 MiB/chip`; average persistent occupancy is about `498 KiB/core`.
  The full-V scan CB plan adds `576 KiB/core`, leaving about `327 KiB/core`
  under the measured `1,434,496 B` limit. Storing all per-chunk transforms
  would add `80 MiB/chip` and is rejected.
- Traffic hypothesis: keeping local compositions in CBs bounds distributed
  L1/NOC traffic near `200 MiB/chip`, requiring approximately `400 GB/s` at
  break-even. This is not yet measured.
- Verdict: the explicit model beats the `500.685 us` serial scan on plausible
  compute and capacity assumptions. Proceed to a device prototype; first
  falsify summary construction correctness and measured phase cost before
  integrating the full prefix.


### 2026-07-24 21:04:00 UTC — Retain scan-reuse affine group summaries

- Hypothesis: at target `K=V=128`, evaluate each eight-chunk affine transform
  with the existing recurrence twice: zero initial state gives B and block
  identity gives A+B. A device subtraction then gives A. This replaces custom
  transform construction and maps 20 groups/head over 80 whole-V workers.
- Implementation: private `QWEN_KDA_GROUP_SUMMARY=1` reshapes the seven
  prepared tensors into 80 pseudo-heads x 8 chunks. The phased scan gained
  generated-zero/generated-identity initial states and state-only output
  draining. Ordinary scan behavior remains the default.
- First failure: the T=256 gate failed at device-kernel compile because the
  small-head heuristic split V four ways (`Vt=1`, `Vt_full=4`). Identity state
  requires a whole square state. State-only scans now force whole-V ownership.
- Second failure: an ordinary split-V kernel still evaluated a non-dependent
  `static_assert` inside a discarded `if constexpr` branch. Removing that
  device assertion fixed compilation; the host mapping enforces whole-V for
  identity mode.
- Consumed-summary correctness command: `QWEN_KDA_GROUP_SUMMARY=1
  scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s -k
  ' 256 and 4 and 128 '` -> PASS. T=256 K=V=128 output/state PCC was
  `0.999992/0.999993`; final state came from `A @ S0 + B` with nonzero S0.
- Regression command: `scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s` ->
  6 passed.
- Summary profile: `QWEN_KDA_GROUP_SUMMARY=1 PERF_TRACE=1 PERF_SEQ=5120
  PERF_REPS=10 scripts/run_safe_pytest.sh --profile
  models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
  -q -s` -> PASS. CSV
  `generated/profiler/reports/2026_07_24_21_00_22/ops_perf_results_2026_07_24_21_00_22.csv`,
  SHA-256 `50455a0e39fd3c6548beefdce3f211c8b405ed52ae9db5c33be3e7b23e9f0471`.
  Wall median `3557.576 us`; spans `[3550.477, 3545.151, 3558.251,
  3556.901, 3525.836, 3577.817, 3553.974, 3560.619, 3580.649,
  3575.392] us`. Zero scan, identity scan, and subtraction are
  `70.869/91.912/21.706 us`, totaling `184.487 us`.
- Matched control: same command without the env -> PASS. CSV
  `generated/profiler/reports/2026_07_24_21_03_12/ops_perf_results_2026_07_24_21_03_12.csv`,
  SHA-256 `b00f96f5c893f6fa85dbbe0bdeb5891eef9f61bc5bd1224b457464f66f28728c`.
  Wall median `3380.644 us`; spans `[3378.048, 3380.180, 3376.323,
  3381.107, 3376.248, 3372.413, 3382.379, 3397.456, 3388.239,
  3393.462] us`; serial scan `500.140 us`.
- Verdict: retain the opt-in prototype. Summary construction adds only
  `176.932 us` wall while the original scan remains. Assuming the measured
  `70.869 us` grouped final scan, prefix must be below `210.978 us` for a 1%
  whole-layer win. Proceed to the summary-prefix kernel.

### 2026-07-24 21:16 UTC — Reject generic-op affine prefix as a performance path

- Hypothesis: a five-stage Hillis-Steele inclusive scan assembled from existing batched matmul, add, slice, and concat operations can validate the full grouped algorithm and might fit the `210.978 us` prefix budget.
- Correctness: `QWEN_KDA_GROUP_PREFIX=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s -k '512 and 4 and 128'` -> PASS. Two groups/head reproduce the serial output/final state at PCC `0.999991/0.999993` (max abs `9.182990e-4/9.382606e-3`). This validates composition order, entry-state shifting, grouped output layout, and final-state selection.
- First target failure: L1-backed intermediates collided with scan CBs (`681,728 B` allocation while the static CB region ended at `701,312 B`). This is allocator evidence, not a compute failure. Moving only the generic prefix temporaries to DRAM removed the collision.
- Target command: `QWEN_KDA_GROUP_PREFIX=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> test PASS. CSV `generated/profiler/reports/2026_07_24_21_15_28/ops_perf_results_2026_07_24_21_15_28.csv`, SHA-256 `a01dc4385d35643dd00233a1ee0f51dd1df5aab28f25afe3f28123e987e667de`.
- Profiler qualification: device marker buffers overflowed late. Sessions 2-8 are complete at 101 calls/device; sessions 9-11 are incomplete and excluded. Complete-session slowest-device spans are `[6275.149, 6261.050, 6276.779, 6293.264, 6267.061, 6271.001, 6288.919] us`, median `6275.149 us`.
- Attribution: summary scans/subtraction cost `70.748 + 91.930 + 13.469 = 176.147 us`; the 62 generic prefix/correction ops cost `3120.528 us`; the final eight-chunk grouped scan costs `88.898 us`. Whole-layer wall regresses `2894.505 us` (`85.62%`) from the matched `3380.644 us` control.
- Diagnosis: ten separately materialized matrix multiplies plus 52 staging/data-movement operations dominate. The arithmetic formulation is correct, but general tensor operators neither retain affine pairs in core-local ping-pong buffers nor fuse the five dependency stages.
- Exact-source recheck: the focused prefix command retained PCC `0.999991/0.999993`; the default `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s` regression passed `7/7`.
- Verdict: reject generic operators as the performance implementation. Retain the private opt-in path only as a device correctness oracle. The performance path must be a persistent five-stage program with one group/core, CB-local A/B ping-pong, direct inter-core transfer, and fused composition/correction.

### 2026-07-24 22:16 UTC — Reject persistent point-to-point affine prefix

- Hypothesis: one group/core and five Hillis-Steele stages can replace the generic prefix with a single persistent program. Each worker reads only its dependency worker’s FP32 `(A,B)` pair over NOC, composes locally, and emits the corrected group-entry state.
- Correctness depth: the first two-slot implementation passed T=512 (one stage), T=1024 (two stages), and T=2048 (three stages). The deepest measured gate reproduced output/state PCC `0.999990/0.999993`, with maximum absolute error `1.012996e-3/8.136928e-3`.
- Deadlock evidence: the same program hung at T=3072 (four stages/48 workers), T=4096 (four stages/64 workers), and the target T=5120 (five stages/80 workers). Every hang was detected by `run_safe_pytest.sh` as an unrecoverable worker timeout and required the prescribed eight-device reset; it was not a pytest budget failure.
- Root-cause isolation: stage-specific readiness semaphores did not change the target hang, ruling out cross-stage credit aliasing. Extending state lifetime from two to three CB slots made the previously failing T=3072 replay pass, proving that source-state reuse under dependency skew was one real cause.
- Capacity boundary: five immutable stage slots exceeded L1 by `44,160 B` at T=3072. Four slots at T=5120 exceeded L1 by `105,600 B`. Three slots fit only after moving the 5 MiB/chip group-entry output boundary to DRAM; it still hung at five stages.
- Failed bounded-memory variants: two- and three-slot rings with remote-read acknowledgements were tested using stage counters, a stage bitmask, atomic increments, and barrier-ordered remote semaphore writes. All acknowledgement forms retained the three-stage PCC result but deadlocked at four stages. The platform supports 16 NoC semaphores and descriptor IDs are explicit, so semaphore-count and descriptor-order hypotheses were checked and rejected.
- Validation commands included `QWEN_KDA_GROUP_PREFIX=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_chunk_kda.py -q -s -k 2048` (PASS) and `QWEN_KDA_GROUP_PREFIX=1 PERF_SEQ={3072,4096,5120} PERF_REPS=1 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` (depth-dependent PASS/HANG results above).
- Verdict: reject and fully restore the validated baseline. No unvalidated kernel remains in the worktree. The experiment never reached a target-shape timing, so it makes no performance claim. A future distributed prefix needs a different ownership protocol (for example, receiver-owned stage storage or a framework-native point-to-point primitive), not another ping-pong/ACK variation.


### 2026-07-24 22:30 UTC — Reject static-core prep-to-scan fusion

- Hypothesis: merge prep and scan into one program, reserve 94 workers for
  chunk-parallel prep and 16 workers for four-head/four-V-block scan, and let
  each scan worker consume a chunk after its exact readiness bit is published.
  This should overlap the measured `306.028 us` prep with the `500.140 us`
  scan while removing one launch.
- Implementation: the experimental program retained all seven distributed-L1
  intermediates, used disjoint prep/scan core ranges, and allocated five 32-bit
  readiness words per scan core (160 chunks). A prep writer published its
  unique chunk bit only after all seven output barriers completed. This avoided
  count aliasing under out-of-order producer completion.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Correctness: `QWEN_KDA_FUSED_PIPELINE=1 scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc
  -q -s` -> PASS at T=32 with output/recurrent/convolution PCC
  `0.999965/0.999910/0.999997`. The same command with
  `KDA_TP_TEST_SEQ=672` -> PASS at 21 chunks with PCC
  `0.999967/0.999920/0.999997`.
- Performance command: `QWEN_KDA_FUSED_PIPELINE=1 PERF_TRACE=1 PERF_SEQ=5120
  PERF_REPS=10 scripts/run_safe_pytest.sh --profile
  models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
  -q -s` -> PASS. CSV
  `generated/profiler/reports/2026_07_24_22_29_50/ops_perf_results_2026_07_24_22_29_50.csv`,
  SHA-256 `8c15e584ba6673f50654c6ebd07c9db57f26467633cd486c5ebba120500999e7`.
- Result: calls/device/replay fell `35 -> 34`, but median slowest-device wall
  regressed `3380.644 -> 3421.029 us` (`+40.385 us`, `+1.19%`). Replay spans
  were `[3417.246, 3416.790, 3429.692, 3415.025, 3440.213, 3414.435,
  3433.797, 3428.980, 3420.494, 3421.564] us`. The merged recurrence measured
  `842.027 us`, versus `306.028 + 500.140 = 806.168 us` for sequential prep
  and scan in the matched control.
- Diagnosis: correctness and multi-chunk readiness rule out the synchronization
  protocol. The static 94/16 partition removes 16 prep workers for the entire
  program and cannot hand them back after prep; that throughput loss exceeds
  the recovered overlap and launch saving. Elastic reassignment remains a
  distinct hypothesis, but this disjoint static-core design is rejected.
- Verdict: reject and fully restore source. No experimental kernel remains in
  the worktree.


### 2026-07-24 22:48 UTC — Reject tiled-input convolution prefix variants

- Hypothesis: the withdrawn tiled convolution can be recovered by first fixing
  its proven projected-width stride error, then replacing its three-row
  cross-block prefix gather without changing the correct four-tap compute body.
- Reproduction: restored the private historical program, changed projected tile
  stride from floor to ceiling division, and ran `QWEN_KDA_TILED_CONV=1
  KDA_TP_TEST_SEQ=672 scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc
  -q -s`. It reproduced the historical output/recurrent/convolution PCC exactly:
  `0.951851/0.999912/0.999997`.
- Row oracle: a focused primitive comparison against host depthwise convolution
  showed RMSE around `8e-4` to `1e-3` on ordinary rows, but `0.084-0.185` on
  exactly rows `32-34`, `64-66`, ..., `640-642`. This proves the four-tap
  compute body and ordinary current-tile path are correct and isolates failure
  to each core first-block prefix.
- Full-tile staging variant 1 added separate previous-tile tiled/row-major CBs
  and a hardware untilize before the current tile. The T=672 row oracle hung
  with broad BRISC stalls and required the prescribed eight-device reset.
- Full-tile staging variant 2 reused the existing projected tiled/row-major CB
  pair sequentially to remove the new-CB protocol. It hung identically and also
  required reset, isolating the additional untilize phase as the common failed
  mechanism rather than CB identity.
- Face-order diagnostic restored the single-untilize program and tested the
  alternate bottom-left/right physical offsets. It completed but preserved the
  same three-row periodic error and worsened peak row RMSE `0.185 -> 0.201`;
  reject that layout hypothesis.
- Verdict: reject and fully restore all tiled-input variants. Partial face-row
  prefix reads are not a validated boundary, and a second untilize deadlocks.
  Preserve the already-correct row-major custom convolution as the viable next
  route; make its writer emit Q/K/V directly so the measured `~94.5 us` three-
  slice boundary is eliminated before reassessing wall latency.


### 2026-07-24 23:10 UTC — Retain direct-output row-major convolution

- Hypothesis: the numerically correct row-major custom four-tap convolution
  missed the wall gate because three post-convolution Q/K/V slices remained.
  Writing Q/K/V tiles directly should recover their approximately `94.5 us`
  boundary without taking on the rejected tiled-prefix protocol.
- Implementation: one work item owns one 32-token block. Up to all 110 compute
  cores read four row-major activation windows and the four tap rows, perform
  row-broadcast multiply/add plus SiLU, and write Q, K, and V tiles directly to
  separate DRAM outputs. The layer retains native convolution for
  `sequence <= 640`; long aligned prefill uses the custom program.
- Root-cause correction: the recovered prototype initially used ordinary tile
  multiply and produced per-shard PCC near `0.17`. A direct Q/K/V oracle proved
  the error was inside convolution, not the writer. Restoring
  `mul_tiles_bcast_rows` matched the channel-wise tap semantics and raised all
  local Q/K/V shard PCCs to `0.999988-0.999990`.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Long correctness: `KDA_TP_TEST_SEQ=5120
  scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc
  -q -s` -> PASS with output/recurrent/convolution PCC
  `0.999958/0.999890/0.999997`. The T=672 gate also passed at
  `0.999958/0.999912/0.999997`.
- Performance: `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10
  scripts/run_safe_pytest.sh --profile
  models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
  -q -s` -> PASS. CSV
  `generated/profiler/reports/2026_07_24_23_07_40/ops_perf_results_2026_07_24_23_07_40.csv`,
  SHA-256 `7293b8cdd6daa99d868c16b4192587d6682e105e9c7559e8033695902110adf4`.
- Result: calls/device/replay fell `24 -> 21`; the custom convolution median
  was `413.165 us`. Slowest-device replay spans were `[3330.362, 3330.574,
  3334.687, 3333.137, 3337.141, 3340.451, 3339.620, 3334.622, 3333.856,
  3330.698] us`, median `3334.239 us`. Against the matched `3380.644 us`
  control, wall improves `46.405 us` (`1.37%`).
- Verdict: retain. This validates direct multi-output production but not tiled
  projection consumption; the input QKV crop/untilize boundary remains.


### 2026-07-24 23:17 UTC — Retain minimal-core gated-RMS mapping

- Hypothesis: T=5,120 has 640 independent `(head,32-token-block)` RMS items.
  Mapping them to 107 cores preserves the all-core critical load of six items
  per worker while avoiding three underfilled workers.
- Implementation: derive the all-core ceiling, then select the smallest worker
  count that preserves it. This remains shape-generic rather than hard-coding
  the T=5,120 core count.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Performance: `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10
  scripts/run_safe_pytest.sh --profile
  models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
  -q -s` -> PASS. CSV
  `generated/profiler/reports/2026_07_24_23_16_12/ops_perf_results_2026_07_24_23_16_12.csv`,
  SHA-256 `317912b3cf9378a719fbd8d3cef2df91235009b9f66817c794f8fe165769b14f`.
- Result: gated-RMS kernel median improved `86.339 -> 83.557 us`; slowest-
  device replay spans were `[3330.481, 3324.707, 3329.104, 3327.917,
  3330.619, 3333.578, 3332.932, 3330.176, 3332.660, 3329.805] us`, median
  `3330.328 us`. Matched wall improves `3.911 us` (`0.117%`).
- Long correctness: `KDA_TP_TEST_SEQ=5120 ...::test_tp_layer_pcc -q -s` ->
  PASS at output/recurrent/convolution PCC `0.999958/0.999890/0.999997`.
- Verdict: retain. The critical load, not maximum physical core occupancy, is
  the correct mapping objective for this epilogue.


### 2026-07-24 23:22 UTC — Reject BF16 gated-RMS projection boundary

- Hypothesis: store the 10 MiB/chip gated-RMS output in BF16 while keeping the
  fused output matmul and reduce-scatter persistent buffers in FP32. This was
  intended to isolate local boundary traffic from the previously rejected BF16
  communication-partial experiment.
- Implementation: added an experimental gated-RMS output dtype, selected BF16
  only for long TP prefill, and passed FP32 to the existing MMRS buffer cache.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Correctness: `KDA_TP_TEST_SEQ=5120 scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc
  -q -s` -> FAIL. Output PCC was `-0.000049`; recurrent and convolution-state
  PCC remained `0.999890/0.999997`.
- Diagnosis: this is format incompatibility, not BF16 quantization drift. The
  fused MMRS wrapper chooses FP32 persistent buffers from its `dtype` argument,
  but the underlying matmul derives its produced format from the BF16 input.
  The op has no independent output/accumulation dtype, so BF16 tiles alias FP32
  buffer pages. Inserting a typecast would restore the removed DRAM pass and no
  longer test the proposed boundary.
- Restoration: rebuilt the exact reverted source and reran the T=672 long-path
  gate -> PASS at output/recurrent/convolution PCC
  `0.999958/0.999912/0.999997`.
- Verdict: reject and fully restore source. Revisit only after MMRS exposes an
  independent matmul-output/persistent-buffer dtype contract.


### 2026-07-24 23:37 UTC — Retain bounded scan-to-gated-RMS pipeline

- Hypothesis: preserve the 16 scan producers and use the other 94 Tensix cores
  as gated-RMS consumers in the same program. Fine-grained readiness should
  overlap most of the standalone `83.557 us` RMS stage with the serial scan.
- Implementation: each scan producer writes one FP32 V-block to the existing
  recurrence output, completes its NOC barrier, and signals the cyclically
  assigned RMS consumer. A consumer waits for all four V-block signals for its
  `(head, chunk)` item, reads the complete row plus gate/weight, and writes the
  token-major gated output. T=5,120 distributes 640 items as six or seven per
  consumer; T=672 distributes 84 items across the first 84 consumers.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Correctness: `KDA_TP_TEST_SEQ=672 scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc
  -q -s` -> PASS at output/recurrent/convolution PCC
  `0.999958/0.999912/0.999997`. The identical T=5,120 command also passed at
  `0.999958/0.999890/0.999997`.
- Performance: `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10
  scripts/run_safe_pytest.sh --profile
  models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
  -q -s` -> PASS. CSV
  `generated/profiler/reports/2026_07_24_23_36_58/ops_perf_results_2026_07_24_23_36_58.csv`,
  SHA-256 `94c235956f94bcfa0995f185521be47f8d9fce90d177065034e7cdc9808c6086`.
- Result: calls/device/replay fell `21 -> 20`. Slowest-device replay spans for
  sessions 2-11 were `[3265.436, 3258.931, 3256.424, 3261.347, 3261.990,
  3267.553, 3265.654, 3266.059, 3265.044, 3267.196] us`, median `3265.240 us`.
  Against the retained `3330.328 us` endpoint, wall improves `65.088 us`
  (`1.954%`). The combined scan program median is `514.395 us`, so the overlap
  hides about 78% of the former RMS stage.
- Verdict: retain. This proves the bounded producer/consumer schedule and
  synchronization. It does not yet eliminate the intermediate FP32 DRAM
  write/read; direct producer-to-consumer L1 transfer is the next experiment.


### 2026-07-24 23:55 UTC — Retain bounded direct-L1 scan-to-RMS handoff

- Hypothesis: the retained fused program still writes each 10 MiB/chip FP32
  scan output to DRAM and rereads it on the RMS consumers. Fixed consumer-local
  L1 slots can remove that round trip without introducing a producer/consumer
  credit cycle.
- Implementation: CB0 is allocated identically across all 110 Tensix cores.
  Each of the 94 RMS consumers owns `ceil(640/94) = 7` full-V slots at T=5,120,
  or `112 KiB/core`. Scan producer `(head, V-block)` writes its unique
  `(consumer, local item, V-block)` slot, executes a NOC write barrier, then
  increments the consumer semaphore. The consumer waits for all four V-block
  signals and publishes that fixed slot into its local CB. Capacity covers the
  complete invocation, so producers never overwrite an unconsumed slot.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS for
  the candidate, matched control, and restored candidate.
- Correctness: T=672 and T=5,120 `test_tp_layer_pcc` hardware gates both PASS.
  Output/recurrent/convolution PCC remained `0.999958/0.999912/0.999997` and
  `0.999958/0.999890/0.999997`, respectively.
- Candidate profile 1: CSV
  `generated/profiler/reports/2026_07_24_23_50_45/ops_perf_results_2026_07_24_23_50_45.csv`,
  SHA-256 `14ce4ddfd5c77a98d231a9fe150e7f9832aa58dbeb61687c9ae4d85a95473979`,
  median wall `3259.237 us`, scan `512.911 us`.
- Candidate profile 2: CSV
  `generated/profiler/reports/2026_07_24_23_52_16/ops_perf_results_2026_07_24_23_52_16.csv`,
  SHA-256 `d8ca9c233fc17063198707ca1f0791d528707a47ba50c322827a4539aa9959b7`,
  median wall `3259.763 us`, scan `512.991 us`.
- Freshly rebuilt DRAM-handoff control: CSV
  `generated/profiler/reports/2026_07_24_23_54_32/ops_perf_results_2026_07_24_23_54_32.csv`,
  SHA-256 `7e2b9e843425266e0361a898b8b7f36d53dbd4d32e202664a9bf0fa1acba31ef`,
  median wall `3264.154 us`, scan `514.549 us`.
- Result: the replicated candidate center is `3259.500 us`, improving the
  matched control by `4.655 us` (`0.143%`). The combined scan program improves
  `1.598 us` (`0.31%`). The smaller kernel delta and wider whole-replay spread
  explain why the original single-profile estimate was not accepted alone.
- Verdict: retain. The gain is reproducible and above the requested `0.1%`
  whole-layer threshold, but it is a micro-optimization, not a material change
  to the roofline. The FP32 DRAM intermediate is no longer transferred.


### 2026-07-25 00:24 UTC — Reject isolated receiver-owned affine prefix

- Hypothesis: replace the failed sender-owned stage protocol with one
  receiver-owned inbound `(A,B)` slot per core and a global barrier between
  Hillis-Steele stages. This should remove the four-stage deadlock without
  exceeding L1 and reduce the serial 160-chunk scan dependency chain.
- Implementation tested: one persistent worker per eight-chunk group (80 cores
  at T=5,120), compute-owned A/B ping-pong, one receiver-owned inbound A/B
  slot, sender flush-before-ready, and a global stage barrier before slot reuse.
  The surrounding oracle used the existing two group-summary scans plus
  subtraction, then an eight-chunk grouped final scan.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Progressive hardware gates: T=512 (one stage) passed direct output/state PCC
  `0.999991/0.999993`; TP=8 T=1,024, 2,048, 3,072, and 5,120 all passed.
  T=5,120 output/recurrent/convolution PCC was
  `0.999958/0.999890/0.999997`, identical to the retained endpoint. The
  four-stage T=3,072 pass rejects stage depth as the cause of the earlier
  sender-owned deadlock; unsafe sender slot reuse was the differentiator.
- Integration diagnosis: the first profiler attempt segfaulted because the
  grouped scan returned two tensors while the fused-RMS caller indexed
  `scan[2]`. A standalone gated-RMS result restored the public return contract;
  rebuilt T=5,120 PCC then passed unchanged. This was a host integration bug,
  not a device protocol hang.
- Performance command: `QWEN_KDA_GROUP_PREFIX_PERSISTENT=1 PERF_TRACE=1
  PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile --run-all
  models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
  -q -s` -> PASS. CSV
  `generated/profiler/reports/2026_07_25_00_23_43/ops_perf_results_2026_07_25_00_23_43.csv`,
  SHA-256 `dfdbae7cd91373b81dce8ce502d9e1cc4c56439679e71b64c8b0ab55d0ba9ab9`.
- Sessions 2-11 slowest-device spans were `[3296.615, 3305.999, 3306.177,
  3310.197, 3297.297, 3291.047, 3309.246, 3298.207, 3297.561, 3307.103] us`;
  median `3302.103 us`. Against retained `3259.500 us`, wall regresses
  `42.603 us` (`1.307%`). The affine-prefix kernel median is `197.867 us`;
  grouped scans total `249.170 us`, standalone gated RMS is `84.584 us`, and
  prep remains `305.979 us`.
- Verdict: reject and fully remove the spike. The protocol is correct but an
  isolated prefix is not profitable. A retry must first fuse summary
  construction with the prefix owner or fuse prefix correction into the
  grouped recurrence; optimizing this 197.867 us kernel alone cannot recover
  the surrounding materialization and launch costs.


### 2026-07-25 00:52 UTC — Reject chunk size 64 before profiling

- Hypothesis: doubling the chunk from 32 to 64 tokens halves the ordered scan
  from 160 to 80 steps at T=5,120, potentially recovering roughly half of the
  retained scan span. The risk was extra Ct=2 prep work and numerical drift.
- Initial implementation: parameterized the layer and recurrence for C=64,
  disabled fused scan-to-RMS for the candidate, and lifted the host Ct=1 guard.
  The first host validation failure correctly showed the operation still
  required C=32; after lifting it, the q/k-flat validation required adjustment
  because normalization was temporarily performed before the fused operation.
- Hang diagnosis 1: the prep compute reserved and the writer consumed
  `Kt*Ct` tiles for `k_dec_t`, but the transpose loop pushed only `Kt`. The
  exact missing-producer count caused the first device hang. Publishing all
  `(ki,ci)` transposed tiles removed it.
- Hang diagnosis 2: the retained doubling inverse produces one 32x32 tile,
  while Ct=2 scan waits for `Ct*Ct = 4` inverse tiles. Reusing the existing
  generic Ct=2 blocked triangular solver removed this second deterministic
  wait. Its first adaptation also exposed two KDA-specific live-range
  collisions (`cb_supd` still held the anchor factor and `cb_scr1` still held
  the right-decayed key); moving solver temporaries off those live tensors
  eliminated corruption.
- Correctness with external Q/K normalization: `QWEN_KDA_CHUNK_SIZE=64
  KDA_TP_TEST_SEQ=64 scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc
  -q -s` -> FAIL at output/recurrent/convolution PCC
  `0.996686/0.929095/0.999997`. A generalized full-64 exact-doubling inverse
  produced the identical PCC, rejecting the inherited blocked solver as the
  source of the remaining state error.
- Normalization control: enabling the retained in-prep Q/K normalization for
  Ct=2 regressed output/recurrent PCC to `0.949638/0.907320`; convolution
  remained `0.999997`. This rejects the external-normalization path as the sole
  cause and identifies Ct=2 recurrence numerics or state handoff as unresolved.
- Performance: not measured. Profiling an implementation that fails the state
  correctness gate would not support a retention decision.
- Restoration: all candidate source was removed, then
  `./build_metal.sh --build-tests --build-type Release` passed. The restored
  `KDA_TP_TEST_SEQ=5120` TP=8 gate passed at output/recurrent/convolution PCC
  `0.999958/0.999890/0.999997`.
- Verdict: reject C=64 in the current optimization campaign. A future retry
  requires a standalone Ct=2 oracle that attributes prepared tensors and final
  state before any whole-layer performance work.


### 2026-07-25 01:09 UTC — Retain one-pass affine summary builder

- Hypothesis: carry zero and identity recurrent states through one state-only grouped scan, then compute `A = (A+B) - B` on-core. This should remove the second scan, standalone subtraction launch, and intermediate DRAM handoff from the eight-chunk affine-summary oracle.
- Implementation: added a private `summary_pair` scan specialization. Each of 80 group owners at T=5,120 retains two state accumulators, skips token-output math and unused q-decay/intra reads, writes A and B directly, and leaves the default KDA path unchanged. The experimental grouped-prefix layer now disables in-scan RMS and applies the existing standalone gated RMS after regrouping, matching the raw-output contract.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Correctness: `QWEN_KDA_GROUP_PREFIX=1 KDA_TP_TEST_SEQ=5120 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> PASS at output/recurrent/convolution PCC `0.999958/0.999890/0.999997`. T=512 summary-produced and summary-consumed gates also passed at `0.999967/0.999923/0.999997`.
- Candidate profile: `QWEN_KDA_GROUP_SUMMARY=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_25_01_00_09/ops_perf_results_2026_07_25_01_00_09.csv`, SHA-256 `36a5c5f12ecb2a31776456b557319ba420d5e8abb604c35794ccaa6502b6843c`. Sessions 2-11 median wall was `3391.428 us`; the fused summary scan median was `134.994 us`.
- Matched control: identical command without the experiment -> PASS. CSV `generated/profiler/reports/2026_07_25_01_02_04/ops_perf_results_2026_07_25_01_02_04.csv`, SHA-256 `8da573ad50efd7c672f89d9ff90d939a1f8b5fb0a9c7eadabfe7af43780ce6bf`. Median wall was `3265.230 us`. Candidate overhead is `126.198 us` (`3.865%`).
- Result: the summary block improves from the prior `70.869 + 91.912 + 21.706 = 184.487 us` to `134.994 us`, saving `49.493 us` (`26.83%`). This validates the fused algebra and removes one program launch, but it remains additive relative to the retained serial endpoint.
- Regression coverage: the full directory run reported `36 passed, 1 failed`; the failure is deterministic and precedes recurrence in the pre-existing single-device perf fixture. Its conv1d requests a `17,301,504 B` L1 buffer requiring `1,572,864 B` per bank, exceeding the empty-bank capacity `1,436,800 B`. The same test fails alone with the identical allocator fatal; all KDA correctness, TP, recurrent, CCL, and recurrence perf cases passed.
- Verdict: retain the opt-in summary specialization as the profitable first half of the distributed-prefix design, not as the default endpoint. The next experiment must fuse prefix correction with the grouped final recurrence; otherwise the added `134.994 us` pass cannot beat the serial scan.


### 2026-07-25 01:38 UTC — Retain receiver-owned distributed affine prefix

- Hypothesis: combining the one-pass affine summaries with the receiver-owned persistent prefix should amortize the protocol cost and replace the 512 us serial recurrence with a grouped final scan.
- Implementation: one core owns each of 80 eight-chunk groups. An inclusive affine scan uses receiver-owned inbound L1 slots, stage-wide arrival/release barriers, and ping-pong local prefixes; each owner emits its corrected group-entry state to the existing grouped recurrence. The path is the default for tile-aligned prefill at T>=5,120; `QWEN_KDA_SERIAL_SCAN=1` is the diagnostic control.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS; 51 build/install steps completed.
- Protocol/correctness: TP=8 T=3,072 (four prefix stages) passed at output/recurrent/convolution PCC `0.999959/0.999912/0.999997`, ruling out the prior sender-owned four-stage deadlock. Promoted-default T=5,120 passed at `0.999958/0.999890/0.999997`.
- Candidate profile: `QWEN_KDA_GROUP_PREFIX_PERSISTENT=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_25_01_31_17/ops_perf_results_2026_07_25_01_31_17.csv`, SHA-256 `17885f711e6ab306dcbd05ecd132cd909e5b9d772b2dc599c7f4bd91a2204124`. Sessions 2-11 median slowest-device wall was `3216.229 us`; affine prefix was `150.427 us`, grouped final recurrence `111.330 us`, and calls/device/replay were 24.
- Matched serial control: `PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_25_01_32_30/ops_perf_results_2026_07_25_01_32_30.csv`, SHA-256 `cf37d309f525e5ccb9bcce08a19caed08362d3673350fae4485df4e6f38451ff`. Median wall was `3262.281 us`, serial recurrence `511.606 us`, and calls/device/replay were 20.
- Result: wall improved `3262.281 -> 3216.229 us`, saving `46.052 us` (`1.41%`). The new recurrence chain costs `134.994 + 150.427 + 111.330 = 396.751 us`, `114.855 us` below the serial scan; standalone gated RMS and four extra launches reduce the wall realization to 40.1% of that block gain.
- Regression coverage: `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests -q -s` -> `36 passed, 1 failed`. The only failure is the deterministic pre-existing single-device perf fixture: Conv1d requests `17,301,504 B` L1 / `1,572,864 B` per bank above the `1,436,800 B` capacity; all correctness, TP, recurrent, CCL, and recurrence tests passed.
- Verdict: retain and promote. The result clears the 1% wall gate and fixes the prior four-stage synchronization failure. Next target is the DRAM group-entry-state boundary between prefix and grouped recurrence.


### 2026-07-25 01:49 UTC — Reject disjoint-consumer grouped fused RMS

- Hypothesis: reuse the scan kernel fused-RMS pipeline for the grouped final recurrence and remove the standalone `KdaGatedRmsOperation` (`83.417 us` ceiling).
- Indexing diagnosis: physical grouped geometry is 80 pseudo-heads x 8 chunks, while logical RMS geometry is 4 heads x 160 chunks. Their flattened work-item order is identical, but batch/head/token decoding must use the logical chunk count. A private logical-RMS-chunk attribute fixed that mapping; T=5,120 then passed at the exact endpoint output/state/conv PCC `0.999958/0.999890/0.999997`.
- Profiler-only capacity failure: the first trace failed because 80 producer cores leave 30 RMS consumers, so uniform staging reserved 22 full-V items (`360,448 B`) on every producer. Tracy then found static CBs ending at `1,061,760` clashing with an L1 buffer at `1,054,464`. Allocating one CB0 tile on producers and full staging only on consumers preserved the common base and made profiling pass.
- Candidate: `QWEN_KDA_GROUP_RMS=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_25_01_46_55/ops_perf_results_2026_07_25_01_46_55.csv`, SHA-256 `76e7b7c58b244d6230ffc8e4d4e89ddb0e44fcc7e75eaa6f3d76c5e2de58d9b3`; median wall `3283.410 us`, 23 calls, grouped scan `187.476 us`.
- Matched control: identical command without the experiment -> PASS. CSV `generated/profiler/reports/2026_07_25_01_48_04/ops_perf_results_2026_07_25_01_48_04.csv`, SHA-256 `0107d0d851179bc77c5b8d65b4e3d4666715ada4cf3722c9158dcceffacb7755`; median wall `3218.490 us`, 24 calls, grouped scan `111.262 us`, standalone RMS `83.560 us`.
- Result: although one launch disappeared, grouped scan grew `76.214 us`; wall regressed `64.920 us` (`2.02%`). With only 30 disjoint consumers, synchronization and reduced RMS parallelism overwhelm launch savings.
- Verdict: reject and fully restore source. Do not retry disjoint streaming consumers for grouped scan. A distinct local-fusion design can execute RMS on each of the 80 full-V group owners after its eight chunks, avoiding cross-core staging; evaluate that separately.


### 2026-07-25 02:06 UTC — Reject owner-local grouped RMS

- Hypothesis: each of the 80 grouped recurrence owners already holds full V=128 output, so applying gated RMS locally can remove the raw-output DRAM round-trip and standalone `83.7 us` RMS launch without the rejected 30-consumer synchronization topology.
- Implementation experiment: the existing RMS arithmetic ran after each group-owned chunk. The reader mapped physical `(pseudo-head, local chunk)` to logical `(batch, token, head)` for gate tiles, and the writer emitted normalized tiles directly in logical token-major order. T=5,120 passed at the exact endpoint output/state/conv PCC `0.999958/0.999890/0.999997`.
- Diagnostic correction: the first profile specified the output as grouped shape and accidentally launched a `68.295 us` reshape despite already writing token-major tiles; that invalid candidate regressed to `3389.026 us`. Giving the output its true logical spec removed the reshape and reduced calls `24 -> 23`.
- Corrected pair 1: candidate CSV `generated/profiler/reports/2026_07_25_02_03_00/ops_perf_results_2026_07_25_02_03_00.csv` (SHA-256 `163fa25647cb0700144bf88baaf973bfa7f8507747d76a91ce77bca8e213ce96`) measured `3216.281 us`; control CSV `generated/profiler/reports/2026_07_25_02_00_21/ops_perf_results_2026_07_25_02_00_21.csv` (SHA-256 `af26f4083ac15c3bc22108bc6619095c0ff852da58767229bc6b021e1dc029bb`) measured `3219.678 us`. The apparent gain was `3.397 us` (`0.106%`), at the continuation threshold.
- Independent pair 2: candidate CSV `generated/profiler/reports/2026_07_25_02_04_24/ops_perf_results_2026_07_25_02_04_24.csv` (SHA-256 `9f16fa8915704707bc36b30867697acc59097ae6a5c07fab40ffab7fed209037`) measured `3227.923 us`; control CSV `generated/profiler/reports/2026_07_25_02_05_38/ops_perf_results_2026_07_25_02_05_38.csv` (SHA-256 `5fb75490d37c05b34da46ebe071f0d3f55d3beeb5ac77adf4ab502027dd3dfba`) measured `3209.753 us`.
- Pooled result: 20 candidate replays had median/mean `3220.836/3222.700 us`; 20 control replays had `3216.827/3215.411 us`. The pooled median regresses `4.009 us` (`0.125%`). Local fusion expands grouped scan from about `111.3 us` to `153.3 us`; removing the standalone RMS does not produce a stable wall gain because that compute is moved onto the recurrence critical path.
- Verdict: reject and fully restore source. Both available grouped RMS topologies are now measured and rejected; do not revisit without overlapping RMS on otherwise idle cores without reducing consumer parallelism.


### 2026-07-25 02:14 UTC — Retain distributed-L1 group-entry states

- Hypothesis: the prefix and grouped recurrence already use the same 80-core row-major ownership, so keeping the 80 FP32 `[128,128]` corrected entry states in distributed L1 should remove their 5 MiB DRAM write/read boundary without merging the two programs.
- Implementation: `kda_affine_prefix` now emits its entry-state tensor to interleaved L1 by default; the existing grouped scan consumes it unchanged. `QWEN_KDA_PREFIX_DRAM=1` preserves the matched diagnostic control.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Correctness: promoted-default TP=8 T=5,120 passed at output/recurrent/convolution PCC `0.999958/0.999890/0.999997`.
- Candidate: `QWEN_KDA_PREFIX_L1=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_25_02_09_42/ops_perf_results_2026_07_25_02_09_42.csv`, SHA-256 `3bd3b9409d4853a7a46942ec7be85aa1fe9b921e5ca4b539b9673eba922f8514`; median wall `3195.611 us`, prefix `129.276 us`, grouped scan `121.559 us`.
- Matched DRAM control: identical command without the experiment -> PASS. CSV `generated/profiler/reports/2026_07_25_02_10_55/ops_perf_results_2026_07_25_02_10_55.csv`, SHA-256 `011ec4c83b83983c38fbec824ba979de6e2fdbae2e56e9bf4f6cb34c874d248c`; median wall `3211.482 us`, prefix `150.211 us`, grouped scan `111.439 us`.
- Result: wall improved `3211.482 -> 3195.611 us`, saving `15.871 us` (`0.494%`). Prefix writes improve `20.935 us`; scan reads regress `10.120 us`, for a net measured recurrence-block gain of `10.815 us`; scheduling/overlap realizes a larger `15.871 us` wall gain.
- Regression coverage: full directory -> `36 passed, 1 failed`; the only failure is the unchanged single-device Conv1d perf fixture whose `1,572,864 B` per-bank request exceeds `1,436,800 B` L1 capacity.
- Verdict: retain and promote. A full same-program handoff is not justified yet; this low-risk storage change captures the measurable boundary reward while preserving both validated protocols.


### 2026-07-25 02:23 UTC — Retain owner-sharded affine summaries

- Corrected hypothesis: affine summaries were already in interleaved L1, not DRAM. Because summary construction and prefix both distribute the same 80 groups row-major, height-sharding A and B as one `[128,128]` shard per owner should eliminate the interleaved-L1 NOC write/read handoff without merging programs.
- Implementation: the default summary memory config is HEIGHT_SHARDED L1 over the first 80 row-major cores with explicit `[128,128]` shards. `QWEN_KDA_SUMMARY_INTERLEAVED=1` preserves the prior interleaved-L1 control.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Correctness: promoted-default TP=8 T=5,120 passed at output/recurrent/convolution PCC `0.999958/0.999890/0.999997`.
- Candidate: `QWEN_KDA_SUMMARY_SHARDED=1 PERF_TRACE=1 PERF_SEQ=5120 PERF_REPS=10 scripts/run_safe_pytest.sh --profile --run-all models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s` -> PASS. CSV `generated/profiler/reports/2026_07_25_02_18_57/ops_perf_results_2026_07_25_02_18_57.csv`, SHA-256 `b72bd0cea8ea5a129de946b631088012c3e6212e2e14a862b765377304812524`; median wall `3183.263 us`, prefix `115.753 us`, grouped scan `121.325 us`.
- Matched interleaved control: identical command without the experiment -> PASS. CSV `generated/profiler/reports/2026_07_25_02_20_13/ops_perf_results_2026_07_25_02_20_13.csv`, SHA-256 `7b48a60387890afe53d016f4158376483c32d6d061e246f2a5020051adfeb759`; median wall `3200.471 us`, prefix `129.370 us`, grouped scan `119.907 us`.
- Result: wall improved `3200.471 -> 3183.263 us`, saving `17.208 us` (`0.538%`). Prefix improved `13.617 us`; grouped scan changed only `+1.418 us`. This validates same-owner sharding as the relevant handoff optimization and rules out the stale DRAM model.
- Verdict: retain and promote. Full summary/prefix kernel fusion now has only launch and CB-handoff upside left; its remaining ceiling is below the original 20 MiB traffic estimate and should be reprioritized behind larger unfused blocks.


### 2026-07-25 02:31 UTC — Reject canonical-offset tiled-input convolution retry

- Hypothesis: the prior residual error came from combining the corrected ceiling-divided projected-row stride with an alternate physical face order. Restoring the canonical BF16 tile layout (TL/TR/BL/BR) should make rows 29-31 use byte offsets `1440-1504` and `1952-2016`, repairing each core first-block prefix.
- Source evidence: production tile-layout kernels document TL/TR/BL/BR at byte ranges `0/512/1024/1536`; the restored reader used bottom-left offset `(row + 16) * 32` and bottom-right `+512`. The host factory used `Pt = ceil(projected_width / 32)`.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS; 53 build/install steps completed.
- Mandatory gate: `QWEN_KDA_TILED_CONV=1 KDA_TP_TEST_SEQ=672 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> FAIL at output/recurrent/convolution PCC `0.951854/0.999912/0.999997`. This exactly reproduces the corrected-stride historical failure (`0.951851` output within measurement variation).
- Diagnosis: the correct recurrent and convolution states rule out projection stride and state update. Canonical physical offsets fail identically, while the earlier alternate offsets worsened row error; therefore face ordering is not the residual cause. The existing row oracle already confines the error to rows 0-2 of each core first block, so the unvalidated raw cross-core prefix import remains the proven fault boundary.
- Restoration: removed the complete private operation, binding, host factory, kernels, and env route. No experiment code remains. Rebuilt the exact restored source with `./build_metal.sh --build-tests --build-type Release` -> PASS, then `KDA_TP_TEST_SEQ=672 scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc -q -s` -> PASS at `0.999958/0.999912/0.999997`.
- Verdict: reject and close direct tiled-input convolution for this campaign. The remaining ~95 us pre-convolution untilize ceiling is not actionable without a standalone validated prefix-import primitive or producer-exported boundary rows.


### 2026-07-25 02:47 UTC — Reject elastic prep/summary split and close the queue

- Hypothesis: use a work-balanced static approximation to elastic
  reassignment—83 prep cores and 27 affine-summary cores—to measure whether a
  one-pass producer/consumer pipeline can overlap the retained `~306 us` prep
  and `~135 us` summary phases. The core-work lower bound is about `404 us`
  versus `441 us` serialized, so perfect scheduling can hide at most `~37 us`.
- Implementation experiment: parameterized prep core count and distributed
  multiple summary heads over fewer cores. Each summary core processed its
  assigned heads sequentially through the existing CB protocol.
- Host build: `./build_metal.sh --build-tests --build-type Release` -> PASS.
- Correctness: `QWEN_KDA_PREP_CORES=83 QWEN_KDA_SUMMARY_CORES=27
  KDA_TP_TEST_SEQ=5120 scripts/run_safe_pytest.sh --run-all
  models/experimental/kimi_delta_attention/tests/test_tp_weights.py::test_tp_layer_pcc
  -q -s` -> PASS at output/recurrent/convolution PCC
  `0.999958/0.999890/0.999997`.
- Profiler result: the corresponding `PERF_TRACE=1 PERF_SEQ=5120
  PERF_REPS=10 ... --profile --run-all` command hung during warm replay twice.
  The second attempt followed a full device reset and reported `260/260` JIT
  cache hits before the same fetch-queue timeout, ruling out first-build
  latency. Both runs required safe-wrapper device reset.
- Diagnosis: numerical work partitioning is valid, but the multi-head reuse of
  the existing summary CB protocol is timing-sensitive under trace replay and
  cannot be retained. Because no stable profile exists, no speed claim is
  made. The experiment source was fully removed.
- Restoration: `./build_metal.sh --build-tests --build-type Release` -> PASS;
  the exact retained `KDA_TP_TEST_SEQ=5120` TP=8 gate passed at
  `0.999958/0.999890/0.999997`.
- Residual-fusion bound: in retained CSV
  `generated/profiler/reports/2026_07_25_02_18_57/ops_perf_results_2026_07_25_02_18_57.csv`,
  the summary-to-prefix `OP TO OP LATENCY` across 88 device/replay observations
  is `0.539 us` median and `0.577 us` maximum. Since summaries are already
  owner-sharded in L1, that dispatch gap is the remaining fusion reward:
  `0.0169%` median (`0.0181%` maximum) of the `3183.263 us` retained wall.
- Verdict: reject the elastic split and stop this optimization campaign. No
  identified actionable idea remains above the `0.1%` (`3.183 us`) whole-layer
  threshold.
