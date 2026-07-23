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
