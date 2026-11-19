# Prefill norm families: bounded independent probe, UNTRIED hand-off

Status: source preparation and CPU plan checks only. No TTNN import, device execution, build, production edit, policy/default change, or performance claim was made by the preparing agent. Frozen at the user's wrap-up cap. Parent owns any later hardware queue.

## Artifacts and contract

- `tests/probe_prefill_norm_families.py`: one real layer, one norm phase, one public workload per process.
- `prefill_norm_families27_cpu.json`: layer 0/input/B3S33 CPU check.
- `prefill_norm_families27_full_cpu.json`: layer 3/post/B32S513 CPU check.

Three coherent paths start with TP4 hidden shards and end with the same selected real next projection:

| Family | Existing implementation | Movement |
|---|---|---|
| distributed | `MultichipDecoder._norm`, gather disabled | BF16 partial-stat AG, local post-normalization, normalized-activation AG |
| gather_explicit | activation AG then frozen `OptimizedDecoder._norm` explicit FP32 sum branch | one full-activation AG; FP32 arithmetic and gamma, BF16 result |
| gather_sfpu | activation AG then frozen `prefill_hidden_rmsnorm_sfpu` | one full-activation AG; existing frozen kernel, no copied or changed arithmetic |

This is a focused norm/projection experiment, not a whole-decoder correctness or latency substitute. No intermediate restore to an obsolete replicated residual contract is introduced. All paths consume sharded residual input; the replicated normalized activation is the actual projection input. The distributed family is explicit even where the selected B1 policy normally gathers first. Each candidate's complete resolved policy is recorded.

## Independent quality evidence the probe will require

The probe loads real layer weights using the existing model/revision loader. It reads uploaded BF16 shards back from all four ranks, reconstructs the full hidden vector, and requires bitwise equality to the fixture. The norm oracle computes `x * rsqrt(mean(x**2) + epsilon) * gamma` in CPU FP64 using those reconstructed values and actual FP32 `1 + weight` gamma. Every logical row on every rank must reach PCC >= .995. Nonfinite output fails; constant rows require exact equality rather than an invented PCC.

Projection reference inputs are the independent FP64-normalized values rounded to BF16. They pass through the unchanged selected real-weight projection. Projection gates include every logical row/rank and the meaningful packed head/beta/decay slices (or post-norm SwiGLU output). This isolates norm errors; it does not independently revalidate the native matmul implementation. All eager family metric failures are collected and saved before the quality assertion, so an early distributed failure cannot conceal a gather-first failure.

Recorded text fixtures are autodiscovered for B1S512 and B3S33 (first three users of the existing B32S33 fixture). An explicit `--activation-file` is also supported. B32S513 uses labelled seeded synthetic input if no matching fixture exists. In post phase, recorded layer-input activation is explicitly labelled as a post-gamma fixture, not misrepresented as actual post-attention residual. `--supplemental` adds nearzero (`1e-6`), scaled (`4096`), and NaN-poisoned physical dead rows and is bounded to <=2048 logical rows. The poison upload is read back to prove the expected NaN count; only logical outputs enter the quality gate.

## Shape and projection fidelity

The probe owns physical tile padding and logical metadata. B3S33 uses `[1,99,1280]` with 128 physical rows. B1S512 retains `[1,512,1280]`. B32S513 follows the current temporal schedule: 512-token bulk `[1,16384,1280]`, then original single-token-tail `[32,1,1280]`; it does not label a synthetic 16416-row matmul as the measured model path. CPU checks prove exact temporal coverage and legal padded extents. Admission requires current flat-prefill policy and 512-token chunks; hardware setup verifies `_can_flatten` agrees with this plan.

Input GDN projection uses `_prefill_packed_gdn` for flattened bulk and `_linear(...,"packed_gdn")` for legacy/B1/tail paths; full attention uses selected packed QKVG. Post projection is extracted as an AST from the actual `_finish_flat` or `_finish` h-to-act statements. This preserves selected minimal-matmul configuration, optional large-row SwiGLU chunking, and optional packed tail epilogue without copying a stale configuration into the harness. The extracted AST is hashed and rejects inclusion of norm/reduction/collective calls. No production function is modified. This extraction is a test-harness technique, not a proposed runtime implementation.

## Lifetime, cache and timing design

Each family has original/rebound owners with independent gamma, input, CCL semaphores, persistent CCL buffers, and output slots. Both owners use actual gamma. Poisoning the old gamma to zero must affect only the old owner; invoking the rebound owner with changed input must reproduce its independently gated result and leave old outputs untouched. Gamma is restored before capture.

All input banks, aliases, gamma, verification outputs and lazily prepared CCL buffers are allocated before the first capture. Four complete warm passes cover semaphore ping-pong and every shape. Program-cache misses are then prohibited. Quality traces copy norm/projection outputs into preallocated slots; timing traces contain the same boundary without those verification copies. Capture outputs are temporary and released before ending capture; no exemptions or acknowledged-corruptible scopes exist. Allocation tracking must reject a deliberately live post-capture buffer and recover after its release. Changed-input replay, 50 queued alternating-input launches per owner/window, and eager rebound invocation followed by original-owner trace replay require exact equality to independently gated eager outputs. Input/output/CCL addresses, keys and policies must remain stable.

Timing is disabled under tracker, watcher (environment presence), or profiler. An uninstrumented run reports seven synchronized single-boundary samples for eager and trace; B32S513 samples include both sequential windows. Host API policing is outside the timed eager interval. Timed results exclude verification copies, host conversion, and input resets; inputs are stateless. The probe always reports that these are boundary timings rather than whole-layer timings. Whole-model policy adoption requires the existing whole-decoder gates and profile evidence afterwards.

## Bounded proposed commands (NOT executed)

Run the first smoke only after the parent releases the queue. Substitute the then-final frozen policies and record their hashes; the following existing files are source-preparation examples, not assertions that an old policy is the final default:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
P0=$D/combine_window27_b32_s513_window.policy.json
P3=$D/long_prefill_profile25_l3_b32_s513.policy.json
MOD=models.autoports.qwen_qwen3_6_27b.tests.probe_prefill_norm_families
env -u TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m "$MOD" --layer 0 --phase input --batch 3 --length 33 --policy "$P0" \
  --supplemental --queued 50 --skip-timing --output "$D/prefill_norm_families28_l0_input_b3_s33_quality.json"
```

If that passes, a six-case representative matrix is enough for this family, without a broad shape sweep:

| Layer | Phase | B,S | Supplemental |
|---|---|---|---|
| 0 GDN | input | 3,33 | yes, first smoke |
| 3 full attention | post | 3,33 | yes |
| 0 GDN | post | 32,513 | no |
| 3 full attention | input | 32,513 | no |
| 0 GDN | input | 1,512 | no, recorded text |
| 3 full attention | post | 1,512 | no, recorded text |

Use the same command structure for each quality row. After its quality pass, run the same row uninstrumented without `--skip-timing`:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER -u TT_METAL_TRACE_ALLOC_TRACKING \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m "$MOD" \
  --layer 0 --phase input --batch 3 --length 33 --policy "$P0" --queued 50 --samples 7 \
  --output "$D/prefill_norm_families28_l0_input_b3_s33_perf.json"
```

The complete original/rebound eager oracle matrix is intentionally stronger than relying on the whole-TTNN reference. Large-row runs retain host projection references and six owners' fixed output buffers; budget several GiB of host and device memory and allow CPU metric computation time. No hardware memory/runtime estimate is represented as measured.

## Open risks / untried items

- Metadata views preserving NaN physical dead rows, all selected projection shapes, AST-extracted optional epilogues, and strict tracker behavior are source-admissible but hardware-unverified in this new harness. A construction/API error must be diagnosed and adapted, not treated as rejection of the family.
- SFPU output padding is not asserted to be initialized; only valid rows matter, and poisoned-input padding tests whether later selected projection preparation respects that contract.
- Narrow beta/decay per-row projection gates can expose cancellation missed by flattened PCC; the .995 gate is unchanged and must not be waived.
- Public B32S513 tail retains per-user padded shape; no compact-tail speed claim is made.
- Future retained family needs whole-layer integration/performance, both norm phases on final policies, final-default reproduction, and stage-review. This hand-off supplies none of those completed claims.
