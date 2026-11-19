# Focused saved-stream capture for the layer34 regression

[tests/v5_layer_stream_probe.py](../../../tests/v5_layer_stream_probe.py) is a new diagnostic based on the existing routing probe. Production and the original harness are unchanged. Its default run captures positions137/165/181 and **stops after position181:54 decode steps**. Optional `--include-worst` additionally captures position7944, selected from the failure JSON, and extends the same replay to that point. It never regenerates inputs with a shorter RNG request.

The root's [V5 failure](v5_representative_layer34_stress_seed154.json) and [V4 control](v4_layer34_stress_seed154.json) share the worst position7944, with contribution PCC0.9125414665356525 and approximately0.91268105 respectively. This shared failure is not explained solely by V5's new native output normalization. The probe does not declare a diagnosis or change any acceptance gate.

## Input and execution identity

The probe loads `doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt` verbatim, checks layer/seed/revision/Torch version and all frozen oracle source hashes, checks shapes/dtypes, and matches the failure report's oracle key. It hashes the entire oracle and each full prefix/token/continuation tensor. Fresh HF prefill and every captured HF output must be bit-identical to the frozen oracle. Layer34, seed154 and the full8192-token tensor remain fixed even when only54 decode steps execute.

The root-run device mode uses the actual current `FusedDecoder` methods and actual HF decoder on every executed token. **Execution is eager and serial with synchronous host observers; it is not the accepted trace replay and is not a timing or acceptance run.** A mismatch between this eager diagnostic and traced failure must remain visible, not be interpreted as a repair. The captured JSON records source hashes, stream fingerprints, execution regime, final device-close status and source-unchanged status. Each `.pos*.pt` capture has its own SHA256 in the JSON.

## Mutation-safe capture

Every TT producer capture calls `ttnn.to_torch(...).detach().cpu().clone()` immediately, before subsequent TT operations execute. No live TT intermediate is retained for delayed readback. In particular, `_recurrent_step`'s incoming FP32 state is copied to CPU before the original method runs; its final add may safely write into the original stable cache without changing the saved input. Both `round_state` and `output_state_is_input` are recorded. HF hooks also clone immediately, including convolution history and incoming recurrence state before their cache updates.

Captured stages include inputnorm, packed Q/K/V/Z/A/B projections, convolution input history/output/FP32 accumulator, normalized Q/K, V, beta, actual decay, recurrence incoming state, decay/memory/delta/new state/core, gated core, attention output, postnorm and router logits/IDs/scores. HF g is captured; TT g is compared only if actually computed (current decode supplies decay directly). Native16 Q/K heads are repeated only on the diagnostic CPU side to compare with HF32 heads. HF cache BF16 and TT raw FP32 cache are reported separately from TT's effective BF16 public-boundary values. Core comparisons include the raw FP32 reconstruction and the actual HF BF16 output boundary.

Actual HF core/state outputs remain the primary reference. Intermediate canonical recurrence reconstruction is explicitly stored under `canonical_reconstruction`; its next state must equal the actual HF returned next state. Local CPU controls evaluate HF gated norm on TT core/Z, HF postnorm on the TT attention residual, and CPU router projection on TT postnorm. Direct module `forward` calls bypass hooks so these diagnostic controls cannot overwrite the actual HF captures. Router reports include actual HF/native IDs and scores, canonical softmax/top8 selection from each logit source, and the eighth/ninth logit gap. These distinguish input drift, local arithmetic, and selection discontinuity.

## Host verification and root command

[CPU validation](v5_layer_stream_probe_cpu.json) passes full saved-stream/source identity checks without importing TTNN or opening a device. A direct CPU execution of the probe's actual HF recurrence callback on a synthetic one-token fixture passed its reconstruction assertion and confirmed the saved incoming state owns separate storage. Pre-commit passes all applicable hooks for the new script. No hardware run was performed by the preparer; native hook/readback behavior remains to be exercised by the root.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_layer_stream_probe \
  --layer 34 --seed 154 --positions 137 165 181 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_layer34_stream_probe.json
```

If the early captures do not localize the failure, run the same command with `--include-worst` and a distinct output basename such as `v5_layer34_stream_probe_worst.json`. This reuses the full saved token tensor and extends replay through7944; no short-stream oracle is created. Root owns all device execution.

## Snapshot-loader follow-up and completed capture audit

The original root-run [V5 early capture](v5_layer34_stream_probe.json) completed and closed its device. Its measured probe hash remains `df0cc13240e0250bf067b9ab8dc84ecb88677d9c6f13f48a1fa2c2b0071d66fd` in the unchanged result and [provenance](v5_layer34_stream_probe.json.provenance.json). The original tensors/results are retained. Current probe SHA256 is `8c0da5535cedc1c40fae9ef6ad17e2498975950beae0182be38565121847f34e` after adding explicit `--snapshot`; default execution still imports the production class.

For an explicit snapshot, a `SourceFileLoader` imports only the selected isolated class. The probe validates the source exists and defines `FusedDecoder`, records the authoritative `measured_decoder_source` and SHA256 in the JSON, provenance sidecar and each tensor capture, checks the imported class's actual file, and verifies source stability. The generic provenance's production-file hash is not substituted for the executed candidate hash. CPU-only mode parses/hashes the selected candidate without importing it or TTNN. Existing output/provenance paths are rejected to preserve prior evidence.

[Snapshot CPU validation](v5_layer_stream_probe_snapshot_cpu.json) passes against `v5_ordered_qz_postnorm_bf16_gate.py.snapshot`, candidate SHA256 `16fdf3f632f685af1b8e279f933fb28137308d8dc1f3965dafbb50c797401870`, with the original full-stream fingerprints and54-step default. Pre-commit passes for the updated probe. No snapshot device execution was performed by this agent.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_layer_stream_probe \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_qz_postnorm_bf16_gate.py.snapshot \
  --positions 137 165 181 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_combined_repair_stream_probe.json
```

### Decay and state observation defects are absent

The old routing probe's decay predicate also matched any multiply with an input activation; its later scaled-Q product could therefore overwrite the saved decay. This new probe instead captures only `multiply` whose first Python operand **is the incoming state object**. The later Q product receives a reshaped rank5 state object, so it does not match. All observations are immediate CPU clones; incoming state is read before the in-place final writer.

A fresh CPU check of the saved137 and181 tensor captures proves, separately at each point:

- `decayed_state == incoming_state.bfloat16().float() * decay_value` exactly, maximum absolute error0.
- Saved incoming and next-state tensors have independent storage and different contents.
- Saved `core == reduced_output.permute(0,2,1,3)` exactly; saved public FP32 state equals the captured new state exactly.
- On identical saved TT operands, canonical CPU `(decayed*k).sum(-2)`, delta, new-state multiply/add, and scaled-Q FP32 core all equal their TT captures bit-for-bit.
- Saved TT decay equals HF decay exactly at these two points.

These checks use actual root-run device captures, without rerunning hardware. They establish the mapping and local recurrence arithmetic at these points, not universal equivalence or a model pass. The observed core discrepancy against HF is compatible with different incoming state and Q, rather than a mislabeled decay or a new local reduction error. Incoming effective BF16 cache differs from HF by105 codes at137 (maximum absolute difference0.0009765625) and404 at181 (0.001953125); current K/V are exact but Q differs. This does not yet distinguish prefix-state differences from earlier uncaptured frontend differences. An additional early capture at128 can inspect the first public incoming cache if that distinction remains necessary after the combined frontend repair.

Actual routing confirms the discontinuity: position137 selects HF expert89 versus TT59 in the eighth slot; position181 includes HF184 versus TT52. The root owns the combined Q/Z, norm and BF16 gate controls; these observations do not independently select a repair.

## Actual norm-input capture and gate-substage follow-up

The [combined candidate capture](v5_combined_layer34_stream_probe.json) is preserved with its original script/source provenance. Its previous `post_norm_local_arithmetic` comparison used a **CPU-reconstructed BF16 residual**, not the actual TT norm input. Consequently its392/490 differing BF16 codes at137/181 can include residual ADD error; it did not by itself establish a local norm error.

Current probe SHA256 `bf54d1d716bee3d3edca2ac37157931b03ede677058a7d14e1ecdba2a78b598d` adds immediate, pre-call CPU copies of actual TT `input_norm_input` and `post_norm_input`, and corresponding HF pre-hooks. It now reports:

- `post_norm_input_vs_cpu_bf16_residual`: actual TT input versus CPU BF16 token+captured attention.
- `post_norm_local_arithmetic`: canonical HF norm on the **actual TT input** versus actual TT norm output.
- `post_norm_local_arithmetic_assuming_bf16_residual`: the original calculation retained under an explicit name.

This distinguishes residual addition from normalization without weakening either comparison. Captures retain both CPU residual reconstruction and CPU norm results on actual/assumed inputs. The existing in-place-state and decay predicates remain unchanged.

For a candidate that executes the combined gate's native unweighted `rms_norm` on shape `[1,32,128]`, optional immediate captures also save the actual unweighted RMS input/output, its gamma-product output and its SiLU-product output. The producer identities are scoped to one attention call and cleared afterward. No delayed TT readback is used. Comparisons independently test unweighted RMS against canonical FP32 mean/rsqrt then BF16, gamma multiplication on the actual normalized values, SiLU multiplication on the actual weighted values, and the final head merge. Original V5's fused native gate does not execute this exposed RMS substage, so no nonexistent substage is invented.

[CPU validation](v5_layer_stream_probe_norminput_cpu.json) passes positions128/129/130/137/181, including the first public post-prefill state; pre-commit passes. New device execution remains root-owned. Use a fresh output basename, for example `v5_combined_norminput_stream_probe.json`, with the same `--snapshot` command above and `--positions 128 129 130 137 181`. Source is frozen after this update.

### CPU/source observations, pending actual residual and RMS captures

The combined candidate's saved gated-core local comparison differs by315 BF16 codes at181 (maximum absolute difference0.00390625), on identical saved TT core/Z. This already excludes upstream core/Z differences as the explanation for those **local** codes, but the previous captures cannot separate unweighted RMS, gamma and SiLU arithmetic. Native layernorm source uses an FP32-reduction path plus SFPU rsqrt (`ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp:282–303`) and a `ckl::BinaryFpu<Mul>` normalization product (`:314–335`). That FPU product is a concrete precision distinction from the canonical CPU FP32 product; it is a hypothesis, not a measured attribution. The new substage captures discriminate it.

The residual-ADD hypothesis has additional CPU support. On saved TT attention/token operands, replacing CPU BF16 ties-to-even addition with a simulated nearest/ties-away BF16 result reduces the resulting canonical-postnorm code mismatch from392 to28 at137 and490 to49 at181. Simple truncation instead worsens it to1026/1056. This simulation does not prove the actual hardware input codes and does not replace their capture. `binary_ng/device/binary_ng_device_operation.cpp:21–43` explicitly documents that ADD with `fast_and_approximate_mode=False` selects the BF16 nearest-even SFPU path. The parent owns that isolated residual control; no production change is made here.
