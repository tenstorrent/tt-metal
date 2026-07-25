# Functional Decoder Stage Review — Round 1

Verdict: **MORE WORK NEEDED**

Review date: 2026-07-25

The emitted graph translation, dense TP4 weight reconstruction, host-fallback audit, context contract, and recorded numerical results are substantially complete. Three stage-relevant gaps remain before this functional-decoder checkpoint can be accepted.

## Required Work

### P2 — Restore the emitted/default compute-kernel policy

`tt/functional_decoder.py:115-121` creates one global `WormholeComputeKernelConfig` with HiFi4, approximation disabled, FP32 destination accumulation enabled, and L1 packer accumulation enabled. That same configuration is passed broadly at `tt/functional_decoder.py:310-489`, including QKV and output projections, prefill/decode SDPA, router projection, and expert gate/up/down matmuls.

This does not match the selected EmitPy layer. In the representative prefill graph, only the RMS norms at `g0_prefill/main.py:3879` and `g0_prefill/main.py:3972` carry the HiFi4/FP32-accumulation configuration; the QKV, attention, output, router, and expert matmuls use `compute_kernel_config=None` at `g0_prefill/main.py:3883`, `3911`, `3943`, `3949`, `3977`, `4007`, and `4018`. The decode graph has the same distinction: configured RMS norms at `g1_decode/main.py:3318` and `3378`, with default-config linear/matmul/attention operations around `3322-3424`.

The functional-decoder skill requires framework-default compute-kernel behavior unless an emitted or evidence-backed numerically sensitive operation needs an override. The current global override silently changes the emitted per-op policy and locks a high-fidelity/FP32-accumulation choice into the functional baseline without an ablation showing it is necessary.

Required resolution:

- Omit the custom compute configuration for operations whose emitted configuration is `None`.
- Preserve the emitted configuration for RMS norm, or document and narrowly scope any additional exception with a default-versus-override PCC result.
- Rerun the six-test decoder suite and record the synthetic, official-HF prefill/decode, capacity, and context results after the change.

### P2 — Add a decode control at the sliding-window boundary

Layer 12 is a sliding-attention layer with window size 128. The emitted decode graph consumes an explicit mask and calls decode SDPA with `is_causal=False` at `g1_decode/main.py:3349-3350`. The implementation intentionally maps that representation to causal decode SDPA using `cur_pos_tensor` and `sliding_window_size=128` at `tt/functional_decoder.py:409-419`.

The only official-HF decode comparison populates a 17-token prefix and decodes position 17 at `tests/test_functional_decoder.py:257-292`. Since no cached token is outside a 128-token window, this test cannot distinguish correct sliding-window masking from full causal attention. The S128 and S256 synthetic results exercise prefill, not the decode cache/mask mapping. Consequently, the claim in `doc/functional_decoder/README.md:97-103` that both emitted-path equivalences are covered is stronger than the evidence.

Required resolution:

- Add and record at least one HF-reference decode PCC control with a populated cache at or beyond the 128-token boundary, so that an old key must be excluded.
- Prefer controls immediately around the boundary plus one clearly beyond it, while retaining the existing position-17 control.
- Keep the official layer PCC floor at 0.99 and record the exact cache position, prefix length, and observed PCC.

### P2 — Make multichip tensor provenance exhaustive for the selected layer

`doc/functional_decoder/multichip_provenance.json` gives strong coverage of the 17 parameter tensors, cache tensors, boundary tensors, and the representative layer's source collectives. However, its activation/cache inventory at lines `292-440` records `query_heads`, cache storage, RoPE/mask/position inputs, sinks, and routing indices/bases without recording all sharded intermediates in the selected flat-graph range.

Examples absent from the tensor inventory include the TP-local fused QKV activation, the split key/value head activations, post-RoPE key activation, local attention/head-concatenation activation, and expert-axis intermediate activations. These appear in the prefill range at `g0_prefill/main.py:3887-3949` and the decode range at `g1_decode/main.py:3326-3355`. They are material to reconstructing the original TP4 axis and to distinguishing tensors that are dense only after the single-device collapse.

The IR-to-functional-decoder skill's provenance contract calls for every sharded tensor and collective, including boundary activations, attention heads, KV/cache tensors, and MoE intermediates. A collective-only inventory is insufficient for this downstream multichip prior.

Required resolution:

- Extend the JSON with every TP-sharded intermediate in the representative prefill and decode ranges, recording global shape, per-device shape, shard axis, TP degree, and single-device collapse.
- Include the fused/split QKV activations, attention-output intermediates, and expert-axis/router intermediates, or provide a demonstrably complete graph-derived tensor inventory that subsumes them.
- Revalidate the JSON and update the documentation counts after the inventory is complete.

## Other Concerns

- `doc/functional_decoder/README.md:138-139` says there is no fused implementation, while `tt/functional_decoder.py:181-186` deliberately fuses the reconstructed Q/K/V weights into one dense QKV projection. This appears to mean “no optimized fused-kernel stage,” but the wording should be made explicit to avoid contradicting the implementation.

## Hard-Check Gaps

- Per the review mandate, no TT device command, pytest invocation, target-decoder import, reset, or profiler command was run in this review. The persisted JUnit artifact parses successfully and records 6 tests, 0 failures, 0 errors, and 0 skips. The detailed PCC/capacity values are recorded in README/work log and were supplied as direct stage evidence, but are not embedded in the JUnit XML itself.
- The official checkpoint test pre-dequantizes packed expert weights before calling `load_state_dict`. It validates the dense canonical expert math, but it does not directly exercise the raw packed-weight branch in `_dense_expert_weight` at `tt/functional_decoder.py:52-61`. This is residual load-path risk, not a blocker for the stated dense-canonical functional contract.

## Anomaly Ledger

### Global compute policy differs from the emit

- **Observed anomaly:** One high-fidelity/FP32-accumulation configuration is applied to nearly every compute operation although the emit specifies it only for RMS norm.
- **Affected path:** Prefill and decode; attention, dense projections, router, and experts.
- **Evidence:** `tt/functional_decoder.py:115-121,310-489`; `g0_prefill/main.py:3879-4018`; `g1_decode/main.py:3318-3424`.
- **Simplest control:** Remove the override from one non-norm operation or all emitted-`None` operations and compare the existing PCC suite.
- **Likely subsystem:** Emit-to-functional translation of per-op execution attributes.
- **Investigation performed:** Compared the selected prefill/decode layer call sites operation by operation and checked the functional-decoder skill's compute-config rule.
- **Outcome:** Confirmed mismatch; not explained by the current documentation or test evidence.
- **Resolution:** Required before acceptance.

### Sliding decode mapping is tested only below the window size

- **Observed anomaly:** The implementation replaces emitted explicit-mask decode with TTNN causal/sliding-window decode, but the only real decode control is at position 17 for a window of 128.
- **Affected path:** Decode attention and KV-cache semantics for sliding layers.
- **Evidence:** `tt/functional_decoder.py:409-419`; `tests/test_functional_decoder.py:257-292`; `g1_decode/main.py:3349-3350`.
- **Simplest control:** Compare TTNN and HF at a cache position where at least one old token lies outside the window.
- **Likely subsystem:** Decode mask lowering and cache-position semantics.
- **Investigation performed:** Traced the emitted decode attention call, implementation call, layer configuration, and every recorded decode test position.
- **Outcome:** Existing evidence cannot exercise the semantic difference.
- **Resolution:** Required before acceptance.

### Provenance omits sharded transient activations

- **Observed anomaly:** Parameter and collective provenance is detailed, but several TP-local attention and MoE intermediates are absent from the tensor inventory.
- **Affected path:** Downstream multichip reconstruction from the selected prefill/decode layer.
- **Evidence:** `doc/functional_decoder/multichip_provenance.json:292-440`; `g0_prefill/main.py:3887-3949`; `g1_decode/main.py:3326-3355`.
- **Simplest control:** Enumerate every intermediate tensor in the selected layer ranges and reconcile that list against the JSON.
- **Likely subsystem:** Provenance extraction/documentation completeness.
- **Investigation performed:** Cross-checked the JSON tensor categories and collective records against both selected flat-graph ranges.
- **Outcome:** Confirmed missing transient sharded tensors.
- **Resolution:** Required before acceptance.

## Verified Strengths

- The documented source hashes match the four pre-generated EmitPy inputs. No MLIR or `ir_to_emit` conversion artifact is used by the stage.
- The representative layer-12 ranges contain the expected two RMS norms, QKV attention, RoPE/sinks/cache operations, output projection, router/top-4 selection, eight local experts, source collectives, and residual ordering.
- The layer-12 parameter transformations reconstruct dense canonical HF shapes, including Q/K/V concatenation order, output/router transposes, expert gate/up interleave, biases, norm weights, and attention sinks.
- `FunctionalDecoder` subclasses `LightweightModule`, provides both prefill and decode entry points, enforces the documented batch/shape/context constraints, and uses a 1x1 mesh after dense TP4 collapse.
- An independent AST/static scan of the runtime methods found no torch execution, TTNN host transfers, host fallback, layout/memory-conversion glue, resharding, or runtime collective calls.
- The context checker passes both exact-contract and strict-cap modes with target context 131072 and honest supported capacity 21248.
- Python AST parsing, JSON parsing, XML parsing, and all four source-hash checks pass.
- The recorded device evidence exceeds the required 0.99 PCC floor: synthetic S17/S128/S256 near 0.99999, official-HF prefill 0.9997057, and official-HF decode at position 17 of 0.9996046. The recorded capacity boundary is S21248 success and S21249 expected device OOM.

## Scope Inspected

- Goal contracts: `forge-functional-decoder`, `forge-functional-decoder-from-ir`, `tt-device-usage`, and `stage-review` skill instructions.
- Implementation: `tt/functional_decoder.py`, target package initializers, the functional test suite, and capacity probe.
- Stage artifacts: README, work log, context contract, multichip provenance, and JUnit XML.
- Emit sources: layer-12 ranges and constant-evaluation transforms in `/home/mvasiljevic/emit-gptoss/g0_prefill` and `/home/mvasiljevic/emit-gptoss/g1_decode`.
- Reference architecture: local GPT-OSS-20B config and installed Hugging Face GPT-OSS modeling source.
- Repository state: branch `mvasiljevic/gpt-oss-pipeline-progress`, inspected at HEAD `dd34ac32928d704bf0aff87fd25f047c5fbb6af0`; the visible worktree scope was confined to `models/autoports/openai_gpt_oss_20b/`.
- Read-only checks: repository status/inventory, source hashes and line counts, targeted source inspection, AST operation/token scans, JSON/XML parsing, and both context-contract checker modes.

## Residual Risk

After the three required items are addressed, the main remaining risk is hardware-path regression from changing the broad compute configuration. The existing six-test suite is the appropriate control, supplemented by the new boundary-position decode comparison. No evidence in this review suggests a defect in the decoder math, dense TP4 reconstruction, or current sub-window numerical results.
