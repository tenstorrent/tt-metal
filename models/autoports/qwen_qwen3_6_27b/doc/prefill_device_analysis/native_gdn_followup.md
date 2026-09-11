# Native GDN recurrence follow-up

The existing `ttnn.transformer.chunk_gated_delta_rule` changes the optimization ceiling substantially. A reduced real-weight run with outer chunk128 takes **5.475 ms per linear layer at S128**, versus 51.119 ms for the original sequential graph (9.34×). At S4096, outer chunk512 takes **69.316 ms**, versus 1,670 ms (24.1×). These are synchronized warmed per-layer measurements, not full-model or serving TTFT claims. Parent investigation owns full-model integration, HF validation, and CI readiness.

The native core is also closer to a full-sequence FP64 recurrence oracle than the existing sequential core. It is not bitwise equivalent, so agreement with the original implementation is insufficient as the sole correctness gate.

## Focused AutoFix experiment

Starting evidence: the affine scan variants in `README.md` / `artifacts/kda_scan*_s128.json` had first32 final-state relative L2 around 6.7%, versus sequential 0.85%, even with FP32 A/B tensors. Rather than continue changing the dense scan's precision, source discovery found a specialized chunked delta-rule implementation. The native algorithm avoids explicitly composing dense 128×128 affine transforms per token.

Hypothesis: the repository's native GDN op implements this exact recurrence and will remove most launches and improve numerical behavior. The bounded experiment monkeypatches only `_sequential_recurrence`; projections, convolutions, normalizations, masks, real weights, CCLs, and generator remain those in the original autoport.

The adapter in `scan_followup.py` converts existing head-major Q/K/V into the native token-major contract, converts beta to FP32, and reconstructs log-decay with `log(decay)` in FP32. Query is already normalized and scaled, so `scale=1`, `use_qk_l2norm=False`. Native state is FP32; existing caller still writes its selected BFP8 recurrent cache. Internal native chunk size is always32; the independent outer chunk sizes below amortize the rest of the Python graph. Constant eye/tril/ones/quadrant-mask tiles are allocated once per generator before warmup.

| Sequence | Outer chunk | Linear layer median | Reduced generator median | Final cache PCC vs original | Final cache relative L2 vs original |
|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 15.171 ms | 27.650 ms | 0.999829 | 1.95% |
| 128 | 128 | 5.475 ms | 17.083 ms | 0.999812 | 2.03% |
| 4096 | 32 | 424.238 ms | 467.918 ms | 0.968700 | 24.88% |
| 4096 | 512 | 69.316 ms | 96.667 ms | 0.968580 | 24.93% |

All four runs used three warmed samples, batch1, TP4, real layers0/3 and selected precision. All checked tensors were finite; terminal logit top1 matched the original. At S4096/outer512, layer0 PCC was0.999882 and terminal logit PCC0.999670. These reduced checks do not establish autoregressive full-model accuracy.

The initial suspicion that the large long-context state difference came mainly from less frequent BFP8 cache writes was **refuted**: forcing the native path back to the original outer32 cadence barely changed the difference.

## Oracle localization

First32 native state relative L2 against FP64 on identical TT-preprocessed inputs is **0.1773%**, versus original sequential **0.8513%**. With outer128, the complete128-token native state relative L2 is **0.1677%** and output relative L2 **0.1171%**.

For the stronger long-context control, one warmup captured every Q/K/V/beta/decay chunk for real layer0 at S4096. `scan_oracle.py` then ran FP64 sequential recurrence on the CPU from a zero initial state across all4096 tokens. This oracle performs no intermediate BFP8 cache quantization. Both compared device implementations use the selected BFP8 cache and outer32. The original and native paths have identical upstream computation at this outer size; convolution cache comparison is exact.

| Final state against full4096 FP64 oracle | Relative L2 | PCC | Max absolute error |
|---|---:|---:|---:|
| Original sequential graph | 26.1024% | 0.968738 | 14.9361 |
| Native chunk GDN | 2.0292% | 0.999794 | 0.5406 |

The result supports the native rewrite and shows why failing a strict original-cache equivalence check cannot be equated with worse accuracy. It does not attribute all original error to a single operation or cache boundary; the native implementation changes intermediate precision, rounding, and algorithm together. The dense affine scan's specific residual error remains unlocalized; it is superseded as the primary performance candidate, not declared fixed.

Artifacts: `artifacts/native_chunk*_s*.json` and `artifacts/native_full4096_oracle.json`. Large reproducibility tensors remain in `/tmp/native_real_inputs4096.pt`, `/tmp/native_chunk32_s4096.pt`, and `/tmp/qwen_baseline_s4096.pt`. Capturing inputs is outside the timed measurements; the capture run is separately labeled and is not the table's performance result.

## Reproduction

From the repository root, prepend this environment to each device command:

```bash
env PYTHONPATH=. HF_HUB_OFFLINE=1 \
  QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B \
  QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  QWEN36_LINEAR_PREFILL_CHUNK_SIZE=128 \
  python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/scan_followup.py \
  --candidate baseline --sequence 128 --oracle --iterations 3 \
  --reference /tmp/qwen_real_prefill_baseline.pt \
  --result models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/native_chunk128_s128.json
```

`--candidate baseline` configures the original graph before the isolated wrapper replaces recurrence; the wrapper labels the resulting JSON `native_chunk`. Change outer chunk/sequence to32/128,32/4096,512/4096 for the other rows, and use `/tmp/qwen_baseline_s4096.pt` for long-context comparison. Omit `--oracle` for the long performance commands.

Capture command uses outer32/S4096, `SCAN_CAPTURE_INPUTS=/tmp/native_real_inputs4096.pt`, `--iterations 1`, and `--save /tmp/native_chunk32_s4096.pt`. Then:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/scan_oracle.py \
  --inputs /tmp/native_real_inputs4096.pt \
  --baseline /tmp/qwen_baseline_s4096.pt \
  --native /tmp/native_chunk32_s4096.pt \
  --result models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/native_full4096_oracle.json
```

No device recovery was needed. All five hardware jobs closed the mesh successfully. No native C++ source was changed; these experiments used the installed native op. Python files were Black-formatted and parsed; no C++ build is required for this isolated Python experiment. Watcher/profiler/full-model validation are not claimed here.

## Why this was missed, and what remains

The dedicated op was added by commit `045f77046d6` on July17 2026, before the August bringup. The family demo wires it in `models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py`, with tested contracts in `models/demos/blackhole/qwen36/tests/test_gdn_tp.py`. The earlier analysis searched affine-scan/KDA operators without completing the broader delta-rule/family-demo discovery. This was a substantive investigation miss.

The current `.agents/skills/graph-fusing/SKILL.md` already says “Always explore the tt-metal repo existing ops as potential fusing candidates” and ranks dedicated fused ops first. More generic instruction text would duplicate an existing requirement. A useful evidence gate would require an explicit inventory of matching native operators and family-demo integration paths, plus measured adoption/rejection, before declaring the graph optimized or projecting a source-op latency floor.

Further graph integration can eliminate work that this deliberately narrow adapter retains: preserve log-decay before exp/BF16/log; avoid query/key head repetition, padding, and head-major/token-major round trips by feeding the op's flat QKV path; use its in-kernel Q/K L2 normalization; replace composite convolution with the native fused convolution; and retest packed projections after the recurrence bottleneck has been removed. Each needs independent accuracy and latency checks. A complete native layer trace could remove the residual launch overhead; per-recurrence tracing is no longer the main opportunity. S128 full-model target60ms still requires an average budget below1ms across64 layers, so a5.5ms linear-layer result is substantial progress but is not enough.

## Flat QKV, direct log decay, and fused convolution

A second isolated wrapper, `flat_followup.py`, moves the native-op boundary earlier. The op accepts rank3 flat `[B,T,H*D]` Q/K/V and performs Q/K normalization and value-head mapping in its prep kernel. This removes explicit Q/K head splitting, repeated heads, L2 normalization, and the padding/permute round trips. It also retains the computed log decay directly instead of exp → BF16 → log. Beta and log decay are FP32 before multiplication by the generator's scalar sequence mask; padded positions have beta0 and log decay0, preserving recurrent state.

The native flat path requires `QWEN_GDN_PHASED=1` (also the current native default) and time length divisible by32. The wrapper falls back before modifying convolution state for a direct non-aligned input. The probe supplies S33 as a padded physical128 tensor, so that ragged test exercises the native path and masks. The generator itself does not add that padding: direct physical33 inputs exercise the rank4 native adapter and composite convolution. Logical and physical lengths must therefore both be recorded.

| Candidate | Sequence / outer chunk | Linear median | Full-attention control median | Evidence |
|---|---|---:|---:|---|
| Flat QKV + direct log decay | 128 / 128 | **4.076 ms** | 2.214 ms | `native_flat_s128_uncontended.json` |
| Flat QKV + direct log decay | 4096 / 512 | **39.696 ms** | 16.915 ms | `native_flat_s4096.json` |
| Flat QKV + direct log decay, allocation32, active slot17 | 33 / 128 | 5.748 ms | 2.219 ms | `native_flat_b32_s33.json` |
| Flat + fused causal convolution | 128 / 128 | **3.402 ms** | 2.192 ms | `native_flat_fusedconv_s128.json` |

These are three-sample warmed medians. The first flat S128 attempt overlapped an eight-thread CPU HF reference run: unchanged full-attention latency increased to6.600ms. Its artifact `native_flat_s128.json` is explicitly retained as contended, and its9.863ms linear time is excluded from optimization comparisons. HF reference completed before the other listed measured windows; the unchanged full-attention controls match the original uncontended times.

Flat S128 is25.6% faster than the narrow native adapter; flat S4096 is42.7% faster. Relative to the original graph, flat S4096 improves linear latency approximately42×. This remains a per-layer claim.

Every listed result had finite checked tensors, matching terminal logit top1, and exact convolution state. Flat S128 layer0 PCC vs original is0.999933. Flat S4096 layer0 PCC is0.999878 and terminal logit PCC0.999648. The S4096 recurrent state differs from the original with PCC0.96178, relative L2 27.47%; the previous identical-input FP64 oracle cannot be reused as an exact reference here because normalization and decay preprocessing changed. Full-model HF accuracy remains necessary.

For the ragged allocation32 test, layer0 PCC is0.9999845, recurrent-state PCC0.9999507, and convolution state is exact. Only slot17 was active. This is one serving-style mask/slot smoke, not complete mixed-batch or decode-interleaving coverage.

`FLAT_FUSED_CONV=1` additionally replaces the composite four-tap convolution + SiLU + Q/K/V split with existing `ttnn.experimental.kda.qkv_causal_conv1d_silu`. It uses already-sharded `weights['conv_tap0'..'conv_tap3']`, channel chunk256, row-major BF16 current tokens and three-token history. The generator's selector-based next convolution-state update remains intact. The native outputs feed the flat GDN op directly, without concatenating them back. The16.5% S128 gain is measured; long-sequence and ragged fused-convolution checks are still outstanding at this handoff. Direct batch>1 and non-aligned calls retain the original path before state mutation.

Reproduce by replacing `scan_followup.py` with `flat_followup.py` in the earlier commands. Use outer128/S128 or outer512/S4096. Ragged flags are `--sequence 33 --batch 32 --active-slot 17 --reference /tmp/qwen_b32_s33.pt`. Add `FLAT_FUSED_CONV=1` only for the fused-convolution candidate. No C++ changes were required. All device jobs closed successfully without recovery; hardware was released to the parent for full-model integration and validation.
