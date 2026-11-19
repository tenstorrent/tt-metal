# AutoFix: isolated B32 input-normalization discriminator

Status: updated host validation passed after the parent's first acceptance run exposed a known input transport mapping. **No TTNN import, device execution, device compilation, timing, server, or production edit was performed by this investigator**. The parent owns hardware execution. This note is the scoped AutoFix handoff, in place of a competing shared `AUTOFIX.md` write.

## Follow-up: exact transport mapping at the packed input boundary

The parent's first run is preserved at `readiness_vllm/p4_batch_norm_01/acceptance.json` and `doc/optimized_vllm/p4_floor_repair/norm_acceptance.log`. Both recorded cases passed exact intermediate/output checks. The finite-edge case completed all physical final-output comparisons before failing the first intermediate check: `finite_edges/wide/rank0: 2175 unequal words; first=[0,1,0]`. Later finite-edge intermediates and traces were not reached, so they remain unverified.

Native `concat.cpp:96` (`build_untilize_rm_retilize_concat`) selects untilize/unpadding followed by tilize-with-padding when the concatenated dimension is padded. The singleton input rows and intermediate16-row groups satisfy that condition. Existing independent `readiness_vllm/optimized_hidden_row_transport/result_maps.pt` exhaustively measures the prior slice/tilize transport on all65536 BF16 patterns per rank; `tests/run_hidden_row_mask_probe.py:old_split_canonical_bits` documents its mapping. For finite words, exponent-zero patterns map to positive zero and normal finite patterns remain bit-identical. The map is idempotent across both concat levels.

The revised probe's host check independently loads that measured mapping and verifies the finite rule against all65280 finite patterns on each of four ranks, with exactly255 changed patterns per rank (254 signed subnormal patterns and negative zero). It then predicts exactly2175 changed words in the current edge fixture:2048 negative-zero words in row1 and127 positive subnormal words in row2. Both recorded fixtures predict zero changed words. This agrees with the observed failure count and first coordinate; individual current-run transitions must still be proven by the next device run.

The revised eager and traced checks require **exact full-tensor equality** between candidate `wide` and this independently validated finite transport mapping of the actual singleton `wide` readback. They record changed-word counts per rank. No other stage is canonicalized: square, grouped square, groups, total, inverse, normalized FP32 and pre-split BF16 output still compare directly to the raw singleton intermediate bits. All final physical BF16 output comparisons remain unchanged. This tests the known boundary explicitly rather than skipping it or introducing a numeric tolerance. A second input mapping or downstream difference still fails.

Use fresh `p4_batch_norm_02` output paths for the next acceptance/benchmark pair; preserve `p4_batch_norm_01` as the failed diagnostic. The exact commands below are updated accordingly.

## Contract and candidate

Finding 2 in the updated `AUTODEBUG.md` identifies 32 serialized compensated input norms in each linear layer. The lowered native reduction uses dim1 then dim0 and sums matching tile elements; B1 and B32 have identical padded `Ht=Wt=1`. Batching is plausible, but packing, permutation, helper output, broadcast and transport bits remain unverified.

Only new `tests/run_p4_batch_norm_probe.py` contains the candidate. It concatenates the existing 32 singleton DRAM BF16 residuals with production `_bounded_concat(fan_in=16)`, groups squares as `[1,32,64,32] → permute(0,2,1,3) → [4,16,32,32]`, calls the unchanged `fast_reduce_nc(dims=[0,1])` defaults, `NormTerminalScan` and `CompensatedNormEpilogue`, then the separate FP32 normalization and BF16 output multiply. Gamma is the actual checkpoint input-layernorm weight plus one in FP32, matching production upload. Output split uses the established row-major slice/tilize transport boundary. No arithmetic fusion, reduction reassociation, epsilon, precision policy or context change.

The oracle is 32 actual `normalized_singleton` calls. A separate diagnostic copy exposes wide input, square, grouped square, reduced groups, terminal sum, refined inverse, normalized FP32 and BF16 output; its physical output must first match the production singleton. Candidate logical arithmetic intermediate bits then match the concatenated diagnostics exactly; the wide input separately matches the measured finite transport rule above. All final physical BF16 words (including padding) match production on every rank. Any difference beyond that single known input mapping fails, even if its logical error would be numerically small.

## Fixtures and coverage

The default existing artifact `doc/multichip_decoder/natural_b32.pt` contains real HF layer-input activations for layers 0 and 3. Layer 0 is linear; layer 3 is full attention and rejected by the probe. The probe selects 64 distinct recorded values across `xd` then `x`, records every source coordinate, and makes two B32 cases without changing their values. Selection across positions is necessary because repeated token IDs have identical layer-0 embeddings; these are not presented as 32 simultaneous serving rows. A third case adds finite zero, signed-zero, subnormal, alternating-sign, scale and BF16-boundary rows. No synthetic values replace the real cases.

Coverage includes changed-input eager program-cache reuse, original production outputs, exact intermediate comparisons, two retained reference/candidate traces, A→B→B→edges→A replay, cache-miss prohibition, exact stable input refresh before each variant, and a deliberate live-allocation positive control in tracked acceptance. Only exact captured output/stage tensors are acknowledged, with each read completed to CPU before another trace. No broad tracker exemptions. The selected context remains recorded as 262144; this component allocates no KV cache or model context.

The benchmark is a separate tracking-disabled process, with all correctness controls still run. Five alternating-order samples of 20 nonblocking replays are bounded by device synchronizations before/after timing. Candidate timing includes pack/split; input host refresh and readback are outside timing. No profiler is enabled. It reports batch32 normalization elapsed time and speedup only, never full-model latency or P4-floor acceptance.

## Host checks actually run

From the repository root:

```bash
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_35b_a3b/tests/run_p4_batch_norm_probe.py
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_p4_batch_norm_probe --mode cpu
```

Both passed. CPU mode loaded real checkpoint gamma and the existing activation artifact, validated all 65,536 logical element indices in the row mapping, proved the omitted-permute negative control differs, and asserted TTNN was absent from `sys.modules`. Python syntax compilation does not compile device kernels. Black formatted the new file. Ruff was unavailable (`No module named ruff`); no lint pass is claimed.

## Exact bounded owner commands

Run serially with no live server or other hardware owner, after the standard environment setup. These commands preserve old results by requiring fresh `--output` paths. A tracked acceptance pass is required before considering benchmark evidence.

```bash
source python_env/bin/activate
source ../lane-env.sh
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_METAL_DEVICE_PROFILER=0 \
  timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_p4_batch_norm_probe \
  --mode acceptance \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/p4_batch_norm_02/acceptance.json
```

Only after acceptance succeeds and the mesh is closed:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_METAL_DEVICE_PROFILER=0 \
  timeout 900 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_p4_batch_norm_probe \
  --mode benchmark --samples 5 --repeats 20 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/p4_batch_norm_02/benchmark.json
```

The probe rejects Watcher/profiler-enabled runs. Capture reserves 128 MiB per device, configurable with `--trace-region-size`. A failure or timeout is not a passing discriminator and should be preserved before any hardware recovery.

## Uncertainty before the corrected acceptance run

The initial parent's run establishes eager exactness for the two recorded cases and physical final-output exactness for the finite-edge case, with the input-mapping failure described above. At that point acceptance after the harness correction, subsequent edge arithmetic intermediates, trace allocation tracking/replay and elapsed time were unverified; the later corrected acceptance and benchmark below now cover them. Only layer-0 gamma is covered by the default recorded-real-input probe; `--layer`, `--activations` and `--snapshot` permit additional compatible artifacts, and unsupported/full-attention layers fail. Intermediate unused padding is unspecified; final output physical padding is strictly checked. A passing component probe is only grounds to consider a bounded production patch, followed by all30 linear-layer/whole-model trace and accuracy validation and the unchanged P4 useful-work floor. It is not proof that batching alone removes the approximately 214 ms required.

## Bounded production candidate for owner validation

The parent's corrected component acceptance and separate benchmark both passed, preserved under `readiness_vllm/p4_batch_norm_02/`. Median synchronized elapsed time was1.3598291ms for32 singleton norms and0.4645922ms for the batched norm including pack/split, a2.92693x component ratio. These are isolated component measurements; they do not establish model savings, serving correctness or P4 acceptance.

At the parent's explicit request, a minimal production candidate is now present in two files:

- `tt/mesh_compensated_norm.py:245`: new `normalized_batch32_input` reproduces the probe's batched arithmetic with the existing gamma, terminal scan and compensated epilogue. The existing `normalized_singleton` remains byte/AST unchanged.
- `tt/multichip_decoder.py:35` imports the new helper. At the linear-loop branch aroundline1973, the candidate runs only when attention reductions are already batched, the layer is linear, B=32, residuals are replicated, compensated normalization is enabled, and every input is DRAM BF16 TILE `[1,1,2048]`. It packs with the same fan-in16 concat, normalizes once, and creates32 singleton outputs using the exact row-major slice/tilize boundary. Packing temporaries are released before attention, as in the measured probe. Existing per-row attention calls and activity/state arguments are unchanged. Other geometry, memory configurations, dtypes, layer kinds, batch counts and compensated-disabled paths retain existing normalization.

No environment lookups, new configuration switches, precision changes, context changes, commits or adoption claim were added. The parent may test this together with the independent combine candidate and revert either if the required floor remains unmet.

Host verification after editing: Python compilation and `git diff --check` pass. A standard-library AST/symbolic execution check verified the production helper's exact14-operation sequence and every argument against the accepted test candidate (without importing TTNN); the pre-existing singleton function and all `_linear_attention` call ASTs match HEAD. A guard matrix covered80 combinations of layer kind, batch, residual layout, compensated flag and DRAM/L1 memory, selecting only the intended branch. No device execution was performed by this investigator.

Remaining production risks: different gamma values across all30 linear layers, integration into persistent full-model traces, native concat/split allocation lifetimes amid other layers, and the cost of retaining32 normalized singleton outputs until per-row attention consumes them. B32 inactive slots retain the same attention/activity handling; full-model mixed-shape/state validation is still required. The component trace test does not cover whole-model trace address reuse or server scheduler state. A shortened first1000-token diagnostic can reject the candidate quickly but cannot replace the canonical6000-token P4 result or justify promotion.

## Owner disposition

The temporary production candidate was measured through mixed serving and a32-user1000-token diagnostic, then reverted after the floor remained failed. See `serving_disposition.json` and `AUTOFIX.md`. The active production files no longer contain this helper/branch; `combined_candidate.patch` preserves it. The component probe remains self-contained. No final acceptance is claimed for the prototype.
