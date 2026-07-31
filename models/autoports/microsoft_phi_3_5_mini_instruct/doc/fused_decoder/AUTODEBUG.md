# AutoDebug: Phi-3.5 fused decoder evidence gaps

Inspection only; no hardware was run.

## Headline findings

1. **The SiLU fusion's traced-decode win is not established.** The sole recorded
   A/B is functional-first, fused-second (`tests/fused_decoder_perf.py:100-134`)
   and accepts any lower fused mean (`:136-137`). The reported differences are
   only 0.001961 ms at B1 (0.19%) and 0.003208 ms at B32 (0.26%);
   functional/fused minimums differ by similarly tiny amounts
   (`perf_before_after.log:65,89`). There are no repetitions, confidence
   intervals, order reversal, or paired/interleaved samples. The implementation
   does remove the standalone `ttnn.silu`: functional uses
   `multiply(silu(gate), up)` (`tt/functional_decoder.py:218`), whereas fused
   passes `input_tensor_a_activations=[SILU]` to multiply
   (`tt/fused_decoder.py:33-37`). That proves graph fusion, not a robust
   end-to-end latency improvement.

2. **The README's “no tilize/untilize” assertion is false and the profiler data
   is not attributable to source contracts.** Every bounded CSV contains
   `TilizeWithValPadding`, `Untilize`, `UntilizeWithUnpadding`, and `Permute`:

   | bounded range | tilize-pad | untilize | untilize-unpad | permute |
   |---|---:|---:|---:|---:|
   | decode B1 | 24 | 8 | 16 | 24 |
   | decode B32 | 16 | 8 | 16 | 24 |
   | prefill B1 | 16 | 8 | 16 | 24 |
   | prefill B32 | 16 | 8 | 16 | 24 |

   This directly contradicts `README.md:76`. The likely source families are:
   row-major norm/RoPE weights uploaded at `functional_decoder.py:166-190`;
   explicit `to_layout(mask, TILE)` at `:253-266`; head split/concatenate at
   `:279-292,396-397,468-495`; and decode RoPE's shard-to-DRAM, 48-wide
   slice/concat, then DRAM-to-shard round trip at `:400-420` plus `_apply_rope`
   at `:220-236`. These are hypotheses: the current bounded CSVs have no Python
   call-stack/source label, so the report cannot prove which caller emitted
   each family.

## Smallest decisive experiments

### A. Robust traced-decode A/B at B1 and B32

Add a benchmark-only control selecting candidate order, retain each individual
sample (not just mean/min), warm both traces equally, and alternate replay in
paired blocks. Use at least 10 independent processes x 1000 replay pairs per
batch; report paired median, bootstrap 95% CI, and per-order result. Prediction:
if the fusion is real, the fused-minus-functional CI is below zero at both
batches in both orders. If it crosses zero or reverses with order, the current
sub-percent claim is noise/order bias.

Exact proposed commands after adding `FUSED_PROFILE_ORDER` and sample JSON:

```bash
mkdir -p /tmp/phi_fused_ab
for batch in 1 32; do
  for order in functional,fused fused,functional; do
    for run in $(seq 1 10); do
      FUSED_PROFILE_BATCH="$batch" FUSED_PROFILE_ORDER="$order" \
      FUSED_PROFILE_ITERATIONS=1000 FUSED_PROFILE_JSON="/tmp/phi_fused_ab/b${batch}_${order//,/-}_${run}.json" \
      pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_perf.py \
        -k profile_traced_decode_before_after
    done
  done
done
python models/autoports/microsoft_phi_3_5_mini_instruct/tests/analyze_fused_decoder_ab.py /tmp/phi_fused_ab
```

Also compare the traced op lists/hashes for the two variants. Predicted structural
outcome: fused has one fewer standalone SiLU dispatch and one multiply carrying
the unary activation; unrelated ops and layouts remain identical.

### B. Attribute each layout-op family before attempting removal

First collect one replay per case with Python stack traces enabled and unique
signposts around these source regions: QKV split, each Q/K RoPE invocation,
cache fill/update, SDPA, head concat, and MLP. Produce call-stack-bearing CSVs:

```bash
TTNN_CONFIG_OVERRIDES='{"enable_graph_report":true,"enable_graph_python_stack_traces":true}' \
FUSED_PROFILE_ONLY=1 FUSED_PROFILE_ITERATIONS=1 \
python -m tracy -r -p -v -m pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_perf.py
python tt_metal/tools/profiler/process_ops_logs.py
```

Then run one-source-change-at-a-time controls, preserving PCC/cache tests:

- **RoPE DRAM round trip / slice-concat:** replace only `_apply_rope` with a
  proven 96-wide Phi-compatible device primitive, if available. Prediction:
  the Q/K-associated untilize/unpadding/permute groups disappear; if the
  primitive rejects width 96 or changes Phi half-rotation PCC, document this
  family as required by the current op contract.
- **Head split/concat:** bracket
  `split_query_key_value_and_split_heads`,
  `nlp_create_qkv_heads_decode`, `concatenate_heads`, and
  `nlp_concat_heads_decode`. Prediction: remaining Permute groups map here; a
  replacement is acceptable only if output shape/layout and decode's
  one-core-per-user shard contract (`functional_decoder.py:422-441`) remain
  valid.
- **ROW_MAJOR/TILE boundaries:** bracket norm/RoPE table consumers and the
  explicit mask `to_layout`. Try TILE-uploaded weights/tables and eliminate
  only conversions whose consumer accepts the new layout. Prediction:
  tilize-pad/tilize groups disappear exactly at accepting consumers; rejection
  or changed program selection proves necessity for that consumer, not for the
  whole model.
- **Padding/unpadding:** record logical and padded shapes for each attributed
  op. B1's extra eight tilize-pad operations versus B32 strongly predicts a
  decode batch-height padding boundary. Test B1 with padded-32 storage plus a
  final logical slice; keep it only if total traced latency improves and PCC,
  cache rows, and public `[1,1,B,3072]` shape remain unchanged.

For every control, rerun:

```bash
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_fused_decoder.py
FUSED_PROFILE_ONLY=1 FUSED_PROFILE_ITERATIONS=3 python -m tracy -r -p -v -m pytest -q -s \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_perf.py
```

Acceptance requires a before/after table keyed by source region and op family,
with counts and device time at B1/B32 for prefill/decode. Until that exists,
remove the README's categorical claim and describe the observed conversions as
unattributed rather than “documented and necessary.”
