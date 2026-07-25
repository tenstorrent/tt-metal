# AutoFix: missing multichip topology families

## Starting evidence

- Source: `stage_review.md`, P1 finding for the unmeasured attended-input
  all-gather/local-output O decomposition, the EP-down-to-next-QKV residual
  boundary, and the missing BF16/BFP8 payload cross on a persistent
  lower-movement family.
- Selected comparison point: final default `20.583611 ms` warmed prefill and
  `0.503275 ms` traced decode.
- The prior residual probe stopped at the same layer's post-attention norm,
  router, and whole-expert input gather.  It did not carry an EP down partial
  into the next layer's input norm and packed QKV.

## Source and shape derivation

### Attended all-gather plus local-output O

The inherited O decomposition is
`A_rank[M,1024] @ W_rank[1024,2880]`, followed by an all-reduce.  The exact
alternative is:

1. All-gather `A_rank[M,1024]` over TP axis 1 to `A[M,4096]`.
2. Give rank `r` output columns of `W_o.T`, either natural
   `W_r[4096,720]` or padded `W_r[4096,736]` from global H=2944.
3. Compute a rank-local output and either gather it at the current residual
   boundary (the production A/B added here) or retain it as the residual
   fracture (the stack family below).

`ttnn.experimental.all_gather_minimal_matmul_async` is the exact fused API.
Its validation requires TILE device tensors, matched activation and weight
dtype, `K_input_local * ring_size == K_weight`, and, for Ring, a K block that
divides the local physical K tiles.  The first legal O config is
`M_block=1, K_block=8, N_block=2, subblock=1x2, grid=4x4`, because local
K=1024 is 32 tiles.  `force_transpose=True`, four workers, and one link satisfy
the program factory's sender-axis grouping constraint.  A stable persistent
gather buffer has shape `[...,M,4096]` and payload dtype BF16 or BFP8.

The implemented off-by-default controls are:

- `use_fused_o_projection_ag` / `MULTICHIP_FUSED_O_AG=1`;
- `fused_o_ag_pad_hidden` / `MULTICHIP_FUSED_O_AG_PAD_HIDDEN={0,1}`;
- `fused_ag_matmul_payload_dtype` /
  `MULTICHIP_FUSED_AG_PAYLOAD_DTYPE={bfloat16,bfloat8_b}`.

The decoder candidate uses fused attended AG+local O, then gathers at the
residual boundary so existing real-layer PCC and full decoder timing tests can
measure it without changing the selected default.

### EP down fracture through the next packed QKV

A rank owns eight whole experts.  Therefore simply column-sharding its down
weight would lose contributions from active experts on the other ranks.  The
legal algebra is:

1. Each rank computes its active local experts' full-H partial `[M,2880]`.
2. Pad H to 2944 and reduce-scatter the sum to one `[M,736]` shard/rank.
3. Add the corresponding padded residual shard and keep it fractured.
4. Run distributed RMSNorm: local pre-statistics, a tiny statistics
   all-gather, then post-normalization with the local norm-weight shard.
5. Fused-all-gather the normalized `[M,736]` input and multiply by the modeled
   downstream layer's packed-QKV weight `[2944,1280]` on each rank.

The padding is not numerically free.  `rms_norm_pre_all_gather` averages over
23 physical tiles/rank, so the gathered mean uses denominator 2944 rather than
logical H=2880.  The exact probe multiplies gathered statistics by
`2944/2880 = 46/45` before post-normalization.  Padded residual and QKV weight
rows are zero.

For fused packed QKV, local K=736 is 23 tiles.  Ring therefore uses the legal
`K_block=23` with `M_block=1, N_block=2, subblock=1x2, grid=4x4`.  The fused
API's matched-dtype rule means the BFP8 payload candidate necessarily uses a
BFP8 packed-QKV weight; BF16 bias and BF16 output remain legal.  This is an
exact API constraint, not an untracked precision change.

The retained exact test is
`test_ep_fractured_residual_to_fused_qkv_candidate`.  It uses real active EP
experts and real layer-12 norm/QKV weights as the modeled downstream consumer
(the retained fixture has no layer 13); persistent buffers cover
RS intermediate/output, statistics AG, and QKV AG.  It checks every rank's
residual, normalized shard, and packed-QKV PCC and records traced whole-boundary
latency.  Run it once per payload:

```bash
RUN_MULTICHIP_FRACTURED_BOUNDARY=1 \
MULTICHIP_FRACTURED_PAYLOAD_DTYPE=bfloat16 \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_ep_fractured_residual_to_fused_qkv_candidate

RUN_MULTICHIP_FRACTURED_BOUNDARY=1 \
MULTICHIP_FRACTURED_PAYLOAD_DTYPE=bfloat8_b \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_ep_fractured_residual_to_fused_qkv_candidate
```

## Hypothesis experiments

### H1: fused attended AG plus local O is API-feasible

- Prediction: exact TP4 shapes compile with a persistent gather buffer for
  both payloads; the natural and padded local-output candidates reproduce the
  current full residual after their boundary gather.
- Experiment: existing real-layer PCC and warmed prefill/traced-decode tests
  with the three environment controls above.  Cross natural/padded and
  BF16/BFP8; compare against the final default.
- First result: padded H=2944 compiled, but real-layer prefill failed
  (`0.982346` sliding, `0.987415` full).  The isolated real-O diagnostic
  localized this to receiver rank 3: ranks 0-2 were canonical at `0.999997`,
  while rank 3 was `0.982617`; none of 24 input or output rank permutations
  improved it.  This refuted a general matmul-order or fidelity explanation
  and identified the padded 736-wide trailing-AG handoff as the failing
  contract.  Artifacts: `fused_o_ag_bf16_pad64_pcc_ep.junit.xml` and
  `fused_o_ag_isolated_bf16_pad64.junit.xml`.
- Adaptation: retry the smallest legal natural local width, 720, retaining the
  same persistent fused AG, exact real O weight/bias, HiFi4 FP32-DST compute,
  and residual-boundary gather.
- BF16 natural result: isolated real-O PCC was `0.999997199` on every receiver
  with canonical rank order.  Both real layer kinds passed; minimum retained
  layer PCC was above `0.9944`.  Warmed performance was `20.771860 ms`
  prefill and `0.642176 ms` traced decode versus final-default
  `20.583611 ms` and `0.503275 ms`.  It is correct but decode is 27.6% slower.
  Artifacts: `fused_o_ag_isolated_bf16_natural720.junit.xml`,
  `fused_o_ag_bf16_natural720_pcc_ep.junit.xml`, and
  `fused_o_ag_bf16_natural720_perf_seq128.json`.
- BFP8 natural result: the isolated O boundary passed and sliding attention
  passed (`0.993043` prefill; decode `0.994284`-`0.995625`), but full-attention
  prefill was `0.989274`, below the accepted 0.99 floor.  The fused API already
  used its best legal precision arrangement: matched BFP8 activation/weight,
  BF16 bias/output, and HiFi4 FP32-DST.  Artifact:
  `fused_o_ag_bfp8_natural720_pcc_ep.junit.xml`.
- Verdict: **verified and rejected**.  BF16 is correct but materially slower;
  BFP8 fails a meaningful layer-kind accuracy gate.  The selected inherited
  local-MM+all-reduce decomposition remains default.

### H2: the EP output fracture can reach the next packed-QKV consumer

- Prediction: persistent padded RS, corrected distributed RMSNorm, and fused
  AG+packed-QKV preserve per-rank PCC without restoring a replicated residual.
- Experiment: retained exact real-weight test above, BF16 then BFP8, including
  traced boundary A/B against the inherited all-reduce + full norm + local QKV.
- First data error: the fixture contains only layer 12, so selecting layer 13
  as the literal next weights raised `_require_tensor` `KeyError` before any
  CCL.  The retry explicitly reused real layer-12 input-norm/QKV weights as the
  geometrically identical modeled downstream consumer and records
  `fixture_reuses_available_layer_weights=true`.
- Second data error: the candidate completed, but the comparison attempted to
  pad the baseline's 10-core/width-sharded normalized tensor and hit a tile
  shard assertion.  The retry retained the interleaved baseline norm output
  for PCC/padding and used a separate width-sharded handle only for baseline
  QKV.  The measured baseline still includes that conversion.
- BF16 result: pass on every rank.  Minimum PCC was `0.99999583` residual,
  `0.99999321` normalized, and `0.99999279` packed QKV.  Over 100 trace
  replays, replicated baseline was `0.452406 ms` and the fractured family was
  `0.539213 ms` (19.19% slower).  Artifact:
  `fractured_ep_to_qkv_bfloat16.json` and its retry2 JUnit.
- BFP8 result: pass on every rank.  Minimum PCC was `0.99989849` residual,
  `0.99987751` normalized, and `0.99985945` packed QKV.  Replicated baseline
  was `0.452890 ms`; fractured was `0.526221 ms` (16.19% slower).  Artifact:
  `fractured_ep_to_qkv_bfloat8_b.json` and its JUnit.
- Verdict: **verified and rejected**.  The residual can remain fractured
  through distributed norm and fused packed-QKV without any inter-layer
  gather, and BFP8 reduces the cost relative to BF16, but both coherent
  persistent-buffer families lose to the inherited replicated boundary.

## Static verification

```text
python -m py_compile .../tt/multichip_decoder.py .../tests/test_multichip_decoder.py
git diff --check -- .../tt/multichip_decoder.py .../tests/test_multichip_decoder.py
pytest -q ...::test_multichip_runtime_contract_and_fallback_audit \
          ...::test_multichip_perf_candidate_parsing
```

Result: `2 passed`; selected defaults remain fused-AG off and the provenance
parser exposes all candidate controls.

## Final status

**Fixed/closed with evidence.** Both missing topology families were implemented,
shape-adapted after their first failures, and measured on the real 1x4 mesh.
The natural BF16 fused AG/local-O decomposition is correct but slower; BFP8
misses full-attention PCC.  The stronger carried-residual family is correct in
both BF16 and BFP8 and avoids a gather between decoder layers, but traced
boundary latency is 16-19% worse.  No candidate is selected by default; the
off-by-default controls and exact regression probes remain as reproducible
evidence.
