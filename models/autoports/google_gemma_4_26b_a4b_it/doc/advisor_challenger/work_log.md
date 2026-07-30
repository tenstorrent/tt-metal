# Advisor-challenger work log

## Runner-side verification bug report

The independent runner reported:

```text
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh: line 13: 1: model_dir
```

This reproduces when the checker is invoked without its required positional
`<model_dir>` argument. It fails before reading any stage artifact, so it is not
evidence of a decoder, measurement, capture, reconciliation, or oracle failure.

The required invocation was run from the tt-metal repository root:

```bash
bash .agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh \
  models/autoports/google_gemma_4_26b_a4b_it
```

It exited 0 and reported all four substantive sections `ok`:

- incumbent has at least three repeats and freezes the best repeat;
- both layer-kind captures parse, contain matmuls, match shipped dtypes, and
  classify DRAM-sharded consideration;
- every reconciliation disagreement has a measurement;
- the final invariant, incumbent tie policy, real-weight oracle, and bounded
  iteration checks pass.

## Final measured decision

The frozen incumbent is 1.2698322534561157 ms with a
0.001062639057636261 ms noise floor. The coherent sliding-attention
residual/norm challenger measured 1.1695511639118195 ms, but its real-weight
decode PCC was 0.994795, below the shipped 0.995 bar, so it is rejected.
All correct advisor-derived material candidates were slower:

- sliding QKV DRAM sharding: 1.3431021943688393 ms;
- full QKV DRAM sharding: 1.3015633448958397 ms;
- sliding O-projection DRAM sharding: 1.2728888541460037 ms;
- full residual/norm sharding: 1.3626804575324059 ms.

The final policy is therefore the unchanged incumbent:
`final_ms == incumbent_ms == 1.2698322534561157`, with the real-weight oracle
passing at PCC 0.999616902015678 for sliding decode and 0.9997537381107462 for
full-attention decode.
