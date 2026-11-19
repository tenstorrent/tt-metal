# Combined-source position181 CPU counterfactual

**Canonical gated norm on the actual TT core/Z, followed by CPU output projection, residual, postnorm and router, restores HF expert184.** Applying only CPU output projection and downstream arithmetic to the actual TT gated output still selects52. This distinguishes a consequential remaining gated-output discrepancy in the combined capture; it does not establish a passing device decoder or identify one primitive as its cause.

The input is [combined capture181](v5_combined_layer34_stream_probe.json.pos181.pt), executed from source `16fdf3f632f685af1b8e279f933fb28137308d8dc1f3965dafbb50c797401870`. This is the combined ordered Q/Z + ordered postnorm + BF16 gated-tail source, **before** the accurate first-residual-add control. The new [CPU script](v5_combined_layer34_counterfactual.py) and [JSON](v5_combined_layer34_counterfactual.json) are separate from the earlier V5 counterfactual artifacts. Eight CPU threads were used; only four small parameter groups were loaded (gated gamma,4096→2048 output projection, postnorm gamma,2048→256 router). No TTNN import, device execution, expert matrices or full model were used.

All nine checks pass bitwise: canonical HF gated output, attention projection, postnorm and logits reproduce the saved HF boundaries; three CPU same-input tensors reproduce the diagnostic's saved tensors; and actual QKV/Z equal HF. Actual TT BF16 core differs from HF in five values. Actual gated output differs from HF in319 values; canonical gating on that same TT core/Z leaves only four differences. The existing capture's same-input gated comparison reports315 differences, relative L2 .001120642. Thus the residual few core differences do not by themselves prevent recovering HF membership in this counterfactual.

| CPU path starting from captured TT tensors | Selected disputed expert | Logit184 − logit52 |
| --- | ---: | ---: |
| Router on actual TT logits |52 |−.0078125 |
| Canonical router linear on actual postnorm |52 |−.0078125 |
| CPU residual/postnorm/router on actual attention |52 |−.0078125 |
| CPU output projection and downstream on actual gated output |52 |−.0078125 |
| Canonical gate and all downstream on actual TT core/Z |184 |0 |
| Canonical gate and all downstream on HF core/actual TT Z |184 |0 |

The actual native IDs are `[39,94,196,143,11,134,203,52]`; HF and canonical-gate counterfactual topk8 choose `[39,94,196,143,11,134,203,184]`. HF itself ties52 and184 at BF16 logit−1.4296875. The script uses topk8 for membership: topk10 orders this tie differently and is retained only as a diagnostic list. Native top-k tie behavior can still differ from CPU, so recovery of CPU membership does not guarantee that a repaired device producer plus the existing native router will choose184.

After canonical gating of TT core/Z, downstream output projection, postnorm and logits are still not bit identical to HF (202/108/13 differing BF16 values), but the selected expert set matches. No expert-output/MoE replay was performed, so this cannot be called recovered final-output PCC. CPU postnorm here reconstructs CPU BF16 `token + attention`; the actual-device-residual capture remains necessary to attribute the postnorm-local discrepancy.

The bounded next check is primitive-level comparison of the combined gated tail against canonical gating on identical captured TT core/Z/gamma, while retaining the separate accurate-residual experiment. A remaining gated arithmetic difference is useful to investigate; exact intermediate bits are not an additional model acceptance gate.

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_combined_layer34_counterfactual.py
```

The script refuses to overwrite an existing result. Current evidence hashes:

- Script: `9d7f6e1bb7ef05b0b766dc4810661bc274e93d166ea82ad9595631b2646c6043`.
- JSON: `0b788f97ee0810f4e83af883e0a07ad458791e38c4ef394f3789da3bbd53acb9`.
- Capture181: `e23b352ae57d0627ef870f0363813e831c858443550d040a02526833711d286f`.
