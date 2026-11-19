# Layer34 actual-capture component counterfactuals

CPU-only follow-up to `AUTODEBUG_v5_layer34.md`. No TTNN import, device execution, runtime/source change, expert-weight loading or full-MoE replay. Loaded only layer34 gated gamma, attention output projection, postnorm gamma and router weight from the pinned checkpoint. Exact producer captures137/165/181 were used, not synthetic operands.

**At137, canonical postnorm plus CPU router on the actual TT attention output restores HF expert89 instead of59. At181, that substitution and every tested downstream canonicalization still select52 instead of HF184.** A postnorm repair is therefore a useful minimal discriminator for137, but is not a demonstrated solution to the shared failure family. An exact router alone does not repair either failing route.

## Controls and observed facts

Before any counterfactual, the script reproduced captured HF gated output, attention projection, postnorm and logits bitwise from the HF inputs using the four real parameter tensors. It also independently reproduced the capture diagnostic's three same-input CPU tensors: canonical gated norm on TT core/Z, postnorm on TT residual, and router on TT postnorm. All checks passed. Torch used8 CPU threads. Reported core comparisons below explicitly cast the FP32 TT core to BF16, as the actual norm boundary does.

Actual normalized input is bit-equal at all three positions. The earliest captured remaining producer discrepancy is native packed Q/Z: Q differs in260/275/283 of2048 values at137/165/181, with relativeL2 .000909/.001212/.001081; Z differs in571/515/636 of4096 values, relativeL2 .000963/.001138/.001136. This proves a producer difference, not that replacing Q/Z alone fixes output quality.

Actual route memberships:

-137: HF includes89, TT includes59; other7 experts agree.
-165: all8 actual experts agree.
-181: HF includes184, TT includes52; other7 experts agree. Ordering differences among identical membership are not counted as swaps.

CPU canonical routing on captured TT postnorm retains those same memberships. Native router arithmetic is therefore not necessary for these particular swaps. At137 the HF rank8/rank9 logits are .302734375/.294921875; actual TT makes89 and59 tie at .294921875. At181 HF52/184 logits already tie at -1.4296875, with HF topk8 choosing184. Actual TT gives52=-1.4296875 and184=-1.4375. Topk10 can order tied members differently from topk8; the report uses actual topk8 membership and preserves the top10 list only as a margin diagnostic.

## Counterfactual results

Each CPU downstream path uses the actual BF16 rounding boundaries, including BF16 residual addition. These are substitutions on captured tensors, not new TT outputs or proof of final MoE quality.

| Substitution |137 |165 |181 |
|---|---|---|---|
| CPU router on actual TT postnorm |59, wrong |114, matches |52, wrong |
| Canonical postnorm+router on actual TT attention |89, restored |114, matches |52, wrong |
| CPU output projection+postnorm+router on actual TT gated core |89, restored |114, matches |52, wrong |
| Canonical gated norm+all CPU downstream on TT core/Z |89, restored |114, matches |52, wrong |
| Same, TT core and HF Z |89, restored |197, wrong tie choice |52, wrong |
| Same, HF core and TT Z |89, restored |114, matches |52, wrong |
| Full HF core/Z and canonical downstream, bitwise control |89, matches |114, matches |184, matches |

At137 canonical postnorm alone moves89 above59 by .001953125, despite remaining upstream attention differences. Canonical gated norm on TT core/Z plus CPU downstream increases that margin to .0078125, matching HF's margin. At181 neither repairing core alone nor Z alone is sufficient in this sampled counterfactual; both exact HF inputs recover the captured control. At165 swapping only Z to HF while retaining TT core creates a different tied decision. Thus improved component PCC need not monotonically improve expert identity.

Canonical gated norm improves gated-core PCC on TT inputs at all three captures (.99998996→.99999274 at137; .99999131→.99999564 at165; .99999147→.99999571 at181), but those improvements alone do not establish acceptable decoder quality.

## Exact canonical boundaries and bounded recommendation

Installed HF `Qwen3_5MoeRMSNormGated.forward` computes variance and inverse inFP32; rounds unweighted normalized core to the original BF16 dtype; multiplies by the actual BF16 learned gamma, producing another BF16 intermediate; multiplies that by FP32 SiLU(Z); and returns BF16. Gamma has no +1 in this norm. The output projection is BF16 `F.linear` on the flattened4096-vector. Residual is BF16 `token + attention`. Ordinary postnorm widens toFP32, computes mean-square/inverse, multiplies by `1 + checkpoint_gamma.float()`, then casts toBF16. Router computes BF16 linear logits, FP32 softmax, topk8, selected-score renormalization and BF16 scores.

Minimal next discriminator: implement/test canonical postnorm arithmetic on actual TT residual with the current router and capture137/165/181. The CPU result predicts route recovery at137, not181. To address181, separately test restoring precise packed Q/Z projections and canonical gated-norm boundaries while leaving precise KVAB/state arithmetic unchanged. Q/Z is the earliest measured producer difference on exact inputnorm, and the isolated core/Z substitutions show that treating only one branch as exact is insufficient for this sample. This is a hypothesis for a bounded device test, not a claim that Q/Z alone explains all296 failures. No threshold waiver or runtime host fallback is proposed.

No expert matmul/combination counterfactual was run, so recovered IDs cannot be called a recovered final decoder output. The findings establish local route sensitivity and distinguish postnorm-local versus remaining upstream contributions.

## Reproduction and hashes

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_layer34_component_counterfactual.py
```

- CPU script SHA256: `8665469731485bc2ddfe0afc85e6d670d34dc1a45b9d575f099bbcd2f61b34a1`
- CPU report SHA256: `45a525a435d4d4009f201b93ac017a9d0f9a62939c93004178eb33a8ce7d862a`
- Capture137 SHA256: `e627062654d4343b420dc2bd489dc55df7dd35424b33ea7fa9faae361d616990`
- Capture165 SHA256: `2d38d8bdd35ef570a8a33449a7c818ea8bd12cee76dd47ea474c706d0ccc373d`
- Capture181 SHA256: `1c4c7d88f46344482949b0d6162666cd2bad1853c8097bfeb7da047b743f12ce`

The JSON contains every route, score, top10 logit, component comparison, source fingerprint and capture hash. It clearly labels captured native IDs separately from CPU routes.

## Minimal gated-norm guidance after counterfactual review

The available four-way core/Z substitution is already the useful cheap attribution: TTcore/HFZ, HFcore/TTZ and TTcore/TTZ all retain the wrong181 membership, while HFcore/HFZ reproduces the exact control. Therefore neither Z-only replacement nor core-only replacement is supported as a sufficient fix for181. At137 all three canonical gated-norm paths recover89 with CPU downstream, while postnorm alone also recovers89. This does not uniquely identify a gated-norm internal operation as the cause.

If testing a gated-norm repair, the smallest semantically complete candidate is restoration of the two HF BF16 boundaries (after unweighted normalization and after gamma multiplication), followed by FP32 SiLU(Z) multiplication and final BF16 output. Replacing only sigmoid with SiLU while retaining the native fused normalization/gamma chain does not restore that contract. Native intermediate variance/normalized/gamma-product tensors are not captured here, so subdividing those sources further would require a dedicated same-operand native capture/probe; a CPU emulation of native arithmetic would not be a new observation. No additional CPU approximation experiment or runtime patch is justified from these captures alone.
