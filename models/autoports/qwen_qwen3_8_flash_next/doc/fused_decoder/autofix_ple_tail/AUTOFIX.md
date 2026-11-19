# PLE34 expanded-fixture accuracy investigation

Status: independent AutoDebug finished; its requested matched packed-only CPU counterfactual has now completed. No threshold, fixture, functional source, precision or runtime change has been made to suppress this failure.

## Failing contract and hypotheses

New layer1 public34 probe has literal whole row5 PCC0.9944641348820654 and contribution0.9936444270418033, below0.995. Three decode steps, reset state/output and repeated prefill pass. Candidate combines exact residual write-back and compact2-row tails. Hypotheses: writer arithmetic, tail routing, prior fused regression, or inherited packed-backend sensitivity.

## Discriminating controls

`ple34_control_collection_v1.json` records five independent serialized controls, each preserving its failing exit1 and raw metrics. All use bit-identical inputs and literal HF oracle. Functional baseline row5 whole0.9944669133989147/contribution0.9936475639549838. Prior v5, writer-only, tail-only and combined all have the same literal minima as the initial failure. `ple34_control_comparison_v1.json` proves all four fused variants have bit-identical full prefill and all three decode outputs. Baseline versus v5 full-prefill minimum row PCC0.9995484810051845. Thus neither new rewrite introduces this failure.

`ple34_packed_backend_v1.json` runs the existing independent CPU counterfactual on the exact exported fixture. Literal HF rerun is bit-exact. Changing only HF embedding lookups to actual cache-decoded BF16 values reproduces row5 whole0.9946457606798581/contribution0.993853134747022. Twelve cache parity checks pass. The selected router set changes (literal includes88, packed includes325). FP64 controls for the six sensitive projections do not repair the packed mismatch (whole0.9946348156447864). Fused versus packed-HF minimum over all34rows is whole0.999462011875479/contribution0.9994260598093204.

## Acceptance status

The literal34 gate remains failed; this diagnostic does not waive it or claim an unchanged-HF strict pass. Original33-token PLE HF acceptance passes in `integrated_writer_layer1_original_hf_v1.json`. Other original-contract and new-tail/capacity gates continue separately in `integrated_writer_remaining_plan_v1`. The inherited packed-table policy, literal failure, matched functional control and matched packed-HF control must all be presented to stage review. No host fallback, fixture-specific correction, or capability restriction is authorized by these observations.

Exact commands, source hashes, logs and cache paths are in each named provenance/plan. Full output exports reside under `/home/ttuser/qwen38-flash/ple34_control_<policy>_v1.pt`; CPU intermediate exports are `/home/ttuser/qwen38-flash/ple34_packed_backend_v1.pt`.

Independent report `AUTODEBUG.md` confirms complete byte equality of old/new fused outputs and identifies packed lookup as the earliest intentional divergence. Its proposed CPU check is now fulfilled by `ple34_packed_backend_v1.json`; the matched packed-HF result above supports that cause. The report also notes missing standard-tile/padded-shape guards on the new writer; add these before the final source freeze. No observed input violates them.
