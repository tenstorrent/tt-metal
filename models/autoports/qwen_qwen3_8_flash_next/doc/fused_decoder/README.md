# Qwen3.8-Flash-Next fused decoder

**Accepted: independent review clean-pass.** Scope is `tt/fused_decoder.py`, stage tests and documentation. Functional runtime sources are unchanged. No optimized-decoder, multichip, full-model or serving stage has started. The stage is checkpointed locally; nothing is pushed. See manifest.json and the work log for the commit.

[FusedDecoder](../../tt/fused_decoder.py) preserves the public functional API and request-owned paged/recurrent state. Tests assert actual fused component classes and selected flags. Arbitrary positive logical tails remain supported through internal chunking. The inherited [context contract](../context_contract.json) advertises262144 autoregressive context and the prepared fully resident PLE public4768/4767 profile; full-context evidence uses a reduced attention/state prefix followed by a complete public final-token decoder, not full MoE on every prefix row.

## Current implementation and candidate

The selected final graph uses packed shared expert and GDN z/a/b projections; packed40-core routed gate/up; exact packed SiLU/multiply; early activation allocation; compact10-expert indexed decode; dedicated score-weighted expert reduction; accurate SFPU residual/router/GDN-output sums; direct checkpoint grouping with BF16 widening inside the unpacker; exact QKV product/half-tree fusion; exact GDN convolution/SiLU; and compact recurrence I/O. QSA uses packed projections, native main RoPE/head concatenation, gate folding, ordered valid-row indexer broadcasts, merged accurate score reductions, and device FP32 selection/group expansion.

The final default adds the measured `best_writer_tail2` policy: exact FP32 product → BF16 round-to-nearest-even → residual add → BF16 rounding in one device kernel, and compact expert union for two-row MoE calls. Its constructor defaults are exercised directly by `test_fused_decoder.py`, with component-class and policy assertions. Physical padding and standard32×32 tile guards protect the writer's indexing contract. Historical candidate policies pin their flags.

General union and thresholds3/4 were rejected because padded expert slots regressed real full-decoder inputs. No KV-cache dtype, persistent state layout, or advertised capacity changed.

## Measured performance

These are serialized chip0 candidate measurements, not stage acceptance. Host prefill is warmed; decode is warmed traced replay. Timing runs disable allocation tracking, while separate correctness/Watcher runs enable it and retain program-cache allocations.

Final default,32-token prompt; real checkpoint weights. Layer0/3 baselines are bracketed in the same final run. PLE uses the earlier serialized same-harness BFP8 baseline because the resident table precludes simultaneous models.

| Layer kind | Functional prefill ms | Final prefill ms | Functional traced decode ms | Final traced decode ms |
|---|---:|---:|---:|---:|
|GDN|423.757|78.400|46.771|5.019|
|PLE|463.053|95.835|47.680|6.341|
|QSA|970.490|184.512|249.688|9.548|

[Final performance](final_performance.json) binds each value to its measurement. Final decode improves over the prior best correct v5 by1.96% GDN,1.57% PLE and0.69% QSA. The current same-run writer candidate and default agree within timing noise. These small last-step gains end tuning under the owner wall cap; cumulative functional speedups are9.32×/7.52×/26.15×. Clean timing disables allocation tracking; separate correctness and Watcher runs enable it with program-cache allocations included.

Earlier cumulative exact-QKV integration reduced GDN decode8.136646→5.117432ms and prefill248.684662→78.750954ms. QSA cumulative v5 is about9.647ms decode versus249.687ms functional. The integrated writer measures QSA prefill183.859695ms/decode9.515672ms versus functional966.575684/249.601486ms; PLE prefill96.196144ms/decode6.342883ms versus the serialized functional control463.053467/47.679710ms. Final default reruns and all official reports passed in `wrap_final_plan_progress.json`.

Historical validation of the repaired native whole-trace control measures layer0 prefill424.122685ms functional versus78.502730ms v5, including internal gaps. Official tables/CSVs are under `tracy/{baseline,final}_layer0_prefill_final_v5b/report/`. [Final per-op findings](per_op_findings.md) summarize all six completed selected-path reports. No70%-roofline claim is made.

## Correctness and capacity evidence

The prior frozen v5 source passed synthetic8/8, short HF0/1/3, continuation0/1/3, context smokes, batch32GDN/QSA andPLEbatch5, prepared4768/4767, and Watcher0/1/3. [Matched PCC table](final_v5_pcc_comparison.json) records functional/fused minima; all original rows meet the unchanged0.995 bar and inherited row-specific controls. Its gate plan later failed a profiler harness assumption, which was independently diagnosed and repaired; the failed plan remains failed. See [profiler AutoFix](autofix_trace_protocol/AUTOFIX.md).

The actual new residual-writer component is bitwise exact across11 lengths1..65, random and BF16 midpoint/cancellation inputs, and12 alternating-input replays under Watcher. [Component timing](residual_write_clean_v1.json) is0.076272→0.022769ms at1row and0.125030→0.023365ms at32rows. Actual compact routing is exact across11 lengths for BF16/BFP8 experts; repeated identical-route rows also pass Watcher, exercising zero-score duplicate padding.

The combined candidate passed expanded GDN HF, original33-token PLE HF, GDNbatch32, PLEbatch5, preparedPLE4768/4767 and4738/2-tail cases, and Watcher GDN/PLE. The final default passed expanded synthetic lengths1/2/31/33/34/65/66, QSA Watcher and batch32, final candidate/default ranking, and official per-op reports. All three final32K context smokes passed; [gate summary](final_gate_summary.json) records scopes and metrics. See [final queue](wrap_final_plan_progress.json).

The stage already ran one262144 GDN context bridge:262143 actual prefix rows plus repeated complete final decode (`packed_early_context_v1_gdn_context.json`). Per owner wrap-up instruction, this is retained and not repeated; final32K smokes cover the current source. Context smoke scope is reduced attention/state prefix plus complete final token where the harness supports it, not all-prefix full-MoE execution.

## Expanded-fixture accuracy and controlled limitations

The owner clarified the final acceptance bar: whole-output/per-layer-kind aggregate PCC≥0.995; individual rows are diagnostics with floor0.99. The saved tensors pass that bar: PLE34 whole0.99820047, QSA34 whole0.99979485, QSA66 whole0.99980678. [Independent CPU recomputation](wrap_accuracy.json) retains the original failed stricter per-row gate records. Every row is≥0.99, and the three outliers below0.995 preserve hidden-coordinate argmax. Token top-1 cannot be measured at an isolated decoder without vocabulary logits; no token-level agreement is claimed.

| Fixture | Fused whole/contribution PCC | Matched evidence |
|---|---|---|
|PLE34, row5|0.994464/0.993644|Functional fails the same row. All old/new fused full outputs are byte-identical. HF using only the existing actual packed lookup reproduces0.994646/0.993853; fused versus packed-HF all-row minima0.999462/0.999426.|
|QSA34, row30|0.993680/0.993508|Functional fails the same row. All old/new fused full outputs are byte-identical. Changing only HF attention-GR-down accumulation toFP64 reproduces the literal-HF outlier; fused/control all-row minima0.998606/0.998568.|
|QSA66, row44|0.993133/0.992766|Same matched controls; fused/attention-GR-down-control all-row minima0.997069/0.996967.|

[PLE AutoFix](autofix_ple_tail/AUTOFIX.md), [PLE independent diagnosis](autofix_ple_tail/AUTODEBUG.md), [QSA AutoFix](autofix_qsa_tail/AUTOFIX.md), and their raw control artifacts distinguish shared backend sensitivity from new fusion behavior. The controls are diagnostic, not replacement acceptance or unchanged-HF strict passes. These per-row limitations are recorded with exact controls in [eval readiness limitations](../eval_readiness/limitations.json). Literal HF is not claimed to fail self-comparison; the controls isolate the existing packed lookup and GR accumulation sensitivity.

## Artifact map and remaining work

- [Work log](work_log.md): exact plan/command references, decisions, failures, repairs and source identities.
- [Graph inventory](graph_inventory.md) and [all-pattern assessment](pattern_assessment.json): every graph-fusing pattern and additional model-specific opportunities.
- [Runtime boundaries](runtime_boundary_audit.md): setup-only host work and necessary representation conversions.
- [Roofline accounting](roofline_accounting.md): active traffic versus resident capacity; no invented70% utilization.
- [Closure manifest](manifest.json): selected source, logs, gate evidence and report paths. Raw large profiler data remains locally available; no new per-artifact archives are required by the owner wrap-up policy.

The owner wall cap ended tuning. Untried ideas are recorded under `Not tried (stop rule)` in the work log, with no70%-roofline claim. The final queue completed21/21 jobs. [Independent review](stage_review_round1.md) returned clean-pass with no Required Work. The work log records the local checkpoint. Nothing is pushed.
