# GPQA loop conditional-logit discriminator

Both full60 HF and TT support the observed loop continuation on the same captured prefixes. Top1 matches at all six checkpoints; KL(HF||TT) ranges0.000157897–0.007201093. This answers the selected conditional-distribution question without establishing why either trajectory entered a loop. No runtime change accepted; causal entry remains open.

| Actual document | Prefix length | HF / TT raw probability of observed next loop token | KL / JS |
|---|---:|---:|---:|
|7|4585|0.556811 / 0.576067|0.00720109 / 0.00143817|
|7|4601|0.999878 / 0.999772|0.00015790 / 0.00003931|
|7|4602|0.994449 / 0.994998|0.00491008 / 0.00068087|
|20|3437|0.998122 / 0.999432|0.00123339 / 0.00022340|
|20|3453|0.998172 / 0.999562|0.00175766 / 0.00030394|
|20|3454|0.959808 / 0.975664|0.00579915 / 0.00121181|

Both observed interior loop phases have probability1 after the diagnostic CPU top20/top_p.95/temperature1 transform. Raw probabilities differ. This transform is not a verification of device sampler arithmetic or RNG. Full top20 details remain private; compact candidate aliases, ranks, probabilities and overlap counts are in RESULT.json.

Exact native identity correction: request0/doc7 has481prompt tokens and native periodic onset4104; request1/doc20 has258prompt tokens/onset3179. Earlier metadata swapped the prompt-length attribution. Both cases use complete native first tokens and exact retained overlap without splicing. HF short-cache controls each yielded KL0/top1match. All arms used pinned revision842da3794eaa0b77d5f08bae87a17459d91ff475, full60 layers; TT B8/current11a387 selected policy.

Costs: case0 device0.093609712h vs0.08forecast, HF329.345s vs170guess under concurrentCPUreference. Conditional case1 device0.046211711h vs0.06guess, HF121.572s vs330forecast with TTfirst then HF. Total0.139821422deviceh versus0.20box, and0.14h combined forecasts. No timing/performance inference from these different regimes and prefixes. Initial guessed1.3h release saving remains unconfirmed; no change produced a saving.

HF support on an existing TT-generated prefix does not show HF would generate that prefix independently. The point immediately before the exact periodic suffix may already be inside a degenerative trajectory. TT uses teacher-forced direct prefill/cache state, not original served decode KV/APC/scheduler state. Earlier divergence, decode-only errors and sampler effects remain possible; KL alone neither assigns nor excludes eventual cause. No score gate, new scored config, benchmark performance or release-fit claim.
