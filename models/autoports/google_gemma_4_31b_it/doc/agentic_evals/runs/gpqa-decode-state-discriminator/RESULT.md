# Incremental decode-state discriminator

All six exact-prefix checkpoints matched HF and TTprefill top1. HF→incremental KL ranged0.0001589557–0.0200489066; TTprefill→incremental KL0.0000102779–0.0058435478. Prefix hashes bind the complete native captured paths including firstprefill tokens. No arbitrary threshold was applied. The observed alternating nexttokens have unit release-filtered probability inside both loops in incrementalTT as in conditional references; no concrete cumulative incremental-decode discrepancy emerged at these states. Recommend stopping this line absent new served-state evidence.

Full60/B8 ran4121 modeltrace steps with explicit captured input reloads and0sampler replays, reading six checkpoints. Driver elapsed275.560607s, decode114.712591s; these are diagnostic host timings, not serving/device performance. Actual queue cost0.077834787deviceh versus0.12hforecast, within0.25hbox. Exit0 and owners empty. Coordinator used proven cache path and860s+15s bounded execution, reserving25s queue margin; shared runtime unchanged.

This checks cumulative incrementalKV/modeldecode conditional behavior, not original live vLLM state: static pages, forced inputs and sampler bypass omit original APC/scheduler/remaps/buffered feedback/RNG. Agreement cannot rule out those mechanisms or establish task accuracy. No further generic sampler sweep is justified here; no fix or scored change accepted. The provisional1.3h avoided-tail opportunity is shared/nonadditive/unconfirmed, with no speed credit.

Compare-prefill outputs reuse generic hf/kl_hf_tt labels; interpret them as TTprefill reference and KL(TTprefill||TTincremental). Raw logits/tokens remain private. RESULT.json records six distributions, observed-token support, exact hashes and source references.
