# GPQA80 new loop diagnosis

No actionable conditional-logit discrepancy was found at the four tested doc139 transition positions. No serving, precision or sampling change is accepted; capped degeneration remains an observed failure.

The fresh capture contains15,032 native tokens and237 prompt IDs. All15,031 native[1:] IDs match the original retained chain. The first token is observed only for the fresh capture; the original first-prefill token is still unknown. The six-token tail starts at fresh native index14520; tested targets are14519,14520,14536 and14537. Each HF/TT comparison binds the same exact case/prefix hashes. Original release limits126976/14400 remained unchanged; the diagnostic disconnected its own HTTP stream after observation and closed only after freshidle and drain.

All four HF/TT argmax identities match. Full-vocabulary KL ranges0.007823–0.026064 and JS0.001600–0.005854. The observed pre-tail token has rank5 onboth, with filtered probabilities0.07182HF/0.04792TT; entry has0.70579/0.73106; firstinterior0.75491/0.79819; nextdistinct0.30184/0.29885. Thus this is not the prior saturated probability-one period3 case, but all observed samples remain supported by both distributions. HF chunk-control KL/JS are zero. No arbitrary acceptance threshold is applied.

DirectTT uses full60 B8 on accepted f579 source. It is a conditional prefix test, not the original incrementalKV/APC/scheduler/RNG state; CPU release filtering is not exact device-sampler arithmetic. This result cannot establish autonomousHF behavior or identify the cause of earlier trajectory choices.

Doc86 is bound by complete-set elimination after known doc160 identity. Its four-token vocabulary begins at retained index8903, while the final period57 suffix starts76823 and lasts50152 tokens. It was not device-tested: no missing first token was synthesized and that fullperiod capture does not fit this bounded effort.

Capture cost0.178726197devicehours, TT0.047036724, total0.225762920, versus initial0.15guess/revised0.30forecast and0.35box. HF took763.592hostseconds versus1100forecast, peak RSS74163.1MiB. Actual sequence was captureclosed→TT→hostHF→compare, avoiding concurrent CPUreference contention. No release saving is credited.

Stop this conditional-discrepancy line without additional device runs. Normal rung/dry progression remains independent. RESULT.json carries compact measurements and private-file hashes; no raw question, response or token arrays are published.
