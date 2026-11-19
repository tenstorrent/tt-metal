# Additional representative layers for native decode norm

Recommend **layer34 first, then layer14** (zero-based), both `linear_attention`. These are two additional diagnostic representatives, not replacement gates or full-model coverage. Existing layer0/3 quality requirements stay unchanged. No hardware or production edits were performed.

The pinned checkpoint revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7` has40 text decoder layers:30 linear layers and10 full-attention layers, with full attention exactly at `3,7,...,39`. All share hidden size2048, epsilon1e−6; linear attention has32 value heads of dimension128 and four convolution taps. Thus both recommended layers exercise the same native decode gated-normalization geometry as layer0, while layer3 exercises a different attention kind. Another full-attention layer would not directly test this linear-only rewrite.

[Norm-scale survey](v5_layer_norm_scale_survey.json) contains every layer's vector statistics, effective-gamma convention and tensor hash. Reads were restricted to130 small norm vectors: **345600 bytes (337.5KiB) of tensor payload** across the checkpoint. No projection matrices, activations or experts were read, and TTNN was not imported. Config/index hashes and checkpoint revision are included. Static norm scales describe representative parameter diversity; they cannot establish projection magnitudes, runtime activation distributions or numerical sensitivity by themselves.

HF source `modeling_qwen3_5_moe.py:185–199` uses direct `linear_attn.norm.weight` after rounding normalized values to input dtype. Its ordinary RMSNorm at `:817–831` instead multiplies by `1 + weight.float()`. The survey follows those distinct rules, matching decoder setup. The changed native decode adapter acts on the128-element direct gated-norm gamma; generic input/postnorm statistics below supply surrounding scale context.

| Layer | Depth/kind | Direct gated gamma min/max | Gated gamma RMS | Gated gamma population std | Effective input gamma RMS/max | Effective post gamma RMS/max |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 0 | existing early linear control | .523438 / .957031 | .885739 | .052941 | 1.032273 / 1.330078 | .908804 / 1.474609 |
| 14 | middle linear, recommended | **.292969** / 1.062500 | .898365 | .086448 | 1.017383 / 1.617188 | 1.275851 / 1.640625 |
| 21 | middle comparison | .539063 / 1.132813 | .890780 | .074436 | .927832 / 1.535156 | 1.319946 / 1.640625 |
| 34 | late linear, recommended | .585938 / **1.289063** | .944870 | .100751 | .972850 / 1.664063 | 1.543634 / 2.007813 |
| 37 | late comparison | .500000 / 1.156250 | .953786 | .108170 | 1.087584 / 1.757813 | 1.676543 / 2.179688 |

- **34** has the largest direct gated-gamma element among all30 linear layers, above1 where layer0's entire vector is below1. Its gamma variation is about1.9× layer0's population standard deviation, and its post-attention effective gamma RMS is about1.70× layer0's. It is a concrete late-layer test of a different rounding/scaling regime. These ratios are scale descriptors, not predicted PCC degradation.
- **14** has the smallest direct gated-gamma element among all30 linear layers. Its max/min ratio is about3.63 versus1.83 at layer0, covering stronger per-channel attenuation and a broader relative range at middle depth. This complements layer34's high-gamma case. All surveyed direct gated-gamma entries are positive and nonzero; no unsupported negative-gamma scenario is invented.

Layer37 has larger postnorm RMS than34, but34 is selected because its direct gated-norm gamma maximum is the stronger discriminator for this specific rewrite; limiting the recommendation to two avoids a broad layer sweep. Use the existing fresh-reference decoder comparison and unchanged thresholds on these layer indices if root elects to run them. Any isolated external-input layer test remains a decoder generalization check, not evidence about activation distributions in an assembled40-layer model.
