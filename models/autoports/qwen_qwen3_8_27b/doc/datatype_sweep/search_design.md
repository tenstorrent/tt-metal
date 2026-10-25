# Search design and comparison scope

Every ranked point uses all 64 target-model layers, the unchanged main AIME24
chat reference, 203 input tokens and 100 generated positions, batch one on the
same 1x4 Blackhole mesh. The median of three warmed traced teacher-forcing runs
is the ranking statistic. Acceptance also requires prefill and teacher-forcing
top-1 >=0.90, top-5 >=0.98 and top-100 =1.00. A one-problem agreement test is not
an AIME task-accuracy estimate. Reduced probes never enter the Pareto frontier.

The inherited optimized policy already uses BFP4/LoFi for all decoder projection
groups, including the first and last layers. Its packed and separate MLP
representations are kept consistent. Therefore the canonical comparison restores
all three decoder groups together to BFP8, comparing LoFi and HiFi2. The same
BFP4 groups also receive a HiFi2 control. Full-model grouped comparisons measure
the effect on the complete workload, not an isolated group's speed.

| Material group | Model tensors | Required comparisons |
| --- | --- | --- |
| Attention projections | Packed attention/linear-attention input `mix`, output `out` | BFP4 LoFi/HiFi2 and BFP8 LoFi/HiFi2 |
| MLP gate/up | Separate prefill `gate`,`up`, packed decode `gateup` | BFP4 LoFi/HiFi2 and BFP8 LoFi/HiFi2 |
| MLP down | `down` | BFP4 LoFi/HiFi2 and BFP8 LoFi/HiFi2 |
| Vocabulary head | TP vocabulary-sharded `head` | BF16 HiFi4 baseline; BFP8 LoFi/HiFi2; BFP4 LoFi/HiFi2 |

Ordinary projection, fused prefill and fused decode fidelities change together in
the decoder fidelity controls. Head fidelity changes independently. Actual kernel
arguments, tensor inventories and source hashes tie policies to measured paths.
FP32 head destination accumulation is held fixed. Norm and recurrence fidelity,
BF16 residual/logits/sampling and FP32 recurrent state are supported constructor
invariants; unsupported alternatives are rejected explicitly.

KV controls compare BF16, BFP8 and BFP4 after real-layer paged fill/update and
nonaligned replay smokes. CCL controls compare inherited BFP8 with BF16 in the
batch-one decode collective. Fixed BF8 prefill and BF16 native-batched collective
representations remain explicit policy invariants. An ordinary activation BFP8
candidate tests the supported input-conversion path independently. If independent
changes offer compatible wins, combine them and measure the full model again.

Allocation failures are diagnostic attempts, preserved alongside repaired results.
They cannot be ranked as throughput points. The BFP8 and BF16-CCL geometry repairs
preserve mathematical operations and trace ownership; exact dtype-aware local
buffer budgets and real-target replay evidence are in AUTODEBUG.md/AUTOFIX.md.

Final selection remains conditional on prompt-correct free-running replies,
long-prefix factual retrieval, native context and the agent memory ledger. The
selected config must then run the original warmed no-readback autoregressive
benchmark through default construction. That separate token-out result supplies
the full-model/telemetry headline and subsequent serving comparisons.
