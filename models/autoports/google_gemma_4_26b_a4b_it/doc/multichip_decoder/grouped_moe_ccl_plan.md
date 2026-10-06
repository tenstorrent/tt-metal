# Grouped MoE BF8 communication candidate

Unapplied, syntax-checked patch against runtime070613dd. No numerical or latency
claim. The candidate is opt-in by setting `decoder.moe_ccl_bfp8 = True` at setup;
its default remains BF16. Attention communication must remain fixed during the
comparison. It applies to the replicated/grouped path only. Shared and routed
local BF16 outputs are concatenated along dimension1, cast to BFP8 for RS+AG,
then restored to BF16 before splitting and applying their independent norms.
It never adds the branches before normalization or changes expert selection.

Per device, decode logical shape is[1,2,1,2816], padded[1,2,32,2816]:176tiles.
BF16 storage360448bytes becomes191488bytes at1088B/BFP8tile. RS output has
44tiles before AG restores176tiles. Prefill1024 chunk has5632tiles,11534336B
BF16 versus6127616B BFP8. These are tensor storage sizes, not estimates of all
fabric traffic or native scratch allocation. Expected payload reduction is
46.875%; two casts and accuracy cost may erase any benefit. Keep maximum-context
reservation conservative at the existing BF16 sizes during this trial.

First run both ordinary paired4096/128 trace/cache/replica/PCC checks; then
mixed-stack and batch/page controls if the candidate wins. A replay/replica
failure requires AutoFix localization, not a lower PCC threshold. Source and
patch hashes are in grouped_moe_ccl_provenance.json. No current source is edited.
