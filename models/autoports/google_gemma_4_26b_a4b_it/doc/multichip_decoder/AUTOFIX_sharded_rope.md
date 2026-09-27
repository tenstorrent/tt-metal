# AutoFix: optional sharded decode RoPE

## Initial source diagnosis

Runtime `c96c935847572085ef97b67ebd76c79eaaf31a45b3f6b90277d69d4951e3fc8d`
and runner `ac9c1e9d55b66724c00a3432b4affdf0910d6503dfd53bd70e2ab3f88bc50bb0`
introduced optional sharded decode RoPE, disabled by default. The first ordinary
sliding 4096/128 trial (`sliding_router109_sharded_rope.log`) passes exact replica
and repeated trace checks, but key-cache PCC is 0.958–0.965 while value-cache PCC
is above 0.999996. No production/default change is authorized beyond the optional
rotary override. Native C++ edits are excluded.

The existing decoder embeds the current cosine/sine row, supplying tables
[1,1,1,D], and normalized FP32 Q/K [1,1,heads,D]. The interleaved RoPE baseline
repeats the same row across heads. The optional native decode contract instead
broadcasts row zero over heads on a height-sharded [32,D] core. Those logical
contracts appear consistent; the key-only error points at rotary math.

Source hypothesis: the multi-tile sharded kernel computes and packs all Wt=D/32
tiles within a single destination-register acquire. Current FP32 destination
accumulation uses half synchronization. D256 requires eight simultaneous tiles,
D512 sixteen. The factory extracts `dst_full_sync_en` but only forwards
`math_fidelity` and `fp32_dest_acc_en` into its ComputeConfigDescriptor; explicitly
requesting full synchronization therefore cannot change the compiled policy.
This may exceed destination capacity rather than represent a table-layout error.
Relevant sources are `rotary_embedding_hf_sharded.cpp` and
`rotary_embedding_hf_sharded_program_factory.cpp` under the native operation.

A fresh independent source-only CLI review is running under the coordinator,
writing `AUTODEBUG_sharded_rope_fresh.md`. No implementation edit preceded this
initial report. First component controls will use exactly the model's FP32
input, BF16-rounded tables promoted to FP32, HiFi4 compute, head counts and tile
padding, comparing native interleaved RoPE and CPU rotate-half against sharded
D64/128/256/512. Explicit full-sync and BF16 destination controls distinguish
capacity from math/table hypotheses. Any model-local adaptation must pass the
original paired key/value-cache, accuracy and strict replay checks.

Hardware ownership was handed over after reset/list/smoke all exited 0. No C++
changes or unrelated runtime/runner edits are permitted during this investigation.

## First component results

`sharded_rope_contract.json` shows failures even at safe-width D64/D128 with
FP32 inputs/tables (PCC0.57–0.93 for several head counts); interleaved baseline
PCC is approximately0.9999999. D256/D512 and full-sync requests also fail.
Therefore DST capacity is not the sole cause. No implementation change yet.
The sharded kernel omits the explicit data-format switches used by the
interleaved kernel, despite initializing SrcB from FP32 sin and later using a
BF16 scalar, then changing operands for addition. Next controls use homogeneous
BF16 operands and separate input/table dtype changes to test that source lead.
The negative component run closed normally; reset/list/smoke recovery follows.

## Verified format/width control and optional adaptation

`sharded_rope_formats.json`: homogeneous BF16 input/tables with BF16 destination
is bit-exact to native interleaved at D256, CPU PCC0.99999656. Homogeneous BF16
D256 with FP32 destination fails (PCC0.687), and D512 with BF16 destination fails
(PCC0.719), while D64/D128 homogeneous controls pass. Mixed FP32/BF16 D128 fails
(PCC0.089). Thus both uniform data formats and destination-capacity limits matter.
No native C++ changes are needed for a model-local supported-shape control.

Only the optional `_LocalAttention.rotary` is adapted: use homogeneous BF16
input/tables and BF16 destination at width256. For D512, pair128 features from
each global rotate-half side into two D256 native calls, then restore feature
order, original output dtype and original memory config. This explicitly adds
BF16 rounding before rotary; acceptance depends on ordinary real-weight/cache
PCC, strict replays, and measured whole-layer latency. The old runtime is saved
as `rope_runtime_before.py.txt`; unrelated runtime/default/runner fields remain
unchanged. Component correctness of D512 packing is next before paired gates.

`sharded_rope_adapted_contract.json` passes all six combinations of D256/D512
and one/two/four heads: all replicas finite/equal, minimum CPU PCC0.99999421.
The adapted runtime hash is c74f191d446a67ae75a7334dda513bdd4a664ae46e958a1a49d6d1ad79c4f508;
its diff is `sharded_rope_supported_candidate.patch`. Fresh source review in
`AUTODEBUG_sharded_rope_fresh.md` independently confirms the capacity/factory
issues; format controls supply additional runtime evidence. Serialized recovery
rope_probe_reset/list/smoke1 and2 completed exit0 after negative component probes.
Ordinary paired sliding/full accuracy and cache checks are now being measured.

## Ordinary paired sliding result

`sliding_sharded_rope_adapted.json` passes real4096/128 output/cache PCC and
all strict trace/replica checks. Minimum output PCC0.9978613806 (baseline
qkv-n2 0.9987453344), minimum cache PCC0.9999966584. Median TP4 host decode
671.712us versus prior same-geometry flag-off731.481us; TP1 controls826.041
versus825.866us. This is host latency evidence, not a device-profile claim.
The precision tradeoff is explicit in `sharded_rope_precision_ledger.json`: both
Q/K rotary arithmetic use BF16 inside this optional path, then restore FP32
output dtype. Existing caches/weights/fidelities remain unchanged. Full-attention
paired correctness and timing follow; the coordinator will perform final matched
controls and selection.

## Final bounded result and handoff

`full_sharded_rope_adapted.json` also passes ordinary4096/128 output/cache PCC
and exact replicas/replays: output minimum0.9997071333, cache0.9999717667.
Its median TP4 host decode is740.513us versus prior baseline724.293us, about2.2%
slower (TP1 controls876.417 versus875.921us). Reject the chunked full-attention
path as a speed candidate; retain the original full-attention path for selection.
The sliding adapter is a measured candidate at approximately8.2% lower host
latency, with the explicit BF16 rotary precision tradeoff and output PCC0.997861.
Final device-profile performance and matched final defaults remain coordinator
work; no extra hardware scope was taken.

Only the optional rotary override was changed, verified by comparing parsed
runtime ASTs with that method removed. Defaults, gatecore109/grid guard, other
runtime fields and native C++ were untouched. The corrected optional path remains
off by default. Source snapshot and results are in
`sharded_rope_supported_runtime.py.txt`, `sharded_rope_supported_candidate.patch`,
and `sharded_rope_adapted_provenance.json`. Python formatting/compilation checks
pass; no C++ build is needed.

Both paired runs closed normally with exit0. Hardware ownership was explicitly
returned to the coordinator after the full run; no subsequent TTNN/device calls
were made by this agent. The original failing native-FP32 candidate is replaced
by the bounded, numerically verified optional adaptation.
