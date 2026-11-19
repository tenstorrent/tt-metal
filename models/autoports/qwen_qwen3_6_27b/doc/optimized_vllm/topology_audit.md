# Serving topology audit before tuning

Selected policy is datatype_sweep/selected_precision_config.json. The completed integration measures37.633t/s/u versus selected full-model37.8177t/s/u at128in/128out/B1 (99.51%). These prior measurements prioritize serving prefill/interference over decoder math; new same-harness measurements are running.

|Operation sequence|Current contract|Candidate / action|
|---|---|---|
|Scheduler chunk, serial rows, internal chunks≤512|External KV, model-owned GDN; arbitrary starts/lengths|Sweep scheduler512/2048/4096 under31 decoders; internal chunk remains512|
|Unaligned continuation, full-attention layer per token|Norm/QKV/attention/output/MLP and3host uploads each token|AutoDebug inspection; test vectorized projections plus safe cache/attention or general chunk operation|
|Decode/model trace then sampling trace|Nonblocking; resident token/position feedback; B1/B32|Preserve; async boundary and stale tests retained/repeated as affected|
|Scheduler page update|B1 shared prefix table; B32 copies table to16full-attention layers|Secondary shared-buffer candidate; requires trace lifetime and ownership proof|
|Page validation|Repeated min/max and diagnostics over32×8192 host entries|Candidate hoist repeated reductions; preserve complete validation|
|Decode projections/MLP|TP4 packed projections, DRAM sharding, fractured residual, persistent CCL|Inherited selected policy; geometry/topology families deferred until measured serving bottleneck warrants|
|Terminal norm/head/local candidates/gathers/sampler|Vocab sharded BFP8/HiFi2 head; canonical width64 sampling; no full-vocab gather|Preserve full-model path; no adapter sampling substitution|
|Batched prefill|Only exact32rows/equalinterval/slotorder uses dense batch|Optional4096budget off/on candidate; no adoption without bit-equal/watcher/trace evidence|

No device profiler is used in this stage. Earlier full-model and datatype evidence supplies device-kernel context; current serving evidence is benchmark JSON and runtime contract logs. Independent CPU audit supplied the hypotheses; they are not measured performance conclusions.
