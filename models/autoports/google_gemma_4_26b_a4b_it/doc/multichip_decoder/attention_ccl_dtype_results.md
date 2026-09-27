# Attention collective datatype and refreshed EP comparison

All runs target4096 input tokens,128 decode tokens,batch1,1x4 Blackhole,
FABRIC_1D/Linear. Policies: fused tail, optimized shared, shared geometry1,
grouped MoE reduce, LoFi QKV and WO. Hybrid means EP prefill/indexed TP decode;
EP-only disables hybrid and keeps active selected expert execution. Exact
commands, runtime SHA256, all per-step PCC and host timings are in the JSONs.
No runtime or runner files were changed by this experiment.

| Kind | Experts | Attention CCL | Min output PCC | Min cache PCC | TP4 prefill host us | TP4 decode host us | Artifact |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| sliding_attention | hybrid | BF16 | 0.998854134 | 0.999996658 | 93769.717 | 735.835 | sliding_ccl_bf16.json |
| full_attention | hybrid | BF16 | 0.999476209 | 0.999971502 | 79437.910 | 728.622 | full_ccl_bf16.json |
| full_attention | hybrid | BFP8 | 0.999447756 | 0.999971502 | 78994.006 | 723.983 | full_ccl_bfp8.json |
| sliding_attention | EP-only | BF16 | 0.998863212 | 0.999996658 | 93858.658 | 808.398 | sliding_ep_ccl_bf16.json |
| full_attention | EP-only | BFP8 | 0.999450533 | 0.999971502 | 78908.335 | 798.715 | full_ep_ccl_bfp8.json |

All table rows passed the .995 PCC gate, all-rank local KV comparisons,
runtime fallback guard, and exact repeated trace equality for all128 decode
steps. Host timings include the complete layer and are not device metrics.
BF16 attention CCL improves the supplied FP32 controls by about13us for both
kinds. Full BFP8 gains a further4.64us in this run. Refreshed EP-only decode
loses about72.56us sliding and74.73us full against indexed TP decode under
matching dtype/projection/shared/collective policies.

## Failed candidate and recovery

`sliding_ccl_bfp8.log` failed at runner line308:
`AssertionError: Replay is not deterministic`. The process reached actual
multichip execution; this was not dtype/API rejection. No JSON was written
because replay assertion precedes result creation; its PCC, cache and timing
measurements are unavailable and must not be invented. The process closed all
devices normally,exit1; it did not hang and no process kill was needed.

Following the failed multichip run, serialized reset/list/FABRIC_1D 1x4 mesh
open-close all exited0 (`ccl_dtype_reset1.log`, `ccl_dtype_list1.log`,
`ccl_dtype_smoke1.log`). Four ASICs were visible; no second reset or lock
cleanup was needed. The remaining full BFP8 and EP-only runs then passed.
Hardware ownership returned to root after all six requested runs.

Full-attention BFP8 success does not resolve sliding replay instability.
`tests/probe_attention_ccl_replay.py` is prepared but not yet executed. It
checks128 synthetic seeds/three duplicate replays for attention-shaped
logical[1,1,1,2816],physical[1,1,32,2816] local partials. Default captures
FP32->selected dtype cast plus native RS/AG; `--precast` uploads selected dtype
before capture, isolating native RS/AG from conversion. It saves first failing
seed/rank/change count/max absolute difference if instability reproduces.
These controls will not replace real-WO capture if synthetic inputs do not
reproduce the failure. No conclusion that BFP8 is universally unsupported or
inherently nondeterministic is earned by the single failure.
