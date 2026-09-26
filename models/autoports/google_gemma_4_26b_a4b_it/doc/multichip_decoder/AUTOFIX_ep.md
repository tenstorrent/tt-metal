# AutoFix: EP4 expert candidate

Starting diagnosis: `AUTODEBUG_perf.md`. TP4 leaves12 gate/up output tiles per
rank while executing all8 selected experts on every rank. EP4 can keep width704
and44 gate/up cores while computing only locally owned selected experts. A
complete-workload speedup remains a hypothesis because rank imbalance, full-width
down projection, mask scans and expanded buffers can offset the gate gain.

## Implementation

Only `tt/multichip_decoder.py` and the stage-scoped
`tests/probe_multichip_expert_parallel.py` changed for this experiment.
`MultichipDecoder.from_state_dict(..., expert_parallel=True)` enables the
candidate; the TP4 default remains intact. Attention/QKV v2 settings are unchanged.

Each rank owns32 complete experts, with setup-owned global routing-column IDs.
Gather produces the rank's routing mask, with32 repeated index rows for prefill.
Gate/up uses44 cores; down uses88. Decode and prefill use no `nnz` and no sparse
`indices`, preserving active-expert projection and allowing zero active experts
on a rank. Decode mixes32 zero-filled local slots; prefill uses its32-token union.
The existing routed collective remains outside the adapter. Precision policy is
the existing BF16 load followed by BFP8/BFP4 gate and BFP4 down conversion,
LoFi expert compute, and BFP8 decode activation for full-attention layers only.

## Focused verification

The probe uses real checkpoint weights and recorded layer inputs, controlled
top8 routing, a CPU oracle using the dequantized selected device weight policy,
and the same collective used by the decoder. It checks every local output,
allreduce replica equality, exact zero on empty ranks, and exact eager/trace
equality while replay changes local counts between0 and8. It also checks two
prefill unions:32/0/0/0 and32/32/32/32.

`python -m py_compile` passed for both edited Python files. No C++ build needed.

### Watcher attempts and recovery

1. `HF_HUB_OFFLINE=1 TT_METAL_WATCHER=1 timeout 300 python -m
   models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_expert_parallel
   --samples 2 --output .../ep_watch.json`: exited1 before expert setup.
   Fabric ACTIVE_ETH program28224 bytes exceeds26624-byte config buffer.
   Console: `ep_watch.log`.
2. During recovery, a list and a reduced-check retry were mistakenly started
   before the asynchronous reset finished. The retry was terminated during
   topology discovery; it never executed EP. This overlap is invalid evidence.
   Exact terminated retry settings were `TT_METAL_WATCHER=5
   TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1`; no acceptance uses them. Reset/list
   exited0. The following commands were strictly serialized.
3. `timeout 60 tt-smi -ls --local` exited0 with all4 devices. A bounded FABRIC_1D
   1x4 open/close smoke with `TT_METAL_WATCHER=10
   TT_METAL_WATCHER_NOINLINE=1` exited0 (`ep_mesh_watch_noinline.log`). This
   preserves every watcher feature and solves the instrumentation code-size
   problem for this configuration.
4. Full probe with `HF_HUB_OFFLINE=1 TT_METAL_WATCHER=10
   TT_METAL_WATCHER_NOINLINE=1
   TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/ep_watcher_noinline
   timeout 300 python -m
   models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_expert_parallel
   --samples 2 --output .../ep_watch_noinline.json` completed all16 correctness
   comparisons, wrote JSON and printed `EP_PROBE_PASS`, then **exited134** on a
   watcher assert during fabric teardown. This is not a clean watcher pass.

The lowest local CPU PCC was0.9994366683; the lowest total PCC was the same.
All empty-rank outputs were exactly zero, all collective replicas matched,
and decode replay output was bitwise equal to eager under changing routing.
Prefill total PCCs were0.9997378118 and0.9996631495.

The teardown assertion is on device0 acteth core(0,8), virtual(28,25):
`subordinate_erisc detected invalid NOC command buffer state before starting
the next kernel (write-capable NOC packet tags must be zero so implicit
transaction ID users start with transaction ID 0)`; current kernel is
`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`.
Logs and inspector snapshots are under `ep_watcher_noinline/generated/`.
No sparse expert assertion was reported. Causality is not assigned to EP or to
an existing fabric issue without a discriminating control.

Post-abort `timeout 120 tools/tt-triage.py --llm-output
--run=dump_callstacks --run=check_eth_status ...` exited1 because the host was
already gone and the default inspector path did not exist. Its failure is
preserved in `triage/ep_watcher_capture.log`; watcher and inspector artifacts
remain available. No live process was killed for this assert. Bounded reset
`ep_reset_assert.log` and four-device list `ep_list_assert.log` exited0.

## Normal runs

Both commands completed with exit0 and16 passing comparisons:

```bash
HF_HUB_OFFLINE=1 timeout 300 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_expert_parallel --samples 8 --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/ep_normal_layer0.json
HF_HUB_OFFLINE=1 timeout 300 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_expert_parallel --layer 5 --samples 8 --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/ep_normal_layer5.json
```

Matching `.log` files preserve console output. The normal mesh open/close after
assert recovery also exited0 (`ep_mesh_after_assert.log`). Device commands are
complete and hardware is released to the coordinating agent.

Measured source SHA256:

- `tt/multichip_decoder.py`:
  `0dceb74b26029a63dc9a2747c35906de3320e1bf52f4068e4a559eaba2dd577a`
- `tests/probe_multichip_expert_parallel.py`:
  `b8c1ee99658e5de940fe51b9aa2c16802fdeaaf8907ec9faaba0b5098f35c64f`

These component timings cover the EP branch and its allreduce, with eight traced
host-wall samples per case, and exclude input refresh. They are not decoder
latency or a measured improvement against TP4.

| Case | Local active counts | Layer 0 median us | Layer 0 total PCC | Layer 5 median us | Layer 5 total PCC |
| --- | --- | ---: | ---: | ---: | ---: |
| Rank 0 owns all | 8/0/0/0 | 301.037 | .9996426493 | 286.330 | .9997842194 |
| Rank 1 owns all | 0/8/0/0 | 303.603 | .9996634968 | 284.547 | .9998832288 |
| Rank 2 owns all | 0/0/8/0 | 307.059 | .9994366683 | 286.184 | .9998789821 |
| Rank 3 owns all | 0/0/0/8 | 303.768 | .9997436474 | 281.276 | .9998248151 |
| Balanced | 2/2/2/2 | 219.963 | .9996026276 | 219.638 | .9998868884 |
| Mixed | 1/1/1/5 | 260.691 | .9995881507 | 251.313 | .9998790707 |
| Rank 0 active again | 8/0/0/0 | 301.148 | .9996426493 | 282.182 | .9997842194 |
| Prefill empty ranks | union 32/0/0/0 | not timed | .9997378118 | not timed | .9995969773 |
| Prefill changing routes | union 32/32/32/32 | not timed | .9996631495 | not timed | .9996060384 |

All per-rank comparisons, not only summed outputs, passed. Minimum local PCCs
were .9994366683 for layer 0 and .9994856885 for layer 5. These controlled routing
patterns validate the component and expose imbalance; they are not a sample of
production router load distribution.

## Status

The active-count correctness hypothesis is supported, including0/1/8 local
counts, replay changes and prefill union masks. Normal-process verification passed
for layers0 and5. Full decoder A/B and performance verdict remain pending. Watcher acceptance
remains open because of the post-test fabric assertion. No stage completion or
performance improvement is claimed.
