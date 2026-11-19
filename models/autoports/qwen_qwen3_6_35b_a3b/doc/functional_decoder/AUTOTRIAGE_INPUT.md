# Live full-context functional decoder stall

Target: Qwen/Qwen3.6-35B-A3B; functional-decoder stage only. Checkout `/home/ttuser/qwen36-35b-a3b/tt-metal`, baseline 822d04806f292e2cf7e92de7f7be79fa8d83e5aa. All stage changes are under `models/autoports/qwen_qwen3_6_35b_a3b`.

Still-running PID 2087868 is under timeout 14400:
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.long_context --layer 3 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/full_long.json`

Source: `tt/functional_decoder.py`, `tests/long_context.py` under the model directory. No arithmetic change since launch (later formatting and setup checks only). HF reference completed in 751.83 seconds. Blackhole P300c, physical chip 0, 1x1 mesh, all devices discoverable, only chip 0 selected.

Passing controls: both layer kinds real-weight prefill/decode at lengths 1,31,32,33,255,256,257,513; batch 32; tracked decode; continuation negative controls. Linear attention also passes 262143/262144 prefill and decode position 262143. See `linear_long.json`, `linear_battery.json`, `full_battery.json`.

Preserved live evidence here:
- `logs/full_long_01.log`
- `triage/full_long_snapshot.txt`, `triage/full_long_summary.txt`. Summary pass labels mean script completion, not all detailed checks passing.
- `triage/host_stack.txt`, `triage/host_native.txt`
- `triage/host_locals_01.txt` (16:36:19 UTC), `triage/host_locals_02.txt` (16:37:50 UTC): identical absolute/start/offset 93696 in full-attention RoPE concat.

Native host stack is SystemMemoryManager::fetch_queue_reserve_back, polling UMD read_core while enqueueing ConcatDeviceOperation. Prefetch core waits at cq_prefetch.cpp:943 fetch_q_get_cmds; dispatch waits upstream. No running model op; worker cores idle in snapshot. Detailed triage has unused Ethernet NoC counter mismatches and cores resuming after triage halted them; do not infer healthy/faulty hardware solely from those warnings or script exit 0.

No reset, kill, or locks cleared. Further device jobs are held in a serial queue; STOP_QUEUE prevents later jobs. Resets are not authorized. Inspect evidence and sources only: no device use, process mutation, implementation edits, or target-code runs. Diagnose earliest inconsistent contract, separating upstream cause from downstream waits. Suggest discriminating checks and a decoder-local workaround if possible. Context 262144 and PCC >=0.995 must be preserved. Write AUTOTRIAGE.md here.
