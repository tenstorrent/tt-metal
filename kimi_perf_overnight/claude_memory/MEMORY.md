# Memory Index

- [Kimi 61L op2op profiling (ACTIVE)](kimi-61L-op2op-profiling.md) — paused task: per-layer op2op for all 61 layers w/+w/o service; resume from kimi_perf_overnight/RESUME_CONTEXT.md; profiler-buffer + disk blockers

- [Kimi prefill env vars](kimi-prefill-env-vars.md) — env vars for Kimi K2.6 chunked prefill tests; runner needs only PREFILL_MODEL_VARIANT=kimi_k2_6
- [Kimi chunked prefill work state](kimi-chunked-prefill-work-state.md) — branch ppopovic/kimi_chunked_prefill_device_gate; PR #46761 reworked to DEVICE_FP32 grouped-topk gate (cherry-pick f02e7f0dfa8 + b6bcae28665); all tests + runner/producer PASS; CRITICAL build gotcha (refresh build_Release/lib/_ttnncpp.so after ttnncpp rebuild)
- [Firmware/host mailbox mismatch](firmware-host-mailbox-mismatch.md) — profiler sync timeout + 30-min embedding hang = host-lib vs JIT-firmware mismatch (e.g. unreverted dev_msgs.h); reset won't fix; git checkout silent-abort gotcha
- [Kimi perf overnight](kimi-perf-overnight.md) — overnight harness at /home/ppopovic/kimi_perf_overnight (tmux kimi_perf) investigating runner ~3.3s vs test ~1.94s/chunk gap; leading theory = sub-device load/clear; decisive exp = PREFILL_STANDALONE_CHUNKED
