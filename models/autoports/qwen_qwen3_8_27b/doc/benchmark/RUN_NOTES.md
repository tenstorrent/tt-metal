# Stage 11 run notes

Setup began 2026-10-25 14:21 UTC; see setup.json. Stage10 left no live server; its last B1 configuration and prior results are retained under optimized_vllm. Prior startup duration is unknown. The first launch will count inside benchmark_stage run.

Four configured batches give a 7200-second total client budget. All launches, compilation, warmups, verification, inference, phase collection and reporting count.

GPQA uses the frozen common 10/198 subset; exact mirror population hash verified. See protocol.json for all settings. SWE-bench Verified and Terminal-Bench 2 require goal12 agent tools and are not covered by lm-eval. No accuracy acceptance threshold is imposed.

Phase accounting is host wall time, not device time. No live device profiling or added synchronization. Detailed estimated work and limitations are in tests/benchmark_work.py.
