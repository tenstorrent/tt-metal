# BFP8 decoder: eight physical TP4 replicas passed G0

Completed **Oct 9, 10:13:29 UTC**, after 45m19s including serial model loading.
All eight replicas generated identical warmup and repeated greedy outputs,
and passed isolated/concurrent execution. Every concurrency/isolated latency
ratio is within 0.01% of one, below the existing 3% interference limit. The
hardware pytest passed and the parent advanced to serving startup. This is
hardware/replica qualification, **not a GPQA pass**.

The exact-source head-only BFP8 G0 is retained as the matched baseline. All
runtime source hashes other than the effective precision override match. The
only policy differences are decoder weights BFP4 -> BFP8 and projection
arithmetic LoFi -> HiFi2; both keep the BFP8/HiFi2 head, BFP8 KV, FP32 state,
native recurrence and accurate-full-tile attention.

| Short-context traced decode, one user per TP4 | BFP4 decoder | BFP8 decoder |
|---|---:|---:|
| Eight-replica aggregate output tok/s | 281.33 | 211.54 |
| Replica 0 median token latency | 26.35 ms | 34.29 ms |
| Replica 0 output tok/s/user | 37.96 | 29.16 |

Measured aggregate throughput decreased **24.81%**; the initial replica-0
measurement decreased 23.16%. These include device sampling and synchronized
host completion, excluding prefill and host readback. They are not B16/32K
performance or client throughput. The matched [16K/32K comparison](../precision-perf-v1/README.md)
is queued to measure prefill and decode separately.

The pinned runtime is `20619e008a236aaf393937b222a60a5b03e49cdc`, pushed on
`anatarajan/qwen38-bfp8-control-runtime-20261009` from the original head-control
source. All 33 frozen runtime/config files match the active accuracy snapshot.
The bundle preparation additionally verified the effective G0 source hashes
against committed bytes. Neither publishing this source nor G0 changes the
full-GPQA score requirement of 177 naturally completed correct answers out of
all 198.

The full hardware receipts, native test log/XML, baseline and parent queue
are retained here. `full-model.json` SHA256 is
`62871c0fd29b981678e85cd04a0e017a583884ea1bf5ebbd174d4f05e662e54e`.
