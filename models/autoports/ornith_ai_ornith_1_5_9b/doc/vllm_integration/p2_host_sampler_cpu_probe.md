# P2 host sampler CPU preparation

This is a short CPU microbenchmark, not P2 execution, a serving performance fix,
or evidence that a failed acceptance gate may be waived. No device or HTTP
request was made. No runtime/plugin source, running process, or global thread
setting was changed.

The installed `readiness_check/eval_request_profiles.json` reasoning profile is
**temperature 1.0, top-k 40, top-p 0.95, seed 42, n 1**, with thinking enabled
and a 32768-token harness budget. `run_harness_requests.py` resolves this profile
and caps the output budget for the probe. Temperature 0.6 is not this installed
profile. Top-k 40 exceeds the model's current device sampling boundary of 32,
so this workload requires host sampling.

The probe imports the actual `create_host_sampler()` from the clean plugin at
`2a1504554aa7babb1d426dfbca8c7db8c0a795ef`, with installed upstream vLLM code.
It selects `TTPlatform`, and first imports the plugin's official
`tests/host-stubs/ttnn` module, whose device entry points raise. It asserts that
no native TTNN extension was imported. Source hashes and platform/stub paths
are recorded in each result.

Each of the 1/2/4 PyTorch intra-op settings runs in a fresh subprocess, with
interop threads 1. The sampler's own pool remains enabled: 16 pool workers on
this host, and four sampling row workers for B32 at every tested setting.
There are three warmups and ten measured full sampler calls per case. Inputs
are FP32 CPU tensors of shape `[1, 248320]` or `[32, 248320]`, generated once
from a fixed normal distribution and restored outside timing. Each row owns
a persistent CPU generator seeded 42. There are no penalties, logprobs,
grammar constraints, or custom logits processors. The actual unit-temperature
fast path and speculative noise prefetch remain active.

| PyTorch intra-op threads | B1 sampling median ms | B32 sampling median ms | B32 ratio versus 1 thread |
| --- | ---: | ---: | ---: |
| 1 | 18.034 | 152.978 | 1.000 |
| 2 | 18.357 | 146.766 | 1.042 |
| 4 | 18.512 | 152.216 | 1.005 |

These are PyTorch lower medians of ten samples. All measured sampled IDs and
final per-request generator states match the one-thread run exactly. B32 mean
times are 153.412, 153.124, and 160.014 ms respectively, so even the apparent
4.2% median gain at two threads is not compelling. This shared Ryzen 7 9700X
host has 16 logical CPUs; runs were sequential and not isolated from other
workloads. No thread-change recommendation follows from these small samples.

The installed upstream `apply_top_k_top_p_pytorch` sorts the complete vocabulary
when top-p is present, including when top-k is only 40. That explains why
top-k 40 does not make this CPU path proportional to just 40 candidates; this
run is not an operator-level attribution experiment.

Device readback is separate and **unmeasured**. Production
`OrnithGenerator.process_decode_output_host` synchronizes and converts/combines
device logits, slices the vocabulary, and clones the host result. The plugin
then selects rows and builds sampling metadata before the timed sampler entry.
The FP32 host logits alone contain 993280 bytes per active row or 31784960 bytes
at B32. Those are host payload sizes, not claimed device transfer sizes or
transfer latencies. The benchmark also excludes model decode, output processing,
networking, and scheduler effects. The final speculative pool drain is outside
per-call timing and is included only in the recorded whole-case wall time.

Reproduce from the repository root:

```bash
python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/tests/vllm_host_sampler_cpu_probe.py \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/p2_host_sampler_cpu_probe.json
```

Artifacts: `p2_host_sampler_cpu_probe.json` (combined report),
`p2_host_sampler_cpu_probe.threads{1,2,4}.json` (individual subprocess provenance,
all timings, tokens, and RNG hashes), and `p2_host_sampler_cpu_probe.log`.
The new script passes `ruff check` and `ruff format --check`; the actual short
CPU run completed with exact token/RNG cross-thread checks. P2 still requires
the actual serving acceptance and performance run.
