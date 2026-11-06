# Host contention result

The existing contention-only client passed at the largest validated batch32. Both waves completed32 requests, each128 input/256 output tokens. The busy wave used16 unpinned workers on CPUs0–15, matching the available cpuset without a CPU quota; all owned stress workers were cleaned. No numerical change or performance gain is claimed.

| Measurement | Idle | Busy |
|---|---:|---:|
| Decode-only HTTP observation window | 8.245105s | 12.600877s |
| Delivered tokens in window | 8126 | 8128 |
| Aggregate delivered output tokens/s | 985.554417 | 645.034448 |
| Logged decode steps | 254 | 254 |
| Logged prefill steps | 0 | 0 |
| Engine summed process CPU/step | 187.086614ms | 83.149606ms |

Source: contention.json. Both windows pass the existing decode-only validity check and contain active_rows32. They also contain active_rows15 at their completion boundaries, so these are measured delivery windows with boundary effects rather than strictly32-live-row windows throughout. Lower engine CPU/step under contention does not indicate improved wall latency: throughput falls about34.55%. Values are consistent with prior stage10 measurements, without a controlled statistical claim about the difference between runs. CPU is process user+system time summed across all engine threads, not exclusive Python overhead, host-wall step duration or device time.

All64 requests passed; the last request ended32.717seconds after the client workload clock began. The plan predicted60seconds including client setup and assigned0.25device hours after startup. The HTTP workload fits that prediction; exact process/startup/device occupancy must be taken from the coordinator queue, not inferred from the HTTP windows. No errors or missing reasons remain in the successful artifact. This fulfills this effort's measurement confirmation; there is no proposed optimization to accept.

An earlier client attempt ran before readiness and failed connection with zero measurement requests and no surviving stress workers. It is preserved in before-ready-contention.json and before-ready-client.log. startup-retry-AUTOFIX.md diagnoses the coordinator shell falling through after its readiness wait; the correction required a successful health probe before retrying. The server was not restarted and no runtime source was changed. The subsequent successful contention.json supplies the live confirmation of that sequencing correction.

idle_shutdown.json records successful client exit and zero running/waiting requests before the coordinator requested graceful shutdown of its owned API process. It is a shutdown request record; final device-owner clearance and queue closure remain coordinator responsibilities. No additional device tests were started by this effort.

This synthetic raw-continuation check is not agentic accuracy, real-task capacity, or the4096/128/B1 headline workload. No device profiling or roofline fields should be derived from these host measurements.
