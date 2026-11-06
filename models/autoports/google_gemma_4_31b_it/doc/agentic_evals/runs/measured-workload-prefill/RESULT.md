# Measured-workload prefill result

The corrected diagnostic completed all five requests on the full60 stock B8 server: cold prefills at4096,31755 and107585 tokens, plus one2048-token decode anchor overlapping a107585-token prefill. The anchor remained live through the entire long-prefill window. Source: result-retry.json. Model/runtime and release settings were unchanged; only the diagnostic client's premature timeout was repaired.

| Cold prompt tokens | HTTP TTFT | Input tokens/s |
|---|---:|---:|
| 4096 | 2.228286s | 1838.184 |
| 31755 | 16.951805s | 1873.252 |
| 107585 | 72.683447s | 1480.186 |

Each cold request bypassed prefix reuse with an idle cache reset and unique initial prefix. With one decoder already running, the107585-token request's submit-to-first-token window was78.910179seconds, including scheduler wait. The decoder delivered28 tokens during that window, or0.354834tokens/s, and ultimately completed2048 tokens. The window occupied about54% of the anchor's146.243second request lifetime; only28/2048 of its output tokens arrived within it. This is an observed overlap share for this diagnostic, not a task generation-time fraction or a release speedup estimate.

These are host-wall HTTP measurements under concurrent16-thread HF CPU reference load (coordinator recorded HF PID93575). TT work remained serialized on the1×4 mesh. Do not compare these latencies causally against earlier idle-host measurements or use them as CI device performance. The107585-token length is the observed maximum prompt-plus-completion from150 saved SWE usage rows, with11 attempts missing usage; using it as a cold prompt is a conservative diagnostic shape, not an exact original request or universal working-set bound. No task score, device-time roofline or headline4096/128/B1 claim is made.

The initial readiness wait expired at300seconds during server startup without sending measurement requests. The first measurement attempt is preserved in result-first.json:107585 cold and overlap reads failed at60.088 and60.103seconds despite configured600seconds. The anchor continued and completed. AUTODEBUG.md identifies urlopen's hidden min(60,request_timeout) clamp. AUTOFIX.md records local generic defect branch fix/diagnostic-request-timeout, commit536febd8, and the one-line diagnostic-only fix. Three mocked tests validate30/600second propagation and retained total elapsed checking; repository hooks and subsequent Ruff checks passed. The successful retry's72.683/78.910second long responses provide live confirmation that requests now survive beyond60seconds. No shared plugin/runtime publication change was required.

The queue charged992.891927seconds (0.275803313device hours), including startup, readiness failure, failed first measurement and successful retry. Against the original540second forecast this is1.83869×; it remains inside the0.3hour box. Keep failure costs in this estimate, rather than charging only the successful client. The coordinator confirmed zero running/waiting requests, graceful shutdown and empty owners before the next device job.

The generic implication is that long-prefill interference can consume a large portion of a concurrent decoder's wall time, especially under host contention. Release pricing still requires actual task overlap and generation shares on the release route; predicted release savings remain zero. The measurement confirmation is complete, with no numerical optimization proposed and no further device work by this effort.
