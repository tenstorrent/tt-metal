# Diagnostic client timeout diagnosis

Failing contract: --request-timeout600 must allow a long prefill to wait longer than60seconds for HTTP headers or the next streaming data. Coordinator observed cold107585 and overlapping107585 requests fail near60.09seconds while the live anchor continued making progress.

Current source check_serving_load_costs.py used urlopen(timeout=min(60,args.request_timeout)). This independently clamps the socket/header/read wait at60seconds regardless of the configured600second deadline. The loop's elapsed-time check runs only when data arrives, so it cannot relax this earlier transport timeout. Cause is direct and supported by the code and observed failure duration; no model/device failure is implied.

Minimal repair: pass args.request_timeout to urlopen. Preserve the existing total elapsed-time check and outer bounded client timeout. This fixes premature60second rejection; urllib socket timeouts remain per blocking operation, so a strict interrupt-at-total-deadline guarantee while blocked is not newly claimed.
