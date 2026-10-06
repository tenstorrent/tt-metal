# Rejected first-decode scheduling yield

Hypothesis: the separate EngineCore output thread needs an opportunity to
serialize/send the already-completed prefill result before asynchronous decode
host work continues. Source inspection establishes that prefill returns a done
future and is enqueued before the next decode. TTNN execute_trace already
releases the GIL, so whole-decode GIL starvation was never established.

The experimental adapter added default-off
`GEMMA4_PREFILL_HANDOFF_YIELD=1`, a pending flag cleared on every prefill entry,
and armed it only after a successful prepared B1 prefill host read. Decode
consumed the flag before configuring sampling. The intervention was exactly:

```python
if (enabled and pending and not on_host and not read_from_device
        and int(torch.count_nonzero(torch.as_tensor(start_pos) >= 0)) == 1):
    time.sleep(0)
```

No positive sleep interval, global interpreter scheduling change, device call,
steady-decode yield or multi-request yield was added.16 CPU contracts passed
(`handoff_host_tests.log`; archived test source alongside this document).

Full30-layer native runs, unchanged32-row/K22 expert geometry, ten repeated
prompt0 warmups,10 measured S128/O16/C1 requests:

| Case | Initial median TTFT | Repeated median TTFT | Repeated TPOT |
|---|---:|---:|---:|
| Async control |137.657ms (3 warmups)|105.592ms (10 warmups)|18.882ms|
| One-shot yield |131.058ms (10 warmups)|106.205ms (10 warmups)|18.854ms|

S128/O128 repeated control107.092ms/19.107ms TPOT; yield105.712ms/19.123ms.
This is not a consistent primary-target win. The hook and configuration flag
were removed from production code. The archived CPU test is intentionally not
an active deployment test. Raw runs: `selected_async*` and `handoff_async*`;
host-only event exports retain process/source identity and completion times.

Exact launch differs from the normal `ttft_server.py` launch only by Docker exec
environment `GEMMA4_PREFILL_HANDOFF_YIELD=1` and
`GEMMA4_BENCHMARK_CONTROL=/tmp/gemma4-ttft-handoff-observer.sock`.
The benchmark command is `ttft_benchmark.py --lengths 128 --requests 10
--warmups 10 --output <handoff_async or handoff_async_repeat>`.

The initial declining-latency anomaly is not a prefill compute regression:
control runner-prefill medians93.694ms initial and94.428ms repeated; initial
pre-dispatch/post-completion medians24.364/19.640ms versus2.651/8.151ms repeated.
The native warmup repeats only the first prompt, so repeating the experiment
also warms the remaining prompt/tokenizer/frontend inputs. Exact cause remains
under investigation; do not equate a repeated-prompt result with unseen-input
tail latency. All initial distributions are retained, not discarded.

Fresh-process pinned-tokenizer control (`tokenizer_probe.json`) rules out raw
BPE cost as the tens-of-milliseconds delay: all10 first-pass prompt encodes are
sub-millisecond and match the saved128 token IDs exactly; repeat calls are
approximately0.1ms. This does not measure API event-loop or tokenizer-batcher
scheduling. No claim of tokenizer-cache causality is made.
