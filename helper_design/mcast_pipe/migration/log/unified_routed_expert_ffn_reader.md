# unified_routed_expert_ffn_reader.cpp — DEFERRED (design-gap)

Kernel:
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp`

Factory:
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp`

Tier: 2.16a. Status: deferred (design-gap). No production code change.

## API-v11 fit

The ordinary single-payload phases are expressible with existing API-v11 pipes:

- in0 is a fixed per-row, handshaked Flag channel with an outside sender;
- in1-down is a fixed per-column, handshaked Flag channel with an outside sender;
- activated data is a per-row rotating-sender, handshaked Flag channel with loopback.

Phase 1's in1 channel is not a single-payload transaction. With the active
`reader_mcasts_up` modes, one receiver-ready handshake protects two discontiguous L1 payloads
(`gate`, then `up`) and one final valid Flag. Both writes are linked so the final signal cannot
overtake either payload. Receivers acknowledge in0 and in1 up front, allow the two sender axes to
run concurrently, and wait for one in1 valid event only after both payloads land.

`SenderPipe::send()` owns exactly one data multicast plus its signal. The only separate public verb,
`send_signal()`, is signal-only and performs its own pre-handshake when enabled. Two handshaked
`send()` calls would require two sequential receiver acknowledgements and publish two valid events;
the current receivers provide one acknowledgement and expect one event. A second no-handshake pipe
would not protect the destination L1/CB lifetime across repeated K blocks. API v11 exposes no
data-only stage that can retain the reserved linked path until a later payload and signal.

## Generality gate

The required behavior is "multiple discontiguous data stages under one ACK and one final signal."
The audit found no second unrelated production family with that exact invariant. Adding a routed-FFN
batch/stage verb would therefore fail the plan's generality gate. The H2D/D2H service units need a
different capability (arbitrary GlobalSemaphore target addresses), so they do not supply a shared
extension case.

## Coverage

Routed-expert tests exist in:

- `tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_single_routed_expert.py`
- `tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_routed_expert_bias.py`
- `tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_swigluoai_routed_expert.py`

The unit stopped at the design gate before source edits, so correctness and performance were not
credited or run. The earlier ledger claim that this route is necessarily multi-device was stale.

## Claude consultation

The required architecture consultation ran for five minutes and timed out without a verdict. Silence
was not treated as approval. The plan-authorized API-v11 proof above independently requires deferral.

## Verdict

DEFER — DESIGN-GAP. Keep the kernel and factory raw. Do not expand the helper API.
