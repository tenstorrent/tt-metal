# DeepSeek V3 B1 Run-Inference Reference Traces

This directory contains JSON traces used by `test_run_inference.py` to replay
the host-side speculative decode logic without running the full device model.
New reference traces should use `schema_version: 2`, which represents every
MTP depth as a dynamic `tokens` window. MTP1 uses two token slots, MTP2 uses
three, MTP3 uses four, and MTP4 uses five.

The test loader still accepts older schema-version-1 traces so historical
fixtures can be replayed, but those fields are converted to the dynamic packet
shape before they reach `ModelPipeline.run_inference`.

## Trace Fields

`schema_version`
: Integer trace schema version. Use `2` for dynamic-depth MTP traces. Use a new
  value when changing packet shape or validation rules in a non-compatible way.

`name`
: Human-readable trace name. This is only for debugging and test failure output.

`params`
: Inputs passed to `ModelPipeline.run_inference`.

`params.prompt_token_ids`
: Prompt token IDs sent through prefill before speculative decode starts. These
  are model vocabulary token IDs, not byte offsets or text.

`params.max_new_tokens`
: Maximum number of generated tokens the host should emit before stopping.

`params.eos_token_id`
: Token ID that stops generation early when emitted. Use `null` when EOS should
  not be checked.

`params.num_speculative_tokens`
: Number of speculative tokens per host round. This is the MTP depth. MTP1 uses
  `1`, MTP2 uses `2`, MTP3 uses `3`, and MTP4 uses `4`. If omitted in current
  schema-1 traces, the test assumes `1`.

`params.relaxed_accept_topn`
: Optional number of target-model probability entries the host is allowed to
  consider for relaxed acceptance. The default is `10`, which is also the
  current metadata-page storage capacity. A smaller value lets a trace exercise
  behavior where a token is present in the packet but outside the active top-N
  acceptance window.

`params.relaxed_accept_delta`
: Optional relaxed-acceptance probability tolerance. The default is `0.6`. The
  host accepts a speculative token only when the token is within the active
  `relaxed_accept_topn` entries and its probability satisfies
  `prob(token) >= prob(top1) - relaxed_accept_delta`.

`device_to_host`
: Packets the fake model returns to the host. These packets represent what the
  real device pipeline would send after prefill and speculative decode steps.

`device_to_host.prefill_results`
: Output packets from the last prefill token. The host uses these to seed the
  first speculative window before it starts reading normal decode packets.

`device_to_host.read_results`
: Output packets returned by later `read_result()` calls during speculative
  decode.

`expected_host`
: Expected observable host behavior after replaying the trace.

`expected_host.generated_tokens`
: Exact token IDs that `run_inference(..., return_generated_tokens=True)` should
  return and that `on_token` should receive.

`expected_host.writes`
: Input packets the host is expected to write back to the device. These are
  checked in order.

`expected_host.read_count`
: Number of device-to-host packets the host is expected to consume through
  `read_result()`.

`metadata`
: Expected summary counters collected by the host.

`metadata.num_accepts`
: Number of speculative decisions accepted by the host.

`metadata.num_rejects`
: Number of speculative decisions rejected by the host.

## Synthetic Trace Suites

Synthetic corner-case coverage can be grouped in a suite file with this shape:

```json
{
  "suite_schema_version": 1,
  "name": "synthetic_mtp4_corner_cases",
  "traces": [
    {
      "schema_version": 2,
      "name": "synthetic_case_name",
      "params": {},
      "device_to_host": {},
      "expected_host": {},
      "metadata": {}
    }
  ]
}
```

Each entry in `traces` uses the same packet schema as a normal reference trace.
Successful synthetic traces use `expected_host` and `metadata`. Negative
synthetic traces replace those fields with:

```json
{
  "expected_error": {
    "type": "RuntimeError",
    "match": "message regex"
  }
}
```

`expected_error.type`
: Exception type the host is expected to raise while replaying the trace.
  Current tests support `RuntimeError` and `ValueError`.

`expected_error.match`
: Regular expression matched against the exception message. Use this to make
  malformed-packet traces assert the specific contract violation being tested,
  such as duplicate lanes or non-contiguous positions.

## Dynamic Device-To-Host Packet Fields

A dynamic MTP packet should use this shape:

```json
{
  "request_id": 0,
  "type": "BASE",
  "lane_idx": 0,
  "window_start_pos": 16,
  "num_window_tokens": 3,
  "tokens": [
    {"token_id": 201, "pos": 16},
    {"token_id": 2581, "pos": 17},
    {"token_id": 477, "pos": 18}
  ],
  "target_topn_tokens": [201, 17, 42],
  "target_topn_probs": [0.72, 0.31, 0.08]
}
```

`request_id`
: Logical decode slot or batch slot. Current tests use single-user decode, so
  this is always `0`. It is still included because the device metadata page has
  a slot field and future batched traces may use it.

`type`
: Packet kind. Allowed values are `PREFILL`, `BASE`, and `SPEC`.

`type = PREFILL`
: Packet comes from the last prefill output. It is allowed to start the first
  speculative window even though the host does not yet have a pending
  speculation to compare against.

`type = BASE`
: Packet came from the base lane of a speculative round. The host compares
  `tokens[0]` against the previously recorded speculation for `tokens[0].pos`.
  If the comparison accepts, the host may also commit speculative-lane output.
  If it rejects, this packet becomes the owner of the next window.

`type = SPEC`
: Packet came from a speculative lane. The host groups it with the other lanes
  in the same `window_start_pos` round. `lane_idx` identifies which speculative
  lane this packet belongs to in the current window.

`lane_idx`
: Lane number inside one speculative round. `0` is the base lane. `1` is the
  first speculative lane, `2` is the second speculative lane, and so on. For a
  depth-N trace, each complete round should contain lanes `0..N`.

`window_start_pos`
: Sequence position of `tokens[0]` in the base lane for this forward pass.
  This value is shared by every lane belonging to the same user and same
  speculative window. For lane 0, `window_start_pos` must equal
  `tokens[0].pos`. For speculative lanes, `tokens[0].pos` may be later than
  `window_start_pos`; the shared value still points back to the base lane's
  first token. This lets the host group lane packets into one decision round
  without relying on arrival order alone.

`num_window_tokens`
: Number of valid entries in `tokens`. It should be `num_speculative_tokens + 1`
  for a full speculative window because the packet contains one base candidate
  plus N speculative candidates. MTP1 uses `2`, MTP2 uses `3`, MTP3 uses `4`,
  and MTP4 uses `5`.

`tokens`
: Ordered candidate token slots produced by one lane. Entry `tokens[0]` is the
  token produced by normal LM-head sampling for this lane's input position.
  Entries `tokens[1:]` are the speculative continuation tokens predicted from
  that lane. The host uses these slots both for emitting committed tokens and
  for seeding the next speculative window.

`tokens[].token_id`
: Model vocabulary token ID for that candidate slot.

`tokens[].pos`
: Absolute sequence position for `token_id`. Positions are zero-based over the
  full prompt plus generated sequence. For example, if the prompt length is 15,
  the first generated token has position `15`.

`target_topn_tokens`
: Optional relaxed-acceptance candidate token IDs from the target/base model
  distribution for `tokens[0]`. This is the set of target tokens that the host
  may accept instead of requiring exact equality with the top-1 token. It
  should be ordered the same way as `target_topn_probs`, with the top-1 token
  at index 0. The host only considers the first `params.relaxed_accept_topn`
  entries, even when the packet stores more entries.

`target_topn_probs`
: Optional probabilities corresponding one-to-one with `target_topn_tokens`.
  These must be probabilities from the target/base model, not log
  probabilities. The host accepts a speculative token when the token is present
  within the active `params.relaxed_accept_topn` slice of `target_topn_tokens`
  and `target_topn_probs[token_index] >= target_topn_probs[0] -
  params.relaxed_accept_delta`.

## Expected Host Write Fields

`expected_host.writes[].token_id`
: Token ID the host writes back to the device for a new base or speculative
  lane.

`expected_host.writes[].type`
: Lane type for the write. `BASE` means lane 0. `SPEC` means a speculative lane.
  For dynamic MTP traces, include `lane_idx` as well so `SPEC` lane 1 can be
  distinguished from `SPEC` lane 2, 3, or 4.

`expected_host.writes[].pos`
: Absolute sequence position the written token should occupy.

`expected_host.writes[].request_id`
: Logical decode slot for the write. Current traces use `0`.

`expected_host.writes[].prefill_id`
: Next prompt token ID used during prefill metadata forwarding. Decode writes
  should use `-1`.

`expected_host.writes[].lane_idx`
: Dynamic MTP write lane. `0` is base, `1..4` are speculative lanes. Include
  this in schema-version-2 traces.

`expected_host.writes[].window_start_pos`
: Sequence position of the base lane's `tokens[0]` for the window this write
  belongs to. All writes that seed the same forward pass for a user should carry
  the same `window_start_pos`, even when the write is for a speculative lane.
  Include this in schema-version-2 traces.

`expected_host.writes[].num_window_tokens`
: Number of valid token slots in the window being written. Include this in
  schema-version-2 traces.

## Synthetic Error Traces

Synthetic traces can add an `expected_error` object when the correct behavior is
to reject malformed device output instead of generating tokens.

`expected_error.type`
: Exception type expected from the host replay.

`expected_error.match`
: Substring or regular expression that should appear in the exception message.
