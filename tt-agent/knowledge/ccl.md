# CCL (Collective Communications)

Topic knowledge for multi-device collectives on tt-metal — external
references, configuration, numerical correctness, and tuning.

## External references

- **Ethernet and multi-chip guide** — `tech_reports/EthernetMultichip/BasicEthernetGuide.md`
  Ethernet link topology, bandwidth, latency. How to reason about
  multi-chip data movement.
- **CCL operations** — `ttnn/cpp/ttnn/operations/ccl/`
  All-gather, all-reduce, reduce-scatter implementations. Current
  patterns.
- **CCL tests** — `tests/ttnn/unit_tests/operations/ccl/`
  How CCL ops are set up and what parameters control topology.

## Configuration

### Set `num_links` to the physical inter-chip link count

`ttnn.experimental.all_gather_async(..., num_links=N, ...)` and siblings
require `N` to match the device topology's physical ethernet links:

| Part | num_links |
|---|---|
| N150 (1 chip) | N/A (no CCL) |
| N300 (2 chips, Ring) | 1 |
| Galaxy (32 chips, Ring) | 4 |
| TG (Tensor Galaxy) | per-axis; see the model's `tt_ccl.py` |

Wrong values **deadlock the device** (hang until watchdog) — not a soft
failure. Hardcoding Galaxy's value on N300 hung 10+ min before timeout,
requiring `tt_device_reset`.

Safest source: copy from a sibling model's `tt_ccl` module (e.g.
`tt_transformers/tt/ccl.py` auto-detects). Keep `tt_device_reset`
preloaded when tuning CCL params.

## Numerical

### Pre-divide cross-device bias by `num_devices`

When a matmul is followed by AllGather + FastReduceNC (or any sum-reduce
across devices), fusing bias via `ttnn.linear(bias=...)` adds it on each
device, and the cross-device sum multiplies it by `num_devices`.

```python
if args.num_devices > 1:
    bias_torch = state_dict[...] / args.num_devices
else:
    bias_torch = state_dict[...]
# Distinct cache filename (e.g. "bias_div") so the old cache isn't reloaded.
```

Same trick applies to any reduced intermediate followed by a bias/scale
add.

## Tuning

### Sweep CCL tuning knobs — effects are non-monotonic

`chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel` do
**not** improve monotonically. Observed on Gemma3 N300:

- `chunks_per_sync`: 10 → 4 improved -309μs; 4 → 2 regressed +976μs
- `num_workers_per_link`: 2 → 4 improved; 4 → 8 neutral (wasted cores)
- `num_buffers_per_channel`: 2 → 4 regressed +47μs

Small targeted sweep (3-4 values per knob) rather than a directional
tune. Record the full sweep in the caller's notes so future sessions
don't re-walk regressions.
