# Exact typecast with fused final DRAM placement

`tests/v6_projection_cast_placement_probe.py` tests a minimal alternative after direct BF16 reduction packing failed the strict midpoint gate. Source SHA256: `58182574be57b313d6ae236001465e41b45fbdc50b0907397240f37c6f78f6eb`.

The baseline is the complete `linear` method from standalone streaming integration snapshot `700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05`. For router and attention-out only, under the existing `terminal_scan` guard, the candidate replaces:

```python
return ttnn.to_memory_config(ttnn.reshape(ttnn.typecast(result, dtype), (1, 1, width)), DRAM)
```

with:

```python
return ttnn.reshape(ttnn.typecast(result, dtype, memory_config=DRAM), (1, 1, width))
```

The native FP32 block reduction, output FP32 tensor, explicit typecast SFPU rounding, and requested dtype remain unchanged. The typecast writer performs the final DRAM placement, eliminating the separate placement copy. Packed attention/QZ paths have no such final copy and are unchanged. All false/default prefill paths and full-attention V5 guards remain unchanged. Removing the two added guarded returns restores the complete baseline method AST exactly.

`typecast.cpp` explicitly accepts output memory configuration while retaining the same FP32-preserving compute path. This source evidence supports testing the change; it is not device equivalence or performance evidence.

## CPU/source setup

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_projection_cast_placement_probe.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_projection_cast_placement_cpu_setup.json
```

Result: `CPU_SOURCE_SETUP_PASS 32 256 rounding lanes`. Python compilation passes. No TTNN import or device execution occurred. The JSON freezes source/dependency/capture/fixture hashes, and sibling files freeze complete baseline/candidate/setup source plus actual layer34 weights/inputs. The same actual positions227/738/5799, random/zero cases, K512/K1024 components, and three-row non-aligned prefill inputs from the validated streaming probe are retained.

The 256-lane rounding fixture includes both signs and FP32 values immediately below, at, and above BF16 midpoints, plus zeros and exact values. Its strict bit-equality gate is retained. The variant saves native FP32 reduction and both BF16 outputs to `.rounding.pt` before asserting, improving failure diagnosis without changing acceptance.

## Parent-owned run

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_projection_cast_placement_probe.py \
  --run-device --timing \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_projection_cast_placement_device.json
```

Optional `--case-filter actual_227/router` narrows an initial smoke while still running the rounding fixture. Full execution compares complete FP32/BF16 adapter outputs and output memory placement for all32 cases. The runtime audit rejects Torch arithmetic and host/device conversion inside projection calls. Timing covers the complete router/out adapters, including streaming projection and reduction: three warmups, five synchronized eager calls, ten groups of ten trace replays, and exact outputs after 100 replays.

Only CPU/source checks have been performed by this subagent. Device equality, timing, watcher cleanliness, and full-decoder selection remain parent-owned. No speedup or integration claim is made. The rejected direct-pack source/failure artifacts remain unchanged; see `v6_projection_reduce_pack_probe.md` for diagnosis and additional untried exact-order fusion opportunities.

Parent follow-up: `v6_projection_cast_placement_device.json` passed all32 fixtures and the unchanged256-lane midpoint gate, with a clean device close. Complete traced router changed from0.050296215340 ms to0.048975250684 ms; attention-out from0.148276705295 ms to0.146979698911 ms. This makes the placement-only change eligible for an integration experiment. No whole-decoder result is implied, and the failed direct-BF16-pack variant remains rejected.
