# Optimized decoder signoff artifacts

All commands used a single `MeshShape(1, 1)` Blackhole device and
`throw_exception_on_fallback=True`. Watcher and profiler runs were separate.

## Final console evidence

Official weights:

```text
full_attention_real_pcc.py --candidate default --batch 1
OPTIMIZED_POLICY ... sdpa_fidelity=HiFi4 qkv_fidelity=HiFi2
o_fidelity=HiFi2 qkv_w=2 o_w=3 gate_w=5 up_w=5 down_w=17
FULL_ATTENTION_REAL_WEIGHT_DECODE_PCC ... batch=1 0.9976120123684317

full_attention_real_pcc.py --candidate default --batch 32
FULL_ATTENTION_REAL_WEIGHT_DECODE_PCC ... batch=32 0.9980950192619897

linear_attention_real_pcc.py --optimized --candidate default
OPTIMIZED_POLICY ... linear_packed=True linear_outer=True
linear_recurrent=grid4_w4 linear_recurrent_fidelity=HiFi2
linear_recurrent_state_dtype=BFLOAT8_B
linear_input_dtype=BFLOAT4_B linear_input_fidelity=LoFi
linear_input_w=5 linear_output_dtype=BFLOAT4_B
linear_output_fidelity=LoFi linear_output_w=12
LINEAR_ATTENTION_REAL_WEIGHT_DECODE_PCC ... 0.9987165695638567
```

Final ten-step traced runs:

```text
full B1:  PCC 0.9990086830950573 .. 0.9999771193783431
          median_ms=1.268103 min_ms=1.267181
full B32: PCC 0.9995602638581764 .. 0.999979298820375
          median_ms=1.453556 min_ms=1.452093
linear B1:  PCC 0.9999867777338293 .. 0.9999971685351872
            median_ms=1.670179 min_ms=1.667905
linear B32: PCC 0.9999677269441009 .. 0.9999975994891279
            median_ms=15.949088 min_ms=15.925423
```

Watcher-enabled runs reran both B32 ten-step commands. Both returned zero;
all nine replay PCC checks passed; the watcher attached and detached all four
host devices with no watcher error, assert, or hang signature. Retained JSON
records `"watcher_enabled": true`.

Final prefill and capacity gates:

```text
full non-aligned prefill S33 B1/B32 PCC=0.999994
linear non-aligned prefill S5/S65 PCC=0.999996
full and linear fresh-run prefill/decode determinism: torch.equal=True
full optimized capacity-only S32769: passed, nonzero output/cache
full optimized capacity-only S192511:
  output_shape=(1, 192511, 5120)
  output_nonzero=985648951
  cache_nonzero=195301706
linear optimized default capacity-only S192511:
  output_shape=(1, 192511, 5120)
  output_nonzero=985656229
  recurrent_nonzero=129830
  recurrent_dtype=DataType.BFLOAT8_B
post-capacity full S33 B1/B32 PCC=0.999994
```

## Compact profiler signoff

Each directory contains `profile_run.json`, `perf.csv`, `summary.csv`, and
`summary.png`.

| Path | Meaning |
|---|---|
| `tracy/final4_full_decode_b1/` | final post-AutoFix full decode B1 |
| `tracy/final4_full_decode_b32/` | final post-AutoFix full decode B32 |
| `tracy/final4_linear_decode_b1/` | final BFP8-state, BFP4/LoFi-projection linear decode B1 |
| `tracy/final4_linear_decode_b32/` | final BFP8-state, BFP4/LoFi-projection linear decode B32 |
| `tracy/final4_full_prefill_b1/` | final full S33 prefill B1, five iterations |
| `tracy/final4_full_prefill_b32/` | final full S33 prefill B32, five iterations |
| `tracy/final4_linear_prefill_b1/` | final BFP8-state linear S5 prefill B1, three iterations |
| `tracy/final4_linear_prefill_b32/` | final BFP8-state linear S5 prefill B32, three iterations |

The four post-review decode profiler windows report, per replay:

| Window | Device time | DRAM roofline |
|---|---:|---:|
| full B1 | 1.200 ms | 55.9%, 286 GB/s |
| full B32 | 1.278 ms | 52.5%, 269 GB/s |
| linear B1 | 1.521 ms | 25.2%, 129 GB/s |
| linear B32 | 15.894 ms | 4.2%, 22 GB/s |

The four final prefill windows report, per iteration:

| Window | Device time | Wall median | DRAM roofline |
|---|---:|---:|---:|
| full S33 B1 | 3.054999 ms | 3.255676 ms | 22.8%, 117 GB/s |
| full S33 B32 | 16.310691 ms | 16.559895 ms | 8.9%, 46 GB/s |
| linear S5 B1 | 10.517650 ms | 11.086341 ms | 10.7%, 55 GB/s |
| linear S5 B32 | 275.038348 ms | 275.375946 ms | 5.0%, 26 GB/s |

The linear policy line records `linear_recurrent_state_dtype=BFLOAT8_B`,
`linear_input_dtype=BFLOAT4_B`, `linear_input_fidelity=LoFi`,
`linear_input_w=5`, `linear_output_dtype=BFLOAT4_B`,
`linear_output_fidelity=LoFi`, and `linear_output_w=12`.
Each measured iteration contains both `BFP8 => FP32` and `FP32 => BFP8`
typecasts. Their combined device time is 0.022020 ms/iteration at B1 and
0.685789 ms/iteration at B32.

Raw Tracy logs and generated reports were moved to the desktop trash after
these compact reports were generated. They are recoverable there and can be
regenerated with the commands in `work_log.md`.

## Candidate evidence

- `candidates/*.json`: runnable candidate outputs with command, exit status,
  resolved policy, correctness, latency, or exact failure.
- `candidate_matrix.csv`: B1/B32 dtype/fidelity, geometry, row/device latency,
  whole traced latency, correctness, and decision.
- `program_contracts.json`: exact TTNN API/validator blockers.
- `tracy/final_cum_*`: cumulative, one-role full-attention alternatives at
  both batches, including exact failure JSON for illegal candidates.
- `tracy/linear_recurrent_*` and `tracy/linear_state_*`: compact recurrent
  program and persistent-state precision candidate windows.
- `tracy/linear_{proj,input,output,both}_*`: compact independent projection
  precision controls and cumulative winner at B1/B32.

The recurrent-state sweep compared FP32, BF16, BFP8, and BFP4 at both
batches. BFP8 is the fastest candidate that passes the real-weight
prefill-to-decode transition; BFP4 reached only 0.993340 minimum real-weight
PCC and was rejected. The selected BFP8 state also passes long transition,
fresh-run determinism, watcher stress, and the retained S=192511 capacity
artifact.
The projection sweep independently compared input and output weight dtype and
fidelity from BF16/HiFi2 through BFP8 and BFP4. Cumulative BFP4/LoFi is the
fastest passing B1 candidate and also improves B32. Its official transition
minimum PCC is 0.997175.

The precision-locked geometry sweep then compared packed-input widths
1/4/5/10/20, output widths 1/2/3/4/6/8/12/24, cumulative width crosses, and
a four-core storage control at B1/B32. Width 5/12 is selected at
1.670349/15.942844 ms candidate trace and 1.521726/15.890707 ms device time.
Input widths 10/20 and the four-core control retain exact B1/B32 failure JSON.
Every passing contender has paired compact profiler evidence. Output subblock
is not exposed by the DRAM-sharded program API; the internally selected 1x8
input and 1x7 output subblocks and exact source contracts are recorded in
`program_contracts.json`.
The promoted default's real S=65 plus four-step transition has minimum PCC
0.997167 and records `"real_weights": true`.

Final promoted-policy artifacts are
`candidates/final6_linear_{real_b1,transition_real_b1,traced_b1,traced_b32,watcher_b32}.json`
and
`candidates/final6_linear_{decode,prefill}_determinism_b32.json`.
The watcher artifact records all nine PCC checks passing with watcher enabled;
both determinism artifacts record `"bit_exact": true`.
Final static gates passed: `py_compile`, 165 optimized-path pytest cases, CSV
schema validation for all 114 candidate rows, and JSON parsing for all seven
final6 artifacts.
