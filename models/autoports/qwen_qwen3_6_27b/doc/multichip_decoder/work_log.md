# Qwen3.6-27B multichip decoder work log

## 2026-08-13 — target selection before final-path coding

- Baseline: `tt/optimized_decoder.py` at `078fd756719`; representative linear-attention layer 0 and full-attention layer 3 are the required single-chip TTNN controls.
- Hardware discovery: `timeout 60 tt-smi -ls --local` reported four local Blackhole p300c devices (IDs 0–3). A `MeshShape(1, 4)` open/close smoke completed with `MESH_SMOKE_OK` and device IDs `[3, 2, 1, 0]`.
- Selected target: Blackhole p300c `MeshShape(1, 4)`, TP=4, 1-D ring fabric and Ring collectives. The implementation intentionally need not support other meshes.
- Model contract read from pinned HF revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`: hidden 5120, intermediate 17408, 24 Q heads, 4 KV heads, head dim 256, 16 linear key heads, 48 linear value heads, linear head dims 128, 64 layers (48 linear / 16 full), advertised context 262144.
- No MoE/expert plan is needed: Qwen3.6-27B has a dense gated MLP and hybrid linear/full attention, not routed experts.

The detailed calculated tensor, cache, collective, padding, and rejected-alternative plan is in `mesh_plan.md`. This entry and that plan precede creation of the final runtime path.

### Hardware commands

```bash
timeout 60 tt-smi -ls --local
timeout 60 python_env/bin/python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
print("DEVICE_IDS", mesh.get_device_ids())
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

### Scope guard

Only `tt/multichip_decoder.py`, multichip decoder tests, `doc/context_contract.json`, and `doc/multichip_decoder/**` are stage-owned. Full-model and vLLM work is not started.

## AutoFix: paged cache update layout

- Original command: `timeout 300 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_full_attention_smoke.py`
- Failure: `TT_FATAL Expect input_tensor to be sharded` at `paged_update_cache`.
- Fresh source-only diagnosis: `AUTODEBUG.md`.
- Verified hypothesis: the TP4 loader and local-head creation had replaced the optimized baseline's batch-height-sharded decode-attention layout with DRAM interleaved tensors. The cache op requires HEIGHT sharding, row-major orientation, full 256-wide head shards, and one core per user.
- Focused fix: derive the same workload-dependent batch grid as the optimized baseline and use it for local Q/K/V head creation. No precision, cache ownership, page-table, or TP mapping changed.
- Verification rerun passed both paged K and V updates and reached paged SDPA. It exposed a second omitted baseline layout restoration at `nlp_concat_heads_decode`; the SDPA result must be converted from its required DRAM output back to the same batch-height-sharded layout before concat.
- Final focused rerun exited zero: PCC 1.0 against serialized optimized single-chip TTNN, all replicas agreed, local cache shapes were `(1,1,64,256)` on each chip, and fallback hard-failure mode was enabled. See `AUTOFIX.md`.

## Correctness and trace closure

- Full-attention eager decode B1: PCC 1.0 against `OptimizedDecoder`; four replicas agree.
- Full-attention non-aligned prefill S33: PCC 0.9999944769002891.
- Linear-attention eager decode B1: PCC 1.0.
- Linear-attention non-aligned prefill S5: PCC 0.999994255064567.
- The inherited long-prefill implementation was invalid for TP because it used global Q/KV widths and omitted the row-parallel output reduction. The multichip override now uses local Q=1536, KV=256, six Q heads and one KV head, pads only the internal chunk tail, and all-reduces the local O projection.
- Long full-attention prefill S32769 passed at PCC 0.9999947405172742. This crosses the 32768 chunk boundary with a one-token logical tail; all device outputs agree and every local cache head matches the corresponding optimized-baseline KV head. Artifact: `logs/full_prefill_s32769.log`.
- B32 full decode passed at PCC 1.0 with a reversed two-page-per-user table and heterogeneous current positions 0–31. Each device owns cache shape `(64,1,64,256)` and local K/V contents match the corresponding one of four optimized-baseline KV heads.

### Trace AutoFix

- Original full TP4 trace capture failed four times with `Writes are not supported during trace capture` and left capture teardown live. Triage artifacts are under `triage/`.
- Synchronizing cache-restoration writes was a focused but refuted hypothesis: the exact failure remained.
- With `mesh.set_program_cache_misses_allowed(False)`, capture named `MatmulDeviceOperation` as a program-cache miss. Warming the exact full path immediately before capture, restoring cache state, and then capturing proved and fixed the cause. The synchronized cache restore remains required ordering hygiene; teardown releases only successfully ended traces.
- All final trace runs forbid cache misses during capture and hard-fail on runtime fallback.

### Like-for-like traced latency

Both sides use warmed trace replay with blocking execution, eight steps, identical synthetic weights/tokens, page tables, and cache progression. JSON artifacts are in `artifacts/`.

| Layer kind | Batch | Single chip ms | TP4 ms | Speedup | Efficiency | PCC |
|---|---:|---:|---:|---:|---:|---:|
| full | 1 | 1.272827 | 0.773775 | 1.64496× | 41.12% | 1.0 all seven compared steps |
| linear | 1 | 1.664927 | 1.074190 | 1.54994× | 38.75% | 1.0 all seven compared steps |
| full | 32 | 1.438492 | 0.903782 | 1.59164× | 39.79% | 1.0 all seven compared steps |
| linear | 32 | 15.829201 | 4.604416 | 3.43783× | 85.95% | 1.0 all seven compared steps |

Commands:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_traced_decode.py \
  --kind <full|linear> --batch <1|32> --steps 8 --forbid-program-cache-misses \
  --result-json models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/artifacts/<kind>_trace_b<batch>.json
```

## Final specialization and acceptance evidence

The table above records the first replicated-boundary implementation. Profiling then identified interleaved static projection weights as the main avoidable decode cost. The final implementation stages semantic TP shards and reshares every decode projection over the eight local DRAM-bank columns, with L1-width-sharded activations and local block widths 4/3/4/17/5 as recorded in `mesh_plan.md`. Prefill retains interleaved weights. This change preserved all PCC/cache gates and improved every traced whole-layer case:

| Layer kind | Batch | Single chip ms | final TP4 ms | Speedup | Efficiency | deterministic replay |
|---|---:|---:|---:|---:|---:|---:|
| full | 1 | 1.271224 | 0.595245 | 2.13563× | 53.39% | PCC 1.0, 8/8 |
| full | 32 | 1.438653 | 0.720534 | 1.99665× | 49.92% | PCC 1.0, 8/8 |
| linear | 1 | 1.664995 | 0.902041 | 1.84581× | 46.15% | PCC 1.0, 8/8 |
| linear | 32 | 15.829027 | 4.432966 | 3.57075× | 89.27% | PCC 1.0, 8/8 |

Final JSON: `artifacts/full_b1_final.json`, `full_b32_final.json`, `linear_b1_final.json`, and `linear_b32_final.json`. Corresponding complete logs are under `logs/*_trace_*_final.log`.

Pinned official weights reject the synthetic BFP4 full-attention promotion: PCC 0.987017. The final BF16 optimized-baseline policy passes official B1/B32 full PCC 0.999741/0.999671 and linear PCC 0.999906/0.999906, including local cache ownership. Evidence: `official_weights_b1.json`, `official_weights_b32.json` and logs.

The rejected candidate is independently preserved in `official_weights_bfp4_rejected_b1.json`: requested/effective `geometry_w4`, BFP4 projections, output PCC 0.9870167, `output_pass=false`. Final-BF16 stacked B32 decode and non-aligned S5 prefill were rerun successfully. Final BF16 S32769 prefill→decode passes PCC 0.9999999658 and all local K/V cache gates (`logs/full_prefill_decode_s32769_bf16_final.log`).

### Stacked layout

`multichip_stacked_decoder_smoke.py` passes the linear-layer device output directly into a full-attention layer without host materialization. B32 decode and non-aligned S5 prefill both pass output PCC, replica equality, local convolution/recurrent cache ownership, local paged K/V shapes, page tables, and current positions. Artifacts: `logs/stacked_decode_b32.log`, `logs/stacked_prefill_s5.log`.

### Residual topology decision

The coherent fractured candidate uses `reduce_scatter → distributed RMSNorm/stat all-gather → delayed hidden gather`. At exact B32/H5120 it measured 0.293329 ms versus 0.304505 ms for the isolated replicated family (3.7% faster), with rank PCC 0.999822–0.999828. Integrated into the real full layer it measured 0.855622 ms versus 0.773775 ms replicated (10.6% slower). It was therefore rejected; the final stack boundary is replicated hidden 5120. Artifact: `artifacts/residual_topology_b32.json`.

### Capacity bracket

The probe reserves an explicit per-object-family loader sum of 10,599,141,888 bytes/device, 572,522,496 bytes/device B32 linear state, and the measured 1,969,152-byte peak warmed-trace allocator delta. It creates 32 real-shaped BFP8 paged cache objects in fresh subprocesses.

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_capacity_probe.py --batch 1 --max-context 262144 --require-max-pass
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_capacity_probe.py --batch 32 --max-context 262144
```

B1 C=262144 passes. B32 C=82432 passes and adjacent C=82496 fails. Evidence: capacity JSON and isolated worker artifacts.

### Profiling

Final profiles were captured with Tracy only after watcher was disabled, then rendered with `tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-color --csv ...`. Human tables, CSV, summary, raw provenance, and trace result are under:

- `artifacts/tracy/full_b32_bf16_final/`
- `artifacts/tracy/linear_b32_dram_sharded/`

The full profile reports 43–67 μs DRAM-sharded projections and explicit reduce-scatter/all-gather collectives. The linear profile identifies the remaining two ~434 μs recurrent state matmuls plus slice/untilize data movement as the dominant costs; static projections are no longer the primary bottleneck.

AutoFix swept the exact TP-local B32 recurrent shape across automatic selection, grids 4×1/2×1/1×1/2×2, K block widths 1/2/4, N subblocks 1/2/4, and DRAM residency. All candidates passed PCC ≥0.99998. The incumbent `grid4x1_w4_n1_s1` is the measured winner at 0.452824 ms (next best 2×2 w4: 0.467813 ms; auto: 0.638784 ms). It is retained. The profiler `SLOW` classification reflects the inherently small N=128 local problem; it already scales almost ideally from the prior ~1.717 ms single-chip recurrent row. Artifact: `artifacts/recurrent_geometry_dram_b32.json`.

### Watcher, fallback, and health

`ttnn.CONFIG.throw_exception_on_fallback = True` is set in every hardware smoke/trace. Initial fabric+watcher startup with Ethernet watching enabled crashed in Ethernet-kernel teardown; `tt-smi -s` immediately showed all four boards healthy. Per the device workflow, watcher was rerun separately with `TT_METAL_WATCHER_DISABLE_ETH=1`:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_full_attention_smoke.py --mode decode --batch 32
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_traced_decode.py --kind linear --batch 32 --steps 4 --forbid-program-cache-misses
```

Both final logs show watcher attach/check/detach without watcher error and correct numerical output: `logs/watcher_full_b32_clean.log`, `logs/watcher_linear_b32_clean.log`. The failed environmental attempt is retained as `logs/watcher_full_b32_16steps.log`.

### Static tests

`python_env/bin/pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_multichip_decoder.py` → 9 passed. Runtime source is fallback-clean, uses the optimized decoder baseline, and contains no full-model or vLLM work.
