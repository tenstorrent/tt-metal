# Qwen3.6-27B TP4 multichip decoder

This stage provides the layer-stack baseline in `tt/multichip_decoder.py`; it deliberately does not begin full-model or vLLM integration. `MultichipDecoder` derives from the completed `OptimizedDecoder` and targets exactly the four local Blackhole p300c devices as `MeshShape(1,4)`, `FABRIC_1D_RING`, TP=4.

## Final scheme

- Hidden-5120 residuals are replicated at layer boundaries. Column-parallel Q/K/V/gate, linear input, and MLP gate/up weights are split across four ranks. Attention/linear output and MLP down are row-parallel and followed by ring all-reduce.
- Decode weights are additionally width-sharded over each device's eight DRAM-bank columns; activations use eight-core L1 width sharding around those matmuls. Prefill keeps interleaved phase copies.
- Full attention owns six Q heads and one paged KV head/device. Linear attention owns four key and twelve value heads/device, with local convolution and recurrent state.
- Full attention preserves the optimized baseline's BF16 projection policy with BFP8 cache and BF16 activations/CCL. The faster BFP4/LoFi candidate was rejected by official-weight PCC (0.9870). Linear follows the optimized BFP4 policy. Qwen3.6-27B is dense, not MoE.
- Public sequence lengths need not be aligned. S5, S33, and the S32769 long-prefill tail are validated; padding/chunking remains internal.

Exact shapes, padding, DRAM shards, rejected mesh alternatives, and the rejected fractured-residual topology are in `mesh_plan.md`.

## Correctness and contracts

- Representative layers pass official-weight B1/B32 decode. Final BF16 full S32769 prefill→decode passes at PCC 0.99999997 with exact local caches.
- B32 full decode uses a reversed two-page-per-user table and heterogeneous positions 0–31; local cache shapes are `(64,1,64,256)` and head ownership is checked.
- A real linear→full stack passes direct device-tensor handoff for B32 decode and S5 prefill with replicated `[1,1,B,5120]` / `[1,B,S,5120]` logical boundaries.
- Trace capture forbids program-cache misses. Two warm passes close every signature; cache-reset identical replay has PCC 1.0 for every step in all final B1/B32 layer-kind artifacts.
- Runtime fallback hard-failure mode is enabled throughout. Watcher-clean B32 full and linear evidence uses `TT_METAL_WATCHER_DISABLE_ETH=1`; the retained first attempt documents the Blackhole Ethernet-watcher teardown crash, and `tt-smi -s` showed all devices healthy afterward.

## Final warmed trace performance

| Kind | Batch | Single chip | TP4 | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|
| full | 1 | 1.2712 ms | 0.5952 ms | 2.136× | 53.39% |
| full | 32 | 1.4387 ms | 0.7205 ms | 1.997× | 49.92% |
| linear | 1 | 1.6650 ms | 0.9020 ms | 1.846× | 46.15% |
| linear | 32 | 15.8290 ms | 4.4330 ms | 3.571× | 89.27% |

Final `tt-perf-report` text and CSV live under `artifacts/tracy/full_b32_bf16_final/` and `linear_b32_dram_sharded/`. Row collectives are explicit reduce-scatter plus all-gather. Linear B32 remains dominated by two ~434 μs recurrent state matmuls; the exact local geometry sweep proves the retained grid4×1/w4 candidate is fastest.

## Capacity

Physical tile overhead and current loader residency are included. The artifact decomposes 10,599,141,888 bytes/device of loader weights/RoPE, 572,522,496 bytes/device B32 linear state, and the measured 1,969,152-byte peak warmed-trace workspace delta:

- B1 C=262144 passes, preserving the advertised context.
- B32 C=82432 passes; the immediately adjacent C=82496 fails in a fresh process.

See `artifacts/capacity/capacity_b1.json`, `capacity_b32.json`, and `doc/context_contract.json`.

## Reproduction

```bash
python_env/bin/pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_multichip_decoder.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_traced_decode.py --kind full --batch 32 --steps 8 --forbid-program-cache-misses
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_stacked_decoder_smoke.py --mode decode --batch 32
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_stacked_decoder_smoke.py --mode prefill --batch 1 --sequence 5
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_capacity_probe.py --batch 32 --max-context 262144
```

The complete command/evidence chronology and artifact paths are in `work_log.md`.
