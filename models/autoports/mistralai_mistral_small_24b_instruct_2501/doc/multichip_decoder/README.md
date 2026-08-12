# Mistral-Small-24B multichip decoder

Status: TP4 decoder implementation and evidence complete on 2026-07-17. This stage does not claim full-model or vLLM readiness.

## Result

`tt/multichip_decoder.py` is a real `MultichipDecoder(OptimizedDecoder)` implementation specialized for the four local Blackhole p300c devices as a logical `1x4` `FABRIC_1D` mesh. Q/K/V heads and MLP channels are column parallel; output and down projections are row parallel; two Linear, two-link all-reduces restore replicated hidden state per layer. Inputs and outputs remain replicated `[1,batch,logical_seq,5120]`, while K/V are head-sharded with two local KV heads per rank.

The path preserves the optimized decoder's BFP4/LoFi weights, BF16 activations, BFP8 cache, DRAM-sharded decode weights, L1-sharded decode activations, compiler-advised 11-core RMSNorm layout, and bounded prefill MLP chunks. The final decode geometry is attention `(10,12,16,8,10,4)` and MLP `(10,32,40,16,16)`; its three-run real-weight median is 4.934% faster than the initial geometry. Mistral-Small-24B is dense, so no MoE/expert policy applies.

Contiguous and paged caches, replicated page tables, and arbitrary valid logical sequence lengths are supported. One persistent replicated INT32 position vector drives RoPE lookup, cache writes, and SDPA. It can contain different positions per user and can be updated between trace replays without recapture.

The pre-code mesh decision, every per-rank tensor shape, tile-aware memory calculation, TP-local sweep, and rejected alternative are in [mesh_plan.md](mesh_plan.md).

## Correctness and contracts

All layer tests use batch 32 and representative decoder layer 20, the model's only meaningful layer kind.

| Gate | Result |
| --- | --- |
| HF reference, contiguous prefill | PCC 0.995336 at seq 17, 0.995353 at seq 18, 0.995555 at seq 32 |
| HF reference, contiguous decode | PCC 0.995057 at current position 32 on the final geometry |
| Real-weight optimized TP1 TTNN baseline, prefill/decode | PCC 0.999994 / 0.999990 |
| Real-weight optimized TP1 TTNN baseline, K/V | PCC 1.0 / 1.0 |
| Matched paged versus contiguous control | prefill output, logical K/V, and decode output PCC 1.0 |
| Paged cache versus HF | prefill/decode PCC 0.995356 / 0.995011; physical K/V 0.993746 / 0.993669 |
| Mutable nonuniform trace positions | contiguous and reversed-page-table modes pass three changing position vectors; output/K/V PCC >= 0.99; repeated replay bitwise deterministic |
| Warmed decode trace | capture/replay passes; output and current-position K/V writes are bitwise stable across 50 timed replays |
| Full advertised cache envelope | 40 decode-weight layers, 40 local K/V pairs, TP embedding/head, shared RoPE, page table, positions, and physical 4 GiB/rank reserve resident; decode passes at position 32767 |
| Runtime fallback audit | owned prefill/decode/MLP runtime; no host conversions or `OptimizedDecoder.*forward` calls |

Logical lengths 17 and 18 prove that tile alignment is not a public restriction. Paged testing reverses logical-to-physical block order and checks the exact physical block/offset written. All output ranks are checked for equality. The optimized-baseline artifact is produced by a single-device process and consumed by a fresh 1x4 process, preventing shared implementation state.

## Capacity

At context 32,768, the complete per-rank steady state is `26,674,528,384` bytes: one decode matrix representation for 40 layers, tiled norms, TP BF16 embedding and untied LM head, shared prefill/decode RoPE, shared indices, page table, positions, and all 40 BFP8 K/V pairs. This leaves `7,504,202,624` bytes on a `34,178,731,008`-byte device. The capacity test physically reserves another 4 GiB and still decodes at position 32,767, leaving a calculated `3,209,235,328`-byte margin.

The context-scaled lifetime is `697,352` bytes/token, yielding a block-aligned calculated ceiling of 37,344 tokens with the 4 GiB reserve. The advertised 32,768-token cache contract is therefore retained. Interleaved prefill copies are released before the long-lived decode phase, and immutable RoPE tables are shared across layers.

## Performance and profiler

Warmed wall latency uses real layer-20 checkpoint weights, one warmed prefill, four trace warmups, and 50 decode trace replays. It excludes construction, weight loading, compile, capture, host copies, and determinism reads.

| Path | Optimized TP1 | TP4 final | Speedup | TP4 efficiency |
| --- | ---: | ---: | ---: | ---: |
| prefill, batch 32 × seq 18 | 5.399418 ms | 3.717828 ms | 1.452304× | 36.308% |
| decode, batch 32 × 1 | 1.288697 ms | 0.579850 ms | 2.222466× | 55.562% |

The separate geometry acceptance used three fresh 50-replay processes per finalist. The original decode median was 0.609677 ms and the selected median was 0.579591 ms; all selected outputs had PCC at least 0.999989 versus the original.

The final `tt-perf-report` decode capture contains 42 representative device ops, zero host ops, and 575 us of device work. Matmuls total 265.56 us; the two all-reduces appear as reduce-scatter/all-gather components totaling 83.08 us; explicit interleaved/sharded/cache/reshard data movement totals about 27.57 us. The modeled overall DRAM rate is 121 GB/s (23.6% roofline). Representative gate/up/down matmuls are 77/77/80 us at 263–274 GB/s; QKV is 18 us, output projection 14 us, SDPA 13 us, and norms total 17 us.

The merged report attributes a 96.6 ms cross-device row-order gap to a reshape. This is a merge-order artifact: the per-device report contains the same ~575 us kernel chain, the profile has zero host ops, and warmed replay is 0.579850 ms. Both merged and per-device readable/CSV tables plus the complete compressed op stream are retained under `tracy/final`.

Linear beat Ring for the real 327,680-byte BF16 collective payload (0.114246 versus 0.117872 ms). The full consumer-chain comparison also rejected a hidden-sharded residual: replicated `all_reduce → RMSNorm → QKV` was 0.102029 ms versus 0.155569 ms for `reduce_scatter → distributed RMSNorm → all_gather → QKV`.

## Reproduce

Run from the repository root:

```bash
MODEL_DIR=models/autoports/mistralai_mistral_small_24b_instruct_2501
SNAPSHOT=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724

tt-smi -r
pytest -q -s "$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_synthetic_prefill_decode_pcc_layout_and_trace"

tt-smi -r
pytest -q -s "$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_paged_cache_page_table_and_positions"

tt-smi -r
pytest -q -s "$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_mutable_nonuniform_positions_trace"

tt-smi -r
MISTRAL_SMALL_24B_MULTICHIP_CAPACITY=1 pytest -q -s \
  "$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_full_context_paged_cache_capacity"

tt-smi -r
MISTRAL_SMALL_24B_REAL_WEIGHT_DIR="$SNAPSHOT" \
MISTRAL_SMALL_24B_MULTICHIP_PERF_IMPL=multichip \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=50 pytest -q -s \
  "$MODEL_DIR/tests/test_multichip_decoder.py::test_warmed_single_chip_and_multichip_perf"

tt-smi -r
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_LOGGER_LEVEL=info pytest -q -s \
  "$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_synthetic_prefill_decode_pcc_layout_and_trace"
```

The optimized timing command changes `MISTRAL_SMALL_24B_MULTICHIP_PERF_IMPL` to `optimized`. The two-process baseline and profiler commands are recorded verbatim in [work_log.md](work_log.md).

## Limitations and handoff contract

- Only the target 1x4 mesh is supported. Smaller or differently shaped meshes are rejected.
- Trace shapes are fixed, but the device-resident per-user position vector is mutable across replays.
- A batch-32 full-width BF16 residual at 32K is 10,737,418,240 bytes per rank. Full-context prefill must stream/chunk residual work; this stage validates the cache/decode handoff, not a monolithic 32K prefill.
- Full active-Ethernet watcher instrumentation is not stable on this host's firmware 19.8.0: default inlining exceeds the 25 KiB region, while noinline passes the test but aborts during active-ETH base-firmware restoration. The final worker watcher therefore uses `TT_METAL_WATCHER_DISABLE_ETH=1`; fabric and CCL still run, all worker cores are checked, and the process exits 0. Failed full-ETH evidence is retained.
- `/dev/shm` has only about 17 MiB free, so MPI warns and falls back. The inspector also cannot replace an unrelated generated log directory because of its existing permissions. Neither warning affected the final gates.
