# Qwen3.6-27B optimized TP4 multichip decoder

This stage optimizes the completed decoder in place on four local Blackhole
p300c devices, `MeshShape(1,4)`, `FABRIC_1D_RING`. It does not contain full-model
or vLLM work. The final default retains the replicated BF16 hidden-5120 layer
boundary and the inherited packed projections, DRAM-sharded decode weights,
precision policy, and ring row reductions. No collective, gather, or reshard
occurs between decoder layer calls.

## Final results

All decode figures are warmed trace replay with identical inputs, eight steps,
fallback hard-failure, deterministic state/cache progression, and the real TP4
path. Prefill is warmed device execution at logical S128/B1.

| Kind | Batch | before decode ms | final default ms | delta | PCC |
|---|---:|---:|---:|---:|---:|
| full attention | 1 | 0.595597 | 0.593794 | -0.30% | 1.0 |
| full attention | 32 | 0.722288 | 0.722050 | -0.03% | 1.0 |
| linear attention | 1 | 0.900798 | 0.899718 | -0.12% | 1.0 |
| linear attention | 32 | 4.433326 | 4.431181 | -0.05% | 1.0 |

| Kind | before S128 prefill ms | final S128 prefill ms | PCC |
|---|---:|---:|---:|
| full attention | 2.102029 | 2.081797 | 0.99999460 |
| linear attention | 80.039188 | 80.580643 | 0.99999433 |

The linear-prefill movement is unchanged by the decode-only optimization; its
0.68% run-to-run increase is reported rather than hidden. Pinned official
weights pass the accepted gates: full PCC 0.999741/0.999671 and linear PCC
0.999906/0.999906 at B1/B32, including per-rank cache ownership.

## Optimization decisions

- Coherent fractured residual across a traced linear→full stack removed the
  inter-layer collective but was 3.82% slower (5.332280 vs 5.136247 ms), so the
  replicated layer contract remains selected.
- Packed MLP gate/up was 4.15% slower in full B32 and 0.42% slower in linear
  B32. Separate projections remain selected.
- BFP8 attention/MLP CCL payloads preserved PCC but were slower because of the
  typecasts. BF16 CCL remains selected.
- Preallocated synchronous CCL buffers appeared to win isolated A/Bs by
  0.13–0.42%, but the final full-B32 rerun was neutral/slower and full-stack
  residency reduced proven B32 context. They are rejected; the restored final
  default above preserves capacity.
- Shape-faithful async matmul→RS passes Watcher and PCC. Fused matmul→RS needed
  an AutoFix atomic drain, then passed; across 8x4/8x6 and block widths
  1/2/3/4/6 it remained 9.7–11.1% slower than separate async.
- TP4 fused all-gather→matmul was adapted through API, semaphore, layout, and
  transfer-count attempts. Three focused Watcher retries still hit the receiver
  ledger assertion; `$autofix` failed and reverted its speculative source fix.
- `tt-perf-report` advised L1 prefill inputs. The complete TP4 candidate
  regressed full S128 to 2.195965 ms and linear to 81.005829 ms, so interleaved
  prefill inputs remain selected. Its fidelity advice was already covered by
  the inherited precision sweep and pinned official-weight rejection of BFP4
  full projections (PCC 0.987017).

The inherited selected projection geometry remains DRAM-sharded decode weights
with L1 width-sharded activations. Its precision-locked geometry sweeps cover
all dominant projections and recurrent matmuls; the latter retained 4x1/w4 at
0.452824 ms versus 0.467813 ms next-best and 0.638784 ms automatic.

## Correctness, context, and artifacts

Valid logical S5, S33, and S32769 prefill paths pass; padding, masking, cache
fill, and slicing remain internal. The final S32769 prefill→decode crosses the
32768 chunk boundary. B1 still physically fits context 262144. The final hard
B32 bracket remains C82432 pass / C82496 fail. The rejected persistent
candidate's C82240/C82304 bracket is retained as evidence, not public contract.

Final Tracy + `tt-perf-report` human output, CSV, summary, raw report, and run
provenance live under `artifacts/tracy/{full_b32_restored_final,linear_b32_restored_final,
full_prefill_s128_final,linear_prefill_s5_final}/`. Decode profiles report
modeled DRAM rooflines of 143 GB/s full and 20 GB/s linear. Final 64-step B32
full/linear decode and 16-iteration S128 full/linear prefill stresses are
Watcher-clean with Ethernet watching disabled for the firmware config-buffer
limitation. Linear stress initially exposed and AutoFix repaired a slice CB
prefix-capacity bug; focused and existing slice controls pass. Runtime fallback
audit is clean.

Exact commands, every candidate, failed/adapted attempt, artifact paths, and
the anomaly ledger are in `work_log.md`.
