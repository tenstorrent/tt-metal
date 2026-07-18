# Final multichip profiler interpretation

The final Tracy run uses the selected BF16 persistent-collective, O8 path on a
1x4 Blackhole p300c Ring. Watcher is deliberately disabled during profiling.

Raw CSV:
`tracy_final_o8/reports/2026_07_18_02_10_04/ops_perf_results_2026_07_18_02_10_04.csv`

Raw CSV SHA256:
`446de16e5ee09185cd4d0824ac25a95a4f2cff9ea256bfbf0a10457924cc5714`

`results/profile_run.json` records the final source hashes, mesh plan,
checkpoint/activation provenance, and 0.664121 ms profiled wall time per replay.
The longer unprofiled result is 0.629461 ms across nine 200-replay trials.

## Decode

`tt-perf-report` merges four devices and reports 174 device ops across three
trace replays, totaling 1,662.285 us, or 0.554095 ms device-op time/replay. The
single 7,147 us gap on the first merged `ReshapeView` is cross-device lane skew
between the signpost and the first selected op, not replay latency: the entire
three-replay wall measurement is about 1.992 ms. Excluding that one merge
artifact leaves 307.704 us of recurring gaps, or 0.102568 ms/replay, which
reconciles with the profiled wall result.

One replay's dominant rows are:

| Operation | Device time | Observed DRAM | Tool reference |
|---|---:|---:|---:|
| QKV `32x5120x2560`, local BFP4/LoFi | 29 us | 226-227 GB/s | 44% |
| O `32x2048x5120`, local BFP4/LoFi | 23 us | 226-227 GB/s | 44% |
| Gate/up, each `32x5120x6400` | 63 us | 261-262 GB/s | 50-51% |
| Down `32x6400x5120` | 62 us | 265 GB/s | 51-52% |
| Two Ring all-gathers | about 11 + 12 us | CCL | |
| Two Ring reduce-scatters | about 25 + 20 us | CCL | |
| Local BFP8-cache SDPA | 21 us | cache/compute | |

Stacked across three replays:

| Category | Time | Share of device-op time |
|---|---:|---:|
| Five projection matmuls | 717.73 us | 43.18% |
| Reduce-scatter | 136.67 us | 8.22% |
| All-gather | 67.75 us | 4.08% |
| SDPA decode | 62.33 us | 3.75% |
| Explicit reshard | 8.49 us | 0.51% |
| Remaining norm/cache/head/layout/add work | 669.32 us | 40.26% |

The four collectives total 204.42 us over three replays, about 68.14 us/replay.
Communication is material but not dominant. Matmul is the largest remaining
device opportunity. The report's overall modeled DRAM roofline is 21.5%
(110 GB/s) because it includes non-matmul work; the projection rows themselves
reach 226-265 GB/s. There are no host ops inside the signposted decode region.

## Prefill

The signposted real length-17 prefill has 221 merged device ops and 1,970.802 us
device-op time. Its 2,599.639 us merged gap sum contains cross-device scheduling
and serialized host submission of user-wise cache fills, so it cannot be added
naively to device time. The authoritative warmed wall median is 3.127879 ms.

| Operation | Device time | Observed DRAM | Tool reference |
|---|---:|---:|---:|
| QKV `544x5120x2560` | 73 us | 203 GB/s | 40% |
| O `544x2048x5120` | 68 us | 192 GB/s | 38% |
| Gate/up | 160 / 162 us | 179-180 GB/s | 35% |
| Down | 160 us | 181 GB/s | 35% |
| First / second Ring all-gather | 84 / 81 us | CCL | |
| O / down Ring reduce-scatter | 87 / 87 us | CCL | |
| SDPA | 44 us | local Q/K/V | |

Projection time is 623.05 us and CCL time is 338.15 us. The table also exposes
64 `slice -> clone(BFP8) -> cache update` paths: K and V for each of 32 users.
The clone is required because exact tile-aligned slices may alias their parent;
materializing them prevents one user's release from invalidating later cache
fills. No batched TTNN paged-fill API is available for this layout. There are no
host fallback ops inside the prefill signposts.

The overall modeled prefill DRAM roofline is 11.4% (58 GB/s). Prefill keeps
interleaved weights and dynamic logical sequence/page-table handling. For long
contexts, O/down reduce-scatter each bounded row chunk before concatenating the
local result, which limits peak DRAM without changing the public contract.

## Advice disposition

The final report was run with advice enabled. Every actionable item was either
tested or assigned a concrete contract limitation.

| Advice / observation | Disposition | Evidence |
|---|---|---|
| Decode high-gap report says tracing could save 7,205 us | rejected as merge artifact | decode already is trace replay; one 7,147 us first-op cross-device gap exceeds the measured 0.664 ms/replay wall; recurring gaps are 0.103 ms/replay |
| Matmuls marked `SLOW` at 44-52% | role tuning applied | independent QKV/O/gate/down core and block-width sweep; O8 won three alternating pairs; other candidates were slower |
| “No output subblock size found” | not exposed by API | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has no output-subblock field; `mesh_plan_summary` records that factory-owned contract |
| Use HiFi2/HiFi4 for accuracy | rejected on measured latency; unnecessary for PCC | attention HiFi2 0.669230 ms; MLP HiFi2 0.782342 ms; selected LoFi 0.629461 ms at PCC 0.99997244 |
| Prefill high gaps; trace could save 1,442 us | not selected | prefill must accept dynamic non-aligned lengths and caller page tables; decode is the required and validated trace path |
| Put prefill input 0 in L1 | rejected by public capacity contract | the same path supports logical lengths through 12,352; full prefill activations cannot be retained in L1, and a special length-17-only contract is not acceptable |
| 64 cache-fill triplets have host gaps | retained for correctness | per-user logical cache ownership and aligned-slice materialization are required; no matching batched fill op exists |
| Reduce collective launches by fusion | measured and rejected | correct fused matmul+RS makes the whole layer 0.768530 ms; fused AG+matmul is numerically wrong |
| Lower CCL precision | measured and rejected | BFP8 CCL is correct but slower at 0.675635 ms and has less margin |

## Commands and retained CSVs

```bash
tt-perf-report <raw-csv> \
  --start-signpost MULTICHIP_DECODE --end-signpost MULTICHIP_DECODE_END \
  --no-color --no-host-ops --no-summary --raw-op-codes --no-advice \
  --csv decode_perf_report.csv

tt-perf-report <raw-csv> \
  --start-signpost MULTICHIP_PREFILL --end-signpost MULTICHIP_PREFILL_END \
  --no-color --no-host-ops --no-summary --raw-op-codes --no-advice \
  --csv prefill_perf_report.csv
```

`decode_perf_report.csv` SHA256:
`32ab91a07bc282d5b2bf7be1adbe5addc7777cb2d4bdeedea33f83fe64617ec8`

`prefill_perf_report.csv` SHA256:
`03af5dff3bc5dafe96ec7a24199424493f583393cddd833d1f56621355d74e85`

These compact CSVs are the committed machine-readable tables. The raw Tracy
CSV remains available locally and is excluded from the commit because of size.
