# Falcon3-10B-Base multichip decoder

This stage implements a real four-chip TTNN tensor-parallel decoder for
`tiiuae/Falcon3-10B-Base`. `tt/optimized_decoder.py` is the numerical and
single-chip performance baseline. Scope stops at the decoder layer: no block
stack, embeddings, logits, generator, full model, or vLLM work is included.

Final source provenance:

- repo start/head during evidence collection: `6f97f9aa5a9`
- `tt/multichip_decoder.py` SHA256: `25d9b50dd182a626f5195db56e194d897ab495e53b16c7f107465ac62fdd41e9`
- `tests/test_multichip_decoder.py` SHA256: `09f8eec582af3ac057ca0b3d5194fb5bdfb2ab0e8f56b57a99d63ca92a277075`
- final independent `$stage-review`: `clean-pass`

Most initial hardware evidence was collected immediately before the commit hook
applied Black-only line wrapping to those two Python files. Review remediation
then closed the candidate geometry matrix, corrected the performance harness to
compare decode position 17 with the position-17 optimized baseline, and selected
the repeatably faster two-core output projection. Exact source transitions are
retained in `results/source_hash_provenance.json`; final result JSONs name the
final source and test.

## Hardware and selected parallel plan

The target is fixed to the machine's four Blackhole p300c devices, opened as
`MeshShape(1,4)` in device order `[3,2,1,0]`. Each chip has an 11x10 compute
grid and eight DRAM banks. Fabric discovery reports degree two for every chip,
so the selected strategy is TP=4 on mesh axis 1 with `FABRIC_1D_RING`, two
links, and BF16 collective payloads. Smaller or different meshes are not
supported.

Falcon3-10B is dense. There is no router or expert placement; the MoE strategy
is therefore “not applicable,” and every dense MLP executes once per layer.

| Tensor | Global logical shape | Mesh mapping | Per-device logical / physical shape |
|---|---:|---|---:|
| RMSNorm weights | `[3072]` | replicated | `[3072]` |
| packed QKV weight | `[3072,5120]` | rank-group Q/K/V, column TP | `[3072,1280]` |
| Q/K/V heads | Q=12, K/V=4, dim=256 | head TP | Q=3, K/V=1 |
| output weight | `[3072,3072]` | row TP | `[768,3072]` |
| gate/up weights | each `[3072,23040]` | column TP | logical `[3072,5760]`, physical `[3072,6144]` |
| down weight | `[23040,3072]` | row TP | logical `[5760,3072]`, physical `[6144,3072]` |
| residual input/output | `[1,batch,seq,3072]` | replicated | full logical shape |
| K or V cache/layer | `[batch,4,context,256]` | KV-head TP | `[batch,1,context,256]` |

Packed QKV is reordered at load time as `[Q_r,K_r,V_r]` for each rank. A
contiguous shard of global `[Q,K,V]` would assign the wrong head types.

The logical local MLP width 5,760 is not divisible by the eight-bank DRAM
stripe `32*8=256`. Load time pads each rank to 6,144. Gate/up may populate the
384 padded channels, while the corresponding 384 down-weight rows are zero, so
the logical result is unchanged. This padding is internal and does not expose
an aligned-only sequence or activation contract.

### Decode sharding and programs

The selected BFP4/LoFi decode core targets are QKV/O/gate-up/down =
`4/2/24/8`. Batch-32 physical rows are 32.

| Operation | Grid | L1 width shard input -> output | DRAM weight shard per bank |
|---|---:|---:|---:|
| residual/RMSNorm | 8x4 | `[32,96]` | n/a |
| QKV | 4x1 | `[32,768] -> [32,320]` | `[3072,160]` |
| output | 2x1 | `[32,384] -> [32,1536]` | `[768,384]` |
| gate/up | 8x3 | `[32,128] -> [32,256]` | `[3072,768]` |
| down | 8x1 | `[32,768] -> [32,384]` | `[6144,384]` |

BF16/HiFi4 down projection uses 24 cores instead of 8. The 8-core BF16
program requested 2,003,712 bytes of circular buffers/core, above Blackhole's
1,572,864-byte L1 limit. The selected BFP4 production policy fits and is faster
on 8 cores.

Prefill uses an 11x10 grid with `in0_block_w=8` and processes rows in chunks of
at most 1,024. Logical lengths 17, 31, 1,025, and 32,768 are validated; tile
padding and chunking never constrain the public sequence length.

### Collectives and stacked layout

Output and down projections are row-parallel. Each partial is summed with a
BF16 two-link Ring all-reduce. TTNN's profile spells each all-reduce as a
reduce-scatter plus all-gather. The resulting residual remains replicated, so
decoder input and output are both exactly `[1,batch,seq,3072]` and adjacent
layers need no layout repair.

At batch-32 decode, one BF16 residual is 196,608 bytes. A TP4 ring all-reduce
moves approximately 294,912 bytes/rank; both layer boundaries move about
589,824 bytes/rank. At sequence 31, the two boundaries move about 18,284,544
bytes/rank.

## Rejected alternatives

All 27 candidate JSONs use the final source/test hashes, real layer-20 weights,
and five warmed samples of 100 trace replays each.

| Candidate | Prefill ms | Decode ms | Decision |
|---|---:|---:|---|
| Ring BF16, 2 links, 4/2/24/8, prefill 11x10/block8 | 2.70928 | 0.57678 | selected |
| QKV target 2 -> grid 2x1 | 2.77600 | 0.57736 | slower prefill and decode |
| QKV targets 8 / 16 -> grids 8x1 / 8x1 | 2.72623 / 2.78088 | 0.57689 / 0.57701 | target 16 resolves to 8 cores; no balanced improvement |
| O target 4 -> grid 4x1 | 2.80779 | 0.57745 | slower than selected O2 |
| O targets 6 / 8 / 12 / 16 -> grids 6x1 / 8x1 / 6x2 / 6x2 | 2.73032 / 2.74081 / 2.72208 / 2.79173 | 0.57778 / 0.57778 / 0.57788 / 0.57767 | target 16 resolves to 12 cores; slower decode |
| O targets 24 / 48 -> grids 8x3 / 8x3 | 2.82249 / 2.70547 | 0.58343 / 0.58339 | target 48 resolves to 24 cores; slower decode |
| gate/up targets 8 / 12 -> grids 8x1 / 6x2 | 2.75179 / 2.77009 | 0.59418 / 0.58552 | slower |
| gate/up targets 16 / 32 / 48 -> grids 8x2 / 8x4 / 8x6 | 2.79570 / 2.90243 / 2.72490 | 0.58030 / 0.57729 / 0.58599 | no balanced improvement |
| down target 4 -> grid 4x1 | 2.72259 | 0.57886 | slower decode |
| down targets 12 / 24 -> grids 6x2 / 8x3 | 2.75963 / 2.76776 | 0.57769 / 0.57915 | slower |
| prefill block 4 / 12 / 24 | 2.81633 / 2.74101 / 2.71597 | 0.57682 / 0.57663 / 0.57676 | no repeatable balanced win |
| prefill grid8/block8 | 2.77488 | 0.57655 | decode noise does not offset slower prefill |
| Linear BF16, 2 links | 2.83539 | 0.57563 | 0.20% decode gain costs 4.65% prefill and abandons the physical balanced Ring |
| Ring BFP8 CCL | 2.72502 | 0.58661 | cast/accuracy payload path slower |
| Ring BF16, 1 link | 2.85078 | 0.58130 | slower |

O2 and O4 were additionally run as three alternating pairs at the corrected
logical decode position 17. O2 won all three: median 0.576597 ms versus
0.577335 ms, a 0.128% improvement. O-grid selection cannot cause prefill
latency because prefill has a separate program, so noisy prefill samples were
not used to reject O2. The six `confirm_o*_pos17_*.json` files preserve the
pair evidence and their exact pre-default-change source provenance.

The lower-data-movement rewrite was investigated with `$autotriage` and
`$autofix`:

- fused `matmul_reduce_scatter_async` hangs for this TP4 geometry with one or
  two links, although ordinary matmul plus standalone reduce-scatter passes;
- fused all-gather-matmul hardcodes four transfers, which makes its receiver
  expect 24 blocks while TP4 supplies 12 (`ring_size/2` would be two);
- a fully standalone RS -> sharded add/distributed RMSNorm -> AG -> matmul probe
  reached PCC 0.99979438 and a 1.00765x isolated latency win. Two attempts
  failed teardown directly. A third added complete stall-group/sub-device
  cleanup and exited pytest normally, but the immediately following mesh open
  still found active Ethernet core 29-25 without a heartbeat. `$autofix`
  established that queues were drained and no decoder-local cleanup can repair
  firmware state that remains poisoned across process exit; board reset was
  required. An integration prototype also exceeded the worker-watcher TENSIX
  config buffer (72,640 > 70,656 bytes) and reproduced the ERISC teardown stall.

The clean-exit gate therefore rejects the rewrite. Production retains the
stable replicated all-reduce path. See `AUTOTRIAGE.md`, `AUTOFIX.md`, and
`results/graph_rewrite_final_decision.json`; the pass-then-poisoned-next-open
sequence is recorded in `results/graph_rewrite_delayed_health.json`.

## Context and full-stack memory fit

The decoder supports the HF limit 32,768. Batch-1 actually executes a full
32,768-token prefill, cyclic paged cache sampling, and decode at 32,767.
Batch-32 full-stack residency is projected from measured allocator capacity;
this stage does not claim a 40-layer execution.

Per device at batch 32/context 32,768:

| Residency | Bytes/device |
|---|---:|
| all 40 layers, local BFP8 K+V | 22,817,013,760 |
| all 40 layers, separate prefill+decode physical padded projection copies | 2,831,155,200 |
| known subtotal | 25,648,168,960 |
| measured TTNN allocator capacity (8 x 4,272,341,376) | 34,178,731,008 |
| headroom after known subtotal | 8,530,562,048 |

The conservative remaining reserve is 6,576,634,112 bytes: replicated BF16
embedding 805,306,368; replicated untied BF16 LM head 805,306,368; shared RoPE
33,554,432; norms 497,664; page table 131,072; trace region 100,000,000;
activation/residual/CCL reserve 536,870,912; and allocator/program/fragmentation
margin 4,294,967,296. That leaves 1,953,927,936 bytes uncommitted. The later
full-model stage must validate its actual scheduling, but there is no physical
DRAM reason to reduce this decoder's advertised context.

## Correctness and runtime gates

PCC threshold is 0.99.

| Gate | Final result |
|---|---|
| TP4 vs optimized TTNN, batch32/seq17 | prefill 0.99999951; decode 0.99999993; K 0.99999576; V 0.99999854 |
| TP4 vs HF, batch32/seq31 | prefill 0.99994867; decode31/32 0.99932893/0.99923151 |
| paged K/V through position 32 | K 0.99656263; V 0.99498064 |
| BF16/HiFi4 and stacked layout | prefill 0.99938993; stacked 0.99904697; decode 0.99941353 |
| sequence 1,025 | prefill 0.99995054; K 0.99655544; V 0.99496660 |
| heterogeneous positions 17/31 | output >=0.99999249; K >=0.99999381; V 1.0 |
| context 32,768 | K 0.99642031; V 0.99476878; final decode passed |
| trace | paged decode capture/replay; repeated replay bitwise identical |
| fallback | strict `throw_exception_on_fallback=true` passed |
| watcher | worker-watcher run passed; no error/fatal/assert/sanitization marker |

The default suite is 5 passed and 6 intentional manual skips. The hot-path
source audit rejects host tensor conversions and calls into the functional or
single-chip implementation. Paged cache is
`[physical_pages,1,32,256]`/rank, page tables may arbitrarily permute pages,
and per-user positions may differ within a batch.

Watcher and profiler were run separately. Watcher used
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`; full Ethernet watcher
instrumentation exceeds the active-fabric kernel config buffer, while worker
watcher still covers the decoder kernels and Ring CCL remains active.

## Performance and profiler

Authoritative numbers are uninstrumented warmed medians; each decode sample is
100 trace replays.

| Batch | Single-chip decode ms | TP4 decode ms | Speedup | Efficiency | Single-chip prefill ms | TP4 prefill ms | Prefill speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.793496 | 0.576824 | 1.37563x | 34.39% | 3.278021 | 2.771583 | 1.18273x |
| 1 | 0.668364 | 0.370774 | 1.80262x | 45.07% | not recorded | 0.853517 | n/a |

Final Tracy batch32/seq17 results:

| Phase | Wall ms | Device ops us | Gaps us | Matmul us (% device / % device+gap) | CCL us (% device / % device+gap) | DRAM |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 4.05236 | 1500.41 | 2399.11 | 351.38 (23.42% / 9.01%) | 221.08 (14.73% / 5.67%) | 47 GB/s, 9.2% |
| decode | 0.61405 | 518.42 | 82.17 | 136.92 (26.41% / 22.80%) | 76.39 (14.74% / 12.72%) | 61 GB/s, 11.9% |

Decode is neither compute- nor DRAM-roofline limited; CCL, layouts/data
movement, small operations, and gaps explain the remaining scaling loss.
`tracy/profile_summary.json` defines both percentage denominators. Human tables
and phase CSVs are in `tracy/`; the raw 5 MB ops CSV is retained with exact
provenance, while duplicate multi-hundred-MB Tracy runtime logs were removed.
The 93,351.652 us decode CSV gap crossing Tracy iterations is excluded from the
per-replay gap denominator and recorded explicitly in the summary JSON.

## Limitations

Only the detected four-chip p300c mesh is supported. Batch-32 40-layer/context
residency is calculated, not executed. The full-context execution is batch 1.
Full-model residual scheduling, embeddings, logits, generation, and vLLM are
explicitly deferred. Reconsidering the sharded-residual rewrite requires fixes
for async-CCL fabric termination and TP4 fused-AGMM transfer accounting, then a
watcher-clean exact two-layer boundary rerun with a passing subsequent mesh
open/close health check.
