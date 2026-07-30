# Phi-3.5 mini optimized decoder

This stage owns the single-device optimized decoder only. It starts from the
fused decoder and preserves packed QKV, packed gate/up, fused SiLU-multiply,
paged-cache semantics, non-aligned prefill lengths, and the 131072-token
context contract.

## Selected cumulative contract

| Item | Selected implementation |
|---|---|
| Decode projections | DRAM-width-sharded weights and L1-width-sharded activations/outputs over 8 DRAM workers |
| QKV / output / gate-up / down | BFP4 weights, LoFi kernel, BF16 output |
| Block widths | QKV 12, output 12, gate-up 6, down 32; `per_core_M=1` for the tile-padded decode row |
| Prefill | Interleaved BFP4 packed projections; fused MLP elementwise path |
| Cache | BF16 tiled paged cache, 32-token pages |
| SDPA decode | 8x8 grid, exact exponential, unchanged logical batches 1 and 32 |
| Boundary | Decoder output restored to DRAM; no host conversion or fallback in the measured path |

The cache remains BF16. After the first dtype error, an adapted BFP8 trial
passed non-aligned cache-consuming decode at B1/B32 (PCC 0.9999885/0.9999882).
Its full-cache staging made prefill slower: 1.4850/35.9823 ms versus the BF16
path's 1.3816/30.1631 ms. AutoFix also proved that staging would overwrite
prior users on partial prefill calls. The rejected branch was removed, so
capacity and `context_contract.json` remain unchanged.

## Correctness and performance

The final default passes all 12 optimized tests. Non-aligned prefill lengths
31, 33, and 65 have PCC 0.9999827–0.9999856; decode PCC versus the functional
decoder is 0.9999473 at batch 1 and 0.9999885 at batch 32. Real Microsoft
weights pass at 0.9999257. Five trace replays are bitwise deterministic at
both batches. A 33-token prefill followed by cache-consuming position-33
decode passes at B1/B32 with permuted page tables (PCC 0.9999883/0.9999882).

Host-observed, warmed mean results (100 timed iterations, milliseconds):

| Workload | Fused before | Optimized after | Change |
|---|---:|---:|---:|
| Prefill S=128, B=1 | 1.5794 | 1.4304 | -9.4% |
| Prefill S=128, B=32 | 37.3156 | 30.1911 | -19.1% |
| Traced decode C=128, B=1 | 1.0475 | 0.5608 | -46.5% |
| Traced decode C=128, B=32 | 1.2107 | 0.7436 | -38.6% |

Primary batch-1 decode beats the best correct BFP8 interleaved candidate
(0.8277 ms), and batch 32 also improves (0.9963 to 0.7436 ms).

## Operation-topology audit

| Current topology | Candidate/action | Evidence |
|---|---|---|
| Packed QKV same-input projection | Kept packed | One matmul and one head split; splitting adds launches and rereads the same activation |
| Packed gate/up | Kept packed with fused SiLU multiply | Existing fused A/B component probe: 0.374 vs 0.561 ms |
| Decode interleaved weights | Replaced by DRAM-sharded BFP4/LoFi | Whole-layer B1 0.8277 to 0.5678 ms |
| Residual/layout transitions | Carry width-sharded tensors across matmul/add/MLP; restore DRAM once | Final correctness and trace tests pass |
| Explicit Phi RoPE | DRAM rotate-half then exact cache-update layout | Required by 48-wide half slicing; avoids host fallback |
| Decode SDPA | Explicit composite program config retained | Same logical batch/cache contract; no measurable regression |
| Prefill packed matmuls | BFP4 weights retained; decode-only DS config not applied | B1 and B32 prefill both improve |

## Shard advisor and search

The required advisor capture is in `shard_advise/`. Its final report sees all
four dense matmuls (`dram_sharded_considered=4`,
`dram_sharded_advised=4`). The advised DRAM-sharded family was applied. The
advisor's layouts were treated as seeds; measured role-specific block widths
won. Candidate logs show:

- precision-locked BFP4/LoFi all-width 2: 0.6470/0.8293 ms (B1/B32);
- all-width 4: 0.5831/0.7652 ms;
- QKV 6/down 16 with output 12/gate-up 4: 0.5642/0.7468 ms;
- QKV 12/gate-up 6/down 32 with output 12: selected at
  0.5613/0.7430 ms in the candidate run;
- gate/up width 12 reached 1,618,688 bytes versus 1,572,864 bytes of L1;
- BFP4/LoFi 0.5684/0.7515 ms beat BFP4/HiFi2
  0.7426/0.9270 ms.

All four matmuls use `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`,
8 cores, width-sharded DRAM weights, width-sharded L1 input/output,
`per_core_M=1`, and respectively `per_core_N=36, 12, 64, 12`. Input shard K
tiles/core and selected `in0_block_w` are QKV 12/12, output 12/12, gate-up
12/6, and down 32/32. Output-subblock fields are not exposed by this program
class.

The same-contract split gate/up candidate passed all 12 tests. It was
rejected: packed clearly won prefill (1.4447 vs 1.4594 ms at B1 and 30.2453
vs 31.4396 ms at B32), while split's sub-percent decode delta was within run
variance and required a second weight read and matmul.

The final Tracy/`tt-perf-report` artifacts in `tracy/` prove the measured
decode rows are LoFi BF16×BFP4→BF16. At B1, QKV/output/gate-up/down take about
55/20/93/47 µs per replay and reach roughly 234–271 GB/s. The report labels
these rows `SLOW` at 45–52% of its modeled Blackhole DRAM ceiling. They were
not accepted on that label alone: the precision-locked block sweep tried
larger legal widths, including 12, 16, and 32 where applicable; the remaining
larger candidates either regressed whole-layer time or hit recorded L1
limits. The selected cumulative path is consequently the measured winner.

Commands and raw evidence are in [work_log.md](work_log.md), `candidates/`,
`perf/`, `correctness/`, `watcher/`, and `shard_advise/`.
