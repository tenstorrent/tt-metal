# Optimized decoder work log

## Operation-topology audit

| Current topology | Action | Evidence |
|---|---|---|
| separate Q/K/V | pack QKV | kept; split was 1.587523/1.769701 ms vs final 1.168046/1.354522 ms b1/b32 |
| separate gate/up | pack and fuse SiLU multiply | kept |
| interleaved decode linears | DRAM-shard QKV/O/gate-up/down | kept |
| MLP DRAM round-trip | carry L1 through gate/up slices, multiply, down | kept; full b1 improved 1.212745 to 1.197807 ms |
| hand attention | TTNN SDPA decode | already composite; kept |
| BF16 cache | BFP8 cache | kept; paged-boundary PCC 0.999993 |
| phase-shared weights | interleaved prefill copies | required because decode-sharded weights select incompatible prefill programs |
| checkpoint Q rows | reorder per-head q/gate at setup | fixes official-weight PCC from approximately 0.69 to 0.997327 |

There are no collectives, CCL buffers, experts, LM head, or sampling in this
single-device decoder scope.

## Required shard advisor

The exact Part-B setup was used in a separate shell:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source "$TTMLIR_ADVISOR_HOME/tools/ttnn-jit/integrations/agentic-research/shard-advise/scripts/bootstrap.sh"
ttnn-advise capture tests/advise_optimized_decoder.py:decode --out doc/optimized_decoder/shard_advise
SHARD_ADVISE_BATCH=32 ttnn-advise capture tests/advise_optimized_decoder.py:decode --out doc/optimized_decoder/shard_advise/batch32
```

Both final-policy captures completed with 13 ops, 12 final choices, spill
analysis enabled, and zero spills. The batch-1 and batch-32 tile-padded advice
is identical:

- QKV DS: `in0_block_w=4`, `M=1`, `N=5`;
- O DS: `in0_block_w=12`, `M=1`, `N=2`;
- gate/up 11x10 1D: `in0_block_w=2`, `M=1`, `N=10`, subblock 1x5;
- down DS: `in0_block_w=17`, `M=1`, `N=2`;
- L1 width-sharded dense intermediates.

Applied: QKV/O/down seeds and the coherent MLP L1 chain. Measurement overruled
the 1D gate/up choice: DS `in0_block_w=2` won.

Rejected with evidence:

- gate/up DS `in0_block_w=5` exceeded Blackhole L1 by 12,544 bytes at both
  batches; 10 is larger and cannot fit;
- down `in0_block_w=17` was initially slower under the pre-L1/BFP8 comparison,
  but the precision-locked L1-chain rerun won and was promoted;
- QKV `in0_block_w=10,N=7` and O `in0_block_w=24,N=2` exceeded the
  1,572,864-byte L1 limit at both batches;
- block-sharded norm through the entire attention/residual chain requires
  restores at head transforms, SDPA, and residual APIs. The adapted MLP
  subchain was legal and retained.

Initial capture adaptations are recorded for reproducibility: the advisor env
lacked Transformers, so capture uses exact local dense tensors; its tracer
could not subscript `TracedTensor`, so explicit `ttnn.slice` is used.

## Precision, fidelity, geometry

| Candidate | Full b1/b32 | Decision |
|---|---:|---|
| final BFP8 attention, BFP4 MLP, LoFi, DS+L1 | 1.168046 / 1.354522 ms | kept |
| same dtypes/layout, HiFi2 | 1.707306 / 1.896968 ms | reject |
| BFP4 attention, otherwise final | faster synthetic; real PCC 0.987364 | reject |
| BFP8 down, otherwise final | 1.207291 / 1.394629 ms | reject |
| packed interleaved, final precision | 1.471094 / 1.655403 ms | reject |
| split final precision | 1.587523 / 1.769701 ms | reject |
| DS gate/up iw2 before L1 chain | 1.212745 / 1.398149 ms | improved further |

BFP4 gate/up/down have official-weight evidence. LoFi beat the same-dtype,
same-layout HiFi2 candidate. All material candidates were tested at batches 1
and 32; batch 1 is the primary target.

## Commands and gates

Representative commands:

```bash
pytest -q tests/test_optimized_decoder.py tests/test_functional_decoder.py tests/test_fused_decoder.py
python tests/full_attention_real_pcc.py --decoder optimized --candidate default
python tests/linear_attention_real_pcc.py --decoder optimized --candidate default
python tests/full_attention_cache_pcc.py --decoder optimized --candidate default
python tests/traced_synthetic_pcc.py --kind full --batch 1 --decoder optimized
python tests/traced_synthetic_pcc.py --kind linear --batch 32 --decoder optimized
TT_METAL_WATCHER=10 python tests/traced_synthetic_pcc.py --kind full --batch 32 --decoder optimized
python -m tracy -r -p -o doc/optimized_decoder/profiler/decode_b1 tests/traced_synthetic_pcc.py --kind full --batch 1 --decoder optimized --perf-iterations 2
tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END <ops.csv>
```

Profiler and watcher were run separately. Final compact decode and prefill
reports are preserved under `profiler/`; correctness/cache/real-weight logs
and the command-labelled candidate sweep are under `evidence/`; watcher logs
are under `watcher/`.

Prefill before/after values use the same current five-iteration synchronized
host harness; fused controls are preserved in `evidence/fused_prefill_perf.log`
and optimized runs in `evidence/prefill_perf.log`.

Final profiler-run accounting is reconciled in
`profiler/decode_b1/{run.log,tt_perf_report.txt}`: two replays total 2.206 ms
device time plus 0.171 ms op-to-op gaps, or 1.103+0.0855 ms per replay, against
the same run's 1.194660 ms host median. The remaining approximately 0.006 ms
is report/host boundary overhead.

## Optimize checklist

- [x] independent optimized path, no host fallback/conversion
- [x] prefill/decode PCC at bar for both layer kinds and batches
- [x] paged cache, permuted pages, non-aligned lengths, determinism
- [x] ten-replay trace stress and separate clean watcher runs
- [x] warmed before/after prefill and decode at b1/b32
- [x] topology audit, packed-vs-split, SDPA, phase-specific layouts
- [x] same-pass final-policy advisor artifacts at b1/b32
- [x] precision/fidelity crossed with final layout and verified in profiler
- [x] dominant-role geometry and DS gate/up/down sweeps
- [x] final default reproduces the selected best candidate

## Stage review

Final independent verdict: **clean-pass**. Earlier findings drove the
official-weight Q/gate loader repair, final-policy advisor reruns, DS gate/up
and L1-chain work, compact runner evidence, and same-harness prefill controls.
No required work remains. The local checkpoint SHA is recorded below.

- Optimized-decoder checkpoint: `bb3a32d5c3c`
