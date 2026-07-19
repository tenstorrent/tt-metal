# Optimized multichip decoder work log

## Scope and starting point

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/model/qwen-qwen3-32b`
- Starting HEAD: `b9628a153c9` (`Record Qwen3-32B multichip stage commit`)
- Completed multichip stage commit: `9a49fbb0bf1`
- Model: `Qwen/Qwen3-32B`, checkpoint revision
  `9216db5781bf21249d130ec9da846c4624c16137`
- Hardware: four Blackhole p300c, UMD IDs 0-3, `1x4 FABRIC_1D_RING`
- Stage scope: decoder layer only; no full-model or vLLM work

The unrelated pre-existing modification to
`.agents/skills/forge-functional-decoder-from-ir/SKILL.md` was not edited or
staged.

## Health and serialization policy

Every hardware job ran serially. Watcher, profiler, and ordinary measurements
were never mixed. Baseline and final health checks used:

```bash
tt-smi -ls --local
```

All four devices were present before and after the pass. The environment emits
a `/dev/shm` warning requesting 16 MiB with approximately 17.5 MiB available;
the measurements completed and this is recorded as environment provenance.

Two experimental fused-AGMM stalls were handled under `$tt-device-usage`:
bounded observation/triage, kill only the exact process, one bounded
`timeout 60 tt-smi -r all`, then `tt-smi -ls --local`. No second reset was
needed. See `AUTOFIX.md` and `triage/`.

## Common real-weight measurement command

```bash
export QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
export QWEN3_32B_MULTICHIP_BASELINE_PATH=models/autoports/qwen_qwen3_32b/doc/multichip_decoder/artifacts/optimized_single_chip_real_layer32.pt
export QWEN3_32B_MULTICHIP_RESULTS_DIR=models/autoports/qwen_qwen3_32b/doc/optimized_multichip_decoder/results
export QWEN3_32B_MULTICHIP_RUN_PERF=1
export QWEN3_32B_MULTICHIP_RESULT_NAME=<artifact.json>
export QWEN3_32B_MULTICHIP_DECODE_REPLAYS=<100-or-1000>
export QWEN3_32B_MULTICHIP_DECODE_TRIALS=<5-or-9>
export QWEN3_32B_MULTICHIP_PREFILL_TRIALS=<5-or-9>
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_warmed_prefill_and_traced_decode -s
```

The harness uses prompt-derived HF boundary activations captured at layer 32,
repeated across batch 32, and the accepted optimized single-chip output as the
PCC baseline. Each JSON records source hashes, activation/checkpoint metadata,
mesh plan, individual timing samples, and hardware.

## Baseline

`results/baseline_before.json`:

| Prefill PCC | Decode PCC | Warmed prefill | Traced decode |
|---:|---:|---:|---:|
| 0.9999999536268244 | 0.9999724364868658 | 4.641534760594368 ms | 0.6295932177454233 ms |

This pass's environment was noisier for prefill than the prior multichip-stage
record (3.127879 ms), so all pass-local speedups use the current-run baseline.
Historical results are not substituted for the before value.

## Topology audit and graph rewrite

The source and first profiler were audited before knob sweeps. Existing
dedicated ops—rank-packed QKV, QKV head split, SDPA, RoPE, head concat,
RMSNorm, and SiLU folded into multiply—were retained. The pass then:

1. removed the final decode L1-sharded-to-DRAM restore;
2. replaced same-input gate/up matmuls by one rank-packed projection plus
   slices in prefill and decode;
3. measured distributed RMSNorm followed by ordinary AG/matmul;
4. adapted and measured distributed RMSNorm plus fused AGMM;
5. measured fused row matmul+reduce-scatter for O/down;
6. measured the full shard-advisor family including its input layouts and
   output reverts.

The action/evidence table is in `README.md`.

## Candidate ledger

The following environment controls were applied to the common command. A first
API/config error was adapted and retried; only completed trials are used for
performance rejection.

| Artifact | Material controls | Prefill ms | Decode ms | PCC result |
|---|---|---:|---:|---|
| `candidate_sharded_output.json` | `KEEP_OUTPUT_SHARDED=1` | 4.313862 | 0.624113 | baseline preserved |
| `candidate_sharded_output_packed_mlp.json` | packed, DRAM split, 20 cores | 3.476185 | 0.617377 | preserved |
| `candidate_sharded_output_packed_mlp_dram10_in08.json` | packed 10 cores, in0=8 | 3.475837 | 0.616889 | preserved |
| `candidate_sharded_output_packed_mlp_dram40.json` | packed 40 cores | 3.433956 | 0.625113 | decode 0.999966899 |
| `candidate_sharded_output_packed_mlp_l1split20.json` | L1 split, 20 cores | 3.421913 | 0.624330 | preserved |
| `candidate_sharded_output_packed_mlp_l1split40.json` | L1 split, 40 cores | 3.453123 | 0.632892 | decode 0.999966899 |
| `candidate_packed_sharded_dram10_in04.json` | packed 10, in0=4 | 3.496187 | 0.624872 | decode 0.999966899 |
| `candidate_packed_sharded_dram10_in02.json` | packed 10, in0=2 | 3.435271 | 0.645748 | decode 0.999963012 |
| `candidate_packed_sharded_dram10_in08_long.json` | 1000 replays x 9, 10 cores | 3.392129 | 0.616706 | preserved |
| `candidate_packed_sharded_dram20_in08_long.json` | 1000 replays x 9, 20 cores | 3.441192 | 0.617178 | preserved |
| `candidate_packed_sharded_nonpersistent.json` | `PERSISTENT_CCL=0` | 3.441491 | 0.630448 | preserved |
| `candidate_packed_sharded_bfp8_ccl.json` | `CCL_DTYPE=bfp8` | 3.595996 | 0.635099 | lower PCC |
| `candidate_distributed_norm_separate_agmm_retry_stats_sharded.json` | distributed norm; ordinary AG/matmul | 3.409621 | 0.628085 | preserved |
| `candidate_distributed_norm_fused_agmm_1link.json` | distributed norm; one-link fused AGMM | 3.464048 | 0.778513 | decode 0.999962936 |
| `candidate_packed_sharded_fused_matmul_rs.json` | fused O/down matmul-RS | 3.447051 | 0.752589 | preserved |
| `candidate_shard_advisor_rank_local.json` | advisor rank-local 1-D family | 3.476971 | 0.728389 | decode 0.999955656 |
| `candidate_packed_sharded_attention_hifi2.json` | attention HiFi2 | 3.484897 | 0.653609 | lower PCC |
| `candidate_packed_sharded_mlp_hifi2.json` | MLP HiFi2 | 3.630294 | 0.767596 | preserved |
| `candidate_packed_sharded_kv_bf16.json` | BF16 KV cache | 3.560661 | 0.618760 | no PCC benefit |

The 10-core default initially selected an illegal packed in0 block of 16. It
was adapted to all meaningful legal divisors 8/4/2; 8 won. The fused families'
rank, stats layout, link count, and packed geometry adaptations are detailed in
`README.md` and `AUTOFIX.md`.

## Shard advisor

The first capture returned a tuple, which the advisor requires to be a single
tensor. It was adapted into four exact TP-rank projection entry points instead
of treating that API error as rejection.

```bash
cd /home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
export PYTHONPATH=/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages:/home/mvasiljevic/tt-metal:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-}
for role in qkv packed_gate_up o down; do
  ttnn-advise capture \
    /home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_32b/doc/optimized_multichip_decoder/advise_qwen3_32b_multichip.py:decode_${role} \
    --out /home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_32b/doc/optimized_multichip_decoder/shard_advise/${role}
done
```

Each role produced `report.json`, `report.txt`, and `final_ir.mlir`, with spill
pass run and zero spills. The translated plan used:

- QKV: 80-core L1 width-sharded output, `11x8`, in0=8, perN=1;
- packed gate/up: 80-core input, 100-core output, `11x10`, in0=2,
  perN/outblock/subblock=4;
- O: 32-core input, 80-core output, `11x8`, in0=2, perN=2;
- down: 50-core input, 80-core output, `11x8`, in0=2, perN=2;
- all four final-IR output reverts to DRAM interleaved.

The full executable family passed PCC and measured 0.728389 ms decode, so the
DRAM-sharded family remains authoritative.

## Final default measurement

No candidate environment flags were set:

```bash
export QWEN3_32B_MULTICHIP_RESULT_NAME=final_default.json
export QWEN3_32B_MULTICHIP_DECODE_REPLAYS=1000
export QWEN3_32B_MULTICHIP_DECODE_TRIALS=9
export QWEN3_32B_MULTICHIP_PREFILL_TRIALS=9
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_warmed_prefill_and_traced_decode -s
```

Result: prefill PCC 0.9999999536268244, decode PCC
0.9999724364868658, warmed prefill 3.372121136635542 ms, traced decode
0.6167201013304293 ms. Relative to the pass-local baseline: 27.3490% faster
prefill and 2.0447% faster decode. This rerun followed the final stress-harness
edit; its provenance hashes exactly match the current decoder and test
sources.

## Profiler-advised prefill L1 input A/B

The first independent review correctly noted that the report's “place input 0
in L1” advice had not been tried on the final packed graph. The candidate now
copies each already-bounded (at most 640-row) QKV, O, packed gate/up, and down
input to L1 interleaved immediately before matmul:

```bash
export QWEN3_32B_MULTICHIP_PREFILL_INPUT_L1=1
export QWEN3_32B_MULTICHIP_RESULT_NAME=candidate_prefill_input_l1.json
export QWEN3_32B_MULTICHIP_PREFILL_TRIALS=7
export QWEN3_32B_MULTICHIP_DECODE_TRIALS=5
export QWEN3_32B_MULTICHIP_DECODE_REPLAYS=100
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_warmed_prefill_and_traced_decode -s

unset QWEN3_32B_MULTICHIP_PREFILL_INPUT_L1
export QWEN3_32B_MULTICHIP_RESULT_NAME=candidate_prefill_input_dram_ab.json
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_warmed_prefill_and_traced_decode -s
```

Both passed exact baseline PCC. L1 measured 3.5721617750823498 ms versus
3.4525739029049873 ms for the source-identical DRAM run, a 3.4637% regression.
This adapts the advice to the chunked topology and rejects it on measured copy
cost, not on the unchunked full-context allocation argument.

## Final correctness and stress commands

Fallback audit, non-aligned/stacked/contiguous/paged/trace stress:

```bash
export TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_contract_is_optimized_owned_and_host_free -s
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_synthetic_non_aligned_prefill_matches_hf -s
```

Length 31, direct second-layer consumption, contiguous/paged equivalence,
ten-replay determinism, and advancing positions 32->33 all passed. The direct
two-layer prefill PCC is 0.9912523198182539 and the direct two-layer decode PCC
is 0.9909803999302423. The decode output shards are asserted to be L1
width-sharded before direct consumption. The first extended run retained the
extra second-layer output through trace capture and exposed an L1 circular
buffer clash; deallocating that completed test-only lifetime before capture
made the intended sequential-layer contract pass. No fallback exception
fired.

The final consolidated rerun retained `final_gate.xml` and passed four tests:
the static host/fallback audit, non-aligned/direct-two-layer stress, real
layer-32 checkpoints, and paged trace refresh.

Real weights plus paged fixed-address refresh:

```bash
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_real_layer_matches_optimized_single_chip_baseline \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_paged_trace_refresh_matches_eager -s
```

Both passed. `results/paged_trace_refresh.json` records exact PCC 1.0 after
page-table refresh and after position 64->65.

## Context capacity

```bash
export QWEN3_32B_MULTICHIP_RUN_CAPACITY=1
export QWEN3_32B_MULTICHIP_CAPACITY_SEQUENCE=12352
export QWEN3_32B_MULTICHIP_CAPACITY_EXPECT=pass
pytest -q models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_full_stack_capacity -s

export QWEN3_32B_MULTICHIP_CAPACITY_SEQUENCE=12353
export QWEN3_32B_MULTICHIP_CAPACITY_EXPECT=fail
pytest -q models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_full_stack_capacity -s
```

Both contract probes passed. The first retains 28,539,904 free bytes/device;
the second records the expected physical OOM at the prefill live set. Updated
evidence is in `doc/context_contract.json` and `results/capacity_seq*.json`.

## Watcher

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_real_layer_matches_optimized_single_chip_baseline -s
```

The final default real-weight test passed. The retained log contains no
`error`, `assert`, `hang`, `stuck`, or `timeout`; SHA-256
`ee439baa627976def5e7063cc7eba6ca8eaf764c519eb8fd72fb27b4a8df0b9f`.
ETH Watcher is disabled only for the documented 27,920-byte versus 25,600-byte
firmware buffer constraint.

## Tracy and tt-perf-report

Profiler and Watcher were separate:

```bash
export QWEN3_32B_MULTICHIP_RUN_PROFILE=1
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_32b/doc/optimized_multichip_decoder/tracy_final_reviewed \
  -m pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_profile_selected_multichip_decoder -s
```

Raw CSV:
`tracy_final_reviewed/reports/2026_07_19_13_10_14/ops_perf_results_2026_07_19_13_10_14.csv`,
SHA-256
`0fb51eaf87262710217bbc0a6c2dde35468779f58e3f028c08ec5c8515cae86b`.

Advice-enabled filtered tables were generated for
`MULTICHIP_PREFILL..MULTICHIP_PREFILL_END` and
`MULTICHIP_DECODE..MULTICHIP_DECODE_END`:

```bash
tt-perf-report <raw.csv> \
  --start-signpost MULTICHIP_PREFILL --end-signpost MULTICHIP_PREFILL_END \
  --no-color --no-host-ops --raw-op-codes \
  --csv doc/optimized_multichip_decoder/tracy/final_prefill_perf_report.csv \
  --summary-file doc/optimized_multichip_decoder/tracy/final_prefill_summary.csv

tt-perf-report <raw.csv> \
  --start-signpost MULTICHIP_DECODE --end-signpost MULTICHIP_DECODE_END \
  --no-color --no-host-ops --raw-op-codes \
  --csv doc/optimized_multichip_decoder/tracy/final_decode_perf_report.csv \
  --summary-file doc/optimized_multichip_decoder/tracy/final_decode_summary.csv

script -q -e -c \
  "tt-perf-report <raw.csv> --start-signpost MULTICHIP_PREFILL \
   --end-signpost MULTICHIP_PREFILL_END --no-color --no-host-ops \
   --raw-op-codes --no-summary" \
  doc/optimized_multichip_decoder/tracy/final_prefill_perf_report.txt

script -q -e -c \
  "tt-perf-report <raw.csv> --start-signpost MULTICHIP_DECODE \
   --end-signpost MULTICHIP_DECODE_END --no-color --no-host-ops \
   --raw-op-codes --no-summary" \
  doc/optimized_multichip_decoder/tracy/final_decode_perf_report.txt
```

Artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `tracy/final_prefill_perf_report.csv` | `c6975d19745d7f8f8c94d3b9875fa649e2f25ae33a8390db7ecc95942db65d77` |
| `tracy/final_prefill_perf_report.txt` | `57f8f0e4eafdd37b570e3545b968b694276ba8bdfc6e7ee60f4e0543695d83fa` |
| `tracy/final_decode_perf_report.csv` | `8dd0951c1d641239e96e822426c94e027161b845e7ad7742175e9501ed64c6ac` |
| `tracy/final_decode_perf_report.txt` | `bac0a4c344b36f645e72daecfc9d57754cf2cc1593a5ea22ebd0471ff83c7e99` |
| `tracy/final_prefill_summary.csv.csv` | `199a58bd84f23fd474cf11de26565b82fa5bc50368c68f6de9c09a6955a04019` |
| `tracy/final_decode_summary.csv.csv` | `f79b42a28a5ffdca794f741ce27cb13e81d3d5b0d06e2b2466ca76817c7d6ddb` |
| `results/final_default.json` | `b21674a999aaf2baa5c18f83aa6bff0886ad28a99627f0bdfcaabb5cefd85359` |
| `final_gate.xml` | `a97d9e2bcd2d075ed0281ec0e9228e2be2cc4791d2a75ed0c55ef7678827f610` |

Decode profile: matmul 43.03%, reduce-scatter 7.99%, all-gather 4.16%,
22.1% modeled DRAM roofline. Prefill: matmul 29.90%, reduce-scatter 8.76%,
all-gather 8.09%, 10.5% modeled DRAM roofline. The same-run roofline/device/wall
reconciliation and complete CCL contract table are in
`README.md`.

## Review and commit

The first independent `$stage-review` verdict was `more-work-needed` with four
findings. All were addressed before rereview:

- generated and retained advice-enabled human-readable prefill/decode tables;
- added same-run theoretical roofline, device-time, and wall-time
  reconciliation, including gap attribution;
- implemented and measured bounded chunk-local L1 prefill matmul inputs,
  retaining the 3.452574 -> 3.572162 ms rejection evidence;
- added the per-phase AG/RS contract table with axis, byte shapes, memory,
  links, chunk/worker/buffer tuning, persistence, row time, and decisions.

The first rereview found one documentation-only inconsistency: the prefill
reduce-scatter row described nonpersistent DRAM scratch as persistence. The CCL
contract was corrected to say that the scratch is DRAM and persistence is
`none / none`.

The final independent `$stage-review` verdict is **`clean-pass`**. It confirmed
that the corrected prefill reduce-scatter contract matches the source, all
review findings are closed, provenance is current, and no regressions or stage
blockers remain.

The stage-owned implementation and evidence commit is
`ff27647f8f88c7009bb8810a4d0b099fb07bd491`. The follow-up documentation-only
commit records that SHA. No push was performed.
