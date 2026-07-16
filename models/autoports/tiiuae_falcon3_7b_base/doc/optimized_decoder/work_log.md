# Optimized decoder work log

## Scope and starting point

- Model: `tiiuae/Falcon3-7B-Base`
- Stage: single-device optimized decoder only
- Start commit: `125af85ecde`
- Functional checkpoint: `e0f096c91ee`
- Representative dense layer: 14 of 28; Falcon3 has one decoder-layer kind
- Device: one Blackhole p300c selected from the healthy paired board with
  `TT_VISIBLE_DEVICES=2,3`; compute grid 11x10, DRAM grid 8x1
- Preserved contract: prefill/decode, batch 32, cache length 128, arbitrary
  logical prefill lengths through internal padding, dense paged cache updates,
  deterministic replay, and public DRAM-interleaved output

No multichip decoder, full model, generator, or vLLM work is included.

## Operation-topology audit

This audit preceded program-config tuning.

| Region | Functional topology/movement | Candidate/action | Evidence and decision |
|---|---|---|---|
| Input norm + QKV | DRAM RMSNorm followed by already packed QKV | Keep the same-input packed Q/K/V projection; shard norm/residual and tune decode matmul | Kept. No repeated independent Q/K/V matmuls exist in the optimized graph. |
| RoPE + cache update | Head split, RoPE, slice/conversion, two persistent cache updates | Preserve the proven Q/K boundary and reduce only legal movement | Query returns to DRAM for decode SDPA; K remains update-compatible. Cache/query PCC is recorded separately from residual-dominated output PCC. |
| Decode mask | `slice -> le -> where -> repeat` plus explicit mask | Use causal decode SDPA with `cur_pos_tensor` | Applied. Explicit mask remains an A/B and is 1.003755 ms versus the sub-0.78 ms selected family. |
| SDPA + head merge | Dedicated SDPA, concat heads, conversion, O matmul | Retain composite ops and match output to the residual chain | Applied. The advisor's unfixable concat-head proposal violated the runtime sharded-input contract; the required sharded input remains. |
| MLP gate/up | Two same-input matmuls, SiLU, multiply | Compare packed gate/up with two tuned matmuls | Both are executable. Under the final DRAM-48 all-BFP4 family, packed is 0.938405 ms versus 0.773380 ms split, so split wins. |
| MLP down/residual | Product conversion, down matmul, residual add | Sweep advisor and coherent DRAM-sharded BFP4/LoFi geometries | AutoFix repaired the DRAM control's prefill/decode weight-layout bug. Aligning the final DRAM-48 down-input shard removes one reshard and beats advisor: 0.768483/0.773084 ms batch 32 and 0.644047/0.652849 ms batch 1. |
| Residual/norm layouts | Multiple functional DRAM boundaries | Seed exact advisor 11-core norm, 96-core residual, and report-sharded inputs; compare a coherent 32-core chain | Exact advisor residual and input modes are retained as controls at 0.786988/0.788957 ms; coherent 32-core residual is final. |
| Host/layout fallback | Functional path is DRAM-heavy but device-only | Own all hot methods and eliminate host fallback | Static test forbids `torch`, `from_torch`, `to_torch`, and `FunctionalDecoder.` in measured methods. Final profile has zero host ops. |

The remaining conversions are required by head creation, RoPE/SDPA, persistent
cache update, concat-head, residual, or public output contracts. The measured
path has no gratuitous host round trip, tilize/untilize, or reshard fallback.

## Baseline correctness

The unchanged functional decoder passed `3 passed in 8.03s`:

| Path | Weights | PCC |
|---|---|---:|
| Prefill, sequence 17 | synthetic BF16 | 0.99893845 |
| Prefill, sequence 128 | synthetic BF16 | 0.99897713 |
| Decode, position 17 | synthetic BF16 | 0.99911663 |
| Prefill, sequence 17 | real layer 14 | 0.99895504 |
| Decode, position 17 | real layer 14 | 0.99882657 |

The exact BF16 optimized smoke passes at 0.99880755 prefill and 0.99961181
decode. The stage acceptance bar remains PCC 0.99.

## Mandatory shard-advisor gate

The required advisor pass ran in a separate shell after:

```text
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source .agents/skills/shard-advise/scripts/bootstrap.sh
ttnn-advise capture \
  models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/shard_advise/advise_falcon3.py:decode \
  --out /tmp/falcon3-advice-mockmap-fixed
```

It captured the rewritten dense attention+MLP block: 24 ops, 21 final choices,
and spill analysis. Required artifacts:

- `shard_advise/report.json`, SHA256
  `5c991487d3093254dd5602d2a80560ce4ab5294587fd373556dbea501dcff3a5`
- `shard_advise/final_ir.mlir`, SHA256
  `659a5de831386e1e06fd5a48dbc6319aaaff15f7b15ee0b50ee303cd040bc736`

Authoritative matmul recommendations:

| Matmul | Grid | `in0_block_w` | `per_core_N` | `out_subblock_w` |
|---|---:|---:|---:|---:|
| QKV 3072x5120 | 11x8 | 2 | 2 | 2 |
| O 3072x3072 | 11x9 | 8 | 1 | 1 |
| Gate/up 3072x23040 | 11x10 | 2 | 7 | 7 |
| Down 23040x3072 | 11x9 | 8 | 1 | 1 |

Applied as the advisor control: all four exact matmul configs, width-sharded
outputs, explicit legal head/SDPA boundaries, and final DRAM output revert.
The exact 11-core block-sharded norms, 96-core residual adds, and
report-sharded inputs were implemented and measured at 0.786988/0.788957 ms
under the then-selected policy. The advisor concat-head output proposal is
invalid because TTNN requires a sharded input. The report's single spill had
no pressure event and only 4.1% final occupancy.

The final default deliberately rejects the advisor matmul family after a
like-for-like all-BFP4 cross. Before down-input alignment the two were tied;
the selected aligned DRAM-48 path is now 0.768483 versus 0.773084 ms at batch
32 and 0.644047 versus 0.652849 ms at primary batch 1. Keeping advisor output
was not required by the gate; running, preserving, and accounting for its
recommendations was.

The stage runner check passes and verifies that both mandatory artifacts are
present and parseable.

## Recorded activations

Precision and final performance use a reproducible layer-14 activation fixture,
not random inputs. `capture_activations.py` runs HF embeddings and layers 0..13
in eager BF16 CPU. The fixture contains:

- the original length-17 prompt and following token;
- a genuine, unpadded 31-token prompt and two following cache-consuming tokens;
- a genuine 128-token max-contract prefill prompt (no decode beyond the
  128-slot cache contract).

- HF revision: `bf3d7ed586cb22a921520e2d681a9d3d7642cde8`
- Shapes: prefill `[1,17,3072]`, `[1,31,3072]`, `[1,128,3072]`; decode
  inputs `[1,1,3072]` at positions 17, 31, and 32
- `activations/layer14_inputs.safetensors`, SHA256
  `ca6aed21bfae7bbde56d995523d9d70f107564820fdea57cc8d1832af768c40c`
- All token IDs, prompts, producer, and shapes are in
  `activations/layer14_inputs.json`.

Random real-weight inputs remain a sensitivity diagnostic. Per OPT-012, they
are not a precision veto when genuine target activations pass. The final
contract gate therefore uses recorded activations for 17/31/128 prefill,
sequential decode/query/cache checks at 31/32, and repeated writes.

## AutoFix: catastrophic DRAM-BFP4 PCC

The second stage review identified the original near-zero DRAM-BFP4 result as
a bug signature. `$autofix` invoked fresh-context `$autodebug`, then isolated
each hypothesis on hardware. Full details are in `AUTODEBUG.md`, `AUTOFIX.md`,
and `results/autofix/`.

| Hypothesis | Evidence | Result |
|---|---|---|
| Raw BFP4 upload is corrupt | Advisor and DRAM gate/up/down round trips have the same approximately 0.993 PCC | Refuted |
| DRAM MLP linears or 24/48-core geometry are bad | Gate/up/gated/down and clean-cache decode closely match advisor | Refuted |
| Decode itself is corrupt | HF-filled-cache decode is approximately 0.999999 PCC | Refuted |
| Prefill state is corrupt | Failing reproduction diverges in attention and both KV caches before MLP/decode | Confirmed boundary |
| MLP-only weight split is sufficient | Preserved post-partial-fix result remains near zero | Refuted as incomplete |
| Decode-formatted DRAM width-sharded weights leaked into generic large-M prefill | Independent static report plus all-weight prefill/cache A/B | Confirmed root cause |

Fix: prefill QKV/O/gate/up/down are always interleaved DRAM; DRAM decode mode
owns separate width-sharded decode copies. A normal acceptance test asserts the
layout contract. The selected DRAM-48 path intentionally pays this persistent
duplicate-weight cost; length-128 acceptance still passes, so capacity remains
128.

Post-fix focused result:

| Candidate | Prefill PCC | Decode PCC | Warm prefill | Traced decode |
|---|---:|---:|---:|---:|
| Advisor MLP-BFP4 | 0.99998603 | 0.99999903 | 3.248070 ms | 0.782636 ms |
| DRAM BFP4, 24 cores | 0.99998603 | 0.99999904 | 3.264622 ms | 0.813781 ms |
| DRAM BFP4, 48 cores | 0.99998603 | 0.99999909 | 3.250064 ms | 0.813254 ms |

The comprehensive post-fix frontier independently reproduces the same
correctness and ordering.

## Precision and fidelity selection

The decisive frontier uses a genuine 31-token layer-14 activation and two
following tokens. It records full output, query, and K/V PCC so residual
magnitude cannot hide an attention regression.

| Policy | Prefill PCC | Decode 31/32 PCC | Query 31/32 PCC | Final K/V PCC | Decode | Decision |
|---|---:|---:|---:|---:|---:|---|
| MLP BFP4, attention BFP8 HiFi2 | 0.99995269 | 0.99939923/0.99931777 | 0.99994936/0.99995227 | 0.99988989/0.99987223 | 0.782718 ms | Correct higher-precision control |
| MLP BFP4, attention BFP8 LoFi | 0.99995029 | 0.99937439/0.99930600 | 0.99991915/0.99992543 | 0.99986596/0.99983533 | 0.782922 ms | Same-dtype attention fidelity control; slower |
| Attention BFP4, MLP BFP8 | 0.99997051 | 0.99967856/0.99966209 | 0.99732482/0.99740638 | 0.99676925/0.99552013 | 0.969488 ms | Correct, MLP too slow |
| All BFP4, attention HiFi2 | 0.99993527 | 0.99916996/0.99905481 | 0.99732482/0.99740638 | 0.99677802/0.99555735 | 0.774677 ms | Same-dtype attention fidelity control; slower |
| All BFP4/LoFi | 0.99993461 | 0.99917049/0.99904047 | 0.99732482/0.99740638 | 0.99677802/0.99555735 | **0.774206 ms** | Selected precision/fidelity |

MLP BFP4 HiFi2 versus LoFi was separately isolated earlier at
0.785517 versus 0.783240 ms; LoFi wins. Every target-activation metric above
the functional 0.99 bar passes. The earlier random-input sensitivity failure
is retained as a diagnostic, not used as the OPT-012 selection veto.

`results/precision_frontier_seq31/recorded_seq31_precision_frontier.json`
SHA256 is
`51f686712c1103e3f36cd0b1e769f527c8d032e28a8da60b40c227e439883014`.
Thus the selected policy is BFP4/LoFi for every dominant projection and BFP8
for the KV cache.

## Layout, topology, and program configs

| Candidate | Result | Decision |
|---|---|---|
| Maskless decode SDPA | 0.782570 ms versus explicit mask 1.003755 ms | Selected |
| Exact advisor residual chain | 0.786988 ms; report-sharded inputs 0.788957 ms | Correct, slower |
| Wider MLP blocks | 0.807852 and 0.801090 ms | Slower |
| Packed advisor MLP | 0.824883 ms after legal L1-interleaved repair | Earlier correct control |
| DRAM packed BFP8 | 1.163505 ms after CB/divisibility repairs | Earlier correct control |
| Final DRAM-48 all-BFP4 split/packed | 0.773380/0.938405 ms | Split selected |
| Final advisor/DRAM-48 all-BFP4 | 0.773663/0.773380 ms in 100-replay topology run | Batch-32 tie; DRAM wins primary batch 1 |
| Final down-input unaligned/aligned | 0.773251/0.768429 ms, identical PCC | Aligned; removes one reshard |

Final-policy like-for-like prefill controls:

| Config | Prefill PCC | Prefill | Decision |
|---|---:|---:|---|
| Grid 8, block 8, all BFP4 | 0.99998189 | 3.546279 ms | Slower |
| Grid 11, block 8, all BFP4 | 0.99998189 | **3.258952 ms** | Selected |
| Grid 11, block 1, MLP BFP4 control | 0.99998531 | 4.003834 ms | Slower |
| Grid 11, block 2, MLP BFP4 control | 0.99998608 | 3.484926 ms | Slower |
| Grid 11, block 4, MLP BFP4 control | 0.99998626 | 3.413339 ms | Slower |
| Grid 11, block 16 | n/a | n/a | Hard device limit: 1,729,280-byte CB exceeds 1,572,864-byte L1 |

The raw block-16 error is in `results/block16/`; it was not rejected on the
first API error. Prefill chunks rows above 1024 inside the MLP, preserving
length 128 and all non-aligned logical lengths without a public modulo rule.

Selected DRAM-sharded role configs are derived and exercised as follows. The
32/48 counts describe activation/output shard geometry, not total active
kernel participants (`tt-perf-report` reports 80 participants). All weights
are 8-bank DRAM width-sharded; `per_core_M=1` for batch-32 tile-padded decode.
The DRAM-sharded TTNN config type does not expose explicit output-subblock
fields.

| Role | Grid / shard count | `in0_block_w` | `per_core_N` |
|---|---:|---:|---:|
| QKV | 8x4 / 32 | 3 | 5 |
| O | 8x4 / 32 | 3 | 3 |
| Gate/up | 8x6 / 48 | 2 | 15 |
| Down | 8x6 / 48 | 15 | 2 |

The rejected 24-core MLP control uses grid 8x3: gate/up
`in0_block_w=4, per_core_N=30`; down
`in0_block_w=30, per_core_N=4`. Residual/norm stays 32-core L1
width-sharded.

Artifacts and hashes:

- 24-candidate frontier: `results/candidates/candidate_sweep.json`, SHA256
  `a54d293b2164ee5215fd2b11e559693a410cf35187c199efdc0549c380815045`
- Isolated prefill tuning: SHA256
  `8c8438be415737f71e11cf0315f2a2c5036ac906110ecb53704911e8412eed75`
- Recorded fidelity controls: SHA256
  `74c4cd79119a2cd054b58e279b74159008f503c853f63c6c96dc9081859eaf77`
- Block-16 structured failure: SHA256
  `04f117d5be3cb3796b4c0edf8669842ae82e0cdf022a8df2f9e9b77abed5a840`
- Final-policy topology cross:
  `results/final_policy_topology/candidate_sweep.json`, SHA256
  `b62744bf9dda6896e5d0166775aa808ac56151482c5b25fcfd60c4db024e2aa4`
- Down-input alignment A/B:
  `results/down_input_alignment/candidate_sweep.json`, SHA256
  `5c771764a23a53674999f2a8682c5f61428c35c7100c5b9cd508c194f1602f04`

## Final same-harness result

| Batch | Path | Prefill PCC | Decode PCC | Warm prefill | Traced decode |
|---:|---|---:|---:|---:|---:|
| 32 | Functional BF16 | 0.99999636 | 0.99999972 | 4.148475 ms | 1.797630 ms |
| 32 | DRAM BFP8 control | 0.99999637 | 0.99999995 | 4.530043 ms | 1.134656 ms |
| 32 | Advisor all-BFP4 control | 0.99998189 | 0.99999842 | 3.274530 ms | 0.773084 ms |
| 32 | **Final DRAM-48 all-BFP4/LoFi** | 0.99998189 | 0.99999839 | **3.262737 ms** | **0.768483 ms** |
| 1 | Optimized BF16 advisor | n/a | 0.99999995 | n/a | 1.402281 ms |
| 1 | Advisor all-BFP4 control | n/a | 0.99999882 | n/a | 0.652849 ms |
| 1 | DRAM BFP8 control | n/a | 0.99999994 | n/a | 1.009907 ms |
| 1 | **Final DRAM-48 all-BFP4/LoFi** | n/a | 0.99999881 | n/a | **0.644047 ms** |

The aligned-down-input change makes DRAM-48 0.6% faster than advisor at batch
32 and 1.3% faster on primary batch 1. Both final tests use 100 trace replays
and assert these orderings. At batch 32 the final default is 21.4% faster in
prefill and 57.2% faster in decode than functional BF16; it is 28.0%/32.3%
faster than DRAM BFP8.

- `final_batch32.json`, SHA256
  `bfe83404db7f417347272686a5f0c787baada0f8667ddbf5a95d0235ad6f8167`
- `final_batch1.json`, SHA256
  `89102f1ea3ba4d5263aca08b48936574bd59073a085b6fb9a0c2f8f3b86d5b1b`

## Profiler and roofline accounting

The post-fix Tracy capture contains two warmed prefill iterations and three
trace replays in one process, delimited by `PERF_PREFILL`/`PERF_DECODE`
signposts. Explicit `ReadDeviceProfiler` boundaries prevent marker loss. The
test records wall timing in the same run; `profile_summary.json` reconciles it
with `tt-perf-report` device and gap columns.

| Phase | Device ops/iter | Host ops | Kernels/iter | Gaps/iter | Device+gap | Wall/iter | Ideal lower bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 215 | 0 | 2.688721 ms | 1.028796 ms | 3.717517 ms | 3.836255 ms | 0.714702 ms |
| Decode | 42 | 0 | 0.758335 ms | 0.034263 ms | 0.792598 ms | 0.801771 ms | 0.437251 ms |

The signpost spans are 3.938836 ms prefill and 0.860991 ms decode because
signpost emission is outside the timed loop. This capture does not mix numbers
from other processes. `tt-perf-report` attributes prefill matmuls to DRAM
interleaved input/weights and decode matmuls to L1 width-sharded inputs plus
DRAM width-sharded weights. Its dominant matmul DRAM utilization is about
37--41% prefill and 34--54% decode; the saved human-readable reports include
the complete table and advice.

Prefill remains limited by large linears plus the per-user cache-fill contract.
Decode remains dominated by QKV/O and gate/up/down linears. The profile proves
all five dominant linears use BFP4+LoFi. It records no host op, torch
conversion, readback, or functional fallback. Required shard/head,
cache, residual, and public-output boundaries are visible and were A/B tested.

The selected decode has four reshards per replay (first replay IDs
976/978/981/986): concat-head output into the 32-way O input; O output into
the 32-way residual chain; 32-way post-attention norm into the 48-way MLP;
and 48-way down output back into the 32-way residual. These are required
head/O or 32-to-48/48-to-32 boundaries. The prior fifth gate-product-to-down
reshard was within one 48-way family and was removed by sharing the exact
physical shard map.

- `tracy/dense_layer/ops.csv`, SHA256
  `8d97aeceb81ff47df0699c6d311b9c19fe63e8297d848c9284dc6b32c0543ef2`
- `results/final/profile_summary.json`, SHA256
  `a570b19905dce11e5c046b102103f70427133895612f36bff6257154f2ab1f82`

## Correctness, stress, watcher, and context

- Normal full-module command: `5 passed, 9 skipped in 24.31s`; manual perf,
  profiler, frontier, and diagnostics are explicitly environment-gated.
- Selected recorded prefill PCC is 0.99998189 at length 17, 0.99993461 at
  non-aligned length 31, and 0.99980626 at length 128.
- Recorded sequential decode PCC is 0.99916996/0.99905190 at positions 31/32;
  query is 0.99733524/0.99741886 and final K/V cache PCC is
  0.99678000/0.99555947 in the watcher run.
- Eight repeated same-slot decode writes and trace replays are bitwise
  deterministic.
- Every public prefill/eager/warm/trace output asserts DRAM interleaved.
- `TT_METAL_WATCHER=10` semantics/cache/repeated decode passed in 9.66 seconds,
  detached both devices, and has no error/assert/NoC-hang/timeout signature.
  `watcher.log` SHA256 is
  `2fa358bddee8c9692cce73123c7f6be8319805c8de73c52d82b9106c84b073cf`;
  raw device watcher log SHA256 is
  `cf187e21da8af34bb58bbc686466ba9dde12990999cae8f9994e1856cf20e6d2`.
- Post-watcher `tt-smi` reports both paired devices DRAM-healthy at 33.1/34.7 C.
- `doc/context_contract.json` remains cache length 128. BFP8 cache dtype reduces
  storage; persistent phase-specific decode weight copies increase weight
  storage, but selected-default max-length validation passes and no advertised
  capability changed.

An earlier all-tests process mixed the module-device fixture with trace-region
tests and hit an invalid context ID after five passes. The transcript is
preserved under `results/anomalies/`. The harness fix environment-gates manual
trace tests; the normal full module now passes in one command. A subsequent
health check and all acceptance processes close/detach cleanly.

Nanobind prints a process-exit reference-leak warning after device close in
these tests. Every command exits after clean device detach; there is no device
allocation leak, watcher error, or retained process. This binding warning is
preserved in all raw consoles rather than hidden.

## Reproduction commands

```text
TT_VISIBLE_DEVICES=2,3 pytest \
  -q -s models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py

python models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/capture_activations.py

TT_VISIBLE_DEVICES=2,3 FALCON3_RUN_FINAL_PERF=1 \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/results/final \
pytest -q -s models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_candidates

TT_VISIBLE_DEVICES=2,3 FALCON3_RUN_FINAL_PERF=1 \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/results/final \
pytest -q -s models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py::test_batch1_traced_decode_candidates

TT_VISIBLE_DEVICES=2,3 FALCON3_RUN_RECORDED_PRECISION_FRONTIER=1 \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/results/precision_frontier_seq31 \
pytest -q -s models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py::test_recorded_seq31_precision_frontier

TT_VISIBLE_DEVICES=2,3 FALCON3_RUN_CANDIDATE_SWEEP=1 FALCON3_REAL_WEIGHTS=1 \
FALCON3_INPUT_SOURCE=recorded FALCON3_CANDIDATE_DECODE_ITERATIONS=100 \
FALCON3_CANDIDATES=advisor_all_bfp4,advisor_all_bfp4_grid8,dram_all_bfp4_48c,dram_all_bfp4_48c_packed \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/results/final_policy_topology \
pytest -q -s models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py::test_decode_candidate_sweep

TT_VISIBLE_DEVICES=2,3 FALCON3_RUN_CANDIDATE_SWEEP=1 FALCON3_REAL_WEIGHTS=1 \
FALCON3_INPUT_SOURCE=recorded FALCON3_CANDIDATE_DECODE_ITERATIONS=100 \
FALCON3_CANDIDATES=dram_all_bfp4_48c,dram_all_bfp4_48c_unaligned_down \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/results/down_input_alignment \
pytest -q -s models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py::test_decode_candidate_sweep

TT_VISIBLE_DEVICES=2,3 TT_METAL_WATCHER=10 \
pytest models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py::test_selected_decoder_semantics_cache_and_repeated_decode -vv -s

TT_VISIBLE_DEVICES=2,3 FALCON3_RUN_PROFILE=1 \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/results/final \
python -m tracy -r -p -v \
  -o models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/tracy/dense_layer \
  -m pytest -q -s \
  models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_decoder.py::test_profile_selected_decoder

tt-perf-report models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/tracy/dense_layer/ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --tracing-mode --no-color --no-summary \
  > models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/tracy/dense_layer/prefill_perf_report.txt

tt-perf-report models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/tracy/dense_layer/ops.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --tracing-mode --no-color --no-summary \
  > models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/tracy/dense_layer/decode_perf_report.txt
```

Watcher and profiler were never enabled together.

## Optimize checklist

| Requirement | Status and evidence |
|---|---|
| Functional PCC, cache, trace, fallback, stress, watcher | Complete; core, final, repeated, and watcher evidence above |
| Operation-topology audit before knob tuning | Complete; packed QKV/composite SDPA retained, mask graph removed, packed MLP repaired and measured |
| Mandatory shard advice this pass | Complete; report/IR hashes, exact applied/rejected decisions, runner gate pass |
| Recorded activations with real weights | Complete; reproducible HF layer-14 fixture and metadata |
| Attention/MLP precision and group-isolated fidelity | Complete; recorded output/query/cache frontier and same-dtype group controls; random inputs retained as diagnostics, non-aligned target activations validated |
| Large prefill programs and valid logical lengths | Complete; grid 8/11, blocks 1/2/4/8/16, internal row chunking, lengths 17/31/128 |
| Decode sharding and DRAM-sharded BFP4 geometries | Complete; AutoFix diagnosis/repair, correct 24/48-way shard controls, DRAM-48 selected |
| Memory/program/compute configs | Complete; advisor/DRAM configs, residual/input layouts, masks, fidelities, grids, blocks, packed/split, and down-input alignment |
| Final default beats strongest correct baseline | Complete; strict assertions against advisor plus functional/BF16 and DRAM BFP8 controls at batch 32 and batch 1 |
| Runtime dtype reaches dominant ops | Complete; final profile proves QKV/O/gate/up/down are all BFP4/LoFi |
| No unnecessary host/layout fallback | Complete; source ownership guard, zero host ops, fifth reshard removed, four required boundaries classified |
| Same-run roofline/device/gap/wall | Complete; repeated marker-clean capture and structured reconciliation |
| Context/capacity preserved | Complete; cache length 128 and arbitrary logical lengths retained |
| MoE, collectives, CCL, LM head, sampling | Not applicable to this dense single-device decoder stage |

## Review and commit administration

The first independent review found four issues: public output remained L1,
advisor residual claims were unmeasured, raw final-selection artifacts and
assertions were insufficient, and profile accounting mixed runs. All four were
fixed and the second review confirmed those remediations.

The second review then required recorded activations, isolated group fidelity,
a like-for-like grid comparison with raw block-16 evidence, and AutoFix for the
near-zero DRAM-BFP4 result. All are addressed above. The fresh final stage
review returned `clean-pass` with no required work. It independently confirmed
the advisor gate, fallback-free selected path, traced-decode wins at batch 32
and batch 1, clean watcher/device detach, zero-host-op profile, arbitrary
logical lengths through the advertised 128-token context, documentation, and
the final optimize checklist. The stage is committed locally; no push is
performed.
