# Optimized Decoder Work Log

## Scope

Target: `models/autoports/meta_llama_llama_3_1_8b_instruct`

Stage-owned code:

- `tt/optimized_decoder.py`
- `tests/test_optimized_decoder.py`
- `doc/optimized_decoder/**`

Worktree note: `.agents/skills/optimize/SKILL.md` is dirty in the checkout but is outside this stage's ownership and is intentionally left untouched and excluded from the optimized-decoder commit.

Out of scope for this stage: multichip decoder, full model, vLLM.

## Device Setup

`tt-smi -ls --local` is unavailable in this checkout because the `tt_smi` module is missing.  Device smoke used TTNN directly:

```bash
timeout 120 python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[0])
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: passed.

## Operation-Topology Audit

Functional decode topology:

1. RMSNorm.
2. One packed QVK matmul, with layer 31 using Q/K/V order and other layers using Q/V/K order.
3. Split heads, RoPE, paged key/value cache updates.
4. SDPA decode and concat heads.
5. O projection and residual add.
6. RMSNorm.
7. Packed gate/up projection, on-device split, SiLU, multiply, down projection, residual add.

Candidate actions:

| Candidate | Action | Evidence |
| --- | --- | --- |
| Keep packed same-input QVK | Applied | Final report shows one QVK matmul `32 x 4096 x 6144`; no separate Q/K/V matmuls. |
| Pack same-input MLP gate/up | Applied | Real-weight candidate runner shows packed BFP8/HiFi2 is correct and faster than split: default packed layer0/layer31 `2.6969/2.7001 ms`, split `2.7700/2.7278 ms`. Final profiler shows one `32 x 4096 x 28672` packed matmul plus split/SiLU/multiply rows. |
| DRAM-sharded decode matmuls | Applied | Final report shows dominant matmuls as `HiFi2 BF16 x BFP8 => BF16` with 12-core DRAM-sharded program configs. |
| Explicit SDPA program config | Applied | Trace and PCC tests pass; SDPA decode row is 125-128 us. |
| Fused gate SiLU | Rejected for final default | The split gate fallback keeps fused SiLU, but the selected packed gate/up path must split before SiLU. The packed path remains faster than the split fused-gate path. |
| Decode-native QKV head creation | Rejected | Layer-31 candidate failed with `Cosine cache must cover the input sequence length. Input sequence length: 8, Cos cache sequence length: 1.` Artifact: `candidate_precision_trials.json`. |
| BFP4 linear weights | Rejected | Faster on real layer-0 weights, but failed real layer-31 PCC: MLP-only BFP4 `0.9893350005149841`; all-linear BFP4 `0.9733318090438843`. Artifact: `candidate_precision_trials.json`. |
| L1-sharded norm outputs and MLP multiply | Applied | Final code writes norm outputs directly to the projection input sharding and writes MLP multiply to the down-projection input sharding. Focused and full optimized tests pass. |
| Adapted fewer-core L1 activation grids | Rejected | Per-role 32-core and 16-core candidates were correct but did not beat the default; TTNN canonicalized requested output configs back to computed 64-core width sharding for several rows. Artifact: `candidate_precision_trials.json`. |
| Interleaved BFP8 without DRAM-sharded program | Rejected | Correct but slower than the DRAM-sharded default in the candidate runner. |
| Full prefill optimization | Not applicable | Functional source emit did not contain a TTNN prefill graph; optimized prefill preserves that explicit stub. |

## Shard Advisor Gate

Setup followed `.agents/skills/shard-advise/SETUP.md` Part B.  The first environment attempt resolved TTNN from the wrong path after sourcing bootstrap, so the successful command reset the tt-metal paths after bootstrap:

```bash
timeout 900 bash -lc 'export TTMLIR_ADVISOR_HOME=/localdev/mvasiljevic/tt-mlir; source /localdev/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh >/tmp/llama31_shard_advise_bootstrap.log 2>&1; export TT_METAL_HOME=/localdev/mvasiljevic/tt-metal; export TT_METAL_BUILD_HOME=/localdev/mvasiljevic/tt-metal/build; export TT_METAL_RUNTIME_ROOT=/localdev/mvasiljevic/tt-metal; export PYTHONPATH=/localdev/mvasiljevic/tt-metal/ttnn:/localdev/mvasiljevic/tt-metal:${PYTHONPATH}; export TT_METAL_REPO_ROOT=/localdev/mvasiljevic/tt-metal; cd "$TTMLIR_ADVISOR_HOME"; rm -rf /tmp/llama31-8b-shard-advice; ttnn-advise capture /localdev/mvasiljevic/tt-metal/models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/advise_llama31_8b.py:decode --out /tmp/llama31-8b-shard-advice'
```

Result:

```text
[ttnn-advise] ops=14 final_choices=13 spill.ran=True total_spills=0
```

Required artifacts:

- `shard_advise/report.json`
- `shard_advise/final_ir.mlir`

Advisor recommendations and decisions:

| Advisor recommendation | Decision | Evidence |
| --- | --- | --- |
| Width-shard RMSNorm, add, multiply, and dense matmul outputs across 64 L1 cores | Applied where compatible with emitted attention | Final runtime writes RMSNorm outputs to the QVK and MLP projection input sharding and writes MLP multiply to the down-projection input sharding. Residual adds remain DRAM at attention/output boundaries because the emitted attention/cache/head path consumes DRAM/interleaved boundaries. |
| Use `matmul_multi_core_reuse_multi_cast_1d @8x8` for dense matmuls | Rejected for final default | Advisor captured a dense-only block. The full decoder's measured best correct path uses DRAM-sharded matmuls with BFP8 weights. Candidate without DRAM-sharded program is correct but slower. |
| Keep reshape boundaries as DRAM/interleaved where required | Applied | Final runtime still uses reshape/head/cache boundaries around emitted attention semantics. |
| No L1 spill | Observed | `report.json` records `total_spills=0`. |

## Precision And Layout Candidate Ledger

Command:

```bash
timeout 2400 python models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/benchmark_decode.py
```

Final candidate evidence from `candidate_precision_trials.json`:

| Candidate | Layer 0 PCC / ms | Layer 31 PCC / ms | Decision |
| --- | ---: | ---: | --- |
| Default BFP8 HiFi2 DRAM-sharded, packed gate/up | 0.9998587369918823 / 2.6968657970428467 | 0.9988876581192017 / 2.7000725269317627 | Kept |
| Split gate/up BFP8 HiFi2 | 0.9998587369918823 / 2.7700455859303474 | 0.9988876581192017 / 2.727835066616535 | Rejected: correct but slower than packed. |
| Explicit packed gate/up BFP8 HiFi2 | 0.9998587369918823 / 2.7076398953795433 | 0.9988876581192017 / 2.7072155848145485 | Confirms packed default. |
| MLP BFP4, attention/output BFP8 | 0.9936819076538086 / 2.5727450847625732 | 0.9893350005149841 / 2.5625046342611313 | Rejected: real layer 31 below 0.99. |
| MLP BFP4 LoFi, attention/output BFP8 | 0.9936819076538086 / 2.6321522891521454 | 0.9893350005149841 / 2.4430811405181885 | Rejected: real layer 31 below 0.99. |
| MLP BFP4 HiFi4, attention/output BFP8 | 0.9936532974243164 / 3.022025153040886 | 0.9892237782478333 / 3.0280236154794693 | Rejected: real layer 31 below 0.99 and slower. |
| All linear BFP4 | 0.9932551980018616 / 2.5535959750413895 | 0.9733318090438843 / 2.5560500100255013 | Rejected: real layer 31 below 0.99. |
| All linear BFP4 LoFi | 0.9932551980018616 / 2.404213882982731 | 0.9733318090438843 / 2.377113699913025 | Rejected: real layer 31 below 0.99. |
| Interleaved BFP8, no DRAM-sharded program | 0.9998310208320618 / 2.900421619415283 | 0.9988491535186768 / 2.899954654276371 | Rejected: slower than DRAM-sharded default. |
| Decode-create-heads candidate | failed | failed | Rejected: layer-31 RoPE layout error. |

Geometry sweep evidence from `candidate_precision_trials.json`:

| Role | Kept `in0_block_w` | Other passing values | Rejected values |
| --- | ---: | --- | --- |
| QVK | 2 | 1 was slower | 4/8/16 fail: shard K tiles `2` not divisible by requested block. |
| O | 2 | 1 was slower | 4/8/16 fail: shard K tiles `2` not divisible by requested block. |
| Packed gate/up | 2 | 1 was slower | 4/8/16 fail: shard K tiles `2` not divisible by requested block. |
| Down | 7 | 1 was slower | 2/4/8/16 fail: shard K tiles `7` not divisible by requested block. |

Adapted L1 core-count geometry evidence:

| Role | Best adapted candidate | PCC / ms | Decision |
| --- | --- | ---: | --- |
| QVK | 32 cores, auto `in0_block_w` | 0.9998563528060913 / 2.7044561381141343 | Rejected: correct, but not faster than default layer-0 candidate at 2.6968657970428467 ms. |
| O | 32 cores, auto `in0_block_w` | 0.9998599290847778 / 2.694830298423767 | Rejected: correct but not materially faster than default, and TTNN canonicalized several requested output configs. |
| Packed gate/up | 32 cores, auto `in0_block_w` | 0.999854564666748 / 2.7200235053896904 | Rejected: correct but slower than default; 16-core variants fail L1 circular-buffer capacity. |
| Down | 32 cores, `in0_block_w=7` | 0.9998587369918823 / 2.708412396411101 | Rejected: correct but slower than default. |

Several adapted runs emitted TTNN warnings that the requested output `MemoryConfig` differed from the op-computed 64-core width-sharded output and the op used the computed config.  These adapted runs are retained as evidence, but the final default keeps the simpler 64-core width-sharded activation shape with DRAM-sharded matmul weights.

## Correctness Commands

Final optimized suite:

```bash
timeout 1200 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py -s
```

Result: 7 passed.

Watcher-clean subset:

```bash
timeout 900 bash -lc 'TT_METAL_WATCHER=10 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py -k "optimized_decode_real_weights or repeated_stress" -s'
```

Result: 2 passed, watcher attached/detached cleanly.  Artifact: `watcher/watcher.log`.

Context contract:

```bash
timeout 120 python .agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct
```

Result: `Context contract OK ... target=131072, supported=128 (DRAM-limited).`

## Performance Commands

Before/after decode:

```bash
timeout 1200 python models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/benchmark_decode.py
```

Result from `benchmark_decode_latency.json`:

- Functional eager decode: `3.8332337513566017 ms`
- Optimized eager decode: `2.6933498680591583 ms`
- Optimized traced decode: `2.688268944621086 ms`
- Optimized traced speedup over functional eager: `1.4259115551018275x`
- Trace output PCC: `0.9990221858024597`
- Trace key/value cache PCC: `0.9998999238014221` / `0.9999070763587952`

Profiler:

```bash
timeout 900 python -m tracy -r -p -v -m pytest models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py -k 'trace_replay' -s
cp generated/profiler/reports/2026_07_14_15_43_47/ops_perf_results_2026_07_14_15_43_47.csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_ops.csv
tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_perf_report.csv > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_perf_report.console.log
tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/decode/decode_perf_report.txt
```

Result: trace test passed; final signposted window contains 5 traced decode replays.

Stage-review remediation:

- Trace correctness: fixed by comparing the tensor returned by the trace-captured graph after replay and by checking the updated key/value cache slot PCC.
- BFP4 rejection: refreshed candidate evidence with real layer 31 and LoFi/HiFi4 variants; faster BFP4 candidates fail real layer-31 PCC.
- Packed gate/up: implemented packed weight loading, one widened matmul, on-device split, SiLU, and multiply. Packed BFP8/HiFi2 beats split BFP8/HiFi2 and is now the default.
- Matmul geometry: added per-role `in0_block_w` and L1 core-count override fields. Same-shard larger values fail exact TTNN shard divisibility checks; adapted 32-core/16-core layouts are correct but do not beat the default and are partly output-canonicalized by TTNN.
- Advisor L1 chain: applied compatible L1-sharded norm outputs and MLP multiply input; residual attention boundaries remain layout-constrained by emitted head/cache/SDPA topology.

## Optimize Checklist

- [x] Operation-topology audit recorded.
- [x] Packed QVK projection preserved.
- [x] Packed gate/up projection compared against split and selected.
- [x] DRAM-sharded decode matmuls applied and verified in profiler rows.
- [x] Explicit SDPA decode program config applied.
- [x] Precision candidate sweep recorded with BFP8/BFP4 evidence.
- [x] Dominant matmul `in0_block_w` geometry sweep recorded with exact op-contract failures for illegal larger values.
- [x] Adapted L1 core-count geometry sweep recorded and rejected with correctness/performance evidence.
- [x] Required shard-advise hard gate run this pass; `report.json` and `final_ir.mlir` saved.
- [x] Shard-advisor recommendations recorded as applied or rejected with evidence.
- [x] Prefill/decode semantics preserved; prefill remains explicit source-emit stub.
- [x] Paged KV-cache behavior covered by HF cache comparison, real-weight decode, trace replay, and repeated stress.
- [x] Representative layer-kind coverage includes layer 0 and layer 31 reorder behavior.
- [x] Runtime source test rejects functional fallback and host conversion calls in measured forwards.
- [x] Warmed decode before/after latency reported.  Prefill is not measured because no prefill graph exists in functional or optimized stages.
- [x] `tt-perf-report` artifacts saved and reviewed.
- [x] Watcher-clean optimized correctness run completed.
- [x] Context contract checked and left unchanged.
- [x] Stage review clean-pass.  Final rereview returned clean-pass after packed-advisor artifacts and worktree hygiene remediation.
- [x] Local commit created; final SHA is recorded in the stage handoff.
