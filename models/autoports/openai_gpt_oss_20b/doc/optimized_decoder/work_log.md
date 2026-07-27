# Optimized decoder work log

Date: 2026-07-25 UTC

## Starting state

- Branch: `mvasiljevic/gpt-oss-pipeline-progress`
- Starting commit: `225f996e2f05a8639ea78580eddb2b43ac8a70a4`
- No `tt/fused_decoder.py` existed, so `tt/functional_decoder.py` is the
  required source.
- The worktree was clean before stage-owned files were added.
- `timeout 60 tt-smi -ls --local` enumerated four Blackhole P300c devices.
- A bounded `TT_VISIBLE_DEVICES=2,3` 1x1 mesh open/close passed with
  `Arch.BLACKHOLE`; no reset or recovery was needed.
- The complete functional suite passed unchanged: 6 tests in 7.66 seconds.

## Baseline correctness

The functional acceptance floor is PCC >= 0.99 for all paths. The current
checkout reproduced the prior stage evidence exactly:

| Path | PCC |
| --- | ---: |
| synthetic prefill S=17 | 0.9999918034 |
| synthetic prefill S=128 | 0.9999962547 |
| synthetic prefill S=256 | 0.9999969718 |
| real prefill S=17 | 0.9933185739 |
| real decode position 17 | 0.9993172396 |
| real prefill S=256 | 0.9913088292 |
| real decode position 256 | 0.9994801582 |

The position-256 test also checks K/V cache updates and the sliding-attention
residual at PCC > 0.9994.

## Initial performance baseline

Command:

```text
TT_VISIBLE_DEVICES=2,3 HF_HOME=/home/mvasiljevic/hf-cache \
python models/autoports/openai_gpt_oss_20b/tests/optimized_decoder_perf.py \
  --decoder functional
```

Real layer-12 weights, batch 1, S=17:

| Path | Warmed latency |
| --- | ---: |
| prefill, eager mean of 5 | 7.7557 ms |
| decode, traced mean of 20 | 6.1804 ms/token |
| decode, traced minimum of 20 | 6.1732 ms/token |

The final default must beat the 6.1804 ms correct traced-decode baseline in
the same harness.

## Operation-topology and graph-fusion audit

The audit starts from code inspection and will be reconciled with
`tt-perf-report` rows after profiler collection.

| Current subgraph | Repeated input or movement | Candidate | Action / evidence |
| --- | --- | --- | --- |
| RMSNorm | BF16 DRAM input/output | sharded RMSNorm/residual chain | Candidate required; measure as a coherent chain. |
| packed QKV `linear+bias` | one shared-input projection already | split Q/K/V or retained packed QKV | Retain packed topology as the candidate to beat; split is expected to add weight reads and dispatches. |
| reshape + create decode heads | QKV output crosses the head helper boundary | dedicated `nlp_create_qkv_heads_decode` | Already dedicated; audit advisor-required boundary layout. |
| RoPE Q/K | two dedicated rotary calls | fused cache-aware attention path if legal | Search the op library and GPT-OSS implementation; keep separate only with contract/perf evidence. |
| KV update + decode SDPA | cache write and attention read | paged update + explicit SDPA config/BFP8 cache | Required candidates; compare whole attention path. |
| SDPA output reshape | no explicit concat-head kernel | `nlp_concat_heads_decode` | Measure after creating its required sharded input; a first layout error is not rejection. |
| O `linear+bias` + residual | DRAM-interleaved | DRAM-sharded matmul, L1 output, sharded residual add | Required geometry/layout candidate. |
| FP32 router linear + top-k + softmax + scatter | two typecasts and dense routing map | keep exact FP32 router or lower-precision/fused candidate | GPT-OSS 32-expert router cannot use the 128-expert fused router op; precision/fidelity still must be swept. |
| repeat token across 32 experts + packed gate/up dense matmul | full all-expert compute and large DRAM intermediates | routed active-expert split gate/up `sparse_matmul` | Highest-priority topology rewrite; compare against packed dense whole-MLP latency/PCC. |
| clamp + sigmoid/multiply exact GPT-OSS SwiGLU | multiple elementwise ops | fused activation/binary form | Search op contracts; exact alpha 1.703125, clamps, and `(up+1)` must be preserved. |
| dense all-expert down matmul + routing multiply + sum | computes 32 experts although top-k=4 | sparse down (`is_input_a_sparse=True`) + fast expert reduce | Highest-priority topology rewrite; `nnz` must reflect the exact nonzero contract. |

No collectives are present or applicable on the 1x1 mesh. QKV is already
packed, causal/paged SDPA is already a dedicated composite, and attention
projection biases are already folded through `ttnn.linear`. These are retained
unless a faster correct topology is measured.

## Commands

```text
TT_VISIBLE_DEVICES=2,3 HF_HOME=/home/mvasiljevic/hf-cache \
timeout 1800 pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py
```

Result: PASS, 6 tests in 7.66 seconds.

