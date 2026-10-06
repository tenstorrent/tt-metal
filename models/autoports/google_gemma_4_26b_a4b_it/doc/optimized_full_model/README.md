# Gemma4 optimized full model — Stage07

Warmed 4096-input / 128-output / batch-1 / concurrency-1 performance:

| Path | TTFT | Token-out decode |
| --- | ---: | ---: |
| Baseline | 2114.794 ms | 49.2561 tokens/s/user |
| Selected default | **2113.193 ms** | **49.3541 tokens/s/user** |

The selected result is the median of three warmed requests. The throughput change
is small (~0.20%); treat performance as essentially flat. Decode readbacks fall
from 127 to one final transfer. Traced teacher-forcing is separately 49.9661
tokens/s/user on the 161-input / 100-position AIME24 check, with caller token
injection and readback.

Status: **clean-pass** — independent review in `stage_review.md`.

Target google/gemma-4-26B-A4B-it revision4d7ae4984b7db7de8f8457170b3f1a419ee76d52,
four Blackhole P300c ASICs,1x4 mesh, starting checkpoint941b23a758.

The complete30-layer path preserves the accepted Stage05 decoder policy and
its measured topology/precision rejection ledger. See
`../optimized_multichip_decoder/README.md`, `candidate_comparisons.md`,
`residual_contract.md` and `final_perf_findings.md` there. Embeddings remain
hidden-sharded with one entry gather, and the terminal head remains vocab-sharded.
Layer-to-layer residual layout is the accepted replicated BF16 layout; no new
inter-layer collective or replication is introduced.

## Generator change

Fixed-length device sampling (`generate(..., stop_on_eos=False)`) records token
outputs into persistent device storage using a third small recorder trace.
Model and common sampling remain separate traces with `tt_out_tok` feeding the
next decode token on device. Position/RoPE advance, unchanged-page-table behavior,
pooled persistent CCL buffers and private layer semaphores remain intact.
The loop submits nonblocking traces and performs one final output transfer.
The first-token transfer is part of TTFT. EOS stopping and teacher-forcing are
explicit caller-interactive paths and retain their required readbacks.

`buffer_tokens=False` selects the original per-token-readback control. Changing
buffer capacity releases/rebinds traces before allocation; shorter generation
windows reuse capacity. No vLLM work is included.

## Terminal decisions

`sampler_comparison.json`: semantically greedy split sampling508.518us vs
force-argmax3300.512us, isolated warmed host-wall timing. Physical top32 per shard
keeps candidate tensors tile-sized, while normalized k1/p0 gives greedy semantics.
Both tested B1 known-winner cases pass. Keep split sampling; no full-vocab gather.

`terminal_lm_head_candidates.csv` and matching JSON/logs compare whole LM-head
paths at unchanged BF16/HiFi4. DRAM-sharded chunks include input reshard, output
conversion, padding removal and concatenation. Selected11x10 interleaved K4 measures936.85us vs K8 control955.30us.
The best adapted DRAM candidate is1245.46us and loses; wider K blocks through88
work after chunk adaptation but remain slower. All44 candidates/controls and
exact L1 failures are retained. See `terminal_lm_head_plan.md`.

## Validation and limitations

See `work_log.md` for exact commands and `AUTOFIX.md` for the verified output-boundary
hypothesis. Full all-layer prefill top1/5/100=.96/1/1 and traced teacher
forcing=.94/1/1 (`readiness.json`),100 positions from one AIME24 chat prompt.
Before/after accuracy uses the same 100 reference continuation positions:

| Phase | Stage06 top1 / top5 / top100 | Stage07 top1 / top5 / top100 |
| --- | --- | --- |
| Prefill | 96% / 100% / 100% | 96% / 100% / 100% |
| Teacher-forcing decode | 94% / 100% / 100% | 94% / 100% / 100% |

Sources: `../full_model/readiness.json` and `readiness.json`.
The same verified model/revision/tokenizer reference is used before/after; this
is not a full AIME dataset benchmark. All six shared chat outputs match Stage06
exactly and each buffered128-token prefix matches its EOS-stopping control.
Actual text was read with HF controls (`qualitative_verdict.md`). Refreshed
128-token sky explanation and degeneracy check pass (`autoregressive/`,
`degeneracy.json`). Raw repeated-document stress is separate from chat quality.

Reduced real-path Watcher checks pass with trace allocation tracking, including
seeded top-k/top-p, request isolation, buffer growth/reuse, zero/single output,
non-aligned prompts31/32/33,127/129,1023/1024/1025 and4097, plus no-host-boundary
audits (`buffered_extended_watcher.json`). Ethernet Watcher instrumentation alone
is disabled because its28464-byte program exceeds the26624-byte ACTIVE_ETH
config buffer before model execution; worker assertions remain enabled.
List/reset/list and mesh smoke succeeded before the scoped retry.

The4096/128 full default runs produce exactly the same tokens as the current
per-token-readback control. All127 model, sampling and recorder replays are
nonblocking; token/position/RoPE/page-table refreshes and explicit synchronizations
are zero within the decode window. One final buffered output read remains,
included in timing (`performance_comparison.json`).

Full all-layer device timing is not inferred from a reduced profile. The profiling
contract prohibits all-layer collection; full-model device-time and roofline
percentages remain unknown unless matching safe measurements exist.

Profiler tables, CSVs, native-window provenance and checklist/advice dispositions
are in `profile/` and `final_perf_findings.md`. The reduced two-layer decode
window is 3082.769 us; sampler and recorder trace spans are at most 514.067 us
and 9.821 us. These diagnostic spans are not full-model timing.

## Stack and capability accounting

The unchanged layer counts are25 sliding and5 full attention. Stage05 warmed
host replay medians give a19.843ms/token stack diagnostic. Current
full token-out is20.262ms, only2.11% above that stack
even before allowing terminal work; the10–15% overhead trigger is not reached.
This cross-run host-wall comparison is not a device-time subtraction.
`layer_stack_comparison.json` records sources and an approximate DRAM roofline
of2.706ms based on the selected active-expert/cache policy plus
BF16 head weights. It is an estimate, not controller bandwidth or utilization.

Context remains262144; prior full-stack maximum/nonaligned capacity evidence
remains applicable to unchanged weights/cache/RoPE allocations. K4 decreases
transient matmul buffering. Recorder history plus captured scratch require32512
logical bytes/device at128 output tokens and less than64MiB at the largest
possible generation; both fit the existing2GiB trace/activation reserve.
No advertised capability is reduced. `../context_contract.json` retains the
full memory accounting and prior capacity evidence. Output capacity increases
release traces before allocation. Fixed-slot mixed prompts pass on all 30 layers at batch 32, with independently
checked slot logits (PCC >= 0.99999994). Reduced batch 3 verifies inactive-cache
isolation. B1 changed-only page tables, feedback and reset pass with allocation
tracking (`trace_full_batch32.json`, `trace_mixed_slots.json`,
`trace_page_tables.json`). The recorder changes standalone generation.

Python-only changes: no native build needed. Source pre-commit checks pass.
No vLLM adapter, broad datatype frontier search or remote push is included.

Local implementation/evidence checkpoint: `9473a2e1b18be1dc9e5fe0add73a0f62ffcc115f`.
