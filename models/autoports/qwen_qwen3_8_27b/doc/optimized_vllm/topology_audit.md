# Serving topology audit

The baseline already matches full-model token-out decode. Retain selected
`head_bfp8_hifi2`; no decoder math, precision, cache layout or trace change is
proposed. This is a serving-path audit, using source and inherited stage07/08
artifacts, with no serving profiler collection.

| Path | Current operations / movement | Candidate and constraints | Action |
|---|---|---|---|
| Prefill intermediate pieces | Persistent metadata refresh → 64 decoder layers/cache updates → select last row → norm/LM head → merge logits. Adapter discards logits. | Skip terminal and logits reset/merge only where no token is emitted. Decoder/cache behavior must be exact versus same chunk sequence. | Selected; exact-state and full mixed-traffic controls pass. |
| Prefill final piece | Same decoder → packed terminal → canonical split sampler → sampled-ID read | Retain selected head BFP8/HiFi2 and all masks. Never skip final terminal. | Unchanged. |
| Decode | Resident token/position/page tensors → nonblocking decode trace → split-sampling trace with token feedback → deferred sampled-ID read | Async split already exercised in stage09; no full-vocab gather or host greedy argmax in measured path. | Retain; regression evidence required. |
| Scheduler changes | Exact changed-table refresh; tokens/positions reload only at authoritative boundaries | Removing change detection would break allocator growth; no mutation-version API established. | Reject unchecked refresh removal on correctness grounds; inherited page-growth tests apply. |
| Decoder topology | Packed projections, fractured residual, configured fused/separate CCL families, selected BFP4 projection/LoFi, BFP8 KV | Earlier stage05/07 topology and geometry sweeps establish current math path. | Reuse `optimized_full_model/topology_audit.md`, `kernel_policy_audit.json`, and datatype selection. No math retuning for an already closed serving decode gap. |
| Off-grid continuation | Up to127 teacher-forced traced bridge steps, then eager physical bucket | Suppression does not remove terminal inside existing bridge decode trace. Changing it would add trace variants/resources. | Retain bridge correctness; limitation explicitly separate from optimized aligned chunk path. |

No persistent buffer or cache allocation changes. Native262144 context and
logical non-aligned support remain the same. Generator preparation still warms
the terminal-capable superset before any trace capture; skipping terminal ops
introduces no new program signature. This argument requires tracker-enabled
replay and mixed-traffic validation, not just source inspection.

Batched recurrent snapshot experiment: expanded persistent metadata and a count-aware copy kernel reduced metadata uploads75→15 on the112-request mixed fixture, but255.249→255.381s showed no latency gain. B16 also remained flat. Rejected; original one-row copy implementation restored byte-for-byte. Candidate source and primitive tests are archived under `rejected_snapshot_batch/`; `final_b16_*`/`final_b32_*` filenames denote this rejected candidate, not the selected final serving configuration.
