# Canonical64 adapter source audit

Continued AutoFix validation, 2026-11-17; **not a stage-review verdict**. CPU/source inspection found and repaired one unsafe trace-selection case after reporting it and receiving parent authorization. No hardware, server or HTTP action was performed by this audit. Device verification of the repaired transition remains with the parent.

## Finding and authorized repair

The original `tt/generator.py:sampling_trace_key` inspected only active slots. Native sampling processes all32 physical rows before `_sampling_forward` restores inactive tokens and masks seed advancement. `configure_sampling` preserves inactive request parameters. A dormant physical row can therefore retain k40/64 while an active greedy/top20 request selects native32.

Concrete producer/consumer mismatch: `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp:77–81` reads each physical row's k. Its native32 branch at182–215 computes the second-face end as `start+(k-16)` and traverses that range; no native32 clamp exists. Physical row31 has second-face start1008, so k40 produces end1032 and can read offsets1024–1031 beyond its single1024-element tile. With flat32 candidate probabilities and valid p1, their sum is1 and the strict `cum_prob > p` condition need not stop the loop before those offsets. Discarding the output later does not make those reads safe. For earlier rows, the extra ranks cross into another user's row. This is a source-proven bounds mismatch, not a claim that a particular hardware failure occurred.

Authorized minimal fix applied at `tt/generator.py:586–602`: choose native64 whenever **any retained physical row** has temperature>0 and logical k>32. Temperature0 rows have effective k1 in the existing `configure_sampling` mapping and remain eligible for native32. Selection changes no parameter, seed, ownership, tensor, kernel, queue or sampling algorithm. Comments and `P2_MESH_SAMPLER_PROBE.md` now describe the physical-row bound.

A dormant wide row can keep native64 selected until an authoritative update replaces its parameters. This is the conservative safety cost of retaining all physical request state. An eventual neutral inactive parameter staging design would need its own correctness/performance evidence; this repair does not add one.

## Async/state and accounting checks

- One common max_top_k64 sampler owns persistent parameter, seed, logit and token buffers (`generator.py:100–106`, `:219–280`). Both retained traces call the same `_sampling_forward` (`:493–502`). No second RNG authority or host feedback path was added.
- Both native widths eagerly warm before any model/sampler capture; both captures are retained (`:515–569`). Capacity32 B2/B32 now has **four traces**: model_2, model_fixed, sampling, sampling_64. Capture signatures identify candidate64 and native32/64 (`:604–626`). Close iterates every retained trace (`:899–908`).
- `serving.py:349–356` drains pending outputs before remap, authoritative input, parameter or seed-state changes. Slot-state registration moves both device sampler tensors and host policy values (`serving_state.py:289–314`); active layout moves with ownership (`:359–363`). Unchanged steady submissions can remain queued. Every device replay and output snapshot is ordered on CQ0; updating the Python draw-width attribute selects the next already-captured graph and does not rewrite an earlier trace.
- Model replay counts stay in `trace_replays`; sampling counts live separately in `sampling_trace_replays`, with one aggregate increment per actual replay (`generator.py:594–602`). Prefill legitimately adds a sampling replay without a model-decode replay (`serving.py:555–560`). Audited adapter events now carry both maps, and lifecycle summarization checks native-width signatures and both-width execution when wide requests exist.
- No additional async ownership defect was identified by this source audit. CPU checks cannot prove queue/allocator safety on hardware.

## Evidence and tests

`readiness_vllm/serving_overhead/comparison.json` is PASS with all18 token sequences exact. Its scope explicitly says **canonical32**, full40, B2, capacity32/context262144, served8224 pages,128 input/128 output. Native53.8331→53.8065ms/step; adapter serial53.9568→53.9380ms; queued async2 average53.7521→53.7505ms. These small sequential differences support no material speedup claim and are not canonical64 performance evidence.

`readiness_vllm/sampling64_probe/mesh_acceptance64_v3.json` is PASS: two sampler captures/two releases, program entries35 before/after, recorded greedy and top20 token identity across widths for active1/32. However `tests/run_vllm_sampling_mesh_probe.py:207–209` skips profile k>native width and assigns one profile to all rows. It does not cover a dormant wide row with a narrow active row. The repaired selector's adapter transition therefore still needs the parent's reduced hardware gate, followed by the final shared-server workload.

CPU commands run after the repair:

```bash
python_env/bin/python -m pytest --noconftest -q models/autoports/qwen_qwen3_6_35b_a3b/tests/test_sampling_trace_selector_host.py models/autoports/qwen_qwen3_6_35b_a3b/tests/test_vllm_adapter_host.py
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/test_overhead_patch.py --applied
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/run_sampling_policy_cpu.py --source models/autoports/qwen_qwen3_6_35b_a3b/tt/serving.py --output models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/final64_serving_policy_cpu.json
```

Results: **19 host tests PASS**, **6 applied-overhead tests PASS**, **24 policy cases PASS**. New selector cases cover dormant row17/31, k40/64, greedy temperature0, active→dormant preservation, authoritative replacement returning to narrow, and per-signature replay counts without recapture. The overhead test now freezes the exact required per-step logging branch rather than the whole audit method, allowing the new sampling replay telemetry while protecting the mandatory flushed ledger.

After separate parent authorization, updated stale three-trace assertions in `tests/run_vllm_bucket_probe.py`, `tests/run_vllm_logit_determinism.py`, and `sampling_debug/run_matched_b2_overhead.py`. Each requires the expected model keys plus both sampler keys and capture count `len(expected_model_keys)+2`; repeated checks retain that expected count. Compile validation passes. Historical hardware artifacts were preserved unchanged, and no device rerun was performed by this audit.
