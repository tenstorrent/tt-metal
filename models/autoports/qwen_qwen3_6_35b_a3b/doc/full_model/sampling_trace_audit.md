# Sampling And Trace Audit

## Sampler Comparison

| Common sampler | Fit | Decision |
| --- | --- | --- |
| `Sampling1D` | 1D-sharded logits only; rejects the full model's flat 4-way vocab-sharded logits. | Rejected. |
| `SamplingGenerator` / `TTSampling` | Works with flat vocab shard when `cluster_shape=(1,4)`, `sampling_all_gather_axis=1`, `max_top_k=32`, and `ShardTensorToMesh(dim=3)` LM-head logits. | Selected. |

No custom sampler was written.

The selected sampler is greedy in the readiness path: the full model passes
top-k 1 behavior through the common `SamplingGenerator`, with `max_top_k=32`
because the common implementation needs a tile-width candidate buffer. The
greedy split-sampling comparison in `tests/test_full_model.py` compares traced
token-out against explicit host compatibility on the same synthetic model and
prompt. It passed in both fallback and watcher runs.

## Composite Top-K Gather

The first traced token-out watcher run failed inside native all-gather on the
small top-k tensor. `$autofix` reproduced the failure with:

- `logs/probe_watcher_axis_explicit_all_gather.log`
- `logs/probe_watcher_noinline_all_gather.log`

The accepted fix adds an opt-in common sampler flag,
`use_composite_topk_all_gather`. When enabled, the top-k gather tensor is
untilized to row-major, gathered through the composite all-gather path, and
tilized back before `ttnn.sampling`. This passed a standalone watcher probe in
`logs/probe_watcher_rm_composite_all_gather.log` and the final full-model
watcher smoke in `logs/watcher_synthetic_composite_gather.log`.

The flag is enabled only by the Qwen full-model sampling args. Other common
sampler users keep the previous native all-gather path.

## Trace Contract

The optimized traced decode body captures:

- token embedding from a persistent 1-wide TT decode input buffer;
- decode through all 40 multichip decoder layers;
- BF16 linear-state update into existing cache tensors;
- final RMSNorm and flat 4-way LM head;
- common greedy sampler writing the next token into a persistent tile-width TT
  sampler output buffer;
- device-side slice and copy of sampler output slot 0 into the decode input
  buffer for the next replay;
- device-side `plus_one` update of the persistent TT current-position tensor.

The generator compiles the trace with a disposable warmup cache, then releases
that trace before real generation. Real generation prefills into a fresh cache,
samples the first token on device from the last prefill logits, captures decode
plus sampling at the next absolute position, and replays the trace for the
remaining tokens.

The page table is allocated once with the cache and reused across trace replay.
Changed page-table overrides are honored by the cache wrapper and covered by a
traced synthetic decode test. There is no per-token host page-table rebuild.
The current position starts at `prompt_len` for the first decode token after
prefill sampling and reaches the expected exclusive end position after
generation.

Evidence from `logs/token_out_trace_perf_default_prompt_100.log`:

- `prompt_len=59`
- `num_tokens=100`
- `decode_tokens=99`
- `trace_present=true`
- `trace_generated_steps=99`
- `position_end_expected_exclusive=159`
- `trace_decode_t_s_u=16.422862858524724`

Teacher-forcing readiness also uses traced low-level decode when
`enable_trace=True`; the host callback only supplies the next reference token
between trace replays. Evidence:
`logs/run_teacher_forcing_aime24_chat_100.log`, top-1 `99/100`, top-5
`100/100`, top-100 `100/100`, decode `16.35 t/s/u`.

## Terminal Path Profile

Terminal-path profiling is recorded in
`artifacts/terminal_path_profile_summary.json` and the Blackhole-normalized
`tt-perf-report` files under `tracy/terminal_path_reports/`.

| Window | Wall time | Dominant ops |
| --- | ---: | --- |
| Final norm + LM head | `0.510 ms` | `MatmulDeviceOperation` `385 us`, `LayerNormDeviceOperation` `26 us` |
| Sampler | `10.938 ms` | `TopKDeviceOperation` `10,608 us` |
| Terminal subpath | `11.464 ms` | sampler `95.4%` of terminal subpath |

The sampler dominates the terminal subpath, but it does not dominate full
token-out decode: the traced token-out decode step is `60.891 ms/token`, so
the measured sampler cost is `18.0%` of token-out decode.

## Host Boundaries

The optimized measured token-out path reads only the selected active token ID
after on-device sampling so the Python caller can return generated IDs. It does
not read full logits, does not run host argmax, and does not feed tokens back
through a Python-controlled decode loop.

Full logits are materialized only for `run_prefill_check`,
`run_teacher_forcing`, and explicit `host_sampling_compat` tests.
