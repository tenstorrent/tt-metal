# Optimized vLLM work log

## Contract

- Model: `tiiuae/Falcon3-7B-Base`; exact local HF snapshot used by all runs.
- Real serving path: vLLM TT plugin -> `tt/generator_vllm.py`.
- Mesh/config: P300 TP4 1x4, `max-num-seqs=32`, `max-model-len=32768`, block
  size 32, `sample_on_device_mode=all`, 512 MB trace region, 1D ring fabric.
- Precision: selected `all_bfp4_lofi_bf16_kv` datatype-sweep policy.
- Context was not reduced.  Full context remains 32,768; final non-aligned
  37-token serving passed.

## Commands

The before and final gates used the same command:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative,benchmark \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --hf-model /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 \
  --mesh-device P300x2 --max-num-seqs 32 --max-model-len 32768 \
  --block-size 32 --sampling-profile full \
  --tt-config '{"trace_region_size":512000000,"fabric_config":"FABRIC_1D_RING","sample_on_device_mode":"all"}' \
  --additional-server-args '--chat-template models/autoports/tiiuae_falcon3_7b_base/base_chat_template.jinja' \
  --server-timeout 1200
```

Other final checks:

```bash
pytest -q models/autoports/tiiuae_falcon3_7b_base/tests/test_generator_vllm_contract.py
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_vllm_shape_timing.py
python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/tiiuae_falcon3_7b_base --scope vllm --missing-artifacts critical --json models/autoports/tiiuae_falcon3_7b_base/doc/optimized_vllm/results/after/degenerate_output.json
```

## Decisions and results

Primary 128/128/1 before -> final: TTFT 182.47 -> 183.68 ms, mean TPOT
15.888 -> 16.037 ms, p50 ITL 14.566 -> 14.573 ms, p99 ITL 15.121 ->
14.832 ms, aggregate output 58.17 -> 57.64 tokens/s, and TPOT-derived
decode 62.94 -> 62.36 tokens/s/user.  Same workload/config; flat within noise.

Secondary CI burst 100/100/32 before -> final: TTFT p50 414.44 -> 415.41
ms, mean TPOT 16.860 -> 16.876 ms, p50 ITL 15.071 -> 15.056 ms, aggregate
output 1539.74 -> 1537.89 tokens/s, TPOT-derived 59.31 -> 59.26
tokens/s/user.  This is capacity evidence, not the headline decode rate.

The adapter already returned device tensors from async decode and used
nonblocking persistent model/sampling trace replay.  This stage removed the
steady host page-table compare and restricted the deferred sampled-token read
to one replicated shard.  Scheduler reset still refreshes changed tokens,
positions, RoPE and page table; unchanged state stays device-resident.

Rejected options:

- synchronous sampled-token read: 57.5 tokens/s/user, slower;
- external-plugin immutable payload reuse: neutral, reverted;
- force argmax, generic slow sampling, eager sampling, full-logits host read:
  invalid by contract and not attempted;
- profiler collection: prohibited for this serving stage and not attempted.

Comparable optimized full model at physical batch 32 / active batch 1 /
32,768 context / `[32,1024]` table / 4,128 blocks measures 14.732 ms and
67.88 tokens/s/user caller-visible.  VLLM p50 ITL is 14.573 ms and 68.62
tokens/s/user.  The older physical-batch-1 110.38 rate is not comparable.

## Optimize checklist

- [x] Decoder model and sampling paths are traced; persistent inputs and
  nonblocking replay remain active.
- [x] Runtime host-fallback audit is clean for serving token-out decode.
- [x] On-device canonical split greedy sampling is reused; no host argmax or
  full logits transfer.
- [x] Same-harness single-user and CI burst before/after evidence recorded.
- [x] Strongest comparable full-model result measured under serving shapes.
- [x] Final default path reproduced after instrumentation was removed.
- [x] Selected BFP4/LoFi with BF16 KV policy is loaded by the measured path.
- [x] Context, non-aligned prompts, paged cache, stale-input behavior, sampling,
  qualitative output, and repeated burst serving remain covered.
- [x] No Tracy, tt-perf-report, device/adapter profiler, or ReadDeviceProfiler.
- [x] Decoder topology, sharding, CCL, SDPA, projection packing, program configs,
  and precision sweeps are inherited unchanged from the completed optimized
  full-model/datatype-sweep stages; no adapter change reopens those choices.
- [x] Serving-specific movement audit completed: only a single replicated token
  shard crosses to host, after the async boundary required by the plugin.

## Artifacts and cleanup

Evidence is under `doc/optimized_vllm/results/`.  The external vLLM worktree has
no retained changes.  Temporary instrumentation was removed before the final
run.  No vLLM or EngineCore process is intended to remain after the stage.

## Review and checkpoints

Independent fresh-subagent stage review: `clean-pass`; report in
`stage_review.md`.  There was no required work.  Its residual cautions are the
single-request resolution, intentional lack of serving profiler data, and the
documented dependence on the scheduler's steady-page-table guarantee.

Local checkpoint:

- `tt-metal`, branch `mvasiljevic/fmf/tiiuae-falcon3-7b-base`:
  `4e22b862073` (`Optimize Falcon3 vLLM async serving boundary`).
- `/home/mvasiljevic/vllm`: clean after candidate removal; no retained change,
  so no checkpoint commit was created.

Nothing was pushed.  The unrelated pre-existing untracked
`third_party/tt-metal/` directory was excluded.
