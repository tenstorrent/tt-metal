# Sampler follow-up

The reduced allocation32 adapter smoke failed after model prefill, in shared sampling's wide row-major pad: `/tmp/qwen_native_vllm_adapter.log` reports9,024,512 bytes of static dataflow buffers against1,572,864 bytes of L1. This is not evidence of a native recurrence accuracy failure.

## Source diagnosis

The Qwen generator explicitly sets `local_topk_num_chunks=2`. Its documented TP4 contract is two32,768-wide local TopKs, physical index restoration, a64→32 local merge, and only then gathering candidates across devices. `doc/optimized_full_model/AUTOFIX.md` records the original hardware boundary-ID test.

On the current rebased tree, the chunk-plan fields and the later index/merge/gather code survive, but `forward` enters the chunk path only for `multi_step_reduction`, which is true only for a1×1 mesh. TP4 skips it and pads the complete local width instead. The surviving chunk index correction references `split_width`, which has no definition. This makes the purported opt-in ineffective on TP4 and leaves an additional latent NameError in the retained branch.

`git show 62559b467af:models/common/sampling/tt_sampling.py` already contains the broken predicate and undefined reference. These defects therefore predate this native-prefill work. The branch's earlier `doc/vllm_rerun_on_new_base/README.md` documents related rebase losses, including an orphan `_plan_local_topk_chunks` call; commit `cc6d97556db` restored that helper but did not restore this runtime entry/setup. The evidence is consistent with that class of incomplete rebase restoration; no claim is made that a newly reproduced clean rebase isolated the precise conflict decision.

The reduced test supplies temperature0/top_k1/top_p1. That is a valid greedy request. The generator deliberately defaults `force_argmax_greedy=False` because it uses the general sampler. Enabling force-argmax could bypass this failure for greedy requests but would leave the top-k/top-p path broken; no environment knob is required to make the current test semantically valid.

A separate source issue exists in the optional `QWEN36_PREFILL_PER_REQUEST=1` multi-active fallback in `generator.py`: it reads full-vocabulary rows to the host and uploads the assembled full vocabulary with `_upload`, whose mapper replicates it to every TP rank. Device sampling requires local vocabulary shards instead. A full248,064-wide tensor also explains a much larger pad than the intended62,080-wide local shard. The parent confirmed that the failing command enabled this environment flag. It is therefore an exercised defect, not a hypothetical path. Parent repaired this block by retaining per-slot logits on device, selecting each slot row, and concatenating local shards with device-created inactive zeros; host gathering now occurs only when explicitly requested by the caller. The shared sampler also rejects oversized local logits explicitly.

The three issues are distinct: (1) full-vocabulary replication in the optional per-request assembly, (2) unreachable TP4 local-chunk setup plus its undefined physical offset, and (3) a wide row-major pad allocation unsuitable for the device L1 budget. Correcting only greedy configuration or only one issue would not establish the intended general sampling path.

## Scoped repair

`models/common/sampling/tt_sampling.py` now enters the existing chunk body for either single-device multi-step reduction or explicit local chunks. It defines the physical `split_width`, applies the configured local padding plan, and preserves the existing later offset/merge/gather logic. Default single-device behavior (`local_topk_num_chunks=1`) retains its original split widths and path.

The local helper converts prefill's possible row-major input into TILE layout before padding. TopK already consumes TILE. The row-major pad factory allocates multi-row buffers proportional to vocabulary width, while tile pad uses tile-sized buffers. The helper rejects a logits width above the configured local padded width before issuing any device work, with an explicit vocabulary-sharding diagnostic.

## Verification and remaining work

Host tests cover the TP4 runtime branch reachability, padding/layout order, negative logits at physical chunk boundary IDs0/32,767/32,768/62,079, and rejection of replicated full vocabulary, in addition to the existing chunk-plan tests:

```bash
python_env/bin/python -m pytest models/common/sampling/tests/test_local_topk_plan.py -q
```

Result: **6 passed** in1.32seconds (`/tmp/qwen_sampler_host_checks.log`). Black formatting, Python AST parsing, and `git diff --check` passed; no C++ build is required. Host tests use CPU op doubles and do not prove device allocation behavior or native TopK index semantics. Parent owns the serialized hardware follow-up:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/split_topk_sampler_probe.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/vllm_reduced_target.py --batch 32
```

The first probe explicitly disables force-argmax, checks chunk boundaries and invalid-vocabulary masking, and replays the sampled-token feedback trace. It should pass before the broader adapter smoke. Sampling-only and serving tests must remain separate from the parent's profiler job.

## Device validation of the restored branch

The parent's full-model generation run then exposed another retained-call mismatch: the index candidate gather passed `dtype=ttnn.uint16`, but `_perform_all_gather` has no dtype keyword. The input index tensor is alreadyUINT16, and the native all-gather output spec explicitly uses `input_tensor.dtype()` (`ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp`). Removing that obsolete keyword is sufficient; no collective cast or API expansion is needed.

After that correction, the existing dedicated hardware probe passed:

```text
SPLIT_TOPK_SAMPLER_OK [0, 32767, 32768, 248063]
```

Command: `timeout 300 env PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/split_topk_sampler_probe.py`. Full log: `/tmp/qwen_sampler_split_repair.log`. It exercised the restored multi-device branch, padded negative logits, invalid-vocabulary masking, first-stage and merged indices, all-gathers, UINT32 output, and two traced calls with persistent feedback on every TP rank. Thus the tile-padding allocation and the complete sampler call chain are device-validated for this focused shape. The allocation32 adapter run follows separately.

The allocation32 adapter also passed after the parent’s device-only per-request assembly repair:

```text
REDUCED_INIT_OK batch=32 fused_kda_decode=3/3
REDUCED_PREFILL_OK (32,)
REDUCED_DECODE_OK
REDUCED_STALE_INPUT_OK
REDUCED_SLOT_REMAP_OK
```

Exact command:

```bash
timeout 600 env PYTHONPATH=. HF_HUB_OFFLINE=1 QWEN36_PREFILL_PER_REQUEST=1 \
  QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B \
  QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/vllm_reduced_target.py --batch 32
```

The native flat/fused-convolution outer512 implementation was the runtime default. Stale-input replay increased the replay counter from1 to2 with zero token-host, position-host, page-table refreshes or readbacks. The run closed its mesh and exited0. Full log: `/tmp/qwen_native_vllm_adapter_repaired.log`; compact recorded markers and environment: `artifacts/sampler_followup_validation.json`. Predominantly token220 output from this random-input four-layer smoke is not a full-model quality result.

Final status: shared sampler runtime repair verified by host checks, a focused TP4 sampled-token trace probe, and the allocation32 adapter integration smoke. Full-model generation and live-serving benchmarks remain parent-owned checks. No watcher or profiler was enabled in these sampler runs.
