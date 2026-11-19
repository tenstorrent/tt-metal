# Final-v1 fixture audit (CPU/source only)

The requested final stack comparisons have matching capacities/RNG schedules. The target B32 four-step profile is correctly pinned to context5152 and can reuse the first four decode outputs from the1024-step reference. **Two actionable launch issues remain:** `multichip_probe` currently forces old chunk_size1024 despite final job arguments omitting `--chunk-size`; the optional128K/1024-answer reference has not yet been generated. No device jobs ran and no existing files were edited in this audit.

## Stack RNG and reference contract

`tests/multichip_packed_stack.py:59` seeds Torch119. Capacity at line198 is:

```text
max(context, ceil((prefix_length + 2 + 18 + steps)/32)*32)
```

It draws B independent `randperm(capacity/32)` tables at line205 **before** any hidden input. Prefill hidden inputs are regenerated in part-major then request-major order at lines208–213; decode inputs are regenerated in step-major then request-major order at lines291–294. `check` stores output tensors only (line112); line375 saves that output dictionary. `--sample-prefill`, state checks, profiling and timing do not add Torch RNG draws. Real `weights()` reads safetensors; neither decoder setup contains Torch random draws.

CPU mmap inspection confirmed `state_stack_32.pt`, `answer1024_state_stack_32.pt`, `dram_prefill_boundaries_0.pt` and `long_0.pt` contain output/state keys, **not input fixtures or RNG metadata**. Consequently, changing capacity, batch, prefix, length-list order, or refresh RNG consumption can invalidate a reference even when labels still match. The packed and original `multichip_stack.py` harnesses have the same seed/table/prefill/decode RNG order.

| Final job | Reference provenance | Effective capacity | Verdict |
|---|---|---:|---|
| `profile_target32`, prefix4095/steps4/context5152 | `answer1024_reference_32_command.json`: prefix4095/steps1024, implicit context1152; result confirms5152 | 5152 / 5152 | Matched. First4 decode outputs are valid. |
| `answer1024_32` and `_1` | Corresponding answer1024 reference, prefix4095/steps1024 | 5152 / 5152 | Matched. |
| `extended_long_32`, prefix4095/steps128 | `long_stack_32_reference_command.json`, original stack harness; result confirms4256 | 4256 / 4256 | Matched RNG order and capacity. |
| `extended_context64k_32`, prefix65535/steps128 | `context64k_reference_32_command.json` explicitly uses context65696 | 65696 / 65696 | Final implicit capacity equals reference's explicit capacity. |
| Boundary B1..32, prefix31/steps16 | Corresponding state-stack references, default context1152 | 1152 / 1152 | No capacity change. |
| Optional `context128k_answer1024_32` | `context128k_answer1024_reference_jobs.json` has exactly matching prefix131071/context132128/steps1024 | 132128 / 132128 | Commands match, but reference/result absent at inspection; generate successfully before final launch. |

For the target profile, a CPU-only replay of seed119 plus32 table permutations gives identical RNG-state SHA256 for both final profile and answer reference: `4733ff7ec49a2c087fe1673b69b136cbb750a50c5b977f1361426f7a8ab18115`. Thus identical prefix input shapes/orders yield identical prefill inputs and the first four decode inputs. Removing `--context 5152` would instead select4128 and invalidate the fixture. **Keep that explicit context.** The profile intentionally omits `--state-checks`; adding that flag while retaining the1024-step reference would compare four-step terminal state against1024-step terminal state. Use a separate four-step reference with context5152 if final-state checks are desired in the profiler job.

The optional128K reference path `generated/multichip/context128k_answer1024_state_stack_32.pt` and result `context128k_answer1024_reference_32.json` did not exist at inspection. Its generation list is prepared, so this is a dependency to complete, not an RNG mismatch.

## Max-context and large-prefill fixtures

`multichip_long.py:29`–`35` uses seed91, n from the unchanged target config262144, generates one BF16 hidden tensor `[1,1,n,4096]` **before** the page permutation, and tests n−5/n−1/n plus terminal decode. `doc/multichip_decoder/baseline_long_{0,3}.json` confirms lengths262139,262143,262144 and decode position262143. Final max-context jobs use that same harness/config without an explicit chunk override; the RNG contract matches. The `.pt` file contains sampled outputs only, so changing n or the random draw dtype would require fresh references.

The final large-prefill jobs use exactly the same ordered length list as executed `dram_prefill_reference_{0,3}_command.json`:

```text
8191,8192,8193,16383,16384,16385,32767,32768,32769,65535,65536,65537
```

All use steps4, seed73 and effective capacity65632 (2051 pages), from `multichip_probe.py:182`. The baseline draws one page permutation before the length loop, then for each length one prefill input and four decode inputs. Final trace/warmup/repeats/benchmark options consume no extra Torch randomness; refresh is absent. Therefore both final large-boundary and large-prefill Watcher lists match their references. Do not truncate/reorder this list against the same reference: both capacity and previous lengths' input draws affect later inputs.

The smaller final contract list also matches `baseline_final_{0,3}.json`'s ordered labels (1,2,3,4,31,32,33,1023,1024,1025,2051), steps8, trace+refresh behavior. Refresh draws three extra random hidden inputs **and** page permutations after each length (`multichip_probe.py:329`–`351`); preserving those flags is essential. Final stress uses length128/steps128 as its saved baseline.

## Concrete final-default chunk blocker

The final-v1 probe job commands contain no `--chunk-size` override. **That is currently insufficient:** `tests/multichip_probe.py:27` sets parser default1024, and line169 unconditionally forwards `chunk_size=args.chunk_size` to every non-baseline constructor. The pending production defaults32768(linear)/16384(full) is therefore masked by this harness. Large-boundary, contracts, per-layer perf/profile, stress and large-prefill Watcher jobs would still instantiate chunk1024, contradicting a final-default-path claim for large prefill. Max-context `multichip_long` and packed-stack jobs do not inject this override.

Minimal recommended harness change before launching final jobs: make `--chunk-size` default `None`; include the constructor keyword only when explicitly provided; report `layer.chunk_size` as the effective value (optionally record requested override separately). Explicitly passing the kind-specific chunk in every final command would exercise those values but would not test omission/default behavior as directly. The baseline branch already omits this keyword, so preserving that behavior does not change reference fixtures. Changing chunking itself does not consume RNG.

## Recorded inputs and prevention

The recorded-HF jobs are different: `multichip_final_recorded.py` uses current production, while `multichip_recorded_batch.py:32` loads cases containing actual `prefill_input`, decode inputs and HF outputs. Random page permutations at line84 do not shift those stored hidden inputs. These fixtures are not susceptible to the regenerated-hidden-input capacity coupling, although valid cache mappings/capacities still matter.

For future generated fixtures, persist seed/batch/prefix/effective capacity/ordered lengths/steps/refresh metadata and assert it before device initialization. To make capacity-independent hidden inputs, use separate page-table and activation Torch generators, but that changes existing reference input streams and requires new references; do not silently introduce it in final validation. No gate relaxation is needed for the current matching jobs.

## Inspected source hashes

- `tests/multichip_packed_stack.py`: `cfc2449e04fd0982c540bc90bfba08c5e24d9d10fd864a1f5ac15edff054b0b4`
- `tests/multichip_probe.py`: `45ef312bf71b93b9f994511f9452fb1660d2164eaf99b5da14f52cb779a56dd0`
- `tests/multichip_long.py`: `a4bf45daac7750dfb729246678628a79c6ba899310e2d54fa828baff988b8cb9`
- `doc/optimized_multichip_decoder/final_v1_context128k_answer_jobs.json`: `cc420d264ed9af0eaaee361a3a4cf2e25a2a38ea4e725d9c826cbfeddcf10553`
- `doc/optimized_multichip_decoder/final_v1_gates_jobs.json`: `c4bb6e3ba020ae42461edac6ee1027f841052c65c5dbc94491dca109c54a98f6`
- `doc/optimized_multichip_decoder/final_v1_kernel_jobs.json`: `4d17d8f03ef95c176256d57a5243edacf26992e058ab69e46f1dacbc7686b9b1`
- `doc/optimized_multichip_decoder/final_v1_perf_jobs.json`: `941887ead22f63a525b68fa5b3e39d6ffbe34272cc9dcee99718b51a20cf9c62`
- `doc/optimized_multichip_decoder/final_v1_profile_jobs.json`: `3ef66d5b844d12d53b0aa708010d45a521d61752c0a1ff5c76e29a8f8492b0a2`
- `doc/optimized_multichip_decoder/final_v1_recorded_jobs.json`: `cdc29a6b730f2887ba64bbd2ea693882ae0947492b9f7ebdf932153a1d27fcfa`

Owner reconciliation: the already-prepared `final_integration_audit/base_integration_5cdc.patch` includes precisely the None/conditional-keyword/effective-value repair. It remains unapplied while candidate queues use the current source. The selected large-prefill candidates are32768linear/16384full;8192 was an earlier measured control. Final jobs launch only after applying the patch.
