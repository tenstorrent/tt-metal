# Stage 09 plugin contract repairs

Date: 2026-09-27. Status: **host-verified; device serving remains unverified by this investigation**. Starting diagnosis: [AUTODEBUG_contracts.md](AUTODEBUG_contracts.md). The earlier `AUTOFIX.md` describes a previous environment and is not the current plugin-validation result.

The working Python was `/home/container_app_user/tt-metal/python_env/bin/python`, with installed vLLM **0.26.0** and the source TT plugin from `/workspace/tt-metal/vllm/plugins/vllm-tt-plugin/src`. The vLLM checkout base is `5ffebf4128f81ea5cf8413175eabde52cd8c8d75`. No dependency was installed, no accelerator was opened, and no server was launched by this investigation.

## Repairs retained

| Plugin file | Change and reason |
| --- | --- |
| `model_runner.py` | `_kv_cache_shape` skips its legacy TP head division only when the model explicitly declares `kv_cache_specs_use_local_heads=True`. Specs describing device-local sliding heads2/full heads1 remain valid after upstream page-size unification. Missing/false capability retains existing TP/DP behavior. `_update_states` now marks decode layout changed when a cached request appends at least one real block in any group. Its existing `build_model_input` drain applies pending tokens before host input preparation. |
| `async_decode.py` | Scheduler preflight rejects steady overlap for nonempty per-request/per-group block deltas. All-empty group lists and `None` remain eligible. |
| `input_batch.py` | Lane `apply_step_plan` marks real block growth as layout changed, activating the caller's existing drain/reset path. Cache-table construction selects the original `max_model_len` API when present, including the transitional signature that has both arguments; vLLM0.26 instead receives one ceiling-divided `max_num_blocks` count per group. |
| `platform.py` | Registers `TTAutoportGemma4ForCausalLM` to `models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm:AutoportGemma4ForCausalLM`. Existing demo architecture registrations are unchanged. |
| `tests/test_hybrid_cache_contracts.py` | Adds actual unifier/shape tests, legacy behavior tests, all three constructor API shapes, real host block-table append tests, deferred-token drain ordering/idempotence tests, and normal/lane/scheduler growth regressions. |

The page-growth repair deliberately drains at allocation changes. It does not advertise page-only refresh overlap or change model capability flags. It preserves steady decode for unchanged tables.

## Discriminating experiments

1. Before the local-head repair, the actual upstream unifier plus plugin `_kv_cache_shape` produced `(128,1,32,256)` for the requested local sliding geometry `(128,2,32,256)`. Both TP4/DP1 and TP4/DP2 cases failed; eight legacy shape cases passed.
2. Constructing a real `InputBatch` in the installed runtime failed before scheduler execution with `TypeError: MultiGroupBlockTable.__init__() got an unexpected keyword argument 'max_model_len'`. Inspection identified the vLLM0.26 `max_num_blocks` signature. The compatibility branch is separately checked with legacy, transitional, and current API doubles, and exercised against real installed cache tables in the scheduler tests.
3. A negative-control Python process restored only `_kv_cache_shape`, `_update_states`, lane `apply_step_plan`, and `steady_decode_scheduler_invariants_met` from the checkout's original `HEAD` methods using AST compilation. The constructor compatibility repair stayed active so the tests could reach the defects. The then-24-case suite produced **8 failed, 16 passed**: precisely two local-head cases plus growth in the first/non-first group for each of host drain, scheduler preflight, and lane reset. This process did not modify source files.
4. The repaired source then passed the new tests and existing state/lane regression tests: **66 passed, 16 warnings, 6.27 seconds**. The additional final test covers the transitional cache-table signature present in the checked-out vLLM source.

Exact positive command, from `/workspace/tt-metal`:

```bash
python -m pytest \
  vllm/plugins/vllm-tt-plugin/tests/test_hybrid_cache_contracts.py \
  vllm/plugins/vllm-tt-plugin/tests/test_state_slots.py \
  vllm/plugins/vllm-tt-plugin/tests/test_lane_input_batch.py \
  vllm/plugins/vllm-tt-plugin/tests/test_lane_model_runner.py \
  -q --disable-warnings --tb=short
```

Formatting/whitespace checks passed:

```bash
python -m black --check --target-version py310 \
  vllm/plugins/vllm-tt-plugin/tests/test_hybrid_cache_contracts.py
git -C vllm diff --check
```

No C++ or CMake changed, so no build was required. Black was initially run on all edited Python files; its unrelated changes to existing source formatting were removed to retain a focused patch.

## Remaining limits

- Tests use actual plugin state transitions, actual installed host block tables, and the actual async completion/apply machinery with a synthetic completed device result. They do not test real accelerator cache reads or automatic allocator behavior in a live server.
- Per-layer routing, per-layer cache views/storage sharing, current-position/token feedback, sampler logprobs, full supported context, and concurrent request boundary behavior remain the main stage's implementation/serving work. These host results are insufficient to set `supports_async_decode=True`.
- The matching vLLM checkout could not replace the installed wheel: source-first imports failed on missing `gguf`. The tested configuration is installed vLLM0.26 plus the patched TT plugin; later unrelated runtime API differences, if any, are not established by these checks.
