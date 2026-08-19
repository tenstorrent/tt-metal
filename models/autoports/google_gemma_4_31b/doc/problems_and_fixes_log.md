# Complete problem and fix log

Date range: 2026-08-14 to 2026-08-18. Host: `qb2-120-p02t03` (BH QuietBox 2,
2x p300c, 4 chips, 11x10 worker grid).

Everything that went wrong while moving this autoport onto new hardware, into
tt-shield CI, and through a local release run -- including the mistakes that were
mine. Model defects have their own document (`defects_found_by_release_flow.md`);
this is the operational and process record, plus an index.

## 1. Shield CI

Full dispatch history with root causes is in `shield_ci_onboarding.md`. Summary of
the distinct CI failure modes:

| Cause | Fix |
| --- | --- |
| `spec_tests` has no suites for any Gemma variant (`error=no_blocks`) | Do not use that lane; not applicable coverage |
| Base tokenizer has `chat_template=None`; the LLM benchmark driver hardcodes `--backend openai-chat --endpoint /v1/chat/completions` | Supply the autoport's raw-passthrough template as spec data |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0` aborts a cold first compile | Raise to 120 via spec env var, keeping detection and tt-triage wired |
| Runner host disk at 81%, runner shut itself down | Not ours |
| `bh-qb-ge` is a single p300x2 host; three release jobs contended | Not ours; expect queueing |
| `vllm-git-ref=dev` fails `resolve-shas` (HTTP 422) | tt-shield resolves from `vllm-tt-plugin`; use `main` |
| Image build `pathspec 'c127c17' did not match` | Branch's Dockerfile still cloned `tenstorrent/vllm`; merge `origin/main` |
| Benchmarks passed but every block `NA (ungraded)` | Added a `gemma-4-31b` perf reference entry; note it still cannot gate (see below) |
| `--workflow release` dies at setup without an `EVAL_CONFIGS` entry | Added a base-appropriate eval config |
| The LLM benchmark driver posts to `/v1/chat/completions` for **every** model; a base checkpoint defines no chat template, so all 17 sweep points die on `vllm bench serve`'s pre-flight probe with `Bad Request` | Endpoint selection from tokenizer capability: tt-inference-server branch `fix/benchmark-completions-endpoint-for-base-models`. Stopgap meanwhile: a passthrough `chat-template` in the spec |

Two structural facts worth repeating because they bound any green result:

- This model is `status: EXPERIMENTAL`, whose `required_target_tiers` is empty and
  whose `evals_enforced` derives from that list, so **neither perf nor eval
  accuracy gates the outcome**.
- All 381 entries in `model_performance_reference.json` use the `theoretical`
  tier, which appears in no status's required list, so perf gates nothing for any
  model on this platform.

## 2. Host and environment

| Problem | Fix |
| --- | --- |
| `/etc/profile.d/ttop.sh` sets `LD_LIBRARY_PATH` to a different checkout; segfault at precompile-firmware | `unset LD_LIBRARY_PATH TT_METAL_RUNTIME_ROOT` in every shell |
| `ImportError: cannot import name 'ModelRegistry' from 'vllm' (unknown location)` | A `tt-metal/vllm` symlink pointed at the vLLM *repo root* (no `__init__.py`), so `vllm` resolved as an empty namespace package while `TT_METAL_HOME` is first on `PYTHONPATH`. Remove it |
| Reverse trap: putting the vLLM checkout first on `PYTHONPATH` breaks tt-metal pytest, because vLLM ships its own `tests/` | Use `PYTHONPATH=$PWD` for pytest, the vLLM dir only for serving |
| `Timed out while waiting for active ethernet core 29-25` after any crashed run | Bounded `tt-smi -r`, then prove with a mesh open/close |
| Spec paths must reach into tt-metal from the server's cwd | Use the existing `../../tt-metal/...` convention; locally add a `tt-inference-server/tt-metal` symlink so the layout matches the container |
| Checkpoint not found | Symlinked the local weights into the HF cache and verified byte-identity via LFS oids |
| Cold JIT made TTFT look 30x worse (9796 ms vs 318 ms warm), accuracy identical | Always warm before quoting any latency |

## 3. Test-suite defects found in this repo

| Problem | Fix |
| --- | --- |
| `test_decode_head_grid.py` was **146 errors, 0 tests executed**. Its module-scoped fixture held a stub `ttnn` in `sys.modules` for the whole file; upstream `68aa24efb61` added an autouse fixture that does `import ttnn` at every test setup, hitting the stub's missing `ttnn.CONFIG` | Narrow the stub window to `exec_module` only. 146 errors -> 146 passed. This file is the portability coverage for the decode head core grid over both 11x10 and 14x10 grids, so losing it silently mattered |
| `test_multichip_sliding_nonaligned_window_wrap_matches_baseline` PCC-checks only the **decoded token** and asserts nothing about the prefill output activations beyond its shape | Structural blind spot: an error in `_chunked_attention_output_projection` (which produces hidden states after this layer's KV cache is written) cannot move the decode check. This is why a 20-point accuracy loss passed the suite. Added `test_zz_chunked_prefill_pcc.py` to compare prefill outputs across paths |
| Fabric-isolated probes (`fused_mmrs`, `fused_agmm*`, `fractured_*`) call `set_fabric_config(FABRIC_1D_RING)`, which requires no open devices, so they cannot share a pytest session with the rest of the file | Run each in its own pytest invocation. All 11 pass when isolated; they appear as 9 setup errors when batched |
| `tests/test_logging_utils.py` fails with `AttributeError: 'AsyncLogHandler' object has no attribute '_listener'` | **Pre-existing upstream tt-inference-server bug**, order-dependent. Reproduced in a clean worktree at pristine `origin/main`, in a file this work never touched |

## 4. My own mistakes

Recorded because they cost time and because anyone repeating this will hit the
same traps.

| Mistake | Consequence | Correction |
| --- | --- | --- |
| `pkill -f <pattern>` to clean up test processes | The pattern matched my own shell; killed it (exit 144). Happened twice | Never broad-kill. Collect PIDs with `ps -eo pid,cmd` and kill those |
| `pgrep -f` to check whether a job was running | Matched my own command line, producing three false "still running" readings | Same fix; exclude self or match on the script path |
| `sed -i 's/tt_metal_commit: "..."/.../'` to update one pin | Rewrote **8** pins, 7 belonging to other models, and used the wrong repo's HEAD | Caught on the diff, reverted the file, redid it with a script that locates the block declaring `impl: gemma4_31b_autoport` and asserts exactly one substitution |
| Set `GEMMA4_LONG_PREFILL=1` / `GEMMA4_LONG_DECODE=1` / `GEMMA4_LONG_GELU=1` as booleans | They are **values**: sequence lengths and a candidate name. Produced 5 false "failures", e.g. `assert 1 == 262144` | Use the documented values (262144) and candidate names (F4/F2/F1/C2048/C1024/C128). The tests' own sanity assertions caught it |
| Switched git branches while a release run was live | Changes files under a running job | It happened to be safe (server code already loaded, runtime spec snapshotted), but a separate worktree is the correct approach |
| Re-ran a failed readiness probe into the same output directory | Overwrote committed evidence: a passing 200 check became a 404 and `server.log` was truncated 13,981 -> 874 lines | Restored with `git checkout --`; never reuse an evidence directory |
| Lowered `GEMMA4_PREFILL_SDPA_MAX_SEQ` to dodge an L1 clash | Traded a visible crash for **silently wrong answers**: GPQA 37.5 -> 17.5, below the random floor, because the chunked path attends over `bfloat8_b` cache K/V | Reverted, with the measurement recorded in the spec so it is not repeated. Fixed the clash with SDPA L1 headroom instead |
| Hand-rolled sampling-parameter normalisation twice before looking for the platform helper | The first version rejected wide `top_k` instead of clamping, and passed a raw temperature where `ttnn.sampling` needs `1/T` | Search for an existing helper before writing a conversion, and check where comparable code calls it from |
| Amended with `git commit --amend -m` after `git checkout <base> -- <file>` had reset the index | The pushed commit was missing the spec entry; CI failed a second time on the same error | Verify the committed tree (`git diff --stat <base> HEAD`, `git show HEAD:<path>`) and the remote, not the working tree |
| Ran `git reset --hard` with uncommitted documentation in the tree | Reverted edits to two tracked docs; only the new untracked file survived | Commit or stash docs before resetting; `git status` before any reset |
| Scheduled the next run 45 s after the previous finished | The previous server had not released the 4 chips; `ttnn.open_mesh_device` failed | Wait for the process to exit, then reset, then prove the mesh opens |

### Wrong diagnoses I published and later corrected

| Claim | Reality |
| --- | --- |
| "TTI has zero LLM `EVAL_CONFIGS` entries using `apply_chat_template=False`" | Wrong. `meta_ifeval`, `meta_gpqa_cot`, `meta_gpqa`, `meta_math` all use it, and `EvalTask` defaults to `local-completions`. The blocker is the reference score, not the mechanics |
| "The chat-template requirement comes from trace capture (`call_chat_inference`)" | Wrong. That function has no callers; text trace capture uses `use_chat_api=False`. It is the **benchmark driver** that hardcodes the chat endpoint |
| "`CoreCoord(11,10)` in `fused_decoder.py` is on the decode hot path" | Wrong. `MultichipDecoder` replaces `shared_mlp` with `_TPOptimizedSharedMLP`; it is the single-device stage path |
| "The `1.00x` concurrency figure is unexplained, probably TTI deriving `max_num_batched_tokens` badly" | Wrong. It is the per-cache-group figure: `9740 blocks // 6 groups * 64 = 103,872`, and `103,872 / 113,280 = 0.917` |
| "`103,872 < 113,280` is a capacity shortfall" | Wrong. Comparing a per-group token count against the full context length is not apples-to-apples for a hybrid cache |
| "The isl=2048 crash is a direct-vs-chunked path problem" | Wrong. The chunked helper builds an **identical** SDPA config; the clash depends on allocator state, and the CB region ends at 1,374,976 in every case |
| "The 84.3 GPQA figure is transplanted from Qwen3.6-27B" | Half wrong. The **value** is Gemma's own published GPQA Diamond; only TTI's `published_score_ref` is misattributed |
| "The `test_logging_utils` failures come from the concurrently running server" | Wrong. They reproduce on pristine `origin/main` with nothing running |

## 5. Findings from dispatching unmodified

Running the model through CI with **no** spec workarounds (2026-08-19, run
32245795692) surfaced one harness defect, cleared one suspected model defect, and
led to fixing a third. Detail in `agentic_bringup_ci_dispatch.md`.

| Finding | Whose defect | Status |
| --- | --- | --- |
| Benchmark driver hardcodes the chat endpoint, so any base checkpoint fails its pre-flight probe | tt-inference-server (`llm_module/drivers/vllm.py:58-64`) | Fix pushed: `fix/benchmark-completions-endpoint-for-base-models` |
| 5 s `TT_METAL_OPERATION_TIMEOUT_SECONDS` aborts the cold first compile | previously believed ours | **Did not reproduce.** Zero `TT_THROW`/device-timeout in the run; the cold compile completed inside the default. The spec override is unnecessary and stays out |
| Non-greedy request kills EngineCore (`_require_semantic_greedy`) | ours | **Fixed**, tt-metal branch `mvasiljevic/gemma4-31b-nongreedy-sampling`. The generator already supported non-greedy; only the adapter refused. Never actually reached in CI — the chat-template failure came first |
| Raw temperature applied where `ttnn.sampling` needs 1/T | ours, latent | **Fixed by the same change.** The kernel multiplies top-k values by `temp` (`sampling.cpp:465`), so `temp` is the reciprocal; `_make_sampling_params` does not transform it. Greedy hid it because T=1 inverts to itself |

Two lessons worth keeping.

The spec entry carried six settings that no other model in the 58-template catalog
needed, each pre-empting a failure. Removing them turned one into a reproducible
upstream bug report and proved another was never a problem on this host.

And prefer the platform helper over a hand-rolled equivalent.
`models.common.sampling.format_sampling_params` clamps `top_k` into the sampler's
`[1, 32]` and inverts temperature. Two hand-rolled attempts rejected wide `top_k`
instead of clamping it (so a default vLLM request would still have failed) and
would have passed a raw temperature. Placement matters too: that helper is called
5 times in `tt_transformers/tt/generator.py` and 0 times in its
`generator_vllm.py`, so normalisation belongs on the generator — which also makes
it testable without vLLM installed (9 tests, 2.6 s, versus a contract file that
cannot even be collected without it).

## 6. Index of model defects

See `defects_found_by_release_flow.md` for the four defects the local release flow
exposed: the greedy-only adapter versus the upstream bench client's changed
default; prefill SDPA circular buffers clashing with decode-resident L1; a
`gcd`-sized block shard that overflowed L1 for 54% of prompt lengths; and the
accuracy loss caused by rerouting prompts onto the lower-precision chunked path.
