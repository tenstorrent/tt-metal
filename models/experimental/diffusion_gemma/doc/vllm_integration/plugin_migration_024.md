# DiffusionGemma serving: tenstorrent/vllm fork → vllm-tt-plugin (vLLM 0.24.0)

Migration record for moving the #47466 / #47488 block-granular serving work off the
`tenstorrent/vllm` fork (`dev` @ `6b4a3a7`, vLLM `0.1.dev1+g6b4a3a7b4`) and onto the
standalone [`tenstorrent/vllm-tt-plugin`](https://github.com/tenstorrent/vllm-tt-plugin)
against upstream vLLM `0.24.0` built with `VLLM_TARGET_DEVICE=empty`.

Measured on QB2 (`bh-qbge-06`, 4× Blackhole p300c) on 2026-07-30 against plugin `main`
@ `0c4e21c`.

## What actually had to move

Nothing DiffusionGemma-specific ever lived in vLLM core. All three fork commits
(`+294/-29`) touched only files under the fork's vendored `plugins/vllm-tt-plugin/`, so the
port is a path rewrite plus the API drift below:

```text
plugins/vllm-tt-plugin/src/vllm_tt_plugin/X  →  src/vllm_tt_plugin/X
plugins/vllm-tt-plugin/tests/X               →  tests/X
```

`git apply --3way` lands 6 of 8 file-patches cleanly, including the `platform.py`
registration and the `model_runner.py` block-granular changes. Two conflicts, both pure
adjacency, neither semantic:

| File | Conflict | Resolution |
| --- | --- | --- |
| `src/vllm_tt_plugin/scheduler.py` | upstream added `_has_pending_prefill()` and `schedule(throttle_prefills=…)` where `_update_request_with_output` is inserted | keep the new upstream signature, insert the override above it |
| `tests/test_lane_model_runner.py` | import block already had `SchedulerOutput` | keep only `Scheduler` and `RequestStatus` |

The tt-metal side (`models/experimental/diffusion_gemma/tt/generator_vllm.py`) needed
**no** change: it imports only `vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE` and
`vllm.v1.kv_cache_interface.FullAttentionSpec`, both present on 0.24, and it constructs
`FullAttentionSpec(block_size, num_kv_heads, head_size, dtype)` — exactly 0.24's four
required fields, with every field 0.24 added (`head_size_v`, `sliding_window`,
`attention_chunk_size`, `non_causal`, `kv_quant_mode`, `indexes_kv_by_block_stride`)
carrying a default.

## Break 1 — the async-discard API was renamed (ours to fix, fixed)

`AsyncScheduler`'s force-preempt flag changed shape between the fork base and 0.24:

```python
# fork base  vllm/v1/request.py:132     0.24.0  vllm/v1/request.py:142
self.discard_latest_async_tokens = False → self.async_tokens_to_discard = 0  # bool → counter
```

`TTScheduler._update_request_with_output` (the #47488 scheduler half) opens on that flag,
so unported it raises `AttributeError` on the first block commit. Ported to the counter
(decrement one frame per call). The fake request in `tests/test_lane_model_runner.py`
carried the old attribute too — that is what the three failing block-accounting tests were
actually reporting. The discard branch itself had no coverage; it does now.

`assert request.num_output_placeholders >= 0` still exists at
`vllm/v1/core/sched/async_scheduler.py:68` on 0.24, so **#47488 is not obsoleted** by
anything upstream did — the override is still required.

## Break 2 — vLLM 0.24 force-selects a Triton-only runner for *any* model with `canvas_length`

This is the one real DiffusionGemma-specific blocker, and no other TT model can hit it.

`ModelConfig.is_diffusion` on 0.24 is:

```python
# vllm/config/model.py:1529
def is_diffusion(self) -> bool:
    """Detect discrete diffusion (dLLM) models from HF config."""
    return getattr(self.hf_config, "canvas_length", None) is not None
```

The DiffusionGemma checkpoint's `config.json` has `canvas_length: 256`, so
`VllmConfig.use_v2_model_runner` returns `True` **before** the `HAS_TRITON` guard
(`vllm/config/vllm.py:526-534`), and `_validate_v2_model_runner()` then hard-raises:

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
  Value error, Model Runner V2 requires Triton.
```

On TT there is no GPU driver, so vLLM disables Triton at import
(`Triton is installed but 0 active driver(s) found`) and the server dies during
`create_engine_config` — before any device is touched. Non-diffusion TT models take the
`_is_default_v2_model_runner_model()` path instead, which *does* fall back to V1 politely.

Escape hatch: `VLLM_USE_V2_MODEL_RUNNER=0` short-circuits the property (an explicit
non-`None` value wins). The launch contract now sets it. The V2 model runner is a GPU
execution path the plugin does not implement at all, so the durable fix belongs in the
plugin: `TTPlatform.check_and_update_config` should force it off for every TT model rather
than leaving a diffusion checkpoint to trip a Triton requirement.

## Break 3 — 0.24 demands draft tokens from every dLLM, and TT has none

The same `is_diffusion` trigger, a second consequence, and the one that actually killed a
request on hardware. `EngineCore.__init__` on 0.24:

```python
# vllm/v1/engine/core.py:160
self.check_for_draft_tokens = self.use_spec_decode or vllm_config.model_config.is_diffusion
```

versus the fork, whose `post_step` guard was `not async_scheduling and self.use_spec_decode
and model_executed`. DiffusionGemma has no speculative config, so the fork never called it;
on 0.24 `is_diffusion` alone opens the gate and `post_step` issues a
`take_draft_token_ids` collective RPC every step. `TTWorker` had no such method, so
`vllm/v1/serial_utils.py` raised `NotImplementedError: Method 'take_draft_token_ids' is not
implemented` → `EngineCore encountered a fatal error` on the first request. The client saw
an opaque `500 InternalServerError` and the server then exited, which is exactly the
failure shape that produces a plausible-looking eval score from a dead engine.

Upstream's dLLM path reuses the speculative-decode draft plumbing to carry a model's
proposed tokens. TT block-diffusion has nothing to propose: the denoise loop lives inside
the model and commits a whole canvas per step, with nothing for the scheduler to verify.
`TTWorker.take_draft_token_ids()` now returns `None`, which `post_step` already treats as
"no drafts".

Note that this only fires because the plugin *disables* async scheduling for
DiffusionGemma (the model does not declare `supports_async_decode`) — the guard is
`check_for_draft_tokens and not self.async_scheduling`.

## Break 4 — plugin `main` imports a fork-only vLLM symbol (not ours; reported upstream)

`src/vllm_tt_plugin/launcher.py:19` does

```python
from vllm.v1.engine.utils import CoreEngine, CoreEngineLauncher, EngineLaunchPlan
```

`CoreEngineLauncher` and `EngineLaunchPlan` do not exist in upstream vLLM `v0.24.0`
(the version `docs/install-vllm-tt.sh` pins), nor in `v0.25.0`, nor in `main` — they exist
only in the **fork's** modified `vllm/v1/engine/utils.py` (lines 82 and 87 at `6b4a3a7`).
Upstream 0.24 exposes `launch_core_engines` as a function instead. Consequences:

- `tests/test_dp_modes.py` fails at collection on the pinned vLLM.
- `vllm_tt_plugin.launcher.TTCoreEngineLauncher` is unimportable, so the explicit
  `tt-run`/MPI launch path (`platform.py:1033`) cannot start.
- `platform.py:881` unconditionally sets
  `parallel_config.engine_core_launcher_cls = "vllm.v1.engine.utils.CoreEngineLauncher"`.
  Nothing in 0.24 reads that field, so for DiffusionGemma it is a dangling no-op rather
  than a failure.

DiffusionGemma serves single-process with `--max-num-seqs 1` and no MPI launch, so this
does not block us — but it does contradict the plugin's "nothing TT-specific touches vLLM
core" claim for the standard-DP launch path.

## Environment

```bash
# env: see plan.md
V=/home/zni/venvs/dg-vllm-plugin-024        # cloned from tt-diffusion-gemma, shebangs repointed
PLUGIN=/home/zni/tt-vllm-plugin             # tenstorrent/vllm-tt-plugin @ 0c4e21c + the port
VIRTUAL_ENV=$V $V/bin/uv pip uninstall vllm vllm_tt_plugin
cd $PLUGIN && VLLM_TARGET_DEVICE=empty $V/bin/uv pip install --no-binary vllm vllm==0.24.0
$V/bin/uv pip uninstall torchaudio && $V/bin/uv pip install -e .
```

`VLLM_TARGET_DEVICE=empty` takes `_no_device()` → `requirements/common.txt` only, which
pins no torch, so the env's `torch 2.11.0+cpu` / `transformers 5.12.1` / editable `ttnn`
survive. Result: `vllm 0.24.0+empty` + `vllm-tt-plugin 0.1.0`,
`Platform plugin tt is activated`.

Only three package versions moved, and one is worth watching: **`numpy 1.26.4 → 2.3.5`**
(also `xgrammar 0.1.29 → 0.2.3`, `+fastsafetensors 0.3.3`). `import ttnn` and all host
tests pass under numpy 2, but this is the one silent-ABI risk in the migration.

Launch differs from the fork recipe in exactly one way: the plugin no longer needs to be on
`PYTHONPATH` (it is an installed package, so `PYTHONPATH=/home/zni/tt-metal` alone is
enough — and pointing `TT_VLLM_ROOT` at the old fork checkout would *shadow* the installed
0.24 vLLM). No `VLLM_USE_V2_MODEL_RUNNER` is needed: the Break-2 fix lives in the plugin,
and the device run below confirms it in the live path.

## Plugin-side commits

Breaks 2 and 3 are plugin bugs that any dLLM would hit, not DiffusionGemma workarounds, so
they are fixed in `vllm-tt-plugin` rather than papered over in the launcher:

| Commit | Change |
| --- | --- |
| `platform: keep TT on the V1 model runner` | pin `VLLM_USE_V2_MODEL_RUNNER` off unless the operator set it; test guards the hook-before-read ordering |
| `worker: answer the draft-token RPC vLLM 0.24 makes for dLLM models` | `TTWorker.take_draft_token_ids() -> None` |
| `tests: track the vLLM 0.24 async-discard counter rename` | fixture rename + first coverage of the discard branch |

## Break 5 — the reasoning parser now actually fires, and it moves the answer out of `content`

This is the only behaviour difference visible in an eval score, and it is **not** a model
regression. On the paired 2-question smoke below the model output is bit-identical between
fork and plugin — same 9 blocks, same 2304 committed tokens, same `denoise_steps` sequence,
same `block_ids` token streams — yet the plugin scored 1/2 and the fork 2/2.

The chemistry question emits `<channel|>` (id 101) at output token 1434 of 1536, i.e. right
after `\boxed{B}`, then restarts its answer. Decoding those exact device tokens and running
`Gemma4ReasoningParser.extract_reasoning` on them, host-side:

| `skip_special_tokens` | `model_output` | `reasoning` | `content` | has `\boxed{B}` |
| --- | --- | --- | --- | --- |
| `True` (fork behaviour) | 3857 chars | 0 chars | **3857 chars** | in `content` |
| `False` (plugin behaviour) | 3867 chars | **3535 chars** | 322 chars | in `reasoning` |

`Gemma4ReasoningParser.adjust_request()` sets `skip_special_tokens = False` — the parser
explicitly asks to see its own delimiters. vLLM 0.24 honours that; the fork did not, so
`extract_reasoning` never found `<channel|>` and took its
`return None, model_output` early path, handing the whole chain of thought to `content`.
The plugin's behaviour is the parser working as designed.

Nothing is lost — the answer is in `reasoning`. But `lm_eval local-chat-completions` reads
`message.content`, so any response that closes its thinking channel now scores
`[invalid]` under flexible-extract. **Any full 198-question comparison against a fork
number is invalid until the harness reads `reasoning`** (the same trap already recorded for
the reference-server path: read `reasoning`, not `reasoning_content`). This is eval
plumbing, not serving quality.

## Host-side evidence

| Suite | Result |
| --- | --- |
| `tests/test_lane_model_runner.py` + `tests/test_reasoning_parser_registration.py` (plugin, ported) | 15 passed |
| DG `test_serving_block_contract` / `test_sampling_params` / `test_upfront_capture` / `test_served_gumbel_default` | 61 passed, 4 skipped (device-gated) |
| `tests/test_dp_modes.py` (plugin, untouched by us) | collection error — Break 3 |

The 12 `test_lane_input_batch` / `test_gemma4_tool_parser` failures are **pre-existing**:
a pristine clone of plugin `main` @ `0c4e21c` fails the same 12 against the same vLLM
(68 passed there vs 81 with the port — 13 added tests, zero new failures).

## Device evidence — paired 2-question GPQA smoke on QB2

`run_upfront_gpqa.sh smoke`, `MAX_MODEL_LEN=4096`, `max_gen_toks=1536`, thinking on,
`MESH_DEVICE=P150x4`, up-front capture, `gumbel_mode=device`. Plugin run
`smoke_024c`, fork run `smoke_fork_baseline`, same two questions.

| | plugin (0.24.0) | fork (`6b4a3a7`) |
| --- | --- | --- |
| `prefill_block0` events vs questions | 2 / 2 | 2 / 2 |
| blocks, committed tokens | 9, 2304 | 9, 2304 |
| `halted` | `True` ×9 | `True` ×9 |
| `denoise_steps` mean / min / max | 18.0 / 10 / 26 | 18.0 / 10 / 26 |
| block latency p50 / max (s) | 3.52 / 5.19 | 3.55 / 5.19 |
| tok/block/s min / p50 / max | 49.4 / 72.6 / 115.3 | 49.3 / 72.0 / 118.7 |
| DRAM free delta over the run (GiB) | −9.585 | −9.585 |
| GPQA flexible-extract | 0.5 | 1.0 |
| per-question responses | Q1 byte-identical; Q2 `content` truncated to the post-`<channel|>` 322 chars | Q1 identical; Q2 full 3857 chars |

Both runs are true greens by the false-green check (`prefill_block0` count equals the
question count, no fatal engine error, engine alive at the end). The score gap is entirely
Break 5. Serving throughput, memory behaviour, halting and the generated tokens themselves
are indistinguishable — which is the actual migration result.

Registration resolves end to end on 0.24: `TTDiffusionGemmaForBlockDiffusion` →
`models.experimental.diffusion_gemma.tt.generator_vllm.DiffusionGemmaForCausalLM`, MRO
`DiffusionGemmaForCausalLM → HybridAttentionForCausalLM → Generator`, with
`get_kv_cache_spec` / `initialize_vllm_model` / `prefill_forward` / `decode_forward` all
present. Import order matters when testing this by hand: importing
`vllm_tt_plugin.platform` before `vllm` trips a vLLM-internal circular import
(`cannot import name 'current_platform' from 'vllm.platforms'`).
