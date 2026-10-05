# Qwen3-VL on Quasar (craq-sim + emu2x3) — e2e design

Issue: [https://github.com/tenstorrent/tt-metal/issues/59033](https://github.com/tenstorrent/tt-metal/issues/59033)
Branch: `gchoudhary/59033/quasar/get-qwen3_vl-functional-end-to-end-on-emulator-and-craq-sim`

## Goal

Get Qwen3-VL-4B-Instruct running functionally correct on Quasar:

1. **craq-sim** (stepping stone): full pipeline (vision → text prefill → decode), on a 2x3 grid by default and 8x4 on demand.
2. **Quasar emulator** `emu-quasar-2x3` (ultimate goal): same pipeline with 2 vision blocks + 2 text layers, ideally under an hour per run.

Performance is not a goal. Correctness is: every stage must match a host torch reference by PCC.

Model: `Qwen/Qwen3-VL-4B-Instruct` (bf16 checkpoint) only; its config matches the graph captures. Other collection variants (2B/8B/32B dense, Thinking, MoE 30B/235B, FP8, GGUF) are out of scope.

## Constraints

- Model shapes (hidden dims, heads, head_dim, vocab, intermediate sizes) are fixed.
- Memory placement (DRAM/L1, interleaved/sharded) may change only where the captured config does not fit the grid or is not supported on Quasar; keep captured configs otherwise.
- bf16 everywhere — Quasar has no bfp8_b/bfp4_b.
- Repeatable: one command per target, flags for everything that varies, comparable output folders.



## Starting state (2026-10-05)

- Quasar model copy: `models/experimental/ops/quasar/qwen3_vl/` (copy of `models/demos/qwen3_vl`, imports `models/tt_transformers`). Essentially unadapted:
  - `ModelArgs` → `determine_device_name()` raises "Unsupported architecture" on Quasar.
  - Hardcoded 8x8 grids: `tt/vision_attention.py:337-351`, `tt/model_config.py:54-61`, many in `tt_transformers` `ModelArgs`.
  - Hardcoded bf8/bf4: `demo/demo.py:56,369`, `tt/model.py:198`, `tt/vision_attention.py:178,231,450,456,509`, `tt/vision_mlp.py:56,59`, `tt_transformers` `ccl_dtype`/`lm_head` dtype/default optimizations.
  - No layer-count override (text: `n_layers` before `load_state_dict` works; vision: `depth` hardcoded at `demo/demo.py:399`).
  - Vision patches always padded to the next multiple of 2048 (`tt/model.py:270`) although `vision_attention.py:364-372` only needs a multiple of 128 (≤2048) / 2048 (>2048).
- Op-level graph-capture tests: `models/experimental/ops/quasar/tests/qwen3_vl_ops/` (134 cases, 38 ops; 37 tagged `emulator`).
- Precedent: `models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py` (monkeypatch-heavy, does not gate on numerics). gpt_oss Quasar copy has only a layer-count knob.
- Checkpoint `Qwen/Qwen3-VL-4B-Instruct` is not cached locally; HF is reachable and a token is set.



## Approach (option C: hybrid)

Permanent Quasar adaptations go into the Quasar model copy; a thin harness provides presets, a host reference, per-stage PCC, and an op-override registry for bisecting and temporary workarounds.

### 1. Layout

All under `models/experimental/ops/quasar/qwen3_vl/`:

```
tt/quasar_config.py          # QuasarModelArgs / QuasarVisionModelArgs
tests/e2e/
  test_qwen3_vl_e2e.py       # single pytest node, all knobs via CLI options
  presets.py                 # tiny | demo
  host_reference.py          # truncated HF model + golden capture
  op_overrides.py            # host fallbacks + named workarounds
  pcc.py                     # per-stage compare, table, thresholds
  thresholds.json            # PCC floors per stage and preset
  test_fallbacks.py          # host fallback vs real op equivalence (run on WH/BH)
  run_craq.sh / run_emu.sh   # per-target env + flag forwarding
  run_wh_bh.sh               # same flags, no simulator (WH/BH hardware baseline)
  README.md                  # recipes, flags, cherry-pick list
```



### 2. Data flow (one run)

1. Resolve preset → image, prompt, sizes.
2. Host: load HF 4B truncated to V vision blocks / T text layers (deepstack remapped if requested); run once, record goldens per stage.
3. Build TT model via the Quasar config subclasses with the same truncated weights in bf16.
4. Run TT vision → text prefill → K teacher-forced decode steps; hooks record the same stage tensors (sliced to unpadded shape).
5. Compare every stage, print one table, assert at the end.



### 3. Run-length knobs


| Knob                                              | Flag               | Test default          | Script default |
| ------------------------------------------------- | ------------------ | --------------------- | -------------- |
| V — vision blocks (of 24)                         | `--vision-layers`  | 2                     | 2              |
| T — text decoder layers (of 36)                   | `--text-layers`    | 2                     | 2              |
| K — decode steps after prefill (0 = prefill only) | `--decode-steps`   | 1                     | 1              |
| Deepstack tap remap                               | `--deepstack-at I` | real taps (5, 11, 17) | 0              |
| Paged KV-cache blocks of 32 tokens                | `--kv-blocks N`    | preset                | preset         |


`--deepstack-at 0` moves the deepstack taps to vision block 0 (in both TT and HF) so a short vision tower still exercises the deepstack mergers and their add into the first text layers. It is a script default only; the test itself stays model-faithful.

### 4. Model-copy changes (`tt/quasar_config.py` + call sites)

Selected when `is_quasar()` or when `--quasar-config` is passed (forces the same config on WH/BH for baselines); otherwise non-Quasar behavior is unchanged.

- **Device name**: bypass `determine_device_name()`; derive from mesh shape.
- **Grid**: replace hardcoded `(8,8)`, `CoreGrid(8,8)`, `dram_shard_grid_width=8`, `find_prefill_grid` max with `device.compute_with_storage_grid_size()`. Sharded activation configs that do not fit become DRAM interleaved; captured configs are kept where they fit.
- **Dtype**: all weights/activations/KV cache/`ccl_dtype`/`lm_head` bf16, HiFi4, via optimization/`DecodersPrecision` overrides plus the hardcoded sites listed above.
- **Compute kernel config**: `WormholeComputeKernelConfig` → `ttnn.init_device_compute_kernel_config(arch, ...)`.
- **Layer counts**: `n_layers = T` before `load_state_dict`; vision `depth = V`; optional deepstack remap.
- **Vision padding**: multiple of 128 when ≤2048, multiple of 2048 above (demo unchanged at 12288).
- **Weight cache**: separate `TT_CACHE_PATH` subdir keyed by dtype policy, V, T.
- **Shared** `models/tt_transformers`: subclass/override first; guarded `is_quasar()` edits only where unavoidable, each called out in its commit.



### 5. Op-override registry (`tests/e2e/op_overrides.py`)

Installed via pytest `monkeypatch`.

- **Host fallbacks** (off by default, for bisecting): `--host-ops a,b` / `QWEN_QSR_HOST_OPS`. The wrapper copies inputs to host, runs the op's torch reference (see source order below), and returns a device tensor with the op's original dtype/layout/memory config. `all` routes every registered op; `--list-ops` prints names.
- **Workarounds** (on by default, start empty): added only for real failures. Each entry records reason and removal condition (e.g. "drop when #58912 lands"), is narrowly predicated (op + shape/memcfg), never changes model shapes. `--disable-wa a,b` / `QWEN_QSR_DISABLE_WA` turns them off.
- **Visibility**: active overrides logged at start; per-override hit counts and everything that ran off-device appear in the final table.
- Long-lived workarounds graduate into `quasar_config`.

**Trusting fallbacks.** A wrong torch reference must never produce a false fix.

1. **Host fallbacks never pass a run.** Any run with an active host fallback ends `DIAGNOSTIC` (test fails with "ops on host: ..."; scripts exit with a distinct code). A fix counts only with zero host ops and all stages passing PCC against HF, which is independent of anything we write.
2. **Minimize new torch code.** Source order per op: (a) `ttnn.get_golden_function(op)` (exists for SDPA, paged SDPA decode); (b) existing `_ref_*` in `models/experimental/ops/quasar/tests/qwen3_vl_ops/graph_case.py` (already exercised against real ops by that suite); (c) hand-written only where neither exists — expected for `rotary_embedding_llama`, `paged_update_cache`, `paged_fill_cache`.
3. **Every fallback is certified against the real op**, including goldens and `graph_case` refs. `test_fallbacks.py` runs fallback vs real ttnn op on the captured `qwen3_vl_ops` cases (same shapes/dtypes/memcfgs, bf16) on WH/BH (ttsim or hardware), never Quasar: PCC ≥ 0.999 and identical shape/layout. Each registry entry records the certifying test id and the arch/commit it passed on; uncertified fallbacks are refused.
4. **Collective check against HF**: `--host-ops all` must match HF at ≥ 0.999 on every stage (catches fallbacks that pass in isolation but use the wrong layout for real model tensors, e.g. rope).

The same rule governs device-side workarounds: they count only once the zero-host-op run passes against HF.



### 6. Presets (`tests/e2e/presets.py`)


|                           | tiny                                              | demo (captured)                                                      |
| ------------------------- | ------------------------------------------------- | -------------------------------------------------------------------- |
| Image                     | 256×256 → 256 patches → 64 image tokens           | `demo.jpeg` 2048×1365 (cached locally) → 11008 patches → 2752 tokens |
| Vision pad                | 256                                               | 12288                                                                |
| Prompt                    | "Describe this image." (~80 tokens) → prefill 128 | same → 2766 → 4096                                                   |
| `max_seq_len` / KV blocks | 256 / 8                                           | 4096 / 1024                                                          |


Prefill padding uses `tt_transformers` `get_padded_prefill_len` unchanged. Presets are data; an intermediate size gets added once tiny runtimes are measured, targeting <1 h per iteration.

### 7. Run scripts

Thin bash wrappers (bash-safety rules, `shellcheck -o all` clean) around the one pytest node.

Common flags: `--size`, `--vision-layers`, `--text-layers`, `--decode-steps`, `--deepstack-at`, `--kv-blocks`, `--host-ops`, `--disable-wa`, `--debug fast|default|deep`, `--noc-sanitize`, `--timeout S`, `-- <pytest args>`. Every default carries a ≤2-sentence comment explaining it.


|                      | `run_craq.sh`                                                                                                          | `run_emu.sh`                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `TT_METAL_SIMULATOR` | `/localdev/$USER/sim/libttsim.so` (file; env-overridable)                                                              | `/proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-2x3/` (dir; env-overridable) |
| Grid                 | `--grid 2x3` (default, via `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`) or `8x4` (native)                                | native 2x3                                                                               |
| Always               | `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_FORCE_JIT_COMPILE=1`, `TT_METAL_DISABLE_SFPLOADMACRO=1`, `MESH_DEVICE=N150` | same                                                                                     |
| Pre-flight           | simulator file exists                                                                                                  | `NNG_SOCKET_ADDR` set; no other own pytest/zrun on the port                              |
| `--timeout` default  | 3600                                                                                                                   | 14400                                                                                    |


`TT_METAL_QUASAR_NOC_API_VERSION` is left unset (v2).

**Debug profiles** (`--debug`):


| Profile   | Env                                                                                                                                                                                                                         |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fast`    | no watcher / asserts                                                                                                                                                                                                        |
| `default` | `TT_METAL_WATCHER=1`, `TT_METAL_WATCHER_TEST_MODE=1`, `TT_METAL_LLK_ASSERTS=1`, `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1`                                                                                                   |
| `deep`    | `default` + `TT_METAL_WATCHER_DUMP_ALL=1`, `TT_METAL_WATCHER_NOINLINE=1`, `TT_METAL_DPRINT_ONE_FILE_PER_RISC=1`, `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`, `TT_METAL_WATCHER_DISABLE_PAUSE=1`, `TT_METAL_LOGGER_LEVEL=DEBUG` |


NoC sanitize is off in all profiles (20–30× slowdown); `--noc-sanitize` re-enables it.

**Output**: `generated/qwen3_vl_quasar/<target>/<timestamp>/` with full log, resolved command + env, `git rev-parse HEAD` + cherry-pick list, `progress.log`, and `pcc.md` (stage table, per-stage wall time, host ops, workarounds). The script prints the folder and table at the end.

**`run_wh_bh.sh`** (WH/BH baseline, run on a separate machine with hardware): same flags, no simulator, always passes `--quasar-config`, and applies the same grid override (`--grid 2x3` default, `native` for the full grid). The override is generic core-descriptor logic (`tt_metal/llrt/core_descriptor.cpp:199`), so it works on silicon. `--ttsim wh|bh` runs it locally on ttsim (`/localdev/$USER/ttsim/src/_out/release_{wh,bh}/`, staged with the matching soc descriptor by the script) for fast iteration; without it, it targets hardware on a separate machine. Its output folder is self-contained so it can be copied back and diffed against Quasar runs.

### 8. Pass gate

Compared after slicing TT outputs to unpadded shape:


| Stage                                         | Golden                            | Initial PCC floor |
| --------------------------------------------- | --------------------------------- | ----------------- |
| each vision block output                      | `visual.blocks[i]`                | 0.99              |
| deepstack merger outputs                      | `visual.deepstack_merger_list[j]` | 0.99              |
| patch merger (image embeds)                   | `visual.merger`                   | 0.99              |
| each text layer output (prefill)              | `language_model.layers[i]`        | 0.99              |
| final norm / prefill logits (last real token) | `norm`, `lm_head`                 | 0.99 / 0.98       |
| each decode step's logits (teacher-forced)    | HF argmax token as input          | 0.98              |


- Floors live in `thresholds.json` (stage × preset).
- Any NaN/Inf fails regardless of PCC.
- Top-1 token agreement is reported, not gated.
- Any active host fallback turns the result into `DIAGNOSTIC`, never `PASS` (see "Trusting fallbacks").
- All stages are compared and printed; the first failing stage is highlighted. Op exceptions propagate normally.



### 9. Hang handling

1. **Watcher + LLK asserts** (`default`/`deep` profiles): assert trips fail fast via watcher test mode; waypoints in the watcher log.
2. `progress.log` from `ttnn.register_pre_operation_hook` / `register_post_operation_hook`: op name, input shapes/memcfgs, model stage, timestamp. A pre without a post identifies the hung op.
3. **pytest** `--timeout`: the hang detector of record. On simulators `wait_until_cores_done` (`tt_metal/llrt/llrt.cpp` ~L407-412) ignores `TT_METAL_OPERATION_TIMEOUT_SECONDS`, so a pure deadlock is only caught here; the timeout failure points at `progress.log` and the watcher log for diagnosis.

Not committed, local debugging aid only: if hangs get hard to pin down, temporarily patch `wait_until_cores_done` to honor `TT_METAL_OPERATION_TIMEOUT_SECONDS` on a simulator when explicitly set, so a deadlocked op throws listing not-done cores. No commit or PR for this change.

### 10. Harness validation and rollout

1. CPU-only unit tests: preset token counts with the real processor; deepstack remap equivalence.
2. **WH/BH baseline first** (`run_wh_bh.sh`; locally on ttsim WH/BH, then confirmed on hardware by the user on a separate machine): every config we plan to run on Quasar — same preset, V/T/K, deepstack remap, `--quasar-config`, 2x3 grid override (and native grid) — must pass PCC against HF with zero host ops. `test_fallbacks.py` certifies the host fallbacks in the same session (ttsim acceptable for certification; hardware confirmation recorded when available). This proves config + harness are sound, so later Quasar failures are Quasar issues. WH/BH run mainline kernels where Quasar uses `ttnn.experimental.quasar.*` ones, so it does not validate Quasar kernels.
3. `--host-ops all` on craq-sim: PCC ≥ 0.999 everywhere, or the harness is wrong.
4. Rollout, each gated on the previous passing: craq-sim 2x3 tiny → craq-sim 8x4 tiny → emulator 2x3 tiny → demo size on craq-sim → demo size on emulator if within time budget. A new config (preset, V/T/K, grid) gets a WH/BH baseline before its first Quasar run.

Verify during implementation: the exact `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` value (it sets the grid's end coordinate) that reproduces the emulator's `compute_with_storage_grid_size()`.



## Dependencies and cherry-picks

Candidate PRs (vsureshTT) that may be cherry-picked while iterating and dropped on rebase once merged; the README tracks which are applied:

- #58909 `minimal_matmul` Quasar uplift (text prefill matmuls)
- #58912 DRAM-sharded matmul Quasar uplift (10 DRAM width-sharded linears in capture)
- #58913 paged fused `update_cache` packer fix (decode KV update)
- #58914 `reshape_view` tiled on Quasar

Prefer the real op with these applied; add a workaround only if it still fails, naming the PR that obsoletes it.

## Out of scope

- Performance, tracing, multi-device meshes, vLLM path, video inputs.
- Full 24/36-layer runs on the emulator (craq-sim only, if at all).
- Fixing the op-level `qwen3_vl_ops` suite beyond what the e2e run needs.

