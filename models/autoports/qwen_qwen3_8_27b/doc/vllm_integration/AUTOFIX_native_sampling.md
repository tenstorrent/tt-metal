# AutoFix: native sampling seed continuity

## Starting evidence and hypothesis

`AUTODEBUG_mixed_sampling.md` finding 2 identifies a separate native RNG
defect: adapter batch/layout/parameter updates write the original request seed
over the advancing TT seed tensor. This can rewind a continuing request when
a companion leaves, arrives, moves rows, or changes parameters. This repair
does not establish the cause of the original mixed-backend `Letter: ` failure.

The parent explicitly authorized CPU-only proof, the adapter repair, and an
optional-seed argument in `QwenGenerator.set_batch_sampling_params`. The live
server retains its already-loaded original code until the parent restarts it.

## Experiment and result

`tests/test_vllm_seed_continuity_host.py` extracts the actual adapter class and
the actual generator sampling-parameter method via AST, excluding module
imports. A fake generator replaces only device effects: seed writes copy to a
torch tensor, and a sample records then increments that tensor. No TTNN module
is imported and no model/device code executes.

```bash
python_env/bin/python -m unittest \
  models.autoports.qwen_qwen3_8_27b.tests.test_vllm_seed_continuity_host -v
```

- Before: 7 tests ran, 6 failed (10 failure records including seed-value
  subtests), 1 passed. `seed_continuity_before.log` preserves the result.
  Survivor seed after a layout reset was **42 instead of 45**; after moving
  row 1 to row 0 it was **99 instead of 102**. Companion-only top-k changes
  and host-to-device transition also failed. Unchanged steady decode passed.
- After: **9 tests passed** in `seed_continuity_after.log`, including the
  original cases, an intervening prefill, rejecting seed changes without
  authoritative positions, and near-limit positive/negative user seeds.
- A separate fresh-process import check confirmed `ttnn` is absent from
  `sys.modules` after loading the proof module.

Verdict: **native RNG rewind verified; CPU repair verified**.

## Retained changes

`tt/generator_vllm.py` gives each explicit seed this deterministic mapping:

```text
M = 2**31 - 262144 - 1
device_seed = request_seed % M + absolute_output_position
prefill output position = prompt_end
decode output position = start_pos + 1
```

The fixed modulus is independent of the configured serving context. For every
supported output position 0..262144, the largest pre-draw seed is
`INT32_MAX - 1`, and its following increment is at most `INT32_MAX`. There is
no signed overflow or `manual_seed`'s `UINT32_MAX` skip-reseed sentinel within
the supported context. Tests cover the maximum normalized base, signed-int32
boundary, unsigned-int32 maximum, and signed-int64 extremes.

Prefill and authoritative decode refreshes apply that mapping; the transition
from host to device is recognized before seeds are configured. Steady decode
continues advancing the device tensor without seed copies. A change only to
top-k/top-p/temperature updates canonical sampling parameter tensors while
preserving every device seed, even if supplied host positions lag. New seed
values without an authoritative reset are rejected before sampling mutation.
The plugin marks additions, removals and compaction as layout changes
(`model_runner.py:698-710`) and makes that flag `reset_batch` on the next decode
(`model_runner.py:1141-1142`); a preceding adapter prefill also forces refresh.

`tt/generator.py` changes only `set_batch_sampling_params`: `seed=None` means
update validated sampling parameters while retaining the seed tensor. The
adapter delegates this update to that existing canonical generator method.
There is no token feedback, logits readback, sampler replacement, or new
device allocation in the steady path.

```bash
python_env/bin/python -m black --target-version py312 --check \
  models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py \
  models/autoports/qwen_qwen3_8_27b/tt/generator.py \
  models/autoports/qwen_qwen3_8_27b/tests/test_vllm_seed_continuity_host.py
git diff --check
```

Both passed. Python-only change; no build is required. The first formatting
invocation omitted `--target-version py312` and warned about the repository's
newer target; the explicit-target check above completed cleanly.

## Required runtime validation and limits

The parent must restart to load the change, run a native stochastic survivor
alone versus companion admission/departure and row changes, and compare raw
logits plus per-request output streams with identical prefixes. Repeat the
steady adapter counter check: no token, position, or seed writes during
unchanged native replay. Existing prefill-host tests that asserted raw seed
values need their expected values updated to the new documented output-position
mapping (`[7,11]` at ends `[3,35]` becomes `[10,46]`).

This changes the native explicit-seed mapping and fixes its continuity; it
does not promise equivalence with PyTorch host RNG, establish model quality,
or resolve arbitrary mixed-backend reproducibility. The unchanged full shared
suite still needs to succeed under the explicitly selected compatibility
configuration. Native correctness and performance remain separate gates.

## Runtime follow-through, 2026-09-14

Final code passed `readiness_vllm/native_seed_continuity.json`: exact native token
IDs match for A(S67,G100) and B(S69,G47) alone, B(G33)+A(G100), and A(G100)+B(G47),
using k5/T0.7/p0.9/seeds42/43. All6 comparisons pass. HTTP scheduling does not
force slot placement; the CPU tests separately force remaps and admission state.
`native_seed_server.log` contains no explicit-host fallback marker in this run.

The final allocation-tracked reduced adapter probe
`adapter_device_after_seed_fix.json` passes68 replay steps. Only initial binding
and the drained remap refresh token/position/RoPE/seed state; unchanged steps do
not. Full-model `full_concurrent_control.json` independently compares12 diverse
native/synchronous/reordered100-token streams, all exact in text and token IDs.
`full_logit_determinism.json` and `standalone_logit_control.json` show exact full
prefill distributions in repeated serving requests and explicitly swapped
standalone slots. These are concrete limits of the runtime proof, not universal
TT/PyTorch RNG equivalence. The canonical full compatibility suite is separately
run with explicit host-only mode; its final result belongs in README/work_log.
