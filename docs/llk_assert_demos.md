# LLK assert demos (commits `25c5623c26` and `60fc5bac57`)

This document describes two teaching commits on branch `ndivnic/demo_prep` (“Demo 1.” and “Demo 2.”). For why `LLK_ASSERT` exists at all, see [llk_assert_motivation_and_purpose.md](./llk_assert_motivation_and_purpose.md).

---

## Demo 1 — commit `25c5623c26` (“Demo 1.”)

### Files

- `models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py`
- `models/demos/deepseek_v3_b1/unified_kernels/deepseek_moe_gate.hpp`

### What changed

1. **Test** — Removes `@skip_with_llk_assert(...)` (and its import) so **`test_deepseek_moe_gate` runs when LLK asserts are enabled**. Previously the test was skipped under `TT_METAL_LLK_ASSERTS=1` because of a known unpack-configuration assert (issue `#39472`). The demo branch exposes the real path again for teaching.

2. **Kernel** — Adds **comments** above the MoE gate TRISC path that describe how to use the demo:
   - **Break the kernel on purpose:** comment out the `reconfig_data_format` / `pack_reconfig_data_format` calls that keep **input indices, bias, and output** tile shapes and formats aligned before `copy_tile_to_dst_init_short` and `deepseek_moe_gate_init`.
   - **Observe the failure:** run with `TT_METAL_LLK_ASSERTS=1` and the suggested `pytest` node (see in-file comment), so an **LLK assert** fires (or wrong results appear, depending on how the kernel is broken).
   - **Fix:** restore the reconfig calls so hardware state matches what the unpack/init path expects.

### What this demo shows

- A **real model demo** (DeepSeek MoE gate), not a minimal op.
- The lesson that **missing or wrong `reconfig_data_format` / `pack_reconfig_data_format`** relative to later `init` / compute can trigger **unpack (or pack) configuration verification** asserts under LLK asserts.
- A workflow: **do not hide the failure** (`skip_with_llk_assert` removed for the demo), **intentionally regress the kernel**, then **use LLK asserts and pytest** to localize and fix **kernel** code.

---

## Demo 2 — commit `60fc5bac57` (“Demo 2.”)

### File

- `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp`

### What changed

- Replaces the **correct** compile-time logic for `num_faces_in_input_tile` (values `2` or `4` derived from `max_sticks_for_reduction`, `window_size_hw`, and `FACE_HEIGHT`) with a **deliberately invalid** constant:
  - `constexpr uint32_t num_faces_in_input_tile = 3;`
- The real `constexpr` is **commented out** beside a short explanation.
- Comments include a **multi-line** `TT_METAL_LLK_ASSERTS=1 pytest ...` example targeting a specific `test_max_pool2d_output_formats_and_layouts[...]` parametrization (join comment lines into one shell string when running).

That value is passed into `tilizeA_B_reduce_init<...>(..., num_faces_in_input_tile, face_r_dim)`, where the LLK stack enforces **`num_faces` ∈ {1, 2, 4}**.

### What this demo shows

- A **generic TTNN pool** compute kernel.
- **Invalid geometry:** `num_faces == 3` is **never** legal for the tilize/unpack rules, yielding a **clear, reproducible LLK assert** without a subtle ordering bug.
- An **end-to-end** path: pool test → compute kernel → tilize init → assert, with environment and test name documented in the source.

---

## Side-by-side

| | **Demo 1** (`25c5623c26`) | **Demo 2** (`60fc5bac57`) |
|--|---------------------------|---------------------------|
| **Area** | Model demo: DeepSeek MoE gate | Core op: max pool 2D compute |
| **Bug style** | Omit or break **reconfig** → format or face layout **mismatch** across CBs or phases | **Illegal `num_faces`** (`3`) passed into tilize init |
| **Pedagogy** | Real kernel; **assert tied to pack/unpack sequencing** | **Minimal** illegal parameter; **face / tile invariant** assert |
| **Test hint** | `test_deepseek_moe_gate[...]` (exact node id in kernel comment) | Specific `test_maxpool2d` parametrization (in kernel comment) |

---

## How to run either demo

1. Export **`TT_METAL_LLK_ASSERTS=1`**. Optionally add **`TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`** if you want lightweight-assert call stacks (see `llk_asserts.rst`).
2. Use the **pytest invocation copied from the comments** in the touched sources (Demo 2’s string may be split across comment lines; rejoin for the shell).
3. **Demo 1:** follow the kernel comment and **comment out** the indicated `reconfig_*` lines to surface wrong behavior or an assert, then **restore** them to fix the kernel.

---

## End-to-end demo run (Demo 1, with `DEVICE_PRINT` + auto-triage)

This is the recommended “one-shot” way to reproduce Demo 1. It enables the new
`DEVICE_PRINT` system, sends device prints to a file, and wires up
[`tools/tt-triage.py`](../tools/tt-triage.py) so that on a dispatch timeout the
lightweight asserts on every RISC are dumped automatically and the card is
reset.

> Prerequisites
>
> - You are at the repo root and your Python env is active
>   (`source python_env/bin/activate`).
> - The branch contains the Demo 1 kernel break (or you have manually commented
>   out the `reconfig_*` calls per the kernel comment).
> - `tt-smi` is on your `PATH` (used to reset the card after a hang).

### 1. Set up the environment

```bash
# Where DEVICE_PRINT output goes (relative to $PWD).
export TT_METAL_DPRINT_FILE=tt_dprint.log

# Auto-triage on dispatch timeout: dump the lightweight asserts then reset the card.
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="$(pwd)/tools/tt-triage.py --run=dump_lightweight_asserts > assert.txt && tt-smi -r"

# Treat the op as hung quickly so the timeout command actually fires.
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0

# Allow tt-triage to run scripts that are normally CI-disabled.
export TT_RUN_DISABLED_TRIAGE_SCRIPTS_IN_CI=1

# Print from every worker core; use the new DEVICE_PRINT system.
export TT_METAL_DPRINT_CORES=all
export TT_METAL_DEVICE_PRINT=1
```

### 2. Run the test

```bash
TT_METAL_LLK_ASSERTS=1 \
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py::test_deepseek_moe_gate[42-True-1] \
  > test_output.txt
```

### 3. Inspect the artifacts

After the run you should have three files in `$PWD`:

- `test_output.txt` — full `pytest` stdout/stderr, including loguru lines from
  Metal and the test itself.
- `tt_dprint.log` — every `DEVICE_PRINT` line emitted by enabled RISCs, prefixed
  with `device:(x,y):RISC:`. Each `DEVICE_PRINT` **must end with `\n`** to be
  flushed by the host server (see
  [`docs/source/tt-metalium/tools/device_print.rst`](source/tt-metalium/tools/device_print.rst)).
- `assert.txt` — only present if the op timed out and the dispatch-timeout
  command ran. Contains the lightweight-assert dump from
  `tools/tt-triage.py --run=dump_lightweight_asserts`, including the per-RISC
  failed line and template/runtime/local variable values when DWARF supplies
  them (see
  [`docs/source/tt-metalium/tools/llk_asserts.rst`](source/tt-metalium/tools/llk_asserts.rst)).

### 4. What you should see

- **Kernel intact** → the test passes; `tt_dprint.log` shows the smoke-test
  lines added in `deepseek_moe_gate_kernel.cpp` (one per RISC), plus any
  unconditional `DEVICE_PRINT` lines from the LLK config checks (e.g.
  `is_unpacker_A_configured_correctly`).
- **Demo 1 kernel break applied** (the `reconfig_*` calls are commented out) →
  expect either an LLK assert (`ebreak`) inside an unpack-/pack- configuration
  check, or a `tt_dprint.log` line of the form
  `unp_A_src_format mismatch. expected: <X>, actual: <Y>` followed by the
  `LLK_ASSERT` firing. If the op hangs, `assert.txt` will be populated by the
  dispatch-timeout command and the card will be reset.

### Troubleshooting

- **`assert.txt` is empty** → either the op never timed out (no hang) or
  `tools/tt-triage.py` failed; check that `TT_RUN_DISABLED_TRIAGE_SCRIPTS_IN_CI=1`
  is exported.
- **`tt_dprint.log` is empty** → confirm `TT_METAL_DPRINT_CORES`,
  `TT_METAL_DEVICE_PRINT=1`, and that every `DEVICE_PRINT` you expect ends with
  `\n` (partial lines are buffered and dropped at device close).
- **`tt-smi -r` fails / is missing** → drop the `&& tt-smi -r` portion from
  `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE`; you will then need to reset
  the card manually before re-running.

---

## Takeaway

Both commits are **teaching aids** on one theme: **`LLK_ASSERT` turns silent or confusing LLK misuse into a deterministic, kernel-local failure.** Demo 1 stresses **configuration and reconfig discipline** in a model kernel; Demo 2 stresses **valid face and tile parameters** in a shared pool kernel. The commit subjects are only “Demo 1.” / “Demo 2.”; the **intent is in the diffs and in-file comments**.
