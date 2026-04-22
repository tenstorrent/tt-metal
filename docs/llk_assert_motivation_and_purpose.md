# LLK asserts: motivation and purpose

## What `LLK_ASSERT` is

`LLK_ASSERT` is defined and used in the **tt-llk** library: the device-side code that drives unpack, math, and pack around L1 and tile registers. Those paths assume **consistent hardware configuration** (for example `face_r_dim`, `num_faces`, and data formats) and **legal tensor geometry** for each API call.

## Motivation

It is easy for a compute kernel to call the LLK in an order or with parameters that are reachable in C++ but **illegal** for the hardware or for the current unpack/pack state. That can produce **wrong numerics with no Python-level error**, or failures that are hard to attribute without device-side checks.

## Purpose

With `TT_METAL_LLK_ASSERTS=1`, JIT-built kernels include runtime checks that **fail at the bad call** (for example via `ebreak` / the assert path, often together with triage or lightweight kernel asserts). The failure is then tied to **kernel code** (reconfig ordering, wrong `num_faces`, format mismatch), not to a vague “test is flaky” story.

In short, `LLK_ASSERT` is **runtime validation of LLK usage**. It complements Lightweight Kernel Asserts and Watcher, and matches the “HW configure vs. init / execute mismatch” pattern discussed in project issues (for example unpack configuration verification in `#39184`) and in the Sphinx doc `docs/source/tt-metalium/tools/llk_asserts.rst`.

## Related material

- **Hands-on demos** (commits `25c5623c26` and `60fc5bac57` on branch `ndivnic/demo_prep`): see [llk_assert_demos.md](./llk_assert_demos.md).
- **User-facing reference:** `docs/source/tt-metalium/tools/llk_asserts.rst`.
