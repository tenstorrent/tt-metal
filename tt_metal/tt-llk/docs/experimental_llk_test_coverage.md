<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental LLK — Test Coverage Plan (by issue)

Tracking coverage of the **canonical** experimental LLKs under
`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/experimental/`, and the GitHub
issues that close the gaps. Part of the experimental-LLK contract/coverage effort
(tt-metal #47554).

_Last updated: 2026-07-22._

## Scope

- **In scope:** the canonical `experimental/` LLKs that tt-metal owns today.
- **Out of scope (deferred):** LLKs that currently live under `ttnn/.../deepseek/` and
  `models/demos/deepseek_v3_b1/kernel_includes/`. These will be relocated into
  `experimental/` by a separate **upstream kernel-migration effort**; we do not
  transfer or test them here to avoid duplicate work. Their tests will be filed once
  the kernels land in `experimental/` (subset only — not all will be pulled).
- **Definition of an LLK:** a function that wraps Tensix ISA into a reusable ML building
  block (unpack / math / pack / reduce / transpose). Helper/scaffolding functions
  (`*_init_`, `*_uninit_`, `*_mop_config_`, `*_configure_*`, `*_reinit_*`, …) are not
  counted as LLKs.

## Coverage matrix

Legend: ✅ covered · 🟡 partial / being deepened · ❌ no coverage → new test.

| Experimental LLK | Arch | Current coverage | Tracked by |
|---|---|---|---|
| `eltwise_binary_custom` (sub + mul bcast-col reuse) | BH+WH | ✅ `test_eltwise_bcast_col_custom` (both ops) | + uninit sweep #50203 |
| `unpack_AB_sub_bcast_col_custom` | BH+WH | ✅ paired in `test_eltwise_bcast_col_custom` | + uninit sweep #50203 |
| `eltwise_unary_datacopy_custom` | BH | 🟡 `test_eltwise_unary_datacopy_custom` (single format/config) | deepen (stretch) #50562 |
| `unpack_A_custom` | BH | 🟡 only as a helper inside the datacopy_custom kernel | #50562 (standalone) |
| `matmul_custom_no_mop` | BH+WH | 🟡 `test_matmul_custom` = LoFi + HiFi2/3/4 (no throttle) | throttle #50562 · uninit #50203 |
| `mul_reduce_scalar` (math + unpack) | BH | ✅ (PR #50547) | #50200 |
| `reduce_block_max_row` (custom + runtime, math + unpack) | BH+WH | 🟡 fuser/chain only; runtime/reinit/trigger thin | #50202 |
| `generalized_moe_gate` (eltwise + transpose + top-k) | BH+WH | ❌ zero references | #50201 |
| `fast_tilize` (math + pack + unpack) | BH | ✅ `test_fast_tilize*` | — |
| `fast_untilize` (math + pack + unpack) | BH | ✅ `test_fast_untilize` | #50200 (verify) |
| `pack_block_contiguous` | BH | ✅ `test_pack_tiny_tile_block` (+ reconfig) | deepen (stretch) #50562 |
| `topk_xl_copy` (math + unpack) | BH | — | **deferred** (upstream pull) |

## Issue slate

| Issue | Scope | Arch |
|---|---|---|
| [#50200](https://github.com/tenstorrent/tt-metal/issues/50200) | `mul_reduce_scalar` (+ confirm `fast_untilize` already on main) | BH |
| [#50201](https://github.com/tenstorrent/tt-metal/issues/50201) | `generalized_moe_gate` — gate eltwise + single-face transpose + top-k | BH+WH |
| [#50202](https://github.com/tenstorrent/tt-metal/issues/50202) | Deepen `reduce_block_max_row` — standalone sweep + runtime/reinit/reprogram + `respect_trigger`/`overlap_first_half` | BH+WH |
| [#50203](https://github.com/tenstorrent/tt-metal/issues/50203) | Reconfig-escape / empty-`uninit` sweep — `matmul_custom_no_mop`, `eltwise_binary_custom`, `reduce_block_max_row`, `unpack_AB_sub_bcast_col_custom` | BH+WH |
| [#50562](https://github.com/tenstorrent/tt-metal/issues/50562) | `matmul_custom_no_mop` THROTTLE_LEVEL 1–5 + `unpack_A_custom` standalone (+ deepen datacopy_custom / pack_block) | BH+WH / BH |

With these five, every canonical `experimental/` LLK is either already covered or has a
tracking issue. The only deliberate exclusion is `topk_xl_copy`, which arrives via the
upstream kernel-migration effort.

## Test infrastructure (reference)

- **Single-LLK:** `tt_metal/tt-llk/tests/sources/<name>_test.cpp` (sections
  `LLK_TRISC_UNPACK/MATH/PACK`) + `tests/python_tests/test_<name>.py`
  (`TestConfig` + a golden from `helpers/golden_generators.py` + `passed_test`).
- **Fused / chained (LLK interactions):** `tests/python_tests/fuser_config/*.yaml`
  (auto-discovered) for declarative `unpacker → math → packer` chains; hand-written
  `tests/sources/fused_tests/*.cpp` for semaphore-trigger / dest-reuse handshakes.
- **Run** from `tt_metal/tt-llk/tests/`: `pytest --compile-producer -n <N> -x ./python_tests/test_<name>.py`
  then `pytest --compile-consumer -x ./python_tests/test_<name>.py`.
