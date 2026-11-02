# Quasar tt-metal side: branch map, stack, tests, open asks

Updated 2026-11-02. The goal is a Python path from ttnn to the Metal 2.0 ProgramSpec, so that tt-blaze can dispatch on Quasar. This file covers the tt-metal side only. The blaze-facing hand-off is tt-blaze `quasar_e2e/handoff_tt_metal.md`.

**On origin.** `gchoudhary/quasar-for-tt-blaze` (`8e9302b354b`) and `gchoudhary/quasar-for-tt-blaze-pin` (`ea118c6d506`) exist on github. The main line `gchoudhary/quasar-rename-63251` is local only. There are no PRs and no GitHub posts.

## 1. Branch map

All the ProgramSpec branches fork from main at `99406df1b1f` (2026-10-30, "Fix DEVICE_PRINT dispatch for new BH firmware (#62697)"). That base includes #61952, the per-node tensor bindings.

| Branch | Worktree | Head | Purpose | Status |
|---|---|---|---|---|
| `gchoudhary/quasar-rename-63251` | `/localdev/gchoudhary/tt-metal-borrowed` | `a49e757afc8` (latest at time of writing) | **Main line.** Contains the full stack in section 2, with its own `build_Release`. | Active; all new work goes here |
| `gchoudhary/quasar-for-tt-blaze` | `/localdev/gchoudhary/tt-metal` (**shared tree**) | `8e9302b354b` | What blaze's Quasar lane builds against, built into `build_Release`. | Lacks the program-hash fix `f65732e4923` and the test commits after `8e9302b354b`. Fast-forward and rebuild only when the blaze architect says the builder's runs are done. |
| `gchoudhary/quasar-for-tt-blaze-pin` | `/localdev/gchoudhary/tt-blaze/tt-metal` (blaze's submodule; the BH lane) | `ea118c6d506` | Blaze's old pin `f5985995a23` plus the 10 bundle commits (`63a90d203e2..b00a81db4b9`, `dbad9631b7f`, `cf1ae340ae5`) | Lacks per-core tensors (`b75c7dfea80`, `2854792510b`, #61952) and `f65732e4923` |
| `gchoudhary/quasar-borrowed-offset`, `quasar-per-node-tensors`, `quasar-per-core-dfb` | none / `tt-metal-pernode` | various | Earlier single-feature branches | Superseded: re-landed on the main line |
| `gchoudhary/quasar-sram-arena` | `/localdev/gchoudhary/tt-metal-sram-arena` | `6c336567b02` | Prototype that combines #63251, #60652 (SRAMArena) and our SRAMArena bindings | Paused. It depends on #60652, and SRAMArena can't wrap an existing tensor's buffer |

**Environment gotchas:**
- `tt-metal-borrowed` has no `python_env`. Use `source /localdev/gchoudhary/tt-metal/python_env/bin/activate`.
- **PYTHONPATH:** set it to `<borrowed>/ttnn:<borrowed>`, not just `<borrowed>`. The env has an editable ttnn install that points at the shared tree, so with plain `PYTHONPATH=<borrowed>`, `import ttnn` loads the shared tree's package and build. Check `ttnn.__file__`.
- **Earlier runs:** before 2026-11-02, test runs used the shared tree's build. That is harmless, because borrowed's non-test code was identical up to `67c65a24a05`.
- **Rebuild:** `cd build_Release && ninja ttnn && ninja install`.

## 2. Main-line commit stack, oldest first (`99406df1b1f..HEAD`)

| # | Hash | Kind | What it does | Pin needs |
|---|---|---|---|---|
| 1 | `63a90d203e2` | Feature | ttnn.generic_op runs a Metal 2.0 ProgramSpec built in Python | yes |
| 2 | `89db70b1ab4` | Feature | ComputeHardwareConfig: full compute config | yes |
| 3 | `87b769aef59` | Feature | ProgramSpec scratchpads | yes |
| 4 | `26880e9c3ff` | Test-only | Rejected specs reach Python as readable errors | optional |
| 5 | `e5f5a8e05c2` | Feature | ProgramSpec semaphores | yes |
| 6 | `475849ac314` | Feature | Gen2 (Quasar) hardware configs for DM and compute | yes |
| 7 | `bcf11d01dc2` | Feature | DataflowBufferSpec.borrowed_from | yes |
| 8 | `b00a81db4b9` | Test-only | Decode matmul variants | optional |
| 9 | `b307d9f5bac` | Test-only | Decode matmul: row blocks, multicast, custom_mm. Its message claims Blackhole silicon, but custom_mm never ran here (no silicon lane) | optional |
| 10 | `dbad9631b7f` | Feature | **Bundle:** DFBAdvancedOptions.borrowed_offset | **yes** |
| 11 | `cf1ae340ae5` | Feature | **Bundle:** `ttnn.experimental.*` surface with #63251's names | **yes** |
| 12 | `7432e50a209` | Test-only | Two-core gather into a planned arena region | optional |
| 13 | `1ededa9e2db` | Test-only | Compute <-> DM phase barriers through token DFBs | optional |
| 14 | `aec941a7bc2` | Test-only | Token push and pop with dummy_pack/dummy_unpack (TEN-4746) | optional |
| 15 | `b75c7dfea80` | Feature | generic_op accepts per-core allocated tensors | **needs #61952** |
| 16 | `2854792510b` | Feature | Borrowed DFBs on per-core allocated tensors (slow dispatch only) | **needs #61952** |
| 17-29 | `61be8b33563` .. `12552613eea` | Test-only | Compute-API probes and numerics, RoPE, swiglu, Mcast/Matmul/Gather, flash, fused RMSNorm, top-k, FIFO multiplex, arena handoff | no |
| 30 | `dd1482dda05` | Test-only | Per-part format switch inside one compute kernel | no |
| 31 | `9ec1fb0666c` | Test-only | Per-part reconfig switches bf16 and fp32 source registers | no |
| 32 | `07102f31d8d` | Test-only | Mcast -> Matmul -> Gather on 4x2 and 8x4 grids | no |
| 33 | `d9dcf1c514e` | Test-only | Mcast -> Matmul -> Gather -> Copy in node-class work units. The token-order mutant in its docstring is not committed | no |
| 34 | `7ee56aa2058` | Test-only | tilize_block and pack_untilize_block as blaze's KV-cache updates call them | no |
| 35 | `039adf1602a` | Test-only | Raw semaphore pointer misses a NoC increment on Quasar DM | no |
| 36 | `320487edac2` | Test-only | SFPU binary ops on tiny tiles are wrong on Quasar. It asserts the buggy output, so it goes red when the bug is fixed | no |
| 37 | `67c65a24a05` | Test-only | Tiny tile through a compute self-loop DFB | no |
| 38 | `f65732e4923` | **Bug fix** | `hash_program_spec` hashes DFB `alias_with` (sorted) and whether data formats are set | **yes** (blaze uses aliasing) |
| 39 | `04764d09dae` | Test-only | Node-class work-unit test runs on Quasar only (BH allows 2 DM kernels per node) | no |
| 40 | `02782c0d817` | Test-only | TRISC plain load of L1 sees a DM store | no |
| 41 | `90cdb3ff723` | Test-only | CPU stores across a multi-entry DFB write lock | no |
| 42 | `a49e757afc8` | Test-only | `probes/`: L1 budget calculator, blaze conftest shim, probe README | no |

What blaze's pin needs:
- Features 1-3, 5-7 and the bundle (`dbad9631b7f` + `cf1ae340ae5`).
- `f65732e4923`, if specs differ only in `alias_with`.
- `b75c7dfea80` and `2854792510b`, only if blaze uses per-core tensors. Both need #61952 (`97f77fcb63c`), which is not in the pin.

## 3. Test inventory

All files are under `tests/ttnn/unit_tests/operations/debug/`.

| File | What it proves | Quasar | BH |
|---|---|---|---|
| `test_generic_spec_op.py` | Base binding: DFBs, compute config, scratchpads (Quasar DM needs `pad.scoped_lock()`), semaphores (handshake and ping-pong; the missing-wait mutant fails on Quasar), Gen2 configs, readable rejections | craq-sim pass; emu-quasar-2x3: 25/25 (W+A) | ttsim pass with slow and fast dispatch; at pin 22 pass, 5 skip (RVV) |
| `test_generic_spec_op_matmul.py` | Decode matmul with borrowed_from: matmul_block, fp32 DST, NoC read and multicast across 2 cores, persistent loop | craq-sim 44 pass, 4 skip (custom_mm is BH-only); emu 2x3 pass | ttsim pass; custom_mm dense packing needs BH silicon |
| `test_generic_spec_op_borrowed_offset.py` | borrowed_offset: FIFOs side by side, packed, and phase reuse; program-cache hit with new tensors; rejects misaligned and out-of-range offsets and mismatched aliases | craq-sim 8/8 (W+A) | ttsim 8/8, also at the pin |
| `test_generic_spec_op_gather.py` | Two-core gather into a planned arena region | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_token_barrier.py` | Compute <-> DM phase barriers through token DFBs. Needs dummy_pack/dummy_unpack (TEN-4746). | craq-sim and emulator pass with asserts | ttsim pass |
| `test_generic_spec_op_per_node_tensors.py` | Per-core allocated tensors through generic_op | craq-sim pass | ttsim pass |
| `test_generic_spec_op_per_core_borrowed_dfb.py` | Borrowed DFBs at per-core addresses (slow dispatch only); fast dispatch is rejected | craq-sim pass (W+A) | ttsim pass |
| `quasar_api_probe.py` (a script, not pytest) | Compile probes for the compute APIs blaze needs on Quasar | craq-sim, ARCH_NAME=quasar, with LLK asserts | n/a |
| `test_generic_spec_op_compute_apis.py` | Numerics of binary, bcast, reduce, rsqrt, silu, mul_unary and reuse_dest | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_rope_recompose.py` | Decode RoPE. Needs pack_init per output DFB on Quasar. | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_swiglu_no_fence.py` | swiglu without the FPU_SFPU fence, which doesn't exist on Quasar | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_mcast_matmul_gather.py` | Mcast (ready/valid, NoC-1 corner swap, two rectangles) -> matmul -> gather (set_multicast + wait_min) on 2 cores | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_mcast_matmul_gather_grid.py` | The same chain on 4x2 and 8x4 grids | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_flash_recompose.py` | Decode flash attention, and GQA 4:1 with a position mask | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_fused_rmsnorm_matmul.py` | RMSNorm and matmul in one compute kernel. Uses full tiles (#47597); eps comes from an eps tile + add_binary_tile because Quasar has no add_unary_tile. | craq-sim pass (W+A); the NO_REINIT mutant trips the pack guard as expected | ttsim pass |
| `test_generic_spec_op_topk_recompose.py` | Top-k values, then indices with exact one-at-a-time tie handling | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_fifo_multiplex.py` | Option 4c: one DFB carries two logical FIFOs per direction, 2 iterations | craq-sim pass (W+A) | ttsim pass |
| `test_generic_spec_op_arena_handoff.py` | Option 3: A/B program alternation over 3 tokens, arena handoff via borrowed_from + borrowed_offset, exactly 2 cache entries. The wrong-offset mutant fails (PCC 0.70). | craq-sim pass (W+A) | ttsim pass |

New since this list was first written: `test_generic_spec_op_dm_cache_semaphore.py`, `_sfpu_binary_tiny_tile.py`, `_tiny_tile_self_loop.py`, `_trisc_l1_read.py`, `_dfb_multi_entry_lock.py`, `_tilize_untilize.py`, `_mcast_matmul_gather_work_units.py` (Quasar-only) and `_format_switch{,_fp32}.py`, plus `probes/` (see its README).

**Quasar env (craq-sim):**
- `ARCH_NAME=quasar`
- `TT_METAL_SIMULATOR=/localdev/gchoudhary/quasar-sim/libttsim.so` (the user stages it; record its md5 per run)
- `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_DISABLE_SFPLOADMACRO=1`
- `TT_METAL_WATCHER=10ms` (not `=1`: the host poll sleep starves the sim clock; see `probes/README.md`), `TT_METAL_LLK_ASSERTS=1`, `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1` ("W+A" in the table)
- Give each parallel lane its own `TT_DEVICE_LOCK_PATH` and `TT_METAL_CACHE`.
- Run one test file per pytest invocation, with `--timeout` raised (2800 for the larger tests).

Quasar acceptance requires W+A, because without asserts, TEN-4746 and the pack re-init guard fail silently.

**BH env:** `ARCH_NAME=blackhole` and the ttsim v1.15 BH build (stage `/localdev/gchoudhary/ttsim/src/_out/release_bh/libttsim.so` with `blackhole_140_arch.yaml` as `soc_descriptor.yaml`; md5 `951db8b834313ba6d3c32ff8e2dab131`), with slow dispatch.

## 4. Re-cherry-picking onto blaze's pin

1. In blaze's submodule (`/localdev/gchoudhary/tt-blaze/tt-metal`, branch `gchoudhary/quasar-for-tt-blaze-pin`, currently `9447faf8f02` = pin `f5985995a23` + `63a90d203e2..b00a81db4b9`), run `git cherry-pick b307d9f5bac dbad9631b7f cf1ae340ae5`.
   - To start fresh instead: on a branch from `f5985995a23`, run `git cherry-pick 99406df1b1f..cf1ae340ae5`.
   - That set was verified to cherry-pick without conflicts and build ttnn at the pin.
   - On BH ttsim at the pin: borrowed_offset 8 pass; base 22 pass, 5 skip (the RVV test skips because the pin has no RVV options); matmul 43 pass, 5 skip.
2. If blaze needs per-core tensors, cherry-pick #61952 (`97f77fcb63c`) first; its only conflict is a test README. Then cherry-pick `b75c7dfea80 2854792510b`.
3. Test-only commits are optional; cherry-pick them only for the tests blaze wants.
4. Rebuild ttnn (`./build_metal.sh`), then run `test_generic_spec_op_borrowed_offset.py` and `test_generic_spec_op.py` on BH ttsim and Quasar craq-sim (W+A), one file per invocation.

The code tolerates the pin's `uint32_t` vs main's `ArgValue`. Don't move blaze's pin itself without the user.

**Pin check, 2026-10-31.** tt-blaze `origin/main` (`d683d5c2bf`) has moved the tt-metal gitlink from `f5985995a23` to `58afa857945` (2026-10-28, "#62799 WH SFPU atan/asin/acos"). That is 206 commits ahead, and the old pin is an ancestor. The pin branch head is now `ea118c6d506` (pin + the 10 bundle commits), not `9447faf8f02`.
- The bundle cherry-picks onto `58afa857945` without conflicts in a scratch worktree (no build). The bundle is `63a90d203e2..b00a81db4b9` (8 commits), then `dbad9631b7f` and `cf1ae340ae5`.
- Named features in the range: none of #61952, #63251, #60652 or #62318 is in it. Nothing upstream touches `ttnn/cpp/ttnn/operations/generic` or DFB `borrowed_from`.
  - #61952 (`97f77fcb63c`, 2026-10-28 21:36) landed just after the new pin. So step 2 still applies.
- Upstream changes to files the bundle touches, all applied cleanly:
  - #62552 `ArgValue`: the binding still writes `uint32_t` values, which it keeps compiling, and `hash_program_spec` already handles both.
  - #60969: scratchpad multi-bind.
  - #58175: MX data types in TTNN.
  - #59063: RVV on Quasar, so the RVV test should stop skipping at the new pin.
  - #61370: umd include paths.
  - #61626, #61385: kernel hash and runtime-arg-count checks.
- A build and the BH ttsim reruns at the new pin are still unverified. Do them before blaze moves.

## 5. Open upstream items and strategic asks waiting on the user

**Strategic asks.** Each is a runtime/firmware design change, not something to start without the user:
- **DFB phase reconfig.** There is no in-program DFB slot reuse: DFB config is set once per launch. A runtime-owned phase reconfig is about 4-6 weeks and high risk. Until then, Option 3 (split programs plus arena handoff, proven in `12552613eea`) and Option 4c (`87585952c23`) cover the budget: GQA reaches about 22 of 24 TRISC-active, and GLM peaks at 23 slots, with phase 2 as the pinch point.
- **Multi-kernel-per-node / INTER DFBs.** There is one compute KernelSpec per node, threads are SPMD, and every DFB counts on every Neo. So multi-Neo compute gives no DFB headroom. TensixScope::INTER is rejected ("not yet supported").
- **Fast dispatch for per-core borrowed DFBs.** `2854792510b` is slow-dispatch only, and fast dispatch rejects it. Lifting that needs one address per core in the fast-dispatch DFB config path.
- **Multi-chip transport.** Quasar has no Ethernet cores (`eth: []`), so there is no fabric. craq-sim has no inter-chip Quasar model. A multi-chip Quasar pipeline is not possible today.
- **#47597 tiny tiles.** Quasar's binary, bcast, reduce and unary-broadcast paths reject non-32x32 tiles; only copy and the standard matmul accept them. tt-blaze-01 ruled that we pad to full tiles meanwhile. The issue is open with no activity since 2026-07-30. Related open PRs: #63381, #62267, #63370/#63369.

**Upstream PRs to watch** (all open unless noted):
- #63251/#63250: generic_op ProgramSpec. Our `cf1ae340ae5` mirrors its names.
- #60652: SRAMArena.
- #60806: Quasar pack re-aim; it would drop the pack_init-per-output rule.
- #62204: DFB finish moves to firmware.
- #63516: Quasar topk.
- #62164, #63433.
- #61952: per-node tensor bindings. Merged 2026-10-29 as `97f77fcb63c`.

**Quasar compute rules we follow** until those land:
- dummy_pack/dummy_unpack around token-only push or pop (TEN-4746).
- pack_init when changing the output DFB.
- Re-run the op init instead of reconfig_full_operand.
- Use only mul_unary_tile among scalar ops.
- Compute semaphores only pair UNPACK and PACK.
- Pad to full tiles.
