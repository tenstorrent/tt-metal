# Proposed tt-metal issues from QB2 stage-05 (full-model) bring-up

Genuine **tt-metal framework** bugs surfaced while bringing up the 8 QB2 models on Blackhole P300 (1×4 mesh,
TP4), with standalone repros. **Nothing here has been filed** — these are local drafts + repros for the metal
team to review and file. Each entry says whether to **file new**, **augment an existing issue**, or **not a bug**.

Repros live in [`repros/`](repros/); full issue drafts are the `issue-*.md` files beside this one. Detailed
analysis + per-model evidence: agentic-research branch `mvasiljevic/qb2-metal-blockers`
(`forge_experiments/qb2-experiments/metal-blockers-scorecard.md`).

## Summary

| ID | Title | Verdict | Repro | Existing issue | Draft |
| --- | --- | --- | --- | --- | --- |
| **B1** | `all_gather_async` writer asserts on a one-tile scatter-write packet (`minimal_default_writer.cpp` → `api_common.h` `chunk_count>=2`) | **FILE NEW** — framework bug, no exact match | `repros/ccl_one_tile_scatter_assert.py` | none (siblings #48222/#48404/#48469/#40592) | `issue-B1-one-tile-scatter-assert.md` |
| **W1** | watcher+fabric kernel-config overflow on ACTIVE_ETH at `open_mesh_device(1×4)` | **FILE NEW** — framework/tooling, no match (confirm on `main`) | inline in draft (3-line snippet) | none | `issue-watcher-eth-config-overflow.md` |
| **B6** | `ttnn.sampling` inexact over a wide multi-device candidate buffer (scales with TP) | **AUGMENT** — already tracked | in-tree `test_sampling_1d.py::test_ttnn_sampling_isolation` | **#48222** (OPEN) | add BH 1×4 data point to #48222 |
| B2 | host stall: buffers allocated while a captured trace is live | **NOT A BUG** (framework warned correctly, `allocator.cpp:110`) | `repros/trace_unsafe_alloc.py` | (#48469 retracted) | model-side fix |
| B3 | L1 circular-buffer clash w/ persistent L1 buffer | **NOT A BUG** (allocator flagged correctly) | `repros/llama70b_l1_cb_collision.py` | — | model-side deallocs |
| B4 | long-context DRAM per-bank OOM | **NOT A BUG** (physical limit) | `repros/qwen_coder_dram_capacity.py` | — | lower advertised context |
| B5 | host-side shard-grid TT_FATAL (falcon3-10b) | **NOT A BUG** (bad shard spec) | (config-specific) | — | model-side sharding |

## FILE NEW

### B1 — one-tile scatter-write assert in `all_gather_async` (primary)
- **Real core:** `minimal_default_writer.cpp` unconditionally pre-initializes a fabric *scatter*-write header
  with `chunk_count = num_tiles_to_write_per_packet` at kernel start (guarded only on `valid_targets`). When a
  tile ≥ the fabric packet payload (FP32 4 KiB tile → 1 tile/packet), that is `chunk_count==1`, and
  `populate_unicast_scatter_write_fields` (`tt_metal/fabric/hw/inc/api_common.h`) asserts
  `chunk_count>=NOC_SCATTER_WRITE_MIN_CHUNKS(=2)`. The send loop **never uses that header for 1-tile packets**
  (`if (>1) scatter else unicast`), so it is dead pre-init; and `ASSERT` is a no-op without watcher, so the fault
  only fires under the readiness watcher gate.
- **Hit by 7/8 QB2 models**; blocked gpt-oss / qwen3 / qwen-coder at stage-05 (each re-derived it, ~1–2 h/run).
- **Suggested fix (validated on 4 BH devices):** guard the pre-init with
  `if constexpr (num_tiles_to_write_per_packet > 1) { … scatter set_state … }` (mirrors the send loop's own
  condition; leaves the correct `api_common.h` HW contract intact).
- **Repro:** `repros/ccl_one_tile_scatter_assert.py` — 1×4 mesh, no model; bf16 pass-control + FP32 fail-trigger.
- Full draft: `issue-B1-one-tile-scatter-assert.md`.

### W1 — watcher+fabric ACTIVE_ETH kernel-config overflow (secondary)
- `open_mesh_device(1×4)` under `TT_METAL_WATCHER` with any fabric config fails at `program.cpp:2483`
  (`Program size 27776/27920 > 25600 on ACTIVE_ETH`). Seen in real runs (falcon3-7b, mistral). Blocks
  watcher-gated CCL work on 1×4 fabric meshes (incl. the B1 repro).
- **Needs confirmation on current `main`** (checkout was ~1089 commits behind).
- Full draft: `issue-watcher-eth-config-overflow.md`.

## AUGMENT EXISTING

### B6 — `ttnn.sampling` inexact over wide candidate buffer → **#48222** (OPEN)
Do **not** file new. #48222 already localizes this to `ttnn.sampling`'s reduction (in-tree
`test_ttnn_sampling_isolation` mismatches argmax, scaling 4/32→7/32 at 1×2→1×8, v32000; trace ruled out).
Suggested: add a comment to #48222 noting it also reproduces on **Blackhole P300 1×4 (TP4)** during QB2
stage-05, i.e. not T3000-only. (It did not block QB2 — accuracy passed top-5 100% / top-100 100%.)

## NOT FRAMEWORK BUGS (recorded so they are not re-filed)
- **B2** — tt-metal *correctly warned* (`allocator.cpp:110`) that allocating under a live trace is unsafe; the
  gpt-oss harness retained decode traces across `reset()`. Fix = release-before-reset (model-side). The related
  "trace-unsafe barrier" **#48469 was retracted** by the CCL team.
- **B3** — `validate_circular_buffer_region` correctly caught a CB region overlapping a persistent L1 buffer; the
  ported decode path was missing inter-layer last-use deallocations. Model-side.
- **B4** — genuine physical DRAM limit (fragmentation ruled out on a cold mesh); lower advertised context.
- **B5** — `tensor_spec.cpp:153` / `matmul_device_operation.cpp:224` correctly rejected shard specs exceeding the
  core grid. Model-side sharding config.
