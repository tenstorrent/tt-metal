# QUASAR_UPLIFT_REPORT — ttnn.reshape (test_reshape.py)

**Status: GREEN — Quasar uplift not applicable to the path this test exercises.**
Every captured call resolves **host-side** (a zero-cost metadata view or an outright no-op);
no device program is created, so there is nothing to uplift. No changes were made
(a "no changes needed" GREEN per `quasar_porting.md` — do not manufacture changes).

- Recipe executed: `docs/source/ttnn/ttnn/ai/quasar_porting.md` + `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/quasar_audit.md`
- Test: `models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py`
- Op entry point: `ttnn::reshape` — `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp`

## Per-case routing (why every case is host-side)

All routing decisions are in `ttnn::reshape` (`reshape_view/reshape.cpp:562`,
after `shape_corrector` resolves any `-1` dims):

| Case | Call | Path taken |
|---|---|---|
| `00_32x3072_bf16_int-l1` (×308) | `reshape(t[1,1,32,3072] TILE L1-interleaved, logical=(1,1,1,3072), padded=(1,1,32,3072))` | `this_is_view` is false (second-last dim 1 vs 32, `1 % 32 != 0`) and logical volumes differ (3072 vs 98304) → `tile_tensor_view_reshape_possible` (padded inner-2D tile-aligned, last padded dim unchanged) → `ttnn::experimental::view` — the legacy issue-15137/15558 metadata-view path |
| `01_32x2048_bf16_ws-l1` (×307) | `reshape(t[1,1,32,2048] width-sharded, (1,1,32,2048))` | shapes identical → "First Case, No reshape Required" early return of the input tensor |
| `02_1024x2048_bf16_int-dram` (×32) | `reshape(t[1,1,1024,2048], [1,1,1024,-1])` | `infer_dims_for_reshape` → `(1,1,1024,2048)` == input → early return |
| `03_32x1024x64_bf16_int-dram` (×32) | `reshape(t[1,32,1024,64], [1,32,-1,64])` | inferred shape == input → early return |
| `04_1024x2048_bf8_int-dram` (×32) | `reshape(t[1,1,1024,2048] BFLOAT8_B, (1,1,1024,2048))` | shape == input → early return |

The one non-trivial path, `ttnn::experimental::view`
(`ttnn/cpp/ttnn/operations/experimental/reshape/view.cpp` → `tt::tt_metal::view` →
`view_device` in `ttnn/core/tensor/tensor_ops.cpp:321`), constructs a new `TensorSpec`
and re-wraps the **same** `MeshBuffer` address. No `Program`/`ProgramArtifacts`, no
kernels, no CBs/DFBs, no semaphores — nothing architecture-specific executes on device.
It is arch-agnostic host code and works identically on WH/BH/Quasar.

## Metal 2.0 gate (§1 of quasar_porting.md)

Not applicable to the exercised path (no device program). For the record, the underlying
**device** reshape op that these paths bypass IS already Metal 2.0 on Gen1:
`ttnn::prim::reshape_view` at
`ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/` — both
`ReshapeViewTiledProgramFactory` and `ReshapeViewRMProgramFactory` implement
`create_program_artifacts` → `ProgramArtifacts`. That op was **not** audited or uplifted
here because this test never invokes it.

## Underlying device ops (reached only by reshape shapes this test does not capture)

A genuinely data-moving `ttnn.reshape` (last-dim change, layout/sharding mismatch, etc.)
would dispatch to:
- `ttnn::prim::reshape_view` — `ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/` (Metal 2.0)
- and, on some branches, helper ops: `ttnn::sharded_to_interleaved` / `ttnn::interleaved_to_sharded`,
  `ttnn::typecast` (block-float), `ttnn::fill_implicit_tile_padding`, and
  `ttnn::operations::data_movement::reshape_on_device` (separate op, different entry point).

Each of those would need its own Quasar-uplift audit if a future capture exercises it.

## Files changed

**None.** Zero source diff. (This report file is the only artifact; delete before merge.)

## §7–§8 gotchas applied / considered

- **Applied:** none — no device code on the exercised path, and §7–§8 fixes are reactive
  (no device run was performed in this session, per protocol).
- **Considered, not needed:** implicit-sync (§7) — no DFBs exist on this path; non-zero-init
  semaphores (quasar_audit.md check 2) — no semaphores; DM self-loop / CB redesign
  (quasar_audit.md check 1) — no CBs/DFBs; Int32/uint16 format quirks (§7) — the view
  forwards dtype without any format branch; `fifo_page_size` / `get_entry_size` (§5, §8.3) —
  no kernels; NoC/mcast (§11) — no NoC traffic.

## Deferred / follow-up items

- None required for this test. Optional follow-up: run the Quasar-uplift audit on
  `ttnn::prim::reshape_view` (already Metal 2.0) before any model change introduces a
  captured reshape that actually moves data.

## WH/BH parity claim (structural — no device run this session)

The working-tree diff for this task is **empty** (no source file touched), so WH/BH
behavior is unchanged by construction. Confirmation commands (user-run, per recipe §9):

```bash
# BH / WH parity (same command on each machine):
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py

# Broader reshape regression net on WH/BH:
pytest tests/ttnn/unit_tests/operations/data_movement/test_reshape.py

# Quasar (emulator, per the craqsim runbook env):
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_reshape.py
```
