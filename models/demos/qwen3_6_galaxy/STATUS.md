# qwen3_6_galaxy Status

## Skeleton — initial commit

Directory tree mirrors `models/demos/olmo_galaxy/`. No code yet. Next-session
work in the order from `PIVOT_PLAN.md` §Execution order.

## What lives here

- `tt/`            — forks of `models/demos/llama3_70b_galaxy/tt/` files with Qwen3.6 deltas, plus net-new `qwen36_model_config.py`, `qwen36_deltanet.py`
- `reference/`     — standalone PyTorch oracle for module tests
- `tests/`         — per-block PCC tests + full accuracy test, mirroring the `test_qwen_*_ttt.py` pattern
- `demo/`          — generation demo with BH/6U-aware expected outputs
- `conftest.py`    — fabric init for the full 8×4 BH GLX system mesh (then submesh creation if needed)

## What does NOT live here

- The single-chip prototype (kept at `models/demos/qwen3_6_27b/tt/`) — historical reference only.
- The DeltaNet kernel implementations (live at `models/experimental/gated_attention_gated_deltanet/tt/`) — imported, not duplicated.
- Documentation (ARCHITECTURE.md, PIVOT_PLAN.md, BRINGUP_LOG.md, TEST_PLAN.md, QUALIFICATION_PLAN.md) — kept at `models/demos/qwen3_6_27b/` and cross-referenced.
