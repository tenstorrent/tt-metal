# Qwen3.6-27B on Blackhole Galaxy

Mesh-sharded (TP=32, 8×4 BH GLX) implementation of Qwen/Qwen3.6-27B (qwen3-next
architecture: hybrid 48× Gated-DeltaNet + 16× Gated-Attention with output gate,
partial RoPE, MRoPE).

**Pattern**: forked from `models/demos/llama3_70b_galaxy/` (production Qwen3-32B
on Galaxy) following the same fork pattern as `models/demos/olmo_galaxy/`.

## Status

Bring-up in progress. See `BRINGUP_LOG.md` (TBD).

For background, the design rationale, the test plan, and the foresight on
risks, see the cross-referenced docs:

- `models/demos/qwen3_6_27b/ARCHITECTURE.md` — model spec, weight layout,
  parallelization plan, memory budget, bottleneck analysis.
- `models/demos/qwen3_6_27b/PIVOT_PLAN.md` — what we fork from
  `llama3_70b_galaxy`, what we pad-and-slice, what's net-new (DeltaNet block,
  output-gated attention, MRoPE), execution order, risk register.
- `models/demos/qwen3_6_27b/QUALIFICATION_PLAN.md` — branch landscape (which
  pre-existing branches we drew material from).
- `models/demos/qwen3_6_27b/TEST_PLAN.md` — TDD test contract.
- `models/demos/qwen3_6_27b/BRINGUP_LOG.md` — running log of all PCC
  measurements from the single-chip prototype phase (already completed:
  end-to-end "The capital of France is" → " Paris" on single BH chip with all
  64 layers in 238s, PCC chain validated layer-by-layer vs HF).
- `models/demos/qwen3_6_27b/reference/{model,gated_delta_net}.py` — standalone
  PyTorch reference (will be ported into this directory's `reference/qwen36.py`).

## Pad-and-slice strategy

Following the `olmo_galaxy` pattern, dimensions that don't match Qwen3-32B's
tiled layout are padded up to the existing shard configs, then sliced back:

| Dim | Qwen3.6-27B | Qwen3-32B target | Approach |
|---|---:|---:|---|
| n_q per col (4 cols) | 6 | 16 | Zero-pad Q-proj output dim by 10 heads/col; slice WO input accordingly |
| n_kv per col | 1 | 2 | Zero-pad K/V output dim by 1 head/col |
| intermediate per row (8 rows) | 2176 | 3200 | Zero-pad FF1/FF3 output, FF2 input; slice FF2 output back |
| vocab | 248320 | n/a | Pad to 248832 (= 32 × 7776) for tile-aligned LM head |
| head_dim | 256 | 128 | Cannot pad — re-derive ~10 hardcoded sites in `llama_attention.py` |

This preserves every memcfg, matmul program config, and CCL primitive from the
Qwen3-32B production path. Zero-padded regions contribute zero to the output.
