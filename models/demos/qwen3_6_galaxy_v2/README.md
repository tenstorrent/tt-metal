# Qwen3.6-27B Galaxy (v2 — llama3_70b_galaxy fork)

**Status:** under bring-up. See `BRINGUP_LOG.md` for the live checkpoint
state. v1 implementation at `models/demos/qwen3_6_galaxy/` is unchanged
and remains the working end-to-end demo until v2 reaches Stage-3 parity.

## Why v2 exists

The v1 implementation (`models/demos/qwen3_6_galaxy/`) achieved correctness
(PCC > 0.99 per block, e2e Paris generation on 64 layers) but is
unoptimized: decode runs at ~1.5 tok/s eager, 73% of decode device-kernel
time is DRAM-interleaved matmuls, and trace capture is blocked on a residual
host-write site that requires the 70B-galaxy custom CCL infrastructure to
fix cleanly.

v2 is a fresh port that **starts** from `models/demos/llama3_70b_galaxy/`
— which already has sharded matmuls, L1 staging, trace replay, custom CCL,
async-decode all landed and tuned — and adapts it for qwen3.6's deltas
(hybrid decoder, DeltaNet linear attention, partial RoPE, QK-norm, output
gate, zero-centered RMSNorm).

## Bringup pattern

Mirrors `models/demos/olmo_galaxy/` exactly:
- Keep all `llama_*.py` filenames; novel behavior lives behind
  `is_qwen36 = getattr(args, "is_qwen36", False)` branches.
- Sibling `qwen_model_config.py` carries the qwen3.6 hyperparams + the
  `is_qwen36` flag.
- Wholly-novel blocks (e.g. DeltaNet, no llama analogue) live in new files
  with the `qwen36_` prefix.

Prefetcher is **disabled** in v2 (`use_prefetcher=False`). The
prefetcher-worker cores are reclaimed into the compute grid so v2 runs
matmul / CCL / norm kernels on the full device grid (minus dispatcher
cores) — see plan
`/home/tt-admin/.claude/plans/this-is-not-the-elegant-acorn.md`
section 8.

## Targets (calibrated against `models/demos/olmo_galaxy/`)

| metric | target |
|---|---|
| Decode tok/s/user (64L, trace) | ≥ 17 |
| TTFT @ 128 ISL | ≤ 350 ms |
| TTFT @ 16k ISL | ≤ 2200 ms (DeltaNet should help) |
| Decode matmul % on L1 | > 90% |

## Run (once Stage-3 lands)

    cd /home/tt-admin/ssinghal/tt-metal
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)

    python models/demos/qwen3_6_galaxy_v2/demo/demo.py \
        --prompt "The capital of France is" --num-tokens 10

Test invocations are listed in `BRINGUP_LOG.md`.
