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

## Server (tt-inference-server, text-only, BH Galaxy)

Qwen3.6-27B is wired into `tt-inference-server`'s vLLM OpenAI API for
`BLACKHOLE_GALAXY` (32× P150), text-only. The serving class is
`models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py:Qwen3_5ForConditionalGeneration`,
registered in the tt-vllm-plugin as `TTQwen3_5ForConditionalGeneration`.

Start the server (on the BH Galaxy host):

    cd tt-inference-server
    export TT_METAL_HOME=/home/tt-admin/ssinghal/qwen36/new/tt-metal
    MODEL_SPECS_ENV=dev python run.py --model Qwen/Qwen3.6-27B --workflow server \
        --local-server --tt-device tt-galaxy-bh --skip-system-sw-validation

Smoke test:

    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.6-27B","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16}'

Off-device integration tests:

    pytest models/demos/qwen3_6_galaxy_v2/tests/test_generator_vllm_import.py -v
    # in tt-inference-server/tt-vllm-plugin:
    python -m pytest tests/test_qwen3_5_config.py -v
    # in tt-inference-server:
    MODEL_SPECS_ENV=dev python -m pytest tests/test_qwen36_model_spec.py -v

### Known limitations / notes

- **Text-only.** The vision / multimodal path (`qwen36_mm_*`, `vision_*`) and
  MTP speculative decoding are not served.
- **`transformers` stays at 4.53.0.** The checkpoint's `qwen3_5` arch is in no
  public transformers release; the tt-vllm-plugin registers a thin `qwen3_5`
  `AutoConfig` and the generator loads weights from raw safetensors. Do not
  bump transformers or use `AutoModelForCausalLM` for this checkpoint.
- **TP-only:** the generator asserts `tt_data_parallel == 1`; do not set
  `data_parallel_size > 1` in the catalog entry.
- **Device validation pending.** `status=EXPERIMENTAL`. See the
  `2026-06-03` device hand-off checklist in `BRINGUP_LOG.md` for the
  `override_tt_config` / firmware / arch-yaml values to fill in on hardware
  before promoting to FUNCTIONAL.
