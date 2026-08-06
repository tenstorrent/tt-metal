<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 on Tenstorrent (P150x4)

TTNN bring-up + vLLM serving of [`poolside/Laguna-XS-2.1`](https://huggingface.co/poolside/Laguna-XS-2.1),
a ~31B GLM/Qwen3-style MoE (256 experts, top-8, shared expert; 40 layers, 10 full-attention + 30
sliding-window(512); hybrid KV; router `sigmoid(logits)+e_bias`, `norm_topk_prob`, no router bias).

- **Target mesh:** Blackhole **P150x4** (1×4), TP=4 / EP=4, `FABRIC_1D_RING`.
- **Precision (selected policy):** BF16 activations/norms/router, BFP8 attn/dense/shared + KV + LM-head,
  BFP4 routed experts, fp32/HiFi4 SDPA. See `doc/datatype_sweep/`.
- **Serving:** via the Tenstorrent vLLM plugin; adapter is `tt/generator_vllm.py`.

## Context / capability

| Quantity | Value | Note |
|---|---|---|
| HF config context | **262,144** | what the checkpoint declares |
| **Advertised = servable context** | **131,072** | verified on P150x4; the value the server advertises |
| Verified-servable ISL (2026-07-31 sweep) | 128 … 131,072 | `doc/vllm_integration/sweep_vllm.tsv` |
| OOM | 262,144 | restoring it is Tier-2 (hybrid-KV + shared-RoPE frees) |

The advertised context **equals** the verified-servable context by construction (`ADVERTISED_MAX_CONTEXT`
in `tt/generator_vllm.py`): the model never accepts a context it cannot serve. See
`doc/context_contract.json` → `serving_verified` for the recorded limiting reason.

## Determinism contract

> **Prefix caching is ON by default (`TT_LAGUNA_PREFIX_CACHE=1`), and partial-hit reads are NOT
> bit-reproducible.** This is a deliberate, accepted property — read this before interpreting your own
> results (e.g. a pass@1 number or a replayed trajectory).

- **Full-hit and cold (no cache reuse) reads are bit-exact** vs a no-cache baseline. A prompt re-sent
  verbatim reproduces its previous generation exactly.
- **Partial-hit reads (cached prefix + new suffix) are NOT bit-exact.** The output matches the cold
  baseline for the deterministic head of generation, then can diverge at a high-entropy (near-tie)
  token. Cause: the suffix read (one chunked-SDPA call from `chunk_start_idx=K`) accumulates in a
  different floating-point order than the cold path (pipelined multi-chunk / local-bf16 prefill). This
  is the same non-determinism prefix caching exhibits on GPUs, inherent to quantized hardware — **not a
  correctness bug**. Details: `doc/vllm_integration/STATUS.md`.
- **Consequence for agentic coding:** with prefix caching on, a failed trajectory may not be
  bit-reproducible and a pass@1 figure may not be independently re-derivable token-for-token.
- **Bit-reproducible mode:** set **`TT_LAGUNA_PREFIX_CACHE=0`** to force every request onto the cold
  path (bit-exact, at the cost of no cache reuse — long prompts re-prefill in full). Use this when a run
  must be exactly replayable.

## Serving (P150x4)

```bash
cd /tmp
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal:$TT_METAL_HOME/vllm:$TT_METAL_HOME/vllm/plugins/vllm-tt-plugin/src
export TT_LAGUNA_PIPE_CHUNK=2048            # route prefill > 2048 onto the bounded-footprint pipelined path
export TT_LAGUNA_PREFIX_CACHE=1             # default; set 0 for bit-reproducible runs (see contract above)
# NOTE: TT_LAGUNA_* only reach the worker via the "env_passthrough" list in --tt-config below; the worker's
# default allowlist is VLLM_*/MESH_DEVICE only (launcher.py:262), so these exports are otherwise silently dropped.

python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/poolside_laguna_xs_2_1 --hf-model poolside/Laguna-XS-2.1 \
  --mesh-device P150x4 --stages serve \
  --max-num-seqs 8 --block-size 64 --max-model-len 131072 \
  --tt-config '{"trace_region_size":1500000000,"fabric_config":"FABRIC_1D_RING","env_passthrough":["VLLM_*","MESH_DEVICE","TT_LAGUNA_*","TT_METAL_*","PYTHONPATH"]}' \
  --additional-server-args='--trust-remote-code --max-num-batched-tokens 131072 --enable-prefix-caching'
```

Supported concurrency target is **8** (`--max-num-seqs 8`); conc 32 collapses (TTFT into hundreds of
seconds). Recorded evidence + the cap rationale: `doc/vllm_integration/STATUS.md`.

> **Warmup / prefill note (item 1.1):** every prefill program shape is compiled *before* the decode
> trace is captured. `warmup_model_prefill` warms the full power-of-two bucket ladder up to the servable
> context, so a long prompt never first-compiles a pipelined-reassembly program under the resident
> decode trace. `TT_LAGUNA_PREFILL_WARM_CAP` can lower the warmed ceiling for fast dev iteration, but
> prompts longer than it then compile under the trace — a warning is logged; do not set it below
> `--max-model-len` for serving.

## Tests

- `tests/test_prefill_buckets.py` — device-free invariants for the prefill warm-set / advertised-context
  contract (items 1.1 + 1.2). Fast; runs in CI.
- `tests/test_optimized_decoder.py`, `tests/test_multichip_decoder.py` — layer PCC ≥ 0.995 vs HF.
- `tests/full_model_checks.py` — prefill top-1/5/100 vs the AIME24 reference.

## Status docs

**`doc/vllm_integration/STATUS.md`** is the single canonical status record (implemented / current numbers /
caveats / blockers / how-to-serve). Also: `doc/context_contract.json` (machine-readable capability),
`doc/vllm_integration/smoke_test.md` (runbook), `resource_utilization_plan.md` (optimization backlog),
`sweep_vllm.tsv` (served-ISL evidence), and the serving-latency + agent-benchmark HTML report
(https://claude.ai/code/artifact/aa902432-303a-43ee-b387-56dcd6bab3b3).
