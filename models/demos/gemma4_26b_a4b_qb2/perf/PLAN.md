# Gemma 4 26B-A4B on QB2: MoE performance plan

Analysis behind this plan: https://claude.ai/artifact/5Enim7ynjnYWmVfnFbT4Ab

## Goal and baseline

Baseline (WS0, 16 Nov, this branch, BFP8 release policy, native traced decode):
batch 1 **53.8 tok/s/user** (18.6 ms); batches 8 / 16 / 32 step 129 / 255 / 501 ms, so total stays
62-64 tok/s at any batch. Decode runs every user through the layer one after another.

| Stage | Owner | What | Target |
|---|---|---|---|
| WS1 | A | Batched decode: attention, router and experts once for all users; 32 whole experts per chip; batch trace buckets | 32 users: step ≤ 35 ms (from 501) |
| WS2 | B | Glue-op cut at batch 1: norms, residual, dtypes, layouts, RoPE tables, one reduction op | B1 step ~11 ms (from 18.6) |
| WS3 | A | Dense all-expert path from 16 users (gpt-oss style) | 32 users: step ≤ 20 ms |
| WS4 | B | `moe_decode` ttnn op: per-DRAM-bank streaming experts, GeGLU/SwiGLU, BFP8/BFP4 | expert read ≥ 400 GB/s/chip |
| WS5 | B | Fused decode (gpt-oss `fused_decode` style) | B1 3-5 ms |
| WS6 | A | MTP speculative decoding (port from `models/demos/gemma4`, #62806) | B1 ×1.5-2 effective |

Machines: **A** = qb2-120-p01t07 (lead Claude, terminal `qb2-120-p01t07/t21`),
**B** = qb2-120-p04t04 (Herdr address qb2-120-p04t04/gemma26-b; ssh alias qb2-49-14) (stage agent). Each agent uses **only its own QB2's chips**.

## Code ownership (to keep stages from colliding)

All work happens in `models/demos/gemma4_26b_a4b_qb2/` (plus new op dirs for WS4/WS5).
`models/autoports/google_gemma_4_26b_a4b_it/` is reference only: do not edit it.

| Stage | Owns | Must not change |
|---|---|---|
| WS1 | batch dimension in every decode path; per-slot loops (`tt/optimized_decoder.py` decode_forward, `tt/multichip_decoder.py` `_decode_shared_batch`, `tt/model.py` decode); attention batching (RoPE per user, QKV heads, cache update, SDPA); **router**; **expert placement and MoE decode**; trace buckets in `tt/generator.py` / `tt/generator_vllm.py` | norm/residual/dtype/layout code paths except for adding the batch dimension; CCL op form |
| WS2 | RMSNorms and residual adds, layer scalar, typecasts and dtype per segment, sharded↔interleaved moves, RoPE table layout, **CCL op form** (RS+AG → one reduction) | router, experts, attention batching |
| WS3 | a new dense-expert module selected by batch size | sparse path from WS1 |
| WS4 | new op under `ttnn/cpp/ttnn/operations/experimental/moe_decode/` + its tests; then the MoE call site | anything else |
| WS5 | new `tt/fused_decode/` package + kernels; then the decode-layer switch | the non-fused path |
| WS6 | new `tt/mtp/` + vLLM adapter hooks | decode internals |

If a stage needs to change code another stage owns, message the other agent first (`hc send`) and agree
who makes the change. Do not keep private copies of shared helpers: put shared tools in `perf/tools/`.

## Git: one hub, one integration branch

- Hub: bare repo `qb2-120-p01t07:/home/mvasiljev/gemma26-perf.git` (from B: `ssh://mvasiljev@qb2-120-p01t07/home/mvasiljev/gemma26-perf.git`).
- `integration`: only stages that passed their gate **as merged**. Its head is always the best checked combination.
- Stage branches: `ws1-batched`, `ws2-glue`, `ws3-dense`, `ws4-moe-op`, `ws5-fused`, `ws6-mtp`. Push them to the hub often (backup and visibility).
- Never push to GitHub. Never rewrite `integration` history.

**Starting a stage:** `git fetch hub && git switch -c wsN-<name> hub/integration`.

**Between iterations** (at least daily): `git fetch hub`; if `integration` moved, merge it into your stage branch, then rerun the quick check.

**Finishing a stage:**
1. `git fetch hub && git merge hub/integration` into the stage branch; resolve conflicts against the ownership table.
2. Pass the quick check and the stage gate (below) **on this merged commit**.
3. Write `perf/WSN.md`: what changed, numbers before/after, gate results, known limits. Commit it.
4. `git push hub HEAD:integration` (must fast-forward). If rejected, the other agent landed first: fetch, merge, rerun the quick check (and the gate if the merge touched code), retry.
5. `git tag wsN-done && git push hub wsN-done`, update your effort file, and tell the other agent:
   `hc send <other terminal> "WSN landed on integration at <sha>: <one-line result>"`.
6. Start the next stage from the new `hub/integration`.

## Tests

**Quick check, every change, ~2-4 min** (`perf/tools/quick_check.py`, from WS1's first commit):
layers 0 and 5 against stored HF references at the stage's batch sizes; prints PCC per user, µs/layer and op count; fails on PCC drop or slowdown.
Until it lands, use `pytest models/demos/gemma4_26b_a4b_qb2/tests/test_device_numerics.py -k test_checkpoint_decoder_numerics` (~3 min).

**Stage gate, once per stage, ~1.5-2 h:**
1. Accuracy: `test_device_numerics` passes; PCC per layer ≥ 0.995 vs HF and ≤ 0.0005 below the previous `integration`.
2. Full-model timing: `python perf/tools/decode_timing.py --output <json>` (batch 1 at 128 and 4K, batches 8/16/32; ~10 min). Record it in `perf/WSN.md`.
3. Agreement: greedy 128 tokens vs previous `integration`, top-1 ≥ 99%.
4. Serving and GPQA (CI is unavailable; run locally): `perf/tools/serve_bench.sh qb2port <out> perf,gpqa`.
   GPQA-10 ≥ 8/10; serving 128/128 at 1 and 16 users completes.

Release cohort (Terminal/SWE) only at the end of the plan, run locally overnight on whichever machine is free.

## Machine setup (once)

- Clone of the hub with `--reference` to an existing tt-metal checkout, submodules, `./build_metal.sh --build-dir build_Release`, `./create_venv.sh`.
- vLLM TT plugin at `0a76258e14ca7237067ca9a22ec02e257bee570a`: clone into `~/gemma26-eval/vllm-tt-plugin`,
  activate the tt-metal venv, `source docs/install-vllm-tt.sh` from the plugin folder.
- Eval env: `uv venv --python 3.11 ~/gemma26-eval/eval-env` and install `tests/requirements-eval.txt` with `UV_TORCH_BACKEND=cpu`.
- GPQA-10 inputs: `benchmark.py --prepare-only --server-capacity 16 --output-dir ~/gemma26-eval/results/gpqa` with `HF_TOKEN` set (gated dataset; token in `~/run_tt_xla_container.sh`, never print it).
- Weights: `HF_HOME=/mnt/models/huggingface`, revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, `HF_HUB_OFFLINE=1`.

## Status

Each agent keeps `~/.claude/efforts/<slug>.json` on its machine current (Efforts board): A `gemma26-moe-perf`, B `gemma26-moe-b`.
Landed stages are the `wsN-done` tags and `perf/WSN.md` files on `integration`.
