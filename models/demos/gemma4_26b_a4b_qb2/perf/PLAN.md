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
| WS3 | A | **Stopped 16 Nov** (see below): batch 32 is expert-bandwidth-bound after WS1 | - |
| WS4 | B | `moe_decode` op (Python `generic_op` + kernels, gemma4 custom_ops style): per-DRAM-bank streaming experts, GeGLU/SwiGLU, BFP8/BFP4 | expert read ≥ 400 GB/s/chip |
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
| WS4 | new `tt/moe_decode/` package (Python `ProgramDescriptor` op with its kernels beside it, like `models/demos/gemma4/custom_ops`) + its tests; then the MoE call site. A C++ ttnn device op only if another model adopts it | anything else |
| WS5 | new `tt/fused_decode/` package + kernels; then the decode-layer switch | the non-fused path |
| WS6 | new `tt/mtp/` + vLLM adapter hooks | decode internals |

If a stage needs to change code another stage owns, tell the other agent (`hc send`) what and why, then keep
going: make the smallest change your stage needs, keep it in its own commit, and let whoever lands second
reconcile it at merge. Do not keep private copies of shared helpers: put shared tools in `perf/tools/`.

## No waiting

Each machine runs its own chain (A: WS1 → WS6, B: WS2 → WS4 → WS5) and never waits for the other.
No stage needs a stage from the other machine.
- Start each stage from `hub/integration`. If the other machine landed nothing new, that is simply your own
  last landed stage; continue from it.
- Never idle on a message, an answer or the other machine's progress. Questions are notices, not gates.
- Integration points follow whatever is on `integration` when you get there. Example: WS4 plugs `moe_decode`
  into the MoE decode path as it exists then (the tensor-parallel experts if WS1 has not landed, WS1's
  32-experts-per-chip path if it has). Whoever lands second adapts their change to the other's.

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

**Quick check, every change, ~1 min incl. model load** (`perf/tools/quick_check.py --baseline perf/results/WS0/quick_baseline.json`; refs cache in `~/gemma26-eval/refs`):
layers 0 and 5 against stored HF references at the stage's batch sizes; prints PCC per user, µs/layer and op count; fails on PCC drop or slowdown.
Until it lands, use `pytest models/demos/gemma4_26b_a4b_qb2/tests/test_device_numerics.py -k test_checkpoint_decoder_numerics` (~3 min).

**Stage gate, once per stage, ~1.5-2 h.** A stage lands only when every line below passes on the merged commit.

Common to every stage (protects the other machine's work):
1. Accuracy: `test_device_numerics` passes; every PCC ≥ 0.995 vs HF and at most 0.0005 below the previous `integration`.
2. Full-model timing: `python perf/tools/decode_timing.py --output <json>` (batch 1 at 128 and 4K, batches 8/16/32; ~10 min).
   **No regression:** batch-1 step and batch-32 step each at most 3% slower than the previous `integration`'s proof,
   unless the stage's own target is that metric.
3. Agreement: greedy 128 tokens vs the previous `integration`, top-1 agreement ≥ 99%.
4. Serving and GPQA (CI is unavailable; run locally): `perf/tools/serve_bench.sh qb2port <out> perf,gpqa`.
   GPQA-10 ≥ 8/10; serving 128/128 at 1 and 16 users completes with no failed requests.

Stage-specific proof (target met = **done**; gate passes and the metric improves ≥ 20% but target missed = **landed partial**,
stated as such in `perf/WSN.md` with the gap; less than that does not land):

| Stage | Proof of success |
|---|---|
| WS1 | batch-32 native step ≤ 35 ms (WS0: 501); batched vs serialized per user PCC ≥ 0.999 at B 2/3/8/32 with mixed positions and pages; KV writes exact per slot; inactive slots untouched; 16-user serving tok/s/user ≥ 25 ; router + experts ≤ 10 device programs per decode layer at batch 1 and 32 (WS0: 34) |
| WS2 | batch-1 native step ≤ 11.5 ms (WS0: 18.6); device programs per sliding decode layer **in WS2's scope** (everything except router and experts) ≤ 30 in a 2-layer device profile (WS0: ~98 of 132) |
| WS3 | stopped (see "WS3 decision") |
| WS4 | `moe_decode` is the **batch-1** MoE op (batch 2-32 stays on WS1's expert-parallel path, already ~490-500 GB/s); op unit tests pass vs torch at B=1 for Gemma (H 2816, I 704, E 128, k 8) and gpt-oss (H 2880, I 2880, E 32, k 4, bias) shapes, BFP8 and BFP4; expert weight read ≥ 400 GB/s per chip; MoE block ≤ 35 µs per layer in the model |
| WS5 | batch-1 native step ≤ 5 ms; 2-layer fused decode vs HF over 8 steps PCC ≥ 0.995 |
| WS6 | greedy speculative output token-identical on 10 prompts to plain greedy decoded with the verify's numerics (the WS1 batched path, one token per step); teacher-forced top-1 agreement of the batched path vs batch-1 `integration` ≥ 99 % (batch-1 and batched decode differ only at near ties); effective batch-1 tok/s/user ≥ 1.5× the previous `integration` on real prompts |

**Combined check** once WS1 and WS2 have both landed: ≤ 50 device programs per sliding decode layer at batch 1.

**Proof files:** commit the gate's evidence next to the report, in `perf/results/WSN/`: `decode_timing.json`,
`numerics.txt` (the `DECODER_NUMERICS ...` lines), the agreement result, `gpqa/summary.json` and the serving summaries
(small JSON only, no logs over 1 MB). `perf/WSN.md` quotes the numbers and states done or partial.

**Merging someone else's code:** when resolving a conflict in code the other stage owns, keep their behaviour and
rerun the quick check at their stage's batch size as well as yours. Never revert or rewrite the other stage's landed change;
if it breaks yours, tell them and adapt on your side.

The hub enforces part of this: `integration` only fast-forwards, `wsN-done` tags cannot move, and each machine can
push only its own stage branches (A: ws0/ws1/ws3/ws6, B: ws2/ws4/ws5).

Release cohort (Terminal/SWE) only at the end of the plan, run locally overnight on whichever machine is free.

## Machine setup (once)

- Clone of the hub with `--reference` to an existing tt-metal checkout, submodules, `./build_metal.sh --build-dir build_Release`, `./create_venv.sh`.
- vLLM TT plugin at `0a76258e14ca7237067ca9a22ec02e257bee570a`: clone into `~/gemma26-eval/vllm-tt-plugin`,
  activate the tt-metal venv, `source docs/install-vllm-tt.sh` from the plugin folder.
- Eval env: `uv venv --python 3.11 ~/gemma26-eval/eval-env` and install `tests/requirements-eval.txt` with `UV_TORCH_BACKEND=cpu`.
- GPQA-10 inputs: `benchmark.py --prepare-only --server-capacity 16 --output-dir ~/gemma26-eval/results/gpqa` with `HF_TOKEN` set (gated dataset; token in `~/run_tt_xla_container.sh`, never print it).
- Weights: `HF_HOME=/mnt/models/huggingface`, revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, `HF_HUB_OFFLINE=1`.

## WS3 decision (16 Nov)

After WS1 the batch-32 expert matmuls read the 32 local experts at ~490-500 GB/s per chip (`results/WS1/ops_profile_b32.txt`),
close to the DRAM peak. A dense all-expert path reads the same bytes, so WS3 as planned cannot help. Measured on the way:
`RouterTopk` one-pass top-k 65 -> 33 us per layer (bit-identical; carried on `ws6-mtp`); BFP4 batched expert down
weights save 2 ms at batch 32 but raise layer-0 error vs HF ~10x (PCC 0.99962 -> 0.99562): rejected. The remaining
batch-32 gap to 20 ms is ~90 us/layer of small ops for ~15 %; machine A moves to WS6 (larger batch-1 gain) and those
glue cuts stay a backlog.

## Status

Each agent keeps `~/.claude/efforts/<slug>.json` on its machine current (Efforts board): A `gemma26-moe-perf`, B `gemma26-moe-b`.
Landed stages are the `wsN-done` tags and `perf/WSN.md` files on `integration`.
