# ERNIE-4.5-21B-A3B prefill bring-up: breadcrumbs

Append-only log, one section per task attempt. Each entry: what was done, decisions and why,
gotchas, exact re-run command, and gate verdict. The gate spec is `tasks.yaml`, the verdicts
are in `state.json` (written only by `gate.py`), and the per-run metrics are in `results/<id>.json`.

Workflow for any agent picking up a step:
1. `python models/demos/ernie45_d_p/bringup/gate.py --next` shows the runnable tasks (all deps PASS).
2. Implement the task. Record metrics with `bringup.metrics.record(task_id, name, value)`,
   using `os.environ["ERNIE_BRINGUP_TASK"]` as the task id.
3. `python models/demos/ernie45_d_p/bringup/gate.py <id> --commit` commits only on PASS,
   tagged `[ernie45_d_p][<id>]`.
4. Append a section here.

## Global decisions (2026-09-24)
- Model: `baidu/ERNIE-4.5-21B-A3B-PT` (not in the repo; standard GQA + MoE). Agreed with the user.
- Target: chunked prefill 55k@5k (11 x 5120 = 56320 tokens), bf16 weights and activations at the start.
- Validation ladder: 2k->2k (4096) -> 8k->8k (16384) -> (b) golden 50k prefix KV + device 50k->55k -> (a) 11 chunks.
- Thresholds: >=0.99 per op/layer, >=0.98 per block, >=0.97 KV and final hidden, top-5 overlap >= 0.9.
- Input text: A Tale of Two Cities, `models/tt_transformers/tests/tale-of-two-cities.txt.bz2`, BOS-prefixed.
- Golden precision: fp32 activations from bf16 checkpoint weights (stricter than HF bf16).
- KV contract (see `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`, `gpt_oss_d_p` GQA template):
  separate K/V `[users*layers, 1, seq, head_dim]` per chip, one KV head per TP column (TP=4 matches the 4 KV
  heads exactly), bfloat8_b, DRAM 32-token round-robin. K post-RoPE in *interleaved* (Meta) order, which is
  ERNIE's native RoPE, so no permutation is needed. The producer's HF->Meta permutation of golden K must be
  skipped for this model (the ERNIE producer branch).
- Open question for P2: the contract cache is bf8 and ring_joint SDPA requires a bf8 cache. Start with a bf16
  cache and plain chunked SDPA for correctness, then switch to bf8 in P2.15 and record the PCC delta.
