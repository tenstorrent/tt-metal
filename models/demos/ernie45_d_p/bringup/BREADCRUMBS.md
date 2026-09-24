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

## P1.1-P1.5 (2026-09-24): CPU reference and goldens. PASS
- `reference/ernie_ref.py` is independent of HF. It matches HF `Ernie4_5_MoeForCausalLM` at PCC 1.0 per layer (max abs
  1e-3, fp32) with 100% top-1 agreement at 512 tokens (P1.2). Chunked == one-shot on CPU to 1e-12 (P1.3).
- Gotcha: HF ERNIE RoPE is *interleaved* (`x[0::2], x[1::2]`), not rotate-half. cos/sin are repeat_interleaved.
- Router: softmax in fp32. `e_score_correction_bias` is added only for top-k *selection*; weights are the
  unbiased probs renormalized with `clamp(sum, 1e-12)`.
- Golden cost on 16 EPYC cores: 2k chunk about 30 s, 8k chunk about 2-4 min, 5k chunk at 50k context about 3+ min.
- Re-run: `python models/demos/ernie45_d_p/bringup/gate.py P1.4` (output in `generated/ernie45_d_p/golden/`, gitignored).

## P2.1-P2.11 (2026-09-24): device bring-up to full model 2k->2k. PASS
- Mesh (1,4), `FABRIC_1D_RING`. Sync CCLs (`ttnn.all_gather/reduce_scatter/all_reduce`, cluster_axis=1) need no semaphores.
- RoPE: `rotary_embedding_llama` + a custom 32x32 trans-mat (out[2k] = -x[2k+1], out[2k+1] = x[2k]) matches the
  interleaved form exactly. The chunk offset lives in the cos/sin tables for [start, start+S).
- KV cache (dev layout): per layer `[max_seq/64, 1, 64, 128]` per chip, identity page table. With 1 KV head/chip this
  is bit-identical to a contiguous cache. Writes use `paged_fill_cache` with a chunk page-table slice. Chunk 0 uses causal
  `scaled_dot_product_attention`; chunk>0 uses `chunked_scaled_dot_product_attention`.
  Gotcha: pass chunked-SDPA args as **keywords** (positional + `scale=` raised a nanobind TypeError).
- MoE: no single TTNN op, so it is COMPOSED. Router: fp32 linear/softmax/add-bias/topk/gather/renorm.
  `ttnn.scatter` has no fp32-TILE path, so the dense routing matrix is scattered in bf16 (agreement 99.9% of experts).
  Routed experts are "dense-EP": 16 local experts over all tokens, scaled by the routing weight.
  Gotcha: `mesh_partition`/`slice` need 32-aligned widths, so a 16-expert column split fails. Instead, per local slot j,
  `routing @ M_j` (M_j row 16c+j = 1) gives the weight already broadcast to [S, 2560].
  Perf TODO: switch to `sparse_matmul` (6/64 of the FLOPs).
- Gotcha: pre-commit autoflake strips imports unused *at commit time*, which bit `tt/moe.py`.
- Gotcha: `run_safe_pytest.sh` runs a fake-tensor collect pass first. `bringup.metrics.record` ignores it.
- P2.11 results: final hidden 0.9983, worst layer 0.9985, worst KV 0.9979, top-1 98.7%, top-5 100%, about 0.9 s per 2k chunk.
