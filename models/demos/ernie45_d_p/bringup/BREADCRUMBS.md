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

## P1.6, P2.12-P2.16 (2026-09-24): 55k chunked prefill E2E + prefill-server contract. PASS
- P1.6 55k golden: 61 min on CPU (3-7.5 min per 5k chunk). Per-layer dumps for the last chunk only; KV for all 56320.
- P2.13 (b): golden KV prefix [0, 51200) loaded (`TtKVCache.load_prefix`), device prefills 51200->56320. Final 0.9982, top-5 100%.
- P2.14 (a): all 11 x 5120 chunks on device. Final hidden 0.9984, worst layer 0.9987, worst KV (28 layers x 56320)
  0.9960, top-1 97.7%, top-5 100%. Chunks take 2.9 -> 4.0 s (attention grows with context), about 36 s for 56k tokens
  (dense-EP MoE, untraced, bf16).
- P2.15 contract: reuses `gpt_oss_d_p` `allocate_kv_cache` / `write_kv_chunk` / `build_kv_chunk_address_table` directly
  (sp=1, tp=4 = KV heads). Every 32-token chunk read back through `table.read_device_chunk` is byte-identical to the
  device tensor. bf8 contract KV vs golden: 0.9979.
- P2.16 engine: `ernie45_d_p` registered in `common/prefill/adapter.py`. The producer GQA reader skips the HF->Meta K
  permutation via `golden_k_rope_layout = "interleaved"`. Driven only through the adapter/runtime API (slot 1 of 2); layer acks are
  global and in order. Gotchas (both fixed in the gate): precompile-mode `comp_pcc` stub (0.999999) leaking into the producer's
  real-pass check, and ambient PYTHONPATH -> ../tt-metal. Hence `--no-precompile` + pinned PYTHONPATH + an independent PCC.
- Replay the whole ladder: `python models/demos/ernie45_d_p/bringup/gate.py --sweep` (P1.6 alone is about 1 h CPU; use
  `--sweep P2` to replay only the device side against the stored goldens).
- Next steps (not gated yet): sparse_matmul experts (about 2.7x fewer expert FLOPs; needs a tile-aligned 16-expert routing slice),
  trace capture, bf8 weights, attention from the contract bf8 cache (ring_joint SDPA) instead of the extra bf16 attention cache,
  two-process runner + producer over H2D sockets (`PREFILL_MOCK_MIGRATION=1` / `PREFILL_PRODUCER_CHECK_PCC=1`).

## P3.1 (2026-09-25): reuse of ttnn.experimental.deepseek_prefill.unified_routed_expert_moe. Probe PASS, integration blocked
- The op is the fused routed-expert FFN only (all local experts in one program, SiLU-SwiGLU = ERNIE's activation).
  It is fed by the DeepSeek EP pipeline: routing_setup -> dispatch -> unified op -> combine -> reduce (as in gpt_oss_d_p).
- Measured on real routing (layer 1/14, last 5120-token chunk of the 55k golden), bf8 activations, bf16 weights, HiFi2:
  **2.0-2.1 ms vs 26.9 ms** for the dense-EP experts (**about 13x**), worst per-expert PCC 0.9992. Dense-EP experts are about
  0.73 s of the about 2.7 s fixed per-5k-chunk cost, so the expected TTFT gain is about 20-25%, not the dominant term.
- Blockers on a 1x4 mesh (dispatch group = 1 chip on axis 0):
  1. `offset_cumsum` all-gathers over the dispatch axis; all_gather rejects a 1-device axis. Patched (skips the gather
     when the axis size is 1; separate commit touching ttnn C++).
  2. The `dispatch` program factory always wires fabric neighbours on the dispatch axis: TT_FATAL "No neighbors found".
     This is not patched; it needs a local-only dispatch/combine path.
  3. The dispatch op only supports cluster_axis=0, so dispatching along the 4-chip axis 1 is rejected.
- Gotchas: (a) the op writes its output IN PLACE into a TILE dispatched buffer; re-running on the same buffer silently
  computes on its own output. (b) expert_token_counts / expert_region_offsets are indexed by GLOBAL expert id ([1, 64] per
  chip); passing a local [1, 16] makes the kernel read garbage counts and HANG (cb_wait_front deadlock).
- The probe dispatched buffer is built on the host from golden routing (`tests/pcc/test_unified_expert_probe.py`).
- P3.2 (TtMoEUnified, full pipeline) stays TODO until a 1-chip dispatch/combine exists. Options: patch dispatch/combine
  for dispatch_group_size == 1 (local NOC copy, no fabric), or compose a local dispatch from TTNN ops.

## P3.2 (2026-09-25): unified_routed_expert_moe integrated as the default MoE. PASS; full ladder re-run PASS
- ttnn patches for a 1-device dispatch axis (1xN mesh, dispatch group = 1 chip), host side only:
  `offset_cumsum` skips its all_gather; `dispatch` and `combine` program factories gate all fabric setup on
  `use_fabric = num_links > 0 && dispatch-axis devices > 1`. The kernels already had a no-fabric build (`#ifdef DEST_CHIP_ID`)
  and every routed expert is local or absent (-1), so nothing needs the fabric.
- `tt/moe_unified.py` TtMoEUnified: router -> masked_bincount/offset_cumsum -> dispatch -> unified op (ROW_MAJOR bf16
  input = fused fast path) -> combine -> post_combine_reduce (weighted top-k, local) -> + shared partial -> one all_reduce.
  Dispatch/combine are built lazily per chunk length and shared by all layers. `ERNIE_MOE_IMPL=dense` restores dense-EP.
- Gotcha: with TILE input the op returns its input buffer as the output (in place). Do not free the input before combine.
- Results (unified vs dense-EP):
  | | final hidden | worst KV | top-1 | TTFT |
  |---|---|---|---|---|
  | 2k->2k | 0.9958 (0.9983) | 0.9939 (0.9979) | 94.3% (98.7%) | 0.76 s (1.79 s) |
  | 55k@5k | 0.9965 (0.9984) | 0.9926 (0.9960) | 95.7% (97.7%) | **10.9 s (35.9 s)** |
  Top-5 stays 100%. The accuracy cost comes from the kernel packing expert activations to bf8 internally (even for bf16 input).
- Correction: the isolated P3.1 probe timed dense-EP experts at 27 ms per layer, but in the model the dense-EP MoE cost about 2.4 s
  per 5k chunk. The measured TTFT win (3.3x) is far larger than the 20-25% predicted from the probe.
- Chunk time is now 0.44 s at 5k context and grows to 1.45 s at 50k: attention is now the dominant, context-dependent cost.
