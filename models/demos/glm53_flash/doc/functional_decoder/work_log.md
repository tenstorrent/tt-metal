# GLM-5.3-Flash functional decoder: work log

Host: `sjc3-glx-a04u07` (32 x Blackhole Galaxy, FW 19.15.0.0). Checkout: `~/ashai/glm53/tt-metal`
main @ `8cfcaf754d3` (2026-10-19), branch `arg/glm53_flash`. Design of record: `~/ashai/GLM_5.3_DESIGN.md`.
Stage: tt-model-bringup stage 1 (functional decoder, 1x1 mesh) = design Phase 1.

## 2026-10-19

- Fresh clone with submodules; `./build_metal.sh` (Release, clang-20, Tracy on) 10m12s on 64 cores.
- Checkpoint `zai-org/GLM-5.3-Flash` (sha 46a050bec13002ad29ee795c16e61cf4fa3c4a9c, 62 shards, 328.4 GB) into the
  HF cache (`hf download`, 16 workers, ~25 min).
- Verified checkpoint key names/dtypes/shapes for layers 0, 3, 4 directly from the safetensors headers:
  raw Kimi-K3 KDA naming (`q/k/v_conv1d`, `f_a/f_b_proj`, `A_log`, `dt_bias` under `self_attn`), mHC as
  `hc_attn_{fn,base,scale}` / `hc_ffn_*` (fn bf16 [24,16384], base/scale fp32), per-expert fp8 weights with
  128x128 `weight_scale_inv`; fp8 groups = experts, shared experts, dense MLP, DSA `q_a/q_b/kv_a/o_proj`.
  transformers remaps these in `conversion_mapping.py` ("glm5_next").
- Host reference (`reference/hf_layer_reference.py`, CPU venv `~/ashai/glm53/hostref_venv`: torch cpu +
  transformers 5.20.2): builds the real `Glm5NextTextDecoderLayer` per layer from the shards (fp8 dequant on
  host), runs embedding -> layers 0..4 prefill with a `DynamicCache`, then 4 single-token decode steps, and
  dumps per-layer `x_in/x_out` (mHC streams `[B,S,4,4096]`), top-k indices and cache state to
  `bringup/references/glm53_flash/<run>/` (git-excluded locally; not committed).
  - `T256`: layers 0-4, 256 tokens + 4 decode steps: done (layer 3 prefill 38.6 s, layer 4 27.7 s on CPU).
  - `T4097`: layers 0-4, 4097 tokens (sparse DSA path, > 2048) + 4 decode steps: running.
- Note: the config.json names DSA layers `deepseek_sparse_attention`; the modeling code keys its mask mapping
  on `indexed_attention`, so the layer-level harness normalizes `layer_types` (same as from_pretrained).
- `tt/config.py`: frozen config dataclass loaded from config.json (45 layers: 34 KDA, DSA at 3+4k, 42 MoE).
- Build done (10m12s); `create_venv.sh` (uv) + `transformers==5.20.2` + `tt-perf-report` added with
  `uv pip install --python python_env/bin/python`. Single-chip smoke (TT_VISIBLE_DEVICES=0, MeshShape(1,1)): OK.
- `reference/check_kda_reference.py`: the in-tree Kimi-K3 KDA torch reference == HF GLM KDA with real layer-0
  weights (prefill PCC 0.99999, state 0.99999, decode 0.99999) -> ttKDA prefill kernels + state contract reused.
- `tt/kda.py` + `tests/test_kda.py` (1x1 mesh, layer 0, real weights): prefill T=96 PCC 0.99994 / T=1000 0.99995,
  recurrent state 0.9999, conv history 0.99999; 3 decode steps from the device state PCC 0.9999 each. Commit f9ed9df261e.
- `tt/moe.py` (router = moe_grouped_topk single-group; experts = sparse_matmul per 32-token block with a per-expert
  union mask; shared expert = clamped SwiGLU). sparse_matmul needs a rectangular grid whose core count divides the
  output tile count (TT_FATAL "num_cores_with_work must equal in0_mcast_receiver_num_cores"): `pick_sparse_grid`.
  First run (layer 4, T=32, realistic MLP input from the T256 dump): output PCC 0.9964, but top-8 sets identical to
  HF on only 16/32 tokens (dense routing-weight PCC 0.985) -> quantifying whether that is near-tie noise.
- Root cause of the router mismatch: the HF layer builders cast the whole module to bf16 before restoring the
  strict-fp32 tensors (router bias ~5.5 resolves 0.03 in bf16) -> fixed (`cast_like_from_pretrained`), dumps
  regenerated. Separately, `moe_grouped_topk` ranks with bf16 resolution and swaps the 8th expert on ~6/32 real
  tokens (HF biased-score gaps 0.003-0.004) -> router is now an exact fp32 composite (iterative masked argmax).
  Result: top-8 identical 32/32 (T=32) and 255/256 (T=256); MoE output PCC 0.99993 / 0.99995.
- `tt/dsa.py`: dense causal SDPA prefill on materialized K/V (exact for context <= 2048), latent written with
  paged_fill_cache; decode = paged_update_cache (height-sharded [1,B,32,512] input) + absorbed
  paged_flash_multi_latent_attention_decode. L1 trap: the decode op's c_19 staging CB is sized
  (cores_per_user-1)*(out+2) fp32 tiles (1.6 MB at 12 cores/user with the 512-wide latent) -> 4 cores/user,
  k_chunk 32. Layer 3 vs HF: prefill PCC 0.99964 (T=96) / 0.99961 (T=1000), latent cache 0.99997, decode 0.9991-0.9999.
- Commit 25c8b1100f8: moe, dsa, decoder_layer, functional_decoder (stage contract), tests.
- Full-layer tests (`tests/test_decoder_layer.py`, HF dumps T256): layer 0 prefill 0.99984 / decode 0.9999;
  layer 3 0.99992 / 0.99996; layer 4 0.99997 / 0.99999; non-aligned S=200 vs HF built live: layer 0 0.99980,
  layer 3 0.99993. Commit 8e2723a6cea.
- Traced decode (`tests/test_decode_trace.py`): KDA decode writes its carries back in place (ttnn.copy into the
  persistent batched state; prefill writes user slots), the DSA cache is already in place; capture once on a
  dummy token, replay the 4 reference steps updating only the input/position tensors: replay PCC identical to
  eager for layers 0, 3, 4 (0.9999).
- Two-user decode (`test_two_users_decode_vs_hf`, prompts 100 and 37 tokens in slots 0/1): layer 0 per-user PCC
  0.9998-0.9999, layer 3 0.9998-0.99998. Commit d5efa78fb35.
- First perf numbers, layer 3 (DSA+MoE), one chip, functional v1 (`tests/test_perf_layer.py`, wall clock):
  warmed prefill S=1024 = 314 ms (3.3k tok/s for this layer); traced decode B=1 = 10.95 ms/step
  (a 45-layer single-chip extrapolation would be ~2 tok/s/user: the composite KDA/MoE paths are not tuned;
  the design's performance work is Phase 2/3 on the 8x4 mesh). `python -m tracy -r -p` failed to merge
  device data ("Op 1024 not present in cpp_device_perf_report.csv"); re-collecting with
  TT_METAL_DEVICE_PROFILER=1 + process_ops_logs.py.
- Sparse DSA path (`tt/indexer.py`, `tt/dsa.py::_prefill_sparse / decode_forward_sparse`, `tests/test_dsa_sparse.py`):
  pooled-key cache (paged, [B*blocks,1,32,128]) written with paged_fill_cache / paged_update_cache; raw k/gate
  4-slot rings per user complete the pool a decode token closes (pool (a+1)//4-1 rewritten every step, harmless
  before completion because it is not yet selectable); scoring = indexer_score_dsa(key_compression_ratio=4) with
  the 128^-0.5 and 32^-0.5 scales folded into wq_b / weights_proj; selection = repeat_interleave(scores, 4) +
  topk_large_indices(k=2048) (token ids directly, whole pools since ties are contiguous) + host-geometry tail ids
  + 0xFFFFFFFF sentinel pad to 2176; attention = sparse_sdpa over the user's latent rows untilized from the paged
  cache. Op facts: indexer_score_dsa needs Sq%32==0, chunk_start%32==0 and chunk_start < kv_len*4 (decode query
  placed at row a%32 of an aligned block, kv_len extended by one tile); ttnn.pad cannot front-pad tiles (concat);
  update_padded_kv_cache only writes whole-chunk multiples (not usable as a row writer); int32->uint32 typecast
  preserves bits (-1 == sentinel). Layer 3 vs the T4097 dump: prefill PCC 0.99977 (rows >= 2048: 0.99990),
  sparse decode at 4097..4100: 0.99995 / 0.99996 / 0.99993 / 0.99991.
- Pushed `origin/arg/glm-53-flash` @ bf8f2ab1d88 (squashed from the local dev branch `arg/glm53_flash`, docs
  excluded: `models/demos/glm53_flash/doc/` stays local; pre-commit green on all 24 files).
- `tests/test_determinism.py` (layers 0, 3): two identical prefills bit-identical, two traced decode replays from
  the same state bit-identical, `throw_exception_on_fallback=True` raised nothing (no host/torch call inside a pass).
- Watcher run (`TT_METAL_WATCHER=10`, `TT_METAL_LOGS_PATH=~/ashai/glm53/_logs/watcher_layer3`) of the layer-3
  dump test: PASSED, 10 dumps, watcher.log 3315 lines, no exceptions/asserts/NoC/L1/CB messages (only the normal
  "highest stack usage" summaries).
- Tracy (`python -m tracy -r -p`) fails at device close for the full prefill+decode run of layer 3:
  `profiler.cpp:334 "Source location hashes are colliding"` (16-bit zone-location hash collision in the kernel
  set), so no device CSV; `TT_METAL_DEVICE_PROFILER=1` + process_ops_logs hits the same gap. Trying per-phase runs.
- Profiler collision identified offline (same FNV-1a 16-bit hash as profiler.cpp): zones "PACK SUB_EXP"
  (compute_streaming.hpp:632) and "Softmax(Q@KT)@V" (compute_streaming.hpp:1769) of the dense SDPA prefill
  kernel collide at 0x6835 on main 8cfcaf754d3, so every run that compiles `scaled_dot_product_attention`
  throws at device close (tt-metal profiler bug, not model code; worth an upstream issue). Per-phase runs do not
  help for DSA layers (the warm-up prefill compiles the kernel). Evidence plan: tt-perf-report tables from
  layer 4 (KDA+MoE, no SDPA) for both phases; layer 3 keeps wall-clock numbers.
- tt-perf-report (layer 4 KDA+MoE, one chip, `Device Time` column in us from the signposted windows; files in
  `doc/functional_decoder/tracy/layer4_kda_moe/`):
  - prefill: 1002 ops, 293.4 ms device per pass (wall 297.8 ms, S=1024); top op classes: SparseMatmulDeviceOperation 183.29 ms x64, UnaryDeviceOperation 45.46 ms x167, BinaryNgDeviceOperation 17.32 ms x152, SliceDeviceOperation 14.10 ms x194, PermuteDeviceOperation 9.96 ms x16, FastReduceNCDeviceOperation 6.91 ms x32
  - decode: 1250 ops, 9.6 ms device per step (3 traced steps summed 28.7 ms; wall 11.08 ms/step); top op classes: AccumulationDeviceOperation 1.76 ms x5, UnaryDeviceOperation 1.47 ms x17, BinaryNgDeviceOperation 1.26 ms x110, SparseMatmulDeviceOperation 1.11 ms x2, MatmulDeviceOperation 0.99 ms x11, TernaryDeviceOperation 0.61 ms x38

## 2026-10-19 (evening): stage review 1 -> fixes

Independent stage review (fresh subagent, read-only) returned `more-work-needed`: P1 sparse-decode pool bookkeeping
(pool (a+1)//4-1 rewritten every step = the previous COMPLETE pool corrupted from a ring that already holds the new
token; a host simulation showed 230 corrupted pool reads over 40 steps), P1 the dense paths never maintained the
pooled-key index (a short prompt decoded past 2048 would score an empty index), P2s: decode perf table from an
incomplete profiler buffer (1250 rows for 3x471 ops), no layer-3 decode table, contract claimed 16384 untested,
dense decode silently valid past 2048 under trace, identity-only page tables, batch coverage 2.

Fixes (`tt/indexer.py`, `tt/dsa.py`, `tt/functional_decoder.py`, `tt/decoder_layer.py`):
- pool written each decode step = `position // 4` (not selectable until complete; exact at `a % 4 == 3`); raw
  k/gate rings are 32-row paged blocks per user written with `paged_update_cache(update_idxs=position % 4)`;
  the index is maintained on EVERY prefill and decode (dense and sparse) from batched, trace-safe ops driven by
  three persistent [B] int32 inputs (`DecodePositions`: position, position // 4, position % 4 + host values);
  `FunctionalDecoder.make_decode_positions / update_decode_positions`; sparse path auto-selected from the host
  values; `capture_decode_trace` refuses DSA decoders with max_seq_len > 2048 unless `allow_sparse_range`.
- page tables: any unique [max_batch, blocks] mapping; `user_rows_rm` gathers the user's blocks in table order.
- router bias is a broadcast [1,1,1,E] row; TtKDA no longer keeps the raw torch weights; indexer tail/sentinel
  constants cached per geometry (uploaded once); hf_layer_reference: `--random-ids` prompts and the HF indexer's
  selected ids (`index_ids`) recorded via a forward hook.
New references (random token ids, CPU): R4097 (layers 0-4, 12 decode steps), R2040 (layers 0-3, 24 steps),
R16385 (layers 0-3, 2 steps; layer 3 prefill 48 s on the host).
New tests (`tests/test_dsa_long.py`, `tests/test_batch8.py`, log `_logs/tests_long_batch8.log`):
- selection vs HF (R4097, 12 steps 4097..4108): layer PCC 0.9997-0.99997; TT/HF selected-id sets 98.0-99.2 %
  common per step (bf16 indexer scores vs fp32: boundary pools); pooled-key cache after 12 steps vs host
  recomputation 0.999994 (worst of the last 8 rows 0.99999).
- dense->sparse transition (R2040, slot 1 of 2, permuted page table): prefill 0.99988; decode 2040..2063 all
  >= 0.99962 (dense 2040-2047, sparse 2048-2063, tails 0/1/2/3 covered).
- 16K (R16385): sparse prefill S=16385 PCC 0.99906 (last 2048 rows 0.99982), decode 16385/16386 0.99996/0.99995.
- 8 users (layers 3 and 4, lengths 100/37/64/1/200/33/96/129, permuted page table): prefill >= 0.9968 (the
  1-token user), 3 decode steps worst per-user 0.99938 (layer 3) / 0.99936 (layer 4).
Regression after the refactor (`_logs/regress_dsa_refactor.log`): dense DSA T=96, sparse 4097, traced layer 3,
two users: unchanged PCCs.
- Perf re-collected with GLM53_PERF_DECODE_STEPS=1 (one signposted replay, so rows == ops) and the device profiler
  drained around the window; layer 3 decode profiled with GLM53_PERF_FORCE_SPARSE_PREFILL=1 (cache seeded through the
  sparse prefill, so the colliding dense-SDPA kernel is never compiled). `Device Time` column (us):
  - layer4_kda_moe decode (1 replay): 471 ops, 10.76 ms device per step, duplicate ids 0; top: AccumulationDeviceOperation 2.64 ms x8, UnaryDeviceOperation 1.48 ms x17, BinaryNgDeviceOperation 1.28 ms x110, SparseMatmulDeviceOperation 1.11 ms x2, MatmulDeviceOperation 1.03 ms x12, TernaryDeviceOperation 0.62 ms x38
  - layer3_dsa_moe decode: the first forced-sparse seeding at S=1024 failed (topk_large_indices needs width >= k;
    the selection now caps k to the visible width) -> re-profiled below.
- Full logged sweep (`_logs/sweep_<test>.log`, one chip): test_kda 2, test_moe 2, test_dsa 2, test_decoder_layer 5,
  test_decode_trace 5, test_dsa_sparse 1, test_dsa_long 3, test_batch8 2, test_determinism 2 -> 24/24 passed.
- Watcher (`TT_METAL_WATCHER=10`): layer-3 dump test (earlier) clean; traced decode (`test_decode_trace.py -k traced`,
  `_logs/watcher_test_decode_trace`) clean, 4043 lines, 0 suspicious; KDA and sparse tests re-run below.
- Watcher re-runs (`TT_METAL_WATCHER=10`, logs `_logs/watcher_test_kda`, `_logs/watcher_test_dsa_sparse`):
  test_kda.py (ttKDA prefill kernels incl. the fabric history exchange, decode composite, T=96/1000) PASSED, 1012 lines,
  0 suspicious; test_dsa_sparse.py (indexer, topk_large_indices, sparse_sdpa, sparse decode) PASSED, 2981 lines,
  0 suspicious. Together with the layer-3 dump test and the traced decode run, every path kind has a clean watcher run.
- layer3_dsa_moe traced decode (B=1, pos 1024, one replay, GLM53_PERF_SEED_ONLY=1 so no SDPA prefill kernel is
  compiled; wall 11.36 ms/step): 435 ops, 10.93 ms device per step, duplicate ids 0; top op classes: AccumulationDeviceOperation 2.64 ms x8, MatmulDeviceOperation 1.66 ms x13, UnaryDeviceOperation 1.44 ms x12, SparseMatmulDeviceOperation 1.11 ms x2, BinaryNgDeviceOperation 0.96 ms x92, TernaryDeviceOperation 0.63 ms x38, FillPadDeviceOperation 0.55 ms x30
  (files `doc/functional_decoder/tracy/layer3_dsa_moe/`). The earlier forced-sparse seeding also collided: sparse_sdpa
  shares compute_streaming.hpp with the dense prefill kernel.
- Checkpoint after the review fixes: local `arg/glm53_flash` @ 8e965d64e72 (code + docs), upstream
  `origin/arg/glm-53-flash` @ ab2653dbbf5 (code + tests only, pre-commit green). Second stage review requested.

## 2026-10-19 (late): stage review 2 -> fixes

Second independent review: `more-work-needed`. P1: `sparse_sdpa` reads an index row only up to its FIRST sentinel
(`sparse_sdpa_reader.cpp`: "sentinels are a contiguous tail"), while the rows were `[top-k | tail ids | pad]` with the
top-k block ending in sentinels whenever fewer than 2048 tokens were visible -> every query at position < 2047 that
went through the sparse path lost its tail ids (its own token included) and rows 0-2 gathered page 0xFFFFFFFF. That is
why the sparse prefill rows < 2048 scored 0.99966 vs 0.9999 on the dense path and the random-prompt 4097 prefill
0.99657 (a number I had mis-transcribed as 0.9998 in the contract). P2: prefill tails of 2-3 rows were written with one
paged_update_cache call whose per-row cores read-modify-write the same 32-row tile (race; lengths == 2,3 mod 4 were
never tested); README/contract numbers contradicted the logs; `ttnn.from_torch` uploads inside measured passes (KDA
scalars, prefill tail slot/page tensors); plus other concerns (single-shot prefill only, layer-3 S=1 PCC 0.9968, raw
test hook filtering, pool-splitting ties, batch 32 not attempted).

Fixes (`tt/dsa.py`, `tt/indexer.py`, `tt/kda.py`, `tt/decoder_layer.py`, `tt/functional_decoder.py`):
- Sparse rows must be full prefixes: prefill rows [0, 2048) run the dense causal SDPA, rows >= 2048 (they always see
  >= 2048 pooled tokens) run select(query_start=2048) + sparse_sdpa; decode runs the dense flash-MLA for every slot
  (HF-exact through position 2050 = index_topk + kpool - 2) and only users at position >= 2051 get their row replaced
  by the per-user sparse result (mixed batches are therefore exact for the short users). `select` asserts >= 2048
  scored tokens instead of capping k.
- Ring tail rows written one per `paged_update_cache` call with slot/page tensors uploaded once at cache creation;
  KDA `actual_start/actual_end` scalars cached per value; the sparse-decode geometry constants were already cached:
  no `ttnn.from_torch` inside a prefill/decode pass after the first use of a geometry. Audit method: grep of
  `from_torch|to_torch|torch\.` in the forward paths + `throw_exception_on_fallback=True` runs.
- `selected_ids` test hook returns the raw row; the test asserts the contiguous-prefix contract and the id range.
- Trace guard aligned with the dense-exact range (max_seq_len > 2051 refused unless allow_sparse_range).
New references: R2042 / R2043 (random ids, layers 0-3, 12 decode steps: prompt tails 2 and 3, decode through the pool
completion and across 2048/2050). New/extended tests: per-row PCC split (rows < 2048 / >= 2048 / position % 4) on the
4097 and 16385 prefills with every class asserted > 0.995; random-prompt prefill PCC asserted; the transition test
checks the short user 0 at every step against its own HF cache; `test_prefill_tail_2_3_then_decode`; dense DSA
prefill at S = 1, 2, 3, 31, 33 (short-S control); sparse-path determinism.
- Results after the review-2 fixes (`_logs/fix2_*.log`, one chip): dsa (S=1/2/3/31/33/96/1000) 7 passed, attention-only
  S=1 0.99989 (no short-S effect in the DSA sublayer); sparse 4097 repeated prompt 0.99992 (rows<2048 0.99994, >=2048
  0.9999); random prompt 4097 0.99994 (rows<2048 0.99988, >=2048 0.99986; per position%4 all >= 0.9998), selection ids
  98-99 % common with contiguous rows, pool cache 0.999994; transition 2040..2063 both users >= 0.9995; 16K 0.99983
  (0.99988 / 0.99980); traced + two users (layer 3) unchanged; 8 users (layer 3) worst 0.99938; tails 2/3 (R2042/R2043):
  prefill 0.99987/0.99990, 12 decode steps >= 0.9978, pool cache 0.999994 (worst row 0.99999).
- Checkpoint: local `arg/glm53_flash` @ 4cd5f8fe829 (code + docs), upstream `origin/arg/glm-53-flash` @ 3ae53bf70ee.
  Running: full logged sweep (`_logs/sweep2_*.log`), watcher re-runs (`watcher2_*`), layer-3 decode perf re-collection.
- Open design item carried into the next round (reviewer "other concern", design Phase-1 row: "chunked prefill"):
  wire `prefill_forward(streams, user, seq_len, start)` through the API. KDA: `ttKDA.forward(actual_start=start)`
  with the user's carried `KdaState` (already supported; needs the per-user state read from the batched slots).
  DSA: `paged_fill_cache` at a block offset (fill from `start` with a page-table view), index maintenance from
  `start` (pools `start//4 ..`, ring from the previous tail), dense rows via the cached prefix (K/V from
  `user_rows_rm` + chunk) while `start + T <= 2048`, sparse rows via `select(query_start=start)`. The quadratic
  `[T_chunk, 4*pools]` score intermediate then scales with the chunk, not the context.
- Final sweep after the review-2 fixes (`_logs/sweep2_<test>.log`): test_kda 2, test_moe 2, test_dsa 7, test_decoder_layer 5,
  test_decode_trace 5, test_dsa_sparse 1, test_dsa_long 5, test_batch8 2, test_determinism 3 (incl. the sparse path) -> 32/32 passed.
  Per-row PCC distributions (one token row = 4 x 4096 values): random 4097 prefill median 0.9999, p5 0.99981, min 0.98783;
  dense 2040 prefill median 0.99996, p5 0.99985, min 0.99079, 5/2040 rows < 0.997; 16385 prefill min 0.9678 (one row). A single-token
  prefill is one sample of this distribution (the 8-user test's 1-token user at 0.9968); the DSA attention alone at S=1 is 0.99989.
- Watcher re-runs on the fixed paths (`_logs/watcher2_test_dsa_sparse`, `_logs/watcher2_test_decode_trace`): clean, 2328 / 3002 lines, 0 suspicious.
- layer3_dsa_moe traced decode re-profiled on the new decode path (dense flash-MLA for every slot + index maintenance, B=1, pos 1024,
  one replay, seed-only; wall 11.31 ms/step): 435 ops, 10.87 ms device per step, duplicate ids 0; top op classes: AccumulationDeviceOperation 2.64 ms x8, MatmulDeviceOperation 1.66 ms x13, UnaryDeviceOperation 1.42 ms x12, SparseMatmulDeviceOperation 1.11 ms x2, BinaryNgDeviceOperation 0.95 ms x92, TernaryDeviceOperation 0.62 ms x38.
  README decode row: layer 3 10.87 ms/step device (435 ops)

## 2026-10-20: expert dtype gate (branch `arg/glm53_dtype`, chip 5)

- Knobs `expert_dtype` (W13 gate/up) / `expert_down_dtype` (W2) on TtRoutedExperts -> TtMoE -> TtDecoderLayer ->
  FunctionalDecoder.from_state_dict, defaults bfp8_b (unchanged); shared expert and dense MLP stay on `weight_dtype`.
- `tests/test_expert_dtype.py` (6 cases, layers 3 and 4 x {bfp8, bfp4 gate/up, bfp4 all}, ~3 min per layer): MoE-only
  T=256 PCC 0.99995 / 0.99968 / 0.99808 (L3) and 0.99995 / 0.99504 / 0.99141 (L4), router sets identical (255/256
  everywhere); R4097 full-layer prefill 0.99994 / 0.99945 / 0.99767 (L3), 1.00000 / 0.99994 / 0.99990 (L4); decode worst
  step 0.99996 / 0.99947 / 0.99775 (L3), 0.99996 / 0.99747 / 0.99628 (L4). Logs `_logs/dtype_sweep_layer{3,4}.log`.
- Layer-4 traced decode under Tracy (one replay, same chip/session, `tracy/dtype_gate/`): 10.93 -> 10.80 (bfp4 gate/up)
  -> 10.72 ms (bfp4 all) device; only the sparse matmuls change (W13 765 -> 624 us, W2 529 -> 465 us); the W13 matmul
  runs at 35 % of DRAM peak in bfp8 (22 % in bfp4), so halving the bytes saves 18 %, not 50 %; baseline repeat 10.92 ms.
- Evidence and recommendation: `doc/functional_decoder/dtype_gate.md` (keep bfp8 default; only bfp4 gate/up goes to the
  eval-set gate; perf side to be re-measured on the Phase-2 layout).
