# Gemma4 implementation survey: E2B Galaxy decode vs other Tenstorrent implementations

2026-10-09. Branch `rsalman-e2b-device-pli`, tree `/data/rsalman/tt-metal-pli`, HEAD `d2434215e2b`.
Brief: `DEVICE_PLI_HANDOFF.md:74-84`.

**Citation rules**
- `models/demos/gemma4/tt/model.py` line numbers refer to HEAD. The working tree has uncommitted Phase 1 edits that shift lines.
- Other citations are `file:line` in HEAD, a commit, or a gh PR.
- **UNVERIFIED** marks any claim with no artifact behind it. Any number derived from an UNVERIFIED input is also UNVERIFIED.

---

## 1. Scope and method

**Metric.** Per-step decode time at batch 1. Device time is reported where it exists. Tokens per second is shown only as context.

**Baseline**
- Model: Gemma-4-E2B on a 4x8 Blackhole Galaxy (`models/demos/gemma4`).
- Layout: TP=8 on mesh axis 1; generator `data_parallel=1`; the 4 mesh rows replicate the same work (`decode_timing_results.json` `layout`).
- Weights: all bf16. `precision_overrides.json` has entries only for 31B and 12B.
- PLI: computed on host every step.
- Fabric: `FABRIC_1D_RING` (job `device_params`).

**Sources**
- Code reading of `models/demos/*`, `models/tt_transformers`, `models/common/sampling`, `models/experimental` and ttnn CCL sources.
- `README.md`/`PERF.md` files and `models/model_targets.yaml`.
- Read-only `git log`/`git show` on remote branches.
- `gh pr view` and `gh pr diff`.
- Job outputs under `/data/rsalman/jobs/{run,logs}/`.

No hardware was run for this survey.

**Measured today.** Current branch code: host PLI with the Phase 0 quick wins, `trace-device` arm, 188 greedy steps, 3 repeats each. Source: `/data/rsalman/jobs/run/<job>/enhancer_exp/e2b_on_dit_handle/decode_timing_results.json`.

| Job | Node (`/data/rsalman/jobs/logs/<job>.log:1`) | Steady median per repeat | Min | p90 | Summary median |
|---|---|---|---|---|---|
| 128881 | bh-glx-120-b02u08 | 17.40 / 17.36 / 17.49 ms | 17.1 ms | 19.0-20.0 ms | 17.42 ms (55.5 tok/s) |
| 128880 | bh-glx-110-d07u02 | 21.9 (cold repeat) / 20.1 / 20.8 ms | 17.1-17.6 ms | 23.7-26.0 ms | 20.45 ms |

What these runs show:
- All six repeats produce the same 189-token stream, and all match the reference.
- The minimum is the same on both nodes (17.1 ms). d07u02 has a fatter tail, so it is the noisier node. Pin b02u08 for A/B runs.
- The handoff's 26.2 ms vs 43.9 ms gap (`DEVICE_PLI_HANDOFF.md:92-93`) compared a plan-era number with pre-Phase-0 code (`DEVICE_PLI_HANDOFF.md:28`). It was a code difference, not node hardware.

**No device-time measurement exists.**
- The handoff's "device step ~11 ms" (`DEVICE_PLI_HANDOFF.md:77`) is an unsourced assertion: **UNVERIFIED**.
- Both jobs have `config.profile=false`, and no Tracy or ops-perf artifact exists under `/data/rsalman/jobs/run`.
- Every device-vs-host split below is therefore an estimate.

## 2. Implementations found

| Implementation | HW / mesh | Parallelism | Trace scope and token path | Sampling | CCL | Dtypes | Per-step decode (source) |
|---|---|---|---|---|---|---|---|
| **`models/demos/gemma4` E2B (ours)** | BH Galaxy 4x8 | TP=8 on axis 1. Rows 1-3 duplicate row 0 | Forward is traced. Sampling is eager (`model.py:241-249`). Host restages 5 inputs per step: tokens, pos, pos_int32, page_table, PLI. No device token feedback or `plus_one` while `_tt_vllm_always_refresh_decode_trace_inputs` is true, and it is always true for PLI models (`model.py:301`, `:2361-2366`) | TTSampling force-argmax, eager. `sampling_dp`=4; all-gather on axis 1 (`model.py:601,712-713`) | Sync `ttnn.all_reduce`, 2 per layer (`attention/operations.py:617`, `shared_mlp.py:232`, `ccl.py:302-308`). Async RS+AG only with `GEMMA4_CCL_ASYNC=1` (`ccl.py:99-105`) | bf16 | **17.4 ms** wall median on b02u08 (job 128881). Device time **UNVERIFIED** |
| `gemma4` E2B, BH P150 1x1 | 1 chip | none | same code | host sample by default (`README.md:57`) | n/a | bf16 | 22.82 tok/s ≈ 43.8 ms at ISL 4k (`README.md:27,35`, not CI) |
| `gemma4` E2B, WH N150 | 1 chip | none | same | same | n/a | bf16 | 12.24 tok/s ≈ 81.7 ms (`README.md:23`, CI) |
| `gemma4` 12B / 31B, BH LoudBox 1x8 | 8 chips | TP=8 | same | same | same | bf16 + overrides | 12B 47.59 tok/s ≈ 21 ms; 31B 31.69 tok/s ≈ 31.6 ms, ISL 4k (`README.md:30-31`). E2B is 🟢 on QB2 and LoudBox (`README.md:47`) but has no published number |
| PR #58524 (morjun, open), E2B | BH P150 1x1 / 1x2 | 1 chip; TP=2 for spec decode | Device PLI inside the trace (`init_pli_device_weights`, `compute_pli_device`). One fused spec-decode trace chains drafting, PLI, verify and argmax | target argmax on device; acceptance decision on host | — | bf16 PLI; HiFi4, fp32 acc | Plain decode 37.10 → **34.41 ms/token** with `DecodeMatmulTuner`. Spec decode K=3: 68.85 / 69.16 tok/s/u ≈ 14.5 ms per emitted token; 30.94 ms per iteration (PR body) |
| PR #58394 (merged into origin/main 2026-10-06 as `44d66500520`; **not in HEAD**), 31B / 12B | BH QB2 and BH Galaxy | Galaxy release entries use DP=4 lanes. Opt-in single instance over 32 chips: `GEMMA4_GALAXY_FRACTURE` / `GEMMA4_GALAXY_LANES` / `GEMMA4_CP_PREFILL` | — | on-device sampling on the fractured mesh | — | Knobs: `GEMMA4_KV_BFP8`, `GEMMA4_ROPE_PERUSER_B1`, `GEMMA4_SDPA_MAX_CORES_PER_HEAD` | Single instance: 641.7 tok/s aggregate at b32, +33% vs DP=4. Batch-1 per-step **not published**. The knobs and lane tests are present in origin/main (`models/demos/gemma4/{config.py,tt/model.py,tt/attention/decode.py}`, `tests/unit/test_lanes_*.py`) |
| `gemma4_31b_qb2` (31B dense) | BH QB2, 4 chips | TP4 (`README.md:3`) | Two traces per token. The model trace does `plus_one` on positions and RoPE (`tt/generator.py:450-451`). The sampling trace writes into the resident `[1,1,1,32]` token buffer (`tt/generator.py:474,488-491`) | TTSampling, Ring, 2 links, top_k ≤ 32, `allow_force_argmax: False` (`tt/generator.py:70-79`) | `all_reduce_async` with persistent per-slot buffers and semaphores. Both layer types send BF8 payloads; sliding layers keep a BF16 sum and full-attention layers a BF8 sum (`tt/decoder.py:457-469`). `all_gather_async` (`tt/decoder.py:411`) | BF4 LoFi projections, BF8 + HiFi2 LM head, BF8 KV (`README.md:9`) | 46.50 tok/s/u ≈ 21.5 ms at 1/1, 128/128 (`README.md:67`, vLLM). Rejects PLI (`tt/decoder.py:64`) |
| `gemma4_d_p` (31B) | BH Galaxy 8x4 / 4x8 | CP8/TP4 or CP4/TP8 (`README.md:3,61`) | prefill only | none | Same env gate as ours: sync by default, async RS+AG with `GEMMA4_CCL_ASYNC=1` (`tt/ccl.py:47-53,217`). 2 links on BH (`tt/ccl.py:12-14`) | — | n/a (no decode). Rejects PLI (`tt/model_config.py:54-56`) |
| `llama3_70b_galaxy` | WH Galaxy. On BH: no prefetcher, 2 links | 8x4 TP (`tt/model_config.py:608-613,714-724`) | Embedding, forward and position increment are traced. The sampler writes into the trace token input (`tt/generator.py:1707-1717,1974-1981`) | on device | Fused CCL+matmul, double-buffered persistent buffers (`tt/llama_ccl.py:472-611,883-1700`). BH Galaxy decode needs `FABRIC_2D_TORUS_XY` for column-axis collectives (`README.md:177`) | mixed | **14.01 ms/token** at b1/b32, 128/128, WH Galaxy (`PERF.md:33`) |
| Qwen3-32B via the `llama3_70b_galaxy` generator | **BH Galaxy** | DP=4 serving | same generator | on device | `FABRIC_2D_TORUS_XY`, `worker_l1_size` 1345000 (`llama3_70b_galaxy/README.md:177-179`) | — | 34.0 t/s/u ≈ 29 ms at b32, seq 507 (`model_targets.yaml:1460-1471`). The b1 entry has `perf: {}` |
| `llama31_8b_qb2` | BH QB2 | TP4 | Live traces; `plus_one` on device; the sampler writes `tt_out_tok=self.tokens` (`tt/generator.py:262-271`) | on device | — | BFP8 QKV/out/down/head, BFP4 gate/up, BFP8 KV. RMSNorm affine weights are folded into the next linear in fp32 (`README.md:9-12`) | 131.08 tok/s/u ≈ 7.6 ms (`README.md:102`); gate ≥ 110 (`README.md:91`) |
| `gpt_oss` | Galaxy [4,8], P300x2, P150 | `users_row_sharded` gives `sampling_dp=4` (`tt/model.py:242-243,286-292`) | `plus_one` on positions and RoPE indices (`tt/model.py:296-299`) | TTSampling, all-gather on axis 1 | — | — | 120b on P300x2: 24.46 t/s/u ≈ 40.9 ms (`model_targets.yaml:~491`). 20b on P150: 14.73 t/s/u. `bh_galaxy` entries are TODO |
| Gemma3 (`tt_transformers` via `multimodal/gemma3`) | WH N150 / T3K; 27B on WH Galaxy as DP=4 of 1x8 (`tests/pipeline_reorg/vllm_model_tests.yaml:53-62`) | TP per submesh | Sampling trace with `tt_out_tok=device_inputs[0]` (`tt_transformers/tt/generator.py:2316-2329`) | on device | FABRIC_1D_RING | bfp4 MLP, bfp8 attention | 4B N150 ≈ 29 ms; 27B T3K ≈ 59 ms (`multimodal/gemma3/PERF.md:14-16`) |
| PR #59035 (gtobarTT, open), Gemma-3-12B | BH P150 | 1 chip | — | — | — | bfp8 outputs | **Tracy device-kernel time 39.05 → 27.45 ms/token** (PR body). See O5 |
| `experimental/diffusion_gemma` | **UNVERIFIED** (draft said QB2) | — | Uses the precomputed `_decode_pli_combined` (`tt/commit_decode.py:626-628`), sliced per layer (`:559`) | — | — | — | n/a |
| `tt_dit` Gemma-3 encoder; pi0 PaliGemma | — | — | forward only (`tt_dit/encoders/gemma/model_gemma.py:5-8`); pi0 single device **UNVERIFIED** | — | — | — | n/a |

### Other Gemma4 decode work

**On `origin/ign/gemma4_opt_decode`.** None of these commits is in HEAD. All are gated to WH T3K dense models without PLI, so E2B is excluded.

| Commit | Change | Measured |
|---|---|---|
| `3bd380c0837` | Width-sharded L1 residual "island". RMSNorm gains `keep_sharded`, so the residual stops round-tripping through DRAM | not stated |
| `03d218f9bf1`, `f6df2ea8916` | Swept decode matmul program configs, gated by `is_t3k_dense_target` | Notes that the source branch regressed E2B PCC 0.9884 → 0.9802 at 1x2/1x8 when its configs leaked onto other meshes |
| `9abf56471fc` | Ring topology, plus an explicit RS+AG all-reduce with swept (workers, chunks, buffers) = (4, 2, 8). The gather lands directly in the L1 residual shard | 66.6 vs 67.9 µs |
| `c718c1c2194` | Folds GeGLU `gelu` into the multiply's input activation, and `layer_scalar` into the residual add: −2 ops per layer | The scalar fold is not bit-exact. The commit says E2B is fragile to where fp32-dest rounding happens |
| `c15299d85cf` | Drops two identity slices per layer (`apply_rope`, `concat_heads`) | Output-identical; within noise |
| `d1ec6e77698` | `GEMMA4_DECODE_PIPELINE=1` pipelined token reads | See O12 |

**Gemma4 PRs**

| PR | Author | State | Content |
|---|---|---|---|
| #57993 | morjun | closed | Decode matmul configs; folded into #58524 |
| #48659 | lukezTT | closed | Decode layer L1 optimizations |
| #47271 | lukezTT | closed | Shared MLP gate/up plus GeGLU |
| #49854 | smistTT | closed | DRAM-sharded plus spec decode on hybrid cache |
| #56466 | mtairum | merged | Per-head norm HiFi4/fp32 on prefill only |
| #56048 | arginugaTT | merged, in HEAD as `38c46ce0d17` | TT-native MTP + dFlash spec decode. Files in HEAD: `tt/{dflash_drafter,spec_decode,async_decode}.py` |

**Gemma4 branches.** 151 `origin/*gemma4*` branches exist. Decode-relevant ones include:
- `amalkov/gemma4-perf-test`
- `arg/gemma4_decode_norm_fp32`
- `amilovanovic/gemma4_SDPA_perf_chunks`
- `dwadhwani/gemma4-vllm-async-decode`
- `odjuricic/gemma4-production-optimizations`
- `ttuser/gemma4-perf-on-hybrid-cache`
- `adam/tt_symbiote_gemma-4`

None of these was audited.

### How the numbers compare

- **No published E2B Galaxy number.** The E2B targets are `status: TODO, perf: {}` (`model_targets.yaml:436-455`). The gemma4 README marks Galaxy 🔴 "not wired" (`README.md:47,58`). That note is stale upstream: PR #58394 ships BH Galaxy entries for 31B and 12B.
- **Plain decode vs other E2B numbers.** 17.4 ms is faster than every other plain-decode E2B number, but the comparison is not like for like:
  - The P150 number is at ISL 4k; job 128881 used a short prompt with `max_seq_len` 2048.
  - PR #58524's E2B spec decode on one P150 reaches about 14.5 ms per emitted token.
  - TP=8 over 8 chips is only about 2x faster than one P150 chip (17.4 vs 34.41 ms).
- **Larger models are about as fast.** A 70B Llama runs at 14.01 ms (WH Galaxy), 31B Gemma4 at about 21.5 ms (4 BH chips), and 12B Gemma4 at about 21 ms (BH 1x8). E2B at 17.4 ms is therefore far from weight-bandwidth-bound.
- **Device PLI.** Only PR #58524 and this branch's Phase 1 compute PLI on device. `gemma4_31b_qb2` and `gemma4_d_p` reject PLI models.

## 3. Baseline device-time breakdown: what is measured vs hypothesized

**Measured**

| Item | Value | Source |
|---|---|---|
| Wall time per step | 17.4 ms median, 17.1 ms min (b02u08) | job 128881 |
| Host PLI cost before Phase 0 | 9-22 ms/token | `DEVICE_PLI_HANDOFF.md:27` |
| Phase 0 effect | 43.9 → 17.1 ms/step, bit-exact | `DEVICE_PLI_HANDOFF.md:28` |

**Structural facts from code reading (verified; no times attached)**

*Op counts per layer*
- **Matmuls: 6 per layer, about 210 per step plus the LM head.** 4 are `DramShardedLinear`: wqkv, o_proj, fused gate_up and down (`attention/weights.py:173`, `shared_mlp.py:150,177`). 2 are plain PLI `ttnn.linear` calls (`layer.py:322,325`).
- **Each decode `DramShardedLinear` is 5 ops:** pad → L1 reshard → matmul → to DRAM → slice (`dram_sharded.py:361-390`).
- **RMSNorm:**
  - 5 layer norms per layer (`layer.py:239,269,280,310,326`). Each is 3 ops: reshard → rms_norm → sharded_to_interleaved (`rms_norm.py:102-114`).
  - Per-head Q norms run on every layer, and K/V norms on non-shared layers. Each is preceded by a `to_memory_config` to DRAM (`attention/decode.py:105-118`).

*CCL*
- **2 sync all-reduces per layer, 70 per step**, plus an embedding all-gather (`model.py:1298-1315`).
- **Each sync all-reduce lowers to 2 ops.** `ttnn::all_reduce` forwards to `experimental::all_reduce_async` with no semaphores (`ttnn/.../ccl/all_reduce/all_reduce.cpp:41-56`). That lowers to non-minimal `ttnn::reduce_scatter` + `prim::all_gather_async`, and sharded inputs are converted to interleaved first (`all_reduce_async.cpp:332-347,401-450`). So about 140 latency-bound CCL ops per step.
- **Default link count.** With `num_links` unset, the count is `get_num_links(mesh, cluster_axis)`: the minimum number of available routing planes along the axis (`ttnn/.../ccl/common/host/moe_utils.cpp:107-128`). The BH Galaxy fabric log reports "only 2 routing planes" (`tests/models/ltx/prompt_enhancer_experiments/results/e2b_on_dit_handle.md:31`), so expect 2 links. Topology comes from the fabric, which is `FABRIC_1D_RING` here.

*Weight bytes per device (bf16, computed from `models/demos/gemma4/configs/gemma-4-E2B-it/config.json`)*

| Component | Params | Bytes per device | Notes |
|---|---|---|---|
| MLP | ≈ 1.56 G of ≈ 1.88 G layer params | ≈ 389 MB | Layers 15-34 use `use_double_wide_mlp` (intermediate 12288 vs 6144). Dominant |
| Attention | — | ≈ 132 MB | 1 KV head, replicated per device (`attention/weights.py`, `kv_per_dev = head_dim if kv_replicated`) |
| PLI gate/proj | — | ≈ 55 MB | replicated |
| LM head | — | ≈ 101 MB | tied embedding |
| **Total** | | **≈ 677 MB** | MLP is about 57% of the total, and 68% of layer bytes |

*Wasted work*
- **Dead K/V columns on KV-shared layers 15-34.** These layers still compute K/V columns from zero-filled placeholder weights and then discard them (`model.py:184-220`, `attention/decode.py:110-113`). That is about 38 MB/step of reads, given replicated K/V.
- **Replicated PLI gate/proj.** Every TP device does the same PLI gate/proj work (`layer.py:168-199,318-333`).
- **Duplicate rows.** Rows 1-3 repeat all of row 0's work.

*LM-head tail and sampler*
- **LM-head tail.** Softcap is 3 eltwise ops (`model.py:1278-1283`), and the logits are padded to 32 rows inside the trace (`model.py:2369`).
- **Eager force-argmax sampler.** The op sequence is: optional invalid-vocab mask add → `all_gather_async` of 32 rows x 262144 bf16 (16 MiB gathered per device) → valid-vocab slice → chunked untilize (split, N untilizes, concat) → argmax (`models/common/sampling/tt_sampling.py:793-825,852-920`).

*Host path*
- Per step, serially: host PLI → 5 H2D copies → `execute_trace` → eager sampler → blocking token readback (`model.py:2148-2264`, `gemma4/tt/generator.py`).

**Hypothesized (all UNVERIFIED; no profile)**

| Bucket | Estimate | Assumption |
|---|---|---|
| CCL (about 140 ops) | 2-3.5 ms | 15-25 µs per latency-bound op |
| Weight reads (677 MB) | 1.4-2.3 ms | 300-500 GB/s per chip; BH DRAM bandwidth not checked |
| Small-op dispatch (about 2,000+ ops) | 4-8 ms | 2-4 µs per op |
| Eager sampler | 0.2-0.5 ms | — |
| Host gap (wall minus device) | ≈ 6 ms | Equals 17.4 − "~11 ms device". Rests entirely on the UNVERIFIED 11 ms |

**Working hypothesis (UNVERIFIED).** Per-op overhead and latency-bound CCLs dominate the device step; DRAM bandwidth does not. Host serialization makes up the rest of the wall time. A profile is required before acting on either claim.

## 4. Optimizations

"Gain" is the expected per-step gain for E2B on Galaxy. "Measured" means measured on some model or system, as stated in the row; nothing here was measured for E2B Galaxy.

| # | Optimization | Who does it | Expected gain for E2B | Applies? |
|---|---|---|---|---|
| O1 | **Device PLI in the decode trace** | PR #58524 `compute_pli_device` (diff hunk `model.py @@ -757,6 +765,124`), about **17-18 ops**: embedding, mul, linear, mul, 2 folds of (to_layout, reshape, to_layout), rms_norm, add, mul, to_layout, reshape, permute, optional slice, to_layout. **Replicated** 262144x8960 bf16 table = **4.38 GiB per chip**. Output is stacked `[L,1,rows,256]`, so consumers change too (hunks `@@ -889/-930/-1008`). HiFi4 + fp32 acc; PCC 0.9999931 vs host | Removes host PLI and 1 of 5 H2D copies, and is a prerequisite for O2. Adds about 17 device ops; net effect **UNVERIFIED** | **Yes, already decided (Phase 1, in progress):**<br>• Table is TP=8 column-sharded (1120 columns per chip, ≈ 0.55 GiB per chip), replicated over the 4 rows, plus one all-gather (`DEVICE_PLI_HANDOFF.md:51-59`; working tree uses `column_parallel` for the table).<br>• Projection and norm are replicated; the norm weight is used as-is, not (1+w).<br>• Contrast: #58524 replicates the full table and stacks across layers. Its op chain is a reference only, and its layout-op count is a reason to keep ours lean.<br>• Conflict with `4e6eb1f`: **UNVERIFIED**, no merge attempted. |
| O2 | **On-device token feedback and position `plus_one`** | `gemma4_31b_qb2/tt/generator.py:450-451,474`; `llama31_8b_qb2/tt/generator.py:262-271`; `llama3_70b_galaxy/tt/generator.py:1974-1981`; `tt_transformers/tt/generator.py:2316-2329`. Our non-PLI path already does it (`model.py:2180-2190,2361-2366`) | Removes 4 remaining H2D copies (tokens, pos, pos_int32, page_table) and the blocking host-to-trace dependency. Size of gain **UNVERIFIED**: the "≈ 6 ms host gap" rests on the 11 ms claim | **Yes, Phase 2, after O1.** Set `_tt_vllm_always_refresh_decode_trace_inputs=False` (`model.py:301`). This flag also gates `plus_one` on `current_pos`/`rot_mat_idxs` and the `[1,1,1,32]` token pad, so all three come together |
| O3 | **Traced sampling** | `gemma4_31b_qb2` second trace (`tt/generator.py:471-474,531-555`); `tt_transformers` | Removes eager dispatch of the about 5 to 7+ op force-argmax sequence (§3). **UNVERIFIED** | **Yes, but blocked.** Disabled because of frozen `all_gather_async` semaphores on replay (`model.py:241-249`), and there is an open replay hang on the sampling all-gather ring (`DEVICE_PLI_HANDOFF.md:34-45`). qb2 uses persistent buffers plus semaphore slots but `allow_force_argmax: False` (`tt/generator.py:77`), so the pattern transfers but not the op chain |
| O4 | **Async / persistent all-reduce, BF8 payload** | `gemma4_31b_qb2` `all_reduce_async` (`tt/decoder.py:457-469`); `llama3_70b_galaxy` fused CCLs (`tt/llama_ccl.py`); swept RS+AG (4,2,8) in `9abf56471fc` | 140 → 70 CCL ops. Perhaps 1-2 ms if CCL is 2-3.5 ms (**UNVERIFIED**) | **Yes, measure first.** `GEMMA4_CCL_ASYNC=1` was slower than sync on P150x8 (`ccl.py:65-67`) and has never been swept on Galaxy. BF8 payload risks greedy parity |
| O5 | **Cut wrapper/layout ops**: keep activations L1-sharded between ops, drop pad/slice and norm reshards, fold elementwise ops | Concrete code to copy:<br>• `3bd380c0837` (L1 residual island), `c718c1c2194` (−2 ops/layer), `c15299d85cf` (−2 slices/layer).<br>• qb2 fused ops: `ttnn.experimental.dit_fused_distributed_rmsnorm` (`tt/decoder.py:615`, stats buffer `:377`); `nlp_create_qkv_heads_decode` on L1 height-shard (`:970-976`); `paged_fused_update_cache` (`:1031`); fused `GELU_TANH` in the linear (`:539`) and as multiply input activation (`:672`); 56-core L1 width-sharded `[32,96]` activations (`:438-446`).<br>• `llama31_8b_qb2` folds RMSNorm affine weights into the next linear in fp32 (`README.md:11-12`).<br>• PR #59035, 13 kept changes: P150 DRAM-sharded program configs (QKV 6x5 grid; FF1/FF3 6x5 `in0_block_w=2`; FF2 `in0_block_w=4`; `num_workers_per_dram_bank`); QKV `fp32_dest_acc_en` off; SDPA decode `exp_approx_mode=True`; bfloat8_b outputs for attention dense and MLP FF1/FF3. Plus a KV-cache host-build workaround that is not a speed-up | Likely the largest device-side lever if the op-overhead hypothesis holds: 1-3 ms (**UNVERIFIED**).<br>Measured on Gemma-3-12B P150 (#59035): 39.05 → 27.45 ms device; the top 3 changes are −4.03, −3.98 and −1.19 ms. That is weight-bound single-chip, which is a different regime.<br>Caveat: #58524 sets `GEMMA4_TUNE_MATMULS` to no tuning on a multi-device mesh "where the target configs gave no decode gain", with −0.7% at TP=2 | **Yes, profile-guided.** Port the ign changes ungated from T3K and re-validate E2B parity: `c718c1c2194` warns E2B is rounding-fragile and is excluded today by its PLI gate. Expect matmul re-blocking alone to do little at TP=8 |
| O6 | **Skip K/V projection on KV-shared layers 15-34** | none found; our own waste (`model.py:184-220`, `attention/decode.py:110-113`) | About 38 MB/step less read, plus narrower wqkv matmuls. Small | **Yes**, low effort. Needs a separate Q-only weight for shared layers |
| O7 | **Shard PLI gate/proj across TP** | qb2 shards all projections | Small; [1536,256] matmuls | **Probably not.** It adds a CCL per layer |
| O8 | **BFP8/BFP4 weights (MLP first)** | `gemma4_31b_qb2` BF4 LoFi + BF8 head (`README.md:9`); `llama31_8b_qb2` BFP4 gate/up and BFP8 elsewhere (`README.md:9-11`); Gemma3 bfp4 MLP; `precision_overrides.json` (31B/12B only) | MLP is 389 of 677 MB per device. BFP8 MLP saves about 195 MB/step; BFP4 gate/up saves more. Time saved depends on whether reads are bandwidth-bound: **UNVERIFIED** | **Yes, after the profile.** A bigger lever than the draft implied. Gate on greedy parity and PCC, since E2B is rounding-fragile (`c718c1c2194`) |
| O9 | **Local argmax/top-k per device before the gather** | TTSampling non-force-argmax path: local top-k, then gather only the top-k values and indices (`tt_sampling.py:833-836`) | Gathers 8 x small (value, index) instead of 16 MiB of logits per device, and removes the full-width untilize. **UNVERIFIED** | **Yes.** Slicing the logits to 1 row does not help: TILE layout keeps 32 rows, and the trace pads to 32 (`model.py:2369`). Combine with O3 |
| O10 | **Use the 3 duplicate rows** | `gpt_oss` `users_row_sharded`/`sampling_dp=4` (`tt/model.py:242-243`); `tt_transformers` `create_submeshes` (`generator.py:4020`); Gemma3-27B DP=4 of 1x8 | **0 ms per step.** Throughput or chip freeing only (4 users, or 24 chips freed with a 1x8 submesh) | **Not a latency lever.** A 1x8 submesh needs a separate mesh-qualified weight cache (`e2b_on_dit_handle.md:37`) |
| O11 | **DRAM prefetcher / sub-devices / stall groups** | `llama3_70b_galaxy`, WH only; disabled on BH (`tt/model_config.py:608-613`) | **UNVERIFIED** | **No for now.** Unproven on BH; the BH Galaxy path runs without a prefetcher |
| O12 | **Pipelined host token read** (`GEMMA4_DECODE_PIPELINE=1`) | `d1ec6e77698` on `ign/gemma4_opt_decode` | ≈ 1 ms/token on 12B WH T3K (29.82 → 29.01 ms at b1). It wedged 31B b1 twice | **Yes, after O2.** It builds on O2: it engages only when device sampling writes into the trace token input and the host does not refresh inputs. It then overlaps the host loop with device work |
| O13 | **Speculative decoding** | PR #58524 (E2B fused route, about 14.5 ms per emitted token on 1 P150); PR #56048 (merged; MTP + dFlash; `tt/spec_decode.py`, `tt/dflash_drafter.py`, `tt/async_decode.py` in HEAD) | Large in tokens/s; per-step time is not comparable | Out of scope until O1-O5 |
| O14 | **KV-cache dtype BFP8** | PR #58394 `GEMMA4_KV_BFP8`; qb2 and `llama31_8b_qb2` use BFP8 KV | Small at short context (KV reads are small at ≤2k); **UNVERIFIED** | Maybe. Check parity |
| O15 | **SDPA decode program config** | #58394 `GEMMA4_SDPA_MAX_CORES_PER_HEAD`; #59035 `exp_approx_mode=True` | **UNVERIFIED** | Profile first |
| O16 | **RoPE path** | #58394: fused single-position op at b1 vs per-user elementwise op (`GEMMA4_ROPE_PERUSER_B1`; PCC 0.99999, not bitwise) | **UNVERIFIED** | Check which path E2B b1 uses; keep the fused one |
| O17 | **Fabric config / worker L1 / dispatch axis** | BH Galaxy Qwen3-32B uses `FABRIC_2D_TORUS_XY`, `worker_l1_size` 1345000, `dispatch_core_axis: col` (`llama3_70b_galaxy/README.md:177-179`). Our job ran `FABRIC_1D_RING` | **UNVERIFIED** | Worth an A/B. Our CCLs are on axis 1 (rows) only, so 2D torus is not required; device PLI's all-gather is also axis 1 |

## 5. Ranked next steps (gain / effort)

1. **Tracy op profile of one decode step on the pinned bh-glx-120-b02u08.** Effort: low. Gain: decides everything below.
   - Use `E2B_PROFILE=1` in `e2b_bringup.py::test_e2b_decode_timing`. It decodes 1 + `E2B_PROFILE_DECODE_STEPS` steps (default 3) and places `start`/`stop` signposts around the steady steps (`e2b_bringup.py:25-31,434-437,527-624`).
   - Run under `python -m tracy -p -r -v -m pytest ...`.
   - Set `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` above the per-window op count. The default is 1000; one step is estimated at 2,000+ ops, and #59035 used `--op-support-count 40000`.
   - This confirms or kills "~11 ms device", splits CCL vs non-CCL time, and gives per-op counts.
   - #59035's `tt-opt` loop is the reusable method: Tracy window → one edit → accuracy gate → re-measure → keep only if the gate passes and the window falls.
2. **Phase 1 (O1): device PLI, already in progress.** Sharded table, gated by `GEMMA4_DEVICE_PLI=1`. Exit criteria: PCC ≥ 0.999 and greedy parity (`DEVICE_PLI_HANDOFF.md:64-67`). Keep the op chain short.
3. **Phase 2 (O2): token feedback and `plus_one`**, then **O12** on top.
4. **O5 + O6, guided by the profile.** Port `3bd380c0837`, `c718c1c2194` and `c15299d85cf` with parity checks, plus the qb2 fused norm and QKV-heads ops.
5. **O9 + O3: local argmax before the gather, then trace the sampler.** Do this together with the sampling all-gather replay hang.
6. **O8: BFP8 MLP weights**, if the profile shows weight-read time matters.
7. **O4, O17, O15, O16, O14** as A/B tests on the pinned node.
8. **O10 and O13**: throughput levers; separate decision.

**Risks**
- **Parity.** Device PLI is not bit-exact (PCC 0.9999931 in #58524). E2B is rounding-fragile (`c718c1c2194`).
- **Sampling all-gather axis.** TTSampling's default all-gather axis is 0 (`tt_sampling.py:241`). Ours is 1 (`model.py:712`); keep it.
- **Link count.** BH Galaxy has 2 links (`llama3_70b_galaxy/tt/model_config.py:718-721`).

## 6. Open questions

1. **Device time and CCL share.** What is the real device time per step, and how much of it is CCL? (Answered by step 1.)
2. **Fabric choice.** Does `FABRIC_2D_TORUS_XY`, or a different `worker_l1_size`, change axis-1 CCL latency on BH Galaxy? **UNVERIFIED.**
3. **Matmul tuning at TP=8.** Do PR #58524's matmul configs help at TP=8? Its own README says they do not on a mesh.
4. **PR #58524 status.** Will it land, and does it conflict with `4e6eb1f`? It was last updated 2026-10-01 and is mergeable state UNKNOWN. **UNVERIFIED.**
5. **PLI norm convention vs HF.** The norm weight is used as-is: host code (`model.py:762-765`), #58524 and the handoff (`DEVICE_PLI_HANDOFF.md:55`) all agree. The HF upstream source was not checked. **UNVERIFIED.**
6. **Sync all-reduce input layout.** Does any sync all-reduce input arrive sharded? If so, it would take the extra interleave conversion (`all_reduce_async.cpp:343-348`). Check in the profile.
