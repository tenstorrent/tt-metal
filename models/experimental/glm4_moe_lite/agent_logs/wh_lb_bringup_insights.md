# GLM-4.7-Flash on Wormhole LoudBox (T3K 1×8) — Bring-up Insights

*Analysis snapshot. Cross-checked against the code on branch `gtobarTT/glm_47_optimization`.*

## 1. TL;DR — where the bring-up stands

- **Runs end-to-end on the LoudBox as a 1×8 (1-D) mesh**, TP=1 + expert-parallel, PCC vs HF
  **0.93–0.96** across ISL 128→8192. (2×4 drops to ~0.65 — the MoE all-reduce must be single-axis.)
- **Long context works**: chunked prefill + expert weight sharding + bf8 KV reach ISL 32K/64K
  (previously OOM ~30K). 128K is supported by the machinery but not yet captured as a clean WH row.
- **Best decode**: batch-1 ~88 ms/token @ ISL 512 (baseline sweep); a batch-4 ~76.3 ms/token result
  is reported but not yet in the committed tables.
- **Device-op profiling (Tracy) FIXED (2026-07-13)** — was blocked by a build-path-dependent 16-bit
  zone-hash collision; fixed with `-ffile-prefix-map` in `tt_metal/jit_build/build.cpp` (details in §7).
  Device perf sheets populate again.
- **Known perf lever left on the table**: the ring all-reduce is ~37% of decode device time and runs
  on 1 CCL link; 2 links isn't reachable under the standard 1×8 fabric mapping (details in §6).

## 2. Model & hardware

| | |
|---|---|
| Model | zai-org/GLM-4.7-Flash — 47 layers, MLA attention, MoE |
| Layers | layer 0 **dense**; layers 1–46 **MoE** (`first_k_dense_replace=1`) |
| MoE | 64 routed experts, top-4, + 1 shared expert; `moe_intermediate_size=1536` |
| Attention | MLA: 20 heads, `kv_lora_rank=512`, `qk_rope_head_dim=64` |
| Shapes | hidden 2048, vocab 154,880, max positions ~202K |
| Hardware | WH LoudBox / T3K — 8 Wormhole B0 chips, run as **1×8** mesh |
| Dispatch | ETH (frees all 64 Tensix); fabric FABRIC_1D |

## 3. Code structure — how the blocks are distributed & implemented

`models/experimental/glm4_moe_lite/tt/` (sizes give a feel for where complexity lives):

| Module | LoC | Responsibility |
|---|---:|---|
| `model_tt.py` | 3787 | **Top-level runner / orchestration**: prefill (unchunked / chunked / traced / batched), decode (trace-sampling / trace-logits), the per-layer loop, embedding, LM head, KV-cache alloc |
| `moe_tt.py` | 2303 | **Sparse MoE experts** — router, `moe_expert_token_remap`, `sparse_matmul`, expert combine. **Where most collectives live (~20 all_gather/reduce_scatter)** |
| `linear_helpers.py` | 1994 | Matmul/linear helpers, program configs, tuned decode/prefill grids (`_DECODE_MATMUL_TUNED`, `prefill_matmul_tuned_enabled`) |
| `layer0_tt.py` | 1713 | Layer-0 **reference harness** for the `*_optional.py` unit tests + RoPE tensor helpers (`make_rope_tensors`). *Not* the main forward path |
| `decoder_layer_tt.py` | 1446 | **The generic decoder layer.** Entry points `run_decoder_layer_prefill_update_cache_tt` / `..._decode_one_step_...`; dispatches attention block + MLP/MoE block; dense-vs-MoE branch *inside*; `_fused_kv_branch_forward` |
| `layer_weights.py` | 1057 | **Weight conversion (torch→TT) + sharding**: `_tp_mesh_mapper` (TP), `ShardTensorToMesh(dim=0)` (expert parallel), dtype selection, cache keys |
| `attention_decode.py` | 625 | **MLA attention block** — q/kv projections, FlashMLA, o_proj (**1 all_reduce**, after o_proj, when TP-sharded) |
| `generator_vllm.py` | 602 | vLLM adapter (`tt_data_parallel`, model init) |
| `mlp_decode.py` | 315 | **Shared-MLP + MoE forwarding wrapper** (3 collectives) |
| `mtp_forward.py` | 231 | Multi-token-prediction (layer 47) |
| `runtime_config.py` | 272 | All `GLM4_MOE_LITE_*` env flags parsed once into a frozen dataclass |
| `config.py` | 129 | Hyperparameters from HF config |
| `reference_*.py`, `tt_embedding.py`, `weights.py` | — | HF reference blocks (PCC oracles), embedding, weight load/cache |

### Forward-path walkthrough

**Per-layer loop** (`model_tt.py`, e.g. line 931/1904): every layer 0–46 goes through the *same*
generic entry in `decoder_layer_tt.py`. The dense-vs-MoE choice is made **inside** the layer
(`layer_idx < first_k_dense_replace` → dense MLP; else MoE) — there is no separate top-level dense path.

**A decoder layer =**
1. **Attention block** (`attention_decode.py` for decode; prefill path in `decoder_layer_tt.py`):
   RMSNorm → `w_q_a`/`w_kv_a` (optionally fused as QKV_A) → q_a/kv_a layernorms → `w_q_b` →
   RoPE → **FlashMLA** (full-head: 20 heads not divisible by 8) → `w_kv_b2` → `w_o` →
   **one `all_reduce`** (TP-partial → full) when `ATTN_DP=0`.
2. **MLP / MoE block** (`mlp_decode.py` → `moe_tt.py`):
   - Layer 0: dense MLP (gate/up/down).
   - Layers 1–46: shared expert (dense) **+** routed MoE. Router picks top-4 of 64;
     `moe_expert_token_remap` builds per-device sparsity (8 experts/device); `sparse_matmul`
     runs the experts; results combined and **reduce-scattered / all-gathered** across the 8
     columns. `FUSE_MLP_MOE_REDUCE` consolidates the dual RS+AG pairs.

**Prefill vs decode are separate implementations**, not one parametrized path:
- **Prefill** (`_prefill_*` in `model_tt.py`): chunked when `MAX_PREFILL_CHUNK_SIZE>0` and
  `prompt_len>chunk` (128 on WH — keeps per-matmul M small so the MoE gate fits L1); batched path
  for batch-1; an experimental fully-traced chunk loop (`_prefill_chunked_single_user_traced`).
- **Decode** (`decode` @ 1605, `_decode_step_tt_logits`, `_decode_trace_sampling`): trace-captured;
  activations kept in L1 (`DECODE_L1_ACT`), expert intermediates in L1 (`EP_L1`); batch-bucketed
  traces for batch>1; FlashMLA cores pinned to 16 (`DECODE_MLA_CORE_SCALE=0`) so decode fits L1.

### How weights are distributed across the 8 chips
- **Expert parallel (EP)**: 64 experts → **8 per device**, `ShardTensorToMesh(dim=0)`
  (`layer_weights.py:333`). This is the "weight sharding" that unblocked long context.
- **Tensor parallel (TP=8)**: dense-MLP, MoE non-expert, and attention projections are hidden-dim
  sharded across the 8 columns via `_tp_mesh_mapper` (`layer_weights.py:49`).
- **Head-parallel: OFF** — 20 heads ∤ 8, so MLA runs full-head (replicated compute).
- **Data-parallel: none** — the 1×8 has no second mesh axis; a batch is processed cooperatively by
  all 8 chips (TP+EP), not split across replicas. (`ATTN_DP=0` → attention is TP-sharded, not replicated.)

## 4. PCC (correctness) tests

`tests/pipeline_tests/` (full-model, HF-compared) and `tests/*.py` (block/unit):

| Test | Scope | Notes |
|---|---|---|
| `pipeline_tests/test_text_prefill_logits_wh.py::test_text_prefill_logits_wh` | WH full-model prefill logits vs HF, **TP=0** (replicated) | bar 0.95; bf8 experts (bf16 OOMs WH) |
| `…::test_text_prefill_logits_wh_tp1_1x8` | WH full-model prefill logits vs HF, **TP=1 on 1×8** | the production tensor-parallel path; PCC 0.93–0.96 |
| `pipeline_tests/test_text_prefill_logits.py` | **Blackhole** full-model prefill logits (bf16 experts, 0.97) | BH-only |
| `pipeline_tests/test_text_decoder.py` | Decode-path logits | |
| `tests/test_reference_layer0.py`, `test_reference_moe_layer1_optional.py` | HF reference blocks (PCC oracles) | |
| `tests/test_tt_layer0_*_optional.py` (7 files) | Layer-0 attention/decode/cache variants vs reference | incl. batch-32 RoPE, unpaged, update-cache |
| `tests/test_tt_moe_layer1_*_optional.py` | MoE layer-1 (single + mesh) | mesh test exercises all_reduce |
| `tests/test_tt_decoder_layer0_*_optional.py` | Generic decoder layer == layer0 harness | prefill & decode |
| `tests/test_tt_embedding_optional.py` | L1 vs DRAM embedding parity | |
| `tests/test_flash_mla_decode_boundary_optional.py`, `test_pre_sdpa_kernel.py` | FlashMLA cache boundary; PreSDPA fused kernel | |
| `tests/test_tt_golden_truncated_n2_optional.py` | Greedy token matches expected | |
| `scripts/pcc_vs_hf.py`, `scripts/ab_prefill_pcm_pcc.py` | Standalone PCC-vs-HF + A/B (old-vs-new prefill) | |

WH PCC runs need no env prefix now — `apply_wh_correctness_env` / `apply_wh_tp1_env` bake the config,
and the module sets `TT_METAL_GTEST_ETH_DISPATCH=1` at import.

## 5. Perf tests / drivers

| Driver | Produces | Config |
|---|---|---|
| `agent_logs/run_baseline_b1_wh_1x8.sh` | The WH 1×8 batch-1 latency table (README) | Self-contained; drives the greedy script; bf16 KV <32K, bf8 ≥32K |
| `scripts/run_sweep_isl_batch.py` | ISL×batch sweep (Blackhole/Galaxy tables) | Bakes the aggressive perf config internally |
| `scripts/debug_run_full_tt_greedy.py` | Single-run benchmark / debug | **Auto-applies** the tuned per-platform config (WH T3K 1×8 / Galaxy / BH) via `setdefault` |

**Caveat:** never quote Tracy-run wall-clock as latency — instrumented/eager execution inflates it
massively (ISL-512 prefill ~297 s under Tracy vs the real ~1.9 s). Use the baseline sweep for latency.

## 6. CCL links — investigated & CLOSED (the ~37% all-reduce is not fabric-reducible)

The ring all-reduce is ~37% of decode device time and runs on **1 CCL link**. Fully investigated:
- Cluster descriptor shows **every direct connection has 2 Ethernet links** — so "1 link" isn't a
  cable limit.
- **Root cause of the 1-link cap (architectural):** `get_num_links`
  (`ttnn/cpp/ttnn/operations/ccl/common/host/moe_utils.cpp:173`) subtracts 1 routing plane on
  non-all-MMIO meshes. T3K = 4 local + **4 remote chips whose 2nd Ethernet link is reserved for the
  MMIO/fast-dispatch tunnel** → 2−1 = 1 plane for CCL. Matches the `T3K:(1,1)` hardcode
  (`models/common/modules/tt_ccl.py:206`).
- `CCL_NUM_LINKS=2` **hangs at the first all-reduce** (reproduced). A hand-picked 2-link Hamiltonian
  order fails fabric init (`ETH core heartbeat`) — ttnn rejects arbitrary 1×8 orderings, and the
  tunnel reservation is order-independent anyway.
- **`FABRIC_1D_RING` on T3K: tested, negative** — decode **91.8 ms vs 88.8 baseline (~3.4% slower)**
  at ISL512. Links are still 1/hop, so the ring fabric only adds routing overhead.

**Conclusion:** `num_links=1` + FABRIC_1D is optimal for the supported T3K 1×8. Reducing the all-reduce
further is tt-metal fabric-team territory (relax the conservative `-1` for pure-1D meshes), not a
GLM-side change. See `memory/project_glm_t3k_ccl_links.md`.

**Device gotcha:** a killed hung-collective run wedges ETH cores (`heartbeat ... post code 2050000`) —
even the normal baseline then fails at fabric init. Recover with `tt-smi -r 0,1,2,3` and reset
*before* the next run.

## 7. Tracy device-op profiling — FIXED 2026-07-13 (was a build-path zone-hash collision)

**FIX:** `tt_metal/jit_build/build.cpp` `common_flags` now adds `-ffile-prefix-map=<root>=` so `__FILE__`
is relative (build-path-independent). The collision below was caused by the **absolute build path**
folding two firmware zones (`BRISC-FW` brisc.cc:433, `ERISC-KERNEL` erisck.cc:29) to the same 16-bit
hash `0xa8d8` — unique to the `glm47_wh_lb` path (other checkouts on the box were collision-free, which
is why it "worked before" elsewhere). After the fix: rebuild `libtt_metal`
(`cmake --build build_Release --target tt_metal`; then copy `tt_metal/libtt_metal.so` → `lib/`), and the
first profiling run recompiles firmware/kernels once. Verified: tracy exit 0, no collision throw, both
CSVs populate (34,761 rows), full perf sheet renders via `tt-perf-report <csv> --min-percentage 1.5`
(NB: this build's CLI dropped the old `--arch/--group-by` flags). Also patched the local pip
`tt-perf-report` for `HiFi3` fidelity (`tflops_per_core`), not version-controlled.

### Historical root-cause detail (pre-fix)

**Intended flow** (device per-op sheet):
```bash
GLM4_MOE_LITE_PROFILER_READ_INTERVAL=1 GLM4_MOE_LITE_SIGNPOST=1 \
python -m tracy -r -p -o /tmp/glm_prof -- \
  models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --simulate-context-len 512 --max-new-tokens 1 --phase prefill \
  --mesh-rows 1 --mesh-cols 8 --cache-dir ~/.cache/ttnn/models/glm4_moe_lite/wh_1x8
# then:
tt-perf-report /tmp/glm_prof/reports/*/ops_perf_results_*.csv --arch wormhole --group-by category
```
`ReadDeviceProfiler` is wired into `model_tt.py` (gated by `PROFILER_READ_INTERVAL`); `signpost()`
brackets regions (gated by `SIGNPOST`).

**Why it currently fails (reproduced):** during the run, `ttnn.ReadDeviceProfiler` throws
`TT_THROW @ tt_metal/impl/profiler/profiler.cpp:280: "Source location hashes are colliding"`.
Both `profile_log_device.csv` and `cpp_device_perf_report.csv` come back empty, so Tracy
post-processing then asserts `Device N present in host logs but missing from cpp_device_perf_report.csv`.

Root cause: tt-metal hashes each kernel-zone source-location string into **16 bits** (`hash16CT`) and
hard-throws on any collision (the device kernels emit the same 16-bit marker, so it can't be
suppressed). Only ~28 distinct zones exist here — this is *not* a "too many zones" problem; two
**built-in firmware zones** collide at `0xa8d8`:
`BRISC-FW (brisc.cc:433)` vs `ERISC-KERNEL (erisck.cc:29)`. The hashed string includes the
**absolute build path**, so it's specific to this repo's build (`/home/ttuser/sdawle/glm47_wh_lb/…`) —
which is why the identical flow worked under `/home/gtobar/tt-metal`. Independent of the GLM harness,
ISL, `PROFILER_READ_INTERVAL`, and `NUM_LAYERS` (=1 still collides). `ERISC-KERNEL` is present
because it's a multi-chip (ethernet-fabric) run.

**Options for a device sheet** (none is a flag flip):
1. Patch tt-metal — rename one colliding firmware zone (e.g. `ERISC-KERNEL`) or widen the zone hash,
   then rebuild. Device + host must agree on the hash, so probing/suppressing host-side is wrong.
2. Profile on **Blackhole** — arch string is `blackhole`, different hashes → collision unlikely.
3. Fallback: host-side `tracy_ops_times.csv` (it *does* populate) — dispatch-side, Tracy-inflated;
   fine for relative op ranking, not absolute latency.

See `memory/project_glm_tracy_profiling.md`.

## 8. Open opportunities (ranked)

1. ~~**Fix Tracy**~~ — **DONE (2026-07-13)**, `-ffile-prefix-map` in build.cpp (§7). Device-op
   breakdown available again; use it to find the next real decode/prefill levers.
2. ~~**trace + 2cq overlap**~~ — **IMPLEMENTED & CLOSED (2026-07-13).** Added `GLM4_MOE_LITE_DECODE_2CQ=1`
   (**default ON**; `=0` disables; auto-falls back to single-CQ on a 1-queue device): per-step decode
   inputs written on CQ1, `execute_trace` non-blocking on CQ0 with
   `record_event`/`wait_for_event` sync + a single-buffer clobber guard (`_execute_decode_trace_2cq`
   in `model_tt.py`). A/B at ISL=512 decode-only, all greedy-equivalent (byte-identical output):
   sampling-b1 **88.9→89.0**, logits-b1 **106.9→107.8**, sampling-b4 **86.1→86.8** ms/tok — **flat in
   every mode** (within run-to-run noise). Decode is device-bound (~89 ms) with negligible host I/O,
   and two structural facts kill the overlap: (a) decode inputs use a **single persistent buffer** so
   the CQ1 write of step N+1 must wait for step N's compute (serialized, not overlapped); (b) decode
   is **autoregressive** — the next input token is the argmax of the current output, so even the
   154,880-vocab logits readback is on the critical path and can't hide behind the next compute. A real
   win needs **double-buffered inputs + on-device token feedback** (re-embed the sampled token inside
   the trace, no host round-trip). Flag kept as a ready building block; today it's a latency no-op.
3. **Land 128K on WH** as a clean row (trace-region growth is the blocker at 131072).
4. **Benchmark & document batch=4** (the ~76.3 ms/token result) — add `FUSE_QKV_A=1` to the WH auto-env.
5. ~~2-link CCL ring~~ — **CLOSED** (architectural: remote-chip tunnels cap it at 1 link;
   FABRIC_1D_RING tested ~3.4% slower). tt-metal fabric-team only.
6. **Head-parallel** is fundamentally blocked (20 ∤ 8); only a DP-row layout (Galaxy) reclaims it.
