# DeepSeek-V4.1-Flash prefill on Tenstorrent: start here

For a Claude Code session (and its human) picking up this model. Give this file to your Claude first. It covers what
the bring-up is, its current state and defaults, how to set up, run and measure it, the prefill → decode (PD) serving
hand-off, the open questions, and where everything is.

Facts are as of 2026-11-07, branch `dnijemcevic/model_bringup_rebased`, tree `902322e6f77` (task P.42). Items marked
**(unverified)** were not checked against code or a run while this was written.

## 1. What this is

Chunked prefill of `deepseek-ai/DeepSeek-V4.1-Flash` (text decoder only: no vision, no MTP), all 40 layers, on a
Blackhole Galaxy 8x4: 32 x p150b chips, fabric `FABRIC_2D_TORUS_XY`, SP = 8 mesh rows (the sequence is split over
them), TP = 4 mesh columns (the hidden size is split over them). "Prefill" = computing the prompt's caches; a separate
decode machine (tt-blaze) generates the tokens after a hand-off.

- The gated bring-up framework (`models/demos/common/bringup`) organizes the work as tasks with gates (tests that must
  pass) in a ledger (section 10).
- The model code is `models/demos/deepseek_v41_flash_d_p/tt/`. Every switch is in `tt/settings.py`.
- The tt-d-gen inference server drives the model through the adapter in `tt/runners/`.

| Item | State (2026-11-07) |
|---|---|
| Deployment box / spec | Galaxy 8x4: `models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml` |
| Seed run | QuietBox 2x2, partial (layers 0-3, 14, 20, 21, 24, scaled model): `bringup/spec.yaml`, ledger in `bringup/` |
| Last full run | X.3 PASS 2026-11-05 (full 56,320-token ladder, every limit holds) |
| Accuracy now (rung `last`) | err_final_hidden 0.1313 (limit 0.242), top-5 1.0 (P.40 / P.41 agents' ladder runs; P.42 left every layer's error identical to the digit) |
| Perf now (one 5,120-token chunk at 51,200, P.43 tree) | device 189.02 ms, wall 204.4 ms, host overhead 15.4 ms (section 4); +11.7 ms device since the contract work began (section 4, regression table) |
| Tasks done in the last two days | P.37-P.43 accepted and committed (section 3); their full gates are still pending (they run before the final task) |
| Running now | Nothing; the orchestrator stopped after P.43 (`--until P.43`) |
| PD hand-off | Our served table: `bringup/migration_configs.md`. Decode reviewed it (Brian Liu); Kartik Paigwar proposed a different final contract (section 8) |
| Held | P.35, P.36, O.1, Z.1, F.1; the QuietBox run |

## 2. Current defaults

`tt/settings.py` is the source of truth. Each entry carries its owner decision and measurements: read it before
flipping one. Every option below is kept behind its switch; nothing old was deleted.

| Switch | Default | Other values | Since | Why |
|---|---|---|---|---|
| `MOE_BLOCK_SRC` | `main` | `carried` | P.37 (2026-11-06) | Marko Staletovic's reworked MoE block (C++ ops `ttnn.bringup.moe_block_main`, no DRISC). `carried` = the block used since P.17, which pushes expert weights with the DRAM cores' RISCs (DRISC) |
| `CKV_FORMAT` | `bf16` | `bfp4_rows` | P.38 (2026-11-06) | Compressed KV in bf16 for the PD first cut; 4-bit comes back later |
| `WINDOW_KV_FORMAT` | `bf16_tiles` | `bfp8_tiles`, `bfp8_blocks`, `bf16` | P.38 | Window KV computed and read in bf16 |
| `INDEX_K_FORMAT` | `bfp8` | `bfp4`, `bf16` | P.40 (2026-11-07) | The index keys the indexer scores; bfp8 picks almost what bf16 picks |
| `INDEX_K_COPY` | `bfp8` | `bf16` | P.41 (2026-11-07) | One bfp8 copy of the index keys, no 16-bit copy. Needs `INDEX_K_FORMAT=bfp8` and `INDEX_K_SERVED=fp8_row192`, else falls back to bf16 (logged once) |
| `INDEX_K_SERVED` | `fp8_row192` | `bf16` | P.39 (2026-11-06) | Served index keys as Blaze decode's 192-byte fp8 rows |
| `WINDOW_KV_KEEP` | `snapshot128` | `full`, `ring`, `none` | P.42 (2026-11-07) | Serve a 128-row window snapshot per layer at cap (section 8). The model keeps 192 rows a layer for its own attention |
| `REINDEX_SERVE` | `alias` | `copy` | P.42 | Re-index layers 24, 28, 32, 36 served as aliases of layer 20's index rows |
| `KV_BLOCK_SIZE` | 64 | any (32 legal in tt-d-gen) | P.42 | The serving engine's `kv_block_size`; sets cap |
| `DECODER_SKIP` | `off` | `exact`, `replay128` | P.30 / P.31 | Skip layers 20-39 on chunks marked not-last. Off until the owner decides |
| `CCL_IMPL` | `fabric_bf16` | `fabric`, `stock` | P.5 / P.6 | Collectives: gathers on `ttnn.experimental.high_bw_all_gather`, reduce-scatters on `ttnn.bringup.fabric_reduce_scatter`, TP partial sums in bf16 |
| `ENGRAM_TABLE` | `resident` (tests: `mmap`) | `mmap` | FIX.10 | Engram tables in host RAM vs the checkpoint mmap. Bit-identical. Timed runs set resident |
| `EXPERTS_IMPL` | `bfp4_mse` | `bfp4`, `bfp8`, `bfp8_down4` | FIX.22 | Routed experts bfp4 with an MSE-chosen exponent |
| `MOE_IMPL` | `block` | `flat` | FIX.6 | Block MoE with a fused head. `flat` hangs on the torus (DS41-130) |
| `OWN_ROWS`, `POST_COLLAPSE` | `on` | `off` | P.23, P.34 | Residual / mHC streams on each chip's own rows; mhc_post writes the next collapsed input |
| `MAX_WORKER_CORES` | 110 | 0 = no cap | FIX.15 | Galaxy power rule: never more than 110 cores an op. Never 0 on a Galaxy |
| `EXPERTS_WEIGHT_CACHE`, `EXPERTS_PREFETCH` | `on` | `off` | | Bank-major expert cache and read-ahead |

## 3. Tasks of the last two days (2026-11-05 to 2026-11-07)

"Device" and "wall" are one 5,120-token chunk at position 51,200 (p51200), off → on, from the A/B tables
`/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/ab/<task>/table.md`. Device = time
the chips spend; wall = host clock for the chunk; host overhead = wall - device.

| Task | What it did | Commit | Measured effect (off → on) |
|---|---|---|---|
| X.3 | Final accuracy at full target (56,320), warm full prefill, one warm chunk at several positions | `907e8d61160` (pre-rebase) | PASS. Then the rebase onto llk_helper_library `107f838f379` (2026-11-05/06, `bringup/REBASE_STATUS.md`) |
| P.34 | `POST_COLLAPSE`: mhc_post also writes the next sublayer's collapsed input | `f123afcc5a1` | Bit-identical; gate err_final_hidden 0.1394, top-5 1.0, device 157.67 ms, wall 172.4 ms (on its own A/B setup) |
| (port) | Marko's MoE block ported as `ttnn.bringup.moe_block_main` + `models/demos/moe_block_main`, switch `MOE_BLOCK_SRC` | `bc71f592543`..`494262dfccb` | Default stayed `carried` until P.37 |
| P.37 | `MOE_BLOCK_SRC=main` reaches parity with the carried block (SE_PACK_PRECISE, NORM_INTO_MOE, DECODER_SKIP=replay128); made default | `bf49ffcd902` | device 177.60 → 177.27 ms, wall 194.9 → 194.0 ms; MoE 2.67 ms device a layer (carried 2.39) |
| P.38 | Compressed and window KV in bf16 (computed and served) | `53832f4caac` | device 179.09 → 183.57 ms (+4.5, attention 44.2 → 50.0), wall 196.3 → 203.7; compressed-KV error 1.4-4x lower |
| P.39 | Served index keys as Blaze's fp8 192 B rows (new fork `ttnn.bringup.pack_fp8_rows`) | `d70e41c47f0` | device 184.87 → 182.97 ms, wall 199.9 → 200.8 (noise); served index-key error L02 0.016 → 0.053 (limit 0.236) |
| P.40 | `INDEX_K_FORMAT` gains bfp8, made default (A/B bf16 / bfp4 / bfp8) | `2a96853dbc3` | device 183.78 → 183.56 ms; final hidden 0.1334 → 0.1313; top-k overlap ~0.85 → ~0.92 (= bf16); key buffers 1.9x bfp4's memory |
| P.41 | One bfp8 index-key copy, no bf16 copy; served rows built from it | `98463fd81bc` | device 184.93 → 183.80 ms, indexer 7.06 → 6.83 ms; copies 127.95 → 44.39 MB a chip a slot at 1M; accuracy identical |
| DS41-216 fix | One `high_bw_all_gather` semaphore pair per mesh axis (section 9) | in `902322e6f77` | Stress tests: 2,048 wrong replicas → 0 |
| P.42 | Re-index layers as aliases of layer 20; window KV as a 128-row snapshot at cap (decode's review) | `902322e6f77` | device 184.13 → 188.74 ms (+4.6: the snapshot writes, in section ffn_residual 5.06 → 11.69), wall 202.9 → 206.5; window + copies 1,377 → 20 MiB a chip a slot at 1M; accuracy identical |
| P.43 | Decode hand-off clean-up. (1) `SKIP_TAIL_START=cap` (default; `end` = old): under `DECODER_SKIP=exact` / `replay128` the last chunk starts from cap (replay128 at or before min(end - 128, cap - 128); exact at or before min(end - 2688, cap - 2668)); a later last chunk is refused (ValueError); every layer's snapshot is written in both modes. (2) The framework contract check (`testing/contract.py`) checks Snapshot configs only at the request's last chunk, as tt-d-gen ships them (DS41-218); new host unit test `test_contract_acks.py`; full `test_contract` passes (acks_early 0). (3) `0_window_kv` declares `max_sequence_length` 128. Owner accepted (2026-11-07) the snapshot test's limit for the 8 layers with official per-position references taken from the official reference's own error (DS41-193 method; layer 20 rows [0, 64): 0.144 -> ~0.180). Under replay128, layers 21-39's snapshot rows are 0.97-2.2x their limit by design (DS41-221: the tail's window starts empty), recorded, not gated | `f087d1b682a` | device 190.0 → 189.0, wall 208.0 → 204.4 (no change: the harness never marks a last chunk, so the skip does not act); accuracy identical |

Owner process notes: P.41 accepted by the owner 2026-11-07 09:25; P.42 held until DS41-216 was root-caused, then
re-measured on the fixed tree (17:19) and committed (17:26). P.43 accepted 18:45 (commit rebased onto the README commit `cf686613dff`, so `f50b38ac87f` in the orchestrator log is now `f087d1b682a`).

## 4. Perf and accuracy headline (current defaults)

One 5,120-token chunk at 51,200, P.43 tree, `ENGRAM_TABLE=resident`, from `ab/P.43/table.md` (2026-11-07 18:25;
section split below from `ab/P.42/table.md`, unchanged by P.43):

| Number | Value |
|---|---|
| device_ms_total (profiled) | 189.02 ms |
| chunk_wall_ms (clean warm, median of 10) | 204.4 ms |
| host overhead (wall - device) | 15.4 ms |
| last chunk after the golden 51,200 prefix (quick run) | wall 186.9 ms |

### What got slower since the contract work began (P.38 on)

Baseline P.37 (Marko's MoE block, before the contract work). Each row is that task's A/B at 51,200 (on - off).
Run-to-run noise is about ±1 ms device and ±3 ms wall.

| Task | Change | Device ms | Wall ms | Where |
|---|---|---|---|---|
| P.37 (baseline) | MoE block switch | **177.3** | **194.0** | |
| P.38 | compressed + window KV in bf16 (were 4-bit / 8-bit) | +4.5 | +7.4 | attention +5.8 |
| P.39 | index keys as Blaze 8-bit rows | ~0 | ~0 | noise |
| P.40 | index keys scored in bfp8 | ~0 | ~0 | noise |
| P.41 | drop the bf16 index-key copy | ~0 | ~0 | indexer -0.2 |
| P.42 | 128-row window snapshot, layer-20 aliases | +4.6 | +3.6 | ffn_residual +6.6 (the snapshot rewrite), attention -1.8 |
| P.43 | skip-mode tail start, contract check | ~0 | ~0 | noise |
| **Now** | | **189.0** | **204.4** | **+11.7 ms (+6.6%) device, +10.4 ms (+5.4%) wall** |

By section, P.37 → now: attention 44.1 → 48.2 (+4.1), ffn_residual 5.1 → 11.7 (+6.6), experts 106.8 → 107.7 (noise),
compressor 1.8 → 2.0, indexer 7.0 → 6.9.

Both costs are recoverable:
- **bf16 KV (~4 ms)** is the temporary format for the first e2e run. Back to 4-bit (`CKV_FORMAT=bfp4_rows`) once
  decode reads it.
- **The snapshot rewrite (~6.6 ms a chunk)** happens on every chunk because prefill cannot tell the last chunk
  (tt-d-gen's chunk header has no last flag). With a last-chunk flag, or N (the prompt length) in the metadata as
  Kartik's contract proposes, it runs once per request. The same signal lets the decoder skip act
  (`DECODER_SKIP=exact`: layers 21-39 skipped on every chunk but the last).

| Section | Device ms |
|---|---|
| experts (MoE) | 107.36 |
| attention | 48.14 |
| ffn_residual (includes the snapshot writes) | 11.69 |
| indexer | 6.88 |
| attn_residual | 5.31 |
| q_a | 3.41 |
| compressor | 2.01 |
| engram_kv | 0.94 |
| kv, ffn_norm, attn_norm, engram_add, attn_hc | 0.73, 0.70, 0.70, 0.62, 0.25 |

| Accuracy (rung `last`) | Value | Limit | Source |
|---|---|---|---|
| err_final_hidden | 0.1313 | 0.242 | P.40 / P.41 agent ladder runs (bfp8 index keys) |
| top5_overlap | 1.000 | >= 0.994 | same |
| err_layer_L39 | 0.7067 | 1.533 | P.42 gate |
| worst state metric share of limit | 0.42 (topk_idx L24) | | P.42 A/B |

- Final hidden error of the P.42 tree itself was not measured: every layer error is identical to P.41's (P.42 agent).
- The quick A/B's "min layer PCC" (~0.75 at L39) is not a regression signal: L39 is the noisy last layer.
- Full 56k prefill: 2,503 ms on the rebased build (X.3 re-run, before P.34, mmap engram). Not re-measured since.

## 5. Setup on a new Galaxy

Everything under `/scratch` is node-local. A new box has none of it. The shared masters are on NFS `/data` and are
readable by other users.

```bash
# 1. Checkout (origin = github.com/tenstorrent/tt-metal); steps 1-3 were not re-run for this doc (unverified as a sequence)
git clone https://github.com/tenstorrent/tt-metal.git && cd tt-metal
git checkout dnijemcevic/model_bringup_rebased
git submodule update --init --recursive
# tt_ops_code_gen has update=none in .gitmodules. Without it the fork tests that import `eval` fail (the root symlink eval -> it)
git -c submodule.tt_metal/third_party/tt_ops_code_gen.update=checkout submodule update --init tt_metal/third_party/tt_ops_code_gen

# 2. Build: Release, no tests (the build_metal.sh defaults). Afterwards `build` must be a symlink to build_Release.
./build_metal.sh
ls -l build            # build -> build_Release
# After a C++ change to a bring-up fork: cmake --build build --target install (Python loads the installed copies)

# 3. venv (./python_env). If ~/.cache/uv is damaged, put uv's cache on /scratch (UV_CACHE_DIR: uv's own variable, unverified here):
UV_CACHE_DIR=/scratch/$USER/uv-cache ./create_venv.sh

# 4. Every session, from the repo root
export PYTHONPATH=$PWD && source python_env/bin/activate
S=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
```

The data paths come from the spec. `paths.art` is `/scratch/$USER/bringup`, and the framework appends the model
name (`core/spec.py:art`):

| What | Where | Size | How it gets there |
|---|---|---|---|
| HF checkpoint, rev `5254448103b6254fc646dacbca1cbbf20e9216c1`, all 48 shards | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/hf_full` | 476 GiB | You download it once (below). Dense weights and the engram tables are always read from it |
| Goldens (master) | `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4` (s10240_c5120, s56320_c5120, plus `-official`) | 85 GB | Fixed |
| Goldens (node-local mirror tests read) | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4` | 85 GB | `reference.golden sync`. The orchestrator syncs at run/resume start. When the mirror is incomplete, tests read NFS (slow) and print a warning |
| Routed-expert weight cache (master) | `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/tt_cache/8x4/v1` (bank-major `.bm.bin`, bfp4_mse) | 306 GB, 120 files | Fixed |
| Weight cache (node-local mirror the loader reads) | `/scratch/$USER/bringup/deepseek_v41_flash_d_p/tt_cache/8x4/v1` | 306 GB | `core.weight_cache pull`. The orchestrator pulls at run/resume start. A miss rebuilds that layer from the checkpoint (~2-3 min a layer, DS41-121) |
| MoE-main routed weights | beside the expert cache: `<expert cache>.moe_main.<plan>.{w_gu,w_d}.tensorbin` | ~285 GB (HANDOFF_2026-11-06) | Laid out from the checkpoint on first use of `MOE_BLOCK_SRC=main` (unverified: time) |

```bash
# checkpoint (the recipe used here: /scratch/dnijemcevic/hf_dl.py; ~7 min on this node's network (unverified elsewhere))
python -c "from huggingface_hub import snapshot_download as d; d('deepseek-ai/DeepSeek-V4.1-Flash', \
  revision='5254448103b6254fc646dacbca1cbbf20e9216c1', \
  local_dir='/scratch/$USER/bringup/deepseek_v41_flash_d_p/hf_full', max_workers=8)"

python -m models.demos.common.bringup.reference.golden status --spec $S      # which dir each rung reads
python -m models.demos.common.bringup.reference.golden sync   --spec $S      # master -> mirror, verified
python -m models.demos.common.bringup.core.weight_cache status --spec $S     # to pull / to push, with a time estimate
python -m models.demos.common.bringup.core.weight_cache pull   --spec $S     # ~50 min at the code's 100 MB/s NFS estimate
```

- The local disk needs about 870 GB for checkpoint, cache and goldens, plus the MoE-main weights.
- Old `.tensorbin` expert caches are not read any more. To convert one (host only, ~8 min):
  `python models/demos/deepseek_v41_flash_d_p/tools/convert_expert_cache.py --layers $(seq 0 39) [--verify]`.
- `ENGRAM_TABLE=resident` needs ~190-210 GB of host RAM for the two engram tables (the Galaxy host has 566 GB).
- `push` is for the master's owner only. A user who cannot write the master only pulls.

## 6. Run the model

The rules in section 11 apply to every command below. Each device test loads the model: about 1.5-3.5 min with a
warm weight cache (measured model_load_s 82 s and 205 s), plus about 1.5-2.5 min of first-chunk compile.

```bash
# "What is the capital of France?" -> Paris, on the device model (no server)
BRINGUP_SPEC=$S scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_smoke.py

# accuracy: ladder rung `last` (golden prefix 0-51200 loaded, last 5120-token chunk on device, every metric vs the golden)
BRINGUP_SPEC=$S BRINGUP_RESULTS_DIR=/scratch/$USER/ds41_results BRINGUP_TASK=mine BRINGUP_RUNG=last \
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py

# the full 56k prefill, all 11 chunks on device, ~8 min (X.3: 492 s)
BRINGUP_SPEC=$S DS41_ENGRAM_TABLE=resident BRINGUP_RUNG=s56320 \
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py

# a short turn under the decoder skip (in every recent gate)
BRINGUP_SPEC=$S DS41_DECODER_SKIP=replay128 DS41_ENGRAM_TABLE=resident \
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/bringup/test_short_turn.py
```

**Serving runner (tt-d-gen).** The K.1R serving-contract tests drive the model with the real tt-d-gen engine.

- Branch for serving (owner, 2026-11-07 00:15): `jazpur/dsv4-replay-window`, checked out at `48b79127f`
  ("Recurrent handoff at #2384's prefill stop, and a DSv4 P2D snapshot test"; main d8a194417 + 10 commits, tt-blaze pin
  3cefbb45). Clone: `/scratch/dnijemcevic/tt-d-gen` (= the spec's `serving.server_repo`, `/scratch/$USER/tt-d-gen`).
- Built at `/scratch/dnijemcevic/tt-d-gen-build`; the engine's Python is `venv312/bin/python`
  (`testing/dgen_engine.py:find_build`). `BRINGUP_SERVER_REPO=<path>` points at another checkout.
- How to build it: `/scratch/dnijemcevic/tt-d-gen-build/README.md` (this node only). Short version:
  `./build_dgen.sh --bindings --blaze --no-kvm-client`. That build fails at tt-blaze's BFP packer because clang-20
  here has no OpenMP. Build the packer with `CC=g++-12 CXX=g++-12 LDSHARED="g++-12 -shared"`, then re-run with
  `--skip-blaze-build`. Or let the `dgen-build` agent (`models/demos/common/bringup/agents/dgen-build.md`) do it.
- Check the build: `BRINGUP_SPEC=$S python -m models.demos.common.bringup.testing.dgen_engine find`.
- `test_runner_contract` against this branch: PASS (2026-11-07 00:48, and again in the DS41-216 fix round).

```bash
T=models/demos/deepseek_v41_flash_d_p/tests/bringup/contract
BRINGUP_SPEC=$S scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_contract.py
for t in test_kv_layout test_adapter_acks test_runner_contract test_runner_smoke; do   # smoke ~12 min: Paris via tt-d-gen
  BRINGUP_SPEC=$S scripts/run_safe_pytest.sh --run-all --no-precompile $T/$t.py || break; done
```

- `test_contract` passes on the P.43 tree (acks_early 0, contract_checks_failed 0). On P.42 it failed with 40 early
  acks (DS41-218: the check was stricter than tt-d-gen for Snapshot configs); P.43 fixed the check.

## 7. Accuracy

There is no fixed bar. Every gated metric has a limit of `e2e.calib` (2.0) x the official reference's own drift
from full precision. The golden is the fp32 reference. The drift comes from the bf16/fp8/fp4 reference
(`<rung>-official`, `drift-<rung>.json` beside the golden). The code is in `testing/e2e_limits.py` and
`reference/drift.py`.

The ladder (`testing/ladder.py`) writes `$BRINGUP_RESULTS_DIR/$BRINGUP_TASK.json`, by default
`generated/bringup_adhoc/adhoc.json`. Every metric has `{t, value}`, and every gated metric has a `<name>_limit` next
to it. The test fails if any gated metric is over its limit, unless an owner waiver (`tests.waivers`) covers it.

| Metric | Meaning | Latest |
|---|---|---|
| `err_layer_L00..L39` | per-layer output rel L2 (`pcc_layer_*` is diagnostic only) | P.42: 0.019 (L0) to 0.140 (L18), about half of each limit; L39 0.707 / 1.533 |
| `err_state_<cache>_L*`, `overlap_state_*` | the served caches: window_kv, compress_kv, index_k, topk_idx (overlap), candidates, engram_tail | all within limits; worst share 0.42 (P.42) |
| `err_final_hidden` | final norm output rel L2 | 0.1313 / limit 0.242 (P.40 / P.41). X.3: 0.145 / 0.259 |
| `top5_overlap`, `top1_match`, `pcc_logits_tail` | token-level agreement of the last rows' logits | 1.0, 1.0, ~0.993 (X.3) |

- Rungs: `s10240` and `seed` (10k), `last` (golden 51k prefix + 1 chunk, the fast check), `s56320` (the full 56k).
- The quick A/B reports layer and state metrics only; final hidden and top-5 come from a ladder run.
- Token-level study: `tests/bringup/test_logit_kl.py` (FIX.20). It still asserts the old window format `bfp8_tiles`
  and fails under P.38's bf16 default (P.38 agent; not in any gate; owner's call).
- The precision policy is realistic precision only: there is never a max-precision device run. Its rationale is in
  `models/demos/common/bringup/README.md` ("Switches") and in the spec's `agents.rules`.

## 8. The PD serving hand-off (prefill → decode)

**The idea.** Prefill (this model, tt-metal) computes the prompt's caches. tt-d-gen's KV Manager copies ("migrates")
them to the decode machine (tt-blaze), which generates tokens. Both sides publish a KV address table; tt-d-gen moves
the records between them.

| Document | Where | What |
|---|---|---|
| Serving contract | `models/demos/deepseek_v41_flash_d_p/bringup/serving_contract.md` (repo) | How tt-d-gen drives prefill: input, chunk layout, served caches, acks, P.39 / P.42 changes |
| **Migration configs** | `models/demos/deepseek_v41_flash_d_p/bringup/migration_configs.md` (repo) | The table as the code serves it: per config the layers, record bytes, entry counts, cap rule, fp8 row, switches, open points |
| Earlier draft (v2) | `/scratch/dnijemcevic/rebase/dgen_migration_table_draft.md` (scratch) | Superseded by the file above |
| Decode's review | Brian Liu, #blaze-deepseek-v4, 2026-11-06, https://claude.ai/artifact/HG8kwWMJedV7TZsJtPMjgk | Asked for the 128-row snapshot at cap, Blaze fp8 index rows, `alias_rows` fan-out |
| Kartik's proposed final contract | #blaze-deepseek-v4 thread 2026-11-06 18:13 CET; local copy `/scratch/dnijemcevic/DSv4.1_final_tt-d-gen_bounded-replay_contract.html` (scratch) | Bounded replay driven by X / N metadata (below) |
| tt-d-gen tables mock | `/scratch/dnijemcevic/tt-d-gen/engine/tests/integration/dsv4_tables.py` (tt-d-gen clone) | DSv4-shaped mock tables: `swa` as a 128-position snapshot, the P2D byte check |

**What we serve today** (P.42 defaults; details in migration_configs.md):

| Config | Layers | Record bytes | Type |
|---|---|---|---|
| `0_window_kv` | 0-39 | 32,768 (32 bf16 rows of 512) | Snapshot: 128 rows, p at row p % 128, p in [cap - 128, cap) |
| `1_compress_kv_r2` | 2, 8, 14 | 16,384 | History [0, cap) |
| `2_index_k_r2` | 2, 8, 14 | 3,072 (16 fp8 rows of 192 B) | History [0, cap) |
| `3_compress_kv_r1` | 20 | 32,768 | History [0, cap) |
| `4_index_k_r1` | 20 (+ 24, 28, 32, 36 aliases) | 6,144 | History [0, cap) |

- cap = (prompt_len - 1) // `KV_BLOCK_SIZE` x `KV_BLOCK_SIZE` (64). Decode receives [0, cap) and replays
  [cap, prompt_len). Prefill still computes and attends every position up to the prompt's end.
- Per slot at 1,048,576 tokens: 3,044.81 MiB of distinct bytes to migrate; at 56,320: 168.10 MiB.

**Kartik Paigwar's proposed contract vs what we serve.**

| Topic | Kartik's contract | What we serve |
|---|---|---|
| Cut point | C = max(N - 128 - N % 2, 0): even, 128 short of the prompt length N | cap = (N - 1) // 64 x 64: 64-aligned; decode replays [cap, N) |
| Window rings | Only encoder layers 0-19 transfer rings; decoder SWA starts at D on decode | A 128-row snapshot for all 40 layers |
| Prefill's work | Layers 0-19 + layer 20's KV head only (never the full decoder) | All 40 layers on every chunk (`DECODER_SKIP=off`); `exact` / `replay128` exist behind the switch and need a last-chunk flag. P.33 showed prefill to an even cut 128 short of the end is within limits (2026-11-05) |
| Metadata | New X (matched prefix length) and N (prompt length): PrefillMetadata 12 → 16 B, BlazeMetadata 384 → 448 B; host planner derives scenarios A / B / C / A0 | tt-d-gen's 12-byte header `{slot_id, actual_start, actual_end}`; no X, N or last flag |
| Ring indexing | Local q = p - E (encoder origin), row = q % 128 | Absolute row = p % 128 |
| Migration range | P migrates only new rows [X, C); decode loads [0, X) from the store | tt-d-gen's existing rule: decode announces what it lacks, prefill sends it |
| Status | DTO/codec prototype and simulator verified; runtime/device integration pending | Implemented, tested against tt-d-gen 48b79127f with mock migration |

**Jonathan Azpur's 5 objections** (same thread, 2026-11-06 23:58 CET):
1. Model-aware readers: keep model logic out of tt-d-gen; the device can report skipped layers as finished.
2. Agreeing X before prefill starts adds a decode round trip to every request's TTFT; indexing rings by absolute
   position (row = p % 128) would remove the need.
3. Migrating only [X, C) and loading [0, X) from the store adds a store dependency for no gain; tt-d-gen already
   sends only what decode lacks.
4. Gating decode until the hand-off is in is not needed: decode already waits for its migration (the snapshot
   arrives with the last chunk).
5. Bypassing prefill in scenario C: the contract does not say who decides; in tt-d-gen decode decides through its
   decline path.

**Open questions** (owner / decode / tt-d-gen):

| Question | State |
|---|---|
| fp8 quantizer rounding: decode's review says round-to-nearest-even, byte for byte with `pack_index_keys`; P.39 read Blaze's writer as truncating and matches the stock `per_token_cast_to_fp8` (DS41-213) | Open |
| cap from the prompt length vs where prefill stops (DS41-219): tt-d-gen stops PD prefill at prompt_len - 1; differs when prompt_len % 64 == 1 (2,945: ours 2,880, decode's 2,944) | Open, ask decode |
| Last-chunk flag / N for prefill: without it the snapshot is rewritten every chunk (~4.6 ms device a chunk) and the decoder skip cannot act | Owner raising with tt-d-gen |
| Single-turn: a 128-row snapshot at cap cannot serve a follow-up past cap | Open |
| Engram at resume: decode's first tail step needs token ids before cap | Open: is Engram in the first serving version? |
| Device groups: index rows on 4 chips, snapshot on 32; dedup not decided | Open |
| Multi-slot caches on decode (`llong/dsv41-mu-main`) | Decode side, under review |
| Compressed-KV `alias_rows` for consumer layers (decode's review) vs ours (sources only) | **Unverified** what decode expects |
| Real KV Manager test: its kernel (`kv_manager/device_io/kernel/dmk.cpp`) uses the same 4 DRISC tiles a chip as the old `flat_drisc` op; `MOE_BLOCK_SRC=main` avoids it; our runner tests use mock migration | Owner: "not yet" |

## 9. Known issues found (2026-11-05 to 2026-11-07)

Exact entries: `models/demos/deepseek_v41_flash_d_p/bringup/known_issues.md`.

| Id | What | State |
|---|---|---|
| DS41-216 | Stale gather rows: one `high_bw_all_gather` semaphore pair was shared by both mesh axes. TP and SP link workers share cores on 22 of 32 chips, so one axis's counts were taken by the other's call; whole TP columns got wrong rows, even after a full sync | Fixed in `902322e6f77`: one pair per axis (`tt/ccl._sems(mesh, axis)`). Stress tests `tests/test_gather_stress.py`, `tests/test_window_gather_stress.py` (skewed chips, 512 iterations): 2,048 wrong replicas → 0. Notes: `/scratch/dnijemcevic/rebase/ds41_216/NOTES.md` Upstream doc request: https://github.com/tenstorrent/tt-metal/issues/64805 (documentation fix PR, reviewer Marko Staletovic). |
| DS41-220 | Stock `ttnn.all_gather` over the whole mesh (cluster_axis None) of a ROW_MAJOR tensor returns before every row has landed (TILE is fine) | Upstream https://github.com/tenstorrent/tt-metal/issues/64801, assigned to Sheran Cardoza (stock CCL owner, ops-data-movement). Standalone repro `/scratch/dnijemcevic/rebase/ds41_216/test_all_gather_mesh_rm_repro.py`. The model never does this call |
| DS41-217 | `slice_write` refuses a sharded output | Worked around: `update_padded_kv_cache` at start 0 |
| DS41-218 | Framework contract check stricter than tt-d-gen for Snapshot configs (acks_early 40) | Fixed in P.43 |
| DS41-221 | Under `DECODER_SKIP=replay128`, layers 21-39's snapshot rows are 0.97-2.2x their limit (the tail model's window starts empty; same effect as DS41-207) | By design; recorded, not gated. `exact` is within limits |
| DS41-219 | tt-d-gen PD prefill stops at prompt_len - 1; cap may differ from decode's | Open (section 8) |
| DS41-211..215 | Main block's shared expert ran bf16 (fixed P.37); bf16 table + per-chip window needs the sdpa fork's `kv_window` (P.38); stock fp8 cast vs Blaze recipe (P.39); reading fp8 bytes (P.39); bfp8 copy at non-aligned starts (P.41) | Fixed / documented |
| SFPI ICE | sparse_sdpa row_bfp4 at H=32 does not build: SFPI 7.90-7.93 internal compiler error (tenstorrent/tt-metal#64605). H=64 (the 8x4) builds. Owner-waived in the spec's `fork_waivers` | Report draft `/scratch/dnijemcevic/rebase/sfpi_ice_report.md` (not filed); reproducer `ttnn/ttnn/bringup/sdpa/docs/sfpi_ice/` |
| tt-kmd leak | tt-kmd 2.11.0 leaks host memory (#301); ~320 GiB unaccounted before the 2026-11-06 23:45 UTC reboot, 6 test units OOM-killed. After the reboot the driver was **the same 2.11.0** (fix not loaded) | Watch `free -g`; only a reboot clears it |

**Gather-path count** (2026-11-07, one profile run at 51,200, `/scratch/dnijemcevic/rebase/gather_count/counts.json`):
every model gather goes through `ttnn.experimental.high_bw_all_gather` (368 calls a chunk, all from `tt/ccl.py:270`);
stock `ttnn.all_gather` is called 0 times on default settings. Direct stock calls exist only on non-default paths:
`tt/weighted_combine.py:417` (the flat-MoE combine, called from `tt/experts.py`; unverified which switch values reach it), `tt/moe_block_main_adapter.py:411` (with `x_dram`), and
`tt/ccl.py:261` (`CCL_IMPL=stock`).

## 10. Operating the bring-up

**Read first:** `models/demos/common/bringup/README.md` (how-to, task ids, gates, exit codes). Skills:
`models/demos/common/bringup/skill/{bringup,rebase-bringup,bringup-fork-op,bringup-fork-tests}/SKILL.md`; link them into
`~/.claude/skills/`. Agent definitions: `agents/`; link them into `~/.claude/agents/`.

**The ledger** (`models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/`, in the repo):

| File | What |
|---|---|
| `tasks.yaml` | each task's brief, allowed `paths`, `gate.cmd` + `metrics`, A/B env (`ab.env` off side, `ab.change` on side) |
| `approvals.yaml` | the owner's approvals; any edit to what was approved voids it |
| `state.json`, `state_metrics.json` | verdicts and metrics (F253: metrics split out) |
| `results/<task>.json` | the gate's metrics |
| `supervision.md` | the overseer's log, newest at the end. Read the entries from 2026-11-05 on |
| `spec.yaml` | box, paths, limits, waivers, serving |

**The orchestrator** (runs agents, measures A/B, runs gates, commits):

```bash
B="python -m models.demos.common.bringup"; O="python -m models.demos.common.bringup.orchestrator"
$B status --spec $S                         # the ledger
$O run|resume --spec $S [--until <task>]    # resume after a stop; --until runs a task and its deps, then stops
$O pause --spec $S                          # stop before the next task
$O decide --task P.42 --accept|--reject|--rerun --spec $S   # a perf pick, from its A/B table
$B gate <task> --spec $S [--unit U | --continue | --status | --no-commit | --commit-only]
$B rerun --from <task> --spec $S            # reset it and everything downstream, then sweep
```

- The run is started by `/scratch/dnijemcevic/run_ds41_8x4_rebased.sh` (last used `resume --until P.43`) in a tmux
  session; it sources `/scratch/dnijemcevic/node_env.sh` and works in `/scratch/dnijemcevic/tt-metal-rebase`.
- **A/B accept flow (perf picks, P.\*).** (1) An agent makes the change behind a `tt/settings.py` switch. (2) The
  orchestrator measures off and on, one process a side: a quick run (last chunk after the golden prefix, accuracy vs
  limits) and p51200 (device / host time). (3) It writes `ab/<task>/table.md` and stops with exit code 3 (waiting for
  a person). (4) The owner runs `decide --accept|--reject|--rerun`. (5) An accept commits with the quick gate; the
  full gate stays `full_pending` and runs before the final task. Exit codes: 0 done, 3 waiting, 1 stopped.
- **`gate` commits a PASS by default** (F239). A gate never restarts: each test segment is a unit, kept until the
  code under test changes.
- **Logs** (run dir `/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/`):
  `orchestrator.log`, `logs/<task>.log` and `<task>.units.log`, `agents/<task>.<role>.<n>.jsonl` (large: extract the
  last text message with a script, never cat them), `briefs/`, `ab/<task>/{table.md,new,old}`.
- **Never edit `tasks.yaml` or `approvals.yaml` while a task runs.** The orchestrator's path check blames the running
  agent for any change outside its paths (2026-11-07 11:36: queuing P.43 during P.42 tripped it). Queue new tasks at a
  decision pause.

## 11. Rules and pitfalls

**Device**
- Before every device run check that nobody holds a chip. `fuser /dev/shm/TT_UMD_LOCK*` prints nothing when free.
  A finer check (who holds which chip):
  `for f in /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_*_PCIe; do python3 -c "import struct,sys,os; p=struct.unpack_from('<i',open(sys.argv[1],'rb').read(),52)[0]; p and os.path.exists(f'/proc/{p}') and print(sys.argv[1], 'held by pid', p)" "$f"; done`
  Also glance at other users' processes (`ps`).
- Run on the device only through `scripts/run_safe_pytest.sh` (`--run-all --no-precompile` for model tests). One
  device job at a time, and never next to a running orchestrator.
- Never wrap a device command in `timeout`. The safe runner detects hangs, collects triage and resets the box itself.
- Reset only with the safe runner's `recover-hosts` (`scripts/lib/tt_device_pool.sh`) or `tt-smi -glx_reset`. **Never
  `tt-smi -r`** on a Blackhole Galaxy.
- Never loosen a limit to make a test pass.
- If `/tmp/tt-device.lock` is a stale file of another user, point `TTPOOL_LEGACY_LOCK` / `TTPOOL_LEGACY_DIRTY` at
  files of your own. See `/scratch/dnijemcevic/node_env.sh`.

**Known failures and waivers**, all in the galaxy spec's `tests:` section:
- `unit_waivers`: `test_adapter_acks` accepts "last block: 0_window_kv layer N pcc 0.98x < 0.99" (tt-d-gen's fixed
  0.99 rule fails by rounding; the gated rel-L2 checks pass).
- `fork_waivers`: the 4 sparse_sdpa row_bfp4 H=32 cases (SFPI ICE, section 9). On a 2x2 QuietBox, DS41 at TP2 has 32
  heads a chip and hits this until there is a fixed SFPI.
- `waivers` (per step): attn_hc / ffn_hc, experts (bfp4), indexer, router. The reasons are written there.
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_hybrid_routed_expert_combine.py` hangs under the safe
  runner's precompile plugin. With `--no-precompile` it passes. Framework bug, not on DS41's path.
- The Galaxy cannot run 2x2 / 1x4 submesh fork cases (fabric init times out, or skipped). Those go to the QuietBox;
  list in `bringup/REBASE_STATUS.md` ("For the QuietBox").

**Host**
- tt-kmd 2.11.0 leaks host memory (section 9). Check `free -g` before resident-engram runs; ~500 GB available after a
  reboot.
- A test at its peak plus the agents can exceed a low `ulimit -u` (512 on an earlier node).

**Before changing anything, read** `bringup/known_issues.md` (DS41-n entries; shared ones in
`models/demos/common/bringup/knowledge/known_issues.md`), `bringup/repo_map.md` and the latest sections of
`bringup/BREADCRUMBS.md` (append-only, one section per task attempt).

## 12. Where things are

"Repo" = in git on this branch. "Scratch" = on the Galaxy node's shared `/scratch` (bh-glx node, not in git; a new
box has none of it). "/data" = NFS, shared, not in git.

| Path (repo paths relative to `models/demos/deepseek_v41_flash_d_p/` unless absolute) | Where | What |
|---|---|---|
| `DEEPSEEK41_README.md` (repo root) | repo | this file |
| `tt/model.py`, `tt/*.py` | repo | the all-device model (`TtDS41Model`) and its steps |
| `tt/settings.py` | repo | every switch, with its default and its why |
| `tt/ccl.py` | repo | collectives (per-axis high_bw semaphores, DS41-216) |
| `tt/runners/adapter.py`, `tt/runners/kv_contract.py` | repo | the tt-d-gen prefill adapter and the served KV table |
| `tt/moe_block_main_adapter.py`, `models/demos/moe_block_main/`, `ttnn/ttnn/bringup/moe_block_main_ttnn/` | repo | the main MoE block (`MOE_BLOCK_SRC=main`) |
| `reference/` | repo | CPU reference (`ds41_ref.py`, `hf_oracle.py`, `kernels.py`, `weights.py`) |
| `bringup/serving_contract.md` | repo | how tt-d-gen drives prefill |
| `bringup/migration_configs.md` | repo | the PD migration configs (v3) |
| `tests/bringup/contract/` (`caches.py`, `server_rules.py`, `test_*`) | repo | serving-contract tests (K.1R) |
| `tests/test_window_snapshot.py`, `tests/test_gather_stress.py`, `tests/test_window_gather_stress.py`, `tests/test_index_k_copy.py` | repo | P.42 snapshot, DS41-216 stress, P.41 bfp8 copy |
| `tests/bringup/test_c_*`, `test_swap_*` | repo | frozen per-step component and swap tests |
| `tests/bringup/test_decoder_cut.py`, `test_decoder_skip.py`, `test_window_ring.py`, `test_short_turn.py`, `test_logit_kl.py` | repo | decoder cut / skip, window ring, short turn, token-level accuracy |
| `tests/perf/`, `tools/convert_expert_cache.py` | repo | perf sweeps; .tensorbin -> bank-major expert cache |
| `bringup/ledgers/galaxy_8x4-full/` | repo | the 8x4 run: spec, tasks, approvals, state, results, supervision.md |
| `bringup/known_issues.md`, `repo_map.md`, `BREADCRUMBS.md`, `opportunities.md` | repo | knowledge files |
| `bringup/REBASE_STATUS.md` | repo | the 2026-11-06 rebase: verification, decisions, QuietBox list |
| `bringup/spec.yaml`, `bringup/tasks.yaml`, `bringup/plan.md`, `ENGRAM_HOST_PATH.md`, `tt/MOE_BLOCK.md` | repo | 2x2 seed run, sharding plan, engram host path, block MoE |
| `models/demos/common/bringup/` | repo | the framework: `README.md`, `defaults.yaml`, `orchestrator.py`, `tests/`, `testing/` (incl. `contract.py`), `docs/` |
| `ttnn/ttnn/bringup/` (`INDEX.md`) | repo | the forked ops (`ttnn.bringup.*`) |
| `scripts/run_safe_pytest.sh` | repo | the only way to run on the device |
| `/scratch/dnijemcevic/tt-metal-rebase` | scratch | the worktree the orchestrator runs in (do not edit while it runs) |
| `/scratch/dnijemcevic/run_ds41_8x4_rebased.sh`, `/scratch/dnijemcevic/node_env.sh` | scratch | run script; node environment |
| `/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/` | scratch | orchestrator.log, logs/, agents/, briefs/, ab/<task>/table.md |
| `/scratch/dnijemcevic/rebase/HANDOFF_2026-11-06.md` | scratch | hand-off before the 2026-11-06 reboot |
| `/scratch/dnijemcevic/rebase/ds41_216/` (`NOTES.md`, logs, repro) | scratch | DS41-216 / DS41-220 root cause |
| `/scratch/dnijemcevic/rebase/gather_count/counts.json` | scratch | gather-path count |
| `/scratch/dnijemcevic/rebase/sfpi_ice_report.md` | scratch | SFPI ICE report draft |
| `/scratch/dnijemcevic/rebase/dgen_migration_table_draft.md` | scratch | migration table v2 (superseded) |
| `/scratch/dnijemcevic/DSv4.1_final_tt-d-gen_bounded-replay_contract.html` | scratch | Kartik's proposed contract (local copy) |
| `/scratch/dnijemcevic/tt-d-gen` (branch `jazpur/dsv4-replay-window`), `/scratch/dnijemcevic/tt-d-gen-build` (`venv312`, `README.md`) | scratch | tt-d-gen clone and build |
| `/scratch/dnijemcevic/tt-d-gen/engine/tests/integration/dsv4_tables.py` | scratch (tt-d-gen) | the tt-d-gen DSv4 tables mock |
| `/scratch/$USER/bringup/deepseek_v41_flash_d_p/{hf_full,golden_full,tt_cache,runs,profiles}` | scratch | checkpoint, mirrors, run dirs |
| `/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/{golden_full/galaxy_8x4,tt_cache}` | /data | shared masters: goldens and weight cache |
