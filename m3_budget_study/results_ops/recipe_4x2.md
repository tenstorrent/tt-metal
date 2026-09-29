# MiniMax-M3 prefill on a (4,2) sub-mesh: layers 0-6 zone profile + KV PCC

SP=4 (axis 0, rows) x TP=2 (axis 1, cols), EP=8, 128/8 = 16 experts per chip. The (4,2) is carved out of
the opened 8x4 galaxy in one process. Harness: `m3_budget_study/results_ops/tools/profile_4x2.py`. It wraps
`tests/perf/profile_prefill.py` (plan / load_tokens / build_runtime) and monkeypatches two things for TP=2:
the KV cache gets 2 K/V heads per chip, and the MSA cache read gathers that multi-head slot one head at a time.

## How 4x2 ran before

- philei's harness `/data/philei/scripts/m3_sptp/profile_prefill_sptp.py` (PROFILE_MESH=SPxTP), run from
  philei's own tree `/data/philei/tt-metal` with `TT_CACHE_PATH=/data/philei/models/m3_profile_cache`
  (`sweep.sh`). Only layers **0 and 3** (LAYER_IDS=0,3) were run, as one 5k chunk at a 51200-token cache on 1D fabric
  with bf4 experts.
- Result (`/data/philei/tmp/m3_sptp/artifact_data.json`, run `20260918_131219`): device 32.6 ms, wall 36.0 ms,
  warmup 120 s, prefix 69 s. PR #57199's "4x2 pass, 35.8 ms" row is a re-run of this same setup.
- Before that, three runs failed. Each failure explains one of the harness patches:
  1. `update_padded_kv_cache ... cache_shape[1] == input_shape[1]`: the stock cache has 1 K/V head per chip.
  2. `high_bw_all_gather selected-batch path requires singleton dimensions between batch and dim`: this is
     the MSA cache read on a 2-head slot.
  3. `buffer.cpp:259 core.y == 0`: a slice was written into the cache's ND-shard spec. The fix is to slice to DRAM interleaved.
- `/data/philei/models/m3_profile_cache/tensor_cache_bfp8_MeshShape([4, 2])` holds 6.9 GB: embed + layers 0 and 3,
  with bf4 experts `local_0..15`. There is no final norm / lm_head. It is read-only for us and **lacks layers 1, 2, 4, 5 and 6**.
- Weka `/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/` has [2,4], [4,4] and [8,4] caches but **no [4,2]**.
  It is writable for us (group cache-writers).
- No philei/* branch carries the SPxTP harness. `philei/m3-traffic-sim` is only `traffic_sim/`.

## Common env

```bash
cd ~/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/mnt/weka/model-weights/llm/minimax/MiniMax-M3
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill   # [4,2] gets created here
export M3_FABRIC=1d EXPERT_DTYPE=bf4 LOGURU_LEVEL=INFO
export PROFILE_MESH=4x2 PROFILE_STAGE=0 PROFILE_NUM_LAYERS=7 PROFILE_CHUNK=5120   # stage 0 = rows 0-3, cols 0-1; layers [0,15)
G=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden
H=m3_budget_study/results_ops/tools/profile_4x2.py
R=m3_budget_study/results_ops
ulimit -Su "$(ulimit -Hu)"
# preflight: build status exit=0; no sibling session on the galaxy (ps -u $USER | grep -E 'python|tracy'); git status
env -u TT_VISIBLE_DEVICES tt-smi -glx_reset
```

`create_submeshes(MeshShape(4,2))` tiles the 8x4 grid in row-major step order (`mesh_device.cpp:794-821`):
[0] = rows 0-3 / cols 0-1, [1] = rows 0-3 / cols 2-3, [2] = rows 4-7 / cols 0-1, [3] = rows 4-7 / cols 2-3.
A (4,2) is therefore 4 rows x 2 cols of the grid. That block is one BH tray.

## 1. Populate the cache for layers 0-6 and run the KV PCC against the golden (no tracy)

`PROFILE_NUM_LAYERS=7` sets `M3_LOAD_NLAYERS=7 M3_LOAD_LAYER_START=0`. `weight_cache_is_complete` is false on the
first run, so the bf16 shards for layers 0-6 are read and the tilized `tensor_cache_bfp8_MeshShape([4, 2])` is written.
This takes about 20-30 min. The estimate comes from philei's populate run for layers 0+3: 4.8 min, mostly about 4 min of
expert conversion per sparse layer. Here there are 4 sparse and 3 dense layers. Later runs load from the cache in 1-2 min.

```bash
M3_PROFILE_ZONES=0 TT_METAL_DEVICE_PROFILER=0 PROFILE_CACHE=51200 \
PREFILL_TRACE_DIR=$G/longbook_56320 PROFILE_KV_PCC=1 PROFILE_KV_DUMP=$R/bench/kv_4x2_L0-6 \
  python3 $H 2>&1 | tee $R/logs/kv_4x2_L0-6.log
```

The golden has 55218 tokens. The harness tiles them to 56320 and compares the first 55218, which is valid
because a causal prefix does not depend on what follows. The run fails if the min PCC is below
`PROFILE_KV_PCC_MIN` (default 0.88).

## 2. Reference (2,4) dump and the 4x2-vs-2x4 comparison

The weka [2,4] cache is complete.

```bash
env -u TT_VISIBLE_DEVICES tt-smi -glx_reset
PROFILE_MESH=2x4 M3_PROFILE_ZONES=0 TT_METAL_DEVICE_PROFILER=0 PROFILE_CACHE=51200 \
PREFILL_TRACE_DIR=$G/longbook_56320 PROFILE_KV_PCC=1 PROFILE_KV_DUMP=$R/bench/kv_2x4_L0-6 \
  python3 $H 2>&1 | tee $R/logs/kv_2x4_L0-6.log
python3 $H --compare $R/bench/kv_4x2_L0-6 $R/bench/kv_2x4_L0-6     # host only, per-layer K / V / index_k PCC
```

## 3. Zone profile, layers 0-6, 5k chunk at a 51200 cache

```bash
env -u TT_VISIBLE_DEVICES tt-smi -glx_reset
T=/tmp/m3_prefill_perf_traces/synthetic_56320        # same tiled trace run_prefill_profile.sh makes
[ -f $T/metadata.json ] || { mkdir -p $T; python3 - <<'PY'
import json; s=json.load(open("/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden/longbook_qa_eng_prefill_56320_nopad/metadata.json"))["token_ids"]
json.dump({"token_ids":[s[i%len(s)] for i in range(56320)],"n_tokens":56320},open("/tmp/m3_prefill_perf_traces/synthetic_56320/metadata.json","w"))
PY
}
M3_PROFILE_ZONES=1 M3_PROFILE_LEVEL=2 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
PROFILE_CACHE=51200 PREFILL_TRACE_DIR=$T PROFILE_SKIP_COMPILE=1 \
  python3 -m tracy -v -r -p --child-functions "HWCommandQueue_write_buffer,HWCommandQueue_read_buffer,CompileProgram" \
  $H 2>&1 | grep -v '| *DEBUG *|' | tee $R/logs/prof_4x2_L0-6_c51200.log
CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
mkdir -p $R/profiles/4x2_L0-6_c51200 && cp "$CSV" $R/profiles/4x2_L0-6_c51200/
python3 models/demos/minimax_m3/tests/perf/parse_zone_perf.py "$CSV" --html $R/profiles/4x2_L0-6_c51200/zones.html
```

`PROFILE_SKIP_COMPILE=1` warms the profiled chunk twice after the prefix instead of sweeping all compile buckets
under tracy, which keeps the capture small. Leave the setting out for a like-for-like comparison with runs that did compile().
`ag_kv` contains two FINE sub-zones, `head_slice` and `head_concat`. Both are costs of the per-head workaround, so
subtract them before comparing ag_kv with TP=4. The `index_k` gather is unchanged.

## Alternative: open the (4,2) directly (untested)

`TT_VISIBLE_DEVICES=<the 8 PCIe ids of one tray>` plus a one-mesh MGD with `device_topology { dims: [4, 2] }` and
`host_topology [1,1]`, then open `MeshShape(4,2)` (`BUDGET_MESH` in budget_sweep.py is the prior art for a direct
open). The tray ids come from `build/test/tt_metal/tt_fabric/test_physical_discovery
--gtest_filter=*GenerateTrayToPCIeDeviceMapping*`, which needs `--build-tests`. Reset with `env -u TT_VISIBLE_DEVICES`.
The carve is the proven path.

## What profile_prefill.py itself would need (another agent owns it)

- A `PROFILE_MESH=SPxTP` knob that sets stages = (8/SP)*(4/TP) and carves `create_submeshes(MeshShape(SP,TP))[stage]`.
  Today `main()` hard-codes `stages in (1,2,4)` with `MeshShape(8 // stages, 4)`.
- A ceil per-stage split in `build_runtime()`, which asserts `total_layers % stages == 0`. 4x1 / 1x4 give 8
  stages, and 60 % 8 != 0.
- A multi-head KV alloc at the end of `build_runtime()` when `num_key_value_heads // TP > 1`, plus the MSA
  multi-head cache read. Both are the monkeypatches in profile_4x2.py, or the model fix.
- An optional `M3_FORCE_LOAD_WEIGHTS` lo..hi window for `PROFILE_LAYER_IDS`, which today forces a cache-only load.
- In run_prefill_profile.sh: `HARNESS=${HARNESS:-...}` at :86 so the wrapper can drive profile_4x2.py, and a
  cache-dir check and banner at :132 / :251 that stop assuming `(8/S, 4)`.

profile_4x2.py only calls `plan`, `load_tokens`, `_raise_nproc_limit` and
`build_runtime(mesh, chunk, total, num_layers_override, layer_ids, stages=, stage=)`, and reads the module
names `L1_SMALL_SIZE` and `fabric_config_from_env` / `set_fabric_config_from_env`. If those change, update the harness.

## TP=2 status in the model code (branch vmelnykov/m3_moe_fabric2d)

Works as is (generic in TP):
- Head sharding: `MeshConfig.shard_size` gives 32 q / 2 kv heads per chip (config.py:73-75, attention/prefill.py:232-233).
- The q/k/v weight split uses `torch.chunk(.., tp)` (attention/weights.py:114-118). The o_proj local hidden is 3072, which is tile-aligned.
- Index heads: index_q is column-parallel, so 4 index heads become 2 per chip, and `num_groups` = 2. indexer_score accepts G>1
  (indexer_score_device_operation.cpp:672-674).
- Dense ring_joint: the gather buffer shards the global n_kv=4 over the cols (ccl.py:150-158).
- MoE: dispatch_group_size = rows = 4 and 2 dispatch groups (deepseek_v3_d_p/tt/moe/init_helpers.py:44-60). That gives 16 experts
  per chip (mlp.py:155) and a reduce over axis 1 (tt_minimax_moe.py:237). dispatch v2 needs an axis-0 extent that is even and >= 4 (:264-266).
- RoPE is SP-only.
- The weight cache is keyed `tensor_cache_bfp8_{mesh.shape}` (model_config.py:230-235).
- `read_slot_kv` composes dims=(2,1) and so concatenates the heads over TP (tt_prefill_runtime.py:649).

Breaks:
- kv_cache.py:95-96 allocates one K/V head per chip.
- The MSA cache read (msa.py:352-356) uses the batch-select `high_bw_all_gather` on a `[B,1,seq,hd]` slot.
- The runner's KV migration table asserts `num_kv_heads == cols` (tt/runners/kv_chunk_table.py:96).

Untested:
- `num_groups>1` in every MSA unit test (all use 1). The unit tests only parametrize (8,4). MeshConfig only warns (config.py:46-50).

## KV-head sharding change: effort

- (a) Upstream the two monkeypatches: add `n_kv_local` to `allocate_kv_caches` plus per-head slice/gather/concat in
  `msa_sp_attention_cache_read`, and add (4,2) cases to test_kv_cache_write_vs_ref / test_msa_sp_cache_read_vs_ref /
  test_attention_chunked_vs_ref with num_groups=2. About 1-1.5 days.
- (b) Make it native: let `high_bw_all_gather`'s selected-batch path take a non-singleton head dim
  (high_bw_all_gather_device_operation.cpp:352). That removes the extra 2 x seq_local x hd slice+concat per K/V per
  sparse layer. About 2-3 days more.
- (c) Runner / migration: generalize kv_chunk_table (head h -> col h // (n_kv/tp), with a head offset inside the chip) and
  the M3 adapter `allocate_kv_cache` (adapters/minimax_m3.py:128). About 1-2 days, plus an agreement on the decode-side layout.

## Runner: 4 x (4,2) with PREFILL_SP=4 PREFILL_TP=2 (not ready)

- `prefill_runner.py:77-79` reads PREFILL_SP / PREFILL_TP generically. Missing pieces:
  1. A topology yaml, like `pipeline_prefill_request_intragalaxy_4rank.yaml` but with SP=4 TP=2.
  2. A 4-mesh [4,2] MGD. `tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_4x2_mesh_graph_descriptor.textproto`
     has links 0-1, 1-2, 2-3 and 0-3. The z-chain variants exist only for [2,4].
  3. Per-rank TT_VISIBLE_DEVICES, one tray each (rank->tray 1,3,4,2 in tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py:510).
     The ids come from test_physical_discovery.
  4. KV-head change (a)+(c).
  5. A full 60-layer [4,2] weight cache: about 57 sparse layers x ~4 min, split across the 4 ranks' 15-layer windows.
  6. A check that PREFILL_FABRIC_MODE=2d works on 4x2, which has not been validated.
