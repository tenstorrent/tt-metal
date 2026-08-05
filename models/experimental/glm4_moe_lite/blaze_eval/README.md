# blaze_eval — archived tt-blaze benchmark scripts

These are **copies kept for provenance**, not runnable from this tree. They `import blaze`
and depend on tt-blaze's own pinned tt-metal (which carries the `-ftt-nttp` / `-ftt-constinit`
/ `-ftt-consteval` / `-ftt-no-dyninit` SFPI flags that tt-metal main lacks — see F9 in
[../BLAZE_EVALUATION.md](../BLAZE_EVALUATION.md)).

To run them, copy back into a tt-blaze checkout and use its environment:

```bash
cd /path/to/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
python -m pytest ab_oproj_v2_bench.py -q -p no:randomly -s
```

The profiler vars must be exported **before** `ttnn` is imported — under pytest that means
setting them in the shell, not in Python. Otherwise the bench silently returns no samples.

| file | what it does | result |
|---|---|---|
| `ab_oproj_v2_bench.py` | ttnn DRAM-sharded matmul vs blaze `DRAMStreamingMatmul` at GLM o_proj (5120x2048, bf8, bs=1) | **2.45x**, both sides PCC 0.9999 |
| `ab_rmsnorm_bench.py` | ttnn `rms_norm` vs blaze `RMSNorm` at hidden=2048 | 0.92x REGRESSION at equal precision |
| `ab_m32_check.py` | drives blaze's own `_run_and_compare` at m ∈ {1,8,32} | reproduces **F1**: m=32 gives PCC 0.0074 |
| `no_grid_guard_plugin.py` | pytest plugin stubbing `requires_grid_size` | diagnostic for **F7** |
| `remap_col12_plugin.py` | remaps GLM-5's column-12 cores to 11; truncates gate cores to `num_experts/32` | makes **F5** ops pass; guards against **F6**'s hang |
| `test_glm47_routed_expert_dims.py` | drives `glm_routed_expert` at GLM-4.7-Flash dims, substituting `GLM4_FLASH_BLAZE_CONFIG` (no plugin needed) | **F11**: still hangs — blocked, not passing. Belongs in `tests/blaze/glm5_1/` |
| `glm4_flash_blaze_config.py` | `GLM4_FLASH_BLAZE_CONFIG` — 2 gate cores, sender (11,9), empty shared-expert coords | constructs and passes all sanity checks |
| `glm5_routed_expert_gate_bias_dim_general.patch` | zero-pads the gate bias to a face instead of hardcoding 256 experts | **F10**, worth upstreaming |

These copies have been run through this repo's `black` (line wrapping only, no semantic
change), so they differ cosmetically from the originals in the tt-blaze checkout.

`glm4_flash.model_config.json` is GLM-4.7-Flash in blaze's model-config schema — the live copy
belongs at `blaze/models/glm4_flash/` inside a tt-blaze checkout.

Two plugins are diagnostics, not fixes: `no_grid_guard_plugin` bypasses a real precondition,
and `remap_col12_plugin` mutates a frozen `BlazeConfig`, which blaze explicitly forbids
(`__replace__` raises). The supported path is authoring a proper `GLM4_FLASH_BLAZE_CONFIG`.
