# Repo map

Pointers every agent reads before it searches the repo. Grouped by need. Agents add rows under "Proposed" at the end of
their step; a person moves them up at the next approval point. The map is only useful if it is short: one row per need,
the best starting point first.

| Need | Where it is |
|---|---|
| MoE, expert parallel (EP) | `models/demos/deepseek_v3_d_p/tt/moe/` (routing setup, dispatch, `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe`, combine, reduce). `models/demos/ernie45_d_p/tt/moe_unified.py` runs that pipeline on a 1xN mesh (dispatch group of 1 chip, needs the ttnn host patches on `dnijemcevic/ernie45_prefill`). `models/demos/gpt_oss_d_p/tt/moe/` for an external router. `models/demos/blackhole/qwen36/tt/moe/` for `sparse_matmul`. |
| MoE, tensor parallel, decode-style | `models/demos/gemma4/tt/moe.py`, `experts/`, `router.py`, `shared_mlp.py` (Gemma 4 26B-A4B on T3K 1x8 and Blackhole) |
| Chunked causal attention | `models/tt_transformers/tt/attention.py`; `models/demos/ernie45_d_p/tt/attention.py` (chunked SDPA, paged-shaped cache, SDPA presets); ring SDPA in `models/demos/gemma4_d_p/tt/attention/` |
| SDPA program config | `models/demos/ernie45_d_p/tt/attention.py:sdpa_settings` (config A: HiFi2, fp32 acc off, exp approx, q256/k512); `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` |
| Sliding-window + global attention, partial RoPE | `models/demos/gemma4/tt/attention/`, `models/demos/gemma4_d_p/tt/attention/` |
| RoPE, interleaved (Meta) order | `models/demos/ernie45_d_p/tt/ops.py` (rotary_embedding_llama + custom trans-mat) |
| Collectives on 1xN | `models/tt_transformers/tt/ccl.py`; sync `ttnn.all_gather`, `ttnn.reduce_scatter`, `ttnn.all_reduce` with `cluster_axis` |
| Prefill engine contract | `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`, `models/demos/common/prefill/adapter.py`, `models/demos/common/prefill/runners/prefill_producer.py`, `runners/prefill_runner.py` |
| GQA KV layout and address table | `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py`, `models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py`; ERNIE reuse in `models/demos/ernie45_d_p/tt/kv_contract.py` |
| Device safety and hang triage | `scripts/run_safe_pytest.sh`, `scripts/tt-probe.sh`, `generated/tt-triage/triage.txt` |
| Tensor caching | `tensor_caching_research.md` (repo root, untracked), `models/common/weight_cache.py`, `models/demos/deepseek_v3_b1/weights/cache/cache.py`; `ttnn.as_tensor(cache_file_name=...)` |
| Device profile without Tracy | `models/demos/common/bringup/testing/profiler.py` (`signpost`) |
| Reference plans (sharding) | Kimi K2.7 4x4 artifact https://claude.ai/artifact/4MC3c1hvkErCYxByJwdPZ9; `models/demos/ernie45_d_p/SHARDING.md` + `plan.yaml` shape in `models/demos/common/bringup/plan/memory.py` |
| A worked bring-up | `models/demos/ernie45_d_p/` (reference, tt, tests, bringup/ ledger, BREADCRUMBS.md) |

## Proposed
