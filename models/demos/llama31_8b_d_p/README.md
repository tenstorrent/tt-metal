<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B-Instruct — TTNN disaggregated **prefill**

A clean, PCC-verified TTNN prefill implementation of Llama-3.1-8B-Instruct, built module-by-module
against a torch/HF reference with the collectives living **inside** the modules.

> **This file is a stub carrying P8's obligations only.** Phase P9 (cleanliness) owns the full
> README — the "why not `models/common/`" answer, the run instructions, the env-var table and the
> complete "not implemented" section. What is here is the status table `G-MESH-KV` requires
> (`BRINGUP_RECIPE.md:1804-1807`) plus the machine facts a reader needs to reproduce it. The
> authoritative record is [`bringup_log/`](bringup_log/): `06_GATES.md` for the numbers,
> `05_DECISIONS.md` for the reasoning, `07_RISKS.md` for what is still open.

## Status — full-model KV cache vs an fp32 torch golden

Measured on **one** Blackhole Galaxy (4x8, 32 devices), SP=4 x TP=8, `bfloat8_b` weights and cache,
`bfloat16` activations, 1024 real Llama-3.1-8B-Instruct tokens, **race-free** (3 runs in one process
on one `CCLManager` producing one hash):

| Run | attention core | min PCC across 32 layers (K / V) |
|---|---|---|
| one-shot, 1 chunk of 1024 | `sp_bootstrap` | **0.99880 / 0.99427** |
| chunked, 2 chunks of 512 (cache-read ring) | `sp_ring` | **0.99671 / 0.98682** |
| chunked, 4 chunks of 256 (cache-read ring) | `sp_ring` | **0.99678 / 0.98662** |

The golden is fp32 throughout and bit-identical to `LlamaModel`'s own layer loop
(`max|delta| = 0.0` on K, V and the post-norm hidden state over all 32 layers). The ring path carries
**2.74x** the one-shot path's K error, which is the cost of reading the prefix back out of a
`bfloat8_b` cache rather than attending live tensors.

Supporting single-configuration numbers, all from `bringup_log/06_GATES.md`:

| What | Measured |
|---|---|
| head -> mesh column mapping at TP=8 | **bit-exact**, 8/8 columns (`torch.equal`, `rtol=atol=0`) |
| the ring-joint SP attention core alone, vs fp32 torch | 0.99967, **6.05x** its own noise floor |
| ring vs one-shot at **layer 1** (the per-op claim) | K 0.99995 / V 0.99979 |
| every module, multi-device vs its own single-device output | 0.99997-1.00000 across `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)` |
| cache-only weight rebuild at TP=8 | 354 device shards, all SHA-256-identical |

Decode, performance work, trace/2CQ, quantised weights and multi-galaxy pipeline parallelism are
**not** part of this bring-up.

## The machine, and two facts that differ from the kit's description

- Compute grid **(12, 10)**, not 8x8. The CCL core range derives from it (offset x = 11); the SDPA
  *program* grid stays a pinned 8x8. The two look alike and must not be unified.
- **There is no ring fabric on this galaxy.** `ttnn.FabricConfig.FABRIC_1D_RING` cannot be
  initialised — the only single-galaxy RING/RING mesh-graph descriptor does not map to the discovered
  physical topology — and `ttnn.transformer.ring_joint_scaled_dot_product_attention` under
  `Topology.Ring` aborts for want of the SP axis's wrap route. Everything here therefore runs on
  `FABRIC_1D` with `ttnn.Topology.Linear`. `ttnn.Topology.Ring` *collectives* do work on `FABRIC_1D`
  (bit-exact at every shape), so the topology is decided by the ring SDPA alone. Details:
  `bringup_log/07_RISKS.md` R-030, `05_DECISIONS.md` DEC-079 / DEC-081. The in-repo M3 galaxy harness
  points at the same non-torus descriptor (`models/demos/minimax_m3/README.md:48`).
- **Submeshes, never a top-level partial mesh.** Opening `(1,8)` or `(2,8)` directly dies in fabric
  bring-up; open the full `(4,8)` and `create_submesh`. And **two overlapping submeshes with no
  `quiesce_devices()` between their phases hangs the machine** and poisons it until `tt-smi -r` —
  measured, 246 s to the timeout. `tests/test_factory.py::SubmeshPool` makes that unreachable through
  the API.

## Reproducing the status table

```bash
cd /home/mstojkovic/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
export HF_MODEL=/path/to/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=$HOME/.cache/llama31_8b_d_p          # never the checkpoint dir (DEC-048)
export PREFILL_TRACE_DIR=/path/to/golden/s1024

# the golden, once (host only, ~2 min, 0.25 GB)
python models/demos/llama31_8b_d_p/scripts/generate_golden_kv_cache.py \
    --tokens 1024 --out $PREFILL_TRACE_DIR --verify-loop

# the fabric map — run this FIRST on a new machine; it resets the box after the hang case
python models/demos/llama31_8b_d_p/tests/fabric_topology_matrix.py

# the status table
python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py                          # one-shot
PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=512 python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py
PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=256 python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py
PREFILL_RACE_ITERS=3 PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=512 \
    python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py                      # G-RACE

# the gate ladder
pytest models/demos/llama31_8b_d_p/tests -q
```

### Environment variables this package reads

| Variable | Owner | Meaning |
|---|---|---|
| `HF_MODEL` | shared | checkpoint directory (weights only — never the weight cache) |
| `TT_CACHE_PATH` | shared | tilized-weight cache root. The dtype **and** the mesh shape go into the path (`DEC-048`) |
| `PREFILL_TRACE_DIR` | the prefill engine | golden KV trace directory |
| `PREFILL_CHUNKED`, `PREFILL_CHUNK_SIZE`, `PREFILL_NUM_LAYERS` | the prefill engine | `tests/galaxy_prefill_kv_pcc.py` |
| `PREFILL_RACE_ITERS`, `PREFILL_KV_PCC_MIN_K`, `PREFILL_KV_PCC_MIN_V` | this package | `G-RACE` and the optional gate thresholds |
| `PREFILL_TOPOLOGY` | this package (`DEC-027`) | `linear` (default here) or `ring` — also selects the fabric config, deliberately one knob |
| `PREFILL_FABRIC` | this package (`DEC-079`) | `1d` (default here) or `1d_ring`; needs the torus descriptor and does not work on this galaxy |
| `TT_MESH_GRAPH_DESC_PATH` | tt-metal | required only for `PREFILL_FABRIC=1d_ring` |
| `LLAMA_DELTA_PROBE` | this package (`DEC-023`) | per-layer residual-delta statistics, for localising a drifting sublayer |

## Not implemented (P8 scope)

- **decode**, performance work, trace/2CQ, quantised weights — explicit non-goals.
- **multi-galaxy / multi-rank pipelined prefill.** Out of scope by instruction; the runtime *raises*
  on `set_layer_completion_sink`, on a non-first rank's `compile()`, and on the multi-rank half of
  the migration table rather than guessing (`bringup_log/07_RISKS.md` R-032).
- **the disaggregated-prefill engine integration** (adapter, request mode, KV migration) — phase P10.
  The six engine hooks are present and every one raises, naming its owning phase (`R-024`).
- **the deployment chunk/cache pair** (8192 / 131072) has never been run; every measurement above
  used a 1024-token cache (`R-039`).
- **residual scheme B** (a TP-sharded residual stream). The seams are wired and **refuse**
  (`DEC-025`, `DEC-038`, `DEC-041`); scheme A is what ships.
