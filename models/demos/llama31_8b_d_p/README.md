<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B-Instruct — TTNN disaggregated **prefill**

A clean, PCC-verified TTNN **prefill** implementation of Llama-3.1-8B-Instruct for Tenstorrent
Blackhole, built module-by-module against a torch/HF reference with the collectives living
**inside** the modules, and plugged into the model-agnostic disaggregated-prefill engine
(`models/demos/common/prefill/`).

The product of prefill is the **KV cache**, so that is what every top-level number here measures.
Decode, performance work, trace/2CQ, quantised weights and multi-galaxy pipelining are explicit
non-goals — see [Not implemented](#not-implemented).

The authoritative record is [`bringup_log/`](bringup_log/): `06_GATES.md` for every number and its
raw log, `05_DECISIONS.md` for every judgement call, `07_RISKS.md` for what is still open. The
recipe this package was built to is `models/demos/common/bringup/BRINGUP_RECIPE.md`.

## Architecture

Every row read from the bundled `configs/Llama-3.1-8B-Instruct/config.json`, which is byte-identical
to the checkpoint's (`md5 3cd5831d379b509d53afade0e24c36e9`) and asserted so by a test. Full
provenance table: `bringup_log/00_MODEL_CARD.md` §2.

| Fact | Value |
|---|---|
| architecture | `LlamaForCausalLM`, 32 layers |
| hidden / FFN intermediate | 4096 / 14336 |
| attention | GQA, 32 Q heads / **8 KV heads** (group 4), `head_dim` 128 (derived), no bias |
| MLP | dense SwiGLU `down(silu(gate(x)) * up(x))` on **every** layer, no bias |
| norm | plain RMSNorm, eps 1e-05 — **no Gemma `+1` fold** |
| RoPE | θ = 500000.0, **full rotary**, `llama3` scaling (factor 8.0, low 1.0, high 4.0, orig_max_pos 8192) |
| max positions / vocab | 131072 / 128256 |
| embeddings | not tied — a separate `lm_head.weight` exists |
| dtypes | `bfloat8_b` weights and KV cache, `bfloat16` activations, fp32 reference throughout |

**What this model does *not* have**, because the two nearest in-repo templates do and copying them in
is the most likely source of wasted work: no MoE / router / expert parallelism, no attention sinks,
no sliding-window alternation, no QK-norm, no partial RoPE, no MLA, no biases anywhere. Llama 8B is
the simplest shape in this family.

## Deployment path (Blackhole Galaxy, 4×8)

`(mesh_shape, TP, SP) = ((4, 8), 8, 4)` — TP on the 8 columns, SP on the 4 rows. The arithmetic,
not a preference (`bringup_log/00_MODEL_CARD.md` §4, `04_CCL_PLAN.md` §1.1):

- **`TP == num_key_value_heads == 8` is an equality, not a bound.** The packed KV cache holds
  exactly one KV head per chip (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:95-99`), so at
  any smaller TP `update_padded_kv_cache` dies with `cache and input num-heads dim must match`, and
  at any larger TP the KV heads would have to be replicated. **Consequence for gate design:** a
  `(1,1)` KV gate tests the cache *primitive*, not the model → cache path, because `nkv = tp = 1` is
  a head count the deployment mesh never produces. `G-KV-TP8` is what closes that.
- TP divides both feature dims tile-aligned: `4096/8 = 512` and `14336/8 = 1792`, both multiples
  of 32.
- SP is the other axis, with `CHUNK_SIZE % (SP*32) == 0` and `MAX_SEQ_LEN % CHUNK_SIZE == 0`.
- **Collectives go on the TP axis only**, which is what makes every module SP-safe, and they live
  inside the modules — the decoder layer and the model never call one.
- Residual scheme **A** (replicated, full-embedding) ships; the scheme-B seams are wired and
  **refuse** (`DEC-025`, `DEC-038`, `DEC-041`).

## Status

### Full-model KV cache vs an fp32 torch golden — two independent readers

Measured on **one** Blackhole Galaxy (4×8, 32 devices), SP=4 × TP=8, `bfloat8_b` weights and cache,
`bfloat16` activations, 1024 real Llama-3.1-8B-Instruct tokens, **race-free** (3 runs in one process
on one `CCLManager` producing one hash).

| Run | attention core | reader | min PCC across 32 layers (K / V) |
|---|---|---|---|
| one-shot, 1 chunk of 1024 | `sp_bootstrap` | on-device (`G-MESH-KV`) | **0.99880 / 0.99427** |
| chunked, 2 chunks of 512 (cache-read ring) | `sp_ring` | on-device (`G-MESH-KV`) | **0.99671 / 0.98682** |
| chunked, 4 chunks of 256 (cache-read ring) | `sp_ring` | on-device (`G-MESH-KV`) | **0.99678 / 0.98662** |
| chunked, 4 chunks of 256, served through the engine | `sp_ring` | **device-less, second process** (`G-MOCK-MIG`) | **0.996784 / 0.986623** |

**The last row is the strongest single line of evidence in the package.** It is a different process,
a different position → address derivation and a different byte decoder, and it agrees with the
on-device chunk-256 row **to six decimals and on both argmin layers** (K worst at L22, V worst at
L28). The engine's own threshold for it is `PREFILL_STANDALONE_CHUNKED_PCC = 0.93`; the measured
minimum sits 5.6× inside that error budget. Agreeing on the *values* is good; agreeing on *which
layers are worst* is what makes the address table credible.

The golden is fp32 throughout and bit-identical to `LlamaModel`'s own layer loop (`max|delta| = 0.0`
on K, V and the post-norm hidden state over all 32 layers). The ring path carries **2.74×** the
one-shot path's K error, which is the cost of reading the prefix back out of a `bfloat8_b` cache
rather than attending live tensors.

### Per-module PCC, single card, against an fp32 torch reference

Every gate records its input distribution, its reference dtype policy, a **computed noise floor**,
the error ratio to that floor, and a **negative control**. The ratio is the number that matters: an
absolute PCC clearing a threshold while sitting far off the floor is a finding, not a pass.

| Module | Gate | PCC (bf8_b weights) | ratio to floor | negative control |
|---|---|---|---|---|
| RMSNorm | `G-RMS` | 0.9999971 (real layer-0 gain) | 2.11× | zero-gain probe → `max|out| = 0.0` |
| RoPE (llama3-scaled, Meta convention) | `G-ROPE` | 0.9999969 | 1.76× | HF-layout tensor into the Meta op → **0.01367** |
| dense SwiGLU MLP | `G-MLP` | 0.9999133 | 1.10× | SiLU on `up` instead of `gate` → **0.64715** |
| attention block (GQA + RoPE + causal SDPA + o_proj) | `G-ATTN` | 0.9997080 | 2.17× raw | Q/K without the Meta `reverse_permute` → **0.51174** |
| KV cache primitive | `G-KV` | 0.9999743 | 1.00× | 128 positions × 4 offsets asserted **bit-exact** |
| decoder layer | `G-LAYER` | 0.9998273 | 1.65× | swapped norm gains → **0.66830** at real embedding scale |
| full 32-layer stack | `G-MODEL` | 0.9984849 @ 512 | worst per-layer step 1.27× | rotated per-layer weights → **0.16180** |

Two of those numbers need their caveat stated rather than buried:

- **`G-ATTN` is `PASS-WITH-DEVIATION`** (`DEC-042`, `R-015`). The fused
  `ttnn.transformer.scaled_dot_product_attention` contributes a roughly *fixed absolute* error, so
  the block's error **ratio** gets worse as the storage dtype improves — 2.17× at bf8_b and 11.82×
  at bf16 for the same code. The 8× block budget is therefore unsatisfiable at bf16 by any correct
  implementation. What is portable is the **SDPA-attributed residual**: 1.10× / 1.01× / 0.70× at
  bf16, i.e. everything this package wrote sits at its floor. The fused kernel is measured
  standalone (52.8–55.0× its own floor) and kept as a permanent probe so its slack stays a named,
  tracked term.
- **`G-MODEL`'s absolute PCC is scoped to a depth.** 0.9997314 at 2 layers and 0.9984849 at 32, same
  model, same sequence length. Depth-varying quantities are gated on the per-layer *step*
  (≤ 4×, worst measured 1.27×) and top-1 agreement (374 == 374 tokens, 100% at every depth).

### Multi-device and integration

| What | Gate | Measured |
|---|---|---|
| head → mesh column mapping at TP=8 | `G-KV-TP8` | **bit-exact**, 8/8 columns (`torch.equal`, `rtol=atol=0`); rotated-column control scores PCC 0.99887 and fails `torch.equal` — which is why mappings are gated on bit-equality, never correlation |
| every module, multi-device vs its own single-device output | `G-TP-PARITY` | 0.99997–1.00000 across `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)`; control (reference rolled by one TP shard) **0.00307** |
| the ring-joint SP attention core alone, vs fp32 torch | `G-SP-RING` | 0.9996672, **6.05×** its own floor; `fp32_dest_acc_en=True` **refused** by the op, as it must be |
| ring vs one-shot at **layer 1** (the per-op claim) | `G-CHUNK-ATTN` | K 0.9999505 / V 0.9997938; worst gated per-layer step 2.14× against 4× |
| chunked ≡ one-shot for deltas 1–2, per layer | `G-CHUNK` | mutual **1.0000000** at 32/32 layers; delta 1 alone `torch.equal` |
| no semaphore races | `G-RACE` | 3 runs in one process on one `CCLManager` → **one** hash; equal to a separate process's too |
| CCL state allocated once | `G-SEMAPHORE` | 6 RS + 4 AG + 2 barrier + 2 ring-attention = **14**, unchanged after a real 32-layer forward; control (3 managers) → 42 |
| weight loading | `G-WEIGHTS` | **291/291** keys consumed, 0 missing, 0 unused; 12/12 device tensors bit-exact through transpose + Q/K Meta swizzle + dtype ladder |
| cache-only rebuild at TP=8 | `G-WEIGHTS` (P8 ext) | **354** device shards, all SHA-256-identical; 8 tensors genuinely sharded with 8 distinct shard hashes each |
| the KV migration address table alone | `G-KV-TABLE` | **2048 chunks bit-identical** over UMD read-back at two block-cyclic periods; 5 discriminating controls; protobuf round trip preserves every address |
| request-mode serving, end to end | `G-REQUEST` | 11/11 chunks at the gate geometry and 2/2 at the **real deployment geometry** (chunk 8192 / cache 131072), sentinel received, both processes rc 0 |
| the engine's adapter contract | `G-ADAPTER` | 0 abstract methods; 9/9 `model_config` constants equal `config.json`; adapter import **37.8 ms with 0 heavy modules**, **0.0155×** the reference adapter's (64.6× cheaper — `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:25` pulls `torch` and `ttnn` at module scope) |

`G-LOOPBACK` (the engine's real DRAM → transport → DRAM copy) is **out of scope** by `DEC-103`: it
needs the tt-llm-engine binaries, which are not in this tree, and it verifies the *engine's*
model-agnostic byte copy rather than this model. The residual gap it leaves is enumerated as
`R-043`, and the multi-rank path it would have exercised **raises** rather than guessing.

**Suite:** **247 passed, 0 failed** (22:39). **Citations:** 629/629 content-checked, 1225/1225 doc refs,
146/146 cited raw artefacts, and the recipe those refs point into is pinned by fingerprint.

## The machine, and three facts that differ from the recipe's description

- **Compute grid (12, 10), not 8×8.** The CCL core range derives from it (offset x = 11); the SDPA
  *program* grid stays a pinned 8×8 and asserts `sdpa_grid.x <= grid.x - 1` at construction. The two
  grids look alike and must not be unified — a derived SDPA grid passes every single-card gate and
  fails only at SP > 1.
- **There is no ring fabric on this galaxy** (`R-030`). `ttnn.FabricConfig.FABRIC_1D_RING` cannot be
  initialised at all — the only single-galaxy RING/RING mesh-graph descriptor does not map to the
  discovered physical topology — and `ttnn.transformer.ring_joint_scaled_dot_product_attention`
  under `Topology.Ring` aborts for want of the SP axis's wrap route. Everything here therefore runs
  on `FABRIC_1D` with `ttnn.Topology.Linear` (`DEC-079`, `DEC-081`, `DEC-097`). `ttnn.Topology.Ring`
  *collectives* do work on `FABRIC_1D` (bit-exact at every shape), so the topology is decided by the
  ring SDPA alone. The in-repo M3 galaxy harness points at the same non-torus descriptor
  (`models/demos/minimax_m3/README.md:48`).
- **Submeshes, never a top-level partial mesh.** Opening `(1,8)` or `(2,8)` directly dies in fabric
  bring-up; open the full `(4,8)` and `create_submesh`. And **two overlapping submeshes with no
  `quiesce_devices()` between their phases hangs the machine** and poisons it until `tt-smi -r` —
  measured, 246 s to the timeout. `tests/test_factory.py::SubmeshPool` makes that unreachable
  through the API.

## Run

```bash
cd /path/to/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
export HF_MODEL=/path/to/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=$HOME/.cache/llama31_8b_d_p          # never the checkpoint dir (DEC-048)
export PREFILL_TRACE_DIR=/path/to/golden/s1024
```

### The gate ladder

```bash
# the golden trace, once (host only, ~100 s, 128 MB at 512 tokens / 0.25 GB at 1024)
python models/demos/llama31_8b_d_p/scripts/generate_golden_kv_cache.py \
    --tokens 1024 --out $PREFILL_TRACE_DIR --verify-loop
python models/demos/llama31_8b_d_p/scripts/verify_golden_kv.py         # G-GOLDEN, imports no ttnn

# the fabric map — run this FIRST on a new machine; it resets the box after the hang case
python models/demos/llama31_8b_d_p/tests/fabric_topology_matrix.py     # G-FABRIC-MATRIX

# every pytest gate (247 tests). -p no:randomly because two P8 files hold submeshes in order
pytest models/demos/llama31_8b_d_p/tests -q -p no:randomly

# the citation verifier — part of every doc gate
python models/demos/llama31_8b_d_p/scripts/verify_citations.py
```

### The status table

```bash
python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py                          # one-shot
PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=512 python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py
PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=256 python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py
PREFILL_RACE_ITERS=3 PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=512 \
    python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py                      # G-RACE
```

### Serving through the engine (`G-REQUEST`, `G-MOCK-MIG`)

Two processes. **Every shared variable must match on both sides** or the byte layout disagrees, and
`PREFILL_NUM_USERS` is one of them — the runner and the producer default it differently (2 vs 1,
`R-051`), so set it explicitly. Choose `PREFILL_MAX_SEQ_LEN` **strictly greater** than
`PREFILL_CHUNK_SIZE`: at equality the per-chip cache shard leaves the ring op no room and attention
falls back to the one-shot bootstrap — a correct but *different* core, so anything measured there is
measuring the wrong path.

```bash
COMMON="PREFILL_MODEL=llama31_8b_d_p PREFILL_SP=4 PREFILL_TP=8 PREFILL_NUM_LAYERS=32 \
        PREFILL_NUM_USERS=1 PREFILL_CHUNK_SIZE=256 PREFILL_MAX_SEQ_LEN=2816 \
        PREFILL_H2D_SERVICE_ID=llama_prefill PREFILL_TRACE_DIR=$PREFILL_TRACE_DIR \
        PREFILL_MIGRATION_TABLE_PATH=/tmp/llama_kv_chunk_table.pb \
        PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/llama_kv_device_map.json"

# Clear stale maps first, or the runner logs "device map ... not found; skipping KV read" and
# every PCC silently vanishes (the failure `PREFILL_MIGRATION_TESTING.md` Gate 1 warns about).
rm -f /tmp/llama_kv_chunk_table.pb /tmp/llama_kv_device_map.json

# terminal A — runner.  Add PREFILL_ENABLE_MIGRATION=1 for G-MOCK-MIG arm 2, which is the only
# arm that takes the engine's real stage-gather branch and calls kv_migration_base_address.
env $COMMON PREFILL_MOCK_MIGRATION=1 \
    python -m models.demos.common.prefill.runners.prefill_runner

# terminal B — producer
env $COMMON PREFILL_PRODUCER_CHUNKS=11 PREFILL_PRODUCER_CHECK_PCC=1 \
    python -m models.demos.common.prefill.runners.prefill_producer
```

## Environment variables

Generated from an AST scan of every `os.environ` / `os.getenv` / `os.environ[...]` read in the
package (`G-CLEAN` item 6 — a hand-written list misses the ones that matter, and the scan must
resolve indirect reads: `LLAMA_DELTA_PROBE` is read through a module constant and a
literal-argument-only scan does not see it). **16 variables, 0 unresolved.** The package invents
**no** `PREFILL_*` variable of its own — the topology that would have needed one is pinned in code
(`DEC-097`).

| Variable | Owner | Read at | Meaning |
|---|---|---|---|
| `HF_MODEL` | shared | `conftest.py:36`, `tt/model_config.py:123`, `tt/runners/adapters/llama.py:341`, `tests/test_factory.py:57`, `scripts/generate_golden_kv_cache.py:340` | checkpoint directory (weights only — never the weight cache) |
| `TT_CACHE_PATH` | shared | `tt/model_config.py:240`, `tt/runners/adapters/llama.py:241` | tilized-weight cache root. The dtype **and** the mesh shape go into the path, because a tilized tensor is already sharded (`DEC-048`) |
| `PREFILL_TRACE_DIR` | the prefill engine | `scripts/{generate_golden_kv_cache,verify_golden_kv}.py`, `tests/galaxy_prefill_kv_pcc.py:336`, 4 test files | golden KV trace directory. Unset makes `G-CHUNK-ATTN`'s vs-golden arm skip rather than fail |
| `PREFILL_HF_MODEL` | the prefill engine | `tt/runners/adapters/llama.py:196` | engine-defined override of the adapter's `hf_model_default` (`models/demos/common/prefill/adapter.py:116`) |
| `PREFILL_TTNN_CACHE` | the prefill engine | `tt/runners/adapters/llama.py:241` | engine-defined override of the adapter's `ttnn_cache_default` (`adapter.py:117`) |
| `PREFILL_MODEL` | the prefill engine | `tests/unit/test_kv_chunk_table.py:48` (`setdefault`) | which adapter the registry resolves. The test pins it *before* importing the producer so the import cannot resolve another model's adapter |
| `PREFILL_CHUNKED` | the prefill engine | `tests/galaxy_prefill_kv_pcc.py:353` | run the harness chunked rather than one-shot |
| `PREFILL_CHUNK_SIZE` | the prefill engine | `tests/galaxy_prefill_kv_pcc.py:354` | tokens per chunk |
| `PREFILL_NUM_LAYERS` | the prefill engine | `tests/galaxy_prefill_kv_pcc.py:356` | layer count. Must match between runner and producer or the ack drain hangs (ack count = layers × chunks) |
| `PREFILL_RACE_ITERS` | this package | `tests/galaxy_prefill_kv_pcc.py:355` | `G-RACE`: repeat the whole harness N times in one process on one `CCLManager` and hash the result |
| `PREFILL_KV_PCC_MIN_K` | this package | `tests/galaxy_prefill_kv_pcc.py:526` | override the K threshold (default 0.99, carried from `G-CHUNK` rather than fitted here) |
| `PREFILL_KV_PCC_MIN_V` | this package | `tests/galaxy_prefill_kv_pcc.py:527` | override the V threshold (default 0.98) |
| `PREFILL_TOPOLOGY` | this package (`DEC-027`) | `tests/test_factory.py:194` | `linear` (**the default here**) or `ring`. Deliberately one knob: it selects the fabric config too |
| `PREFILL_FABRIC` | this package (`DEC-079`) | `tests/test_factory.py:233` | `1d` (the default here) or `1d_ring`. Needs the torus descriptor and **does not work on this galaxy** (`R-030`) |
| `TT_MESH_GRAPH_DESC_PATH` | tt-metal | `tests/test_factory.py:258` | required only for `PREFILL_FABRIC=1d_ring`. A model manifest cannot set it — the runner applies the manifest after `import ttnn`, which is already too late |
| `LLAMA_DELTA_PROBE` | this package (`DEC-023`) | `tt/layer.py:54` | per-layer residual-delta statistics (L2 / mean-abs / signed-mean), for localising a drifting sublayer. A layer- or model-level PCC cannot localise; a growing signed mean is the fingerprint of a directional bias in one sublayer |

Variables the *gates* set but the package never reads — they belong to the engine and are listed
here only so a reader can reproduce a transcript: `PREFILL_SP`, `PREFILL_TP`, `PREFILL_NUM_USERS`,
`PREFILL_MAX_SEQ_LEN`, `PREFILL_H2D_SERVICE_ID`, `PREFILL_PRODUCER_CHUNKS`,
`PREFILL_PRODUCER_CHECK_PCC`, `PREFILL_MOCK_MIGRATION`, `PREFILL_ENABLE_MIGRATION`,
`PREFILL_STANDALONE_CHUNKED_PCC`, `PREFILL_LAYER_ACK_D2H`. Note that `tt-run` forwards only
`TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes, so a `PREFILL_*` set in the shell has no effect
under it — set it in the binding's `global_env` or the model manifest.

## Layout

```
models/demos/llama31_8b_d_p/
├── README.md                       this file
├── conftest.py                     session `state_dict` fixture + --skip-model-load
├── configs/Llama-3.1-8B-Instruct/  the checkpoint's config.json, bundled verbatim
├── bringup_log/                    the evidence: gates, decisions, risks, raw logs
├── tt/
│   ├── config.py                   MeshConfig: parallelism decision, collective wrappers, and the
│   │                               package's ONE compute-kernel-config factory
│   ├── ccl.py                      CCLManager: subdevice, ping-pong semaphores, ring scratch
│   ├── model_config.py             ModelArgs + the one normalised hf_config constructor
│   ├── rms_norm.py  rope.py  mlp.py  embedding.py  lm_head.py
│   ├── attention/                  config · weights · operations · prefill · kv_cache · dense_sp
│   ├── layer.py                    DecoderLayer (+ the LLAMA_DELTA_PROBE bring-up probe)
│   ├── model.py                    Model: embedding → 32 layers → norm → (lm_head)
│   ├── tt_prefill_runtime.py       the chunked runtime the engine drives
│   └── runners/                    adapters/llama.py · kv_chunk_table.py · manifests/*.json
├── scripts/                        verify_citations · generate_golden_kv_cache · verify_golden_kv
└── tests/
    ├── test_factory.py             fixtures, SubmeshPool, and the ONE definition of the
    │                               noise-floor helpers (quantize_like_device, err_ratio)
    ├── unit/                       one file per gate, each with a floor and a negative control
    ├── fabric_topology_matrix.py   G-FABRIC-MATRIX, subprocess-isolated with a timeout
    └── galaxy_prefill_kv_pcc.py    G-MESH-KV, G-RACE
```

57 files (excluding `bringup_log/` and `generated/`), exactly the tree `bringup_log/03_OUTLINE.md`
contracts. Every `tt/` module owns a test; the three 3-line `__init__.py` shims own nothing by
convention. Conventions honoured, and the two the templates do not actually follow themselves, are
tabulated in `03_OUTLINE.md` §5.

## Why not `models/common/` (TTTv2)?

This is the first question a reviewer asks, and the answer is evidence rather than taste
(`bringup_log/02_SURVEY.md` §2). `models/common/` ships a shared module library *and* a complete
Llama-3.1-8B, so the obvious move is to use it. Two things stop it:

- **`MLP2D`'s "2D" is 2D *tensor* parallelism, not TP × SP.** Its prefill path reduce-scatters on
  `cluster_axis=1` and closes with `all_reduce(cluster_axis=0)`
  (`models/common/modules/mlp/mlp_2d.py:461`). With SP on the row axis that all-reduce would sum
  activations belonging to **different tokens** — silently wrong, and it would still produce
  plausible PCC on a one-row mesh. The tempting shortcut "an MLP is token-pointwise, so SP looks
  like DP to it" holds for the math but not for this module's collectives.
- **There is no `Attention2D`**, and `models/common/models/llama3_8b/model.py:890` raises
  `ValueError("Llama3Transformer1D only supports 1D mesh topologies.")` on a 32-device cluster.
  There is no chunked-prefill runtime and no `models/demos/common/prefill` adapter.

So the structural templates are `models/demos/minimax_m3/tt/dense_mlp.py` for the MLP — it
collectives on the **TP axis only**, which is what makes it SP-safe — and
`models/demos/gpt_oss_d_p/tt/attention/` for attention. What is *imported* rather than reimplemented
is listed row by row in `02_SURVEY.md`: the HF→Meta key mapping and `reverse_permute`, the llama3
RoPE frequency math, the chunked-KV and indexed-RoPE ops, `substate`, `get_cache_file_name`,
`get_default_num_links`, and the block-cyclic address walk.

## Not implemented

- **decode**, performance work, trace/2CQ, quantised weights — explicit non-goals of this iteration.
  The runtime **refuses** `use_trace` rather than ignoring it.
- **multi-galaxy / multi-rank pipelined prefill** (`R-032`). Out of scope by instruction. The runtime
  raises on `set_layer_completion_sink`, on a non-first rank's `compile()`, and on the multi-rank
  half of the migration table — `assert_single_rank_stage` refuses a non-zero `first_layer_idx`, a
  partial `num_my_layers`, a gathered stage list carrying more than one rank, a bare dict and an
  empty list. Note that the *first* version of that guard was vacuous on every engine path and would
  additionally have crashed every real migration run (`DEC-111`); the protection this bullet claims
  was not real until that was fixed, which is the sharpest method lesson in the log.
- **`G-LOOPBACK`** — the engine's real DRAM → transport → DRAM copy. Out of scope by `DEC-103`,
  residual gap enumerated in `R-043`.
- **residual scheme B** (a TP-sharded residual stream). The seams are wired and refuse; scheme A is
  what ships (`DEC-025`).
- **`Topology.Ring` end to end.** The deployment collectives have been measured under
  `Topology.Linear` only, because this galaxy has no ring fabric (`R-030`, `R-031`). A torus-cabled
  machine needs a code edit, not a flag.
- **the deployment KV capacity at 131072 tokens has no PCC.** `G-REQUEST` arm 2 served the real
  chunk 8192 / cache 131072 geometry, but there is no fp32 golden that deep (`R-026`), so every PCC
  in this README is measured at a 1024-token cache (`R-039`). Building the address table at that
  capacity — 2,097,152 entries — has also never been timed (`R-052`).
- **`bfloat8_b` KV is a measured choice, not a free one.** It costs 17.8× on K and 17.7× on V against
  bf16 at the cache primitive (`G-KV`, `DEC-021`).

### Known-imperfect in the record itself

Stated here rather than in the log alone, because a reader of this README should not have to find it
(`DEC-120`, `R-053`, `R-054`):

- **247 of this package's 317 prose citations into the recipe had silently drifted** by the end of
  P10, because the kit's recipe grew 1896 → 2235 lines *during* the run. The verifier's doc pass only
  *range*-checks a prose `path:line`, so every one of them reported `resolved` while pointing at
  unrelated text. They were re-pointed mechanically in P9 and the recipe is now fingerprinted, so
  the next drift fails the gate instead of hiding.
- A second class survived that: refs that were **wrong when written**. Sampled after the mechanical
  pass, 5 of 15 were still wrong. **The 125 refs in `tt/`, `tests/`, `scripts/` and this file were
  then read one by one** (57 re-pointed; a fresh sample of 12 is 12/12 correct). The 202 refs inside
  `bringup_log/` got the mechanical pass only, and a 14-ref sample puts their residual error rate at
  about **20%** — so expect roughly 40 wrong line numbers in the logs, and treat a log citation as a
  pointer to a section rather than to a line.
- **431 abbreviated `` `:NNN` `` continuation refs are checked by nothing** — the verifier's regex
  needs a filename, so they are neither resolved nor counted.
