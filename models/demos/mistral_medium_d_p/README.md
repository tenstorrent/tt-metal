# Mistral-Medium-3.5-128B — prefill blocks (`mistral_medium_d_p`)

TTNN **prefill-only** implementation of the Mistral-Medium-3.5-128B *text backbone* for Tenstorrent
Blackhole. Target: one **Blackhole Galaxy (8×4 mesh)** running **TP=4 × SP=8**.

Config: [`configs/Mistral-Medium-3.5-128B/config.json`](configs/Mistral-Medium-3.5-128B/config.json)
(flattened from the published `text_config`; the pixtral vision tower is out of scope).

**Current scope: the attention and MLP blocks only**, so the two can be taken forward by two
engineers in parallel. The decoder layer, model assembly, `TtPrefillRuntime`, the prefill adapter and
the galaxy harness are deliberately not here yet.

## Architecture

| | |
|---|---|
| Decoder layers | 88, **all identical** — no hybrid schedule, no sliding/full alternation |
| Hidden / FFN | 12288 / 28672 |
| Attention | GQA 96 q / 8 kv heads, head_dim 128, **dense causal**, **full rotary**, theta 1e6 |
| RoPE scaling | **YaRN** factor 64 over orig ctx 4096, beta 4/1, attention_factor 1.4158883 |
| MLP | dense **plain SwiGLU** (silu) — *not* the clamped `swigluoai` of gpt-oss / M3 / Kimi |
| Norm | plain RMSNorm, eps 1e-5 (no Gemma `(1+w)` fold) |
| Vocab | 131072, embeddings untied |
| Checkpoint | **per-tensor fp8** (scalar `weight_scale_inv`); lm_head / embeddings / norms bf16 |
| Params | 125.0B text (121.8B ex-embeddings), 243.6 GFLOP/token, KV 352 KiB/token bf16 |

Two mechanisms that no config field advertises, both guarded in code:

* `Ministral3Attention` applies a Llama-4 Q temperature `q *= 1 + beta*log(1 + floor(pos/orig_max))`.
  This checkpoint ships `llama_4_scaling_beta = 0` (exact no-op); the config **class** default is
  `0.1`. `checkpoint.assert_supported` refuses to build if it is non-zero.
* YaRN's `attention_factor` multiplies **cos and sin**, so scores run ~2.0× hot. Baked into the host
  table; dropping it is silently wrong.

## The split: TP=4 × SP=8

```
mesh (8, 4) = 32 Blackhole chips     SP = 8 (rows)      TP = 4 (cols)
```

Same split every other Galaxy prefill model in the repo runs (deepseek_v3/v3.2, kimi_k2_6/k2_7/k3,
glm_5_1/5_2, minimax_m3). TP=4 rather than TP=8 because the TP all-reduce volume per chip is

```
2·(T−1)/T · s_loc·H·2   with  s_loc = S/sp = S·T/32    ->    (S·H·4/32)·(T−1)
```

i.e. **linear in (T−1)** — the `1/T` ring-efficiency gain is exactly cancelled by the per-chip
sequence shard growing with T. At 128K over 88 layers: 231 GB/chip at TP=8 vs 99 GB/chip at TP=4.
The countervailing ring-SDPA KV gather only grows 2.1 → 4.8 GB (GQA has just 8 KV heads). Net **2.25×
less fabric traffic**. See [`config.py`](config.py).

| per chip | value | tiles |
|---|---|---|
| hidden 12288/4 | 3072 | 96 |
| ffn 28672/4 | 7168 | 224 |
| Q heads 96/4 | 24 | — |
| **KV heads 8/4** | **2** | — |
| fused QKV (24+2+2)×128 | 3584 | 112 |
| o_proj K = 24×128 | 3072 | 96 |
| fused gate\|up 2×7168 | 14336 | 448 |

All tile-aligned; no padding anywhere.

### `n_kv_local = 2` is the one genuinely new thing

Every other GQA model in the repo lands on exactly one KV head per chip (minimax_m3 4/4, gpt_oss
8/8), and `deepseek_v3_d_p/utils/kv_cache_utils.py::init_kvpe_cache` hardcodes that `1`. Two is
legal — `update_padded_kv_cache` only requires `cache_shape[1] == input_shape[1]`, and ring-joint
SDPA supports grouped GQA (`NKH == NVH < NQH && NQH % NKH == 0`, ours is 2/2/24) — but unexercised.
`tests/unit/test_ring_joint_sp_vs_ref.py` pins it, **on 4 chips**, before any Galaxy time is spent.

## The block contract

Both blocks have the same signature, under the sharded-residual layout the rest of the family uses:

```
in :  [1, 1, s_local, 12288]   full emb, replicated across the 4 TP cols   (a post-norm activation)
out:  [1, 1, s_local,  3072]   emb/tp,   reduce-scattered across the TP cols
```

Column-parallel weights (QKV, gate|up) consume the full 12288. Row-parallel weights (o_proj,
down_proj) emit a partial sum over the full 12288, closed with a **reduce-scatter only** — no
trailing all-gather. That all-gather belongs in front of the *next* norm, which the (deferred)
decoder layer will own. Fabric bytes are the same as an all-reduce; what is bought is residual adds
on 3072 instead of 12288 and a live residual of 0.094 rather than 0.375 GiB/chip at 128K.

At `tp == 1` this degenerates to the replicated layout (`emb/tp == emb`, the reduce-scatter is
skipped), so the 1-chip tests exercise the same code, not a second branch.

## Ownership

| | files | tests |
|---|---|---|
| **Attention** | `tt/attention/*`, `tt/rope.py`, `tt/rope_tables.py` | `test_rope_vs_hf`, `test_ring_joint_sp_vs_ref` |
| **MLP** | `tt/mlp.py` | `test_swiglu_vs_ref`, `test_mlp_vs_ref` |
| **Shared — frozen** | `config.py`, `tt/ccl.py`, `utils/`, `configs/`, `reference/`, `tt/checkpoint.py`, `tt/rms_norm.py`, `tt/model_config.py`, `tests/test_factory.py`, `tests/unit/shapes.py`, `conftest.py` | `test_checkpoint_ingest`, `test_reference_model`, `test_rms_norm_vs_ref` |

Neither engineer needs to touch the other's files, and neither needs a decoder layer to test: each
test feeds a full-emb activation in and checks the reduce-scattered output.

## Hardware ladder

| tier | devices | what it retires |
|---|---|---|
| host | **0** | YaRN tables vs HF bit-for-bit, fp8 dequant + Meta-RoPE swizzle, torch ref vs `Ministral3DecoderLayer` |
| `2x4` | **8** | **ring-joint SDPA at production TP=4** (24 Q / 2 KV heads per chip) with the ring shortened to SP=2 — BH LoudBox |
| `8x4` | **32** | the SP=8 × TP=4 target — Blackhole Galaxy |

The ladder is deliberately short. Only the rung that runs on hardware we can actually get is wired
up; the rest are added back one at a time as boxes become reachable, so a green run always means
something was really tested rather than skipped. Currently parked, in the order they should return:

| rung | what it needs | what it would retire |
|---|---|---|
| QKV + o_proj on `2x4` | the LoudBox we already target | column/row-parallel splits and the TP reduce-scatter close |
| full block on `8x4` | Galaxy | SP=8 × TP=4 together, the only shape that tests both axes at once |

### Why the smaller rung shrinks SP, not TP

`2x4` holds TP at its production width of 4, so the head split, the cache head count and every
per-chip shape are exactly Galaxy's. Only the ring is shortened, SP 8 → 2 — and because
`chunk_global = sp * chunk_local`, the *global* sequence shortens with it, leaving per-chip load
**identical**: 128 Q rows and a 256-token cache shard per chip on either shape.

That is the whole point. Ring length is the only variable that moves between `2x4` and `8x4`, so a
PCC drop on Galaxy has exactly one candidate cause. Shrinking TP instead to fit a 4-chip box would
change the GQA head grouping at the same time, and a failure would then be ambiguous.

The host tier is the trust anchor: it pins the YaRN tables to HF at 0 ULP out to 262144 positions,
proves the fp8-dequant + Meta-swizzle round-trip, and pins `reference/torch_reference.py` to the real
`Ministral3DecoderLayer`. The device tests then PCC against that reference at 0.99, so a device
failure is never ambiguous about which side is wrong.

## Running

```bash
# 0 devices — run these first, anywhere
pytest models/demos/mistral_medium_d_p/tests/unit/test_rope_vs_hf.py \
       models/demos/mistral_medium_d_p/tests/unit/test_checkpoint_ingest.py \
       models/demos/mistral_medium_d_p/tests/unit/test_reference_model.py --noconftest

pytest models/demos/mistral_medium_d_p/tests/unit -k 2x4   #  8 chips — BH LoudBox
```

Every device test declares its mesh shapes through `tests/test_factory.py::parametrize_mesh_with_fabric`,
which auto-filters to what fits on the current system — a test declaring `[(2,4),(8,4)]` skips the
Galaxy case cleanly on an 8-chip box rather than failing.

## Layout

```
config.py       MeshConfig (TP=4 x SP=8, + reduce_scatter)
conftest.py     session state_dict fixture (real weights; unused by the block tests)
configs/        flattened text-backbone config.json
reference/      torch_reference.py — pure-torch golden, pinned to transformers Ministral3
tt/
  rope_tables.py  YaRN math (pure torch)        rope.py      device rope builders
  rms_norm.py     plain RMSNorm                 mlp.py       dense SwiGLU (col/row parallel)
  attention/      config, weights, operations, kv_cache, dense_sp (ring SDPA), prefill
  checkpoint.py   fp8 dequant + prefix strip + mechanism guards (pure torch)
  model_config.py ModelArgs (config + weights + cache paths)
  ccl.py          CCLManager
tests/          test_factory.py (mesh/fabric parametrize + block helpers), unit/
```

## Not yet done

* **No decoder layer, no model assembly.** The blocks return the sharded residual; whoever
  integrates owns the all-gather in front of each norm and the two residual adds.
* **No `TtPrefillRuntime` / `PrefillModelAdapter`** — wiring into `models/demos/common/prefill/`
  (one line in `ADAPTER_PATHS` plus a runtime) is what makes chunked multi-user serving work.
* **No parallel embedding / lm_head** — a replicated `[131072, 12288]` bf16 table would be 3.22
  GiB/chip; sharding emb_dim over TP brings it to 0.81 GiB and its output *is* the sharded residual.
* Weight dtype is bfp8 everywhere for v1; it is a per-block constructor argument, so bfp4 for the
  MLP (76% of the 125B params) is a one-line change once PCC is green.
* No decode path — prefill only, by design.
