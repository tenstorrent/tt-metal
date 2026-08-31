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
`tests/unit/test_ring_joint_sp_vs_ref.py` pins it on the **32-chip Galaxy**, at the production shape.

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
| **Attention** | `tt/attention/*`, `tt/rope.py`, `tt/rope_tables.py` | `test_rope_vs_hf`, `test_ring_joint_sp_vs_ref`, `test_attention_vs_ref` |
| **MLP** | `tt/mlp.py` | `test_swiglu_vs_ref`, `test_mlp_vs_ref` |
| **Shared — frozen** | `config.py`, `tt/ccl.py`, `utils/`, `configs/`, `reference/`, `tt/checkpoint.py`, `tt/rms_norm.py`, `tt/model_config.py`, `tests/test_factory.py`, `tests/unit/shapes.py`, `conftest.py` | `test_checkpoint_ingest`, `test_reference_model`, `test_rms_norm_vs_ref` |

Neither engineer needs to touch the other's files, and neither needs a decoder layer to test: each
test feeds a full-emb activation in and checks the reduce-scattered output.

## Hardware ladder

| tier | devices | what it retires |
|---|---|---|
| host | **0** | YaRN tables vs HF bit-for-bit, fp8 dequant + Meta-RoPE swizzle, torch ref vs `Ministral3DecoderLayer` |
| `8x4` | **32** | **ring-joint SDPA at the SP=8 × TP=4 target** (24 Q / 2 KV heads per chip), and the **whole attention block** end to end — Blackhole Galaxy |

The ladder is deliberately short: the host tier, and the real shape. Rungs are added back one at a
time as hardware becomes reachable, so a green run always means something was really tested rather
than skipped. Currently parked, in the order they should return:

| rung | what it needs | what it would retire |
|---|---|---|
| a first green run of `test_attention_vs_ref` | Galaxy time | the whole block — it is written but **unrun** |

### There is no scaled-down rung, and there cannot be one on a Galaxy

A `2x4` LoudBox rung was tried and removed. The idea was sound — hold TP at its production width of 4
so the head split and every per-chip shape stay Galaxy's, and shorten only the ring, SP 8 → 2; since
`chunk_global = sp * chunk_local` the global sequence shortens with it and per-chip load is
*identical* (128 Q rows, a 256-token cache shard). Ring length would have been the only variable.

It does not run on a Galaxy. A `2x4` submesh opens fine with the fabric disabled, but **both**
`FABRIC_1D_RING` and `FABRIC_1D` die in the ethernet router handshake:

```
Fabric Router Sync: Timeout after 10000 ms on Device 0: expected status 0xa2b2c2d2.
Master chan=3 got 0xa0b0c0d0. furthest-behind stage: STARTED
```

Carving 8 chips out of the 32-chip fabric leaves their ethernet partners outside the submesh with no
router kernel running, so the handshake never completes. It is not ring-specific, and not specific to
`2x4` — the MLP block's `1x4` rung (`test_mlp_vs_ref`) fails with the identical timeout on this box.
**Only the full `8x4` allocates.** Not fixable from this package: a smaller rung needs a smaller
machine. Note `parametrize_mesh_with_fabric` filters by **device count only**, so it cannot detect
this — a shape that fits may still be unallocatable, which is why `MESH_SHAPES` is declared against
real SKUs rather than chip budgets.

So on this hardware the choice is the production shape or nothing, and `8x4` is what runs. That is
also the shape where the correctness claim has teeth: each chip's 128 Q rows must see all 1024
positions, i.e. every one of the 8 ring hops.

The host tier is the trust anchor: it pins the YaRN tables to HF at 0 ULP out to 262144 positions,
proves the fp8-dequant + Meta-swizzle round-trip, and pins `reference/torch_reference.py` to the real
`Ministral3DecoderLayer`. The device tests then PCC against that reference at 0.99, so a device
failure is never ambiguous about which side is wrong.

## Performance (2026-08-31, HiFi2)

Measured on a 32-chip Blackhole Galaxy with `tests/test_attention_perf.py` under the light profiler
(`python3 -m tracy -p -r -a device_kernel_duration`), 8 iterations after 2 warmups, signpost-bracketed.
Device kernel time, critical path across the 32 chips.

| case | chunk_global | cached_len | µs/call | ns/token | SDPA share |
|---|---|---|---|---|---|
| c128_p0 | 1,024 | 0 | 1087.9 | 1062.4 | 21.9% |
| c640_p0 | 5,120 | 0 | 1854.8 | 362.3 | 35.8% |
| c1024_p0 | 8,192 | 0 | 2672.2 | 326.2 | 37.7% |
| c640_p4 | 5,120 | 20,480 | 4054.6 | 791.9 | **70.2%** |
| c1024_p4 | 8,192 | 32,768 | 6182.9 | 754.7 | **72.9%** |

**Ring SDPA is the budget at realistic depth.** Everything else — matmul, RoPE, head reshape, the TP
reduce-scatter — is flat in `cached_len`, because only the ring touches the whole prefix.

**Lever 1 — fidelity.** The ring SDPA now runs `HiFi2 + packer_l1_acc=True`, matching
`deepseek_v3_d_p`'s `mla.py` default. The KV cache is bf8, so HiFi4 spent four mantissa passes on
~7-bit operands while halving peak throughput (110 vs 220 TFLOP/s across the 110 SDPA cores):

| case | SDPA HiFi4 → HiFi2 | speedup | block speedup |
|---|---|---|---|
| c640_p0 | 885.5 → 663.1 | 1.34x | 1.12x |
| c640_p4 | 4105.9 → 2847.5 | 1.44x | 1.31x |
| c1024_p4 | 6535.0 → 4510.3 | **1.45x** | 1.33x |

Accuracy did not move: ring-SDPA PCC 0.9998847 (HiFi2) vs 0.9998858 (HiFi4); chunked-vs-reference
0.9976430 vs 0.9976288. All 8 correctness tests pass either way. Utilization at HiFi2 reaches
**102.8 TFLOP/s = 46.7% of peak** at the deepest point, so the op is not inefficient — it was
fidelity-capped. The 1.45x (not 2x) says the ring gather contributes real non-compute time.

**Lever 2 — chunk size.** SDPA fits `t = k * chunk_global * logical_n`, with `k` fit **per chunk size**:

```
chunk_global 5120   k = 2.530e-5 (logical_n 5120) -> 2.172e-5 (25600)
chunk_global 8192   k = 1.501e-5 (logical_n 8192) -> 1.344e-5 (40960)     39% cheaper per Q*KV unit
```

Projected 128K prefill over 88 layers: **28.3 s** (HiFi4 @5120) -> **21.2 s** (HiFi2 @5120) ->
**13.8 s** (HiFi2 @8192). Attention-only; 5120 is a shared default in `prefill_runner.py:124`
(`PREFILL_CHUNK_SIZE`) that likely encodes D2D/migration constraints this profile cannot see.

**Accuracy at depth (open issue).** The bf8 KV cache loses accuracy as the prefix grows — measured on
a single *unchunked* call, so it is not a chunking effect:

| attended prefix | 1,024 | 2,048 | 4,096 | 51,200 |
|---|---|---|---|---|
| tail PCC | 0.99185 | 0.98379 | 0.93517 | **0.60253** |

The 51,200 point is from `tests/unit/test_attention_accuracy_at_depth.py` and was measured under
HiFi4; it has not been re-run at HiFi2 (at 2048 the two agree to five decimals). Random activations
are the worst case for this, so treat 0.61 as a lower bound — but real-weight PCC at length should be
measured before trusting bf8 at 128K. Re-running the depth sweep with a bf16 cache would confirm the
cause.

**Caveat that outweighs the rest:** this box has no wrap-around links, so everything ran on a *linear*
fabric (`MISTRAL_LINEAR_FABRIC=1`). The ring SDPA's KV gather is exactly what a torus changes, and it
is the dominant cost — absolute numbers are a ceiling, ratios hold. Drop the env var on a
ring-capable Galaxy.

Reduce a run with `tests/summarize_attention_perf.py <iters> <label>=<csv>`.

## Running

```bash
# 0 devices — run these first, anywhere
pytest models/demos/mistral_medium_d_p/tests/unit/test_rope_vs_hf.py \
       models/demos/mistral_medium_d_p/tests/unit/test_checkpoint_ingest.py \
       models/demos/mistral_medium_d_p/tests/unit/test_reference_model.py --noconftest

pytest models/demos/mistral_medium_d_p/tests/unit -k 8x4   # 32 chips — BH Galaxy (the target)
```

Every device test declares its mesh shapes through `tests/test_factory.py::parametrize_mesh_with_fabric`,
which auto-filters by device count to what fits on the current system, so the Galaxy tests skip
cleanly on a smaller box rather than failing.

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
