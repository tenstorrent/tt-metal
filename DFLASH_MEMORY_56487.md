# #56487 — DFlash drafter KV: how many users still fit

Working note for https://github.com/tenstorrent/tt-metal/issues/56487
("Dflash prefill - memory usage loss due to full kv cache for drafter model", pavlepopovic).

Status: **theory only, nothing measured on hardware.** Every number below is derived from
config + source constants. Written 2026-09-18 on branch `nmilicevic/dflash-e2e`, untracked
on purpose — this is a scratch analysis, not a deliverable yet.

---

## The question

Kimi-K2.7 prefill runs pipeline-parallel across 4 Galaxies (SC4) or 1 Galaxy (SC1).
DFlash adds a 6-layer drafter whose K/V cache lives **only on the last pipeline rank**.
The drafter attends to a ~4K sliding window, but we retain the **full 256K** so rewind
works. Question: how many concurrent users fit, with and without DFlash?

Answer up front: **SC4 86 -> 69 users (-19.8%)**, **SC1 9 -> 8 (-11%)**.
The cost is the *retention policy*, not DFlash itself — see § Break-even.

---

## Machine + model constants (verified)

| Quantity | Value | Source |
|---|---|---|
| DRAM per chip | 8 x 4,278,190,080 = **34,225,520,640 B** (31.875 GiB) | `tt_metal/soc_descriptors/blackhole_140_arch.yaml:112` |
| Mesh per galaxy | 8x4 — `sp`=8 rows (seq parallel), `tp`=4 cols | `prefill_runner.py` |
| Per-chip seq slice | 256000 / sp = **32000 tokens** | `MAX_SEQ_LEN` / `sp` |
| Max seq len | 256000 (manifest), arch cap 262144 | `manifests/kimi27.json`, `reference/kimi_k2_7_config.py` |
| Shipped users | **86** | `manifests/kimi27.json` `PREFILL_NUM_USERS` |
| Layers | 61 (1 dense + 60 MoE) | `kimi_k2_7_config.py` |
| Drafter taps | layers 1, 12, 24, 35, 47, 58 | `tt/runners/adapters/kimi_k2_7.py:40-42` |

Note the two 8s are **unrelated**: `sp=8` is mesh rows, `BH_NUM_DRAM_BANKS=8` is channels
inside a chip. The 32000 comes from `sp`. Banks only round-robin the 1000 32-token blocks
below that (125/bank). If banks were 6, per-chip seq would still be 32000.

Quantization byte rates: bfp8_b = 1088/1024 = **1.0625 B/elem**, bfp4_b = 576/1024 =
**0.5625**, bf16 = 2.

---

## Per-chip weight footprint (derived)

Placement: routed experts bfp4_b EP-sharded 12/chip (`tt_prefill_runtime.py:188`,
`NUM_ROUTED_EXPERTS // num_devices` = 384/32); MLA + shared expert + dense FFN bfp8_b
TP-sharded /4 and SP-replicated x8; gate + embedding bf16 /4.

| Item | Bytes/chip |
|---|---|
| MoE layer (MLA 26,860,544 + experts 297,271,296 + shared 11,698,176 + gate/norms 1,404,928) | **337,234,944** |
| Dense layer 0 (MLA + dense FFN + norms) | **132,172,800** |
| Embedding (rank 0 only) | **587,202,560** |
| DFlash fc slice, per tap (7168x7168 bfp8 /4) | **13,647,872** |
| DFlash k/v proj tail, all 6 layers (last rank only) | **23,396,352** |

No LM head — removed in #55796.

---

## Per-user KV footprint (derived)

Per chip, per user:

- **Verifier KVPE**, per layer: 32000 x 576 x 1.0625 = **19,584,000 B** (= 76.5 B/token/layer/chip).
  576 = `KV_LORA_RANK` 512 + `QK_ROPE_HEAD_DIM` 64.
- **DFlash drafter K+V**, all 6 layers: 6 x 2 heads x 32000 x 128 x 1.0625 x 2 = **104,448,000 B**
  (= 408 B/token/chip). 8 kv heads TP-sharded /4 = 2 heads/chip.

**Ratio: 104,448,000 / 19,584,000 = exactly 5.333 verifier layers per user.** The drafter's
6 layers cost what 5.33 verifier layers cost, because the drafter is dense-head MHA while
the verifier is latent (576 vs 2x2x128=512... but x6 layers and no LoRA compression).

## Calibration: the 1.50 GB reserve

Everything not weights and not KV (trace region, CBs, activations, fragmentation) is folded
into one empirical constant, fitted so the shipped config lands exactly on 86 users:

```
reserve = 34,225,520,640 - 5,777,899,520 (rank0 weights) - 86 x 313,344,000 (rank0 KV/user)
        = 1,500,037,120 B (1.50 GB)
```

256 MB of that is the trace region (`prefill_runner.py:97`, `PREFILL_USE_TRACE=1` in the
kimi27 manifest). **This is circular for rank 0 by construction** — its value is that it
transfers to the other ranks, which are not fitted. Biggest soft spot in the model.

---

## SC4 result

Split from `compute_layer_split(61, 4, ...)` = **[16, 15, 15, 15]**, starts 0/16/31/46.
The MLA adapter imposes no `valid_starts` (only glm_5_2 does). Taps 1,12 -> rank 0;
24 -> rank 1; 35 -> rank 2; 47,58 -> rank 3.

| Rank | Layers | Weights (GB) | KV/user (MB) | Max users |
|---|---|---|---|---|
| 0 (embedding, 2 fc) | 16 | 5.805 | 313.3 | **85** |
| 1 (1 fc) | 15 | 5.072 | 293.8 | 94 |
| 2 (1 fc) | 15 | 5.072 | 293.8 | 94 |
| 3 (2 fc + kv tail + **drafter cache**) | 15 | 5.109 | 398.2 | **69** |

**86 -> 69 users (-17, -19.8%).** DFlash flips the binding rank from 0 to 3.

At the shipped 86 users the drafter wants 8.98 GB/chip on rank 3, which has 3.85 GB free
(2.35 GB after the reserve). Overage = 5.13 GB/chip raw, 6.63 GB/chip reserve-held —
**164 GB / 212 GB over on that one Galaxy**. Galaxies 1-3 lose only 0.44-0.87 GB each
(the fc slices); the whole loss is concentrated on the last rank.

## SC1 result

One rank, all 61 layers: weights 20.95 GB/chip, +105.3 MB for the drafter fc + kv tail.
KV/user 1,194.6 MB, +104.4 MB with the drafter. **9 -> 8 users (-11%).**

CI pins SC1 to `num_users=1` (`run_multirank_pcc.sh:58-64`), so this is unmeasured and
unmeasurable in the current jobs.

---

## Break-even: the retention policy is the cost, not DFlash

`context_len = 4096` but we keep all 256K for rewind. Rank 3 can absorb a drafter cache of
up to **31,137,262 B/user/chip** before it becomes the binding rank (below that, rank 0's
85 still binds and DFlash is *free* in user terms).

At 408 B/token/chip that is **~76K tokens of drafter retention**. So:

- retain <= 76K -> **0 users lost**
- retain 256K -> **17 users lost**

This is the headline for the issue. If rewind never reaches back further than ~76K, the
full-256K retention is buying nothing and costing 20% of capacity.

## Second lever: rebalance the pipeline split

`PREFILL_PP_LAYER_COUNTS` (read at `prefill_runner.py:401`) overrides the even split.
Moving layers *off* rank 3:

| Split | Binding rank | Max users |
|---|---|---|
| [16,15,15,15] (default) | 3 | 69 |
| **[16,17,16,12]** | 1 | **81 (+17%)** |
| [17,16,16,12] | 0 | 79 |

One env var recovers most of the loss. Cost: pipeline imbalance (rank 1 does 17 layers
while rank 3 does 12), so throughput drops even as capacity rises — **not modelled here**.
Do **not** rebalance onto rank 0; it already carries the embedding.

## Separate finding: host-RAM address table is 91% waste

`KvChunkAddressTable` holds `std::vector<KvCacheLocation> entries` (16 B each, indexed
[slot][layer][chunk]) in **host RAM**, not device DRAM — it does not compete with the above.
See `tt_metal/api/internal/disaggregation/kv_chunk_address_table.hpp:89`.

The dflash configs publish `num_layers = 61 + 6 = 67` across 16 configs:
16 x 67 x 8000 chunks x 16 B = **137.2 MB/user** -> **11.80 GB at 86 users**, of which the
verifier's 61 layers are never filled by the drafter path. Trimming the drafter configs to
their own 6 layers: 16 x 6 x 8000 x 16 = 12.3 MB/user -> 1.06 GB. **~10.7 GB of host RAM
recovered**, no device impact. Probably the cheapest fix in the whole issue.

---

## Caveats — read before quoting any number

1. The 1.50 GB reserve is an **empirical fit to rank 0**, not a measurement. Every
   other rank's user count inherits that assumption.
2. No fragmentation, alignment, or allocator overhead is modelled.
3. Kimi **cannot** TP-shard its KVPE cache — `supports_tp_shard_kv` is True only on
   `glm_5_2.py:47`. That lever is unavailable here. (The *drafter* cache is already
   TP-sharded on the kv-head axis, 8 heads -> 2/chip.)
4. The pipeline-rebalance throughput cost is unquantified.
5. The "verifier's own table is 165 MB" figure from the first pass was **not re-derived**
   and does not reconcile with 86 x 61 x 8000 x 16 = 671 MB. Recheck before using.

---

## How to resume

1. Re-derive nothing — the tables above are recomputed and self-consistent as of this file.
   Start from § Caveats: items 1 and 5 are the two open holes.
2. To validate the reserve empirically: run SC4 at increasing `PREFILL_NUM_USERS` until OOM
   and compare against the predicted 69. **Needs reserved machines** — at the time of
   writing the boxes are unreserved and nothing may be run locally.
3. To post to the issue: the deliverable is § SC4 result + § SC1 result + § Break-even.
   The other two sections are recommendations, not answers to what was asked.
4. Decide with pavlepopovic whether rewind actually needs >76K of drafter history. That
   single answer determines whether this issue is a capacity problem or a config typo.

## Key files

- `models/demos/common/prefill/runners/prefill_runner.py` — `:79` MAX_SEQ_LEN, `:88` TP_SHARD_KV,
  `:97` trace region, `:401` PREFILL_PP_LAYER_COUNTS, `:480` layer_split, `:499` num_layers=num_my_layers
- `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:188` — experts_per_chip
- `models/demos/deepseek_v3_d_p/reference/kimi_k2_7_config.py` — all model dims
- `models/demos/deepseek_v3_d_p/tt/runners/adapters/kimi_k2_7.py:40-42` — drafter tap layers
- `models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi27.json` — 256000 / 86 users / trace on
- `models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi27_dflash.json` — dflash, trace off, 1 user
- `tt_metal/api/internal/disaggregation/kv_chunk_address_table.hpp:89` — host-side table
- `tt_metal/soc_descriptors/blackhole_140_arch.yaml:112` — dram_bank_size
