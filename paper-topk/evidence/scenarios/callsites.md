# Real LLM top-k call sites in tt-metal — shapes, engines, repro calls

Miner report for the ledger's new "real model scenarios" region. Repo: `/home/nachiket/tt-metal` @ `nkapre/sorting` (read-only survey; no device runs). All paths absolute-relative to repo root.

## Engine gates verified from source (applied below)

- **Stock factory selection** (`ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp:59-76`, `device/topk_constants.hpp:11`): multi-core bitonic iff padded W >= 8192 AND padded W < 65535 AND padded W pow2 AND k <= 64 (plus L1 cost check). Everything else → single-core insertion (~linear in W).
- **Routed path** (`ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:258-320`): BH + bf16 + TILE + interleaved + last-dim + `largest` + `!stable` + **no `indices_tensor`, no preallocated outputs, no `sub_core_grids`** + k <= 2048; k > 64 → route if k <= W <= 2^19; k <= 64 → route only if padded W is non-pow2-or->=65535 AND >= 4096.
- **`ttnn.experimental.topk_large_indices` direct** (`.../experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp:26-50`): BH-only, ROW_MAJOR bf16 interleaved, k in [16, 2048] multiple of 16, optional `valid_length`.

**The single most consequential call-site fact:** every production *sampling* `ttnn.topk` call passes `indices_tensor=` (and usually `sub_core_grids=`), and the `tt_sampling.py` family additionally passes `stable=True` on WH/BH (`models/common/sampling/tt_sampling.py:127,819,859`). Each of those independently disqualifies the routed path (`topk.cpp:271-279`). **No production sampling top-k currently reaches our routed path** — the routing only helps these models if the call sites are later relaxed. Ledger rows must therefore measure both the "canonical" form (no extra args → routing applies) and the "faithful" form (with `indices_tensor`/`sub_core_grids`/`stable` → stock always).

---

## 1. Call-site inventory

### A. Decode sampling over the (sharded) vocab — `ttnn.topk`, k=32, rows=32

Two parallel implementations of the same chain (lm_head vocab-sharded → per-device top-32 → all-gather [1,1,32,32·ndev] tuples → `ttnn.sampling`):

| Call site | Form | Shape at call | Flags |
|---|---|---|---|
| `models/common/sampling/tt_sampling.py:847` (multi-device) | `ttnn.topk(x_bf16, k=max_top_k, dim=-1, sub_core_grids=…, indices_tensor=…, stable=True-on-WH/BH)` | `[1,1,32, W_pd]` bf16 TILE | largest=True, sorted=True (defaults), **stable=True on BH** |
| `models/common/sampling/tt_sampling.py:807` (single-device `multi_step_reduction`, mesh 1×1) | same, over each vocab **half** | `[1,1,32, vocab/2]` ×2 | same |
| `models/common/modules/sampling/sampling_1d.py:568` (multi-device) | `ttnn.topk(…, sub_core_grids=…, indices_tensor=…)` — **no stable** | `[1,1,32, W_pd]` bf16 TILE | largest/sorted defaults, stable=False |
| `models/common/modules/sampling/sampling_1d.py:530` (single-device split) | same, over each vocab half | `[1,1,32, vocab/2]` ×2 | stable=False |

`W_pd` = per-device vocab shard, tile-aligned then padded to next pow2 (`pad_to_power_of_2` / `pad_logits_to_power_of_2`; e.g. `sampling_1d.py:556-566`, `tt_sampling.py:838-845`). Resolved per model:

| Model (config source) | vocab | mesh/TP | per-dev | padded W | Engine today |
|---|---|---|---|---|---|
| Llama-3.3-70B T3K (`models/common/models/llama33_70b/model.py:653-680`, vocab 128256) | 128256 | 8 dev | 16032 | **16384** | multi-core bitonic |
| Qwen2.5-72B / Qwen3-32B (`models/common/models/qwen25_72b/model.py:690`) | 152064/151936 | 8 dev | 19008 | **32768** | multi-core bitonic |
| gpt-oss (`models/demos/gpt_oss/tt/model.py:21-31,198-204`, vocab 201088) | 201088 | TP=8 | 25152 | **32768** | multi-core bitonic |
| MiniMax-M3 (`models/demos/minimax_m3/tt/model.py:22-31,267-272`, vocab 200064) | 200064 | TP=8 | 25024 | **32768** | multi-core bitonic |
| **Qwen3 on BH p150 boxes, TP=4** (`models/demos/blackhole/qwen36/tt/model.py:43-61`, vocab 151936) | 151936 | 1×4 | 37984 | **65536** | **single-core insertion** — 65536 fails the `< 65535` bitonic gate (`topk_device_operation.cpp:70`); routing blocked by `indices_tensor`+`sub_core_grids`+`stable=True`. This is the worst production sampling shape in tree (~9 ms/row-batch at 137 ns/elem scale). TP=8 → 18992→32768 → bitonic. |
| Llama-3.2-1B/3B, Phi-4 single chip (`models/common/models/llama32_1b/model.py:667`; `.../phi4/model.py:577`) | 128256/100352 | 1×1 | 64128/50176 per half | unpadded (split path deliberately unpadded, `sampling_1d.py:74-78`) | **single-core insertion** (non-pow2), ×2 calls |

Cadence: once per decode step per device (inside the decode trace), the hottest recurring top-k in every text demo. Note logits are typecast bf16 before the call (`tt_sampling.py:797`), so dtype is always bf16 here even when lm_head emits bfp8.

Adjacent narrow call: `models/common/sampling/tt_log_probs.py:586` — `ttnn.topk([1,1,B,256], k=32)` narrowing gathered 256 tuples; per decode step when logprobs enabled; tiny-N, single-core, fine.

### B. DeepSeek-class DSA/NSA indexer — `ttnn.experimental.topk_large_indices` (direct)

- `models/demos/deepseek_v3_d_p/tt/mla/indexer.py:737`: `topk_large_indices(logits, k=index_topk_capacity, valid_length=end_pos)`.
- Input: `[1, 1, S/(sp·tp), T]` bf16 ROW_MAJOR interleaved — the ring-indexer score output (`indexer.py:707-723`); tail `[end_pos, T)` is stale, bounded via `valid_length`.
- k: `min(index_topk, seq_len)`, asserted `16 <= k <= 2048, k%16==0` (`indexer.py:314-317`). `index_topk` = **2048** for DeepSeek-V3.2 / GLM-5.1 / GLM-5.2 / Kimi-K2.6 (`reference/glm_5_2_config.py:51`, `reference/cpu_deepseek_v32/model.py:76`), **512** for DeepSeek-V4 (`reference/deepseek_v4/configuration_deepseek_v4.py:173`).
- Production geometry (`tests/sparse_mla/test_sparse_mla_perf.py:13-15`): Galaxy SP=8×TP=4, chunk 5120 → **rows = 160** per call; cache depth per chip 50k warm, 0.5M "long"; GLM-5.2 max context 1M (`glm_5_2_config.py:40`), Kimi-K2.6 262144 (`reference/kimi_k2_6_config.py:47`). Per sparse MLA layer (61 for DS-v3.2; GLM-5.2 shared layers reuse a prior layer's indices, `mla.py:1348-1363`) per prefill chunk; sparse decode uses the same op with 1-row-scale queries.
- Output indices are all-gathered TILE (RM composite-all-gather deadlocks fabric, `indexer.py:742-755`) — a known caveat for any sweep that adds a gather.

### C. MiniMax-M3 MSA block selection — `topk_large_indices`, k=16

- `models/demos/minimax_m3/tt/attention/msa.py:147`: `topk_large_indices(block_scores, k=topk_blocks)`; `topk_blocks=16`, `block_size=128`, defaults from `tt/layer.py:120-121`.
- Input: indexer_score_msa output, bf16 ROW_MAJOR, rows = per-device query chunk Sq, N = ceil(T/128) blocks → **N=1024 @128k ctx, 8192 @1M ctx**. Layers 3-59 (57 sparse layers) per prefill chunk (`tt/layer.py:105-113`).
- k=16 is the op's floor — exercises the smallest-k corner of the direct op.

### D. MoE expert gates

| Call site | Shape | k | Engine today |
|---|---|---|---|
| `models/demos/gpt_oss/tt/topk.py:26` `ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)` | `[tokens, 128]` bf16 (typecast from bfp8 at :24) | 4 | single-core stock (N=128 < 8192); decode B=32 takes the **fused** `ttnn.experimental.topk_router_gpt` instead (`topk.py:88,169-175`, requires exactly 128 experts) |
| `models/common/modules/moe/tt_moe_gate.py:639` fallback `ttnn.topk(rank_key, self.k, dim=-1)` | `[1,1,batch,N]`; fallback fires for k∉{4,6,8} or N>512 — e.g. **qwen3.5: N=512, k=10** (`tt_moe_gate.py:161-164`) | 10 | single-core stock |
| `models/demos/minimax_m3/tt/topk.py:59` | fused `deepseek_prefill.moe_grouped_topk` (n_groups=1) over `[tokens,128]`, k=4 | 4 | dedicated fused kernel — no ttnn.topk |
| `models/demos/deepseek_v3_d_p/tt/moe/tt_moe_gate_prefill.py:585,767` | fused `deepseek_grouped_gate` / `moe_grouped_topk`, 256 experts, k=8, groups 8/topk_groups 4 | 8 | dedicated fused kernels |
| Legacy: `models/tt_transformers/tt/mixtral_moe.py:117`, `models/experimental/grok/tt/grok_moe.py:119` | `[1,1,32,64]` (8 experts padded to 64), k=32-then-slice | 32 | single-core stock |

Cadence: per MoE layer per forward (heavier than sampling in call count: 36-62 layers/step) but N is tiny.

### E. Minor / fallback sites

- `models/demos/informer/tt/ops.py:249-267`: ProbSparse attention `ttnn.topk(values, k=tile-aligned, largest=True, sorted=False)`; **fp32 inputs are deliberately detoured to an argmax loop** (`:258-260`) — evidence that fp32 topk is not trusted; N = seq len (~hundreds).
- `models/experimental/uniad/tt/ttnn_nms_free_coder.py:35`: falls back to *torch* topk with comment "issue in ttnn topk".
- `models/experimental/tt_symbiote/core/dispatchers/default_dispatcher.py:963,982`: generic dispatcher passthrough, any shape.

---

## 2. Top-5 scenarios (ranked by real-world weight)

### S1 — DeepSeek-V3.2/GLM/K2 DSA indexer top-2048 (the flagship direct-op consumer)
- **Shape**: `[1,1,160,T]` bf16 RM, k=2048, `valid_length=end_pos`; T from 55296 (warm 50k+5k) to 1M (GLM-5.2). DS-V4 variant k=512 at T up to 262144 — this **is** the "k512@262144" scenario, and yes a model calls it.
- **Engine today**: `topk_large_indices` direct — our op, called by name. Row-parallel multi-row path (160 rows) benefits directly from the new chunk-skip.
- **Repro**:
  `ttnn.experimental.topk_large_indices(x, k=2048, valid_length=225280)` with `x = ttnn.from_torch(torch.randn(1,1,160,262144).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)`; variants k=512, T=1048576 (GLM-5.2 long), rows∈{32 (decode-ish), 160, 640 (TP=1)}.
- **Caveats**: measure the `valid_length < T` case (stale tail skip) separately from full-width; ledger routed-composite envelope is only silicon-validated to 131072 (`topk.cpp:253`) but the direct op has no such cap — this scenario stresses beyond the synthetic grid's W range.

### S2 — Multi-device LLM decode sampling, per-device vocab top-32 (the hottest recurring call)
- **Shape**: `[1,1,32,W]` bf16 TILE, k=32, dim=-1, largest, sorted; W ∈ {16384 (llama-70B T3K), 32768 (qwen-72B/gpt-oss/minimax TP=8), **65536 (qwen36 BH TP=4)**}.
- **Engine today**: W=16384/32768 → stock multi-core bitonic. **W=65536 → stock single-core insertion** (fails `<65535`), and the routed rescue is blocked at the call site by `indices_tensor`+`sub_core_grids` (+`stable=True` in tt_sampling). This is a live production perf bug shape on exactly our BH p150 hardware.
- **Repro (canonical)**: `ttnn.topk(x, k=32, dim=-1)` on `[1,1,32,W]` bf16 TILE for the three W. At W=65536 the canonical form routes (k<=64 arm: pow2 but >=65535 → structurally ineligible for bitonic, W>=4096). **Faithful form**: add `indices_tensor=` (uint16 TILE iota `[1,1,32,W]`), `sub_core_grids=`, `stable=True` — always stock. Measure both; the delta is the case for relaxing the call-site gates.
- **Caveats**: `indices_tensor` changes index dtype semantics (uint16 vs uint32 per `tt_sampling.py:101-105`); the pow2 padding is done by the model, so sweep the padded width, not the raw vocab.

### S3 — Single-chip decode sampling split path (N150/P100/P150 llama-1B/3B, phi4)
- **Shape**: two calls of `[1,1,32,64128]` bf16 TILE, k=32 (llama vocab 128256 halved, deliberately unpadded — `sampling_1d.py:74-78`); phi4: 50176.
- **Engine today**: single-core insertion (non-pow2) — twice per token. Canonical form (no `indices_tensor`) hits the routed small-k arm (non-pow2, >=4096) → our path is eligible and should be a large win.
- **Repro**: `ttnn.topk(x, k=32, dim=-1)` on `[1,1,32,64128]`; faithful form adds `indices_tensor`/`stable=True`.
- **Caveats**: 64128 is non-tile? (64128 = 2004×32, tile-aligned) so padded==logical; also compare against a hypothetical pow2-padded 65536 single call — the model's own split-in-half exists purely to dodge stock's weaknesses.

### S4 — MiniMax-M3 MSA block top-16 (direct-op floor-k consumer)
- **Shape**: `[1,1,Sq,nblk]` bf16 RM, k=16; nblk=1024 @128k, 8192 @1M; Sq = per-device chunk (e.g. 2048).
- **Engine today**: `topk_large_indices` direct (`msa.py:147`). k=16 exercises the minimum-k / many-rows corner (row-parallel + chunk-skip).
- **Repro**: `ttnn.experimental.topk_large_indices(x, k=16)` with `x=[1,1,2048,8192]` bf16 RM.
- **Caveats**: rows >> N/k here — completely different balance from the synthetic grid; small-N-many-rows may be where row-parallel overhead shows.

### S5 — MoE expert gate top-k (honest no-change region)
- **Shapes**: gpt-oss `[32,128]` bf16 k=4 sorted=True (`topk.py:26`); qwen3.5 fallback `[1,1,32,512]` bf16 k=10 (`tt_moe_gate.py:639`); mixtral/grok legacy `[1,1,32,64]` k=32.
- **Engine today**: single-core stock for all (N < 8192); gpt-oss decode B=32 uses the fused `topk_router_gpt` kernel instead. DeepSeek/MiniMax gates use dedicated fused grouped-topk kernels, not `ttnn.topk` at all.
- **Eligibility**: **out of scope for our ops** — k=4/10 violate the k%16==0/k>=16 direct-op gate; N=128-512 is below every routing threshold; single-core at N=128 is ~18 µs-scale and fine. Worth ONE ledger row each purely as "no-change" proof that routing never regresses tiny-N (the routing gates provably can't fire here, so any measured change would be a bug).
- **Repro**: `ttnn.topk(x, k=4, dim=-1, sorted=True)` on `[32,128]` bf16 TILE; `ttnn.topk(x, k=10, dim=-1)` on `[1,1,32,512]`.

---

## 3. Cross-cutting caveats for the sweep

1. **dtype**: every surveyed production call is bf16 at the call (explicit typecasts at `tt_sampling.py:797`, `gpt_oss/tt/topk.py:24`); fp32 topk is actively avoided (`informer/tt/ops.py:258`). fp32 rows in the ledger are synthetic-only.
2. **dim**: every surveyed call is last-dim. No production dim!=last site found.
3. **stable**: `tt_sampling.py` family sets `stable=True` on WH/BH (`:127`) — a routed-path disqualifier by design (`topk.cpp:271-274`). `sampling_1d.py` (the newer 1D module used by llama33_70b/qwen/phi4 common models) does NOT set it. Any "make sampling use the routed path" proposal must resolve the stable-tiebreak contract (issue #33492 referenced at `tt_sampling.py:122-126`).
4. **indices_tensor/sub_core_grids**: universal in sampling; a sweep should report both canonical and faithful variants (S2/S3) — the faithful variant is what the model actually gets today.
5. **Non-pow2 vocab**: models pre-pad to pow2 themselves (gpt_oss/minimax `compute_per_device_vocab`, sampling `pad_to_power_of_2`) — the padding cost (a `ttnn.pad` per step when not pre-padded into lm_head) is part of the real pipeline but outside the topk kernel time.
6. **CSA/1M**: nothing named "CSA" calls ttnn.topk; the 1M-context reality is GLM-5.2 via S1 (k=2048, T→1M) and MiniMax MSA via S4 (nblk=8192).
