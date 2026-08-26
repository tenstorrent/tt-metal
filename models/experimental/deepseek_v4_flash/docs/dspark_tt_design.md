# DSpark speculative decoding on Blackhole Galaxy — architecture design

**Status**: design document. No implementation.
**Target model**: DeepSeek-V4-Flash-0731 (284B total / 13B active, 43 layers, hybrid attention, ships DSpark in-checkpoint)
**Target hardware**: Blackhole Galaxy (32× BH, `galaxy32` profile)
**Code baseline**: `origin/smanoj/ds_v4_flash` @ `51cd595eba3` (2026-08-25, "Updated README. DeepSeek V4 Flash demo on BH LB. 15.9 tok/s/u")
**Branch**: `zni/dspark-tt-design`, cut from `origin/smanoj/ds_v4_flash`
**Why this path**: the repo-root `docs/` is the Sphinx build tree (`docs/source`, `docs/Doxyfile`), which is the wrong home for a design document. This lives under the model directory, alongside `VLLM_INTEGRATION_PLAN.md` and `WEIGHT_PLACEMENT.md`.

Unless stated otherwise, every `path:line` in this document is relative to `models/experimental/deepseek_v4_flash/` and points at the commit above.

---

## Contents

| § | Contents |
|---|---|
| [§0](#0-tldr--three-findings-that-change-the-conclusion) | **TL;DR — three findings that change the conclusion** (read this first) |
| §1 | Scope |
| §2 | **Phase 0: state of the code** (Q1–Q8 + the drafter, with line numbers) |
| §3 | Phase 0 addendum: Q9 — tt-metal mainline |
| §4 | **Phase 1: targeted research** (8 sources, retrieval status stated per source) |
| §4.10 | **What we can reuse directly from the SGLang implementation** |
| §5 | **Phase 2: design** (adjudicating the zero hypothesis + design points a–g) |
| §6 | **Decision log** (D-01 … D-16) |
| §7 | **Baseline definition** (three arms + configuration matrix + metrics) |
| §8 | **Phase 3: staged plan** (Phase 0 / A / A' / B / C, incl. "how fast to a number") |
| §9 | **Open questions / unknowns** (U1 … U14, each with a falsifying experiment) |
| §10 | Risk register |
| §11 | What this document did not do + corrections to the brief's premises |

---

## 0. TL;DR — three findings that change the conclusion

After reading the code, the zero hypothesis in the brief — *"port SGLang verbatim: bucket by total verify tokens + front-pack"* — **mostly does not hold**. Not because TT is weaker, but because this codebase is structured differently from a GPU stack. Three findings:

**Finding 1 — a TT-NN trace here is not "fully static"; it has an in-trace H2D socket.**
Every decode step's input is packed into one `[1,1,1,16]` INT32 packet, received by a `recv_async_h2d` **inside** submesh 0's trace (`tt/model.py:1577-1599`, `tt/model.py:1937`); the output is pushed back to host by a `send_async_d2h` inside the last submesh's trace (`tt/model.py:1952`). Position, RoPE, the attention mask, the compressor slot/row indices and SDPA's `cur_pos` are **all derived on device from that packet's `pos`** (`tt/model.py:1632`, `1649`, `1697`, `1707`).
So the accurate statement of the constraint is not "there is no host sync point inside a trace", but: **shapes and the op sequence are frozen at capture; values can be injected from host every step, at the cost of one PCIe write that never touches the command queue.** That is *more* capable than a CUDA graph's dynamic parameters, not less. Reasoning as "whatever a CUDA graph can do, TT cannot" produces wrong conclusions here.

**Finding 2 — the per-slot position semantics a multi-token verify needs already exist.**
The packet's position regions are **B wide**, and the comment states exactly why: *"The two position regions are B wide even though a step's users share one position, because the ops that consume them want a value per user (`paged_update_cache`'s update index, SDPA-decode's `cur_pos`)"* (`tt/model.py:1471-1478`). `_update_cache_at` takes `[B]` INT32 (`tt/attention.py:441-448`); `_sdpa_decode`'s causal mode takes `cur_pos [B]` (`tt/attention.py:1149-1153`). And `_pack_tokens`'s signature is already `[B,S,1,D] → [1,1,B*S,D]` (`tt/attention.py:337-350`).
In other words: **reinterpret the batch slots as "K consecutive positions of one user" and the existing op chain *is* a multi-token verify kernel** — KV is written before it is read (write at `tt/attention.py:1413`, read at `tt/attention.py:1440`), and each slot's `cur_pos = pos+i` gives intra-block causality for free, with no extra mask. What blocks it is four hard-coded assumptions, not the architecture (see §5.1).

**Finding 3 — the expensive thing is not the verify shape, it is MoE's per-token linear cost.**
`fused_experts` is decode-only (`T == 1`), and T tokens are split into **T single-token ops**: *"`T` tokens run as `T` single-token ops rather than one wider one -- they route to different experts, so there is no weight to share between them"* (`tt/moe.py:578-586`). Meanwhile every dense projection / norm / mHC packs ≤32 tokens into **one tile-row**, which *"makes a B-user step cost the same projections / norms / RoPE as a one-user step"* (`tt/attention.py:337-345`).
So the TT verify cost model is **T(M) = f + c·M**, where `f` is the whole 43-layer dense path plus the socket hops (paid once) and `c` is the per-token MoE plus compressor-window writes. This answers design point (g) directly: θ on TT is not "some strange staircase" — its **envelope is affine**, and the staircase comes only from rounding up to a trace bucket. And **`f` and `c` are measurable today**, with no DSpark code written (see §7 and experiment E1 in §9).

The conclusion: **no front-packing, no bucketing on total verify tokens.** Instead, **fixed-K static verify with B×K inside the existing 32-token tile-row budget** gets an end-to-end number in the first phase. There is exactly one genuinely hard problem, and it is the same problem as SGLang's #33412 lineage: **rollback semantics for CSA's 4-token compression window under partial acceptance** (§5.4).

---

## 1. Scope

**In scope**: this design document, a decision log, open questions, a staged plan, a baseline definition.
**Out of scope**: no kernels, no changes to implementation code.

Discipline on performance claims: every quoted number carries its source hardware and configuration; everything derived here is labelled **derived** and shown with its arithmetic; everything the code does not answer goes into an unknown with a minimal experiment that could falsify it.

---

## 2. Phase 0: state of the code

### 2.0 One-line summary

DeepSeek-V4-Flash on `smanoj/ds_v4_flash` is a **pipeline-parallel (one submesh per chip), fully traced, paged-KV, one-token-per-step** decode engine, plus a DSpark drafter that is **written but not wired in at all** (a PyTorch reference plus a ttnn port). There is no prefill op; a prompt is fed by replaying the decode trace once per token (`README.md:4-6`).

### 2.1 Q1 — is decode traced or eager? Where is the trace created and replayed, and what are the buckets keyed on?

| Item | Answer | Evidence |
|---|---|---|
| Traced? | Yes, traced by default; an eager fallback exists | `tt/generator.py:92` `traced: bool = True`; `--no-trace` (`README.md:112`) goes through `model.decode()` and is **single-user only**, `tt/generator.py:224-228` |
| Capture | `_capture_traces()`, **lazy** (on the first `decode_traced_async`) | `tt/model.py:2093` `tid = ttnn.begin_trace_capture(device, cq_id=0)`; `tt/model.py:2096` `ttnn.end_trace_capture(...)`; trigger at `tt/model.py:2159-2160` |
| Replay | `replay_traced()`, **per-submesh, in order, non-blocking** | `tt/model.py:2201` `ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)` |
| Bucket (variant) key | **Not batch size and not token count, but `(SDPA mode, compressor pooling phase)`** | `tt/model.py:722` `def _variant_key(self, pos: int) -> tuple[bool, int]` |
| Number of variants | **5** globally; fewer after per-submesh dedup | `tt/model.py:2185-2188`: "With the default rates that is five (three causal phases — pool nothing / CSA / CSA+HCA — plus the two masked phases reachable below the sliding window)" |
| Where the phases come from | `(pos+1) % cr == 0` over `crs=[4,128]`, period `lcm=128`; since 4\|128, an HCA closure always coincides with a CSA closure, so there are 3 phases rather than 4 | `tt/model.py:690-710`, esp. `694-699` |
| Per-submesh dedup | A submesh only distinguishes what its own layers observe; a sliding-only submesh captures exactly one | `tt/model.py:746-752` `_sm_pool_key` |

**Inside the traced region**: the embedding (`tt/model.py:1864`), all 43 layers, the mHC head + norm (`tt/model.py:1891`), **the LM head** (`tt/model.py:1892-1893`; "folded into the last submesh's trace so a step returns logits directly", `tt/model.py:1388-1389`), the KV-cache writes, the compressor pooling, the **cross-submesh socket send/recv**, the **H2D packet receive** and the **D2H output send**.

**Outside it** — exactly three things:
1. `_write_packet()`: one direct PCIe write into the socket FIFO, **not command-queue work**, issuable before or during the replays (`tt/model.py:1937-1949`);
2. `read_decoded_output()`: a blocking read off the D2H socket (`tt/model.py:1978-1993`) — **the only synchronization point in a step**;
3. `ensure_session_capacity()` → `_write_page_tables()`: only when some group needs a new block, i.e. once every `compress_rate × block_size` tokens (`tt/model.py:939-952`; the write is `ttnn.copy_host_to_device_tensor` at `tt/model.py:989`).

**What the host writes per step**: only that one packet — `[1,1,1,16]` INT32 ROW_MAJOR, laid out as `[tokens(B) | pos_sliding(B) | pos_compress(B)]`, width rounded up to the PCIe alignment (64 B) (`tt/model.py:1460-1487`).

**A three-stage pipeline already exists**: `write_step_packet` / `replay_traced` / `read_decoded_output` are split apart deliberately, and the docstring says they *"touch disjoint state, so they can run concurrently (on separate threads) for different steps: push step n+1's packet while step n's traces replay and step n-1's output is read back"* (`tt/model.py:2172-2175`). **This is the ready-made infrastructure for overlap on TT.**

**Cap on steps in flight**: the D2H FIFO is one pinned host page (`d2h_fifo_bytes: 4032`, `configs/system_configs.yaml:81`). With single-threaded dispatch, *"on a 43-layer stack across eight submeshes: four steps in flight run, six wedge"* (`tt/model.py:2137-2143`); moving the readback onto its own thread lifts the limit (`tt/model.py:2145-2149`).

**Capture cost**: the README says startup runs one throwaway step so kernel compilation and trace capture ("minutes of it") happen first (`README.md:92-95`). **No number for the seconds or the trace DRAM appears anywhere in the code** → unknown U1. Circumstantial: `tt/system_config.py:71` has `trace_region_size: int = 0` (ttnn's dynamic allocation), while mainline budgets 70–100 MB for comparable LLMs on Blackhole (`models/model_trace_region_sizes.yaml:19-20`, `128-129`), with a single 250 MB outlier. **A Blackhole chip has 32 GB of DRAM, so the trace region is not the scarce resource; capture time is.**

**On "no prefill op"**: confirmed. `README.md:4-6` and `tests/test_full_model_decode_demo.py:6-11` both state it; the server feeds prompt tokens through the same traced step (`demo/server.py:1103-1125`), just dispatched `prefill_chunk=16` at a time (`demo/server.py:828`). A P-token prompt costs **P decode steps**.

### 2.2 Q2 — is the KV cache paged or contiguous per user? Block size? How are the several block geometries handled?

**Paged**, and it is the most solidly designed part of this codebase.

| Item | Answer | Evidence |
|---|---|---|
| Layout | One block pool per layer, `[num_blocks, 1, block_size, head_dim]`, DRAM, **bfloat16** | `tt/model.py:1308-1324`; dtype at `1320` |
| head_dim | 512 (MQA; K==V share a single KV head) | `configuration_deepseek_v4.py:137`; `tt/attention.py:1141` "``kv`` is the shared K==V ``[B, 1, Skv, Dh]`` (MQA, one KV head)" |
| Page table | `[batch, logical_blocks]` INT32, **one per layer_type**, a persistent buffer whose address the trace bakes in; a session switch only rewrites its contents | `tt/model.py:1327-1331`; `tt/paged_cache.py:246-256` |
| Block 0 | Reserved all-zero block; every unmapped tail points at it | `tt/paged_cache.py:33-36`, `tt/model.py:1310-1312` |
| Several geometries | **Already solved.** Grouped by `layer_type`, each group with its own `block_size` counted in *rows of that layer's KV axis*, not tokens | `tt/paged_cache.py:60-101` `PagedGroup` |

**How the three geometries coexist** (`tt/paged_cache.py:16-31`, `155-165`):
- `sliding_attention`: the axis is just the `sliding_window=128` ring, wrapped by `cache_position_modulo`, so a session only ever needs `128/block_size` blocks.
- `compressed_sparse_attention` (cr=4) / `heavily_compressed_attention` (cr=128): the axis is `[sliding ring | one row per closed window]`, addressed by **already-wrapped** indices, so no modulo applies.
- `rows_per_block(tokens_per_block, cr, W) = tokens_per_block // cr` (`tt/paged_cache.py:157-166`) — **each group's rows-per-block is scaled down by its compress rate so that every group consumes exactly one block per `tokens_per_block` tokens of context.** At the default rates {4, 128} the HCA block is 1/32 the size of the CSA one. This is precisely the brief's "they do not share a block size"; the code reconciles them through a common *context-per-block*.
- `min_tokens_per_block(rates) = TILE_SIZE × max(rates)` = 32×128 = **4096 tokens/block** (`tt/paged_cache.py:171-175`), because the paged ops floor a block at one tile of rows.
- Alternatively `block_size` (the same row count for every group), default `block_size=32` (`tt/model.py:1375`, `tt/generator.py:219`).

**The brief's "MLA latent stored fp8/uint8, sparse indexer group stored fp32" does not exist in this code**: the pools are uniformly bfloat16 (`tt/model.py:1320`), and **there is no indexer group at all** (see Q3). This premise needs correcting against the code.

**The op that writes the cache each step**: `ttnn.experimental.paged_update_cache(pool, row_sharded, update_idxs_tensor=pos_tensor, page_table=..., cache_position_modulo=...)`, with `row` as `[1,B,1,F]` and `pos_tensor` as `[B]` INT32 (`tt/attention.py:441-468`). **One independent index per slot — exactly what a multi-token verify needs.**

**Can more than one token be written at once? Is there any rollback?**
- Writing multiple tokens: **`paged_update_cache` itself can** (one row per slot, slot count = B), but the layer above blocks it: `tt/attention.py:1378` `assert s == 1, f"decode attends one token per user, but S == {s}"`.
- Rollback / invalidate: **none at all.** Grepping `rollback|invalidate|truncate` under `tt/` returns nothing. The closest thing is `reset_session()` (rewind the whole session to position 0, freeing its compressed blocks and clearing its window state, `tt/model.py:864-882`) — session granularity, so it cannot roll back k tokens.

**Per-token KV footprint (derived)**: pools are bf16 with head_dim=512 → **1024 B/row**.
- CSA layer: 1 row per 4 tokens → **256 B/token/layer**
- HCA layer: 1 row per 128 tokens → **8 B/token/layer**
- Sliding layer: the ring is bounded, so **0 B/token** at long context (a fixed 128 rows = 128 KiB per layer per session)

The default `layer_types` (`configuration_deepseek_v4.py:262-267`: two HCA to bootstrap, then alternating) gives **23 HCA + 20 CSA + 0 sliding** at n=43:
`20×256 + 23×8 = 5120 + 184 = 5304 B/token ≈ 5.18 KiB/token` across all 43 layers, spread over the chips that own them.
1M of context ≈ **5.56 GB**; `galaxy32_server`'s `total_context: 16777216` (`configs/system_configs.yaml:289`) ≈ **89 GB**, or ≈ 2.8 GB/chip across 32 chips.
⚠️ This layer-type mix is what the **default construction** produces; if the 0731 `config.json` states `layer_types` explicitly, that wins → unknown U2.

### 2.3 Q3 — is CSA's top-k indexer on device or host? How many round trips per layer per step?

**Neither: the Lightning Indexer is not implemented on the ttnn side at all.**

- The HF reference has it: `modular_deepseek_v4.py:382` `class DeepseekV4Indexer` (paper §2.3.1 eqs. 13–17), with `index_topk: int = 512`, `index_n_heads: 64`, `index_head_dim: 128` (`configuration_deepseek_v4.py:162-164`).
- The ttnn side does not: grepping `topk|indexer` under `tt/` returns nothing, and `WEIGHT_PLACEMENT.md:16` explicitly lists "Lightning Indexer (never loaded by `tt/`)" under **NOT resident**.
- What CSA actually does today is **dense SDPA over every closed compressed entry**, and the reasoning is stated at `tt/attention.py:643-647`: *"for `seq_len <= index_topk * compress_rate` its top-k selects every entry, so the block_bias reduces to plain causal masking over windows"*.

**Consequence (important, and not among the brief's premises)**: `index_topk × compress_rate = 512 × 4 = **2048** tokens`. **Beyond 2048 tokens of context, the CSA layers are no longer equivalent to the reference model** — they attend to entries that top-k should have discarded. This is a silent accuracy loss, not an error. → unknown U3, and it will **confound any long-context accept-rate measurement** (drafter and target both run on the same deviating implementation, so accept rate could be over- or under-stated).

**Host↔device round trips**: across the per-layer decode path in `tt/model.py`, `tt/attention.py`, `tt/layers.py` and `tt/moe.py`, **`ttnn.to_torch` appears zero times** (the only `to_torch` is on the eager `--no-trace` branch, `demo/server.py:1160`). The whole host traffic of a traced step is one packet write plus one output read. **Zero round trips per layer per step.**

**The 4-token boundary state machine (the most important paragraph in this document)**, `tt/attention.py:634-799` and `100-127`:
Each layer holds, per user:
- `win_kv` / `win_gate` `[B,1,4,2·Dh]` — the current window; a token is written at slot `pos % 4`;
- `prev_kv` / `prev_gate` `[B,1,4,2·Dh]` — the **previous** window (**CSA only**; HCA has none);
- `combined` `[B,1,128 + max_seq/4, Dh]` — `[sliding ring | compressed entries]` on one axis.

Every step: `_update_cache_at(win_kv, kv, win_slot)` writes this token's projection (`tt/attention.py:784-785`).
On the step that closes a window (`pool=True`): `_pool_window(prev, win)` combines **window w−1's Ca half** with **window w's Cb half** as a softmax-weighted sum over a width-`2·cr=8` axis → one entry → written into `combined` row `128+w`; then **`_retire_window(prev, win)` copies win into prev in place** (`tt/attention.py:786-793`).

**Two properties here are fatal for DSpark:**
1. **A CSA entry spans two windows** (the Ca/Cb overlap, `tt/attention.py:637-641`), so rolling back an entry requires restoring both `win` and `prev`.
2. **`_retire_window` is destructive** (`tt/attention.py:482-497`, an in-place overwrite via `ttnn.fill_cache`). **Once a speculative step closes a window, the old `prev` needed to rebuild it is gone forever.** This is the same family of problem as the "c4 compressor store" IMA in sglang #33412, seen from the other side (see §5.4).

HCA's 128-token boundary is the same but simpler: there is no `prev` (`tt/attention.py:499-508`); an entry is pooled from the current window's 128 tokens alone.

**Does any shape change with the number of tokens per step?** No. The mask is `[1,1,1,Skv]` with `Skv = 128 + max_seq/cr` fixed; in causal mode there is no mask at all, only `cur_pos [B]` (`tt/attention.py:1147-1157`).

### 2.4 Q4 — where does sampling happen? Does decode materialize only the last position's logits?

| Item | Answer | Evidence |
|---|---|---|
| Sampling | **Host.** There is no device-side argmax / top-k / multinomial anywhere | `demo/server.py:438-449` (`torch.topk` + `torch.multinomial`); `demo/server.py:1092` (`int(out[0].argmax().item())`); `tt/generator_vllm.py:43` `"supports_sample_on_device": False` |
| Positions of logits | Decode produces **one position per user**, so only the last one is materialized by construction | `tt/attention.py:1378` `assert s == 1` |
| Logits shape | `[batch, 1, vocab]`, dtype = the device dtype (bf16) | `tt/model.py:2110-2112` |
| LM head inside the trace | Yes | `tt/model.py:1892-1893`; `tt/model.py:1388` |
| How logits reach host | `send_async_d2h`, after untilize + reshape into PCIe-aligned rows of `[1,1,-1,2020]` | `tt/model.py:1952-1976`; the width `2020` at `1970` |
| Vocab | 129280 | `configuration_deepseek_v4.py:131` |
| LM-head weights | 129280×4096 = 529.53M params; at bf4 (0.5625 B/elem, `tt/l1_placement.py:44` `BF4_TILE_BYTES = 576`) → **297.9 MB** (derived) | `tt/generator.py:180-184`, dtype = `weight_dtype` |
| Where it lives | `model.last_device` | `tt/generator.py:182` |

**Would N>1 positions of logits work as-is?** Partly. `_pack_tokens` already handles `[B,S,1,D] → [1,1,B·S,D]`, and the output is reshaped back to `[b,s,1,d]` (`tt/attention.py:1379`, `1440`); the D2H `_d2h_page_plan` pages by numel (`tt/model.py:1955-1962`) and `read_decoded_output` reshapes to `[batch,1,-1]` (`tt/model.py:1994-1996`). **The only place that hard-codes "1 token" is the assert at `tt/attention.py:1378`, plus the scalar RoPE / mask / pool assumptions around it** (enumerated in §5.1).

**How the sampled token becomes the next step's input**: the host samples, then writes it into the next packet (`demo/server.py:1092` → `demo/server.py:1154` → `tt/model.py:1937`). **That is one host round trip per step, on the critical path.**
The code has already tried to bypass it and failed: `decode_sampled_burst()` raises `NotImplementedError` (`tt/model.py:2203-2219`) because *"the socket packet overwrites the device-sampled token slot"*, and it states the fix: *"splitting the packet: keep the host-fed positions on the socket and read the token from a separate device-written buffer"* (`tt/model.py:2213-2214`). **That comment is the feasibility argument for the fully on-device speculation loop in §5.3.**

### 2.5 Q5 — scheduling in `demo/server.py`

`demo/server.py` is 1945 lines; the core is `_Scheduler` (`demo/server.py:805`).

| Item | Answer | Evidence |
|---|---|---|
| Continuous batching | **Yes, but it is not batching — it is round-robin interleaving.** New requests are admitted at a round boundary; a finished slot is immediately reusable with no re-capture or re-warm | `demo/server.py:859-867` `submit()`; `demo/server.py:949` `_admit()`; `demo/server.py:1063-1078` `_round_turns()` |
| Batch dimension | **`batch: 1`** in the server profiles (`galaxy32_server` extends `galaxy32`, whose `decode.batch: 1`) | `configs/system_configs.yaml:254`, `282-289` |
| Overlap / double buffering | **Yes.** A turn's output is only read at the start of its *next* round, by which point every other turn's step is already queued behind it | `demo/server.py:815-820`: "the host never waits on one turn before feeding the next, and device work overlaps host work" |
| Host↔device sync points per decode step | **Two**: ① the `_write_packet` PCIe write (non-blocking, not command-queue work, outside the trace); ② the blocking D2H socket read in `read_decoded_output` (outside the trace, but deferred to the next round by the pipeline). Plus a **conditional** third: `ensure_session_capacity` → `_write_page_tables`, once per `cr × block_size` tokens | ① `tt/model.py:1937-1949`; ② `tt/model.py:1978-1993`; ③ `tt/model.py:939-952`, `985-990` |
| Slot model | `_SlotPool`, slot count = `decode.num_users`; `galaxy32_server` sets 32 | `demo/server.py:546-580`; `configs/system_configs.yaml:287` |
| Idle slots | Not wasted — it is interleaving, not batching, so an empty slot is simply a turn that is not dispatched | `demo/server.py:1063` iterates only `self._active` |
| Prefill | Replay decode per token, dispatched `prefill_chunk=16` at a time (prompt-token logits are discarded, so they carry no step-to-step dependency and can all be in flight) | `demo/server.py:822-826`, `1103-1125` |
| Queue / preemption / eviction | An admit queue (`_new` deque) and a capacity check exist; **no preemption, no eviction** | `demo/server.py:833`, `1023-1032` |

**The cost of switching session every step (important for DSpark)**: `_send` calls `activate_session` on every step (`demo/server.py:1146-1152`), which repoints the page tables and swaps in that session's compressor window state. The code quantifies it: *"a per-session swap moves one row per slot per compressor buffer, which on a 43-layer stack is thousands of device ops on the critical path of every step (**measured at ~100 ms**, more than the step itself), while a block swap is one copy per buffer regardless of batch"* (`tt/model.py:912-917`). So it does a **whole-group swap** (a seat group), at the price that co-seated sessions must enter and leave together **and must sit at the same position**.

**Throughput numbers**: see §7. `README.md:1` titles 15.9 tok/s/u; the last row of `PERFORMANCE_LOG.md` reports **16.2 tok/s/u** (commit `a4b967209214`, 8× BH P150, batch=1, 43 layers).

### 2.6 Q6 — is the server on TT's vLLM fork generator interface, or bespoke?

**`demo/server.py` is entirely bespoke; `tt/generator_vllm.py` is a parallel adapter that nothing currently imports.**

- `demo/server.py` holds a `ChatEngine` → `DeepSeekV4Generator` → `DeepSeekV4Model` directly and drives the trace itself (`demo/server.py:1154` calls `model.decode_traced_async`).
- `tt/generator_vllm.py` is the "Phase 1: functional bringup" vLLM adapter (`tt/generator_vllm.py:4`), mapping vLLM's batched forward onto the generator's per-slot sessions: *"Prefill replays decode one token at a time. Sampling is done on host by vLLM"* (`tt/generator_vllm.py:9-11`).
- **It explicitly declares speculative decoding unsupported**: *"prefix caching, async decode, on-device sampling, **speculative decode** and LoRA are all unsupported"* (`tt/generator_vllm.py:23-25`), and all three `model_capabilities` entries are `False` (`tt/generator_vllm.py:41-44`).
- Registration requires editing the plugin clone's `model_registry.py` (`tt/generator_vllm.py:14-22`), which lives **outside this repo**.

→ **This design does not depend on the vLLM path.** Phase A lands in `demo/server.py` plus tests; vLLM integration comes after Phase C.

### 2.7 Q7 — weight requantization; the size, placement and shareability of embedding / lm_head

| Item | Answer | Evidence |
|---|---|---|
| Requant entry point | `dequantize_weight(tensor, scale)` dequantizes the checkpoint's fp8/fp4 to torch, then `ttnn.as_tensor(..., dtype=bfloat4_b)` requantizes | `tt/quant.py`; call site `tests/test_full_model_decode_demo.py:69` (the `_w` thunk) |
| Target dtype | `bfloat4_b` for the whole model (`_WEIGHT_DTYPE = ttnn.bfloat4_b`, `tests/test_full_model_decode_demo.py:52`); norms / position_bias / sinks stay bf16 (`WEIGHT_PLACEMENT.md:15`) | as left |
| Offline or at load | **Converted at load, cached to disk.** The first run takes over an hour; `DEEPSEEK_V4_CACHE_DIR` reuses it | `README.md:46-47`; `tt/weight_cache.py` |
| bf4 density | 576 B / (32×32) = **0.5625 B/elem** | `tt/l1_placement.py:44` `BF4_TILE_BYTES = 576  # 512 B of 4-bit mantissas + 64 B shared exponents` |
| Embedding | 129280×4096 = 529.53M params, **bfloat16 + ROW_MAJOR** (required by `ttnn.embedding`) → **1.059 GB** (derived) | `tt/embedding.py:22-29` |
| Embedding placement | **`self.first_device`** (submesh 0) | `tt/model.py:365` |
| LM head | The same 529.53M params at bf4 → **297.9 MB** (derived) | `tt/generator.py:180-184` |
| LM-head placement | **`model.last_device`** (the last submesh), folded into that submesh's trace | `tt/generator.py:182`; `tt/model.py:447` |
| Expert weights per layer | 256 experts × 3 × (4096×2048) = 6.442B params → **3.62 GB/layer @ bf4** (derived, matching the doc) | `WEIGHT_PLACEMENT.md:16` "Routed experts (3.62 GB/layer)" |
| Non-expert per layer | 75.25 MB (HCA) / 77.50 MB (CSA) @ bf4 | `WEIGHT_PLACEMENT.md:20-21` |
| Whole model | 43×3.62 + 43×0.076 + 1.06 + 0.30 ≈ **160.5 GB** (derived). 8 chips → 20.1 GB/chip (of 32 GB); 32 chips → **5.0 GB/chip** | derived |

**Can the drafter reuse the same physical embedding / lm_head?**
- **LM head: yes, and almost for free.** It already sits on `last_device`, and the target hidden states the drafter needs come from layers 40/41/42 (`dspark.py:142` `dspark_target_layer_ids=(40, 41, 42)`), i.e. the end of a 43-layer stack, which lands on the same last submesh or two (`plan_layer_placement`, `tt/model.py:62-92`).
- **Embedding: not directly, because it lives on `first_device` while the drafter must run on `last_device`.** Three options: (i) a second bf16 copy on `last_device`, **+1.059 GB/chip**; (ii) drop the drafter's embedding to bf8 (does `ttnn.embedding` accept a bf8 row-major table? → unknown U4); (iii) have the target's submesh-0 embedding also embed the drafter's anchor token and ship it down the existing device-to-device socket — **but that adds a whole pipeline depth and drags the drafter back onto the critical path**. → option (i), see D-06.
- The ttnn drafter currently owns its own embedding and its own LM head (`tt/dspark.py:355-367`); sharing is **not implemented**. The PyTorch reference defines the semantics via `share_from_target(embed_tokens, lm_head)` (`dspark.py:389-398`).

### 2.8 Q8 — mesh configuration and the TP/EP/DP split

**In one line: pipeline parallelism only. One submesh is one chip. No TP, no EP, no DP.**

| Item | Answer | Evidence |
|---|---|---|
| Submesh shape | `MeshShape(1, 1)` — **a submesh is one chip** | `tt/model.py:327` |
| How the mesh is opened | `mesh_shape: null` by default = the whole system flattened to a 1×N line | `configs/system_configs.yaml:48-51` |
| Layer→chip map | `plan_layer_placement(num_layers, num_devices, group_size)`; `group_size=1` = one contiguous slice of layers per chip | `tt/model.py:62-92`; `configs/system_configs.yaml:190-191` |
| Chip-to-chip link | One socket pair per adjacent submesh, `send_direct_async` / `recv_direct_async`, **device-to-device with no host round trip**, 16 KB of L1 per socket | `tt/model.py:337-358`, `1157-1169`; `configs/system_configs.yaml:71-72` |
| Fabric | `FABRIC_2D` (required by the submesh pipeline sockets) | `configs/system_configs.yaml:40-41` |
| Attention TP | **None.** All heads of a layer live on one chip | no mesh mapper and no CCL in `tt/attention.py` |
| MoE EP | **None.** Each chip holds **all 256 experts** for the layers it owns, DRAM ND-sharded across 8 banks and 64 cores | `tt/moe.py:451`, `481-507`; `configs/system_configs.yaml:105-106` |
| Collectives per decode step | **Zero** (no all-reduce / all-gather / reduce-scatter). The only cross-chip traffic is activation handoff on the pipeline sockets | grepping `all_reduce\|all_gather\|reduce_scatter` under `tt/` returns nothing |
| DP / replication | None | as above |
| Batch semantics | Batch is **not sharded across chips**; every chip processes all B users (B ≤ 32, packed into one tile-row) | `tt/attention.py:337-350` |

**Profile table** (`configs/system_configs.yaml`):

| profile | line | num_devices | batch | num_users | max_context | total_context | PGS | depth |
|---|---|---|---|---|---|---|---|---|
| `p150x8` | 185 | 8 | 1 | 2 | 131072 | 0 | 1 | 0 |
| `p150x8_server` | 271 | 0 (variant) | 1 | 8 | 524288 | 2097152 | 1 | 0 |
| `p150x8_throughput` | 295 | 0 (variant) | 8 | 64 | 32768 | 2097152 | 1 | 0 |
| **`galaxy32`** | **225** | **32** | **1** | **8** | **524288** | **4194304** | **1** | **4** |
| `galaxy32_server` | 282 | 0 (variant) | 1 | 32 | 524288 | 16777216 | 1 | 4 |
| `galaxy32_throughput` | 305 | 0 (variant) | 8 | 256 | 32768 | 8388608 | 1 | 4 |
| `single_chip` | 318 | 1 | — | 1 | 8192 | — | 1 | — |

**The `galaxy32` profile carries a decisive self-assessment** (`configs/system_configs.yaml:210-224`):
> *"**STARTING POINT, NOT A MEASURED TUNE.** … with the pipeline-parallel-only implementation, 32 chips does not by itself improve tok/s/user — per-token latency is the sum of all 43 layers' compute wherever those layers live, and 4x more chips means ~1-2 layers per chip plus ~31 socket hops instead of ~5 layers and 7 hops. What 32 chips does buy immediately is capacity: ~4x the DRAM for KV cache … Turning the extra chips into per-user speed needs intra-stage tensor/expert parallelism (8 pipeline stages x 4-chip submeshes with the 256 experts sharded 64-per-chip), which is **not implemented yet**."*

That paragraph is the entire basis for design point (e), and it implies that **DSpark's benefit structure on Galaxy differs from 8 chips** (§5.5).

### 2.9 Q9 — tt-metal mainline: a vLLM overlap scheduler? Issue #50475?

See **§3** (separate section, because it covers mainline rather than this branch).

### 2.10 Addendum: the state of the DSpark drafter (not in the brief, but the highest-value fact about the current state)

**The drafter is already written twice, and wired to the target zero times.**

| File | What it is | Key points |
|---|---|---|
| `dspark.py` (514 lines) | **PyTorch reference**, standalone and unit-testable | header at `dspark.py:1-27`; `flash_0731()` at `dspark.py:128-145` |
| `tt/dspark.py` (535 lines) | **ttnn implementation** using `LinearDecode` + the DRISC prefetcher | `tt/dspark.py:1-19` |
| `tests/test_dspark.py` (243) / `test_dspark_ttnn.py` (56) | Unit tests | — |

**The 0731 DSpark geometry** (`dspark.py:128-145`, transcribed by the author from `config.json`):
```
hidden_size=4096      num_stages=3          num_target_layers=3
num_attention_heads=64  head_dim=64        intermediate_size=2048
vocab_size=129280     sliding_window=128
dspark_block_size=5   dspark_markov_rank=256
dspark_noise_token_id=128799
dspark_target_layer_ids=(40, 41, 42)
```

⚠️ **One conflict with the brief**: the brief says the parallel backbone has **5 layers**; the code says `num_stages=3` (`dspark.py:133`) with `dspark_block_size=5` (`dspark.py:139`). By the code, the 5 in "DSpark-5" is the **draft block length γ**, and the backbone is **3 stages**. **§4.6.5 shows the paper agrees with the code**; what remains is a checkpoint cross-check (U5', experiment E5). This document takes `num_stages=3, γ=5`.

**The algorithm** (`dspark.py:6-24`, `427-500`):
1. Fuse target layers 40/41/42's hidden states into a context via `main_proj`/`main_norm` (`dspark.py:400-415`);
2. Inject that context as extra K/V; the draft block is `[anchor, noise, noise, noise, noise]` (`dspark.py:416-426`, noise id 128799), and **attention inside the block is bidirectional** (`dspark.py:226-244` `dspark_block_mask`: every block query sees the last 128 context tokens plus the whole block);
3. One forward produces γ positions of hidden → `lm_head` → `base_logits`;
4. **The Markov head runs γ steps serially**: each step looks the previous sampled id up in `markov_w1 [vocab, 256]`, passes it through `markov_w2 [256, vocab]` to get a logit bias, adds it to `base_logits[k]` and takes an argmax (`dspark.py:487-500`, `tt/dspark.py:508-535`);
5. **The confidence head** emits a sigmoid confidence per position; `prefix_survival = cumprod(confidence)`, and `truncate_prefix(conf, min_survival)` takes the **longest prefix whose survival stays above the threshold**, length ∈ 1..γ, with causal truncation (everything after the first drop is discarded) (`dspark.py:162-179`).

**Why this is unusually friendly to TT, and worth stating separately**:
- The draft **block length is fixed at γ=5**; it is semi-autoregressive, not a tree, so the drafter's input and output shapes are **constant**;
- Variability appears in exactly two places: the **host-side confidence truncation `k_send ∈ 1..5`**, and the **post-verify accept length `k_acc ∈ 1..k_send`**.

**Problems with the current ttnn port (all integration work, not architectural blockers)**:
- `forward()` takes `target_hiddens` as a **torch tensor** (`tt/dspark.py:446`, with `ttnn.from_torch` inside) → a device→host→device round trip;
- the γ-step Markov loop does **4 host round trips per step** (`from_torch` of the prev id, `to_torch` of the bias, `to_torch` of the confidence, host argmax), so **≈20 round trips per round** at γ=5 (`tt/dspark.py:508-535`);
- `batch != 1` raises `NotImplementedError` (`tt/dspark.py:465-466`);
- **no trace at all**;
- **the confidence head is padded to `[vocab, hidden+rank]` but only row 0 is used** (`tt/dspark.py:405-413`) — a 129280×4352 matmul to produce one scalar, i.e. **562.6M params of wasted weight** (derived). This must be fixed (D-08).

---

## 3. Phase 0 addendum: Q9 — tt-metal mainline

Checked this session with an authenticated `gh` (2026-08-26).

**Overlap scheduler**: mainline's `tech_reports/LLMs/vLLM_integration.md` is 68 lines of integration guidance and **does not describe overlap / async scheduling**. tt-metal's vLLM integration goes through a fork (`tenstorrent/vllm`) plus `vllm-tt-plugin`, with models declaring capability via `model_capabilities` (`supports_async_decode` and friends); DeepSeek-V4-Flash has all three `False` (`tt/generator_vllm.py:41-44`).
→ **There is no directly reusable overlap scheduler at the mainline level.** That does not matter for this design: **we do not need one**, because `tt/model.py`'s own three-stage pipeline (`write_step_packet` / `replay_traced` / `read_decoded_output`, `tt/model.py:2172-2175`) already provides the equivalent and fits trace semantics better.

**Issue #50475 (qwen3.6 multi-token GDN decode/verify kernel)**, and whether mainline already has a multi-token paged-cache update or a ragged attention op: see unknown U9 in §9 — the evidence chain the retrieval returned for this one is weaker than the others, so no conclusion is drawn here. **How to verify**: `gh issue view 50475 --repo tenstorrent/tt-metal --comments`, plus `git grep -n 'paged_update_cache\|paged_fill_cache' origin/main -- ttnn`.

**Existing speculative-decoding work on mainline**: `origin/dchrysostomou/experiment_speculative_deepseek`, `origin/smanoj/ds_v4_dspark` (2026-07-04, the older fork of this branch, carrying the original "Implemented DSpark, but not integrated"), and deepseek_v3's MTP branches (`origin/yieldthought/deepseek-mtp-validation`, `deepseek-mtp-max-seq-len`). **deepseek_v3's MTP path is the closest internal precedent and is worth reading before starting** (out of scope here).

---

## 4. Phase 1: targeted research

Each source is reduced to "what constraint does this place on our design, or what conclusion can we reuse". Every quotation was retrieved this session (2026-08-26); retrieval status is stated per source.

### 4.1 LMSYS blog "DSpark in SGLang" (2026-07-06) — ✅ retrieved
`https://www.lmsys.org/blog/2026-07-06-dspark-sglang/`

1. **Ragged verify is keyed on the total token count, not batch size**: *"we keep the batch ragged and key the graph on the total token count — front-pack the variable-length requests into one compact buffer and round up to the nearest captured tier"*. This part **does** port, because the frozen quantity collapses to a single scalar.
2. **But the precondition that makes it free is one we lack**: *"The packed buffer is a cu_seqlens-style varlen input, so the compact verify reuses attention kernels the backend already has — on DeepSeek-V4 the model's own sparse-MLA path (flash_mla), with no new kernel"*. **TT-NN has no varlen/cu_seqlens op for the CSA/HCA path**, so "port it verbatim" silently smuggles in "first write a varlen sparse-MLA for Blackhole".
3. **The tier ladder is not tuned**: `buckets = sorted({bs * captured_req_width for bs in capture_bs})` — mechanically the existing CUDA-graph batch ladder times the per-request width. For us that means trace count = tier count, and the ladder is coarse at the top.
4. **Under DP, every step costs a CPU all_gather to agree on the tier.** We are **structurally exempt**: the whole mesh is driven by a single host scheduler, so the tier is decided once.
5. **Two-step-back confidence relay**: a pinned ring plus a non-blocking event query; if the copy has not landed it degrades to survival = 1.0 (full budget). **Fail-safe by construction** — staleness mis-sizes the budget, it never corrupts output.
6. **Provenance must be pinned**: 383.7 tok/s @ accept≈5 is **B300 TP8, V4-Pro, bs=1, `--cuda-graph-max-bs 4`**; the trimming curves are **H200 DP4, V4-Flash, batch 1..256**. **The two cannot be chained into one story**, and the prose gives no DSpark-over-MTP ratio (it lives only in the figures).

### 4.2 sglang PR #30261 (MERGED 2026-07-12, +17700/−287, 84 files) — ✅ retrieved

1. **A tier miss is not a graceful pad; it raises.** `round_up_grid` raises above the top tier: "the caller must reject this batch before selecting a graph tier". **TT has no cheap eager fallback** (a 1M-context DSV4 target step), so we must **guarantee by construction that the top tier is never exceeded**, rather than rejecting at runtime.
2. **Per-request verify lengths never round-trip to host**: verify_lens live only on device; tier selection uses only host-known scalars (bs + the integer budget). There is a slow D2H fallback `verify_lens.to("cpu").tolist()`. **This split is what makes DSpark graph/trace-compatible at all, and must be copied.**
3. **With ragged batches, the attention metadata is rebuilt outside the graph every step**: *"the fused in-graph kernel cannot express (scalar `seqlen_offset` only), so this runs out-of-graph on every capture/replay-prep"*. **That is not viable on TT** — anything outside a trace is a host-issued op costing a full dispatch. We must build the metadata inside the trace (computed from persistent input tensors), or accept one pre-trace write.
4. **Bucketing alone buys nothing**: without a profiled SPS table the budget degenerates to verify-all, "zero throughput gain by itself". **The SPS table is a prerequisite deliverable.**
5. **There is a flag that turns tier padding into real verification**: `--speculative-dspark-align-verify-tokens-to-graph-tier` (off by default) — round the total up to the tier, then let the confidence-ordered top-k allocator spend the headroom on additional real draft tokens, "at the same step time". **Our tiers are coarser, so the payoff is larger.**
6. **Mutual-exclusion list**: compact is incompatible with two-batch-overlap, LoRA, disabled graph padding, the FlashInfer graph path and context-parallel; and `SGLANG_PREP_IN_CUDA_GRAPH=1` is mandatory.

### 4.3 sglang issue #33412 + root-cause PR #32467 — ✅ retrieved (**the brief's read of the root cause needs correcting**)

1. **The root cause is not the 4-token boundary.** It is a write-write race in `plan_compress_prefill_kernel0`: warp 0 initializes every `warp_min`/`warp_max` scratch slot while other warps later write their own, with no barrier in between → **a genuinely ragged `extend_lens` looks uniform**, which selects the MTP fast path `ragged_id = batch_id * s_max + j` and emits indices up to `B*s_max-1 = 383` when the real row count is 360. The c4 compressor's store is only where it **surfaces**.
2. **The "partially-filled compression block / padded-vs-unpadded boundary index" hypothesis was explicitly tested and refuted by the reporter**: *"we hardened `fused_norm_rope_v2.cuh` with an `if (position < 0) return;` guard … but the IMA persists — negative `position` from sub-ratio plan entries is not (the only) cause."* Legal `extend_lens` values include 2 and 3 (**not multiples of 4**). → **We do not need to align verify segments to 4, and we do not need to pad each segment to a multiple of 4. Do not spend design budget there.**
3. **Compact is not a throughput win on GPU** (the table in PR #32467 itself: V4-Flash-DSpark / block 5 / TP2 / Marlin MoE / MRR96): bs=1 336.46 vs default 333.14; bs=32 48.12 vs 48.49; bs=64 32.78 vs 33.21; **bs=96 24.75 vs 25.73 (compact 3.8% slower)**. → **The zero hypothesis's implicit premise that porting SGLang is faster is not supported by data. Front-packing can only be justified on trace count / DRAM, never on tok/s.**
4. **The consequence is worse on TT**: TT has no MMU and no IMA trap. What is a loud out-of-bounds access on GPU is, for us, a **silent overwrite of another tensor**, inside a trace, with no host sync point at which to notice. On GPU the fault threshold even drifts with allocator configuration ("shapes that work may be silently corrupting"). → **Every index derived from the verify layout must be clamped/masked on device.**
5. **A reusable fix pattern**: the correct pattern is not "add a barrier" but "never write another warp's scratch slot at all" (the reviewer variant deletes the redundant initialization and also gets 0/30000 out-of-bounds). The same trap applies if we ever do a cross-core min/max reduction on TT.

### 4.4 sglang #30344 / #23602 / sub-roadmap #34297 — ✅ retrieved

1. **#30344 contains no task list** (668 characters of body, five open-ended headings); **the real decomposition is in #34297**: eight contracts C0–C8, all **Open or In progress as of 2026-08-11, none landed**. **Five of the eight are graph-capture / verify-shape problems, not model-math problems.**
2. **C1 = "Verifier state rewrite window" (#32183, Open)**: *"DSpark verifier width -> compressed-state rewrite window -> every verifier-committed row must be reconstructed"*. **That is exactly our §5.4, and upstream has not solved it either.** We cannot expect to copy an answer.
3. **The precise definition of the three stages** (from source, not from the issues): `RaggedVerifyMode = {STATIC, CAP_ACCEPT, COMPACT}`, with `SGLANG_RAGGED_VERIFY_MODE` defaulting to **`static`**. **CAP_ACCEPT is a dense `[bs, cap]` layout with a per-row cap — the shape is still static, so it is inherently trace-compatible**, making it the ideal second stage for TT.
4. **Verify lengths are uniform in real traffic**: a natural 60k in / 1k out workload "produced uniform verify lengths and therefore does not count as ragged coverage"; it took `dspark_force_budget_frac=0.5` to force 90.7% ragged blocks. → **static covers the overwhelming majority of real steps; and a ragged path needs a forcing knob to be testable at all.**
5. **Do not cite SGLang's throughput tables as evidence that speculation pays**: their only published sustained run shows `acceptance length: ~1.00-1.12` (random-token workload, simulated acceptance disabled) — **essentially no speculative gain**. Those numbers are capacity/robustness evidence.

### 4.5 vllm-ascend RFC #11126 (+ upstream vllm PR #46995, merged) — ✅ retrieved (**the most important source for us**)

1. **They explicitly list the confidence scheduler and dynamic verify length as Non-goals**, and ship **a fixed 5 speculative tokens with the drafter's FULL ACLGraph on by default**. They also **implemented and measured** the confidence head: *"the acceptance length has increased by approximately 0.x, but performance has degraded."*
   → **Two independent non-NVIDIA/upstream teams concluded that fixed γ + graph mode beats dynamic γ + eager. Nobody has shipped "full graph mode WITH dynamic verify length".**
2. **With fixed γ there is no variable total verify token count.** Upstream code comment: *"Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs"*; Ascend derives capture sizes "from target request rows". → **The zero hypothesis's bucketing axis (total tokens) is empty under fixed γ; the only free variable is the request-row count, and they pad whole rows, not tokens.**
3. **Component decomposition (usable directly as our WBS)** — 4 PRs, ~4000–4900 lines, on a backend that **already had V4 + MTP**:
   - PR1: non-causal DSA/SAS visibility + `ori_sparse_indices` (~1000 lines)
   - PR2: draft model + target aux-hidden-state extraction + weight loading (~1100–1300 lines)
   - PR3: eager proposer (serial Markov sampling + logits-space rejection)
   - PR4: graph capture
   **The scheduler / KV manager / verify step / rejection sampler all come from the framework; we do not write them.**
4. **No new attention kernel is required**: express the draft block's non-causal visibility as **explicit visible-slot indices fed to the existing sparse attention** (`ori_sparse_indices`), and **refuse to introduce a second request-local KV cache** — the standard paged cache stays authoritative and rejected-token rollback is just slot reuse.
5. **`num_lookahead_tokens = num_spec_tokens` (not +1)**, because the anchor is itself the first prediction position. **This is the classic off-by-one.**
6. **Measured gains** (16 Ascend NPUs, TP4/DP4/EP16, W4A8, V4-Flash-DSpark, γ=5):

   | BS | no-spec tok/s | MTP1 tok/s | DSpark tok/s | DSpark vs no-spec | DSpark vs MTP1 |
   |---|---|---|---|---|---|
   | 1 | 29.15 | 47.71 | 65.53 | **+124.8%** | +37.4% |
   | 4 | 110.41 | 175.60 | 217.80 | +97.3% | +24.0% |
   | 8 | 220.61 | 309.49 | 382.45 | +73.4% | +23.6% |
   | 16 | 410.94 | 579.53 | 619.70 | **+50.8%** | +6.9% |

   AL 3.356 (SPEED-Bench) / 3.935 (coding80); per-position accept rate **[80.73, 67.28, 57.11, 47.70, 40.65]%**.
   → **This is the closest measured analogue to our situation in the whole set: a non-NVIDIA backend, the same checkpoint, γ=5. It sets the realistic expectation at ~2.2× at low concurrency, collapsing to ~1.5× at high concurrency.**
7. **The graph boundary is drawn exactly where a TT trace needs it**: *"the fixed-shape draft forward is captured and replayed with persistent buffers; dynamic input preparation and final output slicing remain outside the graph."* But **the two backends disagree on where the serial Markov loop belongs**: upstream CUDA puts it inside the graph, the Ascend prototype leaves it eager (to avoid nested torch.compile — a reason that does not exist on TT). → **We should put it inside the trace.**

### 4.6 DSpark paper arXiv:2607.05147 — ✅ retrieved

1. **The cost model is a single-argument `SPS(B)`, where B = Σ_r (1+ℓ_r) is the batch-wide total verify token count**, with objective `Θ = τ·SPS(B)`, `τ = Σ_r (1 + Σ_j a_{r,j})`, `a_{r,j} = Π_{i≤j} c_{r,i}`. The table is profiled once at engine initialization.
   ⚠️ **The brief's `T(bs,K)=bias+alpha(bs)+theta(M)` has no source in anything retrieved this session**; the paper uses the form above. This document follows the paper and explains the relationship in §5.7.
2. **The scheduler is a global sort**: order every `(request, position)` prefix-survival descending, admit one token at a time (B += 1), stop when Θ drops. Because `a_{r,j}` is monotonically non-increasing in j, **the global sort automatically respects intra-block prefix dependencies.** → Pure host-side O(Rγ log Rγ), zero device impact. **And this lemma *is* the correctness argument for front-packing**: once a tier pins B, the optimal fill is the top-B tokens by cumulative survival.
3. **§5.2 names both of our conflicts explicitly**: (i) the real `SPS(B)` is **discrete / jagged / step-wise**, so greedy early-stopping gets trapped before a cliff; (ii) per-step dynamic token counts clash with graph replay and ZOS. **Their fix is to make the scheduler asynchronous with a two-step lag, and to remove the early-stopping break as a result.**
4. **The two-step lag is not a performance hack; it is a necessary condition for losslessness.** Removing early-stopping and doing a retrospective global search would leak future information (Appendix A counterexample: with SPS = (1.0, 0.5, 0.45) and a₁ = 0.8 the output distribution becomes (0.85, 0.15) when the truth is (0.7, 0.3)); **the asynchronous two-step lag forms a causal barrier** that buys losslessness back.
   → **Our cliffs on TT are self-inflicted and sparser, so early-stopping is near-certain to get trapped. Hence: "if we do dynamic γ, the lag is mandatory."**
5. **The production V4 drafter is 3 MoE layers + mHC + a 128 sliding window, γ=5**: *"The parallel backbone comprises three MoE layers (Dai et al., 2024) with mHC (Xie et al., 2026) and a sliding window attention of 128. We configure the maximum block size to γ=5"*. **"DSpark-5" means γ=5, not 5 layers.** The "5 layers" figure comes from the offline Qwen3/Gemma4 ablation (*"we set 1 for Eagle3 and 5 for DSpark and DFlash"*). → **U5 resolved: the code's `num_stages=3` is right.**
6. **The 60–85% is per-user generation speed against MTP-1 at matched throughput**, not against non-speculative decoding. The paper itself discounts its largest figure (661%) as "extending the feasible interactivity frontier" rather than a representative multiplier. **No hardware model is named anywhere in the paper.**
7. **The implementation recipe for variable-length verification**: flatten all requests' tokens into one undifferentiated element stream and convey sequence structure only through a **marker tensor** inside the sparse attention; **on DeepSeek-V4 only two kernels needed modification — index-attention and compress.**
   → **A marker tensor is a fixed-shape DATA input, which is exactly what is trace-compatible**: one trace per total-B tier covers every partition of B, collapsing the tier count from combinatorial to linear.
8. Serial-head overhead: **0.2%–1.3% of full-round latency** (draft length 4→16, bs=128, averaged over context {512, 1024, 2048, 4096}), buying up to +30% accepted length.

### 4.7 deepseek-ai/DeepSpec — ✅ retrieved (HEAD `005e03b8`, 2026-07-09)

1. **The acceptance rule is standard rejection sampling, not greedy matching**: `accept_prob = clamp(p_target/p_draft, max=1)`, `rand < accept_prob`, `accept_prefix_mask = accept_mask.cumprod(dim=1)`; on rejection the bonus token is drawn from the normalized residual `max(p_target - p_draft, 0)`. Verification is a **linear chain**, not a tree.
   → **Cost: the target must return the full softmax for every verify position** (verify_len × vocab), not just an argmax. **Unless we run greedy (temperature=0), where acceptance degenerates to argmax equality and only K integers need to come back.** (See D-10.)
2. **Target-side rollback is a single scalar truncation**: `past_key_values_target.crop(start)`, where `start` is the committed token count. No per-entry selection, no gather.
   ⚠️ **But DeepSpec's target is a stock HF model on `DynamicCache` with no compressed cache at all. So it gives zero guidance on CSA/HCA rollback.** (The retrieval itself flagged this as an open question.) → **§5.4 is ours to design.**
3. **The drafter never keeps speculative KV**: `past_key_values_draft.crop(start)` immediately after the block forward. The drafter's cache holds exactly one entry per **committed** token.
4. **It is hidden-state injection, not KV injection** (correcting the brief's wording): the drafter receives target hidden states from several tapped layers, concatenates along the feature dim, passes them through a shared `fc(L·d → d)` + RMSNorm, and then **each draft layer applies its own k_proj/v_proj over that sequence to manufacture context K/V**. Our `dspark.py`'s `main_proj` / `main_norm` / `fuse_target_hiddens` matches this.
5. **The γ serial steps are only the Markov head.** The backbone produces all block_size hidden states in one non-causal forward; the serial loop runs only over vocab-sized logits, and the only state carried between steps is the previously sampled id (memoryless: `bias_k = W2(W1[x_{k-1}])`).
6. **Confidence truncation is off by default in the reference** (`eval.py` defaults to `--confidence-threshold 0.0`), and truncation only ever **shortens** the proposal. → **Disabling the threshold on TT and always verifying a fixed K matches the reference's default behaviour; it is not cutting a corner.**
7. **`assert_no_final_target_layer`: the target's last decoder layer is explicitly forbidden as a tap point** (HF's `output_hidden_states` stores the final normalized hidden in that slot). Released tap points leave headroom, e.g. Qwen3-4B (36 layers) = [1,9,17,25,33], Gemma4-12B (48 layers) = [5,17,29,41,46].
   ⚠️ **Our code uses `(40, 41, 42)` — the last three layers of a 43-layer model, including the last one.** → U8.

### 4.8 SwiftSpec arXiv:2506.11309 — ✅ retrieved

1. **There is exactly one graph-reuse trick, and it is purely host-side**: pad the tree mask to a fixed `(w, max_seqlen)`, copy the real mask into the **suffix**, and mark the prefix attend-all. **>1000 distinct mask shapes compress into <20 graphs.**
2. **No trick in the paper requires a kernel to read a control value from device memory at execution time.** They do not split the graph, do not use persistent kernels reading shape from a device buffer, and do not use a fixed grid with device-side early exit. → **Strongly positive for TT: variable accept length needs no device-side control flow; it needs (a) a max-shape mask buffer and (b) a small trace set.**
3. **They suffix-align rather than front-pack**, and it is forced by a KV layout invariant: *"the KV states of the verified tokens are stored continuously in the prefix … and the KV states of the tree are stored right after the prefix"*. → **Under a trace, suffix packing is cheaper than front-packing (no gather/scatter). This is a real design fork, not a wording difference.**
4. **Padding up to the hardware's minimum matmul tile is free; padding past it is expensive**: Llama3-70B on 8×H800, bs=4 → 10.32 ms, bs=8 → 10.39 ms (the tensor-core minimum M is 8, so bs<8 saves nothing); bs=16 → 13.25 ms (+29%) for only +24.5% compression ratio, i.e. a net loss. **They therefore pin target batch at 8 and do no bucketing on the target side at all.**
   → **The TT analogue is the 32-row tile (`_pack_tokens`). The entire γ=5 verify fits inside one tile-row, so padding to 32 should be nearly free — which directly supports "one trace on the target side, no tier ladder".** (Needs E1 to confirm.)
5. **Asynchronous drafting (the drafter does not wait for the verify result) costs only ~9% compression ratio and buys +37% end-to-end**; and they do **asymmetric device allocation**: on Qwen2-72B, draft on 2 GPUs / target on 6 GPUs in parallel = 275 tok/s, versus serial sharing of all 8 GPUs = 200 tok/s. → **Direct support for §5.6's "drafter on its own chips, replaying concurrently with the target".**
6. **Honesty guardrail**: the paper contains **no measurement** of pad-to-max versus bucketing (the word "bucket" appears zero times), and its evaluated contexts only reach 500/1000. **SwiftSpec cannot be cited as evidence that pad-to-max is cheap.**

### 4.9 tt-metal `tech_reports/LLMs/vLLM_integration.md` — ✅ read locally (68 lines)

See §3. Conclusion: **this design does not depend on it**.

---

### 4.10 What we can reuse directly from the SGLang implementation

SGLang is the only place where DSpark exists end-to-end in a graph-captured runtime, so it is our best source of *contracts* even where its *mechanisms* do not port. This table is the concrete reuse list.

**Caveat on the references**: the module paths and symbol names below come from this session's retrieval of PR #30261 at merge commit `6cc9352dfe6c5c013750e72b39c127870ef5b54f`. Line numbers drift — **locate by symbol name, not by line**, and re-check against the tree before relying on any of it.

| SGLang artifact | What it does there | Verdict for us |
|---|---|---|
| `RaggedVerifyMode = {STATIC, CAP_ACCEPT, COMPACT}` + `SGLANG_RAGGED_VERIFY_MODE` (`environ.py`, default `"static"`) | The three-stage progression | **Copy the staging verbatim** → our Phase A / B / C (§8). Their default being `static` is also the precedent for shipping static and calling it correct |
| `RaggedVerifyLayout` / `from_verify_lens_device()` (`speculative/dspark_components/dspark_planner.py`) | The host/device split: per-request `verify_lens` stay as device tensors; the *tier scalar* is computed host-side from (bs, budget) alone, with a D2H `.tolist()` only as a slow fallback | **Copy the split exactly.** This is the single most portable idea in the PR and the thing that makes DSpark graph/trace-compatible at all (§4.2.2, §5.2) |
| `_build_ragged_verify_token_buckets()` (`decode_cuda_graph_runner.py`) | `buckets = sorted({bs * captured_req_width for bs in capture_bs})` | **Read, do not copy.** Their ladder is derived from a GPU batch-size list shaped by wave quantization. Ours is `{16, 32}`, derived from the 32-row tile (§5.1) |
| `round_up_grid()` (`speculative/ragged_verify.py`) | Tier selection; raises above the top tier, expecting the caller to have already refused the batch | **Copy the guard, replace the miss path.** TT has no cheap eager fallback, so cap `max_running_requests × (γ+1) ≤ top tier` at server start (§4.2.1) |
| `build_ragged_target_verify_geometry()` | Produces `cache_seqlens_int32`, `cu_seqlens_q`, `cu_seqlens_k`, `max_seq_len_q` | **Read as the metadata *contract*; do not port the mechanism.** Our per-slot `cur_pos` / `update_idxs` scalars already carry the same information (§5.2) |
| `BuildQoIndptr` / `PaddedToBucket` / `dspark_attn_metadata` (Triton kernels) | Device-side metadata builders, so metadata construction is not on the host critical path | **The right precedent** for "metadata must be computed on device". We get it for free from the packet's `pos`, so we need no equivalent kernels (§5.2) |
| `pad_verify_lens_to_bucket()` | Tail padding to the tier; has an **unclamped** `padded[-1] += leftover` branch when there are no pad rows | **Read as a hazard list.** If we ever do compact, use the `cap=N` ghost-slot variant so every row is bounded by a compile-time constant (§4.4, §5.2, D-05) |
| `graph_tier_fill_budget()` + `--speculative-dspark-align-verify-tokens-to-graph-tier` | Turns tier padding into *real* draft tokens at the same step time; off by default | **Copy the mechanism and flip the default to ON** — our tiers are coarser so the payoff is larger (D-09) |
| `overlap_utils.py`: `CONFIDENCE_RELAY_RING_LAG = 2`, `CONFIDENCE_RELAY_RING_DEPTH`, `conf_ring` (pinned `(depth, req_pool_size, γ)` fp32), `copy_done[slot].query()` | The two-step-back confidence relay; returns `None` if the copy has not landed, and the planner falls back to full budget | **Copy the algorithm and the fail-safe semantics** for Phase B. Their pinned-ring + CUDA-event mechanism maps onto our D2H socket plus a reader thread (§5.3) |
| Per-request `generation` counter relayed alongside confidence, with `torch.where(fresh, k_survival, ones_like)` | Stops a preempted/rescheduled request from inheriting another request's confidence | **Copy verbatim.** Pure host-side, ~50 lines, and it is the part that is easy to forget (§5.3) |
| The greedy admission loop in `dspark_planner.py` (global descending sort over `a_{r,j}`, `B += 1` per admission) | The scheduler's argmax | **Copy for Phase C, minus the early-stopping break** — on our sparse tier set early-stopping is near-certain to get trapped (§5.7) |
| `verify_layout_graph_num_tokens_floor()` + the "uninitialized flat SPS table" warning text | Bucketing without a profiled cost table degenerates to verify-all | **Copy the warning as a hard precondition** on Phase C (§5.7) |
| `python/sglang/benchmark/dspark_sps_profiler.py` (defaults `DEFAULT_MAX_BATCH_SIZE = 256`; conversion `batch_tokens = num_running_reqs × verify_num_draft_tokens`, `steps_per_sec = 1/median(step_time)`) | The offline SPS profiler, run against a live server with `SGLANG_RAGGED_VERIFY_MODE=static` and `SGLANG_DSPARK_ENABLE_SPS_RECORD=1` | **Copy the shape of it; re-profile on Blackhole.** Our sample points must be exactly our captured tiers, since those are the only shapes that can run. Note their own caveat: profiling in static mode over-estimates steps/sec relative to compact |
| `dspark_force_budget_frac` (debug knob) | Forces ragged verify lengths, because natural traffic produces uniform ones | **Copy.** Without an equivalent, a ragged TT path would go untested (§4.4.4) |
| The construction-time `raise` listing compact's incompatibilities (two-batch-overlap / LoRA / disabled padding), and each backend's `NotImplementedError` | Refuses loudly instead of degrading silently | **Copy the posture.** Given TT has no IMA trap, refusing loudly matters more for us than for them |
| The standalone plan-builder harness from PR #32467 (1 GPU, no model weights, tiny inputs, invariant `max(ragged_id) < sum(extend_lens)`) | How the reporter localized the root cause | **Copy this methodology *before* writing any packing code.** On TT a corrupted index tensor produces no crash to localize, so this harness is not optional (§4.3, Phase C verification) |
| `full_cuda_graph_backend.py`, `trtllm_mha_backend.py::_write_ragged_verify_graph_metadata` | Where per-backend ragged support is gated and where the static metadata buffers are refreshed before replay | **Read for the boundary, do not port.** Their refresh happens outside the graph, which is the one thing we cannot afford (§4.2.3) |

**Also worth reading, in this order, before implementation** (closest structural analogues, not SGLang):
1. **upstream vLLM PR #46995** — `dspark/speculator.py` subclassing `DFlashSpeculator`, `dspark/utils.py`, the 5-line `scheduler.py` patch setting `num_lookahead_tokens = num_spec_tokens`, `mla/sparse_swa.py`'s `ori_sparse_indices` handling, and `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` (529 lines) as a correctness-test template. This is the smallest complete DSpark integration in existence (+1821/−94 across 24 files).
2. **vllm-ascend `dspark_proposer.py`** (196 lines subclassing `dflash_proposer.py`) — the non-NVIDIA graph-mode precedent, and the source of the measured numbers in §4.5.6.
3. **DeepSpec** `_forward_backbone`, `_confident_prefix_length`, `sample_residual`, `assert_no_final_target_layer`, and `config/dspark/*.py` — the correctness oracle for accept semantics and the tap-point constraint (§4.7).

---

## 5. Phase 2: design

### 5.0 Adjudicating the zero hypothesis, item by item

Zero hypothesis: *bucket the trace by total verify token count, front-pack variable-length requests into a compact buffer, and round only the total up to the nearest tier.*

| Component | Does it hold on TT? | Basis |
|---|---|---|
| Bucket on **total verify tokens** rather than batch size | **The premise does not exist in v1** | Under fixed γ, total = R×γ, and the only free variable is the request-row count (upstream vLLM code comment, §4.5.2). A variable total requires the confidence scheduler, which Ascend measured as a net loss (§4.5.1) |
| **Front-packing** variable-length requests | **Not doing it.** Per-slot positions instead (equivalent to the paper's marker tensor) | ① their front-packing is free only because flash_mla already consumes cu_seqlens, and TT has no such op (§4.1.2); ② compact is not a throughput win on GPU (§4.3.3); ③ our per-slot `cur_pos` / `update_idx` are already varlen semantics (§0, Finding 2) |
| **Round up to the nearest tier** | **Holds, but there is only one tier**: `_pack_tokens`'s tile-row is 32 tokens, and the whole γ=5 verify fits in one | `tt/attention.py:337-350`; SwiftSpec's isomorphic conclusion that padding to the hardware minimum M is free and past it is not (§4.8.4) |
| **One trace covering many interior partitions** | **Holds, and is mandatory** | The paper's marker tensor (§4.6.7): a fixed-shape DATA input lets one tier cover every partition of B |
| **Host writes metadata into a buffer ahead of time** | **Holds, and does not break trace semantics** | This code does it every step (the H2D packet socket, `tt/model.py:1937-1949`); and the better version is to send only `pos` and compute metadata inside the trace (`tt/model.py:1632/1649/1697/1707`) |
| **Per-request verify segments must be 4-aligned** | **Not required** — the hypothesis was explicitly tested and refuted by #33412's reporter | §4.3.2 |

**Net conclusion**: the zero hypothesis's **mechanism** (one fixed-shape trace plus variability expressed as data) holds and must be adopted; its **parameterization** (bucket by total tokens + front-pack) solves, in v1, a problem we do not have. And the genuinely hard part is one the zero hypothesis skips entirely: **rewriting compressed state under partial acceptance** (SGLang's own C1, still open).

---

### 5.1 (a) How to cut the trace buckets

**Conclusion: v1's verify path adds exactly one trace variant, and introduces no token-count bucket axis at all.**

#### Today
Five global variants = (2 SDPA causal modes) × (3 pooling phases, minus the unreachable), deduplicated per submesh by `_sm_pool_key` (`tt/model.py:745-752`). The variant key is **not a shape** but **which ops appear** — whether pooling runs, and whether SDPA is causal or masked.

#### What happens if we add verify and do nothing else
A verify step writes K tokens at positions q..q+K-1. The number of CSA windows closed in one step depends on `q mod 4` and is in {1,2}; HCA is in {0,1}. So the pooling phases go from 3 to **4 (q mod 4) × 2 (HCA) = 8**, times causal/masked = **16**, plus the existing 5 single-token variants = **21**. Every variant needs a compile run **on every submesh** before capture (`tt/model.py:2060-2100`). That capture time is unacceptable.

#### New design: demote the pooling phase from control flow to data (scratch-row redirection)

Three observations, all with code backing:

1. **A compressed entry is written to `combined` row `win_row`, and `win_row` is already a device-computed INT32 `[B]`** (`tt/model.py:1707-1738` `_device_compressor_indices`).
2. **The tail rows of the KV axis are permanently invalid.** The `a` table is filled with `max_seq` past `axis_rows`, with the comment: *"Filling A past the axis with a position no step can reach makes `A > pos` true there, so the tail is masked out however the compressor compare falls"* (`tt/model.py:1559-1563`). In causal mode, `cur_pos = 128 + (pos+1)//cr - 1 ≤ axis_rows - 1 < kv_len`, so the tail is excluded there too.
3. Therefore: **take one row from the KV axis tail as a SCRATCH row and map it to a real physical block (not `ZERO_BLOCK`, which is globally shared), and you have a write-only-never-read wastebasket.**

The verify trace then **unconditionally** performs 2 CSA poolings and 1 HCA pooling, with the i-th destination row computed on device:

```
row_i = (i < n_close) ? (sliding_window + w_i) : SCRATCH_ROW
```

`n_close` and `w_i` are pure arithmetic on `q` and `q_prev` (floor / multiply / subtract), the same class as the existing `_device_compressor_indices`. **No branch, therefore no new variant.**

Add "do not speculate below `sliding_window`" (D-01) and the verify trace is causal-only → **the verify path costs +1 trace per submesh.**

#### Quantitative: tier count × per-tier DRAM versus hit rate

- **DRAM is not the constraint.** Every profile uses `trace_region_size: 0` (`tt/system_config.py:71`, `configs/system_configs.yaml:47`), i.e. ttnn allocates dynamically from the top of DRAM with **no reserved region and no bound**. Mainline budgets 70–100 MB for comparable LLMs on Blackhole (`models/model_trace_region_sizes.yaml:19-20`, `128-129`), with one 250 MB outlier. A Blackhole chip has **32 GB**, and weights occupy only ~5.0 GB/chip on galaxy32 (§2.7, derived). **+1 variant ≈ +15–20 MB/chip, i.e. 0.06% of DRAM.**
- **The real cost is capture time**, and this is a substantive TT-versus-GPU difference: the README says "minutes" (`README.md:91`), and SGLang's 35+35 shapes on 8 GPUs take ~5 minutes of pure capture (§4.3). Every variant of ours requires **a compile run on every submesh first** (to JIT), then capture (`tt/model.py:2060-2100`'s two-pass structure — and the passes cannot interleave, because once a device holds a trace, allocating buffers on it is unsafe).
- **Hit rate**: v1's hit rate is **100%**, because there is only one verify shape. That is the direct dividend of fixed γ.

> **Decision: no token-count bucketing in v1.** If we ever do (v3 compact), a two-point tier set `{16, 32}` suffices — the tile-row is 32 rows, and `B×K ≤ 32` compresses the whole feasible space into six points `{5,10,15,20,25,30}`. **Do not copy SGLang's `{bs × width}` ladder**; it is shaped for GPU wave quantization.

---

### 5.2 (b) How front-packing metadata would be rebuilt at replay

**Conclusion: no front-packing. The per-slot position scalars *are* our cu_seqlens, and the metadata is computed inside the trace on device from the packet's `pos`.**

#### Three possible approaches, and why the third

| Approach | On TT | Verdict |
|---|---|---|
| Host computes page table / qo_indptr / cu_seqlens each step and writes them into device buffers | Goes through `copy_host_to_device_tensor` = **command-queue work**, outside the trace, one full dispatch. This is exactly what SGLang does (*"runs out-of-graph on every capture/replay-prep"*, §4.2.3) — **they can afford it, we cannot** | ❌ not for per-layer, per-step metadata |
| Device computes it from a packed index array | Requires a varlen attention op that consumes a device-resident indptr. TT-NN has none (§4.1.2). And this is exactly the class of code that failed in #33412 | ❌ a separate kernel project |
| **Device computes it from per-slot scalars in the packet** | **This is what the existing code does**: the packet carries `[tokens(B) \| pos_sliding(B) \| pos_compress(B)]` (`tt/model.py:1471-1478`), and RoPE / mask / win_slot / win_row / cur_pos are all derived from `pos` inside the trace | ✅ |

**"One position per slot" *is* cu_seqlens semantically**: `paged_scaled_dot_product_attention_decode`'s `cur_pos_tensor [B]` makes slot u read only its own KV prefix (`tt/attention.py:1147-1151`), and `paged_update_cache`'s `update_idxs_tensor [B]` makes slot u write to its own position (`tt/attention.py:441-448`). **On TT, variable length is B scalars, not an indptr array.** This is also the simplest form of the paper's marker tensor (§4.6.7).

#### The one thing the host must still write: the page table

`ensure_session_capacity` → `_write_page_tables` is command-queue work (`tt/model.py:939-952`, `985-990`), triggered today once every `cr × block_size` tokens. Speculation makes positions jump by 0..K per step, which raises the trigger frequency without changing its order of magnitude.

> **D-04: pre-grow.** Grow by one extra block on the worst case of "K+1 more tokens", which pushes the trigger frequency back to today's and lets the host's knowledge of position lag without affecting correctness (a prerequisite for Plan X in §5.3). Cost: one extra block per session per group.

#### Three hard constraints from #33412

1. **TT has no MMU and no IMA trap.** An out-of-bounds scatter that crashes loudly on GPU is, for us, a **silent overwrite of another tensor**, inside a trace. → **D-05: every write index derived from the verify layout must be clamped on device.** SCRATCH_ROW is a natural clamp target: `row = min(row, SCRATCH_ROW)`.
2. **Never re-derive raggedness on device**, and never leave a fast-path index of the form `B * s_max + j`. Our scheme has no raggedness derivation (K is fixed), but this belongs in the review checklist.
3. **Never let one core initialize another core's scratch slot** (the reviewer-variant lesson, §4.3.5). The same holds if we ever do a cross-core reduction on TT.

---

### 5.3 (c) The host sitting on the draft→confidence→schedule→verify critical path

#### First, quantify "critical path"

A traced step's host work today: one packet PCIe write (not command-queue work), one blocking D2H read, and a host argmax. The three-stage pipeline already overlaps it (`tt/model.py:2172-2175`; the server's round structure at `demo/server.py:815-820`).

With DSpark, one round becomes:
```
[drafter trace replay] → D2H(draft ids + confidence) → host decision → H2D(packet)
                       → [target verify trace replay] → D2H(K logits) → host accept → next round
```
**One extra full D2H→host→H2D round trip per round.**

#### Can the two-step-back confidence relay be done on TT? Yes. Preconditions, one by one:

| Precondition | Do we have it? | Basis |
|---|---|---|
| On-device page table | **Not needed** | SGLang needs it because their page table changes every step; ours changes only on growth, and pre-grow restores today's frequency (D-04) |
| Overlap scheduler | **Already have it** | `write_step_packet` / `replay_traced` / `read_decoded_output` are separable across threads (`tt/model.py:2172-2175`); the server's round structure already overlaps |
| Non-blocking D2H + event query | **Equivalent exists**: the D2H socket read on its own thread does not block dispatch (`tt/model.py:2145-2149`) | as left |
| Per-request generation counter | **To be written**, pure host-side, ~50 lines | SGLang's approach (§4.1.5, §4.10) |

**Effort: if v1 uses fixed γ, the relay needs zero lines.** It is only required at v3 (dynamic γ), where it is ~200 host-side lines (ring + generation counter + SPS table lookup).

**But the paper gives a stronger reason that must be in the design**: the two-step lag is **not a performance optimization, it is a necessary condition for losslessness** (§4.6.4, the Appendix A counterexample). And TT's trace tiers are inherently a jagged cliff, which the paper says early-stopping cannot cross. → **On TT, "do dynamic γ" and "do asynchronous lag" are one decision, not two.**

#### Plan X (recommended as the v3 target shape, not v1): take the host out of the speculation loop entirely

The zero hypothesis and every external implementation assume the host is in the loop. **TT has an option a GPU does not**: fuse draft + verify + accept into one trace, and leave the host only to read outputs asynchronously.

The chain of evidence, each link sourced:
- **The author has already written down both the obstacle and the fix**: `decode_sampled_burst` raises `NotImplementedError` because the in-trace `recv_async_h2d` overwrites the device-sampled token slot; the fix is *"splitting the packet: keep the host-fed positions on the socket and read the token from a separate device-written buffer"* (`tt/model.py:2213-2214`, `2216-2219`).
- **Device-side sampling exists**: `ttnn.argmax`; and the drafter already does an embedding lookup on device (`tt/dspark.py:436`).
- **Under greedy, the accept decision is pure element-wise ops**: `eq(target_argmax[j], draft[j+1])` then `cumprod` → k.
- **Position advance can be a device-side add**: `pos += k`, and RoPE / mask / win_slot / win_row / cur_pos are already derived from `pos` (`tt/model.py:1632/1649/1697/1707`).
- **The page table's lag is covered by pre-grow** (D-04).

**Payoff**: one draft+verify round has exactly one host interaction (an asynchronous output read), and the host is not in the speculation loop at all — we would not need a two-step relay because we would not need the host. This is structurally impossible for a CUDA graph, and it is precisely why SGLang had to invent ZOS + the relay.

**Risk (why this is not v1)**: it depends on three unverified things — `ttnn.argmax`'s behaviour inside a trace, a device-resident position counter coexisting with the packet, and greedy-only sampling. → experiment **E7**.

---

### 5.4 (d) CSA's 4-token compression boundary versus variable-length accept — rollback semantics

This is the hardest section, and it is also SGLang's own **C1 (#32183, still open)**. DeepSpec's reference implementation offers **no guidance at all** here (its target is a stock HF model on `DynamicCache` with no compressed cache, §4.7.2).

#### First, separate "committed" from "correct"

A verify step's query positions are `q .. q+K-1` (K = γ = 5 draft tokens). Let `a ∈ [0, K]` drafts be accepted:

- Positions `q .. q+a-1`: the tokens are committed **and their KV/projections in the buffer are correct**.
- Position `q+a`: the token is committed (either the target's corrected token, or the bonus token when all were accepted), **but that buffer slot holds the rejected draft's projection, or was never written — it is wrong**.
- Positions `q+a+1 .. q+K-1`: rejected; the buffer holds garbage.

So the next verify starts at `q' = q + a` (**not q+a+1**). **Output advances by a+1 tokens per step, while the "correct KV prefix" advances by a.**

> **D-02 (the off-by-one convention)**: query length = γ = 5, and a step emits at most γ+1 = 6 tokens. This agrees with upstream vLLM's `num_lookahead_tokens = num_spec_tokens` (not +1) (§4.5.5). SGLang's "verify window = γ+1 = 6" is a different accounting. **This is a classic off-by-one bug source and must be pinned by E5 before implementation.**

#### Rollback theorems (three, each with a proof)

**Theorem 1 (the sliding ring rolls back automatically, at zero cost).**
Rejected tokens occupy `combined`'s ring slots `(q+j) mod 128` for j ∈ [a, K). The next verify writes positions `q' .. q'+K-1 = q+a .. q+a+K-1`, whose slot set contains all of `(q+a) .. (q+K-1)`. And `decode_static` **writes before it reads within a layer** (write at `tt/attention.py:1413`, read at `tt/attention.py:1440`). Hence those slots are overwritten with correct values before any query reads them. ∎
⚠️ **This requires K to be fixed.** Under cap-accept / compact, K can shrink and coverage may be incomplete — **a hard reason to choose static**, not merely "easy first".

**Theorem 2 (the CSA window slots roll back automatically, at zero cost).** As above, with window slots `(q+j) mod 4`; K = 5 > 4, so the next step fills all four slots. ∎

**Theorem 3 (compressed entries and `prev` genuinely break).**
If a window closes inside the rejected range (there exists j ≥ a with `(q+j+1) mod 4 == 0`), then: (i) an entry derived from uncommitted tokens is written into `combined[128+w]`; and (ii) `_retire_window(prev, win)` **destructively** copies win into prev via `ttnn.fill_cache` (`tt/attention.py:482-497`), and the old prev is gone forever. (i) can be repaired by recomputing and overwriting; **(ii) is irreversible.** ∎

#### Design: deferred pooling + a merged ring buffer (eliminating `prev` and `_retire_window`)

**Step 1: defer pooling.**
A verify step **writes only, never pools**. Pooling moves to **the start of the next verify step**, and covers only windows fully closed inside the "correct KV prefix":

> A verify step beginning at position `q'` first pools every window w satisfying `q ≤ (w+1)·cr ≤ q'`, and only then writes this step's K tokens.

Because `q' - q = a ∈ [0, K]` with cr=4 and K=5, **the number of CSA windows to pool in one step is in {0,1,2}; HCA (cr=128 > K) is in {0,1}**. §5.1's SCRATCH_ROW pins this to "always 2 CSA + 1 HCA", and the control flow disappears.

The ordering also works out: this step pools w (and possibly w+1) first, then writes tokens belonging to w+2; and w+2 shares w's residue class, so writing it cannot damage w+1's pooling inputs.

**Step 2: merge `prev` and `win` into one `[B, 1, 4·cr, 2·Dh]` ring buffer, and delete `_retire_window` entirely.**

Window w's tokens are written to row `((pos // cr) mod 4)·cr + (pos mod cr)`, which is exactly **`pos mod 16`** at cr=4 — pure arithmetic, the same class as the existing `_device_compressor_indices` (`tt/model.py:1707-1738`).

**Why 4·cr and not 2·cr (an error I made and corrected while deriving this; recorded so it is not repeated)**: deferred pooling means that at the pooling point the buffer already holds **all K tokens of the previous step**, including those belonging to later windows. Quantitatively: in a step beginning at `q'`, the written prefix reaches `q_prev + K - 1 = q_prev + 4`, while the oldest window w to be pooled needs window `w-1` intact, whose oldest position is `≥ q_prev - 7`. **The live span is 12 positions.** 2·cr = 8 rows is **not enough** (window w+1 would overwrite window w−1's rows); 3·cr = 12 rows sits **exactly on the boundary**; **4·cr = 16 rows** leaves 4 positions of headroom and keeps the modulus a power of two.
**Memory goes down, not up**: `TILE_LAYOUT` pads the second-to-last dim to 32 rows, so today's two `[B,1,4,F]` buffers occupy 32 rows each = 64 rows, whereas a single `[B,1,16,F]` occupies 32.

Pooling entry w needs "window w−1's Ca half plus window w's Cb half".
The key: `_pool_window` ultimately is just `sum(kv · softmax(gate, dim=2), dim=2)` (`tt/attention.py:471-479`) — **a sum over the row axis, so row order is irrelevant**; and non-participating rows only need their gate driven to `-inf` for a softmax weight of 0 (**the existing code already uses this trick**: window 0's missing Ca half is cancelled by `prev_gate` still holding `_MASK_NEG`). So no concat and no parity-based reordering is needed — only two `[1,1,4cr,1]` selection masks computed on device from `pos`:

```
w        = pos // cr                                   # device arithmetic
cur[r]   = (r // cr == w      mod 4)                   # the cr rows of the current window
prv[r]   = (r // cr == (w-1)  mod 4)                   # the cr rows of the previous window
kv       = cur · Cb_all + prv · Ca_all                 # Ca_all/Cb_all = the buffer's two feature halves
gate     = cur · Cbg_all + prv · Cag_all + tile(position_bias, 4)
gate     = where(cur + prv, gate, _MASK_NEG)           # zero the other 2·cr rows
entry    = sum(kv · softmax(gate, dim=2), dim=2)
```

- `position_bias` is `[1,1,cr,2Dh]` today (`tt/attention.py:688-693`); tile it 4× along dim 2 into `[1,1,4cr,2Dh]` and the broadcast semantics are unchanged (each token still picks up the row for its offset within its own window).
- **The `prev_kv` / `prev_gate` buffers are deleted, and `_retire_window` (`tt/attention.py:482-497`) is deleted** — and that is exactly the irreversible destructive operation in Theorem 3.
- **Rollback becomes "do nothing"**: the 5−a rejected tokens only dirty a contiguous run of rows at `pos mod 16`, and the next step writes 5 consecutive positions from the same start, overwriting them (Theorem 2 holds mod 16 as well: a run of ≤5 consecutive positions is injective mod 16, and the rejected run is a prefix of the next step's write run); and pooling only ever reads committed content (deferred pooling).
- **Side benefit: the single-token path gets faster too** — it drops the whole-buffer `fill_cache` that 20 of the 43 layers pay once every 4 steps.

**Step 3: HCA.** cr=128 > K, so at most one window closes per step, and HCA has no `prev` (`tt/attention.py:499-508`). But it **needs the same ring headroom**: under deferred pooling, the first tokens of window w+1 arrive before window w is pooled. Grow HCA's window buffer from `[B,1,128,Dh]` to `[B,1,256,Dh]` and index it `pos mod 256` (2× memory ≈ B×5.9 MB across 23 HCA layers, negligible at B=1). No Ca/Cb selection masks are needed — only deferred pooling plus SCRATCH_ROW.

#### Answering "what state is the partially-filled compression block in when 3 of 5 are accepted?"

Take `q ≡ 0 (mod 16)` (so the row numbers can be written out), K=5, a=3.
- Writes: positions q..q+4's projections land in merged-buffer rows `{0,1,2,3, 4}` (the first four are window w=q/4's slots 0..3; the fifth is window w+1's slot 0).
- **Pooling: none happens on this step** (deferred). So although window w "looks full", **no compressed entry is produced** and `combined[128+w]` is untouched.
- The host obtains a=3 → `q' = q+3`.
- The next step (q'=q+3) begins by pooling windows satisfying `q ≤ (w'+1)·4 ≤ q+3` — **there are none** (`(w+1)·4 = q+4 > q+3`). So **both** CSA pooling slots redirect to SCRATCH_ROW, writing garbage into a row nothing can read.
- It then writes positions q+3..q+7, covering rows `{3, 4, 5, 6, 7}` — **row 3 was the rejected q+3, and it is now overwritten with the correct value** (Theorem 2).
- The step after that (say a'=4, q''=q+7) pools `q+3 ≤ (w'+1)·4 ≤ q+7` → w'=w (`(w+1)·4=q+4` ✓) and w'=w+1 (`(w+2)·4=q+8 > q+7` ✗) → **exactly one**, with the second slot going to SCRATCH. Window w's four rows are by now entirely correct projections of committed tokens ✓.

**The state of the partially-filled compression block: it does not exist.** A compression block is only materialized once all cr of its tokens are inside the "correct KV prefix"; before that, the corresponding `combined` row has never been written, and the mask / `cur_pos` already mark entries with `w ≥ (pos+1)//cr` invalid (`tt/attention.py:245-250`, `tt/model.py:1655-1662`) — **the semantics line up for free**.

#### Relationship to #33412

Their IMA's root cause was a plan-builder warp race that made a ragged batch look uniform, taking an out-of-bounds `B·s_max+j` fast path (§4.3.1) — **not** the 4-token boundary. Our scheme has no raggedness derivation (K is fixed) and every index is computed directly from the packet's per-slot `pos`. But **their lesson applies in a different way**: TT has no IMA trap, so an out-of-bounds write is silent. → **D-05's clamp is mandatory, not defensive programming.**

---

### 5.5 (e) Parallelism: do we actually need a different split from the GPU?

**Answer: TP/EP does not need to change for DSpark. DSpark needs exactly one change to the split — give the drafter its own chip and replay it concurrently with the target.**

#### Why TP/EP should not be coupled to DSpark

1. The `galaxy32` profile says it itself (`configs/system_configs.yaml:210-224`): under pipeline-parallel-only, 32 chips **do not improve tok/s/user**, because per-token latency is the sum of all 43 layers' compute plus ~31 socket hops (versus ~5 layers/chip and 7 hops at 8 chips); 32 chips buy capacity only. The intended shape (8 pipeline stages × 4-chip submeshes with 256 experts sharded 64-per-chip) is **written in the comment and not implemented**.
2. That is work **independent of DSpark**. Coupling them lets either one's failure kill the other.
3. **And DSpark's payoff is actually larger under pipeline-parallel-only** — which is this section's positive conclusion.

#### Why DSpark's benefit structure is better on Galaxy's PP architecture

The step-time envelope is `T(M) = f + c·M` (§5.7). `f` contains **the pure latency of ~31 socket hops**, entirely independent of M. Speculation is the only lever that amortizes `f` across several tokens: one step emits `a+1` tokens, for an amortization ratio of `(a+1)·T(1) / T(K)`.

**More chips → a larger hop share of `f` → a larger relative payoff from speculation.** In other words: **DSpark and TP/EP are two orthogonal routes to the same problem (`f` is too large on Galaxy)**, and DSpark is far cheaper (it changes no attention/MoE parallel split).

The counter-evidence belongs here too: vllm-ascend measured DSpark's advantage collapsing with concurrency (versus no-spec: BS1 +124.8% → BS16 +50.8%, §4.5.6), while `galaxy32_server` is positioned for 32 concurrent users. → **DSpark's target regime on Galaxy is low-concurrency, low-latency, not high throughput.** The baseline must measure the two regimes separately.

#### The one thing that must change: lock batch=1

`activate_sessions` requires co-seated sessions to sit at the **same position** (`tt/model.py:917-926`, `raise ValueError(f"resident sessions must resume at one position, got {at}")`), because one trace bakes in the pooling schedule for the whole batch. Under speculation each user advances by 1..γ+1 tokens per step, so **positions diverge immediately**.

- The `batch=1` server interleaving mode (`galaxy32_server`: `batch: 1`, `num_users: 32`): **each session is its own group, so it is unaffected** ✓
- The `*_throughput` profiles (`batch: 8`): **broken outright** ❌

> **D-07: lock `batch=1` in v1.** batch>1 with DSpark first requires making the pooling phase per-slot data — and §5.4's SCRATCH_ROW plus merged ring buffer **already solves exactly that** (each slot gets its own `w` and its own masks), so this is a natural v4 extension rather than a dead end.

#### One thing we do not need to do

SGLang needs a CPU all_gather every step to agree on the tier under DP (§4.1.4), and C0 records a real bug where TP ranks selected different graph tiers and produced an asynchronous IMA (§4.4's C0). **We are structurally immune**: the whole mesh is driven by a single host scheduler thread (`demo/server.py:805-812`), so trace selection is one decision. → **C0 is work we can skip**, but "the tier decision must be fully resolved on host before replay" should be written as an assertion.

---

### 5.6 (f) Which chips the drafter lives on, and how it shares embedding / lm_head with the target

#### What the drafter needs

| Component | Size (derived) | Source |
|---|---|---|
| The 3-stage backbone | see U7 below | The paper: 3 **MoE** layers + mHC + a 128 sliding window (§4.6.5). ⚠️ Our `dspark.py` implements a **dense MLP with a standard residual**, which the author labels "so it can be unit-tested alone (no 256-expert MoE...)" (`dspark.py:17-18`) — **that is not the production drafter** |
| `main_proj` (3·4096 → 4096) | 50.3M params | `dspark.py:400-415` |
| `markov_w1` `[129280, 256]` | 33.1M params, bf16 ROW_MAJOR = **66.2 MB** | `tt/dspark.py:390-397` |
| `markov_w2` `[256, 129280]` | 33.1M params, bf4 = **18.6 MB** | `tt/dspark.py:398-404` |
| Confidence head | **Really only needs `[1, 4096+256]`**; currently padded to `[129280, 4352]` = 562.6M params | `tt/dspark.py:405-413` ⚠️ |
| Embedding (shared) | 129280×4096 bf16 ROW_MAJOR = **1.059 GB** | `tt/embedding.py:22-29` |
| LM head (shared) | 529.5M params, bf4 = **297.9 MB** | `tt/generator.py:180-184` |

#### Placement decision

> **D-06: give the drafter its own chip past the end of the pipeline (on galaxy32, set `pipeline.max_devices` to 31 and give chip 31 to the drafter); put a physical copy of the embedding and of the LM head on that chip.**

Reasoning:
- **The LM head is on `model.last_device`** (`tt/generator.py:182`) and the embedding is on `first_device` (`tt/model.py:365`) — **they are not on the same chip**, so wherever the drafter goes, at least one copy is unavoidable.
- Reusing across chips means going through a socket, i.e. one more pipeline depth, which drags the drafter back onto the critical path.
- Copy cost: 1.059 GB + 0.298 GB = **1.36 GB**, entirely acceptable against 32 GB/chip (galaxy32 weights occupy only ~5.0 GB/chip).
- **For contrast**: upstream vLLM shares the target's modules by pointer aliasing (§4.5.5); the DeepSpec reference **copies and freezes by default** (§4.7.7). Both have precedent; the physical distribution forces us to copy.

**How target hidden states get there**: add a `send_direct_async` to the last submesh's trace, concatenating the three layers' hidden states along the feature dim and shipping them to the drafter chip (an existing mechanism, `tt/model.py:1157-1169`).
- Volume: 3 × 4096 × 2 B = **24 KB/token**, so 120 KB at K=5.
- ⚠️ The socket's L1 buffer defaults to **16384 B** (`configs/system_configs.yaml:72`), so either raise it or page the transfer. → an implementation detail, not a design risk.

**⚠️ The tap-point problem**: DeepSpec carries a hard assertion `assert_no_final_target_layer` — the target's **last** decoder layer is explicitly forbidden as a tap point (§4.7.7), and released configs all leave headroom (Qwen3-4B's 36 layers use [1,9,17,25,33]). Our code uses `(40, 41, 42)` (`dspark.py:142`), the last three layers of a 43-layer model, **including the last**. → **U8; check the checkpoint's `config.json` before implementing.**

#### Serial Markov head: inside or outside the trace?

**Inside.** The two backends disagree here: upstream CUDA puts it in the graph, the Ascend prototype leaves it out (§4.5.7), but Ascend's stated reason is "avoid nested torch.compile" — **a reason that does not exist on TT**, where per-op host dispatch costs more.
The current implementation makes **4 host round trips per Markov step** (`tt/dspark.py:508-535`), so ~20 per round at γ=5; that must go. Moving it inside requires device-side argmax plus device-side `ttnn.embedding` (the drafter already uses the latter).
Budget reference: the paper puts the serial head at **0.2%–1.3% of full-round latency** (§4.6.8); upstream on 8×B300 reports 1.2 ms of draft against 11–13 ms of verify (§4.5.6). → **the drafter's total budget should stay within 5–10% of the verify step.**

#### Concurrency

SwiftSpec measured asymmetric device allocation plus asynchronous drafting: draft on 2 GPUs / target on 6 in parallel = 275 tok/s versus serial sharing of 8 GPUs = 200 tok/s (**+37% e2e for −9% compression ratio**, §4.8.5). Our PP architecture supports this natively: the drafter trace and the target trace live on **disjoint chip sets** and can replay concurrently.
→ **v1 is serial (drafter → verify); asynchronous drafting arrives at v3.**

---

### 5.7 (g) SPS cost table: what shape θ takes on TT, and what that does to the scheduler's argmax

#### First, a correction of provenance

The brief's `T(bs,K) = bias + alpha(bs) + theta(M), M = bs+K` **has no source in anything retrieved this session**. The paper uses a **single-argument** `SPS(B)` with `B = Σ_r (1 + ℓ_r)` the batch-wide total verify token count, and objective `Θ = τ · SPS(B)` (arXiv:2607.05147 §3.2.2, §4.6.1). SGLang's implementation likewise indexes a one-dimensional table by total tokens (`batch_tokens = num_running_reqs × verify_num_draft_tokens`). This document follows the paper's form.

#### The TT cost model: an affine envelope with a staircase we impose ourselves

```
T(M) = f + c·M          (envelope)
T_realized = f + c·M_trace   ,  M_trace = the token count baked into that trace
```

- **`c` (the per-token marginal cost) comes from MoE**: `fused_experts` is decode-only (`T == 1`), and M tokens become **M single-token ops** — *"they route to different experts, so there is no weight to share between them"* (`tt/moe.py:578-586`). Each op fetches six experts' weights from DRAM.
- **`f` (the fixed cost) comes from the dense path plus the pipeline**: every projection / norm / mHC packs ≤32 tokens into **one tile-row**, which *"makes a B-user step cost the same projections / norms / RoPE as a one-user step"* (`tt/attention.py:337-345`); plus the latency of ~31 socket hops.
- **The staircase has a different origin than on GPU**: the paper describes GPU `SPS(B)` as "inherently discrete, jagged, step-wise" (citing Yan et al. 2020, from kernel tiling / wave quantization, §4.6.3); **TT's staircase comes entirely from rounding up to a trace tier**, with a smooth affine curve underneath. SwiftSpec's isomorphic finding supports this: padding to the hardware minimum M is free, past it is not (§4.8.4).

#### What this does to the scheduler's argmax

1. **There are very few feasible points.** `B×K ≤ 32` (the tile-row) with γ=5 gives feasible M ∈ {5,10,15,20,25,30} — **six points**. The paper's greedy admits one token at a time and stops when Θ drops; on a six-point discrete set, **early-stopping is near-certain to stop before the first cliff**.
2. **So if we do dynamic γ, we must copy §5.2 of the paper**: remove the early-stopping break and use an asynchronous **global top-k** (descending over all `(request, position)` values of `a_{r,j}`, taking the top B). And **the asynchronous two-step lag is a necessary condition for losslessness** (the Appendix A counterexample, §4.6.4). → **On TT, "dynamic γ" entails "two-step lag"; they are inseparable.**
3. **The tier round-up waste should become real verification.** SGLang has an off-by-default flag `--speculative-dspark-align-verify-tokens-to-graph-tier` that, after rounding the total up to the tier, lets the confidence-ordered allocator spend the extra slots on real draft tokens "at the same step time" (§4.2.5). **TT's tiers are coarser (the tile is 32 rows), so the payoff is larger.**
   > **D-09: if we reach v3, tier-fill should default to ON, unlike SGLang's default-off.**
4. **Without a profiled SPS table, bucketing gains nothing** (SGLang's own warning text, §4.2.4). → The SPS table is a **prerequisite deliverable** for v3.

#### The good news: the SPS table is measurable today, with no DSpark code

**An M-token verify step's op sequence ≈ today's batch=M single-token decode step**: the same M rows packed into one tile-row, the same M `fused_experts` calls, the same M per-slot cache writes and per-slot `cur_pos`. The only differences are the number of compressor poolings (fixed at 2+1) and the KV read length.

> **Experiment E1: run `tests/test_multi_user_paged_decode_demo.py` with batch ∈ {1,2,4,6,8,12,16,24,32} and record the step time per point.** This directly yields `f`, `c` and the staircase positions. It is the first thing to do in the whole plan, and the only hard gate before D-01.

**The only anchor available today (derived, a two-point extrapolation — treat with care)**:
`PERFORMANCE_LOG.md` gives batch=1 on 8× BH P150 → **16.2 tok/s/u = 61.7 ms/step**, and commit `ceebba9f3ed`'s "Batch=64, 4.3 tok/s/u" → **232.6 ms/step**. Solving `f + 1·c = 61.7` and `f + 64·c = 232.6` gives **c ≈ 2.71 ms/token, f ≈ 59.0 ms**.
⚠️ Three caveats that must travel with it: (1) this is **8 chips**, not Galaxy; (2) `_pack_tokens` asserts `B·S ≤ 32`, so **the batch semantics of that B=64 point are questionable** (it may predate the assert, or be two steps); (3) a two-point fit cannot show the staircase. **Until E1 has run, any speedup derived from this is an order-of-magnitude estimate only.**

Taking it provisionally: `T(5) = 59.0 + 5×2.71 = 72.5 ms`, against `5 × 61.7 = 308.5 ms` for five independent single-token steps.
**Speedup formula (derived)**: `speedup = (a+1) · T(1) / (T(K) + d)`, with `a` the number of accepted drafts and `d` the drafter cost.

| a (accepted drafts) | tokens per step | speedup (d=0) | speedup (d = 10% of T(K)) |
|---|---|---|---|
| 1 | 2 | 1.70× | 1.55× |
| 2 | 3 | 2.55× | 2.32× |
| 2.36 (Ascend AL 3.356) | 3.36 | **2.86×** | **2.60×** |
| 2.94 (Ascend coding80 AL 3.935) | 3.94 | 3.35× | 3.05× |
| 5 (all accepted) | 6 | 5.11× | 4.64× |

For comparison: vllm-ascend measured DSpark versus no-spec at **+124.8% (2.25×)** at BS1 and **+50.8% (1.51×)** at BS16 on 16 NPUs (§4.5.6). **Our estimate is the same order and slightly optimistic** — the gap most likely comes from the drafter cost and from parts of `f` we are under-counting.

---

## 6. Decision log

Format: **[alternatives / basis / cost / reversibility]**. Reversibility has three levels: **easy** (a flag or a constant), **medium** (one module), **hard** (a cross-module data layout or the trace structure).

| # | Decision | Alternatives | Basis | Cost | Reversibility |
|---|---|---|---|---|---|
| **D-01** | **Do not speculate below `sliding_window` (128)**; use the existing single-token path there | Build a verify trace for masked mode too | Masked mode's mask is `[1,1,1,W]` broadcast over batch (`tt/attention.py:1155-1157`), while K distinct positions need `[1,1,K,W]`, and `_sdpa_decode`'s `ttnn.repeat(mask,[1,1,H,1])` assumes leading dims of 1. Causal mode (pos+1≥128) uses no mask at all | The first 128 tokens of each session are not accelerated: 0.012% at 1M context | **easy** |
| **D-02** | **Query length = γ = 5; a step emits at most γ+1 = 6 tokens** | γ+1 query tokens | Upstream vLLM: `num_lookahead_tokens = num_spec_tokens` (not +1), because the anchor is itself the first prediction position (§4.5.5); our block at `dspark.py:416-426` is `[anchor] + noise×(γ-1)` = 5 positions | Getting it off by one is a silent wrong answer | **easy** (but must be pinned by E5 first) |
| **D-03** | **Deferred pooling + a merged ring buffer**: the verify step writes but does not pool; pooling moves to the start of the next step and covers only the "correct KV prefix"; `prev_kv/prev_gate` and `_retire_window` are deleted in favour of a `[B,1,4·cr,2·Dh]` ring (index = `pos mod 16`) with device-computed `cur`/`prv` row masks selecting Ca/Cb; HCA's window buffer likewise grows to `pos mod 256` | ① snapshot all compressor window state and restore it on rollback; ② replay accepted tokens through single-token steps after each verify | ①: snapshotting/restoring 43 layers × 4 buffers each sits on the critical path (`tt/model.py:912-917` measures a comparable operation at ~100 ms); ②: k extra single-token steps × 61.7 ms cancels the entire gain; this design exploits `_softmax_weighted_sum` summing over the row axis, hence **row-order independence** (`tt/attention.py:471-479`) | Every step unconditionally pays 2 CSA + 1 HCA poolings (today CSA is 1-in-4 steps, HCA 1-in-128). CSA buffer memory **goes down** (64 rows → 32 after tile padding); HCA doubles (≈ B×5.9 MB); and **one destructive op disappears**, so the single-token path may get faster | **hard** (changes `_StaticLayerCache`'s layout) |
| **D-04** | **Pre-grow the page table** on the worst case of "K+1 more tokens" | Grow exactly, every step | `_write_page_tables` is command-queue work outside the trace (`tt/model.py:985-990`); pre-grow restores today's trigger frequency and lets the host's position knowledge lag (a prerequisite for Plan X, §5.3) | One extra block per session per group | **easy** |
| **D-05** | **Clamp every write index derived from the verify layout, on device**, to SCRATCH_ROW | Rely on hardware bounds checking | **TT has no MMU and no IMA trap**: the out-of-bounds scatter that crashes loudly on GPU silently overwrites for us, unobservable inside a trace (§4.3.4) | One extra `ttnn.minimum` per index | **easy** |
| **D-06** | **Give the drafter its own chip past the end of the pipeline** (galaxy32: `pipeline.max_devices=31`, chip 31 for the drafter); **a physical copy of embedding + lm_head on that chip** | ① pointer aliasing (upstream vLLM's approach); ② squeeze the drafter into the target's last submesh | The embedding is on `first_device` and the LM head on `last_device` (`tt/model.py:365`, `tt/generator.py:182`) — **not the same chip**, so a copy is unavoidable either way; cross-chip reuse adds a pipeline depth. SwiftSpec measured asymmetric device allocation at +37% e2e (§4.8.5) | +1.36 GB/chip (1.059 GB bf16 embedding + 0.298 GB bf4 LM head), 4.3% of 32 GB; the target pipeline shrinks from 32 chips to 31 | **medium** |
| **D-07** | **Lock `batch=1` in v1** (the server interleaving mode) | Support the `*_throughput` batch=8 directly | `activate_sessions` requires co-seated sessions at the same position (`tt/model.py:926`), and speculation makes positions diverge immediately | `*_throughput` profiles do not get DSpark yet | **medium** (D-03's per-slot form is exactly the unlock) |
| **D-08** | **Rewrite the confidence head**: from a `[vocab, hidden+rank]` pad to `[tile, hidden+rank]` | Leave it | The current form does a 129280×4352 matmul to produce one scalar, i.e. **562.6M params of wasted weight** (`tt/dspark.py:405-413`), five times per round at γ=5 | Requires relaxing `LinearDecode`'s N alignment or switching op | **easy** |
| **D-09** | **If we reach v3, tier-fill (spending tier padding on real draft tokens) defaults to ON** | Copy SGLang's default-off | SGLang's `--speculative-dspark-align-verify-tokens-to-graph-tier`, "at the same step time" (§4.2.5); **TT's tiers are coarser (32-row tile), so the payoff is larger** | Only takes effect at v3 | **easy** |
| **D-10** | **v1 does greedy (temperature=0) accept only**: compare the target's argmax against the draft token | Implement DeepSpec's rejection sampling directly | DeepSpec's rule is `min(1, p_target/p_draft)` plus residual resampling (§4.7.1), which requires the target to return **the full softmax for all K positions** = 5 × 129280 floats/step, while the D2H socket's page is 4032 B (`configs/system_configs.yaml:81`). Greedy needs only K integers | No lossless speculation at temperature>0; the server's existing top-k/top-p sampling (`demo/server.py:438-449`) is mutually exclusive with speculation in v1 | **medium** |
| **D-11** | **Do not change the single-token decode path's variant structure** | Apply the SCRATCH_ROW trick there too and collapse 5 variants into 2 | Unconditional pooling on the single-token path would raise CSA pooling from 1-in-4 steps to every step across 20 CSA layers — a **net cost increase**; and variant count is not that path's bottleneck | Keep 5 variants | **easy** |
| **D-12** | **v1 fixes γ=5 and does not implement the confidence scheduler** (the confidence head still runs, as host telemetry only) | Do dynamic γ from the start | ① Ascend lists dynamic verify length as a Non-goal and measured the confidence head as a **net loss** (+0.x AL, worse wall clock, §4.5.1); ② the DeepSpec reference defaults to `--confidence-threshold 0.0`, i.e. off (§4.7.6); ③ SGLang defaults to `static` (§4.4.3); ④ real traffic produces uniform verify lengths anyway (§4.4.4); ⑤ **fixed K is a precondition for Theorems 1 and 2** | Forgoes the share of the paper's gain attributed to the scheduler | **easy** |
| **D-13** | **No front-packing and no token-count bucket axis** | Port the zero hypothesis as-is | See the full table in §5.0; the core points are ① compact is not a throughput win on GPU (§4.3.3), ② their version is free because a cu_seqlens op already exists and ours does not (§4.1.2), ③ per-slot `cur_pos` is already varlen semantics | No per-request variable verify length before v3 | **medium** |
| **D-14** | **The γ serial Markov steps go inside the drafter's trace** | Outside the trace (Ascend's choice) | Ascend's stated reason is "avoid nested torch.compile", which **does not exist on TT**; upstream CUDA puts it in the graph; the current implementation makes 4 host round trips per step × 5 = ~20 per round (`tt/dspark.py:508-535`) and must be eliminated | Needs device-side argmax and `ttnn.embedding` usable inside a trace (the latter already is) | **medium** |
| **D-15** | **Do not change TP/EP for DSpark** | Do TP/EP first, then DSpark | TP/EP is independent work whose target shape is already written in the yaml and not implemented (`configs/system_configs.yaml:218-223`); **DSpark's payoff is actually larger under PP-only** (speculation is the only lever that amortizes the socket-hop latency in `f`, §5.5) | Galaxy's tok/s/u ceiling stays PP-limited | **easy** |
| **D-16** | **v1 lands in `demo/server.py` plus a new test, not via vLLM** | Go through `tt/generator_vllm.py` | `generator_vllm.py` explicitly declares speculation unsupported (`tt/generator_vllm.py:23-25`, `41-44`), and registration requires editing a plugin clone outside this repo (`tt/generator_vllm.py:14-22`) | vLLM integration deferred | **easy** |

---

## 7. Baseline definition

The DSpark paper's **60–85% is per-user generation speed against MTP-1 at matched throughput**, not against non-speculative decoding (§4.6.6), and **the paper names no hardware at all**. So we must define our own baselines and report all of them.

### 7.1 Three arms

| Arm | Definition | Purpose |
|---|---|---|
| **A0 — no-spec** | The existing traced single-token decode (`decode_traced`), unchanged | The true floor. **Every "×N speedup" must use this as the denominator.** |
| **A1 — γ=1** | DSpark with a draft block length of 1 (draft one token, verify two positions) | **The MTP-1 equivalent.** 0731 has no MTP head, so this is the only way to line up with the paper's and Ascend's tables |
| **A2 — γ=5** | The production configuration | The target |

**Report `A2/A0` and `A2/A1` together**, because every external number is in the `A2/A1` accounting.

### 7.2 Configuration matrix

| Dimension | Value | Note |
|---|---|---|
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark` (`README.md:38`) | Must be the DSpark one |
| Layers | **43 (full)** | Truncating layers changes `f`, so numbers from a truncated stack cannot be used for speedups |
| Weight dtype | `bfloat4_b` (`tests/test_full_model_decode_demo.py:52`) | |
| Hardware (primary) | **galaxy32**: 32× Blackhole Galaxy | The target platform. **Note the galaxy32 profile has never been measured** (`configs/system_configs.yaml:210`) → A0 on Galaxy is itself a new number |
| Hardware (control) | **p150x8**: 8× BH P150 | The only configuration with historical numbers (16.2 tok/s/u @ B=1, `PERFORMANCE_LOG.md`) |
| Trace batch | **1** (D-07) | |
| Concurrent users R | **{1, 4, 8, 32}** (server interleaving) | Covers the range where Ascend's table shows DSpark's advantage collapsing (§4.5.6) |
| Context | **{2048, 8192, 32768, 131072}** | ⚠️ **CSA is no longer equivalent to the reference above 2048** (`index_topk × cr = 512×4 = 2048`, U3). Accept-rate numbers above 2048 must carry that caveat |
| Sampling | **greedy (temperature=0)** | D-10; also removes sampling noise so accept rate is reproducible |
| Workload | Two fixed prompt sets: **the first 100 GSM8K questions** (reasoning) + **coding 80**. 8k in / 1k out | Matches Ascend's accounting (§4.5.6) for cross-comparison |

### 7.3 Metrics

**Performance**
- `TPOT` (ms/token) and `tok/s/user` — the headline metrics
- `aggregate tok/s` (across all R concurrent users)
- `T(M)`: step time versus tokens per step, M ∈ {1,2,4,6,8,12,16,24,32} → fit `f` and `c` (**this *is* the SPS table**)

**Speculation quality**
- `AL` (acceptance length, tokens committed per verify step) — **must be reported, or tok/s is uninterpretable**
- The **per-position accept-rate** vector (length γ). Ascend's reference values: `[80.73, 67.28, 57.11, 47.70, 40.65]%` (§4.5.6)
- The drafter's cost as a fraction of the verify step (target ≤ 10%, §5.6)

**Correctness**
- **Token-for-token identity with A0** (under greedy, speculation must be lossless: same prompt, same seed, A2's output must be **token-for-token identical** to A0's). This is the strongest correctness criterion, stronger than PCC.
- On a mismatch, fall back to layered PCC: a verify step's K logits versus the logits from K individual single-token steps

**Resources**
- Trace capture wall-clock time (report compile runs and captures separately)
- Trace DRAM per chip, and total DRAM occupancy

### 7.4 Reporting rules

Every number must carry the tuple **(profile, chips, layers, batch, R, context, γ, arm)**.
Every external number must carry its original hardware and configuration. The verified external anchors:

| Source | Number | Hardware / configuration |
|---|---|---|
| This repo's `PERFORMANCE_LOG.md` | 16.2 tok/s/u | 8× BH P150, batch=1, 43 layers, `a4b967209214` |
| This repo's `PERFORMANCE_LOG.md` | 4.3 tok/s/u @ "B=64" | 8× BH P150, `ceebba9f3ed` (⚠️ batch semantics questionable, §5.7) |
| vllm-ascend §4.5.6 | DSpark vs no-spec: BS1 **+124.8%**, BS16 **+50.8%** | 16 Ascend NPUs, TP4/DP4/EP16, W4A8, V4-Flash-DSpark, γ=5 |
| vllm-ascend §4.5.6 | AL **3.356** (SPEED-Bench) / **3.935** (coding80) | as above |
| LMSYS blog §4.1.6 | 383.7 tok/s @ accept≈5 | **B300 TP8, V4-Pro**, bs=1, `--cuda-graph-max-bs 4` |
| DSpark paper §4.6.6 | +51% aggregate @ 80 tok/s/u SLA; 60–85% per-user | **hardware unstated**; the baseline is **MTP-1** |
| sglang #34297 §4.4.5 | AL ~1.00–1.12 | random-token workload, simulated acceptance disabled — **not evidence of speculative gain** |

---

## 8. Phase 3: staged plan

Split along SGLang's `RaggedVerifyMode = {STATIC, CAP_ACCEPT, COMPACT}` (§4.4.3), with a Phase 0 in front.

### Phase 0 — measure the cost curve (**the only hard gate**)

| | |
|---|---|
| **Entry condition** | An available Galaxy (or p150x8) plus a populated weight cache |
| **Deliverables** | ① a `T(M)` table for M ∈ {1,2,4,6,8,12,16,24,32}, one for Galaxy and one for p150x8; ② fits for `f` and `c`; ③ the first measured A0 tok/s/u on galaxy32 |
| **Verification** | Run `tests/test_multi_user_paged_decode_demo.py` with a batch sweep (experiment E1). **No code changes.** |
| **Effort** | **~2 person-days** (including queueing and trace capture time) |
| **Fallback on failure** | If `c` is large enough that `T(5) ≈ 5·T(1)` (i.e. MoE's per-token cost dominates), then **the whole DSpark scheme is capped at 1.0× and should be stopped immediately** in favour of TP/EP. This gate must be passed first |

### Phase A — `static`: fixed γ, fixed K verify

| | |
|---|---|
| **Entry condition** | Phase 0 shows `T(5) / T(1) < 2.5` (i.e. any accept length above 2.5 is a net win) |
| **Deliverables** | ① relax the `assert s == 1` at `tt/attention.py:1378` to support `S=K`; ② per-slot positions in the packet; ③ per-token RoPE (`_apply_rope` already accepts a per-row table); ④ **D-03's deferred pooling + merged ring buffer**; ⑤ SCRATCH_ROW allocation and device-side clamping (D-05); ⑥ one new verify trace variant; ⑦ an **oracle drafter** (feed A0's ground-truth tokens as the draft, 100% accept) to isolate correctness and performance measurement |
| **Verification** | **Token-for-token identity**: under the oracle drafter, the j-th logits of a K-token verify step must be **token-for-token identical** to the j-th logits from individual single-token steps (greedy argmax equality suffices; PCC as a diagnostic). Cross-window cases must cover all four residues `q mod 4 ∈ {0,1,2,3}` |
| **Effort** | **~15–20 person-days** (estimate; D-03 is half of it) |
| **Fallback on failure** | If D-03's merged ring buffer turns out infeasible (e.g. `ttnn.where` is unusable at that shape), fall back to "row-wise conditional retire" (§5.4's alternative b): `cr=4` extra small writes per CSA layer per pooling. If that also fails, fall back to "the verify step touches no compressed state, with a separate commit trace", which loses most of the gain → the project degrades to a research attempt at "speculate on sliding layers only" |

### Phase A' — connect the real drafter

| | |
|---|---|
| **Entry condition** | Phase A's oracle identity check passes |
| **Deliverables** | ① drafter on chip 31 with embedding/lm_head copies (D-06); ② hidden-state tap on the target's last submesh plus the socket send (including the U8 tap-point check); ③ **the drafter fully traced, with the γ Markov steps inside the trace** (D-14) and device-side argmax; ④ fix the confidence head's vocab padding (D-08); ⑤ host-side greedy accept (D-10) |
| **Verification** | ① **a full generation token-for-token identical to A0** (the losslessness criterion); ② AL and per-position accept rate against Ascend's `[80.73, 67.28, 57.11, 47.70, 40.65]%`; ③ drafter cost ≤ 10% of verify |
| **Effort** | **~15–20 person-days** |
| **Fallback on failure** | If the drafter's backbone really is 256-expert MoE (U7) and does not fit one chip, fall back to two chips (target shrinks to 30). If AL < 2, check the tap point (U8) and CSA's 2048 equivalence (U3) before deciding whether to continue |

### Phase B — `cap-accept`: scheduling without changing shapes

| | |
|---|---|
| **Entry condition** | Phase A''s end-to-end number is positive, and the confidence head's output is being recorded as telemetry |
| **Deliverables** | ① a per-request accept cap (dense `[B, cap]` layout — **the shape stays static, so no new trace**); ② turn confidence telemetry into actual truncation decisions; ③ the host-side ring plus generation counter for the two-step-back relay |
| **Verification** | ① assert the shapes did not change (trace count does not grow); ② the tok/s delta against Phase A'; ③ **an A/B with the relay disabled and fresh confidence is mandatory**, since the paper states the lag is a necessary condition for losslessness (§4.6.4) — we must verify our implementation really is lossless |
| **Effort** | **~10 person-days** |
| **Fallback on failure** | Ascend measured the confidence head as a **net loss** (§4.5.1). **If we measure a net loss too, stop at Phase A' and write the conclusion down** — that is not a failure, it is agreement with two independent backends |

### Phase C — `compact`: token-count tiers and variable length

| | |
|---|---|
| **Entry condition** | Phase B shows dynamic γ is a net win (**if B is a net loss, C is not done**); and per-slot pooling for `batch>1` is complete (unlocking D-07) |
| **Deliverables** | ① an offline SPS-table profiler (sample points = our actually-captured tiers); ② an asynchronous global top-k allocator (**no early-stopping**, §5.7); ③ the tier set `{16, 32}`; ④ tier-fill on by default (D-09); ⑤ a constructive guarantee that the top tier cannot be exceeded (§4.2.1) |
| **Verification** | ① a knob to force ragged shapes (the equivalent of SGLang's `dspark_force_budget_frac`), because natural traffic does not produce them (§4.4.4); ② a **standalone plan harness** that loads no model and only checks `max(write index) < legal row count` — the method #33412's reporter used to localize the root cause (§4.3, §4.10) |
| **Effort** | **~20 person-days** |
| **Fallback on failure** | Compact is not a throughput win on GPU to begin with (§4.3.3). **The default expectation is that we do not do it.** |

### 8.1 Answering separately: with static fixed-length verify only, how fast can we get a measurable end-to-end number?

Two milestones, because their value and risk differ a lot:

**Milestone 1 — "what is a verify step worth", ~2 person-days, no code changes.**
Run experiment E1 (the batch sweep) to get `T(M)`. This directly gives **the ceiling of the whole project**: `ceiling = (γ+1)·T(1) / T(γ)`. If that number is below 1.5, none of the rest is worth doing. **This is something to do today.**

**Milestone 2 — "the first real end-to-end tok/s", estimated ~4–6 weeks (Phase A + A').**
But there is an earlier number, at ~2.5–3 weeks (Phase A alone): **end-to-end under the oracle drafter**.
- It runs the full K-token verify with real KV and compressed-state maintenance; only the draft tokens come from A0's ground truth (100% accept).
- It yields the **upper bound of the speedup** (the measured version of `(γ+1)·T(1)/T(γ)`), and **fully validates the hardest part of this design (§5.4's rollback semantics)**.
- It needs no drafter on device, no hidden tap, and no in-trace Markov head.

> **Recommended shortest path: E1 (2 days) → Phase A + oracle drafter (~3 weeks) → decision point → Phase A' (~3 weeks).**
> Estimates assume one engineer already familiar with this codebase; without that familiarity, D-03 alone doubles. **These are estimates, not commitments.**

---

## 9. Open questions / unknowns, each with a minimal falsifying experiment

**Rule**: every entry here is something the code does not answer, or something that conflicts with an external source. **None of them is filled in with a plausible guess.**

| # | Question | Why it matters | Minimal falsifying experiment |
|---|---|---|---|
| **U1** | The wall-clock time of trace capture, and how much DRAM each variant actually occupies | Determines the tier budget in §5.1. The code has only prose ("minutes", `README.md:91`), no numbers; every profile uses `trace_region_size: 0` = dynamic allocation, no reserved region, no bound | **E2**: timestamp around `_capture_traces` (`tt/model.py:1997`) and log per variant, while reading `mesh_device`'s DRAM usage before and after capture. One run of `test_full_model_decode_demo.py` suffices |
| **U2** | The actual `layer_types` distribution in the 0731 checkpoint | §2.2's per-token KV footprint (5304 B/token) is computed from the **default construction** (`configuration_deepseek_v4.py:262-267` → 23 HCA + 20 CSA + **0 sliding**). If the checkpoint states something else, both the KV budget and the pooling phases must be recomputed | **E3**: `python -c "import json;c=json.load(open('<snapshot>/config.json'));print(c.get('layer_types') or c.get('compress_ratios'))"` |
| **U3** | The ttnn side **does not implement** the Lightning Indexer (top-k=512); CSA does dense SDPA over every compressed entry. The author states this is equivalent only for `seq_len ≤ index_topk × cr = 2048` (`tt/attention.py:643-647`) | **Above 2048 context our target silently deviates from the reference model.** This affects the drafter's accept rate too (drafter and target run on the same deviating implementation, so accept rate could be over- or under-stated), contaminating **every long-context speculation number** | **E4**: run A0 on the same prompt at context = 1024 (equivalent regime) and 8192 (non-equivalent), comparing token-for-token against the HF reference (`modular_deepseek_v4.py`). If 1024 matches and 8192 does not, U3 is confirmed |
| **U4** | Whether `ttnn.embedding` accepts a `bfloat8_b` ROW_MAJOR weight table | Determines whether the drafter's embedding copy is 1.059 GB (bf16) or about half that (D-06 chose the bf16 copy; if bf8 works it saves ~0.5 GB/chip) | **E10**: a small unit test with no model — `ttnn.as_tensor(w, dtype=ttnn.bfloat8_b, layout=ttnn.ROW_MAJOR_LAYOUT)` then `ttnn.embedding`, compared by PCC against bf16 |
| **U5'** | The brief says the drafter has **5 layers**; the code says `num_stages=3` (`dspark.py:133`). **The paper agrees with the code**: the production V4 drafter is 3 MoE layers + mHC + a 128 sliding window, and "DSpark-5" means γ=5 (§4.6.5). **What remains is only a checkpoint cross-check** | Determines the drafter's size and placement | **E5**: read the checkpoint's `config.json` and confirm the stage count under the `mtp.*` namespace, plus `dspark_block_size`, `dspark_markov_rank` and `dspark_target_layer_ids`. **Also pins D-02's off-by-one convention** |
| **U6** | A0's measured tok/s/u on galaxy32 | The profile labels itself "**STARTING POINT, NOT A MEASURED TUNE**" (`configs/system_configs.yaml:210`). **We currently have no baseline on Galaxy at all** | **E1** (below) |
| **U7** | Whether the 0731 DSpark backbone really is 256-expert MoE | The paper says **3 MoE layers + mHC** (§4.6.5), but our `dspark.py` implements a **dense MLP with a standard residual**, labelled by the author as being for standalone unit testing (`dspark.py:17-18`). **If it really is 256 experts, the drafter weighs 3 × 3.62 GB = 10.9 GB and will not fit one chip** (which also has to hold the 1.06 GB embedding plus KV) | **E5** (the same `config.json`): check whether expert weights exist under `mtp.*`, and what `n_routed_experts` is |
| **U8** | `dspark_target_layer_ids=(40, 41, 42)` (`dspark.py:142`) **includes the last layer** of a 43-layer model, while DeepSpec carries a hard assertion `assert_no_final_target_layer` (§4.7.7) and released configs leave headroom (Qwen3-4B's 36 layers use [1,9,17,25,33]) | If the tap point is wrong, the drafter receives the final normalized hidden instead of a layer output, and **accept rate will be low without any error** | **E5**: `dspark_target_layer_ids` in `config.json`, reconciled with the HF reference's `output_hidden_states` indexing convention |
| **U9** | Progress on tt-metal mainline issue #50475 (qwen3.6 multi-token GDN decode/verify kernel); whether mainline already has a multi-token paged-cache update or a ragged attention op | If it does, part of Phase A can be reused | **E6**: `gh issue view 50475 --repo tenstorrent/tt-metal --comments`; `git grep -n 'paged_update_cache\|paged_fill_cache\|ragged' origin/main -- ttnn \| head -40`; `git log --oneline -15 origin/dchrysostomou/experiment_speculative_deepseek` |
| **U10** | Whether the D2H socket can carry K positions of logits | The page size is `d2h_fifo_bytes: 4032` (`configs/system_configs.yaml:81`), whose comment reads "One 4 KB system page minus the bytes_acked counter: larger fails to pin on an IOMMU-less host". K=5 × 129280 × 2 B = **1.29 MB/step = 320 pages** (greedy needs only K integers, but top-k sampling needs the full logits) | **E8**: compute the page count for K=5 via `_d2h_page_plan` (`tt/model.py:1955-1962`), and measure how `read_decoded_output`'s duration scales with batch on the existing batch=8 demo |
| **U11** | Whether `ttnn.where` / `ttnn.ge` / `ttnn.minimum` work on `[1,1,16,1024]` bf16 **inside trace capture**, without introducing a host sync | **The critical dependency of D-03 (the merged ring buffer).** If not, the whole §5.4 design falls back to alternative (b) | **E9**: a small unit test with no model that runs `where(ge(idx_table, dev_scalar), a, b)` inside a trace, captures once and replays twice with different `dev_scalar`, checking the output changes accordingly |
| **U12** | `ttnn.argmax`'s behaviour inside a trace, and whether a device-resident position counter can coexist with the in-trace `recv_async_h2d` | **The precondition for Plan X (taking the host out of the speculation loop, §5.3).** The author has already identified the obstacle and the fix (`tt/model.py:2213-2214`) but did not implement it | **E7**: split the packet as the author describes (positions over the socket, the token from a separate device-written buffer) and get a minimal two-step `decode_sampled_burst` working. This experiment's outcome decides v3's shape |
| **U13** | Whether the 0731 DSpark checkpoint actually ships **trained** confidence-head weights | SGLang raises a `ValueError` telling you to fall back to `static` when the confidence head is missing (§4.4). If absent, Phase B is impossible from the start | **E5**: look for `mtp.*.confidence_head.*` in `config.json` and the safetensors index |
| **U14** | The real cost of the per-step seat-group swap on galaxy32 | The code quantifies a comparable operation (per-session swap ~100 ms, `tt/model.py:912-917`), but that was 8 chips; on 32 chips layers are more spread out. Speculation does not change the session-switch frequency, but **this becomes the dominant overhead at R>1** | **An extension of E1**: alongside the batch sweep, record how `activate_sessions`'s duration scales with R |

### Experiment E1 (highest priority, no code changes)

```
pytest -s models/experimental/deepseek_v4_flash/tests/test_multi_user_paged_decode_demo.py
  # sweep batch ∈ {1,2,4,6,8,12,16,24,32}, num_users ≥ batch
  # ≥ 200 steady-state steps per point; take the median step time
```
**Output**: the `T(M)` table → fits for `f` and `c` → the project's ceiling `(γ+1)·T(1)/T(γ)`.
**Falsification condition**: if `T(5) / T(1) ≥ 2.5` (MoE's per-token cost dominates), **DSpark is not worth doing on this implementation** and the effort should move to TP/EP.
**Note**: `_pack_tokens` asserts `B·S ≤ 32` (`tt/attention.py:347-349`), so 32 is the batch ceiling; also settle what the "B=64" point in `PERFORMANCE_LOG.md` actually meant.

---

## 10. Risk register (by severity)

| Risk | Severity | Mitigation |
|---|---|---|
| **R1 — D-03's merged ring buffer depends on `ttnn.where`'s behaviour inside a trace (U11)** | **High**: it is the only clean solution in §5.4 | Run E9 first; it takes half a day. The alternative is "row-wise conditional retire" (4 extra small writes per CSA layer per pooling) |
| **R2 — TT has no IMA trap, so an out-of-bounds index is a silent overwrite** | **High**: the loud crash a GPU user gets is, for us, a silently wrong answer with no observation point inside the trace (§4.3.4) | D-05's device-side clamp is mandatory; copy #33412's standalone plan-harness method (no model loaded, only checking `max(index) < legal row count`) |
| **R3 — U3: CSA is not equivalent to the reference above 2048 context** | **High**: contaminates every long-context accept rate and gain number | Confirm with E4 first; until then report accept rates only at context ≤ 2048 |
| **R4 — U7: if the drafter is full 256-expert MoE it will not fit one chip** | **Medium** | E5; the fallback is two chips for the drafter |
| **R5 — the confidence scheduler may be a net loss** | **Medium**: Ascend measured exactly that (§4.5.1) | D-12 keeps it out of v1; Phase B explicitly says "if we measure a net loss, stop at A'" |
| **R6 — galaxy32 has never been measured, so A0 is itself unknown** | **Medium** | E1 produces A0 at the same time |
| **R7 — capture time grows linearly with variant count, and compile runs must precede all captures** | **Low–medium** | §5.1 holds verify to +1 variant per submesh; E2 quantifies it |
| **R8 — a losslessness regression** (under greedy, A2 must be token-for-token identical to A0) | **Medium** | Make "token-for-token identical" a CI criterion, not PCC |

---

## 11. What this document did not do

- **No kernels were written and no implementation code was changed.** The only change on this branch is this file.
- **No device experiments were run.** Every conclusion that needs hardware is written up as a U/E entry.
- **No performance claims without provenance.** §5.7's speedup table is derived, with three caveats attached to the two-point extrapolation; every external number carries its hardware and configuration.
- **No CUDA-graph conclusion was transplanted onto TT-NN trace.** §0's Finding 1 distinguishes the two explicitly (the in-trace H2D socket makes TT's *value* dynamism stronger than a CUDA graph's, while its shape freezing is stricter).
- **Three of the brief's premises were corrected** (all with sources):
  1. "5-layer parallel backbone" → **3 MoE layers**; "DSpark-5" means **γ=5** (paper §4.6.5 + code `dspark.py:133`)
  2. "KV injection from target layers" → **hidden-state injection**; the drafter manufactures context K/V with its own k_proj/v_proj over the projected target hidden states (DeepSpec, §4.7.4)
  3. "#33412's IMA is in the c4 compressor store" → that is where it **surfaces**; the root cause is a plan-builder warp race, and the "4-alignment" hypothesis was explicitly refuted by the reporter (§4.3.1–2)
  Additionally: the brief's `T(bs,K)=bias+alpha(bs)+theta(M)` has **no source** in anything retrieved this session; the paper uses a single-argument `SPS(B)` (§5.7).
  And: the brief's "MLA latent stored fp8/uint8, sparse indexer group stored fp32" — in this code the **KV pools are uniformly bfloat16** and the **indexer group does not exist** (§2.2, §2.3).
