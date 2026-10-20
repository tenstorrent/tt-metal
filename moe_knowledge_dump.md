# MoE prefill on Blackhole: knowledge dump

What was worked out, how the numbers were reached, why they matter, and what follows for buffers,
dispatch, combine and scheduling. Written 2026-10-15 for future agents picking up this work. Numbers
are roofline floors unless they are explicitly marked as measured.

---

## 0. Where things live

| What | Where |
|---|---|
| MoE Block Roofline tracker (main's K2.7 op graph, Table 1 stage floors, Tables 3–4 residual layouts) | https://claude.ai/artifact/6yYVqSgsx5Z7LfWJPgKDwx |
| MoE Dispatch Traffic (1D column dispatch model, overlap pipeline, buffers, DRAM traffic; full-mesh 2D Monte Carlo with link map) | https://claude.ai/artifact/6MVAyNb16j1bZ9ptQwCKuF (`#mesh` for 2D) |
| MoE Residual Layouts (6 layouts, op-by-op TP-chip diagrams with per-op floors and formulas, rooflines, attention cost, weights per chip and KV headroom, MoE peak buffers, router) | https://claude.ai/artifact/BqHotRgTguApZidjdH69Q2 |
| K3 MoE Block Bring-up tracker (db-backed: op status, measured vs floor, overlap %, worklog, plan; seeded with EXAMPLE data) | https://claude.ai/artifact/1s4GCUpB8iw73asETjLtKM |
| Text versions | repo root: `moe_residual_layouts_shared_expert.md`, `moe_token_split_tp_se_plan.md`, this file |
| MiMo reference implementation of the token-split ("sequence-parallel") residual | `/localdev/mstaletovic/tt-metal-mimo/models/demos/mimo_v2_d_p/tt/` (`decoder.py` `forward_sp`, `ffn.py` `reduce_scatter_tp` / `TpGather` / `partition_tp`, `moe_ag.py` `gather_full`) |
| Main's MoE block | `origin/main:models/demos/deepseek_v3_d_p/tt/` (`tt_prefill_block.py`, `moe/tt_moe.py`, `moe/init_helpers.py`, `kimi_k3/`) |
| Agent memory (preferences, artifact schemas) | `~/.claude/projects/-localdev-mstaletovic-tt-metal/memory/` (`moe-roofline-tracker.md`, `k3-small-stage-focus.md`, `k3-moe-bringup-tracker.md`) |

The pages are interactive: model (K3 / GLM-5.3 / K2.7), grid (8 × 4 / 4 × 4), chunk size, TP axis
ring or line, fused or sequential matmul + collective, layers per stage. Their sources are kept in
the session scratchpad, which gets wiped between days; to edit a page, pull it with the Artifact
tool (`action: read`, `path: index.html`), strip the published `<!doctype …><body>` wrapper line and
the trailing `</body></html>`, edit, republish to the same URL.

---

## 1. Scope, hardware and the user's priorities

- **Target**: prefill MoE on Blackhole Galaxy stages. Rooflines are for **8 × 4** (SP = 8 rows,
  TP = 4 columns) and **4 × 4** stages only. The LoudBox (2 × 4, P150) is for measurement and
  bring-up, never a roofline target.
- **Model priority**: Kimi **K3** on a **4 × 4** stage first, GLM-5.3 second, K2.7 least.
- **Block scope** for the current work: the MoE block only. Entry right after the norm that feeds
  the MoE; exit after routed + shared contributions are summed (before the residual add).
  Attention, AttnRes and the residual adds are out of scope.
- **Hardware constants used everywhere** (state them when you quote a floor):

| Constant | Value |
|---|---|
| Ethernet link | 25 GB/s per direction (Galaxy, 200G), 2 links per direction per neighbour → 50 GB/s per direction |
| DRAM | 512 GB/s per chip, 32 GB per chip |
| Compute | LoFi 4,096 FLOP/cycle/core at 1.35 GHz; 120 cores (12 × 10 Galaxy grid) for edge ops, 108 for the expert/middle phase (one row reserved for dispatch) |
| Topology | rings with wrap on both axes (8 × 4 production is `fabric2d-torus-xy-8x4-2link`); TP can also be modelled as a line |

- Working style the user asked for: act on validated results; minimal repros; perf-only iterations
  with the KV ladder at the end; worklog per experiment; report DRAM % of 512 GB/s and math % for
  expert ops; few numbers in chat when explaining concepts.

---

## 2. The models

| | K3 | GLM-5.3 | K2.7 |
|---|---|---|---|
| Hidden H | 7,168 | 6,144 | 7,168 |
| Routed experts E, top-k | 896, 16 | 256, 8 | 384, 8 |
| Routed width (LatentMoE) | **3,584** (latent down H → 3,584, up 3,584 → H) | 6,144 | 7,168 |
| Expert intermediate I | 3,072 | 2,048 | 2,048 |
| Shared expert I_s | **6,144** (2 × 3,072, one MLP) | 2,048 | 2,048 |
| Layers (MoE) | 93 (92); 24 MLA + 69 KDA | 78 (75) | 61 (60) |
| MLA q_lora, kv | 1,536, 512 + 64 rope, 96 heads, output gate | 2,048, 512 + 64, 64 heads, DSA indexer | 1,536, 512 + 64, 64 heads |
| KDA | 96 heads × 128, bf16 weights, head-split over TP | – | – |
| Other | AttnRes residual (softmax over 12-layer block snapshots) | | |

Dtypes on main: routed experts bfp4 (0.5625 B/value), shared expert / MLA / latent projections bf8
(1.0625 B), router weight bf16, KDA bf16, rows on the wire bf16.

K3's routed weights are about 16.6 GB per layer in total: on a 4 × 4 stage that is about 1.04 GB per
chip per layer, so a 4 × 4 stage holds roughly 20 layers. K3 needs several stages.

---

## 3. Main's MoE block today (K2.7 on Galaxy 8 × 4, `TtMoe.forward`)

Residual between ops is **split over H**: `[S/SP, H/TP]` per chip (block docstring: "TP-sharded on
last dim").

1. Distributed RMSNorm: local Σx², all-gather of per-row sums over TP, normalize.
2. Router: `[q, H/TP] · W_r[H/TP, E]` (weight split over H) → partial logits → **fp32 logits
   all-reduce** over TP → `moe_grouped_topk` on all q tokens on every TP chip (4× redundant).
3. **All-gather x over H** → `[q, H]` on every chip of the row.
4. **Column dispatch** (`dispatch` v1 or `dispatch_fabric2d` v2): a dispatch group is one TP column
   (8 chips along SP); each column holds E/TP experts and sends every token's ~k/TP slots that land in
   its column, one row per slot, along the SP ring.
5. Shared expert (column-split gate/up, row-split down) on the gathered x, overlapped with dispatch on
   separate sub-devices (dispatch on 1 grid row, shared on the rest), then **reduce-scatter over H**.
   That reduce-scatter is **forced to Linear** while it overlaps dispatch: a ring reduce-scatter
   running next to dispatch on shared EDM routers deadlocked (credit cycle).
6. Routed experts, combine back along SP (lands every slot: `[q, k, H]`), `post_combine_reduce`
   (weighted sum of the column's slots), **reduce-scatter over H**, add shared.

Facts about main's dispatch ops:
- v1 `dispatch` uses only its own row of the count table; relays happen inside the fabric routers.
- v2 `dispatch_fabric2d` / `combine_fabric2d` take the full count table but their schedule is a
  **static host plan** from the ring size and `num_links` (equal fractions, own destinations
  furthest-first, then forwards). No data-dependent planning. Relays go through a **DRAM forwarding
  buffer** sized for the per-pair worst case, q·min(k, E/chips)·(m−1)² (664 MB per chip on 8 × 4
  against 20.6 MB used in the happy case), because there are no credits, only an arrival counter.
- Main sizes the dispatch output as **capacity factor × every token of the dispatch group** plus a
  tile of padding per extra local expert (`compute_constants` in `init_helpers.py`; production
  cf = 2). Rows past capacity are **dropped silently** (`reader_worker_dispatch.cpp`: the offset still
  advances, no entry is written).
- Main's MLA never gathers x: q_a and the kv stem take H slices, the q latent is reduce-scattered +
  all-gathered, the kv partials all-gathered (narrow latents, 2,112 wide for K2.7). Main's K3 KDA
  layers do all-gather x over TP on the way in and reduce-scatter on the way out. Main's AttnRes
  all-reduces partial scores over TP on every read.

---

## 4. Roofline method (how every number on the pages is made)

### 4.1 Collectives on one axis (TP = 4)
- All-gather or reduce-scatter of a `[rows, w]` bf16 tensor: busiest link carries **1.5 of 4
  shards on a ring**, **2 on the middle link of a line**: `sh × rows × w × 2 B / 4 ÷ 50 GB/s`.
  For K2.7 on 8 × 4, an x-sized collective (`[640, 7168]`) is about **69 µs**; on 4 × 4 (1,280 rows)
  about 138 µs.
- All-reduce = reduce-scatter + all-gather (2×). Router logits are fp32 (the selection bias sits
  where bf16 steps are coarser than the 8th/9th expert gap, so bf16 flips choices).
- All-to-all over TP = 4: 2 parts per directed link on a ring, 4 on the middle link of a line, each
  part `q/TP × w/TP × 2 B`. An all-to-all of reduced data costs about a third of a reduce-scatter
  but does not reduce.
- Matmul: max(DRAM bytes ÷ 512 GB/s, FLOPs ÷ LoFi peak).
- Matmul + collective pairs at the edges: either summed or **fused (max of the two)**; the pages
  default to fused. Fusing saves only the matmul share (~25–30 µs per pair) because the collectives
  dominate.

### 4.2 Column dispatch (1D, one row per slot)
Ring of G chips along SP, N = 4 groups, m = G/2, q tokens per chip.
- r = q·k/(N·G) rows per (source, destination) pair.
- Busiest directed link: **r·m²/2 = T·k/(8N)** rows (opposite chip split both ways). Independent
  of G at fixed chunk T. K2.7 8 × 4: 1,280 rows × 14 KB ÷ 50 GB/s = **367 µs**; the same on 4 × 4.
- All-gather reference: q·(m − ½) rows per link. Per-slot dispatch puts more on a link than an
  all-gather above **k\* = 4N(G−1)/G** (14 at G = 8, 12 at G = 4).
- One row per destination chip instead of per slot saves only a few percent (two slots rarely land
  on the same chip).

### 4.3 Full-mesh (2D) dispatch
Each chip owns q/TP tokens and sends each to its experts anywhere on the stage.
- **Per-axis load, one row per slot, uniform routing**: rows per directed link = q_own·k·n/8 for a
  ring of n chips. The longer ring sets the floor; route order does not change per-axis totals. On
  8 × 4 the SP load equals column dispatch's (367 µs); the TP leg adds half of it in parallel.
- **What is sent** is the lever:
  - one row per slot;
  - one row per (token, destination chip);
  - **drop-off**: one row travels along a ring, every chip that needs it keeps a copy and forwards it.
    Cost on a ring = the cheapest "a hops one way + b hops the other" covering every needed chip
    (ties split evenly).
- **Why 2D beats 1D (the intuitive answer)**: column dispatch fans the token out over TP first (the x
  all-gather), so the long SP ring is travelled once per column (TP times per token). SP-first
  drop-off travels the long ring **once** and fans out on the short TP ring. Fan-out is cheap on a
  short ring and expensive on a long one. Per slot, 1D and 2D are identical.
- **Fixed split**: send a fixed share of tokens SP leg first (picked by a hash of the token index),
  the rest TP leg first; the share balances the busiest TP link against SP under **uniform** routing,
  so it depends only on mesh, k and E: 50% on a square stage, ~88% on 8 × 4 (K2.7), ~75% K3 8 × 4,
  ~43% with a TP line. One row per slot or per chip puts the same totals on each axis in either
  order, so those use 50%.
- Monte Carlo (dispatch page; 20–40k tokens, seeded) gives hops per token. K2.7 on 8 × 4:

| Sent | TP rows / link | SP rows / link | Busiest |
|---|---|---|---|
| One row per slot | 640 | 1,280 | 367 µs |
| One row per chip | 579 | 1,159 | 332 µs |
| Drop-off, TP leg first | 217 | 898 | 258 µs |
| Drop-off, SP leg first | 526 | 436 | 151 µs |
| Drop-off, fixed 88% split | ~490 | ~490 | ~140 µs |

  Sanity checks: per slot matches the closed form; TP-first drop-off costs 2.7 TP hops ≈ an
  all-gather (3); as k → ∞ drop-off becomes the all-gather (2,240 SP rows). 4 × 4 K2.7: per slot
  367, drop-off 251 either pure order, 50/50 split ~188 µs. K3 8 × 4: 367 → ~109 µs; K3 4 × 4:
  drop-off ~127 µs, one row per slot back ~367 µs.
- **TP as a line**: the middle links carry about 2× the ring mean (Monte Carlo over owner columns).
- **Skew** (synthetic Zipf / hot set on the dispatch page): senders stay balanced by construction
  (each owns q tokens); receivers skew. A hot chip has 4 incoming directions in 2D vs 2 in column
  dispatch. Hot experts **packed** on one chip (consecutive expert ids are placed together today) hurt;
  **spread** placement removes almost all of it.

### 4.4 Routed experts
Per chip: weights `eC × 3 × H_r × I × 0.5625 B` + slot rows in and out ÷ 512 GB/s, against LoFi
FLOPs. K2.7 8 × 4: 297 MB → ~652 µs. K3 4 × 4: 56 experts, ~1.04 GB → **~2.2 ms**, DRAM-bound.
The weight read is fixed per chunk; link time and compute grow with the chunk.

### 4.5 Overlapped middle vs edges
- **Middle** = dispatch ∥ routed experts ∥ combine ∥ shared expert. Its floor is the busiest resource
  summed over all of them: DRAM (weights, rows, shared expert, dispatch/combine DRAM traffic), SP
  links, TP links, compute.
- **Edges** = everything before dispatch can start or after combine finishes. Nothing hides them.
- **Serial** = dispatch phase (shared expert under it), then experts, then combine.

---

## 5. Key results (floors, per chip, chunk 5,120)

### 5.1 Main's structure (MoE Block Roofline, Table 1)
| | 8 × 4 K2.7 | GLM-5.3 | K3 | 4 × 4 K2.7 | GLM-5.3 | K3 |
|---|---|---|---|---|---|---|
| Dispatch (= combine) | 367 | 315 | 367 | 367 | 315 | 367 |
| Routed experts | 652 | 393 | 1,088 | 1,305 | 786 | 2,175 |
| Serial | 1,594 | 1,197 | 2,162 | 2,453 | 1,765 | 3,563 |
| Overlapped | 1,044 | 804 | 1,649 | 2,065 | 1,432 | 3,202 |

Overlapped, everything is DRAM-bound by bfp4 expert weights except GLM-5.3 on 8 × 4 (SP-link-bound).
So dispatch/combine are judged by the **DRAM bytes they add** when overlapped, not by link time;
relays in DRAM (v2) erase most of the overlap gain.

### 5.2 Residual layouts, K3 on 4 × 4, chunk 5,120, fused edges (MoE Residual Layouts)
| Layout | Overlapped | Serial |
|---|---|---|
| Current main | ~3.12 ms | ~3.48 ms |
| Main, tuned (top-k once, combine adds on arrival, one reduce-scatter) | ~2.89 | ~3.38 |
| Token-split · replicated shared expert | ~2.73 | ~2.89 |
| **Token-split · TP-split shared expert** | **~2.57** | ~2.95 |
| Gather → 2D owners | ~2.77 | ~3.01 |
| Sliced, never gathered | ~2.81 | ~3.19 |

The middle is DRAM-bound by the ~2.2 ms weight stream in every layout. The 15–20% between layouts
is mostly the **edges**: current main spends ~0.6 ms per layer on router all-reduce (896-wide fp32
logits), x all-gather, latent projections with collectives, weighted sum and reduce-scatters; the
token-split layouts spend ~0.14 ms. **Even serially, every 2D layout beats current main fully
overlapped**, so for K3 on a small stage the overlap work is worth ~13% and can come last.

---

## 6. Residual layouts and the shared expert (why token-split)

- After o_proj every TP chip holds a full-width partial for every token of its SP row, so the
  finishing reduce-scatter can scatter **H** (`[S/SP, H/TP]`, main) or **tokens**
  (`[S/(SP·TP), H]`, MiMo's `sp_residual`) at the same cost. Splitting by tokens makes the norm,
  residual add, router and AttnRes local, and leaves each chip owning whole tokens, which is what 2D
  dispatch needs.
- Row split alone (with column dispatch) saves almost nothing: the x all-gather and reduce-scatters
  cost the same on rows as on H; only the logits all-reduce goes.
- **Layouts compared** (page and `moe_residual_layouts_shared_expert.md`):
  - *Current main*; *Main, tuned* (fair reference).
  - *Token-split · replicated SE*: no collectives, but the whole shared expert is read on every
    chip (K3: 140 MB per layer).
  - *Token-split · TP-split SE*: the shared expert does its own row all-gather and reduce-scatter
    over tokens, alongside dispatch. Best for K3.
  - *Gather → 2D owners*: main's layout up to the x all-gather, then each chip declares ownership of
    a quarter of the row's tokens and dispatches in 2D; combine lands H slices on every chip. Same
    collectives as token-split TP SE, but the gather sits on the critical path; attention unchanged.
  - *Sliced*: never gather; shared expert split on H (the K dim of gate/up), reduce-scatter the
    2·I_s-wide partials instead of gathering H-wide x (the MLA trick); dispatch H slices. Pays only
    when **2·I_s < H** (K2.7, GLM yes; **K3 no**).
- **Attention cost** of a token split (outside the MoE scope): MLA's narrow latents favour the H
  split (K2.7 8 × 4: latent collectives ~52 µs vs x all-gather 69 µs; local projections with
  replicated bf8 q_a/kv weights + latent row gather ~51 µs). KDA is neutral (it gathers x anyway).
  AttnRes gets cheaper (score all-reduces disappear).
- **Router**: H-split weight + fp32 logits all-reduce + top-k on every chip, against a replicated
  weight (K3: 12.8 MB bf16 per layer) on own tokens with top-k once. Same FLOPs per chip. 2D dispatch
  needs a small **count exchange** (all-gather of `[chips × E]` histograms) before sending.
- **Weights per chip and KV headroom** (K3, 4 × 4, 20 layers): ~25.5 GB of weights in current main
  (routed 20.8 GB, attention ~3.6 GB, mostly bf16 KDA); +3.2 GB for replicated shared expert and
  latent projections, +1.1 GB for token-split TP SE (replicated router and latent projections).
  Left for the KV cache after the MoE peak buffers: ~6.1–6.3 GB (main, tuned, gather → 2D owners,
  sliced), ~5.3 GB (token-split TP SE), ~3.2 GB (replicated SE). Capacity, not DRAM time, is the
  real cost of replication on a small stage.
- **Compute-bound?** Every matmul is costed as max(DRAM, compute). The shared expert is
  DRAM-bound in the model mostly because its activations go through DRAM; with x and its output in
  L1, weights and compute balance at about 620 tokens per chip-row and it is compute-bound above.
  Routed experts at LoFi peak turn compute-bound around 16k tokens on 8 × 4. Real math utilisation is
  well below peak, so in practice ops turn compute-bound earlier.

---

## 7. Buffers (routing-independent sizes; nothing dropped, trace-safe)

| Buffer | Safe size | Grows with |
|---|---|---|
| Expert input, one row per slot | T × min(k, E/chips) | chunk, top-k, experts per chip (8× the all-gather buffer for K2.7) |
| Expert input, one row per (token, chip) or drop-off | **T** | chunk only (= the all-gather buffer) |
| Home landing, land then reduce | q_own × min(k, chips) | local tokens × top-k (≤ all-gather when k < chips) |
| Home accumulator, add on arrival | q_own × H_r in fp32 | local tokens only |
| Relay slot rings (L1, credits) | fixed: ≈ bandwidth × round trip + 2 slots per link | nothing (skew becomes waiting) |
| Main's dispatch output | cf × G × q (+ padding), overflow dropped | chunk |

- **Why sizes depend on routing at all**: rows landing on a chip = popularity of its experts; the
  buffer is allocated before routing is known, so it is a bet against the worst chip. One row per
  (token, chip) caps it at T by construction (a token arrives at a chip at most once).
- **Ring depth**: bandwidth × (hop + relay handling + credit return) + 2 slots for store-and-forward.
  At 25 GB/s and a few µs that is ~4–16 slots of 14 KB per incoming link, ~0.5–2 MB per chip for all
  8 incoming links: one or two cores' L1, not a problem. The unknown is the round-trip time; measure
  it with a two-chip ping-pong and grow the slot count until throughput stops rising.
- **MoE peak buffers** (K3, 4 × 4, chunk 5,120): current main ~370 MB (~1.4 GB if column dispatch
  must never drop); main tuned ~240 MB; 2D layouts ~130–170 MB (expert input 37 MB, fp32 per-(token,
  chip) output 73 MB, home accumulator 5 MB, shared intermediates 12–58 MB, gathered x 18 MB). All
  ×6.4 at chunk 32,768. Allocated once, shared by all layers.
- **Expert output, settled design**: do **not** accumulate per (token, chip) on the expert side
  (read-modify-write, ordering between experts of a chip, couples experts and breaks streaming).
  Experts write **one bf16 row per slot**; combine sends each as soon as its expert finishes; the
  home chip adds into its fp32 accumulator. With streaming, the output buffer is a **ring of about
  two experts' outputs (≤ 2T rows)**; serial bring-up allocates T × min(k, experts per chip) or
  sizes from the counts.

---

## 8. Dispatch and combine design decisions

- **Combine options**: one row per slot (independent rows, streams perfectly), one row per (token,
  chip) (rows whose chip holds only one of the token's experts stream; the rest wait for the last
  local expert; with top-8 over 32 chips almost all rows are single-expert, while K3's 16 slots on
  4 × 4 land on ~10 chips, so it saves ~35% of rows there but couples more), reduction on the way home (mirror of drop-off, halves bytes but couples experts on different
  chips: a relay can only forward a merged row after all its branches arrive). Once the middle is
  DRAM-bound the reduction's saving is worthless. **Use one row per slot for combine.**
- **Combine, chosen (2026-10-16): level 1 streamed, the fabric routers relaying**
  (`tt-metal-dispatch` `sparse_combine_rf.py`, worklog "Level 1 combine with the fabric routers
  relaying"):
  - Each row goes to its token's home as one multi-hop 2D unicast. The 2D route tables are
    Y-then-X, the same order as the planned paths.
  - Rows land straight in a home accumulator's L1 ring, one ring of D slots per (source chip, link),
    so each ring has exactly one producer: the source's first-hop port.
  - Credits go back as one small packet per source chip: a snapshot of the cumulative consumed counts.
  - Home adds the rows into fp32 accumulators in L1; nothing touches DRAM.
  - On the LoudBox 2 × 4 it beats port-core relays (`sparse_combine_l1`) on 19 of 22 routings:
    uniform 77% of the link floor vs 72%, packed Zipf 1.2 90% vs 72%.
  - Needs a 7232 B fabric payload (more router slots; dispatch is unchanged at it).
  - Needs rows in expert order (as the flat op emits them). Token-major order head-of-line blocks the
    sender on one accumulator's ring.
  - L1 per accumulator is N × L × D × 7.2 KB: D 8 on 2 × 4, D 4 on 4 × 4.
  - Untested: the torus (router deadlock freedom).
- **When level 1 becomes link bound (2026-10-18, CPU model calibrated on 2 × 4;
  `scratch_combine_linkbound_model.py`, worklog "When the level-1 combine becomes link bound")**:
  the flat expert stays weight-stream bound up to 16K global tokens, the combine grows linearly.
  Uniform routing crosses (combine ≥ flat, today's 64–77% link efficiency) at ~15K (K3) / ~10.5K
  (V4.1) on 2 × 4 and 4 × 4, ~11K / ~7.5K on 8 × 4; exposure starts at ~0.8–0.9 of that. Packed
  skew: a constant 0.45–0.72 (never). Spread: plateaus at 0.8–0.97 (K3 8 × 4 at the edge from 8K).
  Levers: fp8 rows (~1.9×), level 2 (K3 2 × 4 2.3× fewer rows), link efficiency.
- **Dispatch**: drop-off out (big link saving, mild coupling), unicast per destination chip as the
  first implementation.
- **Send order for 160–320 owned tokens**: rounds by local expert index (with drop-off, a token goes
  in the round of its smallest index), furthest destinations first within a round (store-and-forward
  makes far rows slow; sending them first keeps every link busy), one queue per outgoing direction.
- **Relays and credits** (all local, no global orchestration):
  - each incoming link has a **private** slot ring; only that neighbour holds its credits, so the
    four writers into a chip never compete;
  - each outgoing link's sender picks: **forwarded rows first** (they hold slots; own rows hold
    nothing), own rows only while credits exceed a **reserve** (≥ 1 slot kept for forwarding, the
    ring deadlock rule), oldest round first;
  - with SP-then-TP routing an SP output has 1 forward source, a TP output at most 3; **a turn counts
    as an injection** into the TP ring;
  - backpressure is automatic (slots stay full, credits stop).
- The alternative used by v2 today is no runtime flow control: a precomputed plan plus enough DRAM
  to hold every relay. Credits replace that buffer with small rings and waiting.
- Smart scheduling is not needed for uniform routing (header bitmask + lookup tables per row). Under
  skew a plan can still be deterministic: every chip computes the same plan from the global counts.
- **The latent projection writes straight into the dispatch buffer** (user's design, 2026-10-15). A custom K3
  latent down-projection writes each owned token's latent into its own slot (origin x S + row) of the dispatch
  output buffer, so dispatch never copies own rows: port readers read own rows from those slots, no copy cores. Why
  it matters: on the LoudBox the own-row copy was a fixed ~100 us (K3 rows) / ~180 us (K2.7 rows) per call, the whole
  op when little leaves the chip. Requirements and consequences:
  - the projection emits row-major rows (one per token) into the interleaved DRAM buffer: untilize inside the op;
  - the experts read own tokens from the same buffer (own tokens never move);
  - overlap: with a per-batch "rows written" counter, dispatch can start sending the first rows while the projection
    still computes later ones (dispatch under the latent down-projection, next to planning under the shared expert).
- **Top-k emits a destination chip mask per row** (2026-10-15, from the 1D stress test). Turning K expert ids into
  the set of destination chips on the dispatch port cores cost ~90 us per call for K3 rows (16 ids per row) before
  the first send. The router's top-k already has the ids in registers: it should also write one 16-bit chip mask per
  token; dispatch then reads 2 bytes per row. Together with own rows written in place by the latent projection, a
  dispatch with no traffic costs ~35 us on the LoudBox (start skew + end markers) instead of ~100-185 us.
- **Never worse than the dense all-gather**: with packed hot experts a chip needs nearly every row and the
  stateless half split loaded the busiest link above the dense all-gather's (+10% time). The plan should fall back
  to dense-style balance in that case.
- **2D dispatch routing: a cheap plan is enough** (CPU model, 2026-10-15; worklog in tt-metal-dispatch
  `SPARSE_AG_WORKLOG.md`). Per row, evaluate the 4 stateless dimension-order trees (XY/YX x link) against the current
  link loads and take the least loaded. On a 4x4 torus this lands within 1-3 points of the lower bound for every
  routing tried and never meaningfully above the dense all-gather (<= 102%), also when 64 rows are decided in parallel
  against one stale load snapshot (a multi-core device planner). The stateless rule alone is fine for uniform routing
  (K3 76% of dense vs bound 70%) but loses badly under skew (Zipf 116-150%). Under heavy skew the bound itself equals
  the dense all-gather (a hot chip needs every row).
- **Dispatch plan creation hides under almost anything** (user's call, 2026-10-15). A plan (per-row
  route, per-port send lists, per-expert token lists) needs only the top-k ids, which are tiny and
  ready right after the router; it reads none of the block's big tensors and uses no links, just a
  few cores or idle RISCs. So it can run under whatever follows the router and doesn't need routing:
  the **shared expert** first (the natural partner, it never needs routing), the K3 latent
  down-projection, the first experts' weight prefetch.
  - Schedule: router -> small all-gather of the top-k ids -> [plan || shared expert || latent down]
    -> dispatch. Every chip computes the same global plan from the gathered ids, so no plan
    broadcast. The same pass gives each chip its per-expert token lists, which the source-routed
    ("plan-free") dispatch otherwise lacks.
  - Consequence: planning cost stops mattering as long as it fits under its partner. Then the
    better balance of a global plan (host-plan prototype ~10% faster than source-routed at 200G) is
    free, and source-routing is only the fallback when nothing is there to hide under.
  - Budget = the partner's duration. The single-RISC global greedy (milliseconds) does not fit;
    it needs splitting across cores or a cheaper balanced rule. It can also be pipelined: plan the
    first experts' rows, start dispatching them, plan the rest.
  - Catch: the shared expert was also meant to hide under dispatch and the experts. Whichever
    window it fills, the other needs a partner; splitting it (gather + gate/up under the plan,
    down + reduce-scatter under dispatch) is the obvious answer.

---

## 9. Overlap schedules (expert side)

- An expert needs **all** its rows before it starts (weight-stationary: each weight tile is read
  once for all rows).
1. **Expert-ordered dispatch**: everyone sends in the experts' run order; per-expert arrival
   counters (counts from the count exchange) start each expert as soon as its rows are in.
2. **Weights first**: stream the first few experts' weights into L1 from the start (K2.7: ~230 KB
   per core per expert, 3–4 experts fit), so the DRAM stream runs under dispatch. Rule of thumb:
   ring depth ≥ dispatch time ÷ one expert's weight time (≈4 for drop-off, ≈8 for per slot, which
   does not fit, so per slot needs ordering). With enough depth the send order stops mattering.
3. **Weights even earlier**: preload during attention (compute-bound), limited by L1.
- Combine streams per expert in all of them; only the last expert's share remains at the end.
- Floor without starting earlier: weights + last expert's compute + last combine piece.
- **Don't** pipeline by token micro-chunks (each chunk re-reads all weights in a DRAM-bound block);
  **don't** relay through DRAM while overlapped.
- With main's existing ops, the practical first step is **splitting experts into groups** (dispatch
  skips experts whose `expert_dispatch_table` entry is −1; combine and the expert op iterate by
  per-expert counts) and pipelining groups on two sub-devices.
- For K3 on 4 × 4 a serial 2D implementation already wins; overlap is worth ~13% and comes last.
- First experiment for any overlap: run the routed-expert op alone, then with a dummy background
  kernel imitating dispatch's DRAM and NoC traffic, to measure interference before designing.

---

## 10. Measured data so far (LoudBox, stock 400G firmware unless noted)

- GLM-5.3-Flash layer 4, chunk 5,120, 1 link, per-op device time:
  - all-gather MoE block 3.45 ms (flat expert 1.74, gather x 0.28, to_layout 0.22, send-back 0.29,
    local reduce 0.18 + 0.31, fabric reduce-scatter 0.26);
  - standard block with the unified expert 8.85 ms (dispatch v1 2.28, unified expert 3.20, combine v1
    1.88, post_combine_reduce 0.51, fp32 reduce-scatter 0.58);
  - standard block with the flat expert 7.38 ms (dispatch 2.27, flat 1.75, combine 1.86,
    post_combine_reduce 0.51, reduce-scatter 0.58), same accuracy as the all-gather block.
  Measured dispatch + combine cost more than the experts: implementations are far from the floor.
- Fabric: ~40.7 GB/s at 14.4 KB packets on one link, ~200 ns per packet below 10 KiB,
  `dispatch_fabric2d` ~17 GB/s per stream; combine v2 = 0.54–0.58× combine v1 per launch.
- Some LoudBoxes run 200G Ethernet firmware (see memory `loudbox-eth-200g-firmware.md`), matching
  the Galaxy's 25 GB/s.

---

## 11. Current plan and state

- Plan: `moe_token_split_tp_se_plan.md` (token-split MoE block with a TP-split shared expert, K3
  4 × 4). Phases: 0 harness/baseline, 1 router + count exchange, 2 shared expert, 3 2D dispatch
  (serial, DRAM relays, unicast first), 4 routed experts on the 2D input (one row per slot out),
  5 2D combine + home fp32 accumulator + latent up + sum, 6 overlap, 7 into the model.
- If the surrounding model stays H-split, the block's entry adapter is the shared expert's own
  all-gather (owners keep their quarter) and the exit adapter an all-to-all: that is the
  "Gather → 2D owners" layout.
- Bring-up tracker artifact holds op status, measured µs vs floor, overlap %, worklog and plan
  (schema in memory `k3-moe-bringup-tracker.md`). It currently shows **example data**; replace it
  with measurements and set `meta/block.example = false`.
- Open choices: K3 latent projections replicated (no collective, ~55 MB per chip per layer) or
  column-split + all-to-all; router weight bf16 or bf8; MLA input under a token split (outside MoE
  scope); whether the shared expert's reduce-scatter can run as a ring next to 2D dispatch.
- Not modelled anywhere: per-packet cost, NoC contention, launch/sync latency, GLM's DSA indexer,
  K3 KDA compute, real routing traces (main has an env-gated `_dump_routing` in `TtMoe` that could
  feed the dispatch model's skew cases).

---

## 12. When the combine catches the routed experts (chunk size x stage, 2026-10-19)

Model (scratchpad linkbound3.py of session 12382886; uniform routing): flat routed expert per chip = experts/chip x
max(weight read, compute), from the dims, calibrated on K3 / V4.1 single-chip runs (weights at 98% of 512 GB/s,
~407 TFLOP/s effective LoFi, row streaming 0.062 us x H / 3584); combine = the busiest (chip, direction, link) rows
from simulated routing on the stage's topology (2x4 mesh, 4x4 wrapped on the 4-wide axis, 8x4 torus), bf16 rows,
one per (token, expert) - no source pre-reduce - at 25 GB/s with 2 links per direction, divided by the measured
2x4 combine efficiency (K3 0.72, V4.1-Flash 0.64, GLM-5.3-Flash 0.66; others assumed 0.66). Check: GLM-5.3-Flash
S 640 on 2x4 predicts combine / flat 0.93, measured 0.98 (flat predicted 1378 vs 1305 us).
Global chunk where the combine catches the flat expert (ratio 1.0), where it starts to show (0.85, from the 2x4
fused overheads), and the catch point at the link floor (combine at 100% of link rate):
| model (E, top-k, H x I) | 2x4 | 4x4 | 8x4 |
|---|---|---|---|
| Kimi K2.7 (384, 8, 7168 x 2048) | 7.5K / 6.0K / 14.0K | 7.2K / 6.0K / 13.8K | 5.2K / 4.5K / 9.5K |
| GLM-5.3 (256, 8, 6144 x 2048) | 5.0K / 4.0K / 9.5K | 5.0K / 4.0K / 9.2K | **3.5K / 3.0K / 6.2K** |
| Kimi K3 (896, 16, latent 3584 x 3072) | 14.8K / 11.8K / never | 14.8K / 11.8K / never | 10.8K / 8.8K / 17.0K |
| MiniMax M3 (128, 4, 6144 x 3072) | 7.2K / 6.0K / never | 7.2K / 5.8K / never | 5.2K / 4.2K / 9.0K |
| DSv4.1-Flash (384, 6, 5120 x 2304) | 10.8K / 8.8K / 21.0K | 10.5K / 8.5K / 20.5K | 7.8K / 6.2K / 14.0K |
| GLM-5.3-Flash (288, 8, 4096 x 2048) | 5.5K / 4.5K / 10.5K | 5.5K / 4.5K / 10.2K | 4.0K / 3.2K / 7.0K |
- Why: the weight read is fixed per chunk (DRAM-bound at every chunk here except MiniMax M3, whose 4 experts / chip go
  compute-bound past ~16K - its ratio then plateaus at 1.37 / 1.79), while combine bytes grow with tokens x top-k x
  row width. 8x4 crosses ~30% earlier than 4x4: half the experts per chip, but the busiest link carries ~64% as many
  rows. Small-I models (GLM-5.3 and its Flash, K2.7) cross first: the combine-to-compute ratio ~ 1 / I.
- At chunk 5120 on 8x4 (160 tokens / chip) GLM-5.3 (1.31) and GLM-5.3-Flash (1.20) are already combine-bound with
  today's combine; K2.7 (0.96) and MiniMax M3 (0.99) are at the edge; K3 (0.54) and V4.1-Flash (0.72) are clear.
- Skewed routing: the hot chip's flat work grows with the chunk too (packed: constant ratio 0.4-0.7; spread plateaus
  0.8-0.97, 2026-10-18 linkbound2). Levers: link efficiency (0.64-0.72 today), source pre-reduce (K3 2.3x fewer rows,
  V4.1 1.36x), an fp8 / bfp8 wire (~1.9x fewer bytes) - each pushes the crossovers out by that factor.

---

## 13. Layouts re-costed with measured efficiencies (2026-10-20)

Model: `models/demos/deepseek_v3_d_p/moe_model/global_block.py` (uniform routing simulated per stage: 4x4 = TP wraps / SP line,
8x4 torus; LoudBox-measured efficiencies: combine 0.81 K3 / 0.75 GLM / 0.70 V4.1 of link rate, sparse dispatch 0.82,
TP AG 0.87 / RS 0.74, lin2d SE 338 GB/s, fused = max(flat, combine) + 5% of the smaller; small ops from the K3 trace).
Check: K3 block ~3.5 ms modelled under LoudBox conditions vs 3.32 ms traced. Tables on the Residual Layouts page,
section "With measured efficiencies".
- **1D (column dispatch + combine along SP, one TP reduce-scatter for routed + shared) never wins**: 3-39% behind the
  best 2D layout. Combine hot link is equal to 2D once ring ties are split (8x4) and the same middle-of-line link on
  4x4; 1D dispatch carries 1.4-2x the link load (one copy per TP column); AG + RS land on the critical path.
- **Shared expert: never replicated (user, 2026-10-20)**, always TP-split. Serially its all-gather + matmuls +
  reduce-scatter cost 17-41% of the block (K3 4x4 5120: 3852 vs 3163 us; LoudBox traced 4402 TP vs 3700 replicated),
  so it must overlap: all-gather + matmuls under router / routing gather / planner / dispatch, reduce-scatter on the TP
  links beside the combine. The K3 block's replicated lin2d SE (K3_SE default) is a bring-up shortcut to replace.
- **Gather -> 2D owners** costs about the same as token-split once overlapped: the residual layout can follow attention.
- **Combine at 16K is the bottleneck almost everywhere**: tie-splitting -11..-16% on the 8x4 torus where link-bound
  (0 on 4x4), fp8/bfp8 rows -34..-43% at 16K, source pre-reduce (K3 2.3x fewer rows).
- **Implemented (2026-10-20, GLM-5.3-Flash token-split block, LoudBox)**: the TP-split shared expert folded into the
  block with no collectives of its own - dispatch_prep ORs the group-mates into every token's mask (x lands in the
  dispatch buffer), the slice runs between the dispatch and the fused op (stock matmuls + one clamped-SiLU-GLU op,
  bf8 / HiFi2), the combine carries the partials home (static per-port rows, credits and expected counts on top of
  the device plan; the own block added by the accumulators). The fused op starts with its own cross-chip barrier,
  so the arena is zeroed after the shared expert and the shared expert has all of L1. Group: mesh rows on Galaxy
  (TP), SP pairs on the LoudBox (its TP line carries the combine). Router: one fused top-k writing the block's routing
  rows. L4 MoE block 4058 -> 2811 us (chunk 5120), 2423 -> 1791 (2048). Details: FLAT_COMBINE_OVERLAP_WORKLOG.md.
  Next: the fused shared expert and its overlap with the dispatch - ttnn/ttnn/bringup/combine_rf/FUSED_SHARED_EXPERT_PLAN.md.
