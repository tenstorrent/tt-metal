# TtMoe (disaggregated prefill) -- block diagram

What this covers: `TtMoe.forward` in `tt_moe.py`, shown as a graph of ops, plus how every tensor
and weight is sharded across the mesh. MiMo-V2 (`mimo_v2_d_p/tt/ffn.py::TtMoE`) builds on the same
modules; its differences are listed at the end.

Notation:

```
  SP   = mesh axis 0 (rows)  = sequence parallel = dispatch axis   (dispatch_group_size = #rows)
  TP   = mesh axis 1 (cols)  = tensor parallel   = dispatch groups (num_dispatch_groups = #cols)
  S    = seq_len_per_chip          H  = emb_dim           I = hidden_dim (per expert)
  E    = num_routed_experts        K  = num_experts_per_tok
  EPC  = experts_per_chip = E / (#rows * #cols)
  BUF  = max_dispatch_buffer_token_size (flat, shared by the EPC local experts, capacity-factor sized)
```

The running example is **DeepSeek-V3 on a Galaxy (8 x 4)**:
H=7168, I=2048, E=256, K=8, 1 shared expert (I_sh=2048), so EPC = 256/32 = 8.
All shapes are **per device**.

---

## 1. Op graph

```
 x  (1, S, H/TP)   ROW_MAJOR, sharded: seq over SP rows, emb over TP cols
 |
 |=====================================================================================  GATE
 |
 +--> [ttnn.matmul]  x @ W_gate          W_gate (H/TP, E): emb rows split over TP
 |       |           partial logits (S, E)
 |       v
 |    [all_reduce_async]  cluster_axis=TP (sum)     --> full logits (S, E), replicated in the row
 |       |
 |       v
 |    [moe_grouped_topk]  sigmoid + e_score_bias, group-limited top-K, normalize, * route_scale
 |       |                (optional moe_padding_config -> padded rows get sentinel expert id E)
 |       |
 |       +--> scores  (S, K) bf16
 |       +--> indices (S, K) uint16
 |
 |=====================================================================================  ROUTING SETUP
 |       |
 |       v
 |    [masked_bincount]  indices, mask = dispatch_table[my column]
 |       |               hist (E,)  -- only experts of MY dispatch group are non-zero
 |       v
 |    [offset_cumsum]    internal all-gather of hist over SP  -> (#rows, E)
 |       |               exclusive prefix-sum over source rows + tile-aligned region starts
 |       |
 |       +--> expert_offsets        (1, E)    where THIS chip writes into each dest buffer
 |       +--> expert_token_counts   (1, E)    total tokens per expert (summed over the column)
 |       +--> expert_region_offsets (1, E)    start row of each expert's region in its dest BUF
 |
 |=====================================================================================  DISPATCH + SHARED (overlapped)
 |
 +--> [all_gather_async]  dim=-1, cluster_axis=TP   --> x_full (1, S, H), replicated in the row
         |
         |   (Kimi-K3 only: [to_latent] H -> routed_emb_dim, feeds dispatch; shared still reads x_full)
         |
         |   load_sub_device_manager: core row 0 = "dispatch SD", rows 1.. = "shared SD"
         |
         +-------------------------------------------+-------------------------------------+
         |                                           |                                     |
         v   dispatch SD (1 core row)                 v   shared SD (rest of the grid)      |
    [deepseek_prefill.dispatch]                  SHARED EXPERT (TP-sharded MLP)            |
      in:  x_full, scores, indices,                [matmul] x @ W_gate_sh (H, I_sh/TP)    |
           expert_offsets, dispatch_table          [matmul] x @ W_up_sh   (H, I_sh/TP)    |
      for each (token, k):                         [silu(gate) * up]  (or SiTU / clamped)  |
        dest = table[col][expert]                  [matmul] h @ W_down_sh (I_sh/TP, H)    |
        -> NOC (local) / fabric along SP (remote)      partial (1, S, H)                  |
      out: dispatched_buffer (1,1,BUF,H) RM        [reduce_scatter_minimal_async]          |
           metadata          (1,1,BUF,3)            dim=-1, TP axis (forced Linear        |
                (src_chip, token_idx, topk_idx)        when overlapped on a TP ring)       |
         |                                           |                                     |
         |                                           v                                     |
         |                                      shared_out (1, S, H/TP)  ------------------+--+
         |   clear_loaded_sub_device_manager                                                   |
         |                                                                                     |
 |=====================================================================================  ROUTED EXPERTS
         v                                                                                     |
    [unified_routed_expert_moe]   (or moe_fused_swiglu, or hybrid split by token count)         |
      in:  dispatched_buffer (BUF, H), expert_token_counts, expert_region_offsets              |
      per local expert e (EPC of them), only rows [region_e, region_e + count_e):               |
         silu(x @ W_gate[e]) * (x @ W_up[e])  @ W_down[e]                                      |
         W_gate/W_up[e]: (H, I)   W_down[e]: (I, H)   -- full experts, no TP split, no CCL     |
      out: expert_outputs (1,1,BUF,H)                                                          |
         |                                                                                     |
 |=====================================================================================  COMBINE
         v                                                                                     |
    [deepseek_prefill.combine]                                                                 |
      in:  expert_outputs, metadata, counts, region_offsets                                    |
      row -> (src_chip, token_idx, topk_idx): NOC / fabric back along SP                       |
      out: combined (1, 1, S, K, H) RM   -- slot k valid only if expert k lives in MY group    |
         |                                                                                     |
 |=====================================================================================  REDUCE
         v                                                                                     |
    [deepseek_prefill.post_combine_reduce]                                                     |
      sum_k scores[:, k] * combined[:, k, :]   (skips slots whose expert is not in my group)   |
      out: (1, 1, S, H)  = partial sum over MY group's experts only                            |
         |                                                                                     |
         v                                                                                     |
    [reduce_scatter]  dim=-1, cluster_axis=TP                                                  |
      sums the #cols group-partials (= all K experts) AND splits emb over TP                   |
      out: routed_out (1, 1, S, H/TP)                                                          |
         |                                                                                     |
         |   (Kimi-K3 only: [from_latent] distributed RMSNorm + up-proj back to H)             |
         v                                                                                     |
      [ttnn.add]  <----------------------------------------------------------------------------+
         |
         v
 y  (1, S, H/TP)   same sharding as x
```

Key trick: after the TP all-gather, every chip in a row holds the same S tokens. Each column is
its own dispatch group owning E/#cols experts, so chip (r, c) sends its tokens only to experts in
column c. Column c ends up with the part of each token's top-K sum that its own experts produce.
The final TP `reduce_scatter` adds those partials across columns and splits emb over TP in the
same op. No fabric traffic ever crosses columns during dispatch or combine.

---

## 2. Mesh layout and expert placement (Galaxy 8 x 4, DeepSeek-V3)

```
                     TP col 0            TP col 1            TP col 2            TP col 3
                  dispatch group 0    dispatch group 1    dispatch group 2    dispatch group 3
                  experts   0.. 63    experts  64..127    experts 128..191    experts 192..255
                 +------------------+------------------+------------------+------------------+
  SP row 0       | tok chunk 0      | tok chunk 0      | tok chunk 0      | tok chunk 0      |
  (logical 0)    | emb [   0,1792)  | emb [1792,3584)  | emb [3584,5376)  | emb [5376,7168)  |
                 | E   0..  7       | E  64.. 71       | E 128..135       | E 192..199       |
                 +------------------+------------------+------------------+------------------+
  SP row 1       | tok chunk 1      | tok chunk 1      | tok chunk 1      | tok chunk 1      |
                 | emb [   0,1792)  | emb [1792,3584)  | emb [3584,5376)  | emb [5376,7168)  |
                 | E   8.. 15       | E  72.. 79       | E 136..143       | E 200..207       |
                 +------------------+------------------+------------------+------------------+
      ...        |       ...        |       ...        |       ...        |       ...        |
                 +------------------+------------------+------------------+------------------+
  SP row 7       | tok chunk 7      | tok chunk 7      | tok chunk 7      | tok chunk 7      |
                 | emb [   0,1792)  | emb [1792,3584)  | emb [3584,5376)  | emb [5376,7168)  |
                 | E  56.. 63       | E 120..127       | E 184..191       | E 248..255       |
                 +------------------+------------------+------------------+------------------+
                        ^                                                            ^
                        |  dispatch / combine fabric traffic runs                    |
                        |  UP/DOWN a column only (cluster_axis = 0)                  |
                        v                                                            v

   <---------------  gate all-reduce, x all-gather, shared-expert RS, final RS  --------------->
                      run ACROSS a row (cluster_axis = 1)
```

Expert id for chip (r, c), local slot e:  `c * (E / #cols) + r * EPC + e`  (column-major; see
`ExpertMapping.create_global_expert_idx_table`). "tok chunk r" is sequential for single-shot
prefill and block-cyclic / rotated for chunked prefill (`is_balanced` = zigzag).

Dispatch table (one row per column, replicated down the column; -1 = not in my group, the last
column is the padding sentinel):

```
  group 0: [ 0 x8, 1 x8, ... 7 x8 | -1 x192             | -1 ]
  group 1: [ -1 x64 | 0 x8, 1 x8, ... 7 x8 | -1 x128    | -1 ]
  group 2: [ -1 x128 | 0 x8, ... 7 x8 | -1 x64          | -1 ]
  group 3: [ -1 x192 | 0 x8, ... 7 x8                   | -1 ]
```

---

## 3. What happens to one token (row 2, col 1)

```
  token t on SP row 2, top-8 = {5, 70, 77, 130, 131, 200, 64, 250}

  col 0 (group 0: 0..63)    : chip(2,0) sends t -> expert 5    on row 0          (1 copy)
  col 1 (group 1: 64..127)  : chip(2,1) sends t -> 64, 70 row 0 ; 77 row 1       (3 copies)
  col 2 (group 2: 128..191) : chip(2,2) sends t -> 130, 131 row 0                (2 copies)
  col 3 (group 3: 192..255) : chip(2,3) sends t -> 200 row 1 ; 250 row 7         (2 copies)
                                                                           total = K = 8

  after combine + post_combine_reduce, chip(2,c) holds  sum_{k in group c} w_k * FFN_k(t)  (full H)
  reduce_scatter over the row:    chip(2,c) <- sum over c'  [ emb slice c ]
                                  = sum_{all 8 k} w_k * FFN_k(t)    restricted to emb slice c
```

---

## 4. Per-device tensor / weight sharding summary

```
 tensor / weight              per-device shape            SP (rows)        TP (cols)        where
 ---------------------------  --------------------------  ---------------  ---------------  ------------
 x (in)                       (1, S, H/TP)                seq chunk        emb slice        DRAM/L1
 W_gate (router)              (H/TP, E)                   replicated       emb rows split   DRAM
 e_score_correction_bias      (E,)                        replicated       replicated
 logits / scores / indices    (S, E) / (S, K) / (S, K)    seq chunk        replicated       L1 -> DRAM
 dispatch_table               (E+1,)                      replicated       per group        DRAM
 expert_offsets               (1, E)                      per source row   per group        DRAM
 expert_token_counts/regions  (1, E)                      same in column   per group        DRAM
 x_full (after AG)            (1, S, H)                   seq chunk        replicated
 dispatched_buffer / metadata (1, 1, BUF, H) / (.., 3)    per chip         per chip         DRAM
 W_gate/up routed             EPC x (H, I)                per chip         per chip         DRAM (bf4/bf8)
 W_down routed                EPC x (I, H)                per chip         per chip         DRAM
 expert_outputs               (1, 1, BUF, H)              per chip         per chip
 combined                     (1, 1, S, K, H)             seq chunk        partial (group)  DRAM
 W_gate/up shared             (H, I_sh/TP)                replicated       hidden split     DRAM
 W_down shared                (I_sh/TP, H)                replicated       hidden split     DRAM
 shared_out / routed_out / y  (1, S, H/TP)                seq chunk        emb slice
```

---

## 5. CCL inventory per MoE layer

```
  op                              axis      what moves                              topology
  ------------------------------  --------  --------------------------------------  -----------------
  gate all_reduce_async           TP        logits (S, E)                            col_topology
  offset_cumsum (internal AG)     SP        per-row histograms (E,)                  row
  x all_gather_async              TP        x -> full H                              col_topology
  dispatch                        SP        up to K * S * H per chip (only to experts row_topology
                                            in my group; local ones stay on NOC)
  shared-expert reduce_scatter    TP        (S, H) partial -> (S, H/TP)              Linear if overlapped
  combine                         SP        expert rows back to their source chip    row_topology
  final reduce_scatter            TP        (S, H) group-partial -> (S, H/TP)        col_topology
```

---

## 6. MiMo-V2 d_p differences (`mimo_v2_d_p/tt/ffn.py::TtMoE`)

```
  - no shared expert, no sub-device overlap: gate -> routing_setup -> dispatch -> routed expert
    -> combine -> reduce, run in sequence
  - x comes in already full-H (replicated across TP), so there is no pre-dispatch all-gather.
    After the reduce_scatter, an all_gather over TP restores full H.
  - own TtGate (fp32 gate bias), dispatch buffer tilized to bf8 before the experts
  - 256 experts: 2x2 -> EPC = 64, capacity factor 4 ; Galaxy 8x4 -> EPC = 8, capacity factor 2
```
