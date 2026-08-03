# Design: TP-sharding the GLM-5.2 KV cache (Tier 1 — storage dedup)

Status: implementation plan, decided. Scope: **GLM-5.2 sparse (DSA) path only** — dense models are a
later generalization. Touches shared MLA/DSA infra (`models/demos/deepseek_v3_d_p` + the
`update_padded_kv_cache` device op), so any C++ change needs regression coverage across the family.

All file:line citations below were re-verified against `main` on 2026-07-23 (this code churns; re-check
before editing — see [[feedback_ci_bisect_verification_rigor]]).

---

## 1. Problem & payoff

On the production Galaxy mesh (`SP=8, TP=4`, `sp_axis=0`, `tp_axis=1`), GLM-5.2's KV cache is sharded on
sequence across the **SP axis only** and **replicated 4× across TP** — every TP-column device in an SP
row stores an identical copy of that row's tokens. Two caches, both from `init_kvpe_cache`
(`utils/kv_cache_utils.py:306-391`), allocated by `GLM52Adapter.allocate_kv_cache` (`glm_5_2.py:56-98`):
- MLA latent cache (`kvpe`, width 576 = `kv_lora_rank(512) + qk_rope_head_dim(64)`), bf16/ROW_MAJOR.
- DSA lightning-indexer key cache (`index`, width `index_head_dim=128`), bfp8/TILE.

**Payoff (from [[project_glm52_galaxy_capacity]]):** weights are a fixed 411 GiB; KV is 362 GiB/user @ 1M
(= 90.5 GiB logical × 4 TP-replication). Deduplicating to the logical 90.5 GiB changes concurrent 1M-context
users per Galaxy from **1 → ~6** (`(1020−411)/90.5 ≈ 6.7`); per-device KV drops **11.3 → 2.83 GiB**. This
is the accuracy-safe lever (bit-identical, pure storage dedup). It is **orthogonal and stackable** with the
fp8_e4m3 KVPE lever (~halves to 44 GiB/user, but trades precision) — do both for ~8×.

Caveat: real concurrency is also bounded by the transient per-device prefix-gather ceiling (~1.17 GiB/dev
@ 1M, grows with context×batch), so "~6 users" is a storage upper bound, not a delivered number — measure.

## 2. Why it's replicated today (root cause)

`kv_a_proj_with_mqa` is column-parallel across TP (`_kv_stem`, `mla.py:908-913`). Each TP device computes a
partial sum over the same token range, then a TP all-gather + `fast_reduce_nc` (`mla.py:916-929`; duplicated
in `indexer.write_k` via `_tp_rs_ag`, `indexer.py:461`) completes the sum — so every TP device ends up with
the **identical complete** value, which is what gets written. The cache topology is then explicitly forced
to mimic `ReplicateTensorToMesh` (`kv_cache_utils.py:379-389`). Reads gather over `sp_axis` only
(`_gather_kvpe_prefix:1523`, `_gather_index_kbuf:440`); writes address `cluster_axis=self.sp_axis` only.

**Tier 1 keeps this compute unchanged.** The stem still produces the full TP-replicated value; we only
change *storage* so each TP device persists a distinct 1/tp slice. (Removing the redundant compute is
Tier 2, §7 — deferred; the matmul is tiny, so the win is marginal.)

## 3. The design: linearized `sp·tp` block-cyclic, split inside the write op

**Decision 1 — the write op writes only its own TP part; no model-side reshard.** The op receives the full
TP-replicated value (`[1,1,seq_len_local,D]`, identical on all TP chips) and internally reads only its own
`1/tp` window and writes that. The sharding lives entirely in `update_padded_kv_cache`; the model just
passes `tp_axis`. (Rejected alternative: `mesh_partition` in the model before the write — a free local slice,
but puts sharding logic in the model and allocates an extra tensor. The op-owns-it form is cleaner.)

**Decision 2 — linearize the two axes into one effective block-cyclic axis of size `sp·tp`.** Order chips as
`linear = sp_coord·tp + tp_coord`. Within a prefill chunk (`sp·seq_len_local` global tokens), SP chip `j`'s
contiguous `seq_len_local`-token block is sub-split into `tp` contiguous `seq_len_local/tp` sub-blocks,
sub-block `k` → chip `(j,k)` = linear `j·tp+k`. This is **exactly single-axis block-cyclic over `sp·tp`
chips** with `chunk_local' = chunk_local/tp`. Consequence: the block-cyclic **remap math is unchanged** — the
read ops (`sparse_sdpa`, `indexer_score_dsa`) just get fed `sp·tp` and `chunk_local/tp` instead of `sp` and
`chunk_local`. **No new 2-axis modulus.** (This is why the C++ surface is ~1 op, not 3.)

**Decision 3 — read reconstructs with two all-gathers, TP-inner then SP-outer.** That gather order
reproduces linear order `j·tp+k`, after which the existing SP-style remap decodes it. A single combined
2D-mesh gather is not available (ROW_MAJOR over a partial cluster-axis line routes through
`composite_all_gather`→`all_broadcast` and deadlocks the fabric), so 2 sequential AGs is the floor — see §5.

**One C++ subtlety (the input is TP-replicated but the cache is TP-sharded):** the write op can't collapse to a
single factor. `chunk_global = sp · input_rows` uses **physical SP** (TP chips share the same input rows, they
don't add tokens); `global_cache_capacity` and the destination address use **`sp·tp`**. The op needs both
`sp`/`sp_coord` and `tp`/`tp_coord`, plus the input-window offset `tp_coord·(chunk_local/tp)`.

## 4. Change-site map (Tier 1)

### 4.1 Device-op C++ — the one real change
`update_padded_kv_cache` (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/`):
- `operation_attributes_t` (`device/..._device_operation.hpp:33`): add a `tp_axis` field alongside the
  existing single `cluster_axis`.
- device op (`..._device_operation.cpp:63-65,233-234`): keep `chunk_global` on physical `sp`; use `sp·tp` for
  `global_cache_capacity` and the destination linear coord `sp_coord·tp + tp_coord` (extend
  `get_linearized_index_from_physical_coord` to combine two axes, or compose the two per-axis indices).
- writer kernel (`kernels/dataflow/writer_update_padded_kv_cache.cpp:50-59`): read only this chip's input
  window `[tp_coord·(chunk_local/tp), +chunk_local/tp)`, write `chunk_local/tp` rows at the linearized slot.
  The `% sp_factor` modular math is otherwise unchanged, driven by the `sp·tp` linear factor.

`sparse_sdpa` / `indexer_score_dsa` / `block_cyclic_remap.hpp` / `block_cyclic_layout.hpp`: **no math change** —
resolve the block-cyclic `sp` as `sp·tp` and `chunk_local` as `chunk_local/tp` (§3, Decision 2). Confirm with
the op owner that this holds and that `get_linearized_index_from_physical_coord` can combine axes.

### 4.2 Host / model Python
1. **Allocation** — `init_kvpe_cache` (`kv_cache_utils.py:306-391`): `seq_len_local = seq_len //
   (mesh_shape[sp_axis] * mesh_shape[tp_axis])` (`:338`); replace the forced-replicate topology (`:379-389`)
   with a genuine SP×TP shard topology. Ripples to ~15 test callers and the two
   `cache_shard_dims[sp_axis]=2 … TP-replicate` lines in `test_prefill_transformer_chunked.py:291,345`.
2. **Write sites** — add `tp_axis=self.tp_axis` (input unchanged, stays TP-replicated):
   - `mla.py:1246-1254` (`_sparse_chunked_attn`), `indexer.py:479-486` (`write_k`).
   - Excluded (dense / KV-only, not on GLM's path): `mla.py:746-754`, `mla.py:1336-1344` — add an assert so a
     flipped `PREFILL_KV_ONLY_LAST_LAYER` fails loud instead of hitting an unmigrated path.
3. **Read gathers** — add a TP leg (TP-inner, then the existing SP gather):
   - `_gather_kvpe_prefix` (`mla.py:1472-1526`): new `_all_gather(dim=2, cluster_axis=tp_axis)` before `:1523`
     (the helper `mla.py:1355-1372` is already axis-parameterized); update the slab-trim at `:1509` to
     `chunk_local/tp`, `sp·tp`. Apply the ROW_MAJOR→TILE round-trip on the TP leg **if** it doesn't take the
     native path (open question, §6).
   - `_gather_index_kbuf` (`indexer.py:412-443`): add `_tp_all_gather` (exists, `:662`) before `_sp_all_gather`
     at `:440`.
4. **Read-op scalars** — pass `sp·tp` / `chunk_local/tp` (or `seq_len/tp`):
   - `sparse_sdpa` call (`mla.py:1446-1456`), `indexer_score_dsa` call (`indexer.py:618-632`).
5. **Alignment** — generalize `chunk_size_global % (tile_size*sp_factor)==0` (`mla.py:738-741`) to include tp;
   add a short-sequence fallback to replication when there are `< tp` slabs. Production config is clean:
   `5120/(8·4)=160` tokens/device/chunk, `160 % 32 == 0`.

### 4.3 Disaggregation KV chunk address table (block-cyclic, kimi variant)
GLM builds the **kimi block-cyclic** table (runner `kv_chunk_table.py:114-124`; DSA uses the merged variant
`_build_and_serialize_merged_kv_chunk_table:137-174` — config 0 = KVPE, config 1 = index, sharing one
device-group side table). This is the *same* layout the writer produces, so extending it to `sp·tp` is a
natural change, not a rework. **Make it cover all 32 devices via singleton per-`(row,col)` groups.** The
`KvChunkAddressTable` API already supports arbitrary per-device groups — no C++ table change.

Edits to `populate_kv_chunk_address_table_kimi` (`kv_cache_utils.py:204-303`):

```python
# (1) device groups: per-(row,col) singleton, not one replicated group per row  (:245-253)
device_group_idx = {}                                   # (row, col) -> group_idx
for row in range(rank_row_start, rank_row_end):
    for col in range(mesh_shape[1]):
        fid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))
        all_fabric_node_ids.append(fid)
        device_group_idx[(row, col)] = lookup_table.add_device_group([fid])   # singleton

# (2) TP sub-divisor + asserts  (:261)
tokens_per_chunk_local = PREFILL_CHUNK_OUTPUT_TOKENS // mesh_shape[sp_axis]    # 640
tokens_per_chunk_tp    = tokens_per_chunk_local // mesh_shape[tp_axis]         # 160
assert tokens_per_chunk_local % mesh_shape[tp_axis] == 0                       # 640 % 4 == 0
assert tokens_per_chunk_tp % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0           # 160 % 32 == 0

# (3) col loop + col offset + 160-wide ranges  (:282-301)
for global_row in range(rank_row_start, rank_row_end):
    for col in range(mesh_shape[tp_axis]):                                     # NEW
        group_idx = device_group_idx[(global_row, col)]
        curr_bank_id = 0
        curr_bank_offset = 0                                                   # each device: own 160/chunk
        for slot in range(num_users):
            for layer in range(num_layers):
                for seq_chunk in range(num_chunks_per_seq_len):
                    start = (seq_chunk * PREFILL_CHUNK_OUTPUT_TOKENS
                             + global_row * tokens_per_chunk_local
                             + col * tokens_per_chunk_tp)                       # NEW: + col*160
                    end = start + tokens_per_chunk_tp                           # 160, not 640
                    for position in range(start, end, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                        location = ttnn.experimental.disaggregation.KvCacheLocation()
                        location.noc_addr = (curr_bank_id << 32) | (dram_bank_base_addr + curr_bank_offset)
                        location.size_bytes = chunk_size_bytes
                        location.device_group_index = group_idx
                        lookup_table.set(layer, position, slot, location, config_id)
                        curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                        if curr_bank_id == 0:
                            curr_bank_offset += chunk_size_bytes
```
Unchanged: `dram_bank_base_addr` (`:279`) stays a single value — the mesh allocator places every device's
shard at the same address; each device just holds 160 tokens/chunk, so `curr_bank_offset` grows 4× slower.
The bank round-robin restarts per `(row,col)`, matching `init_kvpe_cache`'s per-device ND shard. Both merged
configs share the same 32 groups.

### 4.4 Explicitly unchanged
KV stem all-gather+reduce (`mla.py:916-929`), indexer K-stem all-reduce (`indexer.py:461`), all dense paths
(`ring_mla`, `ring_joint_scaled_dot_product_attention`, `_dense_*`), `fill_cache_for_user_`, GLM-5.1.

## 5. Read-side cost: yes, we add one all-gather (bump to an existing barrier)

Reconstructing across a 2D mesh needs **2 sequential AGs** (TP-inner, SP-outer) — inherent, not avoidable
today. But today's read *already* does 1 AG (SP); TP-sharding adds the **second (TP) leg**, which is the
smaller one — it moves only `(tp−1)/(sp·tp)` of the prefix. Per-chip gather-receive goes `7/8 → 31/32`
(**~+9-11%** at 8×4), plus one gather's fixed/barrier overhead, plus a possible TILE round-trip (§6).

- **Indexer:** PR [#49899](https://github.com/tenstorrent/tt-metal/pull/49899) adds `ring_indexer_score_dsa`,
  fusing the SP gather into scoring (feasible because scoring has *no key-axis reduction*) — 1.47× on GLM-5.1,
  hides 86% of gather latency. The added TP AG rides in front (small; check if it folds into the fused op's
  first band). So on the indexer path TP-sharding is ~free.
- **Attention (`sparse_sdpa`):** *not* ring-fused today — it consumes a pre-materialized prefix
  (`_gather_kvpe_prefix` is a blocking barrier). So both AGs block. Fusing attention is the *harder*
  flash/online-softmax form (attention *has* a key-axis reduction), like the dense `ring_mla` already does —
  a separate, bigger op effort, flagged as a perf TODO at `indexer.py:425-430`. **Follow-on, gated on
  measurement** at GLM's serving context lengths: at long context sparse compute dominates and +11% gather is
  noise; at short context the barrier is more visible.

## 6. Open questions to close before/while implementing
1. Confirm with the `update_padded_kv_cache` owner that (a) the linearized `sp·tp` scheme is the right shape,
   and (b) `get_linearized_index_from_physical_coord` can combine two axes.
2. Does the ROW_MAJOR **KVPE TP-axis** gather take the native all-gather path (like the SP gather does in
   production today) or does it need the TILE round-trip (the composite→`all_broadcast` deadlock that bit the
   indexer's uint32 gather at `indexer.py:654-659`)? Determines whether the TP leg pays a retile.
3. KVPE-first vs both caches at once — recommend both (they tile identically; the merged table already covers
   both), but KVPE is the big win (576 vs 128 wide).

## 7. Tier 2 (deferred): remove the redundant compute
Replicate `kv_a_proj_with_mqa` across TP and feed each TP device a distinct 1/tp sequence slice, so the stem
computes only its own 1/tp directly — no TP all-gather+reduce for KV. The matmul is *tiny*
(`hidden × 576`), so the FLOP saving is negligible; the only plausible value is cutting the stem's
all-gather+reduce **latency**, and it requires different sequence-sharding for KV/indexer vs Q/MLP within one
layer (a real structural change). Also note the read-side TP gather is still required (both tiers). **Not
recommended** unless profiling shows the stem CCL is a bottleneck.

## 8. Sequencing & validation
1. Settle §6.1 with the op owner.
2. Land the `update_padded_kv_cache` writer C++ change + op-unit test in
   `tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py` — the **load-bearing test**: write a
   known ramp through an SP×TP case and assert TP-inner/SP-outer gather + `sp·tp` remap round-trips to natural
   order. This pins the whole scheme before any model wiring.
3. `init_kvpe_cache` topology + the two write sites.
4. The two read gathers + read-op scalars.
5. Regenerate the block-cyclic table (§4.3); extend `test_glm_kv_cache_table` / the merged readback to an
   SP×TP case — proves table + writer + allocation agree byte-for-byte.
6. End-to-end PCC on `tests/sparse_mla/` + the `vllm-glm52-indexer-kcache-55k` trace (`glm_5_2.py:120`).
   Regression-gate DeepSeek V3.2 / Kimi K2.6 / GLM-5.1 since they share `update_padded_kv_cache`.
