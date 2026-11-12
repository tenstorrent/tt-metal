# KDA state migration contract (Kimi-K3)

Kimi-K3 keeps a per-user state in each of its 69 KDA layers. Migration exposes it through the same
KV chunk address table that carries the MLA kvpe cache, as two extra configs:

| config | name | holds | position axis | segment |
|---|---|---|---|---|
| 0 | `"00"` | MLA kvpe cache | tokens (32 per chunk) | `[32, 576]` bfp8, 19584 B |
| 1 | `"01"` | KDA recurrent state | synthetic, `chunk_n_tokens = 96` | `[128, 32]` FP32 V-band, 16384 B |
| 2 | `"02"` | KDA convolution tail | synthetic, `chunk_n_tokens = 64` | `[3, 64]` BF16 rectangle, 384 B |

Every config is published on the model's layer axis (rows 3, 7, 11, ... for kvpe; every other row
for KDA). A KDA config has no token axis, so it uses the synthetic axis it shares with decode
(tt-blaze `blaze/models/kimi_k3/cache_migration.py`, tt-blaze #6203): both states span one window of
`W = lcm(384, 576) * 32 = 36864` positions, global segment `i` sits at `i * 96` (recurrent) or `i * 64`
(convolution) inside it, and `max_sequence_length = W` (`kda_position` in `utils/kv_cache_utils.py`).
Both configs are tagged `Snapshot`: the KV Manager moves a layer's whole window whatever range
/migrate names, and refuses a migration unless both sides tag `Snapshot` with equal extents and
`chunk_n_tokens`. Decode keeps one KDA state per user and publishes the same single window
(tt-blaze #6203). The runtime declares the three caches as migration stages in
this order (`TtKimiK3Runtime.kv_migration_stages`), numbered in compacted slot space; the adapter
maps configs back to layers with `cache_kind` / `cache_layer_rows`.

## Native form and contract form

`ttKDA` reads and writes the native form and accepts nothing else:

* `recurrent`: `[1, H_local, D, D]` FP32 TILE, interleaved DRAM, stored `[k, v]`
* `convolution`: `[1, SEG, (K-1) * 64]` BF16 ROW_MAJOR, interleaved DRAM, channel-paged with one
  64-channel segment per page (`KDAProgramConfig.convolution_page_channels` = `KDA_CONVOLUTION_PAGE_CHANNELS`,
  `tt/kda/state_contract.py`): page `c`
  holds the `K-1` history rows of channels `[64c, 64c + 64)`, history row `r` at element `64r`; channels run
  `[q_local | k_local | v_local]`, `D` per head inside a branch, `SEG = 3 * H_local * D / 64`

with `H_local = 96 / TP`, `D = 128`, `K - 1 = 3`. Heads are TP-sharded; every SP row holds a replica.

The contract form is the carries' only store, owned by the engine (`KdaStates`, allocated by
`KimiK3Adapter.allocate_kv_cache`, one consolidated slab per state kind so a `KvCacheStage` can name
it by one base address):

* recurrent slab `[slots * L, H_local, D, D]` FP32 TILE, ND-sharded DRAM, shard `[1, 1, D, 32]`
* convolution slab `[slots * L, SEG, (K-1) * 64]` BF16 ROW_MAJOR, interleaved DRAM,
  `SEG = 3 * H_local * D / 64` (144 at TP4)

where `L` is the rank's KDA layer count and the leading index is `slot * L + layer_position`
(user-major, as the kvpe cache). The convolution slab's 384-byte page `row0[64c:64c+64] |
row1[...] | row2[...]` is byte-identical to the ND `[1, 1, (K-1) * 64]` shard, and an interleaved page `p`
sits at bank `p % N`, offset `(p // N) * 384`, where ND shard `s = p` would sit; the two forms are
interchangeable for the table. The interleaved form keeps each batch's rows contiguous logical pages,
so one whole-batch copy writes them. The native convolution tail uses the same `[SEG, (K-1) * 64]`
rows, so a slab batch is a byte copy of the native tensor and the per-layer convolution export and
import are one `indexed_entry_copy` each, with no relayout.

## Segment numbering

`g = tp_col * H_local + h_local` is the global head, `branch` is q/k/v = 0/1/2, `bands = D / 32`,
`halves = D / 64`. The global numbering, and with it every table key, depends only on the model:

```
recurrent    G = g * bands + band
convolution  G = (branch * 96 + g) * halves + half
```

The mesh stores each segment once: chip `(sp_row, tp_col)` holds `H_store = H_local / SP = 96 / (SP * TP)`
heads of its TP column; its stored head `j` is local head `h_local = sp_row * H_store + j`, and the table's
device group of a segment is that one chip.

```
recurrent    local shard s = (batch * H_store + j) * bands + band
convolution  local shard s = batch * (3 * H_store * halves) + (branch * H_store + j) * halves + half
address      bank = s % num_banks ; offset = base + (s // num_banks) * segment_bytes
```

The convolution order is the golden's `[all q | all k | all v]`. The recurrent golden is stored
`[heads, v, k]`, the transpose of the device's `[k, v]`.

## SP-sharded storage

Every SP row holding its column's full `H_local` heads (the slabs before SP sharding) was 7/8 redundant on
Galaxy SP8xTP4. Sharded, SP row `r` of column `c` stores only global heads `c * H_local + r * H_store + [0, H_store)`:

| layout | heads | `H_local` | `H_store` | per chip, layer and slot |
|---|---:|---:|---:|---|
| Galaxy SP8xTP4 (K3) | 96 | 24 | 3 | 192 KiB recurrent + 6912 B convolution |
| LoudBox SP8xTP1 perf layout | 24 | 24 | 3 | same as Galaxy |
| QuietBox SP4xTP1 perf layout | 24 | 24 | 6 | 384 KiB + 13824 B |
| QuietBox SP2xTP2 report layout | 48 | 24 | 12 | 768 KiB + 27648 B |

What changes and what does not, outside prefill:

| published item | sharded vs the earlier replicated slabs |
|---|---|
| config names, `Snapshot` tag, window `W`, `chunk_n_tokens` 96 / 64, segment bytes 16384 / 384, layer rows, global segment numbering, positions | unchanged |
| device group per segment | one device `(sp_row, tp_col)` instead of a TP column spanning every SP row |
| `noc_addr` (bank, offset) | the same formula over the smaller store (`H_store` instead of `H_local`) |

The KV Manager checks only the tag, `chunk_n_tokens`, the Snapshot extent and the piece bytes, and decode
agrees with prefill only on positions and the byte order inside a piece, so decode needs no change
(**to be confirmed with the KV Manager / tt-blaze owner**). Singleton groups are already published by
the TP-deduped MLA cache and the dflash caches. Migration reads then spread over every chip.

The stored convolution layer keeps the native branch-major order over its own heads, so the layer's
import gathers the SP rows' `[1, 3, H_store * halves, 192]` views on dim 2 into the native
`[1, 3, H_local * halves, 192]` view (a metadata view of the `[1, SEG, 192]` tail) with no permute
(`test_gather_over_sp_rebuilds_the_native_carry`). The export copies each chip's rows of every branch straight from
the `[1, SEG, 192]` tail: `indexed_entry_copy` with `source_part` (the chip's SP coordinate, read on device) and
`source_part_groups=3`, no view and no partition.

Per layer and chunk the K3 layer imports its slot's carry (`KdaStates.import_layer`: one `indexed_entry_copy` per
state kind of this chip's shard into a fixed staging buffer, zero-filled at a request head, then a native
`ttnn.all_gather` over SP into the 24-head scratch carry), runs its unchanged forward, and exports (one `indexed_entry_copy` per
kind, which reads the chip's heads of the SP-replicated outgoing carry, selected on device from the chip's SP
coordinate `KdaStates.sp_rank`, and writes them into the slab). Outputs and carries are
bit-identical to the replicated slabs (compared byte for byte on Galaxy SP8xTP4). Every chip's export precedes the layer's ack on all
three ack transports (each device's ack record follows its own export on its CQ, and the trace and eager paths
synchronize the whole mesh). For the D2H ack (metadata-only `D2HStreamService` + `LayerAckService`), eager and
traced, `tests/kda/test_kda_export_ack_order.py` tests this on device with one chip held back by a spin kernel ahead of
its export: while the other 31 chips' records are in, the slow chip still holds the previous chunk (both kinds) and no
ack has arrived; on the ack, all 32 chips hold the new chunk in both kinds. The trace-controller and eager host-callback
transports are covered by code reading only.

`models/demos/deepseek_v3_d_p/tt/kda/state_adapter.py` owns the geometry, the maps, the per-layer
export (one `indexed_entry_copy` each of the chip's recurrent tiles and convolution pages, which are the slab's
segments, selected from the native carry by the chip's SP coordinate), the import (the copies back into staging, then the SP all-gather) and the
host decoders (`recurrent_segment_to_torch`, `convolution_segment_to_torch`, `assemble_*`). Both take
the slot as a device tensor and resolve batch `slot * L + layer_position` on device, so a captured
load or store follows the slot in the traced metadata. The slabs are the KDA carries' only store:
every K3 KDA layer loads its slot's carry from them and stores the replacement back inside the
captured region, so the state a reader sees is the one after the last committed chunk, and it lands
before any later MLA layer acks that chunk, and before the KDA layer's own ack. Nothing clears a
slot: a request's head chunk (`actual_start == 0`, evaluated on device) loads zeros instead of the
slot's carry, so from the head until a layer's first store its regions still hold the previous
request's state, and the slabs start uninitialized.

## Reading it back

`models/demos/deepseek_v3_d_p/tt/runners/kda_state_readback.py` opens a published table and device
map, walks configs 1 and 2 by model layer, reads every segment with `read_dram_umd` from the device
group the table names for that segment (the one chip storing it; a table with larger groups, published
before the sharding, is refused), reassembles the
per-layer state and scores it against the head/tail golden
(`k3_vllm_code_debug_1M_head_tail/kda/kda_{recurrent,conv}_state_layer_N`, 1024-token period; the
state after `real_len` tokens is row `real_len / 1024 - 1`). A rank's host only reaches its own
devices, so run it once per host of a multi-rank run.

The CLI reads device-less over UMD (`read_dram_umd`), whose bare UMD cluster pins 1 GiB of host
memory for every visible chip: about 32 GiB on a Galaxy, with regular pages unless hugepages are
pre-allocated. On a Galaxy host with less memory free the CLI cannot run (on a Galaxy host with
~23 GiB available, the process was OOM-killed). The same holds for any other device-less UMD reader
in that process, such as the producer's `check_pcc`. A process that holds the devices reads the same
entries in-process instead: `score_layers(..., read_chunk=table_chunk_reader(table))`, which
`tests/kimi_k3/test_kda_migration_readback.py` runs on a real prefill (published table, golden scores,
and every layer bit-identical to the carry the layer imports).

## For the decode side

* A KDA config is a whole-state `Snapshot`: a KDA layer run migrates the one window `[0, W)`, not
  `[0, real_len)`, and the KV Manager moves that window whatever range the call names. Both sides must
  tag configs 1 and 2 `Snapshot` and span one window, so decode runs one KDA state per user
  (`TT_K3_KDA_STATE_SLOTS=1`). `KimiK3Adapter.layer_position_range` gives the in-repo migration driver
  `[0, W)`: one call per run of consecutive layers with equal ranges, and the byte verify walks each
  layer's own range. At `real_len == W` that range equals the MLA layers' `[0, real_len)`, so every
  layer goes in one call; the Snapshot configs still move whole.
* The state is a fold over the prefix, not position-addressed: decode must resume at exactly the
  prefilled length; replaying a chunk advances it twice.
* Only the final state after the last chunk is meaningful; per-chunk copies are valid but wasted.

## Measured cost (PR #56443, 8 Blackhole devices, real layer-1 weights, T = 5120)

Whole-tensor round trip native -> contract -> native, bit-identical on every layout:

| layout | layer (ms) | export + import (us) | of layer |
|---|---:|---:|---:|
| SP1xTP8 | 9.63 | 29.6 | 0.31% |
| SP2xTP4 | 9.56 | 36.6 | 0.38% |
| SP4xTP2 | 9.98 | 56.5 | 0.57% |

The row-major page redistribution in the copy op was parallelised across aligned page units for this
(`copy_default_row_major_program_factory.cpp`), which is what brought the convolution half down.
