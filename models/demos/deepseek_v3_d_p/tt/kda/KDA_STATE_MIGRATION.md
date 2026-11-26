# KDA state migration contract (Kimi-K3)

Kimi-K3 keeps a per-user state in each of its 69 KDA layers. Migration exposes it through the same
KV chunk address table that carries the MLA kvpe cache, as two extra configs:

| config | name | holds | position axis | segment |
|---|---|---|---|---|
| 0 | `"00"` | MLA kvpe cache | tokens (32 per chunk) | `[32, 576]` bfp8, 19584 B |
| 1 | `"01"` | KDA recurrent state | synthetic, `chunk_n_tokens = 96` | `[128, 32]` FP32 V-band, 16384 B |
| 2 | `"02"` | KDA convolution tail | synthetic, `chunk_n_tokens = 64` | `[3, 64]` BF16 rectangle, 384 B |

Kimi-K3.1 publishes four: its Gated DSA layers add the indexer key cache as `"01"` (tokens, 32 per chunk,
`[32, 128]` bfp8, 4352 B, on the same layer rows and slabs as kvpe, whose chunk is scaled FP8, 32 x 704 B),
and the two KDA configs move to `"02"` and `"03"` unchanged. Consumers find a config by the adapter's
`cache_kind`, never by its number. `tests/test_kv_cache_table.py::test_kimi_k3_kv_cache_table` publishes
both tables from device caches through the runner's `setup_kv_chunk_table` and reads every chunk back.

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
interchangeable for the table. The native convolution tail uses the same `[SEG, (K-1) * 64]` rows.

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

Every SP row holding its column's full `H_local` heads (the slabs before SP sharding) was redundant: 7/8 of it on
Galaxy SP8xTP4, 3/4 on each 4x4 stage (SP4xTP4). Sharded, SP row `r` of column `c` stores only global heads `c * H_local + r * H_store + [0, H_store)`:

| layout | heads | `H_local` | `H_store` | per chip, layer and slot |
|---|---:|---:|---:|---|
| Galaxy 4x4 stage SP4xTP4 (K3, K3.1 default) | 96 | 24 | 6 | 384 KiB recurrent + 13824 B convolution |
| Galaxy SP8xTP4 (K3, K3.1 `*_8x4`) | 96 | 24 | 3 | 192 KiB recurrent + 6912 B convolution |
| LoudBox SP8xTP1 perf layout | 24 | 24 | 3 | same as Galaxy SP8xTP4 |
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

The stored convolution layer keeps the native branch-major order over its own heads: per branch, SP row `r`
stores the branch's channel slice `[r * W_b / SP, (r + 1) * W_b / SP)`.

Every K3 KDA layer (`ttKDA.forward` on `KdaStates.entry`) runs on the sharded slabs directly, as on the replicated
ones: the convolution and the SP relay (which also forms the final carry) address batch `slot * L + layer_position`
(slot read on device) inside the captured region. Only the chunk's chronologically first SP rank reads the old carry,
so inside the same programs the other ranks send it their heads of the entry over the fabric: the relay's senders
write each head's tile columns into the first rank's landing buffer, and the convolution's exchange core writes each
rank's channel slices of the old tail into the first rank's exchange buffer. The first rank's relay sends each head's
final carry to the rank storing it only, then a carry arrival to every rank on the same routes; every rank writes its
own channel slices of the new tail. Outputs and carries are bit-identical to the replicated slabs
(`tests/kda/components/test_relay_chunk_scan.py`, `test_qkv_convolution_exchange.py`, sharded-slab cases, on Galaxy
SP8xTP4 and on a 4x4 stage, SP on mesh axis 1).

Every chip's slab writes precede the layer's ack: a chip's relay program ends only after the carry arrivals of every
link, each sent after that link's writes to the chip on the same route, and its convolution program writes its own
tail pages; each device's ack record follows its own programs on its CQ, and the trace and eager paths synchronize
the whole mesh. This ordering is argued from the programs, not tested on device for the sharded slabs: an earlier
copy-based store had a D2H ack test with one chip held back by a spin kernel, which does not carry over, since a held
back chip now holds back its whole SP ring.

`models/demos/deepseek_v3_d_p/tt/kda/state_adapter.py` owns the geometry, the maps, the slabs and the host decoders
(`recurrent_segment_to_torch`, `convolution_segment_to_torch`, `assemble_*`). The state a reader sees is the one after
the last committed chunk, and it lands before any later MLA layer acks that chunk, and before the KDA layer's own
ack. Nothing clears a slot: on a request's head chunk (`actual_start == 0`, evaluated on device) the layer's ops start
from zeros instead of the slot's carry, so from the head until a layer's first store its regions still hold the
previous request's state, and the slabs start uninitialized.

## Reading it back

`models/demos/deepseek_v3_d_p/tt/runners/kda_state_readback.py` opens a published table and device
map, walks the KDA configs (the adapter's `cache_kind`: 1 and 2 on K3, 2 and 3 on K3.1; `--model`) by model layer,
reads every segment with `read_dram_umd` from the device group the table names for that segment (the one chip storing
it; a table with larger groups, published before the sharding, is refused), reassembles the
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
