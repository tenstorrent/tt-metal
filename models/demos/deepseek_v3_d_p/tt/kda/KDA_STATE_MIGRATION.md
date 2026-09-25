# KDA state migration contract (Kimi-K3)

Kimi-K3 keeps a per-user state in each of its 69 KDA layers. Migration exposes it through the same
KV chunk address table that carries the MLA kvpe cache, as two extra configs:

| config | name | holds | position axis | segment |
|---|---|---|---|---|
| 0 | `"0"` | MLA kvpe cache | tokens (32 per chunk) | `[32, 576]` bfp8, 19584 B |
| 1 | `"1"` | KDA recurrent state | synthetic, `chunk_n_tokens = 96` | `[128, 32]` FP32 V-band, 16384 B |
| 2 | `"2"` | KDA convolution tail | synthetic, `chunk_n_tokens = 64` | `[3, 64]` BF16 rectangle, 384 B |

Every config is published on the model's layer axis (rows 3, 7, 11, ... for kvpe; every other row
for KDA). A KDA config has no token axis, so it uses the synthetic axis of the K3 disaggregation
contract (`k3_disagg_contract.md`, tt-blaze #3634): one version of both states spans a window of
`W = lcm(384, 576) * 32 = 36864` positions, global segment `i` sits at `i * 96` (recurrent) or `i * 64`
(convolution) inside it, and the table repeats each segment in all 8 windows (`v * W + i * stride`,
`max_sequence_length = 8 * W`) because decode keeps 8 versions and every window aliases the one
prefill state (`kda_position` in `utils/kv_cache_utils.py`). The KV Manager needs this: it requires
equal `chunk_n_tokens` on both sides and walks one position range for every config of a layer, so
both states must fill the same window. The runtime declares the three caches as migration stages in
this order (`TtKimiK3Runtime.kv_migration_stages`), numbered in compacted slot space; the adapter
maps configs back to layers with `cache_kind` / `cache_layer_rows`.

## Native form and contract form

`ttKDA` reads and writes the native form and accepts nothing else:

* `recurrent`: `[1, H_local, D, D]` FP32 TILE, interleaved DRAM, stored `[k, v]`
* `convolution`: `[1, K-1, 3 * H_local * D]` BF16 ROW_MAJOR, interleaved DRAM, channels
  `[q_local | k_local | v_local]`, `D` channels per head inside a branch

with `H_local = 96 / TP`, `D = 128`, `K - 1 = 3`. Heads are TP-sharded; every SP row holds a replica.

The contract form is a second copy the engine owns (`KdaStates`, allocated by
`KimiK3Adapter.allocate_kv_cache`, one consolidated slab per state kind so a `KvCacheStage` can name
it by one base address):

* recurrent slab `[slots * L, H_local, D, D]` FP32 TILE, ND-sharded DRAM, shard `[1, 1, D, 32]`
* convolution slab `[slots * L, SEG, (K-1) * 64]` BF16 ROW_MAJOR, interleaved DRAM,
  `SEG = 3 * H_local * D / 64` (144 at TP4)

where `L` is the rank's KDA layer count and the leading index is `slot * L + layer_position`
(user-major, as the kvpe cache). The convolution slab's 384-byte page `row0[64c:64c+64] |
row1[...] | row2[...]` is byte-identical to the ND `[1, K-1, 64]` shard, and an interleaved page `p`
sits at bank `p % N`, offset `(p // N) * 384`, where ND shard `s = p` would sit; the two forms are
interchangeable for the table. The interleaved form is used because `slice_write` can write whole
rows into it per layer, while no op writes a row-major sub-region into an ND-sharded tensor.

## Segment numbering

`g = tp_col * H_local + h_local` is the global head, `branch` is q/k/v = 0/1/2, `bands = D / 32`,
`halves = D / 64`:

```
recurrent    G = g * bands + band
             local shard s = (batch * H_local + h_local) * bands + band
convolution  G = (branch * 96 + g) * halves + half
             local shard s = batch * (3 * H_local * halves) + (branch * H_local + h_local) * halves + half
address      bank = s % num_banks ; offset = base + (s // num_banks) * segment_bytes
replicas     one device group per TP column g // H_local, spanning every SP row
```

The convolution order is the golden's `[all q | all k | all v]`. The recurrent golden is stored
`[heads, v, k]`, the transpose of the device's `[k, v]`.

`models/demos/deepseek_v3_d_p/tt/kda/state_adapter.py` owns the geometry, the maps, the per-layer
export (`fill_cache_for_user_` for recurrent, reshape/permute/`slice_write` for convolution), the
import (`slice` back to a native carry) and the host decoders (`recurrent_segment_to_torch`,
`convolution_segment_to_torch`, `assemble_*`). `KdaStateCache.commit` exports every committed layer
inside the captured region, so the state a reader sees is the one after the last committed chunk, and
it lands before any later MLA layer acks that chunk. `reset(slot)` zeroes the slot's regions too.

## Reading it back

`models/demos/deepseek_v3_d_p/tt/runners/kda_state_readback.py` opens a published table and device
map, walks configs 1 and 2 by model layer, reads every segment with `read_dram_umd`, reassembles the
per-layer state and scores it against the head/tail golden
(`k3_vllm_code_debug_1M_head_tail/kda/kda_{recurrent,conv}_state_layer_N`, 1024-token period; the
state after `real_len` tokens is row `real_len / 1024 - 1`). A rank's host only reaches its own
devices, so run it once per host of a multi-rank run.

## For the decode side

* A KDA config is whole-state: a KDA layer run migrates one whole window, not `[0, real_len)`. With
  `v = (real_len - 1) % 8`, the version decode reads next, the call is `[v * W, (v + 1) * W)` on both
  sides (contract section 5), or source `[0, W)` to destination `[v * W, (v + 1) * W)` where the
  caller can pass separate ranges. MLA and KDA layers need separate calls.
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
