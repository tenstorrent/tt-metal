# DeepSeek-V4.1-Flash P→D migration configs (prefill side, v3, 2026-11-07)

What the prefill model serves to decode at commit `902322e6f77` (task P.42), with the default settings. Every number
below was generated from the code (`tt/runners/kv_contract.py:served_configs` / `address_table`, `tt/attention.py:
snapshot_cap`, `tt/settings.py`), host only, against the real checkpoint config. The command is at the end.

v3 replaces the v2 draft (`/scratch/dnijemcevic/rebase/dgen_migration_table_draft.md`, prefill tree 285af97abd1).
Like v2 it uses the config names of our serving contract, which Brian Liu's decode-side review also uses (artifact
https://claude.ai/artifact/HG8kwWMJedV7TZsJtPMjgk, #blaze-deepseek-v4, 2026-11-06). What changed since v2: P.42 made
the re-index aliases and the 128-row window snapshot real (both were "to do" in v2).

## Terms

| Term | Meaning |
|---|---|
| P→D, PD | Prefill → decode: prefill runs the prompt on one machine, then the caches move ("migrate") to the decode machine |
| Config | One table of the tt-d-gen KV address table (`KvChunkAddressTableConfig`): one cache kind, a set of layers |
| Record | One table entry: one layer, 32 tokens (`chunk_n_tokens` = 32 for every config), contiguous in one DRAM bank, offset a multiple of 64 B |
| Ratio r | The compressed caches keep one row per r tokens, so a record holds 32 / r rows |
| History tensor | `tensor_type` KV: a record per 32 positions, decode pulls the records it lacks |
| Snapshot tensor | `tensor_type` Snapshot (tt-d-gen#1955): a fixed-size state moved whole, once, with the last chunk |
| `KV_BLOCK_SIZE` | The serving engine's `kv_block_size`: 64 in every shipped tt-d-gen worker config (32 is legal) |
| cap | The last migrated position, decode's `reusable_prefix_cap`: `cap = (prompt_len - 1) // KV_BLOCK_SIZE * KV_BLOCK_SIZE`. Under PD tt-d-gen hands prefill `actual_end = prompt_len - 1`, so prefill computes it as `actual_end // KV_BLOCK_SIZE * KV_BLOCK_SIZE` (P.49, DS41-219). Decode receives rows [0, cap) and replays [cap, prompt_len) itself; since P.49 prefill computes nothing at or past cap. Examples (prompt_len → cap): 100 → 64, 128 → 64, 129 → 128, 2,945 → 2,944, 56,320 → 56,256, 1,048,576 → 1,048,512, 64 → 0 |
| Alias entry | A table entry that points at another layer's record (same address, size, device group, slot and position). No bytes of its own |
| Device group | The chips that hold the same bytes of a record. Decode may read any one of them |
| TP-dedup / TP-replicated | 8x4 mesh = 8 SP rows (sequence split) x 4 TP columns. TP-dedup: each record on one chip. TP-replicated: on the 4 chips of a mesh row |
| Blaze fp8 row | tt-blaze's indexer key-cache row, 192 B (below) |

## The table (defaults)

| # | Config | Layers (alias entries) | Rows / record | Row format, bytes | Record bytes | Records per (slot, layer) | `max_sequence_length` declared | Tensor type | Served range | Device group |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `0_window_kv` | 0-39 | 32 | bf16 x 512, row-major, 1024 B | 32,768 | 4 (positions 0, 32, 64, 96) | 128 (P.43) | Snapshot | 128 rows: position p at row p % 128 for p in [cap - 128, cap); rows with no such p are zero | whole mesh (32 chips) |
| 1 | `1_compress_kv_r2` | 2, 8, 14 | 16 | bf16 x 512, row-major, 1024 B | 16,384 | max_seq / 32 | max_seq | History | [0, cap) | one chip (TP-dedup) |
| 2 | `2_index_k_r2` | 2, 8, 14 | 16 | Blaze fp8 row, 192 B | 3,072 | max_seq / 32 | max_seq | History | [0, cap) | mesh row (TP-replicated) |
| 3 | `3_compress_kv_r1` | 20 | 32 | bf16 x 512, row-major, 1024 B | 32,768 | max_seq / 32 | max_seq | History | [0, cap) | one chip (TP-dedup) |
| 4 | `4_index_k_r1` | 20 (+ 24, 28, 32, 36 as aliases of 20) | 32 | Blaze fp8 row, 192 B | 6,144 | max_seq / 32 | max_seq | History | [0, cap) | mesh row (TP-replicated) |

- Config names sort in table order: `KvChunkAddressTable` numbers configs in sorted name order (DS41-6).
- Shards are ND `[1, 1, 32 / r, width]` ROW_MAJOR, round-robin over every DRAM bank: record j in bank j % banks at
  `base + (j // banks) x record bytes`.
- The snapshot holds every layer, including the decoder layers 20-39, because prefill runs all 40 layers on every
  chunk by default (`DECODER_SKIP=off`). Under `DECODER_SKIP=decode` (P.51) only layers 0-19's are written.
- Only the KV sources (2, 8, 14, 20) carry compressed KV. The re-index layers 24, 28, 32, 36 read layer 20's index
  keys, so they get alias entries in config 4 and no bytes.

### Entry counts per slot

Table entries are set for every position up to the declared length; decode reads only the records below cap.
Distinct bytes count one copy of each record (aliases and replicas add none).

| Config | Entries per slot at 56,320 | Records below cap (56,256) | Distinct MiB per slot | Entries per slot at 1,048,576 | Records below cap (1,048,512) | Distinct MiB per slot |
|---|---|---|---|---|---|---|
| `0_window_kv` | 160 (40 x 4) | 160 | 5.00 | 160 (40 x 4) | 160 | 5.00 |
| `1_compress_kv_r2` | 5,280 (3 x 1,760) | 5,274 | 82.41 | 98,304 (3 x 32,768) | 98,298 | 1,535.91 |
| `2_index_k_r2` | 5,280 (3 x 1,760) | 5,274 | 15.45 | 98,304 (3 x 32,768) | 98,298 | 287.98 |
| `3_compress_kv_r1` | 1,760 | 1,758 | 54.94 | 32,768 | 32,766 | 1,023.94 |
| `4_index_k_r1` | 8,800 (5 x 1,760; 7,040 are aliases) | 8,790 | 10.30 | 163,840 (5 x 32,768; 131,072 aliases) | 163,830 | 191.99 |
| **Total** | 21,280 | | **168.10** | 393,376 | | **3,044.81** |

56,320 is the bring-up's test length (11 chunks of 5,120). 1,048,576 (1M) is the deployment's maximum.

## The Blaze fp8 index-key row (192 B)

One row per compressed position, uint8:

| Bytes | Content |
|---|---|
| 0-127 | e4m3 of key[i] / scale, 128 values in the bf16 key's order |
| 128-131 | scale, fp32 little endian: `clamp(amax |key|, 1e-4, 3e38) x float32(1/448)` (not a power of two) |
| 132-191 | zero |

- A pad row (a group past the prompt's end) is all zero, its scale too.
- Made on the device: the stock `ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x,
  round_scale_to_power_of_two=False)`, then `ttnn.bringup.pack_fp8_rows`. Byte-exact with that stock cast, which is what
  tt-blaze's own test `test_dsa_indexer_direct_k_cache_update.py` checks its writer against.
- The cast takes the chunk's bf16 keys as they are: fp8(key), the stock cast of the bf16 key (P.44,
  `INDEX_K_SERVED_FROM=bf16`, the default). The indexer's own bfp8 copy (P.41) is rounded from the same bf16 rows and
  is not what is served. P.41 to P.43 cast from the bfp8 copy instead (fp8(bfp8(key)); still selectable with
  `INDEX_K_SERVED_FROM=bfp8`).
- Not byte-exact with tt-blaze's torch recipe `_quantize_scaled_keys`: about 1.1 % of values are one e4m3 step higher,
  and e4m3 subnormals are kept (DS41-213). See open point 1.

## Switches that change the table

All in `tt/settings.py` (environment variable `DS41_<NAME>`). Verified by calling `served_configs` with each value.

| Switch | Default | Other values and their effect on the table |
|---|---|---|
| `WINDOW_KV_KEEP` | `snapshot128` | `full`: `0_window_kv` becomes a History config of every position, TP-dedup, max_seq / 32 records per layer (1.25 GiB a chip a slot at 1M). `ring` / `none`: no `0_window_kv` at all (the serving runtime refuses both) |
| `REINDEX_SERVE` | `alias` | `copy`: layers 24-36 in `4_index_k_r1` get slabs of their own, written with layer 20's rows (same entry counts, ~96 MiB more a chip a slot at 1M) |
| `INDEX_K_SERVED` | `fp8_row192` | `bf16`: index rows are bf16 x 128 (256 B), records 4,096 / 8,192 B, and `4_index_k_r1` covers layer 20 only (no re-index entries) |
| `CKV_FORMAT` | `bf16` | `bfp4_rows`: changes what attention reads and the values' precision, **not the table**. The served records stay bf16 rows (holding 4-bit-rounded values). Checked: `served_configs` gives the same table |
| `PREFILL_END` | `cap` | `end`: prefill computes and writes every position up to the engine's actual_end (P.42's path); the served bytes below cap are the same (P.49 test), the table is the same |
| `DECODER_SKIP` | `off` | `decode` (P.51): the table is the same; prefill serves history [0, P) and the snapshot of layers 0-19 at P (P = (N - 128) // KV_BLOCK_SIZE x KV_BLOCK_SIZE, tt-d-gen's stop); layers 20-39's snapshot records are not written (decode rebuilds those windows). A chunk's global_write_start keeps every history record below it unwritten |
| `KV_BLOCK_SIZE` | 64 | Changes only cap (and so which window rows the snapshot holds and how many history records decode reads). Set it to the deployed worker's value |

## Changed in P.51 (2026-11-12, owner-authorized contract change, decode's proposal; behind `DECODER_SKIP=decode`, default off)

- **Prefill stops at P and never runs layers 21-39.** tt-d-gen (not yet changed) will stop prefill at
  P = (N - 128) // KV_BLOCK_SIZE x KV_BLOCK_SIZE and send chunks with actual_end <= P; decode runs [P, N) through all
  40 layers. Under `DECODER_SKIP=decode` prefill runs layers 0-19 and layer 20's KV / index head; every layer is still
  acked once a chunk.
- **Served:** configs 1-4 for [0, P) as before; `0_window_kv` holds valid rows only for layers 0-19 (rows [P - 128, P)
  at p % 128). Layers 20-39's 4 snapshot records keep whatever the slot held (zeros on a fresh slot): decode must not
  read them. The table itself (configs, entries, layers, sizes) is unchanged.
- **global_write_start (new header field).** No history record (configs 1-4) of a position below it is written by
  the chunk; positions in [actual_start, global_write_start) are an encoder replay (layers 0-19, windows truncated at
  actual_start). Missing = actual_start.

## Changed in P.49 (2026-11-12, owner-authorized contract change; pending the owner's acceptance)

- **Prefill stops at cap** (`PREFILL_END=cap`, the new default; `end` keeps the old path). Prefill computes only
  positions [0, cap) of a request: in the chunk the engine ends at `actual_end`, positions at or past cap are treated
  as padding (not attended, not written to any cache, not hashed for Engram). The chunk still runs at its fixed size and
  still sends one ack per layer naming the engine's `(actual_start, actual_end)`. A last chunk with no position below
  cap computes nothing: it rewrites each layer's snapshot at cap and acks. History records in [cap, actual_end) are no
  longer written (they hold whatever the slot held before); decode replays those positions anyway.
- **cap equals decode's for every prompt length (DS41-219).** tt-d-gen at `48b79127fb1`: under PD_DISAGG a prefill
  node's prompt ends at `prompt_compute_end = max(prompt_len - 1, 1)` (`engine/include/engine/runtime/types.hpp:
  101-110`), the last chunk ends there (`engine/src/runtime/prefill_writer.cpp:125-131`, `hi == prefill_end`), decode
  asks for `[0, prompt_len - 1)` (`backend_runtime.cpp:817`, `decode_prompt_start`) and decode's cap is
  `reusable_prefix_cap(kv_block, prompt_len) = (prompt_len - 1) // kv_block * kv_block`
  (`engine/include/engine/control/prefix_indexer.hpp:78-81`). So prefill's cap = `actual_end // kv_block * kv_block`.
  P.42 used `(actual_end - 1) // 64 * 64`, 64 positions early when prompt_len % 64 == 1 (2,945 tokens: 2,880 instead
  of 2,944). Checked for every prompt length 1..258 at blocks 32 and 64 (`server_rules.self_check`).
- **Non-last chunks are unchanged.** A recurrent (snapshot) slot always starts cold (`backend_runtime.cpp:1063-1064`),
  so chunks start at multiples of the chunk size (5120, a multiple of 64) and a non-last chunk's cap is its end.
- **Bit-identity.** Every served byte below cap (the 4 snapshot records of every layer, every history record of
  [0, cap)) is the same under `end` and `cap` (`tests/test_prefill_end.py`, both prompts, 6 prompt lengths).

## Changed in P.43 (committed 2026-11-07)

- `0_window_kv` now declares `max_sequence_length` = 128, as decode's table does. P.42 had declared the prefill max
  sequence length, because the framework contract check looked up every config at every chunk and tt-metal's table
  refuses a lookup past the declared length. P.43 made the check treat Snapshot configs as tt-d-gen does (DS41-218:
  checked only at the request's last chunk).
- Under `DECODER_SKIP=exact` / `replay128` the last chunk's start now comes from cap (`SKIP_TAIL_START=cap`), so
  every layer's snapshot exists in those modes too. `DECODER_SKIP` stays `off` by default.

## Open points

| # | Point | State |
|---|---|---|
| 1 | **fp8 quantizer rounding.** Brian's review asks for a byte-for-byte match with decode's `pack_index_keys` (`tests/blaze/deepseekv4_1/csa2_harness.py:402`), "round-to-nearest-even". The served rows are the stock cast of the bf16 key (P.44: fp8(key), no bfp8 rounding first), which is what tt-blaze's writer test (`test_dsa_indexer_direct_k_cache_update.py:300`, bytes 0..131, atol 0) compares Blaze's writer `DsaIndexerDirectKCacheUpdate` with. `pack_index_keys` is a different recipe from both the writer and the stock cast: it rounds to nearest even and divides by 448; about half of the e4m3 values and the scale bytes differ from the writer's recipe on random keys (CPU, 2026-11-10). We do not change the cast to match it. | Open: decode to confirm `pack_index_keys` is a test helper, not the writer's contract |
| 2 | **cap from prompt length vs where prefill stops (DS41-219).** At tt-d-gen 48b79127fb1 a PD prefill stops at prompt_len - 1. P.42's cap used that end as if it were the prompt length, 64 positions early when prompt_len % 64 == 1. | Closed in P.49: cap = actual_end // 64 x 64 = decode's (see Changed in P.49) |
| 3 | **No last-chunk flag.** tt-d-gen's chunk header `{slot_id, actual_start, actual_end}` has no "last chunk" bit, so prefill rewrites the snapshot on every chunk (+4.6 ms device a chunk at 51,200, P.42 A/B). The last chunk's version is the one shipped. | Owner raising the flag with tt-d-gen |
| 4 | **Single-turn only.** A 128-row snapshot clamped at cap cannot serve a follow-up turn that starts past cap. | Open: agree single-turn with tt-d-gen, or recompute on follow-up |
| 5 | **Engram at resume.** Decode's first tail step at cap hashes token ids before cap (Engram layers 1 and 14); the hand-off does not carry them. | Open: is Engram in the first serving version? |
| 6 | **Multi-slot caches on decode.** Branch `llong/dsv41-mu-main`, under review. | Decode side |
| 7 | **Device groups.** Index rows are TP-replicated (4 copies), the snapshot is on all 32 chips. Deduplicating is not decided. | Open |
| 8 | **Compressed-KV fan-out.** Brian's review says reuse layers get compressed KV through `alias_rows` too. Our table publishes compressed KV only under the sources 2, 8, 14, 20. Whether decode expects alias entries for the consumer layers is **unverified**. | Open: ask decode |
| 9 | **Empty rows.** Brian's review: confirm the deployed tt-d-gen worker skips destination rows left unfilled (older workers wrote them to noc_addr 0). | Unverified |
| 10 | **Temporary formats.** bf16 compressed and window KV is a first cut (P.38). 4-bit compute (`CKV_FORMAT=bfp4_rows`) does not change the served bf16 records. Decode keeps its window in bf16 on layers 0-1 ("being converted to bf16", Brian's review). | Owner returns to 4-bit later |
| 11 | **Owner sign-off** to drop "bf16 KV cache" for the two index configs. | Given 2026-11-06 |
| 12 | **Real KV Manager test.** tt-d-gen's KV Manager kernel (`kv_manager/device_io/kernel/dmk.cpp`) uses the same 4 DRISC tiles a chip as our old `flat_drisc` op. The default `MOE_BLOCK_SRC=main` uses no DRISC. Our runner tests use mock migration. | Owner: "not yet" |

## How the numbers were generated

```bash
source python_env/bin/activate; export PYTHONPATH=$PWD
python - <<'EOF'
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
from models.demos.deepseek_v41_flash_d_p.tt.runners.kv_contract import served_configs
from models.demos.deepseek_v41_flash_d_p.tt.attention import snapshot_cap
cfg = Config.from_dir("/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/hf_full")
for c in served_configs(cfg, list(range(40))):
    print(c["name"], c["layers"], c["alloc"], c["rows"], c["row_bytes"], c["record_bytes"], c["dedup"], c.get("snapshot"), c.get("alias"))
print([snapshot_cap(n - 1) for n in (100, 128, 129, 2945, 56320, 1048576)])  # actual_end = prompt_len - 1 (P.49)
EOF
```

Host only, no device. Set `DS41_<SWITCH>=<value>` in the environment to see another value's table.
