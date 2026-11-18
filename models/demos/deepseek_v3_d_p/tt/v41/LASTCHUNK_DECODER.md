# DS41F-0073: the decoder layers over the prefix's final cone only (marked decoder chunks)

Goal: remove the decode ring's fixed replay of R >= 2541 tokens per P/D request. At AgentX c20 that replay is ~41 % of
the ring's replayed prompt pages (DS41F-0073 11-17). Run decoder layers 20..39 in the prefill only on the chunks that cover
the migrated prefix's last 2541 tokens, not on every chunk. Running them on every chunk ("tree D", `V41_PREFILL_DECODER=1`)
fits at 1M with BFP4 experts but costs 0.69-0.79 s per 4k chunk vs 0.41-0.44 s (chain 69b).

Branches: tt-metal `sdawle/dsv41p_lastchunk_dec` (on the serving runner 06c29d03031), tt-d-gen
`sdawle/dsv41_lastchunk_split` (on ff91ef431, the serving row-streaming engine). Default off on both sides. With the
flags unset, the runner does the same device work, call for call, as before (`tests/v41_prefix_reuse`, 238 tests).

## 1. The mechanism today (READ)

- **How chunks reach the runner.**
  - tt-d-gen's prefill writer cuts the request's tokens into `chunk_size` chunks from 0 (`prefill_writer.cpp` `step`).
    On the prefill role, `handle_admit` has already cut the prompt to the migrated prefix S' (`backend_runtime.cpp`,
    `migrated_cap`).
  - Each chunk travels with a 12-byte header (slot word, actual_start, actual_end).
  - With `device.prefill.mark_last_chunk`, the slot word carries bit 29 ("marked") and, on the burst's last chunk,
    bit 30 (`pipeline_types.hpp` `prefill_slot_word`).
  - The runner splits the word (`common/prefill/prefix_reuse.py` `split_slot_word`). It learns that a chunk is the last
    only on that chunk, and it never sees S'.
- **Tree D (`V41_PREFILL_DECODER=1`).**
  - It builds all 40 layers as blocks (`prefill.py`: `n_layers=cfg.n_layers`, `kv_only=False`). Layer 20 becomes a full
    ratio-1 CSA source with a path-A slab; 21..39 are its Reuse / Reindex consumers.
  - It allocates the export with 38 window rings (`kv_export.py` `allocate_v41_kv_export(decoder=True)`). The chunk
    table then points the decoder layers' window chunks at `win` instead of `zero` (`table_rows`, `exp.decoder`).
- **Why 2541.**
  - In the decoder, the only cross-position paths are each layer's window (127 back) and layer 20's entries / index
    keys. Those keys come from the encoder output, so they are exact. Top-k / candidate picks are per query.
  - So a decoder started at c from zero windows has layer l's input exact at p >= c + (l - 20) x 127.
  - The export ships each layer's window rows of the prefix's last 128 positions. Layer 39 needs p >= c + 19 x 127 for
    p >= S' - 128, i.e. S' - c >= 20 x 127 + 1 = 2541 (`disagg_prefill_plan.md` lines 13-16; `decoder_cone.py`).
- **What the decoder layers read from earlier positions.**
  - Only layer 20's rows: entries, index keys, and (path A) its slab entry rows. The Reindex layers 24 / 28 / 32 / 36
    score layer 20's keys (`attention.py` `V41CSAReindex.forward`; the export points their key chunks at layer 20's).
  - Tree A computes these on every chunk via `kv_only`, into a separate `kv_state` without a slab.
  - The block's path-A forward reads entries from the SLAB (`csa.py` `forward` writes it every chunk; `_sparse_core`
    reads `state.slab_rm`). So a KV-only chunk on the block state must also write the slab rows, or every decoder chunk
    after it gathers zeros. That is not a window effect: it would be wrong at every position.
- **Prefix reuse (H1 / H2) today.**
  - H1 skips the chunks the device still holds and resumes at the first new one (checkpointed carries + silent chunks).
  - H2 restores stored chunks' rows: for a block source with `sparse_path` it also writes the slab (`_restore_rows`).
  - A partial last chunk is always computed.
  - `_build_stores` refused any layout but the encoder + layer-20 KV-only one.

## 2. Design

- **tt-d-gen (both roles, `RuntimeConfig`).**
  - `pd_decoder_cone` (0 = off; V4.1: 2541), `pd_decoder_chunk` (= the prefill's chunk_size), `pd_decoder_tail`
    (default 1), `pd_decoder_single_chunk` (policy L).
  - The migrated prefix becomes `PrefixIndexer::decoder_prefix_cap`: floor(S - tail, align) instead of
    floor(S - 2541, align). Both roles compute it, as they do `migrated_prefix_cap` today, via `migrated_cap`.
  - The prefill writer marks every chunk whose end lies past S' - cone (`PrefixIndexer::decoder_chunk`) with
    `kPrefillDecoderChunk` (bit 28, on marked headers only).
  - `holder_copy_tail` (0 = pd_migrate_tail) sets the replay a frozen-holder copy must leave. The 2541 there exists only
    because P/D landings had empty decoder windows. With the cone, landings are exact, so 1 is exact too. It is a
    separate knob so it can be A/B'd.
- **Runner (`V41_PREFILL_DECODER=1` + `V41_PREFILL_DECODER_MARKED=1`).**
  - All 40 layers are built (tree D's weights and memory).
  - A marked decoder chunk runs every layer. Any other chunk runs the encoder, then layer 20 KV-only into the BLOCK's
    state (`V41PrefillBlock.kv_only` / `kv_only_islands`: island A replay + `kv_only(slab=True)`), and skips 21..39.
  - The first decoder chunk of a prompt starts 21..39 at its start position from zero carries (`_decoder_begin`; a zero
    carry cloned before any capture). Zero, not the slot's previous windows: no exact row changes, and runs compare byte
    for byte whatever the slot held.
  - The burst's last chunk checks that the decoder started at 0 or >= cone positions before the end (`_check_cone`,
    logged).
  - An unmarked header runs the decoder (an engine without the marks: slow, never inexact).
  - Prefix reuse: a decoder chunk is never skipped or restored (`PrefixRetention.begin_chunk(force_compute=True)`: resumed
    at its start, then computed). Silent rebuild chunks run KV-only.
  - Checkpoints and H2 tips keep the default prefill's 21 carries (layers 0..20, `window_blocks`). Decoder windows are
    written only by decoder chunks, so the export equals a cold run's.
  - H2 is now allowed in this mode. Layer 20's restore writes its slab through the existing block-source path.

### Policies (`pd_decoder_single_chunk`)

| policy | prefix S' | decoder chunks | ring replay |
|---|---|---|---|
| M (false) | floor128(S - 1) | 1-2 (chunk >= cone) | 1..128 |
| L (true, chunk >= cone) | M's, cut back to its last chunk start when that chunk is shorter than the cone | 1 | < cone + 128 |

Estimate, INFERRED from the rule (`decoder_cone.py`): uniform S over [8k, 1M), chunk 4096, a tree-A chunk 0.42 s, the
decoder +0.35 s per chunk (chain 69b at 0-131k; ~+0.55 s near 1M).

| | ring replay per P/D request | prefill chunks | decoder chunks | prefill s / request vs today |
|---|---|---|---|---|
| today | 2604.5 | 127.80 | 0 | 0 |
| M | 64.5 (-97.5 %) | 128.42 (+0.62) | 1.59 | +0.82 |
| L | 824.8 (-68 %) | 127.83 (+0.03) | 1.00 | +0.36 |

L's prefill chunk count is today's: the cut-back prefix ends in the same chunk as today's floor128(S - 2541).

## 3. Turning it on

- **tt-d-gen, both workers' `runtime`.**
  - `"pd_decoder_cone": 2541, "pd_decoder_chunk": 4096, "pd_decoder_tail": 1, "pd_decoder_single_chunk": false|true`.
  - Keep `pd_migrate_tail: 2541` (the tail rule's switch).
  - Optional on the decode worker: `"holder_copy_tail": 1`.
  - The prefill worker needs `device.prefill.mark_last_chunk: true` (the serving config has it).
- **Runner env (`PREFILL_FLAGS`).** `V41_PREFILL_DECODER=1 V41_PREFILL_DECODER_MARKED=1
  V41_PREFILL_DECODER_CACHE=/mnt/tt-data/sdawle/v41p_1017_weight_cache/8x4 V41_PREFILL_DECODER_EXPERT_DTYPE=bfp4`, plus
  the serving CP2 / H1 / H2 flags (H2 is no longer refused).
  `V41_PREFILL_DECODER_CONE` (default 2541) must equal `pd_decoder_cone`.

## 4. UNVERIFIED (host tests only; nothing ran on a device)

- Bitwise equality of layer 20's rows (entries, index keys, carry, slab) between the block's `kv_only` (eager and via island
  A) and tree A's separate KV-only layer: same weights and ops, never compared on device.
- `kv_only_islands` replays island A without B. No state of B outlives a replay (by reading), never run.
- Warm-up: the compile now runs one KV-only chunk after the decoder warm-up (position 0, fresh state). The first real
  KV-only chunk on the islands may still pay a first-replay cost.
- Memory: tree D's (chain 69b: fits at 1M with BFP4). The per-slot layer-20 slab is now allocated at build, not lazily
  (the same ~1 GB per chip tree D allocated at its warm-up).
- Device plan:
  1. One cold 32k and one 131k probe. Measure TTFT, ring seat -> first token, and the replayed tokens (decode counters).
     The runner log shows "(decoder)" on 1-2 chunks per request and no `_check_cone` warning.
  2. Answer check (6 / 6) and the gate texts (answer-level: the decoder windows now come from the prefill).
  3. AgentX c20 x 2 per policy (M, L; then `holder_copy_tail` 1).
