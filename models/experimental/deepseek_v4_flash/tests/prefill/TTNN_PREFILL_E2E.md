# ttnn prefill + traced decode on ONE galaxy — status and hand-off

Branch `sdawle/dsv4-flash-prefill-decode-1glx` = the pure-ttnn DeepSeek-V4-Flash prefill
(`models/demos/deepseek_v3_d_p/tt/v4`, branch `sdawle/dsv4-flash-prefill-ttnn`) merged with Sankar Manoj's ttnn decode
(`models/experimental/deepseek_v4_flash`, branch `smanoj/ds_v4_flash`), plus the hand-off between them.

**Status (2026-09-29): merged, built, and pushed. The e2e has NOT run end to end yet.** Three attempts on host 41 each
stopped before any model forward. Two causes are fixed; the third fix is committed but has not run on a device yet (see
"Blockers" below).

## Layout on the galaxy (one process, one `(8,4)` mesh open)

| rows | what | notes |
|---|---|---|
| 0-1 | Sankar's traced decode, TP4, two 1x4 stages (layers 0-21 / 22-42) | his default; the 2-stage cap is `tt/model.py` `min(pipeline_devices, 2*tp_size)` |
| 2-3 | idle, or his prefill when `DEEPSEEK_V4_E2E_COMPARE=sankar` | |
| 4-7 | our ttnn prefill on a `(4,4)` submesh (SP4, TP4, EP 16 experts/chip) | `create_submesh(MeshShape(4,4), MeshCoordinate(4,0))` |

Mesh: `FABRIC_2D_TORUS_XY`, 2 command queues, `l1_small_size=1152` (the decode's requirements; the prefill never ran
under TORUS_XY before this branch). Chosen over "both models on all 32 chips" because the decode's prefetcher GCB rings are
permanent L1 (288 KB per receiver core) on every decode chip, and our prefill is already L1-tight.

## Flow (`tests/prefill/test_ttnn_prefill_decode_e2e.py`)

1. tokenizer + prompt (`DEEPSEEK_V4_E2E_PROMPT_LEN`, default 1000)
2. decode model on rows 0-1 → `prepare_static_decode` → session → throw-away `decode_traced(pad, 0)` (captures traces)
3. our prefill on rows 4-7: `DeepSeekV4FlashAdapter.build_runtime` (mesh_shape `(4,4)`, `kv_only_last_layer=True`,
   `use_trace=False`) → `allocate_kv_cache` → `compile`
4. prefill `T0 = floor((len-1)/128)*128` tokens (chunk `DEEPSEEK_V4_E2E_CHUNK`, default 2048)
5. hand-off (`tt/prefill/handoff_ttnn_prefill.py`): read each layer's WORKING state (bf16/fp32, not the tt-blaze export
   caches) → write into the decode buffers `prepare_static_decode` allocated:
   - SWA / every layer: `sliding_carry [1,1,128,512]` (token order = ring order since `T0 % 128 == 0`) → ring rows 0..127
   - CSA: `compressed_kv[:entry_count]` → rows `128 + w`; `prior_c` last 4 rows → `prev_kv`, `prev_gate` with the
     position bias SUBTRACTED (decode adds it inside `csa_pool_window`); Cb gate = `_MASK_NEG`
   - HCA: entries + ring → the paged pool through the session page table (reuses `handoff._pool_rows`)
6. replay the ragged tail `T0..len-1` through `decode_traced`; the last step gives the first generated token
7. decode `DEEPSEEK_V4_MAX_NEW_TOKENS` tokens, print text + TTFT / hand-off s / tok/s

Checks: `DEEPSEEK_V4_E2E_COMPARE=decode` replays the same `T0` tokens through the decode itself, reads its caches back and
compares our hand-off against them per layer (PCC), then rewinds the session. `=sankar` compares against Sankar's own
prefill on rows 2-3 instead.

Limit: **the whole conversation must stay under 2048 tokens.** The lightning indexer's state (`idx_key_cache`,
`comp_kv`, indexer compressor windows) is not handed over yet; past 2048 the decode would score an empty index.

## Run it

```bash
cd <this repo>   # host 41: /home/ttuser/sdawle/tt-metal-dsv4-1glx
TAG=run4 NEW=64 PLEN=1000 COMPARE=decode \
  models/experimental/deepseek_v4_flash/tests/prefill/run_ttnn_prefill_decode_e2e.sh
```

The launcher resets the galaxy (`tt-smi -glx_reset_auto`), gates on an `(8,4)` TORUS_XY mesh open (resetting up to 4 times;
after a reset host 41 sometimes cannot map the mesh), then runs the pytest under a 14000 s timeout. Paths are env overrides
(`DEEPSEEK_V4_CACHE_DIR`, `PREFILL_TTNN_CACHE`, `PREFILL_HF_MODEL`, `E2E_LOG_DIR`).

## Prerequisites per host

| item | host 41 | why |
|---|---|---|
| KMD `options tenstorrent dma_address_bits=58` (`/etc/modprobe.d/tenstorrent.conf`, module reload) | **set** | default (64-bit) hangs every large host→device write: a 1 GB embedding upload never completed; with 58 it takes 5.1 s. quad29 hosts already had 58; **hosts 42/43 still have the default** |
| Blackhole firmware bundle ≥ 19.12.0.0 | **19.11.0.0** | decode's DRISC weight prefetcher needs DRAM programmable cores (see Blockers) |
| `transformers==5.12.1` in `python_env` | installed | decode imports `transformers.models.deepseek_v4` |
| HF checkpoint in the hub layout | symlink `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731` | decode test hard-codes the hub path |
| decode bf4 tile cache `DEEPSEEK_V4_CACHE_DIR` | **empty** (only the checkpoint link) | first decode build converts every weight (> 1 h, from the decode's docs; not measured here) |
| prefill tile cache for `(4,4)` under `PREFILL_TTNN_CACHE/deepseek_v4_flash_bh_32dev/4x4` | **missing** (only `8x4`, 282 GB) | first prefill build on the submesh converts every weight (unmeasured) |

## Attempts so far (host 41)

| run | reached | stopped by | status |
|---|---|---|---|
| run1 | decode weight upload | hang in `wait_for_outstanding_reads` on the 1 GB embedding write: KMD 64-bit DMA mask | fixed on host 41 (`dma_address_bits=58`) |
| run2 | mesh open | `Graph specified in MGD could not fit` after a reset (an eth link came up untrained) | launcher gates on a mesh open and resets again |
| run3 | step 2/8, decode build (17 s in) | `TT_FATAL: DRAM-sender GlobalCircularBuffer requires programmable DRAM cores, which auto-enable on Blackhole with firmware >= 19.12.0.0` (`global_circular_buffer.cpp:173`) | patch committed, device-untested |

## Blockers and the fix in flight

**Decode prefetcher vs firmware 19.11.** The decode streams its projection weights through DRAM-sender global circular
buffers (the DRISC tensor prefetcher). tt-metal registers DRAM programmable cores only on firmware bundle ≥ 19.12.0.0
(`tt_metal/llrt/firmware_capability.cpp`); older firmware runs syseng firmware on a DRAM core the prefetcher wants
(#45751). **Do not** force it with `TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1` on 19.11.

The fix on this branch turns the prefetcher off where the device cannot run it:
`prefetcher.enabled` / `DEEPSEEK_V4_PREFETCHER` (unset = `ttnn.experimental.is_tensor_prefetcher_supported`, `0`/`1` pin it).
With it off, no GCB is built and every projection takes its existing DRAM→L1-copy-per-call path (the one the MTP stack
and the TP projections the shared GCB cannot serve already use). The CSA indexer was the one component hard-wired to the
prefetcher; it now honours the flag. Host checks pass (imports, env parsing). **Not run on a device yet.** Risks:
L1 pressure from the hoisted DRAM→L1 copies, and decode tok/s lower than Sankar's prefetched numbers (not comparable).
The alternative is flashing firmware ≥ 19.12 on the galaxy.

## Plan status

| milestone | status |
|---|---|
| M0 workspace (fresh clone on host 41) | done |
| M1 merge + build + push | done: 30 conflicts + 7 breakages outside the markers (tt-blaze postmortem DS4F-0282) |
| M2 baselines on the merged tree | **not run**: (a) decode alone, (b) Sankar's own e2e, (c/d/e) our prefill 4-layer golden PCC on FABRIC_2D / TORUS_XY / the 4x4 submesh, (f) decode on 4 stages |
| M3 hand-off module | written; not validated (no host unit test, no device comparison yet) |
| M4 co-resident e2e | test + launcher written; 3 attempts, none reached a model forward (table above) |
| M5 both models on all 32 chips | not started (optional) |
| M6 past 2048 tokens (indexer hand-off) | not started |

## Next steps, in order

1. **Decode alone with the prefetcher off** (M2a): `tests/decode/test_full_model_decode_demo.py -k tp4_32chip` with
   `DEEPSEEK_V4_CACHE_DIR` set. Validates the patch and populates the bf4 cache (> 1 h the first time).
2. **Our prefill on the 4x4 submesh under TORUS_XY**, 4 layers first (golden PCC ≥ 0.9997), then 43 layers (populates the
   `4x4` cache; watch per-chip DRAM).
3. **e2e with `COMPARE=decode`** at `PLEN=1000`: the per-layer hand-off PCC is the first correctness gate, then coherent
   text, then ≥ 3 consecutive requests without a reload.
4. Record TTFT / hand-off s / decode tok/s; then M6 (indexer hand-off) for long context.

Estimated: ~4-5 h wall clock if nothing new breaks (two cache builds are sequential on one galaxy), 1-2 working days
realistically. The `4x4` prefill cache can be built on a second galaxy in parallel: `/mnt/tt-data` is one NFS export
(`10.82.97.2:/mnt/TT-Data`) on hosts 29/41/42, the quad29 hosts already have `dma_address_bits=58`, and 42/43 would need it
first. The prefill does not use the prefetcher, so firmware 19.11 is fine for it.

## Record

Defects, fixes and refuted hypotheses: tt-blaze `docs/deepseek_v4_flash_postmortem.md` (branch
`sdawle/dsv4-flash-prefill-ttnn`), entries DS4F-0282 (merge), DS4F-0283 (DMA mask), DS4F-0284 (prefetcher firmware floor).
