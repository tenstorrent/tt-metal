# h47i merge notes (prefill -> paged hand-off -> decode generator, demo)

`changes.diff` applies cleanly (`git apply -p1`) to main 45d3858ddd7 (checked with `git apply --check`).

## New files (mine)
tt/dsv41_model.py (Model), tt/prefill_handoff.py (PagedStateSink + GenPrefillModel), tt/engram_ragged.py, tt/generator.py (Generator),
tt/common.py (create_tt_model), tt/model_args.py (HF tokenizer / chat template), configs/gate_cutoffs.json, demo/text_demo.py (+ demo/sample_prompts/*),
tests/test_e2e_prefill_decode.py.

## Dependencies
- h44p's prefill files (chunked prefill + `state_sink` hook + dyn/traced-chunk code) are ALREADY in main's staged tree: not part of this diff. My verified runs used the earlier h44p snapshot
  (single-chunk path + sink); main's newer snapshot only adds the dyn path (unvalidated by me; default path unchanged: `forward_device(dyn=False)`).
- tt/paged_ops.py + paged_kernels/row_scatter.cpp: h46k's `paged_scatter_rows(..., base_offset=)` re-applied minimally on main's formatting (h46k's own diff has the same change; identical semantics).

## VERIFIED on device (host .45, logs /mnt/tt-data/ssinghal/dsv4-logs/h47i_*.log)
- Full 40 layers, B=16 (U=4), ISL 128, prefill (eager chunk path, 1 chunk) -> paged pool -> traced device-loop decode: first-token logits PCC 0.972 (argmax 14/16);
  teacher-forced decode logits PCC 0.983/0.975/0.980 (state written only by the prefill); closed loop 47.1 ms/token = 21.2 tok/s/user at B=16.
- Per-user RAGGED prompt lengths 2..128 (odd+even), layers 0-3: ring PCC >= 0.9986, compressed latents >= 0.9996, ratio-2 prev_cs >= 0.9995.
- Real GSM8K prompts through the demo (chat template, 16 users of different lengths 30..111 tokens): 12/12 completed answers equal the gold answers (4 not finished in 256 tokens).
  Decode 47.1 ms/token (21.2 tok/s/user, B=16); same prompt x16: 37.6 ms/token.
- U=8 / U=32 (batch 32 / 128) build + prefill + decode run on 4 layers (smoke only, no accuracy).

## NOT verified / gated
- Traced chunk prefill (`Model.prefill_forward_dyn`, `enable_trace=True`) is gated by `DSV41_PREFILL_DYN=1` (needs h44p's run_traced_chunks, not yet validated); default = verified eager path, so
  the demo's TTFT is EAGER (ISL 128 b16: ~5.8 s) until the trace lands. `PagedStateSink` already uses persistent per-chunk tensors + `update(s0, C)` for it.
- Multi-chunk prefill into the pool (S > one chunk) is only exercised by chunk-vs-whole in h44p's tests, not vs the dumps by me.
- Context > 512 tokens (ratio-1 layers; > 1024 for ratio-2): needs prefill indexer (+ key-slab writes) and per-user-length decode indexer: demo scenarios isl4k..64k xfail early;
  Model.check_context_supported raises NotImplementedError unless DSV41_ALLOW_DENSE=1 (experiments only). 128k/256k/1M scenarios are `skip` (never executed, per instruction).
- First-token prefill logits at ragged lengths for 40 layers (test code exists via DSV41_LENS, not run at 40 layers).
- Greedy only (temperature 0); top-k/top-p sampling not implemented.
- NOTE: the demo module imports tt_transformers simple_text_demo, which opens the device driver at import time (like GPT-OSS' demo): do not import it on a host where the device is busy/needs a reset (I hit `Read 0xffffffff over PCIe ID 22` importing it on .44: .44 may need `tt-smi -glx_reset`).


## Increment 2 (against main 2b71af27ee1)
- tt/dsv41_model.py: traced-chunk prefill driver `prefill_forward_dyn` (env DSV41_PREFILL_DYN=1, still OFF by default), `prepare_for_traces` (all persistent tensors + decode compile pass before any capture),
  ragged last-token head INSIDE the chunk trace (per-(row,user,subchunk) 0/1 mask tensors, outputs read host-side in `_post_chunk`, no eager device ops between replays).
- tt/prefill_handoff.py: PagedStateSink uses persistent per-chunk index tensors refreshed by `update(s0,C)` (trace safe); env debug DSV41_SINKDBG / DSV41_SINKSKIP.
- tests/test_e2e_prefill_decode.py: DSV41_PREFILL_TRACE, ring/state diagnostics.
- STATUS of traced prefill on device (4 layers, ragged lens 2..128, chunk 128): trace replays work and are fast (TTFT 1.39 s vs 3.86 s eager for 16 users, 976 tok/s) and the 2nd call no longer hangs once the
  head is inside the trace; the hand-off of layers 0-2 is correct (ring >= 0.9995, latents 0.9996, prev_cs 0.9995). BUT the dyn path (h44p's forward_dyn) produces an exactly-zero stream after the first
  ratio-2 owner layer (layer 2) at S=C=128 (reported to h44p with repro; independent of my sink), so layers >= 3 get no state. Do NOT enable DSV41_PREFILL_DYN by default until h44p fixes it.
- Default path (eager legacy chunks) unchanged and verified (see above).

## Increment 3
- VERIFIED 40 layers, B=16, ISL 128 (log dsv4-logs/h47i_full_dyn2.log): TRACED chunk prefill (h44p S_pad==C fix, tt/prefill_attention.py in diff) + PagedStateSink inside the replayed trace + paged traced decode:
  first-token PCC 0.972 (14/16), teacher-forced 0.983/0.975/0.980 (15/16,15/16,13/16) = identical to the eager path; TTFT warm 2.46 s (834 tok/s) vs 7.03 s eager; closed loop 46.8 ms/token (21.4 tok/s/user).
  4 layers ragged lens 2..128 traced: ring >=0.9976, latents 0.9996, prev_cs 0.9995.
- Traced prefill is now the DEFAULT (DSV41_PREFILL_DYN=0 -> eager reference). launch.sh: hangwatch attached only to the real pytest pid.

## Increment 4 (against main bdde8dcb6e0): indexer wiring + integrated gates
- tt/dsv41_model.py: decode indexers built per index-source layer (`_build_indexer`, key slab on owners, aliases on 24..36), step state with per-user valid lengths, prefill sparse (h46x) enabled automatically when max_ctx > dense limit
  (DSV41_INDEXER=auto|0|1), prefill key slabs exported to the decode key slabs after every prefill (`_export_index_keys`).
- demo/text_demo.py: trace_region_size 1.6e9 (40-layer chunk trace with sparse needs 1.11 GB), small scenarios use max_seq_len 512 (indexer stays off), session mode.
- tests/test_e2e_prefill_decode.py: DSV41_LAYERCHECK=1 per-layer decode check vs dump dec_out; dumps with fewer users than the batch are tiled.
- Gates: G1 PASS (main, 40 layers, ISL128 B16: first token 0.972, teacher-forced 0.983/0.975/0.980, TTFT 2.46 s, 47.4 ms/token).
  G3 PASS layers 0-20, ISL 2048 (traced chunked sparse prefill C=512 -> key export -> decode indexer): per-layer decode out PCC min 0.992 (layer 10) .. 0.99999, layers 2/3 (first indexer) 0.997/0.9999, layer 20 0.99996.
  G4: GSM8K b16 (40 layers): 14/16 finished, 14/14 correct, TTFT 1.22 s, 54.6 ms/token; b64: 60/64 correct (63 finished), TTFT 9.5 s, 68.7 ms/token (932 tok/s); ISL 3720 b16: TTFT 38.6 s (1540 tok/s traced), 45.8 ms/token.
  Known: running a second, longer ISL in the same process OOMs DRAM (teardown_dyn/sparse alloc_dyn leak, reported to h46x): run one ISL per process.
- G5 (spec decode): NOT hooked: merged spec decode (spec_attention.py) uses its own contiguous paged TILE caches, contexts < 128 and seeds its state by feeding the prompt through the device; it does not use the RM paged pool / prefill hand-off.
- G4 ISL ladder at 40 layers, B=16, one ISL per process (traced chunked prefill + sparse indexer, TTFT = warm replay, warmup excluded):
  ISL 3720: TTFT 38.6 s (1540 tok/s), decode 45.8 ms/token; ISL 7443: 75.0 s (1588 tok/s), 44.2 ms/token (22.6 tok/s/user); ISL 16038: 149.4 s (1718 tok/s), 46.3 ms/token (21.6 tok/s/user).
  Not accuracy-checked beyond 2k (no reference); 32k/64k, b32+ at long ISL not run.
