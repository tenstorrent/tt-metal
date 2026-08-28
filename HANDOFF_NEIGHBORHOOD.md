# LTX-2.5 DiffVAE / neighborhood attention -- handoff

Machine: 4x8 Blackhole galaxy, `/home/nwoodall/tt-metal`, branch `na-integration`.
Deep detail and every measurement: `models/tt_dit/layers/NEIGHBORHOOD_STRIDE1_FINDINGS.md`.

## Terminology (these four get confused constantly)

| term | value here | what it is | changes the output? |
|---|---|---|---|
| **window** (context_window, kernel) | `(11,11,11)` | how many keys a query attends -- 1331. The architecture. | yes, it IS the model |
| **stride** (GNA stride) | `(1,1,1)` exact NA, or e.g. `(6,8,4)` | how far the window JUMPS between query groups. stride 1 = every query centred on its own window. stride > 1 = a group shares ONE window (GNA) | **yes** -- this is what causes the rectangle artifacts |
| **brick** | `(2,4,4)` = 32 sites | memory layout: 32 consecutive sites as a compact 3D box, one tile row | no |
| **chunk** (`query_chunk_bricks`) | derived from stride | how many query BRICKS share one K/V gather. Pure scheduling | no (once per-brick masks are on) |

Others: **gather / box** = the union of a chunk's windows, what gets fetched.
**keys/query** in the plan log = gathered sites / queries = fetch cost per query (NOT 1331).
**slot** = one key brick within the gather. **band / slab** = frame-banding of stage 5
(`DIFFVAE_SLAB_FRAMES`). **regime** = the old 27-class mask table (a GNA construct).

## The two executors

- **reference / "old"**: `op_sp_w_sharded` in `models/tt_dit/layers/na3d.py`. Full-W kv-allgather + retile, fused SDPA.
- **ours / "the op"**: `bricked_sp_w_sharded` in `models/tt_dit/layers/neighborhood_attention.py` + the C++ op under `ttnn/cpp/ttnn/operations/transformer/sdpa/device/` (`neighborhood_*`). Halo exchange + bricked layout + in-kernel gather.

## Env vars

Required on this host (weights are NOT at /mnt/MLPerf):

    export LTX25_ROOT="$HOME/.cache/ltx-checkpoints/ltx-2.5"                       # decode-only tests
    export DIFFVAE_CHECKPOINT="$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors"
    # full pipeline instead wants the HF snapshot:
    # LTX25_ROOT=/home/nwoodall/.cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/6c7e5e573ac1667efc83407806fe9b0b93730e60

Selecting the executor / attention:

    DIFFVAE_STAGE5_BACKEND=bricked_sp_w_sharded   # ours; unset = op_sp_w_sharded (reference)
    DIFFVAE_DET_NA3D_BACKEND=gather               # stages 1-4 ONLY, separate from the above
    DIFFVAE_S5_GNA_STRIDE=1,1,1                   # exact NA (stage 5 only). 6,8,4 = GNA
    DIFFVAE_GNA=0                                 # puts the REFERENCE at stride 1 (its own knob)
    DIFFVAE_BLOCK_PROF=1                          # deep spans in the decode tree; needed for any breakdown
    DIFFVAE_TP_HEADS=1                            # TP over heads, 4x. Canonical default. Worth ~4x
    DIFFVAE_SLAB_FRAMES=73                        # canonical. 48 if a band OOMs

Diagnostics added this session (all default off):

    DIFFVAE_NA_TABLE_ALWAYS=1      bypass the mask gate -- WRONG at edges, right timing (17.5 s)
    DIFFVAE_NA_MASK_MEMSET_ONLY=1  fill masks with a constant -- WRONG output, gives the content-free floor (11.0 s)
    DIFFVAE_NA_CHUNK_BRICKS=t,h,w  force the chunk, in BRICKS
    DIFFVAE_NA_UNSAFE_CHUNK=1      lift the plan's chunk==stride check (needed with the above at stride 1)
    DIFFVAE_NA_PER_BRICK_MASK=0|1  force per-brick masks; defaults on when chunk != stride
    DIFFVAE_NA_HALO_TOPOLOGY=ring  retest the halo on ring once neighbor_pad is fixed
    DIFFVAE_EXCLUSIVE=1            restore the DiffVAE's old evict-the-DiT behaviour

## Commands

Decode-only timing (this is what all the numbers below are):

    cd /home/nwoodall/tt-metal
    export LTX25_ROOT="$HOME/.cache/ltx-checkpoints/ltx-2.5"
    export DIFFVAE_CHECKPOINT="$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors"

    # ours, exact NA
    DIFFVAE_BLOCK_PROF=1 DIFFVAE_STAGE5_BACKEND=bricked_sp_w_sharded DIFFVAE_S5_GNA_STRIDE=1,1,1 \
      bash models/tt_dit/experimental/scripts/run_ltx25_diffvae.sh --timeout=0

    # ours, GNA          -> add DIFFVAE_S5_GNA_STRIDE=6,8,4 instead
    # reference, GNA     -> drop both DIFFVAE_STAGE5_BACKEND and DIFFVAE_S5_GNA_STRIDE
    # reference, exact NA-> DIFFVAE_GNA=0, drop DIFFVAE_STAGE5_BACKEND

Same thing with a heartbeat so a hang is visible (`tail -f generated/current.log`):

    DIFFVAE_BLOCK_PROF=1 DIFFVAE_STAGE5_BACKEND=bricked_sp_w_sharded DIFFVAE_S5_GNA_STRIDE=1,1,1 \
      WATCH_TIMEOUT=1800 ./watch_run.sh "label" --timeout=0

Full pipeline + video (3 gens when traced; gen #2 is the steady-state one):

    S=/home/nwoodall/.cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/6c7e5e573ac1667efc83407806fe9b0b93730e60
    LTX25_ROOT=$S LTX_TRACED=1 OUTPUT_PATH=/home/nwoodall/tt-metal/generated/out.mp4 \
    DIFFVAE_STAGE5_BACKEND=bricked_sp_w_sharded DIFFVAE_S5_GNA_STRIDE=6,8,4 DIFFVAE_TP_HEADS=1 \
      bash models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh --timeout=0

    # clear the weight cache first if the video looks wrong or a fusion flag changed:
    rm -rf "$HOME/.cache/tt-dit"

PCC against the torch reference (single chip, seconds):

    python_env/bin/python /tmp/.../scratchpad/vs_torch.py 0     # uploaded table
    python_env/bin/python /tmp/.../scratchpad/vs_torch.py 1     # device-generated
    # the op's own suite:
    python_env/bin/python -m pytest models/tt_dit/tests/unit/test_neighborhood_sdpa.py -q

Rebuilds: host C++ needs `cmake --build build_Release --target ttnn` then
`cp build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so`. Anything under `device/kernels/` is
JIT-compiled at runtime and needs NO rebuild.

## State: numbers (145 frames, 1920x1088, 4x8, ring/2-link, TP4)

| config | decode | quality |
|---|---|---|
| reference, GNA (its own stride) | **8.8 s** | artifacty (shares one window across 352-480 queries) |
| ours, GNA (6,8,4) | **9.2 s** | artifacty, but 9.00 keys/query vs their 15.2 |
| reference, exact NA | **13.8 s** | clean |
| ours, exact NA | **32.3 s** | clean, and the mask is now CORRECT (PCC 0.996) |
| ours, exact NA, gate bypassed | 17.5 s | diagnostic (edges wrong) |
| ours, content-free floor | 11.0 s | diagnostic |

Full traced pipeline, steady state (gen #2): **15.97 s/video** with ours at GNA
(enc 0.56 / s1 2.45 / upsample 0.19 / s2 2.87 / **VAE 9.39** / audio 0.51).
Cold first run was 624 s -- most of that is JIT, not model cost.

## The one open item

`brick_window_is_unclamped()` in `neighborhood_reader.cpp` admits ~10% of bricks; it should admit
~90% (only true edge bricks clamp). That alone is 32.3 s -> ~17.5 s, proven by
`DIFFVAE_NA_TABLE_ALWAYS=1`. Re-deriving the bounds against `window_origin_on_axis` changed
nothing, so it is not the arithmetic. Next suspect: `extents.shard_origin`, read per chunk from
the gather-origin table -- if its T/H components are not 0 on the W-sharded path, every brick
reads as clamped. One instrumented run would settle it.
After that, 17.5 s vs the 11.0 s floor is DMA cost for the mask reads.

## Bugs found, worth filing separately

1. `neighbor_pad_async` **deadlocks on `Topology.Ring`**, works on Linear. Not the channel width, link count or persistent buffer -- each ruled out. Worked around by pinning that one call to Linear (`_halo_exchange`); everything else still runs ring. The reference never hits it because it uses `all_gather`.
2. The uploaded **regime mask table is wrong at stride 1** (PCC 0.914) -- fixed here by the relative table, but the regime path still exists for GNA.
3. **tt-dit weight cache key omits the qkv fusion choice**, so flipping `DIFFVAE_DET_FUSED_QKV` gives a hard `LoadingError` on a stale cache instead of a rebuild.
4. `requires_exclusive_residency` on DiffVAEDecoder was a static `True` sized for the REPLICATED decoder (~8.5 GB); W-sharded it is ~1/8 of that. Now conditional -- which is what allows the DiT to stay resident and be traced. Note exact NA still OOMs co-resident; use `DIFFVAE_SLAB_FRAMES=48` or `DIFFVAE_EXCLUSIVE=1`.
