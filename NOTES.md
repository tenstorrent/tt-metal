# t476 notes (run 1795, split at 3.0 M context)

Branch ttp/t476-turbo-sla-0-85-on-real-data-find-and-fix = t473 head ae9534f5fb2 + t454 SLA/VSA commits
(d39d16e5c9b 53f7b6b707b 22d7d4a6299 04cbd225529 ff705742990 48614457ac0 a59de73a1ef b27c257ed7c), code head 8a01f4b7fc8.
Conflict in pipeline_minimax_h3.py (~2962): kept t473 tilerow AdaLN gather AND the VSA geometry permute
(row_adaln permuted by vsa_geometry.permute_metadata before _row_indices). This combination is untested on device
(t452 runs predate tilerow AdaLN), so check it first if the e2e looks wrong.

Facts found:
- 4x8 is untraced (trace_denoise is quad-only, pipeline preset ~line 344/578), so a plug wrapping
  ttnn.transformer.vsa_sdpa sees every call of every step, incl. step 4.
- blx01: / has 540 GB free. /var/tmp/fasth3/models has NO MiniMax-H3 now, cache/ is empty, t48 venv is gone
  (t48_keep only). t466/cp466.sh is a ready, gated, throttled copy script (source
  /mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3 -> $F/models/MiniMax-H3);
  t466/build466.sh builds a tree + venv. Reuse both (copy under t476/ names).
- t452/run452.sh arms dense/sla<s>; needs VENV and overlay W rebuilt for this branch.

Leading hypothesis (not yet verified): block-sparse depends on token order, dense does not. If our packed order
or patchify differs from LightX2V (packing.py: [text | cond | audio | video], video frame-major
permute(0,2,4,6,1,3,5,7) = (f, h/2, w/2) raster), the 64/128-row blocks group different tokens, LoRA v0.1 was trained
for LightX2V blocks, and #456's layer-0 check (on OUR order) could not see it. Second hypothesis: selection wrong only at
later layers / steps (stale ids) - replay layer 49 and step 4.

Next steps: (1) CPU: compare our _assembly_indices / patchify / prompt-cap padding with LightX2V packing.py;
(2) stage weights + h3hf cache with a t476 copy of cp466.sh (detached on blx01); (3) build overlay + venv;
(4) replay plug (from t457_sweep.py / t456_dump.py) at 0.85, layers 0/49, steps 1/4, device ids vs LightX2V ids,
per-head PCC vs torch block-sparse (smooth-K, BLKQ128/BLKK64) and dense, scale, mass kept.

# t479 notes (run 1799, split at 3.0 M context)

Branch reset to t476 head cf509c4371f (code head 8a01f4b7fc8); t479 commits on top.

Step 1 done, NO mismatch: our packed order equals LightX2V's.
- LightX2V lightx2v/models/networks/minimax_h3/packing.py: [text | cond | audio | video], patchify permute(0,2,4,6,1,3,5,7)
  (same as ours, packing.py:181); _assembly_indices (pipeline ~1679) builds [text(l_len, real length) | cond | audio | video],
  prompt_cap only sizes the arena source table; build_sla_geometry is identity (raster 64-row tiles, pad tiles appended).
- LightX2V SP (model.py _seq_parallel_pre_process, ulysses_prepost._join_tokens aux_first=True): aux = text prefix,
  main = rest split contiguously; the attention input is cat(aux, main) = original order. Sparse op sees full seq per head.
- SLA config (dmd/minimax_h3_fp8_4step_5090_vae_fp8_sla.json): sage2, sparsity 0.85, BLKQ128/BLKK64 (non-sm90),
  shift 6/3, training_euler, LoRA v0.1 sla alpha 128 strength 1. Same shift/step update as the dense v1.0 config.
- SLA LoRA file header = v1.0's: rank 128, metadata alpha 128 -> scale 1. Not a LoRA-scale issue.
- t452 stills: sla0.85 AND v10sla0.85 (v1.0 LoRA + SLA mode) are both full colored-noise garbage; vsa0.9 keeps the fox;
  slalora (dense attn) is clean; #456: v10sla0.0 ~19 dB (fine). So the fault is the SLA selection/op path at
  sparsity>0, independent of LoRA, and it is catastrophic (not mass loss): look for a later-layer/step or
  text-row/q-block (4-tile) bug, or a cap on selected tiles per query in vsa_sdpa at ~177 of ~1180 K blocks.

Step 2 running: blx01 /var/tmp/fasth3/t479/drv479.sh (detached, pid 111314, log t479/detach/stage479.log, rc stage479.rc):
cp479.sh (copy of t466/cp466.sh -> $F/models/MiniMax-H3, gated, throttled, log t479/cp479.log) then build479.sh
(tree at t479/b from head.tar = git archive cf509c4371f minus tt-project + submodule tars at pinned SHAs, same as #477).
MiniMax-H3 had been deleted from blx01 since #466 (copied 13:09 UTC, gone by 15:13); h3hf cache is empty, so the first
device run is a cold cache build. Venv: reuse /var/tmp/fasth3/t460/python_env (exists; t48 venv gone).

Next: when stage479.rc = 0, adapt t452/run452.sh (W=$F/t479/b, VENV=t460, T409_* knobs) plus t456/plug + replay456.py
for layers 0/49, steps 1/4; device-lint; submit via blx01 broker (t288/blx01-submit.sh). Leave MiniMax-H3 staged.

# t479 run 1810
- Staging: MiniMax-H3 copy done (19:16 UTC). Build failed once at 1136/1446: ld.lld `pthread_create has failed:
  Resource temporarily unavailable` linking tracy-capture (transient; limits are huge, #482 started a build at the
  same time). Resumed with t479/rb479.sh (no re-extract), detached on blx01, log t479/detach/rb479.log.
- SLA LoRA v0.1 holds only to_q/k/v/out + ff LoRA (no proj_l / linear-branch weights), so no SLA linear branch to port.
- Step 3 set up: tt-project/t479/{run479.sh, plug/t479_dump.py, replay479.py, env479.yaml, spec_d1.txt, chain479.sh}.
  chain479.sh (detached on g15blx02) waits for rb479.rc, collect-only, submits d1 (sla0.85, dumps real vsa_sdpa
  calls 0/49/150/199 = step 1 L0/L49, step 4 L0/L49, heads 0-3 of device 0) via t288/blx01-submit.sh, up to 3 tries
  (first run is cold h3hf cache), then runs replay479.py on each dump on blx01 CPU -> replay479.txt.
  Replay per call: (a) device out vs CPU sparse on device ids, (b) device ids vs LightX2V ids overlap, (c) vs dense,
  mass kept by device ids and by LightX2V ids.
- Read: if (a) and (b) hold at all four calls, our SLA path is faithful to LightX2V's selection math and the garbage
  is not a device bug (then check LightX2V's real sparse kernel semantics: sage2 SLA, block sizes, k rounding).
  If (b) drops at L49/step 4: stale or mis-staged selection inputs; if (a) drops: vsa_sdpa kernel at that call.
- chain479 detached on g15blx02 at 12:23 PDT (pid in ttp detach), files in /home/smarton/fasth3/tt-metal/tt-project/state/runs/1810:
  chain479.log, chain479.done (exit code: 0 all dumps, 10 build failed, 15 collect failed, 11 submit), replay479.txt.
  Next: read replay479.txt, write results here, then bisect/fix (step 4) or stop per the spec's stop condition.

# t479 run 1822 (after g15blx02 reboot 16:10 PDT)
- Run 1810 chain: build rc 0; d1 jobs 221/229/231 on blx01 all exit 1, no dumps. Cause (job 231 run.log, line 1893):
  `RuntimeError: [t286] warm deadline 400 s passed in Warming prompt encoder keyframe layouts`. Weight caches load warm
  (10 s); audio decode warm took 6m27s (lengths 7-12 cold, ~60 s each, +2180 programs). Cold JIT, not a code bug:
  compiled kernels persist in /var/tmp/fasth3/cache/tt-metal-cache-h3hf, so each try gets further.
  Earlier blx01 holds while waiting: 3/32 then 8/32 chips off the bus (broker degraded-hold, 19:33-20:28 UTC 10-13,
  not our job; trays owns recovery).
- Relaunched chain479 unchanged at 20:00 PDT 10-13 (detached, run 1822): d1 -> blx01 job 307.
  Files: $TTP_RUN_DIR=state/runs/1822/{chain479.log,chain479.rc,replay479.txt}.
  Next: read replay479.txt. If d1 fails 3x again past audio warm, add a T286 audio-length skip (keep only the 10 s length).
