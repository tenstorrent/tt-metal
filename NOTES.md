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
