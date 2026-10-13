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
