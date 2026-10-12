# t456: SLA mode noise, real-layer replay (continues #455)

Branch ttp/t456-continue-455-sla-mode-fix-real-layer-rep (from pr_branch ttp/t454-... head ddb510fedad).

## Step 1 (CPU arg comparison): no mismatch
Model path and t455 device test build the stage with the same args and call vsa_sdpa with the same kwargs;
SLA geometry permutation is identity; rung = padded_len 75776 (shared with VSA mode, which works).

## Step 2: real-layer dump, blx01 job 032 (submitted 19:23 PDT 2026-10-12, queued behind ltx-host 031)
- spec s-dump.txt, run456.sh (= t452/run452.sh + `-p t456_dump`, outputs /var/tmp/fasth3/t456/out_s0-sla0.85/),
  plug/t456_dump.py wraps ttnn.transformer.vsa_sdpa: first call with 9472 local q rows (the 75776 rung, step 1,
  untraced) saves device 0's q, gathered k/v (4 heads), raw ids, block counts, output, kwargs to
  out_s0-sla0.85/dump/layer0.pt, then raises T456_DUMP_DONE (the job is expected to FAIL with that).
- Next: `scp g15blx01:/var/tmp/fasth3/t456/out_s0-sla0.85/dump/layer0.pt /tmp/t456/` then
  `python_env/bin/python tt-project/t456/replay456.py /tmp/t456/layer0.pt` (kernel, selection, sparsity checks).

## Step 2 result: job 032 ended as designed (T456_DUMP_DONE, 199.6 s, no drops); replay of layer0.pt (device 0, heads 0-3)
- q [1,14,9472,128], gathered k/v [1,14,75776,128], ids [1,14,148,176] (0..1178, no dups), list_len 176.
- device out vs CPU sparse attention on the device's own ids: PCC mean 0.9997, min 0.9956 (kernel correct).
- device ids vs LightX2V get_block_map (BLKQ 128): overlap mean 0.9865, min 0.9545.
- vs dense: device 0.9897 mean / 0.9038 min; LightX2V sparse 0.9904 / 0.9051 (same). Softmax mass kept mean 0.59.
- LightX2V upstream (minimax_h3 transformer_infer + ulysses_attn): whole packed sequence goes sparse, aux tokens
  only reordered (aux_first), no step/layer gating, no linear branch. 4x8 preset has no denoise trace.
=> layer 0 on real data is faithful to LightX2V. Not yet explained: the 12 dB e2e result.

## Step 2b: sparsity-0 SLA arm (all 1179 blocks selected), blx01, spec s0-v10sla0.0.txt
v1.0 LoRA + MINIMAX_H3_VSA_MODE=sla + sparsity 0.0 (k=1179), T456_ROWS=-1 (no dump), latents saved.
Compare with /var/tmp/fasth3/t452/out_s0-dense via t452/psnr452.py. ~dense PSNR => SLA plumbing is fine and the
noise is the sparse approximation; ~12 dB => plumbing bug (75776 rung, row_source pad rows, stage) to bisect.
