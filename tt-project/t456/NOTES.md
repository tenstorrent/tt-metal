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
