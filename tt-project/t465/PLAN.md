# t465: next FastH3 Turbo speed steps (plan, from #460)

Written 2026-10-13 PDT. Evidence: #460 (blx01 broker jobs 077-097, branch ttp/t460-integrate-fasth3-turbo-gains-on-one-bran
at dbd6d66048f), #453 (blx01 jobs 995/996), #412, #423/#427, upstream origin/main dc0b7bea9ad.
All numbers are from blx01 with AICLK clamped at 900 MHz: relative only (user #230).

## Where we are (test's own timing table, verbatim)

int 10 s seed 0, blx01 job 080, /var/tmp/fasth3/t460/out_i10s0/run.log:
```
================================================================================
MINIMAX-H3 PERFORMANCE RESULTS
================================================================================
Task: fl2va-turbo
Image Size: 1344x768
Num Frames: 243 @ 24 fps (10.12 s)
Inference Steps: 5 (4 denoise steps)
sequence_length:75449, bucket:76800, waste:1.8%
DiT Configuration: sp=8 axis 1, tp=4 axis 0
Mesh Shape: (4, 8)
Topology: Ring, 2 links
adapter minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
video shift 6.0, audio shift 3.0
--------------------------------------------------------------------------------
Text Encoding             |   0.6944s
VAE Encode                |   0.1297s
Denoising (total)         |  11.9112s
Denoising (per step)      |   2.9778s
VAE Decoding              |   2.1072s
Audio Decoding            |   0.0977s
Total Pipeline            |  14.9818s
================================================================================
```
int 6 s seed 1, blx01 job 085, out_i6s1/run.log:
```
================================================================================
MINIMAX-H3 PERFORMANCE RESULTS
================================================================================
Task: fl2va-turbo
Image Size: 1344x768
Num Frames: 158 @ 24 fps (6.58 s)
Inference Steps: 5 (4 denoise steps)
sequence_length:49965, bucket:52736, waste:5.3%
DiT Configuration: sp=8 axis 1, tp=4 axis 0
Mesh Shape: (4, 8)
Topology: Ring, 2 links
adapter minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
video shift 6.0, audio shift 3.0
--------------------------------------------------------------------------------
Text Encoding             |   0.4911s
VAE Encode                |   0.1668s
Denoising (total)         |   7.3296s
Denoising (per step)      |   1.8324s
VAE Decoding              |   1.2506s
Audio Decoding            |   0.0822s
Total Pipeline            |   9.3528s
================================================================================
```
(6 s seeds 1-4 mean total: 9.600 s; seed 0 decode was cold/noisy.)

## Gap and breakdown

| clip | total | target | gap | needed | text | denoise (per step) | VAE dec | VAE enc + audio + other |
|---|---|---|---|---|---|---|---|---|
| 10 s 1080p | 14.98 s | 7 s | 7.98 s | -53% (2.14x) | 0.69 (5%) | 11.91 (2.93 steady), 80% | 2.11 (14%) | 0.27 (2%) |
| 6 s 1080p | 9.35 s | 5 s | 4.35 s | -47% (1.87x) | 0.49 (5%) | 7.33 (1.75 steady), 78% | 1.25 (13%) | 0.28 (3%) |

- At 10 s the denoise alone (11.9 s) is above the 7 s target. Host gaps are ~0.04 s (#412); trace gives nothing more.
- Attention is ~80% of a 10 s step: 47.0 ms/layer x 50 layers = 2.35 s of 2.93 s (#453 job 995; ring SDPA op 39.0 ms).
  Even with all non-attention work free, 10 s needs attention cut by ~85% (7.98 s / 9.4 s of attention over 4 steps).
- So only attention levers can close the gap. VAE decode, text encode and upstream bit-identical gains are marginal
  (each <=0.6 s). The clamp itself is unmeasured: a full-AICLK run may move both targets, but is not a headline (user #230).

## Ranked next tasks (expected e2e saving, 10 s / 6 s, upper bounds)

1. Ring-joint SDPA LoFi matmul fidelity (upstream #58223 + #57979 + #58032), opt-in knob.  <= 2.6 s / 1.0 s.
2. VSA/SLA fused ring path e2e (fix job 996 stall), Turbo-SLA LoRA at 0.85.  <= 2.0 s / 0.7 s. Reopens VSA per #457 only if it pays.
3. Sparse attention kernel efficiency (work ~ kept tile pairs). <= 4 s / 1.5 s more on top of 2. Depends on 2.
4. FAST_H3_FP8 bf8 matmul subset (to_qkv/to_out/ff1/ff2, PR #59880) on Turbo. <= 1.2 s / 0.4 s. Quality risk.
5. Bit-identical upstream perf bundle (#56023, #55431, #59325, #59349). <= 0.5 s / 0.3 s. Marginal.

Not queued (marginal, <=0.6 s each): VAE decode matmul program_configs (matmuls at 26-52% of peak, 37% of decode),
Qwen3-VL text encoder trace (qwenimage text_encoder.py as prior art). Re-plan after 1 and 2 report.

Upstream: integrate selectively (cherry-pick -x the named commits into task branches off ttp/t460-...), not a full
merge of main (629 commits since the Turbo base c2a4d40104d); #134 already showed the LoFi chain applies cleanly on t48.

Do not repeat: VSA-0.9 two-op/dense-compare (#453), precomputed AdaLN default (#421), full bf8 (#444), v1.2 adapter (#396).

## Box readiness (blx01, checked read-only 2026-10-13)
/var/tmp/fasth3 is 44 GB; t422/b build and t460/python_env are present, but models/MiniMax-H3 and cache/ are empty
(#463 cleanup). The first device task must re-fetch MiniMax-H3 under ionice -c3 nice rsync --bwlimit=100M and keep
the footprint <=150 GB, then warm caches (first job cold). blx03 out (trays #364); g15blx02 paused; exabox only via
self-ending Slurm batch while the Mac tunnel is up.

## Daily video
/home/smarton/fasth3/tt-metal/tt-project/state/t460/mp4/int-10s-s0.mp4 (1920x1080, 243 frames)
still at 5 s: /home/smarton/fasth3/tt-metal/tt-project/state/t460/still_int-10s-s0_t5s.png (clean, no artifacts).
