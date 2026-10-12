# t457 — ~19 dB floor: vsa_sdpa numerics on real data (blx01)

Branch ttp/t457-..., based on t456 head da96dd1c143 (ff-merged).

## Step 1: per-tile sweep, blx01 job 039 (submitted 12:55 PDT 2026-10-12)
- Spec t457/s-sweep0.txt -> /var/tmp/fasth3/t457/run457.sh (run456.sh with -p t457_sweep), plug t457/plug/t457_sweep.py.
- At 9472-row vsa_sdpa calls 0 and 49 (layer 0 and 49 of step 1), devs 0 and 24, heads 0-13, every q tile:
  PCC of vsa_sdpa vs fp32 over real keys, and PCC of plain bf16 ttnn SDPA (HiFi2, no fp32 acc, exact exp) on
  the same q/k/v vs fp32 over all keys (the dense kernel's bf16 floor on the same data). Ends T457_SWEEP_DONE.
- Log /var/log/tt-device-broker/2026-10-12_195511_039.log; read the `[t457]` lines.

## Step 1 result: job 039 PASS as designed (12:55-13:01 PDT, 330 s, ended T457_SWEEP_DONE, no drops)
- Call 0 (layer 0): 2072 tiles; vsa<0.999: 304, sdpa<0.999: 331; mean PCC vsa 0.99948, sdpa 0.99946.
  Worst vsa tiles are heads 0/1 (dev 0 h0 t6 vsa 0.99287 rel 0.145 vs sdpa 0.99866 rel 0.052; dev 24 h0 t0
  0.99340/0.99871). Caveat: the sdpa reference includes the 327 pad keys, vsa's does not.
- Call 49 (layer 49): vsa<0.999: 1082, sdpa<0.999: 1118; mean 0.99689 vs 0.99658. Worst tiles identical for both
  kernels (dev 0 h10 t0 0.86910/0.86913, rel 0.504 both; dev 24 h5 t1 0.87944/0.87921; dev 0 h9 t2 0.89745/0.89742).
- Verdict: vsa_sdpa per-call numerics sit at the dense bf16 SDPA floor (HiFi2, no fp32 acc). No kernel bug, so the
  spec's step 2 (fix the kernel config) does not trigger. The dense e2e path masks pads via logical_n, like vsa.
- Hypothesis: ~19 dB is the trajectory floor of this 4-step model: any equal-precision attention substitution
  drifts to ~19 dB over 200 layer calls. Then #453's 19.1 dB says nothing about "dense-trained weights".

## Step 1b: noise-floor control, blx01 (spec s-k256.txt)
- Dense v1.0 arm, seed 0, with plug t457_k256 (exp-ring SDPA k_chunk capped 256 instead of 512: same precision,
  different accumulation order). Compare: python t452/psnr452.py /var/tmp/fasth3/t452/out_s0-dense
  /var/tmp/fasth3/t457/out_s0k256-dense (on blx01, t48 venv). ~19 dB => floor confirmed; >>19 dB => vsa has a
  systematic difference beyond per-call numerics (next suspect: lazy max / bf16 O L1-acc on heads 0/1).
