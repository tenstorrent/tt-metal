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
- Submitted blx01 job 041 at 13:03 PDT (log /var/log/tt-device-broker/2026-10-12_200329_041.log). Next on wake:
  check `[t457] exp sdpa` line shows k_chunk 256, PASS in out_s0k256-dense, run psnr452.py, grab a still, then
  decide: no kernel fix => step-4 A/B not applicable (record), publish notes via ttp push --own --detach? (notes
  stay off the push branch: push only if a code commit exists), write result.json.

## Step 1b result: job 041 was a no-op (13:03-13:07 PDT, PASS, 244 s, no drops, 900 MHz clamp)
- No `[t457]` line: Turbo 4x8 runs SP 8, so `use_exp_ring_sdpa` is False and dense attention takes
  ring_joint_sdpa with `_sdpa_program_config(ring=True)` (q 256, k 512 default for this length). The plug patched
  the exp-ring config only. psnr452: out_s0k256-dense vs out_s0-dense = inf dB, latent PCC 1.0000.
- Useful side fact: the dense Turbo pipeline is bit-deterministic run to run on blx01 (jobs 008 and 041).
- Fixed plug t457_k256.py now wraps `_sdpa_program_config` and caps k_chunk at 256 (logs `[t457] sdpa ... k_chunk 512 -> 256`).

## Step 1c: corrected control, blx01 job 042 (submitted 13:11 PDT, spec s-k256b.txt, TIMEOUT 420 = 244 s +50% + JIT)
- Log /var/log/tt-device-broker/2026-10-12_201118_042.log; output /var/tmp/fasth3/t457/out_s0k256b-dense.
- Next on wake: check the `[t457] sdpa` line and PASS, then on blx01:
  /var/tmp/fasth3/t48/python_env/bin/python -I /var/tmp/fasth3/t452/psnr452.py /var/tmp/fasth3/t452/out_s0-dense /var/tmp/fasth3/t457/out_s0k256b-dense
  ~19 dB => floor confirmed (no kernel fix; step-4 A/B not applicable). >>19 dB => vsa has a systematic error beyond
  per-call numerics; next suspects: lazy max / bf16 O L1-acc on heads 0/1.

## Step 1c result: job 042 PASS (13:11-13:15 PDT, no drops, 900 MHz clamp; `[t457] sdpa ... k_chunk 512 -> 256` seen)
- out_s0k256b-dense vs out_s0-dense: Y PSNR 23.00 dB (per-frame min 19.67, median 23.08), yuv 24.61 dB, latent PCC 0.9628.
- So an equal-precision reorder of the dense kernel alone drifts to ~23 dB: the trajectory is very sensitive.
  SLA sparsity 0 (vsa_sdpa) is 18.98 dB / PCC 0.9222, ~4 dB and 2x (1-PCC) further. Part of the "floor" is
  trajectory sensitivity, but vsa adds more drift than a reorder does.
- vsa_sdpa config already matches the dense ring path: HiFi2, exact exp, fp32_dest_acc off, bf16 intermediates
  (vsa_sdpa.cpp defaults). fp32_dest_acc cannot be turned on for the streaming kernel (Sqt 2 x vDHt 4 = 8 > 4 DEST).
- Suspect: a uniform scale bias in O (bf16 O accumulation over ~1179 visits per row absorbing small increments;
  the row sum is exact via the int64 sum service, O is not). PCC is blind to a uniform scale.

## Step 1d: scale-bias check, blx01 job 043 (submitted 13:19 PDT, spec s-bias0.txt, TIMEOUT 540 = 330 s +50% + margin)
- Plug t457_sweep.py now also prints per head `bias: vsa scale <o,ref>/<ref,ref> rel ... | sdpa scale ... rel ...`
  and a `bias summary` line per call. Log /var/log/tt-device-broker/2026-10-12_201928_043.log.
- Reading: vsa scale clearly < 1 (e.g. <0.995) while sdpa ~1.000 => O attenuation in vsa's accumulation (kernel fix
  target: accumulate O in fp32/L1 or rescale). Both ~1 and similar rel => no per-call bias; the 4 dB is the
  different kernel's (equal-size) rounding noise, and no kernel fix is warranted.

## Step 1d result: job 043 PASS (13:19-13:26 PDT, ended T457_SWEEP_DONE, no drops, 900 MHz clamp)
- Per-head scale <o,ref>/<ref,ref> vs fp32 over the real keys, 28 heads (devs 0, 24) per call:
  - call 0: vsa mean 0.9883 (min 0.9665), rel mean 0.0433 | dense bf16 sdpa mean 1.0104 (min 0.9838), rel mean 0.0399
  - call 49: vsa mean 0.9928 (min 0.9523), rel mean 0.0679 | sdpa mean 0.9984 (min 0.9609), rel mean 0.0704
- Both kernels scatter +-2-4% per head with equal rel error; vsa leans ~1% low, sdpa ~1% high at layer 0, and both
  are ~1 at layer 49. No vsa-specific O attenuation stands out from the dense kernel's own bf16 noise.
- Stills at 5 s (tt-project/t457/still_dense_k256_vsa0.jpg: dense | k256 reorder | vsa sparsity 0): all sharp, same
  fox and scene; vsa0 shows only pose/position drift, with no blur or ghosting.

## Conclusion (13:30 PDT)
- The ~19 dB "floor" is trajectory sensitivity, not a vsa_sdpa bug: per call, vsa matches the dense bf16 SDPA floor
  (PCC, worst tiles, scale), and a plain equal-precision reorder of the dense kernel already drifts to 23 dB.
  A different kernel with its own tiling and pad handling drifting 4 dB further is in the same class.
- No kernel fix, so the spec's step 2 (config fix plus CPU test) and step 4 (fix off/on A/B) do not apply.
- #453: PSNR vs dense cannot separate "sparsity hurts" from this floor, since VSA-0.9 (19.1 dB) equals sparsity 0
  (18.98 dB). Its blur/ghosting claim needs a visual and VBench check against a sparsity-0 vsa arm on 5 seeds.
