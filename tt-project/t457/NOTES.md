# t457 — ~19 dB floor: vsa_sdpa numerics on real data (blx01)

Branch ttp/t457-..., based on t456 head da96dd1c143 (ff-merged).

## Step 1: per-tile sweep, blx01 job 039 (submitted 12:55 PDT 2026-10-12)
- Spec t457/s-sweep0.txt -> /var/tmp/fasth3/t457/run457.sh (run456.sh with -p t457_sweep), plug t457/plug/t457_sweep.py.
- At 9472-row vsa_sdpa calls 0 and 49 (layer 0 and 49 of step 1), devs 0 and 24, heads 0-13, every q tile:
  PCC of vsa_sdpa vs fp32 over real keys, and PCC of plain bf16 ttnn SDPA (HiFi2, no fp32 acc, exact exp) on
  the same q/k/v vs fp32 over all keys (the dense kernel's bf16 floor on the same data). Ends T457_SWEEP_DONE.
- Log /var/log/tt-device-broker/2026-10-12_195511_039.log; read the `[t457]` lines.
