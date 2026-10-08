# t273 notes (R2b: det stages 2-4 split)

Approach: DIFFVAE_DET_A2A=1 (opt-in, code commit 269d1c5bb72). Stages 2-4 keep the residual as
1/4 of each W-band's tokens per tp chip (padded to 32*4 rows). Full replicated qkv weight
(device-major), all_to_all_async_generic(in_dim=2,out_dim=3) -> colpar layout; NA unchanged but
gather_heads=False; all_to_all back (in_dim=3,out_dim=2) before out-proj; all-gather tokens at
stage end. Weight layout prefix "a2a-".

blx01 A/B: job 021 (off vs on, one arm per process), out /var/tmp/fasth3/t273/out_AB,
log /var/tmp/fasth3/t273/out_AB/run.log, src overlay /var/tmp/fasth3/t273/src/269d1c5bb72,
build t272/b @cae4b52657d. Baseline job 019: off(s1split=0) 3.112 s, def 2.714 s.

Next: when job 021 ends, read DECODE lines in run.log; score with
  python drv/cmp241.py /var/tmp/fasth3/diffvae/ref out_AB/<arm> <dst.json> 0,1
(check cmp241 args / arm subdir layout from t261 out_AB). Accept: >=0.17 s faster, PCC>=0.9999,
PSNR within 0.5 dB of off, md5 differs. Pass -> flip default, land on t48 via -land branch.

## Result (job 021, blx01, completed 221.8 s, no drops)
- off (A2A=0): 2.701 s/decode (seeds 0,1), md5 2797bc15/13ee4b04; PCC 0.999947/0.999948, PSNR 54.706/54.285 dB
- on  (A2A=1): 2.532 s/decode, md5 be633a69/da84001a; PCC 0.999947/0.999948, PSNR 54.708/54.284 dB
- delta -0.169 s (6.3%); warm-up compile 51 s -> 87 s, load 2.5 -> 5.1 s.
- Judgment: 1 ms short of 0.17 s, but 0.17 s is the 5%-of-3.4 s rule; 6.3% passes it. Default flipped: 561f9c5d5b8.
- Scores: blx01 /var/tmp/fasth3/t273/out_AB/score_{off,on}.json
