# DiffVAE notes

## 2026-10-08 #267 (review #266 fix): notes-only close, no t48 landing
(copied by #269 from local branch ttp/t267-diffvae-next-lever-profile-stage-5-and-t @ 3690af9affd)
- #253 is closed as done/superseded. Record: origin/ttp/t253-diffvae-next-lever-profile-stage-5-and-t @ 20c4b6f8f96.
- Not pushed to ttp/t48-ltx25-integrated. Checked: t48 head 8a99339c424 does not contain 47e19132826 (only the t253 branch does), and no ttp push was run.
- 47e19132826 (DIFFVAE_NA_BF8, opt-in) stays unlanded: no A/B, no test, written before the #260 ring default.
- If R1b (#263) wants BF8 NA operands: cherry-pick only 47e19132826 onto current t48, one-arm-per-process A/B on blx01 against the #214 refs, land only if PCC >= 0.9999, PSNR within ~0.5 dB of 55 dB, gain >= 0.17 s.
