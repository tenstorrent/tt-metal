# REVIEW_RECORD-352aee85ec92 — silicon authorization for the pin-cycle-7 build

Pin: cc1plus sha256 `87688520648a3a706ff1f739db472858bc3f8204cf71bf1fcfc8333f11ee8872`
Built from: sfpi-gcc `7b4e4d96fb6` (13 lanes: +BM counted-row/pricing, +BQ crosscall-hoist, +BT cgraph-edge TU-scan fix) via sfpi (`scripts/build.sh`; log `~/sfpi-uplift/toolchain-rebuild-pin12d.log`; single-writer verified after a build-overlap incident (12a/12b killed, stage2 wiped); stage2 stamp removed, cc1plus sha VERIFIED CHANGED from pin-10 `2911f0e680e4…`)
Date: `2026-08-18`
Reviewer: orchestrator session (Claude, session a64aef93, operated by nkapre@tenstorrent.com)
Independence: NOT independent — the reviewer spawned and merged all nine lanes. Every lane ran adversarial gates (byte-identity vs same-recipe bases, frozen-FAIL DejaGnu, renamed/varied/near-miss twins, paired CRAQ on the corrected sims; BS additionally ran an internal adversarial review that found and fixed 3 bugs pre-push). An independent re-review should supersede this record.

## Reviewed commits/branches
- sfpi-gcc `aee50405ae3` = pin-10 base f433762fe5b + merges: AY drain-aware
  scheduling (2df2fdc9295), BN descriptor residency (e59ba9f2303,
  conflict-union vs AY compile-verified + focused 271/0), BO IMS repair +
  arbitration (bce914a56cb, .opt Var verified, WP renumbered, 934/0),
  BS const-remat/residency/spill-diag (aee50405ae3, families 49/0 + 6/0),
  BL mop_cfg derivation + M3 (60bc8dc84be) — all off the BD/BG/BC/BF/BH
  pin-10 union.
- tt-metal batch: BL + AY conf branches merged, BN/BS ON entries added
  with fire witnesses, pin #11 values/prose/history/baseline anchors.
  NOT included: lanes BM (counted-row formation + BP pricing fix) and BQ
  (cross-call hoist) — still in flight; they ride pin 12.

## Gates checked
- Full union rvtt.exp at aee50405ae3 (fresh cc1plus + drivers): 3080
  PASS; unexpected-FAIL set byte-identical to the frozen environmental 14
  (zaamo, delay-34602 x3, unused-46063 x2, sfpxloadi).
- Per-lane gates at merge (all green; evidence dirs laneAY/BN/BO/BS/BL
  -evidence-20260818 + SHA256SUMS verified): corpus flags-off identity
  1858-3169/all per lane; ON inventories fully attributed and CRAQ'd.
- Fire witnesses on record for every pin-11 ON-set entry: prgm-const
  (exp capture 17->16 dump), drain-schedule (minmax SFPNOP 12->3),
  planner-residency (where resident=elided, 5-ELF inventory),
  const-remat (ICE-repro-now-compiles), const-residency (12-row designed
  fire). macro-ims deliberately NOT in the ON set (no wired-row fire;
  silicon A/B pending).
- Installed toolchain accepts all six new flags.
- NOT checked: no pin-11 silicon exists yet (this record authorizes the
  first run); expected ON-inventory movements are pre-registered in the
  baseline header and lane evidence (exp ~75.5, minmax drain gain, where
  drain-elision check, BS 12-row fire, sdpa recovered bytes + dedup).

## Pin-12 addendum
- Union rvtt.exp at 7b4e4d96fb6: 3171 PASS, frozen-14 byte-identical.
- INSTALLED-BINARY fire witnesses (new requirement, BT corrective #3):
  exp M3 'allocated PRGM L14' (artifact e0ef2f9c = predicted fired
  bytes), welford 'Formed counted-row record [24,+5): 5 launch sites',
  sigmoid-tree 'hoisted 6 contract materializations' — all compiled with
  cc1plus 352aee85ec92 on the shipping (comdat-1) recipe.
- Pre-registered flips this pin: welford 300-320 vs 325; sigmoid-tree
  27.7-27.8 vs 27.856; exp ~75.5.
