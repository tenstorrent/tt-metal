# REVIEW_RECORD-2911f0e680e4 — silicon authorization for the pin-cycle-6 build

Pin: cc1plus sha256 `2911f0e680e4a22c9f2e494f6b902ee7b14b9a9bbf5170d5b94c3e6c1fcb7fe4`
Built from: sfpi-gcc `f433762fe5b` via sfpi `HEAD` (`scripts/build.sh`; log `~/sfpi-uplift/toolchain-rebuild-pin10b.log`; stage2 stamp removed, cc1plus sha VERIFIED CHANGED from pin-9 `ac81ede3827d…`; a first rebuild attempt failed on a merge-lost riscv.opt record separator — fixed as f433762fe5b, both rebuilds then green)
Date: `2026-08-18`
Reviewer: orchestrator session (Claude, session a64aef93, operated by nkapre@tenstorrent.com)
Independence: NOT independent — the reviewer spawned and merged all five lanes. Each lane ran adversarial gates (byte-identity vs same-recipe bases, frozen-FAIL DejaGnu, renamed/varied/near-miss twins, paired CRAQ on the corrected sim); the orchestrator verified evidence manifests, charter greps, and scopes at merge. An independent re-review should supersede this record.

## Reviewed commits/branches
- sfpi-gcc `f433762fe5b` = pin-9 base `69a885b9f67` + merges:
  BD `agent/raw-boundary-decode` (033011a5267 merge), BG `agent/exp-win`
  (eeb0e5297b3), BC `agent/laneBC-prgm-const-scan-retire` (580cbaf96ce),
  BF `agent/welford-win` (44ee0ef8af9), BH `agent/mulint32-win`
  (be2edcd512e), + the riscv.opt separator fix (f433762fe5b).
- tt-metal `0fa5b7f87e`-era batch: AZ 87-row expansion (eed9c37961), BB
  welford clean rewrite (ef5b5fe8ee), BC audited lbs, BG exp restructure
  (coupled), BF sweep wiring, BH mul_int semantic body + row, ON-set
  entries with fire witnesses (ccmask, interlock-schedule,
  transp-involution, replay-exec-record).

## Gates checked
- Union DejaGnu at f433762fe5b (fresh cc1plus + xg++/xgcc): 2814 PASS,
  unexpected-FAIL set byte-identical to the frozen environmental 14.
- Per-lane gates at merge review (all green; details in each lane's
  evidence dir laneB{C,D,F,G,H}-evidence-20260818 + SHA256SUMS verified):
  corpus flags-off byte-identity 3156-3160/all per lane; ON changed-rows
  inventories fully attributed and CRAQ'd (BG 42/42, BF full matrix,
  BH 21/21, BC pairs); LLK-pristine R7 GREEN throughout.
- Fire witnesses on record for every new ON-set flag (the pin-10 rule):
  ccmask == exactly the exp node; interlock 18-op set CRAQ'd;
  transp-involution welford dump (parks eliminated); replay-exec-record
  sigmoid 33->32 shape.
- Installed-toolchain sanity: all four new flags accepted by the
  installed driver+cc1plus.
- NOT checked: no pin-10 silicon exists yet (this record authorizes the
  first run — the 87-row expanded nightly); welford/mulint32-fresh first
  measurements are pre-registered predictions, not bookings; two lanes
  disclosed brief shared-toolchain clobber incidents (both restored and
  sha-verified; enforcement follow-up: hardlink-farm write protection).
