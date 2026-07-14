# W1 verdict — the one experiment that convicts or clears the dedup

## Question
Did `8016104c27e` (LTX_DEDUP_GATE_GATHER, the −248 ms W1 win) break prompt adherence at 1080p?

## Why it is the prime suspect
- Landed 01:53; the 1080p failure was measured at 07:55 — it is in the window.
- Only two default-on `models/` changes predate the failure: W1 and the all-zero guard (`be06ca2c222`).
- Both cache commits are EXONERATED by timeline (08:43/08:57, 32+ min AFTER the failure).
- All C++ changes are exonerated by construction: the bisect stages only `models/` against the tip
  `_ttnn.so`, so they were already live in the run that scored 30.85 WITH a guitar.
- W1 was already in when 704p rendered the guitar correctly (07:10) — so it does not break low token
  counts, which matches the token-count-dependent signal (880 tok 32.13 / 2040 25.81 / 9690 18.71).

## Design
Two arms, identical but for one env flag, seed 10, 1088x1920x145, LTX_QUALITY=high:
- `env_w1_on.yaml`   LTX_DEDUP_GATE_GATHER=1  (shipping default — expect no guitar, CLIP ~19)
- `env_w1_off.yaml`  LTX_DEDUP_GATE_GATHER=0  (W1 disabled — if the guitar returns, W1 is the bug)

Score CLIP on the HOST from the dumped frames, and LOOK AT THE FRAME. CLIP alone is not enough: a
bisect step scored 27.43 (below the 28.0 gate) while clearly showing the guitar — a speckle artifact
was depressing it. The guitar is the ground truth; CLIP is the corroboration.

## THE CAVEAT THAT MUST NOT BE FORGOTTEN
Every prior CLIP number (good end 30.85, bad tip 18.71, all bisect steps) was measured at
**LTX_TRACED=0**. These arms run **LTX_TRACED=1** (traced denoise is fast, and with LTX_VIDEO_ONLY=1
there is no audio decode, so no cold-vocoder build — the tax that made untraced look attractive).

"Traced is numerically identical to untraced by design" is an ASSERTION, not a measurement. So:
- The A/B is **internally valid** — both arms are traced, so the DELTA convicts or clears W1.
- The **absolute** CLIP is NOT directly comparable to the 18.71/30.85 baselines until traced==untraced
  is verified once. Do not report "CLIP went 18.7 -> 31" across a traced boundary as if it were one scale.
- If budget allows, add a third arm (dedup=1, LTX_TRACED=0) to tie the two scales together. One run.

## Run it (no timeout_sec — leave the broker default)
LTX_TRACED=1 + LTX_VIDEO_ONLY=1 is the config that should have been used from the start: the untraced
denoise is ~40x slower on-device and the eager vocoder costs ~172 s/run, for audio nobody scores.
