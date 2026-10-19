# Fused FP32 K-chunk prototype for ACCURATE/BALANCED (2026-10-16)
Goal: real speedup of fused FP32 chunk (ref max, no S pack/sub, redo fallback) vs ACCURATE/BALANCED. Scratch only.
Base: cglagovich/retire-int-p5 5dd8d6a0fa1. Bound (resident Q256/K512): 1.22-1.25x BALANCED.
## Log
- 17:32 bh-qbae-16 still has no clang-20/ninja; bh-38 reservation gone. Fresh IRD reserve (yyzc-wh-03, p150) -> bh-30
  (sel 1, job 141295, 14h): toolchain OK (clang-20 in /usr/lib/llvm-20, cmake 4.2.3, ninja, ULFM MPI). ssh alias bh-30.
- bh-30: worktree /localdev/cglagovich/ffp (detached 5dd8d6a0fa1, from /localdev/cglagovich/tt-metal), work dir
  /localdev/cglagovich/ffpw (env.sh, build.sh, logs/). Base build started 21:40 (UTC container clock).
- Local scratch worktree: /Users/cglagovich/.claude/jobs/873667bb/tmp/ffp32 branch ffp32-proto (off 5dd8d6a0fa1).
- 2026-10-19: agent was terminated (auth) after design reading; bh-30 reservation expired. Re-reserving p150 (~/ffp_reserve2.log on yyzc-wh-03).
## Design (decided before termination)
- m_ref = exact TF32 row max of the Q chunk's first (reduce-path) chunk. Fused QK step: extra inner step
  [Q|M2]x[K^T;-e0-e1] at 1 phase, INNER_HALF: M2 (BF16, CB30, built by UNPACK scalar code) col0 = hi (7 sig bits),
  col1 = lo = m - hi (<=4 bits for TF32 m) -> DEST gets s - m exactly in FP32. neg tile CB29 (K fmt BF16) cols 0,1 = -1.
- Exp: FP32 grid trick has a 31-octave window [-21.46, +0.72] nat -> no headroom. Option: 9-fraction-bit grid
  (a=512 log2e scale, shift 14) = 63 octaves: window [-21.5, +22.9] nat, grid error +-0.068% (vs +-0.034%). Applied to
  all chunks (same K constant). Redo when chunk row sum (P.1, scratch CB14) >= 2^24 (UNPACK scans 32 words).
- Denominator kept as P.1 on no-mop matmul (PHASES=1 BALANCED / 3 ACCURATE), folded into l by L1 acc.
- O,l in place (pop+swap banks like inplace_numerator path); segment keeps prev.max when fused.
- 10-19 09:31: bh-30 now draining (NHC card test failed); yyzc-wh-03 /home full (log in /tmp/ffp_reserve_cg.log).
  Reserved qb -> bh-31 (aus). ssh alias bh-31 added. Need fresh clone+build there.
- Prototype committed locally: ffp32-proto 208213093c2 (recipe_fused_fp32.hpp new; recipe_streaming.hpp dispatch/segment/
  FFP_QK_FID knob; recipe_sfpu.hpp grid9 (FFP_GRID9/FFP_GRID10) + FFP_QUAD; host sdpa_recipe.cpp env TT_SDPA_FFP32=1 adds
  CB29/30 + SDPA_RECIPE_FFP32 (BF16 K only), env TT_SDPA_EXTRA_DEFINES="A;B=v"). Patch on bh-31 /localdev/cglagovich/ffp.patch.
- bh-31: tree /localdev/cglagovich/ffp (worktree of /localdev/cglagovich/tt-metal, 5dd8d6a0fa1 + git am patch), work dir
  /localdev/cglagovich/ffpw (env.sh, build.sh, logs/status.log, bf/rbench.py with FFP=1 option, bf/kernels resident rd/wr).
  Build launched ~09:45. Default env => baseline (same binary).
- Test plan: rbench C/D +-FFP; unit large_logits/offset_rows; nightly rising_max (28/25 nat above first max -> redo path);
  bfrun front/single/full with TT_SDPA_FFP32=1.
- 13:43 bh-31 build OK (ccache, 8 min); venv needed XDG_DATA_HOME/UV_PYTHON_INSTALL_DIR on /localdev (/home full).
- First resident Q256/K512 D128 (rbench, worked first try): BALANCED 1.211 (l2 .429) -> fused 1.244 (+2.7%, l2 .364);
  ACCURATE 0.876 (.180) -> fused 1.004 (+14.6%, .184). BALANCED far below bound -> knockout batch.
- GOTCHA: JIT cache keys on defines, not source: rm -rf ffpw/jit/rb after editing kernel sources.
- Knockouts (fused BALANCED 1.244): KO fold 1.320 (+6%: per-group copy/init switch), KO mstep 1.269, KO exp 1.293,
  KO den 1.252, KO PV 1.430, KO PV+exp 1.627, KO all(PV,exp,den,fold) 1.901 -> MATH-bound (QK HiFi4 dominates).
- Fix: denominator packs its DEST tile twice (scratch CB14 for the check + L1-acc onto l): no fold. Redo then does
  l = (l - bad) * c + new (cancellation error ~ulp(bad)*c, negligible).
- Resident Q256/K512 D128 after fix (TF/s, l2): BALANCED fused 1.320 (.364) [+9.0%]; one-piece 1.313;
  QK HiFi3 fused 1.400 (.363) [+15.6% vs BAL]; QK3+quad fused 1.479 (.409) [+22.1% vs BAL, +13.6% vs unfused q3q 1.302];
  ACCURATE fused 1.078 (.184) [+23% vs 0.876]. q3q fused hits the notes' fused bound (1.480) ~100%.
- Correctness: per-group FOLD variant passes unit large_logits (6) + offset_rows (2). Dual-pack (l += bad, redo
  unadds) FAILS large_logits (non-finite) and ACCURATE rising_max even with clamping to 1.27e30 and FORCE_REDO ->
  abandoned (kept as FFP_DUAL knob). Forced redo on resident bench: l2 .380 (works), 0.50 TF/s.
- New default: late fold (STANDARD's scheme): non-last chunks fold all unchanged groups' sums once at chunk end
  (4 tiles per DEST pass); last chunk folds per group before normalization.
- Late fold: unit large_logits/offset_rows/accuracy 10/10 pass; resident BALANCED 1.301, ACCURATE 1.065, q3q 1.454.
- rising_max (nightly) failed 4/4 for every sum mode; FORCE_REDO passes. Per-tile-row diag (bf/rise.py): random redone
  groups 99% wrong -> RACE: after recipe_pack_to_unpack_barrier the UNPACK *RISC* read the new max from L1 for the M2
  rebuild before the pack landed (the SEMWAIT stalls the unpacker, not the RISC). Fix: UNPACK tensix_sync() first.
  (SFPSTOCHRND fp32->int16 clamps at +-32767 per ISA doc, so grid overflow saturates; detection via sums is sound.)
- After race fix: nightly rising_max balanced+accurate 4/4 pass (q3q 2/2); per-row diag all rows <=0.44%.
- KERNEL SIZE (FFP_PROGSIZE log in program.cpp, env in ffpw/env.sh): dense subtile_tails BAL/ACC base ~56.3K -> fused
  ~65.3-67.1K (+9-11 KB; first try hit 70672 > 70656 until redo lambda -Os); psize dense set (D96 Q160/K512 etc) base max
  69536 -> fused max 65920 (fits; inlining changed). ACCURATE causal (KRANGE): base 61.4K -> fused 90.7K (overflow by
  20 KB); whole fused chunk -Os: 84.8K (still +14 KB over) and -Os costs 10% speed (1.179 vs 1.301) -> off by default.
- BALANCED key-range (nightly causal/sliding/chunked): fused programs 92.8-94.4K -> overflow (~24 KB over). Key-range
  fused is NOT measurable without size work; dense non-causal is.
- Launched resident grid sweep (Q128-320 x K128-512, D128+D64; bal/ffp/q3q/ffpq3q/acc/ffpacc), tags g_*.
- Resident grid (geo over Q128-320 x K128-512; D128 n=9, Q320/K512 does not fit resident L1):
  D128 vs BAL: ffp 0.992 (min .874 @K128, max 1.128 @K512), q3q 1.072, ffpq3q 1.111 (max 1.251); ACC: ffp 1.194 vs ACC.
  D64 vs BAL: ffp 1.089 (.939-1.193), q3q 1.073, ffpq3q 1.208 (1.02-1.32); ACC ffp 1.304 vs ACC.
  K512 rows: D128 BAL 1.211->1.301 (+7.4%), q3q-fused 1.454 (+20%), ACC .876->1.065 (+21.6%);
  D64 Q256/K512: BAL .703->.827 (+18%), q3q-fused .919 (+31%), ACC .549->.747 (+36%).
  Fused LOSES at K128/K256 for BALANCED D128 (per-group fixed cost: inits, den, check): K256 ~-6%, K128 -6..-13%.
- One-piece (FFP_ONE_PIECE) at short K: 256x256 1.103 (vs 2-piece 1.077, BAL 1.143) -> short K still loses.
- FULL GRID (bh-31 p150, ms; base reproduces qbae notes): 
  BAL      wan 37.88 h3 24.96 causal8k 1.73 d64 5.62
  BAL-ffp  wan 33.39 (1.134x) h3 21.42 (1.165x) causal n/a (kernel config overflow) d64 4.38 (1.283x)
  q3q      wan 35.61 h3 23.48 causal 1.60 d64 5.28
  q3q-ffp  wan 30.24 (1.253x vs BAL, 1.178x vs q3q) h3 19.71 (1.266x/1.191x) causal n/a d64 3.99 (1.41x/1.32x)
  ACC      wan 50.24 h3 32.95 causal 2.22 d64 6.74
  ACC-ffp  wan 39.85 (1.261x) h3 25.56 (1.289x) causal n/a d64 4.81 (1.40x)
- 18-CASE SUITE (worst/median rel-L2 %, full grid Q256/K512): BAL .483/.382 -> fused .430/.318; q3q .599/.345 ->
  fused .449/.324; ACC .420/.180 -> fused .407/.183. Fused is as accurate or better (S - m exact in FP32 DEST, no TF32
  re-read of S; fewer rescales).
- Captured bound: resident Q256/K512 q3q-fused 1.454 vs fused bound 1.480 (notes) -> (1.454-1.211)/(1.480-1.211)= 90%.
## Final artifacts
- Local branch ffp32-proto d6ce98a2b9b (3 commits on 5dd8d6a0fa1) in /Users/cglagovich/.claude/jobs/873667bb/tmp/ffp32;
  patches + scripts + res/ copied to lp-headroom/retire/fused-fp32/. bh-31 tree /localdev/cglagovich/ffp (built),
  work dir /localdev/cglagovich/ffpw. bh-31 reservation (qb, aus) still held (14h from 09:31 UTC-ish 10-19).
