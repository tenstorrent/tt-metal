You are running an independent model bring-up in this repo checkout (own branch, own worktree).
Long task — likely many hours including real device time, a real weights download, and CPU-bound
golden-trace generation. Work autonomously; only stop for a genuine blocker (missing files, real
hardware fault, something the recipe is silent/wrong about that you can't safely resolve) — not for
routine decisions.

## Mission

Follow `models/demos/common/prefill/docs/MODEL_BRINGUP_RECIPE.md` (read it in full — it's already
final, including the mesh-coverage table and coding conventions in §2.3/§4) to bring up Qwen3.8-27B
prefill at `models/demos/qwen_3_8_27b_d_p`.

**Keep `qwen_3_8_27b.spec.json` as-is** (the binding spec, per recipe §1). Delete everything else in
that directory and rebuild from scratch, genuinely independently: explore per §2 as if this were a
first attempt. **Do not consult, diff against, or take shortcuts from `models/demos/blackhole/qwen36`**
— a complete Qwen implementation already exists there, and this run must not look at it, or at any
other existing Qwen package in this repo. This run stands on its own.

**Pay special attention to Gated DeltaNet.** Unlike a plain-attention model, this architecture's
Gated DeltaNet layers are a recurrent/chunked-scan mechanism, not standard SDPA — get the reference
(HF `transformers`) semantics exactly right before writing the TT module, and give it more PCC
scrutiny than a typical block: this is the part most likely to hide a subtle correctness bug behind a
plausible-looking number.

## Practical setup

- Point `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` at this worktree's root, `LD_LIBRARY_PATH` at
  `$TT_METAL_HOME/build/lib`. Verify with `models/demos/common/prefill/tools/check_runtime_env.sh`
  before any device work.
- No local weights exist yet — fetch `Qwen/Qwen3.8-27B` from the Hugging Face Hub yourself (the
  `hf_repo` field in the spec). Check whether HF credentials are already configured on this machine
  before assuming anonymous access works. This is a large, real download: treat it like the golden
  trace generation — `setsid`, backgrounded, NO wrapping `timeout` that could kill a multi-GB transfer
  partway. Cache it somewhere persistent so a restart doesn't re-download.
- Device tests always through `scripts/run_safe_pytest.sh`, never bare `pytest` (flocks device
  access, resets on hang).
- Stage order and gates per the recipe: E -> D1 -> D2 -> D3 -> M1 -> M2 -> M3 -> P1 -> P2. Write
  `bringup_log.jsonl` per §7 as you go; satisfy §8 for whichever shapes this pod lets you grade.

## Git

No checkpoint commits along the way — work uncommitted until the end. At the end of the run —
**whether it finished or you stopped because of an irrecoverable blocker** — make exactly ONE commit
summarizing what was built, which shapes are graded vs. skip-only, and where PCC numbers /
bringup_log entries live. If you stopped on a blocker, say so plainly in that commit message rather
than presenting it as done. Than you must push that commit to remote!!!

## Report

Stage reached, PCC tables for whichever shapes this pod actually graded, confirmation the rest skip
cleanly, final commit hash, and — separately — how Gated DeltaNet went specifically: what PCC it hit,
what was tricky about it, and any judgment call worth a human's attention.
