---
name: tt-moe-bringup
description: Entry map for bringing up a new MoE LLM end to end on a Tenstorrent Blackhole Galaxy (4x8 mesh, 32 chips). Use first, when starting or resuming a bring-up, to pick the stage, the track and the next skill to open.
---

# MoE bring-up map (Blackhole Galaxy 4x8)

Last validated against commit: <none yet>

Goal: a new MoE model runs prefill -> paged KV -> traced decode on 32 chips with CPU-reference-matching output, then gets faster
(perf), gets speculative decode, and is served. Source of this procedure: the DeepSeek-V4.1-Flash bring-up
(`models/demos/blackhole/deepseek_v41_flash`, branch `ssinghal/dsv4p1`). Evidence for every rule: `bringup/GUIDE_dsv4_lessons.md`.

## First five minutes (every session, also after a restart)
1. Open skill `tt-bringup-log`; read `bringup/<model>/LOG.md` FRESH entries only. Stale entries are hypotheses.
2. `git log --oneline -15`, `git worktree list`, `git branch --list 'ssinghal/<model>*'`: what exists already.
3. Find your track below, check its dependencies are merged, open the listed skill.
4. Never work in a tree another agent or a running job uses; make your own worktree (skill `tt-lab-ops`, `tt-multiagent`).

## Stages, tracks, dependencies, merge gates
Branch layout: bring-up branch `ssinghal/<model>` (integration); one branch + worktree per track `ssinghal/<model>-<track>`.

| # | Track | Needs | Open skill | Merge gate (all must hold, numbers in the log with commit + log path) |
|---|---|---|---|---|
| 0 | Reference + dumps | HF code/config | tt-model-reference | single-layer oracle per layer type matches HF; short CPU chain (4-8 layers or one of each type) matches HF; no full-model CPU run needed (stream layer by layer; host RAM 566 GB); per-layer dumps for a real prompt exist, hashed in LOG |
| 1 | Attention module(s), one per layer type | 0 | tt-moe-correctness (PCC section) | each layer type PCC >= 0.999 vs dump on real activations, several shapes (S, batch) incl. the extremes |
| 2 | MoE module: router, experts, dispatch/combine | 0 | tt-moe-correctness | router expert-set match >= 99% on real tokens; layer PCC >= 0.999; no dropped tokens (capacity check); determinism test passes |
| 3 | Layer + chain assembly (N layers, then all) | 1, 2 | tt-moe-correctness | chained PCC vs the CPU prefix chain with noise floor; first token sane on real prompts; GSM8K gate (required, n/threshold in scorecard) and determinism pass |
| 4 | Decode path + paged KV (traced) | 3 | tt-moe-correctness, tt-lab-ops | closed-loop decode text equals reference (greedy, near-tie divergences explained), steady ms/token measured after warm-up |
| 5 | Prefill path (traced, chunked, writes paged KV) | 3, 4 for hand-off | tt-moe-correctness | prefill -> decode hand-off exact vs dump; wide shape ladder (see below) passes at 40+ layers |
| 6 | Perf | 4, 5 | tt-lab-ops (paired A/B) | each change: flag, default, paired in-process gain, accuracy gate re-run |
| 7 | Spec decode (MTP drafter) | 4 | tt-moe-correctness (exactness) | accepted/round matches CPU reference statistics; output stream equals plain decode or divergences are near-ties |
| 8 | Serving (vLLM / interleave) | 4, 5 | tt-multiagent | end-to-end request test; decode unchanged within noise |

Tracks 1 and 2 and 0-dumps run in parallel; 4 and 5 run in parallel after 3. Perf (6) starts only after a correct
baseline is merged (a perf agent without a correctness gate produced wrong claims every time).

## Rules that apply to every track
* Reference policy: per-layer teacher-forced checks + short CPU chain + separate prefill/decode phase checks + GSM8K/behavioural gates (required: they are the only guard against compounding error beyond the CPU prefix). See `tt-model-reference`.
* A track is done when its gate is met on the MERGE-CANDIDATE commit, evidence in LOG.md. "It ran" is not a gate.
* Test the whole shape range before declaring a path working: batch/users per row 1..max, sequence 1 chunk and many chunks,
  chunk sizes at both ends, 2 different context lengths in ONE process (DRAM leak check), 4 layers AND all layers.
* Cheap check before a long run: a 4-layer or 2-ISL run (minutes) before any 40+ layer run (30-90 min, build alone is 25-90 min).
* Do not state a number without commit + host + log path. Do not reuse a number across commits or hosts.
* Ask the human BEFORE: changing scope/priorities, lowering precision (bfp4, fp8 KV), skipping a gate, deleting data,
  resetting or killing anything not yours, rebuilding C++ in the shared build, installing packages. Then log the decision.
* Stop and report when: a gate fails twice with the same cause, a host is faulty, you are blocked on another track,
  or your time box (default 3 h without a measured result) is over. Report = what ran, evidence paths, what blocks.

## MiMo-V2.6-Flash specifics to verify in track 0/1 (not yet validated, treat as hypotheses)
* 48 layers = 39 sliding-window(128) + 9 global GQA: two attention types, two KV layouts (ring vs full paged).
* Head dims 192 (QK) vs 128 (V): check that SDPA/paged ops accept unequal dims before designing the attention module.
* 256 experts, top-8, no shared expert, sigmoid router with aux-free bias: 256/32 = 8 experts per chip; check the op's
  expert-count and top-k limits first (DSV4 used 384 experts, top-6, 12/chip).
* fp8 / mxfp4 checkpoint: write the dequant/loader and compare against the HF dequantised weights before any device work.
* MTP drafter: reuse the spec-decode procedure only after track 4 is exact.

## Where things live (DSV4 reference implementation, copy patterns, not code blindly)
`tt/` model code, `tests/` device tests, `reference/` CPU oracle + dump scripts, `tools/` launchers, `demo/text_demo.py` scenario table,
`*_NOTES.md` per-feature notes. Kit tools: `models/common/bringup_kit/` (README; see `tt-bringup-log`, `tt-lab-ops`). Define "done" per model as a scorecard: `scorecard.py template --model <m> --layer-types ... > bringup/<m>/scorecard.json`.
