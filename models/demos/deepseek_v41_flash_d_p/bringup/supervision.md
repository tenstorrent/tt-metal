# Supervision log: deepseek_v41_flash_d_p

Every intervention by the supervising session: time, task, trigger, classification, action, resulting commit.

## 2026-10-14 intake (owner delegated; owner away overnight)

- Owner's decisions (2026-10-14): deployment box galaxy_4x4 at 1M tokens; boxes galaxy_4x4 / galaxy_8x4 /
  loudbox_2x4 (partial only) / quietbox_2x2 (this run); partial mode, layers 0-3, 14, 20, 21, 24; engram on the host
  from the checkpoint files; vision and DSpark skipped; DeepSeek's inference code as ground truth; max precision,
  experts in a format exact for the checkpoint's fp4; the supervisor approves the intake spec, the run stops at the
  plan for the owner; framework commits stay local until the owner looks.
- 16:00 framework F65 (layers that read other layers' state: hooks state_sources) built before the run, since this
  model's attention shares KV caches, top-k indices and a candidate pool across layers; selftests 377 pass; commit
  a67c6435d58 (local). F64 layer pattern fix for bare `layers.N.` names: 5a7d562f62a (local).
- 16:15 intake approved by the supervisor on the owner's delegation. Checkpoint download (510 GB, revision
  5254448103b6) running in tmux `ds41dl`; the run starts when it completes.
- 16:50 framework F65 part 2 (token ids for steps keyed on them: ctx.extra tokens, device-model begin_chunk), for
  engram; selftests 379 pass; commit eb615f880ed (local). Cloned tt-d-gen (8a4a0042, no build; SC.1 reads it) to
  /localdev/dnijemcevic/tt-d-gen; it has no DeepSeek-V4.1 code.
