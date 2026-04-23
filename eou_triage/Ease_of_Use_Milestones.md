LLK \- Ease of Use 2026 Milestones

1. 1.0 Refactoring
   1. Current progress
      1. LLK API Refactor [\#33823](https://github.com/tenstorrent/tt-metal/issues/33823)
         1. Was the main quest before the SDPA Q1 sprint, no we need to continue
         2. DEST bit 11  - offloaded to Anil, no action, just follow his work in the issue [\#1568](https://github.com/tenstorrent/tt-llk/issues/1568)
         3. Quasar Uninits \- keep or ditch? Low effort low value
      2. Closing LLK Contract Gaps [\#911](https://github.com/tenstorrent/tt-llk/issues/911)
         1. 7 state handling issues
         2. 10 cleanup issues
         3. We could start exploring for the candidates here to pick up next
      3. Compute API Split [\#35739](https://github.com/tenstorrent/tt-metal/issues/35739)
         1. Two streams: add missing block variants and cleanup redundant
         2. Does it still make sense given the 2.0 effort described below?
      4. Compute API Cleanup [\#22907](https://github.com/tenstorrent/tt-metal/issues/22907)
         1. Many uncategorized issues here
   2. Plan Q2/Q3 \- Shift to Milestones:
      1. Q2 Finish API Refactor
      2. Q2 Contract Gaps \- All HW States Categorized
      3. Q2 Contract Gaps \- start
      4. Q2 Split \- Sync with Kernel Helper Lib Team
      5. Q3 Contract Gaps \- extended
      6. Q3 Split \- start
      7. Maybe Q4 Cleanup
# Not as important in Q2:
<!-- 2. Asserts
   1. Current progress \-
      1. We have sanity test nightly debug run 1/day
      2. 3 PRs shown \- codesize and make Green APC/BPC
      3. **Find original assert class list \- did we cover all/some?**
      4. **Which checks asserts cover, where are they?**
   2. Plan Q2/Q3
      1. Q2 Plan
      2. Nemanja will shift to perf infrastructure
      3. Run LLK Unit tests with asserts on by default on PR gate
      4. What’s the blocker there? Remove configure check asserts that caught many errors, but hw config check should sanitizer catch and not assers
3. Sanitizer
   1. Current progress
      1. Lazar and Straja not working on it
   2. Plan Q2/Q3
      1. Q2 Straja come back to this? When? Poc/MVP?
      2. Q2 Should re-evaluate what the Sanitizer brings with Sentinel and Asserts in the main already covering subset of functionalities
      3. Q2 Stopped on assert/dpring decision, should transition to new Vuk’s DPRINT
      4. Later \- Should expand to new APIs
      5. Later \- Sanitizer has some important diffs \- FSM, dest width tracking
4. Compute API 2.0
   1. Poll sent 3.20 \- 4.6.
   2. Form requirements for new APIs
   3. See the diff from 1.0 to make this happen
   4. Generate a proposal after interviews with poll contributors (customers)
   5. PoC end of May
   6. Send over all pain points with current APIs -->
