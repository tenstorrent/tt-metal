# Pending prefill ownership repair

The diagnosed final-chunk failure is repaired in the model generator, with no plugin changes. A request can retain an unfinished prefill in slot 1 while its decoding sibling releases slot 0. The pinned plugin may then submit that request's continuation to slot 0. The former generator retained the old owner and correctly failed the shared-page overlap guard, because it omitted the state transition.

`tt/pending_prefill.py` now plans that transition without mutation. Intermediate prefills are tracked separately from decoding requests. A cross-slot continuation requires a unique pending source at the committed start position whose ordered live page prefix matches in every attention layer. The existing normalized submission tables supply per-layer precedence, row ordering, and bounds validation. Padding and newly appended pages are excluded from the identity proof, but remain subject to the unchanged live-page validator. Same-slot authoritative COW changes retain their previous behavior. An unprovable moved COW prefix fails explicitly.

The planner completes a full destination-to-source permutation, preserving unrelated holders and rejecting duplicate sources, ambiguous identity, and overwriting unrelated live state. The generator validates the complete post-move page ownership and all sampling arguments before applying the existing fixed-address SlotStateMover. Checkpoints, page-table shadows, and pending metadata follow that permutation. No sampler-wide reset, feedback remap, new cache allocation, host device-tensor read, or new trace capture was introduced. A transition-only `pending_prefill_relocation` event records source, destination, and committed position. Final prefill, reset, and release clear pending ownership; accepted decode permutations preserve dormant pending ownership.

CPU validation ran against production AST-extracted orchestration, the real shared-page validator, real checkpoint manager, and real SlotStateMover with CPU tensor copies:

```bash
python_env/bin/python -m unittest \
  models.autoports.qwen_qwen3_8_flash_next.tests.test_pending_prefill_cpu \
  models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_validation_cpu \
  models.autoports.qwen_qwen3_8_flash_next.tests.test_decode_validation_cpu \
  models.autoports.qwen_qwen3_8_flash_next.tests.test_generator_pool_cpu \
  models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_checkpoint_cpu \
  models.autoports.qwen_qwen3_8_flash_next.tests.test_slot_state_cpu -q
```

Result: **60 tests passed in 0.264 seconds**. Eight new regressions cover release then 120→128 relocation, sentinel contents in every mutable state tensor before tail execution, fixed tensor identities, no unrelated sampling remap, wrong/active/occupied source rejection, invalid pages and sampling before mutation, odd 119→128 checkpoint restoration at 118, same-slot COW, per-layer prefix identity and growth, ambiguity and duplicate sources, duplicate newly-live pages, invalid later-row tokens, simultaneous cycles preserving an omitted holder, and pending status through decode remap and release. Existing dormant-overlap tests still pass. These are CPU orchestration assertions, not hardware numerical validation.

The exact generator diff applies to baseline `86130b35` with `patch --fuzz=0`, with line offsets only. The corrected baseline generator is `/tmp/qwen-pending-baseline-m8ldof22/models/autoports/qwen_qwen3_8_flash_next/tt/generator.py`, SHA256 `d92b52d0907f02aea8f0f2878ed2d53baac83d6e69e83c6f9ff1252315d82c3e`. Its patch is `/tmp/qwen-pending-baseline-m8ldof22/generator_correctness.patch`. Eight pending-prefill regressions also passed against that baseline generator's AST in 0.033 seconds, using a read-only `Path.read_text` substitution during test extraction; checkout files were not replaced. The helper is shared unchanged by both arms. The parent will persist corrected baseline provenance before measurements.

**Hardware acceptance remains pending at this handoff.** The existing parent-owned gates will test the actual relocated continuation against solo token output, both matched measurement arms, tracked mixed-shape/async serving, and final acceptance. No device was opened and no serving success is claimed by this repair report. Optional tuning remains stopped under the owner's stage cap; this change addresses a required correctness gate only.

Parent validation update: the corrected baseline and optimized full48 servers both pass the 128-input/32-output relocation-versus-solo check, including an observed slot1→0 move at position120 and exact raw output-token agreement. Both official primary, CI and serial-prefill burst measurements pass, as do final tracked sync/async relocation, history and nine-wave mixed checks. Final lifecycle state is summarized in `final_trace_async_summary.json`; matching performance provenance is in `perf_summary.json`. The persisted, formatted baseline and commit are recorded in `corrected_baseline.json`, superseding the temporary pre-format path/hash above. Full sampling/eval acceptance and stage review are still pending.
