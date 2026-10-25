# AutoFix: continuing-prefill slot ownership

Diagnosis: AUTODEBUG.md section5, verified against standalone TT plugin
1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf. The allocator excluded continuing
prefills from held slots and assigned their packed rows as new owners. No prefill
state transport accompanied that ownership change. A peer finishing could move a
continuing request from slot1 to slot0 while its recurrent state remained in slot1.

Repair: preserve each existing prefill owner's slot and reserve all existing owners
before placing new requests. New owners still prefer their packed row when free.
Decode retains responsibility for explicit state permutation. No capability,
environment-variable read, configuration key, or model identity gate was added.

Changed files in `bringup/dependencies/vllm-tt-plugin`:
- `src/vllm_tt_plugin/model_runner.py`: allocator ownership repair.
- `tests/test_state_slots.py`: peer-finish continuation, reordered continuations,
  and new-before-existing tests. Each verifies owner/state correspondence through
  the following decode remap, as well as the returned prefill slots.
- `docs/MODEL_LIFECYCLE.md`: continuing-prefill ownership contract.

Validation used the installed container interpreter and import-only TTNN stubs.
No device was opened, server started, dependency changed, or commit created.

Common invocation prefix:
`docker exec -w /workspace/bringup/dependencies/vllm-tt-plugin -e USER=mvasiljev -e PYTHONPATH=tests/host-stubs qwen38-functional-build /opt/venv/bin/python -m pytest --confcutdir=. --rootdir=.`

Before source repair, `tests/test_state_slots.py -q -k 'continuing_prefill or reordered_prefill or new_prefill_reserves'`
failed all3 new regressions. After repair,
`tests/test_state_slots.py tests/test_decode_reload_contract.py tests/test_prefill_chunk_alignment.py tests/test_chunked_prefill_policy.py -q`
passed74 tests in3.34s. Logs: `readiness_vllm/plugin_state_before.log` and
`readiness_vllm/plugin_state_after.log`.

Formatting/lint command:
`docker exec -w /workspace/bringup/dependencies/vllm-tt-plugin -e USER=mvasiljev -e PRE_COMMIT_HOME=/workspace/.cache/pre-commit qwen38-functional-build /opt/venv/bin/python -m pre_commit run --files src/vllm_tt_plugin/model_runner.py tests/test_state_slots.py docs/MODEL_LIFECYCLE.md`
All applicable hooks passed. Log: `readiness_vllm/plugin_state_precommit.log`.

Initial test invocation needed `--confcutdir=.` to avoid importing the enclosing
tt-metal conftest with import-only TTNN stubs, and `USER=mvasiljev` because the
container UID has no passwd entry. These were test-launch corrections, not model
or device failures.

Result: host ownership defect repaired. Hardware manifestation and generated-text
correctness still require the parent stage's concurrent chunked-prefill serving
check. Existing slot-map validity and scheduler admission invariants are unchanged.
