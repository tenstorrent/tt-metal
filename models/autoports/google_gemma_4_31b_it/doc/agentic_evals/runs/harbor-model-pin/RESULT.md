# Model Harbor source pin confirmation

Active separate TTI publication checkout was clean at eac746513e4ced539f4462d0dc5762d76e04fc56 before editing. Added only fresh autoport metadata.harbor_git_ref=e76c762fca5267dfb36e1f900f0a7724ed72e735 in workflows/model_specs/dev/llm.yaml and a resolved-model metadata assertion in tests/workflows/test_gemma4_31b_autoport_v3.py. No other implementation entries, pins, parsers, task lists or limits changed.

Source identity: bringup/artifacts/stage12-swe-live-0745.log line9355 records HEAD e76c762f at07:33:57 for the live SWE job. Read-only official `git ls-remote https://github.com/tenstorrent/harbor.git refs/heads/tt-inference-server` returned full e76c762fca5267dfb36e1f900f0a7724ed72e735, matching the inspected source. This establishes the source being executed, not task correctness or successful suite completion.

58 focused model/Harbor tests passed. Full hooks passed (Ruff, format, pytest, copyright) with both tt_metal_home and TT_METAL_HOME unset. Initial hook run only reformatted the assertion; final clean hook log is bringup/artifacts/stage12-harbor-model-pin-hooks-final.log. Actual setup checkout verification with the new registration is deferred to the next normal rung; no devices used here.

No commit/push/dispatch. Two source files remain ready for coordinator integration. Device hours zero; same observed source is now selected immutably rather than through the moving branch. This does not claim transitive Python dependency locking.
