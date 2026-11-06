# SWE activation live confirmation

Valid original smoke2 at C2/B8; **0/2, Wilson95 [0%, 65.76%]**. Nightly5 is eligible immediately subject to coordinator scheduling. Scores are reported, not gated. No verified same-suite/version reference supplied.

Evaluation374.988095s (0.104163h), device job803s, workflow4195s (completed monitor timestamps; manifest was stale at audit start). The prior quarantined evaluation7306.65975s is not a valid-score baseline or isolated serving-performance comparison.

- django__django-11265: 47 calls, 8448 output tokens, max971 per call; agent307.649815s, trial332.204385s; Submitted, reward0.
- pylint-dev__pylint-7080: 44 calls, 9491 output tokens, max1272 per call; agent342.937653s, trial374.869093s; Submitted, reward0.

Both configs carry agent-scoped BASH_ENV=/root/.bashrc. Django actually executes Python under /opt/miniconda3/envs/testbed/lib/python3.6 and its baseline31tests pass before edits. Pylint actually reports Python3.9.20/astroid2.11.7 and runs analysis successfully. This confirms the intended activation behavior and removes the original missing-dependency/network validity problem for these tasks.

All188 messages and91 model/tool actions were checked. No pip install, external retrieval or fetched fix is observed. Source-license URLs are inert text. Official verifier package downloads occur after agent completion and are distinguished from agent retrieval. Both explicit final submission commands produce Submitted; mini's synthetic returncode-1/action-was-not-executed for the final tool is the normal Submitted exception path. All91 response finish reasons are tool_calls, none length; maximum completion1272 is below32768, and agent durations are below21600s. No observed request/tool timeout. Incorrect local patches and test failures are model outcomes.

Exact frozen tasks django__django-11265 and pylint-dev__pylint-7080 retain task source86723674f04e4209ac479d0fb75d9d9f44b4377e. Actual Harbor lock e76c762fca5267dfb36e1f900f0a7724ed72e735, mini2.2.8. Runtime registration confirms fresh autoport all60layers0..59,1x4/context262144, native gemma4 parsers. TT439c22b0e991e5af626efafa7224edf28f84795b and TTI2d738ec3910f7f13d1614e6196d9ff5f646b6fe1 match report; plugin full dispatch pinca7a18246e4111629ed6cb4a7d1192002749fccf matches artifact abbreviationca7a182. HF/tokenizer842da3794eaa0b77d5f08bae87a17459d91ff475. Release request3600s/output32768/agent21600s unchanged, serveddefaulttop_k32; task CPU/memory overrides absent.

Original network-contaminated attempts remain quarantined. This confirmation claims no packet-level audit, score improvement or isolated speedup. Evidence in audit.json references the complete source trajectories/results. No central docs/source/device/launch changes.
