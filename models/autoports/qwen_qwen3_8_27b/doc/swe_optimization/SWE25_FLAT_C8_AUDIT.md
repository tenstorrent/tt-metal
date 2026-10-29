# SWE25 combined-flag C8 audit

Run 40607188492 used the frozen 25-task SWE-bench Verified subset, isolated agent
network, medium reasoning effort, and the combined grouped, masked and flat
prefill flags. The matched control was run 40572176883. This is one paired
pilot, not an estimate of full SWE100/SWE500 accuracy.

| Measure | Control | Combined candidate |
|---|---:|---:|
| Verified reward | 16/25 | 18/25 |
| Agent completions without exception | 25/25 | 23/25 |
| Agent timeouts | 0 | 2 |
| Trial envelope | 9,323.502 s | 10,933.654 s |

The candidate solved three tasks that the control missed and lost one that the
control solved. Its trial envelope was 17.3% longer. The scored improvement is
therefore insufficient to select it as a faster large-suite configuration.

Evidence audit on the candidate:

- All 25 saved trajectory files matched the SHA-256 values in
  `swe25-flat-c8-trajectory-review.json`. The frozen task identities and
  paired outcomes are in `swe25-flat-c8-paired.json`.
- The network-phase ledger passed for all 25 agent tasks. The stable server
  audit has 869 HTTP starts and ends, all HTTP headers 200. It contains 867
  complete responses, of which 863 were saved in trajectories. Four complete,
  capped 16,384-token responses were not saved there but remain in the server
  audit; two other request ends lacked complete-response accounting. These six
  are retained in the response review rather than silently dropped.
- Manual review of all 49 flagged retrieval/history commands found no observed
  successful external download. Eleven commands attempted external GitHub,
  PyPI or package retrieval; outputs show DNS failures, empty `curl` output,
  or a failed `curl` exit. Some shell pipelines returned zero after a failed
  network command, so the tool return code alone is not evidence of a fetch.
  The other candidates inspect local repository history, source, or tests.
  Four tool-format feedback messages are present in saved trajectories.

Local evidence is under `/home/mvasiljev/build/qwen38-swe/`:
`swe25-flat-c8-summary.json`, `swe25-flat-c8-paired.json`,
`swe25-flat-c8-isolation.json`, `swe25-flat-c8-response-review.json`,
`swe25-flat-c8-trajectory-review.json`, and
`swe25-flat-c8-stable-audits/`. The raw trajectories and audit streams stay
outside Git.

The review establishes no observed external answer retrieval in this run. It
does not prove absence of pretraining contamination, packet-level network
isolation, GPU parity, or statistical superiority. The trial envelope excludes
server startup and artifact collection.

The [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B) reports 61.7
on **SWE-bench Pro** using the Claude Code harness. This pilot uses
**SWE-bench Verified** with mini-swe-agent, so that published score is a
reference to a different benchmark and cannot be used as a parity target.
