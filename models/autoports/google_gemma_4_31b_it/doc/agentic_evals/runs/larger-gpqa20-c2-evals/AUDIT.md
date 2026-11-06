# GPQA20 completed-run audit

Functional_at remains **10**. The frozen20 run completed in 2.0430 evaluator hours (2.1300 device-job wall hours, 2.1353 workflow hours). Raw score **16/20 = 80%**, Wilson95 **[58.40%, 91.93%]**, is reported but not banked.

Actual dataset indices **7 and 20** returned empty visible answers. Each first concurrent phase slot ran **126975 consecutive decode steps** over **5455.5122 host seconds**, consistent with the126976 output cap after the prefill token. Hidden reasoning, native token IDs, usage and finish reasons are absent. This evidence does not decide looping versus unfinished useful reasoning; cap failures remain unresolved. No request timeout was observed. The other18 have conclusive answers; their16 successes do not replace the frozen20 cohort.

Pinned lm-eval af401ef selects sample membership in ascending dataset order but assigns logged doc_id from the unsorted supplied list (api/task.py:574–604; evaluator.py:624–650). Thus logged152/189 are actual7/20. This is a label defect, not incorrect metric association: the evaluator scores each actual document against its corresponding response. All ten earlier question and prompt hashes are present unchanged. audit.json preserves both identifiers and compact hashes, without private questions or reasoning.

Runtime phase identity confirms this autoport, all60 layers, B8, context262144 and head_bfp4; frozen pins and C2 limits match the manifest. Phase export has 7402 snapshots, 186277 matched submit/complete steps and no truncation or pending final work. These are host boundaries, never device-time rooflines. Exact saved requests and asynchronous serializer source were handed to the coordinator's replay owner; seed42 and stop[] are source-backed. No new execution occurred in this audit.
