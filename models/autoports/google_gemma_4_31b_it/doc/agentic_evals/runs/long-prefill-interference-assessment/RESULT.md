# Long-prefill interference assessment result

No additional device allocation. Existing evidence does not establish a serving-config change with positive release makespan benefit. Change:none; problem remains open. Used0deviceh; credited saving0h, actual potential saving unknown. This is not proof that interference has zero cost.

Stock scheduler already uses2prefill:1decode steps. A2:4 candidate could reduce decoder stalls while delaying waiting-agent prefills; no measured aggregate benefit supports selecting it. Synthetic107K overlap49.259s/28anchor tokens proves stalls but cannot price actual task critical paths. Valid SWE2 prefill-labeled27.577s of338.480s is host accounting, not removable-time evidence, and current SWE5 repeat4041.919s has different long/capped behavior. GPQA C4ten's8prefill/19863decode steps and observed caploops do not support long-prefill priority. Existing shape checks would be duplicated.

Revisit after repaired normal-run phase/native artifacts identify repeated long uncached prefills overlapping release-critical decodes. Price nonoverlapping agent-makespan savings against added waiting and confirmation cost. Only then consider matched2:1/2:4 exact same-arrival replay; accept exact native outputs plus confirmation, or same-current-cohort scored pair if outputs change. Preserve context/caps/stock runtime. Prior0.25h estimate is unallocated and must be repriced.

PROPOSAL.md and RESULT.json retain detailed evidence and conditions. No source/device/publication/central changes.
