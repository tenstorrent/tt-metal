# Round r06 summary (policy v5, W=4, depth cap 2, root r05-b01-a01 = 1.5696) - STOPPED BY HUMAN

Stopped after 3 of 4 step-1 attempts at the human's request ("enough for this experiment"). r06-b01-a01 was
interrupted before committing and is recorded as lost. No r06 node beat the root.

| branch | a01 |
|---|---|
| b01 | lost (stopped) |
| b02 | 1.5419: uneven 9/11 wave split |
| b03 | 1.5512: uneven 9/11 wave split (same idea, independently) |
| b04 | 1.4264: congestion-aware out-of-order dual-NoC drain |

Notes: the 9/11 split lost slightly on the small shapes, and both reflections say to revert to 10/10 (or try 11/9).
Fifth failed output-drain mechanism; posted writes (r04) remain the only drain change that paid.
