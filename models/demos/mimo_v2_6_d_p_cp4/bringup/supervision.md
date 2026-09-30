# mimo_v2_6_d_p_cp4 supervision log

- 2026-09-30 22:59, intake. Owner (sjovic) set the configuration up front: mesh 1x4, CP=4, attention TP=1, experts EP=4 TP=1
  (64 complete experts per chip), dense MLP TP=1, chunk 5120, experts at HiFi4. Written into agents.rules. Local paths
  (hf, 1x4 goldens, art) set in the spec because /localdev does not exist on this machine. Intake approved on the owner's word.
- 2026-09-30, PL.1. Plan checked against the owner's configuration: CP=4 slices, ring_joint SDPA (full) + ring halo
  (sliding, window 128), TP=1 attention and dense MLP, EP=4 dispatch group of 4 on axis 1 through the 2x2 forks,
  unified_routed_expert_moe high_precision HiFi4. Memory 15.95 of 27.20 GB. No ttnn/cpp edits, no OPGEN. Planner
  choices beyond the configuration: V padded 128 -> 192 again (ring op needs V dim == QK dim), sliding KV cache bfp8
  (ring sliding path). Plan approved on the owner's standing approval of this configuration.
