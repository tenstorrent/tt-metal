# Stage 5 owner decision

Owner decision: accept the linear-attention decode speed (0.7195 ms/token vs the 0.5183 ms floor, 7.2% of roofline) as a known limitation. Record it in limitations with this decision and the evidence, finish the review and complete stage 5. No further tuning for this floor.

Source: `/home/ttuser/qwen36-35b-a3b/multigoal_logs/05-05-optimized-multichip-decoder.resume-2.message.txt`, received 2026-11-14.

This accepts only the recorded linear performance shortfall. It does not claim a speedup or an HF-reproduced limitation, change the 10% target, relax numerical checks, or approve the rejected native candidates. The selected implementation and raw measurements remain unchanged. See `../eval_readiness/limitations.json` for the scoped record and evidence.
