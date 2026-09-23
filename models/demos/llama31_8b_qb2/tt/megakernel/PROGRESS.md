# Experiment 2 resumable checkpoint

Updated 2026-09-23 09:37 UTC. Matching source build installed; four-chip connectivity and FABRIC_1D_RING mesh passed. **Matched model baseline suite is running; no new latency result yet**.

Branch `codex/llama31-qb2-megakernel`, starting source `f776a26ce77921cc84331cafb1434ba20c6ec46b`, original base `b8915544692d8f9feb2c890afbc2f22791560cd2`. Prior results/instructions are preserved in EXPERIMENT1_PROGRESS.md and the immutable starting SHA.

Exclusive job 114624, host qb2-120-p01t01, expiry 2026-09-23 17:12:44 UTC, freshly verified by local scontrol. Begin final preservation by 16:52:44 and finish all builds/tests/copies/checkpoint by 17:02:44. No allocation changes. Verify allocation before any hardware action; commands run serially under artifacts/run_device.py. Resets are authorized for this owned QB2; preserve triage and stop only own workloads.

Root /home/moconnor/llama-minlat-114624; durable evidence /data/moconnor/llama-minlat-114624. Prior mirror /data/moconnor/llama-megakernel-113796 is read-only. Local build/env/cache under the new root. Run environment artifacts/run-env.sh. See WORK_LOG.md for plan and artifact command scripts for exact setup.

Current: artifacts/commands/baselines.sh runs native then original decode_token serially at 128, 2048, 8192 and 128/256 outputs (five repetitions), using the historical teacher references. Kernel source remains unchanged. Source/installed libtt_metal BuildID 467a3c083d79401b9b46f83db7a46430ca790654 matches the actual loaded library. Next: reader and projection subblock interventions, focused numerics and full-model measurements. No credentials or external posting; parent transfers bundles using its own authentication.
