# Experiment 2 resumable checkpoint

Updated 2026-09-23 09:51 UTC. All eight matched baseline runs passed exact logits/touched KV/greedy outputs. Native/original resident medians: 128 = 7.635744/9.243841 ms; 2048 = 8.067759/9.665413; 8192 = 8.677946/10.334085; 128 with256 outputs = 7.678514/9.287472. Five warmed trials each; 31 or255 decode steps. No candidate result yet.

Branch `codex/llama31-qb2-megakernel`, starting source `f776a26ce77921cc84331cafb1434ba20c6ec46b`, original base `b8915544692d8f9feb2c890afbc2f22791560cd2`. Prior results/instructions are preserved in EXPERIMENT1_PROGRESS.md and the immutable starting SHA.

Exclusive job 114624, host qb2-120-p01t01, expiry 2026-09-23 17:12:44 UTC, freshly verified by local scontrol. Begin final preservation by 16:52:44 and finish all builds/tests/copies/checkpoint by 17:02:44. No allocation changes. Verify allocation before any hardware action; commands run serially under artifacts/run_device.py. Resets are authorized for this owned QB2; preserve triage and stop only own workloads.

Root /home/moconnor/llama-minlat-114624; durable evidence /data/moconnor/llama-minlat-114624. Prior mirror /data/moconnor/llama-megakernel-113796 is read-only. Local build/env/cache under the new root. Run environment artifacts/run-env.sh. See WORK_LOG.md for plan and artifact command scripts for exact setup.

Initial suite complete, sources unchanged for baseline measurements; summary artifacts/initial-benchmark-summary.json. Source checkpoint 8355ae31. Reader/subblock/bounded-barrier candidates now installed, awaiting focused device tests under artifacts/commands/components.sh; default controls preserve the original path. Source/installed libtt_metal BuildID 467a3c083d79401b9b46f83db7a46430ca790654 matches the actual loaded library. Next: reader and projection subblock interventions, focused numerics and full-model measurements. No credentials or external posting; parent transfers bundles using its own authentication.
