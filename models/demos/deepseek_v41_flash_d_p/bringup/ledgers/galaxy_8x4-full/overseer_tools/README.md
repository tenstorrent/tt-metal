# Overseer tools of the galaxy_8x4-full run (copied from the node-local /scratch of bh-glx-110-c10u02, 2026-10-28)

The live copies are in /scratch/dnijemcevic (local disk: nothing there survives a move to another node). On a new node:
copy this folder back to /scratch/dnijemcevic (posprof/ -> /scratch/dnijemcevic/posprof/), adjust node_env.sh (its lock
workaround is specific to this node's stale /tmp/tt-device.lock), and make the paths in the scripts match.

- node_env.sh: environment every launcher sources (TTPOOL_LEGACY_LOCK workaround; node-specific).
- run_ds41_chain.sh: Paris (resident engram tables) -> if PASS the 5-position profile (posprof/series_engram.sh, tag
  ENG) -> orchestrator --until P.6.
- run_ds41_engram_check.sh: the 16:50 ladder 56k + Paris check of the 2x2 engram commits.
- k1_smoke.sh: K.1 runner smoke once, keeping the tt-triage dump.
- posprof/: the 5-position profile (0 / 51200 / 102400 / 204800 / 512000 + one 5120 chunk): run.sh (device time, real-time
  profiler, one process a position), run_wall.sh (F196 warm wall + F205 settle), pos_profile_plugin.py (pytest plugin,
  needs PYTHONPATH=<repo>:<posprof>), table.py / summary.py (tables from results/).
- profile_tblock_20261028/: the baseline for the comparison: block MoE, fused head + tail, FABRIC_2D_TORUS_XY, engram
  tables memory-mapped (before MADV_RANDOM / resident), TBLOCK<pos>.json / _profile.json, TBLOCKWALL<pos>.json and the
  wall medians in TBLOCKwall.log. Published: https://claude.ai/artifact/HRfMf31S4781YkU2M8NDUf
