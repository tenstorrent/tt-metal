# Closure metadata repair

The first host closure check tried to read owned_server.json after benchmark_server --stop had correctly removed it. No device command failed. The repaired check uses the retained final server identity PID, verifies that process is gone, scans /proc device FDs and asserts no owners, and records that the owned state was removed. Raw archive creation/restoration completed with897 files. All evidence gates are run below this ownership check.
