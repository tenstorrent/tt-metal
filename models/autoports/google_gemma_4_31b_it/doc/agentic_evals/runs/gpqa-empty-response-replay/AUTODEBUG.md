# Offline pending-snapshot false alarm

Initial private analysis.json reported any_exported_loss=true despite final pending0 and lossfalse. Static diagnosis: analyzer included every intermediate pending_incomplete flag in permanent loss aggregation.592 periodic snapshots correctly recorded in-flight asynchronous work. That flag is a closure requirement only at final drain; it is not historical lost evidence. No device/runtime failure occurred.
