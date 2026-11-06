# CI startup cost investigation

Goal: attribute cold CI versus warm local startup using logs and source, without assuming causes from elapsed gaps. Host-only box0devicehours. Inspect timestamps for weight conversion/load and warmup, cache paths/mount lifetimes and stock TTI source. Price four release server launches conservatively; propose a bounded discriminating test only if it changes a serving decision. No source, serving config, publication or device changes.

Candidate cache-env.patch changes only two model env values. Host check uses actual stock SetupConfig and generate_docker_run_command: both candidate paths are children of the named volume's exact cache_root destination (host-check.json). No guessed external mount and no source checkout edits.

Coordinator confirmation box0.25devicehours: first host-only cache-path/local-file identity at pinned revision, then same candidate serving configuration with paired exact token outputs. Local confirmation can validate path semantics and numerical invariance only. Actual CI startup saving remains unknown until a subsequent already-planned rung carries the confirmed candidate; do not delay current nightly work.
