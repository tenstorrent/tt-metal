# CI disk admission basis

Result: observed free space is real but does not prove the requested task-storage-plus-admission-reserve inequality. Filesystem identity, Docker data root and task admission reserve remain unknown. No hardware, probes, workflow changes or dispatches were performed (0 device hours).

## Observed evidence

SWE run42200513517, runner qb2-120-p05t02: `bringup/artifacts/stage12-swe-activation-live.log:72` identifies the stock host hook `/opt/tt-forge-infra/scripts/reset.sh`. Its output at11:07:09UTC reports `Free space:953 GB`, `Current disk usage:73%`, and after cleanup still73% (:580-583). The log does not show its df command, filesystem/mount identity, block unit, DockerRootDir or daemon storage configuration. Therefore retain953 with reported unitGB and timestamp; do not relabel it exact GiB or bytes, or assume it describes /var/lib/docker.

The same log:1215-1220 reports model storage `PERSISTENT_VOLUME_ROOT=/localdev/persistent-volume` and `HOST_HF_HOME=/localdev/hf_home`. These establish model/cache paths, not Docker image/layer/volume placement or the filesystem measured by the host hook. A read-only source retrieval attempt for `tenstorrent/tt-forge-infra/contents/scripts/reset.sh` returned404; it does not establish that the private host script is absent. Terminal's earlier980GB belongs to a different observation and must not replace this runner's measurement.

## Source-defined checks

TTI `workflows/setup_host.py:492-513` checks `shutil.disk_usage(check_path)` only for model_source=huggingface with host_hf_cache or host_model_volume_root, bypasses for host_weights_dir, other model sources, or no qualifying path. Its threshold is model_spec.min_disk_gb. `workflows/model_spec.py:775+` includes `MIN_DISK_GB_AFTER_DOWNLOAD=20` in an inferred model-download footprint. This is not a task-container admission reserve and cannot be reused as one. The inspected log has host_hf_cache=None and host_weights_dir=None and no `Sufficient disk space` line; it provides no affirmative task admission check.

Harbor checkout `bringup/artifacts/parser-protocol-harbor` is exactly e76c762fca5267dfb36e1f900f0a7724ed72e735, matching live log:1578-1579. Docker preflight (`src/harbor/environments/docker/docker.py:183-200`) checks Docker availability and successful `docker info`, captures its output internally and does not report a task-disk reserve or disk-admission inequality. Resource compose writer (`src/harbor/environments/docker/__init__.py:36+`) implements CPU/memory limits and reservations, not a disk reserve. Source searches of this Harbor checkout found no disk_usage/free-disk task admission calculation. This proves no reserve in the inspected mechanisms, not a zero reserve across externally configured runner services.

## Task demand and comparison

Frozen SWE smoke contains2 tasks, each declared storage10G. Harbor `models/task/config.py:506-514,553-568` parses10G as10240MB using factor1024. Declared sum=20480MB (20GiB in that convention). Frozen SWE nightly5 tasks sum51200MB (50GiB). Frozen Terminal smoke2 tasks each10240MB also sums20480MB; its nightly10 tasks sum102400MB (100GiB). Source: `setup/task_resources.json` and original task.toml metadata. These are declared task footprints, not measured total Docker image/build/cache/log growth, and no blanket overrides were added.

Requested comparison is `declared task storage + admission reserve <= free space on task-storage filesystem`. For SWE smoke this is `20480MB + R <= F_docker`. R is unknown; F_docker is unknown. Reported953GB is a separate host cleanup observation with unresolved mount and units. Consequently neither the inequality nor a numeric headroom/reserve can honestly be certified. Passing a smoke demonstrates that those actual tasks launched, not an admission guarantee for subsequent larger lists. Under operator instructions, continue using unchanged stock setup/cleanup and preserve each future run's reported free space; do not add a stage-owned admission probe or reinterpret missing values as0.
