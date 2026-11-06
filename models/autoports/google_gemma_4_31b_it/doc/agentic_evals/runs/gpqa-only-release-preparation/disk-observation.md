# Existing CI storage observation

The repaired GPQA20 startup on `qb2-120-p04t05` reports **1006 GB free and72% used**, unchanged72% after cleanup, at2026-11-06T19:43:16Z. Evidence: `bringup/artifacts/stage12-gpqa20-repair-current.log:576`–579. These are the hook's reported units/rounded values, not a measured Docker-root reserve.

The log attributes these lines to the runner-administrator job-start hook `/opt/tt-forge-infra/scripts/reset.sh` (lines73–77), **before** the tt-shield checkout begins at582. They are not emitted by the pinned tt-shield workflow's stale-container cleanup. The hook's exact installed source/revision is not present in the collected source or log; the candidate GitHub repository lookup returned404 and indexed source searches did not locate the message. Consequently the `df` target/mount, rounding method, deletion thresholds and minimum free-space reserve remain **unknown**. Do not infer `/`, `/localdev`, or `/var/lib/docker` from this line.

Read-only inspection of `tenstorrent/tt-shield` at `ba2f03318608be52c5cb2085469598a89dcfa0fc`, `.github/workflows/workflow_run-tests-with-inference-server.yml`, establishes the separate stock workflow policy:

- Lines230–237 stop and remove stale Docker containers. This block has no numeric free-space admission/reserve rule and does not establish Docker's storage location.
- Lines60–62 set fallback STORAGE_PATH `/localdev` and preferred shared persistent/HF directories under `/mnt/MLPerf`.
- Lines251–269 choose shared paths if both exist, otherwise create/use `/localdev` paths, otherwise working-directory paths. This is directory availability selection, not a free-space check.
- Current-run lines1213–1218 prove the selected paths were `/localdev/persistent-volume` and `/localdev/hf_home`. These paths are known; their filesystem free space is not identified by the earlier unqualified hook measurement.

Docker Root Dir, daemon data-root configuration, mount/device relationship between Docker root and `/localdev`, and per-filesystem inode/reserve values are absent from this evidence. Thus the observation improves host admission evidence but **does not prove a1006GB Docker-root budget or a storage reserve policy**. Establishing that relationship would require the installed hook source plus existing mount/daemon-root evidence, or a separately authorized future stock observation; none was run for this task. No CI dispatch, device command, cleanup, source mutation, raw log copy, or credential collection was performed.
