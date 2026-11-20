# bh-glx-110-c10u02 (slurm 158554): /tmp/tt-device.lock is a stale file of a previous user (dlukic, 0660, 19:00) that
# we cannot open; no legacy runner can share this exclusive node, so the pool's legacy interop lock is our own file.
export TTPOOL_LEGACY_LOCK=/tmp/tt-device.legacy.dnijemcevic.lock TTPOOL_LEGACY_DIRTY=/tmp/tt-device.legacy.dnijemcevic.dirty
# 2026-11-18 (overseer): on a HUNG mesh, recover-hosts' sys-triage step "tt-fw-terminal" took the host down twice
# (2026-11-17 14:30:41, 2026-11-18 04:36:53: the triage file ends at that header, the boot ends seconds later). The safe
# runner's automatic recovery skips sys-triage until the cluster admins have looked at it.
export TTPOOL_RECOVER_CMD="source /etc/profile.d/exabox-helpers.sh && HOSTS=\$(hostname) recover-hosts --skip-mpi-stress-test --skip-sys-triage"
