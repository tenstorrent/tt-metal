#!/usr/bin/env bash
# Bootstraps the pipeline-config bundle for the single-pod tests.
#
# This script wraps the one-time per-cluster bring-up step so the runner
# scripts (run_1block.sh / run_10blocks.sh / run_pipeline.sh) can call it
# automatically when the bundle dir is missing or stale.
#
# What the bootstrap does:
#   1. Creates the bundle dir (default: $TT_METAL_HOME/generated/single_pod_pipeline_dir/,
#      gitignored). Override the path with SINGLE_POD_PIPELINE_DIR. Wipes
#      and recreates the dir on every invocation so we never accumulate
#      stale bundles.
#   2. Drops symlinks back to $TT_METAL_HOME so each host can see the same
#      tt_metal/build/python_env/etc. paths.
#   3. Writes a hostfile with the current HOSTS list.
#   4. Runs models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py
#      to produce two artifacts in the bundle dir:
#        - blitz_decode_pipeline_rank_binding_single_pod_ci.yaml
#        - blitz_decode_pipeline_rank_file_single_pod_ci
#      The generator probes the actual cluster (via mpirun + the
#      test_physical_discovery binary) to discover per-host PCIe device IDs
#      for each (4,2) slice — these become per-rank TT_VISIBLE_DEVICES.
#      That probe is the part that genuinely cannot be skipped: which 8 of
#      each host's 32 devices belong to which slice is cluster-specific
#      runtime info.
#   5. Resets chips on every host (via reset_chips.sh) so the discovery
#      probes don't leave stale device state for the next runner. Skip
#      with BOOTSTRAP_SKIP_RESET=1.

case "${1:-}" in
  -h|--help)
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Regenerates the pipeline-config bundle. Wipes and recreates the dir each
time it runs.

Default bundle path:  \$TT_METAL_HOME/generated/single_pod_pipeline_dir/
                      (gitignored; lives with the repo so /tmp aging out
                       between sessions doesn't break the next run).

The runner scripts (run_1block.sh, run_10blocks.sh, run_pipeline.sh) call
this automatically when the bundle dir is missing, so you usually don't
need to invoke it directly.

Required environment:
  TT_METAL_HOME    Repo root. Must be set; \$TT_METAL_HOME/build must contain
                   a built test_physical_discovery binary
                   (from --build-tests, included in --build-tt-train).

  HOSTS            Space- or comma-separated 4-host list. NO DEFAULT —
                   set per-shell, e.g.
                     export HOSTS="hostA hostB hostC hostD"
                   The scripts refuse to run without it.

Optional environment:

  SINGLE_POD_PIPELINE_DIR
                   Override the bundle path (e.g. /tmp/x or /scratch/x).
                   Default: \$TT_METAL_HOME/generated/single_pod_pipeline_dir.

  BOOTSTRAP_SKIP_RESET=1
                   Skip the chips-reset step at the end. Useful only if
                   you've just reset and want to save ~60s. The runner
                   will then likely need its own reset_chips.sh before
                   the first test.

When to run manually:
  - You changed the host list and want to refresh the bundle without
    rebooting hosts (pre-bootstrap step before the next test run).
  - The PCIe device IDs changed (rare; usually only after a full host
    reboot) and you want to regenerate before any test run hits the
    stale binding.

Examples:
  bash $0
  HOSTS="h1 h2 h3 h4" bash $0
EOF
    exit 0
    ;;
  "") ;;
  *)
    echo "[error] unexpected argument: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_hosts.sh"

: "${TT_METAL_HOME:?set TT_METAL_HOME first}"
GENERATOR="${TT_METAL_HOME}/models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py"
PIPELINE_CONFIG="${TT_METAL_HOME}/models/demos/deepseek_v3_b1/scaleout_configs/blitz_pipeline_config_single_pod_ci.yaml"
DISCOVERY_BIN="${TT_METAL_HOME}/build/test/tt_metal/tt_fabric/test_physical_discovery"

# Bundle path. Defaults to a gitignored subdir of $TT_METAL_HOME so we
# don't rely on /tmp (which can age out between sessions and also requires
# all worker hosts to have it available at the same path). The match
# between this and _run_common.sh is via the SINGLE_POD_PIPELINE_DIR env
# var when _run_common invokes us.
PIPELINE_DIR="${SINGLE_POD_PIPELINE_DIR:-${TT_METAL_HOME}/generated/single_pod_pipeline_dir}"

# Detect whether the launching host is one of HOSTS. If it is,
# we must not ssh to ourselves with `rm -rf $PIPELINE_DIR` because that
# would wipe the local bundle we're building (including the hostfile
# the generator needs to read). We mirror on remote hosts only.
LOCAL_HOSTNAME="$(hostname -s)"
LOCAL_FQDN="$(hostname -f 2>/dev/null || hostname)"

is_self() {
    local h="$1"
    [ "$h" = "$LOCAL_HOSTNAME" ] || [ "$h" = "$LOCAL_FQDN" ] || [ "$h" = "localhost" ]
}

# Populate $1 with .benchmarks, generated/, and symlinks back to $2
# (used as $TT_METAL_HOME). Shared between the local bundle and the
# per-host ssh-mirrored copies — the remote call ships this function
# definition over stdin via `declare -f` (see ssh loop below).
_create_bundle_layout() {
    local pdir="$1"
    local tt_home="$2"
    rm -rf "$pdir"
    mkdir -p "$pdir"
    (
        cd "$pdir"
        mkdir -p .benchmarks generated/{fabric,inspector,test_reports,watcher}
        for sub in build models python_env runtime tests tt_metal ttnn; do
            [ -e "$sub" ] || ln -s "$tt_home/$sub" "$sub"
        done
    )
}

# ---- pre-flight ----------------------------------------------------------

if [ ! -f "$GENERATOR" ]; then
    echo "[bootstrap] ERROR: generator not found at $GENERATOR" >&2
    exit 1
fi
if [ ! -f "$PIPELINE_CONFIG" ]; then
    echo "[bootstrap] ERROR: pipeline config not found at $PIPELINE_CONFIG" >&2
    exit 1
fi
if [ ! -x "$DISCOVERY_BIN" ]; then
    cat >&2 <<EOF
[bootstrap] ERROR: PCIe-discovery binary not found at:
    $DISCOVERY_BIN

Build it with:
    cd \$TT_METAL_HOME
    ./build_metal.sh --build-tests

Or, if you've already done a full --build-tt-train, --build-tests is implied.
EOF
    exit 1
fi

# ---- BH Galaxy revision detection ---------------------------------------
# The slice probe in tests/tt_metal/tt_fabric/physical_discovery/
# test_physical_system_descriptor.cpp (TEST Generate2x4SliceToPCIeDeviceMapping)
# hard-codes the Rev A&B tray layout. PR #41414 added Rev C support
# (is_bh_galaxy_rev_c() ? swap tray ids 2<->3 : keep), but PR #41072
# ("auto-rank-bindings allocation", commit cd8ab9fb0fd) silently dropped
# that hunk in a stale base-merge, restoring the Rev A&B-only hardcoding.
#
# We don't modify tt_metal sources; instead we detect the rev here and, if
# Rev C, post-process the slicer's output below to apply the swap.
#
# Detection: revision_bits = (board_id >> 32) & 0xF >= 3 means Rev C. We
# read board_id from `tt-smi -ls`'s "Board Number" column (16 hex chars
# per row; bits [35:32] = the 8th hex char from the left, 0-indexed pos 7).
TT_SMI="${TT_METAL_HOME}/python_env/bin/tt-smi"
GALAXY_REV="unknown"
BOARD_ID_HEX=""
if [ -x "$TT_SMI" ]; then
    BOARD_ID_HEX=$("$TT_SMI" -ls 2>/dev/null | grep -m1 -oE '[0-9a-f]{16}' || true)
    if [ -n "$BOARD_ID_HEX" ]; then
        REV_INT=$((16#${BOARD_ID_HEX:7:1}))
        if [ "$REV_INT" -ge 3 ]; then
            GALAXY_REV="rev_c"
        else
            GALAXY_REV="rev_ab"
        fi
    fi
fi
case "$GALAXY_REV" in
    rev_c)
        echo "[bootstrap] BH Galaxy revision = Rev C (board_id=0x$BOARD_ID_HEX) — will post-process slicer output."
        ;;
    rev_ab)
        echo "[bootstrap] BH Galaxy revision = Rev A/B (board_id=0x$BOARD_ID_HEX) — slicer output used as-is."
        ;;
    *)
        echo "[bootstrap] WARN: could not detect BH Galaxy revision (no tt-smi or no board id parsed); slicer output used as-is." >&2
        ;;
esac

# ---- (re)create the bundle dir + symlinks --------------------------------
# Wipe any existing dir so a stale bundle doesn't shadow today's discovery.
#
# Two cwd-safety steps below:
#   1. `cd /` before the wipe — if the user ran this script from inside
#      $PIPELINE_DIR (or a subdir), `rm -rf $PIPELINE_DIR` would unlink
#      our cwd inode and every subsequent child shell would print
#      'shell-init: getcwd: No such file or directory' warnings.
#   2. Subshell for the post-create `cd "$PIPELINE_DIR"` — keeps the
#      top-level cwd at `/`, so even if a later step (e.g. the per-host
#      ssh-self loop, when is_self misses) wipes-and-recreates the dir,
#      our process never sat on the unlinked inode.
cd /
_create_bundle_layout "$PIPELINE_DIR" "$TT_METAL_HOME"

echo "[bootstrap] PIPELINE_DIR  = $PIPELINE_DIR"
echo "[bootstrap] hosts         = $HOSTS"
echo "[bootstrap] generator     = $GENERATOR"
echo "[bootstrap] local host    = $LOCAL_HOSTNAME"
echo "[bootstrap] running PCIe-device discovery + rank-binding generation..."
echo "[bootstrap]   (this calls mpirun on all 4 hosts; ~30-60s)"

# ---- mirror the bundle dir on every remote host -------------------------
# Each rank does `cd $PIPELINE_DIR` before invoking pytest, so the dir must
# exist on every host with the same path. We always recreate (-rf + mkdir)
# so a stale remote dir from a prior bootstrap can't shadow today's bundle.
# Skip self — the local dir was just created above; ssh-to-self with
# `rm -rf` would wipe it (including the hostfile the generator reads).
for h in $HOSTS; do
    if is_self "$h"; then
        echo "[bootstrap] $h: skipping (local host; already created)"
        continue
    fi
    ssh -o BatchMode=yes "$h" bash -s -- "$PIPELINE_DIR" "$TT_METAL_HOME" <<EOF
$(declare -f _create_bundle_layout)
_create_bundle_layout "\$1" "\$2"
EOF
done

# Write the hostfile the generator's --hostfile expects (one host per line).
# Done AFTER the remote-mirror loop because if the launcher host is in
# HOSTS (or matched by `is_self`), an earlier write would risk
# being clobbered.
HOSTFILE="$PIPELINE_DIR/single_pod_hosts.txt"
echo "$HOSTS" | tr ' ' '\n' > "$HOSTFILE"

# ---- run the generator ---------------------------------------------------

# Activate the project python_env so 'python' has the right deps (pyyaml, loguru).
PY="$TT_METAL_HOME/python_env/bin/python"
if [ ! -x "$PY" ]; then
    PY=python3
fi

(
    cd "$PIPELINE_DIR"
    "$PY" "$GENERATOR" "$PIPELINE_CONFIG" --hostfile "$HOSTFILE" --output-dir "$PIPELINE_DIR"
)

# Sanity-check what we just produced.
EXPECTED_RB="$PIPELINE_DIR/blitz_decode_pipeline_rank_binding_single_pod_ci.yaml"
EXPECTED_RF="$PIPELINE_DIR/blitz_decode_pipeline_rank_file_single_pod_ci"
if [ ! -f "$EXPECTED_RB" ] || [ ! -f "$EXPECTED_RF" ]; then
    echo "[bootstrap] ERROR: generator did not produce expected files." >&2
    echo "[bootstrap]   missing: $EXPECTED_RB or $EXPECTED_RF" >&2
    exit 1
fi

# ---- Rev C slicing correction (only when GALAXY_REV=rev_c) --------------
# The slicer hard-codes the Rev A&B tray-id <-> chassis-position map. On
# Rev C, trays 2 and 3 are physically swapped, so the slicer mis-groups
# chips: each per-rank "(4,2) RING-LINE" submesh ends up containing chips
# that don't actually have the expected intra-mesh connectivity. The
# topology mapper then refuses to map them downstream.
#
# Fix in this scope (without touching tt_metal sources) by:
#   1. ssh each host -> tt-smi -glx_list_tray_to_device, parse the table
#      to learn tray_id -> {chip ids} per host.
#   2. Re-derive each slice's chip set by the Rev C composition:
#        new[0] = (old[0] ∩ T1) ∪ (old[3] ∩ T2)   # top row, top-right asics
#        new[1] = (old[1] ∩ T1) ∪ (old[2] ∩ T2)   # top row, bot-left asics
#        new[2] = (old[1] ∩ T3) ∪ (old[2] ∩ T4)   # bottom row, bot-left
#        new[3] = (old[0] ∩ T3) ∪ (old[3] ∩ T4)   # bottom row, top-right
#      (Equivalent to swapping tray-2 / tray-3 contributions across pairs
#      of slices — same effect as the rev_c branch in PR #41414.)
#   3. Rewrite slice_to_pcie_device_mapping.yaml AND regenerate the
#      rank-binding yaml's per-rank TT_VISIBLE_DEVICES from it.
if [ "$GALAXY_REV" = "rev_c" ]; then
    echo "[bootstrap] Rev C — applying slicer correction..."
    TRAY_DIR="$PIPELINE_DIR/_tray_maps"
    mkdir -p "$TRAY_DIR"

    # Collect per-host tray-to-chip-id maps in the SAME number space the
    # slicer's slice yaml uses (UMD logical IDs, BDF-sorted). The C++
    # discovery binary's GenerateTrayToPCIeDeviceMapping test writes
    # tray_to_pcie_device_mapping.yaml with `tray_id -> [logical_id, ...]`
    # (the filename is misleading; it's already PCIe->logical-converted via
    # chips_with_mmio inside the test). We invoke it standalone (no MPI) on
    # each host via ssh and `cat` the result back. NOTE: we deliberately
    # avoid `tt-smi -glx_list_tray_to_device`, which reports raw PCIe IDs —
    # mixing those with the slicer's logical IDs in a set intersection
    # produces silently-wrong, non-disjoint per-rank slices on this cluster.
    DISC_BIN="$TT_METAL_HOME/build/test/tt_metal/tt_fabric/test_physical_discovery"
    for h in $HOSTS; do
        ssh_cmd="cd /tmp && rm -f tray_to_pcie_device_mapping.yaml && \
            LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:\$LD_LIBRARY_PATH \
            TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME \
            $DISC_BIN --gtest_filter='*GenerateTrayToPCIeDeviceMapping*' >/dev/null 2>&1 && \
            cat tray_to_pcie_device_mapping.yaml"
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$ssh_cmd" \
                > "$TRAY_DIR/${h}.yaml"; then
            echo "[bootstrap] ERROR: failed to fetch tray map from $h" >&2
            exit 1
        fi
        if [ ! -s "$TRAY_DIR/${h}.yaml" ]; then
            echo "[bootstrap] ERROR: tray map from $h is empty" >&2
            exit 1
        fi
    done

    # Corrector reads its inputs from environment variables (see header of
    # _rev_c_slice_correction.py).
    if ! \
        PIPELINE_DIR="$PIPELINE_DIR" \
        PIPELINE_CONFIG="$PIPELINE_CONFIG" \
        HOSTFILE="$HOSTFILE" \
        TRAY_DIR="$TRAY_DIR" \
        EXPECTED_RB="$EXPECTED_RB" \
        "$PY" "$SCRIPT_DIR/_rev_c_slice_correction.py"
    then
        echo "[bootstrap] ERROR: Rev C correction failed" >&2
        exit 1
    fi
    rm -rf "$TRAY_DIR"
    echo "[bootstrap] Rev C correction done."
fi

# Workaround for an OpenMPI 5.0.7 parser quirk: --map-by rankfile:file=PATH
# treats `-` as a qualifier separator. Any rankfile path containing a hyphen
# (e.g. anything under TT_METAL_HOME=/.../tt-metal/) gets rejected with
# "unrecognized qualifier". tt-run's resolve_path() dereferences symlinks
# before passing to mpirun, so symlinks don't help. We copy the (small,
# 500-byte) rankfile to a fixed hyphen-free path; _run_common.sh passes
# that copy to mpirun.
RANKFILE_FOR_MPI="${SINGLE_POD_RANKFILE_PATH:-/var/tmp/single_pod_rankfile}"
RANKFILE_FOR_MPI_DIR="$(dirname "$RANKFILE_FOR_MPI")"
mkdir -p "$RANKFILE_FOR_MPI_DIR" 2>/dev/null
cp -f "$EXPECTED_RF" "$RANKFILE_FOR_MPI"

# Reset chips on every host. The discovery probes above (the generator's
# mpirun-spawned test_physical_discovery, plus on Rev C the per-host
# GenerateTrayToPCIeDeviceMapping pass) open devices and can leave chip
# locks / FW state stale. Without this, the first test run after bootstrap
# typically fails with "Device N init: failed to initialize FW! Try
# resetting the board." Skip with BOOTSTRAP_SKIP_RESET=1.
if [ "${BOOTSTRAP_SKIP_RESET:-0}" != "1" ]; then
    echo "[bootstrap] resetting chips on all hosts so the runner sees fresh devices..."
    "$SCRIPT_DIR/reset_chips.sh"
fi

echo "[bootstrap] done."
echo "[bootstrap]   rank binding:        $EXPECTED_RB"
echo "[bootstrap]   rank file (bundle):  $EXPECTED_RF"
echo "[bootstrap]   rank file (mpirun):  $RANKFILE_FOR_MPI  (hyphen-free copy)"
echo "[bootstrap]   bundle dir:          $PIPELINE_DIR"
