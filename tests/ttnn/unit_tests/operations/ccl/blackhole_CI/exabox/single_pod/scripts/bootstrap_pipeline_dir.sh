#!/bin/bash
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
#   3. Writes a hostfile with the current SINGLE_POD_HOSTS list.
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

Optional environment:
  SINGLE_POD_HOSTS Space- or comma-separated 4-host list. Default in _hosts.sh.
                   *** OVERRIDE THIS for a different cluster. ***

  SINGLE_POD_PIPELINE_DIR
                   Override the bundle path (e.g. /tmp/x or /scratch/x).
                   Default: \$TT_METAL_HOME/generated/single_pod_pipeline_dir.

When to run manually:
  - You changed the host list and want to refresh the bundle without
    rebooting hosts (pre-bootstrap step before the next test run).
  - The PCIe device IDs changed (rare; usually only after a full host
    reboot) and you want to regenerate before any test run hits the
    stale binding.

Examples:
  bash $0
  SINGLE_POD_HOSTS="h1 h2 h3 h4" bash $0
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

# Detect whether the launching host is one of SINGLE_POD_HOSTS. If it is,
# we must not ssh to ourselves with `rm -rf $PIPELINE_DIR` because that
# would wipe the local bundle we're building (including the hostfile
# the generator needs to read). We mirror on remote hosts only.
LOCAL_HOSTNAME="$(hostname -s)"
LOCAL_FQDN="$(hostname -f 2>/dev/null || hostname)"

is_self() {
    local h="$1"
    [ "$h" = "$LOCAL_HOSTNAME" ] || [ "$h" = "$LOCAL_FQDN" ] || [ "$h" = "localhost" ]
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

# ---- (re)create the bundle dir + symlinks --------------------------------
# Wipe any existing dir so a stale bundle doesn't shadow today's discovery.
rm -rf "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR"
cd "$PIPELINE_DIR"
mkdir -p .benchmarks generated/{fabric,inspector,test_reports,watcher}
for sub in build models python_env runtime tests tt_metal ttnn; do
    [ -e "$sub" ] || ln -s "$TT_METAL_HOME/$sub" "$sub"
done

echo "[bootstrap] PIPELINE_DIR  = $PIPELINE_DIR"
echo "[bootstrap] hosts         = $SINGLE_POD_HOSTS"
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
for h in $SINGLE_POD_HOSTS; do
    if is_self "$h"; then
        echo "[bootstrap] $h: skipping (local host; already created)"
        continue
    fi
    ssh -o BatchMode=yes "$h" "
        rm -rf $PIPELINE_DIR
        mkdir -p $PIPELINE_DIR
        cd $PIPELINE_DIR && mkdir -p .benchmarks generated/{fabric,inspector,test_reports,watcher}
        for sub in build models python_env runtime tests tt_metal ttnn; do
            [ -e \$sub ] || ln -s $TT_METAL_HOME/\$sub \$sub
        done
    "
done

# Write the hostfile the generator's --hostfile expects (one host per line).
# Done AFTER the remote-mirror loop because if the launcher host is in
# SINGLE_POD_HOSTS (or matched by `is_self`), an earlier write would risk
# being clobbered.
HOSTFILE="$PIPELINE_DIR/single_pod_hosts.txt"
echo "$SINGLE_POD_HOSTS" | tr ' ' '\n' > "$HOSTFILE"

# ---- run the generator ---------------------------------------------------

# Activate the project python_env so 'python' has the right deps (pyyaml, loguru).
PY="$TT_METAL_HOME/python_env/bin/python"
[ -x "$PY" ] || PY=python3

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

echo "[bootstrap] done."
echo "[bootstrap]   rank binding:        $EXPECTED_RB"
echo "[bootstrap]   rank file (bundle):  $EXPECTED_RF"
echo "[bootstrap]   rank file (mpirun):  $RANKFILE_FOR_MPI  (hyphen-free copy)"
echo "[bootstrap]   bundle dir:          $PIPELINE_DIR"
