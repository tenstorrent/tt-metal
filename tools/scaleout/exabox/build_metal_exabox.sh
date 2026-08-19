#!/usr/bin/env bash
#
# build_metal_exabox.sh — build tt-metal for Exabox without paying NFS latency.
#
# WHY THIS EXISTS
# ---------------
# On Exabox, multi-node Slurm jobs can only see /data (cluster-wide NFS), so the
# documented convention is to clone into /data/<user>/tt-metal and build there.
# But /data is very non-performant for many-small-file I/O: the CPM dependency
# fetch/extract phase alone (tidy, usearch, pugixml, boost -> <clone>/.cpmcache)
# has been measured at 18+ minutes wall clock for ~2 minutes of CPU — i.e. almost
# pure I/O wait, versus 7-12 minutes for a *complete* build on faster storage.
#
# The obvious workaround — build on fast local disk, then copy to /data — does
# NOT work, because tt-metal bakes absolute build paths into its artifacts:
#   * tt_metal/CMakeLists.txt sets INSTALL_RPATH from the literal
#     ${PROJECT_BINARY_DIR} (tt-metal's own comment there says:
#     "FIXME: Install RPATH should not have a build path!"), so .so/executables
#     hard-code wherever they were built.
#   * create_venv.sh writes shebangs in python_env/bin/* pointing at the venv's
#     own absolute path.
# Move the tree afterwards and you break dynamic linking and the venv.
#
# Upstream states the same constraint explicitly. From the base stage of
# tt-metal's own Dockerfile, verbatim:
#   "For system installs of uv, UV_PYTHON_INSTALL_DIR ensures Python is in an
#    accessible shared path. When the venv is copied to NFS for multi-host use,
#    the Python interpreter must be reachable from all nodes."
# i.e. path-correctness-at-build-time is a known, acknowledged property of this
# toolchain, not an inference on our part.
#
# THE TRICK
# ---------
# Build inside a container where a *fast local* directory is bind-mounted so it
# appears at the exact absolute path the build must eventually live at. cmake,
# the RPATH it bakes, and the venv shebangs all see /data/<user>/tt-metal and
# record that path — while every byte is actually read/written on local disk.
#
# USAGE (the intended flow)
#   mkdir -p /tmp/$USER && cd /tmp/$USER
#   git clone --recursive https://github.com/tenstorrent/tt-metal
#   cd tt-metal
#   ./tools/scaleout/exabox/build_metal_exabox.sh
#
# SCOPE / KNOWN LIMITATION (read this)
# ------------------------------------
# By default this script finishes with a complete build whose baked-in paths say
# /data/<user>/tt-metal, but whose bytes still live on local scratch. That build
# is NOT yet visible to other Slurm nodes, so it is NOT yet usable for multi-node
# jobs. Because the baked paths are already correct for the real destination, an
# rsync to that exact same path is safe and is the intended last mile — that is
# what the opt-in --sync-to flag does. It is opt-in (and refuses any destination
# other than --target-path) precisely because that copy is the one unavoidably
# slow NFS operation, and you should choose when to pay for it.
#
# ccache is deliberately NOT wired up in this first draft (no CCACHE_DIR /
# CCACHE_BASEDIR). Deferred, not forgotten. Note that the tool itself is already
# present in the default image — it has a dedicated ccache-layer and ships
# ENV CCACHE_TEMPDIR=/tmp/ccache — so enabling this later is purely a question of
# which env vars/flags to set (and how CCACHE_BASEDIR should interact with the
# path remapping above), not "does the container even have ccache".
#
# FUTURE DIRECTION (not implemented, do not action from this comment)
# tt-metal's release/release-models image targets build their venv with
# `uv venv --relocatable` plus a patch_activate_posix.sh — a genuinely
# relocatable venv mechanism that would sidestep the shebang problem entirely,
# but which is not exposed through the dev-facing create_venv.sh (whose only
# relevant option is --bundle-python). Worth asking upstream whether it can be;
# unverified outside the release-image context.
#
# See also: TROUBLESHOOTING.md in this directory.

set -euo pipefail

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_IMAGE="ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-ci-build-amd64"
DEFAULT_TAG="latest"

IMAGE="${DEFAULT_IMAGE}"
TAG="${DEFAULT_TAG}"
TARGET_PATH=""          # default computed below, needs whoami
HOME_DIR=""             # host dir backing $HOME inside the container
ENTRYPOINT=""           # empty => use the image's own entrypoint (default image has none)
SYNC_TO=""              # opt-in rsync destination (must equal TARGET_PATH)
DRY_RUN=0
EXTRA_BUILD_ARGS=()     # everything after `--`, forwarded to build_metal.sh

# .cpmcache is CMake Package Manager's cache of downloaded/extracted third-party
# sources (tidy, usearch, pugixml, boost, ...), consulted only during the build
# itself to avoid re-fetching from the network. Nothing at runtime references it
# -- RPATHs point at build*/lib, not .cpmcache -- so it's dead weight on a sync
# meant to make the build usable elsewhere, not to let someone keep incrementally
# rebuilding from the synced copy. Excluded by default; add more with
# --sync-exclude (repeatable) if you find other build-only bulk worth dropping.
SYNC_EXCLUDES=(".cpmcache")

# How many concurrent rsync workers the --sync-to copy fans out over. The
# destination is latency-bound (many synchronous metadata round trips per file),
# not bandwidth-bound, and rsync is single-threaded per invocation, so overlapping
# several is the only lever that addresses the actual bottleneck. 4 is a
# deliberately modest default: /data is shared by the whole Exabox pool and the
# server has finite nfsd threads and RPC slots, so hammering it degrades the mount
# for everyone else. Returns flatten well before the warn threshold anyway.
SYNC_JOBS=4
SYNC_JOBS_WARN=8

# 0 = build a fresh copy at ${SYNC_TO}.new and atomically rename it into place
#     (default; see the big comment on the sync step for why).
# 1 = legacy behavior: rsync straight over ${SYNC_TO}. Slower on a populated
#     destination and briefly visible to readers in a half-written state.
SYNC_IN_PLACE=0

# Host-local scratch for sync bookkeeping (worker failure list). Removed by the
# EXIT trap; declared here so cleanup() can reference it unconditionally.
SYNC_STATE_DIR=""

# Soft warning threshold for free space on the local scratch filesystem (GiB).
MIN_FREE_GIB=50

usage() {
  cat <<EOF
Usage: ./tools/scaleout/exabox/build_metal_exabox.sh [options] [-- <build_metal.sh args>]

Builds tt-metal in a container, bind-mounting the current directory (which
should be a tt-metal clone on FAST LOCAL DISK, e.g. /tmp/\$USER/tt-metal) so it
appears at the slow-but-canonical Exabox path, so RPATHs and venv shebangs are
baked correctly for that destination.

Options:
  --image <name>        Docker image name
                        (default: ${DEFAULT_IMAGE})
  --tag <tag>           Docker image tag (default: ${DEFAULT_TAG})
  --target-path <path>  Absolute path the build must believe it lives at
                        (default: /data/\$(whoami)/tt-metal)
  --home-dir <path>     Host directory to back \$HOME inside the container
                        (default: <parent of this clone>/.exabox-home)
  --entrypoint <path>   Override the image entrypoint. Not needed for the default
                        ci-build image, which declares no ENTRYPOINT (ENTRYPOINT
                        first appears at the dev-light stage, which ci-build does
                        not inherit from). Provided for pointing --image/--tag at
                        a variant that does set one, e.g. dev or release.
  --sync-to <path>      After a successful build, rsync the tree to <path> on
                        the real filesystem (i.e. the actual NFS mount). Must be
                        identical to --target-path or the script refuses, because
                        any other destination invalidates the baked-in RPATHs.
                        Builds a fresh copy at <path>.new and renames it into
                        place, so readers never see a partial tree. NOTE: this
                        needs room for two copies of the tree on the destination
                        until the swap completes.
  --sync-exclude <pat>  Additional rsync --exclude pattern for the --sync-to copy
                        (repeatable). .cpmcache is excluded by default (build-only
                        source cache, not needed to use the built tree). Patterns
                        behave exactly as they would in a single whole-tree rsync,
                        anchored ("/foo") ones included.
  --sync-jobs <N>       Concurrent rsync workers for the --sync-to copy
                        (default: ${SYNC_JOBS}). Warns above ${SYNC_JOBS_WARN}: /data is shared,
                        and saturating it is antisocial for little extra speed.
                        N=1 does a single whole-tree rsync.
  --sync-in-place       Skip the temp-dir-and-rename dance and rsync directly over
                        <path>. Only for when you cannot spare the transient 2x
                        space. Slower on a populated destination, and concurrent
                        readers on other nodes may see a half-written tree.
  --dry-run             Print the docker command that would run, then exit.
  -h, --help            Show this help.

Everything after a literal '--' is forwarded verbatim to build_metal.sh, e.g.
  ... build_metal_exabox.sh -- --build-type Release --enable-ccache
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
info() { printf '==> %s\n' "$*"; }

# ----------------------------------------------------------------------------
# Argument parsing (supports both "--flag value" and "--flag=value")
# ----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  arg="$1"
  val=""
  if [[ "$arg" == *=* && "$arg" == --* ]]; then
    val="${arg#*=}"
    arg="${arg%%=*}"
    set -- "$arg" "$val" "${@:2}"
  fi
  case "$1" in
    --image)       [[ $# -ge 2 ]] || die "--image needs a value";       IMAGE="$2"; shift 2 ;;
    --tag)         [[ $# -ge 2 ]] || die "--tag needs a value";         TAG="$2"; shift 2 ;;
    --target-path) [[ $# -ge 2 ]] || die "--target-path needs a value"; TARGET_PATH="$2"; shift 2 ;;
    --home-dir)    [[ $# -ge 2 ]] || die "--home-dir needs a value";    HOME_DIR="$2"; shift 2 ;;
    --entrypoint)  [[ $# -ge 2 ]] || die "--entrypoint needs a value";  ENTRYPOINT="$2"; shift 2 ;;
    --sync-to)     [[ $# -ge 2 ]] || die "--sync-to needs a value";     SYNC_TO="$2"; shift 2 ;;
    --sync-exclude) [[ $# -ge 2 ]] || die "--sync-exclude needs a value"; SYNC_EXCLUDES+=("$2"); shift 2 ;;
    --sync-jobs)   [[ $# -ge 2 ]] || die "--sync-jobs needs a value";   SYNC_JOBS="$2"; shift 2 ;;
    --sync-in-place) SYNC_IN_PLACE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; EXTRA_BUILD_ARGS=("$@"); break ;;
    *)             usage >&2; die "unknown argument: $1" ;;
  esac
done

# ----------------------------------------------------------------------------
# Identity and paths
# ----------------------------------------------------------------------------
UID_N="$(id -u)"
GID_N="$(id -g)"
UNAME="$(id -un)"
GNAME="$(id -gn)"

[[ "${UID_N}" != "0" ]] || warn "you are running as root on the host; container output will be root-owned."

SRC="$(pwd -P)"
: "${TARGET_PATH:=/data/${UNAME}/tt-metal}"

[[ "${TARGET_PATH}" == /* ]]  || die "--target-path must be absolute (got '${TARGET_PATH}')"
[[ "${TARGET_PATH}" != "/" ]] || die "--target-path must not be /"
TARGET_PATH="${TARGET_PATH%/}"

# $HOME inside the container. The doc convention on Exabox is HOME=/data/<user>,
# i.e. the *parent* of the clone. We honour that, but back it with its own writable
# bind mount: if we left it as a bare mountpoint parent, Docker would create it
# root-owned and pip/CPM/git writes into $HOME would fail for a non-root user.
CONTAINER_HOME="$(dirname "${TARGET_PATH}")"
if [[ "${CONTAINER_HOME}" == "/" || -z "${CONTAINER_HOME}" ]]; then
  # Degenerate target like /tt-metal: nowhere sane to put a separate HOME, so
  # fall back to HOME == the build tree itself (works, just leaves dotfiles there).
  CONTAINER_HOME="${TARGET_PATH}"
fi

if [[ -z "${HOME_DIR}" ]]; then
  HOME_DIR="$(dirname "${SRC}")/.exabox-home"
fi

IMAGE_REF="${IMAGE}:${TAG}"

# ----------------------------------------------------------------------------
# Fail-fast sanity checks (cheap ones before anything expensive)
# ----------------------------------------------------------------------------
info "Sanity-checking the working directory"
for f in build_metal.sh create_venv.sh; do
  [[ -f "${SRC}/${f}" ]] || die "'${SRC}' does not look like a tt-metal checkout (missing ${f}). \
Run this script from the root of your clone."
done
[[ -x "${SRC}/build_metal.sh" ]] || warn "build_metal.sh is not executable; will invoke via bash."

if [[ "${SRC}" == "${TARGET_PATH}" ]]; then
  warn "the clone is already at the target path (${SRC}); the bind mount is a no-op \
and you will be building directly on that filesystem."
fi

# Are we actually on fast local storage? If the source is already NFS, the whole
# point of this script is defeated — warn loudly but don't block (someone may be
# deliberately testing).
if command -v stat >/dev/null 2>&1; then
  fstype="$(stat -f -c %T "${SRC}" 2>/dev/null || echo unknown)"
  case "${fstype}" in
    nfs*|autofs|cifs|smb*)
      warn "source directory is on '${fstype}' — this script exists to AVOID building on \
network storage. Clone to local scratch (e.g. /tmp/${UNAME}/tt-metal) instead." ;;
    tmpfs)
      warn "source directory is on tmpfs (RAM). A tt-metal build tree is tens of GB; \
make sure you actually have that much memory." ;;
  esac
fi

# Free-space check (soft).
if command -v df >/dev/null 2>&1; then
  avail_kb="$(df -P "${SRC}" 2>/dev/null | awk 'NR==2 {print $4}')" || avail_kb=""
  if [[ -n "${avail_kb}" ]] && (( avail_kb < MIN_FREE_GIB * 1024 * 1024 )); then
    warn "only $(( avail_kb / 1024 / 1024 )) GiB free on the filesystem holding ${SRC}; \
a full tt-metal build tree typically wants >= ${MIN_FREE_GIB} GiB."
  fi
fi

info "Checking Docker"
command -v docker >/dev/null 2>&1 || die "docker not found in PATH."
docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon as $(id -un). \
This script deliberately does not use sudo (that would defeat --user and produce \
root-owned build output). Make sure you are in the 'docker' group and the daemon is up."

if [[ -n "${SYNC_TO}" ]]; then
  SYNC_TO="${SYNC_TO%/}"
  if [[ "${SYNC_TO}" != "${TARGET_PATH}" ]]; then
    die "--sync-to '${SYNC_TO}' != --target-path '${TARGET_PATH}'.
The build hard-codes '${TARGET_PATH}' into RPATHs and venv shebangs, so placing the
tree anywhere else produces a subtly broken build (dynamic linking and python_env
will fail at runtime). Either sync to '${TARGET_PATH}', or re-run the build with
--target-path '${SYNC_TO}'."
  fi
  command -v rsync >/dev/null 2>&1 || die "--sync-to requires rsync, which was not found."

  [[ "${SYNC_JOBS}" =~ ^[0-9]+$ ]] || die "--sync-jobs must be a positive integer (got '${SYNC_JOBS}')."
  (( SYNC_JOBS >= 1 )) || die "--sync-jobs must be >= 1."
  if (( SYNC_JOBS > SYNC_JOBS_WARN )); then
    warn "--sync-jobs ${SYNC_JOBS} exceeds the recommended ${SYNC_JOBS_WARN}. /data is shared by the \
whole Exabox pool and the server has finite nfsd threads and RPC slots; this can degrade \
the mount for other users for little additional speed. Proceeding anyway."
  fi
  # The concurrency gate below uses `wait -n`, which needs bash >= 4.3.
  if (( SYNC_JOBS > 1 )) \
     && (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    warn "bash ${BASH_VERSION} lacks 'wait -n' (need >= 4.3); falling back to --sync-jobs 1."
    SYNC_JOBS=1
  fi
fi

# ----------------------------------------------------------------------------
# Pull the image (explicitly, so a stale local copy is never silently reused)
# ----------------------------------------------------------------------------
if (( DRY_RUN )); then
  info "[dry-run] would: docker pull ${IMAGE_REF}"
else
  info "Pulling ${IMAGE_REF}"
  docker pull "${IMAGE_REF}" || die "docker pull failed for ${IMAGE_REF}. \
If this image is private, run 'docker login ghcr.io' first."
fi

# ----------------------------------------------------------------------------
# Non-root user support: synthesize /etc/passwd and /etc/group for the container
# ----------------------------------------------------------------------------
# This image runs as root: its ci-build target inherits base -> ci-build-light ->
# ci-build and none of those stages declares a USER, so Docker's root default
# applies. Building as root would leave root-owned artifacts on the host, so we
# run the container as the invoking host UID:GID instead.
#
# The classic failure mode of `--user <uid>:<gid>` is that the UID has no passwd
# entry inside the container, so anything calling getpwuid() ("I have no name!",
# git's "unable to look up current user", tools resolving $HOME from the passwd
# DB) breaks.
#
# The usual fix is to bind-mount the host's /etc/passwd and /etc/group read-only.
# We do something slightly stronger, for two reasons:
#   (1) On cluster login nodes, users are frequently served by LDAP/SSSD and are
#       NOT present in the literal /etc/passwd file — bind-mounting it would
#       therefore not fix the problem at all. We resolve via getent (which does
#       consult NSS) and synthesize an entry.
#   (2) We want the passwd entry's home field to be the *container* home
#       (${CONTAINER_HOME}), not the host home, so getpwuid-based lookups agree
#       with the HOME env var instead of pointing at a path that does not exist
#       in the container.
# We base the file on the *image's* own /etc/passwd where possible so that the
# image's system accounts survive, then append our user.
TMP_ETC="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_ETC}"
  # Sync bookkeeping scratch, if the sync step got as far as creating it.
  [[ -n "${SYNC_STATE_DIR:-}" ]] && rm -rf "${SYNC_STATE_DIR}"
  return 0
}
trap cleanup EXIT

fetch_from_image() { # $1 = file path inside image
  # --entrypoint cat is belt-and-braces: the default ci-build image declares no
  # ENTRYPOINT, but --image/--tag may point somewhere that does.
  docker run --rm --entrypoint cat "${IMAGE_REF}" "$1" 2>/dev/null
}

base_passwd=""
base_group=""
if (( ! DRY_RUN )); then
  base_passwd="$(fetch_from_image /etc/passwd || true)"
  base_group="$(fetch_from_image /etc/group || true)"
fi
# Fall back to the host's files (the documented approach) if we could not read
# the image's, and finally to a minimal skeleton.
[[ -n "${base_passwd}" ]] || base_passwd="$(cat /etc/passwd 2>/dev/null || echo 'root:x:0:0:root:/root:/bin/bash')"
[[ -n "${base_group}"  ]] || base_group="$(cat /etc/group  2>/dev/null || echo 'root:x:0:')"

# Drop any line that collides with our user/uid (glibc uses the FIRST match, so a
# stale colliding entry would win over ours), then append the entry we want.
printf '%s\n' "${base_passwd}" \
  | awk -F: -v u="${UNAME}" -v uid="${UID_N}" '$1 != u && $3 != uid' > "${TMP_ETC}/passwd"
printf '%s:x:%s:%s:%s:%s:/bin/bash\n' \
  "${UNAME}" "${UID_N}" "${GID_N}" "exabox build user" "${CONTAINER_HOME}" >> "${TMP_ETC}/passwd"

printf '%s\n' "${base_group}" \
  | awk -F: -v g="${GNAME}" -v gid="${GID_N}" '$1 != g && $3 != gid' > "${TMP_ETC}/group"
printf '%s:x:%s:\n' "${GNAME}" "${GID_N}" >> "${TMP_ETC}/group"

chmod 0755 "${TMP_ETC}"
chmod 0644 "${TMP_ETC}/passwd" "${TMP_ETC}/group"

# ----------------------------------------------------------------------------
# Host-side directories that back the mounts
# ----------------------------------------------------------------------------
if [[ "${CONTAINER_HOME}" != "${TARGET_PATH}" ]]; then
  mkdir -p "${HOME_DIR}" || die "could not create home directory '${HOME_DIR}' (override with --home-dir)."
fi

# ----------------------------------------------------------------------------
# The build script that runs *inside* the container
# ----------------------------------------------------------------------------
# Fully single-quoted heredoc: nothing is expanded here on the host. Everything
# it needs arrives as an environment variable, which avoids a layer of quoting
# bugs.
CONTAINER_SCRIPT="$(cat <<'INNER_EOF'
set -euo pipefail

echo "==> Container identity: $(id)"
echo "==> HOME=${HOME}  PWD=$(pwd)"

# Verify the passwd shim actually worked before doing 20 minutes of work.
if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  echo "ERROR: uid $(id -u) still has no passwd entry inside the container." >&2
  echo "       Tools using getpwuid() (git, python, pip) will misbehave." >&2
  exit 1
fi
if [ ! -w "${HOME}" ]; then
  echo "ERROR: HOME=${HOME} is not writable inside the container." >&2
  exit 1
fi

# The bind-mounted tree is owned by this same UID, so git ownership checks should
# pass; set safe.directory anyway so nested submodule paths can never trip it.
# This writes to ${HOME}/.gitconfig, which is a throwaway per-build home, not the
# user's real dotfiles and not inside the source tree.
git config --global --add safe.directory '*' >/dev/null 2>&1 || true

# tt-metal's own scripts derive this, but set it explicitly: by construction the
# repo root IS the target path, which is the whole point of the bind mount.
export TT_METAL_HOME="$(pwd)"

# Forwarded `-- ...` arguments for build_metal.sh, printf %q-quoted on the host.
eval "set -- ${EXABOX_BUILD_ARGS:-}"

# Defensive submodule init. `git clone --recursive` on the host should already
# have done this; mirror tt-blaze's install.sh and re-check rather than assume.
if git rev-parse --git-dir >/dev/null 2>&1; then
  if git submodule status --recursive 2>/dev/null | grep -qE '^-'; then
    echo "==> Uninitialized submodules found; running git submodule update --init --recursive"
    git submodule update --init --recursive
  else
    echo "==> Submodules already initialized"
  fi
else
  echo "WARNING: not a git working tree; skipping submodule check." >&2
fi

echo "==> Running build_metal.sh $*"
bash ./build_metal.sh "$@"

# NOTE: this image already ships a pre-built venv at /opt/venv (built by a
# separate Dockerfile.python / ci-build-venv-layer, world-writable via umask 000,
# and already on PATH/VIRTUAL_ENV). It is intentionally unused here: it is a
# different venv from the <clone>/python_env one create_venv.sh builds inside the
# mount, and the two do not conflict. Reusing it could be a future speedup, but
# only if the image's pinned tt-metal commit matches the user's clone — which
# this script has no way to verify — so v1 always builds its own.
#
# --bundle-python deep-copies the interpreter into the venv instead of symlinking
# it. Same default (and same order) as tt-blaze's install.sh, for the same reason:
# a symlinked venv resolves to an interpreter that may not exist on other nodes and
# dies with "python3: not found" once the tree is shared.
# See https://github.com/tenstorrent/tt-blaze/issues/1516
#
# IMPORTANT: the ci-build image bakes ENV PYTHON_ENV_DIR=/opt/venv (pointing at
# its own pre-built venv from a separate Dockerfile.python layer). create_venv.sh
# defaults to that env var when --env-dir is not passed, so without this it tries
# to rebuild/overwrite the image's venv at /opt/venv instead of inside our
# checkout — and fails, since /opt/venv isn't writable for the mapped host UID.
# Explicit --env-dir always overrides the env var (per create_venv.sh --help), so
# pass it, and unset the leaking env vars so nothing else picks up /opt/venv either.
unset PYTHON_ENV_DIR VIRTUAL_ENV
echo "==> Running create_venv.sh --bundle-python --env-dir $(pwd)/python_env"
bash ./create_venv.sh --bundle-python --env-dir "$(pwd)/python_env"

echo "==> Build complete inside container"
INNER_EOF
)"

BUILD_ARGS_Q=""
if (( ${#EXTRA_BUILD_ARGS[@]} > 0 )); then
  BUILD_ARGS_Q="$(printf '%q ' "${EXTRA_BUILD_ARGS[@]}")"
fi

# ----------------------------------------------------------------------------
# Assemble and run the container
# ----------------------------------------------------------------------------
DOCKER_ARGS=(run --rm)
if [[ -t 1 ]]; then DOCKER_ARGS+=(-t); fi
# The default ci-build image declares no ENTRYPOINT, so `bash -c ...` below is
# executed directly and nothing needs overriding. Only set when the user asked.
if [[ -n "${ENTRYPOINT}" ]]; then DOCKER_ARGS+=(--entrypoint "${ENTRYPOINT}"); fi

DOCKER_ARGS+=(
  --user "${UID_N}:${GID_N}"
  -v "${TMP_ETC}/passwd:/etc/passwd:ro"
  -v "${TMP_ETC}/group:/etc/group:ro"
)

# Order note: Docker mounts destinations parent-first, so mounting the home dir at
# ${CONTAINER_HOME} and the source at ${CONTAINER_HOME}/<name> nests correctly and
# the source mount is NOT shadowed.
if [[ "${CONTAINER_HOME}" != "${TARGET_PATH}" ]]; then
  DOCKER_ARGS+=(-v "${HOME_DIR}:${CONTAINER_HOME}")
fi

DOCKER_ARGS+=(
  -v "${SRC}:${TARGET_PATH}"
  --workdir "${TARGET_PATH}"
  # HOME is set explicitly rather than relying on the passwd lookup alone: some
  # tools read the env var, some call getpwuid(), and we want both to agree.
  -e "HOME=${CONTAINER_HOME}"
  -e "USER=${UNAME}"
  -e "LOGNAME=${UNAME}"
  -e "EXABOX_BUILD_ARGS=${BUILD_ARGS_Q}"
  # NOTE: ccache intentionally not configured in v1 (no CCACHE_DIR/CCACHE_BASEDIR).
  # The binary IS present in the image (dedicated ccache-layer, ENV
  # CCACHE_TEMPDIR=/tmp/ccache), so this is an env-var/flag decision for v2.
  # NOTE: if your tt-metal revision still requires ARCH_NAME, add it here.
  "${IMAGE_REF}"
  bash -c "${CONTAINER_SCRIPT}"
)

cat <<EOF

  source (host, fast local):  ${SRC}
  appears in container as:    ${TARGET_PATH}
  container HOME:             ${CONTAINER_HOME}  (backed by ${HOME_DIR})
  image:                      ${IMAGE_REF}
  build_metal.sh args:        ${BUILD_ARGS_Q:-<none>}
  sync-to after build:        ${SYNC_TO:-<disabled>}
EOF
if [[ -n "${SYNC_TO}" ]]; then
  cat <<EOF
  sync mode:                  $( (( SYNC_IN_PLACE )) && echo "in-place (non-atomic)" || echo "fresh copy + atomic rename" )
  sync concurrency:           ${SYNC_JOBS} rsync worker(s)
  sync excludes:              ${SYNC_EXCLUDES[*]:-<none>}
EOF
fi
printf '\n'

if (( DRY_RUN )); then
  info "[dry-run] would run:"
  printf 'docker'; printf ' %q' "${DOCKER_ARGS[@]}"; printf '\n'
  exit 0
fi

info "Starting containerized build"
start_ts=${SECONDS}
docker "${DOCKER_ARGS[@]}"
info "Container build finished in $(( (SECONDS - start_ts) / 60 ))m $(( (SECONDS - start_ts) % 60 ))s"

# ----------------------------------------------------------------------------
# Optional last mile: put the tree on the real (slow) shared filesystem
# ----------------------------------------------------------------------------
# Two things make this safer and more deterministic than a plain in-place rsync,
# and (for the fan-out) genuinely faster.
#
# (1) ATOMIC SWAP (default; --sync-in-place opts out)
#     We never rsync over the live tree. We build a complete copy at
#     "${SYNC_TO}.new" and then rename it into place. Two independent reasons:
#
#     * Determinism, plus a performance effect we have NOT cleanly isolated.
#       Writing into a guaranteed-empty .new directory makes every run behave
#       like a first-ever copy: an exact mirror of the source, with no residue
#       from previous syncs and no dependence on what was already there.
#
#       Be careful with the performance story. An earlier version of this comment
#       (and of the commit message that introduced it) claimed that overwriting
#       an existing file pays a write-to-temp-then-rename cost that a brand-new
#       file avoids. That is wrong. Without --inplace, rsync ALWAYS stages a
#       transfer into a temp file (".<name>.XXXXXX" in the destination directory)
#       and renames it into place, whether or not the destination file already
#       existed -- that is its default safety mechanism, and --inplace is the
#       opt-in flag that skips it. So temp+rename cannot distinguish the two
#       cases. Note also that --whole-file is already the default for
#       local-to-local transfers, so rsync does not read existing destination
#       files back to checksum them either.
#
#       What genuinely differs with a populated destination is smaller: (a) the
#       generator's lstat returns attributes instead of ENOENT, and the quick
#       check that follows is a local comparison of data already fetched, not an
#       extra round trip; (b) the final rename replaces an existing directory
#       entry, so the server must also drop the old inode's link count and free
#       its blocks -- real work that a rename into an unused name skips; (c)
#       mutating already-populated directories churns more metadata and
#       invalidates the client's directory/attribute cache more often, which can
#       turn lookups that would have been cache hits into round trips. All real
#       on a latency-bound mount; none obviously a 2x effect.
#
#       The live numbers (42m59s into an empty destination, 95m45s into a
#       populated one) are NOT a controlled experiment: n=1 each, different times
#       of night, and independently observed higher cluster load on the second.
#       The second run also copied strictly LESS data (it had the .cpmcache
#       exclude), which cuts against a purely mechanical explanation. Treat
#       "fresh destination is faster" as plausible and directionally supported,
#       not established; the correctness argument below is what actually carries
#       this design. To settle the perf question, A/B both modes back-to-back in
#       one load window and diff `nfsstat -c` RPC counts around each.
#
#       Honest accounting: the swap is not free. It trades overwrite cost for an
#       `rm -rf` of the old tree -- thousands of NFS unlinks -- but moves that
#       cost AFTER the new tree is live, so it never delays usability.
#
#     * Correctness. Another node reading ${SYNC_TO} while a plain rsync is
#       mid-flight sees a half-written, unusable tree. rename(2) is atomic, so a
#       concurrent reader sees either the whole old tree or the whole new one.
#
#     COST, stated plainly: until the swap completes and .old is removed, the
#     destination filesystem holds up to TWO copies of the build. On a tight
#     /data quota that can fail; --sync-in-place is the escape hatch.
#
#     BEHAVIOR CHANGE vs in-place, worth knowing: the swapped-in tree contains
#     exactly what was copied this run, so anything left at ${SYNC_TO} by an
#     earlier sync but excluded now (e.g. a .cpmcache from a run predating the
#     default exclude) is gone afterwards. That is rsync --delete-like semantics
#     for free, and usually what you want -- but it is a change from the additive
#     in-place behavior, so it is called out rather than discovered.
#
# (2) PARALLEL FAN-OUT (--sync-jobs, default 4)
#     This destination is latency-bound: cost tracks file *count*, because each
#     file needs several synchronous metadata round trips (LOOKUP, CREATE, WRITE,
#     SETATTR, COMMIT). Note what this rules out -- rsync already defaults to
#     --whole-file for local-to-local transfers, so disabling the delta algorithm
#     is a no-op here, and -a does not checksum. There is no flag that fixes this.
#     The only lever is more requests in flight, and rsync is single-threaded per
#     invocation, so we run several concurrently over disjoint top-level entries.
if [[ -n "${SYNC_TO}" ]]; then
  RSYNC_EXCLUDE_ARGS=()
  for pat in "${SYNC_EXCLUDES[@]}"; do
    RSYNC_EXCLUDE_ARGS+=(--exclude="${pat}")
  done
  RSYNC_BASE=(rsync -a --human-readable "${RSYNC_EXCLUDE_ARGS[@]}")

  SYNC_OLD=""
  if (( SYNC_IN_PLACE )); then
    SYNC_DEST="${SYNC_TO}"
    warn "--sync-in-place: writing directly over '${SYNC_TO}'. Concurrent readers on other \
nodes may observe a partially written tree. This path may also be slower than the fresh-copy \
default when the destination is already populated, but that effect is not cleanly measured -- \
see the comment above the sync step."
  else
    SYNC_DEST="${SYNC_TO}.new"
    SYNC_OLD="${SYNC_TO}.old"

    # Clean up leftovers from an interrupted earlier run -- loudly, because a
    # stale .new means a previous sync did not finish and someone may care.
    for stale in "${SYNC_DEST}" "${SYNC_OLD}"; do
      if [[ -e "${stale}" ]]; then
        warn "removing leftover '${stale}' from an earlier interrupted run (this can itself \
take a while on NFS)"
        rm -rf "${stale}" || die "could not remove '${stale}'; clear it manually and re-run."
      fi
    done
  fi

  # Rough space advisory. du's --exclude globbing is not identical to rsync's, so
  # treat the number as approximate; it is only used to warn, never to refuse.
  if command -v du >/dev/null 2>&1 && command -v df >/dev/null 2>&1; then
    du_excl=()
    for pat in "${SYNC_EXCLUDES[@]}"; do du_excl+=(--exclude="${pat}"); done
    src_kb="$(du -sk "${du_excl[@]}" "${SRC}" 2>/dev/null | awk '{print $1}')" || src_kb=""
    dest_avail_kb="$(df -P "$(dirname "${SYNC_TO}")" 2>/dev/null | awk 'NR==2 {print $4}')" || dest_avail_kb=""
    if [[ -n "${src_kb}" && -n "${dest_avail_kb}" ]]; then
      info "About to copy ~$(( src_kb / 1024 / 1024 )) GiB; $(( dest_avail_kb / 1024 / 1024 )) GiB free at $(dirname "${SYNC_TO}")"
      if (( ! SYNC_IN_PLACE )) && (( dest_avail_kb < src_kb )); then
        warn "the atomic-swap path needs room for the new copy alongside whatever is already \
at '${SYNC_TO}' (briefly ~2x), and the destination looks tighter than that. Consider \
--sync-in-place, or free space first."
      fi
    fi
  fi

  if (( ${#SYNC_EXCLUDES[@]} > 0 )); then
    info "Syncing to ${SYNC_TO} via ${SYNC_DEST} (excluding: ${SYNC_EXCLUDES[*]}) — this is the slow NFS part"
  else
    info "Syncing to ${SYNC_TO} via ${SYNC_DEST} — this is the slow NFS part"
  fi

  sync_start=${SECONDS}
  mkdir -p "${SYNC_DEST}" || die "could not create '${SYNC_DEST}'."

  SYNC_STATE_DIR="$(mktemp -d)"
  FAIL_FILE="${SYNC_STATE_DIR}/failures"
  : > "${FAIL_FILE}"

  # One worker. Failures are recorded in FAIL_FILE rather than propagated as an
  # exit status: a background job's status cannot be reliably collected once the
  # concurrency gate below has reaped it with `wait -n`, and O_APPEND writes of
  # short lines from separate processes do not interleave.
  sync_worker() { # $1 = top-level directory name
    local d="$1"
    if "${RSYNC_BASE[@]}" "${SRC}/${d}" "${SYNC_DEST}/"; then
      printf '  [ok]   %s\n' "${d}"
    else
      printf '  [FAIL] %s\n' "${d}" >&2
      printf '%s\n' "${d}" >> "${FAIL_FILE}"
    fi
  }

  if (( SYNC_JOBS <= 1 )); then
    info "Copying whole tree with a single rsync"
    "${RSYNC_BASE[@]}" "${SRC}/" "${SYNC_DEST}/" || die "rsync failed; '${SYNC_DEST}' left for inspection."
  else
    # Partition: one worker per real top-level directory, plus a single
    # non-recursive pass for everything else at the top level (loose files,
    # symlinks). '--exclude=/*/' matches only real directories directly under the
    # transfer root, and `find -type d` (no -L) matches only real directories, so
    # the two halves are exhaustive and non-overlapping -- a symlink-to-directory
    # is carried by the root pass and never by a worker.
    #
    # Note the source form in sync_worker: "${SRC}/${d}" with NO trailing slash.
    # That puts ${d} itself at the top of the worker's transfer, so every path
    # relative to the transfer root is byte-identical to what it would be in a
    # single whole-tree rsync ("python_env/lib/..." either way). That is what
    # makes --sync-exclude patterns behave identically in every worker, anchored
    # ("/foo") patterns included, with no per-worker pattern rewriting. Using the
    # "${SRC}/${d}/" -> "${SYNC_DEST}/${d}/" form instead WOULD shift the anchor
    # root per worker and silently change what anchored patterns mean.
    info "Copying top-level files and symlinks"
    "${RSYNC_BASE[@]}" --exclude='/*/' "${SRC}/" "${SYNC_DEST}/" \
      || die "rsync of top-level entries failed; '${SYNC_DEST}' left for inspection."

    top_dirs=()
    while IFS= read -r -d '' d; do top_dirs+=("${d}"); done \
      < <(find "${SRC}" -mindepth 1 -maxdepth 1 -type d -printf '%f\0' | sort -z)

    # Do not spawn a worker for a top-level directory an exclude kills outright.
    # rsync would correctly copy nothing, but the process and its local scan are
    # pure waste. Exact-name matching only; globs stay rsync's job.
    worker_dirs=()
    for d in "${top_dirs[@]}"; do
      skip=0
      for pat in "${SYNC_EXCLUDES[@]}"; do
        p="${pat#/}"; p="${p%/}"
        if [[ "${d}" == "${p}" ]]; then skip=1; break; fi
      done
      if (( skip )); then
        info "skipping excluded top-level directory: ${d}"
      else
        worker_dirs+=("${d}")
      fi
    done

    info "Copying ${#worker_dirs[@]} top-level directories, ${SYNC_JOBS} at a time"
    running=0
    for d in "${worker_dirs[@]}"; do
      if (( running >= SYNC_JOBS )); then
        wait -n || true
        running=$(( running - 1 ))
      fi
      sync_worker "${d}" &
      running=$(( running + 1 ))
    done
    wait
  fi

  if [[ -s "${FAIL_FILE}" ]]; then
    warn "these top-level entries failed to copy:"
    sed 's/^/  /' "${FAIL_FILE}" >&2
    die "sync incomplete. '${SYNC_DEST}' is being left in place for inspection and was NOT \
swapped in, so '${SYNC_TO}' still holds whatever it held before."
  fi

  copy_elapsed=$(( SECONDS - sync_start ))
  info "Copy finished in $(( copy_elapsed / 60 ))m $(( copy_elapsed % 60 ))s"

  if (( ! SYNC_IN_PLACE )); then
    # .new and .old are siblings of ${SYNC_TO}, so they are guaranteed to be on
    # the same filesystem and each mv is a plain rename(2): atomic, and with no
    # requirement that the target be empty. (That requirement only applies when
    # replacing a directory *in place*, which is precisely why the existing tree
    # is moved aside first rather than being overwritten.)
    info "Swapping '${SYNC_DEST}' into place at '${SYNC_TO}'"
    if [[ -e "${SYNC_TO}" ]]; then
      mv "${SYNC_TO}" "${SYNC_OLD}" \
        || die "could not move existing '${SYNC_TO}' aside. '${SYNC_DEST}' is complete and \
can be swapped in by hand."
    fi
    if ! mv "${SYNC_DEST}" "${SYNC_TO}"; then
      # Never leave nothing at ${SYNC_TO}: put the old tree back.
      if [[ -e "${SYNC_OLD}" ]]; then
        mv "${SYNC_OLD}" "${SYNC_TO}" || warn "rollback of '${SYNC_OLD}' -> '${SYNC_TO}' also failed."
      fi
      die "could not move '${SYNC_DEST}' to '${SYNC_TO}'."
    fi
    # Between those two renames there is a brief window where ${SYNC_TO} does not
    # exist, so a concurrent reader gets ENOENT rather than a corrupt tree.
    # Closing it completely needs renameat2(RENAME_EXCHANGE), which coreutils mv
    # does not expose and NFS may not support.

    if [[ -e "${SYNC_OLD}" ]]; then
      # Unlinking the old tree is itself thousands of NFS round trips. It happens
      # AFTER the swap, so the new build is already live and usable -- interrupting
      # here is safe, it just leaves ${SYNC_OLD} to remove later. Best effort:
      # failing to reclaim space must not fail an otherwise successful sync.
      info "Removing previous tree at '${SYNC_OLD}' (new build is already live; safe to interrupt)"
      old_start=${SECONDS}
      rm -rf "${SYNC_OLD}" \
        || warn "could not fully remove '${SYNC_OLD}'; remove it manually to reclaim space."
      info "Removed in $(( (SECONDS - old_start) / 60 ))m $(( (SECONDS - old_start) % 60 ))s"
    fi
  fi

  sync_total=$(( SECONDS - sync_start ))
  info "Sync finished in $(( sync_total / 60 ))m $(( sync_total % 60 ))s"
  cat <<EOF

Build is now at ${SYNC_TO}, which matches the path baked into its RPATHs and venv
shebangs, so it should be usable from other nodes. Activate with:
    export HOME=${CONTAINER_HOME}
    source ${SYNC_TO}/python_env/bin/activate
EOF
else
  cat <<EOF

Build complete at (container view) ${TARGET_PATH}, physically ${SRC}.

LIMITATION: the bytes are still on local scratch, so this build is not visible to
other Slurm nodes and is not yet usable for multi-node jobs. Its baked-in paths
already say ${TARGET_PATH}, so the correct last step is to copy it to exactly that
path on the shared mount:
    ./tools/scaleout/exabox/build_metal_exabox.sh --sync-to ${TARGET_PATH}
(or just: rsync -a --exclude=.cpmcache ${SRC}/ ${TARGET_PATH}/)
EOF
fi
