#!/usr/bin/env bash
#
# build_metal_exabox.sh — build tt-metal for Exabox without paying NFS latency.
#
# Multi-node Slurm jobs on Exabox can only see /data (cluster-wide NFS), so the
# documented convention is to clone and build at /data/<user>/tt-metal. But
# /data is very slow for many-small-file I/O: the CPM dependency fetch/extract
# phase alone has been measured at 18+ minutes of almost pure I/O wait, versus
# 7-12 minutes for a *complete* build on fast storage.
#
# Building on fast disk and copying to /data afterwards does NOT work, because
# tt-metal bakes absolute build paths into its artifacts:
#   * tt_metal/CMakeLists.txt sets INSTALL_RPATH from ${PROJECT_BINARY_DIR}
#   * create_venv.sh writes absolute-path shebangs into python_env/bin/*
# Move the tree afterwards and you break dynamic linking and the venv.
#
# THE TRICK: build inside a container where a fast *local* directory is
# bind-mounted at the exact absolute path the build must eventually live at.
# CMake, the RPATHs, and the venv shebangs all see (and bake) that path while
# every byte is actually read/written on local disk. The result is NOT yet
# visible to other Slurm nodes; the opt-in --sync-to flag rsyncs it to the real
# NFS path afterwards — the one unavoidably slow step, so you choose when to
# pay for it.
#
# USAGE
#   mkdir -p /scratch/$USER && cd /scratch/$USER
#   git clone --recursive https://github.com/tenstorrent/tt-metal
#   cd tt-metal
#   ./tools/scaleout/exabox/build_metal_exabox.sh --sync-to /data/$USER/tt-metal
#
# ccache is deliberately NOT wired up yet: the image ships ccache (and
# ENV CCACHE_TEMPDIR=/tmp/ccache) but nothing here sets CCACHE_DIR /
# CCACHE_BASEDIR. Deferred, not forgotten.
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

# .cpmcache is CMake's cache of downloaded third-party sources, read only at
# build time; nothing at runtime references it, so it's dead weight on a sync.
SYNC_EXCLUDES=(".cpmcache")

# Parallel rsyncs for --sync-to. rsync is single-threaded per invocation and the
# destination is latency-bound, so a few concurrent jobs is what helps.
# Deliberately NOT derived from nproc: the shared resource is the NFS server,
# not this machine's CPUs.
SYNC_JOBS=4

# Soft warning threshold for free space on the local scratch filesystem (GiB).
MIN_FREE_GIB=50

usage() {
  cat <<EOF
Usage: ./tools/scaleout/exabox/build_metal_exabox.sh [options] [-- <build_metal.sh args>]

Builds tt-metal in a container, bind-mounting the current directory (which
should be a tt-metal clone on FAST LOCAL DISK, e.g. /scratch/\$USER/tt-metal) so it
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
  --entrypoint <path>   Override the image entrypoint. Must be a shell that
                        accepts -c (e.g. /bin/bash); the build script is handed
                        to it as -c <script>. Not needed for the default
                        ci-build image (which declares none); for variants that
                        do set one, e.g. dev or release.
  --sync-to <path>      After a successful build, rsync the tree to <path> on
                        the real filesystem (i.e. the actual NFS mount). Must be
                        identical to --target-path or the script refuses, because
                        any other destination invalidates the baked-in RPATHs.
  --sync-exclude <pat>  Additional rsync --exclude pattern for the --sync-to copy
                        (repeatable). .cpmcache is excluded by default (build-only
                        source cache, not needed to use the built tree).
  --sync-jobs <N>       Parallel rsync jobs for the --sync-to copy (default: ${SYNC_JOBS}).
  --dry-run             Print the docker command that would run, then exit.
  -h, --help            Show this help.

Everything after a literal '--' is forwarded verbatim to build_metal.sh, e.g.
  ... build_metal_exabox.sh -- --build-type Release
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

# $HOME inside the container. Exabox convention is HOME=/data/<user>, i.e. the
# parent of the clone. Back it with its own writable bind mount: left as a bare
# mountpoint parent, Docker would create it root-owned and pip/CPM/git writes
# into $HOME would fail for a non-root user.
CONTAINER_HOME="$(dirname "${TARGET_PATH}")"
if [[ "${CONTAINER_HOME}" == "/" || -z "${CONTAINER_HOME}" ]]; then
  # Degenerate target like /tt-metal: fall back to HOME == the build tree
  # (works, just leaves dotfiles there).
  CONTAINER_HOME="${TARGET_PATH}"
fi

if [[ -z "${HOME_DIR}" ]]; then
  HOME_DIR="$(dirname "${SRC}")/.exabox-home"
fi
# Docker treats a relative -v source as a named VOLUME, not a path, so a
# relative --home-dir would silently mount something other than the directory
# this script creates below.
[[ "${HOME_DIR}" == /* ]] || die "--home-dir must be absolute (got '${HOME_DIR}')"

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

# Building from network storage defeats the point — warn loudly but don't block
# (someone may be deliberately testing).
if command -v stat >/dev/null 2>&1; then
  fstype="$(stat -f -c %T "${SRC}" 2>/dev/null || echo unknown)"
  case "${fstype}" in
    nfs*|autofs|cifs|smb*)
      warn "source directory is on '${fstype}' — this script exists to AVOID building on \
network storage. Clone to local scratch (e.g. /scratch/${UNAME}/tt-metal) instead." ;;
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
  # Checked now so a typo fails before the build, not after it.
  [[ "${SYNC_JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--sync-jobs must be a positive integer (got '${SYNC_JOBS}')."
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
# The ci-build image declares no USER, so it runs as root; building as root
# would leave root-owned artifacts on the host. We run as the invoking UID:GID
# instead — which breaks anything calling getpwuid() ("I have no name!", git,
# $HOME lookups) unless that UID has a passwd entry in the container.
#
# Bind-mounting the host's /etc/passwd (the usual fix) is not enough here:
# cluster login nodes typically serve users via LDAP/SSSD, so the invoking user
# may not appear in the literal file at all. So we resolve identity via NSS
# (id/getent) and synthesize an entry, with its home field set to the
# *container* home so getpwuid() agrees with the HOME env var. We base the file
# on the image's own /etc/passwd where possible so its system accounts survive.
TMP_ETC="$(mktemp -d)"
cleanup() { rm -rf "${TMP_ETC}"; }
trap cleanup EXIT

fetch_from_image() { # $1 = file path inside image
  # --entrypoint cat in case --image/--tag points at an image that sets an
  # ENTRYPOINT (the default ci-build image does not).
  docker run --rm --entrypoint cat "${IMAGE_REF}" "$1" 2>/dev/null
}

base_passwd=""
base_group=""
if (( ! DRY_RUN )); then
  base_passwd="$(fetch_from_image /etc/passwd || true)"
  base_group="$(fetch_from_image /etc/group || true)"
fi
# Fall back to the host's files, then to a minimal skeleton.
[[ -n "${base_passwd}" ]] || base_passwd="$(cat /etc/passwd 2>/dev/null || echo 'root:x:0:0:root:/root:/bin/bash')"
[[ -n "${base_group}"  ]] || base_group="$(cat /etc/group  2>/dev/null || echo 'root:x:0:')"

# Drop any line colliding with our user/uid (glibc uses the FIRST match, so a
# stale entry would win over ours), then append the entry we want.
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
# Fully single-quoted heredoc: nothing expands on the host; everything the
# script needs arrives as an environment variable.
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

# Belt-and-braces: the tree is owned by this UID, but set safe.directory so
# nested submodule paths can never trip git's ownership check. Writes to the
# throwaway per-build $HOME, not the user's real dotfiles.
git config --global --add safe.directory '*' >/dev/null 2>&1 || true

# tt-metal's scripts derive this, but set it explicitly: by construction the
# repo root IS the target path.
export TT_METAL_HOME="$(pwd)"

# Forwarded `-- ...` arguments for build_metal.sh, printf %q-quoted on the host.
eval "set -- ${EXABOX_BUILD_ARGS:-}"

# Defensive: `git clone --recursive` on the host should already have done this;
# re-check rather than assume.
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

# The image ships a pre-built venv at /opt/venv; we deliberately build our own
# in the checkout instead, since the image's venv is pinned to whatever
# tt-metal commit the image was built from, which we cannot verify against
# this clone.
#
# --bundle-python deep-copies the interpreter into the venv instead of
# symlinking it: a symlinked venv resolves to an interpreter that may not exist
# on other nodes once the tree is on shared NFS ("python3: not found").
# See https://github.com/tenstorrent/tt-blaze/issues/1516
#
# IMPORTANT: the ci-build image bakes ENV PYTHON_ENV_DIR=/opt/venv, and
# create_venv.sh defaults to that env var — without an explicit --env-dir it
# tries to overwrite /opt/venv and fails (not writable for the mapped host
# UID). Keep both the --env-dir flag and the unset below; removing either
# reintroduces that bug.
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
# Only override the entrypoint when asked; the default image declares none.
if [[ -n "${ENTRYPOINT}" ]]; then DOCKER_ARGS+=(--entrypoint "${ENTRYPOINT}"); fi

DOCKER_ARGS+=(
  --user "${UID_N}:${GID_N}"
  -v "${TMP_ETC}/passwd:/etc/passwd:ro"
  -v "${TMP_ETC}/group:/etc/group:ro"
)

# Docker mounts destinations parent-first, so mounting the home dir at
# ${CONTAINER_HOME} and the source at ${CONTAINER_HOME}/<name> nests correctly
# (the source mount is NOT shadowed).
if [[ "${CONTAINER_HOME}" != "${TARGET_PATH}" ]]; then
  DOCKER_ARGS+=(-v "${HOME_DIR}:${CONTAINER_HOME}")
fi

DOCKER_ARGS+=(
  -v "${SRC}:${TARGET_PATH}"
  --workdir "${TARGET_PATH}"
  # HOME set explicitly so tools reading the env var and tools calling
  # getpwuid() agree.
  -e "HOME=${CONTAINER_HOME}"
  -e "USER=${UNAME}"
  -e "LOGNAME=${UNAME}"
  -e "EXABOX_BUILD_ARGS=${BUILD_ARGS_Q}"
  # NOTE: ccache intentionally not configured yet (see header).
  # NOTE: if your tt-metal revision still requires ARCH_NAME, add it here.
  "${IMAGE_REF}"
)
# Everything after the image name becomes ARGUMENTS to an overridden
# entrypoint, so with --entrypoint set the shell named there gets just
# -c <script> — a leading 'bash' token would be read by it as a script
# *filename*. Without an override, this is the whole command.
if [[ -n "${ENTRYPOINT}" ]]; then
  DOCKER_ARGS+=(-c "${CONTAINER_SCRIPT}")
else
  DOCKER_ARGS+=(bash -c "${CONTAINER_SCRIPT}")
fi

cat <<EOF

  source (host, fast local):  ${SRC}
  appears in container as:    ${TARGET_PATH}
  container HOME:             ${CONTAINER_HOME}  (backed by ${HOME_DIR})
  image:                      ${IMAGE_REF}
  build_metal.sh args:        ${BUILD_ARGS_Q:-<none>}
  sync-to after build:        ${SYNC_TO:-<disabled>}

EOF

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
if [[ -n "${SYNC_TO}" ]]; then
  RSYNC_EXCLUDE_ARGS=()
  for pat in "${SYNC_EXCLUDES[@]}"; do
    RSYNC_EXCLUDE_ARGS+=(--exclude="${pat}")
  done
  if (( ${#SYNC_EXCLUDES[@]} > 0 )); then
    info "Syncing to ${SYNC_TO} with ${SYNC_JOBS} parallel rsync jobs (excluding: ${SYNC_EXCLUDES[*]}) — this is the slow NFS part; expect it to take a while"
  else
    info "Syncing to ${SYNC_TO} with ${SYNC_JOBS} parallel rsync jobs — this is the slow NFS part; expect it to take a while"
  fi
  mkdir -p "${SYNC_TO}"
  sync_start=${SECONDS}
  # One rsync per top-level entry, ${SYNC_JOBS} at a time. Sources have no
  # trailing slash, so directories copy recursively and plain files copy as
  # themselves, and --exclude patterns mean the same as in a single rsync of
  # the whole tree — including for the top-level entries themselves: rsync
  # applies filters to explicitly named source arguments too (only trailing-
  # slash "dot dir" sources are exempt), so e.g. the .cpmcache job transfers
  # nothing. --delete makes reruns converge instead of accreting files that
  # were since removed or renamed in ${SRC}. Syncs in place: a concurrent
  # reader on another node could see a partially-written tree mid-sync.
  find "${SRC}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${SYNC_JOBS}" -I{} rsync -a --delete --human-readable "${RSYNC_EXCLUDE_ARGS[@]}" "${SRC}/{}" "${SYNC_TO}/" \
    || die "sync to ${SYNC_TO} failed; re-run to continue (rsync skips what already matches)."
  # --delete only reaches entries that still exist in ${SRC}; a top-level
  # entry removed or renamed since an earlier sync never gets a job at all,
  # so its stale copy is reaped here. Excluded names are left alone, matching
  # rsync's own rule that --exclude protects destination paths from --delete.
  while IFS= read -r entry; do
    [[ -e "${SRC}/${entry}" ]] && continue
    for pat in "${SYNC_EXCLUDES[@]}"; do
      [[ "${entry}" == ${pat} ]] && continue 2
    done
    rm -rf -- "${SYNC_TO}/${entry}"
  done < <(find "${SYNC_TO}" -mindepth 1 -maxdepth 1 -printf '%f\n')
  info "Sync finished in $(( (SECONDS - sync_start) / 60 ))m $(( (SECONDS - sync_start) % 60 ))s"
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
