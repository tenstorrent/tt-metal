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
  --sync-exclude <pat>  Additional rsync --exclude pattern for the --sync-to copy
                        (repeatable). .cpmcache is excluded by default (build-only
                        source cache, not needed to use the built tree).
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
cleanup() { rm -rf "${TMP_ETC}"; }
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
    info "Syncing to ${SYNC_TO} (excluding: ${SYNC_EXCLUDES[*]}) — this is the slow NFS part; expect it to take a while"
  else
    info "Syncing to ${SYNC_TO} (this is the slow NFS part; expect it to take a while)"
  fi
  mkdir -p "${SYNC_TO}"
  sync_start=${SECONDS}
  rsync -a --human-readable "${RSYNC_EXCLUDE_ARGS[@]}" "${SRC}/" "${SYNC_TO}/"
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
