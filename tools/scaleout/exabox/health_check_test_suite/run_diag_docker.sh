#!/bin/bash

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Run the Blackhole Galaxy health check (health_check_test_suite/run_diag.sh)
# inside the tt-metal upstream image, with every artifact landing on the host.

set -euo pipefail

# Pinned to the same image the scheduled fleet run uses.
DEFAULT_IMAGE="ghcr.io/tenstorrent/tt-metal/upstream-tests-bh:v0.79.0-dev20260915-25-gb9198bce432"

# The suite as it ships in the image. --entrypoint "" means nothing else sets
# the environment up, so run_diag.sh does it itself off its own location.
RUN_DIAG="/home/user/tt-metal/tools/scaleout/exabox/health_check_test_suite/run_diag.sh"

# tt-syseng-diag, for the QSFP phase. Same defaults as the Ansible role.
DEFAULT_DIAG_PKG_REPO="tenstorrent/tt-syseng-diag-packages"
DEFAULT_DIAG_PKG_VERSION="v0.0.1"

# Where host files handed to the container (input snapshot, QSFP descriptor,
# triage checkout) are mounted read-only.
IN_DIR="/tmp/hc-inputs"

IMAGE="${DEFAULT_IMAGE}"
TIER="medium"
OUTPUT_DIR=""
DIAG_PKG_REPO="${DEFAULT_DIAG_PKG_REPO}"
DIAG_PKG_VERSION="${DEFAULT_DIAG_PKG_VERSION}"
GH_TOKEN_VALUE="${GH_TOKEN:-}"
SKIP_QSFP=false
NEED_LSPCI=true

FORWARD=()
EXTRA_DOCKER=()

say()  { printf '[wrap] %s\n' "$*"; }
warn() { printf '[wrap] WARNING: %s\n' "$*" >&2; }
die()  { printf '[wrap] ERROR: %s\n' "$*" >&2; exit 1; }

show_help() {
    cat <<EOF
Usage: $(basename "$0") [options] [-- <extra run_diag.sh args>]

Runs health_check_test_suite/run_diag.sh inside the tt-metal upstream image and
keeps the JSON report, the per-step logs and the console transcript on the host.

Every option has a default, so a bare \`$(basename "$0")\` runs the medium tier
into ./test_output_<date>-<time>/.

Wrapper options:
  -t, --tier {light|medium|deploy}
                         Tier to run (default: ${TIER}). Also accepted
                         positionally, like run_diag.sh itself.
  -o, --output-dir PATH  Host directory for the report, the per-step logs and
                         the console transcript. Created if absent and chmod
                         0777 so the image user can write to it.
                         (default: ./test_output_<date>-<time>)
  -i, --image IMAGE      Container image, pulled if it is not present locally
                         (default: the pinned upstream-tests-bh tag; see
                         DEFAULT_IMAGE at the top of this script)
      --docker-arg ARG   Extra argument for \`docker run\`, repeatable. Use for
                         one-offs such as --docker-arg --shm-size=4g
  -h, --help             This text

QSFP package (tt-bh-glx-cluster-debug, needed by the QSFP phase):
      --gh-token TOKEN   GitHub token with read access to the private release.
                         Forwarded to the container as GH_TOKEN via a 0600
                         --env-file, so it never appears in the process list.
                         Defaults to \$GH_TOKEN when that is set. Without a
                         token the package is not installed and the QSFP phase
                         reports SKIP; every other check still runs.
      --diag-pkg-repo REPO
                         Release repo (default: ${DEFAULT_DIAG_PKG_REPO})
      --diag-pkg-version TAG
                         Release tag (default: ${DEFAULT_DIAG_PKG_VERSION});
                         empty string skips the install

Forwarded to run_diag.sh (see its --help, and diag_runner.py --help, for detail):
      --dry-run              Print intended subprocess calls without executing
                             destructive steps
      --skip-reset           Skip the reset loop phase entirely
      --skip-tests           Skip the gtest phase entirely
      --skip-triage          Skip the post-test reset and the triage phase
      --triage-gating        Let triage FAILs gate the run (default: held at WARN)
      --triage-dir PATH      Host triage scripts dir; mounted read-only
      --skip-qsfp-tests      Skip the QSFP tests
      --qsfp-gating          Report QSFP findings at their real severity and let
                             them gate the run
      --qsfp-descriptor PATH Host factory_system_descriptor.textproto; mounted
                             read-only
      --qsfp-tool-path PATH  In-container path to the tt-bh-glx-cluster-debug
                             binary
      --input-snapshot PATH  Host tt-smi snapshot JSON; mounted read-only
      --tt-smi-path PATH     In-container tt-smi binary or repo path
      --snapshot-out PATH    In-container path for the raw tt-smi snapshot
                             (default: <output-dir>/diag_snapshot.json)

  Anything after \`--\` is passed to run_diag.sh untouched.

  --output is not forwardable: it is the bind mount this script manages. Note
  that it also wants a *file* path, not a directory — \`--output output/\` is
  what produced "IsADirectoryError: Is a directory" on a manual run.

Host requirements checked before starting:
  /dev/tenstorrent (required), /dev/ipmi0, /dev/hugepages, /dev/hugepages-1G,
  /etc/udev/rules.d, /lib/modules, /sys/kernel/debug. A missing optional path is
  reported with what it costs rather than failing the container outright.

Examples:
  # Medium tier into a timestamped directory
  $(basename "$0")

  # Deploy tier with the QSFP phase, results in a named directory
  $(basename "$0") deploy --output-dir /data/hc-run-1 --gh-token "\$GH_TOKEN"

  # Snapshot-only smoke, no resets and no gtests
  $(basename "$0") light --skip-reset --skip-tests
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

# Host paths that have to be reachable from inside the container. Collected
# during parsing and turned into read-only mounts once, below, so the flag stays
# a host path for the caller and an in-container path for the suite.
HOST_INPUT_SNAPSHOT=""
HOST_QSFP_DESCRIPTOR=""
HOST_TRIAGE_DIR=""

require_value() {
    [[ -n ${2:-} ]] || die "$1 needs a value"
}

while (( $# )); do
    case "$1" in
        -h|--help) show_help; exit 0 ;;

        light|medium|deploy) TIER="$1"; shift ;;
        -t|--tier)
            require_value "$1" "${2:-}"
            case "$2" in
                light|medium|deploy) ;;
                *) die "unknown tier '$2'. Expected: light, medium, deploy" ;;
            esac
            TIER="$2"; shift 2 ;;

        -o|--output-dir)   require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        -i|--image)        require_value "$1" "${2:-}"; IMAGE="$2"; shift 2 ;;
        --docker-arg)      require_value "$1" "${2:-}"; EXTRA_DOCKER+=("$2"); shift 2 ;;

        --gh-token)          require_value "$1" "${2:-}"; GH_TOKEN_VALUE="$2"; shift 2 ;;
        --diag-pkg-repo)     require_value "$1" "${2:-}"; DIAG_PKG_REPO="$2"; shift 2 ;;
        --diag-pkg-version)  DIAG_PKG_VERSION="${2:-}"; shift 2 ;;

        # Forwarded, no value.
        --dry-run)
            # A dry run executes nothing, so the lspci gate below would be
            # gatekeeping a count the suite never takes.
            NEED_LSPCI=false; FORWARD+=("$1"); shift ;;
        --skip-reset)
            NEED_LSPCI=false; FORWARD+=("$1"); shift ;;
        --skip-qsfp-tests)
            SKIP_QSFP=true; FORWARD+=("$1"); shift ;;
        --skip-tests|--skip-triage|--triage-gating|--qsfp-gating)
            FORWARD+=("$1"); shift ;;

        # Forwarded, in-container value.
        --tt-smi-path|--qsfp-tool-path|--snapshot-out)
            require_value "$1" "${2:-}"; FORWARD+=("$1" "$2"); shift 2 ;;

        # Forwarded, host value that needs mounting.
        --input-snapshot)   require_value "$1" "${2:-}"; HOST_INPUT_SNAPSHOT="$2"; shift 2 ;;
        --qsfp-descriptor)  require_value "$1" "${2:-}"; HOST_QSFP_DESCRIPTOR="$2"; shift 2 ;;
        --triage-dir)       require_value "$1" "${2:-}"; HOST_TRIAGE_DIR="$2"; shift 2 ;;

        --output|--output=*)
            die "--output is managed by this script: it is the host directory bind-mounted
       into the container. Use --output-dir instead. (It also takes a file path,
       not a directory — 'run_diag.sh --output output/' is what raises
       IsADirectoryError.)" ;;

        --) shift; FORWARD+=("$@"); break ;;
        *)  die "unknown option '$1'. Try --help" ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Output directory. Bind-mounted at the same path inside the container so the
# log_file entries in diag_report.json read as valid host paths.
# ─────────────────────────────────────────────────────────────────────────────

[[ -n ${OUTPUT_DIR} ]] || OUTPUT_DIR="test_output_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUTPUT_DIR}" || die "cannot create ${OUTPUT_DIR}"
OUTPUT_DIR="$( cd "${OUTPUT_DIR}" && pwd )"
# The image runs as its own non-root user, whose uid is not the caller's, so the
# mount has to be writable by anyone for the suite to write its report into it.
chmod 0777 "${OUTPUT_DIR}" 2>/dev/null \
    || warn "could not chmod 0777 ${OUTPUT_DIR}; the container may not be able to write there"

# Mirror everything from here on into the run directory. Teed rather than copied
# at exit, so a killed run still keeps its transcript.
exec > >(tee -a "${OUTPUT_DIR}/console.log") 2>&1

# ─────────────────────────────────────────────────────────────────────────────
# Devices and volumes
# ─────────────────────────────────────────────────────────────────────────────

DOCKER_ARGS=(
    --rm
    --entrypoint ""
    --network host
    --cap-add SYSLOG
)

add_device() {
    local path=$1 perms=$2 required=$3 why=$4
    if [[ -e ${path} ]]; then
        DOCKER_ARGS+=(--device "${path}:${path}:${perms}")
    elif [[ ${required} == required ]]; then
        die "${path} is not present on this host. ${why}"
    else
        warn "${path} is not present on this host; ${why}"
    fi
}

add_volume() {
    local path=$1 opts=$2 why=$3
    if [[ -e ${path} ]]; then
        DOCKER_ARGS+=(-v "${path}:${path}${opts:+:${opts}}")
    else
        warn "${path} is not present on this host; ${why}"
    fi
}

add_device /dev/tenstorrent rwm required \
    "Without the chips there is nothing to check — is tt-kmd loaded?"
# tt-smi -glx_reset drives the reset over IPMI (`ipmitool raw 0x30 0x8b ...`),
# and the snapshot phase reads the chassis FRU the same way.
add_device /dev/ipmi0 rw optional \
    "tt-smi -glx_reset and the host_fru_info check both go through ipmitool and will fail"
add_volume /dev/hugepages "" \
    "tt-metal allocates from hugepages; the gtest phase will fail to bring devices up"
add_volume /dev/hugepages-1G "" \
    "1G hugepages are unavailable; only a concern if this host is configured for them"
add_volume /etc/udev/rules.d ro \
    "the tt-kmd udev rules are not visible to the container"
add_volume /lib/modules ro \
    "tt-kmd module metadata is not visible; triage cannot report the driver version"
add_volume /sys/kernel/debug ro \
    "debugfs is unavailable; the triage device mappings section will be empty"

DOCKER_ARGS+=(-v "${OUTPUT_DIR}:${OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Staged temporary files: the virt stub and the token env-file. Both cleaned up
# on exit; neither may outlive the run.
# ─────────────────────────────────────────────────────────────────────────────

VIRT_STUB=""
TOKEN_ENV_FILE=""

cleanup() {
    [[ -z ${VIRT_STUB} ]]      || rm -f "${VIRT_STUB}"
    [[ -z ${TOKEN_ENV_FILE} ]] || rm -f "${TOKEN_ENV_FILE}"
}
trap cleanup EXIT

# systemd-detect-virt stub. Best-effort, like the rest of the optional setup: a
# host that cannot stage it still runs every check that needs no ASIC location.
# Both lookup paths are covered — PATH and absolute.
if ! ${SKIP_QSFP}; then
    if VIRT_STUB="$(mktemp 2>/dev/null)" \
       && printf '#!/bin/sh\nexit 1\n' > "${VIRT_STUB}" \
       && chmod 0755 "${VIRT_STUB}"; then
        DOCKER_ARGS+=(
            -v "${VIRT_STUB}:/usr/local/bin/systemd-detect-virt:ro"
            -v "${VIRT_STUB}:/usr/bin/systemd-detect-virt:ro"
        )
    else
        VIRT_STUB=""
        warn "could not stage the systemd-detect-virt stub; QSFP checks that read an" \
             "ASIC location will see a virtualised environment and decline to"
    fi
fi

# The token goes in via --env-file rather than -e so it stays out of the process
# list, matching the fleet launcher.
if [[ -n ${GH_TOKEN_VALUE} ]]; then
    TOKEN_ENV_FILE="$(mktemp)"
    chmod 600 "${TOKEN_ENV_FILE}"
    printf 'GH_TOKEN=%s\n' "${GH_TOKEN_VALUE}" > "${TOKEN_ENV_FILE}"
    DOCKER_ARGS+=(--env-file "${TOKEN_ENV_FILE}")
fi

# ─────────────────────────────────────────────────────────────────────────────
# Host inputs
# ─────────────────────────────────────────────────────────────────────────────

# Mount a host path read-only under IN_DIR and report its in-container path in
# MOUNTED_PATH, so a caller can hand the suite a host path and have it resolve.
# The answer comes back in a variable rather than on stdout because a command
# substitution would run this in a subshell, where the mount it appends to
# DOCKER_ARGS — and any die() — would be discarded with that subshell.
MOUNTED_PATH=""
mount_input() {
    local host=$1 name=$2 kind=$3
    [[ -e ${host} ]] || die "${host} does not exist"
    if [[ ${kind} == dir ]]; then
        [[ -d ${host} ]] || die "${host} is not a directory"
    else
        [[ -f ${host} ]] || die "${host} is not a file"
    fi
    local abs
    abs="$( cd "$( dirname "${host}" )" && printf '%s/%s' "$(pwd)" "$(basename "${host}")" )"
    DOCKER_ARGS+=(-v "${abs}:${IN_DIR}/${name}:ro")
    MOUNTED_PATH="${IN_DIR}/${name}"
}

if [[ -n ${HOST_INPUT_SNAPSHOT} ]]; then
    mount_input "${HOST_INPUT_SNAPSHOT}" snapshot.json file
    FORWARD+=(--input-snapshot "${MOUNTED_PATH}")
fi
if [[ -n ${HOST_QSFP_DESCRIPTOR} ]]; then
    mount_input "${HOST_QSFP_DESCRIPTOR}" factory_system_descriptor.textproto file
    FORWARD+=(--qsfp-descriptor "${MOUNTED_PATH}")
fi
if [[ -n ${HOST_TRIAGE_DIR} ]]; then
    mount_input "${HOST_TRIAGE_DIR}" triage dir
    FORWARD+=(--triage-dir "${MOUNTED_PATH}")
fi

# ─────────────────────────────────────────────────────────────────────────────
# Container payload
# ─────────────────────────────────────────────────────────────────────────────

DOCKER_ARGS+=(
    -e "HC_TIER=${TIER}"
    -e "HC_OUTPUT_DIR=${OUTPUT_DIR}"
    -e "HC_RUN_DIAG=${RUN_DIAG}"
    -e "HC_DIAG_PKG_REPO=${DIAG_PKG_REPO}"
    -e "HC_DIAG_PKG_VERSION=${DIAG_PKG_VERSION}"
    -e "HC_NEED_LSPCI=${NEED_LSPCI}"
    -e "HC_SKIP_QSFP=${SKIP_QSFP}"
)

CONTAINER_SCRIPT='
set -o pipefail

_say()  { printf "[wrap] %s\n" "$*"; }
_warn() { printf "[wrap] WARNING: %s\n" "$*" >&2; }

# The upstream image ships neither pciutils nor, on some tags, ipmitool, and the
# diag suite shells out to both.
_need=""
command -v lspci    >/dev/null 2>&1 || _need="${_need} pciutils"
command -v ipmitool >/dev/null 2>&1 || _need="${_need} ipmitool"
if [ -n "${_need}" ]; then
    _say "installing:${_need}"
    if ! { sudo apt-get update -qq \
        && sudo apt-get install -y -qq --no-install-recommends ${_need}; }; then
        _warn "apt-get failed; continuing with whatever is already present"
    fi
fi

# The reset phase counts chips with a shell pipeline:
#     lspci -d 1e52: | wc -l
# A missing lspci does not fail that pipeline. wc reports 0 and the exit status
# is wc-s, so the suite reads a real "0 chips enumerated" where it should read
# 32, marks every reset FAIL with post_pcie=0, and stops the reset loop on a
# perfectly healthy machine. Better to refuse than to hand back that report.
if ! command -v lspci >/dev/null 2>&1; then
    if [ "${HC_NEED_LSPCI}" = "true" ]; then
        _warn "lspci is still missing after the install attempt"
        {
            echo "ERROR: the reset phase counts Tenstorrent chips with"
            echo "         lspci -d 1e52: | wc -l"
            echo "       which prints 0 and exits 0 when lspci is absent. Every reset would"
            echo "       be recorded as post_pcie=0 and FAIL, and the reset loop would stop,"
            echo "       on a healthy system. Refusing to produce that report."
            echo "       Fix by giving the container network access so pciutils installs,"
            echo "       baking pciutils into the image, or rerunning with --skip-reset."
        } >&2
        exit 1
    fi
    _warn "lspci is missing, but this run takes no post-reset chip count"
fi
command -v ipmitool >/dev/null 2>&1 \
    || _warn "ipmitool is missing: tt-smi -glx_reset and the host_fru_info check need it"

# tt-syseng-diag ships tt-bh-glx-cluster-debug, which the QSFP phase needs. It
# is a private release asset, so it cannot be baked into the upstream image and
# is fetched per run with a token.
#
# Nothing here is fatal. A run without a token performs every other check rather
# than refusing to start, and a download, checksum or dpkg failure costs the
# QSFP phase alone, which then reports SKIP. Each outcome says so explicitly:
# this is the only place the package could have gone missing, and the suite
# reports its absence only as a phase that did not run.
if [ "${HC_SKIP_QSFP}" = "true" ]; then
    _say "QSFP package: --skip-qsfp-tests given; not installing tt-syseng-diag"
elif [ -z "${HC_DIAG_PKG_VERSION}" ]; then
    _say "QSFP package: no version configured; skipping the install"
elif [ -z "${GH_TOKEN:-}" ]; then
    _say "QSFP package: no GitHub token supplied, so the private release cannot be"
    _say "QSFP package: fetched. The QSFP phase will report SKIP; pass --gh-token"
    _say "QSFP package: to enable it. Every other check is unaffected."
elif ! command -v gh >/dev/null 2>&1; then
    _warn "QSFP package: gh is not present in this image, so ${HC_DIAG_PKG_REPO}" \
          "${HC_DIAG_PKG_VERSION} cannot be downloaded; skipping the install"
elif ! _pkg_dir="$(mktemp -d 2>/dev/null)"; then
    _warn "QSFP package: could not create a download directory; skipping the install"
else
    if (
        cd "${_pkg_dir}" \
        && gh release download "${HC_DIAG_PKG_VERSION}" -R "${HC_DIAG_PKG_REPO}" \
             -p "*.deb" -p "*.sha256" \
        && sha256sum -c tt-syseng-diag_*.deb.sha256 \
        && sudo dpkg -i tt-syseng-diag_*.deb
    ); then
        _say "QSFP package: tt-syseng-diag ${HC_DIAG_PKG_VERSION} installed from" \
             "${HC_DIAG_PKG_REPO} (checksum verified)"
    else
        _warn "QSFP package: could not install tt-syseng-diag ${HC_DIAG_PKG_VERSION}" \
              "from ${HC_DIAG_PKG_REPO} — see the download, sha256sum and dpkg output" \
              "above. The run continues; the QSFP phase will report SKIP."
    fi
    rm -rf "${_pkg_dir}"
fi

[ -f "${HC_RUN_DIAG}" ] || {
    echo "ERROR: ${HC_RUN_DIAG} is not present in this image." >&2
    exit 1
}

# --output takes a file path, not a directory. --snapshot-out is placed in the
# mount too, so the raw tt-smi snapshot survives the container; both come before
# the forwarded arguments so an explicit --snapshot-out still wins.
_say "starting the health check"
exec bash "${HC_RUN_DIAG}" "${HC_TIER}" \
    --output "${HC_OUTPUT_DIR}/diag_report.json" \
    --snapshot-out "${HC_OUTPUT_DIR}/diag_snapshot.json" \
    "$@"
'

# ─────────────────────────────────────────────────────────────────────────────
# Go
# ─────────────────────────────────────────────────────────────────────────────

command -v docker >/dev/null 2>&1 || die "docker is not on PATH"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    say "image not present locally; pulling"
    docker pull "${IMAGE}" || die "docker pull ${IMAGE} failed"
fi

say "tier:       ${TIER}"
say "image:      ${IMAGE}"
say "output dir: ${OUTPUT_DIR}"
say "forwarded:  ${FORWARD[*]:-(none)}"
say ""

rc=0
docker run \
    "${DOCKER_ARGS[@]}" \
    ${EXTRA_DOCKER[@]+"${EXTRA_DOCKER[@]}"} \
    "${IMAGE}" \
    bash -c "${CONTAINER_SCRIPT}" _ ${FORWARD[@]+"${FORWARD[@]}"} || rc=$?

# The suite exits 1 on FAIL and 0 on PASS/WARN; that status is propagated
# unchanged, so a caller can gate on it.
say ""
say "health check exit code: ${rc}"
say "results on this host:"
if [[ -f ${OUTPUT_DIR}/diag_report.json ]]; then
    say "  report      ${OUTPUT_DIR}/diag_report.json"
else
    say "  report      (none written — the run did not reach the end)"
fi
say "  step logs   ${OUTPUT_DIR}/logs/"
say "  console     ${OUTPUT_DIR}/console.log"

exit "${rc}"
