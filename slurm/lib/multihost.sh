#!/usr/bin/env bash
# multihost.sh - Native Slurm multi-node setup for TT-Run tests
# Replaces TTOP (Kubernetes-based) allocation for environments where
# Slurm handles node scheduling directly via --nodes=N.
#
# Generates hostfile.txt, rankfile.txt, and rank_bindings.yaml from
# SLURM_JOB_NODELIST, then distributes them to worker nodes via SCP.

set -euo pipefail

[[ -n "${_SLURM_CI_MULTIHOST_SH:-}" ]] && return 0
_SLURM_CI_MULTIHOST_SH=1

SLURM_CI_LIB_DIR="${SLURM_CI_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SLURM_CI_LIB_DIR}/common.sh"

_SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

# ---------------------------------------------------------------------------
# reset_devices - Run tt-smi -r on every node in the Slurm allocation
# ---------------------------------------------------------------------------
# Resets Tenstorrent accelerators before tests, matching the GitHub Actions
# runner setup behaviour. Uses srun when available (preferred), falls back
# to parallel SSH.
reset_devices() {
    if ! command -v tt-smi &>/dev/null; then
        log_warn "tt-smi not found in PATH, skipping device reset"
        return 0
    fi

    log_info "Resetting Tenstorrent devices on all allocated nodes..."

    if is_slurm_job && command -v srun &>/dev/null; then
        srun --ntasks-per-node=1 bash -c \
            'echo "[$(hostname)] Resetting devices..." && tt-smi -r' || {
            log_warn "srun tt-smi -r returned non-zero (rc=$?), continuing anyway"
        }
    elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        local -a hosts
        if command -v scontrol &>/dev/null; then
            mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
        else
            IFS=',' read -ra hosts <<< "$SLURM_JOB_NODELIST"
        fi
        local local_host
        local_host="$(hostname -s)"
        local host
        for host in "${hosts[@]}"; do
            log_info "  Resetting devices on ${host}..."
            if [[ "${host}" == "${local_host}" || "${host}" == "$(hostname)" ]]; then
                tt-smi -r || log_warn "tt-smi -r failed on ${host} (rc=$?)"
            else
                ssh "${_SSH_OPTS[@]}" "${host}" "tt-smi -r" || \
                    log_warn "tt-smi -r failed on ${host} (rc=$?)"
            fi
        done
    else
        log_info "  Resetting devices on $(hostname)..."
        tt-smi -r || log_warn "tt-smi -r failed (rc=$?)"
    fi

    log_info "Device reset complete"
}

# ---------------------------------------------------------------------------
# multihost_setup - Generate TT-Run config from the Slurm node allocation
# ---------------------------------------------------------------------------
# Usage: multihost_setup <output_dir> [mgd_file]
#   Writes:
#     output_dir/hostfile.txt
#     output_dir/rankfile.txt
#     output_dir/rank_bindings.yaml
#   Requires SLURM_JOB_NODELIST to be set (automatic inside sbatch jobs).
multihost_setup() {
    local output_dir="${1:?multihost_setup: output_dir is required}"
    local mgd_file="${2:-}"

    require_env SLURM_JOB_NODELIST

    mkdir -p "${output_dir}"

    # Expand compact nodelist (e.g. "node[01-04]") into one-per-line hostnames
    local -a hosts
    if command -v scontrol &>/dev/null; then
        mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    else
        # Fallback: comma-separated list (works for simple "host1,host2" lists)
        IFS=',' read -ra hosts <<< "$SLURM_JOB_NODELIST"
    fi

    local num_hosts=${#hosts[@]}
    if (( num_hosts == 0 )); then
        log_error "multihost_setup: SLURM_JOB_NODELIST expanded to zero hosts"
        return 1
    fi

    log_info "Generating TT-Run configuration in ${output_dir} (${num_hosts} hosts)..."

    # hostfile.txt: one hostname per line, ordered by rank
    printf '%s\n' "${hosts[@]}" > "${output_dir}/hostfile.txt"

    # rankfile.txt: OpenMPI rank mapping
    {
        local i
        for (( i = 0; i < num_hosts; i++ )); do
            echo "rank ${i}=${hosts[$i]} slot=0"
        done
    } > "${output_dir}/rankfile.txt"

    # rank_bindings.yaml
    {
        echo "rank_bindings:"
        local i
        for (( i = 0; i < num_hosts; i++ )); do
            echo "  - rank: ${i}"
            echo "    mesh_id: 0"
            echo "    mesh_host_rank: ${i}"
        done
        if [[ -n "${mgd_file}" ]]; then
            echo "mesh_graph_desc_path: \"${mgd_file}\""
        fi
    } > "${output_dir}/rank_bindings.yaml"

    log_info "Generated hostfile.txt (${num_hosts} hosts), rankfile.txt, rank_bindings.yaml"

    # Distribute config files to worker nodes (skip head node = hosts[0])
    if [[ "${MULTIHOST_SHARED_FS:-1}" != "1" ]] && (( num_hosts > 1 )); then
        _multihost_distribute "${output_dir}" "${hosts[@]:1}"
    else
        log_info "Shared filesystem detected, skipping SCP distribution"
    fi
}

# ---------------------------------------------------------------------------
# _multihost_distribute - SCP config files to remote worker nodes
# ---------------------------------------------------------------------------
_multihost_distribute() {
    local output_dir="$1"; shift
    local -a workers=("$@")

    log_info "Copying TT-Run config to ${#workers[@]} worker node(s)..."
    local hostname
    for hostname in "${workers[@]}"; do
        log_info "  -> ${hostname}"
        ssh "${_SSH_OPTS[@]}" "${hostname}" "mkdir -p ${output_dir}" 2>/dev/null || {
            log_warn "Failed to create ${output_dir} on ${hostname}, trying with sudo"
            ssh "${_SSH_OPTS[@]}" "${hostname}" "sudo mkdir -p ${output_dir}"
        }
        for f in hostfile.txt rankfile.txt rank_bindings.yaml; do
            [[ -f "${output_dir}/${f}" ]] || continue
            scp "${_SSH_OPTS[@]}" \
                "${output_dir}/${f}" "${hostname}:${output_dir}/${f}" 2>/dev/null || {
                log_warn "Direct SCP failed for ${f} on ${hostname}, trying via /tmp"
                scp "${_SSH_OPTS[@]}" "${output_dir}/${f}" "${hostname}:/tmp/${f}"
                ssh "${_SSH_OPTS[@]}" "${hostname}" "sudo mv /tmp/${f} ${output_dir}/${f}"
            }
        done
    done
    log_info "TT-Run config distributed to all workers"
}
