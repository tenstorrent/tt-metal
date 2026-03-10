#!/usr/bin/env bash
# ttop.sh - TTOP (TT Operator) allocation management for multi-node jobs
# Port of .github/actions/ttop-{create,delete,get}-allocation,
#          .github/actions/ttop-{create,delete}-environment,
#          .github/actions/ttop-configure-tt-run.
# Uses kubectl for TTOP CRD operations. For simple multi-node scheduling,
# prefer native Slurm --nodes=N with srun instead.

set -euo pipefail

[[ -n "${_SLURM_CI_TTOP_SH:-}" ]] && return 0
_SLURM_CI_TTOP_SH=1

TTOP_KUBECONFIG="${TTOP_KUBECONFIG:-${HOME}/.kube/config}"
TTOP_NAMESPACE="${TTOP_NAMESPACE:-ttop}"
TTOP_WAIT_TIMEOUT="${TTOP_WAIT_TIMEOUT:-999m}"
TTOP_SSH_SETUP="${TTOP_SSH_SETUP:-${HOME}/ttop/ssh-setup.sh}"
TT_CARDS_PER_HOST="${TT_CARDS_PER_HOST:-8}"

_kubectl() {
    if ! command -v kubectl &>/dev/null; then
        log_error "kubectl not found in PATH (required for TTOP operations)"
        return 1
    fi
    kubectl --kubeconfig="${TTOP_KUBECONFIG}" "$@"
}

_kubectl_ns() {
    _kubectl -n "${TTOP_NAMESPACE}" "$@"
}

# ---------------------------------------------------------------------------
# ttop_create_allocation - Create a TTOP Allocation CR and wait for it
# ---------------------------------------------------------------------------
# Usage: ttop_create_allocation <name> <spec_yaml>
#   spec_yaml is inline YAML for the spec section (indented under spec:).
#   Outputs the allocation name on success.
ttop_create_allocation() {
    local name="$1"
    local spec_yaml="${2:-}"

    log_info "Creating TTOP allocation: ${name}"

    local manifest
    if [[ -n "${spec_yaml}" ]]; then
        local indented_spec
        indented_spec=$(printf '%s\n' "${spec_yaml}" | sed 's/^/    /')
        manifest=$(cat <<EOF
apiVersion: tenstorrent.com/v1alpha1
kind: Allocation
metadata:
  name: ${name}
spec:
  retryPolicy: Always
${indented_spec}
EOF
)
    else
        manifest=$(cat <<EOF
apiVersion: tenstorrent.com/v1alpha1
kind: Allocation
metadata:
  name: ${name}
spec:
  retryPolicy: Always
EOF
)
    fi

    printf '%s' "${manifest}" | _kubectl apply -f -

    log_info "Waiting for allocation ${name} to reach Allocated state (timeout=${TTOP_WAIT_TIMEOUT})..."
    _kubectl wait \
        --for=jsonpath='{.status.phase}'=Allocated \
        "allocation/${name}" \
        --timeout="${TTOP_WAIT_TIMEOUT}"

    log_info "Allocation ${name} is Allocated"
    printf '%s' "${name}"
}

# ---------------------------------------------------------------------------
# ttop_delete_allocation - Delete a TTOP Allocation CR
# ---------------------------------------------------------------------------
# Usage: ttop_delete_allocation <name>
ttop_delete_allocation() {
    local name="$1"
    log_info "Deleting TTOP allocation: ${name}"
    _kubectl delete "allocation/${name}" --ignore-not-found || true
}

# ---------------------------------------------------------------------------
# ttop_get_allocation_data - Get node-to-rank mapping as JSON
# ---------------------------------------------------------------------------
# Usage: ttop_get_allocation_data <name>
#   Outputs compact JSON: {"node1": "0", "node2": "1", ...}
ttop_get_allocation_data() {
    local name="$1"

    log_info "Getting allocation rank data for: ${name}"

    local nodes_json
    nodes_json=$(_kubectl get nodes -o json \
        --selector "allocation.tenstorrent.com/allocated-by=${name}")

    local ranks_json
    ranks_json=$(printf '%s' "${nodes_json}" \
        | jq '.items | map({(.metadata.name): .metadata.labels["allocation.tenstorrent.com/allocation-rank"]}) | add')

    log_info "Node ranks:"
    printf '%s' "${ranks_json}" | jq . >&2

    printf '%s' "${ranks_json}" | jq -c .
}

# ---------------------------------------------------------------------------
# ttop_create_environment - Create SSH environment on allocated nodes
# ---------------------------------------------------------------------------
# Usage: ttop_create_environment <allocation_name> <docker_image> [env_vars]
#   docker_image format: repository:tag
#   env_vars is an optional newline-separated list of KEY=VALUE pairs.
ttop_create_environment() {
    local allocation_name="$1"
    local docker_image="${2:-${DOCKER_IMAGE:-}}"
    local env_vars="${3:-}"

    if [[ -z "${docker_image}" ]]; then
        log_fatal "ttop_create_environment: docker_image is required"
    fi

    local repository tag
    IFS=':' read -r repository tag <<< "${docker_image}"

    log_info "Generating SSH keypair for environment..."
    local ssh_dir="${HOME}/.ssh"
    mkdir -p "${ssh_dir}"
    if [[ ! -f "${ssh_dir}/id_ed25519" ]]; then
        ssh-keygen -t ed25519 -f "${ssh_dir}/id_ed25519" -N ""
    fi
    local ssh_public_key
    ssh_public_key=$(cat "${ssh_dir}/id_ed25519.pub")

    log_info "Creating TTOP environment: ${allocation_name}"

    local manifest
    manifest=$(cat <<EOF
apiVersion: helm.toolkit.fluxcd.io/v2
kind: HelmRelease
metadata:
  name: ${allocation_name}
spec:
  interval: 5m
  chart:
    spec:
      chart: ./charts/ttop-ssh
      reconcileStrategy: Revision
      sourceRef:
        kind: GitRepository
        name: tt-orchestration
  values:
    nodeSelector:
      allocation.tenstorrent.com/allocated-by: ${allocation_name}
    image:
      repository: ${repository}
      tag: ${tag}
    deviceMountTypes:
      hostPath: true
      devicePlugin: false
    cardsPerHost: ${TT_CARDS_PER_HOST}
    clientAuthorizedKeys: ${ssh_public_key}
    volume:
      enabled: false
EOF
)

    printf '%s' "${manifest}" | _kubectl_ns apply -f -

    log_info "Waiting for environment ${allocation_name} to be ready..."
    _kubectl_ns wait \
        --for=condition=Ready \
        "helmrelease/${allocation_name}" \
        --timeout="${TTOP_WAIT_TIMEOUT}"

    if [[ -x "${TTOP_SSH_SETUP}" ]]; then
        log_info "Running SSH setup: ${TTOP_SSH_SETUP} ${allocation_name}"
        "${TTOP_SSH_SETUP}" "${allocation_name}"
    else
        log_warn "SSH setup script not found at ${TTOP_SSH_SETUP}; skipping"
    fi

    log_info "Environment ${allocation_name} is ready"
}

# ---------------------------------------------------------------------------
# ttop_delete_environment - Delete TTOP SSH environment
# ---------------------------------------------------------------------------
# Usage: ttop_delete_environment <allocation_name>
ttop_delete_environment() {
    local name="$1"
    log_info "Deleting TTOP environment: ${name}"
    _kubectl_ns delete "helmrelease/${name}" --ignore-not-found || true
}

# ---------------------------------------------------------------------------
# ttop_configure_tt_run - Generate hostfile, rankfile, rank_bindings.yaml
# ---------------------------------------------------------------------------
# Usage: ttop_configure_tt_run <allocation_name> <output_dir> [mgd_file]
#   Fetches rank data from the allocation, then writes:
#     output_dir/hostfile
#     output_dir/rankfile
#     output_dir/rank_bindings.yaml
ttop_configure_tt_run() {
    local allocation_name="$1"
    local output_dir="${2:-.}"
    local mgd_file="${3:-}"

    mkdir -p "${output_dir}"

    local ranks_json
    ranks_json=$(ttop_get_allocation_data "${allocation_name}")

    log_info "Generating TT-Run configuration in ${output_dir}..."

    # hostfile: one hostname per line, ordered by rank
    printf '%s' "${ranks_json}" \
        | jq -r 'to_entries | sort_by(.value | tonumber) | .[].key' \
        > "${output_dir}/hostfile"

    # rankfile: OpenMPI rank mapping
    printf '%s' "${ranks_json}" \
        | jq -r 'to_entries | sort_by(.value | tonumber) | .[] | "rank \(.value)=\(.key) slot=0"' \
        > "${output_dir}/rankfile"

    # rank_bindings.yaml
    local num_hosts
    num_hosts=$(wc -l < "${output_dir}/hostfile")

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

    log_info "Generated hostfile (${num_hosts} hosts), rankfile, rank_bindings.yaml"

    # Copy files to worker nodes
    log_info "Copying TT-Run config to workers..."
    while IFS= read -r hostname; do
        log_info "  -> ${hostname}"
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
            "${hostname}" "sudo mkdir -p ${output_dir}"
        for f in hostfile rankfile rank_bindings.yaml; do
            [[ -f "${output_dir}/${f}" ]] || continue
            scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
                "${output_dir}/${f}" "${hostname}:/tmp/${f}"
            ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
                "${hostname}" "sudo mv /tmp/${f} ${output_dir}/${f}"
        done
    done < "${output_dir}/hostfile"
    log_info "TT-Run data distributed to all workers"
}

# ---------------------------------------------------------------------------
# ttop_cleanup - Delete environment then allocation (use in trap handlers)
# ---------------------------------------------------------------------------
# Usage: ttop_cleanup <allocation_name>
#   Errors are logged but do not cause the function to fail, so it is safe
#   to call from EXIT traps.
ttop_cleanup() {
    local name="$1"
    log_info "Running TTOP cleanup for: ${name}"
    ttop_delete_environment "${name}" || log_warn "Failed to delete environment: ${name}"
    ttop_delete_allocation "${name}" || log_warn "Failed to delete allocation: ${name}"
    log_info "TTOP cleanup complete for: ${name}"
}
