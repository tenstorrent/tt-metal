#!/bin/bash
set -eo pipefail

# Pick cnx1 when present, else first up non-virtual NIC.
default_mpi_tcp_interface() {
    if [[ -d /sys/class/net/cnx1 ]]; then
        echo "cnx1"
        return 0
    fi
    local n state
    for n in /sys/class/net/*; do
        n="${n##*/}"
        case "${n}" in
            lo | docker* | br-* | veth* | tailscale* | cali* | flannel*) continue ;;
        esac
        state="$(cat "/sys/class/net/${n}/operstate" 2>/dev/null || true)"
        if [[ "${state}" == "up" ]]; then
            echo "${n}"
            return 0
        fi
    done
    echo "cnx1"
}

extract_hosts_from_hostfile() {
    local host_count="$1"
    local hostfile="${2:-/etc/mpirun/hostfile}"

    awk '!/^#/ && NF {print $1}' "$hostfile" | head -n "$host_count" | paste -sd,
}
