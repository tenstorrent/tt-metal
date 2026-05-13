#!/usr/bin/env bash
# Install /etc/sudoers.d/alex-mlx-smoke granting passwordless sudo for
# the narrow set of operations needed by the WH↔Mellanox smoke-test
# debug loop (kill receivers, inspect DPDK runtime dir).
#
# Usage:
#   bash install_sudoers_mlx_smoke.sh              # install locally
#   bash install_sudoers_mlx_smoke.sh --remote HOST  # also install on remote (via ssh -t)
#
# Run as your normal user; will sudo internally. Each host prompts for
# its own sudo password the first time.
set -euo pipefail

REMOTE_HOST=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote) REMOTE_HOST="${2:-}"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SUDOERS_DST=/etc/sudoers.d/alex-mlx-smoke
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

cat >"$TMP" <<'EOF'
# Smoke-test receiver cleanup + DPDK runtime-dir inspection.
# Scope: only the test_external_cmac_smoke_mlx binary at its known path.
alex ALL=(root) NOPASSWD: /usr/bin/pkill -9 -f /tmp/mlx_smoke_bin/test_external_cmac_smoke_mlx
alex ALL=(root) NOPASSWD: /usr/bin/pkill -f /tmp/mlx_smoke_bin/test_external_cmac_smoke_mlx
alex ALL=(root) NOPASSWD: /usr/bin/ls /var/run/dpdk/rte/
alex ALL=(root) NOPASSWD: /usr/bin/ls -la /var/run/dpdk/rte/

# Smoke-test binary itself: receiver mode AND --tx-probe mode (reverse-direction
# sanity check). Wildcard tail covers EAL args (-l, -n, -a) and app args (--).
alex ALL=(root) NOPASSWD: /tmp/mlx_smoke_bin/test_external_cmac_smoke_mlx *

# Hugepages setup helper — reserves 2MB pages and mounts hugetlbfs for DPDK.
alex ALL=(root) NOPASSWD: /home/alex/mpi-shfs/tenstorrent/tt-metal-external-eth/tests/tt_metal/llrt/setup_hugepages_mlx_smoke.sh
alex ALL=(root) NOPASSWD: /home/alex/mpi-shfs/tenstorrent/tt-metal-external-eth/tests/tt_metal/llrt/setup_hugepages_mlx_smoke.sh *

# Mellanox NIC bring-up: link state, address flush, speed/duplex, FEC.
alex ALL=(root) NOPASSWD: /usr/sbin/ip link set enp2s0np0 *
alex ALL=(root) NOPASSWD: /usr/sbin/ip addr flush dev enp2s0np0
alex ALL=(root) NOPASSWD: /usr/sbin/ethtool *

# TT-RDMA v1 §8 — PFC / lossless wire config (Mellanox side).
# mlnx_qos: priority -> PFC + prio_tc + trust mode + buffer.
# dcb: newer iproute2 equivalent (kernel >= 5.15).
alex ALL=(root) NOPASSWD: /usr/bin/mlnx_qos *
alex ALL=(root) NOPASSWD: /usr/sbin/dcb *

# TT-RDMA v1 §9 — eSwitch bypass via switchdev mode.
# devlink: PF eswitch mode toggle (legacy <-> switchdev).
# tee onto specific sysfs paths: SR-IOV VF allocation + driver bind/unbind.
alex ALL=(root) NOPASSWD: /usr/sbin/devlink *
alex ALL=(root) NOPASSWD: /usr/bin/tee /sys/class/net/enp2s0np0/device/sriov_numvfs
alex ALL=(root) NOPASSWD: /usr/bin/tee /sys/bus/pci/drivers/mlx5_core/bind
alex ALL=(root) NOPASSWD: /usr/bin/tee /sys/bus/pci/drivers/mlx5_core/unbind
alex ALL=(root) NOPASSWD: /usr/bin/tee /sys/bus/pci/drivers/vfio-pci/bind
alex ALL=(root) NOPASSWD: /usr/bin/tee /sys/bus/pci/drivers/vfio-pci/unbind
EOF

# Validate syntax against the proposed file before installing.
sudo visudo -c -f "$TMP"

# visudo -c passed; install with correct mode/owner.
sudo install -m 0440 -o root -g root "$TMP" "$SUDOERS_DST"

echo "Installed $SUDOERS_DST:"
sudo cat "$SUDOERS_DST"

echo
echo "Smoke-testing one of the new permissions (should run with no prompt):"
sudo -n /usr/bin/ls /var/run/dpdk/rte/ || echo "(directory may not exist yet — that's fine)"

if [[ -n "$REMOTE_HOST" ]]; then
    echo
    echo "=== Installing on remote: $REMOTE_HOST ==="
    # Source file is on mpi-shfs (NFS-shared between hosts), so the same path
    # exists on both. Invoke the same script remotely without --remote to avoid
    # recursion.
    ssh -t "$REMOTE_HOST" "bash '$0'"
fi
