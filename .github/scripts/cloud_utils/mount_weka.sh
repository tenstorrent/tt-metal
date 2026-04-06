#!/bin/bash

set -eo pipefail

sudo systemctl restart mnt-MLPerf.mount
ls -al /mnt/MLPerf/bit_error_tests

check_hugepages_service_status=0 && ( sudo systemctl status tenstorrent-hugepages.service ) || check_hugepages_service_status=$?
# Exit code 4 for systemctl means not found
if [ $check_hugepages_service_status -eq 4 ]; then
    echo "::warning title=weka-mount-hugepages-service-not-found::Hugepages service not found. Using old rc.local method"
    sudo /etc/rc.local
else
    echo "::notice title=weka-mount-hugepages-service-found::Hugepages service found. Command returned with exit code $check_hugepages_service_status. Restarting it so we can ensure hugepages are available"
    sudo systemctl restart tenstorrent-hugepages.service
fi

# Wait until the hugepages are written as the above are not blocking
hugepages_check_start=$(date +%s)
hugepages_check_timeout=60
while [[ "$(cat "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages")" -eq 0 ]]; do
  sleep 1
  if (( $(date +%s) - hugepages_check_start > hugepages_check_timeout )); then
    echo "::error title=weka-mount-hugepages-not-set::nr_hugepages is still 0 after $hugepages_check_timeout seconds. Please let infra team know via issue."
    exit 1
  fi
done
