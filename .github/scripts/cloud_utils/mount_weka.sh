#!/bin/bash

set -eo pipefail

sudo systemctl restart mnt-MLPerf.mount
ls -al /mnt/MLPerf/bit_error_tests

check_hugepages_service_status=0 && ( sudo systemctl status tenstorrent-hugepages.service ) || check_hugepages_service_status=$?
if [ $check_hugepages_service_status -eq 0 ]; then
    echo "::notice title=weka-mount-hugepages-service-found::Hugepages service found. Restarting it so we can ensure hugepages are available"
    sudo systemctl restart tenstorrent-hugepages.service
else
    echo "::warning title=weka-mount-hugepages-service-not-found::Hugepages service found. Restarting it so we can ensure hugepages are available"
    echo "Hugepages service not found. Using old rc.local method"
    sudo /etc/rc.local
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
