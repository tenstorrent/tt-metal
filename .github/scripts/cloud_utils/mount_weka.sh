#!/bin/bash

set -eo pipefail

sudo systemctl restart mnt-MLPerf.mount
sudo /etc/rc.local
ls -al /mnt/MLPerf/bit_error_tests
