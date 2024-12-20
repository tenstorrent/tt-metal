#!/bin/bash

set -eo pipefail

sudo systemctl restart mnt-MLPerf.mount
ls -al /mnt/MLPerf/bit_error_tests
