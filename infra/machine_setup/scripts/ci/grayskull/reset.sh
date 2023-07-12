set -eo pipefail

tt-smi -tr all

WEKA_CHECK_DIR="/mnt/MLPerf/bit_error_tests"

if [ -d $WEKA_CHECK_DIR ]; then
  echo "Weka check passed"
else
  echo "$WEKA_CHECK_DIR does exist - Weka is likely disconnected from this runner"
  exit 1
fi

if python3 /opt/tt_metal_infra/scripts/setup_hugepages.py check; then
  echo "Hugepages check passed"
else
  echo "Hugepages check did not pass - error"
  exit 1
fi

echo Current date / time is $(date)
