set -eo pipefail

reset_output=$(tt-smi -wr all)

echo "${reset_output}"

if [[ $reset_output == *"No chips detected"* ]]; then
  echo "Error: Unsuccessful board reset"
  exit 1
fi

tt-smi -s -f smi.log

echo "Printing out SMI information..."

cat smi.log

echo "Printing out cpu information..."

lscpu

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
