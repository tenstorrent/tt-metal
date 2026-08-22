#!/bin/sh

set -e

LOGDIR="."
ITERS="${ITERS:-5}"
PYTHON="$(command -v python3 || command -v python)"
DEPLOYMENT=0
DEPLOYMENT_CYCLES=5
CONTINUE_ON_FAILURE=0

usage() {
	echo "Usage: $0 [-l logdir] [--deployment] [--continue-on-failure]"
	echo "\t-l <logdir>\t\t\tThe directory where to save the log file"
	echo "\t--deployment\t\t\tRun $DEPLOYMENT_CYCLES deployment cycles with a board reset between each."
	echo "\t\t\t\t\tEach cycle produces a separate log file in <logdir>."
	echo "\t\t\t\t\tStops before starting a new cycle if the previous one failed."
	echo "\t--continue-on-failure\t\tRun all $DEPLOYMENT_CYCLES cycles even if a cycle fails (implies --deployment)."
	echo "	Environment variable ITERS controls the number of iterations of each test"
	echo "		(default 5)"
}

while [ -n "$1" ]
do
	case "$1" in
	-l)
		if [ -z "$2" ]; then echo "Missing argument to $1"; exit 1; fi
		LOGDIR="$2"
		shift
		;;
	--deployment)
		DEPLOYMENT=1
		;;
	--continue-on-failure)
		DEPLOYMENT=1
		CONTINUE_ON_FAILURE=1
		;;
	-h)
		usage
		exit
		;;
	*)
		echo "Unknown option: $1"
		usage
		exit 1
		;;
	esac
	shift
done

mkdir -p "$LOGDIR"

GREEN="$(printf '\033[32m')"
RED="$(printf '\033[31m')"
RESET="$(printf '\033[m')"

FAIL=failed
PASS=passed

if [ -t 1 ]
then
	FAIL="$RED$FAIL$RESET"
	PASS="$GREEN$PASS$RESET"
fi

# run_test: runs a command $ITERS times, appending output to $LOGFILE.
run_test() {
	for i in $(seq "$ITERS")
	do
		printf "$MESSAGE loop $i "
		if ! "$@" >> "$LOGFILE" 2>&1
		then
			failures=$((failures + 1))
			echo "$FAIL"
		else
			passes=$((passes + 1))
			echo "$PASS"
		fi
	done
}

# run_tests <logfile>: runs all tests and writes output to <logfile>.
# Sets last_eth_ok, last_dram_ok, last_pcie_read_ok, last_pcie_write_ok (1=pass, 0=fail).
# Returns 1 if any test failed, 0 otherwise.
run_tests() {
	LOGFILE="$1"
	failures=0
	passes=0
	prev_failures=0

	truncate -s0 "$LOGFILE"

	MESSAGE='Ethernet tests\t'
	run_test $PYTHON tests/tt_metal/tt_metal/deployment/eth/test_runner.py
	last_eth_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)
	prev_failures=$failures

	MESSAGE='DRAM tests\t'
	run_test $PYTHON tests/tt_metal/tt_metal/deployment/dram/test_runner.py
	last_dram_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)
	prev_failures=$failures

	MESSAGE='PCIe read test\t'
	run_test ./build/tools/mem_bench --benchmark_filter='Device Reading Host/1073741824/32768/1/0/0/iterations:5/manual_time' --device-id=0
	last_pcie_read_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)
	prev_failures=$failures

	MESSAGE='PCIe write test\t'
	run_test ./build/tools/mem_bench --benchmark_filter='Device Writing Host/1073741824/32768/0/1/0/iterations:5/manual_time' --device-id=0
	last_pcie_write_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)

	if [ "$passes" -gt 0 ]
	then
		echo "$passes tests $PASS"
	fi

	if [ "$failures" -gt 0 ]
	then
		echo "$failures tests $FAIL"
		return 1
	fi

	return 0
}

if [ "$DEPLOYMENT" -eq 1 ]
then
	ITERS=1
	deployment_failures=0
	cycles_run=0
	depl_eth_pass=0
	depl_dram_pass=0
	depl_pcie_read_pass=0
	depl_pcie_write_pass=0

	if tt-smi -s 2>/dev/null | grep -q '"board_type": "tt-galaxy-bh"'; then
		RESET_CMD="tt-smi -glx_reset"
	else
		RESET_CMD="tt-smi -r"
	fi
	echo "Running: $RESET_CMD"

	echo "Resetting boards before deployment..."
	$RESET_CMD

	for cycle in $(seq 1 "$DEPLOYMENT_CYCLES")
	do
		cycle_log="$LOGDIR/deployment_$(hostname)_cycle_${cycle}_$(date +%4Y-%m-%d-%H-%M-%S).log"
		echo "=== Deployment cycle $cycle/$DEPLOYMENT_CYCLES (log: $(basename "$cycle_log")) ==="
		if run_tests "$cycle_log"
		then
			echo "Cycle $cycle PASSED"
		else
			deployment_failures=$((deployment_failures + 1))
			echo "Cycle $cycle FAILED"
		fi
		cycles_run=$((cycles_run + 1))
		depl_eth_pass=$((depl_eth_pass + last_eth_ok))
		depl_dram_pass=$((depl_dram_pass + last_dram_ok))
		depl_pcie_read_pass=$((depl_pcie_read_pass + last_pcie_read_ok))
		depl_pcie_write_pass=$((depl_pcie_write_pass + last_pcie_write_ok))
		if [ "$cycle" -lt "$DEPLOYMENT_CYCLES" ]
		then
			echo "Resetting boards..."
			$RESET_CMD
			if [ "$CONTINUE_ON_FAILURE" -eq 0 ] && [ "$deployment_failures" -gt 0 ]
			then
				echo "Stopping: cycle $cycle failed."
				break
			fi
		fi
	done

	echo ""
	echo "=== Deployment Results Summary (${cycles_run}/${DEPLOYMENT_CYCLES} cycles ran) ==="
	printf "%-20s %s\n" "Ethernet tests:"   "$depl_eth_pass/$cycles_run cycles passed"
	printf "%-20s %s\n" "DRAM tests:"       "$depl_dram_pass/$cycles_run cycles passed"
	printf "%-20s %s\n" "PCIe read test:"   "$depl_pcie_read_pass/$cycles_run cycles passed"
	printf "%-20s %s\n" "PCIe write test:"  "$depl_pcie_write_pass/$cycles_run cycles passed"
	echo ""
	if [ "$deployment_failures" -gt 0 ]
	then
		printf "%-20s %s\n" "Overall:" "$((cycles_run - deployment_failures))/$cycles_run cycles passed"
		exit 1
	fi
	printf "%-20s %s\n" "Overall:" "All $cycles_run cycles passed"
	exit 0
fi

run_tests "$LOGDIR/deployment_$(hostname)_$(date +%4Y-%m-%d-%H-%M-%S).log"
