#!/bin/sh

set -e

LOGDIR="."
ITERS="${ITERS:-5}"
PYTHON="$(command -v python3 || command -v python)"
DEPLOYMENT=0
DEPLOYMENT_CYCLES=5
CONTINUE_ON_FAILURE=0

usage() {
	cat << EOF
Usage: $0 [-l <logdir>] [--deployment] [--continue-on-failure] [--no-eth-links]

Run the deployment test suite (Ethernet, DRAM, PCIe read/write).
Must be run from the repository root with the tests already built.
Everything printed to the console is also written to the log files.

Optional:
    -l <logdir>                             Directory where log files are written
                                            (default: current directory)
    --deployment                            Run $DEPLOYMENT_CYCLES deployment cycles with a board reset
                                            between each. Each cycle also produces a separate log
                                            file in <logdir>. Stops before starting a new cycle
                                            if the previous one failed. Forces ITERS=1.
    --continue-on-failure                   Run all $DEPLOYMENT_CYCLES cycles even if a cycle fails
                                            (implies --deployment)
    --no-eth-links                          Do not require a specific number of Ethernet links per
                                            chip (sets ETH_TEST_EXPECTED_LINKS=0). Use on partially
                                            cabled systems, otherwise 10 links per chip are expected.
    -h                                      Display this help message and exit

Environment variables:
    ITERS                                   Number of iterations of each test (default: 5,
                                            ignored in deployment mode)

Examples:
    Regular run (all tests, $ITERS iterations each, logs in ./logs):
        $0 -l ./logs

    Deployment run ($DEPLOYMENT_CYCLES reset cycles, one iteration per test per cycle):
        $0 -l ./logs --deployment

    Deployment run on a partially cabled system, without stopping on failure:
        $0 -l ./logs --continue-on-failure --no-eth-links
EOF
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
	--no-eth-links)
		ETH_TEST_EXPECTED_LINKS=0
		export ETH_TEST_EXPECTED_LINKS
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

RUN_LOG="$LOGDIR/deployment_$(hostname)_$(date +%4Y-%m-%d-%H-%M-%S).log"
: > "$RUN_LOG"
LOGFILES="$RUN_LOG"

# Carries a command's exit status out of the tee pipeline
RCFILE="$(mktemp)"
trap 'rm -f "$RCFILE"' EXIT HUP INT TERM

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

# emit: print a line.
emit() {
	printf '%s\n' "$*"
	for f in $LOGFILES
	do
		printf '%s\n' "$*" >> "$f"
	done
}

# emit_status <text> <passed|failed>: print a line ending in a pass/fail verdict.
emit_status() {
	case "$2" in
	passed) colored="$PASS" ;;
	*) colored="$FAIL" ;;
	esac
	printf '%s %s\n' "$1" "$colored"
	for f in $LOGFILES
	do
		printf '%s %s\n' "$1" "$2" >> "$f"
	done
}

# run_logged: run a command and set $rc to its exit status.
run_logged() {
	echo 0 > "$RCFILE"
	{ "$@" 2>&1 || echo "$?" > "$RCFILE"; } | tee -a $LOGFILES
	rc="$(cat "$RCFILE")"
}

# run_reset <message>: reset the boards, aborting the run if the reset fails.
run_reset() {
	emit ""
	emit "$1"
	run_logged $RESET_CMD
	if [ "$rc" -ne 0 ]
	then
		emit "Reset failed (exit code $rc)"
		exit "$rc"
	fi
}

# run_test: runs a command $ITERS times.
run_test() {
	for i in $(seq "$ITERS")
	do
		emit ""
		emit "--- $MESSAGE: iteration $i/$ITERS ---"
		run_logged "$@"
		if [ "$rc" -ne 0 ]
		then
			failures=$((failures + 1))
			emit_status "$MESSAGE iteration $i:" failed
		else
			passes=$((passes + 1))
			emit_status "$MESSAGE iteration $i:" passed
		fi
	done
}

# run_tests [cycle_logfile]: runs all tests.
# Sets last_eth_ok, last_dram_ok, last_pcie_read_ok, last_pcie_write_ok (1=pass, 0=fail).
# Returns 1 if any test failed, 0 otherwise.
run_tests() {
	LOGFILES="$RUN_LOG${1:+ $1}"
	failures=0
	passes=0
	prev_failures=0

	MESSAGE='Ethernet tests'
	run_test $PYTHON tests/tt_metal/tt_metal/deployment/eth/test_runner.py
	last_eth_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)
	prev_failures=$failures

	MESSAGE='DRAM tests'
	run_test $PYTHON tests/tt_metal/tt_metal/deployment/dram/test_runner.py
	last_dram_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)
	prev_failures=$failures

	MESSAGE='PCIe read test'
	run_test ./build/tools/mem_bench --benchmark_filter='Device Reading Host/1073741824/32768/1/0/0/iterations:5/manual_time' --device-id=0
	last_pcie_read_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)
	prev_failures=$failures

	MESSAGE='PCIe write test'
	run_test ./build/tools/mem_bench --benchmark_filter='Device Writing Host/1073741824/32768/0/1/0/iterations:5/manual_time' --device-id=0
	last_pcie_write_ok=$([ "$failures" -eq "$prev_failures" ] && echo 1 || echo 0)

	emit ""

	if [ "$passes" -gt 0 ]
	then
		emit_status "$passes tests" passed
	fi

	if [ "$failures" -gt 0 ]
	then
		emit_status "$failures tests" failed
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

	RESET_CMD="tt-smi -glx_reset"
	emit "Run log: $RUN_LOG"
	run_reset "Resetting boards before deployment ($RESET_CMD)..."

	for cycle in $(seq 1 "$DEPLOYMENT_CYCLES")
	do
		cycle_log="$LOGDIR/deployment_$(hostname)_cycle_${cycle}_$(date +%4Y-%m-%d-%H-%M-%S).log"
		: > "$cycle_log"
		LOGFILES="$RUN_LOG $cycle_log"
		emit ""
		emit "=== Deployment cycle $cycle/$DEPLOYMENT_CYCLES (log: $(basename "$cycle_log")) ==="
		if run_tests "$cycle_log"
		then
			emit "Cycle $cycle PASSED"
		else
			deployment_failures=$((deployment_failures + 1))
			emit "Cycle $cycle FAILED"
		fi
		LOGFILES="$RUN_LOG"
		cycles_run=$((cycles_run + 1))
		depl_eth_pass=$((depl_eth_pass + last_eth_ok))
		depl_dram_pass=$((depl_dram_pass + last_dram_ok))
		depl_pcie_read_pass=$((depl_pcie_read_pass + last_pcie_read_ok))
		depl_pcie_write_pass=$((depl_pcie_write_pass + last_pcie_write_ok))
		if [ "$cycle" -lt "$DEPLOYMENT_CYCLES" ]
		then
			run_reset "Resetting boards ($RESET_CMD)..."
			if [ "$CONTINUE_ON_FAILURE" -eq 0 ] && [ "$deployment_failures" -gt 0 ]
			then
				emit "Stopping: cycle $cycle failed."
				break
			fi
		fi
	done

	emit ""
	emit "=== Deployment Results Summary (${cycles_run}/${DEPLOYMENT_CYCLES} cycles ran) ==="
	emit "$(printf '%-20s %s' 'Ethernet tests:'  "$depl_eth_pass/$cycles_run cycles passed")"
	emit "$(printf '%-20s %s' 'DRAM tests:'      "$depl_dram_pass/$cycles_run cycles passed")"
	emit "$(printf '%-20s %s' 'PCIe read test:'  "$depl_pcie_read_pass/$cycles_run cycles passed")"
	emit "$(printf '%-20s %s' 'PCIe write test:' "$depl_pcie_write_pass/$cycles_run cycles passed")"
	emit ""
	if [ "$deployment_failures" -gt 0 ]
	then
		emit "$(printf '%-20s %s' 'Overall:' "$((cycles_run - deployment_failures))/$cycles_run cycles passed")"
		emit "Run log: $RUN_LOG"
		exit 1
	fi
	emit "$(printf '%-20s %s' 'Overall:' "All $cycles_run cycles passed")"
	emit "Run log: $RUN_LOG"
	exit 0
fi

emit "Run log: $RUN_LOG"
run_tests
