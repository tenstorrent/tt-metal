#!/bin/bash

# <run-config>
#   <scenario name="first-run" description="First time running or environment changed">
#     <var name="ENV_SETUP" value="1"/>
#     <var name="COMPILED" value="1"/>
#     <var name="RUN_TEST" value="1"/>
#   </scenario>
#   <scenario name="code-changed" description="Code changed, need recompile but env is ready">
#     <var name="ENV_SETUP" value="0"/>
#     <var name="COMPILED" value="1"/>
#     <var name="RUN_TEST" value="1"/>
#   </scenario>
#   <scenario name="rerun-only" description="Nothing changed, just rerun tests">
#     <var name="ENV_SETUP" value="0"/>
#     <var name="COMPILED" value="0"/>
#     <var name="RUN_TEST" value="1"/>
#   </scenario>
#   <option name="QUIET" default="1" description="Set to 1 to disable terminal output (logs still saved to /tmp/llk_test/)"/>
# </run-config>

# Override these based on the scenario above (uses env vars if provided)
ENV_SETUP=${ENV_SETUP:-1}
COMPILED=${COMPILED:-1}
RUN_TEST=${RUN_TEST:-1}
FILE_NAME=${FILE_NAME:-""}
TEST_PATH=${TEST_PATH:-""}
QUIET=${QUIET:-1}
PARALLEL_JOBS=${PARALLEL_JOBS:-10}
FAIL_FAST=${FAIL_FAST:-1}
PYTEST_ARGS=${PYTEST_ARGS:-""}
COVERAGE_FLAG=""

if [ "${COVERAGE:-0}" -eq 1 ]; then
    COVERAGE_FLAG="--coverage"
fi

if [ $ENV_SETUP -eq 1 ]; then
    echo "Setting up environment..."
    ./setup_testing_env.sh
fi

cd ./python_tests

if [ -n "$TEST_PATH" ]; then
    TEST_PATH="${TEST_PATH#./}"
    case "$TEST_PATH" in
        python_tests/*) TEST_PATH="${TEST_PATH#python_tests/}" ;;
    esac
fi

if [ $COMPILED -eq 1 ]; then
    echo "Compiling..."
    mkdir -p /tmp/llk_test
    if [ -n "$TEST_PATH" ]; then
        TARGET="$TEST_PATH"
    else
        TARGET="$FILE_NAME"
    fi
    if [ $FAIL_FAST -eq 1 ]; then
        FAIL_FAST_ARG="-x"
    else
        FAIL_FAST_ARG=""
    fi
    if [ $QUIET -eq 1 ]; then
        pytest $COVERAGE_FLAG --compile-producer -n $PARALLEL_JOBS $FAIL_FAST_ARG $PYTEST_ARGS $TARGET > /tmp/llk_test/compile.log 2>&1
        SUCCESS=$?
    else
        pytest $COVERAGE_FLAG --compile-producer -n $PARALLEL_JOBS $FAIL_FAST_ARG $PYTEST_ARGS $TARGET 2>&1 | tee /tmp/llk_test/compile.log
        SUCCESS=${PIPESTATUS[0]}
    fi

    if [ $SUCCESS -ne 0 ]; then
        echo "Compilation failed"
        exit 1
    fi

fi

if [ $RUN_TEST -eq 1 ]; then
    echo "Running tests..."
    mkdir -p /tmp/llk_test
    if [ -n "$TEST_PATH" ]; then
        TARGET="$TEST_PATH"
    else
        TARGET="$FILE_NAME"
    fi
    if [ $FAIL_FAST -eq 1 ]; then
        FAIL_FAST_ARG="-x"
    else
        FAIL_FAST_ARG=""
    fi
    if [ $QUIET -eq 1 ]; then
        pytest $COVERAGE_FLAG --compile-consumer $FAIL_FAST_ARG $PYTEST_ARGS $TARGET > /tmp/llk_test/run.log 2>&1 && tail -10 /tmp/llk_test/run.log
    else
        script -q -c "pytest $COVERAGE_FLAG --compile-consumer $FAIL_FAST_ARG $PYTEST_ARGS $TARGET" /tmp/llk_test/run.log
    fi
fi
