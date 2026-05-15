#!/usr/bin/env bash
# LLK smoke: pytest CLI fragments from LLK_PYTEST_LOG_MODE (set by llk-smoke-impl.yaml).
# Sourced from tests/pipeline_reorg/llk_smoke_tests.yaml matrix commands after cd tests/python_tests.
case "${LLK_PYTEST_LOG_MODE:-verbose}" in
  ci-quiet)
    PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
    PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"
    ;;
  verbose|*)
    PYTEST_COMPILE_EXTRA=""
    PYTEST_RUN_EXTRA='--override-ini=addopts=-v'
    ;;
esac
