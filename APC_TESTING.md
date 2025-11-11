# APC Testing Guide

This document describes how to run APC (Automated Post-Commit) tests on this branch.

## Automated Schedule

The APC tests are automatically scheduled to run daily at 2:00 AM UTC via the `apc-schedule-copilot-branch.yaml` workflow.

## Manual Triggering

You can manually trigger APC tests in several ways:

### Option 1: Using the Scheduled Workflow
1. Go to the Actions tab in GitHub
2. Select "Schedule APC Run for Copilot Branch" workflow
3. Click "Run workflow"
4. Select build type (default: Release) and OS version (default: 22.04)
5. Click "Run workflow"

### Option 2: Using the APC Select Tests Workflow
1. Go to the Actions tab in GitHub
2. Select "(APC) Select and Run Post-Commit Tests" workflow
3. Click "Run workflow"
4. Configure options:
   - Build type: Release, Debug, RelWithDebInfo, ASan, or TSan
   - OS version: 22.04 or 24.04
   - Tests JSON: Specify which test suites to run (default runs all)
   - Commit: Specify a commit SHA (defaults to HEAD)
   - Enable watcher: Enable for long-running tests
5. Click "Run workflow"

### Option 3: Using All Post-Commit Tests
1. Go to the Actions tab in GitHub
2. Select "All post-commit tests" workflow
3. Click "Run workflow"
4. Select build type and version
5. Click "Run workflow"

## Available Test Suites

The following test suites are available in APC:
- `sd-unit-tests`: Standard unit tests
- `fast-dispatch-unit-tests`: Fast dispatch unit tests
- `fabric-unit-tests`: Fabric unit tests
- `cpp-unit-tests`: C++ unit tests
- `ttnn-unit-tests`: TTNN unit tests
- `models-unit-tests`: Model unit tests
- `tt-train-cpp-unit-tests`: TT-Train C++ unit tests
- `run-profiler-regression`: Profiler regression tests
- `t3000-apc-fast-tests`: T3000 APC fast tests
- `test-ttnn-tutorials`: TTNN tutorial tests

## Build Requirements

Before running locally, ensure you have the following dependencies installed:
- libnuma-dev: `sudo apt-get install -y libnuma-dev`
- OpenMPI: `sudo apt-get install -y openmpi-bin libopenmpi-dev`

To build locally:
```bash
git submodule update --init --recursive
./build_metal.sh
```
