# CMake Configuration Tests

This directory contains tests for CMake configuration functionality, including:
- Distro detection (`detect_distro()`)
- MPI configuration (`tt_configure_mpi()`)
- Packaging type detection

## Running the Tests

### Prerequisites

- Python 3.8+
- pytest (install with `pip install pytest`)

### Run All Tests

```bash
# From project root
pytest tests/cmake/ -v
```

### Run Specific Test Files

```bash
# Test distro detection
pytest tests/cmake/test_distro_detection.py -v

# Test MPI configuration
pytest tests/cmake/test_mpi_configuration.py -v
```

### Run Specific Test Cases

```bash
# Run a specific test
pytest tests/cmake/test_distro_detection.py::TestDistroDetection::test_detect_ubuntu_22_04 -v
```

## Test Structure

### `test_distro_detection.py`

Unit tests for the `detect_distro()` CMake function. Tests various `/etc/os-release` file formats and verifies correct detection of:
- Ubuntu, Debian, Fedora, RHEL, CentOS, openSUSE
- Quoted and unquoted values
- Case insensitivity
- Missing or partial files

### `test_mpi_configuration.py`

Integration tests for MPI configuration. Tests:
- ULFM MPI detection
- System MPI detection
- MPI library directory extraction
- RPATH handling

**Note:** Some MPI tests may require CMake and MPI to be installed. Tests will skip gracefully if dependencies are missing.

## CI Integration

These tests are automatically run in CI via `.github/workflows/cmake-tests.yaml`. The workflow:
1. Runs distro detection unit tests
2. Runs MPI configuration integration tests
3. Validates RPATH/RUNPATH in built libraries (if build artifact is available)

## Adding New Tests

When adding new CMake functionality, add corresponding tests here:

1. Create a new test file: `test_<feature>.py`
2. Follow the existing test structure
3. Add the test file to this directory
4. Tests will be automatically discovered by pytest

## RPATH Validation

The `validate_rpath.sh` script (in `tests/scripts/`) validates RPATH/RUNPATH in built binaries:
- Checks that `$ORIGIN` appears first (Fedora compliance)
- Verifies no absolute build paths leak into RPATH
- Validates MPI library paths are included when appropriate

Run manually:
```bash
./tests/scripts/validate_rpath.sh [build_dir] [library_name]
```

Example:
```bash
./tests/scripts/validate_rpath.sh build libtt_metal.so
```
