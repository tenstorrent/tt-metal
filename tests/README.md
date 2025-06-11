# Test Infrastructure Guide

To run the test infrastructure, ensure your system has one of the following devices installed:

- **Blackhole** (or any derivative)
- **Wormhole** (or any derivative)

> ⚠️ The device must be flashed with the original firmware.

When the testing environment is correctly initialized, it will auto-detect the underlying hardware.

---

## Steps to Run Tests

### 1. Set Up the Environment

- **If using the `tt-llk` Docker image**, run:

    ```bash
    ./setup_testing_env.sh
    ```

- **If you are an external developer or not using the Docker image**, run:

    ```bash
    source ./setup_external_testing_env.sh
    ```

> 🔄 **Note**: Always use `source` to ensure the environment is activated in your current shell session.

---

### 2. Navigate to the Python Test Directory

```bash
cd python_tests
```

### 3. Run any test using `pytest`

```bash
pytest <test_file_name>
```

Replace <test_file_name> with the specific test script you want to execute, e.g., test_sfpu_binary.py.

### 4. Run all tests using `pytest`

```bash
pytest
```
