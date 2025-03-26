# Test Infrastructure Guide

To run the test infrastructure, ensure you have one of the following devices installed on your system:
- **Blackhole**
- **Wormhole**

Device must be flashed with the original firmware.

## Steps to Run Tests

1. Source the environment setup script:
    ```bash
    source setup_env.sh
    ```

2. Navigate to the `python_tests` directory:
    ```bash
    cd python_tests
    ```

3. Run any test using `pytest`:
    ```bash
    pytest <test_file_name>
    ```

Replace `<test_file_name>` with the name of the test file you wish to execute.
