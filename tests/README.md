# Test Infrastructure Guide

To run the test infrastructure, ensure you have one of the following devices installed on your system:

- **Blackhole** (any derivative)
- **Wormhole** (any derivative)

Device must be flashed with the original firmware.

Testing environment, when properly initialized, can detect the underlying hardware.

## Steps to Run Tests

1. Execute the environment setup script:
    - If you are using `tt-llk` docker image, run:

    ```bash
    ./setup_testing_env.sh
    ```

    - If you are an external developer, or don't use `tt-llk` docker image, run:

    ```bash
    ./setup_external_testing_env.sh
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
