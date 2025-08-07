import argparse
import sys
import os
import pathlib
import enlighten
import importlib
import datetime as dt
from multiprocessing import Process, Queue
from framework.statuses import VectorValidity, TestStatus
import framework.tt_smi_util as tt_smi_util
from framework.sweeps_logger import sweeps_logger as logger
from framework.vector_source import VectorSource, VectorSourceFactory
from framework.database import deserialize_for_postgres, deserialize


def validate_arguments(args, parser):
    # Define validation rules as tuples of (condition, error_message)
    validation_rules = [
        # Module name dependencies
        (args.vector_id and not args.module_name, "Module name is required if vector id is specified."),
        (args.file_path and not args.module_name, "Module name is required if file path is specified."),
        (
            args.vector_source == "file" and not args.module_name,
            "Module name is required when test vector source is 'file'.",
        ),
        # File path constraints
        (
            args.file_path and args.vector_source in ["elastic", "vectors_export"],
            "File path should not be specified when test vector source is 'elastic' or 'vectors_export'.",
        ),
    ]

    # Check each validation rule
    for condition, error_message in validation_rules:
        if condition:
            parser.print_help()
            logger.error(error_message)
            exit(1)

    # Environment variable validation for elastic database
    if args.vector_source == "elastic" or args.result_dest == "elastic":
        elastic_username = os.getenv("ELASTIC_USERNAME")
        elastic_password = os.getenv("ELASTIC_PASSWORD")

        if not elastic_username or not elastic_password:
            logger.error("ELASTIC_USERNAME and ELASTIC_PASSWORD must be set in the environment variables.")
            exit(1)

    # Validate that skip modules is only used when running all modules
    if args.skip_modules and args.module_name:
        logger.error("Skip modules is only supported when running all modules.")
        exit(1)

    logger.info("All argument validations passed successfully.")


def get_all_modules():
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"
    for file in sorted(sweeps_path.glob("**/*.py")):
        sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
        yield sweep_name


def get_timeout(test_module):
    try:
        timeout = test_module.TIMEOUT
    except:
        timeout = 30
    return timeout


def test_vector_source(vector_source, module_names, vector_source_type):
    """Test the vector source functionality with comprehensive validation"""
    logger.info(f"Testing vector source: {vector_source_type}")

    # Test 1: Validate connection
    logger.info("Testing connection validation...")
    is_connected = vector_source.validate_connection()
    if is_connected:
        logger.info("✓ Connection validation successful")
    else:
        logger.error("✗ Connection validation failed")
        return

    # Test 2: Test with each module
    if isinstance(module_names, str):
        test_modules = [module_names]
    else:
        test_modules = module_names if module_names else []

    if not test_modules:
        logger.info("No modules specified for testing")
        return

    for module_name in test_modules:
        logger.info(f"\n--- Testing module: {module_name} ---")

        # Test 3: Get available suites
        logger.info("Testing get_available_suites...")
        try:
            available_suites = vector_source.get_available_suites(module_name)
            if available_suites:
                logger.info(f"✓ Found {len(available_suites)} suites: {available_suites}")
            else:
                logger.warning(f"✓ No suites found for module {module_name} (this may be expected)")

            # Test 4: Load vectors for each suite
            if available_suites:
                for suite_name in available_suites[:2]:  # Test first 2 suites to avoid too much output
                    logger.info(f"Testing load_vectors for suite: {suite_name}")
                    try:
                        vectors = vector_source.load_vectors(module_name, suite_name)
                        if vectors:
                            logger.info(f"✓ Loaded {len(vectors)} vectors from suite '{suite_name}'")
                            # Show sample vector info
                            sample_vector = vectors[0]
                            vector_id = sample_vector.get("vector_id", "N/A")
                            logger.info(f"  Sample vector ID: {vector_id}")
                            logger.info(f"  Sample vector keys: {list(sample_vector.keys())}")
                        else:
                            logger.warning(f"✓ No vectors found in suite '{suite_name}' (this may be expected)")
                    except Exception as e:
                        logger.error(f"✗ Error loading vectors from suite '{suite_name}': {e}")

            # Test 5: Load all vectors for the module (no specific suite)
            logger.info("Testing load_vectors for entire module...")
            try:
                all_vectors = vector_source.load_vectors(module_name)
                if all_vectors:
                    logger.info(f"✓ Loaded {len(all_vectors)} total vectors from module '{module_name}'")
                else:
                    logger.warning(f"✓ No vectors found for module '{module_name}' (this may be expected)")
            except Exception as e:
                logger.error(f"✗ Error loading all vectors for module '{module_name}': {e}")

        except Exception as e:
            logger.error(f"✗ Error getting available suites for module '{module_name}': {e}")
            continue

    # Test 6: Test vector ID loading (if we have any vectors)
    if test_modules:
        first_module = test_modules[0]
        logger.info(f"\n--- Testing vector ID loading for module: {first_module} ---")
        try:
            # Get a sample vector ID
            sample_vectors = vector_source.load_vectors(first_module)
            if sample_vectors:
                sample_vector_id = sample_vectors[0].get("vector_id")
                if sample_vector_id:
                    logger.info(f"Testing load_vectors with vector_id: {sample_vector_id}")
                    specific_vector = vector_source.load_vectors(first_module, vector_id=sample_vector_id)
                    if specific_vector:
                        logger.info(f"✓ Successfully loaded specific vector by ID")
                    else:
                        logger.warning(f"✗ Could not load vector by ID: {sample_vector_id}")
                else:
                    logger.warning("No vector_id found in sample vectors")
            else:
                logger.warning("No vectors available to test vector ID loading")
        except Exception as e:
            logger.error(f"✗ Error testing vector ID loading: {e}")

    logger.info(f"\n--- Vector source testing completed for {vector_source_type} ---")


def sanitize_inputs(test_vectors):
    info_field_names = ["sweep_name", "suite_name", "vector_id", "input_hash"]
    header_info = []
    for vector in test_vectors:
        header = dict()
        for field in info_field_names:
            if field in vector:
                header[field] = vector.pop(field)
        if "timestamp" in vector:
            vector.pop("timestamp")
        if "tag" in vector:
            vector.pop("tag")
        header_info.append(header)
    return header_info, test_vectors


def run(test_module, input_queue, output_queue):
    device_generator = get_devices(test_module)
    try:
        device, device_name = next(device_generator)
        logger.info(f"Opened device configuration, {device_name}.")
    except AssertionError as e:
        output_queue.put([False, "DEVICE EXCEPTION: " + str(e), None, None])
        return
    try:
        while True:
            test_vector = input_queue.get(block=True, timeout=1)
            # Use appropriate deserialization based on database backend
            if DATABASE_BACKEND == "postgres":
                test_vector = deserialize_vector_for_postgres(test_vector)
            else:  # DATABASE_BACKEND == "elastic"
                test_vector = deserialize_vector(test_vector)
            try:
                results = test_module.run(**test_vector, device=device)
                if type(results) == list:
                    status, message = results[0]
                    e2e_perf = results[1] / 1000000  # Nanoseconds to milliseconds
                else:
                    status, message = results
                    e2e_perf = None
            except Exception as e:
                status, message = False, str(e)
                e2e_perf = None
            if MEASURE_DEVICE_PERF:
                perf_result = gather_single_test_perf(device, status)
                message = get_updated_message(message, perf_result)
                output_queue.put([status, message, e2e_perf, perf_result])
            else:
                output_queue.put([status, message, e2e_perf, None])
    except Empty as e:
        try:
            # Run teardown in mesh_device_fixture
            next(device_generator)
        except StopIteration:
            logger.info(f"Closed device configuration, {device_name}.")


def execute_suite(test_module, test_vectors, pbar_manager, suite_name, module_name, header_info):
    results = []
    input_queue = Queue()
    output_queue = Queue()
    p = None
    timeout = get_timeout(test_module)
    suite_pbar = pbar_manager.counter(total=len(test_vectors), desc=f"Suite: {suite_name}", leave=False)
    arch = ttnn.get_arch_name()
    reset_util = tt_smi_util.ResetUtil(arch)

    if len(test_vectors) > 1 and not DRY_RUN:
        p = Process(target=run, args=(test_module, input_queue, output_queue))
        p.start()

    for i, test_vector in enumerate(test_vectors):
        vector_id = header_info[i].get("vector_id", "N/A")
        logger.info(f"Executing test: Module='{module_name}', Suite='{suite_name}', Vector ID='{vector_id}'")
        if DRY_RUN:
            print(f"Would have executed test for vector {test_vector}")
            suite_pbar.update()
            continue
        result = dict()

        result["start_time_ts"] = dt.datetime.now()

        # Capture the original test vector data BEFORE any modifications
        original_vector_data = test_vector.copy()

        # Use appropriate deserialization based on database backend
        if DATABASE_BACKEND == "postgres":
            validity = deserialize_for_postgres(test_vector["validity"])
        else:  # DATABASE_BACKEND == "elastic"
            validity = deserialize(test_vector["validity"])

        if validity == VectorValidity.INVALID:
            result["status"] = TestStatus.NOT_RUN
            result["exception"] = "INVALID VECTOR: " + test_vector["invalid_reason"]
            result["e2e_perf"] = None
        else:
            test_vector.pop("invalid_reason")
            test_vector.pop("status")
            test_vector.pop("validity")

            try:
                if MEASURE_PERF:
                    # Run one time before capturing result to deal with compile-time slowdown of perf measurement
                    input_queue.put(test_vector)
                    if p is None:
                        logger.info(
                            "Executing test (first run, e2e perf is enabled) on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                        )
                        run(test_module, input_queue, output_queue)
                    output_queue.get(block=True, timeout=timeout)
                input_queue.put(test_vector)
                if p is None:
                    logger.info(
                        "Executing test on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                    )
                    run(test_module, input_queue, output_queue)
                response = output_queue.get(block=True, timeout=timeout)
                status, message, e2e_perf, device_perf = (
                    response[0],
                    response[1],
                    response[2],
                    response[3],
                )
                if status and MEASURE_DEVICE_PERF and device_perf is None:
                    result["status"] = TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF
                    result["message"] = message
                elif status and MEASURE_DEVICE_PERF:
                    result["status"] = TestStatus.PASS
                    result["message"] = message
                    result["device_perf"] = device_perf
                elif status:
                    result["status"] = TestStatus.PASS
                    result["message"] = message
                else:
                    if "DEVICE EXCEPTION" in message:
                        logger.error(
                            "DEVICE EXCEPTION: Device could not be initialized. The following assertion was thrown: "
                            + message,
                        )
                        logger.info("Device error detected. The suite will be aborted after this test.")
                    if "Out of Memory: Not enough space to allocate" in message:
                        result["status"] = TestStatus.FAIL_L1_OUT_OF_MEM
                    elif "Watcher" in message:
                        result["status"] = TestStatus.FAIL_WATCHER
                    else:
                        result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION
                    result["exception"] = message
                if e2e_perf and MEASURE_PERF:
                    result["e2e_perf"] = e2e_perf
                else:
                    result["e2e_perf"] = None
            except Empty as e:
                if p:
                    logger.warning(f"TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
                    p.terminate()
                    p.join(PROCESS_TERMINATION_TIMEOUT_SECONDS)  # Wait for graceful process termination
                    if p.is_alive():
                        logger.error(f"Child process {p.pid} did not terminate, killing it.")
                        p.kill()
                        p.join()
                    p = None
                    reset_util.reset()

                result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
                result["e2e_perf"] = None
                result["original_vector_data"] = original_vector_data
                result["end_time_ts"] = dt.datetime.now()
                result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                result["host"] = get_hostname()
                result["user"] = get_username()
                suite_pbar.update()
                results.append(result)

                # Check if we should skip remaining tests in the suite
                if SKIP_REMAINING_ON_TIMEOUT:
                    # Skip all remaining tests in the suite
                    logger.info("Skipping remaining tests in suite due to timeout.")
                    for j in range(i + 1, len(test_vectors)):
                        remaining_vector = test_vectors[j]
                        skipped_result = dict()
                        skipped_result["start_time_ts"] = dt.datetime.now()
                        skipped_result["original_vector_data"] = remaining_vector.copy()
                        skipped_result["status"] = TestStatus.NOT_RUN
                        skipped_result["exception"] = "SKIPPED DUE TO PREVIOUS TIMEOUT"
                        skipped_result["e2e_perf"] = None
                        skipped_result["end_time_ts"] = dt.datetime.now()
                        skipped_result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        skipped_result["host"] = get_hostname()
                        skipped_result["user"] = get_username()
                        results.append(skipped_result)
                        suite_pbar.update()

                    # Abort the suite
                    break
                else:
                    logger.info("Continuing with remaining tests in suite despite timeout.")
                    # Continue to the next test vector without breaking

        # Add the original test vector data to the result
        result["original_vector_data"] = original_vector_data

        result["end_time_ts"] = dt.datetime.now()
        result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result["host"] = get_hostname()
        result["user"] = get_username()

        suite_pbar.update()
        results.append(result)

        # Abort the suite if a fatal device error was encountered
        if "DEVICE EXCEPTION" in result.get("exception", ""):
            logger.error("Aborting test suite due to fatal device error.")
            if p and p.is_alive():
                p.terminate()
                p.join()
            break

    if p is not None:
        p.join()

    suite_pbar.close()
    return results


def run_sweeps(
    module_names,
    run_contents=None,
    vector_source_type="elastic",
    elastic_connection_string=None,
    elastic_username=None,
    elastic_password=None,
    sweeps_tag=None,
    file_path=None,
):
    pbar_manager = enlighten.get_manager()
    # Only create the vector source with the credentials if needed
    if vector_source_type == "elastic":
        if not all([elastic_connection_string, elastic_username, elastic_password, sweeps_tag]):
            raise ValueError("Elastic credentials are required when using elastic vector source")

        vector_source = VectorSourceFactory.create_source(
            vector_source_type,
            connection_string=elastic_connection_string,
            username=elastic_username,
            password=elastic_password,
            tag=sweeps_tag,
        )
    elif vector_source_type == "file":
        if not file_path:
            raise ValueError("File path is required when using file vector source")
        vector_source = VectorSourceFactory.create_source(
            vector_source_type,
            file_path=file_path,
        )
    elif vector_source_type == "vectors_export":
        vector_source = VectorSourceFactory.create_source(vector_source_type)
    else:
        raise ValueError(f"Unknown vector source: {vector_source_type}")

    # Test the vector source functionality
    test_vector_source(vector_source, module_names, vector_source_type)

    # TODO: Implement actual sweep execution logic here
    logger.info("TODO: Implement unified execution engine for running the actual tests")

    # Unified processing regardless of source
    for module_name in module_names:
        test_module = importlib.import_module("sweeps." + module_name)
        suites = vector_source.get_available_suites(module_name)
        for suite in suites:
            vectors = vector_source.load_vectors(module_name, suite)
            header_info, test_vectors = sanitize_inputs(vectors)
            execute_suite(test_module, test_vectors, pbar_manager, suite, module_name, header_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Runner",
        description="Run test vector suites from generated vector database.",
    )
    parser.add_argument(
        "--module-name",
        required=False,
        help="Test Module Name(s). When vector source is 'elastic' or 'vectors_export', this can be a comma-separated list (e.g., 'eltwise.unary.relu.relu,matmul.short.matmul').",
    )
    parser.add_argument("--suite-name", required=False, help="Suite of Test Vectors to run, or all tests if omitted.")

    parser.add_argument(
        "--vector-source",
        required=False,
        default="elastic",
        choices=["elastic", "file", "vectors_export"],
        help="Test vector source. Available presets are ['elastic', 'file', 'vectors_export']",
    )

    parser.add_argument("--file-path", required=False, help="Read and execute test vectors from a specified file path.")

    parser.add_argument(
        "--vector-id", required=False, help="Specify vector id with a module name to run an individual test vector."
    )

    parser.add_argument(
        "--result-dest",
        required=False,
        default="elastic",
        choices=["elastic", "postgres", "results_export"],
        help="Specify test result destination. Available presets are ['elastic', 'postgres', 'results_export']",
    )

    parser.add_argument(
        "--watcher", action="store_true", required=False, help="Add this flag to run sweeps with watcher enabled."
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        required=False,
        help="Add this flag to measure e2e perf, for op tests with performance markers.",
    )

    parser.add_argument(
        "--device-perf",
        required=False,
        action="store_true",
        help="Measure device perf using device profiler. REQUIRES PROFILER BUILD!",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=False,
        help="Add this flag to perform a dry run.",
    )

    parser.add_argument(
        "--tag",
        required=False,
        default=os.getenv("USER"),
        help="Custom tag for the vectors you are running. This is to keep copies seperate from other people's test vectors. By default, this will be your username. You are able to specify a tag when generating tests using the generator.",
    )

    parser.add_argument(
        "--skip-modules",
        required=False,
        help="Comma-separated list of modules to skip when running all modules.",
    )

    parser.add_argument(
        "--skip-on-timeout",
        action="store_true",
        required=False,
        help="Skip remaining tests in suite when a test times out. Default behavior is to not skip.",
    )

    args = parser.parse_args(sys.argv[1:])

    # VALIDATION
    validate_arguments(args, parser)

    # SET GLOBAL VARIABLES
    global MODULE_NAME, SUITE_NAME, VECTOR_SOURCE, FILE_PATH, VECTOR_ID
    global RESULT_DESTINATION, WATCHER, MEASURE_PERF, MEASURE_DEVICE_PERF
    global DRY_RUN, SWEEPS_TAG, SKIP_MODULES, SKIP_ON_TIMEOUT, ELASTIC_CONNECTION_STRING

    MODULE_NAME = args.module_name
    SUITE_NAME = args.suite_name
    VECTOR_SOURCE = args.vector_source
    FILE_PATH = args.file_path
    VECTOR_ID = args.vector_id
    RESULT_DESTINATION = args.result_dest
    WATCHER = args.watcher
    MEASURE_PERF = args.perf
    MEASURE_DEVICE_PERF = args.device_perf
    DRY_RUN = args.dry_run
    SWEEPS_TAG = args.tag
    SKIP_MODULES = args.skip_modules
    SKIP_ON_TIMEOUT = args.skip_on_timeout

    # Import Elasticsearch if using elastic database
    if VECTOR_SOURCE == "elastic" or RESULT_DESTINATION == "elastic":
        from elasticsearch import Elasticsearch, NotFoundError
        from framework.elastic_config import *

        ELASTIC_CONNECTION_STRING = get_elastic_url("corp")
    else:
        ELASTIC_CONNECTION_STRING = None

    if WATCHER:
        enable_watcher()

    if MEASURE_DEVICE_PERF:
        enable_profiler()

    # Parse module names and suite names to get run_command
    if MODULE_NAME or SUITE_NAME:
        run_contents_details = []
        if MODULE_NAME:
            run_contents_details.append(f"{MODULE_NAME}")
        if SUITE_NAME:
            run_contents_details.append(f"{SUITE_NAME}")
        run_contents = ", ".join(run_contents_details)
    else:
        run_contents = "all_sweeps"

    logger.info(
        f"Running current sweeps with tag: {SWEEPS_TAG} using {VECTOR_SOURCE} test vector source, outputting to {RESULT_DESTINATION}."
    )

    if SKIP_ON_TIMEOUT:
        logger.info("Timeout behavior: Skip remaining tests in suite when a test times out.")
    else:
        logger.info("Timeout behavior: Continue running remaining tests in suite when a test times out.")

    # Extract credentials only when needed
    elastic_username = None
    elastic_password = None
    elastic_connection_string = None

    if args.vector_source == "elastic" or args.result_dest == "elastic":
        from framework.elastic_config import get_elastic_url

        elastic_username = os.getenv("ELASTIC_USERNAME")
        elastic_password = os.getenv("ELASTIC_PASSWORD")
        # You'll need to determine how to get the elastic connection string
        elastic_connection_string = get_elastic_url("corp")  # or based on args

    # Parse modules for running specific tests
    module_names = None
    if MODULE_NAME:
        # Always support comma-separated module names now
        if "," in MODULE_NAME:
            module_names = [name.strip() for name in MODULE_NAME.split(",")]
            logger.info(f"Running multiple modules: {module_names}")
        else:
            module_names = MODULE_NAME
    else:
        module_names = list(get_all_modules())
        logger.info(f"Running all modules.")
        if SKIP_MODULES:
            skip_modules_set = {name.strip() for name in SKIP_MODULES.split(",")}
            module_names = [name for name in module_names if name not in skip_modules_set]
            logger.info(f"But skipping: {', '.join(skip_modules_set)}")

    import ttnn

    run_sweeps(
        module_names,
        run_contents=run_contents,
        vector_source_type=args.vector_source,
        elastic_connection_string=elastic_connection_string,
        elastic_username=elastic_username,
        elastic_password=elastic_password,
        sweeps_tag=args.tag,
        file_path=args.file_path,
    )

    # if WATCHER:
    #     disable_watcher()

    # if MEASURE_DEVICE_PERF:
    #     disable_profiler()
