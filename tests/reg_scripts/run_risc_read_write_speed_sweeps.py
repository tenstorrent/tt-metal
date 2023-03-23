from reg_scripts.common import run_single_test, run_process_and_get_result, report_tests, TestEntry, run_single_test_capture_output

from operator import pow
from functools import partial

import re
import random
import itertools
import dataclasses

def extract_float_from_text(text, pattern_before_float):
    # the floating point number regex is from the Python docs
    try:
        m = re.search(pattern_before_float+'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', text)
    except:
        raise Exception("Couldn't find the float number I was searching for")

    float_num = m.group(1)

    return float_num

def run_test_with_params_and_extract_floats(test_entry, patterns_before_float):

    print(f"RUNNING LLRT TEST - {test_entry}")
    result = run_single_test_capture_output("llrt", test_entry, timeout=60)
    assert result.returncode == 0, f"{test_entry} failed"

    extracted_floats = ()
    for pattern in patterns_before_float:
        extracted_floats = extracted_floats + (extract_float_from_text(result.stdout.decode("utf-8"), pattern),)

    return extracted_floats


@dataclasses.dataclass(frozen=True)
class SweepResult:
    buffer_size: int
    num_repetitions: int
    total_bytes_moved: int
    transaction_size: int
    extra_params: list
    stats: list


def sweep_transaction_size(test_name, lower_bound, upper_bound, buffer_size, num_repetitions, stats_patterns_to_extract, *extra_params):
    sweep_results = []
    transaction_size = lower_bound

    while transaction_size <= upper_bound:
        extra_params_str = " ".join(map(str, extra_params))
        test_entry = TestEntry("llrt/tests/" + test_name, test_name, f'{buffer_size} {num_repetitions} {transaction_size} {extra_params_str}')
        total_bytes_moved = buffer_size * num_repetitions

        assert isinstance(stats_patterns_to_extract, list)
        assert len(stats_patterns_to_extract)

        stats = run_test_with_params_and_extract_floats(test_entry, stats_patterns_to_extract)

        sweep_result = SweepResult(
            buffer_size=buffer_size,
            num_repetitions=num_repetitions,
            total_bytes_moved=total_bytes_moved,
            transaction_size=transaction_size,
            extra_params=extra_params,
            stats=stats,
        )
        sweep_results.append(sweep_result)
        transaction_size *= 2

    return sweep_results


def print_sweep_results(sweep_results, print_extra_params: bool = True):
    stringify_list = lambda list_: list(map(str, list_))

    print()
    for result in sweep_results:
        result_str = ", ".join(stringify_list([
            result.buffer_size,
            result.num_repetitions,
            result.total_bytes_moved,
            result.transaction_size,
            ]))

        if print_extra_params:
            result_str += ", " + ", ".join(stringify_list(result.extra_params))

        result_str += ", " + ", ".join(stringify_list(result.stats))

        print(result_str)
    print()

def run_sweep(test_name, num_repetitions):
    sweep_results = sweep_transaction_size(test_name, 32, 8192*4, 393216, num_repetitions, ["BRISC time: ", "speed GB/s: "], 1, 1)
    print_sweep_results(sweep_results)

    sweep_results = sweep_transaction_size(test_name, 32, 8192, 65536, num_repetitions, ["BRISC time: ", "speed GB/s: "], 1, 1)
    print_sweep_results(sweep_results)

    sweep_results = sweep_transaction_size(test_name, 32, 8192, 393216, num_repetitions, ["BRISC time: ", "speed GB/s: "], 1, 0)
    print_sweep_results(sweep_results)

    sweep_results = sweep_transaction_size(test_name, 32, 8192, 65536, num_repetitions, ["BRISC time: ", "speed GB/s: "], 1, 0)
    print_sweep_results(sweep_results)

def run_sweep_banked_single_core(test_name):
    print("-------- SWEEP BANKED SINGLE CORE --------")
    print("Will print read time, speed, write time, speed at the end of test entries\n\n")

    num_dram_channels = 8
    extra_params_for_banked_sweep = create_extra_args_for_banked_sweep(num_dram_channels, [5, 4])

    sweep_results = sweep_transaction_size(test_name, 32, 8192, 393216, 10000, ["Read speed GB/s: ", "Write speed GB/s: "], *extra_params_for_banked_sweep)
    print_sweep_results(sweep_results)

    sweep_results = sweep_transaction_size(test_name, 32, 8192, 65536, 10000, ["Read speed GB/s: ", "Write speed GB/s: "], *extra_params_for_banked_sweep)
    print_sweep_results(sweep_results)

def create_extra_args_for_banked_sweep(num_dram_channels, unrolled_cores):
    assert num_dram_channels > 0

    pow_of_2 = partial(pow, 2)
    assert num_dram_channels in map(pow_of_2, range(4))

    num_dram_channels_in_bits = num_dram_channels.bit_length() - 1
    return [num_dram_channels_in_bits] + list(unrolled_cores)

def run_sweep_banked_multi_core(test_name):
    print("-------- SWEEP BANKED MULTI CORE --------")
    possible_x_cores = tuple(range(1, 13))
    possible_y_cores = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
    possible_cores = tuple(itertools.product(possible_x_cores, possible_y_cores))
    # By default is sys time
    random.seed()

    get_n_random_cores = lambda n_: random.sample(possible_cores, k=n_)
    unroll_list_of_cores = lambda cores_: itertools.chain.from_iterable(cores_)

    # under 800KB limit in L1
    base_buffer_size = 8192*97

    print("---- SWEEP BANKED MULTI CORE (2 core)----")
    cores = get_n_random_cores(2)

    sweep_results = sweep_transaction_size(test_name, 256, 8192, base_buffer_size * 2, 1000, ["Read speed GB/s: ", "Write speed GB/s: "], *create_extra_args_for_banked_sweep(8, unroll_list_of_cores(cores)))
    print_sweep_results(sweep_results, print_extra_params=False)

    print("---- SWEEP BANKED MULTI CORE (4 core)----")
    cores = get_n_random_cores(4)

    sweep_results = sweep_transaction_size(test_name, 256, 8192, base_buffer_size * 4, 1000, ["Read speed GB/s: ", "Write speed GB/s: "], *create_extra_args_for_banked_sweep(8, unroll_list_of_cores(cores)))
    print_sweep_results(sweep_results, print_extra_params=False)

    print("---- SWEEP BANKED MULTI CORE (8 core)----")
    cores = get_n_random_cores(8)

    sweep_results = sweep_transaction_size(test_name, 256, 8192, base_buffer_size * 8, 1000, ["Read speed GB/s: ", "Write speed GB/s: "], *create_extra_args_for_banked_sweep(8, unroll_list_of_cores(cores)))
    print_sweep_results(sweep_results, print_extra_params=False)

    print("---- SWEEP BANKED MULTI CORE (16 core)----")
    cores = get_n_random_cores(16)

    sweep_results = sweep_transaction_size(test_name, 256, 8192, base_buffer_size * 16, 1000, ["Read speed GB/s: ", "Write speed GB/s: "], *create_extra_args_for_banked_sweep(8, unroll_list_of_cores(cores)))
    print_sweep_results(sweep_results, print_extra_params=False)

    print("---- SWEEP BANKED MULTI CORE (32 core)----")
    cores = get_n_random_cores(32)

    sweep_results = sweep_transaction_size(test_name, 256, 8192, base_buffer_size * 32, 1000, ["Read speed GB/s: ", "Write speed GB/s: "], *create_extra_args_for_banked_sweep(8, unroll_list_of_cores(cores)))
    print_sweep_results(sweep_results, print_extra_params=False)

    print("---- SWEEP BANKED MULTI CORE (64 core)----")
    cores = get_n_random_cores(64)

    sweep_results = sweep_transaction_size(test_name, 256, 8192, base_buffer_size * 64, 1000, ["Read speed GB/s: ", "Write speed GB/s: "], *create_extra_args_for_banked_sweep(8, unroll_list_of_cores(cores)))
    print_sweep_results(sweep_results, print_extra_params=False)

    print("---- SWEEP BANKED MULTI CORE (120 core)----")
    cores = get_n_random_cores(120)

    sweep_results = sweep_transaction_size(test_name, 256, 8192, base_buffer_size * 120, 1000, ["Read speed GB/s: ", "Write speed GB/s: "], *create_extra_args_for_banked_sweep(8, unroll_list_of_cores(cores)))
    print_sweep_results(sweep_results, print_extra_params=False)

if __name__ == "__main__":
    run_sweep("test_run_risc_read_speed", 10000)

    run_sweep("test_run_risc_write_speed", 10000)

    #run_sweep_banked_single_core("test_run_risc_rw_speed_banked_dram")

    #run_sweep_banked_multi_core("test_run_risc_rw_speed_banked_dram")
