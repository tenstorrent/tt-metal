def parse_log_file(filename="log.txt"):
    """Parse log file and return lines starting with 'FAILED ', 'PASSED ' and '0:(x=0,y=0):TR0: version '"""
    failed_lines = []
    passed_lines = []
    tr0_lines = []

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("FAILED "):
                    failed_lines.append(line)
                if line.startswith("PASSED "):
                    passed_lines.append(line)
                if line.startswith("0:(x=0,y=0):TR0: version "):
                    tr0_lines.append(line)
    except FileNotFoundError:
        print(f"Error: {filename} not found")

    return failed_lines, passed_lines, tr0_lines


def cosolidate_tr0(tr0_lines):
    test_dict = {}
    for line in tr0_lines:
        version_idx = line.index("version")
        line = line[version_idx:]
        parts = line.rsplit(" ", 1)
        variant = parts[0]
        time = parts[1]
        if variant not in test_dict:
            test_dict[variant] = [int(time)]
        else:
            test_dict[variant].append(int(time))
    return test_dict


# Usage
failed_lines, passed_lines, tr0_lines = parse_log_file()

perf_results = cosolidate_tr0(tr0_lines)

# Validate counts and print summary
total_results = len(perf_results)
total_tests = len(failed_lines) + len(passed_lines)
passed_count = len(passed_lines)
failed_count = len(failed_lines)

if total_tests != total_results:
    print(f"Warning: Test count mismatch. Found {total_tests} test results but {total_results} TR0 lines")

print(f"Total tests: {total_tests}")
print(f"Passed: {passed_count}")
print(f"Failed: {failed_count}")

if failed_count == 0:
    print("====all tests  PASSED====")
else:
    print("!!!!SOME TESTS FAILED!!!!")

if failed_count > 0:
    print("\nFailed Tests:")
    for line in failed_lines:
        print(line)
    print("\nPassed Tests:")
    for line in passed_lines:
        print(line)

print("\nPerf results:")
for variant, times in perf_results.items():
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(
        f"{variant} \t=> \tavg: {avg_time:7.1f}, \tmin: {min_time:4d}, \tmax: {max_time:4d}, \tdev: {max_time - min_time:4d}"
    )
    # for t in times:
    #     print(f"{t}")
