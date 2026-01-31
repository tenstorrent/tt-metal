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


# Usage
failed_lines, passed_lines, tr0_lines = parse_log_file()

# Validate counts and print summary
total_tr0 = len(tr0_lines)
total_tests = len(failed_lines) + len(passed_lines)
passed_count = len(passed_lines)
failed_count = len(failed_lines)

if total_tests != total_tr0:
    print(f"Warning: Test count mismatch. Found {total_tests} test results but {total_tr0} TR0 lines")

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

print("\nTR0 Lines:")
for line in tr0_lines:
    print(line)
