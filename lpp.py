import os
import sys
from warnings import warn


def parse_values(line, target_type):
    result = {}
    tokens = line.split()

    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Expected token in format key=value, got '{token}'")
        key, value = token.split("=", 1)
        result[key] = target_type(value)

    return result


def parse_log_file(filename):
    print(f"Parsing log file: {filename}")
    failed_lines = []
    passed_lines = []
    results = []
    id = 0

    with open(filename, "r") as f:
        all_lines = f.readlines()
        lines = [line.strip() for line in all_lines if line.strip()]
        for i in range(len(lines)):
            if "0:(x=0,y=0):TR0: Case: " in lines[i]:
                case_line = lines[i + 0]
                zone_line = lines[i + 1]
                sync_line = lines[i + 2]
                case_start = case_line.find("0:(x=0,y=0):TR0: Case: ")
                zone_start = zone_line.find("0:(x=0,y=0):TR0: Zone: ")
                sync_start = sync_line.find("0:(x=0,y=0):TR0: Sync: ")

                if case_start == -1 or zone_start == -1 or sync_start == -1:
                    raise ValueError(
                        f"Could not find expected case/zone/sync lines at index {i}:\n  {lines[i]}\n  {lines[i+1]}\n  {lines[i+2]}"
                    )

                case_dict = parse_values(case_line[case_start + len("0:(x=0,y=0):TR0: Case: ") :], int)
                zone_dict = parse_values(zone_line[zone_start + len("0:(x=0,y=0):TR0: Zone: ") :], int)
                sync_dict = parse_values(sync_line[sync_start + len("0:(x=0,y=0):TR0: Sync: ") :], int)

                case_dict["ID"] = id
                id += 1

                results.append((case_dict, zone_dict, sync_dict))
            if lines[i].startswith("FAILED "):
                failed_lines.append(lines[i])
            if lines[i].startswith("PASSED "):
                passed_lines.append(lines[i])

    return failed_lines, passed_lines, results


def consolidate_sync(result):
    sync_data = {}
    for _, _, sync_dict in result:
        if len(sync_data) == 0:
            sync_data = dict(sync_dict)
        else:
            for k, v in sync_dict.items():
                if k.endswith("_min") or k == "min":
                    sync_data[k] = min(sync_data[k], v)
                elif k.endswith("_max") or k == "max":
                    sync_data[k] = max(sync_data[k], v)
                elif k.endswith("_sum") or k == "sum" or k == "count":
                    sync_data[k] += v
                else:
                    raise ValueError(f"Unexpected key in sync dict: {k}")

    count = sync_data.pop("count", 1)

    prefixes = {}
    for k, v in sync_data.items():
        if "_min" in k:
            prefix = k.replace("_min", "")
            suffix = "min"
        elif "_max" in k:
            prefix = k.replace("_max", "")
            suffix = "max"
        elif "_sum" in k:
            prefix = k.replace("_sum", "")
            suffix = "sum"
        else:
            raise ValueError(f"Unexpected key in sync data: {k}")

        if prefix not in prefixes:
            prefixes[prefix] = {}
        prefixes[prefix][suffix] = v

    for prefix, metrics in prefixes.items():
        if "sum" in metrics and count > 0:
            metrics["avg"] = metrics["sum"] / count
        if "min" in metrics and "max" in metrics:
            metrics["spread"] = metrics["max"] - metrics["min"]

    return prefixes


def log_to_results(parse_result):
    print("Converting parsed log to results")
    failed_lines, passed_lines, results = parse_result

    total_results = len(results)
    passed_count = len(passed_lines)
    failed_count = len(failed_lines)
    total_tests = failed_count + passed_count

    if total_results != total_tests:
        warn(f"Number of parsed results ({total_results}) does not match total tests ({total_tests})")

    print("\n====TEST SUMMARY====")
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

    inner_loops = set()
    outer_loops = set()
    for result in results:
        case_dict, zone_dict, sync_dict = result
        zone_count = zone_dict.get("count", 0)
        sync_count = sync_dict.get("count", 0)
        inner_count = zone_dict.get("inner", 0)

        if zone_count == 0 or sync_count == 0 or inner_count == 0:
            warn(
                f"Zone count, sync count, or inner count is zero for case ID={case_dict.get('ID', 'unknown')} (zone_count={zone_count}, sync_count={sync_count}, inner_count={inner_count})"
            )

        if zone_count != sync_count:
            warn(
                f"Zone count ({zone_count}) does not match sync count ({sync_count}) for case ID={case_dict.get('ID', 'unknown')}"
            )

        if inner_count % zone_count != 0:
            warn(
                f"Inner count ({inner_count}) is not a multiple of zone count ({zone_count}) for case ID={case_dict.get('ID', 'unknown')}"
            )

        inner_loops.add(inner_count // zone_count)
        outer_loops.add(zone_count)

    if len(inner_loops) > 1:
        warn(f"Multiple different inner loop counts found: {inner_loops}")
        print(f"Inner loop counts: {inner_loops}")
    if len(outer_loops) > 1:
        warn(f"Multiple different outer loop counts found: {outer_loops}")
        print(f"Outer loop counts: {outer_loops}")

    print(f"\nLoop factor (inner loops per zone): {inner_loops.pop()}")
    print(f"Outer loop count (zones per test): {outer_loops.pop()}")

    sync = consolidate_sync(results)

    print("\nSync results:")
    for key, stats in sync.items():
        print(
            f"{key:<12}\t=>\tmin={stats['min']:7.1f}\tmax={stats['max']:7.1f}\tspread={stats['spread']:7.1f}\tavg={stats['avg']:7.1f}"
        )

    result_dict = {}

    for case_dict, zone_dict, sync_dict in results:
        case_id = case_dict["ID"]
        avg_time = zone_dict["sum"] / (zone_dict["inner"])
        min_time = zone_dict["min"] / (zone_dict["inner"] // zone_dict["count"])
        max_time = zone_dict["max"] / (zone_dict["inner"] // zone_dict["count"])
        dev_time = max_time - min_time
        perf_dict = {"avg": avg_time, "min": min_time, "max": max_time, "dev": dev_time}
        result_dict[case_id] = (case_dict, perf_dict)

    return result_dict


def load_results(filename):
    print(f"Loading results from: {filename}")
    result_dict = {}

    with open(filename, "r") as f:
        all_lines = f.readlines()
        lines = [line.strip().replace("\t", " ") for line in all_lines if line.strip()]
        for line in lines:
            splits = [s.strip() for s in line.strip().split(" ") if s.strip()]

            arrow_idx = splits.index("=>")
            pipe_idx = splits.index("|")

            if arrow_idx == -1 or pipe_idx == -1:
                raise ValueError(f"Could not find expected '=>' or '|' in line: {line}")

            case_part = splits[:arrow_idx]
            perf_part = splits[arrow_idx + 1 : pipe_idx]
            case_part = [case_part[i] + case_part[i + 1] for i in range(0, len(case_part) - 1, 2)]
            perf_part = [perf_part[i] + perf_part[i + 1] for i in range(0, len(perf_part) - 1, 2)]

            case_dict = parse_values(" ".join(case_part), int)
            perf_dict = parse_values(" ".join(perf_part), float)

            result_dict[case_dict["ID"]] = (case_dict, perf_dict)

    return result_dict


def load_from_file(filename):
    if filename.endswith(".res"):
        results = load_results(filename)
        return results
    else:
        parse_result = parse_log_file(filename)
        results = log_to_results(parse_result)
        return results


def print_results(result_dict, baseline_dict, file=sys.stdout):
    db_cyc_values = []
    db_imp_values = []
    db_dev_values = []

    print("\nPerf results:")
    for case_id, (case_dict, perf_dict) in result_dict.items():
        baseline = baseline_dict.get(case_id, None) if baseline_dict else None

        variant_string = [f"ID={case_id:4d}"]
        for key in case_dict:
            if key != "ID":
                variant_string.append(f"{key}={case_dict[key]:4d}")
        variant_string = " ".join(variant_string)

        if baseline:
            baseline_case_dict, baseline_perf_dict = baseline
            if baseline_case_dict != case_dict:
                warn(f"Baseline case dict does not match result case dict for ID={case_id}")
                print(f"  Baseline case dict: {baseline_case_dict}")
                print(f"  Result case dict:   {case_dict}")
            db_cyc = baseline_perf_dict["avg"] - perf_dict["avg"]
            db_imp = (db_cyc / baseline_perf_dict["avg"]) * 100
            db_dev = baseline_perf_dict["dev"] - perf_dict["dev"]
            db_cyc_values.append(db_cyc)
            db_imp_values.append(db_imp)
            db_dev_values.append(db_dev)
        else:
            db_cyc, db_imp, db_dev = (float("nan"), float("nan"), float("nan"))

        print(
            f"{variant_string:<12}\t=>\tavg={perf_dict['avg']:7.1f}\tmin={perf_dict['min']:7.1f}\tmax={perf_dict['max']:7.1f}\tdev={perf_dict['dev']:7.1f}\t|\tdb={db_cyc:7.1f}\tdb_imp={db_imp:7.1f}\tdb_dev={db_dev:7.1f}",
            file=file,
        )

    if baseline_dict:
        print(f"\nBaseline comparison statistics:")
        db_cyc_stats = {
            "min": min(db_cyc_values),
            "max": max(db_cyc_values),
            "avg": sum(db_cyc_values) / len(db_cyc_values),
            "spread": max(db_cyc_values) - min(db_cyc_values),
        }
        db_imp_stats = {
            "min": min(db_imp_values),
            "max": max(db_imp_values),
            "avg": sum(db_imp_values) / len(db_imp_values),
            "spread": max(db_imp_values) - min(db_imp_values),
        }
        db_dev_stats = {
            "min": min(db_dev_values),
            "max": max(max(db_dev_values), 0),
            "avg": sum(db_dev_values) / len(db_dev_values),
            "spread": max(db_dev_values) - min(db_dev_values),
        }
        print(
            f"db    \t=>\tmin={db_cyc_stats['min']:7.1f}\tmax={db_cyc_stats['max']:7.1f}\tspread={db_cyc_stats['spread']:7.1f}\tavg={db_cyc_stats['avg']:7.1f}"
        )
        print(
            f"db_imp\t=>\tmin={db_imp_stats['min']:7.1f}\tmax={db_imp_stats['max']:7.1f}\tspread={db_imp_stats['spread']:7.1f}\tavg={db_imp_stats['avg']:7.1f}"
        )
        print(
            f"db_dev\t=>\tmin={db_dev_stats['min']:7.1f}\tmax={db_dev_stats['max']:7.1f}\tspread={db_dev_stats['spread']:7.1f}\tavg={db_dev_stats['avg']:7.1f}"
        )


if __name__ == "__main__":
    COMMANDS = ["load", "save", "base"]
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Commands:")
        print("  load <input_file='log.log'> <baseline_file='baseline.res'>")
        print("  save <input_file='log.log'> <results_file='results.res'>")
        print("  base <input_file='log.log'> <results_file='baseline.res'>")
    command = sys.argv[1]
    if command == "load":
        input_file = sys.argv[2] if len(sys.argv) > 2 else "log.log"
        baseline_file = sys.argv[3] if len(sys.argv) > 3 else "baseline.res"

        input_data = load_from_file(input_file)
        if os.path.exists(baseline_file):
            baseline_data = load_from_file(baseline_file)
        else:
            baseline_data = None

        print_results(input_data, baseline_data)
    if command == "save":
        input_file = sys.argv[2] if len(sys.argv) > 2 else "log.log"
        results_file = sys.argv[3] if len(sys.argv) > 3 else "results.res"

        input_data = load_from_file(input_file)
        print_results(input_data, None, file=open(results_file, "w"))
    if command == "base":
        input_file = sys.argv[2] if len(sys.argv) > 2 else "log.log"
        results_file = sys.argv[3] if len(sys.argv) > 3 else "baseline.res"

        input_data = load_from_file(input_file)
        print_results(input_data, None, file=open(results_file, "w"))
