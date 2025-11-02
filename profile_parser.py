import sys

TRACE_NAME = "profile_log_device.csv"
TRACE_NAME = sys.argv[1] if len(sys.argv) > 1 else TRACE_NAME


def open_trace(fname):
    file = open(fname, "r")
    lines = file.readlines()
    file.close()
    lines = lines[2:]
    for i in range(len(lines)):
        lines[i] = lines[i].split(",")
    return lines


def is_start_of_test(trace):
    return trace[8] == "BRISC-FW" and trace[9] == "ZONE_START"


def split_tests(trace):
    tests = []
    test = []
    for line in trace:
        if is_start_of_test(line):
            if test != []:
                tests.append(test)
            test = []
        test.append(line)
    if test != []:
        tests.append(test)
    return tests


def filter_traces(traces, blacklist):
    for i in range(len(traces)):
        tmp = traces[i]
        traces[i] = []
        for trace in tmp:
            blacklisted = False
            for kw in blacklist:
                if kw in trace[8]:
                    blacklisted = True
                    break
            if not blacklisted:
                traces[i].append(trace)
    return traces


def condense_traces(traces):
    for i in range(len(traces)):
        tmp = traces[i]
        traces[i] = []
        zonedict = {}
        for trace in tmp:
            name = trace[8].lower().replace(" ", "-") + ("-s" if trace[9] == "ZONE_START" else "-e")
            tt = (trace[3][-1], name, trace[5])
            if name not in zonedict:
                zonedict[name] = [tt]
            else:
                zonedict[name].append(tt)
        for key in zonedict:
            values = sorted(zonedict[key], key=lambda x: x[0])
            values = [int(x[2]) for x in values]
            traces[i].append((key, values))
        traces[i].sort(key=lambda x: min(x[1]))

    return traces


def generate_test_pattern():
    tests = []
    for test in ["TilizeMatmul", "TilizeMatmulFused"]:
        for rt_dim in [1, 2, 4]:
            for ct_dim in [1, 2, 4]:
                for kt_dim in [1, 2, 4]:
                    for reuse_a in [0, 1, 2]:
                        if rt_dim == 4 and ct_dim == 4:
                            continue
                        tests.append((test, rt_dim, ct_dim, kt_dim, reuse_a))
    return tests


def get_test_header():
    return "test,rt_dim,ct_dim,kt_dim,reuse_a,"


def get_zone(zone, test):
    parts = zone.split(" ")
    parts = [x.strip() for x in parts]
    operator = parts[0]
    zone = parts[1]
    values = -1
    for line in test[1]:
        if zone == line[0]:
            values = line[1]
            break
    if values == -1:
        return None
    if operator == "min":
        return min(values)
    elif operator == "max":
        return max(values)
    elif operator == "avg":
        return sum(values) / len(values)
    elif operator == "cross":
        return max(values) - min(values)
    elif operator == "unp":
        return values[0]
    elif operator == "mth":
        return values[1]
    elif operator == "pck":
        return values[2]
    else:
        return None


def run_query_on_test(test, query):
    lf = 1
    if " lf " in query:
        parts = query.split("lf")
        parts = [x.strip() for x in parts]
        lf = int(parts[1])
        query = parts[0]
    if " to " in query:
        parts = query.split("to")
        parts = [x.strip() for x in parts]
        start_point = get_zone(parts[0], test)
        end_point = get_zone(parts[1], test)
        if start_point == None or end_point == None:
            value = None
        else:
            value = end_point - start_point
    else:
        value = get_zone(query, test)
    if value != None:
        value = value / lf
    print(value, end=",")


def run_queries_on_test(test, queries):
    print(*test[0], sep=",", end=",")
    for query in queries:
        run_query_on_test(test, query)
    print("")


def run_queries(test_traces, queries):
    header = get_test_header()
    for query in queries:
        header += query + ","
    print(header)
    for test in test_traces:
        run_queries_on_test(test, queries)


def get_zones(test_traces):
    zones = []
    for test in test_traces:
        for zone in test[1]:
            if zone[0] not in zones:
                zones.append(zone[0])
    return zones


trace = open_trace(TRACE_NAME)
test_traces = split_tests(trace)
test_traces = filter_traces(test_traces, ["-FW", "-KERNEL"])
test_traces = condense_traces(test_traces)
tests = generate_test_pattern()

for i in range(len(test_traces)):
    test_traces[i] = (tests[i], test_traces[i])

# print(*test_traces, sep="\n")

print(*get_zones(test_traces), sep="\n")

# write query here

queries = [
    "cross b0-e",
    "cross b1-e",
    "cross b2-e",
    "min tilize-seq-s to max tilize-sync-e lf 64",
    "unp tilize-sync-e to mth tilize-sync-e",
    "mth tilize-sync-e to pck tilize-sync-e",
    "min matmul-loop-s to max matmul-sync-e lf 64",
    "unp matmul-sync-e to mth matmul-sync-e",
    "mth matmul-sync-e to pck matmul-sync-e",
]

run_queries(test_traces, queries)
