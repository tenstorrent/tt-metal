# Manually copy sandbox/ll_buda/tests/estimate_core_clock_host.cpp
# to ll_buda and temporarly add it to makefil to be built
# After running the ll_buda test, run this script to get the estimate on clock freq
import csv

hostData = {}

with open("tools/profiler/profile_log.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            sessionName = row[0].strip()
            funcName = row[1].strip()
            timeData = {
                "startTime": int(row[2]),
                "stopTime": int(row[3]),
                "timeDiff": int(row[4]),
            }

            if sessionName in hostData.keys():
                hostData[sessionName][funcName] = timeData
            else:
                hostData[sessionName] = {funcName: timeData}

        line_count += 1

deviceData = {}

with open("tools/profiler/profile_log_kernel.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count > 1:
            chipID = int(row[0])
            core = (int(row[1]), int(row[2]))
            risc = row[3].strip()
            timerID = int(row[4])
            timeData = int(row[5])

            if chipID in deviceData.keys():
                if core in deviceData[chipID].keys():
                    if risc in deviceData[chipID][core].keys():
                        deviceData[chipID][core][risc]["timeSeries"].append(
                            (timerID, timeData)
                        )
                    else:
                        deviceData[chipID][core][risc] = {
                            "timeSeries": [(timerID, timeData)]
                        }
                else:
                    deviceData[chipID][core] = {
                        risc: {"timeSeries": [(timerID, timeData)]}
                    }
            else:
                deviceData[chipID] = {
                    core: {risc: {"timeSeries": [(timerID, timeData)]}}
                }

        line_count += 1


def calculate_diffs_timeseries(timeSeries, startID, endID):
    timeSeries.sort(key=lambda x: x[1])
    diffs = {}
    for i, startTrav in enumerate(timeSeries):
        if startID == startTrav[0] and startTrav[1] not in diffs.keys():
            for endTrav in timeSeries[i + 1 :]:
                if endID == endTrav[0]:
                    diffs[startTrav[1]] = endTrav[1] - startTrav[1]
                    break
    return diffs


for dev in deviceData.keys():
    for core in deviceData[dev].keys():
        for risc in deviceData[dev][core].keys():
            enitre_main = (1, 4)
            deviceData[dev][core][risc]["diffs"] = {
                (1, 4): calculate_diffs_timeseries(
                    deviceData[dev][core][risc]["timeSeries"], 1, 4
                )
            }


briscDiffs = deviceData[0][(0, 0)]["BRISC"]["diffs"][(1, 4)].values()

deviceTimeIncrease = max(briscDiffs) - min(briscDiffs)
hostTimeIncrease = (
    hostData["Long"]["LaunchKernels"]["timeDiff"]
    - hostData["Short"]["LaunchKernels"]["timeDiff"]
)
print(f"Core frequency estimate: {deviceTimeIncrease / hostTimeIncrease:.2f} GHz")
