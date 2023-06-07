import csv

import pandas as pd


def import_host_profile_log(logPath):
    hostData = {}
    with open(logPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        for lineCount, row in enumerate(csvReader):
            if lineCount > 0:
                tmpDict = {}
                sectionName = row[0].strip()
                functionName = row[1].strip()
                tmpDict["start"] = int(row[2])
                tmpDict["end"] = int(row[3])
                tmpDict["diff"] = int(row[4])
                if functionName in hostData.keys():
                    hostData[functionName]["timeseries"].append(tmpDict)
                else:
                    hostData[functionName] = {"timeseries": [tmpDict]}
            else:
                assert "Start timer count [ns]" in row[2], f"CSV {logPath} has bad header format"
    return hostData


def host_analysis(hostData):
    for functionName, calls in hostData.items():
        tmpDF = pd.DataFrame(calls["timeseries"])
        calls["stats"] = {
            "Count": tmpDF.loc[:, "diff"].count(),
            "Average": tmpDF.loc[:, "diff"].mean(),
            "Max": tmpDF.loc[:, "diff"].max(),
            "Min": tmpDF.loc[:, "diff"].min(),
            "Range": tmpDF.loc[:, "diff"].max() - tmpDF.loc[:, "diff"].min(),
            "Median": tmpDF.loc[:, "diff"].median(),
            "Sum": tmpDF.loc[:, "diff"].sum(),
        }
    return hostData


def import_host_log_run_stats(logPath):
    hostData = import_host_profile_log(logPath)
    return host_analysis(hostData)
