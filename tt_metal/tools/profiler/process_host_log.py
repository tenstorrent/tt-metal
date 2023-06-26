import csv

import pandas as pd


def import_host_profile_log(logPath):
    hostData = {}
    with open(logPath) as csvFile:
        csvReader = csv.DictReader(csvFile, delimiter=",")
        for row in csvReader:
            try:
                tmpDict = {}
                functionName = row["Name"].strip()
                tmpDict["start"] = int(row[" Start timer count [ns]"].strip())
                tmpDict["end"] = int(row[" Stop timer count [ns]"].strip())
                tmpDict["diff"] = int(row[" Delta timer count [ns]"].strip())
                if functionName in hostData.keys():
                    hostData[functionName]["timeseries"].append(tmpDict)
                else:
                    hostData[functionName] = {"timeseries": [tmpDict]}
            except KeyError as e:
                assert False, f"CSV {opLogPath} has bad header format"
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
