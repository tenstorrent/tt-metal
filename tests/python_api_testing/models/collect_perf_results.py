from os import listdir
from os.path import isfile, join


def merge_perf_files():
    mypath = "./"
    csvfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and ".csv" in f]
    merge_res = open("perf.csv", "w")
    merge_res.write(
        "model, reference_time(s), first_iter_time (s), second_iter_time (s), compiler_time (s), throughput (it/s) \n"
    )

    for csvfile in csvfiles:
        row_name = csvfile.replace("perf_", "")
        row_name = row_name.replace(".csv", "")

        f = open(f"./{csvfile}", "r")
        f.readline()
        row = f.readline()
        merge_res.write(f"{row_name}, {row}\n")

    merge_res.close()


merge_perf_files()
