from os import listdir, environ
from os.path import isfile, join
import time
environ['TZ'] = 'America/Toronto'
time.tzset()
today = time.strftime("%Y_%m_%d")

def merge_perf_files():
    mypath = "./"
    csvfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f"{today}.csv" in f]
    merge_res = open(f"Models_Perf_{today}.csv", "w")
    merge_res.write(
        "Model, Setting, Batch, First Run (sec), Second Run (sec), Compiler Time (sec), Inference Time GS (sec), Throughput GS (Batch*inf/sec), Inference Time CPU (sec), Throughput CPU (Batch*inf/sec) \n"
                    )

    for csvfile in csvfiles:
        row_name = csvfile.replace("perf_", "")
        row_name = row_name.replace(f"{today}", "")
        row_name = row_name.replace(".csv", "")

        f = open(f"./{csvfile}", "r")
        f.readline()
        row = f.readline()
        merge_res.write(f"{row}\n")

    merge_res.close()


if __name__ == "__main__":
    merge_perf_files()
