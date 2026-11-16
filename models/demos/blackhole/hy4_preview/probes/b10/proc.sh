#!/bin/bash
# proc.sh <tag> <opid_lo> <opid_hi> : extract per-op/per-chip tables of a profile run (cpp csv, raw per-core zones of ops lo..hi, dispatch go signals)
cd /mnt/tt-data/ssinghal/hy4-logs/program/b10; T=$1; LO=$2; HI=$3
G=/home/ttuser/ssinghal/combo_cache/b10_run/$T/gen/profiler/.logs
cp $G/cpp_device_perf_report.csv data/${T}_cpp.csv
python3 gapan.py data/${T}_cpp.csv logs/${T}_ops.json --json data/${T}_gap.json > data/${T}_gap.txt
awk -F, -v lo=$LO -v hi=$HI 'NR>2 { id=int($8/1024); if (id>=lo && id<=hi && ($11=="BRISC-FW"||$11=="BRISC-KERNEL"||$11=="NCRISC-FW"||$11=="NCRISC-KERNEL"||$11=="TRISC-KERNEL")) print $1","$2","$3","$4","$6","$8","$11","$12 }' $G/profile_log_device.csv > data/${T}_raw_f.csv
awk -F, 'NR>2 && ($11 ~ /SEND_GO_SIGNAL/) {print $1","$2","$3","$4","$5","$6","$7","$8","$11","$12","$15}' $G/profile_log_device.csv > data/${T}_disp.csv
python3 rawan.py data/${T}_raw_f.csv data/${T}_raw.json > data/${T}_rawtab.txt
python3 goan.py data/${T}_raw.json data/${T}_disp.csv data/${T}_go.json > data/${T}_gotab.txt
wc -l data/${T}_*.csv | tail -4
