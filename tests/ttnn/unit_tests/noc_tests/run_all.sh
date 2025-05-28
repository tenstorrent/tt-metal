set -e
#tt_metal/tools/profiler/profile_this.py --collect-noc-traces -c 'pytest /localdev/sohaibnadeem/tt-metal/tests/ttnn/unit_tests/test_memory_config.py'

test_dir="/localdev/sohaibnadeem/tt-metal/tests/ttnn/unit_tests/noc_tests"
metal=/localdev/sohaibnadeem/tt-metal
npe=/localdev/sohaibnadeem/tt-npe
device=blackhole
test_files="l1_tiled_shard.py" #$(ls -p $test_dir | grep -v "/" | grep py)
reuse_traces=false
#old_headers="Opname,Op ID,NoC Util,DRAM BW Util,Cong Impact,% Overall Cycles"
headers="test name, test config, trace file, max_transfer_dispatch_delay, congestion impact, estimated cycles, golden cycles, cycle pred error, DRAM BW Util (using golden), DRAM BW Util (using estimated), avg Link util, max Link util, avg Link demand, max Link demand, avg NIU demand, max NIU demand, num timesteps, wallclock time"

rm -rf $test_dir/npe_results && mkdir -p $test_dir/npe_results
#mkdir -p $test_dir/trace_results

echo $headers > $test_dir/npe_results/results.csv

for file in $test_files; do
    truncated_filename=$(echo $file | sed  "s/.py//")
    echo "---------------------------------- " $truncated_filename " ----------------------------------"
    if [[ $reuse_traces != "true" ]]; then
        # get traces
        #$metal/tt_metal/tools/profiler/profile_this.py --collect-noc-traces -c "pytest $test_dir/$file" -o "$test_dir/trace_results/$truncated_filename"
        rm -rf "$metal/generated/"
        TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 pytest $test_dir/$file | tee tmp.txt
        cat tmp.txt | grep "test config"| sed "s/test config: //" > $test_dir/npe_results/test_configs.txt
        rm tmp.txt
    fi

    # run npe
    #$npe/install/bin/npe_analyze_noc_trace_dir.py "$metal/generated/profiler/.logs/" -s --device_name blackhole -m 1000 -s > $test_dir/npe_results/${truncated_filename}.txt
    #cat $test_dir/npe_results/${truncated_filename}.txt | grep "ID[0-9]" | sed -E "s/ID[0-9]+/$truncated_filename/" | sed -E "s/ +/,/g"  >> results.txt
    #trace_files=$(ls -p "$metal/generated/profiler/.logs/" | grep -v "/" | grep json)
    num_trace_files=$(ls -p "$metal/generated/profiler/.logs/" | grep -v "/" | grep json | wc -l)
    for ((i = 1 ; i <= $num_trace_files ; i++)); do
        # check for max trasnfer delay
        test_config=$(cat $test_dir/npe_results/test_configs.txt | head -n $i | tail -1)
        id=$(( i * 1024 ))
        trace_file=noc_trace_dev0_ID${id}.json

        max_transfer_dispatch_delay=$(python trace_timeline.py $metal/generated/profiler/.logs/$trace_file | grep "max_transfer_dispatch_delay" | awk "{ print \$2 }")
        #if [[ $max_transfer_dispatch_delay -gt 1000 ]]; then
        #    echo "max_transfer_dispatch_delay is too large! Please modify tests to use smaller tensor sizes."
        #    exit 1
        #fi

        # --cong-model none
        $npe/install/bin/tt_npe.py -w $metal/generated/profiler/.logs/$trace_file -d $device -t > $test_dir/npe_results/${truncated_filename}_${trace_file}.txt

        # add test name + op name
        echo -n "${truncated_filename},${test_config},${trace_file},${max_transfer_dispatch_delay}," >> $test_dir/npe_results/results.csv

        #take output from npe and do some formatting
        cat $test_dir/npe_results/${truncated_filename}_${trace_file}.txt | grep  "\-\-\- stats \-\-\-" -A 20 | tail -n +2 | grep ":" | sed -E "s/.*://" | sed -E "s/\(.*\)//" | sed -E "s/ +//" | tr '\n' ',' >> $test_dir/npe_results/results.csv

        # add newline
        echo '' >> $test_dir/npe_results/results.csv
    done
done
