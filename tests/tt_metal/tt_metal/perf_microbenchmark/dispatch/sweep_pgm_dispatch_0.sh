#/bin/bash

if [ "$ARCH_NAME" = "grayskull" ]; then
    echo "Configured core range for grayskull"
    max_x="11"
    max_y="8"
elif [ "$ARCH_NAME" = "wormhole_b0" ]; then
    echo "Configured core range for wormhole_b0"
    max_x="7"
    max_y="6"
elif [ "$ARCH_NAME" = "blackhole" ]; then
    echo "Configured core range for blackhole"
    max_x="12"
    max_y="9"
else
    echo "Unknown arch: $ARCH_NAME"
    exit 1
fi

# Initialize the string variable
trace_option=""
eth_dispatch_option=""

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --trace)
            trace_option="-tr"
            shift
            ;;
        --eth)
            eth_dispatch_option="-de"
            shift
            ;;
        *)
            # Handle other arguments if necessary
            ;;
    esac
done

set -x

# skips ncrisc to reduce uncovered kernel init time on WH
function shadow_test() {
 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 256 -x $max_x -y $max_y -rs 40000 $trace_option $eth_dispatch_option $@
 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 2048 -x $max_x -y $max_y -rs 40000 $trace_option $eth_dispatch_option $@
 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 8192 -x $max_x -y $max_y -rs 40000 $trace_option $eth_dispatch_option $@

 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 256 -x $max_x -y $max_y -rs 40000 -a 1 $trace_option $eth_dispatch_option $@
 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 2048 -x $max_x -y $max_y -rs 40000 -a 1 $trace_option $eth_dispatch_option $@
 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 8192 -x $max_x -y $max_y -rs 40000 -a 1 $trace_option $eth_dispatch_option $@

 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 256 -x $max_x -y $max_y -kg $max_x -rs 40000 -a 1 $trace_option $eth_dispatch_option $@
 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 2048 -x $max_x -y $max_y -kg $max_x -rs 40000 -a 1 $trace_option $eth_dispatch_option $@
 build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --custom -w 5000 -n -s 8192 -x $max_x -y $max_y -kg $max_x -rs 40000 -a 1 $trace_option $eth_dispatch_option $@
}

# Test w/ n shadow kernels
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 0
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 1
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 2
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 3
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 4
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 5
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 6
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 7
echo "###" kernel groups w/ 4 shadow kernels
  shadow_test -nf 8
echo "###" done
