#/bin/bash

if [ "$ARCH_NAME" = "grayskull" ]; then
    echo "Configured core range for grayskull"
    max_x="11"
    max_y="8"
elif [ "$ARCH_NAME" = "wormhole_b0" ]; then
    echo "Configured core range for wormhole_b0"
    max_x="7"
    max_y="6"
else
    echo "Unknown arch: $ARCH_NAME"
    exit
fi


# Buffering
echo "###" buffering minor
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 0
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 2
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 6
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 8

# Buffering
echo "###" buffering major
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 0
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 2
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 6
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 8

# Buffering large kernels
echo "###" buffering minor large
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 20000 -nf 0
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 20000 -nf 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 20000 -nf 2
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 20000 -nf 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 20000 -nf 6
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 20000 -nf 8

# Buffering large kernels
echo "###" buffering major large
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 100000 -nf 0
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 100000 -nf 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 100000 -nf 2
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 100000 -nf 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 100000 -nf 6
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -rs 100000 -nf 8

# Buffering RT args
echo "###" buffering minor rtargs
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 0 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 1 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 2 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 4 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 6 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 20000 -nf 8 -x $max_x -y $max_y -a 128

# Buffering large kernels
echo "###" buffering major rtargs
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 0 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 1 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 2 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 4 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 6 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -rs 100000 -nf 8 -x $max_x -y $max_y -a 128
