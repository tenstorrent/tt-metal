#/bin/bash

if [ "$ARCH_NAME" = "grayskull" ]; then
    echo "Configured core range for grayskull"
    $max_x = 11
    $max_y = 8
elif [ "$ARCH_NAME" = "wormhole_b0" ]; then
    echo "Configured core range for wormhole_b0"
    $max_x = 7
    $max_y = 6
else
    echo "Unknown arch: $ARCH_NAME"
    exit
fi

# brisc only
echo "###" brisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -n -t
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -n -t
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -n -t


# ncrisc only
echo "###" ncrisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -b -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -b -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -b -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -b -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -b -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -b -t
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -b -t
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -b -t
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -b -t

#trisc only
echo "###" trisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -b -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -b -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -b -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -b -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -b -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -b -n
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -b -n
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -b -n
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -b -n

#brisc+trisc only
echo "###" brisc+trisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -n
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -n
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -n

#all processors
echo "###" all procesors
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336

#all processors, all cores
echo "###" all procesors all cores
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x 11 -y 8
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x 11 -y 8
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x 11 -y 8
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x 11 -y 8
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x 11 -y 8
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x 11 -y 8
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x 11 -y 8
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x 11 -y 8
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x 11 -y 8

#all processors, all cores, 1 CB
echo "###" all procesors all cores 1cb
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x 11 -y 8 -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x 11 -y 8 -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x 11 -y 8 -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x 11 -y 8 -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x 11 -y 8 -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x 11 -y 8 -c 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x 11 -y 8 -c 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x 11 -y 8 -c 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x 11 -y 8 -c 1

#all processors, all cores, 32 CB
echo "###" all procesors all cores 32cb
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x 11 -y 8 -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x 11 -y 8 -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x 11 -y 8 -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x 11 -y 8 -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x 11 -y 8 -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x 11 -y 8 -c 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x 11 -y 8 -c 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x 11 -y 8 -c 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x 11 -y 8 -c 32

#all processors, 1 core, 1 rt arg
echo "###" all procesors 1 core 1 rt
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -a 1

#all processors, alls core, 1 rt arg
echo "###" all procesors all cores 1 rt
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x 11 -y 8 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x 11 -y 8 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x 11 -y 8 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x 11 -y 8 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x 11 -y 8 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x 11 -y 8 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x 11 -y 8 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x 11 -y 8 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x 11 -y 8 -a 1

#all processors, all cores, 32 args
echo "###" all procesors all cores 32 rt
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x 11 -y 8 -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x 11 -y 8 -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x 11 -y 8 -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x 11 -y 8 -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x 11 -y 8 -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x 11 -y 8 -a 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x 11 -y 8 -a 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x 11 -y 8 -a 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x 11 -y 8 -a 32

#all processors, all cores, 128 args
echo "###" all procesors all cores 128 rt
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x 11 -y 8 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x 11 -y 8 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x 11 -y 8 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x 11 -y 8 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x 11 -y 8 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x 11 -y 8 -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x 11 -y 8 -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x 11 -y 8 -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x 11 -y 8 -a 128

# sems
echo "###" sems 1 core 1 processor
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -S 4

echo "###" sems all cores 1 processors
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -x 11 -y 8 -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -x 11 -y 8 -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -x 11 -y 8 -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -x 11 -y 8 -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -x 11 -y 8 -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -x 11 -y 8 -S 4

# Worst case
echo "###" worst case
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x 11 -y 8 -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x 11 -y 8 -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x 11 -y 8 -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x 11 -y 8 -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x 11 -y 8 -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x 11 -y 8 -S 4 -c 32 -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x 11 -y 8 -S 4 -c 32 -a 128
