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
    exit
fi

# brisc only
echo "###" brisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t
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
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -b -t
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
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -b -n
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
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n
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
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336

#all processors, all cores
echo "###" all procesors all cores
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x $max_x -y $max_y
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x $max_x -y $max_y
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y

#all processors, all cores, 1 CB
echo "###" all procesors all cores 1cb
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -c 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -c 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x $max_x -y $max_y -c 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x $max_x -y $max_y -c 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y -c 1

#all processors, all cores, 32 CB
echo "###" all procesors all cores 32cb
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -c 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -c 32
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -c 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x $max_x -y $max_y -c 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x $max_x -y $max_y -c 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y -c 32

#all processors, 1 core, 1 rt arg
echo "###" all procesors 1 core 1 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -a 1
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -a 1

#1 processors, alls core, 128 rt arg
echo "###" all procesors all cores 128 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -x $max_x -y $max_y -a 128
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -x $max_x -y $max_y -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -n -t -x $max_x -y $max_y -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -n -t -x $max_x -y $max_y -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -n -t -x $max_x -y $max_y -a 128

#1 processors, alls core, 1 rt arg
echo "###" all procesors all cores 1 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -x $max_x -y $max_y -a 1
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -x $max_x -y $max_y -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -n -t -x $max_x -y $max_y -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -n -t -x $max_x -y $max_y -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -n -t -x $max_x -y $max_y -a 1

#all processors, alls core, 1 rt arg
echo "###" all procesors all cores 1 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -a 1
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -a 1
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x $max_x -y $max_y -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x $max_x -y $max_y -a 1
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y -a 1

#all processors, all cores, 32 args
echo "###" all procesors all cores 32 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -a 32
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -a 32
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -a 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x $max_x -y $max_y -a 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x $max_x -y $max_y -a 32
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y -a 32

#all processors, all cores, 128 args
echo "###" all procesors all cores 128 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -a 128
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x $max_x -y $max_y -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 12880 -x $max_x -y $max_y -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y -a 128

# sems
echo "###" sems 1 core 1 processor
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -S 4
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -S 4

echo "###" sems all cores 1 processors
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -x $max_x -y $max_y -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -x $max_x -y $max_y -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -x $max_x -y $max_y -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -x $max_x -y $max_y -S 4
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -x $max_x -y $max_y -S 4
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -x $max_x -y $max_y -S 4

# Worst case
echo "###" worst case
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -S 4 -c 32 -a 128
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -S 4 -c 32 -a 128
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -S 4 -c 32 -a 128
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y -S 4 -c 32 -a 128

# Kernel groups (perhaps even worse)
echo "###" worst case
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -kg $max_x
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -kg $max_x
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -kg $max_x
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -kg $max_x
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -kg $max_x
# build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -kg $max_x
#build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 14336 -x $max_x -y $max_y -kg $max_x
