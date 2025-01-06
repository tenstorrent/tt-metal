#/bin/bash

if [ "$ARCH_NAME" = "wormhole_b0" ]; then
    echo "Configured core range for wormhole_b0"
    width=8
    height=8
elif [ "$ARCH_NAME" = "blackhole" ]; then
    echo "Configured core range for blackhole"
    width=12
    height=10
else
    echo "Unknown arch: $ARCH_NAME"
    exit 1
fi

function run_set() {
    echo "running: $@"
    TT_METAL_SLOW_DISPATCH_MODE=1 build/test/tt_metal/test_stress_noc_mcast -t 120 $@
}

function run_all() {
    run_set $@ -u 32 -m 32
    run_set $@ -u 32 -m 256
    run_set $@ -u 32 -m 2048
    run_set $@ -u 32 -m 4096
    run_set $@ -u 32 -m 8192
    run_set $@ -u 256 -m 32
    run_set $@ -u 2048 -m 256
    run_set $@ -u 4096 -m 2048
    run_set $@ -u 8192 -m 4096
}

# sweep w/ randomized noc address, tensix mcast
for (( i=0; i<$width; i++ )); do
    h=$((height -1))
    run_all -x 0 -y 0 -width $width -height $h -mx $i -my $h
done

# sweep w/ randomized delay+noc address, tensix mcast
for (( i=0; i<=11; i++ )); do
    h=$((height -1))
    run_all -x 0 -y 0 -width $width -height $h -mx $i -my $h -rdelay
done

# sweep w/ randomized noc address, eth mcast
for (( i=0; i<=11; i++ )); do
    run_all -e $i -width $width -height $height
done

# sweep w/ randomized delay+noc address, eth mcast
for (( i=0; i<=11; i++ )); do
    run_all -e $i -width $width -height $height -rdelay
done
