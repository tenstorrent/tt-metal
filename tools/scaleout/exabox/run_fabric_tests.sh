#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 --hosts <comma-separated-host-list> --image <docker-image> [OPTIONS]

Run fabric tests on 4x8, 4x32, or 8x16 cluster configuration.

Required Options:
    --hosts <host-list>                 Comma-separated list of hosts (single host for 4x8)
    --image <docker-image>              Docker image to use ("none" to use local build)

Optional:
    --config <4x8|4x32|8x16|4x8z|2x4x4z|4x32z|16x4x4z>  Mesh configuration (default: 4x32)
                                        The *z configs are multi-mesh layouts that exercise Z links
                                        (inter-mesh) in addition to the intra-mesh N/S/E/W links.
                                        They launch one MPI rank per mesh, each with its own TT_MESH_ID.
                                        4x8z    = single galaxy as 4 Z-connected 4x2 meshes (4 ranks).
                                        2x4x4z  = single galaxy as 2 Z-connected 4x4 meshes (2 ranks, dual_4x4 layout).
                                        16x4x4z = full quad: 4 hosts x 2 Z-connected 4x4 meshes each (8 ranks).
    --output <directory>                Output directory for log files (default: fabric_test_logs)
    --mesh-graph-desc-path <path>       Path to mesh graph descriptor file (overrides --config)
                                        4x8 default:   tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        4x32 default:  tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        8x16 default:  tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
                                        4x8z default:  tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x4x2_z_graph_descriptor.textproto
                                                       (single galaxy split into 4 Z-connected 4x2 meshes)
                                        2x4x4z default: tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_2x4x4_z_graph_descriptor.textproto
                                                       (single galaxy split into 2 Z-connected 4x4 meshes)
                                        4x32z default:  tt_metal/fabric/mesh_graph_descriptors/quad_bh_galaxy_4x4x8_z_torus_graph_descriptor.textproto
                                                       (4 galaxies as 4 Z-connected 8x4 torus meshes)
                                        16x4x4z default: tt_metal/fabric/mesh_graph_descriptors/quad_bh_galaxy_8x4x4_z_graph_descriptor.textproto
                                                       (4 galaxies x 2 Z-connected 4x4 meshes each)
    --test-binary <path>                Path to test binary
                                        (default: ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric)
    --test-config <path>                Path to test configuration file
                                        (default: tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml)
                                        (4x8z/2x4x4z/4x32z/16x4x4z default: test_fabric_multi_mesh_sanity_common.yaml, whose
                                         neighbor_exchange/all_to_all patterns route across mesh boundaries / Z links)
    --filter <pattern>                  Filter pattern passed to test_tt_fabric --filter
    --mpi-if <interface>                Network interface for MPI TCP transport (default: ens5f0np0)
    --mpi-args <args>                   Extra arguments passed directly to mpirun (quoted string)
                                        e.g. --mpi-args "--tag-output"
    --help                              Display this help message and exit

Example:
    $0 --hosts bh-glx-c01u02,bh-glx-c01u08 \\
       --image ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest
EOF
}

# Parse command line arguments
HOSTS=""
DOCKER_IMAGE=""
OUTPUT_DIR="fabric_test_logs"
MESH_GRAPH_DESC_PATH_4x8="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_4x32="tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_8x16="tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_4x8z="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x4x2_z_graph_descriptor.textproto"
# 2x4x4z: single galaxy split into 2 Z-connected 4x4 meshes (dual_4x4 layout).
MESH_GRAPH_DESC_PATH_2x4x4z="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_2x4x4_z_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_4x32z="tt_metal/fabric/mesh_graph_descriptors/quad_bh_galaxy_4x4x8_z_torus_graph_descriptor.textproto"
MESH_GRAPH_DESC_PATH_16x4x4z="tt_metal/fabric/mesh_graph_descriptors/quad_bh_galaxy_8x4x4_z_graph_descriptor.textproto"
CONFIG="4x32"
MESH_GRAPH_DESC_PATH=""
MESH_GRAPH_DESC_PATH_EXPLICIT=false
TEST_BINARY="./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"
TEST_CONFIG="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml"
TEST_CONFIG_EXPLICIT=false
# Multi-mesh (Z) configs default to the multi-mesh sanity config, whose
# neighbor_exchange/all_to_all patterns route across mesh boundaries (Z links).
# The single-mesh test_fabric_sanity_neighbor_exchange.yaml is NOT compatible
# with an inter-mesh Z fabric (its Linear/Ring/Torus setups trip the tensix
# datamover buffer-index assert), so it must not be the default here.
TEST_CONFIG_Z="tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_multi_mesh_sanity_common.yaml"
FILTER=""
MPI_IF="ens5f0np0"
MPI_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --hosts)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --hosts requires a non-empty value"
                exit 1
            fi
            HOSTS="$2"
            shift 2
            ;;
        --image)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --image requires a non-empty value"
                exit 1
            fi
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --config requires a non-empty value"
                exit 1
            fi
            CONFIG="$2"
            if [[ "$CONFIG" != "4x8" && "$CONFIG" != "4x32" && "$CONFIG" != "8x16" && "$CONFIG" != "4x8z" && "$CONFIG" != "2x4x4z" && "$CONFIG" != "4x32z" && "$CONFIG" != "16x4x4z" ]]; then
                echo "Error: --config must be one of '4x8', '4x32', '8x16', '4x8z', '2x4x4z', '4x32z', or '16x4x4z'"
                echo ""
                show_help
                exit 1
            fi
            shift 2
            ;;
        --output)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --output requires a non-empty value"
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mesh-graph-desc-path)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --mesh-graph-desc-path requires a non-empty value"
                exit 1
            fi
            MESH_GRAPH_DESC_PATH="$2"
            MESH_GRAPH_DESC_PATH_EXPLICIT=true
            shift 2
            ;;
        --test-binary)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --test-binary requires a non-empty value"
                exit 1
            fi
            TEST_BINARY="$2"
            shift 2
            ;;
        --test-config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --test-config requires a non-empty value"
                exit 1
            fi
            TEST_CONFIG="$2"
            TEST_CONFIG_EXPLICIT=true
            shift 2
            ;;
        --filter)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --filter requires a non-empty value"
                exit 1
            fi
            FILTER="$2"
            shift 2
            ;;
        --mpi-if)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --mpi-if requires a non-empty value"
                exit 1
            fi
            MPI_IF="$2"
            shift 2
            ;;
        --mpi-args)
            if [[ -z "$2" ]]; then
                echo "Error: --mpi-args requires a non-empty value"
                exit 1
            fi
            read -ra _extra <<< "$2"
            MPI_EXTRA_ARGS+=("${_extra[@]}")
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$HOSTS" ]]; then
    echo "Error: --hosts is required"
    echo ""
    show_help
    exit 1
fi

if [[ -z "$DOCKER_IMAGE" ]]; then
    echo "Error: --image is required"
    echo ""
    show_help
    exit 1
fi

# Set mesh graph descriptor path based on config if not explicitly provided
if [[ "$MESH_GRAPH_DESC_PATH_EXPLICIT" == false ]]; then
    if [[ "$CONFIG" == "4x8" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x8"
    elif [[ "$CONFIG" == "4x32" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x32"
    elif [[ "$CONFIG" == "8x16" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_8x16"
    elif [[ "$CONFIG" == "4x8z" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x8z"
    elif [[ "$CONFIG" == "2x4x4z" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_2x4x4z"
    elif [[ "$CONFIG" == "4x32z" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_4x32z"
    elif [[ "$CONFIG" == "16x4x4z" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_16x4x4z"
    fi
fi

# Multi-mesh (Z) configs need a multi-mesh-aware test config; fall back to the
# multi-mesh sanity config unless the user explicitly passed --test-config.
if [[ "$TEST_CONFIG_EXPLICIT" == false && ( "$CONFIG" == "4x8z" || "$CONFIG" == "2x4x4z" || "$CONFIG" == "4x32z" || "$CONFIG" == "16x4x4z" ) ]]; then
    TEST_CONFIG="$TEST_CONFIG_Z"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/fabric_tests_${RUN_TIMESTAMP}.log"
Z_RANKFILE=""   # 4x32z OpenMPI rankfile; set below when CONFIG=4x32z

echo "=========================================="
echo "Running fabric tests..."
echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
echo "Configuration: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Mesh graph descriptor: $MESH_GRAPH_DESC_PATH"
echo "Test binary: $TEST_BINARY"
echo "Test config: $TEST_CONFIG"
if [[ -n "$FILTER" ]]; then
    echo "Filter: $FILTER"
fi
echo "MPI interface: $MPI_IF"
if [[ "${#MPI_EXTRA_ARGS[@]}" -gt 0 ]]; then
    echo "MPI extra args: ${MPI_EXTRA_ARGS[*]}"
fi
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

EXTRA_BINARY_ARGS=""
if [[ "$TEST_BINARY" == *test_tt_fabric ]]; then
    EXTRA_BINARY_ARGS="--show-progress-detail --show-workers --progress-interval 1"
fi
if [[ -n "$FILTER" ]]; then
    EXTRA_BINARY_ARGS="$EXTRA_BINARY_ARGS --filter $FILTER"
fi

# Resolve TT_VISIBLE_DEVICES for single-host multi-mesh Z configs from tray discovery
# (tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py --fabric-config).
#
# No-docker runs use TT_METAL_HOME (or pwd) on the host, like bootstrap_pipeline_dir.sh.
# Docker runs use the image's TT_METAL_HOME and python3, like run_dispatch_tests.sh /
# run_validation.sh (upstream test images set both in dockerfile/upstream_test_images/).
run_fabric_discovery_local() {
    local fabric_config="$1"

    if [[ "$DOCKER_IMAGE" == "none" ]]; then
        local tt_home="${TT_METAL_HOME:-$(pwd)}"
        local gen_rb="${tt_home}/tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py"
        cd "$tt_home" && \
        LD_LIBRARY_PATH="${tt_home}/build/lib:${LD_LIBRARY_PATH:-}" \
        TT_METAL_HOME="$tt_home" \
        python3 "$gen_rb" --fabric-config "$fabric_config" --print-devices --work-dir "$tt_home"
    else
        docker run --rm --net=host --privileged \
            -v /tmp:/tmp \
            -v /dev/hugepages-1G:/dev/hugepages-1G \
            -v "$HOME:$HOME" \
            --user "$(id -u):$(id -g)" \
            -v /etc/passwd:/etc/passwd:ro \
            -v /etc/group:/etc/group:ro \
            --entrypoint="" \
            "$DOCKER_IMAGE" \
            bash -c 'cd "$TT_METAL_HOME" && python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py --fabric-config '"$fabric_config"' --print-devices --work-dir "$TT_METAL_HOME"'
    fi
}

is_local_fabric_host() {
    local host="$1"
    local short_name fqdn host_short
    short_name="$(hostname -s 2>/dev/null || hostname)"
    fqdn="$(hostname -f 2>/dev/null || true)"
    host_short="${host%%.*}"
    [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$short_name" || "$host" == "$fqdn" || "$host_short" == "$short_name" ]]
}

run_fabric_discovery_on_host() {
    local host="$1"
    local fabric_config="$2"
    local tt_home="${TT_METAL_HOME:-$(pwd)}"
    local ssh_opts=(-o StrictHostKeyChecking=false -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

    if is_local_fabric_host "$host"; then
        run_fabric_discovery_local "$fabric_config"
        return
    fi

    if [[ "$DOCKER_IMAGE" == "none" ]]; then
        local gen_rb="${tt_home}/tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py"
        ssh "${ssh_opts[@]}" "$host" \
            "cd '${tt_home}' && LD_LIBRARY_PATH='${tt_home}/build/lib:'\${LD_LIBRARY_PATH:-} TT_METAL_HOME='${tt_home}' python3 '${gen_rb}' --fabric-config '${fabric_config}' --print-devices --work-dir '${tt_home}'"
    else
        ssh "${ssh_opts[@]}" "$host" \
            "docker run --rm --net=host --privileged \
                -v /tmp:/tmp \
                -v /dev/hugepages-1G:/dev/hugepages-1G \
                -v '${HOME}:${HOME}' \
                --user '$(id -u):$(id -g)' \
                -v /etc/passwd:/etc/passwd:ro \
                -v /etc/group:/etc/group:ro \
                --entrypoint='' \
                '${DOCKER_IMAGE}' \
                bash -c 'cd \"\$TT_METAL_HOME\" && python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py --fabric-config ${fabric_config} --print-devices --work-dir \"\$TT_METAL_HOME\"'"
    fi
}

resolve_single_host_z_visible_devices() {
    local fabric_config="$1"
    local expected_ranks="$2"

    if [[ "$DOCKER_IMAGE" == "none" ]]; then
        local gen_rb="${TT_METAL_HOME:-$(pwd)}/tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py"
        if [[ ! -f "$gen_rb" ]]; then
            echo "Error: rank binding helper not found: $gen_rb" >&2
            exit 1
        fi
    fi

    echo "Resolving TT_VISIBLE_DEVICES via tray discovery (--fabric-config ${fabric_config})..."
    Z_VISIBLE_DEVICES=()

    mapfile -t Z_VISIBLE_DEVICES < <(
        run_fabric_discovery_local "$fabric_config" \
            | grep '^FABRIC_VISIBLE_DEVICES:' | sed 's/^FABRIC_VISIBLE_DEVICES://'
    )

    if [[ "${#Z_VISIBLE_DEVICES[@]}" -ne "$expected_ranks" ]]; then
        echo "Error: expected ${expected_ranks} TT_VISIBLE_DEVICES entries for ${fabric_config}, got ${#Z_VISIBLE_DEVICES[@]}" >&2
        exit 1
    fi
    for ((i = 0; i < expected_ranks; i++)); do
        echo "  rank $i -> TT_VISIBLE_DEVICES=${Z_VISIBLE_DEVICES[$i]}"
    done
}

resolve_quad_host_z_visible_devices() {
    local per_host_fabric_config="$1"
    local num_hosts="$2"
    local ranks_per_host="$3"
    local expected_total=$((num_hosts * ranks_per_host))

    if [[ "$DOCKER_IMAGE" == "none" ]]; then
        local gen_rb="${TT_METAL_HOME:-$(pwd)}/tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py"
        if [[ ! -f "$gen_rb" ]]; then
            echo "Error: rank binding helper not found: $gen_rb" >&2
            exit 1
        fi
    fi

    IFS=',' read -ra Z_RANK_HOSTS <<< "$HOSTS"
    if [[ "${#Z_RANK_HOSTS[@]}" -ne "$num_hosts" ]]; then
        echo "Error: --config requires exactly $num_hosts hosts in --hosts (got ${#Z_RANK_HOSTS[@]})" >&2
        exit 1
    fi

    echo "Resolving TT_VISIBLE_DEVICES via tray discovery on ${num_hosts} hosts (--fabric-config ${per_host_fabric_config})..."
    Z_VISIBLE_DEVICES=()

    local host_idx=0
    for host in "${Z_RANK_HOSTS[@]}"; do
        local host_devices=()
        mapfile -t host_devices < <(
            run_fabric_discovery_on_host "$host" "$per_host_fabric_config" \
                | grep '^FABRIC_VISIBLE_DEVICES:' | sed 's/^FABRIC_VISIBLE_DEVICES://'
        )
        if [[ "${#host_devices[@]}" -ne "$ranks_per_host" ]]; then
            echo "Error: expected ${ranks_per_host} TT_VISIBLE_DEVICES entries from ${host}, got ${#host_devices[@]}" >&2
            exit 1
        fi
        for ((local_rank = 0; local_rank < ranks_per_host; local_rank++)); do
            local global_rank=$((host_idx * ranks_per_host + local_rank))
            Z_VISIBLE_DEVICES+=("${host_devices[$local_rank]}")
            echo "  rank $global_rank (host ${host}, local mesh ${local_rank}) -> TT_VISIBLE_DEVICES=${host_devices[$local_rank]}"
        done
        ((host_idx++))
    done

    if [[ "${#Z_VISIBLE_DEVICES[@]}" -ne "$expected_total" ]]; then
        echo "Error: expected ${expected_total} TT_VISIBLE_DEVICES entries, got ${#Z_VISIBLE_DEVICES[@]}" >&2
        exit 1
    fi
}

write_quad_z_rankfile() {
    local ranks_per_host="$1"
    Z_RANKFILE="$(mktemp)"
    local global_rank=0
    for ((h = 0; h < ${#Z_RANK_HOSTS[@]}; h++)); do
        for ((slot = 0; slot < ranks_per_host; slot++)); do
            echo "rank $global_rank=${Z_RANK_HOSTS[$h]} slot=$slot" >> "$Z_RANKFILE"
            ((global_rank++))
        done
    done
    Z_GLOBAL_HOST=(--hostfile "$Z_RANKFILE" --map-by "rankfile:file=$Z_RANKFILE")
}

# Marker used to detect reports written during this run (vs. stale ones from a
# previous run). We compare report mtimes against this file with bash's `-nt`.
RUN_START_MARKER="$(mktemp)"
cleanup_run_artifacts() {
    rm -f "$RUN_START_MARKER"
    [[ -n "$Z_RANKFILE" ]] && rm -f "$Z_RANKFILE"
}
trap cleanup_run_artifacts EXIT

if [[ "$CONFIG" == "4x8z" || "$CONFIG" == "2x4x4z" || "$CONFIG" == "4x32z" || "$CONFIG" == "16x4x4z" ]]; then
    # Multi-mesh Z configs: launch one MPI rank per mesh, each with its own
    # TT_MESH_ID, so the descriptor's inter-mesh (Z) connections are exercised
    # alongside the intra-mesh N/S/E/W links during neighbor exchange.
    Z_VISIBLE_DEVICES=()
    Z_RANK_HOSTS=()
    Z_GLOBAL_HOST=()
    Z_RANKS_PER_HOST=1

    if [[ "$CONFIG" == "4x8z" ]]; then
        NUM_MESHES=4
        SINGLE_HOST="${HOSTS%%,*}"
        Z_GLOBAL_HOST=(--host "${SINGLE_HOST}:${NUM_MESHES}")
        resolve_single_host_z_visible_devices "4x8z" "$NUM_MESHES"
        echo "Running multi-mesh 4x8z (4 Z-connected 4x2 meshes) on single host: $SINGLE_HOST"
    elif [[ "$CONFIG" == "2x4x4z" ]]; then
        NUM_MESHES=2
        SINGLE_HOST="${HOSTS%%,*}"
        Z_GLOBAL_HOST=(--host "${SINGLE_HOST}:${NUM_MESHES}")
        resolve_single_host_z_visible_devices "2x4x4z" "$NUM_MESHES"
        echo "Running multi-mesh 2x4x4z (2 Z-connected 4x4 meshes) on single host: $SINGLE_HOST"
    elif [[ "$CONFIG" == "16x4x4z" ]]; then
        NUM_MESHES=8
        Z_RANKS_PER_HOST=2
        IFS=',' read -ra Z_RANK_HOSTS <<< "$HOSTS"
        if [[ "${#Z_RANK_HOSTS[@]}" -ne 4 ]]; then
            echo "Error: --config 16x4x4z requires exactly 4 hosts in --hosts (got ${#Z_RANK_HOSTS[@]})"
            exit 1
        fi
        resolve_quad_host_z_visible_devices "2x4x4z" 4 "$Z_RANKS_PER_HOST"
        write_quad_z_rankfile "$Z_RANKS_PER_HOST"
        echo "Running multi-mesh 16x4x4z (4 hosts x 2 Z-connected 4x4 meshes, 8 ranks total); rank i pinned via rankfile:"
        for ((i = 0; i < NUM_MESHES; i++)); do
            local_host_idx=$((i / Z_RANKS_PER_HOST))
            local_mesh=$((i % Z_RANKS_PER_HOST))
            echo "  rank $i -> mesh_id $i -> ${Z_RANK_HOSTS[$local_host_idx]} (local mesh ${local_mesh})"
        done
    else
        NUM_MESHES=4
        # 4x32z: one full galaxy per host. The Z ring (mesh 0->1->2->3->0) only
        # exists between specific physically-adjacent galaxies, so rank i (==
        # mesh i) MUST land on a deterministic host. We therefore pin rank i to
        # the i-th --hosts entry (do NOT rely on MPI round-robin). Provide hosts
        # in the same order as scaleout_configs/full_rankfile.
        IFS=',' read -ra Z_RANK_HOSTS <<< "$HOSTS"
        if [[ "${#Z_RANK_HOSTS[@]}" -ne "$NUM_MESHES" ]]; then
            echo "Error: --config 4x32z requires exactly $NUM_MESHES hosts in --hosts (got ${#Z_RANK_HOSTS[@]})"
            exit 1
        fi
        Z_RANKFILE="$(mktemp)"
        for ((i = 0; i < NUM_MESHES; i++)); do
            echo "rank $i=${Z_RANK_HOSTS[$i]} slot=0" >> "$Z_RANKFILE"
        done
        Z_GLOBAL_HOST=(--hostfile "$Z_RANKFILE" --map-by "rankfile:file=$Z_RANKFILE")
        echo "Running multi-mesh 4x32z (4 Z-connected 8x4 torus galaxies); rank i pinned to host i:"
        for ((i = 0; i < NUM_MESHES; i++)); do
            echo "  rank $i -> mesh_id $i -> ${Z_RANK_HOSTS[$i]}"
        done
    fi
    echo ""

    # Assemble the per-rank ":"-separated MPMD segments shared by the docker
    # and no-docker launch paths. Each segment carries its own TT_MESH_ID (and,
    # for 4x8z/2x4x4z, TT_VISIBLE_DEVICES). Host placement for 4x32z is handled
    # by the OpenMPI rankfile in Z_GLOBAL_HOST, not per-segment --host.
    # TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS matches full_rank_binding.yaml so the
    # slowest ethernet handshakes don't trip the Fabric Router Sync timeout.
    Z_SEGMENTS=()
    for ((i = 0; i < NUM_MESHES; i++)); do
        [[ $i -gt 0 ]] && Z_SEGMENTS+=(":")
        Z_SEGMENTS+=(-np 1)
        Z_SEGMENTS+=(-x TT_MESH_ID="$i")
        if [[ "${#Z_VISIBLE_DEVICES[@]}" -gt 0 ]]; then
            Z_SEGMENTS+=(-x TT_VISIBLE_DEVICES="${Z_VISIBLE_DEVICES[$i]}")
        fi
        Z_SEGMENTS+=(-x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH")
        Z_SEGMENTS+=(-x TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=1000000)
        Z_SEGMENTS+=("$TEST_BINARY" --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS)
    done

    # Single-host and multi-rank-per-host Z configs via mpi-docker: one container
    # per MPMD segment. OpenMPI's default sm BTL needs a shared IPC namespace
    # across containers on the same host; force TCP instead (same as tt-run).
    Z_DOCKER_MPI_ARGS=()
    if [[ "$CONFIG" == "4x8z" || "$CONFIG" == "2x4x4z" || "$CONFIG" == "16x4x4z" ]]; then
        Z_DOCKER_MPI_ARGS=(--mca btl self,tcp)
    fi

    if [[ "$DOCKER_IMAGE" == "none" ]]; then
        mpirun-ulfm \
            --tag-output \
            --mca plm_ssh_args "-o StrictHostKeyChecking=false -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            --mca btl_tcp_if_include "$MPI_IF" \
            "${MPI_EXTRA_ARGS[@]}" \
            --bind-to none \
            "${Z_GLOBAL_HOST[@]}" \
            "${Z_SEGMENTS[@]}" |& tee "$LOG_FILE"
    else
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            --mpi-interface "$MPI_IF" \
            "${Z_DOCKER_MPI_ARGS[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            --bind-to none \
            "${Z_GLOBAL_HOST[@]}" \
            "${Z_SEGMENTS[@]}" |& tee "$LOG_FILE"
    fi
elif [[ "$DOCKER_IMAGE" == "none" ]]; then
    # No-docker path: invoke mpirun-ulfm directly against the local build.
    if [[ "$CONFIG" == "4x8" ]]; then
        SINGLE_HOST="${HOSTS%%,*}"
        echo "Running single-host 4x8 on: $SINGLE_HOST (no docker)"
        echo ""

        mpirun-ulfm \
            --tag-output \
            --mca plm_ssh_args "-o StrictHostKeyChecking=false -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            --mca btl_tcp_if_include "$MPI_IF" \
            "${MPI_EXTRA_ARGS[@]}" \
            --bind-to none \
            --host "$SINGLE_HOST" \
            -np 1 \
            -x TT_MESH_ID=0 \
            -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
            "$TEST_BINARY" \
            --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
    else
        mpirun-ulfm \
            --tag-output \
            --mca plm_ssh_args "-o StrictHostKeyChecking=false -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            --mca btl_tcp_if_include "$MPI_IF" \
            "${MPI_EXTRA_ARGS[@]}" \
            --bind-to none \
            --host "$HOSTS" \
            -np 1 \
            -x TT_MESH_ID=0 \
            -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
            "$TEST_BINARY" \
            --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
            -np 1 \
            -x TT_MESH_ID=0 \
            -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
            "$TEST_BINARY" \
            --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
            -np 1 \
            -x TT_MESH_ID=0 \
            -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
            "$TEST_BINARY" \
            --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
            -np 1 \
            -x TT_MESH_ID=0 \
            -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
            "$TEST_BINARY" \
            --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
    fi
elif [[ "$CONFIG" == "4x8" ]]; then
    SINGLE_HOST="${HOSTS%%,*}"
    echo "Running single-host 4x8 on: $SINGLE_HOST"
    echo ""

    ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
        --empty-entrypoint \
        --mpi-interface "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        --bind-to none \
        --host "$SINGLE_HOST" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
else
    ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
        --empty-entrypoint \
        --mpi-interface "$MPI_IF" \
        "${MPI_EXTRA_ARGS[@]}" \
        --bind-to none \
        --host "$HOSTS" \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS : \
        -np 1 \
        -x TT_MESH_ID=0 \
        -x TT_MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH" \
        "$TEST_BINARY" \
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE"
fi

echo ""
echo "=========================================="
echo "Tests completed at $(date)"
echo "Results logged to: $LOG_FILE"

# Copy any pairwise-validation reports written by test_tt_fabric (only rank 0
# writes them, and only when a hang is detected) into the user's --output dir
# so all artifacts for this run live in one place. Only copy reports that were
# written during this run (newer than $RUN_START_MARKER) so we don't pick up
# stale files from a previous invocation.
REPORT_SRC_DIR="${TT_METAL_HOME:-.}/generated/fabric"
for report in pairwise_validation_summary.log pairwise_validation_detailed.log; do
    if [[ -f "$REPORT_SRC_DIR/$report" && "$REPORT_SRC_DIR/$report" -nt "$RUN_START_MARKER" ]]; then
        cp "$REPORT_SRC_DIR/$report" "$OUTPUT_DIR/"
        echo "Copied report: $OUTPUT_DIR/$report"
    fi
done
echo "=========================================="
