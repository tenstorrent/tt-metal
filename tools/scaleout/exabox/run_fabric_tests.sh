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
    --config <4x8|4x32|8x16|4x8z|2x4x4z|4x32z|8x4x4z>  Mesh configuration (default: 4x32)
                                        The *z configs are multi-mesh layouts that exercise Z links
                                        (inter-mesh) in addition to the intra-mesh N/S/E/W links.
                                        They launch one MPI rank per mesh, each with its own TT_MESH_ID.
                                        4x8z    = single galaxy as 4 Z-connected 4x2 meshes (4 ranks).
                                        2x4x4z  = single galaxy as 2 Z-connected 4x4 meshes (2 ranks, dual_4x4 layout).
                                        8x4x4z = full quad: 8 Z-connected 4x4 meshes across 4 hosts (12 ranks);
                                                  even mesh ids are single-host (slices {1,2}), odd mesh ids are
                                                  split across two adjacent ring hosts ({3} + next-host {0}).
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
                                        8x4x4z default: tt_metal/fabric/mesh_graph_descriptors/quad_bh_galaxy_8x4x4_z_graph_descriptor.textproto
                                                       (8 Z-connected 4x4 meshes across 4 galaxies; even single-host, odd split 2x1)
    --test-binary <path>                Path to test binary
                                        (default: ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric)
    --test-config <path>                Path to test configuration file
                                        (default: tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml)
                                        (4x8z/2x4x4z/4x32z/8x4x4z default: test_fabric_multi_mesh_sanity_common.yaml, whose
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
MESH_GRAPH_DESC_PATH_8x4x4z="tt_metal/fabric/mesh_graph_descriptors/quad_bh_galaxy_8x4x4_z_graph_descriptor.textproto"
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
            if [[ "$CONFIG" != "4x8" && "$CONFIG" != "4x32" && "$CONFIG" != "8x16" && "$CONFIG" != "4x8z" && "$CONFIG" != "2x4x4z" && "$CONFIG" != "4x32z" && "$CONFIG" != "8x4x4z" ]]; then
                echo "Error: --config must be one of '4x8', '4x32', '8x16', '4x8z', '2x4x4z', '4x32z', or '8x4x4z'"
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
    elif [[ "$CONFIG" == "8x4x4z" ]]; then
        MESH_GRAPH_DESC_PATH="$MESH_GRAPH_DESC_PATH_8x4x4z"
    fi
fi

# Multi-mesh (Z) configs need a multi-mesh-aware test config; fall back to the
# multi-mesh sanity config unless the user explicitly passed --test-config.
if [[ "$TEST_CONFIG_EXPLICIT" == false && ( "$CONFIG" == "4x8z" || "$CONFIG" == "2x4x4z" || "$CONFIG" == "4x32z" || "$CONFIG" == "8x4x4z" ) ]]; then
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

# Resolve TT_VISIBLE_DEVICES for 2x4x4z from the multi-host 2x4 slice discovery
# (generate_rank_bindings.py --slice-config). One mpirun across all hosts
# produces a per-host slice mapping (Rev C tray swap handled in the gtest), so no
# per-host SSH loop is needed. Prints one TT_VISIBLE_DEVICES line per rank in
# host-major order (host0 local mesh 0, host0 local mesh 1, host1 ...).
#
# Discovery ALWAYS runs natively on the launch host, even in docker mode: it only
# reads the hardware topology and emits logical device ids (identical whether run
# natively or inside a container), and the multi-host discovery gtest cannot be
# launched across per-rank containers. In docker mode the resolved
# TT_VISIBLE_DEVICES are forwarded into the containers via -x at launch time.
# Requires a local host build of test_physical_discovery + python3 (loguru/yaml).
resolve_slice_z_visible_devices() {
    local slice_config="$1"
    local hosts_csv="$2"
    local expected_ranks="$3"
    local tt_home="${TT_METAL_HOME:-$(pwd)}"
    local gen_rb="${tt_home}/tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py"

    if [[ ! -f "$gen_rb" ]]; then
        echo "Error: rank binding helper not found: $gen_rb" >&2
        exit 1
    fi

    echo "Resolving TT_VISIBLE_DEVICES via 2x4 slice discovery (--slice-config ${slice_config}, hosts ${hosts_csv})..."
    Z_VISIBLE_DEVICES=()
    mapfile -t Z_VISIBLE_DEVICES < <(
        cd "$tt_home" && \
        LD_LIBRARY_PATH="${tt_home}/build/lib:${LD_LIBRARY_PATH:-}" \
        TT_METAL_HOME="$tt_home" \
        python3 "$gen_rb" --slice-config "$slice_config" --hosts "$hosts_csv" --mpi-if "$MPI_IF" \
            --print-devices --work-dir "$tt_home" \
            | grep '^FABRIC_VISIBLE_DEVICES:' | sed 's/^FABRIC_VISIBLE_DEVICES://'
    )

    if [[ "${#Z_VISIBLE_DEVICES[@]}" -ne "$expected_ranks" ]]; then
        echo "Error: expected ${expected_ranks} TT_VISIBLE_DEVICES entries for ${slice_config}, got ${#Z_VISIBLE_DEVICES[@]}" >&2
        exit 1
    fi
    for ((i = 0; i < expected_ranks; i++)); do
        echo "  rank $i -> TT_VISIBLE_DEVICES=${Z_VISIBLE_DEVICES[$i]}"
    done
}

# Resolve the 8x4x4z quad split layout: 8 Z-connected 4x4 meshes across 4 ring
# hosts as a 12-rank table (even meshes single-host {1,2}; odd meshes split
# {3}+next-host {0}). Populates parallel per-rank arrays in canonical
# (mesh_id, host_rank) -> mpi_rank order:
#   Z_RANK_MESH_ID, Z_RANK_HOST_RANK, Z_RANK_PIN_HOSTS, Z_VISIBLE_DEVICES.
#
# Like resolve_slice_z_visible_devices, the slice discovery always runs natively
# on the launch host (even in docker mode); only the test launch is containerized.
resolve_quad_split_rank_table() {
    local hosts_csv="$1"
    local expected_ranks="$2"
    local tt_home="${TT_METAL_HOME:-$(pwd)}"
    local gen_rb="${tt_home}/tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py"

    if [[ ! -f "$gen_rb" ]]; then
        echo "Error: rank binding helper not found: $gen_rb" >&2
        exit 1
    fi

    echo "Resolving quad split rank table via 2x4 slice discovery (--slice-config 8x4x4z, hosts ${hosts_csv})..."
    local rank_lines=()
    mapfile -t rank_lines < <(
        cd "$tt_home" && \
        LD_LIBRARY_PATH="${tt_home}/build/lib:${LD_LIBRARY_PATH:-}" \
        TT_METAL_HOME="$tt_home" \
        python3 "$gen_rb" --slice-config 8x4x4z --hosts "$hosts_csv" --mpi-if "$MPI_IF" \
            --print-rank-table --work-dir "$tt_home" \
            | grep '^FABRIC_RANK:' | sed 's/^FABRIC_RANK://'
    )

    if [[ "${#rank_lines[@]}" -ne "$expected_ranks" ]]; then
        echo "Error: expected ${expected_ranks} FABRIC_RANK entries for 8x4x4z, got ${#rank_lines[@]}" >&2
        exit 1
    fi

    Z_RANK_MESH_ID=()
    Z_RANK_HOST_RANK=()
    Z_RANK_PIN_HOSTS=()
    Z_VISIBLE_DEVICES=()
    local idx=0
    local mesh_id host_rank host devices
    for line in "${rank_lines[@]}"; do
        IFS=';' read -r mesh_id host_rank host devices <<< "$line"
        Z_RANK_MESH_ID[$idx]="$mesh_id"
        Z_RANK_HOST_RANK[$idx]="$host_rank"
        Z_RANK_PIN_HOSTS[$idx]="$host"
        Z_VISIBLE_DEVICES[$idx]="$devices"
        echo "  rank $idx -> mesh_id ${mesh_id} host_rank ${host_rank} host ${host} TT_VISIBLE_DEVICES=${devices}"
        ((idx++))
    done
}

# Build an OpenMPI rankfile that pins each rank to its host (from
# Z_RANK_PIN_HOSTS), assigning per-host slots in increasing order.
write_quad_split_rankfile() {
    Z_RANKFILE="$(mktemp)"
    declare -A host_slot
    local i h slot
    for ((i = 0; i < NUM_RANKS; i++)); do
        h="${Z_RANK_PIN_HOSTS[$i]}"
        slot="${host_slot[$h]:-0}"
        echo "rank $i=${h} slot=$slot" >> "$Z_RANKFILE"
        host_slot[$h]=$((slot + 1))
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

# ---------------------------------------------------------------------------
# Success highlighting helpers (used live during test output and at end).
# Defined here, before the launch logic, so the pipelines below can call them.
# ---------------------------------------------------------------------------

FABRIC_SUCCESS_MARKER="Test | All tests completed successfully"

# Print an unmissable banner when a rank reports success.
_print_fabric_success_banner() {
    local plain_line="$1"
    local success_count="$2"
    local rank_label=""

    if [[ "$plain_line" =~ \[1,([0-9]+)\] ]]; then
        rank_label=" (MPI rank ${BASH_REMATCH[1]})"
    fi

    echo ""
    echo -e "\033[42m\033[1;30m                                                                                \033[0m"
    echo -e "\033[42m\033[1;30m                                                                                \033[0m"
    echo -e "\033[42m\033[1;30m   >>>>>  FABRIC TESTS PASSED${rank_label}  <<<<<                              \033[0m"
    echo -e "\033[42m\033[1;30m                                                                                \033[0m"
    echo -e "\033[42m\033[1;30m   Detected: ${FABRIC_SUCCESS_MARKER}                                       \033[0m"
    echo -e "\033[42m\033[1;30m   Success signals so far: ${success_count}                                   \033[0m"
    echo -e "\033[42m\033[1;30m                                                                                \033[0m"
    echo -e "\033[42m\033[1;30m                                                                                \033[0m"
    echo ""
}

# Tee filter: pass output through unchanged, but shout when success is detected.
# The success phrase is never split by ANSI codes mid-string, so we substring-match
# the raw line directly instead of spawning a sed per line (keeps the live pipe fast).
highlight_fabric_test_success() {
    local line success_count=0

    while IFS= read -r line; do
        printf '%s\n' "$line"
        if [[ "$line" == *"All tests completed successfully"* ]]; then
            success_count=$((success_count + 1))
            _print_fabric_success_banner "$line" "$success_count"
        fi
    done
}

# After the run, summarize pass/fail from the log (one success line per MPI rank).
print_fabric_final_summary() {
    local log_file="$1"
    local success_count=0
    local expected_hosts=0
    local all_passed=false

    if [[ ! -f "$log_file" ]]; then
        echo ""
        echo -e "\033[1;31mNo log file found; cannot verify fabric test results.\033[0m"
        return 1
    fi

    # Single awk pass over the log: strip ANSI inline, count success markers and
    # distinct MPI rank tags ([1,N] from --tag-output). This replaces a bash
    # while-loop that spawned printf|sed per line (two processes per log line),
    # which was very slow on large multi-rank logs.
    read -r success_count expected_hosts < <(
        awk '
        {
            line = $0
            gsub(/\033\[[0-9;]*m/, "", line)
            if (line ~ /All tests completed successfully/) success++
            if (match(line, /^\[1,[0-9]+\]/)) ranks[substr(line, RSTART, RLENGTH)] = 1
        }
        END { n = 0; for (k in ranks) n++; print success + 0, n + 0 }
        ' "$log_file"
    )

    if [[ "$expected_hosts" -eq 0 ]]; then
        expected_hosts=1
    fi

    if [[ "$success_count" -eq "$expected_hosts" && "$success_count" -gt 0 ]]; then
        all_passed=true
    fi

    echo ""
    if [[ "$all_passed" == true ]]; then
        echo -e "\033[42m\033[1;30m                                                                                \033[0m"
        echo -e "\033[42m\033[1;30m                                                                                \033[0m"
        echo -e "\033[42m\033[1;30m                                                                                \033[0m"
        echo -e "\033[42m\033[1;30m   ##########################  ALL FABRIC TESTS PASSED  ######################## \033[0m"
        echo -e "\033[42m\033[1;30m                                                                                \033[0m"
        echo -e "\033[42m\033[1;30m   Every MPI rank (${success_count}/${expected_hosts}) reported: ${FABRIC_SUCCESS_MARKER} \033[0m"
        echo -e "\033[42m\033[1;30m                                                                                \033[0m"
        echo -e "\033[42m\033[1;30m                                                                                \033[0m"
        echo -e "\033[42m\033[1;30m                                                                                \033[0m"
        echo ""
    else
        echo -e "\033[1;31m================================================================================\033[0m"
        echo -e "\033[1;31m FABRIC TESTS DID NOT FULLY PASS \033[0m"
        echo -e "\033[1;31m Success signals: ${success_count}/${expected_hosts} MPI rank(s)\033[0m"
        echo -e "\033[1;31m Expected '${FABRIC_SUCCESS_MARKER}' once per rank.\033[0m"
        echo -e "\033[1;31m See log: ${log_file}\033[0m"
        echo -e "\033[1;31m================================================================================\033[0m"
        echo ""
    fi
}

if [[ "$CONFIG" == "4x8z" || "$CONFIG" == "2x4x4z" || "$CONFIG" == "4x32z" || "$CONFIG" == "8x4x4z" ]]; then
    # Multi-mesh Z configs: launch one MPI rank per mesh, each with its own
    # TT_MESH_ID, so the descriptor's inter-mesh (Z) connections are exercised
    # alongside the intra-mesh N/S/E/W links during neighbor exchange.
    Z_VISIBLE_DEVICES=()
    Z_RANK_HOSTS=()
    Z_GLOBAL_HOST=()
    Z_RANK_MESH_ID=()
    Z_RANK_HOST_RANK=()
    Z_RANK_PIN_HOSTS=()
    NUM_RANKS=""

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
        resolve_slice_z_visible_devices "2x4x4z" "$SINGLE_HOST" "$NUM_MESHES"
        echo "Running multi-mesh 2x4x4z (2 Z-connected 4x4 meshes, slice-based) on single host: $SINGLE_HOST"
    elif [[ "$CONFIG" == "8x4x4z" ]]; then
        # 8 Z-connected 4x4 meshes across 4 ring hosts, composed from 2x4 slices:
        # even meshes are single-host 4x4 {1,2}; odd meshes are split across two
        # adjacent hosts ({3} on host H + {0} on host H+1). That yields 12 ranks
        # (3 per host) ordered mesh-major then host-rank-minor, matching
        # control_plane's (mesh_id, host_rank) -> mpi_rank assignment.
        NUM_MESHES=8
        NUM_RANKS=12
        IFS=',' read -ra Z_RANK_HOSTS <<< "$HOSTS"
        if [[ "${#Z_RANK_HOSTS[@]}" -ne 4 ]]; then
            echo "Error: --config 8x4x4z requires exactly 4 hosts in --hosts (got ${#Z_RANK_HOSTS[@]})"
            exit 1
        fi
        resolve_quad_split_rank_table "$HOSTS" "$NUM_RANKS"
        write_quad_split_rankfile
        echo "Running multi-mesh 8x4x4z (8 Z-connected 4x4 meshes across 4 hosts, slice-based split; 12 ranks, even=single-host {1,2}, odd=split {3}+next-host {0}); ranks pinned via rankfile."
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

    # Configs other than 8x4x4z launch one rank per mesh (mesh_id == rank, no
    # explicit host rank). Fill the per-rank arrays so the segment builder below
    # is shared by all Z configs.
    if [[ -z "$NUM_RANKS" ]]; then
        NUM_RANKS="$NUM_MESHES"
        for ((i = 0; i < NUM_RANKS; i++)); do
            Z_RANK_MESH_ID[$i]="$i"
            Z_RANK_HOST_RANK[$i]=""
        done
    fi

    # Assemble the per-rank ":"-separated MPMD segments shared by the docker
    # and no-docker launch paths. Each segment carries its own TT_MESH_ID (and,
    # where resolved, TT_VISIBLE_DEVICES). 8x4x4z additionally sets
    # TT_MESH_HOST_RANK (required on multi-host systems, and distinguishes the
    # two ranks of a split mesh). Host placement for 4x32z/8x4x4z is handled by
    # the OpenMPI rankfile in Z_GLOBAL_HOST, not per-segment --host.
    # TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS matches full_rank_binding.yaml so the
    # slowest ethernet handshakes don't trip the Fabric Router Sync timeout.
    Z_SEGMENTS=()
    for ((i = 0; i < NUM_RANKS; i++)); do
        [[ $i -gt 0 ]] && Z_SEGMENTS+=(":")
        Z_SEGMENTS+=(-np 1)
        Z_SEGMENTS+=(-x TT_MESH_ID="${Z_RANK_MESH_ID[$i]}")
        if [[ -n "${Z_RANK_HOST_RANK[$i]:-}" ]]; then
            Z_SEGMENTS+=(-x TT_MESH_HOST_RANK="${Z_RANK_HOST_RANK[$i]}")
        fi
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
    if [[ "$CONFIG" == "4x8z" || "$CONFIG" == "2x4x4z" || "$CONFIG" == "8x4x4z" ]]; then
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
            "${Z_SEGMENTS[@]}" |& tee "$LOG_FILE" | highlight_fabric_test_success
    else
        ./tools/scaleout/exabox/mpi-docker --image "$DOCKER_IMAGE" \
            --empty-entrypoint \
            --mpi-interface "$MPI_IF" \
            "${Z_DOCKER_MPI_ARGS[@]}" \
            "${MPI_EXTRA_ARGS[@]}" \
            --bind-to none \
            "${Z_GLOBAL_HOST[@]}" \
            "${Z_SEGMENTS[@]}" |& tee "$LOG_FILE" | highlight_fabric_test_success
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
            --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE" | highlight_fabric_test_success
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
            --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE" | highlight_fabric_test_success
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
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE" | highlight_fabric_test_success
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
        --test_config "$TEST_CONFIG" $EXTRA_BINARY_ARGS |& tee "$LOG_FILE" | highlight_fabric_test_success
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

print_fabric_final_summary "$LOG_FILE"
echo "=========================================="
