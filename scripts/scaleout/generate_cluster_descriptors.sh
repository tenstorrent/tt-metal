#!/bin/bash
# Script to generate cluster descriptors from multiple hosts using MPI
# Usage: ./generate_cluster_descriptors.sh --hostnames <hostfile_or_list> --mapping-file <mapping.yaml> --output-dir <output_dir> [--base-name <base_name>]

set -euo pipefail

HOSTNAMES=""
MAPPING_FILE=""
OUTPUT_DIR=""
BASE_NAME="cluster_desc"
TOPOLOGY_TOOL=""
MPI_LAUNCHER="mpirun"
DRY_RUN=false

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate cluster descriptors from multiple hosts using MPI and the UMD topology tool.

OPTIONS:
    --hostnames <file_or_list>    Hostfile path (with @ prefix) or comma-separated host list
    --mapping-file <file>         Output mapping YAML file path (with @ prefix)
    --output-dir <dir>            Directory to save cluster descriptor files (with @ prefix)
    --base-name <name>            Base name for cluster descriptor files (default: cluster_desc)
    --topology-tool <path>        Path to topology tool (default: auto-detect)
    --mpi-launcher <cmd>          MPI launcher command (default: mpirun)
    --dry-run                     Show what would be executed without running
    -h, --help                    Show this help message

EXAMPLES:
    ./generate_cluster_descriptors.sh \\
      --hostnames "host1,host2,host3" \\
      --mapping-file mapping.yaml \\
      --output-dir ./cluster_descs \\
      --base-name my_cluster_desc
EOF
    exit 1
}

# Remove @ prefix if present
remove_at_prefix() {
    local path="$1"
    if [[ "$path" =~ ^@ ]]; then
        echo "${path#@}"
    else
        echo "$path"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hostnames)
            HOSTNAMES="$2"
            shift 2
            ;;
        --mapping-file)
            MAPPING_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --base-name)
            BASE_NAME="$2"
            shift 2
            ;;
        --topology-tool)
            TOPOLOGY_TOOL="$2"
            shift 2
            ;;
        --mpi-launcher)
            MPI_LAUNCHER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$HOSTNAMES" ]] || [[ -z "$MAPPING_FILE" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo -e "${RED}Error: --hostnames, --mapping-file, and --output-dir are required${NC}" >&2
    usage
fi

# Remove @ prefixes
HOSTNAMES=$(remove_at_prefix "$HOSTNAMES")
MAPPING_FILE=$(remove_at_prefix "$MAPPING_FILE")
OUTPUT_DIR=$(remove_at_prefix "$OUTPUT_DIR")

# Find topology tool
if [[ -z "$TOPOLOGY_TOOL" ]]; then
    if [[ -f "./build/tools/umd/topology" ]]; then
        TOPOLOGY_TOOL="./build/tools/umd/topology"
    elif [[ -f "../build/tools/umd/topology" ]]; then
        TOPOLOGY_TOOL="../build/tools/umd/topology"
    elif command -v topology &> /dev/null; then
        TOPOLOGY_TOOL="topology"
    else
        echo -e "${RED}Error: topology tool not found. Please specify with --topology-tool${NC}" >&2
        exit 1
    fi
fi

if [[ ! -f "$TOPOLOGY_TOOL" ]] && ! command -v "$TOPOLOGY_TOOL" &> /dev/null; then
    echo -e "${RED}Error: Topology tool not found at: $TOPOLOGY_TOOL${NC}" >&2
    exit 1
fi

# Get list of hostnames
get_hostnames() {
    local hosts_list=()
    
    if [[ -f "$HOSTNAMES" ]]; then
        # Check if it's a YAML mapping file
        if grep -q "rank_to_cluster_mock_cluster_desc" "$HOSTNAMES" 2>/dev/null; then
            # Extract hostnames from cluster descriptor filenames
            while IFS= read -r line; do
                if [[ "$line" =~ \".*_([a-zA-Z0-9-]+)\.yaml\" ]]; then
                    filename=$(echo "$line" | grep -oE '[^/"]+\.yaml' | head -1)
                    if [[ "$filename" =~ _([a-zA-Z0-9-]+)\.yaml$ ]]; then
                        potential_host="${BASH_REMATCH[1]}"
                        if [[ "$potential_host" =~ ^[a-zA-Z0-9-]+$ ]] && [[ ${#potential_host} -gt 2 ]]; then
                            hosts_list+=("$potential_host")
                        fi
                    fi
                fi
            done < "$HOSTNAMES"
        fi
        
        # If no hosts found, try hostfile format
        if [[ ${#hosts_list[@]} -eq 0 ]]; then
            while IFS= read -r line; do
                [[ "$line" =~ ^[[:space:]]*# ]] && continue
                [[ -z "${line// }" ]] && continue
                hostname=$(echo "$line" | awk '{print $1}')
                [[ -n "$hostname" ]] && hosts_list+=("$hostname")
            done < "$HOSTNAMES"
        fi
    else
        # Comma-separated list
        IFS=',' read -ra ADDR <<< "$HOSTNAMES"
        for host in "${ADDR[@]}"; do
            hosts_list+=("$(echo "$host" | xargs)")
        done
    fi
    
    printf '%s\n' "${hosts_list[@]}" | sort -u
}

# Get hostnames
mapfile -t HOST_LIST < <(get_hostnames)

if [[ ${#HOST_LIST[@]} -eq 0 ]]; then
    echo -e "${RED}Error: No hostnames found${NC}" >&2
    exit 1
fi

NUM_HOSTS=${#HOST_LIST[@]}
HOSTS_STR=$(IFS=','; echo "${HOST_LIST[*]}")

echo -e "${BLUE}=== Generating Cluster Descriptors ===${NC}"
echo "Hosts: $HOSTS_STR"
echo "Number of hosts: $NUM_HOSTS"
echo "Topology tool: $TOPOLOGY_TOOL"
echo "Output directory: $OUTPUT_DIR"
echo "Base name: $BASE_NAME"
echo ""

# Create output directory
if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build the inline script
INLINE_SCRIPT="
RANK=\${OMPI_COMM_WORLD_RANK:-0}
HOSTS='${HOSTS_STR}'
OUTPUT_DIR='${OUTPUT_DIR}'
BASE_NAME='${BASE_NAME}'
TOPOLOGY_TOOL='${TOPOLOGY_TOOL}'

IFS=',' read -ra HOST_ARRAY <<< \"\$HOSTS\"
CURRENT_HOST=\"\${HOST_ARRAY[\$RANK]}\"
OUTPUT_FILE=\"\${OUTPUT_DIR}/\${BASE_NAME}_\${CURRENT_HOST}.yaml\"

echo \"[Rank \$RANK] Generating cluster descriptor for \$CURRENT_HOST...\" >&2
mkdir -p \"\$OUTPUT_DIR\"
\"\$TOPOLOGY_TOOL\" --path \"\$OUTPUT_FILE\"

if [[ -f \"\$OUTPUT_FILE\" ]]; then
    echo \"RANK_\${RANK}_FILE:\$OUTPUT_FILE\"
else
    echo \"ERROR: Failed to generate \$OUTPUT_FILE\" >&2
    exit 1
fi
"

# Build MPI command
MPI_CMD=("$MPI_LAUNCHER")
MPI_CMD+=("--host" "$HOSTS_STR")
MPI_CMD+=("--mca" "btl_tcp_if_exclude" "docker0,lo,tailscale0")
MPI_CMD+=("--tag-output")
MPI_CMD+=("-np" "$NUM_HOSTS")
MPI_CMD+=("bash" "-c" "$INLINE_SCRIPT")

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
    echo "${MPI_CMD[@]}"
    exit 0
fi

# Execute MPI command
echo -e "${YELLOW}Running MPI command...${NC}"
echo ""

set +e
"${MPI_CMD[@]}" 2>&1 | tee /tmp/mpi_output_$$.log
MPI_EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ $MPI_EXIT_CODE -ne 0 ]]; then
    echo -e "${RED}Error: MPI command failed with exit code $MPI_EXIT_CODE${NC}" >&2
    rm -f /tmp/mpi_output_$$.log
    exit 1
fi

# Extract generated file paths
declare -A RANK_TO_FILE
if [[ -f /tmp/mpi_output_$$.log ]]; then
    while IFS= read -r line; do
        if [[ "$line" =~ RANK_([0-9]+)_FILE:(.+)$ ]]; then
            rank="${BASH_REMATCH[1]}"
            file="${BASH_REMATCH[2]}"
            RANK_TO_FILE["$rank"]="$file"
        fi
    done < /tmp/mpi_output_$$.log
    rm -f /tmp/mpi_output_$$.log
fi

# Also check for files that were created
for rank in $(seq 0 $((NUM_HOSTS - 1))); do
    host="${HOST_LIST[$rank]}"
    expected_file="${OUTPUT_DIR}/${BASE_NAME}_${host}.yaml"
    if [[ -f "$expected_file" ]] && [[ -z "${RANK_TO_FILE[$rank]:-}" ]]; then
        RANK_TO_FILE["$rank"]="$expected_file"
    fi
done

# Generate mapping YAML file
echo ""
echo -e "${YELLOW}Generating mapping file: $MAPPING_FILE${NC}"

mkdir -p "$(dirname "$MAPPING_FILE")"

cat > "$MAPPING_FILE" << EOF
rank_to_cluster_mock_cluster_desc:
EOF

for rank in $(printf '%s\n' "${!RANK_TO_FILE[@]}" | sort -n); do
    file="${RANK_TO_FILE[$rank]}"
    if [[ "$file" =~ ^$(pwd)/ ]]; then
        rel_file="${file#$(pwd)/}"
    else
        rel_file="$file"
    fi
    echo "  $rank: \"$rel_file\"" >> "$MAPPING_FILE"
done

echo -e "${GREEN}✓ Mapping file created: $MAPPING_FILE${NC}"
echo ""
echo "Generated ${#RANK_TO_FILE[@]} cluster descriptor file(s)"
