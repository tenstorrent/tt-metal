#!/bin/bash
# Script to generate cluster descriptors from multiple hosts using MPI
# Usage: ./generate_cluster_descriptors.sh --rankfile <rankfile> --mapping-file <mapping.yaml> --output-dir <output_dir> --base-name <base_name> --topology-tool <path> --mpi-launcher <cmd>

set -euo pipefail

MAPPING_FILE=""
OUTPUT_DIR=""
BASE_NAME=""
TOPOLOGY_TOOL=""
MPI_LAUNCHER=""
DRY_RUN=false
RANK_BINDINGS_FILE=""
RANKFILE=""

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
    --mapping-file <file>         Output mapping YAML file path (with @ prefix)
    --output-dir <dir>            Directory to save cluster descriptor files (with @ prefix)
    --base-name <name>            Base name for cluster descriptor files (required)
    --rank-bindings-file <file>   Rank bindings YAML file (with @ prefix) - sets env vars per rank
    --rankfile <file>              MPI rankfile (with @ prefix) - maps ranks to hosts/slots (required)
    --topology-tool <path>        Path to topology tool (default: auto-detect)
    --mpi-launcher <cmd>          MPI launcher command (default: mpirun)
    --dry-run                     Show what would be executed without running
    -h, --help                    Show this help message

EXAMPLES:
    ./generate_cluster_descriptors.sh \\
      --rankfile rankfile.txt \\
      --mapping-file mapping.yaml \\
      --output-dir ./cluster_descs \\
      --base-name my_cluster_desc \\
      --topology-tool ./build/tools/umd/topology \\
      --mpi-launcher mpirun
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
        --rank-bindings-file)
            RANK_BINDINGS_FILE="$2"
            shift 2
            ;;
        --rankfile)
            RANKFILE="$2"
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
if [[ -z "$MAPPING_FILE" ]] || [[ -z "$OUTPUT_DIR" ]] || [[ -z "$BASE_NAME" ]]; then
    echo -e "${RED}Error: --mapping-file, --output-dir, and --base-name are required${NC}" >&2
    usage
fi

# Set default MPI launcher if not provided
if [[ -z "$MPI_LAUNCHER" ]]; then
    MPI_LAUNCHER="mpirun"
fi

# Require rankfile
if [[ -z "$RANKFILE" ]]; then
    echo -e "${RED}Error: --rankfile is required${NC}" >&2
    usage
fi

# Validate rankfile exists
if [[ ! -f "$RANKFILE" ]]; then
    echo -e "${RED}Error: Rankfile not found: $RANKFILE${NC}" >&2
    exit 1
fi

# Remove @ prefixes
MAPPING_FILE=$(remove_at_prefix "$MAPPING_FILE")
OUTPUT_DIR=$(remove_at_prefix "$OUTPUT_DIR")
if [[ -n "$RANK_BINDINGS_FILE" ]]; then
    RANK_BINDINGS_FILE=$(remove_at_prefix "$RANK_BINDINGS_FILE")
fi
RANKFILE=$(remove_at_prefix "$RANKFILE")

# Find topology tool if not provided
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

# Parse rank bindings file and extract environment variables per rank
declare -A RANK_ENV_VARS
if [[ -n "$RANK_BINDINGS_FILE" ]]; then
    if [[ ! -f "$RANK_BINDINGS_FILE" ]]; then
        echo -e "${RED}Error: Rank bindings file not found: $RANK_BINDINGS_FILE${NC}" >&2
        exit 1
    fi

    echo -e "${BLUE}Parsing rank bindings file: $RANK_BINDINGS_FILE${NC}"

    # Use Python to parse YAML and extract env vars for each rank
    # Create a temporary Python script to parse the YAML
    PYTHON_PARSE_SCRIPT=$(cat << 'PYTHON_EOF'
import sys
import yaml
import json

try:
    with open(sys.argv[1], 'r') as f:
        data = yaml.safe_load(f)

    rank_envs = {}
    if 'rank_bindings' in data:
        for binding in data['rank_bindings']:
            rank = binding.get('rank')
            env_overrides = binding.get('env_overrides', {})
            if rank is not None:
                rank_envs[str(rank)] = env_overrides

    # Output as JSON for bash to parse
    print(json.dumps(rank_envs))
except Exception as e:
    print(f"Error parsing rank bindings file: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
)

    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found. Required for parsing rank bindings file.${NC}" >&2
        exit 1
    fi

    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi

    # Parse the YAML file and get JSON representation
    RANK_ENV_JSON=$("$PYTHON_CMD" -c "$PYTHON_PARSE_SCRIPT" "$RANK_BINDINGS_FILE")
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Error: Failed to parse rank bindings file${NC}" >&2
        exit 1
    fi

    # Escape the JSON for safe embedding in bash script
    # Replace single quotes with '\'' and wrap in single quotes
    RANK_ENV_JSON_ESCAPED=$(echo "$RANK_ENV_JSON" | sed "s/'/'\\\\''/g")

    # Count ranks
    RANK_COUNT=$("$PYTHON_CMD" -c "import json, sys; print(len(json.loads(sys.argv[1])))" "$RANK_ENV_JSON")

    echo -e "${GREEN}✓ Parsed environment variables for $RANK_COUNT rank(s)${NC}"
else
    RANK_ENV_JSON_ESCAPED=""
    RANK_COUNT=0  # Will be set to NUM_HOSTS later if not provided
fi

# Parse rankfile to count ranks, extract hostnames, and map rank -> (hostname, slot)
declare -A RANK_TO_HOST
declare -A RANK_TO_SLOT
declare -A HOST_SET
RANKFILE_RANK_COUNT=0

while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    if [[ "$line" =~ ^rank[[:space:]]+([0-9]+)=([^[:space:]]+)[[:space:]]+slot=([0-9]+) ]]; then
        rank_num="${BASH_REMATCH[1]}"
        hostname="${BASH_REMATCH[2]}"
        slot="${BASH_REMATCH[3]}"
        HOST_SET["$hostname"]=1
        RANK_TO_HOST["$rank_num"]="$hostname"
        RANK_TO_SLOT["$rank_num"]="$slot"
        # Track max rank number
        if [[ $rank_num -ge $RANKFILE_RANK_COUNT ]]; then
            RANKFILE_RANK_COUNT=$((rank_num + 1))
        fi
    elif [[ "$line" =~ ^rank[[:space:]]+([0-9]+)=([^[:space:]]+) ]]; then
        # Rankfile must specify slot
        echo -e "${RED}Error: Rankfile entry for rank ${BASH_REMATCH[1]} missing slot specification. Format: rank N=hostname slot=X${NC}" >&2
        exit 1
    fi
done < "$RANKFILE"

# Convert host set to sorted array
mapfile -t HOST_LIST < <(printf '%s\n' "${!HOST_SET[@]}" | sort -u)
NUM_HOSTS=${#HOST_LIST[@]}

# Use rank count from rankfile if rank bindings not provided, otherwise use rank bindings count
if [[ $RANK_COUNT -gt 0 ]]; then
    TOTAL_RANKS=$RANK_COUNT
    if [[ $TOTAL_RANKS -ne $RANKFILE_RANK_COUNT ]]; then
        echo -e "${YELLOW}Warning: Rank bindings file has $TOTAL_RANKS ranks, but rankfile has $RANKFILE_RANK_COUNT ranks${NC}" >&2
        echo -e "${YELLOW}Using rank bindings count: $TOTAL_RANKS${NC}" >&2
    fi
else
    TOTAL_RANKS=$RANKFILE_RANK_COUNT
fi

echo -e "${BLUE}Parsed rankfile: $RANKFILE${NC}"
echo -e "${BLUE}Found $RANKFILE_RANK_COUNT ranks across $NUM_HOSTS host(s)${NC}"

echo -e "${BLUE}=== Generating Cluster Descriptors ===${NC}"
echo "Rankfile: $RANKFILE"
echo "Number of hosts: $NUM_HOSTS"
echo "Total ranks: $TOTAL_RANKS"
echo "Topology tool: $TOPOLOGY_TOOL"
echo "Output directory: $OUTPUT_DIR"
echo "Base name: $BASE_NAME"
echo ""

# Create output directory
if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build rank-to-host and rank-to-slot mappings for inline script
RANK_HOST_SLOT_JSON=""
if [[ ${#RANK_TO_HOST[@]} -gt 0 ]]; then
    # Create JSON mapping rank -> {host, slot}
    RANK_HOST_SLOT_JSON=$(python3 -c "
import json
rank_to_host = {}
rank_to_slot = {}
$(for rank in "${!RANK_TO_HOST[@]}"; do
    echo "rank_to_host['$rank'] = '${RANK_TO_HOST[$rank]}'"
    echo "rank_to_slot['$rank'] = '${RANK_TO_SLOT[$rank]}'"
done)
result = {}
for rank in rank_to_host:
    result[rank] = {'host': rank_to_host[rank], 'slot': rank_to_slot[rank]}
print(json.dumps(result))
" 2>/dev/null || echo "{}")
    RANK_HOST_SLOT_JSON_ESCAPED=$(echo "$RANK_HOST_SLOT_JSON" | sed "s/'/'\\\\''/g")
fi

# Build the inline script
# If rank bindings file is provided, set environment variables per rank
if [[ -n "$RANK_ENV_JSON_ESCAPED" ]]; then
    INLINE_SCRIPT="
RANK=\${OMPI_COMM_WORLD_RANK:-0}
export RANK
OUTPUT_DIR='${OUTPUT_DIR}'
BASE_NAME='${BASE_NAME}'
TOPOLOGY_TOOL='${TOPOLOGY_TOOL}'
RANK_ENV_JSON='${RANK_ENV_JSON_ESCAPED}'
RANK_HOST_SLOT_JSON='${RANK_HOST_SLOT_JSON_ESCAPED}'

# Set environment variables from rank bindings
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    PYTHON_CMD=\"\"
fi

if [[ -n \"\$PYTHON_CMD\" ]] && [[ -n \"\$RANK_ENV_JSON\" ]]; then
    # Use Python to generate export statements, then eval them
    while IFS= read -r export_line; do
        if [[ -n \"\$export_line\" ]] && [[ \"\$export_line\" =~ ^export ]]; then
            eval \"\$export_line\"
        fi
    done < <(\"\$PYTHON_CMD\" -c \"
import json
import os
import sys
import shlex
rank = int(os.environ.get('RANK', '0'))
rank_envs = json.loads(sys.argv[1])
rank_str = str(rank)
if rank_str in rank_envs:
    env_vars = rank_envs[rank_str]
    for key, value in env_vars.items():
        # Escape the value properly for bash
        escaped_value = shlex.quote(str(value))
        print(f'export {key}={escaped_value}')
        print(f'[Rank {rank}] Set {key}={value}', file=sys.stderr)
\" \"\$RANK_ENV_JSON\")
fi

# Determine hostname and slot from rankfile mapping or hostname command
if [[ -n \"\$RANK_HOST_SLOT_JSON\" ]] && [[ -n \"\$PYTHON_CMD\" ]]; then
    RANK_HOST_SLOT=\$(\"\$PYTHON_CMD\" -c \"
import json
import os
import sys
rank = int(os.environ.get('RANK', '0'))
rank_host_slot = json.loads(sys.argv[1])
rank_str = str(rank)
if rank_str in rank_host_slot:
    host = rank_host_slot[rank_str]['host']
    slot = rank_host_slot[rank_str]['slot']
    print(f'{host} {slot}')
\" \"\$RANK_HOST_SLOT_JSON\")
    if [[ -n \"\$RANK_HOST_SLOT\" ]]; then
        CURRENT_HOST=\$(echo \"\$RANK_HOST_SLOT\" | awk '{print \$1}')
        CURRENT_SLOT=\$(echo \"\$RANK_HOST_SLOT\" | awk '{print \$2}')
    else
        echo \"ERROR: Rank \$RANK not found in rankfile mapping\" >&2
        exit 1
    fi
else
    echo \"ERROR: Rankfile mapping not available\" >&2
    exit 1
fi

# Generate filename with hostname and slot
OUTPUT_FILE=\"\${OUTPUT_DIR}/\${BASE_NAME}_\${CURRENT_HOST}_slot_\${CURRENT_SLOT}.yaml\"

echo \"[Rank \$RANK] Generating cluster descriptor on \$CURRENT_HOST slot \$CURRENT_SLOT...\" >&2
mkdir -p \"\$OUTPUT_DIR\"
\"\$TOPOLOGY_TOOL\" --path \"\$OUTPUT_FILE\"

if [[ -f \"\$OUTPUT_FILE\" ]]; then
    echo \"RANK_\${RANK}_FILE:\$OUTPUT_FILE\"
else
    echo \"ERROR: Failed to generate \$OUTPUT_FILE\" >&2
    exit 1
fi
"
else
    # Build rank-to-host and rank-to-slot mappings for inline script (no rank bindings case)
    RANK_HOST_SLOT_JSON=""
    if [[ ${#RANK_TO_HOST[@]} -gt 0 ]]; then
        RANK_HOST_SLOT_JSON=$(python3 -c "
import json
rank_to_host = {}
rank_to_slot = {}
$(for rank in "${!RANK_TO_HOST[@]}"; do
    echo "rank_to_host['$rank'] = '${RANK_TO_HOST[$rank]}'"
    echo "rank_to_slot['$rank'] = '${RANK_TO_SLOT[$rank]}'"
done)
result = {}
for rank in rank_to_host:
    result[rank] = {'host': rank_to_host[rank], 'slot': rank_to_slot[rank]}
print(json.dumps(result))
" 2>/dev/null || echo "{}")
        RANK_HOST_SLOT_JSON_ESCAPED=$(echo "$RANK_HOST_SLOT_JSON" | sed "s/'/'\\\\''/g")
    fi

    INLINE_SCRIPT="
RANK=\${OMPI_COMM_WORLD_RANK:-0}
OUTPUT_DIR='${OUTPUT_DIR}'
BASE_NAME='${BASE_NAME}'
TOPOLOGY_TOOL='${TOPOLOGY_TOOL}'
RANK_HOST_SLOT_JSON='${RANK_HOST_SLOT_JSON_ESCAPED}'

# Determine hostname and slot from rankfile mapping or hostname command
if [[ -n \"\$RANK_HOST_SLOT_JSON\" ]]; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        PYTHON_CMD=\"\"
    fi
    if [[ -n \"\$PYTHON_CMD\" ]]; then
        RANK_HOST_SLOT=\$(\"\$PYTHON_CMD\" -c \"
import json
import os
import sys
rank = int(os.environ.get('RANK', '0'))
rank_host_slot = json.loads(sys.argv[1])
rank_str = str(rank)
if rank_str in rank_host_slot:
    host = rank_host_slot[rank_str]['host']
    slot = rank_host_slot[rank_str]['slot']
    print(f'{host} {slot}')
\" \"\$RANK_HOST_SLOT_JSON\")
        if [[ -n \"\$RANK_HOST_SLOT\" ]]; then
            CURRENT_HOST=\$(echo \"\$RANK_HOST_SLOT\" | awk '{print \$1}')
            CURRENT_SLOT=\$(echo \"\$RANK_HOST_SLOT\" | awk '{print \$2}')
        else
            echo \"ERROR: Rank \$RANK not found in rankfile mapping\" >&2
            exit 1
        fi
    else
        echo \"ERROR: Python not available for rankfile mapping\" >&2
        exit 1
    fi
else
    echo \"ERROR: Rankfile mapping not available\" >&2
    exit 1
fi

# Generate filename with hostname and slot from rankfile
OUTPUT_FILE=\"\${OUTPUT_DIR}/\${BASE_NAME}_\${CURRENT_HOST}_slot_\${CURRENT_SLOT}.yaml\"

echo \"[Rank \$RANK] Generating cluster descriptor on \$CURRENT_HOST slot \$CURRENT_SLOT...\" >&2
mkdir -p \"\$OUTPUT_DIR\"
\"\$TOPOLOGY_TOOL\" --path \"\$OUTPUT_FILE\"

if [[ -f \"\$OUTPUT_FILE\" ]]; then
    echo \"RANK_\${RANK}_FILE:\$OUTPUT_FILE\"
else
    echo \"ERROR: Failed to generate \$OUTPUT_FILE\" >&2
    exit 1
fi
"
fi

# Build MPI command
MPI_CMD=("$MPI_LAUNCHER")
# Use rankfile - it specifies rank-to-host mapping
MPI_CMD+=("--rankfile" "$RANKFILE")
MPI_CMD+=("--oversubscribe")
MPI_CMD+=("--mca" "btl_tcp_if_exclude" "docker0,lo,tailscale0")
MPI_CMD+=("--tag-output")
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
for rank in $(seq 0 $((TOTAL_RANKS - 1))); do
    # Check for files named with hostname and slot (from rankfile)
    if [[ -n "${RANK_TO_HOST[$rank]:-}" ]] && [[ -n "${RANK_TO_SLOT[$rank]:-}" ]]; then
        host="${RANK_TO_HOST[$rank]}"
        slot="${RANK_TO_SLOT[$rank]}"
        expected_file="${OUTPUT_DIR}/${BASE_NAME}_${host}_slot_${slot}.yaml"
        if [[ -f "$expected_file" ]] && [[ -z "${RANK_TO_FILE[$rank]:-}" ]]; then
            RANK_TO_FILE["$rank"]="$expected_file"
        fi
    fi
done

# Generate mapping YAML file
echo ""
echo -e "${YELLOW}Generating mapping file: $MAPPING_FILE${NC}"

mkdir -p "$(dirname "$MAPPING_FILE")"

cat > "$MAPPING_FILE" << EOF
rank_to_cluster_mock_cluster_desc:
EOF

# Verify all ranks from rank bindings have corresponding files
MISSING_RANKS=()
if [[ $RANK_COUNT -gt 0 ]]; then
    for rank in $(seq 0 $((RANK_COUNT - 1))); do
        if [[ -z "${RANK_TO_FILE[$rank]:-}" ]]; then
            MISSING_RANKS+=("$rank")
        fi
    done

    if [[ ${#MISSING_RANKS[@]} -gt 0 ]]; then
        echo -e "${RED}Warning: Missing cluster descriptors for ranks: ${MISSING_RANKS[*]}${NC}" >&2
    fi
fi

# Write mapping file - ensure ranks are in order matching rank bindings
if [[ $RANK_COUNT -gt 0 ]]; then
    # Use rank bindings order (0 to RANK_COUNT-1)
    for rank in $(seq 0 $((RANK_COUNT - 1))); do
        file="${RANK_TO_FILE[$rank]:-}"
        if [[ -z "$file" ]]; then
            # Try to generate expected filename from rankfile mapping
            if [[ -n "${RANK_TO_HOST[$rank]:-}" ]] && [[ -n "${RANK_TO_SLOT[$rank]:-}" ]]; then
                host="${RANK_TO_HOST[$rank]}"
                slot="${RANK_TO_SLOT[$rank]}"
                file="${OUTPUT_DIR}/${BASE_NAME}_${host}_slot_${slot}.yaml"
                echo -e "${YELLOW}Warning: Rank $rank file not found, using expected path from rankfile: $file${NC}" >&2
            else
                echo -e "${RED}Error: Rank $rank file not found and no rankfile mapping available${NC}" >&2
                exit 1
            fi
        fi
        if [[ "$file" =~ ^$(pwd)/ ]]; then
            rel_file="${file#$(pwd)/}"
        else
            rel_file="$file"
        fi
        echo "  $rank: \"$rel_file\"" >> "$MAPPING_FILE"
    done
else
    # When rank bindings not provided, use rankfile to determine ranks
    if [[ $RANKFILE_RANK_COUNT -gt 0 ]]; then
        for rank in $(seq 0 $((RANKFILE_RANK_COUNT - 1))); do
            file="${RANK_TO_FILE[$rank]:-}"
            if [[ -z "$file" ]]; then
                # Generate expected filename from rankfile mapping
                if [[ -n "${RANK_TO_HOST[$rank]:-}" ]] && [[ -n "${RANK_TO_SLOT[$rank]:-}" ]]; then
                    host="${RANK_TO_HOST[$rank]}"
                    slot="${RANK_TO_SLOT[$rank]}"
                    file="${OUTPUT_DIR}/${BASE_NAME}_${host}_slot_${slot}.yaml"
                    echo -e "${YELLOW}Warning: Rank $rank file not found, using expected path from rankfile: $file${NC}" >&2
                else
                    echo -e "${RED}Error: Rank $rank not found in rankfile mapping${NC}" >&2
                    exit 1
                fi
            fi
            if [[ "$file" =~ ^$(pwd)/ ]]; then
                rel_file="${file#$(pwd)/}"
            else
                rel_file="$file"
            fi
            echo "  $rank: \"$rel_file\"" >> "$MAPPING_FILE"
        done
    else
        echo -e "${RED}Error: Cannot generate mapping file without rank bindings or rankfile${NC}" >&2
        exit 1
    fi
fi

echo -e "${GREEN}✓ Mapping file created: $MAPPING_FILE${NC}"
echo ""
echo "Generated ${#RANK_TO_FILE[@]} cluster descriptor file(s)"

# Verify mapping matches rank bindings
if [[ $RANK_COUNT -gt 0 ]] && [[ -n "$RANK_BINDINGS_FILE" ]]; then
    echo -e "${BLUE}Verifying mapping matches rank bindings...${NC}"
    MAPPED_COUNT=$(grep -c "^  [0-9]" "$MAPPING_FILE" 2>/dev/null || echo "0")
    if [[ $MAPPED_COUNT -eq $RANK_COUNT ]]; then
        echo -e "${GREEN}✓ Mapping file has $MAPPED_COUNT ranks matching rank bindings file${NC}"
    else
        echo -e "${YELLOW}Warning: Mapping file has $MAPPED_COUNT ranks, but rank bindings has $RANK_COUNT ranks${NC}" >&2
    fi
fi
