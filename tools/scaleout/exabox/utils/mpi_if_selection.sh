#!/bin/bash

# MPI network interface validation and auto-detection utility
# Used by exabox cluster scripts to ensure valid MPI interface selection

# Function to test if a given network interface works for MPI communication
test_mpi_interface() {
    local interface="$1"
    local test_host="${2:-$(hostname)}"

    # Quick local test: does the interface exist?
    if ! ip link show "$interface" &>/dev/null; then
        return 1
    fi

    # MPI OOB (out-of-band) connectivity test
    timeout 3 mpirun --host "$test_host" \
        --mca oob_tcp_if_include "$interface" \
        --mca btl_tcp_if_include "$interface" \
        -np 1 hostname &>/dev/null

    return $?
}

# Function to validate and auto-detect MPI network interface
validate_mpi_interface() {
    local interface="$1"
    local explicit="$2"
    local first_host="${3:-$(hostname)}"

    # Get list of available interfaces (excluding loopback)
    local available_interfaces
    available_interfaces=$(ip link show | grep -E '^[0-9]+:' | awk -F': ' '{print $2}' | cut -d'@' -f1 | grep -v '^lo$' || true)

    # Check if explicitly provided interface exists and works
    if [[ "$explicit" == "true" ]]; then
        if ! echo "$available_interfaces" | grep -qx "$interface"; then
            echo "Error: Specified MPI interface '$interface' not found on this system" >&2
            echo "" >&2
            echo "Available network interfaces:" >&2
            echo "$available_interfaces" | sed 's/^/  - /' >&2
            echo "" >&2
            echo "Use --mpi-if <interface> to specify a different interface" >&2
            echo "Example: --mpi-if enp2s0f0np0" >&2
            exit 1
        fi

        # Test if the interface actually works for MPI
        if ! test_mpi_interface "$interface" "$first_host"; then
            echo "Error: Specified MPI interface '$interface' exists but cannot be used for MPI communication" >&2
            echo "" >&2
            echo "This usually means the interface is down or misconfigured." >&2
            echo "Check interface status with: ip link show $interface" >&2
            echo "" >&2

            # Try to auto-detect a working interface to suggest to the user
            echo "Attempting to find a working alternative interface..." >&2
            local suggested_interface=""
            local candidate_interfaces
            candidate_interfaces=$(echo "$available_interfaces" | grep -vE '^(docker|flannel|cali|veth|br-|virbr)' || true)
            [[ -z "$candidate_interfaces" ]] && candidate_interfaces="$available_interfaces"

            while IFS= read -r iface; do
                [[ -z "$iface" ]] && continue
                if test_mpi_interface "$iface" "$first_host"; then
                    suggested_interface="$iface"
                    break
                fi
            done <<< "$candidate_interfaces"

            if [[ -n "$suggested_interface" ]]; then
                echo "Suggested working interface: $suggested_interface" >&2
                echo "Try: --mpi-if $suggested_interface" >&2
                echo "Or remove --mpi-if to use auto-detection" >&2
            else
                echo "No working MPI interface found on this system." >&2
            fi

            exit 1
        fi

        return 0
    fi

    # Auto-detect: Prioritize physical-looking interfaces (exclude known virtual patterns)
    local candidate_interfaces
    candidate_interfaces=$(echo "$available_interfaces" | grep -vE '^(docker|flannel|cali|veth|br-|virbr)' || true)

    if [[ -z "$candidate_interfaces" ]]; then
        candidate_interfaces="$available_interfaces"
    fi

    # Build priority list: UP interfaces first, sorted by speed if ethtool available
    local priority_interfaces=()

    if command -v ethtool &> /dev/null; then
        # Get UP interfaces with speed info
        while IFS= read -r iface; do
            if ip link show "$iface" 2>/dev/null | grep -q 'state UP'; then
                local speed
                speed=$(ethtool "$iface" 2>/dev/null | awk '/Speed:/ {print $2}' | sed 's/Mb\/s//;s/Gb\/s/000/' | grep -E '^[0-9]+$')
                if [[ -n "$speed" ]]; then
                    priority_interfaces+=("$speed:$iface")
                else
                    priority_interfaces+=("0:$iface")
                fi
            fi
        done <<< "$candidate_interfaces"

        # Sort by speed (descending) and extract interface names
        local sorted_interfaces=()
        while IFS= read -r entry; do
            sorted_interfaces+=("${entry#*:}")
        done < <(printf '%s\n' "${priority_interfaces[@]}" | sort -t: -k1 -rn)

        # Add DOWN interfaces at the end
        while IFS= read -r iface; do
            if ! ip link show "$iface" 2>/dev/null | grep -q 'state UP'; then
                sorted_interfaces+=("$iface")
            fi
        done <<< "$candidate_interfaces"

        candidate_interfaces=$(printf '%s\n' "${sorted_interfaces[@]}")
    fi

    # Test each candidate interface with actual MPI connectivity
    local best_interface=""
    while IFS= read -r iface; do
        [[ -z "$iface" ]] && continue

        if test_mpi_interface "$iface" "$first_host"; then
            best_interface="$iface"
            break
        fi
    done <<< "$candidate_interfaces"

    if [[ -z "$best_interface" ]]; then
        if [[ -z "$interface" ]]; then
            echo "Error: No suitable MPI network interface could be auto-detected" >&2
        else
            echo "Error: No suitable MPI network interface found (default '$interface' not available)" >&2
        fi
        echo "" >&2
        echo "Available network interfaces:" >&2
        echo "$available_interfaces" | sed 's/^/  - /' >&2
        echo "" >&2
        echo "None of the available interfaces passed MPI connectivity test." >&2
        echo "Use --mpi-if <interface> to specify an interface explicitly" >&2
        echo "Example: --mpi-if <interface-name>" >&2
        exit 1
    fi

    echo "$best_interface"
}
