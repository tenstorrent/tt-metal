#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: host-load-sidecar.sh [options] [--] <command> [args...]

Wrap a command and print a host load summary after it exits.

Options:
  --interval SECONDS  Sampling interval. Defaults to HOST_LOAD_INTERVAL_SECONDS or 1.
  --label LABEL       Label shown in the summary. Defaults to HOST_LOAD_LABEL or "host load".
  --output FILE       Also append the summary to FILE.
  -h, --help          Show this help.

Environment:
  HOST_LOAD_INTERVAL_SECONDS  Default sampling interval.
  HOST_LOAD_LABEL             Default summary label.
  HOST_LOAD_SUMMARY_FILE      Default file to append the summary to.
  HOST_LOAD_DISK_DEVICES      Comma-separated /sys/block device names to include.
                             Defaults to whole non-loop, non-ram, non-dm block devices.
                             Disk metrics are physical block-device I/O; cached
                             filesystem reads are not counted as disk reads.
  HOST_LOAD_NET_INTERFACES    Comma-separated network interface names to include.
                             Defaults to all interfaces except lo.

Examples:
  host-load-sidecar.sh -- make -j32
  host-load-sidecar.sh --interval 2 --label build --output "$GITHUB_STEP_SUMMARY" -- pytest tests/
EOF
}

die() {
    echo "host-load-sidecar: $*" >&2
    exit 2
}

interval="${HOST_LOAD_INTERVAL_SECONDS:-1}"
label="${HOST_LOAD_LABEL:-host load}"
summary_file="${HOST_LOAD_SUMMARY_FILE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)
            [[ $# -ge 2 ]] || die "--interval requires a value"
            interval="$2"
            shift 2
            ;;
        --label)
            [[ $# -ge 2 ]] || die "--label requires a value"
            label="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || die "--output requires a value"
            summary_file="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        --*)
            die "unknown option: $1"
            ;;
        *)
            break
            ;;
    esac
done

[[ $# -gt 0 ]] || {
    usage >&2
    exit 2
}

if ! awk -v value="$interval" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0) }'; then
    die "--interval must be a positive number"
fi

for required_file in /proc/stat /proc/meminfo /proc/net/dev /proc/uptime; do
    [[ -r "$required_file" ]] || die "$required_file is required; this script is intended for Ubuntu/Linux hosts"
done
[[ -d /sys/block ]] || die "/sys/block is required; this script is intended for Ubuntu/Linux hosts"

command_to_run=("$@")
tmp_dir="$(mktemp -d)"
samples_file="$tmp_dir/samples.tsv"
stats_file="$tmp_dir/stats.env"
summary_tmp="$tmp_dir/summary.txt"
stop_file="$tmp_dir/stop"
ready_file="$tmp_dir/ready"
monitor_pid=""
command_pid=""

# shellcheck disable=SC2329  # Invoked by the EXIT trap.
cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT

now_seconds() {
    awk '{ print $1 }' /proc/uptime
}

cpu_capacity() {
    getconf _NPROCESSORS_ONLN 2>/dev/null \
        || nproc --all 2>/dev/null \
        || awk -F: '/^processor[[:space:]]*:/ { n++ } END { print n ? n : "unknown" }' /proc/cpuinfo
}

memory_capacity_bytes() {
    awk '/^MemTotal:/ { printf "%.0f\n", $2 * 1024 }' /proc/meminfo
}

swap_capacity_bytes() {
    awk '/^SwapTotal:/ { printf "%.0f\n", $2 * 1024 }' /proc/meminfo
}

read_cpu_jiffies() {
    local user nice system idle iowait irq softirq steal guest guest_nice
    read -r _ user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat

    iowait="${iowait:-0}"
    irq="${irq:-0}"
    softirq="${softirq:-0}"
    steal="${steal:-0}"
    guest="${guest:-0}"
    guest_nice="${guest_nice:-0}"

    local total=$((user + nice + system + idle + iowait + irq + softirq + steal + guest + guest_nice))
    local idle_total=$((idle + iowait))

    printf "%s %s\n" "$total" "$idle_total"
}

read_memory_percent() {
    awk '
        /^MemTotal:/ { total = $2 }
        /^MemAvailable:/ { available = $2 }
        END {
            if (total > 0) {
                printf "%.2f\n", ((total - available) * 100) / total
            } else {
                print "0.00"
            }
        }
    ' /proc/meminfo
}

read_swap_percent() {
    awk '
        /^SwapTotal:/ { total = $2 }
        /^SwapFree:/ { free = $2 }
        END {
            if (total > 0) {
                printf "%.2f\n", ((total - free) * 100) / total
            } else {
                print "0.00"
            }
        }
    ' /proc/meminfo
}

selected_disk_stat_files() {
    if [[ -n "${HOST_LOAD_DISK_DEVICES:-}" ]]; then
        local -a configured_devices
        local device
        IFS=',' read -r -a configured_devices <<< "$HOST_LOAD_DISK_DEVICES"
        for device in "${configured_devices[@]}"; do
            device="${device#/dev/}"
            [[ -r "/sys/block/$device/stat" ]] && printf "%s\n" "/sys/block/$device/stat"
        done
        return
    fi

    local stat_file device
    for stat_file in /sys/block/*/stat; do
        [[ -r "$stat_file" ]] || continue
        device="${stat_file%/stat}"
        device="${device##*/}"
        case "$device" in
            loop*|ram*|zram*|fd*|sr*|dm-*)
                continue
                ;;
        esac
        printf "%s\n" "$stat_file"
    done
}

read_disk_bytes() {
    local read_sectors=0
    local write_sectors=0
    local stat_file device_read_sectors device_write_sectors

    # /sys/block/*/stat reports block-layer sectors, so this is physical device
    # I/O. It intentionally does not count filesystem reads served from cache.
    while IFS= read -r stat_file; do
        read -r _ _ device_read_sectors _ _ _ device_write_sectors _ _ < "$stat_file" || continue
        read_sectors=$((read_sectors + device_read_sectors))
        write_sectors=$((write_sectors + device_write_sectors))
    done < <(selected_disk_stat_files)

    printf "%s %s\n" "$((read_sectors * 512))" "$((write_sectors * 512))"
}

read_network_bytes() {
    awk -v configured_interfaces="${HOST_LOAD_NET_INTERFACES:-}" '
        BEGIN {
            if (configured_interfaces != "") {
                split(configured_interfaces, configured, ",")
                for (i in configured) {
                    wanted[configured[i]] = 1
                }
                use_wanted = 1
            }
        }
        NR > 2 {
            split($0, parts, ":")
            iface = parts[1]
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", iface)

            if (use_wanted) {
                if (!(iface in wanted)) {
                    next
                }
            } else if (iface == "lo") {
                next
            }

            split(parts[2], fields)
            rx += fields[1]
            tx += fields[9]
        }
        END {
            printf "%.0f %.0f\n", rx, tx
        }
    ' /proc/net/dev
}

sleep_until_interval_or_stop() {
    local target_interval="$1"
    local slept="0"
    local chunk

    while awk -v slept="$slept" -v target="$target_interval" 'BEGIN { exit !(slept < target) }'; do
        [[ -f "$stop_file" ]] && break
        chunk="$(
            awk -v slept="$slept" -v target="$target_interval" '
                BEGIN {
                    remaining = target - slept
                    if (remaining > 0.2) {
                        remaining = 0.2
                    }
                    if (remaining < 0.01) {
                        remaining = 0.01
                    }
                    printf "%.3f\n", remaining
                }
            '
        )"
        sleep "$chunk"
        slept="$(awk -v slept="$slept" -v chunk="$chunk" 'BEGIN { printf "%.6f\n", slept + chunk }')"
    done
}

monitor_load() {
    printf "elapsed_s\tcpu_pct\tmem_pct\tswap_pct\tdisk_read_bps\tdisk_write_bps\tnet_rx_bps\tnet_tx_bps\tdisk_read_bytes\tdisk_write_bytes\tnet_rx_bytes\tnet_tx_bytes\n" > "$samples_file"

    local previous_time previous_cpu_total previous_cpu_idle previous_disk_read previous_disk_write
    local previous_net_rx previous_net_tx

    previous_time="$(now_seconds)"
    read -r previous_cpu_total previous_cpu_idle < <(read_cpu_jiffies)
    read -r previous_disk_read previous_disk_write < <(read_disk_bytes)
    read -r previous_net_rx previous_net_tx < <(read_network_bytes)
    : > "$ready_file"

    while true; do
        sleep_until_interval_or_stop "$interval"

        local current_time current_cpu_total current_cpu_idle current_disk_read current_disk_write
        local current_net_rx current_net_tx elapsed_s cpu_delta idle_delta cpu_pct mem_pct swap_pct
        local disk_read_delta disk_write_delta net_rx_delta net_tx_delta
        local disk_read_bps disk_write_bps net_rx_bps net_tx_bps

        current_time="$(now_seconds)"
        read -r current_cpu_total current_cpu_idle < <(read_cpu_jiffies)
        read -r current_disk_read current_disk_write < <(read_disk_bytes)
        read -r current_net_rx current_net_tx < <(read_network_bytes)

        elapsed_s="$(awk -v current="$current_time" -v previous="$previous_time" 'BEGIN { elapsed = current - previous; if (elapsed <= 0) elapsed = 0.001; printf "%.6f\n", elapsed }')"
        cpu_delta=$((current_cpu_total - previous_cpu_total))
        idle_delta=$((current_cpu_idle - previous_cpu_idle))
        cpu_pct="$(
            awk -v total="$cpu_delta" -v idle="$idle_delta" '
                BEGIN {
                    if (total > 0) {
                        printf "%.2f\n", ((total - idle) * 100) / total
                    } else {
                        print "0.00"
                    }
                }
            '
        )"
        mem_pct="$(read_memory_percent)"
        swap_pct="$(read_swap_percent)"

        disk_read_delta=$((current_disk_read - previous_disk_read))
        disk_write_delta=$((current_disk_write - previous_disk_write))
        net_rx_delta=$((current_net_rx - previous_net_rx))
        net_tx_delta=$((current_net_tx - previous_net_tx))

        disk_read_bps="$(awk -v bytes="$disk_read_delta" -v elapsed="$elapsed_s" 'BEGIN { printf "%.2f\n", bytes / elapsed }')"
        disk_write_bps="$(awk -v bytes="$disk_write_delta" -v elapsed="$elapsed_s" 'BEGIN { printf "%.2f\n", bytes / elapsed }')"
        net_rx_bps="$(awk -v bytes="$net_rx_delta" -v elapsed="$elapsed_s" 'BEGIN { printf "%.2f\n", bytes / elapsed }')"
        net_tx_bps="$(awk -v bytes="$net_tx_delta" -v elapsed="$elapsed_s" 'BEGIN { printf "%.2f\n", bytes / elapsed }')"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$elapsed_s" "$cpu_pct" "$mem_pct" "$swap_pct" "$disk_read_bps" "$disk_write_bps" \
            "$net_rx_bps" "$net_tx_bps" "$disk_read_delta" "$disk_write_delta" \
            "$net_rx_delta" "$net_tx_delta" >> "$samples_file"

        previous_time="$current_time"
        previous_cpu_total="$current_cpu_total"
        previous_cpu_idle="$current_cpu_idle"
        previous_disk_read="$current_disk_read"
        previous_disk_write="$current_disk_write"
        previous_net_rx="$current_net_rx"
        previous_net_tx="$current_net_tx"

        [[ -f "$stop_file" ]] && break
    done
}

summarize_samples() {
    awk '
        BEGIN { FS = "\t" }
        NR == 1 { next }
        {
            n++
            elapsed += $1

            cpu = $2
            mem = $3
            swap = $4
            disk_read_bps = $5
            disk_write_bps = $6
            net_rx_bps = $7
            net_tx_bps = $8
            disk_read_bytes += $9
            disk_write_bytes += $10
            net_rx_bytes += $11
            net_tx_bytes += $12

            if (n == 1) {
                cpu_min = cpu_max = cpu
                mem_min = mem_max = mem
                swap_min = swap_max = swap
                disk_read_bps_min = disk_read_bps_max = disk_read_bps
                disk_write_bps_min = disk_write_bps_max = disk_write_bps
                net_rx_bps_min = net_rx_bps_max = net_rx_bps
                net_tx_bps_min = net_tx_bps_max = net_tx_bps
            }

            if (cpu < cpu_min) cpu_min = cpu
            if (cpu > cpu_max) cpu_max = cpu
            if (mem < mem_min) mem_min = mem
            if (mem > mem_max) mem_max = mem
            if (swap < swap_min) swap_min = swap
            if (swap > swap_max) swap_max = swap
            if (disk_read_bps < disk_read_bps_min) disk_read_bps_min = disk_read_bps
            if (disk_read_bps > disk_read_bps_max) disk_read_bps_max = disk_read_bps
            if (disk_write_bps < disk_write_bps_min) disk_write_bps_min = disk_write_bps
            if (disk_write_bps > disk_write_bps_max) disk_write_bps_max = disk_write_bps
            if (net_rx_bps < net_rx_bps_min) net_rx_bps_min = net_rx_bps
            if (net_rx_bps > net_rx_bps_max) net_rx_bps_max = net_rx_bps
            if (net_tx_bps < net_tx_bps_min) net_tx_bps_min = net_tx_bps
            if (net_tx_bps > net_tx_bps_max) net_tx_bps_max = net_tx_bps

            cpu_sum += cpu
            mem_sum += mem
            swap_sum += swap
            disk_read_bps_sum += disk_read_bps
            disk_write_bps_sum += disk_write_bps
            net_rx_bps_sum += net_rx_bps
            net_tx_bps_sum += net_tx_bps
        }
        END {
            if (n == 0) {
                print "samples=0"
                print "duration_s=0.00"
                print "cpu_min=0.00"; print "cpu_max=0.00"; print "cpu_mean=0.00"
                print "mem_min=0.00"; print "mem_max=0.00"; print "mem_mean=0.00"
                print "swap_min=0.00"; print "swap_max=0.00"; print "swap_mean=0.00"
                print "disk_read_bps_min=0.00"; print "disk_read_bps_max=0.00"; print "disk_read_bps_mean=0.00"
                print "disk_write_bps_min=0.00"; print "disk_write_bps_max=0.00"; print "disk_write_bps_mean=0.00"
                print "net_rx_bps_min=0.00"; print "net_rx_bps_max=0.00"; print "net_rx_bps_mean=0.00"
                print "net_tx_bps_min=0.00"; print "net_tx_bps_max=0.00"; print "net_tx_bps_mean=0.00"
                print "disk_read_total=0"; print "disk_write_total=0"; print "net_rx_total=0"; print "net_tx_total=0"
                print "disk_read_avg_bps=0.00"; print "disk_write_avg_bps=0.00"; print "net_rx_avg_bps=0.00"; print "net_tx_avg_bps=0.00"
                exit
            }

            printf "samples=%d\n", n
            printf "duration_s=%.2f\n", elapsed
            printf "cpu_min=%.2f\ncpu_max=%.2f\ncpu_mean=%.2f\n", cpu_min, cpu_max, cpu_sum / n
            printf "mem_min=%.2f\nmem_max=%.2f\nmem_mean=%.2f\n", mem_min, mem_max, mem_sum / n
            printf "swap_min=%.2f\nswap_max=%.2f\nswap_mean=%.2f\n", swap_min, swap_max, swap_sum / n
            printf "disk_read_bps_min=%.2f\ndisk_read_bps_max=%.2f\ndisk_read_bps_mean=%.2f\n", disk_read_bps_min, disk_read_bps_max, disk_read_bps_sum / n
            printf "disk_write_bps_min=%.2f\ndisk_write_bps_max=%.2f\ndisk_write_bps_mean=%.2f\n", disk_write_bps_min, disk_write_bps_max, disk_write_bps_sum / n
            printf "net_rx_bps_min=%.2f\nnet_rx_bps_max=%.2f\nnet_rx_bps_mean=%.2f\n", net_rx_bps_min, net_rx_bps_max, net_rx_bps_sum / n
            printf "net_tx_bps_min=%.2f\nnet_tx_bps_max=%.2f\nnet_tx_bps_mean=%.2f\n", net_tx_bps_min, net_tx_bps_max, net_tx_bps_sum / n
            printf "disk_read_total=%.0f\n", disk_read_bytes
            printf "disk_write_total=%.0f\n", disk_write_bytes
            printf "net_rx_total=%.0f\n", net_rx_bytes
            printf "net_tx_total=%.0f\n", net_tx_bytes
            printf "disk_read_avg_bps=%.2f\n", disk_read_bytes / elapsed
            printf "disk_write_avg_bps=%.2f\n", disk_write_bytes / elapsed
            printf "net_rx_avg_bps=%.2f\n", net_rx_bytes / elapsed
            printf "net_tx_avg_bps=%.2f\n", net_tx_bytes / elapsed
        }
    ' "$samples_file" > "$stats_file"
}

format_bytes() {
    awk -v bytes="$1" '
        BEGIN {
            split("B KiB MiB GiB TiB PiB", units, " ")
            value = bytes + 0
            unit = 1
            while (value >= 1024 && unit < 6) {
                value /= 1024
                unit++
            }
            if (unit == 1) {
                printf "%.0f %s\n", value, units[unit]
            } else {
                printf "%.2f %s\n", value, units[unit]
            }
        }
    '
}

format_rate() {
    printf "%s/s\n" "$(format_bytes "$1")"
}

emit_summary() {
    local command_status="$1"
    summarize_samples

    local samples duration_s cpu_min cpu_max cpu_mean mem_min mem_max mem_mean
    local swap_min swap_max swap_mean
    local disk_read_bps_min disk_read_bps_max disk_read_bps_mean disk_write_bps_min
    local disk_write_bps_max disk_write_bps_mean net_rx_bps_min net_rx_bps_max
    local net_rx_bps_mean net_tx_bps_min net_tx_bps_max net_tx_bps_mean
    local disk_read_total disk_write_total net_rx_total net_tx_total
    local disk_read_avg_bps disk_write_avg_bps net_rx_avg_bps net_tx_avg_bps

    # shellcheck disable=SC1090
    source "$stats_file"

    local cpu_cores ram_total swap_total command_display
    local disk_read_total_h disk_write_total_h net_rx_total_h net_tx_total_h
    local disk_read_avg_h disk_write_avg_h net_rx_avg_h net_tx_avg_h
    local disk_read_min_h disk_read_max_h disk_read_mean_h
    local disk_write_min_h disk_write_max_h disk_write_mean_h
    local net_rx_min_h net_rx_max_h net_rx_mean_h
    local net_tx_min_h net_tx_max_h net_tx_mean_h

    cpu_cores="$(cpu_capacity)"
    ram_total="$(format_bytes "$(memory_capacity_bytes)")"
    swap_total="$(format_bytes "$(swap_capacity_bytes)")"
    command_display="$(printf "%q " "${command_to_run[@]}")"
    command_display="${command_display% }"

    disk_read_total_h="$(format_bytes "$disk_read_total")"
    disk_write_total_h="$(format_bytes "$disk_write_total")"
    net_rx_total_h="$(format_bytes "$net_rx_total")"
    net_tx_total_h="$(format_bytes "$net_tx_total")"

    disk_read_avg_h="$(format_rate "$disk_read_avg_bps")"
    disk_write_avg_h="$(format_rate "$disk_write_avg_bps")"
    net_rx_avg_h="$(format_rate "$net_rx_avg_bps")"
    net_tx_avg_h="$(format_rate "$net_tx_avg_bps")"

    disk_read_min_h="$(format_rate "$disk_read_bps_min")"
    disk_read_max_h="$(format_rate "$disk_read_bps_max")"
    disk_read_mean_h="$(format_rate "$disk_read_bps_mean")"
    disk_write_min_h="$(format_rate "$disk_write_bps_min")"
    disk_write_max_h="$(format_rate "$disk_write_bps_max")"
    disk_write_mean_h="$(format_rate "$disk_write_bps_mean")"
    net_rx_min_h="$(format_rate "$net_rx_bps_min")"
    net_rx_max_h="$(format_rate "$net_rx_bps_max")"
    net_rx_mean_h="$(format_rate "$net_rx_bps_mean")"
    net_tx_min_h="$(format_rate "$net_tx_bps_min")"
    net_tx_max_h="$(format_rate "$net_tx_bps_max")"
    net_tx_mean_h="$(format_rate "$net_tx_bps_mean")"

    {
        echo ""
        echo "=== Host Load Sidecar: $label ==="
        echo "Command: $command_display"
        echo "Exit code: $command_status"
        echo "Samples: $samples over ${duration_s}s (target interval: ${interval}s)"
        echo ""
        echo "CPU utilization (capacity: ${cpu_cores} cores):"
        echo "  min: ${cpu_min}% | max: ${cpu_max}% | mean: ${cpu_mean}%"
        echo "Memory utilization (capacity: ${ram_total} RAM):"
        echo "  min: ${mem_min}% | max: ${mem_max}% | mean: ${mem_mean}%"
        echo "Swap utilization (capacity: ${swap_total} swap):"
        echo "  min: ${swap_min}% | max: ${swap_max}% | mean: ${swap_mean}%"
        echo "Physical disk reads:"
        echo "  total: ${disk_read_total_h} | average rate: ${disk_read_avg_h}"
        echo "  sample rate min: ${disk_read_min_h} | max: ${disk_read_max_h} | mean: ${disk_read_mean_h}"
        echo "Physical disk writes:"
        echo "  total: ${disk_write_total_h} | average rate: ${disk_write_avg_h}"
        echo "  sample rate min: ${disk_write_min_h} | max: ${disk_write_max_h} | mean: ${disk_write_mean_h}"
        echo "Network receive:"
        echo "  total: ${net_rx_total_h} | average rate: ${net_rx_avg_h}"
        echo "  sample rate min: ${net_rx_min_h} | max: ${net_rx_max_h} | mean: ${net_rx_mean_h}"
        echo "Network transmit:"
        echo "  total: ${net_tx_total_h} | average rate: ${net_tx_avg_h}"
        echo "  sample rate min: ${net_tx_min_h} | max: ${net_tx_max_h} | mean: ${net_tx_mean_h}"
    } > "$summary_tmp"

    cat "$summary_tmp" >&2
    if [[ -n "$summary_file" ]]; then
        {
            printf '\n```\n'
            cat "$summary_tmp"
            printf '```\n'
        } >> "$summary_file"
    fi
}

# shellcheck disable=SC2329  # Invoked by INT/TERM traps.
forward_signal() {
    local signal="$1"
    : > "$stop_file"
    if [[ -n "$command_pid" ]] && kill -0 "$command_pid" 2>/dev/null; then
        kill "-$signal" "$command_pid" 2>/dev/null || true
    fi
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

monitor_load &
monitor_pid="$!"
while [[ ! -f "$ready_file" ]]; do
    if ! kill -0 "$monitor_pid" 2>/dev/null; then
        wait "$monitor_pid" || true
        die "monitor failed to start"
    fi
    sleep 0.05
done

"${command_to_run[@]}" &
command_pid="$!"

if wait "$command_pid"; then
    command_status=0
else
    command_status="$?"
fi

: > "$stop_file"
wait "$monitor_pid" || true

emit_summary "$command_status"
exit "$command_status"
